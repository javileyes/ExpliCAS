//! Rewrite engine core: `LocalSimplificationTransformer` and `apply_rules`.
//!
//! This module contains the bottom-up expression transformer that recursively
//! simplifies children, then applies matching rules at each node. It includes
//! depth-guarding, cycle detection, budget tracking, and step recording.

mod domain_airbag;
mod rule_application;
mod step_recording;
mod transform_helpers;

use super::hold::is_hold_all_function;
use crate::options::StepsMode;
use crate::profiler::RuleProfiler;
use crate::rule::Rule;
use crate::step::Step;
use cas_ast::{symbol::SymbolId, Context, Expr, ExprId};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use tracing::debug;

/// Maximum recursion depth for simplification to prevent stack overflow.
///
/// V2.15.9: REVERTED to 30 for 8MB stack stability after observing stack overflow
/// in expressions like sin((3-x+sin(x))^4). The KI documents that each recursive
/// frame in `transform_expr_recursive` consumes ~150KB even with `#[inline(never)]`
/// optimizations, limiting safe depth to ~50 on 16MB stacks and ~30 on 8MB stacks.
///
/// For deeper expressions, use `RUST_MIN_STACK=16777216` (16MB).
const MAX_SIMPLIFY_DEPTH: usize = 50;

/// Path to log expressions that exceed the depth limit for later investigation.
const DEPTH_OVERFLOW_LOG_PATH: &str = "/tmp/cas_depth_overflow_expressions.log";

/// Cached env var check: avoids heap-allocated String lookup on every rule match.
static CAS_TRACE_RULES_ENABLED: std::sync::LazyLock<bool> =
    std::sync::LazyLock::new(|| std::env::var("CAS_TRACE_RULES").is_ok());

/// Binary operation type for transform_binary helper
#[derive(Clone, Copy)]
#[allow(dead_code)] // Div is kept for consistency, may be used if Div early-detection is removed
enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

pub(super) struct LocalSimplificationTransformer<'a> {
    // ── Core context (borrowed from Engine) ──────────────────────────────
    pub(super) context: &'a mut Context,
    pub(super) rules: &'a HashMap<String, Vec<Arc<dyn Rule>>>,
    pub(super) global_rules: &'a Vec<Arc<dyn Rule>>,
    pub(super) disabled_rules: &'a HashSet<String>,

    // ── Step recording & diagnostics ─────────────────────────────────────
    pub(super) steps_mode: StepsMode,
    pub(super) steps: Vec<Step>,
    /// Domain warnings collected regardless of steps_mode (for Off mode warning survival)
    pub(super) domain_warnings: Vec<(String, String)>, // (message, rule_name)

    // ── Traversal state ──────────────────────────────────────────────────
    pub(super) cache: HashMap<ExprId, ExprId>,
    pub(super) current_path: Vec<crate::step::PathStep>,
    /// Stack of ancestor ExprIds for parent context propagation to rules
    pub(super) ancestor_stack: Vec<ExprId>,
    /// Current recursion depth for stack overflow prevention
    pub(super) current_depth: usize,
    /// Flag to track if we already warned about depth overflow (to avoid spamming)
    pub(super) depth_overflow_warned: bool,
    /// The current root expression being simplified, used to compute global_after for steps
    pub(super) root_expr: ExprId,

    // ── Rule context & filtering ─────────────────────────────────────────
    pub(super) profiler: &'a mut RuleProfiler,
    #[allow(dead_code)]
    pub(super) pattern_marks: crate::pattern_marks::PatternMarks, // For context-aware guards (used via initial_parent_ctx)
    pub(super) initial_parent_ctx: crate::parent_context::ParentContext, // Carries marks to rules
    /// Current phase of the simplification pipeline (controls which rules can run)
    pub(super) current_phase: crate::phase::SimplifyPhase,
    /// Purpose of simplification: controls which rules are filtered by solve_safety()
    pub(super) simplify_purpose: crate::solve_safety::SimplifyPurpose,

    // ── Cycle detection ──────────────────────────────────────────────────
    /// Cycle detector for ping-pong detection (always-on as of V2.14.30)
    pub(super) cycle_detector: Option<crate::cycle_detector::CycleDetector>,
    /// Phase that the cycle detector was initialized for (reset when phase changes)
    pub(super) cycle_phase: Option<crate::phase::SimplifyPhase>,
    /// Fingerprint memoization cache (cleared per phase)
    pub(super) fp_memo: crate::cycle_detector::FingerprintMemo,
    /// Last detected cycle info (for PhaseStats)
    pub(super) last_cycle: Option<crate::cycle_detector::CycleInfo>,
    /// Blocked (fingerprint, rule) pairs to prevent cycle re-entry
    pub(super) blocked_rules: std::collections::HashSet<(u64, String)>,

    // ── Budget tracking ──────────────────────────────────────────────────
    /// Count of rewrites accepted in this pass (charged to Budget at end of pass)
    pub(super) rewrite_count: u64,
    /// Snapshot of nodes_created at start of pass (for delta charging)
    pub(super) nodes_snap: u64,
    /// Operation type for budget charging (SimplifyCore or SimplifyTransform)
    pub(super) budget_op: crate::budget::Operation,
    /// Set when budget exceeded - contains the error details for the caller
    pub(super) stop_reason: Option<crate::budget::BudgetExceeded>,

    // ── Performance caches ───────────────────────────────────────────────
    /// PERF: Reusable cache for normalize_core, avoids per-call HashMap allocation.
    pub(super) normalize_cache: std::collections::HashMap<cas_ast::ExprId, cas_ast::ExprId>,
}

use cas_ast::visitor::Transformer;

// NOTE on ancestor_stack pattern:
//
// We cannot use RAII guards (like AncestorScope) for push/pop because:
// 1. AncestorScope would borrow &mut self.ancestor_stack
// 2. transform_expr_recursive needs &mut self
// 3. Rust doesn't allow split borrows of struct fields through methods
//
// REQUIRED PATTERN for all operator cases that recurse into children:
// ```
// self.ancestor_stack.push(id);  // Before transform_expr_recursive
// let result = self.transform_expr_recursive(child);
// self.ancestor_stack.pop();     // After transform_expr_recursive (balanced!)
// ```
//
// This is critical for context-aware rules (like AutoExpandPowSumRule) that check
// in_auto_expand_context() - they need to see their ancestor chain.
// Test: test_auto_expand_step_visible_in_sub_context

impl<'a> Transformer for LocalSimplificationTransformer<'a> {
    fn transform_expr(&mut self, _context: &mut Context, id: ExprId) -> ExprId {
        self.transform_expr_recursive(id)
    }
}

impl<'a> LocalSimplificationTransformer<'a> {
    pub(super) fn indent(&self) -> String {
        "  ".repeat(self.current_path.len())
    }

    pub(super) fn transform_expr_recursive(&mut self, id: ExprId) -> ExprId {
        // Depth guard: prevent stack overflow by limiting recursion depth
        self.current_depth += 1;
        if self.current_depth > MAX_SIMPLIFY_DEPTH {
            if !self.depth_overflow_warned {
                self.depth_overflow_warned = true;

                // Log the expression to file for later investigation
                let display = cas_ast::DisplayExpr {
                    context: self.context,
                    id: self.root_expr,
                };
                let expr_str = display.to_string();
                let log_entry = format!(
                    "[{:?}] Depth overflow at phase {:?}, depth {}: {}\n",
                    std::time::SystemTime::now(),
                    self.current_phase,
                    self.current_depth,
                    expr_str
                );

                // Append to log file (ignore errors - this is best-effort)
                use std::io::Write;
                if let Ok(mut file) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(DEPTH_OVERFLOW_LOG_PATH)
                {
                    let _ = file.write_all(log_entry.as_bytes());
                }

                // Emit warning via tracing
                tracing::warn!(
                    target: "simplify",
                    depth = self.current_depth,
                    phase = ?self.current_phase,
                    expr = %expr_str,
                    "depth_overflow - returning expression unsimplified"
                );
            }

            // Return expression as-is without further simplification
            self.current_depth -= 1;
            return id;
        }

        // Use tracing for debug logging
        let expr = self.context.get(id);
        debug!("{}[DEBUG] Visiting: {:?}", self.indent(), expr);

        // println!("Visiting: {:?} {:?}", id, self.context.get(id));
        // println!("Simplifying: {:?}", id);
        if let Some(&cached) = self.cache.get(&id) {
            self.current_depth -= 1;
            return cached;
        }

        // 1. Simplify children first (Bottom-Up)
        let expr = self.context.get(id).clone();

        let expr_with_simplified_children = match expr {
            Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) => id,
            Expr::Add(l, r) => self.transform_binary(id, l, r, BinaryOp::Add),
            Expr::Sub(l, r) => self.transform_binary(id, l, r, BinaryOp::Sub),
            Expr::Mul(l, r) => self.transform_binary(id, l, r, BinaryOp::Mul),
            Expr::Div(l, r) => self.transform_div(id, l, r),
            Expr::Pow(b, e) => self.transform_pow(id, b, e),
            Expr::Neg(e) => {
                if self.steps_mode != StepsMode::Off {
                    self.current_path.push(crate::step::PathStep::Inner);
                }
                self.ancestor_stack.push(id); // Track current node as parent for children
                let new_e = self.transform_expr_recursive(e);
                self.ancestor_stack.pop();
                if self.steps_mode != StepsMode::Off {
                    self.current_path.pop();
                }

                if new_e != e {
                    self.context.add(Expr::Neg(new_e))
                } else {
                    id
                }
            }
            Expr::Function(fn_id, args) => self.transform_function(id, fn_id, args.clone()),
            Expr::Matrix { rows, cols, data } => {
                // Recursively simplify matrix elements
                let mut new_data = Vec::new();
                let mut changed = false;
                for (i, elem) in data.iter().enumerate() {
                    if self.steps_mode != StepsMode::Off {
                        self.current_path.push(crate::step::PathStep::Arg(i));
                    }
                    self.ancestor_stack.push(id); // Track current node as parent for children
                    let new_elem = self.transform_expr_recursive(*elem);
                    self.ancestor_stack.pop();
                    if self.steps_mode != StepsMode::Off {
                        self.current_path.pop();
                    }
                    if new_elem != *elem {
                        changed = true;
                    }
                    new_data.push(new_elem);
                }
                if changed {
                    self.context.add(Expr::Matrix {
                        rows,
                        cols,
                        data: new_data,
                    })
                } else {
                    id
                }
            }
            // SessionRef is a leaf - return as-is (should be resolved before simplification)
            Expr::SessionRef(_) => id,
            // Hold barrier: simplify contents but preserve the wrapper (blocks expansive rules)
            Expr::Hold(inner) => {
                if self.steps_mode != StepsMode::Off {
                    self.current_path.push(crate::step::PathStep::Inner);
                }
                self.ancestor_stack.push(id);
                let new_inner = self.transform_expr_recursive(inner);
                self.ancestor_stack.pop();
                if self.steps_mode != StepsMode::Off {
                    self.current_path.pop();
                }

                if new_inner != inner {
                    self.context.add(Expr::Hold(new_inner))
                } else {
                    id
                }
            }
        };

        // 2. Apply rules
        let result = self.apply_rules(expr_with_simplified_children);
        self.cache.insert(id, result);
        result
    }
}
