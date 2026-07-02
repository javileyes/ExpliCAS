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
use smallvec::SmallVec;
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
    pub(super) rules: &'a HashMap<crate::target_kind::TargetKind, Vec<Arc<dyn Rule>>>,
    pub(super) global_rules: &'a Vec<Arc<dyn Rule>>,
    pub(super) phase_prefiltered: bool,
    pub(super) disabled_rules: &'a HashSet<String>,

    // ── Step recording & diagnostics ─────────────────────────────────────
    pub(super) steps_mode: StepsMode,
    pub(super) steps: Vec<Step>,
    /// Domain warnings collected regardless of steps_mode (for Off mode warning survival)
    pub(super) domain_warnings: Vec<(String, String)>, // (message, rule_name)
    /// Required domain conditions introduced by rewrites, preserved even when steps are off.
    pub(super) required_conditions: Vec<crate::ImplicitCondition>,

    // ── Traversal state ──────────────────────────────────────────────────
    pub(super) cache: HashMap<ExprId, ExprId>,
    pub(super) current_path: SmallVec<[crate::step::PathStep; 8]>,
    /// Stack of ancestor ExprIds for parent context propagation to rules
    pub(super) ancestor_stack: SmallVec<[ExprId; 8]>,
    /// Current recursion depth for stack overflow prevention
    pub(super) current_depth: usize,
    /// Flag to track if we already warned about depth overflow (to avoid spamming)
    pub(super) depth_overflow_warned: bool,
    /// Suppress best-effort depth overflow warnings for internal/proof-only simplify calls.
    pub(super) suppress_depth_overflow_warnings: bool,
    /// The current root expression being simplified, used to compute global_after for steps
    pub(super) root_expr: ExprId,
    /// Optional event listener for rule-application events.
    pub(super) event_listener:
        Option<&'a mut (dyn cas_solver_core::engine_events::StepListener + 'static)>,

    // ── Rule context & filtering ─────────────────────────────────────────
    pub(super) profiler: &'a mut RuleProfiler,
    pub(super) initial_parent_ctx: crate::parent_context::ParentContext, // Carries marks to rules
    /// Current phase of the simplification pipeline (controls which rules can run)
    pub(super) current_phase: crate::phase::SimplifyPhase,
    /// Shared deadline inherited from the orchestrator for cooperative timeouts.
    pub(super) deadline: Option<std::time::Instant>,
    /// Sticky timeout bit for this local pass.
    pub(super) timed_out: bool,
    /// Poll counter so we do not hit `Instant::now()` on every node/rule dispatch.
    pub(super) deadline_check_counter: u32,

    // ── Cycle detection ──────────────────────────────────────────────────
    /// Cycle detector for ping-pong detection (always-on as of V2.14.30)
    pub(super) cycle_detector: Option<cas_solver_core::cycle_detection::CycleDetector>,
    /// Phase that the cycle detector was initialized for (reset when phase changes)
    pub(super) cycle_phase: Option<crate::phase::SimplifyPhase>,
    /// Fingerprint memoization cache (cleared per phase)
    pub(super) fp_memo: cas_solver_core::cycle_detection::FingerprintMemo,
    /// Last detected cycle info (for PhaseStats)
    pub(super) last_cycle: Option<cas_solver_core::cycle_models::CycleInfo>,
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
    #[inline]
    pub(super) fn time_budget_exceeded(&mut self) -> bool {
        if self.timed_out {
            return true;
        }
        let Some(deadline) = self.deadline else {
            return false;
        };
        self.deadline_check_counter = self.deadline_check_counter.wrapping_add(1);
        if (self.deadline_check_counter & 0x3f) != 0 {
            return false;
        }
        if std::time::Instant::now() >= deadline {
            self.timed_out = true;
            return true;
        }
        false
    }

    pub(super) fn indent(&self) -> String {
        "  ".repeat(self.current_path.len())
    }

    #[inline]
    pub(super) fn collect_steps_enabled(&self) -> bool {
        self.steps_mode != StepsMode::Off
    }

    #[inline]
    fn push_path_step_if_recording(&mut self, step: crate::step::PathStep) {
        if self.collect_steps_enabled() {
            self.current_path.push(step);
        }
    }

    #[inline]
    fn pop_path_step_if_recording(&mut self) {
        if self.collect_steps_enabled() {
            self.current_path.pop();
        }
    }

    #[inline]
    fn transform_child_at(
        &mut self,
        parent: ExprId,
        path_step: crate::step::PathStep,
        child: ExprId,
    ) -> ExprId {
        self.push_path_step_if_recording(path_step);
        self.ancestor_stack.push(parent);
        let transformed = self.transform_expr_recursive(child);
        self.ancestor_stack.pop();
        self.pop_path_step_if_recording();
        transformed
    }

    pub(super) fn transform_expr_recursive(&mut self, id: ExprId) -> ExprId {
        if self.time_budget_exceeded() {
            return id;
        }

        // Depth guard: prevent stack overflow by limiting recursion depth
        self.current_depth += 1;
        if self.current_depth > MAX_SIMPLIFY_DEPTH {
            if !self.depth_overflow_warned {
                self.depth_overflow_warned = true;

                if !self.suppress_depth_overflow_warnings
                    && !crate::are_depth_overflow_warnings_suppressed()
                {
                    // Log the expression to file for later investigation
                    let display = cas_formatter::DisplayExpr {
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
                let new_e = self.transform_child_at(id, crate::step::PathStep::Inner, e);

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
                    let new_elem =
                        self.transform_child_at(id, crate::step::PathStep::Arg(i), *elem);
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
            // Strong internal hold: keep protected calculus presentation outputs
            // out of the phase pipeline. The function-form `__hold(...)` remains
            // available for the older transparent barrier contract.
            Expr::Hold(_) => {
                self.current_depth -= 1;
                return id;
            }
        };

        // 2. Apply rules
        let mut result = self.apply_rules(expr_with_simplified_children);

        // Universal soundness backstop: rule application must not collapse a
        // non-finite/undefined additive combination into a purely finite value.
        // `inf - inf`, `sqrt(inf) - sqrt(inf)`, `ln(inf) - ln(inf) + 7` and
        // `x/0 - x/0 + y/0 - y/0` are indeterminate, not `0`/finite. The same
        // "these terms cancel" conclusion is reached by a large family of
        // independent rules; rather than gate each, reject any rule result that
        // drops the non-finite here, keeping the pre-rule form so a sound path
        // (folding to `undefined`, or staying symbolic) wins instead.
        // The backstop can only fire when a rewrite actually changed the node (it
        // compares `before` vs `after` for a dropped non-finite), so skip the full
        // subtree walk on the common fixpoint case where nothing changed — the
        // hottest path in the engine (P1 of the saneamiento audit).
        if result != expr_with_simplified_children
            && cas_math::arithmetic_cancel_support::rewrite_unsoundly_drops_nonfinite_in_domain(
                self.context,
                expr_with_simplified_children,
                result,
                cas_math::abs_support::value_domain_mode_from_flag(
                    self.initial_parent_ctx.value_domain().is_real_only(),
                ),
            )
        {
            result = expr_with_simplified_children;
        }

        self.cache.insert(id, result);
        result
    }
}
