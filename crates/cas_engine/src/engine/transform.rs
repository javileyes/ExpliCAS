//! Rewrite engine core: `LocalSimplificationTransformer` and `apply_rules`.
//!
//! This module contains the bottom-up expression transformer that recursively
//! simplifies children, then applies matching rules at each node. It includes
//! depth-guarding, cycle detection, budget tracking, and step recording.

use super::hold::is_hold_all_function;
use crate::canonical_forms::normalize_core_with_cache;
use crate::options::StepsMode;
use crate::profiler::RuleProfiler;
use crate::rule::{Rewrite, Rule};
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
    pub(super) context: &'a mut Context,
    pub(super) rules: &'a HashMap<String, Vec<Arc<dyn Rule>>>,
    pub(super) global_rules: &'a Vec<Arc<dyn Rule>>,
    pub(super) disabled_rules: &'a HashSet<String>,
    pub(super) steps_mode: StepsMode,
    pub(super) steps: Vec<Step>,
    /// Domain warnings collected regardless of steps_mode (for Off mode warning survival)
    pub(super) domain_warnings: Vec<(String, String)>, // (message, rule_name)
    pub(super) cache: HashMap<ExprId, ExprId>,
    pub(super) current_path: Vec<crate::step::PathStep>,
    pub(super) profiler: &'a mut RuleProfiler,
    #[allow(dead_code)]
    pub(super) pattern_marks: crate::pattern_marks::PatternMarks, // For context-aware guards (used via initial_parent_ctx)
    pub(super) initial_parent_ctx: crate::parent_context::ParentContext, // Carries marks to rules
    /// The current root expression being simplified, used to compute global_after for steps
    pub(super) root_expr: ExprId,
    /// Current phase of the simplification pipeline (controls which rules can run)
    pub(super) current_phase: crate::phase::SimplifyPhase,
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
    /// Current recursion depth for stack overflow prevention
    pub(super) current_depth: usize,
    /// Flag to track if we already warned about depth overflow (to avoid spamming)
    pub(super) depth_overflow_warned: bool,
    /// Stack of ancestor ExprIds for parent context propagation to rules
    pub(super) ancestor_stack: Vec<ExprId>,
    // === Budget tracking (Phase 2 unified) ===
    /// Count of rewrites accepted in this pass (charged to Budget at end of pass)
    pub(super) rewrite_count: u64,
    /// Snapshot of nodes_created at start of pass (for delta charging)
    pub(super) nodes_snap: u64,
    /// Operation type for budget charging (SimplifyCore or SimplifyTransform)
    pub(super) budget_op: crate::budget::Operation,
    /// Set when budget exceeded - contains the error details for the caller
    pub(super) stop_reason: Option<crate::budget::BudgetExceeded>,
    /// Purpose of simplification: controls which rules are filtered by solve_safety()
    pub(super) simplify_purpose: crate::solve_safety::SimplifyPurpose,
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
    fn indent(&self) -> String {
        "  ".repeat(self.current_path.len())
    }

    /// Build the ParentContext for the current node position.
    ///
    /// PERF: This is called once per `apply_rules()` invocation (i.e., once per node),
    /// and the result is shared by all rule attempts on that node. Previously, this
    /// construction was duplicated inside both the specific-rules and global-rules loops,
    /// causing O(rules × nodes × ancestors) redundant work.
    fn build_parent_context(&self) -> crate::parent_context::ParentContext {
        let mut ctx = if let Some(marks) = self.initial_parent_ctx.pattern_marks() {
            crate::parent_context::ParentContext::with_marks(marks.clone())
        } else {
            crate::parent_context::ParentContext::root()
        };
        // Copy expand_mode (enables BinomialExpansionRule when Simplifier::expand() is called)
        if self.initial_parent_ctx.is_expand_mode() {
            ctx = ctx.with_expand_mode_flag(true);
        }
        // Copy auto_expand (enables AutoExpandPowSumRule)
        if self.initial_parent_ctx.is_auto_expand() {
            ctx = ctx
                .with_auto_expand_flag(true, self.initial_parent_ctx.auto_expand_budget().cloned());
        }
        ctx = ctx.with_domain_mode(self.initial_parent_ctx.domain_mode());
        ctx = ctx.with_inv_trig(self.initial_parent_ctx.inv_trig_policy());
        ctx = ctx.with_value_domain(self.initial_parent_ctx.value_domain());
        ctx = ctx.with_goal(self.initial_parent_ctx.goal());
        if let Some(root) = self.initial_parent_ctx.root_expr() {
            ctx = ctx.with_root_expr_only(root);
        }
        // Propagate context_mode and simplify_purpose for Solve mode blocking
        ctx = ctx.with_context_mode(self.initial_parent_ctx.context_mode());
        ctx = ctx.with_simplify_purpose(self.initial_parent_ctx.simplify_purpose());
        // Propagate implicit_domain for domain-aware simplifications
        ctx = ctx.with_implicit_domain(self.initial_parent_ctx.implicit_domain().cloned());
        // Build ancestor chain from stack (for Div tracking)
        for &ancestor in &self.ancestor_stack {
            ctx = ctx.extend_with_div_check(ancestor, self.context);
        }
        ctx = ctx.with_autoexpand_binomials(self.initial_parent_ctx.autoexpand_binomials());
        ctx = ctx.with_heuristic_poly(self.initial_parent_ctx.heuristic_poly());
        ctx
    }

    /// Reconstruct the global expression by substituting `replacement` at the given path
    fn reconstruct_at_path(&mut self, replacement: ExprId) -> ExprId {
        use crate::step::PathStep;

        fn reconstruct_recursive(
            context: &mut Context,
            root: ExprId,
            path: &[PathStep],
            replacement: ExprId,
        ) -> ExprId {
            if path.is_empty() {
                return replacement;
            }

            let current_step = &path[0];
            let remaining_path = &path[1..];
            let expr = context.get(root).clone();

            match (expr, current_step) {
                (Expr::Add(l, r), PathStep::Left) => {
                    let new_l = reconstruct_recursive(context, l, remaining_path, replacement);
                    context.add_raw(Expr::Add(new_l, r)) // Use add_raw to preserve structure
                }
                (Expr::Add(l, r), PathStep::Right) => {
                    // Follow AST literally - don't do magic Neg unwrapping.
                    // If we need to modify inside a Neg, the path should include PathStep::Inner.
                    let new_r = reconstruct_recursive(context, r, remaining_path, replacement);
                    context.add_raw(Expr::Add(l, new_r)) // Use add_raw to preserve structure
                }
                (Expr::Sub(l, r), PathStep::Left) => {
                    let new_l = reconstruct_recursive(context, l, remaining_path, replacement);
                    context.add_raw(Expr::Sub(new_l, r)) // Use add_raw to preserve structure
                }
                (Expr::Sub(l, r), PathStep::Right) => {
                    let new_r = reconstruct_recursive(context, r, remaining_path, replacement);
                    context.add_raw(Expr::Sub(l, new_r)) // Use add_raw to preserve structure
                }
                (Expr::Mul(l, r), PathStep::Left) => {
                    let new_l = reconstruct_recursive(context, l, remaining_path, replacement);
                    context.add_raw(Expr::Mul(new_l, r)) // Use add_raw to preserve structure
                }
                (Expr::Mul(l, r), PathStep::Right) => {
                    let new_r = reconstruct_recursive(context, r, remaining_path, replacement);
                    context.add_raw(Expr::Mul(l, new_r)) // Use add_raw to preserve structure
                }
                (Expr::Div(l, r), PathStep::Left) => {
                    let new_l = reconstruct_recursive(context, l, remaining_path, replacement);
                    context.add_raw(Expr::Div(new_l, r)) // Use add_raw to preserve structure
                }
                (Expr::Div(l, r), PathStep::Right) => {
                    let new_r = reconstruct_recursive(context, r, remaining_path, replacement);
                    context.add_raw(Expr::Div(l, new_r)) // Use add_raw to preserve structure
                }
                (Expr::Pow(b, e), PathStep::Base) => {
                    let new_b = reconstruct_recursive(context, b, remaining_path, replacement);
                    context.add_raw(Expr::Pow(new_b, e)) // Use add_raw to preserve structure
                }
                (Expr::Pow(b, e), PathStep::Exponent) => {
                    let new_e = reconstruct_recursive(context, e, remaining_path, replacement);
                    context.add_raw(Expr::Pow(b, new_e)) // Use add_raw to preserve structure
                }
                (Expr::Neg(e), PathStep::Inner) => {
                    let new_e = reconstruct_recursive(context, e, remaining_path, replacement);
                    context.add_raw(Expr::Neg(new_e)) // Use add_raw to preserve structure
                }
                (Expr::Function(name, args), PathStep::Arg(idx)) => {
                    let mut new_args = args;
                    if *idx < new_args.len() {
                        new_args[*idx] = reconstruct_recursive(
                            context,
                            new_args[*idx],
                            remaining_path,
                            replacement,
                        );
                        context.add_raw(Expr::Function(name, new_args)) // Use add_raw to preserve structure
                    } else {
                        root
                    }
                }
                (Expr::Hold(inner), PathStep::Inner) => {
                    let new_inner =
                        reconstruct_recursive(context, inner, remaining_path, replacement);
                    context.add_raw(Expr::Hold(new_inner))
                }
                // Leaves — no children, path cannot descend further
                (Expr::Number(_), _)
                | (Expr::Constant(_), _)
                | (Expr::Variable(_), _)
                | (Expr::SessionRef(_), _)
                | (Expr::Matrix { .. }, _) => root,
                // Path mismatch: valid expr but wrong PathStep direction
                _ => root,
            }
        }

        let new_root = reconstruct_recursive(
            self.context,
            self.root_expr,
            &self.current_path,
            replacement,
        );
        self.root_expr = new_root; // Update root for next step
        new_root
    }

    /// Record a step without inflating the recursive frame.
    /// Using #[inline(never)] to ensure Step construction stays out of transform_expr_recursive.
    #[inline(never)]
    fn record_step(
        &mut self,
        name: &'static str,
        description: &'static str,
        before: ExprId,
        after: ExprId,
    ) {
        if self.steps_mode != StepsMode::Off {
            let step = crate::step::Step::new(
                name,
                description,
                before,
                after,
                self.current_path.clone(),
                Some(self.context),
            );
            self.steps.push(step);
        }
    }

    /// Transform binary expression (Add/Sub/Mul) by simplifying children.
    /// Extracted to reduce stack frame size in transform_expr_recursive.
    #[inline(never)]
    fn transform_binary(&mut self, id: ExprId, l: ExprId, r: ExprId, op: BinaryOp) -> ExprId {
        if self.steps_mode != StepsMode::Off {
            self.current_path.push(crate::step::PathStep::Left);
        }
        self.ancestor_stack.push(id);
        let new_l = self.transform_expr_recursive(l);
        self.ancestor_stack.pop();
        if self.steps_mode != StepsMode::Off {
            self.current_path.pop();
        }

        if self.steps_mode != StepsMode::Off {
            self.current_path.push(crate::step::PathStep::Right);
        }
        self.ancestor_stack.push(id);
        let new_r = self.transform_expr_recursive(r);
        self.ancestor_stack.pop();
        if self.steps_mode != StepsMode::Off {
            self.current_path.pop();
        }

        if new_l != l || new_r != r {
            let expr = match op {
                BinaryOp::Add => Expr::Add(new_l, new_r),
                BinaryOp::Sub => Expr::Sub(new_l, new_r),
                BinaryOp::Mul => Expr::Mul(new_l, new_r),
                BinaryOp::Div => Expr::Div(new_l, new_r),
            };
            self.context.add(expr)
        } else {
            id
        }
    }

    /// Transform Pow expression with early detection for sqrt-of-square patterns.
    /// Extracted with #[inline(never)] to reduce stack frame size.
    #[inline(never)]
    fn transform_pow(&mut self, id: ExprId, base: ExprId, exp: ExprId) -> ExprId {
        // EARLY DETECTION: sqrt-of-square pattern (u^2)^(1/2) -> |u|
        // Must check BEFORE recursing into children to prevent binomial expansion
        if crate::helpers::is_half(self.context, exp) {
            // Try (something^2)^(1/2) -> |something|
            if let Some(result) = self.try_sqrt_of_square(id, base) {
                return result;
            }
            // Try (u * u)^(1/2) -> |u|
            if let Some(result) = self.try_sqrt_of_product(id, base) {
                return result;
            }
        }

        // Check if this Pow is canonical before recursing into children
        if crate::canonical_forms::is_canonical_form(self.context, id) {
            debug!(
                "Skipping simplification of canonical Pow: {:?}",
                self.context.get(id)
            );
            return id;
        }

        // Simplify children
        if self.steps_mode != StepsMode::Off {
            self.current_path.push(crate::step::PathStep::Base);
        }
        self.ancestor_stack.push(id);
        let new_b = self.transform_expr_recursive(base);
        self.ancestor_stack.pop();
        if self.steps_mode != StepsMode::Off {
            self.current_path.pop();
        }

        if self.steps_mode != StepsMode::Off {
            self.current_path.push(crate::step::PathStep::Exponent);
        }
        self.ancestor_stack.push(id);
        let new_e = self.transform_expr_recursive(exp);
        self.ancestor_stack.pop();
        if self.steps_mode != StepsMode::Off {
            self.current_path.pop();
        }

        if new_b != base || new_e != exp {
            self.context.add(Expr::Pow(new_b, new_e))
        } else {
            id
        }
    }

    /// Try to simplify (u^2)^(1/2) -> |u|
    #[inline(never)]
    fn try_sqrt_of_square(&mut self, id: ExprId, base: ExprId) -> Option<ExprId> {
        if let Expr::Pow(inner_base, inner_exp) = self.context.get(base) {
            if let Expr::Number(n) = self.context.get(*inner_exp) {
                if n.is_integer() && *n == num_rational::BigRational::from_integer(2.into()) {
                    let abs_expr = self.context.call("abs", vec![*inner_base]);
                    self.record_step(
                        "sqrt(u^2) = |u|",
                        "Simplify Square Root of Square",
                        id,
                        abs_expr,
                    );
                    return Some(self.transform_expr_recursive(abs_expr));
                }
            }
        }
        None
    }

    /// Try to simplify (u * u)^(1/2) -> |u|
    #[inline(never)]
    fn try_sqrt_of_product(&mut self, id: ExprId, base: ExprId) -> Option<ExprId> {
        if let Expr::Mul(left, right) = self.context.get(base) {
            if crate::ordering::compare_expr(self.context, *left, *right)
                == std::cmp::Ordering::Equal
            {
                let abs_expr = self.context.call("abs", vec![*left]);
                self.record_step(
                    "sqrt(u * u) = |u|",
                    "Simplify Square Root of Product",
                    id,
                    abs_expr,
                );
                return Some(self.transform_expr_recursive(abs_expr));
            }
        }
        None
    }

    /// Transform Function expression by simplifying children.
    /// Extracted with #[inline(never)] to reduce stack frame size.
    #[inline(never)]
    fn transform_function(&mut self, id: ExprId, fn_id: SymbolId, args: Vec<ExprId>) -> ExprId {
        let name = self.context.sym_name(fn_id);
        // Check if this function is canonical before recursing into children
        if (name == "sqrt" || name == "abs")
            && crate::canonical_forms::is_canonical_form(self.context, id)
        {
            debug!(
                "Skipping simplification of canonical Function: {:?}",
                self.context.get(id)
            );
            return id;
        }

        // HoldAll semantics: do NOT simplify arguments for these functions
        if is_hold_all_function(name) {
            debug!(
                "HoldAll function, skipping child simplification: {:?}",
                self.context.get(id)
            );
            return id;
        }

        // Simplify children
        let mut new_args = Vec::with_capacity(args.len());
        let mut changed = false;
        for (i, arg) in args.iter().enumerate() {
            if self.steps_mode != StepsMode::Off {
                self.current_path.push(crate::step::PathStep::Arg(i));
            }
            self.ancestor_stack.push(id);
            let new_arg = self.transform_expr_recursive(*arg);
            self.ancestor_stack.pop();
            if self.steps_mode != StepsMode::Off {
                self.current_path.pop();
            }

            if new_arg != *arg {
                changed = true;
            }
            new_args.push(new_arg);
        }

        if changed {
            self.context.add(Expr::Function(fn_id, new_args))
        } else {
            id
        }
    }

    /// Transform Div expression with early detection for difference-of-squares pattern.
    /// Extracted with #[inline(never)] to reduce stack frame size.
    #[inline(never)]
    fn transform_div(&mut self, id: ExprId, l: ExprId, r: ExprId) -> ExprId {
        // EARLY DETECTION: (A² - B²) / (A ± B) pattern
        if let Some(early_result) = crate::rules::algebra::try_difference_of_squares_preorder(
            self.context,
            id,
            l,
            r,
            self.steps_mode != StepsMode::Off,
            &mut self.steps,
            &self.current_path,
        ) {
            // Note: don't decrement depth here - transform_expr_recursive manages it
            return self.transform_expr_recursive(early_result);
        }

        // Simplify children
        if self.steps_mode != StepsMode::Off {
            self.current_path.push(crate::step::PathStep::Left);
        }
        self.ancestor_stack.push(id);
        let new_l = self.transform_expr_recursive(l);
        self.ancestor_stack.pop();
        if self.steps_mode != StepsMode::Off {
            self.current_path.pop();
        }

        if self.steps_mode != StepsMode::Off {
            self.current_path.push(crate::step::PathStep::Right);
        }
        self.ancestor_stack.push(id);
        let new_r = self.transform_expr_recursive(r);
        self.ancestor_stack.pop();
        if self.steps_mode != StepsMode::Off {
            self.current_path.pop();
        }

        if new_l != l || new_r != r {
            self.context.add(Expr::Div(new_l, new_r))
        } else {
            id
        }
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

    /// Check the Domain Delta Airbag: whether a rewrite expands the analytic domain.
    ///
    /// Returns `true` if the rewrite should be **skipped** (blocked in Strict mode).
    /// In Generic mode, attaches `required_conditions` to the rewrite.
    /// In Assume mode, attaches `assumption_events` to the rewrite.
    ///
    /// `skip_unless_analytic`: if `true`, only check rules with `NeedsCondition(Analytic)`.
    /// Specific rules use `true` (gated), global rules use `false` (unconditional).
    #[inline(never)]
    fn check_domain_airbag(
        &mut self,
        rule: &dyn Rule,
        parent_ctx: &crate::parent_context::ParentContext,
        expr_id: ExprId,
        rewrite: &mut Rewrite,
        skip_unless_analytic: bool,
    ) -> bool {
        // Determine which parent context to read domain info from
        let vd = parent_ctx.value_domain();
        let mode = parent_ctx.domain_mode();

        // Optionally gate on NeedsCondition(Analytic) solve_safety
        if skip_unless_analytic {
            let needs_analytic_check = matches!(
                rule.solve_safety(),
                crate::solve_safety::SolveSafety::NeedsCondition(
                    crate::assumptions::ConditionClass::Analytic
                )
            );
            if !needs_analytic_check {
                return false; // Not blocked
            }
        }

        if parent_ctx.implicit_domain().is_none() {
            return false; // No domain to check
        }

        use crate::implicit_domain::{
            check_analytic_expansion, AnalyticExpansionResult, ImplicitCondition,
        };

        let expansion =
            check_analytic_expansion(self.context, self.root_expr, expr_id, rewrite.new_expr, vd);

        if let AnalyticExpansionResult::WouldExpand { dropped, sources } = expansion {
            match mode {
                crate::domain::DomainMode::Strict => {
                    debug!(
                        "{}[DEBUG] Rule '{}' would expand analytic domain ({}), blocked in Strict mode",
                        self.indent(),
                        rule.name(),
                        sources.join(", ")
                    );
                    return true; // Blocked
                }
                crate::domain::DomainMode::Generic => {
                    rewrite.required_conditions.extend(dropped.clone());
                    debug!(
                        "{}[DEBUG] Rule '{}' expands analytic domain, allowed in Generic mode with required conditions: {}",
                        self.indent(),
                        rule.name(),
                        sources.join(", ")
                    );
                }
                crate::domain::DomainMode::Assume => {
                    for cond in dropped {
                        match cond {
                            ImplicitCondition::NonNegative(t) => {
                                rewrite.assumption_events.push(
                                    crate::assumptions::AssumptionEvent::nonnegative(
                                        self.context,
                                        t,
                                    ),
                                );
                            }
                            ImplicitCondition::Positive(t) => {
                                rewrite.assumption_events.push(
                                    crate::assumptions::AssumptionEvent::positive(self.context, t),
                                );
                            }
                            ImplicitCondition::NonZero(_) => {} // Skip definability
                        }
                    }
                    debug!(
                        "{}[DEBUG] Rule '{}' expands analytic domain, allowed in Assume mode with assumptions: {}",
                        self.indent(),
                        rule.name(),
                        sources.join(", ")
                    );
                }
            }
        }

        false // Not blocked
    }

    /// Record a rewrite as one or more Steps (main + chained), handling `steps_mode` gating.
    ///
    /// Returns the `final_result` ExprId (last of chained, or main `new_expr`).
    /// When `steps_mode == Off`, skips all Step construction for performance.
    #[inline(never)]
    fn record_rewrite_step(
        &mut self,
        rule: &dyn Rule,
        expr_id: ExprId,
        rewrite: &Rewrite,
    ) -> ExprId {
        if self.steps_mode != StepsMode::Off {
            let main_new_expr = rewrite.new_expr;
            let main_description = &rewrite.description;
            let main_before_local = rewrite.before_local;
            let main_after_local = rewrite.after_local;
            let main_assumptions = rewrite.assumption_events.clone();
            let main_required = rewrite.required_conditions.clone();
            let main_poly_proof = rewrite.poly_proof.clone();
            let main_substeps = rewrite.substeps.clone();
            let chained_rewrites = rewrite.chained.clone();

            // Determine final result (last of chained, or main rewrite)
            let final_result = chained_rewrites
                .last()
                .map(|c| c.after)
                .unwrap_or(main_new_expr);

            let global_before = self.root_expr;
            let main_global_after = self.reconstruct_at_path(main_new_expr);

            // Main step
            let mut step = Step::with_snapshots(
                main_description,
                rule.name(),
                expr_id,
                main_new_expr,
                self.current_path.clone(),
                Some(self.context),
                global_before,
                main_global_after,
            );
            step.before_local = main_before_local;
            step.after_local = main_after_local;
            step.assumption_events = main_assumptions;
            step.required_conditions = main_required;
            step.poly_proof = main_poly_proof;
            step.substeps = main_substeps;
            step.importance = rule.importance();
            self.steps.push(step);

            // Trace coherence verification
            debug_assert_eq!(
                main_global_after,
                self.root_expr,
                "[Trace Coherence] Step global_after doesn't match updated root_expr. \
                 Rule: {}, This will cause trace mismatch for next step.",
                rule.name()
            );

            // Process chained rewrites sequentially
            let mut current = main_new_expr;
            for chain_rw in chained_rewrites {
                let chain_global_before = self.reconstruct_at_path(current);
                let chain_global_after = self.reconstruct_at_path(chain_rw.after);

                let mut chain_step = Step::with_snapshots(
                    &chain_rw.description,
                    rule.name(),
                    current,
                    chain_rw.after,
                    self.current_path.clone(),
                    Some(self.context),
                    chain_global_before,
                    chain_global_after,
                );
                chain_step.before_local = chain_rw.before_local;
                chain_step.after_local = chain_rw.after_local;
                chain_step.assumption_events = chain_rw.assumption_events;
                chain_step.required_conditions = chain_rw.required_conditions;
                chain_step.poly_proof = chain_rw.poly_proof;
                chain_step.importance = chain_rw.importance.unwrap_or_else(|| rule.importance());
                chain_step.is_chained = true;
                self.steps.push(chain_step);

                current = chain_rw.after;
            }

            final_result
        } else {
            // Without steps, just compute final result
            rewrite
                .chained
                .last()
                .map(|c| c.after)
                .unwrap_or(rewrite.new_expr)
        }
    }

    fn apply_rules(&mut self, mut expr_id: ExprId) -> ExprId {
        // Note: This loop pattern with early returns is intentional for structured exit points
        #[allow(clippy::never_loop)]
        loop {
            let mut changed = false;
            let variant = crate::helpers::get_variant_name(self.context.get(expr_id));

            // PERF: Build ParentContext ONCE per node, shared by all rules.
            // Previously this was rebuilt inside the rule loop (O(rules × nodes × ancestors)).
            let parent_ctx = self.build_parent_context();

            // println!("apply_rules for {:?} variant: {}", expr_id, variant);
            // Try specific rules
            if let Some(specific_rules) = self.rules.get(variant) {
                for rule in specific_rules {
                    if self.disabled_rules.contains(rule.name()) {
                        self.profiler
                            .record_rejected_disabled(self.current_phase, rule.name());
                        continue;
                    }
                    // Phase ownership: only run rule if allowed in current phase
                    let phase_mask = self.current_phase.mask();
                    if !rule.allowed_phases().contains(phase_mask) {
                        self.profiler
                            .record_rejected_phase(self.current_phase, rule.name());
                        continue;
                    }
                    // SolveSafety filter: in SolvePrepass, only allow Always-safe rules
                    // In SolveTactic, use domain_mode to determine if conditional rules are allowed
                    match self.simplify_purpose {
                        crate::solve_safety::SimplifyPurpose::Eval => {
                            // Eval: all rules allowed (default behavior)
                        }
                        crate::solve_safety::SimplifyPurpose::SolvePrepass => {
                            // Pre-pass: only SolveSafety::Always rules
                            if !rule.solve_safety().safe_for_prepass() {
                                continue;
                            }
                        }
                        crate::solve_safety::SimplifyPurpose::SolveTactic => {
                            // Tactic: check against domain_mode
                            let domain_mode = self.initial_parent_ctx.domain_mode();
                            if !rule.solve_safety().safe_for_tactic(domain_mode) {
                                continue;
                            }
                        }
                    }

                    if let Some(mut rewrite) = rule.apply(self.context, expr_id, &parent_ctx) {
                        // Check semantic equality - skip if no real change
                        // EXCEPTION: Didactic rules should always generate steps
                        // even if result is semantically equivalent (e.g., sqrt(12) → 2*√3)
                        let is_didactic_rule = rule.name() == "Evaluate Numeric Power"
                            || rule.name() == "Sum Exponents";

                        if !is_didactic_rule {
                            use crate::semantic_equality::SemanticEqualityChecker;
                            let checker = SemanticEqualityChecker::new(self.context);
                            if checker.are_equal(expr_id, rewrite.new_expr) {
                                debug!(
                                    "{}[DEBUG] Rule '{}' produced semantically equal result, skipping",
                                    self.indent(),
                                    rule.name()
                                );
                                self.profiler
                                    .record_rejected_semantic(self.current_phase, rule.name());
                                continue;
                            }
                        }

                        // ANTI-WORSEN BUDGET: Reject rewrites that grow expression beyond threshold.
                        // This is a GLOBAL SAFETY NET against exponential explosion (e.g., sin(16*x) expansion).
                        // Budget policy: Block if BOTH:
                        // - Absolute growth > 30 nodes
                        // - Relative growth > 1.5x (50% larger)
                        // Exception: expand_mode bypasses this check (user explicitly requested expansion)
                        if !parent_ctx.is_expand_mode()
                            && crate::helpers::rewrite_worsens_too_much(
                                self.context,
                                expr_id,
                                rewrite.new_expr,
                                30,  // max_growth_abs
                                1.5, // max_growth_ratio
                            )
                        {
                            debug!(
                                "{}[DEBUG] Rule '{}' blocked by anti-worsen budget (expression grew too much)",
                                self.indent(),
                                rule.name()
                            );
                            continue;
                        }

                        // Domain Delta Airbag: skip_unless_analytic=true for specific rules
                        if self.check_domain_airbag(
                            rule.as_ref(),
                            &parent_ctx,
                            expr_id,
                            &mut rewrite,
                            true,
                        ) {
                            continue;
                        }

                        // Record rule application with delta_nodes for health metrics
                        let delta = if self.profiler.is_health_enabled() {
                            let before = crate::helpers::count_all_nodes(self.context, expr_id);
                            let after =
                                crate::helpers::count_all_nodes(self.context, rewrite.new_expr);
                            after as i64 - before as i64
                        } else {
                            0
                        };
                        self.profiler
                            .record_with_delta(self.current_phase, rule.name(), delta);

                        // TRACE: Log applied rules for debugging cycles
                        if *CAS_TRACE_RULES_ENABLED {
                            use std::io::Write;
                            if let Ok(mut f) = std::fs::OpenOptions::new()
                                .create(true)
                                .append(true)
                                .open("/tmp/rule_trace.log")
                            {
                                let node_count_before =
                                    crate::helpers::node_count(self.context, expr_id);
                                let node_count_after =
                                    crate::helpers::node_count(self.context, rewrite.new_expr);
                                let _ = writeln!(
                                    f,
                                    "APPLIED depth={} rule={} nodes={}->{}",
                                    self.current_depth,
                                    rule.name(),
                                    node_count_before,
                                    node_count_after
                                );
                                let _ = f.flush();
                            }
                        }

                        // println!(
                        //     "Rule '{}' applied: {:?} -> {:?}",
                        //     rule.name(),
                        //     expr_id,
                        //     rewrite.new_expr
                        // );
                        debug!(
                            "{}[DEBUG] Rule '{}' applied: {:?} -> {:?}",
                            self.indent(),
                            rule.name(),
                            expr_id,
                            rewrite.new_expr
                        );
                        expr_id = self.record_rewrite_step(rule.as_ref(), expr_id, &rewrite);

                        // Budget tracking: count this rewrite (charged at end of pass)
                        self.rewrite_count += 1;

                        // Note: Rule application tracking for rationalization is now handled by phase, not flag
                        // Apply canonical normalization to prevent loops
                        expr_id = normalize_core_with_cache(
                            self.context,
                            expr_id,
                            &mut self.normalize_cache,
                        );

                        // V2.14.30: Always-On Cycle Detection with blocklist
                        // Reset detector if phase changed since last initialization
                        if self.cycle_phase != Some(self.current_phase) {
                            self.cycle_detector = Some(crate::cycle_detector::CycleDetector::new(
                                self.current_phase,
                            ));
                            self.cycle_phase = Some(self.current_phase);
                            self.fp_memo.clear();
                            // Note: blocked_rules persists across phases (conservative)
                        }

                        let h = crate::cycle_detector::expr_fingerprint(
                            self.context,
                            expr_id,
                            &mut self.fp_memo,
                        );
                        if let Some(detector) = self.cycle_detector.as_mut() {
                            if let Some(info) = detector.observe(h) {
                                // Add to blocklist to prevent re-entry
                                let rule_name_static = rule.name();
                                if self.blocked_rules.insert((h, rule_name_static.to_string())) {
                                    // First time seeing this (fingerprint, rule) - emit hint
                                    // But don't emit for constants/numbers (they're not meaningful cycles)
                                    let is_constant = matches!(
                                        self.context.get(expr_id),
                                        cas_ast::Expr::Number(_) | cas_ast::Expr::Constant(_)
                                    );
                                    if !is_constant {
                                        crate::domain::register_blocked_hint(crate::domain::BlockedHint {
                                        key: crate::assumptions::AssumptionKey::Defined {
                                            expr_fingerprint: h,
                                        },
                                        expr_id,
                                        rule: rule_name_static.to_string(),
                                        suggestion: "cycle detected; consider disabling heuristic rules or tightening budget",
                                    });
                                    }
                                }
                                self.last_cycle = Some(info);
                                // Treat as fixed-point: stop this phase early
                                self.current_depth -= 1;
                                return expr_id;
                            }
                        }

                        changed = true;
                        break;
                    }
                }
            }

            if changed {
                return self.transform_expr_recursive(expr_id);
            }

            // Try global rules
            for rule in self.global_rules {
                if self.disabled_rules.contains(rule.name()) {
                    continue;
                }
                // Phase ownership: only run rule if allowed in current phase
                let phase_mask = self.current_phase.mask();
                if !rule.allowed_phases().contains(phase_mask) {
                    self.profiler
                        .record_rejected_phase(self.current_phase, rule.name());
                    continue;
                }

                // PERF: Reuse the parent_ctx built once per apply_rules() call
                if let Some(mut rewrite) = rule.apply(self.context, expr_id, &parent_ctx) {
                    // Fast path: if rewrite produces identical ExprId, skip entirely
                    if rewrite.new_expr == expr_id {
                        continue;
                    }

                    // Semantic equality check to prevent infinite loops
                    // Skip rewrites that produce semantically equivalent results without improvement
                    let is_didactic_rule =
                        rule.name() == "Evaluate Numeric Power" || rule.name() == "Sum Exponents";

                    if !is_didactic_rule {
                        use crate::semantic_equality::SemanticEqualityChecker;
                        let checker = SemanticEqualityChecker::new(self.context);
                        if checker.are_equal(expr_id, rewrite.new_expr) {
                            // Provably equal - only accept if it improves normal form
                            if !crate::helpers::nf_score_after_is_better(
                                self.context,
                                expr_id,
                                rewrite.new_expr,
                            ) {
                                continue; // Skip - no improvement
                            }
                        }
                    }

                    // Domain Delta Airbag: skip_unless_analytic=false for global rules (unconditional)
                    if self.check_domain_airbag(
                        rule.as_ref(),
                        &parent_ctx,
                        expr_id,
                        &mut rewrite,
                        false,
                    ) {
                        continue;
                    }

                    // Record rule application for profiling
                    self.profiler.record(self.current_phase, rule.name());

                    debug!(
                        "{}[DEBUG] Global Rule '{}' applied: {:?} -> {:?}",
                        self.indent(),
                        rule.name(),
                        expr_id,
                        rewrite.new_expr
                    );
                    expr_id = self.record_rewrite_step(rule.as_ref(), expr_id, &rewrite);

                    // Budget tracking: count this rewrite (charged at end of pass)
                    self.rewrite_count += 1;

                    // Note: Rule application tracking for rationalization is now handled by phase, not flag
                    // Apply canonical normalization to prevent loops
                    expr_id =
                        normalize_core_with_cache(self.context, expr_id, &mut self.normalize_cache);
                    changed = true;
                    break;
                }
            }

            if changed {
                return self.transform_expr_recursive(expr_id);
            }

            self.current_depth -= 1;
            return expr_id;
        }
    }
}
