//! Simplification entry points and loop orchestration.
//!
//! These are the `impl Simplifier` methods that build a `LocalSimplificationTransformer`
//! and delegate to it. Includes `simplify`, `expand`, `apply_rules_loop`, etc.

use super::hold::unwrap_hold_top;
use super::simplifier::Simplifier;
use super::transform::LocalSimplificationTransformer;
use crate::step::Step;
use cas_ast::{hold::strip_all_holds, ExprId};
use std::collections::HashMap;

/// Configuration for a simplification loop pass.
///
/// Replaces the 14-parameter signature of `apply_rules_loop_with_phase_and_mode`.
/// Use `LoopConfig::default()` and override only the fields you need.
#[derive(Debug, Clone)]
pub struct LoopConfig {
    pub phase: crate::phase::SimplifyPhase,
    pub expand_mode: bool,
    pub auto_expand: bool,
    pub expand_budget: crate::phase::ExpandBudget,
    pub domain_mode: crate::domain::DomainMode,
    pub inv_trig: crate::semantics::InverseTrigPolicy,
    pub value_domain: crate::semantics::ValueDomain,
    pub goal: crate::semantics::NormalFormGoal,
    pub simplify_purpose: crate::solve_safety::SimplifyPurpose,
    pub context_mode: crate::options::ContextMode,
    pub autoexpand_binomials: crate::options::AutoExpandBinomials,
    pub heuristic_poly: crate::options::HeuristicPoly,
}

impl Default for LoopConfig {
    fn default() -> Self {
        Self {
            phase: crate::phase::SimplifyPhase::Transform,
            expand_mode: false,
            auto_expand: false,
            expand_budget: crate::phase::ExpandBudget::default(),
            domain_mode: crate::domain::DomainMode::default(),
            inv_trig: crate::semantics::InverseTrigPolicy::default(),
            value_domain: crate::semantics::ValueDomain::default(),
            goal: crate::semantics::NormalFormGoal::default(),
            simplify_purpose: crate::solve_safety::SimplifyPurpose::default(),
            context_mode: crate::options::ContextMode::default(),
            autoexpand_binomials: crate::options::AutoExpandBinomials::Off,
            heuristic_poly: crate::options::HeuristicPoly::On,
        }
    }
}

impl Simplifier {
    pub fn local_simplify(
        &mut self,
        expr_id: ExprId,
        pattern_marks: &crate::pattern_marks::PatternMarks,
    ) -> (ExprId, Vec<Step>) {
        // Default to Core phase for local_simplify (safe, non-expanding)
        self.local_simplify_with_phase(expr_id, pattern_marks, crate::phase::SimplifyPhase::Core)
    }

    pub fn local_simplify_with_phase(
        &mut self,
        expr_id: ExprId,
        pattern_marks: &crate::pattern_marks::PatternMarks,
        phase: crate::phase::SimplifyPhase,
    ) -> (ExprId, Vec<Step>) {
        // Create initial ParentContext with pattern marks
        let initial_parent_ctx =
            crate::parent_context::ParentContext::with_marks(pattern_marks.clone());

        let mut local_transformer = LocalSimplificationTransformer {
            context: &mut self.context,
            rules: &self.rules,
            global_rules: &self.global_rules,
            disabled_rules: &self.disabled_rules,
            steps_mode: self.steps_mode,
            steps: Vec::new(),
            domain_warnings: Vec::new(),
            cache: HashMap::new(),
            current_path: Vec::new(),
            profiler: &mut self.profiler,
            pattern_marks: pattern_marks.clone(),
            initial_parent_ctx,
            root_expr: expr_id,
            current_phase: phase,
            cycle_detector: None,
            cycle_phase: None,
            fp_memo: std::collections::HashMap::new(),
            last_cycle: None,
            blocked_rules: std::collections::HashSet::new(),
            current_depth: 0,
            depth_overflow_warned: false,
            ancestor_stack: Vec::new(),
            // Budget tracking (unified)
            rewrite_count: 0,
            nodes_snap: 0,
            budget_op: crate::budget::Operation::SimplifyCore,
            stop_reason: None,
            simplify_purpose: crate::solve_safety::SimplifyPurpose::default(),
            normalize_cache: std::collections::HashMap::new(),
        };

        // PERF: Set steps-enabled flag so rules can skip description computation
        crate::rule::set_steps_enabled(self.steps_mode != crate::options::StepsMode::Off);

        let new_expr = local_transformer.transform_expr_recursive(expr_id);

        // Extract steps from transformer
        let steps = std::mem::take(&mut local_transformer.steps);
        // Copy domain_warnings to self (survives even in Off mode)
        self.last_domain_warnings = std::mem::take(&mut local_transformer.domain_warnings);
        drop(local_transformer);

        (new_expr, steps)
    }

    pub fn simplify(&mut self, expr_id: ExprId) -> (ExprId, Vec<Step>) {
        self.simplify_with_options(expr_id, crate::phase::SimplifyOptions::default())
    }

    /// Simplify with custom options controlling phases and policies.
    pub fn simplify_with_options(
        &mut self,
        expr_id: ExprId,
        options: crate::phase::SimplifyOptions,
    ) -> (ExprId, Vec<Step>) {
        let (result, steps, _stats) = self.simplify_with_stats(expr_id, options);
        // Unwrap any top-level hold() wrapper so user sees clean result
        let unwrapped = unwrap_hold_top(&self.context, result);
        (unwrapped, steps)
    }

    /// Simplify with options and return pipeline statistics for diagnostics.
    pub fn simplify_with_stats(
        &mut self,
        expr_id: ExprId,
        options: crate::phase::SimplifyOptions,
    ) -> (ExprId, Vec<Step>, crate::phase::PipelineStats) {
        // Clear blocked hints from previous simplifications
        crate::domain::clear_blocked_hints();

        let mut orchestrator = crate::orchestrator::Orchestrator::new();
        orchestrator.enable_polynomial_strategy = self.enable_polynomial_strategy;
        orchestrator.options = options;
        orchestrator.options.collect_steps = self.collect_steps();
        let result = orchestrator.simplify_pipeline(expr_id, self);

        // Collect blocked hints from thread-local to Simplifier field
        self.last_blocked_hints = crate::domain::take_blocked_hints();

        result
    }

    /// Expand without rationalization.
    /// After expansion, recursively strips all __hold() wrappers so user sees clean result.
    pub fn expand(&mut self, expr_id: ExprId) -> (ExprId, Vec<Step>) {
        let (result, steps) =
            self.simplify_with_options(expr_id, crate::phase::SimplifyOptions::for_expand());
        // Strip ALL nested __hold wrappers (not just top-level)
        let clean_result = strip_all_holds(&mut self.context, result);
        (clean_result, steps)
    }

    /// Simplify for solver pre-pass: only safe rules (SolveSafety::Always).
    /// Blocks rules that could corrupt solution sets (those requiring assumptions).
    /// Steps are not collected (pre-pass is invisible to user).
    pub fn simplify_for_solve(&mut self, expr_id: ExprId) -> ExprId {
        let (result, _steps) =
            self.simplify_with_options(expr_id, crate::phase::SimplifyOptions::for_solve_prepass());
        result
    }

    pub fn apply_rules_loop(
        &mut self,
        expr_id: ExprId,
        pattern_marks: &crate::pattern_marks::PatternMarks,
    ) -> (ExprId, Vec<Step>, crate::budget::PassStats) {
        // Default to Transform phase (allows all rules including distribution)
        self.apply_rules_loop_with_phase(
            expr_id,
            pattern_marks,
            crate::phase::SimplifyPhase::Transform,
        )
    }

    pub fn apply_rules_loop_with_phase(
        &mut self,
        expr_id: ExprId,
        pattern_marks: &crate::pattern_marks::PatternMarks,
        phase: crate::phase::SimplifyPhase,
    ) -> (ExprId, Vec<Step>, crate::budget::PassStats) {
        let config = LoopConfig {
            phase,
            ..LoopConfig::default()
        };
        self.apply_rules_loop_with_config(expr_id, pattern_marks, &config)
    }

    /// Apply rules loop with full configuration via `LoopConfig`.
    /// Returns PassStats for the caller to charge the Budget.
    pub fn apply_rules_loop_with_config(
        &mut self,
        expr_id: ExprId,
        pattern_marks: &crate::pattern_marks::PatternMarks,
        config: &LoopConfig,
    ) -> (ExprId, Vec<Step>, crate::budget::PassStats) {
        let phase = config.phase;
        let expand_mode = config.expand_mode;
        let auto_expand = config.auto_expand;
        let expand_budget = config.expand_budget;
        let domain_mode = config.domain_mode;
        let inv_trig = config.inv_trig;
        let value_domain = config.value_domain;
        let goal = config.goal;
        let simplify_purpose = config.simplify_purpose;
        let context_mode = config.context_mode;
        let autoexpand_binomials = config.autoexpand_binomials;
        let heuristic_poly = config.heuristic_poly;
        let rules = &self.rules;
        let global_rules = &self.global_rules;
        let steps_mode = self.steps_mode;

        // Create initial ParentContext with pattern marks, expand_mode, auto-expand, domain_mode, inv_trig, value_domain, goal, simplify_purpose, and context_mode
        // V2.15: Reset domain inference call counter for regression testing
        crate::implicit_domain::infer_domain_calls_reset();

        // V2.15.8: Use sticky implicit domain if set (from original input), otherwise calculate from current expr
        // This preserves inherited requires (e.g., x≥0 from sqrt) across all phases
        let initial_parent_ctx = if let Some(sticky_domain) = &self.sticky_implicit_domain {
            let sticky_root = self.sticky_root_expr.unwrap_or(expr_id);
            crate::parent_context::ParentContext::with_expand_mode(
                pattern_marks.clone(),
                expand_mode,
            )
            .with_auto_expand_flag(
                auto_expand,
                if auto_expand {
                    Some(expand_budget)
                } else {
                    None
                },
            )
            .with_domain_mode(domain_mode)
            .with_inv_trig(inv_trig)
            .with_value_domain(value_domain)
            .with_goal(goal)
            .with_simplify_purpose(simplify_purpose)
            .with_context_mode(context_mode)
            .with_root_expr_only(sticky_root)
            .with_implicit_domain(Some(sticky_domain.clone()))
            .with_autoexpand_binomials(autoexpand_binomials)
            .with_heuristic_poly(heuristic_poly)
        } else {
            crate::parent_context::ParentContext::with_expand_mode(
                pattern_marks.clone(),
                expand_mode,
            )
            .with_auto_expand_flag(
                auto_expand,
                if auto_expand {
                    Some(expand_budget)
                } else {
                    None
                },
            )
            .with_domain_mode(domain_mode)
            .with_inv_trig(inv_trig)
            .with_value_domain(value_domain)
            .with_goal(goal)
            .with_simplify_purpose(simplify_purpose)
            .with_context_mode(context_mode)
            .with_root_expr(&self.context, expr_id)
            .with_autoexpand_binomials(autoexpand_binomials)
            .with_heuristic_poly(heuristic_poly)
        };

        // Capture nodes_created BEFORE creating transformer (can't access while borrowed)
        let nodes_snap = self.context.stats().nodes_created;

        let mut local_transformer = LocalSimplificationTransformer {
            context: &mut self.context,
            rules,
            global_rules,
            disabled_rules: &self.disabled_rules,
            steps_mode,
            steps: Vec::new(),
            domain_warnings: Vec::new(),
            cache: HashMap::new(),
            current_path: Vec::new(),
            profiler: &mut self.profiler,
            pattern_marks: pattern_marks.clone(),
            initial_parent_ctx,
            root_expr: expr_id,
            current_phase: phase,
            cycle_detector: None,
            cycle_phase: None,
            fp_memo: std::collections::HashMap::new(),
            last_cycle: None,
            blocked_rules: std::collections::HashSet::new(),
            current_depth: 0,
            depth_overflow_warned: false,
            ancestor_stack: Vec::new(),
            // Budget tracking (unified)
            rewrite_count: 0,
            nodes_snap,
            budget_op: match phase {
                crate::phase::SimplifyPhase::Core | crate::phase::SimplifyPhase::PostCleanup => {
                    crate::budget::Operation::SimplifyCore
                }
                crate::phase::SimplifyPhase::Transform
                | crate::phase::SimplifyPhase::Rationalize => {
                    crate::budget::Operation::SimplifyTransform
                }
            },
            stop_reason: None,
            simplify_purpose,
            normalize_cache: std::collections::HashMap::new(),
        };

        // PERF: Set steps-enabled flag so rules can skip description computation
        crate::rule::set_steps_enabled(steps_mode != crate::options::StepsMode::Off);

        let new_expr = local_transformer.transform_expr_recursive(expr_id);

        // Extract budget tracking stats BEFORE dropping transformer
        let rewrite_count = local_transformer.rewrite_count;
        let budget_op = local_transformer.budget_op;
        let nodes_snap = local_transformer.nodes_snap;
        let stop_reason = local_transformer.stop_reason.take();

        // Extract steps from transformer
        let steps = std::mem::take(&mut local_transformer.steps);
        // Copy domain_warnings to self (survives even in Off mode)
        self.last_domain_warnings
            .append(&mut local_transformer.domain_warnings);
        drop(local_transformer);

        // Calculate nodes delta AFTER dropping transformer (now we can borrow self.context)
        let nodes_delta = self
            .context
            .stats()
            .nodes_created
            .saturating_sub(nodes_snap);

        // Build PassStats for caller to use with Budget.charge()
        let pass_stats = crate::budget::PassStats {
            op: budget_op,
            rewrite_count,
            nodes_delta,
            terms_materialized: 0, // Simplify doesn't expand terms
            poly_ops: 0,           // Simplify doesn't do poly ops
            stop_reason,
        };

        (new_expr, steps, pass_stats)
    }
}
