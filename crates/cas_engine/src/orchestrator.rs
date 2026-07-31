use crate::best_so_far::{BestSoFar, BestSoFarBudget};
use crate::expand::eager_eval_expand_calls;
use crate::phase::{SimplifyOptions, SimplifyPhase};
use crate::rule::Rule;
use crate::{Simplifier, Step};
use cas_ast::ordering::compare_expr;
use cas_ast::{BuiltinFn, Constant, Context, Expr, ExprId};
use cas_formatter::render_expr;
use cas_math::abs_support::try_unwrap_abs_arg;
use cas_math::arithmetic_rule_support::try_rewrite_combine_constants_expr;
use cas_math::build::mul2_raw;
use cas_math::expansion_rule_support::{
    try_expand_binomial_pow_expr, try_expand_small_pow_sum_expr, SmallPowExpandPolicy,
};
use cas_math::expr_extract::{extract_exp_argument, extract_i64_integer};
use cas_math::expr_nary::{build_balanced_add, build_balanced_mul, AddView, MulView, Sign};
use cas_math::expr_rewrite::smart_mul;
use cas_math::factoring_support::try_rewrite_automatic_factor_expr;
use cas_math::fraction_power_cancel_support::try_rewrite_cancel_same_base_powers_div_expr;
use cas_math::hyperbolic_identity_support::{
    try_rewrite_hyperbolic_double_angle_sum, try_rewrite_hyperbolic_triple_angle,
    try_rewrite_recognize_hyperbolic_from_exp, try_rewrite_tanh_double_angle_expansion,
    try_rewrite_tanh_to_sinh_cosh,
};
use cas_math::infinity_support::{is_negative_literal, is_positive_literal};
use cas_math::logarithm_inverse_support::{
    expand_logs_collect_positive_assumptions, log_exp_inverse_policy_mode_from_flags,
    plan_log_power_base_numeric_policy, try_rewrite_exponential_log_inverse_expr,
    try_rewrite_log_power_base_numeric_expr,
};
use cas_math::perfect_square_support::rational_sqrt;
use cas_math::pi_helpers::extract_rational_pi_multiple;
use cas_math::poly_lowering;
use cas_math::poly_store::clear_thread_local_store;
use cas_math::reciprocal_sqrt_canon_support::try_rewrite_reciprocal_sqrt_canon_expr;
use cas_math::root_forms::{
    extract_square_root_base, try_rewrite_canonical_root_expr,
    try_rewrite_extract_perfect_power_from_radicand_expr, try_rewrite_simplify_square_root_expr,
    SimplifySquareRootRewriteKind,
};
use cas_math::semantic_equality::SemanticEqualityChecker;
use cas_math::trig_canonicalization_support::{
    try_rewrite_cot_to_csc_pythagorean_identity_expr,
    try_rewrite_csc_cot_pythagorean_identity_expr, try_rewrite_sec_tan_pythagorean_identity_expr,
    try_rewrite_tan_to_sec_pythagorean_identity_expr, try_rewrite_tan_to_sin_cos_function_expr,
};
use cas_math::trig_contraction_support::try_rewrite_angle_sum_fraction_to_tan_expr;
use cas_math::trig_core_identity_support::{
    try_rewrite_legacy_evaluate_trig_expr, try_rewrite_pythagorean_identity_add_expr,
};
use cas_math::trig_eval_table_support::lookup_trig_or_inverse;
use cas_math::trig_half_angle_support::{
    extract_trig_half_angle, try_rewrite_hyperbolic_half_angle_squares_expr,
};
use cas_math::trig_identity_zero_support::try_rewrite_sin_sum_triple_identity_zero_expr;
use cas_math::trig_inverse_expansion_support::try_rewrite_trig_inverse_composition_expr;
use cas_math::trig_linear_support::{
    build_coef_times_base, extract_coef_and_base, extract_linear_coefficients,
};
use cas_math::trig_multi_angle_support::{
    try_rewrite_double_angle_function_expr, try_rewrite_quintuple_angle_expr,
    try_rewrite_triple_angle_expr,
};
use cas_math::trig_phase_shift_support::try_rewrite_trig_phase_shift_function_expr;
use cas_math::trig_power_identity_support::{
    extract_coeff_trig_pow2, extract_coeff_trig_pow4, try_rewrite_pythagorean_chain_add_expr,
    try_rewrite_pythagorean_factor_form_add_expr,
    try_rewrite_pythagorean_generic_coefficient_add_expr,
    try_rewrite_reciprocal_product_pythagorean_zero_add_expr,
    try_rewrite_trig_fourth_power_difference_add_expr,
};
use cas_math::trig_roots_flatten::flatten_mul_chain;
use cas_math::trig_roots_flatten::{extract_double_angle_arg_relaxed, extract_triple_angle_arg};
use cas_math::trig_sum_product_support::{
    try_rewrite_product_to_sum_expr, try_rewrite_sum_to_product_contraction_expr,
};
use cas_math::trig_value_detection_support::detect_special_angle;
use cas_math::trig_values::lookup_trig_value;
use cas_math::trig_weierstrass_support::try_rewrite_weierstrass_contraction_div_expr;
use cas_solver_core::rationalize_policy::AutoRationalizeLevel;
use cas_solver_core::rule_names::{RULE_CANCEL_EXACT_ADDITIVE_PAIRS, RULE_EXPAND_LOG_ABS_MUL_DIV};
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};
use std::cmp::Ordering;
use std::collections::HashSet;
use std::time::Duration;
use web_time::Instant;

#[derive(Clone, Copy, Default)]
struct HotDirectSmallZeroFamilyFlags {
    has_log: bool,
    has_trig: bool,
    has_hyperbolic: bool,
    has_division: bool,
}

#[derive(Clone, Copy)]
struct NumericGeneralPhaseShiftTargetRoot {
    trig_fn: BuiltinFn,
    base_arg: ExprId,
    subtract_shift: bool,
    global_sign: i8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OddHalfPowerProductFormRoot {
    base: ExprId,
    outside_power: i64,
}

thread_local! {
    static ISOLATED_SIMPLIFY_NESTING: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Isolated probe pipelines nest geometrically on mixed trig+rational sums:
/// the root-shortcut layer probes each additive child with a FULL pipeline,
/// whose own root-shortcut layer probes ITS children the same way
/// (`cos(2x)/2 + 1/(u(u+1)) - 1/u + 1/(u+1)` spun >30s in BOTH steps modes,
/// while `run_default_simplify`'s twin guard already capped its probes at
/// two levels). Deeper levels only re-derive the same subprobes; declining
/// is sound — an unproven probe keeps the expression unsimplified.
const ISOLATED_SIMPLIFY_MAX_NESTING: usize = 1;

struct IsolatedSimplifyNestingGuard;

impl Drop for IsolatedSimplifyNestingGuard {
    fn drop(&mut self) {
        ISOLATED_SIMPLIFY_NESTING.with(|depth| depth.set(depth.get().saturating_sub(1)));
    }
}

thread_local! {
    // Exact-zero probe memo, keyed by (Context::instance_tag, options
    // fingerprint, expr, target). The root shortcuts re-ask the same
    // "does this subtree simplify to that target?" question from every
    // pipeline the solver runs: measured 54 calls over 7 distinct probes
    // on the cubic rational-root solve, and 1211 over 47 on the abs-split
    // solve. Each miss is a FULL fresh pipeline, so a hit pays for the
    // whole scheme immediately. The options fingerprint keeps answers
    // computed under different semantic axes (SolvePrepass vs Eval,
    // domain/value modes, ...) apart; nesting-guard refusals are NOT
    // cached; sized-capped, never cleared (tag keys cannot go stale).
    static ISOLATED_SIMPLIFY_PROBE_MEMO: std::cell::RefCell<
        rustc_hash::FxHashMap<(u64, u64, ExprId, ExprId), bool>,
    > = std::cell::RefCell::new(rustc_hash::FxHashMap::default());
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SmallTrigOrHyperbolicNumericSubsetTermRootKind {
    Number,
    TrigOrHyperbolic,
    Other,
}

pub struct Orchestrator {
    // Configuration for the pipeline
    pub max_iterations: usize,
    pub enable_polynomial_strategy: bool,
    /// Pre-scanned pattern marks for context-aware guards
    pub pattern_marks: crate::pattern_marks::PatternMarks,
    /// Expr these marks were last computed for; reused when the tree is unchanged.
    pub pattern_marks_expr: Option<ExprId>,
    /// Pipeline options (budgets, transform/rationalize control)
    pub options: SimplifyOptions,
}

impl Default for Orchestrator {
    fn default() -> Self {
        Self::new()
    }
}

impl Orchestrator {
    pub fn new() -> Self {
        Self {
            max_iterations: 10,
            enable_polynomial_strategy: true,
            pattern_marks: crate::pattern_marks::PatternMarks::new(),
            pattern_marks_expr: None,
            options: SimplifyOptions::default(),
        }
    }

    /// Create orchestrator for expand() command (no rationalization)
    pub fn for_expand() -> Self {
        let mut o = Self::new();
        o.options = SimplifyOptions::for_expand();
        o
    }

    #[inline]
    fn initialize_deadline_if_needed(&mut self) {
        if self.options.deadline.is_some() {
            return;
        }
        let Some(time_budget_ms) = self.options.time_budget_ms else {
            return;
        };
        let now = Instant::now();
        self.options.deadline = Some(
            now.checked_add(Duration::from_millis(time_budget_ms))
                .unwrap_or(now),
        );
    }

    #[inline]
    fn time_budget_exceeded(&self) -> bool {
        self.options
            .deadline
            .is_some_and(|deadline| Instant::now() >= deadline)
    }

    fn finish_timed_out_pipeline(
        &self,
        simplifier: &mut Simplifier,
        current: ExprId,
        all_steps: Vec<Step>,
        mut pipeline_stats: crate::phase::PipelineStats,
        best: Option<&BestSoFar>,
    ) -> (ExprId, Vec<Step>, crate::phase::PipelineStats) {
        let current_score = crate::best_so_far::score_expr(&simplifier.context, current);
        let (final_expr, final_steps) = match best {
            Some(best) if best.best_score() < current_score => {
                let best_expr = best.best_expr();
                let best_steps = if self.options.collect_steps {
                    best.best_steps_prefix(&all_steps)
                } else {
                    all_steps
                };
                (best_expr, best_steps)
            }
            _ => (current, all_steps),
        };

        if self.options.shared.assumption_reporting != crate::AssumptionReporting::Off {
            pipeline_stats.assumptions = crate::collect_assumption_records_from_iter(
                final_steps
                    .iter()
                    .flat_map(|step| step.assumption_events().iter().cloned()),
            );
        }
        pipeline_stats.cycle_events = cas_solver_core::cycle_event_registry::take_cycle_events();
        pipeline_stats.timed_out = true;
        simplifier.clear_sticky_implicit_domain();
        (final_expr, final_steps, pipeline_stats)
    }

    /// Run a single phase of the pipeline until fixed point or budget exhausted.
    ///
    /// Returns the simplified expression, steps, and phase statistics.
    fn run_phase(
        &mut self,
        simplifier: &mut Simplifier,
        start: ExprId,
        phase: SimplifyPhase,
        max_iters: usize,
    ) -> (ExprId, Vec<Step>, crate::phase::PhaseStats) {
        let phase_label = pipeline_phase_profile_label(phase);
        let sample =
            crate::orchestrator_shortcut_profiler::orchestrator_shortcut_profiling_enabled()
                .then(|| render_expr_for_orchestrator_profile(&simplifier.context, start));
        run_profiled_orchestrator_section(phase_label, sample, || {
            use crate::phase::PhaseStats;

            let mut current = start;
            let mut all_steps = Vec::new();
            let mut seen_hashes: HashSet<u64> = HashSet::new();
            let mut stats = PhaseStats::new(phase);

            tracing::debug!(
                target: "simplify",
                phase = %phase,
                budget = max_iters,
                "phase_start"
            );

            for iter in 0..max_iters {
                if self.time_budget_exceeded() {
                    stats.timed_out = true;
                    stats.iters_used = iter;
                    tracing::warn!(
                        target: "simplify",
                        phase = %phase,
                        iters = stats.iters_used,
                        "phase_timeout_before_iteration"
                    );
                    break;
                }

                let is_solve_mode =
                    self.options.shared.context_mode == crate::options::ContextMode::Solve;
                if self.pattern_marks_expr != Some(current) {
                    self.pattern_marks = crate::pattern_marks::PatternMarks::new();
                    crate::pattern_scanner::scan_and_mark_patterns(
                        &simplifier.context,
                        current,
                        &mut self.pattern_marks,
                    );

                    // Auto-expand scanner: mark cancellation contexts (difference quotients)
                    // Only skip in Solve mode (which should never auto-expand to preserve structure)
                    // The scanner has its own strict budgets (n=2, base_terms<=3) so it's safe to always run
                    if !is_solve_mode {
                        let math_budget =
                            to_math_auto_expand_budget(&self.options.shared.expand_budget);
                        cas_math::auto_expand_scan::mark_auto_expand_candidates(
                            &simplifier.context,
                            current,
                            &math_budget,
                            &mut self.pattern_marks,
                        );
                    }
                    self.pattern_marks_expr = Some(current);
                }
                let global_auto_expand = self.options.shared.expand_policy
                    == crate::phase::ExpandPolicy::Auto
                    && !is_solve_mode;
                let config = crate::engine::LoopConfig {
                    phase,
                    deadline: self.options.deadline,
                    expand_mode: self.options.expand_mode,
                    auto_expand: global_auto_expand,
                    expand_budget: self.options.shared.expand_budget,
                    domain_mode: self.options.shared.semantics.domain_mode,
                    inv_trig: self.options.shared.semantics.inv_trig,
                    value_domain: self.options.shared.semantics.value_domain,
                    goal: self.options.goal,
                    simplify_purpose: self.options.simplify_purpose,
                    context_mode: self.options.shared.context_mode,
                    autoexpand_binomials: self.options.shared.autoexpand_binomials,
                    heuristic_poly: self.options.shared.heuristic_poly,
                    suppress_depth_overflow_warnings: self.options.suppress_depth_overflow_warnings,
                };
                let pass_profile_labels = active_pipeline_phase_pass_profile_labels(phase);
                let pass_profile_sample = pass_profile_labels.is_some().then(|| {
                    format!(
                        "iter={} {}",
                        iter,
                        render_expr_for_orchestrator_profile(&simplifier.context, current)
                    )
                });
                let pass_profile_start =
                    pass_profile_labels.is_some().then(std::time::Instant::now);
                let (next, steps, pass_stats) =
                    simplifier.apply_rules_loop_with_config(current, &self.pattern_marks, &config);
                if let (Some(pass_profile_start), Some((changed_label, fixed_label))) =
                    (pass_profile_start, pass_profile_labels)
                {
                    let pass_changed =
                        next != current || !steps.is_empty() || pass_stats.rewrite_count > 0;
                    let pass_profile_label = if pass_changed {
                        changed_label
                    } else {
                        fixed_label
                    };
                    if let Some(pass_profile_sample) = pass_profile_sample {
                        crate::orchestrator_shortcut_profiler::record_orchestrator_shortcut_sample(
                            pass_profile_label,
                            pass_profile_sample,
                        );
                    }
                    crate::orchestrator_shortcut_profiler::record_orchestrator_shortcut_attempt(
                        pass_profile_label,
                        true,
                        pass_profile_start.elapsed(),
                    );
                }

                if phase == SimplifyPhase::Core && next != current {
                    profile_root_exact_zero_multiterm_trig_numeric_subset_status(
                        &self.options,
                        &mut simplifier.context,
                        next,
                        "phase.core.post_pass.exact_zero.multiterm_trig_numeric_subset",
                    );
                }

                // Log budget stats for this iteration (actual charging done by caller if Budget provided)
                if pass_stats.rewrite_count > 0 || pass_stats.nodes_delta > 0 {
                    tracing::trace!(
                        target: "budget",
                        op = %pass_stats.op,
                        rewrites = pass_stats.rewrite_count,
                        nodes_delta = pass_stats.nodes_delta,
                        "pass_budget_stats"
                    );
                }

                // Warn user when budget limit was reached (best-effort mode)
                if let Some(ref exceeded) = pass_stats.stop_reason {
                    tracing::warn!(
                        target: "budget",
                        op = %exceeded.op,
                        metric = %exceeded.metric,
                        used = exceeded.used,
                        limit = exceeded.limit,
                        "Budget limit reached: {}/{} (used {}, limit {}). Returned partial result.",
                        exceeded.op,
                        exceeded.metric,
                        exceeded.used,
                        exceeded.limit
                    );
                }

                stats.rewrites_used += steps.len();
                all_steps.extend(steps);

                if self.time_budget_exceeded() {
                    current = next;
                    stats.iters_used = iter + 1;
                    stats.timed_out = true;
                    tracing::warn!(
                        target: "simplify",
                        phase = %phase,
                        iters = stats.iters_used,
                        rewrites = stats.rewrites_used,
                        "phase_timeout_after_pass"
                    );
                    break;
                }

                // Hidden solve fast path: once Core collapses to a terminal value or a
                // plain symbolic closed form, another full Core pass is only paying the
                // fixed-point check. Later pipeline decisions are still made by the
                // caller after this phase returns.
                if phase == SimplifyPhase::Core
                    && !self.options.collect_steps
                    && is_solve_mode
                    && next != current
                    && (is_terminal_after_core(&simplifier.context, next)
                        || is_plain_symbolic_binomial_after_core(&simplifier.context, next)
                        || is_plain_symbolic_cube_trinomial_after_core(&simplifier.context, next)
                        || (!self.options.shared.semantics.domain_mode.is_strict()
                            && matches!(simplifier.context.get(current), Expr::Div(_, _))
                            && is_plain_symbolic_power_after_core(&simplifier.context, next)))
                {
                    current = next;
                    stats.iters_used = iter + 1;
                    tracing::debug!(
                        target: "simplify",
                        phase = %phase,
                        iters = stats.iters_used,
                        rewrites = stats.rewrites_used,
                        "phase_early_exit_after_closed_form"
                    );
                    break;
                }

                // Fixed point check
                if next == current {
                    stats.iters_used = iter + 1;
                    tracing::debug!(
                        target: "simplify",
                        phase = %phase,
                        iters = stats.iters_used,
                        rewrites = stats.rewrites_used,
                        "phase_fixed_point"
                    );
                    break;
                }

                // Cycle detection: HashSet catches cycles of any period
                let hash =
                    cas_math::expr_semantic_hash::semantic_hash(&simplifier.context, current);
                if !seen_hashes.insert(hash) {
                    // Emit cycle event for the registry
                    cas_solver_core::cycle_event_registry::register_cycle_event_for_expr(
                        &simplifier.context,
                        current,
                        phase,
                        0, // unknown period at inter-iteration level
                        cas_solver_core::cycle_models::CycleLevel::InterIteration,
                        "(inter-iteration)",
                        hash,
                        iter,
                    );
                    stats.iters_used = iter + 1;
                    tracing::warn!(
                        target: "simplify",
                        phase = %phase,
                        iters = stats.iters_used,
                        "cycle_detected"
                    );
                    break;
                }

                current = next;
                stats.iters_used = iter + 1;
            }

            stats.changed = current != start;

            tracing::debug!(
                target: "simplify",
                phase = %phase,
                iters = stats.iters_used,
                rewrites = stats.rewrites_used,
                changed = stats.changed,
                "phase_end"
            );

            (current, all_steps, stats)
        })
    }

    /// Simplify using explicit phase pipeline.
    ///
    /// Pipeline order: Core → Transform → Rationalize → PostCleanup
    ///
    /// Key invariant: Transform never runs after Rationalize.
    pub fn simplify_pipeline(
        &mut self,
        expr: ExprId,
        simplifier: &mut Simplifier,
    ) -> (ExprId, Vec<Step>, crate::phase::PipelineStats) {
        let (result, steps, stats) = self.simplify_pipeline_inner(expr, simplifier);
        // Universal soundness backstop: the pipeline (root shortcuts + phases) must
        // not collapse a non-finite/undefined additive combination into a purely
        // finite value. `inf - inf`, `1/0 - 1/0 + 2/0 - 2/0`, `sqrt(inf) - sqrt(inf)`
        // are indeterminate, not `0`. If a shortcut produced such a result, discard
        // it and keep the input symbolic (a per-node sound path may still fold it to
        // `undefined`).
        if cas_math::arithmetic_cancel_support::rewrite_unsoundly_drops_nonfinite_in_domain(
            &simplifier.context,
            expr,
            result,
            cas_math::abs_support::value_domain_mode_from_flag(
                self.options.shared.semantics.value_domain.is_real_only(),
            ),
        ) {
            return (expr, Vec::new(), stats);
        }
        (result, steps, stats)
    }

    fn simplify_pipeline_inner(
        &mut self,
        expr: ExprId,
        simplifier: &mut Simplifier,
    ) -> (ExprId, Vec<Step>, crate::phase::PipelineStats) {
        self.initialize_deadline_if_needed();
        let _probe_budget_scope =
            crate::rules::arithmetic::enter_default_simplify_probe_budget_scope(
                self.options.shared.semantics.value_domain,
            );

        if self.time_budget_exceeded() {
            return self.finish_timed_out_pipeline(
                simplifier,
                expr,
                Vec::new(),
                crate::phase::PipelineStats::default(),
                None,
            );
        }

        // Extract collect_steps early so pre-passes can skip Step construction
        let collect_steps = self.options.collect_steps;
        let is_solve_mode = self.options.shared.context_mode == crate::options::ContextMode::Solve;
        self.pattern_marks_expr = None;

        // SOUNDNESS / CONSISTENCY: fold `∞/∞` — any quotient of two infinite-factor-bearing
        // expressions (`(2·∞)/(5·∞)`, `(x·∞)/(2·x·∞)`, `(2·∞·sin x)/(5·∞·sin x)`) — to `undefined`
        // HERE, before any common-factor / scalar-multiple cancellation shortcut or Core rule can
        // mis-cancel the `∞` factor and fabricate a finite value (`2/5`, `1`, `tan x`). Several
        // distinct cancellation primitives (plain-mode root shortcuts AND per-node Core rules) race
        // the Core `InfDivInfRule`; in plain mode (no step listener) a cancellation wins, so the
        // result used to depend on whether steps were collected. The fold is RECURSIVE so a NESTED
        // `∞/∞` (`((2·∞)/(5·∞))^2`, `sqrt((2·∞)/(5·∞))`, `1 + (2·∞)/(3·∞)`) cannot escape via a
        // cancellation on the inner `Div`; `undefined` is propagated through the enclosing arithmetic
        // exactly as the engine's undefined-propagation rules do. Runs in both modes -> they agree.
        if let Some(folded) =
            cas_math::infinity_support::fold_inf_div_inf_recursive(&mut simplifier.context, expr)
        {
            // Return early only when the fold fully collapses to `undefined` (every divergence case);
            // a partial fold blocked by `Hold`/`Matrix` falls through to the normal pipeline.
            if matches!(
                simplifier.context.get(folded),
                Expr::Constant(cas_ast::Constant::Undefined)
            ) {
                let shortcut_steps = if collect_steps {
                    vec![build_root_shortcut_compact_step(
                        expr,
                        folded,
                        "∞ / ∞ is indeterminate",
                        "Indeterminate Infinity Quotient",
                    )]
                } else {
                    Vec::new()
                };
                return (
                    folded,
                    shortcut_steps,
                    crate::phase::PipelineStats::default(),
                );
            }
        }

        if let Some(zero) =
            try_div_add_common_factor_residual_root_zero(&mut simplifier.context, expr)
        {
            let shortcut_steps = if collect_steps {
                vec![build_root_shortcut_compact_step(
                    expr,
                    zero,
                    "Collapse factored quotient residual",
                    "Factored Quotient Residual",
                )]
            } else {
                Vec::new()
            };
            return (zero, shortcut_steps, crate::phase::PipelineStats::default());
        }

        if let Some(rewrite) =
            crate::rules::arithmetic::try_build_hyperbolic_pythagorean_factor_root_zero_rewrite(
                &mut simplifier.context,
                expr,
            )
        {
            let (zero, shortcut_steps) = finish_root_shortcut_with_rewrite_meta(
                &simplifier.context,
                expr,
                rewrite,
                "Hyperbolic Pythagorean Residual",
                collect_steps,
            );
            return (zero, shortcut_steps, crate::phase::PipelineStats::default());
        }

        if let Some(zero) = crate::calculus_residual_support::try_diff_hyperbolic_residual_root_zero(
            &mut simplifier.context,
            expr,
        ) {
            let shortcut_steps = if collect_steps {
                vec![build_root_shortcut_compact_step(
                    expr,
                    zero,
                    "Cancel matching hyperbolic derivative residual",
                    "Hyperbolic Diff Residual",
                )]
            } else {
                Vec::new()
            };
            return (zero, shortcut_steps, crate::phase::PipelineStats::default());
        }

        if let Some((zero, required_conditions)) =
            crate::calculus_residual_support::try_diff_sqrt_acosh_split_radical_residual_root_zero(
                &mut simplifier.context,
                expr,
            )
        {
            simplifier.extend_required_conditions(required_conditions.clone());
            let shortcut_steps = if collect_steps {
                let mut step = build_root_shortcut_compact_step(
                    expr,
                    zero,
                    "Cancel matching acosh square-root derivative residual",
                    "Acosh Sqrt Diff Residual",
                );
                step.meta_mut().required_conditions = required_conditions;
                vec![step]
            } else {
                Vec::new()
            };
            return (zero, shortcut_steps, crate::phase::PipelineStats::default());
        }

        if let Some((zero, required_conditions)) = crate::calculus_residual_support::
            try_diff_acosh_affine_reciprocal_sqrt_residual_root_zero(
                &mut simplifier.context,
                expr,
            )
        {
            simplifier.extend_required_conditions(required_conditions.clone());
            let shortcut_steps = if collect_steps {
                let mut step = build_root_shortcut_compact_step(
                    expr,
                    zero,
                    "Cancel matching acosh reciprocal-root derivative residual",
                    "Acosh Reciprocal-Root Diff Residual",
                );
                step.meta_mut().required_conditions = required_conditions;
                vec![step]
            } else {
                Vec::new()
            };
            return (zero, shortcut_steps, crate::phase::PipelineStats::default());
        }

        if let Some((zero, required_conditions)) =
            crate::calculus_residual_support::try_diff_ln_sqrt_polynomial_gap_residual_root_zero(
                &mut simplifier.context,
                expr,
            )
        {
            simplifier.extend_required_conditions(required_conditions.clone());
            let shortcut_steps = if collect_steps {
                let mut step = build_root_shortcut_compact_step(
                    expr,
                    zero,
                    "Cancel matching logarithmic square-root derivative residual",
                    "Ln Sqrt Diff Residual",
                );
                step.meta_mut().required_conditions = required_conditions;
                vec![step]
            } else {
                Vec::new()
            };
            return (zero, shortcut_steps, crate::phase::PipelineStats::default());
        }

        if let Some(zero) =
            crate::calculus_residual_support::try_diff_reciprocal_trig_residual_root_zero(
                &mut simplifier.context,
                expr,
            )
        {
            let shortcut_steps = if collect_steps {
                vec![build_root_shortcut_compact_step(
                    expr,
                    zero,
                    "Cancel matching reciprocal trig derivative residual",
                    "Reciprocal Trig Diff Residual",
                )]
            } else {
                Vec::new()
            };
            return (zero, shortcut_steps, crate::phase::PipelineStats::default());
        }

        if let Some((zero, required_conditions)) =
            crate::calculus_residual_support::try_diff_inverse_reciprocal_trig_residual_root_zero(
                &mut simplifier.context,
                expr,
            )
        {
            simplifier.extend_required_conditions(required_conditions.clone());
            let shortcut_steps = if collect_steps {
                let mut step = build_root_shortcut_compact_step(
                    expr,
                    zero,
                    "Cancel matching inverse reciprocal trig derivative residual",
                    "Inverse Reciprocal Trig Diff Residual",
                );
                step.meta_mut().required_conditions = required_conditions;
                vec![step]
            } else {
                Vec::new()
            };
            return (zero, shortcut_steps, crate::phase::PipelineStats::default());
        }

        if let Some((one, required_conditions)) =
            crate::calculus_residual_support::try_diff_inverse_reciprocal_trig_shifted_quotient_root_one(
                &mut simplifier.context,
                expr,
            )
        {
            simplifier.extend_required_conditions(required_conditions.clone());
            let shortcut_steps = if collect_steps {
                let mut step = build_root_shortcut_compact_step(
                    expr,
                    one,
                    "Collapse shifted quotient of matching inverse reciprocal trig derivative",
                    "Inverse Reciprocal Trig Diff Shifted Quotient",
                );
                step.meta_mut().required_conditions = required_conditions;
                vec![step]
            } else {
                Vec::new()
            };
            return (one, shortcut_steps, crate::phase::PipelineStats::default());
        }

        if let Some((compact, required_conditions)) =
            crate::calculus_residual_support::try_diff_inverse_reciprocal_trig_shifted_quotient_compact_mismatch(
                &mut simplifier.context,
                expr,
            )
        {
            simplifier.extend_required_conditions(required_conditions.clone());
            let shortcut_steps = if collect_steps {
                let mut step = build_root_shortcut_compact_step(
                    expr,
                    compact,
                    "Compact nonmatching shifted quotient of inverse reciprocal trig derivative",
                    "Inverse Reciprocal Trig Diff Shifted Quotient Mismatch",
                );
                step.meta_mut().required_conditions = required_conditions;
                vec![step]
            } else {
                Vec::new()
            };
            return (compact, shortcut_steps, crate::phase::PipelineStats::default());
        }

        if let Some((zero, required_conditions)) =
            crate::calculus_residual_support::try_diff_integral_reciprocal_trig_residual_root_zero(
                &mut simplifier.context,
                expr,
            )
        {
            simplifier.extend_required_conditions(required_conditions.clone());
            let shortcut_steps = if collect_steps {
                let mut step = build_root_shortcut_compact_step(
                    expr,
                    zero,
                    "Verify matching reciprocal trig antiderivative residual",
                    "Reciprocal Trig Integral Residual",
                );
                step.meta_mut().required_conditions = required_conditions;
                vec![step]
            } else {
                Vec::new()
            };
            return (zero, shortcut_steps, crate::phase::PipelineStats::default());
        }

        if let Some((compact, required_conditions)) = crate::calculus_residual_support::
            try_diff_integral_reciprocal_trig_residual_constant_passthrough_quotient(
                &mut simplifier.context,
                expr,
            )
        {
            simplifier.extend_required_conditions(required_conditions.clone());
            let shortcut_steps = if collect_steps {
                let mut step = build_root_shortcut_compact_step(
                    expr,
                    compact,
                    "Compact quotient with matching reciprocal trig antiderivative residual",
                    "Reciprocal Trig Integral Residual Quotient",
                );
                step.meta_mut().required_conditions = required_conditions;
                vec![step]
            } else {
                Vec::new()
            };
            return (compact, shortcut_steps, crate::phase::PipelineStats::default());
        }

        if let Some((one, required_conditions)) = crate::calculus_residual_support::
            try_diff_integral_residual_passthrough_quotient_root_one(
                &mut simplifier.context,
                expr,
            )
        {
            simplifier.extend_required_conditions(required_conditions.clone());
            let shortcut_steps = if collect_steps {
                let mut step = build_root_shortcut_compact_step(
                    expr,
                    one,
                    "Compact quotient with matching antiderivative residual",
                    "Integral Residual Quotient",
                );
                step.meta_mut().required_conditions = required_conditions;
                vec![step]
            } else {
                Vec::new()
            };
            return (one, shortcut_steps, crate::phase::PipelineStats::default());
        }

        if let Some((zero, required_conditions)) =
            crate::calculus_residual_support::try_diff_integral_plain_trig_residual_root_zero(
                &mut simplifier.context,
                expr,
            )
        {
            simplifier.extend_required_conditions(required_conditions.clone());
            let shortcut_steps = if collect_steps {
                let mut step = build_root_shortcut_compact_step(
                    expr,
                    zero,
                    "Verify matching plain trig antiderivative residual",
                    "Plain Trig Integral Residual",
                );
                step.meta_mut().required_conditions = required_conditions;
                vec![step]
            } else {
                Vec::new()
            };
            return (zero, shortcut_steps, crate::phase::PipelineStats::default());
        }

        if let Some((compact, required_conditions)) = crate::calculus_residual_support::
            try_diff_integral_plain_trig_residual_constant_passthrough_quotient(
                &mut simplifier.context,
                expr,
            )
        {
            simplifier.extend_required_conditions(required_conditions.clone());
            let shortcut_steps = if collect_steps {
                let mut step = build_root_shortcut_compact_step(
                    expr,
                    compact,
                    "Compact quotient with matching plain trig antiderivative residual",
                    "Plain Trig Integral Residual Quotient",
                );
                step.meta_mut().required_conditions = required_conditions;
                vec![step]
            } else {
                Vec::new()
            };
            return (compact, shortcut_steps, crate::phase::PipelineStats::default());
        }

        if let Some((zero, required_conditions)) =
            crate::calculus_residual_support::try_diff_integral_inverse_trig_residual_root_zero(
                &mut simplifier.context,
                expr,
            )
        {
            simplifier.extend_required_conditions(required_conditions.clone());
            let shortcut_steps = if collect_steps {
                let mut step = build_root_shortcut_compact_step(
                    expr,
                    zero,
                    "Verify matching inverse trig antiderivative residual",
                    "Inverse Trig Integral Residual",
                );
                step.meta_mut().required_conditions = required_conditions;
                vec![step]
            } else {
                Vec::new()
            };
            return (zero, shortcut_steps, crate::phase::PipelineStats::default());
        }

        if let Some((compact, required_conditions)) = crate::calculus_residual_support::
            try_diff_integral_inverse_trig_residual_constant_passthrough_quotient(
                &mut simplifier.context,
                expr,
            )
        {
            simplifier.extend_required_conditions(required_conditions.clone());
            let shortcut_steps = if collect_steps {
                let mut step = build_root_shortcut_compact_step(
                    expr,
                    compact,
                    "Compact quotient with matching inverse trig antiderivative residual",
                    "Inverse Trig Integral Residual Quotient",
                );
                step.meta_mut().required_conditions = required_conditions;
                vec![step]
            } else {
                Vec::new()
            };
            return (compact, shortcut_steps, crate::phase::PipelineStats::default());
        }

        if let Some((zero, required_conditions)) =
            crate::calculus_residual_support::try_explicit_log_abs_antiderivative_residual_root_zero(
                &mut simplifier.context,
                expr,
            )
        {
            simplifier.extend_required_conditions(required_conditions.clone());
            let shortcut_steps = if collect_steps {
                let mut step = build_root_shortcut_compact_step(
                    expr,
                    zero,
                    "Verify matching log-absolute antiderivative residual",
                    "Log-Absolute Diff Residual",
                );
                step.meta_mut().required_conditions = required_conditions;
                vec![step]
            } else {
                Vec::new()
            };
            return (zero, shortcut_steps, crate::phase::PipelineStats::default());
        }

        if let Some((zero, required_conditions)) =
            crate::calculus_residual_support::try_diff_integral_quadratic_exp_residual_root_zero(
                &mut simplifier.context,
                expr,
            )
        {
            simplifier.extend_required_conditions(required_conditions.clone());
            let shortcut_steps = if collect_steps {
                let mut step = build_root_shortcut_compact_step(
                    expr,
                    zero,
                    "Verify matching quadratic exponential antiderivative residual",
                    "Quadratic Exp Integral Residual",
                );
                step.meta_mut().required_conditions = required_conditions;
                vec![step]
            } else {
                Vec::new()
            };
            return (zero, shortcut_steps, crate::phase::PipelineStats::default());
        }

        if let Some((zero, required_conditions)) =
            crate::calculus_residual_support::try_diff_integral_hyperbolic_residual_root_zero(
                &mut simplifier.context,
                expr,
            )
        {
            simplifier.extend_required_conditions(required_conditions.clone());
            let shortcut_steps = if collect_steps {
                let mut step = build_root_shortcut_compact_step(
                    expr,
                    zero,
                    "Verify matching hyperbolic antiderivative residual",
                    "Hyperbolic Integral Residual",
                );
                step.meta_mut().required_conditions = required_conditions;
                vec![step]
            } else {
                Vec::new()
            };
            return (zero, shortcut_steps, crate::phase::PipelineStats::default());
        }

        if let Some((compact, required_conditions)) = crate::calculus_residual_support::
            try_diff_integral_hyperbolic_residual_constant_passthrough_quotient(
                &mut simplifier.context,
                expr,
            )
        {
            simplifier.extend_required_conditions(required_conditions.clone());
            let shortcut_steps = if collect_steps {
                let mut step = build_root_shortcut_compact_step(
                    expr,
                    compact,
                    "Compact quotient with matching hyperbolic antiderivative residual",
                    "Hyperbolic Integral Residual Quotient",
                );
                step.meta_mut().required_conditions = required_conditions;
                vec![step]
            } else {
                Vec::new()
            };
            return (compact, shortcut_steps, crate::phase::PipelineStats::default());
        }

        if let Some((zero, required_conditions)) = crate::calculus_residual_support::
            try_diff_integral_residual_reciprocal_shifted_difference_root_zero(
                &mut simplifier.context,
                expr,
            )
        {
            simplifier.extend_required_conditions(required_conditions.clone());
            let shortcut_steps = if collect_steps {
                let mut step = build_root_shortcut_compact_step(
                    expr,
                    zero,
                    "Verify reciprocal wrapper with matching antiderivative residual",
                    "Integral Residual Reciprocal Wrapper",
                );
                step.meta_mut().required_conditions = required_conditions;
                vec![step]
            } else {
                Vec::new()
            };
            return (zero, shortcut_steps, crate::phase::PipelineStats::default());
        }

        if let Some((zero, required_conditions)) = crate::calculus_residual_support::
            try_diff_integral_residual_shifted_quotient_difference_root_zero(
                &mut simplifier.context,
                expr,
            )
        {
            simplifier.extend_required_conditions(required_conditions.clone());
            let shortcut_steps = if collect_steps {
                let mut step = build_root_shortcut_compact_step(
                    expr,
                    zero,
                    "Verify shifted quotient with matching antiderivative residual",
                    "Integral Residual Shifted Quotient",
                );
                step.meta_mut().required_conditions = required_conditions;
                vec![step]
            } else {
                Vec::new()
            };
            return (zero, shortcut_steps, crate::phase::PipelineStats::default());
        }

        if let Some((zero, required_conditions)) = crate::calculus_residual_support::
            try_diff_integral_residual_product_zero_factor_root_zero(
                &mut simplifier.context,
                expr,
            )
        {
            simplifier.extend_required_conditions(required_conditions.clone());
            let shortcut_steps = if collect_steps {
                let mut step = build_root_shortcut_compact_step(
                    expr,
                    zero,
                    "Collapse product with zero factor around matching antiderivative residual",
                    "Integral Residual Product Zero Wrapper",
                );
                step.meta_mut().required_conditions = required_conditions;
                vec![step]
            } else {
                Vec::new()
            };
            return (zero, shortcut_steps, crate::phase::PipelineStats::default());
        }

        if let Some((zero, required_conditions)) =
            crate::calculus_residual_support::try_integral_hyperbolic_reciprocal_shifted_difference_root_zero(
                &mut simplifier.context,
                expr,
            )
        {
            simplifier.extend_required_conditions(required_conditions.clone());
            let shortcut_steps = if collect_steps {
                let mut step = build_root_shortcut_compact_step(
                    expr,
                    zero,
                    "Verify reciprocal wrapper with matching hyperbolic antiderivative",
                    "Hyperbolic Integral Reciprocal Wrapper",
                );
                step.meta_mut().required_conditions = required_conditions;
                vec![step]
            } else {
                Vec::new()
            };
            return (zero, shortcut_steps, crate::phase::PipelineStats::default());
        }

        if let Some((zero, required_conditions)) = crate::calculus_residual_support::
            try_diff_integral_rational_quadratic_residual_root_zero(&mut simplifier.context, expr)
        {
            simplifier.extend_required_conditions(required_conditions.clone());
            let shortcut_steps = if collect_steps {
                let mut step = build_root_shortcut_compact_step(
                    expr,
                    zero,
                    "Verify matching rational quadratic antiderivative residual",
                    "Rational Quadratic Integral Residual",
                );
                step.meta_mut().required_conditions = required_conditions;
                vec![step]
            } else {
                Vec::new()
            };
            return (zero, shortcut_steps, crate::phase::PipelineStats::default());
        }

        if let Some((zero, required_conditions)) = crate::calculus_residual_support::
            try_diff_integral_rational_quadratic_residual_reciprocal_shifted_difference_root_zero(
                &mut simplifier.context,
                expr,
            )
        {
            simplifier.extend_required_conditions(required_conditions.clone());
            let shortcut_steps = if collect_steps {
                let mut step = build_root_shortcut_compact_step(
                    expr,
                    zero,
                    "Verify reciprocal wrapper with matching rational antiderivative residual",
                    "Rational Integral Residual Reciprocal Wrapper",
                );
                step.meta_mut().required_conditions = required_conditions;
                vec![step]
            } else {
                Vec::new()
            };
            return (zero, shortcut_steps, crate::phase::PipelineStats::default());
        }

        if let Some((compact, required_conditions)) = crate::calculus_residual_support::
            try_diff_integral_rational_quadratic_residual_constant_passthrough_quotient(
                &mut simplifier.context,
                expr,
            )
        {
            simplifier.extend_required_conditions(required_conditions.clone());
            let shortcut_steps = if collect_steps {
                let mut step = build_root_shortcut_compact_step(
                    expr,
                    compact,
                    "Compact quotient with matching rational quadratic antiderivative residual",
                    "Rational Quadratic Integral Residual Quotient",
                );
                step.meta_mut().required_conditions = required_conditions;
                vec![step]
            } else {
                Vec::new()
            };
            return (compact, shortcut_steps, crate::phase::PipelineStats::default());
        }

        if let Some((zero, required_conditions)) = crate::calculus_residual_support::
            try_explicit_positive_quadratic_cube_antiderivative_residual_root_zero(
                &mut simplifier.context,
                expr,
            )
        {
            simplifier.extend_required_conditions(required_conditions.clone());
            let shortcut_steps = if collect_steps {
                let mut step = build_root_shortcut_compact_step(
                    expr,
                    zero,
                    "Verify matching positive quadratic cube antiderivative residual",
                    "Positive Quadratic Cube Integral Residual",
                );
                step.meta_mut().required_conditions = required_conditions;
                vec![step]
            } else {
                Vec::new()
            };
            return (zero, shortcut_steps, crate::phase::PipelineStats::default());
        }

        if let Some((compact, required_conditions)) = crate::calculus_residual_support::
            try_explicit_positive_quadratic_cube_antiderivative_residual_constant_passthrough_quotient(
                &mut simplifier.context,
                expr,
            )
        {
            simplifier.extend_required_conditions(required_conditions.clone());
            let shortcut_steps = if collect_steps {
                let mut step = build_root_shortcut_compact_step(
                    expr,
                    compact,
                    "Compact quotient with matching positive quadratic cube antiderivative residual",
                    "Positive Quadratic Cube Integral Residual Quotient",
                );
                step.meta_mut().required_conditions = required_conditions;
                vec![step]
            } else {
                Vec::new()
            };
            return (compact, shortcut_steps, crate::phase::PipelineStats::default());
        }

        if let Some((zero, required_conditions)) = crate::calculus_residual_support::
            try_explicit_positive_quadratic_square_antiderivative_residual_root_zero(
                &mut simplifier.context,
                expr,
            )
        {
            simplifier.extend_required_conditions(required_conditions.clone());
            let shortcut_steps = if collect_steps {
                let mut step = build_root_shortcut_compact_step(
                    expr,
                    zero,
                    "Verify matching positive quadratic square antiderivative residual",
                    "Positive Quadratic Square Integral Residual",
                );
                step.meta_mut().required_conditions = required_conditions;
                vec![step]
            } else {
                Vec::new()
            };
            return (zero, shortcut_steps, crate::phase::PipelineStats::default());
        }

        if let Some((compact, required_conditions)) = crate::calculus_residual_support::
            try_explicit_positive_quadratic_square_antiderivative_residual_constant_passthrough_quotient(
                &mut simplifier.context,
                expr,
            )
        {
            simplifier.extend_required_conditions(required_conditions.clone());
            let shortcut_steps = if collect_steps {
                let mut step = build_root_shortcut_compact_step(
                    expr,
                    compact,
                    "Compact quotient with matching positive quadratic square antiderivative residual",
                    "Positive Quadratic Square Integral Residual Quotient",
                );
                step.meta_mut().required_conditions = required_conditions;
                vec![step]
            } else {
                Vec::new()
            };
            return (compact, shortcut_steps, crate::phase::PipelineStats::default());
        }

        if let Some((zero, required_conditions)) = crate::calculus_residual_support::
            try_explicit_high_log_power_product_antiderivative_residual_root_zero(
                &mut simplifier.context,
                expr,
            )
        {
            simplifier.extend_required_conditions(required_conditions.clone());
            let shortcut_steps = if collect_steps {
                let mut step = build_root_shortcut_compact_step(
                    expr,
                    zero,
                    "Verify matching high log-power product antiderivative residual",
                    "High Log-Power Product Integral Residual",
                );
                step.meta_mut().required_conditions = required_conditions;
                vec![step]
            } else {
                Vec::new()
            };
            return (zero, shortcut_steps, crate::phase::PipelineStats::default());
        }

        if let Some((compact, required_conditions)) = crate::calculus_residual_support::
            try_explicit_high_log_power_product_antiderivative_residual_constant_passthrough_quotient(
                &mut simplifier.context,
                expr,
            )
        {
            simplifier.extend_required_conditions(required_conditions.clone());
            let shortcut_steps = if collect_steps {
                let mut step = build_root_shortcut_compact_step(
                    expr,
                    compact,
                    "Compact quotient with matching high log-power product antiderivative residual",
                    "High Log-Power Product Integral Residual Quotient",
                );
                step.meta_mut().required_conditions = required_conditions;
                vec![step]
            } else {
                Vec::new()
            };
            return (compact, shortcut_steps, crate::phase::PipelineStats::default());
        }

        if let Some((zero, required_conditions)) = crate::calculus_residual_support::
            try_explicit_quadratic_affine_log_antiderivative_residual_root_zero(
                &mut simplifier.context,
                expr,
            )
        {
            simplifier.extend_required_conditions(required_conditions.clone());
            let shortcut_steps = if collect_steps {
                let mut step = build_root_shortcut_compact_step(
                    expr,
                    zero,
                    "Verify matching quadratic affine-log antiderivative residual",
                    "Quadratic Affine-Log Integral Residual",
                );
                step.meta_mut().required_conditions = required_conditions;
                vec![step]
            } else {
                Vec::new()
            };
            return (zero, shortcut_steps, crate::phase::PipelineStats::default());
        }

        if let Some((zero, required_conditions)) = crate::calculus_residual_support::
            try_explicit_quadratic_positive_quadratic_log_antiderivative_residual_root_zero(
                &mut simplifier.context,
                expr,
            )
        {
            simplifier.extend_required_conditions(required_conditions.clone());
            let shortcut_steps = if collect_steps {
                let mut step = build_root_shortcut_compact_step(
                    expr,
                    zero,
                    "Verify matching quadratic positive-quadratic log antiderivative residual",
                    "Quadratic Positive-Quadratic Log Integral Residual",
                );
                step.meta_mut().required_conditions = required_conditions;
                vec![step]
            } else {
                Vec::new()
            };
            return (zero, shortcut_steps, crate::phase::PipelineStats::default());
        }

        if let Some((one, required_conditions)) =
            crate::calculus_residual_support::try_diff_arctan_sqrt_positive_polynomial_quotient_shifted_one_root(
                &mut simplifier.context,
                expr,
            )
        {
            simplifier.extend_required_conditions(required_conditions.clone());
            let shortcut_steps = if collect_steps {
                let mut step = build_root_shortcut_compact_step(
                    expr,
                    one,
                    "Collapse shifted quotient of matching derivative presentation",
                    "Arctan Sqrt Diff Shifted Quotient",
                );
                step.meta_mut().required_conditions = required_conditions;
                vec![step]
            } else {
                Vec::new()
            };
            return (one, shortcut_steps, crate::phase::PipelineStats::default());
        }

        if let Some((compact, required_conditions)) =
            crate::calculus_residual_support::try_diff_arctan_sqrt_positive_polynomial_quotient_shifted_compact_mismatch(
                &mut simplifier.context,
                expr,
            )
        {
            simplifier.extend_required_conditions(required_conditions.clone());
            let shortcut_steps = if collect_steps {
                let mut step = build_root_shortcut_compact_step(
                    expr,
                    compact,
                    "Compact nonmatching shifted quotient of arctan sqrt derivative",
                    "Arctan Sqrt Diff Shifted Quotient Mismatch",
                );
                step.meta_mut().required_conditions = required_conditions;
                vec![step]
            } else {
                Vec::new()
            };
            return (compact, shortcut_steps, crate::phase::PipelineStats::default());
        }

        if let Some((zero, required_conditions)) =
            crate::calculus_residual_support::try_diff_sqrt_log_plus_constant_residual_root_zero(
                &mut simplifier.context,
                expr,
            )
        {
            simplifier.extend_required_conditions(required_conditions.clone());
            let shortcut_steps = if collect_steps {
                let mut step = build_root_shortcut_compact_step(
                    expr,
                    zero,
                    "Cancel matching sqrt-log derivative residual",
                    "Sqrt Log Diff Residual",
                );
                step.meta_mut().required_conditions = required_conditions;
                vec![step]
            } else {
                Vec::new()
            };
            return (zero, shortcut_steps, crate::phase::PipelineStats::default());
        }

        if let Some((zero, required_conditions)) = crate::calculus_residual_support::
            try_reciprocal_half_power_shared_denominator_residual_root_zero(
                &mut simplifier.context,
                expr,
            )
        {
            simplifier.extend_required_conditions(required_conditions.clone());
            let shortcut_steps = if collect_steps {
                let mut step = build_root_shortcut_compact_step(
                    expr,
                    zero,
                    "Cancel matching reciprocal half-power residual",
                    "Reciprocal Half-Power Residual",
                );
                step.meta_mut().required_conditions = required_conditions;
                vec![step]
            } else {
                Vec::new()
            };
            return (zero, shortcut_steps, crate::phase::PipelineStats::default());
        }

        // Narrow hidden solve root shortcuts. Keep them limited to the
        // no-steps, no-listener solve path and dispatch by root kind so we do
        // not pay unrelated matchers on every expression.
        if self.options.shared.context_mode == crate::options::ContextMode::Standard
            && self.options.shared.semantics.value_domain == crate::semantics::ValueDomain::RealOnly
            && !simplifier.has_step_listener()
            && is_real_domain_complex_noop_root(&simplifier.context, expr)
        {
            return (expr, Vec::new(), crate::phase::PipelineStats::default());
        }

        if let Some(result) =
            try_finish_dirichlet_kernel_root_shortcut(simplifier, expr, collect_steps)
        {
            return result;
        }

        // Matrix multiplication is non-commutative, but the root-shortcut layer
        // below is a scalar/trig/polynomial fast path: several of its exact-zero
        // and equivalent-pair matchers compare products as commutative factor
        // multisets, which collapses the commutator `A·B − B·A` to 0 even though
        // it is generally nonzero. Skip the entire shortcut layer whenever a
        // matrix participates as a product factor and let the normal pipeline
        // evaluate the expression to its true matrix value.
        let root_shortcut_matrix_guard =
            crate::rules::arithmetic::term_has_matrix_product_factor(&simplifier.context, expr);

        macro_rules! return_profiled_root_shortcut {
            ($name:literal, $call:expr) => {
                if self.time_budget_exceeded() {
                    return self.finish_timed_out_pipeline(
                        simplifier,
                        expr,
                        Vec::new(),
                        crate::phase::PipelineStats::default(),
                        None,
                    );
                }
                if let Some((result, shortcut_steps)) = run_profiled_root_shortcut($name, || $call)
                {
                    // SOUNDNESS: skip a shortcut whose result is unsound (a non-zero collapse to 0, or
                    // an `∞/∞` cancelled to a finite value) so the honest rule pipeline runs.
                    if !root_shortcut_result_is_unsound(&mut simplifier.context, expr, result) {
                        if self.time_budget_exceeded() {
                            return self.finish_timed_out_pipeline(
                                simplifier,
                                result,
                                shortcut_steps,
                                crate::phase::PipelineStats::default(),
                                None,
                            );
                        }
                        return (
                            result,
                            shortcut_steps,
                            crate::phase::PipelineStats::default(),
                        );
                    }
                }
                if self.time_budget_exceeded() {
                    return self.finish_timed_out_pipeline(
                        simplifier,
                        expr,
                        Vec::new(),
                        crate::phase::PipelineStats::default(),
                        None,
                    );
                }
            };
        }

        macro_rules! return_root_shortcut_pair {
            ($call:expr) => {
                if self.time_budget_exceeded() {
                    return self.finish_timed_out_pipeline(
                        simplifier,
                        expr,
                        Vec::new(),
                        crate::phase::PipelineStats::default(),
                        None,
                    );
                }
                if let Some((result, shortcut_steps)) = $call {
                    // SOUNDNESS: skip a shortcut whose result is unsound (a non-zero collapse to 0, or
                    // an `∞/∞` cancelled to a finite value) so the honest rule pipeline runs.
                    if !root_shortcut_result_is_unsound(&mut simplifier.context, expr, result) {
                        if self.time_budget_exceeded() {
                            return self.finish_timed_out_pipeline(
                                simplifier,
                                result,
                                shortcut_steps,
                                crate::phase::PipelineStats::default(),
                                None,
                            );
                        }
                        return (
                            result,
                            shortcut_steps,
                            crate::phase::PipelineStats::default(),
                        );
                    }
                }
                if self.time_budget_exceeded() {
                    return self.finish_timed_out_pipeline(
                        simplifier,
                        expr,
                        Vec::new(),
                        crate::phase::PipelineStats::default(),
                        None,
                    );
                }
            };
        }

        if !root_shortcut_matrix_guard
            && matches!(
                self.options.shared.context_mode,
                crate::options::ContextMode::Standard | crate::options::ContextMode::Auto
            )
        {
            let add_root = matches!(simplifier.context.get(expr), Expr::Add(_, _));
            let sub_root = matches!(simplifier.context.get(expr), Expr::Sub(_, _));
            let div_root = matches!(simplifier.context.get(expr), Expr::Div(_, _));
            let mul_root = matches!(simplifier.context.get(expr), Expr::Mul(_, _));
            let add_term_count = if add_root || sub_root {
                AddView::from_expr(&simplifier.context, expr).terms.len()
            } else {
                0
            };
            let raw_binary_pythagorean_identity = if add_root || sub_root {
                crate::rules::arithmetic::extract_two_term_core_difference(
                    &mut simplifier.context,
                    expr,
                )
                .is_some_and(|(lhs_core, rhs_core)| {
                    matches_direct_pythagorean_identity_pair_root(
                        &mut simplifier.context,
                        lhs_core,
                        rhs_core,
                    )
                })
            } else {
                false
            };

            // These exact-equivalence shortcuts emit proper didactic steps, so keep
            // them available even when the caller requested step collection.
            if mul_root {
                return_profiled_root_shortcut!(
                    "root.mul.01.direct_scaled_half_angle_square.early",
                    try_standard_direct_scaled_half_angle_square_shortcut(
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                );
                return_profiled_root_shortcut!(
                    "root.mul.02.collapsed_fraction_direct_pair_factor",
                    try_standard_collapsed_fraction_direct_pair_factor_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                );
                return_profiled_root_shortcut!(
                    "root.mul.03.collapsed_fraction_factored_numerator",
                    try_standard_collapsed_fraction_factored_numerator_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                );
                return_profiled_root_shortcut!(
                    "root.mul.04.collapsed_fraction_partner_canonicalization",
                    try_standard_collapsed_fraction_partner_canonicalization_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                );
                return_profiled_root_shortcut!(
                    "root.mul.05.embedded_trig_product_to_sum.early",
                    try_standard_embedded_trig_product_to_sum_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                );
                return_profiled_root_shortcut!(
                    "root.mul.29.zero_product_with_exact_zero_child",
                    try_standard_zero_product_with_exact_zero_child_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                );
                return_profiled_root_shortcut!(
                    "root.mul.25.direct_small_zero_pair",
                    try_standard_direct_small_zero_pair_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                );
                return_profiled_root_shortcut!(
                    "root.mul.06.collapsed_fraction_hyperbolic_half_angle_factor",
                    try_standard_collapsed_fraction_hyperbolic_half_angle_factor_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                );
                return_profiled_root_shortcut!(
                    "root.mul.07.tangent_addition_fraction_product",
                    try_standard_tangent_addition_fraction_product_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                );
                return_profiled_root_shortcut!(
                    "root.mul.08.trig_product_to_sum_subset_factor",
                    try_standard_trig_product_to_sum_subset_factor_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                );
                return_profiled_root_shortcut!(
                    "root.mul.09.perfect_square_trinomial_factor",
                    try_standard_perfect_square_trinomial_factor_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                );
                return_profiled_root_shortcut!(
                    "root.mul.10.sum_of_squares_product_subset_factor",
                    try_standard_sum_of_squares_product_subset_factor_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                );
                return_profiled_root_shortcut!(
                    "root.mul.11.square_anchor_linear_shift_partner",
                    try_standard_square_anchor_linear_shift_partner_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                );
                return_profiled_root_shortcut!(
                    "root.mul.12.three_linear_shift_anchor_direct_partner",
                    try_standard_three_linear_shift_anchor_direct_partner_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                );
                return_profiled_root_shortcut!(
                    "root.mul.13.two_factor_small_partner_canonicalization",
                    try_standard_two_factor_small_partner_canonicalization_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                );
                return_profiled_root_shortcut!(
                    "root.mul.14.two_factor_direct_pair_anchor",
                    try_standard_two_factor_direct_pair_anchor_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                );
                return_profiled_root_shortcut!(
                    "root.mul.15.inverse_trig_anchor_small_polynomial_partner",
                    try_standard_inverse_trig_anchor_small_polynomial_partner_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                );
                return_profiled_root_shortcut!(
                    "root.mul.16.safe_anchor_small_polynomial_partner",
                    try_standard_safe_anchor_small_polynomial_partner_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                );
                return_profiled_root_shortcut!(
                    "root.mul.17.scaled_half_angle_anchor_direct_partner",
                    try_standard_scaled_half_angle_anchor_direct_partner_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                );
                return_profiled_root_shortcut!(
                    "root.mul.18.safe_anchor_direct_partner",
                    try_standard_safe_anchor_direct_partner_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                );
                return_profiled_root_shortcut!(
                    "root.mul.19.special_angle_exact_value_factor",
                    try_standard_special_angle_exact_value_factor_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                );
                return_profiled_root_shortcut!(
                    "root.mul.20.tangent_addition_factor",
                    try_standard_tangent_addition_factor_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                );
                return_profiled_root_shortcut!(
                    "root.mul.21.direct_scaled_half_angle_square.late",
                    try_standard_direct_scaled_half_angle_square_shortcut(
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                );
                return_profiled_root_shortcut!(
                    "root.mul.22.half_angle_square_factor",
                    try_standard_half_angle_square_factor_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                );
                return_profiled_root_shortcut!(
                    "root.mul.23.trig_power_reduction_factor",
                    try_standard_trig_power_reduction_factor_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                );
                return_profiled_root_shortcut!(
                    "root.mul.24.positive_double_cos_square_diff_factor",
                    try_standard_positive_double_cos_square_diff_factor_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                );
                return_profiled_root_shortcut!(
                    "root.mul.26.reciprocal_trig_zero_pair",
                    try_standard_reciprocal_trig_zero_pair_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                );
                return_profiled_root_shortcut!(
                    "root.mul.27.small_trig_zero_pair",
                    try_standard_small_trig_zero_pair_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                );
                return_profiled_root_shortcut!(
                    "root.mul.28.direct_trig_mixed_zero_pair",
                    try_standard_direct_trig_mixed_zero_pair_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                );
                return_profiled_root_shortcut!(
                    "root.mul.30.embedded_trig_product_to_sum.late",
                    try_standard_embedded_trig_product_to_sum_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                );
                return_profiled_root_shortcut!(
                    "root.mul.31.assumed_dyadic_cos_product",
                    try_standard_assumed_dyadic_cos_product_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                );
            }
            // A difference whose terms carry a literal non-finite or undefined
            // value never collapses to zero: `inf - inf`, `(1/0) - (1/0)` and
            // `undefined - undefined` are indeterminate, not `0`. Skip the whole
            // additive exact-zero / common-scale shortcut family so these stay
            // symbolic instead of being folded by one of its many routes.
            if (add_root || sub_root)
                && !crate::rules::arithmetic::additive_term_is_nonfinite_or_undefined(
                    &simplifier.context,
                    expr,
                )
            {
                if collect_steps
                    && simplifier.get_steps_mode() == crate::options::StepsMode::Compact
                {
                    return_profiled_root_shortcut!(
                        "root.addsub.00.tan_cot_half_angle_pair.compact_first",
                        try_standard_compact_tan_cot_half_angle_zero_pair_shortcut(
                            &self.options,
                            &mut simplifier.context,
                            expr,
                            collect_steps,
                        )
                    );
                    record_orchestrator_shortcut_profile_sample(
                        &simplifier.context,
                        expr,
                        "root.addsub.00.direct_small_zero_pair.compact_first",
                    );
                    return_profiled_root_shortcut!(
                        "root.addsub.00.direct_small_zero_pair.compact_first",
                        try_standard_direct_small_zero_pair_shortcut(
                            &self.options,
                            &mut simplifier.context,
                            expr,
                            collect_steps,
                        )
                    );
                }
                return_root_shortcut_pair!(
                    try_standard_nested_fraction_zero_hyperbolic_identity_pair_shortcut(
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                );
                return_root_shortcut_pair!(
                    try_standard_log_zero_hyperbolic_cosh_cubic_pair_shortcut(
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                );
                return_root_shortcut_pair!(
                    try_standard_targeted_direct_small_zero_additive_combination_shortcut(
                        simplifier,
                        expr,
                        collect_steps,
                    )
                );
                record_orchestrator_shortcut_profile_sample(
                    &simplifier.context,
                    expr,
                    "root.addsub.00.direct_small_zero_pair",
                );
                return_profiled_root_shortcut!(
                    "root.addsub.00.direct_small_zero_pair",
                    try_standard_direct_small_zero_pair_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                );
                return_profiled_root_shortcut!(
                    "root.addsub.00.same_denominator_distribution_pair",
                    try_standard_same_denominator_distribution_pair_zero_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                );
                return_profiled_root_shortcut!(
                    "root.addsub.00.direct_cos_square_diff_zero",
                    try_standard_direct_cos_square_diff_zero_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                );
                return_root_shortcut_pair!(try_standard_direct_pythagorean_extended_zero_shortcut(
                    &mut simplifier.context,
                    expr,
                    collect_steps,
                ));
                return_profiled_root_shortcut!(
                    "root.addsub.00.atanh_square_ratio_log_zero",
                    try_standard_atanh_square_ratio_log_zero_shortcut(
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                );
                return_root_shortcut_pair!(try_standard_shared_passthrough_direct_pair_shortcut(
                    &self.options,
                    &mut simplifier.context,
                    expr,
                    collect_steps,
                ));
                return_root_shortcut_pair!(try_standard_small_composed_additive_pair_shortcut(
                    &self.options,
                    &mut simplifier.context,
                    expr,
                    collect_steps,
                ));
                return_root_shortcut_pair!(
                    try_standard_subtract_expanded_sum_diff_cubes_quotient_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                );
                if !collect_steps {
                    if let Expr::Add(lhs, rhs) | Expr::Sub(lhs, rhs) = simplifier.context.get(expr)
                    {
                        let lhs = *lhs;
                        let rhs = *rhs;
                        if matches!(
                            simplifier.context.get(lhs),
                            Expr::Add(_, _) | Expr::Sub(_, _)
                        ) && matches!(
                            simplifier.context.get(rhs),
                            Expr::Add(_, _) | Expr::Sub(_, _)
                        ) && expr_contains_trig_or_hyperbolic_builtin_local(
                            &simplifier.context,
                            lhs,
                        ) && expr_contains_trig_or_hyperbolic_builtin_local(
                            &simplifier.context,
                            rhs,
                        ) && is_supported_small_trig_zero_pair_side_root(
                            &simplifier.context,
                            lhs,
                            true,
                        ) && is_supported_small_trig_zero_pair_side_root(
                            &simplifier.context,
                            rhs,
                            true,
                        ) {
                            return_root_shortcut_pair!(try_standard_small_trig_zero_pair_shortcut(
                                &self.options,
                                &mut simplifier.context,
                                expr,
                                collect_steps,
                            ));
                        }
                        return_root_shortcut_pair!(try_standard_direct_small_trig_zero_child_with_supported_zero_partner_shortcut(
                            &self.options,
                            &mut simplifier.context,
                            expr,
                            collect_steps,
                        ));
                    }
                    if add_term_count >= 5 {
                        return_profiled_root_shortcut!(
                            "root.addsub.01.multiterm_trig_numeric_subset_zero",
                            try_standard_multiterm_trig_numeric_subset_zero_shortcut(
                                &self.options,
                                &mut simplifier.context,
                                expr,
                                collect_steps,
                            )
                        );
                    }
                    return_profiled_root_shortcut!(
                        "root.addsub.01.symbolic_root_denesting_subset_zero",
                        try_standard_symbolic_root_denesting_subset_zero_shortcut(
                            &self.options,
                            &mut simplifier.context,
                            expr,
                            collect_steps,
                        )
                    );
                    return_profiled_root_shortcut!(
                        "root.addsub.01.sqrt_perfect_square_abs_subset_zero",
                        try_standard_sqrt_perfect_square_abs_subset_zero_shortcut(
                            &self.options,
                            &mut simplifier.context,
                            expr,
                            collect_steps,
                        )
                    );
                    return_profiled_root_shortcut!(
                        "root.addsub.01.inverse_trig_plus_sqrt_subset_zero",
                        try_standard_inverse_trig_plus_sqrt_subset_zero_shortcut(
                            &self.options,
                            &mut simplifier.context,
                            expr,
                            collect_steps,
                        )
                    );
                    return_profiled_root_shortcut!(
                        "root.addsub.01.inverse_trig_composition_subset_zero",
                        try_standard_inverse_trig_composition_subset_zero_shortcut(
                            &self.options,
                            &mut simplifier.context,
                            expr,
                            collect_steps,
                        )
                    );
                    return_profiled_root_shortcut!(
                        "root.addsub.01.atanh_square_ratio_log_subset_zero",
                        try_standard_atanh_square_ratio_log_subset_zero_shortcut(
                            &self.options,
                            &mut simplifier.context,
                            expr,
                            collect_steps,
                        )
                    );
                    if add_term_count > 2
                        && !raw_binary_pythagorean_identity
                        && has_structural_numeric_pythagorean_pair(&simplifier.context, expr)
                    {
                        return_root_shortcut_pair!(
                            try_standard_pythagorean_additive_pipeline_shortcut(
                                &self.options,
                                &mut simplifier.context,
                                expr,
                                collect_steps,
                            )
                        );
                    }
                    return_root_shortcut_pair!(try_standard_nested_exact_zero_child_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    ));
                }
                return_root_shortcut_pair!(try_standard_small_trig_zero_pair_shortcut(
                    &self.options,
                    &mut simplifier.context,
                    expr,
                    collect_steps,
                ));
                if add_term_count >= 5 {
                    return_profiled_root_shortcut!(
                        "root.addsub.01.multiterm_trig_numeric_subset_zero",
                        try_standard_multiterm_trig_numeric_subset_zero_shortcut(
                            &self.options,
                            &mut simplifier.context,
                            expr,
                            collect_steps,
                        )
                    );
                }
                return_profiled_root_shortcut!(
                    "root.addsub.01.symbolic_root_denesting_subset_zero",
                    try_standard_symbolic_root_denesting_subset_zero_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                );
                return_profiled_root_shortcut!(
                    "root.addsub.01.sqrt_perfect_square_abs_subset_zero",
                    try_standard_sqrt_perfect_square_abs_subset_zero_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                );
                return_profiled_root_shortcut!(
                    "root.addsub.01.inverse_trig_plus_sqrt_subset_zero",
                    try_standard_inverse_trig_plus_sqrt_subset_zero_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                );
                return_profiled_root_shortcut!(
                    "root.addsub.01.inverse_trig_composition_subset_zero",
                    try_standard_inverse_trig_composition_subset_zero_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                );
                return_profiled_root_shortcut!(
                    "root.addsub.01.atanh_square_ratio_log_subset_zero",
                    try_standard_atanh_square_ratio_log_subset_zero_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                );
                return_root_shortcut_pair!(
                    try_standard_rational_half_angle_target_passthrough_shortcut(
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                );
                return_root_shortcut_pair!(
                    try_standard_scaled_sin_fourth_power_reduction_zero_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                );
                return_root_shortcut_pair!(try_standard_trig_fourth_power_difference_shortcut(
                    &mut simplifier.context,
                    expr,
                    collect_steps,
                ));
                return_root_shortcut_pair!(try_standard_reciprocal_pythagorean_pair_shortcut(
                    &mut simplifier.context,
                    expr,
                    collect_steps,
                ));
                return_root_shortcut_pair!(try_standard_shared_passthrough_direct_pair_shortcut(
                    &self.options,
                    &mut simplifier.context,
                    expr,
                    collect_steps,
                ));
                return_root_shortcut_pair!(try_standard_two_factor_product_pair_zero_shortcut(
                    &self.options,
                    &mut simplifier.context,
                    expr,
                    collect_steps,
                ));
                if add_term_count > 2
                    && !raw_binary_pythagorean_identity
                    && has_structural_numeric_pythagorean_pair(&simplifier.context, expr)
                {
                    return_root_shortcut_pair!(
                        try_standard_pythagorean_additive_pipeline_shortcut(
                            &self.options,
                            &mut simplifier.context,
                            expr,
                            collect_steps,
                        )
                    );
                }
                return_root_shortcut_pair!(try_standard_repeated_phase_shift_pair_shortcut(
                    &mut simplifier.context,
                    expr,
                    collect_steps,
                ));
                return_root_shortcut_pair!(try_standard_direct_known_pair_zero_shortcut(
                    &self.options,
                    &mut simplifier.context,
                    expr,
                    collect_steps,
                ));
                return_root_shortcut_pair!(
                    try_standard_exact_additive_pair_chain_pipeline_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                );
                return_root_shortcut_pair!(try_standard_common_scale_known_pair_shortcut(
                    &self.options,
                    &mut simplifier.context,
                    expr,
                    collect_steps,
                ));
                return_root_shortcut_pair!(
                    try_standard_trig_double_angle_cos_variant_zero_shortcut(
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                );
                return_root_shortcut_pair!(try_standard_direct_sum_to_product_root_shortcut(
                    &self.options,
                    &mut simplifier.context,
                    expr,
                    collect_steps,
                ));
                return_root_shortcut_pair!(try_standard_direct_trig_sum_product_zero_shortcut(
                    &mut simplifier.context,
                    expr,
                    collect_steps,
                ));
                return_root_shortcut_pair!(try_standard_exact_zero_subset_passthrough_shortcut(
                    &self.options,
                    &mut simplifier.context,
                    expr,
                    collect_steps,
                ));
                return_root_shortcut_pair!(
                    try_standard_binary_exact_zero_subset_passthrough_pair_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                );
                return_root_shortcut_pair!(try_standard_direct_small_zero_identity_shortcut(
                    &self.options,
                    &mut simplifier.context,
                    expr,
                    collect_steps,
                ));
                return_root_shortcut_pair!(
                    try_standard_direct_small_zero_additive_combination_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                );
                if let Some((result, shortcut_steps)) =
                    try_standard_reciprocal_pythagorean_zero_shortcut(
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_trig_power_reduction_zero_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_subtract_expanded_sum_diff_cubes_quotient_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_shared_passthrough_small_pow_expansion_shortcut(
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_small_composed_additive_pair_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_shared_passthrough_direct_pair_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_common_scale_known_pair_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_reciprocal_trig_zero_pair_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_direct_trig_mixed_zero_pair_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_guarded_small_zero_pair_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_symbolic_root_denesting_subset_zero_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_sqrt_perfect_square_abs_subset_zero_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_inverse_trig_plus_sqrt_subset_zero_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_inverse_trig_composition_subset_zero_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_atanh_square_ratio_log_subset_zero_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_hyperbolic_cosh_cubic_subset_zero_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) = try_standard_half_angle_subset_zero_shortcut(
                    &self.options,
                    &mut simplifier.context,
                    expr,
                    collect_steps,
                ) {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_shared_passthrough_pythagorean_factor_form_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if collect_steps {
                    if let Some((result, shortcut_steps)) =
                        try_standard_nested_exact_zero_child_shortcut(
                            &self.options,
                            &mut simplifier.context,
                            expr,
                            collect_steps,
                        )
                    {
                        return (
                            result,
                            shortcut_steps,
                            crate::phase::PipelineStats::default(),
                        );
                    }
                }
                if !is_nested_additive_pair_root(&simplifier.context, expr) {
                    if let Some((result, shortcut_steps)) =
                        try_standard_pythagorean_generic_coefficient_shortcut(
                            &self.options,
                            &mut simplifier.context,
                            expr,
                            collect_steps,
                        )
                    {
                        return (
                            result,
                            shortcut_steps,
                            crate::phase::PipelineStats::default(),
                        );
                    }
                }
                if is_mixed_sign_trig_square_difference_root(&simplifier.context, expr) {
                    return (expr, Vec::new(), crate::phase::PipelineStats::default());
                }
                if has_negative_numeric_pythagorean_pair(&simplifier.context, expr) {
                    if let Some((result, shortcut_steps)) =
                        try_standard_pythagorean_additive_pipeline_shortcut(
                            &self.options,
                            &mut simplifier.context,
                            expr,
                            collect_steps,
                        )
                    {
                        return (
                            result,
                            shortcut_steps,
                            crate::phase::PipelineStats::default(),
                        );
                    }
                }
                if has_numeric_pythagorean_complement_pair(&simplifier.context, expr) {
                    if let Some((result, shortcut_steps)) =
                        try_standard_pythagorean_additive_pipeline_shortcut(
                            &self.options,
                            &mut simplifier.context,
                            expr,
                            collect_steps,
                        )
                    {
                        return (
                            result,
                            shortcut_steps,
                            crate::phase::PipelineStats::default(),
                        );
                    }
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_reciprocal_product_pythagorean_zero_shortcut(
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_trig_binomial_square_double_angle_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_sin_sum_triple_identity_zero_shortcut(
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_trig_fourth_power_difference_shortcut(
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                let sticky_domain = simplifier.sticky_implicit_domain().cloned();
                if let Some((result, shortcut_steps)) =
                    try_standard_abs_domain_add_sub_cancellation_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                        sticky_domain,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if !is_nested_additive_pair_root(&simplifier.context, expr) {
                    if let Some((result, shortcut_steps)) =
                        try_standard_exact_zero_equivalence_shortcut(
                            &self.options,
                            &mut simplifier.context,
                            expr,
                            collect_steps,
                        )
                    {
                        return (
                            result,
                            shortcut_steps,
                            crate::phase::PipelineStats::default(),
                        );
                    }
                }
            }
            if div_root {
                if let Some((result, shortcut_steps)) = run_profiled_root_shortcut(
                    "root.div.00.reciprocal_trig_half_angle_fraction_passthrough",
                    || {
                        try_standard_reciprocal_trig_half_angle_fraction_passthrough_shortcut(
                            &mut simplifier.context,
                            expr,
                            collect_steps,
                        )
                    },
                ) {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_direct_positive_double_cos_square_diff_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_direct_trig_power_reduction_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_direct_half_angle_square_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                let div_pair = match simplifier.context.get(expr) {
                    Expr::Div(numerator, denominator) => Some((*numerator, *denominator)),
                    _ => None,
                };
                let div_passthrough_cores = div_pair.and_then(|(numerator, denominator)| {
                    extract_shifted_quotient_positive_one_passthrough_cores_root(
                        &mut simplifier.context,
                        numerator,
                        denominator,
                    )
                });
                let try_direct_small_zero_exact_one_first = run_profiled_orchestrator_bool_section(
                    "root.div.00.shifted_quotient_direct_small_zero_exact_one_gate",
                    || {
                        div_passthrough_cores.is_some_and(|(numerator_core, denominator_core)| {
                            run_profiled_orchestrator_bool_section(
                                "root.div.00a.shifted_quotient_direct_small_zero_exact_one_candidate_gate",
                                || {
                                    matches_shifted_quotient_direct_small_zero_hot_gate_root(
                                        &mut simplifier.context,
                                        numerator_core,
                                        denominator_core,
                                    ) || matches_direct_small_zero_pair_root(
                                        &mut simplifier.context,
                                        numerator_core,
                                        denominator_core,
                                    )
                                },
                            )
                        })
                    },
                );
                if try_direct_small_zero_exact_one_first {
                    if let Some((result, shortcut_steps)) = run_profiled_root_shortcut(
                        "root.div.02.shifted_quotient_exact_one",
                        || {
                            try_standard_shifted_quotient_exact_one_shortcut_with_direct_small_zero_hint(
                                &self.options,
                                &mut simplifier.context,
                                expr,
                                collect_steps,
                                true,
                            )
                        },
                    ) {
                        return (
                            result,
                            shortcut_steps,
                            crate::phase::PipelineStats::default(),
                        );
                    }
                }
                let try_nested_zero_first =
                    div_passthrough_cores.is_some_and(|(numerator_core, denominator_core)| {
                        matches_shifted_quotient_nested_zero_fast_gate_candidate_from_cores_root(
                            &mut simplifier.context,
                            numerator_core,
                            denominator_core,
                        )
                    });
                if try_nested_zero_first {
                    if let Some((result, shortcut_steps)) = run_profiled_root_shortcut(
                        "root.div.03.shifted_quotient_nested_zero_core",
                        || {
                            try_standard_shifted_quotient_nested_zero_core_shortcut(
                                &self.options,
                                &mut simplifier.context,
                                expr,
                                collect_steps,
                            )
                        },
                    ) {
                        return (
                            result,
                            shortcut_steps,
                            crate::phase::PipelineStats::default(),
                        );
                    }
                }
                let try_exact_one_first = run_profiled_orchestrator_bool_section(
                    "root.div.01.shifted_quotient_exact_one_gate",
                    || {
                        if try_direct_small_zero_exact_one_first {
                            return false;
                        }
                        div_passthrough_cores.is_some_and(|(numerator_core, denominator_core)| {
                            run_profiled_orchestrator_bool_section(
                                "root.div.01a.shifted_quotient_exact_one_candidate_gate",
                                || {
                                    matches_shifted_quotient_exact_one_root_gate_candidate(
                                        &mut simplifier.context,
                                        numerator_core,
                                        denominator_core,
                                    )
                                },
                            )
                        })
                    },
                );
                if try_exact_one_first {
                    if let Some((result, shortcut_steps)) =
                        run_profiled_root_shortcut("root.div.02.shifted_quotient_exact_one", || {
                            try_standard_shifted_quotient_exact_one_shortcut(
                                &self.options,
                                &mut simplifier.context,
                                expr,
                                collect_steps,
                            )
                        })
                    {
                        return (
                            result,
                            shortcut_steps,
                            crate::phase::PipelineStats::default(),
                        );
                    }
                }
                if !try_nested_zero_first {
                    if let Some((result, shortcut_steps)) = run_profiled_root_shortcut(
                        "root.div.03.shifted_quotient_nested_zero_core",
                        || {
                            try_standard_shifted_quotient_nested_zero_core_shortcut(
                                &self.options,
                                &mut simplifier.context,
                                expr,
                                collect_steps,
                            )
                        },
                    ) {
                        return (
                            result,
                            shortcut_steps,
                            crate::phase::PipelineStats::default(),
                        );
                    }
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_sum_diff_cubes_fraction_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                let try_exact_one_late = run_profiled_orchestrator_bool_section(
                    "root.div.04a.shifted_quotient_exact_one_candidate_gate_late",
                    || {
                        div_pair.is_some_and(|(numerator, denominator)| {
                            strip_positive_one_passthrough_root(&mut simplifier.context, numerator)
                                .zip(strip_positive_one_passthrough_root(
                                    &mut simplifier.context,
                                    denominator,
                                ))
                                .is_some_and(|(numerator_core, denominator_core)| {
                                    matches_shifted_quotient_exact_one_root_gate_candidate(
                                        &mut simplifier.context,
                                        numerator_core,
                                        denominator_core,
                                    )
                                })
                        })
                    },
                );
                if try_exact_one_late {
                    if let Some((result, shortcut_steps)) = run_profiled_root_shortcut(
                        "root.div.04.shifted_quotient_exact_one_late",
                        || {
                            try_standard_shifted_quotient_exact_one_shortcut(
                                &self.options,
                                &mut simplifier.context,
                                expr,
                                collect_steps,
                            )
                        },
                    ) {
                        return (
                            result,
                            shortcut_steps,
                            crate::phase::PipelineStats::default(),
                        );
                    }
                }
            }
        }

        if !root_shortcut_matrix_guard
            && matches!(
                self.options.shared.context_mode,
                crate::options::ContextMode::Standard | crate::options::ContextMode::Auto
            )
            && !simplifier.has_step_listener()
        {
            let mut shortcut_steps = Vec::new();
            let allow_definability_shortcuts = allow_definability_root_shortcuts(&self.options);
            let add_root = matches!(simplifier.context.get(expr), Expr::Add(_, _));
            let sub_root = matches!(simplifier.context.get(expr), Expr::Sub(_, _));
            let div_root = matches!(simplifier.context.get(expr), Expr::Div(_, _));
            let pow_root = matches!(simplifier.context.get(expr), Expr::Pow(_, _));
            if !is_nested_additive_pair_root(&simplifier.context, expr) {
                if let Some((result, shortcut_steps)) =
                    try_standard_pythagorean_generic_coefficient_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
            }
            if div_root {
                return_root_shortcut_pair!(
                    try_standard_small_polynomial_denominator_factor_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                );
            }
            if add_root || sub_root {
                let sticky_domain = simplifier.sticky_implicit_domain().cloned();
                return_root_shortcut_pair!(try_standard_abs_domain_add_sub_cancellation_shortcut(
                    &self.options,
                    &mut simplifier.context,
                    expr,
                    collect_steps,
                    sticky_domain,
                ));
                return_root_shortcut_pair!(
                    try_standard_nested_fraction_zero_hyperbolic_identity_pair_shortcut(
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                );
                return_root_shortcut_pair!(
                    try_standard_log_zero_hyperbolic_cosh_cubic_pair_shortcut(
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                );
                return_root_shortcut_pair!(
                    try_standard_targeted_direct_small_zero_additive_combination_shortcut(
                        simplifier,
                        expr,
                        collect_steps,
                    )
                );
                return_root_shortcut_pair!(try_standard_direct_small_zero_identity_shortcut(
                    &self.options,
                    &mut simplifier.context,
                    expr,
                    collect_steps,
                ));
                return_root_shortcut_pair!(
                    try_standard_direct_small_zero_additive_combination_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                );
                return_root_shortcut_pair!(
                    try_standard_partitioned_direct_small_zero_sum_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                );
                return_root_shortcut_pair!(try_standard_direct_small_zero_pair_shortcut(
                    &self.options,
                    &mut simplifier.context,
                    expr,
                    collect_steps,
                ));
                return_root_shortcut_pair!(try_standard_reciprocal_pythagorean_zero_shortcut(
                    &mut simplifier.context,
                    expr,
                    collect_steps,
                ));
                return_root_shortcut_pair!(
                    try_standard_subtract_expanded_sum_diff_cubes_quotient_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                );
            }
            if sub_root {
                return_root_shortcut_pair!(try_standard_sub_self_cancel_shortcut(
                    &self.options,
                    &mut simplifier.context,
                    expr,
                    collect_steps,
                ));
            }
            if add_root {
                if is_symbolic_atom_plus_nonzero_literal_root(&simplifier.context, expr) {
                    return (expr, Vec::new(), crate::phase::PipelineStats::default());
                }
                return_root_shortcut_pair!(try_standard_numeric_add_chain_shortcut(
                    &mut simplifier.context,
                    expr,
                    collect_steps,
                ));
                return_root_shortcut_pair!(
                    try_standard_reciprocal_product_pythagorean_zero_shortcut(
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                );
                return_root_shortcut_pair!(
                    try_standard_trig_binomial_square_double_angle_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                );
                return_root_shortcut_pair!(try_standard_sin_sum_triple_identity_zero_shortcut(
                    &mut simplifier.context,
                    expr,
                    collect_steps,
                ));
                return_root_shortcut_pair!(try_standard_trig_fourth_power_difference_shortcut(
                    &mut simplifier.context,
                    expr,
                    collect_steps,
                ));
                if try_rewrite_pythagorean_chain_add_expr(&mut simplifier.context, expr).is_some() {
                    return_root_shortcut_pair!(
                        try_standard_pythagorean_additive_pipeline_shortcut(
                            &self.options,
                            &mut simplifier.context,
                            expr,
                            collect_steps,
                        )
                    );
                }
            }

            if matches!(simplifier.context.get(expr), Expr::Function(_, _)) {
                return_root_shortcut_pair!(try_standard_abs_shortcut(
                    &self.options,
                    &mut simplifier.context,
                    expr,
                    collect_steps,
                ));
                return_root_shortcut_pair!(try_standard_simplify_square_root_shortcut(
                    &self.options,
                    &mut simplifier.context,
                    expr,
                    collect_steps,
                ));
                return_root_shortcut_pair!(try_standard_extract_perfect_square_root_shortcut(
                    &self.options,
                    &mut simplifier.context,
                    expr,
                    collect_steps,
                ));
            }

            if pow_root {
                if !extract_shortcut_declines_for_value_domain(
                    &self.options,
                    &simplifier.context,
                    expr,
                ) {
                    if let Some(extract) = try_rewrite_extract_perfect_power_from_radicand_expr(
                        &mut simplifier.context,
                        expr,
                    ) {
                        let rewrite = crate::rule::Rewrite::new(extract.rewritten)
                            .desc("Extract perfect square from under radical");
                        let (result, shortcut_steps) = finish_standard_root_shortcut(
                            &simplifier.context,
                            expr,
                            rewrite,
                            "Extract Perfect Square from Radicand",
                            collect_steps,
                        );
                        return (
                            result,
                            shortcut_steps,
                            crate::phase::PipelineStats::default(),
                        );
                    }
                }

                let parent_ctx =
                    build_root_shortcut_parent_ctx(&self.options, &simplifier.context, expr);
                let root_pow_cancel = crate::rules::exponents::RootPowCancelRule;
                if let Some(rewrite) = crate::rule::Rule::apply(
                    &root_pow_cancel,
                    &mut simplifier.context,
                    expr,
                    &parent_ctx,
                ) {
                    let (result, shortcut_steps) = finish_standard_root_shortcut(
                        &simplifier.context,
                        expr,
                        rewrite,
                        "Root Power Cancel",
                        collect_steps,
                    );
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
            }

            let div_parts = match simplifier.context.get(expr) {
                Expr::Div(num, den) => Some((*num, *den)),
                _ => None,
            };
            if let Some((num, den)) = div_parts {
                if allow_definability_shortcuts {
                    if let Some(result) = crate::rules::algebra::try_difference_of_squares_preorder(
                        &mut simplifier.context,
                        expr,
                        num,
                        den,
                        self.options.shared.semantics.domain_mode,
                        self.options.shared.semantics.value_domain,
                        self.options.shared.semantics.value_domain
                            == crate::semantics::ValueDomain::RealOnly,
                        collect_steps,
                        &mut shortcut_steps,
                        &[],
                    ) {
                        return (
                            result,
                            shortcut_steps,
                            crate::phase::PipelineStats::default(),
                        );
                    }
                    if let Some(result) = crate::rules::algebra::try_sum_diff_of_cubes_preorder(
                        &mut simplifier.context,
                        expr,
                        num,
                        den,
                        collect_steps,
                        &mut shortcut_steps,
                        &[],
                    ) {
                        return (
                            result,
                            shortcut_steps,
                            crate::phase::PipelineStats::default(),
                        );
                    }
                    if let Some(result) =
                        crate::rules::algebra::try_exact_common_factor_mul_fraction_preorder(
                            &mut simplifier.context,
                            expr,
                            num,
                            den,
                            self.options.shared.semantics.domain_mode,
                            self.options.shared.semantics.value_domain,
                            collect_steps,
                            &mut shortcut_steps,
                            &[],
                        )
                    {
                        return (
                            result,
                            shortcut_steps,
                            crate::phase::PipelineStats::default(),
                        );
                    }
                    if let Some((result, shortcut_steps)) =
                        try_standard_exact_two_term_scalar_multiple_shortcut(
                            &mut simplifier.context,
                            expr,
                            self.options.shared.semantics.domain_mode,
                            self.options.shared.semantics.value_domain,
                            collect_steps,
                        )
                    {
                        return (
                            result,
                            shortcut_steps,
                            crate::phase::PipelineStats::default(),
                        );
                    }
                    if let Some(result) =
                        crate::rules::algebra::try_exact_scalar_multiple_fraction_preorder(
                            &mut simplifier.context,
                            expr,
                            num,
                            den,
                            self.options.shared.semantics.domain_mode,
                            self.options.shared.semantics.value_domain,
                            collect_steps,
                            &mut shortcut_steps,
                            &[],
                        )
                    {
                        return (
                            result,
                            shortcut_steps,
                            crate::phase::PipelineStats::default(),
                        );
                    }
                }
            }
        }

        if !collect_steps && is_solve_mode && !simplifier.has_step_listener() {
            let domain_is_strict = self.options.shared.semantics.domain_mode.is_strict();
            let allow_scalar_root = allow_hidden_solve_root_scalar_multiple_shortcut(&self.options);
            let allow_definability_shortcuts = allow_definability_root_shortcuts(&self.options);
            let (is_pow_root, is_function_root, div_parts) = match simplifier.context.get(expr) {
                Expr::Pow(_, _) => (true, false, None),
                Expr::Function(_, _) => (false, true, None),
                Expr::Div(num, den) => (false, false, Some((*num, *den))),
                _ => (false, false, None),
            };

            if is_function_root && !domain_is_strict {
                if let Some((result, shortcut_steps)) =
                    try_hidden_solve_root_log_power_base_shortcut(
                        &mut simplifier.context,
                        expr,
                        self.options.shared.semantics.domain_mode,
                        self.options.shared.semantics.value_domain,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
            }

            if is_pow_root && !domain_is_strict {
                if let Some(result) =
                    try_hidden_solve_root_exp_ln_shortcut(&mut simplifier.context, expr)
                {
                    return (result, Vec::new(), crate::phase::PipelineStats::default());
                }
                if is_symbolic_pow_zero_root(&simplifier.context, expr) {
                    return (
                        simplifier.context.num(1),
                        Vec::new(),
                        crate::phase::PipelineStats::default(),
                    );
                }
            }

            if let Some((num, den)) = div_parts {
                if !domain_is_strict {
                    if is_symbolic_power_over_same_atom_noop_root(&simplifier.context, expr) {
                        return (expr, Vec::new(), crate::phase::PipelineStats::default());
                    }
                    match simplifier.context.get(den) {
                        Expr::Variable(_) | Expr::Constant(_) => {
                            if allow_scalar_root {
                                if let Some(result) =
                                    try_hidden_solve_root_identical_atom_fraction_shortcut(
                                        &mut simplifier.context,
                                        expr,
                                    )
                                {
                                    return (
                                        result,
                                        Vec::new(),
                                        crate::phase::PipelineStats::default(),
                                    );
                                }
                            }
                        }
                        Expr::Pow(_, _) => {
                            if let Some(result) = try_hidden_solve_root_binomial_square_shortcut(
                                &mut simplifier.context,
                                expr,
                            ) {
                                return (
                                    result,
                                    Vec::new(),
                                    crate::phase::PipelineStats::default(),
                                );
                            }
                            if allow_scalar_root {
                                if let Some(result) = try_hidden_solve_root_power_quotient_shortcut(
                                    &mut simplifier.context,
                                    expr,
                                    self.options.shared.semantics.domain_mode,
                                ) {
                                    return (
                                        result,
                                        Vec::new(),
                                        crate::phase::PipelineStats::default(),
                                    );
                                }
                            }
                        }
                        Expr::Add(_, _) => {
                            if allow_definability_shortcuts {
                                if let Some(result) =
                                    crate::rules::algebra::try_exact_sum_diff_of_cubes_preorder(
                                        &mut simplifier.context,
                                        num,
                                        den,
                                    )
                                {
                                    return (
                                        result,
                                        Vec::new(),
                                        crate::phase::PipelineStats::default(),
                                    );
                                }
                            }
                            if allow_scalar_root {
                                if let Some(result) =
                                    try_hidden_solve_root_exact_two_term_scalar_multiple_shortcut(
                                        &mut simplifier.context,
                                        expr,
                                    )
                                {
                                    return (
                                        result,
                                        Vec::new(),
                                        crate::phase::PipelineStats::default(),
                                    );
                                }
                                if let Some(result) =
                                    crate::rules::algebra::try_structural_scalar_multiple_preorder(
                                        &mut simplifier.context,
                                        num,
                                        den,
                                        self.options.shared.semantics.domain_mode,
                                        self.options.shared.semantics.value_domain,
                                    )
                                {
                                    return (
                                        result,
                                        Vec::new(),
                                        crate::phase::PipelineStats::default(),
                                    );
                                }
                            }
                        }
                        Expr::Sub(_, _) => {
                            if allow_definability_shortcuts {
                                if let Some(result) =
                                    crate::rules::algebra::try_exact_sum_diff_of_cubes_preorder(
                                        &mut simplifier.context,
                                        num,
                                        den,
                                    )
                                {
                                    return (
                                        result,
                                        Vec::new(),
                                        crate::phase::PipelineStats::default(),
                                    );
                                }
                                if let Some(result) =
                                    try_hidden_solve_root_difference_of_squares_shortcut(
                                        &mut simplifier.context,
                                        expr,
                                    )
                                {
                                    return (
                                        result,
                                        Vec::new(),
                                        crate::phase::PipelineStats::default(),
                                    );
                                }
                                if let Some(result) =
                                    try_hidden_solve_root_perfect_square_minus_shortcut(
                                        &mut simplifier.context,
                                        expr,
                                    )
                                {
                                    return (
                                        result,
                                        Vec::new(),
                                        crate::phase::PipelineStats::default(),
                                    );
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        // Clear cycle events only when we are about to enter the heavy phase
        // pipeline. Hidden root shortcuts above do not register or consume
        // cycle events, so clearing here avoids fixed overhead on the hot early
        // return paths without changing final stats on full runs.
        cas_solver_core::cycle_event_registry::clear_cycle_events();

        // Clear thread-local PolyStore before evaluation
        clear_thread_local_store();

        // V2.15.8: Set sticky implicit domain from original input to propagate inherited
        // requires across the phase pipeline. Hidden solve root shortcuts above do not need it,
        // because final diagnostics re-derive implicit conditions from input/result.
        if simplifier.sticky_implicit_domain().is_none() {
            simplifier.set_sticky_implicit_domain(expr, self.options.shared.semantics.value_domain);
        }

        // PRE-PASS 1: Eager eval for expand() calls using fast mod-p path
        // This runs BEFORE any simplification to avoid budget exhaustion on huge arguments
        let (current, expand_steps) =
            eager_eval_expand_calls(&mut simplifier.context, expr, collect_steps);
        let mut all_steps = expand_steps;

        // PRE-PASS 2: Eager eval for special functions (poly_gcd_modp)
        let (current, eager_steps) =
            run_poly_gcd_modp_eager_pass(&mut simplifier.context, current, collect_steps);
        all_steps.extend(eager_steps);

        // PRE-PASS 3: Poly lowering - combine poly_result operations before simplification
        // This handles poly_result(0) + poly_result(1) → poly_result(2) internally
        let (current, lower_steps) =
            run_poly_lower_pass(&mut simplifier.context, current, collect_steps);
        all_steps.extend(lower_steps);

        if self.time_budget_exceeded() {
            return self.finish_timed_out_pipeline(
                simplifier,
                current,
                all_steps,
                crate::phase::PipelineStats::default(),
                None,
            );
        }

        // Check for specialized strategies first
        if let Some((zero, mut shortcut_steps, stats)) =
            try_finish_dirichlet_kernel_root_shortcut(simplifier, current, collect_steps)
        {
            all_steps.append(&mut shortcut_steps);
            return (zero, all_steps, stats);
        }

        let mut pipeline_stats = crate::phase::PipelineStats::default();

        // Copy values to avoid borrow conflicts with &mut self in run_phase
        let budgets = self.options.budgets;
        let enable_transform = self.options.enable_transform;
        let auto_level = self.options.rationalize.auto_level;

        // V2.15.25: Best-So-Far tracking to prevent returning worse expressions
        // Initialize BSF AFTER Core phase (not from raw input) to preserve Phase 1 canonicalizations
        // This prevents reverting beneficial transformations like tan→sin/cos, arcsec→arccos, etc.
        let budget = BestSoFarBudget::default();

        // Phase 1: Core - Safe simplifications (canonicalizations, basic identities)
        let (next, steps, stats) =
            self.run_phase(simplifier, current, SimplifyPhase::Core, budgets.core_iters);
        let mut current = next;
        all_steps.extend(steps);
        pipeline_stats.core = stats;
        pipeline_stats.total_rewrites += pipeline_stats.core.rewrites_used;
        if pipeline_stats.core.timed_out {
            return self.finish_timed_out_pipeline(
                simplifier,
                current,
                all_steps,
                pipeline_stats,
                None,
            );
        }
        // Fast path: when Core already collapses to a terminal exact value and the
        // caller is not collecting steps, later phases are pure fixed-cost noise.
        if !collect_steps && is_terminal_after_core(&simplifier.context, current) {
            pipeline_stats.rationalize_level = Some(auto_level);
            pipeline_stats.rationalize_outcome = Some(if auto_level != AutoRationalizeLevel::Off {
                cas_solver_core::rationalize_policy::RationalizeOutcome::NotApplied(
                    cas_solver_core::rationalize_policy::RationalizeReason::NoBinomialFound,
                )
            } else {
                cas_solver_core::rationalize_policy::RationalizeOutcome::NotApplied(
                    cas_solver_core::rationalize_policy::RationalizeReason::PolicyDisabled,
                )
            });
            pipeline_stats.cycle_events =
                cas_solver_core::cycle_event_registry::take_cycle_events();
            simplifier.clear_sticky_implicit_domain();
            return (current, all_steps, pipeline_stats);
        }

        if !collect_steps
            && is_solve_mode
            && !self.options.shared.semantics.domain_mode.is_strict()
            && matches!(simplifier.context.get(expr), Expr::Div(_, _))
            && !self.pattern_marks.has_root_in_denominator()
            && !self.pattern_marks.has_auto_expand_contexts()
            && is_plain_symbolic_power_after_core(&simplifier.context, current)
        {
            pipeline_stats.rationalize_level = Some(auto_level);
            pipeline_stats.rationalize_outcome = Some(if auto_level != AutoRationalizeLevel::Off {
                cas_solver_core::rationalize_policy::RationalizeOutcome::NotApplied(
                    cas_solver_core::rationalize_policy::RationalizeReason::NoBinomialFound,
                )
            } else {
                cas_solver_core::rationalize_policy::RationalizeOutcome::NotApplied(
                    cas_solver_core::rationalize_policy::RationalizeReason::PolicyDisabled,
                )
            });
            pipeline_stats.cycle_events =
                cas_solver_core::cycle_event_registry::take_cycle_events();
            simplifier.clear_sticky_implicit_domain();
            return (current, all_steps, pipeline_stats);
        }

        // Narrow solve fast path: symbolic atom^x / atom with no didactic work.
        // Current solve generic/assume behavior leaves this unchanged, and the
        // later phases are pure overhead on the plain result-only path.
        if !collect_steps
            && is_solve_mode
            && !self.pattern_marks.has_root_in_denominator()
            && !self.pattern_marks.has_auto_expand_contexts()
            && is_symbolic_power_over_same_atom_noop_after_core(&simplifier.context, current)
        {
            pipeline_stats.rationalize_level = Some(auto_level);
            pipeline_stats.rationalize_outcome = Some(if auto_level != AutoRationalizeLevel::Off {
                cas_solver_core::rationalize_policy::RationalizeOutcome::NotApplied(
                    cas_solver_core::rationalize_policy::RationalizeReason::NoBinomialFound,
                )
            } else {
                cas_solver_core::rationalize_policy::RationalizeOutcome::NotApplied(
                    cas_solver_core::rationalize_policy::RationalizeReason::PolicyDisabled,
                )
            });
            pipeline_stats.cycle_events =
                cas_solver_core::cycle_event_registry::take_cycle_events();
            simplifier.clear_sticky_implicit_domain();
            return (current, all_steps, pipeline_stats);
        }

        // Another narrow solve fast path: after Core, symbolic sums like
        // `x + y` do not benefit from later phases on the hidden
        // result-only path.
        if !collect_steps
            && is_solve_mode
            && !self.pattern_marks.has_root_in_denominator()
            && !self.pattern_marks.has_auto_expand_contexts()
            && is_plain_symbolic_binomial_after_core(&simplifier.context, current)
        {
            pipeline_stats.rationalize_level = Some(auto_level);
            pipeline_stats.rationalize_outcome = Some(if auto_level != AutoRationalizeLevel::Off {
                cas_solver_core::rationalize_policy::RationalizeOutcome::NotApplied(
                    cas_solver_core::rationalize_policy::RationalizeReason::NoBinomialFound,
                )
            } else {
                cas_solver_core::rationalize_policy::RationalizeOutcome::NotApplied(
                    cas_solver_core::rationalize_policy::RationalizeReason::PolicyDisabled,
                )
            });
            pipeline_stats.cycle_events =
                cas_solver_core::cycle_event_registry::take_cycle_events();
            simplifier.clear_sticky_implicit_domain();
            return (current, all_steps, pipeline_stats);
        }

        // Same hidden solve fast path for exact cube outputs like
        // `x^2 + y^2 +/- x*y`, which are already in their plain final form
        // after Core and only pay late-phase overhead.
        if !collect_steps
            && is_solve_mode
            && !self.pattern_marks.has_root_in_denominator()
            && !self.pattern_marks.has_auto_expand_contexts()
            && is_plain_symbolic_cube_trinomial_after_core(&simplifier.context, current)
        {
            pipeline_stats.rationalize_level = Some(auto_level);
            pipeline_stats.rationalize_outcome = Some(if auto_level != AutoRationalizeLevel::Off {
                cas_solver_core::rationalize_policy::RationalizeOutcome::NotApplied(
                    cas_solver_core::rationalize_policy::RationalizeReason::NoBinomialFound,
                )
            } else {
                cas_solver_core::rationalize_policy::RationalizeOutcome::NotApplied(
                    cas_solver_core::rationalize_policy::RationalizeReason::PolicyDisabled,
                )
            });
            pipeline_stats.cycle_events =
                cas_solver_core::cycle_event_registry::take_cycle_events();
            simplifier.clear_sticky_implicit_domain();
            return (current, all_steps, pipeline_stats);
        }

        // Initialize BSF lazily from the post-Core baseline.
        // Many solve hot paths stop changing after Core; deferring the score work
        // avoids paying BSF overhead when later phases are pure no-ops.
        let best_baseline_expr = current;
        let best_baseline_steps_len = all_steps.len();
        let mut best: Option<BestSoFar> = None;

        // Phase 2: Transform - Distribution, expansion (if enabled)
        if enable_transform {
            let (next, steps, stats) = self.run_phase(
                simplifier,
                current,
                SimplifyPhase::Transform,
                budgets.transform_iters,
            );
            current = next;
            all_steps.extend(steps);
            pipeline_stats.transform = stats;
            pipeline_stats.total_rewrites += pipeline_stats.transform.rewrites_used;
            if pipeline_stats.transform.changed {
                let best_ref = best.get_or_insert_with(|| {
                    BestSoFar::new(
                        best_baseline_expr,
                        &all_steps[..best_baseline_steps_len],
                        &simplifier.context,
                        budget,
                    )
                });
                best_ref.consider(current, &all_steps, &simplifier.context);
            }
            if pipeline_stats.transform.timed_out {
                return self.finish_timed_out_pipeline(
                    simplifier,
                    current,
                    all_steps,
                    pipeline_stats,
                    best.as_ref(),
                );
            }
        }

        // Narrow hidden solve fast path: if Transform lands on a plain symbolic
        // binomial, later phases are fixed-cost overhead on the result-only path.
        if !collect_steps
            && is_solve_mode
            && pipeline_stats.transform.changed
            && !self.pattern_marks.has_root_in_denominator()
            && !self.pattern_marks.has_auto_expand_contexts()
            && is_plain_symbolic_binomial_after_core(&simplifier.context, current)
        {
            pipeline_stats.rationalize_level = Some(auto_level);
            pipeline_stats.rationalize_outcome = Some(if auto_level != AutoRationalizeLevel::Off {
                cas_solver_core::rationalize_policy::RationalizeOutcome::NotApplied(
                    cas_solver_core::rationalize_policy::RationalizeReason::NoBinomialFound,
                )
            } else {
                cas_solver_core::rationalize_policy::RationalizeOutcome::NotApplied(
                    cas_solver_core::rationalize_policy::RationalizeReason::PolicyDisabled,
                )
            });
            pipeline_stats.cycle_events =
                cas_solver_core::cycle_event_registry::take_cycle_events();
            simplifier.clear_sticky_implicit_domain();
            return (current, all_steps, pipeline_stats);
        }

        // Phase 3: Rationalize - Auto-rationalization per policy
        // Skip the whole phase when the pre-scan proves there is no root-like
        // form anywhere inside a denominator subtree.
        let should_run_rationalize =
            auto_level != AutoRationalizeLevel::Off && self.pattern_marks.has_root_in_denominator();
        if should_run_rationalize {
            let (next, steps, stats) = self.run_phase(
                simplifier,
                current,
                SimplifyPhase::Rationalize,
                budgets.rationalize_iters,
            );

            // Track rationalization outcome
            pipeline_stats.rationalize_level = Some(auto_level);
            if stats.changed {
                pipeline_stats.rationalize_outcome =
                    Some(cas_solver_core::rationalize_policy::RationalizeOutcome::Applied);
            } else {
                // If enabled but didn't change, it was blocked for some reason
                // We don't have detailed reason here; would need deeper integration
                pipeline_stats.rationalize_outcome = Some(
                    cas_solver_core::rationalize_policy::RationalizeOutcome::NotApplied(
                        cas_solver_core::rationalize_policy::RationalizeReason::NoBinomialFound,
                    ),
                );
            }

            current = next;
            all_steps.extend(steps);
            pipeline_stats.rationalize = stats;
            pipeline_stats.total_rewrites += pipeline_stats.rationalize.rewrites_used;
            if pipeline_stats.rationalize.changed {
                let best_ref = best.get_or_insert_with(|| {
                    BestSoFar::new(
                        best_baseline_expr,
                        &all_steps[..best_baseline_steps_len],
                        &simplifier.context,
                        budget,
                    )
                });
                best_ref.consider(current, &all_steps, &simplifier.context);
            }
            if pipeline_stats.rationalize.timed_out {
                return self.finish_timed_out_pipeline(
                    simplifier,
                    current,
                    all_steps,
                    pipeline_stats,
                    best.as_ref(),
                );
            }
        } else {
            pipeline_stats.rationalize_level = Some(auto_level);
            pipeline_stats.rationalize_outcome = Some(if auto_level == AutoRationalizeLevel::Off {
                cas_solver_core::rationalize_policy::RationalizeOutcome::NotApplied(
                    cas_solver_core::rationalize_policy::RationalizeReason::PolicyDisabled,
                )
            } else {
                cas_solver_core::rationalize_policy::RationalizeOutcome::NotApplied(
                    cas_solver_core::rationalize_policy::RationalizeReason::NoBinomialFound,
                )
            });
        }

        // Phase 4: PostCleanup - Final cleanup
        let (next, steps, stats) = self.run_phase(
            simplifier,
            current,
            SimplifyPhase::PostCleanup,
            budgets.post_iters,
        );
        current = next;
        all_steps.extend(steps);
        pipeline_stats.post_cleanup = stats;
        pipeline_stats.total_rewrites += pipeline_stats.post_cleanup.rewrites_used;
        if pipeline_stats.post_cleanup.changed {
            let best_ref = best.get_or_insert_with(|| {
                BestSoFar::new(
                    best_baseline_expr,
                    &all_steps[..best_baseline_steps_len],
                    &simplifier.context,
                    budget,
                )
            });
            best_ref.consider(current, &all_steps, &simplifier.context);
        }
        if pipeline_stats.post_cleanup.timed_out || self.time_budget_exceeded() {
            return self.finish_timed_out_pipeline(
                simplifier,
                current,
                all_steps,
                pipeline_stats,
                best.as_ref(),
            );
        }

        // Log pipeline summary
        tracing::info!(
            target: "simplify",
            core_iters = pipeline_stats.core.iters_used,
            transform_iters = pipeline_stats.transform.iters_used,
            rationalize_iters = pipeline_stats.rationalize.iters_used,
            post_iters = pipeline_stats.post_cleanup.iters_used,
            total_rewrites = pipeline_stats.total_rewrites,
            "pipeline_complete"
        );

        // Final collection for canonical form - RESPECTS domain mode
        // Use collect_with_semantics to preserve Strict definedness invariant
        let final_parent_ctx = crate::parent_context::ParentContext::root()
            .with_domain_mode(self.options.shared.semantics.domain_mode);
        let final_collected = match crate::collect::collect_with_semantics(
            &mut simplifier.context,
            current,
            &final_parent_ctx,
        ) {
            Some(result) => result.new_expr,
            None => current, // No change (blocked by Strict mode or same result)
        };
        if final_collected != current {
            if crate::ordering::compare_expr(&simplifier.context, final_collected, current)
                != std::cmp::Ordering::Equal
                && collect_steps
            {
                all_steps.push(Step::new(
                    "Final Collection",
                    "Collect",
                    current,
                    final_collected,
                    Vec::new(),
                    Some(&simplifier.context),
                ));
            }
            current = final_collected;
        }

        let late_log_zero = simplifier.context.num(0);
        let late_log_parent_ctx =
            build_root_shortcut_parent_ctx(&self.options, &simplifier.context, current);
        let late_log_rule = crate::rules::arithmetic::ExpandLogAbsMulDivToEnableCancellationRule;
        if let Some(rewrite) = crate::rule::Rule::apply(
            &late_log_rule,
            &mut simplifier.context,
            current,
            &late_log_parent_ctx,
        ) {
            if compare_expr(&simplifier.context, rewrite.new_expr, late_log_zero) == Ordering::Equal
            {
                if collect_steps {
                    let mut step = Step::with_snapshots(
                        &rewrite.description,
                        late_log_rule.name(),
                        current,
                        rewrite.new_expr,
                        smallvec::SmallVec::<[crate::step::PathStep; 8]>::new(),
                        Some(&simplifier.context),
                        current,
                        rewrite.new_expr,
                    );
                    step.importance = late_log_rule.importance();
                    {
                        let meta = step.meta_mut();
                        meta.before_local = rewrite.before_local;
                        meta.after_local = rewrite.after_local;
                        meta.assumption_events = rewrite.assumption_events.clone();
                        meta.required_conditions = rewrite.required_conditions.clone();
                        meta.poly_proof = rewrite.poly_proof.clone();
                        meta.substeps = rewrite.substeps.clone();
                    }
                    all_steps.push(step);
                }
                current = rewrite.new_expr;
            }
        }

        if let Some((rewritten, mut late_steps)) = try_finalize_trivial_additive_closure_root(
            &self.options,
            &mut simplifier.context,
            current,
            collect_steps,
        ) {
            current = rewritten;
            all_steps.append(&mut late_steps);
        }

        // Late closure for the half-power residual shortcut: the early
        // root probe runs before any phase, but Sub(diff(...), target)
        // roots only EXPOSE the scaled half-power sum after Core
        // evaluates the derivative (verification residuals). The
        // matcher gates are cheap and only fire when the residual is
        // exactly zero as a Polynomial.
        if matches!(
            simplifier.context.get(current),
            Expr::Add(_, _) | Expr::Sub(_, _) | Expr::Mul(_, _) | Expr::Div(_, _)
        ) {
            if let Some((zero, required_conditions)) = crate::calculus_residual_support::
                try_reciprocal_half_power_shared_denominator_residual_root_zero(
                    &mut simplifier.context,
                    current,
                )
            {
                simplifier.extend_required_conditions(required_conditions.clone());
                if collect_steps {
                    let mut step = build_root_shortcut_compact_step(
                        current,
                        zero,
                        "Cancel matching reciprocal half-power residual",
                        "Reciprocal Half-Power Residual",
                    );
                    step.meta_mut().required_conditions = required_conditions;
                    all_steps.push(step);
                }
                current = zero;
            }
        }

        // Filter and optimize steps
        let filtered_steps = if collect_steps {
            cas_solver_core::step_productivity_runtime::filter_non_productive_solver_steps_with_runtime_recompose_mul(
                &mut simplifier.context,
                expr,
                all_steps,
                crate::build::mul2_raw,
            )
        } else {
            all_steps
        };

        let optimized_steps = if collect_steps {
            match cas_solver_core::step_optimization_runtime::optimize_steps_semantic(
                filtered_steps,
                &simplifier.context,
                expr,
                current,
            ) {
                cas_solver_core::step_optimization_runtime::StepOptimizationResult::Steps(steps) => {
                    steps
                }
                cas_solver_core::step_optimization_runtime::StepOptimizationResult::NoSimplificationNeeded => vec![],
            }
        } else {
            filtered_steps
        };

        // Collect assumptions from steps if reporting is enabled
        // Priority: 1) structured assumption_events, 2) legacy domain_assumption string parsing
        if self.options.shared.assumption_reporting != crate::AssumptionReporting::Off {
            pipeline_stats.assumptions = crate::collect_assumption_records_from_iter(
                optimized_steps
                    .iter()
                    .flat_map(|step| step.assumption_events().iter().cloned()),
            );
        }

        // Collect cycle events detected during this pipeline run
        pipeline_stats.cycle_events = cas_solver_core::cycle_event_registry::take_cycle_events();

        // V2.15.8: Clear sticky domain when pipeline completes
        simplifier.clear_sticky_implicit_domain();

        // V2.15.25: Best-So-Far guard - use best if current is worse
        // After all processing, compare current to best seen during phases
        let Some(best) = best else {
            return (current, optimized_steps, pipeline_stats);
        };
        let best_expr = best.best_expr();
        let current_score = crate::best_so_far::score_expr(&simplifier.context, current);
        let best_score = best.best_score();

        // V2.15.35: Skip rollback for explicit expand() calls
        // When user explicitly calls expand(), they want the expanded form even if "worse"
        let has_explicit_expand =
            if let cas_ast::Expr::Function(name, _) = simplifier.context.get(expr) {
                simplifier.context.is_builtin(*name, BuiltinFn::Expand)
            } else {
                false
            };

        // Only rollback if:
        // 1. Best is strictly better AND
        // 2. Current has significantly more nodes (> 12 extra) to avoid reverting expansions
        // 3. NOT an explicit expand() call (user wants expansion)
        // Moderate-to-large increases (1-12 nodes) are allowed to preserve:
        // - Canonicalizations (tan→sin/cos, arcsec→arccos)
        // - Deliberate expansions (AutoExpandBinomials::On)
        let significant_increase = current_score.nodes > best_score.nodes + 12;

        if best_score < current_score && significant_increase && !has_explicit_expand {
            // The best seen during phases is better than final result
            // This can happen when expansion rules don't close with cancellation
            tracing::debug!(
                target: "simplify",
                best_nodes = best_score.nodes,
                current_nodes = current_score.nodes,
                "best_so_far_rollback"
            );
            // Use best expression but keep optimized steps for now
            // TODO: In phase 2, also use best_steps for consistency
            (best_expr, optimized_steps, pipeline_stats)
        } else {
            (current, optimized_steps, pipeline_stats)
        }
    }
}

#[cfg(test)]
mod tests;

mod fractions;
mod general;
mod hyperbolic;
mod logs_exp;
mod pairing;
mod radicals_powers;
mod support;
mod trig;
mod trig_angles;
mod zero_detection;

use fractions::*;
use general::*;
use hyperbolic::*;
use logs_exp::*;
use pairing::*;
use radicals_powers::*;
use support::*;
use trig::*;
use trig_angles::*;
use zero_detection::*;
