use crate::best_so_far::{BestSoFar, BestSoFarBudget};
use crate::expand::eager_eval_expand_calls;
use crate::phase::{SimplifyOptions, SimplifyPhase};
use crate::rationalize_policy::AutoRationalizeLevel;
use crate::rules::algebra::gcd_modp::eager_eval_poly_gcd_calls;
use crate::{Simplifier, Step};
use cas_ast::{BuiltinFn, Context, ExprId};
use cas_math::poly_lowering::{self, PolyLowerStepKind};
use cas_math::poly_lowering_ops::PolyBinaryOp;
use cas_math::poly_store::clear_thread_local_store;
use std::collections::HashSet;

fn to_math_auto_expand_budget(
    budget: &crate::phase::ExpandBudget,
) -> cas_math::auto_expand_scan::ExpandBudget {
    cas_math::auto_expand_scan::ExpandBudget {
        max_pow_exp: budget.max_pow_exp,
        max_base_terms: budget.max_base_terms,
        max_generated_terms: budget.max_generated_terms,
        max_vars: budget.max_vars,
    }
}

fn poly_step_message(kind: PolyLowerStepKind) -> &'static str {
    match kind {
        PolyLowerStepKind::Direct { op } => match op {
            PolyBinaryOp::Add => "Poly lowering: combined poly_result + poly_result",
            PolyBinaryOp::Sub => "Poly lowering: combined poly_result - poly_result",
            PolyBinaryOp::Mul => "Poly lowering: combined poly_result * poly_result",
        },
        PolyLowerStepKind::Promoted => "Poly lowering: promoted and combined expressions",
    }
}

fn run_poly_lower_pass(
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> (ExprId, Vec<Step>) {
    let result = poly_lowering::poly_lower_pass(ctx, expr, collect_steps);
    let steps = if collect_steps {
        result
            .steps
            .into_iter()
            .map(|step| {
                Step::new(
                    poly_step_message(step.kind),
                    "Polynomial Combination",
                    step.before,
                    step.after,
                    Vec::new(),
                    Some(ctx),
                )
            })
            .collect()
    } else {
        Vec::new()
    };

    (result.expr, steps)
}

pub struct Orchestrator {
    // Configuration for the pipeline
    pub max_iterations: usize,
    pub enable_polynomial_strategy: bool,
    /// Pre-scanned pattern marks for context-aware guards
    pub pattern_marks: crate::pattern_marks::PatternMarks,
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
            options: SimplifyOptions::default(),
        }
    }

    /// Create orchestrator for expand() command (no rationalization)
    pub fn for_expand() -> Self {
        let mut o = Self::new();
        o.options = SimplifyOptions::for_expand();
        o
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
            // Re-scan for patterns each iteration since expression may have changed
            // This ensures marks are up-to-date for the current expression tree
            self.pattern_marks = crate::pattern_marks::PatternMarks::new();
            crate::pattern_scanner::scan_and_mark_patterns(
                &simplifier.context,
                current,
                &mut self.pattern_marks,
            );

            // Auto-expand scanner: mark cancellation contexts (difference quotients)
            // Only skip in Solve mode (which should never auto-expand to preserve structure)
            // The scanner has its own strict budgets (n=2, base_terms<=3) so it's safe to always run
            let is_solve_mode =
                self.options.shared.context_mode == crate::options::ContextMode::Solve;
            let global_auto_expand = self.options.shared.expand_policy
                == crate::phase::ExpandPolicy::Auto
                && !is_solve_mode;

            // Always scan for cancellation contexts (unless in Solve mode)
            // This enables Smart Expansion: auto-expand only when it leads to cancellation
            if !is_solve_mode {
                let math_budget = to_math_auto_expand_budget(&self.options.shared.expand_budget);
                cas_math::auto_expand_scan::mark_auto_expand_candidates(
                    &simplifier.context,
                    current,
                    &math_budget,
                    &mut self.pattern_marks,
                );
            }
            let config = crate::engine::LoopConfig {
                phase,
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
            };
            let (next, steps, pass_stats) =
                simplifier.apply_rules_loop_with_config(current, &self.pattern_marks, &config);

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
            let hash = CycleDetector::semantic_hash(&simplifier.context, current);
            if !seen_hashes.insert(hash) {
                // Emit cycle event for the registry
                let expr_str = format!(
                    "{}",
                    cas_formatter::DisplayExpr {
                        context: &simplifier.context,
                        id: current,
                    }
                );
                crate::cycle_events::register_cycle_event(crate::cycle_events::CycleEvent {
                    phase,
                    period: 0, // unknown period at inter-iteration level
                    level: crate::cycle_events::CycleLevel::InterIteration,
                    rule_name: "(inter-iteration)".to_string(),
                    expr_fingerprint: hash,
                    expr_display: crate::cycle_events::truncate_display(&expr_str, 120),
                    rewrite_step: iter,
                });
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
        // PRE-ANALYSIS: Scan for patterns
        self.pattern_marks = crate::pattern_marks::PatternMarks::new();
        crate::pattern_scanner::scan_and_mark_patterns(
            &simplifier.context,
            expr,
            &mut self.pattern_marks,
        );

        // V2.15.8: Set sticky implicit domain from original input to propagate inherited requires
        // across all phases. This allows AbsNonNegativeSimplifyRule to see x≥0 from sqrt(x)
        // even after the sqrt witness is consumed.
        simplifier.set_sticky_implicit_domain(expr, self.options.shared.semantics.value_domain);

        // Clear thread-local PolyStore before evaluation
        clear_thread_local_store();

        // Clear cycle events from any previous pipeline run
        crate::cycle_events::clear_cycle_events();

        // Extract collect_steps early so pre-passes can skip Step construction
        let collect_steps = self.options.collect_steps;

        // PRE-PASS 1: Eager eval for expand() calls using fast mod-p path
        // This runs BEFORE any simplification to avoid budget exhaustion on huge arguments
        let (current, expand_steps) =
            eager_eval_expand_calls(&mut simplifier.context, expr, collect_steps);
        let mut all_steps = expand_steps;

        // PRE-PASS 2: Eager eval for special functions (poly_gcd_modp)
        let (current, eager_steps) =
            eager_eval_poly_gcd_calls(&mut simplifier.context, current, collect_steps);
        all_steps.extend(eager_steps);

        // PRE-PASS 3: Poly lowering - combine poly_result operations before simplification
        // This handles poly_result(0) + poly_result(1) → poly_result(2) internally
        let (current, lower_steps) =
            run_poly_lower_pass(&mut simplifier.context, current, collect_steps);
        all_steps.extend(lower_steps);

        // Check for specialized strategies first
        if let Some(result) =
            crate::telescoping::try_dirichlet_kernel_identity_pub(&simplifier.context, current)
        {
            let zero = simplifier.context.num(0);
            if self.options.collect_steps {
                all_steps.push(Step::new(
                    &format!(
                        "Dirichlet Kernel Identity: 1 + 2Σcos(kx) = sin((n+½)x)/sin(x/2) for n={}",
                        result.n
                    ),
                    "Trig Summation Identity",
                    current,
                    zero,
                    Vec::new(),
                    Some(&simplifier.context),
                ));
            }
            simplifier.clear_sticky_implicit_domain();
            return (zero, all_steps, crate::phase::PipelineStats::default());
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

        // Initialize BSF with post-Core state as baseline
        // This ensures canonicalizations from Core are preserved
        let mut best = BestSoFar::new(current, &all_steps, &simplifier.context, budget);

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
            best.consider(current, &all_steps, &simplifier.context);
        }

        // Phase 3: Rationalize - Auto-rationalization per policy
        if auto_level != AutoRationalizeLevel::Off {
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
                    Some(crate::rationalize_policy::RationalizeOutcome::Applied);
            } else {
                // If enabled but didn't change, it was blocked for some reason
                // We don't have detailed reason here; would need deeper integration
                pipeline_stats.rationalize_outcome =
                    Some(crate::rationalize_policy::RationalizeOutcome::NotApplied(
                        crate::rationalize_policy::RationalizeReason::NoBinomialFound,
                    ));
            }

            current = next;
            all_steps.extend(steps);
            pipeline_stats.rationalize = stats;
            pipeline_stats.total_rewrites += pipeline_stats.rationalize.rewrites_used;
            best.consider(current, &all_steps, &simplifier.context);
        } else {
            pipeline_stats.rationalize_level = Some(AutoRationalizeLevel::Off);
            pipeline_stats.rationalize_outcome =
                Some(crate::rationalize_policy::RationalizeOutcome::NotApplied(
                    crate::rationalize_policy::RationalizeReason::PolicyDisabled,
                ));
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
        best.consider(current, &all_steps, &simplifier.context);

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

        // Filter and optimize steps
        let filtered_steps = if collect_steps {
            crate::strategies::filter_non_productive_steps(&mut simplifier.context, expr, all_steps)
        } else {
            all_steps
        };

        let optimized_steps = if collect_steps {
            match crate::step_optimization::optimize_steps_semantic(
                filtered_steps,
                &simplifier.context,
                expr,
                current,
            ) {
                crate::step_optimization::StepOptimizationResult::Steps(steps) => steps,
                crate::step_optimization::StepOptimizationResult::NoSimplificationNeeded => vec![],
            }
        } else {
            filtered_steps
        };

        // Collect assumptions from steps if reporting is enabled
        // Priority: 1) structured assumption_events, 2) legacy domain_assumption string parsing
        if self.options.shared.assumption_reporting != crate::assumptions::AssumptionReporting::Off
        {
            let mut collector = crate::assumptions::AssumptionCollector::new();
            for step in &optimized_steps {
                // Collect structured assumption_events
                for event in step.assumption_events() {
                    collector.note(event.clone());
                }
            }
            pipeline_stats.assumptions = collector.finish();
        }

        // Collect cycle events detected during this pipeline run
        pipeline_stats.cycle_events = crate::cycle_events::take_cycle_events();

        // V2.15.8: Clear sticky domain when pipeline completes
        simplifier.clear_sticky_implicit_domain();

        // V2.15.25: Best-So-Far guard - use best if current is worse
        // After all processing, compare current to best seen during phases
        let (best_expr, _best_steps) = best.into_parts();
        let current_score = crate::best_so_far::score_expr(&simplifier.context, current);
        let best_score = crate::best_so_far::score_expr(&simplifier.context, best_expr);

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

/// Helper struct for computing semantic hashes for cycle detection.
/// The actual cycle tracking is done by a `HashSet<u64>` in `run_phase`.
struct CycleDetector;

impl CycleDetector {
    /// Compute a hash based on the semantic structure of the expression
    /// This allows us to detect when expressions are equivalent even with different ExprIds
    /// Uses iterative traversal with depth limit to prevent stack overflow.
    fn semantic_hash(ctx: &cas_ast::Context, expr: ExprId) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        const MAX_HASH_DEPTH: usize = 200;

        // Use a work stack: (expr_id, pending_hash) to process nodes iteratively
        // For simplicity, we'll use a depth-limited recursive approach with explicit limit
        fn hash_expr_depth(
            ctx: &cas_ast::Context,
            expr: ExprId,
            depth: usize,
            hasher: &mut DefaultHasher,
        ) {
            if depth == 0 {
                // Depth limit exceeded - hash the ExprId directly as fallback
                format!("{:?}", expr).hash(hasher);
                return;
            }

            match ctx.get(expr) {
                cas_ast::Expr::Number(n) => {
                    0u8.hash(hasher);
                    n.to_string().hash(hasher);
                }
                cas_ast::Expr::Constant(c) => {
                    1u8.hash(hasher);
                    format!("{:?}", c).hash(hasher);
                }
                cas_ast::Expr::Variable(v) => {
                    2u8.hash(hasher);
                    v.hash(hasher);
                }
                cas_ast::Expr::Add(l, r) => {
                    3u8.hash(hasher);
                    // Compute child hashes for commutative sorting
                    let mut h1 = DefaultHasher::new();
                    let mut h2 = DefaultHasher::new();
                    hash_expr_depth(ctx, *l, depth - 1, &mut h1);
                    hash_expr_depth(ctx, *r, depth - 1, &mut h2);
                    let hash_l = h1.finish();
                    let hash_r = h2.finish();
                    if hash_l <= hash_r {
                        hash_l.hash(hasher);
                        hash_r.hash(hasher);
                    } else {
                        hash_r.hash(hasher);
                        hash_l.hash(hasher);
                    }
                }
                cas_ast::Expr::Sub(l, r) => {
                    4u8.hash(hasher);
                    hash_expr_depth(ctx, *l, depth - 1, hasher);
                    hash_expr_depth(ctx, *r, depth - 1, hasher);
                }
                cas_ast::Expr::Mul(l, r) => {
                    5u8.hash(hasher);
                    // Compute child hashes for commutative sorting
                    let mut h1 = DefaultHasher::new();
                    let mut h2 = DefaultHasher::new();
                    hash_expr_depth(ctx, *l, depth - 1, &mut h1);
                    hash_expr_depth(ctx, *r, depth - 1, &mut h2);
                    let hash_l = h1.finish();
                    let hash_r = h2.finish();
                    if hash_l <= hash_r {
                        hash_l.hash(hasher);
                        hash_r.hash(hasher);
                    } else {
                        hash_r.hash(hasher);
                        hash_l.hash(hasher);
                    }
                }
                cas_ast::Expr::Div(l, r) => {
                    6u8.hash(hasher);
                    hash_expr_depth(ctx, *l, depth - 1, hasher);
                    hash_expr_depth(ctx, *r, depth - 1, hasher);
                }
                cas_ast::Expr::Pow(b, e) => {
                    7u8.hash(hasher);
                    hash_expr_depth(ctx, *b, depth - 1, hasher);
                    hash_expr_depth(ctx, *e, depth - 1, hasher);
                }
                cas_ast::Expr::Neg(e) => {
                    8u8.hash(hasher);
                    hash_expr_depth(ctx, *e, depth - 1, hasher);
                }
                cas_ast::Expr::Function(name, args) => {
                    9u8.hash(hasher);
                    name.hash(hasher);
                    args.len().hash(hasher);
                    for arg in args {
                        hash_expr_depth(ctx, *arg, depth - 1, hasher);
                    }
                }
                cas_ast::Expr::Matrix { rows, cols, data } => {
                    10u8.hash(hasher);
                    rows.hash(hasher);
                    cols.hash(hasher);
                    data.len().hash(hasher);
                    for elem in data {
                        hash_expr_depth(ctx, *elem, depth - 1, hasher);
                    }
                }
                cas_ast::Expr::SessionRef(id) => {
                    11u8.hash(hasher);
                    id.hash(hasher);
                }
                // Hold is transparent for hashing - hash inner like Neg
                cas_ast::Expr::Hold(e) => {
                    12u8.hash(hasher);
                    hash_expr_depth(ctx, *e, depth - 1, hasher);
                }
            }
        }

        let mut hasher = DefaultHasher::new();
        hash_expr_depth(ctx, expr, MAX_HASH_DEPTH, &mut hasher);
        hasher.finish()
    }
}
