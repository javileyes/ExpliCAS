use crate::phase::{SimplifyOptions, SimplifyPhase};
use crate::rationalize_policy::AutoRationalizeLevel;
use crate::{Simplifier, Step};
use cas_ast::ExprId;
use std::collections::VecDeque;

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
        let mut cycle_detector = CycleDetector::new(10);
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
            // Only scan if ExpandPolicy::Auto is enabled AND we're not in Solve mode
            // (Solve mode should never auto-expand to preserve structure)
            let auto_expand = self.options.expand_policy == crate::phase::ExpandPolicy::Auto;
            if auto_expand {
                crate::auto_expand_scan::mark_auto_expand_candidates(
                    &simplifier.context,
                    current,
                    &self.options.expand_budget,
                    &mut self.pattern_marks,
                );
            }
            let (next, steps) = simplifier.apply_rules_loop_with_phase_and_mode(
                current,
                &self.pattern_marks,
                phase,
                self.options.expand_mode,
                auto_expand,
                self.options.expand_budget,
            );

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

            // Cycle detection
            if cycle_detector.check(&simplifier.context, current).is_some() {
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

        // Check for specialized strategies first
        if let Some(result) =
            crate::telescoping::try_dirichlet_kernel_identity_pub(&simplifier.context, expr)
        {
            let zero = simplifier.context.num(0);
            let mut steps = Vec::new();
            if self.options.collect_steps {
                steps.push(Step::new(
                    &format!(
                        "Dirichlet Kernel Identity: 1 + 2Σcos(kx) = sin((n+½)x)/sin(x/2) for n={}",
                        result.n
                    ),
                    "Trig Summation Identity",
                    expr,
                    zero,
                    Vec::new(),
                    Some(&simplifier.context),
                ));
            }
            return (zero, steps, crate::phase::PipelineStats::default());
        }

        let mut current = expr;
        let mut all_steps = Vec::new();
        let mut pipeline_stats = crate::phase::PipelineStats::default();

        // Copy values to avoid borrow conflicts with &mut self in run_phase
        let budgets = self.options.budgets;
        let enable_transform = self.options.enable_transform;
        let auto_level = self.options.rationalize.auto_level;
        let collect_steps = self.options.collect_steps;

        // Phase 1: Core - Safe simplifications
        let (next, steps, stats) =
            self.run_phase(simplifier, current, SimplifyPhase::Core, budgets.core_iters);
        current = next;
        all_steps.extend(steps);
        pipeline_stats.core = stats;
        pipeline_stats.total_rewrites += pipeline_stats.core.rewrites_used;

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

        // Final collection for canonical form
        let final_collected = crate::collect::collect(&mut simplifier.context, current);
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

        (current, optimized_steps, pipeline_stats)
    }
}

/// Helper struct to detect cycles in simplification
/// Tracks recent expressions to detect if we're looping
struct CycleDetector {
    history: VecDeque<u64>, // Store hashes instead of ExprIds
    max_history: usize,
}

impl CycleDetector {
    fn new(max_history: usize) -> Self {
        Self {
            history: VecDeque::with_capacity(max_history),
            max_history,
        }
    }

    /// Check if expr has appeared before (based on semantic content)
    /// Returns Some(cycle_length) if cycle detected
    fn check(&mut self, ctx: &cas_ast::Context, expr: ExprId) -> Option<usize> {
        // Compute semantic hash of expression
        let hash = Self::semantic_hash(ctx, expr);

        if let Some(pos) = self.history.iter().position(|&h| h == hash) {
            return Some(self.history.len() - pos);
        }

        self.history.push_back(hash);
        if self.history.len() > self.max_history {
            self.history.pop_front();
        }

        None
    }

    /// Compute a hash based on the semantic structure of the expression
    /// This allows us to detect when expressions are equivalent even with different ExprIds
    fn semantic_hash(ctx: &cas_ast::Context, expr: ExprId) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        fn hash_expr(ctx: &cas_ast::Context, expr: ExprId, hasher: &mut DefaultHasher) {
            match ctx.get(expr) {
                cas_ast::Expr::Number(n) => {
                    0u8.hash(hasher);
                    n.to_string().hash(hasher); // Hash string representation for consistency
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
                    // Hash commutatively: sort the hashes to make a+b == b+a
                    let hash_l = CycleDetector::semantic_hash(ctx, *l);
                    let hash_r = CycleDetector::semantic_hash(ctx, *r);
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
                    hash_expr(ctx, *l, hasher);
                    hash_expr(ctx, *r, hasher);
                }
                cas_ast::Expr::Mul(l, r) => {
                    5u8.hash(hasher);
                    //  Hash commutatively: sort the hashes to make a*b == b*a
                    let hash_l = CycleDetector::semantic_hash(ctx, *l);
                    let hash_r = CycleDetector::semantic_hash(ctx, *r);
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
                    hash_expr(ctx, *l, hasher);
                    hash_expr(ctx, *r, hasher);
                }
                cas_ast::Expr::Pow(b, e) => {
                    7u8.hash(hasher);
                    hash_expr(ctx, *b, hasher);
                    hash_expr(ctx, *e, hasher);
                }
                cas_ast::Expr::Neg(e) => {
                    8u8.hash(hasher);
                    hash_expr(ctx, *e, hasher);
                }
                cas_ast::Expr::Function(name, args) => {
                    9u8.hash(hasher);
                    name.hash(hasher);
                    args.len().hash(hasher);
                    for arg in args {
                        hash_expr(ctx, *arg, hasher);
                    }
                }
                cas_ast::Expr::Matrix { rows, cols, data } => {
                    10u8.hash(hasher);
                    rows.hash(hasher);
                    cols.hash(hasher);
                    data.len().hash(hasher);
                    for elem in data {
                        hash_expr(ctx, *elem, hasher);
                    }
                }
                cas_ast::Expr::SessionRef(id) => {
                    11u8.hash(hasher);
                    id.hash(hasher);
                }
            }
        }

        let mut hasher = DefaultHasher::new();
        hash_expr(ctx, expr, &mut hasher);
        hasher.finish()
    }
}
