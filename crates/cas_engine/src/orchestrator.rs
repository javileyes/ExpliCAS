use crate::{Simplifier, Step};
use cas_ast::ExprId;
use num_traits::ToPrimitive;
use std::collections::VecDeque;

pub struct Orchestrator {
    // Configuration for the pipeline
    pub max_iterations: usize,
    pub enable_polynomial_strategy: bool,
    /// Pre-scanned pattern marks for context-aware guards
    pub pattern_marks: crate::pattern_marks::PatternMarks,
}

impl Orchestrator {
    pub fn new() -> Self {
        Self {
            max_iterations: 10,
            enable_polynomial_strategy: true,
            pattern_marks: crate::pattern_marks::PatternMarks::new(),
        }
    }

    pub fn simplify(&mut self, expr: ExprId, simplifier: &mut Simplifier) -> (ExprId, Vec<Step>) {
        // PRE-ANALYSIS PASS: Scan entire tree and mark Pythagorean patterns
        // This allows context-aware guards to work correctly even with bottom-up processing
        self.pattern_marks = crate::pattern_marks::PatternMarks::new();
        crate::pattern_scanner::scan_and_mark_patterns(
            &simplifier.context,
            expr,
            &mut self.pattern_marks,
        );

        let mut steps = Vec::new();
        let mut current = expr;

        // 1. Initial Collection (Normalize)
        let collected = crate::collect::collect(&mut simplifier.context, current);
        if collected != current {
            // Only add step if structurally different (collect regenerates IDs)
            if crate::ordering::compare_expr(&simplifier.context, collected, current)
                != std::cmp::Ordering::Equal
            {
                if simplifier.collect_steps {
                    steps.push(Step::new(
                        "Initial Collection",
                        "Collect",
                        current,
                        collected,
                        Vec::new(),
                        Some(&simplifier.context),
                    ));
                }
            }
            current = collected;
        }

        // 2. Rule-based Simplification
        // Commutative operations are now AUTOMATICALLY canonicalized in Context::add()
        // so we no longer need a post-process canonicalization pass.

        let max_passes = 20; // Increased safely thanks to automatic canonicalization
        let mut pass_count = 0;
        let mut cycle_detector = CycleDetector::new(10);

        loop {
            // Apply one pass of simplification
            let (simplified, pass_steps) =
                simplifier.apply_rules_loop(current, &self.pattern_marks);

            // OPTIMIZATION: Fast path with ExprId comparison (O(1))
            if simplified == current {
                // Fixed point reached - no more changes
                break;
            }

            // Cycle detection (semantic, not just ExprId)
            if let Some(_cycle_len) = cycle_detector.check(&simplifier.context, current) {
                eprintln!(
                    "⚠️  CYCLE DETECTED at pass {} (length {})",
                    pass_count, _cycle_len
                );
                break;
            }

            steps.extend(pass_steps);
            current = simplified;
            pass_count += 1;

            // Safety: prevent runaway loops
            if pass_count >= max_passes {
                #[cfg(debug_assertions)]
                eprintln!(
                    "WARNING: Reached max simplification passes ({})",
                    max_passes
                );
                break;
            }
        }

        // 3. High-Level Strategies (Heuristics)
        // Try polynomial simplification (expand -> simplify -> factor)
        // This handles cases like (x-1)(x+1)... which need full expansion to simplify.
        if self.enable_polynomial_strategy {
            let skip_poly = should_skip_polynomial_strategy(&simplifier.context, current, 6, 4);

            if !skip_poly {
                let (poly_simplified, poly_steps) =
                    crate::strategies::simplify_polynomial(&mut simplifier.context, current);
                if poly_simplified != current {
                    // Only add steps if structurally different
                    if crate::ordering::compare_expr(&simplifier.context, poly_simplified, current)
                        != std::cmp::Ordering::Equal
                    {
                        if simplifier.collect_steps {
                            steps.extend(poly_steps);
                        }
                        current = poly_simplified;
                    }
                }
            }
        }

        // 4. Final Collection (Ensure canonical form)
        let final_collected = crate::collect::collect(&mut simplifier.context, current);
        if final_collected != current {
            if crate::ordering::compare_expr(&simplifier.context, final_collected, current)
                != std::cmp::Ordering::Equal
            {
                if simplifier.collect_steps {
                    steps.push(Step::new(
                        "Final Collection",
                        "Collect",
                        current,
                        final_collected,
                        Vec::new(),
                        Some(&simplifier.context),
                    ));
                }
            }
            current = final_collected;
        }

        // 5. Filter out non-productive steps (where global state doesn't change)
        let filtered_steps = if simplifier.collect_steps {
            crate::strategies::filter_non_productive_steps(&mut simplifier.context, expr, steps)
        } else {
            steps
        };

        // 6. Optimize Steps with semantic cycle detection
        let optimized_steps = if simplifier.collect_steps {
            match crate::step_optimization::optimize_steps_semantic(
                filtered_steps,
                &simplifier.context,
                expr,    // original expression
                current, // final expression
            ) {
                crate::step_optimization::StepOptimizationResult::Steps(steps) => steps,
                crate::step_optimization::StepOptimizationResult::NoSimplificationNeeded => vec![],
            }
        } else {
            filtered_steps
        };

        (current, optimized_steps)
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
            }
        }

        let mut hasher = DefaultHasher::new();
        hash_expr(ctx, expr, &mut hasher);
        hasher.finish()
    }
}

fn should_skip_polynomial_strategy(
    ctx: &cas_ast::Context,
    expr: ExprId,
    power_threshold: i32,
    div_threshold: usize,
) -> bool {
    use cas_ast::Expr;
    let mut stack = vec![expr];
    let mut div_count = 0;
    let mut has_add_sub = false;

    while let Some(id) = stack.pop() {
        match ctx.get(id) {
            Expr::Pow(b, e) => {
                if let Expr::Number(n) = ctx.get(*e) {
                    if let Some(val) = n.to_integer().to_i32() {
                        if val.abs() > power_threshold {
                            return true;
                        }
                    }
                }
                if let Expr::Pow(b_inner, _) = ctx.get(*b) {
                    stack.push(*b_inner);
                } else {
                    stack.push(*b);
                }
            }
            Expr::Div(l, r) => {
                div_count += 1;
                if div_count > div_threshold {
                    return true;
                }
                stack.push(*l);
                stack.push(*r);
            }
            Expr::Add(l, r) | Expr::Sub(l, r) => {
                has_add_sub = true;
                stack.push(*l);
                stack.push(*r);
            }
            Expr::Mul(l, r) => {
                stack.push(*l);
                stack.push(*r);
            }
            Expr::Neg(e) => stack.push(*e),
            Expr::Function(_, args) => stack.extend(args),
            _ => {}
        }
    }

    // If there are no additions or subtractions, polynomial simplification (expansion/factorization)
    // is likely unnecessary or redundant with other rules.
    if !has_add_sub {
        return true;
    }

    false
}
