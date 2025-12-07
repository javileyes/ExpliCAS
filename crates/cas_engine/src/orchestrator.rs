use crate::{Simplifier, Step};
use cas_ast::ExprId;
use num_traits::ToPrimitive;
use std::collections::{HashSet, VecDeque};

pub struct Orchestrator {
    // Configuration for the pipeline
    pub max_iterations: usize,
    pub enable_polynomial_strategy: bool,
    /// Rules that trigger multi-pass simplification
    /// These rules often create new opportunities for other rules
    cascade_triggers: HashSet<&'static str>,
    /// Pre-scanned pattern marks for context-aware guards
    pub pattern_marks: crate::pattern_marks::PatternMarks,
}

impl Orchestrator {
    pub fn new() -> Self {
        let mut cascade_triggers = HashSet::new();

        // Rules that create new simplification opportunities
        cascade_triggers.insert("Rationalize Denominator");
        cascade_triggers.insert("Expand Polynomial");
        cascade_triggers.insert("Factor");
        cascade_triggers.insert("Distribute");
        cascade_triggers.insert("Add fractions: a/b + c/d -> (ad+bc)/bd");

        Self {
            max_iterations: 10,
            enable_polynomial_strategy: true,
            cascade_triggers,
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
        eprintln!("📊 Orchestrator: Scanned expression, marks created");

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

        // 2. Rule-based Simplification - Conditional Multi-Pass
        // OPTIMIZATION: Only use multi-pass when "cascade trigger" rules fire
        // These rules create new opportunities that weren't visible before

        // First pass (always executed)
        let (first_pass, first_steps) = simplifier.apply_rules_loop(current, &self.pattern_marks);

        // Fast path: No changes at all (common for already-simplified expressions)
        if first_pass == current {
            current = first_pass;
        } else {
            // Check if any cascade trigger rule fired
            let needs_multi_pass = first_steps
                .iter()
                .any(|step| self.cascade_triggers.contains(step.rule_name.as_str()));

            if !needs_multi_pass {
                // FAST PATH: Single pass sufficient (95% of cases)
                // No cascade triggers, so no new opportunities created
                steps.extend(first_steps);
                current = first_pass;
            } else {
                // SLOW PATH: Multi-pass needed (5% of cases)
                // Cascade triggers fired, re-simplify to catch new opportunities
                steps.extend(first_steps);
                current = first_pass;

                let max_passes = 5;
                let mut pass_count = 1;
                let mut cycle_detector = CycleDetector::new(5);

                loop {
                    // 2. Local Simplification (Rule-based transformation)
                    let (simplified, mut local_steps) =
                        simplifier.local_simplify(current, &self.pattern_marks);

                    // OPTIMIZATION: Fast path with ExprId comparison (O(1))
                    if simplified == current {
                        break; // No changes, done
                    }

                    // Cycle detection
                    if let Some(_cycle_len) = cycle_detector.check(current) {
                        #[cfg(debug_assertions)]
                        eprintln!(
                            "WARNING: Simplification cycle detected (length {})",
                            _cycle_len
                        );
                        break;
                    }

                    steps.extend(local_steps);
                    current = simplified;
                    pass_count += 1;

                    // Safety: prevent runaway loops
                    if pass_count >= max_passes {
                        break;
                    }
                }
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

        // 6. Optimize Steps
        let optimized_steps = if simplifier.collect_steps {
            crate::step_optimization::optimize_steps(filtered_steps)
        } else {
            filtered_steps
        };

        (current, optimized_steps)
    }
}

/// Helper struct to detect cycles in simplification
/// Tracks recent expressions to detect if we're looping
struct CycleDetector {
    history: VecDeque<ExprId>,
    max_history: usize,
}

impl CycleDetector {
    fn new(max_history: usize) -> Self {
        Self {
            history: VecDeque::with_capacity(max_history),
            max_history,
        }
    }

    /// Check if expr has appeared before (indicating a cycle)
    /// Returns Some(cycle_length) if cycle detected
    fn check(&mut self, expr: ExprId) -> Option<usize> {
        if let Some(pos) = self.history.iter().position(|&e| e == expr) {
            return Some(self.history.len() - pos);
        }

        self.history.push_back(expr);
        if self.history.len() > self.max_history {
            self.history.pop_front();
        }

        None
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
