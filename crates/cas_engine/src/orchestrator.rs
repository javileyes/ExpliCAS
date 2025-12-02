use cas_ast::ExprId;
use crate::{Simplifier, Step};
use num_traits::{ToPrimitive, Signed};

pub struct Orchestrator {
    // Configuration for the pipeline
    pub max_iterations: usize,
    pub enable_polynomial_strategy: bool,
}

impl Orchestrator {
    pub fn new() -> Self {
        Self {
            max_iterations: 10,
            enable_polynomial_strategy: true,
        }
    }

    pub fn simplify(&self, expr: ExprId, simplifier: &mut Simplifier) -> (ExprId, Vec<Step>) {
        let mut steps = Vec::new();
        let mut current = expr;
        
        // 1. Initial Collection (Normalize)
        let collected = crate::collect::collect(&mut simplifier.context, current);
        if collected != current {
            // Only add step if structurally different (collect regenerates IDs)
            if crate::ordering::compare_expr(&simplifier.context, collected, current) != std::cmp::Ordering::Equal {
                if simplifier.collect_steps {
                    steps.push(Step::new(
                        "Initial Collection",
                        "Collect",
                        current,
                        collected,
                        Vec::new(),
                    ));
                }
            }
            current = collected;
        }

        // 2. Rule-based Simplification Loop
        let (simplified, rule_steps) = simplifier.apply_rules_loop(current);
        steps.extend(rule_steps);
        current = simplified;

        // 3. High-Level Strategies (Heuristics)
        // Try polynomial simplification (expand -> simplify -> factor)
        // This handles cases like (x-1)(x+1)... which need full expansion to simplify.
        if self.enable_polynomial_strategy {
            let skip_poly = should_skip_polynomial_strategy(&simplifier.context, current, 6, 4);
            
            if !skip_poly {
                let poly_simplified = crate::strategies::simplify_polynomial(&mut simplifier.context, current);
                if poly_simplified != current {
                    // Only add step if structurally different
                    if crate::ordering::compare_expr(&simplifier.context, poly_simplified, current) != std::cmp::Ordering::Equal {
                         if simplifier.collect_steps {
                            steps.push(Step::new(
                                "Polynomial Strategy",
                                "Simplify Polynomial",
                                current,
                                poly_simplified,
                                Vec::new(),
                            ));
                        }
                        current = poly_simplified;
                    }
                }
            }
        }

        // 3. Final Collection (Ensure canonical form)
        let final_collected = crate::collect::collect(&mut simplifier.context, current);
        if final_collected != current {
            if crate::ordering::compare_expr(&simplifier.context, final_collected, current) != std::cmp::Ordering::Equal {
                if simplifier.collect_steps {
                    steps.push(Step::new(
                        "Final Collection",
                        "Collect",
                        current,
                        final_collected,
                        Vec::new(),
                    ));
                }
            }
            current = final_collected;
        }

        // 4. Optimize Steps
        let optimized_steps = if simplifier.collect_steps {
            crate::step_optimization::optimize_steps(steps)
        } else {
            steps
        };

        (current, optimized_steps)
    }
}

fn should_skip_polynomial_strategy(ctx: &cas_ast::Context, expr: ExprId, power_threshold: i32, div_threshold: usize) -> bool {
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
            },
            Expr::Div(l, r) => {
                div_count += 1;
                if div_count > div_threshold {
                    return true;
                }
                stack.push(*l);
                stack.push(*r);
            },
            Expr::Add(l, r) | Expr::Sub(l, r) => {
                has_add_sub = true;
                stack.push(*l);
                stack.push(*r);
            },
            Expr::Mul(l, r) => {
                stack.push(*l);
                stack.push(*r);
            },
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
