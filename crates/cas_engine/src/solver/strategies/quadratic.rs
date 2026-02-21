use crate::engine::Simplifier;
use crate::error::CasError;
use crate::ordering::compare_expr;
use crate::solver::isolation::contains_var;
use crate::solver::solution_set::{compare_values, neg_inf, pos_inf};
use crate::solver::solve_core::solve_with_ctx;
use crate::solver::strategy::SolverStrategy;
use crate::solver::{SolveCtx, SolveStep, SolverOptions};
use cas_ast::{BoundType, Context, Equation, Expr, ExprId, Interval, RelOp, SolutionSet};
use num_rational::BigRational;
use num_traits::{Signed, Zero};
use std::cmp::Ordering;

pub struct QuadraticStrategy;

impl SolverStrategy for QuadraticStrategy {
    fn name(&self) -> &str {
        "Quadratic Formula"
    }

    fn apply(
        &self,
        eq: &Equation,
        var: &str,
        simplifier: &mut Simplifier,
        _opts: &SolverOptions,
        ctx: &SolveCtx,
    ) -> Option<Result<(SolutionSet, Vec<SolveStep>), CasError>> {
        let mut steps = Vec::new();

        // Move everything to LHS: lhs - rhs = 0
        let poly_expr = simplifier.context.add(Expr::Sub(eq.lhs, eq.rhs));
        // Simplify first (this might factor it)
        let (sim_poly_expr, _) = simplifier.simplify(poly_expr);

        // Check for Zero Product Property: A * B = 0 => A = 0 or B = 0
        // Only if RHS is 0 (which it is, we moved everything to LHS)
        // We need to be careful not to recurse infinitely if A or B are not simpler.
        // But solving A=0 and B=0 breaks it down.

        let zero = simplifier.context.num(0);

        // Helper to check if we can split
        let split_factors = |ctx: &Context, expr: ExprId| -> Option<Vec<ExprId>> {
            match ctx.get(expr) {
                Expr::Mul(l, r) => Some(vec![*l, *r]), // We could flatten more
                Expr::Pow(b, e) => {
                    if let Expr::Number(n) = ctx.get(*e) {
                        if n.is_positive() {
                            return Some(vec![*b]);
                        }
                    }
                    None
                }
                _ => None,
            }
        };

        if let Some(factors) = split_factors(&simplifier.context, sim_poly_expr) {
            // We found factors.
            if simplifier.collect_steps() {
                steps.push(SolveStep {
                    description: format!(
                        "Factorized equation: {} = 0",
                        cas_formatter::DisplayExpr {
                            context: &simplifier.context,
                            id: sim_poly_expr
                        }
                    ),
                    equation_after: Equation {
                        lhs: sim_poly_expr,
                        rhs: zero,
                        op: RelOp::Eq,
                    },
                    importance: crate::step::ImportanceLevel::Medium,
                    substeps: vec![],
                });
            }

            // For inequalities, splitting is complex (sign analysis).
            // For Eq, it's simple union.
            if eq.op == RelOp::Eq {
                let mut all_solutions = Vec::new();
                for factor in factors {
                    // Skip factors that don't contain the variable (e.g., constant multipliers)
                    // This fixes cases like x/10000 = 5/10000 where simplified is (x-5)/10000 = Mul(1/10000, x-5)
                    // The 1/10000 factor has no variable, so we skip it
                    if !contains_var(&simplifier.context, factor, var) {
                        continue;
                    }

                    if simplifier.collect_steps() {
                        steps.push(SolveStep {
                            description: format!(
                                "Solve factor: {} = 0",
                                cas_formatter::DisplayExpr {
                                    context: &simplifier.context,
                                    id: factor
                                }
                            ),
                            equation_after: Equation {
                                lhs: factor,
                                rhs: zero,
                                op: RelOp::Eq,
                            },
                            importance: crate::step::ImportanceLevel::Medium,
                            substeps: vec![],
                        });
                    }
                    let factor_eq = Equation {
                        lhs: factor,
                        rhs: zero,
                        op: RelOp::Eq,
                    };
                    // Recursive solve
                    // We need to be careful about depth.
                    match solve_with_ctx(&factor_eq, var, simplifier, ctx) {
                        Ok((sol_set, mut sub_steps)) => {
                            steps.append(&mut sub_steps);
                            match sol_set {
                                SolutionSet::Discrete(sols) => all_solutions.extend(sols),
                                SolutionSet::Empty => {
                                    // No solutions from this factor — skip
                                }
                                SolutionSet::AllReals => {
                                    // Factor is identically zero → entire equation AllReals
                                    return Some(Ok((SolutionSet::AllReals, steps)));
                                }
                                _ => {
                                    // Residual/Conditional/Interval: can't extract discrete
                                    // roots. Return Residual for the whole equation rather
                                    // than crashing with SolverError.
                                    let residual =
                                        simplifier.context.add(Expr::Sub(eq.lhs, eq.rhs));
                                    let (sim, _) = simplifier.simplify(residual);
                                    return Some(Ok((SolutionSet::Residual(sim), steps)));
                                }
                            }
                        }
                        Err(e) => return Some(Err(e)),
                    }
                }
                // Remove duplicates
                all_solutions.sort_by(|a, b| compare_expr(&simplifier.context, *a, *b));
                all_solutions
                    .dedup_by(|a, b| compare_expr(&simplifier.context, *a, *b) == Ordering::Equal);

                return Some(Ok((SolutionSet::Discrete(all_solutions), steps)));
            }
        }

        // Ensure expanded form for coefficient extraction
        // QuadraticStrategy relies on A*x^2 + B*x + C structure (Add/Sub chain)
        let expanded_expr = crate::expand::expand(&mut simplifier.context, sim_poly_expr);

        if let Some((a, b, c)) = cas_solver_core::quadratic_coeffs::extract_quadratic_coefficients(
            &mut simplifier.context,
            expanded_expr,
            var,
        ) {
            // Check if it is actually quadratic (a != 0)
            // We need to simplify 'a' to check if it's zero.
            let (sim_a, _) = simplifier.simplify(a);
            let (sim_b, _) = simplifier.simplify(b);
            let (sim_c, _) = simplifier.simplify(c);

            // If a is zero, it's linear, not quadratic.
            // But if 'a' is symbolic, we might assume it's non-zero or return a conditional solution?
            // For now, if 'a' simplifies to explicit 0, we reject.
            if let Expr::Number(n) = simplifier.context.get(sim_a) {
                if n.is_zero() {
                    return None;
                }
            }

            if simplifier.collect_steps() {
                // Build didactic substeps showing completing-the-square derivation
                let is_real_only =
                    matches!(_opts.value_domain, crate::semantics::ValueDomain::RealOnly);
                let mut substeps = crate::solver::quadratic_steps::build_quadratic_substeps(
                    simplifier,
                    var,
                    sim_a,
                    sim_b,
                    sim_c,
                    is_real_only,
                );

                // Post-pass: apply didactic simplification to clean up expressions
                crate::solver::quadratic_steps::didactic_simplify_substeps(
                    simplifier,
                    &mut substeps,
                );

                // Main step with substeps attached
                let main_step = SolveStep {
                    description: "Detected quadratic equation. Applying quadratic formula."
                        .to_string(),
                    equation_after: Equation {
                        lhs: sim_poly_expr,
                        rhs: simplifier.context.num(0),
                        op: RelOp::Eq,
                    },
                    importance: crate::step::ImportanceLevel::Medium,
                    substeps,
                };
                steps.push(main_step);
            }

            // Check if coefficients are all numeric to support inequalities
            let a_num = crate::solver::solution_set::get_number(&simplifier.context, sim_a);
            let b_num = crate::solver::solution_set::get_number(&simplifier.context, sim_b);
            let c_num = crate::solver::solution_set::get_number(&simplifier.context, sim_c);

            if let (Some(a_val), Some(b_val), Some(c_val)) = (a_num, b_num, c_num) {
                // Use numeric logic for better inequality support
                // delta = b^2 - 4ac
                let b2 = b_val.clone() * b_val.clone();
                let four_ac = BigRational::from_integer(4.into()) * a_val.clone() * c_val;
                let delta = b2 - four_ac;

                // We need to return solutions in terms of Expr
                // x = (-b +/- sqrt(delta)) / 2a

                let neg_b = -b_val;
                let two_a = BigRational::from_integer(2.into()) * a_val.clone();

                let delta_expr = simplifier.context.add(Expr::Number(delta.clone()));
                let neg_b_expr = simplifier.context.add(Expr::Number(neg_b));
                let two_a_expr = simplifier.context.add(Expr::Number(two_a));

                // sqrt(delta)
                let one = simplifier.context.num(1);
                let two = simplifier.context.num(2);
                let half = simplifier.context.add(Expr::Div(one, two));
                let sqrt_delta = simplifier.context.add(Expr::Pow(delta_expr, half));

                // x1 = (-b - sqrt(delta)) / 2a (Smaller root if a > 0)
                let num1 = simplifier.context.add(Expr::Sub(neg_b_expr, sqrt_delta));
                let sol1 = simplifier.context.add(Expr::Div(num1, two_a_expr));

                // x2 = (-b + sqrt(delta)) / 2a (Larger root if a > 0)
                let num2 = simplifier.context.add(Expr::Add(neg_b_expr, sqrt_delta));
                let sol2 = simplifier.context.add(Expr::Div(num2, two_a_expr));

                let (sim_sol1, _) = simplifier.simplify(sol1);
                let (sim_sol2, _) = simplifier.simplify(sol2);

                // Ensure r1 <= r2
                let (r1, r2) = if compare_values(&simplifier.context, sim_sol1, sim_sol2)
                    == Ordering::Greater
                {
                    (sim_sol2, sim_sol1)
                } else {
                    (sim_sol1, sim_sol2)
                };

                // Determine parabola direction
                let opens_up = a_val > BigRational::zero();

                // Helper for intervals
                let mk_interval = |min, min_type, max, max_type| {
                    SolutionSet::Continuous(Interval {
                        min,
                        min_type,
                        max,
                        max_type,
                    })
                };

                let result = if delta > BigRational::zero() {
                    // Two distinct roots r1 < r2
                    match eq.op {
                        RelOp::Eq => SolutionSet::Discrete(vec![r1, r2]),
                        RelOp::Neq => {
                            // (-inf, r1) U (r1, r2) U (r2, inf)
                            let i1 = Interval {
                                min: neg_inf(&mut simplifier.context),
                                min_type: BoundType::Open,
                                max: r1,
                                max_type: BoundType::Open,
                            };
                            let i2 = Interval {
                                min: r1,
                                min_type: BoundType::Open,
                                max: r2,
                                max_type: BoundType::Open,
                            };
                            let i3 = Interval {
                                min: r2,
                                min_type: BoundType::Open,
                                max: pos_inf(&mut simplifier.context),
                                max_type: BoundType::Open,
                            };
                            SolutionSet::Union(vec![i1, i2, i3])
                        }
                        RelOp::Lt => {
                            if opens_up {
                                // Parabola < 0 between roots: (r1, r2)
                                mk_interval(r1, BoundType::Open, r2, BoundType::Open)
                            } else {
                                // Parabola < 0 outside roots: (-inf, r1) U (r2, inf)
                                let i1 = Interval {
                                    min: neg_inf(&mut simplifier.context),
                                    min_type: BoundType::Open,
                                    max: r1,
                                    max_type: BoundType::Open,
                                };
                                let i2 = Interval {
                                    min: r2,
                                    min_type: BoundType::Open,
                                    max: pos_inf(&mut simplifier.context),
                                    max_type: BoundType::Open,
                                };
                                SolutionSet::Union(vec![i1, i2])
                            }
                        }
                        RelOp::Leq => {
                            if opens_up {
                                // [r1, r2]
                                mk_interval(r1, BoundType::Closed, r2, BoundType::Closed)
                            } else {
                                // (-inf, r1] U [r2, inf)
                                let i1 = Interval {
                                    min: neg_inf(&mut simplifier.context),
                                    min_type: BoundType::Open,
                                    max: r1,
                                    max_type: BoundType::Closed,
                                };
                                let i2 = Interval {
                                    min: r2,
                                    min_type: BoundType::Closed,
                                    max: pos_inf(&mut simplifier.context),
                                    max_type: BoundType::Open,
                                };
                                SolutionSet::Union(vec![i1, i2])
                            }
                        }
                        RelOp::Gt => {
                            if opens_up {
                                // Parabola > 0 outside roots: (-inf, r1) U (r2, inf)
                                let i1 = Interval {
                                    min: neg_inf(&mut simplifier.context),
                                    min_type: BoundType::Open,
                                    max: r1,
                                    max_type: BoundType::Open,
                                };
                                let i2 = Interval {
                                    min: r2,
                                    min_type: BoundType::Open,
                                    max: pos_inf(&mut simplifier.context),
                                    max_type: BoundType::Open,
                                };
                                SolutionSet::Union(vec![i1, i2])
                            } else {
                                // Parabola > 0 between roots: (r1, r2)
                                mk_interval(r1, BoundType::Open, r2, BoundType::Open)
                            }
                        }
                        RelOp::Geq => {
                            if opens_up {
                                // (-inf, r1] U [r2, inf)
                                let i1 = Interval {
                                    min: neg_inf(&mut simplifier.context),
                                    min_type: BoundType::Open,
                                    max: r1,
                                    max_type: BoundType::Closed,
                                };
                                let i2 = Interval {
                                    min: r2,
                                    min_type: BoundType::Closed,
                                    max: pos_inf(&mut simplifier.context),
                                    max_type: BoundType::Open,
                                };
                                SolutionSet::Union(vec![i1, i2])
                            } else {
                                // [r1, r2]
                                mk_interval(r1, BoundType::Closed, r2, BoundType::Closed)
                            }
                        }
                    }
                } else if delta == BigRational::zero() {
                    // One repeated root r1
                    match eq.op {
                        RelOp::Eq => SolutionSet::Discrete(vec![r1]),
                        RelOp::Neq => {
                            // (-inf, r1) U (r1, inf)
                            let i1 = Interval {
                                min: neg_inf(&mut simplifier.context),
                                min_type: BoundType::Open,
                                max: r1,
                                max_type: BoundType::Open,
                            };
                            let i2 = Interval {
                                min: r1,
                                min_type: BoundType::Open,
                                max: pos_inf(&mut simplifier.context),
                                max_type: BoundType::Open,
                            };
                            SolutionSet::Union(vec![i1, i2])
                        }
                        RelOp::Lt => {
                            if opens_up {
                                // (x-r)^2 < 0 -> Empty
                                SolutionSet::Empty
                            } else {
                                // -(x-r)^2 < 0 -> All Reals except r
                                let i1 = Interval {
                                    min: neg_inf(&mut simplifier.context),
                                    min_type: BoundType::Open,
                                    max: r1,
                                    max_type: BoundType::Open,
                                };
                                let i2 = Interval {
                                    min: r1,
                                    min_type: BoundType::Open,
                                    max: pos_inf(&mut simplifier.context),
                                    max_type: BoundType::Open,
                                };
                                SolutionSet::Union(vec![i1, i2])
                            }
                        }
                        RelOp::Leq => {
                            if opens_up {
                                // (x-r)^2 <= 0 -> x = r
                                SolutionSet::Discrete(vec![r1])
                            } else {
                                // -(x-r)^2 <= 0 -> All Reals
                                SolutionSet::AllReals
                            }
                        }
                        RelOp::Gt => {
                            if opens_up {
                                // (x-r)^2 > 0 -> All Reals except r
                                let i1 = Interval {
                                    min: neg_inf(&mut simplifier.context),
                                    min_type: BoundType::Open,
                                    max: r1,
                                    max_type: BoundType::Open,
                                };
                                let i2 = Interval {
                                    min: r1,
                                    min_type: BoundType::Open,
                                    max: pos_inf(&mut simplifier.context),
                                    max_type: BoundType::Open,
                                };
                                SolutionSet::Union(vec![i1, i2])
                            } else {
                                // -(x-r)^2 > 0 -> Empty
                                SolutionSet::Empty
                            }
                        }
                        RelOp::Geq => {
                            if opens_up {
                                // (x-r)^2 >= 0 -> All Reals
                                SolutionSet::AllReals
                            } else {
                                // -(x-r)^2 >= 0 -> x = r
                                SolutionSet::Discrete(vec![r1])
                            }
                        }
                    }
                } else {
                    // delta < 0, no real roots
                    // Parabola is always positive (if a > 0) or always negative (if a < 0)
                    let always_pos = opens_up;
                    match eq.op {
                        RelOp::Eq => SolutionSet::Empty,
                        RelOp::Neq => SolutionSet::AllReals,
                        RelOp::Lt => {
                            if always_pos {
                                SolutionSet::Empty
                            } else {
                                SolutionSet::AllReals
                            }
                        }
                        RelOp::Leq => {
                            if always_pos {
                                SolutionSet::Empty
                            } else {
                                SolutionSet::AllReals
                            }
                        }
                        RelOp::Gt => {
                            if always_pos {
                                SolutionSet::AllReals
                            } else {
                                SolutionSet::Empty
                            }
                        }
                        RelOp::Geq => {
                            if always_pos {
                                SolutionSet::AllReals
                            } else {
                                SolutionSet::Empty
                            }
                        }
                    }
                };

                // Emit scope for display transforms (sqrt display in quadratic context)
                crate::solver::emit_scope(cas_formatter::display_transforms::ScopeTag::Rule(
                    "QuadraticFormula",
                ));

                return Some(Ok((result, steps)));
            }

            // Symbolic coefficients

            // delta = b^2 - 4ac
            let two_num = simplifier.context.num(2);
            let b2 = simplifier.context.add(Expr::Pow(sim_b, two_num));
            let four = simplifier.context.num(4);
            let four_a = simplifier.context.add(Expr::Mul(four, sim_a));
            let four_ac = simplifier.context.add(Expr::Mul(four_a, sim_c));
            let delta_raw = simplifier.context.add(Expr::Sub(b2, four_ac));

            // POST-SIMPLIFY: Expand then simplify discriminant for cleaner form
            // This converts "16 + 4*(y - 4)" → "4*y"
            let delta_expanded = crate::expand::expand(&mut simplifier.context, delta_raw);
            let (sim_delta, _) = simplifier.simplify(delta_expanded);

            // x = (-b +/- sqrt(delta)) / 2a

            let neg_b = simplifier.context.add(Expr::Neg(sim_b));
            let two = simplifier.context.num(2);
            let two_a = simplifier.context.add(Expr::Mul(two, sim_a));

            let one = simplifier.context.num(1);
            let half = simplifier.context.add(Expr::Div(one, two));
            let sqrt_delta_raw = simplifier.context.add(Expr::Pow(sim_delta, half));

            // POST-SIMPLIFY: Pull perfect square numeric factors from sqrt
            // This converts sqrt(4*y) → 2*sqrt(y)
            let sqrt_delta = cas_solver_core::quadratic_sqrt::pull_square_from_sqrt(
                &mut simplifier.context,
                sqrt_delta_raw,
            );

            // x1 = (-b - sqrt(delta)) / 2a
            let num1 = simplifier.context.add(Expr::Sub(neg_b, sqrt_delta));
            let sol1_raw = simplifier.context.add(Expr::Div(num1, two_a));

            // x2 = (-b + sqrt(delta)) / 2a
            let num2 = simplifier.context.add(Expr::Add(neg_b, sqrt_delta));
            let sol2_raw = simplifier.context.add(Expr::Div(num2, two_a));

            // POST-SIMPLIFY: Expand and simplify solutions for cleaner form
            // This converts "(4 ± 2*sqrt(y)) / 2" → "2 ± sqrt(y)"
            let sol1_expanded = crate::expand::expand(&mut simplifier.context, sol1_raw);
            let (sim_sol1, _) = simplifier.simplify(sol1_expanded);

            let sol2_expanded = crate::expand::expand(&mut simplifier.context, sol2_raw);
            let (sim_sol2, _) = simplifier.simplify(sol2_expanded);

            // For symbolic solutions, we can't easily order them or determine intervals.
            // We just return them as a discrete set.
            // TODO: Handle inequalities with symbolic coefficients (requires assumptions/cases).
            // For now, only support Eq.
            if eq.op != RelOp::Eq {
                return Some(Err(CasError::SolverError(
                    "Inequalities with symbolic coefficients not yet supported".to_string(),
                )));
            }

            // Emit scope for display transforms (sqrt display in quadratic context)
            crate::solver::emit_scope(cas_formatter::display_transforms::ScopeTag::Rule(
                "QuadraticFormula",
            ));

            return Some(Ok((SolutionSet::Discrete(vec![sim_sol1, sim_sol2]), steps)));
        }

        None
    }

    fn should_verify(&self) -> bool {
        false
    }
}
