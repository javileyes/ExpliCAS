use crate::build::mul2_raw;
use crate::engine::Simplifier;
use crate::error::CasError;
use crate::ordering::compare_expr;
use crate::solver::isolation::{contains_var, isolate};
use crate::solver::solution_set::{compare_values, neg_inf, pos_inf};
use crate::solver::solve; // Needed for recursive solve in substitution
use crate::solver::strategy::SolverStrategy;
use crate::solver::{SolveStep, SolverOptions};
use cas_ast::{BoundType, Context, Equation, Expr, ExprId, Interval, RelOp, SolutionSet};
use num_rational::BigRational;
use num_traits::{Signed, Zero};
use std::cmp::Ordering;

// --- Helper Functions (Keep these as they are useful helpers) ---

pub fn detect_substitution(ctx: &mut Context, eq: &Equation, var: &str) -> Option<ExprId> {
    // ... (Keep existing implementation)
    // Heuristic: Look for e^x.
    // Collect all Pow(e, ...) terms.
    let terms = collect_exponential_terms(ctx, eq.lhs, var);
    if terms.is_empty() {
        return None;
    }

    // If we found e^x, return it.
    // For e^(2x) - 3e^x + 2, we find e^(2x) and e^x.
    // We want the "base" unit, which is e^x.
    // Simple heuristic: pick the one with the simplest exponent (e.g. x).

    let mut found_complex = false;
    let mut base_term = None;

    for term in &terms {
        if let Expr::Pow(_, exp) = ctx.get(*term) {
            // Check if exponent is exactly var (e^x)
            if let Expr::Variable(v) = ctx.get(*exp) {
                if v == var {
                    base_term = Some(*term);
                }
            } else {
                // Check if exponent is k*var (e^2x)
                // If so, it's complex.
                if contains_var(ctx, *exp, var) {
                    found_complex = true;
                }
            }
        }
    }

    if found_complex {
        // If we found e^(2x), we need e^x as base.
        // If e^x was found, return it.
        // If not, maybe we need to infer it?
        // For now, assume e^x is present or implied.
        // If e^x is not present explicitly (e.g. e^2x = 1), we can still substitute u=e^x.
        // But we need to construct it.
        if let Some(base) = base_term {
            return Some(base);
        } else {
            // Construct e^x
            // We need to know the base.
            let base_id = if let Expr::Pow(b, _) = ctx.get(terms[0]) {
                Some(*b)
            } else {
                None
            };

            if let Some(b) = base_id {
                let var_expr = ctx.var(var);
                return Some(ctx.add(Expr::Pow(b, var_expr)));
            }
        }
    }

    None
}

pub fn collect_exponential_terms(ctx: &Context, expr: ExprId, var: &str) -> Vec<ExprId> {
    let mut terms = Vec::new();
    match ctx.get(expr) {
        Expr::Pow(b, e) => {
            let is_e = match ctx.get(*b) {
                Expr::Variable(name) => name == "e",
                Expr::Constant(c) => matches!(c, cas_ast::Constant::E),
                _ => false,
            };
            if is_e && contains_var(ctx, *e, var) {
                terms.push(expr);
            }
        }
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) => {
            terms.extend(collect_exponential_terms(ctx, *l, var));
            terms.extend(collect_exponential_terms(ctx, *r, var));
        }
        _ => {}
    }
    terms
}

pub fn substitute_expr(
    ctx: &mut Context,
    expr: ExprId,
    target: ExprId,
    replacement: ExprId,
) -> ExprId {
    if compare_expr(ctx, expr, target) == Ordering::Equal {
        return replacement;
    }

    let expr_data = ctx.get(expr).clone();

    // Handle e^(2x) -> (e^x)^2 -> u^2
    let target_pow = if let Expr::Pow(tb, te) = ctx.get(target) {
        Some((*tb, *te))
    } else {
        None
    };

    if let Expr::Pow(b, e) = &expr_data {
        if let Some((tb, te)) = target_pow {
            if compare_expr(ctx, *b, tb) == Ordering::Equal {
                // Check if e = k * te
                let e_mul = if let Expr::Mul(l, r) = ctx.get(*e) {
                    Some((*l, *r))
                } else {
                    None
                };

                if let Some((l, r)) = e_mul {
                    let l_is_num = matches!(ctx.get(l), Expr::Number(_));
                    let r_is_num = matches!(ctx.get(r), Expr::Number(_));

                    let l_matches = compare_expr(ctx, l, te) == Ordering::Equal;
                    let r_matches = compare_expr(ctx, r, te) == Ordering::Equal;

                    if (l_matches && r_is_num) || (r_matches && l_is_num) {
                        let coeff = if l_matches { r } else { l };
                        return ctx.add(Expr::Pow(replacement, coeff));
                    }
                }
            }
        }
    }

    match expr_data {
        Expr::Add(l, r) => {
            let new_l = substitute_expr(ctx, l, target, replacement);
            let new_r = substitute_expr(ctx, r, target, replacement);
            if new_l != l || new_r != r {
                return ctx.add(Expr::Add(new_l, new_r));
            }
        }
        Expr::Sub(l, r) => {
            let new_l = substitute_expr(ctx, l, target, replacement);
            let new_r = substitute_expr(ctx, r, target, replacement);
            if new_l != l || new_r != r {
                return ctx.add(Expr::Sub(new_l, new_r));
            }
        }
        Expr::Mul(l, r) => {
            let new_l = substitute_expr(ctx, l, target, replacement);
            let new_r = substitute_expr(ctx, r, target, replacement);
            if new_l != l || new_r != r {
                return mul2_raw(ctx, new_l, new_r);
            }
        }
        Expr::Div(l, r) => {
            let new_l = substitute_expr(ctx, l, target, replacement);
            let new_r = substitute_expr(ctx, r, target, replacement);
            if new_l != l || new_r != r {
                return ctx.add(Expr::Div(new_l, new_r));
            }
        }
        Expr::Pow(b, e) => {
            let new_b = substitute_expr(ctx, b, target, replacement);
            let new_e = substitute_expr(ctx, e, target, replacement);
            if new_b != b || new_e != e {
                return ctx.add(Expr::Pow(new_b, new_e));
            }
        }
        Expr::Neg(e) => {
            let new_e = substitute_expr(ctx, e, target, replacement);
            if new_e != e {
                return ctx.add(Expr::Neg(new_e));
            }
        }
        Expr::Function(name, args) => {
            let mut new_args = Vec::new();
            let mut changed = false;
            for arg in args {
                let new_arg = substitute_expr(ctx, arg, target, replacement);
                if new_arg != arg {
                    changed = true;
                }
                new_args.push(new_arg);
            }
            if changed {
                return ctx.add(Expr::Function(name, new_args));
            }
        }
        _ => {}
    }

    expr
}

// --- Strategies ---

pub struct SubstitutionStrategy;

impl SolverStrategy for SubstitutionStrategy {
    fn name(&self) -> &str {
        "Substitution"
    }

    fn apply(
        &self,
        eq: &Equation,
        var: &str,
        simplifier: &mut Simplifier,
        _opts: &SolverOptions,
    ) -> Option<Result<(SolutionSet, Vec<SolveStep>), CasError>> {
        if let Some(sub_var_expr) = detect_substitution(&mut simplifier.context, eq, var) {
            let mut steps = Vec::new();
            if simplifier.collect_steps() {
                steps.push(SolveStep {
                    description: format!("Detected substitution: u = {:?}", sub_var_expr), // Debug format
                    equation_after: eq.clone(),
                });
            }

            // Rewrite equation in terms of u
            let u_sym = "u";
            let u_var = simplifier.context.var(u_sym);
            let new_lhs = substitute_expr(&mut simplifier.context, eq.lhs, sub_var_expr, u_var);
            let new_rhs = substitute_expr(&mut simplifier.context, eq.rhs, sub_var_expr, u_var);

            let new_eq = Equation {
                lhs: new_lhs,
                rhs: new_rhs,
                op: eq.op.clone(),
            };

            if simplifier.collect_steps() {
                steps.push(SolveStep {
                    description: format!(
                        "Substituted equation: {:?} {} {:?}",
                        new_eq.lhs, new_eq.op, new_eq.rhs
                    ),
                    equation_after: new_eq.clone(),
                });
            }

            // Solve for u
            let (u_solutions, mut u_steps) = match solve(&new_eq, u_sym, simplifier) {
                Ok(res) => res,
                Err(e) => return Some(Err(e)),
            };
            steps.append(&mut u_steps);

            // eprintln!("Substitution u_solutions: {:?}", u_solutions);

            // Now solve u = val for each solution
            match u_solutions {
                SolutionSet::Discrete(vals) => {
                    let mut final_solutions = Vec::new();
                    for val in vals {
                        // Solve sub_var_expr = val
                        let sub_eq = Equation {
                            lhs: sub_var_expr,
                            rhs: val,
                            op: RelOp::Eq,
                        };
                        if simplifier.collect_steps() {
                            steps.push(SolveStep {
                                description: format!(
                                    "Back-substitute: {:?} = {:?}",
                                    sub_var_expr, val
                                ),
                                equation_after: sub_eq.clone(),
                            });
                        }
                        let (x_sol, mut x_steps) = match solve(&sub_eq, var, simplifier) {
                            Ok(res) => res,
                            Err(e) => return Some(Err(e)),
                        };
                        // eprintln!("Back-substitute result for val {:?}: {:?}", val, x_sol);
                        steps.append(&mut x_steps);

                        if let SolutionSet::Discrete(xs) = x_sol {
                            final_solutions.extend(xs);
                        }
                    }
                    return Some(Ok((SolutionSet::Discrete(final_solutions), steps)));
                }
                _ => {
                    // Handle intervals? Too complex for now.
                    return Some(Err(CasError::SolverError(
                        "Substitution strategy currently only supports discrete solutions"
                            .to_string(),
                    )));
                }
            }
        }
        None
    }
}

// Revised helper with mutable context
fn extract_quadratic_coefficients_impl(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<(ExprId, ExprId, ExprId)> {
    let zero = ctx.num(0);
    let mut a = zero;
    let mut b = zero;
    let mut c = zero;

    let mut stack = vec![(expr, true)]; // (expr, is_positive)

    while let Some((curr, pos)) = stack.pop() {
        // We need to inspect `curr` without holding a borrow on `ctx` for too long
        let curr_data = ctx.get(curr).clone();

        match curr_data {
            Expr::Add(l, r) => {
                stack.push((r, pos));
                stack.push((l, pos));
            }
            Expr::Sub(l, r) => {
                stack.push((r, !pos));
                stack.push((l, pos));
            }
            _ => {
                // Analyze term
                let (coeff, degree) = analyze_term_mut(ctx, curr, var)?;

                let term_val = if pos {
                    coeff
                } else {
                    ctx.add(Expr::Neg(coeff))
                };

                match degree {
                    2 => a = ctx.add(Expr::Add(a, term_val)),
                    1 => b = ctx.add(Expr::Add(b, term_val)),
                    0 => c = ctx.add(Expr::Add(c, term_val)),
                    _ => return None,
                }
            }
        }
    }

    // eprintln!("Extracted coeffs for {}: a={:?}, b={:?}, c={:?}", var, ctx.get(a), ctx.get(b), ctx.get(c));
    Some((a, b, c))
}

fn analyze_term_mut(ctx: &mut Context, term: ExprId, var: &str) -> Option<(ExprId, i32)> {
    if !contains_var(ctx, term, var) {
        return Some((term, 0));
    }

    let term_data = ctx.get(term).clone();

    match term_data {
        Expr::Variable(v) if v == var => Some((ctx.num(1), 1)),
        Expr::Pow(base, exp) => {
            if let Expr::Variable(v) = ctx.get(base) {
                if v == var && !contains_var(ctx, exp, var) {
                    let degree = if let Expr::Number(n) = ctx.get(exp) {
                        if n.is_integer() {
                            Some(n.to_integer())
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                    if let Some(d) = degree {
                        return Some((ctx.num(1), d.try_into().ok()?));
                    }
                }
            }
            None
        }
        Expr::Mul(l, r) => {
            let l_has = contains_var(ctx, l, var);
            let r_has = contains_var(ctx, r, var);

            if l_has && r_has {
                let (c1, d1) = analyze_term_mut(ctx, l, var)?;
                let (c2, d2) = analyze_term_mut(ctx, r, var)?;
                let new_coeff = mul2_raw(ctx, c1, c2);
                Some((new_coeff, d1 + d2))
            } else if l_has {
                let (c, d) = analyze_term_mut(ctx, l, var)?;
                let new_coeff = mul2_raw(ctx, c, r);
                Some((new_coeff, d))
            } else if r_has {
                let (c, d) = analyze_term_mut(ctx, r, var)?;
                let new_coeff = mul2_raw(ctx, l, c);
                Some((new_coeff, d))
            } else {
                Some((term, 0))
            }
        }
        Expr::Div(l, r) => {
            if contains_var(ctx, r, var) {
                return None;
            }
            let (c, d) = analyze_term_mut(ctx, l, var)?;
            let new_coeff = ctx.add(Expr::Div(c, r));
            Some((new_coeff, d))
        }
        Expr::Neg(inner) => {
            let (c, d) = analyze_term_mut(ctx, inner, var)?;
            let new_coeff = ctx.add(Expr::Neg(c));
            Some((new_coeff, d))
        }
        _ => None,
    }
}

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
                        cas_ast::DisplayExpr {
                            context: &simplifier.context,
                            id: sim_poly_expr
                        }
                    ),
                    equation_after: Equation {
                        lhs: sim_poly_expr,
                        rhs: zero,
                        op: RelOp::Eq,
                    },
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
                                cas_ast::DisplayExpr {
                                    context: &simplifier.context,
                                    id: factor
                                }
                            ),
                            equation_after: Equation {
                                lhs: factor,
                                rhs: zero,
                                op: RelOp::Eq,
                            },
                        });
                    }
                    let factor_eq = Equation {
                        lhs: factor,
                        rhs: zero,
                        op: RelOp::Eq,
                    };
                    // Recursive solve
                    // We need to be careful about depth.
                    match solve(&factor_eq, var, simplifier) {
                        Ok((sol_set, mut sub_steps)) => {
                            steps.append(&mut sub_steps);
                            match sol_set {
                                SolutionSet::Discrete(sols) => all_solutions.extend(sols),
                                _ => {
                                    return Some(Err(CasError::SolverError(
                                        "Continuous solution in factor split not supported yet"
                                            .to_string(),
                                    )))
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

        if let Some((a, b, c)) =
            extract_quadratic_coefficients_impl(&mut simplifier.context, expanded_expr, var)
        {
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
                steps.push(SolveStep {
                    description: "Detected quadratic equation. Applying quadratic formula."
                        .to_string(),
                    equation_after: Equation {
                        lhs: sim_poly_expr,
                        rhs: simplifier.context.num(0),
                        op: RelOp::Eq,
                    },
                });
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
                crate::solver::emit_scope(cas_ast::display_transforms::ScopeTag::Rule(
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
            let delta = simplifier.context.add(Expr::Sub(b2, four_ac));

            let (sim_delta, _) = simplifier.simplify(delta);

            // x = (-b +/- sqrt(delta)) / 2a

            let neg_b = simplifier.context.add(Expr::Neg(sim_b));
            let two = simplifier.context.num(2);
            let two_a = simplifier.context.add(Expr::Mul(two, sim_a));

            let one = simplifier.context.num(1);
            let half = simplifier.context.add(Expr::Div(one, two));
            let sqrt_delta = simplifier.context.add(Expr::Pow(sim_delta, half));

            // x1 = (-b - sqrt(delta)) / 2a
            let num1 = simplifier.context.add(Expr::Sub(neg_b, sqrt_delta));
            let sol1 = simplifier.context.add(Expr::Div(num1, two_a));

            // x2 = (-b + sqrt(delta)) / 2a
            let num2 = simplifier.context.add(Expr::Add(neg_b, sqrt_delta));
            let sol2 = simplifier.context.add(Expr::Div(num2, two_a));

            let (sim_sol1, _) = simplifier.simplify(sol1);
            let (sim_sol2, _) = simplifier.simplify(sol2);

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
            crate::solver::emit_scope(cas_ast::display_transforms::ScopeTag::Rule(
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

pub struct IsolationStrategy;

impl SolverStrategy for IsolationStrategy {
    fn name(&self) -> &str {
        "Isolation"
    }

    fn apply(
        &self,
        eq: &Equation,
        var: &str,
        simplifier: &mut Simplifier,
        opts: &SolverOptions,
    ) -> Option<Result<(SolutionSet, Vec<SolveStep>), CasError>> {
        // Isolation strategy expects variable on LHS.
        // The main solve loop handles swapping, but we should check here or just assume?
        // Let's check and swap if needed, or just rely on isolate to handle it?
        // isolate() assumes we are isolating FROM lhs.

        // If var is on RHS and not LHS, we should swap.
        // If var is on both, isolation might fail or we need to collect first.

        let lhs_has = contains_var(&simplifier.context, eq.lhs, var);
        let rhs_has = contains_var(&simplifier.context, eq.rhs, var);

        if !lhs_has && !rhs_has {
            return Some(Err(CasError::VariableNotFound(var.to_string())));
        }

        if lhs_has && rhs_has {
            // Isolation cannot handle var on both sides directly without collection
            return None; // Or error? Strategy doesn't apply if not isolated.
        }

        if !lhs_has && rhs_has {
            // Swap
            let new_op = match eq.op {
                RelOp::Eq => RelOp::Eq,
                RelOp::Neq => RelOp::Neq,
                RelOp::Lt => RelOp::Gt,
                RelOp::Gt => RelOp::Lt,
                RelOp::Leq => RelOp::Geq,
                RelOp::Geq => RelOp::Leq,
            };
            let mut steps = Vec::new();
            if simplifier.collect_steps() {
                steps.push(SolveStep {
                    description: "Swap sides to put variable on LHS".to_string(),
                    equation_after: Equation {
                        lhs: eq.rhs,
                        rhs: eq.lhs,
                        op: new_op.clone(),
                    },
                });
            }
            // V2.0: Pass opts through to propagate budget
            match isolate(eq.rhs, eq.lhs, new_op, var, simplifier, *opts) {
                Ok((set, mut iso_steps)) => {
                    steps.append(&mut iso_steps);
                    return Some(Ok((set, steps)));
                }
                Err(e) => return Some(Err(e)),
            }
        }

        // LHS has var
        // V2.0: Pass opts through to propagate budget
        match isolate(eq.lhs, eq.rhs, eq.op.clone(), var, simplifier, *opts) {
            Ok((set, steps)) => Some(Ok((set, steps))),
            Err(e) => Some(Err(e)),
        }
    }

    // Note: We use the default should_verify() = true here.
    // Selective verification in solve() handles symbolic solutions.
}

/// Check if an exponential equation needs complex logarithm in Wildcard mode.
/// Returns Some(Ok(Residual)) if Wildcard mode should return a residual.
/// Returns Some(Err) if an error should be returned.
/// Returns None if this case doesn't apply (normal processing should continue).
fn check_exponential_needs_complex(
    eq: &Equation,
    var: &str,
    simplifier: &mut Simplifier,
    opts: &SolverOptions,
    lhs_has: bool,
    rhs_has: bool,
) -> Option<Result<(SolutionSet, Vec<SolveStep>), CasError>> {
    use crate::domain::DomainMode;
    use crate::semantics::AssumeScope;
    use crate::solver::domain_guards::{classify_log_solve, LogSolveDecision};

    // Check LHS for exponential a^x pattern
    if lhs_has && !rhs_has {
        if let Expr::Pow(base, exp) = simplifier.context.get(eq.lhs).clone() {
            // Check if exponent contains var and base doesn't
            if contains_var(&simplifier.context, exp, var)
                && !contains_var(&simplifier.context, base, var)
            {
                let decision = classify_log_solve(&simplifier.context, base, eq.rhs, opts);

                if let LogSolveDecision::NeedsComplex(msg) = decision {
                    // Check if we're in Wildcard mode
                    if opts.domain_mode == DomainMode::Assume
                        && opts.assume_scope == AssumeScope::Wildcard
                    {
                        // Create a solve(eq, var) residual
                        let eq_expr = create_equation_expr(simplifier, eq);
                        let var_expr = simplifier.context.var(var);
                        let residual = simplifier
                            .context
                            .add(Expr::Function("solve".to_string(), vec![eq_expr, var_expr]));

                        // Create step with warning
                        let mut steps = Vec::new();
                        if simplifier.collect_steps() {
                            steps.push(SolveStep {
                                description: format!("{} - use 'semantics preset complex'", msg),
                                equation_after: eq.clone(),
                            });
                        }

                        return Some(Ok((SolutionSet::Residual(residual), steps)));
                    }
                    // If not Wildcard, let other handlers deal with it
                }
            }
        }
    }

    // Check RHS for exponential pattern (symmetric case)
    if rhs_has && !lhs_has {
        if let Expr::Pow(base, exp) = simplifier.context.get(eq.rhs).clone() {
            if contains_var(&simplifier.context, exp, var)
                && !contains_var(&simplifier.context, base, var)
            {
                let decision = classify_log_solve(&simplifier.context, base, eq.lhs, opts);

                if let LogSolveDecision::NeedsComplex(msg) = decision {
                    if opts.domain_mode == DomainMode::Assume
                        && opts.assume_scope == AssumeScope::Wildcard
                    {
                        let eq_expr = create_equation_expr(simplifier, eq);
                        let var_expr = simplifier.context.var(var);
                        let residual = simplifier
                            .context
                            .add(Expr::Function("solve".to_string(), vec![eq_expr, var_expr]));

                        let mut steps = Vec::new();
                        if simplifier.collect_steps() {
                            steps.push(SolveStep {
                                description: format!("{} - use 'semantics preset complex'", msg),
                                equation_after: eq.clone(),
                            });
                        }

                        return Some(Ok((SolutionSet::Residual(residual), steps)));
                    }
                }
            }
        }
    }

    None
}

/// Create an expression representing the equation for residual notation
fn create_equation_expr(simplifier: &mut Simplifier, eq: &Equation) -> ExprId {
    // We represent eq as Function("__eq__", [lhs, rhs])
    // This is just for internal residual representation
    simplifier
        .context
        .add(Expr::Function("__eq__".to_string(), vec![eq.lhs, eq.rhs]))
}

pub struct UnwrapStrategy;

impl SolverStrategy for UnwrapStrategy {
    fn name(&self) -> &str {
        "Unwrap"
    }

    fn apply(
        &self,
        eq: &Equation,
        var: &str,
        simplifier: &mut Simplifier,
        opts: &SolverOptions,
    ) -> Option<Result<(SolutionSet, Vec<SolveStep>), CasError>> {
        // Try to unwrap functions on LHS or RHS to expose the variable or transform the equation.
        // This is useful when var is on both sides, e.g. sqrt(2x+3) = x.

        let lhs_has = contains_var(&simplifier.context, eq.lhs, var);
        let rhs_has = contains_var(&simplifier.context, eq.rhs, var);

        // Only apply if var is on both sides?
        // If var is only on one side, IsolationStrategy handles it.
        // But IsolationStrategy might be later in the list.
        // Let's apply if top-level is a function/pow that we can invert.

        if !lhs_has && !rhs_has {
            return None;
        }

        // EARLY CHECK: Handle exponential NeedsComplex + Wildcard -> Residual
        // This must be before the closure to be able to return SolutionSet::Residual
        if let Some(result) =
            check_exponential_needs_complex(eq, var, simplifier, opts, lhs_has, rhs_has)
        {
            return Some(result);
        }

        // Helper to invert
        let mut invert = |target: ExprId,
                          other: ExprId,
                          op: RelOp,
                          is_lhs: bool|
         -> Option<(Equation, String)> {
            let target_data = simplifier.context.get(target).clone();
            match target_data {
                Expr::Function(name, args) if args.len() == 1 => {
                    let arg = args[0];
                    match name.as_str() {
                        "sqrt" => {
                            // sqrt(A) = B -> A = B^2
                            // Check domain? sqrt(A) >= 0. So B must be >= 0.
                            // We should add a constraint or verify later.
                            // For now, just transform. Verification step in solve() handles extraneous roots.
                            let two = simplifier.context.num(2);
                            let new_other = simplifier.context.add(Expr::Pow(other, two));
                            let new_eq = if is_lhs {
                                Equation {
                                    lhs: arg,
                                    rhs: new_other,
                                    op,
                                }
                            } else {
                                Equation {
                                    lhs: new_other,
                                    rhs: arg,
                                    op,
                                }
                            };
                            Some((new_eq, "Square both sides".to_string()))
                        }
                        "ln" => {
                            // ln(A) = B -> A = e^B
                            let e = simplifier.context.add(Expr::Constant(cas_ast::Constant::E));
                            let new_other = simplifier.context.add(Expr::Pow(e, other));
                            let new_eq = if is_lhs {
                                Equation {
                                    lhs: arg,
                                    rhs: new_other,
                                    op,
                                }
                            } else {
                                Equation {
                                    lhs: new_other,
                                    rhs: arg,
                                    op,
                                }
                            };
                            Some((new_eq, "Exponentiate (base e)".to_string()))
                        }
                        "exp" => {
                            // exp(A) = B -> A = ln(B)
                            let new_other = simplifier
                                .context
                                .add(Expr::Function("ln".to_string(), vec![other]));
                            let new_eq = if is_lhs {
                                Equation {
                                    lhs: arg,
                                    rhs: new_other,
                                    op,
                                }
                            } else {
                                Equation {
                                    lhs: new_other,
                                    rhs: arg,
                                    op,
                                }
                            };
                            Some((new_eq, "Take natural log".to_string()))
                        }
                        _ => None,
                    }
                }
                Expr::Pow(b, e) => {
                    // A^n = B -> A = B^(1/n) (if n is const)
                    // If A contains var and n does not.
                    if contains_var(&simplifier.context, b, var)
                        && !contains_var(&simplifier.context, e, var)
                    {
                        // Prevent unwrapping positive integer powers (handled by Polynomial/Quadratic)
                        // e.g. x^2 = ... don't turn into x = sqrt(...)
                        let is_pos_int = |ctx: &Context, e_id: ExprId| -> bool {
                            match ctx.get(e_id) {
                                Expr::Number(n) => {
                                    n.is_integer()
                                        && *n > num_rational::BigRational::from_integer(0.into())
                                }
                                Expr::Div(n_id, d_id) => {
                                    if let (Expr::Number(n), Expr::Number(d)) =
                                        (ctx.get(*n_id), ctx.get(*d_id))
                                    {
                                        if !d.is_zero() {
                                            let val = n / d;
                                            return val.is_integer()
                                                && val
                                                    > num_rational::BigRational::from_integer(
                                                        0.into(),
                                                    );
                                        }
                                    }
                                    false
                                }
                                _ => false,
                            }
                        };

                        if is_pos_int(&simplifier.context, e) {
                            // Don't unwrap x^2, x^4 etc.
                            return None;
                        }

                        // A^n = B -> A = B^(1/n)
                        let one = simplifier.context.num(1);
                        let inv_exp = simplifier.context.add(Expr::Div(one, e));
                        let new_other = simplifier.context.add(Expr::Pow(other, inv_exp));
                        let new_eq = if is_lhs {
                            Equation {
                                lhs: b,
                                rhs: new_other,
                                op,
                            }
                        } else {
                            Equation {
                                lhs: new_other,
                                rhs: b,
                                op,
                            }
                        };
                        Some((
                            new_eq,
                            format!("Raise both sides to 1/{:?}", simplifier.context.get(e)),
                        ))
                    } else if !contains_var(&simplifier.context, b, var)
                        && contains_var(&simplifier.context, e, var)
                    {
                        // A^x = B -> x * ln(A) = ln(B)
                        // Use domain classifier for semantic-aware solving

                        use crate::solver::domain_guards::{classify_log_solve, LogSolveDecision};

                        // PRE-CHECK: Handle base = 1 before classifier
                        // 1^x = 1 -> AllReals, 1^x = b (b≠1) -> Empty
                        if let Expr::Number(n) = simplifier.context.get(b) {
                            if *n == num_rational::BigRational::from_integer(1.into()) {
                                // Base is 1
                                if let Expr::Number(rhs_n) = simplifier.context.get(other) {
                                    if *rhs_n == num_rational::BigRational::from_integer(1.into()) {
                                        // 1^x = 1 -> AllReals (handled specially)
                                        // We can't return AllReals directly from invert closure,
                                        // so skip and let IsolationStrategy handle it
                                        return None;
                                    } else {
                                        // 1^x = b (b≠1) -> Empty (also skip)
                                        return None;
                                    }
                                }
                                // 1^x = symbolic -> skip (can be 1 or not)
                                return None;
                            }
                        }

                        // Use the domain classifier
                        let decision = classify_log_solve(&simplifier.context, b, other, opts);

                        match decision {
                            LogSolveDecision::Ok => {
                                // Safe to take ln - no assumptions needed
                            }
                            LogSolveDecision::OkWithAssumptions(assumptions) => {
                                // Record each assumption via the thread-local collector
                                for assumption in assumptions {
                                    let event = assumption.to_assumption_event(
                                        &simplifier.context,
                                        b,
                                        other,
                                    );
                                    crate::solver::note_assumption(event);
                                }
                            }
                            LogSolveDecision::EmptySet(_) => {
                                // No solutions - skip, let IsolationStrategy handle
                                return None;
                            }
                            LogSolveDecision::NeedsComplex(msg) => {
                                // In RealOnly, can't proceed
                                // In wildcard scope: should return residual (not implemented here)
                                // For now, skip and let IsolationStrategy handle
                                let _ = msg; // suppress warning
                                return None;
                            }
                            LogSolveDecision::Unsupported(_, _) => {
                                // Cannot justify in current mode - skip
                                return None;
                            }
                        }

                        // Safe to take ln of both sides
                        let ln_str = "ln".to_string();

                        // ln(A^x) -> x * ln(A)
                        // We construct x * ln(A) directly
                        let ln_b = simplifier
                            .context
                            .add(Expr::Function(ln_str.clone(), vec![b]));
                        let new_lhs_part = simplifier.context.add(Expr::Mul(e, ln_b));

                        // ln(B)
                        let ln_other = simplifier.context.add(Expr::Function(ln_str, vec![other]));

                        let new_eq = if is_lhs {
                            Equation {
                                lhs: new_lhs_part,
                                rhs: ln_other,
                                op,
                            }
                        } else {
                            Equation {
                                lhs: ln_other,
                                rhs: new_lhs_part,
                                op,
                            }
                        };
                        Some((new_eq, "Take log base e of both sides".to_string()))
                    } else {
                        None
                    }
                }
                _ => None,
            }
        };

        // Try LHS
        if lhs_has {
            if let Some((new_eq, desc)) = invert(eq.lhs, eq.rhs, eq.op.clone(), true) {
                let mut steps = Vec::new();
                if simplifier.collect_steps() {
                    steps.push(SolveStep {
                        description: desc,
                        equation_after: new_eq.clone(),
                    });
                }
                match solve(&new_eq, var, simplifier) {
                    Ok((set, mut sub_steps)) => {
                        steps.append(&mut sub_steps);
                        return Some(Ok((set, steps)));
                    }
                    Err(e) => return Some(Err(e)),
                }
            }
        }

        // Try RHS
        if rhs_has {
            if let Some((new_eq, desc)) = invert(eq.rhs, eq.lhs, eq.op.clone(), false) {
                let mut steps = Vec::new();
                if simplifier.collect_steps() {
                    steps.push(SolveStep {
                        description: desc,
                        equation_after: new_eq.clone(),
                    });
                }
                match solve(&new_eq, var, simplifier) {
                    Ok((set, mut sub_steps)) => {
                        steps.append(&mut sub_steps);
                        return Some(Ok((set, steps)));
                    }
                    Err(e) => return Some(Err(e)),
                }
            }
        }

        None
    }

    // Note: We use the default should_verify() = true here.
    // Selective verification in solve() handles symbolic solutions.
}

// --- Helper for CollectTermsStrategy (currently unused) ---

// fn is_zero(ctx: &Context, expr: ExprId) -> bool {
//     matches!(ctx.get(expr), Expr::Number(n) if n.is_zero())
// }

// --- CollectTermsStrategy: Handles linear equations with variables on both sides ---

pub struct CollectTermsStrategy;

impl SolverStrategy for CollectTermsStrategy {
    fn name(&self) -> &str {
        "Collect Terms"
    }

    fn apply(
        &self,
        eq: &Equation,
        var: &str,
        simplifier: &mut Simplifier,
        _opts: &SolverOptions,
    ) -> Option<Result<(SolutionSet, Vec<SolveStep>), CasError>> {
        let lhs_has = contains_var(&simplifier.context, eq.lhs, var);
        let rhs_has = contains_var(&simplifier.context, eq.rhs, var);

        // Only apply if variable is on BOTH sides
        if !lhs_has || !rhs_has {
            return None;
        }

        let mut steps = Vec::new();

        // Strategy: Subtract RHS from both sides to move everything to LHS
        // ax + b = cx + d  ->  ax + b - (cx + d) = cx + d - (cx + d)
        //                  ->  ax - cx + b - d = 0

        let neg_rhs = simplifier.context.add(Expr::Neg(eq.rhs));
        let new_lhs = simplifier.context.add(Expr::Add(eq.lhs, neg_rhs));
        let new_rhs = simplifier.context.add(Expr::Add(eq.rhs, neg_rhs));

        // Simplify both sides
        let (simp_lhs, _) = simplifier.simplify(new_lhs);
        let (simp_rhs, _) = simplifier.simplify(new_rhs);

        if simplifier.collect_steps() {
            steps.push(SolveStep {
                description: format!(
                    "Subtract {} from both sides",
                    cas_ast::DisplayExpr {
                        context: &simplifier.context,
                        id: eq.rhs
                    }
                ),
                equation_after: Equation {
                    lhs: simp_lhs,
                    rhs: simp_rhs,
                    op: eq.op.clone(),
                },
            });
        }

        // Now recursively solve the simplified equation
        // This should now have variable only on one side
        let new_eq = Equation {
            lhs: simp_lhs,
            rhs: simp_rhs,
            op: eq.op.clone(),
        };
        match solve(&new_eq, var, simplifier) {
            Ok((set, mut solve_steps)) => {
                steps.append(&mut solve_steps);
                Some(Ok((set, steps)))
            }
            Err(e) => Some(Err(e)),
        }
    }
}

// --- RationalExponentStrategy: Handles equations like x^(p/q) = rhs ---
// Converts x^(p/q) = rhs to x^p = rhs^q to avoid infinite loops with fractional exponents

pub struct RationalExponentStrategy;

impl SolverStrategy for RationalExponentStrategy {
    fn name(&self) -> &str {
        "Rational Exponent"
    }

    fn apply(
        &self,
        eq: &Equation,
        var: &str,
        simplifier: &mut Simplifier,
        _opts: &SolverOptions,
    ) -> Option<Result<(SolutionSet, Vec<SolveStep>), CasError>> {
        // Only handle equality for now
        if eq.op != RelOp::Eq {
            return None;
        }

        // Check if LHS is Pow(base, exp) where base contains var and exp is rational p/q
        let lhs_has = contains_var(&simplifier.context, eq.lhs, var);
        let rhs_has = contains_var(&simplifier.context, eq.rhs, var);

        // We need var only on one side in a power
        if rhs_has {
            return None;
        }
        if !lhs_has {
            return None;
        }

        // Try to match Pow(base, p/q) on LHS
        let (base, p, q) = match_rational_power(&simplifier.context, eq.lhs, var)?;

        let mut steps = Vec::new();

        // Raise both sides to power q: (base^(p/q))^q = rhs^q → base^p = rhs^q
        let q_expr = simplifier.context.num(q);

        // New LHS: base^p
        let p_expr = simplifier.context.num(p);
        let new_lhs = simplifier.context.add(Expr::Pow(base, p_expr));

        // New RHS: rhs^q
        let new_rhs = simplifier.context.add(Expr::Pow(eq.rhs, q_expr));

        // Simplify both sides
        let (sim_lhs, _) = simplifier.simplify(new_lhs);
        let (sim_rhs, _) = simplifier.simplify(new_rhs);

        let new_eq = Equation {
            lhs: sim_lhs,
            rhs: sim_rhs,
            op: RelOp::Eq,
        };

        if simplifier.collect_steps() {
            steps.push(SolveStep {
                description: format!(
                    "Raise both sides to power {} to eliminate fractional exponent",
                    q
                ),
                equation_after: new_eq.clone(),
            });
        }

        // Recursively solve the new equation
        match solve(&new_eq, var, simplifier) {
            Ok((set, mut sub_steps)) => {
                steps.append(&mut sub_steps);

                // For even q, we need to verify solutions (could introduce extraneous)
                // The main solve() already verifies against original equation
                Some(Ok((set, steps)))
            }
            Err(e) => Some(Err(e)),
        }
    }
}

/// Match Pow(base, p/q) where base contains var and p/q is a non-integer rational
pub fn match_rational_power(ctx: &Context, expr: ExprId, var: &str) -> Option<(ExprId, i64, i64)> {
    if let Expr::Pow(base, exp) = ctx.get(expr) {
        // Check that base contains the variable
        if !contains_var(ctx, *base, var) {
            return None;
        }

        // Check if exp is a rational number p/q with q != 1
        let exp_data = ctx.get(*exp);

        match exp_data {
            Expr::Number(n) => {
                // Check if denominator != 1 (not an integer)
                let denom = n.denom();
                let numer = n.numer();
                if *denom == 1.into() {
                    // Integer exponent, not what we're looking for
                    return None;
                }
                // Convert to i64 (if possible)
                let p: i64 = numer.try_into().ok()?;
                let q: i64 = denom.try_into().ok()?;
                if q <= 0 {
                    return None;
                }
                Some((*base, p, q))
            }
            Expr::Div(num_id, den_id) => {
                // Check if it's a simple p/q
                if let (Expr::Number(p_rat), Expr::Number(q_rat)) =
                    (ctx.get(*num_id), ctx.get(*den_id))
                {
                    if !p_rat.is_integer() || !q_rat.is_integer() {
                        return None;
                    }
                    let p: i64 = p_rat.numer().try_into().ok()?;
                    let q: i64 = q_rat.numer().try_into().ok()?;
                    if q <= 1 {
                        return None;
                    }
                    Some((*base, p, q))
                } else {
                    None
                }
            }
            _ => None,
        }
    } else {
        None
    }
}
