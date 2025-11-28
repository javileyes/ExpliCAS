use cas_ast::{Expr, Equation, RelOp, SolutionSet, Interval, BoundType, ExprId, Context};
use crate::engine::Simplifier;
use crate::solver::SolveStep;
use crate::polynomial::Polynomial;
use num_rational::BigRational;
use num_traits::Zero;
use std::cmp::Ordering;
use crate::solver::solution_set::{neg_inf, pos_inf, compare_values};
use crate::solver::isolation::{contains_var, isolate};
use crate::error::CasError;
use crate::solver::strategy::SolverStrategy;
use crate::solver::solve; // Needed for recursive solve in substitution
use crate::ordering::compare_expr;

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
                _ => false
            };
            if is_e && contains_var(ctx, *e, var) {
                terms.push(expr);
            }
        },
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) => {
            terms.extend(collect_exponential_terms(ctx, *l, var));
            terms.extend(collect_exponential_terms(ctx, *r, var));
        },
        _ => {}
    }
    terms
}

pub fn substitute_expr(ctx: &mut Context, expr: ExprId, target: ExprId, replacement: ExprId) -> ExprId {
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
        },
        Expr::Sub(l, r) => {
            let new_l = substitute_expr(ctx, l, target, replacement);
            let new_r = substitute_expr(ctx, r, target, replacement);
            if new_l != l || new_r != r {
                return ctx.add(Expr::Sub(new_l, new_r));
            }
        },
        Expr::Mul(l, r) => {
            let new_l = substitute_expr(ctx, l, target, replacement);
            let new_r = substitute_expr(ctx, r, target, replacement);
            if new_l != l || new_r != r {
                return ctx.add(Expr::Mul(new_l, new_r));
            }
        },
        Expr::Div(l, r) => {
            let new_l = substitute_expr(ctx, l, target, replacement);
            let new_r = substitute_expr(ctx, r, target, replacement);
            if new_l != l || new_r != r {
                return ctx.add(Expr::Div(new_l, new_r));
            }
        },
        Expr::Pow(b, e) => {
            let new_b = substitute_expr(ctx, b, target, replacement);
            let new_e = substitute_expr(ctx, e, target, replacement);
            if new_b != b || new_e != e {
                return ctx.add(Expr::Pow(new_b, new_e));
            }
        },
        Expr::Neg(e) => {
            let new_e = substitute_expr(ctx, e, target, replacement);
            if new_e != e {
                return ctx.add(Expr::Neg(new_e));
            }
        },
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
        },
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

    fn apply(&self, eq: &Equation, var: &str, simplifier: &mut Simplifier) -> Option<Result<(SolutionSet, Vec<SolveStep>), CasError>> {
        if let Some(sub_var_expr) = detect_substitution(&mut simplifier.context, eq, var) {
            let mut steps = Vec::new();
            if simplifier.collect_steps {
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
            
            let new_eq = Equation { lhs: new_lhs, rhs: new_rhs, op: eq.op.clone() };
            
            if simplifier.collect_steps {
                steps.push(SolveStep {
                    description: format!("Substituted equation: {:?} {} {:?}", new_eq.lhs, new_eq.op, new_eq.rhs),
                    equation_after: new_eq.clone(),
                });
            }
            
            // Solve for u
            let (u_solutions, mut u_steps) = match solve(&new_eq, u_sym, simplifier) {
                Ok(res) => res,
                Err(e) => return Some(Err(e)),
            };
            steps.append(&mut u_steps);
            
            // Now solve u = val for each solution
            match u_solutions {
                SolutionSet::Discrete(vals) => {
                    let mut final_solutions = Vec::new();
                    for val in vals {
                        // Solve sub_var_expr = val
                        let sub_eq = Equation { lhs: sub_var_expr, rhs: val, op: RelOp::Eq };
                        if simplifier.collect_steps {
                            steps.push(SolveStep {
                                description: format!("Back-substitute: {:?} = {:?}", sub_var_expr, val),
                                equation_after: sub_eq.clone(),
                            });
                        }
                        let (x_sol, mut x_steps) = match solve(&sub_eq, var, simplifier) {
                            Ok(res) => res,
                            Err(e) => return Some(Err(e)),
                        };
                        steps.append(&mut x_steps);
                        
                        if let SolutionSet::Discrete(xs) = x_sol {
                            final_solutions.extend(xs);
                        }
                    }
                    return Some(Ok((SolutionSet::Discrete(final_solutions), steps)));
                },
                _ => {
                    // Handle intervals? Too complex for now.
                    return Some(Err(CasError::SolverError("Substitution strategy currently only supports discrete solutions".to_string())));
                }
            }
        }
        None
    }
}

pub struct QuadraticStrategy;

impl SolverStrategy for QuadraticStrategy {
    fn name(&self) -> &str {
        "Quadratic Formula"
    }

    fn apply(&self, eq: &Equation, var: &str, simplifier: &mut Simplifier) -> Option<Result<(SolutionSet, Vec<SolveStep>), CasError>> {
        let mut steps = Vec::new();
        
        // Move everything to LHS: lhs - rhs = 0
        let poly_expr = simplifier.context.add(Expr::Sub(eq.lhs, eq.rhs));
        let (sim_poly_expr, _) = simplifier.simplify(poly_expr);
        
        if let Ok(poly) = Polynomial::from_expr(&simplifier.context, sim_poly_expr, var) {
            if poly.degree() == 2 {
                if simplifier.collect_steps {
                    steps.push(SolveStep {
                        description: "Detected quadratic equation. Applying quadratic formula.".to_string(),
                        equation_after: Equation { lhs: sim_poly_expr, rhs: simplifier.context.num(0), op: RelOp::Eq }
                    });
                }
                
                // ax^2 + bx + c = 0
                // coeffs[0] = c, coeffs[1] = b, coeffs[2] = a
                let c = poly.coeffs.get(0).cloned().unwrap_or_else(BigRational::zero);
                let b = poly.coeffs.get(1).cloned().unwrap_or_else(BigRational::zero);
                let a = poly.coeffs.get(2).cloned().unwrap_or_else(BigRational::zero);
                
                // delta = b^2 - 4ac
                let b2 = b.clone() * b.clone();
                let four_ac = BigRational::from_integer(4.into()) * a.clone() * c.clone();
                let delta = b2 - four_ac;
                
                // We need to return solutions in terms of Expr
                // x = (-b +/- sqrt(delta)) / 2a
                
                let neg_b = -b.clone();
                let two_a = BigRational::from_integer(2.into()) * a.clone();
                
                let delta_expr = simplifier.context.add(Expr::Number(delta.clone()));
                let neg_b_expr = simplifier.context.add(Expr::Number(neg_b));
                let two_a_expr = simplifier.context.add(Expr::Number(two_a.clone()));
                
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
                let (r1, r2) = if compare_values(&simplifier.context, sim_sol1, sim_sol2) == Ordering::Greater {
                    (sim_sol2, sim_sol1)
                } else {
                    (sim_sol1, sim_sol2)
                };

                // Determine parabola direction
                let opens_up = a > BigRational::zero();

                // Helper for intervals
                let mk_interval = |min, min_type, max, max_type| {
                    SolutionSet::Continuous(Interval { min, min_type, max, max_type })
                };
                
                let result = if delta > BigRational::zero() {
                    // Two distinct roots r1 < r2
                    match eq.op {
                        RelOp::Eq => SolutionSet::Discrete(vec![r1, r2]),
                        RelOp::Neq => {
                            // (-inf, r1) U (r1, r2) U (r2, inf)
                            let i1 = Interval { min: neg_inf(&mut simplifier.context), min_type: BoundType::Open, max: r1, max_type: BoundType::Open };
                            let i2 = Interval { min: r1, min_type: BoundType::Open, max: r2, max_type: BoundType::Open };
                            let i3 = Interval { min: r2, min_type: BoundType::Open, max: pos_inf(&mut simplifier.context), max_type: BoundType::Open };
                            SolutionSet::Union(vec![i1, i2, i3])
                        },
                        RelOp::Lt => {
                            if opens_up {
                                // Parabola < 0 between roots: (r1, r2)
                                mk_interval(r1, BoundType::Open, r2, BoundType::Open)
                            } else {
                                // Parabola < 0 outside roots: (-inf, r1) U (r2, inf)
                                let i1 = Interval { min: neg_inf(&mut simplifier.context), min_type: BoundType::Open, max: r1, max_type: BoundType::Open };
                                let i2 = Interval { min: r2, min_type: BoundType::Open, max: pos_inf(&mut simplifier.context), max_type: BoundType::Open };
                                SolutionSet::Union(vec![i1, i2])
                            }
                        },
                        RelOp::Leq => {
                            if opens_up {
                                // [r1, r2]
                                mk_interval(r1, BoundType::Closed, r2, BoundType::Closed)
                            } else {
                                // (-inf, r1] U [r2, inf)
                                let i1 = Interval { min: neg_inf(&mut simplifier.context), min_type: BoundType::Open, max: r1, max_type: BoundType::Closed };
                                let i2 = Interval { min: r2, min_type: BoundType::Closed, max: pos_inf(&mut simplifier.context), max_type: BoundType::Open };
                                SolutionSet::Union(vec![i1, i2])
                            }
                        },
                        RelOp::Gt => {
                            if opens_up {
                                // Parabola > 0 outside roots: (-inf, r1) U (r2, inf)
                                let i1 = Interval { min: neg_inf(&mut simplifier.context), min_type: BoundType::Open, max: r1, max_type: BoundType::Open };
                                let i2 = Interval { min: r2, min_type: BoundType::Open, max: pos_inf(&mut simplifier.context), max_type: BoundType::Open };
                                SolutionSet::Union(vec![i1, i2])
                            } else {
                                // Parabola > 0 between roots: (r1, r2)
                                mk_interval(r1, BoundType::Open, r2, BoundType::Open)
                            }
                        },
                        RelOp::Geq => {
                            if opens_up {
                                // (-inf, r1] U [r2, inf)
                                let i1 = Interval { min: neg_inf(&mut simplifier.context), min_type: BoundType::Open, max: r1, max_type: BoundType::Closed };
                                let i2 = Interval { min: r2, min_type: BoundType::Closed, max: pos_inf(&mut simplifier.context), max_type: BoundType::Open };
                                SolutionSet::Union(vec![i1, i2])
                            } else {
                                // [r1, r2]
                                mk_interval(r1, BoundType::Closed, r2, BoundType::Closed)
                            }
                        },
                    }
                } else if delta == BigRational::zero() {
                    // One repeated root r1
                    match eq.op {
                        RelOp::Eq => SolutionSet::Discrete(vec![r1]),
                        RelOp::Neq => {
                            // (-inf, r1) U (r1, inf)
                            let i1 = Interval { min: neg_inf(&mut simplifier.context), min_type: BoundType::Open, max: r1, max_type: BoundType::Open };
                            let i2 = Interval { min: r1, min_type: BoundType::Open, max: pos_inf(&mut simplifier.context), max_type: BoundType::Open };
                            SolutionSet::Union(vec![i1, i2])
                        },
                        RelOp::Lt => {
                            if opens_up {
                                // (x-r)^2 < 0 -> Empty
                                SolutionSet::Empty
                            } else {
                                // -(x-r)^2 < 0 -> All Reals except r
                                let i1 = Interval { min: neg_inf(&mut simplifier.context), min_type: BoundType::Open, max: r1, max_type: BoundType::Open };
                                let i2 = Interval { min: r1, min_type: BoundType::Open, max: pos_inf(&mut simplifier.context), max_type: BoundType::Open };
                                SolutionSet::Union(vec![i1, i2])
                            }
                        },
                        RelOp::Leq => {
                            if opens_up {
                                // (x-r)^2 <= 0 -> x = r
                                SolutionSet::Discrete(vec![r1])
                            } else {
                                // -(x-r)^2 <= 0 -> All Reals
                                SolutionSet::AllReals
                            }
                        },
                        RelOp::Gt => {
                            if opens_up {
                                // (x-r)^2 > 0 -> All Reals except r
                                let i1 = Interval { min: neg_inf(&mut simplifier.context), min_type: BoundType::Open, max: r1, max_type: BoundType::Open };
                                let i2 = Interval { min: r1, min_type: BoundType::Open, max: pos_inf(&mut simplifier.context), max_type: BoundType::Open };
                                SolutionSet::Union(vec![i1, i2])
                            } else {
                                // -(x-r)^2 > 0 -> Empty
                                SolutionSet::Empty
                            }
                        },
                        RelOp::Geq => {
                            if opens_up {
                                // (x-r)^2 >= 0 -> All Reals
                                SolutionSet::AllReals
                            } else {
                                // -(x-r)^2 >= 0 -> x = r
                                SolutionSet::Discrete(vec![r1])
                            }
                        },
                    }
                } else {
                    // delta < 0, no real roots
                    // Parabola is always positive (if a > 0) or always negative (if a < 0)
                    let always_pos = opens_up;
                    match eq.op {
                        RelOp::Eq => SolutionSet::Empty,
                        RelOp::Neq => SolutionSet::AllReals,
                        RelOp::Lt => if always_pos { SolutionSet::Empty } else { SolutionSet::AllReals },
                        RelOp::Leq => if always_pos { SolutionSet::Empty } else { SolutionSet::AllReals },
                        RelOp::Gt => if always_pos { SolutionSet::AllReals } else { SolutionSet::Empty },
                        RelOp::Geq => if always_pos { SolutionSet::AllReals } else { SolutionSet::Empty },
                    }
                };
                
                return Some(Ok((result, steps)));
            }
        }
        None
    }
}

pub struct IsolationStrategy;

impl SolverStrategy for IsolationStrategy {
    fn name(&self) -> &str {
        "Isolation"
    }

    fn apply(&self, eq: &Equation, var: &str, simplifier: &mut Simplifier) -> Option<Result<(SolutionSet, Vec<SolveStep>), CasError>> {
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
            if simplifier.collect_steps {
                steps.push(SolveStep {
                    description: "Swap sides to put variable on LHS".to_string(),
                    equation_after: Equation { lhs: eq.rhs, rhs: eq.lhs, op: new_op.clone() },
                });
            }
            match isolate(eq.rhs, eq.lhs, new_op, var, simplifier) {
                Ok((set, mut iso_steps)) => {
                    steps.append(&mut iso_steps);
                    return Some(Ok((set, steps)));
                },
                Err(e) => return Some(Err(e)),
            }
        }
        
        // LHS has var
        match isolate(eq.lhs, eq.rhs, eq.op.clone(), var, simplifier) {
            Ok((set, steps)) => Some(Ok((set, steps))),
            Err(e) => Some(Err(e)),
        }
    }
}

pub struct UnwrapStrategy;

impl SolverStrategy for UnwrapStrategy {
    fn name(&self) -> &str {
        "Unwrap"
    }

    fn apply(&self, eq: &Equation, var: &str, simplifier: &mut Simplifier) -> Option<Result<(SolutionSet, Vec<SolveStep>), CasError>> {
        // Try to unwrap functions on LHS or RHS to expose the variable or transform the equation.
        // This is useful when var is on both sides, e.g. sqrt(2x+3) = x.
        
        let lhs_has = contains_var(&simplifier.context, eq.lhs, var);
        let rhs_has = contains_var(&simplifier.context, eq.rhs, var);
        
        // Only apply if var is on both sides? 
        // If var is only on one side, IsolationStrategy handles it.
        // But IsolationStrategy might be later in the list.
        // Let's apply if top-level is a function/pow that we can invert.
        
        if !lhs_has && !rhs_has { return None; }
        
        // Helper to invert
        let mut invert = |target: ExprId, other: ExprId, op: RelOp, is_lhs: bool| -> Option<(Equation, String)> {
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
                                Equation { lhs: arg, rhs: new_other, op }
                            } else {
                                Equation { lhs: new_other, rhs: arg, op }
                            };
                            Some((new_eq, "Square both sides".to_string()))
                        },
                        "ln" => {
                            // ln(A) = B -> A = e^B
                            let e = simplifier.context.add(Expr::Constant(cas_ast::Constant::E));
                            let new_other = simplifier.context.add(Expr::Pow(e, other));
                            let new_eq = if is_lhs {
                                Equation { lhs: arg, rhs: new_other, op }
                            } else {
                                Equation { lhs: new_other, rhs: arg, op }
                            };
                            Some((new_eq, "Exponentiate (base e)".to_string()))
                        },
                        "exp" => {
                            // exp(A) = B -> A = ln(B)
                            let new_other = simplifier.context.add(Expr::Function("ln".to_string(), vec![other]));
                            let new_eq = if is_lhs {
                                Equation { lhs: arg, rhs: new_other, op }
                            } else {
                                Equation { lhs: new_other, rhs: arg, op }
                            };
                            Some((new_eq, "Take natural log".to_string()))
                        },
                        _ => None
                    }
                },
                Expr::Pow(b, e) => {
                    // A^n = B -> A = B^(1/n) (if n is const)
                    // If A contains var and n does not.
                    if contains_var(&simplifier.context, b, var) && !contains_var(&simplifier.context, e, var) {
                         // Prevent unwrapping positive integer powers (handled by Polynomial/Quadratic)
                         // e.g. x^2 = ... don't turn into x = sqrt(...)
                         if let Expr::Number(n) = simplifier.context.get(e) {
                             if n.is_integer() && *n > num_rational::BigRational::from_integer(0.into()) {
                                 return None;
                             }
                         }

                         let one = simplifier.context.num(1);
                         let inv_exp = simplifier.context.add(Expr::Div(one, e));
                         let new_other = simplifier.context.add(Expr::Pow(other, inv_exp));
                         // Handle even powers? |A| = ...
                         // For now, simple inversion.
                         let new_eq = if is_lhs {
                             Equation { lhs: b, rhs: new_other, op }
                         } else {
                             Equation { lhs: new_other, rhs: b, op }
                         };
                         Some((new_eq, "Take root".to_string()))
                    } else {
                        None
                    }
                },
                _ => None
            }
        };

        // Try LHS
        if lhs_has {
            if let Some((new_eq, desc)) = invert(eq.lhs, eq.rhs, eq.op.clone(), true) {
                 let mut steps = Vec::new();
                 if simplifier.collect_steps {
                     steps.push(SolveStep {
                         description: desc,
                         equation_after: new_eq.clone(),
                     });
                 }
                 match solve(&new_eq, var, simplifier) {
                     Ok((set, mut sub_steps)) => {
                         steps.append(&mut sub_steps);
                         return Some(Ok((set, steps)));
                     },
                     Err(e) => return Some(Err(e)),
                 }
            }
        }
        
        // Try RHS
        if rhs_has {
            if let Some((new_eq, desc)) = invert(eq.rhs, eq.lhs, eq.op.clone(), false) {
                 let mut steps = Vec::new();
                 if simplifier.collect_steps {
                     steps.push(SolveStep {
                         description: desc,
                         equation_after: new_eq.clone(),
                     });
                 }
                 match solve(&new_eq, var, simplifier) {
                     Ok((set, mut sub_steps)) => {
                         steps.append(&mut sub_steps);
                         return Some(Ok((set, steps)));
                     },
                     Err(e) => return Some(Err(e)),
                 }
            }
        }

        None
    }
}

