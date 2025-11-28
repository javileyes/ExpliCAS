use cas_ast::{Expr, Equation, RelOp, SolutionSet, Interval, BoundType};
use std::rc::Rc;
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

// --- Helper Functions (Keep these as they are useful helpers) ---

pub fn detect_substitution(eq: &Equation, var: &str) -> Option<Rc<Expr>> {
    // ... (Keep existing implementation)
    // Heuristic: Look for e^x.
    // Collect all Pow(e, ...) terms.
    let terms = collect_exponential_terms(&eq.lhs, var);
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
        if let Expr::Pow(_, exp) = term.as_ref() {
            // Check if exponent is exactly var (e^x)
            if let Expr::Variable(v) = exp.as_ref() {
                if v == var {
                    base_term = Some(term.clone());
                }
            } else {
                // Check if exponent is k*var (e^2x)
                // If so, it's complex.
                if contains_var(exp, var) {
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
            if let Expr::Pow(b, _) = terms[0].as_ref() {
                return Some(Expr::pow(b.clone(), Expr::var(var)));
            }
        }
    }
    
    None
}

pub fn collect_exponential_terms(expr: &Rc<Expr>, var: &str) -> Vec<Rc<Expr>> {
    let mut terms = Vec::new();
    match expr.as_ref() {
        Expr::Pow(b, e) => {
            let is_e = match b.as_ref() {
                Expr::Variable(name) => name == "e",
                Expr::Constant(c) => matches!(c, cas_ast::Constant::E),
                _ => false
            };
            if is_e && contains_var(e, var) {
                terms.push(expr.clone());
            }
        },
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) => {
            terms.extend(collect_exponential_terms(l, var));
            terms.extend(collect_exponential_terms(r, var));
        },
        _ => {}
    }
    terms
}

pub fn substitute_expr(expr: &Rc<Expr>, target: &Rc<Expr>, replacement_name: &str) -> Rc<Expr> {
    if expr == target {
        return Expr::var(replacement_name);
    }
    
    // Handle e^(2x) -> (e^x)^2 -> u^2
    if let Expr::Pow(b, e) = expr.as_ref() {
        if let Expr::Pow(tb, te) = target.as_ref() {
            if b == tb {
                // Check if e = k * te
                // e.g. 2x = 2 * x
                if let Expr::Mul(l, r) = e.as_ref() {
                    if (l == te && matches!(r.as_ref(), Expr::Number(_))) || (r == te && matches!(l.as_ref(), Expr::Number(_))) {
                        let coeff = if l == te { r } else { l };
                        return Expr::pow(Expr::var(replacement_name), coeff.clone());
                    }
                }
            }
        }
    }

    match expr.as_ref() {
        Expr::Add(l, r) => Expr::add(substitute_expr(l, target, replacement_name), substitute_expr(r, target, replacement_name)),
        Expr::Sub(l, r) => Expr::sub(substitute_expr(l, target, replacement_name), substitute_expr(r, target, replacement_name)),
        Expr::Mul(l, r) => Expr::mul(substitute_expr(l, target, replacement_name), substitute_expr(r, target, replacement_name)),
        Expr::Div(l, r) => Expr::div(substitute_expr(l, target, replacement_name), substitute_expr(r, target, replacement_name)),
        Expr::Pow(b, e) => Expr::pow(substitute_expr(b, target, replacement_name), substitute_expr(e, target, replacement_name)),
        _ => expr.clone()
    }
}

// --- Strategies ---

pub struct SubstitutionStrategy;

impl SolverStrategy for SubstitutionStrategy {
    fn name(&self) -> &str {
        "Substitution"
    }

    fn apply(&self, eq: &Equation, var: &str, simplifier: &Simplifier) -> Option<Result<(SolutionSet, Vec<SolveStep>), CasError>> {
        if let Some(sub_var_expr) = detect_substitution(eq, var) {
            let mut steps = Vec::new();
            if simplifier.collect_steps {
                steps.push(SolveStep {
                    description: format!("Detected substitution: u = {}", sub_var_expr),
                    equation_after: eq.clone(),
                });
            }
            
            // Rewrite equation in terms of u
            let u_sym = "u";
            let new_lhs = substitute_expr(&eq.lhs, &sub_var_expr, u_sym);
            let new_rhs = substitute_expr(&eq.rhs, &sub_var_expr, u_sym);
            
            let new_eq = Equation { lhs: new_lhs, rhs: new_rhs, op: eq.op.clone() };
            
            if simplifier.collect_steps {
                steps.push(SolveStep {
                    description: format!("Substituted equation: {} {} {}", new_eq.lhs, new_eq.op, new_eq.rhs),
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
                        let sub_eq = Equation { lhs: sub_var_expr.clone(), rhs: val.clone(), op: RelOp::Eq };
                        if simplifier.collect_steps {
                            steps.push(SolveStep {
                                description: format!("Back-substitute: {} = {}", sub_var_expr, val),
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

    fn apply(&self, eq: &Equation, var: &str, simplifier: &Simplifier) -> Option<Result<(SolutionSet, Vec<SolveStep>), CasError>> {
        let mut steps = Vec::new();
        
        // Move everything to LHS: lhs - rhs = 0
        let poly_expr = Expr::sub(eq.lhs.clone(), eq.rhs.clone());
        let (sim_poly_expr, _) = simplifier.simplify(poly_expr);
        
        if let Ok(poly) = Polynomial::from_expr(&sim_poly_expr, var) {
            if poly.degree() == 2 {
                if simplifier.collect_steps {
                    steps.push(SolveStep {
                        description: "Detected quadratic equation. Applying quadratic formula.".to_string(),
                        equation_after: Equation { lhs: sim_poly_expr.clone(), rhs: Expr::num(0), op: RelOp::Eq }
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
                
                let delta_expr = Rc::new(Expr::Number(delta.clone()));
                let neg_b_expr = Rc::new(Expr::Number(neg_b));
                let two_a_expr = Rc::new(Expr::Number(two_a.clone()));
                
                // sqrt(delta)
                let sqrt_delta = Expr::pow(delta_expr, Expr::rational(1, 2));
                
                // x1 = (-b - sqrt(delta)) / 2a (Smaller root if a > 0)
                let num1 = Expr::sub(neg_b_expr.clone(), sqrt_delta.clone());
                let sol1 = Expr::div(num1, two_a_expr.clone());
                
                // x2 = (-b + sqrt(delta)) / 2a (Larger root if a > 0)
                let num2 = Expr::add(neg_b_expr, sqrt_delta);
                let sol2 = Expr::div(num2, two_a_expr);
                
                let (sim_sol1, _) = simplifier.simplify(sol1);
                let (sim_sol2, _) = simplifier.simplify(sol2);
                
                // Ensure r1 <= r2
                let (r1, r2) = if compare_values(&sim_sol1, &sim_sol2) == Ordering::Greater {
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
                            let i1 = Interval { min: neg_inf(), min_type: BoundType::Open, max: r1.clone(), max_type: BoundType::Open };
                            let i2 = Interval { min: r1.clone(), min_type: BoundType::Open, max: r2.clone(), max_type: BoundType::Open };
                            let i3 = Interval { min: r2.clone(), min_type: BoundType::Open, max: pos_inf(), max_type: BoundType::Open };
                            SolutionSet::Union(vec![i1, i2, i3])
                        },
                        RelOp::Lt => {
                            if opens_up {
                                // Parabola < 0 between roots: (r1, r2)
                                mk_interval(r1, BoundType::Open, r2, BoundType::Open)
                            } else {
                                // Parabola < 0 outside roots: (-inf, r1) U (r2, inf)
                                let i1 = Interval { min: neg_inf(), min_type: BoundType::Open, max: r1, max_type: BoundType::Open };
                                let i2 = Interval { min: r2, min_type: BoundType::Open, max: pos_inf(), max_type: BoundType::Open };
                                SolutionSet::Union(vec![i1, i2])
                            }
                        },
                        RelOp::Leq => {
                            if opens_up {
                                // [r1, r2]
                                mk_interval(r1, BoundType::Closed, r2, BoundType::Closed)
                            } else {
                                // (-inf, r1] U [r2, inf)
                                let i1 = Interval { min: neg_inf(), min_type: BoundType::Open, max: r1, max_type: BoundType::Closed };
                                let i2 = Interval { min: r2, min_type: BoundType::Closed, max: pos_inf(), max_type: BoundType::Open };
                                SolutionSet::Union(vec![i1, i2])
                            }
                        },
                        RelOp::Gt => {
                            if opens_up {
                                // Parabola > 0 outside roots: (-inf, r1) U (r2, inf)
                                let i1 = Interval { min: neg_inf(), min_type: BoundType::Open, max: r1, max_type: BoundType::Open };
                                let i2 = Interval { min: r2, min_type: BoundType::Open, max: pos_inf(), max_type: BoundType::Open };
                                SolutionSet::Union(vec![i1, i2])
                            } else {
                                // Parabola > 0 between roots: (r1, r2)
                                mk_interval(r1, BoundType::Open, r2, BoundType::Open)
                            }
                        },
                        RelOp::Geq => {
                            if opens_up {
                                // (-inf, r1] U [r2, inf)
                                let i1 = Interval { min: neg_inf(), min_type: BoundType::Open, max: r1, max_type: BoundType::Closed };
                                let i2 = Interval { min: r2, min_type: BoundType::Closed, max: pos_inf(), max_type: BoundType::Open };
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
                            let i1 = Interval { min: neg_inf(), min_type: BoundType::Open, max: r1.clone(), max_type: BoundType::Open };
                            let i2 = Interval { min: r1, min_type: BoundType::Open, max: pos_inf(), max_type: BoundType::Open };
                            SolutionSet::Union(vec![i1, i2])
                        },
                        RelOp::Lt => {
                            if opens_up {
                                // (x-r)^2 < 0 -> Empty
                                SolutionSet::Empty
                            } else {
                                // -(x-r)^2 < 0 -> All Reals except r
                                let i1 = Interval { min: neg_inf(), min_type: BoundType::Open, max: r1.clone(), max_type: BoundType::Open };
                                let i2 = Interval { min: r1, min_type: BoundType::Open, max: pos_inf(), max_type: BoundType::Open };
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
                                let i1 = Interval { min: neg_inf(), min_type: BoundType::Open, max: r1.clone(), max_type: BoundType::Open };
                                let i2 = Interval { min: r1, min_type: BoundType::Open, max: pos_inf(), max_type: BoundType::Open };
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

    fn apply(&self, eq: &Equation, var: &str, simplifier: &Simplifier) -> Option<Result<(SolutionSet, Vec<SolveStep>), CasError>> {
        // Isolation strategy expects variable on LHS.
        // The main solve loop handles swapping, but we should check here or just assume?
        // Let's check and swap if needed, or just rely on isolate to handle it?
        // isolate() assumes we are isolating FROM lhs.
        
        // If var is on RHS and not LHS, we should swap.
        // If var is on both, isolation might fail or we need to collect first.
        
        let lhs_has = contains_var(&eq.lhs, var);
        let rhs_has = contains_var(&eq.rhs, var);
        
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
                    equation_after: Equation { lhs: eq.rhs.clone(), rhs: eq.lhs.clone(), op: new_op.clone() },
                });
            }
            match isolate(&eq.rhs, &eq.lhs, new_op, var, simplifier) {
                Ok((set, mut iso_steps)) => {
                    steps.append(&mut iso_steps);
                    return Some(Ok((set, steps)));
                },
                Err(e) => return Some(Err(e)),
            }
        }
        
        // LHS has var
        match isolate(&eq.lhs, &eq.rhs, eq.op.clone(), var, simplifier) {
            Ok((set, steps)) => Some(Ok((set, steps))),
            Err(e) => Some(Err(e)),
        }
    }
}

