use cas_ast::{Expr, Equation, RelOp};
use std::rc::Rc;
use crate::engine::Simplifier;


#[derive(Debug, Clone)]
pub struct SolveStep {
    pub description: String,
    pub equation_after: Equation,
}

pub fn solve(eq: &Equation, var: &str, simplifier: &Simplifier) -> Result<Vec<(Equation, Vec<SolveStep>)>, String> {
    // We want to isolate 'var' on LHS.
    let mut steps = Vec::new();

    let lhs_has_var = contains_var(&eq.lhs, var);
    let rhs_has_var = contains_var(&eq.rhs, var);

    if !lhs_has_var && !rhs_has_var {
        return Err(format!("Variable '{}' not found in equation", var));
    }

    if !lhs_has_var && rhs_has_var {
        // Swap to make LHS have the variable
        let new_op = match eq.op {
            RelOp::Eq => RelOp::Eq,
            RelOp::Neq => RelOp::Neq,
            RelOp::Lt => RelOp::Gt,
            RelOp::Gt => RelOp::Lt,
            RelOp::Leq => RelOp::Geq,
            RelOp::Geq => RelOp::Leq,
        };
        let new_eq = Equation { lhs: eq.rhs.clone(), rhs: eq.lhs.clone(), op: new_op };
        if simplifier.collect_steps {
            steps.push(SolveStep {
                description: "Swap sides to put variable on LHS".to_string(),
                equation_after: new_eq.clone(),
            });
        }
        
        // Recursive call (tail recursion effectively)
        // We need to merge steps from recursive call
        let results = solve_internal(&new_eq, var, simplifier)?;
        
        // Prepend current steps to all results
        let mut final_results = Vec::new();
        for (res_eq, mut res_steps) in results {
            let mut combined_steps = steps.clone();
            combined_steps.append(&mut res_steps);
            final_results.push((res_eq, combined_steps));
        }
        return Ok(final_results);
    }

    if lhs_has_var && rhs_has_var {
        return Err("Variable appears on both sides. Please simplify/collect first.".to_string());
    }

    // Now LHS has var, RHS does not.
    let results = isolate(&eq.lhs, &eq.rhs, eq.op.clone(), var, simplifier)?;
    
    // Prepend current steps (empty here, but for consistency)
    let mut final_results = Vec::new();
    for (res_eq, mut res_steps) in results {
        let mut combined_steps = steps.clone();
        combined_steps.append(&mut res_steps);
        final_results.push((res_eq, combined_steps));
    }
    Ok(final_results)
}

// Internal helper to avoid re-checking var presence unnecessarily if we already know it
fn solve_internal(eq: &Equation, var: &str, simplifier: &Simplifier) -> Result<Vec<(Equation, Vec<SolveStep>)>, String> {
    // This is just a wrapper to call isolate directly since we know var is on LHS from the swap logic
    isolate(&eq.lhs, &eq.rhs, eq.op.clone(), var, simplifier)
}

fn isolate(lhs: &Rc<Expr>, rhs: &Rc<Expr>, op: RelOp, var: &str, simplifier: &Simplifier) -> Result<Vec<(Equation, Vec<SolveStep>)>, String> {
    let mut steps = Vec::new();
    
    match lhs.as_ref() {
        Expr::Variable(v) if v == var => {
            Ok(vec![(Equation { lhs: lhs.clone(), rhs: rhs.clone(), op }, steps)])
        }
        Expr::Add(l, r) => {
            // (A + B) = RHS
            if contains_var(l, var) {
                // A = RHS - B
                let new_rhs = Expr::sub(rhs.clone(), r.clone());
                let new_eq = Equation { lhs: l.clone(), rhs: new_rhs.clone(), op: op.clone() };
                if simplifier.collect_steps {
                    steps.push(SolveStep {
                        description: format!("Subtract {} from both sides", r),
                        equation_after: new_eq.clone(),
                    });
                }
                let results = isolate(l, &new_rhs, op, var, simplifier)?;
                prepend_steps(results, steps)
            } else {
                // B = RHS - A
                let new_rhs = Expr::sub(rhs.clone(), l.clone());
                let new_eq = Equation { lhs: r.clone(), rhs: new_rhs.clone(), op: op.clone() };
                if simplifier.collect_steps {
                    steps.push(SolveStep {
                        description: format!("Subtract {} from both sides", l),
                        equation_after: new_eq.clone(),
                    });
                }
                let results = isolate(r, &new_rhs, op, var, simplifier)?;
                prepend_steps(results, steps)
            }
        }
        Expr::Sub(l, r) => {
            // (A - B) = RHS
            if contains_var(l, var) {
                // A = RHS + B
                let new_rhs = Expr::add(rhs.clone(), r.clone());
                let new_eq = Equation { lhs: l.clone(), rhs: new_rhs.clone(), op: op.clone() };
                if simplifier.collect_steps {
                    steps.push(SolveStep {
                        description: format!("Add {} to both sides", r),
                        equation_after: new_eq.clone(),
                    });
                }
                let results = isolate(l, &new_rhs, op, var, simplifier)?;
                prepend_steps(results, steps)
            } else {
                // -B = RHS - A -> B = A - RHS
                // Multiply by -1 flips inequality
                let new_rhs = Expr::sub(l.clone(), rhs.clone());
                let new_op = flip_inequality(op);
                let new_eq = Equation { lhs: r.clone(), rhs: new_rhs.clone(), op: new_op.clone() };
                if simplifier.collect_steps {
                    steps.push(SolveStep {
                        description: format!("Move {} and multiply by -1 (flips inequality)", l),
                        equation_after: new_eq.clone(),
                    });
                }
                let results = isolate(r, &new_rhs, new_op, var, simplifier)?;
                prepend_steps(results, steps)
            }
        }
        Expr::Mul(l, r) => {
            // A * B = RHS
            if contains_var(l, var) {
                // A = RHS / B
                // Check if B is negative constant to flip inequality
                let mut new_op = op.clone();
                if is_negative(r) {
                    new_op = flip_inequality(new_op);
                }

                let new_rhs = Expr::div(rhs.clone(), r.clone());
                let new_eq = Equation { lhs: l.clone(), rhs: new_rhs.clone(), op: new_op.clone() };
                if simplifier.collect_steps {
                    steps.push(SolveStep {
                        description: format!("Divide both sides by {}", r),
                        equation_after: new_eq.clone(),
                    });
                }
                let results = isolate(l, &new_rhs, new_op, var, simplifier)?;
                prepend_steps(results, steps)
            } else {
                // B = RHS / A
                let mut new_op = op.clone();
                if is_negative(l) {
                    new_op = flip_inequality(new_op);
                }

                let new_rhs = Expr::div(rhs.clone(), l.clone());
                let new_eq = Equation { lhs: r.clone(), rhs: new_rhs.clone(), op: new_op.clone() };
                if simplifier.collect_steps {
                    steps.push(SolveStep {
                        description: format!("Divide both sides by {}", l),
                        equation_after: new_eq.clone(),
                    });
                }
                let results = isolate(r, &new_rhs, new_op, var, simplifier)?;
                prepend_steps(results, steps)
            }
        }
        Expr::Div(l, r) => {
            // A / B = RHS
            if contains_var(l, var) {
                // A = RHS * B
                // Check if B is negative constant to flip inequality
                let mut new_op = op.clone();
                if is_negative(r) {
                    new_op = flip_inequality(new_op);
                }

                let new_rhs = Expr::mul(rhs.clone(), r.clone());
                let new_eq = Equation { lhs: l.clone(), rhs: new_rhs.clone(), op: new_op.clone() };
                if simplifier.collect_steps {
                    steps.push(SolveStep {
                        description: format!("Multiply both sides by {}", r),
                        equation_after: new_eq.clone(),
                    });
                }
                let results = isolate(l, &new_rhs, new_op, var, simplifier)?;
                prepend_steps(results, steps)
            } else {
                // B = A / RHS
                // This is tricky for inequalities. 
                // 10 / x < 2
                // If x > 0: 10 < 2x -> 5 < x -> x > 5
                // If x < 0: 10 > 2x -> 5 > x -> x < 5
                // We assume x is positive for now or just standard algebraic manipulation?
                // Standard: x = A / RHS
                // But inequality direction depends on sign of RHS and x.
                // Let's assume standard equation logic for now, or warn about inequalities with variable in denominator.
                // For now, let's just implement standard isolation.
                
                let new_rhs = Expr::div(l.clone(), rhs.clone());
                let new_eq = Equation { lhs: r.clone(), rhs: new_rhs.clone(), op: op.clone() };
                if simplifier.collect_steps {
                    steps.push(SolveStep {
                        description: format!("Isolate denominator {}", r),
                        equation_after: new_eq.clone(),
                    });
                }
                let results = isolate(r, &new_rhs, op, var, simplifier)?;
                prepend_steps(results, steps)
            }
        }
        Expr::Pow(b, e) => {
            // B^E = RHS
            if contains_var(b, var) {
                // B = RHS^(1/E)
                let inv_exp = Expr::div(Expr::num(1), e.clone());
                let new_rhs = Expr::pow(rhs.clone(), inv_exp);
                let new_eq = Equation { lhs: b.clone(), rhs: new_rhs.clone(), op: op.clone() };
                if simplifier.collect_steps {
                    steps.push(SolveStep {
                        description: format!("Take {}-th root of both sides", e),
                        equation_after: new_eq.clone(),
                    });
                }
                let results = isolate(b, &new_rhs, op, var, simplifier)?;
                prepend_steps(results, steps)
            } else {
                // E = log(B, RHS)
                let new_rhs = Expr::log(b.clone(), rhs.clone());
                let new_eq = Equation { lhs: e.clone(), rhs: new_rhs.clone(), op: op.clone() };
                if simplifier.collect_steps {
                    steps.push(SolveStep {
                        description: format!("Take log base {} of both sides", b),
                        equation_after: new_eq.clone(),
                    });
                }
                let results = isolate(e, &new_rhs, op, var, simplifier)?;
                prepend_steps(results, steps)
            }
        }
        Expr::Function(name, args) => {
            if name == "abs" && args.len() == 1 {
                // |A| = B
                // Branch 1: A = B
                // Branch 2: A = -B
                // For inequalities:
                // |A| < B -> -B < A < B (Intersection) -> A < B AND A > -B
                // |A| > B -> A > B OR A < -B
                // Currently we return a list of equations.
                // For |A| = B, we return [A=B, A=-B].
                // For |A| < B, we return [A < B, A > -B] (Implicit AND? Or just list of constraints?)
                // For |A| > B, we return [A > B, A < -B] (Implicit OR?)
                // The solver returns a Vec of (Equation, Steps).
                // Interpretation of Vec depends on context.
                // For equations, it's OR (solutions).
                // For inequalities, it's context dependent.
                // Let's implement the standard splitting.
                
                let arg = &args[0];
                
                // Branch 1: Positive case
                let eq1 = Equation { lhs: arg.clone(), rhs: rhs.clone(), op: op.clone() };
                let mut steps1 = steps.clone();
                if simplifier.collect_steps {
                    steps1.push(SolveStep {
                        description: format!("Split absolute value (Case 1): {} {} {}", arg, op, rhs),
                        equation_after: eq1.clone(),
                    });
                }
                let results1 = isolate(arg, rhs, op.clone(), var, simplifier)?;
                let final_results1 = prepend_steps(results1, steps1);

                // Branch 2: Negative case
                // |A| < B -> A > -B (Flip op)
                // |A| > B -> A < -B (Flip op)
                // |A| = B -> A = -B
                
                let neg_rhs = Expr::neg(rhs.clone());
                let op2 = match op {
                    RelOp::Eq => RelOp::Eq,
                    RelOp::Neq => RelOp::Neq,
                    RelOp::Lt => RelOp::Gt,  // |x| < 5 -> x > -5
                    RelOp::Leq => RelOp::Geq,
                    RelOp::Gt => RelOp::Lt,  // |x| > 5 -> x < -5
                    RelOp::Geq => RelOp::Leq,
                };
                
                let eq2 = Equation { lhs: arg.clone(), rhs: neg_rhs.clone(), op: op2.clone() };
                let mut steps2 = steps.clone();
                if simplifier.collect_steps {
                    steps2.push(SolveStep {
                        description: format!("Split absolute value (Case 2): {} {} {}", arg, op2, neg_rhs),
                        equation_after: eq2.clone(),
                    });
                }
                let results2 = isolate(arg, &neg_rhs, op2, var, simplifier)?;
                let final_results2 = prepend_steps(results2, steps2);
                
                let mut all_results = final_results1?;
                all_results.extend(final_results2?);
                
                Ok(all_results)
            } else if name == "log" && args.len() == 2 {
                let base = &args[0];
                let arg = &args[1];
                
                if contains_var(arg, var) && !contains_var(base, var) {
                    // log(b, x) = RHS -> x = b^RHS
                    let new_rhs = Expr::pow(base.clone(), rhs.clone());
                    let new_eq = Equation { lhs: arg.clone(), rhs: new_rhs.clone(), op: op.clone() };
                    if simplifier.collect_steps {
                        steps.push(SolveStep {
                            description: format!("Exponentiate both sides with base {}", base),
                            equation_after: new_eq.clone(),
                        });
                    }
                    let results = isolate(arg, &new_rhs, op, var, simplifier)?;
                    prepend_steps(results, steps)
                } else if contains_var(base, var) && !contains_var(arg, var) {
                    let inv_rhs = Expr::div(Expr::num(1), rhs.clone());
                    let new_rhs = Expr::pow(arg.clone(), inv_rhs);
                    let new_eq = Equation { lhs: base.clone(), rhs: new_rhs.clone(), op: op.clone() };
                    if simplifier.collect_steps {
                        steps.push(SolveStep {
                            description: "Isolate base of logarithm".to_string(),
                            equation_after: new_eq.clone(),
                        });
                    }
                    let results = isolate(base, &new_rhs, op, var, simplifier)?;
                    prepend_steps(results, steps)
                } else {
                     Err(format!("Cannot isolate '{}' from log function", var))
                }
            } else if args.len() == 1 {
                let arg = &args[0];
                if contains_var(arg, var) {
                    match name.as_str() {
                        "ln" => {
                            let new_rhs = Expr::pow(Expr::e(), rhs.clone());
                            let new_eq = Equation { lhs: arg.clone(), rhs: new_rhs.clone(), op: op.clone() };
                            if simplifier.collect_steps {
                                steps.push(SolveStep {
                                    description: "Exponentiate both sides with base e".to_string(),
                                    equation_after: new_eq.clone(),
                                });
                            }
                            let results = isolate(arg, &new_rhs, op, var, simplifier)?;
                            prepend_steps(results, steps)
                        },
                        "exp" => {
                            let new_rhs = Expr::ln(rhs.clone());
                            let new_eq = Equation { lhs: arg.clone(), rhs: new_rhs.clone(), op: op.clone() };
                            if simplifier.collect_steps {
                                steps.push(SolveStep {
                                    description: "Take natural log of both sides".to_string(),
                                    equation_after: new_eq.clone(),
                                });
                            }
                            let results = isolate(arg, &new_rhs, op, var, simplifier)?;
                            prepend_steps(results, steps)
                        },
                        "sqrt" => {
                            let new_rhs = Expr::pow(rhs.clone(), Expr::num(2));
                            let new_eq = Equation { lhs: arg.clone(), rhs: new_rhs.clone(), op: op.clone() };
                            if simplifier.collect_steps {
                                steps.push(SolveStep {
                                    description: "Square both sides".to_string(),
                                    equation_after: new_eq.clone(),
                                });
                            }
                            let results = isolate(arg, &new_rhs, op, var, simplifier)?;
                            prepend_steps(results, steps)
                        },
                        "sin" => {
                            // sin(x) = y -> x = arcsin(y)
                            let new_rhs = Rc::new(Expr::Function("arcsin".to_string(), vec![rhs.clone()]));
                            let new_eq = Equation { lhs: arg.clone(), rhs: new_rhs.clone(), op: op.clone() };
                            if simplifier.collect_steps {
                                steps.push(SolveStep {
                                    description: "Take arcsin of both sides".to_string(),
                                    equation_after: new_eq.clone(),
                                });
                            }
                            
                            let (simplified_rhs, sim_steps) = simplify_rhs(new_rhs, arg.clone(), op.clone(), simplifier);
                            steps.extend(sim_steps);

                            let results = isolate(arg, &simplified_rhs, op, var, simplifier)?;
                            prepend_steps(results, steps)
                        },
                        "cos" => {
                            // cos(x) = y -> x = arccos(y)
                            let new_rhs = Rc::new(Expr::Function("arccos".to_string(), vec![rhs.clone()]));
                            let new_eq = Equation { lhs: arg.clone(), rhs: new_rhs.clone(), op: op.clone() };
                            if simplifier.collect_steps {
                                steps.push(SolveStep {
                                    description: "Take arccos of both sides".to_string(),
                                    equation_after: new_eq.clone(),
                                });
                            }

                            let (simplified_rhs, sim_steps) = simplify_rhs(new_rhs, arg.clone(), op.clone(), simplifier);
                            steps.extend(sim_steps);

                            let results = isolate(arg, &simplified_rhs, op, var, simplifier)?;
                            prepend_steps(results, steps)
                        },
                        "tan" => {
                            // tan(x) = y -> x = arctan(y)
                            let new_rhs = Rc::new(Expr::Function("arctan".to_string(), vec![rhs.clone()]));
                            let new_eq = Equation { lhs: arg.clone(), rhs: new_rhs.clone(), op: op.clone() };
                            if simplifier.collect_steps {
                                steps.push(SolveStep {
                                    description: "Take arctan of both sides".to_string(),
                                    equation_after: new_eq.clone(),
                                });
                            }

                            let (simplified_rhs, sim_steps) = simplify_rhs(new_rhs, arg.clone(), op.clone(), simplifier);
                            steps.extend(sim_steps);

                            let results = isolate(arg, &simplified_rhs, op, var, simplifier)?;
                            prepend_steps(results, steps)
                        },
                        _ => Err(format!("Cannot invert function '{}'", name)),
                    }
                } else {
                     Err(format!("Variable '{}' not found in function argument", var))
                }
            } else {
                 Err(format!("Cannot invert function '{}' with {} arguments", name, args.len()))
            }
        }
        _ => Err(format!("Cannot isolate '{}' from {:?}", var, lhs)),
    }
}

fn prepend_steps(
    results: Vec<(Equation, Vec<SolveStep>)>,
    steps: Vec<SolveStep>
) -> Result<Vec<(Equation, Vec<SolveStep>)>, String> {
    let mut final_results = Vec::new();
    for (res_eq, mut res_steps) in results {
        let mut combined_steps = steps.clone();
        combined_steps.append(&mut res_steps);
        final_results.push((res_eq, combined_steps));
    }
    Ok(final_results)
}

fn is_negative(expr: &Expr) -> bool {
    match expr {
        Expr::Number(n) => *n < num_rational::BigRational::from_integer(0.into()),
        Expr::Neg(_) => true, // Simple check, might be Neg(Neg(x))
        Expr::Mul(l, r) => is_negative(l) ^ is_negative(r),
        _ => false, // Conservative
    }
}

fn simplify_rhs(rhs: Rc<Expr>, lhs: Rc<Expr>, op: RelOp, simplifier: &Simplifier) -> (Rc<Expr>, Vec<SolveStep>) {
    let (simplified_rhs, sim_steps) = simplifier.simplify(rhs);
    let mut steps = Vec::new();

    if simplifier.collect_steps {
        for step in sim_steps {
            steps.push(SolveStep {
                description: step.description,
                equation_after: Equation {
                    lhs: lhs.clone(),
                    rhs: step.after, // This is correct, each step produces a new RHS
                    op: op.clone(),
                }
            });
        }
    }
    (simplified_rhs, steps)
}

pub fn contains_var(expr: &Rc<Expr>, var: &str) -> bool {
    match expr.as_ref() {
        Expr::Variable(v) => v == var,
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) | Expr::Pow(l, r) => {
            contains_var(l, var) || contains_var(r, var)
        }
        Expr::Neg(e) => contains_var(e, var),
        Expr::Function(_, args) => args.iter().any(|a| contains_var(a, var)),
        _ => false,
    }
}

fn flip_inequality(op: RelOp) -> RelOp {
    match op {
        RelOp::Eq => RelOp::Eq,
        RelOp::Neq => RelOp::Neq,
        RelOp::Lt => RelOp::Gt,
        RelOp::Gt => RelOp::Lt,
        RelOp::Leq => RelOp::Geq,
        RelOp::Geq => RelOp::Leq,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse; // Note: parse returns Expr, we need to construct Equation manually for tests or update parser
    // Since we updated parser but it's in another crate, we might need to use parse_statement if available or construct manually.
    // For unit tests inside cas_engine, we depend on cas_parser? Yes.
    
    // Helper to make equation from strings
    fn make_eq(lhs: &str, rhs: &str) -> Equation {
        Equation {
            lhs: parse(lhs).unwrap(),
            rhs: parse(rhs).unwrap(),
            op: RelOp::Eq,
        }
    }

    #[test]
    fn test_solve_linear() {
        // x + 2 = 5 -> x = 3
        let eq = make_eq("x + 2", "5");
        let mut simplifier = Simplifier::new();
        simplifier.collect_steps = true;
        let results = solve(&eq, "x", &simplifier).unwrap();
        assert_eq!(results.len(), 1);
        let (res, _) = &results[0];
        // Result: x = 5 - 2
        // We don't simplify automatically in solve, so RHS is "5 - 2".
        assert_eq!(format!("{}", res.lhs), "x");
        assert_eq!(format!("{}", res.rhs), "5 - 2");
    }

    #[test]
    fn test_solve_mul() {
        // 2 * x = 6 -> x = 6 / 2
        let eq = make_eq("2 * x", "6");
        let mut simplifier = Simplifier::new();
        simplifier.collect_steps = true;
        let results = solve(&eq, "x", &simplifier).unwrap();
        assert_eq!(results.len(), 1);
        let (res, _) = &results[0];
        assert_eq!(format!("{}", res.rhs), "6 / 2");
    }
    
    #[test]
    fn test_solve_pow() {
        // x^2 = 4 -> x = 4^(1/2)
        let eq = make_eq("x^2", "4");
        let mut simplifier = Simplifier::new();
        simplifier.collect_steps = true;
        let results = solve(&eq, "x", &simplifier).unwrap();
        assert_eq!(results.len(), 1);
        let (res, _) = &results[0];
        assert_eq!(format!("{}", res.rhs), "4^(1 / 2)");
    }
}
