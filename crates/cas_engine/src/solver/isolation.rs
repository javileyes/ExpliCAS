use cas_ast::{Expr, Equation, RelOp, SolutionSet, Interval, BoundType};
use std::rc::Rc;
use crate::engine::Simplifier;
use crate::solver::{SolveStep, solve};
use crate::solver::solution_set::{neg_inf, pos_inf, intersect_solution_sets, union_solution_sets};

use crate::error::CasError;

pub fn isolate(lhs: &Rc<Expr>, rhs: &Rc<Expr>, op: RelOp, var: &str, simplifier: &Simplifier) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    let mut steps = Vec::new();
    
    match lhs.as_ref() {
        Expr::Variable(v) if v == var => {
            // Simplify RHS before returning
            let (sim_rhs, _) = simplifier.simplify(rhs.clone());
            
            let set = match op {
                RelOp::Eq => SolutionSet::Discrete(vec![sim_rhs.clone()]),
                RelOp::Neq => {
                    // x != 5 -> (-inf, 5) U (5, inf)
                    let i1 = Interval { min: neg_inf(), min_type: BoundType::Open, max: sim_rhs.clone(), max_type: BoundType::Open };
                    let i2 = Interval { min: sim_rhs.clone(), min_type: BoundType::Open, max: pos_inf(), max_type: BoundType::Open };
                    SolutionSet::Union(vec![i1, i2])
                },
                RelOp::Lt => SolutionSet::Continuous(Interval {
                    min: neg_inf(), min_type: BoundType::Open,
                    max: sim_rhs.clone(), max_type: BoundType::Open
                }),
                RelOp::Gt => SolutionSet::Continuous(Interval {
                    min: sim_rhs.clone(), min_type: BoundType::Open,
                    max: pos_inf(), max_type: BoundType::Open
                }),
                RelOp::Leq => SolutionSet::Continuous(Interval {
                    min: neg_inf(), min_type: BoundType::Open,
                    max: sim_rhs.clone(), max_type: BoundType::Closed
                }),
                RelOp::Geq => SolutionSet::Continuous(Interval {
                    min: sim_rhs.clone(), min_type: BoundType::Closed,
                    max: pos_inf(), max_type: BoundType::Open
                }),
            };
            Ok((set, steps))
        }
        Expr::Add(l, r) => {
            // (A + B) = RHS
            if contains_var(l, var) {
                // A = RHS - B
                let new_rhs = Expr::sub(rhs.clone(), r.clone());
                let (sim_rhs, _) = simplifier.simplify(new_rhs.clone());
                let new_eq = Equation { lhs: l.clone(), rhs: new_rhs.clone(), op: op.clone() };
                if simplifier.collect_steps {
                    steps.push(SolveStep {
                        description: format!("Subtract {} from both sides", r),
                        equation_after: new_eq.clone(),
                    });
                }
                let results = isolate(l, &sim_rhs, op, var, simplifier)?;
                prepend_steps(results, steps)
            } else {
                // B = RHS - A
                let new_rhs = Expr::sub(rhs.clone(), l.clone());
                let (sim_rhs, _) = simplifier.simplify(new_rhs.clone());
                let new_eq = Equation { lhs: r.clone(), rhs: new_rhs.clone(), op: op.clone() };
                if simplifier.collect_steps {
                    steps.push(SolveStep {
                        description: format!("Subtract {} from both sides", l),
                        equation_after: new_eq.clone(),
                    });
                }
                let results = isolate(r, &sim_rhs, op, var, simplifier)?;
                prepend_steps(results, steps)
            }
        }
        Expr::Sub(l, r) => {
            // (A - B) = RHS
            if contains_var(l, var) {
                // A = RHS + B
                let new_rhs = Expr::add(rhs.clone(), r.clone());
                let (sim_rhs, _) = simplifier.simplify(new_rhs.clone());
                let new_eq = Equation { lhs: l.clone(), rhs: new_rhs.clone(), op: op.clone() };
                if simplifier.collect_steps {
                    steps.push(SolveStep {
                        description: format!("Add {} to both sides", r),
                        equation_after: new_eq.clone(),
                    });
                }
                let results = isolate(l, &sim_rhs, op, var, simplifier)?;
                prepend_steps(results, steps)
            } else {
                // -B = RHS - A -> B = A - RHS
                // Multiply by -1 flips inequality
                let new_rhs = Expr::sub(l.clone(), rhs.clone());
                let (sim_rhs, _) = simplifier.simplify(new_rhs.clone());
                let new_op = flip_inequality(op);
                let new_eq = Equation { lhs: r.clone(), rhs: new_rhs.clone(), op: new_op.clone() };
                if simplifier.collect_steps {
                    steps.push(SolveStep {
                        description: format!("Move {} and multiply by -1 (flips inequality)", l),
                        equation_after: new_eq.clone(),
                    });
                }
                let results = isolate(r, &sim_rhs, new_op, var, simplifier)?;
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
                if contains_var(r, var) && matches!(op, RelOp::Lt | RelOp::Gt | RelOp::Leq | RelOp::Geq) {
                    // Denominator contains variable. Split into cases.
                    // Case 1: Denominator > 0
                    let op_pos = op.clone();
                    let new_rhs = Expr::mul(rhs.clone(), r.clone());
                    let (sim_rhs, _) = simplifier.simplify(new_rhs);
                    
                    if simplifier.collect_steps {
                        steps.push(SolveStep {
                            description: format!("Case 1: Assume {} > 0. Multiply by positive denominator.", r),
                            equation_after: Equation { lhs: l.clone(), rhs: sim_rhs.clone(), op: op_pos.clone() }
                        });
                    }
                    
                    let results_pos = isolate(l, &sim_rhs, op_pos, var, simplifier)?;
                    let (set_pos, steps_pos) = prepend_steps(results_pos, steps.clone())?;
                    
                    // Domain: r > 0
                    let domain_eq = Equation { lhs: r.clone(), rhs: Expr::num(0), op: RelOp::Gt };
                    let (domain_pos_set, _) = solve(&domain_eq, var, simplifier)?; // Solve r > 0
                    let final_pos = intersect_solution_sets(set_pos, domain_pos_set);
                    
                    // Case 2: Denominator < 0
                    let op_neg = flip_inequality(op.clone());
                    // new_rhs is same
                    let (sim_rhs, _) = simplifier.simplify(Expr::mul(rhs.clone(), r.clone()));
                    
                    if simplifier.collect_steps {
                        steps.push(SolveStep {
                            description: format!("Case 2: Assume {} < 0. Multiply by negative denominator (flips inequality).", r),
                            equation_after: Equation { lhs: l.clone(), rhs: sim_rhs.clone(), op: op_neg.clone() }
                        });
                    }
                    
                    let results_neg = isolate(l, &sim_rhs, op_neg, var, simplifier)?;
                    let (set_neg, steps_neg) = prepend_steps(results_neg, steps.clone())?;
                    
                    // Domain: r < 0
                    let domain_eq_neg = Equation { lhs: r.clone(), rhs: Expr::num(0), op: RelOp::Lt };
                    let (domain_neg_set, _) = solve(&domain_eq_neg, var, simplifier)?; // Solve r < 0
                    let final_neg = intersect_solution_sets(set_neg, domain_neg_set);
                    
                    // Combine
                    let final_set = union_solution_sets(final_pos, final_neg);
                    
                    // Combine steps
                    let mut all_steps = steps_pos;
                    all_steps.push(SolveStep { description: "--- End of Case 1 ---".to_string(), equation_after: Equation { lhs: l.clone(), rhs: sim_rhs.clone(), op: op.clone() }});
                    all_steps.extend(steps_neg);
                    
                    return Ok((final_set, all_steps));

                } else {
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
                }
            } else {
                // B = A / RHS
                let new_rhs = Expr::div(l.clone(), rhs.clone());
                let (sim_rhs, _) = simplifier.simplify(new_rhs.clone());
                let new_eq = Equation { lhs: r.clone(), rhs: new_rhs.clone(), op: op.clone() };
                
                // Check if denominator is just the variable (simple case)
                if let Expr::Variable(v) = r.as_ref() {
                    if v == var && matches!(op, RelOp::Lt | RelOp::Gt | RelOp::Leq | RelOp::Geq) {
                         // Split into x > 0 and x < 0
                         
                         // Case 1: x > 0. Multiply by x (positive) -> Inequality direction preserved for 1 < 2x?
                         // Wait. 1/x < 2.
                         // x > 0: 1 < 2x -> 1/2 < x -> x > 1/2.
                         // Original op <. Result >.
                         // So if we isolate x from 1/2 < x, we get x > 1/2.
                         // But we are calling isolate(x, 1/2, op).
                         // If we pass op (<), we get x < 1/2.
                         // So we need to pass FLIPPED op for x > 0 case?
                         // 1/x < 2. x>0. 1 < 2x. 1/2 < x. x > 1/2.
                         // isolate(x, 1/2, >) -> x > 1/2.
                         // So yes, flip op for x > 0.
                         
                         let op_pos = flip_inequality(op.clone());
                         
                         // Add step for Case 1
                         if simplifier.collect_steps {
                             steps.push(SolveStep {
                                 description: format!("Case 1: Assume {} > 0. Multiply by {} (positive). Inequality direction preserved (flipped from isolation logic).", r, r),
                                 equation_after: Equation { lhs: r.clone(), rhs: sim_rhs.clone(), op: op_pos.clone() } // Showing the isolated form roughly
                             });
                         }

                         let results_pos = isolate(r, &sim_rhs, op_pos, var, simplifier)?;
                         let (set_pos, steps_pos) = prepend_steps(results_pos, steps.clone())?;
                         
                         // Intersect with (0, inf)
                         let domain_pos = SolutionSet::Continuous(Interval {
                             min: Expr::num(0), min_type: BoundType::Open,
                             max: pos_inf(), max_type: BoundType::Open
                         });
                         let final_pos = intersect_solution_sets(set_pos, domain_pos);
                         
                         // Case 2: x < 0. Multiply by x (negative) -> Inequality flips.
                         // 1/x < 2. x < 0. 1 > 2x. 1/2 > x. x < 1/2.
                         // isolate(x, 1/2, <) -> x < 1/2.
                         // So pass ORIGINAL op for x < 0.
                         
                         let op_neg = op.clone();
                         
                         // Add step for Case 2
                         if simplifier.collect_steps {
                             steps.push(SolveStep {
                                 description: format!("Case 2: Assume {} < 0. Multiply by {} (negative). Inequality flips.", r, r),
                                 equation_after: Equation { lhs: r.clone(), rhs: sim_rhs.clone(), op: op_neg.clone() }
                             });
                         }

                         let results_neg = isolate(r, &sim_rhs, op_neg, var, simplifier)?;
                         let (set_neg, steps_neg) = prepend_steps(results_neg, steps.clone())?;
                         
                         // Intersect with (-inf, 0)
                         let domain_neg = SolutionSet::Continuous(Interval {
                             min: neg_inf(), min_type: BoundType::Open,
                             max: Expr::num(0), max_type: BoundType::Open
                         });
                         let final_neg = intersect_solution_sets(set_neg, domain_neg);
                         
                         // Union
                         let final_set = union_solution_sets(final_pos, final_neg);
                         
                         // Combine steps: We should probably show them sequentially or structured.
                         // For now, appending them is okay, but maybe we can add a separator step?
                         let mut all_steps = steps_pos;
                         all_steps.push(SolveStep { description: "--- End of Case 1 ---".to_string(), equation_after: Equation { lhs: r.clone(), rhs: sim_rhs.clone(), op: op.clone() }}); // Dummy eq
                         all_steps.extend(steps_neg);
                         
                         return Ok((final_set, all_steps));
                    }
                }

                if simplifier.collect_steps {
                    steps.push(SolveStep {
                        description: format!("Isolate denominator {}", r),
                        equation_after: new_eq.clone(),
                    });
                }
                let results = isolate(r, &sim_rhs, op, var, simplifier)?;
                prepend_steps(results, steps)
            }
        }
        Expr::Pow(b, e) => {
            // B^E = RHS
            if contains_var(b, var) {
                // Check if exponent is an even integer
                let is_even = if let Some(n) = crate::solver::solution_set::get_number(e) {
                    n.is_integer() && (n.to_integer() % 2 == 0.into())
                } else {
                    false
                };

                if is_even {
                    // Check if RHS is negative
                    if is_negative(rhs) {
                        let result = match op {
                            RelOp::Eq => SolutionSet::Empty,
                            RelOp::Gt | RelOp::Geq | RelOp::Neq => SolutionSet::AllReals,
                            RelOp::Lt | RelOp::Leq => SolutionSet::Empty,
                        };
                         if simplifier.collect_steps {
                            steps.push(SolveStep {
                                description: format!("Even power cannot be negative ({} {} {})", b, op, rhs),
                                equation_after: Equation { lhs: lhs.clone(), rhs: rhs.clone(), op: op.clone() }, // No change
                            });
                        }
                        return Ok((result, steps));
                    }

                     // B^E = RHS -> |B| = RHS^(1/E)
                     let inv_exp = Expr::div(Expr::num(1), e.clone());
                     let new_rhs = Expr::pow(rhs.clone(), inv_exp);
                     
                     // Construct |B|
                     let abs_b = Rc::new(Expr::Function("abs".to_string(), vec![b.clone()]));
                     
                     let new_eq = Equation { lhs: abs_b.clone(), rhs: new_rhs.clone(), op: op.clone() };
                     if simplifier.collect_steps {
                        steps.push(SolveStep {
                            description: format!("Take {}-th root of both sides (even root implies absolute value)", e),
                            equation_after: new_eq.clone(),
                        });
                     }
                     
                     // Isolate |B|
                     // Note: We pass 'op' as is. |B| < RHS will be handled by isolate(|B|...) logic.
                     let results = isolate(&abs_b, &new_rhs, op, var, simplifier)?;
                     prepend_steps(results, steps)
                } else {
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
                    
                    // Check if exponent is negative to flip inequality
                    let mut new_op = op.clone();
                    if is_negative(e) {
                         new_op = flip_inequality(new_op);
                    }
                    
                    let results = isolate(b, &new_rhs, new_op, var, simplifier)?;
                    prepend_steps(results, steps)
                }
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
                // |A| < B -> -B < A < B (Intersection)
                // |A| > B -> A > B OR A < -B (Union)
                
                let arg = &args[0];
                
                // Branch 1: Positive case (A op B)
                let eq1 = Equation { lhs: arg.clone(), rhs: rhs.clone(), op: op.clone() };
                let mut steps1 = steps.clone();
                if simplifier.collect_steps {
                    steps1.push(SolveStep {
                        description: format!("Split absolute value (Case 1): {} {} {}", arg, op, rhs),
                        equation_after: eq1.clone(),
                    });
                }
                let results1 = isolate(arg, rhs, op.clone(), var, simplifier)?;
                let (set1, steps1_out) = prepend_steps(results1, steps1)?;

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
                let (set2, steps2_out) = prepend_steps(results2, steps2)?;
                
                // Combine sets
                let final_set = match op {
                    RelOp::Eq | RelOp::Neq | RelOp::Gt | RelOp::Geq => union_solution_sets(set1, set2),
                    RelOp::Lt | RelOp::Leq => intersect_solution_sets(set1, set2),
                };
                
                // Combine steps (just append for now, maybe separate them?)
                let mut all_steps = steps1_out;
                all_steps.extend(steps2_out);
                
                Ok((final_set, all_steps))
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
                     Err(CasError::IsolationError(var.to_string(), "Cannot isolate from log function".to_string()))
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
                        _ => Err(CasError::UnknownFunction(name.clone())),
                    }
                } else {
                     Err(CasError::VariableNotFound(var.to_string()))
                }
            } else {
                 Err(CasError::IsolationError(var.to_string(), format!("Cannot invert function '{}' with {} arguments", name, args.len())))
            }
        }
        _ => Err(CasError::IsolationError(var.to_string(), format!("Cannot isolate from {:?}", lhs))),
    }
}

pub fn prepend_steps(
    (set, mut res_steps): (SolutionSet, Vec<SolveStep>),
    mut steps: Vec<SolveStep>
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    steps.append(&mut res_steps);
    Ok((set, steps))
}

pub fn is_negative(expr: &Expr) -> bool {
    match expr {
        Expr::Number(n) => *n < num_rational::BigRational::from_integer(0.into()),
        Expr::Neg(_) => true, // Simple check, might be Neg(Neg(x))
        Expr::Mul(l, r) => is_negative(l) ^ is_negative(r),
        _ => false, // Conservative
    }
}

pub fn simplify_rhs(rhs: Rc<Expr>, lhs: Rc<Expr>, op: RelOp, simplifier: &Simplifier) -> (Rc<Expr>, Vec<SolveStep>) {
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

pub fn flip_inequality(op: RelOp) -> RelOp {
    match op {
        RelOp::Eq => RelOp::Eq,
        RelOp::Neq => RelOp::Neq,
        RelOp::Lt => RelOp::Gt,
        RelOp::Gt => RelOp::Lt,
        RelOp::Leq => RelOp::Geq,
        RelOp::Geq => RelOp::Leq,
    }
}
