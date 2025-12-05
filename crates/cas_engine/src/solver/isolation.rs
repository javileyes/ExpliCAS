use crate::engine::Simplifier;
use crate::solver::solution_set::{intersect_solution_sets, neg_inf, pos_inf, union_solution_sets};
use crate::solver::{solve, SolveStep};
use cas_ast::{BoundType, Context, Equation, Expr, ExprId, Interval, RelOp, SolutionSet};
use num_traits::Zero;

use crate::error::CasError;

pub fn isolate(
    lhs: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    simplifier: &mut Simplifier,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    let mut steps = Vec::new();

    let lhs_expr = simplifier.context.get(lhs).clone();

    match lhs_expr {
        Expr::Variable(v) if v == var => {
            // Simplify RHS before returning
            let (sim_rhs, _) = simplifier.simplify(rhs);

            let set = match op {
                RelOp::Eq => SolutionSet::Discrete(vec![sim_rhs]),
                RelOp::Neq => {
                    // x != 5 -> (-inf, 5) U (5, inf)
                    let i1 = Interval {
                        min: neg_inf(&mut simplifier.context),
                        min_type: BoundType::Open,
                        max: sim_rhs,
                        max_type: BoundType::Open,
                    };
                    let i2 = Interval {
                        min: sim_rhs,
                        min_type: BoundType::Open,
                        max: pos_inf(&mut simplifier.context),
                        max_type: BoundType::Open,
                    };
                    SolutionSet::Union(vec![i1, i2])
                }
                RelOp::Lt => SolutionSet::Continuous(Interval {
                    min: neg_inf(&mut simplifier.context),
                    min_type: BoundType::Open,
                    max: sim_rhs,
                    max_type: BoundType::Open,
                }),
                RelOp::Gt => SolutionSet::Continuous(Interval {
                    min: sim_rhs,
                    min_type: BoundType::Open,
                    max: pos_inf(&mut simplifier.context),
                    max_type: BoundType::Open,
                }),
                RelOp::Leq => SolutionSet::Continuous(Interval {
                    min: neg_inf(&mut simplifier.context),
                    min_type: BoundType::Open,
                    max: sim_rhs,
                    max_type: BoundType::Closed,
                }),
                RelOp::Geq => SolutionSet::Continuous(Interval {
                    min: sim_rhs,
                    min_type: BoundType::Closed,
                    max: pos_inf(&mut simplifier.context),
                    max_type: BoundType::Open,
                }),
            };
            Ok((set, steps))
        }
        Expr::Add(l, r) => {
            // (A + B) = RHS
            if contains_var(&simplifier.context, l, var) {
                // A = RHS - B
                let new_rhs = simplifier.context.add(Expr::Sub(rhs, r));
                let (sim_rhs, _) = simplifier.simplify(new_rhs);
                let new_eq = Equation {
                    lhs: l,
                    rhs: new_rhs,
                    op: op.clone(),
                };
                if simplifier.collect_steps {
                    steps.push(SolveStep {
                        description: format!(
                            "Subtract {} from both sides",
                            cas_ast::DisplayExpr {
                                context: &simplifier.context,
                                id: r
                            }
                        ),
                        equation_after: new_eq.clone(),
                    });
                }
                let results = isolate(l, sim_rhs, op, var, simplifier)?;
                prepend_steps(results, steps)
            } else {
                // B = RHS - A
                let new_rhs = simplifier.context.add(Expr::Sub(rhs, l));
                let (sim_rhs, _) = simplifier.simplify(new_rhs);
                let new_eq = Equation {
                    lhs: r,
                    rhs: new_rhs,
                    op: op.clone(),
                };
                if simplifier.collect_steps {
                    steps.push(SolveStep {
                        description: format!(
                            "Subtract {} from both sides",
                            cas_ast::DisplayExpr {
                                context: &simplifier.context,
                                id: l
                            }
                        ),
                        equation_after: new_eq.clone(),
                    });
                }
                let results = isolate(r, sim_rhs, op, var, simplifier)?;
                prepend_steps(results, steps)
            }
        }
        Expr::Sub(l, r) => {
            // (A - B) = RHS
            if contains_var(&simplifier.context, l, var) {
                // A = RHS + B
                let new_rhs = simplifier.context.add(Expr::Add(rhs, r));
                let (sim_rhs, _) = simplifier.simplify(new_rhs);
                let new_eq = Equation {
                    lhs: l,
                    rhs: new_rhs,
                    op: op.clone(),
                };
                if simplifier.collect_steps {
                    steps.push(SolveStep {
                        description: format!(
                            "Add {} to both sides",
                            cas_ast::DisplayExpr {
                                context: &simplifier.context,
                                id: r
                            }
                        ),
                        equation_after: new_eq.clone(),
                    });
                }
                let results = isolate(l, sim_rhs, op, var, simplifier)?;
                prepend_steps(results, steps)
            } else {
                // -B = RHS - A -> B = A - RHS
                // Multiply by -1 flips inequality
                let new_rhs = simplifier.context.add(Expr::Sub(l, rhs));
                let (sim_rhs, _) = simplifier.simplify(new_rhs);
                let new_op = flip_inequality(op);
                let new_eq = Equation {
                    lhs: r,
                    rhs: new_rhs,
                    op: new_op.clone(),
                };
                if simplifier.collect_steps {
                    steps.push(SolveStep {
                        description: format!(
                            "Move {} and multiply by -1 (flips inequality)",
                            cas_ast::DisplayExpr {
                                context: &simplifier.context,
                                id: l
                            }
                        ),
                        equation_after: new_eq.clone(),
                    });
                }
                let results = isolate(r, sim_rhs, new_op, var, simplifier)?;
                prepend_steps(results, steps)
            }
        }
        Expr::Mul(l, r) => {
            // A * B = RHS

            // CRITICAL: For inequalities with products, need sign analysis
            // if both factors contain variables or if RHS != 0
            let both_have_var = contains_var(&simplifier.context, l, var)
                && contains_var(&simplifier.context, r, var);
            let is_inequality = matches!(op, RelOp::Lt | RelOp::Gt | RelOp::Leq | RelOp::Geq);

            // Check if RHS is zero - special case for sign analysis
            let rhs_is_zero = matches!(simplifier.context.get(rhs), Expr::Number(n) if n.is_zero());

            if both_have_var && is_inequality && rhs_is_zero {
                // Product inequality: A * B > 0 (or < 0, etc.)
                // For A * B > 0: need (A > 0 AND B > 0) OR (A < 0 AND B < 0)
                // For A * B < 0: need (A > 0 AND B < 0) OR (A < 0 AND B > 0)

                let zero = simplifier.context.num(0);

                match op {
                    RelOp::Gt | RelOp::Geq => {
                        // A * B > 0: Both same sign
                        // Case 1: Both positive
                        let eq_a_pos = Equation {
                            lhs: l,
                            rhs: zero,
                            op: if matches!(op, RelOp::Gt) {
                                RelOp::Gt
                            } else {
                                RelOp::Geq
                            },
                        };
                        let eq_b_pos = Equation {
                            lhs: r,
                            rhs: zero,
                            op: if matches!(op, RelOp::Gt) {
                                RelOp::Gt
                            } else {
                                RelOp::Geq
                            },
                        };

                        let (set_a_pos, _) = solve(&eq_a_pos, var, simplifier)?;
                        let (set_b_pos, _) = solve(&eq_b_pos, var, simplifier)?;
                        let case_pos =
                            intersect_solution_sets(&simplifier.context, set_a_pos, set_b_pos);

                        // Case 2: Both negative
                        let eq_a_neg = Equation {
                            lhs: l,
                            rhs: zero,
                            op: if matches!(op, RelOp::Gt) {
                                RelOp::Lt
                            } else {
                                RelOp::Leq
                            },
                        };
                        let eq_b_neg = Equation {
                            lhs: r,
                            rhs: zero,
                            op: if matches!(op, RelOp::Gt) {
                                RelOp::Lt
                            } else {
                                RelOp::Leq
                            },
                        };

                        let (set_a_neg, _) = solve(&eq_a_neg, var, simplifier)?;
                        let (set_b_neg, _) = solve(&eq_b_neg, var, simplifier)?;
                        let case_neg =
                            intersect_solution_sets(&simplifier.context, set_a_neg, set_b_neg);

                        let final_set =
                            union_solution_sets(&simplifier.context, case_pos, case_neg);

                        return Ok((final_set, steps));
                    }
                    RelOp::Lt | RelOp::Leq => {
                        // A * B < 0: Different signs
                        // Case 1: A positive, B negative
                        let eq_a_pos = Equation {
                            lhs: l,
                            rhs: zero,
                            op: if matches!(op, RelOp::Lt) {
                                RelOp::Gt
                            } else {
                                RelOp::Geq
                            },
                        };
                        let eq_b_neg = Equation {
                            lhs: r,
                            rhs: zero,
                            op: if matches!(op, RelOp::Lt) {
                                RelOp::Lt
                            } else {
                                RelOp::Leq
                            },
                        };

                        let (set_a_pos, _) = solve(&eq_a_pos, var, simplifier)?;
                        let (set_b_neg, _) = solve(&eq_b_neg, var, simplifier)?;
                        let case1 =
                            intersect_solution_sets(&simplifier.context, set_a_pos, set_b_neg);

                        // Case 2: A negative, B positive
                        let eq_a_neg = Equation {
                            lhs: l,
                            rhs: zero,
                            op: if matches!(op, RelOp::Lt) {
                                RelOp::Lt
                            } else {
                                RelOp::Leq
                            },
                        };
                        let eq_b_pos = Equation {
                            lhs: r,
                            rhs: zero,
                            op: if matches!(op, RelOp::Lt) {
                                RelOp::Gt
                            } else {
                                RelOp::Geq
                            },
                        };

                        let (set_a_neg, _) = solve(&eq_a_neg, var, simplifier)?;
                        let (set_b_pos, _) = solve(&eq_b_pos, var, simplifier)?;
                        let case2 =
                            intersect_solution_sets(&simplifier.context, set_a_neg, set_b_pos);

                        let final_set = union_solution_sets(&simplifier.context, case1, case2);

                        return Ok((final_set, steps));
                    }
                    _ => {
                        // Equality - fall through to regular division
                    }
                }
            }

            // Default behavior: divide by one factor (for equations or when only one has var)
            if contains_var(&simplifier.context, l, var) {
                // A = RHS / B
                // Check if B is negative constant to flip inequality
                let mut new_op = op.clone();
                if is_negative(&simplifier.context, r) {
                    new_op = flip_inequality(new_op);
                }

                let new_rhs = simplifier.context.add(Expr::Div(rhs, r));
                let new_eq = Equation {
                    lhs: l,
                    rhs: new_rhs,
                    op: new_op.clone(),
                };
                if simplifier.collect_steps {
                    steps.push(SolveStep {
                        description: format!(
                            "Divide both sides by {}",
                            cas_ast::DisplayExpr {
                                context: &simplifier.context,
                                id: r
                            }
                        ),
                        equation_after: new_eq.clone(),
                    });
                }
                let results = isolate(l, new_rhs, new_op, var, simplifier)?;
                prepend_steps(results, steps)
            } else {
                // B = RHS / A
                let mut new_op = op.clone();
                if is_negative(&simplifier.context, l) {
                    new_op = flip_inequality(new_op);
                }

                let new_rhs = simplifier.context.add(Expr::Div(rhs, l));
                let new_eq = Equation {
                    lhs: r,
                    rhs: new_rhs,
                    op: new_op.clone(),
                };
                if simplifier.collect_steps {
                    steps.push(SolveStep {
                        description: format!(
                            "Divide both sides by {}",
                            cas_ast::DisplayExpr {
                                context: &simplifier.context,
                                id: l
                            }
                        ),
                        equation_after: new_eq.clone(),
                    });
                }
                let results = isolate(r, new_rhs, new_op, var, simplifier)?;
                prepend_steps(results, steps)
            }
        }
        Expr::Div(l, r) => {
            // A / B = RHS
            if contains_var(&simplifier.context, l, var) {
                if contains_var(&simplifier.context, r, var)
                    && matches!(op, RelOp::Lt | RelOp::Gt | RelOp::Leq | RelOp::Geq)
                {
                    // Denominator contains variable. Split into cases.
                    // Case 1: Denominator > 0
                    let op_pos = op.clone();
                    let new_rhs = simplifier.context.add(Expr::Mul(rhs, r));
                    let (sim_rhs, _) = simplifier.simplify(new_rhs);

                    if simplifier.collect_steps {
                        steps.push(SolveStep {
                            description: format!(
                                "Case 1: Assume {} > 0. Multiply by positive denominator.",
                                cas_ast::DisplayExpr {
                                    context: &simplifier.context,
                                    id: r
                                }
                            ),
                            equation_after: Equation {
                                lhs: l,
                                rhs: sim_rhs,
                                op: op_pos.clone(),
                            },
                        });
                    }

                    let results_pos = isolate(l, sim_rhs, op_pos, var, simplifier)?;
                    let (set_pos, steps_pos) = prepend_steps(results_pos, steps.clone())?;

                    // Domain: r > 0
                    let zero = simplifier.context.num(0);
                    let domain_eq = Equation {
                        lhs: r,
                        rhs: zero,
                        op: RelOp::Gt,
                    };
                    let (domain_pos_set, _) = solve(&domain_eq, var, simplifier)?; // Solve r > 0
                    let final_pos =
                        intersect_solution_sets(&simplifier.context, set_pos, domain_pos_set);

                    // Case 2: Denominator < 0
                    let op_neg = flip_inequality(op.clone());
                    // new_rhs is same
                    let temp_rhs = simplifier.context.add(Expr::Mul(rhs, r));
                    let (sim_rhs, _) = simplifier.simplify(temp_rhs);

                    if simplifier.collect_steps {
                        steps.push(SolveStep {
                            description: format!("Case 2: Assume {} < 0. Multiply by negative denominator (flips inequality).",
                                cas_ast::DisplayExpr { context: &simplifier.context, id: r }),
                            equation_after: Equation { lhs: l, rhs: sim_rhs, op: op_neg.clone() }
                        });
                    }

                    let results_neg = isolate(l, sim_rhs, op_neg, var, simplifier)?;
                    let (set_neg, steps_neg) = prepend_steps(results_neg, steps.clone())?;

                    // Domain: r < 0
                    let zero = simplifier.context.num(0);
                    let domain_eq_neg = Equation {
                        lhs: r,
                        rhs: zero,
                        op: RelOp::Lt,
                    };
                    let (domain_neg_set, _) = solve(&domain_eq_neg, var, simplifier)?; // Solve r < 0
                    let final_neg =
                        intersect_solution_sets(&simplifier.context, set_neg, domain_neg_set);

                    // Combine
                    let final_set = union_solution_sets(&simplifier.context, final_pos, final_neg);

                    // Combine steps
                    let mut all_steps = steps_pos;
                    all_steps.push(SolveStep {
                        description: "--- End of Case 1 ---".to_string(),
                        equation_after: Equation {
                            lhs: l,
                            rhs: sim_rhs,
                            op: op.clone(),
                        },
                    });
                    all_steps.extend(steps_neg);

                    return Ok((final_set, all_steps));
                } else {
                    // A = RHS * B
                    // Check if B is negative constant to flip inequality
                    let mut new_op = op.clone();
                    if is_negative(&simplifier.context, r) {
                        new_op = flip_inequality(new_op);
                    }

                    let new_rhs = simplifier.context.add(Expr::Mul(rhs, r));
                    let new_eq = Equation {
                        lhs: l,
                        rhs: new_rhs,
                        op: new_op.clone(),
                    };
                    if simplifier.collect_steps {
                        steps.push(SolveStep {
                            description: format!(
                                "Multiply both sides by {}",
                                cas_ast::DisplayExpr {
                                    context: &simplifier.context,
                                    id: r
                                }
                            ),
                            equation_after: new_eq.clone(),
                        });
                    }
                    let results = isolate(l, new_rhs, new_op, var, simplifier)?;
                    prepend_steps(results, steps)
                }
            } else {
                // B = A / RHS
                // CRITICAL FIX: Check if RHS is zero to avoid creating undefined (1/0)
                // This happens for inequalities like 1/x > 0 where RHS=0
                let is_rhs_zero =
                    matches!(simplifier.context.get(rhs), Expr::Number(n) if n.is_zero());

                let sim_rhs = if is_rhs_zero {
                    // Division by zero case: A/0
                    // The result depends on the sign of A:
                    // - If A > 0: result is +infinity
                    // - If A < 0: result is -infinity
                    // For now, we'll use pos_inf since this typically comes from cases like 1/x > 0
                    // where we're multiplying by x to get x > 1/0
                    pos_inf(&mut simplifier.context)
                } else {
                    let new_rhs = simplifier.context.add(Expr::Div(l, rhs));
                    let (simplified, _) = simplifier.simplify(new_rhs);
                    simplified
                };

                let new_eq = Equation {
                    lhs: r,
                    rhs: sim_rhs, // Use sim_rhs directly, not new_rhs
                    op: op.clone(),
                };

                // Check if denominator is just the variable (simple case)
                if let Expr::Variable(v) = simplifier.context.get(r) {
                    if v == var && matches!(op, RelOp::Lt | RelOp::Gt | RelOp::Leq | RelOp::Geq) {
                        // Split into x > 0 and x < 0

                        let op_pos = flip_inequality(op.clone());

                        // Add step for Case 1
                        if simplifier.collect_steps {
                            steps.push(SolveStep {
                                 description: format!("Case 1: Assume {} > 0. Multiply by {} (positive). Inequality direction preserved (flipped from isolation logic).",
                                  cas_ast::DisplayExpr { context: &simplifier.context, id: r },
                                  cas_ast::DisplayExpr { context: &simplifier.context, id: r }),
                                 equation_after: Equation { lhs: r, rhs: sim_rhs, op: op_pos.clone() } // Showing the isolated form roughly
                             });
                        }

                        let results_pos = isolate(r, sim_rhs, op_pos, var, simplifier)?;
                        let (set_pos, steps_pos) = prepend_steps(results_pos, steps.clone())?;

                        // Intersect with (0, inf)
                        let domain_pos = SolutionSet::Continuous(Interval {
                            min: simplifier.context.num(0),
                            min_type: BoundType::Open,
                            max: pos_inf(&mut simplifier.context),
                            max_type: BoundType::Open,
                        });
                        let final_pos =
                            intersect_solution_sets(&simplifier.context, set_pos, domain_pos);

                        // Case 2: x < 0. Multiply by x (negative) -> Inequality flips.

                        let op_neg = op.clone();

                        // Add step for Case 2
                        if simplifier.collect_steps {
                            steps.push(SolveStep {
                                 description: format!("Case 2: Assume {} < 0. Multiply by {} (negative). Inequality flips.",
                                  cas_ast::DisplayExpr { context: &simplifier.context, id: r },
                                  cas_ast::DisplayExpr { context: &simplifier.context, id: r }),
                                 equation_after: Equation { lhs: r, rhs: sim_rhs, op: op_neg.clone() }
                             });
                        }

                        let results_neg = isolate(r, sim_rhs, op_neg, var, simplifier)?;
                        let (set_neg, steps_neg) = prepend_steps(results_neg, steps.clone())?;

                        // Intersect with (-inf, 0)
                        let domain_neg = SolutionSet::Continuous(Interval {
                            min: neg_inf(&mut simplifier.context),
                            min_type: BoundType::Open,
                            max: simplifier.context.num(0),
                            max_type: BoundType::Open,
                        });
                        let final_neg =
                            intersect_solution_sets(&simplifier.context, set_neg, domain_neg);

                        // Union
                        let final_set =
                            union_solution_sets(&simplifier.context, final_pos, final_neg);

                        // Combine steps
                        let mut all_steps = steps_pos;
                        all_steps.push(SolveStep {
                            description: "--- End of Case 1 ---".to_string(),
                            equation_after: Equation {
                                lhs: r,
                                rhs: sim_rhs,
                                op: op.clone(),
                            },
                        }); // Dummy eq
                        all_steps.extend(steps_neg);

                        return Ok((final_set, all_steps));
                    }
                }

                if simplifier.collect_steps {
                    steps.push(SolveStep {
                        description: format!(
                            "Isolate denominator {}",
                            cas_ast::DisplayExpr {
                                context: &simplifier.context,
                                id: r
                            }
                        ),
                        equation_after: new_eq.clone(),
                    });
                }
                let results = isolate(r, sim_rhs, op, var, simplifier)?;
                prepend_steps(results, steps)
            }
        }
        Expr::Pow(b, e) => {
            // B^E = RHS
            if contains_var(&simplifier.context, b, var) {
                // Check if exponent is an even integer
                let is_even = if let Some(n) =
                    crate::solver::solution_set::get_number(&simplifier.context, e)
                {
                    n.is_integer() && (n.to_integer() % 2 == 0.into())
                } else {
                    false
                };

                if is_even {
                    // Check if RHS is negative
                    if is_negative(&simplifier.context, rhs) {
                        let result = match op {
                            RelOp::Eq => SolutionSet::Empty,
                            RelOp::Gt | RelOp::Geq | RelOp::Neq => SolutionSet::AllReals,
                            RelOp::Lt | RelOp::Leq => SolutionSet::Empty,
                        };
                        if simplifier.collect_steps {
                            steps.push(SolveStep {
                                description: format!(
                                    "Even power cannot be negative ({} {} {})",
                                    cas_ast::DisplayExpr {
                                        context: &simplifier.context,
                                        id: b
                                    },
                                    op,
                                    cas_ast::DisplayExpr {
                                        context: &simplifier.context,
                                        id: rhs
                                    }
                                ),
                                equation_after: Equation {
                                    lhs,
                                    rhs,
                                    op: op.clone(),
                                }, // No change
                            });
                        }
                        return Ok((result, steps));
                    }

                    // B^E = RHS -> |B| = RHS^(1/E)
                    let one = simplifier.context.num(1);
                    let inv_exp = simplifier.context.add(Expr::Div(one, e));
                    let new_rhs = simplifier.context.add(Expr::Pow(rhs, inv_exp));

                    // Construct |B|
                    let abs_b = simplifier
                        .context
                        .add(Expr::Function("abs".to_string(), vec![b]));

                    let new_eq = Equation {
                        lhs: abs_b,
                        rhs: new_rhs,
                        op: op.clone(),
                    };
                    if simplifier.collect_steps {
                        steps.push(SolveStep {
                            description: format!(
                                "Take {}-th root of both sides (even root implies absolute value)",
                                cas_ast::DisplayExpr {
                                    context: &simplifier.context,
                                    id: e
                                }
                            ),
                            equation_after: new_eq.clone(),
                        });
                    }

                    // Isolate |B|
                    // Note: We pass 'op' as is. |B| < RHS will be handled by isolate(|B|...) logic.
                    let results = isolate(abs_b, new_rhs, op, var, simplifier)?;
                    prepend_steps(results, steps)
                } else {
                    // B = RHS^(1/E)
                    let one = simplifier.context.num(1);
                    let inv_exp = simplifier.context.add(Expr::Div(one, e));
                    let new_rhs = simplifier.context.add(Expr::Pow(rhs, inv_exp));
                    let new_eq = Equation {
                        lhs: b,
                        rhs: new_rhs,
                        op: op.clone(),
                    };
                    if simplifier.collect_steps {
                        steps.push(SolveStep {
                            description: format!(
                                "Take {}-th root of both sides",
                                cas_ast::DisplayExpr {
                                    context: &simplifier.context,
                                    id: e
                                }
                            ),
                            equation_after: new_eq.clone(),
                        });
                    }

                    // Check if exponent is negative to flip inequality
                    let mut new_op = op.clone();
                    if is_negative(&simplifier.context, e) {
                        new_op = flip_inequality(new_op);
                    }

                    let results = isolate(b, new_rhs, new_op, var, simplifier)?;
                    prepend_steps(results, steps)
                }
            } else {
                // E = log(B, RHS)
                let new_rhs = simplifier
                    .context
                    .add(Expr::Function("log".to_string(), vec![b, rhs]));
                let new_eq = Equation {
                    lhs: e,
                    rhs: new_rhs,
                    op: op.clone(),
                };
                if simplifier.collect_steps {
                    steps.push(SolveStep {
                        description: format!(
                            "Take log base {} of both sides",
                            cas_ast::DisplayExpr {
                                context: &simplifier.context,
                                id: b
                            }
                        ),
                        equation_after: new_eq.clone(),
                    });
                }
                let results = isolate(e, new_rhs, op, var, simplifier)?;
                prepend_steps(results, steps)
            }
        }
        Expr::Function(name, args) => {
            if name == "abs" && args.len() == 1 {
                // |A| = B
                // |A| < B -> -B < A < B (Intersection)
                // |A| > B -> A > B OR A < -B (Union)

                let arg = args[0];

                // Branch 1: Positive case (A op B)
                let eq1 = Equation {
                    lhs: arg,
                    rhs: rhs,
                    op: op.clone(),
                };
                let mut steps1 = steps.clone();
                if simplifier.collect_steps {
                    steps1.push(SolveStep {
                        description: format!(
                            "Split absolute value (Case 1): {} {} {}",
                            cas_ast::DisplayExpr {
                                context: &simplifier.context,
                                id: arg
                            },
                            op,
                            cas_ast::DisplayExpr {
                                context: &simplifier.context,
                                id: rhs
                            }
                        ),
                        equation_after: eq1.clone(),
                    });
                }
                let results1 = isolate(arg, rhs, op.clone(), var, simplifier)?;
                let (set1, steps1_out) = prepend_steps(results1, steps1)?;

                // Branch 2: Negative case
                // |A| < B -> A > -B (Flip op)
                // |A| > B -> A < -B (Flip op)
                // |A| = B -> A = -B

                let neg_rhs = simplifier.context.add(Expr::Neg(rhs));
                let op2 = match op {
                    RelOp::Eq => RelOp::Eq,
                    RelOp::Neq => RelOp::Neq,
                    RelOp::Lt => RelOp::Gt, // |x| < 5 -> x > -5
                    RelOp::Leq => RelOp::Geq,
                    RelOp::Gt => RelOp::Lt, // |x| > 5 -> x < -5
                    RelOp::Geq => RelOp::Leq,
                };

                let eq2 = Equation {
                    lhs: arg,
                    rhs: neg_rhs,
                    op: op2.clone(),
                };
                let mut steps2 = steps.clone();
                if simplifier.collect_steps {
                    steps2.push(SolveStep {
                        description: format!(
                            "Split absolute value (Case 2): {} {} {}",
                            cas_ast::DisplayExpr {
                                context: &simplifier.context,
                                id: arg
                            },
                            op2,
                            cas_ast::DisplayExpr {
                                context: &simplifier.context,
                                id: neg_rhs
                            }
                        ),
                        equation_after: eq2.clone(),
                    });
                }
                let results2 = isolate(arg, neg_rhs, op2, var, simplifier)?;
                let (set2, steps2_out) = prepend_steps(results2, steps2)?;

                // Combine sets
                let final_set = match op {
                    RelOp::Eq | RelOp::Neq | RelOp::Gt | RelOp::Geq => {
                        union_solution_sets(&simplifier.context, set1, set2)
                    }
                    RelOp::Lt | RelOp::Leq => {
                        intersect_solution_sets(&simplifier.context, set1, set2)
                    }
                };

                // Combine steps (just append for now, maybe separate them?)
                let mut all_steps = steps1_out;
                all_steps.extend(steps2_out);

                Ok((final_set, all_steps))
            } else if name == "log" && args.len() == 2 {
                let base = args[0];
                let arg = args[1];

                if contains_var(&simplifier.context, arg, var)
                    && !contains_var(&simplifier.context, base, var)
                {
                    // log(b, x) = RHS -> x = b^RHS
                    let new_rhs = simplifier.context.add(Expr::Pow(base, rhs));
                    let new_eq = Equation {
                        lhs: arg,
                        rhs: new_rhs,
                        op: op.clone(),
                    };
                    if simplifier.collect_steps {
                        steps.push(SolveStep {
                            description: format!(
                                "Exponentiate both sides with base {}",
                                cas_ast::DisplayExpr {
                                    context: &simplifier.context,
                                    id: base
                                }
                            ),
                            equation_after: new_eq.clone(),
                        });
                    }
                    let results = isolate(arg, new_rhs, op, var, simplifier)?;
                    prepend_steps(results, steps)
                } else if contains_var(&simplifier.context, base, var)
                    && !contains_var(&simplifier.context, arg, var)
                {
                    let one = simplifier.context.num(1);
                    let inv_rhs = simplifier.context.add(Expr::Div(one, rhs));
                    let new_rhs = simplifier.context.add(Expr::Pow(arg, inv_rhs));
                    let new_eq = Equation {
                        lhs: base,
                        rhs: new_rhs,
                        op: op.clone(),
                    };
                    if simplifier.collect_steps {
                        steps.push(SolveStep {
                            description: "Isolate base of logarithm".to_string(),
                            equation_after: new_eq.clone(),
                        });
                    }
                    let results = isolate(base, new_rhs, op, var, simplifier)?;
                    prepend_steps(results, steps)
                } else {
                    Err(CasError::IsolationError(
                        var.to_string(),
                        "Cannot isolate from log function".to_string(),
                    ))
                }
            } else if args.len() == 1 {
                let arg = args[0];
                if contains_var(&simplifier.context, arg, var) {
                    match name.as_str() {
                        "ln" => {
                            let e = simplifier.context.add(Expr::Constant(cas_ast::Constant::E));
                            let new_rhs = simplifier.context.add(Expr::Pow(e, rhs));
                            let new_eq = Equation {
                                lhs: arg,
                                rhs: new_rhs,
                                op: op.clone(),
                            };
                            if simplifier.collect_steps {
                                steps.push(SolveStep {
                                    description: "Exponentiate both sides with base e".to_string(),
                                    equation_after: new_eq.clone(),
                                });
                            }
                            let results = isolate(arg, new_rhs, op, var, simplifier)?;
                            prepend_steps(results, steps)
                        }
                        "exp" => {
                            let new_rhs = simplifier
                                .context
                                .add(Expr::Function("ln".to_string(), vec![rhs]));
                            let new_eq = Equation {
                                lhs: arg,
                                rhs: new_rhs,
                                op: op.clone(),
                            };
                            if simplifier.collect_steps {
                                steps.push(SolveStep {
                                    description: "Take natural log of both sides".to_string(),
                                    equation_after: new_eq.clone(),
                                });
                            }
                            let results = isolate(arg, new_rhs, op, var, simplifier)?;
                            prepend_steps(results, steps)
                        }
                        "sqrt" => {
                            let two = simplifier.context.num(2);
                            let new_rhs = simplifier.context.add(Expr::Pow(rhs, two));
                            let new_eq = Equation {
                                lhs: arg,
                                rhs: new_rhs,
                                op: op.clone(),
                            };
                            if simplifier.collect_steps {
                                steps.push(SolveStep {
                                    description: "Square both sides".to_string(),
                                    equation_after: new_eq.clone(),
                                });
                            }
                            let results = isolate(arg, new_rhs, op, var, simplifier)?;
                            prepend_steps(results, steps)
                        }
                        "sin" => {
                            // sin(x) = y -> x = arcsin(y)
                            let new_rhs = simplifier
                                .context
                                .add(Expr::Function("arcsin".to_string(), vec![rhs]));
                            let new_eq = Equation {
                                lhs: arg,
                                rhs: new_rhs,
                                op: op.clone(),
                            };
                            if simplifier.collect_steps {
                                steps.push(SolveStep {
                                    description: "Take arcsin of both sides".to_string(),
                                    equation_after: new_eq.clone(),
                                });
                            }

                            let (simplified_rhs, sim_steps) =
                                simplify_rhs(new_rhs, arg, op.clone(), simplifier);
                            steps.extend(sim_steps);

                            let results = isolate(arg, simplified_rhs, op, var, simplifier)?;
                            prepend_steps(results, steps)
                        }
                        "cos" => {
                            // cos(x) = y -> x = arccos(y)
                            let new_rhs = simplifier
                                .context
                                .add(Expr::Function("arccos".to_string(), vec![rhs]));
                            let new_eq = Equation {
                                lhs: arg,
                                rhs: new_rhs,
                                op: op.clone(),
                            };
                            if simplifier.collect_steps {
                                steps.push(SolveStep {
                                    description: "Take arccos of both sides".to_string(),
                                    equation_after: new_eq.clone(),
                                });
                            }

                            let (simplified_rhs, sim_steps) =
                                simplify_rhs(new_rhs, arg, op.clone(), simplifier);
                            steps.extend(sim_steps);

                            let results = isolate(arg, simplified_rhs, op, var, simplifier)?;
                            prepend_steps(results, steps)
                        }
                        "tan" => {
                            // tan(x) = y -> x = arctan(y)
                            let new_rhs = simplifier
                                .context
                                .add(Expr::Function("arctan".to_string(), vec![rhs]));
                            let new_eq = Equation {
                                lhs: arg,
                                rhs: new_rhs,
                                op: op.clone(),
                            };
                            if simplifier.collect_steps {
                                steps.push(SolveStep {
                                    description: "Take arctan of both sides".to_string(),
                                    equation_after: new_eq.clone(),
                                });
                            }

                            let (simplified_rhs, sim_steps) =
                                simplify_rhs(new_rhs, arg, op.clone(), simplifier);
                            steps.extend(sim_steps);

                            let results = isolate(arg, simplified_rhs, op, var, simplifier)?;
                            prepend_steps(results, steps)
                        }
                        _ => Err(CasError::UnknownFunction(name.clone())),
                    }
                } else {
                    Err(CasError::VariableNotFound(var.to_string()))
                }
            } else {
                Err(CasError::IsolationError(
                    var.to_string(),
                    format!(
                        "Cannot invert function '{}' with {} arguments",
                        name,
                        args.len()
                    ),
                ))
            }
        }
        Expr::Neg(inner) => {
            // -A = RHS -> A = -RHS
            // -A < RHS -> A > -RHS (Flip op)

            let new_rhs = simplifier.context.add(Expr::Neg(rhs));
            let new_op = flip_inequality(op);
            let new_eq = Equation {
                lhs: inner,
                rhs: new_rhs,
                op: new_op.clone(),
            };

            if simplifier.collect_steps {
                steps.push(SolveStep {
                    description: "Multiply both sides by -1 (flips inequality)".to_string(),
                    equation_after: new_eq.clone(),
                });
            }

            let results = isolate(inner, new_rhs, new_op, var, simplifier)?;
            prepend_steps(results, steps)
        }
        _ => Err(CasError::IsolationError(
            var.to_string(),
            format!("Cannot isolate from {:?}", lhs_expr),
        )),
    }
}

pub fn prepend_steps(
    (set, mut res_steps): (SolutionSet, Vec<SolveStep>),
    mut steps: Vec<SolveStep>,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    steps.append(&mut res_steps);
    Ok((set, steps))
}

pub fn is_negative(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Number(n) => *n < num_rational::BigRational::from_integer(0.into()),
        Expr::Neg(_) => true, // Simple check, might be Neg(Neg(x))
        Expr::Mul(l, r) => is_negative(ctx, *l) ^ is_negative(ctx, *r),
        _ => false, // Conservative
    }
}

pub fn simplify_rhs(
    rhs: ExprId,
    lhs: ExprId,
    op: RelOp,
    simplifier: &mut Simplifier,
) -> (ExprId, Vec<SolveStep>) {
    let (simplified_rhs, sim_steps) = simplifier.simplify(rhs);
    let mut steps = Vec::new();

    if simplifier.collect_steps {
        for step in sim_steps {
            steps.push(SolveStep {
                description: step.description,
                equation_after: Equation {
                    lhs: lhs,
                    rhs: step.after, // This is correct, each step produces a new RHS
                    op: op.clone(),
                },
            });
        }
    }
    (simplified_rhs, steps)
}

pub fn contains_var(ctx: &Context, expr: ExprId, var: &str) -> bool {
    match ctx.get(expr) {
        Expr::Variable(v) => v == var,
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) | Expr::Pow(l, r) => {
            contains_var(ctx, *l, var) || contains_var(ctx, *r, var)
        }
        Expr::Neg(e) => contains_var(ctx, *e, var),
        Expr::Function(_, args) => args.iter().any(|a| contains_var(ctx, *a, var)),
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
