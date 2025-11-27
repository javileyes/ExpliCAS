use cas_ast::{Expr, Equation, RelOp, SolutionSet, Interval, BoundType, Constant};
use std::rc::Rc;
use crate::engine::Simplifier;
use std::cmp::Ordering;
use num_rational::BigRational;



#[derive(Debug, Clone)]
pub struct SolveStep {
    pub description: String,
    // equation_after is less relevant now that we return SolutionSet, 
    // but we can keep it to show the state of the equation being manipulated.
    // For branching, it might be the equation of that specific branch.
    pub equation_after: Equation, 
}

// Helper to create -infinity
fn neg_inf() -> Rc<Expr> {
    Rc::new(Expr::Neg(Rc::new(Expr::Constant(Constant::Infinity))))
}

// Helper to create +infinity
fn pos_inf() -> Rc<Expr> {
    Rc::new(Expr::Constant(Constant::Infinity))
}

fn is_infinity(expr: &Expr) -> bool {
    matches!(expr, Expr::Constant(Constant::Infinity))
}

fn is_neg_infinity(expr: &Expr) -> bool {
    match expr {
        Expr::Neg(inner) => is_infinity(inner),
        _ => false,
    }
}

fn get_number(expr: &Expr) -> Option<BigRational> {
    match expr {
        Expr::Number(n) => Some(n.clone()),
        Expr::Neg(inner) => get_number(inner).map(|n| -n),
        _ => None,
    }
}

fn compare_values(a: &Expr, b: &Expr) -> Ordering {
    // Handle Infinity
    let a_inf = is_infinity(a);
    let b_inf = is_infinity(b);
    let a_neg_inf = is_neg_infinity(a);
    let b_neg_inf = is_neg_infinity(b);
    
    if a_neg_inf {
        if b_neg_inf { return Ordering::Equal; }
        return Ordering::Less;
    }
    if b_neg_inf { return Ordering::Greater; }
    
    if a_inf {
        if b_inf { return Ordering::Equal; }
        return Ordering::Greater;
    }
    if b_inf { return Ordering::Less; }
    
    // Handle Numbers
    if let (Some(n1), Some(n2)) = (get_number(a), get_number(b)) {
        return n1.cmp(&n2);
    }
    
    // Fallback: Use structural comparison if we can't compare values
    // This is risky but better than nothing for symbolic bounds
    // But we need to be careful. For now, let's use structural but warn?
    // Actually, let's just use structural as a fallback.
    // But structural says Neg(Inf) > Number. We handled Inf above.
    // What about Variable(x) vs Number(5)?
    // We don't know.
    // For this task, we assume solvable inequalities result in numeric bounds.
    crate::ordering::compare_expr(a, b)
}

fn intersect_intervals(i1: &Interval, i2: &Interval) -> SolutionSet {
    // Intersection of [a, b] and [c, d] is [max(a,c), min(b,d)]
    
    println!("Intersecting {} and {}", i1, i2);

    // Compare mins
    let (min, min_type) = match compare_values(&i1.min, &i2.min) {
        Ordering::Less => (i2.min.clone(), i2.min_type.clone()), // i1.min < i2.min -> take i2
        Ordering::Greater => (i1.min.clone(), i1.min_type.clone()), // i1.min > i2.min -> take i1
        Ordering::Equal => {
            let type_ = if i1.min_type == BoundType::Open || i2.min_type == BoundType::Open {
                BoundType::Open
            } else {
                BoundType::Closed
            };
            (i1.min.clone(), type_)
        }
    };

    // Compare maxs
    let (max, max_type) = match compare_values(&i1.max, &i2.max) {
        Ordering::Less => (i1.max.clone(), i1.max_type.clone()), // i1.max < i2.max -> take i1
        Ordering::Greater => (i2.max.clone(), i2.max_type.clone()), // i1.max > i2.max -> take i2
        Ordering::Equal => {
            let type_ = if i1.max_type == BoundType::Open || i2.max_type == BoundType::Open {
                BoundType::Open
            } else {
                BoundType::Closed
            };
            (i1.max.clone(), type_)
        }
    };
    
    println!("Result min: {}, max: {}", min, max);
    println!("Compare min/max: {:?}", compare_values(&min, &max));

    // Check if valid interval (min < max)
    match compare_values(&min, &max) {
        Ordering::Less => SolutionSet::Continuous(Interval { min, min_type, max, max_type }),
        Ordering::Equal => {
            if min_type == BoundType::Closed && max_type == BoundType::Closed {
                SolutionSet::Discrete(vec![min])
            } else {
                SolutionSet::Empty
            }
        },
        Ordering::Greater => SolutionSet::Empty,
    }
}

fn union_solution_sets(s1: SolutionSet, s2: SolutionSet) -> SolutionSet {
    match (s1, s2) {
        (SolutionSet::Empty, s) => s,
        (s, SolutionSet::Empty) => s,
        (SolutionSet::AllReals, _) => SolutionSet::AllReals,
        (_, SolutionSet::AllReals) => SolutionSet::AllReals,
        (SolutionSet::Continuous(i1), SolutionSet::Continuous(i2)) => {
            // Check if they overlap or touch
            // If they do, merge. If not, return Union.
            // Simplified: Just return Union for now unless we implement full interval arithmetic
            SolutionSet::Union(vec![i1, i2])
        },
        (SolutionSet::Union(mut u1), SolutionSet::Union(u2)) => {
            u1.extend(u2);
            SolutionSet::Union(u1)
        },
        (SolutionSet::Continuous(i), SolutionSet::Union(mut u)) => {
            u.push(i);
            SolutionSet::Union(u)
        },
        (SolutionSet::Union(mut u), SolutionSet::Continuous(i)) => {
            u.push(i);
            SolutionSet::Union(u)
        },
        (SolutionSet::Discrete(mut d1), SolutionSet::Discrete(d2)) => {
            d1.extend(d2);
            SolutionSet::Discrete(d1)
        },
        // Fallback for mixed types (Discrete + Continuous) -> Union?
        // SolutionSet::Union currently only holds Intervals.
        // We might need to update SolutionSet definition to hold mixed types or just keep them separate.
        // For this iteration, let's assume we mostly deal with same types or just return a list.
        // But SolutionSet::Union is Vec<Interval>.
        // Let's just return s1 for now if incompatible (TODO: Fix SolutionSet definition)
        (s1, _) => s1, 
    }
}

fn intersect_solution_sets(s1: SolutionSet, s2: SolutionSet) -> SolutionSet {
    match (s1, s2) {
        (SolutionSet::Empty, _) => SolutionSet::Empty,
        (_, SolutionSet::Empty) => SolutionSet::Empty,
        (SolutionSet::AllReals, s) => s,
        (s, SolutionSet::AllReals) => s,
        (SolutionSet::Continuous(i1), SolutionSet::Continuous(i2)) => {
            intersect_intervals(&i1, &i2)
        },
        (SolutionSet::Continuous(i), SolutionSet::Union(u)) => {
            // Intersect i with each interval in u
            let mut new_u = Vec::new();
            for interval in u {
                let res = intersect_intervals(&i, &interval);
                match res {
                    SolutionSet::Continuous(new_i) => new_u.push(new_i),
                    SolutionSet::Discrete(_d) => {
                        // Convert discrete points to tiny intervals or handle separately?
                        // SolutionSet::Union currently only holds Intervals.
                        // We need to handle mixed types properly.
                        // For now, let's ignore discrete points in Union or upgrade Union to hold SolutionSets?
                        // Or just return Discrete if it's the only result?
                        // If we have multiple discrete points, we need a Discrete set.
                        // Complex. Let's assume for now we get Continuous intervals.
                    },
                    _ => {}
                }
            }
            if new_u.is_empty() {
                SolutionSet::Empty
            } else if new_u.len() == 1 {
                SolutionSet::Continuous(new_u[0].clone())
            } else {
                SolutionSet::Union(new_u)
            }
        },
        (SolutionSet::Union(u), SolutionSet::Continuous(i)) => {
            intersect_solution_sets(SolutionSet::Continuous(i), SolutionSet::Union(u))
        },
        (SolutionSet::Union(u1), SolutionSet::Union(u2)) => {
            // Distributive property: (A U B) n (C U D) = (A n C) U (A n D) U (B n C) U (B n D)
            let mut new_u = Vec::new();
            for i1 in &u1 {
                for i2 in &u2 {
                    let res = intersect_intervals(i1, i2);
                    match res {
                        SolutionSet::Continuous(new_i) => new_u.push(new_i),
                        _ => {}
                    }
                }
            }
            if new_u.is_empty() {
                SolutionSet::Empty
            } else if new_u.len() == 1 {
                SolutionSet::Continuous(new_u[0].clone())
            } else {
                SolutionSet::Union(new_u)
            }
        },
        _ => SolutionSet::Empty,
    }
}


pub fn solve(eq: &Equation, var: &str, simplifier: &Simplifier) -> Result<(SolutionSet, Vec<SolveStep>), String> {
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
        
        let (result_set, mut res_steps) = solve_internal(&new_eq, var, simplifier)?;
        steps.append(&mut res_steps);
        return Ok((result_set, steps));
    }

    if lhs_has_var && rhs_has_var {
        return Err("Variable appears on both sides. Please simplify/collect first.".to_string());
    }

    // Now LHS has var, RHS does not.
    let (result_set, mut res_steps) = isolate(&eq.lhs, &eq.rhs, eq.op.clone(), var, simplifier)?;
    steps.append(&mut res_steps);
    Ok((result_set, steps))
}

// Internal helper to avoid re-checking var presence unnecessarily if we already know it
fn solve_internal(eq: &Equation, var: &str, simplifier: &Simplifier) -> Result<(SolutionSet, Vec<SolveStep>), String> {
    // This is just a wrapper to call isolate directly since we know var is on LHS from the swap logic
    isolate(&eq.lhs, &eq.rhs, eq.op.clone(), var, simplifier)
}

fn isolate(lhs: &Rc<Expr>, rhs: &Rc<Expr>, op: RelOp, var: &str, simplifier: &Simplifier) -> Result<(SolutionSet, Vec<SolveStep>), String> {
    let mut steps = Vec::new();
    
    match lhs.as_ref() {
        Expr::Variable(v) if v == var => {
            let set = match op {
                RelOp::Eq => SolutionSet::Discrete(vec![rhs.clone()]),
                RelOp::Neq => {
                    // x != 5 -> (-inf, 5) U (5, inf)
                    let i1 = Interval { min: neg_inf(), min_type: BoundType::Open, max: rhs.clone(), max_type: BoundType::Open };
                    let i2 = Interval { min: rhs.clone(), min_type: BoundType::Open, max: pos_inf(), max_type: BoundType::Open };
                    SolutionSet::Union(vec![i1, i2])
                },
                RelOp::Lt => SolutionSet::Continuous(Interval {
                    min: neg_inf(), min_type: BoundType::Open,
                    max: rhs.clone(), max_type: BoundType::Open
                }),
                RelOp::Gt => SolutionSet::Continuous(Interval {
                    min: rhs.clone(), min_type: BoundType::Open,
                    max: pos_inf(), max_type: BoundType::Open
                }),
                RelOp::Leq => SolutionSet::Continuous(Interval {
                    min: neg_inf(), min_type: BoundType::Open,
                    max: rhs.clone(), max_type: BoundType::Closed
                }),
                RelOp::Geq => SolutionSet::Continuous(Interval {
                    min: rhs.clone(), min_type: BoundType::Closed,
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
                         let results_pos = isolate(r, &sim_rhs, op_pos, var, simplifier)?;
                         let (set_pos, mut steps_pos) = prepend_steps(results_pos, steps.clone())?;
                         
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
                         
                         steps_pos.extend(steps_neg);
                         return Ok((final_set, steps_pos));
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
    (set, mut res_steps): (SolutionSet, Vec<SolveStep>),
    mut steps: Vec<SolveStep>
) -> Result<(SolutionSet, Vec<SolveStep>), String> {
    steps.append(&mut res_steps);
    Ok((set, steps))
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
        let (result, _) = solve(&eq, "x", &simplifier).unwrap();
        
        if let SolutionSet::Discrete(solutions) = result {
            assert_eq!(solutions.len(), 1);
            assert_eq!(format!("{}", solutions[0]), "5 - 2");
        } else {
            panic!("Expected Discrete solution");
        }
    }

    #[test]
    fn test_solve_mul() {
        // 2 * x = 6 -> x = 6 / 2
        let eq = make_eq("2 * x", "6");
        let mut simplifier = Simplifier::new();
        simplifier.collect_steps = true;
        let (result, _) = solve(&eq, "x", &simplifier).unwrap();
        
        if let SolutionSet::Discrete(solutions) = result {
            assert_eq!(solutions.len(), 1);
            assert_eq!(format!("{}", solutions[0]), "6 / 2");
        } else {
            panic!("Expected Discrete solution");
        }
    }
    
    #[test]
    fn test_solve_pow() {
        // x^2 = 4 -> x = 4^(1/2)
        let eq = make_eq("x^2", "4");
        let mut simplifier = Simplifier::new();
        simplifier.collect_steps = true;
        let (result, _) = solve(&eq, "x", &simplifier).unwrap();
        
        if let SolutionSet::Discrete(solutions) = result {
            assert_eq!(solutions.len(), 1);
            assert_eq!(format!("{}", solutions[0]), "4^(1 / 2)");
        } else {
            panic!("Expected Discrete solution");
        }
    }
    
    #[test]
    fn test_solve_abs() {
        // |x| = 5 -> x=5, x=-5
        let eq = make_eq("|x|", "5");
        let simplifier = Simplifier::new();
        let (result, _) = solve(&eq, "x", &simplifier).unwrap();
        
        if let SolutionSet::Discrete(solutions) = result {
            assert_eq!(solutions.len(), 2);
            // Order might vary
            let s: Vec<String> = solutions.iter().map(|e| format!("{}", e)).collect();
            assert!(s.contains(&"5".to_string()));
            assert!(s.contains(&"-5".to_string()));
        } else {
            panic!("Expected Discrete solution");
        }
    }

    #[test]
    fn test_solve_inequality_flip() {
        // -2x < 10 -> x > -5
        let eq = Equation {
            lhs: parse("-2*x").unwrap(),
            rhs: parse("10").unwrap(),
            op: RelOp::Lt,
        };
        let simplifier = Simplifier::new();
        let (result, _) = solve(&eq, "x", &simplifier).unwrap();
        
        if let SolutionSet::Continuous(interval) = result {
            // (-5, inf)
            assert_eq!(format!("{}", interval.min), "10 / -2"); // Not simplified
            assert_eq!(interval.min_type, BoundType::Open);
            assert_eq!(format!("{}", interval.max), "infinity");
        } else {
            panic!("Expected Continuous solution, got {:?}", result);
        }
    }

    #[test]
    fn test_solve_abs_inequality() {
        // |x| < 5 -> (-5, 5)
        let eq = Equation {
            lhs: parse("|x|").unwrap(),
            rhs: parse("5").unwrap(),
            op: RelOp::Lt,
        };
        let simplifier = Simplifier::new();
        let (result, _) = solve(&eq, "x", &simplifier).unwrap();
        
        if let SolutionSet::Continuous(interval) = result {
            assert_eq!(format!("{}", interval.min), "-5");
            assert_eq!(format!("{}", interval.max), "5");
        } else {
            panic!("Expected Continuous solution, got {:?}", result);
        }
    }
}
