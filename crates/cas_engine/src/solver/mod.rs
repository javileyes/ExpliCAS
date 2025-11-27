pub mod solution_set;
pub mod isolation;
pub mod strategies;

use cas_ast::{Equation, RelOp, SolutionSet};
use crate::engine::Simplifier;

use self::isolation::isolate;
pub use self::isolation::contains_var;
use self::strategies::{detect_substitution, substitute_expr, solve_quadratic};

#[derive(Debug, Clone)]
pub struct SolveStep {
    pub description: String,
    pub equation_after: Equation, 
}

use crate::error::CasError;

pub fn solve(eq: &Equation, var: &str, simplifier: &Simplifier) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    // 1. Check if variable exists in equation
    if !contains_var(&eq.lhs, var) && !contains_var(&eq.rhs, var) {
        return Err(CasError::VariableNotFound(var.to_string()));
    }

    // 2. Try strategies
    // ...
    solve_internal(eq, var, simplifier)
}

fn solve_internal(eq: &Equation, var: &str, simplifier: &Simplifier) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    // We want to isolate 'var' on LHS.
    let mut steps = Vec::new();

    let lhs_has_var = contains_var(&eq.lhs, var);
    let rhs_has_var = contains_var(&eq.rhs, var);

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
        return Err(CasError::VariableNotFound("Variable appears on both sides. Please simplify/collect first.".to_string()));
    }

    // 0. Try Substitution (Hidden Quadratic)
    if let Some(sub_var_expr) = detect_substitution(eq, var) {
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
        let (u_solutions, mut u_steps) = solve(&new_eq, u_sym, simplifier)?;
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
                    let (x_sol, mut x_steps) = solve(&sub_eq, var, simplifier)?;
                    steps.append(&mut x_steps);
                    
                    if let SolutionSet::Discrete(xs) = x_sol {
                        final_solutions.extend(xs);
                    }
                }
                return Ok((SolutionSet::Discrete(final_solutions), steps));
            },
            _ => {
                // Handle intervals? Too complex for now.
            }
        }
    }

    // 0.5 Try Polynomial Solver (Quadratic Formula)
    if let Some(result) = solve_quadratic(eq, var, simplifier) {
        let (res_set, mut res_steps) = result?;
        steps.append(&mut res_steps);
        return Ok((res_set, steps));
    }

    // Now LHS has var, RHS does not.
    let (result_set, mut res_steps) = isolate(&eq.lhs, &eq.rhs, eq.op.clone(), var, simplifier)?;
    steps.append(&mut res_steps);
    Ok((result_set, steps))
}



#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse; 
    use cas_ast::BoundType; 
    
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
        simplifier.add_rule(Box::new(crate::rules::exponents::EvaluatePowerRule));
        simplifier.add_rule(Box::new(crate::rules::canonicalization::CanonicalizeNegationRule));
        simplifier.add_rule(Box::new(crate::rules::arithmetic::CombineConstantsRule));
        simplifier.collect_steps = true;
        let (result, _) = solve(&eq, "x", &simplifier).unwrap();
        
        if let SolutionSet::Discrete(mut solutions) = result {
            assert_eq!(solutions.len(), 2);
            // Sort to ensure order
            solutions.sort_by(|a, b| format!("{}", a).cmp(&format!("{}", b)));
            
            let s1 = format!("{}", solutions[0]);
            let s2 = format!("{}", solutions[1]);
            
            // We want to eventually see "-2" and "2".
            assert_eq!(s1, "-2");
            assert_eq!(s2, "2");
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
