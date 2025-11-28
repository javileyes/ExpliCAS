pub mod solution_set;
pub mod isolation;
pub mod strategies;
pub mod strategy;

use cas_ast::{Equation, RelOp, SolutionSet};
use crate::engine::Simplifier;

pub use self::isolation::contains_var;

#[derive(Debug, Clone)]
pub struct SolveStep {
    pub description: String,
    pub equation_after: Equation, 
}

use crate::error::CasError;

use crate::solver::strategy::SolverStrategy;
use crate::solver::strategies::{SubstitutionStrategy, QuadraticStrategy, IsolationStrategy};

pub fn solve(eq: &Equation, var: &str, simplifier: &Simplifier) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    // 1. Check if variable exists in equation
    if !contains_var(&eq.lhs, var) && !contains_var(&eq.rhs, var) {
        return Err(CasError::VariableNotFound(var.to_string()));
    }

    // 2. Define strategies
    // In a real app, these might be configured in Simplifier or passed in.
    let strategies: Vec<Box<dyn SolverStrategy>> = vec![
        Box::new(SubstitutionStrategy),
        Box::new(QuadraticStrategy),
        Box::new(IsolationStrategy),
    ];

    // 3. Try strategies
    for strategy in strategies {
        if let Some(result) = strategy.apply(eq, var, simplifier) {
            return result;
        }
    }

    Err(CasError::SolverError("No strategy could solve this equation.".to_string()))
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
