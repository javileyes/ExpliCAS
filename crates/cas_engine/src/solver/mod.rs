pub mod solution_set;
pub mod isolation;
pub mod strategies;
pub mod strategy;

use cas_ast::{Equation, SolutionSet, Context, ExprId, DisplayExpr};
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

pub fn solve(eq: &Equation, var: &str, simplifier: &mut Simplifier) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    // 1. Check if variable exists in equation
    if !contains_var(&simplifier.context, eq.lhs, var) && !contains_var(&simplifier.context, eq.rhs, var) {
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
    use cas_ast::{BoundType, RelOp}; 
    
    // Helper to make equation from strings
    fn make_eq(ctx: &mut Context, lhs: &str, rhs: &str) -> Equation {
        Equation {
            lhs: parse(lhs, ctx).unwrap(),
            rhs: parse(rhs, ctx).unwrap(),
            op: RelOp::Eq,
        }
    }

    #[test]
    fn test_solve_linear() {
        // x + 2 = 5 -> x = 3
        let mut simplifier = Simplifier::new();
        let eq = make_eq(&mut simplifier.context, "x + 2", "5");
        simplifier.collect_steps = true;
        let (result, _) = solve(&eq, "x", &mut simplifier).unwrap();
        
        if let SolutionSet::Discrete(solutions) = result {
            assert_eq!(solutions.len(), 1);
            let s = format!("{}", DisplayExpr { context: &simplifier.context, id: solutions[0] });
            assert_eq!(s, "5 - 2");
        } else {
            panic!("Expected Discrete solution");
        }
    }

    #[test]
    fn test_solve_mul() {
        // 2 * x = 6 -> x = 6 / 2
        let mut simplifier = Simplifier::new();
        let eq = make_eq(&mut simplifier.context, "2 * x", "6");
        simplifier.collect_steps = true;
        let (result, _) = solve(&eq, "x", &mut simplifier).unwrap();
        
        if let SolutionSet::Discrete(solutions) = result {
            assert_eq!(solutions.len(), 1);
            let s = format!("{}", DisplayExpr { context: &simplifier.context, id: solutions[0] });
            assert_eq!(s, "6 / 2");
        } else {
            panic!("Expected Discrete solution");
        }
    }
    
    #[test]
    fn test_solve_pow() {
        // x^2 = 4 -> x = 4^(1/2)
        let mut simplifier = Simplifier::new();
        let eq = make_eq(&mut simplifier.context, "x^2", "4");
        simplifier.add_rule(Box::new(crate::rules::exponents::EvaluatePowerRule));
        simplifier.add_rule(Box::new(crate::rules::canonicalization::CanonicalizeNegationRule));
        simplifier.add_rule(Box::new(crate::rules::arithmetic::CombineConstantsRule));
        simplifier.collect_steps = true;
        let (result, _) = solve(&eq, "x", &mut simplifier).unwrap();
        
        if let SolutionSet::Discrete(mut solutions) = result {
            assert_eq!(solutions.len(), 2);
            // Sort to ensure order
            solutions.sort_by(|a, b| {
                let sa = format!("{}", DisplayExpr { context: &simplifier.context, id: *a });
                let sb = format!("{}", DisplayExpr { context: &simplifier.context, id: *b });
                sa.cmp(&sb)
            });
            
            let s1 = format!("{}", DisplayExpr { context: &simplifier.context, id: solutions[0] });
            let s2 = format!("{}", DisplayExpr { context: &simplifier.context, id: solutions[1] });
            
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
        let mut simplifier = Simplifier::new();
        let eq = make_eq(&mut simplifier.context, "|x|", "5");
        let (result, _) = solve(&eq, "x", &mut simplifier).unwrap();
        
        if let SolutionSet::Discrete(solutions) = result {
            assert_eq!(solutions.len(), 2);
            // Order might vary
            let s: Vec<String> = solutions.iter().map(|e| format!("{}", DisplayExpr { context: &simplifier.context, id: *e })).collect();
            assert!(s.contains(&"5".to_string()));
            assert!(s.contains(&"-5".to_string()));
        } else {
            panic!("Expected Discrete solution");
        }
    }

    #[test]
    fn test_solve_inequality_flip() {
        // -2x < 10 -> x > -5
        let mut simplifier = Simplifier::new();
        let eq = Equation {
            lhs: parse("-2*x", &mut simplifier.context).unwrap(),
            rhs: parse("10", &mut simplifier.context).unwrap(),
            op: RelOp::Lt,
        };
        let (result, _) = solve(&eq, "x", &mut simplifier).unwrap();
        
        if let SolutionSet::Continuous(interval) = result {
            // (-5, inf)
            let s_min = format!("{}", DisplayExpr { context: &simplifier.context, id: interval.min });
            let s_max = format!("{}", DisplayExpr { context: &simplifier.context, id: interval.max });
            assert_eq!(s_min, "10 / -2"); // Not simplified
            assert_eq!(interval.min_type, BoundType::Open);
            assert_eq!(s_max, "infinity");
        } else {
            panic!("Expected Continuous solution, got {:?}", result);
        }
    }

    #[test]
    fn test_solve_abs_inequality() {
        // |x| < 5 -> (-5, 5)
        let mut simplifier = Simplifier::new();
        let eq = Equation {
            lhs: parse("|x|", &mut simplifier.context).unwrap(),
            rhs: parse("5", &mut simplifier.context).unwrap(),
            op: RelOp::Lt,
        };
        let (result, _) = solve(&eq, "x", &mut simplifier).unwrap();
        
        if let SolutionSet::Continuous(interval) = result {
            let s_min = format!("{}", DisplayExpr { context: &simplifier.context, id: interval.min });
            let s_max = format!("{}", DisplayExpr { context: &simplifier.context, id: interval.max });
            assert_eq!(s_min, "-5");
            assert_eq!(s_max, "5");
        } else {
            panic!("Expected Continuous solution, got {:?}", result);
        }
    }
}
