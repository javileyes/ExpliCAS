pub mod solution_set;
pub mod isolation;
pub mod strategies;
pub mod strategy;

use cas_ast::{Equation, SolutionSet, Context, ExprId};
use crate::engine::Simplifier;

pub use self::isolation::contains_var;

#[derive(Debug, Clone)]
pub struct SolveStep {
    pub description: String,
    pub equation_after: Equation, 
}

use crate::error::CasError;

use crate::solver::strategy::SolverStrategy;
use crate::solver::strategies::{SubstitutionStrategy, QuadraticStrategy, IsolationStrategy, UnwrapStrategy};

pub fn solve(eq: &Equation, var: &str, simplifier: &mut Simplifier) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    // 1. Check if variable exists in equation
    if !contains_var(&simplifier.context, eq.lhs, var) && !contains_var(&simplifier.context, eq.rhs, var) {
        return Err(CasError::VariableNotFound(var.to_string()));
    }

    // 2. Define strategies
    // In a real app, these might be configured in Simplifier or passed in.
    let strategies: Vec<Box<dyn SolverStrategy>> = vec![
        Box::new(SubstitutionStrategy),
        Box::new(UnwrapStrategy),
        Box::new(QuadraticStrategy),
        Box::new(IsolationStrategy),
    ];

    // 3. Try strategies
    for strategy in strategies {
        if let Some(res) = strategy.apply(eq, var, simplifier) {
            match res {
                Ok((result, steps)) => {
                    // Verify solutions if Discrete
                    if let SolutionSet::Discrete(sols) = result {
                        if !strategy.should_verify() {
                             return Ok((SolutionSet::Discrete(sols), steps));
                        }
                        let mut valid_sols = Vec::new();
                        for sol in sols {
                            if verify_solution(eq, var, sol, simplifier) {
                                valid_sols.push(sol);
                            }
                        }
                        return Ok((SolutionSet::Discrete(valid_sols), steps));
                    }
                    return Ok((result, steps));
                },
                Err(e) => return Err(e),
            }
        }
    }

    Err(CasError::SolverError("No strategy could solve this equation.".to_string()))
}

fn verify_solution(eq: &Equation, var: &str, sol: ExprId, simplifier: &mut Simplifier) -> bool {
    // 1. Substitute
    let lhs_sub = substitute(&mut simplifier.context, eq.lhs, var, sol);
    let rhs_sub = substitute(&mut simplifier.context, eq.rhs, var, sol);
    
    // 2. Simplify
    let (lhs_sim, _) = simplifier.simplify(lhs_sub);
    let (rhs_sim, _) = simplifier.simplify(rhs_sub);
    
    // 3. Check equality
    simplifier.are_equivalent(lhs_sim, rhs_sim)
}

fn substitute(ctx: &mut Context, expr: ExprId, var: &str, val: ExprId) -> ExprId {
    use cas_ast::Expr;
    let expr_data = ctx.get(expr).clone();
    match expr_data {
        Expr::Variable(v) if v == var => val,
        Expr::Add(l, r) => {
            let nl = substitute(ctx, l, var, val);
            let nr = substitute(ctx, r, var, val);
            if nl != l || nr != r { ctx.add(Expr::Add(nl, nr)) } else { expr }
        },
        Expr::Sub(l, r) => {
            let nl = substitute(ctx, l, var, val);
            let nr = substitute(ctx, r, var, val);
            if nl != l || nr != r { ctx.add(Expr::Sub(nl, nr)) } else { expr }
        },
        Expr::Mul(l, r) => {
            let nl = substitute(ctx, l, var, val);
            let nr = substitute(ctx, r, var, val);
            if nl != l || nr != r { ctx.add(Expr::Mul(nl, nr)) } else { expr }
        },
        Expr::Div(l, r) => {
            let nl = substitute(ctx, l, var, val);
            let nr = substitute(ctx, r, var, val);
            if nl != l || nr != r { ctx.add(Expr::Div(nl, nr)) } else { expr }
        },
        Expr::Pow(b, e) => {
            let nb = substitute(ctx, b, var, val);
            let ne = substitute(ctx, e, var, val);
            if nb != b || ne != e { ctx.add(Expr::Pow(nb, ne)) } else { expr }
        },
        Expr::Neg(e) => {
            let ne = substitute(ctx, e, var, val);
            if ne != e { ctx.add(Expr::Neg(ne)) } else { expr }
        },
        Expr::Function(name, args) => {
            let mut new_args = Vec::new();
            let mut changed = false;
            for arg in args {
                let new_arg = substitute(ctx, arg, var, val);
                if new_arg != arg { changed = true; }
                new_args.push(new_arg);
            }
            if changed { ctx.add(Expr::Function(name, new_args)) } else { expr }
        },
        _ => expr
    }
}



#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse; 
    use cas_ast::{BoundType, RelOp, DisplayExpr, Context}; 
    
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
            assert_eq!(s, "3");
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
