use cas_ast::{Constant, Equation, Expr, RelOp, SolutionSet};
use cas_engine::engine::Simplifier;
use cas_engine::solver::solve;
use cas_parser::parse;

#[test]
fn test_rational_inequality_undefined_debug() {
    let mut simplifier = Simplifier::with_default_rules();

    // 1/x > 0
    let lhs = parse("1/x", &mut simplifier.context).unwrap();
    let rhs = parse("0", &mut simplifier.context).unwrap();

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Gt,
    };

    let result = solve(&eq, "x", &mut simplifier);
    println!("Result for 1/x > 0: {:?}", result);

    assert!(result.is_ok());
    let (solution, steps) = result.unwrap();

    println!("Solution: {:?}", solution);

    // Check the solution set
    match &solution {
        SolutionSet::Continuous(interval) => {
            println!("Min ExprId: {:?}", interval.min);
            println!("Max ExprId: {:?}", interval.max);

            // Check what the min and max actually are
            let min_expr = simplifier.context.get(interval.min);
            let max_expr = simplifier.context.get(interval.max);

            println!("Min Expr: {:?}", min_expr);
            println!("Max Expr: {:?}", max_expr);

            // Check if max contains Undefined
            match max_expr {
                Expr::Constant(Constant::Undefined) => {
                    panic!("Max is Constant::Undefined!");
                }
                Expr::Constant(Constant::Infinity) => {
                    println!("Max is correctly Infinity");
                }
                _ => {
                    println!("Max is something else: {:?}", max_expr);
                    // Check recursively
                    check_for_undefined(&simplifier.context, interval.max, "max");
                }
            }
        }
        _ => println!("Got solution type: {:?}", solution),
    }

    // Print all steps
    for (i, step) in steps.iter().enumerate() {
        println!("Step {}: {}", i + 1, step.description);

        let lhs_expr = simplifier.context.get(step.equation_after.lhs);
        let rhs_expr = simplifier.context.get(step.equation_after.rhs);

        println!("  LHS: {:?}", lhs_expr);
        println!("  RHS: {:?}", rhs_expr);

        // Check for undefined in RHS
        check_for_undefined(
            &simplifier.context,
            step.equation_after.rhs,
            &format!("step {} RHS", i + 1),
        );
    }
}

fn check_for_undefined(ctx: &cas_ast::Context, expr_id: cas_ast::ExprId, label: &str) {
    let expr = ctx.get(expr_id);
    match expr {
        Expr::Constant(Constant::Undefined) => {
            println!("FOUND UNDEFINED in {}: ExprId({:?})", label, expr_id);
        }
        Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::Div(a, b) | Expr::Pow(a, b) => {
            check_for_undefined(ctx, *a, label);
            check_for_undefined(ctx, *b, label);
        }
        Expr::Neg(inner) => {
            check_for_undefined(ctx, *inner, label);
        }
        Expr::Function(_, args) => {
            for arg in args {
                check_for_undefined(ctx, *arg, label);
            }
        }
        _ => {}
    }
}
