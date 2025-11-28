use crate::rule::Rewrite;
use crate::define_rule;
use cas_ast::Expr;
use num_traits::Signed;

define_rule!(
    EvaluateAbsRule,
    "Evaluate Absolute Value",
    |ctx, expr| {
        if let Expr::Function(name, args) = ctx.get(expr) {
            if name == "abs" && args.len() == 1 {
                let arg = args[0];
                
                // Case 1: abs(number)
                let arg_data = ctx.get(arg).clone();
                if let Expr::Number(n) = arg_data {
                    // Always evaluate to positive number
                    let abs_val = ctx.add(Expr::Number(n.abs()));
                    return Some(Rewrite {
                        new_expr: abs_val,
                        description: format!("abs({}) = {}", n, n.abs()),
                    });
                }
                
                // Case 2: abs(-x) -> abs(x)
                if let Expr::Neg(inner) = ctx.get(arg) {
                    // If inner is a number, we can simplify fully: abs(-5) -> 5
                    let inner_data = ctx.get(*inner).clone();
                    if let Expr::Number(n) = inner_data {
                        let abs_val = ctx.add(Expr::Number(n.clone())); // n is already positive if it was inside Neg? No, Neg(5) means -5.
                        // Wait, Expr::Neg(inner) means the expression is -inner.
                        // If inner is 5, then arg is -5.
                        // But we already handled Expr::Number above.
                        // Expr::Number(-5) is a single node.
                        // Expr::Neg(Expr::Number(5)) is also possible depending on parser/simplifier.
                        // Let's handle it.
                        return Some(Rewrite {
                            new_expr: abs_val,
                            description: format!("abs(-{}) = {}", n, n),
                        });
                    }

                    let abs_inner = ctx.add(Expr::Function("abs".to_string(), vec![*inner]));
                    return Some(Rewrite {
                        new_expr: abs_inner,
                        description: "abs(-x) = abs(x)".to_string(),
                    });
                }
            }
        }
        None
    }
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rule::Rule;
    use cas_ast::Context;
    use cas_parser::parse;
    use cas_ast::DisplayExpr;

    #[test]
    fn test_evaluate_abs() {
        let mut ctx = Context::new();
        let rule = EvaluateAbsRule;

        // abs(-5) -> 5
        // Note: Parser might produce Number(-5) or Neg(Number(5)).
        // Our parser likely produces Number(-5) for literals.
        let expr1 = parse("abs(-5)", &mut ctx).expect("Failed to parse abs(-5)");
        let rewrite1 = rule.apply(&mut ctx, expr1).expect("Rule failed to apply");
        assert_eq!(format!("{}", DisplayExpr { context: &ctx, id: rewrite1.new_expr }), "5");

        // abs(5) -> 5
        let expr2 = parse("abs(5)", &mut ctx).expect("Failed to parse abs(5)");
        let rewrite2 = rule.apply(&mut ctx, expr2).expect("Rule failed to apply");
        assert_eq!(format!("{}", DisplayExpr { context: &ctx, id: rewrite2.new_expr }), "5");

        // abs(-x) -> abs(x)
        let expr3 = parse("abs(-x)", &mut ctx).expect("Failed to parse abs(-x)");
        let rewrite3 = rule.apply(&mut ctx, expr3).expect("Rule failed to apply");
        assert_eq!(format!("{}", DisplayExpr { context: &ctx, id: rewrite3.new_expr }), "|x|");
    }
}

pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(EvaluateAbsRule));
}
