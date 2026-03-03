use super::pythagorean::TrigPythagoreanSimplifyRule;
use crate::rule::SimpleRule;
use cas_ast::{Context, Expr};

#[test]
fn test_one_minus_sin_squared() {
    let mut ctx = Context::new();
    // 1 - sin²(x) should become cos²(x)
    let x = ctx.var("x");
    let sin_x = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![x]);
    let two = ctx.num(2);
    let sin_sq = ctx.add(Expr::Pow(sin_x, two));
    let neg_sin_sq = ctx.add(Expr::Neg(sin_sq));
    let one = ctx.num(1);
    let expr = ctx.add(Expr::Add(one, neg_sin_sq)); // 1 + (-sin²(x))

    let rule = TrigPythagoreanSimplifyRule;
    let result = rule.apply_simple(&mut ctx, expr);

    assert!(result.is_some(), "Rule should apply to 1 - sin²(x)");
    let rewrite = result.unwrap();

    // Result should be cos²(x)
    if let Expr::Mul(_coeff, pow) = ctx.get(rewrite.new_expr) {
        if let Expr::Pow(base, _) = ctx.get(*pow) {
            if let Expr::Function(name_id, _) = ctx.get(*base) {
                assert_eq!(ctx.sym_name(*name_id), "cos");
            }
        }
    }
}
