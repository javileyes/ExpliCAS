use super::functions::EvaluateAbsRule;
use crate::rule::Rule;
use cas_ast::Context;
use cas_formatter::DisplayExpr;
use cas_parser::parse;

#[test]
fn test_evaluate_abs() {
    let mut ctx = Context::new();
    let rule = EvaluateAbsRule;

    // abs(-5) -> 5
    // Note: Parser might produce Number(-5) or Neg(Number(5)).
    // Our parser likely produces Number(-5) for literals.
    let expr1 = parse("abs(-5)", &mut ctx).expect("Failed to parse abs(-5)");
    let rewrite1 = rule
        .apply(
            &mut ctx,
            expr1,
            &crate::parent_context::ParentContext::root(),
        )
        .expect("Rule failed to apply");
    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite1.new_expr
            }
        ),
        "5"
    );

    // abs(5) -> 5
    let expr2 = parse("abs(5)", &mut ctx).expect("Failed to parse abs(5)");
    let rewrite2 = rule
        .apply(
            &mut ctx,
            expr2,
            &crate::parent_context::ParentContext::root(),
        )
        .expect("Rule failed to apply");
    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite2.new_expr
            }
        ),
        "5"
    );

    // abs(-x) -> abs(x)
    let expr3 = parse("abs(-x)", &mut ctx).expect("Failed to parse abs(-x)");
    let rewrite3 = rule
        .apply(
            &mut ctx,
            expr3,
            &crate::parent_context::ParentContext::root(),
        )
        .expect("Rule failed to apply");
    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite3.new_expr
            }
        ),
        "|x|"
    );
}
