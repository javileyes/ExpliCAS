use cas_ast::{latex::LaTeXExpr, Context, Expr};

#[test]
fn test_negative_addition() {
    let mut ctx = Context::new();

    // Test: -3 + -1 + 4 should render as -3 - 1 + 4
    let neg_three = ctx.num(-3);
    let neg_one = ctx.num(-1);
    let four = ctx.num(4);

    // (-3) + (-1)
    let sum1 = ctx.add(Expr::Add(neg_three, neg_one));
    // ((-3) + (-1)) + 4
    let sum2 = ctx.add(Expr::Add(sum1, four));

    let latex = LaTeXExpr {
        context: &ctx,
        id: sum2,
    };

    let result = latex.to_latex();
    println!("Result with Number(-1): {}", result);

    // Should be "-3 - 1 + 4", not "-3 + -1 + 4"
    assert!(result.contains("- 1"), "Expected '- 1' but got: {}", result);
    assert!(
        !result.contains("+ -"),
        "Should not contain '+ -', got: {}",
        result
    );
}

#[test]
fn test_negative_addition_with_neg_expr() {
    let mut ctx = Context::new();

    // Test: -3 + Neg(1) + 4 should render as -3 - 1 + 4
    let neg_three = ctx.num(-3);
    let one = ctx.num(1);
    let neg_one = ctx.add(Expr::Neg(one)); // Using Neg expression
    let four = ctx.num(4);

    // (-3) + Neg(1)
    let sum1 = ctx.add(Expr::Add(neg_three, neg_one));
    // ((-3) + Neg(1)) + 4
    let sum2 = ctx.add(Expr::Add(sum1, four));

    let latex = LaTeXExpr {
        context: &ctx,
        id: sum2,
    };

    let result = latex.to_latex();
    println!("Result with Neg(1): {}", result);

    // Should be "-3 - 1 + 4", not "-3 + -1 + 4"
    assert!(result.contains("- 1"), "Expected '- 1' but got: {}", result);
    assert!(
        !result.contains("+ -"),
        "Should not contain '+ -', got: {}",
        result
    );
}

#[test]
fn test_addition_with_mul_negative() {
    let mut ctx = Context::new();

    // Test: x + (-1)*y should render as x - y, not x + -1y
    let x = ctx.var("x");
    let neg_one = ctx.num(-1);
    let y = ctx.var("y");

    // (-1) * y
    let neg_y = ctx.add(Expr::Mul(neg_one, y));
    // x + ((-1) * y)
    let sum = ctx.add(Expr::Add(x, neg_y));

    let latex = LaTeXExpr {
        context: &ctx,
        id: sum,
    };

    let result = latex.to_latex();
    println!("Result with Mul(-1, y): {}", result);

    // Should NOT contain "+ -"
    assert!(
        !result.contains("+ -"),
        "Should not contain '+ -', got: {}",
        result
    );
}

#[test]
fn test_addition_with_neg_one_times_function() {
    let mut ctx = Context::new();

    // Test: -3 + (-1)*cos(4x) should render as -3 - cos(4x), not -3 + -cos(4x)
    let neg_three = ctx.num(-3);
    let neg_one = ctx.num(-1);
    let four = ctx.num(4);
    let x = ctx.var("x");
    let four_x = ctx.add(Expr::Mul(four, x));
    let cos_4x = ctx.add(Expr::Function("cos".to_string(), vec![four_x]));

    // (-1) * cos(4x)
    let neg_cos = ctx.add(Expr::Mul(neg_one, cos_4x));
    // -3 + (-1)*cos(4x)
    let sum = ctx.add(Expr::Add(neg_three, neg_cos));

    let latex = LaTeXExpr {
        context: &ctx,
        id: sum,
    };

    let result = latex.to_latex();
    println!("Result with -3 + (-1)*cos(4x): {}", result);

    // Should NOT contain "+ -"
    assert!(
        !result.contains("+ -"),
        "Should not contain '+ -', got: {}",
        result
    );
    // Should contain "- \"
    assert!(
        result.contains("- "),
        "Should contain '- ', got: {}",
        result
    );
}
