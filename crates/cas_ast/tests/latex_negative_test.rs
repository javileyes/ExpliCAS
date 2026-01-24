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
    let cos_4x = ctx.call("cos", vec![four_x]);

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

// ============================================================================
// Non-regression tests for Neg(Add/Sub) parenthesization
// Bug fix: Ensures -(a + b) is rendered as "-(a + b)" not "-a + b"
// ============================================================================

/// Test that Neg(Add) in subtraction context has parentheses in LaTeX
#[test]
fn test_latex_neg_add_has_parentheses() {
    let mut ctx = Context::new();

    // Build: x - (a + b) as Add(x, Neg(Add(a, b)))
    let x = ctx.var("x");
    let a = ctx.var("a");
    let b = ctx.var("b");

    let a_plus_b = ctx.add(Expr::Add(a, b));
    let neg_sum = ctx.add(Expr::Neg(a_plus_b));
    let expr = ctx.add(Expr::Add(x, neg_sum));

    let latex = LaTeXExpr {
        context: &ctx,
        id: expr,
    };

    let result = latex.to_latex();
    println!("LaTeX Neg(Add) test: {}", result);

    // Must have parentheses around the subtracted sum
    assert!(
        result.contains("(a + b)") || result.contains("(b + a)"),
        "Expected parentheses around '(a + b)' in LaTeX. Got: '{}'",
        result
    );
}

/// Test that unary Neg(Add) has parentheses: `-(a + b)` in LaTeX
#[test]
fn test_latex_unary_neg_add_has_parentheses() {
    let mut ctx = Context::new();

    // Build: -(a + b)
    let a = ctx.var("a");
    let b = ctx.var("b");

    let a_plus_b = ctx.add(Expr::Add(a, b));
    let neg_sum = ctx.add(Expr::Neg(a_plus_b));

    let latex = LaTeXExpr {
        context: &ctx,
        id: neg_sum,
    };

    let result = latex.to_latex();
    println!("LaTeX unary Neg(Add) test: {}", result);

    // Must have -(...)
    assert!(
        result.starts_with("-(") || result.contains("-("),
        "Expected '-(...)' format for unary neg of sum in LaTeX. Got: '{}'",
        result
    );
}

/// Regression test: fraction numerator with subtracted sum in LaTeX
#[test]
fn test_latex_fraction_numerator_neg_sum_regression() {
    let mut ctx = Context::new();

    // Build: x + 1 - (3x + 2) - the problematic numerator from the bug report
    let x = ctx.var("x");
    let one = ctx.num(1);
    let two = ctx.num(2);
    let three = ctx.num(3);

    let x_plus_1 = ctx.add(Expr::Add(x, one));
    let three_x = ctx.add(Expr::Mul(three, x));
    let three_x_plus_2 = ctx.add(Expr::Add(three_x, two));
    let neg_sum = ctx.add(Expr::Neg(three_x_plus_2));
    let numerator = ctx.add(Expr::Add(x_plus_1, neg_sum));

    let latex = LaTeXExpr {
        context: &ctx,
        id: numerator,
    };

    let result = latex.to_latex();
    println!("LaTeX regression test result: {}", result);

    // The critical check: should NOT have unparenthesized "+ 2"
    // Pattern "- 3" followed by "+ 2" (without parens) indicates missing parentheses
    // Correct output has "+ 2" INSIDE parentheses like "- (3\cdot x + 2)"
    assert!(
        result.contains("(3\\cdot x + 2)") || result.contains("(2 + 3\\cdot x)"),
        "REGRESSION: Missing parentheses around '3x + 2'. \
         Expected something like 'x + 1 - (3\\cdot x + 2)'. Got: '{}'",
        result
    );
}
