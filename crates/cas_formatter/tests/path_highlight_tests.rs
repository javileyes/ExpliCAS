//! Tests for path-based highlighting (V2.9.16)

use cas_ast::{Context, Expr};
use cas_formatter::{HighlightColor, PathHighlightConfig, PathHighlightedLatexRenderer};

#[test]
fn test_path_highlight_single_occurrence() {
    let mut ctx = Context::new();
    let two = ctx.num(2);
    let x = ctx.var("x");
    // 2 + x
    let expr = ctx.add(Expr::Add(two, x));

    // Highlight only the left child (the 2) at path [0]
    let mut config = PathHighlightConfig::new();
    config.add(vec![0], HighlightColor::Red);

    let renderer = PathHighlightedLatexRenderer {
        context: &ctx,
        id: expr,
        path_highlights: &config,
        hints: None,
        style_prefs: None,
    };

    let latex = renderer.to_latex();
    // Should highlight only the first 2, not any subsequent occurrences
    assert!(
        latex.contains("{\\color{red}{2}}"),
        "Expected red 2, got: {}",
        latex
    );
    assert!(latex.contains("x"), "Expected x in output");
}

#[test]
fn test_path_highlight_avoids_duplicate_values() {
    let mut ctx = Context::new();
    let two1 = ctx.num(2);
    let two2 = ctx.num(2); // Same value, will have same ExprId in DAG
                           // 2 + 2
    let expr = ctx.add(Expr::Add(two1, two2));

    // Highlight only the RIGHT 2 at path [1]
    let mut config = PathHighlightConfig::new();
    config.add(vec![1], HighlightColor::Green);

    let renderer = PathHighlightedLatexRenderer {
        context: &ctx,
        id: expr,
        path_highlights: &config,
        hints: None,
        style_prefs: None,
    };

    let latex = renderer.to_latex();
    // Should have: "2 + {color green 2}" NOT "{color green 2} + {color green 2}"
    // The first 2 should NOT be colored
    assert!(
        !latex.starts_with("{\\color"),
        "Left 2 should NOT be highlighted: {}",
        latex
    );
    assert!(
        latex.contains("{\\color{green}{2}}"),
        "Right 2 should be green: {}",
        latex
    );
}

#[test]
fn test_path_highlight_nested_expression() {
    let mut ctx = Context::new();
    let a = ctx.var("a");
    let b = ctx.var("b");
    // a + b - just test Add which doesn't reorder like Mul
    let expr = ctx.add(Expr::Add(a, b));

    // Highlight the 'b' which is at path [1] (right child of root)
    let mut config = PathHighlightConfig::new();
    config.add(vec![1], HighlightColor::Red);

    let renderer = PathHighlightedLatexRenderer {
        context: &ctx,
        id: expr,
        path_highlights: &config,
        hints: None,
        style_prefs: None,
    };

    let latex = renderer.to_latex();
    assert!(
        latex.contains("{\\color{red}{b}}"),
        "b should be highlighted red: {}",
        latex
    );
    // a should NOT be highlighted
    assert!(
        !latex.contains("{\\color{red}{a}}"),
        "a should NOT be highlighted: {}",
        latex
    );
}

#[test]
fn test_path_highlight_root_node() {
    let mut ctx = Context::new();
    let x = ctx.var("x");

    // Highlight the root (empty path)
    let mut config = PathHighlightConfig::new();
    config.add(vec![], HighlightColor::Green);

    let renderer = PathHighlightedLatexRenderer {
        context: &ctx,
        id: x,
        path_highlights: &config,
        hints: None,
        style_prefs: None,
    };

    let latex = renderer.to_latex();
    assert_eq!(latex, "{\\color{green}{x}}");
}

#[test]
fn test_path_highlight_root_negative_fraction_lifts_sign_outside_fraction() {
    let mut ctx = Context::new();
    let neg_two = ctx.num(-2);
    let x = ctx.var("x");
    let denominator = ctx.call("sqrt", vec![x]);
    let expr = ctx.add(Expr::Div(neg_two, denominator));

    let mut config = PathHighlightConfig::new();
    config.add(vec![], HighlightColor::Green);

    let renderer = PathHighlightedLatexRenderer {
        context: &ctx,
        id: expr,
        path_highlights: &config,
        hints: None,
        style_prefs: None,
    };

    let latex = renderer.to_latex();
    assert_eq!(latex, "{\\color{green}{-\\frac{2}{\\sqrt{x}}}}");
}

#[test]
fn test_path_highlight_root_denominator_product_prefers_numeric_factor_first() {
    let mut ctx = Context::new();
    let one = ctx.num(1);
    let two = ctx.num(2);
    let x = ctx.var("x");
    let sqrt_x = ctx.call("sqrt", vec![x]);
    let cos_sqrt_x = ctx.call("cos", vec![sqrt_x]);
    let sqrt_times_two = ctx.add(Expr::Mul(sqrt_x, two));
    let denominator = ctx.add(Expr::Mul(sqrt_times_two, cos_sqrt_x));
    let expr = ctx.add(Expr::Div(one, denominator));

    let mut config = PathHighlightConfig::new();
    config.add(vec![], HighlightColor::Green);

    let renderer = PathHighlightedLatexRenderer {
        context: &ctx,
        id: expr,
        path_highlights: &config,
        hints: None,
        style_prefs: None,
    };

    let latex = renderer.to_latex();
    assert_eq!(
        latex,
        "{\\color{green}{\\frac{1}{2\\cdot \\cos(\\sqrt{x})\\cdot \\sqrt{x}}}}"
    );
}

#[test]
fn test_path_highlight_renders_negative_mul_factor_as_subtraction() {
    let mut ctx = Context::new();
    let a = ctx.var("a");
    let b = ctx.var("b");
    let c = ctx.var("c");
    let three = ctx.num(3);
    let a3 = ctx.add(Expr::Pow(a, three));
    let left = ctx.add(Expr::Mul(a3, b));
    let neg_c = ctx.add(Expr::Neg(c));
    let right = ctx.add(Expr::Mul(a3, neg_c));
    let expr = ctx.add(Expr::Add(left, right));

    let mut config = PathHighlightConfig::new();
    config.add(vec![], HighlightColor::Green);

    let renderer = PathHighlightedLatexRenderer {
        context: &ctx,
        id: expr,
        path_highlights: &config,
        hints: None,
        style_prefs: None,
    };

    let latex = renderer.to_latex();
    assert!(
        latex.contains("\\color{green}") && latex.contains(" - "),
        "expected highlighted subtraction, got: {latex}"
    );
    assert!(
        !latex.contains("\\cdot -c"),
        "negative factor should not stay inside a product: {latex}"
    );
}

#[test]
fn test_path_highlight_does_not_parenthesize_simple_subtracted_product() {
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let a = ctx.var("a");
    let b = ctx.var("b");
    let product = ctx.add(Expr::Mul(a, b));
    let expr = ctx.add(Expr::Sub(x, product));

    let mut config = PathHighlightConfig::new();
    config.add(vec![1], HighlightColor::Red);

    let renderer = PathHighlightedLatexRenderer {
        context: &ctx,
        id: expr,
        path_highlights: &config,
        hints: None,
        style_prefs: None,
    };

    let latex = renderer.to_latex();
    assert!(
        latex.contains("x - {\\color{red}{a\\cdot b}}"),
        "expected direct subtraction of the highlighted product, got: {latex}"
    );
    assert!(
        !latex.contains("- ({\\color{red}{a\\cdot b}})"),
        "simple product should not be wrapped in subtraction parentheses: {latex}"
    );
}
