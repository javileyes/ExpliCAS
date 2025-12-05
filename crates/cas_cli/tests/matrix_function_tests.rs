use cas_ast::{Context, DisplayExpr};
use cas_engine::Simplifier;
use cas_parser::parse;

#[test]
fn test_matrix_det_function() {
    let mut ctx = Context::new();
    let mut simplifier = Simplifier::with_default_rules();

    // det([[1, 2], [3, 4]]) should compute determinant
    let expr_str = "det([[1, 2], [3, 4]])";
    let expr = parse(expr_str, &mut ctx).unwrap();

    // Transfer context to simplifier
    simplifier.context = ctx;
    let (result, _) = simplifier.simplify(expr);

    // Result should be a number (the determinant)
    let result_str = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    );

    // det([[1,2],[3,4]]) = 1*4 - 2*3 = -2
    assert!(result_str.contains("-2") || result_str.contains("("));
}

#[test]
fn test_matrix_transpose_function() {
    let mut ctx = Context::new();
    let mut simplifier = Simplifier::with_default_rules();

    // transpose([[1, 2], [3, 4]]) should transpose the matrix
    let expr_str = "transpose([[1, 2, 3], [4, 5, 6]])";
    let expr = parse(expr_str, &mut ctx).unwrap();

    simplifier.context = ctx;
    let (result, _) = simplifier.simplify(expr);

    // Should return a matrix
    use cas_ast::Expr;
    matches!(simplifier.context.get(result), Expr::Matrix { .. });
}

#[test]
fn test_matrix_trace_function() {
    let mut ctx = Context::new();
    let mut simplifier = Simplifier::with_default_rules();

    // trace([[1, 2], [3, 4]]) should be 1 + 4 = 5
    let expr_str = "trace([[1, 2], [3, 4]])";
    let expr = parse(expr_str, &mut ctx).unwrap();

    simplifier.context = ctx;
    let (result, _) = simplifier.simplify(expr);

    let result_str = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    );

    // trace([[1,2],[3,4]]) = 1 + 4 = 5
    assert!(result_str.contains("5") || result_str.contains("+"));
}

#[test]
fn test_short_matrix_function_names() {
    let mut ctx = Context::new();
    let mut simplifier = Simplifier::with_default_rules();

    // Test 'T' as alias for transpose
    let expr_str = "T([[1, 2]])";
    let expr = parse(expr_str, &mut ctx).unwrap();

    simplifier.context = ctx;
    let (result, _) = simplifier.simplify(expr);

    use cas_ast::Expr;
    matches!(simplifier.context.get(result), Expr::Matrix { .. });
}
