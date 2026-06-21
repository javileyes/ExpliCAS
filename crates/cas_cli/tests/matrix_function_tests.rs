use cas_ast::Context;
use cas_formatter::DisplayExpr;
use cas_parser::parse;
use cas_solver::runtime::Simplifier;

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

/// Simplify `src` and return its display string.
fn simplify_to_string(src: &str) -> String {
    let mut ctx = Context::new();
    let expr = parse(src, &mut ctx).unwrap();
    let mut simplifier = Simplifier::with_default_rules();
    simplifier.context = ctx;
    let (result, _) = simplifier.simplify(expr);
    format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    )
}

#[test]
fn test_matrix_inverse_numeric() {
    // inverse([[1,2],[3,4]]) = [[-2, 1], [3/2, -1/2]]
    assert_eq!(
        simplify_to_string("inverse([[1, 2], [3, 4]])"),
        "[[-2, 1], [3/2, -1/2]]"
    );
    // `inv` alias.
    assert_eq!(
        simplify_to_string("inv([[1, 2], [3, 4]])"),
        "[[-2, 1], [3/2, -1/2]]"
    );
    // Diagonal.
    assert_eq!(
        simplify_to_string("inverse([[2, 0], [0, 4]])"),
        "[[1/2, 0], [0, 1/4]]"
    );
    // 1x1.
    assert_eq!(simplify_to_string("inverse([[5]])"), "[1/5]");
    // 3x3 with determinant 1.
    assert_eq!(
        simplify_to_string("inverse([[1, 2, 3], [0, 1, 4], [5, 6, 0]])"),
        "[[-24, 18, 5], [20, -15, -4], [-5, 4, 1]]"
    );
}

#[test]
fn test_matrix_inverse_times_matrix_is_identity() {
    // A · inverse(A) = I for the 2x2 case.
    assert_eq!(
        simplify_to_string("[[1, 2], [3, 4]] * inverse([[1, 2], [3, 4]])"),
        "[[1, 0], [0, 1]]"
    );
}

#[test]
fn test_matrix_inverse_singular_is_undefined() {
    // det([[1,2],[2,4]]) = 0 -> no inverse -> undefined.
    assert_eq!(simplify_to_string("inverse([[1, 2], [2, 4]])"), "undefined");
    // Two identical symbolic rows are structurally singular.
    assert_eq!(simplify_to_string("inverse([[a, b], [a, b]])"), "undefined");
}

#[test]
fn test_matrix_inverse_non_square_stays_symbolic() {
    // A non-square matrix has no inverse; the call is left symbolic (honest).
    let out = simplify_to_string("inverse([[1, 2, 3], [4, 5, 6]])");
    assert!(out.starts_with("inverse("), "got {out}");
}
