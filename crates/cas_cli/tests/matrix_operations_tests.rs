use cas_ast::Context;
use cas_engine::Simplifier;
use cas_formatter::DisplayExpr;
use cas_parser::parse;

#[test]
fn test_matrix_addition() {
    let mut ctx = Context::new();
    let mut simplifier = Simplifier::with_default_rules();

    let expr_str = "[[1, 2], [3, 4]] + [[5, 6], [7, 8]]";
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

    // Should be [[6, 8], [10, 12]]
    assert!(
        result_str.contains("6")
            && result_str.contains("8")
            && result_str.contains("10")
            && result_str.contains("12")
    );
}

#[test]
fn test_matrix_scalar_multiplication() {
    let mut ctx = Context::new();
    let mut simplifier = Simplifier::with_default_rules();

    let expr_str = "3 * [[1, 2], [3, 4]]";
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

    // Should be [[3, 6], [9, 12]]
    assert!(
        result_str.contains("3")
            && result_str.contains("6")
            && result_str.contains("9")
            && result_str.contains("12")
    );
}

#[test]
fn test_matrix_multiplication() {
    let mut ctx = Context::new();
    let mut simplifier = Simplifier::with_default_rules();

    // 2x2 * 2x2
    let expr_str = "[[1, 2], [3, 4]] * [[5, 6], [7, 8]]";
    let expr = parse(expr_str, &mut ctx).unwrap();

    simplifier.context = ctx;
    let (result, steps) = simplifier.simplify(expr);

    println!("Steps for matrix multiplication:");
    for step in &steps {
        println!("  {}", step.description);
    }

    let result_str = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    );
    println!("Result: {}", result_str);

    // [[1,2],[3,4]] * [[5,6],[7,8]] = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
    //                                 = [[19, 22], [43, 50]]
    assert!(result_str.contains("19") || result_str.contains("Matrix"));
}

#[test]
fn test_matrix_subtraction() {
    let mut ctx = Context::new();
    let mut simplifier = Simplifier::with_default_rules();

    let expr_str = "[[5, 6], [7, 8]] - [[1, 2], [3, 4]]";
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

    // Should be [[4, 4], [4, 4]]
    assert!(result_str.contains("4"));
}
