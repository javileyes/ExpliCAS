use cas_ast::DisplayExpr;
use cas_engine::Simplifier;

#[test]
fn test_sqrt_simplification() {
    let mut simplifier = Simplifier::with_default_rules();
    let expr = cas_parser::parse("sqrt(8) + sqrt(2)", &mut simplifier.context).unwrap();
    let (res, _) = simplifier.simplify(expr);
    let res_str = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: res
        }
    );
    println!("Result: {}", res_str);

    // Expected: 3 * 2^(1/2) or 3 * sqrt(2)
    // Current behavior (according to user): 2^(1/2) + 8^(1/2)
    assert!(
        res_str.contains("3") && (res_str.contains("sqrt(2)") || res_str.contains("2^(1/2)")),
        "Expected simplified sqrt, got: {}",
        res_str
    );
}
