use cas_engine::Simplifier;
use cas_formatter::DisplayExpr;

#[test]
fn test_prevent_numeric_combine() {
    let mut simplifier = Simplifier::with_default_rules();
    // 2 * 2^(1/2)
    let expr = cas_parser::parse("2 * 2^(1/2)", &mut simplifier.context).unwrap();
    let (res, _) = simplifier.simplify(expr);
    let res_str = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: res
        }
    );
    println!("Result: {}", res_str);

    // We WANT "2 * 2^(1/2)" or "2 * sqrt(2)"
    // We DO NOT want "2^(3/2)" or "sqrt(8)"
    assert!(
        res_str.contains("2 *"),
        "Expected separated coefficient, got: {}",
        res_str
    );
}
