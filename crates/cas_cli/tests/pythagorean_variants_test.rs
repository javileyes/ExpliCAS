use cas_engine::Simplifier;
use cas_formatter::DisplayExpr;
use cas_parser::parse;

fn simplify_str(input: &str) -> String {
    let mut simplifier = Simplifier::with_default_rules();
    let expr = parse(input, &mut simplifier.context).unwrap();
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
fn test_sec_tan_minus_one_equals_zero() {
    let result = simplify_str("sec(x)^2 - tan(x)^2 - 1");
    assert_eq!(result, "0", "sec²-tan²-1 should simplify to 0");
}

#[test]
fn test_sec_tan_equals_one() {
    let result = simplify_str("sec(x)^2 - tan(x)^2");
    assert_eq!(result, "1", "sec²-tan² should simplify to 1");
}

#[test]
fn test_csc_cot_minus_one_equals_zero() {
    let result = simplify_str("csc(x)^2 - cot(x)^2 - 1");
    assert_eq!(result, "0", "csc²-cot²-1 should simplify to 0");
}
