use cas_engine::Simplifier;
use cas_parser::parse;

#[test]
fn test_hyperbolic_double_angle_exponential() {
    let mut simplifier = Simplifier::with_default_rules();
    simplifier.enable_debug();

    // (e^(2*x) + e^(-2*x))/2 - (((e^x + e^(-x))/2)^2 + ((e^x - e^(-x))/2)^2)
    // This is cosh(2x) - (cosh^2(x) + sinh^2(x)) which should be 0.
    let input_str = "(e^(2*x) + e^(-2*x))/2 - (((e^x + e^(-x))/2)^2 + ((e^x - e^(-x))/2)^2)";
    let input = parse(input_str, &mut simplifier.context).expect("Failed to parse input");

    let (result_id, _) = simplifier.simplify(input);
    let result_expr = simplifier.context.get(result_id).clone();

    println!("Result: {:?}", result_expr);

    // Check if result is 0
    let zero = simplifier.context.num(0);
    use cas_engine::ordering::compare_expr;
    use std::cmp::Ordering;
    assert_eq!(compare_expr(&simplifier.context, result_id, zero), Ordering::Equal, "Expected 0, got {:?}", simplifier.context.get(result_id));
}
