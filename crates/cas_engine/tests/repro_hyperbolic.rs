use cas_engine::options::EvalOptions;
use cas_engine::phase::ExpandPolicy;
use cas_engine::Simplifier;
use cas_parser::parse;

/// Options with auto-expand enabled for identity tests
fn opts_autoexpand() -> EvalOptions {
    EvalOptions {
        expand_policy: ExpandPolicy::Auto,
        ..Default::default()
    }
}

#[test]
fn test_hyperbolic_double_angle_exponential() {
    let opts = opts_autoexpand();
    let mut simplifier = Simplifier::with_profile(&opts);
    simplifier.enable_debug();

    // (e^(2*x) + e^(-2*x))/2 - (((e^x + e^(-x))/2)^2 + ((e^x - e^(-x))/2)^2)
    // This is cosh(2x) - (cosh^2(x) + sinh^2(x)) which should be 0.
    // Using auto-expand via EvalOptions to fully simplify the identity.
    let input_str = "(e^(2*x) + e^(-2*x))/2 - (((e^x + e^(-x))/2)^2 + ((e^x - e^(-x))/2)^2)";
    let input = parse(input_str, &mut simplifier.context).expect("Failed to parse input");

    // Use simplify_with_options to propagate expand_policy
    let simplify_opts = opts.to_simplify_options();
    let (result_id, _) = simplifier.simplify_with_options(input, simplify_opts);
    let result_expr = simplifier.context.get(result_id).clone();

    println!("Result: {:?}", result_expr);

    // Check if result is 0
    let zero = simplifier.context.num(0);
    use cas_engine::ordering::compare_expr;
    use std::cmp::Ordering;
    assert_eq!(
        compare_expr(&simplifier.context, result_id, zero),
        Ordering::Equal,
        "Expected 0, got {:?}",
        simplifier.context.get(result_id)
    );
}
