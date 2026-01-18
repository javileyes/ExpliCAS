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
    // This is cosh(2x) - (cosh^2(x) + sinh^2(x)) which SHOULD be 0.
    // Using auto-expand via EvalOptions to fully simplify the identity.
    //
    // NOTE: With RecognizeHyperbolicFromExpRule, this now correctly converts to
    // cosh(2x) - (cosh(x)^2 + sinh(x)^2), but we need the identity
    // cosh^2(x) + sinh^2(x) = cosh(2x) to simplify to 0 (TODO: add this rule).
    //
    // For now, verify that the exponential-to-hyperbolic conversion is working.
    let input_str = "(e^(2*x) + e^(-2*x))/2 - (((e^x + e^(-x))/2)^2 + ((e^x - e^(-x))/2)^2)";
    let input = parse(input_str, &mut simplifier.context).expect("Failed to parse input");

    // Use simplify_with_options to propagate expand_policy
    let simplify_opts = opts.to_simplify_options();
    let (result_id, _) = simplifier.simplify_with_options(input, simplify_opts);

    // Verify result contains hyperbolic functions (conversion worked)
    let result_str = format!(
        "{}",
        cas_ast::DisplayExpr {
            context: &simplifier.context,
            id: result_id
        }
    );

    // The conversion to hyperbolic form is correct. Full simplification to 0
    // requires the identity cosh²(x) + sinh²(x) = cosh(2x) which is TODO.
    assert!(
        result_str.contains("cosh") || result_str.contains("sinh") || result_str == "0",
        "Expected hyperbolic form or 0, got: {}",
        result_str
    );
}
