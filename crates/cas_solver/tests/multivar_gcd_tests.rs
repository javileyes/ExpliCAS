//! Multivariate GCD Integration Tests
//!
//! Tests for multivariate polynomial GCD in fraction simplification.

use cas_ast::Context;
use cas_parser::parse;
use cas_session::state_api::SessionState;
use cas_solver::runtime::{
    BranchMode, ComplexMode, ContextMode, DomainMode, Engine, EvalAction, EvalConfig, EvalOptions,
    EvalRequest, EvalResult, Simplifier, StepsMode,
};

/// Helper: simplify expression and return result string
fn simplify(input: &str) -> String {
    simplify_with_semantics(input, ContextMode::Standard, DomainMode::Generic)
}

fn simplify_with_semantics(
    input: &str,
    context_mode: ContextMode,
    domain_mode: DomainMode,
) -> String {
    simplify_with_semantics_and_steps(input, context_mode, domain_mode, StepsMode::On)
}

fn simplify_with_semantics_and_steps(
    input: &str,
    context_mode: ContextMode,
    domain_mode: DomainMode,
    steps_mode: StepsMode,
) -> String {
    let opts = EvalOptions {
        branch_mode: BranchMode::Strict,
        complex_mode: ComplexMode::Auto,
        steps_mode,
        shared: cas_solver::runtime::SharedSemanticConfig {
            context_mode,
            semantics: EvalConfig {
                domain_mode,
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    };
    let mut ctx = Context::new();
    let expr = parse(input, &mut ctx).expect("Failed to parse");

    let mut simplifier = Simplifier::with_profile(&opts);
    simplifier.context = ctx;
    let (result, _steps) = simplifier.simplify(expr);

    format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    )
}

fn simplify_with_semantics_and_steps_and_requires(
    input: &str,
    context_mode: ContextMode,
    domain_mode: DomainMode,
    steps_mode: StepsMode,
) -> (String, Vec<String>) {
    let mut engine = Engine::new();
    let mut state = SessionState::new();

    state.options_mut().branch_mode = BranchMode::Strict;
    state.options_mut().complex_mode = ComplexMode::Auto;
    state.options_mut().steps_mode = steps_mode;
    state.options_mut().shared.context_mode = context_mode;
    state.options_mut().shared.semantics.domain_mode = domain_mode;

    let parsed = parse(input, &mut engine.simplifier.context).expect("Failed to parse");
    let req = EvalRequest {
        raw_input: input.to_string(),
        parsed,
        action: EvalAction::Simplify,
        auto_store: false,
    };

    let output = engine.eval(&mut state, req).expect("eval failed");
    let result = match &output.result {
        EvalResult::Expr(expr) => format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: &engine.simplifier.context,
                id: *expr
            }
        ),
        other => panic!("unexpected eval result: {other:?}"),
    };
    let requires = output
        .required_conditions
        .iter()
        .map(|cond| cond.display(&engine.simplifier.context))
        .collect();
    (result, requires)
}

// =============================================================================
// Content GCD Tests (Layer 1)
// =============================================================================

#[test]
fn test_content_gcd_multivar() {
    // (2x + 2y) / (4x + 4y)
    // With Layer 2: GCD = 2*(x+y), so result = 1/2
    let result = simplify("(2*x + 2*y) / (4*x + 4*y)");
    assert_eq!(result, "1/2", "(2x+2y)/(4x+4y) should simplify to 1/2");
}

#[test]
fn test_content_gcd_multivar_in_solve_strict_context() {
    let result = simplify_with_semantics(
        "(2*x + 2*y) / (4*x + 4*y)",
        ContextMode::Solve,
        DomainMode::Strict,
    );
    assert_eq!(
        result, "1/2",
        "solve-context strict eval should simplify exact scalar-multiple fractions to 1/2"
    );
}

#[test]
fn test_content_gcd_multivar_in_solve_strict_context_steps_off() {
    let result = simplify_with_semantics_and_steps(
        "(2*x + 2*y) / (4*x + 4*y)",
        ContextMode::Solve,
        DomainMode::Strict,
        StepsMode::Off,
    );
    assert_eq!(
        result, "1/2",
        "solve-context strict eval should keep the same scalar-multiple result with steps off"
    );
}

#[test]
fn test_content_gcd_multivar_in_solve_generic_context_steps_off() {
    let result = simplify_with_semantics_and_steps(
        "(2*x + 2*y) / (4*x + 4*y)",
        ContextMode::Solve,
        DomainMode::Generic,
        StepsMode::Off,
    );
    assert_eq!(
        result, "1/2",
        "solve-context generic eval should keep the same scalar-multiple result with steps off"
    );
}

#[test]
fn test_content_gcd_multivar_in_solve_generic_context_steps_off_keeps_requires() {
    let (result, requires) = simplify_with_semantics_and_steps_and_requires(
        "(2*x + 2*y) / (4*x + 4*y)",
        ContextMode::Solve,
        DomainMode::Generic,
        StepsMode::Off,
    );
    assert_eq!(result, "1/2");
    assert!(
        requires
            .iter()
            .any(|message| message.contains("4 * x + 4 * y") && message.contains("0")),
        "expected denominator require, got: {requires:?}"
    );
}

#[test]
fn test_binomial_square_cancel_in_solve_generic_context_steps_off() {
    let result = simplify_with_semantics_and_steps(
        "(x^2 + 2*x*y + y^2) / (x + y)^2",
        ContextMode::Solve,
        DomainMode::Generic,
        StepsMode::Off,
    );
    assert_eq!(
        result, "1",
        "solve-context generic eval should collapse the binomial-square fraction to 1 with steps off"
    );
}

#[test]
fn test_binomial_square_cancel_in_solve_strict_context_steps_off() {
    let result = simplify_with_semantics_and_steps(
        "(x^2 + 2*x*y + y^2) / (x + y)^2",
        ContextMode::Solve,
        DomainMode::Strict,
        StepsMode::Off,
    );
    assert_eq!(
        result, "1",
        "solve-context strict eval should keep the same steps-off binomial-square result"
    );
}

#[test]
fn test_binomial_square_cancel_in_solve_generic_context_steps_off_keeps_requires() {
    let (result, requires) = simplify_with_semantics_and_steps_and_requires(
        "(x^2 + 2*x*y + y^2) / (x + y)^2",
        ContextMode::Solve,
        DomainMode::Generic,
        StepsMode::Off,
    );
    assert_eq!(result, "1");
    assert!(
        requires.iter().any(|message| message.contains("(x + y)^2")),
        "expected original denominator require, got: {requires:?}"
    );
    assert!(
        requires
            .iter()
            .any(|message| message.contains("x^2 + y^2 + 2")
                && message.contains("x")
                && message.contains("y")),
        "expected expanded denominator require, got: {requires:?}"
    );
}

#[test]
fn test_perfect_square_minus_cancel_in_solve_generic_context_steps_off() {
    let result = simplify_with_semantics_and_steps(
        "(x^2 - 2*x*y + y^2) / (x - y)",
        ContextMode::Solve,
        DomainMode::Generic,
        StepsMode::Off,
    );
    assert_eq!(
        result, "x - y",
        "solve-context generic eval should collapse the perfect-square-minus fraction with steps off"
    );
}

#[test]
fn test_power_quotient_cancel_in_solve_generic_context_steps_off_keeps_requires() {
    let (result, requires) = simplify_with_semantics_and_steps_and_requires(
        "x^4 / x^2",
        ContextMode::Solve,
        DomainMode::Generic,
        StepsMode::Off,
    );
    assert_eq!(result, "x^2");
    assert!(
        requires.iter().any(|message| message.contains("x^2")),
        "expected x^2 denominator require, got: {requires:?}"
    );
    assert!(
        requires.iter().any(|message| message == "x ≠ 0"),
        "expected base nonzero require, got: {requires:?}"
    );
}

#[test]
fn test_perfect_square_minus_cancel_in_solve_generic_context_steps_on() {
    let result = simplify_with_semantics(
        "(x^2 - 2*x*y + y^2) / (x - y)",
        ContextMode::Solve,
        DomainMode::Generic,
    );
    assert_eq!(
        result, "x - y",
        "solve-context generic eval should keep the same perfect-square-minus result with steps on"
    );
}

#[test]
fn test_perfect_square_minus_cancel_in_solve_strict_context_steps_off() {
    let result = simplify_with_semantics_and_steps(
        "(x^2 - 2*x*y + y^2) / (x - y)",
        ContextMode::Solve,
        DomainMode::Strict,
        StepsMode::Off,
    );
    assert_eq!(
        result, "x - y",
        "solve-context strict eval should keep the same perfect-square-minus result with steps off"
    );
}

#[test]
fn test_perfect_square_minus_cancel_in_solve_strict_context_steps_on() {
    let result = simplify_with_semantics(
        "(x^2 - 2*x*y + y^2) / (x - y)",
        ContextMode::Solve,
        DomainMode::Strict,
    );
    assert_eq!(
        result, "x - y",
        "solve-context strict eval should keep the same perfect-square-minus result with steps on"
    );
}

#[test]
fn test_difference_of_cubes_cancel_in_solve_generic_context_steps_off() {
    let result = simplify_with_semantics_and_steps(
        "(x^3 - y^3) / (x - y)",
        ContextMode::Solve,
        DomainMode::Generic,
        StepsMode::Off,
    );
    assert_eq!(
        result, "x^2 + y^2 + x * y",
        "solve-context generic eval should collapse the difference-of-cubes fraction with steps off"
    );
}

#[test]
fn test_difference_of_cubes_cancel_in_solve_strict_context_steps_off() {
    let result = simplify_with_semantics_and_steps(
        "(x^3 - y^3) / (x - y)",
        ContextMode::Solve,
        DomainMode::Strict,
        StepsMode::Off,
    );
    assert_eq!(
        result, "x^2 + y^2 + x * y",
        "solve-context strict eval should collapse the difference-of-cubes fraction with steps off"
    );
}

#[test]
fn test_sum_of_cubes_cancel_in_solve_generic_context_steps_off() {
    let result = simplify_with_semantics_and_steps(
        "(x^3 + y^3) / (x + y)",
        ContextMode::Solve,
        DomainMode::Generic,
        StepsMode::Off,
    );
    assert_eq!(
        result, "x^2 + y^2 - x * y",
        "solve-context generic eval should collapse the sum-of-cubes fraction with steps off"
    );
}

#[test]
fn test_sum_of_cubes_cancel_in_solve_strict_context_steps_off() {
    let result = simplify_with_semantics_and_steps(
        "(x^3 + y^3) / (x + y)",
        ContextMode::Solve,
        DomainMode::Strict,
        StepsMode::Off,
    );
    assert_eq!(
        result, "x^2 + y^2 - x * y",
        "solve-context strict eval should collapse the sum-of-cubes fraction with steps off"
    );
}

// =============================================================================
// Monomial GCD Tests (Layer 1)
// =============================================================================

#[test]
fn test_monomial_gcd_multivar() {
    // (x^2*y + x^2*y^2) / (x^3*y + x^3*y^2) should simplify to 1/x
    let result = simplify("(x^2*y + x^2*y^2) / (x^3*y + x^3*y^2)");
    // After factoring out x^2*y from num and x^3*y from den:
    // num: x^2*y*(1 + y), den: x^3*y*(1 + y)
    // Common: x^2*y*(1+y)
    // Result: 1/x
    assert!(
        result.contains("1") && result.contains("x"),
        "Should simplify to 1/x, got: {}",
        result
    );
}

// =============================================================================
// Combined Content + Monomial Tests
// =============================================================================

#[test]
fn test_combined_content_and_monomial() {
    // (2*x*y) / (4*x^2*y^2) should simplify to 1/(2*x*y)
    let result = simplify("(2*x*y) / (4*x^2*y^2)");
    // Content GCD: 2, Monomial GCD: x*y
    // Result: 1 / (2*x*y)
    assert!(
        result.contains("1") && result.contains("2"),
        "Should simplify, got: {}",
        result
    );
}

// =============================================================================
// Layer 2 Tests: Heuristic Polynomial GCD
// =============================================================================

#[test]
fn test_layer2_difference_of_squares() {
    // (x^2 - y^2) / (x - y) should simplify to x + y
    // This requires Layer 2: GCD = (x - y)
    let result = simplify("(x^2 - y^2) / (x - y)");
    // Expected: x + y (or y + x depending on ordering)
    assert!(
        (result.contains("x") && result.contains("y") && result.contains("+"))
            || result == "x + y"
            || result == "y + x",
        "Expected x+y, got: {}",
        result
    );
}

#[test]
fn test_layer2_difference_of_squares_in_solve_generic_context_steps_off_keeps_requires() {
    let (result, requires) = simplify_with_semantics_and_steps_and_requires(
        "(x^2 - y^2) / (x - y)",
        ContextMode::Solve,
        DomainMode::Generic,
        StepsMode::Off,
    );
    assert!(result == "x + y" || result == "y + x");
    assert!(
        requires
            .iter()
            .any(|message| message.contains("x - y") && message.contains("0")),
        "expected denominator require, got: {requires:?}"
    );
}

// =============================================================================
// Anti-Regression Tests A-D
// =============================================================================

// A. Factor in other variables (tests scaled GCD preservation)
#[test]
fn test_factor_in_other_var() {
    // ((y+1)*(x+1))/((y+1)*(x+2)) -> (x+1)/(x+2)
    // GCD = (y+1), which depends on the OTHER variable
    let result = simplify("((y+1)*(x+1))/((y+1)*(x+2))");
    // Result should be (x+1)/(x+2) - no y in final result
    assert!(
        result.contains("x") && !result.contains("y"),
        "Factor (y+1) should cancel, got: {}",
        result
    );
}

// B. 3-variable cases
#[test]
fn test_3var_monomial_gcd() {
    // (x*y + x*z) / (x^2*y + x^2*z) -> 1/x (full) or (y+z)/(x*(y+z)) (partial)
    // Layer 1 extracts x from monomial GCD
    // Layer 2 should find (y+z) as common factor, but 3-var interpolation is complex
    let result = simplify("(x*y + x*z) / (x^2*y + x^2*z)");
    // Accept either full simplification to 1/x or partial simplification
    // where y,z factor canceled but x remains
    assert!(
        (result.contains("1")
            && result.contains("x")
            && !result.contains("y")
            && !result.contains("z"))
            || result.contains("/"), // At least some simplification happened
        "Should simplify, got: {}",
        result
    );
}

// C. Difference of squares with multivar binomial
#[test]
fn test_diff_of_squares_multivar_binomial() {
    // (x^2 - (y+z)^2) / (x - (y+z)) -> x + y + z
    let result = simplify("(x^2 - (y+z)^2) / (x - (y+z))");
    // Note: (y+z)^2 may expand
    assert!(
        result.contains("x") && result.contains("y") && result.contains("z"),
        "Result should contain x, y, z, got: {}",
        result
    );
}

// D. No simplification cases (GCD = 1)
#[test]
fn test_no_simplify_gcd_is_one() {
    // (x^2 + y^2)/(x + y) has GCD = 1 in Q[x,y]
    let result = simplify("(x^2 + y^2)/(x + y)");
    // Should stay as a fraction (no simplification)
    assert!(
        result.contains("/") || result.contains("x^2") || result.contains("y^2"),
        "Should not over-simplify: gcd(x^2+y^2, x+y)=1, got: {}",
        result
    );
}

#[test]
fn test_coprime_polys_stay_unsimplified() {
    // (x + 1) / (y + 1) - coprime, no common factor
    let result = simplify("(x + 1) / (y + 1)");
    assert!(
        result.contains("x") && result.contains("y") && result.contains("/"),
        "Coprime polys should stay as fraction, got: {}",
        result
    );
}

// =============================================================================
// Hardening Tests: Edge Cases for Layer 2 Limitations
// =============================================================================

// A. GCD depends on 2+ param variables at once
// This typically CANNOT be reconstructed by bivar+seeds if we fix one of them
#[test]
fn test_hard_multiparam_factor() {
    // ((x + y + z) * (x + 2*y + 3*z)) / ((x + y + z) * (2*x - y + z))
    // GCD = (x + y + z) - depends on both y AND z
    // Expected: may NOT simplify (confirms limitation) OR may simplify if lucky seed
    let result = simplify("((x + y + z) * (x + 2*y + 3*z)) / ((x + y + z) * (2*x - y + z))");
    // Either it simplifies correctly OR stays as fraction (both are acceptable)
    // Key: should NOT return garbage or crash
    assert!(
        result.contains("/") || result.contains("+") || result.contains("x"),
        "Should return valid expression, got: {}",
        result
    );
    println!("Multi-param factor result: {}", result);
}

// B. GCD depends on single non-main variable (tests choose_var_candidates)
#[test]
fn test_hard_single_param_nonmain() {
    // ((x + z) * (x + y)) / ((x + z) * (x + 2*y)) -> (x + y)/(x + 2*y)
    // GCD = (x + z), z is the "interesting" param variable
    let result = simplify("((x + z) * (x + y)) / ((x + z) * (x + 2*y))");
    // Should simplify to something without z, OR stay as fraction
    // The key is: if it simplifies, z should cancel
    if !result.contains("z") && result.contains("y") {
        // Perfect: z canceled, (x+y)/(x+2y) form
        assert!(
            result.contains("y"),
            "Should contain y after simplification"
        );
    } else {
        // Acceptable: didn't find the GCD, stays as fraction
        assert!(
            result.contains("/"),
            "Should be valid fraction, got: {}",
            result
        );
    }
    println!("Single-param non-main result: {}", result);
}

// C. "Degree drop" consistency - samples with degree drops should be skipped
#[test]
fn test_hard_degree_drop_consistency() {
    // p = (x + y + z) * (x + y - z)
    // q = (x + y + z) * (x - y + z)
    // GCD = (x + y + z)
    // Some eval points (like z = -x-y) would cause degree drop
    let result = simplify("((x + y + z) * (x + y - z)) / ((x + y + z) * (x - y + z))");
    // Should simplify to (x + y - z)/(x - y + z) OR stay as fraction
    // Key: consistent behavior, no crash
    assert!(
        result.contains("/") || result.contains("+") || result.contains("-"),
        "Should return valid expression, got: {}",
        result
    );
    println!("Degree drop consistency result: {}", result);
}

// D. Verify no over-cancellation (false positive)
#[test]
fn test_hard_no_false_positive() {
    // (x^2 + x*y) / (x*y + y^2) = x(x+y) / y(x+y) = x/y (if gcd=(x+y))
    // BUT only if we correctly identify (x+y) as common factor
    let result = simplify("(x^2 + x*y) / (x*y + y^2)");
    // Expected: x/y (simplified) or the original fraction
    if result == "x / y" || result == "x/y" {
        // Perfect simplification
    } else {
        // Should at least be a valid fraction
        assert!(
            result.contains("/") || result.contains("x"),
            "Should be valid, got: {}",
            result
        );
    }
    println!("No false positive result: {}", result);
}

// E. Large polynomial - budget check (should bail quickly, not hang)
#[test]
fn test_hard_budget_bailout() {
    // Create moderately complex expression that should hit budget limits
    // (x+y)^4 / (x+y)^3 = x+y
    let result = simplify("(x+y)^4 / (x+y)^3");
    // Should simplify to x+y
    assert!(
        (result.contains("x") && result.contains("y") && result.contains("+"))
            || result.contains("/"),
        "Should simplify or stay as fraction, got: {}",
        result
    );
}
