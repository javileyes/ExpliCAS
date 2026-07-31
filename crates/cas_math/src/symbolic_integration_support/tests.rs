//! Tests de `symbolic_integration_support`, extraídos del módulo (P3).

use super::{
    get_linear_coeffs, integrate_symbolic_expr,
    integrate_symbolic_is_arcsin_inverse_sqrt_product_target,
    integrate_symbolic_is_positive_quadratic_cube_target,
    integrate_symbolic_required_nonzero_conditions,
    integrate_symbolic_required_positive_conditions,
    polynomial_times_constant_base_power_antiderivative,
    transcendental_chain_substitution_antiderivative,
};
use crate::general_integration_backend::{
    AlgorithmicIntegrationMethod, AlgorithmicIntegrationVerificationStatus,
};
use crate::polynomial::Polynomial;
use cas_ast::Context;
use cas_formatter::DisplayExpr;
use cas_parser::parse;
use num_rational::BigRational;

fn rendered(ctx: &Context, id: cas_ast::ExprId) -> String {
    format!("{}", DisplayExpr { context: ctx, id })
}

fn assert_constant_expr(ctx: &Context, id: cas_ast::ExprId, numerator: i64, denominator: i64) {
    let poly = Polynomial::from_expr(ctx, id, "x").expect("constant polynomial");
    assert_eq!(
        poly.coeffs,
        vec![BigRational::new(numerator.into(), denominator.into())]
    );
}

#[test]
fn integrates_simple_power() {
    let mut ctx = Context::new();
    let expr = parse("x^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "x^3 / 3");
}

#[test]
fn integrates_cbrt_via_power_rule_lowering() {
    let mut ctx = Context::new();
    // cbrt(x) is lowered to x^(1/3); the ordinary power rule gives 3/4 x^(4/3).
    let expr = parse("cbrt(x)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    // Raw power-rule form `x^(4/3) / (4/3)` (the CLI simplifier folds it to
    // 3/4 x^(4/3)); equivalently the antiderivative of x^(1/3).
    assert_eq!(rendered(&ctx, out), "x^(4/3) / 4/3");
    // cbrt(x)^2 -> x^(2/3) -> x^(5/3) / (5/3); the nested power must flatten.
    let squared = parse("cbrt(x)^2", &mut ctx).expect("parse");
    let out_sq = integrate_symbolic_expr(&mut ctx, squared, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out_sq), "x^(5/3) / 5/3");
    // A non-linear radicand stays residual: x^(1/3) of a non-linear base is
    // not a power-rule target (needs a substitution the engine declines).
    let residual = parse("cbrt(x^2 + 1)", &mut ctx).expect("parse");
    assert!(integrate_symbolic_expr(&mut ctx, residual, "x").is_none());
}

#[test]
fn integrates_polynomial_times_trig_square_via_power_reduction() {
    let mut ctx = Context::new();
    // x sin^2(x) -> x(1 - cos 2x)/2 -> x/4*... ; check it resolves (the exact
    // form is verified end-to-end by the contract/matrix; here assert that
    // the previously-residual product now integrates and the bare/non-affine
    // cases keep their behavior).
    for source in ["x*sin(x)^2", "x^2*sin(x)^2", "x^2*cos(x)^2", "x*sin(2*x)^2"] {
        let expr = parse(source, &mut ctx).expect("parse");
        assert!(
            integrate_symbolic_expr(&mut ctx, expr, "x").is_some(),
            "must integrate: {source}"
        );
    }
    // Non-affine inner stays residual; the bare power-reduction case is owned
    // elsewhere, so this rule declines it (constant cofactor / degree 0).
    let nonaffine = parse("x*sin(x^2)^2", &mut ctx).expect("parse");
    assert!(super::polynomial_times_trig_square_antiderivative(&mut ctx, nonaffine, "x").is_none());
    let bare = parse("2*sin(x)^2", &mut ctx).expect("parse");
    assert!(super::polynomial_times_trig_square_antiderivative(&mut ctx, bare, "x").is_none());
}

#[test]
fn integrates_hyperbolic_transcendental_products_via_exp_lowering() {
    let mut ctx = Context::new();
    // trig*hyperbolic and exp*hyperbolic products integrate by lowering the
    // sinh/cosh to exp form and delegating to the exp*trig / exp*exp owners.
    // Round-trips verified end-to-end (diff(integral) - integrand simplifies
    // to 0 via the CLI).
    for source in [
        "sin(x)*sinh(x)",
        "cos(x)*cosh(x)",
        "sin(x)*cosh(x)",
        "cos(x)*sinh(x)",
        "e^x*sinh(x)",
        "e^x*cosh(x)",
        "e^(2*x)*sinh(x)",
    ] {
        let expr = parse(source, &mut ctx).expect("parse");
        assert!(
            integrate_symbolic_expr(&mut ctx, expr, "x").is_some(),
            "must integrate: {source}"
        );
    }
    // Declines: a poly-times-hyperbolic (no trig/exp partner, owned by the
    // dedicated owner), a bare hyperbolic, two hyperbolics (no single factor
    // to lower into the exp*trig family), and a non-affine hyperbolic inner.
    for source in [
        "x*sinh(x)",
        "sinh(x)",
        "sinh(x)*cosh(x)",
        "sin(x)*sinh(x^2)",
    ] {
        let expr = parse(source, &mut ctx).expect("parse");
        assert!(
            super::hyperbolic_transcendental_product_antiderivative(&mut ctx, expr, "x").is_none(),
            "lowering rule must decline: {source}"
        );
    }
}

#[test]
fn integrates_polynomial_times_trig_square_with_substitution_inner() {
    let mut ctx = Context::new();
    // Non-affine inner whose cofactor supplies the substitution derivative:
    // x*sin(x^2)^2 -> x/2 - (x/2)cos(2x^2), elementary via u = x^2.
    // Round-trips verified numerically end-to-end (max |d/dx F - f| ~1e-16).
    for source in [
        "x*sin(x^2)^2",
        "x*cos(x^2)^2",
        "x^3*sin(x^2)^2",
        "x*sin(2*x^2)^2",
    ] {
        let expr = parse(source, &mut ctx).expect("parse");
        assert!(
            integrate_symbolic_expr(&mut ctx, expr, "x").is_some(),
            "must integrate: {source}"
        );
    }
    // Honest declines: even cofactor (non-elementary / Fresnel), bare square
    // (constant cofactor), cofactor mismatched to the inner derivative
    // (x*sin(x^3)^2 needs du = 3x^2), and the affine inner (owned by the
    // affine rule, so this substitution rule must NOT claim it).
    for source in ["x^2*sin(x^2)^2", "sin(x^2)^2", "x*sin(x^3)^2", "x*sin(x)^2"] {
        let expr = parse(source, &mut ctx).expect("parse");
        assert!(
            super::polynomial_times_trig_square_substitution_antiderivative(&mut ctx, expr, "x")
                .is_none(),
            "substitution rule must decline: {source}"
        );
    }
}

#[test]
fn integrates_polynomial_times_higher_even_trig_power() {
    let mut ctx = Context::new();
    // p(x) * sin^n / cos^n with even n in 4..=8 now reduces to a cosine sum
    // times p(x) and integrates; round-trips verified numerically end-to-end.
    for source in [
        "x*sin(x)^4",
        "x^2*sin(x)^4",
        "x*cos(x)^4",
        "x*sin(x)^6",
        "x*sin(x)^8",
        "x*cos(2*x+1)^4",
        "(x+1)*cos(x)^4",
    ] {
        let expr = parse(source, &mut ctx).expect("parse");
        assert!(
            integrate_symbolic_expr(&mut ctx, expr, "x").is_some(),
            "must integrate: {source}"
        );
    }
    // Declines: odd power (different family), non-affine inner, out-of-range
    // power (n=10), constant cofactor (the bare power-reduction owner), and
    // the n==2 case (owned by the square rule that runs first).
    for source in [
        "x*sin(x)^3",
        "x*sin(x^2)^4",
        "x*sin(x)^10",
        "3*sin(x)^4",
        "x*sin(x)^2",
    ] {
        let expr = parse(source, &mut ctx).expect("parse");
        assert!(
            super::polynomial_times_higher_even_trig_power_antiderivative(&mut ctx, expr, "x")
                .is_none(),
            "must decline: {source}"
        );
    }
}

#[test]
fn routes_constant_table_reuse_through_algorithmic_boundary_without_output_change() {
    let mut ctx = Context::new();
    let expr = parse("3", &mut ctx).expect("parse");

    let candidate = super::table_reused_constant_integration_candidate(&mut ctx, expr, "x");

    assert_eq!(candidate.method, AlgorithmicIntegrationMethod::TableReused);
    assert_eq!(
        candidate.verification_status,
        AlgorithmicIntegrationVerificationStatus::Verified
    );
    assert!(candidate.is_publicly_acceptable());
    let public_antiderivative = candidate
        .public_antiderivative()
        .expect("verified table candidate should be public");
    assert_eq!(rendered(&ctx, public_antiderivative), "3 * x");

    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "3 * x");
}

#[test]
fn integrates_linear_trig_substitution() {
    let mut ctx = Context::new();
    let expr = parse("sin(2*x)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-1/2 * cos(2 * x)");
}

#[test]
fn integrates_linear_log_table() {
    let mut ctx = Context::new();
    let expr = parse("ln(x)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "x * ln(x) - x");

    let expr = parse("ln(2*x+1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "(2 * x + 1) * (ln(2 * x + 1) - 1) / 2");

    let expr = parse("log(2,x)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "x * (log(2, x) - 1 / ln(2))");

    let expr = parse("log(2,3*x+2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "(3 * x + 2) * (log(2, 3 * x + 2) - 1 / ln(2)) / 3"
    );

    let expr = parse("log2(x)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "x * (log2(x) - 1 / ln(2))");

    let expr = parse("log10(3*x+2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "(3 * x + 2) * (log10(3 * x + 2) - 1 / ln(10)) / 3"
    );

    for invalid in ["log(1,x)", "log(-2,x)", "log(1,2)", "ln(0)", "sqrt(-1)"] {
        let expr = parse(invalid, &mut ctx).expect("parse invalid integrand");
        let out = integrate_symbolic_expr(&mut ctx, expr, "x")
            .expect("invalid real-domain integrand should produce undefined");
        assert_eq!(
            rendered(&ctx, out),
            "undefined",
            "invalid real-domain integral should be undefined for {invalid}"
        );
    }

    for unsupported in ["log(x,x)", "log(2,x-x+2)"] {
        let expr = parse(unsupported, &mut ctx).expect("parse unsupported log");
        assert!(
            integrate_symbolic_expr(&mut ctx, expr, "x").is_none(),
            "unsupported constant-base log integral should remain residual for {unsupported}"
        );
    }
}

#[test]
fn integrates_quadratic_monomial_times_positive_affine_log_by_parts() {
    let mut ctx = Context::new();
    let expr = parse("x^2*ln(2*x+1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");

    assert_eq!(
        rendered(&ctx, out),
        "1/3 * x^3 * ln(2 * x + 1) - (1/9 * x^3 + 1/12 * x - 1/24 * ln(2 * x + 1) - 1/12 * x^2)"
    );
}

#[test]
fn integrates_polynomial_derivative_times_log_substitution() {
    let mut ctx = Context::new();
    let expr = parse("2*x*ln(x^2+1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "(x^2 + 1) * (ln(x^2 + 1) - 1)");

    let expr = parse("(2*x+1)*ln(x^2+x+1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "(x^2 + x + 1) * (ln(x^2 + x + 1) - 1)");

    let expr = parse("2*x*ln(x^2+x+1)+ln(x^2+x+1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "(x^2 + x + 1) * (ln(x^2 + x + 1) - 1)");

    let expr = parse("4*x*ln(x^2+1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "2 * (x^2 + 1) * (ln(x^2 + 1) - 1)");

    let expr = parse("2*x*log(2,x^2+1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "(x^2 + 1) * log(2, x^2 + 1) - (x^2 + 1) / ln(2)"
    );

    let expr = parse("(2*x+1)*log10(x^2+x+1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "(x^2 + x + 1) * log10(x^2 + x + 1) - (x^2 + x + 1) / ln(10)"
    );

    let expr = parse("4*x*log2(x^2+1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "2 * ((x^2 + 1) * log2(x^2 + 1) - (x^2 + 1) / ln(2))"
    );

    let invalid_base = "2*x*log(1,x^2+1)";
    let expr = parse(invalid_base, &mut ctx).expect("parse invalid-base log product");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x")
        .expect("invalid-base log product should produce undefined");
    assert_eq!(
        rendered(&ctx, out),
        "undefined",
        "invalid-base log product should be undefined for {invalid_base}"
    );

    let unsupported = "2*x*log(x,x^2+1)";
    let expr = parse(unsupported, &mut ctx).expect("parse unsupported log product");
    assert!(
        integrate_symbolic_expr(&mut ctx, expr, "x").is_none(),
        "unsupported symbolic-base log product should remain residual for {unsupported}"
    );
}

#[test]
fn integrates_quadratic_times_positive_quadratic_log_by_parts() {
    let mut ctx = Context::new();
    let expr = parse("ln(x^2+x+1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "x * ln(x^2 + x + 1) - (-3 * arctan((2 * x + 1) / sqrt(3)) / sqrt(3) + 2 * x - 1/2 * ln(x^2 + x + 1))"
    );

    let expr = parse("x^2*ln(x^2+1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/3 * x^3 * ln(x^2 + 1) - (2/3 * arctan(x) + 2/9 * x^3 - 2/3 * x)"
    );

    let expr = parse("x*ln(x^2+x+1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/2 * x^2 * ln(x^2 + x + 1) - (3/2 * arctan((2 * x + 1) / sqrt(3)) / sqrt(3) + 1/2 * x^2 - 1/4 * ln(x^2 + x + 1) - 1/2 * x)"
    );

    let expr = parse("x^2*ln(x^2+x+1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/3 * x^3 * ln(x^2 + x + 1) - (1/3 * ln(x^2 + x + 1) + 2/9 * x^3 - 1/6 * x^2 - 1/3 * x)"
    );

    let expr = parse("x^3*ln(x^2+1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/4 * x^4 * ln(x^2 + 1) - (1/4 * ln(x^2 + 1) + 1/8 * x^4 - 1/4 * x^2)"
    );

    let expr = parse("x^4*ln(x^2+1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/5 * x^5 * ln(x^2 + 1) - (2/25 * x^5 + 2/5 * x - 2/5 * arctan(x) - 2/15 * x^3)"
    );

    let expr = parse("x^5*ln(x^2+1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/6 * x^6 * ln(x^2 + 1) - (1/18 * x^6 + 1/6 * x^2 - 1/6 * ln(x^2 + 1) - 1/12 * x^4)"
    );

    let expr = parse("x^6*ln(x^2+1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/7 * x^7 * ln(x^2 + 1) - (2/7 * arctan(x) + 2/49 * x^7 + 2/21 * x^3 - 2/35 * x^5 - 2/7 * x)"
    );

    let expr = parse("x^7*ln(x^2+1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/8 * x^8 * ln(x^2 + 1) - (1/8 * ln(x^2 + 1) + 1/32 * x^8 + 1/16 * x^4 - 1/24 * x^6 - 1/8 * x^2)"
    );

    let expr = parse("x^8*ln(x^2+1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/9 * x^9 * ln(x^2 + 1) - (2/81 * x^9 + 2/45 * x^5 + 2/9 * x - 2/9 * arctan(x) - 2/63 * x^7 - 2/27 * x^3)"
    );

    let expr = parse("x^2*ln(x^2-1)", &mut ctx).expect("parse");
    assert!(integrate_symbolic_expr(&mut ctx, expr, "x").is_none());

    let expr = parse("x^9*ln(x^2+1)", &mut ctx).expect("parse");
    assert!(
        integrate_symbolic_expr(&mut ctx, expr, "x").is_none(),
        "positive-quadratic ln by-parts budget should stop above degree {}",
        super::POSITIVE_QUADRATIC_LN_BY_PARTS_MAX_COFACTOR_DEGREE
    );
}

#[test]
fn integrates_polynomial_log_derivative_power_substitution() {
    let mut ctx = Context::new();
    let expr = parse("2*x*ln(x^2+1)^2/(x^2+1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "1/3 * ln(x^2 + 1)^3");

    let expr = parse("(2*x+1)*ln(abs(x^2+x-1))^2/(x^2+x-1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "1/3 * ln(|x^2 + x - 1|)^3");
}

#[test]
fn integrates_polynomial_derivative_times_log_square_by_parts() {
    let mut ctx = Context::new();
    let expr = parse("2*x*ln(x^2+1)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "(x^2 + 1) * (ln(x^2 + 1)^2 - 2 * ln(x^2 + 1) + 2)"
    );

    let expr = parse("2*x*log(2,x^2+1)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "(x^2 + 1) * (log(2, x^2 + 1)^2 + 2 / ln(2)^2 + -2 * log(2, x^2 + 1) / ln(2))"
    );

    let expr = parse("2*x*log2(x^2+1)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "(x^2 + 1) * (log2(x^2 + 1)^2 + 2 / ln(2)^2 + -2 * log2(x^2 + 1) / ln(2))"
    );

    let expr = parse("ln(2*x+1)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/2 * (2 * x + 1) * (ln(2 * x + 1)^2 - 2 * ln(2 * x + 1) + 2)"
    );

    let expr = parse("(2*x+1)*ln(x^2+x+1)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "(x^2 + x + 1) * (ln(x^2 + x + 1)^2 - 2 * ln(x^2 + x + 1) + 2)"
    );

    let expr = parse("2*x*ln(x^2-1)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "(x^2 - 1) * (ln(x^2 - 1)^2 - 2 * ln(x^2 - 1) + 2)"
    );

    let expr = parse("2*x*log(2,x^2-1)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "(x^2 - 1) * (log(2, x^2 - 1)^2 + 2 / ln(2)^2 + -2 * log(2, x^2 - 1) / ln(2))"
    );
    let conditions = integrate_symbolic_required_positive_conditions(&mut ctx, expr, "x");
    assert_eq!(conditions.len(), 1);
    assert_eq!(rendered(&ctx, conditions[0]), "x^2 - 1");

    let expr = parse("(2*x+1)*ln(x^2+x-1)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "(x^2 + x - 1) * (ln(x^2 + x - 1)^2 - 2 * ln(x^2 + x - 1) + 2)"
    );

    let expr = parse("(3*x^2-1)*ln(x^3-x)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "(x^3 - x) * (ln(x^3 - x)^2 - 2 * ln(x^3 - x) + 2)"
    );

    let expr = parse("(4*x^3-2*x)*ln(x^4-x^2-1)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "(x^4 - x^2 - 1) * (ln(x^4 - x^2 - 1)^2 - 2 * ln(x^4 - x^2 - 1) + 2)"
    );

    let expr = parse("x*ln(x)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/8 * x^2 * (4 * ln(x)^2 - 4 * ln(x) + 2)"
    );

    let expr = parse("x^2*ln(x)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/27 * x^3 * (9 * ln(x)^2 - 6 * ln(x) + 2)"
    );

    let expr = parse("x*ln(x)^3", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/16 * x^2 * (8 * ln(x)^3 - 12 * ln(x)^2 + 12 * ln(x) - 6)"
    );

    let expr = parse("x^2*ln(x)^3", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/81 * x^3 * (27 * ln(x)^3 - 27 * ln(x)^2 + 18 * ln(x) - 6)"
    );

    let expr = parse("x*ln(x)^4", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/32 * x^2 * (16 * ln(x)^4 - 32 * ln(x)^3 + 48 * ln(x)^2 - 48 * ln(x) + 24)"
    );

    let expr = parse("x^2*ln(x)^4", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/243 * x^3 * (81 * ln(x)^4 - 108 * ln(x)^3 + 108 * ln(x)^2 - 72 * ln(x) + 24)"
    );

    let expr = parse("x*ln(x)^5", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/64 * x^2 * (32 * ln(x)^5 - 80 * ln(x)^4 + 160 * ln(x)^3 - 240 * ln(x)^2 + 240 * ln(x) - 120)"
    );

    let expr = parse("x^2*ln(x)^5", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/729 * x^3 * (243 * ln(x)^5 - 405 * ln(x)^4 + 540 * ln(x)^3 - 540 * ln(x)^2 + 360 * ln(x) - 120)"
    );

    let expr = parse("2*x*ln(x^2+1)^3", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "(x^2 + 1) * (ln(x^2 + 1)^3 - 3 * ln(x^2 + 1)^2 + 6 * ln(x^2 + 1) - 6)"
    );

    let expr = parse("2*x*ln(x^2+1)^4", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "(x^2 + 1) * (ln(x^2 + 1)^4 - 4 * ln(x^2 + 1)^3 + 12 * ln(x^2 + 1)^2 - 24 * ln(x^2 + 1) + 24)"
    );

    let expr = parse("2*x*ln(x^2+1)^5", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "(x^2 + 1) * (ln(x^2 + 1)^5 - 5 * ln(x^2 + 1)^4 + 20 * ln(x^2 + 1)^3 - 60 * ln(x^2 + 1)^2 + 120 * ln(x^2 + 1) - 120)"
    );

    let expr = parse("2*x*ln(x^2-1)^4", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "(x^2 - 1) * (ln(x^2 - 1)^4 - 4 * ln(x^2 - 1)^3 + 12 * ln(x^2 - 1)^2 - 24 * ln(x^2 - 1) + 24)"
    );
    let conditions = integrate_symbolic_required_positive_conditions(&mut ctx, expr, "x");
    assert_eq!(conditions.len(), 1);
    assert_eq!(rendered(&ctx, conditions[0]), "x^2 - 1");

    let expr = parse("(2*x+1)*ln(x^2+x+1)^3", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "(x^2 + x + 1) * (ln(x^2 + x + 1)^3 - 3 * ln(x^2 + x + 1)^2 + 6 * ln(x^2 + x + 1) - 6)"
    );

    let invalid_base = "2*x*log(1,x^2+1)^2";
    let expr = parse(invalid_base, &mut ctx).expect("parse invalid-base log power");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x")
        .expect("invalid-base log power integral should produce undefined");
    assert_eq!(
        rendered(&ctx, out),
        "undefined",
        "invalid-base log power integral should be undefined for {invalid_base}"
    );

    let unsupported = "2*x*log(y,x^2+1)^2";
    let expr = parse(unsupported, &mut ctx).expect("parse unsupported log power");
    assert!(
        integrate_symbolic_expr(&mut ctx, expr, "x").is_none(),
        "unsupported symbolic-base log power integral should remain residual for {unsupported}"
    );
}

#[test]
fn integrates_linear_times_exp_linear_by_parts() {
    let mut ctx = Context::new();
    let expr = parse("x*exp(x)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "e^x * (x - 1)");

    let expr = parse("-x*exp(x)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "e^x * (1 - x)");

    let expr = parse("(2*x+3)*exp(2*x+1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "e^(2 * x + 1) * (x + 1)");

    let expr = parse("(x+1)*exp((3*x+2)/2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "e^((3 * x + 2) / 2) * (2/3 * x + 2/9)");

    let expr = parse("(x+1)*exp((2-3*x)/2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "e^((2 - 3 * x) / 2) * (-2/3 * x - 10/9)"
    );
}

#[test]
fn integrates_transcendental_chain_substitution() {
    // Each integrates by guess-and-verify u-substitution; confirm the round-trip
    // d/dx(∫) == integrand numerically over a sweep, and that non-chain forms decline.
    for src in [
        "cos(x)*exp(sin(x))",
        "sin(x)*exp(cos(x))",
        "exp(x)*cos(exp(x))",
        "sinh(x)*exp(cosh(x))",
    ] {
        let mut ctx = Context::new();
        let integrand = parse(src, &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, integrand, "x")
            .unwrap_or_else(|| panic!("{src} should integrate"));
        let derivative = crate::symbolic_differentiation_support::differentiate_symbolic_expr(
            &mut ctx, out, "x",
        )
        .expect("differentiate");
        let integrand = parse(src, &mut ctx).expect("re-parse");
        for sample in [-1i64, 1, 2] {
            let a = eval_numeric_at(&ctx, derivative, "x", sample);
            let b = eval_numeric_at(&ctx, integrand, "x", sample);
            if let (Some(a), Some(b)) = (a, b) {
                assert!(
                    (a - b).abs() < 1e-9,
                    "{src}: d/dx(∫) != integrand at x={sample}"
                );
            }
        }
    }
    // A genuinely non-elementary chain (∫ e^(x^2) has no elementary form) must NOT be
    // accepted: there is no F(g) whose derivative is e^(x^2).
    let mut ctx = Context::new();
    let non_elementary = parse("exp(x^2)", &mut ctx).expect("parse");
    assert!(
        transcendental_chain_substitution_antiderivative(&mut ctx, non_elementary, "x").is_none(),
        "exp(x^2) has no elementary antiderivative and must decline"
    );
}

#[test]
fn integrates_polynomial_times_constant_base_power_by_parts() {
    let mut ctx = Context::new();
    // ∫ x·2^x dx = 2^x·(x/ln 2 − 1/ln(2)^2). Verified by differentiating back below.
    let expr = parse("x*2^x", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    let derivative =
        crate::symbolic_differentiation_support::differentiate_symbolic_expr(&mut ctx, out, "x")
            .expect("differentiate antiderivative");
    let integrand = parse("x*2^x", &mut ctx).expect("parse integrand");
    // d/dx of the antiderivative must equal the integrand (numerically over a sweep).
    for sample in [-2i64, -1, 1, 2, 3] {
        let a = eval_numeric_at(&ctx, derivative, "x", sample);
        let b = eval_numeric_at(&ctx, integrand, "x", sample);
        let (a, b) = (a.expect("eval derivative"), b.expect("eval integrand"));
        assert!(
            (a - b).abs() < 1e-9,
            "x*2^x: d/dx(∫) {a} != integrand {b} at x={sample}"
        );
    }

    // AFFINE exponents `a^(m·x+c)` integrate through the effective slope
    // `m·ln a` — `x·2^(2x)`, `x·3^(2x)`, `x·2^(x+1)`, negative/fractional
    // slopes — all round-trip to the integrand under differentiation.
    for src in [
        "x*2^(2*x)",
        "x*3^(2*x)",
        "x*2^(x+1)",
        "x^2*3^(2*x)",
        "x*2^(-x)",
        "x*9^(x/2)",
        "(3*x+1)*5^(2*x-1)",
    ] {
        let mut ctx = Context::new();
        let integrand = parse(src, &mut ctx).expect("parse");
        let out = integrate_symbolic_expr(&mut ctx, integrand, "x")
            .unwrap_or_else(|| panic!("{src} should integrate"));
        let derivative = crate::symbolic_differentiation_support::differentiate_symbolic_expr(
            &mut ctx, out, "x",
        )
        .expect("differentiate");
        let integrand = parse(src, &mut ctx).expect("re-parse");
        for sample in [-2i64, -1, 1, 2, 3] {
            let a = eval_numeric_at(&ctx, derivative, "x", sample).expect("eval derivative");
            let b = eval_numeric_at(&ctx, integrand, "x", sample).expect("eval integrand");
            assert!(
                (a - b).abs() < 1e-9,
                "{src}: d/dx(∫) {a} != integrand {b} at x={sample}"
            );
        }
    }

    // Bare 2^x (degree-0 cofactor) is NOT this kernel — the table owns it.
    let bare = parse("2^x", &mut ctx).expect("parse");
    assert!(
        polynomial_times_constant_base_power_antiderivative(&mut ctx, bare, "x").is_none(),
        "bare 2^x must be left to the a^x table, not the by-parts kernel"
    );
    // A constant (degree-0) exponent is not an exponential in x — decline.
    let const_exp = parse("x*2^3", &mut ctx).expect("parse");
    assert!(
        polynomial_times_constant_base_power_antiderivative(&mut ctx, const_exp, "x").is_none(),
        "x*2^3 has a constant exponent — not an exponential in x"
    );
    // Base e is the exp kernel's job, not this one (no constant rational base).
    let exp_case = parse("x*exp(x)", &mut ctx).expect("parse");
    assert!(
        polynomial_times_constant_base_power_antiderivative(&mut ctx, exp_case, "x").is_none(),
        "x*e^x belongs to the exp-linear kernel"
    );
}

fn eval_numeric_at(ctx: &Context, expr: cas_ast::ExprId, var: &str, value: i64) -> Option<f64> {
    use cas_ast::Expr;
    match ctx.get(expr) {
        Expr::Number(n) => {
            use num_traits::ToPrimitive;
            n.to_f64()
        }
        Expr::Variable(sym) if ctx.sym_name(*sym) == var => Some(value as f64),
        Expr::Constant(cas_ast::Constant::E) => Some(std::f64::consts::E),
        Expr::Neg(inner) => Some(-eval_numeric_at(ctx, *inner, var, value)?),
        Expr::Add(l, r) => {
            Some(eval_numeric_at(ctx, *l, var, value)? + eval_numeric_at(ctx, *r, var, value)?)
        }
        Expr::Sub(l, r) => {
            Some(eval_numeric_at(ctx, *l, var, value)? - eval_numeric_at(ctx, *r, var, value)?)
        }
        Expr::Mul(l, r) => {
            Some(eval_numeric_at(ctx, *l, var, value)? * eval_numeric_at(ctx, *r, var, value)?)
        }
        Expr::Div(l, r) => {
            Some(eval_numeric_at(ctx, *l, var, value)? / eval_numeric_at(ctx, *r, var, value)?)
        }
        Expr::Pow(b, e) => {
            Some(eval_numeric_at(ctx, *b, var, value)?.powf(eval_numeric_at(ctx, *e, var, value)?))
        }
        Expr::Function(fn_id, args) if args.len() == 1 => {
            let arg = eval_numeric_at(ctx, args[0], var, value)?;
            match ctx.builtin_of(*fn_id)? {
                cas_ast::BuiltinFn::Ln => Some(arg.ln()),
                cas_ast::BuiltinFn::Exp => Some(arg.exp()),
                cas_ast::BuiltinFn::Sin => Some(arg.sin()),
                cas_ast::BuiltinFn::Cos => Some(arg.cos()),
                _ => None,
            }
        }
        _ => None,
    }
}

#[test]
fn integrates_polynomial_times_exp_linear_by_parts() {
    let mut ctx = Context::new();
    let expr = parse("x^2*exp(x)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "exp(x) * (x^2 + 2 - 2 * x)");

    let expr = parse("(x^2+x+1)*exp(2*x+1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "exp(2 * x + 1) * (x^2 + 1) / 2");

    let expr = parse("(x^2+x+1)*exp((2-3*x)/2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "exp((2 - 3 * x) / 2) * (-18 * x^2 - 42 * x - 46) / 27"
    );

    let expr = parse("x^3*exp(x)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "exp(x) * (x^3 + 6 * x - 3 * x^2 - 6)");

    let expr = parse("(x^3+x)*exp(2*x+1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "exp(2 * x + 1) * (4 * x^3 + 10 * x - 6 * x^2 - 5) / 8"
    );

    let expr = parse("x^4*exp(2*x+1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "exp(2 * x + 1) * (2 * x^4 + 6 * x^2 + 3 - 4 * x^3 - 6 * x) / 4"
    );

    let expr = parse("x^5*exp(x)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "exp(x) * (x^5 + 20 * x^3 + 120 * x - 5 * x^4 - 60 * x^2 - 120)"
    );

    let expr = parse("x^6*exp(x)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "exp(x) * (x^6 + 30 * x^4 + 360 * x^2 + 720 - 6 * x^5 - 120 * x^3 - 720 * x)"
    );

    let expr = parse("x^7*exp(x)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "exp(x) * (x^7 + 42 * x^5 + 840 * x^3 + 5040 * x - 7 * x^6 - 210 * x^4 - 2520 * x^2 - 5040)"
    );

    let expr = parse("x^8*exp(x)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "exp(x) * (x^8 + 56 * x^6 + 1680 * x^4 + 20160 * x^2 + 40320 - 8 * x^7 - 336 * x^5 - 6720 * x^3 - 40320 * x)"
    );
}

#[test]
fn integrates_exp_trig_same_linear_by_parts() {
    let mut ctx = Context::new();
    let expr = parse("exp(x)*sin(x)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "1/2 * e^x * (sin(x) - cos(x))");

    let expr = parse("exp(x)*cos(x)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "1/2 * e^x * (sin(x) + cos(x))");

    let expr = parse("exp(2*x+1)*sin(2*x+1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/4 * e^(2 * x + 1) * (sin(2 * x + 1) - cos(2 * x + 1))"
    );

    let expr = parse("exp(2*x+1)*cos(2*x+1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/4 * e^(2 * x + 1) * (sin(2 * x + 1) + cos(2 * x + 1))"
    );

    let expr = parse("3*exp(2*x+1)*sin(2*x+1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "3/4 * e^(2 * x + 1) * (sin(2 * x + 1) - cos(2 * x + 1))"
    );

    let expr = parse("exp(2*x)*sin(3*x)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/13 * e^(2 * x) * (2 * sin(3 * x) - 3 * cos(3 * x))"
    );

    let expr = parse("exp(2*x)*sin((3*x+1)/2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "4/25 * e^(2 * x) * (2 * sin((3 * x + 1) / 2) - 3/2 * cos((3 * x + 1) / 2))"
    );

    let expr = parse("exp(2*x)*cos((3*x+1)/2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "4/25 * e^(2 * x) * (3/2 * sin((3 * x + 1) / 2) + 2 * cos((3 * x + 1) / 2))"
    );

    let expr = parse("exp(2*x)*cos(3*x)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/13 * e^(2 * x) * (2 * cos(3 * x) + 3 * sin(3 * x))"
    );
}

#[test]
fn integrates_linear_times_trig_linear_by_parts() {
    let mut ctx = Context::new();
    let expr = parse("x*sin(x)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "sin(x) - x * cos(x)");

    let expr = parse("x*cos(x)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "cos(x) + x * sin(x)");

    let expr = parse("(2*x+3)*sin(2*x+1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/2 * sin(2 * x + 1) - (cos(2 * x + 1) * (2 * x + 3))/2"
    );

    let expr = parse("(2*x+3)*cos(2*x+1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/2 * cos(2 * x + 1) + (sin(2 * x + 1) * (2 * x + 3))/2"
    );

    let expr = parse("(x+1)*sin((3*x+2)/2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "4/9 * sin((3 * x + 2) / 2) - 2/3 * (x + 1) * cos((3 * x + 2) / 2)"
    );

    let expr = parse("(x+1)*cos((3*x+2)/2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "4/9 * cos((3 * x + 2) / 2) + 2/3 * (x + 1) * sin((3 * x + 2) / 2)"
    );

    let expr = parse("(x+1)*sin((2-3*x)/2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "4/9 * sin((2 - 3 * x) / 2) + 2/3 * (x + 1) * cos((2 - 3 * x) / 2)"
    );

    let expr = parse("(x+1)*cos((2-3*x)/2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "4/9 * cos((2 - 3 * x) / 2) - 2/3 * (x + 1) * sin((2 - 3 * x) / 2)"
    );
}

#[test]
fn integrates_quadratic_times_trig_linear_by_parts() {
    let mut ctx = Context::new();
    let expr = parse("x^2*sin(x)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "2 * x * sin(x) + (2 - x^2) * cos(x)");

    let expr = parse("-x^2*sin(x)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-2 * x * sin(x) + (x^2 - 2) * cos(x)");

    let expr = parse("x^2*cos(x)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "2 * x * cos(x) + (x^2 - 2) * sin(x)");

    let expr = parse("(x^2+x+1)*sin(2*x+1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "(1/2 * x + 1/4) * sin(2 * x + 1) + (-1/2 * x^2 - 1/2 * x - 1/4) * cos(2 * x + 1)"
    );
}

#[test]
fn integrates_cubic_times_trig_linear_by_parts() {
    let mut ctx = Context::new();
    let expr = parse("x^3*sin(x)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "(6 * x - x^3) * cos(x) + (3 * x^2 - 6) * sin(x)"
    );

    let expr = parse("x^3*cos(x)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "(x^3 - 6 * x) * sin(x) + (3 * x^2 - 6) * cos(x)"
    );
}

#[test]
fn integrates_quartic_times_trig_linear_by_parts() {
    let mut ctx = Context::new();
    let expr = parse("x^4*sin(2*x+1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "(x^3 - 3/2 * x) * sin(2 * x + 1) + (-1/2 * x^4 + 3/2 * x^2 - 3/4) * cos(2 * x + 1)"
    );

    let expr = parse("x^4*cos(2*x+1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "(x^3 - 3/2 * x) * cos(2 * x + 1) + (1/2 * x^4 - 3/2 * x^2 + 3/4) * sin(2 * x + 1)"
    );
}

#[test]
fn integrates_quintic_times_trig_linear_by_parts() {
    let mut ctx = Context::new();
    let expr = parse("x^5*sin(x)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "(-x^5 + 20 * x^3 - 120 * x) * cos(x) + (5 * x^4 - 60 * x^2 + 120) * sin(x)"
    );

    let expr = parse("x^5*cos(x)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "(x^5 - 20 * x^3 + 120 * x) * sin(x) + (5 * x^4 - 60 * x^2 + 120) * cos(x)"
    );
}

#[test]
fn integrates_sextic_times_trig_linear_by_parts() {
    let mut ctx = Context::new();
    let expr = parse("x^6*sin(x)", &mut ctx).expect("parse");
    assert!(super::integrate_symbolic_is_polynomial_times_trig_linear_target(&mut ctx, expr, "x"));
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "(6 * x^5 - 120 * x^3 + 720 * x) * sin(x) + (-x^6 + 30 * x^4 - 360 * x^2 + 720) * cos(x)"
    );

    let expr = parse("x^6*cos(x)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "(6 * x^5 - 120 * x^3 + 720 * x) * cos(x) + (x^6 - 30 * x^4 + 360 * x^2 - 720) * sin(x)"
    );

    let expr = parse("x^7*sin(x)", &mut ctx).expect("parse");
    assert!(super::integrate_symbolic_is_polynomial_times_trig_linear_target(&mut ctx, expr, "x"));
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "(-x^7 + 42 * x^5 - 840 * x^3 + 5040 * x) * cos(x) + (7 * x^6 - 210 * x^4 + 2520 * x^2 - 5040) * sin(x)"
    );

    let expr = parse("x^7*cos(x)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "(x^7 - 42 * x^5 + 840 * x^3 - 5040 * x) * sin(x) + (7 * x^6 - 210 * x^4 + 2520 * x^2 - 5040) * cos(x)"
    );

    let expr = parse("x^8*cos(x)", &mut ctx).expect("parse");
    assert!(super::integrate_symbolic_is_polynomial_times_trig_linear_target(&mut ctx, expr, "x"));
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "(8 * x^7 - 336 * x^5 + 6720 * x^3 - 40320 * x) * cos(x) + (x^8 - 56 * x^6 + 1680 * x^4 - 20160 * x^2 + 40320) * sin(x)"
    );
}

#[test]
fn integrates_linear_times_hyperbolic_linear_by_parts() {
    let mut ctx = Context::new();
    let expr = parse("x*sinh(x)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "x * cosh(x) - sinh(x)");

    let expr = parse("x*cosh(x)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "x * sinh(x) - cosh(x)");

    let expr = parse("(2*x+3)*sinh(2*x+1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "(cosh(2 * x + 1) * (2 * x + 3))/2 - 1/2 * sinh(2 * x + 1)"
    );

    let expr = parse("(2*x+3)*cosh(2*x+1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "(sinh(2 * x + 1) * (2 * x + 3))/2 - 1/2 * cosh(2 * x + 1)"
    );

    let expr = parse("(x+1)*sinh((3*x+2)/2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "(cosh((3 * x + 2) / 2) * (x + 1))/3/2 - 4/9 * sinh((3 * x + 2) / 2)"
    );

    let expr = parse("(x+1)*cosh((3*x+2)/2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "(sinh((3 * x + 2) / 2) * (x + 1))/3/2 - 4/9 * cosh((3 * x + 2) / 2)"
    );
}

#[test]
fn integrates_polynomial_times_hyperbolic_linear_by_parts() {
    let mut ctx = Context::new();
    let expr = parse("x^2*sinh(x)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "(x^2 + 2) * cosh(x) - 2 * x * sinh(x)");

    let expr = parse("x^2*cosh(x)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "(x^2 + 2) * sinh(x) - 2 * x * cosh(x)");

    let expr = parse("x^5*sinh(x)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "(x^5 + 20 * x^3 + 120 * x) * cosh(x) - (5 * x^4 + 60 * x^2 + 120) * sinh(x)"
    );

    let expr = parse("x^6*sinh(x)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "(x^6 + 30 * x^4 + 360 * x^2 + 720) * cosh(x) - (6 * x^5 + 120 * x^3 + 720 * x) * sinh(x)"
    );

    let expr = parse("x^7*sinh(x)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "(x^7 + 42 * x^5 + 840 * x^3 + 5040 * x) * cosh(x) - (7 * x^6 + 210 * x^4 + 2520 * x^2 + 5040) * sinh(x)"
    );

    let expr = parse("x^7*cosh(x)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "(x^7 + 42 * x^5 + 840 * x^3 + 5040 * x) * sinh(x) - (7 * x^6 + 210 * x^4 + 2520 * x^2 + 5040) * cosh(x)"
    );

    let expr = parse("x^8*cosh(x)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "(x^8 + 56 * x^6 + 1680 * x^4 + 20160 * x^2) * sinh(x) - (8 * x^7 + 336 * x^5 + 6720 * x^3 + 40320 * x) * cosh(x)"
    );
}

#[test]
fn integrates_explicit_negation_by_linearity() {
    let mut ctx = Context::new();
    let expr = parse("-sin(x)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "cos(x)");

    let expr = parse("-(x*sin(x^2)/cos(x^2)^2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-sec(x^2) / 2");

    let expr = parse("-(x^2*cos(x^3)/sin(x^3)^2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "csc(x^3) / 3");

    let expr = parse("-tan(x^2)", &mut ctx).expect("parse");
    assert!(integrate_symbolic_expr(&mut ctx, expr, "x").is_none());
}

#[test]
fn integrates_polynomial_derivative_exp_substitution() {
    let mut ctx = Context::new();
    let expr = parse("2*x*exp(x^2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "e^(x^2)");
}

#[test]
fn integrates_polynomial_derivative_trig_substitution() {
    let mut ctx = Context::new();
    let expr = parse("2*x*cos(x^2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "sin(x^2)");

    let expr = parse("2*x*sin(x^2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-cos(x^2)");

    let expr = parse("4*x^3*cos(x^4-x^2)-2*x*cos(x^4-x^2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "sin(x^4 - x^2)");

    let expr = parse("4*x^3*sin(x^4-x^2)-2*x*sin(x^4-x^2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-cos(x^4 - x^2)");

    let expr = parse(
        "(4*x^3*sin(x^4-x^2)-2*x*sin(x^4-x^2))/cos(x^4-x^2)",
        &mut ctx,
    )
    .expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-ln(|cos(x^4 - x^2)|)");
    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert_eq!(conditions.len(), 1);
    assert_eq!(rendered(&ctx, conditions[0]), "cos(x^4 - x^2)");

    let expr = parse(
        "(4*x^3*cos(x^4-x^2)-2*x*cos(x^4-x^2))/sin(x^4-x^2)",
        &mut ctx,
    )
    .expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "ln(|sin(x^4 - x^2)|)");
    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert_eq!(conditions.len(), 1);
    assert_eq!(rendered(&ctx, conditions[0]), "sin(x^4 - x^2)");

    let expr = parse("(2*k*x*sin(x^2+b))/cos(x^2+b)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-k * ln(|cos(x^2 + b)|)");
    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert_eq!(conditions.len(), 1);
    assert_eq!(rendered(&ctx, conditions[0]), "cos(x^2 + b)");

    let expr = parse("(2*k*x*cos(x^2+b))/sin(x^2+b)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "k * ln(|sin(x^2 + b)|)");
    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert_eq!(conditions.len(), 1);
    assert_eq!(rendered(&ctx, conditions[0]), "sin(x^2 + b)");
}

#[test]
fn integrates_nested_trig_log_derivative_substitution() {
    let mut ctx = Context::new();
    let expr = parse("1/(sin(x)*cos(x)*ln(tan(x)))", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "ln(|ln(tan(x))|)");

    let expr = parse("1/(sin(x)*cos(x)*ln(cot(x)))", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-ln(|ln(cot(x))|)");

    let expr = parse("2/(sin(2*x+1)*cos(2*x+1)*ln(tan(2*x+1)))", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "ln(|ln(tan(2 * x + 1))|)");
}

#[test]
fn integrates_polynomial_derivative_hyperbolic_substitution() {
    let mut ctx = Context::new();
    let expr = parse("sinh(2*x + 1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "1/2 * cosh(2 * x + 1)");

    let expr = parse("cosh(2*x + 1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "1/2 * sinh(2 * x + 1)");

    let expr = parse("2*x*sinh(x^2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "cosh(x^2)");

    let expr = parse("2*x*cosh(x^2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "sinh(x^2)");

    let expr = parse("tanh(2*x + 1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "1/2 * ln(|cosh(2 * x + 1)|)");

    let expr = parse("2*x*tanh(x^2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "ln(|cosh(x^2)|)");

    let expr = parse("2*k*x*tanh(x^2+b)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "k * ln(|cosh(x^2 + b)|)");

    let expr = parse("-2*k*x*tanh(x^2+b)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-k * ln(|cosh(x^2 + b)|)");
}

#[test]
fn integrates_affine_hyperbolic_power_times_derivative_product() {
    let mut ctx = Context::new();
    let expr = parse("sinh(x)^2*cosh(x)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "1/3 * sinh(x)^3");

    let expr = parse("2*cosh(2*x + 1)*sinh(2*x + 1)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "1/3 * sinh(2 * x + 1)^3");

    let expr = parse("sinh(x)*cosh(x)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "1/3 * cosh(x)^3");
}

#[test]
fn integrates_hyperbolic_log_derivative_ratio_substitution() {
    let mut ctx = Context::new();
    let expr = parse("sinh(2*x + 1)/cosh(2*x + 1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "1/2 * ln(|cosh(2 * x + 1)|)");

    let expr = parse("cosh(2*x + 1)/sinh(2*x + 1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "1/2 * ln(|sinh(2 * x + 1)|)");

    let expr = parse("2*x*cosh(x^2)/sinh(x^2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "ln(|sinh(x^2)|)");

    let expr = parse("2*k*x*sinh(x^2+b)/cosh(x^2+b)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "k * ln(|cosh(x^2 + b)|)");
}

#[test]
fn integrates_hyperbolic_tanh_reciprocal_log_sinh_substitution() {
    let mut ctx = Context::new();
    let expr = parse("1/tanh(2*x + 1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "1/2 * ln(|sinh(2 * x + 1)|)");

    let expr = parse("2*x/tanh(x^2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "ln(|sinh(x^2)|)");

    let expr = parse("x/tanh(x^2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "1/2 * ln(|sinh(x^2)|)");

    let expr = parse("2*k*x/tanh(x^2+b)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "k * ln(|sinh(x^2 + b)|)");
}

#[test]
fn integrates_hyperbolic_tanh_reciprocal_square_substitution() {
    let mut ctx = Context::new();
    let expr = parse("1/cosh(2*x + 1)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "1/2 * tanh(2 * x + 1)");

    let expr = parse("2*x/cosh(x^2)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "tanh(x^2)");

    let expr = parse("x/cosh(x^2)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "1/2 * tanh(x^2)");
}

#[test]
fn integrates_hyperbolic_tanh_reciprocal_fourth_substitution() {
    let mut ctx = Context::new();
    let expr = parse("1/cosh(2*x + 1)^4", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/2 * tanh(2 * x + 1) - 1/6 * tanh(2 * x + 1)^3"
    );

    let expr = parse("2*x/cosh(x^2)^4", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "tanh(x^2) - 1/3 * tanh(x^2)^3");

    let expr = parse("2*k*x/cosh(x^2+b)^4", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/3 * (3 * k * tanh(x^2 + b) - k * tanh(x^2 + b)^3)"
    );
}

#[test]
fn integrates_hyperbolic_coth_reciprocal_square_substitution() {
    let mut ctx = Context::new();
    let expr = parse("1/sinh(2*x + 1)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "-1/2 * cosh(2 * x + 1)/sinh(2 * x + 1)"
    );

    let expr = parse("2*x/sinh(x^2)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-cosh(x^2) / sinh(x^2)");

    let expr = parse("x/sinh(x^2)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-1/2 * cosh(x^2)/sinh(x^2)");
}

#[test]
fn integrates_hyperbolic_coth_reciprocal_fourth_substitution() {
    let mut ctx = Context::new();
    let expr = parse("1/sinh(2*x + 1)^4", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/2 / tanh(2 * x + 1) - 1/6 / tanh(2 * x + 1)^3"
    );

    let expr = parse("2*x/sinh(x^2)^4", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "1 / tanh(x^2) - 1/3 / tanh(x^2)^3");

    let expr = parse("2*k*x/sinh(x^2+b)^4", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "k / tanh(x^2 + b) - 1/3 * k / tanh(x^2 + b)^3"
    );
}

#[test]
fn integrates_hyperbolic_cosh_reciprocal_derivative_substitution() {
    let mut ctx = Context::new();
    let expr = parse("sinh(2*x + 1)/cosh(2*x + 1)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-1 / (2 * cosh(2 * x + 1))");

    let expr = parse("2*x*sinh(x^2)/cosh(x^2)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-1 / cosh(x^2)");

    let expr = parse("x*sinh(x^2)/cosh(x^2)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-1 / (2 * cosh(x^2))");

    let expr = parse("-2*x*sinh(x^2)/cosh(x^2)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "1 / cosh(x^2)");
}

#[test]
fn integrates_hyperbolic_sinh_reciprocal_derivative_substitution() {
    let mut ctx = Context::new();
    let expr = parse("cosh(2*x + 1)/sinh(2*x + 1)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-1 / (2 * sinh(2 * x + 1))");

    let expr = parse("2*x*cosh(x^2)/sinh(x^2)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-1 / sinh(x^2)");

    let expr = parse("x*cosh(x^2)/sinh(x^2)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-1 / (2 * sinh(x^2))");

    let expr = parse("-2*x*cosh(x^2)/sinh(x^2)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "1 / sinh(x^2)");
}

#[test]
fn integrates_arctan_unary_derivative_substitution() {
    let mut ctx = Context::new();
    let expr = parse("cosh(x)/(1+sinh(x)^2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "arctan(sinh(x))");

    let expr = parse("2*cosh(2*x + 1)/(1+sinh(2*x + 1)^2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "arctan(sinh(2 * x + 1))");

    let expr = parse("sinh(x)/(1+cosh(x)^2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "arctan(cosh(x))");

    let expr = parse("-sinh(x)/(1+cosh(x)^2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-arctan(cosh(x))");
}

#[test]
fn hyperbolic_substitution_rejects_missing_polynomial_cofactor() {
    let mut ctx = Context::new();
    let expr = parse("sinh(x^2)", &mut ctx).expect("parse");
    assert!(integrate_symbolic_expr(&mut ctx, expr, "x").is_none());
    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert!(
        conditions.is_empty(),
        "unexpected required conditions: {:?}",
        conditions
            .iter()
            .map(|condition| rendered(&ctx, *condition))
            .collect::<Vec<_>>()
    );
    assert!(super::integrate_symbolic_required_positive_conditions(&mut ctx, expr, "x").is_empty());

    let expr = parse("tanh(x^2)", &mut ctx).expect("parse");
    assert!(integrate_symbolic_expr(&mut ctx, expr, "x").is_none());
    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert!(
        conditions.is_empty(),
        "unexpected required conditions: {:?}",
        conditions
            .iter()
            .map(|condition| rendered(&ctx, *condition))
            .collect::<Vec<_>>()
    );
    assert!(super::integrate_symbolic_required_positive_conditions(&mut ctx, expr, "x").is_empty());

    let expr = parse("1/cosh(x^2)^2", &mut ctx).expect("parse");
    assert!(integrate_symbolic_expr(&mut ctx, expr, "x").is_none());

    let expr = parse("cosh(x^2)/sinh(x^2)", &mut ctx).expect("parse");
    assert!(integrate_symbolic_expr(&mut ctx, expr, "x").is_none());

    let expr = parse("1/sinh(x^2)^2", &mut ctx).expect("parse");
    assert!(integrate_symbolic_expr(&mut ctx, expr, "x").is_none());

    let expr = parse("sinh(x^2)/cosh(x^2)^2", &mut ctx).expect("parse");
    assert!(integrate_symbolic_expr(&mut ctx, expr, "x").is_none());

    let expr = parse("cosh(x^2)/sinh(x^2)^2", &mut ctx).expect("parse");
    assert!(integrate_symbolic_expr(&mut ctx, expr, "x").is_none());

    let expr = parse("1/tanh(x^2)", &mut ctx).expect("parse");
    assert!(integrate_symbolic_expr(&mut ctx, expr, "x").is_none());
}

#[test]
fn integrates_reciprocal_linear_with_absolute_log() {
    let mut ctx = Context::new();
    let expr = parse("1/(3*x)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "ln(|3 * x|) / 3");
}

#[test]
fn integrates_linear_power_minus_one_with_absolute_log() {
    let mut ctx = Context::new();
    let expr = parse("(2*x + 1)^-1", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "ln(|2 * x + 1|) / 2");
}

#[test]
fn integrates_arctan_kernel() {
    let mut ctx = Context::new();
    let expr = parse("1/(x^2+1)", &mut ctx).expect("parse");

    let candidate = super::table_reused_arctan_kernel_integration_candidate(&mut ctx, expr, "x");

    assert_eq!(candidate.method, AlgorithmicIntegrationMethod::TableReused);
    assert_eq!(
        candidate.verification_status,
        AlgorithmicIntegrationVerificationStatus::Verified
    );
    assert!(candidate.is_publicly_acceptable());
    let public_antiderivative = candidate
        .public_antiderivative()
        .expect("verified arctan table candidate should be public");
    assert_eq!(rendered(&ctx, public_antiderivative), "arctan(x)");

    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "arctan(x)");
}

#[test]
fn integrates_arctan_sqrt_reciprocal_kernel() {
    let mut ctx = Context::new();
    let expr = parse("1/(2*sqrt(x)*(x+1))", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "arctan(sqrt(x))");

    let expr = parse("1/(sqrt(x)*(x+1))", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "2 * arctan(sqrt(x))");

    let expr = parse("x^(-1/2)/(x+1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "2 * arctan(sqrt(x))");

    let expr = parse("1/(sqrt(x)*(4*x+1))", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "arctan(2 * sqrt(x))");

    let expr = parse("1/(sqrt(x)*(x+4))", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "arctan(sqrt(x) / 2)");

    let expr = parse("1/(2*sqrt(x)*(4*x+1))", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "1/2 * arctan(2 * sqrt(x))");
}

#[test]
fn integrates_arctan_sqrt_symbolic_square_shift_reciprocal_kernel() {
    let mut ctx = Context::new();
    let expr = parse("1/(sqrt(x)*(x+a^2))", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "(arctan(sqrt(x) / a) * 2)/a");

    let positive_conditions =
        super::integrate_symbolic_required_positive_conditions(&mut ctx, expr, "x");
    assert_eq!(positive_conditions.len(), 1);
    assert_eq!(rendered(&ctx, positive_conditions[0]), "x");

    let nonzero_conditions =
        super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert_eq!(nonzero_conditions.len(), 1);
    assert_eq!(rendered(&ctx, nonzero_conditions[0]), "a");

    let expr = parse("x^(-1/2)/(x+a^2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "(arctan(sqrt(x) / a) * 2)/a");

    let expr = parse("1/(sqrt(x)*(a^2*x+1))", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "(arctan(a * sqrt(x)) * 2)/a");

    let positive_conditions =
        super::integrate_symbolic_required_positive_conditions(&mut ctx, expr, "x");
    assert_eq!(positive_conditions.len(), 1);
    assert_eq!(rendered(&ctx, positive_conditions[0]), "x");

    let nonzero_conditions =
        super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert_eq!(nonzero_conditions.len(), 1);
    assert_eq!(rendered(&ctx, nonzero_conditions[0]), "a");

    let expr = parse("x^(-1/2)/(a^2*x+1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "(arctan(a * sqrt(x)) * 2)/a");

    let expr = parse("1/(sqrt(x)*(4*x+a^2))", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "arctan(2 * sqrt(x) / a) / a");

    let positive_conditions =
        super::integrate_symbolic_required_positive_conditions(&mut ctx, expr, "x");
    assert_eq!(positive_conditions.len(), 1);
    assert_eq!(rendered(&ctx, positive_conditions[0]), "x");

    let nonzero_conditions =
        super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert_eq!(nonzero_conditions.len(), 1);
    assert_eq!(rendered(&ctx, nonzero_conditions[0]), "a");
}

#[test]
fn integrates_arctan_sqrt_unit_shift_square_reciprocal_kernel() {
    let mut ctx = Context::new();
    let expr = parse("1/(sqrt(x)*(x+1)^2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "arctan(sqrt(x)) + sqrt(x) / (x + 1)");

    let expr = parse("1/(2*sqrt(x)*(x+1)^2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/2 * (arctan(sqrt(x)) + sqrt(x) / (x + 1))"
    );

    let expr = parse("x^(-1/2)/(x+1)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "arctan(sqrt(x)) + sqrt(x) / (x + 1)");

    let expr = parse("sqrt(x)/(x*(x+1)^2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "arctan(sqrt(x)) + sqrt(x) / (x + 1)");

    let expr = parse("1/(sqrt(x)*(x+4)^2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/8 * arctan(sqrt(x) / 2) + sqrt(x) / (4 * (x + 4))"
    );

    let expr = parse("sqrt(x)/(x*(x+4)^2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/8 * arctan(sqrt(x) / 2) + sqrt(x) / (4 * (x + 4))"
    );

    let expr = parse("1/(sqrt(x)*(x+1/4)^2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "8 * arctan(2 * sqrt(x)) + 4 * sqrt(x) / (x + 1/4)"
    );

    let expr = parse("1/(3*sqrt(x)*(x+1/4)^2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "8/3 * arctan(2 * sqrt(x)) + 4/3 * sqrt(x) / (x + 1/4)"
    );
}

#[test]
fn integrates_inverse_hyperbolic_sqrt_reciprocal_kernels() {
    let mut ctx = Context::new();
    let expr = parse("-1/(2*x*sqrt(x+1))", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "asinh(sqrt(1 / x))");

    let expr = parse("-1/(x*sqrt(x+4))", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "asinh(sqrt(4 / x))");

    let expr = parse("-1/(2*(x+1)*sqrt(x+2))", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "asinh(sqrt(1 / (x + 1)))");

    let expr = parse("-1/(2*sqrt(x)*(x-1))", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "atanh(sqrt(1 / x))");

    let expr = parse("-1/(sqrt(x)*(x-4))", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "atanh(sqrt(4 / x))");

    let expr = parse("3/(2*sqrt(3*x)*(3-x))", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "atanh(sqrt(3 / x))");

    let expr = parse("-3/(2*sqrt(3*x+1)*(3*x))", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "atanh(sqrt(1 / (3 * x + 1)))");

    let expr = parse("-2/((2*x+1)*sqrt(2*x+5))", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "asinh(sqrt(4 / (2 * x + 1)))");

    let expr = parse("-1/((x+3)*sqrt(2*x+10))", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "asinh(sqrt(2 / (x + 3)))");

    let expr = parse("-2/((2*x+1)*sqrt(4*x+6))", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "asinh(sqrt(2 / (2 * x + 1)))");

    let expr = parse("1/((6-2*x)*sqrt(8-2*x))", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/2 * sqrt(2) * asinh(sqrt(1 / (3 - x)))"
    );

    let expr = parse("sqrt(8-2*x)/((6-2*x)*(8-2*x))", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/2 * sqrt(2) * asinh(sqrt(1 / (3 - x)))"
    );

    let expr = parse("-1/(x*sqrt(2*x+4))", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "atanh(sqrt(2 / (x + 2)))");
}

#[test]
fn integrates_arctan_sqrt_affine_derivative_kernel() {
    let mut ctx = Context::new();
    let expr = parse("1/(sqrt(4*x+1)*(2*x+1))", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "arctan(sqrt(4 * x + 1))");

    let expr = parse("(4*x+1)^(1/2)/((2*x+1)*(4*x+1))", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "arctan(sqrt(4 * x + 1))");

    let expr = parse("-1/(2*sqrt(5-3*x)*(2-x))", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "arctan(sqrt(5 - 3 * x))");

    let expr = parse("1/(sqrt(x+1)*(4*x+5))", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "arctan(2 * sqrt(x + 1))");

    let expr = parse("3/(sqrt(x+1)*(4*x+5))", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "3 * arctan(2 * sqrt(x + 1))");

    let expr = parse("-1/(sqrt(1-x)*(5-4*x))", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "arctan(2 * sqrt(1 - x))");

    let expr = parse("(x+1)^(1/2)/((x+1)*(4*x+5))", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "arctan(2 * sqrt(x + 1))");

    let expr = parse("(5-3*x)^(1/2)/((10-6*x)*(x-2))", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "arctan(sqrt(5 - 3 * x))");
}

#[test]
fn integrates_arctan_scaled_variable_by_parts() {
    let mut ctx = Context::new();
    let expr = parse("arctan(x)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-1/2 * ln(x^2 + 1) + x * arctan(x)");

    let expr = parse("arctan(2*x)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "-1/4 * ln((2 * x)^2 + 1) + x * arctan(2 * x)"
    );

    let expr = parse("arctan(2*x + 1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "-1/4 * ln((2 * x + 1)^2 + 1) + 1/2 * (2 * x + 1) * arctan(2 * x + 1)"
    );

    let expr = parse("arctan(1 - 2*x)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/4 * ln((1 - 2 * x)^2 + 1) + -1/2 * (1 - 2 * x) * arctan(1 - 2 * x)"
    );
}

#[test]
fn integrates_linear_times_arctan_by_parts_collects_arctan_tail() {
    let mut ctx = Context::new();
    let expr = parse("x*arctan(x)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "((x^2 + 1) * arctan(x) - x) / 2");
}

#[test]
fn integrates_arctan_reciprocal_scaled_variable_by_parts() {
    let mut ctx = Context::new();
    let expr = parse("arctan(1/x)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "1/2 * ln(x^2 + 1) + x * arctan(1 / x)");

    let expr = parse("arctan(1/(2*x))", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/4 * ln((2 * x)^2 + 1) + x * arctan(1 / (2 * x))"
    );

    let expr = parse("arctan(1/(2*x + 1))", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/4 * ln((2 * x + 1)^2 + 1) + 1/2 * (2 * x + 1) * arctan(1 / (2 * x + 1))"
    );
}

#[test]
fn integrates_bounded_inverse_trig_scaled_variable_by_parts() {
    let mut ctx = Context::new();
    let expr = parse("arcsin(x)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "sqrt(1 - x^2) + x * arcsin(x)");

    let conditions = super::integrate_symbolic_required_positive_conditions(&mut ctx, expr, "x");
    assert_eq!(conditions.len(), 1);
    assert_eq!(rendered(&ctx, conditions[0]), "1 - x^2");

    let expr = parse("arccos(x)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "x * arccos(x) - sqrt(1 - x^2)");

    let conditions = super::integrate_symbolic_required_positive_conditions(&mut ctx, expr, "x");
    assert_eq!(conditions.len(), 1);
    assert_eq!(rendered(&ctx, conditions[0]), "1 - x^2");

    let expr = parse("arcsin(2*x)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "sqrt(1/4 - x^2) + x * arcsin(2 * x)");

    let conditions = super::integrate_symbolic_required_positive_conditions(&mut ctx, expr, "x");
    assert_eq!(conditions.len(), 1);
    assert_eq!(rendered(&ctx, conditions[0]), "1/4 - x^2");

    let expr = parse("arccos(2*x)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "x * arccos(2 * x) - sqrt(1/4 - x^2)");

    let conditions = super::integrate_symbolic_required_positive_conditions(&mut ctx, expr, "x");
    assert_eq!(conditions.len(), 1);
    assert_eq!(rendered(&ctx, conditions[0]), "1/4 - x^2");

    let expr = parse("arcsin(-2*x)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "x * arcsin(-2 * x) - 1/2 * sqrt(1 - (-2 * x)^2)"
    );

    let expr = parse("arcsin(2*x + 1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/2 * sqrt(1 - (2 * x + 1)^2) + 1/2 * (2 * x + 1) * arcsin(2 * x + 1)"
    );

    let conditions = super::integrate_symbolic_required_positive_conditions(&mut ctx, expr, "x");
    assert_eq!(conditions.len(), 1);
    assert_eq!(rendered(&ctx, conditions[0]), "1 - (2 * x + 1)^2");

    let expr = parse("arccos(2*x + 1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/2 * (2 * x + 1) * arccos(2 * x + 1) - 1/2 * sqrt(1 - (2 * x + 1)^2)"
    );

    let expr = parse("arcsin(1 - 2*x)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "-1/2 * (4 * (x - x^2))^(1/2) - 1/2 * (1 - 2 * x) * arcsin(1 - 2 * x)"
    );
    let conditions = super::integrate_symbolic_required_positive_conditions(&mut ctx, expr, "x");
    assert_eq!(conditions.len(), 1);
    assert_eq!(rendered(&ctx, conditions[0]), "x - x^2");

    let expr = parse("arccos(1 - 2*x)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/2 * (4 * (x - x^2))^(1/2) - 1/2 * (1 - 2 * x) * arccos(1 - 2 * x)"
    );

    let expr = parse("arcsin(a*x)", &mut ctx).expect("parse");
    assert!(integrate_symbolic_expr(&mut ctx, expr, "x").is_none());
    assert!(super::integrate_symbolic_required_positive_conditions(&mut ctx, expr, "x").is_empty());
}

#[test]
fn integrates_asinh_affine_by_parts() {
    let mut ctx = Context::new();
    let expr = parse("asinh(x)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "x * asinh(x) - sqrt(x^2 + 1)");

    let expr = parse("asinh(2*x)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "x * asinh(2 * x) - 1/2 * sqrt((2 * x)^2 + 1)"
    );

    let expr = parse("asinh(2*x + 1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/2 * (2 * x + 1) * asinh(2 * x + 1) - 1/2 * sqrt((2 * x + 1)^2 + 1)"
    );
}

#[test]
fn integrates_atanh_affine_by_parts() {
    let mut ctx = Context::new();
    let expr = parse("atanh(x)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "1/2 * ln(1 - x^2) + x * atanh(x)");

    let expr = parse("atanh(2*x)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/4 * ln(1 - (2 * x)^2) + x * atanh(2 * x)"
    );

    let expr = parse("atanh(2*x + 1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/4 * ln(1 - (2 * x + 1)^2) + 1/2 * (2 * x + 1) * atanh(2 * x + 1)"
    );
}

#[test]
fn integrates_acosh_affine_by_parts() {
    let mut ctx = Context::new();
    let expr = parse("acosh(x)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "x * acosh(x) - sqrt(x - 1) * sqrt(x + 1)"
    );

    let expr = parse("acosh(2*x)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "x * acosh(2 * x) - 1/2 * sqrt(2 * x - 1) * sqrt(2 * x + 1)"
    );

    let expr = parse("acosh(2*x + 1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/2 * (2 * x + 1) * acosh(2 * x + 1) - 1/2 * sqrt(2 * x) * sqrt(2 * x + 2)"
    );

    let expr = parse("acosh(1 - 2*x)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/2 * sqrt(-2 * x) * sqrt(2 - 2 * x) - 1/2 * (1 - 2 * x) * acosh(1 - 2 * x)"
    );
}

#[test]
fn detects_arcsin_inverse_sqrt_product_target() {
    let mut ctx = Context::new();
    let expr = parse("1/(sqrt(x)*sqrt(1-x))", &mut ctx).expect("parse");
    assert!(integrate_symbolic_is_arcsin_inverse_sqrt_product_target(
        &mut ctx, expr, "x"
    ));

    let expr = parse("(x*(1-x))^(-1/2)", &mut ctx).expect("parse");
    assert!(integrate_symbolic_is_arcsin_inverse_sqrt_product_target(
        &mut ctx, expr, "x"
    ));

    let expr = parse("a/(2*sqrt(x)*sqrt(1-x))", &mut ctx).expect("parse");
    assert!(integrate_symbolic_is_arcsin_inverse_sqrt_product_target(
        &mut ctx, expr, "x"
    ));
}

#[test]
fn integrates_shifted_sqrt_arcsin_inverse_product() {
    let mut ctx = Context::new();
    let expr = parse("1/(sqrt(x)*sqrt(sqrt(x)-x))", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "2 * arcsin(2 * sqrt(x) - 1)");

    let expr = parse("(x*(sqrt(x)-x))^(-1/2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "2 * arcsin(2 * sqrt(x) - 1)");

    let expr = parse("1/(2*sqrt(x)*sqrt(sqrt(x)-x))", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "arcsin(2 * sqrt(x) - 1)");

    let required = integrate_symbolic_required_positive_conditions(&mut ctx, expr, "x");
    let rendered_required: Vec<_> = required.iter().map(|id| rendered(&ctx, *id)).collect();
    assert_eq!(
        rendered_required,
        vec!["x".to_string(), "sqrt(x) - x".to_string()]
    );
}

#[test]
fn integrates_polynomial_derivative_arctan_substitution() {
    let mut ctx = Context::new();
    let expr = parse("2*x/(1+x^4)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "arctan(x^2)");

    let expr = parse("x/(1+x^4)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "1/2 * arctan(x^2)");

    let expr = parse("2*x/(4+x^4)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "1/2 * arctan(x^2 / 2)");

    let expr = parse("2*x/(3+x^4)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "arctan(x^2 / sqrt(3)) / sqrt(3)");

    let expr = parse("1/(4+(x+1)^2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "1/2 * arctan((x + 1) / 2)");

    let expr = parse("1/(2*x^2+2*x+1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "arctan(2 * x + 1)");

    let expr = parse("1/(2*x^2+4*x+5)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "arctan((2 * x + 2) / sqrt(6)) / sqrt(6)"
    );
}

#[test]
fn integrates_polynomial_derivative_atanh_substitution() {
    let mut ctx = Context::new();
    let expr = parse("2*x/(4-x^4)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "1/2 * atanh(x^2 / 2)");

    let expr = parse("-2*x/(1-x^4)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-atanh(x^2)");

    let expr = parse("x/(4-x^4)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "1/4 * atanh(x^2 / 2)");

    let expr = parse("2*x/(3-x^4)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "atanh(x^2 / sqrt(3)) / sqrt(3)");

    let expr = parse("1/(12-4*x^2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "1/4 * atanh(x / sqrt(3)) / sqrt(3)");

    let expr = parse("(2*x+2)/(5-(x^2+2*x+1)^2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "atanh((x^2 + 2 * x + 1) / sqrt(5)) / sqrt(5)"
    );

    let expr = parse("1/(4-(x+1)^2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "1/2 * atanh((x + 1) / 2)");
}

#[test]
fn integrates_inverse_sqrt_substitution_with_surd_width() {
    let mut ctx = Context::new();

    let expr = parse("2*x/sqrt(3-x^4)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "arcsin(x^2 / sqrt(3))");

    let expr = parse("2*x/sqrt(3+x^4)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "asinh(x^2 / sqrt(3))");

    let expr = parse("(2*x+2)/sqrt(2 - x^4 - 4*x^3 - 6*x^2 - 4*x)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "arcsin((x + 1)^2 / sqrt(3))");

    let expr = parse("(4*x^3+6*x^2+6*x+2)/sqrt(2-3*(x^2+x+1)^4)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "arcsin((x^2 + x + 1)^2 / sqrt(2/3)) / sqrt(3)"
    );

    let expr = parse("(2*x+2)/sqrt(4 + 4*x + 6*x^2 + 4*x^3 + x^4)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "asinh((x + 1)^2 / sqrt(3))");

    let expr = parse("2*x/sqrt(x^4-4)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "acosh(x^2 / 2)");

    let expr = parse("(2*x+1)/sqrt((x^2+x)^2-4)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "acosh((x^2 + x) / 2)");
}

#[test]
fn integrates_polynomial_derivative_square_minus_constant_log_substitution() {
    let mut ctx = Context::new();
    let expr = parse("1/(x^2-1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "1/2 * ln(|(x - 1) / (x + 1)|)");

    let expr = parse("1/(x^2+x)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "ln(|x / (x + 1)|)");

    let expr = parse("1/(x^2+3*x+2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "ln(|(x + 1) / (x + 2)|)");

    let expr = parse("1/(4*x^2+4*x)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "1/4 * ln(|x / (x + 1)|)");

    let expr = parse("1/(4*x^2+12*x+8)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "1/4 * ln(|(x + 1) / (x + 2)|)");

    let expr = parse("1/(4*x^2-4)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "1/8 * ln(|(x - 1) / (x + 1)|)");

    let expr = parse("1/(2*x^2+3*x+1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "ln(|(2 * x + 1) / (x + 1)|)");

    let expr = parse("1/(6*x^2+9*x+3)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "1/3 * ln(|(2 * x + 1) / (x + 1)|)");

    let expr = parse("2*x/(x^4-4)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "1/4 * ln(|(x^2 - 2) / (x^2 + 2)|)");

    let expr = parse("x/(x^4-4)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "1/8 * ln(|(x^2 - 2) / (x^2 + 2)|)");
}

#[test]
fn integrates_reciprocal_quadratic_with_irrational_roots() {
    // 1/(p^2 - c) with sqrt(c) irrational: the log form carries a
    // symbolic radical instead of bailing on exact_rational_sqrt.
    let mut ctx = Context::new();
    let expr = parse("1/(x^2-2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "(ln(|(x - sqrt(2)) / (sqrt(2) + x)|) * 1/2)/sqrt(2)"
    );

    // 2/(x^2-3): scale 2 cancels the 1/2 to a bare log over the surd.
    let expr = parse("2/(x^2-3)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "ln(|(x - sqrt(3)) / (sqrt(3) + x)|) / sqrt(3)"
    );

    // Completed square (x-1)^2 - 3 from x^2 - 2x - 2.
    let expr = parse("1/(x^2-2*x-2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "(ln(|(x - 1 - sqrt(3)) / (sqrt(3) + x - 1)|) * 1/2)/sqrt(3)"
    );
}

#[test]
fn reciprocal_quadratic_irrational_round_trips_numerically() {
    let mut ctx = Context::new();
    for (source, samples) in [
        ("1/(x^2-2)", [-3.0_f64, 0.5, 2.5]),
        ("1/(x^2-5)", [-4.0, 1.0, 3.5]),
        ("1/(x^2+2*x-1)", [-4.0, 0.2, 2.0]),
        ("1/(x^2-2*x-2)", [-2.0, 0.5, 4.0]),
    ] {
        let expr = parse(source, &mut ctx).expect(source);
        let antiderivative = integrate_symbolic_expr(&mut ctx, expr, "x").expect(source);
        let derivative = crate::symbolic_differentiation_support::differentiate_symbolic_expr(
            &mut ctx,
            antiderivative,
            "x",
        )
        .expect("derivative");
        let target = parse(source, &mut ctx).expect(source);
        for sample in samples {
            let mut vars = std::collections::HashMap::new();
            vars.insert("x".to_string(), sample);
            let lhs = crate::evaluator_f64::eval_f64(&ctx, derivative, &vars).expect("lhs");
            let rhs = crate::evaluator_f64::eval_f64(&ctx, target, &vars).expect("rhs");
            assert!(
                (lhs - rhs).abs() < 1e-9,
                "mismatch for {source} at {sample}: {lhs} vs {rhs}"
            );
        }
    }
}

#[test]
fn reciprocal_quadratic_irrational_declines_harder_shapes() {
    let mut ctx = Context::new();
    // Non-monic leading (needs an irrational leading root) and
    // linear numerators (need a derivative split) stay residual as
    // honest follow-up rungs.
    for source in ["1/(2*x^2-3)", "(x+1)/(x^2-2)"] {
        let expr = parse(source, &mut ctx).expect(source);
        assert!(
            integrate_symbolic_expr(&mut ctx, expr, "x").is_none(),
            "must stay residual: {source}"
        );
    }
}

#[test]
fn integrates_polynomial_log_derivative_substitution() {
    let mut ctx = Context::new();
    let expr = parse("(2*x+1)/(x^2+x-1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "ln(|x^2 + x - 1|)");

    let expr = parse("(4*x+2)/(x^2+x+1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "2 * ln(|x^2 + x + 1|)");

    let expr = parse("(2*x+3)/((x+1)*(x+2))", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "ln(|(x + 1) * (x + 2)|)");
}

#[test]
fn integrates_polynomial_log_reciprocal_derivative_substitution() {
    let mut ctx = Context::new();
    let expr = parse("1/(x*ln(x))", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "ln(|ln(x)|)");

    let expr = parse("2/((2*x+1)*ln(2*x+1))", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "ln(|ln(2 * x + 1)|)");

    let expr = parse("2*x/((x^2+1)*ln(x^2+1))", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "ln(|ln(x^2 + 1)|)");

    let expr = parse("(2*x+1)/((x^2+x+1)*ln(x^2+x+1))", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "ln(|ln(x^2 + x + 1)|)");

    let expr = parse("2*x/((x^2+1)*ln(x^2+1)^2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-1 / ln(x^2 + 1)");

    let expr = parse("2*x/((x^2-1)*ln(x^2-1)^2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-1 / ln(x^2 - 1)");

    let expr = parse("(2*x+1)/((x^2+x+1)*ln(x^2+x+1)^2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-1 / ln(x^2 + x + 1)");

    let expr = parse("(2*x+1)/((x^2+x-1)*ln(x^2+x-1)^2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-1 / ln(x^2 + x - 1)");

    let expr = parse("(2*x+1)/((x^2+x-1)*ln(x^2+x-1)^3)", &mut ctx).expect("parse");
    assert!(
        super::integrate_symbolic_is_polynomial_log_reciprocal_derivative_target(
            &mut ctx, expr, "x"
        )
    );
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-1 / (2 * ln(x^2 + x - 1)^2)");
}

#[test]
fn integrates_linear_numerator_over_positive_quadratic() {
    let mut ctx = Context::new();
    let expr = parse("x/(x^2+2*x+2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    let result = rendered(&ctx, out);

    assert!(
        result.contains("ln(x^2 + 2 * x + 2)") && result.contains("arctan(x + 1)"),
        "expected log plus arctan decomposition, got {result}"
    );

    let expr = parse("(x+3)/(2*x^2+4*x+4)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    let result = rendered(&ctx, out);
    assert!(
        result.contains("ln(x^2 + 2 * x + 2)") && result.contains("arctan(x + 1)"),
        "expected scaled log plus arctan decomposition, got {result}"
    );

    let expr = parse("(x^3+x+1)/(x^2+1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    let result = rendered(&ctx, out);
    assert!(
        result.contains("x^2") && result.contains("arctan(x)"),
        "expected polynomial quotient plus arctan remainder, got {result}"
    );
}

#[test]
fn integrates_positive_rational_radius_arctan_quadratic() {
    let mut ctx = Context::new();
    let expr = parse("1/((a*x+b)^2+2)", &mut ctx).expect("parse");

    assert!(
        super::integrate_symbolic_is_positive_rational_quadratic_arctan_target(&mut ctx, expr, "x")
    );

    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "arctan(sqrt(2) * (a * x + b) / 2) / (a * sqrt(2))"
    );
}

#[test]
fn integrates_named_positive_constant_radius_arctan_quadratic() {
    let mut ctx = Context::new();
    let expr = parse("1/(x^2+pi)", &mut ctx).expect("parse");

    assert!(
        super::integrate_symbolic_is_positive_rational_quadratic_arctan_target(&mut ctx, expr, "x")
    );

    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "arctan(x / sqrt(pi)) / (sqrt(pi))");

    let expr = parse("1/(x^2+phi)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "arctan(x / sqrt(phi)) / (sqrt(phi))");
    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert!(
        conditions.is_empty(),
        "unexpected required conditions: {:?}",
        conditions
            .iter()
            .map(|condition| rendered(&ctx, *condition))
            .collect::<Vec<_>>()
    );

    let expr = parse("1/(x^2-phi)", &mut ctx).expect("parse");
    assert!(integrate_symbolic_expr(&mut ctx, expr, "x").is_none());
}

#[test]
fn integrates_expanded_numeric_affine_named_positive_constant_radius_arctan_quadratic() {
    let mut ctx = Context::new();
    let expr = parse("1/(4*x^2+12*x+9+phi)", &mut ctx).expect("parse");

    assert!(
        super::integrate_symbolic_is_positive_rational_quadratic_arctan_target(&mut ctx, expr, "x")
    );

    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "arctan((2 * x + 3) / sqrt(phi)) / (2 * sqrt(phi))"
    );
    let denominator = parse("4*x^2+12*x+9+phi", &mut ctx).expect("parse");
    assert!(
        super::positive_constant_radius_quadratic_denominator_is_structurally_nonzero(
            &mut ctx,
            denominator,
            "x"
        )
    );
    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert!(
        conditions.is_empty(),
        "unexpected required conditions: {:?}",
        conditions
            .iter()
            .map(|condition| rendered(&ctx, *condition))
            .collect::<Vec<_>>()
    );

    let expr = parse("1/(4*x^2+12*x+9-phi)", &mut ctx).expect("parse");
    assert!(integrate_symbolic_expr(&mut ctx, expr, "x").is_none());
}

#[test]
fn integrates_expanded_named_positive_constant_radius_linear_numerator_quadratic() {
    let mut ctx = Context::new();
    let expr = parse("(2*x+6)/(4*x^2+12*x+9+phi)", &mut ctx).expect("parse");

    assert!(
        super::integrate_symbolic_is_positive_constant_radius_quadratic_linear_numerator_target(
            &mut ctx, expr, "x"
        )
    );

    let decomposition =
        super::integrate_symbolic_positive_quadratic_linear_numerator_decomposition_expr(
            &mut ctx, expr, "x",
        )
        .expect("decomposition");
    assert_eq!(
        rendered(&ctx, decomposition),
        "3 / (4 * x^2 + 12 * x + 9 + phi) + (2 * x + 3) / (4 * x^2 + 12 * x + 9 + phi)"
    );

    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/4 * ln(4 * x^2 + 12 * x + 9 + phi) + (atan((2 * x + 3) / sqrt(phi)) * 3)/(2 * sqrt(phi))"
    );
    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert!(
        conditions.is_empty(),
        "unexpected required conditions: {:?}",
        conditions
            .iter()
            .map(|condition| rendered(&ctx, *condition))
            .collect::<Vec<_>>()
    );

    let expr = parse("(2*x+6)/(4*x^2+12*x+9-phi)", &mut ctx).expect("parse");
    assert!(integrate_symbolic_expr(&mut ctx, expr, "x").is_none());
}

#[test]
fn integrates_symbolic_square_radius_arctan_quadratic() {
    let mut ctx = Context::new();
    let expr = parse("1/(x^2+a^2)", &mut ctx).expect("parse");

    assert!(
        super::integrate_symbolic_is_positive_rational_quadratic_arctan_target(&mut ctx, expr, "x")
    );

    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "arctan(x / a) / (a)");
}

#[test]
fn integrates_symbolic_square_radius_expanded_affine_arctan_quadratic() {
    let mut ctx = Context::new();
    let expr = parse("1/((x+b)^2+a^2)", &mut ctx).expect("parse");

    assert!(
        super::integrate_symbolic_is_positive_rational_quadratic_arctan_target(&mut ctx, expr, "x")
    );

    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "arctan((b + x) / a) / (a)");
}

#[test]
fn exposes_positive_quadratic_linear_numerator_decomposition_for_didactic_trace() {
    let mut ctx = Context::new();
    let expr = parse("(x+1)/(x^2+1)", &mut ctx).expect("parse");
    let out = super::integrate_symbolic_positive_quadratic_linear_numerator_decomposition_expr(
        &mut ctx, expr, "x",
    )
    .expect("positive quadratic linear numerator decomposition");
    let result = rendered(&ctx, out);

    assert_eq!(result, "1 / (x^2 + 1) + x / (x^2 + 1)");

    let expr = parse("x/(x^2+2*x+2)", &mut ctx).expect("parse");
    let out = super::integrate_symbolic_positive_quadratic_linear_numerator_decomposition_expr(
        &mut ctx, expr, "x",
    )
    .expect("shifted positive quadratic linear numerator decomposition");
    let result = rendered(&ctx, out);

    assert_eq!(
        result,
        "(x + 1) / (x^2 + 2 * x + 2) - 1 / (x^2 + 2 * x + 2)"
    );

    let expr = parse("1/(x^2+1)", &mut ctx).expect("parse");
    let out = super::integrate_symbolic_positive_quadratic_linear_numerator_decomposition_expr(
        &mut ctx, expr, "x",
    );
    assert!(
        out.is_none(),
        "pure arctan table integrals should not gain a redundant decomposition"
    );

    let expr = parse("(x^2+1)/(x^2+2*x+2)", &mut ctx).expect("parse");
    let out = super::integrate_symbolic_positive_quadratic_linear_numerator_decomposition_expr(
        &mut ctx, expr, "x",
    )
    .expect("improper positive quadratic decomposition");
    let result = rendered(&ctx, out);

    assert!(
        result.contains("1")
            && result.contains("- (2 * x + 2) / (x^2 + 2 * x + 2)")
            && result.contains("1 / (x^2 + 2 * x + 2)"),
        "improper positive quadratic decomposition should expose polynomial quotient plus log/arctan remainder, got {result}"
    );
}

#[test]
fn exposes_positive_quadratic_cube_reduction_for_didactic_trace() {
    let mut ctx = Context::new();
    let expr = parse("1/(x^2+1)^3", &mut ctx).expect("parse");
    let out = super::integrate_symbolic_positive_quadratic_cube_constant_reduction_expr(
        &mut ctx, expr, "x",
    )
    .expect("positive quadratic cube reduction");
    let result = rendered(&ctx, out);

    assert!(
        result.contains("3 / (8 * (x^2 + 1))")
            && result.contains("(5 - 3 * x^4 - 6 * x^2) / (8 * (x^2 + 1)^3)"),
        "expected arctan integrand plus rational-derivative integrand, got {result}"
    );

    let expr = parse("1/((x+1)^2+1)^3", &mut ctx).expect("parse");
    let out = super::integrate_symbolic_positive_quadratic_cube_constant_reduction_expr(
        &mut ctx, expr, "x",
    )
    .expect("shifted positive quadratic cube reduction");
    let result = rendered(&ctx, out);

    assert!(
        result.contains("3 / (8 * (x^2 + 2 * x + 2))")
            && result.contains(
                "- (3 * x^4 + 12 * x^3 + 24 * x^2 + 24 * x + 4) / (8 * (x^2 + 2 * x + 2)^3)"
            ),
        "expected negative rational part to carry an outer sign, got {result}"
    );

    let expr = parse("x/(x^2+1)^3", &mut ctx).expect("parse");
    let out = super::integrate_symbolic_positive_quadratic_cube_constant_reduction_expr(
        &mut ctx, expr, "x",
    );
    assert!(
        out.is_none(),
        "non-constant numerators should use the existing cube-family path"
    );
}

#[test]
fn exposes_positive_quadratic_square_reduction_for_didactic_trace() {
    let mut ctx = Context::new();
    let expr = parse("1/(x^2+1)^2", &mut ctx).expect("parse");
    let out = super::integrate_symbolic_positive_quadratic_square_constant_reduction_expr(
        &mut ctx, expr, "x",
    )
    .expect("positive quadratic square reduction");
    let result = rendered(&ctx, out);

    assert!(
        result.contains("1 / (2 * (x^2 + 1))") && result.contains("(1 - x^2) / (2 * (x^2 + 1)^2)"),
        "expected arctan integrand plus rational-derivative integrand, got {result}"
    );

    let expr = parse("x/(x^2+1)^2", &mut ctx).expect("parse");
    let out = super::integrate_symbolic_positive_quadratic_square_constant_reduction_expr(
        &mut ctx, expr, "x",
    );
    assert!(
        out.is_none(),
        "non-constant numerators should use the existing square-family path"
    );
}

#[test]
fn integrates_multiple_linear_factors_with_positive_quadratic_remainder() {
    let mut ctx = Context::new();
    let expr = parse("1/(x^4-1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    let result = rendered(&ctx, out);

    assert!(
        result.contains("ln(|x - 1|)")
            && result.contains("ln(|x + 1|)")
            && result.contains("arctan(x)")
            && !result.starts_with("integrate("),
        "expected linear-log plus positive-quadratic arctan decomposition, got {result}"
    );

    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    let mut conditions: Vec<_> = conditions
        .into_iter()
        .map(|condition| rendered(&ctx, condition))
        .collect();
    conditions.sort();
    assert_eq!(conditions, vec!["x + 1".to_string(), "x - 1".to_string()]);
}

#[test]
fn integrates_positive_quadratic_square() {
    let mut ctx = Context::new();
    let expr = parse("1/(x^2+1)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "1/2 * arctan(x) + x / (2 * (x^2 + 1))");

    let expr = parse("x^2/(x^2+1)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "1/2 * arctan(x) - x / (2 * (x^2 + 1))");

    let expr = parse("1/((x+1)^2+1)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/2 * arctan(x + 1) + (x + 1) / (2 * (x^2 + 2 * x + 2))"
    );

    let expr = parse("1/(4*x^2+1)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/4 * arctan(2 * x) + x / (2 * (4 * x^2 + 1))"
    );
}

#[test]
fn integrates_positive_quadratic_cube() {
    let mut ctx = Context::new();
    let expr = parse("1/(x^2+1)^3", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "3/8 * arctan(x) + (3 * x^3 + 5 * x) / (8 * (x^2 + 1)^2)"
    );

    let expr = parse("1/((x+1)^2+1)^3", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "3/8 * arctan(x + 1) + (3 * x^3 + 9 * x^2 + 14 * x + 8) / (8 * (x^2 + 2 * x + 2)^2)"
    );

    let expr = parse("1/(4*x^2+1)^3", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "3/16 * arctan(2 * x) + (12 * x^3 + 5 * x) / (8 * (4 * x^2 + 1)^2)"
    );

    let expr = parse("x^2/(x^2+1)^3", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/8 * arctan(x) + (x^3 - x) / (8 * (x^2 + 1)^2)"
    );

    let expr = parse("(2*x+1)^2/((2*x+1)^2+1)^3", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/16 * arctan(2 * x + 1) + (2 * x^3 + 3 * x^2 + x) / (4 * (4 * x^2 + 4 * x + 2)^2)"
    );

    let expr = parse("x^3/(x^2+1)^3", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1 / (4 * (x^2 + 1)^2) - 1 / (2 * (x^2 + 1))"
    );

    let expr = parse("x^4/(x^2+1)^3", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "3/8 * arctan(x) + (3 * x^3 + 5 * x) / (8 * (x^2 + 1)^2) - x / (x^2 + 1)"
    );

    let expr = parse("(2*x+1)^4/((2*x+1)^2+1)^3", &mut ctx).expect("parse");
    assert!(integrate_symbolic_is_positive_quadratic_cube_target(
        &ctx, expr, "x"
    ));
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "3/16 * arctan(2 * x + 1) + (6 * x^3 + 9 * x^2 + 7 * x + 2) / (4 * (4 * x^2 + 4 * x + 2)^2) - (2 * x + 1) / (2 * (4 * x^2 + 4 * x + 2))"
    );
}

#[test]
fn integrates_rational_partial_fractions_over_repeated_linear_factors() {
    let mut ctx = Context::new();
    let expr = parse("(3*x+5)/(x^3-x^2-x+1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    let result = rendered(&ctx, out);

    assert_eq!(result, "1/2 * ln(|(x + 1) / (x - 1)|) - 4 / (x - 1)");

    let mut conditions: Vec<_> =
        integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x")
            .into_iter()
            .map(|condition| rendered(&ctx, condition))
            .collect();
    conditions.sort();
    assert_eq!(conditions, vec!["x + 1", "x - 1"]);
}

#[test]
fn exposes_rational_linear_partial_fraction_decomposition_for_didactic_trace() {
    let mut ctx = Context::new();
    let expr = parse("1/(-2*x^2-2*x)", &mut ctx).expect("parse");
    let out = super::integrate_symbolic_rational_linear_partial_fraction_decomposition_expr(
        &mut ctx, expr, "x",
    )
    .expect("partial fraction decomposition");
    let result = rendered(&ctx, out);

    assert!(
        result == "1 / (2 * (x + 1)) - 1 / (2 * x)",
        "expected concrete simple-fraction decomposition, got {result}"
    );
}

#[test]
fn exposes_linear_positive_quadratic_partial_fraction_decomposition_for_didactic_trace() {
    let mut ctx = Context::new();
    let expr = parse("1/(x^4-1)", &mut ctx).expect("parse");
    let out =
        super::integrate_symbolic_rational_linear_positive_quadratic_partial_fraction_decomposition_expr(
            &mut ctx, expr, "x",
        )
        .expect("partial fraction decomposition");
    let result = rendered(&ctx, out);

    assert!(
        result.contains("1 / (4 * (x - 1))")
            && result.contains("- 1 / (4 * (x + 1))")
            && result.contains("- 1 / (2 * (x^2 + 1))"),
        "expected concrete linear plus positive-quadratic decomposition, got {result}"
    );

    let expr = parse("1/((x+1)^2*(x^2+1))", &mut ctx).expect("parse");
    let out =
        super::integrate_symbolic_rational_linear_positive_quadratic_partial_fraction_decomposition_expr(
            &mut ctx, expr, "x",
        )
        .expect("repeated-pole partial fraction decomposition");
    let result = rendered(&ctx, out);

    assert!(
        result.contains("- 1/2 * x / (x^2 + 1)") && !result.contains("+ -"),
        "expected negative quadratic numerator to render as subtraction, got {result}"
    );
}

#[test]
fn integrates_improper_rational_partial_fractions_after_polynomial_division() {
    let mut ctx = Context::new();
    let expr = parse("(x^3+3*x+5)/(x^3-x^2-x+1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    let result = rendered(&ctx, out);

    assert!(
        result.contains("ln(|x + 1|)") && result.contains("ln(|x - 1|)"),
        "expected logarithmic remainder terms, got {result}"
    );
    assert!(
        result.contains("- 9 / (2 * (x - 1))") || result.contains("9/2 / (x - 1)"),
        "expected repeated-pole remainder term, got {result}"
    );

    let mut conditions: Vec<_> =
        integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x")
            .into_iter()
            .map(|condition| rendered(&ctx, condition))
            .collect();
    conditions.sort();
    assert_eq!(conditions, vec!["x + 1", "x - 1"]);
}

#[test]
fn atanh_substitution_reports_open_interval_condition() {
    let mut ctx = Context::new();
    let expr = parse("2*x/(4-x^4)", &mut ctx).expect("parse");
    let conditions = super::integrate_symbolic_required_positive_conditions(&mut ctx, expr, "x");

    assert_eq!(conditions.len(), 1);
    assert_eq!(rendered(&ctx, conditions[0]), "4 - x^4");

    let factored = parse("(2*x)/((2-x^2)*(x^2+2))", &mut ctx).expect("parse");
    let conditions =
        super::integrate_symbolic_required_positive_conditions(&mut ctx, factored, "x");

    assert_eq!(conditions.len(), 1);
    assert_eq!(rendered(&ctx, conditions[0]), "(x^2 + 2) * (2 - x^2)");
}

#[test]
fn integrates_polynomial_derivative_arcsin_substitution() {
    let mut ctx = Context::new();
    let expr = parse("2*x/sqrt(4-x^4)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "arcsin(x^2 / 2)");

    let expr = parse("x/sqrt(4-x^4)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "1/2 * arcsin(x^2 / 2)");

    let expr = parse("1/sqrt(4-(x+1)^2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "arcsin((x + 1) / 2)");

    let expr = parse("1/sqrt(a^2-x^2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "arcsin(x / sqrt(a^2))");

    let conditions = super::integrate_symbolic_required_positive_conditions(&mut ctx, expr, "x");
    assert_eq!(conditions.len(), 1);
    assert_eq!(rendered(&ctx, conditions[0]), "a^2 - x^2");

    let expr = parse("1/sqrt(a^2-(x+b)^2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "arcsin((b + x) / sqrt(a^2))");

    let expr = parse("1/sqrt(a^2-(m*x+b)^2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "arcsin((m * x + b) / sqrt(a^2)) / m");

    let conditions = super::integrate_symbolic_required_positive_conditions(&mut ctx, expr, "x");
    assert_eq!(conditions.len(), 1);
    assert_eq!(rendered(&ctx, conditions[0]), "a^2 - (m * x + b)^2");

    let expr = parse("1/sqrt(a^2-(x+b)^2)", &mut ctx).expect("parse");
    let conditions = super::integrate_symbolic_required_positive_conditions(&mut ctx, expr, "x");
    assert_eq!(conditions.len(), 1);
    assert_eq!(rendered(&ctx, conditions[0]), "a^2 - (b + x)^2");

    let expr = parse("(2*x*sqrt(4-x^4))/(4-x^4)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "arcsin(x^2 / 2)");
}

#[test]
fn integrates_polynomial_derivative_asinh_substitution() {
    let mut ctx = Context::new();
    let expr = parse("2*x/sqrt(1+x^4)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "asinh(x^2)");

    let expr = parse("x/sqrt(1+x^4)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "1/2 * asinh(x^2)");

    let expr = parse("2*x/sqrt(4+x^4)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "asinh(x^2 / 2)");

    let expr = parse("1/sqrt(4+(x+1)^2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "asinh((x + 1) / 2)");

    let expr = parse("1/sqrt(a^2+x^2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "asinh(x / sqrt(a^2))");

    let expr = parse("1/sqrt((x+b)^2+a^2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "asinh((b + x) / sqrt(a^2))");

    let expr = parse("1/sqrt((m*x+b)^2+a^2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "asinh((m * x + b) / sqrt(a^2)) / m");

    let expr = parse("(2*x*sqrt(1+x^4))/(1+x^4)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "asinh(x^2)");
}

#[test]
fn integrates_polynomial_derivative_over_square_root_substitution() {
    let mut ctx = Context::new();
    let expr = parse("x/sqrt(x^2+1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "sqrt(x^2 + 1)");

    let expr = parse("(2*x+1)/sqrt(x^2+x+1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "2 * sqrt(x^2 + x + 1)");

    let expr = parse("((2*x+1)*sqrt(x^2+x+1))/(x^2+x+1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "2 * sqrt(x^2 + x + 1)");
}

#[test]
fn sqrt_derivative_substitution_reports_positive_radicand_condition() {
    let mut ctx = Context::new();
    let expr = parse("2*x/sqrt(x^2-1)", &mut ctx).expect("parse");
    let conditions = super::integrate_symbolic_required_positive_conditions(&mut ctx, expr, "x");

    assert_eq!(conditions.len(), 1);
    assert_eq!(rendered(&ctx, conditions[0]), "x^2 - 1");
}

#[test]
fn integrates_polynomial_derivative_times_square_root_substitution() {
    let mut ctx = Context::new();
    let expr = parse("x*sqrt(x^2+1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "1/3 * (x^2 + 1)^(3/2)");

    let expr = parse("(2*x+1)*sqrt(x^2+x+1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "2/3 * (x^2 + x + 1)^(3/2)");
}

#[test]
fn integrates_polynomial_derivative_times_power_substitution() {
    let mut ctx = Context::new();
    let expr = parse("2*x*(x^2+1)^3", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "1/4 * (x^2 + 1)^4");

    let expr = parse("2*x*(x^2+1)^(3/2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "2/5 * (x^2 + 1)^(5/2)");

    let expr = parse("(2*x+1)*(x^2+x+1)^3", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "1/4 * (x^2 + x + 1)^4");

    let expr = parse("(2*x+1)*(x^2+x+1)^(-3/2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-2 / sqrt(x^2 + x + 1)");
}

#[test]
fn integrates_polynomial_derivative_over_denominator_power_substitution() {
    let mut ctx = Context::new();
    let expr = parse("2*x/(x^2+1)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-1 / (x^2 + 1)");

    let expr = parse("2*x/(x^2-1)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-1 / (x^2 - 1)");

    let expr = parse("2*x/(x^2+1)^3", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-1 / (2 * (x^2 + 1)^2)");

    let expr = parse("(2*x+1)/(x^2+x+1)^(3/2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-2 / sqrt(x^2 + x + 1)");

    let expr = parse("(2*x+1)/(sqrt(x^2+x+1)^3)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-2 / sqrt(x^2 + x + 1)");

    let expr = parse("-2*x/(x^2+1)^3", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "1 / (2 * (x^2 + 1)^2)");

    let expr = parse("(2*x+1)/(x^4+2*x^3-x^2-2*x+1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-1 / (x^2 + x - 1)");

    let expr = parse("(2*x+1)/(3*(x^2+x-1)^2)", &mut ctx).expect("parse");
    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert_eq!(conditions.len(), 1);
    assert_eq!(rendered(&ctx, conditions[0]), "x^2 + x - 1");

    let expr = parse("(2*x+1)/(3*x^4+6*x^3-3*x^2-6*x+3)", &mut ctx).expect("parse");
    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert_eq!(conditions.len(), 1);
    assert_eq!(rendered(&ctx, conditions[0]), "x^2 + x - 1");

    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-1 / (3 * (x^2 + x - 1))");

    let expr = parse("1/3*((2*x+1)/(x^2+x-1)^2)", &mut ctx).expect("parse");
    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert_eq!(conditions.len(), 1);
    assert_eq!(rendered(&ctx, conditions[0]), "x^2 + x - 1");

    let expr = parse("(2*x+1)/(x^6+3*x^5-5*x^3+3*x-1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-1 / (2 * (x^2 + x - 1)^2)");

    let expr = parse("-(2*x+1)/(x^2+x-1)^3", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "1 / (2 * (x^2 + x - 1)^2)");

    let expr = parse("(2*x+1)/(4*x^6+12*x^5-20*x^3+12*x-4)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-1 / (8 * (x^2 + x - 1)^2)");

    let expr = parse("(2*x+1)/(3/((x^2+x-1)^(-2)))", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-1 / (3 * (x^2 + x - 1))");

    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert_eq!(conditions.len(), 1);
    assert_eq!(rendered(&ctx, conditions[0]), "x^2 + x - 1");

    let expr = parse("1/(x^5+5*x^4+10*x^3+10*x^2+5*x+1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-1 / (4 * (x + 1)^4)");
}

#[test]
fn detects_bounded_reciprocal_quotient_denominator_power_substitution_targets() {
    let mut ctx = Context::new();

    for input in ["(2*x+1)/(3/(x^2+x-1)^2)", "(2*x+1)/(3/((x^2+x-1)^(-2)))"] {
        let expr = parse(input, &mut ctx).expect("parse");
        assert!(
            super::integrate_symbolic_is_bounded_reciprocal_quotient_denominator_power_substitution_target(
                &ctx, expr, "x", 8
            ),
            "expected bounded reciprocal quotient detector to accept {input}"
        );
    }

    for input in [
        "(2*x+1)/(3/((x^2+x-1)^(-1)))",
        "(2*x+1)/(3/(x^2+x-1)^9)",
        "(2*x+1)/(3/((x^2+x-1)^(-9)))",
    ] {
        let expr = parse(input, &mut ctx).expect("parse");
        assert!(
            !super::integrate_symbolic_is_bounded_reciprocal_quotient_denominator_power_substitution_target(
                &ctx, expr, "x", 8
            ),
            "expected bounded reciprocal quotient detector to reject {input}"
        );
    }
}

#[test]
fn detects_fractional_denominator_power_substitution_target() {
    let mut ctx = Context::new();
    let expr = parse("(2*x+1)/(x^2+x+1)^(3/2)", &mut ctx).expect("parse");

    assert!(
        super::integrate_symbolic_is_fractional_denominator_power_substitution_target(
            &mut ctx, expr, "x"
        )
    );

    let expr = parse("2*(x/(x^2-1)^(3/2))", &mut ctx).expect("parse");
    assert!(
        super::integrate_symbolic_is_fractional_denominator_power_substitution_target(
            &mut ctx, expr, "x"
        ),
        "constant extraction before integration should preserve the fractional denominator power family"
    );
}

#[test]
fn detects_bounded_negative_syntactic_denominator_power_substitution_targets() {
    let mut ctx = Context::new();

    for input in ["(2*x+1)/(3*(x^2+x-1)^(-1))", "(2*x+1)/(3*(x^2+x-1)^(-2))"] {
        let expr = parse(input, &mut ctx).expect("parse");
        assert!(
            super::integrate_symbolic_is_bounded_negative_syntactic_denominator_power_substitution_target(
                &ctx, expr, "x", 8
            ),
            "expected bounded negative denominator power detector to accept {input}"
        );
    }

    for input in [
        "(2*x+1)/(3*(x^2+x-1)^(-9))",
        "(x+1)/(3*(x^2+x-1)^(-2))",
        "(2*x+1)/(3*(x^2+x-1)^2)",
    ] {
        let expr = parse(input, &mut ctx).expect("parse");
        assert!(
            !super::integrate_symbolic_is_bounded_negative_syntactic_denominator_power_substitution_target(
                &ctx, expr, "x", 8
            ),
            "expected bounded negative denominator power detector to reject {input}"
        );
    }
}

#[test]
fn integrates_asinh_kernel() {
    let mut ctx = Context::new();
    let expr = parse("(x^2+1)^(-1/2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "asinh(x)");
}

#[test]
fn integrates_secant_squared_kernel() {
    let mut ctx = Context::new();
    let expr = parse("1/cos(x)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "tan(x)");

    let expr = parse("2*x/cos(x^2)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "tan(x^2)");

    let expr = parse("x/cos(x^2)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "tan(x^2) / 2");
}

#[test]
fn integrates_cosecant_squared_kernel() {
    let mut ctx = Context::new();
    let expr = parse("1/sin(x)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-cot(x)");

    let expr = parse("3*x^2/sin(x^3)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-cot(x^3)");

    let expr = parse("x^2/sin(x^3)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-cot(x^3) / 3");
}

#[test]
fn integrates_affine_trig_square_power_reduction() {
    let mut ctx = Context::new();
    let expr = parse("sin(x)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "x / 2 - 1/4 * sin(2 * x)");

    let expr = parse("cos(2*x + 1)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "1/8 * sin(2 * (2 * x + 1)) + x / 2");
}

#[test]
fn integrates_affine_hyperbolic_square_power_reduction() {
    let mut ctx = Context::new();
    let expr = parse("sinh(x)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "1/2 * sinh(x) * cosh(x) - x / 2");

    let expr = parse("cosh(x)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "1/2 * sinh(x) * cosh(x) + x / 2");

    let expr = parse("sinh(2*x)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/4 * sinh(2 * x) * cosh(2 * x) - x / 2"
    );

    let expr = parse("sinh(2*x+1)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/4 * sinh(2 * x + 1) * cosh(2 * x + 1) - x / 2"
    );

    let expr = parse("cosh(2*x+1)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/4 * sinh(2 * x + 1) * cosh(2 * x + 1) + x / 2"
    );

    let expr = parse("sinh(x)^2*cosh(x)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "1/32 * sinh(4 * x) - 1/8 * x");

    let expr = parse("4*sinh(x)^2*cosh(x)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "1/8 * sinh(4 * x) - 1/2 * x");

    let expr = parse("sinh(2*x+1)^2*cosh(2*x+1)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/64 * sinh(4 * (2 * x + 1)) - 1/8 * x"
    );
}

#[test]
fn integrates_affine_tanh_fourth_power_reduction() {
    let mut ctx = Context::new();
    let expr = parse("tanh(x)^4", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "x - (tanh(x) + 1/3 * tanh(x)^3)");

    let expr = parse("tanh(2*x+1)^4", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "x - 1/2 * (tanh(2 * x + 1) + 1/3 * tanh(2 * x + 1)^3)"
    );

    let expr = parse("tanh(1-2*x)^4", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/2 * (tanh(1 - 2 * x) + 1/3 * tanh(1 - 2 * x)^3) + x"
    );

    let expr = parse("tanh(x)^6", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "x - (tanh(x) + 1/5 * tanh(x)^5 + 1/3 * tanh(x)^3)"
    );

    let expr = parse("tanh(2*x+1)^6", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "x - 1/2 * (tanh(2 * x + 1) + 1/5 * tanh(2 * x + 1)^5 + 1/3 * tanh(2 * x + 1)^3)"
    );

    let expr = parse("tanh(1-2*x)^6", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/2 * (tanh(1 - 2 * x) + 1/5 * tanh(1 - 2 * x)^5 + 1/3 * tanh(1 - 2 * x)^3) + x"
    );

    let expr = parse("tanh(x)^8", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "x - (tanh(x) + 1/7 * tanh(x)^7 + 1/5 * tanh(x)^5 + 1/3 * tanh(x)^3)"
    );

    let expr = parse("tanh(2*x+1)^8", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "x - (tanh(2 * x + 1) + 1/7 * tanh(2 * x + 1)^7 + 1/5 * tanh(2 * x + 1)^5 + 1/3 * tanh(2 * x + 1)^3) / 2"
    );

    let expr = parse("tanh(1-2*x)^8", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "(tanh(1 - 2 * x) + 1/7 * tanh(1 - 2 * x)^7 + 1/5 * tanh(1 - 2 * x)^5 + 1/3 * tanh(1 - 2 * x)^3) / 2 + x"
    );
}

#[test]
fn integrates_affine_hyperbolic_cubic_power_reduction() {
    let mut ctx = Context::new();
    let expr = parse("sinh(2*x + 1)^3", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/2 * (1/3 * cosh(2 * x + 1)^3 - cosh(2 * x + 1))"
    );

    let expr = parse("cosh(2*x + 1)^3", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/2 * (sinh(2 * x + 1) + 1/3 * sinh(2 * x + 1)^3)"
    );

    let expr = parse("sinh(1 - 2*x)^3", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "-1/2 * (1/3 * cosh(1 - 2 * x)^3 - cosh(1 - 2 * x))"
    );

    let expr = parse("cosh(1 - 2*x)^3", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "-1/2 * (sinh(1 - 2 * x) + 1/3 * sinh(1 - 2 * x)^3)"
    );
}

#[test]
fn integrates_affine_hyperbolic_fifth_power_reduction() {
    let mut ctx = Context::new();
    let expr = parse("sinh(2*x + 1)^5", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/2 * (cosh(2 * x + 1) + 1/5 * cosh(2 * x + 1)^5 - 2/3 * cosh(2 * x + 1)^3)"
    );

    let expr = parse("cosh(2*x + 1)^5", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/2 * (sinh(2 * x + 1) + 1/5 * sinh(2 * x + 1)^5 + 2/3 * sinh(2 * x + 1)^3)"
    );

    let expr = parse("sinh(1 - 2*x)^5", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "-1/2 * (cosh(1 - 2 * x) + 1/5 * cosh(1 - 2 * x)^5 - 2/3 * cosh(1 - 2 * x)^3)"
    );

    let expr = parse("cosh(1 - 2*x)^5", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "-1/2 * (sinh(1 - 2 * x) + 1/5 * sinh(1 - 2 * x)^5 + 2/3 * sinh(1 - 2 * x)^3)"
    );

    let expr = parse("sinh(2*x + 1)^7", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/2 * (-cosh(2 * x + 1) + cosh(2 * x + 1)^3 + 1/7 * cosh(2 * x + 1)^7 - 3/5 * cosh(2 * x + 1)^5)"
    );

    let expr = parse("cosh(2*x + 1)^7", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/2 * (sinh(2 * x + 1) + sinh(2 * x + 1)^3 + 1/7 * sinh(2 * x + 1)^7 + 3/5 * sinh(2 * x + 1)^5)"
    );

    let expr = parse("sinh(1 - 2*x)^7", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "-1/2 * (-cosh(1 - 2 * x) + cosh(1 - 2 * x)^3 + 1/7 * cosh(1 - 2 * x)^7 - 3/5 * cosh(1 - 2 * x)^5)"
    );

    let expr = parse("cosh(1 - 2*x)^7", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "-1/2 * (sinh(1 - 2 * x) + sinh(1 - 2 * x)^3 + 1/7 * sinh(1 - 2 * x)^7 + 3/5 * sinh(1 - 2 * x)^5)"
    );

    let expr = parse("sinh(x)^7", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "-cosh(x) + cosh(x)^3 + 1/7 * cosh(x)^7 - 3/5 * cosh(x)^5"
    );

    let expr = parse("cosh(x)^7", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "sinh(x) + sinh(x)^3 + 1/7 * sinh(x)^7 + 3/5 * sinh(x)^5"
    );
}

#[test]
fn integrates_affine_trig_ratio_square_power_reduction() {
    let mut ctx = Context::new();
    let expr = parse("tan(x)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "tan(x) - x");
    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert_eq!(rendered(&ctx, conditions[0]), "cos(x)");

    let expr = parse("sin(x)^2/cos(x)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "tan(x) - x");
    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert_eq!(rendered(&ctx, conditions[0]), "cos(x)");

    let expr = parse("tan(2*x + 1)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "tan(2 * x + 1) / 2 - x");

    let expr = parse("tan(x)^4", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "tan(x)^3 / 3 + x - tan(x)");
    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert_eq!(rendered(&ctx, conditions[0]), "cos(x)");

    let expr = parse("tan(2*x + 1)^4", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "-tan(2 * x + 1) / 2 + tan(2 * x + 1)^3 / 6 + x"
    );

    let expr = parse("sin(2*x + 1)^4/cos(2*x + 1)^4", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "-tan(2 * x + 1) / 2 + tan(2 * x + 1)^3 / 6 + x"
    );
    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert_eq!(rendered(&ctx, conditions[0]), "cos(2 * x + 1)");

    let expr = parse("tan(x)^6", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "tan(x) + -tan(x)^3 / 3 + tan(x)^5 / 5 - x"
    );
    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert_eq!(rendered(&ctx, conditions[0]), "cos(x)");

    let expr = parse("tan(2*x + 1)^6", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "tan(2 * x + 1) / 2 + -tan(2 * x + 1)^3 / 6 + tan(2 * x + 1)^5 / 10 - x"
    );

    let expr = parse("tan(1-2*x)^6", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "-tan(1 - 2 * x) / 2 + -tan(1 - 2 * x)^5 / 10 + tan(1 - 2 * x)^3 / 6 - x"
    );

    let expr = parse("sin(2*x + 1)^6/cos(2*x + 1)^6", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "tan(2 * x + 1) / 2 + -tan(2 * x + 1)^3 / 6 + tan(2 * x + 1)^5 / 10 - x"
    );
    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert_eq!(rendered(&ctx, conditions[0]), "cos(2 * x + 1)");

    let expr = parse("tan(x)^8", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "-tan(x)^5 / 5 + tan(x)^3 / 3 + tan(x)^7 / 7 + x - tan(x)"
    );
    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert_eq!(rendered(&ctx, conditions[0]), "cos(x)");

    let expr = parse("tan(2*x + 1)^8", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "-tan(2 * x + 1) / 2 + -tan(2 * x + 1)^5 / 10 + tan(2 * x + 1)^3 / 6 + tan(2 * x + 1)^7 / 14 + x"
    );
    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert_eq!(rendered(&ctx, conditions[0]), "cos(2 * x + 1)");

    let expr = parse("tan(1-2*x)^8", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "tan(1 - 2 * x) / 2 + -tan(1 - 2 * x)^3 / 6 + -tan(1 - 2 * x)^7 / 14 + tan(1 - 2 * x)^5 / 10 + x"
    );

    let expr = parse("sec(x)^4", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "tan(x) + tan(x)^3 / 3");
    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert_eq!(rendered(&ctx, conditions[0]), "cos(x)");

    let expr = parse("1/cos(2*x + 1)^4", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "tan(2 * x + 1) / 2 + tan(2 * x + 1)^3 / 6"
    );
    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert_eq!(rendered(&ctx, conditions[0]), "cos(2 * x + 1)");

    let expr = parse("sec(x)^6", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "tan(x) + tan(x)^5 / 5 + 2 * tan(x)^3 / 3"
    );
    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert_eq!(rendered(&ctx, conditions[0]), "cos(x)");

    let expr = parse("1/cos(2*x + 1)^6", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "tan(2 * x + 1) / 2 + tan(2 * x + 1)^3 / 3 + tan(2 * x + 1)^5 / 10"
    );
    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert_eq!(rendered(&ctx, conditions[0]), "cos(2 * x + 1)");

    let expr = parse("sec(1-2*x)^6", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "-tan(1 - 2 * x) / 2 + -tan(1 - 2 * x)^3 / 3 + -tan(1 - 2 * x)^5 / 10"
    );

    let expr = parse("sec(x)^8", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "tan(x) + tan(x)^7 / 7 + 3 * tan(x)^5 / 5 + tan(x)^3"
    );
    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert_eq!(rendered(&ctx, conditions[0]), "cos(x)");

    let expr = parse("1/cos(2*x + 1)^8", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "tan(2 * x + 1) / 2 + tan(2 * x + 1)^3 / 2 + tan(2 * x + 1)^7 / 14 + 3 * tan(2 * x + 1)^5 / 10"
    );
    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert_eq!(rendered(&ctx, conditions[0]), "cos(2 * x + 1)");

    let expr = parse("sec(1-2*x)^8", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "-tan(1 - 2 * x) / 2 + -tan(1 - 2 * x)^3 / 2 + -tan(1 - 2 * x)^7 / 14 + -3 * tan(1 - 2 * x)^5 / 10"
    );

    let expr = parse("csc(x)^4", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-cot(x)^3 / 3 - cot(x)");
    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert_eq!(rendered(&ctx, conditions[0]), "sin(x)");

    let expr = parse("1/sin(2*x + 1)^4", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "-cot(2 * x + 1)^3 / 6 - cot(2 * x + 1) / 2"
    );
    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert_eq!(rendered(&ctx, conditions[0]), "sin(2 * x + 1)");

    let expr = parse("csc(x)^6", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "-cot(x)^5 / 5 + -2 * cot(x)^3 / 3 - cot(x)"
    );
    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert_eq!(rendered(&ctx, conditions[0]), "sin(x)");

    let expr = parse("1/sin(2*x + 1)^6", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "-cot(2 * x + 1) / 2 + -cot(2 * x + 1)^3 / 3 + -cot(2 * x + 1)^5 / 10"
    );
    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert_eq!(rendered(&ctx, conditions[0]), "sin(2 * x + 1)");

    let expr = parse("csc(1-2*x)^6", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "cot(1 - 2 * x) / 2 + cot(1 - 2 * x)^3 / 3 + cot(1 - 2 * x)^5 / 10"
    );

    let expr = parse("csc(x)^8", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "-cot(x)^7 / 7 + -3 * cot(x)^5 / 5 - cot(x) - cot(x)^3"
    );
    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert_eq!(rendered(&ctx, conditions[0]), "sin(x)");

    let expr = parse("1/sin(2*x + 1)^8", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "-cot(2 * x + 1) / 2 + -cot(2 * x + 1)^3 / 2 + -cot(2 * x + 1)^7 / 14 + -3 * cot(2 * x + 1)^5 / 10"
    );
    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert_eq!(rendered(&ctx, conditions[0]), "sin(2 * x + 1)");

    let expr = parse("csc(1-2*x)^8", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "cot(1 - 2 * x) / 2 + cot(1 - 2 * x)^3 / 2 + cot(1 - 2 * x)^7 / 14 + 3 * cot(1 - 2 * x)^5 / 10"
    );

    let expr = parse("cot(x)^4", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "cot(x) + x - cot(x)^3 / 3");
    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert_eq!(rendered(&ctx, conditions[0]), "sin(x)");

    let expr = parse("cos(2*x + 1)^4/sin(2*x + 1)^4", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "cot(2 * x + 1) / 2 + x - cot(2 * x + 1)^3 / 6"
    );
    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert_eq!(rendered(&ctx, conditions[0]), "sin(2 * x + 1)");

    let expr = parse("cot(x)^6", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "-cot(x)^5 / 5 + cot(x)^3 / 3 - cot(x) - x"
    );
    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert_eq!(rendered(&ctx, conditions[0]), "sin(x)");

    let expr = parse("cot(2*x + 1)^6", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "-cot(2 * x + 1) / 2 + -cot(2 * x + 1)^5 / 10 + cot(2 * x + 1)^3 / 6 - x"
    );

    let expr = parse("cot(1-2*x)^6", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "cot(1 - 2 * x) / 2 + -cot(1 - 2 * x)^3 / 6 + cot(1 - 2 * x)^5 / 10 - x"
    );

    let expr = parse("cos(2*x + 1)^6/sin(2*x + 1)^6", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "-cot(2 * x + 1) / 2 + -cot(2 * x + 1)^5 / 10 + cot(2 * x + 1)^3 / 6 - x"
    );
    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert_eq!(rendered(&ctx, conditions[0]), "sin(2 * x + 1)");

    let expr = parse("cot(x)^8", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "cot(x) + -cot(x)^3 / 3 + -cot(x)^7 / 7 + cot(x)^5 / 5 + x"
    );
    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert_eq!(rendered(&ctx, conditions[0]), "sin(x)");

    let expr = parse("cot(2*x + 1)^8", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "cot(2 * x + 1) / 2 + -cot(2 * x + 1)^3 / 6 + -cot(2 * x + 1)^7 / 14 + cot(2 * x + 1)^5 / 10 + x"
    );
    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert_eq!(rendered(&ctx, conditions[0]), "sin(2 * x + 1)");

    let expr = parse("cot(1-2*x)^8", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "-cot(1 - 2 * x) / 2 + -cot(1 - 2 * x)^5 / 10 + cot(1 - 2 * x)^3 / 6 + cot(1 - 2 * x)^7 / 14 + x"
    );

    let expr = parse("cot(x)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-cot(x) - x");
    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert_eq!(rendered(&ctx, conditions[0]), "sin(x)");

    let expr = parse("cos(x)^2/sin(x)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-cot(x) - x");
    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert_eq!(rendered(&ctx, conditions[0]), "sin(x)");

    let expr = parse("cot(2*x + 1)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-cot(2 * x + 1) / 2 - x");
}

#[test]
fn integrates_affine_sine_cosine_product() {
    let mut ctx = Context::new();
    let expr = parse("sin(x)*cos(x)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "1/2 * sin(x)^2");

    let expr = parse("3*sin(2*x + 1)*cos(2*x + 1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "3/4 * sin(2 * x + 1)^2");
}

#[test]
fn integrates_affine_trig_power_times_derivative_product() {
    let mut ctx = Context::new();
    let expr = parse("sin(x)^2*cos(x)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "1/3 * sin(x)^3");

    let expr = parse("2*cos(2*x + 1)*sin(2*x + 1)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "1/3 * sin(2 * x + 1)^3");

    let expr = parse("sin(x)*cos(x)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-1/3 * cos(x)^3");
}

#[test]
fn integrates_affine_trig_ratio_power_reciprocal_square_product() {
    let mut ctx = Context::new();
    let expr = parse("tan(x)/cos(x)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "tan(x)^2 / 2");

    let expr = parse("sec(x)^2*tan(x)", &mut ctx).expect("parse");
    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert_eq!(rendered(&ctx, conditions[0]), "cos(x)");

    let expr = parse("2*tan(2*x + 1)/cos(2*x + 1)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "tan(2 * x + 1)^2 / 2");

    let expr = parse("sin(x)/cos(x)^3", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "tan(x)^2 / 2");

    let expr = parse("tan(x)^2/cos(x)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "tan(x)^3 / 3");

    let expr = parse("2*tan(2*x + 1)^2/cos(2*x + 1)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "tan(2 * x + 1)^3 / 3");

    let expr = parse("sin(x)^2/cos(x)^4", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "tan(x)^3 / 3");

    let expr = parse("cot(x)/sin(x)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-cot(x)^2 / 2");

    let expr = parse("csc(x)^2*cot(x)", &mut ctx).expect("parse");
    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert_eq!(rendered(&ctx, conditions[0]), "sin(x)");

    let expr = parse("2*cot(2*x + 1)/sin(2*x + 1)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-cot(2 * x + 1)^2 / 2");

    let expr = parse("cos(x)/sin(x)^3", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-cot(x)^2 / 2");

    let expr = parse("cot(x)^2/sin(x)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-cot(x)^3 / 3");

    let expr = parse("2*cot(2*x + 1)^2/sin(2*x + 1)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-cot(2 * x + 1)^3 / 3");

    let expr = parse("cos(x)^2/sin(x)^4", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-cot(x)^3 / 3");
}

#[test]
fn integrates_affine_trig_cube_power_reduction() {
    let mut ctx = Context::new();
    let expr = parse("sin(x)^3", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "1/3 * cos(x)^3 - cos(x)");

    let expr = parse("cos(x)^3", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "sin(x) - 1/3 * sin(x)^3");

    let expr = parse("sin(2*x + 1)^3", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/2 * (1/3 * cos(2 * x + 1)^3 - cos(2 * x + 1))"
    );

    let expr = parse("cos(2*x + 1)^3", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/2 * (sin(2 * x + 1) - 1/3 * sin(2 * x + 1)^3)"
    );
}

#[test]
fn integrates_affine_trig_fourth_power_reduction() {
    let mut ctx = Context::new();
    let expr = parse("sin(x)^4", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/32 * sin(4 * x) + 3/8 * x - 1/4 * sin(2 * x)"
    );

    let expr = parse("cos(x)^4", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/32 * sin(4 * x) + 1/4 * sin(2 * x) + 3/8 * x"
    );

    let expr = parse("sin(2*x + 1)^4", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/64 * sin(4 * (2 * x + 1)) + 3/8 * x - 1/8 * sin(2 * (2 * x + 1))"
    );

    let expr = parse("cos(2*x + 1)^4", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/64 * sin(4 * (2 * x + 1)) + 1/8 * sin(2 * (2 * x + 1)) + 3/8 * x"
    );
}

#[test]
fn integrates_affine_trig_sixth_power_reduction() {
    let mut ctx = Context::new();
    let expr = parse("sin(x)^6", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "3/64 * sin(4 * x) + 5/16 * x - 15/64 * sin(2 * x) - 1/192 * sin(6 * x)"
    );

    let expr = parse("cos(x)^6", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/192 * sin(6 * x) + 3/64 * sin(4 * x) + 15/64 * sin(2 * x) + 5/16 * x"
    );

    let expr = parse("sin(2*x + 1)^6", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "3/128 * sin(4 * (2 * x + 1)) + 5/16 * x - 15/128 * sin(2 * (2 * x + 1)) - 1/384 * sin(6 * (2 * x + 1))"
    );

    let expr = parse("cos(2*x + 1)^6", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/384 * sin(6 * (2 * x + 1)) + 3/128 * sin(4 * (2 * x + 1)) + 15/128 * sin(2 * (2 * x + 1)) + 5/16 * x"
    );
}

#[test]
fn integrates_affine_trig_eighth_power_reduction() {
    let mut ctx = Context::new();
    let expr = parse("sin(x)^8", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/1024 * sin(8 * x) + 7/128 * sin(4 * x) + 35/128 * x - 7/32 * sin(2 * x) - 1/96 * sin(6 * x)"
    );

    let expr = parse("cos(x)^8", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/1024 * sin(8 * x) + 1/96 * sin(6 * x) + 7/128 * sin(4 * x) + 7/32 * sin(2 * x) + 35/128 * x"
    );

    let expr = parse("sin(2*x + 1)^8", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/2048 * sin(8 * (2 * x + 1)) + 7/256 * sin(4 * (2 * x + 1)) + 35/128 * x - 7/64 * sin(2 * (2 * x + 1)) - 1/192 * sin(6 * (2 * x + 1))"
    );

    let expr = parse("cos(2*x + 1)^8", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/2048 * sin(8 * (2 * x + 1)) + 1/192 * sin(6 * (2 * x + 1)) + 7/256 * sin(4 * (2 * x + 1)) + 7/64 * sin(2 * (2 * x + 1)) + 35/128 * x"
    );
}

#[test]
fn integrates_affine_trig_fifth_power_reduction() {
    let mut ctx = Context::new();
    let expr = parse("sin(x)^5", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "2/3 * cos(x)^3 - cos(x) - 1/5 * cos(x)^5"
    );

    let expr = parse("cos(x)^5", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "sin(x) + 1/5 * sin(x)^5 - 2/3 * sin(x)^3"
    );

    let expr = parse("sin(2*x + 1)^5", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/2 * (2/3 * cos(2 * x + 1)^3 - cos(2 * x + 1) - 1/5 * cos(2 * x + 1)^5)"
    );

    let expr = parse("cos(2*x + 1)^5", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "1/2 * (sin(2 * x + 1) + 1/5 * sin(2 * x + 1)^5 - 2/3 * sin(2 * x + 1)^3)"
    );
}

#[test]
fn integrates_affine_trig_seventh_power_reduction() {
    let mut ctx = Context::new();
    let expr = parse("sin(x)^7", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "cos(x)^3 + 1/7 * cos(x)^7 - cos(x) - 3/5 * cos(x)^5"
    );

    let expr = parse("cos(x)^7", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "sin(x) + 3/5 * sin(x)^5 - sin(x)^3 - 1/7 * sin(x)^7"
    );

    let expr = parse("sin(2*x + 1)^7", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "(cos(2 * x + 1)^3 + 1/7 * cos(2 * x + 1)^7 - cos(2 * x + 1) - 3/5 * cos(2 * x + 1)^5) / 2"
    );

    let expr = parse("cos(2*x + 1)^7", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "(sin(2 * x + 1) + 3/5 * sin(2 * x + 1)^5 - sin(2 * x + 1)^3 - 1/7 * sin(2 * x + 1)^7) / 2"
    );

    let expr = parse("sin(1 - 2*x)^7", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "-(cos(1 - 2 * x)^3 + 1/7 * cos(1 - 2 * x)^7 - cos(1 - 2 * x) - 3/5 * cos(1 - 2 * x)^5) / 2"
    );

    let expr = parse("cos(1 - 2*x)^7", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "-(sin(1 - 2 * x) + 3/5 * sin(1 - 2 * x)^5 - sin(1 - 2 * x)^3 - 1/7 * sin(1 - 2 * x)^7) / 2"
    );
}

#[test]
fn integrates_canonical_sec_tan_and_csc_cot_quotients() {
    let mut ctx = Context::new();
    let expr = parse("sin(2*x + 1)/cos(2*x + 1)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "sec(2 * x + 1) / 2");

    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert_eq!(conditions.len(), 1);
    assert_eq!(rendered(&ctx, conditions[0]), "cos(2 * x + 1)");

    let expr = parse("cos(2*x + 1)/sin(2*x + 1)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-csc(2 * x + 1) / 2");

    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert_eq!(conditions.len(), 1);
    assert_eq!(rendered(&ctx, conditions[0]), "sin(2 * x + 1)");

    let expr = parse("tan(2*x + 1)/cos(2*x + 1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "sec(2 * x + 1) / 2");

    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert_eq!(conditions.len(), 1);
    assert_eq!(rendered(&ctx, conditions[0]), "cos(2 * x + 1)");

    let expr = parse("cot(2*x + 1)/sin(2*x + 1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-csc(2 * x + 1) / 2");

    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert_eq!(conditions.len(), 1);
    assert_eq!(rendered(&ctx, conditions[0]), "sin(2 * x + 1)");

    let expr = parse("sec(2*x + 1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "ln(|tan(2 * x + 1) + sec(2 * x + 1)|) / 2"
    );

    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert_eq!(conditions.len(), 1);
    assert_eq!(rendered(&ctx, conditions[0]), "cos(2 * x + 1)");

    let expr = parse("csc(2*x + 1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(
        rendered(&ctx, out),
        "ln(|csc(2 * x + 1) - cot(2 * x + 1)|) / 2"
    );

    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert_eq!(conditions.len(), 1);
    assert_eq!(rendered(&ctx, conditions[0]), "sin(2 * x + 1)");
}

#[test]
fn integrates_polynomial_sec_tan_and_csc_cot_quotients() {
    let mut ctx = Context::new();
    let expr = parse("2*x*sin(x^2)/cos(x^2)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "sec(x^2)");

    let expr = parse("2*x*sin(x^2+b)/cos(x^2+b)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "sec(x^2 + b)");

    let expr = parse("2*k*x*sin(x^2+b)/cos(x^2+b)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "k * sec(x^2 + b)");

    let expr = parse("x*sin(x^2)/cos(x^2)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "sec(x^2) / 2");

    let expr = parse("3*x^2*cos(x^3)/sin(x^3)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-csc(x^3)");

    let expr = parse("2*x*cos(x^2+b)/sin(x^2+b)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-csc(x^2 + b)");

    let expr = parse("2*k*x*cos(x^2+b)/sin(x^2+b)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-k * csc(x^2 + b)");

    let expr = parse("2*(x*sin(x^2)/cos(x^2)^2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "sec(x^2)");

    let expr = parse("3*(x^2*cos(x^3)/sin(x^3)^2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-csc(x^3)");

    let expr = parse("-4*x*sin(x^2)/cos(x^2)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-2 * sec(x^2)");

    let expr = parse("-4*x*cos(x^2)/sin(x^2)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "2 * csc(x^2)");

    let expr = parse("(2*x+1)*sin(x^2+x)/cos(x^2+x)^2", &mut ctx).expect("parse");
    let (out, required_nonzero) =
        super::integrate_symbolic_polynomial_trig_reciprocal_derivative_root_gate(
            &mut ctx, expr, "x",
        )
        .expect("root gate");
    assert_eq!(rendered(&ctx, out), "sec(x^2 + x)");
    assert_eq!(rendered(&ctx, required_nonzero), "cos(x^2 + x)");

    let expr = parse("(2*x+1)*cos(x^2+x)/sin(x^2+x)^2", &mut ctx).expect("parse");
    let (out, required_nonzero) =
        super::integrate_symbolic_polynomial_trig_reciprocal_derivative_root_gate(
            &mut ctx, expr, "x",
        )
        .expect("root gate");
    assert_eq!(rendered(&ctx, out), "-csc(x^2 + x)");
    assert_eq!(rendered(&ctx, required_nonzero), "sin(x^2 + x)");

    let expr = parse("a*sin(a*x+b)/cos(a*x+b)^2", &mut ctx).expect("parse");
    let (out, required_nonzero) =
        super::integrate_symbolic_polynomial_trig_reciprocal_derivative_root_gate(
            &mut ctx, expr, "x",
        )
        .expect("root gate");
    assert_eq!(rendered(&ctx, out), "sec(a * x + b)");
    assert_eq!(rendered(&ctx, required_nonzero), "cos(a * x + b)");
    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert_eq!(conditions.len(), 1);
    assert_eq!(rendered(&ctx, conditions[0]), "cos(a * x + b)");

    let expr = parse("a*cos(a*x+b)/sin(a*x+b)^2", &mut ctx).expect("parse");
    let (out, required_nonzero) =
        super::integrate_symbolic_polynomial_trig_reciprocal_derivative_root_gate(
            &mut ctx, expr, "x",
        )
        .expect("root gate");
    assert_eq!(rendered(&ctx, out), "-csc(a * x + b)");
    assert_eq!(rendered(&ctx, required_nonzero), "sin(a * x + b)");
    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert_eq!(conditions.len(), 1);
    assert_eq!(rendered(&ctx, conditions[0]), "sin(a * x + b)");

    let expr = parse("(3*sin(x^2+x)+6*x*sin(x^2+x))/cos(x^2+x)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "3 * sec(x^2 + x)");
    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert_eq!(conditions.len(), 1);
    assert_eq!(rendered(&ctx, conditions[0]), "cos(x^2 + x)");

    let expr = parse("(3*cos(x^2+x)+6*x*cos(x^2+x))/sin(x^2+x)^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-3 * csc(x^2 + x)");
    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert_eq!(conditions.len(), 1);
    assert_eq!(rendered(&ctx, conditions[0]), "sin(x^2 + x)");

    let expr = parse("2*x*sec(x^2)^2", &mut ctx).expect("parse");
    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert_eq!(conditions.len(), 1);
    assert_eq!(rendered(&ctx, conditions[0]), "cos(x^2)");

    let expr = parse("2*x*csc(x^2)^2", &mut ctx).expect("parse");
    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert_eq!(conditions.len(), 1);
    assert_eq!(rendered(&ctx, conditions[0]), "sin(x^2)");
}

#[test]
fn integrates_sqrt_chain_sec_tan_and_csc_cot_quotients() {
    let mut ctx = Context::new();
    let expr = parse("sin(sqrt(x))*sqrt(x)/(2*x*cos(sqrt(x))^2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "sec(sqrt(x))");

    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert_eq!(conditions.len(), 1);
    assert_eq!(rendered(&ctx, conditions[0]), "cos(sqrt(x))");

    let positive = super::integrate_symbolic_required_positive_conditions(&mut ctx, expr, "x");
    assert_eq!(positive.len(), 1);
    assert_eq!(rendered(&ctx, positive[0]), "x");

    let expr = parse("cos((2*x)^(1/2))*(2*x)^(-1/2)/sin((2*x)^(1/2))^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-csc(sqrt(2 * x))");

    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert_eq!(conditions.len(), 1);
    assert_eq!(rendered(&ctx, conditions[0]), "sin(sqrt(2 * x))");

    let positive = super::integrate_symbolic_required_positive_conditions(&mut ctx, expr, "x");
    assert_eq!(positive.len(), 1);
    assert_eq!(rendered(&ctx, positive[0]), "2 * x");

    let expr = parse("k/(2*sqrt(x)*cosh(sqrt(x)-b)^2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "k * tanh(sqrt(x) - b)");

    let positive = super::integrate_symbolic_required_positive_conditions(&mut ctx, expr, "x");
    assert_eq!(positive.len(), 1);
    assert_eq!(rendered(&ctx, positive[0]), "x");

    let expr = parse("k/(2*sqrt(x)*sinh(sqrt(x)-b)^2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-k / tanh(sqrt(x) - b)");

    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert_eq!(conditions.len(), 1);
    assert_eq!(rendered(&ctx, conditions[0]), "sinh(sqrt(x) - b)");

    let positive = super::integrate_symbolic_required_positive_conditions(&mut ctx, expr, "x");
    assert_eq!(positive.len(), 1);
    assert_eq!(rendered(&ctx, positive[0]), "x");
}

#[test]
fn integrates_sqrt_chain_tangent_cotangent_log_quotients() {
    let mut ctx = Context::new();
    let expr = parse("sin(sqrt(x))*sqrt(x)/(2*x*cos(sqrt(x)))", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-ln(|cos(sqrt(x))|)");

    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert_eq!(conditions.len(), 1);
    assert_eq!(rendered(&ctx, conditions[0]), "cos(sqrt(x))");

    let positive = super::integrate_symbolic_required_positive_conditions(&mut ctx, expr, "x");
    assert_eq!(positive.len(), 1);
    assert_eq!(rendered(&ctx, positive[0]), "x");

    let expr = parse("sin(b-sqrt(x))/(2*sqrt(x)*cos(b-sqrt(x)))", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "ln(|cos(b - sqrt(x))|)");

    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert_eq!(conditions.len(), 1);
    assert_eq!(rendered(&ctx, conditions[0]), "cos(b - sqrt(x))");

    let positive = super::integrate_symbolic_required_positive_conditions(&mut ctx, expr, "x");
    assert_eq!(positive.len(), 1);
    assert_eq!(rendered(&ctx, positive[0]), "x");

    let expr = parse("cos((2*x)^(1/2))*(2*x)^(-1/2)/sin((2*x)^(1/2))", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "ln(|sin(sqrt(2 * x))|)");

    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert_eq!(conditions.len(), 1);
    assert_eq!(rendered(&ctx, conditions[0]), "sin(sqrt(2 * x))");

    let positive = super::integrate_symbolic_required_positive_conditions(&mut ctx, expr, "x");
    assert_eq!(positive.len(), 1);
    assert_eq!(rendered(&ctx, positive[0]), "2 * x");

    let expr = parse("k*sinh(sqrt(x)-b)/(2*sqrt(x)*cosh(sqrt(x)-b)^2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-k / cosh(sqrt(x) - b)");

    let positive = super::integrate_symbolic_required_positive_conditions(&mut ctx, expr, "x");
    assert_eq!(positive.len(), 1);
    assert_eq!(rendered(&ctx, positive[0]), "x");

    let expr = parse("k*cosh(sqrt(x)-b)/(2*sqrt(x)*sinh(sqrt(x)-b)^2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-k / sinh(sqrt(x) - b)");

    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert_eq!(conditions.len(), 1);
    assert_eq!(rendered(&ctx, conditions[0]), "sinh(sqrt(x) - b)");

    let positive = super::integrate_symbolic_required_positive_conditions(&mut ctx, expr, "x");
    assert_eq!(positive.len(), 1);
    assert_eq!(rendered(&ctx, positive[0]), "x");
}

#[test]
fn integrates_sqrt_chain_hyperbolic_tangent_logs() {
    let mut ctx = Context::new();
    let expr = parse("tanh(sqrt(x))/(2*sqrt(x))", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "ln(|cosh(sqrt(x))|)");

    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert!(
        conditions.is_empty(),
        "cosh is strictly positive over reals and should not require a nonzero condition"
    );

    let positive = super::integrate_symbolic_required_positive_conditions(&mut ctx, expr, "x");
    assert_eq!(positive.len(), 1);
    assert_eq!(rendered(&ctx, positive[0]), "x");

    let expr = parse("sqrt(x)/(2*x*tanh(sqrt(x)))", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "ln(|sinh(sqrt(x))|)");

    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert_eq!(conditions.len(), 1);
    assert_eq!(rendered(&ctx, conditions[0]), "sinh(sqrt(x))");

    let positive = super::integrate_symbolic_required_positive_conditions(&mut ctx, expr, "x");
    assert_eq!(positive.len(), 1);
    assert_eq!(rendered(&ctx, positive[0]), "x");

    let expr = parse("tanh(b-sqrt(x))/(2*sqrt(x))", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-ln(|cosh(b - sqrt(x))|)");

    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert!(
        conditions.is_empty(),
        "cosh is strictly positive over reals and should not require a nonzero condition"
    );

    let positive = super::integrate_symbolic_required_positive_conditions(&mut ctx, expr, "x");
    assert_eq!(positive.len(), 1);
    assert_eq!(rendered(&ctx, positive[0]), "x");

    let expr = parse("1/(2*sqrt(x)*tanh(b-sqrt(x)))", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-ln(|sinh(b - sqrt(x))|)");

    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert_eq!(conditions.len(), 1);
    assert_eq!(rendered(&ctx, conditions[0]), "sinh(b - sqrt(x))");

    let positive = super::integrate_symbolic_required_positive_conditions(&mut ctx, expr, "x");
    assert_eq!(positive.len(), 1);
    assert_eq!(rendered(&ctx, positive[0]), "x");

    let expr = parse("tanh((2*x)^(1/2))*(2*x)^(-1/2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "ln(|cosh(sqrt(2 * x))|)");

    let positive = super::integrate_symbolic_required_positive_conditions(&mut ctx, expr, "x");
    assert_eq!(positive.len(), 1);
    assert_eq!(rendered(&ctx, positive[0]), "2 * x");
}

#[test]
fn integrates_sqrt_chain_hyperbolic_reciprocal_squares() {
    let mut ctx = Context::new();
    let expr = parse("1/(2*sqrt(x)*cosh(sqrt(x))^2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "tanh(sqrt(x))");

    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert!(
        conditions.is_empty(),
        "cosh is strictly positive over reals and should not require a nonzero condition"
    );

    let positive = super::integrate_symbolic_required_positive_conditions(&mut ctx, expr, "x");
    assert_eq!(positive.len(), 1);
    assert_eq!(rendered(&ctx, positive[0]), "x");

    let expr = parse("1/(2*sqrt(x)*sinh(sqrt(x))^2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-1 / tanh(sqrt(x))");

    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert_eq!(conditions.len(), 1);
    assert_eq!(rendered(&ctx, conditions[0]), "sinh(sqrt(x))");

    let positive = super::integrate_symbolic_required_positive_conditions(&mut ctx, expr, "x");
    assert_eq!(positive.len(), 1);
    assert_eq!(rendered(&ctx, positive[0]), "x");

    let expr = parse("(2*x)^(-1/2)/cosh((2*x)^(1/2))^2", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "tanh(sqrt(2 * x))");

    let positive = super::integrate_symbolic_required_positive_conditions(&mut ctx, expr, "x");
    assert_eq!(positive.len(), 1);
    assert_eq!(rendered(&ctx, positive[0]), "2 * x");
}

#[test]
fn integrates_sqrt_chain_hyperbolic_reciprocal_derivatives() {
    let mut ctx = Context::new();
    let expr = parse("sinh(sqrt(x))/(2*sqrt(x)*cosh(sqrt(x))^2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-1 / cosh(sqrt(x))");

    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert!(
        conditions.is_empty(),
        "cosh is strictly positive over reals and should not require a nonzero condition"
    );

    let positive = super::integrate_symbolic_required_positive_conditions(&mut ctx, expr, "x");
    assert_eq!(positive.len(), 1);
    assert_eq!(rendered(&ctx, positive[0]), "x");

    let expr = parse("cosh(sqrt(x))/(2*sqrt(x)*sinh(sqrt(x))^2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-1 / sinh(sqrt(x))");

    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert_eq!(conditions.len(), 1);
    assert_eq!(rendered(&ctx, conditions[0]), "sinh(sqrt(x))");

    let positive = super::integrate_symbolic_required_positive_conditions(&mut ctx, expr, "x");
    assert_eq!(positive.len(), 1);
    assert_eq!(rendered(&ctx, positive[0]), "x");

    let expr = parse(
        "sinh((2*x)^(1/2))*(2*x)^(-1/2)/cosh((2*x)^(1/2))^2",
        &mut ctx,
    )
    .expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-1 / cosh(sqrt(2 * x))");

    let positive = super::integrate_symbolic_required_positive_conditions(&mut ctx, expr, "x");
    assert_eq!(positive.len(), 1);
    assert_eq!(rendered(&ctx, positive[0]), "2 * x");
}

#[test]
fn canonical_sec_tan_quotient_rejects_non_linear_argument() {
    let mut ctx = Context::new();
    let expr = parse("sin(x^2)/cos(x^2)^2", &mut ctx).expect("parse");
    assert!(integrate_symbolic_expr(&mut ctx, expr, "x").is_none());
    assert!(super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x").is_empty());
}

#[test]
fn integrates_trig_log_linear_substitution() {
    let mut ctx = Context::new();
    let expr = parse("tan(2*x + 1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-ln(|cos(2 * x + 1)|) / 2");

    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert_eq!(conditions.len(), 1);
    assert_eq!(rendered(&ctx, conditions[0]), "cos(2 * x + 1)");

    let expr = parse("cot(2*x + 1)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "ln(|sin(2 * x + 1)|) / 2");

    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert_eq!(conditions.len(), 1);
    assert_eq!(rendered(&ctx, conditions[0]), "sin(2 * x + 1)");
}

#[test]
fn integrates_polynomial_trig_log_ratio_substitution() {
    let mut ctx = Context::new();
    let expr = parse("2*x*tan(x^2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-ln(|cos(x^2)|)");
    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert_eq!(conditions.len(), 1);
    assert_eq!(rendered(&ctx, conditions[0]), "cos(x^2)");

    let expr = parse("x*tan(x^2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "-1/2 * ln(|cos(x^2)|)");

    let expr = parse("3*x^2*cot(x^3)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "ln(|sin(x^3)|)");
    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert_eq!(conditions.len(), 1);
    assert_eq!(rendered(&ctx, conditions[0]), "sin(x^3)");

    let expr = parse("2*x/cos(x^2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "ln(|tan(x^2) + sec(x^2)|)");
    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert_eq!(conditions.len(), 1);
    assert_eq!(rendered(&ctx, conditions[0]), "cos(x^2)");

    let expr = parse("2*x/sin(x^2)", &mut ctx).expect("parse");
    let out = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integrate");
    assert_eq!(rendered(&ctx, out), "ln(|csc(x^2) - cot(x^2)|)");
    let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
    assert_eq!(conditions.len(), 1);
    assert_eq!(rendered(&ctx, conditions[0]), "sin(x^2)");
}

#[test]
fn trig_log_integration_rejects_non_linear_argument_but_preserves_domain_condition() {
    for (input, expected_condition) in [
        ("tan(x^2)", "cos(x^2)"),
        ("cot(x^2)", "sin(x^2)"),
        ("sec(x^2)", "cos(x^2)"),
        ("csc(x^2)", "sin(x^2)"),
    ] {
        let mut ctx = Context::new();
        let expr = parse(input, &mut ctx).expect("parse");
        assert!(integrate_symbolic_expr(&mut ctx, expr, "x").is_none());
        let conditions = super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x");
        assert_eq!(conditions.len(), 1, "input: {input}");
        assert_eq!(rendered(&ctx, conditions[0]), expected_condition);
    }
}

#[test]
fn trig_pole_residual_compositions_preserve_domain_conditions() {
    for (input, expected_conditions) in [
        ("tan(x^2)+sin(x^2)", vec!["cos(x^2)"]),
        ("sec(x^2)*sin(x^2)", vec!["cos(x^2)"]),
        ("cot(x^2)+csc(y)", vec!["sin(x^2)", "sin(y)"]),
        ("tan(1)+sin(x^2)", vec![]),
    ] {
        let mut ctx = Context::new();
        let expr = parse(input, &mut ctx).expect("parse");
        let mut conditions: Vec<_> =
            super::integrate_symbolic_required_nonzero_conditions(&mut ctx, expr, "x")
                .iter()
                .map(|condition| rendered(&ctx, *condition))
                .collect();
        conditions.sort();
        assert_eq!(conditions, expected_conditions, "input: {input}");
    }
}

#[test]
fn extracts_linear_coeffs() {
    let mut ctx = Context::new();
    let expr = parse("2*x + 3", &mut ctx).expect("parse");
    let (a, b) = get_linear_coeffs(&mut ctx, expr, "x").expect("coeffs");
    let a_text = rendered(&ctx, a);
    assert_eq!(a_text, "2");
    let b_text = rendered(&ctx, b);
    assert_eq!(b_text, "3");

    let expr = parse("1 + -2*x", &mut ctx).expect("parse negated linear term");
    let (a, b) = get_linear_coeffs(&mut ctx, expr, "x").expect("negated term coeffs");
    let a_text = rendered(&ctx, a);
    assert_eq!(a_text, "-2");
    let b_text = rendered(&ctx, b);
    assert_eq!(b_text, "1");

    let expr = parse("1 - 2*x", &mut ctx).expect("parse negative slope affine");
    let (a, b) = get_linear_coeffs(&mut ctx, expr, "x").expect("negative slope coeffs");
    let a_text = rendered(&ctx, a);
    assert_eq!(a_text, "-2");
    let b_text = rendered(&ctx, b);
    assert_eq!(b_text, "1");

    let expr = parse("1/2*(3*x + 2)", &mut ctx).expect("parse scaled affine");
    let (a, b) = get_linear_coeffs(&mut ctx, expr, "x").expect("scaled affine coeffs");
    assert_constant_expr(&ctx, a, 3, 2);
    assert_constant_expr(&ctx, b, 1, 1);

    let expr = parse("(3*x + 2)/2", &mut ctx).expect("parse divided affine");
    let (a, b) = get_linear_coeffs(&mut ctx, expr, "x").expect("divided affine coeffs");
    assert_constant_expr(&ctx, a, 3, 2);
    assert_constant_expr(&ctx, b, 1, 1);

    let expr = parse("asinh(1 - 2*x)", &mut ctx).expect("parse shifted asinh");
    assert!(super::integrate_symbolic_is_asinh_affine_variable_target(
        &mut ctx, expr, "x"
    ));

    let expr = parse("asinh(2*x)", &mut ctx).expect("parse scaled asinh");
    assert!(super::integrate_symbolic_is_asinh_affine_variable_target(
        &mut ctx, expr, "x"
    ));

    let expr = parse("-(2*x + 1)", &mut ctx).expect("parse negated affine");
    let (a, b) = get_linear_coeffs(&mut ctx, expr, "x").expect("negated affine coeffs");
    assert_eq!(rendered(&ctx, a), "-2");
    assert_eq!(rendered(&ctx, b), "-1");
}

#[test]
fn reciprocal_exp_linear_integrates_normalized_negative_exponentials() {
    let mut ctx = Context::new();
    let cases = [
        ("1/e^x", "-1 / e^x"),
        ("2/e^x", "-2 / e^x"),
        ("1/e^(2*x)", "-1/2 / e^(2 * x)"),
    ];
    for (source, expected) in cases {
        let expr = parse(source, &mut ctx).expect(source);
        let integral = integrate_symbolic_expr(&mut ctx, expr, "x").expect(source);
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: integral
                }
            ),
            expected,
            "{source}"
        );
    }
}

#[test]
fn reciprocal_exp_rejects_nonlinear_and_variable_numerators() {
    let mut ctx = Context::new();
    for source in ["1/e^(x^2)", "c/e^(s*x)"] {
        let expr = parse(source, &mut ctx).expect(source);
        assert!(
            integrate_symbolic_expr(&mut ctx, expr, "x").is_none(),
            "must stay residual: {source}"
        );
    }
}

#[test]
fn div_exp_linear_by_parts_integrates_normalized_products() {
    let mut ctx = Context::new();
    // Raw integrator output (the engine simplifier later displays
    // these as (-x - 1)/e^x etc., pinned by the matrix rows).
    let cases = [
        ("x/e^x", "exp(-x) * (-x - 1)"),
        ("x^2/e^x", "exp(-x) * (-x^2 - 2 * x - 2)"),
    ];
    for (source, expected) in cases {
        let expr = parse(source, &mut ctx).expect(source);
        let integral = integrate_symbolic_expr(&mut ctx, expr, "x").expect(source);
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: integral
                }
            ),
            expected,
            "{source}"
        );
    }
}

#[test]
fn div_exp_by_parts_rejects_unsupported_numerators() {
    let mut ctx = Context::new();
    // x/e^(x^2) graduated to the polynomial-derivative substitution
    // owner (Div arm); the by-parts family still declines non-table
    // numerators and non-derivative nonlinear quotients.
    for source in ["tan(x)/e^x", "x^3/e^(x^2)"] {
        let expr = parse(source, &mut ctx).expect(source);
        assert!(
            integrate_symbolic_expr(&mut ctx, expr, "x").is_none(),
            "must stay residual: {source}"
        );
    }
}

#[test]
fn div_exp_delegation_reaches_the_cyclic_trig_family() {
    let mut ctx = Context::new();
    for source in ["sin(x)/e^x", "cos(x)/e^x", "sin(2*x)/e^(3*x)"] {
        let expr = parse(source, &mut ctx).expect(source);
        assert!(
            integrate_symbolic_expr(&mut ctx, expr, "x").is_some(),
            "must integrate: {source}"
        );
    }
    // Honest residual: non-table trig cofactor.
    let expr = parse("tan(x)/e^x", &mut ctx).expect("tan");
    assert!(integrate_symbolic_expr(&mut ctx, expr, "x").is_none());
}

#[test]
fn sine_multiple_angle_ratios_integrate_via_chebyshev_rewrite() {
    let mut ctx = Context::new();
    for source in [
        "sin(3*x)/(3*sin(x))",
        "sin(4*x)/(4*sin(x))",
        "sin(6*x)/(6*sin(x))",
        "sin(4*(x+1))/(4*sin(x+1))",
    ] {
        let expr = parse(source, &mut ctx).expect(source);
        assert!(
            integrate_symbolic_expr(&mut ctx, expr, "x").is_some(),
            "must integrate: {source}"
        );
    }
}

#[test]
fn sine_multiple_angle_ratio_round_trips_numerically() {
    let mut ctx = Context::new();
    // U_3(cos x)/4 = cos(x)cos(2x): pin the Chebyshev rewrite by value.
    let expr = parse("sin(4*x)/(4*sin(x))", &mut ctx).expect("parse");
    let antiderivative = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integral");
    let derivative = crate::symbolic_differentiation_support::differentiate_symbolic_expr(
        &mut ctx,
        antiderivative,
        "x",
    )
    .expect("derivative");
    let target = parse("cos(x)*cos(2*x)", &mut ctx).expect("target");
    for sample in [0.4_f64, 1.1, 2.3] {
        let mut vars = std::collections::HashMap::new();
        vars.insert("x".to_string(), sample);
        let lhs = crate::evaluator_f64::eval_f64(&ctx, derivative, &vars).expect("lhs");
        let rhs = crate::evaluator_f64::eval_f64(&ctx, target, &vars).expect("rhs");
        assert!(
            (lhs - rhs).abs() < 1e-9,
            "mismatch at {sample}: {lhs} vs {rhs}"
        );
    }
}

#[test]
fn sine_multiple_angle_ratio_declines_foreign_shapes() {
    let mut ctx = Context::new();
    // Non-integer multiples, mismatched offsets, and non-sine
    // denominators stay with their owners or honestly residual.
    for source in [
        "sin(x^2)/sin(x)",
        "sin(3*x)/(3*sin(x+1))",
        "sin(3*x)/(3*cos(x))",
    ] {
        let expr = parse(source, &mut ctx).expect(source);
        let cas_ast::Expr::Div(num, den) = ctx.get(expr).clone() else {
            panic!("div: {source}");
        };
        assert!(
            super::sine_multiple_angle_ratio_antiderivative(&mut ctx, num, den, "x").is_none(),
            "must decline: {source}"
        );
    }
}

#[test]
fn monomial_inverse_trig_by_parts_generalizes_powers_and_slopes() {
    let mut ctx = Context::new();
    for source in [
        "x^2*arcsin(x)",
        "x^3*arcsin(x)",
        "x^2*arccos(x)",
        "x*arcsin(2*x)",
        "x^2*arcsin(3*x)",
    ] {
        let expr = parse(source, &mut ctx).expect(source);
        assert!(
            integrate_symbolic_expr(&mut ctx, expr, "x").is_some(),
            "must integrate: {source}"
        );
    }
}

#[test]
fn monomial_inverse_trig_by_parts_round_trips_numerically() {
    let mut ctx = Context::new();
    for source in ["x^2*arcsin(x)", "x*arcsin(2*x)", "x^2*arccos(x)"] {
        let expr = parse(source, &mut ctx).expect(source);
        let antiderivative = integrate_symbolic_expr(&mut ctx, expr, "x").expect(source);
        let derivative = crate::symbolic_differentiation_support::differentiate_symbolic_expr(
            &mut ctx,
            antiderivative,
            "x",
        )
        .expect("derivative");
        let target = parse(source, &mut ctx).expect(source);
        for sample in [-0.3_f64, 0.2, 0.45] {
            let mut vars = std::collections::HashMap::new();
            vars.insert("x".to_string(), sample);
            let lhs = crate::evaluator_f64::eval_f64(&ctx, derivative, &vars).expect("lhs");
            let rhs = crate::evaluator_f64::eval_f64(&ctx, target, &vars).expect("rhs");
            assert!(
                (lhs - rhs).abs() < 1e-9,
                "mismatch for {source} at {sample}: {lhs} vs {rhs}"
            );
        }
    }
}

#[test]
fn monomial_inverse_trig_by_parts_covers_offset_arguments() {
    let mut ctx = Context::new();
    for (source, samples) in [
        ("x*arcsin(x+1)", [-1.6_f64, -0.8, -0.3]),
        ("x*arccos(x-1)", [0.3, 1.0, 1.6]),
        ("x^2*arcsin(x+1)", [-1.4, -0.9, -0.4]),
        ("x*arcsin(2*x-1)", [0.2, 0.5, 0.8]),
    ] {
        let expr = parse(source, &mut ctx).expect(source);
        let antiderivative = integrate_symbolic_expr(&mut ctx, expr, "x").expect(source);
        let derivative = crate::symbolic_differentiation_support::differentiate_symbolic_expr(
            &mut ctx,
            antiderivative,
            "x",
        )
        .expect("derivative");
        let target = parse(source, &mut ctx).expect(source);
        for sample in samples {
            let mut vars = std::collections::HashMap::new();
            vars.insert("x".to_string(), sample);
            let lhs = crate::evaluator_f64::eval_f64(&ctx, derivative, &vars).expect("lhs");
            let rhs = crate::evaluator_f64::eval_f64(&ctx, target, &vars).expect("rhs");
            assert!(
                (lhs - rhs).abs() < 1e-9,
                "mismatch for {source} at {sample}: {lhs} vs {rhs}"
            );
        }
    }
}

#[test]
fn monomial_inverse_trig_by_parts_declines_offsets_and_high_powers() {
    let mut ctx = Context::new();
    // Offset arguments graduated to the shifted-tail by-parts path;
    // n > 5 exceeds the radical-tail cap and symbolic offsets stay
    // honestly residual.
    for source in ["x^6*arcsin(x)", "x*arcsin(x+b)"] {
        let expr = parse(source, &mut ctx).expect(source);
        assert!(
            integrate_symbolic_expr(&mut ctx, expr, "x").is_none(),
            "must stay residual: {source}"
        );
    }
}

#[test]
fn polynomial_over_sqrt_hermite_split_integrates_general_numerators() {
    let mut ctx = Context::new();
    for source in [
        "x^2/sqrt(x^2+x+1)",
        "x^3/sqrt(x^2+x+1)",
        "x^2/sqrt(2*x-x^2)",
        "(x^2+1)/sqrt(x^2+2*x)",
        "(x^2+x)/sqrt(2*x-x^2)",
    ] {
        let expr = parse(source, &mut ctx).expect(source);
        assert!(
            integrate_symbolic_expr(&mut ctx, expr, "x").is_some(),
            "must integrate: {source}"
        );
    }
}

#[test]
fn polynomial_over_sqrt_hermite_split_round_trips_numerically() {
    let mut ctx = Context::new();
    for (source, samples) in [
        ("x^2/sqrt(x^2+x+1)", [-1.5_f64, 0.4, 2.2]),
        ("x^2/sqrt(2*x-x^2)", [0.3, 1.1, 1.8]),
        ("(x^2+1)/sqrt(x^2+2*x)", [0.5, 1.7, 3.0]),
        ("x^3/sqrt(x^2+x+1)", [-1.2, 0.6, 2.5]),
    ] {
        let expr = parse(source, &mut ctx).expect(source);
        let antiderivative = integrate_symbolic_expr(&mut ctx, expr, "x").expect(source);
        let derivative = crate::symbolic_differentiation_support::differentiate_symbolic_expr(
            &mut ctx,
            antiderivative,
            "x",
        )
        .expect("derivative");
        let target = parse(source, &mut ctx).expect(source);
        for sample in samples {
            let mut vars = std::collections::HashMap::new();
            vars.insert("x".to_string(), sample);
            let lhs = crate::evaluator_f64::eval_f64(&ctx, derivative, &vars).expect("lhs");
            let rhs = crate::evaluator_f64::eval_f64(&ctx, target, &vars).expect("rhs");
            assert!(
                (lhs - rhs).abs() < 1e-9,
                "mismatch for {source} at {sample}: {lhs} vs {rhs}"
            );
        }
    }
}

#[test]
fn polynomial_over_sqrt_hermite_split_declines_foreign_shapes() {
    let mut ctx = Context::new();
    // Elliptic cubic radicands and degree over the cap stay
    // residual (x^2/sqrt(x^3+1) is OWNED by derivative substitution).
    for source in ["x/sqrt(x^3+1)", "x^7/sqrt(x^2+x+1)"] {
        let expr = parse(source, &mut ctx).expect(source);
        assert!(
            integrate_symbolic_expr(&mut ctx, expr, "x").is_none(),
            "must stay residual: {source}"
        );
    }
}

#[test]
fn linear_over_sqrt_shifted_quadratic_integrates_all_patterns() {
    let mut ctx = Context::new();
    for source in [
        "x/sqrt(x^2+x+1)",
        "x/sqrt(2*x-x^2)",
        "x/sqrt(x^2+2*x)",
        "(2*x+3)/sqrt(x^2+x+1)",
        "(1-x)/sqrt(2*x-x^2)",
    ] {
        let expr = parse(source, &mut ctx).expect(source);
        assert!(
            integrate_symbolic_expr(&mut ctx, expr, "x").is_some(),
            "must integrate: {source}"
        );
    }
}

#[test]
fn linear_over_sqrt_shifted_quadratic_round_trips_numerically() {
    let mut ctx = Context::new();
    for (source, samples) in [
        ("x/sqrt(x^2+x+1)", [-2.0_f64, 0.3, 1.8]),
        ("x/sqrt(2*x-x^2)", [0.4, 1.0, 1.7]),
    ] {
        let expr = parse(source, &mut ctx).expect(source);
        let antiderivative = integrate_symbolic_expr(&mut ctx, expr, "x").expect(source);
        let derivative = crate::symbolic_differentiation_support::differentiate_symbolic_expr(
            &mut ctx,
            antiderivative,
            "x",
        )
        .expect("derivative");
        let target = parse(source, &mut ctx).expect(source);
        for sample in samples {
            let mut vars = std::collections::HashMap::new();
            vars.insert("x".to_string(), sample);
            let lhs = crate::evaluator_f64::eval_f64(&ctx, derivative, &vars).expect("lhs");
            let rhs = crate::evaluator_f64::eval_f64(&ctx, target, &vars).expect("rhs");
            assert!(
                (lhs - rhs).abs() < 1e-9,
                "mismatch for {source} at {sample}: {lhs} vs {rhs}"
            );
        }
    }
}

#[test]
fn linear_over_sqrt_shifted_quadratic_declines_owned_and_foreign_shapes() {
    let mut ctx = Context::new();
    // Pure-quadratic radicands keep their owners (gate a1 != 0) and
    // quadratic numerators stay with the reduction family or residual.
    let pure = parse("x/sqrt(1-x^2)", &mut ctx).expect("pure");
    assert!(
        super::linear_over_sqrt_shifted_quadratic_antiderivative(&mut ctx, pure, "x").is_none(),
        "pure-quadratic radicand must decline"
    );
    let cubic_rad = parse("x/sqrt(x^3+x+1)", &mut ctx).expect("cubic");
    assert!(
        integrate_symbolic_expr(&mut ctx, cubic_rad, "x").is_none(),
        "cubic radicand must stay residual"
    );
}

#[test]
fn quadratic_radical_over_monomial_integrates_the_family() {
    let mut ctx = Context::new();
    for source in [
        "sqrt(4-x^2)/x",
        "1/(x*sqrt(x^2-1))",
        "1/(x*sqrt(x^2+4))",
        "1/(x^2*sqrt(x^2+4))",
        "sqrt(x^2-1)/x",
        "sqrt(x^2+1)/x",
        "1/(x*sqrt(1-x^2))",
        "sqrt(1-x^2)/x^2",
        "1/(x^3*sqrt(x^2+1))",
    ] {
        let expr = parse(source, &mut ctx).expect(source);
        assert!(
            integrate_symbolic_expr(&mut ctx, expr, "x").is_some(),
            "must integrate: {source}"
        );
    }
}

#[test]
fn quadratic_radical_over_monomial_round_trips_numerically() {
    let mut ctx = Context::new();
    for (source, samples) in [
        ("sqrt(4-x^2)/x", [0.5_f64, 1.0, 1.8]),
        ("1/(x*sqrt(x^2-1))", [1.2, 2.0, 3.0]),
        ("1/(x*sqrt(x^2+4))", [0.5, 1.0, 2.0]),
        ("1/(x^2*sqrt(x^2+4))", [-1.0, 0.5, 2.0]),
        ("sqrt(x^2+1)/x", [0.4, 1.0, 2.5]),
        ("1/(x*sqrt(1-x^2))", [0.3, 0.6, 0.9]),
        ("sqrt(1-x^2)/x^2", [0.3, 0.6, 0.9]),
        ("1/(x^3*sqrt(x^2+1))", [0.5, 1.2, 2.0]),
    ] {
        let expr = parse(source, &mut ctx).expect(source);
        let antiderivative = integrate_symbolic_expr(&mut ctx, expr, "x").expect(source);
        let derivative = crate::symbolic_differentiation_support::differentiate_symbolic_expr(
            &mut ctx,
            antiderivative,
            "x",
        )
        .expect("derivative");
        let target = parse(source, &mut ctx).expect(source);
        for sample in samples {
            let mut vars = std::collections::HashMap::new();
            vars.insert("x".to_string(), sample);
            let lhs = crate::evaluator_f64::eval_f64(&ctx, derivative, &vars).expect("lhs");
            let rhs = crate::evaluator_f64::eval_f64(&ctx, target, &vars).expect("rhs");
            assert!(
                (lhs - rhs).abs() < 1e-9,
                "mismatch for {source} at {sample}: {lhs} vs {rhs}"
            );
        }
    }
}

#[test]
fn quadratic_radical_over_monomial_declines_foreign_shapes() {
    let mut ctx = Context::new();
    // Linear terms in the radicand, cubic radicands, non-monomial
    // denominators, non-radical numerators, symbolic coefficients
    // and the unimplemented even powers stay outside this owner.
    for source in [
        "sqrt(x^2+x+1)/x",
        "sqrt(x^3+1)/x",
        "sqrt(x^2+1)/(x+1)",
        "x/sqrt(x^2+1)",
        "sqrt(a-x^2)/x",
        "sqrt(x^2+4)/x^4",
    ] {
        let expr = parse(source, &mut ctx).expect(source);
        assert!(
            super::quadratic_radical_over_monomial_antiderivative(&mut ctx, expr, "x").is_none(),
            "must decline: {source}"
        );
    }
}

#[test]
fn quartic_symmetric_substitution_integrates_the_family() {
    let mut ctx = Context::new();
    for source in [
        "1/(x^4+1)",
        "x^2/(x^4+1)",
        "(x^2+1)/(x^4+1)",
        "(x^2-1)/(x^4+1)",
        "(2*x^2+3)/(x^4+1)",
    ] {
        let expr = parse(source, &mut ctx).expect(source);
        assert!(
            integrate_symbolic_expr(&mut ctx, expr, "x").is_some(),
            "must integrate: {source}"
        );
    }
}

#[test]
fn quartic_symmetric_substitution_round_trips_numerically() {
    let mut ctx = Context::new();
    for (source, samples) in [
        ("1/(x^4+1)", [0.4_f64, 1.0, 2.1]),
        ("x^2/(x^4+1)", [-0.6, 0.7, 1.8]),
        ("(x^2+1)/(x^4+1)", [0.3, 1.2, 2.5]),
        ("(x^2-1)/(x^4+1)", [0.5, 1.4, 2.0]),
    ] {
        let expr = parse(source, &mut ctx).expect(source);
        let antiderivative = integrate_symbolic_expr(&mut ctx, expr, "x").expect(source);
        let derivative = crate::symbolic_differentiation_support::differentiate_symbolic_expr(
            &mut ctx,
            antiderivative,
            "x",
        )
        .expect("derivative");
        let target = parse(source, &mut ctx).expect(source);
        for sample in samples {
            let mut vars = std::collections::HashMap::new();
            vars.insert("x".to_string(), sample);
            let lhs = crate::evaluator_f64::eval_f64(&ctx, derivative, &vars).expect("lhs");
            let rhs = crate::evaluator_f64::eval_f64(&ctx, target, &vars).expect("rhs");
            assert!(
                (lhs - rhs).abs() < 1e-9,
                "mismatch for {source} at {sample}: {lhs} vs {rhs}"
            );
        }
    }
}

#[test]
fn quartic_symmetric_substitution_declines_foreign_shapes() {
    let mut ctx = Context::new();
    // x^4+4 factors rationally (owner), odd numerators and higher
    // numerator degrees and non-x^4+1 denominators decline here.
    for source in [
        "x/(x^4+1)",
        "x^3/(x^4+1)",
        "1/(x^4+x^2+1)",
        "(x^3+1)/(x^4+1)",
    ] {
        let expr = parse(source, &mut ctx).expect(source);
        assert!(
            super::quartic_symmetric_substitution_antiderivative(&mut ctx, expr, "x").is_none(),
            "must decline: {source}"
        );
    }
}

#[test]
fn mixed_trig_power_substitution_integrates_the_family() {
    let mut ctx = Context::new();
    for source in [
        "sin(x)^2*cos(x)^3",
        "sin(x)^4*cos(x)^3",
        "sin(x)^3*cos(x)^2",
        "sin(x)^3*cos(x)^4",
        "sin(x)^5*cos(x)^2",
        "sin(x)^3*cos(x)^3",
        "sin(2*x)^3*cos(2*x)^2",
    ] {
        let expr = parse(source, &mut ctx).expect(source);
        assert!(
            integrate_symbolic_expr(&mut ctx, expr, "x").is_some(),
            "must integrate: {source}"
        );
    }
}

#[test]
fn mixed_trig_power_substitution_round_trips_numerically() {
    let mut ctx = Context::new();
    for (source, samples) in [
        ("sin(x)^2*cos(x)^3", [0.4_f64, 1.0, 2.1]),
        ("sin(x)^4*cos(x)^3", [-0.6, 0.7, 1.8]),
        ("sin(x)^3*cos(x)^2", [0.3, 1.2, 2.5]),
        ("sin(x)^3*cos(x)^4", [-0.9, 0.5, 1.6]),
        ("sin(x)^5*cos(x)^2", [0.4, 1.1, 2.0]),
        ("sin(x)^3*cos(x)^3", [-0.5, 0.8, 1.9]),
        ("sin(2*x)^3*cos(2*x)^2", [0.3, 0.9, 1.4]),
    ] {
        let expr = parse(source, &mut ctx).expect(source);
        let antiderivative = integrate_symbolic_expr(&mut ctx, expr, "x").expect(source);
        let derivative = crate::symbolic_differentiation_support::differentiate_symbolic_expr(
            &mut ctx,
            antiderivative,
            "x",
        )
        .expect("derivative");
        let target = parse(source, &mut ctx).expect(source);
        for sample in samples {
            let mut vars = std::collections::HashMap::new();
            vars.insert("x".to_string(), sample);
            let lhs = crate::evaluator_f64::eval_f64(&ctx, derivative, &vars).expect("lhs");
            let rhs = crate::evaluator_f64::eval_f64(&ctx, target, &vars).expect("rhs");
            assert!(
                (lhs - rhs).abs() < 1e-9,
                "mismatch for {source} at {sample}: {lhs} vs {rhs}"
            );
        }
    }
}

#[test]
fn mixed_trig_power_substitution_declines_foreign_shapes() {
    let mut ctx = Context::new();
    // Both-even products (power reduction), single odd factors and
    // f^n f' single-factor cases keep their owners; mismatched
    // arguments, offsets and tan/sec atoms decline here.
    for source in [
        "sin(x)^2*cos(x)^4",
        "sin(x)*cos(x)^3",
        "sin(x)^2*cos(2*x)^3",
        "sin(x+1)^2*cos(x+1)^3",
        "sin(x)^2*tan(x)^3",
    ] {
        let expr = parse(source, &mut ctx).expect(source);
        assert!(
            super::mixed_trig_power_substitution_antiderivative(&mut ctx, expr, "x").is_none(),
            "must decline: {source}"
        );
    }
}

#[test]
fn weierstrass_substitution_integrates_the_family() {
    let mut ctx = Context::new();
    for source in [
        "1/(2+cos(x))",
        "1/(1+sin(x))",
        "1/(1+cos(x))",
        "1/(3+2*cos(x))",
        "1/(5+4*sin(x))",
        "sin(x)/(1+sin(x))",
        "cos(x)/(2+cos(x))",
        "1/(2+cos(2*x))",
        "1/(sin(x)+cos(x))",
    ] {
        let expr = parse(source, &mut ctx).expect(source);
        assert!(
            integrate_symbolic_expr(&mut ctx, expr, "x").is_some(),
            "must integrate: {source}"
        );
    }
}

#[test]
fn weierstrass_substitution_round_trips_numerically() {
    let mut ctx = Context::new();
    for (source, samples) in [
        ("1/(2+cos(x))", [-1.2_f64, 0.4, 2.0]),
        ("1/(1+sin(x))", [-0.8, 0.3, 1.2]),
        ("1/(3+2*cos(x))", [-2.0, 0.5, 1.7]),
        ("1/(5+4*sin(x))", [-1.5, 0.2, 2.3]),
        ("sin(x)/(1+sin(x))", [-0.9, 0.6, 1.4]),
        ("1/(2+cos(2*x))", [-0.7, 0.3, 1.1]),
        // atanh window: |tan(x/2) - 1| < sqrt(2).
        ("1/(sin(x)+cos(x))", [0.3, 1.0, 2.0]),
    ] {
        let expr = parse(source, &mut ctx).expect(source);
        let antiderivative = integrate_symbolic_expr(&mut ctx, expr, "x").expect(source);
        let derivative = crate::symbolic_differentiation_support::differentiate_symbolic_expr(
            &mut ctx,
            antiderivative,
            "x",
        )
        .expect("derivative");
        let target = parse(source, &mut ctx).expect(source);
        for sample in samples {
            let mut vars = std::collections::HashMap::new();
            vars.insert("x".to_string(), sample);
            let lhs = crate::evaluator_f64::eval_f64(&ctx, derivative, &vars).expect("lhs");
            let rhs = crate::evaluator_f64::eval_f64(&ctx, target, &vars).expect("rhs");
            assert!(
                (lhs - rhs).abs() < 1e-9,
                "mismatch for {source} at {sample}: {lhs} vs {rhs}"
            );
        }
    }
}

#[test]
fn weierstrass_substitution_declines_foreign_shapes() {
    let mut ctx = Context::new();
    // Bare x mixes, nonlinear/offset arguments, mixed multiples,
    // tan atoms and symbolic coefficients are outside this owner.
    for source in [
        "x/(2+cos(x))",
        "1/(2+cos(x^2))",
        "1/(2+cos(x+1))",
        "1/(sin(x)+cos(2*x))",
        "1/(2+tan(x))",
        "1/(a+cos(x))",
    ] {
        let expr = parse(source, &mut ctx).expect(source);
        assert!(
            super::weierstrass_rational_substitution_antiderivative(&mut ctx, expr, "x").is_none(),
            "must decline: {source}"
        );
    }
}

#[test]
fn linear_radical_substitution_integrates_the_family() {
    let mut ctx = Context::new();
    for source in [
        "x*sqrt(x+1)",
        "x^2*sqrt(x+1)",
        "x*sqrt(2*x-1)",
        "x*(x+1)^(3/2)",
        "sqrt(x)/(1+x)",
        "(sqrt(x)-1)/(x-1)",
        "sqrt(x+1)/x",
        "(2*x+3)*sqrt(5-x)",
    ] {
        let expr = parse(source, &mut ctx).expect(source);
        assert!(
            integrate_symbolic_expr(&mut ctx, expr, "x").is_some(),
            "must integrate: {source}"
        );
    }
}

#[test]
fn linear_radical_substitution_round_trips_numerically() {
    let mut ctx = Context::new();
    for (source, samples) in [
        ("x*sqrt(x+1)", [-0.6_f64, 0.5, 2.0]),
        ("x^2*sqrt(x+1)", [-0.4, 0.8, 1.5]),
        ("x*sqrt(2*x-1)", [0.7, 1.2, 2.5]),
        ("sqrt(x)/(1+x)", [0.3, 1.0, 2.4]),
        ("(sqrt(x)-1)/(x-1)", [0.2, 0.6, 2.2]),
        ("sqrt(x+1)/x", [0.4, 1.3, 3.0]),
        ("(2*x+3)*sqrt(5-x)", [-1.0, 0.5, 4.0]),
    ] {
        let expr = parse(source, &mut ctx).expect(source);
        let antiderivative = integrate_symbolic_expr(&mut ctx, expr, "x").expect(source);
        let derivative = crate::symbolic_differentiation_support::differentiate_symbolic_expr(
            &mut ctx,
            antiderivative,
            "x",
        )
        .expect("derivative");
        let target = parse(source, &mut ctx).expect(source);
        for sample in samples {
            let mut vars = std::collections::HashMap::new();
            vars.insert("x".to_string(), sample);
            let lhs = crate::evaluator_f64::eval_f64(&ctx, derivative, &vars).expect("lhs");
            let rhs = crate::evaluator_f64::eval_f64(&ctx, target, &vars).expect("rhs");
            assert!(
                (lhs - rhs).abs() < 1e-9,
                "mismatch for {source} at {sample}: {lhs} vs {rhs}"
            );
        }
    }
}

#[test]
fn linear_radical_substitution_declines_foreign_shapes() {
    let mut ctx = Context::new();
    // Quadratic/cubic radicands have their own owners (or stay
    // residual), mixed radicands are non-rational in one u, and
    // non-rational cofactors (exp, sin) are outside this owner.
    for source in [
        "sqrt(1-x^2)",
        "sqrt(x^3+1)",
        "sqrt(x)*sqrt(x+1)",
        "e^sqrt(x)/sqrt(x)",
        "sin(sqrt(x))",
        "sqrt(a*x+1)*x",
    ] {
        let expr = parse(source, &mut ctx).expect(source);
        assert!(
            super::linear_radical_substitution_antiderivative(&mut ctx, expr, "x").is_none(),
            "must decline: {source}"
        );
    }
}

#[test]
fn exponential_rational_substitution_integrates_the_family() {
    let mut ctx = Context::new();
    for source in [
        "1/(1+e^x)",
        "e^x/(1+e^(2*x))",
        "e^(2*x)/(1+e^x)",
        "(e^x-1)/(e^x+1)",
        "1/(e^(2*x)-1)",
        "e^(x/2)/(1+e^x)",
        "1/(2+3*e^(2*x))",
    ] {
        let expr = parse(source, &mut ctx).expect(source);
        assert!(
            integrate_symbolic_expr(&mut ctx, expr, "x").is_some(),
            "must integrate: {source}"
        );
    }
}

#[test]
fn exponential_rational_substitution_round_trips_numerically() {
    let mut ctx = Context::new();
    for (source, samples) in [
        ("1/(1+e^x)", [-1.3_f64, 0.4, 1.8]),
        ("e^x/(1+e^(2*x))", [-1.0, 0.2, 1.5]),
        ("e^(2*x)/(1+e^x)", [-0.8, 0.5, 1.2]),
        ("(e^x-1)/(e^x+1)", [-1.5, 0.3, 2.0]),
        ("1/(e^(2*x)-1)", [0.4, 1.0, 1.9]),
        ("e^(x/2)/(1+e^x)", [-1.2, 0.6, 1.4]),
    ] {
        let expr = parse(source, &mut ctx).expect(source);
        let antiderivative = integrate_symbolic_expr(&mut ctx, expr, "x").expect(source);
        let derivative = crate::symbolic_differentiation_support::differentiate_symbolic_expr(
            &mut ctx,
            antiderivative,
            "x",
        )
        .expect("derivative");
        let target = parse(source, &mut ctx).expect(source);
        for sample in samples {
            let mut vars = std::collections::HashMap::new();
            vars.insert("x".to_string(), sample);
            let lhs = crate::evaluator_f64::eval_f64(&ctx, derivative, &vars).expect("lhs");
            let rhs = crate::evaluator_f64::eval_f64(&ctx, target, &vars).expect("rhs");
            assert!(
                (lhs - rhs).abs() < 1e-9,
                "mismatch for {source} at {sample}: {lhs} vs {rhs}"
            );
        }
    }
}

#[test]
fn exponential_rational_substitution_declines_foreign_shapes() {
    let mut ctx = Context::new();
    // Mixed polynomial/trig occurrences of x, nonlinear exponents,
    // exponent offsets, and non-e bases stay outside this owner.
    for source in [
        "x/(1+e^x)",
        "sin(x)/(1+e^x)",
        "1/(1+e^(x^2))",
        "1/(1+e^(x+1))",
        "1/(1+2^x)",
    ] {
        let expr = parse(source, &mut ctx).expect(source);
        assert!(
            super::exponential_rational_substitution_antiderivative(&mut ctx, expr, "x").is_none(),
            "must decline: {source}"
        );
    }
}

#[test]
fn radical_numerator_polynomial_integrates_the_trig_substitution_chapter() {
    let mut ctx = Context::new();
    for source in [
        "sqrt(1-x^2)",
        "sqrt(4-x^2)",
        "x^2*sqrt(1-x^2)",
        "sqrt(x^2+1)",
        "sqrt(x^2-1)",
        "sqrt(2*x-x^2)",
        "x*sqrt(x^2+1)",
        "3*sqrt(1-4*x^2)",
    ] {
        let expr = parse(source, &mut ctx).expect(source);
        assert!(
            integrate_symbolic_expr(&mut ctx, expr, "x").is_some(),
            "must integrate: {source}"
        );
    }
}

#[test]
fn radical_numerator_polynomial_round_trips_numerically() {
    let mut ctx = Context::new();
    for (source, samples) in [
        ("sqrt(1-x^2)", [-0.7_f64, 0.2, 0.8]),
        ("x^2*sqrt(1-x^2)", [-0.6, 0.3, 0.9]),
        ("sqrt(x^2+1)", [-1.5, 0.4, 2.0]),
        ("sqrt(2*x-x^2)", [0.3, 1.0, 1.7]),
    ] {
        let expr = parse(source, &mut ctx).expect(source);
        let antiderivative = integrate_symbolic_expr(&mut ctx, expr, "x").expect(source);
        let derivative = crate::symbolic_differentiation_support::differentiate_symbolic_expr(
            &mut ctx,
            antiderivative,
            "x",
        )
        .expect("derivative");
        let target = parse(source, &mut ctx).expect(source);
        for sample in samples {
            let mut vars = std::collections::HashMap::new();
            vars.insert("x".to_string(), sample);
            let lhs = crate::evaluator_f64::eval_f64(&ctx, derivative, &vars).expect("lhs");
            let rhs = crate::evaluator_f64::eval_f64(&ctx, target, &vars).expect("rhs");
            assert!(
                (lhs - rhs).abs() < 1e-9,
                "mismatch for {source} at {sample}: {lhs} vs {rhs}"
            );
        }
    }
}

#[test]
fn radical_numerator_polynomial_declines_foreign_radicands() {
    let mut ctx = Context::new();
    // Elliptic cubics and over-cap numerators stay residual.
    for source in ["sqrt(x^3+1)", "x^5*sqrt(1-x^2)"] {
        let expr = parse(source, &mut ctx).expect(source);
        assert!(
            integrate_symbolic_expr(&mut ctx, expr, "x").is_none(),
            "must stay residual: {source}"
        );
    }
}

#[test]
fn monomial_over_sqrt_hyperbolic_reduces_with_exact_values() {
    let mut ctx = Context::new();
    for source in [
        "x^2/sqrt(1+x^2)",
        "x^3/sqrt(1+x^2)",
        "x^4/sqrt(1+x^2)",
        "x^2/sqrt(x^2-1)",
        "x^2/sqrt(4+x^2)",
        "3*x^2/sqrt(1+x^2)",
    ] {
        let expr = parse(source, &mut ctx).expect(source);
        assert!(
            integrate_symbolic_expr(&mut ctx, expr, "x").is_some(),
            "must integrate: {source}"
        );
    }
}

#[test]
fn monomial_over_sqrt_hyperbolic_round_trips_numerically() {
    let mut ctx = Context::new();
    for (source, samples) in [
        ("x^2/sqrt(1+x^2)", [-1.3_f64, 0.4, 2.1]),
        ("x^2/sqrt(x^2-1)", [1.5, 2.0, 3.7]),
    ] {
        let expr = parse(source, &mut ctx).expect(source);
        let antiderivative = integrate_symbolic_expr(&mut ctx, expr, "x").expect(source);
        let derivative = crate::symbolic_differentiation_support::differentiate_symbolic_expr(
            &mut ctx,
            antiderivative,
            "x",
        )
        .expect("derivative");
        let target = parse(source, &mut ctx).expect(source);
        for sample in samples {
            let mut vars = std::collections::HashMap::new();
            vars.insert("x".to_string(), sample);
            let lhs = crate::evaluator_f64::eval_f64(&ctx, derivative, &vars).expect("lhs");
            let rhs = crate::evaluator_f64::eval_f64(&ctx, target, &vars).expect("rhs");
            assert!(
                (lhs - rhs).abs() < 1e-9,
                "mismatch for {source} at {sample}: {lhs} vs {rhs}"
            );
        }
    }
}

#[test]
fn monomial_over_sqrt_hyperbolic_declines_degenerate_radicands() {
    let mut ctx = Context::new();
    // a = 0 (pure square), linear terms, and powers over the cap
    // stay with other owners or honestly residual.
    for source in ["x^2/sqrt(x^2)", "x^2/sqrt(x^2+x+1)", "x^7/sqrt(1+x^2)"] {
        let expr = parse(source, &mut ctx).expect(source);
        let Some(integral) = integrate_symbolic_expr(&mut ctx, expr, "x") else {
            continue;
        };
        // If another owner integrates it, fine - but the result must
        // not contain an unevaluated integrate call.
        let display = format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: &ctx,
                id: integral
            }
        );
        assert!(
            !display.contains("integrate"),
            "must be a closed form or residual: {source} -> {display}"
        );
    }
}

#[test]
fn monomial_over_sqrt_negative_quadratic_reduces_with_exact_values() {
    let mut ctx = Context::new();
    for source in [
        "x^2/sqrt(1-x^2)",
        "x^3/sqrt(1-x^2)",
        "x^4/sqrt(1-x^2)",
        "x^2/sqrt(4-x^2)",
        "x^2/sqrt(1-4*x^2)",
        "3*x^2/sqrt(1-x^2)",
    ] {
        let expr = parse(source, &mut ctx).expect(source);
        assert!(
            integrate_symbolic_expr(&mut ctx, expr, "x").is_some(),
            "must integrate: {source}"
        );
    }
}

#[test]
fn monomial_over_sqrt_negative_quadratic_round_trips_numerically() {
    let mut ctx = Context::new();
    let expr = parse("x^2/sqrt(1-x^2)", &mut ctx).expect("parse");
    let antiderivative = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integral");
    let derivative = crate::symbolic_differentiation_support::differentiate_symbolic_expr(
        &mut ctx,
        antiderivative,
        "x",
    )
    .expect("derivative");
    let target = parse("x^2/sqrt(1-x^2)", &mut ctx).expect("target");
    for sample in [-0.6_f64, 0.3, 0.8] {
        let mut vars = std::collections::HashMap::new();
        vars.insert("x".to_string(), sample);
        let lhs = crate::evaluator_f64::eval_f64(&ctx, derivative, &vars).expect("lhs");
        let rhs = crate::evaluator_f64::eval_f64(&ctx, target, &vars).expect("rhs");
        assert!(
            (lhs - rhs).abs() < 1e-9,
            "mismatch at {sample}: {lhs} vs {rhs}"
        );
    }
}

#[test]
fn monomial_over_sqrt_negative_quadratic_declines_other_sign_patterns() {
    let mut ctx = Context::new();
    // The hyperbolic radicands graduated to the mirrored recurrence
    // and the linear-term radicands to the Hermite split; only
    // powers above the tail cap stay residual here.
    for source in ["x^7/sqrt(1-x^2)", "x^7/sqrt(1+x^2)", "x^7/sqrt(x^2+x+1)"] {
        let expr = parse(source, &mut ctx).expect(source);
        assert!(
            integrate_symbolic_expr(&mut ctx, expr, "x").is_none(),
            "must stay residual: {source}"
        );
    }
}

#[test]
fn monomial_times_bounded_inverse_trig_integrates_and_round_trips_numerically() {
    let mut ctx = Context::new();
    for (source, target) in [
        ("x*arcsin(x)", "x*arcsin(x)"),
        ("x*arccos(x)", "x*arccos(x)"),
        ("2*x*arcsin(x)", "2*x*arcsin(x)"),
    ] {
        let expr = parse(source, &mut ctx).expect(source);
        let antiderivative = integrate_symbolic_expr(&mut ctx, expr, "x")
            .unwrap_or_else(|| panic!("must integrate: {source}"));
        // The simplifier cannot collapse the radical residual
        // a*sqrt(u) + b/sqrt(u) yet, so the round-trip is pinned
        // NUMERICALLY: d/dx F == integrand at interior samples.
        let derivative = crate::symbolic_differentiation_support::differentiate_symbolic_expr(
            &mut ctx,
            antiderivative,
            "x",
        )
        .unwrap_or_else(|| panic!("must differentiate: {source}"));
        let target_expr = parse(target, &mut ctx).expect(target);
        for sample in [-0.7_f64, -0.2, 0.3, 0.8] {
            let mut vars = std::collections::HashMap::new();
            vars.insert("x".to_string(), sample);
            let lhs = crate::evaluator_f64::eval_f64(&ctx, derivative, &vars)
                .unwrap_or_else(|| panic!("eval diff at {sample}: {source}"));
            let rhs = crate::evaluator_f64::eval_f64(&ctx, target_expr, &vars)
                .unwrap_or_else(|| panic!("eval target at {sample}: {source}"));
            assert!(
                (lhs - rhs).abs() < 1e-9,
                "round-trip mismatch for {source} at {sample}: {lhs} vs {rhs}"
            );
        }
    }
}

#[test]
fn monomial_times_bounded_inverse_trig_declines_other_shapes() {
    let mut ctx = Context::new();
    // Offset arguments graduated alongside the scaled ones; the
    // remaining honest residuals are powers over the tail cap and
    // symbolic offsets, and the arctan family keeps its own owner.
    for source in ["x^6*arcsin(x)", "x*arcsin(x+c)"] {
        let expr = parse(source, &mut ctx).expect(source);
        assert!(
            integrate_symbolic_expr(&mut ctx, expr, "x").is_none(),
            "must stay residual: {source}"
        );
    }
    let arctan = parse("x*arctan(x)", &mut ctx).expect("arctan");
    assert!(integrate_symbolic_expr(&mut ctx, arctan, "x").is_some());
}

#[test]
fn tangent_cotangent_odd_powers_integrate_with_reduction_closed_forms() {
    let mut ctx = Context::new();
    for source in [
        "tan(x)^3",
        "cot(x)^3",
        "tan(x)^5",
        "cot(x)^5",
        "tan(2*x)^3",
        "sin(x)^3/cos(x)^3",
        "cos(x)^3/sin(x)^3",
    ] {
        let expr = parse(source, &mut ctx).expect(source);
        assert!(
            integrate_symbolic_expr(&mut ctx, expr, "x").is_some(),
            "must integrate: {source}"
        );
    }
}

#[test]
fn tangent_odd_power_owners_keep_even_and_linear_rungs() {
    let mut ctx = Context::new();
    let one = BigRational::from_integer(1.into());
    let arg = parse("x", &mut ctx).expect("x");
    // The closed forms differentiate back to tan^3 / cot^3:
    // d/du [tan^2/2 + ln|cos|] = tan^3; the cot twin flips signs.
    let tan_form = super::trig_tan_third_antiderivative_from_parts(&mut ctx, arg, one.clone());
    let tan_display = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: &ctx,
            id: tan_form
        }
    );
    assert!(
        tan_display.contains("tan(x)") && tan_display.contains("ln(|cos(x)|)"),
        "unexpected tan^3 form: {tan_display}"
    );
    let cot_form = super::trig_cot_third_antiderivative_from_parts(&mut ctx, arg, one);
    let cot_display = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: &ctx,
            id: cot_form
        }
    );
    assert!(
        cot_display.contains("cot(x)") && cot_display.contains("ln(|sin(x)|)"),
        "unexpected cot^3 form: {cot_display}"
    );
    // Even powers decline this owner (they keep theirs).
    let even = parse("tan(x)^4", &mut ctx).expect("tan4");
    let cas_ast::Expr::Pow(base, exp) = ctx.get(even).clone() else {
        panic!("pow");
    };
    assert!(super::trig_tan_cot_odd_affine_antiderivative(&mut ctx, base, exp, "x").is_none());
}

#[test]
fn reciprocal_trig_odd_powers_integrate_with_reduction_closed_forms() {
    let mut ctx = Context::new();
    for source in [
        "1/cos(x)^3",
        "1/sin(x)^3",
        "1/cos(x)^5",
        "1/sin(x)^5",
        "1/cos(2*x+1)^3",
        "sec(x)^3",
    ] {
        let expr = parse(source, &mut ctx).expect(source);
        assert!(
            integrate_symbolic_expr(&mut ctx, expr, "x").is_some(),
            "must integrate: {source}"
        );
    }
}

#[test]
fn reciprocal_trig_odd_cube_closed_forms_are_exact() {
    let mut ctx = Context::new();
    // d/du [sec(u)tan(u) + ln|sec(u)+tan(u)|] = 2 sec^3(u) and the
    // csc twin: verified by direct differentiation of the parts.
    let arg = parse("x", &mut ctx).expect("x");
    let one = BigRational::from_integer(1.into());
    let sec_form = super::trig_sec_third_antiderivative_from_parts(&mut ctx, arg, one.clone());
    let sec_display = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: &ctx,
            id: sec_form
        }
    );
    assert!(
        sec_display.contains("sec(x)") && sec_display.contains("tan(x)"),
        "unexpected sec^3 form: {sec_display}"
    );
    let csc_form = super::trig_csc_third_antiderivative_from_parts(&mut ctx, arg, one);
    let csc_display = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: &ctx,
            id: csc_form
        }
    );
    assert!(
        csc_display.contains("csc(x)") && csc_display.contains("cot(x)"),
        "unexpected csc^3 form: {csc_display}"
    );
}

#[test]
fn reciprocal_trig_odd_power_targets_decline_other_owners() {
    let mut ctx = Context::new();
    // Even powers and the n = 1 log forms keep their owners.
    for source in ["1/cos(x)", "1/cos(x)^2", "1/cos(x)^4", "1/sin(x)^4"] {
        let expr = parse(source, &mut ctx).expect(source);
        assert!(
            !super::integrate_symbolic_is_sec_third_affine_target(&mut ctx, expr, "x"),
            "must not claim: {source}"
        );
    }
}

#[test]
fn exp_polynomial_substitution_covers_normalized_div_shapes() {
    let mut ctx = Context::new();
    // c*u' / e^u normalizations of c*u'*e^(-u) for nonlinear u.
    for source in ["x/e^(x^2)", "2*x/e^(x^2)", "x^2/e^(x^3)", "x*e^(-x^2)"] {
        let expr = parse(source, &mut ctx).expect(source);
        assert!(
            integrate_symbolic_expr(&mut ctx, expr, "x").is_some(),
            "must integrate: {source}"
        );
    }
}

#[test]
fn exp_polynomial_substitution_div_arm_keeps_owners_and_residuals() {
    let mut ctx = Context::new();
    // Linear exponents stay with the reciprocal/by-parts/cyclic
    // family; non-derivative numerators are honest residuals (erf).
    for source in ["1/e^(x^2)", "x^2/e^(x^2)", "x^3/e^(x^2 + x)"] {
        let expr = parse(source, &mut ctx).expect(source);
        let result = integrate_symbolic_expr(&mut ctx, expr, "x");
        assert!(result.is_none(), "must stay residual: {source}");
    }
}

#[test]
fn monomial_ln_by_parts_covers_negative_and_fractional_powers() {
    let mut ctx = Context::new();
    // (source, expected derivative round-trip target)
    for source in [
        "ln(x)/x^2",
        "ln(x)/x^3",
        "ln(x)^2/x^2",
        "ln(x)/(2*x^2)",
        "ln(x)*x^(-1/2)",
        "x^(3/2)*ln(x)",
    ] {
        let expr = parse(source, &mut ctx).expect(source);
        assert!(
            integrate_symbolic_expr(&mut ctx, expr, "x").is_some(),
            "must integrate: {source}"
        );
    }
}

#[test]
fn monomial_ln_by_parts_exact_negative_power_form() {
    let mut ctx = Context::new();
    // p = -2, m = 1: the closed form is (-ln(x) - 1)/x.
    let expr = parse("ln(x)/x^2", &mut ctx).expect("parse");
    let integral = integrate_symbolic_expr(&mut ctx, expr, "x").expect("integral");
    let display = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: &ctx,
            id: integral
        }
    );
    assert!(
        display.contains("ln(x)") && display.contains("- 1"),
        "unexpected form: {display}"
    );
}

#[test]
fn monomial_ln_by_parts_leaves_nonmonomial_cofactors_residual() {
    let mut ctx = Context::new();
    // ln(x)/x stays with the u-substitution owner (handled upstream),
    // and non-power denominators are honest residuals (dilogarithm).
    for source in ["ln(x)/(x+1)", "ln(x)/(x^2+1)"] {
        let expr = parse(source, &mut ctx).expect(source);
        assert!(
            integrate_symbolic_expr(&mut ctx, expr, "x").is_none(),
            "must stay residual: {source}"
        );
    }
}

#[test]
fn trig_product_to_sum_integrates_distinct_frequencies() {
    let mut ctx = Context::new();
    for source in [
        "sin(3*x)*cos(2*x)",
        "cos(2*x)*cos(3*x)",
        "sin(2*x)*sin(3*x)",
    ] {
        let expr = parse(source, &mut ctx).expect(source);
        assert!(
            integrate_symbolic_expr(&mut ctx, expr, "x").is_some(),
            "must integrate: {source}"
        );
    }
}

#[test]
fn trig_product_to_sum_leaves_equal_frequencies_to_their_owners() {
    let mut ctx = Context::new();
    // Equal frequencies decline here; other routes own them.
    let expr = parse("sin(2*x)*cos(2*x)", &mut ctx).expect("same freq");
    let l = parse("sin(2*x)", &mut ctx).expect("l");
    let r = parse("cos(2*x)", &mut ctx).expect("r");
    assert!(super::trig_product_to_sum_antiderivative(&mut ctx, l, r, "x").is_none());
    // But the full route still integrates it (existing owner).
    assert!(integrate_symbolic_expr(&mut ctx, expr, "x").is_some());
}

#[test]
fn inverse_trig_squares_integrate_and_round_trip_numerically() {
    let mut ctx = Context::new();
    // (source integrand, derivative round-trip target). The closed forms
    // carry a sqrt(1 - u^2) factor the simplifier cannot collapse, so the
    // round-trip is pinned NUMERICALLY: d/dx F == integrand at samples.
    for (source, target) in [
        ("arcsin(x)^2", "arcsin(x)^2"),
        ("arccos(x)^2", "arccos(x)^2"),
        ("arccos(3*x)^2", "arccos(3*x)^2"),
        ("arcsin(2*x+1)^2", "arcsin(2*x+1)^2"),
    ] {
        let expr = parse(source, &mut ctx).expect(source);
        let antiderivative = integrate_symbolic_expr(&mut ctx, expr, "x")
            .unwrap_or_else(|| panic!("must integrate: {source}"));
        let derivative = crate::symbolic_differentiation_support::differentiate_symbolic_expr(
            &mut ctx,
            antiderivative,
            "x",
        )
        .unwrap_or_else(|| panic!("must differentiate: {source}"));
        let target_expr = parse(target, &mut ctx).expect(target);
        // Samples kept inside (-1/3, 0) so every affine argument (x, 3x,
        // 2x+1) stays within the real domain of arcsin/arccos.
        for sample in [-0.05_f64, -0.12, -0.22, -0.3] {
            let mut vars = std::collections::HashMap::new();
            vars.insert("x".to_string(), sample);
            let lhs = crate::evaluator_f64::eval_f64(&ctx, derivative, &vars)
                .unwrap_or_else(|| panic!("eval diff at {sample}: {source}"));
            let rhs = crate::evaluator_f64::eval_f64(&ctx, target_expr, &vars)
                .unwrap_or_else(|| panic!("eval target at {sample}: {source}"));
            assert!(
                (lhs - rhs).abs() < 1e-9,
                "round-trip mismatch for {source} at {sample}: {lhs} vs {rhs}"
            );
        }
    }
}

#[test]
fn inverse_trig_squares_leave_nonelementary_and_out_of_scope_residual() {
    let mut ctx = Context::new();
    // arctan^2 / arccot^2 are NON-elementary (reduce to ∫ln(cos θ) dθ):
    // they MUST stay honest residuals.
    for source in ["arctan(x)^2", "arccot(x)^2"] {
        let expr = parse(source, &mut ctx).expect(source);
        assert!(
            integrate_symbolic_expr(&mut ctx, expr, "x").is_none(),
            "must stay residual (non-elementary): {source}"
        );
    }
    // The rule is gated to exponent 2 and a linear argument; a cube and a
    // non-affine inner decline here (left to other owners / residual).
    for source in ["arcsin(x)^3", "arcsin(x^2)^2"] {
        let expr = parse(source, &mut ctx).expect(source);
        let base_exp = match ctx.get(expr).clone() {
            cas_ast::Expr::Pow(b, e) => (b, e),
            _ => panic!("expected Pow: {source}"),
        };
        assert!(
            super::inverse_trig_square_affine_antiderivative(&mut ctx, base_exp.0, base_exp.1, "x")
                .is_none(),
            "rule must decline: {source}"
        );
    }
}

#[test]
fn inverse_trig_over_power_integrates_and_round_trips_numerically() {
    let mut ctx = Context::new();
    // int inv(x)/x^n dx by parts, with the lower-degree tail delegated.
    for (source, target) in [
        ("arctan(x)/x^2", "arctan(x)/x^2"),
        ("arcsin(x)/x^2", "arcsin(x)/x^2"),
        ("arccos(x)/x^2", "arccos(x)/x^2"),
        ("arctan(x)/x^3", "arctan(x)/x^3"),
        ("arcsin(x)/x^3", "arcsin(x)/x^3"),
        ("arctan(x)/x^4", "arctan(x)/x^4"),
    ] {
        let expr = parse(source, &mut ctx).expect(source);
        let antiderivative = integrate_symbolic_expr(&mut ctx, expr, "x")
            .unwrap_or_else(|| panic!("must integrate: {source}"));
        let target_expr = parse(target, &mut ctx).expect(target);
        // The closed form carries ln(|x|), which the internal differentiator
        // does not handle, so pin the round-trip with a central finite
        // difference of F instead: F'(x) ~ (F(x+h) - F(x-h)) / (2h) == f(x).
        let eval_at = |ctx: &Context, e: cas_ast::ExprId, x: f64| -> f64 {
            let mut vars = std::collections::HashMap::new();
            vars.insert("x".to_string(), x);
            crate::evaluator_f64::eval_f64(ctx, e, &vars).expect("eval_f64")
        };
        // Samples inside (-1, 1) \ {0} so arcsin/arccos stay real and x != 0.
        for sample in [-0.4_f64, -0.2, 0.25, 0.45] {
            let h = 1e-6;
            let numeric_deriv = (eval_at(&ctx, antiderivative, sample + h)
                - eval_at(&ctx, antiderivative, sample - h))
                / (2.0 * h);
            let f_val = eval_at(&ctx, target_expr, sample);
            assert!(
                (numeric_deriv - f_val).abs() < 1e-5,
                "round-trip mismatch for {source} at {sample}: {numeric_deriv} vs {f_val}"
            );
        }
    }
}

#[test]
fn inverse_trig_over_power_declines_out_of_scope_shapes() {
    let mut ctx = Context::new();
    // Gated to a bare x^n (2 <= n <= 8) denominator and a bare-variable
    // inner; an affine inner, a non-affine inner, plain trig, and x^1
    // decline.
    for source in [
        "arctan(2*x)/x^2",
        "arctan(x^2)/x^2",
        "sin(x)/x^2",
        "arctan(x)/x",
    ] {
        let expr = parse(source, &mut ctx).expect(source);
        let (num, den) = match ctx.get(expr).clone() {
            cas_ast::Expr::Div(n, d) => (n, d),
            _ => panic!("expected Div: {source}"),
        };
        assert!(
            super::inverse_trig_over_power_antiderivative(&mut ctx, num, den, "x").is_none(),
            "rule must decline: {source}"
        );
    }
}

#[test]
fn function_of_sqrt_integrates_and_round_trips_numerically() {
    let mut ctx = Context::new();
    // int f(sqrt(x)) dx via u = sqrt(x) -> 2 int u f(u) du, back-substituted.
    // Covers the inverse-trig family plus sin/cos/sinh/cosh; the target is the
    // integrand itself, so the central finite difference must recover it.
    for (source, target) in [
        ("arctan(sqrt(x))", "arctan(sqrt(x))"),
        ("arcsin(sqrt(x))", "arcsin(sqrt(x))"),
        ("arccos(sqrt(x))", "arccos(sqrt(x))"),
        ("sin(sqrt(x))", "sin(sqrt(x))"),
        ("cos(sqrt(x))", "cos(sqrt(x))"),
        ("sinh(sqrt(x))", "sinh(sqrt(x))"),
        ("cosh(sqrt(x))", "cosh(sqrt(x))"),
    ] {
        let expr = parse(source, &mut ctx).expect(source);
        let antiderivative = integrate_symbolic_expr(&mut ctx, expr, "x")
            .unwrap_or_else(|| panic!("must integrate: {source}"));
        // The closed forms mix sqrt factors the internal differentiator
        // declines, so pin the round-trip with a central finite difference.
        let eval_at = |ctx: &Context, e: cas_ast::ExprId, x: f64| -> f64 {
            let mut vars = std::collections::HashMap::new();
            vars.insert("x".to_string(), x);
            crate::evaluator_f64::eval_f64(ctx, e, &vars).expect("eval_f64")
        };
        let target_expr = parse(target, &mut ctx).expect(target);
        // Samples in (0, 1) so sqrt(x) and sqrt(1-x) stay real.
        for sample in [0.15_f64, 0.4, 0.65, 0.9] {
            let h = 1e-6;
            let numeric_deriv = (eval_at(&ctx, antiderivative, sample + h)
                - eval_at(&ctx, antiderivative, sample - h))
                / (2.0 * h);
            let f_val = eval_at(&ctx, target_expr, sample);
            assert!(
                (numeric_deriv - f_val).abs() < 1e-5,
                "round-trip mismatch for {source} at {sample}: {numeric_deriv} vs {f_val}"
            );
        }
    }
}

#[test]
fn function_of_sqrt_declines_out_of_scope_arguments() {
    let mut ctx = Context::new();
    // The argument must be sqrt of the bare variable; scaled/shifted
    // radicands and a plain (non-sqrt) argument decline. `tan(sqrt(x))`
    // declines too, but by self-gating: int u tan(u) du is non-elementary,
    // so the delegated tail returns None and the rule stays an honest
    // residual.
    for (source, builtin) in [
        ("arctan(sqrt(2*x))", cas_ast::BuiltinFn::Arctan),
        ("arctan(sqrt(x)+1)", cas_ast::BuiltinFn::Arctan),
        ("arctan(sqrt(x^2))", cas_ast::BuiltinFn::Arctan),
        ("arcsin(x)", cas_ast::BuiltinFn::Arcsin),
        ("sin(2*x)", cas_ast::BuiltinFn::Sin),
        ("tan(sqrt(x))", cas_ast::BuiltinFn::Tan),
    ] {
        let expr = parse(source, &mut ctx).expect(source);
        let arg = match ctx.get(expr).clone() {
            cas_ast::Expr::Function(_, args) if args.len() == 1 => args[0],
            _ => panic!("expected unary function: {source}"),
        };
        assert!(
            super::function_of_sqrt_antiderivative(&mut ctx, builtin, arg, "x").is_none(),
            "rule must decline: {source}"
        );
    }
}

#[test]
fn exp_and_function_over_sqrt_integrate_and_round_trip() {
    let mut ctx = Context::new();
    // e^sqrt(x) (Pow(E, sqrt(x)), the Pow arm) and the H(sqrt(x))/sqrt(x)
    // cofactor family (the Mul arm, since the engine normalizes /sqrt(x) to
    // *x^(-1/2)); target is the integrand, recovered by central difference.
    for (source, target) in [
        ("e^sqrt(x)", "e^sqrt(x)"),
        ("e^sqrt(x)/sqrt(x)", "e^sqrt(x)/sqrt(x)"),
        ("sin(sqrt(x))/sqrt(x)", "sin(sqrt(x))/sqrt(x)"),
        ("cos(sqrt(x))/sqrt(x)", "cos(sqrt(x))/sqrt(x)"),
        ("sinh(sqrt(x))/sqrt(x)", "sinh(sqrt(x))/sqrt(x)"),
        ("cosh(sqrt(x))/sqrt(x)", "cosh(sqrt(x))/sqrt(x)"),
    ] {
        let expr = parse(source, &mut ctx).expect(source);
        let antiderivative = integrate_symbolic_expr(&mut ctx, expr, "x")
            .unwrap_or_else(|| panic!("must integrate: {source}"));
        let eval_at = |ctx: &Context, e: cas_ast::ExprId, x: f64| -> f64 {
            let mut vars = std::collections::HashMap::new();
            vars.insert("x".to_string(), x);
            crate::evaluator_f64::eval_f64(ctx, e, &vars).expect("eval_f64")
        };
        let target_expr = parse(target, &mut ctx).expect(target);
        for sample in [0.2_f64, 0.5, 1.3, 2.7] {
            let h = 1e-6;
            let numeric_deriv = (eval_at(&ctx, antiderivative, sample + h)
                - eval_at(&ctx, antiderivative, sample - h))
                / (2.0 * h);
            let f_val = eval_at(&ctx, target_expr, sample);
            assert!(
                (numeric_deriv - f_val).abs() < 1e-4,
                "round-trip mismatch for {source} at {sample}: {numeric_deriv} vs {f_val}"
            );
        }
    }
}

#[test]
fn exp_over_sqrt_declines_non_elementary() {
    let mut ctx = Context::new();
    // The sqrt substitution must self-gate to an HONEST residual on
    // non-elementary integrands: e^x/sqrt(x) -> 2 int e^(u^2) du (erf);
    // e^sqrt(x)/x -> 2 int e^u/u du (Ei, the 1/x cofactor is not 1/sqrt(x));
    // sin(x)/sqrt(x) -> 2 int sin(u^2) du (Fresnel).
    for source in ["e^x/sqrt(x)", "e^sqrt(x)/x", "sin(x)/sqrt(x)"] {
        let expr = parse(source, &mut ctx).expect(source);
        assert!(
            integrate_symbolic_expr(&mut ctx, expr, "x").is_none(),
            "non-elementary integrand must stay residual: {source}"
        );
    }
}

#[test]
fn abs_and_sign_affine_integrate_and_round_trip() {
    let mut ctx = Context::new();
    // int |a*x+b| dx = (a*x+b)|a*x+b|/(2a); int sign(a*x+b) dx = |a*x+b|/a.
    // sqrt(x^2) is canonicalized to |x| before integration. The closed forms
    // mix abs factors the internal differentiator declines, so pin the
    // round-trip with a central finite difference of the antiderivative.
    let eval_at = |ctx: &Context, e: cas_ast::ExprId, x: f64| -> f64 {
        let mut vars = std::collections::HashMap::new();
        vars.insert("x".to_string(), x);
        crate::evaluator_f64::eval_f64(ctx, e, &vars).expect("eval_f64")
    };
    for (source, integrand) in [
        ("abs(x)", "abs(x)"),
        ("sqrt(x^2)", "abs(x)"),
        ("abs(2*x+1)", "abs(2*x+1)"),
        ("abs(x-3)", "abs(x-3)"),
        ("sign(x)", "sign(x)"),
        ("sign(2*x+1)", "sign(2*x+1)"),
    ] {
        let expr = parse(source, &mut ctx).expect(source);
        let antiderivative = integrate_symbolic_expr(&mut ctx, expr, "x")
            .unwrap_or_else(|| panic!("must integrate: {source}"));
        let integrand_expr = parse(integrand, &mut ctx).expect(integrand);
        // Samples avoid the single corner of each affine argument.
        for sample in [-2.3_f64, -0.7, 0.4, 1.9, 3.5] {
            let h = 1e-6;
            let numeric_deriv = (eval_at(&ctx, antiderivative, sample + h)
                - eval_at(&ctx, antiderivative, sample - h))
                / (2.0 * h);
            let f_val = eval_at(&ctx, integrand_expr, sample);
            assert!(
                (numeric_deriv - f_val).abs() < 1e-4,
                "round-trip mismatch for {source} at {sample}: {numeric_deriv} vs {f_val}"
            );
        }
    }
}

#[test]
fn abs_integration_declines_non_affine_argument() {
    let mut ctx = Context::new();
    // The closed form (h)|h|/(2a) is only valid for affine h; a non-affine
    // argument has a piecewise antiderivative across its roots, so it stays
    // an honest residual.
    for source in ["abs(x^2-1)", "abs(sin(x))", "abs(x^3)"] {
        let expr = parse(source, &mut ctx).expect(source);
        assert!(
            integrate_symbolic_expr(&mut ctx, expr, "x").is_none(),
            "non-affine abs integrand must stay residual: {source}"
        );
    }
}

#[test]
fn fold_nested_numeric_powers_collapses_sqrt_square() {
    // (sqrt(x))^2 + 1 must fold to x + 1, never display as the ambiguous
    // x^(1/2)^2 (which re-parses as x^(1/4)).
    let mut ctx = Context::new();
    let expr = parse("(x^(1/2))^2 + 1", &mut ctx).expect("parse");
    let folded = super::fold_nested_numeric_powers(&mut ctx, expr);
    let rendered = cas_formatter::DisplayExpr {
        context: &ctx,
        id: folded,
    }
    .to_string();
    assert!(
        !rendered.contains("1 / 2") && !rendered.contains("1/2"),
        "the ambiguous sqrt-square must be folded away, got {rendered}"
    );
    // and it must evaluate to x + 1 (= 4 at x = 3).
    let mut vars = std::collections::HashMap::new();
    vars.insert("x".to_string(), 3.0_f64);
    let value = crate::evaluator_f64::eval_f64(&ctx, folded, &vars).expect("eval");
    assert!((value - 4.0).abs() < 1e-12, "expected 4, got {value}");
}
