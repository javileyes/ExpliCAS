use cas_ast::{Context, Expr, ExprId};
use cas_formatter::DisplayExpr;
use cas_parser::parse;
use cas_session::SessionState;
use cas_solver::api::render_conditions_normalized;
use cas_solver::runtime::{
    Engine, EvalAction, EvalRequest, EvalResult, ImportanceLevel, Simplifier, StepsMode,
};
use serde_json::Value;
use std::process::Command;

fn render_expr(ctx: &Context, id: ExprId) -> String {
    format!("{}", DisplayExpr { context: ctx, id })
}

fn cli_eval_json_with_stderr(input: &str) -> (Value, String) {
    cli_eval_json_with_stderr_args(input, &[])
}

fn cli_eval_json_with_stderr_args(input: &str, extra_args: &[&str]) -> (Value, String) {
    let output = Command::new(env!("CARGO_BIN_EXE_cas_cli"))
        .args(["eval", input])
        .args(extra_args)
        .args(["--format", "json"])
        .output()
        .expect("execute cas_cli");
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    assert!(
        output.status.success(),
        "cas_cli failed for {input}\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );

    let wire = serde_json::from_str(stdout.trim())
        .unwrap_or_else(|err| panic!("parse CLI JSON for {input}: {err}\nstdout:\n{stdout}"));
    (wire, stderr)
}

fn assert_u_du_substep_labels(substeps: &[Value], input: &str) {
    let u_du_substep = substeps
        .iter()
        .find(|substep| substep["title"] == "Identificar u y du")
        .expect("expected concrete u/du substep");
    let before_latex = u_du_substep["before_latex"]
        .as_str()
        .expect("u/du substep should expose before_latex");
    let after_latex = u_du_substep["after_latex"]
        .as_str()
        .expect("u/du substep should expose after_latex");
    assert!(
        before_latex.contains("u =") && after_latex.contains("du ="),
        "u/du substep should label substitution evidence for {input}, got {u_du_substep:?}"
    );
}

fn simplified_integral(input: &str) -> String {
    let mut simplifier = Simplifier::with_default_rules();
    simplifier.disable_rule("Double Angle Identity");
    let expr = parse(input, &mut simplifier.context).expect("parse integration input");
    let (result, _) = simplifier.simplify(expr);
    render_expr(&simplifier.context, result)
}

fn rationalize_rewrites_for_simplify(input: &str) -> usize {
    let mut simplifier = Simplifier::with_default_rules();
    simplifier.disable_rule("Double Angle Identity");
    let expr = parse(input, &mut simplifier.context).expect("parse input");
    let (_, _, stats) = simplifier.simplify_with_stats(expr, Default::default());
    stats.rationalize.rewrites_used
}

fn explicit_integrate_call_parts(ctx: &Context, expr: ExprId) -> (ExprId, String) {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        panic!(
            "expected integrate(...) call, got {}",
            render_expr(ctx, expr)
        );
    };
    assert_eq!(
        ctx.sym_name(*fn_id),
        "integrate",
        "expected integrate(...) call, got {}",
        render_expr(ctx, expr)
    );
    assert_eq!(
        args.len(),
        2,
        "antiderivative verification currently requires explicit integrate(expr, var)"
    );

    let var = args[1];
    let Expr::Variable(sym_id) = ctx.get(var) else {
        panic!(
            "antiderivative verification requires a variable integration target, got {}",
            render_expr(ctx, var)
        );
    };

    (args[0], ctx.sym_name(*sym_id).to_string())
}

#[derive(Debug, PartialEq, Eq)]
enum AntiderivativeVerificationRoute {
    PublicResidual,
    InternalDerivative,
}

fn should_verify_antiderivative_with_public_integrate_residual(
    ctx: &mut Context,
    integrand: ExprId,
    var_name: &str,
) -> bool {
    cas_math::symbolic_integration_support::integrate_symbolic_is_polynomial_times_exp_linear_target(
        ctx, integrand, var_name,
    ) || cas_math::symbolic_integration_support::integrate_symbolic_is_exp_trig_same_linear_target(
        ctx, integrand, var_name,
    ) || cas_math::symbolic_integration_support::integrate_symbolic_is_linear_times_trig_linear_target(
        ctx, integrand, var_name,
    ) || cas_math::symbolic_integration_support::integrate_symbolic_is_polynomial_times_trig_linear_target(
        ctx, integrand, var_name,
    ) || cas_math::symbolic_integration_support::integrate_symbolic_is_linear_times_hyperbolic_linear_target(
        ctx, integrand, var_name,
    ) || cas_math::symbolic_integration_support::integrate_symbolic_is_polynomial_times_hyperbolic_linear_target(
        ctx, integrand, var_name,
    ) || cas_math::symbolic_integration_support::integrate_symbolic_is_hyperbolic_quotient_substitution_target(
        ctx, integrand, var_name,
    ) || cas_math::symbolic_integration_support::integrate_symbolic_is_affine_trig_seventh_target(
        ctx, integrand, var_name,
    ) || cas_math::symbolic_integration_support::integrate_symbolic_is_tan_fourth_affine_target(
        ctx, integrand, var_name,
    ) || cas_math::symbolic_integration_support::integrate_symbolic_is_cot_fourth_affine_target(
        ctx, integrand, var_name,
    ) || cas_math::symbolic_integration_support::integrate_symbolic_is_tan_sixth_affine_target(
        ctx, integrand, var_name,
    ) || cas_math::symbolic_integration_support::integrate_symbolic_is_cot_sixth_affine_target(
        ctx, integrand, var_name,
    ) || cas_math::symbolic_integration_support::integrate_symbolic_is_tan_eighth_affine_target(
        ctx, integrand, var_name,
    ) || cas_math::symbolic_integration_support::integrate_symbolic_is_cot_eighth_affine_target(
        ctx, integrand, var_name,
    ) || cas_math::symbolic_integration_support::integrate_symbolic_is_sec_fourth_affine_target(
        ctx, integrand, var_name,
    ) || cas_math::symbolic_integration_support::integrate_symbolic_is_csc_fourth_affine_target(
        ctx, integrand, var_name,
    ) || cas_math::symbolic_integration_support::integrate_symbolic_is_sec_sixth_affine_target(
        ctx, integrand, var_name,
    ) || cas_math::symbolic_integration_support::integrate_symbolic_is_csc_sixth_affine_target(
        ctx, integrand, var_name,
    ) || cas_math::symbolic_integration_support::integrate_symbolic_is_sec_eighth_affine_target(
        ctx, integrand, var_name,
    ) || cas_math::symbolic_integration_support::integrate_symbolic_is_csc_eighth_affine_target(
        ctx, integrand, var_name,
    ) || cas_math::symbolic_integration_support::integrate_symbolic_is_polynomial_times_arctan_affine_target(
        ctx, integrand, var_name,
    ) || cas_math::symbolic_integration_support::integrate_symbolic_is_positive_quadratic_cube_target(
        ctx, integrand, var_name,
    ) || cas_math::symbolic_integration_support::integrate_symbolic_is_high_log_power_product_substitution_target(
        ctx, integrand, var_name,
    ) || cas_math::symbolic_integration_support::integrate_symbolic_is_verifiable_log_power_product_substitution_target(
        ctx, integrand, var_name,
    ) || cas_math::symbolic_integration_support::integrate_symbolic_is_polynomial_log_reciprocal_derivative_target(
        ctx, integrand, var_name,
    ) || cas_math::symbolic_integration_support::integrate_symbolic_is_trig_log_substitution_target(
        ctx, integrand, var_name,
    ) || cas_math::symbolic_integration_support::integrate_symbolic_is_affine_sqrt_product_derivative_target(
        ctx, integrand, var_name,
    ) || cas_math::symbolic_integration_support::integrate_symbolic_is_acosh_polynomial_substitution_target(
        ctx, integrand, var_name,
    ) || cas_math::symbolic_integration_support::integrate_symbolic_is_monomial_times_ln_var_by_parts_target(
        ctx, integrand, var_name,
    ) || cas_math::symbolic_integration_support::integrate_symbolic_is_linear_times_affine_ln_by_parts_target(
        ctx, integrand, var_name,
    ) || cas_math::symbolic_integration_support::integrate_symbolic_is_quadratic_times_affine_ln_by_parts_target(
        ctx, integrand, var_name,
    ) || cas_math::symbolic_integration_support::integrate_symbolic_is_quadratic_times_positive_quadratic_ln_by_parts_target(
        ctx, integrand, var_name,
    ) || cas_math::symbolic_integration_support::integrate_symbolic_is_arctan_sqrt_var_unit_shift_square_target(
        ctx, integrand, var_name,
    ) || cas_math::symbolic_integration_support::integrate_symbolic_is_affine_hyperbolic_cubic_target(
        ctx, integrand, var_name,
    ) || cas_math::symbolic_integration_support::integrate_symbolic_is_affine_hyperbolic_fifth_target(
        ctx, integrand, var_name,
    ) || cas_math::symbolic_integration_support::integrate_symbolic_is_affine_hyperbolic_seventh_target(
        ctx, integrand, var_name,
    )
}

fn assert_antiderivative_verifies(input: &str) -> AntiderivativeVerificationRoute {
    let mut simplifier = Simplifier::with_default_rules();
    simplifier.disable_rule("Double Angle Identity");
    let expr = parse(input, &mut simplifier.context).expect("parse integration input");
    let (integrand, var_name) = explicit_integrate_call_parts(&simplifier.context, expr);

    if should_verify_antiderivative_with_public_integrate_residual(
        &mut simplifier.context,
        integrand,
        &var_name,
    ) {
        let public_residual = integrate_call_antiderivative_residual_result(input);
        if public_residual == "0" {
            return AntiderivativeVerificationRoute::PublicResidual;
        }
    }

    let (antiderivative, _) = simplifier.simplify(expr);
    let antiderivative_rendered = render_expr(&simplifier.context, antiderivative);
    let var = simplifier.context.var(&var_name);
    let diff_call = simplifier.context.call("diff", vec![antiderivative, var]);
    let (derivative, _) = simplifier.simplify(diff_call);
    let (expected_integrand, _) = simplifier.simplify(integrand);
    let residual = simplifier
        .context
        .add(Expr::Sub(derivative, expected_integrand));
    let (residual, _) = simplifier.simplify(residual);
    let residual_rendered = render_expr(&simplifier.context, residual);
    if residual_rendered == "0" {
        return AntiderivativeVerificationRoute::InternalDerivative;
    }

    let public_residual = rendered_antiderivative_residual_result(input, &antiderivative_rendered);
    if public_residual == "0" {
        return AntiderivativeVerificationRoute::PublicResidual;
    }

    panic!(
        "antiderivative verification failed for {input}\nintegral result: {}\nderivative: {}\nexpected integrand: {}\npublic residual: {}",
        antiderivative_rendered,
        render_expr(&simplifier.context, derivative),
        render_expr(&simplifier.context, expected_integrand),
        public_residual,
    );
}

fn assert_rendered_antiderivative_verifies(input: &str, rendered_antiderivative: &str) {
    let result = rendered_antiderivative_residual_result(input, rendered_antiderivative);

    assert_eq!(
        result,
        "0",
        "antiderivative verification failed for {input}\nintegral result: {rendered_antiderivative}"
    );
}

fn integrate_call_antiderivative_residual_result(input: &str) -> String {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let expr = parse(input, &mut engine.simplifier.context).expect("parse integration input");
    let (integrand, var_name) = explicit_integrate_call_parts(&engine.simplifier.context, expr);
    let var = engine.simplifier.context.var(&var_name);
    let diff_call = engine.simplifier.context.call("diff", vec![expr, var]);
    let residual = engine
        .simplifier
        .context
        .add(Expr::Sub(diff_call, integrand));

    let output = engine
        .eval(
            &mut state,
            EvalRequest {
                raw_input: format!("diff({input}, {var_name}) - integrand"),
                parsed: residual,
                action: EvalAction::Simplify,
                auto_store: false,
            },
        )
        .expect("eval derivative residual");

    match output.result {
        EvalResult::Expr(expr) => render_expr(&engine.simplifier.context, expr),
        other => panic!("expected expression result, got {other:?}"),
    }
}

fn rendered_antiderivative_residual_result(input: &str, rendered_antiderivative: &str) -> String {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let expr = parse(input, &mut engine.simplifier.context).expect("parse integration input");
    let (integrand, var_name) = explicit_integrate_call_parts(&engine.simplifier.context, expr);

    let antiderivative = parse(rendered_antiderivative, &mut engine.simplifier.context)
        .expect("parse antiderivative");
    let var = engine.simplifier.context.var(&var_name);
    let diff_call = engine
        .simplifier
        .context
        .call("diff", vec![antiderivative, var]);
    let residual = engine
        .simplifier
        .context
        .add(Expr::Sub(diff_call, integrand));

    let output = engine
        .eval(
            &mut state,
            EvalRequest {
                raw_input: format!("diff({rendered_antiderivative}, {var_name}) - integrand"),
                parsed: residual,
                action: EvalAction::Simplify,
                auto_store: false,
            },
        )
        .expect("eval derivative residual");

    match output.result {
        EvalResult::Expr(expr) => render_expr(&engine.simplifier.context, expr),
        other => panic!("expected expression result, got {other:?}"),
    }
}

const REPRESENTATIVE_ANTIDERIVATIVE_VERIFICATION_CASES: &[&str] = &[
    "integrate(2*x + 3, x)",
    "integrate(sin(2*x), x)",
    "integrate(x^2*sin(x), x)",
    "integrate(x^2*cos(x), x)",
    "integrate(x^3*sin(x), x)",
    "integrate(x^3*cos(x), x)",
    "integrate(x^5*sin(x), x)",
    "integrate(x^5*cos(x), x)",
    "integrate(x^6*sin(x), x)",
    "integrate(x^6*cos(x), x)",
    "integrate((x^6+1)*sin(2*x+1), x)",
    "integrate((x^6+1)*cos(2*x+1), x)",
    "integrate(x^7*sin(x), x)",
    "integrate(x^7*cos(x), x)",
    "integrate((x^7+1)*sin(2*x+1), x)",
    "integrate((x^7+1)*cos(2*x+1), x)",
    "integrate((2*x+3)*exp(2*x+1), x)",
    "integrate(x^2*exp(x), x)",
    "integrate(x^3*exp(x), x)",
    "integrate(x^5*exp(x), x)",
    "integrate((x^3+x)*exp(2*x+1), x)",
    "integrate((x^6+1)*exp(2*x+1), x)",
    "integrate(x^7*exp(x), x)",
    "integrate((x^7+1)*exp(2*x+1), x)",
    "integrate(x^2*sinh(x), x)",
    "integrate(x^2*cosh(x), x)",
    "integrate((x^3+x)*sinh(2*x+1), x)",
    "integrate((x^3+x)*cosh(2*x+1), x)",
    "integrate(x^6*sinh(x), x)",
    "integrate(x^6*cosh(x), x)",
    "integrate((x^6+1)*sinh(2*x+1), x)",
    "integrate((x^6+1)*cosh(2*x+1), x)",
    "integrate(x^7*sinh(x), x)",
    "integrate(x^7*cosh(x), x)",
    "integrate((x^7+1)*sinh(2*x+1), x)",
    "integrate((x^7+1)*cosh(2*x+1), x)",
    "integrate(sinh(x)^2*cosh(x), x)",
    "integrate(2*x*exp(x^2), x)",
    "integrate(cosh(x)/(1+sinh(x)^2), x)",
    "integrate(2*cosh(2*x+1)/(1+sinh(2*x+1)^2), x)",
    "integrate(sinh(2*x + 1)/cosh(2*x + 1), x)",
    "integrate(1/cosh(2*x + 1)^2, x)",
    "integrate(ln(2*x+1), x)",
    "integrate(1/(2*x + 1), x)",
    "integrate(1/(x^2+1), x)",
    // G1 Cap. A: real-root quadratic factor (x^2-2) renders a real-log ratio;
    // the sqrt(2) coefficient folds under the differentiate-back verifier.
    "integrate(1/(x^4-4), x)",
    // G1 Cap. B: irreducible even quartic as a factor (x^4-x^2+1 in x^6+1, x^4+1
    // in x^8-1) integrates via the surd symmetric split; the constant-numerator
    // targets fold under the differentiate-back verifier.
    "integrate(1/(x^6+1), x)",
    "integrate(1/(x^8-1), x)",
    "integrate(1/(4*x^2+1)^2, x)",
    "integrate(1/(2*sqrt(x)*(x+1)), x)",
    "integrate(arcsin(2*x+1), x)",
    "integrate(asinh(2*x+1), x)",
    "integrate(1/(x^2-1), x)",
    "integrate((2*x+1)/(x^2+x-1)^3, x)",
    "integrate(2*x/sqrt(4-x^4), x)",
    "integrate(1/sqrt(4-(x+1)^2), x)",
    "integrate(1/sqrt(4+(x+1)^2), x)",
    "integrate(sin(x)*cos(x), x)",
    "integrate(sin(x)^2*cos(x), x)",
    "integrate(sin(2*x + 1)^3, x)",
    "integrate(cos(2*x + 1)^3, x)",
    "integrate(sinh(2*x + 1)^3, x)",
    "integrate(cosh(2*x + 1)^3, x)",
    "integrate(sinh(2*x + 1)^5, x)",
    "integrate(cosh(2*x + 1)^5, x)",
    "integrate(sec(2*x + 1)^2, x)",
    "integrate(sec(2*x + 1)*tan(2*x + 1), x)",
    "integrate(sec(x)^2*tan(x), x)",
    "integrate(tan(x)^2/cos(x)^2, x)",
];

fn assert_antiderivative_equiv_verifies(input: &str) {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let expr = parse(input, &mut engine.simplifier.context).expect("parse integration input");
    let (integrand, var_name) = explicit_integrate_call_parts(&engine.simplifier.context, expr);

    let integral_output = engine
        .eval(
            &mut state,
            EvalRequest {
                raw_input: input.to_string(),
                parsed: expr,
                action: EvalAction::Simplify,
                auto_store: false,
            },
        )
        .expect("eval integral");
    let antiderivative = match integral_output.result {
        EvalResult::Expr(expr) => expr,
        other => panic!("expected expression result, got {other:?}"),
    };

    let var = engine.simplifier.context.var(&var_name);
    let diff_call = engine
        .simplifier
        .context
        .call("diff", vec![antiderivative, var]);
    let verify_output = engine
        .eval(
            &mut state,
            EvalRequest {
                raw_input: format!("diff(integrate(...), {var_name}) equiv integrand"),
                parsed: diff_call,
                action: EvalAction::Equiv { other: integrand },
                auto_store: false,
            },
        )
        .expect("eval derivative equivalence");
    let result = match verify_output.result {
        EvalResult::Bool(value) => value,
        other => panic!("expected bool result, got {other:?}"),
    };

    assert!(
        result,
        "antiderivative equivalence verification failed for {input}\nintegral result: {}",
        render_expr(&engine.simplifier.context, antiderivative)
    );
}

fn evaluated_equiv_with_required_conditions(lhs: &str, rhs: &str) -> (bool, Vec<String>) {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let parsed = parse(lhs, &mut engine.simplifier.context).expect("parse lhs");
    let other = parse(rhs, &mut engine.simplifier.context).expect("parse rhs");

    let output = engine
        .eval(
            &mut state,
            EvalRequest {
                raw_input: format!("{lhs} equiv {rhs}"),
                parsed,
                action: EvalAction::Equiv { other },
                auto_store: false,
            },
        )
        .expect("eval equivalence");
    let result = match output.result {
        EvalResult::Bool(value) => value,
        other => panic!("expected bool result, got {other:?}"),
    };
    let required =
        render_conditions_normalized(&mut engine.simplifier.context, &output.required_conditions);

    (result, required)
}

fn evaluated_expr_with_required_conditions(input: &str) -> (String, Vec<String>) {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let parsed = parse(input, &mut engine.simplifier.context).expect("parse input");

    let req = EvalRequest {
        raw_input: input.to_string(),
        parsed,
        action: EvalAction::Simplify,
        auto_store: false,
    };

    let output = engine.eval(&mut state, req).expect("eval failed");
    let result = match output.result {
        EvalResult::Expr(expr) => format!(
            "{}",
            DisplayExpr {
                context: &engine.simplifier.context,
                id: expr,
            }
        ),
        other => panic!("expected expression result, got {other:?}"),
    };
    let required =
        render_conditions_normalized(&mut engine.simplifier.context, &output.required_conditions);

    (result, required)
}

fn evaluated_expr_with_required_conditions_and_blocked_count(
    input: &str,
) -> (String, Vec<String>, usize) {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let parsed = parse(input, &mut engine.simplifier.context).expect("parse input");

    let output = engine
        .eval(
            &mut state,
            EvalRequest {
                raw_input: input.to_string(),
                parsed,
                action: EvalAction::Simplify,
                auto_store: false,
            },
        )
        .expect("eval failed");
    let result = match output.result {
        EvalResult::Expr(expr) => render_expr(&engine.simplifier.context, expr),
        other => panic!("expected expression result, got {other:?}"),
    };
    let required =
        render_conditions_normalized(&mut engine.simplifier.context, &output.required_conditions);

    (result, required, output.blocked_hints.len())
}

fn evaluated_integral_with_required_conditions(input: &str) -> (String, Vec<String>) {
    evaluated_expr_with_required_conditions(input)
}

fn evaluated_integral_step_rules(input: &str) -> Vec<String> {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().steps_mode = StepsMode::On;
    let parsed = parse(input, &mut engine.simplifier.context).expect("parse integration input");

    let output = engine
        .eval(
            &mut state,
            EvalRequest {
                raw_input: input.to_string(),
                parsed,
                action: EvalAction::Simplify,
                auto_store: false,
            },
        )
        .expect("eval failed");

    output
        .steps
        .iter()
        .map(|step| step.rule_name.to_string())
        .collect()
}

fn evaluated_expr_step_summaries(input: &str) -> Vec<(String, String, ImportanceLevel)> {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().steps_mode = StepsMode::On;
    let parsed = parse(input, &mut engine.simplifier.context).expect("parse input");

    let output = engine
        .eval(
            &mut state,
            EvalRequest {
                raw_input: input.to_string(),
                parsed,
                action: EvalAction::Simplify,
                auto_store: false,
            },
        )
        .expect("eval failed");

    output
        .steps
        .iter()
        .map(|step| {
            (
                step.description.to_string(),
                step.rule_name.to_string(),
                step.get_importance(),
            )
        })
        .collect()
}

#[test]
fn integrate_contract_invalid_real_domain_integrands_return_undefined() {
    for input in [
        "integrate(log(1,2), x)",
        "integrate(log(1,x), x)",
        "integrate(log(-2,x), x)",
        "integrate(ln(0), x)",
        "integrate(sqrt(-1), x)",
        "integrate(infinity, x)",
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);
        assert!(
            stderr.is_empty(),
            "invalid real-domain integrand should not emit stderr for {input}: {stderr}"
        );
        assert_eq!(
            wire["result"].as_str(),
            Some("undefined"),
            "invalid real-domain integrand should not produce a formal primitive for {input}: {wire:?}"
        );
        assert_eq!(
            wire["required_display"]
                .as_array()
                .expect("required_display should be an array")
                .len(),
            0,
            "undefined integrands should not introduce conditional assumptions for {input}"
        );

        let step_text = wire["steps"].to_string();
        assert!(
            step_text.contains("undefined"),
            "integration trace should expose the undefined result for {input}: {step_text}"
        );
        assert!(
            !step_text.contains("x·log") && !step_text.contains("infinity·x"),
            "integration trace should not present a false formal primitive for {input}: {step_text}"
        );
    }

    let (result, required) = evaluated_integral_with_required_conditions("integrate(2, x)");
    assert_eq!(result, "2 * x");
    assert!(
        required.is_empty(),
        "finite constants should still integrate without domain assumptions: {required:?}"
    );
}

#[test]
fn integrate_contract_supported_antiderivatives_verify_by_differentiation() {
    for input in REPRESENTATIVE_ANTIDERIVATIVE_VERIFICATION_CASES {
        assert_antiderivative_verifies(input);
    }
}

/// G1 sub-cycle Cap. A (2026-07-14): a rational denominator whose squarefree
/// factorization over ℚ leaves a quadratic with a POSITIVE discriminant (real
/// irrational roots, e.g. `x^2 - 2` inside `x^4 - 4 = (x^2-2)(x^2+2)`) now
/// integrates to a real-log ratio `ln|(x-√2)/(x+√2)|` instead of declining. The
/// irreducible (Δ<0) factor still renders as arctan, unchanged.
/// See docs/G1_RATIONAL_INTEGRATION_SCOPING.md.
///
/// The `x^4-4` case is also in REPRESENTATIVE_ANTIDERIVATIVE_VERIFICATION_CASES,
/// where its antiderivative is confirmed by differentiate-back. The odd-surd
/// variants (`x^4-9`, the `(x^2-3)` factor) are equally correct — verified
/// numerically — but the differentiate-back simplifier does not yet fold
/// `sqrt(3)*sqrt(3)` inside the rational cancellation, so they are asserted here
/// by render form + support only (the surd self-verification is a simplifier
/// residual, not an integration one — no wrong answer is emitted).
#[test]
fn integrate_contract_real_root_quadratic_factor_renders_real_log_ratio() {
    for input in [
        "integrate(1/(x^4-4), x)",
        "integrate(1/(x^4-9), x)",
        "integrate(1/((x^2-2)*(x^2+1)), x)",
        "integrate(1/((x^2-2)*(x^2-3)), x)",
        "integrate(x/(x^4-4), x)",
    ] {
        let (result, _required) = evaluated_integral_with_required_conditions(input);
        assert!(
            !result.contains("integrate("),
            "should no longer be residual: {input} -> {result}"
        );
        // Real-root factor contributes a log ratio (ln of an absolute value).
        assert!(
            result.contains("ln(|"),
            "expected a real-log ratio term for {input} -> {result}"
        );
    }

    // The mixed denominator keeps the arctan term for its irreducible factor.
    let (mixed, _) =
        evaluated_integral_with_required_conditions("integrate(1/((x^2-2)*(x^2+1)), x)");
    assert!(
        mixed.contains("arctan"),
        "irreducible factor should still render arctan: {mixed}"
    );
}

/// Honest-residual contract for the later G1 sub-cycles: denominators needing
/// factorization over an algebraic extension not yet covered (√5-quadratics,
/// cube roots, deeper even quartics) must stay residual — declining, never
/// emitting a wrong antiderivative.
#[test]
fn integrate_contract_algebraic_extension_denominators_stay_residual() {
    for input in [
        "integrate(1/(x^5-1), x)", // Φ5 needs √5-quadratics (Cap. C)
        "integrate(1/(x^3-2), x)", // needs ∛2 (Cap. D)
        "integrate(1/(x^8+1), x)", // resolvent u^4+1 needs a surd factorization
        "integrate(1/(x^4-2), x)", // resolvent root √2 is itself irrational
    ] {
        let (result, _required) = evaluated_integral_with_required_conditions(input);
        assert!(
            result.contains("integrate("),
            "should stay an honest residual until its sub-cycle lands: {input} -> {result}"
        );
    }
}

/// G1 sub-cycle Cap. B (2026-07-14): an irreducible-over-ℝ even quartic
/// `x^4 + p*x^2 + r` appearing as a FACTOR (e.g. `x^4-x^2+1` in `x^6+1`,
/// `x^4+1` in `x^8-1`) is kept whole so the partial-fraction residue solve stays
/// over ℚ; the surd split `(x^2+a*x+s)(x^2-a*x+s)` appears only in the render.
/// See docs/G1_RATIONAL_INTEGRATION_SCOPING.md. The named constant-numerator
/// targets are confirmed by differentiate-back (they are also in
/// REPRESENTATIVE_ANTIDERIVATIVE_VERIFICATION_CASES); odd/even numerator variants
/// are verified numerically here by support + render form (the combined-surd
/// differentiate-back is verifier-limited, a documented residual, never a wrong
/// answer).
#[test]
fn integrate_contract_irreducible_even_quartic_factor_integrates_via_surd_split() {
    // Named G1 exit probes plus odd-numerator variants routed through the surd
    // even-quartic render (x^2/(x^6+1) takes the cleaner u=x^3 substitution and
    // is covered elsewhere).
    for input in [
        "integrate(1/(x^6+1), x)",
        "integrate(1/(x^8-1), x)",
        "integrate(x^3/(x^4-x^2+1), x)",
        "integrate(x/(x^6+1), x)",
    ] {
        let (result, _required) = evaluated_integral_with_required_conditions(input);
        assert!(
            !result.contains("integrate("),
            "should integrate via the even-quartic surd split: {input} -> {result}"
        );
        assert!(
            result.contains("sqrt(3)") || result.contains("sqrt(2)"),
            "expected a surd coefficient in the even-quartic render: {input} -> {result}"
        );
    }

    // The standalone constant-numerator even quartic stays owned by the earlier
    // symmetric-surd closed form (byte-identical), not the general route.
    let (standalone, _) =
        evaluated_integral_with_required_conditions("integrate(1/(x^4-x^2+1), x)");
    assert!(
        standalone.contains("arctan(sqrt(3) + 2") || standalone.contains("arctan(sqrt(3)+2"),
        "standalone even quartic keeps its symmetric-surd render: {standalone}"
    );
}

/// G1 verification-budget lift (2026-07-14): a GENERAL (non-constant) numerator
/// over an even-quartic factor produces a correct antiderivative whose combined
/// surd render is larger than the constant-numerator case. The exact algebraic
/// zero-test (`algebraic_rational_zero_test`) previously exceeded its node/term
/// budget on that residual and declined; the raised budget lets it verify (still
/// an EXACT decision procedure — a larger budget only decides bigger inputs, never
/// a false positive), so these now emit. Numerically confirmed correct.
#[test]
fn integrate_contract_general_numerator_even_quartic_now_verifies() {
    for input in [
        "integrate((x^3+5)/(x^6+1), x)",
        "integrate((x^3-x)/(x^6+1), x)",
        "integrate((3*x^3-2*x+5)/(x^6+1), x)",
        "integrate((x^2+1)/(x^4-x^2+1), x)",
        "integrate((2*x^2+3)/(x^8-1), x)",
    ] {
        let (result, _required) = evaluated_integral_with_required_conditions(input);
        assert!(
            !result.contains("integrate("),
            "raised algebraic zero-test budget should verify and emit: {input} -> {result}"
        );
    }
    // Denominators needing an algebraic extension (Φ5 √5-quadratics, ∛2) are still
    // out of scope — the budget does not create the missing ℚ(√n) render.
    for input in [
        "integrate(1/(x^5-1), x)",
        "integrate(1/(x^4-5), x)",
        "integrate(1/(x^3-2), x)",
    ] {
        let (result, _required) = evaluated_integral_with_required_conditions(input);
        assert!(
            result.contains("integrate("),
            "algebraic-extension render is a later sub-cycle, still residual: {input} -> {result}"
        );
    }
}

#[test]
fn integrate_contract_positive_half_power_antiderivatives_render_as_sqrt() {
    for (input, expected) in [
        ("integrate(x/sqrt(x^2+3), x)", "sqrt(x^2 + 3)"),
        (
            "integrate((2*x+1)/sqrt(x^2+x+1), x)",
            "2 * sqrt(x^2 + x + 1)",
        ),
    ] {
        let (result, required) = evaluated_integral_with_required_conditions(input);
        assert_eq!(result, expected, "input: {input}");
        assert!(
            required.is_empty(),
            "positive half-power presentation should not add domain conditions for {input}: {required:?}"
        );
        assert!(
            !result.contains("^(1/2)"),
            "post-integration presentation should prefer sqrt notation: {result}"
        );
        assert_antiderivative_verifies(input);
    }
}

#[test]
fn integrate_contract_reciprocal_sqrt_antiderivative_rationalized_residual_collapses() {
    let input = "integrate((2*x+1)/(x^2+x+1)^(3/2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);
    assert_eq!(result, "-2 / sqrt(x^2 + x + 1)");
    assert!(
        required.is_empty(),
        "positive quadratic reciprocal-root primitive should not add domain conditions: {required:?}"
    );

    let residual = "integrate((2*x+1)/(x^2+x+1)^(3/2), x) - (-2*sqrt(x^2+x+1)/(x^2+x+1))";
    let (residual_result, residual_required) =
        evaluated_integral_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert!(
        residual_required.is_empty(),
        "positive quadratic rationalized residual should not add domain conditions: {residual_required:?}"
    );

    let step_summaries = evaluated_expr_step_summaries(residual);
    assert!(
        step_summaries
            .iter()
            .any(
                |(description, rule, _)| description == "Post-calculus residual simplification"
                    || rule == "Post-calculus residual simplification"
            ),
        "expected a visible post-calculus residual simplification step, got {step_summaries:?}"
    );
}

#[test]
fn integrate_contract_positive_sqrt_antiderivative_rationalized_residual_collapses() {
    let input = "integrate((2*x+1)/sqrt(x^2+x+1), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);
    assert_eq!(result, "2 * sqrt(x^2 + x + 1)");
    assert!(
        required.is_empty(),
        "positive quadratic sqrt primitive should not add domain conditions: {required:?}"
    );

    let residual = "integrate((2*x+1)/sqrt(x^2+x+1), x) - 2*(x^2+x+1)/sqrt(x^2+x+1)";
    let (residual_result, residual_required) =
        evaluated_integral_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert!(
        residual_required.is_empty(),
        "positive quadratic rationalized sqrt residual should not add domain conditions: {residual_required:?}"
    );

    let mismatch = "integrate((2*x+1)/sqrt(x^2+x+1), x) - 3*(x^2+x+1)/sqrt(x^2+x+1)";
    let (mismatch_result, _) = evaluated_integral_with_required_conditions(mismatch);
    assert_ne!(
        mismatch_result, "0",
        "mismatched rationalized sqrt scale must not collapse"
    );
}

#[test]
fn integrate_contract_positive_sqrt_antiderivative_rationalized_residual_survives_quotient_wrapper()
{
    let residual = "(integrate((2*x+1)/sqrt(x^2+x+1), x) - 2*(x^2+x+1)/sqrt(x^2+x+1))/(x+2)";
    let (residual_result, residual_required) =
        evaluated_integral_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert_eq!(
        residual_required,
        vec!["x ≠ -2".to_string()],
        "quotient-wrapped rationalized sqrt residual should preserve denominator domain"
    );

    let step_summaries = evaluated_expr_step_summaries(residual);
    assert!(
        step_summaries
            .iter()
            .any(
                |(description, rule, _)| description == "Post-calculus residual simplification"
                    || rule == "Post-calculus residual simplification"
            ),
        "expected a visible post-calculus residual simplification step, got {step_summaries:?}"
    );
}

#[test]
fn integrate_contract_positive_sqrt_antiderivative_rationalized_residual_survives_shifted_reciprocal_difference(
) {
    let residual =
        "1/((integrate((2*x+1)/sqrt(x^2+x+1), x) - 2*(x^2+x+1)/sqrt(x^2+x+1))+x+2)-1/(x+2)";
    let (residual_result, residual_required) =
        evaluated_integral_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert_eq!(
        residual_required,
        vec!["x ≠ -2".to_string()],
        "shifted reciprocal residual should preserve the compact denominator domain"
    );

    let mismatch =
        "1/((integrate((2*x+1)/sqrt(x^2+x+1), x) - 3*(x^2+x+1)/sqrt(x^2+x+1))+x+2)-1/(x+2)";
    let (mismatch_result, _) = evaluated_integral_with_required_conditions(mismatch);
    assert_ne!(
        mismatch_result, "0",
        "mismatched rationalized sqrt scale must not collapse under reciprocal shift"
    );

    let step_summaries = evaluated_expr_step_summaries(residual);
    assert!(
        step_summaries
            .iter()
            .any(
                |(description, rule, _)| description == "Post-calculus residual simplification"
                    || rule == "Post-calculus residual simplification"
            ),
        "expected a visible post-calculus residual simplification step, got {step_summaries:?}"
    );
}

#[test]
fn integrate_contract_antiderivative_verification_uses_bounded_public_residual_for_hyperbolic_by_parts(
) {
    for input in [
        "integrate(x^2*sinh(x), x)",
        "integrate(x^2*cosh(x), x)",
        "integrate((x^3+x)*sinh(2*x+1), x)",
        "integrate((x^3+x)*cosh(2*x+1), x)",
        "integrate(x^6*sinh(x), x)",
        "integrate(x^6*cosh(x), x)",
        "integrate((x^6+1)*sinh(2*x+1), x)",
        "integrate((x^6+1)*cosh(2*x+1), x)",
        "integrate((x+1)*sinh((3*x+2)/2), x)",
        "integrate((x+1)*cosh((3*x+2)/2), x)",
        "integrate((x+1)*sinh((2-3*x)/2), x)",
        "integrate((x+1)*cosh((2-3*x)/2), x)",
        "integrate(x^7*sinh(x), x)",
        "integrate(x^7*cosh(x), x)",
        "integrate((x^7+1)*sinh(2*x+1), x)",
        "integrate((x^7+1)*cosh(2*x+1), x)",
    ] {
        assert_eq!(
            assert_antiderivative_verifies(input),
            AntiderivativeVerificationRoute::PublicResidual,
            "{input} should verify through the bounded public residual route"
        );
    }
}

#[test]
fn integrate_contract_hyperbolic_by_parts_double_nested_residual_compacts_without_timeout() {
    for input in [
        "((((diff(integrate(x^5*sinh(2*x+1), x), x) - x^5*sinh(2*x+1)) + 1)/(x+2))/(x+3))/(x+4)",
        "((((diff(integrate(x^4*cosh(2*x+1), x), x) - x^4*cosh(2*x+1)) + 1)/(x+2))/(x+3))/(x+4)",
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr(input);
        assert!(
            stderr.is_empty(),
            "unexpected stderr for hyperbolic double-nested residual: {stderr}"
        );
        assert_eq!(wire["result"], "1 / ((x + 2)·(x + 3)·(x + 4))", "{input}");
        assert_eq!(
            wire["required_display"],
            serde_json::json!(["x ≠ -2", "x ≠ -3", "x ≠ -4"]),
            "{input}"
        );
    }
}

#[test]
fn integrate_contract_positive_quadratic_rational_double_nested_residual_stays_quiet() {
    let input =
        "(((((diff(integrate((3*x + 5)/(x^2+x+1),x),x)-((3*x + 5)/(x^2+x+1)))+1)/(x+2))/(x+3))/(x+4))";
    let (wire, stderr) = cli_eval_json_with_stderr(input);
    assert!(
        stderr.is_empty(),
        "positive-quadratic rational residual should avoid depth overflow: {stderr}"
    );
    assert_eq!(wire["result"], "1 / ((x + 2)·(x + 3)·(x + 4))");
    assert_eq!(
        wire["required_display"],
        serde_json::json!(["x ≠ -2", "x ≠ -3", "x ≠ -4"])
    );
}

#[test]
fn integrate_contract_antiderivative_verification_uses_bounded_public_residual_for_exp_by_parts() {
    for input in [
        "integrate(x^2*exp(x), x)",
        "integrate(x^3*exp(x), x)",
        "integrate(x^5*exp(x), x)",
        "integrate((x^3+x)*exp(2*x+1), x)",
        "integrate((x^6+1)*exp(2*x+1), x)",
        "integrate(x^7*exp(x), x)",
        "integrate((x^7+1)*exp(2*x+1), x)",
    ] {
        assert_eq!(
            assert_antiderivative_verifies(input),
            AntiderivativeVerificationRoute::PublicResidual,
            "{input} should verify through the bounded public residual route"
        );
    }
}

#[test]
fn integrate_contract_exp_trig_same_linear_antiderivatives_verify_by_differentiation() {
    for (input, expected) in [
        (
            "integrate(exp(x)*sin(x), x)",
            "1/2 * e^x * (sin(x) - cos(x))",
        ),
        (
            "integrate(exp(x)*cos(x), x)",
            "1/2 * e^x * (sin(x) + cos(x))",
        ),
        (
            "integrate(exp(2*x+1)*sin(2*x+1), x)",
            "1/4 * e^(2 * x + 1) * (sin(2 * x + 1) - cos(2 * x + 1))",
        ),
        (
            "integrate(exp(2*x+1)*cos(2*x+1), x)",
            "1/4 * e^(2 * x + 1) * (sin(2 * x + 1) + cos(2 * x + 1))",
        ),
        (
            "integrate(3*exp(2*x+1)*sin(2*x+1), x)",
            "3/4 * e^(2 * x + 1) * (sin(2 * x + 1) - cos(2 * x + 1))",
        ),
        (
            "integrate(exp(2*x)*sin(3*x), x)",
            "1/13 * e^(2 * x) * (2 * sin(3 * x) - 3 * cos(3 * x))",
        ),
        (
            "integrate(exp(2*x)*sin((3*x+1)/2), x)",
            "4/25 * e^(2 * x) * (2 * sin((3 * x + 1) / 2) - 3/2 * cos((3 * x + 1) / 2))",
        ),
        (
            "integrate(exp(2*x)*cos((3*x+1)/2), x)",
            "4/25 * e^(2 * x) * (3/2 * sin((3 * x + 1) / 2) + 2 * cos((3 * x + 1) / 2))",
        ),
    ] {
        let (result, required) = evaluated_integral_with_required_conditions(input);
        assert_eq!(result, expected, "input: {input}");
        assert!(
            required.is_empty(),
            "exp-trig same-linear antiderivative should not add domain conditions for {input}: {required:?}"
        );
        assert_rendered_antiderivative_verifies(input, expected);
        assert_eq!(
            assert_antiderivative_verifies(input),
            AntiderivativeVerificationRoute::PublicResidual,
            "{input} should verify through the bounded public residual route"
        );
    }
}

#[test]
fn integrate_contract_exp_trig_integer_multiple_cos_verifies_by_public_residual() {
    let input = "integrate(exp(2*x)*cos(3*x), x)";
    let expected = "1/13 * e^(2 * x) * (2 * cos(3 * x) + 3 * sin(3 * x))";

    let (result, required) = evaluated_integral_with_required_conditions(input);
    assert_eq!(result, expected);
    assert!(
        required.is_empty(),
        "exp-trig integer-multiple cosine should not add domain conditions: {required:?}"
    );
    assert_eq!(
        assert_antiderivative_verifies(input),
        AntiderivativeVerificationRoute::PublicResidual,
        "{input} should verify through the bounded public residual route"
    );
    assert_rendered_antiderivative_verifies(input, expected);
    assert_rendered_antiderivative_verifies(input, &format!("{expected} + 7"));
    assert_rendered_antiderivative_verifies(input, &format!("{expected} + C"));
}

#[test]
fn integrate_contract_exp_trig_wrong_sign_antiderivative_residual_compacts_without_depth_warning() {
    let primitive = "1/13*exp(2*x)*(2*cos(3*x)+3*sin(3*x))";
    let integrand = "exp(2*x)*cos(3*x)";
    let input = format!("diff(7 - {primitive}, x) - {integrand}");

    let (wire, stderr) = cli_eval_json_with_stderr(&input);
    assert_eq!(wire["result"], "-2·cos(3·x)·e^(2·x)");
    assert!(
        wire["warnings"]
            .as_array()
            .is_some_and(|warnings| warnings.is_empty()),
        "wrong-sign exp-trig residual should not emit warnings: {wire:#}"
    );
    assert!(
        !stderr.contains("depth_overflow"),
        "wrong-sign exp-trig residual should not emit depth_overflow\nstderr:\n{stderr}"
    );
}

#[test]
fn integrate_contract_antiderivative_verification_uses_bounded_public_residual_for_trig_by_parts() {
    for input in [
        "integrate(x*sin(x), x)",
        "integrate(x*cos(x), x)",
        "integrate(x^2*sin(x), x)",
        "integrate(x^2*cos(x), x)",
        "integrate(x^5*sin(x), x)",
        "integrate(x^5*cos(x), x)",
        "integrate(x^6*sin(x), x)",
        "integrate(x^6*cos(x), x)",
        "integrate((x^6+1)*sin(2*x+1), x)",
        "integrate((x^6+1)*cos(2*x+1), x)",
        "integrate(x^7*sin(x), x)",
        "integrate(x^7*cos(x), x)",
        "integrate((x^7+1)*sin(2*x+1), x)",
        "integrate((x^7+1)*cos(2*x+1), x)",
        "integrate((2*x+3)*sin(2*x+1), x)",
        "integrate((2*x+3)*cos(2*x+1), x)",
    ] {
        assert_eq!(
            assert_antiderivative_verifies(input),
            AntiderivativeVerificationRoute::PublicResidual,
            "{input} should verify through the bounded public residual route"
        );
    }
}

#[test]
fn integrate_contract_antiderivative_verification_uses_bounded_public_residual_for_log_by_parts() {
    for input in ["integrate(x*ln(x), x)", "integrate((2*x+1)*ln(2*x+1), x)"] {
        assert_eq!(
            assert_antiderivative_verifies(input),
            AntiderivativeVerificationRoute::PublicResidual,
            "{input} should verify through the bounded public residual route"
        );
    }
}

#[test]
fn integrate_contract_antiderivative_verification_uses_bounded_public_residual_for_log_reciprocal_derivative(
) {
    for input in [
        "integrate(1/(x*ln(x)), x)",
        "integrate(2*x/((x^2+1)*ln(x^2+1)^2), x)",
    ] {
        assert_eq!(
            assert_antiderivative_verifies(input),
            AntiderivativeVerificationRoute::PublicResidual,
            "{input} should verify through the bounded public residual route"
        );
    }
}

#[test]
fn integrate_contract_antiderivative_verification_uses_bounded_public_residual_for_trig_log_substitution(
) {
    for input in [
        "integrate(tan(2*x+1), x)",
        "integrate(cot(2*x+1), x)",
        "integrate(sec(2*x+1), x)",
        "integrate(csc(2*x+1), x)",
    ] {
        assert_eq!(
            assert_antiderivative_verifies(input),
            AntiderivativeVerificationRoute::PublicResidual,
            "{input} should verify through the bounded public residual route"
        );
    }
}

#[test]
fn integrate_contract_antiderivative_verification_uses_bounded_public_residual_for_hyperbolic_quotient_substitution(
) {
    for input in [
        "integrate(sinh(2*x+1)/cosh(2*x+1), x)",
        "integrate(cosh(2*x+1)/sinh(2*x+1), x)",
    ] {
        assert_eq!(
            assert_antiderivative_verifies(input),
            AntiderivativeVerificationRoute::PublicResidual,
            "{input} should verify through the bounded public residual route"
        );
    }
}

#[test]
fn integrate_contract_antiderivative_verification_uses_bounded_public_residual_for_stable_hyperbolic_reciprocal_square_subset(
) {
    for input in [
        "integrate(1/cosh(2*x+1)^2, x)",
        "integrate(1/sinh(2*x+1)^2, x)",
        "integrate(sinh(2*x+1)/cosh(2*x+1)^2, x)",
        "integrate(cosh(2*x+1)/sinh(2*x+1)^2, x)",
    ] {
        assert_eq!(
            assert_antiderivative_verifies(input),
            AntiderivativeVerificationRoute::PublicResidual,
            "{input} should verify through the bounded public residual route"
        );
    }
}

#[test]
fn integrate_contract_reciprocal_hyperbolic_first_power_resolves_sech_csch() {
    // n == 1: sech -> arctan(sinh) (no domain condition, cosh > 0), csch ->
    // ln|tanh(x/2)| (the generic sinh != 0 domain carries over). Affine arguments
    // get the 1/u' scale; a constant numerator scales the primitive.
    for (input, expected_result, expected_required) in [
        (
            "integrate(1/cosh(x), x)",
            "arctan(sinh(x))",
            Vec::<String>::new(),
        ),
        ("integrate(sech(x), x)", "arctan(sinh(x))", Vec::new()),
        (
            "integrate(1/cosh(2*x+1), x)",
            "1/2 * arctan(sinh(2 * x + 1))",
            Vec::new(),
        ),
        ("integrate(3/cosh(x), x)", "3 * arctan(sinh(x))", Vec::new()),
        (
            "integrate(1/sinh(x), x)",
            "ln(|tanh(x / 2)|)",
            vec!["sinh(x) ≠ 0".to_string()],
        ),
    ] {
        let (result, required) = evaluated_expr_with_required_conditions(input);
        assert_eq!(result, expected_result, "result for {input}");
        assert_eq!(required, expected_required, "required for {input}");
    }

    // Soundness: each emitted antiderivative differentiates back to the integrand.
    for integrand in ["1/cosh(x)", "1/sinh(x)", "1/cosh(2*x+1)"] {
        let (result, _) = evaluated_expr_with_required_conditions(&format!(
            "diff(integrate({integrand}, x), x) - ({integrand})"
        ));
        assert_eq!(result, "0", "round-trip for {integrand}");
    }

    // The n=1 branch must NOT disturb the n>=2 table route nor hijack tanh/coth,
    // and must leave a constant-argument denominator as a constant multiple.
    assert_eq!(
        evaluated_expr_with_required_conditions("integrate(1/cosh(x)^2, x)").0,
        "tanh(x)"
    );
    assert_eq!(
        evaluated_expr_with_required_conditions("integrate(1/sinh(x)^2, x)").0,
        "-1 / tanh(x)"
    );
    assert_eq!(
        evaluated_expr_with_required_conditions("integrate(1/tanh(x), x)").0,
        "ln(|sinh(x)|)"
    );
    assert_eq!(
        evaluated_expr_with_required_conditions("integrate(1/cosh(5), x)").0,
        "x / cosh(5)"
    );
}

#[test]
fn integrate_contract_trig_of_logarithm_resolves_and_respects_boundary() {
    // cos/sin of ln(affine) integrate via the cyclic substitution u = ln(inner),
    // carrying the ln-positivity domain. The result round-trips to the integrand.
    for (input, expected_result, expected_required) in [
        (
            "integrate(cos(ln(x)), x)",
            "1/2 * x * (sin(ln(x)) + cos(ln(x)))",
            vec!["x > 0".to_string()],
        ),
        (
            "integrate(sin(ln(x)), x)",
            "1/2 * x * (sin(ln(x)) - cos(ln(x)))",
            vec!["x > 0".to_string()],
        ),
        (
            "integrate(cos(ln(2*x+1)), x)",
            "1/4 * (2 * x + 1) * (sin(ln(2 * x + 1)) + cos(ln(2 * x + 1)))",
            vec!["x > -1/2".to_string()],
        ),
    ] {
        let (result, required) = evaluated_expr_with_required_conditions(input);
        assert_eq!(result, expected_result, "result for {input}");
        assert_eq!(required, expected_required, "required for {input}");
        let (rt, _) = evaluated_expr_with_required_conditions(&format!(
            "diff(integrate({}, x), x) - ({})",
            &input[10..input.len() - 4],
            &input[10..input.len() - 4]
        ));
        assert_eq!(rt, "0", "round-trip for {input}");
    }

    // Boundary: only a bare cos/sin of ln(affine) fires. A non-logarithmic or
    // non-affine inner stays an honest residual, and the u=ln/x and bare-ln
    // owners keep their results (the new detector must not hijack them).
    for input in [
        "integrate(cos(ln(x) + 1), x)",
        "integrate(cos(x*ln(x)), x)",
        "integrate(cos(ln(x^2)), x)",
    ] {
        let (result, _) = evaluated_expr_with_required_conditions(input);
        assert!(
            result.starts_with("integrate("),
            "{input} should stay an honest residual, got {result}"
        );
    }
    assert_eq!(
        evaluated_expr_with_required_conditions("integrate(ln(x)/x, x)").0,
        "1/2 * ln(x)^2"
    );
    assert_eq!(
        evaluated_expr_with_required_conditions("integrate(ln(x), x)").0,
        "x * ln(x) - x"
    );
    assert_eq!(
        evaluated_expr_with_required_conditions("integrate(cos(x), x)").0,
        "sin(x)"
    );
}

#[test]
fn integrate_contract_odd_power_quadratic_inner_substitution_resolves_and_respects_boundary() {
    // x^(2k+1) f(x^2) for f in {exp,sin,cos}, odd power >= 3: substitute u = x^2
    // and delegate u^k f(c u) to the poly*{exp,trig} by-parts owner.
    for (input, expected) in [
        ("integrate(x^3*exp(x^2), x)", "1/2 * e^(x^2) * (x^2 - 1)"),
        (
            "integrate(x^5*exp(x^2), x)",
            "1/2 * e^(x^2) * (x^4 + 2 - 2 * x^2)",
        ),
        (
            "integrate(x^3*sin(x^2), x)",
            "1/2 * (sin(x^2) - cos(x^2) * x^2)",
        ),
        (
            "integrate(x^3*cos(x^2), x)",
            "1/2 * (cos(x^2) + sin(x^2) * x^2)",
        ),
        (
            "integrate(x^3*exp(2*x^2), x)",
            "1/8 * e^(2 * x^2) * (2 * x^2 - 1)",
        ),
    ] {
        let (result, required) = evaluated_expr_with_required_conditions(input);
        assert_eq!(result, expected, "result for {input}");
        assert!(required.is_empty(), "no domain condition for {input}");
        let integrand = &input["integrate(".len()..input.len() - ", x)".len()];
        let (rt, _) = evaluated_expr_with_required_conditions(&format!(
            "diff(integrate({integrand}, x), x) - ({integrand})"
        ));
        assert_eq!(rt, "0", "round-trip for {input}");
    }

    // Boundary: the k=0 case x*f(x^2) is owned by the derivative-substitution
    // rule; an even power, a non-x^2 inner, and a non-elementary f all decline.
    assert_eq!(
        evaluated_expr_with_required_conditions("integrate(x*exp(x^2), x)").0,
        "1/2 * e^(x^2)"
    );
    for residual in [
        "integrate(x^2*exp(x^2), x)", // even power, no elementary closed form
        "integrate(x^3*ln(x^2), x)",  // f = ln, not exp/sin/cos
    ] {
        let (result, _) = evaluated_expr_with_required_conditions(residual);
        assert!(
            result.starts_with("integrate("),
            "{residual} should stay an honest residual, got {result}"
        );
    }
    // x^3*exp(x) (linear inner) keeps the ordinary polynomial-by-parts result.
    assert_eq!(
        evaluated_expr_with_required_conditions("integrate(x^3*exp(x), x)").0,
        "e^x * (x^3 + 6 * x - 3 * x^2 - 6)"
    );
}

#[test]
fn integrate_contract_antiderivative_verification_uses_bounded_public_residual_for_nonlinear_hyperbolic_reciprocal_square_subset(
) {
    for input in [
        "integrate(2*x/cosh(x^2)^2, x)",
        "integrate(2*x/sinh(x^2)^2, x)",
        "integrate(2*x*sinh(x^2)/cosh(x^2)^2, x)",
        "integrate(2*x*cosh(x^2)/sinh(x^2)^2, x)",
    ] {
        assert_eq!(
            assert_antiderivative_verifies(input),
            AntiderivativeVerificationRoute::PublicResidual,
            "{input} should verify through the bounded public residual route"
        );
    }
}

#[test]
fn integrate_contract_antiderivative_verification_uses_bounded_public_residual_for_hyperbolic_reciprocal_fourth_subset(
) {
    for input in [
        "integrate(1/cosh(2*x+1)^4, x)",
        "integrate(2*x/cosh(x^2)^4, x)",
        "integrate(1/sinh(2*x+1)^4, x)",
        "integrate(2*x/sinh(x^2)^4, x)",
        "integrate(2*k*x/sinh(x^2+b)^4, x)",
    ] {
        assert_eq!(
            assert_antiderivative_verifies(input),
            AntiderivativeVerificationRoute::PublicResidual,
            "{input} should verify through the bounded public residual route"
        );
    }
}

#[test]
fn integrate_contract_nonlinear_hyperbolic_reciprocal_square_residual_survives_wrappers() {
    for (input, expected_result, expected_required_display) in [
        (
            "(diff(integrate(2*x/sinh(x^2)^2, x), x) - 2*x/sinh(x^2)^2) + y - y",
            "0",
            serde_json::json!(["sinh(x^2) ≠ 0"]),
        ),
        (
            "(diff(integrate(2*x/sinh(x^2)^2, x), x) - 2*x/sinh(x^2)^2)/(x+1)",
            "0",
            serde_json::json!(["sinh(x^2) ≠ 0", "x ≠ -1"]),
        ),
        (
            "((diff(integrate(2*x/sinh(x^2)^2, x), x) - 2*x/sinh(x^2)^2) + x + 1)/(x+1)",
            "1",
            serde_json::json!(["sinh(x^2) ≠ 0", "x ≠ -1"]),
        ),
        (
            "1/((diff(integrate(2*x/sinh(x^2)^2, x), x) - 2*x/sinh(x^2)^2) + x + 1) - 1/(x+1)",
            "0",
            serde_json::json!(["sinh(x^2) ≠ 0", "x ≠ -1"]),
        ),
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr(input);
        assert!(
            stderr.is_empty(),
            "unexpected stderr for wrapped nonlinear hyperbolic reciprocal-square residual: {stderr}"
        );
        assert_eq!(wire["result"], expected_result);
        assert_eq!(
            wire["required_display"], expected_required_display,
            "unexpected required_display for {input}: {:?}",
            wire["required_display"]
        );
    }
}

#[test]
fn integrate_contract_hyperbolic_reciprocal_fourth_substitution_verifies() {
    let cases = [
        (
            "integrate(1/cosh(x)^4, x)",
            "1/3 * (3 * tanh(x) - tanh(x)^3)",
            "diff(integrate(1/cosh(x)^4, x), x) - 1/cosh(x)^4",
            vec![],
        ),
        (
            "integrate(1/cosh(2*x+1)^4, x)",
            "1/6 * (3 * tanh(2 * x + 1) - tanh(2 * x + 1)^3)",
            "diff(integrate(1/cosh(2*x+1)^4, x), x) - 1/cosh(2*x+1)^4",
            vec![],
        ),
        (
            "integrate(2*x/cosh(x^2)^4, x)",
            "1/3 * (3 * tanh(x^2) - tanh(x^2)^3)",
            "diff(integrate(2*x/cosh(x^2)^4, x), x) - 2*x/cosh(x^2)^4",
            vec![],
        ),
        (
            "integrate(1/sinh(2*x+1)^4, x)",
            "1/2 / tanh(2 * x + 1) - 1/6 / tanh(2 * x + 1)^3",
            "diff(integrate(1/sinh(2*x+1)^4, x), x) - 1/sinh(2*x+1)^4",
            vec!["sinh(2 * x + 1) ≠ 0"],
        ),
        (
            "integrate(2*x/sinh(x^2)^4, x)",
            "1 / tanh(x^2) - 1/3 / tanh(x^2)^3",
            "diff(integrate(2*x/sinh(x^2)^4, x), x) - 2*x/sinh(x^2)^4",
            vec!["sinh(x^2) ≠ 0"],
        ),
        (
            "integrate(2*k*x/sinh(x^2+b)^4, x)",
            "k / tanh(x^2 + b) - k / (3 * tanh(x^2 + b)^3)",
            "diff(integrate(2*k*x/sinh(x^2+b)^4, x), x) - 2*k*x/sinh(x^2+b)^4",
            vec!["sinh(x^2 + b) ≠ 0"],
        ),
    ];

    for (input, expected, residual, expected_required) in cases {
        let (result, required) = evaluated_integral_with_required_conditions(input);
        assert_eq!(result, expected, "input: {input}");
        assert_eq!(
            required, expected_required,
            "unexpected required conditions for {input}: {required:?}"
        );
        let (residual_result, residual_required) =
            evaluated_integral_with_required_conditions(residual);
        assert_eq!(residual_result, "0", "residual: {residual}");
        assert_eq!(
            residual_required, expected_required,
            "unexpected residual required conditions for {input}: {residual_required:?}"
        );
    }
}

#[test]
fn integrate_contract_reciprocal_shifted_trig_by_parts_residual_keeps_compact_requires() {
    let input = "1/((diff(integrate(x^3*sin(x), x), x) - x^3*sin(x)) + x + 2) - 1/(x+2)";
    let (wire, stderr) = cli_eval_json_with_stderr(input);

    assert!(
        stderr.is_empty(),
        "unexpected stderr for reciprocal shifted trig by-parts residual: {stderr}"
    );
    assert_eq!(wire["result"], "0");
    assert_eq!(wire["required_display"], serde_json::json!(["x ≠ -2"]));
}

#[test]
fn integrate_contract_reciprocal_shifted_log_reciprocal_residual_keeps_domain_requires() {
    let input = "1/((diff(integrate(1/(x*ln(x)), x), x) - 1/(x*ln(x))) + x + 2) - 1/(x+2)";
    let (wire, stderr) = cli_eval_json_with_stderr(input);

    assert!(
        stderr.is_empty(),
        "unexpected stderr for reciprocal shifted log reciprocal residual: {stderr}"
    );
    assert_eq!(wire["result"], "0");
    assert_eq!(
        wire["required_display"],
        serde_json::json!(["x > 0", "x ≠ 1"])
    );
}

#[test]
fn integrate_contract_reciprocal_shifted_arcsin_residual_keeps_radical_domain_requires() {
    let input = "1/((diff(integrate(1/sqrt(1-x^2), x), x) - 1/sqrt(1-x^2)) + x + 2) - 1/(x+2)";
    let (wire, stderr) = cli_eval_json_with_stderr(input);

    assert!(
        stderr.is_empty(),
        "unexpected stderr for reciprocal shifted arcsin residual: {stderr}"
    );
    assert_eq!(wire["result"], "0");
    assert_eq!(
        wire["required_display"],
        serde_json::json!(["-1 < x < 1", "x ≠ -2"])
    );
}

#[test]
fn integrate_contract_reciprocal_shifted_affine_arcsin_residual_keeps_radical_domain_requires() {
    let input =
        "1/((diff(integrate(1/sqrt(4-(x+1)^2), x), x) - 1/sqrt(4-(x+1)^2)) + x + 2) - 1/(x+2)";
    let (wire, stderr) = cli_eval_json_with_stderr(input);

    assert!(
        stderr.is_empty(),
        "unexpected stderr for reciprocal shifted affine arcsin residual: {stderr}"
    );
    assert_eq!(wire["result"], "0");
    assert_eq!(
        wire["required_display"],
        serde_json::json!(["-3 < x < 1", "x ≠ -2"])
    );
}

#[test]
fn integrate_contract_reciprocal_shifted_arctan_sqrt_residual_keeps_positive_domain_requires() {
    let input =
        "1/((diff(integrate(1/(sqrt(x)*(x+1)), x), x) - 1/(sqrt(x)*(x+1))) + x + 2) - 1/(x+2)";
    let (wire, stderr) = cli_eval_json_with_stderr(input);

    assert!(
        stderr.is_empty(),
        "unexpected stderr for reciprocal shifted arctan sqrt residual: {stderr}"
    );
    assert_eq!(wire["result"], "0");
    assert_eq!(wire["required_display"], serde_json::json!(["x > 0"]));
}

#[test]
fn integrate_contract_reciprocal_shifted_root_product_residual_compacts_without_timeout() {
    let input =
        "1/((diff(integrate(1/(sqrt(2*x)*sqrt(2*x+6)), x), x) - 1/(sqrt(2*x)*sqrt(2*x+6))) + x + 2) - 1/(x+2)";
    let (wire, stderr) = cli_eval_json_with_stderr(input);

    assert!(
        stderr.is_empty(),
        "unexpected stderr for reciprocal shifted root-product residual: {stderr}"
    );
    assert_eq!(wire["result"], "0");
    assert_eq!(wire["required_display"], serde_json::json!(["x > 0"]));
}

#[test]
fn integrate_contract_reciprocal_shifted_root_product_residual_additive_noise_compacts_without_timeout(
) {
    let input =
        "1/((diff(integrate(1/(sqrt(2*x)*sqrt(2*x+6)), x), x) - 1/(sqrt(2*x)*sqrt(2*x+6))) + x + 2) - 1/(x+2) + y - y";
    let (wire, stderr) = cli_eval_json_with_stderr(input);

    assert!(
        stderr.is_empty(),
        "unexpected stderr for reciprocal shifted root-product residual with additive noise: {stderr}"
    );
    assert_eq!(wire["result"], "0");
    assert_eq!(wire["required_display"], serde_json::json!(["x > 0"]));
}

#[test]
fn integrate_contract_reciprocal_shifted_asinh_residual_compacts_without_timeout() {
    let input =
        "1/((diff(integrate(1/sqrt(4+(x+1)^2), x), x) - 1/sqrt(4+(x+1)^2)) + x + 2) - 1/(x+2)";
    let (wire, stderr) = cli_eval_json_with_stderr(input);

    assert!(
        stderr.is_empty(),
        "unexpected stderr for reciprocal shifted asinh residual: {stderr}"
    );
    assert_eq!(wire["result"], "0");
    assert_eq!(wire["required_display"], serde_json::json!(["x ≠ -2"]));
}

#[test]
fn integrate_contract_reciprocal_shifted_csc_residual_compacts_without_timeout() {
    let input = "1/((diff(integrate(csc(2*x+1), x), x) - csc(2*x+1)) + x + 2) - 1/(x+2)";
    let (wire, stderr) = cli_eval_json_with_stderr(input);

    assert!(
        stderr.is_empty(),
        "unexpected stderr for reciprocal shifted csc residual: {stderr}"
    );
    assert_eq!(wire["result"], "0");
    assert_eq!(
        wire["required_display"],
        serde_json::json!(["sin(2·x + 1) ≠ 0", "x ≠ -2"])
    );
}

#[test]
fn integrate_contract_shifted_quotient_asinh_residual_compacts_without_timeout() {
    let input =
        "1 - (x+2)/((diff(integrate(1/sqrt(4+(x+1)^2), x), x) - 1/sqrt(4+(x+1)^2)) + x + 2)";
    let (wire, stderr) = cli_eval_json_with_stderr(input);

    assert!(
        stderr.is_empty(),
        "unexpected stderr for shifted quotient asinh residual: {stderr}"
    );
    assert_eq!(wire["result"], "0");
    assert_eq!(wire["required_display"], serde_json::json!(["x ≠ -2"]));
}

#[test]
fn integrate_contract_shifted_quotient_csc_residual_compacts_without_timeout() {
    let input = "1 - (x+2)/((diff(integrate(csc(2*x+1), x), x) - csc(2*x+1)) + x + 2)";
    let (wire, stderr) = cli_eval_json_with_stderr(input);

    assert!(
        stderr.is_empty(),
        "unexpected stderr for shifted quotient csc residual: {stderr}"
    );
    assert_eq!(wire["result"], "0");
    assert_eq!(
        wire["required_display"],
        serde_json::json!(["sin(2·x + 1) ≠ 0", "x ≠ -2"])
    );
}

#[test]
fn integrate_contract_root_product_residual_constant_passthrough_quotient_compacts_without_timeout()
{
    let input =
        "((diff(integrate(1/(sqrt(2*x)*sqrt(2*x+6)), x), x) - 1/(sqrt(2*x)*sqrt(2*x+6))) + x + 2)/(x+2)";
    let (wire, stderr) = cli_eval_json_with_stderr(input);

    assert!(
        stderr.is_empty(),
        "unexpected stderr for root-product residual passthrough quotient: {stderr}"
    );
    assert_eq!(wire["result"], "1");
    assert_eq!(wire["required_display"], serde_json::json!(["x > 0"]));
}

#[test]
fn integrate_contract_root_product_residual_reciprocal_shifted_quotient_compacts_without_timeout() {
    let input =
        "1/(((diff(integrate(1/(sqrt(2*x)*sqrt(2*x+6)), x), x) - 1/(sqrt(2*x)*sqrt(2*x+6))) + x + 2)/(x+2)) - 1";
    let (wire, stderr) = cli_eval_json_with_stderr(input);

    assert!(
        stderr.is_empty(),
        "unexpected stderr for root-product residual reciprocal shifted quotient: {stderr}"
    );
    assert_eq!(wire["result"], "0");
    assert_eq!(wire["required_display"], serde_json::json!(["x > 0"]));
}

#[test]
fn integrate_contract_root_product_residual_reciprocal_shifted_quotient_additive_noise_compacts_without_timeout(
) {
    let input =
        "1/(((diff(integrate(1/(sqrt(2*x)*sqrt(2*x+6)), x), x) - 1/(sqrt(2*x)*sqrt(2*x+6))) + x + 2)/(x+2)) + y - (1+y)";
    let (wire, stderr) = cli_eval_json_with_stderr(input);

    assert!(
        stderr.is_empty(),
        "unexpected stderr for root-product residual reciprocal shifted quotient with additive noise: {stderr}"
    );
    assert_eq!(wire["result"], "0");
    assert_eq!(wire["required_display"], serde_json::json!(["x > 0"]));
}

#[test]
fn integrate_contract_root_product_residual_squared_shifted_quotient_compacts_without_timeout() {
    let input =
        "(((diff(integrate(1/(sqrt(2*x)*sqrt(2*x+6)), x), x) - 1/(sqrt(2*x)*sqrt(2*x+6))) + x + 2)/(x+2))^2 - 1";
    let (wire, stderr) = cli_eval_json_with_stderr(input);

    assert!(
        stderr.is_empty(),
        "unexpected stderr for root-product residual squared shifted quotient: {stderr}"
    );
    assert_eq!(wire["result"], "0");
    assert_eq!(wire["required_display"], serde_json::json!(["x > 0"]));
}

#[test]
fn integrate_contract_product_zero_csc_residual_compacts_without_timeout() {
    for input in [
        "((diff(integrate(csc(2*x+1), x), x) - csc(2*x+1)) + x + 2)*(y-y)",
        "(y-y)*((diff(integrate(csc(2*x+1), x), x) - csc(2*x+1)) + x + 2)",
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr(input);

        assert!(
            stderr.is_empty(),
            "unexpected stderr for product-zero csc residual: {stderr}"
        );
        assert_eq!(wire["result"], "0", "{input}");
        assert_eq!(
            wire["required_display"],
            serde_json::json!(["sin(2·x + 1) ≠ 0"]),
            "{input}"
        );
    }
}

#[test]
fn integrate_contract_product_zero_by_parts_residuals_compact_without_timeout() {
    for input in [
        "((diff(integrate(x^6*sin(x), x), x) - x^6*sin(x)) + x + 2)*(y-y)",
        "((diff(integrate((x^3+x)*cosh(2*x+1), x), x) - ((x^3+x)*cosh(2*x+1))) + x + 2)*(y-y)",
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr(input);

        assert!(
            stderr.is_empty(),
            "unexpected stderr for product-zero by-parts residual: {stderr}"
        );
        assert_eq!(wire["result"], "0", "{input}");
        assert_eq!(wire["required_display"], serde_json::json!([]), "{input}");
    }
}

#[test]
fn integrate_contract_negative_by_parts_primitives_keep_compact_public_form() {
    let cases = [
        (
            "integrate(-x*exp(x), x)",
            "e^x·(1 - x)",
            "{e}^{x}\\cdot (1 - x)",
            "diff(integrate(-x*exp(x), x), x) + x*exp(x)",
        ),
        (
            "integrate(-x^2*sin(x), x)",
            "-2·x·sin(x) + (x^2 - 2)·cos(x)",
            "-2\\cdot x\\cdot \\sin(x) + ({x}^{2} - 2)\\cdot \\cos(x)",
            "diff(integrate(-x^2*sin(x), x), x) + x^2*sin(x)",
        ),
    ];

    for (input, expected_result, expected_latex, residual) in cases {
        let (wire, stderr) = cli_eval_json_with_stderr(input);
        assert!(
            stderr.is_empty(),
            "unexpected stderr for negative by-parts primitive: {stderr}"
        );
        assert_eq!(wire["result"], expected_result);
        assert_eq!(wire["result_latex"], expected_latex);
        assert_eq!(wire["required_display"], serde_json::json!([]));

        let (residual_wire, residual_stderr) = cli_eval_json_with_stderr(residual);
        assert!(
            residual_stderr.is_empty(),
            "unexpected stderr for negative by-parts residual: {residual_stderr}"
        );
        assert_eq!(residual_wire["result"], "0");
        assert_eq!(residual_wire["required_display"], serde_json::json!([]));
    }
}

#[test]
fn integrate_contract_negative_affine_trig_by_parts_keeps_compact_public_form() {
    let input = "integrate(-(2*x+3)*sin(2*x+1), x)";
    let (wire, stderr) = cli_eval_json_with_stderr(input);
    assert!(
        stderr.is_empty(),
        "unexpected stderr for negative affine trig by-parts primitive: {stderr}"
    );
    assert_eq!(
        wire["result"],
        "1/2·(cos(2·x + 1)·(2·x + 3) - sin(2·x + 1))"
    );
    assert_eq!(
        wire["result_latex"],
        "\\frac{\\cos(2\\cdot x + 1)\\cdot (2\\cdot x + 3) - \\sin(2\\cdot x + 1)}{2}"
    );
    assert_eq!(wire["required_display"], serde_json::json!([]));

    let residual = "diff(integrate(-(2*x+3)*sin(2*x+1), x), x) + (2*x+3)*sin(2*x+1)";
    let (residual_wire, residual_stderr) = cli_eval_json_with_stderr(residual);
    assert!(
        residual_stderr.is_empty(),
        "unexpected stderr for negative affine trig by-parts residual: {residual_stderr}"
    );
    assert_eq!(residual_wire["result"], "0");
    assert_eq!(residual_wire["required_display"], serde_json::json!([]));
}

#[test]
fn integrate_contract_quadratic_trig_by_parts_presents_without_blocked_hint() {
    let (sin_result, sin_required, sin_blocked) =
        evaluated_expr_with_required_conditions_and_blocked_count("integrate(x^2*sin(x), x)");
    assert_eq!(sin_result, "2 * x * sin(x) + (2 - x^2) * cos(x)");
    assert!(
        sin_required.is_empty(),
        "unexpected required conditions: {sin_required:?}"
    );
    assert_eq!(sin_blocked, 0, "unexpected blocked hints for x^2*sin(x)");
    assert_antiderivative_verifies("integrate(x^2*sin(x), x)");

    let (cos_result, cos_required, cos_blocked) =
        evaluated_expr_with_required_conditions_and_blocked_count("integrate(x^2*cos(x), x)");
    assert_eq!(cos_result, "2 * x * cos(x) + (x^2 - 2) * sin(x)");
    assert!(
        cos_required.is_empty(),
        "unexpected required conditions: {cos_required:?}"
    );
    assert_eq!(cos_blocked, 0, "unexpected blocked hints for x^2*cos(x)");
    assert_antiderivative_verifies("integrate(x^2*cos(x), x)");

    let expanded = "integrate(x^2*sin(2*x+1)+x*sin(2*x+1), x)";
    let (expanded_result, expanded_required, expanded_blocked) =
        evaluated_expr_with_required_conditions_and_blocked_count(expanded);
    assert!(
        !expanded_result.starts_with("integrate("),
        "expected additive common-trig by-parts primitive, got {expanded_result}"
    );
    assert!(
        expanded_required.is_empty(),
        "unexpected required conditions for additive common-trig by-parts: {expanded_required:?}"
    );
    assert_eq!(
        expanded_blocked, 0,
        "unexpected blocked hints for additive common-trig by-parts"
    );

    let residual =
        "diff(integrate(x^2*sin(2*x+1)+x*sin(2*x+1), x), x) - (x^2*sin(2*x+1)+x*sin(2*x+1))";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert!(
        residual_required.is_empty(),
        "unexpected residual required conditions for additive common-trig by-parts: {residual_required:?}"
    );

    let (cos_equiv, cos_equiv_required) = evaluated_equiv_with_required_conditions(
        "diff(integrate(x^2*cos(2*x+1)+x*cos(2*x+1), x), x)",
        "x^2*cos(2*x+1)+x*cos(2*x+1)",
    );
    assert!(
        cos_equiv,
        "public equivalence should reuse the direct residual proof for additive common-trig by-parts"
    );
    assert!(
        cos_equiv_required.is_empty(),
        "unexpected public equivalence required conditions for additive common-trig by-parts: {cos_equiv_required:?}"
    );
}

#[test]
fn integrate_contract_by_parts_residual_dedupes_integrate_denominator_condition() {
    let residual = "1/((integrate(x^2*cos(x),x))+c) - 1/((2*x*cos(x)+(x^2-2)*sin(x))+c)";
    let (wire, stderr) = cli_eval_json_with_stderr(residual);

    assert!(
        stderr.is_empty(),
        "unexpected stderr for by-parts denominator residual: {stderr}"
    );
    assert_eq!(wire["result"], "0");
    assert_eq!(
        wire["required_display"],
        serde_json::json!(["2·x·cos(x) + (x^2 - 2)·sin(x) + c ≠ 0"])
    );
}

#[test]
fn integrate_contract_cubic_trig_by_parts_presents_without_blocked_hint() {
    let (sin_result, sin_required, sin_blocked) =
        evaluated_expr_with_required_conditions_and_blocked_count("integrate(x^3*sin(x), x)");
    assert_eq!(
        sin_result,
        "(6 * x - x^3) * cos(x) + (3 * x^2 - 6) * sin(x)"
    );
    assert!(
        sin_required.is_empty(),
        "unexpected required conditions: {sin_required:?}"
    );
    assert_eq!(sin_blocked, 0, "unexpected blocked hints for x^3*sin(x)");
    assert_antiderivative_verifies("integrate(x^3*sin(x), x)");

    let (cos_result, cos_required, cos_blocked) =
        evaluated_expr_with_required_conditions_and_blocked_count("integrate(x^3*cos(x), x)");
    assert_eq!(
        cos_result,
        "(x^3 - 6 * x) * sin(x) + (3 * x^2 - 6) * cos(x)"
    );
    assert!(
        cos_required.is_empty(),
        "unexpected required conditions: {cos_required:?}"
    );
    assert_eq!(cos_blocked, 0, "unexpected blocked hints for x^3*cos(x)");
    assert_antiderivative_verifies("integrate(x^3*cos(x), x)");

    for (lhs, rhs) in [
        (
            "diff(integrate(x^3*sin(2*x+1)+x*sin(2*x+1), x), x)",
            "x^3*sin(2*x+1)+x*sin(2*x+1)",
        ),
        (
            "diff(integrate(x^3*cos(2*x+1)+x*cos(2*x+1), x), x)",
            "x^3*cos(2*x+1)+x*cos(2*x+1)",
        ),
    ] {
        let (equivalent, required) = evaluated_equiv_with_required_conditions(lhs, rhs);
        assert!(
            equivalent,
            "public equivalence should reuse the direct residual proof for {lhs} equiv {rhs}"
        );
        assert!(
            required.is_empty(),
            "unexpected public equivalence required conditions for {lhs} equiv {rhs}: {required:?}"
        );
    }
}

#[test]
fn integrate_contract_quintic_trig_by_parts_presents_without_blocked_hint() {
    let (sin_wire, sin_stderr) = cli_eval_json_with_stderr("integrate(x^5*sin(x), x)");
    assert_eq!(
        sin_wire["result"],
        "(-x^5 + 20·x^3 - 120·x)·cos(x) + (5·x^4 - 60·x^2 + 120)·sin(x)"
    );
    assert!(
        sin_wire["required_conditions"]
            .as_array()
            .expect("required_conditions should be an array")
            .is_empty(),
        "unexpected required conditions for x^5*sin(x): {:?}",
        sin_wire["required_conditions"]
    );
    assert!(
        !sin_stderr.contains("depth_overflow"),
        "quintic sin by-parts presentation should not emit depth_overflow warning\nstderr:\n{sin_stderr}"
    );
    assert_antiderivative_verifies("integrate(x^5*sin(x), x)");

    let (cos_wire, cos_stderr) = cli_eval_json_with_stderr("integrate(x^5*cos(x), x)");
    assert_eq!(
        cos_wire["result"],
        "(x^5 - 20·x^3 + 120·x)·sin(x) + (5·x^4 - 60·x^2 + 120)·cos(x)"
    );
    assert!(
        cos_wire["required_conditions"]
            .as_array()
            .expect("required_conditions should be an array")
            .is_empty(),
        "unexpected required conditions for x^5*cos(x): {:?}",
        cos_wire["required_conditions"]
    );
    assert!(
        !cos_stderr.contains("depth_overflow"),
        "quintic cos by-parts presentation should not emit depth_overflow warning\nstderr:\n{cos_stderr}"
    );
    assert_antiderivative_verifies("integrate(x^5*cos(x), x)");
}

#[test]
fn integrate_contract_quintic_trig_by_parts_nested_residual_verifies_publicly() {
    for residual in [
        "diff(integrate(x^5*sin(x), x), x) - x^5*sin(x)",
        "diff(integrate(x^5*cos(x), x), x) - x^5*cos(x)",
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr(residual);
        assert_eq!(wire["result"], "0");
        assert!(
            wire["required_conditions"]
                .as_array()
                .expect("required_conditions should be an array")
                .is_empty(),
            "unexpected required conditions for {residual}: {:?}",
            wire["required_conditions"]
        );
        assert!(
            !stderr.contains("depth_overflow"),
            "quintic trig by-parts nested residual should not emit depth_overflow warning\nstderr:\n{stderr}"
        );
    }
}

#[test]
fn integrate_contract_sextic_trig_by_parts_verifies_publicly() {
    let (sin_wire, sin_stderr) =
        cli_eval_json_with_stderr_args("integrate(x^6*sin(x), x)", &["--steps", "on"]);
    assert_eq!(
        sin_wire["result"],
        "(6·x^5 - 120·x^3 + 720·x)·sin(x) + (-x^6 + 30·x^4 - 360·x^2 + 720)·cos(x)"
    );
    assert!(
        sin_wire["required_conditions"]
            .as_array()
            .expect("required_conditions should be an array")
            .is_empty(),
        "unexpected required conditions for x^6*sin(x): {:?}",
        sin_wire["required_conditions"]
    );
    assert!(
        !sin_stderr.contains("depth_overflow"),
        "sextic sin by-parts presentation should not emit depth_overflow warning\nstderr:\n{sin_stderr}"
    );
    let steps = sin_wire["steps"]
        .as_array()
        .expect("steps should be present with --steps on");
    let integration_step = steps
        .iter()
        .find(|step| step["rule"] == "Calcular la integral")
        .expect("expected public symbolic integration step");
    let substeps = integration_step["substeps"]
        .as_array()
        .expect("integration step should expose didactic substeps");
    assert!(
        substeps
            .iter()
            .any(|substep| substep["title"] == "Usar integración por partes repetida"),
        "expected repeated integration-by-parts substep for x^6*sin(x), got {substeps:?}"
    );
    assert_eq!(
        assert_antiderivative_verifies("integrate(x^6*sin(x), x)"),
        AntiderivativeVerificationRoute::PublicResidual
    );

    let (cos_wire, cos_stderr) =
        cli_eval_json_with_stderr_args("integrate(x^6*cos(x), x)", &["--steps", "on"]);
    assert_eq!(
        cos_wire["result"],
        "(6·x^5 - 120·x^3 + 720·x)·cos(x) + (x^6 - 30·x^4 + 360·x^2 - 720)·sin(x)"
    );
    assert!(
        cos_wire["required_conditions"]
            .as_array()
            .expect("required_conditions should be an array")
            .is_empty(),
        "unexpected required conditions for x^6*cos(x): {:?}",
        cos_wire["required_conditions"]
    );
    assert!(
        !cos_stderr.contains("depth_overflow"),
        "sextic cos by-parts presentation should not emit depth_overflow warning\nstderr:\n{cos_stderr}"
    );
    assert_eq!(
        assert_antiderivative_verifies("integrate(x^6*cos(x), x)"),
        AntiderivativeVerificationRoute::PublicResidual
    );

    for residual in [
        "diff(integrate(x^6*sin(x), x), x) - x^6*sin(x)",
        "diff(integrate(x^6*cos(x), x), x) - x^6*cos(x)",
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr(residual);
        assert_eq!(wire["result"], "0");
        assert!(
            wire["required_conditions"]
                .as_array()
                .expect("required_conditions should be an array")
                .is_empty(),
            "unexpected required conditions for {residual}: {:?}",
            wire["required_conditions"]
        );
        assert!(
            !stderr.contains("depth_overflow"),
            "sextic trig by-parts nested residual should not emit depth_overflow warning\nstderr:\n{stderr}"
        );
    }
}

#[test]
fn integrate_contract_sparse_affine_sextic_trig_by_parts_verifies_publicly() {
    for (input, expected_result) in [
        (
            "integrate((x^6+1)*sin(2*x+1), x)",
            "(3/2·x^5 - 15/2·x^3 + 45/4·x)·sin(2·x + 1) + (-1/2·x^6 + 15/4·x^4 - 45/4·x^2 + 41/8)·cos(2·x + 1)",
        ),
        (
            "integrate((x^6+1)*cos(2*x+1), x)",
            "(3/2·x^5 - 15/2·x^3 + 45/4·x)·cos(2·x + 1) + (1/2·x^6 - 15/4·x^4 + 45/4·x^2 - 41/8)·sin(2·x + 1)",
        ),
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

        assert_eq!(wire["result"], expected_result);
        assert!(
            wire["required_conditions"]
                .as_array()
                .expect("required_conditions should be an array")
                .is_empty(),
            "unexpected required conditions for {input}: {:?}",
            wire["required_conditions"]
        );
        assert!(
            !stderr.contains("depth_overflow"),
            "sparse affine sextic trig by-parts presentation should not emit depth_overflow warning for {input}\nstderr:\n{stderr}"
        );

        let steps = wire["steps"]
            .as_array()
            .expect("steps should be present with --steps on");
        assert_eq!(
            steps.len(),
            1,
            "expected direct integration trace without expansion noise for {input}, got {steps:?}"
        );
        assert!(
            steps
                .iter()
                .all(|step| step["rule"] != "Expandir la expresión"),
            "sparse affine sextic trig by-parts should not expand before integrating, got {steps:?}"
        );
        let integration_step = steps
            .iter()
            .find(|step| step["rule"] == "Calcular la integral")
            .expect("expected public symbolic integration step");
        let substeps = integration_step["substeps"]
            .as_array()
            .expect("integration step should expose didactic substeps");
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == "Usar integración por partes repetida"),
            "expected repeated integration-by-parts substep for {input}, got {substeps:?}"
        );
        assert_eq!(
            assert_antiderivative_verifies(input),
            AntiderivativeVerificationRoute::PublicResidual,
            "{input} should verify through the bounded public residual route"
        );
    }

    for residual in [
        "diff(integrate((x^6+1)*sin(2*x+1), x), x) - (x^6+1)*sin(2*x+1)",
        "diff(integrate((x^6+1)*cos(2*x+1), x), x) - (x^6+1)*cos(2*x+1)",
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr(residual);
        assert_eq!(wire["result"], "0");
        assert!(
            wire["required_conditions"]
                .as_array()
                .expect("required_conditions should be an array")
                .is_empty(),
            "unexpected required conditions for {residual}: {:?}",
            wire["required_conditions"]
        );
        assert!(
            !stderr.contains("depth_overflow"),
            "sparse affine sextic trig by-parts nested residual should not emit depth_overflow warning for {residual}\nstderr:\n{stderr}"
        );
    }
}

#[test]
fn integrate_contract_septic_trig_by_parts_verifies_publicly() {
    for (input, expected_result) in [
        (
            "integrate(x^7*sin(x), x)",
            "(-x^7 + 42·x^5 - 840·x^3 + 5040·x)·cos(x) + (7·x^6 - 210·x^4 + 2520·x^2 - 5040)·sin(x)",
        ),
        (
            "integrate(x^7*cos(x), x)",
            "(x^7 - 42·x^5 + 840·x^3 - 5040·x)·sin(x) + (7·x^6 - 210·x^4 + 2520·x^2 - 5040)·cos(x)",
        ),
        (
            "integrate((x^7+1)*sin(2*x+1), x)",
            "(7/4·x^6 - 105/8·x^4 + 315/8·x^2 - 315/16)·sin(2·x + 1) + (-1/2·x^7 + 21/4·x^5 - 105/4·x^3 + 315/8·x - 1/2)·cos(2·x + 1)",
        ),
        (
            "integrate((x^7+1)*cos(2*x+1), x)",
            "(7/4·x^6 - 105/8·x^4 + 315/8·x^2 - 315/16)·cos(2·x + 1) + (1/2·x^7 - 21/4·x^5 + 105/4·x^3 - 315/8·x + 1/2)·sin(2·x + 1)",
        ),
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

        assert_eq!(wire["result"], expected_result);
        assert!(
            wire["required_conditions"]
                .as_array()
                .expect("required_conditions should be an array")
                .is_empty(),
            "unexpected required conditions for {input}: {:?}",
            wire["required_conditions"]
        );
        assert!(
            !stderr.contains("depth_overflow"),
            "septic trig by-parts presentation should not emit depth_overflow warning for {input}\nstderr:\n{stderr}"
        );

        let steps = wire["steps"]
            .as_array()
            .expect("steps should be present with --steps on");
        assert_eq!(
            steps.len(),
            1,
            "expected direct integration trace without expansion noise for {input}, got {steps:?}"
        );
        let integration_step = steps
            .iter()
            .find(|step| step["rule"] == "Calcular la integral")
            .expect("expected public symbolic integration step");
        let substeps = integration_step["substeps"]
            .as_array()
            .expect("integration step should expose didactic substeps");
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == "Usar integración por partes repetida"),
            "expected repeated integration-by-parts substep for {input}, got {substeps:?}"
        );
        assert_eq!(
            assert_antiderivative_verifies(input),
            AntiderivativeVerificationRoute::PublicResidual,
            "{input} should verify through the bounded public residual route"
        );
    }

    for residual in [
        "diff(integrate(x^7*sin(x), x), x) - x^7*sin(x)",
        "diff(integrate(x^7*cos(x), x), x) - x^7*cos(x)",
        "diff(integrate((x^7+1)*sin(2*x+1), x), x) - (x^7+1)*sin(2*x+1)",
        "diff(integrate((x^7+1)*cos(2*x+1), x), x) - (x^7+1)*cos(2*x+1)",
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr(residual);
        assert_eq!(wire["result"], "0");
        assert!(
            wire["required_conditions"]
                .as_array()
                .expect("required_conditions should be an array")
                .is_empty(),
            "unexpected required conditions for {residual}: {:?}",
            wire["required_conditions"]
        );
        assert!(
            !stderr.contains("depth_overflow"),
            "septic trig by-parts nested residual should not emit depth_overflow warning for {residual}\nstderr:\n{stderr}"
        );
    }
}

#[test]
fn integrate_contract_octic_cos_by_parts_verifies_publicly() {
    let input = "integrate(x^8*cos(x), x)";
    let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

    assert_eq!(
        wire["result"],
        "(8·x^7 - 336·x^5 + 6720·x^3 - 40320·x)·cos(x) + (x^8 - 56·x^6 + 1680·x^4 - 20160·x^2 + 40320)·sin(x)"
    );
    assert!(
        wire["required_conditions"]
            .as_array()
            .expect("required_conditions should be an array")
            .is_empty(),
        "unexpected required conditions for {input}: {:?}",
        wire["required_conditions"]
    );
    assert!(
        !stderr.contains("depth_overflow"),
        "octic cosine by-parts presentation should not emit depth_overflow warning\nstderr:\n{stderr}"
    );

    let steps = wire["steps"]
        .as_array()
        .expect("steps should be present with --steps on");
    let integration_step = steps
        .iter()
        .find(|step| step["rule"] == "Calcular la integral")
        .expect("expected public symbolic integration step");
    let substeps = integration_step["substeps"]
        .as_array()
        .expect("integration step should expose didactic substeps");
    assert!(
        substeps
            .iter()
            .any(|substep| substep["title"] == "Usar integración por partes repetida"),
        "expected repeated integration-by-parts substep for {input}, got {substeps:?}"
    );

    let residual = "diff(integrate(x^8*cos(x), x), x) - x^8*cos(x)";
    let (wire, stderr) = cli_eval_json_with_stderr(residual);
    assert_eq!(wire["result"], "0");
    assert!(
        wire["required_conditions"]
            .as_array()
            .expect("required_conditions should be an array")
            .is_empty(),
        "unexpected residual required conditions for x^8*cos(x): {:?}",
        wire["required_conditions"]
    );
    assert!(
        !stderr.contains("depth_overflow"),
        "nested octic cosine verification should not emit depth_overflow warning\nstderr:\n{stderr}"
    );
}

#[test]
fn integrate_contract_affine_octic_sin_by_parts_verifies_publicly() {
    let input = "integrate(x^8*sin(2*x+1), x)";
    let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

    assert_eq!(
        wire["result"],
        "(2·x^7 - 21·x^5 + 105·x^3 - 315/2·x)·sin(2·x + 1) + (-1/2·x^8 + 7·x^6 - 105/2·x^4 + 315/2·x^2 - 315/4)·cos(2·x + 1)"
    );
    assert!(
        wire["required_conditions"]
            .as_array()
            .expect("required_conditions should be an array")
            .is_empty(),
        "unexpected required conditions for {input}: {:?}",
        wire["required_conditions"]
    );
    assert!(
        !stderr.contains("depth_overflow"),
        "affine octic sine by-parts presentation should not emit depth_overflow warning\nstderr:\n{stderr}"
    );

    let steps = wire["steps"]
        .as_array()
        .expect("steps should be present with --steps on");
    let integration_step = steps
        .iter()
        .find(|step| step["rule"] == "Calcular la integral")
        .expect("expected public symbolic integration step");
    let substeps = integration_step["substeps"]
        .as_array()
        .expect("integration step should expose didactic substeps");
    assert!(
        substeps
            .iter()
            .any(|substep| substep["title"] == "Usar integración por partes repetida"),
        "expected repeated integration-by-parts substep for {input}, got {substeps:?}"
    );

    let residual = "diff(integrate(x^8*sin(2*x+1), x), x) - x^8*sin(2*x+1)";
    let (wire, stderr) = cli_eval_json_with_stderr(residual);
    assert_eq!(wire["result"], "0");
    assert!(
        wire["required_conditions"]
            .as_array()
            .expect("required_conditions should be an array")
            .is_empty(),
        "unexpected residual required conditions for affine octic sine: {:?}",
        wire["required_conditions"]
    );
    assert!(
        !stderr.contains("depth_overflow"),
        "nested affine octic sine verification should not emit depth_overflow warning\nstderr:\n{stderr}"
    );
}

fn integration_substeps(input: &str) -> Vec<Value> {
    let (wire, _) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);
    let steps = wire["steps"]
        .as_array()
        .expect("steps should be present with --steps on")
        .clone();
    steps
        .iter()
        .find(|step| step["rule"] == "Calcular la integral")
        .and_then(|step| step["substeps"].as_array())
        .cloned()
        .unwrap_or_default()
}

fn substep_after_latex<'a>(substeps: &'a [Value], title: &str) -> Option<&'a str> {
    substeps
        .iter()
        .find(|substep| substep["title"] == title)
        .and_then(|substep| substep["after_latex"].as_str())
}

#[test]
fn integrate_contract_linear_elementary_by_parts_narrates_u_dv_du_v() {
    // The polynomial(linear) * {exp,sin,cos,sinh} family now narrates the full
    // by-parts choice (u = polynomial, dv = elementary factor), mirroring the
    // log narrator with the opposite u/dv assignment.
    for (input, dv_latex, v_latex) in [
        ("integrate(x*cos(x), x)", "\\cos(x)", "\\sin(x)"),
        ("integrate(x*sin(x), x)", "\\sin(x)", "-\\cos(x)"),
        ("integrate(x*exp(x), x)", "{e}^{x}", "{e}^{x}"),
        ("integrate(x*sinh(x), x)", "\\sinh(x)", "\\cosh(x)"),
    ] {
        let substeps = integration_substeps(input);
        assert_eq!(
            substep_after_latex(&substeps, "Elegir u y dv"),
            Some(format!("u = x,\\; dv = {dv_latex}\\,dx").as_str()),
            "u/dv narration mismatch for {input}, got {substeps:?}"
        );
        assert_eq!(
            substep_after_latex(&substeps, "Calcular du y v"),
            Some(format!("du = 1\\,dx,\\; v = {v_latex}").as_str()),
            "du/v narration mismatch for {input}, got {substeps:?}"
        );
        assert!(
            substeps
                .iter()
                .any(|s| s["title"] == "Aplicar la fórmula de integración por partes"),
            "missing apply-formula substep for {input}, got {substeps:?}"
        );
    }

    // Affine inner argument: v carries the 1/2 chain factor; non-unit du shows.
    let affine = integration_substeps("integrate((2*x+3)*exp(x), x)");
    assert_eq!(
        substep_after_latex(&affine, "Elegir u y dv"),
        Some("u = 2\\cdot x + 3,\\; dv = {e}^{x}\\,dx"),
        "u/dv for (2x+3)exp(x), got {affine:?}"
    );
    assert_eq!(
        substep_after_latex(&affine, "Calcular du y v"),
        Some("du = 2\\,dx,\\; v = {e}^{x}"),
        "du/v for (2x+3)exp(x), got {affine:?}"
    );

    // Regression: the ln family keeps ITS narration (u = ln), not duplicated by
    // the new poly-elementary narrator.
    let log = integration_substeps("integrate(x*ln(x), x)");
    assert_eq!(
        substep_after_latex(&log, "Elegir u y dv"),
        Some("u = \\ln(x),\\; dv = x\\,dx"),
        "ln by-parts narration regressed, got {log:?}"
    );

    // The repeated degree>=2 case now unrolls each by-parts application: the
    // master title stays "repetida" and the FIRST "Elegir u y dv" chooses the
    // full polynomial (u = x^2), with v = e^x and du = 2x.
    let repeated = integration_substeps("integrate(x^2*exp(x), x)");
    assert!(
        repeated
            .iter()
            .any(|s| s["title"] == "Usar integración por partes repetida"),
        "expected repeated title for x^2*exp(x), got {repeated:?}"
    );
    assert_eq!(
        substep_after_latex(&repeated, "Elegir u y dv"),
        Some("u = {x}^{2},\\; dv = {e}^{x}\\,dx"),
        "repeated case must now narrate the first u/dv, got {repeated:?}"
    );
    assert_eq!(
        substep_after_latex(&repeated, "Calcular du y v"),
        Some("du = 2\\cdot x\\,dx,\\; v = {e}^{x}"),
        "repeated case must narrate the first du/v, got {repeated:?}"
    );

    // No polynomial factor: the new narrator must not fire (different route).
    let product = integration_substeps("integrate(cos(x)*exp(x), x)");
    assert!(
        substep_after_latex(&product, "Elegir u y dv").is_none(),
        "cos(x)*exp(x) must not get poly-elementary narration, got {product:?}"
    );

    // Results are untouched (presentation-only change).
    assert_antiderivative_verifies("integrate(x*cos(x), x)");
    assert_antiderivative_verifies("integrate(x*exp(x), x)");
    assert_antiderivative_verifies("integrate((2*x+3)*exp(x), x)");
}

#[test]
fn integrate_contract_single_inverse_by_parts_narrates_u_dv_du_v() {
    // A bare inverse function integrates by parts with u = f(x), dv = dx, so
    // v = x and du = f'(x) dx. The narration now exposes that choice for the
    // arc-trig and inverse-hyperbolic family.
    for (input, u_latex, du_latex) in [
        (
            "integrate(arctan(x), x)",
            "\\arctan(x)",
            "\\frac{1}{{x}^{2} + 1}",
        ),
        (
            "integrate(arcsin(x), x)",
            "\\arcsin(x)",
            "{(1 - {x}^{2})}^{-\\frac{1}{2}}",
        ),
        (
            "integrate(arccos(x), x)",
            "\\arccos(x)",
            "-{(1 - {x}^{2})}^{-\\frac{1}{2}}",
        ),
        (
            "integrate(asinh(x), x)",
            "\\operatorname{asinh}(x)",
            "{({x}^{2} + 1)}^{-\\frac{1}{2}}",
        ),
    ] {
        let substeps = integration_substeps(input);
        assert_eq!(
            substep_after_latex(&substeps, "Elegir u y dv"),
            Some(format!("u = {u_latex},\\; dv = dx").as_str()),
            "u/dv narration mismatch for {input}, got {substeps:?}"
        );
        assert_eq!(
            substep_after_latex(&substeps, "Calcular du y v"),
            Some(format!("du = {du_latex}\\,dx,\\; v = x").as_str()),
            "du/v narration mismatch for {input}, got {substeps:?}"
        );
        assert!(
            substeps
                .iter()
                .any(|s| s["title"] == "Aplicar la fórmula de integración por partes"),
            "missing apply-formula substep for {input}, got {substeps:?}"
        );
    }

    // Affine argument: u carries the inner, v stays x, du folds the chain factor.
    let affine = integration_substeps("integrate(atan(2*x+1), x)");
    assert_eq!(
        substep_after_latex(&affine, "Elegir u y dv"),
        Some("u = \\arctan(2\\cdot x + 1),\\; dv = dx"),
        "u/dv for atan(2x+1), got {affine:?}"
    );

    // The new narrator must not fire on a product (owned by the other narrators)
    // and must leave plain trig/exp (which integrate directly) un-narrated.
    let product = integration_substeps("integrate(x*cos(x), x)");
    assert_eq!(
        substep_after_latex(&product, "Elegir u y dv"),
        Some("u = x,\\; dv = \\cos(x)\\,dx"),
        "x*cos(x) must keep its poly-elementary narration, got {product:?}"
    );
    assert!(
        integration_substeps("integrate(cos(x), x)")
            .iter()
            .all(|s| s["title"] != "Elegir u y dv"),
        "plain cos(x) integrates directly and must not be narrated by parts"
    );

    // Results are untouched (presentation-only change).
    assert_antiderivative_verifies("integrate(arctan(x), x)");
    assert_antiderivative_verifies("integrate(arcsin(x), x)");
}

#[test]
fn integrate_contract_bare_logarithm_by_parts_narrates_u_dv_du_v() {
    // A bare ln(affine) -- previously emitting NO by-parts substeps -- now
    // narrates u = ln, dv = dx, v = x, du = (ln arg)' dx via the single-inverse
    // narrator (the bare-ln by-parts gate makes the title fire).
    for (input, u_latex, du_latex) in [
        ("integrate(ln(x), x)", "\\ln(x)", "\\frac{1}{x}"),
        (
            "integrate(ln(2*x+1), x)",
            "\\ln(2\\cdot x + 1)",
            "\\frac{2}{2\\cdot x + 1}",
        ),
    ] {
        let substeps = integration_substeps(input);
        assert_eq!(
            substep_after_latex(&substeps, "Elegir u y dv"),
            Some(format!("u = {u_latex},\\; dv = dx").as_str()),
            "u/dv narration mismatch for {input}, got {substeps:?}"
        );
        assert_eq!(
            substep_after_latex(&substeps, "Calcular du y v"),
            Some(format!("du = {du_latex}\\,dx,\\; v = x").as_str()),
            "du/v narration mismatch for {input}, got {substeps:?}"
        );
    }

    // Regression: x*ln(x) keeps its polynomial*ln narration (u = ln, dv = x dx,
    // v = x^2/2), NOT the bare-ln dv = dx, proving the new branch did not steal it.
    let product = integration_substeps("integrate(x*ln(x), x)");
    assert_eq!(
        substep_after_latex(&product, "Elegir u y dv"),
        Some("u = \\ln(x),\\; dv = x\\,dx"),
        "x*ln(x) must keep its poly*ln narration, got {product:?}"
    );

    // Results untouched.
    assert_eq!(
        evaluated_expr_with_required_conditions("integrate(ln(x), x)").0,
        "x * ln(x) - x"
    );
}

#[test]
fn integrate_contract_cbrt_elementary_calculus_differentiates_and_integrates() {
    // cbrt is now first-class elementary calculus, like sqrt: it differentiates
    // and integrates as x^(1/3) (Pow form, no domain condition -- cbrt is defined
    // on all reals). cbrt stays Function(Cbrt) for display and the limit rules.
    // The derivative 1/(3 x^(2/3)) is undefined at x=0 (cbrt has a vertical
    // tangent there), so diff carries x != 0; the antiderivative x^(4/3) is
    // defined everywhere, so integrate carries no condition.
    for (input, expected, expected_required) in [
        (
            "diff(cbrt(x), x)",
            "1/3 * x^(-2/3)",
            vec!["x ≠ 0".to_string()],
        ),
        (
            "diff(5*cbrt(x), x)",
            "5/3 * x^(-2/3)",
            vec!["x ≠ 0".to_string()],
        ),
        ("integrate(cbrt(x), x)", "3/4 * x^(4/3)", Vec::new()),
        ("integrate(cbrt(x)^2, x)", "3/5 * x^(5/3)", Vec::new()),
        ("integrate(2*cbrt(x), x)", "3/2 * x^(4/3)", Vec::new()),
    ] {
        let (result, required) = evaluated_expr_with_required_conditions(input);
        assert_eq!(result, expected, "result for {input}");
        assert_eq!(required, expected_required, "required for {input}");
    }

    // Soundness in Pow form: diff(antiderivative) - x^(1/3) and the derivative
    // minus its closed form both reduce to 0 (the engine does not yet simplify
    // x^(1/3) == cbrt(x), so the cbrt-form round-trip stays open -- a separate
    // simplification gap, not an integration error).
    for (expr, want) in [
        ("diff(integrate(cbrt(x), x), x) - x^(1/3)", "0"),
        ("diff(integrate(cbrt(x)^2, x), x) - x^(2/3)", "0"),
        ("diff(cbrt(x), x) - 1/3*x^(-2/3)", "0"),
    ] {
        assert_eq!(
            evaluated_expr_with_required_conditions(expr).0,
            want,
            "soundness {expr}"
        );
    }

    // Boundary: a non-linear radicand stays an honest residual (no power-rule
    // target), bare cbrt(x) still displays as cbrt(x) (not lowered), and the
    // cube-root LIMIT rule (Function(Cbrt)) is untouched.
    for residual in ["integrate(cbrt(x^2+1), x)", "integrate(cbrt(x^2), x)"] {
        assert!(
            evaluated_expr_with_required_conditions(residual)
                .0
                .starts_with("integrate("),
            "{residual} should stay an honest residual"
        );
    }
    assert_eq!(
        evaluated_expr_with_required_conditions("cbrt(x)").0,
        "cbrt(x)"
    );
    // sqrt and x^(1/3) are untouched.
    assert_eq!(
        evaluated_expr_with_required_conditions("integrate(sqrt(x), x)").0,
        "2/3 * sqrt(x) * x"
    );
}

#[test]
fn integrate_contract_affine_octic_cos_by_parts_verifies_publicly() {
    let input = "integrate(x^8*cos(2*x+1), x)";
    let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

    assert_eq!(
        wire["result"],
        "(2·x^7 - 21·x^5 + 105·x^3 - 315/2·x)·cos(2·x + 1) + (1/2·x^8 - 7·x^6 + 105/2·x^4 - 315/2·x^2 + 315/4)·sin(2·x + 1)"
    );
    assert!(
        wire["required_conditions"]
            .as_array()
            .expect("required_conditions should be an array")
            .is_empty(),
        "unexpected required conditions for {input}: {:?}",
        wire["required_conditions"]
    );
    assert!(
        !stderr.contains("depth_overflow"),
        "affine octic cosine by-parts presentation should not emit depth_overflow warning\nstderr:\n{stderr}"
    );

    let steps = wire["steps"]
        .as_array()
        .expect("steps should be present with --steps on");
    let integration_step = steps
        .iter()
        .find(|step| step["rule"] == "Calcular la integral")
        .expect("expected public symbolic integration step");
    let substeps = integration_step["substeps"]
        .as_array()
        .expect("integration step should expose didactic substeps");
    assert!(
        substeps
            .iter()
            .any(|substep| substep["title"] == "Usar integración por partes repetida"),
        "expected repeated integration-by-parts substep for {input}, got {substeps:?}"
    );

    let residual = "diff(integrate(x^8*cos(2*x+1), x), x) - x^8*cos(2*x+1)";
    let (wire, stderr) = cli_eval_json_with_stderr(residual);
    assert_eq!(wire["result"], "0");
    assert!(
        wire["required_conditions"]
            .as_array()
            .expect("required_conditions should be an array")
            .is_empty(),
        "unexpected residual required conditions for affine octic cosine: {:?}",
        wire["required_conditions"]
    );
    assert!(
        !stderr.contains("depth_overflow"),
        "nested affine octic cosine verification should not emit depth_overflow warning\nstderr:\n{stderr}"
    );
}

#[test]
fn integrate_contract_quadratic_exp_by_parts_presents_without_depth_overflow() {
    let input = "integrate((x^2+x+1)*exp(2*x+1), x)";
    let (wire, stderr) = cli_eval_json_with_stderr(input);

    assert_eq!(wire["result"], "1/2·e^(2·x + 1)·(x^2 + 1)");
    assert!(
        !stderr.contains("depth_overflow"),
        "quadratic exp by-parts presentation should not emit depth_overflow warning\nstderr:\n{stderr}"
    );
    assert_antiderivative_verifies(input);

    let nested = "diff(integrate((x^2+x+1)*exp(2*x+1), x), x) - (x^2+x+1)*exp(2*x+1)";
    let (wire, stderr) = cli_eval_json_with_stderr(nested);
    assert_eq!(wire["result"], "0");
    assert!(
        !stderr.contains("depth_overflow"),
        "nested quadratic exp by-parts verification should not emit depth_overflow warning\nstderr:\n{stderr}"
    );
}

#[test]
fn integrate_contract_quadratic_exp_by_parts_exposes_didactic_substep() {
    let input = "integrate(x^2*exp(x), x)";
    let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

    assert_eq!(wire["result"], "e^x·(x^2 + 2 - 2·x)");
    assert!(
        !stderr.contains("depth_overflow"),
        "quadratic exp by-parts didactic trace should not emit depth_overflow warning\nstderr:\n{stderr}"
    );

    let steps = wire["steps"]
        .as_array()
        .expect("steps should be present with --steps on");
    let integration_step = steps
        .iter()
        .find(|step| step["rule"] == "Calcular la integral")
        .expect("expected public symbolic integration step");
    let substeps = integration_step["substeps"]
        .as_array()
        .expect("integration step should expose didactic substeps");
    assert!(
        substeps
            .iter()
            .any(|substep| substep["title"] == "Usar integración por partes repetida"),
        "expected repeated integration-by-parts substep, got {substeps:?}"
    );
    // x^2 e^x reduces in two by-parts applications, so the repeated narration
    // unrolls two "Elegir u y dv"/"Aplicar la fórmula" blocks and closes with a
    // single "Integrar el término restante" landing on the final antiderivative.
    let count_title = |title: &str| {
        substeps
            .iter()
            .filter(|substep| substep["title"] == title)
            .count()
    };
    assert_eq!(count_title("Elegir u y dv"), 2, "got {substeps:?}");
    assert_eq!(
        count_title("Aplicar la fórmula de integración por partes"),
        2,
        "got {substeps:?}"
    );
    let first_choice = substeps
        .iter()
        .find(|substep| substep["title"] == "Elegir u y dv")
        .expect("first u/dv choice");
    assert_eq!(
        first_choice["after_latex"], "u = {x}^{2},\\; dv = {e}^{x}\\,dx",
        "got {substeps:?}"
    );
    let closer = substeps
        .last()
        .expect("repeated narration should not be empty");
    assert_eq!(closer["title"], "Integrar el término restante");
    assert_eq!(
        closer["after_latex"], "e^{x}\\cdot ({x}^{2} + 2 - 2\\cdot x)",
        "closer should land on the final antiderivative, got {substeps:?}"
    );
    assert_antiderivative_verifies(input);
}

#[test]
fn integrate_contract_cubic_exp_by_parts_presents_without_depth_overflow() {
    for (input, expected_result, expected_substep_title) in [
        (
            "integrate(x^3*exp(x), x)",
            "e^x·(x^3 + 6·x - 3·x^2 - 6)",
            "Usar integración por partes repetida",
        ),
        (
            "integrate((x^3+x)*exp(2*x+1), x)",
            "1/8·e^(2·x + 1)·(4·x^3 + 10·x - 6·x^2 - 5)",
            "Usar integración por partes repetida",
        ),
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

        assert_eq!(wire["result"], expected_result);
        assert!(
            !stderr.contains("depth_overflow"),
            "cubic exp by-parts presentation should not emit depth_overflow warning for {input}\nstderr:\n{stderr}"
        );

        let steps = wire["steps"]
            .as_array()
            .expect("steps should be present with --steps on");
        let integration_step = steps
            .iter()
            .find(|step| step["rule"] == "Calcular la integral")
            .expect("expected public symbolic integration step");
        let substeps = integration_step["substeps"]
            .as_array()
            .expect("integration step should expose didactic substeps");
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == expected_substep_title),
            "expected {expected_substep_title} substep for {input}, got {substeps:?}"
        );
        assert_antiderivative_verifies(input);
    }
}

#[test]
fn integrate_contract_quartic_exp_by_parts_verifies() {
    let input = "integrate(x^4*exp(2*x+1), x)";
    let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

    assert_eq!(
        wire["result"],
        "1/4·e^(2·x + 1)·(2·x^4 + 6·x^2 + 3 - 4·x^3 - 6·x)"
    );
    assert!(
        !stderr.contains("depth_overflow"),
        "quartic exp by-parts presentation should not emit depth_overflow warning\nstderr:\n{stderr}"
    );

    let steps = wire["steps"]
        .as_array()
        .expect("steps should be present with --steps on");
    let integration_step = steps
        .iter()
        .find(|step| step["rule"] == "Calcular la integral")
        .expect("expected public symbolic integration step");
    let substeps = integration_step["substeps"]
        .as_array()
        .expect("integration step should expose didactic substeps");
    assert!(
        substeps
            .iter()
            .any(|substep| substep["title"] == "Usar integración por partes repetida"),
        "expected repeated integration-by-parts substep for {input}, got {substeps:?}"
    );

    let residual = "diff(integrate(x^4*exp(2*x+1), x), x) - x^4*exp(2*x+1)";
    let (wire, stderr) = cli_eval_json_with_stderr(residual);
    assert_eq!(wire["result"], "0");
    assert!(
        !stderr.contains("depth_overflow"),
        "nested quartic exp verification should not emit depth_overflow warning\nstderr:\n{stderr}"
    );
}

#[test]
fn integrate_contract_quintic_exp_by_parts_verifies() {
    let input = "integrate(x^5*exp(x), x)";
    let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

    assert_eq!(
        wire["result"],
        "e^x·(x^5 + 20·x^3 + 120·x - 5·x^4 - 60·x^2 - 120)"
    );
    assert!(
        wire["required_conditions"]
            .as_array()
            .expect("required_conditions should be an array")
            .is_empty(),
        "unexpected required conditions for x^5*exp(x): {:?}",
        wire["required_conditions"]
    );
    assert!(
        !stderr.contains("depth_overflow"),
        "quintic exp by-parts presentation should not emit depth_overflow warning\nstderr:\n{stderr}"
    );

    let steps = wire["steps"]
        .as_array()
        .expect("steps should be present with --steps on");
    let integration_step = steps
        .iter()
        .find(|step| step["rule"] == "Calcular la integral")
        .expect("expected public symbolic integration step");
    let substeps = integration_step["substeps"]
        .as_array()
        .expect("integration step should expose didactic substeps");
    assert!(
        substeps
            .iter()
            .any(|substep| substep["title"] == "Usar integración por partes repetida"),
        "expected repeated integration-by-parts substep for {input}, got {substeps:?}"
    );

    let residual = "diff(integrate(x^5*exp(x), x), x) - x^5*exp(x)";
    let (wire, stderr) = cli_eval_json_with_stderr(residual);
    assert_eq!(wire["result"], "0");
    assert!(
        !stderr.contains("depth_overflow"),
        "nested quintic exp verification should not emit depth_overflow warning\nstderr:\n{stderr}"
    );
}

#[test]
fn integrate_contract_sextic_exp_by_parts_verifies() {
    let input = "integrate(x^6*exp(x), x)";
    let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

    assert_eq!(
        wire["result"],
        "e^x·(x^6 + 30·x^4 + 360·x^2 + 720 - 6·x^5 - 120·x^3 - 720·x)"
    );
    assert!(
        wire["required_conditions"]
            .as_array()
            .expect("required_conditions should be an array")
            .is_empty(),
        "unexpected required conditions for x^6*exp(x): {:?}",
        wire["required_conditions"]
    );
    assert!(
        !stderr.contains("depth_overflow"),
        "sextic exp by-parts presentation should not emit depth_overflow warning\nstderr:\n{stderr}"
    );

    let steps = wire["steps"]
        .as_array()
        .expect("steps should be present with --steps on");
    let integration_step = steps
        .iter()
        .find(|step| step["rule"] == "Calcular la integral")
        .expect("expected public symbolic integration step");
    let substeps = integration_step["substeps"]
        .as_array()
        .expect("integration step should expose didactic substeps");
    assert!(
        substeps
            .iter()
            .any(|substep| substep["title"] == "Usar integración por partes repetida"),
        "expected repeated integration-by-parts substep for {input}, got {substeps:?}"
    );

    let residual = "diff(integrate(x^6*exp(x), x), x) - x^6*exp(x)";
    let (wire, stderr) = cli_eval_json_with_stderr(residual);
    assert_eq!(wire["result"], "0");
    assert!(
        !stderr.contains("depth_overflow"),
        "nested sextic exp verification should not emit depth_overflow warning\nstderr:\n{stderr}"
    );
}

#[test]
fn integrate_contract_septic_exp_by_parts_verifies() {
    let input = "integrate(x^7*exp(x), x)";
    let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

    assert_eq!(
        wire["result"],
        "e^x·(x^7 + 42·x^5 + 840·x^3 + 5040·x - 7·x^6 - 210·x^4 - 2520·x^2 - 5040)"
    );
    assert!(
        wire["required_conditions"]
            .as_array()
            .expect("required_conditions should be an array")
            .is_empty(),
        "unexpected required conditions for x^7*exp(x): {:?}",
        wire["required_conditions"]
    );
    assert!(
        !stderr.contains("depth_overflow"),
        "septic exp by-parts presentation should not emit depth_overflow warning\nstderr:\n{stderr}"
    );

    let steps = wire["steps"]
        .as_array()
        .expect("steps should be present with --steps on");
    let integration_step = steps
        .iter()
        .find(|step| step["rule"] == "Calcular la integral")
        .expect("expected public symbolic integration step");
    let substeps = integration_step["substeps"]
        .as_array()
        .expect("integration step should expose didactic substeps");
    assert!(
        substeps
            .iter()
            .any(|substep| substep["title"] == "Usar integración por partes repetida"),
        "expected repeated integration-by-parts substep for {input}, got {substeps:?}"
    );

    for residual in [
        "diff(integrate(x^7*exp(x), x), x) - x^7*exp(x)",
        "diff(integrate((x^7+1)*exp(2*x+1), x), x) - (x^7+1)*exp(2*x+1)",
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr(residual);
        assert_eq!(wire["result"], "0");
        assert!(
            wire["required_conditions"]
                .as_array()
                .expect("required_conditions should be an array")
                .is_empty(),
            "unexpected required conditions for {residual}: {:?}",
            wire["required_conditions"]
        );
        assert!(
            !stderr.contains("depth_overflow"),
            "nested septic exp verification should not emit depth_overflow warning for {residual}\nstderr:\n{stderr}"
        );
    }
}

#[test]
fn integrate_contract_octic_exp_by_parts_verifies() {
    let input = "integrate(x^8*exp(x), x)";
    let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

    assert_eq!(
        wire["result"],
        "e^x·(x^8 + 56·x^6 + 1680·x^4 + 20160·x^2 + 40320 - 8·x^7 - 336·x^5 - 6720·x^3 - 40320·x)"
    );
    assert!(
        wire["required_conditions"]
            .as_array()
            .expect("required_conditions should be an array")
            .is_empty(),
        "unexpected required conditions for x^8*exp(x): {:?}",
        wire["required_conditions"]
    );
    assert!(
        !stderr.contains("depth_overflow"),
        "octic exp by-parts presentation should not emit depth_overflow warning\nstderr:\n{stderr}"
    );

    let steps = wire["steps"]
        .as_array()
        .expect("steps should be present with --steps on");
    let integration_step = steps
        .iter()
        .find(|step| step["rule"] == "Calcular la integral")
        .expect("expected public symbolic integration step");
    let substeps = integration_step["substeps"]
        .as_array()
        .expect("integration step should expose didactic substeps");
    assert!(
        substeps
            .iter()
            .any(|substep| substep["title"] == "Usar integración por partes repetida"),
        "expected repeated integration-by-parts substep for {input}, got {substeps:?}"
    );

    let residual = "diff(integrate(x^8*exp(x), x), x) - x^8*exp(x)";
    let (wire, stderr) = cli_eval_json_with_stderr(residual);
    assert_eq!(wire["result"], "0");
    assert!(
        wire["required_conditions"]
            .as_array()
            .expect("required_conditions should be an array")
            .is_empty(),
        "unexpected residual required conditions for x^8*exp(x): {:?}",
        wire["required_conditions"]
    );
    assert!(
        !stderr.contains("depth_overflow"),
        "nested octic exp verification should not emit depth_overflow warning\nstderr:\n{stderr}"
    );
}

#[test]
fn integrate_contract_sparse_quartic_exp_by_parts_keeps_direct_trace() {
    let input = "integrate((x^4+x^2)*exp(2*x+1), x)";
    let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

    assert_eq!(
        wire["result"],
        "1/2·e^(2·x + 1)·(x^4 + 4·x^2 + 2 - 2·x^3 - 4·x)"
    );
    assert!(
        !stderr.contains("depth_overflow"),
        "sparse quartic exp by-parts presentation should not emit depth_overflow warning\nstderr:\n{stderr}"
    );

    let steps = wire["steps"]
        .as_array()
        .expect("steps should be present with --steps on");
    assert_eq!(
        steps.len(),
        2,
        "expected direct integration trace without expansion noise for {input}, got {steps:?}"
    );
    assert!(
        steps
            .iter()
            .all(|step| step["rule"] != "Expandir la expresión"),
        "sparse quartic exp by-parts should not expand before integrating, got {steps:?}"
    );
    let integration_step = steps
        .iter()
        .find(|step| step["rule"] == "Calcular la integral")
        .expect("expected public symbolic integration step");
    let substeps = integration_step["substeps"]
        .as_array()
        .expect("integration step should expose didactic substeps");
    assert!(
        substeps
            .iter()
            .any(|substep| substep["title"] == "Usar integración por partes repetida"),
        "expected repeated integration-by-parts substep for {input}, got {substeps:?}"
    );

    let residual = "diff(integrate((x^4+x^2)*exp(2*x+1), x), x) - (x^4+x^2)*exp(2*x+1)";
    let (wire, stderr) = cli_eval_json_with_stderr(residual);
    assert_eq!(wire["result"], "0");
    assert!(
        !stderr.contains("depth_overflow"),
        "nested sparse quartic exp verification should not emit depth_overflow warning\nstderr:\n{stderr}"
    );
}

#[test]
fn integrate_contract_sparse_affine_sextic_exp_by_parts_keeps_direct_trace() {
    let input = "integrate((x^6+1)*exp(2*x+1), x)";
    let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

    assert_eq!(
        wire["result"],
        "1/8·e^(2·x + 1)·(4·x^6 + 30·x^4 + 90·x^2 + 49 - 12·x^5 - 60·x^3 - 90·x)"
    );
    assert!(
        wire["required_conditions"]
            .as_array()
            .expect("required_conditions should be an array")
            .is_empty(),
        "unexpected required conditions for {input}: {:?}",
        wire["required_conditions"]
    );
    assert!(
        !stderr.contains("depth_overflow"),
        "sparse affine sextic exp by-parts presentation should not emit depth_overflow warning\nstderr:\n{stderr}"
    );

    let steps = wire["steps"]
        .as_array()
        .expect("steps should be present with --steps on");
    assert!(
        steps.len() <= 2,
        "expected compact integration trace without expansion noise for {input}, got {steps:?}"
    );
    assert!(
        steps
            .iter()
            .all(|step| step["rule"] != "Expandir la expresión"),
        "sparse affine sextic exp by-parts should not expand before integrating, got {steps:?}"
    );
    let integration_step = steps
        .iter()
        .find(|step| step["rule"] == "Calcular la integral")
        .expect("expected public symbolic integration step");
    let substeps = integration_step["substeps"]
        .as_array()
        .expect("integration step should expose didactic substeps");
    assert!(
        substeps
            .iter()
            .any(|substep| substep["title"] == "Usar integración por partes repetida"),
        "expected repeated integration-by-parts substep for {input}, got {substeps:?}"
    );
    assert_eq!(
        assert_antiderivative_verifies(input),
        AntiderivativeVerificationRoute::PublicResidual,
        "{input} should verify through the bounded public residual route"
    );

    let residual = "diff(integrate((x^6+1)*exp(2*x+1), x), x) - (x^6+1)*exp(2*x+1)";
    let (wire, stderr) = cli_eval_json_with_stderr(residual);
    assert_eq!(wire["result"], "0");
    assert!(
        wire["required_conditions"]
            .as_array()
            .expect("required_conditions should be an array")
            .is_empty(),
        "unexpected required conditions for {residual}: {:?}",
        wire["required_conditions"]
    );
    assert!(
        !stderr.contains("depth_overflow"),
        "nested sparse affine sextic exp verification should not emit depth_overflow warning\nstderr:\n{stderr}"
    );
}

#[test]
fn integrate_contract_repeated_trig_by_parts_exposes_didactic_substep() {
    for (input, expected_result) in [
        ("integrate(x^2*sin(x), x)", "2·x·sin(x) + (2 - x^2)·cos(x)"),
        ("integrate(x^2*cos(x), x)", "2·x·cos(x) + (x^2 - 2)·sin(x)"),
        (
            "integrate(x^3*sin(x), x)",
            "(6·x - x^3)·cos(x) + (3·x^2 - 6)·sin(x)",
        ),
        (
            "integrate(x^3*cos(x), x)",
            "(x^3 - 6·x)·sin(x) + (3·x^2 - 6)·cos(x)",
        ),
        (
            "integrate(x^4*sin(2*x+1), x)",
            "(x^3 - 3/2·x)·sin(2·x + 1) + (-1/2·x^4 + 3/2·x^2 - 3/4)·cos(2·x + 1)",
        ),
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

        assert_eq!(wire["result"], expected_result);
        assert!(
            !stderr.contains("depth_overflow"),
            "repeated trig by-parts didactic trace should not emit depth_overflow warning for {input}\nstderr:\n{stderr}"
        );

        let steps = wire["steps"]
            .as_array()
            .expect("steps should be present with --steps on");
        let integration_step = steps
            .iter()
            .find(|step| step["rule"] == "Calcular la integral")
            .expect("expected public symbolic integration step");
        let substeps = integration_step["substeps"]
            .as_array()
            .expect("integration step should expose didactic substeps");
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == "Usar integración por partes repetida"),
            "expected repeated integration-by-parts substep for {input}, got {substeps:?}"
        );
        assert_antiderivative_verifies(input);
    }
}

#[test]
fn integrate_contract_sparse_quartic_trig_by_parts_keeps_direct_trace() {
    let input = "integrate((x^4+x^2)*sin(2*x+1), x)";
    let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

    assert_eq!(
        wire["result"],
        "(x^3 - x)·sin(2·x + 1) + (-1/2·x^4 + x^2 - 1/2)·cos(2·x + 1)"
    );
    assert!(
        !stderr.contains("depth_overflow"),
        "sparse quartic trig by-parts presentation should not emit depth_overflow warning\nstderr:\n{stderr}"
    );

    let steps = wire["steps"]
        .as_array()
        .expect("steps should be present with --steps on");
    assert_eq!(
        steps.len(),
        1,
        "expected direct integration trace without expansion noise for {input}, got {steps:?}"
    );
    assert!(
        steps
            .iter()
            .all(|step| step["rule"] != "Expandir la expresión"),
        "sparse quartic trig by-parts should not expand before integrating, got {steps:?}"
    );
    let integration_step = steps
        .iter()
        .find(|step| step["rule"] == "Calcular la integral")
        .expect("expected public symbolic integration step");
    let substeps = integration_step["substeps"]
        .as_array()
        .expect("integration step should expose didactic substeps");
    assert!(
        substeps
            .iter()
            .any(|substep| substep["title"] == "Usar integración por partes repetida"),
        "expected repeated integration-by-parts substep for {input}, got {substeps:?}"
    );

    let residual = "diff(integrate((x^4+x^2)*sin(2*x+1), x), x) - (x^4+x^2)*sin(2*x+1)";
    let (wire, stderr) = cli_eval_json_with_stderr(residual);
    assert_eq!(wire["result"], "0");
    assert!(
        !stderr.contains("depth_overflow"),
        "nested sparse quartic trig verification should not emit depth_overflow warning\nstderr:\n{stderr}"
    );
}

#[test]
fn integrate_contract_affine_cubic_trig_by_parts_avoids_depth_overflow() {
    let input = "integrate((x^3+x)*sin(2*x+1), x)";
    let (wire, stderr) = cli_eval_json_with_stderr(input);

    assert_eq!(
        wire["result"],
        "(1/4·x - 1/2·x^3)·cos(2·x + 1) + (3/4·x^2 - 1/8)·sin(2·x + 1)"
    );
    assert!(
        !stderr.contains("depth_overflow"),
        "affine cubic trig by-parts presentation should not emit depth_overflow warning\nstderr:\n{stderr}"
    );

    let residual = "diff(integrate((x^3+x)*sin(2*x+1), x), x) - (x^3+x)*sin(2*x+1)";
    let (wire, stderr) = cli_eval_json_with_stderr(residual);
    assert_eq!(wire["result"], "0");
    assert!(
        !stderr.contains("depth_overflow"),
        "nested affine cubic trig verification should not emit depth_overflow warning\nstderr:\n{stderr}"
    );
}

#[test]
fn integrate_contract_affine_quartic_trig_by_parts_verifies() {
    for (input, expected_result, residual) in [
        (
            "integrate(x^4*sin(2*x+1), x)",
            "(x^3 - 3/2·x)·sin(2·x + 1) + (-1/2·x^4 + 3/2·x^2 - 3/4)·cos(2·x + 1)",
            "diff(integrate(x^4*sin(2*x+1), x), x) - x^4*sin(2*x+1)",
        ),
        (
            "integrate(x^4*cos(2*x+1), x)",
            "(x^3 - 3/2·x)·cos(2·x + 1) + (1/2·x^4 - 3/2·x^2 + 3/4)·sin(2·x + 1)",
            "diff(integrate(x^4*cos(2*x+1), x), x) - x^4*cos(2*x+1)",
        ),
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr(input);

        assert_eq!(wire["result"], expected_result);
        assert!(
            !stderr.contains("depth_overflow"),
            "affine quartic trig by-parts presentation should not emit depth_overflow warning for {input}\nstderr:\n{stderr}"
        );

        let (wire, stderr) = cli_eval_json_with_stderr(residual);
        assert_eq!(wire["result"], "0");
        assert!(
            !stderr.contains("depth_overflow"),
            "nested affine quartic trig verification should not emit depth_overflow warning for {input}\nstderr:\n{stderr}"
        );
    }
}

#[test]
fn integrate_contract_linear_by_parts_exposes_didactic_substep() {
    for (input, expected_result) in [
        ("integrate(x*exp(x), x)", "(x - 1)·e^x"),
        ("integrate(x*sin(x), x)", "sin(x) - x·cos(x)"),
        ("integrate(x*cos(x), x)", "cos(x) + x·sin(x)"),
        ("integrate(x*sinh(x), x)", "x·cosh(x) - sinh(x)"),
        ("integrate(x*cosh(x), x)", "x·sinh(x) - cosh(x)"),
        ("integrate((2*x+3)*exp(2*x+1), x)", "(x + 1)·e^(2·x + 1)"),
        (
            "integrate((2*x+3)*sin(2*x+1), x)",
            "1/2·sin(2·x + 1) - (cos(2·x + 1)·(2·x + 3))/2",
        ),
        (
            "integrate((2*x+3)*cos(2*x+1), x)",
            "1/2·cos(2·x + 1) + (sin(2·x + 1)·(2·x + 3))/2",
        ),
        (
            "integrate((2*x+3)*sinh(2*x+1), x)",
            "(cosh(2·x + 1)·(2·x + 3))/2 - 1/2·sinh(2·x + 1)",
        ),
        (
            "integrate((2*x+3)*cosh(2*x+1), x)",
            "(sinh(2·x + 1)·(2·x + 3))/2 - 1/2·cosh(2·x + 1)",
        ),
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

        assert_eq!(wire["result"], expected_result);
        assert!(
            !stderr.contains("depth_overflow"),
            "linear by-parts didactic trace should not emit depth_overflow warning for {input}\nstderr:\n{stderr}"
        );

        let steps = wire["steps"]
            .as_array()
            .expect("steps should be present with --steps on");
        if input.contains("2*x+3")
            && ["exp", "sin", "cos", "sinh", "cosh"]
                .iter()
                .any(|kernel| input.contains(kernel))
        {
            assert_eq!(
                steps.first().and_then(|step| step["rule"].as_str()),
                Some("Calcular la integral"),
                "affine by-parts should not expand before integration for {input}"
            );
            assert!(
                !steps
                    .iter()
                    .any(|step| step["rule"] == "Expandir la expresión"),
                "affine by-parts should preserve compact presentation for {input}, got {steps:?}"
            );
        }
        let integration_step = steps
            .iter()
            .find(|step| step["rule"] == "Calcular la integral")
            .expect("expected public symbolic integration step");
        let substeps = integration_step["substeps"]
            .as_array()
            .expect("integration step should expose didactic substeps");
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == "Usar integración por partes"),
            "expected integration-by-parts substep for {input}, got {substeps:?}"
        );
        assert_antiderivative_verifies(input);
    }
}

#[test]
fn integrate_contract_inverse_trig_by_parts_exposes_didactic_substep() {
    for (input, expected_result, expected_required_display) in [
        (
            "integrate(arcsin(x), x)",
            "sqrt(1 - x^2) + x·arcsin(x)",
            serde_json::json!(["-1 < x < 1"]),
        ),
        (
            "integrate(arccos(x), x)",
            "x·arccos(x) - sqrt(1 - x^2)",
            serde_json::json!(["-1 < x < 1"]),
        ),
        (
            "integrate(arctan(x), x)",
            "-1/2·ln(x^2 + 1) + x·arctan(x)",
            serde_json::json!([]),
        ),
        (
            "integrate(arctan(1/(2*x+1)), x)",
            "1/4·ln((2·x + 1)^2 + 1) + 1/2·(2·x + 1)·arctan(1 / (2·x + 1))",
            serde_json::json!(["x ≠ -1/2"]),
        ),
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

        assert_eq!(wire["result"], expected_result, "input: {input}");
        assert_eq!(
            wire["required_display"], expected_required_display,
            "unexpected required_display for {input}: {:?}",
            wire["required_display"]
        );
        assert!(
            !stderr.contains("depth_overflow"),
            "inverse-trig by-parts trace should not emit depth_overflow warning for {input}\nstderr:\n{stderr}"
        );

        let steps = wire["steps"]
            .as_array()
            .expect("steps should be present with --steps on");
        assert_eq!(
            steps.len(),
            1,
            "expected compact direct integration trace for {input}, got {steps:?}"
        );
        let integration_step = steps
            .iter()
            .find(|step| step["rule"] == "Calcular la integral")
            .expect("expected public symbolic integration step");
        let substeps = integration_step["substeps"]
            .as_array()
            .expect("integration step should expose didactic substeps");
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == "Usar integración por partes"),
            "expected integration-by-parts substep for {input}, got {substeps:?}"
        );
        assert_antiderivative_verifies(input);
    }
}

#[test]
fn integrate_contract_inverse_hyperbolic_affine_by_parts_exposes_didactic_substep() {
    for (input, expected_result, expected_required_display) in [
        (
            "integrate(asinh(2*x+1), x)",
            "1/2·(2·x + 1)·asinh(2·x + 1) - 1/2·sqrt((2·x + 1)^2 + 1)",
            serde_json::json!([]),
        ),
        (
            "integrate(atanh(2*x+1), x)",
            "1/4·ln(1 - (2·x + 1)^2) + 1/2·(2·x + 1)·atanh(2·x + 1)",
            serde_json::json!(["-1 < x < 0"]),
        ),
        (
            "integrate(acosh(2*x+1), x)",
            "1/2·(2·x + 1)·acosh(2·x + 1) - 1/2·sqrt(2·x)·sqrt(2·x + 2)",
            serde_json::json!(["x > 0"]),
        ),
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

        assert_eq!(wire["result"], expected_result, "input: {input}");
        assert_eq!(
            wire["required_display"], expected_required_display,
            "unexpected required_display for {input}: {:?}",
            wire["required_display"]
        );
        assert!(
            !stderr.contains("depth_overflow"),
            "inverse-hyperbolic by-parts trace should not emit depth_overflow warning for {input}\nstderr:\n{stderr}"
        );

        let steps = wire["steps"]
            .as_array()
            .expect("steps should be present with --steps on");
        assert_eq!(
            steps.len(),
            1,
            "expected compact direct integration trace for {input}, got {steps:?}"
        );
        let integration_step = steps
            .iter()
            .find(|step| step["rule"] == "Calcular la integral")
            .expect("expected public symbolic integration step");
        let substeps = integration_step["substeps"]
            .as_array()
            .expect("integration step should expose didactic substeps");
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == "Usar integración por partes"),
            "expected integration-by-parts substep for {input}, got {substeps:?}"
        );
        assert_antiderivative_verifies(input);
    }
}

#[test]
fn integrate_contract_linear_exp_by_parts_steps_keep_compact_presentation() {
    let input = "integrate(x*exp(x), x)";
    let (wire, _stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);
    let steps = wire["steps"]
        .as_array()
        .expect("steps should be present with --steps on");
    let rules: Vec<_> = steps.iter().map(|step| step["rule"].as_str()).collect();

    assert_eq!(wire["result"], "(x - 1)·e^x");
    assert!(
        !rules.contains(&Some("Expandir la expresión")),
        "linear exp by-parts should not expand the compact antiderivative: {rules:?}"
    );
    assert!(
        !rules.contains(&Some("Sacar factor común")),
        "linear exp by-parts should not refactor immediately after expansion: {rules:?}"
    );
}

#[test]
fn integrate_contract_scaled_inverse_sqrt_polynomial_power_substitution() {
    let input = "integrate((4*x^3+6*x^2+6*x+2)/sqrt(2-3*(x^2+x+1)^4), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "arcsin((x^2 + x + 1)^2 / sqrt(2/3)) / sqrt(3)");
    assert_eq!(required, vec!["2 - 3 * (x^2 + x + 1)^4 > 0".to_string()]);
    assert_antiderivative_equiv_verifies(input);
    assert_inverse_trig_polynomial_substitution_keeps_compact_steps(input);

    let scaled = "integrate(2*(2*x^3+3*x^2+3*x+1)*sqrt(3)/sqrt(2-3*(x^2+x+1)^4), x)";
    let (result, required) = evaluated_integral_with_required_conditions(scaled);
    assert_eq!(result, "arcsin((x^2 + x + 1)^2 / sqrt(2/3))");
    assert_eq!(required, vec!["2 - 3 * (x^2 + x + 1)^4 > 0".to_string()]);
    assert_antiderivative_equiv_verifies(scaled);
    assert_inverse_trig_polynomial_substitution_keeps_compact_steps(scaled);
}

fn assert_inverse_trig_polynomial_substitution_keeps_compact_steps(input: &str) {
    let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

    assert!(
        !stderr.contains("depth_overflow"),
        "inverse-trig polynomial substitution should not emit depth_overflow warning for {input}\nstderr:\n{stderr}"
    );
    let steps = wire["steps"]
        .as_array()
        .expect("steps should be present with --steps on");
    assert_eq!(
        steps.len(),
        1,
        "expected compact direct substitution trace for {input}, got {steps:?}"
    );
    assert!(
        steps.iter().all(|step| step["rule"] != "Expandir binomio"),
        "inverse-trig polynomial substitution should not expand the radicand before integrating: {steps:?}"
    );
    let integration_step = steps
        .iter()
        .find(|step| step["rule"] == "Calcular la integral")
        .expect("expected public symbolic integration step");
    let substeps = integration_step["substeps"]
        .as_array()
        .expect("integration step should expose didactic substeps");
    assert!(
        substeps
            .iter()
            .any(|substep| substep["title"] == "Usar sustitución"),
        "expected substitution substep for {input}, got {substeps:?}"
    );
}

#[test]
fn integrate_contract_symbolic_constant_verification_preserves_independent_domain_conditions() {
    let input = "integrate(ln(y)*(z+1)^(-2), x)";
    let (result, mut required) = evaluated_integral_with_required_conditions(input);
    required.sort();

    assert_eq!(result, "x * ln(y) / (z + 1)^2");
    assert_eq!(
        required,
        vec!["y > 0".to_string(), "z ≠ -1".to_string()],
        "symbolic constant integration should publish independent domain conditions"
    );

    let (nested_residual, mut nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate(ln(y)*(z+1)^(-2), x), x) - ln(y)*(z+1)^(-2)",
    );
    nested_required.sort();
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["y > 0".to_string(), "z ≠ -1".to_string()],
        "nested antiderivative verification should preserve independent domain conditions"
    );
}

#[test]
#[ignore = "exhaustive debug verification is intentionally slower; CI runs the representative smoke test"]
fn integrate_contract_supported_antiderivatives_verify_by_differentiation_exhaustive() {
    let public_residual_inputs: Vec<_> = [
        "integrate(2*x + 3, x)",
        "integrate((3*x)^2, x)",
        "integrate(sin(2*x), x)",
        "integrate(-(sin(x)), x)",
        "integrate(cos(x), x)",
        "integrate(exp(3*x + 1), x)",
        "integrate(x*exp(x), x)",
        "integrate(x^2*exp(x), x)",
        "integrate(x^3*exp(x), x)",
        "integrate((2*x+3)*exp(2*x+1), x)",
        "integrate((x^2+x+1)*exp(2*x+1), x)",
        "integrate((x^3+x)*exp(2*x+1), x)",
        "integrate((x+1)*exp((3*x+2)/2), x)",
        "integrate((x+1)*exp((2-3*x)/2), x)",
        "integrate(x*sin(x), x)",
        "integrate(x*cos(x), x)",
        "integrate(x^2*sin(x), x)",
        "integrate(x^2*cos(x), x)",
        "integrate((2*x+3)*sin(2*x+1), x)",
        "integrate((2*x+3)*cos(2*x+1), x)",
        "integrate((x+1)*sin((3*x+2)/2), x)",
        "integrate((x+1)*cos((3*x+2)/2), x)",
        "integrate((x+1)*sin((2-3*x)/2), x)",
        "integrate((x+1)*cos((2-3*x)/2), x)",
        "integrate(x*sinh(x), x)",
        "integrate(x*cosh(x), x)",
        "integrate((2*x+3)*sinh(2*x+1), x)",
        "integrate((2*x+3)*cosh(2*x+1), x)",
        "integrate((x+1)*sinh((3*x+2)/2), x)",
        "integrate((x+1)*cosh((3*x+2)/2), x)",
        "integrate((x+1)*sinh((2-3*x)/2), x)",
        "integrate((x+1)*cosh((2-3*x)/2), x)",
        "integrate(x^2*sinh(x), x)",
        "integrate(x^2*cosh(x), x)",
        "integrate(2*x*exp(x^2), x)",
        "integrate(2*x*cos(x^2), x)",
        "integrate(2*x*sin(x^2), x)",
        "integrate(sinh(2*x + 1), x)",
        "integrate(cosh(2*x + 1), x)",
        "integrate(tanh(2*x + 1), x)",
        "integrate(2*x*sinh(x^2), x)",
        "integrate(2*x*cosh(x^2), x)",
        "integrate(2*x*tanh(x^2), x)",
        "integrate(cosh(x)/(1+sinh(x)^2), x)",
        "integrate(2*cosh(2*x+1)/(1+sinh(2*x+1)^2), x)",
        "integrate(sinh(x)/(1+cosh(x)^2), x)",
        "integrate(sinh(2*x + 1)/cosh(2*x + 1), x)",
        "integrate(cosh(2*x + 1)/sinh(2*x + 1), x)",
        "integrate(1/tanh(2*x + 1), x)",
        "integrate(2*x*cosh(x^2)/sinh(x^2), x)",
        "integrate(2*x/tanh(x^2), x)",
        "integrate(1/cosh(2*x + 1)^2, x)",
        "integrate(1/sinh(2*x + 1)^2, x)",
        "integrate(2*x/cosh(x^2)^2, x)",
        "integrate(2*x/sinh(x^2)^2, x)",
        "integrate(sinh(2*x + 1)/cosh(2*x + 1)^2, x)",
        "integrate(2*x*sinh(x^2)/cosh(x^2)^2, x)",
        "integrate(-x*sinh(x^2)/cosh(x^2)^2, x)",
        "integrate(cosh(2*x + 1)/sinh(2*x + 1)^2, x)",
        "integrate(2*x*cosh(x^2)/sinh(x^2)^2, x)",
        "integrate(-x*cosh(x^2)/sinh(x^2)^2, x)",
        "integrate(ln(x), x)",
        "integrate(ln(2*x+1), x)",
        "integrate(2*x*ln(x^2+1), x)",
        "integrate((2*x+1)*ln(x^2+x+1), x)",
        "integrate(2*x*ln(x^2-1), x)",
        "integrate(1/x, x)",
        "integrate(1/(2*x + 1), x)",
        "integrate(1/(x^2+1), x)",
        "integrate(2*x/(1+x^4), x)",
        "integrate(2*x/(4+x^4), x)",
        "integrate(2*x/(4-x^4), x)",
        "integrate(arcsin(x), x)",
        "integrate(arccos(x), x)",
        "integrate(arcsin(2*x+1), x)",
        "integrate(arccos(2*x+1), x)",
        "integrate(arcsin(1-2*x), x)",
        "integrate(arccos(1-2*x), x)",
        "integrate(arctan(x), x)",
        "integrate(arctan(2*x), x)",
        "integrate(arctan(1/x), x)",
        "integrate(arctan(1/(2*x+1)), x)",
        "integrate(arccot(x), x)",
        "integrate(arccot(2*x), x)",
        "integrate(arccot(2*x+1), x)",
        "integrate(arccot(1-2*x), x)",
        "integrate(asinh(x), x)",
        "integrate(asinh(2*x), x)",
        "integrate(asinh(2*x+1), x)",
        "integrate(asinh(1-2*x), x)",
        "integrate(atanh(x), x)",
        "integrate(atanh(2*x), x)",
        "integrate(atanh(2*x+1), x)",
        "integrate(atanh(1-2*x), x)",
        "integrate(1/(x^2-1), x)",
        "integrate(2*x/(x^2-1)^2, x)",
        "integrate((2*x+1)/(x^2+x-1)^2, x)",
        "integrate((2*x+1)/(3*(x^2+x-1)^2), x)",
        "integrate((2*x+1)/(x^2+x-1)^3, x)",
        "integrate((3*x+3/2)/(x^2+x-1)^4, x)",
        "integrate((8*x+2)/(3*(2*x^2+x-1)^3), x)",
        "integrate((2*x+1)/(x^4+2*x^3-x^2-2*x+1), x)",
        "integrate((2*x+1)/(x^6+3*x^5-5*x^3+3*x-1), x)",
        "integrate((2*x+1)/(4*x^6+12*x^5-20*x^3+12*x-4), x)",
        "integrate((2*x+1)/(-4*x^6-12*x^5+20*x^3-12*x+4), x)",
        "integrate((3*x+3/2)/(x^8+4*x^7+2*x^6-8*x^5-5*x^4+8*x^3+2*x^2-4*x+1), x)",
        "integrate(1/(x^5+5*x^4+10*x^3+10*x^2+5*x+1), x)",
        "integrate(2*x/(x^4-4), x)",
        "integrate((2*x + 1)/(x^2 + x - 1), x)",
        "integrate((4*x + 2)/(x^2 + x + 1), x)",
        "integrate(2*x/sqrt(4-x^4), x)",
        "integrate(2*x/sqrt(3-x^4), x)",
        "integrate(1/sqrt(4-(x+1)^2), x)",
        "integrate(2*x/sqrt(1+x^4), x)",
        "integrate(2*x/sqrt(4+x^4), x)",
        "integrate(2*x/sqrt(3+x^4), x)",
        "integrate(1/sqrt(4+(x+1)^2), x)",
        "integrate(3/(sqrt(5-3*x)*(1-3*x)), x)",
        "integrate(x/sqrt(x^2+1), x)",
        "integrate((3*x+5)/(2*sqrt(x+2)), x)",
        "integrate(2*x/sqrt(x^2-1), x)",
        "integrate(x*sqrt(x^2+1), x)",
        "integrate(2*x*sqrt(x^2-1), x)",
        "integrate(2*x*(x^2+1)^3, x)",
        "integrate(2*x*(x^2-1)^(3/2), x)",
        "integrate((x^2+1)^(-1/2), x)",
        "integrate(1/(x^2+1)^2, x)",
        "integrate(1/((x+1)^2+1)^2, x)",
        "integrate(1/(4*x^2+1)^2, x)",
        "integrate(sin(x)^2, x)",
        "integrate(cos(x)^2, x)",
        "integrate(sin(2*x + 1)^2, x)",
        "integrate(cos(2*x + 1)^2, x)",
        "integrate(sin(x)^3, x)",
        "integrate(cos(x)^3, x)",
        "integrate(sin(2*x + 1)^3, x)",
        "integrate(cos(2*x + 1)^3, x)",
        "integrate(sec(x)^2, x)",
        "integrate(csc(x)^2, x)",
        "integrate(sec(2*x + 1)^2, x)",
        "integrate(csc(2*x + 1)^2, x)",
        "integrate(x/(cos(x^2)^2), x)",
        "integrate(x^2/(sin(x^3)^2), x)",
        "integrate(sec(x), x)",
        "integrate(csc(x), x)",
        "integrate(csc(x)*cot(x), x)",
        "integrate(tan(2*x + 1), x)",
        "integrate(cot(2*x + 1), x)",
        "integrate(2*x*tan(x^2), x)",
        "integrate(3*x^2*cot(x^3), x)",
        "integrate(2*(x*sin(x^2)/cos(x^2)), x)",
        "integrate(3*(x^2*cos(x^3)/sin(x^3)), x)",
        "integrate(sec((3*x+2)/2), x)",
        "integrate(csc((2-3*x)/2), x)",
        "integrate(sec(2*x + 1)*tan(2*x + 1), x)",
        "integrate(x*sec(x^2)*tan(x^2), x)",
        "integrate(2*x*sec(x^2)*tan(x^2), x)",
        "integrate(-x*sec(x^2)*tan(x^2), x)",
        "integrate(csc(2*x + 1)*cot(2*x + 1), x)",
        "integrate(x^2*csc(x^3)*cot(x^3), x)",
        "integrate(3*x^2*csc(x^3)*cot(x^3), x)",
        "integrate(-x^2*csc(x^3)*cot(x^3), x)",
    ]
    .into_iter()
    .filter(|input| {
        assert_antiderivative_verifies(input) == AntiderivativeVerificationRoute::PublicResidual
    })
    .collect();

    assert_eq!(
        public_residual_inputs,
        vec![
            "integrate(x^2*exp(x), x)",
            "integrate(x^3*exp(x), x)",
            "integrate((x^2+x+1)*exp(2*x+1), x)",
            "integrate((x^3+x)*exp(2*x+1), x)",
            "integrate(x*sin(x), x)",
            "integrate(x*cos(x), x)",
            "integrate(x^2*sin(x), x)",
            "integrate(x^2*cos(x), x)",
            "integrate((2*x+3)*sin(2*x+1), x)",
            "integrate((2*x+3)*cos(2*x+1), x)",
            "integrate((x+1)*sin((3*x+2)/2), x)",
            "integrate((x+1)*cos((3*x+2)/2), x)",
            "integrate((x+1)*sin((2-3*x)/2), x)",
            "integrate((x+1)*cos((2-3*x)/2), x)",
            "integrate(x*sinh(x), x)",
            "integrate(x*cosh(x), x)",
            "integrate((2*x+3)*sinh(2*x+1), x)",
            "integrate((2*x+3)*cosh(2*x+1), x)",
            "integrate((x+1)*sinh((3*x+2)/2), x)",
            "integrate((x+1)*cosh((3*x+2)/2), x)",
            "integrate((x+1)*sinh((2-3*x)/2), x)",
            "integrate((x+1)*cosh((2-3*x)/2), x)",
            "integrate(x^2*sinh(x), x)",
            "integrate(x^2*cosh(x), x)",
            "integrate(sinh(2*x + 1)/cosh(2*x + 1), x)",
            "integrate(cosh(2*x + 1)/sinh(2*x + 1), x)",
            "integrate(1/tanh(2*x + 1), x)",
            "integrate(2*x*cosh(x^2)/sinh(x^2), x)",
            "integrate(2*x/tanh(x^2), x)",
            "integrate(1/cosh(2*x + 1)^2, x)",
            "integrate(1/sinh(2*x + 1)^2, x)",
            "integrate(2*x/cosh(x^2)^2, x)",
            "integrate(2*x/sinh(x^2)^2, x)",
            "integrate(sinh(2*x + 1)/cosh(2*x + 1)^2, x)",
            "integrate(2*x*sinh(x^2)/cosh(x^2)^2, x)",
            "integrate(-x*sinh(x^2)/cosh(x^2)^2, x)",
            "integrate(cosh(2*x + 1)/sinh(2*x + 1)^2, x)",
            "integrate(2*x*cosh(x^2)/sinh(x^2)^2, x)",
            "integrate(-x*cosh(x^2)/sinh(x^2)^2, x)",
            "integrate(2*x*ln(x^2+1), x)",
            "integrate((2*x+1)*ln(x^2+x+1), x)",
            "integrate((3*x+5)/(2*sqrt(x+2)), x)",
            "integrate(sec(x), x)",
            "integrate(csc(x), x)",
            "integrate(tan(2*x + 1), x)",
            "integrate(cot(2*x + 1), x)",
            "integrate(sec((3*x+2)/2), x)",
            "integrate(csc((2-3*x)/2), x)",
        ],
        "exhaustive debug antiderivative verification should only use the bounded public residual route for known public-residual families"
    );
}

#[test]
fn integrate_contract_linear_sine_substitution() {
    assert_eq!(
        simplified_integral("integrate(sin(2*x), x)"),
        "-1/2 * cos(2 * x)"
    );
}

#[test]
fn integrate_contract_affine_trig_square_power_reduction() {
    assert_eq!(
        simplified_integral("integrate(sin(x)^2, x)"),
        "1/4 * (2 * x - sin(2 * x))"
    );
    assert_eq!(
        simplified_integral("integrate(cos(x)^2, x)"),
        "1/4 * (sin(2 * x) + 2 * x)"
    );
    assert_eq!(
        simplified_integral("integrate(sin(2*x + 1)^2, x)"),
        "1/8 * (4 * x - sin(4 * x + 2))"
    );
    assert_eq!(
        simplified_integral("integrate(cos(2*x + 1)^2, x)"),
        "1/8 * (sin(4 * x + 2) + 4 * x)"
    );

    let public_cases = [
        ("integrate(sin(x)^2, x)", "1/2 * x - 1/4 * sin(2 * x)"),
        ("integrate(cos(x)^2, x)", "1/4 * sin(2 * x) + 1/2 * x"),
        (
            "integrate(sin(2*x + 1)^2, x)",
            "1/2 * x - 1/8 * sin(4 * x + 2)",
        ),
        (
            "integrate(cos(2*x + 1)^2, x)",
            "1/8 * sin(4 * x + 2) + 1/2 * x",
        ),
    ];

    for (input, expected) in public_cases {
        let (result, required) = evaluated_integral_with_required_conditions(input);
        assert_eq!(result, expected, "input: {input}");
        assert!(
            required.is_empty(),
            "input: {input}, required: {required:?}"
        );
        assert_rendered_antiderivative_verifies(input, expected);
    }
}

#[test]
fn integrate_contract_affine_hyperbolic_square_power_reduction() {
    let public_cases = [
        ("integrate(sinh(x)^2, x)", "1/2 * sinh(x) * cosh(x) - x / 2"),
        ("integrate(cosh(x)^2, x)", "1/2 * sinh(x) * cosh(x) + x / 2"),
        ("integrate(sinh(2*x)^2, x)", "1/8 * sinh(4 * x) - 1/2 * x"),
        (
            "integrate(sinh(2*x + 1)^2, x)",
            "1/4 * sinh(2 * x + 1) * cosh(2 * x + 1) - x / 2",
        ),
        (
            "integrate(cosh(2*x + 1)^2, x)",
            "1/4 * sinh(2 * x + 1) * cosh(2 * x + 1) + x / 2",
        ),
        ("integrate(tanh(x)^2, x)", "x - tanh(x)"),
        ("integrate(tanh(2*x + 1)^2, x)", "x - 1/2 * tanh(2 * x + 1)"),
        ("integrate(tanh(1-2*x)^2, x)", "1/2 * tanh(1 - 2 * x) + x"),
        (
            "integrate(tanh(x)^4, x)",
            "1/3 * (3 * x - tanh(x)^3 - 3 * tanh(x))",
        ),
        (
            "integrate(tanh(2*x + 1)^4, x)",
            "1/6 * (6 * x - tanh(2 * x + 1)^3 - 3 * tanh(2 * x + 1))",
        ),
        (
            "integrate(tanh(1-2*x)^4, x)",
            "1/6 * (tanh(1 - 2 * x)^3 + 3 * tanh(1 - 2 * x) + 6 * x)",
        ),
        (
            "integrate(tanh(x)^6, x)",
            "1/15 * (15 * x - 15 * tanh(x) - 5 * tanh(x)^3 - 3 * tanh(x)^5)",
        ),
        (
            "integrate(tanh(2*x + 1)^6, x)",
            "1/30 * (30 * x - 15 * tanh(2 * x + 1) - 5 * tanh(2 * x + 1)^3 - 3 * tanh(2 * x + 1)^5)",
        ),
        (
            "integrate(tanh(1-2*x)^6, x)",
            "1/30 * (3 * tanh(1 - 2 * x)^5 + 5 * tanh(1 - 2 * x)^3 + 15 * tanh(1 - 2 * x) + 30 * x)",
        ),
        (
            "integrate(tanh(x)^8, x)",
            "x - (tanh(x) + 1/7 * tanh(x)^7 + 1/5 * tanh(x)^5 + 1/3 * tanh(x)^3)",
        ),
        (
            "integrate(tanh(2*x + 1)^8, x)",
            "x - (tanh(2 * x + 1) + 1/7 * tanh(2 * x + 1)^7 + 1/5 * tanh(2 * x + 1)^5 + 1/3 * tanh(2 * x + 1)^3) / 2",
        ),
        (
            "integrate(tanh(1-2*x)^8, x)",
            "(tanh(1 - 2 * x) + 1/7 * tanh(1 - 2 * x)^7 + 1/5 * tanh(1 - 2 * x)^5 + 1/3 * tanh(1 - 2 * x)^3) / 2 + x",
        ),
        (
            "integrate(4*sinh(x)^2*cosh(x)^2, x)",
            "1/8 * sinh(4 * x) - 1/2 * x",
        ),
        (
            "integrate(sinh(2*x + 1)^2*cosh(2*x + 1)^2, x)",
            "1/64 * sinh(4 * (2 * x + 1)) - 1/8 * x",
        ),
    ];

    for (input, expected) in public_cases {
        let (result, required) = evaluated_integral_with_required_conditions(input);
        assert_eq!(result, expected, "input: {input}");
        assert!(
            required.is_empty(),
            "input: {input}, required: {required:?}"
        );
        assert_eq!(integrate_call_antiderivative_residual_result(input), "0");
    }
}

#[test]
fn integrate_contract_affine_tanh_eight_explicit_primitives_verify_publicly() {
    let residuals = [
        "diff(integrate(tanh(x)^8, x), x) - tanh(x)^8",
        "diff(integrate(tanh(2*x+1)^8, x), x) - tanh(2*x+1)^8",
        "diff(integrate(tanh(1-2*x)^8, x), x) - tanh(1-2*x)^8",
        "diff(x - 1/2*(tanh(2*x+1) + tanh(2*x+1)^3/3 + tanh(2*x+1)^5/5 + tanh(2*x+1)^7/7), x) - tanh(2*x+1)^8",
        "diff(x + 1/2*(tanh(1-2*x) + tanh(1-2*x)^3/3 + tanh(1-2*x)^5/5 + tanh(1-2*x)^7/7), x) - tanh(1-2*x)^8",
    ];

    for input in residuals {
        let (wire, stderr) = cli_eval_json_with_stderr(input);
        assert!(
            stderr.is_empty(),
            "tanh eighth primitive residual should stay quiet for {input}: {stderr}"
        );
        assert_eq!(wire["result"], "0", "{input}");
        assert_eq!(wire["required_display"], serde_json::json!([]), "{input}");
        assert!(
            wire["warnings"]
                .as_array()
                .is_some_and(|warnings| warnings.is_empty()),
            "tanh eighth primitive residual should not warn for {input}: {wire:#}"
        );
    }
}

#[test]
fn integrate_contract_affine_tanh_six_explicit_primitives_verify_publicly() {
    let residuals = [
        "diff(x - tanh(x) - tanh(x)^3/3 - tanh(x)^5/5, x) - tanh(x)^6",
        "diff(x - 1/2*(tanh(2*x+1) + tanh(2*x+1)^3/3 + tanh(2*x+1)^5/5), x) - tanh(2*x+1)^6",
        "diff(x + 1/2*(tanh(1-2*x) + tanh(1-2*x)^3/3 + tanh(1-2*x)^5/5), x) - tanh(1-2*x)^6",
    ];

    for input in residuals {
        let (wire, stderr) = cli_eval_json_with_stderr(input);
        assert!(
            stderr.is_empty(),
            "tanh sixth primitive residual should stay quiet for {input}: {stderr}"
        );
        assert_eq!(wire["result"], "0", "{input}");
        assert_eq!(wire["required_display"], serde_json::json!([]), "{input}");
        assert!(
            wire["warnings"]
                .as_array()
                .is_some_and(|warnings| warnings.is_empty()),
            "tanh sixth primitive residual should not warn for {input}: {wire:#}"
        );
    }
}

#[test]
fn integrate_contract_affine_hyperbolic_cubic_power_reduction() {
    let public_cases = [
        (
            "integrate(sinh(2*x + 1)^3, x)",
            "1/2 * (1/3 * cosh(2 * x + 1)^3 - cosh(2 * x + 1))",
        ),
        (
            "integrate(cosh(2*x + 1)^3, x)",
            "1/2 * (sinh(2 * x + 1) + 1/3 * sinh(2 * x + 1)^3)",
        ),
        (
            "integrate(sinh(1 - 2*x)^3, x)",
            "1/6 * (3 * cosh(1 - 2 * x) - cosh(1 - 2 * x)^3)",
        ),
        (
            "integrate(cosh(1 - 2*x)^3, x)",
            "-1/2 * (sinh(1 - 2 * x) + 1/3 * sinh(1 - 2 * x)^3)",
        ),
    ];

    for (input, expected) in public_cases {
        let (result, required) = evaluated_integral_with_required_conditions(input);
        assert_eq!(result, expected, "input: {input}");
        assert!(
            required.is_empty(),
            "input: {input}, required: {required:?}"
        );
        assert_eq!(
            assert_antiderivative_verifies(input),
            AntiderivativeVerificationRoute::PublicResidual,
            "{input} should verify through the bounded public residual route"
        );
    }
}

#[test]
fn integrate_contract_affine_trig_ratio_square_power_reduction() {
    let public_cases = [
        ("integrate(tan(x)^2, x)", "tan(x) - x", vec!["cos(x) ≠ 0"]),
        (
            "integrate(tan(2*x + 1)^2, x)",
            "1/2 * (tan(2 * x + 1) - 2 * x)",
            vec!["cos(2 * x + 1) ≠ 0"],
        ),
        (
            "integrate(tan(x)^4, x)",
            "tan(x)^3 / 3 + x - tan(x)",
            vec!["cos(x) ≠ 0"],
        ),
        (
            "integrate(tan(2*x + 1)^4, x)",
            "-tan(2 * x + 1) / 2 + tan(2 * x + 1)^3 / 6 + x",
            vec!["cos(2 * x + 1) ≠ 0"],
        ),
        (
            "integrate(sin(2*x + 1)^4/cos(2*x + 1)^4, x)",
            "-tan(2 * x + 1) / 2 + tan(2 * x + 1)^3 / 6 + x",
            vec!["cos(2 * x + 1) ≠ 0"],
        ),
        (
            "integrate(tan(x)^6, x)",
            "tan(x) + -tan(x)^3 / 3 + tan(x)^5 / 5 - x",
            vec!["cos(x) ≠ 0"],
        ),
        (
            "integrate(tan(2*x + 1)^6, x)",
            "tan(2 * x + 1) / 2 + -tan(2 * x + 1)^3 / 6 + tan(2 * x + 1)^5 / 10 - x",
            vec!["cos(2 * x + 1) ≠ 0"],
        ),
        (
            "integrate(tan(1-2*x)^6, x)",
            "-tan(1 - 2 * x) / 2 + -tan(1 - 2 * x)^5 / 10 + tan(1 - 2 * x)^3 / 6 - x",
            vec!["cos(1 - 2 * x) ≠ 0"],
        ),
        (
            "integrate(sin(2*x + 1)^6/cos(2*x + 1)^6, x)",
            "tan(2 * x + 1) / 2 + -tan(2 * x + 1)^3 / 6 + tan(2 * x + 1)^5 / 10 - x",
            vec!["cos(2 * x + 1) ≠ 0"],
        ),
        (
            "integrate(tan(x)^8, x)",
            "-tan(x)^5 / 5 + tan(x)^3 / 3 + tan(x)^7 / 7 + x - tan(x)",
            vec!["cos(x) ≠ 0"],
        ),
        (
            "integrate(tan(2*x + 1)^8, x)",
            "-tan(2 * x + 1) / 2 + -tan(2 * x + 1)^5 / 10 + tan(2 * x + 1)^3 / 6 + tan(2 * x + 1)^7 / 14 + x",
            vec!["cos(2 * x + 1) ≠ 0"],
        ),
        (
            "integrate(tan(1-2*x)^8, x)",
            "tan(1 - 2 * x) / 2 + -tan(1 - 2 * x)^3 / 6 + -tan(1 - 2 * x)^7 / 14 + tan(1 - 2 * x)^5 / 10 + x",
            vec!["cos(1 - 2 * x) ≠ 0"],
        ),
        (
            "integrate(sin(2*x + 1)^8/cos(2*x + 1)^8, x)",
            "-tan(2 * x + 1) / 2 + -tan(2 * x + 1)^5 / 10 + tan(2 * x + 1)^3 / 6 + tan(2 * x + 1)^7 / 14 + x",
            vec!["cos(2 * x + 1) ≠ 0"],
        ),
        (
            "integrate(sec(x)^4, x)",
            "tan(x) + tan(x)^3 / 3",
            vec!["cos(x) ≠ 0"],
        ),
        (
            "integrate(1/cos(2*x + 1)^4, x)",
            "tan(2 * x + 1) / 2 + tan(2 * x + 1)^3 / 6",
            vec!["cos(2 * x + 1) ≠ 0"],
        ),
        (
            "integrate(sec(x)^6, x)",
            "tan(x) + tan(x)^5 / 5 + 2 * tan(x)^3 / 3",
            vec!["cos(x) ≠ 0"],
        ),
        (
            "integrate(1/cos(2*x + 1)^6, x)",
            "tan(2 * x + 1) / 2 + tan(2 * x + 1)^3 / 3 + tan(2 * x + 1)^5 / 10",
            vec!["cos(2 * x + 1) ≠ 0"],
        ),
        (
            "integrate(sec(1-2*x)^6, x)",
            "-tan(1 - 2 * x) / 2 + -tan(1 - 2 * x)^3 / 3 + -tan(1 - 2 * x)^5 / 10",
            vec!["cos(1 - 2 * x) ≠ 0"],
        ),
        (
            "integrate(sec(x)^8, x)",
            "tan(x) + tan(x)^7 / 7 + 3 * tan(x)^5 / 5 + tan(x)^3",
            vec!["cos(x) ≠ 0"],
        ),
        (
            "integrate(1/cos(2*x + 1)^8, x)",
            "tan(2 * x + 1) / 2 + tan(2 * x + 1)^3 / 2 + tan(2 * x + 1)^7 / 14 + 3 * tan(2 * x + 1)^5 / 10",
            vec!["cos(2 * x + 1) ≠ 0"],
        ),
        (
            "integrate(sec(1-2*x)^8, x)",
            "-tan(1 - 2 * x) / 2 + -tan(1 - 2 * x)^3 / 2 + -tan(1 - 2 * x)^7 / 14 + -3 * tan(1 - 2 * x)^5 / 10",
            vec!["cos(1 - 2 * x) ≠ 0"],
        ),
        (
            "integrate(csc(x)^4, x)",
            "-cot(x)^3 / 3 - cot(x)",
            vec!["sin(x) ≠ 0"],
        ),
        (
            "integrate(1/sin(2*x + 1)^4, x)",
            "-cot(2 * x + 1)^3 / 6 - cot(2 * x + 1) / 2",
            vec!["sin(2 * x + 1) ≠ 0"],
        ),
        (
            "integrate(csc(x)^6, x)",
            "-cot(x)^5 / 5 + -2 * cot(x)^3 / 3 - cot(x)",
            vec!["sin(x) ≠ 0"],
        ),
        (
            "integrate(1/sin(2*x + 1)^6, x)",
            "-cot(2 * x + 1) / 2 + -cot(2 * x + 1)^3 / 3 + -cot(2 * x + 1)^5 / 10",
            vec!["sin(2 * x + 1) ≠ 0"],
        ),
        (
            "integrate(csc(1-2*x)^6, x)",
            "cot(1 - 2 * x) / 2 + cot(1 - 2 * x)^3 / 3 + cot(1 - 2 * x)^5 / 10",
            vec!["sin(1 - 2 * x) ≠ 0"],
        ),
        (
            "integrate(csc(x)^8, x)",
            "-cot(x)^7 / 7 + -3 * cot(x)^5 / 5 - cot(x) - cot(x)^3",
            vec!["sin(x) ≠ 0"],
        ),
        (
            "integrate(1/sin(2*x + 1)^8, x)",
            "-cot(2 * x + 1) / 2 + -cot(2 * x + 1)^3 / 2 + -cot(2 * x + 1)^7 / 14 + -3 * cot(2 * x + 1)^5 / 10",
            vec!["sin(2 * x + 1) ≠ 0"],
        ),
        (
            "integrate(csc(1-2*x)^8, x)",
            "cot(1 - 2 * x) / 2 + cot(1 - 2 * x)^3 / 2 + cot(1 - 2 * x)^7 / 14 + 3 * cot(1 - 2 * x)^5 / 10",
            vec!["sin(1 - 2 * x) ≠ 0"],
        ),
        (
            "integrate(cot(x)^4, x)",
            "cot(x) + x - cot(x)^3 / 3",
            vec!["sin(x) ≠ 0"],
        ),
        (
            "integrate(cos(2*x + 1)^4/sin(2*x + 1)^4, x)",
            "cot(2 * x + 1) / 2 + x - cot(2 * x + 1)^3 / 6",
            vec!["sin(2 * x + 1) ≠ 0"],
        ),
        (
            "integrate(cot(x)^6, x)",
            "-cot(x)^5 / 5 + cot(x)^3 / 3 - cot(x) - x",
            vec!["sin(x) ≠ 0"],
        ),
        (
            "integrate(cot(2*x + 1)^6, x)",
            "-cot(2 * x + 1) / 2 + -cot(2 * x + 1)^5 / 10 + cot(2 * x + 1)^3 / 6 - x",
            vec!["sin(2 * x + 1) ≠ 0"],
        ),
        (
            "integrate(cot(1-2*x)^6, x)",
            "cot(1 - 2 * x) / 2 + -cot(1 - 2 * x)^3 / 6 + cot(1 - 2 * x)^5 / 10 - x",
            vec!["sin(1 - 2 * x) ≠ 0"],
        ),
        (
            "integrate(cos(2*x + 1)^6/sin(2*x + 1)^6, x)",
            "-cot(2 * x + 1) / 2 + -cot(2 * x + 1)^5 / 10 + cot(2 * x + 1)^3 / 6 - x",
            vec!["sin(2 * x + 1) ≠ 0"],
        ),
        (
            "integrate(cot(x)^8, x)",
            "cot(x) + -cot(x)^3 / 3 + -cot(x)^7 / 7 + cot(x)^5 / 5 + x",
            vec!["sin(x) ≠ 0"],
        ),
        (
            "integrate(cot(2*x + 1)^8, x)",
            "cot(2 * x + 1) / 2 + -cot(2 * x + 1)^3 / 6 + -cot(2 * x + 1)^7 / 14 + cot(2 * x + 1)^5 / 10 + x",
            vec!["sin(2 * x + 1) ≠ 0"],
        ),
        (
            "integrate(cot(1-2*x)^8, x)",
            "-cot(1 - 2 * x) / 2 + -cot(1 - 2 * x)^5 / 10 + cot(1 - 2 * x)^3 / 6 + cot(1 - 2 * x)^7 / 14 + x",
            vec!["sin(1 - 2 * x) ≠ 0"],
        ),
        (
            "integrate(cos(2*x + 1)^8/sin(2*x + 1)^8, x)",
            "cot(2 * x + 1) / 2 + -cot(2 * x + 1)^3 / 6 + -cot(2 * x + 1)^7 / 14 + cot(2 * x + 1)^5 / 10 + x",
            vec!["sin(2 * x + 1) ≠ 0"],
        ),
        ("integrate(cot(x)^2, x)", "-cot(x) - x", vec!["sin(x) ≠ 0"]),
        (
            "integrate(cot(2*x + 1)^2, x)",
            "1/2 * (-cot(2 * x + 1) - 2 * x)",
            vec!["sin(2 * x + 1) ≠ 0"],
        ),
    ];

    for (input, expected, expected_required) in public_cases {
        let (result, required) = evaluated_integral_with_required_conditions(input);
        assert_eq!(result, expected, "input: {input}");
        assert_eq!(required, expected_required, "input: {input}");
        assert_rendered_antiderivative_verifies(input, expected);
    }

    for input in [
        "integrate(tan(x)^4, x)",
        "integrate(tan(2*x + 1)^4, x)",
        "integrate(sin(2*x + 1)^4/cos(2*x + 1)^4, x)",
        "integrate(tan(x)^6, x)",
        "integrate(tan(2*x + 1)^6, x)",
        "integrate(tan(1-2*x)^6, x)",
        "integrate(sin(2*x + 1)^6/cos(2*x + 1)^6, x)",
        "integrate(tan(x)^8, x)",
        "integrate(tan(2*x + 1)^8, x)",
        "integrate(tan(1-2*x)^8, x)",
        "integrate(sin(2*x + 1)^8/cos(2*x + 1)^8, x)",
        "integrate(sec(x)^4, x)",
        "integrate(1/cos(2*x + 1)^4, x)",
        "integrate(sec(x)^6, x)",
        "integrate(1/cos(2*x + 1)^6, x)",
        "integrate(sec(1-2*x)^6, x)",
        "integrate(sec(x)^8, x)",
        "integrate(1/cos(2*x + 1)^8, x)",
        "integrate(sec(1-2*x)^8, x)",
        "integrate(csc(x)^4, x)",
        "integrate(1/sin(2*x + 1)^4, x)",
        "integrate(csc(x)^6, x)",
        "integrate(1/sin(2*x + 1)^6, x)",
        "integrate(csc(1-2*x)^6, x)",
        "integrate(csc(x)^8, x)",
        "integrate(1/sin(2*x + 1)^8, x)",
        "integrate(csc(1-2*x)^8, x)",
        "integrate(cot(x)^4, x)",
        "integrate(cos(2*x + 1)^4/sin(2*x + 1)^4, x)",
        "integrate(cot(x)^6, x)",
        "integrate(cot(2*x + 1)^6, x)",
        "integrate(cot(1-2*x)^6, x)",
        "integrate(cos(2*x + 1)^6/sin(2*x + 1)^6, x)",
        "integrate(cot(x)^8, x)",
        "integrate(cot(2*x + 1)^8, x)",
        "integrate(cot(1-2*x)^8, x)",
        "integrate(cos(2*x + 1)^8/sin(2*x + 1)^8, x)",
    ] {
        assert_eq!(
            assert_antiderivative_verifies(input),
            AntiderivativeVerificationRoute::PublicResidual,
            "{input} should verify through the bounded public residual route"
        );
    }
}

#[test]
fn integrate_contract_affine_sine_cosine_product() {
    let public_cases = [
        ("integrate(sin(x)*cos(x), x)", "1/2 * sin(x)^2"),
        (
            "integrate(3*sin(2*x + 1)*cos(2*x + 1), x)",
            "3/4 * sin(2 * x + 1)^2",
        ),
        (
            "integrate(3*cos(2*x + 1)*sin(2*x + 1), x)",
            "3/4 * sin(2 * x + 1)^2",
        ),
        (
            "integrate(-3*sin(2*x + 1)*cos(2*x + 1), x)",
            "-3/4 * sin(2 * x + 1)^2",
        ),
    ];

    for (input, expected) in public_cases {
        let (result, required) = evaluated_integral_with_required_conditions(input);
        assert_eq!(result, expected, "input: {input}");
        assert!(
            required.is_empty(),
            "input: {input}, required: {required:?}"
        );
        assert_rendered_antiderivative_verifies(input, expected);
    }
}

#[test]
fn integrate_contract_affine_trig_power_times_derivative_product() {
    let public_cases = [
        ("integrate(sin(x)^2*cos(x), x)", "1/3 * sin(x)^3"),
        (
            "integrate(2*cos(2*x + 1)*sin(2*x + 1)^2, x)",
            "1/3 * sin(2 * x + 1)^3",
        ),
        (
            "integrate(2*sin(2*x + 1)^2*cos(2*x + 1), x)",
            "1/3 * sin(2 * x + 1)^3",
        ),
        (
            "integrate(-2*cos(2*x + 1)*sin(2*x + 1)^2, x)",
            "-1/3 * sin(2 * x + 1)^3",
        ),
        ("integrate(sin(x)*cos(x)^2, x)", "-1/3 * cos(x)^3"),
        ("integrate(-sin(x)*cos(x)^2, x)", "1/3 * cos(x)^3"),
    ];

    for (input, expected) in public_cases {
        let (result, required) = evaluated_integral_with_required_conditions(input);
        assert_eq!(result, expected, "input: {input}");
        assert!(
            required.is_empty(),
            "input: {input}, required: {required:?}"
        );
        assert_rendered_antiderivative_verifies(input, expected);
    }
}

#[test]
fn integrate_contract_affine_trig_ratio_power_reciprocal_square_product() {
    let public_cases = [
        (
            "integrate(sec(x)^2*tan(x), x)",
            "tan(x)^2 / 2",
            vec!["cos(x) ≠ 0"],
        ),
        (
            "integrate(2*sec(2*x + 1)^2*tan(2*x + 1), x)",
            "tan(2 * x + 1)^2 / 2",
            vec!["cos(2 * x + 1) ≠ 0"],
        ),
        (
            "integrate(sin(x)/cos(x)^3, x)",
            "tan(x)^2 / 2",
            vec!["cos(x) ≠ 0"],
        ),
        (
            "integrate(tan(x)^2/cos(x)^2, x)",
            "tan(x)^3 / 3",
            vec!["cos(x) ≠ 0"],
        ),
        (
            "integrate(2*tan(2*x + 1)^2/cos(2*x + 1)^2, x)",
            "tan(2 * x + 1)^3 / 3",
            vec!["cos(2 * x + 1) ≠ 0"],
        ),
        (
            "integrate(sin(x)^2/cos(x)^4, x)",
            "tan(x)^3 / 3",
            vec!["cos(x) ≠ 0"],
        ),
        (
            "integrate(csc(x)^2*cot(x), x)",
            "-cot(x)^2 / 2",
            vec!["sin(x) ≠ 0"],
        ),
        (
            "integrate(2*csc(2*x + 1)^2*cot(2*x + 1), x)",
            "-cot(2 * x + 1)^2 / 2",
            vec!["sin(2 * x + 1) ≠ 0"],
        ),
        (
            "integrate(cos(x)/sin(x)^3, x)",
            "-cot(x)^2 / 2",
            vec!["sin(x) ≠ 0"],
        ),
        (
            "integrate(cot(x)^2/sin(x)^2, x)",
            "-cot(x)^3 / 3",
            vec!["sin(x) ≠ 0"],
        ),
        (
            "integrate(2*cot(2*x + 1)^2/sin(2*x + 1)^2, x)",
            "-cot(2 * x + 1)^3 / 3",
            vec!["sin(2 * x + 1) ≠ 0"],
        ),
        (
            "integrate(cos(x)^2/sin(x)^4, x)",
            "-cot(x)^3 / 3",
            vec!["sin(x) ≠ 0"],
        ),
    ];

    for (input, expected, expected_required) in public_cases {
        let (result, required) = evaluated_integral_with_required_conditions(input);
        assert_eq!(result, expected, "input: {input}");
        assert_eq!(
            required, expected_required,
            "input: {input}, required: {required:?}"
        );
        assert_rendered_antiderivative_verifies(input, expected);
    }
}

#[test]
fn integrate_contract_reciprocal_trig_power_verification_avoids_depth_overflow() {
    for input in [
        "diff(tan(x)^2/2, x) - sec(x)^2*tan(x)",
        "diff(-cot(x)^2/2, x) - csc(x)^2*cot(x)",
        "diff(tan(2*x+1)^2/2, x) - 2*sec(2*x+1)^2*tan(2*x+1)",
        "diff(-cot(2*x+1)^2/2, x) - 2*csc(2*x+1)^2*cot(2*x+1)",
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr(input);
        assert_eq!(wire["result"], "0", "unexpected residual for {input}");
        assert!(
            !stderr.contains("depth_overflow"),
            "reciprocal trig power verification should not emit depth_overflow for {input}\nstderr:\n{stderr}"
        );
    }
}

#[test]
fn integrate_contract_affine_trig_cube_power_reduction() {
    assert_eq!(
        simplified_integral("integrate(sin(x)^3, x)"),
        "1/3 * (cos(x)^3 - 3 * cos(x))"
    );
    assert_eq!(
        simplified_integral("integrate(cos(x)^3, x)"),
        "1/3 * (3 * sin(x) - sin(x)^3)"
    );
    assert_eq!(
        simplified_integral("integrate(sin(2*x + 1)^3, x)"),
        "1/6 * (cos(2 * x + 1)^3 - 3 * cos(2 * x + 1))"
    );
    assert_eq!(
        simplified_integral("integrate(cos(2*x + 1)^3, x)"),
        "1/6 * (3 * sin(2 * x + 1) - sin(2 * x + 1)^3)"
    );

    let public_cases = [
        ("integrate(sin(x)^3, x)", "1/3 * cos(x)^3 - cos(x)"),
        ("integrate(cos(x)^3, x)", "sin(x) - 1/3 * sin(x)^3"),
        (
            "integrate(sin(2*x + 1)^3, x)",
            "1/6 * cos(2 * x + 1)^3 - 1/2 * cos(2 * x + 1)",
        ),
        (
            "integrate(cos(2*x + 1)^3, x)",
            "1/2 * sin(2 * x + 1) - 1/6 * sin(2 * x + 1)^3",
        ),
    ];

    for (input, expected) in public_cases {
        let (result, required) = evaluated_integral_with_required_conditions(input);
        assert_eq!(result, expected, "input: {input}");
        assert!(
            required.is_empty(),
            "input: {input}, required: {required:?}"
        );
        assert_antiderivative_verifies(input);
    }
}

#[test]
fn integrate_contract_affine_trig_fifth_power_reduction() {
    let cases = [
        (
            "integrate(sin(x)^5, x)",
            "sin(x)^5",
            "2/3 * cos(x)^3 - cos(x) - 1/5 * cos(x)^5",
        ),
        (
            "integrate(cos(x)^5, x)",
            "cos(x)^5",
            "sin(x) + 1/5 * sin(x)^5 - 2/3 * sin(x)^3",
        ),
        (
            "integrate(sin(2*x + 1)^5, x)",
            "sin(2*x + 1)^5",
            "1/3 * cos(2 * x + 1)^3 - 1/2 * cos(2 * x + 1) - 1/10 * cos(2 * x + 1)^5",
        ),
        (
            "integrate(cos(2*x + 1)^5, x)",
            "cos(2*x + 1)^5",
            "1/10 * sin(2 * x + 1)^5 + 1/2 * sin(2 * x + 1) - 1/3 * sin(2 * x + 1)^3",
        ),
    ];

    for (input, integrand, expected) in cases {
        let (antiderivative, required) = evaluated_integral_with_required_conditions(input);
        assert!(
            required.is_empty(),
            "trig fifth primitive should not add domain conditions for {input}: {required:?}"
        );
        assert_eq!(antiderivative, expected, "{input}");
        assert!(
            !antiderivative.contains("10/3"),
            "post-calculus presentation should distribute nested fifth-power primitive coefficients: {antiderivative}"
        );
        assert_antiderivative_verifies(input);
        assert_rendered_antiderivative_verifies(input, &antiderivative);

        let residual = format!("diff({input}, x) - {integrand}");
        let (wire, _stderr) = cli_eval_json_with_stderr(&residual);
        assert_eq!(wire["result"], "0", "{residual}");
    }
}

#[test]
fn integrate_contract_affine_trig_seventh_power_reduction() {
    let cases = [
        (
            "integrate(sin(x)^7, x)",
            "sin(x)^7",
            "1/7 * (cos(x)^7 + 7 * cos(x)^3 - 7 * cos(x) - 21/5 * cos(x)^5)",
        ),
        (
            "integrate(cos(x)^7, x)",
            "cos(x)^7",
            "1/7 * (21/5 * sin(x)^5 + 7 * sin(x) - sin(x)^7 - 7 * sin(x)^3)",
        ),
        (
            "integrate(sin(2*x + 1)^7, x)",
            "sin(2*x + 1)^7",
            "(cos(2 * x + 1)^3 + 1/7 * cos(2 * x + 1)^7 - cos(2 * x + 1) - 3/5 * cos(2 * x + 1)^5) / 2",
        ),
        (
            "integrate(cos(2*x + 1)^7, x)",
            "cos(2*x + 1)^7",
            "(sin(2 * x + 1) + 3/5 * sin(2 * x + 1)^5 - sin(2 * x + 1)^3 - 1/7 * sin(2 * x + 1)^7) / 2",
        ),
        (
            "integrate(sin(1 - 2*x)^7, x)",
            "sin(1 - 2*x)^7",
            "-(cos(1 - 2 * x)^3 + 1/7 * cos(1 - 2 * x)^7 - cos(1 - 2 * x) - 3/5 * cos(1 - 2 * x)^5) / 2",
        ),
        (
            "integrate(cos(1 - 2*x)^7, x)",
            "cos(1 - 2*x)^7",
            "-(sin(1 - 2 * x) + 3/5 * sin(1 - 2 * x)^5 - sin(1 - 2 * x)^3 - 1/7 * sin(1 - 2 * x)^7) / 2",
        ),
    ];

    for (input, integrand, expected) in cases {
        let (antiderivative, required) = evaluated_integral_with_required_conditions(input);
        assert!(
            required.is_empty(),
            "trig seventh primitive should not add domain conditions for {input}: {required:?}"
        );
        assert_eq!(antiderivative, expected, "{input}");
        assert_antiderivative_verifies(input);
        assert_rendered_antiderivative_verifies(input, &antiderivative);

        let residual = format!("diff({input}, x) - {integrand}");
        let (wire, stderr) = cli_eval_json_with_stderr(&residual);
        assert!(
            stderr.is_empty(),
            "trig seventh residual should stay quiet for {input}: {stderr}"
        );
        assert_eq!(wire["result"], "0", "{residual}");
        assert_eq!(
            wire["required_display"],
            serde_json::json!([]),
            "{residual}"
        );
    }
}

#[test]
fn integrate_contract_hyperbolic_odd_power_reduction() {
    let cases = [
        ("integrate(sinh(x)^3, x)", "1/3 * cosh(x)^3 - cosh(x)"),
        ("integrate(cosh(x)^3, x)", "sinh(x) + 1/3 * sinh(x)^3"),
        (
            "integrate(sinh(x)^5, x)",
            "1/5 * (cosh(x)^5 + 5 * cosh(x) - 10/3 * cosh(x)^3)",
        ),
        (
            "integrate(cosh(x)^5, x)",
            "1/5 * (sinh(x)^5 + 10/3 * sinh(x)^3 + 5 * sinh(x))",
        ),
        (
            "integrate(sinh(x)^7, x)",
            "-cosh(x) + cosh(x)^3 + 1/7 * cosh(x)^7 - 3/5 * cosh(x)^5",
        ),
        (
            "integrate(cosh(x)^7, x)",
            "sinh(x) + sinh(x)^3 + 1/7 * sinh(x)^7 + 3/5 * sinh(x)^5",
        ),
    ];

    for (input, expected) in cases {
        let (antiderivative, required) = evaluated_integral_with_required_conditions(input);
        assert!(
            required.is_empty(),
            "hyperbolic odd-power primitive should not add domain conditions for {input}: {required:?}"
        );
        assert_eq!(antiderivative, expected, "{input}");
        assert!(
            !antiderivative.contains("integrate("),
            "expected a closed-form primitive for {input}, got {antiderivative}"
        );
    }

    for input in ["integrate(sinh(x)^3, x)", "integrate(cosh(x)^7, x)"] {
        assert_antiderivative_equiv_verifies(input);
    }

    for input in ["integrate(sinh(x)^7, x)", "integrate(cosh(x)^7, x)"] {
        assert_eq!(
            assert_antiderivative_verifies(input),
            AntiderivativeVerificationRoute::PublicResidual,
            "{input} should verify through the bounded public residual route"
        );
    }
}

#[test]
fn integrate_contract_affine_hyperbolic_fifth_explicit_primitives_verify_publicly() {
    let residuals = [
        "diff(1/2*(1/5*cosh(2*x+1)^5 - 2/3*cosh(2*x+1)^3 + cosh(2*x+1)), x) - sinh(2*x+1)^5",
        "diff(1/2*(sinh(2*x+1) + 2/3*sinh(2*x+1)^3 + 1/5*sinh(2*x+1)^5), x) - cosh(2*x+1)^5",
        "diff(-1/2*(1/5*cosh(1-2*x)^5 - 2/3*cosh(1-2*x)^3 + cosh(1-2*x)), x) - sinh(1-2*x)^5",
        "diff(-1/2*(sinh(1-2*x) + 2/3*sinh(1-2*x)^3 + 1/5*sinh(1-2*x)^5), x) - cosh(1-2*x)^5",
    ];

    for input in residuals {
        let (wire, stderr) = cli_eval_json_with_stderr(input);
        assert!(stderr.is_empty(), "unexpected stderr for {input}: {stderr}");
        assert_eq!(wire["result"], "0", "{input}");
        assert_eq!(wire["required_display"], serde_json::json!([]), "{input}");
        assert!(
            wire["warnings"]
                .as_array()
                .is_some_and(|warnings| warnings.is_empty()),
            "affine fifth primitive residual should not warn for {input}: {wire:#}"
        );
    }
}

#[test]
fn integrate_contract_affine_hyperbolic_fifth_power_reduction() {
    let public_cases = [
        (
            "integrate(sinh(2*x + 1)^5, x)",
            "1/2 * (cosh(2 * x + 1) + 1/5 * cosh(2 * x + 1)^5 - 2/3 * cosh(2 * x + 1)^3)",
        ),
        (
            "integrate(cosh(2*x + 1)^5, x)",
            "1/2 * (sinh(2 * x + 1) + 1/5 * sinh(2 * x + 1)^5 + 2/3 * sinh(2 * x + 1)^3)",
        ),
        (
            "integrate(sinh(1 - 2*x)^5, x)",
            "-1/2 * (cosh(1 - 2 * x) + 1/5 * cosh(1 - 2 * x)^5 - 2/3 * cosh(1 - 2 * x)^3)",
        ),
        (
            "integrate(cosh(1 - 2*x)^5, x)",
            "-1/2 * (sinh(1 - 2 * x) + 1/5 * sinh(1 - 2 * x)^5 + 2/3 * sinh(1 - 2 * x)^3)",
        ),
    ];

    for (input, expected) in public_cases {
        let (result, required) = evaluated_integral_with_required_conditions(input);
        assert_eq!(result, expected, "input: {input}");
        assert!(
            required.is_empty(),
            "input: {input}, required: {required:?}"
        );
        assert_eq!(
            assert_antiderivative_verifies(input),
            AntiderivativeVerificationRoute::PublicResidual,
            "{input} should verify through the bounded public residual route"
        );
    }
}

#[test]
fn integrate_contract_affine_hyperbolic_seventh_explicit_primitives_verify_publicly() {
    let residuals = [
        "diff(1/2*(1/7*cosh(2*x+1)^7 - 3/5*cosh(2*x+1)^5 + cosh(2*x+1)^3 - cosh(2*x+1)), x) - sinh(2*x+1)^7",
        "diff(1/2*(sinh(2*x+1) + sinh(2*x+1)^3 + 3/5*sinh(2*x+1)^5 + 1/7*sinh(2*x+1)^7), x) - cosh(2*x+1)^7",
        "diff(-1/2*(1/7*cosh(1-2*x)^7 - 3/5*cosh(1-2*x)^5 + cosh(1-2*x)^3 - cosh(1-2*x)), x) - sinh(1-2*x)^7",
        "diff(-1/2*(sinh(1-2*x) + sinh(1-2*x)^3 + 3/5*sinh(1-2*x)^5 + 1/7*sinh(1-2*x)^7), x) - cosh(1-2*x)^7",
    ];

    for input in residuals {
        let (wire, stderr) = cli_eval_json_with_stderr(input);
        assert!(stderr.is_empty(), "unexpected stderr for {input}: {stderr}");
        assert_eq!(wire["result"], "0", "{input}");
        assert_eq!(wire["required_display"], serde_json::json!([]), "{input}");
        assert!(
            wire["warnings"]
                .as_array()
                .is_some_and(|warnings| warnings.is_empty()),
            "affine seventh primitive residual should not warn for {input}: {wire:#}"
        );
    }
}

#[test]
fn integrate_contract_affine_hyperbolic_seventh_power_reduction() {
    let public_cases = [
        (
            "integrate(sinh(2*x + 1)^7, x)",
            "1/2 * (-cosh(2 * x + 1) + cosh(2 * x + 1)^3 + 1/7 * cosh(2 * x + 1)^7 - 3/5 * cosh(2 * x + 1)^5)",
        ),
        (
            "integrate(cosh(2*x + 1)^7, x)",
            "1/2 * (sinh(2 * x + 1) + sinh(2 * x + 1)^3 + 1/7 * sinh(2 * x + 1)^7 + 3/5 * sinh(2 * x + 1)^5)",
        ),
        (
            "integrate(sinh(1 - 2*x)^7, x)",
            "-1/2 * (-cosh(1 - 2 * x) + cosh(1 - 2 * x)^3 + 1/7 * cosh(1 - 2 * x)^7 - 3/5 * cosh(1 - 2 * x)^5)",
        ),
        (
            "integrate(cosh(1 - 2*x)^7, x)",
            "-1/2 * (sinh(1 - 2 * x) + sinh(1 - 2 * x)^3 + 1/7 * sinh(1 - 2 * x)^7 + 3/5 * sinh(1 - 2 * x)^5)",
        ),
    ];

    for (input, expected) in public_cases {
        let (result, required) = evaluated_integral_with_required_conditions(input);
        assert_eq!(result, expected, "input: {input}");
        assert!(
            required.is_empty(),
            "input: {input}, required: {required:?}"
        );
        assert_eq!(
            assert_antiderivative_verifies(input),
            AntiderivativeVerificationRoute::PublicResidual,
            "{input} should verify through the bounded public residual route"
        );
    }
}

#[test]
fn integrate_contract_affine_trig_fourth_power_reduction() {
    let cases = [
        (
            "integrate(sin(x)^4, x)",
            "sin(x)^4",
            "1/32 * sin(4 * x) + 3/8 * x - 1/4 * sin(2 * x)",
        ),
        (
            "integrate(cos(x)^4, x)",
            "cos(x)^4",
            "1/32 * sin(4 * x) + 1/4 * sin(2 * x) + 3/8 * x",
        ),
        (
            "integrate(sin(2*x + 1)^4, x)",
            "sin(2*x + 1)^4",
            "1/64 * sin(4 * (2 * x + 1)) + 3/8 * x - 1/8 * sin(2 * (2 * x + 1))",
        ),
        (
            "integrate(cos(2*x + 1)^4, x)",
            "cos(2*x + 1)^4",
            "1/64 * sin(4 * (2 * x + 1)) + 1/8 * sin(2 * (2 * x + 1)) + 3/8 * x",
        ),
    ];

    for (input, integrand, expected) in cases {
        let (antiderivative, required) = evaluated_integral_with_required_conditions(input);
        assert!(
            required.is_empty(),
            "trig fourth primitive should not add domain conditions for {input}: {required:?}"
        );
        assert_eq!(antiderivative, expected, "{input}");
        assert_rendered_antiderivative_verifies(input, &antiderivative);

        let residual = format!("diff({input}, x) - {integrand}");
        let (wire, stderr) = cli_eval_json_with_stderr(&residual);
        assert_eq!(wire["result"], "0", "{residual}");
        assert_eq!(wire["required_display"], serde_json::json!([]));
        assert!(
            !stderr.contains("depth_overflow"),
            "trig fourth nested residual should not emit depth_overflow for {residual}\nstderr:\n{stderr}"
        );
    }
}

#[test]
fn integrate_contract_affine_trig_sixth_power_reduction() {
    let cases = [
        (
            "integrate(sin(x)^6, x)",
            "sin(x)^6",
            "3/64 * sin(4 * x) + 5/16 * x - 15/64 * sin(2 * x) - 1/192 * sin(6 * x)",
        ),
        (
            "integrate(cos(x)^6, x)",
            "cos(x)^6",
            "1/192 * sin(6 * x) + 3/64 * sin(4 * x) + 15/64 * sin(2 * x) + 5/16 * x",
        ),
        (
            "integrate(sin(2*x + 1)^6, x)",
            "sin(2*x + 1)^6",
            "3/128 * sin(4 * (2 * x + 1)) + 5/16 * x - 15/128 * sin(2 * (2 * x + 1)) - 1/384 * sin(6 * (2 * x + 1))",
        ),
        (
            "integrate(cos(2*x + 1)^6, x)",
            "cos(2*x + 1)^6",
            "1/384 * sin(6 * (2 * x + 1)) + 3/128 * sin(4 * (2 * x + 1)) + 15/128 * sin(2 * (2 * x + 1)) + 5/16 * x",
        ),
    ];

    for (input, integrand, expected) in cases {
        let (antiderivative, required) = evaluated_integral_with_required_conditions(input);
        assert!(
            required.is_empty(),
            "trig sixth primitive should not add domain conditions for {input}: {required:?}"
        );
        assert_eq!(antiderivative, expected, "{input}");
        assert_rendered_antiderivative_verifies(input, &antiderivative);

        let residual = format!("diff({input}, x) - {integrand}");
        let (wire, stderr) = cli_eval_json_with_stderr(&residual);
        assert_eq!(wire["result"], "0", "{residual}");
        assert_eq!(wire["required_display"], serde_json::json!([]));
        assert!(
            !stderr.contains("depth_overflow"),
            "trig sixth nested residual should not emit depth_overflow for {residual}\nstderr:\n{stderr}"
        );
    }
}

#[test]
fn integrate_contract_affine_trig_eighth_power_reduction() {
    let cases = [
        (
            "integrate(sin(x)^8, x)",
            "sin(x)^8",
            "1/1024 * sin(8 * x) + 7/128 * sin(4 * x) + 35/128 * x - 7/32 * sin(2 * x) - 1/96 * sin(6 * x)",
        ),
        (
            "integrate(cos(x)^8, x)",
            "cos(x)^8",
            "1/1024 * sin(8 * x) + 1/96 * sin(6 * x) + 7/128 * sin(4 * x) + 7/32 * sin(2 * x) + 35/128 * x",
        ),
        (
            "integrate(sin(2*x + 1)^8, x)",
            "sin(2*x + 1)^8",
            "1/2048 * sin(8 * (2 * x + 1)) + 7/256 * sin(4 * (2 * x + 1)) + 35/128 * x - 7/64 * sin(2 * (2 * x + 1)) - 1/192 * sin(6 * (2 * x + 1))",
        ),
        (
            "integrate(cos(2*x + 1)^8, x)",
            "cos(2*x + 1)^8",
            "1/2048 * sin(8 * (2 * x + 1)) + 1/192 * sin(6 * (2 * x + 1)) + 7/256 * sin(4 * (2 * x + 1)) + 7/64 * sin(2 * (2 * x + 1)) + 35/128 * x",
        ),
    ];

    for (input, integrand, expected) in cases {
        let (antiderivative, required) = evaluated_integral_with_required_conditions(input);
        assert!(
            required.is_empty(),
            "trig eighth primitive should not add domain conditions for {input}: {required:?}"
        );
        assert_eq!(antiderivative, expected, "{input}");
        assert_rendered_antiderivative_verifies(input, &antiderivative);

        let residual = format!("diff({input}, x) - {integrand}");
        let (wire, stderr) = cli_eval_json_with_stderr(&residual);
        assert_eq!(wire["result"], "0", "{residual}");
        assert_eq!(wire["required_display"], serde_json::json!([]));
        assert!(
            !stderr.contains("depth_overflow"),
            "trig eighth nested residual should not emit depth_overflow for {residual}\nstderr:\n{stderr}"
        );
    }
}

#[test]
fn integrate_contract_explicit_trig_fourth_power_antiderivative_residual_verifies() {
    for residual in [
        "diff(3*x/8 - sin(2*x)/4 + sin(4*x)/32, x) - sin(x)^4",
        "diff(3*x/8 + sin(2*x)/4 + sin(4*x)/32, x) - cos(x)^4",
        "diff(3*x/8 - sin(2*(2*x+1))/8 + sin(4*(2*x+1))/64, x) - sin(2*x+1)^4",
        "diff(3*x/8 + sin(2*(2*x+1))/8 + sin(4*(2*x+1))/64, x) - cos(2*x+1)^4",
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr(residual);
        assert_eq!(wire["result"], "0", "{residual}");
        assert_eq!(wire["required_display"], serde_json::json!([]));
        assert!(
            !stderr.contains("depth_overflow"),
            "fourth-power trig residual should not emit depth_overflow for {residual}\nstderr:\n{stderr}"
        );
    }
}

#[test]
fn integrate_contract_explicit_trig_sixth_power_antiderivative_residual_verifies() {
    for residual in [
        "diff(5*x/16 - 15*sin(2*x)/64 + 3*sin(4*x)/64 - sin(6*x)/192, x) - sin(x)^6",
        "diff(5*x/16 + 15*sin(2*x)/64 + 3*sin(4*x)/64 + sin(6*x)/192, x) - cos(x)^6",
        "diff(5*x/16 - 15*sin(2*(2*x+1))/128 + 3*sin(4*(2*x+1))/128 - sin(6*(2*x+1))/384, x) - sin(2*x+1)^6",
        "diff(5*x/16 + 15*sin(2*(2*x+1))/128 + 3*sin(4*(2*x+1))/128 + sin(6*(2*x+1))/384, x) - cos(2*x+1)^6",
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr(residual);
        assert_eq!(wire["result"], "0", "{residual}");
        assert_eq!(wire["required_display"], serde_json::json!([]));
        assert!(
            !stderr.contains("depth_overflow"),
            "sixth-power trig residual should not emit depth_overflow for {residual}\nstderr:\n{stderr}"
        );
    }
}

#[test]
fn integrate_contract_explicit_trig_eighth_power_antiderivative_residual_verifies() {
    for residual in [
        "diff(35*x/128 - 7*sin(2*x)/32 + 7*sin(4*x)/128 - sin(6*x)/96 + sin(8*x)/1024, x) - sin(x)^8",
        "diff(35*x/128 + 7*sin(2*x)/32 + 7*sin(4*x)/128 + sin(6*x)/96 + sin(8*x)/1024, x) - cos(x)^8",
        "diff(35*x/128 - 7*sin(2*(2*x+1))/64 + 7*sin(4*(2*x+1))/256 - sin(6*(2*x+1))/192 + sin(8*(2*x+1))/2048, x) - sin(2*x+1)^8",
        "diff(35*x/128 + 7*sin(2*(2*x+1))/64 + 7*sin(4*(2*x+1))/256 + sin(6*(2*x+1))/192 + sin(8*(2*x+1))/2048, x) - cos(2*x+1)^8",
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr(residual);
        assert_eq!(wire["result"], "0", "{residual}");
        assert_eq!(wire["required_display"], serde_json::json!([]));
        assert!(
            !stderr.contains("depth_overflow"),
            "eighth-power trig residual should not emit depth_overflow for {residual}\nstderr:\n{stderr}"
        );
    }
}

#[test]
fn integrate_contract_explicit_negated_sine_uses_linearity() {
    assert_eq!(simplified_integral("integrate(-(sin(x)), x)"), "cos(x)");
}

#[test]
fn integrate_contract_linear_exp_substitution() {
    assert_eq!(
        simplified_integral("integrate(exp(3*x + 1), x)"),
        "1/3 * e^(3 * x + 1)"
    );
}

#[test]
fn integrate_contract_linear_times_exp_linear_by_parts() {
    let (result, required) = evaluated_integral_with_required_conditions("integrate(x*exp(x), x)");
    assert_eq!(result, "(x - 1) * e^x");
    assert!(
        required.is_empty(),
        "unexpected required_conditions: {required:?}"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate((2*x+3)*exp(2*x+1), x)");
    assert_eq!(result, "(x + 1) * e^(2 * x + 1)");
    assert!(
        required.is_empty(),
        "unexpected required_conditions: {required:?}"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate((x+1)*exp((3*x+2)/2), x)");
    assert_eq!(result, "(2/3 * x + 2/9) * e^((3 * x + 2) / 2)");
    assert!(
        required.is_empty(),
        "unexpected required_conditions: {required:?}"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate((x+1)*exp((2-3*x)/2), x)");
    assert_eq!(result, "(-2/3 * x - 10/9) * e^((2 - 3 * x) / 2)");
    assert!(
        required.is_empty(),
        "unexpected required_conditions: {required:?}"
    );
}

#[test]
fn integrate_contract_linear_times_trig_linear_by_parts() {
    let (result, required) = evaluated_integral_with_required_conditions("integrate(x*sin(x), x)");
    assert_eq!(result, "sin(x) - x * cos(x)");
    assert!(
        required.is_empty(),
        "unexpected required_conditions: {required:?}"
    );

    let (result, required) = evaluated_integral_with_required_conditions("integrate(x*cos(x), x)");
    assert_eq!(result, "cos(x) + x * sin(x)");
    assert!(
        required.is_empty(),
        "unexpected required_conditions: {required:?}"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate((2*x+3)*sin(2*x+1), x)");
    assert_eq!(
        result,
        "1/2 * sin(2 * x + 1) - (cos(2 * x + 1) * (2 * x + 3))/2"
    );
    assert!(
        required.is_empty(),
        "unexpected required_conditions: {required:?}"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate((2*x+3)*cos(2*x+1), x)");
    assert_eq!(
        result,
        "1/2 * cos(2 * x + 1) + (sin(2 * x + 1) * (2 * x + 3))/2"
    );
    assert!(
        required.is_empty(),
        "unexpected required_conditions: {required:?}"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate((x+1)*sin((3*x+2)/2), x)");
    assert_eq!(
        result,
        "4/9 * sin((3 * x + 2) / 2) - 2/3 * (x + 1) * cos((3 * x + 2) / 2)"
    );
    assert!(
        required.is_empty(),
        "unexpected required_conditions: {required:?}"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate((x+1)*cos((3*x+2)/2), x)");
    assert_eq!(
        result,
        "4/9 * cos((3 * x + 2) / 2) + 2/3 * (x + 1) * sin((3 * x + 2) / 2)"
    );
    assert!(
        required.is_empty(),
        "unexpected required_conditions: {required:?}"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate((x+1)*sin((2-3*x)/2), x)");
    assert_eq!(
        result,
        "4/9 * sin((2 - 3 * x) / 2) + 2/3 * (x + 1) * cos((2 - 3 * x) / 2)"
    );
    assert!(
        required.is_empty(),
        "unexpected required_conditions: {required:?}"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate((x+1)*cos((2-3*x)/2), x)");
    assert_eq!(
        result,
        "4/9 * cos((2 - 3 * x) / 2) - 2/3 * (x + 1) * sin((2 - 3 * x) / 2)"
    );
    assert!(
        required.is_empty(),
        "unexpected required_conditions: {required:?}"
    );
}

#[test]
fn integrate_contract_linear_times_hyperbolic_linear_by_parts() {
    let (result, required) = evaluated_integral_with_required_conditions("integrate(x*sinh(x), x)");
    assert_eq!(result, "x * cosh(x) - sinh(x)");
    assert!(
        required.is_empty(),
        "unexpected required_conditions: {required:?}"
    );

    let (result, required) = evaluated_integral_with_required_conditions("integrate(x*cosh(x), x)");
    assert_eq!(result, "x * sinh(x) - cosh(x)");
    assert!(
        required.is_empty(),
        "unexpected required_conditions: {required:?}"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate((2*x+3)*sinh(2*x+1), x)");
    assert_eq!(
        result,
        "(cosh(2 * x + 1) * (2 * x + 3))/2 - 1/2 * sinh(2 * x + 1)"
    );
    assert!(
        required.is_empty(),
        "unexpected required_conditions: {required:?}"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate((2*x+3)*cosh(2*x+1), x)");
    assert_eq!(
        result,
        "(sinh(2 * x + 1) * (2 * x + 3))/2 - 1/2 * cosh(2 * x + 1)"
    );
    assert!(
        required.is_empty(),
        "unexpected required_conditions: {required:?}"
    );

    for (input, expected) in [
        (
            "integrate((2*x+3)*sinh(1-2*x), x)",
            "1/2·(-sinh(1 - 2·x) - cosh(1 - 2·x)·(2·x + 3))",
        ),
        (
            "integrate((2*x+3)*cosh(1-2*x), x)",
            "1/2·(-cosh(1 - 2·x) - sinh(1 - 2·x)·(2·x + 3))",
        ),
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr(input);
        assert!(stderr.is_empty(), "unexpected stderr for {input}: {stderr}");
        assert_eq!(wire["result"], expected, "{input}");
        assert_eq!(wire["required_display"], serde_json::json!([]), "{input}");
    }

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate((x+1)*sinh(2*x+1), x)");
    assert_eq!(
        result,
        "(cosh(2 * x + 1) * (x + 1))/2 - 1/4 * sinh(2 * x + 1)"
    );
    assert!(
        required.is_empty(),
        "unexpected required_conditions: {required:?}"
    );

    let residual = "diff(integrate((x+1)*sinh(2*x+1), x), x) - (x+1)*sinh(2*x+1)";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert!(
        residual_required.is_empty(),
        "linear hyperbolic residual should not add conditions: {residual_required:?}"
    );

    for residual in [
        "diff(integrate((2*x+3)*sinh(1-2*x), x), x) - (2*x+3)*sinh(1-2*x)",
        "diff(integrate((2*x+3)*cosh(1-2*x), x), x) - (2*x+3)*cosh(1-2*x)",
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr(residual);
        assert!(
            stderr.is_empty(),
            "negative-affine hyperbolic residual should not emit stderr: {stderr}"
        );
        assert_eq!(wire["result"], "0", "{residual}");
        assert_eq!(
            wire["required_display"],
            serde_json::json!([]),
            "{residual}"
        );
    }
}

#[test]
fn integrate_contract_polynomial_times_hyperbolic_linear_by_parts_verifies() {
    for (input, expected_result, residual) in [
        (
            "integrate(x^2*sinh(x), x)",
            "(x^2 + 2)·cosh(x) - 2·x·sinh(x)",
            "diff(integrate(x^2*sinh(x), x), x) - x^2*sinh(x)",
        ),
        (
            "integrate(x^2*cosh(x), x)",
            "(x^2 + 2)·sinh(x) - 2·x·cosh(x)",
            "diff(integrate(x^2*cosh(x), x), x) - x^2*cosh(x)",
        ),
        (
            "integrate((x^2+x)*sinh(2*x+1), x)",
            "(1/2·x^2 + 1/2·x + 1/4)·cosh(2·x + 1) - (1/2·x + 1/4)·sinh(2·x + 1)",
            "diff(integrate((x^2+x)*sinh(2*x+1), x), x) - (x^2+x)*sinh(2*x+1)",
        ),
        (
            "integrate((x^3+x)*sinh(2*x+1), x)",
            "(1/2·x^3 + 5/4·x)·cosh(2·x + 1) - (3/4·x^2 + 5/8)·sinh(2·x + 1)",
            "diff(integrate((x^3+x)*sinh(2*x+1), x), x) - (x^3+x)*sinh(2*x+1)",
        ),
        (
            "integrate((x^3+x)*cosh(2*x+1), x)",
            "(1/2·x^3 + 5/4·x)·sinh(2·x + 1) - (3/4·x^2 + 5/8)·cosh(2·x + 1)",
            "diff(integrate((x^3+x)*cosh(2*x+1), x), x) - (x^3+x)*cosh(2*x+1)",
        ),
        (
            "integrate(x^2*sinh(2*x+1)+x*sinh(2*x+1), x)",
            "(1/2·x^2 + 1/2·x + 1/4)·cosh(2·x + 1) - (1/2·x + 1/4)·sinh(2·x + 1)",
            "diff(integrate(x^2*sinh(2*x+1)+x*sinh(2*x+1), x), x) - (x^2*sinh(2*x+1)+x*sinh(2*x+1))",
        ),
        (
            "integrate(x^2*cosh(2*x+1)+x*cosh(2*x+1), x)",
            "(1/2·x^2 + 1/2·x + 1/4)·sinh(2·x + 1) - (1/2·x + 1/4)·cosh(2·x + 1)",
            "diff(integrate(x^2*cosh(2*x+1)+x*cosh(2*x+1), x), x) - (x^2*cosh(2*x+1)+x*cosh(2*x+1))",
        ),
        (
            "integrate(x^6*sinh(x), x)",
            "(x^6 + 30·x^4 + 360·x^2 + 720)·cosh(x) - (6·x^5 + 120·x^3 + 720·x)·sinh(x)",
            "diff(integrate(x^6*sinh(x), x), x) - x^6*sinh(x)",
        ),
        (
            "integrate(x^6*cosh(x), x)",
            "(x^6 + 30·x^4 + 360·x^2 + 720)·sinh(x) - (6·x^5 + 120·x^3 + 720·x)·cosh(x)",
            "diff(integrate(x^6*cosh(x), x), x) - x^6*cosh(x)",
        ),
        (
            "integrate((x^6+1)*sinh(2*x+1), x)",
            "(1/2·x^6 + 15/4·x^4 + 45/4·x^2 + 49/8)·cosh(2·x + 1) - (3/2·x^5 + 15/2·x^3 + 45/4·x)·sinh(2·x + 1)",
            "diff(integrate((x^6+1)*sinh(2*x+1), x), x) - (x^6+1)*sinh(2*x+1)",
        ),
        (
            "integrate((x^6+1)*cosh(2*x+1), x)",
            "(1/2·x^6 + 15/4·x^4 + 45/4·x^2 + 49/8)·sinh(2·x + 1) - (3/2·x^5 + 15/2·x^3 + 45/4·x)·cosh(2·x + 1)",
            "diff(integrate((x^6+1)*cosh(2*x+1), x), x) - (x^6+1)*cosh(2*x+1)",
        ),
        (
            "integrate(x^7*sinh(x), x)",
            "(x^7 + 42·x^5 + 840·x^3 + 5040·x)·cosh(x) - (7·x^6 + 210·x^4 + 2520·x^2 + 5040)·sinh(x)",
            "diff(integrate(x^7*sinh(x), x), x) - x^7*sinh(x)",
        ),
        (
            "integrate(x^7*cosh(x), x)",
            "(x^7 + 42·x^5 + 840·x^3 + 5040·x)·sinh(x) - (7·x^6 + 210·x^4 + 2520·x^2 + 5040)·cosh(x)",
            "diff(integrate(x^7*cosh(x), x), x) - x^7*cosh(x)",
        ),
        (
            "integrate((x^7+1)*sinh(2*x+1), x)",
            "(1/2·x^7 + 21/4·x^5 + 105/4·x^3 + 315/8·x + 1/2)·cosh(2·x + 1) - (7/4·x^6 + 105/8·x^4 + 315/8·x^2 + 315/16)·sinh(2·x + 1)",
            "diff(integrate((x^7+1)*sinh(2*x+1), x), x) - (x^7+1)*sinh(2*x+1)",
        ),
        (
            "integrate((x^7+1)*cosh(2*x+1), x)",
            "(1/2·x^7 + 21/4·x^5 + 105/4·x^3 + 315/8·x + 1/2)·sinh(2·x + 1) - (7/4·x^6 + 105/8·x^4 + 315/8·x^2 + 315/16)·cosh(2·x + 1)",
            "diff(integrate((x^7+1)*cosh(2*x+1), x), x) - (x^7+1)*cosh(2*x+1)",
        ),
        (
            "integrate(x^8*cosh(x), x)",
            "(x^8 + 56·x^6 + 1680·x^4 + 20160·x^2)·sinh(x) - (8·x^7 + 336·x^5 + 6720·x^3 + 40320·x)·cosh(x)",
            "diff(integrate(x^8*cosh(x), x), x) - x^8*cosh(x)",
        ),
        (
            "integrate(x^8*cosh(2*x+1), x)",
            "(1/2·x^8 + 7·x^6 + 105/2·x^4 + 315/2·x^2)·sinh(2·x + 1) - (2·x^7 + 21·x^5 + 105·x^3 + 315/2·x)·cosh(2·x + 1)",
            "diff(integrate(x^8*cosh(2*x+1), x), x) - x^8*cosh(2*x+1)",
        ),
        (
            "integrate(x^8*sinh(2*x+1), x)",
            "(1/2·x^8 + 7·x^6 + 105/2·x^4 + 315/2·x^2)·cosh(2·x + 1) - (2·x^7 + 21·x^5 + 105·x^3 + 315/2·x)·sinh(2·x + 1)",
            "diff(integrate(x^8*sinh(2*x+1), x), x) - x^8*sinh(2*x+1)",
        ),
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

        assert_eq!(wire["result"], expected_result);
        assert_eq!(
            wire["required_conditions"]
                .as_array()
                .expect("required_conditions array")
                .len(),
            0,
            "unexpected required_conditions for {input}: {:?}",
            wire["required_conditions"]
        );
        assert!(
            !stderr.contains("depth_overflow"),
            "quadratic hyperbolic by-parts presentation should not emit depth_overflow warning for {input}\nstderr:\n{stderr}"
        );
        assert!(
            wire["steps"]
                .as_array()
                .expect("steps array")
                .iter()
                .flat_map(|step| step["substeps"].as_array().into_iter().flatten())
                .any(|substep| substep["title"] == "Usar integración por partes repetida"),
            "quadratic hyperbolic by-parts should expose repeated integration-by-parts substep: {wire:?}"
        );

        let (wire, stderr) = cli_eval_json_with_stderr(residual);
        assert_eq!(wire["result"], "0");
        assert!(
            !stderr.contains("depth_overflow"),
            "quadratic hyperbolic by-parts verification should not emit depth_overflow warning for {input}\nstderr:\n{stderr}"
        );
    }

    for (lhs, rhs) in [
        (
            "diff(integrate(x^2*sinh(2*x+1)+x*sinh(2*x+1), x), x)",
            "x^2*sinh(2*x+1)+x*sinh(2*x+1)",
        ),
        (
            "diff(integrate(x^2*cosh(2*x+1)+x*cosh(2*x+1), x), x)",
            "x^2*cosh(2*x+1)+x*cosh(2*x+1)",
        ),
    ] {
        let (equivalent, required) = evaluated_equiv_with_required_conditions(lhs, rhs);
        assert!(
            equivalent,
            "public equivalence should reuse the direct residual proof for {lhs} equiv {rhs}"
        );
        assert!(
            required.is_empty(),
            "unexpected public equivalence required conditions for {lhs} equiv {rhs}: {required:?}"
        );
    }
}

#[test]
fn integrate_contract_linear_times_hyperbolic_rational_linear_by_parts() {
    for (input, expected_result, residual) in [
        (
            "integrate((x+1)*sinh((3*x+2)/2), x)",
            "2/3 * cosh((3 * x + 2) / 2) * (x + 1) - 4/9 * sinh((3 * x + 2) / 2)",
            "diff(integrate((x+1)*sinh((3*x+2)/2), x), x) - (x+1)*sinh((3*x+2)/2)",
        ),
        (
            "integrate((x+1)*cosh((3*x+2)/2), x)",
            "2/3 * sinh((3 * x + 2) / 2) * (x + 1) - 4/9 * cosh((3 * x + 2) / 2)",
            "diff(integrate((x+1)*cosh((3*x+2)/2), x), x) - (x+1)*cosh((3*x+2)/2)",
        ),
        (
            "integrate((x+1)*sinh((2-3*x)/2), x)",
            "-2/3 * cosh((2 - 3 * x) / 2) * (x + 1) - 4/9 * sinh((2 - 3 * x) / 2)",
            "diff(integrate((x+1)*sinh((2-3*x)/2), x), x) - (x+1)*sinh((2-3*x)/2)",
        ),
        (
            "integrate((x+1)*cosh((2-3*x)/2), x)",
            "-2/3 * sinh((2 - 3 * x) / 2) * (x + 1) - 4/9 * cosh((2 - 3 * x) / 2)",
            "diff(integrate((x+1)*cosh((2-3*x)/2), x), x) - (x+1)*cosh((2-3*x)/2)",
        ),
    ] {
        let (result, required) = evaluated_integral_with_required_conditions(input);
        assert_eq!(result, expected_result);
        assert!(
            required.is_empty(),
            "unexpected required_conditions for {input}: {required:?}"
        );
        assert_eq!(
            assert_antiderivative_verifies(input),
            AntiderivativeVerificationRoute::PublicResidual,
            "{input} should verify through the bounded public residual route"
        );

        let (residual_result, residual_required) =
            evaluated_expr_with_required_conditions(residual);
        assert_eq!(residual_result, "0");
        assert!(
            residual_required.is_empty(),
            "unexpected residual required_conditions for {input}: {residual_required:?}"
        );
    }
}

#[test]
fn integrate_contract_polynomial_derivative_exp_substitution() {
    assert_eq!(simplified_integral("integrate(2*x*exp(x^2), x)"), "e^(x^2)");
}

#[test]
fn integrate_contract_polynomial_derivative_substitution_exposes_didactic_substep() {
    for (input, expected_result, expected_substep_title) in [
        (
            "integrate(2*x*exp(x^2), x)",
            "e^(x^2)",
            "Usar la regla de exp(u) -> exp(u)",
        ),
        (
            "integrate(2*x*cos(x^2), x)",
            "sin(x^2)",
            "Usar la regla de cos(u) -> sin(u)",
        ),
        (
            "integrate(2*x*sin(x^2), x)",
            "-cos(x^2)",
            "Usar la regla de sin(u) -> -cos(u)",
        ),
        (
            "integrate(2*x*sinh(x^2), x)",
            "cosh(x^2)",
            "Usar la regla de sinh(u) -> cosh(u)",
        ),
        (
            "integrate(2*x*cosh(x^2), x)",
            "sinh(x^2)",
            "Usar la regla de cosh(u) -> sinh(u)",
        ),
        (
            "integrate(2*x*tanh(x^2), x)",
            "ln(cosh(x^2))",
            "Usar la regla de tanh(u) -> ln(cosh(u))",
        ),
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

        assert_eq!(wire["result"], expected_result, "input: {input}");
        assert!(
            wire["required_display"]
                .as_array()
                .expect("required_display should be an array")
                .is_empty(),
            "unexpected required_display for {input}: {:?}",
            wire["required_display"]
        );
        assert!(
            !stderr.contains("depth_overflow"),
            "polynomial derivative substitution trace should not emit depth_overflow warning for {input}\nstderr:\n{stderr}"
        );

        let steps = wire["steps"]
            .as_array()
            .expect("steps should be present with --steps on");
        let integration_step = steps
            .iter()
            .find(|step| step["rule"] == "Calcular la integral")
            .expect("expected public symbolic integration step");
        let substeps = integration_step["substeps"]
            .as_array()
            .expect("integration step should expose didactic substeps");
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == expected_substep_title),
            "expected {expected_substep_title} substep for {input}, got {substeps:?}"
        );
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == "Identificar u y du"),
            "expected concrete u/du substep for {input}, got {substeps:?}"
        );
        let u_du_substep = substeps
            .iter()
            .find(|substep| substep["title"] == "Identificar u y du")
            .expect("expected concrete u/du substep");
        let before_latex = u_du_substep["before_latex"]
            .as_str()
            .expect("u/du substep should expose before_latex");
        let after_latex = u_du_substep["after_latex"]
            .as_str()
            .expect("u/du substep should expose after_latex");
        assert!(
            before_latex.contains("u =") && after_latex.contains("du ="),
            "u/du substep should label substitution evidence for {input}, got {u_du_substep:?}"
        );
        assert!(
            substeps
                .iter()
                .all(|substep| substep["title"] != "Usar sustitución"),
            "polynomial derivative substitution should not use only the generic substitution substep for {input}: {substeps:?}"
        );
        assert_antiderivative_verifies(input);
    }
}

#[test]
fn integrate_contract_symbolic_scale_tanh_substitution_exposes_concrete_trace() {
    let input = "integrate(2*k*x*tanh(x^2+b), x)";
    let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

    assert_eq!(wire["result"], "k·ln(cosh(x^2 + b))");
    assert!(
        wire["required_display"]
            .as_array()
            .expect("required_display should be an array")
            .is_empty(),
        "symbolic tanh substitution should not invent domain conditions: {:?}",
        wire["required_display"]
    );
    assert!(
        !stderr.contains("depth_overflow"),
        "symbolic tanh substitution trace should not emit depth_overflow warning\nstderr:\n{stderr}"
    );

    let steps = wire["steps"]
        .as_array()
        .expect("steps should be present with --steps on");
    let integration_step = steps
        .iter()
        .find(|step| step["rule"] == "Calcular la integral")
        .expect("expected public symbolic integration step");
    let substeps = integration_step["substeps"]
        .as_array()
        .expect("integration step should expose didactic substeps");
    assert!(
        substeps
            .iter()
            .any(|substep| substep["title"] == "Usar la regla de tanh(u) -> ln(cosh(u))"),
        "expected tanh table substep for symbolic scale case, got {substeps:?}"
    );
    assert_u_du_substep_labels(substeps, input);
    assert!(
        substeps
            .iter()
            .any(|substep| substep["title"] == "Ajustar el factor constante"),
        "expected symbolic scale adjustment substep, got {substeps:?}"
    );
    assert!(
        substeps
            .iter()
            .all(|substep| substep["title"] != "Usar sustitución"),
        "symbolic tanh substitution should not fall back to generic substitution only: {substeps:?}"
    );
    assert!(
        steps
            .iter()
            .any(|step| step["rule"] == "Abs Under Positivity"),
        "expected cosh positivity cleanup step for symbolic tanh substitution, got {steps:?}"
    );
    assert_antiderivative_verifies(input);
}

#[test]
fn integrate_contract_log_power_product_substitution_exposes_didactic_substep() {
    for (input, expected_result, expected_substep_title) in [
        (
            "integrate(2*x*ln(x^2+1), x)",
            "(ln(x^2 + 1) - 1)·(x^2 + 1)",
            "Usar la regla de u'·ln(u) -> u·(ln(u)-1)",
        ),
        (
            "integrate(2*x*ln(x^2+1)^2, x)",
            "(x^2 + 1)·(ln(x^2 + 1)^2 - 2·ln(x^2 + 1) + 2)",
            "Usar la regla de u'·ln(u)^n por partes",
        ),
        (
            "integrate(2*x*ln(x^2+1)^3, x)",
            "(ln(x^2 + 1)^3 - 3·ln(x^2 + 1)^2 + 6·ln(x^2 + 1) - 6)·(x^2 + 1)",
            "Usar la regla de u'·ln(u)^n por partes",
        ),
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

        assert_eq!(wire["result"], expected_result, "input: {input}");
        assert!(
            wire["required_display"]
                .as_array()
                .expect("required_display should be an array")
                .is_empty(),
            "unexpected required_display for {input}: {:?}",
            wire["required_display"]
        );
        assert!(
            !stderr.contains("depth_overflow"),
            "log-power product substitution trace should not emit depth_overflow warning for {input}\nstderr:\n{stderr}"
        );

        let steps = wire["steps"]
            .as_array()
            .expect("steps should be present with --steps on");
        assert_eq!(
            steps.len(),
            1,
            "expected compact direct substitution trace for {input}, got {steps:?}"
        );
        let integration_step = steps
            .iter()
            .find(|step| step["rule"] == "Calcular la integral")
            .expect("expected public symbolic integration step");
        let substeps = integration_step["substeps"]
            .as_array()
            .expect("integration step should expose didactic substeps");
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == expected_substep_title),
            "expected log-power product table substep for {input}, got {substeps:?}"
        );
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == "Identificar u y du"),
            "expected concrete u/du substep for {input}, got {substeps:?}"
        );
        assert!(
            substeps
                .iter()
                .all(|substep| substep["title"] != "Usar sustitución"),
            "log-power product table case should not use generic substitution substep for {input}: {substeps:?}"
        );
        assert_antiderivative_verifies(input);
    }
}

#[test]
fn integrate_contract_constant_base_log_power_product_substitution_exposes_didactic_substep() {
    for (input, expected_result, expected_required) in [
        (
            "integrate(2*x*log(2,x^2+1)^2, x)",
            "(x^2 + 1)·(log(2, x^2 + 1)^2 + 2 / ln(2)^2 - 2·log(2, x^2 + 1) / ln(2))",
            vec![],
        ),
        (
            "integrate(2*x*log2(x^2+1)^2, x)",
            "(x^2 + 1)·(log2(x^2 + 1)^2 + 2 / ln(2)^2 - 2·log2(x^2 + 1) / ln(2))",
            vec![],
        ),
        (
            "integrate(2*x*log(2,x^2-1)^2, x)",
            "(x^2 - 1)·(log(2, x^2 - 1)^2 + 2 / ln(2)^2 - 2·log(2, x^2 - 1) / ln(2))",
            vec!["x < -1 or x > 1"],
        ),
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

        assert_eq!(wire["result"], expected_result, "input: {input}");
        assert_eq!(
            wire["required_display"],
            serde_json::json!(expected_required),
            "unexpected required_display for {input}: {:?}",
            wire["required_display"]
        );
        assert!(
            !stderr.contains("depth_overflow"),
            "constant-base log-power substitution should not emit depth_overflow for {input}\nstderr:\n{stderr}"
        );

        let steps = wire["steps"]
            .as_array()
            .expect("steps should be present with --steps on");
        let integration_step = steps
            .iter()
            .find(|step| step["rule"] == "Calcular la integral")
            .expect("expected public symbolic integration step");
        let substeps = integration_step["substeps"]
            .as_array()
            .expect("integration step should expose didactic substeps");
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == "Usar la regla de u'·log_b(u)^n por partes"),
            "expected constant-base log-power substep for {input}, got {substeps:?}"
        );
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == "Identificar u y du"),
            "expected concrete u/du substep for {input}, got {substeps:?}"
        );
        assert!(
            substeps
                .iter()
                .all(|substep| substep["title"] != "Usar sustitución"),
            "constant-base log-power table case should not use generic substitution for {input}: {substeps:?}"
        );
        assert_antiderivative_verifies(input);
    }

    let (invalid_base_result, invalid_base_required) =
        evaluated_integral_with_required_conditions("integrate(2*x*log(1,x^2+1)^2, x)");
    assert_eq!(
        invalid_base_result, "undefined",
        "invalid log base should produce an undefined integrand, not a residual"
    );
    assert!(
        invalid_base_required.is_empty(),
        "undefined invalid-base log integrand should not add assumptions: {invalid_base_required:?}"
    );

    let (symbolic_base_result, _required) =
        evaluated_integral_with_required_conditions("integrate(2*x*log(y,x^2+1)^2, x)");
    assert!(
        symbolic_base_result.starts_with("integrate("),
        "symbolic log base should remain residual, got {symbolic_base_result}"
    );
}

#[test]
fn integrate_contract_polynomial_derivative_cos_substitution() {
    assert_eq!(
        simplified_integral("integrate(2*x*cos(x^2), x)"),
        "sin(x^2)"
    );
}

#[test]
fn integrate_contract_polynomial_derivative_sin_substitution() {
    assert_eq!(
        simplified_integral("integrate(2*x*sin(x^2), x)"),
        "-cos(x^2)"
    );
}

#[test]
fn integrate_contract_linear_hyperbolic_substitution() {
    assert_eq!(
        simplified_integral("integrate(sinh(2*x + 1), x)"),
        "1/2 * cosh(2 * x + 1)"
    );
    assert_eq!(
        simplified_integral("integrate(cosh(2*x + 1), x)"),
        "1/2 * sinh(2 * x + 1)"
    );
}

#[test]
fn integrate_contract_linear_tanh_uses_abs_log_cosh_and_nonzero_domain() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(tanh(2*x + 1), x)");

    assert_eq!(result, "1/2 * ln(cosh(2 * x + 1))");
    assert_eq!(
        required,
        Vec::<String>::new(),
        "unexpected required_conditions: {required:?}"
    );
}

#[test]
fn integrate_contract_polynomial_derivative_hyperbolic_substitution() {
    assert_eq!(
        simplified_integral("integrate(2*x*sinh(x^2), x)"),
        "cosh(x^2)"
    );
    assert_eq!(
        simplified_integral("integrate(2*x*cosh(x^2), x)"),
        "sinh(x^2)"
    );
    assert_eq!(
        simplified_integral("integrate(2*x*tanh(x^2), x)"),
        "ln(cosh(x^2))"
    );
}

#[test]
fn integrate_contract_affine_hyperbolic_power_times_derivative_product() {
    let public_cases = [
        ("integrate(sinh(x)^2*cosh(x), x)", "1/3 * sinh(x)^3"),
        (
            "integrate(2*cosh(2*x + 1)*sinh(2*x + 1)^2, x)",
            "1/3 * sinh(2 * x + 1)^3",
        ),
        (
            "integrate(-2*cosh(2*x + 1)*sinh(2*x + 1)^2, x)",
            "-1/3 * sinh(2 * x + 1)^3",
        ),
        ("integrate(sinh(x)*cosh(x)^2, x)", "1/3 * cosh(x)^3"),
        ("integrate(-sinh(x)*cosh(x)^2, x)", "-1/3 * cosh(x)^3"),
    ];

    for (input, expected) in public_cases {
        let (result, required) = evaluated_integral_with_required_conditions(input);
        assert_eq!(result, expected, "input: {input}");
        assert!(
            required.is_empty(),
            "input: {input}, required: {required:?}"
        );
        assert_rendered_antiderivative_verifies(input, expected);
    }
}

#[test]
fn integrate_contract_hyperbolic_arctan_derivative_substitution() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(cosh(x)/(1+sinh(x)^2), x)");

    assert_eq!(result, "arctan(sinh(x))");
    assert!(
        required.is_empty(),
        "positive one-plus-square denominator should not require source conditions: {required:?}"
    );
    assert_antiderivative_verifies("integrate(cosh(x)/(1+sinh(x)^2), x)");

    let (result, required) = evaluated_integral_with_required_conditions(
        "integrate(2*cosh(2*x+1)/(1+sinh(2*x+1)^2), x)",
    );

    assert_eq!(result, "arctan(sinh(2 * x + 1))");
    assert!(
        required.is_empty(),
        "positive one-plus-square denominator should not require source conditions: {required:?}"
    );
    assert_antiderivative_verifies("integrate(2*cosh(2*x+1)/(1+sinh(2*x+1)^2), x)");

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(2*x*cosh(x^2)/(1+sinh(x^2)^2), x)");

    assert_eq!(result, "arctan(sinh(x^2))");
    assert!(
        required.is_empty(),
        "positive one-plus-square denominator should not require source conditions: {required:?}"
    );
    assert_antiderivative_verifies("integrate(2*x*cosh(x^2)/(1+sinh(x^2)^2), x)");

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(2*x*cosh(x^2)/(sinh(x^2)^2+1), x)");

    assert_eq!(result, "arctan(sinh(x^2))");
    assert!(
        required.is_empty(),
        "positive one-plus-square denominator should not require source conditions: {required:?}"
    );
    assert_antiderivative_verifies("integrate(2*x*cosh(x^2)/(sinh(x^2)^2+1), x)");

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(sinh(x)/(1+cosh(x)^2), x)");

    assert_eq!(result, "arctan(cosh(x))");
    assert!(
        required.is_empty(),
        "positive one-plus-square denominator should not require source conditions: {required:?}"
    );
    assert_antiderivative_verifies("integrate(sinh(x)/(1+cosh(x)^2), x)");

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(-2*x*sinh(x^2)/(1+cosh(x^2)^2), x)");

    assert_eq!(result, "-arctan(cosh(x^2))");
    assert!(
        required.is_empty(),
        "positive one-plus-square denominator should not require source conditions: {required:?}"
    );
    assert_antiderivative_verifies("integrate(-2*x*sinh(x^2)/(1+cosh(x^2)^2), x)");
}

#[test]
fn integrate_contract_trig_arctan_derivative_substitution() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(cos(x)/(1+sin(x)^2), x)");

    assert_eq!(result, "arctan(sin(x))");
    assert!(
        required.is_empty(),
        "positive one-plus-square denominator should not require source conditions: {required:?}"
    );
    assert_rendered_antiderivative_verifies("integrate(cos(x)/(1+sin(x)^2), x)", &result);

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(2*cos(2*x+1)/(1+sin(2*x+1)^2), x)");

    assert_eq!(result, "arctan(sin(2 * x + 1))");
    assert!(
        required.is_empty(),
        "positive one-plus-square denominator should not require source conditions: {required:?}"
    );
    assert_rendered_antiderivative_verifies("integrate(2*cos(2*x+1)/(1+sin(2*x+1)^2), x)", &result);

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(2*cos(2*x+1)/(sin(2*x+1)^2+1), x)");

    assert_eq!(result, "arctan(sin(2 * x + 1))");
    assert!(
        required.is_empty(),
        "positive one-plus-square denominator should not require source conditions: {required:?}"
    );
    assert_rendered_antiderivative_verifies("integrate(2*cos(2*x+1)/(sin(2*x+1)^2+1), x)", &result);

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(2*x*cos(x^2)/(1+sin(x^2)^2), x)");

    assert_eq!(result, "arctan(sin(x^2))");
    assert!(
        required.is_empty(),
        "positive one-plus-square denominator should not require source conditions: {required:?}"
    );
    assert_rendered_antiderivative_verifies("integrate(2*x*cos(x^2)/(1+sin(x^2)^2), x)", &result);

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(2*x*cos(x^2)/(sin(x^2)^2+1), x)");

    assert_eq!(result, "arctan(sin(x^2))");
    assert!(
        required.is_empty(),
        "positive one-plus-square denominator should not require source conditions: {required:?}"
    );
    assert_rendered_antiderivative_verifies("integrate(2*x*cos(x^2)/(sin(x^2)^2+1), x)", &result);

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(-sin(x)/(1+cos(x)^2), x)");

    assert_eq!(result, "arctan(cos(x))");
    assert!(
        required.is_empty(),
        "positive one-plus-square denominator should not require source conditions: {required:?}"
    );
    assert_rendered_antiderivative_verifies("integrate(-sin(x)/(1+cos(x)^2), x)", &result);

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(-2*sin(2*x+1)/(cos(2*x+1)^2+1), x)");

    assert_eq!(result, "arctan(cos(2 * x + 1))");
    assert!(
        required.is_empty(),
        "positive one-plus-square denominator should not require source conditions: {required:?}"
    );
    assert_rendered_antiderivative_verifies(
        "integrate(-2*sin(2*x+1)/(cos(2*x+1)^2+1), x)",
        &result,
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(-2*x*sin(x^2)/(cos(x^2)^2+1), x)");

    assert_eq!(result, "arctan(cos(x^2))");
    assert!(
        required.is_empty(),
        "positive one-plus-square denominator should not require source conditions: {required:?}"
    );
    assert_rendered_antiderivative_verifies("integrate(-2*x*sin(x^2)/(cos(x^2)^2+1), x)", &result);
}

#[test]
fn integrate_contract_hyperbolic_non_linear_argument_without_cofactor_remains_residual() {
    let (result, required) = evaluated_integral_with_required_conditions("integrate(sinh(x^2), x)");

    assert_eq!(result, "integrate(sinh(x^2), x)");
    assert!(
        required.is_empty(),
        "unsupported hyperbolic integral should not invent conditions: {required:?}"
    );

    let (result, required) = evaluated_integral_with_required_conditions("integrate(tanh(x^2), x)");

    assert_eq!(result, "integrate(tanh(x^2), x)");
    assert!(
        required.is_empty(),
        "unsupported tanh integral should not invent conditions: {required:?}"
    );
}

#[test]
fn integrate_contract_hyperbolic_ratio_uses_tanh_kernel_and_preserves_source_domain() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(sinh(2*x + 1)/cosh(2*x + 1), x)");

    assert_eq!(result, "1/2 * ln(cosh(2 * x + 1))");
    assert_eq!(
        required,
        Vec::<String>::new(),
        "unexpected required_conditions: {required:?}"
    );
}

#[test]
fn integrate_contract_hyperbolic_coth_ratio_uses_log_sinh_and_preserves_source_domain() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(cosh(2*x + 1)/sinh(2*x + 1), x)");

    assert_eq!(result, "1/2 * ln(|sinh(2 * x + 1)|)");
    assert_eq!(
        required,
        vec!["sinh(2 * x + 1) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}

#[test]
fn integrate_contract_hyperbolic_tanh_reciprocal_uses_log_sinh_and_preserves_domains() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(1/tanh(2*x + 1), x)");

    assert_eq!(result, "1/2 * ln(|sinh(2 * x + 1)|)");
    assert_eq!(
        required,
        vec!["sinh(2 * x + 1) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}

#[test]
fn integrate_contract_polynomial_hyperbolic_coth_ratio_substitution() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(2*x*cosh(x^2)/sinh(x^2), x)");

    assert_eq!(result, "ln(|sinh(x^2)|)");
    assert_eq!(
        required,
        vec!["sinh(x^2) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}

#[test]
fn integrate_contract_polynomial_hyperbolic_tanh_reciprocal_substitution() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(2*x/tanh(x^2), x)");

    assert_eq!(result, "ln(|sinh(x^2)|)");
    assert_eq!(
        required,
        vec!["sinh(x^2) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}

#[test]
fn integrate_contract_linear_hyperbolic_tanh_derivative_square_substitution() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(1/cosh(2*x + 1)^2, x)");

    assert_eq!(result, "1/2 * tanh(2 * x + 1)");
    assert_eq!(
        required,
        Vec::<String>::new(),
        "unexpected required_conditions: {required:?}"
    );
}

#[test]
fn integrate_contract_polynomial_hyperbolic_tanh_derivative_square_substitution() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(2*x/cosh(x^2)^2, x)");

    assert_eq!(result, "tanh(x^2)");
    assert_eq!(
        required,
        Vec::<String>::new(),
        "unexpected required_conditions: {required:?}"
    );
}

#[test]
fn integrate_contract_hyperbolic_quotient_substitution_exposes_didactic_substep() {
    for (input, expected_result, expected_required_display, expected_substep_title) in [
        (
            "integrate(2*x*cosh(x^2)/sinh(x^2), x)",
            "ln(|sinh(x^2)|)",
            serde_json::json!(["sinh(x^2) ≠ 0"]),
            "Usar la regla de cosh(u)/sinh(u) -> ln|sinh(u)|",
        ),
        (
            "integrate(2*x/tanh(x^2), x)",
            "ln(|sinh(x^2)|)",
            serde_json::json!(["sinh(x^2) ≠ 0"]),
            "Usar la regla de 1/tanh(u) -> ln|sinh(u)|",
        ),
        (
            "integrate(2*x/cosh(x^2)^2, x)",
            "tanh(x^2)",
            serde_json::json!([]),
            "Usar la regla de 1/cosh(u)^2 -> tanh(u)",
        ),
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

        assert_eq!(wire["result"], expected_result, "input: {input}");
        assert_eq!(
            wire["required_display"], expected_required_display,
            "unexpected required_display for {input}: {:?}",
            wire["required_display"]
        );
        assert!(
            !stderr.contains("depth_overflow"),
            "hyperbolic quotient substitution trace should not emit depth_overflow warning for {input}\nstderr:\n{stderr}"
        );

        let steps = wire["steps"]
            .as_array()
            .expect("steps should be present with --steps on");
        let integration_step = steps
            .iter()
            .find(|step| step["rule"] == "Calcular la integral")
            .expect("expected public symbolic integration step");
        let substeps = integration_step["substeps"]
            .as_array()
            .expect("integration step should expose didactic substeps");
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == expected_substep_title),
            "expected hyperbolic quotient table substep for {input}, got {substeps:?}"
        );
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == "Identificar u y du"),
            "expected concrete u/du substep for {input}, got {substeps:?}"
        );
        assert!(
            substeps
                .iter()
                .all(|substep| substep["title"] != "Usar sustitución"),
            "expected specific hyperbolic quotient trace without generic substitution for {input}, got {substeps:?}"
        );
        assert_antiderivative_verifies(input);
    }
}

#[test]
fn integrate_contract_trig_quotient_substitution_exposes_didactic_substep() {
    for (input, expected_result, expected_required_display, expected_substep_title) in [
        (
            "integrate(2*x*tan(x^2), x)",
            "-ln(|cos(x^2)|)",
            serde_json::json!(["cos(x^2) ≠ 0"]),
            "Usar la regla de tan(u) -> -ln|cos(u)|",
        ),
        (
            "integrate(3*x^2*cot(x^3), x)",
            "ln(|sin(x^3)|)",
            serde_json::json!(["sin(x^3) ≠ 0"]),
            "Usar la regla de cot(u) -> ln|sin(u)|",
        ),
        (
            "integrate(2*x/cos(x^2)^2, x)",
            "tan(x^2)",
            serde_json::json!(["cos(x^2) ≠ 0"]),
            "Usar la regla de 1/cos(u)^2 -> tan(u)",
        ),
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

        assert_eq!(wire["result"], expected_result, "input: {input}");
        assert_eq!(
            wire["required_display"], expected_required_display,
            "unexpected required_display for {input}: {:?}",
            wire["required_display"]
        );
        assert!(
            !stderr.contains("depth_overflow"),
            "trig quotient substitution trace should not emit depth_overflow warning for {input}\nstderr:\n{stderr}"
        );

        let steps = wire["steps"]
            .as_array()
            .expect("steps should be present with --steps on");
        let integration_step = steps
            .iter()
            .find(|step| step["rule"] == "Calcular la integral")
            .expect("expected public symbolic integration step");
        let substeps = integration_step["substeps"]
            .as_array()
            .expect("integration step should expose didactic substeps");
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == expected_substep_title),
            "expected trig quotient table substep for {input}, got {substeps:?}"
        );
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == "Identificar u y du"),
            "expected u/du identification substep for {input}, got {substeps:?}"
        );
        assert!(
            substeps
                .iter()
                .all(|substep| substep["title"] != "Usar sustitución"),
            "expected specific trig quotient trace without generic substitution for {input}, got {substeps:?}"
        );
        assert_antiderivative_verifies(input);
    }

    let (wire, _stderr) =
        cli_eval_json_with_stderr_args("integrate(tan(x^2), x)", &["--steps", "on"]);
    assert_eq!(wire["result"], "integrate(tan(x^2), x)");
    let steps = wire["steps"]
        .as_array()
        .expect("steps should be present with --steps on");
    assert!(
        !steps
            .iter()
            .any(|step| step["rule"] == "Calcular la integral"),
        "unsupported missing-cofactor tan(x^2) should not emit a fake integration step: {steps:?}"
    );
}

#[test]
fn integrate_contract_direct_sec_csc_derivative_quotients_expose_didactic_substep() {
    for (input, expected_result, expected_required_display) in [
        (
            "integrate(2*x*sin(x^2)/cos(x^2)^2, x)",
            "sec(x^2)",
            serde_json::json!(["cos(x^2) ≠ 0"]),
        ),
        (
            "integrate(3*x^2*cos(x^3)/sin(x^3)^2, x)",
            "-csc(x^3)",
            serde_json::json!(["sin(x^3) ≠ 0"]),
        ),
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

        assert_eq!(wire["result"], expected_result, "input: {input}");
        assert_eq!(
            wire["required_display"], expected_required_display,
            "unexpected required_display for {input}: {:?}",
            wire["required_display"]
        );
        assert!(
            !stderr.contains("depth_overflow"),
            "direct sec/csc derivative quotient trace should not emit depth_overflow warning for {input}\nstderr:\n{stderr}"
        );

        let steps = wire["steps"]
            .as_array()
            .expect("steps should be present with --steps on");
        assert_eq!(
            steps.len(),
            1,
            "expected direct public integration step for {input}, got {steps:?}"
        );
        let substeps = steps[0]["substeps"]
            .as_array()
            .expect("direct integration step should expose didactic substeps");
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == "Usar sustitución"),
            "expected substitution substep for {input}, got {substeps:?}"
        );
        assert_antiderivative_verifies(input);
    }

    let (wire, _stderr) =
        cli_eval_json_with_stderr_args("integrate(sin(x^2)/cos(x^2)^2, x)", &["--steps", "on"]);
    assert_eq!(wire["result"], "integrate(sin(x^2) / cos(x^2)^2, x)");
    if let Some(steps) = wire["steps"].as_array() {
        assert!(
            !steps.iter().any(|step| step["substeps"].is_array()),
            "unsupported missing-cofactor sec derivative quotient should not emit a fake substep: {steps:?}"
        );
    }
}

#[test]
fn integrate_contract_direct_trig_log_substitution_exposes_didactic_substep() {
    for (input, expected_result, expected_required_display, expected_substep_title) in [
        (
            "integrate(tan(2*x+1), x)",
            "-1/2·ln(|cos(2·x + 1)|)",
            serde_json::json!(["cos(2·x + 1) ≠ 0"]),
            "Usar la regla de tan(u) -> -ln|cos(u)|",
        ),
        (
            "integrate(cot(2*x+1), x)",
            "1/2·ln(|sin(2·x + 1)|)",
            serde_json::json!(["sin(2·x + 1) ≠ 0"]),
            "Usar la regla de cot(u) -> ln|sin(u)|",
        ),
        (
            "integrate(sec(2*x+1), x)",
            "1/2·ln(|tan(2·x + 1) + sec(2·x + 1)|)",
            serde_json::json!(["cos(2·x + 1) ≠ 0"]),
            "Usar la regla de sec(u) -> ln|sec(u)+tan(u)|",
        ),
        (
            "integrate(csc(2*x+1), x)",
            "1/2·ln(|csc(2·x + 1) - cot(2·x + 1)|)",
            serde_json::json!(["sin(2·x + 1) ≠ 0"]),
            "Usar la regla de csc(u) -> ln|csc(u)-cot(u)|",
        ),
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

        assert_eq!(wire["result"], expected_result, "input: {input}");
        assert_eq!(
            wire["required_display"], expected_required_display,
            "unexpected required_display for {input}: {:?}",
            wire["required_display"]
        );
        assert!(
            !stderr.contains("depth_overflow"),
            "trig log substitution trace should not emit depth_overflow warning for {input}\nstderr:\n{stderr}"
        );

        let steps = wire["steps"]
            .as_array()
            .expect("steps should be present with --steps on");
        let integration_step = steps
            .iter()
            .find(|step| step["rule"] == "Calcular la integral")
            .expect("expected public symbolic integration step");
        let substeps = integration_step["substeps"]
            .as_array()
            .expect("integration step should expose didactic substeps");
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == expected_substep_title),
            "expected {expected_substep_title} substep for {input}, got {substeps:?}"
        );
        if expected_substep_title != "Usar sustitución" {
            assert!(
                substeps
                    .iter()
                    .all(|substep| substep["title"] != "Usar sustitución"),
                "direct trig log table case should not use the generic substitution substep for {input}: {substeps:?}"
            );
            assert!(
                substeps
                    .iter()
                    .any(|substep| substep["title"] == "Identificar el argumento afín"),
                "expected affine argument substep for {input}, got {substeps:?}"
            );
        }
        assert!(
            steps.iter().filter(|step| step["rule"] != "Calcular la integral").all(|step| {
                step["substeps"]
                    .as_array()
                    .is_none_or(|substeps| {
                        !substeps
                            .iter()
                            .any(|substep| substep["title"] == "Usar sustitución")
                    })
            }),
            "non-integration prep steps should not get substitution substeps for {input}: {steps:?}"
        );
    }

    let (wire, _stderr) =
        cli_eval_json_with_stderr_args("integrate(tan(x^2), x)", &["--steps", "on"]);
    assert_eq!(wire["result"], "integrate(tan(x^2), x)");
    if let Some(steps) = wire["steps"].as_array() {
        assert!(
            steps.iter().all(|step| {
                step["substeps"].as_array().is_none_or(|substeps| {
                    substeps.iter().all(|substep| {
                        substep["title"] != "Usar sustitución"
                            && substep["title"] != "Identificar u y du"
                            && substep["title"] != "Usar la regla de tan(u) -> -ln|cos(u)|"
                    })
                })
            }),
            "unsupported nonlinear tan(x^2) should not emit a fake substitution substep: {steps:?}"
        );
        assert!(
            steps.iter().any(|step| {
                step["rule"] == "Conservar integral residual"
                    && step["substeps"].as_array().is_some_and(|substeps| {
                        substeps.iter().any(|substep| {
                            substep["title"] == "Registrar polo del integrando"
                                && substep["after_latex"] == "\\cos({x}^{2}) \\ne 0"
                        })
                    })
            }),
            "unsupported nonlinear tan(x^2) should expose only the residual domain policy: {steps:?}"
        );
    }
}

#[test]
fn integrate_contract_polynomial_base_substitution_exposes_didactic_substep() {
    for (input, expected_result, expected_required_display, expected_rule_title) in [
        (
            "integrate((2*x+1)/(x^2+x-1), x)",
            "ln(|x^2 + x - 1|)",
            serde_json::json!(["x^2 + x - 1 ≠ 0"]),
            "Usar la regla de u'/u -> ln|u|",
        ),
        (
            "integrate((2*x+1)/(x^2+x-1)^3, x)",
            "-1 / (2·(x^2 + x - 1)^2)",
            serde_json::json!(["x^2 + x - 1 ≠ 0"]),
            "Usar la regla de u'/u^n -> u^(1-n)/(1-n)",
        ),
        (
            "integrate(x/sqrt(x^2+1), x)",
            "sqrt(x^2 + 1)",
            serde_json::json!([]),
            "Usar la regla de u'/sqrt(u) -> 2*sqrt(u)",
        ),
        (
            "integrate(2*x/sqrt(x^2-1), x)",
            "2·sqrt(x^2 - 1)",
            serde_json::json!(["x < -1 or x > 1"]),
            "Usar la regla de u'/sqrt(u) -> 2*sqrt(u)",
        ),
        (
            "integrate(2*x*(x^2-1)^(3/2), x)",
            "2/5·(x^2 - 1)^(5/2)",
            serde_json::json!(["x ≤ -1 or x ≥ 1"]),
            "Usar la regla de u'·u^p -> u^(p+1)/(p+1)",
        ),
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

        assert_eq!(wire["result"], expected_result, "input: {input}");
        assert_eq!(
            wire["required_display"], expected_required_display,
            "unexpected required_display for {input}: {:?}",
            wire["required_display"]
        );
        assert!(
            !stderr.contains("depth_overflow"),
            "polynomial-base substitution trace should not emit depth_overflow warning for {input}\nstderr:\n{stderr}"
        );

        let steps = wire["steps"]
            .as_array()
            .expect("steps should be present with --steps on");
        let integration_step = steps
            .iter()
            .find(|step| step["rule"] == "Calcular la integral")
            .expect("expected public symbolic integration step");
        let substeps = integration_step["substeps"]
            .as_array()
            .expect("integration step should expose didactic substeps");
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == expected_rule_title),
            "expected polynomial-base table substep for {input}, got {substeps:?}"
        );
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == "Identificar u y du"),
            "expected concrete u/du substep for {input}, got {substeps:?}"
        );
        assert_u_du_substep_labels(substeps, input);
        assert!(
            substeps
                .iter()
                .all(|substep| substep["title"] != "Usar sustitución"),
            "polynomial-base table case should not use generic substitution substep for {input}: {substeps:?}"
        );
        assert!(
            steps.iter().filter(|step| step["rule"] != "Calcular la integral").all(|step| {
                step["substeps"]
                    .as_array()
                    .is_none_or(|substeps| {
                        !substeps
                            .iter()
                            .any(|substep| substep["title"] == "Usar sustitución")
                    })
            }),
            "non-integration prep steps should not get substitution substeps for {input}: {steps:?}"
        );
    }

    for input in ["integrate(1/(x^2+1), x)", "integrate(1/sqrt(x^2+1), x)"] {
        let (wire, _stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);
        let steps = wire["steps"]
            .as_array()
            .expect("steps should be present with --steps on");
        assert!(
            steps.iter().all(|step| {
                step["substeps"]
                    .as_array()
                    .is_none_or(|substeps| {
                        !substeps
                            .iter()
                            .any(|substep| substep["title"] == "Usar sustitución")
                    })
            }),
            "table-form integral without polynomial cofactor should not emit a fake substitution substep for {input}: {steps:?}"
        );
    }
}

#[test]
fn integrate_contract_nested_inverse_polynomial_substitution_exposes_didactic_substep() {
    for (input, expected_result, expected_required_display, expected_rule_title) in [
        (
            "integrate(2*x/sqrt(4-x^4), x)",
            "arcsin(x^2 / 2)",
            serde_json::json!(["4 - x^4 > 0"]),
            "Usar la regla de u'/sqrt(1-u^2) -> arcsin(u)",
        ),
        (
            "integrate(2*x/sqrt(1+x^4), x)",
            "asinh(x^2)",
            serde_json::json!([]),
            "Usar la regla de u'/sqrt(1+u^2) -> asinh(u)",
        ),
        (
            "integrate(2*x/(1+x^4), x)",
            "arctan(x^2)",
            serde_json::json!([]),
            "Usar la regla de u'/(1+u^2) -> arctan(u)",
        ),
        (
            "integrate(2*x/(4-x^4), x)",
            "1/2·atanh(x^2 / 2)",
            serde_json::json!(["4 - x^4 > 0"]),
            "Usar la regla de u'/(1-u^2) -> atanh(u)",
        ),
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

        assert_eq!(wire["result"], expected_result, "input: {input}");
        assert_eq!(
            wire["required_display"], expected_required_display,
            "unexpected required_display for {input}: {:?}",
            wire["required_display"]
        );
        assert!(
            !stderr.contains("depth_overflow"),
            "nested inverse polynomial substitution trace should not emit depth_overflow warning for {input}\nstderr:\n{stderr}"
        );

        let steps = wire["steps"]
            .as_array()
            .expect("steps should be present with --steps on");
        let integration_step = steps
            .iter()
            .find(|step| step["rule"] == "Calcular la integral")
            .expect("expected public symbolic integration step");
        let substeps = integration_step["substeps"]
            .as_array()
            .expect("integration step should expose didactic substeps");
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == expected_rule_title),
            "expected nested inverse polynomial table substep for {input}, got {substeps:?}"
        );
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == "Identificar u y du"),
            "expected concrete u/du substep for {input}, got {substeps:?}"
        );
        assert!(
            substeps
                .iter()
                .all(|substep| substep["title"] != "Usar sustitución"),
            "nested inverse polynomial table case should not use generic substitution substep for {input}: {substeps:?}"
        );
        assert!(
            steps.iter().filter(|step| step["rule"] != "Calcular la integral").all(|step| {
                step["substeps"]
                    .as_array()
                    .is_none_or(|substeps| {
                        !substeps
                            .iter()
                            .any(|substep| substep["title"] == "Usar sustitución")
                    })
            }),
            "non-integration prep steps should not get substitution substeps for {input}: {steps:?}"
        );
    }

    for input in [
        "integrate(1/(x^2+1), x)",
        "integrate(1/(1-x^2), x)",
        "integrate(1/(4-x^2), x)",
        "integrate(1/sqrt(x^2+1), x)",
        "integrate(1/sqrt(1-x^2), x)",
        "integrate(1/sqrt(4-(x+1)^2), x)",
        "integrate(1/sqrt(4+(x+1)^2), x)",
        "integrate(2/(1+(2*x+1)^2), x)",
    ] {
        let (wire, _stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);
        let steps = wire["steps"]
            .as_array()
            .expect("steps should be present with --steps on");
        assert!(
            steps.iter().all(|step| {
                step["substeps"]
                    .as_array()
                    .is_none_or(|substeps| {
                        !substeps
                            .iter()
                            .any(|substep| substep["title"] == "Usar sustitución")
                    })
            }),
            "table or linear inverse primitive should not emit a fake substitution substep for {input}: {steps:?}"
        );
    }
}

#[test]
fn integrate_contract_linear_inverse_table_explains_internal_derivative_without_substitution() {
    for (
        input,
        expected_result,
        expected_required_display,
        expected_substep_title,
        expect_constant_adjustment,
    ) in [
        (
            "integrate(1/sqrt(9-(3*x-2)^2), x)",
            "1/3·arcsin((3·x - 2) / 3)",
            serde_json::json!(["-1/3 < x < 5/3"]),
            "Usar la regla de arcsin con derivada interna",
            true,
        ),
        (
            "integrate(-1/sqrt(9-(3*x-2)^2), x)",
            "-1/3·arcsin((3·x - 2) / 3)",
            serde_json::json!(["-1/3 < x < 5/3"]),
            "Usar la regla de arcsin con derivada interna",
            true,
        ),
        (
            "integrate(3/sqrt(9-(3*x-2)^2), x)",
            "arcsin((3·x - 2) / 3)",
            serde_json::json!(["-1/3 < x < 5/3"]),
            "Usar la regla de arcsin con derivada interna",
            false,
        ),
        (
            "integrate(1/(1+(2*x+1)^2), x)",
            "1/2·arctan(2·x + 1)",
            serde_json::json!([]),
            "Usar la regla de arctan con derivada interna",
            true,
        ),
        (
            "integrate(1/(1+(1-2*x)^2), x)",
            "1/2·arctan(2·x - 1)",
            serde_json::json!([]),
            "Usar la regla de arctan con derivada interna",
            true,
        ),
        (
            "integrate(2/(4-(2*x+1)^2), x)",
            "1/2·atanh((2·x + 1) / 2)",
            serde_json::json!(["-3/2 < x < 1/2"]),
            "Usar la regla de atanh con derivada interna",
            true,
        ),
        (
            "integrate(2/(4-(1-2*x)^2), x)",
            "1/2·atanh((2·x - 1) / 2)",
            serde_json::json!(["-1/2 < x < 3/2"]),
            "Usar la regla de atanh con derivada interna",
            true,
        ),
        (
            "integrate(1/sqrt((2*x+1)^2+1), x)",
            "1/2·asinh(2·x + 1)",
            serde_json::json!([]),
            "Usar la regla de asinh con derivada interna",
            true,
        ),
        (
            "integrate(1/sqrt((1-2*x)^2+1), x)",
            "1/2·asinh(2·x - 1)",
            serde_json::json!([]),
            "Usar la regla de asinh con derivada interna",
            true,
        ),
        (
            "integrate(2/(sqrt(2*x)*sqrt(2*x+2)), x)",
            "acosh(2·x + 1)",
            serde_json::json!(["x > 0"]),
            "Usar la regla de acosh con derivada interna",
            false,
        ),
        (
            "integrate(-1/(2*sqrt(x)*sqrt(x+1)), x)",
            "-1/2·acosh(2·x + 1)",
            serde_json::json!(["x > 0"]),
            "Usar la regla de acosh con derivada interna",
            true,
        ),
        (
            "integrate(2/sqrt((2*x-1)^2-1), x)",
            "acosh(2·x - 1)",
            serde_json::json!(["x > 1"]),
            "Usar la regla de acosh con derivada interna",
            false,
        ),
        (
            "integrate(-2/(sqrt(-2*x)*sqrt(2-2*x)), x)",
            "acosh(1 - 2·x)",
            serde_json::json!(["x < 0"]),
            "Usar la regla de acosh con derivada interna",
            false,
        ),
        (
            "integrate(-2/sqrt((2*x-1)^2-1), x)",
            "acosh(1 - 2·x)",
            serde_json::json!(["x < 0"]),
            "Usar la regla de acosh con derivada interna",
            false,
        ),
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

        assert_eq!(wire["result"], expected_result, "input: {input}");
        assert_eq!(
            wire["required_display"], expected_required_display,
            "unexpected required_display for {input}: {:?}",
            wire["required_display"]
        );
        assert!(
            !stderr.contains("depth_overflow"),
            "linear inverse table trace should not emit depth_overflow warning for {input}\nstderr:\n{stderr}"
        );

        let steps = wire["steps"]
            .as_array()
            .expect("steps should be present with --steps on");
        let integration_step = steps
            .iter()
            .find(|step| step["rule"] == "Calcular la integral")
            .expect("expected public symbolic integration step");
        let substeps = integration_step["substeps"]
            .as_array()
            .expect("integration step should expose didactic substeps");
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == expected_substep_title),
            "expected {expected_substep_title} substep for {input}, got {substeps:?}"
        );
        assert!(
            substeps
                .iter()
                .all(|substep| substep["title"] != "Usar sustitución"),
            "linear inverse table case should not use the polynomial-substitution substep for {input}: {substeps:?}"
        );
        assert_eq!(
            substeps
                .iter()
                .any(|substep| substep["title"] == "Ajustar el factor constante"),
            expect_constant_adjustment,
            "unexpected constant adjustment substep presence for {input}: {substeps:?}"
        );
        assert_antiderivative_verifies(input);
    }
}

#[test]
fn integrate_contract_affine_sqrt_product_public_diff_verifies_antiderivative() {
    for (input, expected_diff, expected_required) in [
        (
            "integrate(1/(sqrt(x)*sqrt(x+5)), x)",
            "1 / (sqrt(x) * sqrt(x + 5))",
            vec!["x > 0".to_string()],
        ),
        (
            "integrate(1/(sqrt(x-1)*sqrt(x+2)), x)",
            "1 / (sqrt(x + 2) * sqrt(x - 1))",
            vec!["x > 1".to_string()],
        ),
        (
            "integrate(1/(sqrt(x)*sqrt(2*x+4)), x)",
            "1 / (sqrt(x) * sqrt(2 * x + 4))",
            vec!["x > 0".to_string()],
        ),
        (
            "integrate(1/(sqrt(2*x)*sqrt(2*x+6)), x)",
            "1 / (sqrt(2) * sqrt(x) * sqrt(2 * x + 6))",
            vec!["x > 0".to_string()],
        ),
    ] {
        let diff_input = format!("diff({input}, x)");
        let (result, required) = evaluated_expr_with_required_conditions(&diff_input);

        assert_eq!(result, expected_diff, "input: {diff_input}");
        assert_eq!(required, expected_required, "input: {diff_input}");
        assert_eq!(
            assert_antiderivative_verifies(input),
            AntiderivativeVerificationRoute::PublicResidual,
            "affine sqrt-product antiderivative should verify through the public residual"
        );
    }
}

#[test]
fn integrate_contract_acosh_root_product_uses_compact_unit_offset_affine_arg() {
    for (input, expected_result) in [
        ("integrate(1/(sqrt(x)*sqrt(x+3)), x)", "acosh(2/3 * x + 1)"),
        ("integrate(1/(sqrt(x)*sqrt(x+5)), x)", "acosh(2/5 * x + 1)"),
        (
            "integrate(1/(sqrt(x)*sqrt(2*x+4)), x)",
            "acosh(x + 1) / sqrt(2)",
        ),
    ] {
        let (result, required) = evaluated_integral_with_required_conditions(input);

        assert_eq!(result, expected_result, "input: {input}");
        assert!(
            !result.contains("*(x +") && !result.contains("* (x +"),
            "post-calculus presentation should not leave a scaled shifted group in acosh: {result}"
        );
        assert_eq!(
            required,
            vec!["x > 0".to_string()],
            "unexpected required_conditions for {input}: {required:?}"
        );
        assert_antiderivative_verifies(input);
        assert_rendered_antiderivative_verifies(input, &result);

        let residual = format!(
            "diff({result}, x) - {}",
            input
                .strip_prefix("integrate(")
                .and_then(|rest| rest.strip_suffix(", x)"))
                .expect("test input should be explicit integrate(expr, x)")
        );
        let (residual_result, residual_required) =
            evaluated_expr_with_required_conditions(&residual);
        assert_eq!(residual_result, "0", "residual: {residual}");
        assert_eq!(
            residual_required,
            vec!["x > 0".to_string()],
            "unexpected residual required_conditions for {input}: {residual_required:?}"
        );
    }
}

#[test]
fn integrate_contract_linear_elementary_table_explains_internal_derivative_without_substitution() {
    for (input, expected_result, expected_substep_title, expect_constant_adjustment) in [
        (
            "integrate(sin(x+1), x)",
            "-cos(x + 1)",
            "Usar la regla de sin con derivada interna",
            false,
        ),
        (
            "integrate(sin(2*x+1), x)",
            "-1/2·cos(2·x + 1)",
            "Usar la regla de sin con derivada interna",
            true,
        ),
        (
            "integrate(sin(1-2*x), x)",
            "1/2·cos(1 - 2·x)",
            "Usar la regla de sin con derivada interna",
            true,
        ),
        (
            "integrate(cos(2*x+1), x)",
            "1/2·sin(2·x + 1)",
            "Usar la regla de cos con derivada interna",
            true,
        ),
        (
            "integrate(cos(1-2*x), x)",
            "-1/2·sin(1 - 2·x)",
            "Usar la regla de cos con derivada interna",
            true,
        ),
        (
            "integrate(exp(2*x+1), x)",
            "1/2·e^(2·x + 1)",
            "Usar la regla de exp con derivada interna",
            true,
        ),
        (
            "integrate(exp(1-2*x), x)",
            "-1/2·e^(1 - 2·x)",
            "Usar la regla de exp con derivada interna",
            true,
        ),
        (
            "integrate(sinh(2*x+1), x)",
            "1/2·cosh(2·x + 1)",
            "Usar la regla de sinh con derivada interna",
            true,
        ),
        (
            "integrate(sinh(1-2*x), x)",
            "-1/2·cosh(1 - 2·x)",
            "Usar la regla de sinh con derivada interna",
            true,
        ),
        (
            "integrate(cosh(2*x+1), x)",
            "1/2·sinh(2·x + 1)",
            "Usar la regla de cosh con derivada interna",
            true,
        ),
        (
            "integrate(cosh(1-2*x), x)",
            "-1/2·sinh(1 - 2·x)",
            "Usar la regla de cosh con derivada interna",
            true,
        ),
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

        assert_eq!(wire["result"], expected_result, "input: {input}");
        assert_eq!(
            wire["required_display"],
            serde_json::json!([]),
            "unexpected required_display for {input}: {:?}",
            wire["required_display"]
        );
        assert!(
            !stderr.contains("depth_overflow"),
            "linear elementary table trace should not emit depth_overflow warning for {input}\nstderr:\n{stderr}"
        );

        let steps = wire["steps"]
            .as_array()
            .expect("steps should be present with --steps on");
        let integration_step = steps
            .iter()
            .find(|step| step["rule"] == "Calcular la integral")
            .expect("expected public symbolic integration step");
        let substeps = integration_step["substeps"]
            .as_array()
            .expect("integration step should expose didactic substeps");
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == expected_substep_title),
            "expected {expected_substep_title} substep for {input}, got {substeps:?}"
        );
        assert!(
            substeps
                .iter()
                .all(|substep| substep["title"] != "Usar sustitución"),
            "linear elementary table case should not use the polynomial-substitution substep for {input}: {substeps:?}"
        );
        assert_eq!(
            substeps
                .iter()
                .any(|substep| substep["title"] == "Ajustar el factor constante"),
            expect_constant_adjustment,
            "unexpected constant adjustment substep presence for {input}: {substeps:?}"
        );
        assert_antiderivative_verifies(input);
    }
}

#[test]
fn integrate_contract_reciprocal_linear_log_table_explains_internal_derivative_without_substitution(
) {
    for (input, expected_result, expected_required_display, expect_constant_adjustment) in [
        (
            "integrate(1/(x+1), x)",
            "ln(|x + 1|)",
            serde_json::json!(["x ≠ -1"]),
            false,
        ),
        (
            "integrate(1/(2*x+1), x)",
            "1/2·ln(|2·x + 1|)",
            serde_json::json!(["x ≠ -1/2"]),
            true,
        ),
        (
            "integrate(1/(1-2*x), x)",
            "-1/2·ln(|1 - 2·x|)",
            serde_json::json!(["x ≠ 1/2"]),
            true,
        ),
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

        assert_eq!(wire["result"], expected_result, "input: {input}");
        assert_eq!(
            wire["required_display"], expected_required_display,
            "unexpected required_display for {input}: {:?}",
            wire["required_display"]
        );
        assert!(
            !stderr.contains("depth_overflow"),
            "linear log table trace should not emit depth_overflow warning for {input}\nstderr:\n{stderr}"
        );

        let steps = wire["steps"]
            .as_array()
            .expect("steps should be present with --steps on");
        let integration_step = steps
            .iter()
            .find(|step| step["rule"] == "Calcular la integral")
            .expect("expected public symbolic integration step");
        let substeps = integration_step["substeps"]
            .as_array()
            .expect("integration step should expose didactic substeps");
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == "Usar la regla de ln|u| con derivada interna"),
            "expected ln|u| substep for {input}, got {substeps:?}"
        );
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == "Identificar el denominador afín"),
            "expected affine denominator substep for {input}, got {substeps:?}"
        );
        assert_eq!(
            substeps
                .iter()
                .any(|substep| substep["title"] == "Ajustar el factor constante"),
            expect_constant_adjustment,
            "unexpected constant adjustment substep presence for {input}: {substeps:?}"
        );
        assert!(
            substeps
                .iter()
                .all(|substep| substep["title"] != "Usar sustitución"),
            "linear log table case should not use the polynomial-substitution substep for {input}: {substeps:?}"
        );
        assert_antiderivative_verifies(input);
    }
}

#[test]
fn integrate_contract_linear_hyperbolic_coth_derivative_square_substitution() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(1/sinh(2*x + 1)^2, x)");

    assert_eq!(result, "-1 / (2 * tanh(2 * x + 1))");
    assert_eq!(
        required,
        vec!["sinh(2 * x + 1) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}

#[test]
fn integrate_contract_polynomial_hyperbolic_coth_derivative_square_substitution() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(2*x/sinh(x^2)^2, x)");

    assert_eq!(result, "-1 / tanh(x^2)");
    assert_eq!(
        required,
        vec!["sinh(x^2) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}

#[test]
fn integrate_contract_linear_hyperbolic_cosh_reciprocal_derivative_substitution() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(sinh(2*x + 1)/cosh(2*x + 1)^2, x)");

    assert_eq!(result, "-1 / (2 * cosh(2 * x + 1))");
    assert_eq!(
        required,
        Vec::<String>::new(),
        "unexpected required_conditions: {required:?}"
    );
}

#[test]
fn integrate_contract_polynomial_hyperbolic_cosh_reciprocal_derivative_substitution() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(2*x*sinh(x^2)/cosh(x^2)^2, x)");

    assert_eq!(result, "-1 / cosh(x^2)");
    assert_eq!(
        required,
        Vec::<String>::new(),
        "unexpected required_conditions: {required:?}"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(-x*sinh(x^2)/cosh(x^2)^2, x)");

    assert_eq!(result, "1 / (2 * cosh(x^2))");
    assert_eq!(
        required,
        Vec::<String>::new(),
        "unexpected required_conditions: {required:?}"
    );
}

#[test]
fn integrate_contract_linear_hyperbolic_sinh_reciprocal_derivative_substitution() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(cosh(2*x + 1)/sinh(2*x + 1)^2, x)");

    assert_eq!(result, "-1 / (2 * sinh(2 * x + 1))");
    assert_eq!(
        required,
        vec!["sinh(2 * x + 1) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}

#[test]
fn integrate_contract_polynomial_hyperbolic_sinh_reciprocal_derivative_substitution() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(2*x*cosh(x^2)/sinh(x^2)^2, x)");

    assert_eq!(result, "-1 / sinh(x^2)");
    assert_eq!(
        required,
        vec!["sinh(x^2) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(-x*cosh(x^2)/sinh(x^2)^2, x)");

    assert_eq!(result, "1 / (2 * sinh(x^2))");
    assert_eq!(
        required,
        vec!["sinh(x^2) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}

#[test]
fn integrate_contract_negative_scaled_hyperbolic_reciprocal_primitives_keep_domain_signal() {
    let cases = [
        (
            "integrate(-2*x*sinh(x^2)/cosh(x^2)^2, x)",
            "1 / cosh(x^2)",
            "\\frac{1}{\\cosh({x}^{2})}",
            serde_json::json!([]),
            "diff(integrate(-2*x*sinh(x^2)/cosh(x^2)^2, x), x) + 2*x*sinh(x^2)/cosh(x^2)^2",
        ),
        (
            "integrate(-2*x*cosh(x^2)/sinh(x^2)^2, x)",
            "1 / sinh(x^2)",
            "\\frac{1}{\\sinh({x}^{2})}",
            serde_json::json!(["sinh(x^2) ≠ 0"]),
            "diff(integrate(-2*x*cosh(x^2)/sinh(x^2)^2, x), x) + 2*x*cosh(x^2)/sinh(x^2)^2",
        ),
    ];

    for (input, expected_result, expected_latex, expected_required, residual) in cases {
        let (wire, stderr) = cli_eval_json_with_stderr(input);
        assert!(
            stderr.is_empty(),
            "unexpected stderr for negative scaled hyperbolic reciprocal primitive: {stderr}"
        );
        assert_eq!(wire["result"], expected_result);
        assert_eq!(wire["result_latex"], expected_latex);
        assert_eq!(wire["required_display"], expected_required);

        let (residual_wire, residual_stderr) = cli_eval_json_with_stderr(residual);
        assert!(
            residual_stderr.is_empty(),
            "unexpected stderr for negative scaled hyperbolic reciprocal residual: {residual_stderr}"
        );
        assert_eq!(residual_wire["result"], "0");
        assert_eq!(residual_wire["required_display"], expected_required);
    }
}

#[test]
fn integrate_contract_affine_sum_linearity() {
    assert_eq!(simplified_integral("integrate(2*x + 3, x)"), "x^2 + 3 * x");
}

#[test]
fn integrate_contract_direct_cos_table() {
    assert_eq!(simplified_integral("integrate(cos(x), x)"), "sin(x)");
}

#[test]
fn integrate_contract_linear_log_table_preserves_positive_domain() {
    let (result, required) = evaluated_integral_with_required_conditions("integrate(ln(x), x)");

    assert_eq!(result, "x * ln(x) - x");
    assert_eq!(
        required,
        vec!["x > 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_rendered_antiderivative_verifies("integrate(ln(x), x)", &result);

    let (result, required) = evaluated_integral_with_required_conditions("integrate(ln(2*x+1), x)");

    assert_eq!(result, "1/2 * (2 * x + 1) * (ln(2 * x + 1) - 1)");
    assert_eq!(
        required,
        vec!["x > -1/2".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_antiderivative_verifies("integrate(ln(2*x+1), x)");
    assert_rendered_antiderivative_verifies("integrate(ln(2*x+1), x)", &result);
}

#[test]
fn integrate_contract_log_reciprocal_derivative_preserves_real_domain() {
    let cases = [
        (
            "integrate(1/(x*ln(x)), x)",
            "ln(|ln(x)|)",
            "diff(integrate(1/(x*ln(x)), x), x) - 1/(x*ln(x))",
            vec!["x > 0".to_string(), "x ≠ 1".to_string()],
        ),
        (
            "integrate(2/((2*x+1)*ln(2*x+1)), x)",
            "ln(|ln(2 * x + 1)|)",
            "diff(integrate(2/((2*x+1)*ln(2*x+1)), x), x) - 2/((2*x+1)*ln(2*x+1))",
            vec!["x > -1/2".to_string(), "x ≠ 0".to_string()],
        ),
        (
            "integrate(2*x/((x^2+1)*ln(x^2+1)), x)",
            "ln(|ln(x^2 + 1)|)",
            "diff(integrate(2*x/((x^2+1)*ln(x^2+1)), x), x) - 2*x/((x^2+1)*ln(x^2+1))",
            vec!["x ≠ 0".to_string()],
        ),
        (
            "integrate((2*x+1)/((x^2+x+1)*ln(x^2+x+1)), x)",
            "ln(|ln(x^2 + x + 1)|)",
            "diff(integrate((2*x+1)/((x^2+x+1)*ln(x^2+x+1)), x), x) - (2*x+1)/((x^2+x+1)*ln(x^2+x+1))",
            vec!["x ≠ -1".to_string(), "x ≠ 0".to_string()],
        ),
        (
            "integrate(2*x/((x^2-1)*ln(x^2-1)), x)",
            "ln(|ln(x^2 - 1)|)",
            "diff(integrate(2*x/((x^2-1)*ln(x^2-1)), x), x) - 2*x/((x^2-1)*ln(x^2-1))",
            vec!["x < -1 or x > 1".to_string(), "x^2 - 2 ≠ 0".to_string()],
        ),
        (
            "integrate(2*x/((x^2+1)*ln(x^2+1)^2), x)",
            "-1 / ln(x^2 + 1)",
            "diff(integrate(2*x/((x^2+1)*ln(x^2+1)^2), x), x) - 2*x/((x^2+1)*ln(x^2+1)^2)",
            vec!["x ≠ 0".to_string()],
        ),
        (
            "integrate(2*x/((x^2-1)*ln(x^2-1)^2), x)",
            "-1 / ln(x^2 - 1)",
            "diff(integrate(2*x/((x^2-1)*ln(x^2-1)^2), x), x) - 2*x/((x^2-1)*ln(x^2-1)^2)",
            vec!["x < -1 or x > 1".to_string(), "x^2 - 2 ≠ 0".to_string()],
        ),
        (
            "integrate((2*x+1)/((x^2+x-1)*ln(x^2+x-1)^2), x)",
            "-1 / ln(x^2 + x - 1)",
            "diff(integrate((2*x+1)/((x^2+x-1)*ln(x^2+x-1)^2), x), x) - (2*x+1)/((x^2+x-1)*ln(x^2+x-1)^2)",
            vec![
                "x < -1/2 - sqrt(5)/2 or x > -1/2 + sqrt(5)/2".to_string(),
                "x ≠ -2".to_string(),
                "x ≠ 1".to_string(),
            ],
        ),
        (
            "integrate((2*x+1)/((x^2+x-1)*ln(x^2+x-1)^3), x)",
            "-1 / (2 * ln(x^2 + x - 1)^2)",
            "diff(integrate((2*x+1)/((x^2+x-1)*ln(x^2+x-1)^3), x), x) - (2*x+1)/((x^2+x-1)*ln(x^2+x-1)^3)",
            vec![
                "x < -1/2 - sqrt(5)/2 or x > -1/2 + sqrt(5)/2".to_string(),
                "x ≠ -2".to_string(),
                "x ≠ 1".to_string(),
            ],
        ),
    ];

    for (input, expected, residual, expected_required) in cases {
        let (result, mut required) = evaluated_integral_with_required_conditions(input);
        required.sort();
        assert_eq!(result, expected, "input: {input}");
        assert_eq!(required, expected_required, "input: {input}");
        assert_rendered_antiderivative_verifies(input, &result);

        let (residual_result, mut residual_required) =
            evaluated_expr_with_required_conditions(residual);
        residual_required.sort();
        assert_eq!(residual_result, "0", "input: {input}");
        assert_eq!(residual_required, expected_required, "input: {input}");
    }
}

#[test]
fn integrate_contract_log_reciprocal_derivative_preserves_compact_prep_trace() {
    let input = "integrate(2*x/((x^2+1)*ln(x^2+1)), x)";
    let step_rules = evaluated_integral_step_rules(input);

    assert_eq!(
        step_rules,
        vec![
            "Pull Constant From Fraction".to_string(),
            "Symbolic Integration".to_string(),
        ],
        "log-reciprocal substitution should not expand and refactor its compact denominator: {step_rules:?}"
    );
}

#[test]
fn integrate_contract_monomial_times_log_by_parts_preserves_positive_domain() {
    let cases = [
        ("integrate(x*ln(x), x)", "1/4 * x^2 * (2 * ln(x) - 1)"),
        ("integrate(x^2*ln(x), x)", "1/9 * x^3 * (3 * ln(x) - 1)"),
        (
            "integrate(x*ln(x)^2, x)",
            "1/8 * x^2 * (4 * ln(x)^2 - 4 * ln(x) + 2)",
        ),
        (
            "integrate(x^2*ln(x)^2, x)",
            "1/27 * x^3 * (9 * ln(x)^2 - 6 * ln(x) + 2)",
        ),
        (
            "integrate(x*ln(x)^3, x)",
            "1/16 * x^2 * (8 * ln(x)^3 - 12 * ln(x)^2 + 12 * ln(x) - 6)",
        ),
        (
            "integrate(x^2*ln(x)^3, x)",
            "1/81 * x^3 * (27 * ln(x)^3 - 27 * ln(x)^2 + 18 * ln(x) - 6)",
        ),
        (
            "integrate(x*ln(x)^4, x)",
            "1/32 * x^2 * (16 * ln(x)^4 - 32 * ln(x)^3 + 48 * ln(x)^2 - 48 * ln(x) + 24)",
        ),
        (
            "integrate(x^2*ln(x)^4, x)",
            "1/243 * x^3 * (81 * ln(x)^4 - 108 * ln(x)^3 + 108 * ln(x)^2 - 72 * ln(x) + 24)",
        ),
        (
            "integrate(x*ln(x)^5, x)",
            "1/64 * x^2 * (32 * ln(x)^5 - 80 * ln(x)^4 + 160 * ln(x)^3 - 240 * ln(x)^2 + 240 * ln(x) - 120)",
        ),
        (
            "integrate(x^2*ln(x)^5, x)",
            "1/729 * x^3 * (243 * ln(x)^5 - 405 * ln(x)^4 + 540 * ln(x)^3 - 540 * ln(x)^2 + 360 * ln(x) - 120)",
        ),
    ];

    for (input, expected) in cases {
        let (result, required) = evaluated_integral_with_required_conditions(input);

        assert_eq!(result, expected, "input: {input}");
        assert_eq!(
            required,
            vec!["x > 0".to_string()],
            "unexpected required_conditions for {input}: {required:?}"
        );
        assert_antiderivative_verifies(input);
        assert_rendered_antiderivative_verifies(input, &result);
    }
}

#[test]
fn integrate_contract_log_by_parts_exposes_didactic_substep_and_keeps_compact_trace() {
    for (input, expected_result, expected_required) in [
        ("integrate(x*ln(x), x)", "1/4·x^2·(2·ln(x) - 1)", "x > 0"),
        (
            "integrate((2*x+1)*ln(2*x+1), x)",
            "1/4·((2·x + 1)^2·ln(2·x + 1) - 2·x^2 - 2·x)",
            "x > -1/2",
        ),
        (
            "integrate((x^2+x+1)*ln(2*x+1), x)",
            "(1/3·x^3 + 1/2·x^2 + x)·ln(2·x + 1) - 1/9·x^3 - 1/6·x^2 - 5/6·x + 5/12·ln(2·x + 1)",
            "x > -1/2",
        ),
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);
        assert!(
            stderr.is_empty(),
            "log by-parts presentation should stay quiet for {input}\nstderr:\n{stderr}"
        );
        assert_eq!(wire["result"], expected_result, "input: {input}");
        assert_eq!(
            wire["required_display"],
            serde_json::json!([expected_required]),
            "unexpected required_display for {input}: {:?}",
            wire["required_display"]
        );

        let steps = wire["steps"]
            .as_array()
            .expect("steps should be present with --steps on");
        assert_eq!(
            steps.len(),
            1,
            "expected compact direct integration trace for {input}, got {steps:?}"
        );
        assert!(
            steps
                .iter()
                .all(|step| step["rule"] != "Expandir la expresión"),
            "log by-parts trace should not expand before integrating for {input}, got {steps:?}"
        );
        let integration_step = steps
            .iter()
            .find(|step| step["rule"] == "Calcular la integral")
            .expect("expected public symbolic integration step");
        let substeps = integration_step["substeps"]
            .as_array()
            .expect("integration step should expose didactic substeps");
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == "Usar integración por partes"),
            "expected integration-by-parts substep for {input}, got {substeps:?}"
        );
        for title in [
            "Elegir u y dv",
            "Calcular du y v",
            "Aplicar la fórmula de integración por partes",
        ] {
            assert!(
                substeps.iter().any(|substep| substep["title"] == title),
                "expected concrete by-parts substep {title:?} for {input}, got {substeps:?}"
            );
        }

        let choice_substep = substeps
            .iter()
            .find(|substep| substep["title"] == "Elegir u y dv")
            .expect("expected u/dv choice substep");
        let choice_latex = choice_substep["after_latex"]
            .as_str()
            .expect("choice substep should expose after_latex");
        assert!(
            choice_latex.contains("\\ln") && choice_latex.contains("dv"),
            "choice substep should show concrete u and dv, got {choice_latex:?}"
        );

        let derivative_substep = substeps
            .iter()
            .find(|substep| substep["title"] == "Calcular du y v")
            .expect("expected du/v calculation substep");
        let derivative_latex = derivative_substep["after_latex"]
            .as_str()
            .expect("du/v substep should expose after_latex");
        assert!(
            derivative_latex.contains("du =") && derivative_latex.contains("v ="),
            "du/v substep should show concrete du and v, got {derivative_latex:?}"
        );
        if input == "integrate(x*ln(x), x)" {
            assert!(
                derivative_latex.contains("\\frac{1}{x}") && derivative_latex.contains("{x}^{2}"),
                "du/v substep should show concrete du and v, got {derivative_latex:?}"
            );
        } else {
            assert!(
                derivative_latex.contains("\\frac{2}{2\\cdot x + 1}"),
                "affine log by-parts substep should show concrete du, got {derivative_latex:?}"
            );
        }
    }
}

#[test]
fn integrate_contract_positive_quadratic_log_by_parts_keeps_compact_trace() {
    let input = "integrate((x^2+1)*ln(x^2+1), x)";
    let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);
    assert!(
        stderr.is_empty(),
        "positive quadratic log by-parts presentation should stay quiet\nstderr:\n{stderr}"
    );
    assert_eq!(
        wire["result"],
        "(1/3·x^3 + x)·ln(x^2 + 1) - 2/9·x^3 - 4/3·x + 4/3·arctan(x)"
    );
    assert_eq!(
        wire["required_display"],
        serde_json::json!([]),
        "positive quadratic log by-parts should not add domain requirements"
    );

    let steps = wire["steps"]
        .as_array()
        .expect("steps should be present with --steps on");
    assert_eq!(
        steps.len(),
        1,
        "expected compact direct integration trace, got {steps:?}"
    );
    assert!(
        steps
            .iter()
            .all(|step| step["rule"] != "Expandir la expresión"),
        "positive quadratic log by-parts trace should not expand before integrating, got {steps:?}"
    );
    let integration_step = steps
        .iter()
        .find(|step| step["rule"] == "Calcular la integral")
        .expect("expected public symbolic integration step");
    let substeps = integration_step["substeps"]
        .as_array()
        .expect("integration step should expose didactic substeps");
    assert!(
        substeps
            .iter()
            .any(|substep| substep["title"] == "Usar integración por partes"),
        "expected integration-by-parts substep, got {substeps:?}"
    );

    assert_eq!(
        assert_antiderivative_verifies(input),
        AntiderivativeVerificationRoute::PublicResidual
    );
}

#[test]
fn integrate_contract_positive_quadratic_log_by_parts_collects_repeated_log_factor() {
    let input = "integrate((x^2+x+1)*ln(x^2+1), x)";
    let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);
    assert!(
        stderr.is_empty(),
        "positive quadratic log by-parts presentation should stay quiet\nstderr:\n{stderr}"
    );
    assert_eq!(
        wire["result"],
        "ln(x^2 + 1)·(1/3·x^3 + 1/2·x^2 + x + 1/2) + 4/3·arctan(x) - 2/9·x^3 - 1/2·x^2 - 4/3·x"
    );
    assert_eq!(wire["required_display"], serde_json::json!([]));

    let steps = wire["steps"]
        .as_array()
        .expect("steps should be present with --steps on");
    assert_eq!(
        steps.len(),
        1,
        "expected compact direct integration trace, got {steps:?}"
    );
    assert!(
        steps
            .iter()
            .all(|step| step["rule"] != "Expandir la expresión"),
        "positive quadratic log by-parts trace should not expand before integrating, got {steps:?}"
    );
    let integration_step = steps
        .iter()
        .find(|step| step["rule"] == "Calcular la integral")
        .expect("expected public symbolic integration step");
    let substeps = integration_step["substeps"]
        .as_array()
        .expect("integration step should expose didactic substeps");
    assert!(
        substeps
            .iter()
            .any(|substep| substep["title"] == "Usar integración por partes"),
        "expected integration-by-parts substep, got {substeps:?}"
    );

    assert_eq!(
        assert_antiderivative_verifies(input),
        AntiderivativeVerificationRoute::PublicResidual
    );
    let residual = "diff(integrate((x^2+x+1)*ln(x^2+1), x), x) - (x^2+x+1)*ln(x^2+1)";
    let (residual_wire, residual_stderr) = cli_eval_json_with_stderr(residual);
    assert!(
        residual_stderr.is_empty(),
        "positive quadratic log by-parts residual should stay quiet\nstderr:\n{residual_stderr}"
    );
    assert_eq!(residual_wire["result"], "0");
    assert_eq!(residual_wire["required_display"], serde_json::json!([]));
}

#[test]
fn integrate_contract_linear_monomial_times_affine_log_by_parts_preserves_positive_domain() {
    let cases = [
        (
            "integrate(x*ln(2*x+1), x)",
            "(x^2 / 2 - 1/8) * ln(2 * x + 1) + 1/4 * x - 1/4 * x^2",
            "x > -1/2",
        ),
        (
            "integrate(3*x*ln(2*x+1), x)",
            "3 * ((x^2 / 2 - 1/8) * ln(2 * x + 1) + 1/4 * x - 1/4 * x^2)",
            "x > -1/2",
        ),
        (
            "integrate(x*ln(x+1), x)",
            "(x^2 / 2 - 1/2) * ln(x + 1) + 1/2 * x - 1/4 * x^2",
            "x > -1",
        ),
        (
            "integrate((x+1)*ln(2*x+1), x)",
            "1/8 * (ln(2 * x + 1) * (2 * x + 1) * (2 * x + 3) - 2 * x^2 - 6 * x)",
            "x > -1/2",
        ),
        (
            "integrate((2*x+3)*ln(2*x+1), x)",
            "1/4 * (ln(2 * x + 1) * (2 * x + 1) * (2 * x + 5) - 2 * x^2 - 10 * x)",
            "x > -1/2",
        ),
        (
            "integrate((1-2*x)*ln(1-2*x), x)",
            "1/4 * (2 * x^2 - ln(1 - 2 * x) * (1 - 2 * x)^2 - 2 * x)",
            "x < 1/2",
        ),
    ];

    for (input, expected, required_condition) in cases {
        let (result, required) = evaluated_integral_with_required_conditions(input);

        assert_eq!(result, expected, "input: {input}");
        assert_eq!(
            required,
            vec![required_condition.to_string()],
            "unexpected required_conditions for {input}: {required:?}"
        );
        assert_antiderivative_verifies(input);
        assert_rendered_antiderivative_verifies(input, &result);
    }
}

#[test]
fn integrate_contract_quadratic_monomial_times_affine_log_by_parts_verifies() {
    let input = "integrate(x^2*ln(2*x+1), x)";
    let (wire, stderr) = cli_eval_json_with_stderr(input);

    assert!(
        stderr.is_empty(),
        "quadratic affine-log by-parts integration should stay quiet\nstderr:\n{stderr}"
    );
    assert_eq!(
        wire["result"],
        "1/3·x^3·ln(2·x + 1) - 1/9·x^3 - 1/12·x + 1/24·ln(2·x + 1) + 1/12·x^2"
    );
    assert_eq!(wire["required_display"], serde_json::json!(["x > -1/2"]));

    let residual = "diff(integrate(x^2*ln(2*x+1), x), x) - x^2*ln(2*x+1)";
    let (residual_wire, residual_stderr) = cli_eval_json_with_stderr(residual);

    assert!(
        residual_stderr.is_empty(),
        "quadratic affine-log by-parts residual should stay quiet\nstderr:\n{residual_stderr}"
    );
    assert_eq!(residual_wire["result"], "0");
    assert_eq!(
        residual_wire["required_display"],
        serde_json::json!(["x > -1/2"])
    );

    let negative_input = "integrate(x^2*ln(1-2*x), x)";
    let (negative_wire, negative_stderr) = cli_eval_json_with_stderr_args(
        negative_input,
        &["--budget", "small", "--time-budget-ms", "100"],
    );

    assert!(
        negative_stderr.is_empty(),
        "negative-slope quadratic affine-log integration should stay quiet\nstderr:\n{negative_stderr}"
    );
    assert_eq!(
        negative_wire["result"],
        "1/3·x^3·ln(1 - 2·x) - (1/24·ln(1 - 2·x) + 1/9·x^3 + 1/12·x^2 + 1/12·x)"
    );
    assert_eq!(
        negative_wire["required_display"],
        serde_json::json!(["x < 1/2"])
    );

    let negative_residual = "diff(integrate(x^2*ln(1-2*x), x), x) - x^2*ln(1-2*x)";
    let (negative_residual_wire, negative_residual_stderr) = cli_eval_json_with_stderr_args(
        negative_residual,
        &["--budget", "small", "--time-budget-ms", "100"],
    );

    assert!(
        negative_residual_stderr.is_empty(),
        "negative-slope quadratic affine-log residual should stay quiet\nstderr:\n{negative_residual_stderr}"
    );
    assert_eq!(negative_residual_wire["result"], "0");
    assert_eq!(
        negative_residual_wire["required_display"],
        serde_json::json!(["x < 1/2"])
    );
}

#[test]
fn integrate_contract_cubic_monomial_times_affine_log_by_parts_verifies() {
    let input = "integrate(x^3*ln(2*x+1), x)";
    let (wire, stderr) = cli_eval_json_with_stderr(input);

    assert!(
        stderr.is_empty(),
        "cubic affine-log by-parts integration should stay quiet\nstderr:\n{stderr}"
    );
    assert_eq!(
        wire["result"],
        "ln(2·x + 1)·(1/4·x^4 - 1/64) + 1/24·x^3 + 1/32·x - 1/16·x^4 - 1/32·x^2"
    );
    assert_eq!(wire["required_display"], serde_json::json!(["x > -1/2"]));
    assert!(wire.get("blocked_hints").is_none());

    let residual = "diff(integrate(x^3*ln(2*x+1), x), x) - x^3*ln(2*x+1)";
    let (residual_wire, residual_stderr) = cli_eval_json_with_stderr(residual);

    assert!(
        residual_stderr.is_empty(),
        "cubic affine-log by-parts residual should stay quiet\nstderr:\n{residual_stderr}"
    );
    assert_eq!(residual_wire["result"], "0");
    assert_eq!(
        residual_wire["required_display"],
        serde_json::json!(["x > -1/2"])
    );
}

#[test]
fn integrate_contract_sparse_cubic_affine_log_by_parts_stays_compact() {
    let input = "integrate((x^3+x)*ln(x+1), x)";
    let (wire, stderr) = cli_eval_json_with_stderr(input);

    assert!(
        stderr.is_empty(),
        "sparse cubic affine-log by-parts integration should stay quiet\nstderr:\n{stderr}"
    );
    assert_eq!(
        wire["result"],
        "ln(x + 1)·(1/4·x^4 + 1/2·x^2 - 3/4) + 1/12·x^3 + 3/4·x - 1/16·x^4 - 3/8·x^2"
    );
    assert_eq!(wire["required_display"], serde_json::json!(["x > -1"]));
    assert!(wire.get("blocked_hints").is_none());

    let residual = "diff(integrate((x^3+x)*ln(x+1), x), x) - (x^3+x)*ln(x+1)";
    let (residual_wire, residual_stderr) = cli_eval_json_with_stderr(residual);

    assert!(
        residual_stderr.is_empty(),
        "sparse cubic affine-log by-parts residual should stay quiet\nstderr:\n{residual_stderr}"
    );
    assert_eq!(residual_wire["result"], "0");
    assert_eq!(
        residual_wire["required_display"],
        serde_json::json!(["x > -1"])
    );
}

#[test]
fn integrate_contract_quadratic_times_positive_quadratic_log_by_parts_verifies_shifted_argument() {
    let cases = [
        (
            "integrate(ln(x^2+x+1), x)",
            "1/2·ln(x^2 + x + 1) + ln(x^2 + x + 1)·x + 3·arctan((2·x + 1) / sqrt(3)) / sqrt(3) - 2·x",
            "diff(integrate(ln(x^2+x+1), x), x) - ln(x^2+x+1)",
            true,
        ),
        (
            "integrate(x*ln(x^2+x+1), x)",
            "1/4·ln(x^2 + x + 1) + 1/2·ln(x^2 + x + 1)·x^2 - 3/2·arctan((2·x + 1) / sqrt(3)) / sqrt(3) + 1/2·x - 1/2·x^2",
            "diff(integrate(x*ln(x^2+x+1), x), x) - x*ln(x^2+x+1)",
            true,
        ),
        (
            "integrate(x^2*ln(x^2+x+1), x)",
            "1/3·x^3·ln(x^2 + x + 1) - 1/3·ln(x^2 + x + 1) - 2/9·x^3 + 1/6·x^2 + 1/3·x",
            "diff(integrate(x^2*ln(x^2+x+1), x), x) - x^2*ln(x^2+x+1)",
            true,
        ),
        (
            "integrate(x^3*ln(x^2+1), x)",
            "1/4·x^4·ln(x^2 + 1) - 1/4·ln(x^2 + 1) - 1/8·x^4 + 1/4·x^2",
            "diff(integrate(x^3*ln(x^2+1), x), x) - x^3*ln(x^2+1)",
            true,
        ),
        (
            "integrate(x^4*ln(x^2+1), x)",
            "1/5·x^5·ln(x^2 + 1) - 2/25·x^5 - 2/5·x + 2/5·arctan(x) + 2/15·x^3",
            "diff(integrate(x^4*ln(x^2+1), x), x) - x^4*ln(x^2+1)",
            true,
        ),
        (
            "integrate(x^5*ln(x^2+1), x)",
            "1/6·x^6·ln(x^2 + 1) - 1/18·x^6 - 1/6·x^2 + 1/6·ln(x^2 + 1) + 1/12·x^4",
            "diff(integrate(x^5*ln(x^2+1), x), x) - x^5*ln(x^2+1)",
            true,
        ),
        (
            "integrate(x^6*ln(x^2+1), x)",
            "1/7·x^7·ln(x^2 + 1) - 2/7·arctan(x) - 2/49·x^7 - 2/21·x^3 + 2/35·x^5 + 2/7·x",
            "diff(integrate(x^6*ln(x^2+1), x), x) - x^6*ln(x^2+1)",
            true,
        ),
        (
            "integrate(x^7*ln(x^2+1), x)",
            "1/8·x^8·ln(x^2 + 1) - 1/8·ln(x^2 + 1) - 1/32·x^8 - 1/16·x^4 + 1/24·x^6 + 1/8·x^2",
            "diff(integrate(x^7*ln(x^2+1), x), x) - x^7*ln(x^2+1)",
            true,
        ),
        (
            "integrate(x^8*ln(x^2+1), x)",
            "1/9·x^9·ln(x^2 + 1) - 2/81·x^9 - 2/45·x^5 - 2/9·x + 2/9·arctan(x) + 2/63·x^7 + 2/27·x^3",
            "diff(integrate(x^8*ln(x^2+1), x), x) - x^8*ln(x^2+1)",
            true,
        ),
    ];

    for (input, expected, residual, verify_rendered) in cases {
        let (wire, stderr) = cli_eval_json_with_stderr(input);

        assert!(
            stderr.is_empty(),
            "positive-quadratic log by-parts integration should stay quiet for {input}\nstderr:\n{stderr}"
        );
        assert_eq!(wire["result"], expected, "input: {input}");
        assert_eq!(wire["required_display"], serde_json::json!([]));
        assert_antiderivative_verifies(input);
        if verify_rendered {
            assert_rendered_antiderivative_verifies(input, expected);
        }

        let (residual_wire, residual_stderr) = cli_eval_json_with_stderr(residual);

        assert!(
            residual_stderr.is_empty(),
            "positive-quadratic log by-parts residual should stay quiet for {input}\nstderr:\n{residual_stderr}"
        );
        assert_eq!(residual_wire["result"], "0", "input: {input}");
        assert_eq!(residual_wire["required_display"], serde_json::json!([]));
    }

    let explicit_quintic_residual = "diff(1/6*x^6*ln(x^2+1) - 1/18*x^6 - 1/6*x^2 + 1/6*ln(x^2+1) + 1/12*x^4, x) - x^5*ln(x^2+1)";
    let (wire, stderr) = cli_eval_json_with_stderr(explicit_quintic_residual);
    assert!(
        stderr.is_empty(),
        "positive-quadratic log rendered quintic residual should stay quiet\nstderr:\n{stderr}"
    );
    assert_eq!(wire["result"], "0");
    assert_eq!(wire["required_display"], serde_json::json!([]));

    let explicit_sextic_residual = "diff(1/7*x^7*ln(x^2+1) - 2/7*arctan(x) - 2/49*x^7 - 2/21*x^3 + 2/35*x^5 + 2/7*x, x) - x^6*ln(x^2+1)";
    let (wire, stderr) = cli_eval_json_with_stderr(explicit_sextic_residual);
    assert!(
        stderr.is_empty(),
        "positive-quadratic log rendered sextic residual should stay quiet\nstderr:\n{stderr}"
    );
    assert_eq!(wire["result"], "0");
    assert_eq!(wire["required_display"], serde_json::json!([]));

    let explicit_septic_residual = "diff(1/8*x^8*ln(x^2+1) - 1/8*ln(x^2+1) - 1/32*x^8 - 1/16*x^4 + 1/24*x^6 + 1/8*x^2, x) - x^7*ln(x^2+1)";
    let (wire, stderr) = cli_eval_json_with_stderr(explicit_septic_residual);
    assert!(
        stderr.is_empty(),
        "positive-quadratic log rendered septic residual should stay quiet\nstderr:\n{stderr}"
    );
    assert_eq!(wire["result"], "0");
    assert_eq!(wire["required_display"], serde_json::json!([]));

    let explicit_octic_residual = "diff(1/9*x^9*ln(x^2+1) - 2/81*x^9 - 2/45*x^5 - 2/9*x + 2/9*arctan(x) + 2/63*x^7 + 2/27*x^3, x) - x^8*ln(x^2+1)";
    let (wire, stderr) = cli_eval_json_with_stderr(explicit_octic_residual);
    assert!(
        stderr.is_empty(),
        "positive-quadratic log rendered octic residual should stay quiet\nstderr:\n{stderr}"
    );
    assert_eq!(wire["result"], "0");
    assert_eq!(wire["required_display"], serde_json::json!([]));

    let (wire, stderr) = cli_eval_json_with_stderr("integrate(x^9*ln(x^2+1), x)");
    assert!(
        stderr.is_empty(),
        "positive-quadratic log by-parts budget boundary should stay quiet\nstderr:\n{stderr}"
    );
    assert_eq!(wire["result"], "integrate(ln(x^2 + 1)·x^9, x)");
    assert_eq!(wire["required_display"], serde_json::json!([]));
}

#[test]
fn integrate_contract_positive_quadratic_log_by_parts_flattens_compound_remainder() {
    let input = "integrate((x+1)*ln(x^2+x+1), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(
        result,
        "ln(x^2 + x + 1) * (1/2 * x^2 + x + 3/4) + 3/2 * arctan((2 * x + 1) / sqrt(3)) / sqrt(3) - 1/2 * x^2 - 3/2 * x"
    );
    assert!(
        !result.contains(" - ("),
        "compound positive-quadratic log by-parts presentation should flatten subtracting a remainder group, got {result}"
    );
    assert!(
        required.is_empty(),
        "positive quadratic log argument should not add conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
    assert_rendered_antiderivative_verifies(input, &result);

    let residual = "diff(integrate((x+1)*ln(x^2+x+1), x), x) - (x+1)*ln(x^2+x+1)";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert!(
        residual_required.is_empty(),
        "positive-quadratic log by-parts residual should not add conditions: {residual_required:?}"
    );
}

#[test]
fn integrate_contract_positive_quadratic_log_by_parts_recombines_expanded_orientation() {
    let input = "integrate((x+1)*ln(x^2-x+1), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(
        result,
        "ln(x^2 + 1 - x) * (1/2 * x^2 + x - 1/4) + 9/2 * arctan((2 * x - 1) / sqrt(3)) / sqrt(3) - 1/2 * x^2 - 5/2 * x"
    );
    assert!(
        !result.contains(" - (") && !result.contains("1/2 * ("),
        "expanded-orientation positive-quadratic log by-parts presentation should stay flat, got {result}"
    );
    assert!(
        required.is_empty(),
        "positive quadratic log argument should not add conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);

    let residual = "diff(integrate((x+1)*ln(x^2-x+1), x), x) - (x+1)*ln(x^2-x+1)";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert!(
        residual_required.is_empty(),
        "expanded-orientation positive-quadratic log by-parts residual should not add conditions: {residual_required:?}"
    );
}

#[test]
fn integrate_contract_positive_quadratic_self_log_by_parts_flattens_remainder() {
    let input = "integrate((x^2+x+1)*ln(x^2+x+1), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(
        result,
        "ln(x^2 + x + 1) * (1/3 * x^3 + 1/2 * x^2 + x + 5/12) + 3/2 * arctan((2 * x + 1) / sqrt(3)) / sqrt(3) - 2/9 * x^3 - 1/3 * x^2 - 7/6 * x"
    );
    assert!(
        !result.contains(" - ("),
        "self positive-quadratic log by-parts presentation should flatten subtracting a remainder group, got {result}"
    );
    assert!(
        required.is_empty(),
        "positive quadratic self-log argument should not add conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
    assert_rendered_antiderivative_verifies(input, &result);

    let residual = "diff(integrate((x^2+x+1)*ln(x^2+x+1), x), x) - (x^2+x+1)*ln(x^2+x+1)";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert!(
        residual_required.is_empty(),
        "positive-quadratic self-log by-parts residual should not add conditions: {residual_required:?}"
    );
}

#[test]
fn integrate_contract_cubic_times_shifted_positive_quadratic_log_flattens_remainder() {
    let input = "integrate((x^3+x^2+x+1)*ln(x^2+x+1), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(
        result,
        "ln(x^2 + x + 1) * (1/4 * x^4 + 1/3 * x^3 + 1/2 * x^2 + x + 13/24) + 9/4 * arctan((2 * x + 1) / sqrt(3)) / sqrt(3) - 1/8 * x^4 - 5/36 * x^3 - 5/24 * x^2 - 5/3 * x"
    );
    assert!(
        !result.contains(" - ("),
        "cubic shifted positive-quadratic log by-parts presentation should flatten subtracting a remainder group, got {result}"
    );
    assert!(
        required.is_empty(),
        "shifted positive quadratic log argument should not add conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
    assert_rendered_antiderivative_verifies(input, &result);

    let residual = "diff(integrate((x^3+x^2+x+1)*ln(x^2+x+1), x), x) - (x^3+x^2+x+1)*ln(x^2+x+1)";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert!(
        residual_required.is_empty(),
        "cubic shifted positive-quadratic log by-parts residual should not add conditions: {residual_required:?}"
    );
}

#[test]
fn integrate_contract_quartic_times_shifted_positive_quadratic_log_flattens_remainder() {
    let input = "integrate((x^4+x^3+x^2+x+1)*ln(x^2+x+1), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(
        result,
        "ln(x^2 + x + 1) * (1/5 * x^5 + 1/4 * x^4 + 1/3 * x^3 + 1/2 * x^2 + x + 77/120) + 33/20 * arctan((2 * x + 1) / sqrt(3)) / sqrt(3) - 2/25 * x^5 - 3/40 * x^4 - 13/180 * x^3 - 49/120 * x^2 - 22/15 * x"
    );
    assert!(
        !result.contains(" - ("),
        "quartic shifted positive-quadratic log by-parts presentation should flatten subtracting a remainder group, got {result}"
    );
    assert!(
        required.is_empty(),
        "quartic shifted positive quadratic log argument should not add conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
    assert_rendered_antiderivative_verifies(input, &result);

    let residual =
        "diff(integrate((x^4+x^3+x^2+x+1)*ln(x^2+x+1), x), x) - (x^4+x^3+x^2+x+1)*ln(x^2+x+1)";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert!(
        residual_required.is_empty(),
        "quartic shifted positive-quadratic log by-parts residual should not add conditions: {residual_required:?}"
    );
}

#[test]
fn integrate_contract_affine_log_by_parts_presentation_stays_quiet() {
    for input in ["integrate(x*ln(2*x+1), x)", "integrate((x+1)*ln(2*x+1), x)"] {
        let (_wire, stderr) = cli_eval_json_with_stderr(input);
        assert!(
            !stderr.contains("depth_overflow"),
            "affine log by-parts presentation should not emit depth_overflow for {input}\nstderr:\n{stderr}"
        );
    }
}

#[test]
fn integrate_contract_affine_log_by_parts_offset_residual_stays_quiet() {
    for input in [
        "diff(integrate((x+1)*ln(2*x+1), x), x) - (x+1)*ln(2*x+1)",
        "diff(integrate((1-2*x)*ln(1-2*x), x), x) - (1-2*x)*ln(1-2*x)",
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr(input);

        assert_eq!(wire["result"], "0", "input: {input}");
        assert!(
            !stderr.contains("depth_overflow"),
            "offset affine log by-parts residual should not emit depth_overflow for {input}\nstderr:\n{stderr}"
        );
    }
}

#[test]
fn integrate_contract_polynomial_log_product_preserves_source_domain() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(2*x*ln(x^2-1), x)");

    assert_eq!(result, "(ln(x^2 - 1) - 1) * (x^2 - 1)");
    assert_eq!(
        required,
        vec!["x < -1 or x > 1".to_string()],
        "unexpected required_conditions: {required:?}"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate((2*x+1)*ln(x^2+x+1), x)");

    assert_eq!(result, "(ln(x^2 + x + 1) - 1) * (x^2 + x + 1)");
    assert!(
        required.is_empty(),
        "unexpected required_conditions: {required:?}"
    );
}

#[test]
fn integrate_contract_quadratic_times_positive_quadratic_log_by_parts_verifies() {
    let input = "integrate(x^2*ln(x^2+1), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(
        result,
        "1/3 * x^3 * ln(x^2 + 1) - (2/3 * arctan(x) + 2/9 * x^3 - 2/3 * x)"
    );
    assert!(
        required.is_empty(),
        "positive quadratic log argument should not add conditions: {required:?}"
    );
    assert_rendered_antiderivative_verifies(input, &result);

    let residual = "diff(integrate(x^2*ln(x^2+1), x), x) - x^2*ln(x^2+1)";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert!(
        residual_required.is_empty(),
        "positive quadratic log residual should not add conditions: {residual_required:?}"
    );

    assert_eq!(
        simplified_integral("integrate(x^2*ln(x^2-1), x)"),
        "integrate(ln(x^2 - 1) * x^2, x)",
        "indefinite-sign quadratic log arguments must stay unsupported"
    );
}

#[test]
fn integrate_contract_reciprocal_linear_uses_abs_log() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(1/(2*x + 1), x)");

    assert_eq!(result, "1/2 * ln(|2 * x + 1|)");
    assert_eq!(
        required,
        vec!["x ≠ -1/2".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_rendered_antiderivative_verifies("integrate(1/(2*x + 1), x)", &result);
}

#[test]
fn integrate_contract_constant_base_affine_logs_use_table_and_preserve_domain() {
    for (input, expected_result, expected_required) in [
        (
            "integrate(log(2, x), x)",
            "x * (log(2, x) - 1 / ln(2))",
            vec!["x > 0".to_string()],
        ),
        (
            "integrate(log(2, 3*x+2), x)",
            "1/3 * (3 * x + 2) * (log(2, 3 * x + 2) - 1 / ln(2))",
            vec!["x > -2/3".to_string()],
        ),
        (
            "integrate(log(1/2, x), x)",
            "x * (log(1/2, x) - 1 / ln(1/2))",
            vec!["x > 0".to_string()],
        ),
        (
            "integrate(log(e, x), x)",
            "x * (ln(x) - 1)",
            vec!["x > 0".to_string()],
        ),
        (
            "integrate(log2(x), x)",
            "x * (log2(x) - 1 / ln(2))",
            vec!["x > 0".to_string()],
        ),
        (
            "integrate(log10(3*x+2), x)",
            "1/3 * (3 * x + 2) * (log10(3 * x + 2) - 1 / ln(10))",
            vec!["x > -2/3".to_string()],
        ),
    ] {
        let (result, required) = evaluated_integral_with_required_conditions(input);
        assert_eq!(result, expected_result, "input: {input}");
        assert_eq!(
            required, expected_required,
            "unexpected required_conditions for {input}: {required:?}"
        );
        assert_antiderivative_verifies(input);
        assert_rendered_antiderivative_verifies(input, &result);
    }
}

#[test]
fn integrate_contract_constant_base_log_handles_invalid_and_symbolic_bases() {
    for input in [
        "integrate(log(1, x), x)",
        "integrate(log(-2, x), x)",
        "integrate(log(0, x), x)",
    ] {
        let (result, required) = evaluated_integral_with_required_conditions(input);
        assert_eq!(
            result, "undefined",
            "invalid log base should make the real-domain integrand undefined for {input}"
        );
        assert!(
            required.is_empty(),
            "invalid log base should not add conditional assumptions for {input}: {required:?}"
        );
    }

    let input = "integrate(log(y, x), x)";
    let (result, _required) = evaluated_integral_with_required_conditions(input);
    assert!(
        result.starts_with("integrate("),
        "symbolic log base should remain residual for {input}, got {result}"
    );
}

#[test]
fn integrate_contract_constant_base_affine_log_trace_stays_compact() {
    let input = "integrate(log(2, 3*x+2), x)";
    let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

    assert!(
        stderr.is_empty(),
        "constant-base affine log integration should not emit stderr warnings: {stderr}"
    );
    assert_eq!(
        wire["result"],
        "1/3·(3·x + 2)·(log(2, 3·x + 2) - 1 / ln(2))"
    );
    assert_eq!(wire["required_display"], serde_json::json!(["x > -2/3"]));
    let steps = wire["steps"]
        .as_array()
        .expect("steps should be present with --steps on");
    assert_eq!(steps.len(), 1, "expected one integration step: {steps:?}");
    assert_eq!(steps[0]["rule"], "Calcular la integral");
    assert!(
        steps[0]["substeps"].is_null(),
        "constant-base affine log table should not invent didactic substeps: {steps:?}"
    );
}

#[test]
fn integrate_contract_constant_base_polynomial_log_substitution_preserves_domain() {
    for (input, expected_result, expected_required) in [
        (
            "integrate(2*x*log(2,x^2+1), x)",
            "(x^2 + 1) * log(2, x^2 + 1) - (x^2 + 1) / ln(2)",
            vec![],
        ),
        (
            "integrate(2*x*log(2,x^2-1), x)",
            "(x^2 - 1) * log(2, x^2 - 1) - (x^2 - 1) / ln(2)",
            vec!["x < -1 or x > 1".to_string()],
        ),
        (
            "integrate(2*x*log2(x^2+1), x)",
            "(x^2 + 1) * log2(x^2 + 1) - (x^2 + 1) / ln(2)",
            vec![],
        ),
        (
            "integrate(2*x*log(2,x^2+x+1)+log(2,x^2+x+1), x)",
            "(x^2 + x + 1) * log(2, x^2 + x + 1) - (x^2 + x + 1) / ln(2)",
            vec![],
        ),
    ] {
        let (result, required) = evaluated_integral_with_required_conditions(input);
        assert_eq!(result, expected_result, "input: {input}");
        assert_eq!(
            required, expected_required,
            "unexpected required_conditions for {input}: {required:?}"
        );
        assert_antiderivative_verifies(input);
        assert_rendered_antiderivative_verifies(input, &result);
    }

    let (result, _required) =
        evaluated_integral_with_required_conditions("integrate(2*x*log(x,x^2+1), x)");
    assert!(
        result.starts_with("integrate("),
        "symbolic log base should remain residual for polynomial substitution, got {result}"
    );
}

#[test]
fn integrate_contract_constant_base_polynomial_log_trace_uses_substitution() {
    let input = "integrate(2*x*log(2,x^2+1), x)";
    let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

    assert!(
        stderr.is_empty(),
        "constant-base polynomial log integration should not emit stderr warnings: {stderr}"
    );
    assert_eq!(
        wire["result"],
        "(x^2 + 1)·log(2, x^2 + 1) - (x^2 + 1) / ln(2)"
    );
    assert!(wire["required_display"]
        .as_array()
        .expect("required_display array")
        .is_empty());
    let steps = wire["steps"]
        .as_array()
        .expect("steps should be present with --steps on");
    assert_eq!(steps.len(), 1, "expected one integration step: {steps:?}");
    assert_eq!(steps[0]["rule"], "Calcular la integral");
    let substeps = steps[0]["substeps"]
        .as_array()
        .expect("substitution should be visible as a compact substep");
    assert_eq!(
        substeps.len(),
        1,
        "expected one substitution substep: {steps:?}"
    );
    assert_eq!(substeps[0]["title"], "Usar sustitución");
}

#[test]
fn integrate_contract_constant_base_additive_polynomial_log_trace_uses_common_factor_then_substitution(
) {
    let input = "integrate(2*x*log(2,x^2+x+1)+log(2,x^2+x+1), x)";
    let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

    assert!(
        stderr.is_empty(),
        "constant-base additive polynomial log integration should not emit stderr warnings: {stderr}"
    );
    assert_eq!(
        wire["result"],
        "(x^2 + x + 1)·log(2, x^2 + x + 1) - (x^2 + x + 1) / ln(2)"
    );
    assert!(wire["required_display"]
        .as_array()
        .expect("required_display array")
        .is_empty());
    let steps = wire["steps"]
        .as_array()
        .expect("steps should be present with --steps on");
    assert_eq!(
        steps.len(),
        2,
        "expected factor then integration steps: {steps:?}"
    );
    assert_eq!(steps[0]["rule"], "Sacar factor común");
    assert_eq!(steps[1]["rule"], "Calcular la integral");
    let substeps = steps[1]["substeps"]
        .as_array()
        .expect("substitution should be visible as a compact substep");
    assert_eq!(
        substeps.len(),
        1,
        "expected one substitution substep: {steps:?}"
    );
    assert_eq!(substeps[0]["title"], "Usar sustitución");
}

#[test]
fn integrate_contract_simple_linear_partial_fractions_normalize_negative_factor() {
    let input = "integrate((x+2)/((x-1)*(-x-1)), x)";
    let (result, mut required) = evaluated_integral_with_required_conditions(input);

    assert!(
        result.contains("ln(|x + 1|)")
            && result.contains("ln(|x - 1|)")
            && !result.contains("ln(|-x - 1|)"),
        "expected normalized logarithmic simple-pole terms for negative factor orientation, got {result}"
    );
    required.sort();
    assert_eq!(
        required,
        vec!["x ≠ -1".to_string(), "x ≠ 1".to_string()],
        "unexpected negative-factor simple-pole required_conditions: {required:?}"
    );

    let residual = "diff(integrate((x+2)/((x-1)*(-x-1)), x), x) - (x+2)/((x-1)*(-x-1))";
    let (residual_result, mut residual_required) =
        evaluated_expr_with_required_conditions(residual);
    residual_required.sort();
    assert_eq!(residual_result, "0");
    assert_eq!(
        residual_required,
        vec!["x ≠ -1".to_string(), "x ≠ 1".to_string()],
        "negative-factor simple-pole verification should preserve the source denominator domain"
    );
}

#[test]
fn integrate_contract_linear_partial_fraction_log_result_exposes_didactic_substep() {
    let input = "integrate(2/(1-(2*x+1)^2), x)";
    let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

    assert_eq!(wire["result"], "1/2·ln(|(x + 1) / x|)");
    assert_eq!(
        wire["required_display"],
        serde_json::json!(["x ≠ -1", "x ≠ 0"])
    );
    assert!(
        !stderr.contains("depth_overflow"),
        "partial fraction trace should not emit depth_overflow warning\nstderr:\n{stderr}"
    );

    let steps = wire["steps"]
        .as_array()
        .expect("steps should be present with --steps on");
    let integration_step = steps
        .iter()
        .find(|step| step["rule"] == "Calcular la integral")
        .expect("expected public symbolic integration step");
    let substeps = integration_step["substeps"]
        .as_array()
        .expect("integration step should expose didactic substeps");
    assert!(
        substeps
            .iter()
            .any(|substep| substep["title"] == "Descomponer en fracciones parciales"),
        "expected partial-fraction didactic substep, got {substeps:?}"
    );
    let decomposition_substep = substeps
        .iter()
        .find(|substep| substep["title"] == "Descomponer en fracciones parciales")
        .expect("partial-fraction substep should exist");
    let decomposition_latex = decomposition_substep["after_latex"]
        .as_str()
        .expect("partial-fraction substep should expose concrete after_latex");
    assert!(
        decomposition_latex == "\\frac{1}{2\\cdot (x + 1)} - \\frac{1}{2\\cdot x}",
        "partial-fraction substep should show the decomposed simple fractions, got {decomposition_latex}"
    );
    assert!(
        substeps
            .iter()
            .any(|substep| substep["title"] == "Integrar los términos simples"),
        "expected simple-term integration substep, got {substeps:?}"
    );
    assert!(
        substeps
            .iter()
            .all(|substep| substep["title"] != "Usar sustitución"),
        "partial fractions should not be mislabeled as generic substitution: {substeps:?}"
    );
    assert_antiderivative_verifies(input);
}

#[test]
fn integrate_contract_rational_partial_fraction_with_two_linear_factors_and_positive_quadratic() {
    let input = "integrate(1/(x^4-1), x)";
    let (result, mut required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(
        result,
        "1/4 * ln(|x - 1|) - 1/2 * arctan(x) - 1/4 * ln(|x + 1|)"
    );
    required.sort();
    assert_eq!(
        required,
        vec!["x ≠ -1".to_string(), "x ≠ 1".to_string()],
        "positive quadratic remainder should not add real-domain conditions: {required:?}"
    );

    let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);
    assert!(
        stderr.is_empty(),
        "positive-quadratic partial-fraction trace should stay quiet\nstderr:\n{stderr}"
    );
    let steps = wire["steps"]
        .as_array()
        .expect("steps should be present with --steps on");
    let integration_step = steps
        .iter()
        .find(|step| step["rule"] == "Calcular la integral")
        .expect("expected integration step");
    let substeps = integration_step["substeps"]
        .as_array()
        .expect("integration step should expose didactic substeps");
    let decomposition_substep = substeps
        .iter()
        .find(|substep| substep["title"] == "Descomponer en fracciones parciales")
        .expect("expected positive-quadratic partial-fraction decomposition substep");
    let decomposition_latex = decomposition_substep["after_latex"]
        .as_str()
        .expect("partial-fraction substep should expose concrete after_latex");
    assert!(
        decomposition_latex.contains("x - 1")
            && decomposition_latex.contains("x + 1")
            && decomposition_latex.contains("{x}^{2} + 1"),
        "partial-fraction substep should show linear factors and positive quadratic, got {decomposition_latex}"
    );
    assert!(
        substeps
            .iter()
            .any(|substep| substep["title"] == "Integrar los términos simples"),
        "expected simple-term integration substep, got {substeps:?}"
    );

    let residual = "diff(integrate(1/(x^4-1), x), x) - 1/(x^4-1)";
    let (residual_result, mut residual_required) =
        evaluated_expr_with_required_conditions(residual);
    residual_required.sort();
    assert_eq!(residual_result, "0");
    assert_eq!(
        residual_required,
        vec!["x ≠ -1".to_string(), "x ≠ 1".to_string()],
        "nested verification should preserve only the source denominator domain"
    );

    let input = "integrate((x-1)/(x^4-1), x)";
    let (result, mut required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(
        result,
        "1/2 * arctan(x) - 1/4 * ln(x^2 + 1) + 1/2 * ln(|x + 1|)"
    );
    required.sort();
    assert_eq!(
        required,
        vec!["x ≠ -1".to_string(), "x ≠ 1".to_string()],
        "cancelled pole in the partial-fraction numerator must still retain the source denominator domain: {required:?}"
    );

    let residual = "diff(integrate((x-1)/(x^4-1), x), x) - (x-1)/(x^4-1)";
    let (residual_result, mut residual_required) =
        evaluated_expr_with_required_conditions(residual);
    residual_required.sort();
    assert_eq!(residual_result, "0");
    assert_eq!(
        residual_required,
        vec!["x ≠ -1".to_string(), "x ≠ 1".to_string()],
        "cancelled-pole nested verification should preserve the source denominator domain"
    );

    let input = "integrate((x^2-1)/(x^4-1), x)";
    let (result, mut required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "arctan(x)");
    required.sort();
    assert_eq!(
        required,
        vec!["x ≠ -1".to_string(), "x ≠ 1".to_string()],
        "fully cancelled real poles must still retain the source denominator domain: {required:?}"
    );

    let residual = "diff(integrate((x^2-1)/(x^4-1), x), x) - (x^2-1)/(x^4-1)";
    let (residual_result, mut residual_required) =
        evaluated_expr_with_required_conditions(residual);
    residual_required.sort();
    assert_eq!(residual_result, "0");
    assert_eq!(
        residual_required,
        vec!["x ≠ -1".to_string(), "x ≠ 1".to_string()],
        "fully-cancelled-pole nested verification should preserve the source denominator domain"
    );

    let input = "integrate((x^2-1)/((x-1)*(x+1)*(x^2+1)), x)";
    let (result, mut required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "arctan(x)");
    required.sort();
    assert_eq!(
        required,
        vec!["x ≠ -1".to_string(), "x ≠ 1".to_string()],
        "factored denominator cancellation must still retain only the real source denominator domain: {required:?}"
    );

    let residual =
        "diff(integrate((x^2-1)/((x-1)*(x+1)*(x^2+1)), x), x) - (x^2-1)/((x-1)*(x+1)*(x^2+1))";
    let (residual_result, mut residual_required) =
        evaluated_expr_with_required_conditions(residual);
    residual_required.sort();
    assert_eq!(residual_result, "0");
    assert_eq!(
        residual_required,
        vec!["x ≠ -1".to_string(), "x ≠ 1".to_string()],
        "factored-denominator nested verification should preserve only the real source denominator domain"
    );

    let input = "integrate((x^2-1)/(-(x-1)*(x+1)*(x^2+1)), x)";
    let (result, mut required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "-arctan(x)");
    required.sort();
    assert_eq!(
        required,
        vec!["x ≠ -1".to_string(), "x ≠ 1".to_string()],
        "negative factored denominator cancellation must retain only the real source denominator domain: {required:?}"
    );

    let residual =
        "diff(integrate((x^2-1)/(-(x-1)*(x+1)*(x^2+1)), x), x) - (x^2-1)/(-(x-1)*(x+1)*(x^2+1))";
    let (residual_result, mut residual_required) =
        evaluated_expr_with_required_conditions(residual);
    residual_required.sort();
    assert_eq!(residual_result, "0");
    assert_eq!(
        residual_required,
        vec!["x ≠ -1".to_string(), "x ≠ 1".to_string()],
        "negative-factored-denominator verification should preserve only the real source denominator domain"
    );
}

#[test]
fn integrate_contract_rational_partial_fractions_over_repeated_linear_factors() {
    let input = "integrate((3*x+5)/(x^3-x^2-x+1), x)";
    let (result, mut required) = evaluated_integral_with_required_conditions(input);

    assert!(
        !result.starts_with("integrate("),
        "expected a proper antiderivative, got {result}"
    );
    assert_eq!(result, "1/2 * ln(|(x + 1) / (x - 1)|) - 4 / (x - 1)");
    assert!(
        result.contains("4 / (x - 1)"),
        "expected repeated-pole rational term, got {result}"
    );
    assert!(
        result.contains(" - 4 / (x - 1)") && !result.contains("+ -"),
        "expected a clean subtraction for the repeated-pole rational term, got {result}"
    );
    required.sort();
    assert_eq!(
        required,
        vec!["x ≠ -1".to_string(), "x ≠ 1".to_string()],
        "unexpected required_conditions: {required:?}"
    );

    let residual = "diff(integrate((3*x+5)/(x^3-x^2-x+1), x), x) - (3*x+5)/(x^3-x^2-x+1)";
    let (residual_result, mut residual_required) =
        evaluated_expr_with_required_conditions(residual);
    residual_required.sort();
    assert_eq!(residual_result, "0");
    assert_eq!(
        residual_required,
        vec!["x ≠ -1".to_string(), "x ≠ 1".to_string()],
        "nested verification should preserve the source denominator domain"
    );
    assert_rendered_antiderivative_verifies(input, &result);

    let reordered_residual =
        "diff((-1/2)*ln(abs(x-1)) - 4/(x-1) + (1/2)*ln(abs(x+1)), x) - (3*x+5)/(x^3-x^2-x+1)";
    let (reordered_result, mut reordered_required) =
        evaluated_expr_with_required_conditions(reordered_residual);
    reordered_required.sort();
    assert_eq!(reordered_result, "0");
    assert_eq!(
        reordered_required,
        vec!["x ≠ -1".to_string(), "x ≠ 1".to_string()],
        "reordered rendered antiderivative verification should preserve the source denominator domain"
    );

    let input = "integrate((3*x+5)/((x-2)^2*(x+3)), x)";
    let (result, mut required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(
        result,
        "4/25 * ln(|(x - 2) / (x + 3)|) - 11 / (5 * (x - 2))"
    );
    required.sort();
    assert_eq!(
        required,
        vec!["x ≠ -3".to_string(), "x ≠ 2".to_string()],
        "unexpected shifted repeated-pole required_conditions: {required:?}"
    );

    let residual = "diff(integrate((3*x+5)/((x-2)^2*(x+3)), x), x) - (3*x+5)/((x-2)^2*(x+3))";
    let (residual_result, mut residual_required) =
        evaluated_expr_with_required_conditions(residual);
    residual_required.sort();
    assert_eq!(residual_result, "0");
    assert_eq!(
        residual_required,
        vec!["x ≠ -3".to_string(), "x ≠ 2".to_string()],
        "shifted repeated-pole nested verification should preserve the source denominator domain"
    );

    let input = "integrate((3*x+5)/(x^3-x^2-8*x+12), x)";
    let (result, mut required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(
        result,
        "4/25 * ln(|(x - 2) / (x + 3)|) - 11 / (5 * (x - 2))"
    );
    required.sort();
    assert_eq!(
        required,
        vec!["x ≠ -3".to_string(), "x ≠ 2".to_string()],
        "unexpected expanded shifted repeated-pole required_conditions: {required:?}"
    );

    let residual = "diff(integrate((3*x+5)/(x^3-x^2-8*x+12), x), x) - (3*x+5)/(x^3-x^2-8*x+12)";
    let (residual_result, mut residual_required) =
        evaluated_expr_with_required_conditions(residual);
    residual_required.sort();
    assert_eq!(residual_result, "0");
    assert_eq!(
        residual_required,
        vec!["x ≠ -3".to_string(), "x ≠ 2".to_string()],
        "expanded shifted repeated-pole nested verification should preserve the source denominator domain"
    );

    let input = "integrate((3*x+5)/((1-x)^2*(x+1)), x)";
    let (result, mut required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "1/2 * ln(|(x + 1) / (x - 1)|) - 4 / (x - 1)");
    assert!(
        result.contains(" - 4 / (x - 1)") && !result.contains("+ -"),
        "expected clean repeated-pole term for factored orientation, got {result}"
    );
    required.sort();
    assert_eq!(
        required,
        vec!["x ≠ -1".to_string(), "x ≠ 1".to_string()],
        "unexpected factored-orientation required_conditions: {required:?}"
    );

    let residual = "diff(integrate((3*x+5)/((1-x)^2*(x+1)), x), x) - (3*x+5)/((1-x)^2*(x+1))";
    let (residual_result, mut residual_required) =
        evaluated_expr_with_required_conditions(residual);
    residual_required.sort();
    assert_eq!(residual_result, "0");
    assert_eq!(
        residual_required,
        vec!["x ≠ -1".to_string(), "x ≠ 1".to_string()],
        "factored-orientation nested verification should preserve the source denominator domain"
    );

    let input = "integrate((3*x+5)/((x-1)^2*(-x-1)), x)";
    let (result, mut required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "1/2 * ln(|(x - 1) / (x + 1)|) + 4 / (x - 1)");
    assert!(
        result.contains("4 / (x - 1)") && !result.contains("+ -"),
        "expected clean repeated-pole term for negative factor orientation, got {result}"
    );
    required.sort();
    assert_eq!(
        required,
        vec!["x ≠ -1".to_string(), "x ≠ 1".to_string()],
        "unexpected negative-factor required_conditions: {required:?}"
    );

    let residual = "diff(integrate((3*x+5)/((x-1)^2*(-x-1)), x), x) - (3*x+5)/((x-1)^2*(-x-1))";
    let (residual_result, mut residual_required) =
        evaluated_expr_with_required_conditions(residual);
    residual_required.sort();
    assert_eq!(residual_result, "0");
    assert_eq!(
        residual_required,
        vec!["x ≠ -1".to_string(), "x ≠ 1".to_string()],
        "negative-factor nested verification should preserve the source denominator domain"
    );
}

#[test]
fn integrate_contract_scaled_repeated_linear_partial_fractions_normalize_log_arguments() {
    let input = "integrate((3*x+5)/(2*x^3-2*x^2-2*x+2), x)";
    let (result, mut required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "1/4 * ln(|(x + 1) / (x - 1)|) - 2 / (x - 1)");
    required.sort();
    assert_eq!(
        required,
        vec!["x ≠ -1".to_string(), "x ≠ 1".to_string()],
        "unexpected required_conditions: {required:?}"
    );

    let residual =
        "diff(integrate((3*x+5)/(2*x^3-2*x^2-2*x+2), x), x) - (3*x+5)/(2*x^3-2*x^2-2*x+2)";
    let (residual_result, mut residual_required) =
        evaluated_expr_with_required_conditions(residual);
    residual_required.sort();
    assert_eq!(residual_result, "0");
    assert_eq!(
        residual_required,
        vec!["x ≠ -1".to_string(), "x ≠ 1".to_string()],
        "nested verification should preserve the source denominator domain"
    );

    let input = "integrate((3*x+5)/(1/2*x^3-1/2*x^2-1/2*x+1/2), x)";
    let (result, mut required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "ln(|(x + 1) / (x - 1)|) - 8 / (x - 1)");
    required.sort();
    assert_eq!(
        required,
        vec!["x ≠ -1".to_string(), "x ≠ 1".to_string()],
        "unexpected fractional-scale required_conditions: {required:?}"
    );

    let residual =
        "diff(integrate((3*x+5)/(1/2*x^3-1/2*x^2-1/2*x+1/2), x), x) - (3*x+5)/(1/2*x^3-1/2*x^2-1/2*x+1/2)";
    let (residual_result, mut residual_required) =
        evaluated_expr_with_required_conditions(residual);
    residual_required.sort();
    assert_eq!(residual_result, "0");
    assert_eq!(
        residual_required,
        vec!["x ≠ -1".to_string(), "x ≠ 1".to_string()],
        "fractional nested verification should preserve the source denominator domain"
    );

    let input = "integrate(1/((2*x+2)^2*(x-1)), x)";
    let (result, mut required) = evaluated_integral_with_required_conditions(input);

    assert!(
        !result.starts_with("integrate("),
        "expected scaled repeated linear factors to integrate, got {result}"
    );
    assert_eq!(result, "1/16 * ln(|(x - 1) / (x + 1)|) + 1 / (8 * (x + 1))");
    required.sort();
    assert_eq!(
        required,
        vec!["x ≠ -1".to_string(), "x ≠ 1".to_string()],
        "unexpected scaled repeated-factor required_conditions: {required:?}"
    );

    let residual = "diff(integrate(1/((2*x+2)^2*(x-1)), x), x) - 1/((2*x+2)^2*(x-1))";
    let (residual_result, mut residual_required) =
        evaluated_expr_with_required_conditions(residual);
    residual_required.sort();
    assert_eq!(residual_result, "0");
    assert_eq!(
        residual_required,
        vec!["x ≠ -1".to_string(), "x ≠ 1".to_string()],
        "scaled repeated-factor nested verification should preserve the source denominator domain"
    );
}

#[test]
fn integrate_contract_degree_five_linear_partial_fractions_verify_by_diff() {
    let input = "integrate(1/((x-2)*(x-1)*x*(x+1)*(x+2)), x)";
    let (result, mut required) = evaluated_integral_with_required_conditions(input);

    assert!(
        !result.starts_with("integrate("),
        "expected a proper antiderivative for five linear factors, got {result}"
    );
    for log_arg in [
        "ln(|x - 2|)",
        "ln(|x - 1|)",
        "ln(|x|)",
        "ln(|x + 1|)",
        "ln(|x + 2|)",
    ] {
        assert!(
            result.contains(log_arg),
            "expected {log_arg} in five-factor partial-fraction antiderivative, got {result}"
        );
    }
    required.sort();
    assert_eq!(
        required,
        vec![
            "x ≠ -1".to_string(),
            "x ≠ -2".to_string(),
            "x ≠ 0".to_string(),
            "x ≠ 1".to_string(),
            "x ≠ 2".to_string(),
        ],
        "unexpected five-factor partial-fraction required_conditions: {required:?}"
    );

    let residual =
        "diff(integrate(1/((x-2)*(x-1)*x*(x+1)*(x+2)), x), x) - 1/((x-2)*(x-1)*x*(x+1)*(x+2))";
    let (residual_result, mut residual_required) =
        evaluated_expr_with_required_conditions(residual);
    residual_required.sort();
    assert_eq!(residual_result, "0");
    assert_eq!(
        residual_required,
        vec![
            "x ≠ -1".to_string(),
            "x ≠ -2".to_string(),
            "x ≠ 0".to_string(),
            "x ≠ 1".to_string(),
            "x ≠ 2".to_string(),
        ],
        "five-factor nested verification should preserve the source denominator domain"
    );

    let public_equiv = "equiv(diff(integrate(1/((x-2)*(x-1)*x*(x+1)*(x+2)), x), x), 1/((x-2)*(x-1)*x*(x+1)*(x+2)))";
    let (equiv_wire, equiv_stderr) = cli_eval_json_with_stderr(public_equiv);
    assert!(
        equiv_stderr.is_empty(),
        "public five-factor antiderivative check should avoid depth overflow: {equiv_stderr}"
    );
    assert_eq!(equiv_wire["result"], "true");
    assert_eq!(
        equiv_wire["required_display"],
        serde_json::json!(["x ≠ 0", "x ≠ 1", "x ≠ -1", "x ≠ 2", "x ≠ -2"])
    );

    let direct_diff = "diff(integrate(1/((x-2)*(x-1)*x*(x+1)*(x+2)), x), x)";
    let (direct_wire, direct_stderr) = cli_eval_json_with_stderr(direct_diff);
    assert!(
        direct_stderr.is_empty(),
        "direct five-factor diff/integrate should avoid depth overflow: {direct_stderr}"
    );
    assert_eq!(
        direct_wire["result"],
        "1 / (x·(x + 1)·(x + 2)·(x - 1)·(x - 2))"
    );
    assert_eq!(
        direct_wire["required_display"],
        serde_json::json!(["x ≠ -1", "x ≠ -2", "x ≠ 0", "x ≠ 1", "x ≠ 2"])
    );
    assert_rendered_antiderivative_verifies(input, &result);
}

#[test]
fn integrate_contract_degree_six_linear_partial_fractions_verify_by_diff() {
    let input = "integrate(1/((x-3)*(x-2)*(x-1)*x*(x+1)*(x+2)), x)";
    let (result, mut required) = evaluated_integral_with_required_conditions(input);

    assert!(
        !result.starts_with("integrate("),
        "expected a proper antiderivative for six linear factors, got {result}"
    );
    for log_arg in [
        "ln(|x - 3|)",
        "ln(|x - 2|)",
        "ln(|x - 1|)",
        "ln(|x|)",
        "ln(|x + 1|)",
        "ln(|x + 2|)",
    ] {
        assert!(
            result.contains(log_arg),
            "expected {log_arg} in six-factor partial-fraction antiderivative, got {result}"
        );
    }
    required.sort();
    assert_eq!(
        required,
        vec![
            "x ≠ -1".to_string(),
            "x ≠ -2".to_string(),
            "x ≠ 0".to_string(),
            "x ≠ 1".to_string(),
            "x ≠ 2".to_string(),
            "x ≠ 3".to_string(),
        ],
        "unexpected six-factor partial-fraction required_conditions: {required:?}"
    );

    let residual =
        "diff(integrate(1/((x-3)*(x-2)*(x-1)*x*(x+1)*(x+2)), x), x) - 1/((x-3)*(x-2)*(x-1)*x*(x+1)*(x+2))";
    let (residual_result, mut residual_required) =
        evaluated_expr_with_required_conditions(residual);
    residual_required.sort();
    assert_eq!(residual_result, "0");
    assert_eq!(
        residual_required,
        vec![
            "x ≠ -1".to_string(),
            "x ≠ -2".to_string(),
            "x ≠ 0".to_string(),
            "x ≠ 1".to_string(),
            "x ≠ 2".to_string(),
            "x ≠ 3".to_string(),
        ],
        "six-factor nested verification should preserve the source denominator domain"
    );
    assert_rendered_antiderivative_verifies(input, &result);
}

#[test]
fn integrate_contract_linear_times_positive_quadratic_partial_fraction_verify_by_diff() {
    let input = "integrate(1/((x+1)*(x^2+1)), x)";
    let (result, mut required) = evaluated_integral_with_required_conditions(input);

    assert!(
        !result.starts_with("integrate("),
        "expected a proper antiderivative for linear times positive quadratic, got {result}"
    );
    assert!(
        result.contains("ln(|x + 1|)")
            && result.contains("ln(x^2 + 1)")
            && result.contains("arctan(x)"),
        "expected log-linear plus positive-quadratic log/arctan terms, got {result}"
    );
    required.sort();
    assert_eq!(
        required,
        vec!["x ≠ -1".to_string()],
        "unexpected linear-positive-quadratic required_conditions: {required:?}"
    );

    let residual = "diff(integrate(1/((x+1)*(x^2+1)), x), x) - 1/((x+1)*(x^2+1))";
    let (residual_result, mut residual_required) =
        evaluated_expr_with_required_conditions(residual);
    residual_required.sort();
    assert_eq!(residual_result, "0");
    assert_eq!(
        residual_required,
        vec!["x ≠ -1".to_string()],
        "linear-positive-quadratic verification should keep only the linear-pole domain"
    );
    assert_rendered_antiderivative_verifies(input, &result);

    let input = "integrate(1/((x+2)*(x^2+2*x+5)), x)";
    let (result, mut required) = evaluated_integral_with_required_conditions(input);

    assert!(
        !result.starts_with("integrate("),
        "expected a proper antiderivative for shifted linear-positive-quadratic partial fractions, got {result}"
    );
    assert!(
        result.contains("ln(|x + 2|)")
            && result.contains("ln(x^2 + 2 * x + 5)")
            && result.contains("arctan(1/2 * x + 1/2)"),
        "expected shifted log-linear plus scaled positive-quadratic log/arctan terms, got {result}"
    );
    required.sort();
    assert_eq!(
        required,
        vec!["x ≠ -2".to_string()],
        "shifted positive-quadratic denominator should not add synthetic required conditions: {required:?}"
    );

    let residual = "diff(integrate(1/((x+2)*(x^2+2*x+5)), x), x) - 1/((x+2)*(x^2+2*x+5))";
    let (residual_result, mut residual_required) =
        evaluated_expr_with_required_conditions(residual);
    residual_required.sort();
    assert_eq!(residual_result, "0");
    assert_eq!(
        residual_required,
        vec!["x ≠ -2".to_string()],
        "shifted linear-positive-quadratic verification should keep only the linear-pole domain"
    );
    let (equivalent, mut equiv_required) = evaluated_equiv_with_required_conditions(
        "diff(integrate(1/((x+2)*(x^2+2*x+5)), x), x)",
        "1/((x+2)*(x^2+2*x+5))",
    );
    equiv_required.sort();
    assert!(
        equivalent,
        "public equivalence should verify the shifted linear-positive-quadratic antiderivative by differentiation"
    );
    assert_eq!(
        equiv_required,
        vec!["x ≠ -2".to_string()],
        "public equivalence should keep only the shifted linear-pole domain"
    );

    let direct_diff = "diff(integrate(1/((x+2)*(x^2+2*x+5)), x), x)";
    let (direct_wire, direct_stderr) = cli_eval_json_with_stderr(direct_diff);
    assert!(
        direct_stderr.is_empty(),
        "direct shifted linear-positive-quadratic diff/integrate should avoid depth overflow: {direct_stderr}"
    );
    let direct_result = direct_wire["result"].as_str().unwrap_or_default();
    assert!(
        direct_result.contains("x + 2") && direct_result.contains("x^2 + 2·x + 5"),
        "direct shifted linear-positive-quadratic diff/integrate should preserve a compact denominator, got {direct_result}"
    );
    assert_eq!(
        direct_wire["required_display"],
        serde_json::json!(["x ≠ -2"])
    );
    assert_rendered_antiderivative_verifies(input, &result);

    let input = "integrate((x+2)/(x^3+1), x)";
    let (result, mut required) = evaluated_integral_with_required_conditions(input);

    assert!(
        !result.starts_with("integrate("),
        "expected a proper antiderivative for expanded cubic linear-positive-quadratic partial fractions, got {result}"
    );
    assert!(
        result.contains("ln(|x + 1|)")
            && result.contains("ln(x^2 + 1 - x)")
            && result.contains("arctan((2 * x - 1) / sqrt(3))"),
        "expected expanded cubic to decompose into linear-log plus positive-quadratic log/arctan terms, got {result}"
    );
    required.sort();
    assert_eq!(
        required,
        vec!["x ≠ -1".to_string()],
        "expanded cubic denominator should only require the real linear pole domain: {required:?}"
    );

    let residual = "diff(integrate((x+2)/(x^3+1), x), x) - (x+2)/(x^3+1)";
    let (residual_result, mut residual_required) =
        evaluated_expr_with_required_conditions(residual);
    residual_required.sort();
    assert_eq!(residual_result, "0");
    assert_eq!(
        residual_required,
        vec!["x ≠ -1".to_string()],
        "expanded cubic verification should preserve only the real linear pole domain"
    );

    let input = "integrate((x^2+1)/(x^3+1), x)";
    let (result, mut required) = evaluated_integral_with_required_conditions(input);

    assert!(
        !result.starts_with("integrate("),
        "expected a proper antiderivative for expanded cubic quadratic-numerator partial fractions, got {result}"
    );
    assert!(
        result.contains("ln(|x + 1|)")
            && result.contains("ln(x^2 + 1 - x)")
            && result.contains("arctan((2 * x - 1) / sqrt(3))"),
        "expected expanded cubic quadratic numerator to decompose into linear-log plus positive-quadratic log/arctan terms, got {result}"
    );
    required.sort();
    assert_eq!(
        required,
        vec!["x ≠ -1".to_string()],
        "expanded cubic quadratic numerator should only require the real linear pole domain: {required:?}"
    );

    let residual = "diff(integrate((x^2+1)/(x^3+1), x), x) - (x^2+1)/(x^3+1)";
    let (residual_result, mut residual_required) =
        evaluated_expr_with_required_conditions(residual);
    residual_required.sort();
    assert_eq!(residual_result, "0");
    assert_eq!(
        residual_required,
        vec!["x ≠ -1".to_string()],
        "expanded cubic quadratic-numerator verification should preserve only the real linear pole domain"
    );
}

#[test]
fn integrate_contract_linear_times_definite_quadratic_handles_negative_orientation() {
    let input = "integrate(1/((-x-1)*(x^2+1)), x)";
    let (result, mut required) = evaluated_integral_with_required_conditions(input);

    assert!(
        !result.starts_with("integrate("),
        "expected a proper antiderivative for negatively oriented linear times definite quadratic, got {result}"
    );
    assert!(
        result.contains("ln(|x + 1|)")
            && result.contains("ln(x^2 + 1)")
            && result.contains("arctan(x)"),
        "expected negatively oriented log-linear plus quadratic log/arctan terms, got {result}"
    );
    required.sort();
    assert_eq!(
        required,
        vec!["x ≠ -1".to_string()],
        "unexpected negative-orientation linear-definite-quadratic required_conditions: {required:?}"
    );

    let residual = "diff(integrate(1/((-x-1)*(x^2+1)), x), x) - 1/((-x-1)*(x^2+1))";
    let (residual_result, mut residual_required) =
        evaluated_expr_with_required_conditions(residual);
    residual_required.sort();
    assert_eq!(residual_result, "0");
    assert_eq!(
        residual_required,
        vec!["x ≠ -1".to_string()],
        "negative-orientation verification should keep only the linear-pole domain"
    );
    assert_rendered_antiderivative_verifies(input, &result);
}

#[test]
fn integrate_contract_repeated_linear_times_definite_quadratic_partial_fraction() {
    let input = "integrate((x+2)/((x+1)^2*(x^2+1)), x)";
    let (result, mut required) = evaluated_integral_with_required_conditions(input);

    assert!(
        !result.starts_with("integrate("),
        "expected a proper antiderivative for repeated-linear times definite quadratic, got {result}"
    );
    assert!(
        result.contains("ln(|x + 1|)")
            && result.contains("ln(x^2 + 1)")
            && result.contains("arctan(x)")
            && result.contains("1 / (2 * (x + 1))"),
        "expected repeated-pole reciprocal plus quadratic log/arctan terms, got {result}"
    );
    required.sort();
    assert_eq!(
        required,
        vec!["x ≠ -1".to_string()],
        "unexpected repeated-linear-definite-quadratic required_conditions: {required:?}"
    );

    let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);
    assert!(
        stderr.is_empty(),
        "repeated-pole partial-fraction trace should stay quiet\nstderr:\n{stderr}"
    );
    let steps = wire["steps"]
        .as_array()
        .expect("steps should be present with --steps on");
    let integration_step = steps
        .iter()
        .find(|step| step["rule"] == "Calcular la integral")
        .expect("expected integration step");
    let substeps = integration_step["substeps"]
        .as_array()
        .expect("integration step should expose didactic substeps");
    let decomposition_latex = substeps
        .iter()
        .find(|substep| substep["title"] == "Descomponer en fracciones parciales")
        .and_then(|substep| substep["after_latex"].as_str())
        .expect("expected concrete partial-fraction decomposition substep");
    assert!(
        decomposition_latex.contains("- \\frac{x - \\frac{1}{2}}{{x}^{2} + 1}")
            && !decomposition_latex.contains("+ \\frac{-"),
        "negative quadratic numerator should render as subtraction, got {decomposition_latex}"
    );

    let residual = "diff(integrate((x+2)/((x+1)^2*(x^2+1)), x), x) - (x+2)/((x+1)^2*(x^2+1))";
    let (residual_result, mut residual_required) =
        evaluated_expr_with_required_conditions(residual);
    residual_required.sort();
    assert_eq!(residual_result, "0");
    assert_eq!(
        residual_required,
        vec!["x ≠ -1".to_string()],
        "repeated-linear-definite-quadratic verification should keep only the linear-pole domain"
    );
    assert_rendered_antiderivative_verifies(input, &result);
}

#[test]
fn integrate_contract_cubic_repeated_linear_times_definite_quadratic_partial_fraction() {
    let input = "integrate((x+2)/((x+1)^3*(x^2+1)), x)";
    let (result, mut required) = evaluated_integral_with_required_conditions(input);

    assert!(
        !result.starts_with("integrate("),
        "expected a proper antiderivative for cubic repeated-linear times definite quadratic, got {result}"
    );
    assert!(
        result.contains("ln(|x + 1|)")
            && result.contains("ln(x^2 + 1)")
            && result.contains("arctan(x)")
            && result.contains("1 / (x + 1)")
            && result.contains("1 / (4 * (x + 1)^2)"),
        "expected cubic repeated-pole reciprocal plus quadratic log/arctan terms, got {result}"
    );
    required.sort();
    assert_eq!(
        required,
        vec!["x ≠ -1".to_string()],
        "unexpected cubic repeated-linear-definite-quadratic required_conditions: {required:?}"
    );

    assert_antiderivative_verifies(input);
    assert_rendered_antiderivative_verifies(input, &result);
}

#[test]
fn integrate_contract_shifted_definite_quadratic_cubic_repeated_pole_verifies_by_diff() {
    let input = "integrate(1/((x+1)^3*(x^2+2*x+2)), x)";
    let (result, mut required) = evaluated_integral_with_required_conditions(input);

    assert!(
        !result.starts_with("integrate("),
        "expected a proper antiderivative for shifted quadratic cubic repeated pole, got {result}"
    );
    assert!(
        result.contains("ln(|x + 1|)")
            && result.contains("ln(x^2 + 2 * x + 2)")
            && result.contains("1 / (2 * (x + 1)^2)"),
        "expected shifted quadratic log and compact reciprocal terms, got {result}"
    );
    required.sort();
    assert_eq!(
        required,
        vec!["x ≠ -1".to_string()],
        "unexpected shifted-quadratic repeated-pole required_conditions: {required:?}"
    );

    let residual = "diff(integrate(1/((x+1)^3*(x^2+2*x+2)), x), x) - 1/((x+1)^3*(x^2+2*x+2))";
    let (residual_result, mut residual_required) =
        evaluated_expr_with_required_conditions(residual);
    residual_required.sort();
    assert_eq!(residual_result, "0");
    assert_eq!(
        residual_required,
        vec!["x ≠ -1".to_string()],
        "shifted-quadratic repeated-pole verification should keep only the linear-pole domain"
    );
    assert_rendered_antiderivative_verifies(input, &result);
}

#[test]
fn integrate_contract_improper_rational_partial_fractions_use_polynomial_division() {
    let input = "integrate(x^2/(x+1), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert!(
        !result.starts_with("integrate("),
        "expected a proper antiderivative for linear improper rational integrand, got {result}"
    );
    assert!(
        (result.contains("x^2 / 2") || result.contains("1/2 * x^2"))
            && result.contains("- x")
            && result.contains("ln(|x + 1|)")
            && !result.contains("+ -"),
        "expected polynomial division plus linear-log remainder, got {result}"
    );
    assert_eq!(
        required,
        vec!["x ≠ -1".to_string()],
        "unexpected linear improper rational required_conditions: {required:?}"
    );

    let residual = "diff(integrate(x^2/(x+1), x), x) - x^2/(x+1)";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert_eq!(
        residual_required,
        vec!["x ≠ -1".to_string()],
        "linear improper rational nested verification should preserve the source denominator domain"
    );

    let input = "integrate(x^2/(2*x+2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(
        result, "1/2 * ln(|x + 1|) + 1/4 * x^2 - 1/2 * x",
        "scaled linear improper rational should fold nested rational factors"
    );
    assert_eq!(
        required,
        vec!["x ≠ -1".to_string()],
        "unexpected scaled linear improper rational required_conditions: {required:?}"
    );

    let residual = "diff(integrate(x^2/(2*x+2), x), x) - x^2/(2*x+2)";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert_eq!(
        residual_required,
        vec!["x ≠ -1".to_string()],
        "scaled linear improper rational nested verification should preserve the source denominator domain"
    );

    let input = "integrate(x^2/(-2*x-2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(
        result, "1/2 * x - 1/2 * ln(|x + 1|) - 1/4 * x^2",
        "negative scaled linear improper rational should fold nested rational factors"
    );
    assert_eq!(
        required,
        vec!["x ≠ -1".to_string()],
        "unexpected negative scaled linear improper rational required_conditions: {required:?}"
    );

    let residual = "diff(integrate(x^2/(-2*x-2), x), x) - x^2/(-2*x-2)";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert_eq!(
        residual_required,
        vec!["x ≠ -1".to_string()],
        "negative scaled linear improper rational nested verification should preserve the source denominator domain"
    );

    let input = "integrate((x^3+3*x+5)/(x^3-x^2-x+1), x)";
    let (result, mut required) = evaluated_integral_with_required_conditions(input);

    assert!(
        !result.starts_with("integrate("),
        "expected a proper antiderivative, got {result}"
    );
    assert!(
        result.contains("ln(|x + 1|)") && result.contains("ln(|x - 1|)"),
        "expected logarithmic partial-fraction remainder terms, got {result}"
    );
    assert!(
        result.contains("9 / (2 * (x - 1))")
            && !result.contains("9/2 / (x - 1)")
            && !result.contains("+ -"),
        "expected a clean repeated-pole rational remainder term, got {result}"
    );
    assert!(
        result.contains("+ x") && !result.contains("1 * x"),
        "expected the polynomial quotient term to omit the unit factor, got {result}"
    );
    required.sort();
    assert_eq!(
        required,
        vec!["x ≠ -1".to_string(), "x ≠ 1".to_string()],
        "unexpected required_conditions: {required:?}"
    );

    let residual = "diff(integrate((x^3+3*x+5)/(x^3-x^2-x+1), x), x) - (x^3+3*x+5)/(x^3-x^2-x+1)";
    let (residual_result, mut residual_required) =
        evaluated_expr_with_required_conditions(residual);
    residual_required.sort();
    assert_eq!(residual_result, "0");
    assert_eq!(
        residual_required,
        vec!["x ≠ -1".to_string(), "x ≠ 1".to_string()],
        "nested verification should preserve the source denominator domain"
    );
    let (equivalent, mut equiv_required) = evaluated_equiv_with_required_conditions(
        "diff(integrate((x^3+3*x+5)/(x^3-x^2-x+1), x), x)",
        "(x^3+3*x+5)/(x^3-x^2-x+1)",
    );
    equiv_required.sort();
    assert!(
        equivalent,
        "public equivalence should verify the improper rational antiderivative by differentiation"
    );
    assert_eq!(
        equiv_required,
        vec!["x ≠ -1".to_string(), "x ≠ 1".to_string()],
        "public equivalence should preserve the source denominator domain"
    );

    let input = "integrate((x^3+3*x+5)/(-x^3+x^2+x-1), x)";
    let (result, mut required) = evaluated_integral_with_required_conditions(input);

    assert!(
        !result.starts_with("integrate("),
        "expected a proper antiderivative for negative orientation, got {result}"
    );
    assert!(
        result.contains("ln(|x + 1|)") && result.contains("ln(|x - 1|)"),
        "expected logarithmic partial-fraction remainder terms for negative orientation, got {result}"
    );
    assert!(
        result.contains("9 / (2 * (x - 1))")
            && !result.contains("9/2 / (x - 1)")
            && result.contains("- x")
            && !result.contains("+ -"),
        "expected clean oriented polynomial and repeated-pole terms, got {result}"
    );
    required.sort();
    assert_eq!(
        required,
        vec!["x ≠ -1".to_string(), "x ≠ 1".to_string()],
        "unexpected negative-orientation required_conditions: {required:?}"
    );

    let residual = "diff(integrate((x^3+3*x+5)/(-x^3+x^2+x-1), x), x) - (x^3+3*x+5)/(-x^3+x^2+x-1)";
    let (residual_result, mut residual_required) =
        evaluated_expr_with_required_conditions(residual);
    residual_required.sort();
    assert_eq!(residual_result, "0");
    assert_eq!(
        residual_required,
        vec!["x ≠ -1".to_string(), "x ≠ 1".to_string()],
        "negative-orientation nested verification should preserve the source denominator domain"
    );

    let input = "integrate((x^5+3*x+5)/(x^3-x^2-x+1), x)";
    let (result, mut required) = evaluated_integral_with_required_conditions(input);

    assert!(
        (result.contains("x^3 / 3") || result.contains("1/3 * x^3"))
            && !result.contains("x^(1 + 2) / (1 + 2)"),
        "higher-degree quotient should render folded polynomial power terms, got {result}"
    );
    required.sort();
    assert_eq!(
        required,
        vec!["x ≠ -1".to_string(), "x ≠ 1".to_string()],
        "unexpected higher-degree quotient required_conditions: {required:?}"
    );

    let residual = "diff(integrate((x^5+3*x+5)/(x^3-x^2-x+1), x), x) - (x^5+3*x+5)/(x^3-x^2-x+1)";
    let (residual_result, mut residual_required) =
        evaluated_expr_with_required_conditions(residual);
    residual_required.sort();
    assert_eq!(residual_result, "0");
    assert_eq!(
        residual_required,
        vec!["x ≠ -1".to_string(), "x ≠ 1".to_string()],
        "higher-degree quotient nested verification should preserve the source denominator domain"
    );
}

#[test]
fn integrate_contract_linear_numerator_over_positive_quadratic_decomposes_to_log_plus_arctan() {
    let input = "integrate((x+1)/(x^2+1), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "arctan(x) + 1/2 * ln(x^2 + 1)");
    assert!(
        required.is_empty(),
        "positive quadratic denominator should not add synthetic required conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
    let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);
    assert!(
        stderr.is_empty(),
        "positive quadratic linear-numerator trace should stay quiet\nstderr:\n{stderr}"
    );
    let steps = wire["steps"]
        .as_array()
        .expect("steps should be present with --steps on");
    let integration_step = steps
        .iter()
        .find(|step| step["rule"] == "Calcular la integral")
        .expect("expected public symbolic integration step");
    let substeps = integration_step["substeps"]
        .as_array()
        .expect("integration step should expose didactic substeps");
    let decomposition_substep = substeps
        .iter()
        .find(|substep| substep["title"] == "Descomponer en fracciones parciales")
        .expect("expected positive quadratic numerator decomposition substep");
    let decomposition_latex = decomposition_substep["after_latex"]
        .as_str()
        .expect("decomposition substep should expose concrete after_latex");
    assert!(
        decomposition_latex.contains("\\frac{x}{{x}^{2} + 1}")
            && decomposition_latex.contains("\\frac{1}{{x}^{2} + 1}"),
        "decomposition should expose derivative and arctan parts, got {decomposition_latex}"
    );
    assert!(
        substeps
            .iter()
            .any(|substep| substep["title"] == "Integrar los términos simples"),
        "expected simple-term integration substep, got {substeps:?}"
    );

    let input = "integrate(x/(x^2+2*x+2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "1/2 * ln(x^2 + 2 * x + 2) - arctan(x + 1)");
    assert!(
        required.is_empty(),
        "positive shifted quadratic denominator should not add synthetic required conditions: {required:?}"
    );

    let residual = "diff(integrate(x/(x^2+2*x+2), x), x) - x/(x^2+2*x+2)";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert!(
        residual_required.is_empty(),
        "nested verification should not invent denominator conditions for a positive quadratic"
    );

    let input = "integrate(1/(x^2+x+1), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "2 * arctan((2 * x + 1) / sqrt(3)) / sqrt(3)");
    assert!(
        !result.contains("sqrt(3/4)"),
        "positive quadratic presentation should reduce rational surd width: {result}"
    );
    assert!(
        required.is_empty(),
        "positive shifted quadratic denominator should not add synthetic required conditions: {required:?}"
    );

    let residual = "diff(integrate(1/(x^2+x+1), x), x) - 1/(x^2+x+1)";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert!(
        residual_required.is_empty(),
        "nested verification for reduced surd width should not invent denominator conditions"
    );

    let input = "integrate(1/(1/2*x^2+1/2*x+1/2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "4 * arctan((2 * x + 1) / sqrt(3)) / sqrt(3)");
    assert!(
        !result.contains("sqrt(3/4)"),
        "scaled positive quadratic presentation should reduce rational surd width: {result}"
    );
    assert!(
        required.is_empty(),
        "scaled positive shifted quadratic denominator should not add synthetic required conditions: {required:?}"
    );

    let residual = "diff(integrate(1/(1/2*x^2+1/2*x+1/2), x), x) - 1/(1/2*x^2+1/2*x+1/2)";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert!(
        residual_required.is_empty(),
        "nested verification for scaled reduced surd width should not invent denominator conditions"
    );

    let input = "integrate((x+3)/(2*x^2+4*x+4), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "arctan(x + 1) + 1/4 * ln(x^2 + 2 * x + 2)");
    assert!(
        required.is_empty(),
        "positive scaled quadratic denominator should not add synthetic required conditions: {required:?}"
    );

    let residual = "diff(integrate((x+3)/(2*x^2+4*x+4), x), x) - (x+3)/(2*x^2+4*x+4)";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert!(
        residual_required.is_empty(),
        "scaled nested verification should not invent denominator conditions"
    );

    let input = "integrate((x+1)/(1/2*x^2+1/2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "ln(x^2 + 1) + 2 * arctan(x)");
    assert!(
        required.is_empty(),
        "fractionally scaled positive quadratic denominator should not add synthetic required conditions: {required:?}"
    );

    let residual = "diff(integrate((x+1)/(1/2*x^2+1/2), x), x) - (x+1)/(1/2*x^2+1/2)";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert!(
        residual_required.is_empty(),
        "fractionally scaled nested verification should not invent denominator conditions"
    );

    let input = "integrate((x+1)/(-2*x^2-2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "-1/2 * arctan(x) - 1/4 * ln(x^2 + 1)");
    assert!(
        required.is_empty(),
        "negative scaled positive quadratic denominator should not add synthetic required conditions: {required:?}"
    );

    let residual = "diff(integrate((x+1)/(-2*x^2-2), x), x) - (x+1)/(-2*x^2-2)";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert!(
        residual_required.is_empty(),
        "negative scaled nested verification should not invent denominator conditions"
    );

    let input = "integrate((x+1)/(-1/2*x^2-1/2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "-ln(x^2 + 1) - 2 * arctan(x)");
    assert!(
        required.is_empty(),
        "negative fractionally scaled positive quadratic denominator should not add synthetic required conditions: {required:?}"
    );

    let residual = "diff(integrate((x+1)/(-1/2*x^2-1/2), x), x) - (x+1)/(-1/2*x^2-1/2)";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert!(
        residual_required.is_empty(),
        "negative fractionally scaled nested verification should not invent denominator conditions"
    );

    let input = "integrate(x/(-x^2-2*x-2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "arctan(x + 1) - 1/2 * ln(x^2 + 2 * x + 2)");
    assert!(
        required.is_empty(),
        "negative shifted positive quadratic denominator should not add synthetic required conditions: {required:?}"
    );

    let residual = "diff(integrate(x/(-x^2-2*x-2), x), x) - x/(-x^2-2*x-2)";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert!(
        residual_required.is_empty(),
        "negative shifted nested verification should not invent denominator conditions"
    );

    let input = "integrate((x^3+2*x^2+3*x+4)/(x^2+2*x+2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(
        result,
        "1/2 * ln(x^2 + 2 * x + 2) + 3 * arctan(x + 1) + x^2 / 2"
    );
    assert!(
        required.is_empty(),
        "shifted positive quadratic denominator should not add synthetic required conditions: {required:?}"
    );

    let residual =
        "diff(integrate((x^3+2*x^2+3*x+4)/(x^2+2*x+2), x), x) - (x^3+2*x^2+3*x+4)/(x^2+2*x+2)";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert!(
        residual_required.is_empty(),
        "shifted nested verification should not invent denominator conditions"
    );

    let negative_residual =
        "diff(integrate((x^3+2*x^2+3*x+4)/(-x^2-2*x-2), x), x) - (x^3+2*x^2+3*x+4)/(-x^2-2*x-2)";
    let (wire, stderr) = cli_eval_json_with_stderr(negative_residual);
    assert_eq!(wire["result"].as_str(), Some("0"));
    assert!(
        !stderr.contains("depth_overflow"),
        "negative shifted improper quadratic verification should avoid depth_overflow\nstderr:\n{stderr}"
    );
}

#[test]
fn integrate_contract_positive_quadratic_square_decomposes_to_arctan_plus_rational() {
    let input = "integrate(1/(x^2+1)^2, x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "1/2 * arctan(x) + x / (2 * (x^2 + 1))");
    assert!(
        required.is_empty(),
        "positive quadratic square should not add synthetic required conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
    assert_rendered_antiderivative_verifies(input, &result);
    let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);
    assert!(
        stderr.is_empty(),
        "positive quadratic square trace should stay quiet\nstderr:\n{stderr}"
    );
    let steps = wire["steps"]
        .as_array()
        .expect("steps should be present with --steps on");
    let integration_step = steps
        .iter()
        .find(|step| step["rule"] == "Calcular la integral")
        .expect("expected public symbolic integration step");
    let substeps = integration_step["substeps"]
        .as_array()
        .expect("integration step should expose didactic substeps");
    let reduction_substep = substeps
        .iter()
        .find(|substep| substep["title"] == "Reducir el cuadrático positivo al cuadrado")
        .expect("expected positive quadratic square reduction substep");
    let reduction_latex = reduction_substep["after_latex"]
        .as_str()
        .expect("reduction substep should expose concrete after_latex");
    assert!(
        reduction_latex.contains("\\frac{1}{2\\cdot ({x}^{2} + 1)}")
            && reduction_latex.contains("{({x}^{2} + 1)}^{2}"),
        "reduction should expose the arctan integrand and rational derivative part, got {reduction_latex}"
    );
    assert!(
        substeps
            .iter()
            .any(|substep| substep["title"] == "Integrar la parte arctan y la parte racional"),
        "expected final integration substep, got {substeps:?}"
    );

    let residual = "diff(integrate(1/(x^2+1)^2, x), x) - 1/(x^2+1)^2";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert!(
        residual_required.is_empty(),
        "nested verification should not invent denominator conditions for a positive quadratic square"
    );

    let input = "integrate(x^2/(x^2+1)^2, x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "1/2 * arctan(x) - x / (2 * (x^2 + 1))");
    assert!(
        required.is_empty(),
        "quadratic numerator over a positive quadratic square should not add synthetic required conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
    assert_rendered_antiderivative_verifies(input, &result);

    let residual = "diff(integrate(x^2/(x^2+1)^2, x), x) - x^2/(x^2+1)^2";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert!(
        residual_required.is_empty(),
        "nested verification should not invent denominator conditions for a quadratic numerator over a positive quadratic square"
    );

    let input = "integrate(1/((x+1)^2+1)^2, x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(
        result,
        "1/2 * arctan(x + 1) + (x + 1) / (2 * (x^2 + 2 * x + 2))"
    );
    assert!(
        required.is_empty(),
        "shifted positive quadratic square should not add synthetic required conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
    assert_rendered_antiderivative_verifies(input, &result);

    let input = "integrate(1/(x^2+2*x+5)^2, x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(
        result,
        "1/16 * arctan(1/2 * x + 1/2) + (x + 1) / (8 * (x^2 + 2 * x + 5))"
    );
    assert!(
        required.is_empty(),
        "wide shifted positive quadratic square should not add synthetic required conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
    assert_rendered_antiderivative_verifies(input, &result);

    let residual = "diff(integrate(1/(x^2+2*x+5)^2, x), x) - 1/(x^2+2*x+5)^2";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert!(
        residual_required.is_empty(),
        "nested verification should not invent denominator conditions for a wide shifted positive quadratic square"
    );

    let input = "integrate(1/(4*x^2+1)^2, x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "1/4 * arctan(2 * x) + x / (2 * (4 * x^2 + 1))");
    assert!(
        required.is_empty(),
        "scaled positive quadratic square should not add synthetic required conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
    assert_rendered_antiderivative_verifies(input, &result);

    let direct_diff = "diff(1/4*arctan(2*x)+x/(2*(4*x^2+1)), x)";
    let (direct_diff_result, direct_diff_required) =
        evaluated_expr_with_required_conditions(direct_diff);
    assert_eq!(direct_diff_result, "1 / (4 * x^2 + 1)^2");
    assert!(
        direct_diff_required.is_empty(),
        "compact post-diff presentation should not add synthetic required conditions"
    );

    let residual = "diff(integrate(1/(4*x^2+1)^2, x), x) - 1/(4*x^2+1)^2";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert!(
        residual_required.is_empty(),
        "nested verification should not invent denominator conditions for a scaled positive quadratic square"
    );
}

#[test]
fn integrate_contract_positive_quadratic_cube_uses_recurrence() {
    let input = "integrate(1/(x^2+1)^3, x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(
        result,
        "3/8 * arctan(x) + (3 * x^3 + 5 * x) / (8 * (x^2 + 1)^2)"
    );
    assert!(
        required.is_empty(),
        "positive quadratic cube should not add synthetic required conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
    assert_rendered_antiderivative_verifies(input, &result);
    let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);
    assert!(
        stderr.is_empty(),
        "positive quadratic cube trace should stay quiet\nstderr:\n{stderr}"
    );
    let steps = wire["steps"]
        .as_array()
        .expect("steps should be present with --steps on");
    let integration_step = steps
        .iter()
        .find(|step| step["rule"] == "Calcular la integral")
        .expect("expected public symbolic integration step");
    let substeps = integration_step["substeps"]
        .as_array()
        .expect("integration step should expose didactic substeps");
    let reduction_substep = substeps
        .iter()
        .find(|substep| substep["title"] == "Reducir el cuadrático positivo al cubo")
        .expect("expected positive quadratic cube reduction substep");
    let reduction_latex = reduction_substep["after_latex"]
        .as_str()
        .expect("reduction substep should expose concrete after_latex");
    assert!(
        reduction_latex.contains("\\frac{3}{8\\cdot ({x}^{2} + 1)}")
            && reduction_latex.contains("{({x}^{2} + 1)}^{3}"),
        "reduction should expose the arctan integrand and rational derivative part, got {reduction_latex}"
    );
    assert!(
        substeps
            .iter()
            .any(|substep| substep["title"] == "Integrar la parte arctan y la parte racional"),
        "expected final integration substep, got {substeps:?}"
    );

    let residual = "diff(integrate(1/(x^2+1)^3, x), x) - 1/(x^2+1)^3";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert!(
        residual_required.is_empty(),
        "nested verification should not invent denominator conditions for a positive quadratic cube"
    );

    let input = "integrate(1/((x+1)^2+1)^3, x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(
        result,
        "3/8 * arctan(x + 1) + (3 * x^3 + 9 * x^2 + 14 * x + 8) / (8 * (x^2 + 2 * x + 2)^2)"
    );
    assert!(
        required.is_empty(),
        "shifted positive quadratic cube should not add synthetic required conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
    assert_rendered_antiderivative_verifies(input, &result);
    let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);
    assert!(
        stderr.is_empty(),
        "shifted positive quadratic cube trace should stay quiet\nstderr:\n{stderr}"
    );
    let steps = wire["steps"]
        .as_array()
        .expect("steps should be present with --steps on");
    let integration_step = steps
        .iter()
        .find(|step| step["rule"] == "Calcular la integral")
        .expect("expected shifted positive quadratic cube integration step");
    let substeps = integration_step["substeps"]
        .as_array()
        .expect("shifted positive quadratic cube should expose didactic substeps");
    let reduction_latex = substeps
        .iter()
        .find(|substep| substep["title"] == "Reducir el cuadrático positivo al cubo")
        .and_then(|substep| substep["after_latex"].as_str())
        .expect("expected shifted positive quadratic cube reduction substep");
    assert!(
        reduction_latex.contains(" - \\frac{3\\cdot {x}^{4}")
            && !reduction_latex.contains("\\frac{-"),
        "shifted cube reduction should carry the negative rational sign outside the fraction, got {reduction_latex}"
    );

    let residual = "diff(integrate(1/((x+1)^2+1)^3, x), x) - 1/((x+1)^2+1)^3";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert!(
        residual_required.is_empty(),
        "nested verification should not invent denominator conditions for a shifted positive quadratic cube"
    );
    let direct_diff = "diff(integrate(1/((x+1)^2+1)^3, x), x)";
    let (direct_wire, direct_stderr) = cli_eval_json_with_stderr(direct_diff);
    assert!(
        direct_stderr.is_empty(),
        "direct shifted positive-quadratic cube diff/integrate should avoid depth overflow: {direct_stderr}"
    );
    let direct_result = direct_wire["result"].as_str().unwrap_or_default();
    assert!(
        direct_result.contains("^3") && !direct_result.contains("x^6"),
        "direct shifted positive-quadratic cube diff/integrate should preserve compact denominator, got {direct_result}"
    );
    assert_eq!(direct_wire["required_display"], serde_json::json!([]));

    let input = "integrate(1/(4*x^2+1)^3, x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(
        result,
        "3/16 * arctan(2 * x) + (12 * x^3 + 5 * x) / (8 * (4 * x^2 + 1)^2)"
    );
    assert!(
        required.is_empty(),
        "scaled positive quadratic cube should not add synthetic required conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
    assert_rendered_antiderivative_verifies(input, &result);

    let residual = "diff(integrate(1/(4*x^2+1)^3, x), x) - 1/(4*x^2+1)^3";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert!(
        residual_required.is_empty(),
        "nested verification should not invent denominator conditions for a scaled positive quadratic cube"
    );

    let input = "integrate(x^2/(x^2+1)^3, x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "1/8 * arctan(x) + (x^3 - x) / (8 * (x^2 + 1)^2)");
    assert!(
        required.is_empty(),
        "quadratic numerator over positive quadratic cube should not add synthetic required conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
    assert_rendered_antiderivative_verifies(input, &result);

    let residual = "diff(integrate(x^2/(x^2+1)^3, x), x) - x^2/(x^2+1)^3";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert!(
        residual_required.is_empty(),
        "nested verification should not invent denominator conditions for a quadratic numerator over a positive quadratic cube"
    );
    let direct_diff = "diff(integrate(x^2/(x^2+2*x+2)^3, x), x)";
    let (direct_wire, direct_stderr) = cli_eval_json_with_stderr(direct_diff);
    assert!(
        direct_stderr.is_empty(),
        "direct quadratic-numerator positive-quadratic cube diff/integrate should avoid depth overflow: {direct_stderr}"
    );
    let direct_result = direct_wire["result"].as_str().unwrap_or_default();
    assert!(
        direct_result.contains("x^2")
            && direct_result.contains("(x^2 + 2·x + 2)^3")
            && !direct_result.contains("x^6"),
        "direct quadratic-numerator positive-quadratic cube diff/integrate should preserve compact denominator, got {direct_result}"
    );
    assert_eq!(direct_wire["required_display"], serde_json::json!([]));

    let input = "integrate((2*x+1)^2/((2*x+1)^2+1)^3, x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(
        result,
        "1/16 * arctan(2 * x + 1) + (2 * x^3 + 3 * x^2 + x) / (4 * (4 * x^2 + 4 * x + 2)^2)"
    );
    assert!(
        required.is_empty(),
        "affine quadratic numerator over positive quadratic cube should not add synthetic required conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
    assert_rendered_antiderivative_verifies(input, &result);

    let residual = "diff(integrate((2*x+1)^2/((2*x+1)^2+1)^3, x), x) - (2*x+1)^2/((2*x+1)^2+1)^3";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert!(
        residual_required.is_empty(),
        "nested verification should not invent denominator conditions for an affine quadratic numerator over a positive quadratic cube"
    );

    let input = "integrate(x^3/(x^2+1)^3, x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "1 / (4 * (x^2 + 1)^2) - 1 / (2 * (x^2 + 1))");
    assert!(
        required.is_empty(),
        "cubic numerator over positive quadratic cube should not add synthetic required conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
    assert_rendered_antiderivative_verifies(input, &result);

    let residual = "diff(integrate(x^3/(x^2+1)^3, x), x) - x^3/(x^2+1)^3";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert!(
        residual_required.is_empty(),
        "nested verification should not invent denominator conditions for a cubic numerator over a positive quadratic cube"
    );

    let input = "integrate(x^4/(x^2+1)^3, x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(
        result,
        "3/8 * arctan(x) + (3 * x^3 + 5 * x) / (8 * (x^2 + 1)^2) - x / (x^2 + 1)"
    );
    assert!(
        required.is_empty(),
        "quartic numerator over positive quadratic cube should not add synthetic required conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
    assert_rendered_antiderivative_verifies(input, &result);

    let residual = "diff(integrate(x^4/(x^2+1)^3, x), x) - x^4/(x^2+1)^3";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert!(
        residual_required.is_empty(),
        "nested verification should not invent denominator conditions for a quartic numerator over a positive quadratic cube"
    );

    let input = "integrate((2*x+1)^4/(((2*x+1)^2+1)^3), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(
        result,
        "3/16 * arctan(2 * x + 1) + (6 * x^3 + 9 * x^2 + 7 * x + 2) / (4 * (4 * x^2 + 4 * x + 2)^2) - (2 * x + 1) / (2 * (4 * x^2 + 4 * x + 2))"
    );
    assert!(
        required.is_empty(),
        "scaled affine quartic numerator over positive quadratic cube should not add synthetic required conditions: {required:?}"
    );
    assert_eq!(
        assert_antiderivative_verifies(input),
        AntiderivativeVerificationRoute::PublicResidual
    );
    assert_rendered_antiderivative_verifies(input, &result);

    let residual =
        "diff(integrate((2*x+1)^4/(((2*x+1)^2+1)^3), x), x) - (2*x+1)^4/(((2*x+1)^2+1)^3)";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert!(
        residual_required.is_empty(),
        "nested verification should not invent denominator conditions for a scaled affine quartic numerator over a positive quadratic cube"
    );
}

#[test]
fn integrate_contract_improper_positive_quadratic_uses_polynomial_division() {
    let input = "integrate((x^3+x+1)/(x^2+1), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "arctan(x) + x^2 / 2");
    assert!(
        required.is_empty(),
        "positive quadratic denominator should not add synthetic required conditions: {required:?}"
    );

    let residual = "diff(integrate((x^3+x+1)/(x^2+1), x), x) - (x^3+x+1)/(x^2+1)";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert!(
        residual_required.is_empty(),
        "nested verification should not invent denominator conditions for a positive quadratic"
    );
}

#[test]
fn integrate_contract_direct_reciprocal_uses_abs_log_and_nonzero_domain() {
    let (result, required) = evaluated_integral_with_required_conditions("integrate(1/x, x)");

    assert_eq!(result, "ln(|x|)");
    assert_eq!(
        required,
        vec!["x ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}

#[test]
fn integrate_contract_linear_power_substitution() {
    assert_eq!(simplified_integral("integrate((3*x)^2, x)"), "3 * x^3");
}

#[test]
fn integrate_contract_arctan_kernel() {
    assert_eq!(simplified_integral("integrate(1/(x^2+1), x)"), "arctan(x)");
}

#[test]
fn integrate_contract_arctan_sqrt_kernel_inverts_diff_output() {
    let input = "integrate(1/(2*sqrt(x)*(x+1)), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "arctan(sqrt(x))");
    assert_eq!(
        required,
        vec!["x > 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    let step_rules = evaluated_integral_step_rules(input);
    assert_eq!(
        step_rules,
        vec!["Symbolic Integration".to_string()],
        "arctan sqrt reciprocal kernel should integrate directly without pre-expanding the denominator: {step_rules:?}"
    );
    assert_antiderivative_verifies(input);
    let (nested_derivative, nested_required) =
        evaluated_expr_with_required_conditions("diff(integrate(1/(2*sqrt(x)*(x+1)), x), x)");
    assert_eq!(nested_derivative, "1 / (2 * sqrt(x) * (x + 1))");
    assert_eq!(
        nested_required,
        vec!["x > 0".to_string()],
        "nested arctan sqrt derivative should preserve the positive radicand condition"
    );
    let (nested_residual, nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate(1/(2*sqrt(x)*(x+1)), x), x) - 1/(2*sqrt(x)*(x+1))",
    );
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["x > 0".to_string()],
        "nested arctan sqrt verification should preserve the positive radicand condition"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(1/(sqrt(x)*(x+1)), x)");
    assert_eq!(result, "2 * arctan(sqrt(x))");
    assert_eq!(
        required,
        vec!["x > 0".to_string()],
        "unexpected scaled required_conditions: {required:?}"
    );
    let step_rules = evaluated_integral_step_rules("integrate(1/(sqrt(x)*(x+1)), x)");
    assert_eq!(
        step_rules,
        vec!["Symbolic Integration".to_string()],
        "scaled arctan sqrt reciprocal kernel should integrate directly without pre-expanding the denominator: {step_rules:?}"
    );
    assert_antiderivative_verifies("integrate(1/(sqrt(x)*(x+1)), x)");

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(1/(sqrt(x)*(4*x+1)), x)");
    assert_eq!(result, "arctan(2 * sqrt(x))");
    assert_eq!(
        required,
        vec!["x > 0".to_string()],
        "unexpected scaled linear required_conditions: {required:?}"
    );
    assert_antiderivative_verifies("integrate(1/(sqrt(x)*(4*x+1)), x)");
    let (nested_derivative, nested_required) =
        evaluated_expr_with_required_conditions("diff(integrate(1/(sqrt(x)*(4*x+1)), x), x)");
    assert_eq!(nested_derivative, "1 / (sqrt(x) * (4 * x + 1))");
    assert_eq!(
        nested_required,
        vec!["x > 0".to_string()],
        "scaled linear nested derivative should preserve the positive radicand condition"
    );
    let (nested_residual, nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate(1/(sqrt(x)*(4*x+1)), x), x) - 1/(sqrt(x)*(4*x+1))",
    );
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["x > 0".to_string()],
        "scaled linear verification should preserve the positive radicand condition"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(1/(sqrt(x)*(x+4)), x)");
    assert_eq!(result, "arctan(sqrt(x) / 2)");
    assert_eq!(
        required,
        vec!["x > 0".to_string()],
        "unexpected offset linear required_conditions: {required:?}"
    );
    assert_antiderivative_verifies("integrate(1/(sqrt(x)*(x+4)), x)");

    let input = "integrate(1/(sqrt(4*x+1)*(2*x+1)), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);
    assert_eq!(result, "arctan(sqrt(4 * x + 1))");
    assert_eq!(
        required,
        vec!["x > -1/4".to_string()],
        "unexpected affine required_conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
    let (nested_derivative, nested_required) =
        evaluated_expr_with_required_conditions("diff(integrate(1/(sqrt(4*x+1)*(2*x+1)), x), x)");
    assert!(
        !nested_derivative.contains("integrate("),
        "nested derivative should not leave an integration residual: {nested_derivative}"
    );
    assert_eq!(
        nested_required,
        vec!["x > -1/4".to_string()],
        "affine nested derivative should preserve the positive radicand condition"
    );
    let (nested_residual, nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate(1/(sqrt(4*x+1)*(2*x+1)), x), x) - 1/(sqrt(4*x+1)*(2*x+1))",
    );
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["x > -1/4".to_string()],
        "affine verification should preserve the positive radicand condition"
    );

    let input = "integrate(-1/(2*sqrt(5-3*x)*(2-x)), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);
    assert_eq!(result, "arctan(sqrt(5 - 3 * x))");
    assert_eq!(
        required,
        vec!["x < 5/3".to_string()],
        "unexpected negative-slope affine required_conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
    assert_antiderivative_equiv_verifies(input);
    let (nested_residual, nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate(-1/(2*sqrt(5-3*x)*(2-x)), x), x) - (-1/(2*sqrt(5-3*x)*(2-x)))",
    );
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["x < 5/3".to_string()],
        "negative-slope affine residual verification should preserve the positive radicand condition"
    );
    let (nested_equiv, nested_required) = evaluated_equiv_with_required_conditions(
        "diff(integrate(-1/(2*sqrt(5-3*x)*(2-x)), x), x)",
        "-1/(2*sqrt(5-3*x)*(2-x))",
    );
    assert!(nested_equiv);
    assert_eq!(
        nested_required,
        vec!["x < 5/3".to_string()],
        "negative-slope affine derivative equivalence should preserve the positive radicand condition"
    );

    let (nested_equiv, nested_required) = evaluated_equiv_with_required_conditions(
        "diff(integrate(-1/(2*sqrt(1-x)*(2-x)), x), x)",
        "-1/(2*sqrt(1-x)*(2-x))",
    );
    assert!(
        nested_equiv,
        "public equivalence should accept the directly simplified zero residual"
    );
    assert_eq!(
        nested_required,
        vec!["x < 1".to_string()],
        "unit-slope negative affine derivative equivalence should preserve the positive radicand condition"
    );
}

#[test]
fn integrate_contract_arctan_sqrt_unit_shift_square_inverts_diff_output() {
    let input = "integrate(1/(sqrt(x)*(x+1)^2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "arctan(sqrt(x)) + sqrt(x) / (x + 1)");
    assert_eq!(
        required,
        vec!["x > 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    let step_rules = evaluated_integral_step_rules(input);
    assert!(
        step_rules
            .iter()
            .any(|rule| rule == "Symbolic Integration"),
        "unit-shift square arctan sqrt reciprocal kernel should reach symbolic integration: {step_rules:?}"
    );
    assert_rendered_antiderivative_verifies(input, &result);

    let scaled_input = "integrate(1/(2*sqrt(x)*(x+1)^2), x)";
    let (scaled_result, scaled_required) =
        evaluated_integral_with_required_conditions(scaled_input);
    assert_eq!(scaled_result, "1/2 * (arctan(sqrt(x)) + sqrt(x) / (x + 1))");
    assert_eq!(
        scaled_required,
        vec!["x > 0".to_string()],
        "unexpected scaled required_conditions: {scaled_required:?}"
    );
    assert_eq!(
        integrate_call_antiderivative_residual_result(scaled_input),
        "0"
    );
    assert_rendered_antiderivative_verifies(scaled_input, &scaled_result);

    let shifted_input = "integrate(1/(sqrt(x)*(x+4)^2), x)";
    let (shifted_result, shifted_required) =
        evaluated_integral_with_required_conditions(shifted_input);
    assert_eq!(
        shifted_result,
        "1/8 * arctan(sqrt(x) / 2) + sqrt(x) / (4 * (x + 4))"
    );
    assert_eq!(
        shifted_required,
        vec!["x > 0".to_string()],
        "unexpected shifted required_conditions: {shifted_required:?}"
    );
    assert_eq!(
        integrate_call_antiderivative_residual_result(shifted_input),
        "0"
    );
    assert_eq!(
        assert_antiderivative_verifies(shifted_input),
        AntiderivativeVerificationRoute::PublicResidual
    );
    assert_rendered_antiderivative_verifies(shifted_input, &shifted_result);

    let rational_shift_input = "integrate(1/(sqrt(x)*(x+1/4)^2), x)";
    let (rational_shift_result, rational_shift_required) =
        evaluated_integral_with_required_conditions(rational_shift_input);
    assert_eq!(
        rational_shift_result,
        "8 * arctan(2 * sqrt(x)) + 4 * sqrt(x) / (x + 1/4)"
    );
    assert_eq!(
        rational_shift_required,
        vec!["x > 0".to_string()],
        "unexpected rational shift required_conditions: {rational_shift_required:?}"
    );
    assert_eq!(
        integrate_call_antiderivative_residual_result(rational_shift_input),
        "0"
    );
    assert_rendered_antiderivative_verifies(rational_shift_input, &rational_shift_result);
    let (rational_shift_displayed_derivative, rational_shift_displayed_required) =
        evaluated_expr_with_required_conditions("diff(8*arctan(2*sqrt(x)) + 4*sqrt(x)/(x+1/4), x)");
    assert_eq!(
        rational_shift_displayed_derivative,
        "1 / ((x + 1/4)^2 * sqrt(x))"
    );
    assert_eq!(
        rational_shift_displayed_required,
        vec!["x > 0".to_string()],
        "displayed rational-shift derivative should preserve the positive radicand condition"
    );

    let externally_scaled_rational_shift_input = "integrate(1/(3*sqrt(x)*(x+1/4)^2), x)";
    let (externally_scaled_rational_shift_result, externally_scaled_rational_shift_required) =
        evaluated_integral_with_required_conditions(externally_scaled_rational_shift_input);
    assert_eq!(
        externally_scaled_rational_shift_result,
        "8/3 * arctan(2 * sqrt(x)) + 4/3 * sqrt(x) / (x + 1/4)"
    );
    assert_eq!(
        externally_scaled_rational_shift_required,
        vec!["x > 0".to_string()],
        "unexpected externally scaled rational shift required_conditions: {externally_scaled_rational_shift_required:?}"
    );
    assert_eq!(
        integrate_call_antiderivative_residual_result(externally_scaled_rational_shift_input),
        "0"
    );
    assert_rendered_antiderivative_verifies(
        externally_scaled_rational_shift_input,
        &externally_scaled_rational_shift_result,
    );
}

#[test]
fn integrate_contract_inverse_hyperbolic_sqrt_reciprocal_kernels_invert_diff_output() {
    for (input, expected, expected_required) in [
        (
            "integrate(-1/(2*x*sqrt(x+1)), x)",
            "asinh(sqrt(1 / x))",
            vec!["x > 0".to_string()],
        ),
        (
            "integrate(-1/(x*sqrt(x+4)), x)",
            "asinh(sqrt(4 / x))",
            vec!["x > 0".to_string()],
        ),
        (
            "integrate(-1/(2*(x+1)*sqrt(x+2)), x)",
            "asinh(sqrt(1 / (x + 1)))",
            vec!["x > -1".to_string()],
        ),
        (
            "integrate(-1/(2*sqrt(x)*(x-1)), x)",
            "atanh(sqrt(1 / x))",
            vec!["x > 1".to_string()],
        ),
        (
            "integrate(-1/(sqrt(x)*(x-4)), x)",
            "atanh(sqrt(4 / x))",
            vec!["x > 4".to_string()],
        ),
        (
            "integrate(3/(2*sqrt(3*x)*(3-x)), x)",
            "atanh(sqrt(3 / x))",
            vec!["x > 3".to_string()],
        ),
        (
            "integrate(-3/(2*sqrt(3*x+1)*(3*x)), x)",
            "atanh(sqrt(1 / (3 * x + 1)))",
            vec!["x > 0".to_string()],
        ),
        (
            "integrate(-2/((2*x+1)*sqrt(2*x+5)), x)",
            "asinh(sqrt(4 / (2 * x + 1)))",
            vec!["x > -1/2".to_string()],
        ),
        (
            "integrate(-1/(sqrt(2)*(x+3)*sqrt(x+5)), x)",
            "asinh(sqrt(2 / (x + 3)))",
            vec!["x > -3".to_string()],
        ),
        (
            "integrate(-2/(sqrt(2)*(2*x+1)*sqrt(2*x+3)), x)",
            "asinh(sqrt(2 / (2 * x + 1)))",
            vec!["x > -1/2".to_string()],
        ),
        (
            "integrate(1/((6-2*x)*sqrt(8-2*x)), x)",
            "1/2 * asinh(sqrt(1 / (3 - x))) * sqrt(2)",
            vec!["x < 3".to_string()],
        ),
        (
            "integrate(-1/(x*sqrt(2*x+4)), x)",
            "atanh(sqrt(2 / (x + 2)))",
            vec!["x > 0".to_string()],
        ),
    ] {
        let (result, required) = evaluated_integral_with_required_conditions(input);
        assert_eq!(result, expected, "input: {input}");
        assert_eq!(
            required, expected_required,
            "unexpected required_conditions for {input}: {required:?}"
        );
        assert_antiderivative_verifies(input);

        let nested = format!("diff({input}, x)");
        let (nested_derivative, nested_required) = evaluated_expr_with_required_conditions(&nested);
        assert!(
            !nested_derivative.contains("integrate("),
            "nested derivative should not leave an integration residual for {input}: {nested_derivative}"
        );
        assert_eq!(
            nested_required, expected_required,
            "nested derivative should preserve domain conditions for {input}"
        );
    }
}

#[test]
fn integrate_contract_inverse_hyperbolic_sqrt_reciprocal_kernels_integrate_directly_without_denominator_expansion(
) {
    for input in [
        "integrate(-1/(2*sqrt(x)*(x-1)), x)",
        "integrate(-1/(sqrt(x)*(x-4)), x)",
    ] {
        let step_rules = evaluated_integral_step_rules(input);
        assert_eq!(
            step_rules,
            vec!["Symbolic Integration".to_string()],
            "inverse-hyperbolic sqrt reciprocal kernel should integrate directly without pre-expanding the denominator: {step_rules:?}"
        );
        assert_antiderivative_verifies(input);
    }
}

#[test]
fn integrate_contract_scaled_asinh_residual_stays_quiet_on_stderr() {
    let input = concat!(
        "diff(integrate(1/((6-2*x)*sqrt(8-2*x)), x), x) ",
        "- 1/((6-2*x)*sqrt(8-2*x))"
    );
    let (wire, stderr) = cli_eval_json_with_stderr(input);

    assert_eq!(wire["result"], "0");
    assert_eq!(
        wire["required_display"],
        serde_json::json!(["x < 3"]),
        "scaled asinh residual should preserve the antiderivative domain condition"
    );
    assert!(
        !stderr.contains("depth_overflow"),
        "scaled asinh residual should not emit depth_overflow to stderr, got: {stderr}"
    );
}

#[test]
fn integrate_contract_ambiguous_inverse_hyperbolic_family_selection_verifies_by_diff() {
    let input = "integrate(3/(sqrt(5-3*x)*(1-3*x)), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "asinh(sqrt(4 / (1 - 3 * x)))");
    assert_eq!(
        required,
        vec!["x < 1/3".to_string()],
        "ambiguous inverse-hyperbolic primitive should keep the witnessed positive denominator"
    );
    assert_antiderivative_verifies(input);
    assert_antiderivative_equiv_verifies(input);

    let (nested_residual, nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate(3/(sqrt(5-3*x)*(1-3*x)), x), x) - 3/(sqrt(5-3*x)*(1-3*x))",
    );
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["x < 1/3".to_string()],
        "nested ambiguous primitive verification should preserve the selected-domain condition"
    );
}

#[test]
fn integrate_contract_polynomial_derivative_arctan_substitution() {
    assert_eq!(
        simplified_integral("integrate(2*x/(1+x^4), x)"),
        "arctan(x^2)"
    );
}

#[test]
fn integrate_contract_scaled_polynomial_derivative_arctan_substitution() {
    assert_eq!(
        simplified_integral("integrate(2*x/(4+x^4), x)"),
        "1/2 * arctan(x^2 / 2)"
    );
}

#[test]
fn integrate_contract_scaled_affine_arctan_kernel_survives_quadratic_normalization() {
    let input = "integrate(2/(1+(2*x+1)^2), x)";
    assert_eq!(simplified_integral(input), "arctan(2 * x + 1)");
    assert_antiderivative_verifies(input);
}

#[test]
fn integrate_contract_arctan_scaled_variable_by_parts() {
    let (result, required) = evaluated_integral_with_required_conditions("integrate(arctan(x), x)");
    assert_eq!(result, "-1/2 * ln(x^2 + 1) + x * arctan(x)");
    assert!(
        required.is_empty(),
        "arctan integration should not add required conditions: {required:?}"
    );
    assert_antiderivative_verifies("integrate(arctan(x), x)");

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(arctan(2*x), x)");
    assert_eq!(result, "-1/4 * ln((2 * x)^2 + 1) + x * arctan(2 * x)");
    assert!(
        required.is_empty(),
        "scaled arctan integration should not add required conditions: {required:?}"
    );
    assert_antiderivative_verifies("integrate(arctan(2*x), x)");
    assert_rendered_antiderivative_verifies("integrate(arctan(2*x), x)", &result);

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(arctan(2*x+1), x)");
    assert_eq!(
        result,
        "-1/4 * ln((2 * x + 1)^2 + 1) + 1/2 * (2 * x + 1) * arctan(2 * x + 1)"
    );
    assert!(
        required.is_empty(),
        "shifted arctan integration should not add required conditions: {required:?}"
    );
    assert_antiderivative_verifies("integrate(arctan(2*x+1), x)");
    assert_rendered_antiderivative_verifies("integrate(arctan(2*x+1), x)", &result);

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(arctan(1-2*x), x)");
    assert_eq!(
        result,
        "1/4 * ln((1 - 2 * x)^2 + 1) + -1/2 * (1 - 2 * x) * arctan(1 - 2 * x)"
    );
    assert!(
        required.is_empty(),
        "negative-slope shifted arctan integration should not add required conditions: {required:?}"
    );
    assert_antiderivative_verifies("integrate(arctan(1-2*x), x)");
    assert_rendered_antiderivative_verifies("integrate(arctan(1-2*x), x)", &result);

    let (wire, _stderr) = cli_eval_json_with_stderr("integrate(arctan(1-2*x), x)");
    assert_eq!(
        wire["result"],
        "1/4·ln((1 - 2·x)^2 + 1) - 1/2·(1 - 2·x)·arctan(1 - 2·x)"
    );
    assert!(
        !wire["result"].as_str().unwrap_or_default().contains("+ -"),
        "public negative-slope shifted arctan integration should compact adjacent signs: {}",
        wire["result"]
    );
}

#[test]
fn integrate_contract_polynomial_times_arctan_affine_by_parts() {
    for (input, fragments) in [
        (
            "integrate(x*arctan(x), x)",
            vec!["arctan(x) * (x^2 + 1)", "- x"],
        ),
        (
            "integrate(x^2*arctan(x), x)",
            vec!["x^3 * arctan(x)", "ln(x^2 + 1)"],
        ),
        (
            "integrate(x^3*arctan(x), x)",
            vec!["arctan(x) * (3 * x^4 - 3)", "3 * x - x^3"],
        ),
        (
            "integrate(x^4*arctan(x), x)",
            vec!["x^5 * arctan(x)", "x^4", "ln(x^2 + 1)"],
        ),
        (
            "integrate(x^5*arctan(x), x)",
            vec!["arctan(x) * (15 * x^6 + 15)", "5 * x^3 - 3 * x^5 - 15 * x"],
        ),
        (
            "integrate(x^6*arctan(x), x)",
            vec!["x^7 * arctan(x)", "x^6", "ln(x^2 + 1)"],
        ),
        (
            "integrate(x^2*arctan(x+1), x)",
            vec!["arctan(x + 1) * (2 * x^3 - 4)", "ln(x^2 + 2 * x + 2)"],
        ),
        (
            "integrate(x^3*arctan(x+1), x)",
            vec!["arctan(x + 1) * (3 * x^4 + 12)", "3 * x^2 - x^3 - 6 * x"],
        ),
        (
            "integrate(x^2*arctan(1-x), x)",
            vec!["arctan(1 - x) * (1/3 * x^3 + 2/3)", "ln(x^2 + 2 - 2 * x)"],
        ),
    ] {
        let (result, required) = evaluated_integral_with_required_conditions(input);
        assert!(
            !result.starts_with("integrate("),
            "expected polynomial-arctan affine product to integrate, got {result}"
        );
        for fragment in fragments {
            assert!(
                result.contains(fragment),
                "expected `{fragment}` in polynomial-arctan antiderivative for {input}, got {result}"
            );
        }
        assert!(
            !result.contains("1/3 * x^2 / 2"),
            "polynomial-arctan presentation should not keep nested polynomial fractions, got {result}"
        );
        assert!(
            required.is_empty(),
            "polynomial-arctan integration should not add required conditions: {required:?}"
        );
        assert_antiderivative_verifies(input);
    }

    let (linear_arctan_result, _) =
        evaluated_integral_with_required_conditions("integrate(x*arctan(x), x)");
    assert!(
        linear_arctan_result.matches("arctan(x)").nth(1).is_none(),
        "linear polynomial-arctan by-parts presentation should collect repeated arctan terms, got {linear_arctan_result}"
    );
    let (cubic_arctan_result, _) =
        evaluated_integral_with_required_conditions("integrate(x^3*arctan(x), x)");
    assert!(
        cubic_arctan_result.matches("arctan(x)").nth(1).is_none(),
        "cubic polynomial-arctan by-parts presentation should collect repeated arctan terms, got {cubic_arctan_result}"
    );
    let (quintic_arctan_result, _) =
        evaluated_integral_with_required_conditions("integrate(x^5*arctan(x), x)");
    assert!(
        quintic_arctan_result.matches("arctan(x)").nth(1).is_none(),
        "quintic polynomial-arctan by-parts presentation should collect repeated arctan terms, got {quintic_arctan_result}"
    );
    let (shifted_quadratic_arctan_result, _) =
        evaluated_integral_with_required_conditions("integrate(x^2*arctan(x+1), x)");
    assert!(
        shifted_quadratic_arctan_result
            .matches("arctan(x + 1)")
            .nth(1)
            .is_none(),
        "shifted quadratic polynomial-arctan by-parts presentation should collect repeated arctan terms, got {shifted_quadratic_arctan_result}"
    );
    let (shifted_cubic_arctan_result, _) =
        evaluated_integral_with_required_conditions("integrate(x^3*arctan(x+1), x)");
    assert!(
        shifted_cubic_arctan_result
            .matches("arctan(x + 1)")
            .nth(1)
            .is_none(),
        "shifted cubic polynomial-arctan by-parts presentation should collect repeated arctan terms, got {shifted_cubic_arctan_result}"
    );

    for input in [
        "integrate(x*arctan(x), x)",
        "integrate(x^2*arctan(x), x)",
        "integrate(x^3*arctan(x), x)",
        "integrate(x^4*arctan(x), x)",
        "integrate(x^5*arctan(x), x)",
        "integrate(x^6*arctan(x), x)",
        "integrate(x^2*arctan(x+1), x)",
        "integrate(x^3*arctan(x+1), x)",
        "integrate(x^2*arctan(1-x), x)",
    ] {
        let (wire, _) = cli_eval_json_with_stderr(input);
        let result = wire["result"].as_str().expect("result string");
        assert!(
            !result.contains(" - ("),
            "public polynomial-arctan by-parts presentation should flatten subtracting a difference, got {result}"
        );
        let result_latex = wire["result_latex"].as_str().expect("result_latex string");
        assert!(
            !result_latex.contains(" - ("),
            "public polynomial-arctan by-parts LaTeX should flatten subtracting a difference, got {result_latex}"
        );
    }

    let (negative_shifted_result, _) =
        evaluated_integral_with_required_conditions("integrate(x^2*arctan(1-x), x)");
    assert_eq!(
        negative_shifted_result.matches("arctan(1 - x)").count(),
        1,
        "negative-shifted polynomial-arctan presentation should collect repeated arctan terms, got {negative_shifted_result}"
    );
    assert!(
        negative_shifted_result.contains("1/3 * x^3 + 2/3")
            && !negative_shifted_result.contains("arctan(x - 1)"),
        "negative-shifted polynomial-arctan presentation should orient correction terms toward the input argument, got {negative_shifted_result}"
    );

    let (negative_expanded_result, _) =
        evaluated_integral_with_required_conditions("integrate((x^2+x)*arctan(1-x), x)");
    assert_eq!(
        negative_expanded_result.matches("arctan(1 - x)").count(),
        1,
        "expanded negative-shifted polynomial-arctan presentation should collect repeated arctan terms, got {negative_expanded_result}"
    );
    assert_eq!(
        negative_expanded_result.matches("ln(x^2 + 2 - 2 * x)").count(),
        1,
        "expanded negative-shifted polynomial-arctan presentation should collect repeated log companions, got {negative_expanded_result}"
    );
    assert!(
        negative_expanded_result.contains("5/6 * ln(x^2 + 2 - 2 * x)")
            && negative_expanded_result.contains("7/6 * x")
            && !negative_expanded_result.contains("arctan(x - 1)"),
        "expanded negative-shifted polynomial-arctan presentation should keep compact companions and input orientation, got {negative_expanded_result}"
    );

    let residual = "diff(integrate(x^6*arctan(x), x), x) - x^6*arctan(x)";
    let (wire, stderr) = cli_eval_json_with_stderr(residual);
    assert_eq!(wire["result"].as_str().unwrap_or_default(), "0");
    assert!(
        !stderr.contains("depth_overflow"),
        "degree-six arctan by-parts residual should not emit depth_overflow warning\nstderr:\n{stderr}"
    );

    let shifted_residual = "diff(integrate(x^2*arctan(x+1), x), x) - x^2*arctan(x+1)";
    let (wire, stderr) = cli_eval_json_with_stderr(shifted_residual);
    assert_eq!(wire["result"], "0");
    assert!(
        stderr.is_empty(),
        "shifted polynomial-arctan residual should stay quiet\nstderr:\n{stderr}"
    );
    assert_eq!(wire["required_display"], serde_json::json!([]));

    let shifted_cubic_residual = "diff(integrate(x^3*arctan(x+1), x), x) - x^3*arctan(x+1)";
    let (wire, stderr) = cli_eval_json_with_stderr(shifted_cubic_residual);
    assert_eq!(wire["result"], "0");
    assert!(
        stderr.is_empty(),
        "shifted cubic polynomial-arctan residual should stay quiet\nstderr:\n{stderr}"
    );
    assert_eq!(wire["required_display"], serde_json::json!([]));

    let expanded_residual =
        "diff(integrate(x^2*arctan(x+1)+x*arctan(x+1), x), x) - (x^2*arctan(x+1)+x*arctan(x+1))";
    let (wire, stderr) = cli_eval_json_with_stderr(expanded_residual);
    assert_eq!(wire["result"], "0");
    assert!(
        stderr.is_empty(),
        "expanded shifted polynomial-arctan residual should stay quiet\nstderr:\n{stderr}"
    );
    assert_eq!(wire["required_display"], serde_json::json!([]));

    let negative_shifted_residual = "diff(integrate(x^2*arctan(1-x), x), x) - x^2*arctan(1-x)";
    let (wire, stderr) = cli_eval_json_with_stderr(negative_shifted_residual);
    assert_eq!(wire["result"], "0");
    assert!(
        stderr.is_empty(),
        "negative-shifted polynomial-arctan residual should stay quiet\nstderr:\n{stderr}"
    );
    assert_eq!(wire["required_display"], serde_json::json!([]));

    let compact_negative_shifted_residual =
        "diff(((x^3+2)*arctan(1-x))/3 + ln(x^2+2-2*x)/3 + x^2/6 + 2*x/3, x) - x^2*arctan(1-x)";
    let (wire, stderr) = cli_eval_json_with_stderr(compact_negative_shifted_residual);
    assert_eq!(wire["result"], "0");
    assert!(
        stderr.is_empty(),
        "explicit compact negative-shifted polynomial-arctan residual should stay quiet\nstderr:\n{stderr}"
    );
    assert_eq!(wire["required_display"], serde_json::json!([]));

    let negative_expanded_residual =
        "diff(integrate(x^2*arctan(1-x)+x*arctan(1-x), x), x) - (x^2*arctan(1-x)+x*arctan(1-x))";
    let (wire, stderr) = cli_eval_json_with_stderr(negative_expanded_residual);
    assert_eq!(wire["result"], "0");
    assert!(
        stderr.is_empty(),
        "expanded negative-shifted polynomial-arctan residual should stay quiet\nstderr:\n{stderr}"
    );
    assert_eq!(wire["required_display"], serde_json::json!([]));
}

#[test]
fn integrate_contract_bounded_inverse_trig_variable_by_parts() {
    let (result, required) = evaluated_integral_with_required_conditions("integrate(arcsin(x), x)");
    assert_eq!(result, "sqrt(1 - x^2) + x * arcsin(x)");
    assert_eq!(
        required,
        vec!["-1 < x < 1".to_string()],
        "arcsin integration should publish its open-domain condition"
    );
    assert_antiderivative_verifies("integrate(arcsin(x), x)");
    let (nested_residual, nested_required) =
        evaluated_expr_with_required_conditions("diff(integrate(arcsin(x), x), x) - arcsin(x)");
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["-1 < x < 1".to_string()],
        "nested arcsin verification should preserve the open-domain condition"
    );

    let (result, required) = evaluated_integral_with_required_conditions("integrate(arccos(x), x)");
    assert_eq!(result, "x * arccos(x) - sqrt(1 - x^2)");
    assert_eq!(
        required,
        vec!["-1 < x < 1".to_string()],
        "arccos integration should publish its open-domain condition"
    );
    assert_antiderivative_verifies("integrate(arccos(x), x)");
    let (nested_residual, nested_required) =
        evaluated_expr_with_required_conditions("diff(integrate(arccos(x), x), x) - arccos(x)");
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["-1 < x < 1".to_string()],
        "nested arccos verification should preserve the open-domain condition"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(arcsin(2*x), x)");
    assert_eq!(result, "sqrt(1/4 - x^2) + x * arcsin(2 * x)");
    assert_eq!(
        required,
        vec!["-1/2 < x < 1/2".to_string()],
        "scaled arcsin integration should publish its open-domain condition"
    );
    assert_rendered_antiderivative_verifies("integrate(arcsin(2*x), x)", &result);
    let (nested_residual, nested_required) =
        evaluated_expr_with_required_conditions("diff(integrate(arcsin(2*x), x), x) - arcsin(2*x)");
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["-1/2 < x < 1/2".to_string()],
        "nested scaled arcsin verification should preserve the open-domain condition"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(arccos(2*x), x)");
    assert_eq!(result, "x * arccos(2 * x) - sqrt(1/4 - x^2)");
    assert_eq!(
        required,
        vec!["-1/2 < x < 1/2".to_string()],
        "scaled arccos integration should publish its open-domain condition"
    );
    assert_rendered_antiderivative_verifies("integrate(arccos(2*x), x)", &result);
    let (nested_residual, nested_required) =
        evaluated_expr_with_required_conditions("diff(integrate(arccos(2*x), x), x) - arccos(2*x)");
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["-1/2 < x < 1/2".to_string()],
        "nested scaled arccos verification should preserve the open-domain condition"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(arcsin(-2*x), x)");
    assert_eq!(result, "x * arcsin(-2 * x) - 1/2 * sqrt(1 - (-2 * x)^2)");
    assert_eq!(
        required,
        vec!["-1/2 < x < 1/2".to_string()],
        "negative-scale arcsin integration should publish its open-domain condition"
    );
    assert_rendered_antiderivative_verifies("integrate(arcsin(-2*x), x)", &result);
    let (nested_residual, nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate(arcsin(-2*x), x), x) - arcsin(-2*x)",
    );
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["-1/2 < x < 1/2".to_string()],
        "nested negative-scale arcsin verification should preserve the open-domain condition"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(arcsin(2*x+1), x)");
    assert_eq!(
        result,
        "1/2 * sqrt(1 - (2 * x + 1)^2) + 1/2 * (2 * x + 1) * arcsin(2 * x + 1)"
    );
    assert_eq!(
        required,
        vec!["-1 < x < 0".to_string()],
        "shifted positive-slope arcsin integration should publish its open-domain condition"
    );
    assert_antiderivative_verifies("integrate(arcsin(2*x+1), x)");
    assert_rendered_antiderivative_verifies("integrate(arcsin(2*x+1), x)", &result);
    let (nested_residual, nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate(arcsin(2*x+1), x), x) - arcsin(2*x+1)",
    );
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["-1 < x < 0".to_string()],
        "nested shifted arcsin verification should preserve the open-domain condition"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(arccos(2*x+1), x)");
    assert_eq!(
        result,
        "1/2 * (2 * x + 1) * arccos(2 * x + 1) - 1/2 * sqrt(1 - (2 * x + 1)^2)"
    );
    assert_eq!(
        required,
        vec!["-1 < x < 0".to_string()],
        "shifted positive-slope arccos integration should publish its open-domain condition"
    );
    assert_antiderivative_verifies("integrate(arccos(2*x+1), x)");
    assert_rendered_antiderivative_verifies("integrate(arccos(2*x+1), x)", &result);
    let (nested_residual, nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate(arccos(2*x+1), x), x) - arccos(2*x+1)",
    );
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["-1 < x < 0".to_string()],
        "nested shifted arccos verification should preserve the open-domain condition"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(arcsin(2*x-1), x)");
    assert_eq!(
        result,
        "1/2 * sqrt(1 - (2 * x - 1)^2) + 1/2 * (2 * x - 1) * arcsin(2 * x - 1)"
    );
    assert_eq!(
        required,
        vec!["0 < x < 1".to_string()],
        "opposite-offset positive-slope arcsin integration should publish its open-domain condition"
    );
    assert_antiderivative_verifies("integrate(arcsin(2*x-1), x)");
    assert_rendered_antiderivative_verifies("integrate(arcsin(2*x-1), x)", &result);

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(arcsin(1-2*x), x)");
    assert_eq!(
        result,
        "-1/2 * (4 * (x - x^2))^(1/2) - 1/2 * (1 - 2 * x) * arcsin(1 - 2 * x)"
    );
    assert_eq!(
        required,
        vec!["0 < x < 1".to_string()],
        "negative-slope shifted arcsin integration should publish its open-domain condition"
    );
    assert_antiderivative_verifies("integrate(arcsin(1-2*x), x)");
    assert_rendered_antiderivative_verifies("integrate(arcsin(1-2*x), x)", &result);
    let (nested_residual, nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate(arcsin(1-2*x), x), x) - arcsin(1-2*x)",
    );
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["0 < x < 1".to_string()],
        "nested negative-slope arcsin verification should preserve the open-domain condition"
    );

    assert_eq!(
        simplified_integral("integrate(arcsin(a*x+1), x)"),
        "integrate(arcsin(a * x + 1), x)",
        "symbolic-scale bounded inverse-trig integration remains intentionally deferred"
    );
}

#[test]
fn integrate_contract_arccos_negative_slope_preserves_compact_by_parts_form() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(arccos(1-2*x), x)");
    assert_eq!(
        result,
        "1/2 * (4 * (x - x^2))^(1/2) - 1/2 * (1 - 2 * x) * arccos(1 - 2 * x)"
    );
    assert_eq!(
        required,
        vec!["0 < x < 1".to_string()],
        "negative-slope shifted arccos integration should publish its open-domain condition"
    );
    assert_rendered_antiderivative_verifies("integrate(arccos(1-2*x), x)", &result);
    let (nested_residual, nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate(arccos(1-2*x), x), x) - arccos(1-2*x)",
    );
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["0 < x < 1".to_string()],
        "nested negative-slope arccos verification should preserve the open-domain condition"
    );
}

#[test]
fn integrate_contract_arctan_reciprocal_scaled_variable_by_parts() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(arctan(1/x), x)");
    assert_eq!(result, "1/2 * (ln(x^2 + 1) + 2 * x * arctan(1 / x))");
    assert_eq!(
        required,
        vec!["x ≠ 0".to_string()],
        "reciprocal arctan integration should preserve the reciprocal nonzero condition"
    );
    assert_antiderivative_verifies("integrate(arctan(1/x), x)");
    assert_rendered_antiderivative_verifies("integrate(arctan(1/x), x)", &result);

    let (result, required) = evaluated_integral_with_required_conditions("integrate(arccot(x), x)");
    assert_eq!(result, "1/2 * (ln(x^2 + 1) + 2 * x * arctan(1 / x))");
    assert_eq!(
        required,
        vec!["x ≠ 0".to_string()],
        "arccot canonicalization should keep the explicit reciprocal condition"
    );
    assert_antiderivative_verifies("integrate(arccot(x), x)");
    assert_rendered_antiderivative_verifies("integrate(arccot(x), x)", &result);

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(arccot(2*x), x)");
    assert_eq!(
        result,
        "1/4 * (ln(4 * x^2 + 1) + 4 * x * arctan(1 / (2 * x)))"
    );
    assert_eq!(
        required,
        vec!["x ≠ 0".to_string()],
        "scaled arccot canonicalization should keep the explicit reciprocal condition"
    );
    assert_antiderivative_verifies("integrate(arccot(2*x), x)");
    assert_rendered_antiderivative_verifies("integrate(arccot(2*x), x)", &result);

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(arctan(1/(2*x+1)), x)");
    assert_eq!(
        result,
        "1/4 * ln((2 * x + 1)^2 + 1) + 1/2 * (2 * x + 1) * arctan(1 / (2 * x + 1))"
    );
    assert_eq!(
        required,
        vec!["x ≠ -1/2".to_string()],
        "shifted reciprocal arctan integration should preserve the affine reciprocal condition"
    );
    assert_antiderivative_verifies("integrate(arctan(1/(2*x+1)), x)");
    assert_rendered_antiderivative_verifies("integrate(arctan(1/(2*x+1)), x)", &result);
    let (nested_residual, nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate(arctan(1/(2*x+1)), x), x) - arctan(1/(2*x+1))",
    );
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["x ≠ -1/2".to_string()],
        "nested integrate/diff verification should keep the affine reciprocal condition"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(arccot(2*x+1), x)");
    assert_eq!(
        result,
        "1/4 * ln((2 * x + 1)^2 + 1) + 1/2 * (2 * x + 1) * arctan(1 / (2 * x + 1))"
    );
    assert_eq!(
        required,
        vec!["x ≠ -1/2".to_string()],
        "shifted arccot canonicalization should keep the explicit affine reciprocal condition"
    );
    assert_antiderivative_verifies("integrate(arccot(2*x+1), x)");
    assert_rendered_antiderivative_verifies("integrate(arccot(2*x+1), x)", &result);

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(arccot(1-2*x), x)");
    assert_eq!(
        result,
        "-1/2 * (1 - 2 * x) * arctan(1 / (1 - 2 * x)) - 1/4 * ln((1 - 2 * x)^2 + 1)"
    );
    assert_eq!(
        required,
        vec!["x ≠ 1/2".to_string()],
        "negative-slope shifted arccot canonicalization should keep the explicit affine reciprocal condition"
    );
    assert_antiderivative_verifies("integrate(arccot(1-2*x), x)");
    assert_rendered_antiderivative_verifies("integrate(arccot(1-2*x), x)", &result);
}

#[test]
fn integrate_contract_asinh_affine_by_parts() {
    let (result, required) = evaluated_integral_with_required_conditions("integrate(asinh(x), x)");
    assert_eq!(result, "x * asinh(x) - sqrt(x^2 + 1)");
    assert!(
        required.is_empty(),
        "asinh integration should not add required conditions: {required:?}"
    );
    assert_antiderivative_verifies("integrate(asinh(x), x)");
    assert_rendered_antiderivative_verifies("integrate(asinh(x), x)", &result);

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(asinh(2*x), x)");
    assert_eq!(result, "x * asinh(2 * x) - 1/2 * sqrt((2 * x)^2 + 1)");
    assert!(
        required.is_empty(),
        "scaled asinh integration should not add required conditions: {required:?}"
    );
    assert_antiderivative_verifies("integrate(asinh(2*x), x)");
    assert_rendered_antiderivative_verifies("integrate(asinh(2*x), x)", &result);

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(asinh(2*x+1), x)");
    assert_eq!(
        result,
        "1/2 * (2 * x + 1) * asinh(2 * x + 1) - 1/2 * sqrt((2 * x + 1)^2 + 1)"
    );
    assert!(
        required.is_empty(),
        "affine asinh integration should not add required conditions: {required:?}"
    );
    assert_antiderivative_verifies("integrate(asinh(2*x+1), x)");
    assert_rendered_antiderivative_verifies("integrate(asinh(2*x+1), x)", &result);
    let (nested_residual, nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate(asinh(2*x+1), x), x) - asinh(2*x+1)",
    );
    assert_eq!(nested_residual, "0");
    assert!(
        nested_required.is_empty(),
        "nested shifted asinh verification should not add required conditions: {nested_required:?}"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(asinh(1-2*x), x)");
    assert_eq!(
        result,
        "1/2 * sqrt((1 - 2 * x)^2 + 1) - 1/2 * (1 - 2 * x) * asinh(1 - 2 * x)"
    );
    assert!(
        required.is_empty(),
        "negative-slope affine asinh integration should not add required conditions: {required:?}"
    );
    assert_antiderivative_verifies("integrate(asinh(1-2*x), x)");
    assert_rendered_antiderivative_verifies("integrate(asinh(1-2*x), x)", &result);
    let (nested_residual, nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate(asinh(1-2*x), x), x) - asinh(1-2*x)",
    );
    assert_eq!(nested_residual, "0");
    assert!(
        nested_required.is_empty(),
        "nested negative-slope asinh verification should not add required conditions: {nested_required:?}"
    );
}

#[test]
fn integrate_contract_atanh_affine_by_parts_preserves_open_interval_domain() {
    let (result, required) = evaluated_integral_with_required_conditions("integrate(atanh(x), x)");
    assert_eq!(result, "1/2 * ln(1 - x^2) + x * atanh(x)");
    assert_eq!(
        required,
        vec!["-1 < x < 1".to_string()],
        "atanh integration should publish its open-interval condition"
    );
    assert_antiderivative_verifies("integrate(atanh(x), x)");
    assert_rendered_antiderivative_verifies("integrate(atanh(x), x)", &result);
    let (nested_residual, nested_required) =
        evaluated_expr_with_required_conditions("diff(integrate(atanh(x), x), x) - atanh(x)");
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["-1 < x < 1".to_string()],
        "nested atanh verification should preserve the open-interval condition"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(atanh(2*x), x)");
    assert_eq!(result, "1/4 * ln(1 - (2 * x)^2) + x * atanh(2 * x)");
    assert_eq!(
        required,
        vec!["-1/2 < x < 1/2".to_string()],
        "scaled atanh integration should publish its normalized open-interval condition"
    );
    assert_antiderivative_verifies("integrate(atanh(2*x), x)");
    assert_rendered_antiderivative_verifies("integrate(atanh(2*x), x)", &result);

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(atanh(2*x+1), x)");
    assert_eq!(
        result,
        "1/4 * ln(1 - (2 * x + 1)^2) + 1/2 * (2 * x + 1) * atanh(2 * x + 1)"
    );
    assert_eq!(
        required,
        vec!["-1 < x < 0".to_string()],
        "shifted atanh integration should publish its normalized open-interval condition"
    );
    assert_antiderivative_verifies("integrate(atanh(2*x+1), x)");
    assert_rendered_antiderivative_verifies("integrate(atanh(2*x+1), x)", &result);
    let (nested_residual, nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate(atanh(2*x+1), x), x) - atanh(2*x+1)",
    );
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["-1 < x < 0".to_string()],
        "nested shifted atanh verification should preserve the normalized condition"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(atanh(1-2*x), x)");
    assert_eq!(
        result,
        "-1/2 * (1 - 2 * x) * atanh(1 - 2 * x) - 1/4 * ln(1 - (1 - 2 * x)^2)"
    );
    assert_eq!(
        required,
        vec!["0 < x < 1".to_string()],
        "negative-slope atanh integration should publish its normalized open-interval condition"
    );
    let (wire, _stderr) = cli_eval_json_with_stderr("integrate(atanh(1-2*x), x)");
    assert_eq!(
        wire["result"],
        "-1/2·(1 - 2·x)·atanh(1 - 2·x) - 1/4·ln(1 - (1 - 2·x)^2)"
    );
    assert!(
        !wire["result"]
            .as_str()
            .unwrap_or_default()
            .contains("ln(1 - 1 +"),
        "public negative-slope atanh presentation must not rewrite inside ln: {}",
        wire["result"]
    );
    assert_antiderivative_verifies("integrate(atanh(1-2*x), x)");
    assert_rendered_antiderivative_verifies("integrate(atanh(1-2*x), x)", &result);
    let (nested_residual, nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate(atanh(1-2*x), x), x) - atanh(1-2*x)",
    );
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["0 < x < 1".to_string()],
        "nested negative-slope atanh verification should preserve the normalized condition"
    );
}

#[test]
fn integrate_contract_acosh_affine_by_parts_preserves_real_radical_domain() {
    let (result, required) = evaluated_integral_with_required_conditions("integrate(acosh(x), x)");
    assert_eq!(result, "x * acosh(x) - sqrt(x - 1) * sqrt(x + 1)");
    assert_eq!(
        required,
        vec!["x > 1".to_string()],
        "acosh integration should publish the real radical conditions"
    );
    assert_rendered_antiderivative_verifies("integrate(acosh(x), x)", &result);
    let (nested_residual, nested_required) =
        evaluated_expr_with_required_conditions("diff(integrate(acosh(x), x), x) - acosh(x)");
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["x > 1".to_string()],
        "nested acosh verification should preserve the real radical conditions"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(acosh(2*x), x)");
    assert_eq!(
        result,
        "x * acosh(2 * x) - 1/2 * sqrt(2 * x - 1) * sqrt(2 * x + 1)"
    );
    assert_eq!(
        required,
        vec!["x > 1/2".to_string()],
        "scaled acosh integration should publish the real radical conditions"
    );
    assert_rendered_antiderivative_verifies("integrate(acosh(2*x), x)", &result);

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(acosh(2*x+1), x)");
    assert_eq!(
        result,
        "1/2 * (2 * x + 1) * acosh(2 * x + 1) - 1/2 * sqrt(2 * x) * sqrt(2 * x + 2)"
    );
    assert_eq!(
        required,
        vec!["x > 0".to_string()],
        "shifted acosh integration should publish its normalized real-domain condition"
    );
    assert_rendered_antiderivative_verifies("integrate(acosh(2*x+1), x)", &result);
    let (nested_residual, nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate(acosh(2*x+1), x), x) - acosh(2*x+1)",
    );
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["x > 0".to_string()],
        "nested shifted acosh verification should preserve the normalized condition"
    );
}

#[test]
fn integrate_contract_acosh_negative_slope_preserves_compact_by_parts_form() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(acosh(1-2*x), x)");
    assert_eq!(
        result,
        "1/2 * sqrt(-2 * x) * sqrt(2 - 2 * x) - 1/2 * (1 - 2 * x) * acosh(1 - 2 * x)"
    );
    assert_eq!(
        required,
        vec!["x < 0".to_string()],
        "negative-slope acosh integration should publish its normalized real-domain condition"
    );
    assert_rendered_antiderivative_verifies("integrate(acosh(1-2*x), x)", &result);
    let (nested_residual, nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate(acosh(1-2*x), x), x) - acosh(1-2*x)",
    );
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["x < 0".to_string()],
        "nested negative-slope acosh verification should preserve the normalized condition"
    );
}

#[test]
fn integrate_contract_arctan_positive_quadratic_with_surd_width() {
    let input = "integrate(1/(2*x^2+4*x+5), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "arctan((2 * x + 2) / sqrt(6)) / sqrt(6)");
    assert!(
        required.is_empty(),
        "positive quadratic arctan kernel should not add required conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
}

#[test]
fn integrate_contract_arctan_symbolic_scaled_positive_quadratic() {
    let cases = [
        ("integrate(1/(a^2*x^2+1), x)", "arctan(a * x) / a"),
        (
            "integrate(1/(a^2*(x+b)^2+1), x)",
            "arctan(a * b + a * x) / a",
        ),
        ("integrate(1/((a*x+b)^2+1), x)", "arctan(a * x + b) / a"),
        (
            "integrate(1/((a*x+b)^2+4), x)",
            "arctan((a * x + b) / 2) / (2 * a)",
        ),
        (
            "integrate(1/((a*x+b)^2+2), x)",
            "arctan(sqrt(2) * (a * x + b) / 2) / (sqrt(2) * a)",
        ),
        (
            "integrate(1/(a^2*(2*x+1)^2+1), x)",
            "arctan(2 * a * x + a) / (2 * a)",
        ),
        ("integrate(1/(a^2*(1-x)^2+1), x)", "-arctan(a - a * x) / a"),
    ];

    for (input, expected) in cases {
        let (result, required) = evaluated_integral_with_required_conditions(input);

        assert_eq!(
            result, expected,
            "unexpected symbolic arctan integral for {input}"
        );
        assert_eq!(
            required,
            vec!["a ≠ 0".to_string()],
            "symbolic arctan scale should expose the nonzero parameter condition for {input}"
        );
        assert_antiderivative_verifies(input);
    }

    let (nested_residual, nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate(1/((a*x+b)^2+4), x), x) - 1/((a*x+b)^2+4)",
    );
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["a ≠ 0".to_string()],
        "nested verification should preserve the nonzero parameter condition"
    );

    let (irrational_radius_residual, irrational_radius_required) =
        evaluated_expr_with_required_conditions(
            "diff(integrate(1/((a*x+b)^2+2), x), x) - 1/((a*x+b)^2+2)",
        );
    assert_eq!(irrational_radius_residual, "0");
    assert_eq!(
        irrational_radius_required,
        vec!["a ≠ 0".to_string()],
        "nested irrational-radius verification should preserve the nonzero parameter condition"
    );
}

#[test]
fn integrate_contract_scaled_polynomial_derivative_atanh_substitution() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(2*x/(4-x^4), x)");

    assert_eq!(result, "1/2 * atanh(x^2 / 2)");
    assert_eq!(
        required,
        vec!["4 - x^4 > 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_antiderivative_verifies("integrate(2*x/(4-x^4), x)");
    let (nested_residual, nested_required) =
        evaluated_expr_with_required_conditions("diff(integrate(2*x/(4-x^4), x), x) - 2*x/(4-x^4)");
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["4 - x^4 > 0".to_string()],
        "atanh polynomial residual verification should preserve the open-interval condition"
    );

    let input = "integrate(-2*x/(1-x^4), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "-atanh(x^2)");
    assert_eq!(
        required,
        vec!["1 - x^4 > 0".to_string()],
        "negative atanh substitution should preserve its open-interval domain"
    );
    assert_antiderivative_verifies(input);
}

#[test]
fn integrate_contract_atanh_quadratic_kernel_with_surd_width_preserves_positive_domain() {
    let input = "integrate(1/(3-x^2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "atanh(x / sqrt(3)) / sqrt(3)");
    assert_eq!(
        required,
        vec!["-sqrt(3) < x < sqrt(3)".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);

    let input = "integrate(1/(3/4-(x+1/2)^2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "2 * atanh((2 * x + 1) / sqrt(3)) / sqrt(3)");
    assert!(
        !result.contains("sqrt(3/4)"),
        "atanh quadratic presentation should reduce rational surd width: {result}"
    );
    assert_eq!(
        required,
        vec!["-1/2 - sqrt(3)/2 < x < -1/2 + sqrt(3)/2".to_string()],
        "shifted atanh quadratic should preserve its positive-domain condition"
    );

    let residual = "diff(integrate(1/(3/4-(x+1/2)^2), x), x) - 1/(3/4-(x+1/2)^2)";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert_eq!(
        residual_required,
        vec!["-1/2 - sqrt(3)/2 < x < -1/2 + sqrt(3)/2".to_string()],
        "nested shifted atanh verification should preserve the positive-domain condition"
    );
}

#[test]
fn integrate_contract_scaled_atanh_quadratic_kernel_reduces_surd_width() {
    let input = "integrate(1/(12-4*x^2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "1/4 * atanh(x / sqrt(3)) / sqrt(3)");
    assert_eq!(
        required,
        vec!["-sqrt(3) < x < sqrt(3)".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
}

#[test]
fn integrate_contract_polynomial_atanh_surd_width_uses_compact_positive_domain() {
    let input = "integrate(2*x/(3-x^4), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "atanh(x^2 / sqrt(3)) / sqrt(3)");
    assert_eq!(
        required,
        vec!["3 - x^4 > 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
    let (nested_residual, nested_required) =
        evaluated_expr_with_required_conditions("diff(integrate(2*x/(3-x^4), x), x) - 2*x/(3-x^4)");
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["3 - x^4 > 0".to_string()],
        "nested atanh substitution should preserve its positive open-interval condition"
    );
}

#[test]
fn integrate_contract_shifted_polynomial_atanh_surd_width_uses_compact_positive_domain() {
    let input = "integrate((2*x+2)/(3-(x+1)^4), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "atanh((x^2 + 2 * x + 1) / sqrt(3)) / sqrt(3)");
    assert_eq!(
        required,
        vec!["3 - (x + 1)^4 > 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
}

#[test]
fn integrate_contract_expanded_square_atanh_surd_width_uses_compact_positive_domain() {
    let input = "integrate((2*x+2)/(5-(x^2+2*x+1)^2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "atanh((x^2 + 2 * x + 1) / sqrt(5)) / sqrt(5)");
    assert_eq!(
        required,
        vec!["5 - (x + 1)^4 > 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
}

#[test]
fn integrate_contract_polynomial_derivative_acosh_substitution_preserves_real_domain() {
    let input = "integrate(2*x/sqrt(x^4-4), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "acosh(x^2 / 2)");
    assert_eq!(
        required,
        vec!["x < -sqrt(2) or x > sqrt(2)".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);

    let input = "integrate((2*x+1)/sqrt((x^2+x)^2-4), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "acosh((x^2 + x) / 2)");
    assert_eq!(
        required,
        vec!["x < -2 or x > 1".to_string()],
        "unexpected shifted acosh required_conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);

    let (nested_residual, nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate((2*x+1)/sqrt((x^2+x)^2-4), x), x) - (2*x+1)/sqrt((x^2+x)^2-4)",
    );
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["x < -2 or x > 1".to_string()],
        "nested acosh verification should preserve the real-domain conditions"
    );

    let input = "integrate((2*x+1)/sqrt((x^2+x)^2-5), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "acosh((x^2 + x) / sqrt(5))");
    assert_eq!(
        required,
        vec!["x^2 + x - sqrt(5) > 0".to_string()],
        "unexpected shifted surd-width acosh required_conditions: {required:?}"
    );
    assert_antiderivative_equiv_verifies(input);

    let (nested_equiv, nested_required) = evaluated_equiv_with_required_conditions(
        "diff(integrate((2*x+1)/sqrt((x^2+x)^2-5), x), x)",
        "(2*x+1)/sqrt((x^2+x)^2-5)",
    );
    assert!(nested_equiv);
    assert_eq!(
        nested_required,
        vec!["x^4 + 2 * x^3 + x^2 - 5 > 0".to_string()],
        "surd-width acosh equivalence verification should retain the direct radicand domain"
    );

    let (nested_residual, nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate((2*x+1)/sqrt((x^2+x)^2-5), x), x) - (2*x+1)/sqrt((x^2+x)^2-5)",
    );
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["x^2 + x - sqrt(5) > 0".to_string()],
        "nested surd-width acosh residual should keep the compact real-domain conditions"
    );
}

#[test]
fn integrate_contract_square_minus_constant_uses_abs_log_ratio_and_nonzero_domain() {
    let (result, required) = evaluated_integral_with_required_conditions("integrate(1/(x^2-1), x)");

    assert_eq!(result, "1/2 * ln(|(x - 1) / (x + 1)|)");
    assert_eq!(
        required,
        vec!["x ≠ 1".to_string(), "x ≠ -1".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}

#[test]
fn integrate_contract_factorable_quadratic_log_ratio_simplifies_linear_factors() {
    let cases = [
        (
            "integrate(1/(x^2+x), x)",
            "ln(|x / (x + 1)|)",
            vec!["x ≠ 0".to_string(), "x ≠ -1".to_string()],
        ),
        (
            "integrate(1/(x^2-x), x)",
            "ln(|(x - 1) / x|)",
            vec!["x ≠ 1".to_string(), "x ≠ 0".to_string()],
        ),
        (
            "integrate(1/(x^2+3*x+2), x)",
            "ln(|(x + 1) / (x + 2)|)",
            vec!["x ≠ -1".to_string(), "x ≠ -2".to_string()],
        ),
        (
            "integrate(1/(4*x^2+4*x), x)",
            "1/4 * ln(|x / (x + 1)|)",
            vec!["x ≠ 0".to_string(), "x ≠ -1".to_string()],
        ),
        (
            "integrate(1/(4*x^2+12*x+8), x)",
            "1/4 * ln(|(x + 1) / (x + 2)|)",
            vec!["x ≠ -1".to_string(), "x ≠ -2".to_string()],
        ),
        (
            "integrate(1/(4*x^2-4), x)",
            "1/8 * ln(|(x - 1) / (x + 1)|)",
            vec!["x ≠ 1".to_string(), "x ≠ -1".to_string()],
        ),
        (
            "integrate(1/(2*x^2+3*x+1), x)",
            "ln(|(2 * x + 1) / (x + 1)|)",
            vec!["x ≠ -1/2".to_string(), "x ≠ -1".to_string()],
        ),
        (
            "integrate(1/(6*x^2+9*x+3), x)",
            "1/3 * ln(|(2 * x + 1) / (x + 1)|)",
            vec!["x ≠ -1/2".to_string(), "x ≠ -1".to_string()],
        ),
        (
            "integrate(1/(3*x^2+7*x+4), x)",
            "ln(|(x + 1) / (3 * x + 4)|)",
            vec!["x ≠ -1".to_string(), "x ≠ -4/3".to_string()],
        ),
    ];

    for (input, expected_result, expected_required) in cases {
        let (result, required) = evaluated_integral_with_required_conditions(input);
        assert_eq!(result, expected_result, "input: {input}");
        assert_eq!(
            required, expected_required,
            "input: {input}, unexpected required_conditions: {required:?}"
        );
        assert_rendered_antiderivative_verifies(input, &result);
    }
}

#[test]
fn integrate_contract_repeated_linear_partial_fraction_preserves_nonzero_domain() {
    let input = "integrate((3*x+5)/(x^3-x^2-x+1), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "1/2 * ln(|(x + 1) / (x - 1)|) - 4 / (x - 1)");
    assert_eq!(
        required,
        vec!["x ≠ -1".to_string(), "x ≠ 1".to_string()],
        "unexpected repeated-linear partial fraction required_conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);

    let residual = "diff(integrate((3*x+5)/(x^3-x^2-x+1), x), x) - (3*x+5)/(x^3-x^2-x+1)";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert_eq!(
        residual_required,
        vec!["x ≠ -1".to_string(), "x ≠ 1".to_string()],
        "unexpected repeated-linear partial fraction residual conditions: {residual_required:?}"
    );
}

#[test]
fn integrate_contract_polynomial_derivative_over_denominator_power_preserves_nonzero_domain() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(2*x/(x^2-1)^2, x)");

    assert_eq!(result, "-1 / (x^2 - 1)");
    assert_eq!(
        required,
        vec!["x ≠ -1".to_string(), "x ≠ 1".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}

#[test]
fn integrate_contract_shifted_polynomial_derivative_over_expanded_denominator_square_preserves_nonzero_domain(
) {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate((2*x+1)/(x^4+2*x^3-x^2-2*x+1), x)");

    assert_eq!(result, "-1 / (x^2 + x - 1)");
    assert_eq!(
        required,
        vec!["x^2 + x - 1 ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );

    let (nested_residual, nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate((2*x+1)/(x^4+2*x^3-x^2-2*x+1), x), x) - (2*x+1)/(x^4+2*x^3-x^2-2*x+1)",
    );
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["x^2 + x - 1 ≠ 0".to_string()],
        "expanded denominator antiderivative verification should preserve the compact nonzero domain"
    );
}

#[test]
fn integrate_contract_shifted_polynomial_derivative_over_syntactic_denominator_square_preserves_nonzero_domain(
) {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate((2*x+1)/(x^2+x-1)^2, x)");

    assert_eq!(result, "-1 / (x^2 + x - 1)");
    assert_eq!(
        required,
        vec!["x^2 + x - 1 ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}

#[test]
fn integrate_contract_scaled_syntactic_denominator_square_preserves_nonzero_domain() {
    let input = "integrate((2*x+1)/(3*(x^2+x-1)^2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);
    let step_rules = evaluated_integral_step_rules(input);

    assert_eq!(result, "-1 / (3 * (x^2 + x - 1))");
    assert_eq!(
        required,
        vec!["x^2 + x - 1 ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_eq!(
        step_rules,
        vec!["Symbolic Integration".to_string()],
        "scaled denominator power integration should render as a direct compact reciprocal: {step_rules:?}"
    );
    assert_antiderivative_verifies(input);
}

#[test]
fn integrate_contract_negative_power_denominator_displays_base_nonzero_domain() {
    let input = "integrate((2*x+1)/(3*(x^2+x-1)^(-2)), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);
    let step_rules = evaluated_integral_step_rules(input);

    assert_eq!(result, "1/9 * (x^2 + x - 1)^3");
    assert_eq!(
        required,
        vec!["x^2 + x - 1 ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert!(
        step_rules
            .iter()
            .any(|rule_name| rule_name == "Symbolic Integration"),
        "expected symbolic integration step, got {step_rules:?}"
    );
    assert!(
        !step_rules
            .iter()
            .any(|rule_name| rule_name == "Simplify Complex Fraction"
                || rule_name == "Sacar factor común"),
        "negative-power denominator integration should not expand then refactor before integrating: {step_rules:?}"
    );
    assert_rendered_antiderivative_verifies(input, &result);
}

#[test]
fn integrate_contract_reciprocal_power_denominator_quotient_integrates_directly_with_domain() {
    let input = "integrate((2*x+1)/(3/(x^2+x-1)^2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);
    let step_rules = evaluated_integral_step_rules(input);

    assert_eq!(result, "1/9 * (x^2 + x - 1)^3");
    assert_eq!(
        required,
        vec!["x^2 + x - 1 ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert!(
        step_rules
            .iter()
            .any(|rule_name| rule_name == "Symbolic Integration"),
        "expected symbolic integration step, got {step_rules:?}"
    );
    assert!(
        !step_rules
            .iter()
            .any(|rule_name| rule_name == "Simplify Complex Fraction"
                || rule_name == "Sacar factor común"),
        "reciprocal quotient denominator integration should not expand then refactor before integrating: {step_rules:?}"
    );
    assert_rendered_antiderivative_verifies(input, &result);
}

#[test]
fn integrate_contract_reciprocal_negative_power_denominator_quotient_integrates_directly_with_domain(
) {
    let input = "integrate((2*x+1)/(3/((x^2+x-1)^(-2))), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);
    let step_rules = evaluated_integral_step_rules(input);

    assert_eq!(result, "-1 / (3 * (x^2 + x - 1))");
    assert_eq!(
        required,
        vec!["x^2 + x - 1 ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_eq!(
        step_rules,
        vec!["Symbolic Integration".to_string()],
        "reciprocal negative-power denominator integration should not need a nested-fraction pre-step: {step_rules:?}"
    );
    assert_rendered_antiderivative_verifies(input, &result);
}

#[test]
fn integrate_contract_shifted_polynomial_derivative_over_syntactic_denominator_cube_preserves_nonzero_domain(
) {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate((2*x+1)/(x^2+x-1)^3, x)");

    assert_eq!(result, "-1 / (2 * (x^2 + x - 1)^2)");
    assert_eq!(
        required,
        vec!["x^2 + x - 1 ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}

#[test]
fn integrate_contract_positive_quadratic_denominator_cube_preserves_compact_antiderivative() {
    let input = "integrate(2*x/(x^2+1)^3, x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);
    let step_rules = evaluated_integral_step_rules(input);

    assert_eq!(result, "-1 / (2 * (x^2 + 1)^2)");
    assert!(
        required.is_empty(),
        "positive quadratic denominator should not add required conditions: {required:?}"
    );
    assert!(
        !result.contains("x^4"),
        "post-calculus presentation should keep the denominator factored: {result}"
    );
    assert!(
        !step_rules
            .iter()
            .any(|rule_name| rule_name == "Cancelar factores en una fracción"),
        "scaled denominator-power integration should not expand through cancellation: {step_rules:?}"
    );
    assert_rendered_antiderivative_verifies(input, &result);
}

#[test]
fn integrate_contract_negative_derivative_over_denominator_cube_keeps_compact_domain_signal() {
    let cases = [
        (
            "integrate(-2*x/(x^2+1)^3, x)",
            "1 / (2·(x^2 + 1)^2)",
            "\\frac{1}{2\\cdot {({x}^{2} + 1)}^{2}}",
            serde_json::json!([]),
            "diff(integrate(-2*x/(x^2+1)^3, x), x) + 2*x/(x^2+1)^3",
        ),
        (
            "integrate(-(2*x+1)/(x^2+x-1)^3, x)",
            "1 / (2·(x^2 + x - 1)^2)",
            "\\frac{1}{2\\cdot {({x}^{2} + x - 1)}^{2}}",
            serde_json::json!(["x^2 + x - 1 ≠ 0"]),
            "diff(integrate(-(2*x+1)/(x^2+x-1)^3, x), x) + (2*x+1)/(x^2+x-1)^3",
        ),
    ];

    for (input, expected_result, expected_latex, expected_required, residual) in cases {
        let (wire, stderr) = cli_eval_json_with_stderr(input);
        assert!(
            stderr.is_empty(),
            "unexpected stderr for negative rational denominator-power primitive: {stderr}"
        );
        assert_eq!(wire["result"], expected_result);
        assert_eq!(wire["result_latex"], expected_latex);
        assert_eq!(wire["required_display"], expected_required);

        let (residual_wire, residual_stderr) = cli_eval_json_with_stderr(residual);
        assert!(
            residual_stderr.is_empty(),
            "unexpected stderr for negative rational denominator-power residual: {residual_stderr}"
        );
        assert_eq!(residual_wire["result"], "0");
        assert_eq!(residual_wire["required_display"], expected_required);
    }
}

#[test]
fn integrate_contract_scaled_polynomial_derivative_over_higher_denominator_power_preserves_compact_form_and_nonzero_domain(
) {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate((3*x+3/2)/(x^2+x-1)^4, x)");

    assert_eq!(result, "-1 / (2 * (x^2 + x - 1)^3)");
    assert_eq!(
        required,
        vec!["x^2 + x - 1 ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}

#[test]
fn integrate_contract_scaled_syntactic_denominator_power_preserves_full_nonzero_domain() {
    let (result, mut required) =
        evaluated_integral_with_required_conditions("integrate((8*x+2)/(3*(2*x^2+x-1)^3), x)");

    assert_eq!(result, "-1 / (3 * (2 * x^2 + x - 1)^2)");
    required.sort();
    assert_eq!(
        required,
        vec!["x ≠ -1".to_string(), "x ≠ 1/2".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}

#[test]
fn integrate_contract_shifted_polynomial_derivative_over_expanded_denominator_cube_recovers_compact_base_and_nonzero_domain(
) {
    let (result, required) = evaluated_integral_with_required_conditions(
        "integrate((2*x+1)/(x^6+3*x^5-5*x^3+3*x-1), x)",
    );

    assert_eq!(result, "-1 / (2 * (x^2 + x - 1)^2)");
    assert_eq!(
        required,
        vec!["x^2 + x - 1 ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}

#[test]
fn integrate_contract_shifted_polynomial_derivative_over_scaled_expanded_denominator_cube_recovers_compact_base_and_nonzero_domain(
) {
    let (result, required) = evaluated_integral_with_required_conditions(
        "integrate((2*x+1)/(4*x^6+12*x^5-20*x^3+12*x-4), x)",
    );

    assert_eq!(result, "-1 / (8 * (x^2 + x - 1)^2)");
    assert_eq!(
        required,
        vec!["x^2 + x - 1 ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}

#[test]
fn integrate_contract_shifted_polynomial_derivative_over_negatively_scaled_expanded_denominator_cube_preserves_sign_and_domain(
) {
    let (result, required) = evaluated_integral_with_required_conditions(
        "integrate((2*x+1)/(-4*x^6-12*x^5+20*x^3-12*x+4), x)",
    );

    assert_eq!(result, "1 / (8 * (x^2 + x - 1)^2)");
    assert_eq!(
        required,
        vec!["x^2 + x - 1 ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );

    let (nested_residual, nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate((2*x+1)/(-4*x^6-12*x^5+20*x^3-12*x+4), x), x) - (2*x+1)/(-4*x^6-12*x^5+20*x^3-12*x+4)",
    );
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["x^2 + x - 1 ≠ 0".to_string()],
        "negative expanded denominator antiderivative verification should preserve the compact nonzero domain"
    );
}

#[test]
fn integrate_contract_scaled_polynomial_derivative_over_expanded_denominator_fourth_power_recovers_compact_base_and_nonzero_domain(
) {
    let (result, required) = evaluated_integral_with_required_conditions(
        "integrate((3*x+3/2)/(x^8+4*x^7+2*x^6-8*x^5-5*x^4+8*x^3+2*x^2-4*x+1), x)",
    );

    assert_eq!(result, "-1 / (2 * (x^2 + x - 1)^3)");
    assert_eq!(
        required,
        vec!["x^2 + x - 1 ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}

#[test]
fn integrate_contract_expanded_linear_denominator_fifth_power_recovers_compact_base_and_nonzero_domain(
) {
    let (result, required) = evaluated_integral_with_required_conditions(
        "integrate(1/(x^5+5*x^4+10*x^3+10*x^2+5*x+1), x)",
    );

    assert_eq!(result, "-1 / (4 * (x + 1)^4)");
    assert_eq!(
        required,
        vec!["x ≠ -1".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}

#[test]
fn integrate_contract_near_expanded_denominator_power_remains_residual_without_compact_base_domain()
{
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate((2*x+1)/(x^6+3*x^5-5*x^3+3*x), x)");

    assert!(
        result.starts_with("integrate("),
        "near denominator power should remain residual, got {result}"
    );
    assert!(
        !result.contains("-1 / (2 * (x^2 + x - 1)^2)"),
        "near denominator power must not reuse the exact-power antiderivative"
    );
    assert!(
        !required.contains(&"x^2 + x - 1 ≠ 0".to_string()),
        "near denominator power must not collapse domain to compact base: {required:?}"
    );
}

#[test]
fn integrate_contract_polynomial_derivative_square_minus_constant_log_substitution() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(2*x/(x^4-4), x)");

    assert_eq!(result, "1/4 * ln(|(x^2 - 2) / (x^2 + 2)|)");
    assert_eq!(
        required,
        vec!["x^2 - 2 ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}

#[test]
fn integrate_contract_polynomial_log_derivative_uses_abs_log_and_nonzero_domain() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate((2*x + 1)/(x^2 + x - 1), x)");

    assert_eq!(result, "ln(|x^2 + x - 1|)");
    assert_eq!(
        required,
        vec!["x^2 + x - 1 ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}

#[test]
fn integrate_contract_abs_log_preserves_compact_positive_leading_polynomial_base() {
    let input = "integrate((2*x-1)/(x^2-x-1), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "ln(|x^2 - x - 1|)");
    assert_eq!(
        required,
        vec!["x^2 - x - 1 ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
    assert_rendered_antiderivative_verifies(input, &result);
}

#[test]
fn integrate_contract_scaled_polynomial_log_derivative() {
    assert_eq!(
        simplified_integral("integrate((4*x + 2)/(x^2 + x + 1), x)"),
        "2 * ln(x^2 + x + 1)"
    );
}

#[test]
fn integrate_contract_positive_log_derivative_power_substitution() {
    let input = "integrate((2*x)/(x^2+1)*ln(x^2+1)^2, x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "1/3 * ln(x^2 + 1)^3");
    assert!(
        required.is_empty(),
        "positive log-power substitution should not add required conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
}

#[test]
fn integrate_contract_abs_log_derivative_power_substitution_preserves_nonzero_domain() {
    let input = "integrate((2*x+1)/(x^2+x-1)*ln(abs(x^2+x-1))^2, x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "1/3 * ln(|x^2 + x - 1|)^3");
    assert_eq!(
        required,
        vec!["x^2 + x - 1 ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
}

#[test]
fn integrate_contract_negative_scaled_abs_log_derivative_power_preserves_sign_and_domain() {
    let input = "integrate(-2*((2*x+1)/(x^2+x-1)*ln(abs(x^2+x-1))^2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "-2/3 * ln(|x^2 + x - 1|)^3");
    assert_eq!(
        required,
        vec!["x^2 + x - 1 ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
    assert_rendered_antiderivative_verifies(input, &result);
}

#[test]
fn integrate_contract_positive_log_square_product_by_parts() {
    let input = "integrate(2*x*ln(x^2+1)^2, x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "(x^2 + 1) * (ln(x^2 + 1)^2 - 2 * ln(x^2 + 1) + 2)");
    assert!(
        required.is_empty(),
        "positive log-square product integration should not add required conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
}

#[test]
fn integrate_contract_positive_log_cube_product_by_parts_verifies() {
    let input = "integrate(2*x*ln(x^2+1)^3, x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(
        result,
        "(ln(x^2 + 1)^3 - 3 * ln(x^2 + 1)^2 + 6 * ln(x^2 + 1) - 6) * (x^2 + 1)"
    );
    assert!(
        required.is_empty(),
        "positive log-cube product integration should not add required conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);

    let (nested_residual, nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate(2*x*ln(x^2+1)^3, x), x) - 2*x*ln(x^2+1)^3",
    );
    assert_eq!(nested_residual, "0");
    assert!(
        nested_required.is_empty(),
        "nested log-cube verification should not add required conditions: {nested_required:?}"
    );
}

#[test]
fn integrate_contract_shifted_quadratic_log_cube_product_by_parts_verifies() {
    let input = "integrate((2*x+1)*ln(x^2+x+1)^3, x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(
        result,
        "(ln(x^2 + x + 1)^3 - 3 * ln(x^2 + x + 1)^2 + 6 * ln(x^2 + x + 1) - 6) * (x^2 + x + 1)"
    );
    assert!(
        required.is_empty(),
        "shifted quadratic log-cube product integration should not add required conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);

    let (nested_residual, nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate((2*x+1)*ln(x^2+x+1)^3, x), x) - (2*x+1)*ln(x^2+x+1)^3",
    );
    assert_eq!(nested_residual, "0");
    assert!(
        nested_required.is_empty(),
        "nested shifted quadratic log-cube verification should not add required conditions: {nested_required:?}"
    );
}

#[test]
fn integrate_contract_conditional_log_cube_product_by_parts_verifies() {
    let input = "integrate(2*x*ln(x^2-1)^3, x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(
        result,
        "(ln(x^2 - 1)^3 - 3 * ln(x^2 - 1)^2 + 6 * ln(x^2 - 1) - 6) * (x^2 - 1)"
    );
    assert_eq!(
        required,
        vec!["x < -1 or x > 1".to_string()],
        "conditional log-cube product integration should publish its positive-domain condition"
    );
    assert_antiderivative_verifies(input);

    let (nested_residual, nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate(2*x*ln(x^2-1)^3, x), x) - 2*x*ln(x^2-1)^3",
    );
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["x < -1 or x > 1".to_string()],
        "nested conditional log-cube verification should preserve the positive-domain condition"
    );
}

#[test]
fn integrate_contract_log_high_power_product_by_parts_verifies() {
    let cases: [(&str, &str, &[&str], &str); 4] = [
        (
            "integrate(2*x*ln(x^2+1)^4, x)",
            "(x^2 + 1) * (ln(x^2 + 1)^4 - 4 * ln(x^2 + 1)^3 + 12 * ln(x^2 + 1)^2 - 24 * ln(x^2 + 1) + 24)",
            &[],
            "diff(integrate(2*x*ln(x^2+1)^4, x), x) - 2*x*ln(x^2+1)^4",
        ),
        (
            "integrate(2*x*ln(x^2+1)^5, x)",
            "(x^2 + 1) * (ln(x^2 + 1)^5 - 5 * ln(x^2 + 1)^4 + 20 * ln(x^2 + 1)^3 - 60 * ln(x^2 + 1)^2 + 120 * ln(x^2 + 1) - 120)",
            &[],
            "diff(integrate(2*x*ln(x^2+1)^5, x), x) - 2*x*ln(x^2+1)^5",
        ),
        (
            "integrate((2*x+1)*ln(x^2+x+1)^4, x)",
            "(x^2 + x + 1) * (ln(x^2 + x + 1)^4 - 4 * ln(x^2 + x + 1)^3 + 12 * ln(x^2 + x + 1)^2 - 24 * ln(x^2 + x + 1) + 24)",
            &[],
            "diff(integrate((2*x+1)*ln(x^2+x+1)^4, x), x) - (2*x+1)*ln(x^2+x+1)^4",
        ),
        (
            "integrate((2*x+1)*ln(x^2+x+1)^5, x)",
            "(x^2 + x + 1) * (ln(x^2 + x + 1)^5 - 5 * ln(x^2 + x + 1)^4 + 20 * ln(x^2 + x + 1)^3 - 60 * ln(x^2 + x + 1)^2 + 120 * ln(x^2 + x + 1) - 120)",
            &[],
            "diff(integrate((2*x+1)*ln(x^2+x+1)^5, x), x) - (2*x+1)*ln(x^2+x+1)^5",
        ),
    ];

    for (input, expected, expected_required, residual_input) in cases {
        let (result, required) = evaluated_integral_with_required_conditions(input);
        assert_eq!(result, expected);
        let expected_required: Vec<String> = expected_required
            .iter()
            .map(|condition| condition.to_string())
            .collect();
        assert_eq!(
            required, expected_required,
            "unexpected required_conditions for {input}: {required:?}"
        );
        assert_antiderivative_verifies(input);

        let (residual, residual_required) = evaluated_expr_with_required_conditions(residual_input);
        assert_eq!(residual, "0");
        assert_eq!(
            residual_required, expected_required,
            "unexpected nested required_conditions for {residual_input}: {residual_required:?}"
        );
    }
}

#[test]
fn integrate_contract_conditional_high_log_power_product_by_parts_verifies() {
    let cases: [(&str, &str, &[&str], &str); 3] = [
        (
            "integrate(2*x*ln(x^2-1)^4, x)",
            "(x^2 - 1) * (ln(x^2 - 1)^4 - 4 * ln(x^2 - 1)^3 + 12 * ln(x^2 - 1)^2 - 24 * ln(x^2 - 1) + 24)",
            &["x < -1 or x > 1"],
            "diff(integrate(2*x*ln(x^2-1)^4, x), x) - 2*x*ln(x^2-1)^4",
        ),
        (
            "integrate(2*x*ln(x^2-1)^5, x)",
            "(x^2 - 1) * (ln(x^2 - 1)^5 - 5 * ln(x^2 - 1)^4 + 20 * ln(x^2 - 1)^3 - 60 * ln(x^2 - 1)^2 + 120 * ln(x^2 - 1) - 120)",
            &["x < -1 or x > 1"],
            "diff(integrate(2*x*ln(x^2-1)^5, x), x) - 2*x*ln(x^2-1)^5",
        ),
        (
            "integrate((2*x+1)*ln(x^2+x-1)^4, x)",
            "(x^2 + x - 1) * (ln(x^2 + x - 1)^4 - 4 * ln(x^2 + x - 1)^3 + 12 * ln(x^2 + x - 1)^2 - 24 * ln(x^2 + x - 1) + 24)",
            &["x < -1/2 - sqrt(5)/2 or x > -1/2 + sqrt(5)/2"],
            "diff(integrate((2*x+1)*ln(x^2+x-1)^4, x), x) - (2*x+1)*ln(x^2+x-1)^4",
        ),
    ];

    for (input, expected, expected_required, residual_input) in cases {
        let (result, required) = evaluated_integral_with_required_conditions(input);
        assert_eq!(result, expected, "input: {input}");
        let expected_required: Vec<String> = expected_required
            .iter()
            .map(|condition| condition.to_string())
            .collect();
        assert_eq!(
            required, expected_required,
            "unexpected required_conditions for {input}: {required:?}"
        );
        assert_antiderivative_verifies(input);

        let (residual_wire, residual_stderr) = cli_eval_json_with_stderr(residual_input);
        assert!(
            residual_stderr.is_empty(),
            "conditional high-log-power residual should stay quiet for {input}\nstderr:\n{residual_stderr}"
        );
        assert_eq!(residual_wire["result"], "0", "input: {input}");
        assert_eq!(
            residual_wire["required_display"],
            serde_json::json!(expected_required),
            "unexpected residual required_display for {input}: {:?}",
            residual_wire["required_display"]
        );
    }
}

#[test]
fn integrate_contract_linear_log_square_product_by_parts_preserves_positive_domain() {
    let input = "integrate(ln(2*x+1)^2, x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(
        result,
        "1/2 * (2 * x + 1) * (ln(2 * x + 1)^2 - 2 * ln(2 * x + 1) + 2)"
    );
    assert_eq!(
        required,
        vec!["x > -1/2".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
}

#[test]
fn integrate_contract_quadratic_log_square_product_by_parts() {
    let input = "integrate((2*x+1)*ln(x^2+x+1)^2, x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(
        result,
        "(x^2 + x + 1) * (ln(x^2 + x + 1)^2 - 2 * ln(x^2 + x + 1) + 2)"
    );
    assert!(
        required.is_empty(),
        "positive quadratic log-square integration should not add required conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
}

#[test]
fn integrate_contract_conditional_monomial_log_square_product_by_parts_verifies() {
    let input = "integrate(2*x*ln(x^2-1)^2, x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "(x^2 - 1) * (ln(x^2 - 1)^2 - 2 * ln(x^2 - 1) + 2)");
    assert_eq!(
        required,
        vec!["x < -1 or x > 1".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
}

#[test]
fn integrate_contract_conditional_quadratic_log_square_product_by_parts_verifies() {
    let input = "integrate((2*x+1)*ln(x^2+x-1)^2, x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(
        result,
        "(ln(x^2 + x - 1)^2 - 2 * ln(x^2 + x - 1) + 2) * (x^2 + x - 1)"
    );
    assert_eq!(
        required,
        vec!["x < -1/2 - sqrt(5)/2 or x > -1/2 + sqrt(5)/2".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
}

#[test]
fn integrate_contract_conditional_cubic_log_square_product_by_parts_verifies() {
    let input = "integrate((3*x^2-1)*ln(x^3-x)^2, x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "(x^3 - x) * (ln(x^3 - x)^2 - 2 * ln(x^3 - x) + 2)");
    assert_eq!(
        required,
        vec!["x^3 - x > 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
}

#[test]
fn integrate_contract_conditional_quartic_log_square_product_by_parts_verifies() {
    let input = "integrate((4*x^3-2*x)*ln(x^4-x^2-1)^2, x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(
        result,
        "(ln(x^4 - x^2 - 1)^2 - 2 * ln(x^4 - x^2 - 1) + 2) * (x^4 - x^2 - 1)"
    );
    assert_eq!(
        required,
        vec!["x^4 - x^2 - 1 > 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
}

#[test]
fn integrate_contract_scaled_polynomial_derivative_arcsin_substitution() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(2*x/sqrt(4-x^4), x)");

    assert_eq!(result, "arcsin(x^2 / 2)");
    assert_eq!(
        required,
        vec!["4 - x^4 > 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}

#[test]
fn integrate_contract_polynomial_derivative_arcsin_surd_width_preserves_positive_domain() {
    let input = "integrate(2*x/sqrt(3-x^4), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "arcsin(x^2 / sqrt(3))");
    assert_eq!(
        required,
        vec!["3 - x^4 > 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
    let (nested_residual, nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate(2*x/sqrt(3-x^4), x), x) - 2*x/sqrt(3-x^4)",
    );
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["3 - x^4 > 0".to_string()],
        "nested surd-width arcsin substitution should preserve its positive radicand condition"
    );
}

#[test]
fn integrate_contract_expanded_shifted_polynomial_arcsin_surd_width_dedupes_positive_domain() {
    let input = "integrate((2*x+2)/sqrt(2 - x^4 - 4*x^3 - 6*x^2 - 4*x), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "arcsin((x + 1)^2 / sqrt(3))");
    assert_eq!(
        required,
        vec!["3 - (x + 1)^4 > 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
}

#[test]
fn integrate_contract_factored_shifted_polynomial_arcsin_surd_width_verifies_positive_domain() {
    let input = "integrate((2*x+2)/sqrt(3-(x^2+2*x+1)^2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "arcsin((x + 1)^2 / sqrt(3))");
    assert_eq!(
        required,
        vec!["3 - (x + 1)^4 > 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
}

#[test]
fn integrate_contract_quadratic_affine_log_by_parts_nested_residual_collapses() {
    let input = "integrate((x^2+x)*ln(x+1), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert!(
        result.contains("ln(x + 1)"),
        "quadratic affine-log primitive should retain the log factor: {result}"
    );
    assert_eq!(
        required,
        vec!["x > -1".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);

    let (nested_residual, nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate((x^2+x)*ln(x+1), x), x) - (x^2+x)*ln(x+1)",
    );
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["x > -1".to_string()],
        "nested residual should preserve the log domain condition"
    );

    let (expanded_nested_residual, expanded_nested_required) =
        evaluated_expr_with_required_conditions(
            "diff(integrate(x^2*ln(x+1)+x*ln(x+1), x), x) - (x^2*ln(x+1)+x*ln(x+1))",
        );
    assert_eq!(expanded_nested_residual, "0");
    assert_eq!(
        expanded_nested_required,
        vec!["x > -1".to_string()],
        "expanded nested residual should preserve the log domain condition"
    );

    let (negative_nested_residual, negative_nested_required) =
        evaluated_expr_with_required_conditions(
            "diff(integrate(x^2*ln(x+1)-x*ln(x+1), x), x) - (x^2*ln(x+1)-x*ln(x+1))",
        );
    assert_eq!(negative_nested_residual, "0");
    assert_eq!(
        negative_nested_required,
        vec!["x > -1".to_string()],
        "negative nested residual should preserve the log domain condition"
    );
}

#[test]
fn integrate_contract_shifted_linear_scaled_arcsin_substitution() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(1/sqrt(4-(x+1)^2), x)");

    assert_eq!(result, "arcsin((x + 1) / 2)");
    assert_eq!(
        required,
        vec!["-3 < x < 1".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_antiderivative_verifies("integrate(1/sqrt(4-(x+1)^2), x)");
    let (nested_residual, nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate(1/sqrt(4-(x+1)^2), x), x) - 1/sqrt(4-(x+1)^2)",
    );
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["-3 < x < 1".to_string()],
        "nested shifted arcsin verification should preserve its positive radicand condition"
    );
}

#[test]
fn integrate_contract_polynomial_inverse_sqrt_public_diff_keeps_sqrt_presentation() {
    for (input, expected_diff, expected_required) in [
        (
            "integrate(1/sqrt(1-x^2), x)",
            "1 / sqrt(1 - x^2)",
            vec!["-1 < x < 1".to_string()],
        ),
        (
            "integrate(1/sqrt(4-(x+1)^2), x)",
            "1 / sqrt(4 - (x + 1)^2)",
            vec!["-3 < x < 1".to_string()],
        ),
        (
            "integrate(2*x/sqrt(1+x^4), x)",
            "2 * x / sqrt(x^4 + 1)",
            Vec::new(),
        ),
    ] {
        let direct_diff = format!("diff({input}, x)");
        let (result, required) = evaluated_expr_with_required_conditions(&direct_diff);

        assert_eq!(result, expected_diff, "input: {direct_diff}");
        assert_eq!(required, expected_required, "input: {direct_diff}");

        let residual = format!("{direct_diff} - {expected_diff}");
        let (residual_result, residual_required) =
            evaluated_expr_with_required_conditions(&residual);
        assert_eq!(residual_result, "0", "input: {residual}");
        assert_eq!(residual_required, expected_required, "input: {residual}");
    }
}

#[test]
fn integrate_contract_beta_sqrt_product_kernel_preserves_open_domain_and_verifies() {
    for (input, expected_result, expected_derivative) in [
        (
            "integrate(1/(sqrt(x)*sqrt(1-x)), x)",
            "arcsin(2 * x - 1)",
            "1 / (sqrt(x) * sqrt(1 - x))",
        ),
        (
            "integrate(1/(2*sqrt(x)*sqrt(1-x)), x)",
            "1/2 * arcsin(2 * x - 1)",
            "1 / (2 * sqrt(x) * sqrt(1 - x))",
        ),
    ] {
        let (result, mut required) = evaluated_integral_with_required_conditions(input);
        let mut expected_required = vec!["x < 1".to_string(), "x > 0".to_string()];
        required.sort();
        expected_required.sort();

        assert_eq!(result, expected_result, "input: {input}");
        assert_eq!(
            required, expected_required,
            "sqrt-product beta kernel should preserve both open denominator conditions"
        );
        assert_antiderivative_verifies(input);

        let rendered_derivative = format!("diff({result}, x)");
        let (derivative_result, mut nested_required) =
            evaluated_expr_with_required_conditions(&rendered_derivative);
        nested_required.sort();
        assert_eq!(derivative_result, expected_derivative, "input: {input}");
        assert_eq!(
            nested_required, expected_required,
            "rendered beta-kernel derivative should preserve both open denominator conditions"
        );

        let direct_diff = format!("diff({input}, x)");
        let (direct_result, mut direct_required) =
            evaluated_expr_with_required_conditions(&direct_diff);
        direct_required.sort();
        assert_eq!(direct_result, expected_derivative, "input: {direct_diff}");
        assert_eq!(
            direct_required, expected_required,
            "direct diff(integrate(...)) beta-kernel presentation should preserve both open denominator conditions"
        );
    }

    let direct_affine = "diff(integrate(1/(sqrt(2*x+1)*sqrt(3-2*x)), x), x)";
    let (direct_affine_result, mut direct_affine_required) =
        evaluated_expr_with_required_conditions(direct_affine);
    let mut expected_affine_required = vec!["x < 3/2".to_string(), "x > -1/2".to_string()];
    direct_affine_required.sort();
    expected_affine_required.sort();
    assert_eq!(
        direct_affine_result,
        "1 / (sqrt(2 * x + 1) * sqrt(3 - 2 * x))"
    );
    assert_eq!(
        direct_affine_required, expected_affine_required,
        "affine beta-kernel presentation should preserve both open denominator conditions"
    );

    let direct_symbolic = "diff(integrate(a/(2*sqrt(x)*sqrt(1-x)), x), x)";
    let (direct_symbolic_result, mut direct_symbolic_required) =
        evaluated_expr_with_required_conditions(direct_symbolic);
    let mut expected_required = vec!["x < 1".to_string(), "x > 0".to_string()];
    direct_symbolic_required.sort();
    expected_required.sort();
    assert_eq!(direct_symbolic_result, "a / (2 * sqrt(x) * sqrt(1 - x))");
    assert_eq!(
        direct_symbolic_required, expected_required,
        "symbolic beta-kernel presentation should preserve both open denominator conditions"
    );

    let direct_symbolic_affine = "diff(integrate(a/(sqrt(2*x+1)*sqrt(3-2*x)), x), x)";
    let (direct_symbolic_affine_result, mut direct_symbolic_affine_required) =
        evaluated_expr_with_required_conditions(direct_symbolic_affine);
    direct_symbolic_affine_required.sort();
    assert_eq!(
        direct_symbolic_affine_result,
        "a / (sqrt(2 * x + 1) * sqrt(3 - 2 * x))"
    );
    assert_eq!(
        direct_symbolic_affine_required, expected_affine_required,
        "symbolic affine beta-kernel presentation should preserve both open denominator conditions"
    );

    let (nested_residual, mut nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate(a/(2*sqrt(x)*sqrt(1-x)), x), x) - a/(2*sqrt(x)*sqrt(1-x))",
    );
    nested_required.sort();
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required, expected_required,
        "nested symbolic beta-kernel verification should preserve both open denominator conditions"
    );
}

#[test]
fn integrate_contract_shifted_sqrt_arcsin_kernel_verifies_public_residual() {
    let input = "integrate(1/(sqrt(x)*sqrt(sqrt(x)-x)), x)";
    let (result, mut required) = evaluated_integral_with_required_conditions(input);
    let mut expected_required = vec!["sqrt(x) - x > 0".to_string(), "x > 0".to_string()];
    required.sort();
    expected_required.sort();

    assert_eq!(result, "2 * arcsin(2 * sqrt(x) - 1)");
    assert_eq!(
        required, expected_required,
        "shifted sqrt arcsin kernel should preserve minimal denominator conditions"
    );
    assert_antiderivative_verifies(input);

    let (nested_residual, mut nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate(1/(sqrt(x)*sqrt(sqrt(x)-x)), x), x) - 1/(sqrt(x)*sqrt(sqrt(x)-x))",
    );
    let mut expected_nested_required = vec!["sqrt(x) - x > 0".to_string(), "x > 0".to_string()];
    nested_required.sort();
    expected_nested_required.sort();
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required, expected_nested_required,
        "nested shifted sqrt arcsin residual should preserve denominator conditions"
    );
}

#[test]
fn integrate_contract_polynomial_derivative_asinh_substitution() {
    assert_eq!(
        simplified_integral("integrate(2*x/sqrt(1+x^4), x)"),
        "asinh(x^2)"
    );

    let input = "integrate(-2*x/sqrt(1+x^4), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "-asinh(x^2)");
    assert!(
        required.is_empty(),
        "negative asinh substitution should remain unconditional: {required:?}"
    );
    assert_antiderivative_verifies(input);
}

#[test]
fn integrate_contract_polynomial_derivative_asinh_residual_omits_cycle_blocked_hint_after_zero() {
    let input = "diff(integrate(2*x/sqrt(1+x^4), x), x) - 2*x/sqrt(1+x^4)";
    let (wire, stderr) = cli_eval_json_with_stderr(input);

    assert_eq!(wire["result"], "0");
    assert!(
        wire.get("blocked_hints").is_none(),
        "successful residual should not surface non-actionable cycle blocked hints: {wire:?}"
    );
    assert!(
        !stderr.contains("cycle detected"),
        "successful residual should not print cycle detected hint\nstderr:\n{stderr}"
    );
}

#[test]
fn integrate_contract_scaled_polynomial_derivative_asinh_substitution() {
    assert_eq!(
        simplified_integral("integrate(2*x/sqrt(4+x^4), x)"),
        "asinh(x^2 / 2)"
    );
}

#[test]
fn integrate_contract_polynomial_derivative_asinh_surd_width_remains_unconditional() {
    let input = "integrate(2*x/sqrt(3+x^4), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "asinh(x^2 / sqrt(3))");
    assert!(
        required.is_empty(),
        "unexpected required_conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
}

#[test]
fn integrate_contract_asinh_rational_surd_width_reduces_inner_offset() {
    let input = "integrate(x/sqrt(x^4+3/4), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "1/2 * asinh(2 * x^2 / sqrt(3))");
    assert!(
        !result.contains("sqrt(3/4)"),
        "asinh substitution presentation should reduce rational surd width: {result}"
    );
    assert!(
        required.is_empty(),
        "asinh positive radicand substitution should remain unconditional: {required:?}"
    );
    assert_antiderivative_equiv_verifies(input);
    assert_rendered_antiderivative_verifies(input, &result);

    let residual = "diff(integrate(x/sqrt(x^4+3/4), x), x) - x/sqrt(x^4+3/4)";
    let (residual_result, residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
    assert!(
        residual_required.is_empty(),
        "nested asinh verification should remain unconditional: {residual_required:?}"
    );

    let explicit_residual = "diff(1/2*asinh(2*x^2/sqrt(3)), x) - x/sqrt(x^4+3/4)";
    let (explicit_residual_result, explicit_residual_required) =
        evaluated_expr_with_required_conditions(explicit_residual);
    assert_eq!(explicit_residual_result, "0");
    assert!(
        explicit_residual_required.is_empty(),
        "explicit asinh verification should remain unconditional: {explicit_residual_required:?}"
    );

    let additive_residual = "diff(-1/2*asinh(2*x^2/sqrt(3)), x) + x/sqrt(x^4+3/4)";
    let (additive_residual_result, additive_residual_required) =
        evaluated_expr_with_required_conditions(additive_residual);
    assert_eq!(additive_residual_result, "0");
    assert!(
        additive_residual_required.is_empty(),
        "negative explicit asinh verification should remain unconditional: {additive_residual_required:?}"
    );
}

#[test]
fn integrate_contract_arcsin_rational_surd_width_reduces_inner_offset() {
    let input = "integrate((2*x+1)/sqrt(3/4-(x^2+x+1)^2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "arcsin((2 * x^2 + 2 * x + 2) / sqrt(3))");
    assert!(
        !result.contains("sqrt(3/4)"),
        "arcsin substitution presentation should reduce rational surd width: {result}"
    );
    assert!(
        required == ["3/4 - (x^2 + x + 1)^2 > 0"],
        "arcsin substitution presentation should preserve the real radicand condition: {required:?}"
    );
    assert_antiderivative_equiv_verifies(input);
    assert_rendered_antiderivative_verifies(input, &result);

    let residual = "diff(arcsin((2*x^2+2*x+2)/sqrt(3)), x) - (2*x+1)/sqrt(3/4-(x^2+x+1)^2)";
    let (residual_result, _residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");
}

#[test]
fn integrate_contract_arcsin_scaled_rational_surd_width_verifies() {
    let input = "integrate((x+1/2)/sqrt(3/4-(x^2+x+1)^2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "1/2 * arcsin((2 * x^2 + 2 * x + 2) / sqrt(3))");
    assert!(
        required == ["3/4 - (x^2 + x + 1)^2 > 0"],
        "scaled arcsin substitution should preserve the real radicand condition: {required:?}"
    );
    assert_antiderivative_equiv_verifies(input);
    assert_rendered_antiderivative_verifies(input, &result);

    let residual = "diff(1/2*arcsin((2*x^2+2*x+2)/sqrt(3)), x) - (x+1/2)/sqrt(3/4-(x^2+x+1)^2)";
    let (residual_result, _residual_required) = evaluated_expr_with_required_conditions(residual);
    assert_eq!(residual_result, "0");

    let negative_input = "integrate(-(x+1/2)/sqrt(3/4-(x^2+x+1)^2), x)";
    let (negative_result, negative_required) =
        evaluated_integral_with_required_conditions(negative_input);
    assert_eq!(
        negative_result,
        "-1/2 * arcsin((2 * x^2 + 2 * x + 2) / sqrt(3))"
    );
    assert!(
        negative_required == ["3/4 - (x^2 + x + 1)^2 > 0"],
        "negative scaled arcsin substitution should preserve the real radicand condition: {negative_required:?}"
    );
    assert_antiderivative_equiv_verifies(negative_input);
    assert_rendered_antiderivative_verifies(negative_input, &negative_result);

    let additive_residual =
        "diff(-1/2*arcsin((2*x^2+2*x+2)/sqrt(3)), x) + (x+1/2)/sqrt(3/4-(x^2+x+1)^2)";
    let (additive_residual_result, _additive_residual_required) =
        evaluated_expr_with_required_conditions(additive_residual);
    assert_eq!(additive_residual_result, "0");
}

#[test]
fn integrate_contract_shifted_linear_scaled_asinh_substitution() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(1/sqrt(4+(x+1)^2), x)");

    assert_eq!(result, "asinh((x + 1) / 2)");
    assert!(
        required.is_empty(),
        "unexpected required_conditions: {required:?}"
    );
    assert_antiderivative_verifies("integrate(1/sqrt(4+(x+1)^2), x)");
    let (nested_residual, nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate(1/sqrt(4+(x+1)^2), x), x) - 1/sqrt(4+(x+1)^2)",
    );
    assert_eq!(nested_residual, "0");
    assert!(
        nested_required.is_empty(),
        "nested shifted asinh verification should remain unconditional: {nested_required:?}"
    );
}

#[test]
fn integrate_contract_polynomial_derivative_over_square_root_substitution() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(x/sqrt(x^2+1), x)");

    assert_eq!(result, "sqrt(x^2 + 1)");
    assert!(
        required.is_empty(),
        "unexpected required_conditions: {required:?}"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(2*x/sqrt(x^2-1), x)");

    assert_eq!(result, "2 * sqrt(x^2 - 1)");
    assert_eq!(
        required,
        vec!["x < -1 or x > 1".to_string()],
        "unexpected required_conditions: {required:?}"
    );

    let input = "integrate((3*x^2+2*x+1)/sqrt(x^3+x^2+x+1), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "2 * sqrt(x^3 + x^2 + x + 1)");
    assert_eq!(
        required,
        vec!["x^3 + x^2 + x + 1 > 0".to_string()],
        "unexpected cubic radicand required_conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);

    let (nested_residual, nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate((3*x^2+2*x+1)/sqrt(x^3+x^2+x+1), x), x) - (3*x^2+2*x+1)/sqrt(x^3+x^2+x+1)",
    );
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["x^3 + x^2 + x + 1 > 0".to_string()],
        "cubic radicand residual verification should preserve positive-domain conditions"
    );

    let direct_diff = "diff(integrate((3*x^2+2*x+1)/sqrt(x^3+x^2+x+1), x), x)";
    let (direct_wire, direct_stderr) = cli_eval_json_with_stderr(direct_diff);
    assert!(
        direct_stderr.is_empty(),
        "direct square-root substitution diff/integrate should stay quiet: {direct_stderr}"
    );
    assert_eq!(
        direct_wire["result"],
        "(3·x^2 + 2·x + 1) / sqrt(x^3 + x^2 + x + 1)"
    );
    assert_eq!(
        direct_wire["required_display"],
        serde_json::json!(["x^3 + x^2 + x + 1 > 0"])
    );
}

#[test]
fn integrate_contract_affine_sqrt_product_derivative_inverse() {
    let input = "integrate((3*x+5)/(2*sqrt(x+2)), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "sqrt(x + 2) * (x + 1)");
    assert_eq!(
        required,
        vec!["x > -2".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
    assert_antiderivative_equiv_verifies(input);

    let input = "integrate((3*x+1)/(2*sqrt(x)), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "sqrt(x) * (x + 1)");
    assert_eq!(
        required,
        vec!["x > 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_antiderivative_equiv_verifies(input);

    let input = "integrate((1/2)*(3*x+1)*x^(-1/2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "sqrt(x) * (x + 1)");
    assert_eq!(
        required,
        vec!["x > 0".to_string()],
        "unexpected product-form required_conditions: {required:?}"
    );
    assert_antiderivative_equiv_verifies(input);

    let input = "integrate((2-3*x)*(3-2*x)^(-1/2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "sqrt(3 - 2 * x) * (x + 1)");
    assert_eq!(
        required,
        vec!["x < 3/2".to_string()],
        "unexpected negative-slope product-form required_conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
}

#[test]
fn integrate_contract_polynomial_derivative_times_square_root_substitution() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(x*sqrt(x^2+1), x)");

    assert_eq!(result, "1/3 * sqrt(x^2 + 1) * (x^2 + 1)");
    assert!(
        required.is_empty(),
        "unexpected required_conditions: {required:?}"
    );
    assert_antiderivative_verifies("integrate(x*sqrt(x^2+1), x)");

    let input = "integrate(-x*sqrt(x^2+1), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "-(1/3 * (x^2 + 1) * sqrt(x^2 + 1))");
    assert!(
        required.is_empty(),
        "unexpected negative required_conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);

    let input = "integrate((2*x+1)*sqrt(x^2+x+1), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "2/3 * sqrt(x^2 + x + 1) * (x^2 + x + 1)");
    assert!(
        required.is_empty(),
        "unexpected affine required_conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);

    let input = "integrate(2*x*sqrt(x^2-1), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "2/3 * sqrt(x^2 - 1) * (x^2 - 1)");
    assert_eq!(
        required,
        vec!["x ≤ -1 or x ≥ 1".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);

    let input = "integrate(-2*x*sqrt(x^2-1), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "-(2/3 * sqrt(x^2 - 1) * (x^2 - 1))");
    assert_eq!(
        required,
        vec!["x ≤ -1 or x ≥ 1".to_string()],
        "negated square-root substitution should preserve the same nonnegative-base condition"
    );
    assert_antiderivative_verifies(input);
}

#[test]
fn integrate_contract_polynomial_derivative_times_power_substitution() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(2*x*(x^2+1)^3, x)");

    assert_eq!(result, "1/4 * (x^2 + 1)^4");
    assert!(
        required.is_empty(),
        "unexpected required_conditions: {required:?}"
    );
    assert_antiderivative_verifies("integrate(2*x*(x^2+1)^3, x)");

    let input = "integrate(2*x*(x^2-1)^(3/2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "2/5 * (x^2 - 1)^(5/2)");
    assert_eq!(
        required,
        vec!["x ≤ -1 or x ≥ 1".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);

    let input = "integrate(-2*x*(x^2-1)^(3/2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "-2/5 * (x^2 - 1)^(5/2)");
    assert_eq!(
        required,
        vec!["x ≤ -1 or x ≥ 1".to_string()],
        "negated power substitution should preserve the same nonnegative-base condition"
    );
    assert_antiderivative_verifies(input);
}

#[test]
fn integrate_contract_polynomial_derivative_over_fractional_denominator_power_substitution() {
    let input = "integrate((2*x+1)/(x^2+x+1)^(3/2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "-2 / sqrt(x^2 + x + 1)");
    assert!(
        required.is_empty(),
        "positive quadratic denominator should not emit redundant conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
    let step_summaries = evaluated_expr_step_summaries(input);
    assert_eq!(
        step_summaries
            .iter()
            .filter(|(_, rule_name, _)| rule_name == "Symbolic Integration")
            .count(),
        1,
        "integration should stay as one compact didactic step: {step_summaries:?}"
    );
    assert!(
        !step_summaries.iter().any(|(_, rule_name, _)| {
            rule_name == "Rationalize Product Denominator"
                || rule_name == "Cancel Same Base Powers"
                || rule_name == "Present calculus result in compact form"
        }),
        "compact integration trace should not expose rationalize/cancel/post-presentation roundtrip: {step_summaries:?}"
    );
    assert_eq!(
        rationalize_rewrites_for_simplify(input),
        0,
        "held compact integration result should not take an internal rationalize route"
    );
    let direct_diff = "diff(integrate((2*x+1)/(x^2+x+1)^(3/2), x), x)";
    let (direct_wire, direct_stderr) = cli_eval_json_with_stderr(direct_diff);
    assert!(
        direct_stderr.is_empty(),
        "direct fractional denominator-power diff/integrate should stay quiet: {direct_stderr}"
    );
    assert_eq!(direct_wire["result"], "(2·x + 1) / (x^2 + x + 1)^(3 / 2)");
    assert_eq!(direct_wire["required_display"], serde_json::json!([]));

    let direct_diff = "diff(integrate((2*x+1)/(x^2+x+1)^(5/2), x), x)";
    let (direct_wire, direct_stderr) = cli_eval_json_with_stderr(direct_diff);
    assert!(
        direct_stderr.is_empty(),
        "direct higher fractional denominator-power diff/integrate should stay quiet: {direct_stderr}"
    );
    assert_eq!(direct_wire["result"], "(2·x + 1) / (x^2 + x + 1)^(5 / 2)");
    assert_eq!(direct_wire["required_display"], serde_json::json!([]));
    let input = "integrate((2*x+1)/(x^2+x+1)^(5/2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);
    assert_eq!(result, "-2 / (3 * sqrt(x^2 + x + 1) * (x^2 + x + 1))");
    assert!(
        !result.contains("^(3/2)"),
        "post-integration presentation should prefer a polynomial-sqrt denominator: {result}"
    );
    assert!(
        required.is_empty(),
        "higher positive-quadratic denominator should not emit redundant conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
    let residual = "diff(integrate((2*x+1)/(x^2+x+1)^(5/2), x), x) - (2*x+1)/(x^2+x+1)^(5/2)";
    let (residual_wire, residual_stderr) = cli_eval_json_with_stderr(residual);
    assert!(
        residual_stderr.is_empty(),
        "higher fractional denominator-power residual should stay quiet: {residual_stderr}"
    );
    assert_eq!(residual_wire["result"], "0");
    assert_eq!(residual_wire["required_display"], serde_json::json!([]));

    let input = "integrate((2*x+1)/(x^2+x+1)^(7/2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);
    assert_eq!(result, "-2 / (5 * sqrt(x^2 + x + 1) * (x^2 + x + 1)^2)");
    assert!(
        !result.contains("^(5/2)"),
        "deeper post-integration presentation should prefer a polynomial-sqrt denominator: {result}"
    );
    assert!(
        required.is_empty(),
        "deeper positive-quadratic denominator should not emit redundant conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
    let residual = "diff(integrate((2*x+1)/(x^2+x+1)^(7/2), x), x) - (2*x+1)/(x^2+x+1)^(7/2)";
    let (residual_wire, residual_stderr) = cli_eval_json_with_stderr(residual);
    assert!(
        residual_stderr.is_empty(),
        "deeper fractional denominator-power residual should stay quiet: {residual_stderr}"
    );
    assert_eq!(residual_wire["result"], "0");
    assert_eq!(residual_wire["required_display"], serde_json::json!([]));

    let input = "integrate((2*x+1)/(x^2+x+1)^(9/2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);
    assert_eq!(result, "-2 / (7 * sqrt(x^2 + x + 1) * (x^2 + x + 1)^3)");
    assert!(
        !result.contains("^(7/2)"),
        "deepest post-integration presentation should prefer a polynomial-sqrt denominator: {result}"
    );
    assert!(
        required.is_empty(),
        "deepest positive-quadratic denominator should not emit redundant conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
    let residual = "diff(integrate((2*x+1)/(x^2+x+1)^(9/2), x), x) - (2*x+1)/(x^2+x+1)^(9/2)";
    let (residual_wire, residual_stderr) = cli_eval_json_with_stderr(residual);
    assert!(
        residual_stderr.is_empty(),
        "deepest fractional denominator-power residual should stay quiet: {residual_stderr}"
    );
    assert_eq!(residual_wire["result"], "0");
    assert_eq!(residual_wire["required_display"], serde_json::json!([]));

    let input = "integrate((2*x+1)/(sqrt(x^2+x+1)^3), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "-2 / sqrt(x^2 + x + 1)");
    assert!(
        required.is_empty(),
        "sqrt-denominator spelling should share the same positive-quadratic domain: {required:?}"
    );
    assert_antiderivative_verifies(input);
    assert_eq!(
        rationalize_rewrites_for_simplify(input),
        0,
        "sqrt-denominator spelling should not take an internal rationalize route"
    );

    let input = "integrate(2*x/(x^2-1)^(3/2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "-2 / sqrt(x^2 - 1)");
    assert_eq!(
        required,
        vec!["x < -1 or x > 1".to_string()],
        "fractional denominator power should require the base to be positive"
    );
    assert_antiderivative_verifies(input);
    assert_eq!(
        rationalize_rewrites_for_simplify(input),
        0,
        "conditional fractional denominator power should not take an internal rationalize route"
    );
    let input = "integrate(2*x/(x^2-1)^(5/2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);
    assert_eq!(result, "-2 / (3 * (x^2 - 1) * sqrt(x^2 - 1))");
    assert_eq!(
        required,
        vec!["x < -1 or x > 1".to_string()],
        "conditional higher fractional denominator power should preserve the positive-base condition"
    );
    assert_antiderivative_verifies(input);

    let input = "integrate(2*x/(x^2-1)^(7/2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);
    assert_eq!(result, "-2 / (5 * (x^2 - 1)^2 * sqrt(x^2 - 1))");
    assert_eq!(
        required,
        vec!["x < -1 or x > 1".to_string()],
        "conditional deeper fractional denominator power should preserve the positive-base condition"
    );
    assert_antiderivative_verifies(input);

    let input = "integrate(2*x/(x^2-1)^(11/2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);
    assert_eq!(result, "-2 / (9 * (x^2 - 1)^4 * sqrt(x^2 - 1))");
    assert!(
        !result.contains("^(9/2)"),
        "conditional deepest denominator-power presentation should prefer a polynomial-sqrt denominator: {result}"
    );
    assert_eq!(
        required,
        vec!["x < -1 or x > 1".to_string()],
        "conditional deepest fractional denominator power should preserve the positive-base condition"
    );
    assert_antiderivative_verifies(input);
    let residual = "diff(integrate(2*x/(x^2-1)^(11/2), x), x) - 2*x/(x^2-1)^(11/2)";
    let (residual_wire, residual_stderr) = cli_eval_json_with_stderr(residual);
    assert!(
        residual_stderr.is_empty(),
        "conditional deepest fractional denominator-power residual should stay quiet: {residual_stderr}"
    );
    assert_eq!(residual_wire["result"], "0");
    assert_eq!(
        residual_wire["required_display"],
        serde_json::json!(["x < -1 or x > 1"])
    );

    let input = "integrate(2*x/(x^2-1)^(13/2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);
    assert_eq!(result, "-2 / (11 * (x^2 - 1)^5 * sqrt(x^2 - 1))");
    assert!(
        !result.contains("^(11/2)"),
        "conditional next-depth denominator-power presentation should prefer a polynomial-sqrt denominator: {result}"
    );
    assert_eq!(
        required,
        vec!["x < -1 or x > 1".to_string()],
        "conditional next-depth fractional denominator power should preserve the positive-base condition"
    );
    assert_antiderivative_verifies(input);
    let residual = "diff(integrate(2*x/(x^2-1)^(13/2), x), x) - 2*x/(x^2-1)^(13/2)";
    let (residual_wire, residual_stderr) = cli_eval_json_with_stderr(residual);
    assert!(
        residual_stderr.is_empty(),
        "conditional next-depth fractional denominator-power residual should stay quiet: {residual_stderr}"
    );
    assert_eq!(residual_wire["result"], "0");
    assert_eq!(
        residual_wire["required_display"],
        serde_json::json!(["x < -1 or x > 1"])
    );

    let input = "integrate(2*x/(x^2-1)^(15/2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);
    assert_eq!(result, "-2 / (13 * (x^2 - 1)^6 * sqrt(x^2 - 1))");
    assert!(
        !result.contains("^(13/2)"),
        "conditional odd-half denominator-power presentation should not depend on a manual exponent whitelist: {result}"
    );
    assert_eq!(
        required,
        vec!["x < -1 or x > 1".to_string()],
        "conditional odd-half fractional denominator power should preserve the positive-base condition"
    );
    assert_antiderivative_verifies(input);
    let residual = "diff(integrate(2*x/(x^2-1)^(15/2), x), x) - 2*x/(x^2-1)^(15/2)";
    let (residual_wire, residual_stderr) = cli_eval_json_with_stderr(residual);
    assert!(
        residual_stderr.is_empty(),
        "conditional odd-half fractional denominator-power residual should stay quiet: {residual_stderr}"
    );
    assert_eq!(residual_wire["result"], "0");
    assert_eq!(
        residual_wire["required_display"],
        serde_json::json!(["x < -1 or x > 1"])
    );

    let direct_diff = "diff(integrate(2*x/(x^2-1)^(3/2), x), x)";
    let (direct_wire, direct_stderr) = cli_eval_json_with_stderr(direct_diff);
    assert!(
        direct_stderr.is_empty(),
        "direct conditional fractional denominator-power diff/integrate should stay quiet: {direct_stderr}"
    );
    assert_eq!(direct_wire["result"], "2·x / (x^2 - 1)^(3 / 2)");
    assert_eq!(
        direct_wire["required_display"],
        serde_json::json!(["x < -1 or x > 1"])
    );
    let residual = "diff(integrate(2*x/(x^2-1)^(5/2), x), x) - 2*x/(x^2-1)^(5/2)";
    let (residual_wire, residual_stderr) = cli_eval_json_with_stderr(residual);
    assert!(
        residual_stderr.is_empty(),
        "higher conditional fractional denominator-power residual should stay quiet: {residual_stderr}"
    );
    assert_eq!(residual_wire["result"], "0");
    assert_eq!(
        residual_wire["required_display"],
        serde_json::json!(["x < -1 or x > 1"])
    );

    let input = "integrate((4*x+2)/(2*x^2+2*x-3)^(3/2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "-2 / sqrt(2 * x^2 + 2 * x - 3)");
    assert_eq!(
        required,
        vec!["x < -1/2 - sqrt(7)/2 or x > -1/2 + sqrt(7)/2".to_string()],
        "scaled shifted fractional denominator power should preserve the positive-base condition"
    );
    assert_antiderivative_verifies(input);
    let input = "integrate((4*x+2)/(2*x^2+2*x-3)^(5/2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);
    assert_eq!(
        result,
        "-2 / (3 * sqrt(2 * x^2 + 2 * x - 3) * (2 * x^2 + 2 * x - 3))"
    );
    assert_eq!(
        required,
        vec!["x < -1/2 - sqrt(7)/2 or x > -1/2 + sqrt(7)/2".to_string()],
        "higher scaled shifted fractional denominator power should preserve the positive-base condition"
    );
    assert_antiderivative_verifies(input);
    let input = "integrate((4*x+2)/(2*x^2+2*x-3)^(7/2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);
    assert_eq!(
        result,
        "-2 / (5 * sqrt(2 * x^2 + 2 * x - 3) * (2 * x^2 + 2 * x - 3)^2)"
    );
    assert_eq!(
        required,
        vec!["x < -1/2 - sqrt(7)/2 or x > -1/2 + sqrt(7)/2".to_string()],
        "deeper scaled shifted fractional denominator power should preserve the positive-base condition"
    );
    assert_antiderivative_verifies(input);
    let residual =
        "diff(integrate((4*x+2)/(2*x^2+2*x-3)^(5/2), x), x) - (4*x+2)/(2*x^2+2*x-3)^(5/2)";
    let (residual_wire, residual_stderr) = cli_eval_json_with_stderr(residual);
    assert!(
        residual_stderr.is_empty(),
        "higher scaled shifted fractional denominator-power residual should stay quiet: {residual_stderr}"
    );
    assert_eq!(residual_wire["result"], "0");
    assert_eq!(
        residual_wire["required_display"],
        serde_json::json!(["x < -1/2 - sqrt(7)/2 or x > -1/2 + sqrt(7)/2"])
    );
    let residual =
        "diff(integrate((4*x+2)/(2*x^2+2*x-3)^(7/2), x), x) - (4*x+2)/(2*x^2+2*x-3)^(7/2)";
    let (residual_wire, residual_stderr) = cli_eval_json_with_stderr(residual);
    assert!(
        residual_stderr.is_empty(),
        "deeper scaled shifted fractional denominator-power residual should stay quiet: {residual_stderr}"
    );
    assert_eq!(residual_wire["result"], "0");
    assert_eq!(
        residual_wire["required_display"],
        serde_json::json!(["x < -1/2 - sqrt(7)/2 or x > -1/2 + sqrt(7)/2"])
    );
    assert_eq!(
        rationalize_rewrites_for_simplify(input),
        0,
        "scaled shifted fractional denominator power should not take an internal rationalize route"
    );

    let input = "integrate(-2*x/(x^2-1)^(3/2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "2 / sqrt(x^2 - 1)");
    assert_eq!(
        required,
        vec!["x < -1 or x > 1".to_string()],
        "negated fractional denominator power should preserve the same positive-base condition"
    );
    assert_antiderivative_verifies(input);
    assert_eq!(
        rationalize_rewrites_for_simplify(input),
        0,
        "negated fractional denominator power should not take an internal rationalize route"
    );
}

#[test]
fn integrate_contract_affine_numerator_over_sqrt_quadratic_splits_by_linearity() {
    // `(a·x+b)/√(quadratic)` used to decline even though both pieces work (`x/√q → √q`,
    // `c/√q → asinh/arcsin/acosh`): the integrand normalizes to `(q)^(-1/2)·(a·x+b)` (a product, not
    // a Div) and `expand` will not distribute over the fractional-power factor. Distributing the sum
    // over the radical before integration now closes the whole asinh/arcsin/acosh family. Each is
    // certified by differentiating the antiderivative back to the integrand.
    for input in [
        "integrate((x+1)/sqrt(x^2+1), x)",     // asinh(x) + sqrt(x^2+1)
        "integrate((2*x+3)/sqrt(x^2+1), x)",   // 2 sqrt(x^2+1) + 3 asinh(x)
        "integrate((x+1)/sqrt(1-x^2), x)",     // arcsin(x) - sqrt(1-x^2)
        "integrate((x-2)/sqrt(x^2-1), x)",     // acosh family
        "integrate((x^2+x+1)/sqrt(x^2+1), x)", // higher-degree numerator also splits
    ] {
        assert_antiderivative_verifies(input);
    }
    // The single-term owners are unchanged (still verify), and a product WITH sqrt (positive half) is
    // untouched by the reciprocal-only split.
    for input in [
        "integrate(x/sqrt(x^2+1), x)",
        "integrate(1/sqrt(x^2+1), x)",
        "integrate((x+1)*sqrt(x^2+1), x)",
    ] {
        assert_antiderivative_verifies(input);
    }
}

#[test]
fn integrate_contract_reciprocal_cos_power_via_u_substitution() {
    // `sin(x)/cos(x)^n` for n >= 4 (and `sin^odd/cos^n`) declined: the polynomial-only odd-power owner
    // does not accept a NEGATIVE companion power. The u = cos substitution `∫ sin^p cos^q dx`
    // (p odd) now integrates the companion by the power rule for any integer q, closing the gap.
    // Certified by differentiating the antiderivative back to the integrand.
    let r = |input: &str| evaluated_integral_with_required_conditions(input).0;
    // Result-form pins for the newly-closed cases (the antiderivative differentiates back, checked by
    // hand and numerically): u = cos gives `1/((n-1)cos^(n-1))`. Affine argument scales by 1/a.
    assert_eq!(r("integrate(sin(x)/cos(x)^4, x)"), "1 / (3 * cos(x)^3)");
    assert_eq!(r("integrate(sin(x)/cos(x)^5, x)"), "1 / (4 * cos(x)^4)");
    assert_eq!(
        r("integrate(sin(2*x)/cos(2*x)^4, x)"),
        "1 / (6 * cos(2 * x)^3)"
    );
    // The odd-numerator-power case (the (1-u^2) expansion branch); form verified by hand/numerically
    // to differentiate back to sin(x)^3/cos(x)^4 (a result pin, not the slow simplify-and-diff path).
    assert_eq!(
        r("integrate(sin(x)^3/cos(x)^4, x)"),
        "(-3/2 * (2 * cos(x)^2 - 1) - 1/2) / (3 * cos(x)^3)"
    );
    // Existing owners keep their forms (fallback placement): pins guard against the u-substitution
    // accidentally taking over the canonical sec/tan/polynomial spellings.
    assert_eq!(r("integrate(sin(x)/cos(x)^2, x)"), "sec(x)");
    assert_eq!(r("integrate(sin(x)/cos(x)^3, x)"), "tan(x)^2 / 2");
    assert_eq!(
        r("integrate(sin(x)^3*cos(x)^2, x)"),
        "1/15 * (3 * cos(x)^5 - 5 * cos(x)^3)"
    );
}

#[test]
fn integrate_contract_asinh_kernel() {
    assert_eq!(
        simplified_integral("integrate((x^2+1)^(-1/2), x)"),
        "asinh(x)"
    );
}

#[test]
fn integrate_contract_secant_squared_kernel_uses_tangent_and_nonzero_domain() {
    let (result, required) = evaluated_integral_with_required_conditions("integrate(sec(x)^2, x)");

    assert_eq!(result, "tan(x)");
    assert_eq!(
        required,
        vec!["cos(x) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}

#[test]
fn integrate_contract_cosecant_squared_kernel_uses_cotangent_and_nonzero_domain() {
    let (result, required) = evaluated_integral_with_required_conditions("integrate(csc(x)^2, x)");

    assert_eq!(result, "-cot(x)");
    assert_eq!(
        required,
        vec!["sin(x) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}

#[test]
fn integrate_contract_linear_secant_squared_substitution_preserves_nonzero_domain() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(sec(2*x + 1)^2, x)");

    assert_eq!(result, "1/2 * tan(2 * x + 1)");
    assert_eq!(
        required,
        vec!["cos(2 * x + 1) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}

#[test]
fn integrate_contract_linear_cosecant_squared_substitution_preserves_nonzero_domain() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(csc(2*x + 1)^2, x)");

    assert_eq!(result, "-cot(2 * x + 1) / 2");
    assert_eq!(
        required,
        vec!["sin(2 * x + 1) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}

#[test]
fn integrate_contract_polynomial_secant_squared_substitution_preserves_nonzero_domain() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(x/(cos(x^2)^2), x)");

    assert_eq!(result, "tan(x^2) / 2");
    assert_eq!(
        required,
        vec!["cos(x^2) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}

#[test]
fn integrate_contract_polynomial_cosecant_squared_substitution_preserves_nonzero_domain() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(x^2/(sin(x^3)^2), x)");

    assert_eq!(result, "-cot(x^3) / 3");
    assert_eq!(
        required,
        vec!["sin(x^3) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}

#[test]
fn integrate_contract_raw_reciprocal_trig_derivative_quotients_render_compactly() {
    let cases = [
        (
            "integrate(x*sin(x^2)/cos(x^2)^2, x)",
            "sec(x^2) / 2",
            "cos(x^2) ≠ 0",
            "diff(integrate(x*sin(x^2)/cos(x^2)^2, x), x) - x*sin(x^2)/cos(x^2)^2",
        ),
        (
            "integrate(x^2*cos(x^3)/sin(x^3)^2, x)",
            "-csc(x^3) / 3",
            "sin(x^3) ≠ 0",
            "diff(integrate(x^2*cos(x^3)/sin(x^3)^2, x), x) - x^2*cos(x^3)/sin(x^3)^2",
        ),
        (
            "integrate((2*x+1)*sin(x^2+x)/cos(x^2+x)^2, x)",
            "sec(x^2 + x)",
            "cos(x^2 + x) ≠ 0",
            "diff(integrate((2*x+1)*sin(x^2+x)/cos(x^2+x)^2, x), x) - (2*x+1)*sin(x^2+x)/cos(x^2+x)^2",
        ),
        (
            "integrate((2*x+1)*cos(x^2+x)/sin(x^2+x)^2, x)",
            "-csc(x^2 + x)",
            "sin(x^2 + x) ≠ 0",
            "diff(integrate((2*x+1)*cos(x^2+x)/sin(x^2+x)^2, x), x) - (2*x+1)*cos(x^2+x)/sin(x^2+x)^2",
        ),
        (
            "integrate((3*sin(x^2+x)+6*x*sin(x^2+x))/cos(x^2+x)^2, x)",
            "3 * sec(x^2 + x)",
            "cos(x^2 + x) ≠ 0",
            "diff(integrate(3*(2*x+1)*sin(x^2+x)/cos(x^2+x)^2, x), x) - 3*(2*x+1)*sin(x^2+x)/cos(x^2+x)^2",
        ),
        (
            "integrate((3*cos(x^2+x)+6*x*cos(x^2+x))/sin(x^2+x)^2, x)",
            "-3 * csc(x^2 + x)",
            "sin(x^2 + x) ≠ 0",
            "diff(integrate(3*(2*x+1)*cos(x^2+x)/sin(x^2+x)^2, x), x) - 3*(2*x+1)*cos(x^2+x)/sin(x^2+x)^2",
        ),
    ];

    for (input, expected_result, expected_condition, residual_input) in cases {
        let (result, required) = evaluated_integral_with_required_conditions(input);

        assert_eq!(result, expected_result, "unexpected result for {input}");
        assert_eq!(
            required,
            vec![expected_condition.to_string()],
            "unexpected required_conditions for {input}: {required:?}"
        );
        assert_rendered_antiderivative_verifies(input, &result);

        let (residual_result, residual_required) =
            evaluated_expr_with_required_conditions(residual_input);
        assert_eq!(
            residual_result, "0",
            "unexpected antiderivative residual for {input}"
        );
        assert_eq!(
            residual_required,
            vec![expected_condition.to_string()],
            "residual should preserve required domain for {input}: {residual_required:?}"
        );
    }
}

#[test]
fn integrate_contract_polynomial_trig_derivative_substitution_preserves_compact_arg() {
    let input = "integrate((4*x^3-2*x)*sin(x^4-x^2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "-cos(x^4 - x^2)");
    assert!(
        required.is_empty(),
        "polynomial sine substitution should not add required conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);

    let input = "integrate((4*x^3-2*x)*cos(x^4-x^2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "sin(x^4 - x^2)");
    assert!(
        required.is_empty(),
        "polynomial cosine substitution should not add required conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
}

#[test]
fn integrate_contract_expanded_polynomial_tangent_cotangent_preserves_domain() {
    let input = "integrate((4*x^3-2*x)*tan(x^4-x^2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "-ln(|cos(x^4 - x^2)|)");
    assert_eq!(
        required,
        vec!["cos(x^4 - x^2) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_eq!(integrate_call_antiderivative_residual_result(input), "0");

    let input = "integrate((4*x^3-2*x)*cot(x^4-x^2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "ln(|sin(x^4 - x^2)|)");
    assert_eq!(
        required,
        vec!["sin(x^4 - x^2) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_eq!(integrate_call_antiderivative_residual_result(input), "0");
}

#[test]
fn integrate_contract_linear_secant_uses_abs_log_and_nonzero_domain() {
    let (result, _required) =
        evaluated_integral_with_required_conditions("integrate(sec(2*x + 1), x)");

    assert_eq!(result, "1/2 * ln(|tan(2 * x + 1) + sec(2 * x + 1)|)");
    let (wire, stderr) = cli_eval_json_with_stderr("integrate(sec(2*x + 1), x)");
    assert!(stderr.is_empty(), "unexpected stderr: {stderr}");
    assert_eq!(
        wire["required_display"],
        serde_json::json!(["cos(2·x + 1) ≠ 0"]),
        "unexpected public required_display: {:?}",
        wire["required_display"]
    );
    assert_antiderivative_verifies("integrate(sec(2*x + 1), x)");
    let (nested_wire, nested_stderr) =
        cli_eval_json_with_stderr("diff(integrate(sec(2*x+1), x), x) - sec(2*x+1)");
    assert!(
        nested_stderr.is_empty(),
        "unexpected stderr: {nested_stderr}"
    );
    assert_eq!(nested_wire["result"], "0");
    assert_eq!(
        nested_wire["required_display"],
        serde_json::json!(["cos(2·x + 1) ≠ 0"]),
        "secant log primitive verification should preserve the trig pole condition"
    );
}

#[test]
fn integrate_contract_linear_cosecant_uses_abs_log_and_nonzero_domain() {
    let (result, _required) =
        evaluated_integral_with_required_conditions("integrate(csc(2*x + 1), x)");

    assert_eq!(result, "1/2 * ln(|csc(2 * x + 1) - cot(2 * x + 1)|)");
    let (wire, stderr) = cli_eval_json_with_stderr("integrate(csc(2*x + 1), x)");
    assert!(stderr.is_empty(), "unexpected stderr: {stderr}");
    assert_eq!(
        wire["required_display"],
        serde_json::json!(["sin(2·x + 1) ≠ 0"]),
        "unexpected public required_display: {:?}",
        wire["required_display"]
    );
    let (nested_wire, nested_stderr) =
        cli_eval_json_with_stderr("diff(integrate(csc(2*x+1), x), x) - csc(2*x+1)");
    assert!(
        nested_stderr.is_empty(),
        "unexpected stderr: {nested_stderr}"
    );
    assert_eq!(nested_wire["result"], "0");
    assert_eq!(
        nested_wire["required_display"],
        serde_json::json!(["sin(2·x + 1) ≠ 0"]),
        "cosecant log primitive verification should preserve the trig pole condition"
    );
}

#[test]
fn integrate_contract_scaled_affine_secant_cosecant_uses_abs_log_and_nonzero_domain() {
    let cases = [
        (
            "integrate(sec((3*x+2)/2), x)",
            "2/3 * ln(|tan((3 * x + 2) / 2) + sec((3 * x + 2) / 2)|)",
            "cos((3 * x + 2) / 2) ≠ 0",
        ),
        (
            "integrate(csc((2-3*x)/2), x)",
            "-2/3 * ln(|csc((2 - 3 * x) / 2) - cot((2 - 3 * x) / 2)|)",
            "sin((2 - 3 * x) / 2) ≠ 0",
        ),
    ];

    for (input, expected_result, expected_condition) in cases {
        let (result, _required) = evaluated_integral_with_required_conditions(input);

        assert_eq!(result, expected_result, "unexpected result for {input}");
        let (wire, stderr) = cli_eval_json_with_stderr(input);
        assert!(stderr.is_empty(), "unexpected stderr for {input}: {stderr}");
        assert_eq!(
            wire["required_display"],
            serde_json::json!([expected_condition.replace(" * ", "·")]),
            "unexpected public required_display for {input}: {:?}",
            wire["required_display"]
        );
        assert_rendered_antiderivative_verifies(input, &result);
    }
}

#[test]
fn integrate_contract_nested_reciprocal_trig_residual_verifies_antiderivative() {
    let cases = [
        (
            "diff(integrate(sec((3*x+2)/2), x), x) - sec((3*x+2)/2)",
            "cos((3 * x + 2) / 2) ≠ 0",
        ),
        (
            "diff(integrate(csc((2-3*x)/2), x), x) - csc((2-3*x)/2)",
            "sin((2 - 3 * x) / 2) ≠ 0",
        ),
        (
            "diff(integrate(1/sin((2-3*x)/2), x), x) - 1/sin((2-3*x)/2)",
            "sin((2 - 3 * x) / 2) ≠ 0",
        ),
        (
            "diff(integrate(2*x*sec(x^2)*tan(x^2), x), x) - 2*x*sec(x^2)*tan(x^2)",
            "cos(x^2) ≠ 0",
        ),
        (
            "diff(integrate(2*x*csc(x^2)*cot(x^2), x), x) - 2*x*csc(x^2)*cot(x^2)",
            "sin(x^2) ≠ 0",
        ),
        (
            "diff(integrate((4*x^3-2*x)*sec(x^4-x^2)*tan(x^4-x^2), x), x) - (4*x^3-2*x)*sec(x^4-x^2)*tan(x^4-x^2)",
            "cos(x^4 - x^2) ≠ 0",
        ),
        (
            "diff(integrate((4*x^3-2*x)*csc(x^4-x^2)*cot(x^4-x^2), x), x) - (4*x^3-2*x)*csc(x^4-x^2)*cot(x^4-x^2)",
            "sin(x^4 - x^2) ≠ 0",
        ),
    ];

    for (input, expected_condition) in cases {
        let (wire, stderr) = cli_eval_json_with_stderr(input);
        assert!(stderr.is_empty(), "unexpected stderr for {input}: {stderr}");
        assert_eq!(
            wire["result"], "0",
            "unexpected nested residual for {input}"
        );
        assert_eq!(
            wire["required_display"],
            serde_json::json!([expected_condition.replace(" * ", "·")]),
            "unexpected required_display for nested residual {input}: {:?}",
            wire["required_display"]
        );
    }
}

#[test]
fn integrate_contract_wrapped_nested_reciprocal_trig_residual_verifies_antiderivative() {
    let cases = [
        (
            "(diff(integrate((4*x^3-2*x)*sec(x^4-x^2)*tan(x^4-x^2), x), x) - (4*x^3-2*x)*sec(x^4-x^2)*tan(x^4-x^2)) + 0",
            vec!["cos(x^4 - x^2) ≠ 0"],
        ),
        (
            "2*(diff(integrate((4*x^3-2*x)*csc(x^4-x^2)*cot(x^4-x^2), x), x) - (4*x^3-2*x)*csc(x^4-x^2)*cot(x^4-x^2))",
            vec!["sin(x^4 - x^2) ≠ 0"],
        ),
        (
            "(diff(integrate((4*x^3-2*x)*sec(x^4-x^2)*tan(x^4-x^2), x), x) - (4*x^3-2*x)*sec(x^4-x^2)*tan(x^4-x^2))/(x+1)",
            vec!["cos(x^4 - x^2) ≠ 0", "x ≠ -1"],
        ),
    ];

    for (input, expected_conditions) in cases {
        let (result, required) = evaluated_expr_with_required_conditions(input);
        assert_eq!(result, "0", "unexpected wrapped residual for {input}");
        assert_eq!(
            required, expected_conditions,
            "wrapped residual should preserve required domain for {input}: {required:?}"
        );
    }
}

#[test]
fn integrate_contract_linear_tangent_uses_abs_log_and_nonzero_domain() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(tan(2*x + 1), x)");

    assert_eq!(result, "-1/2 * ln(|cos(2 * x + 1)|)");
    assert_eq!(
        required,
        vec!["cos(2 * x + 1) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}

#[test]
fn integrate_contract_linear_cotangent_uses_abs_log_and_nonzero_domain() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(cot(2*x + 1), x)");

    assert_eq!(result, "1/2 * ln(|sin(2 * x + 1)|)");
    assert_eq!(
        required,
        vec!["sin(2 * x + 1) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}

#[test]
fn integrate_contract_nested_tangent_cotangent_residual_verifies_antiderivative() {
    let cases = [
        (
            "diff(integrate(tan(2*x+1), x), x) - tan(2*x+1)",
            "cos(2 * x + 1) ≠ 0",
        ),
        (
            "diff(integrate(cot(2*x+1), x), x) - cot(2*x+1)",
            "sin(2 * x + 1) ≠ 0",
        ),
        (
            "diff(integrate(cos(2*x+1)/sin(2*x+1), x), x) - cos(2*x+1)/sin(2*x+1)",
            "sin(2 * x + 1) ≠ 0",
        ),
        (
            "diff(integrate(2*x*tan(x^2), x), x) - 2*x*tan(x^2)",
            "cos(x^2) ≠ 0",
        ),
        (
            "diff(integrate(3*x^2*cot(x^3), x), x) - 3*x^2*cot(x^3)",
            "sin(x^3) ≠ 0",
        ),
        (
            "diff(integrate((4*x^3-2*x)*tan(x^4-x^2), x), x) - (4*x^3-2*x)*tan(x^4-x^2)",
            "cos(x^4 - x^2) ≠ 0",
        ),
        (
            "diff(integrate((4*x^3-2*x)*cot(x^4-x^2), x), x) - (4*x^3-2*x)*cot(x^4-x^2)",
            "sin(x^4 - x^2) ≠ 0",
        ),
        (
            "diff(integrate(2*x*sec(x^2), x), x) - 2*x*sec(x^2)",
            "cos(x^2) ≠ 0",
        ),
        (
            "diff(integrate(2*x*csc(x^2), x), x) - 2*x*csc(x^2)",
            "sin(x^2) ≠ 0",
        ),
    ];

    for (input, expected_condition) in cases {
        let (wire, stderr) = cli_eval_json_with_stderr(input);
        assert!(stderr.is_empty(), "unexpected stderr for {input}: {stderr}");
        assert_eq!(
            wire["result"], "0",
            "unexpected nested residual for {input}"
        );
        assert_eq!(
            wire["required_display"],
            serde_json::json!([expected_condition.replace(" * ", "·")]),
            "unexpected required_display for nested residual {input}: {:?}",
            wire["required_display"]
        );
    }
}

#[test]
fn integrate_contract_wrapped_nested_tangent_cotangent_residual_verifies_antiderivative() {
    let cases = [
        (
            "(diff(integrate((4*x^3-2*x)*tan(x^4-x^2), x), x) - (4*x^3-2*x)*tan(x^4-x^2)) + 0",
            vec!["cos(x^4 - x^2) ≠ 0"],
        ),
        (
            "2*(diff(integrate((4*x^3-2*x)*cot(x^4-x^2), x), x) - (4*x^3-2*x)*cot(x^4-x^2))",
            vec!["sin(x^4 - x^2) ≠ 0"],
        ),
        (
            "(diff(integrate((4*x^3-2*x)*cot(x^4-x^2), x), x) - (4*x^3-2*x)*cot(x^4-x^2))/(x+1)",
            vec!["sin(x^4 - x^2) ≠ 0", "x ≠ -1"],
        ),
    ];

    for (input, expected_conditions) in cases {
        let (result, required) = evaluated_expr_with_required_conditions(input);
        assert_eq!(result, "0", "unexpected wrapped residual for {input}");
        assert_eq!(
            required, expected_conditions,
            "wrapped residual should preserve required domain for {input}: {required:?}"
        );
    }
}

#[test]
fn integrate_contract_polynomial_tangent_uses_abs_log_and_nonzero_domain() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(2*x*tan(x^2), x)");

    assert_eq!(result, "-ln(|cos(x^2)|)");
    assert_eq!(
        required,
        vec!["cos(x^2) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}

#[test]
fn integrate_contract_polynomial_cotangent_uses_abs_log_and_nonzero_domain() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(3*x^2*cot(x^3), x)");

    assert_eq!(result, "ln(|sin(x^3)|)");
    assert_eq!(
        required,
        vec!["sin(x^3) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}

#[test]
fn integrate_contract_polynomial_secant_uses_abs_log_and_nonzero_domain() {
    let (result, _required) =
        evaluated_integral_with_required_conditions("integrate(2*x*sec(x^2), x)");

    assert_eq!(result, "ln(|tan(x^2) + sec(x^2)|)");
    let (wire, stderr) = cli_eval_json_with_stderr("integrate(2*x*sec(x^2), x)");
    assert!(stderr.is_empty(), "unexpected stderr: {stderr}");
    assert_eq!(
        wire["required_display"],
        serde_json::json!(["cos(x^2) ≠ 0"]),
        "unexpected public required_display: {:?}",
        wire["required_display"]
    );
}

#[test]
fn integrate_contract_polynomial_cosecant_uses_abs_log_and_nonzero_domain() {
    let (result, _required) =
        evaluated_integral_with_required_conditions("integrate(2*x*csc(x^2), x)");

    assert_eq!(result, "ln(|csc(x^2) - cot(x^2)|)");
    let (wire, stderr) = cli_eval_json_with_stderr("integrate(2*x*csc(x^2), x)");
    assert!(stderr.is_empty(), "unexpected stderr: {stderr}");
    assert_eq!(
        wire["required_display"],
        serde_json::json!(["sin(x^2) ≠ 0"]),
        "unexpected public required_display: {:?}",
        wire["required_display"]
    );
}

#[test]
fn integrate_contract_presimplified_reciprocal_secant_and_cosecant_use_source_domain() {
    let (sec_wire, sec_stderr) = cli_eval_json_with_stderr("integrate(2*x/sec(x^2+0), x)");
    assert!(sec_stderr.is_empty(), "unexpected stderr: {sec_stderr}");
    assert_eq!(sec_wire["result"], serde_json::json!("sin(x^2)"));
    assert_eq!(
        sec_wire["required_display"],
        serde_json::json!(["cos(x^2) ≠ 0"]),
        "unexpected secant reciprocal required_display: {:?}",
        sec_wire["required_display"]
    );

    let (csc_wire, csc_stderr) = cli_eval_json_with_stderr("integrate(2*x/csc(x^2+0), x)");
    assert!(csc_stderr.is_empty(), "unexpected stderr: {csc_stderr}");
    assert_eq!(csc_wire["result"], serde_json::json!("-cos(x^2)"));
    assert_eq!(
        csc_wire["required_display"],
        serde_json::json!(["sin(x^2) ≠ 0"]),
        "unexpected cosecant reciprocal required_display: {:?}",
        csc_wire["required_display"]
    );
}

#[test]
fn integrate_contract_presimplified_reciprocal_tangent_and_cotangent_use_source_domain() {
    let (tan_wire, tan_stderr) = cli_eval_json_with_stderr("integrate(2*x/tan(x^2+0), x)");
    assert!(tan_stderr.is_empty(), "unexpected stderr: {tan_stderr}");
    assert_eq!(tan_wire["result"], serde_json::json!("ln(|sin(x^2)|)"));
    assert_eq!(
        tan_wire["required_display"],
        serde_json::json!(["sin(x^2) ≠ 0", "cos(x^2) ≠ 0"]),
        "unexpected tangent reciprocal required_display: {:?}",
        tan_wire["required_display"]
    );

    let (cot_wire, cot_stderr) = cli_eval_json_with_stderr("integrate(2*x/cot(x^2+0), x)");
    assert!(cot_stderr.is_empty(), "unexpected stderr: {cot_stderr}");
    assert_eq!(cot_wire["result"], serde_json::json!("-ln(|cos(x^2)|)"));
    assert_eq!(
        cot_wire["required_display"],
        serde_json::json!(["cos(x^2) ≠ 0", "sin(x^2) ≠ 0"]),
        "unexpected cotangent reciprocal required_display: {:?}",
        cot_wire["required_display"]
    );
}

#[test]
fn integrate_contract_polynomial_trig_log_substitution_explains_u_and_du() {
    for (
        input,
        expected_result,
        expected_required_display,
        expected_rule_title,
        expects_constant_adjustment,
    ) in [
        (
            "integrate(x*tan(x^2), x)",
            "-1/2·ln(|cos(x^2)|)",
            serde_json::json!(["cos(x^2) ≠ 0"]),
            "Usar la regla de tan(u) -> -ln|cos(u)|",
            true,
        ),
        (
            "integrate(x*cot(x^2), x)",
            "1/2·ln(|sin(x^2)|)",
            serde_json::json!(["sin(x^2) ≠ 0"]),
            "Usar la regla de cot(u) -> ln|sin(u)|",
            true,
        ),
        (
            "integrate(2*x*sec(x^2), x)",
            "ln(|tan(x^2) + sec(x^2)|)",
            serde_json::json!(["cos(x^2) ≠ 0"]),
            "Usar la regla de sec(u) -> ln|sec(u)+tan(u)|",
            false,
        ),
        (
            "integrate(x*sec(x^2), x)",
            "1/2·ln(|tan(x^2) + sec(x^2)|)",
            serde_json::json!(["cos(x^2) ≠ 0"]),
            "Usar la regla de sec(u) -> ln|sec(u)+tan(u)|",
            true,
        ),
        (
            "integrate(2*x*csc(x^2), x)",
            "ln(|csc(x^2) - cot(x^2)|)",
            serde_json::json!(["sin(x^2) ≠ 0"]),
            "Usar la regla de csc(u) -> ln|csc(u)-cot(u)|",
            false,
        ),
        (
            "integrate(x*csc(x^2), x)",
            "1/2·ln(|csc(x^2) - cot(x^2)|)",
            serde_json::json!(["sin(x^2) ≠ 0"]),
            "Usar la regla de csc(u) -> ln|csc(u)-cot(u)|",
            true,
        ),
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

        assert_eq!(wire["result"], expected_result, "input: {input}");
        assert_eq!(
            wire["required_display"], expected_required_display,
            "unexpected required_display for {input}: {:?}",
            wire["required_display"]
        );
        assert!(
            !stderr.contains("depth_overflow"),
            "polynomial trig log trace should not emit depth_overflow warning for {input}\nstderr:\n{stderr}"
        );

        let steps = wire["steps"]
            .as_array()
            .expect("steps should be present with --steps on");
        let integration_step = steps
            .iter()
            .find(|step| step["rule"] == "Calcular la integral")
            .expect("expected public symbolic integration step");
        let substeps = integration_step["substeps"]
            .as_array()
            .expect("integration step should expose didactic substeps");
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == expected_rule_title),
            "expected {expected_rule_title} substep for {input}, got {substeps:?}"
        );
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == "Identificar u y du"),
            "expected concrete u/du substep for {input}, got {substeps:?}"
        );
        assert_u_du_substep_labels(substeps, input);
        assert_eq!(
            substeps
                .iter()
                .any(|substep| substep["title"] == "Ajustar el factor constante"),
            expects_constant_adjustment,
            "unexpected constant adjustment substep presence for {input}: {substeps:?}"
        );
        assert!(
            substeps
                .iter()
                .all(|substep| substep["title"] != "Usar sustitución"),
            "polynomial trig log table case should not use the generic substitution substep for {input}: {substeps:?}"
        );
    }
}

#[test]
fn integrate_contract_reciprocal_trig_derivative_product_explains_u_and_du() {
    for (
        input,
        expected_result,
        expected_required_display,
        expected_rule_title,
        expects_constant_adjustment,
    ) in [
        (
            "integrate(x*sec(x^2)*tan(x^2), x)",
            "sec(x^2) / 2",
            serde_json::json!(["cos(x^2) ≠ 0"]),
            "Usar la regla de sec(u)·tan(u) -> sec(u)",
            true,
        ),
        (
            "integrate(2*x*sec(x^2)*tan(x^2), x)",
            "sec(x^2)",
            serde_json::json!(["cos(x^2) ≠ 0"]),
            "Usar la regla de sec(u)·tan(u) -> sec(u)",
            false,
        ),
        (
            "integrate(x*csc(x^2)*cot(x^2), x)",
            "-csc(x^2) / 2",
            serde_json::json!(["sin(x^2) ≠ 0"]),
            "Usar la regla de csc(u)·cot(u) -> -csc(u)",
            true,
        ),
        (
            "integrate(3*x^2*csc(x^3)*cot(x^3), x)",
            "-csc(x^3)",
            serde_json::json!(["sin(x^3) ≠ 0"]),
            "Usar la regla de csc(u)·cot(u) -> -csc(u)",
            false,
        ),
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

        assert_eq!(wire["result"], expected_result, "input: {input}");
        assert_eq!(
            wire["required_display"], expected_required_display,
            "unexpected required_display for {input}: {:?}",
            wire["required_display"]
        );
        assert!(
            !stderr.contains("depth_overflow"),
            "reciprocal trig derivative product trace should not emit depth_overflow warning for {input}\nstderr:\n{stderr}"
        );

        let steps = wire["steps"]
            .as_array()
            .expect("steps should be present with --steps on");
        let integration_step = steps
            .iter()
            .find(|step| step["rule"] == "Calcular la integral")
            .expect("expected public symbolic integration step");
        let substeps = integration_step["substeps"]
            .as_array()
            .expect("integration step should expose didactic substeps");
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == expected_rule_title),
            "expected {expected_rule_title} substep for {input}, got {substeps:?}"
        );
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == "Identificar u y du"),
            "expected concrete u/du substep for {input}, got {substeps:?}"
        );
        assert_u_du_substep_labels(substeps, input);
        assert_eq!(
            substeps
                .iter()
                .any(|substep| substep["title"] == "Ajustar el factor constante"),
            expects_constant_adjustment,
            "unexpected constant adjustment substep presence for {input}: {substeps:?}"
        );
        assert!(
            substeps
                .iter()
                .all(|substep| substep["title"] != "Usar sustitución"),
            "reciprocal trig derivative product should not use the generic substitution substep for {input}: {substeps:?}"
        );
    }
}

#[test]
fn integrate_contract_polynomial_trig_log_explicit_ratios_preserve_source_domain() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(2*(x*sin(x^2)/cos(x^2)), x)");

    assert_eq!(result, "-ln(|cos(x^2)|)");
    assert_eq!(
        required,
        vec!["cos(x^2) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(3*(x^2*cos(x^3)/sin(x^3)), x)");

    assert_eq!(result, "ln(|sin(x^3)|)");
    assert_eq!(
        required,
        vec!["sin(x^3) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}

#[test]
fn integrate_contract_linear_secant_tangent_product_preserves_nonzero_domain() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(sec(2*x + 1)*tan(2*x + 1), x)");

    assert_eq!(result, "sec(2 * x + 1) / 2");
    assert_eq!(
        required,
        vec!["cos(2 * x + 1) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}

#[test]
fn integrate_contract_linear_cosecant_cotangent_product_preserves_nonzero_domain() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(csc(2*x + 1)*cot(2*x + 1), x)");

    assert_eq!(result, "-csc(2 * x + 1) / 2");
    assert_eq!(
        required,
        vec!["sin(2 * x + 1) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}

#[test]
fn integrate_contract_polynomial_secant_tangent_product_preserves_nonzero_domain() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(x*sec(x^2)*tan(x^2), x)");

    assert_eq!(result, "sec(x^2) / 2");
    assert_eq!(
        required,
        vec!["cos(x^2) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}

#[test]
fn integrate_contract_polynomial_cosecant_cotangent_product_preserves_nonzero_domain() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(x^2*csc(x^3)*cot(x^3), x)");

    assert_eq!(result, "-csc(x^3) / 3");
    assert_eq!(
        required,
        vec!["sin(x^3) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}

#[test]
fn integrate_contract_exact_polynomial_secant_tangent_product_uses_clean_antiderivative() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(2*x*sec(x^2)*tan(x^2), x)");

    assert_eq!(result, "sec(x^2)");
    assert_eq!(
        required,
        vec!["cos(x^2) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}

#[test]
fn integrate_contract_exact_polynomial_cosecant_cotangent_product_uses_clean_antiderivative() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(3*x^2*csc(x^3)*cot(x^3), x)");

    assert_eq!(result, "-csc(x^3)");
    assert_eq!(
        required,
        vec!["sin(x^3) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}

#[test]
fn integrate_contract_exact_polynomial_secant_squared_preserves_nonzero_domain() {
    let input = "integrate(2*x*sec(x^2)^2, x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "tan(x^2)");
    assert_eq!(
        required,
        vec!["cos(x^2) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
}

#[test]
fn integrate_contract_exact_polynomial_cosecant_squared_preserves_nonzero_domain() {
    let input = "integrate(2*x*csc(x^2)^2, x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "-cot(x^2)");
    assert_eq!(
        required,
        vec!["sin(x^2) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
}

#[test]
fn integrate_contract_sqrt_chain_secant_cosecant_products_verify() {
    let cases = [
        (
            "integrate(sec(sqrt(x))*tan(sqrt(x))/(2*sqrt(x)), x)",
            "sec(sqrt(x))",
            "tan(sqrt(x)) * sec(sqrt(x)) / (2 * sqrt(x))",
            vec!["cos(sqrt(x)) ≠ 0", "x > 0"],
        ),
        (
            "integrate(-sec(sqrt(x))*tan(sqrt(x))/(2*sqrt(x)), x)",
            "-sec(sqrt(x))",
            "-tan(sqrt(x)) * sec(sqrt(x)) / (2 * sqrt(x))",
            vec!["cos(sqrt(x)) ≠ 0", "x > 0"],
        ),
        (
            "integrate(csc(sqrt(x))*cot(sqrt(x))/(2*sqrt(x)), x)",
            "-csc(sqrt(x))",
            "csc(sqrt(x)) * cot(sqrt(x)) / (2 * sqrt(x))",
            vec!["sin(sqrt(x)) ≠ 0", "x > 0"],
        ),
        (
            "integrate(-csc(sqrt(x))*cot(sqrt(x))/(2*sqrt(x)), x)",
            "csc(sqrt(x))",
            "-cot(sqrt(x)) * csc(sqrt(x)) / (2 * sqrt(x))",
            vec!["sin(sqrt(x)) ≠ 0", "x > 0"],
        ),
        (
            "integrate(sec(sqrt(2*x))*tan(sqrt(2*x))/sqrt(2*x), x)",
            "sec(sqrt(2 * x))",
            "tan(sqrt(2 * x)) * sec(sqrt(2 * x)) / sqrt(2 * x)",
            vec!["cos(sqrt(2 * x)) ≠ 0", "x > 0"],
        ),
        (
            "integrate(csc(sqrt(2*x))*cot(sqrt(2*x))/sqrt(2*x), x)",
            "-csc(sqrt(2 * x))",
            "csc(sqrt(2 * x)) * cot(sqrt(2 * x)) / sqrt(2 * x)",
            vec!["sin(sqrt(2 * x)) ≠ 0", "x > 0"],
        ),
        (
            "integrate(-sec(sqrt(2*x))*tan(sqrt(2*x))/sqrt(2*x), x)",
            "-sec(sqrt(2 * x))",
            "-tan(sqrt(2 * x)) * sec(sqrt(2 * x)) / sqrt(2 * x)",
            vec!["cos(sqrt(2 * x)) ≠ 0", "x > 0"],
        ),
        (
            "integrate(-csc(sqrt(2*x))*cot(sqrt(2*x))/sqrt(2*x), x)",
            "csc(sqrt(2 * x))",
            "-cot(sqrt(2 * x)) * csc(sqrt(2 * x)) / sqrt(2 * x)",
            vec!["sin(sqrt(2 * x)) ≠ 0", "x > 0"],
        ),
        (
            "integrate(sec(sqrt(3*x+1))*tan(sqrt(3*x+1))*3/(2*sqrt(3*x+1)), x)",
            "sec(sqrt(3 * x + 1))",
            "3 * tan(sqrt(3 * x + 1)) * sec(sqrt(3 * x + 1)) / (2 * sqrt(3 * x + 1))",
            vec!["cos(sqrt(3 * x + 1)) ≠ 0", "x > -1/3"],
        ),
        (
            "integrate(-sec(sqrt(3-2*x))*tan(sqrt(3-2*x))/sqrt(3-2*x), x)",
            "sec(sqrt(3 - 2 * x))",
            "-tan(sqrt(3 - 2 * x)) * sec(sqrt(3 - 2 * x)) / sqrt(3 - 2 * x)",
            vec!["cos(sqrt(3 - 2 * x)) ≠ 0", "x < 3/2"],
        ),
        (
            "integrate(sec(sqrt(3-2*x))*tan(sqrt(3-2*x))/sqrt(3-2*x), x)",
            "-sec(sqrt(3 - 2 * x))",
            "tan(sqrt(3 - 2 * x)) * sec(sqrt(3 - 2 * x)) / sqrt(3 - 2 * x)",
            vec!["cos(sqrt(3 - 2 * x)) ≠ 0", "x < 3/2"],
        ),
        (
            "integrate(csc(sqrt(3*x+1))*cot(sqrt(3*x+1))*3/(2*sqrt(3*x+1)), x)",
            "-csc(sqrt(3 * x + 1))",
            "3 * csc(sqrt(3 * x + 1)) * cot(sqrt(3 * x + 1)) / (2 * sqrt(3 * x + 1))",
            vec!["sin(sqrt(3 * x + 1)) ≠ 0", "x > -1/3"],
        ),
        (
            "integrate(-csc(sqrt(3-2*x))*cot(sqrt(3-2*x))/sqrt(3-2*x), x)",
            "-csc(sqrt(3 - 2 * x))",
            "-cot(sqrt(3 - 2 * x)) * csc(sqrt(3 - 2 * x)) / sqrt(3 - 2 * x)",
            vec!["sin(sqrt(3 - 2 * x)) ≠ 0", "x < 3/2"],
        ),
        (
            "integrate(csc(sqrt(3-2*x))*cot(sqrt(3-2*x))/sqrt(3-2*x), x)",
            "csc(sqrt(3 - 2 * x))",
            "csc(sqrt(3 - 2 * x)) * cot(sqrt(3 - 2 * x)) / sqrt(3 - 2 * x)",
            vec!["sin(sqrt(3 - 2 * x)) ≠ 0", "x < 3/2"],
        ),
        (
            "integrate(-csc(sqrt(3*x+1))*cot(sqrt(3*x+1))*3/(2*sqrt(3*x+1)), x)",
            "csc(sqrt(3 * x + 1))",
            "-3 * cot(sqrt(3 * x + 1)) * csc(sqrt(3 * x + 1)) / (2 * sqrt(3 * x + 1))",
            vec!["x > -1/3", "sin(sqrt(3 * x + 1)) ≠ 0"],
        ),
    ];

    for (input, expected_result, expected_nested_result, expected_conditions) in cases {
        let (result, required) = evaluated_integral_with_required_conditions(input);

        assert_eq!(result, expected_result, "unexpected result for {input}");
        assert_eq!(
            required, expected_conditions,
            "unexpected required_conditions for {input}: {required:?}"
        );
        assert_antiderivative_verifies(input);

        let nested_input = format!("diff({input}, x)");
        let (nested_result, nested_required) =
            evaluated_expr_with_required_conditions(&nested_input);
        assert_eq!(
            nested_result, expected_nested_result,
            "unexpected nested diff/integrate result for {input}"
        );
        assert_eq!(
            nested_required, expected_conditions,
            "unexpected nested required_conditions for {input}: {nested_required:?}"
        );
    }
}

#[test]
fn integrate_contract_sqrt_chain_raw_reciprocal_trig_derivative_quotients_render_compactly() {
    let cases = [
        (
            "integrate(sin(sqrt(x))*sqrt(x)/(2*x*cos(sqrt(x))^2), x)",
            "sec(sqrt(x))",
            vec!["cos(sqrt(x)) ≠ 0", "x > 0"],
            "diff(integrate(sin(sqrt(x))*sqrt(x)/(2*x*cos(sqrt(x))^2), x), x) - sin(sqrt(x))*sqrt(x)/(2*x*cos(sqrt(x))^2)",
        ),
        (
            "integrate(cos(sqrt(2*x))*(2*x)^(-1/2)/sin(sqrt(2*x))^2, x)",
            "-csc(sqrt(2 * x))",
            vec!["sin(sqrt(2 * x)) ≠ 0", "x > 0"],
            "diff(integrate(cos(sqrt(2*x))*(2*x)^(-1/2)/sin(sqrt(2*x))^2, x), x) - cos(sqrt(2*x))*(2*x)^(-1/2)/sin(sqrt(2*x))^2",
        ),
        (
            "integrate(3*sin(sqrt(3*x+1))/(2*sqrt(3*x+1)*cos(sqrt(3*x+1))^2), x)",
            "sec(sqrt(3 * x + 1))",
            vec!["cos(sqrt(3 * x + 1)) ≠ 0", "x > -1/3"],
            "diff(integrate(3*sin(sqrt(3*x+1))/(2*sqrt(3*x+1)*cos(sqrt(3*x+1))^2), x), x) - 3*sin(sqrt(3*x+1))/(2*sqrt(3*x+1)*cos(sqrt(3*x+1))^2)",
        ),
        (
            "integrate(-2*cos(sqrt(3-2*x))/(sqrt(3-2*x)*sin(sqrt(3-2*x))^2), x)",
            "-2 * csc(sqrt(3 - 2 * x))",
            vec!["sin(sqrt(3 - 2 * x)) ≠ 0", "x < 3/2"],
            "diff(integrate(-2*cos(sqrt(3-2*x))/(sqrt(3-2*x)*sin(sqrt(3-2*x))^2), x), x) + 2*cos(sqrt(3-2*x))/(sqrt(3-2*x)*sin(sqrt(3-2*x))^2)",
        ),
    ];

    for (input, expected_result, expected_conditions, residual_input) in cases {
        let (result, required) = evaluated_integral_with_required_conditions(input);

        assert_eq!(result, expected_result, "unexpected result for {input}");
        assert_eq!(
            required, expected_conditions,
            "unexpected required_conditions for {input}: {required:?}"
        );
        assert_rendered_antiderivative_verifies(input, &result);

        let (residual_result, residual_required) =
            evaluated_expr_with_required_conditions(residual_input);
        assert_eq!(
            residual_result, "0",
            "unexpected antiderivative residual for {input}"
        );
        assert_eq!(
            residual_required, expected_conditions,
            "residual should preserve required domain for {input}: {residual_required:?}"
        );
    }
}

#[test]
fn integrate_contract_sqrt_chain_reciprocal_trig_products_explain_u_and_du() {
    for (
        input,
        expected_result,
        expected_required_display,
        expected_rule_title,
        expects_constant_adjustment,
    ) in [
        (
            "integrate(sec(sqrt(x))*tan(sqrt(x))/(2*sqrt(x)), x)",
            "sec(sqrt(x))",
            serde_json::json!(["cos(sqrt(x)) ≠ 0", "x > 0"]),
            "Usar la regla de sec(u)·tan(u) -> sec(u)",
            false,
        ),
        (
            "integrate(csc(sqrt(x))*cot(sqrt(x))/(2*sqrt(x)), x)",
            "-csc(sqrt(x))",
            serde_json::json!(["sin(sqrt(x)) ≠ 0", "x > 0"]),
            "Usar la regla de csc(u)·cot(u) -> -csc(u)",
            false,
        ),
        (
            "integrate(sec(sqrt(2*x))*tan(sqrt(2*x))/sqrt(2*x), x)",
            "sec(sqrt(2·x))",
            serde_json::json!(["cos(sqrt(2·x)) ≠ 0", "x > 0"]),
            "Usar la regla de sec(u)·tan(u) -> sec(u)",
            false,
        ),
        (
            "integrate(sin(sqrt(x))*sqrt(x)/(2*x*cos(sqrt(x))^2), x)",
            "sec(sqrt(x))",
            serde_json::json!(["cos(sqrt(x)) ≠ 0", "x > 0"]),
            "Usar la regla de sec(u)·tan(u) -> sec(u)",
            false,
        ),
        (
            "integrate(-sec(sqrt(x))*tan(sqrt(x))/(2*sqrt(x)), x)",
            "-sec(sqrt(x))",
            serde_json::json!(["cos(sqrt(x)) ≠ 0", "x > 0"]),
            "Usar la regla de sec(u)·tan(u) -> sec(u)",
            true,
        ),
        (
            "integrate(-csc(sqrt(x))*cot(sqrt(x))/(2*sqrt(x)), x)",
            "csc(sqrt(x))",
            serde_json::json!(["sin(sqrt(x)) ≠ 0", "x > 0"]),
            "Usar la regla de csc(u)·cot(u) -> -csc(u)",
            true,
        ),
        (
            "integrate(-3*sec(sqrt(3*x+1))*tan(sqrt(3*x+1))/(2*sqrt(3*x+1)), x)",
            "-sec(sqrt(3·x + 1))",
            serde_json::json!(["cos(sqrt(3·x + 1)) ≠ 0", "x > -1/3"]),
            "Usar la regla de sec(u)·tan(u) -> sec(u)",
            true,
        ),
        (
            "integrate(-3*csc(sqrt(3*x+1))*cot(sqrt(3*x+1))/(2*sqrt(3*x+1)), x)",
            "csc(sqrt(3·x + 1))",
            serde_json::json!(["sin(sqrt(3·x + 1)) ≠ 0", "x > -1/3"]),
            "Usar la regla de csc(u)·cot(u) -> -csc(u)",
            true,
        ),
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

        assert_eq!(wire["result"], expected_result, "input: {input}");
        assert_eq!(
            wire["required_display"], expected_required_display,
            "unexpected required_display for {input}: {:?}",
            wire["required_display"]
        );
        assert!(
            !stderr.contains("depth_overflow"),
            "sqrt-chain reciprocal trig product trace should not emit depth_overflow warning for {input}\nstderr:\n{stderr}"
        );

        let steps = wire["steps"]
            .as_array()
            .expect("steps should be present with --steps on");
        let integration_step = steps
            .iter()
            .find(|step| step["rule"] == "Calcular la integral")
            .expect("expected public symbolic integration step");
        let substeps = integration_step["substeps"]
            .as_array()
            .expect("integration step should expose didactic substeps");
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == expected_rule_title),
            "expected {expected_rule_title} substep for {input}, got {substeps:?}"
        );
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == "Identificar u y du"),
            "expected concrete u/du substep for {input}, got {substeps:?}"
        );
        assert_u_du_substep_labels(substeps, input);
        assert_eq!(
            substeps
                .iter()
                .any(|substep| substep["title"] == "Ajustar el factor constante"),
            expects_constant_adjustment,
            "unexpected constant adjustment substep presence for {input}: {substeps:?}"
        );
        assert!(
            substeps
                .iter()
                .all(|substep| substep["title"] != "Usar sustitución"),
            "sqrt-chain reciprocal trig product should not use the generic substitution substep for {input}: {substeps:?}"
        );
        assert_antiderivative_verifies(input);
    }
}

#[test]
fn integrate_contract_sqrt_chain_substitutions_expose_didactic_substep() {
    for (input, expected_result, expected_required_display, expected_substep_title) in [
        (
            "integrate(sin(sqrt(x))/(sqrt(x)*cos(sqrt(x))^2), x)",
            "2·sec(sqrt(x))",
            serde_json::json!(["cos(sqrt(x)) ≠ 0", "x > 0"]),
            "Usar la regla de sec(u)·tan(u) -> sec(u)",
        ),
        (
            "integrate(tan(sqrt(x))/sqrt(x), x)",
            "-2·ln(|cos(sqrt(x))|)",
            serde_json::json!(["cos(sqrt(x)) ≠ 0", "x > 0"]),
            "Usar la regla de tan(u) -> -ln|cos(u)|",
        ),
        (
            "integrate(tanh(sqrt(x))/sqrt(x), x)",
            "2·ln(cosh(sqrt(x)))",
            serde_json::json!(["x > 0"]),
            "Usar sustitución",
        ),
        (
            "integrate(1/(sqrt(x)*cosh(sqrt(x))^2), x)",
            "2·tanh(sqrt(x))",
            serde_json::json!(["x > 0"]),
            "Usar la regla de 1/cosh(u)^2 -> tanh(u)",
        ),
        (
            "integrate(sinh(sqrt(x))/(sqrt(x)*cosh(sqrt(x))^2), x)",
            "-2 / cosh(sqrt(x))",
            serde_json::json!(["x > 0"]),
            "Usar la regla de sinh(u)/cosh(u)^2 -> -1/cosh(u)",
        ),
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

        assert_eq!(wire["result"], expected_result, "input: {input}");
        assert_eq!(
            wire["required_display"], expected_required_display,
            "unexpected required_display for {input}: {:?}",
            wire["required_display"]
        );
        assert!(
            !stderr.contains("depth_overflow"),
            "sqrt-chain substitution trace should not emit depth_overflow warning for {input}\nstderr:\n{stderr}"
        );

        let steps = wire["steps"]
            .as_array()
            .expect("steps should be present with --steps on");
        assert_eq!(
            steps.len(),
            1,
            "expected compact direct substitution trace for {input}, got {steps:?}"
        );
        let integration_step = steps
            .iter()
            .find(|step| step["rule"] == "Calcular la integral")
            .expect("expected public symbolic integration step");
        let substeps = integration_step["substeps"]
            .as_array()
            .expect("integration step should expose didactic substeps");
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == expected_substep_title),
            "expected {expected_substep_title} substep for {input}, got {substeps:?}"
        );
        assert_antiderivative_verifies(input);
    }
}

#[test]
fn integrate_contract_negative_scaled_cosecant_result_latex_keeps_sign_on_coefficient() {
    let input = "integrate(-2*cos(sqrt(3-2*x))/(sqrt(3-2*x)*sin(sqrt(3-2*x))^2), x)";
    let (wire, stderr) = cli_eval_json_with_stderr(input);
    assert!(
        stderr.is_empty(),
        "unexpected stderr for negative scaled cosecant integral: {stderr}"
    );
    assert_eq!(wire["result"], "-2·csc(sqrt(3 - 2·x))");
    assert_eq!(
        wire["result_latex"],
        "-2\\cdot \\csc(\\sqrt{3 - 2\\cdot x})"
    );
    assert_ne!(
        wire["result_latex"], "2\\cdot -\\csc(\\sqrt{3 - 2\\cdot x})",
        "negative sign should not remain inside the multiplicative factor"
    );
}

#[test]
fn integrate_contract_negative_scaled_raw_reciprocal_trig_primitives_keep_sign_compact() {
    let cases = [
        (
            "integrate(-4*x*sin(x^2)/cos(x^2)^2, x)",
            "-2·sec(x^2)",
            "-2\\cdot \\sec({x}^{2})",
            "cos(x^2) ≠ 0",
            "diff(integrate(-4*x*sin(x^2)/cos(x^2)^2, x), x) + 4*x*sin(x^2)/cos(x^2)^2",
        ),
        (
            "integrate(-4*x*cos(x^2)/sin(x^2)^2, x)",
            "2·csc(x^2)",
            "2\\cdot \\csc({x}^{2})",
            "sin(x^2) ≠ 0",
            "diff(integrate(-4*x*cos(x^2)/sin(x^2)^2, x), x) + 4*x*cos(x^2)/sin(x^2)^2",
        ),
    ];

    for (input, expected_result, expected_latex, expected_condition, residual) in cases {
        let (wire, stderr) = cli_eval_json_with_stderr(input);
        assert!(
            stderr.is_empty(),
            "unexpected stderr for negative scaled raw reciprocal trig primitive: {stderr}"
        );
        assert_eq!(wire["result"], expected_result);
        assert_eq!(wire["result_latex"], expected_latex);
        assert_eq!(
            wire["required_display"],
            serde_json::json!([expected_condition])
        );

        let (residual_wire, residual_stderr) = cli_eval_json_with_stderr(residual);
        assert!(
            residual_stderr.is_empty(),
            "unexpected stderr for negative scaled raw reciprocal trig residual: {residual_stderr}"
        );
        assert_eq!(residual_wire["result"], "0");
        assert_eq!(
            residual_wire["required_display"],
            serde_json::json!([expected_condition])
        );
    }
}

#[test]
fn integrate_contract_sqrt_chain_secant_cosecant_products_integrate_directly() {
    let cases = [
        "integrate(sec(sqrt(x))*tan(sqrt(x))/(2*sqrt(x)), x)",
        "integrate(csc(sqrt(x))*cot(sqrt(x))/(2*sqrt(x)), x)",
    ];

    for input in cases {
        let step_rules = evaluated_integral_step_rules(input);
        assert_eq!(
            step_rules,
            vec!["Symbolic Integration".to_string()],
            "sqrt-chain reciprocal trig products should integrate directly: {step_rules:?}"
        );
        assert_antiderivative_verifies(input);
    }
}

#[test]
fn integrate_contract_sqrt_chain_tangent_cotangent_logs_integrate_directly() {
    let cases = [
        "integrate(tan(sqrt(x))/(2*sqrt(x)), x)",
        "integrate(cot(sqrt(x))/(2*sqrt(x)), x)",
        "integrate(tan(sqrt(3*x+1))*3/(2*sqrt(3*x+1)), x)",
        "integrate(cot(sqrt(3*x+1))*3/(2*sqrt(3*x+1)), x)",
        "integrate(-tan(sqrt(3-2*x))/sqrt(3-2*x), x)",
        "integrate(-cot(sqrt(3-2*x))/sqrt(3-2*x), x)",
    ];

    for input in cases {
        let step_rules = evaluated_integral_step_rules(input);
        assert_eq!(
            step_rules,
            vec!["Symbolic Integration".to_string()],
            "sqrt-chain trig log derivatives should integrate directly: {step_rules:?}"
        );
        assert_antiderivative_verifies(input);
    }
}

#[test]
fn integrate_contract_sqrt_chain_tangent_cotangent_logs_explain_u_and_du() {
    for (
        input,
        expected_result,
        expected_required_display,
        expected_rule_title,
        expects_adjustment,
    ) in [
        (
            "integrate(tan(sqrt(x))/(2*sqrt(x)), x)",
            "-ln(|cos(sqrt(x))|)",
            serde_json::json!(["cos(sqrt(x)) ≠ 0", "x > 0"]),
            "Usar la regla de tan(u) -> -ln|cos(u)|",
            false,
        ),
        (
            "integrate(cot(sqrt(x))/(2*sqrt(x)), x)",
            "ln(|sin(sqrt(x))|)",
            serde_json::json!(["sin(sqrt(x)) ≠ 0", "x > 0"]),
            "Usar la regla de cot(u) -> ln|sin(u)|",
            false,
        ),
        (
            "integrate(tan(sqrt(x))/sqrt(x), x)",
            "-2·ln(|cos(sqrt(x))|)",
            serde_json::json!(["cos(sqrt(x)) ≠ 0", "x > 0"]),
            "Usar la regla de tan(u) -> -ln|cos(u)|",
            true,
        ),
        (
            "integrate(cot(sqrt(x))/sqrt(x), x)",
            "2·ln(|sin(sqrt(x))|)",
            serde_json::json!(["sin(sqrt(x)) ≠ 0", "x > 0"]),
            "Usar la regla de cot(u) -> ln|sin(u)|",
            true,
        ),
        (
            "integrate(tan(sqrt(3*x+1))*3/(2*sqrt(3*x+1)), x)",
            "-ln(|cos(sqrt(3·x + 1))|)",
            serde_json::json!(["cos(sqrt(3·x + 1)) ≠ 0", "x > -1/3"]),
            "Usar la regla de tan(u) -> -ln|cos(u)|",
            false,
        ),
        (
            "integrate(-cot(sqrt(3-2*x))/sqrt(3-2*x), x)",
            "ln(|sin(sqrt(3 - 2·x))|)",
            serde_json::json!(["sin(sqrt(3 - 2·x)) ≠ 0", "x < 3/2"]),
            "Usar la regla de cot(u) -> ln|sin(u)|",
            false,
        ),
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

        assert_eq!(wire["result"], expected_result, "input: {input}");
        assert_eq!(
            wire["required_display"], expected_required_display,
            "unexpected required_display for {input}: {:?}",
            wire["required_display"]
        );
        assert!(
            !stderr.contains("depth_overflow"),
            "sqrt-chain trig log trace should not emit depth_overflow warning for {input}\nstderr:\n{stderr}"
        );

        let steps = wire["steps"]
            .as_array()
            .expect("steps should be present with --steps on");
        let integration_step = steps
            .iter()
            .find(|step| step["rule"] == "Calcular la integral")
            .expect("expected public symbolic integration step");
        let substeps = integration_step["substeps"]
            .as_array()
            .expect("integration step should expose didactic substeps");
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == expected_rule_title),
            "expected {expected_rule_title} substep for {input}, got {substeps:?}"
        );
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == "Identificar u y du"),
            "expected concrete u/du substep for {input}, got {substeps:?}"
        );
        assert_u_du_substep_labels(substeps, input);
        assert_eq!(
            substeps
                .iter()
                .any(|substep| substep["title"] == "Ajustar el factor constante"),
            expects_adjustment,
            "unexpected constant adjustment substep presence for {input}: {substeps:?}"
        );
        assert!(
            substeps
                .iter()
                .all(|substep| substep["title"] != "Usar sustitución"),
            "sqrt-chain trig log table should not use the generic substitution substep for {input}: {substeps:?}"
        );
        assert_antiderivative_verifies(input);
    }
}

#[test]
fn integrate_contract_sqrt_chain_tangent_cotangent_logs_verify() {
    let cases = [
        (
            "integrate(tan(sqrt(x))/(2*sqrt(x)), x)",
            "-ln(|cos(sqrt(x))|)",
            "tan(sqrt(x)) / (2 * sqrt(x))",
            vec!["cos(sqrt(x)) ≠ 0", "x > 0"],
            vec!["cos(sqrt(x)) ≠ 0", "x > 0"],
        ),
        (
            "integrate(cot(sqrt(x))/(2*sqrt(x)), x)",
            "ln(|sin(sqrt(x))|)",
            "cot(sqrt(x)) / (2 * sqrt(x))",
            vec!["sin(sqrt(x)) ≠ 0", "x > 0"],
            vec!["sin(sqrt(x)) ≠ 0", "x > 0"],
        ),
        (
            "integrate(-tan(sqrt(x))/(2*sqrt(x)), x)",
            "ln(|cos(sqrt(x))|)",
            "-tan(sqrt(x)) / (2 * sqrt(x))",
            vec!["cos(sqrt(x)) ≠ 0", "x > 0"],
            vec!["cos(sqrt(x)) ≠ 0", "x > 0"],
        ),
        (
            "integrate(-cot(sqrt(x))/(2*sqrt(x)), x)",
            "-ln(|sin(sqrt(x))|)",
            "-cot(sqrt(x)) / (2 * sqrt(x))",
            vec!["sin(sqrt(x)) ≠ 0", "x > 0"],
            vec!["sin(sqrt(x)) ≠ 0", "x > 0"],
        ),
        (
            "integrate(tan(sqrt(2*x))/sqrt(2*x), x)",
            "-ln(|cos(sqrt(2 * x))|)",
            "tan(sqrt(2 * x)) / sqrt(2 * x)",
            vec!["cos(sqrt(2 * x)) ≠ 0", "x > 0"],
            vec!["cos(sqrt(2 * x)) ≠ 0", "x > 0"],
        ),
        (
            "integrate(-tan(sqrt(2*x))/sqrt(2*x), x)",
            "ln(|cos(sqrt(2 * x))|)",
            "-tan(sqrt(2 * x)) / sqrt(2 * x)",
            vec!["cos(sqrt(2 * x)) ≠ 0", "x > 0"],
            vec!["cos(sqrt(2 * x)) ≠ 0", "x > 0"],
        ),
        (
            "integrate(-cot(sqrt(2*x))/sqrt(2*x), x)",
            "-ln(|sin(sqrt(2 * x))|)",
            "-cot(sqrt(2 * x)) / sqrt(2 * x)",
            vec!["sin(sqrt(2 * x)) ≠ 0", "x > 0"],
            vec!["sin(sqrt(2 * x)) ≠ 0", "x > 0"],
        ),
        (
            "integrate(tan(sqrt(3*x+1))*3/(2*sqrt(3*x+1)), x)",
            "-ln(|cos(sqrt(3 * x + 1))|)",
            "3 * tan(sqrt(3 * x + 1)) / (2 * sqrt(3 * x + 1))",
            vec!["cos(sqrt(3 * x + 1)) ≠ 0", "x > -1/3"],
            vec!["cos(sqrt(3 * x + 1)) ≠ 0", "x > -1/3"],
        ),
        (
            "integrate(cot(sqrt(3*x+1))*3/(2*sqrt(3*x+1)), x)",
            "ln(|sin(sqrt(3 * x + 1))|)",
            "3 * cot(sqrt(3 * x + 1)) / (2 * sqrt(3 * x + 1))",
            vec!["sin(sqrt(3 * x + 1)) ≠ 0", "x > -1/3"],
            vec!["sin(sqrt(3 * x + 1)) ≠ 0", "x > -1/3"],
        ),
        (
            "integrate(-tan(sqrt(3*x+1))*3/(2*sqrt(3*x+1)), x)",
            "ln(|cos(sqrt(3 * x + 1))|)",
            "-3 * tan(sqrt(3 * x + 1)) / (2 * sqrt(3 * x + 1))",
            vec!["cos(sqrt(3 * x + 1)) ≠ 0", "x > -1/3"],
            vec!["cos(sqrt(3 * x + 1)) ≠ 0", "x > -1/3"],
        ),
        (
            "integrate(-tan(sqrt(3-2*x))/sqrt(3-2*x), x)",
            "-ln(|cos(sqrt(3 - 2 * x))|)",
            "-tan(sqrt(3 - 2 * x)) / sqrt(3 - 2 * x)",
            vec!["cos(sqrt(3 - 2 * x)) ≠ 0", "x < 3/2"],
            vec!["cos(sqrt(3 - 2 * x)) ≠ 0", "x < 3/2"],
        ),
        (
            "integrate(-cot(sqrt(3-2*x))/sqrt(3-2*x), x)",
            "ln(|sin(sqrt(3 - 2 * x))|)",
            "-cot(sqrt(3 - 2 * x)) / sqrt(3 - 2 * x)",
            vec!["sin(sqrt(3 - 2 * x)) ≠ 0", "x < 3/2"],
            vec!["sin(sqrt(3 - 2 * x)) ≠ 0", "x < 3/2"],
        ),
        (
            "integrate(-cot(sqrt(3*x+1))*3/(2*sqrt(3*x+1)), x)",
            "-ln(|sin(sqrt(3 * x + 1))|)",
            "-3 * cot(sqrt(3 * x + 1)) / (2 * sqrt(3 * x + 1))",
            vec!["sin(sqrt(3 * x + 1)) ≠ 0", "x > -1/3"],
            vec!["sin(sqrt(3 * x + 1)) ≠ 0", "x > -1/3"],
        ),
    ];

    for (
        input,
        expected_result,
        expected_nested_result,
        expected_conditions,
        expected_nested_conditions,
    ) in cases
    {
        let (result, required) = evaluated_integral_with_required_conditions(input);

        assert_eq!(result, expected_result, "unexpected result for {input}");
        assert_eq!(
            required, expected_conditions,
            "unexpected required_conditions for {input}: {required:?}"
        );
        assert_antiderivative_verifies(input);

        let nested_input = format!("diff({input}, x)");
        let (nested_result, nested_required) =
            evaluated_expr_with_required_conditions(&nested_input);
        assert_eq!(
            nested_result, expected_nested_result,
            "unexpected nested diff/integrate result for {input}"
        );
        assert_eq!(
            nested_required, expected_nested_conditions,
            "unexpected nested required_conditions for {input}: {nested_required:?}"
        );
    }
}

#[test]
fn integrate_contract_sqrt_chain_reciprocal_trig_logs_verify_by_diff() {
    let cases = [
        (
            "integrate(3/(2*sqrt(3*x+1)*cos(sqrt(3*x+1))), x)",
            "ln(|tan(sqrt(3 * x + 1)) + sec(sqrt(3 * x + 1))|)",
            vec!["cos(sqrt(3 * x + 1)) ≠ 0", "x > -1/3"],
        ),
        (
            "integrate(3/(2*sqrt(3*x+1)*sin(sqrt(3*x+1))), x)",
            "ln(|csc(sqrt(3 * x + 1)) - cot(sqrt(3 * x + 1))|)",
            vec!["sin(sqrt(3 * x + 1)) ≠ 0", "x > -1/3"],
        ),
    ];

    for (input, expected_result, expected_conditions) in cases {
        let (result, required) = evaluated_integral_with_required_conditions(input);
        assert_eq!(result, expected_result, "input: {input}");
        assert_eq!(required, expected_conditions, "input: {input}");
        assert_antiderivative_verifies(input);
    }
}

#[test]
fn integrate_contract_sqrt_chain_reciprocal_trig_logs_explain_u_and_du() {
    for (input, expected_result, expected_required_display, expected_rule_title) in [
        (
            "integrate(3/(2*sqrt(3*x+1)*cos(sqrt(3*x+1))), x)",
            "ln(|tan(sqrt(3·x + 1)) + sec(sqrt(3·x + 1))|)",
            serde_json::json!(["cos(sqrt(3·x + 1)) ≠ 0", "x > -1/3"]),
            "Usar la regla de sec(u) -> ln|sec(u)+tan(u)|",
        ),
        (
            "integrate(3/(2*sqrt(3*x+1)*sin(sqrt(3*x+1))), x)",
            "ln(|csc(sqrt(3·x + 1)) - cot(sqrt(3·x + 1))|)",
            serde_json::json!(["sin(sqrt(3·x + 1)) ≠ 0", "x > -1/3"]),
            "Usar la regla de csc(u) -> ln|csc(u)-cot(u)|",
        ),
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

        assert_eq!(wire["result"], expected_result, "input: {input}");
        assert_eq!(
            wire["required_display"], expected_required_display,
            "unexpected required_display for {input}: {:?}",
            wire["required_display"]
        );
        assert!(
            !stderr.contains("depth_overflow"),
            "sqrt-chain reciprocal trig log trace should not emit depth_overflow warning for {input}\nstderr:\n{stderr}"
        );

        let steps = wire["steps"]
            .as_array()
            .expect("steps should be present with --steps on");
        let integration_step = steps
            .iter()
            .find(|step| step["rule"] == "Calcular la integral")
            .expect("expected public symbolic integration step");
        let substeps = integration_step["substeps"]
            .as_array()
            .expect("integration step should expose didactic substeps");
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == expected_rule_title),
            "expected {expected_rule_title} substep for {input}, got {substeps:?}"
        );
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == "Identificar u y du"),
            "expected concrete u/du substep for {input}, got {substeps:?}"
        );
        assert!(
            substeps
                .iter()
                .all(|substep| substep["title"] != "Usar sustitución"),
            "sqrt-chain reciprocal trig log table should not use the generic substitution substep for {input}: {substeps:?}"
        );
        assert_antiderivative_verifies(input);
    }
}

#[test]
fn integrate_contract_sqrt_chain_hyperbolic_tangent_logs_verify() {
    let cases = [
        (
            "integrate(tanh(sqrt(x))/(2*sqrt(x)), x)",
            "ln(cosh(sqrt(x)))",
            "tanh(sqrt(x)) / (2 * sqrt(x))",
            vec!["x > 0"],
        ),
        (
            "integrate(1/(2*sqrt(x)*tanh(sqrt(x))), x)",
            "ln(|sinh(sqrt(x))|)",
            "1 / (2 * tanh(sqrt(x)) * sqrt(x))",
            vec!["sinh(sqrt(x)) ≠ 0", "x > 0"],
        ),
        (
            "integrate(tanh(sqrt(2*x))/sqrt(2*x), x)",
            "ln(cosh(sqrt(2 * x)))",
            "tanh(sqrt(2 * x)) / sqrt(2 * x)",
            vec!["x > 0"],
        ),
        (
            "integrate(tanh(sqrt(3*x+1))*3/(2*sqrt(3*x+1)), x)",
            "ln(cosh(sqrt(3 * x + 1)))",
            "3 * tanh(sqrt(3 * x + 1)) / (2 * sqrt(3 * x + 1))",
            vec!["x > -1/3"],
        ),
        (
            "integrate(3/(2*sqrt(3*x+1)*tanh(sqrt(3*x+1))), x)",
            "ln(|sinh(sqrt(3 * x + 1))|)",
            "3 / (2 * tanh(sqrt(3 * x + 1)) * sqrt(3 * x + 1))",
            vec!["sinh(sqrt(3 * x + 1)) ≠ 0", "x > -1/3"],
        ),
    ];

    for (input, expected_result, expected_nested_result, expected_conditions) in cases {
        let (result, required) = evaluated_integral_with_required_conditions(input);

        assert_eq!(result, expected_result, "unexpected result for {input}");
        assert_eq!(
            required, expected_conditions,
            "unexpected required_conditions for {input}: {required:?}"
        );
        assert_antiderivative_verifies(input);
        assert_rendered_antiderivative_verifies(input, &result);

        let nested_input = format!("diff({input}, x)");
        let (nested_result, nested_required) =
            evaluated_expr_with_required_conditions(&nested_input);
        assert_eq!(
            nested_result, expected_nested_result,
            "unexpected nested diff/integrate result for {input}"
        );
        assert_eq!(
            nested_required, expected_conditions,
            "unexpected nested required_conditions for {input}: {nested_required:?}"
        );
    }
}

#[test]
fn integrate_contract_sqrt_chain_hyperbolic_tangent_logs_explain_u_and_du() {
    for (input, expected_result, expected_required_display, expected_rule_title) in [
        (
            "integrate(tanh(sqrt(3*x+1))*3/(2*sqrt(3*x+1)), x)",
            "ln(cosh(sqrt(3·x + 1)))",
            serde_json::json!(["x > -1/3"]),
            "Usar la regla de tanh(u) -> ln(cosh(u))",
        ),
        (
            "integrate(3/(2*sqrt(3*x+1)*tanh(sqrt(3*x+1))), x)",
            "ln(|sinh(sqrt(3·x + 1))|)",
            serde_json::json!(["sinh(sqrt(3·x + 1)) ≠ 0", "x > -1/3"]),
            "Usar la regla de 1/tanh(u) -> ln|sinh(u)|",
        ),
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

        assert_eq!(wire["result"], expected_result, "input: {input}");
        assert_eq!(
            wire["required_display"], expected_required_display,
            "unexpected required_display for {input}: {:?}",
            wire["required_display"]
        );
        assert!(
            !stderr.contains("depth_overflow"),
            "sqrt-chain hyperbolic log trace should not emit depth_overflow warning for {input}\nstderr:\n{stderr}"
        );

        let steps = wire["steps"]
            .as_array()
            .expect("steps should be present with --steps on");
        let integration_step = steps
            .iter()
            .find(|step| step["rule"] == "Calcular la integral")
            .expect("expected public symbolic integration step");
        let substeps = integration_step["substeps"]
            .as_array()
            .expect("integration step should expose didactic substeps");
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == expected_rule_title),
            "expected {expected_rule_title} substep for {input}, got {substeps:?}"
        );
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == "Identificar u y du"),
            "expected concrete u/du substep for {input}, got {substeps:?}"
        );
        assert!(
            substeps
                .iter()
                .all(|substep| substep["title"] != "Usar sustitución"),
            "sqrt-chain hyperbolic log table should not use only the generic substitution substep for {input}: {substeps:?}"
        );
        assert_antiderivative_verifies(input);
    }
}

#[test]
fn integrate_contract_sqrt_chain_hyperbolic_reciprocal_squares_verify() {
    let cases = [
        (
            "integrate(1/(2*sqrt(x)*cosh(sqrt(x))^2), x)",
            "tanh(sqrt(x))",
            vec!["x > 0"],
        ),
        (
            "integrate(1/(2*sqrt(x)*sinh(sqrt(x))^2), x)",
            "-1 / tanh(sqrt(x))",
            vec!["x > 0", "sinh(sqrt(x)) ≠ 0"],
        ),
        (
            "integrate((2*x)^(-1/2)/cosh(sqrt(2*x))^2, x)",
            "tanh(sqrt(2 * x))",
            vec!["x > 0"],
        ),
        (
            "integrate(3/(2*sqrt(3*x+1)*cosh(sqrt(3*x+1))^2), x)",
            "tanh(sqrt(3 * x + 1))",
            vec!["x > -1/3"],
        ),
        (
            "integrate(3/(2*sqrt(3*x+1)*sinh(sqrt(3*x+1))^2), x)",
            "-1 / tanh(sqrt(3 * x + 1))",
            vec!["x > -1/3", "sinh(sqrt(3 * x + 1)) ≠ 0"],
        ),
        (
            "integrate(k/(2*sqrt(x)*cosh(sqrt(x)-b)^2), x)",
            "tanh(sqrt(x) - b) * k",
            vec!["x > 0"],
        ),
        (
            "integrate(k/(2*sqrt(x)*sinh(sqrt(x)-b)^2), x)",
            "-k / tanh(sqrt(x) - b)",
            vec!["x > 0", "sinh(sqrt(x) - b) ≠ 0"],
        ),
    ];

    for (input, expected_result, expected_conditions) in cases {
        let (result, required) = evaluated_integral_with_required_conditions(input);

        assert_eq!(result, expected_result, "unexpected result for {input}");
        assert_eq!(
            required, expected_conditions,
            "unexpected required_conditions for {input}: {required:?}"
        );
        assert_antiderivative_verifies(input);
    }

    let nested_cases = [
        (
            "diff(integrate(3/(2*sqrt(3*x+1)*cosh(sqrt(3*x+1))^2), x), x)",
            "3 / (2 * sqrt(3 * x + 1) * cosh(sqrt(3 * x + 1))^2)",
            vec!["x > -1/3"],
        ),
        (
            "diff(integrate(3/(2*sqrt(3*x+1)*sinh(sqrt(3*x+1))^2), x), x)",
            "3 / (2 * sqrt(3 * x + 1) * sinh(sqrt(3 * x + 1))^2)",
            vec!["x > -1/3", "sinh(sqrt(3 * x + 1)) ≠ 0"],
        ),
    ];

    for (input, expected_result, expected_conditions) in nested_cases {
        let (result, required) = evaluated_expr_with_required_conditions(input);

        assert_eq!(
            result, expected_result,
            "unexpected nested result for {input}"
        );
        assert!(
            !result.contains("^(-1/2)") && !result.contains("^(1/2)"),
            "nested diff/integrate presentation should keep explicit sqrt forms, got: {result}"
        );
        assert_eq!(
            required, expected_conditions,
            "unexpected nested required_conditions for {input}: {required:?}"
        );
    }

    let step_summaries = evaluated_expr_step_summaries(
        "diff(integrate(3/(2*sqrt(3*x+1)*cosh(sqrt(3*x+1))^2), x), x)",
    );
    assert!(
        step_summaries
            .iter()
            .any(|(description, rule_name, importance)| description
                == "Post-calculus presentation"
                && rule_name == "Present calculus result in compact form"
                && *importance >= ImportanceLevel::Medium),
        "post-calculus presentation should be visible in normal step output: {step_summaries:?}"
    );
    assert!(
        !step_summaries.iter().any(|(_, rule_name, _)| {
            rule_name == "Pull Constant From Fraction"
                || rule_name == "Simplify Multiplication with Division"
        }),
        "post-calculus presentation should hide mechanical fraction cleanup before the compact result: {step_summaries:?}"
    );
}

#[test]
fn integrate_contract_sqrt_chain_hyperbolic_reciprocal_derivatives_verify() {
    let cases = [
        (
            "integrate(sinh(sqrt(x))/(2*sqrt(x)*cosh(sqrt(x))^2), x)",
            "-1 / cosh(sqrt(x))",
            vec!["x > 0"],
        ),
        (
            "integrate(cosh(sqrt(x))/(2*sqrt(x)*sinh(sqrt(x))^2), x)",
            "-1 / sinh(sqrt(x))",
            vec!["x > 0", "sinh(sqrt(x)) ≠ 0"],
        ),
        (
            "integrate(sinh(sqrt(2*x))/(sqrt(2*x)*cosh(sqrt(2*x))^2), x)",
            "-1 / cosh(sqrt(2 * x))",
            vec!["x > 0"],
        ),
        (
            "integrate(3*sinh(sqrt(3*x+1))/(2*sqrt(3*x+1)*cosh(sqrt(3*x+1))^2), x)",
            "-1 / cosh(sqrt(3 * x + 1))",
            vec!["x > -1/3"],
        ),
        (
            "integrate(3*cosh(sqrt(3*x+1))/(2*sqrt(3*x+1)*sinh(sqrt(3*x+1))^2), x)",
            "-1 / sinh(sqrt(3 * x + 1))",
            vec!["x > -1/3", "sinh(sqrt(3 * x + 1)) ≠ 0"],
        ),
        (
            "integrate(k*sinh(sqrt(x)-b)/(2*sqrt(x)*cosh(sqrt(x)-b)^2), x)",
            "-k / cosh(sqrt(x) - b)",
            vec!["x > 0"],
        ),
        (
            "integrate(k*cosh(sqrt(x)-b)/(2*sqrt(x)*sinh(sqrt(x)-b)^2), x)",
            "-k / sinh(sqrt(x) - b)",
            vec!["x > 0", "sinh(sqrt(x) - b) ≠ 0"],
        ),
    ];

    for (input, expected_result, expected_conditions) in cases {
        let (result, required) = evaluated_integral_with_required_conditions(input);

        assert_eq!(result, expected_result, "unexpected result for {input}");
        assert_eq!(
            required, expected_conditions,
            "unexpected required_conditions for {input}: {required:?}"
        );
        assert_antiderivative_verifies(input);
    }

    let nested_cases = [
        (
            "diff(integrate(3*sinh(sqrt(3*x+1))/(2*sqrt(3*x+1)*cosh(sqrt(3*x+1))^2), x), x)",
            "3 * sinh(sqrt(3 * x + 1)) / (2 * sqrt(3 * x + 1) * cosh(sqrt(3 * x + 1))^2)",
            vec!["x > -1/3"],
        ),
        (
            "diff(integrate(3*cosh(sqrt(3*x+1))/(2*sqrt(3*x+1)*sinh(sqrt(3*x+1))^2), x), x)",
            "3 * cosh(sqrt(3 * x + 1)) / (2 * sqrt(3 * x + 1) * sinh(sqrt(3 * x + 1))^2)",
            vec!["x > -1/3", "sinh(sqrt(3 * x + 1)) ≠ 0"],
        ),
    ];

    for (input, expected_result, expected_conditions) in nested_cases {
        let (result, required) = evaluated_expr_with_required_conditions(input);

        assert_eq!(
            result, expected_result,
            "unexpected nested result for {input}"
        );
        assert!(
            !result.contains("^(-1/2)") && !result.contains("^(1/2)"),
            "nested diff/integrate presentation should keep explicit sqrt forms, got: {result}"
        );
        assert_eq!(
            required, expected_conditions,
            "unexpected nested required_conditions for {input}: {required:?}"
        );
    }
}

#[test]
fn integrate_contract_sqrt_chain_hyperbolic_reciprocal_tables_explain_u_and_du() {
    for (input, expected_result, expected_required_display, expected_rule_title) in [
        (
            "integrate(3/(2*sqrt(3*x+1)*cosh(sqrt(3*x+1))^2), x)",
            "tanh(sqrt(3·x + 1))",
            serde_json::json!(["x > -1/3"]),
            "Usar la regla de 1/cosh(u)^2 -> tanh(u)",
        ),
        (
            "integrate(3/(2*sqrt(3*x+1)*sinh(sqrt(3*x+1))^2), x)",
            "-1 / tanh(sqrt(3·x + 1))",
            serde_json::json!(["x > -1/3", "sinh(sqrt(3·x + 1)) ≠ 0"]),
            "Usar la regla de 1/sinh(u)^2 -> -1/tanh(u)",
        ),
        (
            "integrate(3*sinh(sqrt(3*x+1))/(2*sqrt(3*x+1)*cosh(sqrt(3*x+1))^2), x)",
            "-1 / cosh(sqrt(3·x + 1))",
            serde_json::json!(["x > -1/3"]),
            "Usar la regla de sinh(u)/cosh(u)^2 -> -1/cosh(u)",
        ),
        (
            "integrate(3*cosh(sqrt(3*x+1))/(2*sqrt(3*x+1)*sinh(sqrt(3*x+1))^2), x)",
            "-1 / sinh(sqrt(3·x + 1))",
            serde_json::json!(["x > -1/3", "sinh(sqrt(3·x + 1)) ≠ 0"]),
            "Usar la regla de cosh(u)/sinh(u)^2 -> -1/sinh(u)",
        ),
        (
            "integrate(k/(2*sqrt(x)*sinh(sqrt(x)-b)^2), x)",
            "-k / tanh(sqrt(x) - b)",
            serde_json::json!(["x > 0", "sinh(sqrt(x) - b) ≠ 0"]),
            "Usar la regla de 1/sinh(u)^2 -> -1/tanh(u)",
        ),
        (
            "integrate(k*sinh(sqrt(x)-b)/(2*sqrt(x)*cosh(sqrt(x)-b)^2), x)",
            "-k / cosh(sqrt(x) - b)",
            serde_json::json!(["x > 0"]),
            "Usar la regla de sinh(u)/cosh(u)^2 -> -1/cosh(u)",
        ),
        (
            "integrate(k*cosh(sqrt(x)-b)/(2*sqrt(x)*sinh(sqrt(x)-b)^2), x)",
            "-k / sinh(sqrt(x) - b)",
            serde_json::json!(["x > 0", "sinh(sqrt(x) - b) ≠ 0"]),
            "Usar la regla de cosh(u)/sinh(u)^2 -> -1/sinh(u)",
        ),
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

        assert_eq!(wire["result"], expected_result, "input: {input}");
        assert_eq!(
            wire["required_display"], expected_required_display,
            "unexpected required_display for {input}: {:?}",
            wire["required_display"]
        );
        assert!(
            !stderr.contains("depth_overflow"),
            "sqrt-chain hyperbolic reciprocal trace should not emit depth_overflow warning for {input}\nstderr:\n{stderr}"
        );

        let steps = wire["steps"]
            .as_array()
            .expect("steps should be present with --steps on");
        let integration_step = steps
            .iter()
            .find(|step| step["rule"] == "Calcular la integral")
            .expect("expected public symbolic integration step");
        let substeps = integration_step["substeps"]
            .as_array()
            .expect("integration step should expose didactic substeps");
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == expected_rule_title),
            "expected {expected_rule_title} substep for {input}, got {substeps:?}"
        );
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == "Identificar u y du"),
            "expected concrete u/du substep for {input}, got {substeps:?}"
        );
        assert!(
            substeps
                .iter()
                .all(|substep| substep["title"] != "Usar sustitución"),
            "sqrt-chain hyperbolic reciprocal table should not use only the generic substitution substep for {input}: {substeps:?}"
        );
        assert_antiderivative_verifies(input);
    }
}

#[test]
fn integrate_contract_negated_polynomial_secant_tangent_product_preserves_nonzero_domain() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(-x*sec(x^2)*tan(x^2), x)");

    assert_eq!(result, "-sec(x^2) / 2");
    assert_eq!(
        required,
        vec!["cos(x^2) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}

#[test]
fn integrate_contract_negated_polynomial_cosecant_cotangent_product_preserves_nonzero_domain() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(-x^2*csc(x^3)*cot(x^3), x)");

    assert_eq!(result, "csc(x^3) / 3");
    assert_eq!(
        required,
        vec!["sin(x^3) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}

#[test]
fn integrate_contract_secant_tangent_non_linear_argument_remains_residual() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(sec(x^2)*tan(x^2), x)");

    assert_eq!(result, "integrate(sin(x^2) / cos(x^2)^2, x)");
    assert_eq!(
        required,
        vec!["cos(x^2) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}

#[test]
fn integrate_contract_tangent_non_linear_argument_remains_residual_with_pole_condition() {
    let (result, required) = evaluated_integral_with_required_conditions("integrate(tan(x^2), x)");

    assert_eq!(result, "integrate(tan(x^2), x)");
    assert_eq!(
        required,
        vec!["cos(x^2) ≠ 0".to_string()],
        "unsupported tangent residual should preserve pole domain: {required:?}"
    );
}

#[test]
fn integrate_contract_presimplified_tangent_residual_preserves_pole_condition() {
    for (input, expected_result) in [
        ("integrate(tan(x^2+0), x)", "integrate(tan(x^2), x)"),
        (
            "integrate(tan(x^2+0)+sin(x^2), x)",
            "integrate(sin(x^2) + tan(x^2), x)",
        ),
    ] {
        let (result, required) = evaluated_integral_with_required_conditions(input);

        assert_eq!(result, expected_result, "unexpected result for {input}");
        assert_eq!(
            required,
            vec!["cos(x^2) ≠ 0".to_string()],
            "pre-simplified tangent residual should preserve pole domain for {input}: {required:?}"
        );
    }
}

#[test]
fn integrate_contract_presimplified_cosecant_residual_preserves_pole_condition() {
    for (input, expected_result) in [
        ("integrate(csc(x^2+0), x)", "integrate(csc(x^2), x)"),
        ("integrate(cot(x^2+0), x)", "integrate(cot(x^2), x)"),
    ] {
        let (result, required) = evaluated_integral_with_required_conditions(input);

        assert_eq!(result, expected_result, "unexpected result for {input}");
        assert_eq!(
            required,
            vec!["sin(x^2) ≠ 0".to_string()],
            "pre-simplified sine-pole residual should preserve domain for {input}: {required:?}"
        );
    }
}

#[test]
fn integrate_contract_unsupported_non_elementary_residual() {
    assert_eq!(
        simplified_integral("integrate(sin(x^2), x)"),
        "integrate(sin(x^2), x)"
    );
}

#[test]
fn integrate_contract_abs_linear_definite_narrates_root_split() {
    // |linear| has no single elementary antiderivative, so the FTC narration
    // produces nothing; the root-split route must be narrated instead.
    // Root strictly inside the interval -> "Partir el intervalo en la raíz".
    let (wire, stderr) =
        cli_eval_json_with_stderr_args("integrate(abs(2*x-1), x, 0, 1)", &["--steps", "on"]);
    assert!(stderr.is_empty(), "no stderr expected: {stderr}");
    assert_eq!(wire["result"].as_str(), Some("1/2"));
    let step_text = wire["steps"].to_string();
    assert!(
        step_text.contains("Localizar la raíz del valor absoluto"),
        "abs-linear trace should locate the root: {step_text}"
    );
    assert!(
        step_text.contains("Partir el intervalo en la raíz"),
        "root inside the interval should narrate the split: {step_text}"
    );

    // Root outside the interval -> constant-sign narration, no split.
    let (wire_outside, _) =
        cli_eval_json_with_stderr_args("integrate(abs(x-1), x, 2, 5)", &["--steps", "on"]);
    assert_eq!(wire_outside["result"].as_str(), Some("15/2"));
    let outside_text = wire_outside["steps"].to_string();
    assert!(
        outside_text.contains("El interior mantiene signo constante"),
        "root outside the interval should narrate constant sign: {outside_text}"
    );
    assert!(
        !outside_text.contains("Partir el intervalo en la raíz"),
        "root outside the interval should not narrate a split: {outside_text}"
    );

    // A plain FTC definite integral keeps its antiderivative narration.
    let (wire_ftc, _) =
        cli_eval_json_with_stderr_args("integrate(x^2, x, 0, 1)", &["--steps", "on"]);
    let ftc_text = wire_ftc["steps"].to_string();
    assert!(
        ftc_text.contains("Hallar la antiderivada"),
        "ordinary definite integrals keep the FTC narration: {ftc_text}"
    );
}

#[test]
fn integrate_contract_divergent_improper_area_function_is_undefined() {
    // int_0^x ln(t)/t dt DIVERGES at t = 0 (antiderivative (ln t)^2/2 -> +inf):
    // it must be `undefined`, not a form carrying a silent infinity^k term that
    // a later diff drops into a false finite derivative.
    assert_eq!(
        simplified_integral("integrate(ln(t)/t, t, 0, x)"),
        "undefined"
    );
    assert_eq!(
        simplified_integral("integrate(ln(t)^2/t, t, 0, x)"),
        "undefined"
    );
    // The hyperbolic/exp ~1/t family (antiderivative ln|...| -> -inf at 0) too:
    // the boundary ln(|sinh(0)|) / ln(|(e^0-1)/...|) folds numerically to ln(0).
    assert_eq!(
        simplified_integral("integrate(coth(t), t, 0, x)"),
        "undefined"
    );
    assert_eq!(
        simplified_integral("integrate(1/(e^t-1), t, 0, x)"),
        "undefined"
    );
    // and the derivative of the divergent area function is undefined, not ln(x)/x.
    assert_eq!(
        simplified_integral("diff(integrate(ln(t)/t, t, 0, x), x)"),
        "undefined"
    );

    // CONVERGENT improper and ordinary area functions are unaffected.
    assert_eq!(
        simplified_integral("integrate(ln(t), t, 0, x)"),
        "x * ln(x) - x"
    );
    assert_eq!(simplified_integral("integrate(t^2, t, 0, x)"), "1/3 * x^3");
    assert_eq!(
        simplified_integral("integrate(1/(1+t^2), t, 0, x)"),
        "arctan(x)"
    );
}

/// `N/(a² − x²)` definite-integrated over an interval strictly OUTSIDE (−a, a):
/// the `atanh` antiderivative is real only inside (−a, a), so the FTC path used
/// to decline; the equal `−N/(x² − a²)` has a real log antiderivative off the
/// poles. Numeric ground truths (Simpson, verified): `∫₂³ 1/(1−x²) = ½ln(2/3)`,
/// `∫₃⁵ 1/(4−x²) = ¼ln(7/15)`, `∫₄⁵ 1/(9−x²) = ⅙ln(4/7)`.
#[test]
fn definite_rational_reciprocal_difference_of_squares_outside_atanh_domain() {
    // Outside (−a, a), pole-free: now evaluates via the log form.
    assert_eq!(
        simplified_integral("integrate(1/(1-x^2), x, 2, 3)"),
        "1/2 * ln(2/3)"
    );
    // Negative interval reflects (even integrand) then evaluates identically.
    assert_eq!(
        simplified_integral("integrate(1/(1-x^2), x, -3, -2)"),
        "1/2 * ln(2/3)"
    );
    assert_eq!(
        simplified_integral("integrate(1/(4-x^2), x, 3, 5)"),
        "1/4 * ln(7/15)"
    );
    assert_eq!(
        simplified_integral("integrate(1/(9-x^2), x, 4, 5)"),
        "1/6 * ln(4/7)"
    );
    // Scaled leading coefficient: 1/(2−2x²) = (1/2)/(1−x²).
    assert_eq!(
        simplified_integral("integrate(1/(2-2*x^2), x, 2, 3)"),
        "1/4 * ln(2/3)"
    );
    // Reversed bounds negate the oriented integral.
    assert_eq!(
        simplified_integral("integrate(1/(1-x^2), x, 3, 2)"),
        "1/2 * ln(3/2)"
    );
}

/// The gate is strict: an interval INSIDE (−a, a) keeps the elegant `atanh`
/// form, a pole crossing stays `undefined`, and non-`a²−x²` shapes are left to
/// their own owners (arctan for `a²+x²`, residual for a linear term).
#[test]
fn definite_rational_atanh_domain_gate_is_strict() {
    // Inside (−a, a): the atanh antiderivative is real, keep it.
    assert_eq!(
        simplified_integral("integrate(1/(1-x^2), x, 0, 1/2)"),
        "atanh(1/2)"
    );
    assert_eq!(
        simplified_integral("integrate(1/(4-x^2), x, 0, 1)"),
        "1/2 * atanh(1/2)"
    );
    // A pole strictly inside the interval: divergent, undefined.
    assert_eq!(
        simplified_integral("integrate(1/(1-x^2), x, 0, 2)"),
        "undefined"
    );
    assert_eq!(
        simplified_integral("integrate(1/(1/4-x^2), x, -2, 1/4)"),
        "undefined"
    );
    // Not the a²−x² family: `a²+x²` is arctan, a linear term declines — the
    // gate must not hijack either.
    assert_eq!(
        simplified_integral("integrate(1/(-1-x^2), x, 2, 3)"),
        "arctan(2) - arctan(3)"
    );
    assert_eq!(
        simplified_integral("integrate(1/(1-x-x^2), x, 2, 3)"),
        "integrate(1 / (1 - x^2 - x), x, 2, 3)"
    );
}
