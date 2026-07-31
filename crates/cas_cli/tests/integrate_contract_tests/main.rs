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

/// Returns the `after_latex` of the "evaluate the antiderivative at the bounds"
/// substep, i.e. the line where the student reads `F(b) - F(a)`.
fn definite_bounds_substep_after_latex(input: &str) -> String {
    let (wire, _) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);
    let steps = wire["steps"]
        .as_array()
        .expect("steps should be present with --steps on");
    let substep = steps
        .iter()
        .filter_map(|step| step["substeps"].as_array())
        .flatten()
        .find(|substep| substep["title"] == "Evaluar la antiderivada en los límites")
        .unwrap_or_else(|| panic!("no bounds-evaluation substep for {input}"));
    substep["after_latex"]
        .as_str()
        .unwrap_or_else(|| panic!("bounds substep without after_latex for {input}"))
        .to_string()
}

mod by_parts;
mod definite_integrals;
mod exponential_log;
mod hyperbolic;
mod inverse_trig;
mod misc;
mod partial_fractions;
mod polynomial;
mod radicals;
mod rational;
mod steps_narration;
mod substitution;
mod trigonometric;
mod verification;
