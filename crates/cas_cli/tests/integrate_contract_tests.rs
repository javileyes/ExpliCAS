use cas_ast::{Context, Expr, ExprId};
use cas_formatter::DisplayExpr;
use cas_parser::parse;
use cas_session::SessionState;
use cas_solver::api::render_conditions_normalized;
use cas_solver::runtime::{Engine, EvalAction, EvalRequest, EvalResult, Simplifier};

fn render_expr(ctx: &Context, id: ExprId) -> String {
    format!("{}", DisplayExpr { context: ctx, id })
}

fn simplified_integral(input: &str) -> String {
    let mut simplifier = Simplifier::with_default_rules();
    simplifier.disable_rule("Double Angle Identity");
    let expr = parse(input, &mut simplifier.context).expect("parse integration input");
    let (result, _) = simplifier.simplify(expr);
    render_expr(&simplifier.context, result)
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

fn assert_antiderivative_verifies(input: &str) {
    let mut simplifier = Simplifier::with_default_rules();
    simplifier.disable_rule("Double Angle Identity");
    let expr = parse(input, &mut simplifier.context).expect("parse integration input");
    let (integrand, var_name) = explicit_integrate_call_parts(&simplifier.context, expr);

    let (antiderivative, _) = simplifier.simplify(expr);
    let var = simplifier.context.var(&var_name);
    let diff_call = simplifier.context.call("diff", vec![antiderivative, var]);
    let (derivative, _) = simplifier.simplify(diff_call);
    let (expected_integrand, _) = simplifier.simplify(integrand);
    let residual = simplifier
        .context
        .add(Expr::Sub(derivative, expected_integrand));
    let (residual, _) = simplifier.simplify(residual);

    assert_eq!(
        render_expr(&simplifier.context, residual),
        "0",
        "antiderivative verification failed for {input}\nintegral result: {}\nderivative: {}\nexpected integrand: {}",
        render_expr(&simplifier.context, antiderivative),
        render_expr(&simplifier.context, derivative),
        render_expr(&simplifier.context, expected_integrand),
    );
}

fn evaluated_integral_with_required_conditions(input: &str) -> (String, Vec<String>) {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    let parsed = parse(input, &mut engine.simplifier.context).expect("parse integration input");

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

#[test]
fn integrate_contract_supported_antiderivatives_verify_by_differentiation() {
    for input in [
        "integrate(2*x + 3, x)",
        "integrate((3*x)^2, x)",
        "integrate(sin(2*x), x)",
        "integrate(cos(x), x)",
        "integrate(exp(3*x + 1), x)",
        "integrate(x*exp(x), x)",
        "integrate((2*x+3)*exp(2*x+1), x)",
        "integrate((x+1)*exp((3*x+2)/2), x)",
        "integrate((x+1)*exp((2-3*x)/2), x)",
        "integrate(x*sin(x), x)",
        "integrate(x*cos(x), x)",
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
        "integrate(2*x*exp(x^2), x)",
        "integrate(2*x*cos(x^2), x)",
        "integrate(2*x*sin(x^2), x)",
        "integrate(sinh(2*x + 1), x)",
        "integrate(cosh(2*x + 1), x)",
        "integrate(tanh(2*x + 1), x)",
        "integrate(2*x*sinh(x^2), x)",
        "integrate(2*x*cosh(x^2), x)",
        "integrate(2*x*tanh(x^2), x)",
        "integrate(sinh(2*x + 1)/cosh(2*x + 1), x)",
        "integrate(cosh(2*x + 1)/sinh(2*x + 1), x)",
        "integrate(2*x/tanh(x^2), x)",
        "integrate(1/cosh(2*x + 1)^2, x)",
        "integrate(2*x/cosh(x^2)^2, x)",
        "integrate(2*x/sinh(x^2)^2, x)",
        "integrate(2*x*sinh(x^2)/cosh(x^2)^2, x)",
        "integrate(-x*sinh(x^2)/cosh(x^2)^2, x)",
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
        "integrate(1/(x^2-1), x)",
        "integrate(2*x/(x^2-1)^2, x)",
        "integrate((2*x+1)/(x^2+x-1)^2, x)",
        "integrate((2*x+1)/(x^2+x-1)^3, x)",
        "integrate((2*x+1)/(x^4+2*x^3-x^2-2*x+1), x)",
        "integrate(2*x/(x^4-4), x)",
        "integrate((2*x + 1)/(x^2 + x - 1), x)",
        "integrate((4*x + 2)/(x^2 + x + 1), x)",
        "integrate(2*x/sqrt(4-x^4), x)",
        "integrate(1/sqrt(4-(x+1)^2), x)",
        "integrate(2*x/sqrt(1+x^4), x)",
        "integrate(2*x/sqrt(4+x^4), x)",
        "integrate(1/sqrt(4+(x+1)^2), x)",
        "integrate(x/sqrt(x^2+1), x)",
        "integrate(2*x/sqrt(x^2-1), x)",
        "integrate(x*sqrt(x^2+1), x)",
        "integrate(2*x*sqrt(x^2-1), x)",
        "integrate(2*x*(x^2+1)^3, x)",
        "integrate(2*x*(x^2-1)^(3/2), x)",
        "integrate((x^2+1)^(-1/2), x)",
        "integrate(sec(x)^2, x)",
        "integrate(csc(x)^2, x)",
        "integrate(sec(2*x + 1)^2, x)",
        "integrate(csc(2*x + 1)^2, x)",
        "integrate(x/(cos(x^2)^2), x)",
        "integrate(x^2/(sin(x^3)^2), x)",
        "integrate(csc(x)*cot(x), x)",
        "integrate(tan(2*x + 1), x)",
        "integrate(cot(2*x + 1), x)",
        "integrate(2*x*tan(x^2), x)",
        "integrate(3*x^2*cot(x^3), x)",
        "integrate(sec(2*x + 1)*tan(2*x + 1), x)",
        "integrate(2*x*sec(x^2)*tan(x^2), x)",
        "integrate(-x*sec(x^2)*tan(x^2), x)",
        "integrate(csc(2*x + 1)*cot(2*x + 1), x)",
        "integrate(3*x^2*csc(x^3)*cot(x^3), x)",
        "integrate(-x^2*csc(x^3)*cot(x^3), x)",
    ] {
        assert_antiderivative_verifies(input);
    }
}

#[test]
fn integrate_contract_linear_sine_substitution() {
    assert_eq!(
        simplified_integral("integrate(sin(2*x), x)"),
        "-1/2 * cos(2 * x)"
    );
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
    assert_eq!(result, "e^x * (x - 1)");
    assert!(
        required.is_empty(),
        "unexpected required_conditions: {required:?}"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate((2*x+3)*exp(2*x+1), x)");
    assert_eq!(result, "e^(2 * x + 1) * (x + 1)");
    assert!(
        required.is_empty(),
        "unexpected required_conditions: {required:?}"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate((x+1)*exp((3*x+2)/2), x)");
    assert_eq!(result, "e^(1/2 * (3 * x + 2)) * (2/3 * x + 2/9)");
    assert!(
        required.is_empty(),
        "unexpected required_conditions: {required:?}"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate((x+1)*exp((2-3*x)/2), x)");
    assert_eq!(result, "e^(1/2 * (2 - 3 * x)) * (-2/3 * x - 10/9)");
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
        "1/2 * (sin(2 * x + 1) - 3 * cos(2 * x + 1) - 2 * x * cos(2 * x + 1))"
    );
    assert!(
        required.is_empty(),
        "unexpected required_conditions: {required:?}"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate((2*x+3)*cos(2*x+1), x)");
    assert_eq!(
        result,
        "1/2 * (cos(2 * x + 1) + 2 * x * sin(2 * x + 1) + 3 * sin(2 * x + 1))"
    );
    assert!(
        required.is_empty(),
        "unexpected required_conditions: {required:?}"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate((x+1)*sin((3*x+2)/2), x)");
    assert_eq!(
        result,
        "4/9 * sin(1/2 * (3 * x + 2)) - 2/3 * cos(1/2 * (3 * x + 2)) - 2/3 * x * cos(1/2 * (3 * x + 2))"
    );
    assert!(
        required.is_empty(),
        "unexpected required_conditions: {required:?}"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate((x+1)*cos((3*x+2)/2), x)");
    assert_eq!(
        result,
        "4/9 * cos(1/2 * (3 * x + 2)) + 2/3 * sin(1/2 * (3 * x + 2)) + 2/3 * x * sin(1/2 * (3 * x + 2))"
    );
    assert!(
        required.is_empty(),
        "unexpected required_conditions: {required:?}"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate((x+1)*sin((2-3*x)/2), x)");
    assert_eq!(
        result,
        "4/9 * sin(1/2 * (2 - 3 * x)) + 2/3 * cos(1/2 * (2 - 3 * x)) + 2/3 * x * cos(1/2 * (2 - 3 * x))"
    );
    assert!(
        required.is_empty(),
        "unexpected required_conditions: {required:?}"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate((x+1)*cos((2-3*x)/2), x)");
    assert_eq!(
        result,
        "4/9 * cos(1/2 * (2 - 3 * x)) - 2/3 * sin(1/2 * (2 - 3 * x)) - 2/3 * x * sin(1/2 * (2 - 3 * x))"
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
        "1/2 * (cosh(2 * x + 1) * (2 * x + 3) - sinh(2 * x + 1))"
    );
    assert!(
        required.is_empty(),
        "unexpected required_conditions: {required:?}"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate((2*x+3)*cosh(2*x+1), x)");
    assert_eq!(
        result,
        "1/2 * (sinh(2 * x + 1) * (2 * x + 3) - cosh(2 * x + 1))"
    );
    assert!(
        required.is_empty(),
        "unexpected required_conditions: {required:?}"
    );
}

#[test]
fn integrate_contract_linear_times_hyperbolic_rational_linear_by_parts() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate((x+1)*sinh((3*x+2)/2), x)");
    assert_eq!(
        result,
        "cosh(1/2 * (3 * x + 2)) * (2/3 * x + 2/3) - 4/9 * sinh(1/2 * (3 * x + 2))"
    );
    assert!(
        required.is_empty(),
        "unexpected required_conditions: {required:?}"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate((x+1)*cosh((3*x+2)/2), x)");
    assert_eq!(
        result,
        "sinh(1/2 * (3 * x + 2)) * (2/3 * x + 2/3) - 4/9 * cosh(1/2 * (3 * x + 2))"
    );
    assert!(
        required.is_empty(),
        "unexpected required_conditions: {required:?}"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate((x+1)*sinh((2-3*x)/2), x)");
    assert_eq!(
        result,
        "cosh(1/2 * (2 - 3 * x)) * (-2/3 * x - 2/3) - 4/9 * sinh(1/2 * (2 - 3 * x))"
    );
    assert!(
        required.is_empty(),
        "unexpected required_conditions: {required:?}"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate((x+1)*cosh((2-3*x)/2), x)");
    assert_eq!(
        result,
        "sinh(1/2 * (2 - 3 * x)) * (-2/3 * x - 2/3) - 4/9 * cosh(1/2 * (2 - 3 * x))"
    );
    assert!(
        required.is_empty(),
        "unexpected required_conditions: {required:?}"
    );
}

#[test]
fn integrate_contract_polynomial_derivative_exp_substitution() {
    assert_eq!(simplified_integral("integrate(2*x*exp(x^2), x)"), "e^(x^2)");
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

    assert_eq!(result, "1/2 * ln(|cosh(2 * x + 1)|)");
    assert_eq!(
        required,
        vec!["cosh(2 * x + 1) ≠ 0".to_string()],
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
        "ln(|cosh(x^2)|)"
    );
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

    assert_eq!(result, "1/2 * ln(|cosh(2 * x + 1)|)");
    assert_eq!(
        required,
        vec!["cosh(2 * x + 1) ≠ 0".to_string()],
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
        vec!["cosh(2 * x + 1) ≠ 0".to_string()],
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
        vec!["cosh(x^2) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
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

    assert_eq!(result, "-(1 / tanh(x^2))");
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
        vec!["cosh(2 * x + 1) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}

#[test]
fn integrate_contract_polynomial_hyperbolic_cosh_reciprocal_derivative_substitution() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(2*x*sinh(x^2)/cosh(x^2)^2, x)");

    assert_eq!(result, "-(1 / cosh(x^2))");
    assert_eq!(
        required,
        vec!["cosh(x^2) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(-x*sinh(x^2)/cosh(x^2)^2, x)");

    assert_eq!(result, "1 / (2 * cosh(x^2))");
    assert_eq!(
        required,
        vec!["cosh(x^2) ≠ 0".to_string()],
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

    assert_eq!(result, "-(1 / sinh(x^2))");
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

    let (result, required) = evaluated_integral_with_required_conditions("integrate(ln(2*x+1), x)");

    assert_eq!(result, "1/2 * (ln(2 * x + 1) * (2 * x + 1) - 2 * x - 1)");
    assert_eq!(
        required,
        vec!["2 * x + 1 > 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}

#[test]
fn integrate_contract_polynomial_log_product_preserves_source_domain() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(2*x*ln(x^2-1), x)");

    assert_eq!(result, "(ln(x^2 - 1) - 1) * (x^2 - 1)");
    assert_eq!(
        required,
        vec!["x^2 - 1 > 0".to_string()],
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
fn integrate_contract_reciprocal_linear_uses_abs_log() {
    assert_eq!(
        simplified_integral("integrate(1/(2*x + 1), x)"),
        "1/2 * ln(|2 * x + 1|)"
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
        "1/2 * arctan(1/2 * x^2)"
    );
}

#[test]
fn integrate_contract_scaled_polynomial_derivative_atanh_substitution() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(2*x/(4-x^4), x)");

    assert_eq!(result, "1/2 * atanh(1/2 * x^2)");
    assert_eq!(
        required,
        vec!["4 - x^4 > 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}

#[test]
fn integrate_contract_square_minus_constant_uses_abs_log_ratio_and_nonzero_domain() {
    let (result, required) = evaluated_integral_with_required_conditions("integrate(1/(x^2-1), x)");

    assert_eq!(result, "1/2 * ln(|(x - 1) / (x + 1)|)");
    assert_eq!(
        required,
        vec!["x - 1 ≠ 0".to_string(), "x + 1 ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}

#[test]
fn integrate_contract_polynomial_derivative_over_denominator_power_preserves_nonzero_domain() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(2*x/(x^2-1)^2, x)");

    assert_eq!(result, "-1 / (x^2 - 1)");
    assert_eq!(
        required,
        vec!["x - 1 ≠ 0".to_string(), "x + 1 ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}

#[test]
fn integrate_contract_shifted_polynomial_derivative_over_expanded_denominator_square_preserves_nonzero_domain(
) {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate((2*x+1)/(x^4+2*x^3-x^2-2*x+1), x)");

    assert_eq!(result, "-(1 / (x^2 + x - 1))");
    assert_eq!(
        required,
        vec!["x^2 + x - 1 ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}

#[test]
fn integrate_contract_shifted_polynomial_derivative_over_syntactic_denominator_square_preserves_nonzero_domain(
) {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate((2*x+1)/(x^2+x-1)^2, x)");

    assert_eq!(result, "-(1 / (x^2 + x - 1))");
    assert_eq!(
        required,
        vec!["x^2 + x - 1 ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}

#[test]
fn integrate_contract_shifted_polynomial_derivative_over_syntactic_denominator_cube_preserves_nonzero_domain(
) {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate((2*x+1)/(x^2+x-1)^3, x)");

    assert!(
        !result.starts_with("integrate("),
        "expected supported integral, got residual: {result}"
    );
    assert_eq!(
        required,
        vec!["x^2 + x - 1 ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
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
fn integrate_contract_scaled_polynomial_log_derivative() {
    assert_eq!(
        simplified_integral("integrate((4*x + 2)/(x^2 + x + 1), x)"),
        "2 * ln(x^2 + x + 1)"
    );
}

#[test]
fn integrate_contract_scaled_polynomial_derivative_arcsin_substitution() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(2*x/sqrt(4-x^4), x)");

    assert_eq!(result, "arcsin(1/2 * x^2)");
    assert_eq!(
        required,
        vec!["4 - x^4 > 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}

#[test]
fn integrate_contract_shifted_linear_scaled_arcsin_substitution() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(1/sqrt(4-(x+1)^2), x)");

    assert_eq!(result, "arcsin(1/2 * (x + 1))");
    assert_eq!(
        required,
        vec!["3 - x^2 - 2 * x > 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}

#[test]
fn integrate_contract_polynomial_derivative_asinh_substitution() {
    assert_eq!(
        simplified_integral("integrate(2*x/sqrt(1+x^4), x)"),
        "asinh(x^2)"
    );
}

#[test]
fn integrate_contract_scaled_polynomial_derivative_asinh_substitution() {
    assert_eq!(
        simplified_integral("integrate(2*x/sqrt(4+x^4), x)"),
        "asinh(1/2 * x^2)"
    );
}

#[test]
fn integrate_contract_shifted_linear_scaled_asinh_substitution() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(1/sqrt(4+(x+1)^2), x)");

    assert_eq!(result, "asinh(1/2 * (x + 1))");
    assert!(
        required.is_empty(),
        "unexpected required_conditions: {required:?}"
    );
}

#[test]
fn integrate_contract_polynomial_derivative_over_square_root_substitution() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(x/sqrt(x^2+1), x)");

    assert_eq!(result, "(x^2 + 1)^(1/2)");
    assert!(
        required.is_empty(),
        "unexpected required_conditions: {required:?}"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(2*x/sqrt(x^2-1), x)");

    assert_eq!(result, "2 * (x^2 - 1)^(1/2)");
    assert_eq!(
        required,
        vec!["x^2 - 1 > 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}

#[test]
fn integrate_contract_polynomial_derivative_times_square_root_substitution() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(x*sqrt(x^2+1), x)");

    assert_eq!(result, "1/3 * (x^2 + 1)^(3/2)");
    assert!(
        required.is_empty(),
        "unexpected required_conditions: {required:?}"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(2*x*sqrt(x^2-1), x)");

    assert_eq!(result, "2/3 * (x^2 - 1)^(3/2)");
    assert_eq!(
        required,
        vec!["x^2 - 1 ≥ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
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

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(2*x*(x^2-1)^(3/2), x)");

    assert_eq!(result, "2/5 * (x^2 - 1)^(5/2)");
    assert_eq!(
        required,
        vec!["x^2 - 1 ≥ 0".to_string()],
        "unexpected required_conditions: {required:?}"
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

    assert_eq!(result, "sin(2 * x + 1) / (2 * cos(2 * x + 1))");
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

    assert_eq!(result, "-(cos(2 * x + 1) / (2 * sin(2 * x + 1)))");
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

    assert_eq!(result, "sin(x^2) / (2 * cos(x^2))");
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

    assert_eq!(result, "-(cos(x^3) / (3 * sin(x^3)))");
    assert_eq!(
        required,
        vec!["sin(x^3) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
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

    assert_eq!(result, "1 / (2 * cos(2 * x + 1))");
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

    assert_eq!(result, "-(1 / (2 * sin(2 * x + 1)))");
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

    assert_eq!(result, "1 / (2 * cos(x^2))");
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

    assert_eq!(result, "-(1 / (3 * sin(x^3)))");
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
fn integrate_contract_negated_polynomial_secant_tangent_product_preserves_nonzero_domain() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(-x*sec(x^2)*tan(x^2), x)");

    assert_eq!(result, "-(1 / (2 * cos(x^2)))");
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

    assert_eq!(result, "1 / (3 * sin(x^3))");
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
fn integrate_contract_tangent_non_linear_argument_remains_residual_without_condition() {
    let (result, required) = evaluated_integral_with_required_conditions("integrate(tan(x^2), x)");

    assert_eq!(result, "integrate(tan(x^2), x)");
    assert!(
        required.is_empty(),
        "unsupported tangent integral should not invent conditions: {required:?}"
    );
}

#[test]
fn integrate_contract_unsupported_non_elementary_residual() {
    assert_eq!(
        simplified_integral("integrate(sin(x^2), x)"),
        "integrate(sin(x^2), x)"
    );
}
