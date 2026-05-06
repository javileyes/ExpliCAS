use cas_ast::{Context, Expr, ExprId};
use cas_formatter::DisplayExpr;
use cas_parser::parse;
use cas_session::SessionState;
use cas_solver::api::render_conditions_normalized;
use cas_solver::runtime::{
    Engine, EvalAction, EvalRequest, EvalResult, ImportanceLevel, Simplifier, StepsMode,
};

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

fn assert_rendered_antiderivative_verifies(input: &str, rendered_antiderivative: &str) {
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

    let result = match output.result {
        EvalResult::Expr(expr) => render_expr(&engine.simplifier.context, expr),
        other => panic!("expected expression result, got {other:?}"),
    };

    assert_eq!(
        result,
        "0",
        "antiderivative verification failed for {input}\nintegral result: {rendered_antiderivative}"
    );
}

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
fn integrate_contract_supported_antiderivatives_verify_by_differentiation() {
    for input in [
        "integrate(2*x + 3, x)",
        "integrate(sin(2*x), x)",
        "integrate((2*x+3)*exp(2*x+1), x)",
        "integrate(2*x*exp(x^2), x)",
        "integrate(cosh(x)/(1+sinh(x)^2), x)",
        "integrate(2*cosh(2*x+1)/(1+sinh(2*x+1)^2), x)",
        "integrate(sinh(2*x + 1)/cosh(2*x + 1), x)",
        "integrate(1/cosh(2*x + 1)^2, x)",
        "integrate(ln(2*x+1), x)",
        "integrate(1/(2*x + 1), x)",
        "integrate(1/(x^2+1), x)",
        "integrate(1/(2*sqrt(x)*(x+1)), x)",
        "integrate(arcsin(2*x+1), x)",
        "integrate(asinh(2*x+1), x)",
        "integrate(1/(x^2-1), x)",
        "integrate((2*x+1)/(x^2+x-1)^3, x)",
        "integrate(2*x/sqrt(4-x^4), x)",
        "integrate(1/sqrt(4-(x+1)^2), x)",
        "integrate(1/sqrt(4+(x+1)^2), x)",
        "integrate(sec(2*x + 1)^2, x)",
        "integrate(sec(2*x + 1)*tan(2*x + 1), x)",
    ] {
        assert_antiderivative_verifies(input);
    }
}

#[test]
fn integrate_contract_symbolic_constant_verification_preserves_independent_domain_conditions() {
    let input = "integrate(ln(y)*(z+1)^(-2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "x * ln(y) / (z + 1)^2");
    assert_eq!(
        required,
        vec!["z + 1 ≠ 0".to_string(), "y > 0".to_string()],
        "symbolic constant integration should publish independent domain conditions"
    );

    let (nested_residual, nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate(ln(y)*(z+1)^(-2), x), x) - ln(y)*(z+1)^(-2)",
    );
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["z + 1 ≠ 0".to_string(), "y > 0".to_string()],
        "nested antiderivative verification should preserve independent domain conditions"
    );
}

#[test]
#[ignore = "exhaustive debug verification is intentionally slower; CI runs the representative smoke test"]
fn integrate_contract_supported_antiderivatives_verify_by_differentiation_exhaustive() {
    for input in [
        "integrate(2*x + 3, x)",
        "integrate((3*x)^2, x)",
        "integrate(sin(2*x), x)",
        "integrate(-(sin(x)), x)",
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
        vec!["2 * x + 1 > 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_antiderivative_verifies("integrate(ln(2*x+1), x)");
    assert_rendered_antiderivative_verifies("integrate(ln(2*x+1), x)", &result);
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
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(1/(2*x + 1), x)");

    assert_eq!(result, "1/2 * ln(|2 * x + 1|)");
    assert_eq!(
        required,
        vec!["2 * x + 1 ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_rendered_antiderivative_verifies("integrate(1/(2*x + 1), x)", &result);
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
    assert_eq!(result, "arctan(1/2 * sqrt(x))");
    assert_eq!(
        required,
        vec!["x > 0".to_string()],
        "unexpected offset linear required_conditions: {required:?}"
    );
    assert_antiderivative_verifies("integrate(1/(sqrt(x)*(x+4)), x)");

    let input = "integrate(1/(sqrt(4*x+1)*(2*x+1)), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);
    assert_eq!(result, "arctan((4 * x + 1)^(1/2))");
    assert_eq!(
        required,
        vec!["4 * x + 1 > 0".to_string()],
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
        vec!["4 * x + 1 > 0".to_string()],
        "affine nested derivative should preserve the positive radicand condition"
    );
    let (nested_residual, nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate(1/(sqrt(4*x+1)*(2*x+1)), x), x) - 1/(sqrt(4*x+1)*(2*x+1))",
    );
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["4 * x + 1 > 0".to_string()],
        "affine verification should preserve the positive radicand condition"
    );

    let input = "integrate(-1/(2*sqrt(5-3*x)*(2-x)), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);
    assert_eq!(result, "arctan((5 - 3 * x)^(1/2))");
    assert_eq!(
        required,
        vec!["5 - 3 * x > 0".to_string()],
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
        vec!["5 - 3 * x > 0".to_string()],
        "negative-slope affine residual verification should preserve the positive radicand condition"
    );
    let (nested_equiv, nested_required) = evaluated_equiv_with_required_conditions(
        "diff(integrate(-1/(2*sqrt(5-3*x)*(2-x)), x), x)",
        "-1/(2*sqrt(5-3*x)*(2-x))",
    );
    assert!(nested_equiv);
    assert_eq!(
        nested_required,
        vec!["5 - 3 * x > 0".to_string()],
        "negative-slope affine derivative equivalence should preserve the positive radicand condition"
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
        "1/2 * arctan(1/2 * x^2)"
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
}

#[test]
fn integrate_contract_bounded_inverse_trig_variable_by_parts() {
    let (result, required) = evaluated_integral_with_required_conditions("integrate(arcsin(x), x)");
    assert_eq!(result, "sqrt(1 - x^2) + x * arcsin(x)");
    assert_eq!(
        required,
        vec!["1 - x^2 > 0".to_string()],
        "arcsin integration should publish its open-domain condition"
    );
    assert_antiderivative_verifies("integrate(arcsin(x), x)");
    let (nested_residual, nested_required) =
        evaluated_expr_with_required_conditions("diff(integrate(arcsin(x), x), x) - arcsin(x)");
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["1 - x^2 > 0".to_string()],
        "nested arcsin verification should preserve the open-domain condition"
    );

    let (result, required) = evaluated_integral_with_required_conditions("integrate(arccos(x), x)");
    assert_eq!(result, "x * arccos(x) - sqrt(1 - x^2)");
    assert_eq!(
        required,
        vec!["1 - x^2 > 0".to_string()],
        "arccos integration should publish its open-domain condition"
    );
    assert_antiderivative_verifies("integrate(arccos(x), x)");
    let (nested_residual, nested_required) =
        evaluated_expr_with_required_conditions("diff(integrate(arccos(x), x), x) - arccos(x)");
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["1 - x^2 > 0".to_string()],
        "nested arccos verification should preserve the open-domain condition"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(arcsin(2*x), x)");
    assert_eq!(result, "sqrt(1/4 - x^2) + x * arcsin(2 * x)");
    assert_eq!(
        required,
        vec!["1 - 4 * x^2 > 0".to_string()],
        "scaled arcsin integration should publish its open-domain condition"
    );
    assert_rendered_antiderivative_verifies("integrate(arcsin(2*x), x)", &result);
    let (nested_residual, nested_required) =
        evaluated_expr_with_required_conditions("diff(integrate(arcsin(2*x), x), x) - arcsin(2*x)");
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["1 - 4 * x^2 > 0".to_string()],
        "nested scaled arcsin verification should preserve the open-domain condition"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(arccos(2*x), x)");
    assert_eq!(result, "x * arccos(2 * x) - sqrt(1/4 - x^2)");
    assert_eq!(
        required,
        vec!["1 - 4 * x^2 > 0".to_string()],
        "scaled arccos integration should publish its open-domain condition"
    );
    assert_rendered_antiderivative_verifies("integrate(arccos(2*x), x)", &result);
    let (nested_residual, nested_required) =
        evaluated_expr_with_required_conditions("diff(integrate(arccos(2*x), x), x) - arccos(2*x)");
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["1 - 4 * x^2 > 0".to_string()],
        "nested scaled arccos verification should preserve the open-domain condition"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(arcsin(-2*x), x)");
    assert_eq!(result, "x * arcsin(-2 * x) - 1/2 * sqrt(1 - (-2 * x)^2)");
    assert_eq!(
        required,
        vec!["1 - 4 * x^2 > 0".to_string()],
        "negative-scale arcsin integration should publish its open-domain condition"
    );
    assert_rendered_antiderivative_verifies("integrate(arcsin(-2*x), x)", &result);
    let (nested_residual, nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate(arcsin(-2*x), x), x) - arcsin(-2*x)",
    );
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["1 - 4 * x^2 > 0".to_string()],
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
        vec!["-x^2 - x > 0".to_string()],
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
        vec!["-x^2 - x > 0".to_string()],
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
        vec!["-x^2 - x > 0".to_string()],
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
        vec!["-x^2 - x > 0".to_string()],
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
        vec!["x - x^2 > 0".to_string()],
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
        vec!["x - x^2 > 0".to_string()],
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
        vec!["x - x^2 > 0".to_string()],
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
        vec!["x - x^2 > 0".to_string()],
        "negative-slope shifted arccos integration should publish its open-domain condition"
    );
    assert_rendered_antiderivative_verifies("integrate(arccos(1-2*x), x)", &result);
    let (nested_residual, nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate(arccos(1-2*x), x), x) - arccos(1-2*x)",
    );
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["x - x^2 > 0".to_string()],
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
        vec!["2 * x + 1 ≠ 0".to_string()],
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
        vec!["2 * x + 1 ≠ 0".to_string()],
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
        vec!["2 * x + 1 ≠ 0".to_string()],
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
        vec!["2 * x - 1 ≠ 0".to_string()],
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
        vec!["1 - x^2 > 0".to_string()],
        "atanh integration should publish its open-interval condition"
    );
    assert_antiderivative_verifies("integrate(atanh(x), x)");
    assert_rendered_antiderivative_verifies("integrate(atanh(x), x)", &result);
    let (nested_residual, nested_required) =
        evaluated_expr_with_required_conditions("diff(integrate(atanh(x), x), x) - atanh(x)");
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["1 - x^2 > 0".to_string()],
        "nested atanh verification should preserve the open-interval condition"
    );

    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(atanh(2*x), x)");
    assert_eq!(result, "1/4 * ln(1 - (2 * x)^2) + x * atanh(2 * x)");
    assert_eq!(
        required,
        vec!["1 - 4 * x^2 > 0".to_string()],
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
        vec!["-x^2 - x > 0".to_string()],
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
        vec!["-x^2 - x > 0".to_string()],
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
        vec!["x - x^2 > 0".to_string()],
        "negative-slope atanh integration should publish its normalized open-interval condition"
    );
    assert_antiderivative_verifies("integrate(atanh(1-2*x), x)");
    assert_rendered_antiderivative_verifies("integrate(atanh(1-2*x), x)", &result);
    let (nested_residual, nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate(atanh(1-2*x), x), x) - atanh(1-2*x)",
    );
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["x - x^2 > 0".to_string()],
        "nested negative-slope atanh verification should preserve the normalized condition"
    );
}

#[test]
fn integrate_contract_acosh_affine_by_parts_preserves_real_radical_domain() {
    let (result, required) = evaluated_integral_with_required_conditions("integrate(acosh(x), x)");
    assert_eq!(result, "x * acosh(x) - sqrt(x - 1) * sqrt(x + 1)");
    assert_eq!(
        required,
        vec!["x - 1 > 0".to_string()],
        "acosh integration should publish the real radical conditions"
    );
    assert_rendered_antiderivative_verifies("integrate(acosh(x), x)", &result);
    let (nested_residual, nested_required) =
        evaluated_expr_with_required_conditions("diff(integrate(acosh(x), x), x) - acosh(x)");
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["x - 1 > 0".to_string()],
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
        vec!["2 * x - 1 > 0".to_string()],
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
        vec!["-x > 0".to_string()],
        "negative-slope acosh integration should publish its normalized real-domain condition"
    );
    assert_rendered_antiderivative_verifies("integrate(acosh(1-2*x), x)", &result);
    let (nested_residual, nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate(acosh(1-2*x), x), x) - acosh(1-2*x)",
    );
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["-x > 0".to_string()],
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
fn integrate_contract_scaled_polynomial_derivative_atanh_substitution() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(2*x/(4-x^4), x)");

    assert_eq!(result, "1/2 * atanh(x^2 / 2)");
    assert_eq!(
        required,
        vec!["4 - x^4 > 0".to_string()],
        "unexpected required_conditions: {required:?}"
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
        vec!["3 - x^2 > 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
}

#[test]
fn integrate_contract_scaled_atanh_quadratic_kernel_reduces_surd_width() {
    let input = "integrate(1/(12-4*x^2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "1/4 * atanh(x / sqrt(3)) / sqrt(3)");
    assert_eq!(
        required,
        vec!["3 - x^2 > 0".to_string()],
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

    assert_eq!(result, "-1 / (x^2 + x - 1)");
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
                || rule_name == "Extract Common Multiplicative Factor"),
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
                || rule_name == "Extract Common Multiplicative Factor"),
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
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate((8*x+2)/(3*(2*x^2+x-1)^3), x)");

    assert_eq!(result, "-1 / (3 * (2 * x^2 + x - 1)^2)");
    assert_eq!(
        required,
        vec!["x + 1 ≠ 0".to_string(), "2 * x - 1 ≠ 0".to_string()],
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
        vec!["x + 1 ≠ 0".to_string()],
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

    assert_eq!(result, "(x^2 + 1) * (ln(x^2 + 1)^2 + 2 - 2 * ln(x^2 + 1))");
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
        vec!["x^2 - 1 > 0".to_string()],
        "conditional log-cube product integration should publish its positive-domain condition"
    );
    assert_antiderivative_verifies(input);

    let (nested_residual, nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate(2*x*ln(x^2-1)^3, x), x) - 2*x*ln(x^2-1)^3",
    );
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["x^2 - 1 > 0".to_string()],
        "nested conditional log-cube verification should preserve the positive-domain condition"
    );
}

#[test]
fn integrate_contract_linear_log_square_product_by_parts_preserves_positive_domain() {
    let input = "integrate(ln(2*x+1)^2, x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(
        result,
        "1/2 * (2 * x + 1) * (ln(2 * x + 1)^2 + 2 - 2 * ln(2 * x + 1))"
    );
    assert_eq!(
        required,
        vec!["2 * x + 1 > 0".to_string()],
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
        "(x^2 + x + 1) * (ln(x^2 + x + 1)^2 + 2 - 2 * ln(x^2 + x + 1))"
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

    assert_eq!(result, "(x^2 - 1) * (ln(x^2 - 1)^2 + 2 - 2 * ln(x^2 - 1))");
    assert_eq!(
        required,
        vec!["x^2 - 1 > 0".to_string()],
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
        "(ln(x^2 + x - 1)^2 + 2 - 2 * ln(x^2 + x - 1)) * (x^2 + x - 1)"
    );
    assert_eq!(
        required,
        vec!["x^2 + x - 1 > 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
}

#[test]
fn integrate_contract_conditional_cubic_log_square_product_by_parts_verifies() {
    let input = "integrate((3*x^2-1)*ln(x^3-x)^2, x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "(x^3 - x) * (ln(x^3 - x)^2 + 2 - 2 * ln(x^3 - x))");
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
        "(ln(x^4 - x^2 - 1)^2 + 2 - 2 * ln(x^4 - x^2 - 1)) * (x^4 - x^2 - 1)"
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

    assert_eq!(result, "arcsin(1/2 * x^2)");
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
fn integrate_contract_shifted_linear_scaled_arcsin_substitution() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(1/sqrt(4-(x+1)^2), x)");

    assert_eq!(result, "arcsin(1/2 * (x + 1))");
    assert_eq!(
        required,
        vec!["3 - x^2 - 2 * x > 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_antiderivative_verifies("integrate(1/sqrt(4-(x+1)^2), x)");
    let (nested_residual, nested_required) = evaluated_expr_with_required_conditions(
        "diff(integrate(1/sqrt(4-(x+1)^2), x), x) - 1/sqrt(4-(x+1)^2)",
    );
    assert_eq!(nested_residual, "0");
    assert_eq!(
        nested_required,
        vec!["3 - x^2 - 2 * x > 0".to_string()],
        "nested shifted arcsin verification should preserve its positive radicand condition"
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
fn integrate_contract_scaled_polynomial_derivative_asinh_substitution() {
    assert_eq!(
        simplified_integral("integrate(2*x/sqrt(4+x^4), x)"),
        "asinh(1/2 * x^2)"
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
fn integrate_contract_shifted_linear_scaled_asinh_substitution() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(1/sqrt(4+(x+1)^2), x)");

    assert_eq!(result, "asinh(1/2 * (x + 1))");
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

    assert_eq!(result, "-cos(2 * x + 1) / (2 * sin(2 * x + 1))");
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

    assert_eq!(result, "-cos(x^3) / (3 * sin(x^3))");
    assert_eq!(
        required,
        vec!["sin(x^3) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
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
    assert_antiderivative_verifies(input);

    let input = "integrate((4*x^3-2*x)*cot(x^4-x^2), x)";
    let (result, required) = evaluated_integral_with_required_conditions(input);

    assert_eq!(result, "ln(|sin(x^4 - x^2)|)");
    assert_eq!(
        required,
        vec!["sin(x^4 - x^2) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
    assert_antiderivative_verifies(input);
}

#[test]
fn integrate_contract_linear_secant_uses_abs_log_and_nonzero_domain() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(sec(2*x + 1), x)");

    assert_eq!(result, "1/2 * ln(|(sin(2 * x + 1) + 1) / cos(2 * x + 1)|)");
    assert_eq!(
        required,
        vec!["cos(2 * x + 1) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}

#[test]
fn integrate_contract_linear_cosecant_uses_abs_log_and_nonzero_domain() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(csc(2*x + 1), x)");

    assert_eq!(result, "1/2 * ln(|(cos(2 * x + 1) - 1) / sin(2 * x + 1)|)");
    assert_eq!(
        required,
        vec!["sin(2 * x + 1) ≠ 0".to_string()],
        "unexpected required_conditions: {required:?}"
    );
}

#[test]
fn integrate_contract_scaled_affine_secant_cosecant_uses_abs_log_and_nonzero_domain() {
    let cases = [
        (
            "integrate(sec((3*x+2)/2), x)",
            "2/3 * ln(|(sin(1/2 * (3 * x + 2)) + 1) / cos(1/2 * (3 * x + 2))|)",
            "cos(1/2 * (3 * x + 2)) ≠ 0",
        ),
        (
            "integrate(csc((2-3*x)/2), x)",
            "-2/3 * ln(|(cos(1/2 * (2 - 3 * x)) - 1) / sin(1/2 * (2 - 3 * x))|)",
            "sin(1/2 * (2 - 3 * x)) ≠ 0",
        ),
    ];

    for (input, expected_result, expected_condition) in cases {
        let (result, required) = evaluated_integral_with_required_conditions(input);

        assert_eq!(result, expected_result, "unexpected result for {input}");
        assert_eq!(
            required,
            vec![expected_condition.to_string()],
            "unexpected required_conditions for {input}: {required:?}"
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
        let (result, required) = evaluated_expr_with_required_conditions(input);
        assert_eq!(result, "0", "unexpected nested residual for {input}");
        assert_eq!(
            required,
            vec![expected_condition.to_string()],
            "nested residual should preserve required domain for {input}: {required:?}"
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
            vec!["cos(x^4 - x^2) ≠ 0", "x + 1 ≠ 0"],
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
    ];

    for (input, expected_condition) in cases {
        let (result, required) = evaluated_expr_with_required_conditions(input);
        assert_eq!(result, "0", "unexpected nested residual for {input}");
        assert_eq!(
            required,
            vec![expected_condition.to_string()],
            "nested residual should preserve required domain for {input}: {required:?}"
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
            vec!["sin(x^4 - x^2) ≠ 0", "x + 1 ≠ 0"],
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

    assert_eq!(result, "-1 / (2 * sin(2 * x + 1))");
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

    assert_eq!(result, "-1 / (3 * sin(x^3))");
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
fn integrate_contract_sqrt_chain_secant_cosecant_products_verify() {
    let cases = [
        (
            "integrate(sec(sqrt(x))*tan(sqrt(x))/(2*sqrt(x)), x)",
            "sec(sqrt(x))",
            "tan(sqrt(x)) * sec(sqrt(x)) / (2 * sqrt(x))",
            vec!["x > 0", "cos(sqrt(x)) ≠ 0"],
        ),
        (
            "integrate(-sec(sqrt(x))*tan(sqrt(x))/(2*sqrt(x)), x)",
            "-sec(sqrt(x))",
            "-tan(sqrt(x)) * sec(sqrt(x)) / (2 * sqrt(x))",
            vec!["x > 0", "cos(sqrt(x)) ≠ 0"],
        ),
        (
            "integrate(csc(sqrt(x))*cot(sqrt(x))/(2*sqrt(x)), x)",
            "-csc(sqrt(x))",
            "csc(sqrt(x)) * cot(sqrt(x)) / (2 * sqrt(x))",
            vec!["x > 0", "sin(sqrt(x)) ≠ 0"],
        ),
        (
            "integrate(-csc(sqrt(x))*cot(sqrt(x))/(2*sqrt(x)), x)",
            "csc(sqrt(x))",
            "-cot(sqrt(x)) * csc(sqrt(x)) / (2 * sqrt(x))",
            vec!["x > 0", "sin(sqrt(x)) ≠ 0"],
        ),
        (
            "integrate(sec(sqrt(2*x))*tan(sqrt(2*x))/sqrt(2*x), x)",
            "sec(sqrt(2 * x))",
            "tan(sqrt(2 * x)) * sec(sqrt(2 * x)) / sqrt(2 * x)",
            vec!["x > 0", "cos(sqrt(2 * x)) ≠ 0"],
        ),
        (
            "integrate(csc(sqrt(2*x))*cot(sqrt(2*x))/sqrt(2*x), x)",
            "-csc(sqrt(2 * x))",
            "csc(sqrt(2 * x)) * cot(sqrt(2 * x)) / sqrt(2 * x)",
            vec!["x > 0", "sin(sqrt(2 * x)) ≠ 0"],
        ),
        (
            "integrate(-sec(sqrt(2*x))*tan(sqrt(2*x))/sqrt(2*x), x)",
            "-sec(sqrt(2 * x))",
            "-tan(sqrt(2 * x)) * sec(sqrt(2 * x)) / sqrt(2 * x)",
            vec!["x > 0", "cos(sqrt(2 * x)) ≠ 0"],
        ),
        (
            "integrate(-csc(sqrt(2*x))*cot(sqrt(2*x))/sqrt(2*x), x)",
            "csc(sqrt(2 * x))",
            "-cot(sqrt(2 * x)) * csc(sqrt(2 * x)) / sqrt(2 * x)",
            vec!["x > 0", "sin(sqrt(2 * x)) ≠ 0"],
        ),
        (
            "integrate(sec(sqrt(3*x+1))*tan(sqrt(3*x+1))*3/(2*sqrt(3*x+1)), x)",
            "sec(sqrt(3 * x + 1))",
            "3 * tan(sqrt(3 * x + 1)) * sec(sqrt(3 * x + 1)) / (2 * sqrt(3 * x + 1))",
            vec!["3 * x + 1 > 0", "cos(sqrt(3 * x + 1)) ≠ 0"],
        ),
        (
            "integrate(-sec(sqrt(3-2*x))*tan(sqrt(3-2*x))/sqrt(3-2*x), x)",
            "sec(sqrt(3 - 2 * x))",
            "-tan(sqrt(3 - 2 * x)) * sec(sqrt(3 - 2 * x)) / sqrt(3 - 2 * x)",
            vec!["3 - 2 * x > 0", "cos(sqrt(3 - 2 * x)) ≠ 0"],
        ),
        (
            "integrate(sec(sqrt(3-2*x))*tan(sqrt(3-2*x))/sqrt(3-2*x), x)",
            "-sec(sqrt(3 - 2 * x))",
            "tan(sqrt(3 - 2 * x)) * sec(sqrt(3 - 2 * x)) / sqrt(3 - 2 * x)",
            vec!["3 - 2 * x > 0", "cos(sqrt(3 - 2 * x)) ≠ 0"],
        ),
        (
            "integrate(csc(sqrt(3*x+1))*cot(sqrt(3*x+1))*3/(2*sqrt(3*x+1)), x)",
            "-csc(sqrt(3 * x + 1))",
            "3 * csc(sqrt(3 * x + 1)) * cot(sqrt(3 * x + 1)) / (2 * sqrt(3 * x + 1))",
            vec!["3 * x + 1 > 0", "sin(sqrt(3 * x + 1)) ≠ 0"],
        ),
        (
            "integrate(-csc(sqrt(3-2*x))*cot(sqrt(3-2*x))/sqrt(3-2*x), x)",
            "-csc(sqrt(3 - 2 * x))",
            "-cot(sqrt(3 - 2 * x)) * csc(sqrt(3 - 2 * x)) / sqrt(3 - 2 * x)",
            vec!["3 - 2 * x > 0", "sin(sqrt(3 - 2 * x)) ≠ 0"],
        ),
        (
            "integrate(csc(sqrt(3-2*x))*cot(sqrt(3-2*x))/sqrt(3-2*x), x)",
            "csc(sqrt(3 - 2 * x))",
            "csc(sqrt(3 - 2 * x)) * cot(sqrt(3 - 2 * x)) / sqrt(3 - 2 * x)",
            vec!["3 - 2 * x > 0", "sin(sqrt(3 - 2 * x)) ≠ 0"],
        ),
        (
            "integrate(-csc(sqrt(3*x+1))*cot(sqrt(3*x+1))*3/(2*sqrt(3*x+1)), x)",
            "csc(sqrt(3 * x + 1))",
            "-3 * cot(sqrt(3 * x + 1)) * csc(sqrt(3 * x + 1)) / (2 * sqrt(3 * x + 1))",
            vec!["3 * x + 1 > 0", "sin(sqrt(3 * x + 1)) ≠ 0"],
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
fn integrate_contract_sqrt_chain_tangent_cotangent_logs_verify() {
    let cases = [
        (
            "integrate(tan(sqrt(x))/(2*sqrt(x)), x)",
            "-ln(|cos(sqrt(x))|)",
            "tan(sqrt(x)) / (2 * sqrt(x))",
            vec!["x > 0", "cos(sqrt(x)) ≠ 0"],
            vec!["x > 0", "cos(sqrt(x)) ≠ 0"],
        ),
        (
            "integrate(cot(sqrt(x))/(2*sqrt(x)), x)",
            "ln(|sin(sqrt(x))|)",
            "cot(sqrt(x)) / (2 * sqrt(x))",
            vec!["x > 0", "sin(sqrt(x)) ≠ 0"],
            vec!["x > 0", "sin(sqrt(x)) ≠ 0"],
        ),
        (
            "integrate(-tan(sqrt(x))/(2*sqrt(x)), x)",
            "ln(|cos(sqrt(x))|)",
            "-tan(sqrt(x)) / (2 * sqrt(x))",
            vec!["x > 0", "cos(sqrt(x)) ≠ 0"],
            vec!["x > 0", "cos(sqrt(x)) ≠ 0"],
        ),
        (
            "integrate(-cot(sqrt(x))/(2*sqrt(x)), x)",
            "-ln(|sin(sqrt(x))|)",
            "-cot(sqrt(x)) / (2 * sqrt(x))",
            vec!["x > 0", "sin(sqrt(x)) ≠ 0"],
            vec!["x > 0", "sin(sqrt(x)) ≠ 0"],
        ),
        (
            "integrate(tan(sqrt(2*x))/sqrt(2*x), x)",
            "-ln(|cos(sqrt(2 * x))|)",
            "tan(sqrt(2 * x)) / sqrt(2 * x)",
            vec!["x > 0", "cos(sqrt(2 * x)) ≠ 0"],
            vec!["x > 0", "cos(sqrt(2 * x)) ≠ 0"],
        ),
        (
            "integrate(-tan(sqrt(2*x))/sqrt(2*x), x)",
            "ln(|cos(sqrt(2 * x))|)",
            "-tan(sqrt(2 * x)) / sqrt(2 * x)",
            vec!["x > 0", "cos(sqrt(2 * x)) ≠ 0"],
            vec!["x > 0", "cos(sqrt(2 * x)) ≠ 0"],
        ),
        (
            "integrate(-cot(sqrt(2*x))/sqrt(2*x), x)",
            "-ln(|sin(sqrt(2 * x))|)",
            "-cot(sqrt(2 * x)) / sqrt(2 * x)",
            vec!["x > 0", "sin(sqrt(2 * x)) ≠ 0"],
            vec!["x > 0", "sin(sqrt(2 * x)) ≠ 0"],
        ),
        (
            "integrate(tan(sqrt(3*x+1))*3/(2*sqrt(3*x+1)), x)",
            "-ln(|cos(sqrt(3 * x + 1))|)",
            "3 * tan(sqrt(3 * x + 1)) / (2 * sqrt(3 * x + 1))",
            vec!["3 * x + 1 > 0", "cos(sqrt(3 * x + 1)) ≠ 0"],
            vec!["3 * x + 1 > 0", "cos(sqrt(3 * x + 1)) ≠ 0"],
        ),
        (
            "integrate(cot(sqrt(3*x+1))*3/(2*sqrt(3*x+1)), x)",
            "ln(|sin(sqrt(3 * x + 1))|)",
            "3 * cot(sqrt(3 * x + 1)) / (2 * sqrt(3 * x + 1))",
            vec!["3 * x + 1 > 0", "sin(sqrt(3 * x + 1)) ≠ 0"],
            vec!["3 * x + 1 > 0", "sin(sqrt(3 * x + 1)) ≠ 0"],
        ),
        (
            "integrate(-tan(sqrt(3*x+1))*3/(2*sqrt(3*x+1)), x)",
            "ln(|cos(sqrt(3 * x + 1))|)",
            "-3 * tan(sqrt(3 * x + 1)) / (2 * sqrt(3 * x + 1))",
            vec!["3 * x + 1 > 0", "cos(sqrt(3 * x + 1)) ≠ 0"],
            vec!["3 * x + 1 > 0", "cos(sqrt(3 * x + 1)) ≠ 0"],
        ),
        (
            "integrate(-tan(sqrt(3-2*x))/sqrt(3-2*x), x)",
            "-ln(|cos(sqrt(3 - 2 * x))|)",
            "-tan(sqrt(3 - 2 * x)) / sqrt(3 - 2 * x)",
            vec!["3 - 2 * x > 0", "cos(sqrt(3 - 2 * x)) ≠ 0"],
            vec!["3 - 2 * x > 0", "cos(sqrt(3 - 2 * x)) ≠ 0"],
        ),
        (
            "integrate(-cot(sqrt(3*x+1))*3/(2*sqrt(3*x+1)), x)",
            "-ln(|sin(sqrt(3 * x + 1))|)",
            "-3 * cot(sqrt(3 * x + 1)) / (2 * sqrt(3 * x + 1))",
            vec!["3 * x + 1 > 0", "sin(sqrt(3 * x + 1)) ≠ 0"],
            vec!["3 * x + 1 > 0", "sin(sqrt(3 * x + 1)) ≠ 0"],
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
            vec!["3 * x + 1 > 0"],
        ),
        (
            "integrate(3/(2*sqrt(3*x+1)*tanh(sqrt(3*x+1))), x)",
            "ln(|sinh(sqrt(3 * x + 1))|)",
            "3 / (2 * tanh(sqrt(3 * x + 1)) * sqrt(3 * x + 1))",
            vec!["sinh(sqrt(3 * x + 1)) ≠ 0", "3 * x + 1 > 0"],
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
            vec!["3 * x + 1 > 0"],
        ),
        (
            "integrate(3/(2*sqrt(3*x+1)*sinh(sqrt(3*x+1))^2), x)",
            "-1 / tanh(sqrt(3 * x + 1))",
            vec!["3 * x + 1 > 0", "sinh(sqrt(3 * x + 1)) ≠ 0"],
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
            vec!["3 * x + 1 > 0"],
        ),
        (
            "diff(integrate(3/(2*sqrt(3*x+1)*sinh(sqrt(3*x+1))^2), x), x)",
            "3 / (2 * sqrt(3 * x + 1) * sinh(sqrt(3 * x + 1))^2)",
            vec!["3 * x + 1 > 0", "sinh(sqrt(3 * x + 1)) ≠ 0"],
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
            vec!["3 * x + 1 > 0"],
        ),
        (
            "integrate(3*cosh(sqrt(3*x+1))/(2*sqrt(3*x+1)*sinh(sqrt(3*x+1))^2), x)",
            "-1 / sinh(sqrt(3 * x + 1))",
            vec!["3 * x + 1 > 0", "sinh(sqrt(3 * x + 1)) ≠ 0"],
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
            vec!["3 * x + 1 > 0"],
        ),
        (
            "diff(integrate(3*cosh(sqrt(3*x+1))/(2*sqrt(3*x+1)*sinh(sqrt(3*x+1))^2), x), x)",
            "3 * cosh(sqrt(3 * x + 1)) / (2 * sqrt(3 * x + 1) * sinh(sqrt(3 * x + 1))^2)",
            vec!["3 * x + 1 > 0", "sinh(sqrt(3 * x + 1)) ≠ 0"],
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
fn integrate_contract_negated_polynomial_secant_tangent_product_preserves_nonzero_domain() {
    let (result, required) =
        evaluated_integral_with_required_conditions("integrate(-x*sec(x^2)*tan(x^2), x)");

    assert_eq!(result, "-1 / (2 * cos(x^2))");
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
