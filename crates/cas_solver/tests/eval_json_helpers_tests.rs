use cas_solver::json::{
    collect_required_conditions_eval_json, collect_required_display_eval_json,
    collect_warnings_eval_json, parse_eval_json_special_command, EvalJsonSpecialCommand,
};
use cas_solver::{DomainMode, Engine, EvalAction, EvalOptions, EvalRequest, EvalResult};

fn eval_expr(expr: &str, domain_mode: DomainMode) -> (Engine, cas_solver::EvalOutput) {
    let mut engine = Engine::new();
    let parsed = cas_parser::parse(expr, &mut engine.simplifier.context).expect("parse");
    let mut opts = EvalOptions::default();
    opts.shared.semantics.domain_mode = domain_mode;
    let output = engine
        .eval_stateless(
            opts,
            EvalRequest {
                raw_input: expr.to_string(),
                parsed,
                action: EvalAction::Simplify,
                auto_store: false,
            },
        )
        .expect("eval");
    (engine, output)
}

#[test]
fn parse_eval_json_special_command_parses_solve_and_limit() {
    let solve = parse_eval_json_special_command("solve((x+1)=0, x)").expect("solve parse");
    assert_eq!(
        solve,
        EvalJsonSpecialCommand::Solve {
            equation: "(x+1)=0".to_string(),
            var: "x".to_string()
        }
    );

    let limit = parse_eval_json_special_command("limit((x^2+1)/x, x, -inf)").expect("limit parse");
    assert_eq!(
        limit,
        EvalJsonSpecialCommand::Limit {
            expr: "(x^2+1)/x".to_string(),
            var: "x".to_string(),
            approach: cas_solver::Approach::NegInfinity
        }
    );
}

#[test]
fn parse_eval_json_special_command_rejects_invalid_input() {
    assert!(parse_eval_json_special_command("solve(x+1=0)").is_none());
    assert!(parse_eval_json_special_command("limit(x, x, sideways)").is_none());
    assert!(parse_eval_json_special_command("x + 1").is_none());
}

#[test]
fn output_helpers_map_required_conditions_and_warnings() {
    let (engine, output) = eval_expr("1/x", DomainMode::Generic);
    let ctx = &engine.simplifier.context;
    match output.result {
        EvalResult::Expr(_) | EvalResult::Set(_) | EvalResult::SolutionSet(_) => {}
        EvalResult::Bool(_) | EvalResult::None => {}
    }

    let required = collect_required_conditions_eval_json(&output, ctx);
    let display = collect_required_display_eval_json(&output, ctx);
    let warnings = collect_warnings_eval_json(&output);

    assert_eq!(warnings.len(), output.domain_warnings.len());
    assert_eq!(required.len(), output.required_conditions.len());
    assert_eq!(display.len(), output.required_conditions.len());
    if let Some(first) = required.first() {
        assert!(!first.kind.is_empty());
        assert!(!first.expr_display.is_empty());
    }
}
