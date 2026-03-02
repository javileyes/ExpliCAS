use cas_solver::{EvalAction, EvalRequest};

fn build_req(input: &str) -> Result<(cas_ast::Context, EvalRequest), String> {
    let mut ctx = cas_ast::Context::new();
    let req = cas_solver::json::build_eval_request_for_input(input, &mut ctx, true)?;
    Ok((ctx, req))
}

#[test]
fn build_eval_request_for_plain_expression_uses_simplify() {
    let (_ctx, req) = build_req("x + 1").expect("request");
    assert!(matches!(req.action, EvalAction::Simplify));
    assert!(req.auto_store);
}

#[test]
fn build_eval_request_for_plain_equation_uses_solve_with_detected_var() {
    let (ctx, req) = build_req("2*x + 1 = 0").expect("request");
    match req.action {
        EvalAction::Solve { var } => assert_eq!(var, "x"),
        _ => panic!("expected solve action"),
    }
    match ctx.get(req.parsed) {
        cas_ast::Expr::Function(sym, args) => {
            assert_eq!(ctx.sym_name(*sym), "Equal");
            assert_eq!(args.len(), 2);
        }
        _ => panic!("expected Equal(...) function"),
    }
}

#[test]
fn build_eval_request_for_special_solve_treats_expression_as_eq_zero() {
    let (ctx, req) = build_req("solve(x^2 - 4, x)").expect("request");
    match req.action {
        EvalAction::Solve { var } => assert_eq!(var, "x"),
        _ => panic!("expected solve action"),
    }
    let (lhs, rhs) = match ctx.get(req.parsed) {
        cas_ast::Expr::Function(sym, args) => {
            assert_eq!(ctx.sym_name(*sym), "Equal");
            assert_eq!(args.len(), 2);
            (args[0], args[1])
        }
        _ => panic!("expected Equal(...) function"),
    };
    let rhs_display = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: &ctx,
            id: rhs
        }
    );
    assert_eq!(rhs_display, "0");
    let lhs_display = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: &ctx,
            id: lhs
        }
    );
    assert!(!lhs_display.is_empty());
}

#[test]
fn build_eval_request_for_special_limit_uses_limit_action() {
    let (_ctx, req) = build_req("limit((x^2+1)/x, x, -inf)").expect("request");
    match req.action {
        EvalAction::Limit { var, approach } => {
            assert_eq!(var, "x");
            assert_eq!(approach, cas_solver::Approach::NegInfinity);
        }
        _ => panic!("expected limit action"),
    }
}

#[test]
fn build_eval_request_parse_errors_keep_prefix_contract() {
    let mut ctx = cas_ast::Context::new();
    let err = cas_solver::json::build_eval_request_for_input("solve(x+1=, x)", &mut ctx, false)
        .expect_err("should fail");
    assert!(err.starts_with("Parse error in solve equation:"));

    let err_limit =
        cas_solver::json::build_eval_request_for_input("limit((x+), x, inf)", &mut ctx, false)
            .expect_err("should fail");
    assert!(err_limit.starts_with("Parse error in limit expression:"));

    let err_plain = cas_solver::json::build_eval_request_for_input("x + * 2", &mut ctx, false)
        .expect_err("should fail");
    assert!(err_plain.starts_with("Parse error:"));
}

#[test]
fn format_eval_input_latex_formats_equation_and_expression() {
    let mut ctx = cas_ast::Context::new();
    let req_eq =
        cas_solver::json::build_eval_request_for_input("x + 1 = 0", &mut ctx, false).expect("eq");
    let eq_latex = cas_solver::json::format_eval_input_latex(&ctx, req_eq.parsed);
    let expected_eq_latex = cas_formatter::LaTeXExpr {
        context: &ctx,
        id: req_eq.parsed,
    }
    .to_latex();
    assert_eq!(eq_latex, expected_eq_latex);

    let req_expr =
        cas_solver::json::build_eval_request_for_input("x + 1", &mut ctx, false).expect("expr");
    let expr_latex = cas_solver::json::format_eval_input_latex(&ctx, req_expr.parsed);
    assert!(!expr_latex.is_empty());
}
