use crate::eval_input_variable::detect_solve_variable_for_eval_request;

#[test]
fn detect_solve_variable_prefers_x_when_ambiguous() {
    let mut ctx = cas_ast::Context::new();
    let statement = cas_parser::parse_statement("x + y = 0", &mut ctx).expect("parse");
    let (lhs, rhs) = match statement {
        cas_parser::Statement::Equation(eq) => (eq.lhs, eq.rhs),
        _ => panic!("expected equation"),
    };
    let var = detect_solve_variable_for_eval_request(&mut ctx, lhs, rhs);
    assert_eq!(var, "x");
}

#[test]
fn detect_solve_variable_falls_back_to_preferred_order() {
    let mut ctx = cas_ast::Context::new();
    let statement = cas_parser::parse_statement("z + y = 0", &mut ctx).expect("parse");
    let (lhs, rhs) = match statement {
        cas_parser::Statement::Equation(eq) => (eq.lhs, eq.rhs),
        _ => panic!("expected equation"),
    };
    let var = detect_solve_variable_for_eval_request(&mut ctx, lhs, rhs);
    assert_eq!(var, "y");
}

#[test]
fn build_prepared_eval_request_wraps_equation_as_solve_variant() {
    let mut ctx = cas_ast::Context::new();
    let prepared =
        crate::eval_input::build_prepared_eval_request_for_input("x + 1 = 0", &mut ctx, false)
            .expect("request");

    match prepared {
        crate::eval_input::PreparedEvalRequest::Solve {
            var, auto_store, ..
        } => {
            assert_eq!(var, "x");
            assert!(!auto_store);
        }
        _ => panic!("expected solve variant"),
    }
}

#[test]
fn build_prepared_eval_request_preserves_not_equal_relation() {
    let mut ctx = cas_ast::Context::new();
    let prepared =
        crate::eval_input::build_prepared_eval_request_for_input("x != 0", &mut ctx, false)
            .expect("request");

    match prepared {
        crate::eval_input::PreparedEvalRequest::Solve {
            original_equation, ..
        } => {
            let original_equation = original_equation.expect("equation should be preserved");
            assert_eq!(original_equation.op, cas_ast::RelOp::Neq);
        }
        _ => panic!("expected solve variant"),
    }
}

#[test]
fn build_prepared_eval_request_parses_limit_as_non_solve_action() {
    let mut ctx = cas_ast::Context::new();
    let prepared = crate::eval_input::build_prepared_eval_request_for_input(
        "limit(x^2, x, +inf)",
        &mut ctx,
        true,
    )
    .expect("request");

    match prepared {
        crate::eval_input::PreparedEvalRequest::Eval {
            action, auto_store, ..
        } => {
            assert!(auto_store);
            match action {
                crate::eval_input::EvalNonSolveAction::Limit { var, approach } => {
                    assert_eq!(var, "x");
                    assert_eq!(approach, cas_math::limit_types::Approach::PosInfinity);
                }
                _ => panic!("expected limit action"),
            }
        }
        _ => panic!("expected eval variant"),
    }
}
