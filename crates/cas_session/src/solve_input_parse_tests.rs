#[test]
fn parse_solve_invocation_check_honors_flag() {
    let (check, tail) =
        crate::solve_input_parse::parse_solve_invocation_check("--check x+1=2, x", false);
    assert!(check);
    assert_eq!(tail, "x+1=2, x");
}

#[test]
fn parse_solve_command_input_accepts_comma_form() {
    let parsed = crate::solve_input_parse::parse_solve_command_input("x + 2 = 5, x");
    assert_eq!(
        parsed,
        crate::SolveCommandInput {
            equation: "x + 2 = 5".to_string(),
            variable: Some("x".to_string()),
        }
    );
}

#[test]
fn parse_timeline_command_input_routes_solve() {
    let parsed = crate::solve_input_parse::parse_timeline_command_input("solve x + 2 = 5, x");
    assert_eq!(
        parsed,
        crate::TimelineCommandInput::Solve("x + 2 = 5, x".to_string())
    );
}

#[test]
fn prepare_timeline_solve_equation_requires_equation() {
    let mut ctx = cas_ast::Context::new();
    let err = crate::solve_input_parse::prepare_timeline_solve_equation(
        &mut ctx,
        "x + 2",
        Some("x".to_string()),
    )
    .expect_err("expected equation requirement");
    assert_eq!(err, crate::SolvePrepareError::ExpectedEquation);
}
