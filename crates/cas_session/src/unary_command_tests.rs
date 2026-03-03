#[test]
fn evaluate_unary_function_command_lines_runs() {
    let mut simplifier = cas_solver::Simplifier::with_default_rules();
    let lines = crate::unary_command::evaluate_unary_function_command_lines(
        &mut simplifier,
        "trace",
        "[[1,2],[3,4]]",
        crate::SetDisplayMode::Normal,
        false,
    )
    .expect("unary eval");
    assert!(lines.iter().any(|line| line.starts_with("Result:")));
}

#[test]
fn evaluate_unary_command_lines_trims_command_prefix() {
    let mut simplifier = cas_solver::Simplifier::with_default_rules();
    let lines = crate::unary_command::evaluate_unary_command_lines(
        &mut simplifier,
        "trace [[1,2],[3,4]]",
        "trace",
        crate::SetDisplayMode::None,
        false,
        true,
    )
    .expect("unary eval");
    assert!(lines
        .first()
        .is_some_and(|line| line.starts_with("Parsed: trace(")));
}

#[test]
fn evaluate_unary_command_message_joins_lines() {
    let mut simplifier = cas_solver::Simplifier::with_default_rules();
    let message = crate::unary_command::evaluate_unary_command_message(
        &mut simplifier,
        "trace [[1,2],[3,4]]",
        "trace",
        crate::SetDisplayMode::None,
        false,
        true,
    )
    .expect("unary eval");
    assert!(message.contains("Result:"));
}
