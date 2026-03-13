#[test]
fn evaluate_unary_function_command_lines_runs() {
    use crate::unary_command_eval::evaluate_unary_function_command_lines;

    let mut simplifier = crate::Simplifier::with_default_rules();
    let lines = evaluate_unary_function_command_lines(
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
    use crate::unary_command_eval::evaluate_unary_command_lines;

    let mut simplifier = crate::Simplifier::with_default_rules();
    let lines = evaluate_unary_command_lines(
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
    use crate::unary_command_eval::evaluate_unary_command_message;

    let mut simplifier = crate::Simplifier::with_default_rules();
    let message = evaluate_unary_command_message(
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
