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
