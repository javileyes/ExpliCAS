#[test]
fn extract_simplify_command_tail_trims_prefix() {
    assert_eq!(crate::extract_simplify_command_tail("simplify x+1"), "x+1");
}

#[test]
fn evaluate_full_simplify_command_lines_runs() {
    let mut simplifier = cas_solver::Simplifier::with_default_rules();
    let session = crate::SessionState::new();
    let lines = crate::evaluate_full_simplify_command_lines(
        &mut simplifier,
        &session,
        "simplify x + 0",
        crate::SetDisplayMode::Normal,
    )
    .expect("simplify");
    assert!(lines.iter().any(|line| line.starts_with("Result:")));
}
