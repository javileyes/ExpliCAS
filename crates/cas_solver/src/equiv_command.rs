fn evaluate_equiv_input(
    simplifier: &mut crate::Simplifier,
    input: &str,
) -> Result<crate::EquivalenceResult, crate::ParseExprPairError> {
    let (lhs, rhs) = crate::parse_expr_pair(&mut simplifier.context, input)?;
    Ok(simplifier.are_equivalent_extended(lhs, rhs))
}

/// Evaluate equivalence command input and format user-facing output lines.
pub fn evaluate_equiv_command_lines(
    simplifier: &mut crate::Simplifier,
    input: &str,
) -> Result<Vec<String>, crate::ParseExprPairError> {
    let result = evaluate_equiv_input(simplifier, input)?;
    Ok(crate::format_equivalence_result_lines(&result))
}

/// Evaluate `equiv` command input and return user-facing message text.
pub fn evaluate_equiv_command_message(
    simplifier: &mut crate::Simplifier,
    input: &str,
) -> Result<String, crate::ParseExprPairError> {
    Ok(evaluate_equiv_command_lines(simplifier, input)?.join("\n"))
}

/// Evaluate full `equiv ...` invocation and return user-facing message text.
pub fn evaluate_equiv_invocation_message(
    simplifier: &mut crate::Simplifier,
    line: &str,
) -> Result<String, String> {
    let input = crate::extract_equiv_command_tail(line);
    evaluate_equiv_command_message(simplifier, input)
        .map_err(|error| crate::format_expr_pair_parse_error_message(&error, "equiv"))
}
