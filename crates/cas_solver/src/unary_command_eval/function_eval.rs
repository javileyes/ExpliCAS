pub(super) fn evaluate_unary_function_command_lines(
    simplifier: &mut crate::Simplifier,
    function_name: &str,
    input: &str,
    display_mode: crate::SetDisplayMode,
    show_step_assumptions: bool,
) -> Result<Vec<String>, String> {
    let parsed_expr = cas_parser::parse(input.trim(), &mut simplifier.context)
        .map_err(|e| format!("Parse error: {e}"))?;
    let call_expr = simplifier.context.call(function_name, vec![parsed_expr]);
    let (result_expr, steps) = simplifier.simplify(call_expr);
    Ok(crate::format_unary_function_eval_lines(
        &simplifier.context,
        input,
        result_expr,
        &steps,
        function_name,
        display_mode != crate::SetDisplayMode::None,
        show_step_assumptions,
    ))
}
