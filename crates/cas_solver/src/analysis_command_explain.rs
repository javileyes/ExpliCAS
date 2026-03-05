use cas_ast::Context;

use crate::ExplainCommandEvalError;

/// Evaluate explain command input and format user-facing output lines.
pub fn evaluate_explain_command_lines(
    ctx: &mut Context,
    input: &str,
) -> Result<Vec<String>, ExplainCommandEvalError> {
    let parsed_expr = cas_parser::parse(input.trim(), ctx)
        .map_err(|e| ExplainCommandEvalError::Parse(e.to_string()))?;
    let expr_data = ctx.get(parsed_expr).clone();
    let cas_ast::Expr::Function(name_id, args) = expr_data else {
        return Err(ExplainCommandEvalError::ExpectedFunctionCall);
    };
    let function_name = ctx.sym_name(name_id).to_string();
    if function_name != "gcd" {
        return Err(ExplainCommandEvalError::UnsupportedFunction(function_name));
    }
    if args.len() != 2 {
        return Err(ExplainCommandEvalError::InvalidArity {
            function: function_name,
            expected: 2,
            found: args.len(),
        });
    }
    let result = crate::number_theory::explain_gcd(ctx, args[0], args[1]);
    Ok(crate::format_explain_gcd_eval_lines(
        ctx,
        input,
        &result.steps,
        result.value,
    ))
}

/// Evaluate `explain` command input and return cleaned message text.
pub fn evaluate_explain_command_message(
    ctx: &mut Context,
    input: &str,
) -> Result<String, ExplainCommandEvalError> {
    let mut lines = evaluate_explain_command_lines(ctx, input)?;
    crate::clean_result_output_line(&mut lines);
    Ok(lines.join("\n"))
}

/// Evaluate full `explain ...` invocation and return user-facing message text.
pub fn evaluate_explain_invocation_message(
    ctx: &mut cas_ast::Context,
    line: &str,
) -> Result<String, String> {
    let input = crate::extract_explain_command_tail(line);
    evaluate_explain_command_message(ctx, input)
        .map_err(|error| crate::format_explain_command_error_message(&error))
}
