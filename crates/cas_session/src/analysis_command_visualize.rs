use cas_ast::Context;

use crate::analysis_command_types::{VisualizeCommandOutput, VisualizeEvalError};

fn visualize_output_hint_lines(file_name: &str) -> Vec<String> {
    vec![
        format!("Render with: dot -Tsvg {file_name} -o ast.svg"),
        format!("Or: dot -Tpng {file_name} -o ast.png"),
    ]
}

/// Evaluate visualize command input and return DOT graph source.
pub fn evaluate_visualize_command_dot(
    ctx: &mut Context,
    input: &str,
) -> Result<String, VisualizeEvalError> {
    let parsed_expr = cas_parser::parse(input.trim(), ctx)
        .map_err(|e| VisualizeEvalError::Parse(e.to_string()))?;
    let mut viz = cas_solver::visualizer::AstVisualizer::new(ctx);
    Ok(viz.to_dot(parsed_expr))
}

/// Evaluate visualize command input and return session-level output payload.
pub fn evaluate_visualize_command_output(
    ctx: &mut Context,
    input: &str,
) -> Result<VisualizeCommandOutput, VisualizeEvalError> {
    let file_name = "ast.dot";
    let dot_source = evaluate_visualize_command_dot(ctx, input)?;
    Ok(VisualizeCommandOutput {
        file_name: file_name.to_string(),
        dot_source,
        hint_lines: visualize_output_hint_lines(file_name),
    })
}

/// Evaluate full `visualize ...` invocation and return session-level output payload.
pub fn evaluate_visualize_invocation_output(
    ctx: &mut Context,
    line: &str,
) -> Result<VisualizeCommandOutput, String> {
    let input = crate::extract_visualize_command_tail(line);
    evaluate_visualize_command_output(ctx, input)
        .map_err(|error| crate::format_visualize_command_error_message(&error))
}
