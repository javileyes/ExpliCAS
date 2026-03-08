mod header;
mod step_lines;

use crate::substitute_command_types::{SubstituteEvalOutput, SubstituteRenderMode};

/// Format substitute command eval output into display lines.
pub fn format_substitute_eval_lines(
    context: &cas_ast::Context,
    input: &str,
    output: &SubstituteEvalOutput,
    mode: SubstituteRenderMode,
) -> Vec<String> {
    let mut lines = header::build_header_lines(input, output, mode);
    step_lines::append_step_lines(&mut lines, context, output, mode);
    lines.push(format!(
        "Result: {}",
        cas_formatter::DisplayExpr {
            context,
            id: output.simplified_expr
        }
    ));
    lines
}
