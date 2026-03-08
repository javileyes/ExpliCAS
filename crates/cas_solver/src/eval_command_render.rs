mod post;
mod pre;
mod result;

use crate::eval_command_types::{EvalCommandOutput, EvalCommandRenderPlan};

/// Convert an eval output payload into an ordered rendering plan.
pub fn build_eval_command_render_plan(
    output: EvalCommandOutput,
    verbosity_is_none: bool,
) -> EvalCommandRenderPlan {
    let crate::eval_command_types::EvalCommandOutput {
        resolved_expr,
        style_signals,
        steps,
        stored_entry_line,
        metadata,
        result_line,
    } = output;
    let render_steps = !steps.is_empty() || !verbosity_is_none;
    let pre_messages = pre::build_pre_messages(
        stored_entry_line,
        metadata.warning_lines,
        metadata.requires_lines,
    );
    let (result_message, result_terminal) = result::build_result_message(result_line);
    let post_messages = post::build_post_messages(metadata.hint_lines, metadata.assumption_lines);

    EvalCommandRenderPlan {
        pre_messages,
        render_steps,
        resolved_expr,
        style_signals,
        steps,
        result_message,
        result_terminal,
        post_messages,
    }
}
