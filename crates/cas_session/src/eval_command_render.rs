use crate::eval_command_types::{
    EvalCommandOutput, EvalCommandRenderPlan, EvalDisplayMessage, EvalDisplayMessageKind,
};

/// Convert an eval output payload into an ordered rendering plan.
pub fn build_eval_command_render_plan(
    output: EvalCommandOutput,
    verbosity_is_none: bool,
) -> EvalCommandRenderPlan {
    let mut pre_messages = Vec::new();
    if let Some(line) = output.stored_entry_line {
        pre_messages.push(EvalDisplayMessage {
            kind: EvalDisplayMessageKind::Output,
            text: line,
        });
    }
    pre_messages.extend(
        output
            .metadata
            .warning_lines
            .into_iter()
            .map(|line| EvalDisplayMessage {
                kind: EvalDisplayMessageKind::Warn,
                text: line,
            }),
    );
    pre_messages.extend(output.metadata.requires_lines.into_iter().map(|line| {
        EvalDisplayMessage {
            kind: EvalDisplayMessageKind::Info,
            text: line,
        }
    }));

    let render_steps = !output.steps.is_empty() || !verbosity_is_none;

    let (result_message, result_terminal) = match output.result_line {
        Some(result) => (
            Some(EvalDisplayMessage {
                kind: EvalDisplayMessageKind::Output,
                text: result.line,
            }),
            result.terminal,
        ),
        None => (None, false),
    };

    let mut post_messages = Vec::new();
    post_messages.extend(
        output
            .metadata
            .hint_lines
            .into_iter()
            .map(|line| EvalDisplayMessage {
                kind: EvalDisplayMessageKind::Info,
                text: line,
            }),
    );
    post_messages.extend(output.metadata.assumption_lines.into_iter().map(|line| {
        EvalDisplayMessage {
            kind: EvalDisplayMessageKind::Info,
            text: line,
        }
    }));

    EvalCommandRenderPlan {
        pre_messages,
        render_steps,
        resolved_expr: output.resolved_expr,
        style_signals: output.style_signals,
        steps: output.steps,
        result_message,
        result_terminal,
        post_messages,
    }
}
