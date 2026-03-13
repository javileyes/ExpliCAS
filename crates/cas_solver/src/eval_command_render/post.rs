use crate::command_api::eval::{EvalDisplayMessage, EvalDisplayMessageKind};

pub(super) fn build_post_messages(
    hint_lines: Vec<String>,
    assumption_lines: Vec<String>,
) -> Vec<EvalDisplayMessage> {
    let mut post_messages = Vec::new();
    post_messages.extend(hint_lines.into_iter().map(|line| EvalDisplayMessage {
        kind: EvalDisplayMessageKind::Info,
        text: line,
    }));
    post_messages.extend(assumption_lines.into_iter().map(|line| EvalDisplayMessage {
        kind: EvalDisplayMessageKind::Info,
        text: line,
    }));
    post_messages
}
