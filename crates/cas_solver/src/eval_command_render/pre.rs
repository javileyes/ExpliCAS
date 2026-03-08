use crate::eval_command_types::{EvalDisplayMessage, EvalDisplayMessageKind};

pub(super) fn build_pre_messages(
    stored_entry_line: Option<String>,
    warning_lines: Vec<String>,
    requires_lines: Vec<String>,
) -> Vec<EvalDisplayMessage> {
    let mut pre_messages = Vec::new();
    if let Some(line) = stored_entry_line {
        pre_messages.push(EvalDisplayMessage {
            kind: EvalDisplayMessageKind::Output,
            text: line,
        });
    }
    pre_messages.extend(warning_lines.into_iter().map(|line| EvalDisplayMessage {
        kind: EvalDisplayMessageKind::Warn,
        text: line,
    }));
    pre_messages.extend(requires_lines.into_iter().map(|line| EvalDisplayMessage {
        kind: EvalDisplayMessageKind::Info,
        text: line,
    }));
    pre_messages
}
