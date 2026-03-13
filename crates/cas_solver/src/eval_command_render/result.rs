use crate::command_api::eval::{EvalDisplayMessage, EvalDisplayMessageKind, EvalResultLine};

pub(super) fn build_result_message(
    result_line: Option<EvalResultLine>,
) -> (Option<EvalDisplayMessage>, bool) {
    match result_line {
        Some(result) => (
            Some(EvalDisplayMessage {
                kind: EvalDisplayMessageKind::Output,
                text: result.line,
            }),
            result.terminal,
        ),
        None => (None, false),
    }
}
