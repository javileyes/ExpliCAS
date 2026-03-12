//! Public session-owned API for stateless/stateful eval command execution.

pub use crate::eval_command::{
    evaluate_eval_command_pretty_with_session, evaluate_eval_command_with_session,
    EvalCommandConfig,
};
pub use crate::eval_text_command::evaluate_eval_text_command_with_session;
pub use cas_solver_core::eval_display_types::{
    EvalDisplayMessage, EvalDisplayMessageKind, EvalMetadataLines, EvalResultLine,
};
