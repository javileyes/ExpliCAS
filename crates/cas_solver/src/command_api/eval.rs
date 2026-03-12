//! Eval command entrypoints exposed for CLI/frontends.

pub use crate::eval_command_eval::evaluate_eval_command_output;
pub use crate::eval_command_render::build_eval_command_render_plan;
pub use crate::eval_command_text::evaluate_eval_text_simplify_with_session;
pub use crate::eval_command_types::{
    EvalCommandError, EvalCommandOutput, EvalCommandRenderPlan, EvalDisplayMessage,
    EvalDisplayMessageKind, EvalMetadataLines, EvalResultLine,
};
