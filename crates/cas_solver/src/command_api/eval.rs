//! Eval command entrypoints exposed for CLI/frontends.

#[path = "../eval_command_types/error.rs"]
mod error;
#[path = "../eval_command_types/output.rs"]
mod output;
#[path = "../eval_command_types/render_plan.rs"]
mod render_plan;
#[path = "../eval_command_types/view.rs"]
mod view;

pub use crate::eval_command_eval::evaluate_eval_command_output;
pub use crate::eval_command_render::build_eval_command_render_plan;
pub use crate::eval_command_text::evaluate_eval_text_simplify_with_session;
pub use cas_solver_core::eval_display_types::{
    EvalDisplayMessage, EvalDisplayMessageKind, EvalMetadataLines, EvalResultLine,
};
pub use error::EvalCommandError;
pub use output::EvalCommandOutput;
pub use render_plan::EvalCommandRenderPlan;
pub(crate) use view::EvalCommandEvalView;
