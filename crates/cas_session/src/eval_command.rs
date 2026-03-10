//! Session-backed eval command orchestration.

mod pretty;
mod session;

pub use pretty::evaluate_eval_command_pretty_with_session;
pub use session::{evaluate_eval_command_with_session, EvalCommandConfig};
