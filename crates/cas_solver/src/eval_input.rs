//! Input parsing and typed request-building helpers for eval orchestration.

mod build;
mod types;

pub use build::build_prepared_eval_request_for_input;
pub use types::{EvalNonSolveAction, PreparedEvalRequest};
