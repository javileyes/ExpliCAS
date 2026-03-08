//! Input parsing and request-building helpers for eval-json orchestration.

mod build;
mod types;

pub use build::build_eval_json_request_for_input;
pub use types::{EvalJsonNonSolveAction, EvalJsonPreparedRequest};
