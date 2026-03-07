//! Final assembly for session-backed `eval-json` outputs.

mod build;
mod dispatch;

pub(crate) use self::build::{build_eval_json_output, EvalJsonResultPayload};
pub(crate) use self::dispatch::finalize_eval_json_output;
