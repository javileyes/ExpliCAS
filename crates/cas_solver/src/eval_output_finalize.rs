//! Final assembly for typed eval outputs.

mod build;
mod dispatch;

pub(crate) type EvalOutputWire = cas_api_models::EvalWireOutput;
pub(crate) type EvalOutputWireBuild<'a> = cas_api_models::EvalOutputBuild<'a>;

pub(crate) use self::build::{build_eval_output, EvalOutputResultPayload};
pub(crate) use self::dispatch::finalize_eval_output;
