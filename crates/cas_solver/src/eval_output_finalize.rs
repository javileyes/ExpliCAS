//! Final assembly for typed eval outputs.

mod dispatch;
#[path = "eval_output_finalize/build/output.rs"]
mod output;
#[path = "eval_output_finalize/build/payload.rs"]
mod payload;

pub(crate) type EvalOutputWire = cas_api_models::EvalWireOutput;
pub(crate) type EvalOutputWireBuild<'a> = cas_api_models::EvalOutputBuild<'a>;

pub(crate) use self::dispatch::finalize_eval_output;
pub(crate) use self::output::build_eval_output;
pub(crate) use self::payload::EvalOutputResultPayload;
