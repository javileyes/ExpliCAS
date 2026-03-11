//! envelope command handler.
//!
//! Delegates envelope generation to `cas_session` and only handles CLI I/O.

use clap::Args;

/// Arguments for the envelope wire command.
#[derive(Args, Debug)]
pub struct EnvelopeArgs {
    /// Expression to evaluate
    pub expr: String,

    /// Domain mode: strict, generic, assume
    #[arg(long, default_value = "generic")]
    pub domain: String,

    /// Value domain: real, complex
    #[arg(long, default_value = "real")]
    pub value_domain: String,
}

/// Run the envelope command.
pub fn run(args: EnvelopeArgs) {
    let output =
        cas_session::evaluate_envelope_wire_command(&args.expr, &args.domain, &args.value_domain);
    println!("{}", output);
}
