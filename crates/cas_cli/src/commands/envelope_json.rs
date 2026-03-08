//! envelope-json subcommand handler.
//!
//! Delegates envelope generation to `cas_session` and only handles CLI I/O.

use clap::Args;

/// Arguments for envelope-json subcommand
#[derive(Args, Debug)]
pub struct EnvelopeJsonArgs {
    /// Expression to evaluate
    pub expr: String,

    /// Domain mode: strict, generic, assume
    #[arg(long, default_value = "generic")]
    pub domain: String,

    /// Value domain: real, complex
    #[arg(long, default_value = "real")]
    pub value_domain: String,
}

/// Run the envelope-json command
pub fn run(args: EnvelopeJsonArgs) {
    let output =
        cas_session::evaluate_envelope_json_command(&args.expr, &args.domain, &args.value_domain);
    println!("{}", output);
}
