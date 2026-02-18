//! envelope-json subcommand handler.
//!
//! Delegates envelope generation to `cas_solver` and only handles CLI I/O.

use clap::Args;

use cas_solver::{eval_str_to_output_envelope, EnvelopeEvalOptions};

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
    let output = eval_str_to_output_envelope(
        &args.expr,
        &EnvelopeEvalOptions {
            domain: args.domain,
            value_domain: args.value_domain,
        },
    );
    println!("{}", output.to_json_pretty());
}
