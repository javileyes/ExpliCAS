//! envelope command handler.
//!
//! Delegates stateless envelope generation to `cas_solver` and only handles CLI I/O.

use super::output::CommandOutput;
use crate::cli_args::{DomainArg, ValueDomainArg};
use cas_api_models::{EvalDomainMode, EvalValueDomain};
use clap::Args;

/// Arguments for the envelope wire command.
#[derive(Args, Debug)]
pub struct EnvelopeArgs {
    /// Expression to evaluate
    pub expr: String,

    /// Domain mode: strict, generic, assume
    #[arg(long, value_enum, default_value_t = DomainArg::Generic)]
    pub domain: DomainArg,

    /// Value domain: real, complex
    #[arg(long, value_enum, default_value_t = ValueDomainArg::Real)]
    pub value_domain: ValueDomainArg,
}

pub(crate) fn render(args: EnvelopeArgs) -> CommandOutput {
    CommandOutput::from_stdout(cas_solver::wire::evaluate_envelope_wire_command(
        &args.expr,
        domain_mode(args.domain),
        value_domain(args.value_domain),
    ))
}

fn domain_mode(domain: DomainArg) -> EvalDomainMode {
    match domain {
        DomainArg::Strict => EvalDomainMode::Strict,
        DomainArg::Generic => EvalDomainMode::Generic,
        DomainArg::Assume => EvalDomainMode::Assume,
    }
}

fn value_domain(domain: ValueDomainArg) -> EvalValueDomain {
    match domain {
        ValueDomainArg::Real => EvalValueDomain::Real,
        ValueDomainArg::Complex => EvalValueDomain::Complex,
    }
}
