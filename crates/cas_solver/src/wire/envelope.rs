use super::stateless_eval::evaluate_prepared_stateless_request;
use crate::eval_input::build_prepared_eval_request_for_input;
use crate::{Engine, EvalOptions};
use cas_api_models::{EnvelopeEvalOptions, OutputEnvelope};

mod common;
mod success;
#[cfg(test)]
mod tests;
mod transparency;

use self::common::build_request_info;
use self::success::build_success_envelope;

/// Stateless envelope evaluation used by application wrappers.
fn eval_str_to_output_envelope(expr: &str, opts: &EnvelopeEvalOptions) -> OutputEnvelope {
    let mut engine = Engine::new();
    let mut eval_options = EvalOptions::default();
    eval_options.shared.semantics.domain_mode = match opts.domain {
        cas_api_models::EvalDomainMode::Strict => crate::DomainMode::Strict,
        cas_api_models::EvalDomainMode::Generic => crate::DomainMode::Generic,
        cas_api_models::EvalDomainMode::Assume => crate::DomainMode::Assume,
    };
    eval_options.shared.semantics.value_domain = match opts.value_domain {
        cas_api_models::EvalValueDomain::Complex => crate::ValueDomain::ComplexEnabled,
        cas_api_models::EvalValueDomain::Real => crate::ValueDomain::RealOnly,
    };

    let prepared =
        match build_prepared_eval_request_for_input(expr, &mut engine.simplifier.context, false) {
            Ok(request) => request,
            Err(e) => {
                return OutputEnvelope::eval_error(
                    build_request_info(expr, opts),
                    format!("Parse error: {}", e),
                );
            }
        };

    let output_view = match evaluate_prepared_stateless_request(&mut engine, eval_options, prepared)
    {
        Ok(view) => view,
        Err(e) => return OutputEnvelope::eval_error(build_request_info(expr, opts), e.to_string()),
    };

    build_success_envelope(expr, opts, &engine.simplifier.context, &output_view)
}

/// Stateless CLI helper for the envelope wire command.
pub fn evaluate_envelope_wire_command(
    expr: &str,
    domain: cas_api_models::EvalDomainMode,
    value_domain: cas_api_models::EvalValueDomain,
) -> String {
    let opts = EnvelopeEvalOptions {
        domain,
        value_domain,
    };
    eval_str_to_output_envelope(expr, &opts).to_json_pretty()
}
