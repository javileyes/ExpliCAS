mod assumptions;
mod conditions;
mod hints;

use crate::{AssumptionRecord, BlockedHint, DomainWarning, ImplicitCondition};
use cas_api_models::TransparencyDto;

pub struct TransparencyInput<'a> {
    pub required_conditions: &'a [ImplicitCondition],
    pub solver_assumptions: &'a [AssumptionRecord],
    pub domain_warnings: &'a [DomainWarning],
    pub steps: &'a [crate::Step],
    pub blocked_hints: &'a [BlockedHint],
    pub ctx: &'a cas_ast::Context,
    pub raw_input: &'a str,
    pub result_display: Option<&'a str>,
}

pub fn build_transparency(input: TransparencyInput<'_>) -> TransparencyDto {
    let assumptions_used = assumptions::map_assumptions_used(
        input.solver_assumptions,
        input.domain_warnings,
        input.steps,
    );
    TransparencyDto {
        required_conditions: conditions::map_required_conditions(
            input.required_conditions,
            input.ctx,
            &assumptions_used,
            input.raw_input,
            input.result_display,
        ),
        assumptions_used,
        blocked_hints: hints::map_blocked_hints(input.ctx, input.blocked_hints),
    }
}
