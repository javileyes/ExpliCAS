mod assumptions;
mod conditions;
mod hints;

use crate::{AssumptionRecord, BlockedHint, DomainWarning, ImplicitCondition};
use cas_api_models::TransparencyDto;

pub fn build_transparency(
    required_conditions: &[ImplicitCondition],
    solver_assumptions: &[AssumptionRecord],
    domain_warnings: &[DomainWarning],
    blocked_hints: &[BlockedHint],
    ctx: &cas_ast::Context,
) -> TransparencyDto {
    TransparencyDto {
        required_conditions: conditions::map_required_conditions(required_conditions, ctx),
        assumptions_used: assumptions::map_assumptions_used(solver_assumptions, domain_warnings),
        blocked_hints: hints::map_blocked_hints(blocked_hints),
    }
}
