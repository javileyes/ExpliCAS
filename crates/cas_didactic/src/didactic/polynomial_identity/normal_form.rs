mod comparison;
mod conversion;

use super::PolynomialProofData;
use crate::cas_solver::Step;
use crate::didactic::SubStep;
use cas_ast::Context;

pub(super) fn generate_polynomial_identity_normal_form_substeps(
    ctx: &Context,
    step: &Step,
    proof: &PolynomialProofData,
) -> Vec<SubStep> {
    if let (Some(lhs_stats), Some(rhs_stats)) = (&proof.lhs_stats, &proof.rhs_stats) {
        return comparison::generate_identity_comparison_substeps(ctx, lhs_stats, rhs_stats);
    }

    conversion::generate_normal_form_conversion_substeps(ctx, step, proof)
}
