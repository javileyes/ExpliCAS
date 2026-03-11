mod cancellation;
mod substituted;
mod substitution_intro;

use super::{display_expr, latex_expr, PolynomialProofData};
use crate::didactic::SubStep;
use crate::runtime::Step;
use cas_ast::Context;

pub(super) fn generate_polynomial_identity_substitution_substeps(
    ctx: &Context,
    step: &Step,
    proof: &PolynomialProofData,
) -> Vec<SubStep> {
    let mut sub_steps = Vec::new();

    sub_steps.push(substitution_intro::build_substitution_intro_substep(
        ctx,
        step,
        proof,
        display_expr,
        latex_expr,
    ));

    if let Some(display_id) = proof.normal_form_expr {
        sub_steps.push(substituted::build_substituted_expression_substep(
            ctx,
            proof,
            display_id,
            display_expr,
            latex_expr,
        ));
    }

    sub_steps.push(cancellation::build_cancellation_substep());

    sub_steps
}
