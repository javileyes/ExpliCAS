use super::super::gcd_factorization::generate_gcd_factorization_substeps;
use super::super::nested_fractions::generate_nested_fraction_substeps;
use super::super::polynomial_identity::generate_polynomial_identity_substeps;
use super::super::rationalization::generate_rationalization_substeps;
use super::super::root_denesting::generate_root_denesting_substeps;
use super::super::sum_three_cubes::generate_sum_three_cubes_substeps;
use super::super::SubStep;
use crate::cas_solver::Step;
use cas_ast::Context;

pub(super) fn extend_step_specific_substeps(
    ctx: &Context,
    step: &Step,
    sub_steps: &mut Vec<SubStep>,
) {
    if step.description.starts_with("Simplified fraction by GCD") && !step.is_chained() {
        sub_steps.extend(generate_gcd_factorization_substeps(ctx, step));
    }

    if is_nested_fraction_step(step) {
        sub_steps.extend(generate_nested_fraction_substeps(ctx, step));
    }

    if step.description.contains("Rationalize") || step.rule_name.contains("Rationalize") {
        sub_steps.extend(generate_rationalization_substeps(ctx, step));
    }

    if step.poly_proof().is_some() {
        sub_steps.extend(generate_polynomial_identity_substeps(ctx, step));
    }

    if step.rule_name.contains("Sum of Three Cubes") {
        sub_steps.extend(generate_sum_three_cubes_substeps(ctx, step));
    }

    if step.rule_name.contains("Root Denesting") {
        sub_steps.extend(generate_root_denesting_substeps(ctx, step));
    }
}

fn is_nested_fraction_step(step: &Step) -> bool {
    step.rule_name.to_lowercase().contains("complex fraction")
        || step.rule_name.to_lowercase().contains("nested fraction")
        || step.description.to_lowercase().contains("nested fraction")
}
