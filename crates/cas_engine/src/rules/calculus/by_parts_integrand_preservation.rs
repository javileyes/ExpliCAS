use super::exp_by_parts_integrand_presentation::linear_exp_by_parts_integrand_for_calculus_presentation;
use super::hyperbolic_by_parts_integrand_presentation::{
    linear_hyperbolic_by_parts_integrand_for_calculus_presentation,
    repeated_hyperbolic_by_parts_integrand_for_calculus_presentation,
};
use super::log_by_parts_integrand_presentation::log_by_parts_integrand_for_calculus_presentation;
use super::trig_by_parts_integrand_presentation::{
    linear_trig_by_parts_integrand_for_calculus_presentation,
    repeated_trig_by_parts_integrand_for_calculus_presentation,
};
use cas_ast::{Context, ExprId};

pub(super) struct ByPartsIntegrandPreservation {
    active: bool,
    pub(super) preserve_compact_log_by_parts: bool,
}

impl ByPartsIntegrandPreservation {
    pub(super) fn should_preserve_compact_result(&self) -> bool {
        self.active
    }
}

pub(super) fn by_parts_integrand_preservation_gates(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> ByPartsIntegrandPreservation {
    let preserve_compact_linear_exp_by_parts =
        linear_exp_by_parts_integrand_for_calculus_presentation(ctx, target, var_name);
    let preserve_compact_linear_trig_by_parts =
        linear_trig_by_parts_integrand_for_calculus_presentation(ctx, target, var_name);
    let preserve_compact_linear_hyperbolic_by_parts =
        linear_hyperbolic_by_parts_integrand_for_calculus_presentation(ctx, target, var_name);
    let preserve_compact_repeated_trig_by_parts =
        repeated_trig_by_parts_integrand_for_calculus_presentation(ctx, target, var_name);
    let preserve_compact_repeated_hyperbolic_by_parts =
        repeated_hyperbolic_by_parts_integrand_for_calculus_presentation(ctx, target, var_name);
    let preserve_compact_log_by_parts =
        log_by_parts_integrand_for_calculus_presentation(ctx, target, var_name);

    let active = preserve_compact_linear_exp_by_parts
        || preserve_compact_linear_trig_by_parts
        || preserve_compact_linear_hyperbolic_by_parts
        || preserve_compact_repeated_trig_by_parts
        || preserve_compact_repeated_hyperbolic_by_parts
        || preserve_compact_log_by_parts;

    ByPartsIntegrandPreservation {
        active,
        preserve_compact_log_by_parts,
    }
}
