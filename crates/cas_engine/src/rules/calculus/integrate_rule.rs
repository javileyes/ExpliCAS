use crate::define_rule;
use crate::symbolic_calculus_call_support::try_extract_integrate_call;

use super::integration_derivative_cofactor_routes::polynomial_trig_reciprocal_derivative_root_gate_rewrite;
use super::integration_result_pipeline::standard_integration_rewrite;

define_rule!(IntegrateRule, "Symbolic Integration", |ctx, expr| {
    if let Some(definite_call) =
        crate::symbolic_calculus_call_support::try_extract_definite_integrate_call(ctx, expr)
    {
        return super::definite_integration::definite_integration_rewrite(ctx, &definite_call);
    }
    let call = try_extract_integrate_call(ctx, expr)?;
    if let Some(rewrite) = polynomial_trig_reciprocal_derivative_root_gate_rewrite(ctx, &call) {
        return Some(rewrite);
    }

    standard_integration_rewrite(ctx, &call)
});
