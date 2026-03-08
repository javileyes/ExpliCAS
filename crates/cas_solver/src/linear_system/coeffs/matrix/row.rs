mod terms;

use cas_ast::{Context, ExprId};
use cas_math::multipoly::PolyBudget;

use super::super::super::types::LinearSystemError;
use super::super::shared::build_linear_poly;

pub(super) fn extract_linear_row(
    ctx: &Context,
    expr: ExprId,
    vars: &[&str],
) -> Result<(Vec<num_rational::BigRational>, num_rational::BigRational), LinearSystemError> {
    let poly = build_linear_poly(
        ctx,
        expr,
        PolyBudget {
            max_terms: 200,
            max_total_degree: 2,
            max_pow_exp: 2,
        },
    )?;

    let var_indices: Vec<Option<usize>> = vars
        .iter()
        .map(|v| poly.vars.iter().position(|pv| pv == *v))
        .collect();

    terms::extract_linear_terms(&poly, &var_indices, vars.len())
}
