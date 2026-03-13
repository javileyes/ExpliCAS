use cas_ast::{Context, ExprId};
use cas_math::multipoly::PolyBudget;
use num_rational::BigRational;
use num_traits::Zero;

use super::super::super::LinearSystemError;
use super::super::shared::{build_linear_poly, non_linear_degree_error, unexpected_variable_error};
use super::LinearCoeffs;

pub(super) fn extract_linear_coeffs(
    ctx: &Context,
    expr: ExprId,
    var_x: &str,
    var_y: &str,
) -> Result<LinearCoeffs, LinearSystemError> {
    let poly = build_linear_poly(
        ctx,
        expr,
        PolyBudget {
            max_terms: 100,
            max_total_degree: 2,
            max_pow_exp: 2,
        },
    )?;

    let idx_x = poly.vars.iter().position(|v| v == var_x);
    let idx_y = poly.vars.iter().position(|v| v == var_y);
    let num_vars = poly.vars.len();

    let mut a = BigRational::zero();
    let mut b = BigRational::zero();
    let mut c = BigRational::zero();

    for (coef, mono) in &poly.terms {
        let total_exp: u32 = mono.iter().sum();

        if total_exp == 0 {
            c = &c + coef;
        } else if total_exp == 1 {
            let mut found = false;
            for (i, &exp) in mono.iter().enumerate() {
                if exp == 1 {
                    if Some(i) == idx_x {
                        a = &a + coef;
                        found = true;
                    } else if Some(i) == idx_y {
                        b = &b + coef;
                        found = true;
                    } else {
                        return Err(unexpected_variable_error(&poly.vars[i]));
                    }
                }
            }
            if !found && num_vars == 0 {
                c = &c + coef;
            }
        } else {
            return Err(non_linear_degree_error(total_exp));
        }
    }

    Ok(LinearCoeffs { a, b, c })
}
