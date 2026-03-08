use cas_math::multipoly::MultiPoly;
use num_rational::BigRational;
use num_traits::Zero;

use super::super::super::super::types::LinearSystemError;
use super::super::super::shared::{non_linear_degree_error, unexpected_variable_error};

pub(super) fn extract_linear_terms(
    poly: &MultiPoly,
    var_indices: &[Option<usize>],
    arity: usize,
) -> Result<(Vec<BigRational>, BigRational), LinearSystemError> {
    let mut coeffs = vec![BigRational::zero(); arity];
    let mut constant = BigRational::zero();

    for (coef, mono) in &poly.terms {
        let total_exp: u32 = mono.iter().sum();

        if total_exp == 0 {
            constant = &constant + coef;
            continue;
        }

        if total_exp != 1 {
            return Err(non_linear_degree_error(total_exp));
        }

        accumulate_linear_term(poly, var_indices, &mut coeffs, coef, mono)?;
    }

    Ok((coeffs, -constant))
}

fn accumulate_linear_term(
    poly: &MultiPoly,
    var_indices: &[Option<usize>],
    coeffs: &mut [BigRational],
    coef: &BigRational,
    mono: &[u32],
) -> Result<(), LinearSystemError> {
    for (mono_idx, &exp) in mono.iter().enumerate() {
        if exp != 1 {
            continue;
        }

        for (var_idx, opt_idx) in var_indices.iter().enumerate() {
            if *opt_idx == Some(mono_idx) {
                coeffs[var_idx] = &coeffs[var_idx] + coef;
                return Ok(());
            }
        }

        return Err(unexpected_variable_error(&poly.vars[mono_idx]));
    }

    Ok(())
}
