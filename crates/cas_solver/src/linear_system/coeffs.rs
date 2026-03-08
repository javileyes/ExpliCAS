mod matrix;
mod shared;
mod three_var;
mod two_var;

pub(crate) use matrix::build_augmented_matrix;
pub(crate) use three_var::{extract_linear_coeffs_3, LinearCoeffs3};
pub(crate) use two_var::{extract_linear_coeffs, LinearCoeffs};
