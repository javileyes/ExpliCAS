pub mod algebra;
pub mod arithmetic;
pub mod calculus;
pub mod canonicalization;
pub mod complex;
pub mod constants;
pub mod exponents;
pub mod functions;
pub mod grouping;
pub mod hyperbolic;
pub mod infinity;
pub mod integration;
pub mod inv_trig_n_angle;
pub mod inverse_trig;
pub mod logarithms;
pub mod matrix_ops;
pub mod number_theory;
pub mod polynomial;
pub mod rational_canonicalization;
pub mod reciprocal_trig;
pub mod trig_canonicalization;
pub mod trig_inverse_expansion;
pub mod trigonometry;

#[cfg(test)]
mod arithmetic_tests;
#[cfg(test)]
mod canonicalization_tests;
#[cfg(test)]
mod complex_tests;
#[cfg(test)]
mod constants_tests;
#[cfg(test)]
mod functions_tests;
#[cfg(test)]
mod grouping_tests;
#[cfg(test)]
mod hyperbolic_tests;
#[cfg(test)]
mod infinity_tests;
#[cfg(test)]
mod integration_tests;
#[cfg(test)]
mod logarithms_tests;
#[cfg(test)]
mod matrix_ops_tests;
#[cfg(test)]
mod rational_canonicalization_tests;
