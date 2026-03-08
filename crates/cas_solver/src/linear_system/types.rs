use cas_math::multipoly::PolyError;
use num_rational::BigRational;

/// Error type for linear system solving.
#[derive(Debug)]
pub enum LinearSystemError {
    NotLinear(String),
    /// System has infinitely many solutions (dependent equations).
    InfiniteSolutions,
    /// System has no solution (inconsistent equations).
    NoSolution,
    PolyConversion(PolyError),
}

impl std::fmt::Display for LinearSystemError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LinearSystemError::NotLinear(msg) => write!(f, "non-linear term: {}", msg),
            LinearSystemError::InfiniteSolutions => {
                write!(
                    f,
                    "system has infinitely many solutions (dependent equations)"
                )
            }
            LinearSystemError::NoSolution => {
                write!(f, "system has no solution (inconsistent equations)")
            }
            LinearSystemError::PolyConversion(e) => write!(f, "polynomial conversion: {}", e),
        }
    }
}

/// Result of solving a linear system.
#[derive(Debug)]
pub enum LinSolveResult {
    /// Unique solution: values for each variable in order.
    Unique(Vec<BigRational>),
    /// Infinitely many solutions (dependent equations).
    Infinite,
    /// No solution (inconsistent equations).
    Inconsistent,
}

pub(crate) fn with_equation_index(error: LinearSystemError, index: usize) -> LinearSystemError {
    match error {
        LinearSystemError::NotLinear(message) => {
            LinearSystemError::NotLinear(format!("equation {}: {}", index, message))
        }
        other => other,
    }
}
