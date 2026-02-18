//! Compatibility adapter for exact sparse multivariate polynomials over Q.
//!
//! Canonical implementation lives in `cas_math::multipoly`.

use crate::budget::{BudgetExceeded, Metric, Operation};
use crate::error::CasError;

pub mod arithmetic;
pub mod gcd;

pub use cas_math::multipoly::conversion::{
    collect_poly_vars, multipoly_from_expr, multipoly_to_expr,
};
pub use cas_math::multipoly::{
    Exp, GcdBudget, GcdLayer, Layer25Budget, Monomial, MultiPoly, PolyBudget, PolyError,
    PolyOperation, PolyPassStats, Term, VarIdx,
};
pub use gcd::{gcd_multivar_layer2, gcd_multivar_layer25, gcd_multivar_layer2_with_stats};

impl From<PolyError> for CasError {
    fn from(e: PolyError) -> Self {
        match e {
            PolyError::BudgetExceeded => CasError::BudgetExceeded(BudgetExceeded {
                op: Operation::PolyOps,
                metric: Metric::TermsMaterialized,
                used: 0,
                limit: 0,
            }),
            _ => CasError::PolynomialError(e.to_string()),
        }
    }
}
