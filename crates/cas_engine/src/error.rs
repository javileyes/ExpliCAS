use thiserror::Error;

use crate::budget;

#[derive(Error, Debug)]
pub enum CasError {
    #[error("Variable '{0}' not found")]
    VariableNotFound(String),
    #[error("Cannot isolate '{0}': {1}")]
    IsolationError(String, String),
    #[error("Unknown function '{0}'")]
    UnknownFunction(String),
    #[error("Solver error: {0}")]
    SolverError(String),
    #[error("Parse error: {0}")]
    ParseError(String),
    #[error("Polynomial error: {0}")]
    PolynomialError(String),

    // New error variants for panic removal
    #[error("division by zero")]
    DivisionByZero,

    #[error("invalid matrix: {reason}")]
    InvalidMatrix { reason: String },

    #[error("conversion failed: {from} -> {to}")]
    ConversionFailed { from: String, to: String },

    /// Budget exceeded error with full context.
    /// Wraps the unified `budget::BudgetExceeded` struct.
    #[error("{0}")]
    BudgetExceeded(budget::BudgetExceeded),

    #[error("not implemented: {feature}")]
    NotImplemented { feature: String },

    /// For invariant violations in release builds
    #[error("internal error: {0}")]
    InternalError(String),
}

impl From<budget::BudgetExceeded> for CasError {
    fn from(e: budget::BudgetExceeded) -> Self {
        CasError::BudgetExceeded(e)
    }
}

impl CasError {
    /// Create a BudgetExceeded error from operation and metric with default usage info.
    /// Convenience for migration from old code.
    pub fn budget_exceeded(op: budget::Operation, metric: budget::Metric) -> Self {
        CasError::BudgetExceeded(budget::BudgetExceeded {
            op,
            metric,
            used: 0,
            limit: 0,
        })
    }
}

/// Helper macro for invariant assertions.
/// In debug: uses debug_assert!
/// In release: returns Err(CasError::InternalError) if condition fails.
#[macro_export]
macro_rules! ensure_invariant {
    ($cond:expr, $msg:literal $(, $args:expr)* $(,)?) => {
        if cfg!(debug_assertions) {
            debug_assert!($cond, $msg $(, $args)*);
        }
        if !$cond {
            return Err($crate::error::CasError::InternalError(format!($msg $(, $args)*)));
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::budget::{BudgetExceeded, Metric, Operation};

    #[test]
    fn test_budget_exceeded_conversion() {
        let budget_err = BudgetExceeded {
            op: Operation::Expand,
            metric: Metric::TermsMaterialized,
            used: 150,
            limit: 100,
        };

        let cas_err: CasError = budget_err.clone().into();

        // Verify it's wrapped correctly
        match cas_err {
            CasError::BudgetExceeded(inner) => {
                assert_eq!(inner.op, Operation::Expand);
                assert_eq!(inner.metric, Metric::TermsMaterialized);
                assert_eq!(inner.used, 150);
                assert_eq!(inner.limit, 100);
            }
            _ => panic!("Expected BudgetExceeded variant"),
        }
    }

    #[test]
    fn test_budget_exceeded_display() {
        let budget_err = BudgetExceeded {
            op: Operation::GcdZippel,
            metric: Metric::PolyOps,
            used: 500,
            limit: 300,
        };

        let cas_err: CasError = budget_err.into();
        let msg = format!("{}", cas_err);

        assert!(msg.contains("GcdZippel"), "Missing op in: {}", msg);
        assert!(msg.contains("PolyOps"), "Missing metric in: {}", msg);
        assert!(msg.contains("500"), "Missing used in: {}", msg);
        assert!(msg.contains("300"), "Missing limit in: {}", msg);
    }

    #[test]
    fn test_budget_exceeded_helper() {
        let err = CasError::budget_exceeded(Operation::SimplifyCore, Metric::RewriteSteps);

        match err {
            CasError::BudgetExceeded(inner) => {
                assert_eq!(inner.op, Operation::SimplifyCore);
                assert_eq!(inner.metric, Metric::RewriteSteps);
            }
            _ => panic!("Expected BudgetExceeded variant"),
        }
    }

    #[test]
    fn test_poly_error_to_cas_error() {
        use crate::multipoly::PolyError;

        let poly_err = PolyError::BudgetExceeded;
        let cas_err: CasError = poly_err.into();

        match cas_err {
            CasError::BudgetExceeded(inner) => {
                assert_eq!(inner.op, Operation::PolyOps);
            }
            _ => panic!("Expected BudgetExceeded, got: {:?}", cas_err),
        }
    }
}
