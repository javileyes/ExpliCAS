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

impl From<cas_parser::ParseError> for CasError {
    fn from(e: cas_parser::ParseError) -> Self {
        CasError::ParseError(e.to_string())
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

    // =========================================================================
    // Stable Error API (kind/code/span)
    // See POLICY.md "Error API Stability Contract"
    // =========================================================================

    /// Stable error kind for JSON/UI routing.
    ///
    /// # Stability Contract
    ///
    /// These values are **stable** and will not change between minor versions:
    /// - `ParseError` - Input parsing failed
    /// - `DomainError` - Mathematical domain violation
    /// - `SolverError` - Equation solving failed
    /// - `BudgetExceeded` - Resource limit hit
    /// - `NotImplemented` - Feature not available
    /// - `InternalError` - Bug in the engine
    pub fn kind(&self) -> &'static str {
        match self {
            CasError::ParseError(_) => "ParseError",
            CasError::VariableNotFound(_) => "DomainError",
            CasError::IsolationError(_, _) => "SolverError",
            CasError::UnknownFunction(_) => "DomainError",
            CasError::SolverError(_) => "SolverError",
            CasError::PolynomialError(_) => "DomainError",
            CasError::DivisionByZero => "DomainError",
            CasError::InvalidMatrix { .. } => "DomainError",
            CasError::ConversionFailed { .. } => "DomainError",
            CasError::BudgetExceeded(_) => "BudgetExceeded",
            CasError::NotImplemented { .. } => "NotImplemented",
            CasError::InternalError(_) => "InternalError",
        }
    }

    /// Stable error code for UI mapping.
    ///
    /// # Stability Contract
    ///
    /// Codes start with `E_` and are **stable** between minor versions.
    pub fn code(&self) -> &'static str {
        match self {
            CasError::ParseError(_) => "E_PARSE",
            CasError::VariableNotFound(_) => "E_VAR_NOT_FOUND",
            CasError::IsolationError(_, _) => "E_ISOLATION",
            CasError::UnknownFunction(_) => "E_UNKNOWN_FUNC",
            CasError::SolverError(_) => "E_SOLVER",
            CasError::PolynomialError(_) => "E_POLYNOMIAL",
            CasError::DivisionByZero => "E_DIV_ZERO",
            CasError::InvalidMatrix { .. } => "E_MATRIX",
            CasError::ConversionFailed { .. } => "E_CONVERSION",
            CasError::BudgetExceeded(_) => "E_BUDGET",
            CasError::NotImplemented { .. } => "E_NOT_IMPL",
            CasError::InternalError(_) => "E_INTERNAL",
        }
    }

    /// Get budget details if this is a BudgetExceeded error.
    pub fn budget_details(&self) -> Option<&budget::BudgetExceeded> {
        match self {
            CasError::BudgetExceeded(b) => Some(b),
            _ => None,
        }
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

    // =========================================================================
    // Stability tests for kind/code API (Error API Stability Contract)
    // =========================================================================

    #[test]
    fn test_error_kind_stable() {
        // These kind values are STABLE and must not change
        assert_eq!(CasError::ParseError("x".into()).kind(), "ParseError");
        assert_eq!(CasError::DivisionByZero.kind(), "DomainError");
        assert_eq!(CasError::VariableNotFound("x".into()).kind(), "DomainError");
        assert_eq!(CasError::SolverError("x".into()).kind(), "SolverError");
        assert_eq!(
            CasError::NotImplemented {
                feature: "x".into()
            }
            .kind(),
            "NotImplemented"
        );
        assert_eq!(CasError::InternalError("x".into()).kind(), "InternalError");

        let budget_err = CasError::BudgetExceeded(BudgetExceeded {
            op: Operation::Expand,
            metric: Metric::TermsMaterialized,
            used: 0,
            limit: 0,
        });
        assert_eq!(budget_err.kind(), "BudgetExceeded");
    }

    #[test]
    fn test_error_code_stable() {
        // These code values are STABLE and must not change
        assert_eq!(CasError::ParseError("x".into()).code(), "E_PARSE");
        assert_eq!(CasError::DivisionByZero.code(), "E_DIV_ZERO");
        assert_eq!(
            CasError::VariableNotFound("x".into()).code(),
            "E_VAR_NOT_FOUND"
        );
        assert_eq!(CasError::SolverError("x".into()).code(), "E_SOLVER");
        assert_eq!(
            CasError::NotImplemented {
                feature: "x".into()
            }
            .code(),
            "E_NOT_IMPL"
        );
        assert_eq!(CasError::InternalError("x".into()).code(), "E_INTERNAL");

        let budget_err = CasError::BudgetExceeded(BudgetExceeded {
            op: Operation::Expand,
            metric: Metric::TermsMaterialized,
            used: 0,
            limit: 0,
        });
        assert_eq!(budget_err.code(), "E_BUDGET");
    }

    #[test]
    fn test_error_code_prefix() {
        // All codes must start with E_
        let errors: Vec<CasError> = vec![
            CasError::ParseError("x".into()),
            CasError::DivisionByZero,
            CasError::VariableNotFound("x".into()),
            CasError::InternalError("x".into()),
        ];

        for e in errors {
            assert!(
                e.code().starts_with("E_"),
                "Code {} must start with E_",
                e.code()
            );
        }
    }

    #[test]
    fn test_error_kind_known_set() {
        // kind() must return one of the known values
        let valid_kinds = [
            "ParseError",
            "DomainError",
            "SolverError",
            "BudgetExceeded",
            "NotImplemented",
            "InternalError",
        ];

        let errors: Vec<CasError> = vec![
            CasError::ParseError("x".into()),
            CasError::DivisionByZero,
            CasError::VariableNotFound("x".into()),
            CasError::InternalError("x".into()),
        ];

        for e in errors {
            assert!(
                valid_kinds.contains(&e.kind()),
                "Unknown kind: {}",
                e.kind()
            );
        }
    }

    #[test]
    fn test_budget_details_accessor() {
        let budget_err = CasError::BudgetExceeded(BudgetExceeded {
            op: Operation::Expand,
            metric: Metric::TermsMaterialized,
            used: 100,
            limit: 50,
        });

        let details = budget_err.budget_details().unwrap();
        assert_eq!(details.used, 100);
        assert_eq!(details.limit, 50);

        // Non-budget errors should return None
        assert!(CasError::DivisionByZero.budget_details().is_none());
    }

    #[test]
    fn test_parse_error_conversion() {
        let parse_err = cas_parser::ParseError::syntax("unexpected token");
        let cas_err: CasError = parse_err.into();

        assert_eq!(cas_err.kind(), "ParseError");
        assert_eq!(cas_err.code(), "E_PARSE");
    }
}
