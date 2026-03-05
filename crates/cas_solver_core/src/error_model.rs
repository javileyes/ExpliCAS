//! Shared CAS error model.
//!
//! Runtime crates (`cas_engine`, `cas_solver`) can re-export these types while
//! keeping a single canonical error vocabulary.

use thiserror::Error;

/// Convenience alias for `Result<T, CasError>`.
pub type CasResult<T> = Result<T, CasError>;

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
    #[error("division by zero")]
    DivisionByZero,
    #[error("invalid matrix: {reason}")]
    InvalidMatrix { reason: String },
    #[error("conversion failed: {from} -> {to}")]
    ConversionFailed { from: String, to: String },
    /// Budget exceeded error with full context.
    #[error("{0}")]
    BudgetExceeded(crate::budget_model::BudgetExceeded),
    #[error("not implemented: {feature}")]
    NotImplemented { feature: String },
    /// Invariant violation / internal bug.
    #[error("internal error: {0}")]
    InternalError(String),
    /// Operation not supported in real-only value domain.
    #[error("not supported in real domain: {0}")]
    UnsupportedInRealDomain(String),
    /// Expression didn't match expected shape.
    #[error("expression error: {0}")]
    ExpressionError(String),
    /// Numeric conversion overflow (e.g. BigInt -> i64).
    #[error("numeric overflow: {0}")]
    NumericOverflow(String),
}

impl From<crate::budget_model::BudgetExceeded> for CasError {
    fn from(e: crate::budget_model::BudgetExceeded) -> Self {
        CasError::BudgetExceeded(e)
    }
}

impl From<cas_math::multipoly::PolyError> for CasError {
    fn from(e: cas_math::multipoly::PolyError) -> Self {
        match e {
            cas_math::multipoly::PolyError::BudgetExceeded => {
                CasError::BudgetExceeded(crate::budget_model::BudgetExceeded {
                    op: crate::budget_model::Operation::PolyOps,
                    metric: crate::budget_model::Metric::TermsMaterialized,
                    used: 0,
                    limit: 0,
                })
            }
            _ => CasError::PolynomialError(e.to_string()),
        }
    }
}

impl crate::solve_analysis::StrategyErrorMessageParts for CasError {
    fn isolation_detail(&self) -> Option<&str> {
        match self {
            CasError::IsolationError(_, detail) => Some(detail.as_str()),
            _ => None,
        }
    }

    fn solver_detail(&self) -> Option<&str> {
        match self {
            CasError::SolverError(detail) => Some(detail.as_str()),
            _ => None,
        }
    }
}

impl CasError {
    /// Create a BudgetExceeded error from operation and metric with default usage info.
    pub fn budget_exceeded(
        op: crate::budget_model::Operation,
        metric: crate::budget_model::Metric,
    ) -> Self {
        CasError::BudgetExceeded(crate::budget_model::BudgetExceeded {
            op,
            metric,
            used: 0,
            limit: 0,
        })
    }

    /// Stable error kind for JSON/UI routing.
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
            CasError::UnsupportedInRealDomain(_) => "DomainError",
            CasError::ExpressionError(_) => "DomainError",
            CasError::NumericOverflow(_) => "DomainError",
        }
    }

    /// Stable error code for UI mapping.
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
            CasError::UnsupportedInRealDomain(_) => "E_REAL_ONLY",
            CasError::ExpressionError(_) => "E_EXPR",
            CasError::NumericOverflow(_) => "E_OVERFLOW",
        }
    }

    /// Get budget details if this is a BudgetExceeded error.
    pub fn budget_details(&self) -> Option<&crate::budget_model::BudgetExceeded> {
        match self {
            CasError::BudgetExceeded(b) => Some(b),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::CasError;

    #[test]
    fn budget_exceeded_helper_sets_expected_fields() {
        let err = CasError::budget_exceeded(
            crate::budget_model::Operation::Expand,
            crate::budget_model::Metric::NodesCreated,
        );
        let details = err.budget_details().expect("budget details");
        assert_eq!(details.op, crate::budget_model::Operation::Expand);
        assert_eq!(details.metric, crate::budget_model::Metric::NodesCreated);
        assert_eq!(details.used, 0);
        assert_eq!(details.limit, 0);
    }

    #[test]
    fn kind_and_code_are_stable_for_parse_error() {
        let err = CasError::ParseError("bad".to_string());
        assert_eq!(err.kind(), "ParseError");
        assert_eq!(err.code(), "E_PARSE");
    }
}
