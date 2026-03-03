use thiserror::Error;

use crate::budget;

/// Convenience alias for `Result<T, CasError>`.
///
/// Use this throughout the engine to avoid repeating the error type.
/// ```ignore
/// fn my_function() -> CasResult<ExprId> { ... }
/// ```
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

    /// Operation not supported in RealOnly value domain
    /// Includes suggestion to enable complex mode
    #[error("not supported in real domain: {0}")]
    UnsupportedInRealDomain(String),

    /// Expression didn't match expected shape (e.g., expected Add but got Mul).
    /// Used to replace unwrap() on pattern-matching in rules and transforms.
    #[error("expression error: {0}")]
    ExpressionError(String),

    /// Numeric conversion overflow (e.g., BigInt → i64 narrowing failed).
    /// Used to replace unwrap() on `.to_i64()`, `.to_u32()`, etc.
    #[error("numeric overflow: {0}")]
    NumericOverflow(String),
}

impl From<budget::BudgetExceeded> for CasError {
    fn from(e: budget::BudgetExceeded) -> Self {
        CasError::BudgetExceeded(e)
    }
}

impl From<cas_math::multipoly::PolyError> for CasError {
    fn from(e: cas_math::multipoly::PolyError) -> Self {
        match e {
            cas_math::multipoly::PolyError::BudgetExceeded => {
                CasError::BudgetExceeded(budget::BudgetExceeded {
                    op: budget::Operation::PolyOps,
                    metric: budget::Metric::TermsMaterialized,
                    used: 0,
                    limit: 0,
                })
            }
            _ => CasError::PolynomialError(e.to_string()),
        }
    }
}

impl cas_solver_core::solve_analysis::StrategyErrorMessageParts for CasError {
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
            CasError::UnsupportedInRealDomain(_) => "DomainError",
            CasError::ExpressionError(_) => "DomainError",
            CasError::NumericOverflow(_) => "DomainError",
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
            CasError::UnsupportedInRealDomain(_) => "E_REAL_ONLY",
            CasError::ExpressionError(_) => "E_EXPR",
            CasError::NumericOverflow(_) => "E_OVERFLOW",
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
