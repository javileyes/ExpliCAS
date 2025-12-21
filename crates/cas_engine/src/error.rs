use thiserror::Error;

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

    #[error("budget exceeded during {operation}")]
    BudgetExceeded { operation: String },

    #[error("not implemented: {feature}")]
    NotImplemented { feature: String },

    /// For invariant violations in release builds
    #[error("internal error: {0}")]
    InternalError(String),
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
