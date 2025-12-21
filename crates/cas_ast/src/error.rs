//! Error types for cas_ast crate.

use thiserror::Error;

/// Errors that can occur in AST operations.
#[derive(Error, Debug, Clone)]
pub enum AstError {
    /// Matrix dimensions don't match data length
    #[error("invalid matrix: {reason}")]
    InvalidMatrix { reason: String },

    /// Internal invariant violation
    #[error("internal error: {0}")]
    InternalError(String),
}

/// Helper macro for invariant assertions in cas_ast.
/// In debug: uses debug_assert!
/// In release: returns Err(AstError::InternalError) if condition fails.
#[macro_export]
macro_rules! ensure_ast_invariant {
    ($cond:expr, $msg:literal $(, $args:expr)* $(,)?) => {
        if cfg!(debug_assertions) {
            debug_assert!($cond, $msg $(, $args)*);
        }
        if !$cond {
            return Err($crate::error::AstError::InternalError(format!($msg $(, $args)*)));
        }
    };
}
