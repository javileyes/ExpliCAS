//! Backward-compatible engine error module.
//!
//! Canonical error definitions now live in `cas_solver_core::error_model`.
//! `cas_engine::error` remains as a compatibility facade.

pub use cas_solver_core::error_model::{CasError, CasResult};

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
