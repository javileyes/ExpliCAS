//! Local passthrough for engine error module exports.
//!
//! Keeping this bridge in solver-owned modules narrows direct `cas_engine`
//! references to dedicated migration boundaries.

pub mod error {
    pub use cas_solver_core::error_model::{CasError, CasResult};
}
