//! Compatibility shim for equation-level additive cancellation.
//!
//! The semantic fallback implementation lives in
//! `crate::cancel_common_terms_runtime`, while structural cancellation is
//! re-exported directly from `cas_solver_core`.

pub use crate::{cancel_additive_terms_semantic, cancel_common_additive_terms, CancelResult};
