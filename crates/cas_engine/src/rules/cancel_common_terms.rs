//! Compatibility shim for equation-level additive cancellation.
//!
//! The implementation lives in `crate::cancel_common_terms_runtime` because
//! this operation is equation-relational and not a simplifier rewrite rule.

pub use crate::cancel_common_terms_runtime::{
    cancel_additive_terms_semantic, cancel_common_additive_terms, CancelResult,
};
