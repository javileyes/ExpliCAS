//! Compatibility shim for equation-level additive cancellation.
//!
//! The implementation lives in `crate::solver::cancel_common_terms` because
//! this operation is equation-relational and not a simplifier rewrite rule.

pub use crate::solver::cancel_common_terms::{
    cancel_additive_terms_semantic, cancel_common_additive_terms, CancelResult,
};
