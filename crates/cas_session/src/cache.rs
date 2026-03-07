//! Session-owned simplified cache models.
//!
//! These payloads live in `cas_session` (application state layer), while
//! `cas_engine` only reports simplification artifacts through `EvalStore`.

mod domain;
mod key;
mod simplified;

pub use domain::CacheDomainMode;
pub use key::SimplifyCacheKey;
pub use simplified::SimplifiedCache;
