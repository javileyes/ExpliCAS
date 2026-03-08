//! Core session primitives shared across crates.
//!
//! This crate is intentionally separate from `cas_session`.
//!
//! It is not a temporary migration artifact anymore. It exists to keep the
//! dependency graph acyclic:
//!
//! - `cas_engine` needs session-agnostic eval contracts and helpers.
//! - `cas_solver` needs low-level store, snapshot and entry-id models.
//! - `cas_session` depends on both `cas_engine` and `cas_solver`.
//!
//! Moving these primitives back into `cas_session` would force wider
//! dependencies or reintroduce cycles. So `cas_session_core` is the permanent
//! shared kernel for:
//!
//! - stateless eval/session contracts
//! - session-store and snapshot primitives
//! - reference resolution helpers
//! - small shared DTOs used below the stateful session layer

pub mod cache;
pub mod context_snapshot;
pub mod env;
pub mod eval;
pub mod resolve;
pub mod snapshot_error;
pub mod snapshot_header;
pub mod snapshot_io;
pub mod store;
pub mod store_snapshot;
pub mod types;
