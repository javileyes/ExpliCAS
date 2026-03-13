//! Session-related components extracted from `cas_engine`.

use cas_solver_core::diagnostics_model::Diagnostics;

#[cfg(test)]
mod assignment_tests;
#[cfg(test)]
mod bindings_tests;
pub mod cache;
#[cfg(test)]
mod cache_tests;
#[cfg(test)]
mod commands_tests;
mod config;
#[cfg(test)]
mod config_command_apply_tests;
#[cfg(test)]
mod config_tests;
pub mod eval;
#[cfg(test)]
mod eval_command_session_tests;
#[cfg(test)]
mod eval_command_tests;
#[cfg(test)]
mod history_tests;
#[cfg(test)]
mod inspect_tests;
#[cfg(test)]
mod options_tests;
pub mod repl;
#[cfg(test)]
mod repl_command_runtime_tests;
#[cfg(test)]
mod repl_config_runtime_tests;
mod repl_core;
mod repl_core_runtime_impls;
#[cfg(test)]
mod repl_runtime_tests;
#[cfg(test)]
mod repl_semantics_runtime_tests;
#[cfg(test)]
mod repl_set_runtime_tests;
#[cfg(test)]
mod repl_steps_runtime_tests;
pub mod resolve_refs;
#[cfg(test)]
mod resolve_refs_tests;
mod session_io;
#[cfg(test)]
mod show_command_tests;
#[cfg(test)]
mod simplifier_setup_tests;
mod snapshot;
mod snapshot_store_convert;
#[cfg(test)]
mod snapshot_tests;
pub mod state;
mod state_bindings;
mod state_core;
mod state_eval_session;
mod state_eval_store;
mod state_history;
mod state_persistence;
mod state_resolution;
mod store_cache_policy;
#[cfg(test)]
mod store_cache_policy_tests;
#[cfg(test)]
mod timeline_command_tests;

pub type CacheHitEntryId = u64;

pub(crate) type Entry = cas_session_core::store::Entry<Diagnostics, cache::SimplifiedCache>;
pub(crate) type SessionStore =
    cas_session_core::store::SessionStore<Diagnostics, cache::SimplifiedCache>;
pub use cas_session_core::env;
