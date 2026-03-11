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
mod eval_command;
#[cfg(test)]
mod eval_command_session_tests;
#[cfg(test)]
mod eval_command_tests;
mod eval_text_command;
#[cfg(test)]
mod history_tests;
#[cfg(test)]
mod inspect_tests;
#[cfg(test)]
mod options_tests;
#[cfg(test)]
mod repl_command_runtime_tests;
#[cfg(test)]
mod repl_config_runtime_tests;
mod repl_core;
mod repl_core_runtime_impls;
mod repl_runtime_core;
#[cfg(test)]
mod repl_runtime_tests;
mod repl_semantics_runtime;
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

pub(crate) use cache::{SimplifiedCache, SimplifyCacheKey};
pub use config::CasConfig;
pub use repl_core::ReplCore;
pub use repl_runtime_core::{
    build_repl_core_with_config, evaluate_and_apply_config_command_on_repl,
    reset_repl_core_full_with_config, reset_repl_core_with_config,
};
pub use repl_semantics_runtime::{
    evaluate_autoexpand_command_on_repl, evaluate_context_command_on_repl,
    evaluate_semantics_command_on_repl,
};
pub type CacheHitEntryId = u64;

pub(crate) type Entry = cas_session_core::store::Entry<Diagnostics, SimplifiedCache>;
pub(crate) type SessionStore = cas_session_core::store::SessionStore<Diagnostics, SimplifiedCache>;
pub use cas_session_core::env;
pub(crate) use cas_session_core::types::{CacheConfig, EntryId, EntryKind, RefMode, ResolveError};
pub use cas_solver_core::eval_display_types::{
    EvalDisplayMessage, EvalDisplayMessageKind, EvalMetadataLines, EvalResultLine,
};
pub use eval_command::{
    evaluate_eval_command_pretty_with_session, evaluate_eval_command_with_session,
    EvalCommandConfig,
};
pub use eval_text_command::evaluate_eval_text_command_with_session;
pub use state_core::SessionState;
