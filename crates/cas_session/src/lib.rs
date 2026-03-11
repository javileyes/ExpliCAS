//! Session-related components extracted from `cas_engine`.

use cas_solver_core::diagnostics_model::{Diagnostics, RequiredItem};

#[cfg(test)]
mod assignment_tests;
#[cfg(test)]
mod bindings_tests;
mod cache;
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
mod resolve_refs;
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
pub mod solver_exports;
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

pub use cache::{CacheDomainMode, SimplifiedCache, SimplifyCacheKey};
pub use config::{
    apply_solver_toggle_to_cas_config, solver_rule_config_from_cas_config,
    solver_toggle_config_from_cas_config, sync_simplifier_with_cas_config, CasConfig,
};
pub use repl_core::ReplCore;
pub use repl_runtime_core::{
    build_repl_core_with_config, evaluate_and_apply_config_command_on_repl,
    reset_repl_core_full_with_config, reset_repl_core_with_config,
};
pub use repl_semantics_runtime::{
    evaluate_autoexpand_command_on_repl, evaluate_context_command_on_repl,
    evaluate_semantics_command_on_repl,
};
pub use resolve_refs::{
    resolve_session_refs, resolve_session_refs_with_diagnostics, resolve_session_refs_with_env,
    resolve_session_refs_with_mode, resolve_session_refs_with_mode_and_diagnostics,
    resolve_session_refs_with_mode_and_env,
};
pub use session_io::{
    load_or_new_session, run_with_domain_session, run_with_session, save_session,
};
pub type CacheHitEntryId = u64;

pub type ResolvedExpr = cas_session_core::cache::ResolvedExpr<RequiredItem>;

pub type Entry = cas_session_core::store::Entry<Diagnostics, SimplifiedCache>;
pub type SessionStore = cas_session_core::store::SessionStore<Diagnostics, SimplifiedCache>;
pub use cas_engine::{Engine, Simplifier, Step};
pub use cas_session_core::env;
pub use cas_session_core::types::{CacheConfig, EntryId, EntryKind, RefMode, ResolveError};
pub use cas_solver_core::eval_display_types::{
    EvalDisplayMessage, EvalDisplayMessageKind, EvalMetadataLines, EvalResultLine,
};
pub use cas_solver_core::eval_option_axes::{BranchMode, ComplexMode, ContextMode, StepsMode};
pub use cas_solver_core::eval_options::EvalOptions;
pub use cas_solver_core::expand_policy::ExpandPolicy;
pub use cas_solver_core::phase_stats::PipelineStats;
pub use cas_solver_core::simplify_options::{SharedSemanticConfig, SimplifyOptions};
pub use env::{is_reserved, substitute, substitute_with_shadow, Environment};
pub use eval_command::{
    evaluate_eval_command_pretty_with_session, evaluate_eval_command_with_session,
    EvalCommandConfig,
};
pub use eval_text_command::evaluate_eval_text_command_with_session;
pub use snapshot::SnapshotError;
pub use state_core::SessionState;
