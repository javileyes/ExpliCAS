//! Session-related components extracted from `cas_engine`.

use cas_engine::{Diagnostics, RequireOrigin, RequiredItem};

mod assignment;
mod autoexpand_command;
mod bindings;
mod cache;
mod commands;
mod config;
mod context_command;
pub mod env;
mod health_command;
mod health_suite;
mod history;
mod inspect;
mod options;
mod semantics_command;
mod semantics_display;
mod semantics_presets;
mod semantics_set;
mod session_io;
mod simplifier_setup;
mod snapshot;
mod state;

pub use assignment::{
    apply_assignment, format_assignment_error_message, format_assignment_success_message,
    format_let_assignment_parse_error_message, let_assignment_usage_message,
    parse_let_assignment_input, AssignmentError, LetAssignmentParseError, ParsedLetAssignment,
};
pub use autoexpand_command::{
    apply_autoexpand_policy_to_options, autoexpand_budget_view_from_options,
    evaluate_and_apply_autoexpand_command, evaluate_autoexpand_command_input,
    format_autoexpand_current_message, format_autoexpand_set_message,
    format_autoexpand_unknown_mode_message, parse_autoexpand_command_input, AutoexpandBudgetView,
    AutoexpandCommandApplyOutput, AutoexpandCommandInput, AutoexpandCommandResult,
    AutoexpandCommandState,
};
pub use bindings::{
    binding_overview_entries, clear_bindings_command, format_binding_overview_lines,
    format_clear_bindings_result_lines, vars_empty_message, BindingOverviewEntry,
    ClearBindingsResult,
};
pub use cache::{CacheDomainMode, SimplifiedCache, SimplifyCacheKey};
pub use commands::{
    apply_profile_cache_command, apply_profile_command, evaluate_assignment_command,
    evaluate_assignment_command_message_with_simplifier, evaluate_clear_command_lines,
    evaluate_delete_history_command_message, evaluate_history_command_lines,
    evaluate_history_command_lines_with_context, evaluate_let_assignment_command,
    evaluate_let_assignment_command_message_with_simplifier, evaluate_profile_command_input,
    evaluate_vars_command_lines, evaluate_vars_command_lines_with_context,
    format_assignment_command_output_message, format_profile_cache_command_lines,
    format_show_history_command_lines, format_show_history_command_lines_with_context,
    parse_profile_command_input, AssignmentCommandOutput, ProfileCacheCommandResult,
    ProfileCommandInput, ProfileCommandResult,
};
pub use config::CasConfig;
pub use context_command::{
    apply_context_mode_to_options, evaluate_and_apply_context_command,
    evaluate_context_command_input, format_context_current_message, format_context_set_message,
    format_context_unknown_message, parse_context_command_input, ContextCommandApplyOutput,
    ContextCommandInput, ContextCommandResult,
};
pub use health_command::{
    capture_health_report_if_enabled, clear_health_profiler, evaluate_health_command,
    evaluate_health_command_input, evaluate_health_status_lines,
    format_health_failed_tests_warning_line, format_health_invalid_category_message,
    format_health_missing_category_arg_message, format_health_report_lines,
    format_health_status_running_message, format_health_usage_message, health_clear_message,
    health_disable_message, health_enable_message, health_usage_message,
    parse_health_command_input, resolve_health_category_filter, HealthCommandEvalOutput,
    HealthCommandInput, HealthStatusInput,
};
pub use health_suite::{
    category_names as health_suite_category_names, count_results as count_health_results,
    format_report_filtered as format_health_suite_report_filtered,
    list_cases as list_health_suite_cases, run_suite_filtered as run_health_suite_filtered,
    Category as HealthSuiteCategory,
};
pub use history::{
    delete_history_entries, format_delete_history_error_message,
    format_delete_history_result_message, format_history_overview_lines, history_empty_message,
    history_overview_entries, parse_history_ids, DeleteHistoryError, DeleteHistoryResult,
    HistoryOverviewEntry, HistoryOverviewKind,
};
pub use inspect::{
    format_history_entry_inspection_lines, format_inspect_history_entry_error_message,
    inspect_history_entry, inspect_history_entry_input, parse_history_entry_id,
    HistoryEntryDetails, HistoryEntryInspection, HistoryExprInspection,
    InspectHistoryEntryInputError, ParseHistoryEntryIdError,
};
pub use options::{
    apply_solve_budget_command, format_solve_budget_command_message, SolveBudgetCommandResult,
};
pub use semantics_command::{
    evaluate_semantics_command_line, parse_semantics_command_input, SemanticsCommandInput,
    SemanticsCommandOutput,
};
pub use semantics_display::{
    format_semantics_axis_lines, format_semantics_overview_lines,
    format_semantics_unknown_subcommand_message, semantics_help_message,
    semantics_view_state_from_options, SemanticsViewState,
};
pub use semantics_presets::{
    apply_semantics_preset_by_name, apply_semantics_preset_by_name_to_options,
    apply_semantics_preset_state_to_options, evaluate_semantics_preset_args_to_options,
    find_semantics_preset, format_semantics_preset_application_lines,
    format_semantics_preset_help_lines, format_semantics_preset_list_lines,
    semantics_preset_state_from_options, semantics_presets, SemanticsPreset,
    SemanticsPresetApplication, SemanticsPresetApplyError, SemanticsPresetCommandOutput,
    SemanticsPresetState,
};
pub use semantics_set::{
    apply_semantics_set_args_to_options, apply_semantics_set_state_to_options,
    evaluate_semantics_set_args, evaluate_semantics_set_args_to_overview_lines,
    semantics_set_state_from_options, SemanticsSetState,
};
pub use session_io::{
    load_or_new_session, run_with_domain_session, run_with_session, save_session,
};
pub use simplifier_setup::{
    apply_simplifier_toggle_config, build_simplifier_with_rule_config, set_simplifier_toggle_rule,
    SimplifierRuleConfig, SimplifierToggleConfig,
};
pub type CacheHitEntryId = u64;

pub type ResolvedExpr = cas_session_core::cache::ResolvedExpr<RequiredItem>;

pub type Entry = cas_session_core::store::Entry<Diagnostics, SimplifiedCache>;
pub type SessionStore = cas_session_core::store::SessionStore<Diagnostics, SimplifiedCache>;
pub use cas_session_core::types::{CacheConfig, EntryId, EntryKind, RefMode, ResolveError};
pub use env::{is_reserved, substitute, substitute_with_shadow, Environment};
pub use snapshot::SnapshotError;
pub use state::SessionState;

fn mode_entry_from_store_entry(
    entry: &Entry,
) -> cas_session_core::resolve::ModeEntry<SimplifyCacheKey, RequiredItem> {
    cas_session_core::resolve::ModeEntry {
        kind: entry.kind.clone(),
        requires: entry.diagnostics.requires.clone(),
        cache: entry
            .simplified
            .as_ref()
            .map(|cache| cas_session_core::resolve::ModeCacheEntry {
                key: cache.key.clone(),
                expr: cache.expr,
                requires: cache.requires.clone(),
            }),
    }
}

fn same_requirement(lhs: &RequiredItem, rhs: &RequiredItem) -> bool {
    lhs.cond == rhs.cond
}

fn mark_session_propagated(item: &mut RequiredItem) {
    item.merge_origin(RequireOrigin::SessionPropagated);
}

fn mode_resolve_config<'a>(
    mode: RefMode,
    cache_key: &'a SimplifyCacheKey,
    env: &'a Environment,
) -> cas_session_core::resolve::ModeResolveConfig<'a, SimplifyCacheKey> {
    cas_session_core::resolve::ModeResolveConfig {
        mode,
        cache_key,
        env,
    }
}

fn push_session_propagated_requirement(diagnostics: &mut Diagnostics, item: RequiredItem) {
    diagnostics.push_required(item.cond, RequireOrigin::SessionPropagated);
}

fn with_mode_resolution_plumbing<T, F>(store: &SessionStore, run: F) -> T
where
    F: FnOnce(
        &mut dyn FnMut(
            EntryId,
        ) -> Option<
            cas_session_core::resolve::ModeEntry<SimplifyCacheKey, RequiredItem>,
        >,
        &mut dyn FnMut(&RequiredItem, &RequiredItem) -> bool,
        &mut dyn FnMut(&mut RequiredItem),
    ) -> T,
{
    let mut lookup = |id: EntryId| store.get(id).map(mode_entry_from_store_entry);
    let mut same = same_requirement;
    let mut mark = mark_session_propagated;
    run(&mut lookup, &mut same, &mut mark)
}

pub(crate) fn simplify_cache_steps_len(cache: &SimplifiedCache) -> usize {
    cache.steps.as_ref().map(|s| s.len()).unwrap_or(0)
}

pub(crate) fn apply_simplified_light_cache(
    mut cache: SimplifiedCache,
    light_cache_threshold: Option<usize>,
) -> SimplifiedCache {
    if let Some(threshold) = light_cache_threshold {
        if simplify_cache_steps_len(&cache) > threshold {
            cache.steps = None;
        }
    }
    cache
}

pub(crate) fn session_store_with_cache_config(cache_config: CacheConfig) -> SessionStore {
    SessionStore::with_cache_config_and_policy(
        cache_config,
        simplify_cache_steps_len,
        apply_simplified_light_cache,
    )
}

/// Resolve all `Expr::SessionRef` in an expression tree.
pub fn resolve_session_refs(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
    store: &SessionStore,
) -> Result<cas_ast::ExprId, ResolveError> {
    let mut lookup = |id: EntryId| store.get(id).map(|entry| entry.kind.clone());
    cas_session_core::resolve::resolve_session_refs_with_lookup(ctx, expr, &mut lookup)
}

/// Resolve session refs and apply environment substitution.
pub fn resolve_session_refs_with_env(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
    store: &SessionStore,
    env: &Environment,
) -> Result<cas_ast::ExprId, ResolveError> {
    let mut lookup = |id: EntryId| store.get(id).map(|entry| entry.kind.clone());
    cas_session_core::resolve::resolve_all_with_lookup_and_env(ctx, expr, &mut lookup, env)
}

/// Resolve session refs and accumulate inherited diagnostics.
pub fn resolve_session_refs_with_diagnostics(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
    store: &SessionStore,
) -> Result<(cas_ast::ExprId, Diagnostics), ResolveError> {
    let mut lookup = |id: EntryId| store.get(id).map(|entry| entry.kind.clone());
    cas_session_core::resolve::resolve_session_refs_with_lookup_accumulator(
        ctx,
        expr,
        &mut lookup,
        Diagnostics::new(),
        |inherited, id| {
            if let Some(entry) = store.get(id) {
                inherited.inherit_requires_from(&entry.diagnostics);
            }
        },
    )
}

/// Resolve session refs with mode selection and cache checking.
pub fn resolve_session_refs_with_mode(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
    store: &SessionStore,
    mode: RefMode,
    cache_key: &SimplifyCacheKey,
) -> Result<ResolvedExpr, ResolveError> {
    with_mode_resolution_plumbing(store, |mut lookup, mut same, mut mark| {
        cas_session_core::resolve::resolve_session_refs_with_mode_lookup(
            ctx,
            expr,
            mode,
            cache_key,
            &mut lookup,
            &mut same,
            &mut mark,
        )
    })
}

/// Resolve session refs with mode selection and apply environment substitution.
pub fn resolve_session_refs_with_mode_and_env(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
    store: &SessionStore,
    mode: RefMode,
    cache_key: &SimplifyCacheKey,
    env: &Environment,
) -> Result<ResolvedExpr, ResolveError> {
    with_mode_resolution_plumbing(store, |mut lookup, mut same, mut mark| {
        cas_session_core::resolve::resolve_all_with_mode_lookup_and_env(
            ctx,
            expr,
            mode_resolve_config(mode, cache_key, env),
            &mut lookup,
            &mut same,
            &mut mark,
        )
    })
}

/// Resolve session refs with mode + env and return inherited diagnostics + cache hits.
pub fn resolve_session_refs_with_mode_and_diagnostics(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
    store: &SessionStore,
    mode: RefMode,
    cache_key: &SimplifyCacheKey,
    env: &Environment,
) -> Result<(cas_ast::ExprId, Diagnostics, Vec<EntryId>), ResolveError> {
    with_mode_resolution_plumbing(store, |mut lookup, mut same, mut mark| {
        cas_session_core::resolve::resolve_mode_with_env_and_diagnostics(
            ctx,
            expr,
            mode_resolve_config(mode, cache_key, env),
            &mut lookup,
            &mut same,
            &mut mark,
            Diagnostics::new(),
            push_session_propagated_requirement,
        )
    })
}
