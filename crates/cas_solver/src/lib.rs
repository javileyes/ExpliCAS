//! Solver facade crate.
//!
//! During migration this crate hosts the solver entry points while still
//! re-exporting selected `cas_engine` APIs for compatibility.

mod algebra_command_eval;
mod algebra_command_parse;
#[cfg(test)]
mod algebra_command_tests;
#[cfg(test)]
mod analysis_command_eval_tests;
mod analysis_command_explain;
mod analysis_command_format_errors;
mod analysis_command_format_explain;
#[cfg(test)]
mod analysis_command_format_tests;
mod analysis_command_parse;
#[cfg(test)]
mod analysis_command_parse_tests;
mod analysis_command_types;
mod analysis_command_visualize;
mod analysis_input_parse;
#[cfg(test)]
mod analysis_input_parse_tests;
mod assignment_apply;
#[cfg(test)]
mod assignment_apply_tests;
mod assignment_command;
mod assignment_command_runtime;
#[cfg(test)]
mod assignment_command_runtime_tests;
#[cfg(test)]
mod assignment_command_tests;
mod assignment_format;
mod assignment_parse;
mod assignment_types;
mod assumption_format;
mod autoexpand_command_eval;
mod autoexpand_command_format;
mod autoexpand_command_parse;
#[cfg(test)]
mod autoexpand_command_tests;
mod autoexpand_command_types;
mod bindings_command;
mod bindings_command_runtime;
#[cfg(test)]
mod bindings_command_runtime_tests;
#[cfg(test)]
mod bindings_command_tests;
mod bindings_format;
mod bindings_types;
mod blocked_hint_format;
mod budget_runtime_types;
mod cancel_runtime;
mod config_command_apply;
mod config_command_eval;
mod config_command_parse;
#[cfg(test)]
mod config_command_tests;
mod config_command_types;
mod const_fold_local;
mod context_command_eval;
mod context_command_format;
mod context_command_parse;
#[cfg(test)]
mod context_command_tests;
mod context_command_types;
mod display_eval_steps;
mod domain_facade;
mod engine_bridge;
mod engine_runtime_types;
mod equiv_command;
mod equiv_format;
mod error_runtime_types;
mod eval_command_eval;
mod eval_command_format;
mod eval_command_format_metadata;
mod eval_command_format_result;
mod eval_command_render;
mod eval_command_request;
mod eval_command_text;
mod eval_command_types;
mod eval_json_command_runtime;
mod eval_json_finalize;
mod eval_json_finalize_expr;
mod eval_json_finalize_input;
mod eval_json_finalize_nonexpr;
mod eval_json_finalize_wire;
mod eval_json_input;
mod eval_json_input_special;
#[cfg(test)]
mod eval_json_input_tests;
mod eval_json_input_variable;
mod eval_json_options;
#[cfg(test)]
mod eval_json_options_tests;
mod eval_json_presentation;
mod eval_json_presentation_conditions;
mod eval_json_presentation_solution;
mod eval_json_presentation_solution_display;
mod eval_json_presentation_solution_latex;
mod eval_json_presentation_solve;
mod eval_json_presentation_solve_input;
mod eval_json_presentation_solve_steps;
mod eval_json_request_runtime;
mod eval_json_stats;
mod eval_json_stats_format;
mod eval_json_stats_hash;
mod eval_json_stats_metrics;
#[cfg(test)]
mod eval_json_stats_tests;
mod eval_output_adapters;
mod exports;
mod exports_base;
mod exports_commands;
mod exports_repl;
mod full_simplify_command;
mod full_simplify_display;
mod full_simplify_eval;
mod health_command_eval;
mod health_command_format;
mod health_command_messages;
mod health_command_parse;
#[cfg(test)]
mod health_command_tests;
mod health_command_types;
mod health_suite_catalog;
mod health_suite_catalog_core;
mod health_suite_catalog_stress;
mod health_suite_format_catalog;
mod health_suite_format_report;
mod health_suite_runner;
mod health_suite_types;
mod history_command_display;
#[cfg(test)]
mod history_command_display_tests;
mod history_command_runtime;
#[cfg(test)]
mod history_command_runtime_tests;
mod history_delete;
#[cfg(test)]
mod history_delete_tests;
mod history_format;
mod history_metadata_format;
#[cfg(test)]
mod history_metadata_format_tests;
mod history_overview;
#[cfg(test)]
mod history_overview_tests;
mod history_parse;
mod history_show_format;
mod history_types;
mod input_parse_common;
mod inspect_format;
mod inspect_parse;
mod inspect_runtime;
mod inspect_types;
mod json;
#[cfg(test)]
mod json_bridge_tests;
mod limit_command;
mod limit_command_core;
mod limit_command_eval;
#[cfg(test)]
mod limit_command_eval_tests;
mod limit_command_parse;
#[cfg(test)]
mod limit_command_tests;
mod limit_command_types;
mod limit_subcommand;
#[cfg(test)]
mod limit_subcommand_tests;
mod linear_system;
mod linear_system_command_entry;
mod linear_system_command_eval;
mod linear_system_command_format;
mod linear_system_command_parse;
#[cfg(test)]
mod linear_system_command_tests;
mod linear_system_command_types;
#[cfg(test)]
mod linear_system_tests;
mod options_budget_eval;
#[cfg(test)]
mod options_budget_eval_tests;
mod options_budget_format;
mod options_budget_types;
mod output_clean;
#[cfg(test)]
mod output_clean_tests;
mod parse_error_render;
#[cfg(test)]
mod parse_error_render_tests;
mod path_rewrite;
#[cfg(test)]
mod path_rewrite_tests;
mod phase_runtime_types;
mod pipeline_display;
#[cfg(test)]
mod pipeline_display_tests;
mod profile_cache_command;
mod profile_command;
mod prompt_display;
#[cfg(test)]
mod prompt_display_tests;
mod rationalize_command;
mod rationalize_command_eval;
mod rationalize_command_format;
mod rationalize_command_parse;
#[cfg(test)]
mod rationalize_command_tests;
mod rationalize_command_types;
mod repl_command_parse;
mod repl_command_parse_early;
mod repl_command_parse_routing;
mod repl_command_preprocess;
#[cfg(test)]
mod repl_command_routing_tests;
mod repl_command_types;
mod repl_config_runtime;
#[cfg(test)]
mod repl_config_runtime_tests;
mod repl_eval_runtime;
#[cfg(test)]
mod repl_eval_runtime_tests;
mod repl_health_runtime;
#[cfg(test)]
mod repl_health_runtime_tests;
mod repl_runtime_configured;
#[cfg(test)]
mod repl_runtime_configured_tests;
mod repl_runtime_state;
#[cfg(test)]
mod repl_runtime_state_tests;
mod repl_semantics_runtime;
#[cfg(test)]
mod repl_semantics_runtime_tests;
mod repl_session_runtime;
mod repl_set_runtime;
#[cfg(test)]
mod repl_set_runtime_tests;
mod repl_set_types;
mod repl_simplifier_runtime;
#[cfg(test)]
mod repl_simplifier_runtime_tests;
mod repl_solve_runtime;
mod repl_steps_runtime;
#[cfg(test)]
mod repl_steps_runtime_tests;
mod rule_runtime_types;
mod rules_runtime_types;
mod semantics_command_eval;
mod semantics_command_parse;
#[cfg(test)]
mod semantics_command_tests;
mod semantics_command_types;
#[cfg(test)]
mod semantics_display_tests;
mod semantics_preset_apply;
mod semantics_preset_catalog;
mod semantics_preset_format;
mod semantics_preset_labels;
mod semantics_preset_types;
#[cfg(test)]
mod semantics_presets_tests;
mod semantics_set_apply;
mod semantics_set_parse;
mod semantics_set_parse_apply;
mod semantics_set_parse_axis;
#[cfg(test)]
mod semantics_set_tests;
mod semantics_set_types;
mod semantics_view_format;
mod semantics_view_format_axis;
mod semantics_view_format_help;
mod semantics_view_format_overview;
mod semantics_view_types;
mod set_command_apply;
mod set_command_eval;
mod set_command_format;
mod set_command_options;
mod set_command_options_rules;
mod set_command_options_steps;
mod set_command_parse;
#[cfg(test)]
mod set_command_tests;
mod set_command_types;
mod show_command;
#[cfg(test)]
mod show_command_tests;
mod simplifier_setup_build;
mod simplifier_setup_toggle;
mod simplifier_setup_types;
mod solution_display;
mod solve_backend;
mod solve_backend_active;
mod solve_backend_contract;
mod solve_backend_dispatch;
mod solve_backend_local;
mod solve_command_errors;
mod solve_command_eval_core;
mod solve_command_session_eval;
mod solve_core_runtime;
mod solve_display_lines;
mod solve_display_result;
mod solve_display_steps;
mod solve_input_parse_parse;
mod solve_input_parse_prepare;
#[cfg(test)]
mod solve_input_parse_tests;
mod solve_input_types;
mod solve_render_config;
mod solve_safety;
mod solve_verify_display;
mod solver_entrypoints;
mod solver_entrypoints_eval;
mod solver_entrypoints_proof_verify;
mod solver_entrypoints_solve;
mod solver_number_theory;
mod standard_oracle;
mod step_runtime_types;
mod steps_command_eval;
mod steps_command_format;
mod steps_command_parse;
#[cfg(test)]
mod steps_command_tests;
mod steps_command_types;
pub mod substitute;
mod substitute_command_eval;
mod substitute_command_format;
mod substitute_command_parse;
#[cfg(test)]
mod substitute_command_tests;
mod substitute_command_types;
mod substitute_subcommand_eval;
mod substitute_subcommand_json;
#[cfg(test)]
mod substitute_subcommand_tests;
mod substitute_subcommand_text;
mod substitute_subcommand_types;
#[cfg(test)]
mod substitute_tests;
mod symbolic_transforms;
mod telescoping;
mod timeline_command_eval;
mod timeline_simplify_eval;
mod timeline_solve_eval;
mod timeline_types;
mod types;
mod unary_command_eval;
#[cfg(test)]
mod unary_command_tests;
mod unary_display;
mod vars_command_display;
#[cfg(test)]
mod vars_command_display_tests;
mod weierstrass_command;
#[cfg(test)]
mod weierstrass_command_tests;

/// Backward-compatible facade for former `cas_engine::strategies::substitute_expr` imports.
pub mod strategies {
    pub use cas_ast::substitute_expr_by_id as substitute_expr;
}

/// Backward-compatible facade for former `cas_engine::api::*` imports.
pub mod api;

pub use exports::*;
