//! Solver runtime adapters re-exported for session clients.

pub use crate::assignment_apply::apply_assignment_with_context as apply_assignment;
pub use crate::assignment_command_runtime::evaluate_assignment_command_message_with_context as evaluate_assignment_command_message_with_simplifier;
pub use crate::assignment_command_runtime::evaluate_assignment_command_with_context as evaluate_assignment_command;
pub use crate::assignment_command_runtime::evaluate_let_assignment_command_message_with_context as evaluate_let_assignment_command_message_with_simplifier;
pub use crate::assignment_command_runtime::evaluate_let_assignment_command_with_context as evaluate_let_assignment_command;
pub use crate::bindings_command_runtime::evaluate_clear_bindings_command_lines as evaluate_clear_command_lines;
pub use crate::bindings_command_runtime::evaluate_vars_command_lines_from_bindings as evaluate_vars_command_lines;
pub use crate::bindings_command_runtime::evaluate_vars_command_lines_from_bindings_with_context as evaluate_vars_command_lines_with_context;
pub use crate::command_api::eval::{
    build_eval_command_render_plan, evaluate_eval_command_output,
    evaluate_eval_text_simplify_with_session,
};
pub use crate::eval_command_runtime::evaluate_eval_with_session;
pub use crate::health_suite_runner::run_suite_filtered as run_health_suite_filtered;
pub use crate::history_command_runtime::evaluate_history_command_lines_from_history as evaluate_history_command_lines;
pub use crate::history_command_runtime::evaluate_history_command_lines_from_history_with_context as evaluate_history_command_lines_with_context;
pub use crate::history_delete::evaluate_delete_history_command_message;
pub use crate::history_metadata_format::format_history_eval_metadata_sections;
pub use crate::linear_system_command_entry::evaluate_linear_system_command_message;
pub use crate::options_budget_eval::{
    apply_solve_budget_command, evaluate_solve_budget_command_message,
};
pub use crate::output_clean::clean_result_output_line;
pub use crate::rationalize_command::evaluate_rationalize_command_lines;
pub use crate::repl_config_runtime::evaluate_and_apply_config_command_on_runtime;
pub use crate::repl_eval_runtime::{
    evaluate_eval_command_render_plan_on_runtime as evaluate_eval_command_render_plan_on_repl_core,
    evaluate_expand_command_render_plan_on_runtime as evaluate_expand_command_render_plan_on_repl_core,
    profile_cache_len_on_runtime as profile_cache_len_on_repl_core, ReplEvalRuntimeContext,
};
pub use crate::repl_health_runtime::{
    evaluate_health_command_message_on_runtime as evaluate_health_command_message_on_repl_core,
    update_health_report_on_runtime as update_health_report_on_repl_core, ReplHealthRuntimeContext,
};
pub use crate::repl_runtime_configured::{
    build_runtime_with_config, reset_runtime_full_with_config, reset_runtime_with_config,
    ReplConfiguredRuntimeContext,
};
pub use crate::repl_runtime_state::build_repl_prompt_on_runtime as build_repl_prompt;
pub use crate::repl_runtime_state::clear_repl_profile_cache_on_runtime as clear_repl_profile_cache;
pub use crate::repl_runtime_state::eval_options_from_runtime as eval_options_from_repl_core;
pub use crate::repl_runtime_state::{
    reset_repl_runtime_state_on_runtime as reset_repl_runtime_state, ReplRuntimeStateContext,
};
pub use crate::repl_semantics_runtime::{
    apply_autoexpand_command_on_runtime as apply_autoexpand_command_on_repl_core,
    apply_context_command_on_runtime as apply_context_command_on_repl_core,
    apply_semantics_command_on_runtime as apply_semantics_command_on_repl_core,
    evaluate_autoexpand_command_on_runtime as evaluate_autoexpand_command_on_repl_core,
    evaluate_autoexpand_command_with_config_sync_on_runtime,
    evaluate_context_command_on_runtime as evaluate_context_command_on_repl_core,
    evaluate_context_command_with_config_sync_on_runtime,
    evaluate_semantics_command_on_runtime as evaluate_semantics_command_on_repl_core,
    evaluate_semantics_command_with_config_sync_on_runtime, ReplSemanticsRuntimeContext,
};
pub use crate::repl_session_runtime::evaluate_assignment_command_message_on_runtime as evaluate_assignment_command_message_on_repl_core;
pub use crate::repl_session_runtime::evaluate_clear_command_lines_on_runtime as evaluate_clear_command_lines_on_repl_core;
pub use crate::repl_session_runtime::evaluate_delete_history_command_message_on_runtime as evaluate_delete_history_command_message_on_repl_core;
pub use crate::repl_session_runtime::evaluate_history_command_message_on_runtime as evaluate_history_command_message_on_repl_core;
pub use crate::repl_session_runtime::evaluate_let_assignment_command_message_on_runtime as evaluate_let_assignment_command_message_on_repl_core;
pub use crate::repl_session_runtime::{
    evaluate_profile_cache_command_lines_on_runtime as evaluate_profile_cache_command_lines_on_repl_core,
    evaluate_show_command_lines_on_runtime as evaluate_show_command_lines_on_repl_core,
    evaluate_solve_budget_command_message_on_runtime as evaluate_solve_budget_command_message_on_repl_core,
    evaluate_vars_command_message_on_runtime as evaluate_vars_command_message_on_repl_core,
    ReplEngineRuntimeContext, ReplSessionEngineRuntimeContext, ReplSessionRuntimeContext,
    ReplSessionSimplifierRuntimeContext, ReplSessionStateMutRuntimeContext,
    ReplSessionViewRuntimeContext,
};
pub use crate::repl_set_runtime::{
    apply_set_command_plan_on_runtime as apply_set_command_plan_on_repl_core,
    evaluate_set_command_on_runtime as evaluate_set_command_on_repl_core,
    set_command_state_for_runtime as set_command_state_for_repl_core, ReplSetRuntimeContext,
};
pub use crate::repl_simplifier_runtime::{
    apply_profile_command_on_runtime as apply_profile_command_on_repl_core,
    evaluate_det_command_message_on_runtime as evaluate_det_command_message_on_repl_core,
    evaluate_equiv_invocation_message_on_runtime as evaluate_equiv_invocation_message_on_repl_core,
    evaluate_expand_log_invocation_message_on_runtime as evaluate_expand_log_invocation_message_on_repl_core,
    evaluate_explain_invocation_message_on_runtime as evaluate_explain_invocation_message_on_repl_core,
    evaluate_linear_system_command_message_on_runtime as evaluate_linear_system_command_message_on_repl_core,
    evaluate_profile_command_message_on_runtime as evaluate_profile_command_message_on_repl_core,
    evaluate_rationalize_command_lines_on_runtime as evaluate_rationalize_command_lines_on_repl_core,
    evaluate_substitute_invocation_user_message_on_runtime as evaluate_substitute_invocation_user_message_on_repl_core,
    evaluate_telescope_invocation_message_on_runtime as evaluate_telescope_invocation_message_on_repl_core,
    evaluate_trace_command_message_on_runtime as evaluate_trace_command_message_on_repl_core,
    evaluate_transpose_command_message_on_runtime as evaluate_transpose_command_message_on_repl_core,
    evaluate_unary_command_message_on_runtime,
    evaluate_visualize_invocation_output_on_runtime as evaluate_visualize_invocation_output_on_repl_core,
    evaluate_weierstrass_invocation_message_on_runtime as evaluate_weierstrass_invocation_message_on_repl_core,
    ReplSimplifierRuntimeContext,
};
pub use crate::repl_solve_runtime::{
    evaluate_full_simplify_command_lines_on_runtime as evaluate_full_simplify_command_lines_on_repl_core,
    evaluate_solve_command_message_on_runtime as evaluate_solve_command_message_on_repl_core,
    ReplSolveRuntimeContext,
};
pub use crate::repl_steps_runtime::{
    apply_steps_command_update_on_runtime as apply_steps_command_update_on_repl_core,
    steps_command_state_for_runtime as steps_command_state_for_repl_core, ReplStepsRuntimeContext,
};
pub use crate::show_command::evaluate_show_command_lines;
pub use crate::solve_display_lines::format_solve_command_eval_lines;
pub use crate::timeline_command_eval::evaluate_timeline_command_with_session;
pub use cas_solver_core::health_category::Category as HealthSuiteCategory;
pub use cas_solver_core::repl_runtime::ReplSemanticsApplyOutput;
