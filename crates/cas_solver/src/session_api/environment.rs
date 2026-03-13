//! Environment command/session-facing API re-exported for session clients.

pub use crate::autoexpand_command_eval::{
    apply_autoexpand_policy_to_options, autoexpand_budget_view_from_options,
    evaluate_and_apply_autoexpand_command, evaluate_autoexpand_command_input,
};
pub use crate::autoexpand_command_format::{
    format_autoexpand_current_message, format_autoexpand_set_message,
    format_autoexpand_unknown_mode_message,
};
pub use crate::autoexpand_command_parse::parse_autoexpand_command_input;
pub use crate::context_command_eval::{
    apply_context_mode_to_options, evaluate_and_apply_context_command,
    evaluate_context_command_input,
};
pub use crate::context_command_format::{
    format_context_current_message, format_context_set_message, format_context_unknown_message,
};
pub use crate::context_command_parse::parse_context_command_input;
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
pub use crate::semantics_preset_apply::{
    apply_semantics_preset_by_name, apply_semantics_preset_by_name_to_options,
    apply_semantics_preset_state_to_options, evaluate_semantics_preset_args_to_options,
    semantics_preset_state_from_options,
};
pub use crate::semantics_preset_catalog::{find_semantics_preset, semantics_presets};
pub use crate::semantics_preset_format::{
    format_semantics_preset_application_lines, format_semantics_preset_help_lines,
    format_semantics_preset_list_lines,
};
pub use crate::semantics_set_apply::{
    apply_semantics_set_args_to_options, apply_semantics_set_state_to_options,
    evaluate_semantics_set_args_to_overview_lines,
};
pub use crate::semantics_set_parse_apply::evaluate_semantics_set_args;
pub use crate::semantics_view_format::{
    format_semantics_axis_lines, format_semantics_overview_lines,
    format_semantics_unknown_subcommand_message, semantics_help_message,
};
pub use cas_solver_core::autoexpand_command_types::{
    AutoexpandBudgetView, AutoexpandCommandApplyOutput, AutoexpandCommandInput,
    AutoexpandCommandResult, AutoexpandCommandState,
};
pub use cas_solver_core::context_command_types::{
    ContextCommandApplyOutput, ContextCommandInput, ContextCommandResult,
};
pub use cas_solver_core::repl_runtime::ReplSemanticsApplyOutput;
pub use cas_solver_core::semantics_preset_types::{
    SemanticsPreset, SemanticsPresetApplication, SemanticsPresetApplyError,
    SemanticsPresetCommandOutput, SemanticsPresetState,
};
pub use cas_solver_core::semantics_set_types::{
    semantics_set_state_from_options, SemanticsSetState,
};
pub use cas_solver_core::semantics_view_types::{
    semantics_view_state_from_options, SemanticsViewState,
};
