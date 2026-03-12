//! Solver option and settings command APIs re-exported for session clients.

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
pub use crate::profile_cache_command::{
    apply_profile_cache_command, evaluate_profile_cache_command_lines,
    format_profile_cache_command_lines,
};
pub use crate::profile_command::{
    apply_profile_command, evaluate_profile_command_input, parse_profile_command_input,
};
pub use crate::repl_set_runtime::ReplSetRuntimeContext;
pub use crate::repl_steps_runtime::ReplStepsRuntimeContext;
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
pub use crate::semantics_set_parse::evaluate_semantics_set_args;
pub use crate::semantics_set_types::semantics_set_state_from_options;
pub use crate::set_command_apply::apply_set_command_plan;
pub use crate::set_command_eval::evaluate_set_command_input;
pub use crate::set_command_format::{format_set_help_text, format_set_option_value};
pub use crate::set_command_parse::parse_set_command_input;
pub use crate::simplifier_setup_build::build_simplifier_with_rule_config;
pub use crate::simplifier_setup_toggle::apply_simplifier_toggle_config;
pub use crate::steps_command_eval::{apply_steps_command_update, evaluate_steps_command_input};
pub use crate::steps_command_format::{
    format_steps_current_message, format_steps_unknown_mode_message,
};
pub use crate::steps_command_parse::parse_steps_command_input;
pub use crate::steps_command_types::{
    StepsCommandApplyEffects, StepsCommandInput, StepsCommandResult, StepsCommandState,
    StepsDisplayMode,
};
pub use cas_solver_core::autoexpand_command_types::{
    AutoexpandBudgetView, AutoexpandCommandApplyOutput, AutoexpandCommandInput,
    AutoexpandCommandResult, AutoexpandCommandState,
};
pub use cas_solver_core::context_command_types::{
    ContextCommandApplyOutput, ContextCommandInput, ContextCommandResult,
};
pub use cas_solver_core::profile_cache_command_types::ProfileCacheCommandResult;
pub use cas_solver_core::profile_command_types::{ProfileCommandInput, ProfileCommandResult};
pub use cas_solver_core::semantics_preset_types::{
    SemanticsPreset, SemanticsPresetApplication, SemanticsPresetApplyError,
    SemanticsPresetCommandOutput, SemanticsPresetState,
};
pub use cas_solver_core::semantics_set_types::SemanticsSetState;
pub use cas_solver_core::set_command_types::{
    SetCommandApplyEffects, SetCommandInput, SetCommandPlan, SetCommandResult, SetCommandState,
    SetDisplayMode,
};
