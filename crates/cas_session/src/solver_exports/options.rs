//! Solver option and settings command APIs re-exported for session clients.

pub use cas_solver::{
    apply_autoexpand_policy_to_options, autoexpand_budget_view_from_options,
    evaluate_and_apply_autoexpand_command, evaluate_autoexpand_command_input,
    format_autoexpand_current_message, format_autoexpand_set_message,
    format_autoexpand_unknown_mode_message, parse_autoexpand_command_input, AutoexpandBudgetView,
    AutoexpandCommandApplyOutput, AutoexpandCommandInput, AutoexpandCommandResult,
    AutoexpandCommandState,
};
pub use cas_solver::{
    apply_context_mode_to_options, evaluate_and_apply_context_command,
    evaluate_context_command_input, format_context_current_message, format_context_set_message,
    format_context_unknown_message, parse_context_command_input, ContextCommandApplyOutput,
    ContextCommandInput, ContextCommandResult,
};
pub use cas_solver::{
    apply_profile_cache_command, apply_profile_command, evaluate_profile_cache_command_lines,
    evaluate_profile_command_input, format_profile_cache_command_lines,
    parse_profile_command_input, ProfileCacheCommandResult, ProfileCommandInput,
    ProfileCommandResult,
};
pub use cas_solver::{
    apply_semantics_preset_by_name, apply_semantics_preset_by_name_to_options,
    apply_semantics_preset_state_to_options, evaluate_semantics_preset_args_to_options,
    find_semantics_preset, format_semantics_preset_application_lines,
    format_semantics_preset_help_lines, format_semantics_preset_list_lines,
    semantics_preset_state_from_options, semantics_presets, SemanticsPreset,
    SemanticsPresetApplication, SemanticsPresetApplyError, SemanticsPresetCommandOutput,
    SemanticsPresetState,
};
pub use cas_solver::{
    apply_semantics_set_args_to_options, apply_semantics_set_state_to_options,
    evaluate_semantics_set_args, evaluate_semantics_set_args_to_overview_lines,
    semantics_set_state_from_options, SemanticsSetState,
};
pub use cas_solver::{
    apply_set_command_plan, evaluate_set_command_input, format_set_help_text,
    format_set_option_value, parse_set_command_input, SetCommandApplyEffects, SetCommandInput,
    SetCommandPlan, SetCommandResult, SetCommandState, SetDisplayMode,
};
pub use cas_solver::{apply_simplifier_toggle_config, build_simplifier_with_rule_config};
pub use cas_solver::{
    apply_steps_command_update, evaluate_steps_command_input, format_steps_current_message,
    format_steps_unknown_mode_message, parse_steps_command_input, StepsCommandApplyEffects,
    StepsCommandInput, StepsCommandResult, StepsCommandState, StepsDisplayMode,
};
