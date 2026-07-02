//! Environment command/session-facing API re-exported for session clients.

pub use crate::repl_semantics_runtime::{
    apply_autoexpand_command_on_runtime as apply_autoexpand_command_on_repl_core,
    apply_context_command_on_runtime as apply_context_command_on_repl_core,
    apply_semantics_command_on_runtime as apply_semantics_command_on_repl_core,
    evaluate_autoexpand_command_with_config_sync_on_runtime,
    evaluate_context_command_with_config_sync_on_runtime,
    evaluate_semantics_command_with_config_sync_on_runtime, ReplSemanticsRuntimeContext,
};
pub use crate::semantics_preset_types::{
    SemanticsPreset, SemanticsPresetApplication, SemanticsPresetApplyError,
    SemanticsPresetCommandOutput, SemanticsPresetState,
};
pub use crate::semantics_set_types::SemanticsSetState;
pub use crate::semantics_view_types::SemanticsViewState;
pub use cas_api_models::{
    AutoexpandBudgetView, AutoexpandCommandApplyOutput, AutoexpandCommandInput,
    AutoexpandCommandResult, AutoexpandCommandState,
};
pub use cas_api_models::{ContextCommandApplyOutput, ContextCommandInput, ContextCommandResult};
pub use cas_solver_core::repl_runtime::ReplSemanticsApplyOutput;
