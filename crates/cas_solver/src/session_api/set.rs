//! Set command/session-facing API re-exported for session clients.

pub use crate::repl_set_runtime::{
    apply_set_command_plan_on_runtime as apply_set_command_plan_on_repl_core,
    evaluate_set_command_on_runtime as evaluate_set_command_on_repl_core,
    set_command_state_for_runtime as set_command_state_for_repl_core, ReplSetRuntimeContext,
};
pub use crate::repl_set_types::{ReplSetCommandOutput, ReplSetMessageKind};
pub use crate::set_command_apply::apply_set_command_plan;
pub use crate::set_command_eval::evaluate_set_command_input;
pub use crate::set_command_format::{format_set_help_text, format_set_option_value};
pub use crate::set_command_parse::parse_set_command_input;
pub use crate::set_command_types::{
    SetCommandApplyEffects, SetCommandInput, SetCommandPlan, SetCommandResult, SetCommandState,
    SetDisplayMode,
};
