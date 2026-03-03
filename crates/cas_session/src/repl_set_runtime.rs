//! Runtime adapters for `set` command over REPL core state.

pub use crate::repl_set_apply::{
    apply_set_command_plan_on_repl_core, set_command_state_for_repl_core,
};
pub use crate::repl_set_eval::evaluate_set_command_on_repl_core;
pub use crate::repl_set_types::{ReplSetCommandOutput, ReplSetMessageKind};
