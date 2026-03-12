//! Linear-algebra command APIs re-exported for session clients.

pub use crate::linear_system_command_entry::evaluate_linear_system_command_message;
pub use crate::repl_simplifier_runtime::{
    evaluate_det_command_message_on_runtime as evaluate_det_command_message_on_repl_core,
    evaluate_linear_system_command_message_on_runtime as evaluate_linear_system_command_message_on_repl_core,
    evaluate_trace_command_message_on_runtime as evaluate_trace_command_message_on_repl_core,
    evaluate_transpose_command_message_on_runtime as evaluate_transpose_command_message_on_repl_core,
};
