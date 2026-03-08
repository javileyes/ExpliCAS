mod eval;
mod message;
mod types;

pub use eval::{evaluate_assignment_command_with, evaluate_let_assignment_command_with};
pub use message::{
    evaluate_assignment_command_message_with, evaluate_let_assignment_command_message_with,
    format_assignment_command_output_message,
};
pub use types::AssignmentCommandOutput;
