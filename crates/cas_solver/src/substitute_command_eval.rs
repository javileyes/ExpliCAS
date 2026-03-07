mod eval;
mod invocation;

pub use eval::evaluate_substitute_command_lines;
pub use invocation::{
    evaluate_substitute_invocation_lines, evaluate_substitute_invocation_message,
    evaluate_substitute_invocation_user_message,
};
