mod error;
mod result;
mod solution;

pub use solution::display_linear_system_solution;

pub(crate) use error::format_linear_system_command_error_message;
pub(crate) use result::format_linear_system_result_message;
