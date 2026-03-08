mod dispatch;
mod output;
mod status;

pub use dispatch::evaluate_health_command;
pub use status::evaluate_health_status_lines;
// Public API is re-exported from `dispatch` and `status`.
