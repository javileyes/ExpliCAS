mod dispatch;
mod output;
mod status;

pub(crate) use dispatch::evaluate_health_command;
pub(crate) use status::evaluate_health_status_lines;
// Public API is re-exported from `dispatch` and `status`.
