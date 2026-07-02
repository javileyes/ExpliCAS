mod collect;
mod expand;
mod expand_log;
mod telescope;

pub(crate) use collect::evaluate_collect_wrapped_expression;
pub(crate) use expand::evaluate_expand_wrapped_expression;
pub(crate) use expand_log::{
    evaluate_expand_log_command_lines, evaluate_expand_log_invocation_lines,
    evaluate_expand_log_invocation_message,
};
pub(crate) use telescope::{
    evaluate_telescope_command_lines, evaluate_telescope_invocation_lines,
    evaluate_telescope_invocation_message,
};
