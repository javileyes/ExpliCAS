mod eval;
mod runtime;
mod solve;

#[derive(Debug)]
pub(crate) struct LinearSystemCommandEvalOutput {
    pub(crate) vars: Vec<String>,
    pub(crate) result: crate::LinSolveResult,
}

#[derive(Debug)]
pub(crate) enum LinearSystemCommandEvalError {
    Parse(crate::linear_system_command_parse::LinearSystemSpecError),
    Solve(crate::LinearSystemError),
}

mod nonlinear;
mod resultant;
mod steps;

pub(crate) use eval::{
    evaluate_linear_system_command_input, evaluate_linear_system_command_input_with_simplifier,
};
pub(crate) use runtime::evaluate_linear_system_eval_request_with_session;
