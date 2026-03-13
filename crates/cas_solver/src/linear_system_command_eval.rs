mod eval;
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

pub(crate) use eval::evaluate_linear_system_command_input;
