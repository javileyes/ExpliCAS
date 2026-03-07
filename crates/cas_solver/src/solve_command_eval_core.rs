mod diagnostics;
mod eval;
mod prepare;

pub(crate) struct PreparedSolveEvalRequest {
    pub raw_input: String,
    pub parsed_expr: cas_ast::ExprId,
    pub auto_store: bool,
    pub var: String,
    pub original_equation: Option<cas_ast::Equation>,
}

#[derive(Debug, Clone)]
pub struct SolveCommandEvalOutput {
    pub var: String,
    pub original_equation: Option<cas_ast::Equation>,
    pub output: crate::EvalOutputView,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SolveCommandEvalError {
    Prepare(crate::SolvePrepareError),
    Eval(String),
}

pub use eval::evaluate_solve_command_with_session;
pub(crate) use eval::evaluate_solve_parsed_with_session;
pub(crate) use prepare::prepare_solve_eval_request;
