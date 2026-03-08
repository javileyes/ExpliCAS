mod diagnostics;
mod eval;
mod prepare;

pub use cas_solver_core::solve_command_types::SolveCommandEvalError;

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

pub use eval::evaluate_solve_command_with_session;
pub(crate) use eval::evaluate_solve_parsed_with_session;
pub(crate) use prepare::prepare_solve_eval_request;
