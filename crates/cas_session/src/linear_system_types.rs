use cas_ast::{ExprId, RelOp};

#[derive(Debug, Clone)]
pub(crate) struct LinearSystemSpec {
    pub(crate) exprs: Vec<ExprId>,
    pub(crate) vars: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum LinearSystemSpecError {
    InvalidPartCount,
    InvalidVariableName { name: String },
    ParseEquation { position: usize, message: String },
    ExpectedEquation { position: usize, input: String },
    UnsupportedRelation,
}

#[derive(Debug)]
pub(crate) struct LinearSystemCommandEvalOutput {
    pub(crate) vars: Vec<String>,
    pub(crate) result: cas_solver::LinSolveResult,
}

#[derive(Debug)]
pub(crate) enum LinearSystemCommandEvalError {
    Parse(LinearSystemSpecError),
    Solve(cas_solver::LinearSystemError),
}

pub(crate) fn ensure_equation_relation(op: RelOp) -> Result<(), LinearSystemSpecError> {
    if op == RelOp::Eq {
        Ok(())
    } else {
        Err(LinearSystemSpecError::UnsupportedRelation)
    }
}
