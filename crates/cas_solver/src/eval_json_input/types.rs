#[derive(Debug, Clone)]
pub enum EvalJsonNonSolveAction {
    Simplify,
    Limit {
        var: String,
        approach: cas_math::limit_types::Approach,
    },
}

#[derive(Debug, Clone)]
pub enum EvalJsonPreparedRequest {
    Solve {
        raw_input: String,
        parsed: cas_ast::ExprId,
        var: String,
        auto_store: bool,
    },
    Eval {
        raw_input: String,
        parsed: cas_ast::ExprId,
        action: EvalJsonNonSolveAction,
        auto_store: bool,
    },
}

impl EvalJsonPreparedRequest {
    pub fn parsed(&self) -> cas_ast::ExprId {
        match self {
            Self::Solve { parsed, .. } => *parsed,
            Self::Eval { parsed, .. } => *parsed,
        }
    }
}
