//! Input parsing and typed request-building helpers for eval orchestration.

mod build;

pub use build::build_prepared_eval_request_for_input;

#[derive(Debug, Clone)]
pub enum EvalNonSolveAction {
    Simplify,
    Limit {
        var: String,
        approach: cas_math::limit_types::Approach,
    },
}

#[derive(Debug, Clone)]
pub enum PreparedEvalRequest {
    Solve {
        raw_input: String,
        parsed: cas_ast::ExprId,
        original_equation: Option<cas_ast::Equation>,
        var: String,
        auto_store: bool,
    },
    SolveSystem {
        parsed_anchor: cas_ast::ExprId,
        exprs: Vec<cas_ast::ExprId>,
        vars: Vec<String>,
    },
    Derive {
        raw_input: String,
        parsed: cas_ast::ExprId,
        target: cas_ast::ExprId,
        auto_store: bool,
    },
    Eval {
        raw_input: String,
        parsed: cas_ast::ExprId,
        action: EvalNonSolveAction,
        auto_store: bool,
    },
}

impl PreparedEvalRequest {
    pub fn parsed(&self) -> cas_ast::ExprId {
        match self {
            Self::Solve { parsed, .. } => *parsed,
            Self::SolveSystem { parsed_anchor, .. } => *parsed_anchor,
            Self::Derive { parsed, .. } => *parsed,
            Self::Eval { parsed, .. } => *parsed,
        }
    }
}
