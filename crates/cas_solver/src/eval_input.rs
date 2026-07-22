//! Input parsing and typed request-building helpers for eval orchestration.

mod build;

pub(crate) use build::build_prepared_eval_request_for_input;
pub(crate) use build::special::parse_dsolve_conditions;

#[derive(Debug, Clone)]
pub enum EvalNonSolveAction {
    Simplify,
    Equiv {
        other: cas_ast::ExprId,
    },
    Limit {
        var: String,
        approach: cas_math::limit_types::Approach,
    },
    Dsolve {
        func: String,
        var: String,
        conditions: Vec<cas_solver_core::eval_models::DsolveCondition>,
    },
    DsolveSystem {
        second_equation: cas_ast::ExprId,
        funcs: Vec<String>,
        var: String,
        conditions: Vec<cas_solver_core::eval_models::DsolveCondition>,
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
