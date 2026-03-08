//! Shared eval request/action/result models.

use cas_ast::ExprId;

#[derive(Clone, Debug)]
pub enum EvalAction {
    Simplify,
    Expand,
    /// Solve for a variable.
    Solve {
        var: String,
    },
    /// Check equivalence between two expressions.
    Equiv {
        other: ExprId,
    },
    /// Compute limit as variable approaches a value.
    Limit {
        var: String,
        approach: cas_math::limit_types::Approach,
    },
}

#[derive(Clone, Debug)]
pub struct EvalRequest {
    pub raw_input: String,
    pub parsed: ExprId,
    pub action: EvalAction,
    pub auto_store: bool,
}

#[derive(Clone, Debug)]
pub enum EvalResult {
    Expr(ExprId),
    /// For solve multiple roots (legacy).
    Set(Vec<ExprId>),
    /// Full solution set including conditional branches.
    SolutionSet(cas_ast::SolutionSet),
    /// For equivalence checks.
    Bool(bool),
    /// For commands with no output.
    None,
}
