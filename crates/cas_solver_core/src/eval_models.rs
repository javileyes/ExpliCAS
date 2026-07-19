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
    /// Solve an elementary ODE for `func(var)` (Fase 4). The parsed equation
    /// travels as the request expression; conditions stay textual (their heads
    /// like `y(0)` are not parseable expressions).
    Dsolve {
        func: String,
        var: String,
        conditions: Vec<String>,
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
    /// For special commands whose natural output is textual, not an expression.
    Text {
        plain: String,
        latex: Option<String>,
    },
    /// For commands with no output.
    None,
}
