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
    /// travels as the request expression; conditions arrive pre-parsed (their
    /// textual heads like `y(0)` never reach the expression parser — the
    /// solver layer splits them and parses point/value separately).
    Dsolve {
        func: String,
        var: String,
        conditions: Vec<DsolveCondition>,
    },
}

/// One dsolve initial condition `y(point) = value` (order 0) or
/// `y'(point) = value` (order 1).
#[derive(Clone, Copy, Debug)]
pub struct DsolveCondition {
    pub point: ExprId,
    pub value: ExprId,
    pub order: usize,
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
