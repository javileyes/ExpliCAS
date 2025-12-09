pub mod display_context;
pub mod domain;
pub mod expression;
pub mod latex;
pub mod latex_highlight;
pub mod latex_no_roots;
pub mod ordering; // NEW: Canonical ordering utilities
pub mod visitor;

pub use display_context::{DisplayContext, DisplayHint};
pub use domain::{BoundType, Interval, SolutionSet};
pub use expression::{
    Constant, Context, DisplayExpr, DisplayExprWithHints, Expr, ExprId, RawDisplayExpr,
};
pub use latex::LaTeXExpr;
pub use latex_highlight::{HighlightColor, HighlightConfig, LaTeXExprHighlighted};
pub use latex_no_roots::LatexNoRoots;
pub use visitor::{Transformer, Visitor};
// use std::rc::Rc; // Removed Rc usage

#[derive(Debug, Clone, PartialEq)]
pub enum RelOp {
    Eq,  // =
    Neq, // !=
    Lt,  // <
    Gt,  // >
    Leq, // <=
    Geq, // >=
}

impl std::fmt::Display for RelOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RelOp::Eq => write!(f, "="),
            RelOp::Neq => write!(f, "!="),
            RelOp::Lt => write!(f, "<"),
            RelOp::Gt => write!(f, ">"),
            RelOp::Leq => write!(f, "<="),
            RelOp::Geq => write!(f, ">="),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Equation {
    pub lhs: ExprId, // Left Hand Side
    pub rhs: ExprId, // Right Hand Side
    pub op: RelOp,
}
