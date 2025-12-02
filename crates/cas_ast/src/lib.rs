pub mod expression;
pub mod domain;
pub mod visitor;

pub use expression::{Expr, Constant, ExprId, Context, DisplayExpr, RawDisplayExpr};
pub use visitor::{Visitor, Transformer};
pub use domain::{Interval, BoundType, SolutionSet};
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
