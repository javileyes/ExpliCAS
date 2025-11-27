pub mod expression;
pub use expression::{Expr, Constant};
use std::rc::Rc;

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
    pub lhs: Rc<Expr>, // Left Hand Side
    pub rhs: Rc<Expr>, // Right Hand Side
    pub op: RelOp,
}
