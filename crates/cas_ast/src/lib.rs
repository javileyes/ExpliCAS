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

#[derive(Debug, Clone, PartialEq)]
pub struct Equation {
    pub lhs: Rc<Expr>, // Left Hand Side
    pub rhs: Rc<Expr>, // Right Hand Side
    pub op: RelOp,
}
