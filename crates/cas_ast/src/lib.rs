pub mod builtin; // Builtin function identifiers for O(1) comparison
pub mod domain;
pub mod eq; // __eq__ equation wrapper utilities (canonical implementation)
pub mod error; // Error types for AST operations
pub mod expr_path; // Path-based occurrence identification (V2.9.16)
pub mod expression;
pub mod hold; // __hold barrier utilities (canonical implementation)
pub mod ordering; // Canonical ordering utilities
pub mod span; // Canonical source span for error reporting
pub mod symbol; // Symbol interning for variable names
pub mod target_kind; // Expr discriminant mapping for rule dispatch
pub mod traversal; // Canonical traversal utilities (count_nodes, etc.)
pub mod views; // Unified views for pattern matching
pub mod visitor;
pub mod visitors; // Common concrete visitors (depth, variables)

pub use builtin::{BuiltinFn, BuiltinIds, ALL_BUILTINS};

pub use domain::{
    BoundType, Case, ConditionPredicate, ConditionSet, Interval, SolutionSet, SolveResult,
};
pub use expr_path::{path_to_string, ExprPath};
pub use expression::{Constant, Context, ContextStats, Expr, ExprId, MulCommutativity};
// Span re-export (canonical source location)
pub use span::Span;
pub use target_kind::{TargetKind, TargetKindSet};
// Traversal re-exports (canonical implementations)
pub use traversal::{
    collect_variables, count_all_nodes, count_nodes_and_max_depth, count_nodes_matching,
    substitute_expr_by_id,
};
// Legacy alias for backward compatibility
pub use traversal::count_all_nodes as count_nodes;
pub use visitor::{Transformer, Visitor};
pub use visitors::{DepthVisitor, VariableCollector};

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
