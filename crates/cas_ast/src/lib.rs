pub mod builtin; // Builtin function identifiers for O(1) comparison
pub mod display; // NEW: Display formatting
pub mod display_context;
pub mod display_transforms; // Scoped display transforms (sqrt in quadratic, etc.)
pub mod domain;
pub mod eq; // __eq__ equation wrapper utilities (canonical implementation)
pub mod error; // Error types for AST operations
pub mod expr_path; // Path-based occurrence identification (V2.9.16)
pub mod expression;
pub mod hold; // __hold barrier utilities (canonical implementation)
pub mod latex;
pub mod latex_core;
pub mod latex_highlight;
pub mod latex_no_roots;
pub mod latex_parser;
pub mod ordering; // Canonical ordering utilities
pub mod root_style; // Style Sniffing for root notation
pub mod span; // Canonical source span for error reporting
pub mod symbol; // Symbol interning for variable names
pub mod traversal; // Canonical traversal utilities (count_nodes, etc.)
pub mod views; // Unified views for pattern matching
pub mod visitor;

pub use builtin::{BuiltinFn, BuiltinIds, ALL_BUILTINS};

pub use display::{DisplayExpr, DisplayExprStyled, DisplayExprWithHints, RawDisplayExpr};
pub use display_context::{DisplayContext, DisplayHint};
pub use domain::{
    BoundType, Case, ConditionPredicate, ConditionSet, Interval, SolutionSet, SolveResult,
};
pub use expr_path::{path_to_string, ExprPath};
pub use expression::{Constant, Context, ContextStats, Expr, ExprId, MulCommutativity};
pub use latex::{LaTeXExpr, LaTeXExprWithHints};
pub use latex_core::PathHighlightedLatexRenderer;
pub use latex_highlight::{
    HighlightColor, HighlightConfig, LaTeXExprHighlighted, LaTeXExprHighlightedWithHints,
    PathHighlightConfig,
};
pub use latex_no_roots::LatexNoRoots;
pub use latex_parser::parse_latex;
pub use root_style::{
    detect_root_style, ParseStyleSignals, RootStyle, StylePreferences, StyledExpr,
};
// Span re-export (canonical source location)
pub use span::Span;
// Traversal re-exports (canonical implementations)
pub use traversal::{count_all_nodes, count_nodes_and_max_depth, count_nodes_matching};
// Legacy alias for backward compatibility
pub use traversal::count_all_nodes as count_nodes;
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
