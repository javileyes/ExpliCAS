pub mod builtin; // Builtin function identifiers for O(1) comparison
#[path = "../../cas_formatter/src/display/mod.rs"]
pub mod display; // Temporary compatibility path during formatter extraction
#[path = "../../cas_formatter/src/display_context.rs"]
pub mod display_context; // Temporary compatibility path during formatter extraction
#[path = "../../cas_formatter/src/display_transforms.rs"]
pub mod display_transforms; // Temporary compatibility path during formatter extraction
pub mod domain;
pub mod eq; // __eq__ equation wrapper utilities (canonical implementation)
pub mod error; // Error types for AST operations
pub mod expr_path; // Path-based occurrence identification (V2.9.16)
pub mod expression;
pub mod hold; // __hold barrier utilities (canonical implementation)
#[path = "../../cas_formatter/src/latex.rs"]
pub mod latex;
#[path = "../../cas_formatter/src/latex_core.rs"]
pub mod latex_core;
#[path = "../../cas_formatter/src/latex_highlight.rs"]
pub mod latex_highlight;
#[path = "../../cas_formatter/src/latex_no_roots.rs"]
pub mod latex_no_roots;
pub mod ordering; // Canonical ordering utilities
#[path = "../../cas_formatter/src/root_style.rs"]
pub mod root_style; // Temporary compatibility path during formatter extraction
pub mod span; // Canonical source span for error reporting
pub mod symbol; // Symbol interning for variable names
pub mod traversal; // Canonical traversal utilities (count_nodes, etc.)
pub mod views; // Unified views for pattern matching
pub mod visitor;

pub use builtin::{BuiltinFn, BuiltinIds, ALL_BUILTINS};

pub use display::{DisplayExpr, DisplayExprStyled, DisplayExprWithHints, RawDisplayExpr};
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
pub use root_style::{detect_root_style, ParseStyleSignals, RootStyle, StylePreferences};
// Span re-export (canonical source location)
pub use span::Span;
// Traversal re-exports (canonical implementations)
pub use traversal::{
    collect_variables, count_all_nodes, count_nodes_and_max_depth, count_nodes_matching,
    substitute_expr_by_id,
};
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
