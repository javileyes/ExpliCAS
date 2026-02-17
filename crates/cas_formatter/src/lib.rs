//! Formatting facade for AST rendering concerns.
//!
//! This crate centralizes display/LaTeX APIs so callers can avoid importing
//! presentation modules directly from `cas_ast`.

pub use cas_ast::{display_context, eq, expr_path, hold, ordering, root_style, views};
pub use cas_ast::{Constant, Context, Expr, ExprId};

pub mod display;
pub mod display_transforms;
pub mod latex;
pub mod latex_core;
pub mod latex_highlight;
pub mod latex_no_roots;

pub use display::{DisplayExpr, DisplayExprStyled, DisplayExprWithHints, RawDisplayExpr};
pub use display_transforms::{
    DisplayTransform, DisplayTransformRegistry, ScopeTag, ScopedRenderer,
};
pub use latex::{LaTeXExpr, LaTeXExprWithHints};
pub use latex_core::PathHighlightedLatexRenderer;
pub use latex_highlight::{
    HighlightColor, HighlightConfig, LaTeXExprHighlighted, LaTeXExprHighlightedWithHints,
    PathHighlightConfig,
};
pub use latex_no_roots::LatexNoRoots;
