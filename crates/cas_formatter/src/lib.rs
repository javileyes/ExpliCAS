//! Formatting facade for AST rendering concerns.
//!
//! This crate centralizes display/LaTeX APIs so callers can avoid importing
//! presentation modules directly from `cas_ast`.

pub use cas_ast::{eq, expr_path, hold, ordering, views};
pub use cas_ast::{Constant, Context, Expr, ExprId};

pub mod display;
pub mod display_context;
pub mod display_transforms;
pub mod latex;
pub mod latex_core;
pub mod latex_highlight;
pub mod latex_no_roots;
pub mod root_style;

pub use display::{DisplayExpr, DisplayExprStyled, DisplayExprWithHints, RawDisplayExpr};
pub use display_context::{DisplayContext, DisplayHint};
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
pub use root_style::{detect_root_style, ParseStyleSignals, RootStyle, StylePreferences};
