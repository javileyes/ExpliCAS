//! Formatting facade for AST rendering concerns.
//!
//! This crate centralizes display/LaTeX APIs so callers can avoid importing
//! presentation modules directly from `cas_ast`.

pub mod display {
    pub use cas_ast::display::*;
}

pub mod display_transforms {
    pub use cas_ast::display_transforms::*;
}

pub mod latex {
    pub use cas_ast::latex::*;
}

pub mod latex_core {
    pub use cas_ast::latex_core::*;
}

pub mod latex_highlight {
    pub use cas_ast::latex_highlight::*;
}

pub mod latex_no_roots {
    pub use cas_ast::latex_no_roots::*;
}

pub use cas_ast::{
    DisplayExpr, DisplayExprStyled, DisplayExprWithHints, LaTeXExpr, LaTeXExprHighlighted,
    LaTeXExprHighlightedWithHints, LaTeXExprWithHints, LatexNoRoots, PathHighlightConfig,
    PathHighlightedLatexRenderer, RawDisplayExpr,
};
