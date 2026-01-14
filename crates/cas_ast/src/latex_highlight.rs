//! LaTeX expression highlighting support
//!
//! This module provides a way to highlight specific subexpressions in LaTeX output.
//! Used primarily for step-by-step visualization where rule inputs are highlighted
//! in red and outputs in green.
//!
//! Uses the shared LaTeXRenderer trait from latex_core for consistent formatting.

use crate::display_context::DisplayContext;
use crate::latex_core::LaTeXRenderer;
use crate::{Context, ExprId};
use std::collections::HashMap;

// ============================================================================
// Highlight Configuration
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HighlightColor {
    /// Rule input (before transformation)
    Red,
    /// Rule output (after transformation)
    Green,
    /// Optional intermediate highlight
    Blue,
}

impl HighlightColor {
    /// Get the LaTeX color name
    pub fn to_latex(&self) -> &'static str {
        match self {
            HighlightColor::Red => "red",
            HighlightColor::Green => "green",
            HighlightColor::Blue => "blue",
        }
    }
}

/// Configuration for highlighting specific expressions
#[derive(Default)]
pub struct HighlightConfig {
    highlights: HashMap<ExprId, HighlightColor>,
}

impl HighlightConfig {
    /// Create empty highlight configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a highlight for an expression
    pub fn add(&mut self, id: ExprId, color: HighlightColor) -> &mut Self {
        self.highlights.insert(id, color);
        self
    }

    /// Get the highlight color for an expression, if any
    pub fn get(&self, id: ExprId) -> Option<HighlightColor> {
        self.highlights.get(&id).copied()
    }

    /// Check if config is empty
    pub fn is_empty(&self) -> bool {
        self.highlights.is_empty()
    }
}

// ============================================================================
// PathHighlightConfig - Path-based highlighting (V2.9.16)
// ============================================================================

use crate::expr_path::ExprPath;

/// Configuration for highlighting specific occurrences by path.
///
/// Unlike `HighlightConfig` which uses ExprId (causing all identical values
/// to be highlighted), this uses tree paths to highlight exactly one occurrence.
#[derive(Default)]
pub struct PathHighlightConfig {
    /// List of (path, color) pairs. First match wins.
    paths: Vec<(ExprPath, HighlightColor)>,
}

impl PathHighlightConfig {
    /// Create empty path highlight configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a highlight for a specific path
    pub fn add(&mut self, path: ExprPath, color: HighlightColor) -> &mut Self {
        self.paths.push((path, color));
        self
    }

    /// Get the highlight color for a path, if any
    pub fn get(&self, path: &ExprPath) -> Option<HighlightColor> {
        for (p, color) in &self.paths {
            if p == path {
                return Some(*color);
            }
        }
        None
    }

    /// Check if config is empty
    pub fn is_empty(&self) -> bool {
        self.paths.is_empty()
    }
}

// ============================================================================
// LaTeXExprHighlighted - LaTeX with highlighting
// ============================================================================

/// LaTeX expression with optional highlighting
///
/// Wraps expression formatting and applies color to highlighted subexpressions.
pub struct LaTeXExprHighlighted<'a> {
    pub context: &'a Context,
    pub id: ExprId,
    pub highlights: &'a HighlightConfig,
}

impl<'a> LaTeXRenderer for LaTeXExprHighlighted<'a> {
    fn context(&self) -> &Context {
        self.context
    }

    fn root_id(&self) -> ExprId {
        self.id
    }

    fn get_highlight(&self, id: ExprId) -> Option<HighlightColor> {
        self.highlights.get(id)
    }
}

impl<'a> LaTeXExprHighlighted<'a> {
    /// Generate LaTeX string with highlights applied
    pub fn to_latex(&self) -> String {
        LaTeXRenderer::to_latex(self)
    }
}

// ============================================================================
// LaTeXExprHighlightedWithHints - LaTeX with highlighting AND display hints
// ============================================================================

/// LaTeX expression with optional highlighting AND display hints support
///
/// Combines LaTeXExprHighlighted with DisplayHint awareness for sqrt notation.
pub struct LaTeXExprHighlightedWithHints<'a> {
    pub context: &'a Context,
    pub id: ExprId,
    pub highlights: &'a HighlightConfig,
    pub hints: &'a DisplayContext,
}

impl<'a> LaTeXRenderer for LaTeXExprHighlightedWithHints<'a> {
    fn context(&self) -> &Context {
        self.context
    }

    fn root_id(&self) -> ExprId {
        self.id
    }

    fn get_highlight(&self, id: ExprId) -> Option<HighlightColor> {
        self.highlights.get(id)
    }

    fn get_display_hint(&self, _id: ExprId) -> Option<&DisplayContext> {
        Some(self.hints)
    }
    // V2.14.40: format_pow is now handled by the trait default, which renders
    // fractional powers as roots automatically
}

impl<'a> LaTeXExprHighlightedWithHints<'a> {
    /// Generate LaTeX string with highlights and display hints applied
    pub fn to_latex(&self) -> String {
        LaTeXRenderer::to_latex(self)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Context, Expr};

    #[test]
    fn test_no_highlights() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let sum = ctx.add(Expr::Add(x, y));

        let config = HighlightConfig::new();
        let latex = LaTeXExprHighlighted {
            context: &ctx,
            id: sum,
            highlights: &config,
        };
        assert_eq!(latex.to_latex(), "x + y");
    }

    #[test]
    fn test_red_highlight() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let sum = ctx.add(Expr::Add(x, y));

        let mut config = HighlightConfig::new();
        config.add(x, HighlightColor::Red);

        let latex = LaTeXExprHighlighted {
            context: &ctx,
            id: sum,
            highlights: &config,
        };
        assert_eq!(latex.to_latex(), "{\\color{red}{x}} + y");
    }

    #[test]
    fn test_nested_highlight() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let mul = ctx.add(Expr::Mul(two, x));
        let y = ctx.var("y");
        let sum = ctx.add(Expr::Add(mul, y));

        let mut config = HighlightConfig::new();
        config.add(mul, HighlightColor::Green);

        let latex = LaTeXExprHighlighted {
            context: &ctx,
            id: sum,
            highlights: &config,
        };
        // Note: canonicalization may reorder operands
        assert!(latex.to_latex().contains("{\\color{green}{2\\cdot x}}"));
    }

    #[test]
    fn test_subtraction_with_highlight() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let sub = ctx.add(Expr::Sub(x, y));

        let mut config = HighlightConfig::new();
        config.add(y, HighlightColor::Red);

        let latex = LaTeXExprHighlighted {
            context: &ctx,
            id: sub,
            highlights: &config,
        };
        assert_eq!(latex.to_latex(), "x - {\\color{red}{y}}");
    }

    #[test]
    fn test_multiplication_uses_cdot() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");
        let mul = ctx.add(Expr::Mul(a, b));

        let config = HighlightConfig::new();
        let latex = LaTeXExprHighlighted {
            context: &ctx,
            id: mul,
            highlights: &config,
        };
        // Multiplication always uses cdot
        assert_eq!(latex.to_latex(), "a\\cdot b");
    }

    #[test]
    fn test_negative_fraction() {
        let mut ctx = Context::new();
        let neg_half = ctx.rational(-1, 2);

        let config = HighlightConfig::new();
        let latex = LaTeXExprHighlighted {
            context: &ctx,
            id: neg_half,
            highlights: &config,
        };
        // Negative sign should be outside fraction
        assert_eq!(latex.to_latex(), "-\\frac{1}{2}");
    }

    #[test]
    fn test_with_hints_sqrt() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let half = ctx.rational(1, 2);
        let sqrt_expr = ctx.add(Expr::Pow(x, half));

        let config = HighlightConfig::new();
        let hints = DisplayContext::with_root_index(2);

        let latex = LaTeXExprHighlightedWithHints {
            context: &ctx,
            id: sqrt_expr,
            highlights: &config,
            hints: &hints,
        };
        assert_eq!(latex.to_latex(), "\\sqrt{x}");
    }

    #[test]
    fn test_sum_rendering() {
        let mut ctx = Context::new();
        let k = ctx.var("k");
        let one = ctx.num(1);
        let n = ctx.var("n");
        let sum = ctx.add(Expr::Function("sum".to_string(), vec![k, k, one, n]));

        let config = HighlightConfig::new();
        let latex = LaTeXExprHighlighted {
            context: &ctx,
            id: sum,
            highlights: &config,
        };
        assert_eq!(latex.to_latex(), "\\sum_{k=1}^{n} k");
    }
}
