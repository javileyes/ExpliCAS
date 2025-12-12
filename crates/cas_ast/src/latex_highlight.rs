//! LaTeX expression highlighting support
//!
//! This module provides a way to highlight specific subexpressions in LaTeX output.
//! Used primarily for step-by-step visualization where rule inputs are highlighted
//! in red and outputs in green.
//!
//! Uses the shared LaTeXRenderer trait from latex_core for consistent formatting.

use crate::display_context::DisplayContext;
use crate::latex_core::LaTeXRenderer;
use crate::{Context, Expr, ExprId};
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

    /// Override pow formatting to check for root hints
    fn format_pow(&self, base: ExprId, exp: ExprId) -> String {
        // If exponent is 1, just return the base (no ^{1})
        if let Expr::Number(n) = self.context().get(exp) {
            if n.is_integer() && *n == num_rational::BigRational::from_integer(1.into()) {
                return self.expr_to_latex(base, false);
            }
        }

        // Check if this should be rendered as a root based on hints
        if let Some(hints) = self.get_display_hint(self.root_id()) {
            if let Expr::Number(n) = self.context().get(exp) {
                let denom = n.denom();
                if !n.is_integer() && *denom > 1.into() {
                    // Check if we have a matching root hint
                    for root_idx in hints.root_indices() {
                        if *denom == (root_idx as i64).into() {
                            let base_str = self.expr_to_latex(base, false);
                            let numer = n.numer();

                            if *numer == 1.into() {
                                if *denom == 2.into() {
                                    return format!("\\sqrt{{{}}}", base_str);
                                } else {
                                    return format!("\\sqrt[{}]{{{}}}", denom, base_str);
                                }
                            } else {
                                if *denom == 2.into() {
                                    return format!("\\sqrt{{{{{}}}^{{{}}}}}", base_str, numer);
                                } else {
                                    return format!(
                                        "\\sqrt[{}]{{{{{}}}^{{{}}}}}",
                                        denom, base_str, numer
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }

        // Default power rendering
        let base_str = self.expr_to_latex_base(base);
        let exp_str = self.expr_to_latex(exp, false);
        format!("{{{}}}^{{{}}}", base_str, exp_str)
    }
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
