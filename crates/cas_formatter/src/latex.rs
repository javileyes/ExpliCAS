//! LaTeX expression rendering
//!
//! This module provides LaTeX rendering for expressions using the shared
//! LaTeXRenderer trait from latex_core. Two main structures are provided:
//! - LaTeXExpr: Basic LaTeX rendering
//! - LaTeXExprWithHints: LaTeX rendering with display hints (e.g., sqrt notation)

use crate::display_context::DisplayContext;
use crate::latex_core::LaTeXRenderer;
use crate::{Context, ExprId};

// ============================================================================
// LaTeXExpr - Basic LaTeX rendering
// ============================================================================

/// Converts an expression to LaTeX format for rendering with MathJax
///
/// Uses the shared LaTeXRenderer trait for consistent formatting.
pub struct LaTeXExpr<'a> {
    pub context: &'a Context,
    pub id: ExprId,
}

impl<'a> LaTeXRenderer for LaTeXExpr<'a> {
    fn context(&self) -> &Context {
        self.context
    }

    fn root_id(&self) -> ExprId {
        self.id
    }
}

impl<'a> LaTeXExpr<'a> {
    /// Generate LaTeX string
    pub fn to_latex(&self) -> String {
        LaTeXRenderer::to_latex(self)
    }
}

// ============================================================================
// LaTeXExprStyled - LaTeX rendering with global style preferences
// ============================================================================

/// LaTeX expression renderer that respects global style preferences.
///
/// This is the LaTeX sibling of `DisplayExprStyled`: it lets callers preserve
/// user notation such as `x^(1/2)` instead of forcing `\sqrt{x}`.
pub struct LaTeXExprStyled<'a> {
    pub context: &'a Context,
    pub id: ExprId,
    pub style_prefs: &'a crate::root_style::StylePreferences,
}

impl<'a> LaTeXRenderer for LaTeXExprStyled<'a> {
    fn context(&self) -> &Context {
        self.context
    }

    fn root_id(&self) -> ExprId {
        self.id
    }

    fn get_style_prefs(&self) -> Option<&crate::root_style::StylePreferences> {
        Some(self.style_prefs)
    }
}

impl<'a> LaTeXExprStyled<'a> {
    /// Generate LaTeX string with style preferences applied.
    pub fn to_latex(&self) -> String {
        LaTeXRenderer::to_latex(self)
    }
}

// ============================================================================
// LaTeXExprWithHints - LaTeX rendering with display hints
// ============================================================================

/// LaTeX expression renderer that respects display hints
///
/// Similar to LaTeXExpr but checks DisplayContext for AsRoot hints
/// to render fractional powers as roots when appropriate.
pub struct LaTeXExprWithHints<'a> {
    pub context: &'a Context,
    pub id: ExprId,
    pub hints: &'a DisplayContext,
}

impl<'a> LaTeXRenderer for LaTeXExprWithHints<'a> {
    fn context(&self) -> &Context {
        self.context
    }

    fn root_id(&self) -> ExprId {
        self.id
    }

    fn get_display_hint(&self, _id: ExprId) -> Option<&DisplayContext> {
        Some(self.hints)
    }
    // V2.14.40: format_pow is now handled by the trait default, which renders
    // fractional powers as roots automatically
}

impl<'a> LaTeXExprWithHints<'a> {
    /// Generate LaTeX string with hints applied
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
    fn test_latex_basic() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let expr = ctx.add(Expr::Mul(two, x));

        let latex = LaTeXExpr {
            context: &ctx,
            id: expr,
        };
        // Now uses cdot for all multiplication
        assert_eq!(latex.to_latex(), "2\\cdot x");
    }

    #[test]
    fn test_latex_sqrt() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let sqrt = ctx.call("sqrt", vec![x]);

        let latex = LaTeXExpr {
            context: &ctx,
            id: sqrt,
        };
        assert_eq!(latex.to_latex(), "\\sqrt{x}");
    }

    #[test]
    fn test_latex_fraction() {
        let mut ctx = Context::new();
        let one = ctx.num(1);
        let two = ctx.num(2);
        let frac = ctx.add(Expr::Div(one, two));

        let latex = LaTeXExpr {
            context: &ctx,
            id: frac,
        };
        assert_eq!(latex.to_latex(), "\\frac{1}{2}");
    }

    #[test]
    fn test_latex_power() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let power = ctx.add(Expr::Pow(x, two));

        let latex = LaTeXExpr {
            context: &ctx,
            id: power,
        };
        assert_eq!(latex.to_latex(), "{x}^{2}");
    }

    #[test]
    fn test_latex_sqrt_from_power() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let half = ctx.rational(1, 2);
        let sqrt_expr = ctx.add(Expr::Pow(x, half));

        // Now renders as sqrt by default (V2.14.40: consistent root display)
        let latex = LaTeXExpr {
            context: &ctx,
            id: sqrt_expr,
        };
        assert_eq!(latex.to_latex(), "\\sqrt{x}");

        // With hints is still consistent
        let hints = DisplayContext::with_root_index(2);
        let latex_with_hints = LaTeXExprWithHints {
            context: &ctx,
            id: sqrt_expr,
            hints: &hints,
        };
        assert_eq!(latex_with_hints.to_latex(), "\\sqrt{x}");
    }

    #[test]
    fn test_latex_styled_preserves_fractional_power_style() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let half = ctx.rational(1, 2);
        let sqrt_expr = ctx.add(Expr::Pow(x, half));

        let style = crate::root_style::StylePreferences::with_root_style(
            crate::root_style::RootStyle::Exponential,
        );
        let latex = LaTeXExprStyled {
            context: &ctx,
            id: sqrt_expr,
            style_prefs: &style,
        };
        assert_eq!(latex.to_latex(), "{x}^{\\frac{1}{2}}");
    }

    #[test]
    fn test_latex_nth_root() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let third = ctx.rational(1, 3);
        let cbrt = ctx.add(Expr::Pow(x, third));

        let hints = DisplayContext::with_root_index(3);
        let latex = LaTeXExprWithHints {
            context: &ctx,
            id: cbrt,
            hints: &hints,
        };
        assert_eq!(latex.to_latex(), "\\sqrt[3]{x}");
    }

    #[test]
    fn test_latex_fractional_power() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two_thirds = ctx.rational(2, 3);
        let power = ctx.add(Expr::Pow(x, two_thirds));

        let hints = DisplayContext::with_root_index(3);
        let latex = LaTeXExprWithHints {
            context: &ctx,
            id: power,
            hints: &hints,
        };
        assert_eq!(latex.to_latex(), "\\sqrt[3]{{x}^{2}}");
    }

    #[test]
    fn test_latex_negative_fraction() {
        let mut ctx = Context::new();
        let neg_half = ctx.rational(-1, 2);

        let latex = LaTeXExpr {
            context: &ctx,
            id: neg_half,
        };
        // Negative sign should be outside fraction
        assert_eq!(latex.to_latex(), "-\\frac{1}{2}");
    }

    #[test]
    fn test_latex_subtraction() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let sub = ctx.add(Expr::Sub(x, y));

        let latex = LaTeXExpr {
            context: &ctx,
            id: sub,
        };
        assert_eq!(latex.to_latex(), "x - y");
    }

    #[test]
    fn test_latex_sum() {
        let mut ctx = Context::new();
        let k = ctx.var("k");
        let one = ctx.num(1);
        let n = ctx.var("n");
        let sum = ctx.call("sum", vec![k, k, one, n]);

        let latex = LaTeXExpr {
            context: &ctx,
            id: sum,
        };
        assert_eq!(latex.to_latex(), "\\sum_{k=1}^{n} k");
    }

    #[test]
    fn test_latex_product() {
        let mut ctx = Context::new();
        let k = ctx.var("k");
        let one = ctx.num(1);
        let n = ctx.var("n");
        let prod = ctx.call("product", vec![k, k, one, n]);

        let latex = LaTeXExpr {
            context: &ctx,
            id: prod,
        };
        assert_eq!(latex.to_latex(), "\\prod_{k=1}^{n} k");
    }

    // =========================================================================
    // Regression tests for parentheses preservation (Issue: LaTeX omits parens)
    // =========================================================================

    #[test]
    fn test_latex_sub_with_add_rhs() {
        // A - (B + C) must show parentheses, not A - B + C
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");
        let c = ctx.var("c");
        let b_plus_c = ctx.add(Expr::Add(b, c));
        let expr = ctx.add(Expr::Sub(a, b_plus_c));

        let latex = LaTeXExpr {
            context: &ctx,
            id: expr,
        };
        assert_eq!(latex.to_latex(), "a - (b + c)");
    }

    #[test]
    fn test_latex_sub_with_sub_rhs() {
        // A - (B - C) must show parentheses, not A - B - C
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");
        let c = ctx.var("c");
        let b_minus_c = ctx.add(Expr::Sub(b, c));
        let expr = ctx.add(Expr::Sub(a, b_minus_c));

        let latex = LaTeXExpr {
            context: &ctx,
            id: expr,
        };
        assert_eq!(latex.to_latex(), "a - (b - c)");
    }

    #[test]
    fn test_latex_sub_with_complex_add_rhs() {
        // (A + B) - (C + D) must preserve both groups
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");
        let c = ctx.var("c");
        let d = ctx.var("d");
        let a_plus_b = ctx.add(Expr::Add(a, b));
        let c_plus_d = ctx.add(Expr::Add(c, d));
        let expr = ctx.add(Expr::Sub(a_plus_b, c_plus_d));

        let latex = LaTeXExpr {
            context: &ctx,
            id: expr,
        };
        assert_eq!(latex.to_latex(), "a + b - (c + d)");
    }

    #[test]
    fn test_latex_add_renders_negative_mul_factor_as_subtraction() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");
        let c = ctx.var("c");
        let three = ctx.num(3);
        let a3 = ctx.add(Expr::Pow(a, three));
        let neg_c = ctx.add(Expr::Neg(c));
        let left = ctx.add(Expr::Mul(a3, b));
        let right = ctx.add(Expr::Mul(a3, neg_c));
        let expr = ctx.add(Expr::Add(left, right));

        let latex = LaTeXExpr {
            context: &ctx,
            id: expr,
        };
        assert_eq!(latex.to_latex(), "b\\cdot {a}^{3} - c\\cdot {a}^{3}");
    }

    #[test]
    fn test_latex_mul_with_negative_factor_lifts_sign_to_product() {
        let mut ctx = Context::new();
        let y = ctx.var("y");
        let two = ctx.num(2);
        let neg_y = ctx.add(Expr::Neg(y));
        let expr = ctx.add(Expr::Mul(two, neg_y));

        let latex = LaTeXExpr {
            context: &ctx,
            id: expr,
        };
        assert_eq!(latex.to_latex(), "-2\\cdot y");
    }

    #[test]
    fn test_latex_sub_does_not_parenthesize_simple_product_rhs() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let two = ctx.num(2);
        let x_sq = ctx.add(Expr::Pow(x, two));
        let y_sq = ctx.add(Expr::Pow(y, two));
        let two_y_sq = ctx.add(Expr::Mul(two, y_sq));
        let two_x = ctx.add(Expr::Mul(two, x));
        let two_x_y = ctx.add(Expr::Mul(two_x, y));
        let lhs = ctx.add(Expr::Add(x_sq, two_y_sq));
        let expr = ctx.add(Expr::Sub(lhs, two_x_y));

        let latex = LaTeXExpr {
            context: &ctx,
            id: expr,
        };
        assert_eq!(
            latex.to_latex(),
            "{x}^{2} + 2\\cdot {y}^{2} - 2\\cdot x\\cdot y"
        );
    }
}
