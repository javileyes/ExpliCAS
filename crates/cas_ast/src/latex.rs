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

    /// Override pow formatting to check for root hints
    fn format_pow(&self, base: ExprId, exp: ExprId) -> String {
        use crate::Expr;

        // If exponent is 1, just return the base (no ^{1})
        if let Expr::Number(n) = self.context().get(exp) {
            if n.is_integer() && *n == num_rational::BigRational::from_integer(1.into()) {
                return self.expr_to_latex(base, false);
            }
        }

        // Check if this should be rendered as a root based on hints
        if let Some(hints) = self.get_display_hint(self.root_id()) {
            if let Expr::Number(n) = self.context().get(exp) {
                // Check if this power matches a root hint
                let denom = n.denom();
                if !n.is_integer() && *denom > 1.into() {
                    // Check if we have a matching root hint
                    for root_idx in hints.root_indices() {
                        if *denom == (root_idx as i64).into() {
                            let base_str = self.expr_to_latex(base, false);
                            let numer = n.numer();

                            if *numer == 1.into() {
                                // Simple root: x^(1/n) -> nth root of x
                                if *denom == 2.into() {
                                    return format!("\\sqrt{{{}}}", base_str);
                                } else {
                                    return format!("\\sqrt[{}]{{{}}}", denom, base_str);
                                }
                            } else {
                                // Complex root: x^(k/n) -> nth root of x^k
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
        let sqrt = ctx.add(Expr::Function("sqrt".to_string(), vec![x]));

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

        // Without hints, renders as power
        let latex = LaTeXExpr {
            context: &ctx,
            id: sqrt_expr,
        };
        assert_eq!(latex.to_latex(), "{x}^{\\frac{1}{2}}");

        // With hints containing root index 2, renders as sqrt
        let hints = DisplayContext::with_root_index(2);
        let latex_with_hints = LaTeXExprWithHints {
            context: &ctx,
            id: sqrt_expr,
            hints: &hints,
        };
        assert_eq!(latex_with_hints.to_latex(), "\\sqrt{x}");
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
        let sum = ctx.add(Expr::Function("sum".to_string(), vec![k, k, one, n]));

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
        let prod = ctx.add(Expr::Function("product".to_string(), vec![k, k, one, n]));

        let latex = LaTeXExpr {
            context: &ctx,
            id: prod,
        };
        assert_eq!(latex.to_latex(), "\\prod_{k=1}^{n} k");
    }
}
