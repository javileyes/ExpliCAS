//! LaTeX expression highlighting support
//!
//! This module provides a way to highlight specific subexpressions in LaTeX output.
//! Used primarily for step-by-step visualization where rule inputs are highlighted
//! in red and outputs in green.

use crate::{Constant, Context, Expr, ExprId};
use num_traits::Signed;
use std::collections::HashMap;

/// Color for LaTeX highlighting
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

/// LaTeX expression with optional highlighting
///
/// Wraps expression formatting and applies color to highlighted subexpressions.
pub struct LaTeXExprHighlighted<'a> {
    pub context: &'a Context,
    pub id: ExprId,
    pub highlights: &'a HighlightConfig,
}

impl<'a> LaTeXExprHighlighted<'a> {
    /// Generate LaTeX string with highlights applied
    pub fn to_latex(&self) -> String {
        let latex = self.expr_to_latex_internal(self.id, false);
        Self::clean_latex_negatives(&latex)
    }

    /// Post-process LaTeX to fix negative sign patterns
    fn clean_latex_negatives(latex: &str) -> String {
        let mut result = latex.to_string();
        result = result.replace("+ -\\", "- \\");
        result = result.replace("+ -{", "- {");
        result = result.replace("+ -(", "- (");
        result = result.replace("- -\\", "+ \\");
        result = result.replace("- -{", "+ {");
        result = result.replace("- -(", "+ (");
        result
    }

    fn expr_to_latex_internal(&self, id: ExprId, parent_needs_parens: bool) -> String {
        // Format the expression
        let inner = self.format_expr(id, parent_needs_parens);

        // Apply highlighting if this expression is marked
        if let Some(color) = self.highlights.get(id) {
            format!("{{\\color{{{}}}{{{}}}}}", color.to_latex(), inner)
        } else {
            inner
        }
    }

    fn format_expr(&self, id: ExprId, parent_needs_parens: bool) -> String {
        match self.context.get(id) {
            Expr::Number(n) => {
                if n.is_integer() {
                    format!("{}", n.numer())
                } else {
                    format!("\\frac{{{}}}{{{}}}", n.numer(), n.denom())
                }
            }
            Expr::Variable(name) => name.clone(),
            Expr::Constant(c) => match c {
                Constant::Pi => "\\pi".to_string(),
                Constant::E => "e".to_string(),
                Constant::Infinity => "\\infty".to_string(),
                Constant::Undefined => "\\text{undefined}".to_string(),
            },
            Expr::Add(l, r) => {
                let left = self.expr_to_latex_internal(*l, false);

                // Check if right side is negative
                let (is_negative, right_str) = match self.context.get(*r) {
                    Expr::Number(n) if n.is_negative() => {
                        let positive = -n;
                        let positive_str = if positive.is_integer() {
                            format!("{}", positive.numer())
                        } else {
                            format!("\\frac{{{}}}{{{}}}", positive.numer(), positive.denom())
                        };
                        (true, positive_str)
                    }
                    Expr::Neg(inner) => (true, self.expr_to_latex_internal(*inner, true)),
                    _ => (false, self.expr_to_latex_internal(*r, false)),
                };

                if is_negative {
                    format!("{} - {}", left, right_str)
                } else {
                    format!("{} + {}", left, right_str)
                }
            }
            Expr::Sub(l, r) => {
                let left = self.expr_to_latex_internal(*l, false);
                let right = self.expr_to_latex_internal(*r, true);
                format!("{} - {}", left, right)
            }
            Expr::Mul(l, r) => {
                let left = self.expr_to_latex_mul(*l);
                let right = self.expr_to_latex_mul(*r);

                let needs_cdot = self.needs_explicit_mult(*l, *r);
                if needs_cdot {
                    if parent_needs_parens {
                        format!("({}\\cdot {})", left, right)
                    } else {
                        format!("{}\\cdot {}", left, right)
                    }
                } else if parent_needs_parens {
                    format!("({}{})", left, right)
                } else {
                    format!("{}{}", left, right)
                }
            }
            Expr::Div(l, r) => {
                let numer = self.expr_to_latex_internal(*l, false);
                let denom = self.expr_to_latex_internal(*r, false);
                format!("\\frac{{{}}}{{{}}}", numer, denom)
            }
            Expr::Pow(base, exp) => {
                let base_str = self.expr_to_latex_base(*base);
                let exp_str = self.expr_to_latex_internal(*exp, false);
                format!("{{{}}}^{{{}}}", base_str, exp_str)
            }
            Expr::Neg(e) => {
                let inner = self.expr_to_latex_internal(*e, true);
                format!("-{}", inner)
            }
            Expr::Function(name, args) => match name.as_str() {
                "sqrt" if args.len() == 1 => {
                    let arg = self.expr_to_latex_internal(args[0], false);
                    format!("\\sqrt{{{}}}", arg)
                }
                "sqrt" if args.len() == 2 => {
                    let radicand = self.expr_to_latex_internal(args[0], false);
                    let index = self.expr_to_latex_internal(args[1], false);
                    format!("\\sqrt[{}]{{{}}}", index, radicand)
                }
                "sin" | "cos" | "tan" | "cot" | "sec" | "csc" | "arcsin" | "arccos" | "arctan"
                | "sinh" | "cosh" | "tanh" | "ln" | "log" | "exp" => {
                    let args_latex: Vec<String> = args
                        .iter()
                        .map(|a| self.expr_to_latex_internal(*a, false))
                        .collect();
                    format!("\\{}({})", name, args_latex.join(", "))
                }
                "abs" if args.len() == 1 => {
                    let arg = self.expr_to_latex_internal(args[0], false);
                    format!("|{}|", arg)
                }
                _ => {
                    let args_latex: Vec<String> = args
                        .iter()
                        .map(|a| self.expr_to_latex_internal(*a, false))
                        .collect();
                    format!("\\text{{{}}}({})", name, args_latex.join(", "))
                }
            },
            Expr::Matrix { rows, cols, data } => {
                let mut result = format!("\\begin{{pmatrix}}");
                for row in 0..*rows {
                    if row > 0 {
                        result.push_str(" \\\\ ");
                    }
                    for col in 0..*cols {
                        if col > 0 {
                            result.push_str(" & ");
                        }
                        let idx = row * cols + col;
                        result.push_str(&self.expr_to_latex_internal(data[idx], false));
                    }
                }
                result.push_str("\\end{pmatrix}");
                result
            }
        }
    }

    fn expr_to_latex_mul(&self, id: ExprId) -> String {
        match self.context.get(id) {
            Expr::Add(_, _) | Expr::Sub(_, _) => {
                format!("({})", self.expr_to_latex_internal(id, false))
            }
            _ => self.expr_to_latex_internal(id, false),
        }
    }

    fn expr_to_latex_base(&self, id: ExprId) -> String {
        match self.context.get(id) {
            Expr::Add(_, _) | Expr::Sub(_, _) | Expr::Mul(_, _) | Expr::Neg(_) => {
                format!("({})", self.expr_to_latex_internal(id, false))
            }
            _ => self.expr_to_latex_internal(id, false),
        }
    }

    fn needs_explicit_mult(&self, left: ExprId, right: ExprId) -> bool {
        match (self.context.get(left), self.context.get(right)) {
            (Expr::Number(_), Expr::Number(_)) => true,
            (Expr::Number(_), Expr::Add(_, _)) => true,
            (Expr::Number(_), Expr::Sub(_, _)) => true,
            (Expr::Add(_, _), Expr::Number(_)) => true,
            (Expr::Sub(_, _), Expr::Number(_)) => true,
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_highlights() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let expr = ctx.add(Expr::Pow(x, two));

        let config = HighlightConfig::new();
        let highlighted = LaTeXExprHighlighted {
            context: &ctx,
            id: expr,
            highlights: &config,
        };

        let latex = highlighted.to_latex();
        assert_eq!(latex, "{x}^{2}");
    }

    #[test]
    fn test_red_highlight() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let pow = ctx.add(Expr::Pow(x, two));

        let mut config = HighlightConfig::new();
        config.add(pow, HighlightColor::Red);

        let highlighted = LaTeXExprHighlighted {
            context: &ctx,
            id: pow,
            highlights: &config,
        };

        let latex = highlighted.to_latex();
        assert!(latex.contains("\\color{red}"));
    }

    #[test]
    fn test_nested_highlight() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let pow = ctx.add(Expr::Pow(x, two));
        let y = ctx.var("y");
        let sum = ctx.add(Expr::Add(pow, y));

        let mut config = HighlightConfig::new();
        config.add(pow, HighlightColor::Green);

        let highlighted = LaTeXExprHighlighted {
            context: &ctx,
            id: sum,
            highlights: &config,
        };

        let latex = highlighted.to_latex();
        assert!(latex.contains("\\color{green}"));
        assert!(latex.contains("y"));
    }
}
