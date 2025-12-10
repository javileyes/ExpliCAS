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
        // Replace "+ -" patterns with "- " for cleaner display
        result = result.replace("+ -\\", "- \\");
        result = result.replace("+ -{", "- {");
        result = result.replace("+ -(", "- (");
        // Handle numbers: "+ -1", "+ -2", etc.
        for digit in '0'..='9' {
            result = result.replace(&format!("+ -{}", digit), &format!("- {}", digit));
        }
        // Handle highlighted negatives: "+ {\color{...}{-" → "- {\color{...}{"
        for color in &["red", "green", "blue"] {
            // Pattern: + {\color{red}{- → - {\color{red}{
            let from = format!("+ {{\\color{{{}}}{{-", color);
            let to = format!("- {{\\color{{{}}}{{", color);
            result = result.replace(&from, &to);
        }
        // Handle double negatives
        result = result.replace("- -\\", "+ \\");
        result = result.replace("- -{", "+ {");
        result = result.replace("- -(", "+ (");
        for digit in '0'..='9' {
            result = result.replace(&format!("- -{}", digit), &format!("+ {}", digit));
        }
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
                // First check if left side is negative - canonicalization may swap operands
                if let Expr::Neg(left_inner) = self.context.get(*l) {
                    // Left side is negative: Add(Neg(a), b) = b - a (more natural order)
                    let inner_latex = self.expr_to_latex_internal(*left_inner, true);
                    let right_latex = self.expr_to_latex_internal(*r, false);
                    return format!("{} - {}", right_latex, inner_latex);
                }

                let left = self.expr_to_latex_internal(*l, false);

                // Check if right side is negative
                match self.context.get(*r) {
                    Expr::Number(n) if n.is_negative() => {
                        let positive = -n;
                        let positive_str = if positive.is_integer() {
                            format!("{}", positive.numer())
                        } else {
                            format!("\\frac{{{}}}{{{}}}", positive.numer(), positive.denom())
                        };
                        format!("{} - {}", left, positive_str)
                    }
                    Expr::Neg(inner) => {
                        // Check if the Neg has a highlight - if so, apply it to inner with subtraction
                        if let Some(color) = self.highlights.get(*r) {
                            let inner_latex = self.format_expr(*inner, true);
                            format!(
                                "{} - {{\\color{{{}}}{{{}}}}}",
                                left,
                                color.to_latex(),
                                inner_latex
                            )
                        } else {
                            let inner_latex = self.expr_to_latex_internal(*inner, true);
                            format!("{} - {}", left, inner_latex)
                        }
                    }
                    _ => {
                        let right_str = self.expr_to_latex_internal(*r, false);
                        format!("{} + {}", left, right_str)
                    }
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
                // Render as power notation - sqrt() functions are handled separately
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
            (Expr::Number(_), Expr::Div(_, _)) => true,
            (Expr::Number(_), Expr::Pow(_, _)) => true,
            (Expr::Number(_), Expr::Mul(_, _)) => true,
            (Expr::Add(_, _), Expr::Number(_)) => true,
            (Expr::Sub(_, _), Expr::Number(_)) => true,
            (Expr::Pow(_, _), Expr::Number(_)) => true,
            (Expr::Pow(_, _), Expr::Pow(_, _)) => true,
            _ => false,
        }
    }
}

/// LaTeX expression with optional highlighting AND display hints support
/// Combines LaTeXExprHighlighted with DisplayHint awareness for sqrt notation
pub struct LaTeXExprHighlightedWithHints<'a> {
    pub context: &'a Context,
    pub id: ExprId,
    pub highlights: &'a HighlightConfig,
    pub hints: &'a crate::display_context::DisplayContext,
}

impl<'a> LaTeXExprHighlightedWithHints<'a> {
    /// Generate LaTeX string with highlights and display hints applied
    pub fn to_latex(&self) -> String {
        let latex = self.expr_to_latex_internal(self.id, false);
        LaTeXExprHighlighted::clean_latex_negatives(&latex)
    }

    fn expr_to_latex_internal(&self, id: ExprId, parent_needs_parens: bool) -> String {
        // Format the expression (with hint awareness)
        let inner = self.format_expr(id, parent_needs_parens);

        // Apply highlighting if this expression is marked
        if let Some(color) = self.highlights.get(id) {
            format!("{{\\color{{{}}}{{{}}}}}", color.to_latex(), inner)
        } else {
            inner
        }
    }

    fn format_expr(&self, id: ExprId, _parent_needs_parens: bool) -> String {
        // Check for display hint first - if AsRoot, render as sqrt
        if let Some(crate::display_context::DisplayHint::AsRoot { index }) = self.hints.get(id) {
            if let Expr::Pow(base, exp) = self.context.get(id) {
                // Get the numerator of the exponent (for x^(k/n), k is the power inside the root)
                let inner_power: i64 = if let Expr::Number(n) = self.context.get(*exp) {
                    n.numer().try_into().unwrap_or(1)
                } else {
                    1
                };

                let base_str = self.expr_to_latex_internal(*base, false);
                // If numerator > 1, show base^k inside the root
                let radicand = if inner_power != 1 {
                    format!("{{{}}}^{{{}}}", base_str, inner_power)
                } else {
                    base_str
                };

                if *index == 2 {
                    return format!("\\sqrt{{{}}}", radicand);
                } else {
                    return format!("\\sqrt[{}]{{{}}}", index, radicand);
                }
            }
        }

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
                // Check if right is Neg for subtraction display
                if let Expr::Neg(inner) = self.context.get(*r) {
                    let right = self.expr_to_latex_internal(*inner, false);
                    format!("{} - {}", left, right)
                } else {
                    let right = self.expr_to_latex_internal(*r, false);
                    format!("{} + {}", left, right)
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
                if self.needs_explicit_mult(*l, *r) {
                    format!("{} \\cdot {}", left, right)
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
                "sin" | "cos" | "tan" | "cot" | "sec" | "csc" => {
                    let arg = self.expr_to_latex_internal(args[0], false);
                    format!("\\{}({})", name, arg)
                }
                "ln" => {
                    let arg = self.expr_to_latex_internal(args[0], false);
                    format!("\\ln({})", arg)
                }
                "log" if args.len() == 2 => {
                    let arg = self.expr_to_latex_internal(args[0], false);
                    let base = self.expr_to_latex_internal(args[1], false);
                    format!("\\log_{{{}}}({})", base, arg)
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
                let mut result = String::from("\\begin{bmatrix}\n");
                for r in 0..*rows {
                    for c in 0..*cols {
                        if c > 0 {
                            result.push_str(" & ");
                        }
                        let idx = r * cols + c;
                        result.push_str(&self.expr_to_latex_internal(data[idx], false));
                    }
                    if r < rows - 1 {
                        result.push_str(" \\\\\n");
                    }
                }
                result.push_str("\n\\end{bmatrix}");
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
            (Expr::Number(_), Expr::Div(_, _)) => true,
            (Expr::Number(_), Expr::Pow(_, _)) => true,
            (Expr::Number(_), Expr::Mul(_, _)) => true,
            (Expr::Add(_, _), Expr::Number(_)) => true,
            (Expr::Sub(_, _), Expr::Number(_)) => true,
            (Expr::Pow(_, _), Expr::Number(_)) => true,
            (Expr::Pow(_, _), Expr::Pow(_, _)) => true,
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

    #[test]
    fn test_subtraction_with_highlight() {
        // This tests the problematic pattern from timeline:
        // x^(17/24) - x^(17/24) where the second term is highlighted green
        let mut ctx = Context::new();
        let x = ctx.var("x");

        // Create x^(17/24)
        let seventeen = ctx.num(17);
        let twentyfour = ctx.num(24);
        let exp = ctx.add(Expr::Div(seventeen, twentyfour));
        let pow1 = ctx.add(Expr::Pow(x, exp));

        // Create another x^(17/24) for subtraction
        let x2 = ctx.var("x");
        let seventeen2 = ctx.num(17);
        let twentyfour2 = ctx.num(24);
        let exp2 = ctx.add(Expr::Div(seventeen2, twentyfour2));
        let pow2 = ctx.add(Expr::Pow(x2, exp2));

        // Create pow1 - pow2 using Sub
        let sub_expr = ctx.add(Expr::Sub(pow1, pow2));

        // Highlight the second power (the result after transformation)
        let mut config = HighlightConfig::new();
        config.add(pow2, HighlightColor::Green);

        let highlighted = LaTeXExprHighlighted {
            context: &ctx,
            id: sub_expr,
            highlights: &config,
        };

        let latex = highlighted.to_latex();
        println!("LaTeX with highlight: {}", latex);

        // The LaTeX should still contain subtraction, not addition
        assert!(latex.contains(" - "), "Expected subtraction in: {}", latex);
        assert!(
            latex.contains("\\color{green}"),
            "Expected green highlight in: {}",
            latex
        );
    }

    #[test]
    fn test_addition_of_neg_with_highlight() {
        // Test Add(a, Neg(b)) which should render as a - b
        // The bug might be that when Neg(b) is highlighted, it breaks the pattern
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let neg_y = ctx.add(Expr::Neg(y));

        // a + (-b) should display as a - b
        let add_expr = ctx.add(Expr::Add(x, neg_y));

        // Test without highlight first
        let config = HighlightConfig::new();
        let highlighted = LaTeXExprHighlighted {
            context: &ctx,
            id: add_expr,
            highlights: &config,
        };
        let latex_no_hl = highlighted.to_latex();
        println!("No highlight: {}", latex_no_hl);
        assert!(
            latex_no_hl.contains(" - "),
            "Expected subtraction without highlight: {}",
            latex_no_hl
        );

        // Now test WITH highlight on the neg_y
        let mut config_hl = HighlightConfig::new();
        config_hl.add(neg_y, HighlightColor::Green);

        let highlighted_hl = LaTeXExprHighlighted {
            context: &ctx,
            id: add_expr,
            highlights: &config_hl,
        };
        let latex_hl = highlighted_hl.to_latex();
        println!("With highlight on Neg(y): {}", latex_hl);

        // BUG CHECK: Does highlighting break the subtraction detection?
        // The current code checks self.context.get(*r) but if r is highlighted,
        // it may not detect Neg correctly
        assert!(
            latex_hl.contains(" - ") || !latex_hl.contains(" + "),
            "BUG: Subtraction became addition with highlight: {}",
            latex_hl
        );
    }
}
