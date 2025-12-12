//! Core LaTeX rendering logic shared across all LaTeX renderers
//!
//! This module provides a trait-based approach to LaTeX rendering.
//! All renderers implement LaTeXRenderer trait which provides shared
//! implementations for expression rendering with customizable hooks.

use crate::display_context::DisplayContext;
use crate::latex_highlight::{HighlightColor, HighlightConfig};
use crate::{Constant, Context, Expr, ExprId};
use num_traits::Signed;

/// Core trait for LaTeX rendering
///
/// Implementors provide context access and optional hooks for
/// highlighting and display hints. The trait provides default
/// implementations for all expression rendering.
pub trait LaTeXRenderer {
    /// Get the expression context
    fn context(&self) -> &Context;

    /// Get the root expression ID
    fn root_id(&self) -> ExprId;

    /// Check if an expression should be highlighted
    fn get_highlight(&self, _id: ExprId) -> Option<HighlightColor> {
        None
    }

    /// Get display hints for an expression (e.g., render as root)
    fn get_display_hint(&self, _id: ExprId) -> Option<&DisplayContext> {
        None
    }

    /// Post-process LaTeX to fix negative sign patterns
    fn clean_latex_negatives(latex: &str) -> String {
        use regex::Regex;
        let mut result = latex.to_string();

        // Fix "+ -" → "-" in all contexts
        result = result.replace("+ -\\", "- \\");
        result = result.replace("+ -{", "- {");
        result = result.replace("+ -(", "- (");

        // Fix "- -" → "+" (double negative)
        result = result.replace("- -\\", "+ \\");
        result = result.replace("- -{", "+ {");
        result = result.replace("- -(", "+ (");

        // Fix "+ -" before digits
        let re_plus_minus_digit = Regex::new(r"\+ -(\d)").unwrap();
        result = re_plus_minus_digit.replace_all(&result, "- $1").to_string();

        // Fix "- -" before digits
        let re_minus_minus_digit = Regex::new(r"- -(\d)").unwrap();
        result = re_minus_minus_digit
            .replace_all(&result, "+ $1")
            .to_string();

        result
    }

    /// Generate complete LaTeX output
    fn to_latex(&self) -> String {
        let latex = self.expr_to_latex(self.root_id(), false);
        Self::clean_latex_negatives(&latex)
    }

    /// Render an expression to LaTeX
    fn expr_to_latex(&self, id: ExprId, parent_needs_parens: bool) -> String {
        // Check for highlighting first
        if let Some(color) = self.get_highlight(id) {
            let inner = self.format_expr(id, parent_needs_parens);
            return format!("{{\\color{{{}}}{{{}}}}}", color.to_latex(), inner);
        }

        self.format_expr(id, parent_needs_parens)
    }

    /// Format a single expression (without highlight wrapper)
    fn format_expr(&self, id: ExprId, parent_needs_parens: bool) -> String {
        let ctx = self.context();
        match ctx.get(id) {
            Expr::Number(n) => self.format_number(n),
            Expr::Variable(name) => name.clone(),
            Expr::Constant(c) => self.format_constant(c),
            Expr::Add(l, r) => self.format_add(*l, *r),
            Expr::Sub(l, r) => self.format_sub(*l, *r),
            Expr::Mul(l, r) => self.format_mul(*l, *r, parent_needs_parens),
            Expr::Div(l, r) => self.format_div(*l, *r),
            Expr::Pow(base, exp) => self.format_pow(*base, *exp),
            Expr::Neg(e) => self.format_neg(*e),
            Expr::Function(name, args) => self.format_function(name, args),
            Expr::Matrix { rows, cols, data } => self.format_matrix(*rows, *cols, data),
        }
    }

    /// Format a number
    fn format_number(&self, n: &num_rational::BigRational) -> String {
        if n.is_integer() {
            format!("{}", n.numer())
        } else if n.is_negative() {
            let positive = -n;
            format!("-\\frac{{{}}}{{{}}}", positive.numer(), positive.denom())
        } else {
            format!("\\frac{{{}}}{{{}}}", n.numer(), n.denom())
        }
    }

    /// Format a constant
    fn format_constant(&self, c: &Constant) -> String {
        match c {
            Constant::Pi => "\\pi".to_string(),
            Constant::E => "e".to_string(),
            Constant::Infinity => "\\infty".to_string(),
            Constant::Undefined => "\\text{undefined}".to_string(),
        }
    }

    /// Format addition, detecting negative right operand
    fn format_add(&self, l: ExprId, r: ExprId) -> String {
        let ctx = self.context();

        // Check if left side is negative - canonicalization may swap operands
        // Handle Neg(expr) case
        if let Expr::Neg(left_inner) = ctx.get(l) {
            let inner_latex = self.expr_to_latex(*left_inner, true);
            let right_latex = self.expr_to_latex(r, false);
            return format!("{} - {}", right_latex, inner_latex);
        }

        // Handle negative number on left: -2 + x -> x - 2
        if let Expr::Number(n) = ctx.get(l) {
            if n.is_negative() {
                let positive = -n;
                let positive_str = if positive.is_integer() {
                    format!("{}", positive.numer())
                } else {
                    format!("\\frac{{{}}}{{{}}}", positive.numer(), positive.denom())
                };
                let right_latex = self.expr_to_latex(r, false);
                return format!("{} - {}", right_latex, positive_str);
            }
        }

        let left = self.expr_to_latex(l, false);

        // Check if right side is negative
        let (is_negative, right_str) = match ctx.get(r) {
            Expr::Number(n) if n.is_negative() => {
                let positive = -n;
                let positive_str = if positive.is_integer() {
                    format!("{}", positive.numer())
                } else {
                    format!("\\frac{{{}}}{{{}}}", positive.numer(), positive.denom())
                };
                (true, positive_str)
            }
            Expr::Neg(inner) => (true, self.expr_to_latex(*inner, true)),
            Expr::Mul(ml, mr) => {
                if let Expr::Number(coef) = ctx.get(*ml) {
                    if coef.is_negative() {
                        let positive_coef = -coef;
                        let rest_latex = self.expr_to_latex_mul(*mr);

                        if positive_coef.is_integer() && *positive_coef.numer() == 1.into() {
                            (true, rest_latex)
                        } else {
                            let coef_str = if positive_coef.is_integer() {
                                format!("{}", positive_coef.numer())
                            } else {
                                format!(
                                    "\\frac{{{}}}{{{}}}",
                                    positive_coef.numer(),
                                    positive_coef.denom()
                                )
                            };
                            // Always use cdot
                            (true, format!("{}\\cdot {}", coef_str, rest_latex))
                        }
                    } else {
                        (false, self.expr_to_latex(r, false))
                    }
                } else {
                    (false, self.expr_to_latex(r, false))
                }
            }
            _ => (false, self.expr_to_latex(r, false)),
        };

        if is_negative {
            format!("{} - {}", left, right_str)
        } else {
            format!("{} + {}", left, right_str)
        }
    }

    /// Format subtraction
    fn format_sub(&self, l: ExprId, r: ExprId) -> String {
        let left = self.expr_to_latex(l, false);
        let right = self.expr_to_latex(r, true);
        format!("{} - {}", left, right)
    }

    /// Format multiplication with explicit cdot
    fn format_mul(&self, l: ExprId, r: ExprId, parent_needs_parens: bool) -> String {
        let left = self.expr_to_latex_mul(l);
        let right = self.expr_to_latex_mul(r);

        // Always use cdot for consistent formatting
        if parent_needs_parens {
            format!("({}\\cdot {})", left, right)
        } else {
            format!("{}\\cdot {}", left, right)
        }
    }

    /// Format division, handling negative numerators
    fn format_div(&self, l: ExprId, r: ExprId) -> String {
        let ctx = self.context();

        // Check if numerator is negative - put sign outside fraction
        match ctx.get(l) {
            Expr::Neg(inner) => {
                let numer = self.expr_to_latex(*inner, false);
                let denom = self.expr_to_latex(r, false);
                format!("-\\frac{{{}}}{{{}}}", numer, denom)
            }
            Expr::Number(n) if n.is_negative() => {
                let positive = -n;
                let numer_str = if positive.is_integer() {
                    format!("{}", positive.numer())
                } else {
                    format!("\\frac{{{}}}{{{}}}", positive.numer(), positive.denom())
                };
                let denom = self.expr_to_latex(r, false);
                format!("-\\frac{{{}}}{{{}}}", numer_str, denom)
            }
            _ => {
                let numer = self.expr_to_latex(l, false);
                let denom = self.expr_to_latex(r, false);
                format!("\\frac{{{}}}{{{}}}", numer, denom)
            }
        }
    }

    /// Format power expression
    fn format_pow(&self, base: ExprId, exp: ExprId) -> String {
        // If exponent is 1, just return the base (no ^{1})
        if let Expr::Number(n) = self.context().get(exp) {
            if n.is_integer() && *n == num_rational::BigRational::from_integer(1.into()) {
                return self.expr_to_latex(base, false);
            }
        }

        let base_str = self.expr_to_latex_base(base);
        let exp_str = self.expr_to_latex(exp, false);
        format!("{{{}}}^{{{}}}", base_str, exp_str)
    }

    /// Format negation
    fn format_neg(&self, e: ExprId) -> String {
        let inner = self.expr_to_latex(e, true);
        format!("-{}", inner)
    }

    /// Format function calls
    fn format_function(&self, name: &str, args: &[ExprId]) -> String {
        match name {
            "sqrt" if args.len() == 1 => {
                format!("\\sqrt{{{}}}", self.expr_to_latex(args[0], false))
            }
            "sqrt" if args.len() == 2 => {
                let radicand = self.expr_to_latex(args[0], false);
                let index = self.expr_to_latex(args[1], false);
                format!("\\sqrt[{}]{{{}}}", index, radicand)
            }
            "sum" if args.len() == 4 => {
                let expr = self.expr_to_latex(args[0], false);
                let var = self.expr_to_latex(args[1], false);
                let start = self.expr_to_latex(args[2], false);
                let end = self.expr_to_latex(args[3], false);
                format!("\\sum_{{{}={}}}^{{{}}} {}", var, start, end, expr)
            }
            "product" if args.len() == 4 => {
                let expr = self.expr_to_latex(args[0], false);
                let var = self.expr_to_latex(args[1], false);
                let start = self.expr_to_latex(args[2], false);
                let end = self.expr_to_latex(args[3], false);
                format!("\\prod_{{{}={}}}^{{{}}} {}", var, start, end, expr)
            }
            "sin" | "cos" | "tan" | "cot" | "sec" | "csc" => {
                format!("\\{}({})", name, self.expr_to_latex(args[0], false))
            }
            "asin" | "acos" | "atan" => {
                format!("\\{}({})", name, self.expr_to_latex(args[0], false))
            }
            "sinh" | "cosh" | "tanh" => {
                format!("\\{}({})", name, self.expr_to_latex(args[0], false))
            }
            "ln" => format!("\\ln({})", self.expr_to_latex(args[0], false)),
            "log" if args.len() == 1 => {
                format!("\\log({})", self.expr_to_latex(args[0], false))
            }
            "log" if args.len() == 2 => {
                let base = self.expr_to_latex(args[1], false);
                let arg = self.expr_to_latex(args[0], false);
                format!("\\log_{{{}}}({})", base, arg)
            }
            "abs" => format!("|{}|", self.expr_to_latex(args[0], false)),
            "exp" => format!("e^{{{}}}", self.expr_to_latex(args[0], false)),
            "floor" => format!("\\lfloor {} \\rfloor", self.expr_to_latex(args[0], false)),
            "ceil" => format!("\\lceil {} \\rceil", self.expr_to_latex(args[0], false)),
            "diff" if args.len() >= 2 => {
                let expr = self.expr_to_latex(args[0], false);
                let var = self.expr_to_latex(args[1], false);
                format!("\\frac{{d}}{{d{}}}({})", var, expr)
            }
            "integrate" if args.len() >= 2 => {
                let expr = self.expr_to_latex(args[0], false);
                let var = self.expr_to_latex(args[1], false);
                format!("\\int {} \\, d{}", expr, var)
            }
            _ => {
                let args_str: Vec<String> =
                    args.iter().map(|&a| self.expr_to_latex(a, false)).collect();
                format!("\\text{{{}}}({})", name, args_str.join(", "))
            }
        }
    }

    /// Format matrix
    fn format_matrix(&self, rows: usize, cols: usize, data: &[ExprId]) -> String {
        let mut result = String::from("\\begin{bmatrix}\n");
        for r in 0..rows {
            for c in 0..cols {
                if c > 0 {
                    result.push_str(" & ");
                }
                let idx = r * cols + c;
                result.push_str(&self.expr_to_latex(data[idx], false));
            }
            if r < rows - 1 {
                result.push_str(" \\\\\n");
            }
        }
        result.push_str("\n\\end{bmatrix}");
        result
    }

    /// Helper for multiplication operands - adds parens for Add/Sub
    fn expr_to_latex_mul(&self, id: ExprId) -> String {
        match self.context().get(id) {
            Expr::Add(_, _) | Expr::Sub(_, _) => {
                format!("({})", self.expr_to_latex(id, false))
            }
            _ => self.expr_to_latex(id, false),
        }
    }

    /// Helper for power base - adds parens for composite expressions
    fn expr_to_latex_base(&self, id: ExprId) -> String {
        match self.context().get(id) {
            Expr::Add(_, _)
            | Expr::Sub(_, _)
            | Expr::Mul(_, _)
            | Expr::Div(_, _)
            | Expr::Neg(_) => {
                format!("({})", self.expr_to_latex(id, false))
            }
            _ => self.expr_to_latex(id, false),
        }
    }
}

// ============================================================================
// Simple LaTeX Renderer (no highlights, no hints)
// ============================================================================

/// Simple LaTeX renderer - no highlighting or display hints
pub struct SimpleLatexRenderer<'a> {
    pub context: &'a Context,
    pub id: ExprId,
}

impl<'a> LaTeXRenderer for SimpleLatexRenderer<'a> {
    fn context(&self) -> &Context {
        self.context
    }

    fn root_id(&self) -> ExprId {
        self.id
    }
}

// ============================================================================
// LaTeX Renderer with Highlighting
// ============================================================================

/// LaTeX renderer with color highlighting support
pub struct HighlightedLatexRenderer<'a> {
    pub context: &'a Context,
    pub id: ExprId,
    pub highlights: &'a HighlightConfig,
}

impl<'a> LaTeXRenderer for HighlightedLatexRenderer<'a> {
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

// ============================================================================
// LaTeX Renderer with Display Hints
// ============================================================================

/// LaTeX renderer with display hints (e.g., render as root)
pub struct HintedLatexRenderer<'a> {
    pub context: &'a Context,
    pub id: ExprId,
    pub hints: &'a DisplayContext,
}

impl<'a> LaTeXRenderer for HintedLatexRenderer<'a> {
    fn context(&self) -> &Context {
        self.context
    }

    fn root_id(&self) -> ExprId {
        self.id
    }

    fn get_display_hint(&self, _id: ExprId) -> Option<&DisplayContext> {
        // The DisplayContext applies to the root; for now return it for all
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

        // Check if this should be rendered as a root
        if let Some(_hints) = self.get_display_hint(self.root_id()) {
            if let Expr::Number(n) = self.context().get(exp) {
                // Check if denominator matches a root hint
                let denom = n.denom();
                if !n.is_integer() && *denom > 1.into() {
                    // Render as nth root
                    let base_str = self.expr_to_latex(base, false);
                    let numer = n.numer();

                    if *numer == 1.into() {
                        // Simple root: x^(1/n) -> nth root of x
                        if *denom == 2.into() {
                            return format!("\\sqrt{{{}}}", base_str);
                        } else {
                            return format!("\\sqrt[{}]{{{}}}", denom, base_str);
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

// ============================================================================
// Full-Featured LaTeX Renderer (Highlights + Hints)
// ============================================================================

/// LaTeX renderer with both highlighting and display hints
pub struct FullLatexRenderer<'a> {
    pub context: &'a Context,
    pub id: ExprId,
    pub highlights: &'a HighlightConfig,
    pub hints: &'a DisplayContext,
}

impl<'a> LaTeXRenderer for FullLatexRenderer<'a> {
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

        // Check if this should be rendered as a root
        if let Some(_hints) = self.get_display_hint(self.root_id()) {
            if let Expr::Number(n) = self.context().get(exp) {
                let denom = n.denom();
                if !n.is_integer() && *denom > 1.into() {
                    let base_str = self.expr_to_latex(base, false);
                    let numer = n.numer();

                    if *numer == 1.into() {
                        if *denom == 2.into() {
                            return format!("\\sqrt{{{}}}", base_str);
                        } else {
                            return format!("\\sqrt[{}]{{{}}}", denom, base_str);
                        }
                    }
                }
            }
        }

        let base_str = self.expr_to_latex_base(base);
        let exp_str = self.expr_to_latex(exp, false);
        format!("{{{}}}^{{{}}}", base_str, exp_str)
    }
}
