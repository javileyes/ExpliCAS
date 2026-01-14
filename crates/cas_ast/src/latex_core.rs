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

    /// Get style preferences for rendering (e.g., root as radical vs exponential)
    fn get_style_prefs(&self) -> Option<&crate::root_style::StylePreferences> {
        None
    }

    /// Post-process LaTeX to fix negative sign patterns
    fn clean_latex_negatives(latex: &str) -> String {
        use regex::Regex;
        let mut result = latex.to_string();

        // Fix "+ -" → "-" in all contexts
        result = result.replace("+ -\\", "- \\");
        result = result.replace("+ -(", "- (");

        // Fix "- -" → "+" (double negative)
        result = result.replace("- -\\", "+ \\");
        result = result.replace("- -(", "+ (");

        // Fix "+ -" before digits or letters (e.g., "+ -x" → "- x")
        let re_plus_minus = Regex::new(r"\+ -([0-9a-zA-Z])").unwrap();
        result = re_plus_minus.replace_all(&result, "- $1").to_string();

        // Fix "- -" before digits or letters (e.g., "- -x" → "+ x")
        let re_minus_minus = Regex::new(r"- -([0-9a-zA-Z])").unwrap();
        result = re_minus_minus.replace_all(&result, "+ $1").to_string();

        // Fix "+ {color command}{-" patterns (highlighted negatives)
        // e.g., "+ {\color{red}{-..." → "- {\color{red}{"
        // Note: Absorbs only the leading minus of the highlighted expression
        let re_plus_color_minus = Regex::new(r"\+ (\{\\color\{[^}]+\}\{)-").unwrap();
        result = re_plus_color_minus.replace_all(&result, "- $1").to_string();

        // Fix "- {color command}{-" patterns (double negative with highlight)
        let re_minus_color_minus = Regex::new(r"- (\{\\color\{[^}]+\}\{)-").unwrap();
        result = re_minus_color_minus
            .replace_all(&result, "+ $1")
            .to_string();

        // Fix "+ -{" → "- {" (when minus precedes a brace group)
        result = result.replace("+ -{", "- {");

        // Fix "- -{" → "+ {" (double negative before brace)
        result = result.replace("- -{", "+ {");

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
            Expr::SessionRef(id) => format!("\\#{}", id), // LaTeX escape
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
            Constant::I => "i".to_string(),
        }
    }

    /// Format addition - flatten chain, sort by degree, render with proper signs
    fn format_add(&self, l: ExprId, r: ExprId) -> String {
        let ctx = self.context();

        // Flatten the Add chain into individual terms
        let mut terms = Vec::new();
        Self::collect_add_terms_static(ctx, l, &mut terms);
        Self::collect_add_terms_static(ctx, r, &mut terms);

        // Sort by polynomial degree (descending) then sign (positive first)
        terms.sort_by(|a, b| crate::display::cmp_term_for_display(ctx, *a, *b));

        // Render terms with proper sign handling
        let mut result = String::new();
        for (i, term) in terms.iter().enumerate() {
            let (is_neg, term_str) = self.term_to_latex_with_sign(*term);

            if i == 0 {
                // First term: include sign only if negative
                if is_neg {
                    result.push_str(&format!("-{}", term_str));
                } else {
                    result.push_str(&term_str);
                }
            } else if is_neg {
                result.push_str(&format!(" - {}", term_str));
            } else {
                result.push_str(&format!(" + {}", term_str));
            }
        }

        result
    }

    /// Collect all additive terms by flattening nested Add
    fn collect_add_terms_static(ctx: &Context, id: ExprId, terms: &mut Vec<ExprId>) {
        match ctx.get(id) {
            Expr::Add(l, r) => {
                Self::collect_add_terms_static(ctx, *l, terms);
                Self::collect_add_terms_static(ctx, *r, terms);
            }
            _ => terms.push(id),
        }
    }

    /// Convert a term to LaTeX, returning (is_negative, positive_latex)
    fn term_to_latex_with_sign(&self, id: ExprId) -> (bool, String) {
        let ctx = self.context();
        match ctx.get(id) {
            Expr::Neg(inner) => {
                let inner_is_add_sub = matches!(ctx.get(*inner), Expr::Add(_, _) | Expr::Sub(_, _));
                let latex = self.expr_to_latex(*inner, true);
                if inner_is_add_sub {
                    (true, format!("({})", latex))
                } else {
                    (true, latex)
                }
            }
            Expr::Number(n) if n.is_negative() => {
                let positive = -n;
                let positive_str = if positive.is_integer() {
                    format!("{}", positive.numer())
                } else {
                    format!("\\frac{{{}}}{{{}}}", positive.numer(), positive.denom())
                };
                (true, positive_str)
            }
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
                            (true, format!("{}\\cdot {}", coef_str, rest_latex))
                        }
                    } else {
                        (false, self.expr_to_latex(id, false))
                    }
                } else {
                    (false, self.expr_to_latex(id, false))
                }
            }
            _ => (false, self.expr_to_latex(id, false)),
        }
    }

    /// Format subtraction
    fn format_sub(&self, l: ExprId, r: ExprId) -> String {
        let left = self.expr_to_latex(l, false);
        // Right operand needs parentheses if it's Add or Sub to preserve precedence
        // e.g., A - (B + C) must show parens, otherwise it looks like A - B + C
        let r_expr = self.context().get(r);
        let right = if matches!(r_expr, Expr::Add(_, _) | Expr::Sub(_, _)) {
            format!("({})", self.expr_to_latex(r, false))
        } else {
            self.expr_to_latex(r, true)
        };
        format!("{} - {}", left, right)
    }

    /// Format multiplication with explicit cdot
    fn format_mul(&self, l: ExprId, r: ExprId, parent_needs_parens: bool) -> String {
        // Skip multiplication by 1: 1 * x = x, x * 1 = x
        if let Expr::Number(n) = self.context().get(l) {
            if n.is_integer() && *n == num_rational::BigRational::from_integer(1.into()) {
                return self.expr_to_latex(r, parent_needs_parens);
            }
        }
        if let Expr::Number(n) = self.context().get(r) {
            if n.is_integer() && *n == num_rational::BigRational::from_integer(1.into()) {
                return self.expr_to_latex(l, parent_needs_parens);
            }
        }

        // V2.14.40: Absorb fractional coefficient into fraction for cleaner display
        // Pattern: (1/n) * expr -> \frac{expr}{n}
        // Pattern: (k/n) * expr -> \frac{k \cdot expr}{n} or just k/n * expr (simpler)
        if let Expr::Number(n) = self.context().get(l) {
            // Check if left is a simple fraction 1/n (numerator = 1, denominator > 1)
            if !n.is_integer() && *n.numer() == 1.into() && *n.denom() > 1.into() {
                let right_latex = self.expr_to_latex(r, false);
                return format!("\\frac{{{}}}{{{}}}", right_latex, n.denom());
            }
        }
        // Also check right side for (expr * 1/n) pattern
        if let Expr::Number(n) = self.context().get(r) {
            if !n.is_integer() && *n.numer() == 1.into() && *n.denom() > 1.into() {
                let left_latex = self.expr_to_latex(l, false);
                return format!("\\frac{{{}}}{{{}}}", left_latex, n.denom());
            }
        }

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
    /// Respects display hints and style preferences in priority order:
    /// 1. Node-level hint (PreferPower or AsRoot) - highest priority
    /// 2. Global style preferences (Exponential or Radical)
    /// 3. Default: render fractional powers as roots
    fn format_pow(&self, base: ExprId, exp: ExprId) -> String {
        use crate::display_context::DisplayHint;
        use crate::root_style::RootStyle;
        use num_traits::ToPrimitive;

        // If exponent is 1, just return the base (no ^{1})
        if let Expr::Number(n) = self.context().get(exp) {
            if n.is_integer() && *n == num_rational::BigRational::from_integer(1.into()) {
                return self.expr_to_latex(base, false);
            }
        }

        // If base is 1, just return "1" (1^n = 1)
        if let Expr::Number(n) = self.context().get(base) {
            if n.is_integer() && *n == num_rational::BigRational::from_integer(1.into()) {
                return "1".to_string();
            }
        }

        // Helper to extract (numerator, denominator) from exponent
        let get_frac_parts = |ctx: &Context, exp_id: ExprId| -> Option<(i64, i64)> {
            match ctx.get(exp_id) {
                // Expr::Number with rational value
                Expr::Number(n) => {
                    if !n.is_integer() {
                        let numer = n.numer().to_i64()?;
                        let denom = n.denom().to_i64()?;
                        if denom > 1 {
                            return Some((numer, denom));
                        }
                    }
                    None
                }
                // Expr::Div(num, den) form
                Expr::Div(num, den) => {
                    if let (Expr::Number(n), Expr::Number(d)) = (ctx.get(*num), ctx.get(*den)) {
                        if n.is_integer() && d.is_integer() {
                            let numer = n.numer().to_i64()?;
                            let denom = d.numer().to_i64()?;
                            if denom > 1 {
                                return Some((numer, denom));
                            }
                        }
                    }
                    None
                }
                _ => None,
            }
        };

        // Helper to render as power (exponential form)
        let render_as_power = |this: &Self| -> String {
            let base_str = this.expr_to_latex_base(base);
            let exp_str = this.expr_to_latex(exp, false);
            format!("{{{}}}^{{{}}}", base_str, exp_str)
        };

        // Helper to render as root
        let render_as_root = |this: &Self, numer: i64, denom: i64| -> String {
            let base_str = this.expr_to_latex(base, false);
            if numer == 1 {
                // Simple root: x^(1/n) -> \sqrt[n]{x}
                if denom == 2 {
                    format!("\\sqrt{{{}}}", base_str)
                } else {
                    format!("\\sqrt[{}]{{{}}}", denom, base_str)
                }
            } else if numer > 0 {
                // Fractional power: x^(k/n) -> \sqrt[n]{x^k}
                if denom == 2 {
                    format!("\\sqrt{{{{{}}}^{{{}}}}}", base_str, numer)
                } else {
                    format!("\\sqrt[{}]{{{{{}}}^{{{}}}}}", denom, base_str, numer)
                }
            } else {
                // Negative numerators: fall through to power
                render_as_power(this)
            }
        };

        // Check for fractional exponent
        if let Some((numer, denom)) = get_frac_parts(self.context(), exp) {
            // Priority 1: Check node-level display hint
            // Note: We check the root_id's hints, which may contain node-specific preferences
            if let Some(hints) = self.get_display_hint(self.root_id()) {
                // Check if there's a specific hint for the current Pow's base
                // (This is a simplified check - ideally we'd check hints for the Pow node itself)
                if let Some(hint) = hints.get(base) {
                    match hint {
                        DisplayHint::PreferPower => return render_as_power(self),
                        DisplayHint::AsRoot { .. } => return render_as_root(self, numer, denom),
                    }
                }
            }

            // Priority 2: Check global style preferences
            if let Some(prefs) = self.get_style_prefs() {
                match prefs.root_style {
                    RootStyle::Exponential => return render_as_power(self),
                    RootStyle::Radical => return render_as_root(self, numer, denom),
                    RootStyle::Auto => return render_as_root(self, numer, denom), // Auto defaults to radical
                }
            }

            // Priority 3: Default to radical notation
            return render_as_root(self, numer, denom);
        }

        // Non-fractional exponent: render as power
        render_as_power(self)
    }

    /// Format negation
    fn format_neg(&self, e: ExprId) -> String {
        // Check if inner is Add/Sub - needs parentheses to preserve grouping
        // e.g., -(a + b) should display as "-(a + b)" not "-a + b"
        let inner_is_add_sub = matches!(self.context().get(e), Expr::Add(_, _) | Expr::Sub(_, _));
        let inner = self.expr_to_latex(e, true);
        if inner_is_add_sub {
            format!("-({})", inner)
        } else {
            format!("-{}", inner)
        }
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
            // Inverse trig: use \arcsin, \arccos, \arctan (MathJax-compatible)
            "asin" | "arcsin" => {
                format!("\\arcsin({})", self.expr_to_latex(args[0], false))
            }
            "acos" | "arccos" => {
                format!("\\arccos({})", self.expr_to_latex(args[0], false))
            }
            "atan" | "arctan" => {
                format!("\\arctan({})", self.expr_to_latex(args[0], false))
            }
            "sinh" | "cosh" | "tanh" => {
                format!("\\{}({})", name, self.expr_to_latex(args[0], false))
            }
            "ln" => format!("\\ln({})", self.expr_to_latex(args[0], false)),
            "log" if args.len() == 1 => {
                format!("\\log({})", self.expr_to_latex(args[0], false))
            }
            "log" if args.len() == 2 => {
                // log(base, arg) where args[0]=base, args[1]=arg
                let base = self.expr_to_latex(args[0], false);
                let arg = self.expr_to_latex(args[1], false);
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
            // __eq__ is an internal equation representation - display as "lhs = rhs"
            "__eq__" if args.len() == 2 => {
                let lhs = self.expr_to_latex(args[0], false);
                let rhs = self.expr_to_latex(args[1], false);
                format!("{} = {}", lhs, rhs)
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
            // Negative numbers also need parentheses: (-1)^2 not -1^2
            Expr::Number(n) if n.is_negative() => {
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
    // V2.14.40: format_pow is now handled by the trait default, which renders
    // fractional powers as roots automatically
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
    // V2.14.40: format_pow is now handled by the trait default, which renders
    // fractional powers as roots automatically
}

// ============================================================================
// Path-Highlighted LaTeX Renderer (V2.9.16)
// ============================================================================

use crate::expr_path::ExprPath;
use crate::latex_highlight::PathHighlightConfig;

/// LaTeX renderer that highlights by path instead of ExprId.
///
/// This solves the problem where multiple occurrences of the same value
/// (e.g., all `4`s) would be highlighted when only one should be.
///
/// V2.9.16: Path-based occurrence highlighting
pub struct PathHighlightedLatexRenderer<'a> {
    pub context: &'a Context,
    pub id: ExprId,
    pub path_highlights: &'a PathHighlightConfig,
    pub hints: Option<&'a DisplayContext>,
    /// Style preferences for rendering (e.g., root style)
    pub style_prefs: Option<&'a crate::root_style::StylePreferences>,
}

impl<'a> PathHighlightedLatexRenderer<'a> {
    /// Generate complete LaTeX output with path-based highlighting
    pub fn to_latex(&self) -> String {
        let latex = self.render_with_path(self.id, false, &vec![]);
        Self::clean_latex_negatives(&latex)
    }

    /// Post-process LaTeX to fix negative sign patterns (same as LaTeXRenderer)
    fn clean_latex_negatives(latex: &str) -> String {
        use regex::Regex;
        let mut result = latex.to_string();

        // Fix "+ -" → "-" in all contexts
        result = result.replace("+ -\\", "- \\");
        result = result.replace("+ -(", "- (");

        // Fix "- -" → "+" (double negative)
        result = result.replace("- -\\", "+ \\");
        result = result.replace("- -(", "+ (");

        // Fix "+ -" before digits or letters (e.g., "+ -x" → "- x")
        let re_plus_minus = Regex::new(r"\+ -([0-9a-zA-Z])").unwrap();
        result = re_plus_minus.replace_all(&result, "- $1").to_string();

        // Fix "- -" before digits or letters (e.g., "- -x" → "+ x")
        let re_minus_minus = Regex::new(r"- -([0-9a-zA-Z])").unwrap();
        result = re_minus_minus.replace_all(&result, "+ $1").to_string();

        // Fix "+ {color command}{-" patterns (highlighted negatives)
        // e.g., "+ {\color{red}{-..." → "- {\color{red}{"
        let re_plus_color_minus = Regex::new(r"\+ (\{\\color\{[^}]+\}\{)-").unwrap();
        result = re_plus_color_minus.replace_all(&result, "- $1").to_string();

        // Fix "- {color command}{-" patterns (double negative with highlight)
        let re_minus_color_minus = Regex::new(r"- (\{\\color\{[^}]+\}\{)-").unwrap();
        result = re_minus_color_minus
            .replace_all(&result, "+ $1")
            .to_string();

        // Fix "+ -{" → "- {" (when minus precedes a brace group)
        result = result.replace("+ -{", "- {");

        // Fix "- -{" → "+ {" (double negative before brace)
        result = result.replace("- -{", "+ {");

        result
    }

    /// Render expression with path tracking
    fn render_with_path(
        &self,
        id: ExprId,
        parent_needs_parens: bool,
        current_path: &ExprPath,
    ) -> String {
        // Check for path-based highlighting
        if let Some(color) = self.path_highlights.get(current_path) {
            let inner = self.format_at_path(id, parent_needs_parens, current_path);
            return format!("{{\\color{{{}}}{{{}}}}}", color.to_latex(), inner);
        }

        self.format_at_path(id, parent_needs_parens, current_path)
    }

    /// Format expression at a given path (without checking for highlight)
    fn format_at_path(
        &self,
        id: ExprId,
        parent_needs_parens: bool,
        current_path: &ExprPath,
    ) -> String {
        match self.context.get(id) {
            Expr::Number(n) => self.format_number(n),
            Expr::Variable(name) => name.clone(),
            Expr::Constant(c) => self.format_constant(c),
            Expr::Add(l, r) => self.format_add_path(*l, *r, current_path),
            Expr::Sub(l, r) => self.format_sub_path(*l, *r, current_path),
            Expr::Mul(l, r) => self.format_mul_path(*l, *r, parent_needs_parens, current_path),
            Expr::Div(l, r) => self.format_div_path(*l, *r, current_path),
            Expr::Pow(base, exp) => self.format_pow_path(*base, *exp, current_path),
            Expr::Neg(e) => self.format_neg_path(*e, current_path),
            Expr::Function(name, args) => self.format_function_path(name, args, current_path),
            Expr::Matrix { rows, cols, data } => {
                self.format_matrix_path(*rows, *cols, data, current_path)
            }
            Expr::SessionRef(id) => format!("\\#{}", id),
        }
    }

    fn child_path(&self, current: &ExprPath, child_idx: u8) -> ExprPath {
        let mut p = current.clone();
        p.push(child_idx);
        p
    }

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

    fn format_constant(&self, c: &Constant) -> String {
        match c {
            Constant::Pi => "\\pi".to_string(),
            Constant::E => "e".to_string(),
            Constant::Infinity => "\\infty".to_string(),
            Constant::Undefined => "\\text{undefined}".to_string(),
            Constant::I => "i".to_string(),
        }
    }

    fn format_add_path(&self, l: ExprId, r: ExprId, path: &ExprPath) -> String {
        // Simplified version - full version would handle negative detection
        let left = self.render_with_path(l, false, &self.child_path(path, 0));
        let right = self.render_with_path(r, false, &self.child_path(path, 1));
        format!("{} + {}", left, right)
    }

    fn format_sub_path(&self, l: ExprId, r: ExprId, path: &ExprPath) -> String {
        let left = self.render_with_path(l, false, &self.child_path(path, 0));
        let r_expr = self.context.get(r);
        let right = if matches!(r_expr, Expr::Add(_, _) | Expr::Sub(_, _)) {
            format!(
                "({})",
                self.render_with_path(r, false, &self.child_path(path, 1))
            )
        } else {
            self.render_with_path(r, true, &self.child_path(path, 1))
        };
        format!("{} - {}", left, right)
    }

    fn format_mul_path(
        &self,
        l: ExprId,
        r: ExprId,
        parent_needs_parens: bool,
        path: &ExprPath,
    ) -> String {
        // Skip multiplication by 1: 1 * x = x, x * 1 = x
        if let Expr::Number(n) = self.context.get(l) {
            if n.is_integer() && *n == num_rational::BigRational::from_integer(1.into()) {
                return self.render_with_path(r, parent_needs_parens, &self.child_path(path, 1));
            }
        }
        if let Expr::Number(n) = self.context.get(r) {
            if n.is_integer() && *n == num_rational::BigRational::from_integer(1.into()) {
                return self.render_with_path(l, parent_needs_parens, &self.child_path(path, 0));
            }
        }

        // V2.14.40: Absorb fractional coefficient into fraction for cleaner display
        // Pattern: (1/n) * expr -> \frac{expr}{n}
        if let Expr::Number(n) = self.context.get(l) {
            if !n.is_integer() && *n.numer() == 1.into() && *n.denom() > 1.into() {
                let right_latex = self.render_with_path(r, false, &self.child_path(path, 1));
                return format!("\\frac{{{}}}{{{}}}", right_latex, n.denom());
            }
        }
        // Also check right side for (expr * 1/n) pattern
        if let Expr::Number(n) = self.context.get(r) {
            if !n.is_integer() && *n.numer() == 1.into() && *n.denom() > 1.into() {
                let left_latex = self.render_with_path(l, false, &self.child_path(path, 0));
                return format!("\\frac{{{}}}{{{}}}", left_latex, n.denom());
            }
        }

        let left = self.render_mul_operand(l, &self.child_path(path, 0));
        let right = self.render_mul_operand(r, &self.child_path(path, 1));
        if parent_needs_parens {
            format!("({}\\cdot {})", left, right)
        } else {
            format!("{}\\cdot {}", left, right)
        }
    }

    fn render_mul_operand(&self, id: ExprId, path: &ExprPath) -> String {
        match self.context.get(id) {
            Expr::Add(_, _) | Expr::Sub(_, _) => {
                format!("({})", self.render_with_path(id, false, path))
            }
            _ => self.render_with_path(id, false, path),
        }
    }

    fn format_div_path(&self, l: ExprId, r: ExprId, path: &ExprPath) -> String {
        let numer = self.render_with_path(l, false, &self.child_path(path, 0));
        let denom = self.render_with_path(r, false, &self.child_path(path, 1));
        format!("\\frac{{{}}}{{{}}}", numer, denom)
    }

    fn format_pow_path(&self, base: ExprId, exp: ExprId, path: &ExprPath) -> String {
        use crate::display_context::DisplayHint;
        use num_traits::ToPrimitive;

        // Check if exponent is 1, just return the base (no ^{1})
        if let Expr::Number(n) = self.context.get(exp) {
            if n.is_integer() && *n == num_rational::BigRational::from_integer(1.into()) {
                return self.render_with_path(base, false, &self.child_path(path, 0));
            }
        }

        // Check if base is 1, just return "1" (1^n = 1)
        if let Expr::Number(n) = self.context.get(base) {
            if n.is_integer() && *n == num_rational::BigRational::from_integer(1.into()) {
                return "1".to_string();
            }
        }

        // Helper to extract (numerator, denominator) from exponent
        let get_frac_parts = |ctx: &Context, exp_id: ExprId| -> Option<(i64, i64)> {
            match ctx.get(exp_id) {
                // Expr::Number with rational value
                Expr::Number(n) => {
                    if !n.is_integer() {
                        let numer = n.numer().to_i64()?;
                        let denom = n.denom().to_i64()?;
                        if denom > 1 {
                            return Some((numer, denom));
                        }
                    }
                    None
                }
                // Expr::Div(num, den) form
                Expr::Div(num, den) => {
                    if let (Expr::Number(n), Expr::Number(d)) = (ctx.get(*num), ctx.get(*den)) {
                        if n.is_integer() && d.is_integer() {
                            let numer = n.numer().to_i64()?;
                            let denom = d.numer().to_i64()?;
                            if denom > 1 {
                                return Some((numer, denom));
                            }
                        }
                    }
                    None
                }
                _ => None,
            }
        };

        // V2.14.40: Render fractional powers based on style preferences
        // Priority: 1) node hint, 2) style prefs, 3) default radical
        if let Some((numer, denom)) = get_frac_parts(self.context, exp) {
            // Helper to render as power
            let render_power = || {
                let base_str = self.render_base(base, &self.child_path(path, 0));
                let exp_str = self.render_with_path(exp, false, &self.child_path(path, 1));
                format!("{{{}}}^{{{}}}", base_str, exp_str)
            };

            // Helper to render as root
            let render_root = || {
                let base_str = self.render_with_path(base, false, &self.child_path(path, 0));
                if numer == 1 {
                    if denom == 2 {
                        format!("\\sqrt{{{}}}", base_str)
                    } else {
                        format!("\\sqrt[{}]{{{}}}", denom, base_str)
                    }
                } else if numer > 0 {
                    if denom == 2 {
                        format!("\\sqrt{{{{{}}}^{{{}}}}}", base_str, numer)
                    } else {
                        format!("\\sqrt[{}]{{{{{}}}^{{{}}}}}", denom, base_str, numer)
                    }
                } else {
                    // Negative numerator: fall to power
                    render_power()
                }
            };

            // Priority 1: Check node-level hints (via hints field)
            if let Some(hints) = self.hints {
                if let Some(hint) = hints.get(base) {
                    match hint {
                        DisplayHint::PreferPower => return render_power(),
                        DisplayHint::AsRoot { .. } => return render_root(),
                    }
                }
            }

            // Priority 2: Check style preferences
            if let Some(prefs) = self.style_prefs {
                use crate::root_style::RootStyle;
                match prefs.root_style {
                    RootStyle::Exponential => return render_power(),
                    RootStyle::Radical | RootStyle::Auto => return render_root(),
                }
            }

            // Priority 3: Default to radical
            return render_root();
        }

        // Default power rendering (non-fractional exponent)
        let base_str = self.render_base(base, &self.child_path(path, 0));
        let exp_str = self.render_with_path(exp, false, &self.child_path(path, 1));
        format!("{{{}}}^{{{}}}", base_str, exp_str)
    }

    fn render_base(&self, id: ExprId, path: &ExprPath) -> String {
        match self.context.get(id) {
            Expr::Add(_, _)
            | Expr::Sub(_, _)
            | Expr::Mul(_, _)
            | Expr::Div(_, _)
            | Expr::Neg(_) => {
                format!("({})", self.render_with_path(id, false, path))
            }
            // Negative numbers also need parentheses: (-1)^2 not -1^2
            Expr::Number(n) if n.is_negative() => {
                format!("({})", self.render_with_path(id, false, path))
            }
            _ => self.render_with_path(id, false, path),
        }
    }

    fn format_neg_path(&self, e: ExprId, path: &ExprPath) -> String {
        let inner_is_add_sub = matches!(self.context.get(e), Expr::Add(_, _) | Expr::Sub(_, _));
        let inner = self.render_with_path(e, true, &self.child_path(path, 0));
        if inner_is_add_sub {
            format!("-({})", inner)
        } else {
            format!("-{}", inner)
        }
    }

    fn format_function_path(&self, name: &str, args: &[ExprId], path: &ExprPath) -> String {
        match name {
            "sqrt" if args.len() == 1 => {
                format!(
                    "\\sqrt{{{}}}",
                    self.render_with_path(args[0], false, &self.child_path(path, 0))
                )
            }
            "sqrt" if args.len() == 2 => {
                let radicand = self.render_with_path(args[0], false, &self.child_path(path, 0));
                let index = self.render_with_path(args[1], false, &self.child_path(path, 1));
                format!("\\sqrt[{}]{{{}}}", index, radicand)
            }
            "sin" | "cos" | "tan" | "cot" | "sec" | "csc" => {
                format!(
                    "\\{}({})",
                    name,
                    self.render_with_path(args[0], false, &self.child_path(path, 0))
                )
            }
            "ln" => format!(
                "\\ln({})",
                self.render_with_path(args[0], false, &self.child_path(path, 0))
            ),
            "log" if args.len() == 1 => format!(
                "\\log({})",
                self.render_with_path(args[0], false, &self.child_path(path, 0))
            ),
            "log" if args.len() == 2 => {
                // log(base, arg) where args[0]=base, args[1]=arg
                let base = self.render_with_path(args[0], false, &self.child_path(path, 0));
                let arg = self.render_with_path(args[1], false, &self.child_path(path, 1));
                format!("\\log_{{{}}}({})", base, arg)
            }
            "abs" => format!(
                "|{}|",
                self.render_with_path(args[0], false, &self.child_path(path, 0))
            ),
            // __eq__ is an internal equation representation - display as "lhs = rhs"
            "__eq__" if args.len() == 2 => {
                let lhs = self.render_with_path(args[0], false, &self.child_path(path, 0));
                let rhs = self.render_with_path(args[1], false, &self.child_path(path, 1));
                format!("{} = {}", lhs, rhs)
            }
            _ => {
                let args_str: Vec<String> = args
                    .iter()
                    .enumerate()
                    .map(|(i, &a)| self.render_with_path(a, false, &self.child_path(path, i as u8)))
                    .collect();
                format!("\\text{{{}}}({})", name, args_str.join(", "))
            }
        }
    }

    fn format_matrix_path(
        &self,
        rows: usize,
        cols: usize,
        data: &[ExprId],
        path: &ExprPath,
    ) -> String {
        let mut result = String::from("\\begin{bmatrix}\n");
        for r in 0..rows {
            for c in 0..cols {
                if c > 0 {
                    result.push_str(" & ");
                }
                let idx = r * cols + c;
                result.push_str(&self.render_with_path(
                    data[idx],
                    false,
                    &self.child_path(path, idx as u8),
                ));
            }
            if r < rows - 1 {
                result.push_str(" \\\\\n");
            }
        }
        result.push_str("\n\\end{bmatrix}");
        result
    }
}
