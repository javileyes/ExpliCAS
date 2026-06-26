//! Core LaTeX rendering logic shared across all LaTeX renderers
//!
//! This module provides a trait-based approach to LaTeX rendering.
//! All renderers implement LaTeXRenderer trait which provides shared
//! implementations for expression rendering with customizable hooks.

use crate::display_context::DisplayContext;
use crate::latex_highlight::{HighlightColor, HighlightConfig};
use crate::{Constant, Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};
use regex::Regex;
use std::sync::LazyLock;

// ── Static regex constants for LaTeX negative-sign cleanup ──────────────────
static RE_PLUS_MINUS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\+ -([0-9a-zA-Z])").unwrap());
static RE_MINUS_MINUS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"- -([0-9a-zA-Z])").unwrap());
static RE_PLUS_COLOR_MINUS: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\+ (\{\\color\{[^}]+\}\{)-").unwrap());
static RE_MINUS_COLOR_MINUS: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"- (\{\\color\{[^}]+\}\{)-").unwrap());

/// Shared implementation for cleaning LaTeX negative-sign patterns.
/// Used by both `LaTeXRenderer::clean_latex_negatives` and
/// `PathHighlightedLatexRenderer::clean_latex_negatives`.
fn clean_latex_negatives_shared(latex: &str) -> String {
    let mut result = latex.to_string();

    // Fix "+ -" → "-" in all contexts
    result = result.replace("+ -\\", "- \\");
    result = result.replace("+ -(", "- (");

    // Fix "- -" → "+" (double negative)
    result = result.replace("- -\\", "+ \\");
    result = result.replace("- -(", "+ (");

    // Fix "+ -" before digits or letters (e.g., "+ -x" → "- x")
    result = RE_PLUS_MINUS.replace_all(&result, "- $1").to_string();

    // Fix "- -" before digits or letters (e.g., "- -x" → "+ x")
    result = RE_MINUS_MINUS.replace_all(&result, "+ $1").to_string();

    // Fix "+ {color command}{-" patterns (highlighted negatives)
    // e.g., "+ {\color{red}{-..." → "- {\color{red}{"
    result = RE_PLUS_COLOR_MINUS.replace_all(&result, "- $1").to_string();

    // Fix "- {color command}{-" patterns (double negative with highlight)
    result = RE_MINUS_COLOR_MINUS
        .replace_all(&result, "+ $1")
        .to_string();

    // Fix "+ -{" → "- {" (when minus precedes a brace group)
    result = result.replace("+ -{", "- {");

    // Fix "- -{" → "+ {" (double negative before brace)
    result = result.replace("- -{", "+ {");

    result
}

fn unwrap_internal_hold_for_latex(ctx: &Context, id: ExprId) -> ExprId {
    let mut current = id;
    loop {
        match ctx.get(current) {
            Expr::Hold(inner) => current = *inner,
            Expr::Function(fn_id, args)
                if args.len() == 1 && crate::hold::is_internal_hold_name(ctx.sym_name(*fn_id)) =>
            {
                current = args[0];
            }
            _ => return current,
        }
    }
}

fn is_add_sub_after_internal_hold(ctx: &Context, id: ExprId) -> bool {
    let id = unwrap_internal_hold_for_latex(ctx, id);
    matches!(ctx.get(id), Expr::Add(_, _) | Expr::Sub(_, _))
}

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
        clean_latex_negatives_shared(latex)
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
            Expr::Variable(sym_id) => ctx.sym_name(*sym_id).to_string(),
            Expr::Constant(c) => self.format_constant(c),
            Expr::Add(l, r) => self.format_add(*l, *r),
            Expr::Sub(l, r) => self.format_sub(*l, *r),
            Expr::Mul(l, r) => self.format_mul(*l, *r, parent_needs_parens),
            Expr::Div(l, r) => self.format_div(*l, *r),
            Expr::Pow(base, exp) => self.format_pow(*base, *exp),
            Expr::Neg(e) => self.format_neg(*e),
            Expr::Function(fn_id, args) => {
                self.format_function(self.context().sym_name(*fn_id), args)
            }
            Expr::Matrix { rows, cols, data } => self.format_matrix(*rows, *cols, data),
            Expr::SessionRef(id) => format!("\\#{}", id), // LaTeX escape
            // Hold is transparent for display - render inner directly
            Expr::Hold(inner) => self.format_expr(*inner, parent_needs_parens),
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
            Constant::Phi => "\\phi".to_string(),
        }
    }

    /// Format addition - flatten chain, sort by degree, render with proper signs
    fn format_add(&self, l: ExprId, r: ExprId) -> String {
        // Flatten the Add chain into individual terms
        let mut terms = Vec::new();
        self.collect_add_terms(l, &mut terms);
        self.collect_add_terms(r, &mut terms);

        // Sort by polynomial degree (descending) then sign (positive first)
        let ctx = self.context();
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
    fn collect_add_terms(&self, id: ExprId, terms: &mut Vec<ExprId>) {
        if self.get_highlight(id).is_some() {
            terms.push(id);
            return;
        }

        match self.context().get(id) {
            Expr::Add(l, r) => {
                self.collect_add_terms(*l, terms);
                self.collect_add_terms(*r, terms);
            }
            _ => terms.push(id),
        }
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
                let inner_is_add_sub = is_add_sub_after_internal_hold(ctx, *inner);
                if inner_is_add_sub {
                    (true, format!("({})", self.expr_to_latex(*inner, false)))
                } else {
                    (true, self.expr_to_latex(*inner, false))
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
                if let Some((coefficient, radicand)) = reciprocal_sqrt_numerator_for_latex(ctx, id)
                {
                    if !coefficient.is_negative() {
                        return (false, self.expr_to_latex(id, false));
                    }
                    let coefficient = coefficient.abs();
                    if coefficient.is_positive()
                        && coefficient.numer().is_one()
                        && !coefficient.denom().is_one()
                    {
                        let abs_latex =
                            self.format_reciprocal_sqrt_div_latex(coefficient, radicand, &[]);
                        let highlighted = if let Some(color) = self.get_highlight(id) {
                            format!("{{\\color{{{}}}{{{}}}}}", color.to_latex(), abs_latex)
                        } else {
                            abs_latex
                        };
                        return (true, highlighted);
                    }
                }
                if let Some(abs_latex) = self.direct_negative_mul_abs_latex(*ml, *mr) {
                    let highlighted = if let Some(color) = self.get_highlight(id) {
                        format!("{{\\color{{{}}}{{{}}}}}", color.to_latex(), abs_latex)
                    } else {
                        abs_latex
                    };
                    return (true, highlighted);
                }
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

    fn direct_negative_mul_abs_latex(&self, l: ExprId, r: ExprId) -> Option<String> {
        let left_neg = self.direct_negative_factor_latex(l);
        let right_neg = self.direct_negative_factor_latex(r);
        match (left_neg, right_neg) {
            (Some(left), None) => Some(format!("{}\\cdot {}", left, self.expr_to_latex_mul(r))),
            (None, Some(right)) => Some(format!("{}\\cdot {}", self.expr_to_latex_mul(l), right)),
            _ => None,
        }
    }

    fn direct_negative_factor_latex(&self, id: ExprId) -> Option<String> {
        match self.context().get(id) {
            Expr::Neg(inner) => Some(self.expr_to_latex_mul(*inner)),
            Expr::Number(n) if n.is_negative() => {
                let positive = -n;
                if positive.is_integer() {
                    Some(format!("{}", positive.numer()))
                } else {
                    Some(format!(
                        "\\frac{{{}}}{{{}}}",
                        positive.numer(),
                        positive.denom()
                    ))
                }
            }
            _ => None,
        }
    }

    /// Format subtraction
    fn format_sub(&self, l: ExprId, r: ExprId) -> String {
        let left = self.expr_to_latex(l, false);
        // Right operand needs parentheses if it's Add or Sub to preserve precedence
        // e.g., A - (B + C) must show parens, otherwise it looks like A - B + C
        let right = if is_add_sub_after_internal_hold(self.context(), r) {
            format!("({})", self.expr_to_latex(r, false))
        } else {
            self.expr_to_latex(r, false)
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

        if let Some(abs_latex) = self.direct_negative_mul_abs_latex(l, r) {
            return if parent_needs_parens {
                format!("(-{})", abs_latex)
            } else {
                format!("-{}", abs_latex)
            };
        }

        if let Some((coefficient, radicand, denominator)) =
            reciprocal_sqrt_times_unit_fraction_for_latex(self.context(), l, r)
        {
            return self.format_reciprocal_sqrt_div_latex(coefficient, radicand, &[denominator]);
        }
        if let Some((coefficient, radicand, denominators)) =
            reciprocal_sqrt_product_with_unit_fraction_for_latex(self.context(), l, r)
        {
            return self.format_reciprocal_sqrt_div_latex(coefficient, radicand, &denominators);
        }
        if let Some(fraction_product) =
            self.format_root_denominator_fraction_product_latex(l, r, parent_needs_parens)
        {
            return fraction_product;
        }

        // V2.14.40: Absorb fractional coefficient into fraction for cleaner display
        // Pattern: (1/n) * expr -> \frac{expr}{n}
        // Pattern: (k/n) * expr -> \frac{k \cdot expr}{n} or just k/n * expr (simpler)
        if let Expr::Number(n) = self.context().get(l) {
            // Check if left is a simple fraction 1/n (numerator = 1, denominator > 1)
            if !n.is_integer() && *n.numer() == 1.into() && *n.denom() > 1.into() {
                if let Some((coefficient, radicand)) =
                    reciprocal_sqrt_numerator_for_latex(self.context(), r)
                {
                    return self.format_reciprocal_sqrt_div_latex(
                        coefficient * n.clone(),
                        radicand,
                        &[],
                    );
                }
                let right_latex = self.expr_to_latex(r, false);
                return format!("\\frac{{{}}}{{{}}}", right_latex, n.denom());
            }
        }
        // Also check right side for (expr * 1/n) pattern
        if let Expr::Number(n) = self.context().get(r) {
            if !n.is_integer() && *n.numer() == 1.into() && *n.denom() > 1.into() {
                if let Some((coefficient, radicand)) =
                    reciprocal_sqrt_numerator_for_latex(self.context(), l)
                {
                    return self.format_reciprocal_sqrt_div_latex(
                        coefficient * n.clone(),
                        radicand,
                        &[],
                    );
                }
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

        if let Some((coefficient, radicand, denominators)) =
            reciprocal_sqrt_division_for_latex(ctx, l, r)
        {
            return self.format_reciprocal_sqrt_div_latex(coefficient, radicand, &denominators);
        }

        // Check if numerator is negative - put sign outside fraction
        match ctx.get(l) {
            Expr::Neg(inner) => {
                let numer = self.expr_to_latex(*inner, false);
                let denom = self.expr_to_latex_denominator(r);
                format!("-\\frac{{{}}}{{{}}}", numer, denom)
            }
            Expr::Number(n) if n.is_negative() => {
                let positive = -n;
                let numer_str = if positive.is_integer() {
                    format!("{}", positive.numer())
                } else {
                    format!("\\frac{{{}}}{{{}}}", positive.numer(), positive.denom())
                };
                let denom = self.expr_to_latex_denominator(r);
                format!("-\\frac{{{}}}{{{}}}", numer_str, denom)
            }
            _ => {
                let numer = self.expr_to_latex(l, false);
                let denom = self.expr_to_latex_denominator(r);
                format!("\\frac{{{}}}{{{}}}", numer, denom)
            }
        }
    }

    fn format_reciprocal_sqrt_div_latex(
        &self,
        mut coefficient: BigRational,
        radicand: ExprId,
        denominators: &[ExprId],
    ) -> String {
        let sqrt_radicand = self.expr_to_latex(radicand, false);
        let mut denominator_parts = Vec::new();
        let mut rest_denominator_parts = Vec::new();
        let mut only_sin_or_cos_sqrt_rest = true;
        for denominator in denominators {
            let mut denominator_factors = Vec::new();
            collect_mul_factors_for_latex(self.context(), *denominator, &mut denominator_factors);
            for factor in denominator_factors {
                let rendered = self.expr_to_latex_mul(factor);
                if let Some(value) = rational_constant_expr_for_latex(self.context(), factor)
                    .filter(|value| value.is_positive())
                {
                    coefficient /= value;
                } else {
                    only_sin_or_cos_sqrt_rest &=
                        is_sin_or_cos_of_latex_sqrt_factor(self.context(), factor);
                    rest_denominator_parts.push(rendered);
                }
            }
        }
        let sign = if coefficient.is_negative() { "-" } else { "" };
        let coefficient = coefficient.abs();
        let numerator = coefficient.numer().to_string();
        if !coefficient.denom().is_one() {
            denominator_parts.push(coefficient.denom().to_string());
        }
        let sqrt_before_rest = rest_denominator_parts.len() == 1 && only_sin_or_cos_sqrt_rest;
        if sqrt_before_rest {
            denominator_parts.push(format!("\\sqrt{{{}}}", sqrt_radicand));
        }
        denominator_parts.extend(rest_denominator_parts);
        if !sqrt_before_rest {
            denominator_parts.push(format!("\\sqrt{{{}}}", sqrt_radicand));
        }
        let denominator = denominator_parts.join("\\cdot ");
        format!("{sign}\\frac{{{}}}{{{}}}", numerator, denominator)
    }

    fn format_root_denominator_fraction_product_latex(
        &self,
        l: ExprId,
        r: ExprId,
        parent_needs_parens: bool,
    ) -> Option<String> {
        let mut numerator_coeff = BigRational::one();
        let mut numerator_factors = Vec::new();
        let mut denominator_factors = Vec::new();
        let mut saw_fraction = false;
        collect_fraction_product_parts_with_numerators_for_latex(
            self.context(),
            l,
            &mut numerator_coeff,
            &mut numerator_factors,
            &mut denominator_factors,
            &mut saw_fraction,
        )?;
        collect_fraction_product_parts_with_numerators_for_latex(
            self.context(),
            r,
            &mut numerator_coeff,
            &mut numerator_factors,
            &mut denominator_factors,
            &mut saw_fraction,
        )?;
        if !saw_fraction || denominator_factors.is_empty() {
            return None;
        }
        let mut denominator_coeff = BigRational::one();
        let mut structural_denominators = Vec::new();
        for factor in denominator_factors {
            if let Some(value) = rational_constant_expr_for_latex(self.context(), factor) {
                denominator_coeff *= value;
            } else {
                structural_denominators.push(factor);
            }
        }
        if structural_denominators.is_empty()
            || !denominator_has_sqrt_like_factor_for_latex(self.context(), &structural_denominators)
            || denominator_coeff.is_zero()
        {
            return None;
        }

        let coefficient = numerator_coeff / denominator_coeff;
        if coefficient.is_zero() {
            return None;
        }

        let sign = if coefficient.is_negative() { "-" } else { "" };
        let coefficient = coefficient.abs();
        let mut numerator_parts = Vec::new();
        if coefficient.numer() != &1.into() || numerator_factors.is_empty() {
            numerator_parts.push(coefficient.numer().to_string());
        }
        numerator_parts.extend(
            numerator_factors
                .iter()
                .map(|factor| self.expr_to_latex_mul(*factor)),
        );
        let mut denominator_parts = Vec::new();
        if !coefficient.denom().is_one() {
            denominator_parts.push(coefficient.denom().to_string());
        }
        let mut non_root_denominators = Vec::new();
        let mut root_denominators = Vec::new();
        for factor in structural_denominators {
            if is_sqrt_like_factor_for_latex(self.context(), factor) {
                root_denominators.push(factor);
            } else {
                non_root_denominators.push(factor);
            }
        }
        denominator_parts.extend(
            non_root_denominators
                .iter()
                .chain(root_denominators.iter())
                .map(|factor| self.expr_to_latex_mul(*factor)),
        );
        let numerator = numerator_parts.join("\\cdot ");
        let denominator = denominator_parts.join("\\cdot ");
        let rendered = format!("{sign}\\frac{{{numerator}}}{{{denominator}}}");
        Some(if parent_needs_parens {
            format!("({rendered})")
        } else {
            rendered
        })
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
        let inner_is_add_sub = is_add_sub_after_internal_hold(self.context(), e);
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
            "fact" | "factorial" if args.len() == 1 => {
                let needs_parens = matches!(
                    self.context().get(args[0]),
                    Expr::Add(_, _) | Expr::Sub(_, _) | Expr::Mul(_, _) | Expr::Div(_, _)
                );
                let arg = self.expr_to_latex(args[0], false);
                if needs_parens {
                    format!("({})!", arg)
                } else {
                    format!("{}!", arg)
                }
            }
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
            "log10" if args.len() == 1 => {
                format!("\\log_{{10}}({})", self.expr_to_latex(args[0], false))
            }
            "log" if args.len() == 2 => {
                // log(base, arg) where args[0]=base, args[1]=arg
                let base = self.expr_to_latex(args[0], false);
                let arg = self.expr_to_latex(args[1], false);
                format!("\\log_{{{}}}({})", base, arg)
            }
            "abs" => format!("|{}|", self.expr_to_latex(args[0], false)),
            "exp" => format!("e^{{{}}}", self.expr_to_latex(args[0], false)),
            // PlusMinus for quadratic formula display: PlusMinus(a, b) -> a ± b
            "PlusMinus" if args.len() == 2 => {
                let a = self.expr_to_latex(args[0], false);
                let b = self.expr_to_latex(args[1], false);
                format!("{} \\pm {}", a, b)
            }
            "floor" => format!("\\lfloor {} \\rfloor", self.expr_to_latex(args[0], false)),
            "ceil" => format!("\\lceil {} \\rceil", self.expr_to_latex(args[0], false)),
            "diff" if args.len() >= 2 => {
                let expr = self.expr_to_latex(args[0], false);
                let var = self.expr_to_latex(args[1], false);
                format!("\\frac{{d}}{{d{}}}({})", var, expr)
            }
            "integrate" if args.len() == 4 => {
                let expr = self.expr_to_latex(args[0], false);
                let var = self.expr_to_latex(args[1], false);
                let lower = self.expr_to_latex(args[2], false);
                let upper = self.expr_to_latex(args[3], false);
                format!("\\int_{{{}}}^{{{}}} {} \\, d{}", lower, upper, expr, var)
            }
            "integrate" if args.len() >= 2 => {
                let expr = self.expr_to_latex(args[0], false);
                let var = self.expr_to_latex(args[1], false);
                format!("\\int {} \\, d{}", expr, var)
            }
            "integrate" if args.len() == 1 => {
                let expr = self.expr_to_latex(args[0], false);
                format!("\\int {} \\, dx", expr)
            }
            // __eq__ is an internal equation representation - display as "lhs = rhs"
            _ if crate::eq::is_eq_name(name) && args.len() == 2 => {
                let lhs = self.expr_to_latex(args[0], false);
                let rhs = self.expr_to_latex(args[1], false);
                format!("{} = {}", lhs, rhs)
            }
            // ONLY internal __hold barrier is transparent - user-facing hold(...) is displayed
            _ if crate::hold::is_internal_hold_name(name) && args.len() == 1 => {
                self.expr_to_latex(args[0], false)
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
        let display_id = unwrap_internal_hold_for_latex(self.context(), id);
        match self.context().get(display_id) {
            Expr::Add(_, _) | Expr::Sub(_, _) => {
                format!("({})", self.expr_to_latex(id, false))
            }
            _ => self.expr_to_latex(id, false),
        }
    }

    fn expr_to_latex_denominator(&self, id: ExprId) -> String {
        if let Some(factors) = denominator_product_numeric_first_factors(self.context(), id) {
            return factors
                .iter()
                .map(|factor| self.expr_to_latex_mul(*factor))
                .collect::<Vec<_>>()
                .join("\\cdot ");
        }

        self.expr_to_latex(id, false)
    }

    /// Helper for power base - adds parens for composite expressions
    fn expr_to_latex_base(&self, id: ExprId) -> String {
        let display_id = unwrap_internal_hold_for_latex(self.context(), id);
        match self.context().get(display_id) {
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
            // A non-integer rational renders as `\frac{p}{q}`, unambiguous in rich LaTeX but de-LaTeXing
            // to a bare `p/q`; parenthesize so the plain-text form stays `(p/q)^e`, never `p/q^e` (which
            // re-parses as `p/(q^e)`).
            Expr::Number(n) if !n.is_integer() => {
                format!("({})", self.expr_to_latex(id, false))
            }
            _ => self.expr_to_latex(id, false),
        }
    }
}

fn reciprocal_sqrt_numerator_for_latex(ctx: &Context, id: ExprId) -> Option<(BigRational, ExprId)> {
    let mut factors = Vec::new();
    collect_mul_factors_for_latex(ctx, id, &mut factors);

    let mut coefficient = BigRational::one();
    let mut radicand = None;

    for factor in factors {
        match ctx.get(factor) {
            Expr::Number(n) => coefficient *= n.clone(),
            Expr::Pow(base, exp) if is_negative_one_half_exponent_for_latex(ctx, *exp) => {
                if radicand.replace(*base).is_some() {
                    return None;
                }
            }
            _ => return None,
        }
    }

    radicand.map(|radicand| (coefficient, radicand))
}

fn reciprocal_sqrt_division_for_latex(
    ctx: &Context,
    numerator: ExprId,
    denominator: ExprId,
) -> Option<(BigRational, ExprId, Vec<ExprId>)> {
    if let Some((coefficient, radicand)) = reciprocal_sqrt_numerator_for_latex(ctx, numerator) {
        return Some((coefficient, radicand, vec![denominator]));
    }

    match ctx.get(numerator) {
        Expr::Div(inner_num, inner_den) => {
            let (coefficient, radicand) = reciprocal_sqrt_numerator_for_latex(ctx, *inner_num)?;
            Some((coefficient, radicand, vec![*inner_den, denominator]))
        }
        _ => None,
    }
}

fn reciprocal_sqrt_times_unit_fraction_for_latex(
    ctx: &Context,
    left: ExprId,
    right: ExprId,
) -> Option<(BigRational, ExprId, ExprId)> {
    if let (Some((coefficient, radicand)), Some(denominator)) = (
        reciprocal_sqrt_numerator_for_latex(ctx, left),
        unit_fraction_denominator_for_latex(ctx, right),
    ) {
        return Some((coefficient, radicand, denominator));
    }

    if let (Some(denominator), Some((coefficient, radicand))) = (
        unit_fraction_denominator_for_latex(ctx, left),
        reciprocal_sqrt_numerator_for_latex(ctx, right),
    ) {
        return Some((coefficient, radicand, denominator));
    }

    None
}

fn reciprocal_sqrt_product_with_unit_fraction_for_latex(
    ctx: &Context,
    left: ExprId,
    right: ExprId,
) -> Option<(BigRational, ExprId, Vec<ExprId>)> {
    if let (Some((coefficient, radicand, mut denominators)), Some(denominator)) = (
        reciprocal_sqrt_factor_with_denominators_for_latex(ctx, left),
        unit_fraction_denominator_for_latex(ctx, right),
    ) {
        denominators.push(denominator);
        return Some((coefficient, radicand, denominators));
    }

    if let (Some(denominator), Some((coefficient, radicand, mut denominators))) = (
        unit_fraction_denominator_for_latex(ctx, left),
        reciprocal_sqrt_factor_with_denominators_for_latex(ctx, right),
    ) {
        denominators.push(denominator);
        return Some((coefficient, radicand, denominators));
    }

    if let (Some((coefficient, radicand, mut denominators)), Some((scale, mut extra))) = (
        reciprocal_sqrt_factor_with_denominators_for_latex(ctx, left),
        scalar_fraction_product_denominators_for_latex(ctx, right),
    ) {
        denominators.append(&mut extra);
        return Some((coefficient * scale, radicand, denominators));
    }

    if let (Some((scale, mut extra)), Some((coefficient, radicand, mut denominators))) = (
        scalar_fraction_product_denominators_for_latex(ctx, left),
        reciprocal_sqrt_factor_with_denominators_for_latex(ctx, right),
    ) {
        extra.append(&mut denominators);
        return Some((coefficient * scale, radicand, extra));
    }

    None
}

fn scalar_fraction_product_denominators_for_latex(
    ctx: &Context,
    id: ExprId,
) -> Option<(BigRational, Vec<ExprId>)> {
    let mut numerator_coeff = BigRational::one();
    let mut denominator_factors = Vec::new();
    let mut saw_fraction = false;
    collect_fraction_product_parts_for_latex(
        ctx,
        id,
        &mut numerator_coeff,
        &mut denominator_factors,
        &mut saw_fraction,
    )?;
    saw_fraction.then_some((numerator_coeff, denominator_factors))
}

fn reciprocal_sqrt_factor_with_denominators_for_latex(
    ctx: &Context,
    id: ExprId,
) -> Option<(BigRational, ExprId, Vec<ExprId>)> {
    if let Some((coefficient, radicand)) = reciprocal_sqrt_numerator_for_latex(ctx, id) {
        return Some((coefficient, radicand, Vec::new()));
    }

    match ctx.get(id) {
        Expr::Div(num, den) => {
            let (coefficient, radicand) = reciprocal_sqrt_numerator_for_latex(ctx, *num)?;
            Some((coefficient, radicand, vec![*den]))
        }
        _ => None,
    }
}

fn unit_fraction_denominator_for_latex(ctx: &Context, id: ExprId) -> Option<ExprId> {
    match ctx.get(id) {
        Expr::Div(num, den) if matches!(ctx.get(*num), Expr::Number(n) if n.is_one()) => Some(*den),
        _ => None,
    }
}

fn collect_mul_factors_for_latex(ctx: &Context, id: ExprId, out: &mut Vec<ExprId>) {
    let id = unwrap_internal_hold_for_latex(ctx, id);
    match ctx.get(id) {
        Expr::Mul(l, r) => {
            collect_mul_factors_for_latex(ctx, *l, out);
            collect_mul_factors_for_latex(ctx, *r, out);
        }
        _ => out.push(id),
    }
}

fn collect_fraction_product_parts_for_latex(
    ctx: &Context,
    id: ExprId,
    numerator_coeff: &mut BigRational,
    denominator_factors: &mut Vec<ExprId>,
    saw_fraction: &mut bool,
) -> Option<()> {
    let id = unwrap_internal_hold_for_latex(ctx, id);
    match ctx.get(id) {
        Expr::Mul(left, right) => {
            collect_fraction_product_parts_for_latex(
                ctx,
                *left,
                numerator_coeff,
                denominator_factors,
                saw_fraction,
            )?;
            collect_fraction_product_parts_for_latex(
                ctx,
                *right,
                numerator_coeff,
                denominator_factors,
                saw_fraction,
            )
        }
        Expr::Div(num, den) => {
            *numerator_coeff *= rational_constant_expr_for_latex(ctx, *num)?;
            collect_mul_factors_for_latex(ctx, *den, denominator_factors);
            *saw_fraction = true;
            Some(())
        }
        _ => {
            *numerator_coeff *= rational_constant_expr_for_latex(ctx, id)?;
            Some(())
        }
    }
}

fn collect_fraction_product_parts_with_numerators_for_latex(
    ctx: &Context,
    id: ExprId,
    numerator_coeff: &mut BigRational,
    numerator_factors: &mut Vec<ExprId>,
    denominator_factors: &mut Vec<ExprId>,
    saw_fraction: &mut bool,
) -> Option<()> {
    let id = unwrap_internal_hold_for_latex(ctx, id);
    match ctx.get(id) {
        Expr::Mul(left, right) => {
            collect_fraction_product_parts_with_numerators_for_latex(
                ctx,
                *left,
                numerator_coeff,
                numerator_factors,
                denominator_factors,
                saw_fraction,
            )?;
            collect_fraction_product_parts_with_numerators_for_latex(
                ctx,
                *right,
                numerator_coeff,
                numerator_factors,
                denominator_factors,
                saw_fraction,
            )
        }
        Expr::Div(num, den) => {
            collect_fraction_product_numerator_for_latex(
                ctx,
                *num,
                numerator_coeff,
                numerator_factors,
            )?;
            collect_mul_factors_for_latex(ctx, *den, denominator_factors);
            *saw_fraction = true;
            Some(())
        }
        Expr::Number(value) => {
            *numerator_coeff *= value.clone();
            Some(())
        }
        _ => {
            numerator_factors.push(id);
            Some(())
        }
    }
}

fn collect_fraction_product_numerator_for_latex(
    ctx: &Context,
    id: ExprId,
    numerator_coeff: &mut BigRational,
    numerator_factors: &mut Vec<ExprId>,
) -> Option<()> {
    let id = unwrap_internal_hold_for_latex(ctx, id);
    match ctx.get(id) {
        Expr::Mul(left, right) => {
            collect_fraction_product_numerator_for_latex(
                ctx,
                *left,
                numerator_coeff,
                numerator_factors,
            )?;
            collect_fraction_product_numerator_for_latex(
                ctx,
                *right,
                numerator_coeff,
                numerator_factors,
            )
        }
        Expr::Number(value) => {
            *numerator_coeff *= value.clone();
            Some(())
        }
        Expr::Div(_, _) => None,
        _ => {
            numerator_factors.push(id);
            Some(())
        }
    }
}

fn denominator_has_sqrt_like_factor_for_latex(ctx: &Context, factors: &[ExprId]) -> bool {
    factors
        .iter()
        .any(|factor| is_sqrt_like_factor_for_latex(ctx, *factor))
}

fn is_sqrt_like_factor_for_latex(ctx: &Context, factor: ExprId) -> bool {
    let factor = unwrap_internal_hold_for_latex(ctx, factor);
    match ctx.get(factor) {
        Expr::Function(fn_id, args) => args.len() == 1 && ctx.sym_name(*fn_id) == "sqrt",
        Expr::Pow(_, exp) => is_positive_one_half_exponent_for_latex(ctx, *exp),
        _ => false,
    }
}

fn denominator_product_numeric_first_factors(ctx: &Context, id: ExprId) -> Option<Vec<ExprId>> {
    let mut factors = Vec::new();
    collect_mul_factors_for_latex(ctx, id, &mut factors);
    numeric_first_product_factors_latex(ctx, &factors)
}

fn numeric_first_product_factors_latex(ctx: &Context, factors: &[ExprId]) -> Option<Vec<ExprId>> {
    if factors.len() < 2 {
        return None;
    }

    let mut numeric = Vec::new();
    let mut rest = Vec::new();
    for factor in factors {
        if matches!(ctx.get(*factor), Expr::Number(n) if n.is_positive()) {
            numeric.push(*factor);
        } else {
            rest.push(*factor);
        }
    }

    if numeric.is_empty() || rest.is_empty() {
        return None;
    }

    let reordered = numeric.into_iter().chain(rest).collect::<Vec<_>>();
    (reordered != factors).then_some(reordered)
}

fn is_negative_one_half_exponent_for_latex(ctx: &Context, id: ExprId) -> bool {
    match ctx.get(id) {
        Expr::Number(n) => *n == BigRational::new((-1).into(), 2.into()),
        Expr::Div(num, den) => {
            matches!(ctx.get(*num), Expr::Number(n) if *n == BigRational::from_integer((-1).into()))
                && matches!(ctx.get(*den), Expr::Number(n) if *n == BigRational::from_integer(2.into()))
        }
        Expr::Neg(inner) => is_positive_one_half_exponent_for_latex(ctx, *inner),
        Expr::Add(left, right) => rational_constant_expr_for_latex(ctx, *left)
            .zip(rational_constant_expr_for_latex(ctx, *right))
            .is_some_and(|(left, right)| left + right == BigRational::new((-1).into(), 2.into())),
        Expr::Sub(left, right) => rational_constant_expr_for_latex(ctx, *left)
            .zip(rational_constant_expr_for_latex(ctx, *right))
            .is_some_and(|(left, right)| left - right == BigRational::new((-1).into(), 2.into())),
        _ => false,
    }
}

fn rational_constant_expr_for_latex(ctx: &Context, id: ExprId) -> Option<BigRational> {
    match ctx.get(id) {
        Expr::Number(n) => Some(n.clone()),
        Expr::Neg(inner) => Some(-rational_constant_expr_for_latex(ctx, *inner)?),
        Expr::Add(left, right) => Some(
            rational_constant_expr_for_latex(ctx, *left)?
                + rational_constant_expr_for_latex(ctx, *right)?,
        ),
        Expr::Sub(left, right) => Some(
            rational_constant_expr_for_latex(ctx, *left)?
                - rational_constant_expr_for_latex(ctx, *right)?,
        ),
        Expr::Div(num, den) => {
            let denominator = rational_constant_expr_for_latex(ctx, *den)?;
            if denominator.is_zero() {
                None
            } else {
                Some(rational_constant_expr_for_latex(ctx, *num)? / denominator)
            }
        }
        _ => None,
    }
}

fn is_positive_one_half_exponent_for_latex(ctx: &Context, id: ExprId) -> bool {
    match ctx.get(id) {
        Expr::Number(n) => *n == BigRational::new(1.into(), 2.into()),
        Expr::Div(num, den) => {
            matches!(ctx.get(*num), Expr::Number(n) if *n == BigRational::from_integer(1.into()))
                && matches!(ctx.get(*den), Expr::Number(n) if *n == BigRational::from_integer(2.into()))
        }
        _ => false,
    }
}

fn is_sin_or_cos_of_latex_sqrt_factor(ctx: &Context, id: ExprId) -> bool {
    let id = unwrap_internal_hold_for_latex(ctx, id);
    let Expr::Function(fn_id, args) = ctx.get(id) else {
        return false;
    };
    if args.len() != 1 || !matches!(ctx.sym_name(*fn_id), "sin" | "cos") {
        return false;
    }
    let arg = unwrap_internal_hold_for_latex(ctx, args[0]);
    match ctx.get(arg) {
        Expr::Function(inner_fn, inner_args) => {
            inner_args.len() == 1 && ctx.sym_name(*inner_fn) == "sqrt"
        }
        Expr::Pow(_, exp) => is_positive_one_half_exponent_for_latex(ctx, *exp),
        _ => false,
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

type ReciprocalSqrtDivisionLatexParts = (BigRational, ExprId, ExprPath, Vec<(ExprId, ExprPath)>);

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
        clean_latex_negatives_shared(latex)
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
            if is_full_color_highlight(&inner, color) {
                return inner;
            }
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
            Expr::Variable(sym_id) => self.context.sym_name(*sym_id).to_string(),
            Expr::Constant(c) => self.format_constant(c),
            Expr::Add(_, _) | Expr::Sub(_, _) => self.format_additive_path(id, current_path),
            Expr::Mul(l, r) => self.format_mul_path(*l, *r, parent_needs_parens, current_path),
            Expr::Div(l, r) => self.format_div_path(*l, *r, current_path),
            Expr::Pow(base, exp) => self.format_pow_path(*base, *exp, current_path),
            Expr::Neg(e) => self.format_neg_path(*e, current_path),
            Expr::Function(fn_id, args) => {
                self.format_function_path(self.context.sym_name(*fn_id), args, current_path)
            }
            Expr::Matrix { rows, cols, data } => {
                self.format_matrix_path(*rows, *cols, data, current_path)
            }
            Expr::SessionRef(id) => format!("\\#{}", id),
            // Hold is transparent for display - render inner directly
            Expr::Hold(inner) => self.render_with_path(*inner, parent_needs_parens, current_path),
        }
    }

    fn child_path(&self, current: &ExprPath, child_idx: u8) -> ExprPath {
        let mut p = current.clone();
        p.push(child_idx);
        p
    }

    fn collect_signed_add_terms_path(
        &self,
        id: ExprId,
        path: &ExprPath,
        barrier_root: &ExprPath,
        invert_sign: bool,
        terms: &mut Vec<(ExprId, ExprPath, bool)>,
    ) {
        if path != barrier_root && self.path_highlights.get(path).is_some() {
            terms.push((id, path.clone(), invert_sign));
            return;
        }

        match self.context.get(id) {
            Expr::Add(l, r) => {
                self.collect_signed_add_terms_path(
                    *l,
                    &self.child_path(path, 0),
                    barrier_root,
                    invert_sign,
                    terms,
                );
                self.collect_signed_add_terms_path(
                    *r,
                    &self.child_path(path, 1),
                    barrier_root,
                    invert_sign,
                    terms,
                );
            }
            Expr::Sub(l, r) => {
                self.collect_signed_add_terms_path(
                    *l,
                    &self.child_path(path, 0),
                    barrier_root,
                    invert_sign,
                    terms,
                );
                self.collect_signed_add_terms_path(
                    *r,
                    &self.child_path(path, 1),
                    barrier_root,
                    !invert_sign,
                    terms,
                );
            }
            _ => terms.push((id, path.clone(), invert_sign)),
        }
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
            Constant::Phi => "\\phi".to_string(),
        }
    }

    fn format_additive_path(&self, id: ExprId, path: &ExprPath) -> String {
        let mut terms = Vec::new();
        self.collect_signed_add_terms_path(id, path, path, false, &mut terms);

        let mut result = String::new();
        for (i, (term_id, term_path, inherited_neg)) in terms.iter().enumerate() {
            let (term_neg, term_str) = self.term_to_latex_with_sign_path(*term_id, term_path);
            let is_neg = *inherited_neg ^ term_neg;
            if i == 0 {
                if is_neg {
                    result.push('-');
                    result.push_str(&term_str);
                } else {
                    result.push_str(&term_str);
                }
            } else if is_neg {
                result.push_str(" - ");
                result.push_str(&term_str);
            } else {
                result.push_str(" + ");
                result.push_str(&term_str);
            }
        }

        result
    }

    fn term_to_latex_with_sign_path(&self, id: ExprId, path: &ExprPath) -> (bool, String) {
        match self.context.get(id) {
            Expr::Neg(inner) => {
                let inner_is_add_sub = is_add_sub_after_internal_hold(self.context, *inner);
                if inner_is_add_sub {
                    (
                        true,
                        format!(
                            "({})",
                            self.render_with_path(*inner, false, &self.child_path(path, 0))
                        ),
                    )
                } else {
                    (
                        true,
                        self.render_with_path(*inner, false, &self.child_path(path, 0)),
                    )
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
            Expr::Mul(l, r) => {
                if let Some(abs_latex) = self.direct_negative_mul_abs_latex_path(*l, *r, path) {
                    let highlighted = if let Some(color) = self.path_highlights.get(path) {
                        format!("{{\\color{{{}}}{{{}}}}}", color.to_latex(), abs_latex)
                    } else {
                        abs_latex
                    };
                    return (true, highlighted);
                }
                (false, self.render_with_path(id, false, path))
            }
            _ => (false, self.render_with_path(id, false, path)),
        }
    }

    fn direct_negative_mul_abs_latex_path(
        &self,
        l: ExprId,
        r: ExprId,
        path: &ExprPath,
    ) -> Option<String> {
        let left_neg = self.direct_negative_factor_latex_path(l, &self.child_path(path, 0));
        let right_neg = self.direct_negative_factor_latex_path(r, &self.child_path(path, 1));
        match (left_neg, right_neg) {
            (Some(left), None) => Some(format!(
                "{}\\cdot {}",
                left,
                self.render_mul_operand(r, &self.child_path(path, 1))
            )),
            (None, Some(right)) => Some(format!(
                "{}\\cdot {}",
                self.render_mul_operand(l, &self.child_path(path, 0)),
                right
            )),
            _ => None,
        }
    }

    fn direct_negative_factor_latex_path(&self, id: ExprId, path: &ExprPath) -> Option<String> {
        let display_id = unwrap_internal_hold_for_latex(self.context, id);
        match self.context.get(display_id) {
            Expr::Neg(inner) => Some(self.render_mul_operand(*inner, &self.child_path(path, 0))),
            Expr::Number(n) if n.is_negative() => {
                let positive = -n;
                if positive.is_integer() {
                    Some(format!("{}", positive.numer()))
                } else {
                    Some(format!(
                        "\\frac{{{}}}{{{}}}",
                        positive.numer(),
                        positive.denom()
                    ))
                }
            }
            _ => None,
        }
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

        if let Some(abs_latex) = self.direct_negative_mul_abs_latex_path(l, r, path) {
            return if parent_needs_parens {
                format!("(-{})", abs_latex)
            } else {
                format!("-{}", abs_latex)
            };
        }

        if let Some((coefficient, radicand, radicand_path, denominator, denominator_path)) =
            self.reciprocal_sqrt_times_unit_fraction_for_latex_path(l, r, path)
        {
            return self.format_reciprocal_sqrt_with_denominators_path(
                coefficient,
                radicand,
                &radicand_path,
                &[(denominator, denominator_path)],
            );
        }
        if let Some((coefficient, radicand, denominators)) =
            reciprocal_sqrt_product_with_unit_fraction_for_latex(self.context, l, r)
        {
            let fallback_path = self.child_path(path, 0);
            let denominator_paths = denominators
                .into_iter()
                .map(|denominator| (denominator, fallback_path.clone()))
                .collect::<Vec<_>>();
            return self.format_reciprocal_sqrt_with_denominators_path(
                coefficient,
                radicand,
                &fallback_path,
                &denominator_paths,
            );
        }
        if let Some(fraction_product) = self.format_root_denominator_fraction_product_latex_path(
            l,
            r,
            parent_needs_parens,
            path,
        ) {
            return fraction_product;
        }

        // V2.14.40: Absorb fractional coefficient into fraction for cleaner display
        // Pattern: (1/n) * expr -> \frac{expr}{n}
        if let Expr::Number(n) = self.context.get(l) {
            if !n.is_integer() && *n.numer() == 1.into() && *n.denom() > 1.into() {
                if let Some((coefficient, radicand, radicand_path)) =
                    self.reciprocal_sqrt_numerator_for_latex_path(r, &self.child_path(path, 1))
                {
                    return self.format_reciprocal_sqrt_product_path(
                        coefficient * n.clone(),
                        radicand,
                        &radicand_path,
                    );
                }
                let right_latex = self.render_with_path(r, false, &self.child_path(path, 1));
                return format!("\\frac{{{}}}{{{}}}", right_latex, n.denom());
            }
        }
        // Also check right side for (expr * 1/n) pattern
        if let Expr::Number(n) = self.context.get(r) {
            if !n.is_integer() && *n.numer() == 1.into() && *n.denom() > 1.into() {
                if let Some((coefficient, radicand, radicand_path)) =
                    self.reciprocal_sqrt_numerator_for_latex_path(l, &self.child_path(path, 0))
                {
                    return self.format_reciprocal_sqrt_product_path(
                        coefficient * n.clone(),
                        radicand,
                        &radicand_path,
                    );
                }
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
        let display_id = unwrap_internal_hold_for_latex(self.context, id);
        match self.context.get(display_id) {
            Expr::Add(_, _) | Expr::Sub(_, _) => {
                format!("({})", self.render_with_path(id, false, path))
            }
            _ => self.render_with_path(id, false, path),
        }
    }

    fn format_root_denominator_fraction_product_latex_path(
        &self,
        l: ExprId,
        r: ExprId,
        parent_needs_parens: bool,
        path: &ExprPath,
    ) -> Option<String> {
        let mut numerator_coeff = BigRational::one();
        let mut numerator_factors = Vec::new();
        let mut denominator_factors = Vec::new();
        let mut saw_fraction = false;
        collect_fraction_product_parts_with_numerators_for_latex(
            self.context,
            l,
            &mut numerator_coeff,
            &mut numerator_factors,
            &mut denominator_factors,
            &mut saw_fraction,
        )?;
        collect_fraction_product_parts_with_numerators_for_latex(
            self.context,
            r,
            &mut numerator_coeff,
            &mut numerator_factors,
            &mut denominator_factors,
            &mut saw_fraction,
        )?;
        if !saw_fraction || denominator_factors.is_empty() {
            return None;
        }

        let mut denominator_coeff = BigRational::one();
        let mut non_root_denominators = Vec::new();
        let mut root_denominators = Vec::new();
        for factor in denominator_factors {
            if let Some(value) = rational_constant_expr_for_latex(self.context, factor) {
                denominator_coeff *= value;
            } else if is_sqrt_like_factor_for_latex(self.context, factor) {
                root_denominators.push(factor);
            } else {
                non_root_denominators.push(factor);
            }
        }
        if root_denominators.is_empty() || denominator_coeff.is_zero() {
            return None;
        }

        let coefficient = numerator_coeff / denominator_coeff;
        if coefficient.is_zero() {
            return None;
        }

        let sign = if coefficient.is_negative() { "-" } else { "" };
        let coefficient = coefficient.abs();
        let fallback_path = self.child_path(path, 0);
        let mut numerator_parts = Vec::new();
        if coefficient.numer() != &1.into() || numerator_factors.is_empty() {
            numerator_parts.push(coefficient.numer().to_string());
        }
        numerator_parts.extend(
            numerator_factors
                .iter()
                .map(|factor| self.render_mul_operand(*factor, &fallback_path)),
        );
        let mut denominator_parts = Vec::new();
        if !coefficient.denom().is_one() {
            denominator_parts.push(coefficient.denom().to_string());
        }
        denominator_parts.extend(
            non_root_denominators
                .iter()
                .chain(root_denominators.iter())
                .map(|factor| self.render_mul_operand(*factor, &fallback_path)),
        );
        let numerator = numerator_parts.join("\\cdot ");
        let denominator = denominator_parts.join("\\cdot ");
        let rendered = format!("{sign}\\frac{{{numerator}}}{{{denominator}}}");
        Some(if parent_needs_parens {
            format!("({rendered})")
        } else {
            rendered
        })
    }

    fn format_div_path(&self, l: ExprId, r: ExprId, path: &ExprPath) -> String {
        if let Some((coefficient, radicand, radicand_path, denominators)) =
            self.reciprocal_sqrt_division_for_latex_path(l, r, path)
        {
            return self.format_reciprocal_sqrt_with_denominators_path(
                coefficient,
                radicand,
                &radicand_path,
                &denominators,
            );
        }

        let numerator_path = self.child_path(path, 0);
        let denominator_path = self.child_path(path, 1);
        let denom = self.render_denominator_path(r, &denominator_path);

        match self.context.get(l) {
            Expr::Neg(inner) if self.path_highlights.get(&numerator_path).is_none() => {
                let inner_path = self.child_path(&numerator_path, 0);
                let numer = self.render_with_path(*inner, false, &inner_path);
                format!("-\\frac{{{}}}{{{}}}", numer, denom)
            }
            Expr::Number(n)
                if n.is_negative() && self.path_highlights.get(&numerator_path).is_none() =>
            {
                let positive = -n;
                let numer = if positive.is_integer() {
                    positive.numer().to_string()
                } else {
                    format!("\\frac{{{}}}{{{}}}", positive.numer(), positive.denom())
                };
                format!("-\\frac{{{}}}{{{}}}", numer, denom)
            }
            _ => {
                let numer = self.render_with_path(l, false, &numerator_path);
                format!("\\frac{{{}}}{{{}}}", numer, denom)
            }
        }
    }

    fn format_reciprocal_sqrt_with_denominators_path(
        &self,
        mut coefficient: BigRational,
        radicand: ExprId,
        radicand_path: &ExprPath,
        denominators: &[(ExprId, ExprPath)],
    ) -> String {
        let sqrt_radicand = self.render_with_path(radicand, false, radicand_path);
        let mut denominator_parts = Vec::new();
        let mut rest_denominator_parts = Vec::new();
        let mut only_sin_or_cos_sqrt_rest = true;
        for (denominator, denominator_path) in denominators {
            let mut denominator_factors = Vec::new();
            self.collect_mul_factors_for_latex_path(
                *denominator,
                denominator_path,
                &mut denominator_factors,
            );
            for (factor, factor_path) in denominator_factors {
                let rendered = self.render_mul_operand(factor, &factor_path);
                if let Some(value) = rational_constant_expr_for_latex(self.context, factor)
                    .filter(|value| value.is_positive())
                {
                    coefficient /= value;
                } else {
                    only_sin_or_cos_sqrt_rest &=
                        is_sin_or_cos_of_latex_sqrt_factor(self.context, factor);
                    rest_denominator_parts.push(rendered);
                }
            }
        }
        let sign = if coefficient.is_negative() { "-" } else { "" };
        let coefficient = coefficient.abs();
        let numerator = coefficient.numer().to_string();
        if !coefficient.denom().is_one() {
            denominator_parts.push(coefficient.denom().to_string());
        }
        let sqrt_before_rest = rest_denominator_parts.len() == 1 && only_sin_or_cos_sqrt_rest;
        if sqrt_before_rest {
            denominator_parts.push(format!("\\sqrt{{{}}}", sqrt_radicand));
        }
        denominator_parts.extend(rest_denominator_parts);
        if !sqrt_before_rest {
            denominator_parts.push(format!("\\sqrt{{{}}}", sqrt_radicand));
        }
        format!(
            "{sign}\\frac{{{}}}{{{}}}",
            numerator,
            denominator_parts.join("\\cdot ")
        )
    }

    fn format_reciprocal_sqrt_product_path(
        &self,
        coefficient: BigRational,
        radicand: ExprId,
        radicand_path: &ExprPath,
    ) -> String {
        let numerator = coefficient.numer().to_string();
        let sqrt_radicand = self.render_with_path(radicand, false, radicand_path);
        let mut denominator_parts = Vec::new();
        if !coefficient.denom().is_one() {
            denominator_parts.push(coefficient.denom().to_string());
        }
        denominator_parts.push(format!("\\sqrt{{{}}}", sqrt_radicand));
        format!(
            "\\frac{{{}}}{{{}}}",
            numerator,
            denominator_parts.join("\\cdot ")
        )
    }

    fn reciprocal_sqrt_times_unit_fraction_for_latex_path(
        &self,
        left: ExprId,
        right: ExprId,
        path: &ExprPath,
    ) -> Option<(BigRational, ExprId, ExprPath, ExprId, ExprPath)> {
        let left_path = self.child_path(path, 0);
        let right_path = self.child_path(path, 1);

        if let (
            Some((coefficient, radicand, radicand_path)),
            Some((denominator, denominator_path)),
        ) = (
            self.reciprocal_sqrt_numerator_for_latex_path(left, &left_path),
            self.unit_fraction_denominator_for_latex_path(right, &right_path),
        ) {
            return Some((
                coefficient,
                radicand,
                radicand_path,
                denominator,
                denominator_path,
            ));
        }

        if let (
            Some((denominator, denominator_path)),
            Some((coefficient, radicand, radicand_path)),
        ) = (
            self.unit_fraction_denominator_for_latex_path(left, &left_path),
            self.reciprocal_sqrt_numerator_for_latex_path(right, &right_path),
        ) {
            return Some((
                coefficient,
                radicand,
                radicand_path,
                denominator,
                denominator_path,
            ));
        }

        None
    }

    fn unit_fraction_denominator_for_latex_path(
        &self,
        id: ExprId,
        path: &ExprPath,
    ) -> Option<(ExprId, ExprPath)> {
        match self.context.get(id) {
            Expr::Div(num, den) if matches!(self.context.get(*num), Expr::Number(n) if n.is_one()) => {
                Some((*den, self.child_path(path, 1)))
            }
            _ => None,
        }
    }

    fn reciprocal_sqrt_division_for_latex_path(
        &self,
        numerator: ExprId,
        denominator: ExprId,
        path: &ExprPath,
    ) -> Option<ReciprocalSqrtDivisionLatexParts> {
        let numerator_path = self.child_path(path, 0);
        let denominator_path = self.child_path(path, 1);

        if let Some((coefficient, radicand, radicand_path)) =
            self.reciprocal_sqrt_numerator_for_latex_path(numerator, &numerator_path)
        {
            return Some((
                coefficient,
                radicand,
                radicand_path,
                vec![(denominator, denominator_path)],
            ));
        }

        match self.context.get(numerator) {
            Expr::Div(inner_num, inner_den) => {
                let inner_num_path = self.child_path(&numerator_path, 0);
                let inner_den_path = self.child_path(&numerator_path, 1);
                let (coefficient, radicand, radicand_path) =
                    self.reciprocal_sqrt_numerator_for_latex_path(*inner_num, &inner_num_path)?;
                Some((
                    coefficient,
                    radicand,
                    radicand_path,
                    vec![
                        (*inner_den, inner_den_path),
                        (denominator, denominator_path),
                    ],
                ))
            }
            _ => None,
        }
    }

    fn reciprocal_sqrt_numerator_for_latex_path(
        &self,
        id: ExprId,
        path: &ExprPath,
    ) -> Option<(BigRational, ExprId, ExprPath)> {
        let mut factors = Vec::new();
        self.collect_mul_factors_for_latex_path(id, path, &mut factors);

        let mut coefficient = BigRational::one();
        let mut radicand = None;

        for (factor, factor_path) in factors {
            match self.context.get(factor) {
                Expr::Number(n) if n.is_positive() => coefficient *= n.clone(),
                Expr::Pow(base, exp)
                    if is_negative_one_half_exponent_for_latex(self.context, *exp) =>
                {
                    let radicand_path = self.child_path(&factor_path, 0);
                    if radicand.replace((*base, radicand_path)).is_some() {
                        return None;
                    }
                }
                _ => return None,
            }
        }

        radicand.map(|(radicand, radicand_path)| (coefficient, radicand, radicand_path))
    }

    fn collect_mul_factors_for_latex_path(
        &self,
        id: ExprId,
        path: &ExprPath,
        out: &mut Vec<(ExprId, ExprPath)>,
    ) {
        let display_id = unwrap_internal_hold_for_latex(self.context, id);
        match self.context.get(display_id) {
            Expr::Mul(l, r) => {
                self.collect_mul_factors_for_latex_path(*l, &self.child_path(path, 0), out);
                self.collect_mul_factors_for_latex_path(*r, &self.child_path(path, 1), out);
            }
            _ => out.push((display_id, path.clone())),
        }
    }

    fn render_denominator_path(&self, id: ExprId, path: &ExprPath) -> String {
        if let Some(factors) = self.denominator_product_numeric_first_path(id, path) {
            return factors
                .iter()
                .map(|(factor, factor_path)| self.render_mul_operand(*factor, factor_path))
                .collect::<Vec<_>>()
                .join("\\cdot ");
        }

        self.render_with_path(id, false, path)
    }

    fn denominator_product_numeric_first_path(
        &self,
        id: ExprId,
        path: &ExprPath,
    ) -> Option<Vec<(ExprId, ExprPath)>> {
        let mut factors = Vec::new();
        self.collect_mul_factors_for_latex_path(id, path, &mut factors);
        if factors.len() < 2 {
            return None;
        }

        let mut numeric = Vec::new();
        let mut rest = Vec::new();
        for factor in factors.iter() {
            if matches!(self.context.get(factor.0), Expr::Number(n) if n.is_positive()) {
                numeric.push(factor.clone());
            } else {
                rest.push(factor.clone());
            }
        }

        if numeric.is_empty() || rest.is_empty() {
            return None;
        }

        let reordered = numeric.into_iter().chain(rest).collect::<Vec<_>>();
        (reordered != factors).then_some(reordered)
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
            // A non-integer rational de-LaTeXes to a bare `p/q`; parenthesize so the plain-text power
            // stays `(p/q)^e`, never the misparsing `p/q^e`.
            Expr::Number(n) if !n.is_integer() => {
                format!("({})", self.render_with_path(id, false, path))
            }
            _ => self.render_with_path(id, false, path),
        }
    }

    fn format_neg_path(&self, e: ExprId, path: &ExprPath) -> String {
        let inner_is_add_sub = is_add_sub_after_internal_hold(self.context, e);
        let inner = self.render_with_path(e, true, &self.child_path(path, 0));
        if inner_is_add_sub {
            format!("-({})", inner)
        } else {
            format!("-{}", inner)
        }
    }

    fn format_function_path(&self, name: &str, args: &[ExprId], path: &ExprPath) -> String {
        match name {
            "fact" | "factorial" if args.len() == 1 => {
                let needs_parens = matches!(
                    self.context.get(args[0]),
                    Expr::Add(_, _) | Expr::Sub(_, _) | Expr::Mul(_, _) | Expr::Div(_, _)
                );
                let arg = self.render_with_path(args[0], false, &self.child_path(path, 0));
                if needs_parens {
                    format!("({})!", arg)
                } else {
                    format!("{}!", arg)
                }
            }
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
            "log10" if args.len() == 1 => format!(
                "\\log_{{10}}({})",
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
            "diff" if args.len() >= 2 => {
                let expr = self.render_with_path(args[0], false, &self.child_path(path, 0));
                let var = self.render_with_path(args[1], false, &self.child_path(path, 1));
                format!("\\frac{{d}}{{d{}}}({})", var, expr)
            }
            "integrate" if args.len() == 4 => {
                let expr = self.render_with_path(args[0], false, &self.child_path(path, 0));
                let var = self.render_with_path(args[1], false, &self.child_path(path, 1));
                let lower = self.render_with_path(args[2], false, &self.child_path(path, 2));
                let upper = self.render_with_path(args[3], false, &self.child_path(path, 3));
                format!("\\int_{{{}}}^{{{}}} {} \\, d{}", lower, upper, expr, var)
            }
            "integrate" if args.len() >= 2 => {
                let expr = self.render_with_path(args[0], false, &self.child_path(path, 0));
                let var = self.render_with_path(args[1], false, &self.child_path(path, 1));
                format!("\\int {} \\, d{}", expr, var)
            }
            "integrate" if args.len() == 1 => {
                let expr = self.render_with_path(args[0], false, &self.child_path(path, 0));
                format!("\\int {} \\, dx", expr)
            }
            // __eq__ is an internal equation representation - display as "lhs = rhs"
            _ if crate::eq::is_eq_name(name) && args.len() == 2 => {
                let lhs = self.render_with_path(args[0], false, &self.child_path(path, 0));
                let rhs = self.render_with_path(args[1], false, &self.child_path(path, 1));
                format!("{} = {}", lhs, rhs)
            }
            // ONLY internal __hold barrier is transparent - user-facing hold(...) is displayed
            _ if crate::hold::is_internal_hold_name(name) && args.len() == 1 => {
                self.render_with_path(args[0], false, &self.child_path(path, 0))
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

fn is_full_color_highlight(latex: &str, color: HighlightColor) -> bool {
    let prefix = format!("{{\\color{{{}}}{{", color.to_latex());
    latex.starts_with(&prefix) && latex.ends_with("}}")
}
