//! Root Style Detection for Context-Aware Formatting
//!
//! This module implements the "Style Sniffing" pattern: detect the user's preferred
//! notation for roots (√x vs x^(1/2)) from the input expression, then apply that
//! style consistently to the output.

use crate::{Context, Expr, ExprId};

/// User's preferred style for displaying roots
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RootStyle {
    /// Display as radicals: √x, ³√x, etc.
    Radical,
    /// Display as fractional exponents: x^(1/2), x^(1/3), etc.
    Exponential,
    /// Auto-detect or use default (Radical)
    #[default]
    Auto,
}

impl RootStyle {
    /// Resolve Auto to a concrete style (defaults to Radical)
    pub fn resolve(self) -> RootStyle {
        match self {
            RootStyle::Auto => RootStyle::Radical,
            other => other,
        }
    }
}

/// Detect the user's preferred root style from an expression
///
/// Scans the expression tree and counts:
/// - `Function("sqrt", _)` → radical_score
/// - `Pow(_, 1/n)` where n > 1 → exponent_score
///
/// Returns the majority style, defaulting to Radical on tie.
pub fn detect_root_style(ctx: &Context, id: ExprId) -> RootStyle {
    let mut radical_score = 0i32;
    let mut exponent_score = 0i32;

    let mut worklist = vec![id];

    while let Some(curr) = worklist.pop() {
        match ctx.get(curr) {
            Expr::Function(name, args) => {
                if name == "sqrt" {
                    radical_score += 1;
                }
                // Recurse into function arguments
                worklist.extend(args.iter().copied());
            }

            Expr::Pow(base, exp) => {
                // Check if exponent is fractional 1/n or k/n
                if is_fractional_exponent(ctx, *exp) {
                    exponent_score += 1;
                }
                worklist.push(*base);
                worklist.push(*exp);
            }

            Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) => {
                worklist.push(*l);
                worklist.push(*r);
            }

            Expr::Neg(inner) => {
                worklist.push(*inner);
            }

            Expr::Matrix { data, .. } => {
                worklist.extend(data.iter().copied());
            }

            // Terminals: Number, Variable, Constant
            _ => {}
        }
    }

    // Tie-breaker: default to Radical (standard math notation)
    if radical_score > 0 && radical_score >= exponent_score {
        RootStyle::Radical
    } else if exponent_score > radical_score {
        RootStyle::Exponential
    } else {
        RootStyle::Auto // No roots detected, let formatter decide
    }
}

/// Check if an exponent represents a fractional power (1/n or k/n where n > 1)
fn is_fractional_exponent(ctx: &Context, exp: ExprId) -> bool {
    match ctx.get(exp) {
        // Direct fractional number: 0.5, 0.333...
        Expr::Number(n) => {
            if n.is_integer() {
                return false;
            }
            // Check if denominator > 1
            let denom = n.denom();
            *denom > 1.into()
        }

        // Division form: 1/2, 1/3, k/n
        Expr::Div(_num, den) => {
            // Verify denominator is integer > 1
            if let Expr::Number(d) = ctx.get(*den) {
                if d.is_integer() {
                    if let Some(d_int) = d.numer().to_u32_digits().1.first() {
                        return *d_int > 1;
                    }
                }
            }
            false
        }

        // Negative exponent: -1/2
        Expr::Neg(inner) => is_fractional_exponent(ctx, *inner),

        _ => false,
    }
}

// ============================================================================
// StylePreferences: Global formatting preferences
// ============================================================================

/// Global formatting preferences derived from the original expression.
///
/// This replaces per-node hints with a single set of preferences that apply
/// to all output formatting. Preferences are "sniffed" from the input to
/// match the user's notation style.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StylePreferences {
    /// How to display Pow(x, 1/n): as √x or x^(1/n)
    pub root_style: RootStyle,
    /// Whether to prefer a/b over a*b^(-1)
    pub prefer_division: bool,
    /// Whether to prefer a-b over a+(-b)
    pub prefer_subtraction: bool,
    /// Whether to order polynomial sums by degree (descending): x^2 + x + 1
    pub polynomial_order: bool,
}

impl Default for StylePreferences {
    fn default() -> Self {
        Self {
            root_style: RootStyle::Auto,
            prefer_division: true,    // Most users expect fractions
            prefer_subtraction: true, // Most users expect subtraction
            polynomial_order: true,   // Educational CAS: order by degree
        }
    }
}

impl StylePreferences {
    /// Create preferences with all defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Detect preferences from an expression (AST-only sniffing).
    ///
    /// For more accurate detection, use `from_expression_with_signals`
    /// which also considers parser-level information.
    pub fn from_expression(ctx: &Context, id: ExprId) -> Self {
        let root_style = detect_root_style(ctx, id);
        Self {
            root_style,
            ..Self::default()
        }
    }

    /// Detect preferences from expression + optional parser signals.
    ///
    /// Parser signals provide information that may be lost in the AST
    /// (e.g., whether user wrote `sqrt(2)` or `2^(1/2)`).
    pub fn from_expression_with_signals(
        ctx: &Context,
        id: ExprId,
        signals: Option<&ParseStyleSignals>,
    ) -> Self {
        let mut prefs = Self::from_expression(ctx, id);

        // If parser signals are available, they take precedence
        if let Some(sig) = signals {
            if sig.saw_sqrt_token > 0 || sig.saw_caret_fraction > 0 {
                if sig.saw_sqrt_token > sig.saw_caret_fraction {
                    prefs.root_style = RootStyle::Radical;
                } else if sig.saw_caret_fraction > sig.saw_sqrt_token {
                    prefs.root_style = RootStyle::Exponential;
                }
                // On tie, keep AST-based detection
            }
        }

        prefs
    }

    /// Create preferences with explicit root style
    pub fn with_root_style(root_style: RootStyle) -> Self {
        Self {
            root_style,
            ..Self::default()
        }
    }

    /// Resolve Auto values to concrete preferences
    pub fn resolve(&self) -> Self {
        Self {
            root_style: self.root_style.resolve(),
            polynomial_order: self.polynomial_order,
            ..*self
        }
    }
}

/// Signals from the parser about notation preferences.
///
/// These capture information that may be lost when the expression
/// is canonicalized to AST form.
#[derive(Debug, Clone, Default)]
pub struct ParseStyleSignals {
    /// Number of times `sqrt(...)` or `√` was seen
    pub saw_sqrt_token: usize,
    /// Number of times `^(1/n)` or similar fractional exponent was seen
    pub saw_caret_fraction: usize,
    /// Number of `/` division operators seen  
    pub saw_division_slash: usize,
    /// Number of `-` subtraction operators seen
    pub saw_minus: usize,
}

impl ParseStyleSignals {
    pub fn new() -> Self {
        Self::default()
    }

    /// Quick sniff from input string before parsing.
    /// This is a lightweight alternative to parser-level signal collection.
    pub fn from_input_string(input: &str) -> Self {
        let mut signals = Self::new();

        // Count sqrt occurrences
        signals.saw_sqrt_token = input.matches("sqrt").count() + input.matches('√').count();

        // Count potential fractional exponents (rough heuristic)
        // Look for patterns like ^(1/2), ^(1/3), ^(2/3)
        let mut chars = input.chars().peekable();
        while let Some(c) = chars.next() {
            if c == '^' {
                // Check if followed by ( and then a fraction-like pattern
                if chars.peek() == Some(&'(') {
                    signals.saw_caret_fraction += 1;
                }
            }
            if c == '/' {
                signals.saw_division_slash += 1;
            }
            if c == '-' {
                signals.saw_minus += 1;
            }
        }

        signals
    }
}

// ============================================================================
// StyledExpr: Display Wrapper with Style Context
// ============================================================================

use num_traits::ToPrimitive;
use std::fmt;

/// Expression display wrapper that respects RootStyle for consistent output.
///
/// This implements the "Style Sniffing" output pattern: format roots according
/// to the user's detected preference (radical √x or exponential x^(1/2)).
pub struct StyledExpr<'a> {
    pub context: &'a Context,
    pub id: ExprId,
    pub style: RootStyle,
}

impl<'a> StyledExpr<'a> {
    /// Create a new styled expression with given style
    pub fn new(context: &'a Context, id: ExprId, style: RootStyle) -> Self {
        Self {
            context,
            id,
            style: style.resolve(),
        }
    }

    /// Helper to check if number is 1/n (returns n if so)
    fn as_root_degree(&self, exp: ExprId) -> Option<i64> {
        match self.context.get(exp) {
            Expr::Number(n) => {
                if n.is_integer() {
                    return None;
                }
                // Check if numerator is 1
                if n.numer().to_i64() == Some(1) {
                    n.denom().to_i64()
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Recursive formatting helper
    fn fmt_expr(&self, f: &mut fmt::Formatter<'_>, id: ExprId) -> fmt::Result {
        match self.context.get(id) {
            Expr::Number(n) => {
                if n.is_integer() {
                    write!(f, "{}", n.numer())
                } else {
                    write!(f, "{}/{}", n.numer(), n.denom())
                }
            }

            Expr::Variable(name) => write!(f, "{}", name),

            Expr::Constant(c) => match c {
                crate::Constant::Pi => write!(f, "π"),
                crate::Constant::E => write!(f, "e"),
                crate::Constant::Infinity => write!(f, "∞"),
                crate::Constant::Undefined => write!(f, "undefined"),
                crate::Constant::I => write!(f, "i"),
            },
            Expr::Pow(base, exp) => {
                // Check if this is a root (exponent is 1/n)
                if let Some(degree) = self.as_root_degree(*exp) {
                    match self.style {
                        RootStyle::Radical | RootStyle::Auto => {
                            // Display as radical
                            write!(f, "{}(", crate::display::unicode_root_prefix(degree as u64))?;
                            self.fmt_expr(f, *base)?;
                            write!(f, ")")
                        }
                        RootStyle::Exponential => {
                            // Display as power
                            write!(f, "(")?;
                            self.fmt_expr(f, *base)?;
                            write!(f, ")^(1/{})", degree)
                        }
                    }
                } else {
                    // Regular power
                    write!(f, "(")?;
                    self.fmt_expr(f, *base)?;
                    write!(f, ")^(")?;
                    self.fmt_expr(f, *exp)?;
                    write!(f, ")")
                }
            }

            Expr::Add(l, r) => {
                // Collect all additive terms with their signs
                let terms = self.collect_additive_terms(id);
                for (i, (term, is_positive)) in terms.iter().enumerate() {
                    if i == 0 {
                        if *is_positive {
                            self.fmt_term(f, *term)?;
                        } else {
                            write!(f, "-")?;
                            self.fmt_term_abs(f, *term)?;
                        }
                    } else if *is_positive {
                        write!(f, " + ")?;
                        self.fmt_term(f, *term)?;
                    } else {
                        write!(f, " - ")?;
                        self.fmt_term_abs(f, *term)?;
                    }
                }
                let _ = (l, r); // suppress unused
                Ok(())
            }

            Expr::Sub(l, r) => {
                self.fmt_expr(f, *l)?;
                write!(f, " - ")?;
                // Parenthesize r if it's Add/Sub
                if self.needs_parens_in_sub(*r) {
                    write!(f, "(")?;
                    self.fmt_expr(f, *r)?;
                    write!(f, ")")
                } else {
                    self.fmt_expr(f, *r)
                }
            }

            Expr::Mul(l, r) => {
                // Parenthesize Add/Sub factors
                self.fmt_mul_factor(f, *l)?;
                write!(f, "·")?;
                self.fmt_mul_factor(f, *r)
            }

            Expr::Div(l, r) => {
                // Only parenthesize if not atomic
                if self.is_atomic(*l) {
                    self.fmt_expr(f, *l)?;
                } else {
                    write!(f, "(")?;
                    self.fmt_expr(f, *l)?;
                    write!(f, ")")?;
                }
                write!(f, " / ")?;
                if self.is_atomic(*r) {
                    self.fmt_expr(f, *r)
                } else {
                    write!(f, "(")?;
                    self.fmt_expr(f, *r)?;
                    write!(f, ")")
                }
            }

            Expr::Neg(inner) => {
                // Don't wrap atomics in parens
                if self.is_atomic(*inner) {
                    write!(f, "-")?;
                    self.fmt_expr(f, *inner)
                } else {
                    write!(f, "-(")?;
                    self.fmt_expr(f, *inner)?;
                    write!(f, ")")
                }
            }

            Expr::Function(name, args) => {
                write!(f, "{}(", name)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    self.fmt_expr(f, *arg)?;
                }
                write!(f, ")")
            }

            Expr::Matrix { rows, cols, data } => {
                write!(f, "[{}x{} matrix]", rows, cols)?;
                let _ = data; // Suppress unused warning
                Ok(())
            }

            Expr::SessionRef(id) => write!(f, "#{}", id),
        }
    }

    /// Check if expression is atomic (doesn't need parens when negated)
    fn is_atomic(&self, id: ExprId) -> bool {
        matches!(
            self.context.get(id),
            Expr::Number(_)
                | Expr::Variable(_)
                | Expr::Constant(_)
                | Expr::Pow(_, _)
                | Expr::Function(_, _)
        )
    }

    /// Format a factor in multiplication, adding parens if needed
    fn fmt_mul_factor(&self, f: &mut fmt::Formatter<'_>, id: ExprId) -> fmt::Result {
        match self.context.get(id) {
            Expr::Add(_, _) | Expr::Sub(_, _) => {
                write!(f, "(")?;
                self.fmt_expr(f, id)?;
                write!(f, ")")
            }
            _ => self.fmt_expr(f, id),
        }
    }

    /// Check if expression needs parens on RHS of Sub
    fn needs_parens_in_sub(&self, id: ExprId) -> bool {
        matches!(self.context.get(id), Expr::Add(_, _) | Expr::Sub(_, _))
    }

    /// Format a single term (may need parens depending on context)
    fn fmt_term(&self, f: &mut fmt::Formatter<'_>, id: ExprId) -> fmt::Result {
        match self.context.get(id) {
            Expr::Add(_, _) | Expr::Sub(_, _) => {
                write!(f, "(")?;
                self.fmt_expr(f, id)?;
                write!(f, ")")
            }
            _ => self.fmt_expr(f, id),
        }
    }

    /// Format term's absolute value (strip leading negative sign/coeff)
    fn fmt_term_abs(&self, f: &mut fmt::Formatter<'_>, id: ExprId) -> fmt::Result {
        match self.context.get(id) {
            // Neg(x) -> print x
            Expr::Neg(inner) => self.fmt_term(f, *inner),

            // Number(n) -> print |n|
            Expr::Number(n) => {
                let abs_n = if n < &num_rational::BigRational::from_integer(0.into()) {
                    -n.clone()
                } else {
                    n.clone()
                };
                if abs_n.is_integer() {
                    write!(f, "{}", abs_n.numer())
                } else {
                    write!(f, "{}/{}", abs_n.numer(), abs_n.denom())
                }
            }

            // Mul(neg_coeff, rest) -> print |coeff| * rest
            Expr::Mul(l, r) => {
                if let Expr::Number(n) = self.context.get(*l) {
                    if n < &num_rational::BigRational::from_integer(0.into()) {
                        // Print absolute value of coefficient
                        let abs_n = -n.clone();
                        if abs_n == num_rational::BigRational::from_integer(1.into()) {
                            // Skip 1 *
                            self.fmt_mul_factor(f, *r)
                        } else {
                            if abs_n.is_integer() {
                                write!(f, "{}", abs_n.numer())?;
                            } else {
                                write!(f, "{}/{}", abs_n.numer(), abs_n.denom())?;
                            }
                            write!(f, "·")?;
                            self.fmt_mul_factor(f, *r)
                        }
                    } else {
                        // Not negative, print normally
                        self.fmt_term(f, id)
                    }
                } else {
                    self.fmt_term(f, id)
                }
            }

            // Everything else: print as-is
            _ => self.fmt_term(f, id),
        }
    }

    /// Collect all additive terms with their signs
    fn collect_additive_terms(&self, id: ExprId) -> Vec<(ExprId, bool)> {
        let mut terms = Vec::new();
        self.collect_terms_recursive(id, true, &mut terms);
        terms
    }

    fn collect_terms_recursive(&self, id: ExprId, positive: bool, terms: &mut Vec<(ExprId, bool)>) {
        match self.context.get(id) {
            Expr::Add(l, r) => {
                self.collect_terms_recursive(*l, positive, terms);
                self.collect_terms_recursive(*r, positive, terms);
            }
            Expr::Sub(l, r) => {
                self.collect_terms_recursive(*l, positive, terms);
                self.collect_terms_recursive(*r, !positive, terms);
            }
            Expr::Neg(inner) => {
                self.collect_terms_recursive(*inner, !positive, terms);
            }
            // Detect Mul with negative leading coefficient: -1/4 * √2 * √3
            Expr::Mul(l, _) => {
                if let Some(is_neg) = self.is_negative_coefficient(*l) {
                    if is_neg {
                        // Flip sign and store the "positive" version
                        terms.push((id, !positive));
                    } else {
                        terms.push((id, positive));
                    }
                } else {
                    terms.push((id, positive));
                }
            }
            // Detect standalone negative number
            Expr::Number(n) => {
                if n < &num_rational::BigRational::from_integer(0.into()) {
                    terms.push((id, !positive));
                } else {
                    terms.push((id, positive));
                }
            }
            _ => {
                terms.push((id, positive));
            }
        }
    }

    /// Check if an expression is a negative coefficient (for leading term in Mul)
    fn is_negative_coefficient(&self, id: ExprId) -> Option<bool> {
        match self.context.get(id) {
            Expr::Number(n) => Some(n < &num_rational::BigRational::from_integer(0.into())),
            Expr::Neg(_) => Some(true),
            _ => None,
        }
    }
}

impl<'a> fmt::Display for StyledExpr<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_expr(f, self.id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_sqrt_function() {
        let mut ctx = Context::new();
        let two = ctx.num(2);
        let sqrt2 = ctx.add(Expr::Function("sqrt".to_string(), vec![two]));

        assert_eq!(detect_root_style(&ctx, sqrt2), RootStyle::Radical);
    }

    #[test]
    fn test_detect_pow_half() {
        let mut ctx = Context::new();
        let two = ctx.num(2);
        let half = ctx.rational(1, 2);
        let pow_half = ctx.add(Expr::Pow(two, half));

        assert_eq!(detect_root_style(&ctx, pow_half), RootStyle::Exponential);
    }

    #[test]
    fn test_detect_mixed_radical_wins() {
        let mut ctx = Context::new();
        // sqrt(2) + 3^(1/2)
        let two = ctx.num(2);
        let three = ctx.num(3);
        let sqrt2 = ctx.add(Expr::Function("sqrt".to_string(), vec![two]));
        let half = ctx.rational(1, 2);
        let pow3 = ctx.add(Expr::Pow(three, half));
        let sum = ctx.add(Expr::Add(sqrt2, pow3));

        // Tie goes to Radical
        assert_eq!(detect_root_style(&ctx, sum), RootStyle::Radical);
    }

    #[test]
    fn test_detect_no_roots() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let sum = ctx.add(Expr::Add(x, two));

        assert_eq!(detect_root_style(&ctx, sum), RootStyle::Auto);
    }

    #[test]
    fn test_style_preferences_default() {
        let prefs = StylePreferences::default();
        assert_eq!(prefs.root_style, RootStyle::Auto);
        assert!(prefs.prefer_division);
        assert!(prefs.prefer_subtraction);
    }

    #[test]
    fn test_style_preferences_from_expression() {
        let mut ctx = Context::new();
        let two = ctx.num(2);
        let sqrt2 = ctx.add(Expr::Function("sqrt".to_string(), vec![two]));

        let prefs = StylePreferences::from_expression(&ctx, sqrt2);
        assert_eq!(prefs.root_style, RootStyle::Radical);
    }

    #[test]
    fn test_parse_style_signals_from_input() {
        let signals = ParseStyleSignals::from_input_string("sqrt(2) + sqrt(3)");
        assert_eq!(signals.saw_sqrt_token, 2);
        assert_eq!(signals.saw_caret_fraction, 0);

        let signals2 = ParseStyleSignals::from_input_string("2^(1/2) + 3^(1/3)");
        assert_eq!(signals2.saw_sqrt_token, 0);
        assert_eq!(signals2.saw_caret_fraction, 2);
    }

    #[test]
    fn test_style_preferences_with_signals_sqrt_wins() {
        let mut ctx = Context::new();
        let two = ctx.num(2);
        let half = ctx.rational(1, 2);
        let pow_half = ctx.add(Expr::Pow(two, half)); // AST says exponent

        // But signals say sqrt
        let signals = ParseStyleSignals {
            saw_sqrt_token: 3,
            saw_caret_fraction: 1,
            ..Default::default()
        };

        let prefs = StylePreferences::from_expression_with_signals(&ctx, pow_half, Some(&signals));
        assert_eq!(prefs.root_style, RootStyle::Radical); // Signals override AST
    }
}
