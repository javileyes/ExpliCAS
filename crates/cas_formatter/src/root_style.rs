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
            Expr::Function(fn_id, args) => {
                if ctx.sym_name(*fn_id) == "sqrt" {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_sqrt_function() {
        let mut ctx = Context::new();
        let two = ctx.num(2);
        let sqrt2 = ctx.call("sqrt", vec![two]);

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
        let sqrt2 = ctx.call("sqrt", vec![two]);
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
        let sqrt2 = ctx.call("sqrt", vec![two]);

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
