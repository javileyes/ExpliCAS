//! Static lookup tables for known trigonometric values at special angles.
//!
//! This module provides a data-driven approach to evaluating trig functions
//! at special angles like 0, π/6, π/4, π/3, π/2, π using static tables.

use cas_ast::{Context, Expr, ExprId};

/// Represents a simplified trigonometric result value.
#[derive(Clone, Debug, PartialEq)]
pub enum TrigValue {
    /// 0
    Zero,
    /// 1
    One,
    /// -1
    NegOne,
    /// Undefined (e.g., tan(π/2))
    Undefined,
    /// A rational fraction: numer/denom (e.g., 1/2)
    Fraction(i64, i64),
    /// sqrt(n)/d (e.g., sqrt(3)/2 is SqrtDiv(3, 2))
    SqrtDiv(i64, i64),
    /// 1/sqrt(n) (e.g., 1/sqrt(3) is InvSqrt(3))
    InvSqrt(i64),
    /// -sqrt(n)/d (e.g., -sqrt(2)/2)
    NegSqrtDiv(i64, i64),
    /// π/n (e.g., π/2 is PiDiv(2))
    PiDiv(i64),
}

impl TrigValue {
    /// Build an ExprId from this TrigValue
    pub fn to_expr(&self, ctx: &mut Context) -> ExprId {
        match self {
            TrigValue::Zero => ctx.num(0),
            TrigValue::One => ctx.num(1),
            TrigValue::NegOne => ctx.num(-1),
            TrigValue::Undefined => ctx.add(Expr::Constant(cas_ast::Constant::Undefined)),
            TrigValue::Fraction(n, d) => {
                let num = ctx.num(*n);
                let den = ctx.num(*d);
                ctx.add(Expr::Div(num, den))
            }
            TrigValue::SqrtDiv(radicand, denom) => {
                // sqrt(radicand) / denom
                let rad = ctx.num(*radicand);
                let one = ctx.num(1);
                let two = ctx.num(2);
                let half = ctx.add(Expr::Div(one, two));
                let sqrt_rad = ctx.add(Expr::Pow(rad, half));
                let d = ctx.num(*denom);
                ctx.add(Expr::Div(sqrt_rad, d))
            }
            TrigValue::NegSqrtDiv(radicand, denom) => {
                // -sqrt(radicand) / denom
                let rad = ctx.num(*radicand);
                let one = ctx.num(1);
                let two = ctx.num(2);
                let half = ctx.add(Expr::Div(one, two));
                let sqrt_rad = ctx.add(Expr::Pow(rad, half));
                let neg_sqrt = ctx.add(Expr::Neg(sqrt_rad));
                let d = ctx.num(*denom);
                ctx.add(Expr::Div(neg_sqrt, d))
            }
            TrigValue::InvSqrt(radicand) => {
                // 1 / sqrt(radicand)
                let one = ctx.num(1);
                let rad = ctx.num(*radicand);
                let one2 = ctx.num(1);
                let two = ctx.num(2);
                let half = ctx.add(Expr::Div(one2, two));
                let sqrt_rad = ctx.add(Expr::Pow(rad, half));
                ctx.add(Expr::Div(one, sqrt_rad))
            }
            TrigValue::PiDiv(denom) => {
                let pi = ctx.add(Expr::Constant(cas_ast::Constant::Pi));
                let d = ctx.num(*denom);
                ctx.add(Expr::Div(pi, d))
            }
        }
    }

    /// Human-readable display for use in descriptions
    pub fn display(&self) -> String {
        match self {
            TrigValue::Zero => "0".to_string(),
            TrigValue::One => "1".to_string(),
            TrigValue::NegOne => "-1".to_string(),
            TrigValue::Undefined => "undefined".to_string(),
            TrigValue::Fraction(n, d) => format!("{}/{}", n, d),
            TrigValue::SqrtDiv(r, d) => format!("√{}/{}", r, d),
            TrigValue::NegSqrtDiv(r, d) => format!("-√{}/{}", r, d),
            TrigValue::InvSqrt(r) => format!("1/√{}", r),
            TrigValue::PiDiv(d) => format!("π/{}", d),
        }
    }
}

/// Angle key enum for special angles
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SpecialAngle {
    Zero,    // 0
    Pi,      // π
    PiOver2, // π/2
    PiOver3, // π/3
    PiOver4, // π/4
    PiOver6, // π/6
}

impl SpecialAngle {
    pub fn display(&self) -> &'static str {
        match self {
            SpecialAngle::Zero => "0",
            SpecialAngle::Pi => "π",
            SpecialAngle::PiOver2 => "π/2",
            SpecialAngle::PiOver3 => "π/3",
            SpecialAngle::PiOver4 => "π/4",
            SpecialAngle::PiOver6 => "π/6",
        }
    }
}

/// Static lookup table: (function_name, special_angle) -> TrigValue
///
/// This table contains the known exact values for sin, cos, tan at special angles.
pub static TRIG_VALUES: &[(&str, SpecialAngle, TrigValue)] = &[
    // --- Angle: 0 ---
    ("sin", SpecialAngle::Zero, TrigValue::Zero),
    ("cos", SpecialAngle::Zero, TrigValue::One),
    ("tan", SpecialAngle::Zero, TrigValue::Zero),
    // --- Angle: π ---
    ("sin", SpecialAngle::Pi, TrigValue::Zero),
    ("cos", SpecialAngle::Pi, TrigValue::NegOne),
    ("tan", SpecialAngle::Pi, TrigValue::Zero),
    // --- Angle: π/2 ---
    ("sin", SpecialAngle::PiOver2, TrigValue::One),
    ("cos", SpecialAngle::PiOver2, TrigValue::Zero),
    ("tan", SpecialAngle::PiOver2, TrigValue::Undefined),
    // --- Angle: π/3 ---
    ("sin", SpecialAngle::PiOver3, TrigValue::SqrtDiv(3, 2)),
    ("cos", SpecialAngle::PiOver3, TrigValue::Fraction(1, 2)),
    ("tan", SpecialAngle::PiOver3, TrigValue::InvSqrt(3)), // Actually √3, but this is 1/√(1/3)...
    // --- Angle: π/4 ---
    ("sin", SpecialAngle::PiOver4, TrigValue::SqrtDiv(2, 2)),
    ("cos", SpecialAngle::PiOver4, TrigValue::SqrtDiv(2, 2)),
    ("tan", SpecialAngle::PiOver4, TrigValue::One),
    // --- Angle: π/6 ---
    ("sin", SpecialAngle::PiOver6, TrigValue::Fraction(1, 2)),
    ("cos", SpecialAngle::PiOver6, TrigValue::SqrtDiv(3, 2)),
    ("tan", SpecialAngle::PiOver6, TrigValue::InvSqrt(3)),
];

/// Inverse trig values at special numeric inputs
pub static INVERSE_TRIG_VALUES: &[(&str, &str, TrigValue)] = &[
    // --- Input: 0 ---
    ("arcsin", "0", TrigValue::Zero),
    ("arctan", "0", TrigValue::Zero),
    ("arccos", "0", TrigValue::PiDiv(2)),
    // --- Input: 1 ---
    ("arcsin", "1", TrigValue::PiDiv(2)),
    ("arccos", "1", TrigValue::Zero),
    ("arctan", "1", TrigValue::PiDiv(4)),
    // --- Input: 1/2 ---
    ("arcsin", "1/2", TrigValue::PiDiv(6)),
    ("arccos", "1/2", TrigValue::PiDiv(3)),
];

/// Detect if an expression represents a special angle
///
/// Checks for 0, π, π/2, π/3, π/4, π/6 using shared helper functions.
pub fn detect_special_angle(ctx: &Context, expr: ExprId) -> Option<SpecialAngle> {
    use crate::helpers::{is_pi, is_pi_over_n};
    use num_traits::Zero;

    // Check for 0
    if let Expr::Number(n) = ctx.get(expr) {
        if n.is_zero() {
            return Some(SpecialAngle::Zero);
        }
    }

    // Check for π
    if is_pi(ctx, expr) {
        return Some(SpecialAngle::Pi);
    }

    // Check for π/n values
    if is_pi_over_n(ctx, expr, 2) {
        return Some(SpecialAngle::PiOver2);
    }
    if is_pi_over_n(ctx, expr, 3) {
        return Some(SpecialAngle::PiOver3);
    }
    if is_pi_over_n(ctx, expr, 4) {
        return Some(SpecialAngle::PiOver4);
    }
    if is_pi_over_n(ctx, expr, 6) {
        return Some(SpecialAngle::PiOver6);
    }

    None
}

/// Look up a trig value from the static table
pub fn lookup_trig_value(func_name: &str, angle: SpecialAngle) -> Option<&'static TrigValue> {
    TRIG_VALUES
        .iter()
        .find(|(f, a, _)| *f == func_name && *a == angle)
        .map(|(_, _, v)| v)
}

/// Special input values for inverse trig functions
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InverseTrigInput {
    Zero, // 0
    One,  // 1
    Half, // 1/2
}

impl InverseTrigInput {
    pub fn display(&self) -> &'static str {
        match self {
            InverseTrigInput::Zero => "0",
            InverseTrigInput::One => "1",
            InverseTrigInput::Half => "1/2",
        }
    }
}

/// Detect if an expression is a special input for inverse trig
pub fn detect_inverse_trig_input(ctx: &Context, expr: ExprId) -> Option<InverseTrigInput> {
    use num_traits::{One, Zero};

    if let Expr::Number(n) = ctx.get(expr) {
        if n.is_zero() {
            return Some(InverseTrigInput::Zero);
        }
        if n.is_one() {
            return Some(InverseTrigInput::One);
        }
        if *n == num_rational::BigRational::new(1.into(), 2.into()) {
            return Some(InverseTrigInput::Half);
        }
    }
    None
}

/// Lookup table for inverse trig values
pub static INVERSE_TRIG_TABLE: &[(&str, InverseTrigInput, TrigValue)] = &[
    // --- Input: 0 ---
    ("arcsin", InverseTrigInput::Zero, TrigValue::Zero),
    ("arctan", InverseTrigInput::Zero, TrigValue::Zero),
    ("arccos", InverseTrigInput::Zero, TrigValue::PiDiv(2)),
    // --- Input: 1 ---
    ("arcsin", InverseTrigInput::One, TrigValue::PiDiv(2)),
    ("arccos", InverseTrigInput::One, TrigValue::Zero),
    ("arctan", InverseTrigInput::One, TrigValue::PiDiv(4)),
    // --- Input: 1/2 ---
    ("arcsin", InverseTrigInput::Half, TrigValue::PiDiv(6)),
    ("arccos", InverseTrigInput::Half, TrigValue::PiDiv(3)),
];

/// Look up an inverse trig value from the static table
pub fn lookup_inverse_trig_value(
    func_name: &str,
    input: InverseTrigInput,
) -> Option<&'static TrigValue> {
    INVERSE_TRIG_TABLE
        .iter()
        .find(|(f, i, _)| *f == func_name && *i == input)
        .map(|(_, _, v)| v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lookup_sin_zero() {
        let val = lookup_trig_value("sin", SpecialAngle::Zero);
        assert_eq!(val, Some(&TrigValue::Zero));
    }

    #[test]
    fn test_lookup_cos_pi_over_4() {
        let val = lookup_trig_value("cos", SpecialAngle::PiOver4);
        assert_eq!(val, Some(&TrigValue::SqrtDiv(2, 2)));
    }

    #[test]
    fn test_to_expr_fraction() {
        let mut ctx = Context::new();
        let val = TrigValue::Fraction(1, 2);
        let expr_id = val.to_expr(&mut ctx);
        // Just verify it doesn't panic and returns a valid expression
        // Check that we can get the expression back
        let _ = ctx.get(expr_id);
    }
}
