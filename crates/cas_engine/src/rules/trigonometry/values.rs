//! Static lookup tables for known trigonometric values at special angles.
//!
//! This module provides a data-driven approach to evaluating trig functions
//! at special angles like 0, π/6, π/4, π/3, π/2, π using static tables.

use cas_ast::{Context, Expr, ExprId};
pub use cas_math::trig_value_detection_support::{
    detect_inverse_trig_input, detect_special_angle, InverseTrigInput, SpecialAngle,
};

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
    /// √n (just sqrt, e.g., √3)
    Sqrt(i64),
    /// (√a + √b) / d (e.g., (√6 + √2)/4)
    SqrtSumDiv(i64, i64, i64),
    /// (√a - √b) / d (e.g., (√6 - √2)/4)
    SqrtDiffDiv(i64, i64, i64),
    /// √(a + √b) / d (nested radical, e.g., √(2 + √2)/2 = cos(π/8))
    SqrtOfSqrtSumDiv(i64, i64, i64),
    /// √(a - √b) / d (nested radical, e.g., √(2 - √2)/2 = sin(π/8))
    SqrtOfSqrtDiffDiv(i64, i64, i64),
    /// √(a + c*√b) / d (nested with coeff, e.g., √(10 + 2√5)/4 = sin(2π/5))
    CoeffSqrtOfSqrtSumDiv(i64, i64, i64, i64), // a, c, b, d
    /// √(a - c*√b) / d (nested with coeff, e.g., √(10 - 2√5)/4 = sin(π/5))
    CoeffSqrtOfSqrtDiffDiv(i64, i64, i64, i64), // a, c, b, d
    /// n - √r (e.g., 2 - √3 = tan(π/12))
    IntMinusSqrt(i64, i64),
    /// n + √r (e.g., 2 + √3 = tan(5π/12))
    IntPlusSqrt(i64, i64),
    /// √r - n (e.g., √2 - 1 = tan(π/8))
    SqrtMinusInt(i64, i64),
    /// √r + n (e.g., √2 + 1 = tan(3π/8))
    SqrtPlusInt(i64, i64),
    /// (n + √r) / d (e.g., (1 + √5)/4 = cos(π/5))
    IntPlusSqrtDiv(i64, i64, i64),
    /// (n - √r) / d
    IntMinusSqrtDiv(i64, i64, i64),
    /// (√r - n) / d (e.g., (√5 - 1)/4 = cos(2π/5))
    SqrtMinusIntDiv(i64, i64, i64),
    /// (√r + n) / d
    SqrtPlusIntDiv(i64, i64, i64),
    /// √(a - c*√b) (without division, e.g., √(5 - 2√5) = tan(π/5))
    CoeffSqrtOfSqrtDiff(i64, i64, i64), // a, c, b
}

impl TrigValue {
    /// Build an ExprId from this TrigValue
    pub fn to_expr(&self, ctx: &mut Context) -> ExprId {
        // Helper to create sqrt(n) = n^(1/2)
        let make_sqrt = |ctx: &mut Context, n: i64| -> ExprId {
            let num = ctx.num(n);
            let one = ctx.num(1);
            let two = ctx.num(2);
            let half = ctx.add(Expr::Div(one, two));
            ctx.add(Expr::Pow(num, half))
        };

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
                let sqrt_rad = make_sqrt(ctx, *radicand);
                let d = ctx.num(*denom);
                ctx.add(Expr::Div(sqrt_rad, d))
            }
            TrigValue::NegSqrtDiv(radicand, denom) => {
                // -sqrt(radicand) / denom
                let sqrt_rad = make_sqrt(ctx, *radicand);
                let neg_sqrt = ctx.add(Expr::Neg(sqrt_rad));
                let d = ctx.num(*denom);
                ctx.add(Expr::Div(neg_sqrt, d))
            }
            TrigValue::InvSqrt(radicand) => {
                // 1 / sqrt(radicand)
                let one = ctx.num(1);
                let sqrt_rad = make_sqrt(ctx, *radicand);
                ctx.add(Expr::Div(one, sqrt_rad))
            }
            TrigValue::PiDiv(denom) => {
                let pi = ctx.add(Expr::Constant(cas_ast::Constant::Pi));
                let d = ctx.num(*denom);
                ctx.add(Expr::Div(pi, d))
            }
            TrigValue::Sqrt(radicand) => {
                // sqrt(radicand)
                make_sqrt(ctx, *radicand)
            }
            TrigValue::SqrtSumDiv(a, b, d) => {
                // (sqrt(a) + sqrt(b)) / d
                let sqrt_a = make_sqrt(ctx, *a);
                let sqrt_b = make_sqrt(ctx, *b);
                let sum = ctx.add(Expr::Add(sqrt_a, sqrt_b));
                let denom = ctx.num(*d);
                ctx.add(Expr::Div(sum, denom))
            }
            TrigValue::SqrtDiffDiv(a, b, d) => {
                // (sqrt(a) - sqrt(b)) / d
                let sqrt_a = make_sqrt(ctx, *a);
                let sqrt_b = make_sqrt(ctx, *b);
                let diff = ctx.add(Expr::Sub(sqrt_a, sqrt_b));
                let denom = ctx.num(*d);
                ctx.add(Expr::Div(diff, denom))
            }
            TrigValue::SqrtOfSqrtSumDiv(a, b, d) => {
                // sqrt(a + sqrt(b)) / d
                let sqrt_b = make_sqrt(ctx, *b);
                let a_num = ctx.num(*a);
                let inner = ctx.add(Expr::Add(a_num, sqrt_b));
                let outer_sqrt = {
                    let one = ctx.num(1);
                    let two = ctx.num(2);
                    let half = ctx.add(Expr::Div(one, two));
                    ctx.add(Expr::Pow(inner, half))
                };
                let denom = ctx.num(*d);
                ctx.add(Expr::Div(outer_sqrt, denom))
            }
            TrigValue::SqrtOfSqrtDiffDiv(a, b, d) => {
                // sqrt(a - sqrt(b)) / d
                let sqrt_b = make_sqrt(ctx, *b);
                let a_num = ctx.num(*a);
                let inner = ctx.add(Expr::Sub(a_num, sqrt_b));
                let outer_sqrt = {
                    let one = ctx.num(1);
                    let two = ctx.num(2);
                    let half = ctx.add(Expr::Div(one, two));
                    ctx.add(Expr::Pow(inner, half))
                };
                let denom = ctx.num(*d);
                ctx.add(Expr::Div(outer_sqrt, denom))
            }
            TrigValue::CoeffSqrtOfSqrtSumDiv(a, c, b, d) => {
                // sqrt(a + c*sqrt(b)) / d
                let sqrt_b = make_sqrt(ctx, *b);
                let a_num = ctx.num(*a);
                let c_num = ctx.num(*c);
                let c_sqrt_b = ctx.add(Expr::Mul(c_num, sqrt_b));
                let inner = ctx.add(Expr::Add(a_num, c_sqrt_b));
                let outer_sqrt = {
                    let one = ctx.num(1);
                    let two = ctx.num(2);
                    let half = ctx.add(Expr::Div(one, two));
                    ctx.add(Expr::Pow(inner, half))
                };
                let denom = ctx.num(*d);
                ctx.add(Expr::Div(outer_sqrt, denom))
            }
            TrigValue::CoeffSqrtOfSqrtDiffDiv(a, c, b, d) => {
                // sqrt(a - c*sqrt(b)) / d
                let sqrt_b = make_sqrt(ctx, *b);
                let a_num = ctx.num(*a);
                let c_num = ctx.num(*c);
                let c_sqrt_b = ctx.add(Expr::Mul(c_num, sqrt_b));
                let inner = ctx.add(Expr::Sub(a_num, c_sqrt_b));
                let outer_sqrt = {
                    let one = ctx.num(1);
                    let two = ctx.num(2);
                    let half = ctx.add(Expr::Div(one, two));
                    ctx.add(Expr::Pow(inner, half))
                };
                let denom = ctx.num(*d);
                ctx.add(Expr::Div(outer_sqrt, denom))
            }
            TrigValue::CoeffSqrtOfSqrtDiff(a, c, b) => {
                // sqrt(a - c*sqrt(b)) (no division)
                let sqrt_b = make_sqrt(ctx, *b);
                let a_num = ctx.num(*a);
                let c_num = ctx.num(*c);
                let c_sqrt_b = ctx.add(Expr::Mul(c_num, sqrt_b));
                let inner = ctx.add(Expr::Sub(a_num, c_sqrt_b));
                let one = ctx.num(1);
                let two = ctx.num(2);
                let half = ctx.add(Expr::Div(one, two));
                ctx.add(Expr::Pow(inner, half))
            }
            TrigValue::IntMinusSqrt(n, r) => {
                // n - sqrt(r)
                let num = ctx.num(*n);
                let sqrt_r = make_sqrt(ctx, *r);
                ctx.add(Expr::Sub(num, sqrt_r))
            }
            TrigValue::IntPlusSqrt(n, r) => {
                // n + sqrt(r)
                let num = ctx.num(*n);
                let sqrt_r = make_sqrt(ctx, *r);
                ctx.add(Expr::Add(num, sqrt_r))
            }
            TrigValue::SqrtMinusInt(r, n) => {
                // sqrt(r) - n
                let sqrt_r = make_sqrt(ctx, *r);
                let num = ctx.num(*n);
                ctx.add(Expr::Sub(sqrt_r, num))
            }
            TrigValue::SqrtPlusInt(r, n) => {
                // sqrt(r) + n
                let sqrt_r = make_sqrt(ctx, *r);
                let num = ctx.num(*n);
                ctx.add(Expr::Add(sqrt_r, num))
            }
            TrigValue::IntPlusSqrtDiv(n, r, d) => {
                // (n + sqrt(r)) / d
                let num = ctx.num(*n);
                let sqrt_r = make_sqrt(ctx, *r);
                let sum = ctx.add(Expr::Add(num, sqrt_r));
                let denom = ctx.num(*d);
                ctx.add(Expr::Div(sum, denom))
            }
            TrigValue::IntMinusSqrtDiv(n, r, d) => {
                // (n - sqrt(r)) / d
                let num = ctx.num(*n);
                let sqrt_r = make_sqrt(ctx, *r);
                let diff = ctx.add(Expr::Sub(num, sqrt_r));
                let denom = ctx.num(*d);
                ctx.add(Expr::Div(diff, denom))
            }
            TrigValue::SqrtMinusIntDiv(r, n, d) => {
                // (sqrt(r) - n) / d
                let sqrt_r = make_sqrt(ctx, *r);
                let num = ctx.num(*n);
                let diff = ctx.add(Expr::Sub(sqrt_r, num));
                let denom = ctx.num(*d);
                ctx.add(Expr::Div(diff, denom))
            }
            TrigValue::SqrtPlusIntDiv(r, n, d) => {
                // (sqrt(r) + n) / d
                let sqrt_r = make_sqrt(ctx, *r);
                let num = ctx.num(*n);
                let sum = ctx.add(Expr::Add(sqrt_r, num));
                let denom = ctx.num(*d);
                ctx.add(Expr::Div(sum, denom))
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
            TrigValue::Sqrt(r) => format!("√{}", r),
            TrigValue::SqrtSumDiv(a, b, d) => format!("(√{}+√{})/{}", a, b, d),
            TrigValue::SqrtDiffDiv(a, b, d) => format!("(√{}-√{})/{}", a, b, d),
            TrigValue::SqrtOfSqrtSumDiv(a, b, d) => format!("√({}+√{})/{}", a, b, d),
            TrigValue::SqrtOfSqrtDiffDiv(a, b, d) => format!("√({}-√{})/{}", a, b, d),
            TrigValue::CoeffSqrtOfSqrtSumDiv(a, c, b, d) => format!("√({}+{}√{})/{}", a, c, b, d),
            TrigValue::CoeffSqrtOfSqrtDiffDiv(a, c, b, d) => format!("√({}-{}√{})/{}", a, c, b, d),
            TrigValue::IntMinusSqrt(n, r) => format!("{}-√{}", n, r),
            TrigValue::IntPlusSqrt(n, r) => format!("{}+√{}", n, r),
            TrigValue::SqrtMinusInt(r, n) => format!("√{}-{}", r, n),
            TrigValue::SqrtPlusInt(r, n) => format!("√{}+{}", r, n),
            TrigValue::IntPlusSqrtDiv(n, r, d) => format!("({}+√{})/{}", n, r, d),
            TrigValue::IntMinusSqrtDiv(n, r, d) => format!("({}-√{})/{}", n, r, d),
            TrigValue::SqrtMinusIntDiv(r, n, d) => format!("(√{}-{})/{}", r, n, d),
            TrigValue::SqrtPlusIntDiv(r, n, d) => format!("(√{}+{})/{}", r, n, d),
            TrigValue::CoeffSqrtOfSqrtDiff(a, c, b) => format!("√({}-{}√{})", a, c, b),
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
    ("tan", SpecialAngle::PiOver3, TrigValue::Sqrt(3)),
    // --- Angle: π/4 ---
    ("sin", SpecialAngle::PiOver4, TrigValue::SqrtDiv(2, 2)),
    ("cos", SpecialAngle::PiOver4, TrigValue::SqrtDiv(2, 2)),
    ("tan", SpecialAngle::PiOver4, TrigValue::One),
    // --- Angle: π/6 ---
    ("sin", SpecialAngle::PiOver6, TrigValue::Fraction(1, 2)),
    ("cos", SpecialAngle::PiOver6, TrigValue::SqrtDiv(3, 2)),
    ("tan", SpecialAngle::PiOver6, TrigValue::InvSqrt(3)),
    // --- Angle: π/12 (15°) ---
    // sin(π/12) = (√6 - √2) / 4
    // cos(π/12) = (√6 + √2) / 4
    // tan(π/12) = 2 - √3
    (
        "sin",
        SpecialAngle::PiOver12,
        TrigValue::SqrtDiffDiv(6, 2, 4),
    ),
    (
        "cos",
        SpecialAngle::PiOver12,
        TrigValue::SqrtSumDiv(6, 2, 4),
    ),
    ("tan", SpecialAngle::PiOver12, TrigValue::IntMinusSqrt(2, 3)),
    // --- Angle: 5π/12 (75°) ---
    // sin(5π/12) = (√6 + √2) / 4
    // cos(5π/12) = (√6 - √2) / 4
    // tan(5π/12) = 2 + √3
    (
        "sin",
        SpecialAngle::FivePiOver12,
        TrigValue::SqrtSumDiv(6, 2, 4),
    ),
    (
        "cos",
        SpecialAngle::FivePiOver12,
        TrigValue::SqrtDiffDiv(6, 2, 4),
    ),
    (
        "tan",
        SpecialAngle::FivePiOver12,
        TrigValue::IntPlusSqrt(2, 3),
    ),
    // --- Angle: π/8 (22.5°) ---
    // sin(π/8) = √(2 - √2) / 2
    // cos(π/8) = √(2 + √2) / 2
    // tan(π/8) = √2 - 1
    (
        "sin",
        SpecialAngle::PiOver8,
        TrigValue::SqrtOfSqrtDiffDiv(2, 2, 2),
    ),
    (
        "cos",
        SpecialAngle::PiOver8,
        TrigValue::SqrtOfSqrtSumDiv(2, 2, 2),
    ),
    ("tan", SpecialAngle::PiOver8, TrigValue::SqrtMinusInt(2, 1)),
    // --- Angle: 3π/8 (67.5°) ---
    // sin(3π/8) = √(2 + √2) / 2
    // cos(3π/8) = √(2 - √2) / 2
    // tan(3π/8) = √2 + 1
    (
        "sin",
        SpecialAngle::ThreePiOver8,
        TrigValue::SqrtOfSqrtSumDiv(2, 2, 2),
    ),
    (
        "cos",
        SpecialAngle::ThreePiOver8,
        TrigValue::SqrtOfSqrtDiffDiv(2, 2, 2),
    ),
    (
        "tan",
        SpecialAngle::ThreePiOver8,
        TrigValue::SqrtPlusInt(2, 1),
    ),
    // --- Angle: π/5 (36°) - Golden Angle ---
    // sin(π/5) = √(10 - 2√5) / 4
    // cos(π/5) = (1 + √5) / 4
    // tan(π/5) = √(5 - 2√5)
    (
        "sin",
        SpecialAngle::PiOver5,
        TrigValue::CoeffSqrtOfSqrtDiffDiv(10, 2, 5, 4),
    ),
    (
        "cos",
        SpecialAngle::PiOver5,
        TrigValue::IntPlusSqrtDiv(1, 5, 4),
    ),
    (
        "tan",
        SpecialAngle::PiOver5,
        TrigValue::CoeffSqrtOfSqrtDiff(5, 2, 5),
    ),
    // --- Angle: 2π/5 (72°) ---
    // sin(2π/5) = √(10 + 2√5) / 4
    // cos(2π/5) = (√5 - 1) / 4
    (
        "sin",
        SpecialAngle::TwoPiOver5,
        TrigValue::CoeffSqrtOfSqrtSumDiv(10, 2, 5, 4),
    ),
    (
        "cos",
        SpecialAngle::TwoPiOver5,
        TrigValue::SqrtMinusIntDiv(5, 1, 4),
    ),
    // --- Angle: π/10 (18°) ---
    // sin(π/10) = (√5 - 1) / 4
    // cos(π/10) = √(10 + 2√5) / 4
    (
        "sin",
        SpecialAngle::PiOver10,
        TrigValue::SqrtMinusIntDiv(5, 1, 4),
    ),
    (
        "cos",
        SpecialAngle::PiOver10,
        TrigValue::CoeffSqrtOfSqrtSumDiv(10, 2, 5, 4),
    ),
    // =========================================================================
    // COTANGENT (cot = cos/sin = 1/tan)
    // =========================================================================
    ("cot", SpecialAngle::Zero, TrigValue::Undefined),
    ("cot", SpecialAngle::Pi, TrigValue::Undefined),
    ("cot", SpecialAngle::PiOver2, TrigValue::Zero),
    ("cot", SpecialAngle::PiOver6, TrigValue::Sqrt(3)),
    ("cot", SpecialAngle::PiOver4, TrigValue::One),
    ("cot", SpecialAngle::PiOver3, TrigValue::InvSqrt(3)),
    ("cot", SpecialAngle::PiOver12, TrigValue::IntPlusSqrt(2, 3)), // 2 + √3
    (
        "cot",
        SpecialAngle::FivePiOver12,
        TrigValue::IntMinusSqrt(2, 3),
    ), // 2 - √3
    ("cot", SpecialAngle::PiOver8, TrigValue::SqrtPlusInt(2, 1)),  // √2 + 1
    (
        "cot",
        SpecialAngle::ThreePiOver8,
        TrigValue::SqrtMinusInt(2, 1),
    ), // √2 - 1
    // =========================================================================
    // SECANT (sec = 1/cos)
    // =========================================================================
    ("sec", SpecialAngle::Zero, TrigValue::One),
    ("sec", SpecialAngle::Pi, TrigValue::NegOne),
    ("sec", SpecialAngle::PiOver2, TrigValue::Undefined),
    ("sec", SpecialAngle::PiOver6, TrigValue::SqrtDiv(12, 3)), // 2/√3 = 2√3/3 = √12/3
    ("sec", SpecialAngle::PiOver4, TrigValue::Sqrt(2)),
    ("sec", SpecialAngle::PiOver3, TrigValue::Fraction(2, 1)), // 2
    (
        "sec",
        SpecialAngle::PiOver12,
        TrigValue::SqrtDiffDiv(6, 2, 1),
    ), // √6 - √2
    (
        "sec",
        SpecialAngle::FivePiOver12,
        TrigValue::SqrtSumDiv(6, 2, 1),
    ), // √6 + √2
    // =========================================================================
    // COSECANT (csc = 1/sin)
    // =========================================================================
    ("csc", SpecialAngle::Zero, TrigValue::Undefined),
    ("csc", SpecialAngle::Pi, TrigValue::Undefined),
    ("csc", SpecialAngle::PiOver2, TrigValue::One),
    ("csc", SpecialAngle::PiOver6, TrigValue::Fraction(2, 1)), // 2
    ("csc", SpecialAngle::PiOver4, TrigValue::Sqrt(2)),
    ("csc", SpecialAngle::PiOver3, TrigValue::SqrtDiv(12, 3)), // 2/√3 = 2√3/3 = √12/3
    (
        "csc",
        SpecialAngle::PiOver12,
        TrigValue::SqrtSumDiv(6, 2, 1),
    ), // √6 + √2
    (
        "csc",
        SpecialAngle::FivePiOver12,
        TrigValue::SqrtDiffDiv(6, 2, 1),
    ), // √6 - √2
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

/// Look up a trig value from the static table
pub fn lookup_trig_value(func_name: &str, angle: SpecialAngle) -> Option<&'static TrigValue> {
    TRIG_VALUES
        .iter()
        .find(|(f, a, _)| *f == func_name && *a == angle)
        .map(|(_, _, v)| v)
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
    // --- Input: √2/2 ---
    ("arcsin", InverseTrigInput::Sqrt2Over2, TrigValue::PiDiv(4)),
    ("arccos", InverseTrigInput::Sqrt2Over2, TrigValue::PiDiv(4)),
    // --- Input: √3/2 ---
    ("arcsin", InverseTrigInput::Sqrt3Over2, TrigValue::PiDiv(3)),
    ("arccos", InverseTrigInput::Sqrt3Over2, TrigValue::PiDiv(6)),
    // --- Input: 1/√3 = √3/3 ---
    (
        "arctan",
        InverseTrigInput::OneOverSqrt3,
        TrigValue::PiDiv(6),
    ),
    // --- Input: √3 ---
    ("arctan", InverseTrigInput::Sqrt3, TrigValue::PiDiv(3)),
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
