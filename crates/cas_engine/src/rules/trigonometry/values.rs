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

/// Angle key enum for special angles
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SpecialAngle {
    Zero,         // 0
    Pi,           // π
    PiOver2,      // π/2
    PiOver3,      // π/3
    PiOver4,      // π/4
    PiOver6,      // π/6
    PiOver8,      // π/8 = 22.5°
    PiOver12,     // π/12 = 15°
    ThreePiOver8, // 3π/8 = 67.5°
    FivePiOver12, // 5π/12 = 75°
    PiOver5,      // π/5 = 36° (golden angle)
    TwoPiOver5,   // 2π/5 = 72°
    PiOver10,     // π/10 = 18°
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
            SpecialAngle::PiOver8 => "π/8",
            SpecialAngle::PiOver12 => "π/12",
            SpecialAngle::ThreePiOver8 => "3π/8",
            SpecialAngle::FivePiOver12 => "5π/12",
            SpecialAngle::PiOver5 => "π/5",
            SpecialAngle::TwoPiOver5 => "2π/5",
            SpecialAngle::PiOver10 => "π/10",
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
    ("sec", SpecialAngle::PiOver6, TrigValue::SqrtDiv(3, 1)), // 2/√3 = 2√3/3
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
    ("csc", SpecialAngle::PiOver3, TrigValue::SqrtDiv(3, 1)), // 2/√3 = 2√3/3
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

/// Detect if an expression represents a special angle
///
/// Checks for 0, π, π/2, π/3, π/4, π/6, π/8, π/12, 3π/8, 5π/12 using helper functions.
pub fn detect_special_angle(ctx: &Context, expr: ExprId) -> Option<SpecialAngle> {
    use crate::helpers::{extract_rational_pi_multiple, is_pi, is_pi_over_n};
    use num_rational::BigRational;
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

    // Check for simple π/n values first (faster path)
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
    if is_pi_over_n(ctx, expr, 8) {
        return Some(SpecialAngle::PiOver8);
    }
    if is_pi_over_n(ctx, expr, 12) {
        return Some(SpecialAngle::PiOver12);
    }
    if is_pi_over_n(ctx, expr, 5) {
        return Some(SpecialAngle::PiOver5);
    }
    if is_pi_over_n(ctx, expr, 10) {
        return Some(SpecialAngle::PiOver10);
    }

    // For fractional multiples like 3π/8, 5π/12, 2π/5, use extract_rational_pi_multiple
    if let Some(k) = extract_rational_pi_multiple(ctx, expr) {
        // Check for 3/8
        if k == BigRational::new(3.into(), 8.into()) {
            return Some(SpecialAngle::ThreePiOver8);
        }
        // Check for 5/12
        if k == BigRational::new(5.into(), 12.into()) {
            return Some(SpecialAngle::FivePiOver12);
        }
        // Check for 2/5
        if k == BigRational::new(2.into(), 5.into()) {
            return Some(SpecialAngle::TwoPiOver5);
        }
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
    Zero,         // 0
    One,          // 1
    Half,         // 1/2
    Sqrt2Over2,   // √2/2
    Sqrt3Over2,   // √3/2
    OneOverSqrt3, // 1/√3 = √3/3
    Sqrt3,        // √3
}

impl InverseTrigInput {
    pub fn display(&self) -> &'static str {
        match self {
            InverseTrigInput::Zero => "0",
            InverseTrigInput::One => "1",
            InverseTrigInput::Half => "1/2",
            InverseTrigInput::Sqrt2Over2 => "√2/2",
            InverseTrigInput::Sqrt3Over2 => "√3/2",
            InverseTrigInput::OneOverSqrt3 => "√3/3",
            InverseTrigInput::Sqrt3 => "√3",
        }
    }
}

/// Check if expression is n^(1/2) and return n as i64
fn is_sqrt_expr(ctx: &Context, expr: ExprId) -> Option<i64> {
    if let Expr::Pow(base, exp) = ctx.get(expr) {
        // Case 1: exponent is Div(1, 2) — structural form
        if let Expr::Div(num, den) = ctx.get(*exp) {
            if let (Expr::Number(n), Expr::Number(d)) = (ctx.get(*num), ctx.get(*den)) {
                use num_traits::One;
                if n.is_one() && *d == num_rational::BigRational::from_integer(2.into()) {
                    if let Expr::Number(b) = ctx.get(*base) {
                        return b.to_integer().try_into().ok();
                    }
                }
            }
        }
        // Case 2: exponent is Number(1/2) — folded rational form
        if let Expr::Number(exp_val) = ctx.get(*exp) {
            if *exp_val == num_rational::BigRational::new(1.into(), 2.into()) {
                if let Expr::Number(b) = ctx.get(*base) {
                    return b.to_integer().try_into().ok();
                }
            }
        }
    }
    None
}

/// Detect if an expression is a special input for inverse trig
pub fn detect_inverse_trig_input(ctx: &Context, expr: ExprId) -> Option<InverseTrigInput> {
    use num_traits::{One, Zero};

    match ctx.get(expr) {
        // Integer/rational constants: 0, 1, 1/2
        Expr::Number(n) => {
            if n.is_zero() {
                return Some(InverseTrigInput::Zero);
            }
            if n.is_one() {
                return Some(InverseTrigInput::One);
            }
            if *n == num_rational::BigRational::new(1.into(), 2.into()) {
                return Some(InverseTrigInput::Half);
            }
            None
        }

        // Div patterns: √n/d or 1/√n
        Expr::Div(num, den) => {
            if let Expr::Number(d) = ctx.get(*den) {
                let denom: i64 = d.to_integer().try_into().ok()?;

                // √2/2
                if denom == 2 {
                    if let Some(2) = is_sqrt_expr(ctx, *num) {
                        return Some(InverseTrigInput::Sqrt2Over2);
                    }
                }
                // √3/2
                if denom == 2 {
                    if let Some(3) = is_sqrt_expr(ctx, *num) {
                        return Some(InverseTrigInput::Sqrt3Over2);
                    }
                }
                // √3/3 = 1/√3
                if denom == 3 {
                    if let Some(3) = is_sqrt_expr(ctx, *num) {
                        return Some(InverseTrigInput::OneOverSqrt3);
                    }
                }
            }
            // 1/√3
            if let Expr::Number(n) = ctx.get(*num) {
                if n.is_one() {
                    if let Some(3) = is_sqrt_expr(ctx, *den) {
                        return Some(InverseTrigInput::OneOverSqrt3);
                    }
                }
            }
            None
        }

        // Mul patterns: rationalized forms like (1/3)·√3, (1/2)·√2, etc.
        Expr::Mul(l, r) => {
            // Try both orderings
            let (num_id, sqrt_id) = if matches!(ctx.get(*l), Expr::Number(_) | Expr::Div(_, _)) {
                (*l, *r)
            } else {
                (*r, *l)
            };

            if let Some(sqrt_base) = is_sqrt_expr(ctx, sqrt_id) {
                if let Expr::Number(n) = ctx.get(num_id) {
                    // (1/3)·√3 = √3/3 = 1/√3
                    if sqrt_base == 3 && *n == num_rational::BigRational::new(1.into(), 3.into()) {
                        return Some(InverseTrigInput::OneOverSqrt3);
                    }
                    // (1/2)·√2 = √2/2
                    if sqrt_base == 2 && *n == num_rational::BigRational::new(1.into(), 2.into()) {
                        return Some(InverseTrigInput::Sqrt2Over2);
                    }
                    // (1/2)·√3 = √3/2
                    if sqrt_base == 3 && *n == num_rational::BigRational::new(1.into(), 2.into()) {
                        return Some(InverseTrigInput::Sqrt3Over2);
                    }
                }
            }
            None
        }

        // Pow patterns: √3 alone
        Expr::Pow(_, _) => {
            if let Some(3) = is_sqrt_expr(ctx, expr) {
                return Some(InverseTrigInput::Sqrt3);
            }
            None
        }

        _ => None,
    }
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
