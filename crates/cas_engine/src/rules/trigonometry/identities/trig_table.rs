//! Table-driven trigonometric evaluation types and tables.
//!
//! This module provides context-independent "specs" for angles and values,
//! along with lookup tables for special trigonometric values.

use cas_ast::{Context, Expr, ExprId};
use num_traits::{ToPrimitive, Zero};
use std::cmp::Ordering;

// =============================================================================
// AngleSpec - Represents angles as rational multiples of π
// =============================================================================

/// Represents an angle as (num/den) * π
///
/// # Invariants
/// - `den > 0` always
/// - Fraction is reduced (gcd(|num|, den) = 1)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AngleSpec {
    pub num: i32,
    pub den: i32,
}

impl AngleSpec {
    /// Create a new AngleSpec, automatically normalizing the fraction.
    pub fn new(num: i32, den: i32) -> Self {
        assert!(den != 0, "Denominator cannot be zero");

        // Ensure den > 0
        let (num, den) = if den < 0 { (-num, -den) } else { (num, den) };

        // Reduce by gcd
        let g = gcd(num.unsigned_abs(), den.unsigned_abs()) as i32;
        Self {
            num: num / g,
            den: den / g,
        }
    }

    /// Zero angle (0)
    pub const ZERO: Self = Self { num: 0, den: 1 };

    /// π/6
    pub const PI_6: Self = Self { num: 1, den: 6 };

    /// π/4  
    pub const PI_4: Self = Self { num: 1, den: 4 };

    /// π/3
    pub const PI_3: Self = Self { num: 1, den: 3 };

    /// π/2
    pub const PI_2: Self = Self { num: 1, den: 2 };

    /// π
    pub const PI: Self = Self { num: 1, den: 1 };

    /// 2π
    pub const TWO_PI: Self = Self { num: 2, den: 1 };

    /// Negate this angle
    pub fn negate(self) -> Self {
        Self {
            num: -self.num,
            den: self.den,
        }
    }

    /// Add two angles
    pub fn add(self, other: Self) -> Self {
        // a/b + c/d = (ad + bc) / bd
        let num = self.num * other.den + other.num * self.den;
        let den = self.den * other.den;
        Self::new(num, den)
    }

    /// Subtract two angles
    pub fn sub(self, other: Self) -> Self {
        self.add(other.negate())
    }

    /// Reduce angle modulo 2π to range [0, 2π)
    pub fn reduce_mod_2pi(self) -> Self {
        // We're working with (num/den)*π
        // Mod 2π means mod 2 in the coefficient
        // So we want num/den mod 2, which is (num mod (2*den)) / den
        let period = 2 * self.den;
        let mut num = self.num % period;
        if num < 0 {
            num += period;
        }
        Self::new(num, self.den)
    }

    /// Compare to another angle (as rational numbers)
    pub fn cmp_value(&self, other: &Self) -> Ordering {
        // a/b vs c/d: compare a*d vs c*b
        let lhs = (self.num as i64) * (other.den as i64);
        let rhs = (other.num as i64) * (self.den as i64);
        lhs.cmp(&rhs)
    }

    /// Check if this angle is in range [0, π/2]
    pub fn is_first_quadrant(&self) -> bool {
        // 0 <= num/den <= 1/2 (in units of π)
        let zero = Self::ZERO;
        let pi_2 = Self::PI_2;
        self.cmp_value(&zero) != Ordering::Less && self.cmp_value(&pi_2) != Ordering::Greater
    }

    /// Convert to ExprId representing (num/den)*π
    pub fn to_expr(self, ctx: &mut Context) -> ExprId {
        let pi = ctx.add(Expr::Constant(cas_ast::Constant::Pi));

        if self.num == 0 {
            return ctx.num(0);
        }

        if self.num == 1 && self.den == 1 {
            return pi;
        }

        if self.den == 1 {
            // num * π
            let coeff = ctx.num(self.num as i64);
            ctx.add(Expr::Mul(coeff, pi))
        } else if self.num == 1 {
            // π / den
            let d = ctx.num(self.den as i64);
            ctx.add(Expr::Div(pi, d))
        } else {
            // (num/den) * π = num*π/den
            let n = ctx.num(self.num as i64);
            let num_pi = ctx.add(Expr::Mul(n, pi));
            let d = ctx.num(self.den as i64);
            ctx.add(Expr::Div(num_pi, d))
        }
    }
}

/// Compute GCD using Euclidean algorithm
fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

// =============================================================================
// ValueSpec - Represents special trigonometric values
// =============================================================================

/// Represents common special values that appear in trig evaluation.
/// These are the "recipes" that can be materialized into ExprId.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ValueSpec {
    /// 0
    Zero,
    /// 1
    One,
    /// -1
    NegOne,
    /// 1/2
    Half,
    /// -1/2
    NegHalf,
    /// √2/2
    Sqrt2Over2,
    /// -√2/2
    NegSqrt2Over2,
    /// √3/2
    Sqrt3Over2,
    /// -√3/2
    NegSqrt3Over2,
    /// √3
    Sqrt3,
    /// -√3
    NegSqrt3,
    /// 1/√3 = √3/3
    OneOverSqrt3,
    /// -1/√3 = -√3/3
    NegOneOverSqrt3,
    /// Undefined (e.g., tan(π/2))
    Undefined,
}

impl ValueSpec {
    /// Negate this value
    pub fn negate(self) -> Self {
        match self {
            Self::Zero => Self::Zero,
            Self::One => Self::NegOne,
            Self::NegOne => Self::One,
            Self::Half => Self::NegHalf,
            Self::NegHalf => Self::Half,
            Self::Sqrt2Over2 => Self::NegSqrt2Over2,
            Self::NegSqrt2Over2 => Self::Sqrt2Over2,
            Self::Sqrt3Over2 => Self::NegSqrt3Over2,
            Self::NegSqrt3Over2 => Self::Sqrt3Over2,
            Self::Sqrt3 => Self::NegSqrt3,
            Self::NegSqrt3 => Self::Sqrt3,
            Self::OneOverSqrt3 => Self::NegOneOverSqrt3,
            Self::NegOneOverSqrt3 => Self::OneOverSqrt3,
            Self::Undefined => Self::Undefined,
        }
    }

    /// Convert to ExprId
    pub fn to_expr(self, ctx: &mut Context) -> Option<ExprId> {
        Some(match self {
            Self::Zero => ctx.num(0),
            Self::One => ctx.num(1),
            Self::NegOne => ctx.num(-1),
            Self::Half => {
                let one = ctx.num(1);
                let two = ctx.num(2);
                ctx.add(Expr::Div(one, two))
            }
            Self::NegHalf => {
                let one = ctx.num(1);
                let two = ctx.num(2);
                let half = ctx.add(Expr::Div(one, two));
                ctx.add(Expr::Neg(half))
            }
            Self::Sqrt2Over2 => {
                // √2/2
                let one = ctx.num(1);
                let two = ctx.num(2);
                let half = ctx.add(Expr::Div(one, two));
                let base = ctx.num(2);
                let sqrt2 = ctx.add(Expr::Pow(base, half));
                let two2 = ctx.num(2);
                ctx.add(Expr::Div(sqrt2, two2))
            }
            Self::NegSqrt2Over2 => {
                let val = Self::Sqrt2Over2.to_expr(ctx)?;
                ctx.add(Expr::Neg(val))
            }
            Self::Sqrt3Over2 => {
                // √3/2
                let one = ctx.num(1);
                let two = ctx.num(2);
                let half = ctx.add(Expr::Div(one, two));
                let base = ctx.num(3);
                let sqrt3 = ctx.add(Expr::Pow(base, half));
                let two2 = ctx.num(2);
                ctx.add(Expr::Div(sqrt3, two2))
            }
            Self::NegSqrt3Over2 => {
                let val = Self::Sqrt3Over2.to_expr(ctx)?;
                ctx.add(Expr::Neg(val))
            }
            Self::Sqrt3 => {
                // √3
                let one = ctx.num(1);
                let two = ctx.num(2);
                let half = ctx.add(Expr::Div(one, two));
                let base = ctx.num(3);
                ctx.add(Expr::Pow(base, half))
            }
            Self::NegSqrt3 => {
                let val = Self::Sqrt3.to_expr(ctx)?;
                ctx.add(Expr::Neg(val))
            }
            Self::OneOverSqrt3 => {
                // 1/√3 = √3/3
                let sqrt3 = Self::Sqrt3.to_expr(ctx)?;
                let three = ctx.num(3);
                ctx.add(Expr::Div(sqrt3, three))
            }
            Self::NegOneOverSqrt3 => {
                let val = Self::OneOverSqrt3.to_expr(ctx)?;
                ctx.add(Expr::Neg(val))
            }
            Self::Undefined => return None,
        })
    }
}

// =============================================================================
// Lookup Tables - First quadrant values only
// =============================================================================

/// Sin values for angles in [0, π/2]
pub const SIN_TABLE: &[(AngleSpec, ValueSpec)] = &[
    (AngleSpec::ZERO, ValueSpec::Zero),       // sin(0) = 0
    (AngleSpec::PI_6, ValueSpec::Half),       // sin(π/6) = 1/2
    (AngleSpec::PI_4, ValueSpec::Sqrt2Over2), // sin(π/4) = √2/2
    (AngleSpec::PI_3, ValueSpec::Sqrt3Over2), // sin(π/3) = √3/2
    (AngleSpec::PI_2, ValueSpec::One),        // sin(π/2) = 1
];

/// Cos values for angles in [0, π/2]
pub const COS_TABLE: &[(AngleSpec, ValueSpec)] = &[
    (AngleSpec::ZERO, ValueSpec::One),        // cos(0) = 1
    (AngleSpec::PI_6, ValueSpec::Sqrt3Over2), // cos(π/6) = √3/2
    (AngleSpec::PI_4, ValueSpec::Sqrt2Over2), // cos(π/4) = √2/2
    (AngleSpec::PI_3, ValueSpec::Half),       // cos(π/3) = 1/2
    (AngleSpec::PI_2, ValueSpec::Zero),       // cos(π/2) = 0
];

/// Tan values for angles in [0, π/2)
pub const TAN_TABLE: &[(AngleSpec, ValueSpec)] = &[
    (AngleSpec::ZERO, ValueSpec::Zero),         // tan(0) = 0
    (AngleSpec::PI_6, ValueSpec::OneOverSqrt3), // tan(π/6) = 1/√3
    (AngleSpec::PI_4, ValueSpec::One),          // tan(π/4) = 1
    (AngleSpec::PI_3, ValueSpec::Sqrt3),        // tan(π/3) = √3
    (AngleSpec::PI_2, ValueSpec::Undefined),    // tan(π/2) = undefined
];

/// Asin values: value -> angle in [-π/2, π/2]
pub const ASIN_TABLE: &[(ValueSpec, AngleSpec)] = &[
    (ValueSpec::Zero, AngleSpec::ZERO),
    (ValueSpec::Half, AngleSpec::PI_6),
    (ValueSpec::Sqrt2Over2, AngleSpec::PI_4),
    (ValueSpec::Sqrt3Over2, AngleSpec::PI_3),
    (ValueSpec::One, AngleSpec::PI_2),
    // Negative values
    (ValueSpec::NegHalf, AngleSpec { num: -1, den: 6 }),
    (ValueSpec::NegSqrt2Over2, AngleSpec { num: -1, den: 4 }),
    (ValueSpec::NegSqrt3Over2, AngleSpec { num: -1, den: 3 }),
    (ValueSpec::NegOne, AngleSpec { num: -1, den: 2 }),
];

/// Acos values: value -> angle in [0, π]
pub const ACOS_TABLE: &[(ValueSpec, AngleSpec)] = &[
    (ValueSpec::One, AngleSpec::ZERO),
    (ValueSpec::Sqrt3Over2, AngleSpec::PI_6),
    (ValueSpec::Sqrt2Over2, AngleSpec::PI_4),
    (ValueSpec::Half, AngleSpec::PI_3),
    (ValueSpec::Zero, AngleSpec::PI_2),
    (ValueSpec::NegHalf, AngleSpec { num: 2, den: 3 }), // 2π/3
    (ValueSpec::NegSqrt2Over2, AngleSpec { num: 3, den: 4 }), // 3π/4
    (ValueSpec::NegSqrt3Over2, AngleSpec { num: 5, den: 6 }), // 5π/6
    (ValueSpec::NegOne, AngleSpec::PI),
];

/// Atan values: value -> angle in (-π/2, π/2)
pub const ATAN_TABLE: &[(ValueSpec, AngleSpec)] = &[
    (ValueSpec::Zero, AngleSpec::ZERO),
    (ValueSpec::OneOverSqrt3, AngleSpec::PI_6),
    (ValueSpec::One, AngleSpec::PI_4),
    (ValueSpec::Sqrt3, AngleSpec::PI_3),
    // Negative values
    (ValueSpec::NegOneOverSqrt3, AngleSpec { num: -1, den: 6 }),
    (ValueSpec::NegOne, AngleSpec { num: -1, den: 4 }),
    (ValueSpec::NegSqrt3, AngleSpec { num: -1, den: 3 }),
];

// =============================================================================
// Table Lookup Functions
// =============================================================================

/// Lookup sin value for a first-quadrant angle
pub fn lookup_sin(angle: AngleSpec) -> Option<ValueSpec> {
    SIN_TABLE.iter().find(|(a, _)| *a == angle).map(|(_, v)| *v)
}

/// Lookup cos value for a first-quadrant angle
pub fn lookup_cos(angle: AngleSpec) -> Option<ValueSpec> {
    COS_TABLE.iter().find(|(a, _)| *a == angle).map(|(_, v)| *v)
}

/// Lookup tan value for a first-quadrant angle
pub fn lookup_tan(angle: AngleSpec) -> Option<ValueSpec> {
    TAN_TABLE.iter().find(|(a, _)| *a == angle).map(|(_, v)| *v)
}

/// Lookup asin angle for a special value
pub fn lookup_asin(value: ValueSpec) -> Option<AngleSpec> {
    ASIN_TABLE
        .iter()
        .find(|(v, _)| *v == value)
        .map(|(_, a)| *a)
}

/// Lookup acos angle for a special value
pub fn lookup_acos(value: ValueSpec) -> Option<AngleSpec> {
    ACOS_TABLE
        .iter()
        .find(|(v, _)| *v == value)
        .map(|(_, a)| *a)
}

/// Lookup atan angle for a special value
pub fn lookup_atan(value: ValueSpec) -> Option<AngleSpec> {
    ATAN_TABLE
        .iter()
        .find(|(v, _)| *v == value)
        .map(|(_, a)| *a)
}

// =============================================================================
// Angle Normalizer - Reduces any angle to first quadrant + sign info
// =============================================================================

/// Normalized angle: base angle in [0, π/2] plus transformation info.
///
/// This allows lookup in small first-quadrant tables and then
/// applying adjustments based on the original quadrant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NormAngle {
    /// Base angle in [0, π/2] (first quadrant)
    pub base: AngleSpec,
    /// Sign for sin value: +1 or -1
    pub sin_sign: i8,
    /// Sign for cos value: +1 or -1
    pub cos_sign: i8,
    /// If true, swap sin↔cos (cofunction identity)
    pub swap_sin_cos: bool,
}

impl NormAngle {
    /// Apply normalization to get sin value from base lookup
    pub fn apply_sin(&self, base_val: ValueSpec) -> ValueSpec {
        let val = if self.swap_sin_cos {
            // Use cos of base angle (cofunction identity)
            lookup_cos(self.base).unwrap_or(base_val)
        } else {
            base_val
        };
        if self.sin_sign < 0 {
            val.negate()
        } else {
            val
        }
    }

    /// Apply normalization to get cos value from base lookup
    pub fn apply_cos(&self, base_val: ValueSpec) -> ValueSpec {
        let val = if self.swap_sin_cos {
            // Use sin of base angle (cofunction identity)
            lookup_sin(self.base).unwrap_or(base_val)
        } else {
            base_val
        };
        if self.cos_sign < 0 {
            val.negate()
        } else {
            val
        }
    }

    /// Apply normalization to get tan value  
    /// tan sign = sin_sign * cos_sign
    pub fn apply_tan(&self, base_val: ValueSpec) -> ValueSpec {
        let tan_sign = self.sin_sign * self.cos_sign;
        if tan_sign < 0 {
            base_val.negate()
        } else {
            base_val
        }
    }
}

/// Normalize an angle to first quadrant [0, π/2] with sign information.
///
/// Returns `NormAngle` containing:
/// - `base`: the reference angle in [0, π/2]
/// - `sin_sign`, `cos_sign`: signs to apply after lookup
/// - `swap_sin_cos`: whether to use cofunction identity
///
/// # Algorithm
/// 1. Reduce mod 2π to get angle in [0, 2π)
/// 2. Determine quadrant and apply identities:
///    - Q1 [0, π/2]: no change
///    - Q2 [π/2, π]: sin(π-x) = sin(x), cos(π-x) = -cos(x)
///    - Q3 [π, 3π/2]: sin(π+x) = -sin(x), cos(π+x) = -cos(x)
///    - Q4 [3π/2, 2π]: sin(2π-x) = -sin(x), cos(2π-x) = cos(x)
pub fn normalize_angle(angle: AngleSpec) -> NormAngle {
    // First reduce to [0, 2π)
    let reduced = angle.reduce_mod_2pi();

    // Compute quadrant boundaries as AngleSpec
    let pi_2 = AngleSpec::PI_2; // π/2
    let pi = AngleSpec::PI; // π
    let pi_3_2 = AngleSpec::new(3, 2); // 3π/2

    // Determine quadrant and compute base + signs
    if reduced.cmp_value(&pi_2) != Ordering::Greater {
        // Q1: [0, π/2]
        NormAngle {
            base: reduced,
            sin_sign: 1,
            cos_sign: 1,
            swap_sin_cos: false,
        }
    } else if reduced.cmp_value(&pi) != Ordering::Greater {
        // Q2: (π/2, π]
        // sin(π - x) = sin(x), cos(π - x) = -cos(x)
        let base = pi.sub(reduced);
        NormAngle {
            base,
            sin_sign: 1,
            cos_sign: -1,
            swap_sin_cos: false,
        }
    } else if reduced.cmp_value(&pi_3_2) != Ordering::Greater {
        // Q3: (π, 3π/2]
        // sin(x) = -sin(x - π), cos(x) = -cos(x - π)
        let base = reduced.sub(pi);
        NormAngle {
            base,
            sin_sign: -1,
            cos_sign: -1,
            swap_sin_cos: false,
        }
    } else {
        // Q4: (3π/2, 2π)
        // sin(2π - x) = -sin(x), cos(2π - x) = cos(x)
        let two_pi = AngleSpec::TWO_PI;
        let base = two_pi.sub(reduced);
        NormAngle {
            base,
            sin_sign: -1,
            cos_sign: 1,
            swap_sin_cos: false,
        }
    }
}

// =============================================================================
// Evaluators - Connect AST to tables
// =============================================================================

/// Trig function type for unified evaluation
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrigFn {
    Sin,
    Cos,
    Tan,
}

/// Try to extract an AngleSpec from an expression that looks like k*π/n.
///
/// Recognizes patterns:
/// - π -> 1/1
/// - k*π -> k/1
/// - π/n -> 1/n
/// - k*π/n -> k/n
/// - (k/n)*π -> k/n
/// - 0 (the number) -> 0/1
pub fn parse_angle_from_expr(ctx: &Context, expr: ExprId) -> Option<AngleSpec> {
    match ctx.get(expr) {
        // Zero constant
        Expr::Number(n) if n.is_zero() => Some(AngleSpec::ZERO),

        // π alone
        Expr::Constant(cas_ast::Constant::Pi) => Some(AngleSpec::PI),

        // k * π or π * k
        Expr::Mul(a, b) => {
            // Check if one is π and the other is integer
            if matches!(ctx.get(*a), Expr::Constant(cas_ast::Constant::Pi)) {
                // π * k
                if let Expr::Number(n) = ctx.get(*b) {
                    let k = n.to_integer().to_i32()?;
                    return Some(AngleSpec::new(k, 1));
                }
            }
            if matches!(ctx.get(*b), Expr::Constant(cas_ast::Constant::Pi)) {
                // k * π
                if let Expr::Number(n) = ctx.get(*a) {
                    let k = n.to_integer().to_i32()?;
                    return Some(AngleSpec::new(k, 1));
                }
                // (k/n) * π
                if let Expr::Div(num, den) = ctx.get(*a) {
                    if let (Expr::Number(n), Expr::Number(d)) = (ctx.get(*num), ctx.get(*den)) {
                        let k = n.to_integer().to_i32()?;
                        let m = d.to_integer().to_i32()?;
                        if m != 0 {
                            return Some(AngleSpec::new(k, m));
                        }
                    }
                }
            }
            None
        }

        // π / n or (k*π) / n
        Expr::Div(num, den) => {
            if let Expr::Number(d) = ctx.get(*den) {
                let denom = d.to_integer().to_i32()?;
                if denom == 0 {
                    return None;
                }

                // π / n
                if matches!(ctx.get(*num), Expr::Constant(cas_ast::Constant::Pi)) {
                    return Some(AngleSpec::new(1, denom));
                }

                // (k * π) / n
                if let Expr::Mul(a, b) = ctx.get(*num) {
                    if matches!(ctx.get(*b), Expr::Constant(cas_ast::Constant::Pi)) {
                        if let Expr::Number(n) = ctx.get(*a) {
                            let k = n.to_integer().to_i32()?;
                            return Some(AngleSpec::new(k, denom));
                        }
                    }
                    if matches!(ctx.get(*a), Expr::Constant(cas_ast::Constant::Pi)) {
                        if let Expr::Number(n) = ctx.get(*b) {
                            let k = n.to_integer().to_i32()?;
                            return Some(AngleSpec::new(k, denom));
                        }
                    }
                }
            }
            None
        }

        // Neg(-x) -> parse x and negate
        Expr::Neg(inner) => {
            let a = parse_angle_from_expr(ctx, *inner)?;
            Some(a.negate())
        }

        _ => None,
    }
}

/// Evaluate sin/cos/tan at a special angle, returning the simplified ExprId.
///
/// This is the main entry point for table-driven trig evaluation.
/// Returns `None` if the angle is not a special value we know about.
pub fn eval_trig_special(ctx: &mut Context, f: TrigFn, arg: ExprId) -> Option<ExprId> {
    // Step 1: Parse angle from expression
    let angle = parse_angle_from_expr(ctx, arg)?;

    // Step 2: Normalize to first quadrant
    let norm = normalize_angle(angle);

    // Step 3: Lookup base value in table
    let base_val = match f {
        TrigFn::Sin => lookup_sin(norm.base),
        TrigFn::Cos => lookup_cos(norm.base),
        TrigFn::Tan => lookup_tan(norm.base),
    }?;

    // Step 4: Apply normalization (signs, swaps)
    let final_val = match f {
        TrigFn::Sin => norm.apply_sin(base_val),
        TrigFn::Cos => norm.apply_cos(base_val),
        TrigFn::Tan => norm.apply_tan(base_val),
    };

    // Step 5: Convert to ExprId
    final_val.to_expr(ctx)
}

// =============================================================================
// Value Parser - Recognize special values in AST for inverse trig
// =============================================================================

/// Inverse trig function type
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InvTrigFn {
    Asin,
    Acos,
    Atan,
}

/// Try to parse an expression as a special value (for inverse trig evaluation).
///
/// Recognizes patterns:
/// - 0, 1, -1
/// - 1/2, -1/2
/// - sqrt(2)/2, -sqrt(2)/2  (also 2^(1/2)/2)
/// - sqrt(3)/2, -sqrt(3)/2
/// - sqrt(3), -sqrt(3)
/// - sqrt(3)/3 (= 1/sqrt(3)), -sqrt(3)/3
pub fn parse_special_value(ctx: &Context, expr: ExprId) -> Option<ValueSpec> {
    use num_traits::One;

    match ctx.get(expr) {
        // Integer constants: 0, 1, -1
        Expr::Number(n) => {
            if n.is_zero() {
                Some(ValueSpec::Zero)
            } else if n.is_one() {
                Some(ValueSpec::One)
            } else if *n == num_rational::Ratio::from_integer((-1).into()) {
                Some(ValueSpec::NegOne)
            } else if *n == num_rational::Ratio::new(1.into(), 2.into()) {
                Some(ValueSpec::Half)
            } else if *n == num_rational::Ratio::new((-1).into(), 2.into()) {
                Some(ValueSpec::NegHalf)
            } else {
                None
            }
        }

        // Negation: parse inner and negate
        Expr::Neg(inner) => {
            let v = parse_special_value(ctx, *inner)?;
            Some(v.negate())
        }

        // Division patterns: 1/2, sqrt(n)/2, sqrt(n)/3
        Expr::Div(num, den) => {
            // Check denominator
            if let Expr::Number(d) = ctx.get(*den) {
                let denom = d.to_integer().to_i32()?;

                // 1/2 or -1/2
                if denom == 2 {
                    if let Expr::Number(n) = ctx.get(*num) {
                        if n.is_one() {
                            return Some(ValueSpec::Half);
                        }
                        if *n == num_rational::Ratio::from_integer((-1).into()) {
                            return Some(ValueSpec::NegHalf);
                        }
                    }

                    // sqrt(2)/2 or sqrt(3)/2
                    if let Some(sqrt_base) = is_sqrt(ctx, *num) {
                        if sqrt_base == 2 {
                            return Some(ValueSpec::Sqrt2Over2);
                        }
                        if sqrt_base == 3 {
                            return Some(ValueSpec::Sqrt3Over2);
                        }
                    }

                    // -sqrt(2)/2 or -sqrt(3)/2
                    if let Expr::Neg(inner) = ctx.get(*num) {
                        if let Some(sqrt_base) = is_sqrt(ctx, *inner) {
                            if sqrt_base == 2 {
                                return Some(ValueSpec::NegSqrt2Over2);
                            }
                            if sqrt_base == 3 {
                                return Some(ValueSpec::NegSqrt3Over2);
                            }
                        }
                    }
                }

                // sqrt(3)/3 = 1/sqrt(3)
                if denom == 3 {
                    if let Some(sqrt_base) = is_sqrt(ctx, *num) {
                        if sqrt_base == 3 {
                            return Some(ValueSpec::OneOverSqrt3);
                        }
                    }
                    if let Expr::Neg(inner) = ctx.get(*num) {
                        if let Some(sqrt_base) = is_sqrt(ctx, *inner) {
                            if sqrt_base == 3 {
                                return Some(ValueSpec::NegOneOverSqrt3);
                            }
                        }
                    }
                }
            }

            // 1/sqrt(3)
            if let Expr::Number(n) = ctx.get(*num) {
                if n.is_one() {
                    if let Some(sqrt_base) = is_sqrt(ctx, *den) {
                        if sqrt_base == 3 {
                            return Some(ValueSpec::OneOverSqrt3);
                        }
                    }
                }
            }

            None
        }

        // sqrt(n) alone
        Expr::Pow(base, exp) => {
            // Check if this is sqrt(3)
            if let Some(sqrt_base) = is_sqrt_pow(ctx, *base, *exp) {
                if sqrt_base == 3 {
                    return Some(ValueSpec::Sqrt3);
                }
            }
            None
        }

        _ => None,
    }
}

/// Check if expression is sqrt(n) and return n
fn is_sqrt(ctx: &Context, expr: ExprId) -> Option<i32> {
    if let Expr::Pow(base, exp) = ctx.get(expr) {
        is_sqrt_pow(ctx, *base, *exp)
    } else {
        None
    }
}

/// Check if base^exp is sqrt(base) and return the base as i32
fn is_sqrt_pow(ctx: &Context, base: ExprId, exp: ExprId) -> Option<i32> {
    // exp should be 1/2
    if let Expr::Div(num, den) = ctx.get(exp) {
        if let (Expr::Number(n), Expr::Number(d)) = (ctx.get(*num), ctx.get(*den)) {
            if n.to_integer().to_i32()? == 1 && d.to_integer().to_i32()? == 2 {
                // base should be an integer
                if let Expr::Number(b) = ctx.get(base) {
                    return b.to_integer().to_i32();
                }
            }
        }
    }
    None
}

/// Evaluate inverse trig (asin/acos/atan) at a special value.
///
/// Returns the angle as ExprId if the value is recognized.
pub fn eval_inv_trig_special(ctx: &mut Context, f: InvTrigFn, arg: ExprId) -> Option<ExprId> {
    // Step 1: Parse value from expression
    let value = parse_special_value(ctx, arg)?;

    // Step 2: Lookup angle in table
    let angle = match f {
        InvTrigFn::Asin => lookup_asin(value),
        InvTrigFn::Acos => lookup_acos(value),
        InvTrigFn::Atan => lookup_atan(value),
    }?;

    // Step 3: Convert to ExprId
    Some(angle.to_expr(ctx))
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::{One, Zero};

    #[test]
    fn test_angle_spec_new() {
        // Basic normalization
        assert_eq!(AngleSpec::new(2, 4), AngleSpec { num: 1, den: 2 });
        assert_eq!(AngleSpec::new(-2, 4), AngleSpec { num: -1, den: 2 });
        assert_eq!(AngleSpec::new(2, -4), AngleSpec { num: -1, den: 2 });
        assert_eq!(AngleSpec::new(-2, -4), AngleSpec { num: 1, den: 2 });
    }

    #[test]
    fn test_angle_spec_add() {
        let a = AngleSpec::new(1, 6); // π/6
        let b = AngleSpec::new(1, 3); // π/3
        let sum = a.add(b); // π/6 + π/3 = π/2
        assert_eq!(sum, AngleSpec::PI_2);
    }

    #[test]
    fn test_angle_spec_reduce_mod_2pi() {
        // 5π/2 mod 2π = π/2
        let a = AngleSpec::new(5, 2);
        let reduced = a.reduce_mod_2pi();
        assert_eq!(reduced, AngleSpec::PI_2);

        // -π/2 mod 2π = 3π/2
        let b = AngleSpec::new(-1, 2);
        let reduced = b.reduce_mod_2pi();
        assert_eq!(reduced, AngleSpec::new(3, 2));
    }

    #[test]
    fn test_value_spec_negate() {
        assert_eq!(ValueSpec::Zero.negate(), ValueSpec::Zero);
        assert_eq!(ValueSpec::One.negate(), ValueSpec::NegOne);
        assert_eq!(ValueSpec::NegOne.negate(), ValueSpec::One);
        assert_eq!(ValueSpec::Half.negate(), ValueSpec::NegHalf);
        assert_eq!(ValueSpec::Sqrt2Over2.negate(), ValueSpec::NegSqrt2Over2);
    }

    #[test]
    fn test_lookup_sin() {
        assert_eq!(lookup_sin(AngleSpec::ZERO), Some(ValueSpec::Zero));
        assert_eq!(lookup_sin(AngleSpec::PI_6), Some(ValueSpec::Half));
        assert_eq!(lookup_sin(AngleSpec::PI_4), Some(ValueSpec::Sqrt2Over2));
        assert_eq!(lookup_sin(AngleSpec::PI_3), Some(ValueSpec::Sqrt3Over2));
        assert_eq!(lookup_sin(AngleSpec::PI_2), Some(ValueSpec::One));
    }

    #[test]
    fn test_lookup_cos() {
        assert_eq!(lookup_cos(AngleSpec::ZERO), Some(ValueSpec::One));
        assert_eq!(lookup_cos(AngleSpec::PI_6), Some(ValueSpec::Sqrt3Over2));
        assert_eq!(lookup_cos(AngleSpec::PI_4), Some(ValueSpec::Sqrt2Over2));
        assert_eq!(lookup_cos(AngleSpec::PI_3), Some(ValueSpec::Half));
        assert_eq!(lookup_cos(AngleSpec::PI_2), Some(ValueSpec::Zero));
    }

    #[test]
    fn test_lookup_tan() {
        assert_eq!(lookup_tan(AngleSpec::ZERO), Some(ValueSpec::Zero));
        assert_eq!(lookup_tan(AngleSpec::PI_4), Some(ValueSpec::One));
        assert_eq!(lookup_tan(AngleSpec::PI_2), Some(ValueSpec::Undefined));
    }

    #[test]
    fn test_angle_to_expr() {
        let mut ctx = cas_ast::Context::new();

        // 0 -> 0
        let zero = AngleSpec::ZERO.to_expr(&mut ctx);
        assert!(matches!(ctx.get(zero), Expr::Number(n) if n.is_zero()));

        // π -> π
        let pi = AngleSpec::PI.to_expr(&mut ctx);
        assert!(matches!(ctx.get(pi), Expr::Constant(cas_ast::Constant::Pi)));
    }

    #[test]
    fn test_value_to_expr() {
        let mut ctx = cas_ast::Context::new();

        // Zero -> 0
        let zero = ValueSpec::Zero.to_expr(&mut ctx).unwrap();
        assert!(matches!(ctx.get(zero), Expr::Number(n) if n.is_zero()));

        // One -> 1
        let one = ValueSpec::One.to_expr(&mut ctx).unwrap();
        assert!(matches!(ctx.get(one), Expr::Number(n) if n.is_one()));

        // Undefined -> None
        assert!(ValueSpec::Undefined.to_expr(&mut ctx).is_none());
    }

    #[test]
    fn test_normalize_angle_q1() {
        // Q1: π/6 stays as is
        let norm = normalize_angle(AngleSpec::PI_6);
        assert_eq!(norm.base, AngleSpec::PI_6);
        assert_eq!(norm.sin_sign, 1);
        assert_eq!(norm.cos_sign, 1);
    }

    #[test]
    fn test_normalize_angle_q2() {
        // Q2: 2π/3 -> base = π - 2π/3 = π/3
        let angle = AngleSpec::new(2, 3); // 2π/3
        let norm = normalize_angle(angle);
        assert_eq!(norm.base, AngleSpec::PI_3);
        assert_eq!(norm.sin_sign, 1); // sin positive in Q2
        assert_eq!(norm.cos_sign, -1); // cos negative in Q2
    }

    #[test]
    fn test_normalize_angle_q3() {
        // Q3: 7π/6 -> base = 7π/6 - π = π/6
        let angle = AngleSpec::new(7, 6); // 7π/6
        let norm = normalize_angle(angle);
        assert_eq!(norm.base, AngleSpec::PI_6);
        assert_eq!(norm.sin_sign, -1); // sin negative in Q3
        assert_eq!(norm.cos_sign, -1); // cos negative in Q3
    }

    #[test]
    fn test_normalize_angle_q4() {
        // Q4: 11π/6 -> base = 2π - 11π/6 = π/6
        let angle = AngleSpec::new(11, 6); // 11π/6
        let norm = normalize_angle(angle);
        assert_eq!(norm.base, AngleSpec::PI_6);
        assert_eq!(norm.sin_sign, -1); // sin negative in Q4
        assert_eq!(norm.cos_sign, 1); // cos positive in Q4
    }

    #[test]
    fn test_normalize_angle_negative() {
        // -π/6 mod 2π = 11π/6 -> Q4
        let angle = AngleSpec::new(-1, 6);
        let norm = normalize_angle(angle);
        assert_eq!(norm.base, AngleSpec::PI_6);
        assert_eq!(norm.sin_sign, -1);
        assert_eq!(norm.cos_sign, 1);
    }
}
