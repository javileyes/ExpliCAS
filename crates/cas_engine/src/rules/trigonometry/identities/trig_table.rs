//! Table-driven trigonometric evaluation types and tables.
//!
//! This module provides context-independent "specs" for angles and values,
//! along with lookup tables for special trigonometric values.

use cas_ast::{Context, Expr, ExprId};
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
}
