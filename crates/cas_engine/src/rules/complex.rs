//! Complex Number Rules
//!
//! This module provides rules for simplifying expressions involving the
//! imaginary unit `i`. These rules are only registered when `ComplexMode::On`.
//!
//! ## Rules
//! - `ImaginaryPowerRule`: i^n → {1, i, -1, -i} based on n mod 4
//! - `GaussianMulRule`: (a + bi)(c + di) → (ac - bd) + (ad + bc)i
//! - `GaussianAddRule`: (a + bi) + (c + di) → (a + c) + (b + d)i
//! - `GaussianDivRule`: (a + bi)/(c + di) → ... (conjugate method)

use crate::define_rule;
use crate::rule::Rewrite;
use cas_ast::{Constant, Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{One, Zero};

// ============================================================================
// Gaussian Rational Helpers
// ============================================================================

/// A Gaussian rational: a + bi where a, b are BigRational
#[derive(Debug, Clone)]
pub struct GaussianRational {
    pub real: BigRational,
    pub imag: BigRational,
}

impl GaussianRational {
    pub fn new(real: BigRational, imag: BigRational) -> Self {
        Self { real, imag }
    }

    pub fn is_real(&self) -> bool {
        self.imag.is_zero()
    }

    pub fn is_pure_imag(&self) -> bool {
        self.real.is_zero() && !self.imag.is_zero()
    }

    /// Build an expression from this Gaussian rational
    pub fn to_expr(&self, ctx: &mut Context) -> ExprId {
        let zero = BigRational::zero();
        let one = BigRational::one();
        let neg_one = -BigRational::one();

        if self.imag.is_zero() {
            // Pure real: just a
            ctx.add(Expr::Number(self.real.clone()))
        } else if self.real.is_zero() {
            // Pure imaginary: bi
            if self.imag == one {
                ctx.add(Expr::Constant(Constant::I))
            } else if self.imag == neg_one {
                let i = ctx.add(Expr::Constant(Constant::I));
                ctx.add(Expr::Neg(i))
            } else {
                let b = ctx.add(Expr::Number(self.imag.clone()));
                let i = ctx.add(Expr::Constant(Constant::I));
                ctx.add(Expr::Mul(b, i))
            }
        } else {
            // Full: a + bi
            let a = ctx.add(Expr::Number(self.real.clone()));
            let imag_part = if self.imag == one {
                ctx.add(Expr::Constant(Constant::I))
            } else if self.imag == neg_one {
                let i = ctx.add(Expr::Constant(Constant::I));
                ctx.add(Expr::Neg(i))
            } else if self.imag < zero {
                // Negative imaginary: a - |b|i
                let abs_b = ctx.add(Expr::Number(-self.imag.clone()));
                let i = ctx.add(Expr::Constant(Constant::I));
                let bi = ctx.add(Expr::Mul(abs_b, i));
                return ctx.add(Expr::Sub(a, bi));
            } else {
                let b = ctx.add(Expr::Number(self.imag.clone()));
                let i = ctx.add(Expr::Constant(Constant::I));
                ctx.add(Expr::Mul(b, i))
            };
            ctx.add(Expr::Add(a, imag_part))
        }
    }
}

/// Try to extract a Gaussian rational from an expression.
/// Returns Some((a, b)) if expression is of form a + b*i where a, b are rational.
/// Returns None if expression is not in Gaussian rational form.
pub fn extract_gaussian(ctx: &Context, expr: ExprId) -> Option<GaussianRational> {
    match ctx.get(expr) {
        // Pure number: a + 0i
        Expr::Number(n) => Some(GaussianRational::new(n.clone(), BigRational::zero())),

        // Pure i: 0 + 1i
        Expr::Constant(Constant::I) => Some(GaussianRational::new(
            BigRational::zero(),
            BigRational::one(),
        )),

        // -i: 0 + (-1)i
        Expr::Neg(inner) => {
            if let Expr::Constant(Constant::I) = ctx.get(*inner) {
                Some(GaussianRational::new(
                    BigRational::zero(),
                    -BigRational::one(),
                ))
            } else if let Some(g) = extract_gaussian(ctx, *inner) {
                // -expr where expr is gaussian
                Some(GaussianRational::new(-g.real, -g.imag))
            } else {
                None
            }
        }

        // c*i where c is number
        Expr::Mul(l, r) => {
            // Try c*i
            if let Expr::Constant(Constant::I) = ctx.get(*r) {
                if let Expr::Number(n) = ctx.get(*l) {
                    return Some(GaussianRational::new(BigRational::zero(), n.clone()));
                }
            }
            // Try i*c
            if let Expr::Constant(Constant::I) = ctx.get(*l) {
                if let Expr::Number(n) = ctx.get(*r) {
                    return Some(GaussianRational::new(BigRational::zero(), n.clone()));
                }
            }
            None
        }

        // a + bi
        Expr::Add(l, r) => {
            let left = extract_gaussian(ctx, *l)?;
            let right = extract_gaussian(ctx, *r)?;

            // Both sides must contribute to distinct parts
            // e.g., (a) + (bi) → a + bi
            if left.is_real() && right.is_pure_imag() {
                Some(GaussianRational::new(left.real, right.imag))
            } else if left.is_pure_imag() && right.is_real() {
                Some(GaussianRational::new(right.real, left.imag))
            } else if left.is_real() && right.is_real() {
                // a + b → (a+b) + 0i
                Some(GaussianRational::new(
                    left.real + right.real,
                    BigRational::zero(),
                ))
            } else {
                // More general case: (a + bi) + (c + di)
                Some(GaussianRational::new(
                    left.real + right.real,
                    left.imag + right.imag,
                ))
            }
        }

        // a - bi
        Expr::Sub(l, r) => {
            let left = extract_gaussian(ctx, *l)?;
            let right = extract_gaussian(ctx, *r)?;
            Some(GaussianRational::new(
                left.real - right.real,
                left.imag - right.imag,
            ))
        }

        _ => None,
    }
}

// ============================================================================
// ImaginaryPowerRule: i^n → {1, i, -1, -i}
// ============================================================================

define_rule!(ImaginaryPowerRule, "Imaginary Power", |ctx, expr| {
    // i^n where n is integer → 1, i, -1, or -i based on n mod 4
    if let Expr::Pow(base, exp) = ctx.get(expr) {
        let base = *base;
        let exp = *exp;

        // Check if base is i
        if !matches!(ctx.get(base), Expr::Constant(Constant::I)) {
            return None;
        }

        // Check if exponent is integer
        if let Expr::Number(n) = ctx.get(exp) {
            if !n.is_integer() {
                return None;
            }

            // Get n mod 4 - simpler calculation
            let n_int = n.to_integer();
            // Use rem_euclid equivalent: ((n % 4) + 4) % 4 gives 0..3
            use num_bigint::BigInt;
            let four = BigInt::from(4);
            let remainder = ((&n_int % &four) + &four) % &four;
            // Convert to i32 - safe since result is 0..3
            use num_traits::ToPrimitive;
            let normalized = remainder.to_i32().unwrap_or(0) as usize;

            let new_expr = match normalized {
                0 => ctx.num(1),                           // i^0 = 1
                1 => ctx.add(Expr::Constant(Constant::I)), // i^1 = i
                2 => ctx.num(-1),                          // i^2 = -1
                3 => {
                    // i^3 = -i
                    let i = ctx.add(Expr::Constant(Constant::I));
                    ctx.add(Expr::Neg(i))
                }
                _ => unreachable!(),
            };

            let desc = format!(
                "i^{} = {} (using i⁴ = 1)",
                n_int,
                match normalized {
                    0 => "1",
                    1 => "i",
                    2 => "-1",
                    3 => "-i",
                    _ => unreachable!(),
                }
            );

            return Some(Rewrite {
                new_expr,
                description: desc,
                before_local: None,
                after_local: None,
                domain_assumption: None,
            });
        }
    }
    None
});

// ============================================================================
// ISquaredMulRule: i * i → -1 (handles Mul case, not Pow)
// ============================================================================

define_rule!(ISquaredMulRule, "i * i = -1", |ctx, expr| {
    // i * i → -1
    if let Expr::Mul(l, r) = ctx.get(expr) {
        let l = *l;
        let r = *r;

        // Check if both are i
        if matches!(ctx.get(l), Expr::Constant(Constant::I))
            && matches!(ctx.get(r), Expr::Constant(Constant::I))
        {
            let neg_one = ctx.num(-1);
            return Some(Rewrite {
                new_expr: neg_one,
                description: "i · i = -1".to_string(),
                before_local: None,
                after_local: None,
                domain_assumption: None,
            });
        }
    }
    None
});

// ============================================================================
// GaussianMulRule: (a + bi)(c + di) → (ac - bd) + (ad + bc)i
// ============================================================================

define_rule!(GaussianMulRule, "Gaussian Multiplication", |ctx, expr| {
    // (a + bi) * (c + di) → (ac - bd) + (ad + bc)i
    if let Expr::Mul(l, r) = ctx.get(expr) {
        let l = *l;
        let r = *r;

        // Try to extract gaussian rationals from both sides
        let left = extract_gaussian(ctx, l)?;
        let right = extract_gaussian(ctx, r)?;

        // Skip if both are pure real (regular multiplication handles it)
        if left.is_real() && right.is_real() {
            return None;
        }

        // (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        let ac = &left.real * &right.real;
        let bd = &left.imag * &right.imag;
        let ad = &left.real * &right.imag;
        let bc = &left.imag * &right.real;

        let real_part = ac - bd;
        let imag_part = ad + bc;

        let result = GaussianRational::new(real_part, imag_part);
        let new_expr = result.to_expr(ctx);

        return Some(Rewrite {
            new_expr,
            description: "Gaussian multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i".to_string(),
            before_local: None,
            after_local: None,
            domain_assumption: None,
        });
    }
    None
});

// ============================================================================
// GaussianAddRule: (a + bi) + (c + di) → (a + c) + (b + d)i
// Only fires when combining two non-trivial gaussian expressions
// ============================================================================

define_rule!(GaussianAddRule, "Gaussian Addition", |ctx, expr| {
    // (a + bi) + (c + di) → (a + c) + (b + d)i
    if let Expr::Add(l, r) = ctx.get(expr) {
        let l = *l;
        let r = *r;

        // Try to extract gaussian rationals from both sides
        let left = extract_gaussian(ctx, l)?;
        let right = extract_gaussian(ctx, r)?;

        // Skip if both are pure real (regular addition handles it)
        if left.is_real() && right.is_real() {
            return None;
        }

        // Skip if one side is already a + b*i form and the other is just a number
        // This prevents rewriting "a + b*i" to "a + b*i" (no-op)
        // Only combine when BOTH sides have imaginary components OR
        // when we're genuinely combining separate complex numbers
        if (left.is_real() && !right.is_real()) || (!left.is_real() && right.is_real()) {
            // One is real, one is complex - this is already canonical form a + bi
            // Check if this is exactly (real) + (imag*i) pattern
            if left.is_real() && right.is_pure_imag() {
                return None; // Already in a + bi form
            }
            if right.is_real() && left.is_pure_imag() {
                return None; // Already in bi + a form (unusual but valid)
            }
        }

        // (a + bi) + (c + di) = (a + c) + (b + d)i
        let real_part = &left.real + &right.real;
        let imag_part = &left.imag + &right.imag;

        let result = GaussianRational::new(real_part, imag_part);
        let new_expr = result.to_expr(ctx);

        return Some(Rewrite {
            new_expr,
            description: "Gaussian addition: (a+bi) + (c+di) = (a+c) + (b+d)i".to_string(),
            before_local: None,
            after_local: None,
            domain_assumption: None,
        });
    }
    None
});

// ============================================================================
// GaussianDivRule: (a + bi) / (c + di) → conjugate method
// (a+bi)/(c+di) = (a+bi)(c-di) / (c+di)(c-di) = (ac+bd + (bc-ad)i) / (c²+d²)
// ============================================================================

define_rule!(GaussianDivRule, "Gaussian Division", |ctx, expr| {
    // (a + bi) / (c + di) → using conjugate method
    if let Expr::Div(num, den) = ctx.get(expr) {
        let num = *num;
        let den = *den;

        // Try to extract gaussian rationals from both sides
        let numerator = extract_gaussian(ctx, num)?;
        let denominator = extract_gaussian(ctx, den)?;

        // Skip if denominator is pure real (regular division handles it)
        if denominator.is_real() {
            return None;
        }

        // Skip if numerator is pure real and denominator is pure real
        if numerator.is_real() && denominator.is_real() {
            return None;
        }

        // (a + bi) / (c + di) = (ac + bd) / (c² + d²) + (bc - ad) / (c² + d²) · i
        let a = &numerator.real;
        let b = &numerator.imag;
        let c = &denominator.real;
        let d = &denominator.imag;

        // Denominator magnitude squared: c² + d²
        let denom_sq = c * c + d * d;

        // Check for division by zero
        if denom_sq.is_zero() {
            return None; // Let other rules handle undefined
        }

        // Real part: (ac + bd) / (c² + d²)
        let real_num = a * c + b * d;
        let real_part = &real_num / &denom_sq;

        // Imaginary part: (bc - ad) / (c² + d²)
        let imag_num = b * c - a * d;
        let imag_part = &imag_num / &denom_sq;

        let result = GaussianRational::new(real_part, imag_part);
        let new_expr = result.to_expr(ctx);

        return Some(Rewrite {
            new_expr,
            description: "Gaussian division: (a+bi)/(c+di) using conjugate".to_string(),
            before_local: None,
            after_local: None,
            domain_assumption: None,
        });
    }
    None
});

// ============================================================================
// Registration
// ============================================================================

pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(ImaginaryPowerRule));
    simplifier.add_rule(Box::new(ISquaredMulRule));
    simplifier.add_rule(Box::new(GaussianMulRule));
    simplifier.add_rule(Box::new(GaussianAddRule));
    simplifier.add_rule(Box::new(GaussianDivRule));
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rule::Rule;
    use cas_ast::{Context, DisplayExpr};

    #[test]
    fn test_i_squared() {
        let mut ctx = Context::new();
        let rule = ImaginaryPowerRule;

        // i^2 = -1
        let i = ctx.add(Expr::Constant(Constant::I));
        let two = ctx.num(2);
        let i_squared = ctx.add(Expr::Pow(i, two));

        let rewrite = rule
            .apply(
                &mut ctx,
                i_squared,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();

        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "-1"
        );
    }

    #[test]
    fn test_i_cubed() {
        let mut ctx = Context::new();
        let rule = ImaginaryPowerRule;

        // i^3 = -i
        let i = ctx.add(Expr::Constant(Constant::I));
        let three = ctx.num(3);
        let i_cubed = ctx.add(Expr::Pow(i, three));

        let rewrite = rule
            .apply(
                &mut ctx,
                i_cubed,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();

        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "-i"
        );
    }

    #[test]
    fn test_i_fourth() {
        let mut ctx = Context::new();
        let rule = ImaginaryPowerRule;

        // i^4 = 1
        let i = ctx.add(Expr::Constant(Constant::I));
        let four = ctx.num(4);
        let i_fourth = ctx.add(Expr::Pow(i, four));

        let rewrite = rule
            .apply(
                &mut ctx,
                i_fourth,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();

        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "1"
        );
    }

    #[test]
    fn test_i_large_power() {
        let mut ctx = Context::new();
        let rule = ImaginaryPowerRule;

        // i^17 = i^1 = i (17 mod 4 = 1)
        let i = ctx.add(Expr::Constant(Constant::I));
        let seventeen = ctx.num(17);
        let i_17 = ctx.add(Expr::Pow(i, seventeen));

        let rewrite = rule
            .apply(
                &mut ctx,
                i_17,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();

        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "i"
        );
    }

    #[test]
    fn test_extract_gaussian_number() {
        let mut ctx = Context::new();
        let three = ctx.num(3);
        let g = extract_gaussian(&ctx, three).unwrap();
        assert_eq!(g.real, BigRational::from_integer(3.into()));
        assert!(g.imag.is_zero());
    }

    #[test]
    fn test_extract_gaussian_i() {
        let mut ctx = Context::new();
        let i = ctx.add(Expr::Constant(Constant::I));
        let g = extract_gaussian(&ctx, i).unwrap();
        assert!(g.real.is_zero());
        assert!(g.imag.is_one());
    }

    #[test]
    fn test_gaussian_to_expr() {
        let mut ctx = Context::new();
        // 3 + 2i
        let g = GaussianRational::new(
            BigRational::from_integer(3.into()),
            BigRational::from_integer(2.into()),
        );
        let expr = g.to_expr(&mut ctx);
        let display = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: expr
            }
        );
        assert_eq!(display, "3 + 2 * i");
    }
}
