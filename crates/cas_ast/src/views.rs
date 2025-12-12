//! # Expression Views
//!
//! This module provides "views" that normalize different expression representations
//! for easier pattern matching in rules. For example, `Div(a,b)`, `Mul(a, Pow(b,-1))`,
//! and `Pow(x, -1)` are all recognized as having a denominator.
//!
//! ## Key Types
//!
//! - `Factor`: A base with a signed integer exponent
//! - `MulParts`: Collects multiplicative factors with sign extraction
//! - `FractionParts`: Separates numerator/denominator factors
//!
//! ## Builders
//!
//! - `build_as_div`: Creates didactic form `Div(num, den)`
//! - `build_as_mulpow`: Creates canonical form `Mul(factors, Pow(den, -1))`

use crate::{Context, Expr, ExprId};
use num_traits::ToPrimitive;
use std::collections::HashMap;

/// A factor in a multiplicative expression: base^exp where exp is a signed integer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Factor {
    pub base: ExprId,
    pub exp: i32, // Signed exponent: negative means in denominator
}

/// Multiplicative parts of an expression, with sign extracted.
///
/// Recognizes: Mul, Div, Pow with integer exponent, Neg
#[derive(Debug, Clone)]
pub struct MulParts {
    pub sign: i8,             // +1 or -1 (extracted from Neg)
    pub factors: Vec<Factor>, // exp is signed (negative = denominator)
}

/// Fraction representation: separated numerator and denominator factors.
///
/// Both num and den have positive exponents.
#[derive(Debug, Clone)]
pub struct FractionParts {
    pub sign: i8,         // +1 or -1
    pub num: Vec<Factor>, // factors with exp > 0
    pub den: Vec<Factor>, // factors with exp > 0 (originally negative)
}

/// Extract integer exponent from an expression if it's an integer Number.
fn int_exp_i32(ctx: &Context, id: ExprId) -> Option<i32> {
    if let Expr::Number(n) = ctx.get(id) {
        if n.is_integer() {
            return n.to_integer().to_i32();
        }
    }
    None
}

impl MulParts {
    /// Create MulParts by collecting all multiplicative factors from an expression.
    ///
    /// Recognizes:
    /// - `Mul(a, b)` → factors from a and b
    /// - `Div(a, b)` → a in numerator, b in denominator
    /// - `Pow(base, k)` where k is integer → base^k
    /// - `Neg(x)` → flips sign, collects from x
    pub fn from(ctx: &Context, id: ExprId) -> Self {
        let mut out = MulParts {
            sign: 1,
            factors: Vec::new(),
        };
        collect_mul(ctx, id, 1, &mut out);
        out.compress(ctx);
        out
    }

    /// Combine factors with same base, remove factors with exp=0
    fn compress(&mut self, ctx: &Context) {
        let mut map: HashMap<ExprId, i32> = HashMap::new();
        for f in self.factors.drain(..) {
            *map.entry(f.base).or_insert(0) += f.exp;
        }
        self.factors = map
            .into_iter()
            .filter_map(|(base, exp)| {
                if exp != 0 {
                    Some(Factor { base, exp })
                } else {
                    None
                }
            })
            .collect();

        // Sort by expression ordering for determinism
        self.factors
            .sort_by(|a, b| crate::ordering::compare_expr(ctx, a.base, b.base));
    }

    /// Split into FractionParts (numerator with exp>0, denominator with exp>0)
    pub fn split_fraction(mut self) -> FractionParts {
        let mut num = Vec::new();
        let mut den = Vec::new();

        for f in self.factors.drain(..) {
            if f.exp > 0 {
                num.push(Factor {
                    base: f.base,
                    exp: f.exp,
                });
            } else {
                den.push(Factor {
                    base: f.base,
                    exp: -f.exp, // Make positive
                });
            }
        }

        FractionParts {
            sign: self.sign,
            num,
            den,
        }
    }

    /// Check if this represents a non-trivial fraction (has denominator factors)
    pub fn has_denominator(&self) -> bool {
        self.factors.iter().any(|f| f.exp < 0)
    }
}

/// Recursively collect multiplicative factors.
fn collect_mul(ctx: &Context, id: ExprId, mult: i32, out: &mut MulParts) {
    match ctx.get(id) {
        Expr::Mul(l, r) => {
            collect_mul(ctx, *l, mult, out);
            collect_mul(ctx, *r, mult, out);
        }
        Expr::Div(n, d) => {
            collect_mul(ctx, *n, mult, out);
            collect_mul(ctx, *d, -mult, out); // denominator gets negated exponent
        }
        Expr::Neg(x) => {
            out.sign *= -1;
            collect_mul(ctx, *x, mult, out);
        }
        Expr::Pow(b, e) => {
            if let Some(k) = int_exp_i32(ctx, *e) {
                // Integer exponent: recurse into base with scaled exponent
                // But only if base is not a nested Pow (to avoid complexity)
                if matches!(ctx.get(*b), Expr::Mul(_, _) | Expr::Div(_, _)) {
                    collect_mul(ctx, *b, mult * k, out);
                } else {
                    out.factors.push(Factor {
                        base: *b,
                        exp: mult * k,
                    });
                }
            } else {
                // Non-integer exponent: treat Pow as atomic factor
                out.factors.push(Factor {
                    base: id,
                    exp: mult,
                });
            }
        }
        _ => {
            // Atomic factor (Number, Variable, Function, etc.)
            out.factors.push(Factor {
                base: id,
                exp: mult,
            });
        }
    }
}

impl FractionParts {
    /// Create FractionParts directly from an expression.
    pub fn from(ctx: &Context, id: ExprId) -> Self {
        MulParts::from(ctx, id).split_fraction()
    }

    /// Check if this actually represents a fraction (has denominator).
    pub fn is_fraction(&self) -> bool {
        !self.den.is_empty()
    }

    /// Check if this is a simple fraction a/b (single term in num and den)
    pub fn is_simple(&self) -> bool {
        self.num.len() <= 1 && self.den.len() <= 1
    }

    /// Build a product from factors: Π base^exp
    fn build_product(ctx: &mut Context, factors: &[Factor]) -> ExprId {
        if factors.is_empty() {
            return ctx.num(1);
        }

        let mut parts: Vec<ExprId> = Vec::with_capacity(factors.len());
        for f in factors {
            let term = if f.exp == 1 {
                f.base
            } else {
                let e = ctx.num(f.exp as i64);
                ctx.add(Expr::Pow(f.base, e))
            };
            parts.push(term);
        }

        let mut acc = parts[0];
        for p in parts.into_iter().skip(1) {
            acc = ctx.add(Expr::Mul(acc, p));
        }
        acc
    }

    /// Build as didactic division: `Div(num, den)` or just `num` if den=1.
    ///
    /// Use this for pedagogical output that shows fractions as a/b.
    pub fn build_as_div(&self, ctx: &mut Context) -> ExprId {
        let num_expr = Self::build_product(ctx, &self.num);
        let den_expr = Self::build_product(ctx, &self.den);

        let mut result = if self.den.is_empty() {
            // No denominator, just return numerator
            num_expr
        } else {
            // Check if denominator is just 1
            if let Expr::Number(n) = ctx.get(den_expr) {
                if n.is_integer() && *n == num_rational::BigRational::from_integer(1.into()) {
                    num_expr
                } else {
                    ctx.add(Expr::Div(num_expr, den_expr))
                }
            } else {
                ctx.add(Expr::Div(num_expr, den_expr))
            }
        };

        // Apply sign
        if self.sign < 0 {
            result = ctx.add(Expr::Neg(result));
        }

        result
    }

    /// Build as canonical multiplication with negative powers: `Mul(factors..., Pow(den, -exp)...)`.
    ///
    /// Use this for internal canonical form.
    pub fn build_as_mulpow(&self, ctx: &mut Context) -> ExprId {
        // Combine num (positive exp) and den (negative exp)
        let mut all_factors: Vec<Factor> = self.num.clone();
        for d in &self.den {
            all_factors.push(Factor {
                base: d.base,
                exp: -(d.exp), // negate for canonical form
            });
        }

        if all_factors.is_empty() {
            // Just ±1
            return ctx.num(self.sign as i64);
        }

        // Sort for determinism
        all_factors.sort_by(|a, b| crate::ordering::compare_expr(ctx, a.base, b.base));

        // Build product
        let mut parts: Vec<ExprId> = Vec::new();
        for f in &all_factors {
            let term = if f.exp == 1 {
                f.base
            } else {
                let e = ctx.num(f.exp as i64);
                ctx.add(Expr::Pow(f.base, e))
            };
            parts.push(term);
        }

        let mut acc = parts[0];
        for p in parts.into_iter().skip(1) {
            acc = ctx.add(Expr::Mul(acc, p));
        }

        // Apply sign
        if self.sign < 0 {
            ctx.add(Expr::Neg(acc))
        } else {
            acc
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_fraction() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");
        let div = ctx.add(Expr::Div(a, b));

        let fp = FractionParts::from(&ctx, div);
        assert!(fp.is_fraction());
        assert_eq!(fp.sign, 1);
        assert_eq!(fp.num.len(), 1);
        assert_eq!(fp.den.len(), 1);
    }

    #[test]
    fn test_reciprocal_pow() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let neg_one = ctx.num(-1);
        let recip = ctx.add(Expr::Pow(x, neg_one));

        let fp = FractionParts::from(&ctx, recip);
        assert!(fp.is_fraction());
        assert_eq!(fp.num.len(), 0);
        assert_eq!(fp.den.len(), 1);
    }

    #[test]
    fn test_mul_with_div() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");
        let c = ctx.var("c");
        // a * (b/c)
        let frac = ctx.add(Expr::Div(b, c));
        let mul = ctx.add(Expr::Mul(a, frac));

        let fp = FractionParts::from(&ctx, mul);
        assert!(fp.is_fraction());
        assert_eq!(fp.num.len(), 2); // a, b
        assert_eq!(fp.den.len(), 1); // c
    }

    #[test]
    fn test_build_as_div() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");
        let neg_one = ctx.num(-1);

        // a * b^(-1) should build back to a/b
        let recip_b = ctx.add(Expr::Pow(b, neg_one));
        let mul = ctx.add(Expr::Mul(a, recip_b));

        let fp = FractionParts::from(&ctx, mul);
        let result = fp.build_as_div(&mut ctx);

        // Should be Div(a, b)
        if let Expr::Div(n, d) = ctx.get(result) {
            assert_eq!(*n, a);
            assert_eq!(*d, b);
        } else {
            panic!("Expected Div");
        }
    }
}
