//! Light exact ℚ(√n) element: `rat + surd·√n` with `BigRational` components.
//!
//! This is the struct-ification of the `(A, B, n)` triple that
//! [`crate::root_forms::as_linear_surd`] already produces, closed under the
//! operations the G1 Cap. C partial-fraction algebra needs: `+`, `−`, `×`,
//! negation and conjugation (see `docs/G1_RATIONAL_INTEGRATION_SCOPING.md`,
//! sub-cycle C-i). **Division is deliberately absent**: the conjugate-ansatz
//! coefficient solve reduces to rational Gaussian elimination, so a general
//! ℚ(√n) inverse is not needed and would be untested surface.
//!
//! Canonical invariants (enforced by every constructor):
//! - the radicand `n` is a non-negative rational (`new` declines `n < 0`);
//! - `surd == 0 ⟹ n == 0` (a rational value has a canonical zero radicand);
//! - a perfect-square radicand folds away (`3·√4` becomes the rational `6`),
//!   so `is_rational`/`is_zero` and `==` are exact structural checks.
//!
//! Binary operations on two genuinely-irrational operands require the SAME
//! radicand (mixing `√2` with `√3` leaves ℚ(√n) — they return `None`, the same
//! honest decline as `as_linear_surd`). Radicands are compared exactly, so
//! callers must feed a consistent spelling (`2·√2`, not `√8`); trees that went
//! through the simplifier — the `from_expr` path — already are.

use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

use crate::perfect_square_support::rational_sqrt;
use crate::root_forms::as_linear_surd;

/// An exact element `rat + surd·√n` of a real quadratic field ℚ(√n).
///
/// Fields are private so every value in circulation satisfies the canonical
/// invariants documented at the module level; read them back through
/// [`QuadSurd::rational_part`], [`QuadSurd::surd_part`] and
/// [`QuadSurd::radicand`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuadSurd {
    rat: BigRational,
    surd: BigRational,
    n: BigRational,
}

impl QuadSurd {
    /// Build `rat + surd·√n`, normalizing to the canonical form. Returns `None`
    /// when `n` is negative (not a real surd).
    pub fn new(rat: BigRational, surd: BigRational, n: BigRational) -> Option<Self> {
        if n.is_negative() {
            return None;
        }
        // √0 = 0: the surd term vanishes.
        if surd.is_zero() || n.is_zero() {
            return Some(Self {
                rat,
                surd: BigRational::zero(),
                n: BigRational::zero(),
            });
        }
        // A perfect-square radicand is rational in disguise: fold `surd·√(s²)` into `rat`.
        if let Some(root) = rational_sqrt(&n) {
            return Some(Self {
                rat: rat + surd * root,
                surd: BigRational::zero(),
                n: BigRational::zero(),
            });
        }
        Some(Self { rat, surd, n })
    }

    /// The purely rational element `r` (canonical zero radicand).
    pub fn from_rational(rat: BigRational) -> Self {
        Self {
            rat,
            surd: BigRational::zero(),
            n: BigRational::zero(),
        }
    }

    /// Parse a constant expression of the form `A + B·√n` into a `QuadSurd`,
    /// bridging [`as_linear_surd`] so there is exactly one surd-parsing path.
    /// `None` for anything outside a single real quadratic surd (nested
    /// radicals, distinct radicands, variables, transcendentals).
    pub fn from_expr(ctx: &Context, expr: ExprId) -> Option<Self> {
        let (rat, surd, n) = as_linear_surd(ctx, expr)?;
        Self::new(rat, surd, n)
    }

    /// The rational part `rat` of `rat + surd·√n`.
    pub fn rational_part(&self) -> &BigRational {
        &self.rat
    }

    /// The surd coefficient `surd` of `rat + surd·√n` (zero for a rational value).
    pub fn surd_part(&self) -> &BigRational {
        &self.surd
    }

    /// The radicand `n` (canonically zero for a rational value).
    pub fn radicand(&self) -> &BigRational {
        &self.n
    }

    /// Whether the value lies in ℚ (no surd component).
    pub fn is_rational(&self) -> bool {
        self.surd.is_zero()
    }

    /// Whether the value is exactly zero.
    pub fn is_zero(&self) -> bool {
        self.rat.is_zero() && self.surd.is_zero()
    }

    /// The shared radicand for a binary operation: a rational operand adopts the
    /// other side's field, two irrational operands must agree exactly. `None`
    /// when the radicands differ (the result leaves ℚ(√n)).
    fn common_radicand(&self, other: &Self) -> Option<BigRational> {
        if self.is_rational() {
            return Some(other.n.clone());
        }
        if other.is_rational() {
            return Some(self.n.clone());
        }
        if self.n == other.n {
            return Some(self.n.clone());
        }
        None
    }

    /// `self + other`; `None` when the radicands are incompatible.
    pub fn add(&self, other: &Self) -> Option<Self> {
        let n = self.common_radicand(other)?;
        Self::new(&self.rat + &other.rat, &self.surd + &other.surd, n)
    }

    /// `self − other`; `None` when the radicands are incompatible.
    pub fn sub(&self, other: &Self) -> Option<Self> {
        let n = self.common_radicand(other)?;
        Self::new(&self.rat - &other.rat, &self.surd - &other.surd, n)
    }

    /// `self · other = (ac + bd·n) + (ad + bc)·√n`; `None` when the radicands
    /// are incompatible.
    pub fn mul(&self, other: &Self) -> Option<Self> {
        let n = self.common_radicand(other)?;
        let rat = &self.rat * &other.rat + &self.surd * &other.surd * &n;
        let surd = &self.rat * &other.surd + &self.surd * &other.rat;
        Self::new(rat, surd, n)
    }

    /// `−self`.
    pub fn neg(&self) -> Self {
        Self {
            rat: -self.rat.clone(),
            surd: -self.surd.clone(),
            n: self.n.clone(),
        }
    }

    /// The field conjugate `rat − surd·√n` (the `√n ↦ −√n` automorphism).
    pub fn conj(&self) -> Self {
        Self {
            rat: self.rat.clone(),
            surd: -self.surd.clone(),
            n: self.n.clone(),
        }
    }

    /// Render as an expression tree `rat + surd·sqrt(n)` (just `rat` for a
    /// rational value). Emits the plain shape [`from_expr`](Self::from_expr)
    /// reads back, so `from_expr(to_expr(q)) == q`.
    pub fn to_expr(&self, ctx: &mut Context) -> ExprId {
        let rat_expr = ctx.add(Expr::Number(self.rat.clone()));
        if self.is_rational() {
            return rat_expr;
        }
        let n_expr = ctx.add(Expr::Number(self.n.clone()));
        let sqrt = ctx.call_builtin(BuiltinFn::Sqrt, vec![n_expr]);
        let term = if self.surd.is_one() {
            sqrt
        } else {
            let coeff = ctx.add(Expr::Number(self.surd.clone()));
            ctx.add(Expr::Mul(coeff, sqrt))
        };
        if self.rat.is_zero() {
            term
        } else {
            ctx.add(Expr::Add(rat_expr, term))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    fn q(rat: (i64, i64), surd: (i64, i64), n: i64) -> QuadSurd {
        QuadSurd::new(
            BigRational::new(rat.0.into(), rat.1.into()),
            BigRational::new(surd.0.into(), surd.1.into()),
            BigRational::from_integer(n.into()),
        )
        .expect("valid radicand")
    }

    fn rational(v: i64) -> QuadSurd {
        QuadSurd::from_rational(BigRational::from_integer(v.into()))
    }

    #[test]
    fn constructor_normalizes_to_canonical_form() {
        // Perfect-square radicand folds to a rational: 1 + 3·√4 = 7.
        let folded = q((1, 1), (3, 1), 4);
        assert!(folded.is_rational());
        assert_eq!(folded, rational(7));
        // Rational fraction radicand too: 2·√(9/4) = 3.
        assert_eq!(q((0, 1), (2, 1), 0), rational(0));
        let frac = QuadSurd::new(
            BigRational::zero(),
            BigRational::from_integer(2.into()),
            BigRational::new(9.into(), 4.into()),
        )
        .expect("valid");
        assert_eq!(frac, rational(3));
        // Zero surd coefficient gets the canonical zero radicand (exact equality works).
        assert_eq!(q((5, 1), (0, 1), 7), rational(5));
        // A negative radicand is not a real surd.
        assert!(QuadSurd::new(
            BigRational::zero(),
            BigRational::one(),
            BigRational::from_integer((-2).into()),
        )
        .is_none());
        // A genuine surd stays put.
        let phi = q((1, 2), (1, 2), 5);
        assert!(!phi.is_rational());
        assert!(!phi.is_zero());
        assert_eq!(phi.radicand(), &BigRational::from_integer(5.into()));
    }

    #[test]
    fn golden_ratio_identities_via_mul() {
        // φ = (1+√5)/2, ψ = conj(φ) = (1−√5)/2.
        let phi = q((1, 2), (1, 2), 5);
        let psi = phi.conj();
        let one = rational(1);
        // φ² = φ + 1.
        assert_eq!(
            phi.mul(&phi).expect("same field"),
            phi.add(&one).expect("same field")
        );
        // φ·ψ = −1 (the norm), a rational — the surd coefficient cancels exactly.
        let norm = phi.mul(&psi).expect("same field");
        assert!(norm.is_rational());
        assert_eq!(norm, rational(-1));
        // φ + ψ = 1 (the trace).
        assert_eq!(phi.add(&psi).expect("same field"), one);
    }

    #[test]
    fn phi5_factor_pair_multiplies_back_to_the_cyclotomic_coefficients() {
        // (x² + φx + 1)(x² + ψx + 1) = x⁴ + x³ + x² + x + 1 (Φ₅): compute the product's
        // coefficients with QuadSurd arithmetic and check every one is the rational 1.
        let phi = q((1, 2), (1, 2), 5);
        let psi = phi.conj();
        let one = rational(1);
        let x3 = phi.add(&psi).expect("x³ coefficient: φ + ψ");
        let x2 = one
            .add(&one)
            .and_then(|two| two.add(&phi.mul(&psi).expect("φψ")))
            .expect("x² coefficient: 1 + φψ + 1");
        let x1 = phi.add(&psi).expect("x¹ coefficient: φ + ψ");
        for (label, coeff) in [("x³", &x3), ("x²", &x2), ("x¹", &x1)] {
            assert_eq!(coeff, &one, "{label} coefficient of Φ₅");
        }
    }

    #[test]
    fn arithmetic_respects_field_boundaries() {
        let a = q((1, 1), (1, 1), 2); // 1 + √2
        let b = q((0, 1), (1, 1), 3); // √3
                                      // Mixing √2 with √3 leaves ℚ(√n): honest decline on every operation.
        assert!(a.add(&b).is_none());
        assert!(a.sub(&b).is_none());
        assert!(a.mul(&b).is_none());
        // A rational operand adopts the other side's field.
        let two = rational(2);
        assert_eq!(two.add(&a).expect("adopts √2"), q((3, 1), (1, 1), 2));
        assert_eq!(two.mul(&a).expect("adopts √2"), q((2, 1), (2, 1), 2));
        assert_eq!(a.sub(&two).expect("adopts √2"), q((-1, 1), (1, 1), 2));
        // Same-field products fold the √n·√n = n cross term: (2√2)·(3√2) = 12.
        let prod = q((0, 1), (2, 1), 2).mul(&q((0, 1), (3, 1), 2)).expect("√2");
        assert_eq!(prod, rational(12));
        // Subtraction that cancels the surd exactly gives a canonical rational.
        let diff = a.sub(&q((0, 1), (1, 1), 2)).expect("same field");
        assert_eq!(diff, rational(1));
    }

    #[test]
    fn neg_and_conj_are_involutions() {
        let a = q((3, 7), (-2, 5), 3);
        assert_eq!(a.neg().neg(), a);
        assert_eq!(a.conj().conj(), a);
        // a + (−a) = 0 and a − a = 0.
        assert!(a.add(&a.neg()).expect("same field").is_zero());
        assert!(a.sub(&a).expect("same field").is_zero());
        // Conjugation fixes exactly the rational part: a + conj(a) = 2·rat.
        assert_eq!(a.add(&a.conj()).expect("same field"), q((6, 7), (0, 1), 0));
    }

    #[test]
    fn from_expr_bridges_as_linear_surd() {
        let mut ctx = Context::new();
        let cases = [
            ("1/2 + sqrt(5)/2", Some(q((1, 2), (1, 2), 5))),
            ("3", Some(rational(3))),
            ("2*sqrt(2)", Some(q((0, 1), (2, 1), 2))),
            ("1 - sqrt(2)", Some(q((1, 1), (-1, 1), 2))),
            // A perfect-square radicand folds through the constructor.
            ("sqrt(9)", Some(rational(3))),
            // Outside ℚ(√n): nested radical, two radicands, a variable, a transcendental.
            ("sqrt(1 + sqrt(2))", None),
            ("sqrt(2) + sqrt(3)", None),
            ("x + sqrt(2)", None),
            ("pi", None),
        ];
        for (src, expected) in cases {
            let expr = parse(src, &mut ctx).expect("parse");
            assert_eq!(QuadSurd::from_expr(&ctx, expr), expected, "{src}");
        }
    }

    #[test]
    fn to_expr_round_trips_through_from_expr() {
        let mut ctx = Context::new();
        for value in [
            q((1, 2), (1, 2), 5),  // φ
            q((1, 2), (-1, 2), 5), // ψ (negative surd coefficient)
            q((0, 1), (1, 1), 2),  // bare √2 (unit coefficient, zero rational part)
            q((-3, 4), (2, 7), 3),
            rational(42),
            rational(0),
        ] {
            let expr = value.to_expr(&mut ctx);
            assert_eq!(
                QuadSurd::from_expr(&ctx, expr),
                Some(value.clone()),
                "round-trip of {value:?}"
            );
        }
    }
}
