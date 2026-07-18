//! Complex-number rewrite helpers shared by engine rule layers.

use cas_ast::{BuiltinFn, Constant, Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{One, Signed, ToPrimitive, Zero};

/// A Gaussian rational: `a + b i` with rational coefficients.
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

    /// Exact Gaussian product: `(a+bi)(c+di) = (ac-bd) + (ad+bc)i`.
    pub fn mul(&self, other: &Self) -> Self {
        let real = &self.real * &other.real - &self.imag * &other.imag;
        let imag = &self.real * &other.imag + &self.imag * &other.real;
        Self { real, imag }
    }

    /// Exact non-negative integer power by repeated squaring (`z^0 = 1`).
    pub fn pow(&self, mut n: u64) -> Self {
        let mut result = Self::new(BigRational::one(), BigRational::zero());
        let mut base = self.clone();
        while n > 0 {
            if n & 1 == 1 {
                result = result.mul(&base);
            }
            n >>= 1;
            if n > 0 {
                base = base.mul(&base);
            }
        }
        result
    }

    /// Materialize this Gaussian number back into AST form.
    pub fn to_expr(&self, ctx: &mut Context) -> ExprId {
        let zero = BigRational::zero();
        let one = BigRational::one();
        let neg_one = -BigRational::one();

        if self.imag.is_zero() {
            return ctx.add(Expr::Number(self.real.clone()));
        }

        if self.real.is_zero() {
            if self.imag == one {
                return ctx.add(Expr::Constant(Constant::I));
            }
            if self.imag == neg_one {
                let i = ctx.add(Expr::Constant(Constant::I));
                return ctx.add(Expr::Neg(i));
            }
            let b = ctx.add(Expr::Number(self.imag.clone()));
            let i = ctx.add(Expr::Constant(Constant::I));
            return ctx.add(Expr::Mul(b, i));
        }

        let a = ctx.add(Expr::Number(self.real.clone()));
        let imag_part = if self.imag == one {
            ctx.add(Expr::Constant(Constant::I))
        } else if self.imag == neg_one {
            let i = ctx.add(Expr::Constant(Constant::I));
            ctx.add(Expr::Neg(i))
        } else if self.imag < zero {
            // Preserve `a - |b| i` shape for negative imaginary coefficients.
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

#[derive(Debug, Clone)]
pub struct ComplexRewrite {
    pub rewritten: ExprId,
    pub kind: ComplexRewriteKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComplexRewriteKind {
    ImaginaryPower,
    ISquaredMul,
    GaussianMul,
    GaussianAdd,
    GaussianDiv,
    GaussianPower,
    SqrtNegative,
    GaussianAbs,
    Conjugate,
    RealPart,
    ImagPart,
    Euler,
    PrincipalArg,
    PrincipalLog,
    GaussianSqrt,
    ComplexGeneralPower,
}

/// Extract `a + bi` from an expression when possible.
/// Match `|cos θ ± i·sin θ|` and return `θ` — the SAME node in both trig
/// arguments (hash-consing makes ExprId equality exact). Accepts either Add
/// order for the two terms and the conjugate `Sub` form; the `i·sin θ` factor
/// may carry `i` on either side of the `Mul`. The CALLER decides whether θ is
/// provably real (unimodularity is FALSE for complex θ — V0 discipline: under
/// ComplexEnabled a bare symbol may hold a complex value).
pub fn try_match_unimodular_abs(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if ctx.builtin_of(*fn_id) != Some(BuiltinFn::Abs) || args.len() != 1 {
        return None;
    }
    match_cis(ctx, args[0]).map(|(theta, _)| theta)
}

/// Match the cis form `cos θ ± i·sin θ` and return `(θ, imag_sign)` with
/// `imag_sign = +1` for `+i·sin θ` and `-1` for the conjugate. θ must be the
/// SAME node in both trig arguments (hash-consing ⇒ ExprId equality). Accepts
/// either Add order, `i` on either side of the Mul, Sub, and a Neg-wrapped
/// imaginary term. Shared by the unimodular-abs and reciprocal-cis rules.
pub fn match_cis(ctx: &Context, expr: ExprId) -> Option<(ExprId, i8)> {
    let (l, r, base_sign) = match ctx.get(expr) {
        Expr::Add(l, r) => (*l, *r, 1i8),
        Expr::Sub(l, r) => (*l, *r, -1i8),
        _ => return None,
    };

    fn cos_arg(ctx: &Context, e: ExprId) -> Option<ExprId> {
        match ctx.get(e) {
            Expr::Function(f, a) if ctx.builtin_of(*f) == Some(BuiltinFn::Cos) && a.len() == 1 => {
                Some(a[0])
            }
            _ => None,
        }
    }
    fn i_sin_arg(ctx: &Context, e: ExprId) -> Option<(ExprId, bool)> {
        // Report a Neg wrapper as a sign flip (the unimodular consumer ignores it —
        // the modulus is sign-blind — but the reciprocal-cis consumer needs it).
        let (e, negated) = match ctx.get(e) {
            Expr::Neg(inner) => (*inner, true),
            _ => (e, false),
        };
        let Expr::Mul(x, y) = ctx.get(e) else {
            return None;
        };
        let (x, y) = (*x, *y);
        let sin_of = |ctx: &Context, e: ExprId| match ctx.get(e) {
            Expr::Function(f, a) if ctx.builtin_of(*f) == Some(BuiltinFn::Sin) && a.len() == 1 => {
                Some(a[0])
            }
            _ => None,
        };
        let is_i = |ctx: &Context, e: ExprId| matches!(ctx.get(e), Expr::Constant(Constant::I));
        if is_i(ctx, x) {
            return sin_of(ctx, y).map(|t| (t, negated));
        }
        if is_i(ctx, y) {
            return sin_of(ctx, x).map(|t| (t, negated));
        }
        None
    }

    let (theta_cos, theta_sin, sign) = if let Some(tc) = cos_arg(ctx, l) {
        let (ts, neg) = i_sin_arg(ctx, r)?;
        (tc, ts, if neg { -base_sign } else { base_sign })
    } else {
        // sin-term first only occurs in the Add order (Sub keeps cos on the left).
        if base_sign < 0 {
            return None;
        }
        let (ts, neg) = i_sin_arg(ctx, l)?;
        (cos_arg(ctx, r)?, ts, if neg { -1 } else { 1 })
    };
    (theta_cos == theta_sin).then_some((theta_cos, sign))
}

pub fn extract_gaussian(ctx: &Context, expr: ExprId) -> Option<GaussianRational> {
    match ctx.get(expr) {
        Expr::Number(n) => Some(GaussianRational::new(n.clone(), BigRational::zero())),
        Expr::Constant(Constant::I) => Some(GaussianRational::new(
            BigRational::zero(),
            BigRational::one(),
        )),
        Expr::Neg(inner) => {
            if let Expr::Constant(Constant::I) = ctx.get(*inner) {
                Some(GaussianRational::new(
                    BigRational::zero(),
                    -BigRational::one(),
                ))
            } else if let Some(g) = extract_gaussian(ctx, *inner) {
                Some(GaussianRational::new(-g.real, -g.imag))
            } else {
                None
            }
        }
        Expr::Mul(l, r) => {
            if let Expr::Constant(Constant::I) = ctx.get(*r) {
                if let Expr::Number(n) = ctx.get(*l) {
                    return Some(GaussianRational::new(BigRational::zero(), n.clone()));
                }
            }
            if let Expr::Constant(Constant::I) = ctx.get(*l) {
                if let Expr::Number(n) = ctx.get(*r) {
                    return Some(GaussianRational::new(BigRational::zero(), n.clone()));
                }
            }
            None
        }
        Expr::Add(l, r) => {
            let left = extract_gaussian(ctx, *l)?;
            let right = extract_gaussian(ctx, *r)?;

            if left.is_real() && right.is_pure_imag() {
                Some(GaussianRational::new(left.real, right.imag))
            } else if left.is_pure_imag() && right.is_real() {
                Some(GaussianRational::new(right.real, left.imag))
            } else if left.is_real() && right.is_real() {
                Some(GaussianRational::new(
                    left.real + right.real,
                    BigRational::zero(),
                ))
            } else {
                Some(GaussianRational::new(
                    left.real + right.real,
                    left.imag + right.imag,
                ))
            }
        }
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

/// Rewrite `i^n` (integer n) into `{1, i, -1, -i}`.
pub fn try_rewrite_imaginary_power_expr(ctx: &mut Context, expr: ExprId) -> Option<ComplexRewrite> {
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return None;
    };
    let base = *base;
    let exp = *exp;

    if !matches!(ctx.get(base), Expr::Constant(Constant::I)) {
        return None;
    }

    let Expr::Number(n) = ctx.get(exp) else {
        return None;
    };
    if !n.is_integer() {
        return None;
    }

    let n_int = n.to_integer();
    let four = num_bigint::BigInt::from(4);
    let remainder = ((&n_int % &four) + &four) % &four;
    let normalized = remainder.to_i32().unwrap_or(0) as usize;

    let rewritten = match normalized {
        0 => ctx.num(1),
        1 => ctx.add(Expr::Constant(Constant::I)),
        2 => ctx.num(-1),
        3 => {
            let i = ctx.add(Expr::Constant(Constant::I));
            ctx.add(Expr::Neg(i))
        }
        _ => unreachable!(),
    };

    let _ = n_int;

    Some(ComplexRewrite {
        rewritten,
        kind: ComplexRewriteKind::ImaginaryPower,
    })
}

/// Rewrite `i * i` into `-1`.
pub(crate) fn try_rewrite_i_squared_mul_expr(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Mul(l, r) = ctx.get(expr) else {
        return None;
    };
    let l = *l;
    let r = *r;

    if matches!(ctx.get(l), Expr::Constant(Constant::I))
        && matches!(ctx.get(r), Expr::Constant(Constant::I))
    {
        Some(ctx.num(-1))
    } else {
        None
    }
}

/// Rewrite `i * i` into `-1` with canonical rule description.
pub fn try_rewrite_i_squared_mul_identity_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ComplexRewrite> {
    let rewritten = try_rewrite_i_squared_mul_expr(ctx, expr)?;
    Some(ComplexRewrite {
        rewritten,
        kind: ComplexRewriteKind::ISquaredMul,
    })
}

/// Rewrite `(a+bi)(c+di)` into `(ac-bd) + (ad+bc)i`.
pub fn try_rewrite_gaussian_mul_expr(ctx: &mut Context, expr: ExprId) -> Option<ComplexRewrite> {
    let Expr::Mul(l, r) = ctx.get(expr) else {
        return None;
    };
    let l = *l;
    let r = *r;

    let left = extract_gaussian(ctx, l)?;
    let right = extract_gaussian(ctx, r)?;
    if left.is_real() && right.is_real() {
        return None;
    }

    let ac = &left.real * &right.real;
    let bd = &left.imag * &right.imag;
    let ad = &left.real * &right.imag;
    let bc = &left.imag * &right.real;

    let rewritten = GaussianRational::new(ac - bd, ad + bc).to_expr(ctx);
    Some(ComplexRewrite {
        rewritten,
        kind: ComplexRewriteKind::GaussianMul,
    })
}

/// Rewrite `(a+bi) + (c+di)` into `(a+c) + (b+d)i`.
pub fn try_rewrite_gaussian_add_expr(ctx: &mut Context, expr: ExprId) -> Option<ComplexRewrite> {
    let Expr::Add(l, r) = ctx.get(expr) else {
        return None;
    };
    let l = *l;
    let r = *r;

    let left = extract_gaussian(ctx, l)?;
    let right = extract_gaussian(ctx, r)?;

    if left.is_real() && right.is_real() {
        return None;
    }

    if (left.is_real() && !right.is_real()) || (!left.is_real() && right.is_real()) {
        if left.is_real() && !left.real.is_zero() && right.is_pure_imag() {
            return None;
        }
        if right.is_real() && !right.real.is_zero() && left.is_pure_imag() {
            return None;
        }
    }

    let rewritten =
        GaussianRational::new(&left.real + &right.real, &left.imag + &right.imag).to_expr(ctx);
    Some(ComplexRewrite {
        rewritten,
        kind: ComplexRewriteKind::GaussianAdd,
    })
}

/// Rewrite `(a+bi)/(c+di)` using conjugates.
pub fn try_rewrite_gaussian_div_expr(ctx: &mut Context, expr: ExprId) -> Option<ComplexRewrite> {
    let Expr::Div(num, den) = ctx.get(expr) else {
        return None;
    };
    let num = *num;
    let den = *den;

    let numerator = extract_gaussian(ctx, num)?;
    let denominator = extract_gaussian(ctx, den)?;
    if denominator.is_real() {
        return None;
    }
    if numerator.is_real() && denominator.is_real() {
        return None;
    }

    let a = &numerator.real;
    let b = &numerator.imag;
    let c = &denominator.real;
    let d = &denominator.imag;

    let denom_sq = c * c + d * d;
    if denom_sq.is_zero() {
        return None;
    }

    let real_num = a * c + b * d;
    let imag_num = b * c - a * d;
    let rewritten =
        GaussianRational::new(&real_num / &denom_sq, &imag_num / &denom_sq).to_expr(ctx);

    Some(ComplexRewrite {
        rewritten,
        kind: ComplexRewriteKind::GaussianDiv,
    })
}

/// Rewrite `sqrt(-n)` (negative numeric literal) into `i * sqrt(n)`.
pub fn try_rewrite_sqrt_negative_expr(ctx: &mut Context, expr: ExprId) -> Option<ComplexRewrite> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if ctx.builtin_of(*fn_id) != Some(BuiltinFn::Sqrt) || args.len() != 1 {
        return None;
    }

    let arg = args[0];
    let abs_value = match ctx.get(arg) {
        Expr::Number(n) if n.is_negative() => (-n).clone(),
        Expr::Neg(inner) => {
            if let Expr::Number(n) = ctx.get(*inner) {
                if !n.is_negative() {
                    n.clone()
                } else {
                    return None;
                }
            } else {
                return None;
            }
        }
        _ => return None,
    };

    Some(ComplexRewrite {
        rewritten: negative_abs_to_i_sqrt(ctx, abs_value),
        kind: ComplexRewriteKind::SqrtNegative,
    })
}

/// Build `i·√(abs_value)` for a non-negative `abs_value`, folding a perfect-square
/// radicand to an integer coefficient (`abs_value = 4 → 2·i`, `1 → i`, `3 → i·√3`).
fn negative_abs_to_i_sqrt(ctx: &mut Context, abs_value: BigRational) -> ExprId {
    let i = ctx.add(Expr::Constant(Constant::I));
    if abs_value.is_integer() {
        let int_val = abs_value.to_integer();
        if let Some(f) = int_val.to_f64() {
            let sqrt_f = f.sqrt();
            if sqrt_f.fract() == 0.0 && sqrt_f > 0.0 {
                let sqrt_int = sqrt_f as i64;
                let sqrt_num = ctx.num(sqrt_int);
                return if sqrt_int == 1 {
                    i
                } else {
                    ctx.add(Expr::Mul(sqrt_num, i))
                };
            }
        }
    }
    let abs_num = ctx.add(Expr::Number(abs_value));
    let sqrt_abs = ctx.call_builtin(BuiltinFn::Sqrt, vec![abs_num]);
    ctx.add(Expr::Mul(i, sqrt_abs))
}

/// Rewrite the POW form `(-n)^(1/2)` (negative numeric base, exponent exactly `1/2`)
/// into `i·√n` in complex mode. The simplifier canonicalises `sqrt(-n)` to this Pow
/// form, which [`try_rewrite_sqrt_negative_expr`] (matching only the `sqrt(...)` call
/// form) misses -- so this is what actually folds `(-1)^(1/2) → i`, `(-4)^(1/2) → 2i`.
pub fn try_rewrite_negative_base_half_power_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ComplexRewrite> {
    use crate::numeric_eval::as_rational_const;
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return None;
    };
    let base = *base;
    let exp = *exp;
    // Exponent must fold to exactly 1/2 (handles `Number(1/2)` and `Div(1,2)` forms).
    let half = BigRational::new(1.into(), 2.into());
    if as_rational_const(ctx, exp)? != half {
        return None;
    }
    // Base must fold to a negative rational literal (`-1`, `-4`, `Neg(4)`, ...).
    let base_val = as_rational_const(ctx, base)?;
    if !base_val.is_negative() {
        return None;
    }
    Some(ComplexRewrite {
        rewritten: negative_abs_to_i_sqrt(ctx, -base_val),
        kind: ComplexRewriteKind::SqrtNegative,
    })
}

/// Cap on the exponent admitted by [`try_rewrite_gaussian_power_expr`]. Beyond it
/// the fold declines (honest residual): this bounds eager evaluation — coefficient
/// bit-size grows linearly with the exponent — as declared intent, not as a patch.
const MAX_GAUSSIAN_POWER_EXPONENT: u64 = 4096;

/// Rewrite `(a+bi)^n` (true Gaussian binomial base, `a≠0 ∧ b≠0`, integer `n ≥ 2`)
/// into its exact Gaussian value by repeated squaring.
///
/// Ownership guards: a real base defers to ordinary arithmetic; a pure-imaginary
/// base (`(k·i)^n`, bare `i^n`) already folds via power-of-a-product +
/// [`try_rewrite_imaginary_power_expr`] — this rule claims ONLY the binomial gap.
/// `n ∈ {0, 1}` have existing owners; a negative exponent canonicalises to
/// `1/(a+bi)^n`, whose inner power this rule folds and Gaussian division finishes.
pub fn try_rewrite_gaussian_power_expr(ctx: &mut Context, expr: ExprId) -> Option<ComplexRewrite> {
    use crate::numeric_eval::as_rational_const;
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return None;
    };
    let base = *base;
    let exp = *exp;

    let g = extract_gaussian(ctx, base)?;
    if g.is_real() || g.is_pure_imag() {
        return None;
    }

    let n = as_rational_const(ctx, exp)?;
    if !n.is_integer() {
        return None;
    }
    let n_int = n.to_integer();
    if n_int < num_bigint::BigInt::from(2)
        || n_int > num_bigint::BigInt::from(MAX_GAUSSIAN_POWER_EXPONENT)
    {
        return None;
    }
    let n_u64 = n_int.to_u64()?;

    let rewritten = g.pow(n_u64).to_expr(ctx);
    Some(ComplexRewrite {
        rewritten,
        kind: ComplexRewriteKind::GaussianPower,
    })
}

/// Exact principal argument of a CLOSED Gaussian `a+bi` — the 9-case atan2
/// sign table over `(-π, π]`, emitting rational multiples of `π` and symbolic
/// `atan(q)` forms. ZERO f64 (soundness-gates-must-be-exact): every branch
/// decision is a `BigRational` sign test. `Arg(0)` is `None` (undefined).
fn exact_principal_arg(ctx: &mut Context, g: &GaussianRational) -> Option<ExprId> {
    use num_traits::Signed as _;
    let a = &g.real;
    let b = &g.imag;
    let pi = |ctx: &mut Context| ctx.add(Expr::Constant(Constant::Pi));
    let half_pi = |ctx: &mut Context, negate: bool| {
        let p = pi(ctx);
        let two = ctx.num(2);
        let h = ctx.add(Expr::Div(p, two));
        if negate {
            ctx.add(Expr::Neg(h))
        } else {
            h
        }
    };
    let atan_q = |ctx: &mut Context, q: BigRational| {
        let qn = ctx.add(Expr::Number(q));
        ctx.call_builtin(BuiltinFn::Atan, vec![qn])
    };

    Some(match (a.is_zero(), b.is_zero()) {
        (true, true) => return None, // Arg(0): undefined
        (false, true) => {
            if a.is_positive() {
                ctx.num(0)
            } else {
                pi(ctx)
            }
        }
        (true, false) => half_pi(ctx, b.is_negative()),
        (false, false) => {
            let q = b / a;
            if a.is_positive() {
                // Right half-plane: atan(b/a) directly (atan(±1) folds to ±π/4).
                atan_q(ctx, q)
            } else if b.is_positive() {
                // Second quadrant: π + atan(b/a) (atan term is negative).
                let at = atan_q(ctx, q);
                let p = pi(ctx);
                ctx.add(Expr::Add(p, at))
            } else {
                // Third quadrant: atan(b/a) − π.
                let at = atan_q(ctx, q);
                let p = pi(ctx);
                ctx.add(Expr::Sub(at, p))
            }
        }
    })
}

/// Rewrite `arg(a+bi)` (closed Gaussian) into its EXACT principal value.
/// `arg(0)` rewrites to `Undefined` explicitly; symbolic args decline.
pub fn try_rewrite_arg_expr(ctx: &mut Context, expr: ExprId) -> Option<ComplexRewrite> {
    let g = gaussian_unary_builtin_arg(ctx, expr, BuiltinFn::Arg)?;
    let rewritten = match exact_principal_arg(ctx, &g) {
        Some(id) => id,
        None => ctx.add(Expr::Constant(Constant::Undefined)),
    };
    Some(ComplexRewrite {
        rewritten,
        kind: ComplexRewriteKind::PrincipalArg,
    })
}

/// Principal logarithm of a CLOSED Gaussian: `ln(z) = ln|z| + i·Arg(z)`.
/// DECLINES a positive rational argument (the real `EvaluateLogRule` owns
/// `ln(2)` — its narration and footprint must not change) and `z = 0`.
/// `ln(-c)` (negative real) yields `ln(c) + i·π`; `ln(i)` yields `i·π/2`.
pub fn try_rewrite_complex_ln_expr(ctx: &mut Context, expr: ExprId) -> Option<ComplexRewrite> {
    use num_traits::Signed as _;
    let g = gaussian_unary_builtin_arg(ctx, expr, BuiltinFn::Ln)?;
    if g.real.is_zero() && g.imag.is_zero() {
        return None; // ln(0): the real machinery's undefined verdict stands.
    }
    if g.imag.is_zero() && g.real.is_positive() {
        return None; // positive real: owned by the real log machinery.
    }

    // ln|z|: |z|² = a²+b² rational; emit ln(√(a²+b²)) = ln(a²+b²)/2, folding
    // the perfect-square case to ln(rational). For b=0 the modulus is |a|.
    let modulus_ln = if g.imag.is_zero() {
        let abs_a = ctx.add(Expr::Number(g.real.abs()));
        Some(ctx.call_builtin(BuiltinFn::Ln, vec![abs_a]))
    } else {
        let norm = &g.real * &g.real + &g.imag * &g.imag;
        if let Some(root) = crate::perfect_square_support::rational_sqrt(&norm) {
            if root.is_one() {
                None // |z| = 1: the ln|z| term vanishes (ln(i) → i·π/2).
            } else {
                let r = ctx.add(Expr::Number(root));
                Some(ctx.call_builtin(BuiltinFn::Ln, vec![r]))
            }
        } else {
            let n = ctx.add(Expr::Number(norm));
            let ln_norm = ctx.call_builtin(BuiltinFn::Ln, vec![n]);
            let two = ctx.num(2);
            Some(ctx.add(Expr::Div(ln_norm, two)))
        }
    };

    let arg = exact_principal_arg(ctx, &g)?;
    let i = ctx.add(Expr::Constant(Constant::I));
    let i_arg = ctx.add(Expr::Mul(i, arg));
    let rewritten = match modulus_ln {
        Some(m) => ctx.add(Expr::Add(m, i_arg)),
        None => i_arg,
    };
    Some(ComplexRewrite {
        rewritten,
        kind: ComplexRewriteKind::PrincipalLog,
    })
}

/// Pure-imaginary view of an expression: `Some(θ)` iff `expr = i·θ` exactly
/// (no real part), with `θ` i-free. Built on `split_i_factor`. Used by the
/// trig↔hyperbolic bridge (`sin(i·y) = i·sinh(y)` and sisters — ENTIRE-function
/// identities, valid for arbitrary complex `y`, so symbolic arguments are fine).
pub fn split_pure_imaginary(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    match split_i_factor(ctx, expr)? {
        (None, Some(theta)) => Some(theta),
        _ => None,
    }
}

/// Trig/hyperbolic of a PURE-IMAGINARY argument: the entire-function bridge.
/// `sin(iy) = i·sinh(y)`, `cos(iy) = cosh(y)`, `tan(iy) = i·tanh(y)`,
/// `sinh(iy) = i·sin(y)`, `cosh(iy) = cos(y)`, `tanh(iy) = i·tan(y)`.
/// ONE-DIRECTION (imaginary argument → i-free argument); no inverse rule may
/// exist (ping-pong). Returns the rewritten node plus a Spanish description.
pub fn try_rewrite_trig_of_imaginary(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, &'static str)> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }
    let builtin = ctx.builtin_of(*fn_id)?;
    let arg = args[0];
    let theta = split_pure_imaginary(ctx, arg)?;
    let i = ctx.add(Expr::Constant(Constant::I));
    let out = match builtin {
        BuiltinFn::Sin => {
            let h = ctx.call_builtin(BuiltinFn::Sinh, vec![theta]);
            (
                ctx.add(Expr::Mul(i, h)),
                "seno de argumento imaginario: sin(i·y) = i·senh(y)",
            )
        }
        BuiltinFn::Cos => (
            ctx.call_builtin(BuiltinFn::Cosh, vec![theta]),
            "coseno de argumento imaginario: cos(i·y) = cosh(y)",
        ),
        BuiltinFn::Tan => {
            let h = ctx.call_builtin(BuiltinFn::Tanh, vec![theta]);
            (
                ctx.add(Expr::Mul(i, h)),
                "tangente de argumento imaginario: tan(i·y) = i·tanh(y)",
            )
        }
        BuiltinFn::Sinh => {
            let t = ctx.call_builtin(BuiltinFn::Sin, vec![theta]);
            (
                ctx.add(Expr::Mul(i, t)),
                "seno hiperbólico de argumento imaginario: senh(i·y) = i·sen(y)",
            )
        }
        BuiltinFn::Cosh => (
            ctx.call_builtin(BuiltinFn::Cos, vec![theta]),
            "coseno hiperbólico de argumento imaginario: cosh(i·y) = cos(y)",
        ),
        BuiltinFn::Tanh => {
            let t = ctx.call_builtin(BuiltinFn::Tan, vec![theta]);
            (
                ctx.add(Expr::Mul(i, t)),
                "tangente hiperbólica de argumento imaginario: tanh(i·y) = i·tan(y)",
            )
        }
        _ => return None,
    };
    Some(out)
}

/// Trig/hyperbolic of a MIXED complex argument `re + i·θ` (both parts nonzero):
/// the entire angle-sum bridge, composing the real-argument factor with the
/// imaginary-argument one. Valid for ARBITRARY complex `re`/`θ` (sin(z+w) is an
/// entire identity, and cos(iθ)=cosh(θ) holds on all of ℂ) — no realness guard.
/// `sin(re+iθ) = sin(re)·cosh(θ) + i·cos(re)·sinh(θ)`
/// `cos(re+iθ) = cos(re)·cosh(θ) − i·sin(re)·sinh(θ)`
/// `sinh(re+iθ) = sinh(re)·cos(θ) + i·cosh(re)·sin(θ)`
/// `cosh(re+iθ) = cosh(re)·cos(θ) + i·sinh(re)·sin(θ)`
/// `tan`/`tanh` decline (the quotient form is an honest residual). ONE-DIRECTION.
pub fn try_rewrite_trig_complex_angle_sum(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, &'static str)> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }
    let builtin = ctx.builtin_of(*fn_id)?;
    if !matches!(
        builtin,
        BuiltinFn::Sin | BuiltinFn::Cos | BuiltinFn::Sinh | BuiltinFn::Cosh
    ) {
        return None;
    }
    let arg = args[0];
    let (re, theta) = match split_i_factor(ctx, arg)? {
        (Some(re), Some(theta)) => (re, theta),
        // Pure-imaginary is TrigOfImaginaryRule's; i-free is not ours at all.
        _ => return None,
    };
    let i = ctx.add(Expr::Constant(Constant::I));
    let build = |ctx: &mut Context,
                 f_re: BuiltinFn,
                 g_re: BuiltinFn,
                 f_th: BuiltinFn,
                 g_th: BuiltinFn,
                 subtract: bool| {
        let a = ctx.call_builtin(f_re, vec![re]);
        let b = ctx.call_builtin(f_th, vec![theta]);
        let first = ctx.add(Expr::Mul(a, b));
        let c = ctx.call_builtin(g_re, vec![re]);
        let d = ctx.call_builtin(g_th, vec![theta]);
        let cd = ctx.add(Expr::Mul(c, d));
        let second = ctx.add(Expr::Mul(i, cd));
        if subtract {
            ctx.add(Expr::Sub(first, second))
        } else {
            ctx.add(Expr::Add(first, second))
        }
    };
    let out = match builtin {
        BuiltinFn::Sin => (
            build(ctx, BuiltinFn::Sin, BuiltinFn::Cos, BuiltinFn::Cosh, BuiltinFn::Sinh, false),
            "seno de argumento complejo: sin(a+ib) = sin(a)·cosh(b) + i·cos(a)·senh(b)",
        ),
        BuiltinFn::Cos => (
            build(ctx, BuiltinFn::Cos, BuiltinFn::Sin, BuiltinFn::Cosh, BuiltinFn::Sinh, true),
            "coseno de argumento complejo: cos(a+ib) = cos(a)·cosh(b) - i·sen(a)·senh(b)",
        ),
        BuiltinFn::Sinh => (
            build(ctx, BuiltinFn::Sinh, BuiltinFn::Cosh, BuiltinFn::Cos, BuiltinFn::Sin, false),
            "seno hiperbólico de argumento complejo: senh(a+ib) = senh(a)·cos(b) + i·cosh(a)·sen(b)",
        ),
        BuiltinFn::Cosh => (
            build(ctx, BuiltinFn::Cosh, BuiltinFn::Sinh, BuiltinFn::Cos, BuiltinFn::Sin, false),
            "coseno hiperbólico de argumento complejo: cosh(a+ib) = cosh(a)·cos(b) + i·senh(a)·sen(b)",
        ),
        _ => return None,
    };
    Some(out)
}

/// Modulus of a Gaussian with DECIDABLE-REAL surd/transcendental components:
/// `|a + b·i| = √(a² + b²)` when BOTH parts prove decidable-real via
/// `provable_const_sign` (rationals, surds, e/π combos) and at least one is NOT
/// a plain rational (that case already belongs to the exact `GaussianRational`
/// modulus). The emitted squares fold downstream by the exact surd arithmetic
/// (`(√3/2)² → 3/4`), closing the π-rational family (`|1/2 + i·√3/2| → 1`).
/// Symbolic components decline (V0 discipline: a variable may hold a complex
/// value, so `a²+b²` would be the wrong formula).
pub fn try_rewrite_gaussian_surd_abs(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, &'static str)> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if ctx.builtin_of(*fn_id) != Some(BuiltinFn::Abs) || args.len() != 1 {
        return None;
    }
    let arg = args[0];
    let (re, im) = match split_i_factor(ctx, arg)? {
        (Some(re), Some(im)) => (Some(re), im),
        (None, Some(im)) => (None, im),
        // No imaginary part: not a Gaussian modulus — never ours.
        _ => return None,
    };
    let im_sign = crate::const_sign::provable_const_sign(ctx, im)?;
    if let Some(re) = re {
        crate::const_sign::provable_const_sign(ctx, re)?;
    }
    // Both parts plain rationals ⇒ the exact GaussianRational modulus owns it.
    // (`|2·i| → 2·|i|` with the nested `|i|` unfolded is a PRE-EXISTING residual of
    // the multiplicative abs split, bisect-verified — its owner is that split, not
    // this rule: widening here was dead surface because the split preempts.)
    let re_rational = re.is_none_or(|r| crate::numeric_eval::as_rational_const(ctx, r).is_some());
    let im_rational = crate::numeric_eval::as_rational_const(ctx, im).is_some();
    if re_rational && im_rational {
        return None;
    }
    let Some(re) = re else {
        // Pure imaginary: |i·b| = |b|, and the sign is already DECIDED — emit ±b
        // directly (√(b²) would sit unfolded, the lone-radicand quirk).
        let out = match im_sign {
            crate::const_sign::ConstSign::Positive => im,
            crate::const_sign::ConstSign::Negative => ctx.add(Expr::Neg(im)),
            crate::const_sign::ConstSign::Zero => ctx.num(0),
        };
        return Some((
            out,
            "módulo del imaginario puro con componente real decidible: |b·i| = |b|",
        ));
    };
    let two = ctx.num(2);
    let im_sq = ctx.add(Expr::Pow(im, two));
    let re_sq = ctx.add(Expr::Pow(re, two));
    let sum = ctx.add(Expr::Add(re_sq, im_sq));
    Some((
        ctx.call_builtin(BuiltinFn::Sqrt, vec![sum]),
        "módulo del gaussiano con componentes reales decidibles: |a+b·i| = √(a²+b²)",
    ))
}

/// Reciprocal of a cis form: `n / (cos u ± i·sin u) → n·(cos u ∓ i·sin u)` — an
/// ENTIRE identity for arbitrary complex `u` ((cos z + i·sin z)(cos z − i·sin z)
/// = cos²z + sin²z = 1 identically, the Pythagorean identity is entire), so no
/// realness guard. Closes the `e^(-i·x)` residual: the negative-exponent
/// canonicalization turns it into `1/e^(ix)` BEFORE Euler fires, and Euler then
/// expands only the denominator. ONE-DIRECTION.
pub fn try_rewrite_reciprocal_cis(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, &'static str)> {
    let (numer, denom) = match ctx.get(expr) {
        Expr::Div(n, d) => (*n, *d),
        _ => return None,
    };
    let (theta, sign) = match_cis(ctx, denom)?;
    let i = ctx.add(Expr::Constant(Constant::I));
    let cos_t = ctx.call_builtin(BuiltinFn::Cos, vec![theta]);
    let sin_t = ctx.call_builtin(BuiltinFn::Sin, vec![theta]);
    let i_sin = ctx.add(Expr::Mul(i, sin_t));
    let conj = if sign > 0 {
        ctx.add(Expr::Sub(cos_t, i_sin))
    } else {
        ctx.add(Expr::Add(cos_t, i_sin))
    };
    let out = if crate::expr_predicates::is_one_expr(ctx, numer) {
        conj
    } else {
        ctx.add(Expr::Mul(numer, conj))
    };
    Some((
        out,
        "recíproco de cis: 1/(cos u ± i·sen u) = cos u ∓ i·sen u (módulo cis·cis̄ = 1, identidad entera)",
    ))
}

/// STRUCTURAL split of an exponent into `real_part + i·theta` — exact, no
/// folding. Returns `(real_part, theta)` where `None` means "zero part";
/// both components are guaranteed i-free. Declines (`None` overall) on any
/// shape that is not a clean single-`i` split: `i` inside a function call,
/// `i` in both factors of a product (`i·i·x`), `i` in a denominator, `i`
/// under a `Pow`, etc. This is the piece `extract_gaussian` cannot do: it
/// requires NUMERIC coefficients next to `i`, so `i·π` (Mul(Constant,I)),
/// `π·i/2` (Div(Mul(Pi,I),2)) and `i·x` all need this symbolic splitter.
#[allow(clippy::type_complexity)]
fn split_i_factor(ctx: &mut Context, expr: ExprId) -> Option<(Option<ExprId>, Option<ExprId>)> {
    use crate::numeric_eval::contains_i;

    if !contains_i(ctx, expr) {
        return Some((Some(expr), None));
    }
    match ctx.get(expr) {
        Expr::Constant(Constant::I) => {
            let one = ctx.num(1);
            Some((None, Some(one)))
        }
        Expr::Neg(inner) => {
            let inner = *inner;
            let (re, im) = split_i_factor(ctx, inner)?;
            let neg = |ctx: &mut Context, part: Option<ExprId>| part.map(|p| ctx.add(Expr::Neg(p)));
            Some((neg(ctx, re), neg(ctx, im)))
        }
        Expr::Mul(l, r) => {
            let (l, r) = (*l, *r);
            let (with_i, pure) = match (contains_i(ctx, l), contains_i(ctx, r)) {
                (true, false) => (l, r),
                (false, true) => (r, l),
                // `i` in both factors (i·i·x, (1+i)(…)) is not a clean split.
                _ => return None,
            };
            let (re, im) = split_i_factor(ctx, with_i)?;
            let scale =
                |ctx: &mut Context, part: Option<ExprId>| part.map(|p| ctx.add(Expr::Mul(p, pure)));
            Some((scale(ctx, re), scale(ctx, im)))
        }
        Expr::Div(num, den) => {
            let (num, den) = (*num, *den);
            if contains_i(ctx, den) {
                return None;
            }
            let (re, im) = split_i_factor(ctx, num)?;
            let scale =
                |ctx: &mut Context, part: Option<ExprId>| part.map(|p| ctx.add(Expr::Div(p, den)));
            Some((scale(ctx, re), scale(ctx, im)))
        }
        Expr::Add(l, r) => {
            let (l, r) = (*l, *r);
            let (re_l, im_l) = split_i_factor(ctx, l)?;
            let (re_r, im_r) = split_i_factor(ctx, r)?;
            let join = |ctx: &mut Context, a: Option<ExprId>, b: Option<ExprId>| match (a, b) {
                (Some(a), Some(b)) => Some(ctx.add(Expr::Add(a, b))),
                (Some(a), None) => Some(a),
                (None, b) => b,
            };
            Some((join(ctx, re_l, re_r), join(ctx, im_l, im_r)))
        }
        Expr::Sub(l, r) => {
            let (l, r) = (*l, *r);
            let (re_l, im_l) = split_i_factor(ctx, l)?;
            let (re_r, im_r) = split_i_factor(ctx, r)?;
            let join = |ctx: &mut Context, a: Option<ExprId>, b: Option<ExprId>| match (a, b) {
                (Some(a), Some(b)) => Some(ctx.add(Expr::Sub(a, b))),
                (Some(a), None) => Some(a),
                (None, Some(b)) => Some(ctx.add(Expr::Neg(b))),
                (None, None) => None,
            };
            Some((join(ctx, re_l, re_r), join(ctx, im_l, im_r)))
        }
        // `i` under Pow / inside a function call: not a clean linear split.
        _ => None,
    }
}

/// Euler's formula: rewrite `e^(a + i·θ)` (Pow(E,·) — the parser desugars
/// `exp(x)` to this form at parse time — plus a defensive Function(exp,·)
/// arm) into `e^a · (cos θ + i·sin θ)`. Pure-imaginary exponents give the
/// classic `cos θ + i·sin θ`. ONE-DIRECTION by convention: there is
/// deliberately no inverse `cos θ + i·sin θ → e^(iθ)` rule (ping-pong).
pub fn try_rewrite_euler_expr(ctx: &mut Context, expr: ExprId) -> Option<ComplexRewrite> {
    let exponent = match ctx.get(expr) {
        Expr::Pow(base, exp) if matches!(ctx.get(*base), Expr::Constant(Constant::E)) => *exp,
        Expr::Function(fn_id, args)
            if ctx.builtin_of(*fn_id) == Some(BuiltinFn::Exp) && args.len() == 1 =>
        {
            args[0]
        }
        _ => return None,
    };

    let (re_part, theta) = split_i_factor(ctx, exponent)?;
    // Euler only fires on a genuine i-component; theta is i-free by
    // construction of the splitter (defense in depth: decline otherwise).
    let theta = theta?;
    if crate::numeric_eval::contains_i(ctx, theta)
        || re_part.is_some_and(|re| crate::numeric_eval::contains_i(ctx, re))
    {
        return None;
    }

    let cos = ctx.call_builtin(BuiltinFn::Cos, vec![theta]);
    let sin = ctx.call_builtin(BuiltinFn::Sin, vec![theta]);
    let i = ctx.add(Expr::Constant(Constant::I));
    let i_sin = ctx.add(Expr::Mul(i, sin));
    let trig = ctx.add(Expr::Add(cos, i_sin));
    let rewritten = match re_part {
        None => trig,
        Some(re) => {
            let e = ctx.add(Expr::Constant(Constant::E));
            let e_re = ctx.add(Expr::Pow(e, re));
            ctx.add(Expr::Mul(e_re, trig))
        }
    };
    Some(ComplexRewrite {
        rewritten,
        kind: ComplexRewriteKind::Euler,
    })
}

/// `sqrt(a+bi)` (both the `sqrt(...)` call and the `Pow(z, 1/2)` form) with
/// `b != 0` and `a²+b²` a perfect rational square folds to the EXACT
/// principal square root
///   `sqrt((|z|+a)/2) + i·sign(b)·sqrt((|z|-a)/2)`
/// (`Re ≥ 0` — the principal-branch choice). Pure-real radicands decline:
/// the real machinery and `SqrtNegative` keep their ownership. Non-perfect
/// squares decline to the polar route (`z^w`) or stay symbolic.
pub fn try_rewrite_gaussian_sqrt_expr(ctx: &mut Context, expr: ExprId) -> Option<ComplexRewrite> {
    use num_traits::Signed as _;
    let radicand = match ctx.get(expr) {
        Expr::Pow(base, exp) => {
            let base = *base;
            let exp = *exp;
            let half = BigRational::new(1.into(), 2.into());
            if crate::numeric_eval::as_rational_const(ctx, exp)? != half {
                return None;
            }
            base
        }
        Expr::Function(fn_id, args)
            if ctx.builtin_of(*fn_id) == Some(BuiltinFn::Sqrt) && args.len() == 1 =>
        {
            args[0]
        }
        _ => return None,
    };
    let g = extract_gaussian(ctx, radicand)?;
    if g.imag.is_zero() {
        return None;
    }
    let norm = &g.real * &g.real + &g.imag * &g.imag;
    let modulus = crate::perfect_square_support::rational_sqrt(&norm)?;
    let two = BigRational::from_integer(2.into());
    let re_sq = (&modulus + &g.real) / &two;
    let im_sq = (&modulus - &g.real) / &two;
    let re_part = sqrt_of_nonnegative_rational(ctx, re_sq);
    let im_part = sqrt_of_nonnegative_rational(ctx, im_sq);
    let i = ctx.add(Expr::Constant(Constant::I));
    let i_im = ctx.add(Expr::Mul(i, im_part));
    let rewritten = if g.imag.is_positive() {
        ctx.add(Expr::Add(re_part, i_im))
    } else {
        ctx.add(Expr::Sub(re_part, i_im))
    };
    Some(ComplexRewrite {
        rewritten,
        kind: ComplexRewriteKind::GaussianSqrt,
    })
}

/// `sqrt(q)` of a non-negative rational: exact `Number` when `q` is a
/// perfect square, else the canonical `q^(1/2)` power form.
fn sqrt_of_nonnegative_rational(ctx: &mut Context, q: BigRational) -> ExprId {
    if let Some(root) = crate::perfect_square_support::rational_sqrt(&q) {
        return ctx.add(Expr::Number(root));
    }
    let q_expr = ctx.add(Expr::Number(q));
    let half = ctx.add(Expr::Number(BigRational::new(1.into(), 2.into())));
    ctx.add(Expr::Pow(q_expr, half))
}

/// `z^w = e^(w·ln z)` (principal branch) for CLOSED Gaussian base and
/// exponent. Fires when the exponent has a genuine imaginary part
/// (`i^i`, `2^i`) or when a non-real base meets a non-integer rational
/// exponent (`(1+i)^(1/3)` — honest polar form). Everything with an
/// existing owner declines:
///   - base `e` (EulerRule owns `Pow(E, ·)` — this decline is also the
///     anti-churn guard: the rule's OWN output is `Pow(E, ·)`),
///   - integer exponents (GaussianPowRule / ordinary arithmetic),
///   - real rational base with real rational exponent (real machinery,
///     ComplexNegativeBaseRoot, SqrtNegative),
///   - `z^(1/2)` with perfect-square norm (GaussianSqrtRule, exact > polar),
///   - symbolic base or exponent (honest residual).
pub fn try_rewrite_complex_general_power_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ComplexRewrite> {
    use num_traits::Signed as _;
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return None;
    };
    let base = *base;
    let exp = *exp;
    if crate::expr_predicates::is_e_constant_expr(ctx, base) {
        return None;
    }
    let g_base = extract_gaussian(ctx, base)?;
    if g_base.real.is_zero() && g_base.imag.is_zero() {
        return None; // 0^w: ln(0) has no principal value.
    }
    // The exponent may reach us in raw `Div(1,3)` form (the pipeline
    // canonicalizes to `Number(1/3)`, but the helper must not depend on
    // that): fall back to the rational-const folder for real exponents.
    let g_exp = extract_gaussian(ctx, exp).or_else(|| {
        crate::numeric_eval::as_rational_const(ctx, exp)
            .map(|c| GaussianRational::new(c, BigRational::zero()))
    })?;
    if g_exp.imag.is_zero() {
        let c = &g_exp.real;
        if c.is_integer() {
            return None;
        }
        if g_base.imag.is_zero() {
            return None;
        }
        let half = BigRational::new(1.into(), 2.into());
        if *c == half {
            let norm = &g_base.real * &g_base.real + &g_base.imag * &g_base.imag;
            if crate::perfect_square_support::rational_sqrt(&norm).is_some() {
                return None;
            }
        }
    }
    // Positive real base with a genuine i-exponent: emit the Euler form
    // DIRECTLY — z^(c+di) = z^c·(cos(d·ln z) + i·sin(d·ln z)). Routing it
    // through e^(w·ln z) would ping-pong with the real exp-log
    // canonicalizer (e^(a·ln b) -> b^a) whenever ln z stays symbolic
    // (2^i: ln 2 has no closed form for ComplexLogRule to consume).
    if g_base.imag.is_zero() && g_base.real.is_positive() && !g_exp.imag.is_zero() {
        let ln_z = ctx.call_builtin(BuiltinFn::Ln, vec![base]);
        let d = ctx.add(Expr::Number(g_exp.imag.clone()));
        let theta = ctx.add(Expr::Mul(d, ln_z));
        let cos = ctx.call_builtin(BuiltinFn::Cos, vec![theta]);
        let sin = ctx.call_builtin(BuiltinFn::Sin, vec![theta]);
        let i = ctx.add(Expr::Constant(Constant::I));
        let i_sin = ctx.add(Expr::Mul(i, sin));
        let trig = ctx.add(Expr::Add(cos, i_sin));
        let rewritten = if g_exp.real.is_zero() {
            trig
        } else {
            let c = ctx.add(Expr::Number(g_exp.real.clone()));
            let z_c = ctx.add(Expr::Pow(base, c));
            ctx.add(Expr::Mul(z_c, trig))
        };
        return Some(ComplexRewrite {
            rewritten,
            kind: ComplexRewriteKind::ComplexGeneralPower,
        });
    }

    let ln_z = ctx.call_builtin(BuiltinFn::Ln, vec![base]);
    let w_ln = ctx.add(Expr::Mul(exp, ln_z));
    let e = ctx.add(Expr::Constant(Constant::E));
    let rewritten = ctx.add(Expr::Pow(e, w_ln));
    Some(ComplexRewrite {
        rewritten,
        kind: ComplexRewriteKind::ComplexGeneralPower,
    })
}

/// Extract the single Gaussian argument of `Function(builtin, [arg])`.
/// Declines (None) when the shape or the builtin does not match, or when the
/// argument is not a closed Gaussian number (symbolic args stay symbolic —
/// honest residual, never a fabricated value).
fn gaussian_unary_builtin_arg(
    ctx: &Context,
    expr: ExprId,
    builtin: BuiltinFn,
) -> Option<GaussianRational> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if ctx.builtin_of(*fn_id) != Some(builtin) || args.len() != 1 {
        return None;
    }
    extract_gaussian(ctx, args[0])
}

/// Rewrite `abs(a+bi)` into the exact modulus `√(a²+b²)`, folding a
/// perfect-square radicand to a rational (`|3+4i| → 5`, `|i| → 1`,
/// `|1+i| → √2`). Claims only args with a NON-ZERO imaginary part: real
/// arguments keep their existing real-`abs` owner.
pub fn try_rewrite_gaussian_abs_expr(ctx: &mut Context, expr: ExprId) -> Option<ComplexRewrite> {
    let g = gaussian_unary_builtin_arg(ctx, expr, BuiltinFn::Abs)?;
    if g.is_real() {
        return None;
    }
    let norm = &g.real * &g.real + &g.imag * &g.imag;
    let rewritten = if let Some(root) = crate::perfect_square_support::rational_sqrt(&norm) {
        ctx.add(Expr::Number(root))
    } else {
        let radicand = ctx.add(Expr::Number(norm));
        ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand])
    };
    Some(ComplexRewrite {
        rewritten,
        kind: ComplexRewriteKind::GaussianAbs,
    })
}

/// Rewrite `conjugate(a+bi)` into `a-bi` (real args are their own conjugate).
pub fn try_rewrite_conjugate_expr(ctx: &mut Context, expr: ExprId) -> Option<ComplexRewrite> {
    let g = gaussian_unary_builtin_arg(ctx, expr, BuiltinFn::Conjugate)?;
    let rewritten = GaussianRational::new(g.real, -g.imag).to_expr(ctx);
    Some(ComplexRewrite {
        rewritten,
        kind: ComplexRewriteKind::Conjugate,
    })
}

/// Rewrite `Re(a+bi)` into `a`.
pub fn try_rewrite_re_expr(ctx: &mut Context, expr: ExprId) -> Option<ComplexRewrite> {
    let g = gaussian_unary_builtin_arg(ctx, expr, BuiltinFn::Re)?;
    let rewritten = ctx.add(Expr::Number(g.real));
    Some(ComplexRewrite {
        rewritten,
        kind: ComplexRewriteKind::RealPart,
    })
}

/// Rewrite `Im(a+bi)` into `b`.
pub fn try_rewrite_im_expr(ctx: &mut Context, expr: ExprId) -> Option<ComplexRewrite> {
    let g = gaussian_unary_builtin_arg(ctx, expr, BuiltinFn::Im)?;
    let rewritten = ctx.add(Expr::Number(g.imag));
    Some(ComplexRewrite {
        rewritten,
        kind: ComplexRewriteKind::ImagPart,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        extract_gaussian, try_rewrite_arg_expr, try_rewrite_complex_general_power_expr,
        try_rewrite_complex_ln_expr, try_rewrite_conjugate_expr, try_rewrite_euler_expr,
        try_rewrite_gaussian_abs_expr, try_rewrite_gaussian_add_expr,
        try_rewrite_gaussian_div_expr, try_rewrite_gaussian_mul_expr,
        try_rewrite_gaussian_power_expr, try_rewrite_gaussian_sqrt_expr,
        try_rewrite_i_squared_mul_identity_expr, try_rewrite_im_expr,
        try_rewrite_imaginary_power_expr, try_rewrite_re_expr, try_rewrite_sqrt_negative_expr,
        ComplexRewriteKind, GaussianRational,
    };
    use cas_ast::{Constant, Context, Expr};
    use cas_formatter::DisplayExpr;
    use num_rational::BigRational;
    use num_traits::{One, Zero};

    #[test]
    fn extracts_gaussian_i() {
        let mut ctx = Context::new();
        let i = ctx.add(Expr::Constant(Constant::I));
        let g = extract_gaussian(&ctx, i).expect("gaussian");
        assert!(g.real.is_zero());
        assert!(g.imag.is_one());
    }

    #[test]
    fn imaginary_power_mod4() {
        let mut ctx = Context::new();
        let i = ctx.add(Expr::Constant(Constant::I));
        let exp = ctx.num(17);
        let expr = ctx.add(Expr::Pow(i, exp));
        let rewrite = try_rewrite_imaginary_power_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(rewrite.kind, ComplexRewriteKind::ImaginaryPower);
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.rewritten
                }
            ),
            "i"
        );
    }

    #[test]
    fn rewrites_gaussian_mul() {
        let mut ctx = Context::new();
        let left = cas_parser::parse("1 + 2*i", &mut ctx).expect("parse");
        let right = cas_parser::parse("3 + 4*i", &mut ctx).expect("parse");
        let expr = ctx.add(Expr::Mul(left, right));
        let rewrite = try_rewrite_gaussian_mul_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(rewrite.kind, ComplexRewriteKind::GaussianMul);
        let g = extract_gaussian(&ctx, rewrite.rewritten).expect("gaussian");
        assert_eq!(g.real, BigRational::from_integer((-5).into()));
        assert_eq!(g.imag, BigRational::from_integer(10.into()));
    }

    #[test]
    fn rewrites_gaussian_add_after_real_cancellation_to_pure_imag() {
        let mut ctx = Context::new();
        let expr = cas_parser::parse("(1 + 2*i) + (-1 + 3*i)", &mut ctx).expect("parse");
        let rewrite = try_rewrite_gaussian_add_expr(&mut ctx, expr).expect("rewrite");
        let shown = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.rewritten
            }
        );
        assert_eq!(shown, "5 * i");
    }

    #[test]
    fn rewrites_gaussian_div() {
        let mut ctx = Context::new();
        let expr = cas_parser::parse("(1 + i) / (1 - i)", &mut ctx).expect("parse");
        let rewrite = try_rewrite_gaussian_div_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(rewrite.kind, ComplexRewriteKind::GaussianDiv);
        let shown = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.rewritten
            }
        );
        assert!(shown.contains("i"));
    }

    #[test]
    fn rewrites_sqrt_negative() {
        let mut ctx = Context::new();
        let expr = cas_parser::parse("sqrt(-4)", &mut ctx).expect("parse");
        let rewrite = try_rewrite_sqrt_negative_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(rewrite.kind, ComplexRewriteKind::SqrtNegative);
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.rewritten
                }
            ),
            "2 * i"
        );
    }

    #[test]
    fn gaussian_to_expr_roundtrip() {
        let mut ctx = Context::new();
        let g = GaussianRational::new(
            BigRational::from_integer(3.into()),
            BigRational::from_integer(2.into()),
        );
        let expr = g.to_expr(&mut ctx);
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: expr
                }
            ),
            "3 + 2 * i"
        );
    }

    #[test]
    fn rewrites_i_squared_identity_with_desc() {
        let mut ctx = Context::new();
        let i1 = ctx.add(Expr::Constant(Constant::I));
        let i2 = ctx.add(Expr::Constant(Constant::I));
        let expr = ctx.add(Expr::Mul(i1, i2));
        let rewrite = try_rewrite_i_squared_mul_identity_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(rewrite.kind, ComplexRewriteKind::ISquaredMul);
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.rewritten
                }
            ),
            "-1"
        );
    }

    fn gauss(re: i64, im: i64) -> GaussianRational {
        GaussianRational::new(
            BigRational::from_integer(re.into()),
            BigRational::from_integer(im.into()),
        )
    }

    #[test]
    fn gaussian_mul_method_exact() {
        // (1+2i)(3+4i) = 3-8 + (4+6)i = -5+10i
        let p = gauss(1, 2).mul(&gauss(3, 4));
        assert_eq!(p.real, BigRational::from_integer((-5).into()));
        assert_eq!(p.imag, BigRational::from_integer(10.into()));
    }

    #[test]
    fn gaussian_pow_method_exact() {
        // (1+i)^2 = 2i ; (1+i)^4 = -4 ; (2+i)^4 = -7+24i ; z^0 = 1
        let z = gauss(1, 1);
        let sq = z.pow(2);
        assert!(sq.real.is_zero());
        assert_eq!(sq.imag, BigRational::from_integer(2.into()));
        let fourth = z.pow(4);
        assert_eq!(fourth.real, BigRational::from_integer((-4).into()));
        assert!(fourth.imag.is_zero());
        let w = gauss(2, 1).pow(4);
        assert_eq!(w.real, BigRational::from_integer((-7).into()));
        assert_eq!(w.imag, BigRational::from_integer(24.into()));
        let unit = gauss(5, -3).pow(0);
        assert!(unit.real.is_one());
        assert!(unit.imag.is_zero());
    }

    #[test]
    fn rewrites_gaussian_power_binomial() {
        let mut ctx = Context::new();
        let expr = cas_parser::parse("(1 + i)^2", &mut ctx).expect("parse");
        let rewrite = try_rewrite_gaussian_power_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(rewrite.kind, ComplexRewriteKind::GaussianPower);
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.rewritten
                }
            ),
            "2 * i"
        );
    }

    #[test]
    fn rewrites_gaussian_power_rational_coefficients() {
        // (1/2 + i)^2 = 1/4 - 1 + i = -3/4 + i. Built with Number(1/2) directly:
        // raw-parse keeps `Div(1,2)`, which extract_gaussian deliberately does not
        // capture (widening the collector would change all 7 sibling rules'
        // admission); the pipeline's rational canonicalization folds it upstream.
        let mut ctx = Context::new();
        let half = ctx.add(Expr::Number(BigRational::new(1.into(), 2.into())));
        let i = ctx.add(Expr::Constant(Constant::I));
        let base = ctx.add(Expr::Add(half, i));
        let two = ctx.num(2);
        let expr = ctx.add(Expr::Pow(base, two));
        let rewrite = try_rewrite_gaussian_power_expr(&mut ctx, expr).expect("rewrite");
        let g = extract_gaussian(&ctx, rewrite.rewritten).expect("gaussian");
        assert_eq!(g.real, BigRational::new((-3).into(), 4.into()));
        assert!(g.imag.is_one());
    }

    #[test]
    fn gaussian_abs_folds_modulus_exactly() {
        let mut ctx = Context::new();
        // Perfect square: |3+4i| = 5
        let expr = cas_parser::parse("abs(3 + 4*i)", &mut ctx).expect("parse");
        let rewrite = try_rewrite_gaussian_abs_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(rewrite.kind, ComplexRewriteKind::GaussianAbs);
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.rewritten
                }
            ),
            "5"
        );
        // Pure imaginary unit: |i| = 1
        let unit = cas_parser::parse("abs(i)", &mut ctx).expect("parse");
        let rewrite = try_rewrite_gaussian_abs_expr(&mut ctx, unit).expect("rewrite");
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.rewritten
                }
            ),
            "1"
        );
        // Non-perfect square stays exact: |1+i| = sqrt(2)
        let surd = cas_parser::parse("abs(1 + i)", &mut ctx).expect("parse");
        let rewrite = try_rewrite_gaussian_abs_expr(&mut ctx, surd).expect("rewrite");
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.rewritten
                }
            ),
            "sqrt(2)"
        );
    }

    #[test]
    fn gaussian_abs_declines_real_and_symbolic_args() {
        let mut ctx = Context::new();
        // Real argument: the real-abs machinery owns it.
        let real_abs = cas_parser::parse("abs(-3)", &mut ctx).expect("parse");
        assert!(try_rewrite_gaussian_abs_expr(&mut ctx, real_abs).is_none());
        // Symbolic argument: honest residual.
        let sym_abs = cas_parser::parse("abs(x)", &mut ctx).expect("parse");
        assert!(try_rewrite_gaussian_abs_expr(&mut ctx, sym_abs).is_none());
    }

    #[test]
    fn conjugate_re_im_evaluate_closed_gaussians() {
        let mut ctx = Context::new();
        let conj = cas_parser::parse("conjugate(3 + 4*i)", &mut ctx).expect("parse");
        let rewrite = try_rewrite_conjugate_expr(&mut ctx, conj).expect("rewrite");
        assert_eq!(rewrite.kind, ComplexRewriteKind::Conjugate);
        let g = extract_gaussian(&ctx, rewrite.rewritten).expect("gaussian");
        assert_eq!(g.real, BigRational::from_integer(3.into()));
        assert_eq!(g.imag, BigRational::from_integer((-4).into()));

        let re = cas_parser::parse("Re(3 + 4*i)", &mut ctx).expect("parse");
        let rewrite = try_rewrite_re_expr(&mut ctx, re).expect("rewrite");
        assert_eq!(rewrite.kind, ComplexRewriteKind::RealPart);
        let g = extract_gaussian(&ctx, rewrite.rewritten).expect("gaussian");
        assert_eq!(g.real, BigRational::from_integer(3.into()));
        assert!(g.imag.is_zero());

        let im = cas_parser::parse("Im(3 + 4*i)", &mut ctx).expect("parse");
        let rewrite = try_rewrite_im_expr(&mut ctx, im).expect("rewrite");
        assert_eq!(rewrite.kind, ComplexRewriteKind::ImagPart);
        let g = extract_gaussian(&ctx, rewrite.rewritten).expect("gaussian");
        assert_eq!(g.real, BigRational::from_integer(4.into()));

        // Real arg: conjugate(5) = 5, Im(7) = 0 (still closed Gaussians).
        let conj_real = cas_parser::parse("conjugate(5)", &mut ctx).expect("parse");
        assert!(try_rewrite_conjugate_expr(&mut ctx, conj_real).is_some());
        // Symbolic arg declines (honest residual).
        let conj_sym = cas_parser::parse("conjugate(x + i)", &mut ctx).expect("parse");
        assert!(try_rewrite_conjugate_expr(&mut ctx, conj_sym).is_none());
        let re_sym = cas_parser::parse("Re(x)", &mut ctx).expect("parse");
        assert!(try_rewrite_re_expr(&mut ctx, re_sym).is_none());
    }

    #[test]
    fn euler_rewrites_the_splitter_shapes() {
        // The shapes extract_gaussian cannot split (the B2 blocker): i·π,
        // π·i/2 (Div(Mul(Pi,I),2)), 2·π·i (3-factor chain), i·x (symbolic),
        // and the mixed Gaussian-rational exponent 1+i.
        for (src, must_contain) in [
            ("e^(i*pi)", "cos(pi)"),
            ("e^(pi*i/2)", "cos(pi / 2)"),
            ("e^(2*pi*i)", "cos(2 * pi)"),
            ("e^(i*x)", "cos(x)"),
            ("e^(1+i)", "cos(1)"),
        ] {
            let mut ctx = Context::new();
            let expr = cas_parser::parse(src, &mut ctx).expect("parse");
            let rewrite = try_rewrite_euler_expr(&mut ctx, expr)
                .unwrap_or_else(|| panic!("euler should fire for {src}"));
            assert_eq!(rewrite.kind, ComplexRewriteKind::Euler);
            let shown = format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.rewritten
                }
            );
            assert!(
                shown.contains(must_contain),
                "{src} -> {shown} (expected to contain {must_contain})"
            );
        }
    }

    #[test]
    fn euler_declines_unclean_splits_and_real_exponents() {
        for src in [
            "e^x",         // no i at all
            "e^2",         // pure real
            "e^(i*i*x)",   // i in both factors (not a clean linear split)
            "e^(x/(2*i))", // i in a denominator
            "e^(i^3)",     // i under a Pow
            "x^(i*pi)",    // base is not E
        ] {
            let mut ctx = Context::new();
            let expr = cas_parser::parse(src, &mut ctx).expect("parse");
            assert!(
                try_rewrite_euler_expr(&mut ctx, expr).is_none(),
                "euler must decline {src}"
            );
        }
    }

    #[test]
    fn euler_output_matches_complex_evaluator_at_generic_angles() {
        // Independent B1-net verification: at NON-special angles the rewrite
        // must agree numerically with the principal-branch evaluator.
        use crate::evaluator_complex::eval_complex;
        use std::collections::HashMap;
        for src in ["e^(i/5)", "e^(3*i)", "e^(2+5*i)"] {
            let mut ctx = Context::new();
            let expr = cas_parser::parse(src, &mut ctx).expect("parse");
            let before = eval_complex(&ctx, expr, &HashMap::new()).expect("eval before");
            let rewrite = try_rewrite_euler_expr(&mut ctx, expr).expect("euler fires");
            let after = eval_complex(&ctx, rewrite.rewritten, &HashMap::new()).expect("eval after");
            assert!(
                (before.re - after.re).abs() < 1e-12 && (before.im - after.im).abs() < 1e-12,
                "{src}: rewrite changed the value {before:?} -> {after:?}"
            );
        }
    }

    #[test]
    fn principal_arg_nine_case_table() {
        // The exact atan2 sign table over (-π, π]: axes, quadrants, zero.
        for (src, expected) in [
            ("arg(3)", "0"),
            ("arg(-2)", "pi"),
            ("arg(i)", "pi / 2"),
            ("arg(-i)", "-pi / 2"),
            ("arg(1 + i)", "atan(1)"),
            ("arg(-1 + i)", "atan(-1) + pi"),
            ("arg(-1 - i)", "atan(1) - pi"),
            ("arg(2 - 2*i)", "atan(-1)"),
        ] {
            let mut ctx = Context::new();
            let expr = cas_parser::parse(src, &mut ctx).expect("parse");
            let rewrite = try_rewrite_arg_expr(&mut ctx, expr)
                .unwrap_or_else(|| panic!("arg should fire for {src}"));
            let shown = format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.rewritten
                }
            );
            assert_eq!(shown, expected, "{src}");
        }
        // arg(0) -> Undefined explicitly; symbolic declines.
        let mut ctx = Context::new();
        let zero_arg = cas_parser::parse("arg(0)", &mut ctx).expect("parse");
        let rewrite = try_rewrite_arg_expr(&mut ctx, zero_arg).expect("fires");
        assert!(matches!(
            ctx.get(rewrite.rewritten),
            cas_ast::Expr::Constant(Constant::Undefined)
        ));
        let sym = cas_parser::parse("arg(x)", &mut ctx).expect("parse");
        assert!(try_rewrite_arg_expr(&mut ctx, sym).is_none());
    }

    #[test]
    fn principal_log_shapes_and_declines() {
        for (src, must_contain) in [
            ("ln(-1)", "pi"),
            ("ln(i)", "pi"),
            ("ln(-2)", "ln(2)"),
            ("ln(1 + i)", "ln(2)"),
            ("ln(2*i)", "ln(2)"),
        ] {
            let mut ctx = Context::new();
            let expr = cas_parser::parse(src, &mut ctx).expect("parse");
            let rewrite = try_rewrite_complex_ln_expr(&mut ctx, expr)
                .unwrap_or_else(|| panic!("complex ln should fire for {src}"));
            assert_eq!(rewrite.kind, ComplexRewriteKind::PrincipalLog);
            let shown = format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.rewritten
                }
            );
            assert!(shown.contains(must_contain), "{src} -> {shown}");
        }
        // Ownership declines: positive rational, zero, symbolic.
        for src in ["ln(2)", "ln(1)", "ln(0)", "ln(x)"] {
            let mut ctx = Context::new();
            let expr = cas_parser::parse(src, &mut ctx).expect("parse");
            assert!(
                try_rewrite_complex_ln_expr(&mut ctx, expr).is_none(),
                "complex ln must decline {src}"
            );
        }
    }

    #[test]
    fn principal_log_and_arg_match_complex_evaluator() {
        // Independent B1-net verification on generic (non-special) Gaussians.
        use crate::evaluator_complex::eval_complex;
        use std::collections::HashMap;
        for src in ["ln(3 + 4*i)", "ln(-2 + i)", "ln(-3 - 5*i)"] {
            let mut ctx = Context::new();
            let expr = cas_parser::parse(src, &mut ctx).expect("parse");
            let before = eval_complex(&ctx, expr, &HashMap::new()).expect("eval before");
            let rewrite = try_rewrite_complex_ln_expr(&mut ctx, expr).expect("fires");
            let after = eval_complex(&ctx, rewrite.rewritten, &HashMap::new()).expect("eval after");
            assert!(
                (before.re - after.re).abs() < 1e-12 && (before.im - after.im).abs() < 1e-12,
                "{src}: rewrite changed the value {before:?} -> {after:?}"
            );
        }
    }

    #[test]
    fn gaussian_sqrt_folds_perfect_square_norms_exactly() {
        for (src, expected) in [
            ("sqrt(3 + 4*i)", "2 + i"),
            ("sqrt(3 - 4*i)", "2 - i"),
            ("(3 + 4*i)^(1/2)", "2 + i"),
            ("sqrt(-5 + 12*i)", "2 + 3 * i"),
        ] {
            let mut ctx = Context::new();
            let expr = cas_parser::parse(src, &mut ctx).expect("parse");
            let rewrite = try_rewrite_gaussian_sqrt_expr(&mut ctx, expr)
                .unwrap_or_else(|| panic!("sqrt should fire for {src}"));
            assert_eq!(rewrite.kind, ComplexRewriteKind::GaussianSqrt);
            let shown = format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.rewritten
                }
            );
            assert_eq!(
                shown.replace(" · ", "·"),
                expected.replace(" · ", "·"),
                "{src}"
            );
        }
    }

    #[test]
    fn gaussian_sqrt_declines_owned_and_out_of_scope_radicands() {
        // Pure-real radicands (real machinery / SqrtNegative own), norms
        // that are not perfect squares (polar route), symbolic radicands.
        for src in [
            "sqrt(4)",
            "sqrt(-4)",
            "(-4)^(1/2)",
            "sqrt(1 + i)",
            "sqrt(x + i)",
        ] {
            let mut ctx = Context::new();
            let expr = cas_parser::parse(src, &mut ctx).expect("parse");
            assert!(
                try_rewrite_gaussian_sqrt_expr(&mut ctx, expr).is_none(),
                "gaussian sqrt must decline {src}"
            );
        }
    }

    #[test]
    fn complex_general_power_emits_log_form_and_declines_owned_shapes() {
        // i^i -> e^(i·ln(i)): the general branch emits the exponential-log
        // form for the pipeline (ComplexLogRule folds ln(i) next).
        let mut ctx = Context::new();
        let expr = cas_parser::parse("i^i", &mut ctx).expect("parse");
        let rewrite = try_rewrite_complex_general_power_expr(&mut ctx, expr).expect("i^i fires");
        assert_eq!(rewrite.kind, ComplexRewriteKind::ComplexGeneralPower);
        let shown = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.rewritten
            }
        );
        assert!(shown.contains("ln(i)"), "i^i -> {shown}");

        // 2^i takes the DIRECT Euler branch (no e^(w·ln z) intermediate):
        // routing through the log form would ping-pong with the real
        // exp-log canonicalizer since ln(2) stays symbolic.
        let mut ctx = Context::new();
        let expr = cas_parser::parse("2^i", &mut ctx).expect("parse");
        let rewrite = try_rewrite_complex_general_power_expr(&mut ctx, expr).expect("2^i fires");
        let shown = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.rewritten
            }
        );
        assert!(
            shown.contains("cos") && shown.contains("sin") && !shown.contains("e^"),
            "2^i -> {shown}"
        );

        // Owned or out-of-scope shapes decline, each with its owner:
        // base e (EulerRule + anti-churn), integer exponents (GaussianPow),
        // real-real rationals (real machinery), perfect-square half powers
        // (GaussianSqrtRule), zero base (no principal log), symbolics.
        for src in [
            "e^i",
            "(2 + i)^2",
            "(1 + i)^(-2)",
            "2^(1/2)",
            "(-8)^(1/3)",
            "(3 + 4*i)^(1/2)",
            "0^i",
            "x^i",
            "2^x",
        ] {
            let mut ctx = Context::new();
            let expr = cas_parser::parse(src, &mut ctx).expect("parse");
            assert!(
                try_rewrite_complex_general_power_expr(&mut ctx, expr).is_none(),
                "general power must decline {src}"
            );
        }
    }

    #[test]
    fn gaussian_sqrt_and_general_power_match_complex_evaluator() {
        // Independent B1-net verification: the rewrite preserves the value.
        use crate::evaluator_complex::eval_complex;
        use std::collections::HashMap;
        for src in ["sqrt(3 + 4*i)", "sqrt(-5 + 12*i)", "sqrt(i)"] {
            let mut ctx = Context::new();
            let expr = cas_parser::parse(src, &mut ctx).expect("parse");
            let before = eval_complex(&ctx, expr, &HashMap::new()).expect("eval before");
            let rewrite = try_rewrite_gaussian_sqrt_expr(&mut ctx, expr).expect("fires");
            let after = eval_complex(&ctx, rewrite.rewritten, &HashMap::new()).expect("eval after");
            assert!(
                (before.re - after.re).abs() < 1e-12 && (before.im - after.im).abs() < 1e-12,
                "{src}: {before:?} -> {after:?}"
            );
        }
        for src in [
            "i^i",
            "2^i",
            "(-2)^i",
            "(1 + i)^(1/2)",
            "i^(1/3)",
            "2^(1 + i)",
        ] {
            let mut ctx = Context::new();
            let expr = cas_parser::parse(src, &mut ctx).expect("parse");
            let before = eval_complex(&ctx, expr, &HashMap::new()).expect("eval before");
            let rewrite = try_rewrite_complex_general_power_expr(&mut ctx, expr).expect("fires");
            let after = eval_complex(&ctx, rewrite.rewritten, &HashMap::new()).expect("eval after");
            assert!(
                (before.re - after.re).abs() < 1e-10 && (before.im - after.im).abs() < 1e-10,
                "{src}: {before:?} -> {after:?}"
            );
        }
    }

    #[test]
    fn gaussian_power_declines_out_of_scope_bases_and_exponents() {
        let mut ctx = Context::new();
        // Real base: ordinary arithmetic owns it.
        let real_pow = cas_parser::parse("3^2", &mut ctx).expect("parse");
        assert!(try_rewrite_gaussian_power_expr(&mut ctx, real_pow).is_none());
        // Pure-imaginary base: power-of-a-product + i^n own it.
        let imag_pow = cas_parser::parse("(2*i)^3", &mut ctx).expect("parse");
        assert!(try_rewrite_gaussian_power_expr(&mut ctx, imag_pow).is_none());
        // n = 1 has an existing owner; negative and fractional exponents decline.
        let unit_pow = cas_parser::parse("(1 + i)^1", &mut ctx).expect("parse");
        assert!(try_rewrite_gaussian_power_expr(&mut ctx, unit_pow).is_none());
        let neg_pow = cas_parser::parse("(1 + i)^(-2)", &mut ctx).expect("parse");
        assert!(try_rewrite_gaussian_power_expr(&mut ctx, neg_pow).is_none());
        let frac_pow = cas_parser::parse("(1 + i)^(1/2)", &mut ctx).expect("parse");
        assert!(try_rewrite_gaussian_power_expr(&mut ctx, frac_pow).is_none());
        // Symbolic base declines (extract_gaussian never captures symbols).
        let sym_pow = cas_parser::parse("(x + i)^2", &mut ctx).expect("parse");
        assert!(try_rewrite_gaussian_power_expr(&mut ctx, sym_pow).is_none());
        // Exponent beyond the eager-evaluation cap declines (honest residual).
        let huge_pow = cas_parser::parse("(1 + i)^5000", &mut ctx).expect("parse");
        assert!(try_rewrite_gaussian_power_expr(&mut ctx, huge_pow).is_none());
    }
}
