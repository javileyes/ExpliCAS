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
}

/// Extract `a + bi` from an expression when possible.
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
        extract_gaussian, try_rewrite_conjugate_expr, try_rewrite_euler_expr,
        try_rewrite_gaussian_abs_expr, try_rewrite_gaussian_add_expr,
        try_rewrite_gaussian_div_expr, try_rewrite_gaussian_mul_expr,
        try_rewrite_gaussian_power_expr, try_rewrite_i_squared_mul_identity_expr,
        try_rewrite_im_expr, try_rewrite_imaginary_power_expr, try_rewrite_re_expr,
        try_rewrite_sqrt_negative_expr, ComplexRewriteKind, GaussianRational,
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
