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
    SqrtNegative,
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
pub fn try_rewrite_i_squared_mul_expr(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
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
        if left.is_real() && right.is_pure_imag() {
            return None;
        }
        if right.is_real() && left.is_pure_imag() {
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

    let i = ctx.add(Expr::Constant(Constant::I));
    let rewritten = if abs_value.is_integer() {
        let int_val = abs_value.to_integer();
        if let Some(f) = int_val.to_f64() {
            let sqrt_f = f.sqrt();
            if sqrt_f.fract() == 0.0 && sqrt_f > 0.0 {
                let sqrt_int = sqrt_f as i64;
                let sqrt_num = ctx.num(sqrt_int);
                if sqrt_int == 1 {
                    i
                } else {
                    ctx.add(Expr::Mul(sqrt_num, i))
                }
            } else {
                let abs_num = ctx.add(Expr::Number(abs_value));
                let sqrt_abs = ctx.call_builtin(BuiltinFn::Sqrt, vec![abs_num]);
                ctx.add(Expr::Mul(i, sqrt_abs))
            }
        } else {
            let abs_num = ctx.add(Expr::Number(abs_value));
            let sqrt_abs = ctx.call_builtin(BuiltinFn::Sqrt, vec![abs_num]);
            ctx.add(Expr::Mul(i, sqrt_abs))
        }
    } else {
        let abs_num = ctx.add(Expr::Number(abs_value));
        let sqrt_abs = ctx.call_builtin(BuiltinFn::Sqrt, vec![abs_num]);
        ctx.add(Expr::Mul(i, sqrt_abs))
    };

    Some(ComplexRewrite {
        rewritten,
        kind: ComplexRewriteKind::SqrtNegative,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        extract_gaussian, try_rewrite_gaussian_div_expr, try_rewrite_gaussian_mul_expr,
        try_rewrite_i_squared_mul_identity_expr, try_rewrite_imaginary_power_expr,
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
}
