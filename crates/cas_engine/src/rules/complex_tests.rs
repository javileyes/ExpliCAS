use super::complex::{extract_gaussian, GaussianPowRule, GaussianRational, ImaginaryPowerRule};
use crate::rule::Rule;
use cas_ast::{Constant, Context, Expr};
use cas_formatter::DisplayExpr;
use num_rational::BigRational;
use num_traits::{One, Zero};

fn complex_ctx() -> crate::parent_context::ParentContext {
    crate::parent_context::ParentContext::root()
        .with_value_domain(crate::semantics::ValueDomain::ComplexEnabled)
}

#[test]
fn test_i_squared() {
    let mut ctx = Context::new();
    let rule = ImaginaryPowerRule;
    let i = ctx.add(Expr::Constant(Constant::I));
    let two = ctx.num(2);
    let i_squared = ctx.add(Expr::Pow(i, two));

    let rewrite = rule.apply(&mut ctx, i_squared, &complex_ctx()).unwrap();
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
    let i = ctx.add(Expr::Constant(Constant::I));
    let three = ctx.num(3);
    let i_cubed = ctx.add(Expr::Pow(i, three));

    let rewrite = rule.apply(&mut ctx, i_cubed, &complex_ctx()).unwrap();
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
    let i = ctx.add(Expr::Constant(Constant::I));
    let four = ctx.num(4);
    let i_fourth = ctx.add(Expr::Pow(i, four));

    let rewrite = rule.apply(&mut ctx, i_fourth, &complex_ctx()).unwrap();
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
    let i = ctx.add(Expr::Constant(Constant::I));
    let seventeen = ctx.num(17);
    let i_17 = ctx.add(Expr::Pow(i, seventeen));

    let rewrite = rule.apply(&mut ctx, i_17, &complex_ctx()).unwrap();
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

/// Build `(a + i)^n` with integer `a`.
fn int_plus_i_pow(ctx: &mut Context, a: i64, n: i64) -> cas_ast::ExprId {
    let re = ctx.num(a);
    let i = ctx.add(Expr::Constant(Constant::I));
    let base = ctx.add(Expr::Add(re, i));
    let exp = ctx.num(n);
    ctx.add(Expr::Pow(base, exp))
}

#[test]
fn test_gaussian_power_squares_binomial() {
    let mut ctx = Context::new();
    let rule = GaussianPowRule;
    let expr = int_plus_i_pow(&mut ctx, 1, 2);

    let rewrite = rule.apply(&mut ctx, expr, &complex_ctx()).unwrap();
    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        ),
        "2 * i"
    );
}

#[test]
fn test_gaussian_power_fourth_power_lands_real() {
    // (1+i)^4 = -4: the fold may land on a pure-real value (no re-match risk).
    let mut ctx = Context::new();
    let rule = GaussianPowRule;
    let expr = int_plus_i_pow(&mut ctx, 1, 4);

    let rewrite = rule.apply(&mut ctx, expr, &complex_ctx()).unwrap();
    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        ),
        "-4"
    );
}

#[test]
fn test_gaussian_power_general_binomial() {
    // (2+i)^4 = -7 + 24i. Fase 2 C1 landed the cartesian display order:
    // the real part now leads even when negative ("-7 + 24 * i").
    let mut ctx = Context::new();
    let rule = GaussianPowRule;
    let expr = int_plus_i_pow(&mut ctx, 2, 4);

    let rewrite = rule.apply(&mut ctx, expr, &complex_ctx()).unwrap();
    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        ),
        "-7 + 24 * i"
    );
    // Value check independent of display ordering: extract back as Gaussian.
    let g = extract_gaussian(&ctx, rewrite.new_expr).expect("gaussian");
    assert_eq!(g.real, BigRational::from_integer((-7).into()));
    assert_eq!(g.imag, BigRational::from_integer(24.into()));
}

#[test]
fn test_gaussian_power_inert_in_real_only_mode() {
    // Footprint guard: in RealOnly (the default) the rule must return None,
    // keeping real-mode output byte-identical.
    let mut ctx = Context::new();
    let rule = GaussianPowRule;
    let expr = int_plus_i_pow(&mut ctx, 1, 2);

    let real_ctx = crate::parent_context::ParentContext::root();
    assert!(rule.apply(&mut ctx, expr, &real_ctx).is_none());
}

#[test]
fn test_complex_builtins_gated_to_complex_mode() {
    // The four A2 rules (modulus / conjugate / Re / Im) fire only under
    // ComplexEnabled; in RealOnly (default) the functions stay symbolic —
    // footprint guard for real mode.
    use super::complex::{ConjugateRule, GaussianAbsRule, ImagPartRule, RealPartRule};

    let mut ctx = Context::new();
    let three = ctx.num(3);
    let four = ctx.num(4);
    let i = ctx.add(Expr::Constant(Constant::I));
    let four_i = ctx.add(Expr::Mul(four, i));
    let z = ctx.add(Expr::Add(three, four_i));
    let abs_z = ctx.call_builtin(cas_ast::BuiltinFn::Abs, vec![z]);
    let conj_z = ctx.call_builtin(cas_ast::BuiltinFn::Conjugate, vec![z]);
    let re_z = ctx.call_builtin(cas_ast::BuiltinFn::Re, vec![z]);
    let im_z = ctx.call_builtin(cas_ast::BuiltinFn::Im, vec![z]);

    let real_ctx = crate::parent_context::ParentContext::root();
    assert!(GaussianAbsRule.apply(&mut ctx, abs_z, &real_ctx).is_none());
    assert!(ConjugateRule.apply(&mut ctx, conj_z, &real_ctx).is_none());
    assert!(RealPartRule.apply(&mut ctx, re_z, &real_ctx).is_none());
    assert!(ImagPartRule.apply(&mut ctx, im_z, &real_ctx).is_none());

    let show = |ctx: &Context, id| format!("{}", DisplayExpr { context: ctx, id });
    let r = GaussianAbsRule
        .apply(&mut ctx, abs_z, &complex_ctx())
        .unwrap();
    assert_eq!(show(&ctx, r.new_expr), "5");
    let r = ConjugateRule
        .apply(&mut ctx, conj_z, &complex_ctx())
        .unwrap();
    assert_eq!(show(&ctx, r.new_expr), "3 - 4 * i");
    let r = RealPartRule.apply(&mut ctx, re_z, &complex_ctx()).unwrap();
    assert_eq!(show(&ctx, r.new_expr), "3");
    let r = ImagPartRule.apply(&mut ctx, im_z, &complex_ctx()).unwrap();
    assert_eq!(show(&ctx, r.new_expr), "4");
}

#[test]
fn test_gaussian_power_declines_owned_shapes() {
    let mut ctx = Context::new();
    let rule = GaussianPowRule;

    // Pure-imaginary base is owned by power-of-a-product + i^n.
    let two = ctx.num(2);
    let i = ctx.add(Expr::Constant(Constant::I));
    let two_i = ctx.add(Expr::Mul(two, i));
    let three = ctx.num(3);
    let imag_pow = ctx.add(Expr::Pow(two_i, three));
    assert!(rule.apply(&mut ctx, imag_pow, &complex_ctx()).is_none());

    // n = 1 has an existing owner; negative exponents decline here.
    let unit_pow = int_plus_i_pow(&mut ctx, 1, 1);
    assert!(rule.apply(&mut ctx, unit_pow, &complex_ctx()).is_none());
    let neg_pow = int_plus_i_pow(&mut ctx, 1, -2);
    assert!(rule.apply(&mut ctx, neg_pow, &complex_ctx()).is_none());
}
