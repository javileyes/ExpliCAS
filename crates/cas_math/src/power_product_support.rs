//! Support for product/quotient rewrites over powers.

use crate::build::mul2_raw;
use crate::canonical_forms::is_canonical_form;
use crate::exponents_support::{add_exp, has_numeric_factor};
use crate::expr_destructure::{as_div, as_mul, as_pow};
use crate::expr_nary::{build_balanced_mul, mul_leaves};
use crate::expr_predicates::is_e_constant_expr;
use crate::root_forms::can_distribute_root_safely;
use cas_ast::ordering::compare_expr;
use cas_ast::views::FractionParts;
use cas_ast::{Constant, Context, Expr, ExprId};
use num_integer::Integer;
use num_rational::BigRational;
use num_traits::Signed;
use num_traits::{One, Zero};
use std::cmp::Ordering;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PowerProductRewrite {
    pub rewritten: ExprId,
    pub kind: PowerProductRewriteKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowerProductRewriteKind {
    SameBase,
    PowerAndBase,
    BaseAndPower,
    MultiplyIdenticalTerms,
    NestedIdenticalTerms,
    NestedPowers,
    BaseAndNestedPower,
    PowerAndNestedBase,
    CoeffPowerAndPower,
    CoeffPowerAndBase,
    CoeffBaseAndPower,
    NestedBaseAndPower,
    DistributePowerOverProduct,
    SameExponent,
    NestedSameExponent,
    QuotientSameExponent,
    DistributePowerOverQuotient,
    ExpQuotient,
    ExpOverExpPower,
    ExpPowerOverExp,
    AllFactorsCancelled,
    SameBaseNary,
}

/// Try product-of-powers rewrites:
/// - `x^a * x^b -> x^(a+b)`
/// - `x^a * x -> x^(a+1)` and symmetric variants
/// - selected nested variants preserving multiplicative tails
pub fn try_rewrite_product_power_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<PowerProductRewrite> {
    let should_combine = |ctx: &Context, base: ExprId, e1: ExprId, e2: ExprId| -> bool {
        if let Expr::Number(_) = ctx.get(base) {
            if let (Expr::Number(n1), Expr::Number(n2)) = (ctx.get(e1), ctx.get(e2)) {
                let sum = n1 + n2;
                if sum.is_integer() {
                    return true;
                }
                let num = sum.numer().abs();
                let den = sum.denom().abs();
                return num < den;
            }
        }
        true
    };

    if let Some((lhs, rhs)) = as_mul(ctx, expr) {
        let lhs_pow = as_pow(ctx, lhs);
        let rhs_pow = as_pow(ctx, rhs);

        if let (Some((base1, exp1)), Some((base2, exp2))) = (lhs_pow, rhs_pow) {
            if compare_expr(ctx, base1, base2) == Ordering::Equal
                && should_combine(ctx, base1, exp1, exp2)
            {
                let sum_exp = add_exp(ctx, exp1, exp2);
                let rewritten = ctx.add(Expr::Pow(base1, sum_exp));
                return Some(PowerProductRewrite {
                    rewritten,
                    kind: PowerProductRewriteKind::SameBase,
                });
            }
        }

        if let Some((base1, exp1)) = lhs_pow {
            if compare_expr(ctx, base1, rhs) == Ordering::Equal {
                let one = ctx.num(1);
                if should_combine(ctx, base1, exp1, one) {
                    let sum_exp = add_exp(ctx, exp1, one);
                    let rewritten = ctx.add(Expr::Pow(base1, sum_exp));
                    return Some(PowerProductRewrite {
                        rewritten,
                        kind: PowerProductRewriteKind::PowerAndBase,
                    });
                }
            }
        }

        if let Some((base2, exp2)) = rhs_pow {
            if compare_expr(ctx, base2, lhs) == Ordering::Equal {
                let one = ctx.num(1);
                if should_combine(ctx, base2, one, exp2) {
                    let sum_exp = add_exp(ctx, one, exp2);
                    let rewritten = ctx.add(Expr::Pow(base2, sum_exp));
                    return Some(PowerProductRewrite {
                        rewritten,
                        kind: PowerProductRewriteKind::BaseAndPower,
                    });
                }
            }
        }

        if compare_expr(ctx, lhs, rhs) == Ordering::Equal {
            let two = ctx.num(2);
            let rewritten = ctx.add(Expr::Pow(lhs, two));
            return Some(PowerProductRewrite {
                rewritten,
                kind: PowerProductRewriteKind::MultiplyIdenticalTerms,
            });
        }

        if let Some((rl, rr)) = as_mul(ctx, rhs) {
            if compare_expr(ctx, lhs, rl) == Ordering::Equal {
                let two = ctx.num(2);
                let x_squared = ctx.add(Expr::Pow(lhs, two));
                let rewritten = mul2_raw(ctx, x_squared, rr);
                return Some(PowerProductRewrite {
                    rewritten,
                    kind: PowerProductRewriteKind::NestedIdenticalTerms,
                });
            }

            let rl_pow = as_pow(ctx, rl);

            if let (Some((base1, exp1)), Some((base2, exp2))) = (lhs_pow, rl_pow) {
                if compare_expr(ctx, base1, base2) == Ordering::Equal {
                    let sum_exp = add_exp(ctx, exp1, exp2);
                    let new_pow = ctx.add(Expr::Pow(base1, sum_exp));
                    let rewritten = mul2_raw(ctx, new_pow, rr);
                    return Some(PowerProductRewrite {
                        rewritten,
                        kind: PowerProductRewriteKind::NestedPowers,
                    });
                }
            }

            if let Some((base2, exp2)) = rl_pow {
                if compare_expr(ctx, lhs, base2) == Ordering::Equal {
                    let one = ctx.num(1);
                    let sum_exp = add_exp(ctx, exp2, one);
                    let new_pow = ctx.add(Expr::Pow(base2, sum_exp));
                    let rewritten = mul2_raw(ctx, new_pow, rr);
                    return Some(PowerProductRewrite {
                        rewritten,
                        kind: PowerProductRewriteKind::BaseAndNestedPower,
                    });
                }
            }

            if let Some((base1, exp1)) = lhs_pow {
                if compare_expr(ctx, base1, rl) == Ordering::Equal {
                    let one = ctx.num(1);
                    let sum_exp = ctx.add(Expr::Add(exp1, one));
                    let new_pow = ctx.add(Expr::Pow(base1, sum_exp));
                    let rewritten = mul2_raw(ctx, new_pow, rr);
                    return Some(PowerProductRewrite {
                        rewritten,
                        kind: PowerProductRewriteKind::PowerAndNestedBase,
                    });
                }
            }

            if let Some((ll, lr)) = as_mul(ctx, lhs) {
                if let Expr::Number(_) = ctx.get(ll) {
                    let lr_pow = as_pow(ctx, lr);

                    if let (Some((base1, exp1)), Some((base2, exp2))) = (lr_pow, rhs_pow) {
                        if compare_expr(ctx, base1, base2) == Ordering::Equal {
                            let sum_exp = add_exp(ctx, exp1, exp2);
                            let new_pow = ctx.add(Expr::Pow(base1, sum_exp));
                            let rewritten = mul2_raw(ctx, ll, new_pow);
                            return Some(PowerProductRewrite {
                                rewritten,
                                kind: PowerProductRewriteKind::CoeffPowerAndPower,
                            });
                        }
                    }

                    if let Some((base1, exp1)) = lr_pow {
                        if compare_expr(ctx, base1, rhs) == Ordering::Equal {
                            let one = ctx.num(1);
                            let sum_exp = ctx.add(Expr::Add(exp1, one));
                            let new_pow = ctx.add(Expr::Pow(base1, sum_exp));
                            let rewritten = mul2_raw(ctx, ll, new_pow);
                            return Some(PowerProductRewrite {
                                rewritten,
                                kind: PowerProductRewriteKind::CoeffPowerAndBase,
                            });
                        }
                    }
                }
            }

            if let Some((ll, lr)) = as_mul(ctx, lhs) {
                if let Expr::Number(_) = ctx.get(ll) {
                    if let Some((base2, exp2)) = rhs_pow {
                        if compare_expr(ctx, lr, base2) == Ordering::Equal {
                            let one = ctx.num(1);
                            let sum_exp = add_exp(ctx, exp2, one);
                            let new_pow = ctx.add(Expr::Pow(base2, sum_exp));
                            let rewritten = mul2_raw(ctx, ll, new_pow);
                            return Some(PowerProductRewrite {
                                rewritten,
                                kind: PowerProductRewriteKind::CoeffBaseAndPower,
                            });
                        }
                    }
                }
            }

            if let Some((base2, exp2)) = rl_pow {
                if compare_expr(ctx, lhs, base2) == Ordering::Equal {
                    let one = ctx.num(1);
                    let sum_exp = ctx.add(Expr::Add(one, exp2));
                    let new_pow = ctx.add(Expr::Pow(base2, sum_exp));
                    let rewritten = mul2_raw(ctx, new_pow, rr);
                    return Some(PowerProductRewrite {
                        rewritten,
                        kind: PowerProductRewriteKind::NestedBaseAndPower,
                    });
                }
            }
        }
    }
    None
}

/// Try distribution of power over product:
/// `(a*b)^n -> a^n * b^n`.
pub fn try_rewrite_power_product_distribution_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<PowerProductRewrite> {
    if is_canonical_form(ctx, expr) {
        return None;
    }

    if let Some((base, exp)) = as_pow(ctx, expr) {
        if let Some((a, b)) = as_mul(ctx, base) {
            if let Expr::Number(exp_num) = ctx.get(exp) {
                let denom = exp_num.denom();
                if denom > &num_bigint::BigInt::from(1)
                    && !can_distribute_root_safely(ctx, base, denom)
                {
                    return None;
                }
            }

            // `(a*b)^x -> a^x * b^x` for a SYMBOLIC (possibly non-integer) exponent
            // is valid over R only when both bases are >= 0: for negative a,b the
            // product `(a*b)^x` is real but the split factors `a^x`, `b^x` are
            // individually complex, so the identity does NOT hold over the reals.
            // Numeric exponents are already handled above — integers are universally
            // safe (`a^n*b^n = (a*b)^n` for all a,b), fractional ones go through
            // `can_distribute_root_safely`. Decline the symbolic-exponent split
            // unless both bases are provably non-negative, rather than silently
            // dropping the `a>=0 AND b>=0` domain condition.
            let exp_is_numeric = matches!(ctx.get(exp), Expr::Number(_));
            let split_is_sound = exp_is_numeric
                || (base_is_provably_nonnegative(ctx, a) && base_is_provably_nonnegative(ctx, b));
            if !split_is_sound {
                return None;
            }

            let a_pow = ctx.add(Expr::Pow(a, exp));
            let b_pow = ctx.add(Expr::Pow(b, exp));
            let rewritten = mul2_raw(ctx, a_pow, b_pow);
            return Some(PowerProductRewrite {
                rewritten,
                kind: PowerProductRewriteKind::DistributePowerOverProduct,
            });
        }
    }

    None
}

/// True when `expr` folds to a negative rational constant.
fn is_negative_const_value(ctx: &Context, expr: ExprId) -> bool {
    crate::numeric_eval::as_rational_const(ctx, expr).is_some_and(|r| r.is_negative())
}

/// True when `base` is provably non-negative over the reals — so that
/// `base^x` is real-valued for a symbolic exponent and the product-power split
/// `(a*b)^x -> a^x * b^x` is sound. Conservative: a non-negative numeric
/// constant, an even-integer power (`y^(2k) >= 0`), an absolute value, the
/// constant `e` (`> 0`), or a product of provably-non-negative factors. Returns
/// `false` when non-negativity is not provable (e.g. a bare variable), which
/// makes the symbolic-exponent split decline.
pub(crate) fn base_is_provably_nonnegative(ctx: &Context, base: ExprId) -> bool {
    if let Some(r) = crate::numeric_eval::as_rational_const(ctx, base) {
        return !r.is_negative();
    }
    match ctx.get(base) {
        Expr::Constant(Constant::E) => true,
        Expr::Function(fn_id, args) => {
            args.len() == 1 && ctx.builtin_of(*fn_id) == Some(cas_ast::BuiltinFn::Abs)
        }
        Expr::Pow(_, exp) => crate::numeric_eval::as_rational_const(ctx, *exp)
            .is_some_and(|e| e.is_integer() && e.numer().is_even()),
        Expr::Mul(a, b) => {
            base_is_provably_nonnegative(ctx, *a) && base_is_provably_nonnegative(ctx, *b)
        }
        _ => false,
    }
}

/// True when merging `a^exp * b^exp -> (a*b)^exp` is unsound: `exp` is an
/// even-denominator (even-root) rational and EITHER base is a provably-negative
/// constant. An even root of a negative is undefined over the reals, so the
/// product is undefined and must NOT collapse — e.g. `sqrt(-2)*sqrt(-3)` must not
/// become `sqrt(6)` (its complex principal value is `-sqrt(6)`, not `+sqrt(6)`).
/// Gating on EITHER (not both) also blocks the pairwise cascade:
/// `sqrt(-3)*sqrt(-5)*sqrt(x)` must not become `(15·x)^(1/2)` via an intermediate
/// `sqrt(-5)*sqrt(x)` merge whose result base is no longer a bare constant.
fn even_root_merge_of_negatives(ctx: &Context, exp: ExprId, base1: ExprId, base2: ExprId) -> bool {
    let Some(e) = crate::numeric_eval::as_rational_const(ctx, exp) else {
        return false;
    };
    e.denom().is_even()
        && (is_negative_const_value(ctx, base1) || is_negative_const_value(ctx, base2))
}

/// Try same-exponent product rewrites:
/// - `a^n * b^n -> (a*b)^n`
/// - nested variant: `a^n * (b^n * c) -> (a*b)^n * c`
pub fn try_rewrite_product_same_exponent_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<PowerProductRewrite> {
    if let Some((lhs, rhs)) = as_mul(ctx, expr) {
        let lhs_pow = as_pow(ctx, lhs);
        let rhs_pow = as_pow(ctx, rhs);

        if let (Some((base1, exp1)), Some((base2, exp2))) = (lhs_pow, rhs_pow) {
            if compare_expr(ctx, exp1, exp2) == Ordering::Equal {
                let base1_is_num = matches!(ctx.get(base1), Expr::Number(_));
                let base2_is_num = matches!(ctx.get(base2), Expr::Number(_));
                let base1_has_num = base1_is_num || has_numeric_factor(ctx, base1);
                let base2_has_num = base2_is_num || has_numeric_factor(ctx, base2);
                if !base1_has_num && !base2_has_num {
                    return None;
                }
                if even_root_merge_of_negatives(ctx, exp1, base1, base2) {
                    return None;
                }

                let new_base = mul2_raw(ctx, base1, base2);
                let rewritten = ctx.add(Expr::Pow(new_base, exp1));
                return Some(PowerProductRewrite {
                    rewritten,
                    kind: PowerProductRewriteKind::SameExponent,
                });
            }
        }

        if let Some((base1, exp1)) = lhs_pow {
            if let Some((rl, rr)) = as_mul(ctx, rhs) {
                if let Some((base2, exp2)) = as_pow(ctx, rl) {
                    if compare_expr(ctx, exp1, exp2) == Ordering::Equal {
                        let base1_is_num = matches!(ctx.get(base1), Expr::Number(_));
                        let base2_is_num = matches!(ctx.get(base2), Expr::Number(_));
                        let base1_has_num = base1_is_num || has_numeric_factor(ctx, base1);
                        let base2_has_num = base2_is_num || has_numeric_factor(ctx, base2);
                        if !base1_has_num && !base2_has_num {
                            return None;
                        }
                        if even_root_merge_of_negatives(ctx, exp1, base1, base2) {
                            return None;
                        }

                        let new_base = mul2_raw(ctx, base1, base2);
                        let combined_pow = ctx.add(Expr::Pow(new_base, exp1));
                        let rewritten = mul2_raw(ctx, combined_pow, rr);
                        return Some(PowerProductRewrite {
                            rewritten,
                            kind: PowerProductRewriteKind::NestedSameExponent,
                        });
                    }
                }
            }
        }
    }

    None
}

/// Try same-exponent quotient rewrite:
/// `a^n / b^n -> (a/b)^n`.
pub fn try_rewrite_quotient_same_exponent_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<PowerProductRewrite> {
    let fp = FractionParts::from(&*ctx, expr);
    if !fp.is_fraction() {
        return None;
    }
    let (num, den, _) = fp.to_num_den(ctx);

    if let (Expr::Pow(base_num, exp_num), Expr::Pow(base_den, exp_den)) =
        (ctx.get(num), ctx.get(den))
    {
        let (base_num, exp_num, base_den, exp_den) = (*base_num, *exp_num, *base_den, *exp_den);
        if compare_expr(ctx, exp_num, exp_den) == Ordering::Equal {
            let base_num_is_num = matches!(ctx.get(base_num), Expr::Number(_));
            let base_den_is_num = matches!(ctx.get(base_den), Expr::Number(_));
            let base_num_has_num = base_num_is_num || has_numeric_factor(ctx, base_num);
            let base_den_has_num = base_den_is_num || has_numeric_factor(ctx, base_den);
            if !base_num_has_num && !base_den_has_num {
                return None;
            }

            let new_base = ctx.add(Expr::Div(base_num, base_den));
            let rewritten = ctx.add(Expr::Pow(new_base, exp_num));
            return Some(PowerProductRewrite {
                rewritten,
                kind: PowerProductRewriteKind::QuotientSameExponent,
            });
        }
    }

    None
}

/// Try quotient-power distribution rewrite:
/// `(a/b)^n -> a^n / b^n`.
pub fn try_rewrite_power_quotient_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<PowerProductRewrite> {
    if let Some((base, exp)) = as_pow(ctx, expr) {
        if let Expr::Number(exp_num) = ctx.get(exp) {
            let denom = exp_num.denom();
            if denom > &num_bigint::BigInt::from(1) && !can_distribute_root_safely(ctx, base, denom)
            {
                return None;
            }
        }

        if let Some((num, den)) = as_div(ctx, base) {
            // `(a/b)^x -> a^x / b^x` for a SYMBOLIC (possibly non-integer) exponent
            // is valid over R only when both base parts are >= 0 — same reasoning as
            // the product-base split: for negative a,b the split factors are
            // individually complex. Numeric exponents are handled above. Decline the
            // symbolic-exponent quotient split unless both parts are provably
            // non-negative, rather than dropping the `a>=0 AND b>=0` condition.
            let exp_is_numeric = matches!(ctx.get(exp), Expr::Number(_));
            let split_is_sound = exp_is_numeric
                || (base_is_provably_nonnegative(ctx, num)
                    && base_is_provably_nonnegative(ctx, den));
            if split_is_sound {
                let new_num = ctx.add(Expr::Pow(num, exp));
                let new_den = ctx.add(Expr::Pow(den, exp));
                let rewritten = ctx.add(Expr::Div(new_num, new_den));
                return Some(PowerProductRewrite {
                    rewritten,
                    kind: PowerProductRewriteKind::DistributePowerOverQuotient,
                });
            }
        }
    }
    None
}

/// Split a product's leaves into (exponents of base-`e` powers, the other factors).
/// A bare `e` contributes exponent `1`.
fn split_e_power_factors(ctx: &mut Context, leaves: &[ExprId]) -> (Vec<ExprId>, Vec<ExprId>) {
    let mut e_exps = Vec::new();
    let mut others = Vec::new();
    for &leaf in leaves {
        if is_e_constant_expr(ctx, leaf) {
            let one = ctx.num(1);
            e_exps.push(one);
        } else if let Some((base, exp)) = as_pow(ctx, leaf) {
            if is_e_constant_expr(ctx, base) {
                e_exps.push(exp);
            } else {
                others.push(leaf);
            }
        } else {
            others.push(leaf);
        }
    }
    (e_exps, others)
}

/// Sum a non-empty list of exponent expressions into a single additive chain.
fn sum_exponent_chain(ctx: &mut Context, exps: Vec<ExprId>) -> ExprId {
    let mut iter = exps.into_iter();
    let mut acc = iter
        .next()
        .expect("sum_exponent_chain requires a non-empty list");
    for exp in iter {
        acc = ctx.add(Expr::Add(acc, exp));
    }
    acc
}

/// Try exponential quotient rewrites for base `e`:
/// - `e^a / e^b -> e^(a-b)`
/// - `e / e^b -> e^(1-b)`
/// - `e^a / e -> e^(a-1)`
/// - `Mul(.., e^a) / e^b -> .. * e^(a-b)` (and the symmetric / both-product shapes): the same
///   combination when an `e` power sits inside a product on either side, so a co-factor no longer
///   blocks it (e.g. `x·e^(x²)/e^(2x²) -> x·e^(-x²)`, which keeps high-order exp derivatives legible).
pub fn try_rewrite_exp_quotient_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<PowerProductRewrite> {
    if let Some((num, den)) = as_div(ctx, expr) {
        let num_pow = as_pow(ctx, num);
        let den_pow = as_pow(ctx, den);

        if let (Some((num_base, num_exp)), Some((den_base, den_exp))) = (num_pow, den_pow) {
            if is_e_constant_expr(ctx, num_base) && is_e_constant_expr(ctx, den_base) {
                let diff = ctx.add(Expr::Sub(num_exp, den_exp));
                let e = ctx.add(Expr::Constant(Constant::E));
                let rewritten = ctx.add(Expr::Pow(e, diff));
                return Some(PowerProductRewrite {
                    rewritten,
                    kind: PowerProductRewriteKind::ExpQuotient,
                });
            }
        }

        if is_e_constant_expr(ctx, num) {
            if let Some((den_base, den_exp)) = den_pow {
                if is_e_constant_expr(ctx, den_base) {
                    let one = ctx.num(1);
                    let diff = ctx.add(Expr::Sub(one, den_exp));
                    let e = ctx.add(Expr::Constant(Constant::E));
                    let rewritten = ctx.add(Expr::Pow(e, diff));
                    return Some(PowerProductRewrite {
                        rewritten,
                        kind: PowerProductRewriteKind::ExpOverExpPower,
                    });
                }
            }
        }

        if is_e_constant_expr(ctx, den) {
            if let Some((num_base, num_exp)) = num_pow {
                if is_e_constant_expr(ctx, num_base) {
                    let one = ctx.num(1);
                    let diff = ctx.add(Expr::Sub(num_exp, one));
                    let e = ctx.add(Expr::Constant(Constant::E));
                    let rewritten = ctx.add(Expr::Pow(e, diff));
                    return Some(PowerProductRewrite {
                        rewritten,
                        kind: PowerProductRewriteKind::ExpPowerOverExp,
                    });
                }
            }
        }

        // Fallback: an `e` power wrapped in a product on either side. The pure single-power shapes
        // above have already returned, so this only fires when at least one side is a product
        // (`num_leaves.len() > 1 || den_leaves.len() > 1`). We require an `e` power on BOTH sides so
        // this is a genuine quotient combination (not a within-product merge), and we collect the
        // other factors verbatim. SOUNDNESS: `e^x ≠ 0` for all real `x`, so `e^a/e^b = e^(a-b)` is
        // unconditional — no domain gate (consistent with the pure-case branches above).
        let num_leaves = mul_leaves(ctx, num);
        let den_leaves = mul_leaves(ctx, den);
        if num_leaves.len() > 1 || den_leaves.len() > 1 {
            let (num_e_exps, mut num_others) = split_e_power_factors(ctx, &num_leaves);
            let (den_e_exps, den_others) = split_e_power_factors(ctx, &den_leaves);
            if !num_e_exps.is_empty() && !den_e_exps.is_empty() {
                let num_sum = sum_exponent_chain(ctx, num_e_exps);
                let den_sum = sum_exponent_chain(ctx, den_e_exps);
                let diff = ctx.add(Expr::Sub(num_sum, den_sum));
                let e = ctx.add(Expr::Constant(Constant::E));
                let e_pow = ctx.add(Expr::Pow(e, diff));
                num_others.push(e_pow);
                let new_num = build_balanced_mul(ctx, &num_others);
                let rewritten = if den_others.is_empty() {
                    new_num
                } else {
                    let new_den = build_balanced_mul(ctx, &den_others);
                    ctx.add(Expr::Div(new_num, new_den))
                };
                return Some(PowerProductRewrite {
                    rewritten,
                    kind: PowerProductRewriteKind::ExpQuotient,
                });
            }
        }
    }
    None
}

/// Try combining powers with the same base in n-ary multiplication trees.
pub fn try_rewrite_mul_nary_combine_powers_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<PowerProductRewrite> {
    if !matches!(ctx.get(expr), Expr::Mul(_, _)) {
        return None;
    }

    let factors = mul_leaves(ctx, expr);
    if factors.len() > 12 || factors.len() < 2 {
        return None;
    }

    let mut base_exp_pairs: Vec<(ExprId, Option<BigRational>, bool)> = Vec::new();
    for &factor in &factors {
        match ctx.get(factor) {
            Expr::Pow(base, exp) => {
                if let Expr::Number(n) = ctx.get(*exp) {
                    base_exp_pairs.push((*base, Some(n.clone()), true));
                } else {
                    base_exp_pairs.push((factor, None, false));
                }
            }
            Expr::Number(_) => {
                base_exp_pairs.push((factor, Some(BigRational::one()), false));
            }
            _ => {
                base_exp_pairs.push((factor, Some(BigRational::one()), true));
            }
        }
    }

    let mut combined: Vec<(ExprId, BigRational, usize)> = Vec::new();
    let mut absorbed = vec![false; factors.len()];
    let mut any_combined = false;

    for i in 0..base_exp_pairs.len() {
        if absorbed[i] {
            continue;
        }

        let (base_i, exp_i, is_pow_i) = &base_exp_pairs[i];
        let Some(mut sum_exp) = exp_i.clone() else {
            continue;
        };

        if !is_pow_i {
            continue;
        }

        let mut found_match = false;
        for j in (i + 1)..base_exp_pairs.len() {
            if absorbed[j] {
                continue;
            }

            let (base_j, exp_j, is_pow_j) = &base_exp_pairs[j];
            if !is_pow_j {
                continue;
            }

            let Some(exp_j_val) = exp_j else {
                continue;
            };

            if compare_expr(ctx, *base_i, *base_j) == Ordering::Equal {
                sum_exp += exp_j_val;
                absorbed[j] = true;
                found_match = true;
            }
        }

        if found_match {
            absorbed[i] = true;
            combined.push((*base_i, sum_exp, i));
            any_combined = true;
        }
    }

    if !any_combined {
        return None;
    }

    let mut result_factors: Vec<ExprId> = Vec::new();
    let mut combined_map: HashMap<usize, (ExprId, BigRational)> = HashMap::new();
    for (base, sum_exp, first_idx) in &combined {
        combined_map.insert(*first_idx, (*base, sum_exp.clone()));
    }

    for i in 0..factors.len() {
        if let Some((base, sum_exp)) = combined_map.get(&i) {
            let new_factor = if sum_exp.is_one() {
                *base
            } else if sum_exp.is_zero() {
                ctx.num(1)
            } else {
                let exp_id = ctx.add(Expr::Number(sum_exp.clone()));
                ctx.add(Expr::Pow(*base, exp_id))
            };
            result_factors.push(new_factor);
        } else if !absorbed[i] {
            result_factors.push(factors[i]);
        }
    }

    if result_factors.is_empty() {
        return Some(PowerProductRewrite {
            rewritten: ctx.num(1),
            kind: PowerProductRewriteKind::AllFactorsCancelled,
        });
    }

    let (&last, rest) = result_factors.split_last()?;
    let mut rewritten = last;
    for &factor in rest.iter().rev() {
        rewritten = mul2_raw(ctx, factor, rewritten);
    }

    if result_factors.len() < factors.len() {
        Some(PowerProductRewrite {
            rewritten,
            kind: PowerProductRewriteKind::SameBaseNary,
        })
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::{
        try_rewrite_exp_quotient_expr, try_rewrite_mul_nary_combine_powers_expr,
        try_rewrite_power_product_distribution_expr, try_rewrite_power_quotient_expr,
        try_rewrite_product_power_expr, try_rewrite_product_same_exponent_expr,
        try_rewrite_quotient_same_exponent_expr,
    };
    use cas_ast::Context;
    use cas_parser::parse;

    #[test]
    fn rewrites_product_power_same_base() {
        let mut ctx = Context::new();
        let expr = parse("x^2 * x^3", &mut ctx).expect("parse");
        assert!(try_rewrite_product_power_expr(&mut ctx, expr).is_some());
    }

    #[test]
    fn rewrites_product_same_exponent() {
        let mut ctx = Context::new();
        let expr = parse("2^x * 3^x", &mut ctx).expect("parse");
        assert!(try_rewrite_product_same_exponent_expr(&mut ctx, expr).is_some());
    }

    #[test]
    fn product_same_exponent_declines_even_root_with_negative_base() {
        // An even root of a negative is undefined over the reals, so the merge
        // must decline if EITHER base is a negative constant: sqrt(-2)*sqrt(-3)
        // must not become sqrt(6), and the mixed (-2)^(1/2)*3^(1/2) must not merge
        // either (gating on either side blocks the pairwise cascade that produced
        // the sign-wrong sqrt(-3)*sqrt(-5)*sqrt(x) -> (15x)^(1/2)).
        let mut ctx = Context::new();
        for src in [
            "(-2)^(1/2) * (-3)^(1/2)",
            "(-2)^(1/4) * (-8)^(1/4)",
            "(-2)^(1/2) * 3^(1/2)",
            "2^(1/2) * (-3)^(1/2)",
        ] {
            let expr = parse(src, &mut ctx).expect("parse");
            assert!(
                try_rewrite_product_same_exponent_expr(&mut ctx, expr).is_none(),
                "even root with a negative base must not merge: {src}"
            );
        }
    }

    #[test]
    fn product_same_exponent_still_merges_safe_cases() {
        // Positive bases, ODD-root of negatives, and symbolic bases stay mergeable.
        let mut ctx = Context::new();
        for src in [
            "2^(1/2) * 3^(1/2)",       // positive bases
            "(-2)^(1/3) * (-4)^(1/3)", // odd root: real, (8)^(1/3) = 2
            "(-2)^2 * (-3)^2",         // integer exponent: real
        ] {
            let expr = parse(src, &mut ctx).expect("parse");
            assert!(
                try_rewrite_product_same_exponent_expr(&mut ctx, expr).is_some(),
                "safe same-exponent product must still merge: {src}"
            );
        }
    }

    #[test]
    fn rewrites_quotient_same_exponent() {
        let mut ctx = Context::new();
        let expr = parse("2^x / 3^x", &mut ctx).expect("parse");
        assert!(try_rewrite_quotient_same_exponent_expr(&mut ctx, expr).is_some());
    }

    #[test]
    fn rewrites_power_quotient_distribution() {
        let mut ctx = Context::new();
        let expr = parse("(a/b)^3", &mut ctx).expect("parse");
        assert!(try_rewrite_power_quotient_expr(&mut ctx, expr).is_some());
    }

    #[test]
    fn rewrites_exp_quotient() {
        let mut ctx = Context::new();
        let expr = parse("e^a / e^b", &mut ctx).expect("parse");
        assert!(try_rewrite_exp_quotient_expr(&mut ctx, expr).is_some());
    }

    #[test]
    fn rewrites_mul_nary_combine_powers() {
        let mut ctx = Context::new();
        let expr = parse("x^2 * y * x^3", &mut ctx).expect("parse");
        assert!(try_rewrite_mul_nary_combine_powers_expr(&mut ctx, expr).is_some());
    }

    #[test]
    fn rewrites_power_product_distribution() {
        let mut ctx = Context::new();
        let expr = parse("(a*b)^3", &mut ctx).expect("parse");
        assert!(try_rewrite_power_product_distribution_expr(&mut ctx, expr).is_some());
    }

    #[test]
    fn declines_product_power_split_for_symbolic_exponent() {
        // `(a*b)^x -> a^x * b^x` is unsound for symbolic x unless both bases are
        // provably >= 0 (negative bases make a^x, b^x individually complex).
        for src in ["(a*b)^x", "(x*y)^n", "(a*b)^pi"] {
            let mut ctx = Context::new();
            let expr = parse(src, &mut ctx).expect("parse");
            assert!(
                try_rewrite_power_product_distribution_expr(&mut ctx, expr).is_none(),
                "`{src}` must NOT split for a symbolic exponent"
            );
        }
    }

    #[test]
    fn still_splits_integer_exponent_and_provably_nonnegative_bases() {
        // Integer exponents are universally safe; provably-non-negative bases
        // (even powers, abs, e) keep the symbolic-exponent split sound.
        for src in ["(a*b)^2", "(a*b)^3", "(x^2*y^2)^n", "(abs(a)*abs(b))^x"] {
            let mut ctx = Context::new();
            let expr = parse(src, &mut ctx).expect("parse");
            assert!(
                try_rewrite_power_product_distribution_expr(&mut ctx, expr).is_some(),
                "`{src}` should still split"
            );
        }
    }

    #[test]
    fn declines_power_quotient_split_for_symbolic_exponent() {
        // `(a/b)^x -> a^x / b^x` is unsound for a symbolic exponent unless both
        // parts are provably >= 0 — same domain constraint as the product split.
        for src in ["(a/b)^x", "(x/y)^n", "(a/b)^pi", "(a/b)^(2*n)"] {
            let mut ctx = Context::new();
            let expr = parse(src, &mut ctx).expect("parse");
            assert!(
                try_rewrite_power_quotient_expr(&mut ctx, expr).is_none(),
                "`{src}` must NOT split for a symbolic exponent"
            );
        }
        // Integer exponents and provably-non-negative parts still split.
        for src in ["(a/b)^2", "(a/b)^3", "(x^2/y^2)^n"] {
            let mut ctx = Context::new();
            let expr = parse(src, &mut ctx).expect("parse");
            assert!(
                try_rewrite_power_quotient_expr(&mut ctx, expr).is_some(),
                "`{src}` should still split"
            );
        }
    }
}
