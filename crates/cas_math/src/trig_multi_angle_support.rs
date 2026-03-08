use crate::expr_destructure::{as_mul, as_pow};
use crate::expr_nary::{add_terms_signed, Sign};
use crate::expr_rewrite::smart_mul;
use crate::numeric::as_number;
use crate::pattern_marks::PatternMarks;
use crate::pi_helpers::extract_rational_pi_multiple;
use crate::pi_helpers::is_provably_sin_nonzero;
use crate::trig_dyadic_policy_support::{
    decide_dyadic_sin_nonzero_policy, DyadicSinNonzeroPolicyDecision,
};
use crate::trig_roots_flatten::{
    extract_double_angle_arg, extract_quintuple_angle_arg, extract_triple_angle_arg,
};
use crate::trig_sum_product_support::extract_trig_arg;
use cas_ast::ordering::compare_expr;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{One, Zero};
use std::cmp::Ordering;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrigMultiAngleRewrite {
    pub rewritten: ExprId,
    pub kind: TrigMultiAngleRewriteKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrigMultiAngleRewriteKind {
    TripleSin,
    TripleCos,
    TripleTan,
    DoubleSin,
    DoubleCos,
    QuintupleSin,
    QuintupleCos,
    TripleContractionSin,
    TripleContractionCos,
    CanonicalizeCosSquared,
    CanonicalizeCosEvenPower,
    HalfAngleExpansion,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrigMultiAngleRewriteOwned {
    pub rewritten: ExprId,
    pub desc: String,
}

/// Check if a trig argument is "trivial": variable, constant, number,
/// or a simple numeric multiple of one of those.
pub fn is_trivial_angle(ctx: &Context, arg: ExprId) -> bool {
    match ctx.get(arg) {
        Expr::Variable(_) | Expr::Constant(_) | Expr::Number(_) => true,
        Expr::Mul(l, r) => {
            let l_simple = matches!(
                ctx.get(*l),
                Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_)
            );
            let r_simple = matches!(
                ctx.get(*r),
                Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_)
            );
            l_simple && r_simple
        }
        Expr::Neg(inner) => is_trivial_angle(ctx, *inner),
        _ => false,
    }
}

/// Rewrite shortcut for `sin(3x)`, `cos(3x)`, `tan(3x)`.
pub fn try_rewrite_triple_angle_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<TrigMultiAngleRewrite> {
    let (fn_id, arg) = match ctx.get(expr) {
        Expr::Function(fn_id, args) if args.len() == 1 => (*fn_id, args[0]),
        _ => return None,
    };

    let inner_var = extract_triple_angle_arg(ctx, arg)?;
    if !is_trivial_angle(ctx, inner_var) {
        return None;
    }

    match ctx.builtin_of(fn_id) {
        Some(BuiltinFn::Sin) => {
            let three = ctx.num(3);
            let four = ctx.num(4);
            let exp_three = ctx.num(3);
            let sin_x = ctx.call_builtin(BuiltinFn::Sin, vec![inner_var]);

            let term1 = smart_mul(ctx, three, sin_x);
            let sin_cubed = ctx.add(Expr::Pow(sin_x, exp_three));
            let term2 = smart_mul(ctx, four, sin_cubed);
            let rewritten = ctx.add(Expr::Sub(term1, term2));
            Some(TrigMultiAngleRewrite {
                rewritten,
                kind: TrigMultiAngleRewriteKind::TripleSin,
            })
        }
        Some(BuiltinFn::Cos) => {
            let three = ctx.num(3);
            let four = ctx.num(4);
            let exp_three = ctx.num(3);
            let cos_x = ctx.call_builtin(BuiltinFn::Cos, vec![inner_var]);

            let cos_cubed = ctx.add(Expr::Pow(cos_x, exp_three));
            let term1 = smart_mul(ctx, four, cos_cubed);
            let term2 = smart_mul(ctx, three, cos_x);
            let rewritten = ctx.add(Expr::Sub(term1, term2));
            Some(TrigMultiAngleRewrite {
                rewritten,
                kind: TrigMultiAngleRewriteKind::TripleCos,
            })
        }
        Some(BuiltinFn::Tan) => {
            let one = ctx.num(1);
            let three = ctx.num(3);
            let exp_two = ctx.num(2);
            let exp_three = ctx.num(3);
            let tan_x = ctx.call_builtin(BuiltinFn::Tan, vec![inner_var]);

            let three_tan = smart_mul(ctx, three, tan_x);
            let tan_cubed = ctx.add(Expr::Pow(tan_x, exp_three));
            let numer = ctx.add(Expr::Sub(three_tan, tan_cubed));

            let tan_squared = ctx.add(Expr::Pow(tan_x, exp_two));
            let three_tan_squared = smart_mul(ctx, three, tan_squared);
            let denom = ctx.add(Expr::Sub(one, three_tan_squared));

            let rewritten = ctx.add(Expr::Div(numer, denom));
            Some(TrigMultiAngleRewrite {
                rewritten,
                kind: TrigMultiAngleRewriteKind::TripleTan,
            })
        }
        _ => None,
    }
}

/// Rewrite double-angle shortcut forms:
/// - `sin(2x) -> 2sin(x)cos(x)`
/// - `cos(2x) -> cos^2(x) - sin^2(x)`
///
/// Preserves anti-worsen behavior by skipping nested multiple-angle arguments.
pub fn try_rewrite_double_angle_function_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<TrigMultiAngleRewrite> {
    let (fn_id, args) = match ctx.get(expr) {
        Expr::Function(fn_id, args) if args.len() == 1 => (*fn_id, args),
        _ => return None,
    };

    let inner_var = extract_double_angle_arg(ctx, args[0])?;
    if is_multiple_angle(ctx, inner_var) {
        return None;
    }

    match ctx.builtin_of(fn_id) {
        Some(BuiltinFn::Sin) => {
            let two = ctx.num(2);
            let sin_x = ctx.call_builtin(BuiltinFn::Sin, vec![inner_var]);
            let cos_x = ctx.call_builtin(BuiltinFn::Cos, vec![inner_var]);
            let sin_cos = smart_mul(ctx, sin_x, cos_x);
            let rewritten = smart_mul(ctx, two, sin_cos);
            Some(TrigMultiAngleRewrite {
                rewritten,
                kind: TrigMultiAngleRewriteKind::DoubleSin,
            })
        }
        Some(BuiltinFn::Cos) => {
            let two = ctx.num(2);
            let cos_x = ctx.call_builtin(BuiltinFn::Cos, vec![inner_var]);
            let cos2 = ctx.add(Expr::Pow(cos_x, two));

            let sin_x = ctx.call_builtin(BuiltinFn::Sin, vec![inner_var]);
            let sin2 = ctx.add(Expr::Pow(sin_x, two));
            let rewritten = ctx.add(Expr::Sub(cos2, sin2));
            Some(TrigMultiAngleRewrite {
                rewritten,
                kind: TrigMultiAngleRewriteKind::DoubleCos,
            })
        }
        _ => None,
    }
}

/// Unified policy gate for double-angle expansion (`sin(2x)`, `cos(2x)`).
///
/// Returns `true` when expansion should be blocked.
pub fn should_block_double_angle_expr(
    is_expand_mode: bool,
    is_inside_div_context: bool,
    has_sin4x_identity_pattern: bool,
) -> bool {
    !is_expand_mode || is_inside_div_context || has_sin4x_identity_pattern
}

/// Context-aware policy gate for double-angle expansion (`sin(2x)`, `cos(2x)`).
///
/// Returns `true` when expansion should be blocked.
pub fn should_block_double_angle_expr_with_context(
    ctx: &Context,
    is_expand_mode: bool,
    marks: Option<&PatternMarks>,
    ancestors: &[ExprId],
) -> bool {
    let is_inside_div_context = ancestors
        .iter()
        .copied()
        .any(|ancestor| matches!(ctx.get(ancestor), Expr::Div(_, _)));
    let has_sin4x_identity_pattern = marks.is_some_and(|m| m.has_sin4x_identity_pattern);
    should_block_double_angle_expr(
        is_expand_mode,
        is_inside_div_context,
        has_sin4x_identity_pattern,
    )
}

/// Unified policy gate for high-order trig expansions (`3x`, `5x`, recursive `nx`).
///
/// Returns `true` when expansion should be blocked.
pub fn should_block_high_order_trig_expansion_expr(
    ctx: &Context,
    expr: ExprId,
    marks: Option<&PatternMarks>,
    ancestors: &[ExprId],
    block_on_sin4x_pattern: bool,
) -> bool {
    if marks.is_some_and(|m| m.is_sum_quotient_protected(expr)) {
        return true;
    }
    if block_on_sin4x_pattern && marks.is_some_and(|m| m.has_sin4x_identity_pattern) {
        return true;
    }
    is_inside_trig_sum_quotient_with_ancestors(ctx, ancestors)
}

/// Rewrite shortcut for `sin(5x)` and `cos(5x)`.
pub fn try_rewrite_quintuple_angle_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<TrigMultiAngleRewrite> {
    let (fn_id, arg) = match ctx.get(expr) {
        Expr::Function(fn_id, args) if args.len() == 1 => (*fn_id, args[0]),
        _ => return None,
    };

    let inner_var = extract_quintuple_angle_arg(ctx, arg)?;
    match ctx.builtin_of(fn_id) {
        Some(BuiltinFn::Sin) => {
            let five = ctx.num(5);
            let sixteen = ctx.num(16);
            let twenty = ctx.num(20);
            let exp_three = ctx.num(3);
            let exp_five = ctx.num(5);
            let sin_x = ctx.call_builtin(BuiltinFn::Sin, vec![inner_var]);

            let sin_5 = ctx.add(Expr::Pow(sin_x, exp_five));
            let term1 = smart_mul(ctx, sixteen, sin_5);
            let sin_3 = ctx.add(Expr::Pow(sin_x, exp_three));
            let term2 = smart_mul(ctx, twenty, sin_3);
            let term3 = smart_mul(ctx, five, sin_x);
            let sub1 = ctx.add(Expr::Sub(term1, term2));
            let rewritten = ctx.add(Expr::Add(sub1, term3));
            Some(TrigMultiAngleRewrite {
                rewritten,
                kind: TrigMultiAngleRewriteKind::QuintupleSin,
            })
        }
        Some(BuiltinFn::Cos) => {
            let five = ctx.num(5);
            let sixteen = ctx.num(16);
            let twenty = ctx.num(20);
            let exp_three = ctx.num(3);
            let exp_five = ctx.num(5);
            let cos_x = ctx.call_builtin(BuiltinFn::Cos, vec![inner_var]);

            let cos_5 = ctx.add(Expr::Pow(cos_x, exp_five));
            let term1 = smart_mul(ctx, sixteen, cos_5);
            let cos_3 = ctx.add(Expr::Pow(cos_x, exp_three));
            let term2 = smart_mul(ctx, twenty, cos_3);
            let term3 = smart_mul(ctx, five, cos_x);
            let sub1 = ctx.add(Expr::Sub(term1, term2));
            let rewritten = ctx.add(Expr::Add(sub1, term3));
            Some(TrigMultiAngleRewrite {
                rewritten,
                kind: TrigMultiAngleRewriteKind::QuintupleCos,
            })
        }
        _ => None,
    }
}

/// Contract expanded triple-angle forms:
/// - `3*sin(θ) - 4*sin(θ)^3 -> sin(3θ)`
/// - `4*cos(θ)^3 - 3*cos(θ) -> cos(3θ)`
///
/// preserving a common scale factor `k` when present.
pub fn try_rewrite_triple_angle_contraction_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<TrigMultiAngleRewrite> {
    let signed_terms = add_terms_signed(ctx, expr);
    if signed_terms.len() < 2 {
        return None;
    }

    struct TrigTerm {
        index: usize,
        coeff: BigRational,
        builtin: BuiltinFn,
        arg: ExprId,
        power: i64,
    }

    fn decompose_trig_term(
        ctx: &Context,
        term: ExprId,
        sign: Sign,
    ) -> Option<(BigRational, BuiltinFn, ExprId, i64)> {
        let outer_sign = BigRational::from_integer(sign.to_i32().into());

        let (inner, neg_sign) = if let Expr::Neg(i) = ctx.get(term) {
            (*i, BigRational::from_integer((-1).into()))
        } else {
            (term, BigRational::from_integer(1.into()))
        };

        let (coeff, core) = match ctx.get(inner) {
            Expr::Mul(l, r) => {
                if let Expr::Number(n) = ctx.get(*l) {
                    (n.clone(), *r)
                } else if let Expr::Number(n) = ctx.get(*r) {
                    (n.clone(), *l)
                } else {
                    (BigRational::from_integer(1.into()), inner)
                }
            }
            _ => (BigRational::from_integer(1.into()), inner),
        };
        let final_coeff = outer_sign * neg_sign * coeff;

        let (base, power) = if let Expr::Pow(b, e) = ctx.get(core) {
            if let Expr::Number(n) = ctx.get(*e) {
                if n.is_integer() {
                    (*b, n.to_integer().try_into().ok().unwrap_or(0i64))
                } else {
                    return None;
                }
            } else {
                return None;
            }
        } else {
            (core, 1i64)
        };

        if let Expr::Function(fn_id, args) = ctx.get(base) {
            if args.len() == 1 {
                if let Some(b @ (BuiltinFn::Sin | BuiltinFn::Cos)) = ctx.builtin_of(*fn_id) {
                    return Some((final_coeff, b, args[0], power));
                }
            }
        }
        None
    }

    let mut trig_terms: Vec<TrigTerm> = Vec::new();
    for (i, (term, sign)) in signed_terms.iter().enumerate() {
        if let Some((c, b, a, p)) = decompose_trig_term(ctx, *term, *sign) {
            if p == 1 || p == 3 {
                trig_terms.push(TrigTerm {
                    index: i,
                    coeff: c,
                    builtin: b,
                    arg: a,
                    power: p,
                });
            }
        }
    }

    for i in 0..trig_terms.len() {
        for j in 0..trig_terms.len() {
            if i == j {
                continue;
            }

            let t1 = &trig_terms[i];
            let t3 = &trig_terms[j];
            if std::mem::discriminant(&t1.builtin) != std::mem::discriminant(&t3.builtin) {
                continue;
            }
            if t1.power != 1 || t3.power != 3 {
                continue;
            }
            if compare_expr(ctx, t1.arg, t3.arg) != Ordering::Equal {
                continue;
            }

            let three = BigRational::from_integer(3.into());
            let four = BigRational::from_integer(4.into());
            let matched = match t1.builtin {
                BuiltinFn::Sin => (&t1.coeff * &four) == (-(&t3.coeff) * &three),
                BuiltinFn::Cos => (&t3.coeff * &three) == (-(&t1.coeff) * &four),
                _ => false,
            };
            if !matched {
                continue;
            }

            if is_trivial_angle(ctx, t1.arg) {
                continue;
            }

            let scale = match t1.builtin {
                BuiltinFn::Sin => &t1.coeff / &three,
                BuiltinFn::Cos => &t3.coeff / &four,
                _ => continue,
            };

            let three_id = ctx.num(3);
            let triple_arg = smart_mul(ctx, three_id, t1.arg);
            let contracted = ctx.call_builtin(t1.builtin, vec![triple_arg]);

            let one = BigRational::from_integer(1.into());
            let neg_one = -&one;
            let scaled = if scale == one {
                contracted
            } else if scale == neg_one {
                ctx.add(Expr::Neg(contracted))
            } else {
                let scale_id = ctx.add(Expr::Number(scale));
                smart_mul(ctx, scale_id, contracted)
            };

            let kind = match t1.builtin {
                BuiltinFn::Sin => TrigMultiAngleRewriteKind::TripleContractionSin,
                BuiltinFn::Cos => TrigMultiAngleRewriteKind::TripleContractionCos,
                _ => continue,
            };

            if signed_terms.len() == 2 {
                return Some(TrigMultiAngleRewrite {
                    rewritten: scaled,
                    kind,
                });
            }

            let mut new_terms: Vec<ExprId> = Vec::new();
            for (k, (term, sign)) in signed_terms.iter().enumerate() {
                if k != t1.index && k != t3.index {
                    if *sign == Sign::Neg {
                        new_terms.push(ctx.add(Expr::Neg(*term)));
                    } else {
                        new_terms.push(*term);
                    }
                }
            }
            new_terms.push(scaled);

            let mut acc = new_terms[0];
            for &t in new_terms.iter().skip(1) {
                acc = ctx.add(Expr::Add(acc, t));
            }
            return Some(TrigMultiAngleRewrite {
                rewritten: acc,
                kind,
            });
        }
    }

    None
}

/// Canonicalize `cos^(2k)(x)` into `(1 - sin^2(x))^k` for small even powers.
pub fn try_rewrite_canonicalize_trig_square_pow_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<TrigMultiAngleRewrite> {
    let (base, exp) = as_pow(ctx, expr)?;
    let n = match ctx.get(exp) {
        Expr::Number(n) => n.clone(),
        _ => return None,
    };

    if !n.is_integer() || n.to_integer() % 2 != 0.into() || n <= BigRational::zero() {
        return None;
    }
    if n > BigRational::from_integer(4.into()) {
        return None;
    }

    let Expr::Function(fn_id, args) = ctx.get(base) else {
        return None;
    };
    if !matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Cos)) || args.len() != 1 {
        return None;
    }

    let arg = args[0];
    let one = ctx.num(1);
    let sin_x = ctx.call_builtin(BuiltinFn::Sin, vec![arg]);
    let two = ctx.num(2);
    let sin_sq = ctx.add(Expr::Pow(sin_x, two));
    let base_term = ctx.add(Expr::Sub(one, sin_sq));

    let half_n = n / BigRational::from_integer(2.into());
    if half_n.is_one() {
        return Some(TrigMultiAngleRewrite {
            rewritten: base_term,
            kind: TrigMultiAngleRewriteKind::CanonicalizeCosSquared,
        });
    }

    let half_n_expr = ctx.add(Expr::Number(half_n));
    let rewritten = ctx.add(Expr::Pow(base_term, half_n_expr));
    Some(TrigMultiAngleRewrite {
        rewritten,
        kind: TrigMultiAngleRewriteKind::CanonicalizeCosEvenPower,
    })
}

/// Expand `sin(n*x)` / `cos(n*x)` recursively for small integers `n` in `[3, 6]`.
pub fn try_rewrite_recursive_trig_expansion_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<TrigMultiAngleRewriteOwned> {
    let (fn_id, args) = match ctx.get(expr) {
        Expr::Function(fn_id, args) => (*fn_id, args.clone()),
        _ => return None,
    };

    let builtin = ctx.builtin_of(fn_id);
    let is_sin = matches!(builtin, Some(BuiltinFn::Sin));
    let is_cos = matches!(builtin, Some(BuiltinFn::Cos));
    if args.len() != 1 || (!is_sin && !is_cos) {
        return None;
    }

    let inner = args[0];
    let (n_val, x_val) = if let Some((l, r)) = as_mul(ctx, inner) {
        if let Expr::Number(n) = ctx.get(l) {
            if n.is_integer() {
                (n.to_integer(), r)
            } else {
                return None;
            }
        } else if let Expr::Number(n) = ctx.get(r) {
            if n.is_integer() {
                (n.to_integer(), l)
            } else {
                return None;
            }
        } else {
            return None;
        }
    } else {
        return None;
    };

    if n_val <= num_bigint::BigInt::from(2) || n_val > num_bigint::BigInt::from(6) {
        return None;
    }

    let n_minus_1 = n_val.clone() - 1;
    let n_minus_1_expr = ctx.add(Expr::Number(BigRational::from_integer(n_minus_1)));
    let term_nm1 = smart_mul(ctx, n_minus_1_expr, x_val);

    let sin_nm1 = ctx.call_builtin(BuiltinFn::Sin, vec![term_nm1]);
    let cos_nm1 = ctx.call_builtin(BuiltinFn::Cos, vec![term_nm1]);
    let sin_x = ctx.call_builtin(BuiltinFn::Sin, vec![x_val]);
    let cos_x = ctx.call_builtin(BuiltinFn::Cos, vec![x_val]);

    if is_sin {
        let t1 = smart_mul(ctx, sin_nm1, cos_x);
        let t2 = smart_mul(ctx, cos_nm1, sin_x);
        let rewritten = ctx.add(Expr::Add(t1, t2));
        return Some(TrigMultiAngleRewriteOwned {
            rewritten,
            desc: format!("sin({}x) expansion", n_val),
        });
    }

    let t1 = smart_mul(ctx, cos_nm1, cos_x);
    let t2 = smart_mul(ctx, sin_nm1, sin_x);
    let rewritten = ctx.add(Expr::Sub(t1, t2));
    Some(TrigMultiAngleRewriteOwned {
        rewritten,
        desc: format!("cos({}x) expansion", n_val),
    })
}

/// Rewrite helper used by engine `AngleConsistencyRule`.
///
/// Detects pairs of trig arguments `(A, B)` with `A = 2B`, then expands all
/// occurrences of `trig(A)` in `expr` to be consistent with `B`.
pub fn try_rewrite_angle_consistency_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<TrigMultiAngleRewrite> {
    match ctx.get(expr) {
        Expr::Add(_, _) | Expr::Sub(_, _) | Expr::Mul(_, _) | Expr::Div(_, _) => {}
        _ => return None,
    }

    let mut trig_args = Vec::new();
    collect_trig_args_recursive(ctx, expr, &mut trig_args);
    if trig_args.is_empty() {
        return None;
    }

    for i in 0..trig_args.len() {
        for j in 0..trig_args.len() {
            if i == j {
                continue;
            }
            let large_angle = trig_args[i];
            let small_angle = trig_args[j];
            if !is_double_angle_relation(ctx, large_angle, small_angle) {
                continue;
            }

            let rewritten = expand_trig_angle(ctx, expr, large_angle, small_angle);
            if rewritten != expr {
                return Some(TrigMultiAngleRewrite {
                    rewritten,
                    kind: TrigMultiAngleRewriteKind::HalfAngleExpansion,
                });
            }
        }
    }

    None
}

/// Check whether `arg` is a multiple-angle form `n*x` (or `x*n`)
/// with integer `|n| > 1`.
pub fn is_multiple_angle(ctx: &Context, arg: ExprId) -> bool {
    let Expr::Mul(l, r) = ctx.get(arg) else {
        return false;
    };

    let is_large_integer_factor = |id: ExprId| -> bool {
        if let Expr::Number(n) = ctx.get(id) {
            if n.is_integer() {
                let val = n.numer().clone();
                return val > 1.into() || val < (-1).into();
            }
        }
        false
    };

    is_large_integer_factor(*l) || is_large_integer_factor(*r)
}

/// Check whether `arg` has a large trig coefficient.
///
/// Preserves engine behavior:
/// - `n*x` with integer `|n| > 2` is considered large.
/// - For `a+b` / `a-b`, if either side is a multiple-angle (`|n| > 1`),
///   it is considered large.
pub fn has_large_coefficient(ctx: &Context, arg: ExprId) -> bool {
    if let Expr::Mul(l, r) = ctx.get(arg) {
        let is_very_large_integer_factor = |id: ExprId| -> bool {
            if let Expr::Number(n) = ctx.get(id) {
                if n.is_integer() {
                    let val = n.numer().clone();
                    return val > 2.into() || val < (-2).into();
                }
            }
            false
        };

        if is_very_large_integer_factor(*l) || is_very_large_integer_factor(*r) {
            return true;
        }
    }

    if let Expr::Add(lhs, rhs) | Expr::Sub(lhs, rhs) = ctx.get(arg) {
        return is_multiple_angle(ctx, *lhs) || is_multiple_angle(ctx, *rhs);
    }

    false
}

/// Check if `expr` is a binary trig add/sub operation:
/// - `Add(trig(A), trig(B))`
/// - `Sub(trig(A), trig(B))`
/// - `Add(trig(A), Neg(trig(B)))`
pub fn is_binary_trig_op(ctx: &Context, expr: ExprId, fn_name: &str) -> bool {
    match ctx.get(expr) {
        Expr::Add(l, r) => {
            if extract_trig_arg(ctx, *l, fn_name).is_some()
                && extract_trig_arg(ctx, *r, fn_name).is_some()
            {
                return true;
            }
            if let Expr::Neg(inner) = ctx.get(*r) {
                if extract_trig_arg(ctx, *l, fn_name).is_some()
                    && extract_trig_arg(ctx, *inner, fn_name).is_some()
                {
                    return true;
                }
            }
            if let Expr::Neg(inner) = ctx.get(*l) {
                if extract_trig_arg(ctx, *r, fn_name).is_some()
                    && extract_trig_arg(ctx, *inner, fn_name).is_some()
                {
                    return true;
                }
            }
            false
        }
        Expr::Sub(l, r) => {
            extract_trig_arg(ctx, *l, fn_name).is_some()
                && extract_trig_arg(ctx, *r, fn_name).is_some()
        }
        _ => false,
    }
}

/// Check if `expr` is `Add(trig(A), trig(B))`.
pub fn is_trig_sum(ctx: &Context, expr: ExprId, fn_name: &str) -> bool {
    if let Expr::Add(l, r) = ctx.get(expr) {
        return extract_trig_arg(ctx, *l, fn_name).is_some()
            && extract_trig_arg(ctx, *r, fn_name).is_some();
    }
    false
}

/// Check whether an expression matches a trig sum-quotient scaffold:
/// `(sin(A) ± sin(B)) / (cos(C) + cos(D))`.
pub fn is_trig_sum_quotient_div_pattern(ctx: &Context, expr: ExprId) -> bool {
    let Expr::Div(num, den) = ctx.get(expr) else {
        return false;
    };

    let num_is_sin_sum_or_diff = is_binary_trig_op(ctx, *num, "sin");
    let den_is_cos_sum = is_trig_sum(ctx, *den, "cos");
    num_is_sin_sum_or_diff && den_is_cos_sum
}

/// Check whether any ancestor expression matches a trig sum-quotient scaffold.
pub fn is_inside_trig_sum_quotient_with_ancestors(ctx: &Context, ancestors: &[ExprId]) -> bool {
    ancestors
        .iter()
        .copied()
        .any(|ancestor| is_trig_sum_quotient_div_pattern(ctx, ancestor))
}

/// Check whether `large` and `small` satisfy a double-angle relation.
///
/// Recognized shapes:
/// - `large = 2 * small`
/// - `small = large / 2`
/// - `small = (1/2) * large`
pub fn is_double_angle_relation(ctx: &Context, large: ExprId, small: ExprId) -> bool {
    // Case 1: large = 2 * small
    if let Expr::Mul(l, r) = ctx.get(large) {
        if let Expr::Number(n) = ctx.get(*l) {
            if n == &num_rational::BigRational::from_integer(2.into())
                && compare_expr(ctx, *r, small) == Ordering::Equal
            {
                return true;
            }
        }
        if let Expr::Number(n) = ctx.get(*r) {
            if n == &num_rational::BigRational::from_integer(2.into())
                && compare_expr(ctx, *l, small) == Ordering::Equal
            {
                return true;
            }
        }
    }

    // Case 2: small = large / 2
    if let Expr::Div(n, d) = ctx.get(small) {
        if let Expr::Number(val) = ctx.get(*d) {
            if val == &num_rational::BigRational::from_integer(2.into())
                && compare_expr(ctx, *n, large) == Ordering::Equal
            {
                return true;
            }
        }
    }

    // Case 3: small = (1/2) * large
    if let Expr::Mul(l, r) = ctx.get(small) {
        if let Expr::Number(n) = ctx.get(*l) {
            if n == &num_rational::BigRational::new(1.into(), 2.into())
                && compare_expr(ctx, *r, large) == Ordering::Equal
            {
                return true;
            }
        }
        if let Expr::Number(n) = ctx.get(*r) {
            if n == &num_rational::BigRational::new(1.into(), 2.into())
                && compare_expr(ctx, *l, large) == Ordering::Equal
            {
                return true;
            }
        }
    }

    false
}

/// Verify that trig args form a dyadic sequence: `theta, 2*theta, 4*theta, ...`.
///
/// This matcher is robust to normalization by comparing extracted rational
/// coefficients in `k*pi` space rather than raw AST shapes.
pub fn verify_dyadic_pi_sequence(ctx: &Context, theta: ExprId, trig_args: &[ExprId]) -> bool {
    let n = trig_args.len() as u32;
    if n == 0 {
        return false;
    }

    let base_coeff = match extract_rational_pi_multiple(ctx, theta) {
        Some(k) => k,
        None => return false,
    };

    let mut coeffs: Vec<BigRational> = Vec::with_capacity(n as usize);
    for &arg in trig_args {
        match extract_rational_pi_multiple(ctx, arg) {
            Some(k) => coeffs.push(k),
            None => return false,
        }
    }

    let mut expected: Vec<BigRational> = Vec::with_capacity(n as usize);
    for k in 0..n {
        let multiplier = BigRational::from_integer((1u64 << k).into());
        expected.push(&base_coeff * &multiplier);
    }

    let mut used = vec![false; expected.len()];
    for coeff in &coeffs {
        let mut found = false;
        for (i, exp) in expected.iter().enumerate() {
            if !used[i] && coeff == exp {
                used[i] = true;
                found = true;
                break;
            }
        }
        if !found {
            return false;
        }
    }

    used.iter().all(|&u| u)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DyadicCosProductPlan {
    pub rewritten: ExprId,
    pub theta: ExprId,
    pub sin_theta: ExprId,
    pub n: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DyadicCosProductPolicyPlan {
    pub rewritten: ExprId,
    pub sin_theta: ExprId,
    pub n: u32,
    pub policy: DyadicSinNonzeroPolicyDecision,
}

/// Plan dyadic cosine product contraction:
/// `2^n · Π_{k=0}^{n-1} cos(2^k·θ) -> sin(2^n·θ) / sin(θ)`.
///
/// This helper performs structural matching and builds the rewritten expression.
/// Domain assumptions (e.g. `sin(θ) != 0`) are intentionally left to the engine.
pub fn try_plan_dyadic_cos_product_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<DyadicCosProductPlan> {
    let Expr::Mul(_, _) = ctx.get(expr) else {
        return None;
    };

    let factors = crate::expr_nary::mul_leaves(ctx, expr);

    let mut numeric_coeff = BigRational::one();
    let mut cos_args: Vec<ExprId> = Vec::new();
    let mut other_factors: Vec<ExprId> = Vec::new();

    for &factor in &factors {
        if let Some(n) = as_number(ctx, factor) {
            numeric_coeff *= n.clone();
        } else if let Expr::Function(fn_id, args) = ctx.get(factor) {
            if matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Cos)) && args.len() == 1 {
                cos_args.push(args[0]);
            } else {
                other_factors.push(factor);
            }
        } else {
            other_factors.push(factor);
        }
    }

    if !other_factors.is_empty() || cos_args.is_empty() {
        return None;
    }

    let n = cos_args.len() as u32;
    let expected_coeff = BigRational::from_integer(num_bigint::BigInt::from(1u64 << n));
    if numeric_coeff != expected_coeff {
        return None;
    }

    let mut theta: Option<ExprId> = None;
    for &candidate in &cos_args {
        if verify_dyadic_pi_sequence(ctx, candidate, &cos_args) {
            theta = Some(candidate);
            break;
        }
    }
    let theta = theta?;

    let two_pow_n = ctx.num((1u64 << n) as i64);
    let scaled_theta = smart_mul(ctx, two_pow_n, theta);
    let sin_scaled = ctx.call_builtin(BuiltinFn::Sin, vec![scaled_theta]);
    let sin_theta = ctx.call_builtin(BuiltinFn::Sin, vec![theta]);
    let rewritten = ctx.add(Expr::Div(sin_scaled, sin_theta));

    Some(DyadicCosProductPlan {
        rewritten,
        theta,
        sin_theta,
        n,
    })
}

/// Plan dyadic cosine product rewrite plus domain policy for `sin(theta) != 0`.
pub fn try_plan_dyadic_cos_product_with_policy(
    ctx: &mut Context,
    expr: ExprId,
    assume_mode: bool,
    strict_mode: bool,
) -> Option<DyadicCosProductPolicyPlan> {
    let plan = try_plan_dyadic_cos_product_expr(ctx, expr)?;
    let sin_nonzero_proven = is_provably_sin_nonzero(ctx, plan.theta);
    let policy = decide_dyadic_sin_nonzero_policy(assume_mode, strict_mode, sin_nonzero_proven);

    Some(DyadicCosProductPolicyPlan {
        rewritten: plan.rewritten,
        sin_theta: plan.sin_theta,
        n: plan.n,
        policy,
    })
}

/// Collect all arguments from unary `sin`, `cos` and `tan` calls in an expression tree.
pub fn collect_trig_args_recursive(ctx: &Context, expr: ExprId, args: &mut Vec<ExprId>) {
    match ctx.get(expr) {
        Expr::Function(fn_id, fargs) => {
            if matches!(
                ctx.builtin_of(*fn_id),
                Some(BuiltinFn::Sin | BuiltinFn::Cos | BuiltinFn::Tan)
            ) && fargs.len() == 1
            {
                args.push(fargs[0]);
            }
            for arg in fargs {
                collect_trig_args_recursive(ctx, *arg, args);
            }
        }
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) | Expr::Pow(l, r) => {
            collect_trig_args_recursive(ctx, *l, args);
            collect_trig_args_recursive(ctx, *r, args);
        }
        Expr::Neg(e) => collect_trig_args_recursive(ctx, *e, args),
        _ => {}
    }
}

/// Expand `sin/cos/tan(large_angle)` nodes to half-angle forms using `small_angle`.
pub fn expand_trig_angle(
    ctx: &mut Context,
    expr: ExprId,
    large_angle: ExprId,
    small_angle: ExprId,
) -> ExprId {
    if let Expr::Function(fn_id, args) = ctx.get(expr) {
        if args.len() == 1 && compare_expr(ctx, args[0], large_angle) == Ordering::Equal {
            let fn_id = *fn_id;
            match ctx.builtin_of(fn_id) {
                Some(BuiltinFn::Sin) => {
                    let two = ctx.num(2);
                    let sin_half = ctx.call_builtin(BuiltinFn::Sin, vec![small_angle]);
                    let cos_half = ctx.call_builtin(BuiltinFn::Cos, vec![small_angle]);
                    let term = smart_mul(ctx, sin_half, cos_half);
                    return smart_mul(ctx, two, term);
                }
                Some(BuiltinFn::Cos) => {
                    let two = ctx.num(2);
                    let one = ctx.num(1);
                    let cos_half = ctx.call_builtin(BuiltinFn::Cos, vec![small_angle]);
                    let cos_sq = ctx.add(Expr::Pow(cos_half, two));
                    let term = smart_mul(ctx, two, cos_sq);
                    return ctx.add(Expr::Sub(term, one));
                }
                Some(BuiltinFn::Tan) => {
                    let two = ctx.num(2);
                    let one = ctx.num(1);
                    let tan_half = ctx.call_builtin(BuiltinFn::Tan, vec![small_angle]);
                    let num = smart_mul(ctx, two, tan_half);
                    let tan_sq = ctx.add(Expr::Pow(tan_half, two));
                    let den = ctx.add(Expr::Sub(one, tan_sq));
                    return ctx.add(Expr::Div(num, den));
                }
                _ => {}
            }
        }
    }

    enum Shape {
        Add(ExprId, ExprId),
        Sub(ExprId, ExprId),
        Mul(ExprId, ExprId),
        Div(ExprId, ExprId),
        Pow(ExprId, ExprId),
        Neg(ExprId),
        Func(usize, Vec<ExprId>),
        Other,
    }

    let shape = match ctx.get(expr) {
        Expr::Add(l, r) => Shape::Add(*l, *r),
        Expr::Sub(l, r) => Shape::Sub(*l, *r),
        Expr::Mul(l, r) => Shape::Mul(*l, *r),
        Expr::Div(l, r) => Shape::Div(*l, *r),
        Expr::Pow(b, e) => Shape::Pow(*b, *e),
        Expr::Neg(e) => Shape::Neg(*e),
        Expr::Function(fn_id, args) => Shape::Func(*fn_id, args.clone()),
        _ => Shape::Other,
    };

    match shape {
        Shape::Add(l, r) => {
            let nl = expand_trig_angle(ctx, l, large_angle, small_angle);
            let nr = expand_trig_angle(ctx, r, large_angle, small_angle);
            if nl != l || nr != r {
                ctx.add(Expr::Add(nl, nr))
            } else {
                expr
            }
        }
        Shape::Sub(l, r) => {
            let nl = expand_trig_angle(ctx, l, large_angle, small_angle);
            let nr = expand_trig_angle(ctx, r, large_angle, small_angle);
            if nl != l || nr != r {
                ctx.add(Expr::Sub(nl, nr))
            } else {
                expr
            }
        }
        Shape::Mul(l, r) => {
            let nl = expand_trig_angle(ctx, l, large_angle, small_angle);
            let nr = expand_trig_angle(ctx, r, large_angle, small_angle);
            if nl != l || nr != r {
                smart_mul(ctx, nl, nr)
            } else {
                expr
            }
        }
        Shape::Div(l, r) => {
            let nl = expand_trig_angle(ctx, l, large_angle, small_angle);
            let nr = expand_trig_angle(ctx, r, large_angle, small_angle);
            if nl != l || nr != r {
                ctx.add(Expr::Div(nl, nr))
            } else {
                expr
            }
        }
        Shape::Pow(b, e) => {
            let nb = expand_trig_angle(ctx, b, large_angle, small_angle);
            let ne = expand_trig_angle(ctx, e, large_angle, small_angle);
            if nb != b || ne != e {
                ctx.add(Expr::Pow(nb, ne))
            } else {
                expr
            }
        }
        Shape::Neg(e) => {
            let ne = expand_trig_angle(ctx, e, large_angle, small_angle);
            if ne != e {
                ctx.add(Expr::Neg(ne))
            } else {
                expr
            }
        }
        Shape::Func(fn_id, args) => {
            let mut new_args = Vec::with_capacity(args.len());
            let mut changed = false;
            for arg in args {
                let na = expand_trig_angle(ctx, arg, large_angle, small_angle);
                if na != arg {
                    changed = true;
                }
                new_args.push(na);
            }
            if changed {
                ctx.add(Expr::Function(fn_id, new_args))
            } else {
                expr
            }
        }
        Shape::Other => expr,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    #[test]
    fn trivial_angle_detection_handles_basic_shapes() {
        let mut ctx = Context::new();
        let var = parse("x", &mut ctx).expect("x");
        let mul = parse("2*x", &mut ctx).expect("2*x");
        let neg = parse("-pi", &mut ctx).expect("-pi");
        let complex = parse("x+y", &mut ctx).expect("x+y");

        assert!(is_trivial_angle(&ctx, var));
        assert!(is_trivial_angle(&ctx, mul));
        assert!(is_trivial_angle(&ctx, neg));
        assert!(!is_trivial_angle(&ctx, complex));
    }

    #[test]
    fn triple_angle_rewrite_matches_sin_shortcut() {
        let mut ctx = Context::new();
        let expr = parse("sin(3*x)", &mut ctx).expect("expr");
        let expected = parse("3*sin(x) - 4*sin(x)^3", &mut ctx).expect("expected");
        let rewrite = try_rewrite_triple_angle_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(
            compare_expr(&ctx, rewrite.rewritten, expected),
            Ordering::Equal
        );
    }

    #[test]
    fn triple_angle_rewrite_blocks_non_trivial_argument() {
        let mut ctx = Context::new();
        let expr = parse("sin(3*(x+y))", &mut ctx).expect("expr");
        assert!(try_rewrite_triple_angle_expr(&mut ctx, expr).is_none());
    }

    #[test]
    fn double_angle_rewrite_matches_sin_expansion() {
        let mut ctx = Context::new();
        let expr = parse("sin(2*x)", &mut ctx).expect("expr");
        let rewrite = try_rewrite_double_angle_function_expr(&mut ctx, expr).expect("rewrite");
        let rewritten_str = format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: &ctx,
                id: rewrite.rewritten
            }
        );
        assert_eq!(rewritten_str, "2 * sin(x) * cos(x)");
    }

    #[test]
    fn double_angle_rewrite_skips_nested_multiple_angle() {
        let mut ctx = Context::new();
        let expr = parse("sin(2*(8*x))", &mut ctx).expect("expr");
        assert!(try_rewrite_double_angle_function_expr(&mut ctx, expr).is_none());
    }

    #[test]
    fn double_angle_policy_gate_matches_expected_blocks() {
        assert!(should_block_double_angle_expr(false, false, false));
        assert!(should_block_double_angle_expr(true, true, false));
        assert!(should_block_double_angle_expr(true, false, true));
        assert!(!should_block_double_angle_expr(true, false, false));
    }

    #[test]
    fn double_angle_context_gate_blocks_div_ancestor() {
        let mut ctx = Context::new();
        let expr = parse("sin(2*x)", &mut ctx).expect("expr");
        let div_ancestor = parse("sin(2*x)/cos(2*x)", &mut ctx).expect("ancestor");
        assert!(should_block_double_angle_expr_with_context(
            &ctx,
            true,
            None,
            &[expr, div_ancestor]
        ));
    }

    #[test]
    fn quintuple_angle_rewrite_matches_cos_shortcut() {
        let mut ctx = Context::new();
        let expr = parse("cos(5*x)", &mut ctx).expect("expr");
        let expected = parse("16*cos(x)^5 - 20*cos(x)^3 + 5*cos(x)", &mut ctx).expect("expected");
        let rewrite = try_rewrite_quintuple_angle_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(
            compare_expr(&ctx, rewrite.rewritten, expected),
            Ordering::Equal
        );
    }

    #[test]
    fn triple_angle_contraction_rewrites_nontrivial_argument() {
        let mut ctx = Context::new();
        let expr = parse("3*sin(x+y) - 4*sin(x+y)^3", &mut ctx).expect("expr");
        let expected = parse("sin(3*(x+y))", &mut ctx).expect("expected");
        let rewrite = try_rewrite_triple_angle_contraction_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(
            compare_expr(&ctx, rewrite.rewritten, expected),
            Ordering::Equal
        );
    }

    #[test]
    fn triple_angle_contraction_skips_trivial_argument() {
        let mut ctx = Context::new();
        let expr = parse("3*sin(x) - 4*sin(x)^3", &mut ctx).expect("expr");
        assert!(try_rewrite_triple_angle_contraction_expr(&mut ctx, expr).is_none());
    }

    #[test]
    fn canonicalize_trig_square_rewrites_cos_squared() {
        let mut ctx = Context::new();
        let expr = parse("cos(x)^2", &mut ctx).expect("expr");
        let expected = parse("1 - sin(x)^2", &mut ctx).expect("expected");
        let rewrite =
            try_rewrite_canonicalize_trig_square_pow_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(
            compare_expr(&ctx, rewrite.rewritten, expected),
            Ordering::Equal
        );
    }

    #[test]
    fn canonicalize_trig_square_rewrites_cos_fourth() {
        let mut ctx = Context::new();
        let expr = parse("cos(x)^4", &mut ctx).expect("expr");
        let expected = parse("(1 - sin(x)^2)^2", &mut ctx).expect("expected");
        let rewrite =
            try_rewrite_canonicalize_trig_square_pow_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(
            compare_expr(&ctx, rewrite.rewritten, expected),
            Ordering::Equal
        );
    }

    #[test]
    fn recursive_trig_expansion_rewrites_sin_four_x() {
        let mut ctx = Context::new();
        let expr = parse("sin(4*x)", &mut ctx).expect("expr");
        let rewrite = try_rewrite_recursive_trig_expansion_expr(&mut ctx, expr).expect("rewrite");
        assert_ne!(rewrite.rewritten, expr);
        assert!(rewrite.desc.contains("sin(4x) expansion"));
    }

    #[test]
    fn recursive_trig_expansion_skips_large_n() {
        let mut ctx = Context::new();
        let expr = parse("sin(7*x)", &mut ctx).expect("expr");
        assert!(try_rewrite_recursive_trig_expansion_expr(&mut ctx, expr).is_none());
    }

    #[test]
    fn binary_trig_op_matches_add_sub_and_add_neg_forms() {
        let mut ctx = Context::new();
        let add = parse("sin(a)+sin(b)", &mut ctx).expect("add");
        let sub = parse("sin(a)-sin(b)", &mut ctx).expect("sub");
        let add_neg = parse("sin(a)+(-sin(b))", &mut ctx).expect("add_neg");
        let mixed = parse("sin(a)+cos(b)", &mut ctx).expect("mixed");

        assert!(is_binary_trig_op(&ctx, add, "sin"));
        assert!(is_binary_trig_op(&ctx, sub, "sin"));
        assert!(is_binary_trig_op(&ctx, add_neg, "sin"));
        assert!(!is_binary_trig_op(&ctx, mixed, "sin"));
    }

    #[test]
    fn trig_sum_matches_only_add_of_same_trig_family() {
        let mut ctx = Context::new();
        let sum = parse("cos(a)+cos(b)", &mut ctx).expect("sum");
        let diff = parse("cos(a)-cos(b)", &mut ctx).expect("diff");

        assert!(is_trig_sum(&ctx, sum, "cos"));
        assert!(!is_trig_sum(&ctx, diff, "cos"));
    }

    #[test]
    fn trig_sum_quotient_div_pattern_detects_expected_shape() {
        let mut ctx = Context::new();
        let yes = parse("(sin(a)-sin(b))/(cos(a)+cos(b))", &mut ctx).expect("yes");
        let no_den = parse("(sin(a)-sin(b))/(cos(a)-cos(b))", &mut ctx).expect("no_den");
        let no_num = parse("(sin(a)*sin(b))/(cos(a)+cos(b))", &mut ctx).expect("no_num");

        assert!(is_trig_sum_quotient_div_pattern(&ctx, yes));
        assert!(!is_trig_sum_quotient_div_pattern(&ctx, no_den));
        assert!(!is_trig_sum_quotient_div_pattern(&ctx, no_num));
    }

    #[test]
    fn inside_trig_sum_quotient_detects_matching_ancestor() {
        let mut ctx = Context::new();
        let yes = parse("(sin(a)-sin(b))/(cos(a)+cos(b))", &mut ctx).expect("yes");
        let leaf = parse("sin(a)", &mut ctx).expect("leaf");

        assert!(is_inside_trig_sum_quotient_with_ancestors(
            &ctx,
            &[leaf, yes]
        ));
    }

    #[test]
    fn inside_trig_sum_quotient_rejects_non_matching_ancestors() {
        let mut ctx = Context::new();
        let no = parse("(sin(a)-sin(b))/(cos(a)-cos(b))", &mut ctx).expect("no");
        let leaf = parse("sin(a)", &mut ctx).expect("leaf");

        assert!(!is_inside_trig_sum_quotient_with_ancestors(
            &ctx,
            &[leaf, no]
        ));
    }

    #[test]
    fn multiple_angle_detection_matches_integer_multiplier_policy() {
        let mut ctx = Context::new();
        let two_x = parse("2*x", &mut ctx).expect("2*x");
        let neg_three_x = parse("-3*x", &mut ctx).expect("-3*x");
        let half_x = parse("x/2", &mut ctx).expect("x/2");
        let one_x = parse("1*x", &mut ctx).expect("1*x");

        assert!(is_multiple_angle(&ctx, two_x));
        assert!(is_multiple_angle(&ctx, neg_three_x));
        assert!(!is_multiple_angle(&ctx, half_x));
        assert!(!is_multiple_angle(&ctx, one_x));
    }

    #[test]
    fn large_coefficient_detection_matches_existing_engine_behavior() {
        let mut ctx = Context::new();
        let three_x = parse("3*x", &mut ctx).expect("3*x");
        let two_x = parse("2*x", &mut ctx).expect("2*x");
        let sum_with_multiple = parse("x + 2*y", &mut ctx).expect("x+2*y");
        let simple_sum = parse("x + y", &mut ctx).expect("x+y");

        assert!(has_large_coefficient(&ctx, three_x));
        assert!(!has_large_coefficient(&ctx, two_x));
        assert!(has_large_coefficient(&ctx, sum_with_multiple));
        assert!(!has_large_coefficient(&ctx, simple_sum));
    }

    #[test]
    fn double_angle_relation_matches_all_supported_forms() {
        let mut ctx = Context::new();
        let two_x = parse("2*x", &mut ctx).expect("2*x");
        let x = parse("x", &mut ctx).expect("x");
        let x_over_2 = parse("x/2", &mut ctx).expect("x/2");
        let three_x = parse("3*x", &mut ctx).expect("3*x");

        assert!(is_double_angle_relation(&ctx, two_x, x));
        assert!(is_double_angle_relation(&ctx, x, x_over_2));
        assert!(!is_double_angle_relation(&ctx, three_x, x));
    }

    #[test]
    fn dyadic_pi_sequence_detection_matches_expected_patterns() {
        let mut ctx = Context::new();
        let theta = parse("pi/9", &mut ctx).expect("theta");
        let args = vec![
            parse("2*pi/9", &mut ctx).expect("2pi/9"),
            parse("4*pi/9", &mut ctx).expect("4pi/9"),
            parse("pi/9", &mut ctx).expect("pi/9"),
        ];
        let wrong = vec![
            parse("pi/9", &mut ctx).expect("pi/9"),
            parse("3*pi/9", &mut ctx).expect("3pi/9"),
        ];

        assert!(verify_dyadic_pi_sequence(&ctx, theta, &args));
        assert!(!verify_dyadic_pi_sequence(&ctx, theta, &wrong));
    }

    #[test]
    fn dyadic_cos_product_plan_matches_valid_product() {
        let mut ctx = Context::new();
        let expr = parse("8*cos(pi/9)*cos(2*pi/9)*cos(4*pi/9)", &mut ctx).expect("expr");
        let plan = try_plan_dyadic_cos_product_expr(&mut ctx, expr).expect("plan");
        assert_eq!(plan.n, 3);
        let Expr::Div(num, den) = ctx.get(plan.rewritten) else {
            panic!("expected division rewrite");
        };
        assert_eq!(compare_expr(&ctx, *den, plan.sin_theta), Ordering::Equal);

        let Expr::Function(fn_id, args) = ctx.get(*num) else {
            panic!("expected sin(...) numerator");
        };
        assert!(ctx.is_builtin(*fn_id, BuiltinFn::Sin));
        assert_eq!(args.len(), 1);

        let scaled = args[0];
        let Some((l, r)) = as_mul(&ctx, scaled) else {
            panic!("expected scaled angle as multiplication");
        };
        let is_expected_scale = |n_id: ExprId, t_id: ExprId| -> bool {
            if let Expr::Number(n) = ctx.get(n_id) {
                return *n == num_rational::BigRational::from_integer(8.into())
                    && compare_expr(&ctx, t_id, plan.theta) == Ordering::Equal;
            }
            false
        };
        assert!(is_expected_scale(l, r) || is_expected_scale(r, l));
    }

    #[test]
    fn dyadic_cos_product_plan_rejects_wrong_coefficient() {
        let mut ctx = Context::new();
        let expr = parse("4*cos(pi/9)*cos(2*pi/9)*cos(4*pi/9)", &mut ctx).expect("expr");
        assert!(try_plan_dyadic_cos_product_expr(&mut ctx, expr).is_none());
    }

    #[test]
    fn dyadic_cos_product_policy_plan_blocks_unproven_in_strict() {
        let mut ctx = Context::new();
        let expr = parse("8*cos(pi)*cos(2*pi)*cos(4*pi)", &mut ctx).expect("expr");
        let plan = try_plan_dyadic_cos_product_with_policy(&mut ctx, expr, false, true)
            .expect("policy plan");
        assert_eq!(plan.policy, DyadicSinNonzeroPolicyDecision::Block);
        assert_eq!(plan.n, 3);
    }

    #[test]
    fn high_order_expansion_policy_blocks_marked_expr() {
        let mut ctx = Context::new();
        let expr = parse("sin(3*x)", &mut ctx).expect("expr");

        let mut marks = PatternMarks::default();
        marks.mark_sum_quotient(expr);

        assert!(should_block_high_order_trig_expansion_expr(
            &ctx,
            expr,
            Some(&marks),
            &[],
            false
        ));
    }

    #[test]
    fn high_order_expansion_policy_blocks_inside_sum_quotient_ancestor() {
        let mut ctx = Context::new();
        let expr = parse("sin(3*x)", &mut ctx).expect("expr");
        let ancestor = parse("(sin(a)-sin(b))/(cos(a)+cos(b))", &mut ctx).expect("ancestor");
        assert!(should_block_high_order_trig_expansion_expr(
            &ctx,
            expr,
            None,
            &[ancestor],
            false
        ));
    }

    #[test]
    fn high_order_expansion_policy_blocks_sin4x_when_enabled() {
        let mut ctx = Context::new();
        let expr = parse("sin(4*x)", &mut ctx).expect("expr");
        let marks = PatternMarks {
            has_sin4x_identity_pattern: true,
            ..PatternMarks::default()
        };
        assert!(should_block_high_order_trig_expansion_expr(
            &ctx,
            expr,
            Some(&marks),
            &[],
            true
        ));
        assert!(!should_block_high_order_trig_expansion_expr(
            &ctx,
            expr,
            Some(&marks),
            &[],
            false
        ));
    }

    #[test]
    fn collect_trig_args_recursive_finds_nested_unary_trig_arguments() {
        let mut ctx = Context::new();
        let expr = parse("sin(2*x) + cos(x+y) + tan(z) + ln(sin(w))", &mut ctx).expect("expr");
        let mut args = Vec::new();
        collect_trig_args_recursive(&ctx, expr, &mut args);

        let mut expected = [
            parse("2*x", &mut ctx).expect("2*x"),
            parse("x+y", &mut ctx).expect("x+y"),
            parse("z", &mut ctx).expect("z"),
            parse("w", &mut ctx).expect("w"),
        ];

        args.sort_by(|a, b| compare_expr(&ctx, *a, *b));
        expected.sort_by(|a, b| compare_expr(&ctx, *a, *b));

        assert_eq!(args.len(), expected.len());
        for (a, b) in args.iter().zip(expected.iter()) {
            assert_eq!(compare_expr(&ctx, *a, *b), Ordering::Equal);
        }
    }

    #[test]
    fn expand_trig_angle_rewrites_large_angle_trig_nodes_recursively() {
        let mut ctx = Context::new();
        let large = parse("2*x", &mut ctx).expect("large");
        let small = parse("x", &mut ctx).expect("small");

        let sin_expr = parse("sin(2*x)", &mut ctx).expect("sin_expr");
        let sin_got = expand_trig_angle(&mut ctx, sin_expr, large, small);
        let two = ctx.num(2);
        let sin_half = ctx.call_builtin(BuiltinFn::Sin, vec![small]);
        let cos_half = ctx.call_builtin(BuiltinFn::Cos, vec![small]);
        let sin_term = smart_mul(&mut ctx, sin_half, cos_half);
        let sin_expected = smart_mul(&mut ctx, two, sin_term);
        assert_eq!(compare_expr(&ctx, sin_got, sin_expected), Ordering::Equal);

        let cos_expr = parse("cos(2*x)", &mut ctx).expect("cos_expr");
        let cos_got = expand_trig_angle(&mut ctx, cos_expr, large, small);
        let one = ctx.num(1);
        let cos_half = ctx.call_builtin(BuiltinFn::Cos, vec![small]);
        let cos_sq = ctx.add(Expr::Pow(cos_half, two));
        let cos_term = smart_mul(&mut ctx, two, cos_sq);
        let cos_expected = ctx.add(Expr::Sub(cos_term, one));
        assert_eq!(compare_expr(&ctx, cos_got, cos_expected), Ordering::Equal);

        let tan_expr = parse("tan(2*x)", &mut ctx).expect("tan_expr");
        let tan_got = expand_trig_angle(&mut ctx, tan_expr, large, small);
        let tan_half = ctx.call_builtin(BuiltinFn::Tan, vec![small]);
        let tan_num = smart_mul(&mut ctx, two, tan_half);
        let tan_sq = ctx.add(Expr::Pow(tan_half, two));
        let tan_den = ctx.add(Expr::Sub(one, tan_sq));
        let tan_expected = ctx.add(Expr::Div(tan_num, tan_den));
        assert_eq!(compare_expr(&ctx, tan_got, tan_expected), Ordering::Equal);
    }

    #[test]
    fn angle_consistency_rewrite_expands_large_angle_to_half_angle_basis() {
        let mut ctx = Context::new();
        let expr = parse("sin(2*x) + cos(x)", &mut ctx).expect("expr");
        let rewrite = try_rewrite_angle_consistency_expr(&mut ctx, expr).expect("rewrite");
        let out = format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: &ctx,
                id: rewrite.rewritten
            }
        );
        assert!(out.contains("sin(x)"));
        assert!(out.contains("cos(x)"));
        assert!(out.contains("2"));
    }

    #[test]
    fn angle_consistency_rewrite_skips_when_no_double_angle_pair() {
        let mut ctx = Context::new();
        let expr = parse("sin(3*x) + cos(x)", &mut ctx).expect("expr");
        assert!(try_rewrite_angle_consistency_expr(&mut ctx, expr).is_none());
    }
}
