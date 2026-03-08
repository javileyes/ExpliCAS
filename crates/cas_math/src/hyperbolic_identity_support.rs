use crate::expr_predicates::{is_one_expr, is_two_expr};
use crate::trig_roots_flatten::{extract_double_angle_arg, extract_triple_angle_arg};
use cas_ast::ordering::compare_expr;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use std::cmp::Ordering;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HyperbolicPythagoreanValue {
    One,
    NegativeOne,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HyperbolicIdentityRewrite {
    pub rewritten: ExprId,
    pub desc: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SinhCoshToExpRewrite {
    pub rewritten: ExprId,
    pub desc: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HyperbolicDoubleAngleRewrite {
    pub rewritten: ExprId,
    pub desc: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HyperbolicTanhPythagoreanRewrite {
    pub rewritten: ExprId,
    pub desc: &'static str,
}

/// Detects hyperbolic Pythagorean subtraction forms:
/// - `cosh(x)^2 - sinh(x)^2` -> `1`
/// - `sinh(x)^2 - cosh(x)^2` -> `-1`
pub fn detect_hyperbolic_pythagorean_sub(
    ctx: &Context,
    expr: ExprId,
) -> Option<HyperbolicPythagoreanValue> {
    let Expr::Sub(l, r) = ctx.get(expr) else {
        return None;
    };
    let (l, r) = (*l, *r);

    let (Expr::Pow(l_base, l_exp), Expr::Pow(r_base, r_exp)) = (ctx.get(l), ctx.get(r)) else {
        return None;
    };
    let (l_base, l_exp, r_base, r_exp) = (*l_base, *l_exp, *r_base, *r_exp);

    if !is_two_expr(ctx, l_exp) || !is_two_expr(ctx, r_exp) {
        return None;
    }

    let (Expr::Function(l_fn, l_args), Expr::Function(r_fn, r_args)) =
        (ctx.get(l_base), ctx.get(r_base))
    else {
        return None;
    };

    if l_args.len() != 1 || r_args.len() != 1 {
        return None;
    }
    if compare_expr(ctx, l_args[0], r_args[0]) != Ordering::Equal {
        return None;
    }

    if ctx.is_builtin(*l_fn, BuiltinFn::Cosh) && ctx.is_builtin(*r_fn, BuiltinFn::Sinh) {
        return Some(HyperbolicPythagoreanValue::One);
    }

    if ctx.is_builtin(*l_fn, BuiltinFn::Sinh) && ctx.is_builtin(*r_fn, BuiltinFn::Cosh) {
        return Some(HyperbolicPythagoreanValue::NegativeOne);
    }

    None
}

/// Detect and rewrite:
/// - `cosh(x)^2 - sinh(x)^2` -> `1`
/// - `sinh(x)^2 - cosh(x)^2` -> `-1`
pub fn try_rewrite_hyperbolic_pythagorean_sub_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<HyperbolicIdentityRewrite> {
    match detect_hyperbolic_pythagorean_sub(ctx, expr)? {
        HyperbolicPythagoreanValue::One => Some(HyperbolicIdentityRewrite {
            rewritten: ctx.num(1),
            desc: "cosh²(x) - sinh²(x) = 1",
        }),
        HyperbolicPythagoreanValue::NegativeOne => Some(HyperbolicIdentityRewrite {
            rewritten: ctx.num(-1),
            desc: "sinh²(x) - cosh²(x) = -1",
        }),
    }
}

/// Detect and rewrite:
/// - `sinh(x) + cosh(x)` or `cosh(x) + sinh(x)` to `exp(x)`
/// - `cosh(x) - sinh(x)` to `exp(-x)`
/// - `sinh(x) - cosh(x)` to `-exp(-x)`
pub fn try_rewrite_sinh_cosh_to_exp(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<SinhCoshToExpRewrite> {
    match ctx.get(expr) {
        Expr::Add(l, r) => {
            let (l, r) = (*l, *r);
            let (Expr::Function(l_fn, l_args), Expr::Function(r_fn, r_args)) =
                (ctx.get(l), ctx.get(r))
            else {
                return None;
            };
            if l_args.len() != 1 || r_args.len() != 1 {
                return None;
            }
            let is_sinh_plus_cosh =
                ctx.is_builtin(*l_fn, BuiltinFn::Sinh) && ctx.is_builtin(*r_fn, BuiltinFn::Cosh);
            let is_cosh_plus_sinh =
                ctx.is_builtin(*l_fn, BuiltinFn::Cosh) && ctx.is_builtin(*r_fn, BuiltinFn::Sinh);

            if (is_sinh_plus_cosh || is_cosh_plus_sinh)
                && compare_expr(ctx, l_args[0], r_args[0]) == Ordering::Equal
            {
                let exp_x = ctx.call_builtin(BuiltinFn::Exp, vec![l_args[0]]);
                return Some(SinhCoshToExpRewrite {
                    rewritten: exp_x,
                    desc: "sinh(x) + cosh(x) = exp(x)",
                });
            }
            None
        }
        Expr::Sub(l, r) => {
            let (l, r) = (*l, *r);
            let (Expr::Function(l_fn, l_args), Expr::Function(r_fn, r_args)) =
                (ctx.get(l), ctx.get(r))
            else {
                return None;
            };
            if l_args.len() != 1 || r_args.len() != 1 {
                return None;
            }
            let is_cosh_minus_sinh =
                ctx.is_builtin(*l_fn, BuiltinFn::Cosh) && ctx.is_builtin(*r_fn, BuiltinFn::Sinh);
            let is_sinh_minus_cosh =
                ctx.is_builtin(*l_fn, BuiltinFn::Sinh) && ctx.is_builtin(*r_fn, BuiltinFn::Cosh);

            if !(is_cosh_minus_sinh || is_sinh_minus_cosh)
                || compare_expr(ctx, l_args[0], r_args[0]) != Ordering::Equal
            {
                return None;
            }

            let neg_arg = ctx.add(Expr::Neg(l_args[0]));
            let exp_neg_x = ctx.call_builtin(BuiltinFn::Exp, vec![neg_arg]);
            if is_cosh_minus_sinh {
                Some(SinhCoshToExpRewrite {
                    rewritten: exp_neg_x,
                    desc: "cosh(x) - sinh(x) = exp(-x)",
                })
            } else {
                Some(SinhCoshToExpRewrite {
                    rewritten: ctx.add(Expr::Neg(exp_neg_x)),
                    desc: "sinh(x) - cosh(x) = -exp(-x)",
                })
            }
        }
        _ => None,
    }
}

/// Detect and rewrite:
/// `cosh(x)^2 + sinh(x)^2` (or swapped) to `cosh(2x)`.
pub fn try_rewrite_hyperbolic_double_angle_sum(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<HyperbolicDoubleAngleRewrite> {
    let Expr::Add(l, r) = ctx.get(expr) else {
        return None;
    };
    let (l, r) = (*l, *r);

    let (Expr::Pow(l_base, l_exp), Expr::Pow(r_base, r_exp)) = (ctx.get(l), ctx.get(r)) else {
        return None;
    };
    let (l_base, l_exp, r_base, r_exp) = (*l_base, *l_exp, *r_base, *r_exp);

    if !is_two_expr(ctx, l_exp) || !is_two_expr(ctx, r_exp) {
        return None;
    }

    let (Expr::Function(l_fn, l_args), Expr::Function(r_fn, r_args)) =
        (ctx.get(l_base), ctx.get(r_base))
    else {
        return None;
    };
    if l_args.len() != 1 || r_args.len() != 1 {
        return None;
    }

    let is_cosh_sinh =
        ctx.is_builtin(*l_fn, BuiltinFn::Cosh) && ctx.is_builtin(*r_fn, BuiltinFn::Sinh);
    let is_sinh_cosh =
        ctx.is_builtin(*l_fn, BuiltinFn::Sinh) && ctx.is_builtin(*r_fn, BuiltinFn::Cosh);

    if !(is_cosh_sinh || is_sinh_cosh) || compare_expr(ctx, l_args[0], r_args[0]) != Ordering::Equal
    {
        return None;
    }

    let x = l_args[0];
    let two = ctx.num(2);
    let two_x = ctx.add(Expr::Mul(two, x));
    let cosh_2x = ctx.call_builtin(BuiltinFn::Cosh, vec![two_x]);
    Some(HyperbolicDoubleAngleRewrite {
        rewritten: cosh_2x,
        desc: "cosh²(x) + sinh²(x) = cosh(2x)",
    })
}

/// Detect and rewrite additive-chain subtraction form:
/// `cosh(2x) - cosh²(x) - sinh²(x) -> 0`.
///
/// Works on canonicalized add chains where subtraction appears as `Add(..., Neg(...))`.
pub fn try_rewrite_hyperbolic_double_angle_sub_chain(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<HyperbolicDoubleAngleRewrite> {
    let mut terms = Vec::new();
    let mut stack = vec![expr];
    while let Some(id) = stack.pop() {
        if let Expr::Add(l, r) = ctx.get(id) {
            stack.push(*l);
            stack.push(*r);
        } else {
            terms.push(id);
        }
    }

    if terms.len() < 3 {
        return None;
    }

    let find_cosh_double = |terms: &[ExprId]| -> Option<(usize, ExprId)> {
        for (i, &t) in terms.iter().enumerate() {
            if let Expr::Function(fn_id, args) = ctx.get(t) {
                if ctx.builtin_of(*fn_id) == Some(BuiltinFn::Cosh) && args.len() == 1 {
                    if let Some(x) = extract_double_angle_arg(ctx, args[0]) {
                        return Some((i, x));
                    }
                }
            }
        }
        None
    };

    let as_neg_hyp_squared = |e: ExprId| -> Option<(ExprId, bool)> {
        if let Expr::Neg(inner) = ctx.get(e) {
            if let Expr::Pow(base, exp) = ctx.get(*inner) {
                if is_two_expr(ctx, *exp) {
                    if let Expr::Function(fn_id, args) = ctx.get(*base) {
                        if args.len() == 1 {
                            if ctx.is_builtin(*fn_id, BuiltinFn::Cosh) {
                                return Some((args[0], true));
                            }
                            if ctx.is_builtin(*fn_id, BuiltinFn::Sinh) {
                                return Some((args[0], false));
                            }
                        }
                    }
                }
            }
        }
        None
    };

    let (cosh_idx, x_arg) = find_cosh_double(&terms)?;

    let mut neg_cosh_idx = None;
    let mut neg_sinh_idx = None;

    for (i, &t) in terms.iter().enumerate() {
        if i == cosh_idx {
            continue;
        }
        if let Some((arg, is_cosh)) = as_neg_hyp_squared(t) {
            if compare_expr(ctx, arg, x_arg) == Ordering::Equal {
                if is_cosh && neg_cosh_idx.is_none() {
                    neg_cosh_idx = Some(i);
                } else if !is_cosh && neg_sinh_idx.is_none() {
                    neg_sinh_idx = Some(i);
                }
            }
        }
    }

    let nc_idx = neg_cosh_idx?;
    let ns_idx = neg_sinh_idx?;

    let mut remaining: Vec<ExprId> = terms
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != cosh_idx && i != nc_idx && i != ns_idx)
        .map(|(_, &t)| t)
        .collect();

    let rewritten = if remaining.is_empty() {
        ctx.num(0)
    } else {
        let mut result = remaining.pop().expect("non-empty");
        while let Some(t) = remaining.pop() {
            result = ctx.add(Expr::Add(t, result));
        }
        result
    };

    Some(HyperbolicDoubleAngleRewrite {
        rewritten,
        desc: "cosh(2x) - cosh²(x) - sinh²(x) = 0",
    })
}

/// Detect and rewrite additive-chain form of:
/// `1 - tanh(x)^2 -> 1/cosh(x)^2`.
///
/// Works on flattened additive forms such as `1 + (-tanh(x)^2) + rest`.
pub fn try_rewrite_tanh_pythagorean_add_chain(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<HyperbolicTanhPythagoreanRewrite> {
    let terms = crate::expr_nary::add_leaves(ctx, expr);
    if terms.len() < 2 {
        return None;
    }

    let mut one_idx: Option<usize> = None;
    let mut tanh2_idx: Option<usize> = None;
    let mut tanh_arg: Option<ExprId> = None;

    for (i, &term) in terms.iter().enumerate() {
        if is_one_expr(ctx, term) {
            one_idx = Some(i);
            continue;
        }

        if let Expr::Neg(inner) = ctx.get(term) {
            if let Expr::Pow(base, exp) = ctx.get(*inner) {
                if is_two_expr(ctx, *exp) {
                    if let Expr::Function(fn_id, args) = ctx.get(*base) {
                        if ctx.is_builtin(*fn_id, BuiltinFn::Tanh) && args.len() == 1 {
                            tanh2_idx = Some(i);
                            tanh_arg = Some(args[0]);
                        }
                    }
                }
            }
        }
    }

    let (one_i, tanh_i, arg) = (one_idx?, tanh2_idx?, tanh_arg?);

    let cosh = ctx.call_builtin(BuiltinFn::Cosh, vec![arg]);
    let two = ctx.num(2);
    let cosh_squared = ctx.add(Expr::Pow(cosh, two));
    let one = ctx.num(1);
    let sech_squared = ctx.add(Expr::Div(one, cosh_squared));

    let mut new_terms: Vec<ExprId> = Vec::new();
    for (i, &term) in terms.iter().enumerate() {
        if i != one_i && i != tanh_i {
            new_terms.push(term);
        }
    }
    new_terms.push(sech_squared);

    let rewritten = if new_terms.len() == 1 {
        new_terms[0]
    } else {
        let mut acc = new_terms[0];
        for &term in new_terms.iter().skip(1) {
            acc = ctx.add(Expr::Add(acc, term));
        }
        acc
    };

    Some(HyperbolicTanhPythagoreanRewrite {
        rewritten,
        desc: "1 - tanh²(x) = 1/cosh²(x)",
    })
}

/// Detect and rewrite `sinh(x)/cosh(x) -> tanh(x)`.
pub fn try_rewrite_sinh_cosh_to_tanh(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Div(num, den) = ctx.get(expr) else {
        return None;
    };
    let num = *num;
    let den = *den;

    let Expr::Function(num_name, num_args) = ctx.get(num) else {
        return None;
    };
    if !ctx.is_builtin(*num_name, BuiltinFn::Sinh) || num_args.len() != 1 {
        return None;
    }

    let Expr::Function(den_name, den_args) = ctx.get(den) else {
        return None;
    };
    if !ctx.is_builtin(*den_name, BuiltinFn::Cosh) || den_args.len() != 1 {
        return None;
    }

    if compare_expr(ctx, num_args[0], den_args[0]) != Ordering::Equal {
        return None;
    }

    Some(ctx.call_builtin(BuiltinFn::Tanh, vec![num_args[0]]))
}

/// Detect and rewrite `sinh(x)/cosh(x) -> tanh(x)` with canonical description.
pub fn try_rewrite_sinh_cosh_to_tanh_identity_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<HyperbolicIdentityRewrite> {
    let rewritten = try_rewrite_sinh_cosh_to_tanh(ctx, expr)?;
    Some(HyperbolicIdentityRewrite {
        rewritten,
        desc: "sinh(x)/cosh(x) = tanh(x)",
    })
}

/// Detect and rewrite `tanh(x) -> sinh(x)/cosh(x)`.
///
/// Guarded to preserve:
/// - direct composition simplifications like `tanh(atanh(x))`
/// - odd-function normalization `tanh(-x) -> -tanh(x)`
pub fn try_rewrite_tanh_to_sinh_cosh(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if !ctx.is_builtin(*fn_id, BuiltinFn::Tanh) || args.len() != 1 {
        return None;
    }
    let x = args[0];

    // Preserve composition rules for inverse hyperbolic arguments.
    if let Expr::Function(inner_fn, _) = ctx.get(x) {
        if ctx.is_builtin(*inner_fn, BuiltinFn::Atanh)
            || ctx.is_builtin(*inner_fn, BuiltinFn::Asinh)
            || ctx.is_builtin(*inner_fn, BuiltinFn::Acosh)
        {
            return None;
        }
    }

    // Preserve odd-function rewrite path tanh(-x) -> -tanh(x).
    if matches!(ctx.get(x), Expr::Neg(_)) {
        return None;
    }

    let sinh_x = ctx.call_builtin(BuiltinFn::Sinh, vec![x]);
    let cosh_x = ctx.call_builtin(BuiltinFn::Cosh, vec![x]);
    Some(ctx.add(Expr::Div(sinh_x, cosh_x)))
}

/// Detect and rewrite `tanh(x) -> sinh(x)/cosh(x)` with canonical description.
pub fn try_rewrite_tanh_to_sinh_cosh_identity_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<HyperbolicIdentityRewrite> {
    let rewritten = try_rewrite_tanh_to_sinh_cosh(ctx, expr)?;
    Some(HyperbolicIdentityRewrite {
        rewritten,
        desc: "tanh(x) = sinh(x)/cosh(x)",
    })
}

/// Detect and rewrite `sinh(2x) -> 2*sinh(x)*cosh(x)`.
pub fn try_rewrite_sinh_double_angle_expansion(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if !ctx.is_builtin(*fn_id, BuiltinFn::Sinh) || args.len() != 1 {
        return None;
    }

    let x = extract_double_angle_arg(ctx, args[0])?;
    let two = ctx.num(2);
    let sinh_x = ctx.call_builtin(BuiltinFn::Sinh, vec![x]);
    let cosh_x = ctx.call_builtin(BuiltinFn::Cosh, vec![x]);
    let sinh_cosh = crate::expr_rewrite::smart_mul(ctx, sinh_x, cosh_x);
    Some(crate::expr_rewrite::smart_mul(ctx, two, sinh_cosh))
}

/// Detect and rewrite `sinh(2x) -> 2*sinh(x)*cosh(x)` with canonical description.
pub fn try_rewrite_sinh_double_angle_expansion_identity_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<HyperbolicIdentityRewrite> {
    let rewritten = try_rewrite_sinh_double_angle_expansion(ctx, expr)?;
    Some(HyperbolicIdentityRewrite {
        rewritten,
        desc: "sinh(2x) = 2·sinh(x)·cosh(x)",
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HyperbolicTripleAngleRewrite {
    pub rewritten: ExprId,
    pub desc: &'static str,
}

/// Detect and rewrite hyperbolic triple-angle expansions.
///
/// - `sinh(3x) -> 3*sinh(x) + 4*sinh(x)^3`
/// - `cosh(3x) -> 4*cosh(x)^3 - 3*cosh(x)`
pub fn try_rewrite_hyperbolic_triple_angle(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<HyperbolicTripleAngleRewrite> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }

    let x = extract_triple_angle_arg(ctx, args[0])?;

    // Expand only for trivial argument forms to avoid expression blow-up.
    let is_simple = |id: ExprId| {
        matches!(
            ctx.get(id),
            Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_)
        )
    };
    match ctx.get(x) {
        Expr::Variable(_) | Expr::Constant(_) | Expr::Number(_) => {}
        Expr::Mul(l, r) => {
            if !(is_simple(*l) && is_simple(*r)) {
                return None;
            }
        }
        Expr::Neg(inner) => {
            if !is_simple(*inner) {
                return None;
            }
        }
        _ => return None,
    }

    match ctx.builtin_of(*fn_id) {
        Some(BuiltinFn::Sinh) => {
            let three = ctx.num(3);
            let four = ctx.num(4);
            let exp_three = ctx.num(3);
            let sinh_x = ctx.call_builtin(BuiltinFn::Sinh, vec![x]);
            let term1 = crate::expr_rewrite::smart_mul(ctx, three, sinh_x);
            let sinh_cubed = ctx.add(Expr::Pow(sinh_x, exp_three));
            let term2 = crate::expr_rewrite::smart_mul(ctx, four, sinh_cubed);
            let rewritten = ctx.add(Expr::Add(term1, term2));
            Some(HyperbolicTripleAngleRewrite {
                rewritten,
                desc: "sinh(3x) → 3sinh(x) + 4sinh³(x)",
            })
        }
        Some(BuiltinFn::Cosh) => {
            let three = ctx.num(3);
            let four = ctx.num(4);
            let exp_three = ctx.num(3);
            let cosh_x = ctx.call_builtin(BuiltinFn::Cosh, vec![x]);
            let cosh_cubed = ctx.add(Expr::Pow(cosh_x, exp_three));
            let term1 = crate::expr_rewrite::smart_mul(ctx, four, cosh_cubed);
            let term2 = crate::expr_rewrite::smart_mul(ctx, three, cosh_x);
            let rewritten = ctx.add(Expr::Sub(term1, term2));
            Some(HyperbolicTripleAngleRewrite {
                rewritten,
                desc: "cosh(3x) → 4cosh³(x) - 3cosh(x)",
            })
        }
        _ => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecognizeHyperbolicFromExpRewrite {
    pub rewritten: ExprId,
    pub desc: &'static str,
}

/// Detect and rewrite exponential definitions of hyperbolic functions:
/// - `(e^x + e^(-x))/2` or `(1/2)*(...)` -> `cosh(x)`
/// - `(e^x - e^(-x))/2` or `(1/2)*(...)` -> `sinh(x)`
/// - `(e^(-x) - e^x)/2` or `(1/2)*(...)` -> `-sinh(x)`
/// - `(e^x - e^(-x))/(e^x + e^(-x))` -> `tanh(x)` (or negated variant)
pub fn try_rewrite_recognize_hyperbolic_from_exp(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<RecognizeHyperbolicFromExpRewrite> {
    // Pattern 1: Div(sum_or_diff, 2)
    if let Expr::Div(num, den) = ctx.get(expr) {
        if is_two_expr(ctx, *den) {
            if let Some((arg, is_cosh, positive_first)) =
                crate::hyperbolic_exp_support::extract_exp_pair(ctx, *num)
            {
                if is_cosh {
                    let cosh = ctx.call_builtin(BuiltinFn::Cosh, vec![arg]);
                    return Some(RecognizeHyperbolicFromExpRewrite {
                        rewritten: cosh,
                        desc: "(e^x + e^(-x))/2 = cosh(x)",
                    });
                } else if positive_first {
                    let sinh = ctx.call_builtin(BuiltinFn::Sinh, vec![arg]);
                    return Some(RecognizeHyperbolicFromExpRewrite {
                        rewritten: sinh,
                        desc: "(e^x - e^(-x))/2 = sinh(x)",
                    });
                } else {
                    let sinh = ctx.call_builtin(BuiltinFn::Sinh, vec![arg]);
                    return Some(RecognizeHyperbolicFromExpRewrite {
                        rewritten: ctx.add(Expr::Neg(sinh)),
                        desc: "(e^(-x) - e^x)/2 = -sinh(x)",
                    });
                }
            }
        }
    }

    // Pattern 2: Mul(1/2, sum_or_diff) or Mul(sum_or_diff, 1/2)
    if let Expr::Mul(l, r) = ctx.get(expr) {
        let sum_id = if crate::expr_predicates::is_half_expr(ctx, *l) {
            Some(*r)
        } else if crate::expr_predicates::is_half_expr(ctx, *r) {
            Some(*l)
        } else {
            None
        };
        if let Some(sum_id) = sum_id {
            if let Some((arg, is_cosh, positive_first)) =
                crate::hyperbolic_exp_support::extract_exp_pair(ctx, sum_id)
            {
                if is_cosh {
                    let cosh = ctx.call_builtin(BuiltinFn::Cosh, vec![arg]);
                    return Some(RecognizeHyperbolicFromExpRewrite {
                        rewritten: cosh,
                        desc: "(e^x + e^(-x))/2 = cosh(x)",
                    });
                } else if positive_first {
                    let sinh = ctx.call_builtin(BuiltinFn::Sinh, vec![arg]);
                    return Some(RecognizeHyperbolicFromExpRewrite {
                        rewritten: sinh,
                        desc: "(e^x - e^(-x))/2 = sinh(x)",
                    });
                } else {
                    let sinh = ctx.call_builtin(BuiltinFn::Sinh, vec![arg]);
                    return Some(RecognizeHyperbolicFromExpRewrite {
                        rewritten: ctx.add(Expr::Neg(sinh)),
                        desc: "(e^(-x) - e^x)/2 = -sinh(x)",
                    });
                }
            }
        }
    }

    // Pattern 3: (e^x - e^(-x)) / (e^x + e^(-x)) -> tanh(x) (or -tanh(x)).
    if let Expr::Div(num, den) = ctx.get(expr) {
        if let Some((num_arg, false, num_positive_first)) =
            crate::hyperbolic_exp_support::extract_exp_pair(ctx, *num)
        {
            if let Some((den_arg, true, _)) =
                crate::hyperbolic_exp_support::extract_exp_pair(ctx, *den)
            {
                if compare_expr(ctx, num_arg, den_arg) == Ordering::Equal {
                    let tanh_x = ctx.call_builtin(BuiltinFn::Tanh, vec![num_arg]);
                    if num_positive_first {
                        return Some(RecognizeHyperbolicFromExpRewrite {
                            rewritten: tanh_x,
                            desc: "(e^x - e^(-x))/(e^x + e^(-x)) = tanh(x)",
                        });
                    } else {
                        return Some(RecognizeHyperbolicFromExpRewrite {
                            rewritten: ctx.add(Expr::Neg(tanh_x)),
                            desc: "(e^(-x) - e^x)/(e^x + e^(-x)) = -tanh(x)",
                        });
                    }
                }
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::{
        detect_hyperbolic_pythagorean_sub, try_rewrite_hyperbolic_double_angle_sub_chain,
        try_rewrite_hyperbolic_double_angle_sum, try_rewrite_hyperbolic_pythagorean_sub_expr,
        try_rewrite_hyperbolic_triple_angle, try_rewrite_recognize_hyperbolic_from_exp,
        try_rewrite_sinh_cosh_to_exp, try_rewrite_sinh_cosh_to_tanh,
        try_rewrite_sinh_cosh_to_tanh_identity_expr, try_rewrite_sinh_double_angle_expansion,
        try_rewrite_sinh_double_angle_expansion_identity_expr,
        try_rewrite_tanh_pythagorean_add_chain, try_rewrite_tanh_to_sinh_cosh,
        try_rewrite_tanh_to_sinh_cosh_identity_expr, HyperbolicPythagoreanValue,
    };
    use cas_ast::{BuiltinFn, Context, Expr};

    #[test]
    fn detects_cosh2_minus_sinh2() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let cosh = ctx.call_builtin(BuiltinFn::Cosh, vec![x]);
        let sinh = ctx.call_builtin(BuiltinFn::Sinh, vec![x]);
        let lhs = ctx.add(Expr::Pow(cosh, two));
        let rhs = ctx.add(Expr::Pow(sinh, two));
        let expr = ctx.add(Expr::Sub(lhs, rhs));

        assert_eq!(
            detect_hyperbolic_pythagorean_sub(&ctx, expr),
            Some(HyperbolicPythagoreanValue::One)
        );
    }

    #[test]
    fn detects_sinh2_minus_cosh2() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let cosh = ctx.call_builtin(BuiltinFn::Cosh, vec![x]);
        let sinh = ctx.call_builtin(BuiltinFn::Sinh, vec![x]);
        let lhs = ctx.add(Expr::Pow(sinh, two));
        let rhs = ctx.add(Expr::Pow(cosh, two));
        let expr = ctx.add(Expr::Sub(lhs, rhs));

        assert_eq!(
            detect_hyperbolic_pythagorean_sub(&ctx, expr),
            Some(HyperbolicPythagoreanValue::NegativeOne)
        );
    }

    #[test]
    fn rewrites_hyperbolic_pythagorean_sub_with_desc() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let cosh = ctx.call_builtin(BuiltinFn::Cosh, vec![x]);
        let sinh = ctx.call_builtin(BuiltinFn::Sinh, vec![x]);
        let lhs = ctx.add(Expr::Pow(cosh, two));
        let rhs = ctx.add(Expr::Pow(sinh, two));
        let expr = ctx.add(Expr::Sub(lhs, rhs));
        let rewrite = try_rewrite_hyperbolic_pythagorean_sub_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(rewrite.desc, "cosh²(x) - sinh²(x) = 1");
    }

    #[test]
    fn rejects_mismatched_arguments() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let two = ctx.num(2);
        let cosh = ctx.call_builtin(BuiltinFn::Cosh, vec![x]);
        let sinh = ctx.call_builtin(BuiltinFn::Sinh, vec![y]);
        let lhs = ctx.add(Expr::Pow(cosh, two));
        let rhs = ctx.add(Expr::Pow(sinh, two));
        let expr = ctx.add(Expr::Sub(lhs, rhs));

        assert_eq!(detect_hyperbolic_pythagorean_sub(&ctx, expr), None);
    }

    #[test]
    fn rewrites_sinh_plus_cosh_to_exp() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let sinh = ctx.call_builtin(BuiltinFn::Sinh, vec![x]);
        let cosh = ctx.call_builtin(BuiltinFn::Cosh, vec![x]);
        let expr = ctx.add(Expr::Add(sinh, cosh));

        let rewrite = try_rewrite_sinh_cosh_to_exp(&mut ctx, expr).expect("rewrite");
        assert_eq!(rewrite.desc, "sinh(x) + cosh(x) = exp(x)");
    }

    #[test]
    fn rewrites_cosh_minus_sinh_to_exp_neg() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let sinh = ctx.call_builtin(BuiltinFn::Sinh, vec![x]);
        let cosh = ctx.call_builtin(BuiltinFn::Cosh, vec![x]);
        let expr = ctx.add(Expr::Sub(cosh, sinh));

        let rewrite = try_rewrite_sinh_cosh_to_exp(&mut ctx, expr).expect("rewrite");
        assert_eq!(rewrite.desc, "cosh(x) - sinh(x) = exp(-x)");
    }

    #[test]
    fn rewrites_double_angle_sum() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let cosh = ctx.call_builtin(BuiltinFn::Cosh, vec![x]);
        let sinh = ctx.call_builtin(BuiltinFn::Sinh, vec![x]);
        let cosh_sq = ctx.add(Expr::Pow(cosh, two));
        let sinh_sq = ctx.add(Expr::Pow(sinh, two));
        let expr = ctx.add(Expr::Add(cosh_sq, sinh_sq));

        let rewrite = try_rewrite_hyperbolic_double_angle_sum(&mut ctx, expr).expect("rewrite");
        assert_eq!(rewrite.desc, "cosh²(x) + sinh²(x) = cosh(2x)");
    }

    #[test]
    fn rewrites_double_angle_sub_chain_to_zero() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let two_x = ctx.add(Expr::Mul(two, x));
        let cosh_2x = ctx.call_builtin(BuiltinFn::Cosh, vec![two_x]);
        let cosh = ctx.call_builtin(BuiltinFn::Cosh, vec![x]);
        let sinh = ctx.call_builtin(BuiltinFn::Sinh, vec![x]);
        let cosh_sq = ctx.add(Expr::Pow(cosh, two));
        let sinh_sq = ctx.add(Expr::Pow(sinh, two));
        let neg_cosh_sq = ctx.add(Expr::Neg(cosh_sq));
        let neg_sinh_sq = ctx.add(Expr::Neg(sinh_sq));
        let tail = ctx.add(Expr::Add(neg_cosh_sq, neg_sinh_sq));
        let expr = ctx.add(Expr::Add(cosh_2x, tail));

        let rewrite =
            try_rewrite_hyperbolic_double_angle_sub_chain(&mut ctx, expr).expect("rewrite");
        assert_eq!(rewrite.desc, "cosh(2x) - cosh²(x) - sinh²(x) = 0");
        let zero = num_rational::BigRational::from_integer(0.into());
        assert!(matches!(ctx.get(rewrite.rewritten), Expr::Number(n) if n == &zero));
    }

    #[test]
    fn rewrites_tanh_pythagorean_add_chain() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let two = ctx.num(2);
        let tanh_x = ctx.call_builtin(BuiltinFn::Tanh, vec![x]);
        let tanh_sq = ctx.add(Expr::Pow(tanh_x, two));
        let neg_tanh_sq = ctx.add(Expr::Neg(tanh_sq));
        let expr = ctx.add(Expr::Add(one, neg_tanh_sq));

        let rewrite = try_rewrite_tanh_pythagorean_add_chain(&mut ctx, expr).expect("rewrite");
        assert_eq!(rewrite.desc, "1 - tanh²(x) = 1/cosh²(x)");
    }

    #[test]
    fn rewrites_sinh_div_cosh_to_tanh() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let sinh = ctx.call_builtin(BuiltinFn::Sinh, vec![x]);
        let cosh = ctx.call_builtin(BuiltinFn::Cosh, vec![x]);
        let expr = ctx.add(Expr::Div(sinh, cosh));

        let rewrite = try_rewrite_sinh_cosh_to_tanh(&mut ctx, expr).expect("rewrite");
        let Expr::Function(fn_id, args) = ctx.get(rewrite) else {
            panic!("expected function");
        };
        assert!(ctx.is_builtin(*fn_id, BuiltinFn::Tanh));
        assert_eq!(args, &vec![x]);
    }

    #[test]
    fn rewrites_sinh_div_cosh_to_tanh_with_desc() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let sinh = ctx.call_builtin(BuiltinFn::Sinh, vec![x]);
        let cosh = ctx.call_builtin(BuiltinFn::Cosh, vec![x]);
        let expr = ctx.add(Expr::Div(sinh, cosh));
        let rewrite = try_rewrite_sinh_cosh_to_tanh_identity_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(rewrite.desc, "sinh(x)/cosh(x) = tanh(x)");
    }

    #[test]
    fn rewrites_tanh_to_sinh_div_cosh() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let expr = ctx.call_builtin(BuiltinFn::Tanh, vec![x]);

        let rewrite = try_rewrite_tanh_to_sinh_cosh(&mut ctx, expr).expect("rewrite");
        let Expr::Div(num, den) = ctx.get(rewrite) else {
            panic!("expected division");
        };
        let Expr::Function(num_fn, num_args) = ctx.get(*num) else {
            panic!("expected function numerator");
        };
        let Expr::Function(den_fn, den_args) = ctx.get(*den) else {
            panic!("expected function denominator");
        };
        assert!(ctx.is_builtin(*num_fn, BuiltinFn::Sinh));
        assert!(ctx.is_builtin(*den_fn, BuiltinFn::Cosh));
        assert_eq!(num_args, &vec![x]);
        assert_eq!(den_args, &vec![x]);
    }

    #[test]
    fn rewrites_tanh_to_sinh_div_cosh_with_desc() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let expr = ctx.call_builtin(BuiltinFn::Tanh, vec![x]);
        let rewrite = try_rewrite_tanh_to_sinh_cosh_identity_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(rewrite.desc, "tanh(x) = sinh(x)/cosh(x)");
    }

    #[test]
    fn rewrites_sinh_double_angle_expansion() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let two_x = ctx.add(Expr::Mul(two, x));
        let expr = ctx.call_builtin(BuiltinFn::Sinh, vec![two_x]);

        let rewrite = try_rewrite_sinh_double_angle_expansion(&mut ctx, expr).expect("rewrite");
        let Expr::Mul(lhs, rhs) = ctx.get(rewrite) else {
            panic!("expected outer multiplication");
        };
        let (two_factor, inner_mul) = if *lhs == two {
            (*lhs, *rhs)
        } else if *rhs == two {
            (*rhs, *lhs)
        } else {
            panic!("expected numeric factor 2");
        };
        assert_eq!(two_factor, two);

        let Expr::Mul(m1, m2) = ctx.get(inner_mul) else {
            panic!("expected inner multiplication");
        };
        let (f1, f2) = (*m1, *m2);
        let Expr::Function(fn1, args1) = ctx.get(f1) else {
            panic!("expected function factor");
        };
        let Expr::Function(fn2, args2) = ctx.get(f2) else {
            panic!("expected function factor");
        };
        assert_eq!(args1, &vec![x]);
        assert_eq!(args2, &vec![x]);
        let is_sinh_cosh =
            ctx.is_builtin(*fn1, BuiltinFn::Sinh) && ctx.is_builtin(*fn2, BuiltinFn::Cosh);
        let is_cosh_sinh =
            ctx.is_builtin(*fn1, BuiltinFn::Cosh) && ctx.is_builtin(*fn2, BuiltinFn::Sinh);
        assert!(is_sinh_cosh || is_cosh_sinh, "expected sinh/cosh factors");
    }

    #[test]
    fn rewrites_sinh_double_angle_expansion_with_desc() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let two_x = ctx.add(Expr::Mul(two, x));
        let expr = ctx.call_builtin(BuiltinFn::Sinh, vec![two_x]);
        let rewrite =
            try_rewrite_sinh_double_angle_expansion_identity_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(rewrite.desc, "sinh(2x) = 2·sinh(x)·cosh(x)");
    }

    #[test]
    fn rewrites_hyperbolic_triple_angle_sinh() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let three = ctx.num(3);
        let three_x = ctx.add(Expr::Mul(three, x));
        let expr = ctx.call_builtin(BuiltinFn::Sinh, vec![three_x]);

        let rewrite = try_rewrite_hyperbolic_triple_angle(&mut ctx, expr).expect("rewrite");
        assert_eq!(rewrite.desc, "sinh(3x) → 3sinh(x) + 4sinh³(x)");
    }

    #[test]
    fn rewrites_hyperbolic_triple_angle_cosh() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let three = ctx.num(3);
        let three_x = ctx.add(Expr::Mul(three, x));
        let expr = ctx.call_builtin(BuiltinFn::Cosh, vec![three_x]);

        let rewrite = try_rewrite_hyperbolic_triple_angle(&mut ctx, expr).expect("rewrite");
        assert_eq!(rewrite.desc, "cosh(3x) → 4cosh³(x) - 3cosh(x)");
    }

    #[test]
    fn recognize_from_exp_rewrites_div_by_two() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let e = ctx.add(Expr::Constant(cas_ast::Constant::E));
        let neg_x = ctx.add(Expr::Neg(x));
        let exp_x = ctx.add(Expr::Pow(e, x));
        let e2 = ctx.add(Expr::Constant(cas_ast::Constant::E));
        let exp_neg_x = ctx.add(Expr::Pow(e2, neg_x));
        let sum = ctx.add(Expr::Add(exp_x, exp_neg_x));
        let two = ctx.num(2);
        let expr = ctx.add(Expr::Div(sum, two));

        let rewrite = try_rewrite_recognize_hyperbolic_from_exp(&mut ctx, expr).expect("rewrite");
        assert_eq!(rewrite.desc, "(e^x + e^(-x))/2 = cosh(x)");
    }

    #[test]
    fn recognize_from_exp_rewrites_tanh_ratio() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let e = ctx.add(Expr::Constant(cas_ast::Constant::E));
        let neg_x = ctx.add(Expr::Neg(x));
        let exp_x = ctx.add(Expr::Pow(e, x));
        let e2 = ctx.add(Expr::Constant(cas_ast::Constant::E));
        let exp_neg_x = ctx.add(Expr::Pow(e2, neg_x));
        let num = ctx.add(Expr::Sub(exp_x, exp_neg_x));
        let den = ctx.add(Expr::Add(exp_x, exp_neg_x));
        let expr = ctx.add(Expr::Div(num, den));

        let rewrite = try_rewrite_recognize_hyperbolic_from_exp(&mut ctx, expr).expect("rewrite");
        assert_eq!(rewrite.desc, "(e^x - e^(-x))/(e^x + e^(-x)) = tanh(x)");
    }
}
