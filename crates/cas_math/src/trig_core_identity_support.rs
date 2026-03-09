use crate::expr_destructure::{as_add, as_div, as_mul, as_neg, as_sub};
use crate::expr_nary::{add_leaves, mul_leaves};
use crate::expr_predicates::contains_variable;
use crate::expr_relations::extract_negated_inner;
use crate::expr_rewrite::smart_mul;
use crate::pattern_marks::PatternMarks;
use crate::pi_helpers::extract_rational_pi_multiple;
use crate::pi_helpers::{is_pi, is_pi_over_n};
use crate::trig_multi_angle_support::{has_large_coefficient, is_multiple_angle};
use crate::trig_pattern_detection::try_extract_all_negative_sum;
use crate::trig_table::{eval_inv_trig_special, eval_trig_special, InvTrigFn, TrigFn};
use cas_ast::ordering::compare_expr;
use cas_ast::{Context, Expr, ExprId};
use num_traits::{One, Zero};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrigOddEvenParityKind {
    Odd,
    Even,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SinCosIntegerPiKind {
    SinIntegerPi,
    CosIntegerPiEven,
    CosIntegerPiOdd,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SinCosIntegerPiRewrite {
    pub rewritten: ExprId,
    pub kind: SinCosIntegerPiKind,
    pub n: num_bigint::BigInt,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrigOddEvenParityRewrite {
    pub rewritten: ExprId,
    pub fn_name: String,
    pub kind: TrigOddEvenParityKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LegacyTrigEvalRewriteKind {
    TableEval { fn_name: String },
    ZeroEval { fn_name: String },
    CosZero,
    ArccosZero,
    ArcsinOne,
    ArccosOne,
    ArctanOne,
    ArcsinHalf,
    ArccosHalf,
    PiZero { fn_name: String },
    CosPi,
    SinPiOver2,
    CosPiOver2,
    TanPiOver2,
    SinPiOver3,
    CosPiOver3,
    TanPiOver3,
    PiOver4Trig { fn_name: String },
    TanPiOver4,
    SinPiOver6,
    CosPiOver6,
    TanPiOver6,
    SinNegative,
    CosNegative,
    TanNegative,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LegacyTrigEvalRewrite {
    pub rewritten: ExprId,
    pub kind: LegacyTrigEvalRewriteKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PythagoreanIdentityRewriteKind {
    Empty,
    Standard,
    Negated,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PythagoreanIdentityRewrite {
    pub rewritten: ExprId,
    pub kind: PythagoreanIdentityRewriteKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AngleSumDiffIdentityKind {
    SinAdd,
    SinSub,
    SinDivAdd,
    CosAdd,
    CosSub,
    CosDivAdd,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AngleSumDiffIdentityRewrite {
    pub rewritten: ExprId,
    pub kind: AngleSumDiffIdentityKind,
}

/// Rewrite exact values for `sin(n*pi)` and `cos(n*pi)` when `n` is integer.
pub fn try_rewrite_sin_cos_integer_pi_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<SinCosIntegerPiRewrite> {
    let (fn_id, args) = match ctx.get(expr) {
        Expr::Function(fn_id, args) => (*fn_id, args.clone()),
        _ => return None,
    };
    let builtin = ctx.builtin_of(fn_id)?;
    if args.len() != 1 {
        return None;
    }

    let is_sin = matches!(builtin, cas_ast::BuiltinFn::Sin);
    let is_cos = matches!(builtin, cas_ast::BuiltinFn::Cos);
    if !is_sin && !is_cos {
        return None;
    }

    let arg = args[0];
    let k = extract_rational_pi_multiple(ctx, arg)?;
    if !k.is_integer() {
        return None;
    }
    let n = k.to_integer();

    if is_sin {
        return Some(SinCosIntegerPiRewrite {
            rewritten: ctx.num(0),
            kind: SinCosIntegerPiKind::SinIntegerPi,
            n,
        });
    }

    let is_even = &n % 2 == num_bigint::BigInt::from(0);
    let rewritten = if is_even { ctx.num(1) } else { ctx.num(-1) };
    Some(SinCosIntegerPiRewrite {
        rewritten,
        kind: if is_even {
            SinCosIntegerPiKind::CosIntegerPiEven
        } else {
            SinCosIntegerPiKind::CosIntegerPiOdd
        },
        n,
    })
}

/// Rewrite odd/even parity identities:
/// - odd: f(-u) -> -f(u) for sin, tan, csc, cot, sinh, tanh
/// - even: f(-u) -> f(u)  for cos, sec, cosh
pub fn try_rewrite_trig_odd_even_parity_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<TrigOddEvenParityRewrite> {
    let (fn_id, args) = match ctx.get(expr) {
        Expr::Function(fn_id, args) => (*fn_id, args.clone()),
        _ => return None,
    };
    if args.len() != 1 {
        return None;
    }
    let builtin = ctx.builtin_of(fn_id)?;
    let name = builtin.name().to_string();
    let arg = args[0];

    let negated_info: Option<(ExprId, Option<num_rational::BigRational>)> =
        if let Some(inner) = as_neg(ctx, arg) {
            Some((inner, None))
        } else if let Expr::Number(n) = ctx.get(arg) {
            let zero = num_rational::BigRational::from_integer(0.into());
            if *n < zero {
                let pos = ctx.add(Expr::Number(-n.clone()));
                Some((pos, None))
            } else {
                None
            }
        } else if let Some((a, b)) = as_mul(ctx, arg) {
            if let Expr::Number(n) = ctx.get(a) {
                if *n < num_rational::BigRational::from_integer(0.into()) {
                    Some((b, Some(-n.clone())))
                } else {
                    None
                }
            } else if let Expr::Number(n) = ctx.get(b) {
                if *n < num_rational::BigRational::from_integer(0.into()) {
                    Some((a, Some(-n.clone())))
                } else {
                    None
                }
            } else {
                None
            }
        } else if let Some((a, b)) = as_sub(ctx, arg) {
            if compare_expr(ctx, a, b) == std::cmp::Ordering::Less {
                let flipped = ctx.add(Expr::Sub(b, a));
                Some((flipped, None))
            } else {
                None
            }
        } else {
            try_extract_all_negative_sum(ctx, arg)
        };

    let (base, opt_coeff) = negated_info?;
    let positive_arg = if let Some(coeff) = opt_coeff {
        if coeff == num_rational::BigRational::from_integer(1.into()) {
            base
        } else {
            let c = ctx.add(Expr::Number(coeff));
            ctx.add(Expr::Mul(c, base))
        }
    } else {
        base
    };

    match name.as_str() {
        "sin" | "tan" | "csc" | "cot" | "sinh" | "tanh" => {
            let f_u = ctx.add(Expr::Function(fn_id, vec![positive_arg]));
            let neg_f_u = ctx.add(Expr::Neg(f_u));
            Some(TrigOddEvenParityRewrite {
                rewritten: neg_f_u,
                fn_name: name,
                kind: TrigOddEvenParityKind::Odd,
            })
        }
        "cos" | "sec" | "cosh" => {
            let f_u = ctx.add(Expr::Function(fn_id, vec![positive_arg]));
            Some(TrigOddEvenParityRewrite {
                rewritten: f_u,
                fn_name: name,
                kind: TrigOddEvenParityKind::Even,
            })
        }
        _ => None,
    }
}

/// Legacy trig evaluator used by `EvaluateTrigRule` in engine identities.
///
/// Kept as support to thin the engine module while preserving behavior.
pub fn try_rewrite_legacy_evaluate_trig_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<LegacyTrigEvalRewrite> {
    let (fn_id, args) = match ctx.get(expr) {
        Expr::Function(fn_id, args) => (*fn_id, args.clone()),
        _ => return None,
    };
    let name = ctx.builtin_of(fn_id)?.name().to_string();
    if args.len() != 1 {
        return None;
    }
    let arg = args[0];

    let trig_fn = match name.as_str() {
        "sin" => Some(TrigFn::Sin),
        "cos" => Some(TrigFn::Cos),
        "tan" => Some(TrigFn::Tan),
        _ => None,
    };
    if let Some(f) = trig_fn {
        if let Some(result) = eval_trig_special(ctx, f, arg) {
            return Some(LegacyTrigEvalRewrite {
                rewritten: result,
                kind: LegacyTrigEvalRewriteKind::TableEval { fn_name: name },
            });
        }
    }

    let inv_trig_fn = match name.as_str() {
        "arcsin" | "asin" => Some(InvTrigFn::Asin),
        "arccos" | "acos" => Some(InvTrigFn::Acos),
        "arctan" | "atan" => Some(InvTrigFn::Atan),
        _ => None,
    };
    if let Some(f) = inv_trig_fn {
        if let Some(result) = eval_inv_trig_special(ctx, f, arg) {
            return Some(LegacyTrigEvalRewrite {
                rewritten: result,
                kind: LegacyTrigEvalRewriteKind::TableEval { fn_name: name },
            });
        }
    }

    if let Expr::Number(n) = ctx.get(arg) {
        if n.is_zero() {
            match name.as_str() {
                "sin" | "tan" | "arcsin" | "arctan" => {
                    return Some(LegacyTrigEvalRewrite {
                        rewritten: ctx.num(0),
                        kind: LegacyTrigEvalRewriteKind::ZeroEval { fn_name: name },
                    });
                }
                "cos" => {
                    return Some(LegacyTrigEvalRewrite {
                        rewritten: ctx.num(1),
                        kind: LegacyTrigEvalRewriteKind::CosZero,
                    });
                }
                "arccos" => {
                    let pi = ctx.add(Expr::Constant(cas_ast::Constant::Pi));
                    let two = ctx.num(2);
                    return Some(LegacyTrigEvalRewrite {
                        rewritten: ctx.add(Expr::Div(pi, two)),
                        kind: LegacyTrigEvalRewriteKind::ArccosZero,
                    });
                }
                _ => {}
            }
        } else if n.is_one() {
            match name.as_str() {
                "arcsin" => {
                    let pi = ctx.add(Expr::Constant(cas_ast::Constant::Pi));
                    let two = ctx.num(2);
                    return Some(LegacyTrigEvalRewrite {
                        rewritten: ctx.add(Expr::Div(pi, two)),
                        kind: LegacyTrigEvalRewriteKind::ArcsinOne,
                    });
                }
                "arccos" => {
                    return Some(LegacyTrigEvalRewrite {
                        rewritten: ctx.num(0),
                        kind: LegacyTrigEvalRewriteKind::ArccosOne,
                    });
                }
                "arctan" => {
                    let pi = ctx.add(Expr::Constant(cas_ast::Constant::Pi));
                    let four = ctx.num(4);
                    return Some(LegacyTrigEvalRewrite {
                        rewritten: ctx.add(Expr::Div(pi, four)),
                        kind: LegacyTrigEvalRewriteKind::ArctanOne,
                    });
                }
                _ => {}
            }
        } else if *n == num_rational::BigRational::new(1.into(), 2.into()) {
            match name.as_str() {
                "arcsin" => {
                    let pi = ctx.add(Expr::Constant(cas_ast::Constant::Pi));
                    let six = ctx.num(6);
                    return Some(LegacyTrigEvalRewrite {
                        rewritten: ctx.add(Expr::Div(pi, six)),
                        kind: LegacyTrigEvalRewriteKind::ArcsinHalf,
                    });
                }
                "arccos" => {
                    let pi = ctx.add(Expr::Constant(cas_ast::Constant::Pi));
                    let three = ctx.num(3);
                    return Some(LegacyTrigEvalRewrite {
                        rewritten: ctx.add(Expr::Div(pi, three)),
                        kind: LegacyTrigEvalRewriteKind::ArccosHalf,
                    });
                }
                _ => {}
            }
        }
    }

    if is_pi(ctx, arg) {
        match name.as_str() {
            "sin" | "tan" => {
                return Some(LegacyTrigEvalRewrite {
                    rewritten: ctx.num(0),
                    kind: LegacyTrigEvalRewriteKind::PiZero { fn_name: name },
                });
            }
            "cos" => {
                return Some(LegacyTrigEvalRewrite {
                    rewritten: ctx.num(-1),
                    kind: LegacyTrigEvalRewriteKind::CosPi,
                });
            }
            _ => {}
        }
    }

    if is_pi_over_n(ctx, arg, 2) {
        match name.as_str() {
            "sin" => {
                return Some(LegacyTrigEvalRewrite {
                    rewritten: ctx.num(1),
                    kind: LegacyTrigEvalRewriteKind::SinPiOver2,
                });
            }
            "cos" => {
                return Some(LegacyTrigEvalRewrite {
                    rewritten: ctx.num(0),
                    kind: LegacyTrigEvalRewriteKind::CosPiOver2,
                });
            }
            "tan" => {
                return Some(LegacyTrigEvalRewrite {
                    rewritten: ctx.add(Expr::Constant(cas_ast::Constant::Undefined)),
                    kind: LegacyTrigEvalRewriteKind::TanPiOver2,
                });
            }
            _ => {}
        }
    }

    if is_pi_over_n(ctx, arg, 3) {
        match name.as_str() {
            "sin" => {
                let three = ctx.num(3);
                let one = ctx.num(1);
                let two = ctx.num(2);
                let half_exp = ctx.add(Expr::Div(one, two));
                let sqrt3 = ctx.add(Expr::Pow(three, half_exp));
                let two2 = ctx.num(2);
                return Some(LegacyTrigEvalRewrite {
                    rewritten: ctx.add(Expr::Div(sqrt3, two2)),
                    kind: LegacyTrigEvalRewriteKind::SinPiOver3,
                });
            }
            "cos" => {
                let one = ctx.num(1);
                let two = ctx.num(2);
                return Some(LegacyTrigEvalRewrite {
                    rewritten: ctx.add(Expr::Div(one, two)),
                    kind: LegacyTrigEvalRewriteKind::CosPiOver3,
                });
            }
            "tan" => {
                let three = ctx.num(3);
                let one = ctx.num(1);
                let two = ctx.num(2);
                let half_exp = ctx.add(Expr::Div(one, two));
                return Some(LegacyTrigEvalRewrite {
                    rewritten: ctx.add(Expr::Pow(three, half_exp)),
                    kind: LegacyTrigEvalRewriteKind::TanPiOver3,
                });
            }
            _ => {}
        }
    }

    if is_pi_over_n(ctx, arg, 4) {
        match name.as_str() {
            "sin" | "cos" => {
                let two = ctx.num(2);
                let one = ctx.num(1);
                let two2 = ctx.num(2);
                let half_exp = ctx.add(Expr::Div(one, two2));
                let sqrt2 = ctx.add(Expr::Pow(two, half_exp));
                let two3 = ctx.num(2);
                return Some(LegacyTrigEvalRewrite {
                    rewritten: ctx.add(Expr::Div(sqrt2, two3)),
                    kind: LegacyTrigEvalRewriteKind::PiOver4Trig { fn_name: name },
                });
            }
            "tan" => {
                return Some(LegacyTrigEvalRewrite {
                    rewritten: ctx.num(1),
                    kind: LegacyTrigEvalRewriteKind::TanPiOver4,
                });
            }
            _ => {}
        }
    }

    if is_pi_over_n(ctx, arg, 6) {
        match name.as_str() {
            "sin" => {
                let one = ctx.num(1);
                let two = ctx.num(2);
                return Some(LegacyTrigEvalRewrite {
                    rewritten: ctx.add(Expr::Div(one, two)),
                    kind: LegacyTrigEvalRewriteKind::SinPiOver6,
                });
            }
            "cos" => {
                let three = ctx.num(3);
                let one = ctx.num(1);
                let two = ctx.num(2);
                let half_exp = ctx.add(Expr::Div(one, two));
                let sqrt3 = ctx.add(Expr::Pow(three, half_exp));
                let two2 = ctx.num(2);
                return Some(LegacyTrigEvalRewrite {
                    rewritten: ctx.add(Expr::Div(sqrt3, two2)),
                    kind: LegacyTrigEvalRewriteKind::CosPiOver6,
                });
            }
            "tan" => {
                let three = ctx.num(3);
                let one = ctx.num(1);
                let two = ctx.num(2);
                let half_exp = ctx.add(Expr::Div(one, two));
                let sqrt3 = ctx.add(Expr::Pow(three, half_exp));
                let one2 = ctx.num(1);
                return Some(LegacyTrigEvalRewrite {
                    rewritten: ctx.add(Expr::Div(one2, sqrt3)),
                    kind: LegacyTrigEvalRewriteKind::TanPiOver6,
                });
            }
            _ => {}
        }
    }

    if let Some(inner) = extract_negated_inner(ctx, arg) {
        match name.as_str() {
            "sin" => {
                let sin_inner = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![inner]);
                return Some(LegacyTrigEvalRewrite {
                    rewritten: ctx.add(Expr::Neg(sin_inner)),
                    kind: LegacyTrigEvalRewriteKind::SinNegative,
                });
            }
            "cos" => {
                return Some(LegacyTrigEvalRewrite {
                    rewritten: ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![inner]),
                    kind: LegacyTrigEvalRewriteKind::CosNegative,
                });
            }
            "tan" => {
                let tan_inner = ctx.call_builtin(cas_ast::BuiltinFn::Tan, vec![inner]);
                return Some(LegacyTrigEvalRewrite {
                    rewritten: ctx.add(Expr::Neg(tan_inner)),
                    kind: LegacyTrigEvalRewriteKind::TanNegative,
                });
            }
            _ => {}
        }
    }

    None
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PythagoreanTrigTerm {
    term_index: usize,
    coeff: ExprId,
    func_name: String,
    arg: ExprId,
    is_negated: bool,
}

fn extract_pythagorean_trig_parts(
    ctx: &mut Context,
    term: ExprId,
) -> Vec<(ExprId, String, ExprId, bool)> {
    let mut results = Vec::new();

    let (inner_term, is_negated) = match ctx.get(term) {
        Expr::Neg(inner) => (*inner, true),
        _ => (term, false),
    };

    let inner_shape = match ctx.get(inner_term) {
        Expr::Pow(b, e) => Some((*b, *e)),
        _ => None,
    };
    let is_mul = matches!(ctx.get(inner_term), Expr::Mul(_, _));

    if let Some((base, exp)) = inner_shape {
        if let Expr::Number(n) = ctx.get(exp) {
            if n.clone() >= num_rational::BigRational::from_integer(2.into()) && n.is_integer() {
                let trig_info = if let Expr::Function(fn_id, args) = ctx.get(base) {
                    let builtin = ctx.builtin_of(*fn_id);
                    if matches!(
                        builtin,
                        Some(cas_ast::BuiltinFn::Sin | cas_ast::BuiltinFn::Cos)
                    ) && args.len() == 1
                    {
                        builtin.map(|b| (b.name().to_string(), args[0]))
                    } else {
                        None
                    }
                } else {
                    None
                };

                if let Some((name, arg)) = trig_info {
                    let two = num_rational::BigRational::from_integer(2.into());
                    if n.clone() == two {
                        results.push((ctx.num(1), name, arg, is_negated));
                    } else {
                        let rem_exp = n.clone() - two;
                        if rem_exp.is_one() {
                            results.push((base, name, arg, is_negated));
                        } else {
                            let rem_exp_expr = ctx.add(Expr::Number(rem_exp));
                            let rem_pow = ctx.add(Expr::Pow(base, rem_exp_expr));
                            results.push((rem_pow, name, arg, is_negated));
                        }
                    }
                    return results;
                }
            }
        }
    }

    if is_mul {
        let factors = mul_leaves(ctx, inner_term);

        for (i, &factor) in factors.iter().enumerate() {
            let pow_data = match ctx.get(factor) {
                Expr::Pow(b, e) => Some((*b, *e)),
                _ => None,
            };
            if let Some((base, exp)) = pow_data {
                if let Expr::Number(n) = ctx.get(exp) {
                    if n.clone() >= num_rational::BigRational::from_integer(2.into())
                        && n.is_integer()
                    {
                        if let Expr::Function(fn_id, args) = ctx.get(base) {
                            let builtin = ctx.builtin_of(*fn_id);
                            if matches!(
                                builtin,
                                Some(cas_ast::BuiltinFn::Sin | cas_ast::BuiltinFn::Cos)
                            ) && args.len() == 1
                            {
                                let arg = args[0];
                                let Some(b) = builtin else {
                                    continue;
                                };
                                let name_str = b.name().to_string();

                                let two = num_rational::BigRational::from_integer(2.into());
                                let trig_rem = if n.clone() > two {
                                    let rem_exp = n.clone() - two;
                                    if rem_exp.is_one() {
                                        Some(base)
                                    } else {
                                        let rem_exp_expr = ctx.add(Expr::Number(rem_exp));
                                        Some(ctx.add(Expr::Pow(base, rem_exp_expr)))
                                    }
                                } else {
                                    None
                                };

                                let mut coeff_factors = Vec::new();
                                for (j, &f) in factors.iter().enumerate() {
                                    if j != i {
                                        coeff_factors.push(f);
                                    }
                                }
                                if let Some(rem) = trig_rem {
                                    coeff_factors.push(rem);
                                }

                                let coeff = if coeff_factors.is_empty() {
                                    ctx.num(1)
                                } else {
                                    let mut c = coeff_factors[0];
                                    for &f in coeff_factors.iter().skip(1) {
                                        c = smart_mul(ctx, c, f);
                                    }
                                    c
                                };
                                results.push((coeff, name_str, arg, is_negated));
                            }
                        }
                    }
                }
            }
        }
    }

    results
}

/// Rewrite generalized Pythagorean additive pairs:
/// - `a*sin(x)^2 + a*cos(x)^2 -> a`
/// - `-a*sin(x)^2 - a*cos(x)^2 -> -a`
pub fn try_rewrite_pythagorean_identity_add_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<PythagoreanIdentityRewrite> {
    if !matches!(ctx.get(expr), Expr::Add(_, _)) {
        return None;
    }

    let terms = add_leaves(ctx, expr);

    let mut trig_terms = Vec::new();
    for (i, &term) in terms.iter().enumerate() {
        for (coeff, name, arg, is_negated) in extract_pythagorean_trig_parts(ctx, term) {
            trig_terms.push(PythagoreanTrigTerm {
                term_index: i,
                coeff,
                func_name: name,
                arg,
                is_negated,
            });
        }
    }

    for i in 0..trig_terms.len() {
        for j in (i + 1)..trig_terms.len() {
            let t1 = &trig_terms[i];
            let t2 = &trig_terms[j];

            if t1.func_name == t2.func_name {
                continue;
            }
            let args_equal = t1.arg == t2.arg || compare_expr(ctx, t1.arg, t2.arg).is_eq();
            if !args_equal {
                continue;
            }
            let coeff_equal = t1.coeff == t2.coeff || compare_expr(ctx, t1.coeff, t2.coeff).is_eq();
            if !coeff_equal || t1.is_negated != t2.is_negated {
                continue;
            }

            let mut new_terms = Vec::new();
            for (k, &term) in terms.iter().enumerate() {
                if k != t1.term_index && k != t2.term_index {
                    new_terms.push(term);
                }
            }

            let result_coeff = if t1.is_negated {
                ctx.add(Expr::Neg(t1.coeff))
            } else {
                t1.coeff
            };
            new_terms.push(result_coeff);

            if new_terms.is_empty() {
                return Some(PythagoreanIdentityRewrite {
                    rewritten: ctx.num(0),
                    kind: PythagoreanIdentityRewriteKind::Empty,
                });
            }

            let mut rewritten = new_terms[0];
            for &term in new_terms.iter().skip(1) {
                rewritten = ctx.add(Expr::Add(rewritten, term));
            }

            let kind = if t1.is_negated {
                PythagoreanIdentityRewriteKind::Negated
            } else {
                PythagoreanIdentityRewriteKind::Standard
            };
            return Some(PythagoreanIdentityRewrite { rewritten, kind });
        }
    }

    None
}

/// True when angle-sum expansion should be blocked in normal simplify mode.
///
/// In non-expand mode, only allow `sin/cos/tan` over `Add/Sub` where both sides
/// contain at least one variable, or `sin/cos/tan((Add/Sub)/c)` with same property.
pub fn should_block_angle_identity_non_expand_mode(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Function(_, args) if args.len() == 1 => match ctx.get(args[0]) {
            Expr::Add(l, r) | Expr::Sub(l, r) => {
                !contains_variable(ctx, *l) || !contains_variable(ctx, *r)
            }
            Expr::Div(num, _) => match ctx.get(*num) {
                Expr::Add(l, r) | Expr::Sub(l, r) => {
                    !contains_variable(ctx, *l) || !contains_variable(ctx, *r)
                }
                _ => true,
            },
            _ => true,
        },
        _ => true,
    }
}

/// True when angle-sum expansion should be blocked due to large coefficients.
pub fn should_block_angle_identity_large_coeff(ctx: &Context, expr: ExprId) -> bool {
    if let Expr::Function(fn_id, args) = ctx.get(expr) {
        return matches!(
            ctx.builtin_of(*fn_id),
            Some(cas_ast::BuiltinFn::Sin | cas_ast::BuiltinFn::Cos | cas_ast::BuiltinFn::Tan)
        ) && args.len() == 1
            && has_large_coefficient(ctx, args[0]);
    }
    false
}

/// True when angle-sum expansion should be blocked because an Add/Sub operand
/// already contains a multiple-angle term.
pub fn should_block_angle_identity_multiple_angle(ctx: &Context, expr: ExprId) -> bool {
    if let Expr::Function(fn_id, args) = ctx.get(expr) {
        if matches!(
            ctx.builtin_of(*fn_id),
            Some(cas_ast::BuiltinFn::Sin | cas_ast::BuiltinFn::Cos)
        ) && args.len() == 1
        {
            let inner = args[0];
            if let Expr::Add(lhs, rhs) | Expr::Sub(lhs, rhs) = ctx.get(inner) {
                return is_multiple_angle(ctx, *lhs) || is_multiple_angle(ctx, *rhs);
            }
        }
    }
    false
}

/// Unified policy gate for angle-sum/-diff expansion (`sin/cos` over `Add/Sub`).
///
/// Returns `true` when expansion should be blocked for the current context.
pub fn should_block_angle_identity_expr(
    ctx: &Context,
    expr: ExprId,
    is_expand_mode: bool,
    marks: Option<&PatternMarks>,
    trig_large_coeff_protected: bool,
) -> bool {
    if !is_expand_mode && should_block_angle_identity_non_expand_mode(ctx, expr) {
        return true;
    }
    if should_block_angle_identity_large_coeff(ctx, expr) {
        return true;
    }
    if marks.is_some_and(|m| m.is_trig_square_protected(expr)) {
        return true;
    }
    if trig_large_coeff_protected {
        return true;
    }
    should_block_angle_identity_multiple_angle(ctx, expr)
}

/// Rewrite angle-sum/-difference identities for `sin`/`cos`.
///
/// This covers:
/// - `sin(a+b), sin(a-b), sin((a+b)/c)`
/// - `cos(a+b), cos(a-b), cos((a+b)/c)`
pub fn try_rewrite_angle_sum_diff_identity_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<AngleSumDiffIdentityRewrite> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }
    let inner = args[0];

    match ctx.builtin_of(*fn_id) {
        Some(cas_ast::BuiltinFn::Sin) => {
            if let Some((lhs, rhs)) = as_add(ctx, inner) {
                let sin_a = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![lhs]);
                let cos_b = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![rhs]);
                let term1 = smart_mul(ctx, sin_a, cos_b);

                let cos_a = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![lhs]);
                let sin_b = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![rhs]);
                let term2 = smart_mul(ctx, cos_a, sin_b);

                return Some(AngleSumDiffIdentityRewrite {
                    rewritten: ctx.add(Expr::Add(term1, term2)),
                    kind: AngleSumDiffIdentityKind::SinAdd,
                });
            } else if let Some((lhs, rhs)) = as_sub(ctx, inner) {
                let sin_a = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![lhs]);
                let cos_b = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![rhs]);
                let term1 = smart_mul(ctx, sin_a, cos_b);

                let cos_a = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![lhs]);
                let sin_b = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![rhs]);
                let term2 = smart_mul(ctx, cos_a, sin_b);

                return Some(AngleSumDiffIdentityRewrite {
                    rewritten: ctx.add(Expr::Sub(term1, term2)),
                    kind: AngleSumDiffIdentityKind::SinSub,
                });
            } else if let Some((num, den)) = as_div(ctx, inner) {
                if let Some((lhs, rhs)) = as_add(ctx, num) {
                    let a = ctx.add(Expr::Div(lhs, den));
                    let b = ctx.add(Expr::Div(rhs, den));

                    let sin_a = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![a]);
                    let cos_b = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![b]);
                    let term1 = smart_mul(ctx, sin_a, cos_b);

                    let cos_a = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![a]);
                    let sin_b = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![b]);
                    let term2 = smart_mul(ctx, cos_a, sin_b);

                    return Some(AngleSumDiffIdentityRewrite {
                        rewritten: ctx.add(Expr::Add(term1, term2)),
                        kind: AngleSumDiffIdentityKind::SinDivAdd,
                    });
                }
            }
        }
        Some(cas_ast::BuiltinFn::Cos) => {
            if let Some((lhs, rhs)) = as_add(ctx, inner) {
                let cos_a = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![lhs]);
                let cos_b = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![rhs]);
                let term1 = smart_mul(ctx, cos_a, cos_b);

                let sin_a = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![lhs]);
                let sin_b = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![rhs]);
                let term2 = smart_mul(ctx, sin_a, sin_b);

                return Some(AngleSumDiffIdentityRewrite {
                    rewritten: ctx.add(Expr::Sub(term1, term2)),
                    kind: AngleSumDiffIdentityKind::CosAdd,
                });
            } else if let Some((lhs, rhs)) = as_sub(ctx, inner) {
                let cos_a = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![lhs]);
                let cos_b = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![rhs]);
                let term1 = smart_mul(ctx, cos_a, cos_b);

                let sin_a = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![lhs]);
                let sin_b = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![rhs]);
                let term2 = smart_mul(ctx, sin_a, sin_b);

                return Some(AngleSumDiffIdentityRewrite {
                    rewritten: ctx.add(Expr::Add(term1, term2)),
                    kind: AngleSumDiffIdentityKind::CosSub,
                });
            } else if let Some((num, den)) = as_div(ctx, inner) {
                if let Some((lhs, rhs)) = as_add(ctx, num) {
                    let a = ctx.add(Expr::Div(lhs, den));
                    let b = ctx.add(Expr::Div(rhs, den));

                    let cos_a = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![a]);
                    let cos_b = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![b]);
                    let term1 = smart_mul(ctx, cos_a, cos_b);

                    let sin_a = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![a]);
                    let sin_b = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![b]);
                    let term2 = smart_mul(ctx, sin_a, sin_b);

                    return Some(AngleSumDiffIdentityRewrite {
                        rewritten: ctx.add(Expr::Sub(term1, term2)),
                        kind: AngleSumDiffIdentityKind::CosDivAdd,
                    });
                }
            }
        }
        _ => {}
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    fn render(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn rewrites_sin_integer_pi() {
        let mut ctx = Context::new();
        let expr = parse("sin(3*pi)", &mut ctx).expect("parse");
        let rewrite = try_rewrite_sin_cos_integer_pi_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(render(&ctx, rewrite.rewritten), "0");
        assert_eq!(rewrite.kind, SinCosIntegerPiKind::SinIntegerPi);
        assert_eq!(rewrite.n, 3.into());
    }

    #[test]
    fn rewrites_cos_integer_pi() {
        let mut ctx = Context::new();
        let expr = parse("cos(4*pi)", &mut ctx).expect("parse");
        let rewrite = try_rewrite_sin_cos_integer_pi_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(render(&ctx, rewrite.rewritten), "1");
        assert_eq!(rewrite.kind, SinCosIntegerPiKind::CosIntegerPiEven);
        assert_eq!(rewrite.n, 4.into());
    }

    #[test]
    fn rewrites_odd_parity() {
        let mut ctx = Context::new();
        let expr = parse("sin(-x)", &mut ctx).expect("parse");
        let rewrite = try_rewrite_trig_odd_even_parity_expr(&mut ctx, expr).expect("rewrite");
        let rendered = render(&ctx, rewrite.rewritten);
        assert!(rendered.contains("-sin(x)"));
        assert_eq!(rewrite.kind, TrigOddEvenParityKind::Odd);
    }

    #[test]
    fn rewrites_even_parity() {
        let mut ctx = Context::new();
        let expr = parse("cos(-x)", &mut ctx).expect("parse");
        let rewrite = try_rewrite_trig_odd_even_parity_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(render(&ctx, rewrite.rewritten), "cos(x)");
        assert_eq!(rewrite.kind, TrigOddEvenParityKind::Even);
    }

    #[test]
    fn rewrites_odd_parity_after_sub_flip() {
        let mut ctx = Context::new();
        let expr = parse("sin(1 - u)", &mut ctx).expect("parse");
        let rewrite = try_rewrite_trig_odd_even_parity_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(render(&ctx, rewrite.rewritten), "-sin(u - 1)");
        assert_eq!(rewrite.kind, TrigOddEvenParityKind::Odd);
    }

    #[test]
    fn rewrites_even_parity_after_sub_flip() {
        let mut ctx = Context::new();
        let expr = parse("cos(1 - u)", &mut ctx).expect("parse");
        let rewrite = try_rewrite_trig_odd_even_parity_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(render(&ctx, rewrite.rewritten), "cos(u - 1)");
        assert_eq!(rewrite.kind, TrigOddEvenParityKind::Even);
    }

    #[test]
    fn rewrites_pythagorean_identity_basic() {
        let mut ctx = Context::new();
        let expr = parse("sin(x)^2 + cos(x)^2", &mut ctx).expect("parse");
        let rewrite = try_rewrite_pythagorean_identity_add_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(render(&ctx, rewrite.rewritten), "1");
        assert_eq!(rewrite.kind, PythagoreanIdentityRewriteKind::Standard);
    }

    #[test]
    fn rewrites_pythagorean_identity_negated() {
        let mut ctx = Context::new();
        let expr = parse("-sin(x)^2 + -cos(x)^2", &mut ctx).expect("parse");
        let rewrite = try_rewrite_pythagorean_identity_add_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(render(&ctx, rewrite.rewritten), "-1");
        assert_eq!(rewrite.kind, PythagoreanIdentityRewriteKind::Negated);
    }

    #[test]
    fn blocks_angle_identity_non_expand_for_constant_summand() {
        let mut ctx = Context::new();
        let expr = parse("sin(x+1)", &mut ctx).expect("parse");
        assert!(should_block_angle_identity_non_expand_mode(&ctx, expr));
    }

    #[test]
    fn allows_angle_identity_non_expand_for_two_variable_summands() {
        let mut ctx = Context::new();
        let expr = parse("sin(x+y)", &mut ctx).expect("parse");
        assert!(!should_block_angle_identity_non_expand_mode(&ctx, expr));
    }

    #[test]
    fn blocks_angle_identity_large_coeff_guard() {
        let mut ctx = Context::new();
        let expr = parse("sin(16*x)", &mut ctx).expect("parse");
        assert!(should_block_angle_identity_large_coeff(&ctx, expr));
    }

    #[test]
    fn unified_angle_identity_gate_blocks_non_expand_constant_summand() {
        let mut ctx = Context::new();
        let expr = parse("sin(x+1)", &mut ctx).expect("parse");
        assert!(should_block_angle_identity_expr(
            &ctx, expr, false, None, false
        ));
    }

    #[test]
    fn unified_angle_identity_gate_allows_expand_variable_sum() {
        let mut ctx = Context::new();
        let expr = parse("sin(x+y)", &mut ctx).expect("parse");
        assert!(!should_block_angle_identity_expr(
            &ctx, expr, true, None, false
        ));
    }

    #[test]
    fn unified_angle_identity_gate_blocks_parent_large_coeff_protection() {
        let mut ctx = Context::new();
        let expr = parse("sin(x+y)", &mut ctx).expect("parse");
        assert!(should_block_angle_identity_expr(
            &ctx, expr, true, None, true
        ));
    }

    #[test]
    fn rewrites_angle_sum_for_sin() {
        let mut ctx = Context::new();
        let expr = parse("sin(x+y)", &mut ctx).expect("parse");
        let rewrite = try_rewrite_angle_sum_diff_identity_expr(&mut ctx, expr).expect("rewrite");
        let out = render(&ctx, rewrite.rewritten);
        assert!(out.contains("sin(x)"));
        assert!(out.contains("cos(y)"));
    }

    #[test]
    fn rewrites_angle_diff_for_cos() {
        let mut ctx = Context::new();
        let expr = parse("cos(a-b)", &mut ctx).expect("parse");
        let rewrite = try_rewrite_angle_sum_diff_identity_expr(&mut ctx, expr).expect("rewrite");
        let out = render(&ctx, rewrite.rewritten);
        assert!(out.contains("cos(a)"));
        assert!(out.contains("sin(b)"));
    }
}
