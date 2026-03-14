use crate::expr_destructure::{as_add, as_mul};
use crate::expr_nary::{AddView, MulView, Sign};
use crate::pi_helpers::{extract_rational_pi_multiple, is_pi, is_pi_over_n};
use crate::poly_compare::poly_eq;
use cas_ast::{Context, Expr, ExprId};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Zero};

/// Extract the coefficient of π from an expression.
/// - π -> 1
/// - k*π -> k
/// - π*k -> k
pub fn extract_pi_coefficient(ctx: &Context, expr: ExprId) -> Option<i32> {
    if is_pi(ctx, expr) {
        return Some(1);
    }

    if let Expr::Mul(l, r) = ctx.get(expr) {
        if is_pi(ctx, *r) {
            if let Expr::Number(n) = ctx.get(*l) {
                if n.is_integer() {
                    return n.to_integer().try_into().ok();
                }
            }
        }
        if is_pi(ctx, *l) {
            if let Expr::Number(n) = ctx.get(*r) {
                if n.is_integer() {
                    return n.to_integer().try_into().ok();
                }
            }
        }
    }

    None
}

/// Extract `k` from expressions equivalent to `k*π/2` with integer `k`.
pub fn extract_pi_half_multiple(ctx: &Context, expr: ExprId) -> Option<i32> {
    if is_pi_over_n(ctx, expr, 2) {
        return Some(1);
    }

    if is_pi(ctx, expr) {
        return Some(2);
    }

    if let Expr::Mul(l, r) = ctx.get(expr) {
        if let Expr::Number(n) = ctx.get(*l) {
            if is_pi_over_n(ctx, *r, 2) && n.is_integer() {
                if let Ok(k) = n.to_integer().try_into() {
                    return Some(k);
                }
            }
            if is_pi(ctx, *r) && n.is_integer() {
                if let Ok(k_half) = n.to_integer().try_into() {
                    let k: i32 = k_half;
                    return Some(k * 2);
                }
            }
        }

        if let Expr::Number(n) = ctx.get(*r) {
            if is_pi_over_n(ctx, *l, 2) && n.is_integer() {
                if let Ok(k) = n.to_integer().try_into() {
                    return Some(k);
                }
            }
            if is_pi(ctx, *l) && n.is_integer() {
                if let Ok(k_half) = n.to_integer().try_into() {
                    let k: i32 = k_half;
                    return Some(k * 2);
                }
            }
        }
    }

    if let Expr::Div(num, den) = ctx.get(expr) {
        if let Expr::Number(d) = ctx.get(*den) {
            if d.is_integer() && *d == num_rational::BigRational::from_integer(2.into()) {
                if is_pi(ctx, *num) {
                    return Some(1);
                }
                if let Expr::Mul(l, r) = ctx.get(*num) {
                    if let Expr::Number(n) = ctx.get(*l) {
                        if is_pi(ctx, *r) && n.is_integer() {
                            if let Ok(k) = n.to_integer().try_into() {
                                return Some(k);
                            }
                        }
                    }
                    if let Expr::Number(n) = ctx.get(*r) {
                        if is_pi(ctx, *l) && n.is_integer() {
                            if let Ok(k) = n.to_integer().try_into() {
                                return Some(k);
                            }
                        }
                    }
                }
            }
        }
    }

    None
}

fn extract_positive_unit_fraction_denominator(ctx: &Context, expr: ExprId) -> Option<i32> {
    match ctx.get(expr) {
        Expr::Number(coeff) if coeff.numer() == &BigInt::from(1) && !coeff.denom().is_one() => {
            coeff.denom().try_into().ok()
        }
        Expr::Div(num, den) => match (ctx.get(*num), ctx.get(*den)) {
            (Expr::Number(num), Expr::Number(den))
                if num.is_one() && den.is_integer() && den.denom().is_one() =>
            {
                den.to_integer().try_into().ok()
            }
            _ => None,
        },
        _ => None,
    }
}

fn is_zero_expr(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Number(n) if n.is_zero())
}

fn add_signed_term(ctx: &mut Context, acc: ExprId, term: ExprId, sign: Sign) -> ExprId {
    if is_zero_expr(ctx, acc) {
        return match sign {
            Sign::Pos => term,
            Sign::Neg => ctx.add(Expr::Neg(term)),
        };
    }
    match sign {
        Sign::Pos => ctx.add(Expr::Add(acc, term)),
        Sign::Neg => ctx.add(Expr::Sub(acc, term)),
    }
}

fn mul_if_nontrivial(ctx: &mut Context, a: ExprId, b: ExprId) -> ExprId {
    if is_zero_expr(ctx, a) || is_zero_expr(ctx, b) {
        return ctx.num(0);
    }
    if matches!(ctx.get(a), Expr::Number(n) if n.is_one()) {
        return b;
    }
    if matches!(ctx.get(b), Expr::Number(n) if n.is_one()) {
        return a;
    }
    ctx.add(Expr::Mul(a, b))
}

fn split_pi_linear_component(ctx: &mut Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    let zero = ctx.num(0);
    let one = ctx.num(1);

    if is_pi(ctx, expr) {
        return Some((one, zero));
    }

    match ctx.get(expr) {
        Expr::Add(_, _) | Expr::Sub(_, _) => {
            let view = AddView::from_expr(ctx, expr);
            let mut coeff_acc = zero;
            let mut rest_acc = zero;
            for (term, sign) in view.terms {
                let (term_coeff, term_rest) = split_pi_linear_component(ctx, term)?;
                if !is_zero_expr(ctx, term_coeff) {
                    coeff_acc = add_signed_term(ctx, coeff_acc, term_coeff, sign);
                }
                if !is_zero_expr(ctx, term_rest) {
                    rest_acc = add_signed_term(ctx, rest_acc, term_rest, sign);
                }
            }
            Some((coeff_acc, rest_acc))
        }
        Expr::Neg(inner) => {
            let (coeff, rest) = split_pi_linear_component(ctx, *inner)?;
            let neg_coeff = if is_zero_expr(ctx, coeff) {
                coeff
            } else {
                ctx.add(Expr::Neg(coeff))
            };
            let neg_rest = if is_zero_expr(ctx, rest) {
                rest
            } else {
                ctx.add(Expr::Neg(rest))
            };
            Some((neg_coeff, neg_rest))
        }
        Expr::Mul(_, _) => {
            let view = MulView::from_expr(ctx, expr);
            let mut carrier_idx = None;
            let mut carrier_coeff = zero;
            let mut carrier_rest = zero;
            let mut pure_factors: smallvec::SmallVec<[ExprId; 8]> = smallvec::SmallVec::new();

            for (idx, factor) in view.factors.iter().copied().enumerate() {
                let (coeff, rest) = split_pi_linear_component(ctx, factor)?;
                if is_zero_expr(ctx, coeff) {
                    pure_factors.push(rest);
                } else {
                    if carrier_idx.is_some() {
                        return None;
                    }
                    carrier_idx = Some(idx);
                    carrier_coeff = coeff;
                    carrier_rest = rest;
                }
            }

            let Some(_) = carrier_idx else {
                return Some((zero, expr));
            };

            let pure_product = if pure_factors.is_empty() {
                one
            } else {
                MulView {
                    root: expr,
                    factors: pure_factors,
                    commutative: true,
                }
                .rebuild(ctx)
            };

            let coeff_out = mul_if_nontrivial(ctx, pure_product, carrier_coeff);
            let rest_out = if is_zero_expr(ctx, carrier_rest) {
                zero
            } else {
                mul_if_nontrivial(ctx, pure_product, carrier_rest)
            };
            Some((coeff_out, rest_out))
        }
        _ => Some((zero, expr)),
    }
}

fn extract_symbolic_den_phase_shift(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
) -> Option<(ExprId, i32)> {
    let (pi_coeff, base_num) = split_pi_linear_component(ctx, num)?;
    if is_zero_expr(ctx, pi_coeff) {
        return None;
    }
    let two = ctx.num(2);
    let lhs = ctx.add(Expr::Mul(two, pi_coeff));

    for k in [-3_i32, -2, -1, 1, 2, 3] {
        let abs_k = k.abs() as i64;
        let scaled_den = if abs_k == 1 {
            den
        } else {
            let coeff = ctx.num(abs_k);
            ctx.add(Expr::Mul(coeff, den))
        };
        let rhs = if k < 0 {
            ctx.add(Expr::Neg(scaled_den))
        } else {
            scaled_den
        };
        if poly_eq(ctx, lhs, rhs) {
            let base = if matches!(ctx.get(base_num), Expr::Number(n) if n.is_zero()) {
                base_num
            } else {
                ctx.add(Expr::Div(base_num, den))
            };
            return Some((base, k));
        }
    }

    None
}

/// Extract `(base_term, k)` from `expr` such that:
/// `expr = base_term + k*π/2`.
///
/// Handles canonical and n-ary forms:
/// - `Div(Add(n*x, k*pi), m)` when `m | (2k)`
/// - `Mul(1/n, Add(..., k*pi))`
/// - Any n-ary additive expression containing a `k*π/2` term
pub fn extract_phase_shift(ctx: &mut Context, expr: ExprId) -> Option<(ExprId, i32)> {
    // Form 1: Div((coeff*x + k*pi), denom) - canonical quotient form
    if let Expr::Div(num, den) = ctx.get(expr) {
        let num = *num;
        let den = *den;

        let denom_val: i32 = if let Expr::Number(n) = ctx.get(den) {
            if n.is_integer() {
                n.to_integer().try_into().ok()?
            } else {
                return None;
            }
        } else {
            return extract_symbolic_den_phase_shift(ctx, num, den);
        };

        if let Some((l, r)) = as_add(ctx, num) {
            if is_pi(ctx, r) {
                let k = 2 / denom_val;
                if 2 % denom_val == 0 {
                    let base = ctx.add(Expr::Div(l, den));
                    return Some((base, k));
                }
            }

            if is_pi(ctx, l) {
                let k = 2 / denom_val;
                if 2 % denom_val == 0 {
                    let base = ctx.add(Expr::Div(r, den));
                    return Some((base, k));
                }
            }

            if let Some(pi_coeff) = extract_pi_coefficient(ctx, r) {
                let k_times_2 = 2 * pi_coeff;
                if k_times_2 % denom_val == 0 {
                    let k = k_times_2 / denom_val;
                    let base = ctx.add(Expr::Div(l, den));
                    return Some((base, k));
                }
            }

            if let Some(pi_coeff) = extract_pi_coefficient(ctx, l) {
                let k_times_2 = 2 * pi_coeff;
                if k_times_2 % denom_val == 0 {
                    let k = k_times_2 / denom_val;
                    let base = ctx.add(Expr::Div(r, den));
                    return Some((base, k));
                }
            }
        }
    }

    // Form 1b: Mul(1/n, Add(coeff*x, k*pi)) - canonical lowered division form
    if let Some((coeff_id, inner)) = as_mul(ctx, expr) {
        if let Some(denom_val) = extract_positive_unit_fraction_denominator(ctx, coeff_id) {
            let view = AddView::from_expr(ctx, inner);
            if view.terms.len() >= 2 {
                for (i, (term, sign)) in view.terms.iter().enumerate() {
                    let mut k_times_2 = if is_pi(ctx, *term) {
                        2
                    } else if let Some(pi_coeff) = extract_pi_coefficient(ctx, *term) {
                        2 * pi_coeff
                    } else {
                        continue;
                    };

                    if *sign == Sign::Neg {
                        k_times_2 = -k_times_2;
                    }

                    if k_times_2 % denom_val == 0 {
                        let k = k_times_2 / denom_val;
                        let remaining: smallvec::SmallVec<[(ExprId, Sign); 8]> = view
                            .terms
                            .iter()
                            .enumerate()
                            .filter(|(j, _)| *j != i)
                            .map(|(_, t)| *t)
                            .collect();
                        let rest_view = AddView {
                            root: inner,
                            terms: remaining,
                        };
                        let base_inner = rest_view.rebuild(ctx);
                        let base = ctx.add(Expr::Mul(coeff_id, base_inner));
                        return Some((base, k));
                    }
                }
            }
        }
    }

    // Form 2/3: n-ary additive scan
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() >= 2 {
        for (i, (term, sign)) in view.terms.iter().enumerate() {
            if let Some(mut k) = extract_pi_half_multiple(ctx, *term) {
                if *sign == Sign::Neg {
                    k = -k;
                }
                let remaining: smallvec::SmallVec<[(ExprId, Sign); 8]> = view
                    .terms
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .map(|(_, t)| *t)
                    .collect();
                let rest_view = AddView {
                    root: expr,
                    terms: remaining,
                };
                let base = rest_view.rebuild(ctx);
                return Some((base, k));
            }
        }
    }

    None
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrigPhaseShiftRewriteOwned {
    pub rewritten: ExprId,
    pub function: TrigPhaseShiftFunctionKind,
    pub shift: TrigPhaseShiftKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrigSupplementaryAngleRewriteOwned {
    pub rewritten: ExprId,
    pub desc: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrigPhaseShiftFunctionKind {
    Sin,
    Cos,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrigPhaseShiftKind {
    PiOver2,
    NegPiOver2,
    Pi,
    NegPi,
    ThreePiOver2,
    NegThreePiOver2,
    KPiOver2,
}

/// Rewrites phase-shifted sin/cos forms:
/// - `sin(x + k*pi/2)` (or equivalent canonical forms)
/// - `cos(x + k*pi/2)`
pub fn try_rewrite_trig_phase_shift_function_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<TrigPhaseShiftRewriteOwned> {
    let (fn_id, arg) = match ctx.get(expr) {
        Expr::Function(fn_id, args) if args.len() == 1 => (*fn_id, args[0]),
        _ => return None,
    };

    let builtin = ctx.builtin_of(fn_id)?;
    let is_sin = matches!(builtin, cas_ast::BuiltinFn::Sin);
    let is_cos = matches!(builtin, cas_ast::BuiltinFn::Cos);
    if !is_sin && !is_cos {
        return None;
    }

    let (base_term, pi_multiple) = extract_phase_shift(ctx, arg)?;
    if pi_multiple == 0 {
        return None;
    }

    let k = ((pi_multiple % 4) + 4) % 4;
    let (new_builtin, negate) = if is_sin {
        match k {
            0 => (cas_ast::BuiltinFn::Sin, false),
            1 => (cas_ast::BuiltinFn::Cos, false),
            2 => (cas_ast::BuiltinFn::Sin, true),
            3 => (cas_ast::BuiltinFn::Cos, true),
            _ => return None,
        }
    } else {
        match k {
            0 => (cas_ast::BuiltinFn::Cos, false),
            1 => (cas_ast::BuiltinFn::Sin, true),
            2 => (cas_ast::BuiltinFn::Cos, true),
            3 => (cas_ast::BuiltinFn::Sin, false),
            _ => return None,
        }
    };

    let new_trig = ctx.call_builtin(new_builtin, vec![base_term]);
    let rewritten = if negate {
        ctx.add(Expr::Neg(new_trig))
    } else {
        new_trig
    };

    let function = if is_sin {
        TrigPhaseShiftFunctionKind::Sin
    } else {
        TrigPhaseShiftFunctionKind::Cos
    };
    let shift = match pi_multiple {
        1 => TrigPhaseShiftKind::PiOver2,
        -1 => TrigPhaseShiftKind::NegPiOver2,
        2 => TrigPhaseShiftKind::Pi,
        -2 => TrigPhaseShiftKind::NegPi,
        3 => TrigPhaseShiftKind::ThreePiOver2,
        -3 => TrigPhaseShiftKind::NegThreePiOver2,
        _ => TrigPhaseShiftKind::KPiOver2,
    };
    Some(TrigPhaseShiftRewriteOwned {
        rewritten,
        function,
        shift,
    })
}

/// Rewrites supplementary-angle forms:
/// - `sin(k*pi - x)` style rational-π cases to a simpler angle
/// - `cos(k*pi - x)` analogously
pub fn try_rewrite_supplementary_angle_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<TrigSupplementaryAngleRewriteOwned> {
    let (fn_id, args) = match ctx.get(expr) {
        Expr::Function(fn_id, args) => (*fn_id, args.clone()),
        _ => return None,
    };
    if args.len() != 1 {
        return None;
    }

    let builtin = ctx.builtin_of(fn_id);
    let is_sin = matches!(builtin, Some(cas_ast::BuiltinFn::Sin));
    let is_cos = matches!(builtin, Some(cas_ast::BuiltinFn::Cos));
    if !is_sin && !is_cos {
        return None;
    }
    let arg = args[0];

    let k = extract_rational_pi_multiple(ctx, arg)?;
    let p = k.numer();
    let q = k.denom();
    if p <= &BigInt::from(0) {
        return None;
    }

    let one = BigInt::from(1);
    let n_candidate = (p + q - &one) / q;
    let remainder = &n_candidate * q - p;
    if remainder <= BigInt::from(0) || &remainder >= p {
        return None;
    }

    let new_coeff = BigRational::new(remainder.clone(), q.clone());
    let new_angle = if new_coeff == BigRational::from_integer(1.into()) {
        ctx.add(Expr::Constant(cas_ast::Constant::Pi))
    } else {
        let pi = ctx.add(Expr::Constant(cas_ast::Constant::Pi));
        let coeff_expr = ctx.add(Expr::Number(new_coeff));
        ctx.add(Expr::Mul(coeff_expr, pi))
    };

    let n_parity_odd = &n_candidate % 2 == one;
    let (rewritten, desc) = if is_sin {
        let new_trig = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![new_angle]);
        if n_parity_odd {
            (new_trig, format!("sin({}π - x) = sin(x)", n_candidate))
        } else {
            (
                ctx.add(Expr::Neg(new_trig)),
                format!("sin({}π - x) = -sin(x)", n_candidate),
            )
        }
    } else {
        let new_trig = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![new_angle]);
        if n_parity_odd {
            (
                ctx.add(Expr::Neg(new_trig)),
                format!("cos({}π - x) = -cos(x)", n_candidate),
            )
        } else {
            (new_trig, format!("cos({}π - x) = cos(x)", n_candidate))
        }
    };

    Some(TrigSupplementaryAngleRewriteOwned { rewritten, desc })
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    #[test]
    fn extract_pi_coefficient_matches_mul_forms() {
        let mut ctx = Context::new();
        let pi = parse("pi", &mut ctx).expect("pi");
        let three_pi = parse("3*pi", &mut ctx).expect("3*pi");
        let pi_four = parse("pi*4", &mut ctx).expect("pi*4");
        let x = parse("x", &mut ctx).expect("x");

        assert_eq!(extract_pi_coefficient(&ctx, pi), Some(1));
        assert_eq!(extract_pi_coefficient(&ctx, three_pi), Some(3));
        assert_eq!(extract_pi_coefficient(&ctx, pi_four), Some(4));
        assert_eq!(extract_pi_coefficient(&ctx, x), None);
    }

    #[test]
    fn extract_pi_half_multiple_matches_common_forms() {
        let mut ctx = Context::new();
        let pi_half = parse("pi/2", &mut ctx).expect("pi/2");
        let pi = parse("pi", &mut ctx).expect("pi");
        let three_pi_half = parse("3*pi/2", &mut ctx).expect("3*pi/2");
        let five_pi = parse("5*pi", &mut ctx).expect("5*pi");

        assert_eq!(extract_pi_half_multiple(&ctx, pi_half), Some(1));
        assert_eq!(extract_pi_half_multiple(&ctx, pi), Some(2));
        assert_eq!(extract_pi_half_multiple(&ctx, three_pi_half), Some(3));
        assert_eq!(extract_pi_half_multiple(&ctx, five_pi), Some(10));
    }

    #[test]
    fn extract_phase_shift_from_div_add_form() {
        let mut ctx = Context::new();
        let expr = parse("(2*x + pi)/2", &mut ctx).expect("expr");
        let expected_base = parse("(2*x)/2", &mut ctx).expect("(2*x)/2");

        let (base, k) = extract_phase_shift(&mut ctx, expr).expect("phase shift");
        assert_eq!(base, expected_base);
        assert_eq!(k, 1);
    }

    #[test]
    fn extract_phase_shift_from_sub_form() {
        let mut ctx = Context::new();
        let expr = parse("x - pi/2", &mut ctx).expect("expr");
        let expected_base = parse("x", &mut ctx).expect("x");

        let (base, k) = extract_phase_shift(&mut ctx, expr).expect("phase shift");
        assert_eq!(base, expected_base);
        assert_eq!(k, -1);
    }

    #[test]
    fn extract_phase_shift_from_nary_add_form() {
        let mut ctx = Context::new();
        let expr = parse("x + y + pi/2", &mut ctx).expect("expr");
        let expected_base = parse("x + y", &mut ctx).expect("x+y");

        let (base, k) = extract_phase_shift(&mut ctx, expr).expect("phase shift");
        assert_eq!(base, expected_base);
        assert_eq!(k, 1);
    }

    #[test]
    fn extract_phase_shift_from_scaled_nary_add_form() {
        let mut ctx = Context::new();
        let expr = parse("1/2 * (3*pi + 2*x + 2)", &mut ctx).expect("expr");
        let expected_base = parse("1/2 * (2*x + 2)", &mut ctx).expect("expected");

        let (base, k) = extract_phase_shift(&mut ctx, expr).expect("phase shift");
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, base, expected_base),
            std::cmp::Ordering::Equal
        );
        assert_eq!(k, 3);
    }

    #[test]
    fn extract_phase_shift_from_symbolic_den_fraction_form() {
        let mut ctx = Context::new();
        let expr = parse("(pi*u*(u+1) + 2*u + 1)/(u*(u+1))", &mut ctx).expect("expr");
        let expected_base = parse("(2*u + 1)/(u*(u+1))", &mut ctx).expect("expected");

        let (base, k) = extract_phase_shift(&mut ctx, expr).expect("phase shift");
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, base, expected_base),
            std::cmp::Ordering::Equal
        );
        assert_eq!(k, 2);
    }

    #[test]
    fn rewrites_trig_phase_shift_from_symbolic_den_fraction_form() {
        let mut ctx = Context::new();
        let expr = parse("sin((pi*u*(u+1) + 2*u + 1)/(u*(u+1)))", &mut ctx).expect("expr");
        let rewrite = try_rewrite_trig_phase_shift_function_expr(&mut ctx, expr).expect("rewrite");
        let expected = parse("-sin((2*u + 1)/(u*(u+1)))", &mut ctx).expect("expected");
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, rewrite.rewritten, expected),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn rewrites_trig_phase_shift_function_expr() {
        let mut ctx = Context::new();
        let expr = parse("sin(x + pi/2)", &mut ctx).expect("expr");
        let rewrite = try_rewrite_trig_phase_shift_function_expr(&mut ctx, expr).expect("rewrite");
        let expected = parse("cos(x)", &mut ctx).expect("expected");
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, rewrite.rewritten, expected),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn rewrites_trig_phase_shift_from_scaled_nary_add_form() {
        let mut ctx = Context::new();
        let expr = parse("sin(1/2 * (2*x + 2 + pi))", &mut ctx).expect("expr");
        let rewrite = try_rewrite_trig_phase_shift_function_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(rewrite.function, TrigPhaseShiftFunctionKind::Sin);
        assert_eq!(rewrite.shift, TrigPhaseShiftKind::PiOver2);
        let rendered = format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: &ctx,
                id: rewrite.rewritten
            }
        );
        assert!(rendered.starts_with("cos("));
    }

    #[test]
    fn rewrites_supplementary_angle_expr() {
        let mut ctx = Context::new();
        let expr = parse("sin(8*pi/9)", &mut ctx).expect("expr");
        let rewrite = try_rewrite_supplementary_angle_expr(&mut ctx, expr).expect("rewrite");
        let rewritten_str = format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: &ctx,
                id: rewrite.rewritten
            }
        );
        assert_eq!(rewritten_str, "sin(1/9 * pi)");
    }
}
