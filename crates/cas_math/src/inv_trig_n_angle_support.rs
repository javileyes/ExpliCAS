use crate::trig_roots_flatten::extract_int_multiple;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_bigint::BigInt;
use num_rational::BigRational;

/// Build `expr^(1/2)` i.e. `sqrt(expr)`.
pub fn build_sqrt(ctx: &mut Context, expr: ExprId) -> ExprId {
    let half = ctx.add(Expr::Number(BigRational::new(
        BigInt::from(1),
        BigInt::from(2),
    )));
    ctx.add(Expr::Pow(expr, half))
}

/// Build `1 + t^2`.
pub fn build_one_plus_t_sq(ctx: &mut Context, t: ExprId) -> ExprId {
    let one = ctx.num(1);
    let two = ctx.num(2);
    let t_sq = ctx.add(Expr::Pow(t, two));
    ctx.add(Expr::Add(one, t_sq))
}

/// Build `1 - t^2`.
pub fn build_one_minus_t_sq(ctx: &mut Context, t: ExprId) -> ExprId {
    let one = ctx.num(1);
    let two = ctx.num(2);
    let t_sq = ctx.add(Expr::Pow(t, two));
    ctx.add(Expr::Sub(one, t_sq))
}

/// Count unique DAG nodes reachable from `root`.
pub fn count_nodes_dedup(ctx: &Context, root: ExprId) -> usize {
    let mut visited = std::collections::HashSet::new();
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        if !visited.insert(id) {
            continue;
        }
        match ctx.get(id) {
            Expr::Add(l, r)
            | Expr::Sub(l, r)
            | Expr::Mul(l, r)
            | Expr::Div(l, r)
            | Expr::Pow(l, r) => {
                stack.push(*l);
                stack.push(*r);
            }
            Expr::Neg(inner) => stack.push(*inner),
            Expr::Function(_, args) => stack.extend(args.iter()),
            _ => {}
        }
    }
    visited.len()
}

/// Build Weierstrass polynomials `A_n(t), B_n(t)` via recurrence.
pub fn weierstrass_recurrence(ctx: &mut Context, t: ExprId, n: usize) -> (ExprId, ExprId) {
    let mut a = ctx.num(1);
    let mut b = ctx.num(0);

    for _ in 0..n {
        let t_b = ctx.add(Expr::Mul(t, b));
        let t_a = ctx.add(Expr::Mul(t, a));
        let new_a = ctx.add(Expr::Sub(a, t_b));
        let new_b = ctx.add(Expr::Add(b, t_a));
        a = new_a;
        b = new_b;
    }

    (a, b)
}

/// Build Chebyshev `T_n(t)` via recurrence.
pub fn chebyshev_t(ctx: &mut Context, t: ExprId, n: usize) -> ExprId {
    if n == 0 {
        return ctx.num(1);
    }
    if n == 1 {
        return t;
    }
    let two = ctx.num(2);
    let two_t = ctx.add(Expr::Mul(two, t));
    let mut prev = ctx.num(1);
    let mut curr = t;
    for _ in 2..=n {
        let next = ctx.add(Expr::Mul(two_t, curr));
        let next = ctx.add(Expr::Sub(next, prev));
        prev = curr;
        curr = next;
    }
    curr
}

/// Build Chebyshev `U_{n-1}(t)` via recurrence (for `n >= 1`).
pub fn chebyshev_u_nm1(ctx: &mut Context, t: ExprId, n: usize) -> ExprId {
    debug_assert!(n >= 1, "chebyshev_u_nm1 requires n >= 1");
    if n == 1 {
        return ctx.num(1);
    }
    let two = ctx.num(2);
    let two_t = ctx.add(Expr::Mul(two, t));
    let mut prev = ctx.num(1);
    let mut curr = two_t;
    for _ in 2..n {
        let next = ctx.add(Expr::Mul(two_t, curr));
        let next = ctx.add(Expr::Sub(next, prev));
        prev = curr;
        curr = next;
    }
    curr
}

/// Build `sin/cos(n*arcsin(t))` via recurrence.
pub fn arcsin_recurrence(
    ctx: &mut Context,
    t: ExprId,
    cos_theta: ExprId,
    n: usize,
) -> (ExprId, ExprId) {
    let mut s = ctx.num(0);
    let mut c = ctx.num(1);

    for _ in 0..n {
        let s_cos = ctx.add(Expr::Mul(s, cos_theta));
        let c_t = ctx.add(Expr::Mul(c, t));
        let c_cos = ctx.add(Expr::Mul(c, cos_theta));
        let s_t = ctx.add(Expr::Mul(s, t));
        let new_s = ctx.add(Expr::Add(s_cos, c_t));
        let new_c = ctx.add(Expr::Sub(c_cos, s_t));
        s = new_s;
        c = new_c;
    }

    (s, c)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NAngleInverseTrigPlan {
    pub rewritten: ExprId,
    pub assume_nonzero_expr: ExprId,
    pub desc: String,
}

fn match_outer_trig_call(ctx: &Context, expr: ExprId) -> Option<(BuiltinFn, ExprId)> {
    let (fn_id, arg0) = match ctx.get(expr) {
        Expr::Function(fn_id, args) if args.len() == 1 => (*fn_id, args[0]),
        _ => return None,
    };

    match ctx.builtin_of(fn_id) {
        Some(b @ (BuiltinFn::Sin | BuiltinFn::Cos | BuiltinFn::Tan)) => Some((b, arg0)),
        _ => None,
    }
}

fn match_n_multiple(arg0: ExprId, ctx: &Context, max_n: i64) -> Option<(bool, i64, ExprId)> {
    if let Expr::Function(_, _) = ctx.get(arg0) {
        return Some((true, 1, arg0));
    }

    for k in 2..=max_n {
        if let Some((sign, inner)) = extract_int_multiple(ctx, arg0, k) {
            return Some((sign, k, inner));
        }
    }
    None
}

/// Plan `sin/cos/tan(n*atan(t))` via Weierstrass recurrence.
pub fn try_plan_n_angle_atan_expr(
    ctx: &mut Context,
    expr: ExprId,
    max_n: i64,
    max_inner_nodes: usize,
    max_output_nodes: usize,
) -> Option<NAngleInverseTrigPlan> {
    let (trig, arg0) = match_outer_trig_call(ctx, expr)?;
    let (is_positive, n, inner) = match_n_multiple(arg0, ctx, max_n)?;

    let t = match ctx.get(inner) {
        Expr::Function(inv_id, inv_args) if inv_args.len() == 1 => match ctx.builtin_of(*inv_id) {
            Some(BuiltinFn::Atan | BuiltinFn::Arctan) => inv_args[0],
            _ => return None,
        },
        _ => return None,
    };

    if count_nodes_dedup(ctx, t) > max_inner_nodes {
        return None;
    }

    let (a_n, b_n) = weierstrass_recurrence(ctx, t, n as usize);
    let (result, assume_nonzero_expr, desc) = match trig {
        BuiltinFn::Tan => {
            let result = ctx.add(Expr::Div(b_n, a_n));
            (result, a_n, format!("tan({n}·atan(t)) = Bₙ/Aₙ"))
        }
        BuiltinFn::Sin | BuiltinFn::Cos => {
            let one_plus_t_sq = build_one_plus_t_sq(ctx, t);
            let exp = ctx.add(Expr::Number(BigRational::new(
                BigInt::from(n),
                BigInt::from(2),
            )));
            let denom = ctx.add(Expr::Pow(one_plus_t_sq, exp));

            let (numerator, desc) = match trig {
                BuiltinFn::Sin => (b_n, format!("sin({n}·atan(t)) = Bₙ/(1+t²)^({n}/2)")),
                _ => (a_n, format!("cos({n}·atan(t)) = Aₙ/(1+t²)^({n}/2)")),
            };
            (ctx.add(Expr::Div(numerator, denom)), denom, desc)
        }
        _ => return None,
    };

    let rewritten = if !is_positive {
        match trig {
            BuiltinFn::Cos => result,
            _ => ctx.add(Expr::Neg(result)),
        }
    } else {
        result
    };

    if count_nodes_dedup(ctx, rewritten) > max_output_nodes {
        return None;
    }

    Some(NAngleInverseTrigPlan {
        rewritten,
        assume_nonzero_expr,
        desc,
    })
}

/// Plan `sin/cos/tan(n*acos(t))` via Chebyshev recurrence.
pub fn try_plan_n_angle_acos_expr(
    ctx: &mut Context,
    expr: ExprId,
    max_n: i64,
    max_inner_nodes: usize,
    max_output_nodes: usize,
) -> Option<NAngleInverseTrigPlan> {
    let (trig, arg0) = match_outer_trig_call(ctx, expr)?;
    let (is_positive, n, inner) = match_n_multiple(arg0, ctx, max_n)?;

    let t = match ctx.get(inner) {
        Expr::Function(inv_id, inv_args) if inv_args.len() == 1 => match ctx.builtin_of(*inv_id) {
            Some(BuiltinFn::Acos | BuiltinFn::Arccos) => inv_args[0],
            _ => return None,
        },
        _ => return None,
    };

    if count_nodes_dedup(ctx, t) > max_inner_nodes {
        return None;
    }

    let n_usize = n as usize;
    let (result, assume_nonzero_expr, desc) = match trig {
        BuiltinFn::Cos => {
            let tn = chebyshev_t(ctx, t, n_usize);
            (tn, ctx.num(1), format!("cos({n}·arccos(t)) = T_{n}(t)"))
        }
        BuiltinFn::Sin => {
            if n_usize == 0 {
                return None;
            }
            let one_minus = build_one_minus_t_sq(ctx, t);
            let sqrt_part = build_sqrt(ctx, one_minus);
            let u = chebyshev_u_nm1(ctx, t, n_usize);
            (
                ctx.add(Expr::Mul(sqrt_part, u)),
                ctx.num(1),
                format!("sin({n}·arccos(t)) = √(1-t²)·U_{{{n}-1}}(t)"),
            )
        }
        BuiltinFn::Tan => {
            if n_usize == 0 {
                return None;
            }
            let tn = chebyshev_t(ctx, t, n_usize);
            let one_minus = build_one_minus_t_sq(ctx, t);
            let sqrt_part = build_sqrt(ctx, one_minus);
            let u = chebyshev_u_nm1(ctx, t, n_usize);
            let numerator = ctx.add(Expr::Mul(sqrt_part, u));
            (
                ctx.add(Expr::Div(numerator, tn)),
                tn,
                format!("tan({n}·arccos(t)) = √(1-t²)·U_{{{n}-1}}(t)/T_{n}(t)"),
            )
        }
        _ => return None,
    };

    let rewritten = if !is_positive {
        match trig {
            BuiltinFn::Cos => result,
            _ => ctx.add(Expr::Neg(result)),
        }
    } else {
        result
    };

    if count_nodes_dedup(ctx, rewritten) > max_output_nodes {
        return None;
    }

    Some(NAngleInverseTrigPlan {
        rewritten,
        assume_nonzero_expr,
        desc,
    })
}

/// Plan `sin/cos/tan(n*asin(t))` via sin/cos recurrence.
pub fn try_plan_n_angle_asin_expr(
    ctx: &mut Context,
    expr: ExprId,
    max_n: i64,
    max_inner_nodes: usize,
    max_output_nodes: usize,
) -> Option<NAngleInverseTrigPlan> {
    let (trig, arg0) = match_outer_trig_call(ctx, expr)?;
    let (is_positive, n, inner) = match_n_multiple(arg0, ctx, max_n)?;

    let t = match ctx.get(inner) {
        Expr::Function(inv_id, inv_args) if inv_args.len() == 1 => match ctx.builtin_of(*inv_id) {
            Some(BuiltinFn::Asin | BuiltinFn::Arcsin) => inv_args[0],
            _ => return None,
        },
        _ => return None,
    };

    if count_nodes_dedup(ctx, t) > max_inner_nodes {
        return None;
    }

    let one_minus = build_one_minus_t_sq(ctx, t);
    let cos_theta = build_sqrt(ctx, one_minus);
    let (s_n, c_n) = arcsin_recurrence(ctx, t, cos_theta, n as usize);

    let (result, assume_nonzero_expr, desc) = match trig {
        BuiltinFn::Sin => (
            s_n,
            ctx.num(1),
            format!("sin({n}·arcsin(t)) via recurrence"),
        ),
        BuiltinFn::Cos => (
            c_n,
            ctx.num(1),
            format!("cos({n}·arcsin(t)) via recurrence"),
        ),
        BuiltinFn::Tan => (
            ctx.add(Expr::Div(s_n, c_n)),
            c_n,
            format!("tan({n}·arcsin(t)) = Sₙ/Cₙ"),
        ),
        _ => return None,
    };

    let rewritten = if !is_positive {
        match trig {
            BuiltinFn::Cos => result,
            _ => ctx.add(Expr::Neg(result)),
        }
    } else {
        result
    };

    if count_nodes_dedup(ctx, rewritten) > max_output_nodes {
        return None;
    }

    Some(NAngleInverseTrigPlan {
        rewritten,
        assume_nonzero_expr,
        desc,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn count_nodes_dedup_counts_shared_subexpression_once() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let sub = ctx.add(Expr::Add(x, one));
        let expr = ctx.add(Expr::Mul(sub, sub));

        assert_eq!(count_nodes_dedup(&ctx, expr), 4);
    }

    #[test]
    fn one_plus_and_minus_square_build_expected_shapes() {
        let mut ctx = Context::new();
        let t = ctx.var("t");
        let plus = build_one_plus_t_sq(&mut ctx, t);
        let minus = build_one_minus_t_sq(&mut ctx, t);

        match ctx.get(plus) {
            Expr::Add(l, r) => {
                assert!(matches!(ctx.get(*l), Expr::Number(_)));
                assert!(matches!(ctx.get(*r), Expr::Pow(_, _)));
            }
            _ => panic!("expected Add"),
        }

        match ctx.get(minus) {
            Expr::Sub(l, r) => {
                assert!(matches!(ctx.get(*l), Expr::Number(_)));
                assert!(matches!(ctx.get(*r), Expr::Pow(_, _)));
            }
            _ => panic!("expected Sub"),
        }
    }

    #[test]
    fn recurrence_base_cases_hold() {
        let mut ctx = Context::new();
        let t = ctx.var("t");
        let one_minus = build_one_minus_t_sq(&mut ctx, t);
        let cos_theta = build_sqrt(&mut ctx, one_minus);

        let (a0, b0) = weierstrass_recurrence(&mut ctx, t, 0);
        assert!(matches!(ctx.get(a0), Expr::Number(_)));
        assert!(matches!(ctx.get(b0), Expr::Number(_)));

        let t0 = chebyshev_t(&mut ctx, t, 0);
        let u0 = chebyshev_u_nm1(&mut ctx, t, 1);
        assert!(matches!(ctx.get(t0), Expr::Number(_)));
        assert!(matches!(ctx.get(u0), Expr::Number(_)));

        let (s0, c0) = arcsin_recurrence(&mut ctx, t, cos_theta, 0);
        assert!(matches!(ctx.get(s0), Expr::Number(_)));
        assert!(matches!(ctx.get(c0), Expr::Number(_)));
    }
}
