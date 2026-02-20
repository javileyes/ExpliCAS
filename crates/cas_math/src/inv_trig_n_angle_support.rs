use cas_ast::{Context, Expr, ExprId};
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
