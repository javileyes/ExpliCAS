use cas_ast::{Context, Expr, ExprId};
use std::hash::{Hash, Hasher};

/// Compute a deterministic structural hash of an AST subtree.
pub fn expr_fingerprint(ctx: &Context, id: ExprId, h: &mut impl Hasher) {
    let node = ctx.get(id);
    std::mem::discriminant(node).hash(h);
    match node {
        Expr::Number(n) => n.hash(h),
        Expr::Variable(s) => ctx.sym_name(*s).hash(h),
        Expr::Constant(c) => std::mem::discriminant(c).hash(h),
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) => {
            expr_fingerprint(ctx, *l, h);
            expr_fingerprint(ctx, *r, h);
        }
        Expr::Pow(b, e) => {
            expr_fingerprint(ctx, *b, h);
            expr_fingerprint(ctx, *e, h);
        }
        Expr::Neg(e) | Expr::Hold(e) => expr_fingerprint(ctx, *e, h),
        Expr::Function(name, args) => {
            ctx.sym_name(*name).hash(h);
            for a in args {
                expr_fingerprint(ctx, *a, h);
            }
        }
        Expr::Matrix { rows, cols, data } => {
            rows.hash(h);
            cols.hash(h);
            for d in data {
                expr_fingerprint(ctx, *d, h);
            }
        }
        Expr::SessionRef(s) => s.hash(h),
    }
}

/// Compute a fingerprint for a `(var, lhs, rhs)` equation tuple.
pub fn equation_fingerprint(ctx: &Context, lhs: ExprId, rhs: ExprId, var: &str) -> u64 {
    let mut hasher = std::hash::DefaultHasher::new();
    var.hash(&mut hasher);
    expr_fingerprint(ctx, lhs, &mut hasher);
    0xFFu8.hash(&mut hasher);
    expr_fingerprint(ctx, rhs, &mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equation_fingerprint_is_deterministic() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let lhs = ctx.add(Expr::Add(x, one));
        let rhs = ctx.num(0);

        let f1 = equation_fingerprint(&ctx, lhs, rhs, "x");
        let f2 = equation_fingerprint(&ctx, lhs, rhs, "x");
        assert_eq!(f1, f2);
    }

    #[test]
    fn equation_fingerprint_changes_with_var() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let lhs = ctx.add(Expr::Add(x, one));
        let rhs = ctx.num(0);

        let fx = equation_fingerprint(&ctx, lhs, rhs, "x");
        let fy = equation_fingerprint(&ctx, lhs, rhs, "y");
        assert_ne!(fx, fy);
    }
}
