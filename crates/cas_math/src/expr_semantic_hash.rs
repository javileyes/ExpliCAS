//! Semantic hashing helpers for expression trees.
//!
//! The hash is structural but commutativity-aware for `Add`/`Mul`, and uses
//! a depth guard to avoid runaway recursion on malformed trees.

use cas_ast::{Context, Expr, ExprId};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Compute a semantic hash of an expression.
///
/// - `Add` and `Mul` children are sorted by child hash (commutativity-aware).
/// - Uses depth-limited recursion (`MAX_HASH_DEPTH`) as a safety guard.
pub fn semantic_hash(ctx: &Context, expr: ExprId) -> u64 {
    const MAX_HASH_DEPTH: usize = 200;

    fn hash_expr_depth(ctx: &Context, expr: ExprId, depth: usize, hasher: &mut DefaultHasher) {
        if depth == 0 {
            format!("{expr:?}").hash(hasher);
            return;
        }

        match ctx.get(expr) {
            Expr::Number(n) => {
                0u8.hash(hasher);
                n.to_string().hash(hasher);
            }
            Expr::Constant(c) => {
                1u8.hash(hasher);
                format!("{c:?}").hash(hasher);
            }
            Expr::Variable(v) => {
                2u8.hash(hasher);
                v.hash(hasher);
            }
            Expr::Add(l, r) => {
                3u8.hash(hasher);
                let mut h1 = DefaultHasher::new();
                let mut h2 = DefaultHasher::new();
                hash_expr_depth(ctx, *l, depth - 1, &mut h1);
                hash_expr_depth(ctx, *r, depth - 1, &mut h2);
                let hash_l = h1.finish();
                let hash_r = h2.finish();
                if hash_l <= hash_r {
                    hash_l.hash(hasher);
                    hash_r.hash(hasher);
                } else {
                    hash_r.hash(hasher);
                    hash_l.hash(hasher);
                }
            }
            Expr::Sub(l, r) => {
                4u8.hash(hasher);
                hash_expr_depth(ctx, *l, depth - 1, hasher);
                hash_expr_depth(ctx, *r, depth - 1, hasher);
            }
            Expr::Mul(l, r) => {
                5u8.hash(hasher);
                let mut h1 = DefaultHasher::new();
                let mut h2 = DefaultHasher::new();
                hash_expr_depth(ctx, *l, depth - 1, &mut h1);
                hash_expr_depth(ctx, *r, depth - 1, &mut h2);
                let hash_l = h1.finish();
                let hash_r = h2.finish();
                if hash_l <= hash_r {
                    hash_l.hash(hasher);
                    hash_r.hash(hasher);
                } else {
                    hash_r.hash(hasher);
                    hash_l.hash(hasher);
                }
            }
            Expr::Div(l, r) => {
                6u8.hash(hasher);
                hash_expr_depth(ctx, *l, depth - 1, hasher);
                hash_expr_depth(ctx, *r, depth - 1, hasher);
            }
            Expr::Pow(b, e) => {
                7u8.hash(hasher);
                hash_expr_depth(ctx, *b, depth - 1, hasher);
                hash_expr_depth(ctx, *e, depth - 1, hasher);
            }
            Expr::Neg(e) => {
                8u8.hash(hasher);
                hash_expr_depth(ctx, *e, depth - 1, hasher);
            }
            Expr::Function(name, args) => {
                9u8.hash(hasher);
                name.hash(hasher);
                args.len().hash(hasher);
                for arg in args {
                    hash_expr_depth(ctx, *arg, depth - 1, hasher);
                }
            }
            Expr::Matrix { rows, cols, data } => {
                10u8.hash(hasher);
                rows.hash(hasher);
                cols.hash(hasher);
                data.len().hash(hasher);
                for elem in data {
                    hash_expr_depth(ctx, *elem, depth - 1, hasher);
                }
            }
            Expr::SessionRef(id) => {
                11u8.hash(hasher);
                id.hash(hasher);
            }
            Expr::Hold(e) => {
                12u8.hash(hasher);
                hash_expr_depth(ctx, *e, depth - 1, hasher);
            }
        }
    }

    let mut hasher = DefaultHasher::new();
    hash_expr_depth(ctx, expr, MAX_HASH_DEPTH, &mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::semantic_hash;
    use cas_parser::parse;

    #[test]
    fn add_is_commutative_for_hashing() {
        let mut ctx = cas_ast::Context::new();
        let a = parse("x + y", &mut ctx).expect("parse a");
        let b = parse("y + x", &mut ctx).expect("parse b");
        assert_eq!(semantic_hash(&ctx, a), semantic_hash(&ctx, b));
    }

    #[test]
    fn mul_is_commutative_for_hashing() {
        let mut ctx = cas_ast::Context::new();
        let a = parse("2 * x", &mut ctx).expect("parse a");
        let b = parse("x * 2", &mut ctx).expect("parse b");
        assert_eq!(semantic_hash(&ctx, a), semantic_hash(&ctx, b));
    }

    #[test]
    fn non_commutative_ops_keep_order() {
        let mut ctx = cas_ast::Context::new();
        let a = parse("x - y", &mut ctx).expect("parse a");
        let b = parse("y - x", &mut ctx).expect("parse b");
        assert_ne!(semantic_hash(&ctx, a), semantic_hash(&ctx, b));
    }
}
