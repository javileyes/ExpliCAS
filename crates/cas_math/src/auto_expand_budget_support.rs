//! Structural helpers for auto-expand budget checks.

use cas_ast::{Context, Expr, ExprId};

/// Count additive terms in an Add-tree base used by power auto-expansion.
///
/// This intentionally treats only `Add` as additive branching to preserve
/// historical behavior in `AutoExpandPowSumRule`.
pub fn count_add_terms_for_pow_base(ctx: &Context, expr: ExprId) -> u32 {
    match ctx.get(expr) {
        Expr::Add(l, r) => {
            count_add_terms_for_pow_base(ctx, *l) + count_add_terms_for_pow_base(ctx, *r)
        }
        _ => 1,
    }
}

/// Count distinct variables in an expression.
pub fn count_distinct_variables_in_expr(ctx: &Context, expr: ExprId) -> u32 {
    cas_ast::collect_variables(ctx, expr).len() as u32
}

/// Estimate multinomial output terms `C(n+k-1, k-1)` where:
/// - `k` = number of additive terms in base
/// - `n` = exponent
///
/// Returns `None` on overflow.
pub fn estimate_multinomial_terms_for_pow(k: u32, n: u32) -> Option<u32> {
    if k <= 1 {
        return Some(1);
    }

    let total = n.checked_add(k)?.checked_sub(1)? as u64;
    let choose = (k - 1).min(n) as u64;

    let mut result: u64 = 1;
    for i in 0..choose {
        result = result.checked_mul(total - i)?;
        result /= i + 1;
        if result > u32::MAX as u64 {
            return None;
        }
    }

    Some(result as u32)
}

#[cfg(test)]
mod tests {
    use super::{
        count_add_terms_for_pow_base, count_distinct_variables_in_expr,
        estimate_multinomial_terms_for_pow,
    };
    use cas_ast::Context;
    use cas_parser::parse;

    #[test]
    fn count_add_terms_for_pow_base_counts_add_only() {
        let mut ctx = Context::new();
        let expr = parse("a + b + c", &mut ctx).expect("parse");
        assert_eq!(count_add_terms_for_pow_base(&ctx, expr), 3);
    }

    #[test]
    fn count_distinct_variables_in_expr_counts_unique_symbols() {
        let mut ctx = Context::new();
        let expr = parse("x + x + y", &mut ctx).expect("parse");
        assert_eq!(count_distinct_variables_in_expr(&ctx, expr), 2);
    }

    #[test]
    fn estimate_multinomial_terms_for_pow_matches_small_values() {
        // (a+b)^3 -> 4 terms
        assert_eq!(estimate_multinomial_terms_for_pow(2, 3), Some(4));
        // (a+b+c)^4 -> C(6,2)=15 terms
        assert_eq!(estimate_multinomial_terms_for_pow(3, 4), Some(15));
    }
}
