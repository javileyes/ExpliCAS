//! Guard helpers for fold-add fraction rewrites.

use crate::expr_predicates::{contains_div_term, contains_function_or_root};
use crate::fold_add_fraction_support::should_block_fold_add_unit_constant_term;
use cas_ast::{Context, Expr, ExprId};

/// Structural/expression guard for `k + p/q -> (k*q + p)/q`.
///
/// Blocks when:
/// - `term` already contains division terms (delegate to fraction-add rules)
/// - `term` contains functions or roots
/// - denominator contains functions or roots
/// - denominator is purely numeric
pub fn should_block_fold_add_term_or_den(ctx: &Context, term: ExprId, denominator: ExprId) -> bool {
    contains_div_term(ctx, term)
        || contains_function_or_root(ctx, term)
        || contains_function_or_root(ctx, denominator)
        || matches!(ctx.get(denominator), Expr::Number(_))
}

/// Combined policy guard for `k + p/q -> (k*q + p)/q`.
pub fn should_block_fold_add_rewrite(
    ctx: &Context,
    term: ExprId,
    denominator: ExprId,
    numerator_is_constant: bool,
    inside_trig: bool,
    inside_fraction: bool,
) -> bool {
    inside_trig
        || inside_fraction
        || should_block_fold_add_unit_constant_term(ctx, term, numerator_is_constant)
        || should_block_fold_add_term_or_den(ctx, term, denominator)
}

#[cfg(test)]
mod tests {
    use super::{should_block_fold_add_rewrite, should_block_fold_add_term_or_den};
    use cas_ast::Context;
    use cas_parser::parse;

    #[test]
    fn blocks_term_with_division() {
        let mut ctx = Context::new();
        let term = parse("1/x", &mut ctx).expect("parse");
        let den = parse("y", &mut ctx).expect("parse");
        assert!(should_block_fold_add_term_or_den(&ctx, term, den));
    }

    #[test]
    fn blocks_numeric_denominator() {
        let mut ctx = Context::new();
        let term = parse("x", &mut ctx).expect("parse");
        let den = parse("2", &mut ctx).expect("parse");
        assert!(should_block_fold_add_term_or_den(&ctx, term, den));
    }

    #[test]
    fn allows_simple_symbolic_case() {
        let mut ctx = Context::new();
        let term = parse("x", &mut ctx).expect("parse");
        let den = parse("y+1", &mut ctx).expect("parse");
        assert!(!should_block_fold_add_term_or_den(&ctx, term, den));
    }

    #[test]
    fn combined_guard_blocks_inside_contexts() {
        let mut ctx = Context::new();
        let term = parse("x", &mut ctx).expect("parse");
        let den = parse("y+1", &mut ctx).expect("parse");
        assert!(should_block_fold_add_rewrite(
            &ctx, term, den, false, true, false
        ));
        assert!(should_block_fold_add_rewrite(
            &ctx, term, den, false, false, true
        ));
    }

    #[test]
    fn combined_guard_allows_simple_case() {
        let mut ctx = Context::new();
        let term = parse("x", &mut ctx).expect("parse");
        let den = parse("y+1", &mut ctx).expect("parse");
        assert!(!should_block_fold_add_rewrite(
            &ctx, term, den, false, false, false
        ));
    }
}
