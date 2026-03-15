//! Heuristics for fraction-addition simplification decisions.

use crate::expr_predicates::is_simple_number_abs_leq;
use crate::expr_relations::is_negation;
use crate::polynomial::Polynomial;
use cas_ast::{collect_variables, Context, Expr, ExprId};
use num_traits::{One, Zero};

const MAX_NONSIMPLIFYING_FRACTION_ADD_GROWTH_COMPLEXITY: usize = 64;

#[derive(Debug, Clone, Copy)]
pub struct FractionAddAcceptanceInput {
    pub n1: ExprId,
    pub n2: ExprId,
    pub old_complexity: usize,
    pub new_complexity: usize,
    pub opposite_denom: bool,
    pub same_denom: bool,
    pub does_simplify: bool,
    pub is_proper: bool,
    pub same_sign: bool,
}

/// Heuristic assessment used by fraction-addition rules.
///
/// Returns `(does_simplify, is_proper)` where:
/// - `does_simplify`: likely cancellation/algebraic gain exists.
/// - `is_proper`: resulting fraction appears proper (`deg(num) < deg(den)`).
pub fn assess_fraction_add_simplification(
    ctx: &Context,
    numerator: ExprId,
    denominator: ExprId,
) -> (bool, bool) {
    if let Expr::Number(n) = ctx.get(numerator) {
        if n.is_zero() {
            return (true, true);
        }
    }

    if is_negation(ctx, numerator, denominator) {
        return (true, false);
    }

    let vars = collect_variables(ctx, numerator);
    if vars.len() == 1 {
        if let Some(var) = vars.iter().next() {
            if let Ok(p_num) = Polynomial::from_expr(ctx, numerator, var) {
                if let Ok(p_den) = Polynomial::from_expr(ctx, denominator, var) {
                    if !p_den.is_zero() {
                        let gcd = p_num.gcd(&p_den);
                        if gcd.degree() > 0 || !gcd.leading_coeff().is_one() {
                            let is_proper = p_num.degree() < p_den.degree();
                            return (true, is_proper);
                        }
                    }
                }
            }
        }
    }

    (false, false)
}

/// Final acceptance policy for fraction-addition rewrites after candidate build.
pub fn should_accept_fraction_add_rewrite(
    ctx: &Context,
    input: FractionAddAcceptanceInput,
) -> bool {
    let growth_ok = input.new_complexity <= input.old_complexity * 3 / 2 + 2;

    let both_simple_numerators =
        is_simple_number_abs_leq(ctx, input.n1, 2) && is_simple_number_abs_leq(ctx, input.n2, 2);
    let bounded_growth = input.old_complexity <= MAX_NONSIMPLIFYING_FRACTION_ADD_GROWTH_COMPLEXITY;
    let allow_growth = growth_ok
        && bounded_growth
        && !input.opposite_denom
        && !input.same_denom
        && (input.same_sign || both_simple_numerators);

    input.opposite_denom
        || input.same_denom
        || input.new_complexity <= input.old_complexity
        || (input.does_simplify
            && input.is_proper
            && input.new_complexity < (input.old_complexity * 2))
        || allow_growth
}

#[cfg(test)]
mod tests {
    use super::{
        assess_fraction_add_simplification, should_accept_fraction_add_rewrite,
        FractionAddAcceptanceInput,
    };
    use cas_ast::Context;
    use cas_parser::parse;

    #[test]
    fn zero_numerator_is_simplifying_and_proper() {
        let mut ctx = Context::new();
        let num = parse("0", &mut ctx).expect("parse");
        let den = parse("x+1", &mut ctx).expect("parse");
        assert_eq!(
            assess_fraction_add_simplification(&ctx, num, den),
            (true, true)
        );
    }

    #[test]
    fn negation_relation_is_simplifying() {
        let mut ctx = Context::new();
        let num = parse("-x", &mut ctx).expect("parse");
        let den = parse("x", &mut ctx).expect("parse");
        assert_eq!(
            assess_fraction_add_simplification(&ctx, num, den),
            (true, false)
        );
    }

    #[test]
    fn nontrivial_gcd_detected() {
        let mut ctx = Context::new();
        let num = parse("x^2-1", &mut ctx).expect("parse");
        let den = parse("x-1", &mut ctx).expect("parse");
        let (does, _proper) = assess_fraction_add_simplification(&ctx, num, den);
        assert!(does);
    }

    #[test]
    fn proper_fraction_case_detected() {
        let mut ctx = Context::new();
        let num = parse("x", &mut ctx).expect("parse");
        let den = parse("x^2-x", &mut ctx).expect("parse");
        let (does, proper) = assess_fraction_add_simplification(&ctx, num, den);
        assert!(does);
        assert!(proper);
    }

    #[test]
    fn no_simplification_case() {
        let mut ctx = Context::new();
        let num = parse("x+1", &mut ctx).expect("parse");
        let den = parse("x+2", &mut ctx).expect("parse");
        assert_eq!(
            assess_fraction_add_simplification(&ctx, num, den),
            (false, false)
        );
    }

    #[test]
    fn acceptance_allows_related_denominators() {
        let mut ctx = Context::new();
        let n1 = parse("1", &mut ctx).expect("parse");
        let n2 = parse("1", &mut ctx).expect("parse");
        assert!(should_accept_fraction_add_rewrite(
            &ctx,
            FractionAddAcceptanceInput {
                n1,
                n2,
                old_complexity: 10,
                new_complexity: 30,
                opposite_denom: false,
                same_denom: true,
                does_simplify: false,
                is_proper: false,
                same_sign: true,
            }
        ));
    }

    #[test]
    fn acceptance_allows_when_simplifies_and_proper() {
        let mut ctx = Context::new();
        let n1 = parse("x", &mut ctx).expect("parse");
        let n2 = parse("1", &mut ctx).expect("parse");
        assert!(should_accept_fraction_add_rewrite(
            &ctx,
            FractionAddAcceptanceInput {
                n1,
                n2,
                old_complexity: 10,
                new_complexity: 19,
                opposite_denom: false,
                same_denom: false,
                does_simplify: true,
                is_proper: true,
                same_sign: false,
            }
        ));
    }

    #[test]
    fn acceptance_rejects_large_growth_without_gains() {
        let mut ctx = Context::new();
        let n1 = parse("x+3", &mut ctx).expect("parse");
        let n2 = parse("x+5", &mut ctx).expect("parse");
        assert!(!should_accept_fraction_add_rewrite(
            &ctx,
            FractionAddAcceptanceInput {
                n1,
                n2,
                old_complexity: 10,
                new_complexity: 25,
                opposite_denom: false,
                same_denom: false,
                does_simplify: false,
                is_proper: false,
                same_sign: false,
            }
        ));
    }

    #[test]
    fn acceptance_rejects_large_nonsimplifying_growth_even_when_same_sign() {
        let mut ctx = Context::new();
        let n1 = parse("x+3", &mut ctx).expect("parse");
        let n2 = parse("x+5", &mut ctx).expect("parse");
        assert!(!should_accept_fraction_add_rewrite(
            &ctx,
            FractionAddAcceptanceInput {
                n1,
                n2,
                old_complexity: 79,
                new_complexity: 103,
                opposite_denom: false,
                same_denom: false,
                does_simplify: false,
                is_proper: false,
                same_sign: true,
            }
        ));
    }

    #[test]
    fn acceptance_still_allows_small_same_sign_growth() {
        let mut ctx = Context::new();
        let n1 = parse("1", &mut ctx).expect("parse");
        let n2 = parse("1", &mut ctx).expect("parse");
        assert!(should_accept_fraction_add_rewrite(
            &ctx,
            FractionAddAcceptanceInput {
                n1,
                n2,
                old_complexity: 11,
                new_complexity: 15,
                opposite_denom: false,
                same_denom: false,
                does_simplify: false,
                is_proper: false,
                same_sign: true,
            }
        ));
    }
}
