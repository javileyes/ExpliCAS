//! Polynomial comparison helpers for lightweight equivalence checks.
//!
//! These utilities compare expressions by converting to canonical `MultiPoly`
//! under a tight budget, instead of relying on AST shape.

use crate::multipoly::{multipoly_from_expr, PolyBudget};
use cas_ast::{Context, ExprId};

fn compare_budget() -> PolyBudget {
    PolyBudget {
        max_terms: 100,
        max_total_degree: 10,
        max_pow_exp: 5,
    }
}

/// Relation between two polynomial expressions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignRelation {
    /// `a == b`
    Same,
    /// `a == -b`
    Negated,
}

/// Compare two expressions as polynomials (ignoring AST structure/order).
///
/// Returns `false` if conversion fails for either side.
pub fn poly_eq(ctx: &Context, a: ExprId, b: ExprId) -> bool {
    let budget = compare_budget();

    let pa = match multipoly_from_expr(ctx, a, &budget) {
        Ok(p) => p,
        Err(_) => return false,
    };
    let pb = match multipoly_from_expr(ctx, b, &budget) {
        Ok(p) => p,
        Err(_) => return false,
    };

    pa == pb
}

/// Compare two expressions to detect if they are equal or negated.
///
/// Returns:
/// - `Some(SignRelation::Same)` if `a == b`
/// - `Some(SignRelation::Negated)` if `a == -b`
/// - `None` otherwise
pub fn poly_relation(ctx: &Context, a: ExprId, b: ExprId) -> Option<SignRelation> {
    let budget = compare_budget();

    let pa = multipoly_from_expr(ctx, a, &budget).ok()?;
    let pb = multipoly_from_expr(ctx, b, &budget).ok()?;

    if pa == pb {
        return Some(SignRelation::Same);
    }

    if pa == pb.neg() {
        return Some(SignRelation::Negated);
    }

    None
}

/// True when `a == λ·b` for some rational `λ < 0`, i.e. `a` and `b` are non-zero
/// polynomials pointing in opposite directions.
///
/// Then `a > 0 ∧ b > 0` is unsatisfiable (`a` and `b` always have opposite signs),
/// which is how `solve(log(b, -k·x) = log(b, x) + …)` collapses to "No solution":
/// its recorded domain conditions `Positive(-k·x) ∧ Positive(x)` are contradictory.
/// This generalises [`poly_relation`]'s `Negated` (`λ = -1`) case to any negative
/// rational multiple (e.g. `-8·x` vs `x`). Returns `false` (no claim) when either
/// side is not polynomial-convertible or is zero.
pub fn poly_negatively_proportional(ctx: &Context, a: ExprId, b: ExprId) -> bool {
    use num_traits::Signed;

    let budget = compare_budget();
    let (Ok(pa), Ok(pb)) = (
        multipoly_from_expr(ctx, a, &budget),
        multipoly_from_expr(ctx, b, &budget),
    ) else {
        return false;
    };
    if pa.is_zero() || pb.is_zero() {
        return false;
    }
    // For proportional polynomials the leading terms (same monomial order)
    // correspond, so λ = (lead coeff a) / (lead coeff b) is the only candidate.
    let (Some(ta), Some(tb)) = (pa.leading_term_lex(), pb.leading_term_lex()) else {
        return false;
    };
    let lambda = &ta.0 / &tb.0;
    if !lambda.is_negative() {
        return false;
    }
    // Exact proof: a - λ·b == 0  (and a wrong λ from a non-proportional pair makes
    // this non-zero, so the test never yields a false positive).
    matches!(pa.sub(&pb.mul_scalar(&lambda)), Ok(diff) if diff.is_zero())
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    #[test]
    fn poly_eq_matches_commutative_forms() {
        let mut ctx = Context::new();
        let a = parse("x + y", &mut ctx).expect("parse a");
        let b = parse("y + x", &mut ctx).expect("parse b");
        assert!(poly_eq(&ctx, a, b));
    }

    #[test]
    fn poly_relation_detects_negation() {
        let mut ctx = Context::new();
        let a = parse("x - y", &mut ctx).expect("parse a");
        let b = parse("y - x", &mut ctx).expect("parse b");
        assert_eq!(poly_relation(&ctx, a, b), Some(SignRelation::Negated));
    }

    #[test]
    fn poly_negatively_proportional_detects_negative_multiples() {
        let mut ctx = Context::new();
        // -8x = -8·x  (the log(2,-8x)=log(2,x)+k domain pair)
        let a = parse("-8*x", &mut ctx).expect("a");
        let b = parse("x", &mut ctx).expect("b");
        assert!(poly_negatively_proportional(&ctx, a, b));
        // exact negation (λ = -1) is still covered
        let c = parse("1 - x", &mut ctx).expect("c");
        let d = parse("x - 1", &mut ctx).expect("d");
        assert!(poly_negatively_proportional(&ctx, c, d));
        // a multivariable negative multiple
        let e = parse("-3*x - 3*y", &mut ctx).expect("e");
        let f = parse("x + y", &mut ctx).expect("f");
        assert!(poly_negatively_proportional(&ctx, e, f));
    }

    #[test]
    fn poly_negatively_proportional_rejects_compatible_and_unrelated() {
        let mut ctx = Context::new();
        // positive multiple: x>0 and 2x>0 are compatible, NOT contradictory
        let a = parse("2*x", &mut ctx).expect("a");
        let b = parse("x", &mut ctx).expect("b");
        assert!(!poly_negatively_proportional(&ctx, a, b));
        // unrelated variables are not proportional
        let c = parse("x", &mut ctx).expect("c");
        let d = parse("y", &mut ctx).expect("d");
        assert!(!poly_negatively_proportional(&ctx, c, d));
        // not proportional at all
        let e = parse("x + 1", &mut ctx).expect("e");
        let f = parse("x", &mut ctx).expect("f");
        assert!(!poly_negatively_proportional(&ctx, e, f));
    }
}
