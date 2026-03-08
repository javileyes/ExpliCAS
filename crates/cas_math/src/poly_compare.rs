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
}
