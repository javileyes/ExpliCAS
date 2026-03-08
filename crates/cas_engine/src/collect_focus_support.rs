//! Didactic focus helpers for collect/combine-like-terms style rewrites.

use cas_ast::{Context, ExprId};
use cas_math::collect_terms::{CancelledGroup, CombinedGroup};
use cas_math::expr_terms::build_sum;

#[derive(Debug, Clone)]
pub struct CollectDidacticFocus {
    pub before: Option<ExprId>,
    pub after: Option<ExprId>,
    pub description: String,
}

/// Build didactic focus expressions from cancellation/combination groups.
///
/// Shows all affected groups together so UI can highlight the full local change.
pub fn select_collect_didactic_focus(
    ctx: &mut Context,
    cancelled: &[CancelledGroup],
    combined: &[CombinedGroup],
) -> CollectDidacticFocus {
    let mut all_before_terms: Vec<ExprId> = Vec::new();
    let mut all_after_terms: Vec<ExprId> = Vec::new();
    let mut has_cancellation = false;
    let mut has_combination = false;

    for group in cancelled {
        all_before_terms.extend(group.original_terms.iter().copied());
        has_cancellation = true;
    }

    for group in combined {
        all_before_terms.extend(group.original_terms.iter().copied());
        all_after_terms.push(group.combined_term);
        has_combination = true;
    }

    if all_before_terms.is_empty() {
        return CollectDidacticFocus {
            before: None,
            after: None,
            description: "Combine like terms".to_string(),
        };
    }

    let before = build_sum(ctx, &all_before_terms);
    let after = if all_after_terms.is_empty() {
        ctx.num(0)
    } else {
        build_sum(ctx, &all_after_terms)
    };

    let description = if has_cancellation && has_combination {
        "Cancel and combine like terms".to_string()
    } else if has_cancellation {
        "Cancel opposite terms".to_string()
    } else {
        "Combine like terms".to_string()
    };

    CollectDidacticFocus {
        before: Some(before),
        after: Some(after),
        description,
    }
}

#[cfg(test)]
mod tests {
    use super::select_collect_didactic_focus;
    use cas_ast::{Context, Expr};
    use cas_parser::parse;
    use smallvec::smallvec;

    #[test]
    fn focus_from_combined_group_builds_non_empty_before_after() {
        let mut ctx = Context::new();
        let x = parse("x", &mut ctx).expect("parse");
        let two_x = parse("2*x", &mut ctx).expect("parse");
        let three_x = parse("3*x", &mut ctx).expect("parse");
        let group = cas_math::collect_terms::CombinedGroup {
            key: x,
            original_terms: smallvec![x, two_x],
            combined_term: three_x,
        };
        let focus = select_collect_didactic_focus(&mut ctx, &[], &[group]);
        assert!(focus.before.is_some());
        assert!(focus.after.is_some());
        assert_eq!(focus.description, "Combine like terms");
    }

    #[test]
    fn focus_with_only_cancelled_terms_points_to_zero_after() {
        let mut ctx = Context::new();
        let x = parse("x", &mut ctx).expect("parse");
        let neg_x = ctx.add(Expr::Neg(x));
        let group = cas_math::collect_terms::CancelledGroup {
            key: x,
            original_terms: smallvec![x, neg_x],
            is_constant: false,
        };
        let focus = select_collect_didactic_focus(&mut ctx, &[group], &[]);
        assert!(focus.before.is_some());
        assert_eq!(focus.description, "Cancel opposite terms");
        assert_eq!(focus.after.expect("after"), ctx.num(0));
    }
}
