use crate::best_so_far::{score_expr, BestSoFar, BestSoFarBudget, Score};
use cas_ast::Context;
use cas_parser::parse;

#[test]
fn test_score_ordering() {
    // Score with sqrt in den is worse than without
    let s1 = Score {
        sqrt_in_den: 1,
        nested_div: 0,
        add_in_den: false,
        nodes: 10,
    };
    let s2 = Score {
        sqrt_in_den: 0,
        nested_div: 0,
        add_in_den: false,
        nodes: 15,
    };
    assert!(
        s2 < s1,
        "Fewer sqrt_in_den should be better even with more nodes"
    );

    // Score with nested div is worse
    let s3 = Score {
        sqrt_in_den: 0,
        nested_div: 2,
        add_in_den: false,
        nodes: 10,
    };
    let s4 = Score {
        sqrt_in_den: 0,
        nested_div: 0,
        add_in_den: false,
        nodes: 12,
    };
    assert!(
        s4 < s3,
        "Fewer nested_div should be better even with more nodes"
    );
}

#[test]
fn test_budget_enforcement_via_consider() {
    let mut ctx = Context::new();
    let baseline = parse("1/sqrt(x)", &mut ctx).expect("baseline parse should succeed");
    let candidate = parse("sqrt(x)/x + 0", &mut ctx).expect("candidate parse should succeed");

    let base_score = score_expr(&ctx, baseline);
    let cand_score = score_expr(&ctx, candidate);
    assert!(
        cand_score < base_score,
        "candidate should be preferred when budget allows"
    );
    assert!(
        cand_score.nodes > base_score.nodes,
        "candidate must be larger to exercise budget guard"
    );

    // Tight budget: candidate is rejected by node budget.
    let mut tracker_tight =
        BestSoFar::new(baseline, &[], &ctx, BestSoFarBudget { max_extra_nodes: 0 });
    tracker_tight.consider(candidate, &[], &ctx);
    let (best_tight, _) = tracker_tight.into_parts();
    assert_eq!(best_tight, baseline);

    // Relaxed budget: candidate can be selected.
    let mut tracker_relaxed = BestSoFar::new(
        baseline,
        &[],
        &ctx,
        BestSoFarBudget {
            max_extra_nodes: 16,
        },
    );
    tracker_relaxed.consider(candidate, &[], &ctx);
    let (best_relaxed, _) = tracker_relaxed.into_parts();
    assert_eq!(best_relaxed, candidate);
}
