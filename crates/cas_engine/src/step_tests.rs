use crate::assumptions::{AssumptionEvent, AssumptionKey, AssumptionKind};
use crate::step::{ImportanceLevel, Step};

#[test]
fn test_importance_classification() {
    let mut ctx = cas_ast::Context::new();
    let x = ctx.var("x");
    let y = ctx.var("y");

    // Test 1: Default importance is Low (from Step::new)
    let step = Step::new("Some rule", "Some Rule", x, y, vec![], Some(&ctx));
    assert_eq!(step.get_importance(), ImportanceLevel::Low);

    // Test 2: No-op steps (same before/after) are always Trivial
    let step = Step::new("No change", "Any Rule", x, x, vec![], Some(&ctx));
    assert_eq!(step.get_importance(), ImportanceLevel::Trivial);

    // Test 3: Steps with assumption_events are bumped to Medium
    let mut step = Step::new("Rule with assumption", "Rule", x, y, vec![], Some(&ctx));
    step.meta_mut().assumption_events.push(AssumptionEvent {
        key: AssumptionKey::NonZero {
            expr_fingerprint: 12345,
        },
        expr_display: "x".to_string(),
        message: "Assuming x ≠ 0".to_string(),
        kind: AssumptionKind::RequiresIntroduced,
        expr_id: None, // Test doesn't need real ExprId
    });
    assert_eq!(step.get_importance(), ImportanceLevel::Medium);

    // Test 4: Declaratively set importance is respected
    let mut step = Step::new("Medium rule", "Important Rule", x, y, vec![], Some(&ctx));
    step.importance = ImportanceLevel::Medium;
    assert_eq!(step.get_importance(), ImportanceLevel::Medium);

    let mut step = Step::new("High rule", "Major Transform", x, y, vec![], Some(&ctx));
    step.importance = ImportanceLevel::High;
    assert_eq!(step.get_importance(), ImportanceLevel::High);
}
