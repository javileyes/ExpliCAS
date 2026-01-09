//! Test to verify step importance is correctly propagated from rules
use cas_engine::step::ImportanceLevel;
use cas_engine::Simplifier;

#[test]
fn test_identity_property_steps_have_low_importance() {
    let mut simplifier = Simplifier::with_default_rules();
    simplifier.set_collect_steps(true);

    // Expression with multiplication by 1: will trigger MulOneRule
    let expr = cas_parser::parse("x * 1", &mut simplifier.context).unwrap();

    let (_result, steps) = simplifier.simplify(expr);

    // Find Identity Property steps
    let identity_steps: Vec<_> = steps
        .iter()
        .filter(|s| s.rule_name == "Identity Property of Multiplication")
        .collect();

    assert!(
        !identity_steps.is_empty(),
        "Should have at least one Identity Property step"
    );

    for step in &identity_steps {
        eprintln!(
            "Step: {}, importance field: {:?}, get_importance(): {:?}",
            step.description,
            step.importance,
            step.get_importance()
        );

        // Identity Property rules now have Low importance (hidden in normal mode)
        assert_eq!(
            step.importance,
            ImportanceLevel::Low,
            "Step '{}' should have importance field set to Low, got {:?}",
            step.description,
            step.importance
        );

        assert_eq!(
            step.get_importance(),
            ImportanceLevel::Low,
            "Step '{}' get_importance() should return Low, got {:?}",
            step.description,
            step.get_importance()
        );
    }
}
