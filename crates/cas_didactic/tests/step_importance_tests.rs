//! Test to verify step importance is correctly propagated from rules
use cas_solver::runtime::ImportanceLevel;
use cas_solver::runtime::Simplifier;

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

/// Regression test: a visible cancellation step should appear in Sub context.
///
/// Historically, `(x^2+y^2+2xy) - (x+y)^2` needed an explicit
/// `Auto Expand Power Sum` step before cancellation. The current engine may
/// instead close the identity directly via `Polynomial Identity`.
///
/// The contract we care about is that the user sees at least one didactically
/// relevant step for the cancellation in Sub context.
///
/// Prior bug: `(x^2+y^2+2xy) - (x+y)^2` was showing expanded form in a later
/// step "Before" but the actual visible cancellation trigger was missing because:
/// 1. engine.rs didn't track Sub in ancestor_stack (now fixed)
/// 2. AutoExpandPowSumRule couldn't see Sub parent, so in_auto_expand_context() was false
#[test]
fn test_auto_expand_step_visible_in_sub_context() {
    let mut simplifier = Simplifier::with_default_rules();
    simplifier.set_collect_steps(true);

    // This expression should auto-expand (x+y)² because it's in a Sub context
    // marked for auto-expand by auto_expand_scan::try_mark_sub_cancellation
    let expr = cas_parser::parse("(x^2+y^2+2*x*y) - (x+y)^2", &mut simplifier.context).unwrap();

    let (_result, steps) = simplifier.simplify(expr);

    // Find Auto Expand Power Sum steps
    let expand_steps: Vec<_> = steps
        .iter()
        .filter(|s| s.rule_name == "Auto Expand Power Sum")
        .collect();
    let poly_identity_steps: Vec<_> = steps
        .iter()
        .filter(|s| s.rule_name == "Polynomial Identity")
        .collect();

    // There should be at least one visible cancellation route.
    assert!(
        !expand_steps.is_empty() || !poly_identity_steps.is_empty(),
        "Expected at least one visible cancellation step in Sub context. \
         Available steps: {:?}",
        steps.iter().map(|s| &s.rule_name).collect::<Vec<_>>()
    );

    // Find Combine Like Terms steps
    let combine_idx = steps
        .iter()
        .position(|s| s.rule_name == "Combine Like Terms");

    // Find Auto Expand step position
    let expand_idx = steps
        .iter()
        .position(|s| s.rule_name == "Auto Expand Power Sum");

    // Auto Expand should appear BEFORE Combine Like Terms when that path is taken.
    if !expand_steps.is_empty() {
        if let (Some(expand_pos), Some(combine_pos)) = (expand_idx, combine_idx) {
            assert!(
                expand_pos < combine_pos,
                "Auto Expand Power Sum (at {}) should appear before Combine Like Terms (at {})",
                expand_pos,
                combine_pos
            );
        }
    }

    // The expand step should have Medium importance when that route is used.
    for step in &expand_steps {
        assert_eq!(
            step.importance,
            ImportanceLevel::Medium,
            "Auto Expand Power Sum should have Medium importance, got {:?}",
            step.importance
        );
    }
}
