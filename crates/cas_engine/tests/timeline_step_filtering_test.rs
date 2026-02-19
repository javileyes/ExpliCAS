//! Test to verify timeline step filtering for atan expression
use cas_engine::to_display_steps;
use cas_engine::ImportanceLevel;
use cas_engine::Simplifier;

#[test]
fn test_atan_expression_step_importance() {
    let mut simplifier = Simplifier::with_default_rules();
    simplifier.set_collect_steps(true);

    // Expression: atan(3) + (atan(1/3) - pi/2)
    let expr = cas_parser::parse("atan(3) + (atan(1/3) - pi/2)", &mut simplifier.context).unwrap();

    let (_result, raw_steps) = simplifier.simplify(expr);

    eprintln!("\n=== RAW STEPS ({}) ===", raw_steps.len());
    for (i, step) in raw_steps.iter().enumerate() {
        eprintln!(
            "  {}: {} [{}] importance={:?} get_importance={:?}",
            i + 1,
            step.description,
            step.rule_name,
            step.importance,
            step.get_importance()
        );
    }

    // Apply pipeline
    let display_steps = to_display_steps(raw_steps);

    eprintln!("\n=== DISPLAY STEPS ({}) ===", display_steps.len());
    for (i, step) in display_steps.iter().enumerate() {
        eprintln!(
            "  {}: {} [{}] importance={:?} get_importance={:?}",
            i + 1,
            step.description,
            step.rule_name,
            step.importance,
            step.get_importance()
        );
    }

    // Count Identity Property steps
    let identity_steps: Vec<_> = display_steps
        .iter()
        .filter(|s| s.rule_name == "Identity Property of Multiplication")
        .collect();

    eprintln!(
        "\n=== IDENTITY PROPERTY STEPS ({}) ===",
        identity_steps.len()
    );
    for step in &identity_steps {
        eprintln!(
            "  {} importance={:?} get_importance={:?}",
            step.description,
            step.importance,
            step.get_importance()
        );
    }

    // All Identity Property steps should have Low importance (hidden in normal mode)
    for step in &identity_steps {
        assert_eq!(
            step.get_importance(),
            ImportanceLevel::Low,
            "Step '{}' should have Low importance",
            step.description
        );
    }

    // Check that these steps would NOT be shown in Normal verbosity
    let shown_in_normal: Vec<_> = display_steps
        .iter()
        .filter(|s| s.get_importance() >= ImportanceLevel::Medium)
        .collect();

    eprintln!(
        "\n=== STEPS SHOWN IN NORMAL VERBOSITY ({}) ===",
        shown_in_normal.len()
    );

    // Verify Identity Property steps are NOT in the shown list
    let identity_shown = shown_in_normal
        .iter()
        .filter(|s| s.rule_name == "Identity Property of Multiplication")
        .count();

    assert_eq!(
        identity_shown, 0,
        "Identity Property steps should NOT be shown in Normal verbosity, but {} were shown",
        identity_shown
    );

    // Normal mode should show 2 steps for this expression (improved didactics)
    // Previously 5 steps before atan identity fix - AddFractionsRule fired first
    assert_eq!(
        shown_in_normal.len(),
        2,
        "Normal verbosity should show 2 steps (improved didactics), got {}",
        shown_in_normal.len()
    );
}
