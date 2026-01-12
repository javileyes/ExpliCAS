//! Integration test to verify timeline step rendering
use cas_engine::eval_step_pipeline::to_display_steps;
use cas_engine::step::ImportanceLevel;
use cas_engine::timeline::{TimelineHtml, VerbosityLevel};
use cas_engine::Simplifier;

#[test]
fn test_timeline_renders_all_medium_importance_steps() {
    let mut simplifier = Simplifier::with_default_rules();
    simplifier.set_collect_steps(true);

    // Expression: atan(3) + (atan(1/3) - pi/2)
    let expr = cas_parser::parse("atan(3) + (atan(1/3) - pi/2)", &mut simplifier.context).unwrap();

    let (result, raw_steps) = simplifier.simplify(expr);

    // Apply pipeline (same as engine.eval would do)
    let display_steps = to_display_steps(raw_steps);

    // Count steps with Medium+ importance (should be shown in Normal verbosity)
    let expected_count = display_steps
        .iter()
        .filter(|s| s.get_importance() >= ImportanceLevel::Medium)
        .count();

    eprintln!(
        "Expected {} steps to be shown in Normal verbosity",
        expected_count
    );

    // Should be 2 steps: Inverse Tan Identity + Cancel opposite terms
    // (Previously 5 steps before atan identity improvement - AddFractionsRule fired first)
    assert_eq!(
        expected_count, 2,
        "Expected 2 steps in Normal verbosity (improved didactics)"
    );

    // Create timeline with Normal verbosity
    let mut timeline = TimelineHtml::new_with_result(
        &mut simplifier.context,
        display_steps.as_slice(),
        expr,
        Some(result),
        VerbosityLevel::Normal,
    );

    // Generate HTML
    let html = timeline.to_html();

    // Count step cards in HTML (look for step-content class)
    let step_card_count = html.matches("class=\"step-content\"").count();

    eprintln!("Timeline HTML has {} step cards", step_card_count);

    // Identity Property steps should NOT appear in timeline (Low importance)
    let identity_count = html.matches("Identity Property").count();
    eprintln!(
        "Timeline HTML mentions 'Identity Property' {} times",
        identity_count
    );
    assert_eq!(
        identity_count, 0,
        "Identity Property should not appear in Normal timeline"
    );

    assert_eq!(
        step_card_count, expected_count,
        "Timeline should render {} steps, but rendered {}",
        expected_count, step_card_count
    );
}
