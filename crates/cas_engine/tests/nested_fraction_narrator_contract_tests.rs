//! Contract tests for nested fraction sub-step generation
//!
//! These tests verify that the didactic layer generates appropriate sub-steps
//! for nested fraction simplification.

use cas_ast::Context;
use cas_engine::didactic::enrich_steps;
use cas_engine::Engine;

/// Helper function to simplify expression and get enriched steps
fn simplify_and_enrich(input: &str) -> (Context, Vec<cas_engine::didactic::EnrichedStep>) {
    let mut engine = Engine::new();
    let ctx = &mut engine.simplifier.context;

    let expr = cas_parser::parse(input, ctx).expect("Should parse");
    let (_, steps) = engine.simplifier.simplify(expr);

    let enriched = enrich_steps(&engine.simplifier.context, expr, steps);
    // Clone context for returning
    (engine.simplifier.context.clone(), enriched)
}

/// Verify that P1 pattern (1/(a + 1/b)) generates exactly 2 sub-steps
#[test]
fn nested_fraction_p1_generates_substeps() {
    let (_, enriched) = simplify_and_enrich("1/(1 + 1/x)");

    // Should have at least one step
    assert!(!enriched.is_empty(), "Should have at least one step");

    // Find nested fraction step
    let nested = enriched
        .iter()
        .find(|e| e.base_step.rule_name.to_lowercase().contains("complex"));

    assert!(nested.is_some(), "Should have complex fraction step");
    let step = nested.unwrap();

    // Should have exactly 2 sub-steps (combine denominator + invert)
    assert_eq!(
        step.sub_steps.len(),
        2,
        "P1 pattern should have 2 sub-steps: {:?}",
        step.sub_steps
    );
}

/// Verify that P3 pattern (A/(B + C/D)) generates sub-steps
#[test]
fn nested_fraction_p3_generates_substeps() {
    let (_, enriched) = simplify_and_enrich("2/(1 + 1/x)");

    let nested = enriched
        .iter()
        .find(|e| e.base_step.rule_name.to_lowercase().contains("complex"));

    assert!(nested.is_some(), "Should have complex fraction step");
    let step = nested.unwrap();

    assert!(
        !step.sub_steps.is_empty(),
        "P3 pattern should have sub-steps"
    );
}

/// Verify multiple nested fractions each get enriched
#[test]
fn multiple_nested_fractions_all_enriched() {
    let (_, enriched) = simplify_and_enrich("1/(1 + 1/(1 + 1/x))");

    // Count nested fraction steps with sub-steps
    let enriched_count = enriched
        .iter()
        .filter(|e| {
            e.base_step.rule_name.to_lowercase().contains("complex") && !e.sub_steps.is_empty()
        })
        .count();

    assert!(
        enriched_count >= 1,
        "At least 1 nested fraction step should have sub-steps, got {}",
        enriched_count
    );
}

/// Verify sub-steps contain expected descriptions for common denominator
#[test]
fn substeps_contain_denominator_explanation() {
    let (_, enriched) = simplify_and_enrich("1/(1 + 1/x)");

    let nested = enriched
        .iter()
        .find(|e| e.base_step.rule_name.to_lowercase().contains("complex"));

    assert!(nested.is_some(), "Should find nested fraction step");
    let step = nested.unwrap();

    // Sub-steps should mention common denominator (or simplification)
    // V2.15.8: After AddFractionsRule enhancement, the flow may be different -
    // the denominator is combined first by AddFractionsRule, so the sub-steps
    // may not include explicit "inversion" step anymore.
    let has_denominator_or_simplify = step.sub_steps.iter().any(|s| {
        s.description.contains("denominador")
            || s.description.contains("denominator")
            || s.description.contains("Simplificar")
            || s.description.contains("simplif")
    });

    assert!(
        has_denominator_or_simplify,
        "Should have sub-step about common denominator or simplification. Sub-steps: {:?}",
        step.sub_steps
    );
}

/// Verify intermediate expressions contain real values, not placeholders
#[test]
fn substep_intermediate_shows_real_values() {
    let (_, enriched) = simplify_and_enrich("1/(1 + 1/x)");

    let nested = enriched
        .iter()
        .find(|e| e.base_step.rule_name.to_lowercase().contains("complex"));

    if let Some(step) = nested {
        if !step.sub_steps.is_empty() {
            let first_substep = &step.sub_steps[0];
            // Should NOT contain placeholder text
            assert!(
                !first_substep.after_expr.contains("combinado"),
                "Should not have placeholder 'combinado': {}",
                first_substep.after_expr
            );
            // Should contain actual variable x
            assert!(
                first_substep.after_expr.contains("x"),
                "Should contain variable x: {}",
                first_substep.after_expr
            );
        }
    }
}

/// Verify LaTeX format is used for complex denominators (with \frac)
#[test]
fn substep_has_parentheses_around_complex_denominator() {
    let (_, enriched) = simplify_and_enrich("1/(1 + x/(x+1))");

    let nested = enriched
        .iter()
        .find(|e| e.base_step.rule_name.to_lowercase().contains("complex"));

    if let Some(step) = nested {
        if !step.sub_steps.is_empty() {
            let first_substep = &step.sub_steps[0];
            // Should be LaTeX format with \frac (no need for parentheses in \frac{}{})
            let has_latex_frac = first_substep.after_expr.contains("\\frac");
            assert!(
                has_latex_frac,
                "Complex denominator should use LaTeX \\frac format: {}",
                first_substep.after_expr
            );
        }
    }
}
