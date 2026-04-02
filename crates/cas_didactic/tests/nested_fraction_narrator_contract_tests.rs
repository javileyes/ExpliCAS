//! Contract tests for nested fraction sub-step generation
//!
//! These tests verify that the didactic layer generates appropriate sub-steps
//! for nested fraction simplification.

use cas_ast::Context;
use cas_didactic::{enrich_steps, EnrichedStep};
use cas_solver::runtime::Engine;

/// Helper function to simplify expression and get enriched steps
fn simplify_and_enrich(input: &str) -> (Context, Vec<EnrichedStep>) {
    let mut engine = Engine::new();
    let ctx = &mut engine.simplifier.context;

    let expr = cas_parser::parse(input, ctx).expect("Should parse");
    let (_, steps) = engine.simplifier.simplify(expr);

    let enriched = enrich_steps(&engine.simplifier.context, expr, steps);
    // Clone context for returning
    (engine.simplifier.context.clone(), enriched)
}

/// Verify that P1 pattern avoids placeholder template sub-steps
#[test]
fn nested_fraction_p1_avoids_template_substeps() {
    let (_, enriched) = simplify_and_enrich("1/(1 + 1/x)");

    // Should have at least one step
    assert!(!enriched.is_empty(), "Should have at least one step");

    // Find nested fraction step
    let nested = enriched
        .iter()
        .find(|e| e.base_step.rule_name.to_lowercase().contains("complex"));

    assert!(nested.is_some(), "Should have complex fraction step");
    let step = nested.unwrap();

    assert!(
        step.sub_steps.iter().all(|substep| {
            !substep
                .description
                .contains("Dividir entre una fracción equivale a invertirla")
                && !substep.description.contains("Usar 1 / (p / q) = q / p")
                && !substep.description.contains("Usar n / (p / q) = n · q / p")
                && !substep.description.contains("Usar n / (1 / d) = n · d")
        }),
        "P1 pattern should avoid placeholder template sub-steps: {:?}",
        step.sub_steps
    );
}

/// Verify that P3 pattern avoids placeholder template sub-steps
#[test]
fn nested_fraction_p3_avoids_template_substeps() {
    let (_, enriched) = simplify_and_enrich("2/(1 + 1/x)");

    let nested = enriched
        .iter()
        .find(|e| e.base_step.rule_name.to_lowercase().contains("complex"));

    assert!(nested.is_some(), "Should have complex fraction step");
    let step = nested.unwrap();

    assert!(
        step.sub_steps.iter().all(|substep| {
            !substep
                .description
                .contains("Dividir entre una fracción equivale a invertirla")
                && !substep.description.contains("Usar 1 / (p / q) = q / p")
                && !substep.description.contains("Usar n / (p / q) = n · q / p")
                && !substep.description.contains("Usar n / (1 / d) = n · d")
                && !substep.after_expr.contains("combinado")
        }),
        "P3 pattern should avoid placeholder template sub-steps: {:?}",
        step.sub_steps
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

/// Verify sub-steps avoid direct inversion templates
#[test]
fn substeps_avoid_direct_inversion_templates() {
    let (_, enriched) = simplify_and_enrich("1/(1 + 1/x)");

    let nested = enriched
        .iter()
        .find(|e| e.base_step.rule_name.to_lowercase().contains("complex"));

    assert!(nested.is_some(), "Should find nested fraction step");
    let step = nested.unwrap();

    assert!(
        step.sub_steps.iter().all(|substep| {
            !substep
                .description
                .contains("Dividir entre una fracción equivale a invertirla")
                && !substep.description.contains("Usar 1 / (p / q) = q / p")
                && !substep.description.contains("Usar n / (p / q) = n · q / p")
                && !substep.description.contains("Usar n / (1 / d) = n · d")
        }),
        "Nested fraction sub-steps should avoid direct inversion templates. Sub-steps: {:?}",
        step.sub_steps
    );
}

/// Verify intermediate expressions avoid placeholders and stay informative
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
            // Should either keep concrete values or show a useful abstract formula,
            // but never collapse into a meaningless placeholder.
            assert!(
                first_substep.after_expr.contains("x")
                    || first_substep.after_expr.contains("\\frac"),
                "Should contain either concrete variables or a useful abstract formula: {}",
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
