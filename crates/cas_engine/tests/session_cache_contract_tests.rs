//! Session Reference Caching Contract Tests (V2.15.36)
//!
//! These tests verify that:
//! 1. Cache hits generate a synthetic "Use cached result" step
//! 2. Multiple cache hits are aggregated into a single step
//! 3. The synthetic step has Medium importance (visible in timeline)

use cas_engine::eval::{Engine, EvalAction, EvalRequest};
use cas_engine::step::ImportanceLevel;
use cas_session::SessionState;

/// Helper: create request from expression string
fn make_request(engine: &mut Engine, input: &str) -> EvalRequest {
    let expr_id = cas_parser::parse(input, &mut engine.simplifier.context).expect("parse failed");
    EvalRequest {
        parsed: expr_id,
        raw_input: input.to_string(),
        action: EvalAction::Simplify,
        auto_store: true,
    }
}

/// V2.15.36: Verify that referencing a cached entry produces a synthetic step
#[test]
fn test_cache_hit_produces_synthetic_step() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();

    // First: evaluate an identity that simplifies (sin²x + cos²x = 1)
    let req1 = make_request(&mut engine, "sin(x)^2 + cos(x)^2");
    engine.eval(&mut state, req1).unwrap();

    // Second: reference #1 in a new expression
    let req2 = make_request(&mut engine, "#1 + 5");
    let output2 = engine.eval(&mut state, req2).unwrap();

    // Verify: steps should contain a "Use cached result" step
    let has_cache_step = output2.steps.iter().any(|step| {
        step.rule_name == "Use cached result"
            && step
                .description
                .contains("Used cached simplified result from #1")
    });

    assert!(
        has_cache_step,
        "Expected a synthetic 'Use cached result' step for #1, but steps were: {:?}",
        output2
            .steps
            .iter()
            .map(|s| &s.rule_name)
            .collect::<Vec<_>>()
    );
}

/// V2.15.36: Verify that multiple cache hits are aggregated into one step
#[test]
fn test_multiple_cache_hits_aggregated() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();

    // Create two entries
    let req1 = make_request(&mut engine, "x + 1");
    engine.eval(&mut state, req1).unwrap();

    let req2 = make_request(&mut engine, "x + 2");
    engine.eval(&mut state, req2).unwrap();

    // Reference both
    let req3 = make_request(&mut engine, "#1 + #2");
    let output3 = engine.eval(&mut state, req3).unwrap();

    // Verify: one synthetic step mentioning both #1 and #2
    let cache_steps: Vec<_> = output3
        .steps
        .iter()
        .filter(|s| s.rule_name == "Use cached result")
        .collect();

    assert_eq!(
        cache_steps.len(),
        1,
        "Expected exactly one aggregated cache step, got {}",
        cache_steps.len()
    );

    let step = cache_steps[0];
    assert!(
        step.description.contains("#1") && step.description.contains("#2"),
        "Expected both #1 and #2 in description, got: {}",
        step.description
    );
}

/// V2.15.36: Verify synthetic step has Medium importance (visible in timeline)
#[test]
fn test_cache_step_has_medium_importance() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();

    // Create and reference an entry
    let req1 = make_request(&mut engine, "x");
    engine.eval(&mut state, req1).unwrap();

    let req2 = make_request(&mut engine, "#1 + 1");
    let output2 = engine.eval(&mut state, req2).unwrap();

    let cache_step = output2
        .steps
        .iter()
        .find(|s| s.rule_name == "Use cached result");

    assert!(cache_step.is_some(), "Expected cache step to exist");

    let step = cache_step.unwrap();
    assert_eq!(
        step.importance,
        ImportanceLevel::Medium,
        "Cache step should have Medium importance for visibility"
    );
}

/// V2.15.36: Verify dedup - same ref multiple times produces one entry in description
#[test]
fn test_cache_hit_dedup_same_ref() {
    let mut engine = Engine::new();
    let mut state = SessionState::new();

    // Create entry #1
    let req1 = make_request(&mut engine, "5");
    engine.eval(&mut state, req1).unwrap();

    // Reference #1 three times
    let req2 = make_request(&mut engine, "#1 + #1 + #1");
    let output2 = engine.eval(&mut state, req2).unwrap();

    // Verify: only one #1 in the description (deduped)
    let cache_steps: Vec<_> = output2
        .steps
        .iter()
        .filter(|s| s.rule_name == "Use cached result")
        .collect();

    assert_eq!(cache_steps.len(), 1, "Expected one cache step");

    // Count occurrences of "#1" in description - should be exactly 1
    let desc = &cache_steps[0].description;
    let count = desc.matches("#1").count();
    assert_eq!(
        count, 1,
        "Expected #1 to appear once (deduped), but found {} times in: {}",
        count, desc
    );
}
