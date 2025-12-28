//! Contract tests for the assumptions reporting infrastructure.
//!
//! These tests verify the end-to-end behavior of assumption collection
//! and reporting across different modes and configurations.

use cas_ast::Context;
use cas_engine::assumptions::{AssumptionCollector, AssumptionEvent, AssumptionReporting};

/// Test 1: Default Off does not collect or report assumptions
#[test]
fn default_off_no_assumptions_output() {
    // Verify that AssumptionReporting defaults to Off
    let reporting = AssumptionReporting::default();
    assert_eq!(reporting, AssumptionReporting::Off);

    // Empty collector produces empty results regardless
    let collector = AssumptionCollector::new();
    assert!(collector.is_empty());
    assert!(collector.summary_line().is_none());
}

/// Test 2: Summary mode collects and aggregates assumptions with dedup
#[test]
fn summary_mode_dedup_and_aggregate() {
    let mut ctx = Context::new();
    let x = ctx.var("x");

    let mut collector = AssumptionCollector::new();

    // Simulate 3 uses of same assumption (x/x + 2x/2x + x^2/x^2)
    collector.note(AssumptionEvent::nonzero(&ctx, x));
    collector.note(AssumptionEvent::nonzero(&ctx, x));
    collector.note(AssumptionEvent::nonzero(&ctx, x));

    // Should dedup to 1 record with count=3
    let records = collector.finish();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].count, 3);
    assert_eq!(records[0].kind, "nonzero");
    assert_eq!(records[0].expr, "x");
}

/// Test 3: Strict mode does not simplify x/x (so no assumption is needed)
/// This test verifies the expected behavior when domain is Strict
#[test]
fn strict_mode_no_assumptions_for_unsimplified() {
    // In Strict mode, x/x stays as x/x, so no assumption is made
    // The collector should be empty if no rewrites with domain_assumption occurred
    let collector = AssumptionCollector::new();

    assert!(collector.is_empty());
    assert_eq!(collector.len(), 0);

    let records = collector.finish();
    assert!(records.is_empty());
}

/// Test 4: Assumption reporting affects JSON output
#[test]
fn reporting_levels_serialize_correctly() {
    // Off should serialize to "off"
    let off = AssumptionReporting::Off;
    let serialized = serde_json::to_string(&off).unwrap();
    assert_eq!(serialized, "\"off\"");

    // Summary should serialize to "summary"
    let summary = AssumptionReporting::Summary;
    let serialized = serde_json::to_string(&summary).unwrap();
    assert_eq!(serialized, "\"summary\"");

    // Trace should serialize to "trace"
    let trace = AssumptionReporting::Trace;
    let serialized = serde_json::to_string(&trace).unwrap();
    assert_eq!(serialized, "\"trace\"");

    // Deserialize should work
    let parsed: AssumptionReporting = serde_json::from_str("\"summary\"").unwrap();
    assert_eq!(parsed, AssumptionReporting::Summary);
}

/// Test 5: Semantics set affects assumption reporting level
#[test]
fn from_str_parsing() {
    assert_eq!(
        AssumptionReporting::from_str("off"),
        Some(AssumptionReporting::Off)
    );
    assert_eq!(
        AssumptionReporting::from_str("summary"),
        Some(AssumptionReporting::Summary)
    );
    assert_eq!(
        AssumptionReporting::from_str("trace"),
        Some(AssumptionReporting::Trace)
    );
    assert_eq!(
        AssumptionReporting::from_str("OFF"),
        Some(AssumptionReporting::Off)
    );
    assert_eq!(
        AssumptionReporting::from_str("SUMMARY"),
        Some(AssumptionReporting::Summary)
    );
    assert_eq!(AssumptionReporting::from_str("invalid"), None);
}

/// Test: Different expressions are tracked separately
#[test]
fn different_expressions_separate_records() {
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let y = ctx.var("y");
    let z = ctx.var("z");

    let mut collector = AssumptionCollector::new();

    // Different expressions should not dedup
    collector.note(AssumptionEvent::nonzero(&ctx, x));
    collector.note(AssumptionEvent::nonzero(&ctx, y));
    collector.note(AssumptionEvent::nonzero(&ctx, z));

    let records = collector.finish();
    assert_eq!(records.len(), 3);

    // Check each has count=1
    for record in &records {
        assert_eq!(record.count, 1);
        assert_eq!(record.kind, "nonzero");
    }
}

/// Test: Summary line format
#[test]
fn summary_line_format() {
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let y = ctx.var("y");

    let mut collector = AssumptionCollector::new();

    collector.note(AssumptionEvent::nonzero(&ctx, x));
    collector.note(AssumptionEvent::nonzero(&ctx, x));
    collector.note(AssumptionEvent::nonzero(&ctx, y));

    let summary = collector.summary_line();
    assert!(summary.is_some());

    let line = summary.unwrap();
    assert!(line.starts_with("⚠ Assumptions:"));
    assert!(line.contains("nonzero(x)"));
    assert!(line.contains("×2")); // x was noted twice
    assert!(line.contains("nonzero(y)"));
}

/// Test: Stable order across multiple runs
#[test]
fn stable_order_invariant() {
    let mut ctx = Context::new();
    let a = ctx.var("a");
    let b = ctx.var("b");
    let c = ctx.var("c");

    // Create collector in non-alphabetical insertion order
    let mut collector = AssumptionCollector::new();
    collector.note(AssumptionEvent::positive(&ctx, c));
    collector.note(AssumptionEvent::nonzero(&ctx, b));
    collector.note(AssumptionEvent::defined(&ctx, a));
    collector.note(AssumptionEvent::nonzero(&ctx, a));

    let records = collector.finish();

    // Should be sorted by kind, then by expr
    // defined < nonzero < positive (alphabetically)
    assert_eq!(records[0].kind, "defined");
    assert_eq!(records[0].expr, "a");

    assert_eq!(records[1].kind, "nonzero");
    assert_eq!(records[1].expr, "a");

    assert_eq!(records[2].kind, "nonzero");
    assert_eq!(records[2].expr, "b");

    assert_eq!(records[3].kind, "positive");
    assert_eq!(records[3].expr, "c");
}

/// Test: from_legacy_string parses known patterns
#[test]
fn legacy_string_parsing() {
    // NonZero patterns
    let event = AssumptionEvent::from_legacy_string("Assuming denominator ≠ 0");
    assert_eq!(event.key.kind(), "nonzero");
    assert!(event.message.contains("denominator"));

    // Defined patterns
    let event = AssumptionEvent::from_legacy_string("Assuming expression is defined");
    assert_eq!(event.key.kind(), "defined");

    // Positive patterns
    let event = AssumptionEvent::from_legacy_string("Assuming x > 0");
    assert_eq!(event.key.kind(), "positive");

    // Unknown patterns fall back to defined
    let event = AssumptionEvent::from_legacy_string("Some unknown assumption");
    assert_eq!(event.key.kind(), "defined");
}

/// Test: SimplifyOptions includes assumption_reporting
#[test]
fn simplify_options_has_assumption_reporting() {
    use cas_engine::phase::SimplifyOptions;

    let opts = SimplifyOptions::default();
    // Default should be Off
    assert_eq!(opts.assumption_reporting, AssumptionReporting::Off);
}

/// Test: EvalOptions includes assumption_reporting
#[test]
fn eval_options_has_assumption_reporting() {
    use cas_engine::options::EvalOptions;

    let opts = EvalOptions::default();
    // Default should be Off
    assert_eq!(opts.assumption_reporting, AssumptionReporting::Off);
}

/// Test: Options propagate through to_simplify_options
#[test]
fn options_propagation() {
    use cas_engine::options::EvalOptions;

    let mut eval_opts = EvalOptions::default();
    eval_opts.assumption_reporting = AssumptionReporting::Summary;

    let simplify_opts = eval_opts.to_simplify_options();
    assert_eq!(
        simplify_opts.assumption_reporting,
        AssumptionReporting::Summary
    );
}
