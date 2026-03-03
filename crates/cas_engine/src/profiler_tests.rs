use crate::phase::SimplifyPhase;
use crate::profiler::RuleProfiler;

#[test]
fn test_profiler_disabled_by_default() {
    let profiler = RuleProfiler::default();
    assert!(!profiler.is_enabled());
}

#[test]
fn test_profiler_record_and_report() {
    let mut profiler = RuleProfiler::new(true);

    profiler.record(SimplifyPhase::Core, "AddZeroRule");
    profiler.record(SimplifyPhase::Core, "AddZeroRule");
    profiler.record(SimplifyPhase::Transform, "MulOneRule");

    let report = profiler.report();
    assert!(report.contains("AddZeroRule"));
    assert!(report.contains("2"));
    assert!(report.contains("MulOneRule"));
    assert!(report.contains("1"));
}

#[test]
fn test_profiler_clear() {
    let mut profiler = RuleProfiler::new(true);

    profiler.record(SimplifyPhase::Core, "TestRule");
    assert!(profiler.report().contains("TestRule"));

    profiler.clear();
    assert!(profiler.report().contains("No rules"));
}

#[test]
fn test_per_phase_isolation() {
    let mut profiler = RuleProfiler::new(true);

    profiler.record(SimplifyPhase::Core, "CoreRule");
    profiler.record(SimplifyPhase::Core, "CoreRule");
    profiler.record(SimplifyPhase::Transform, "TransformRule");

    // Check Core phase
    let top_core = profiler.top_applied_for_phase(SimplifyPhase::Core, 5);
    assert_eq!(top_core.len(), 1);
    assert_eq!(top_core[0].0, "CoreRule");
    assert_eq!(top_core[0].1, 2);

    // Check Transform phase
    let top_transform = profiler.top_applied_for_phase(SimplifyPhase::Transform, 5);
    assert_eq!(top_transform.len(), 1);
    assert_eq!(top_transform[0].0, "TransformRule");
    assert_eq!(top_transform[0].1, 1);

    // Check Rationalize phase (should be empty)
    let top_rat = profiler.top_applied_for_phase(SimplifyPhase::Rationalize, 5);
    assert!(top_rat.is_empty());
}

#[test]
fn test_health_report_per_phase() {
    let mut profiler = RuleProfiler::new(true);
    profiler.enable_health();

    profiler.record_with_delta(SimplifyPhase::Core, "CoreRule", 5);
    profiler.record_with_delta(SimplifyPhase::Transform, "Distribute", 10);

    // Core report should only show CoreRule
    let core_report = profiler.health_report_for_phase(Some(SimplifyPhase::Core));
    assert!(core_report.contains("CoreRule"));
    assert!(!core_report.contains("Distribute"));

    // Transform report should only show Distribute
    let transform_report = profiler.health_report_for_phase(Some(SimplifyPhase::Transform));
    assert!(transform_report.contains("Distribute"));
    assert!(!transform_report.contains("CoreRule"));

    // Aggregate should show both
    let agg_report = profiler.health_report();
    assert!(agg_report.contains("CoreRule"));
    assert!(agg_report.contains("Distribute"));
}
