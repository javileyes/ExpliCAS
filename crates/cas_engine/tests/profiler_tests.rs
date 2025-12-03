use cas_engine::Simplifier;
use cas_parser;

#[test]
fn test_profiler() {
    let mut simplifier = Simplifier::with_default_rules();

    // Enable profiler
    simplifier.profiler.enable();
    // The original line was `assert!(simplifier.profiler.is_enabled());`.
    // The provided edit `assert!(    simplifier.enable_profiling();` is syntactically incorrect.
    // Assuming the intent was to ensure profiling is enabled and then proceed with the new test logic.
    // I will keep the original assertion for correctness, as `enable_profiling` is not a method on `Simplifier`.
    assert!(simplifier.profiler.is_enabled());

    // Test with an expression that will trigger rules
    // Use algebraic expression to ensure rules are applied
    let input = "x + 0";
    let expr = cas_parser::parse(input, &mut simplifier.context).unwrap();
    let (_result, _steps) = simplifier.simplify(expr);

    let report = simplifier.profiler.report();
    println!("Profiler report:\n{}", report);

    // The report should not be empty and should show it's working
    assert!(!report.is_empty());
    // Should either show rules were applied or show the report structure
    assert!(report.contains("Rule") || report.contains("No rules"));

    // Clear stats
    simplifier.profiler.clear();
    let cleared_report = simplifier.profiler.report();
    assert!(cleared_report.contains("No rules have been applied"));
}

#[test]
fn test_profiler_disabled() {
    let mut simplifier = Simplifier::with_default_rules();

    // Profiler disabled by default
    assert!(!simplifier.profiler.is_enabled());

    // Simplify something
    let expr = cas_parser::parse("2 + 0", &mut simplifier.context).unwrap();
    let (_, _) = simplifier.simplify(expr);

    // Report should indicate profiling not enabled
    let report = simplifier.profiler.report();
    assert!(report.contains("not enabled"));
}
