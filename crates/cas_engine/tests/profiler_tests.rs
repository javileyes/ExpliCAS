use cas_engine::Simplifier;
use cas_parser;

#[test]
fn test_profiler() {
    let mut simplifier = Simplifier::with_default_rules();
    
    // Enable profiler
    simplifier.profiler.enable();
    assert!(simplifier.profiler.is_enabled());
    
    // Parse and simplify a simple expression
    let expr = cas_parser::parse("2 + 0 + x * 1", &mut simplifier.context).unwrap();
    let (_, _) = simplifier.simplify(expr);
    
    // Get profiler report
    let report = simplifier.profiler.report();
    
    // Should contain some rules
    assert!(report.contains("Rule Profiling Report"));
    assert!(report.contains("TOTAL"));
    
    // Should have add zero and mul one rules
    assert!(report.contains("Add Zero") || report.contains("Mul By One"));
    
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
