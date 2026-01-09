//! Test to verify timeline step filtering for atan expression
use cas_engine::Simplifier;
use cas_engine::step::ImportanceLevel;
use cas_engine::eval_step_pipeline::to_display_steps;

#[test]
fn test_atan_expression_step_importance() {
    let mut simplifier = Simplifier::with_default_rules();
    simplifier.set_collect_steps(true);
    
    // Expression: atan(3) + (atan(1/3) - pi/2)
    let expr = cas_parser::parse("atan(3) + (atan(1/3) - pi/2)", &mut simplifier.context).unwrap();
    
    let (_result, raw_steps) = simplifier.simplify(expr);
    
    eprintln!("\n=== RAW STEPS ({}) ===", raw_steps.len());
    for (i, step) in raw_steps.iter().enumerate() {
        eprintln!("  {}: {} [{}] importance={:?} get_importance={:?}", 
            i+1, step.description, step.rule_name, step.importance, step.get_importance());
    }
    
    // Apply pipeline
    let display_steps = to_display_steps(raw_steps);
    
    eprintln!("\n=== DISPLAY STEPS ({}) ===", display_steps.len());
    for (i, step) in display_steps.iter().enumerate() {
        eprintln!("  {}: {} [{}] importance={:?} get_importance={:?}", 
            i+1, step.description, step.rule_name, step.importance, step.get_importance());
    }
    
    // Count Identity Property steps
    let identity_steps: Vec<_> = display_steps.iter()
        .filter(|s| s.rule_name == "Identity Property of Multiplication")
        .collect();
    
    eprintln!("\n=== IDENTITY PROPERTY STEPS ({}) ===", identity_steps.len());
    for step in &identity_steps {
        eprintln!("  {} importance={:?} get_importance={:?}", 
            step.description, step.importance, step.get_importance());
    }
    
    // All Identity Property steps should have Medium importance
    for step in &identity_steps {
        assert_eq!(step.get_importance(), ImportanceLevel::Medium, 
            "Step '{}' should have Medium importance", step.description);
    }
    
    // Check that these steps would be shown in Normal verbosity
    let shown_in_normal: Vec<_> = display_steps.iter()
        .filter(|s| s.get_importance() >= ImportanceLevel::Medium)
        .collect();
    
    eprintln!("\n=== STEPS SHOWN IN NORMAL VERBOSITY ({}) ===", shown_in_normal.len());
    
    // Verify Identity Property steps are in the shown list
    let identity_shown = shown_in_normal.iter()
        .filter(|s| s.rule_name == "Identity Property of Multiplication")
        .count();
    
    assert_eq!(identity_shown, identity_steps.len(), 
        "All {} Identity Property steps should be shown in Normal verbosity", identity_steps.len());
}
