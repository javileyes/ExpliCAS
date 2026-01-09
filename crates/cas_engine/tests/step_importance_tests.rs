//! Test to verify step importance is correctly propagated from rules
use cas_engine::Simplifier;
use cas_engine::step::ImportanceLevel;

#[test]
fn test_identity_property_steps_have_medium_importance() {
    let mut simplifier = Simplifier::with_default_rules();
    simplifier.set_collect_steps(true);
    
    // Expression with multiplication by 1: will trigger MulOneRule
    let expr = cas_parser::parse("x * 1", &mut simplifier.context).unwrap();
    
    let (_result, steps) = simplifier.simplify(expr);
    
    // Find Identity Property steps
    let identity_steps: Vec<_> = steps.iter()
        .filter(|s| s.rule_name == "Identity Property of Multiplication")
        .collect();
    
    assert!(!identity_steps.is_empty(), "Should have at least one Identity Property step");
    
    for step in &identity_steps {
        eprintln!("Step: {}, importance field: {:?}, get_importance(): {:?}", 
            step.description, step.importance, step.get_importance());
        
        assert_eq!(step.importance, ImportanceLevel::Medium, 
            "Step '{}' should have importance field set to Medium, got {:?}", 
            step.description, step.importance);
        
        assert_eq!(step.get_importance(), ImportanceLevel::Medium, 
            "Step '{}' get_importance() should return Medium, got {:?}", 
            step.description, step.get_importance());
    }
}
