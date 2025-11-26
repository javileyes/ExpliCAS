use cas_ast::Expr;
use cas_engine::Simplifier;
use cas_engine::rules::arithmetic::{AddZeroRule, MulOneRule, CombineConstantsRule};
use cas_parser::parse;

#[test]
fn test_end_to_end_simplification() {
    // Setup
    let mut simplifier = Simplifier::new();
    simplifier.add_rule(Box::new(AddZeroRule));
    simplifier.add_rule(Box::new(MulOneRule));
    simplifier.add_rule(Box::new(CombineConstantsRule));

    // Input: 2 * 3 + 0
    let input = "2 * 3 + 0";
    let expr = parse(input).expect("Failed to parse");

    // Simplify
    let (result, steps) = simplifier.simplify(expr);

    // Verify Result
    assert_eq!(format!("{}", result), "6");

    // Verify Steps
    assert_eq!(steps.len(), 2);
    assert_eq!(steps[0].rule_name, "Combine Constants");
    assert_eq!(format!("{}", steps[0].after), "(6 + 0)");
    
    assert_eq!(steps[1].rule_name, "Identity Property of Addition");
    assert_eq!(format!("{}", steps[1].after), "6");
}

#[test]
fn test_nested_simplification() {
    let mut simplifier = Simplifier::new();
    simplifier.add_rule(Box::new(CombineConstantsRule));

    // Input: (1 + 2) * (3 + 4)
    let input = "(1 + 2) * (3 + 4)";
    let expr = parse(input).expect("Failed to parse");

    let (result, steps) = simplifier.simplify(expr);

    assert_eq!(format!("{}", result), "21");
    // Steps: 
    // 1. 1+2 -> 3
    // 2. 3+4 -> 7
    // 3. 3*7 -> 21
    assert_eq!(steps.len(), 3);
}
