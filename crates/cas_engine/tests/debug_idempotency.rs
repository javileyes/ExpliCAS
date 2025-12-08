// Debug test for idempotency failure
// Minimal failing case from property test
use cas_ast::{Constant, Context, DisplayExpr, Expr};
use cas_engine::Simplifier;

#[test]
fn test_idempotency_minimal_case() {
    // Enable verbose comparison logging
    cas_engine::ordering::enable_compare_debug();

    // Minimal failing input from proptest:
    // Function("cos", [Add(Constant(E), Constant(E))])
    let mut ctx = Context::new();
    let e1 = ctx.add(Expr::Constant(Constant::E));
    let e2 = ctx.add(Expr::Constant(Constant::E));
    let add = ctx.add(Expr::Add(e1, e2));
    let cos_expr = ctx.add(Expr::Function("cos".to_string(), vec![add]));

    let mut simplifier = Simplifier::with_default_rules();
    simplifier.context = ctx;

    eprintln!("\n=== FIRST SIMPLIFY ===");
    let (s1, steps1) = simplifier.simplify(cos_expr);
    eprintln!("Steps count: {}", steps1.len());

    let result1 = DisplayExpr {
        context: &simplifier.context,
        id: s1,
    }
    .to_string();
    eprintln!("Result 1: {}", result1);

    eprintln!("\n=== SECOND SIMPLIFY ===");
    let (s2, steps2) = simplifier.simplify(s1);
    eprintln!("Steps count: {}", steps2.len());

    let result2 = DisplayExpr {
        context: &simplifier.context,
        id: s2,
    }
    .to_string();
    eprintln!("Result 2: {}", result2);

    // Print all steps from second simplify to see what changed
    if !steps2.is_empty() {
        eprintln!("\n=== STEPS IN SECOND SIMPLIFY ===");
        for (i, step) in steps2.iter().enumerate() {
            eprintln!("Step {}: {} - {}", i + 1, step.rule_name, step.description);
        }
    }

    if result1 != result2 {
        eprintln!("\n❌ IDEMPOTENCY FAILURE!");
        eprintln!("  Result 1: {}", result1);
        eprintln!("  Result 2: {}", result2);
        panic!("Idempotency violated: results differ");
    } else {
        eprintln!("\n✅ Idempotency OK");
    }
}
