use cas_ast::{Constant, Context, Expr};
use cas_engine::engine::Simplifier;

#[test]
fn test_infinity_simplification() {
    let mut simplifier = Simplifier::with_default_rules();

    // Create infinity
    let inf = simplifier.context.add(Expr::Constant(Constant::Infinity));
    println!("Created infinity ExprId: {:?}", inf);
    println!("Infinity Expr: {:?}", simplifier.context.get(inf));

    // Simplify infinity
    let (simplified, steps) = simplifier.simplify(inf);
    println!("Simplified ExprId: {:?}", simplified);
    println!("Simplified Expr: {:?}", simplifier.context.get(simplified));

    for step in steps {
        println!("Step: {}", step.description);
    }

    // Check if it became undefined
    match simplifier.context.get(simplified) {
        Expr::Constant(Constant::Undefined) => {
            panic!("Infinity simplified to Undefined!");
        }
        Expr::Constant(Constant::Infinity) => {
            println!("Infinity correctly preserved");
        }
        other => {
            println!("Infinity became: {:?}", other);
        }
    }
}

#[test]
fn test_division_by_zero_creates_undefined() {
    let mut ctx = Context::new();
    let mut simplifier = Simplifier::with_default_rules();
    std::mem::swap(&mut simplifier.context, &mut ctx);

    // Create 1/0
    let one = simplifier.context.num(1);
    let zero = simplifier.context.num(0);
    let div = simplifier.context.add(Expr::Div(one, zero));

    println!("Created 1/0 ExprId: {:?}", div);

    // Simplify
    let (simplified, steps) = simplifier.simplify(div);
    println!("Simplified ExprId: {:?}", simplified);
    println!("Simplified Expr: {:?}", simplifier.context.get(simplified));

    for step in steps {
        println!("Step: {}", step.description);
    }

    // This SHOULD create undefined
    match simplifier.context.get(simplified) {
        Expr::Constant(Constant::Undefined) => {
            println!("Division by zero correctly creates Undefined");
        }
        other => {
            panic!("Expected Undefined, got: {:?}", other);
        }
    }
}
