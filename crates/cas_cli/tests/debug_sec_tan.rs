use cas_ast::DisplayExpr;
use cas_engine::Simplifier;
use cas_parser::parse;

/// Debug test to trace what's happening with sec²-tan²-1
/// Run with: RUST_LOG=cas_engine=debug cargo test -p cas_cli --test debug_sec_tan -- --nocapture
#[test]
fn debug_sec_tan_pythagorean() {
    // Initialize tracing
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .try_init();

    println!("\n=== STARTING DEBUG OF sec(x)^2 - tan(x)^2 - 1 ===\n");

    let mut simplifier = Simplifier::with_default_rules();

    // Enable debug mode in simplifier if available
    simplifier.enable_debug();

    let expr = parse("sec(x)^2 - tan(x)^2 - 1", &mut simplifier.context).unwrap();

    println!(
        "Parsed expression: {}",
        DisplayExpr {
            context: &simplifier.context,
            id: expr
        }
    );

    println!("\n--- Starting simplification (this may overflow) ---\n");

    // Try to simplify with manual iteration control
    let (result, steps) = simplifier.simplify(expr);

    println!("\n=== SIMPLIFICATION COMPLETED ===");
    println!("Number of steps: {}", steps.len());
    println!("\nSteps taken:");
    for (i, step) in steps.iter().enumerate().take(50) {
        // Limit output
        println!("{}. {} [{}]", i + 1, step.description, step.rule_name);
    }

    if steps.len() > 50 {
        println!("... and {} more steps", steps.len() - 50);
    }

    println!(
        "\nFinal result: {}",
        DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    );
}

/// Safer version with manual iteration limiting
#[test]
fn debug_sec_tan_limited() {
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .try_init();

    println!("\n=== LIMITED ITERATION DEBUG ===\n");

    let mut simplifier = Simplifier::with_default_rules();
    let expr = parse("sec(x)^2 - tan(x)^2 - 1", &mut simplifier.context).unwrap();

    println!(
        "Initial: {}",
        DisplayExpr {
            context: &simplifier.context,
            id: expr
        }
    );

    let mut current = expr;
    let max_iterations = 20;

    for iteration in 0..max_iterations {
        println!("\n--- Iteration {} ---", iteration + 1);
        let (new_expr, steps) = simplifier.apply_rules_loop(current);

        if steps.is_empty() {
            println!("No more rules applied. Converged!");
            break;
        }

        println!("Rules applied:");
        for step in &steps {
            println!("  - {} [{}]", step.description, step.rule_name);
        }

        println!(
            "After: {}",
            DisplayExpr {
                context: &simplifier.context,
                id: new_expr
            }
        );

        if new_expr == current {
            println!("Expression unchanged, converged!");
            break;
        }

        current = new_expr;
    }

    println!(
        "\n=== Final after {} iterations ===",
        max_iterations.min(20)
    );
    println!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: current
        }
    );
}

/// Test just sec^2 - tan^2 (without -1)
#[test]
fn debug_sec_tan_without_minus_one() {
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .try_init();

    println!("\n=== Testing sec²-tan² (no -1) ===\n");

    let mut simplifier = Simplifier::with_default_rules();
    let expr = parse("sec(x)^2 - tan(x)^2", &mut simplifier.context).unwrap();

    println!(
        "Expression: {}",
        DisplayExpr {
            context: &simplifier.context,
            id: expr
        }
    );

    let (result, steps) = simplifier.simplify(expr);

    println!("\nSteps:");
    for (i, step) in steps.iter().enumerate() {
        println!("{}. {} [{}]", i + 1, step.description, step.rule_name);
    }

    println!(
        "\nResult: {}",
        DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    );
}
