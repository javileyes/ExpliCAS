use cas_engine::Simplifier;
use cas_formatter::DisplayExpr;
use cas_parser::parse;

#[test]
fn debug_test_47() {
    let mut simplifier = Simplifier::with_default_rules();
    let input = "asin(x^2 - 1) + acos(x^2 - 1)";
    let expr = parse(input, &mut simplifier.context).unwrap();

    println!("Parsed expr ID: {:?}", expr);
    println!("Parsed expr: {:?}", simplifier.context.get(expr));

    let (result, steps) = simplifier.simplify(expr);

    println!("\nResult ID: {:?}", result);
    println!("Result expr: {:?}", simplifier.context.get(result));
    println!(
        "Result display: {}",
        DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    );

    println!("\n Steps:");
    for (i, step) in steps.iter().enumerate() {
        println!("  {}. {}", i + 1, step.description);
    }
}
