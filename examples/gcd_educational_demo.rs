use cas_ast::Context;
use cas_engine::rules::number_theory;
use cas_parser::parse;
use num_bigint::BigInt;

// Demo program showing how to use the educational GCD mode
fn main() {
    println!("=== Educational GCD Mode Demo ===\n");

    // Demo 1: Integer GCD with explanation
    println!("Example 1: Integer GCD (48, 18) with educational explanation:\n");

    let (gcd_result, steps) =
        number_theory::verbose_integer_gcd(BigInt::from(48), BigInt::from(18));

    for step in &steps {
        println!("{}", step);
    }
    println!("\nResult: {}\n", gcd_result);
    println!("=".repeat(60));

    // Demo 2: Another integer example
    println!("\nExample 2: Integer GCD (252, 105) with educational explanation:\n");

    let (gcd_result2, steps2) =
        number_theory::verbose_integer_gcd(BigInt::from(252), BigInt::from(105));

    for step in &steps2 {
        println!("{}", step);
    }
    println!("\nResult: {}\n", gcd_result2);
    println!("=".repeat(60));

    // Demo 3: Polynomial GCD (if integrated into main API)
    println!("\nExample 3: Polynomial GCD");
    println!("To use polynomial GCD in educational mode, call:");
    println!("  let result = compute_gcd(ctx, poly1_expr, poly2_expr, true);");
    println!("  for step in result.steps {{ println!(\"{{}}\" step); }}");
    println!("\nExample polynomials: gcd(x² - 1, x² + 2x + 1) = x + 1");
}
