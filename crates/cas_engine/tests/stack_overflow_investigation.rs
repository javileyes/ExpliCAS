// Test to identify stack overflow causing expressions
// Run with: cargo test -p cas_engine --test stack_overflow_investigation -- --nocapture

use cas_engine::Simplifier;
use cas_parser::parse;

/// Simple LCG for deterministic random generation
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.0
    }

    fn pick(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }
}

/// Generate a random expression with given depth
fn gen_expr(vars: &[&str], depth: usize, rng: &mut Lcg) -> String {
    if depth == 0 || rng.pick(10) < 2 {
        // Terminal: variable or small constant
        if vars.is_empty() || rng.pick(3) == 0 {
            let constants = [-3, -2, -1, 0, 1, 2, 3];
            let c = constants[rng.pick(7) as usize];
            c.to_string()
        } else {
            vars[rng.pick(vars.len() as u64) as usize].to_string()
        }
    } else {
        let op = rng.pick(9);
        match op {
            0 | 1 => format!(
                "({}) + ({})",
                gen_expr(vars, depth - 1, rng),
                gen_expr(vars, depth - 1, rng)
            ),
            2 | 3 => format!(
                "({}) - ({})",
                gen_expr(vars, depth - 1, rng),
                gen_expr(vars, depth - 1, rng)
            ),
            4 | 5 => format!(
                "({}) * ({})",
                gen_expr(vars, depth - 1, rng),
                gen_expr(vars, depth - 1, rng)
            ),
            6 => {
                let base = gen_expr(vars, depth - 1, rng);
                let exp = [1, 2, 3, 4][rng.pick(4) as usize];
                format!("({})^({})", base, exp)
            }
            7 => format!("sin({})", gen_expr(vars, depth - 1, rng)),
            8 => format!("cos({})", gen_expr(vars, depth - 1, rng)),
            _ => unreachable!(),
        }
    }
}

#[test]
#[ignore]
fn investigate_stack_overflow() {
    let seed = 0xC0FFEE_u64;
    let vars = &["x"];

    // Test with increasing depth to find where it breaks
    for depth in 3..=6 {
        println!("\n=== Testing depth {} ===", depth);

        for i in 0..50 {
            // Reset RNG to specific position for reproducibility
            let expr_seed = seed.wrapping_add(i * 1000);
            let mut rng = Lcg::new(expr_seed);

            let e = gen_expr(vars, depth, &mut rng);
            let base_expr = format!("(sin(x)^2 + cos(x)^2) + ({})", e);

            println!(
                "  [depth={}, iter={}] Testing: {} chars",
                depth,
                i,
                base_expr.len()
            );

            let mut simplifier = Simplifier::with_default_rules();
            match parse(&base_expr, &mut simplifier.context) {
                Ok(expr) => {
                    // Print BEFORE simplify so we can see which expression hangs
                    eprintln!(
                        "    About to simplify: {}",
                        &base_expr[0..base_expr.len().min(100)]
                    );
                    use std::io::Write;
                    std::io::stderr().flush().unwrap();

                    // Try to simplify with a timeout/recursion guard
                    let start = std::time::Instant::now();
                    let (_result, steps) = simplifier.simplify(expr);
                    let elapsed = start.elapsed();

                    if elapsed.as_millis() > 500 {
                        println!("    ⚠️  SLOW! Took {:?}, {} steps", elapsed, steps.len());
                        println!("    Expression: {}", base_expr);
                    } else if steps.len() > 100 {
                        println!("    ⚠️  MANY STEPS! {} steps in {:?}", steps.len(), elapsed);
                        println!("    Expression: {}", base_expr);
                    }
                }
                Err(e) => {
                    println!("    Parse error: {:?}", e);
                }
            }
        }
    }

    println!("\n=== Investigation complete ===");
}

#[test]
fn test_specific_problematic_case() {
    // Test a known problematic pattern: deeply nested sin/cos
    let cases = [
        "cos(cos(cos(cos(cos(x)))))",
        "sin(sin(sin(sin(sin(x)))))",
        "cos(sin(cos(sin(cos(x)))))",
        "(cos(x) + sin(x))^4",
        "(sin(x)^2 + cos(x)^2 + x)^3",
    ];

    for case in cases {
        println!("\nTesting: {}", case);
        let mut simplifier = Simplifier::with_default_rules();
        match parse(case, &mut simplifier.context) {
            Ok(expr) => {
                let start = std::time::Instant::now();
                let (result, steps) = simplifier.simplify(expr);
                let elapsed = start.elapsed();
                println!("  Steps: {}, Time: {:?}", steps.len(), elapsed);
                println!(
                    "  Result: {}",
                    cas_formatter::DisplayExpr {
                        context: &simplifier.context,
                        id: result
                    }
                );
            }
            Err(e) => println!("  Parse error: {:?}", e),
        }
    }
}
