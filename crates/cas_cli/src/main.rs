use cas_engine::Simplifier;
use cas_engine::rules::arithmetic::{AddZeroRule, MulOneRule, CombineConstantsRule};
use std::io::{self, Write};

fn main() {
    let mut simplifier = Simplifier::new();
    simplifier.add_rule(Box::new(AddZeroRule));
    simplifier.add_rule(Box::new(MulOneRule));
    simplifier.add_rule(Box::new(CombineConstantsRule));

    println!("Rust CAS Step-by-Step Demo");
    println!("Enter an expression (e.g., '2 * 3 + 0'):");

    loop {
        print!("> ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        if io::stdin().read_line(&mut input).unwrap() == 0 {
            break;
        }

        let input = input.trim();
        if input.is_empty() {
            continue;
        }

        match cas_parser::parse(input) {
            Ok(expr) => {
                println!("Parsed: {}", expr);
                let (simplified, steps) = simplifier.simplify(expr);
                
                if steps.is_empty() {
                    println!("No simplification steps needed.");
                } else {
                    println!("Steps:");
                    for (i, step) in steps.iter().enumerate() {
                        println!("{}. {}  [{}]", i + 1, step.description, step.rule_name);
                        println!("   -> {}", step.after);
                    }
                }
                println!("Result: {}", simplified);
            }
            Err(e) => println!("Error: {}", e),
        }
    }
}
