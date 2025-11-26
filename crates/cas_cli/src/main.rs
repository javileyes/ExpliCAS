use cas_engine::Simplifier;
use cas_engine::rules::arithmetic::{AddZeroRule, MulOneRule, CombineConstantsRule};
use cas_engine::rules::polynomial::{DistributeRule, CombineLikeTermsRule, AnnihilationRule};
use cas_engine::rules::exponents::{ProductPowerRule, PowerPowerRule, ZeroOnePowerRule};
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;

fn main() -> rustyline::Result<()> {
    let mut simplifier = Simplifier::new();
    simplifier.add_rule(Box::new(DistributeRule));
    simplifier.add_rule(Box::new(CombineLikeTermsRule));
    simplifier.add_rule(Box::new(AnnihilationRule));
    simplifier.add_rule(Box::new(ProductPowerRule));
    simplifier.add_rule(Box::new(PowerPowerRule));
    simplifier.add_rule(Box::new(ZeroOnePowerRule));
    simplifier.add_rule(Box::new(AddZeroRule));
    simplifier.add_rule(Box::new(MulOneRule));
    simplifier.add_rule(Box::new(CombineConstantsRule));

    println!("Rust CAS Step-by-Step Demo");
    println!("Enter an expression (e.g., '2 * 3 + 0'):");

    let mut rl = DefaultEditor::new()?;
    // Load history if file exists (optional, skipping for simplicity or can add later)
    
    loop {
        let readline = rl.readline("> ");
        match readline {
            Ok(line) => {
                let input = line.trim();
                if input.is_empty() {
                    continue;
                }
                
                rl.add_history_entry(input)?;

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
            },
            Err(ReadlineError::Interrupted) => {
                println!("CTRL-C");
                break;
            },
            Err(ReadlineError::Eof) => {
                println!("CTRL-D");
                break;
            },
            Err(err) => {
                println!("Error: {:?}", err);
                break;
            }
        }
    }
    Ok(())
}
