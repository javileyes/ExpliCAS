use cas_engine::Simplifier;
use cas_engine::rules::arithmetic::{AddZeroRule, MulOneRule, CombineConstantsRule};
use cas_engine::rules::polynomial::{CombineLikeTermsRule, AnnihilationRule};
use cas_engine::rules::exponents::{ProductPowerRule, PowerPowerRule, ZeroOnePowerRule, EvaluatePowerRule};
use cas_engine::rules::canonicalization::{CanonicalizeRootRule, CanonicalizeNegationRule};
use cas_engine::rules::functions::EvaluateAbsRule;
use cas_engine::rules::trigonometry::{EvaluateTrigRule, PythagoreanIdentityRule};
use cas_engine::rules::logarithms::{EvaluateLogRule, ExponentialLogRule};
use cas_engine::rules::algebra::{SimplifyFractionRule, ExpandRule, FactorRule};
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;

fn main() -> rustyline::Result<()> {
    let mut simplifier = Simplifier::new();
    simplifier.add_rule(Box::new(CanonicalizeNegationRule));
    simplifier.add_rule(Box::new(CanonicalizeRootRule));
    simplifier.add_rule(Box::new(EvaluateAbsRule));
    simplifier.add_rule(Box::new(EvaluateTrigRule));
    simplifier.add_rule(Box::new(PythagoreanIdentityRule));
    simplifier.add_rule(Box::new(EvaluateLogRule));
    simplifier.add_rule(Box::new(ExponentialLogRule));
    simplifier.add_rule(Box::new(SimplifyFractionRule));
    simplifier.add_rule(Box::new(ExpandRule));
    simplifier.add_rule(Box::new(FactorRule));
    simplifier.add_rule(Box::new(EvaluatePowerRule));
    simplifier.add_rule(Box::new(EvaluatePowerRule));
    // simplifier.add_rule(Box::new(DistributeRule)); // Disabled to allow factor() to persist. Use expand() to distribute.
    simplifier.add_rule(Box::new(CombineLikeTermsRule));
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
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                
                rl.add_history_entry(line)?;

                rl.add_history_entry(line)?;

                // Check for "quit" or "exit" command
                if line == "quit" || line == "exit" {
                    println!("Goodbye!");
                    break;
                }

                // Check for "help" command
                if line == "help" {
                    println!("Rust CAS Commands:");
                    println!("  <expr>                  Evaluate and simplify an expression");
                    println!("  subst <expr> <var>=<val> Substitute a variable and simplify");
                    println!("  expand <expr>           Expand polynomials");
                    println!("  factor <expr>           Factor polynomials");
                    println!("  help                    Show this help message");
                    println!("  quit / exit             Exit the REPL");
                    println!();
                    println!("Examples:");
                    println!("  2 + 3 * 4");
                    println!("  x + x");
                    println!("  sqrt(27, 3)");
                    println!("  ln(e^2)");
                    println!("  expand((x+1)^2)");
                    println!("  factor(2*x^2 + 4*x)");
                    println!("  subst x+1 x=2");
                    continue;
                }

                // Check for "subst" command
                if line.starts_with("subst ") {
                    // Format: subst <expr> <var>=<val>
                    // Example: subst x+1 x=2
                    // We need to split by space, but expr might contain spaces.
                    // Let's assume the LAST argument is the assignment.
                    if let Some((expr_str, assign_str)) = line[6..].rsplit_once(' ') {
                        if let Some((var, val_str)) = assign_str.split_once('=') {
                            let var = var.trim();
                            let val_str = val_str.trim();
                            
                            match cas_parser::parse(expr_str) {
                                Ok(expr) => {
                                    match cas_parser::parse(val_str) {
                                        Ok(val_expr) => {
                                            println!("Substituting {} = {} into {}", var, val_str, expr_str);
                                            let subbed = expr.substitute(var, &val_expr);
                                            println!("After substitution: {}", subbed);
                                            
                                            let (result, steps) = simplifier.simplify(subbed);
                                            println!("Steps:");
                                            for (i, step) in steps.iter().enumerate() {
                                                println!("{}. {}  [{}]", i + 1, step.description, step.rule_name);
                                                println!("   -> {}", step.after);
                                            }
                                            println!("Result: {}", result);
                                        },
                                        Err(e) => println!("Error parsing value: {:?}", e),
                                    }
                                },
                                Err(e) => println!("Error parsing expression: {:?}", e),
                            }
                            continue;
                        }
                    }
                    println!("Usage: subst <expression> <var>=<value>");
                    continue;
                }

                match cas_parser::parse(line) {
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
