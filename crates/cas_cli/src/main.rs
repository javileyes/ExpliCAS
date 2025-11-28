use cas_engine::Simplifier;
use cas_engine::rules::arithmetic::{AddZeroRule, MulOneRule, CombineConstantsRule};
use cas_engine::rules::polynomial::{CombineLikeTermsRule, AnnihilationRule};
use cas_engine::rules::exponents::{ProductPowerRule, PowerPowerRule, ZeroOnePowerRule, EvaluatePowerRule};
use cas_engine::rules::canonicalization::{CanonicalizeRootRule, CanonicalizeNegationRule, CanonicalizeAddRule, CanonicalizeMulRule, AssociativityRule};
use cas_engine::rules::functions::EvaluateAbsRule;
use cas_engine::rules::trigonometry::{EvaluateTrigRule, PythagoreanIdentityRule, AngleIdentityRule, TanToSinCosRule, DoubleAngleRule};
use cas_engine::rules::logarithms::{EvaluateLogRule, ExponentialLogRule};
use cas_engine::rules::algebra::{SimplifyFractionRule, ExpandRule, FactorRule};
use cas_engine::rules::calculus::IntegrateRule;
use cas_engine::rules::grouping::CollectRule;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;

fn main() -> rustyline::Result<()> {
    let mut simplifier = Simplifier::new();
    simplifier.add_rule(Box::new(CanonicalizeNegationRule));
    simplifier.add_rule(Box::new(CanonicalizeAddRule));
    simplifier.add_rule(Box::new(CanonicalizeMulRule));
    simplifier.add_rule(Box::new(CanonicalizeRootRule));
    simplifier.add_rule(Box::new(AssociativityRule));
    simplifier.add_rule(Box::new(EvaluateAbsRule));
    simplifier.add_rule(Box::new(EvaluateTrigRule));
    simplifier.add_rule(Box::new(PythagoreanIdentityRule));
    simplifier.add_rule(Box::new(AngleIdentityRule));
    simplifier.add_rule(Box::new(TanToSinCosRule));
    simplifier.add_rule(Box::new(DoubleAngleRule));
    simplifier.add_rule(Box::new(EvaluateLogRule));
    simplifier.add_rule(Box::new(ExponentialLogRule));
    simplifier.add_rule(Box::new(SimplifyFractionRule));
    simplifier.add_rule(Box::new(ExpandRule));
    simplifier.add_rule(Box::new(FactorRule));
    simplifier.add_rule(Box::new(CollectRule));
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
    simplifier.add_rule(Box::new(IntegrateRule));

    println!("Rust CAS Step-by-Step Demo");
    println!("Enter an expression (e.g., '2 * 3 + 0'):");

    let mut rl = DefaultEditor::new()?;
    // Load history if file exists (optional, skipping for simplicity or can add later)
    
    // Helper to split string by delimiter, ignoring delimiters inside parentheses
    fn rsplit_ignoring_parens(s: &str, delimiter: char) -> Option<(&str, &str)> {
        let mut balance = 0;
        let mut split_idx = None;
        
        for (i, c) in s.char_indices().rev() {
            if c == ')' {
                balance += 1;
            } else if c == '(' {
                balance -= 1;
            } else if c == delimiter && balance == 0 {
                split_idx = Some(i);
                break;
            }
        }
        
        if let Some(idx) = split_idx {
            Some((&s[..idx], &s[idx+1..]))
        } else {
            None
        }
    }

    let mut show_steps = true;

    loop {
        let readline = rl.readline("> ");
        match readline {
            Ok(line) => {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                
                rl.add_history_entry(line)?;

                // Check for "quit" or "exit" command
                if line == "quit" || line == "exit" {
                    println!("Goodbye!");
                    break;
                }

                // Check for "steps" command
                if line == "steps on" {
                    show_steps = true;
                    simplifier.collect_steps = true;
                    println!("Step-by-step output enabled.");
                    continue;
                } else if line == "steps off" {
                    show_steps = false;
                    simplifier.collect_steps = false;
                    println!("Step-by-step output disabled.");
                    continue;
                }

                // Check for "help" command
                if line == "help" {
                    println!("Rust CAS Commands:");
                    println!("  <expr>                  Evaluate and simplify an expression");
                    println!("  subst <expr>, <var>=<val> Substitute a variable and simplify");
                    println!("  expand <expr>           Expand polynomials");
                    println!("  factor <expr>           Factor polynomials");
                    println!("  collect <expr>, <var>   Group terms by variable");
                    println!("  equiv <e1>, <e2>        Check if two expressions are equivalent");
                    println!("  solve <eq>, <var>       Solve equation for variable");
                    println!("  steps on/off            Toggle step-by-step output");
                    println!("  help                    Show this help message");
                    println!("  quit / exit             Exit the REPL");
                    println!();
                    println!("Examples:");
                    println!("  2 + 3 * 4");
                    println!("  sin(pi) + cos(0)");
                    println!("  ln(e^2)");
                    println!("  expand((x+1)^2)");
                    println!("  factor(2*x^2 + 4*x)");
                    println!("  collect(a*x + b*x, x)");
                    println!("  solve x+2=5, x");
                    println!("  solve |x|=5, x");
                    println!("  solve |x|<5, x");
                    println!("  solve -2*x < 10, x");
                    println!("  subst x+1, x=2");
                    continue;
                }

                // Check for "equiv" command
                if line.starts_with("equiv ") {
                    let rest = line[6..].trim();
                    if let Some((expr1_str, expr2_str)) = rsplit_ignoring_parens(rest, ',') {
                         match (cas_parser::parse(expr1_str.trim()), cas_parser::parse(expr2_str.trim())) {
                             (Ok(e1), Ok(e2)) => {
                                 let are_eq = simplifier.are_equivalent(e1, e2);
                                 if are_eq {
                                     println!("True");
                                 } else {
                                     println!("False");
                                 }
                             },
                             (Err(e), _) => println!("Error parsing first expression: {}", e),
                             (_, Err(e)) => println!("Error parsing second expression: {}", e),
                         }
                    } else {
                        println!("Usage: equiv <expr1>, <expr2>");
                    }
                    continue;
                }

                // Check for "subst" command
                if line.starts_with("subst ") {
                    // Format: subst <expr>, <var>=<val>
                    // Example: subst x+1, x=2
                    let rest = line[6..].trim();
                    
                    // Try splitting by comma first (preferred)
                    let (expr_str, assign_str) = if let Some((e, a)) = rsplit_ignoring_parens(rest, ',') {
                        (e.trim(), a.trim())
                    } else if let Some((e, a)) = rsplit_ignoring_parens(rest, ' ') {
                        // Fallback to last space
                        (e.trim(), a.trim())
                    } else {
                        println!("Usage: subst <expression>, <var>=<value>");
                        continue;
                    };

                    if let Some((var, val_str)) = assign_str.split_once('=') {
                        let var = var.trim();
                        let val_str = val_str.trim();
                        
                        match cas_parser::parse(expr_str) {
                            Ok(expr) => {
                                match cas_parser::parse(val_str) {
                                    Ok(val_expr) => {
                                        if show_steps {
                                            println!("Substituting {} = {} into {}", var, val_str, expr_str);
                                        }
                                        let subbed = expr.substitute(var, &val_expr);
                                        if show_steps {
                                            println!("After substitution: {}", subbed);
                                        }
                                        
                                        let (result, steps) = simplifier.simplify(subbed);
                                        if show_steps {
                                            println!("Steps:");
                                            for (i, step) in steps.iter().enumerate() {
                                                println!("{}. {}  [{}]", i + 1, step.description, step.rule_name);
                                                println!("   -> {}", step.after);
                                            }
                                        }
                                        println!("Result: {}", result);
                                    },
                                    Err(e) => println!("Error parsing value: {}", e),
                                }
                            },
                            Err(e) => println!("Error parsing expression: {}", e),
                        }
                        continue;
                    }
                    println!("Usage: subst <expression>, <var>=<value>");
                    continue;
                }

                // Check for "solve" command
                if line.starts_with("solve ") {
                    // solve <equation>, <var>
                    let rest = line[6..].trim();
                    
                    // Split by comma or space to get equation and var
                    let (eq_str, var) = if let Some((e, v)) = rsplit_ignoring_parens(rest, ',') {
                        (e.trim(), v.trim())
                    } else {
                        // No comma. Try to see if it looks like "eq var"
                        // We only accept "eq var" if "eq" is a valid equation.
                        // Otherwise, we assume the whole string is the equation (e.g. "ln(x) = a + b")
                        let try_split = rsplit_ignoring_parens(rest, ' ');
                        let mut use_split = false;
                        
                        if let Some((e, v)) = try_split {
                             let v = v.trim();
                             // Check if v is a variable name (alphabetic)
                             if !v.is_empty() && v.chars().all(char::is_alphabetic) {
                                 // Check if e is a valid equation
                                 // We need to suppress output or errors? parse_statement just returns Result.
                                 if let Ok(cas_parser::Statement::Equation(_)) = cas_parser::parse_statement(e.trim()) {
                                     use_split = true;
                                 }
                             }
                        }
                        
                        if use_split {
                            let (e, v) = try_split.unwrap();
                            (e.trim(), v.trim())
                        } else {
                            (rest, "x")
                        }
                    };

                    match cas_parser::parse_statement(eq_str) {
                        Ok(cas_parser::Statement::Equation(eq)) => {
                            // Check if variable exists in equation
                            // We should simplify the equation first to handle cases like "ln(x) + ln(x) = 2" -> "2*ln(x) = 2"
                            let (sim_lhs, steps_lhs) = simplifier.simplify(eq.lhs.clone());
                            let (sim_rhs, steps_rhs) = simplifier.simplify(eq.rhs.clone());
                            
                            if show_steps && (!steps_lhs.is_empty() || !steps_rhs.is_empty()) {
                                println!("Simplification Steps:");
                                for (i, step) in steps_lhs.iter().enumerate() {
                                    println!("LHS {}. {}  [{}]", i + 1, step.description, step.rule_name);
                                    println!("   -> {}", step.after);
                                }
                                for (i, step) in steps_rhs.iter().enumerate() {
                                    println!("RHS {}. {}  [{}]", i + 1, step.description, step.rule_name);
                                    println!("   -> {}", step.after);
                                }
                                println!("Solving simplified equation: {} {} {}", sim_lhs, eq.op, sim_rhs);
                            }

                            let simplified_eq = cas_ast::Equation {
                                lhs: sim_lhs.clone(),
                                rhs: sim_rhs.clone(),
                                op: eq.op.clone(),
                            };

                            let lhs_has = cas_engine::solver::contains_var(&simplified_eq.lhs, var);
                            let rhs_has = cas_engine::solver::contains_var(&simplified_eq.rhs, var);

                            if !lhs_has && !rhs_has {
                                // Constant equation (w.r.t var). Evaluate truthiness.
                                // Already simplified above
                                if sim_lhs == sim_rhs {
                                    println!("True (Identity)");
                                } else {
                                    println!("False (Contradiction)");
                                    println!("{} != {}", sim_lhs, sim_rhs);
                                }
                            } else {

                                match cas_engine::solver::solve(&simplified_eq, var, &simplifier) {
                                    Ok((solution_set, steps)) => {
                                        if show_steps {
                                            println!("Steps:");
                                            for (i, step) in steps.iter().enumerate() {
                                                // Simplify the equation for display
                                                let (sim_lhs, _) = simplifier.simplify(step.equation_after.lhs.clone());
                                                let (sim_rhs, _) = simplifier.simplify(step.equation_after.rhs.clone());
                                                
                                                println!("{}. {}", i + 1, step.description);
                                                println!("   -> {} {} {}", sim_lhs, step.equation_after.op, sim_rhs);
                                            }
                                        }
                                        println!("Result: {}", solution_set);
                                    },
                                    Err(e) => println!("Error solving: {}", e),
                                }
                            }
                        },
                        Ok(cas_parser::Statement::Expression(_)) => {
                            println!("Error: Expected an equation, got an expression.");
                        },
                        Err(e) => println!("Error parsing equation: {}", e),
                    }
                    continue;
                }

                match cas_parser::parse(line) {
                    Ok(expr) => {
                        println!("Parsed: {}", expr);
                        let (simplified, steps) = simplifier.simplify(expr);
                        
                        if show_steps {
                            if steps.is_empty() {
                                println!("No simplification steps needed.");
                            } else {
                                println!("Steps:");
                                for (i, step) in steps.iter().enumerate() {
                                    println!("{}. {}  [{}]", i + 1, step.description, step.rule_name);
                                    println!("   -> {}", step.after);
                                }
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
