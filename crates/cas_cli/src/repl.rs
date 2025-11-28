use cas_engine::Simplifier;
use cas_engine::rules::arithmetic::{AddZeroRule, MulOneRule, MulZeroRule, CombineConstantsRule};
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
use cas_ast::DisplayExpr;


use crate::completer::CasHelper;
use crate::config::CasConfig;
use rustyline::config::Configurer;

pub struct Repl {
    simplifier: Simplifier,
    show_steps: bool,
    config: CasConfig,
}

impl Repl {
    pub fn new() -> Self {
        let config = CasConfig::load();
        let mut simplifier = Simplifier::new();
        
        // Always enabled core rules
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
        simplifier.add_rule(Box::new(cas_engine::rules::logarithms::SplitLogExponentsRule));
        
        // Advanced Algebra Rules (Critical for Solver)
        simplifier.add_rule(Box::new(cas_engine::rules::algebra::NestedFractionRule));
        simplifier.add_rule(Box::new(cas_engine::rules::algebra::AddFractionsRule));
        simplifier.add_rule(Box::new(cas_engine::rules::algebra::SimplifyMulDivRule));
        simplifier.add_rule(Box::new(cas_engine::rules::algebra::RationalizeDenominatorRule));
        simplifier.add_rule(Box::new(cas_engine::rules::algebra::CancelCommonFactorsRule));
        
        // Configurable rules
        if config.distribute {
            simplifier.add_rule(Box::new(cas_engine::rules::polynomial::DistributeRule));
        }
        
        if config.distribute_constants {
            simplifier.add_rule(Box::new(cas_engine::rules::polynomial::DistributeConstantRule));
        }
        
        if config.expand_binomials {
            simplifier.add_rule(Box::new(cas_engine::rules::polynomial::BinomialExpansionRule));
        }
        
        if config.factor_difference_squares {
            simplifier.add_rule(Box::new(cas_engine::rules::algebra::FactorDifferenceSquaresRule));
        }

        simplifier.add_rule(Box::new(CombineLikeTermsRule));
        simplifier.add_rule(Box::new(CombineLikeTermsRule));
        simplifier.add_rule(Box::new(AnnihilationRule));
        simplifier.add_rule(Box::new(ProductPowerRule));
        simplifier.add_rule(Box::new(PowerPowerRule));
        simplifier.add_rule(Box::new(ZeroOnePowerRule));
        simplifier.add_rule(Box::new(AddZeroRule));
        simplifier.add_rule(Box::new(MulOneRule));
        simplifier.add_rule(Box::new(MulZeroRule));
        simplifier.add_rule(Box::new(CombineConstantsRule));
        simplifier.add_rule(Box::new(IntegrateRule));

        Self {
            simplifier,
            show_steps: true,
            config,
        }
    }


    // ... (new method remains same)

    pub fn run(&mut self) -> rustyline::Result<()> {
        println!("Rust CAS Step-by-Step Demo");
        println!("Enter an expression (e.g., '2 * 3 + 0'):");

        let helper = CasHelper::new();
        let mut rl = rustyline::Editor::<CasHelper, rustyline::history::DefaultHistory>::new()?;
        rl.set_helper(Some(helper));
        rl.set_completion_type(rustyline::CompletionType::List);

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

                    // Check for "quit" or "exit" command
                    if line == "quit" || line == "exit" {
                        println!("Goodbye!");
                        break;
                    }

                    // Check for "help" command
                    if line.starts_with("help") {
                        self.handle_help(line);
                        continue;
                    }

                    // Check for "steps" command
                    if line == "steps on" {
                        self.show_steps = true;
                        self.simplifier.collect_steps = true;
                        println!("Step-by-step output enabled.");
                        continue;
                    } else if line == "steps off" {
                        self.show_steps = false;
                        self.simplifier.collect_steps = false;
                        println!("Step-by-step output disabled.");
                        continue;
                    }

                    // Check for "help" command
                    if line == "help" {
                        self.print_general_help();
                        continue;
                    }

                    // Check for "equiv" command
                    if line.starts_with("equiv ") {
                        self.handle_equiv(line);
                        continue;
                    }

                    // Check for "subst" command
                    if line.starts_with("subst ") {
                        self.handle_subst(line);
                        continue;
                    }

                    // Check for "solve" command
                    if line.starts_with("solve ") {
                        self.handle_solve(line);
                        continue;
                    }

                    // Check for "simplify" command
                    if line.starts_with("simplify ") {
                        self.handle_full_simplify(line);
                        continue;
                    }

                    // Check for "config" command
                    if line.starts_with("config ") {
                        self.handle_config(line);
                        continue;
                    }

                    self.handle_eval(line);
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

    fn handle_config(&mut self, line: &str) {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 2 {
            println!("Usage: config <list|enable|disable|save|restore> [rule]");
            return;
        }

        match parts[1] {
            "list" => {
                println!("Current Configuration:");
                println!("  distribute: {}", self.config.distribute);
                println!("  expand_binomials: {}", self.config.expand_binomials);
                println!("  distribute_constants: {}", self.config.distribute_constants);
                println!("  factor_difference_squares: {}", self.config.factor_difference_squares);
            },
            "save" => {
                match self.config.save() {
                    Ok(_) => println!("Configuration saved to cas_config.toml"),
                    Err(e) => println!("Error saving configuration: {}", e),
                }
            },
            "restore" => {
                self.config = CasConfig::restore();
                println!("Configuration restored to defaults. Restart CLI to apply changes fully.");
                // We should probably reload rules here, but Repl::new() logic is complex to extract.
                // For now, just warn user.
            },
            "enable" | "disable" => {
                if parts.len() < 3 {
                    println!("Usage: config {} <rule>", parts[1]);
                    return;
                }
                let rule = parts[2];
                let enable = parts[1] == "enable";
                
                let mut changed = true;
                match rule {
                    "distribute" => self.config.distribute = enable,
                    "expand_binomials" => self.config.expand_binomials = enable,
                    "distribute_constants" => self.config.distribute_constants = enable,
                    "factor_difference_squares" => self.config.factor_difference_squares = enable,
                    _ => {
                        println!("Unknown rule: {}", rule);
                        changed = false;
                    }
                }
                
                if changed {
                    println!("Rule '{}' set to {}. Restart CLI to apply changes.", rule, enable);
                }
            },
            _ => println!("Unknown config command: {}", parts[1]),
        }
    }

    fn handle_help(&self, line: &str) {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 2 {
            self.print_general_help();
            return;
        }

        match parts[1] {
            "simplify" => {
                println!("Command: simplify <expr>");
                println!("Description: Simplifies an expression using the full power of the engine.");
                println!("             This includes aggressive distribution and other rules that may");
                println!("             undo factorizations, but guarantee maximum simplification.");
                println!("Example: simplify (x+1)*(x-1) -> x^2 - 1");
            },
            "config" => {
                println!("Command: config <subcommand> [args]");
                println!("Description: Manages CLI configuration.");
                println!("Subcommands:");
                println!("  list             Show current configuration");
                println!("  enable <rule>    Enable a simplification rule");
                println!("  disable <rule>   Disable a simplification rule");
                println!("  save             Save configuration to file");
                println!("  restore          Restore default configuration");
                println!("Rules:");
                println!("  distribute       Aggressive distribution (a*(b+c) -> a*b + a*c)");
                println!("  distribute_constants Safe distribution (-1*(x+y) -> -x-y)");
                println!("  expand_binomials Expand powers ((a+b)^2 -> a^2+2ab+b^2)");
                println!("  factor_difference_squares Factor difference of squares (a^2-b^2 -> (a-b)(a+b))");
            },
            "subst" => {
                println!("Command: subst <expr>, <var>=<val>");
                println!("Description: Substitutes a variable with a value (or another expression) and simplifies.");
                println!("Example: subst x^2 + x, x=3 -> 12");
            },
            "expand" => {
                println!("Command: expand <expr>");
                println!("Description: Expands polynomials and products.");
                println!("Example: expand (x+1)^2 -> x^2 + 2*x + 1");
            },
            "factor" => {
                println!("Command: factor <expr>");
                println!("Description: Factors polynomials.");
                println!("Example: factor x^2 - 1 -> (x - 1) * (x + 1)");
            },
            "collect" => {
                println!("Command: collect <expr>, <var>");
                println!("Description: Groups terms by powers of a variable.");
                println!("Example: collect a*x + b*x + c, x -> (a + b) * x + c");
            },
            "equiv" => {
                println!("Command: equiv <expr1>, <expr2>");
                println!("Description: Checks if two expressions are mathematically equivalent.");
                println!("             Returns true if expr1 - expr2 simplifies to 0.");
            },
            "solve" => {
                println!("Command: solve <equation>, <var>");
                println!("Description: Solves an equation for a variable.");
                println!("Example: solve x + 2 = 5, x -> x = 3");
            },
            "steps" => {
                println!("Command: steps <on|off>");
                println!("Description: Toggles the display of step-by-step simplification details.");
            },
            _ => {
                println!("Unknown command: {}", parts[1]);
                self.print_general_help();
            }
        }
    }

    fn print_general_help(&self) {
        println!("Rust CAS Commands:");
        println!("  <expr>                  Evaluate and simplify an expression");
        println!("  simplify <expr>         Aggressive simplification (full power)");
        println!("  config <subcmd>         Manage configuration (list, enable, disable...)");
        println!("  subst <expr>, <var>=<val> Substitute a variable and simplify");
        println!("  expand <expr>           Expand polynomials");
        println!("  factor <expr>           Factor polynomials");
        println!("  collect <expr>, <var>   Group terms by variable");
        println!("  equiv <e1>, <e2>        Check if two expressions are equivalent");
        println!("  solve <eq>, <var>       Solve equation for variable");
        println!("  steps on/off            Toggle step-by-step output");
        println!("  help [cmd]              Show this help message or details for a command");
        println!("  quit / exit             Exit the REPL");
        println!();
        println!("Type 'help <command>' for more details on a specific command.");
    }

    fn handle_equiv(&mut self, line: &str) {
        let rest = line[6..].trim();
        if let Some((expr1_str, expr2_str)) = rsplit_ignoring_parens(rest, ',') {
             // We need to parse both, but parse takes &mut Context.
             // We can't borrow self.simplifier.context mutably twice.
             // So we parse one, then the other.
             
             let e1_res = cas_parser::parse(expr1_str.trim(), &mut self.simplifier.context);
             match e1_res {
                 Ok(e1) => {
                     let e2_res = cas_parser::parse(expr2_str.trim(), &mut self.simplifier.context);
                     match e2_res {
                         Ok(e2) => {
                             let are_eq = self.simplifier.are_equivalent(e1, e2);
                             if are_eq {
                                 println!("True");
                             } else {
                                 println!("False");
                             }
                         },
                         Err(e) => println!("Error parsing second expression: {}", e),
                     }
                 },
                 Err(e) => println!("Error parsing first expression: {}", e),
             }
        } else {
            println!("Usage: equiv <expr1>, <expr2>");
        }
    }

    fn handle_subst(&mut self, line: &str) {
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
            return;
        };

        if let Some((var, val_str)) = assign_str.split_once('=') {
            let var = var.trim();
            let val_str = val_str.trim();
            
            // Parse expr first
            match cas_parser::parse(expr_str, &mut self.simplifier.context) {
                Ok(expr) => {
                    // Parse val
                    match cas_parser::parse(val_str, &mut self.simplifier.context) {
                        Ok(val_expr) => {
                            if self.show_steps {
                                println!("Substituting {} = {} into {}", var, val_str, expr_str);
                            }
                            // Substitute
                            let target_var = self.simplifier.context.var(var);
                            let subbed = cas_engine::solver::strategies::substitute_expr(&mut self.simplifier.context, expr, target_var, val_expr);
                            
                            if self.show_steps {
                                println!("After substitution: {}", DisplayExpr { context: &self.simplifier.context, id: subbed });
                            }
                            
                            let (result, steps) = self.simplifier.simplify(subbed);
                            if self.show_steps {
                                println!("Steps:");
                                for (i, step) in steps.iter().enumerate() {
                                    println!("{}. {}  [{}]", i + 1, step.description, step.rule_name);
                                    println!("   -> {}", DisplayExpr { context: &self.simplifier.context, id: step.after });
                                }
                            }
                            println!("Result: {}", DisplayExpr { context: &self.simplifier.context, id: result });
                        },
                        Err(e) => println!("Error parsing value: {}", e),
                    }
                },
                Err(e) => println!("Error parsing expression: {}", e),
            }
            return;
        }
        println!("Usage: subst <expression>, <var>=<value>");
    }

    fn handle_solve(&mut self, line: &str) {
        // solve <equation>, <var>
        let rest = line[6..].trim();
        
        // Split by comma or space to get equation and var
        let (eq_str, var) = if let Some((e, v)) = rsplit_ignoring_parens(rest, ',') {
            (e.trim(), v.trim())
        } else {
            // No comma. Try to see if it looks like "eq var"
            // We only accept "eq var" if "eq" is a valid equation.
            // Otherwise, we assume the whole string is the equation (e.g. "ln(x) = a + b")
            if let Some((e, v)) = rsplit_ignoring_parens(rest, ' ') {
                 let v_trim = v.trim();
                 // Check if v is a variable name (alphabetic)
                 if !v_trim.is_empty() && v_trim.chars().all(char::is_alphabetic) {
                     (e.trim(), v_trim)
                 } else {
                     (rest, "x")
                 }
            } else {
                (rest, "x")
            }
        };

        match cas_parser::parse_statement(eq_str, &mut self.simplifier.context) {
            Ok(cas_parser::Statement::Equation(eq)) => {
                // Check if variable exists in equation
                // We should simplify the equation first to handle cases like "ln(x) + ln(x) = 2" -> "2*ln(x) = 2"
                let (sim_lhs, steps_lhs) = self.simplifier.simplify(eq.lhs);
                let (sim_rhs, steps_rhs) = self.simplifier.simplify(eq.rhs);
                
                if self.show_steps && (!steps_lhs.is_empty() || !steps_rhs.is_empty()) {
                    println!("Simplification Steps:");
                    for (i, step) in steps_lhs.iter().enumerate() {
                        println!("LHS {}. {}  [{}]", i + 1, step.description, step.rule_name);
                        println!("   -> {}", DisplayExpr { context: &self.simplifier.context, id: step.after });
                    }
                    for (i, step) in steps_rhs.iter().enumerate() {
                        println!("RHS {}. {}  [{}]", i + 1, step.description, step.rule_name);
                        println!("   -> {}", DisplayExpr { context: &self.simplifier.context, id: step.after });
                    }
                    println!("Solving simplified equation: {} {} {}", DisplayExpr { context: &self.simplifier.context, id: sim_lhs }, eq.op, DisplayExpr { context: &self.simplifier.context, id: sim_rhs });
                }

                let simplified_eq = cas_ast::Equation {
                    lhs: sim_lhs,
                    rhs: sim_rhs,
                    op: eq.op.clone(),
                };

                let lhs_has = cas_engine::solver::contains_var(&self.simplifier.context, simplified_eq.lhs, var);
                let rhs_has = cas_engine::solver::contains_var(&self.simplifier.context, simplified_eq.rhs, var);

                if !lhs_has && !rhs_has {
                    // Constant equation (w.r.t var). Evaluate truthiness.
                    // Already simplified above
                    // We need to compare values.
                    // But sim_lhs and sim_rhs are ExprIds.
                    // We can use are_equivalent? No, that checks symbolic equivalence.
                    // We can use compare_values from solution_set if we expose it?
                    // Or just check if they are same ID?
                    // If simplified, they should be same ID if they are identical.
                    if sim_lhs == sim_rhs {
                        println!("True (Identity)");
                    } else {
                        println!("False (Contradiction)");
                        println!("{} != {}", DisplayExpr { context: &self.simplifier.context, id: sim_lhs }, DisplayExpr { context: &self.simplifier.context, id: sim_rhs });
                    }
                } else {

                    match cas_engine::solver::solve(&simplified_eq, var, &mut self.simplifier) {
                        Ok((solution_set, steps)) => {
                            if self.show_steps {
                                println!("Steps:");
                                for (i, step) in steps.iter().enumerate() {
                                    // Simplify the equation for display
                                    let (sim_lhs, _) = self.simplifier.simplify(step.equation_after.lhs);
                                    let (sim_rhs, _) = self.simplifier.simplify(step.equation_after.rhs);
                                    
                                    println!("{}. {}", i + 1, step.description);
                                    println!("   -> {} {} {}", DisplayExpr { context: &self.simplifier.context, id: sim_lhs }, step.equation_after.op, DisplayExpr { context: &self.simplifier.context, id: sim_rhs });
                                }
                            }
                            // SolutionSet doesn't implement Display with Context.
                            // We need to manually display it.
                            println!("Result: {}", display_solution_set(&self.simplifier.context, &solution_set));
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
    }

    fn handle_eval(&mut self, line: &str) {
        match cas_parser::parse(line, &mut self.simplifier.context) {
            Ok(expr) => {
                println!("Parsed: {}", DisplayExpr { context: &self.simplifier.context, id: expr });
                let (simplified, steps) = self.simplifier.simplify(expr);
                
                if self.show_steps {
                    if steps.is_empty() {
                        println!("No simplification steps needed.");
                    } else {
                        println!("Steps:");
                        for (i, step) in steps.iter().enumerate() {
                            println!("{}. {}  [{}]", i + 1, step.description, step.rule_name);
                            println!("   -> {}", DisplayExpr { context: &self.simplifier.context, id: step.after });
                        }
                    }
                }
                println!("Result: {}", DisplayExpr { context: &self.simplifier.context, id: simplified });
            }
            Err(e) => println!("Error: {}", e),
        }
    }
    fn handle_full_simplify(&mut self, line: &str) {
        // simplify <expr>
        // Uses a temporary simplifier with ALL default rules (including aggressive distribution)
        let expr_str = line[9..].trim();
        
        // We need to use the existing context to parse, but then we want to simplify using a different rule set.
        // The Simplifier struct owns the context.
        // Option 1: Create a new Simplifier, parse into it.
        // Option 2: Swap rules in current simplifier? (Hard)
        // Option 3: Create a new Simplifier, copy context? (Hard)
        
        // Easiest: Create new simplifier, parse string into it. 
        // Note: Variables from previous history won't be available if we don't copy context.
        // But REPL history is just text in rustyline, not context state (unless we implement variable storage).
        // Current implementation: Context is reset per line? No, self.simplifier.context persists.
        // If we want to support "x = 5; simplify x", we need to share context.
        
        // Better approach:
        // 1. Parse expression using current context.
        // 2. Create a temporary Simplifier that SHARES the context? 
        //    Simplifier owns Context. We can't easily share.
        //    But we can temporarily TAKE the context, use it in a new Simplifier, and then put it back.
        
        let mut temp_simplifier = Simplifier::with_default_rules();
        // Swap context
        std::mem::swap(&mut self.simplifier.context, &mut temp_simplifier.context);
        
        // Ensure we have the aggressive rules we want (DistributeRule is in default)
        // Also add DistributeConstantRule just in case (though DistributeRule covers it)
        temp_simplifier.add_rule(Box::new(cas_engine::rules::polynomial::DistributeConstantRule));
        
        // Set steps mode
        temp_simplifier.collect_steps = self.show_steps;
        
        match cas_parser::parse(expr_str, &mut temp_simplifier.context) {
            Ok(expr) => {
                println!("Parsed: {}", DisplayExpr { context: &temp_simplifier.context, id: expr });
                let (simplified, steps) = temp_simplifier.simplify(expr);
                
                if self.show_steps {
                    if steps.is_empty() {
                        println!("No simplification steps needed.");
                    } else {
                        println!("Steps (Aggressive Mode):");
                        for (i, step) in steps.iter().enumerate() {
                            println!("{}. {}  [{}]", i + 1, step.description, step.rule_name);
                            println!("   -> {}", DisplayExpr { context: &temp_simplifier.context, id: step.after });
                        }
                    }
                }
                println!("Result: {}", DisplayExpr { context: &temp_simplifier.context, id: simplified });
            }
            Err(e) => println!("Error: {}", e),
        }
        
        // Swap context back
        std::mem::swap(&mut self.simplifier.context, &mut temp_simplifier.context);
    }
}

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

fn display_solution_set(ctx: &cas_ast::Context, set: &cas_ast::SolutionSet) -> String {
    match set {
        cas_ast::SolutionSet::Empty => "Empty Set".to_string(),
        cas_ast::SolutionSet::AllReals => "All Real Numbers".to_string(),
        cas_ast::SolutionSet::Discrete(exprs) => {
            let s: Vec<String> = exprs.iter().map(|e| format!("{}", DisplayExpr { context: ctx, id: *e })).collect();
            format!("{{ {} }}", s.join(", "))
        },
        cas_ast::SolutionSet::Continuous(interval) => display_interval(ctx, interval),
        cas_ast::SolutionSet::Union(intervals) => {
            let s: Vec<String> = intervals.iter().map(|i| display_interval(ctx, i)).collect();
            s.join(" U ")
        }
    }
}

fn display_interval(ctx: &cas_ast::Context, interval: &cas_ast::Interval) -> String {
    let min_bracket = match interval.min_type {
        cas_ast::BoundType::Open => "(",
        cas_ast::BoundType::Closed => "[",
    };
    let max_bracket = match interval.max_type {
        cas_ast::BoundType::Open => ")",
        cas_ast::BoundType::Closed => "]",
    };
    format!("{}{}, {}{}", min_bracket, DisplayExpr { context: ctx, id: interval.min }, DisplayExpr { context: ctx, id: interval.max }, max_bracket)
}
