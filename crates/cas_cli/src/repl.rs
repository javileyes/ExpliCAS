use cas_engine::Simplifier;
use cas_engine::rules::arithmetic::{AddZeroRule, MulOneRule, MulZeroRule, CombineConstantsRule};
use cas_engine::rules::polynomial::{CombineLikeTermsRule, AnnihilationRule};
use cas_engine::rules::exponents::{ProductPowerRule, PowerPowerRule, EvaluatePowerRule, IdentityPowerRule, PowerProductRule, PowerQuotientRule};


use cas_engine::rules::trigonometry::{EvaluateTrigRule, PythagoreanIdentityRule, AngleIdentityRule, TanToSinCosRule, DoubleAngleRule};
use cas_engine::rules::logarithms::{EvaluateLogRule, ExponentialLogRule};
use cas_engine::rules::algebra::{SimplifyFractionRule, ExpandRule, FactorRule};
use cas_engine::rules::calculus::{IntegrateRule, DiffRule};
use cas_engine::rules::number_theory::NumberTheoryRule;
use cas_engine::rules::grouping::CollectRule;
use rustyline::error::ReadlineError;
use cas_ast::{Context, Expr, DisplayExpr, ExprId};
use cas_engine::step::PathStep;


use crate::completer::CasHelper;
use crate::config::CasConfig;
use rustyline::config::Configurer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verbosity {
    None,
    Low,
    Normal,
    Verbose,
}

pub struct Repl {
    simplifier: Simplifier,
    verbosity: Verbosity,
    config: CasConfig,
}

fn reconstruct_global_expr(context: &mut Context, root: ExprId, path: &[PathStep], replacement: ExprId) -> ExprId {
    if path.is_empty() {
        return replacement;
    }

    let current_step = &path[0];
    let remaining_path = &path[1..];
    let expr = context.get(root).clone();

    match (expr, current_step) {
        (Expr::Add(l, r), PathStep::Left) => {
            let new_l = reconstruct_global_expr(context, l, remaining_path, replacement);
            context.add(Expr::Add(new_l, r))
        },
        (Expr::Add(l, r), PathStep::Right) => {
            let new_r = reconstruct_global_expr(context, r, remaining_path, replacement);
            context.add(Expr::Add(l, new_r))
        },
        (Expr::Sub(l, r), PathStep::Left) => {
            let new_l = reconstruct_global_expr(context, l, remaining_path, replacement);
            context.add(Expr::Sub(new_l, r))
        },
        (Expr::Sub(l, r), PathStep::Right) => {
            let new_r = reconstruct_global_expr(context, r, remaining_path, replacement);
            context.add(Expr::Sub(l, new_r))
        },
        (Expr::Mul(l, r), PathStep::Left) => {
            let new_l = reconstruct_global_expr(context, l, remaining_path, replacement);
            context.add(Expr::Mul(new_l, r))
        },
        (Expr::Mul(l, r), PathStep::Right) => {
            let new_r = reconstruct_global_expr(context, r, remaining_path, replacement);
            context.add(Expr::Mul(l, new_r))
        },
        (Expr::Div(l, r), PathStep::Left) => {
            let new_l = reconstruct_global_expr(context, l, remaining_path, replacement);
            context.add(Expr::Div(new_l, r))
        },
        (Expr::Div(l, r), PathStep::Right) => {
            let new_r = reconstruct_global_expr(context, r, remaining_path, replacement);
            context.add(Expr::Div(l, new_r))
        },
        (Expr::Pow(b, e), PathStep::Base) => {
            let new_b = reconstruct_global_expr(context, b, remaining_path, replacement);
            context.add(Expr::Pow(new_b, e))
        },
        (Expr::Pow(b, e), PathStep::Exponent) => {
            let new_e = reconstruct_global_expr(context, e, remaining_path, replacement);
            context.add(Expr::Pow(b, new_e))
        },
        (Expr::Neg(e), PathStep::Inner) => {
            let new_e = reconstruct_global_expr(context, e, remaining_path, replacement);
            context.add(Expr::Neg(new_e))
        },
        (Expr::Function(name, args), PathStep::Arg(idx)) => {
            let mut new_args = args.clone();
            if *idx < new_args.len() {
                new_args[*idx] = reconstruct_global_expr(context, new_args[*idx], remaining_path, replacement);
                context.add(Expr::Function(name, new_args))
            } else {
                root // Should not happen if path is valid
            }
        },
        _ => root, // Path mismatch or invalid structure
    }
}


fn should_show_step(step: &cas_engine::step::Step, verbosity: Verbosity) -> bool {
    match verbosity {
        Verbosity::None => false,
        Verbosity::Verbose => true,
        Verbosity::Low | Verbosity::Normal => {
            // Filter out "noise" rules
            let name = &step.rule_name;
            if name.starts_with("Canonicalize") || 
               name.starts_with("Sort") || 
               name == "Collect" || 
               name.starts_with("Identity") ||
               name == "Add Zero" ||
               name == "Multiply by One" {
                false
            } else {
                true
            }
        }
    }
}

impl Repl {
    pub fn new() -> Self {
        let config = CasConfig::load();
        let mut simplifier = Simplifier::with_default_rules();

        
        // Always enabled core rules
        simplifier.add_rule(Box::new(cas_engine::rules::functions::AbsSquaredRule));
        simplifier.add_rule(Box::new(EvaluateTrigRule));
        simplifier.add_rule(Box::new(PythagoreanIdentityRule));
        if config.trig_angle_sum {
            simplifier.add_rule(Box::new(AngleIdentityRule));
        }
        simplifier.add_rule(Box::new(TanToSinCosRule));
        if config.trig_double_angle {
            simplifier.add_rule(Box::new(DoubleAngleRule));
        }
        if config.canonicalize_trig_square {
            simplifier.add_rule(Box::new(cas_engine::rules::trigonometry::CanonicalizeTrigSquareRule));
        }
        simplifier.add_rule(Box::new(EvaluateLogRule));
        simplifier.add_rule(Box::new(ExponentialLogRule));
        simplifier.add_rule(Box::new(SimplifyFractionRule));
        simplifier.add_rule(Box::new(ExpandRule));
        simplifier.add_rule(Box::new(cas_engine::rules::algebra::ConservativeExpandRule));
        simplifier.add_rule(Box::new(FactorRule));
        simplifier.add_rule(Box::new(CollectRule));
        simplifier.add_rule(Box::new(EvaluatePowerRule));
        simplifier.add_rule(Box::new(EvaluatePowerRule));
        if config.log_split_exponents {
            simplifier.add_rule(Box::new(cas_engine::rules::logarithms::SplitLogExponentsRule));
        }
        
        // Advanced Algebra Rules (Critical for Solver)
        simplifier.add_rule(Box::new(cas_engine::rules::algebra::NestedFractionRule));
        simplifier.add_rule(Box::new(cas_engine::rules::algebra::AddFractionsRule));
        simplifier.add_rule(Box::new(cas_engine::rules::algebra::SimplifyMulDivRule));
        if config.rationalize_denominator {
            simplifier.add_rule(Box::new(cas_engine::rules::algebra::RationalizeDenominatorRule));
        }
        simplifier.add_rule(Box::new(cas_engine::rules::algebra::CancelCommonFactorsRule));
        
        // Configurable rules
        if config.distribute {
            simplifier.add_rule(Box::new(cas_engine::rules::polynomial::DistributeRule));
        }
        
        if config.distribute_constants {
    
        }
        
        if config.expand_binomials {
            simplifier.add_rule(Box::new(cas_engine::rules::polynomial::BinomialExpansionRule));
        }
        
        if config.factor_difference_squares {
            simplifier.add_rule(Box::new(cas_engine::rules::algebra::FactorDifferenceSquaresRule));
        }

        if config.root_denesting {
            simplifier.add_rule(Box::new(cas_engine::rules::algebra::RootDenestingRule));
        }

        if config.auto_factor {
            simplifier.add_rule(Box::new(cas_engine::rules::algebra::AutomaticFactorRule));
        }

        simplifier.add_rule(Box::new(cas_engine::rules::trigonometry::AngleConsistencyRule));
        simplifier.add_rule(Box::new(CombineLikeTermsRule));
        simplifier.add_rule(Box::new(CombineLikeTermsRule));
        simplifier.add_rule(Box::new(AnnihilationRule));
        simplifier.add_rule(Box::new(ProductPowerRule));
        simplifier.add_rule(Box::new(PowerPowerRule));
        simplifier.add_rule(Box::new(PowerProductRule));
        simplifier.add_rule(Box::new(PowerQuotientRule));
        simplifier.add_rule(Box::new(IdentityPowerRule));
        simplifier.add_rule(Box::new(cas_engine::rules::exponents::NegativeBasePowerRule));
        simplifier.add_rule(Box::new(AddZeroRule));
        simplifier.add_rule(Box::new(MulOneRule));
        simplifier.add_rule(Box::new(MulZeroRule));
        simplifier.add_rule(Box::new(cas_engine::rules::arithmetic::DivZeroRule));
        simplifier.add_rule(Box::new(CombineConstantsRule));
        simplifier.add_rule(Box::new(IntegrateRule));
        simplifier.add_rule(Box::new(DiffRule));
        simplifier.add_rule(Box::new(NumberTheoryRule));

        let mut repl = Self {
            simplifier,
            verbosity: Verbosity::Normal,
            config,
        };
        repl.sync_config_to_simplifier();
        repl
    }


    fn sync_config_to_simplifier(&mut self) {
        let config = &self.config;
        
        // Helper to toggle rule
        let mut toggle = |name: &str, enabled: bool| {
            if enabled {
                self.simplifier.enable_rule(name);
            } else {
                self.simplifier.disable_rule(name);
            }
        };

        toggle("Distributive Property", config.distribute);
        toggle("Binomial Expansion", config.expand_binomials);
        toggle("Distribute Constant", config.distribute_constants);
        toggle("Factor Difference of Squares", config.factor_difference_squares);
        toggle("Root Denesting", config.root_denesting);
        toggle("Double Angle Identity", config.trig_double_angle);
        toggle("Angle Sum/Diff Identity", config.trig_angle_sum);
        toggle("Split Log Exponents", config.log_split_exponents);
        toggle("Rationalize Denominator", config.rationalize_denominator);
        toggle("Canonicalize Trig Square", config.canonicalize_trig_square);
        toggle("Rationalize Denominator", config.rationalize_denominator);
        toggle("Canonicalize Trig Square", config.canonicalize_trig_square);
        
        // Auto Factor Logic:
        // If auto_factor is on, we enable AutomaticFactorRule AND ConservativeExpandRule.
        // We DISABLE the aggressive ExpandRule to prevent loops.
        if config.auto_factor {
            self.simplifier.enable_rule("Automatic Factorization");
            self.simplifier.enable_rule("Conservative Expand");
            self.simplifier.disable_rule("Expand Polynomial");
            self.simplifier.disable_rule("Binomial Expansion");
        } else {
            self.simplifier.disable_rule("Automatic Factorization");
            self.simplifier.disable_rule("Conservative Expand");
            self.simplifier.enable_rule("Expand Polynomial");
            // Re-enable Binomial Expansion if config says so
            if config.expand_binomials {
                self.simplifier.enable_rule("Binomial Expansion");
            }
        }
    }

    pub fn run(&mut self) -> rustyline::Result<()> {
        println!("Rust CAS Step-by-Step Demo");
        println!("Step-by-step output enabled (Normal).");
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
                    
                    if line == "quit" || line == "exit" {
                        println!("Goodbye!");
                        break;
                    }

                    self.handle_command(line);
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

    /// Converts function-style commands to command-style
    /// Examples:
    ///   simplify(...) -> simplify x^2 + 1
    ///   solve(...) -> solve x + 2 = 5, x
    fn preprocess_function_syntax(&self, line: &str) -> String {
        let line = line.trim();
        
        // Check for simplify(...)
        if line.starts_with("simplify(") && line.ends_with(")") {
            let content = &line["simplify(".len()..line.len()-1];
            return format!("simplify {}", content);
        }
        
        // Check for solve(...)
        if line.starts_with("solve(") && line.ends_with(")") {
            let content = &line["solve(".len()..line.len()-1];
            return format!("solve {}", content);
        }
        
        // Return unchanged
        line.to_string()
    }

    pub fn handle_command(&mut self, line: &str) {
        // Preprocess: Convert function-style commands to command-style
        // simplify(...) -> simplify ...
        // solve(...) -> solve ...
        let line = self.preprocess_function_syntax(line);
        
        // Check for "help" command
        if line.starts_with("help") {
            self.handle_help(&line);
            return;
        }

        // Check for "steps" command
        if line.starts_with("steps ") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                match parts[1] {
                    "on" | "normal" => {
                        self.verbosity = Verbosity::Normal;
                        self.simplifier.collect_steps = true;
                        println!("Step-by-step output enabled (Normal).");
                    },
                    "off" | "none" => {
                        self.verbosity = Verbosity::None;
                        self.simplifier.collect_steps = false;
                        println!("Step-by-step output disabled.");
                    },
                    "verbose" => {
                        self.verbosity = Verbosity::Verbose;
                        self.simplifier.collect_steps = true;
                        println!("Step-by-step output enabled (Verbose).");
                    },
                    "low" => {
                        self.verbosity = Verbosity::Low;
                        self.simplifier.collect_steps = true;
                        println!("Step-by-step output enabled (Low).");
                    },
                    _ => println!("Usage: steps <on|off|normal|verbose|low|none>"),
                }
            } else {
                 println!("Usage: steps <on|off|normal|verbose|low|none>");
            }
            return;
        }

        // Check for "help" command (duplicate check in original code?)
        if line == "help" {
            self.print_general_help();
            return;
        }

        // Check for "equiv" command
        if line.starts_with("equiv ") {
            self.handle_equiv(&line);
            return;
        }

        // Check for "subst" command
        if line.starts_with("subst ") {
            self.handle_subst(&line);
            return;
        }

        // Check for "solve" command
        if line.starts_with("solve ") {
            self.handle_solve(&line);
            return;
        }

        // Check for "simplify" command
        if line.starts_with("simplify ") {
            self.handle_full_simplify(&line);
            return;
        }

        // Check for "config" command
        if line.starts_with("config ") {
            self.handle_config(&line);
            return;
        }

        self.handle_eval(&line);
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
                println!("  root_denesting: {}", self.config.root_denesting);
                println!("  trig_double_angle: {}", self.config.trig_double_angle);
                println!("  trig_angle_sum: {}", self.config.trig_angle_sum);
                println!("  log_split_exponents: {}", self.config.log_split_exponents);
                println!("  rationalize_denominator: {}", self.config.rationalize_denominator);
                println!("  canonicalize_trig_square: {}", self.config.canonicalize_trig_square);
                println!("  auto_factor: {}", self.config.auto_factor);
            },
            "save" => {
                match self.config.save() {
                    Ok(_) => println!("Configuration saved to cas_config.toml"),
                    Err(e) => println!("Error saving configuration: {}", e),
                }
            },
            "restore" => {
                self.config = CasConfig::restore();
                self.sync_config_to_simplifier();
                println!("Configuration restored to defaults.");
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
                    "root_denesting" => self.config.root_denesting = enable,
                    "trig_double_angle" => self.config.trig_double_angle = enable,
                    "trig_angle_sum" => self.config.trig_angle_sum = enable,
                    "log_split_exponents" => self.config.log_split_exponents = enable,
                    "rationalize_denominator" => self.config.rationalize_denominator = enable,
                    "canonicalize_trig_square" => self.config.canonicalize_trig_square = enable,
                    "auto_factor" => self.config.auto_factor = enable,
                    _ => {
                        println!("Unknown rule: {}", rule);
                        changed = false;
                    }
                }
                
                if changed {
                    self.sync_config_to_simplifier();
                    println!("Rule '{}' set to {}.", rule, enable);
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
            "diff" => {
                println!("Command: diff <expr> <var>");
                println!("Description: Computes the symbolic derivative of an expression with respect to a variable.");
                println!("             Supports basic arithmetic, power rule, chain rule, and common functions.");
                println!("Example: diff(sin(x^2), x) -> 2*x*cos(x^2)");
            },
            "gcd" => {
                println!("Function: gcd <a, b>");
                println!("Description: Computes the Greatest Common Divisor of two integers.");
                println!("Example: gcd(12, 18) -> 6");
            },
            "lcm" => {
                println!("Function: lcm <a, b>");
                println!("Description: Computes the Least Common Multiple of two integers.");
                println!("Example: lcm(4, 6) -> 12");
            },
            "mod" => {
                println!("Function: mod <a, n>");
                println!("Description: Computes the remainder of a divided by n (Euclidean modulo).");
                println!("Example: mod(10, 3) -> 1");
            },
            "factors" | "prime_factors" => {
                println!("Function: factors <n>");
                println!("Description: Computes the prime factorization of an integer.");
                println!("Example: factors(12) -> 2^2 * 3");
            },
            "fact" | "factorial" => {
                println!("Function: fact <n> or <n>!");
                println!("Description: Computes the factorial of a non-negative integer.");
                println!("Example: fact(5) -> 120, 5! -> 120");
            },
            "choose" | "nCr" => {
                println!("Function: choose <n, k>");
                println!("Description: Computes the binomial coefficient nCk (combinations).");
                println!("Example: choose(5, 2) -> 10");
            },
            "perm" | "nPr" => {
                println!("Function: perm <n, k>");
                println!("Description: Computes the number of permutations nPk.");
                println!("Example: perm(5, 2) -> 20");
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
                println!("Example: expand(x+1)^2 -> x^2 + 2*x + 1");
            },
            "factor" => {
                println!("Command: factor <expr>");
                println!("Description: Factors polynomials.");
                println!("Example: factor(x^2 - 1) -> (x - 1) * (x + 1)");
            },
            "collect" => {
                println!("Command: collect <expr>, <var>");
                println!("Description: Groups terms by powers of a variable.");
                println!("Example: collect(a*x + b*x + c, x) -> (a + b) * x + c");
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
                println!("Command: steps <level>");
                println!("Description: Controls the verbosity of simplification steps.");
                println!("Levels:");
                println!("  normal (or on)   Show clarifying steps (Global state). Default.");
                println!("  low              Minimal output (Global state sequence only).");
                println!("  verbose          Show all steps (Local + Global details).");
                println!("  none (or off)    Disable step output.");
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
        println!("  diff <expr>, <var>      Compute symbolic derivative");
        println!("  gcd <a, b>              Greatest Common Divisor");
        println!("  lcm <a, b>              Least Common Multiple");
        println!("  mod <a, n>              Modular arithmetic");
        println!("  factors <n>             Prime factorization");
        println!("  fact <n>                Factorial (or n!)");
        println!("  choose <n, k>           Binomial coefficient (nCk)");
        println!("  perm <n, k>             Permutations (nPk)");
        println!("  steps <level>           Set step verbosity (normal, low, verbose, none)");
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

                            if self.verbosity != Verbosity::None {
                                println!("Substituting {} = {} into {}", var, val_str, expr_str);
                            }
                            // Substitute
                            let target_var = self.simplifier.context.var(var);
                            let subbed = cas_engine::solver::strategies::substitute_expr(&mut self.simplifier.context, expr, target_var, val_expr);
                            
                            let (result, steps) = self.simplifier.simplify(subbed);
                            if self.verbosity != Verbosity::None {
                                if self.verbosity != Verbosity::Low {
                                    println!("Steps:");
                                }
                                let mut current_root = subbed;
                                let mut step_count = 0;
                                for step in steps.iter() {
                                    if should_show_step(step, self.verbosity) {
                                        step_count += 1;
                                        
                                        if self.verbosity == Verbosity::Low {
                                            // Low mode: just global state
                                            current_root = reconstruct_global_expr(&mut self.simplifier.context, current_root, &step.path, step.after);
                                            println!("-> {}", DisplayExpr { context: &self.simplifier.context, id: current_root });
                                        } else {
                                            // Normal/Verbose
                                            println!("{}. {}  [{}]", step_count, step.description, step.rule_name);
                                            
                                            if self.verbosity == Verbosity::Verbose || self.verbosity == Verbosity::Normal {
                                                let after_disp = if let Some(s) = &step.after_str {
                                                    s.clone()
                                                } else {
                                                    format!("{}", DisplayExpr { context: &self.simplifier.context, id: step.after })
                                                };
                                                println!("   Local: {} -> {}", 
                                                    DisplayExpr { context: &self.simplifier.context, id: step.before },
                                                    after_disp
                                                );
                                            }
                                            
                                            current_root = reconstruct_global_expr(&mut self.simplifier.context, current_root, &step.path, step.after);
                                            println!("   Global: {}", DisplayExpr { context: &self.simplifier.context, id: current_root });
                                        }
                                    } else {
                                        current_root = reconstruct_global_expr(&mut self.simplifier.context, current_root, &step.path, step.after);
                                    }
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
                
                if self.verbosity != Verbosity::None && (!steps_lhs.is_empty() || !steps_rhs.is_empty()) {
                    if self.verbosity != Verbosity::Low {
                        println!("Simplification Steps:");
                    }
                    for (i, step) in steps_lhs.iter().enumerate() {
                        if should_show_step(step, self.verbosity) {
                            if self.verbosity == Verbosity::Low {
                                // Low mode: just global state? No, for solve simplification we don't track global state easily here
                                // because steps_lhs are local to lhs.
                                // We can show the result of the step on LHS.
                                // But wait, solve simplification is just pre-simplification.
                                // Let's just show it if not Low.
                                // Or if Low, maybe we skip pre-simplification steps display?
                                // User said "Low mode only shows global changes".
                                // For solve, the "Global" is the equation.
                                // But here we simplify LHS and RHS separately.
                                // Let's skip detailed steps in Low mode for pre-simplification, 
                                // and just show the simplified equation.
                            } else {
                                println!("LHS {}. {}  [{}]", i + 1, step.description, step.rule_name);
                                let after_disp = if let Some(s) = &step.after_str {
                                    s.clone()
                                } else {
                                    format!("{}", DisplayExpr { context: &self.simplifier.context, id: step.after })
                                };
                                println!("   Local: {} -> {}", 
                                    DisplayExpr { context: &self.simplifier.context, id: step.before },
                                    after_disp
                                );
                            }
                        }
                    }
                    for (i, step) in steps_rhs.iter().enumerate() {
                         if should_show_step(step, self.verbosity) {
                            if self.verbosity != Verbosity::Low {
                                println!("RHS {}. {}  [{}]", i + 1, step.description, step.rule_name);
                                let after_disp = if let Some(s) = &step.after_str {
                                    s.clone()
                                } else {
                                    format!("{}", DisplayExpr { context: &self.simplifier.context, id: step.after })
                                };
                                println!("   Local: {} -> {}", 
                                    DisplayExpr { context: &self.simplifier.context, id: step.before },
                                    after_disp
                                );
                            }
                        }
                    }
                    if self.verbosity != Verbosity::Low {
                        println!("Solving simplified equation: {} {} {}", DisplayExpr { context: &self.simplifier.context, id: sim_lhs }, eq.op, DisplayExpr { context: &self.simplifier.context, id: sim_rhs });
                    }
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

                    // Pass the ORIGINAL equation to solve, so it can check for domain restrictions (singularities).
                    // If we pass simplified_eq, we lose information about e.g. (x-1) in denominator.
                    match cas_engine::solver::solve(&eq, var, &mut self.simplifier) {
                        Ok((solution_set, steps)) => {
                            if self.verbosity != Verbosity::None {
                                if self.verbosity != Verbosity::Low {
                                    println!("Steps:");
                                }
                                for (i, step) in steps.iter().enumerate() {
                                    // SolveStep is different from Step, so we can't use should_show_step directly.
                                    // For now, just show all steps if verbosity is not None/Low?
                                    // Or implement filtering for SolveStep too?
                                    // SolveStep has description but no rule_name in the same way?
                                    // Let's just show it.
                                    if true {
                                        // Simplify the equation for display
                                        let (sim_lhs, _) = self.simplifier.simplify(step.equation_after.lhs);
                                        let (sim_rhs, _) = self.simplifier.simplify(step.equation_after.rhs);
                                        
                                        if self.verbosity == Verbosity::Low {
                                            println!("-> {} {} {}", DisplayExpr { context: &self.simplifier.context, id: sim_lhs }, step.equation_after.op, DisplayExpr { context: &self.simplifier.context, id: sim_rhs });
                                        } else {
                                            println!("{}. {}", i + 1, step.description);
                                            println!("   -> {} {} {}", DisplayExpr { context: &self.simplifier.context, id: sim_lhs }, step.equation_after.op, DisplayExpr { context: &self.simplifier.context, id: sim_rhs });
                                        }
                                    }
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
                
                if self.verbosity != Verbosity::None {
                    if steps.is_empty() {
                        if self.verbosity != Verbosity::Low {
                             println!("No simplification steps needed.");
                        }
                    } else {
                        if self.verbosity != Verbosity::Low {
                            println!("Steps:");
                        }
                        let mut current_root = expr;
                        let mut step_count = 0;
                        for step in steps.iter() {
                            if should_show_step(step, self.verbosity) {
                                step_count += 1;
                                
                                if self.verbosity == Verbosity::Low {
                                    // Low mode: just global state
                                    current_root = reconstruct_global_expr(&mut self.simplifier.context, current_root, &step.path, step.after);
                                    println!("-> {}", DisplayExpr { context: &self.simplifier.context, id: current_root });
                                } else {
                                    // Normal/Verbose
                                    println!("{}. {}  [{}]", step_count, step.description, step.rule_name);
                                    
                                    if self.verbosity == Verbosity::Verbose || self.verbosity == Verbosity::Normal {
                                        let after_disp = if let Some(s) = &step.after_str {
                                            s.clone()
                                        } else {
                                            format!("{}", DisplayExpr { context: &self.simplifier.context, id: step.after })
                                        };
                                        println!("   Local: {} -> {}", 
                                            DisplayExpr { context: &self.simplifier.context, id: step.before },
                                            after_disp
                                        );
                                    }
                                    
                                    current_root = reconstruct_global_expr(&mut self.simplifier.context, current_root, &step.path, step.after);
                                    println!("   Global: {}", DisplayExpr { context: &self.simplifier.context, id: current_root });
                                }
                            } else {
                                current_root = reconstruct_global_expr(&mut self.simplifier.context, current_root, &step.path, step.after);
                            }
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

        
        // Set steps mode
        temp_simplifier.collect_steps = self.verbosity != Verbosity::None;
        
        match cas_parser::parse(expr_str, &mut temp_simplifier.context) {
            Ok(expr) => {
                println!("Parsed: {}", DisplayExpr { context: &temp_simplifier.context, id: expr });
                let (simplified, steps) = temp_simplifier.simplify(expr);
                
                if self.verbosity != Verbosity::None {
                    if steps.is_empty() {
                        if self.verbosity != Verbosity::Low {
                            println!("No simplification steps needed.");
                        }
                    } else {
                        if self.verbosity != Verbosity::Low {
                            println!("Steps (Aggressive Mode):");
                        }
                        let mut current_root = expr;
                        let mut step_count = 0;
                        for step in steps.iter() {
                            if should_show_step(step, self.verbosity) {
                                step_count += 1;
                                
                                if self.verbosity == Verbosity::Low {
                                    // Low mode: just global state
                                    current_root = reconstruct_global_expr(&mut temp_simplifier.context, current_root, &step.path, step.after);
                                    println!("-> {}", DisplayExpr { context: &temp_simplifier.context, id: current_root });
                                } else {
                                    // Normal/Verbose
                                    println!("{}. {}  [{}]", step_count, step.description, step.rule_name);
                                    
                                    if self.verbosity == Verbosity::Verbose || self.verbosity == Verbosity::Normal {
                                        let after_disp = if let Some(s) = &step.after_str {
                                            s.clone()
                                        } else {
                                            format!("{}", DisplayExpr { context: &temp_simplifier.context, id: step.after })
                                        };
                                        println!("   Local: {} -> {}", 
                                            DisplayExpr { context: &temp_simplifier.context, id: step.before },
                                            after_disp
                                        );
                                    }
                                    
                                    current_root = reconstruct_global_expr(&mut temp_simplifier.context, current_root, &step.path, step.after);
                                    println!("   Global: {}", DisplayExpr { context: &temp_simplifier.context, id: current_root });
                                }
                            } else {
                                current_root = reconstruct_global_expr(&mut temp_simplifier.context, current_root, &step.path, step.after);
                            }
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
