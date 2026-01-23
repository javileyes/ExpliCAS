use super::*;

impl Repl {
    pub(crate) fn handle_set_command(&mut self, line: &str) {
        let parts: Vec<&str> = line.split_whitespace().collect();

        // `set` or `set show` → show all settings
        if parts.len() == 1 || (parts.len() == 2 && parts[1] == "show") {
            self.print_set_help();
            return;
        }

        // `set <option>` → show that option's current value
        if parts.len() == 2 {
            match parts[1] {
                "transform" => {
                    println!(
                        "transform: {}",
                        if self.simplify_options.enable_transform {
                            "on"
                        } else {
                            "off"
                        }
                    );
                }
                "rationalize" => {
                    println!(
                        "rationalize: {:?}",
                        self.simplify_options.rationalize.auto_level
                    );
                }
                "heuristic_poly" => {
                    use cas_engine::options::HeuristicPoly;
                    println!(
                        "heuristic_poly: {}",
                        if self.simplify_options.heuristic_poly == HeuristicPoly::On {
                            "on"
                        } else {
                            "off"
                        }
                    );
                }
                "autoexpand" | "autoexpand_binomials" => {
                    use cas_engine::options::AutoExpandBinomials;
                    println!(
                        "autoexpand: {}",
                        if self.simplify_options.autoexpand_binomials == AutoExpandBinomials::On {
                            "on"
                        } else {
                            "off"
                        }
                    );
                }
                "max-rewrites" => {
                    println!(
                        "max-rewrites: {}",
                        self.simplify_options.budgets.max_total_rewrites
                    );
                }
                "debug" => {
                    println!("debug: {}", if self.debug_mode { "on" } else { "off" });
                }
                "steps" => {
                    use cas_engine::options::StepsMode;
                    let mode_str = match self.state.options.steps_mode {
                        StepsMode::On => "on",
                        StepsMode::Off => "off",
                        StepsMode::Compact => "compact",
                    };
                    let verbosity_str = match self.verbosity {
                        Verbosity::None => "none",
                        Verbosity::Succinct => "succinct",
                        Verbosity::Normal => "normal",
                        Verbosity::Verbose => "verbose",
                    };
                    println!("steps: {} (display: {})", mode_str, verbosity_str);
                }
                _ => {
                    println!("Unknown option: {}", parts[1]);
                    println!("Use 'set show' to see available options");
                }
            }
            return;
        }

        // `set <option> <value>` → change value
        match parts[1] {
            "transform" => match parts[2] {
                "on" | "true" | "1" => {
                    self.simplify_options.enable_transform = true;
                    println!("Transform phase ENABLED (distribution, expansion)");
                }
                "off" | "false" | "0" => {
                    self.simplify_options.enable_transform = false;
                    println!("Transform phase DISABLED (no distribution/expansion)");
                }
                _ => println!("Usage: set transform <on|off>"),
            },
            "autoexpand" | "autoexpand_binomials" => match parts[2] {
                "on" | "true" | "1" => {
                    self.state.options.autoexpand_binomials = cas_engine::AutoExpandBinomials::On;
                    self.simplify_options.autoexpand_binomials =
                        cas_engine::AutoExpandBinomials::On;
                    println!("Autoexpand binomials: ON (always expand)");
                    println!("  (x+1)^5 will now expand to x⁵+5x⁴+10x³+10x²+5x+1");
                }
                "off" | "false" | "0" => {
                    self.state.options.autoexpand_binomials = cas_engine::AutoExpandBinomials::Off;
                    self.simplify_options.autoexpand_binomials =
                        cas_engine::AutoExpandBinomials::Off;
                    println!("Autoexpand binomials: OFF (default, keep factored form)");
                }
                _ => println!("Usage: set autoexpand <off|on>"),
            },
            "heuristic_poly" => match parts[2] {
                "on" | "true" | "1" => {
                    self.state.options.heuristic_poly = cas_engine::HeuristicPoly::On;
                    self.simplify_options.heuristic_poly = cas_engine::HeuristicPoly::On;
                    println!("Heuristic polynomial simplification: ON");
                    println!("  - Extract common factors in Add/Sub");
                    println!("  - Poly normalize if no factor found");
                    println!("  Example: (x+1)^4 + 4·(x+1)^3 → (x+1)³·(x+5)");
                }
                "off" | "false" | "0" => {
                    self.state.options.heuristic_poly = cas_engine::HeuristicPoly::Off;
                    self.simplify_options.heuristic_poly = cas_engine::HeuristicPoly::Off;
                    println!("Heuristic polynomial simplification: OFF (default)");
                }
                _ => println!("Usage: set heuristic_poly <off|on>"),
            },
            "rationalize" => match parts[2] {
                "on" | "true" | "auto" => {
                    self.simplify_options.rationalize.auto_level =
                        cas_engine::rationalize_policy::AutoRationalizeLevel::Level15;
                    println!("Rationalization ENABLED (Level 1.5)");
                }
                "off" | "false" => {
                    self.simplify_options.rationalize.auto_level =
                        cas_engine::rationalize_policy::AutoRationalizeLevel::Off;
                    println!("Rationalization DISABLED");
                }
                "0" | "level0" => {
                    self.simplify_options.rationalize.auto_level =
                        cas_engine::rationalize_policy::AutoRationalizeLevel::Level0;
                    println!("Rationalization set to Level 0 (single sqrt)");
                }
                "1" | "level1" => {
                    self.simplify_options.rationalize.auto_level =
                        cas_engine::rationalize_policy::AutoRationalizeLevel::Level1;
                    println!("Rationalization set to Level 1 (binomial conjugate)");
                }
                "1.5" | "level15" => {
                    self.simplify_options.rationalize.auto_level =
                        cas_engine::rationalize_policy::AutoRationalizeLevel::Level15;
                    println!("Rationalization set to Level 1.5 (same-surd products)");
                }
                _ => println!("Usage: set rationalize <on|off|0|1|1.5>"),
            },
            "max-rewrites" => {
                if let Ok(n) = parts[2].parse::<usize>() {
                    self.simplify_options.budgets.max_total_rewrites = n;
                    println!("Max rewrites set to {}", n);
                } else {
                    println!("Usage: set max-rewrites <number>");
                }
            }
            "steps" => {
                use cas_engine::options::StepsMode;
                match parts[2] {
                    "on" => {
                        self.state.options.steps_mode = StepsMode::On;
                        self.engine.simplifier.set_steps_mode(StepsMode::On);
                        self.verbosity = Verbosity::Normal;
                        println!("Steps: on (full collection, normal display)");
                    }
                    "off" => {
                        self.state.options.steps_mode = StepsMode::Off;
                        self.engine.simplifier.set_steps_mode(StepsMode::Off);
                        self.verbosity = Verbosity::None;
                        println!("Steps: off");
                    }
                    "compact" => {
                        self.state.options.steps_mode = StepsMode::Compact;
                        self.engine.simplifier.set_steps_mode(StepsMode::Compact);
                        println!("Steps: compact (no before/after snapshots)");
                    }
                    "verbose" => {
                        self.state.options.steps_mode = StepsMode::On;
                        self.engine.simplifier.set_steps_mode(StepsMode::On);
                        self.verbosity = Verbosity::Verbose;
                        println!("Steps: verbose (all rules, full detail)");
                    }
                    "succinct" => {
                        self.state.options.steps_mode = StepsMode::On;
                        self.engine.simplifier.set_steps_mode(StepsMode::On);
                        self.verbosity = Verbosity::Succinct;
                        println!("Steps: succinct (compact 1-line per step)");
                    }
                    "normal" => {
                        self.state.options.steps_mode = StepsMode::On;
                        self.engine.simplifier.set_steps_mode(StepsMode::On);
                        self.verbosity = Verbosity::Normal;
                        println!("Steps: normal (default display)");
                    }
                    "none" => {
                        self.verbosity = Verbosity::None;
                        println!("Steps display: none (collection still active)");
                    }
                    _ => println!("Usage: set steps <on|off|compact|verbose|succinct|normal|none>"),
                }
            }
            "debug" => match parts[2] {
                "on" | "true" | "1" => {
                    self.debug_mode = true;
                    println!("Debug mode ENABLED (pipeline diagnostics after each operation)");
                }
                "off" | "false" | "0" => {
                    self.debug_mode = false;
                    println!("Debug mode DISABLED");
                }
                _ => println!("Usage: set debug <on|off>"),
            },
            _ => self.print_set_help(),
        }
    }

    pub(crate) fn print_set_help(&self) {
        println!("Pipeline settings:");
        println!("  set transform <on|off>         Enable/disable distribution & expansion");
        println!("  set rationalize <on|off|0|1|1.5>  Set rationalization level");
        println!("  set heuristic_poly <on|off>    Smart polynomial simplification/factorization");
        println!(
            "  set autoexpand <on|off>        Force expansion of binomial powers like (x+1)^n"
        );
        println!("  set steps <on|off|...>         Step collection and display mode");
        println!("  set max-rewrites <N>           Set max total rewrites (safety limit)");
        println!("  set debug <on|off>             Show pipeline diagnostics after operations");
        println!();
        println!("Current settings:");
        println!(
            "  transform: {}",
            if self.simplify_options.enable_transform {
                "on"
            } else {
                "off"
            }
        );
        println!(
            "  rationalize: {:?}",
            self.simplify_options.rationalize.auto_level
        );
        use cas_engine::options::HeuristicPoly;
        println!(
            "  heuristic_poly: {}",
            if self.simplify_options.heuristic_poly == HeuristicPoly::On {
                "on"
            } else {
                "off"
            }
        );
        use cas_engine::options::AutoExpandBinomials;
        println!(
            "  autoexpand: {}",
            if self.simplify_options.autoexpand_binomials == AutoExpandBinomials::On {
                "on"
            } else {
                "off"
            }
        );
        use cas_engine::options::StepsMode;
        let mode_str = match self.state.options.steps_mode {
            StepsMode::On => "on",
            StepsMode::Off => "off",
            StepsMode::Compact => "compact",
        };
        let verbosity_str = match self.verbosity {
            Verbosity::None => "none",
            Verbosity::Succinct => "succinct",
            Verbosity::Normal => "normal",
            Verbosity::Verbose => "verbose",
        };
        println!("  steps: {} (display: {})", mode_str, verbosity_str);
        println!(
            "  max-rewrites: {}",
            self.simplify_options.budgets.max_total_rewrites
        );
        println!("  debug: {}", if self.debug_mode { "on" } else { "off" });
    }

    pub(crate) fn handle_help(&self, line: &str) {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 2 {
            self.print_general_help();
            return;
        }

        match parts[1] {
            "simplify" => {
                println!("Command: simplify <expr>");
                println!(
                    "Description: Simplifies an expression using the full power of the engine."
                );
                println!(
                    "             This includes aggressive distribution and other rules that may"
                );
                println!("             undo factorizations, but guarantee maximum simplification.");
                println!("Example: simplify (x+1)*(x-1) -> x^2 - 1");
            }
            "diff" => {
                println!("Command: diff <expr> <var>");
                println!("Description: Computes the symbolic derivative of an expression with respect to a variable.");
                println!("             Supports basic arithmetic, power rule, chain rule, and common functions.");
                println!("Example: diff(sin(x^2), x) -> 2*x*cos(x^2)");
            }
            "sum" => {
                println!("Function: sum(expr, var, start, end)");
                println!("Description: Evaluates finite summations Σ(var=start to end) expr.");
                println!("             Supports numeric evaluation and telescoping detection.");
                println!("Features:");
                println!("  - Numeric: sum(k, k, 1, 10) -> 55");
                println!("  - Symbolic end: sum(1/(k*(k+1)), k, 1, n) -> 1 - 1/(n+1)");
                println!("  - Telescoping: Automatically detects 1/(k*(k+a)) patterns");
                println!("Examples:");
                println!("  sum(k^2, k, 1, 5)           -> 55");
                println!("  sum(1/(k*(k+1)), k, 1, n)   -> 1 - 1/(1+n)");
                println!("  sum(1/(k*(k+2)), k, 1, n)   -> 1/2 - 1/2/(1+n)");
            }
            "product" => {
                println!("Function: product(expr, var, start, end)");
                println!("Description: Evaluates finite products Π(var=start to end) expr.");
                println!("             Supports numeric evaluation and telescoping detection.");
                println!("Features:");
                println!("  - Numeric: product(k, k, 1, 5) -> 120 (5!)");
                println!("  - Symbolic end: product((k+1)/k, k, 1, n) -> n+1");
                println!("  - Telescoping: Detects (k+a)/(k+b) quotient patterns");
                println!("Examples:");
                println!("  product(k, k, 1, 5)         -> 120");
                println!("  product((k+1)/k, k, 1, n)   -> 1 + n");
                println!("  product((k+1)/k, k, 1, 10)  -> 11");
            }
            "gcd" => {
                println!("Function: gcd <a, b>");
                println!("Description: Computes the Greatest Common Divisor of two integers.");
                println!("Example: gcd(12, 18) -> 6");
            }
            "lcm" => {
                println!("Function: lcm <a, b>");
                println!("Description: Computes the Least Common Multiple of two integers.");
                println!("Example: lcm(4, 6) -> 12");
            }
            "mod" => {
                println!("Function: mod <a, n>");
                println!(
                    "Description: Computes the remainder of a divided by n (Euclidean modulo)."
                );
                println!("Example: mod(10, 3) -> 1");
            }
            "factors" | "prime_factors" => {
                println!("Function: factors <n>");
                println!("Description: Computes the prime factorization of an integer.");
                println!("Example: factors(12) -> 2^2 * 3");
            }
            "fact" | "factorial" => {
                println!("Function: fact <n> or <n>!");
                println!("Description: Computes the factorial of a non-negative integer.");
                println!("Example: fact(5) -> 120, 5! -> 120");
            }
            "choose" | "nCr" => {
                println!("Function: choose <n, k>");
                println!("Description: Computes the binomial coefficient nCk (combinations).");
                println!("Example: choose(5, 2) -> 10");
            }
            "perm" | "nPr" => {
                println!("Function: perm <n, k>");
                println!("Description: Computes the number of permutations nPk.");
                println!("Example: perm(5, 2) -> 20");
            }
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
            }
            "subst" => {
                println!("Command: subst <expr>, <target>, <replacement>");
                println!("Description: Substitutes a pattern with a replacement and simplifies.");
                println!();
                println!("Variable substitution:");
                println!("  subst x^2 + x, x, 3           → 12");
                println!();
                println!("Power-aware expression substitution:");
                println!("  subst x^4 + x^2 + 1, x^2, y   → y² + y + 1");
                println!("  subst x^3, x^2, y             → y·x (with remainder)");
                println!("  subst x^6, x^2, y             → y³");
            }
            "expand" => {
                println!("Command: expand <expr>");
                println!("Description: Expands polynomials and products.");
                println!("Example: expand(x+1)^2 -> x^2 + 2*x + 1");
            }
            "factor" => {
                println!("Command: factor <expr>");
                println!("Description: Factors polynomials.");
                println!("Example: factor(x^2 - 1) -> (x - 1) * (x + 1)");
            }
            "collect" => {
                println!("Command: collect <expr>, <var>");
                println!("Description: Groups terms by powers of a variable.");
                println!("Example: collect(a*x + b*x + c, x) -> (a + b) * x + c");
            }
            "equiv" => {
                println!("Command: equiv <expr1>, <expr2>");
                println!("Description: Checks if two expressions are mathematically equivalent.");
                println!("             Returns true if expr1 - expr2 simplifies to 0.");
            }
            "solve" => {
                println!("Command: solve <equation>, <var>");
                println!("Description: Solves an equation for a variable.");
                println!("Example: solve x + 2 = 5, x -> x = 3");
            }
            "steps" => {
                println!("Command: steps <level>");
                println!("Description: Controls the verbosity of simplification steps.");
                println!("Levels:");
                println!("  normal (or on)   Show clarifying steps (Global state). Default.");
                println!("  succinct              Compact: same steps as normal but 1 line each.");
                println!("  verbose          Show all steps (Local + Global details).");
                println!("  none (or off)    Disable step output.");
            }
            "profile" => {
                println!("Command: profile [subcommand]");
                println!("Description: Rule profiler for debugging and performance analysis.");
                println!("Subcommands:");
                println!("  (none)           Show profiling report");
                println!("  enable           Enable profiler");
                println!("  disable          Disable profiler");
                println!("  clear            Clear statistics");
                println!("Example: profile enable, then run expressions, then profile");
            }
            "visualize" => {
                println!("Command: visualize <expr>");
                println!("Description: Export expression tree to Graphviz DOT format.");
                println!("             Generates ast.dot file for rendering.");
                println!("Example: visualize (x+1)*(x-1)");
                println!();
                println!("To render the generated file, use Graphviz in your terminal:");
                println!("  $ dot -Tsvg ast.dot -o ast.svg");
                println!("  $ open ast.svg");
            }
            "timeline" => {
                println!("Command: timeline <expr>");
                println!("Description: Export simplification steps to interactive HTML.");
                println!("             Generates timeline.html with MathJax rendering.");
                println!("Example: timeline (x+1)^2");
                println!("         Open timeline.html in browser to view.");
            }
            "explain" => {
                println!("Command: explain <function>");
                println!(
                    "Description: Provides step-by-step educational explanations of mathematical"
                );
                println!("             operations. Shows the detailed algorithm steps in Spanish.");
                println!("Supported functions:");
                println!("  gcd(a, b)    Greatest Common Divisor using Euclidean algorithm");
                println!("               Works for both integers and polynomials.");
                println!("Examples:");
                println!("  explain gcd(48, 18)");
                println!("  explain gcd(2*x^2 + 7*x + 3, 2*x^2 + 5*x + 2)");
            }
            "det" => {
                println!("Command: det <matrix>");
                println!("Description: Compute the determinant of a square matrix.");
                println!("             Supports 1×1, 2×2, and 3×3 matrices.");
                println!("Examples:");
                println!("  det [[1, 2], [3, 4]]        → -2");
                println!("  det [[2]]                    → 2");
                println!("  det [[1, 2, 3], [4, 5, 6], [7, 8, 9]]");
            }
            "transpose" => {
                println!("Command: transpose <matrix>");
                println!("Description: Transpose a matrix (swap rows and columns).");
                println!("             Works with any rectangular matrix.");
                println!("Examples:");
                println!("  transpose [[1, 2, 3], [4, 5, 6]]");
                println!("    → [[1, 4], [2, 5], [3, 6]]");
                println!("  transpose [[1, 2], [3, 4]]");
                println!("    → [[1, 3], [2, 4]]");
            }
            "trace" => {
                println!("Command: trace <matrix>");
                println!("Description: Compute the trace of a square matrix.");
                println!("             The trace is the sum of diagonal elements.");
                println!("Examples:");
                println!("  trace [[1, 2], [3, 4]]      → 5");
                println!("  trace [[5, 0, 0], [0, 3, 0], [0, 0, 2]]  → 10");
            }
            "rationalize" => {
                println!("Command: rationalize <expr>");
                println!("Description: Rationalize denominators containing surds (square roots).");
                println!(
                    "             Eliminates irrational numbers from denominators by multiplying"
                );
                println!("             by the conjugate.");
                println!("Examples:");
                println!("  rationalize 1/(1 + sqrt(2))      → √2 - 1");
                println!("  rationalize 1/(3 - 2*sqrt(5))    → -(3 + 2*√5)/11");
                println!("  rationalize x/(sqrt(3) + 1)      → x*(√3 - 1)/2");
            }
            "status" | "health" => {
                println!("Command: health [on|off|reset|status]");
                println!("Description: Engine health monitoring and diagnostic test suite.");
                println!();
                println!("Subcommands:");
                println!("  health on                Enable profiler");
                println!("  health off               Disable profiler");
                println!("  health reset             Reset profiler stats");
                println!("  health status            Run diagnostic test suite");
                println!();
                println!("Test suite options:");
                println!("  health status --list             List all test cases");
                println!("  health status --category <cat>   Run specific category");
                println!("  health status -c <cat>           Shorthand for --category");
                println!();
                println!("Categories: transform, expansion, fractions, rationalization,");
                println!("            mixed, baseline, roots, powers, stress, all");
                println!();
                println!("Examples:");
                println!("  health status                Run all test categories");
                println!("  health status -c stress      Run only stress tests");
                println!("  health status --list         List available tests");
            }
            // Session environment commands
            "let" => {
                println!("Command: let <name> = <expr>");
                println!("Description: Assigns an expression to a variable name.");
                println!("             The variable can be used in subsequent expressions.");
                println!("             Substitution is transitive and cycle-safe.");
                println!();
                println!("Examples:");
                println!("  let a = 5");
                println!("  let b = a + 1        → b becomes 6");
                println!("  let f = x^2 + 1      → f stores symbolic expression");
            }
            "vars" => {
                println!("Command: vars");
                println!("Description: Lists all defined variables and their values.");
                println!();
                println!("Example output:");
                println!("  a = 5");
                println!("  b = 6");
                println!("  f = x^2 + 1");
            }
            "clear" => {
                println!("Command: clear [name ...]");
                println!("Description: Clears variable bindings from the environment.");
                println!("             Without arguments, clears ALL variables.");
                println!("             With arguments, clears only the specified variables.");
                println!();
                println!("Examples:");
                println!("  clear           → clears all variables");
                println!("  clear a b       → clears only a and b");
            }
            "reset" => {
                println!("Command: reset");
                println!("Description: Resets the entire session state.");
                println!("             Clears all variables AND session history (#ids).");
            }
            "history" | "list" => {
                println!("Command: history (or list)");
                println!("Description: Shows all stored session entries with their #ids.");
                println!("             Each expression you evaluate is stored with a unique ID.");
                println!();
                println!("Example output:");
                println!("  #1: x + 1");
                println!("  #2: 2*x - 3");
                println!("  #3: x + 1 = 5  [Eq]");
            }
            "show" => {
                println!("Command: show #<id>");
                println!("Description: Displays a specific session entry by its ID.");
                println!();
                println!("Example:");
                println!("  show #1         → shows the expression stored as #1");
            }
            "del" => {
                println!("Command: del #<id> [#<id> ...]");
                println!("Description: Deletes session entries by their IDs.");
                println!("             IDs are never reused after deletion.");
                println!();
                println!("Examples:");
                println!("  del #1          → deletes entry #1");
                println!("  del #2 #3 #5    → deletes entries #2, #3, and #5");
            }
            "poly_gcd" | "pgcd" => {
                println!("Command: poly_gcd(expr1, expr2)");
                println!("Alias: pgcd(expr1, expr2)");
                println!("Description: Computes the STRUCTURAL GCD of two polynomial expressions.");
                println!(
                    "             Finds common factors that appear explicitly as multiplicands."
                );
                println!("             Does NOT factor expressions to find hidden common factors.");
                println!();
                println!("Examples:");
                println!("  poly_gcd((x+1)*(y+2), (x+1)*(z+3)) → (x + 1)");
                println!("  poly_gcd((x+1)^3, (x+1)^2)         → (x + 1)²");
                println!("  poly_gcd(x*g, y*g)                 → g");
                println!();
                println!("See also: poly_gcd_exact for algebraic GCD");
            }
            "poly_gcd_exact" | "pgcdx" => {
                println!("Command: poly_gcd_exact(expr1, expr2)");
                println!("Alias: pgcdx(expr1, expr2)");
                println!(
                    "Description: Computes the ALGEBRAIC GCD of two polynomials over ℚ[x₁,...,xₙ]."
                );
                println!(
                    "             Interprets expressions as polynomials and finds the true GCD."
                );
                println!("             Uses Euclidean algorithm for univariate, interpolation for multivariate.");
                println!();
                println!("Examples:");
                println!("  poly_gcd_exact(x^2 - 1, x - 1)         → x - 1");
                println!("  poly_gcd_exact(x^2 - 1, x^2 - 2*x + 1) → x - 1");
                println!("  poly_gcd_exact(2*x + 2*y, 4*x + 4*y)   → x + y");
                println!("  poly_gcd_exact(6, 15)                  → 1 (constants over ℚ)");
                println!();
                println!("Result is normalized: primitive (GCD of coefficients = 1), positive leading coefficient.");
                println!();
                println!("See also: poly_gcd for structural (visible factor) GCD");
            }
            "limit" => {
                println!("Command: limit <expr> [, <var> [, <direction>]]");
                println!();
                println!("Description: Compute the limit of an expression as a variable approaches infinity.");
                println!("             Uses polynomial degree comparison for rational functions P(x)/Q(x).");
                println!();
                println!("Arguments:");
                println!("  <expr>       Expression to evaluate the limit of");
                println!("  <var>        Variable approaching the limit (default: x)");
                println!("  <direction>  Direction: infinity or -infinity (default: infinity)");
                println!();
                println!("Examples:");
                println!("  limit x^2                      → infinity");
                println!("  limit (x^2+1)/(2*x^2-3), x     → 1/2");
                println!("  limit x^3/x^2, x, -infinity    → -infinity");
                println!("  limit x^2/x^3                  → 0");
                println!();
                println!("Behavior:");
                println!("  - deg(P) < deg(Q): limit = 0");
                println!("  - deg(P) = deg(Q): limit = leading_coeff(P) / leading_coeff(Q)");
                println!(
                    "  - deg(P) > deg(Q): limit = ±∞ (sign depends on coefficients and approach)"
                );
                println!();
                println!("Residuals:");
                println!("  If the limit cannot be determined (e.g., sin(x)/x, non-polynomial expressions),");
                println!("  returns limit(...) as a symbolic residual with a warning.");
            }
            "expand_log" => {
                println!("Command: expand_log <expr>");
                println!();
                println!("Description: Expands logarithms using log laws (product, quotient, power rules).");
                println!("             Requires positivity assumptions for correctness.");
                println!();
                println!("Laws applied:");
                println!("  ln(a*b)   → ln(a) + ln(b)   (product rule)");
                println!("  ln(a/b)   → ln(a) - ln(b)   (quotient rule)");
                println!("  ln(a^n)   → n*ln(a)         (power rule)");
                println!();
                println!("Examples:");
                println!("  expand_log ln(x^2 * y)     → 2*ln(x) + ln(y)");
                println!("  expand_log ln(a/b)        → ln(a) - ln(b)");
                println!("  expand_log ln(x^3 * y^2)  → 3*ln(x) + 2*ln(y)");
            }
            _ => {
                println!("Unknown command: {}", parts[1]);
                self.print_general_help();
            }
        }
    }

    pub(crate) fn print_general_help(&self) {
        println!("Rust CAS Commands:");
        println!();

        println!("Basic Operations:");
        println!("  <expr>                  Evaluate and simplify an expression");
        println!("  simplify <expr>         Aggressive simplification (full power)");
        println!("  expand <expr>           Expand polynomials");
        println!("  expand_log <expr>       Expand logarithms (log laws)");
        println!("  factor <expr>           Factor polynomials");
        println!("  collect <expr>, <var>   Group terms by variable");
        println!();

        println!("Polynomial GCD:");
        println!("  poly_gcd(a, b)          Structural GCD (visible factors)");
        println!("  poly_gcd_exact(a, b)    Algebraic GCD over ℚ[x₁,...,xₙ]");
        println!("  pgcd                    Alias for poly_gcd");
        println!("  pgcdx                   Alias for poly_gcd_exact");
        println!();

        println!("Equation Solving:");
        println!("  solve <eq>, <var>       Solve equation for variable");
        println!("  equiv <e1>, <e2>        Check if two expressions are equivalent");
        println!("  subst <expr>, <var>=<val> Substitute a variable and simplify");
        println!();

        println!("Calculus:");
        println!("  diff <expr>, <var>      Compute symbolic derivative");
        println!("  limit <expr>            Compute limit at ±∞ (CLI: expli limit)");
        println!("  sum(e, v, a, b)         Finite summation: Σ(v=a to b) e");
        println!("  product(e, v, a, b)     Finite product: Π(v=a to b) e");
        println!();

        println!("Number Theory:");
        println!("  gcd <a, b>              Greatest Common Divisor");
        println!("  lcm <a, b>              Least Common Multiple");
        println!("  mod <a, n>              Modular arithmetic");
        println!("  factors <n>             Prime factorization");
        println!("  fact <n>                Factorial (or n!)");
        println!("  choose <n, k>           Binomial coefficient (nCk)");
        println!("  perm <n, k>             Permutations (nPk)");
        println!();

        println!("Matrix Operations:");
        println!("  det <matrix>            Compute determinant (up to 3×3)");
        println!("  transpose <matrix>      Transpose a matrix");
        println!("  trace <matrix>          Compute trace (sum of diagonal)");
        println!();

        println!("Analysis & Verification:");
        println!("  explain <function>      Show step-by-step explanation");
        println!("  telescope <expr>        Prove telescoping identities (Dirichlet kernel)");
        println!("  steps <level>           Set step verbosity (normal, succinct, verbose, none)");
        println!();

        println!("Visualization & Output:");
        println!("  visualize <expr>        Export AST to Graphviz DOT (generates ast.dot)");
        println!("  timeline <expr>         Export steps to interactive HTML");
        println!();

        println!(
            "  set <option> <value>    Pipeline settings (transform, rationalize, max-rewrites)"
        );
        println!("  semantics [set|help]    Semantic settings (domain, value, inv_trig, branch)");
        println!("  context [mode]          Context mode (auto, standard, solve, integrate)");
        println!("  config <subcmd>         Manage configuration (list, enable, disable...)");
        println!("  profile [cmd]           Rule profiler (enable/disable/clear)");
        println!("  health [cmd]            Health tracking (on/off/reset/status)");
        println!("  help [cmd]              Show this help message or details for a command");
        println!("  quit / exit             Exit the REPL");
        println!();

        println!("Session Environment:");
        println!("  let <name> = <expr>     Assign a variable");
        println!("  <name> := <expr>        Alternative assignment syntax");
        println!("  vars                    List all defined variables");
        println!("  clear [name]            Clear one or all variables");
        println!("  reset                   Clear all session state (keeps cache)");
        println!("  reset full              Clear all session state AND profile cache");
        println!("  budget [N]              Set/show Conditional branching budget (0-3)");
        println!("  cache [status|clear]    View or clear profile cache");
        println!("  history / list          Show session history (#ids)");
        println!("  show #<id>              Display a session entry");
        println!("  del #<id> ...           Delete session entries");
        println!();

        println!("Type 'help <command>' for more details on a specific command.");
    }
}
