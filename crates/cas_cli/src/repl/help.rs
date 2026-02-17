use super::output::{CoreResult, UiDelta};
use super::*;

impl Repl {
    /// Legacy wrapper - calls core, applies UI delta, and prints
    pub(crate) fn handle_set_command(&mut self, line: &str) {
        let result = self.handle_set_command_core(line);
        // Apply UI delta
        if let Some(v) = result.ui_delta.verbosity {
            self.verbosity = v;
        }
        self.print_reply(result.reply);
    }

    /// Core: handle set command, returns CoreResult with UI delta for verbosity changes
    pub(crate) fn handle_set_command_core(&mut self, line: &str) -> CoreResult {
        let parts: Vec<&str> = line.split_whitespace().collect();
        let mut ui_delta = UiDelta::default();

        // `set` or `set show` → show all settings
        if parts.len() == 1 || (parts.len() == 2 && parts[1] == "show") {
            return self.set_help_core().into();
        }

        // `set <option>` → show that option's current value
        if parts.len() == 2 {
            let msg = match parts[1] {
                "transform" => format!(
                    "transform: {}",
                    if self.core.simplify_options.enable_transform {
                        "on"
                    } else {
                        "off"
                    }
                ),
                "rationalize" => format!(
                    "rationalize: {:?}",
                    self.core.simplify_options.rationalize.auto_level
                ),
                "heuristic_poly" => {
                    use cas_solver::HeuristicPoly;
                    format!(
                        "heuristic_poly: {}",
                        if self.core.simplify_options.shared.heuristic_poly == HeuristicPoly::On {
                            "on"
                        } else {
                            "off"
                        }
                    )
                }
                "autoexpand" | "autoexpand_binomials" => {
                    use cas_solver::AutoExpandBinomials;
                    format!(
                        "autoexpand: {}",
                        if self.core.simplify_options.shared.autoexpand_binomials
                            == AutoExpandBinomials::On
                        {
                            "on"
                        } else {
                            "off"
                        }
                    )
                }
                "max-rewrites" => format!(
                    "max-rewrites: {}",
                    self.core.simplify_options.budgets.max_total_rewrites
                ),
                "debug" => format!("debug: {}", if self.core.debug_mode { "on" } else { "off" }),
                "steps" => {
                    use cas_solver::StepsMode;
                    let mode_str = match self.core.state.options.steps_mode {
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
                    format!("steps: {} (display: {})", mode_str, verbosity_str)
                }
                _ => format!(
                    "Unknown option: {}\nUse 'set show' to see available options",
                    parts[1]
                ),
            };
            return vec![ReplMsg::info(msg)].into();
        }

        // `set <option> <value>` → change value
        let msg = match parts[1] {
            "transform" => match parts[2] {
                "on" | "true" | "1" => {
                    self.core.simplify_options.enable_transform = true;
                    "Transform phase ENABLED (distribution, expansion)".to_string()
                }
                "off" | "false" | "0" => {
                    self.core.simplify_options.enable_transform = false;
                    "Transform phase DISABLED (no distribution/expansion)".to_string()
                }
                _ => "Usage: set transform <on|off>".to_string(),
            },
            "autoexpand" | "autoexpand_binomials" => match parts[2] {
                "on" | "true" | "1" => {
                    self.core.state.options.shared.autoexpand_binomials =
                        cas_solver::AutoExpandBinomials::On;
                    self.core.simplify_options.shared.autoexpand_binomials =
                        cas_solver::AutoExpandBinomials::On;
                    "Autoexpand binomials: ON (always expand)\n  (x+1)^5 will now expand to x⁵+5x⁴+10x³+10x²+5x+1".to_string()
                }
                "off" | "false" | "0" => {
                    self.core.state.options.shared.autoexpand_binomials =
                        cas_solver::AutoExpandBinomials::Off;
                    self.core.simplify_options.shared.autoexpand_binomials =
                        cas_solver::AutoExpandBinomials::Off;
                    "Autoexpand binomials: OFF (default, keep factored form)".to_string()
                }
                _ => "Usage: set autoexpand <off|on>".to_string(),
            },
            "heuristic_poly" => match parts[2] {
                "on" | "true" | "1" => {
                    self.core.state.options.shared.heuristic_poly = cas_solver::HeuristicPoly::On;
                    self.core.simplify_options.shared.heuristic_poly =
                        cas_solver::HeuristicPoly::On;
                    "Heuristic polynomial simplification: ON\n  - Extract common factors in Add/Sub\n  - Poly normalize if no factor found\n  Example: (x+1)^4 + 4·(x+1)^3 → (x+1)³·(x+5)".to_string()
                }
                "off" | "false" | "0" => {
                    self.core.state.options.shared.heuristic_poly = cas_solver::HeuristicPoly::Off;
                    self.core.simplify_options.shared.heuristic_poly =
                        cas_solver::HeuristicPoly::Off;
                    "Heuristic polynomial simplification: OFF (default)".to_string()
                }
                _ => "Usage: set heuristic_poly <off|on>".to_string(),
            },
            "rationalize" => match parts[2] {
                "on" | "true" | "auto" => {
                    self.core.simplify_options.rationalize.auto_level =
                        cas_solver::AutoRationalizeLevel::Level15;
                    "Rationalization ENABLED (Level 1.5)".to_string()
                }
                "off" | "false" => {
                    self.core.simplify_options.rationalize.auto_level =
                        cas_solver::AutoRationalizeLevel::Off;
                    "Rationalization DISABLED".to_string()
                }
                "0" | "level0" => {
                    self.core.simplify_options.rationalize.auto_level =
                        cas_solver::AutoRationalizeLevel::Level0;
                    "Rationalization set to Level 0 (single sqrt)".to_string()
                }
                "1" | "level1" => {
                    self.core.simplify_options.rationalize.auto_level =
                        cas_solver::AutoRationalizeLevel::Level1;
                    "Rationalization set to Level 1 (binomial conjugate)".to_string()
                }
                "1.5" | "level15" => {
                    self.core.simplify_options.rationalize.auto_level =
                        cas_solver::AutoRationalizeLevel::Level15;
                    "Rationalization set to Level 1.5 (same-surd products)".to_string()
                }
                _ => "Usage: set rationalize <on|off|0|1|1.5>".to_string(),
            },
            "max-rewrites" => {
                if let Ok(n) = parts[2].parse::<usize>() {
                    self.core.simplify_options.budgets.max_total_rewrites = n;
                    format!("Max rewrites set to {}", n)
                } else {
                    "Usage: set max-rewrites <number>".to_string()
                }
            }
            "steps" => {
                use cas_solver::StepsMode;
                match parts[2] {
                    "on" => {
                        self.core.state.options.steps_mode = StepsMode::On;
                        self.core.engine.simplifier.set_steps_mode(StepsMode::On);
                        ui_delta.verbosity = Some(Verbosity::Normal);
                        "Steps: on (full collection, normal display)".to_string()
                    }
                    "off" => {
                        self.core.state.options.steps_mode = StepsMode::Off;
                        self.core.engine.simplifier.set_steps_mode(StepsMode::Off);
                        ui_delta.verbosity = Some(Verbosity::None);
                        "Steps: off".to_string()
                    }
                    "compact" => {
                        self.core.state.options.steps_mode = StepsMode::Compact;
                        self.core
                            .engine
                            .simplifier
                            .set_steps_mode(StepsMode::Compact);
                        "Steps: compact (no before/after snapshots)".to_string()
                    }
                    "verbose" => {
                        self.core.state.options.steps_mode = StepsMode::On;
                        self.core.engine.simplifier.set_steps_mode(StepsMode::On);
                        ui_delta.verbosity = Some(Verbosity::Verbose);
                        "Steps: verbose (all rules, full detail)".to_string()
                    }
                    "succinct" => {
                        self.core.state.options.steps_mode = StepsMode::On;
                        self.core.engine.simplifier.set_steps_mode(StepsMode::On);
                        ui_delta.verbosity = Some(Verbosity::Succinct);
                        "Steps: succinct (compact 1-line per step)".to_string()
                    }
                    "normal" => {
                        self.core.state.options.steps_mode = StepsMode::On;
                        self.core.engine.simplifier.set_steps_mode(StepsMode::On);
                        ui_delta.verbosity = Some(Verbosity::Normal);
                        "Steps: normal (default display)".to_string()
                    }
                    "none" => {
                        ui_delta.verbosity = Some(Verbosity::None);
                        "Steps display: none (collection still active)".to_string()
                    }
                    _ => {
                        "Usage: set steps <on|off|compact|verbose|succinct|normal|none>".to_string()
                    }
                }
            }
            "debug" => match parts[2] {
                "on" | "true" | "1" => {
                    self.core.debug_mode = true;
                    "Debug mode ENABLED (pipeline diagnostics after each operation)".to_string()
                }
                "off" | "false" | "0" => {
                    self.core.debug_mode = false;
                    "Debug mode DISABLED".to_string()
                }
                _ => "Usage: set debug <on|off>".to_string(),
            },
            _ => return self.set_help_core().into(),
        };
        CoreResult::with_delta(vec![ReplMsg::info(msg)], ui_delta)
    }

    /// Core: generate set help text as ReplReply
    pub(crate) fn set_help_core(&self) -> ReplReply {
        vec![ReplMsg::output(self.set_help_text())]
    }

    /// Generate set help text as String
    fn set_help_text(&self) -> String {
        use cas_solver::{AutoExpandBinomials, HeuristicPoly, StepsMode};

        let mut s = String::new();
        s.push_str("Pipeline settings:\n");
        s.push_str("  set transform <on|off>         Enable/disable distribution & expansion\n");
        s.push_str("  set rationalize <on|off|0|1|1.5>  Set rationalization level\n");
        s.push_str(
            "  set heuristic_poly <on|off>    Smart polynomial simplification/factorization\n",
        );
        s.push_str(
            "  set autoexpand <on|off>        Force expansion of binomial powers like (x+1)^n\n",
        );
        s.push_str("  set steps <on|off|...>         Step collection and display mode\n");
        s.push_str("  set max-rewrites <N>           Set max total rewrites (safety limit)\n");
        s.push_str(
            "  set debug <on|off>             Show pipeline diagnostics after operations\n\n",
        );
        s.push_str("Current settings:\n");
        s.push_str(&format!(
            "  transform: {}\n",
            if self.core.simplify_options.enable_transform {
                "on"
            } else {
                "off"
            }
        ));
        s.push_str(&format!(
            "  rationalize: {:?}\n",
            self.core.simplify_options.rationalize.auto_level
        ));
        s.push_str(&format!(
            "  heuristic_poly: {}\n",
            if self.core.simplify_options.shared.heuristic_poly == HeuristicPoly::On {
                "on"
            } else {
                "off"
            }
        ));
        s.push_str(&format!(
            "  autoexpand: {}\n",
            if self.core.simplify_options.shared.autoexpand_binomials == AutoExpandBinomials::On {
                "on"
            } else {
                "off"
            }
        ));
        let mode_str = match self.core.state.options.steps_mode {
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
        s.push_str(&format!(
            "  steps: {} (display: {})\n",
            mode_str, verbosity_str
        ));
        s.push_str(&format!(
            "  max-rewrites: {}\n",
            self.core.simplify_options.budgets.max_total_rewrites
        ));
        s.push_str(&format!(
            "  debug: {}",
            if self.core.debug_mode { "on" } else { "off" }
        ));
        s
    }

    /// Legacy wrapper - calls core and prints
    pub(crate) fn handle_help(&self, line: &str) {
        let reply = self.handle_help_core(line);
        self.print_reply(reply);
    }

    /// Core: handle help command, returns ReplReply (no I/O)
    pub(crate) fn handle_help_core(&self, line: &str) -> ReplReply {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 2 {
            return self.print_general_help_core();
        }

        let text = match parts[1] {
            "simplify" => "\
Command: simplify <expr>
Description: Simplifies an expression using the full power of the engine.
             This includes aggressive distribution and other rules that may
             undo factorizations, but guarantee maximum simplification.
Example: simplify (x+1)*(x-1) -> x^2 - 1"
                .to_string(),

            "diff" => "\
Command: diff <expr> <var>
Description: Computes the symbolic derivative of an expression with respect to a variable.
             Supports basic arithmetic, power rule, chain rule, and common functions.
Example: diff(sin(x^2), x) -> 2*x*cos(x^2)"
                .to_string(),

            "sum" => "\
Function: sum(expr, var, start, end)
Description: Evaluates finite summations Σ(var=start to end) expr.
             Supports numeric evaluation and telescoping detection.
Features:
  - Numeric: sum(k, k, 1, 10) -> 55
  - Symbolic end: sum(1/(k*(k+1)), k, 1, n) -> 1 - 1/(n+1)
  - Telescoping: Automatically detects 1/(k*(k+a)) patterns
Examples:
  sum(k^2, k, 1, 5)           -> 55
  sum(1/(k*(k+1)), k, 1, n)   -> 1 - 1/(1+n)
  sum(1/(k*(k+2)), k, 1, n)   -> 1/2 - 1/2/(1+n)"
                .to_string(),

            "product" => "\
Function: product(expr, var, start, end)
Description: Evaluates finite products Π(var=start to end) expr.
             Supports numeric evaluation and telescoping detection.
Features:
  - Numeric: product(k, k, 1, 5) -> 120 (5!)
  - Symbolic end: product((k+1)/k, k, 1, n) -> n+1
  - Telescoping: Detects (k+a)/(k+b) quotient patterns
Examples:
  product(k, k, 1, 5)         -> 120
  product((k+1)/k, k, 1, n)   -> 1 + n
  product((k+1)/k, k, 1, 10)  -> 11"
                .to_string(),

            "gcd" => "\
Function: gcd <a, b>
Description: Computes the Greatest Common Divisor of two integers.
Example: gcd(12, 18) -> 6"
                .to_string(),

            "lcm" => "\
Function: lcm <a, b>
Description: Computes the Least Common Multiple of two integers.
Example: lcm(4, 6) -> 12"
                .to_string(),

            "mod" => "\
Function: mod <a, n>
Description: Computes the remainder of a divided by n (Euclidean modulo).
Example: mod(10, 3) -> 1"
                .to_string(),

            "factors" | "prime_factors" => "\
Function: factors <n>
Description: Computes the prime factorization of an integer.
Example: factors(12) -> 2^2 * 3"
                .to_string(),

            "fact" | "factorial" => "\
Function: fact <n> or <n>!
Description: Computes the factorial of a non-negative integer.
Example: fact(5) -> 120, 5! -> 120"
                .to_string(),

            "choose" | "nCr" => "\
Function: choose <n, k>
Description: Computes the binomial coefficient nCk (combinations).
Example: choose(5, 2) -> 10"
                .to_string(),

            "perm" | "nPr" => "\
Function: perm <n, k>
Description: Computes the number of permutations nPk.
Example: perm(5, 2) -> 20"
                .to_string(),

            "config" => "\
Command: config <subcommand> [args]
Description: Manages CLI configuration.
Subcommands:
  list             Show current configuration
  enable <rule>    Enable a simplification rule
  disable <rule>   Disable a simplification rule
  save             Save configuration to file
  restore          Restore default configuration
Rules:
  distribute       Aggressive distribution (a*(b+c) -> a*b + a*c)
  distribute_constants Safe distribution (-1*(x+y) -> -x-y)
  expand_binomials Expand powers ((a+b)^2 -> a^2+2ab+b^2)
  factor_difference_squares Factor difference of squares (a^2-b^2 -> (a-b)(a+b))"
                .to_string(),

            "subst" => "\
Command: subst <expr>, <target>, <replacement>
Description: Substitutes a pattern with a replacement and simplifies.

Variable substitution:
  subst x^2 + x, x, 3           → 12

Power-aware expression substitution:
  subst x^4 + x^2 + 1, x^2, y   → y² + y + 1
  subst x^3, x^2, y             → y·x (with remainder)
  subst x^6, x^2, y             → y³"
                .to_string(),

            "expand" => "\
Command: expand <expr>
Description: Expands polynomials and products.
Example: expand(x+1)^2 -> x^2 + 2*x + 1"
                .to_string(),

            "factor" => "\
Command: factor <expr>
Description: Factors polynomials.
Example: factor(x^2 - 1) -> (x - 1) * (x + 1)"
                .to_string(),

            "collect" => "\
Command: collect <expr>, <var>
Description: Groups terms by powers of a variable.
Example: collect(a*x + b*x + c, x) -> (a + b) * x + c"
                .to_string(),

            "equiv" => "\
Command: equiv <expr1>, <expr2>
Description: Checks if two expressions are mathematically equivalent.
             Returns true if expr1 - expr2 simplifies to 0."
                .to_string(),

            "solve" => "\
Command: solve <equation>, <var>
Description: Solves an equation for a variable.
Example: solve x + 2 = 5, x -> x = 3"
                .to_string(),

            "steps" => "\
Command: steps <level>
Description: Controls the verbosity of simplification steps.
Levels:
  normal (or on)   Show clarifying steps (Global state). Default.
  succinct              Compact: same steps as normal but 1 line each.
  verbose          Show all steps (Local + Global details).
  none (or off)    Disable step output."
                .to_string(),

            "profile" => "\
Command: profile [subcommand]
Description: Rule profiler for debugging and performance analysis.
Subcommands:
  (none)           Show profiling report
  enable           Enable profiler
  disable          Disable profiler
  clear            Clear statistics
Example: profile enable, then run expressions, then profile"
                .to_string(),

            "visualize" => "\
Command: visualize <expr>
Description: Export expression tree to Graphviz DOT format.
             Generates ast.dot file for rendering.
Example: visualize (x+1)*(x-1)

To render the generated file, use Graphviz in your terminal:
  $ dot -Tsvg ast.dot -o ast.svg
  $ open ast.svg"
                .to_string(),

            "timeline" => "\
Command: timeline <expr>
Description: Export simplification steps to interactive HTML.
             Generates timeline.html with MathJax rendering.
Example: timeline (x+1)^2
         Open timeline.html in browser to view."
                .to_string(),

            "explain" => "\
Command: explain <function>
Description: Provides step-by-step educational explanations of mathematical
             operations. Shows the detailed algorithm steps in Spanish.
Supported functions:
  gcd(a, b)    Greatest Common Divisor using Euclidean algorithm
               Works for both integers and polynomials.
Examples:
  explain gcd(48, 18)
  explain gcd(2*x^2 + 7*x + 3, 2*x^2 + 5*x + 2)"
                .to_string(),

            "det" => "\
Command: det <matrix>
Description: Compute the determinant of a square matrix.
             Supports 1×1, 2×2, and 3×3 matrices.
Examples:
  det [[1, 2], [3, 4]]        → -2
  det [[2]]                    → 2
  det [[1, 2, 3], [4, 5, 6], [7, 8, 9]]"
                .to_string(),

            "transpose" => "\
Command: transpose <matrix>
Description: Transpose a matrix (swap rows and columns).
             Works with any rectangular matrix.
Examples:
  transpose [[1, 2, 3], [4, 5, 6]]
    → [[1, 4], [2, 5], [3, 6]]
  transpose [[1, 2], [3, 4]]
    → [[1, 3], [2, 4]]"
                .to_string(),

            "trace" => "\
Command: trace <matrix>
Description: Compute the trace of a square matrix.
             The trace is the sum of diagonal elements.
Examples:
  trace [[1, 2], [3, 4]]      → 5
  trace [[5, 0, 0], [0, 3, 0], [0, 0, 2]]  → 10"
                .to_string(),

            "rationalize" => "\
Command: rationalize <expr>
Description: Rationalize denominators containing surds (square roots).
             Eliminates irrational numbers from denominators by multiplying
             by the conjugate.
Examples:
  rationalize 1/(1 + sqrt(2))      → √2 - 1
  rationalize 1/(3 - 2*sqrt(5))    → -(3 + 2*√5)/11
  rationalize x/(sqrt(3) + 1)      → x*(√3 - 1)/2"
                .to_string(),

            "status" | "health" => "\
Command: health [on|off|reset|status]
Description: Engine health monitoring and diagnostic test suite.

Subcommands:
  health on                Enable profiler
  health off               Disable profiler
  health reset             Reset profiler stats
  health status            Run diagnostic test suite

Test suite options:
  health status --list             List all test cases
  health status --category <cat>   Run specific category
  health status -c <cat>           Shorthand for --category

Categories: transform, expansion, fractions, rationalization,
            mixed, baseline, roots, powers, stress, all

Examples:
  health status                Run all test categories
  health status -c stress      Run only stress tests
  health status --list         List available tests"
                .to_string(),

            // Session environment commands
            "let" => "\
Command: let <name> = <expr>
Description: Assigns an expression to a variable name.
             The variable can be used in subsequent expressions.
             Substitution is transitive and cycle-safe.

Examples:
  let a = 5
  let b = a + 1        → b becomes 6
  let f = x^2 + 1      → f stores symbolic expression"
                .to_string(),

            "vars" => "\
Command: vars
Description: Lists all defined variables and their values.

Example output:
  a = 5
  b = 6
  f = x^2 + 1"
                .to_string(),

            "clear" => "\
Command: clear [name ...]
Description: Clears variable bindings from the environment.
             Without arguments, clears ALL variables.
             With arguments, clears only the specified variables.

Examples:
  clear           → clears all variables
  clear a b       → clears only a and b"
                .to_string(),

            "reset" => "\
Command: reset
Description: Resets the entire session state.
             Clears all variables AND session history (#ids)."
                .to_string(),

            "history" | "list" => "\
Command: history (or list)
Description: Shows all stored session entries with their #ids.
             Each expression you evaluate is stored with a unique ID.

Example output:
  #1: x + 1
  #2: 2*x - 3
  #3: x + 1 = 5  [Eq]"
                .to_string(),

            "show" => "\
Command: show #<id>
Description: Displays a specific session entry by its ID.

Example:
  show #1         → shows the expression stored as #1"
                .to_string(),

            "del" => "\
Command: del #<id> [#<id> ...]
Description: Deletes session entries by their IDs.
             IDs are never reused after deletion.

Examples:
  del #1          → deletes entry #1
  del #2 #3 #5    → deletes entries #2, #3, and #5"
                .to_string(),

            "poly_gcd" | "pgcd" => "\
Command: poly_gcd(expr1, expr2)
Alias: pgcd(expr1, expr2)
Description: Computes the STRUCTURAL GCD of two polynomial expressions.
             Finds common factors that appear explicitly as multiplicands.
             Does NOT factor expressions to find hidden common factors.

Examples:
  poly_gcd((x+1)*(y+2), (x+1)*(z+3)) → (x + 1)
  poly_gcd((x+1)^3, (x+1)^2)         → (x + 1)²
  poly_gcd(x*g, y*g)                 → g

See also: poly_gcd_exact for algebraic GCD"
                .to_string(),

            "poly_gcd_exact" | "pgcdx" => "\
Command: poly_gcd_exact(expr1, expr2)
Alias: pgcdx(expr1, expr2)
Description: Computes the ALGEBRAIC GCD of two polynomials over ℚ[x₁,...,xₙ].
             Interprets expressions as polynomials and finds the true GCD.
             Uses Euclidean algorithm for univariate, interpolation for multivariate.

Examples:
  poly_gcd_exact(x^2 - 1, x - 1)         → x - 1
  poly_gcd_exact(x^2 - 1, x^2 - 2*x + 1) → x - 1
  poly_gcd_exact(2*x + 2*y, 4*x + 4*y)   → x + y
  poly_gcd_exact(6, 15)                  → 1 (constants over ℚ)

Result is normalized: primitive (GCD of coefficients = 1), positive leading coefficient.

See also: poly_gcd for structural (visible factor) GCD"
                .to_string(),

            "limit" => "\
Command: limit <expr> [, <var> [, <direction>]]

Description: Compute the limit of an expression as a variable approaches infinity.
             Uses polynomial degree comparison for rational functions P(x)/Q(x).

Arguments:
  <expr>       Expression to evaluate the limit of
  <var>        Variable approaching the limit (default: x)
  <direction>  Direction: infinity or -infinity (default: infinity)

Examples:
  limit x^2                      → infinity
  limit (x^2+1)/(2*x^2-3), x     → 1/2
  limit x^3/x^2, x, -infinity    → -infinity
  limit x^2/x^3                  → 0

Behavior:
  - deg(P) < deg(Q): limit = 0
  - deg(P) = deg(Q): limit = leading_coeff(P) / leading_coeff(Q)
  - deg(P) > deg(Q): limit = ±∞ (sign depends on coefficients and approach)

Residuals:
  If the limit cannot be determined (e.g., sin(x)/x, non-polynomial expressions),
  returns limit(...) as a symbolic residual with a warning."
                .to_string(),

            "expand_log" => "\
Command: expand_log <expr>

Description: Expands logarithms using log laws (product, quotient, power rules).
             Requires positivity assumptions for correctness.

Laws applied:
  ln(a*b)   → ln(a) + ln(b)   (product rule)
  ln(a/b)   → ln(a) - ln(b)   (quotient rule)
  ln(a^n)   → n*ln(a)         (power rule)

Examples:
  expand_log ln(x^2 * y)     → 2*ln(x) + ln(y)
  expand_log ln(a/b)        → ln(a) - ln(b)
  expand_log ln(x^3 * y^2)  → 3*ln(x) + 2*ln(y)"
                .to_string(),

            _ => {
                let mut reply = vec![ReplMsg::error(format!("Unknown command: {}", parts[1]))];
                reply.extend(self.print_general_help_core());
                return reply;
            }
        };
        vec![ReplMsg::output(text)]
    }

    /// Core version: returns help text as ReplReply (no I/O)
    pub(crate) fn print_general_help_core(&self) -> ReplReply {
        vec![ReplMsg::output(self.general_help_text())]
    }

    /// Generate general help text as a String
    fn general_help_text(&self) -> String {
        let mut s = String::new();
        s.push_str("Rust CAS Commands:\n\n");

        s.push_str("Basic Operations:\n");
        s.push_str("  <expr>                  Evaluate and simplify an expression\n");
        s.push_str("  simplify <expr>         Aggressive simplification (full power)\n");
        s.push_str("  expand <expr>           Expand polynomials\n");
        s.push_str("  expand_log <expr>       Expand logarithms (log laws)\n");
        s.push_str("  factor <expr>           Factor polynomials\n");
        s.push_str("  collect <expr>, <var>   Group terms by variable\n\n");

        s.push_str("Polynomial GCD:\n");
        s.push_str("  poly_gcd(a, b)          Structural GCD (visible factors)\n");
        s.push_str("  poly_gcd_exact(a, b)    Algebraic GCD over ℚ[x₁,...,xₙ]\n");
        s.push_str("  pgcd                    Alias for poly_gcd\n");
        s.push_str("  pgcdx                   Alias for poly_gcd_exact\n\n");

        s.push_str("Equation Solving:\n");
        s.push_str("  solve <eq>, <var>       Solve equation for variable\n");
        s.push_str("  equiv <e1>, <e2>        Check if two expressions are equivalent\n");
        s.push_str("  subst <expr>, <var>=<val> Substitute a variable and simplify\n\n");

        s.push_str("Calculus:\n");
        s.push_str("  diff <expr>, <var>      Compute symbolic derivative\n");
        s.push_str("  limit <expr>            Compute limit at ±∞ (CLI: expli limit)\n");
        s.push_str("  sum(e, v, a, b)         Finite summation: Σ(v=a to b) e\n");
        s.push_str("  product(e, v, a, b)     Finite product: Π(v=a to b) e\n\n");

        s.push_str("Number Theory:\n");
        s.push_str("  gcd <a, b>              Greatest Common Divisor\n");
        s.push_str("  lcm <a, b>              Least Common Multiple\n");
        s.push_str("  mod <a, n>              Modular arithmetic\n");
        s.push_str("  factors <n>             Prime factorization\n");
        s.push_str("  fact <n>                Factorial (or n!)\n");
        s.push_str("  choose <n, k>           Binomial coefficient (nCk)\n");
        s.push_str("  perm <n, k>             Permutations (nPk)\n\n");

        s.push_str("Matrix Operations:\n");
        s.push_str("  det <matrix>            Compute determinant (up to 3×3)\n");
        s.push_str("  transpose <matrix>      Transpose a matrix\n");
        s.push_str("  trace <matrix>          Compute trace (sum of diagonal)\n\n");

        s.push_str("Analysis & Verification:\n");
        s.push_str("  explain <function>      Show step-by-step explanation\n");
        s.push_str("  telescope <expr>        Prove telescoping identities (Dirichlet kernel)\n");
        s.push_str(
            "  steps <level>           Set step verbosity (normal, succinct, verbose, none)\n\n",
        );

        s.push_str("Visualization & Output:\n");
        s.push_str("  visualize <expr>        Export AST to Graphviz DOT (generates ast.dot)\n");
        s.push_str("  timeline <expr>         Export steps to interactive HTML\n\n");

        s.push_str(
            "  set <option> <value>    Pipeline settings (transform, rationalize, max-rewrites)\n",
        );
        s.push_str(
            "  semantics [set|help]    Semantic settings (domain, value, inv_trig, branch)\n",
        );
        s.push_str("  context [mode]          Context mode (auto, standard, solve, integrate)\n");
        s.push_str("  config <subcmd>         Manage configuration (list, enable, disable...)\n");
        s.push_str("  profile [cmd]           Rule profiler (enable/disable/clear)\n");
        s.push_str("  health [cmd]            Health tracking (on/off/reset/status)\n");
        s.push_str("  help [cmd]              Show this help message or details for a command\n");
        s.push_str("  quit / exit             Exit the REPL\n\n");

        s.push_str("Session Environment:\n");
        s.push_str("  let <name> = <expr>     Assign a variable\n");
        s.push_str("  <name> := <expr>        Alternative assignment syntax\n");
        s.push_str("  vars                    List all defined variables\n");
        s.push_str("  clear [name]            Clear one or all variables\n");
        s.push_str("  reset                   Clear all session state (keeps cache)\n");
        s.push_str("  reset full              Clear all session state AND profile cache\n");
        s.push_str("  budget [N]              Set/show Conditional branching budget (0-3)\n");
        s.push_str("  cache [status|clear]    View or clear profile cache\n");
        s.push_str("  history / list          Show session history (#ids)\n");
        s.push_str("  show #<id>              Display a session entry\n");
        s.push_str("  del #<id> ...           Delete session entries\n\n");

        s.push_str("Type 'help <command>' for more details on a specific command.");
        s
    }

    /// Legacy print_general_help - calls core and prints
    pub(crate) fn print_general_help(&self) {
        let reply = self.print_general_help_core();
        self.print_reply(reply);
    }
}
