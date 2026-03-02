/// Resolve detailed help text for a specific command/topic.
pub fn help_topic_text(topic: &str) -> Option<String> {
    let text = match topic {
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
        _ => return None,
    };
    Some(text)
}

#[cfg(test)]
mod tests {
    use super::help_topic_text;

    #[test]
    fn help_topic_text_returns_known_topic() {
        let text = help_topic_text("simplify").expect("topic text");
        assert!(text.contains("Command: simplify"));
    }

    #[test]
    fn help_topic_text_returns_none_for_unknown_topic() {
        assert!(help_topic_text("unknown_topic").is_none());
    }
}
