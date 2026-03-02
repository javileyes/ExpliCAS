/// Generate general REPL help text.
pub fn general_help_text() -> String {
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
    s.push_str("  semantics [set|help]    Semantic settings (domain, value, inv_trig, branch)\n");
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

#[cfg(test)]
mod tests {
    use super::general_help_text;

    #[test]
    fn general_help_text_contains_core_sections() {
        let text = general_help_text();
        assert!(text.contains("Rust CAS Commands:"));
        assert!(text.contains("Basic Operations:"));
        assert!(text.contains("Session Environment:"));
    }
}
