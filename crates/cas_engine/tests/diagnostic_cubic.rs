use cas_engine::solver::solve;
/// Diagnostic test: understand why x^3 - x = 0 + T1 identities → non-discrete
use cas_engine::Simplifier;
use cas_parser::parse;

fn solve_and_print(label: &str, eq_str: &str, var: &str) {
    let mut simplifier = Simplifier::with_default_rules();
    simplifier.set_collect_steps(false);

    let parts: Vec<&str> = eq_str.splitn(2, '=').collect();
    let lhs = parse(parts[0].trim(), &mut simplifier.context).unwrap();
    let rhs = parse(parts[1].trim(), &mut simplifier.context).unwrap();
    let eq = cas_ast::Equation {
        lhs,
        rhs,
        op: cas_ast::RelOp::Eq,
    };

    match solve(&eq, var, &mut simplifier) {
        Ok((set, _)) => eprintln!("  {}: {:?}", label, set),
        Err(e) => eprintln!("  {}: ERROR {:?}", label, e),
    }
}

#[test]
fn diagnostic_cubic_failures() {
    eprintln!();
    eprintln!("=== Cubic diagnostic ===");

    // Base case: x^3 - x = 0  (should give Discrete {-1, 0, 1})
    solve_and_print("BASE", "x^3 - x = 0", "x");

    eprintln!();

    // Case #63: + (exp(2*x)/exp(x) ≡ exp(x)) → lhs + exp(2*x)/exp(x) = rhs + exp(x)
    solve_and_print("#63 trans", "x^3 - x + exp(2*x)/exp(x) = 0 + exp(x)", "x");

    // Case #83: + ((x^2-1)/(x+1) ≡ x-1) → lhs + (x^2-1)/(x+1) = rhs + (x-1)
    solve_and_print(
        "#83 trans",
        "x^3 - x + (x^2 - 1)/(x + 1) = 0 + (x - 1)",
        "x",
    );

    // Case #219: + (x^2 + 2*x ≡ x*(x+2))
    solve_and_print("#219 trans", "x^3 - x + x^2 + 2*x = 0 + x*(x+2)", "x");

    // Case #255: + (x^3 - 1 ≡ (x-1)*(x^2+x+1))
    solve_and_print(
        "#255 trans",
        "x^3 - x + x^3 - 1 = 0 + (x-1)*(x^2 + x + 1)",
        "x",
    );

    // Case #340: + (e^(ln(x)) ≡ x)
    solve_and_print("#340 trans", "x^3 - x + exp(ln(x)) = 0 + x", "x");

    // Case #388: T0 + ((-1)^5 ≡ -1) → constant identity, should simplify away
    solve_and_print("#388 trans", "x^3 - x + (-1)^5 = 0 + (-1)", "x");

    // Case #450: T0 + (csc(pi/4) ≡ sqrt(2))
    solve_and_print("#450 trans", "x^3 - x + csc(pi/4) = 0 + sqrt(2)", "x");

    // Case #488: T0 + (sin(pi/10) ≡ (sqrt(5)-1)/4)
    solve_and_print(
        "#488 trans",
        "x^3 - x + sin(pi/10) = 0 + (sqrt(5) - 1)/4",
        "x",
    );
}
