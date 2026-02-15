/// Diagnostic test: cycle and isolation failures
use cas_engine::engine::Simplifier;
use cas_engine::solver::solve;
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
fn diagnostic_cycle_failures() {
    eprintln!();
    eprintln!("=== Cycle diagnostic ===");

    // #7: x^2+1=0 + (x^(1/2)*x^(1/3) ≡ x^(5/6))
    solve_and_print("x^2+1 base", "x^2 + 1 = 0", "x");
    solve_and_print("#7 trans", "x^2 + 1 + x^(1/2)*x^(1/3) = 0 + x^(5/6)", "x");

    // #109: x^2-6x+9=0 + (sqrt(x^3) ≡ x*sqrt(x))
    solve_and_print("x^2-6x+9 base", "x^2 - 6*x + 9 = 0", "x");
    solve_and_print(
        "#109 trans",
        "x^2 - 6*x + 9 + sqrt(x^3) = 0 + x*sqrt(x)",
        "x",
    );

    // #315: (x+1)*(x-1)=0 + (sqrt(x^3) ≡ x*sqrt(x))
    solve_and_print("(x+1)(x-1) base", "(x + 1)*(x - 1) = 0", "x");
    solve_and_print(
        "#315 trans",
        "(x + 1)*(x - 1) + sqrt(x^3) = 0 + x*sqrt(x)",
        "x",
    );

    // #353: (x-2)*(x+3)=0 + (x^(1/2)*x^(1/3) ≡ x^(5/6))
    solve_and_print("(x-2)(x+3) base", "(x - 2)*(x + 3) = 0", "x");
    solve_and_print(
        "#353 trans",
        "(x - 2)*(x + 3) + x^(1/2)*x^(1/3) = 0 + x^(5/6)",
        "x",
    );

    // #375: x^2+1=0 + (sqrt(x^3) ≡ x*sqrt(x))
    solve_and_print("#375 trans", "x^2 + 1 + sqrt(x^3) = 0 + x*sqrt(x)", "x");
}

#[test]
fn diagnostic_isolation_failures() {
    eprintln!();
    eprintln!("=== Isolation diagnostic ===");

    // #76: exp(x)=1 + ((x+1)*(x+2) ≡ x^2+3*x+2)
    solve_and_print("exp(x)=1 base", "exp(x) = 1", "x");
    solve_and_print("#76 trans", "exp(x) + (x+1)*(x+2) = 1 + x^2 + 3*x + 2", "x");

    // #298: 2^x=8 + (x^(-1/2) ≡ 1/sqrt(x))
    solve_and_print("2^x=8 base", "2^x = 8", "x");
    solve_and_print("#298 trans", "2^x + x^(-1/2) = 8 + 1/sqrt(x)", "x");

    // #339: exp(x)=1 + (tan(arcsin(x)) ≡ x/sqrt(1-x^2))
    solve_and_print(
        "#339 trans",
        "exp(x) + tan(arcsin(x)) = 1 + x/sqrt(1 - x^2)",
        "x",
    );
}
