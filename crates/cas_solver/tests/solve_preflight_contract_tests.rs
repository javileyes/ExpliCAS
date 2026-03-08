use cas_ast::{Equation, RelOp};
use cas_parser::parse;
use cas_solver::{solve_with_display_steps, Simplifier, SolverOptions};

fn parse_eq(simplifier: &mut Simplifier, lhs: &str, rhs: &str) -> Equation {
    Equation {
        lhs: parse(lhs, &mut simplifier.context).expect("lhs parse should succeed"),
        rhs: parse(rhs, &mut simplifier.context).expect("rhs parse should succeed"),
        op: RelOp::Eq,
    }
}

#[test]
fn solve_with_display_steps_collects_required_conditions_from_preflight() {
    let mut simplifier = Simplifier::new();
    let equation = parse_eq(&mut simplifier, "sqrt(x)", "0");

    let (_solutions, _steps, diagnostics) =
        solve_with_display_steps(&equation, "x", &mut simplifier, SolverOptions::default())
            .expect("solve should succeed");

    assert!(
        !diagnostics.required.is_empty(),
        "expected required conditions from sqrt(x) preflight",
    );
}

#[test]
fn solve_with_display_steps_keeps_required_empty_for_plain_linear_equation() {
    let mut simplifier = Simplifier::new();
    let equation = parse_eq(&mut simplifier, "x", "0");

    let (_solutions, _steps, diagnostics) =
        solve_with_display_steps(&equation, "x", &mut simplifier, SolverOptions::default())
            .expect("solve should succeed");

    assert!(
        diagnostics.required.is_empty(),
        "did not expect required conditions for x = 0",
    );
}
