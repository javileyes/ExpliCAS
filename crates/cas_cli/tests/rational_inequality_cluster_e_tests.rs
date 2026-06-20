//! Round-4 audit Cluster E: rational inequalities `f/g (op) c` with the solve
//! variable only in the (linear) denominator must take the denominator-sign case
//! split, excluding the pole, instead of demoting to the boundary equation and
//! emitting a single ray disjoint from the truth.

use cas_ast::{BoundType, Equation, Interval, RelOp, SolutionSet};
use cas_formatter::DisplayExpr;
use cas_parser::parse;
use cas_solver::api::solve;
use cas_solver::runtime::Simplifier;

fn render_interval(simplifier: &mut Simplifier, interval: &Interval) -> String {
    let min = simplifier.simplify(interval.min).0;
    let max = simplifier.simplify(interval.max).0;
    let lo = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: min
        }
    );
    let hi = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: max
        }
    );
    let lb = if interval.min_type == BoundType::Closed {
        "["
    } else {
        "("
    };
    let rb = if interval.max_type == BoundType::Closed {
        "]"
    } else {
        ")"
    };
    format!("{lb}{lo}, {hi}{rb}")
}

/// Render a solution set to the canonical textual form, e.g. `(2, 3)` or
/// `(-infinity, 2) U (3, infinity)`.
fn render_set(simplifier: &mut Simplifier, set: &SolutionSet) -> String {
    match set {
        SolutionSet::Empty => "Empty".to_string(),
        SolutionSet::AllReals => "AllReals".to_string(),
        SolutionSet::Continuous(interval) => render_interval(simplifier, interval),
        SolutionSet::Union(intervals) => intervals
            .iter()
            .map(|i| render_interval(simplifier, i))
            .collect::<Vec<_>>()
            .join(" U "),
        other => format!("{other:?}"),
    }
}

fn solve_render(input: &str, op: RelOp, rhs_src: &str) -> String {
    let mut simplifier = Simplifier::with_default_rules();
    let lhs = parse(input, &mut simplifier.context).unwrap();
    let rhs = parse(rhs_src, &mut simplifier.context).unwrap();
    let eq = Equation { lhs, rhs, op };
    let (result, _) = solve(&eq, "x", &mut simplifier).expect("solve failed");
    render_set(&mut simplifier, &result)
}

#[test]
fn cluster_e_five_audit_probes() {
    // 1/(x-2) > 1  ->  (2, 3)
    assert_eq!(solve_render("1 / (x - 2)", RelOp::Gt, "1"), "(2, 3)");
    // 1/(x-2) < 1  ->  (-inf, 2) U (3, inf)
    assert_eq!(
        solve_render("1 / (x - 2)", RelOp::Lt, "1"),
        "(-infinity, 2) U (3, infinity)"
    );
    // 1/(x+1) > 2  ->  (-1, -1/2)
    assert_eq!(solve_render("1 / (x + 1)", RelOp::Gt, "2"), "(-1, -1/2)");
    // 2/(x-1) > 1  ->  (1, 3)
    assert_eq!(solve_render("2 / (x - 1)", RelOp::Gt, "1"), "(1, 3)");
    // 1/(x-2) < -1  ->  (1, 2)
    assert_eq!(solve_render("1 / (x - 2)", RelOp::Lt, "-1"), "(1, 2)");
}

#[test]
fn cluster_e_excludes_pole_and_handles_non_strict_endpoints() {
    // The pole is ALWAYS open, but a numerator-root endpoint follows the operator:
    // 1/(x-2) >= 1  ->  (2, 3]  (closed at the boundary root 3, open at the pole 2)
    assert_eq!(solve_render("1 / (x - 2)", RelOp::Geq, "1"), "(2, 3]");
    // 2/(x-1) <= 1  ->  (-inf, 1) U [3, inf)  (open at pole 1, closed at root 3)
    assert_eq!(
        solve_render("2 / (x - 1)", RelOp::Leq, "1"),
        "(-infinity, 1) U [3, infinity)"
    );
}

#[test]
fn cluster_e_negative_fraction_rhs_is_sign_correct() {
    // 1/(x-2) > -1/2  ->  (-inf, 0) U (2, inf)
    // A negative fractional RHS must fold sign-correctly into p = f - c*g.
    assert_eq!(
        solve_render("1 / (x - 2)", RelOp::Gt, "-1/2"),
        "(-infinity, 0) U (2, infinity)"
    );
}

#[test]
fn cluster_e_zero_rhs_guards_unchanged() {
    // c = 0 is NOT folded (kept on the legacy path); these were already correct
    // and must stay so.
    assert_eq!(
        solve_render("1 / x", RelOp::Gt, "0"),
        "(0, infinity)",
        "1/x > 0 guard"
    );
    assert_eq!(
        solve_render("1 / x", RelOp::Lt, "0"),
        "(-infinity, 0)",
        "1/x < 0 guard"
    );
}

#[test]
fn cluster_e_variable_in_numerator_route_unchanged() {
    // The already-correct numerator-side split must be untouched.
    assert_eq!(
        solve_render("(x - 1) / (x - 3)", RelOp::Geq, "0"),
        "(-infinity, 1] U (3, infinity)"
    );
    assert_eq!(solve_render("(3 - x) / (x - 2)", RelOp::Gt, "0"), "(2, 3)");
}
