//! Contract tests for affine-in-tan equations keeping periodicity (2026-07-10).
//! The simplifier folds `tan(x) + 1` into `(sin(x) + cos(x)) / cos(x)` BEFORE the
//! periodic handler's entry simplify, destroying the affine structure its peel
//! matches — so `solve(tan(x) + 1 = 2)` fell to the principal-only arctan
//! isolation and reported `{π/4}` with no `+kπ` family. The peel now also probes
//! the RAW equation sides (the work-on-the-raw-tree lesson).

use cas_ast::{Equation, RelOp};
use cas_parser::parse;
use cas_solver::api::solve;
use cas_solver::command_api::solve::display_solution_set;
use cas_solver::runtime::Simplifier;

fn solve_display(lhs: &str, rhs: &str) -> String {
    let mut simplifier = Simplifier::with_default_rules();
    let lhs = parse(lhs, &mut simplifier.context).expect("parse lhs");
    let rhs = parse(rhs, &mut simplifier.context).expect("parse rhs");
    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let (set, _) = solve(&eq, "x", &mut simplifier).expect("solve");
    display_solution_set(&simplifier.context, &set)
}

#[test]
fn additive_tan_forms_keep_the_periodic_family() {
    let quarter = solve_display("tan(x) + 1", "2");
    assert!(
        quarter.contains("k·pi") && quarter.contains("1/4"),
        "tan(x)+1=2 must keep the +k*pi family, got {quarter}"
    );
    assert_eq!(solve_display("2*tan(x) + 1", "3"), quarter);
    assert_eq!(solve_display("3 - tan(x)", "2"), quarter);
    let negative = solve_display("tan(x) + 1", "0");
    assert!(
        negative.contains("k·pi") && negative.contains("-1/4"),
        "tan(x)+1=0 must keep the family, got {negative}"
    );
    let scaled = solve_display("tan(2*x) + 1", "2");
    assert!(
        scaled.contains("1/8") && scaled.contains("k·1/2"),
        "tan(2x)+1=2 must map the family back through the slope, got {scaled}"
    );
}

#[test]
fn sibling_trig_owners_are_unchanged() {
    assert_eq!(
        solve_display("tan(x)", "1"),
        solve_display("tan(x) + 1", "2")
    );
    // sin/cos affine forms never regressed (their structure survives simplify).
    let sin_form = solve_display("sin(x) + 1", "3/2");
    assert!(sin_form.contains("1/6") && sin_form.contains("5/6"));
    // The harmonic-addition owner keeps sin+cos.
    let harmonic = solve_display("sin(x) + cos(x)", "1");
    assert!(harmonic.contains("k·2"));
}
