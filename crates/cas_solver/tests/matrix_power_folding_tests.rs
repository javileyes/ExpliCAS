//! End-to-end contract tests for integer matrix powers (2026-07-14, F1 of
//! docs/AUDITORIA_FRONTERA_2026-07-14.md). `try_matrix_power_expr` multiplies the
//! matrix by itself repeatedly via `Matrix::multiply`, which built raw
//! (unfolded) `Mul`/`Add` trees for each dot product. Numeric entries never
//! collapsed, so after a few multiplications each entry was an exponentially
//! large symbolic tree — `[[1,1],[1,0]]^14` took seconds and left one entry as a
//! multi-thousand-term sum; larger exponents hung / hit depth overflow. Eager
//! numeric folding in `Matrix::multiply` keeps integer matrices exact and
//! instant. The Fibonacci matrix `[[1,1],[1,0]]^n = [[F(n+1),F(n)],[F(n),F(n-1)]]`
//! gives an exact reference.

use cas_parser::parse;
use cas_solver::runtime::{Engine, EvalAction, EvalOptions, EvalRequest, EvalResult};

fn eval_render(input: &str) -> String {
    let mut engine = Engine::new();
    let parsed = parse(input, &mut engine.simplifier.context).expect("parse");
    let req = EvalRequest {
        raw_input: input.to_string(),
        parsed,
        action: EvalAction::Simplify,
        auto_store: false,
    };
    let output = engine
        .eval_stateless(EvalOptions::default(), req)
        .expect("eval");
    let id = match output.result {
        EvalResult::Expr(id) => id,
        other => panic!("expected Expr, got {other:?}"),
    };
    format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: &engine.simplifier.context,
            id
        }
    )
}

#[test]
fn fibonacci_matrix_power_folds_to_exact_integers() {
    // [[1,1],[1,0]]^n = [[F(n+1), F(n)], [F(n), F(n-1)]].
    assert_eq!(eval_render("[[1,1],[1,0]]^5"), "[[8, 5], [5, 3]]");
    assert_eq!(eval_render("[[1,1],[1,0]]^10"), "[[89, 55], [55, 34]]");
    // ^14 previously left [1][1] as a giant unfolded Add-tree.
    assert_eq!(eval_render("[[1,1],[1,0]]^14"), "[[610, 377], [377, 233]]");
    // Large exponents previously hung; F(51)=20365011074.
    assert_eq!(
        eval_render("[[1,1],[1,0]]^50"),
        "[[20365011074, 12586269025], [12586269025, 7778742049]]"
    );
}

#[test]
fn other_numeric_matrix_powers_are_exact() {
    assert_eq!(eval_render("[[2,0],[0,2]]^3"), "[[8, 0], [0, 8]]");
    assert_eq!(eval_render("[[1,2],[3,4]]^2"), "[[7, 10], [15, 22]]");
    // Multiplication (not a power) stays exact too.
    assert_eq!(
        eval_render("[[1,2],[3,4]]*[[5,6],[7,8]]"),
        "[[19, 22], [43, 50]]"
    );
}

#[test]
fn identity_multiply_preserves_symbolic_matrix() {
    // Symbolic entries pass through the raw path; numeric 0/1 fold — the identity
    // product is the original symbolic matrix.
    assert_eq!(
        eval_render("[[a,b],[c,d]]*[[1,0],[0,1]]"),
        "[[a, b], [c, d]]"
    );
}
