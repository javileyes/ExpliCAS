use cas_ast::Context;

#[test]
fn infer_solve_variable_returns_single_free_variable() {
    let mut ctx = Context::new();
    let expr = cas_parser::parse("x^2 + 2*x + 1", &mut ctx).expect("parse expression");

    let inferred = cas_solver::infer_solve_variable(&ctx, expr).expect("single variable");
    assert_eq!(inferred, Some("x".to_string()));
}

#[test]
fn infer_solve_variable_ignores_known_constants() {
    let mut ctx = Context::new();
    let expr = cas_parser::parse("pi + e + i + 3", &mut ctx).expect("parse expression");

    let inferred = cas_solver::infer_solve_variable(&ctx, expr).expect("no free variables");
    assert_eq!(inferred, None);
}

#[test]
fn infer_solve_variable_returns_sorted_ambiguous_variables() {
    let mut ctx = Context::new();
    let expr = cas_parser::parse("y + x + _tmp", &mut ctx).expect("parse expression");

    let inferred = cas_solver::infer_solve_variable(&ctx, expr).expect_err("ambiguous variables");
    assert_eq!(inferred, vec!["x".to_string(), "y".to_string()]);
}
