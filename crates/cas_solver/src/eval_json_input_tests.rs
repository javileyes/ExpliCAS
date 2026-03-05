use crate::eval_json_input_variable::detect_solve_variable_eval_json;

#[test]
fn detect_solve_variable_prefers_x_when_ambiguous() {
    let mut ctx = cas_ast::Context::new();
    let statement = cas_parser::parse_statement("x + y = 0", &mut ctx).expect("parse");
    let (lhs, rhs) = match statement {
        cas_parser::Statement::Equation(eq) => (eq.lhs, eq.rhs),
        _ => panic!("expected equation"),
    };
    let var = detect_solve_variable_eval_json(&mut ctx, lhs, rhs);
    assert_eq!(var, "x");
}

#[test]
fn detect_solve_variable_falls_back_to_preferred_order() {
    let mut ctx = cas_ast::Context::new();
    let statement = cas_parser::parse_statement("z + y = 0", &mut ctx).expect("parse");
    let (lhs, rhs) = match statement {
        cas_parser::Statement::Equation(eq) => (eq.lhs, eq.rhs),
        _ => panic!("expected equation"),
    };
    let var = detect_solve_variable_eval_json(&mut ctx, lhs, rhs);
    assert_eq!(var, "y");
}
