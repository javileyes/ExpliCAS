use cas_ast::{Context, SolutionSet};

#[test]
fn detect_solve_variable_prefers_x_then_fallbacks() {
    let mut ctx = Context::new();
    let lhs = cas_parser::parse("2*y + x", &mut ctx).expect("parse lhs");
    let rhs = cas_parser::parse("0", &mut ctx).expect("parse rhs");
    let var = cas_solver::json::detect_solve_variable_eval_json(&ctx, lhs, rhs);
    assert_eq!(var, "x");

    let lhs_only_y = cas_parser::parse("3*y + 1", &mut ctx).expect("parse lhs y");
    let var_y = cas_solver::json::detect_solve_variable_eval_json(&ctx, lhs_only_y, rhs);
    assert_eq!(var_y, "y");

    let lhs_no_vars = cas_parser::parse("3 + 1", &mut ctx).expect("parse lhs const");
    let var_default = cas_solver::json::detect_solve_variable_eval_json(&ctx, lhs_no_vars, rhs);
    assert_eq!(var_default, "x");
}

#[test]
fn format_solution_set_eval_json_matches_contract_strings() {
    let mut ctx = Context::new();
    let x = cas_parser::parse("x", &mut ctx).expect("parse x");
    let residual = cas_parser::parse("x + 1", &mut ctx).expect("parse residual");

    let empty = cas_solver::json::format_solution_set_eval_json(&ctx, &SolutionSet::Empty);
    assert_eq!(empty, "No solution");

    let all = cas_solver::json::format_solution_set_eval_json(&ctx, &SolutionSet::AllReals);
    assert_eq!(all, "All real numbers");

    let discrete =
        cas_solver::json::format_solution_set_eval_json(&ctx, &SolutionSet::Discrete(vec![x]));
    assert_eq!(discrete, "{ x }");

    let residual_display =
        cas_solver::json::format_solution_set_eval_json(&ctx, &SolutionSet::Residual(residual));
    assert!(
        residual_display.starts_with("Solve: "),
        "expected residual solve string, got: {residual_display}"
    );
}

#[test]
fn solution_set_to_latex_eval_json_matches_contract_strings() {
    let mut ctx = Context::new();
    let x = cas_parser::parse("x", &mut ctx).expect("parse x");

    let empty = cas_solver::json::solution_set_to_latex_eval_json(&ctx, &SolutionSet::Empty);
    assert_eq!(empty, r"\emptyset");

    let all = cas_solver::json::solution_set_to_latex_eval_json(&ctx, &SolutionSet::AllReals);
    assert_eq!(all, r"\mathbb{R}");

    let discrete =
        cas_solver::json::solution_set_to_latex_eval_json(&ctx, &SolutionSet::Discrete(vec![x]));
    assert!(discrete.contains(r"\left\{"));
    assert!(discrete.contains(r"\right\}"));
}
