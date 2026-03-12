use cas_ast::{Case, ConditionPredicate, ConditionSet, Context, SolutionSet, SolveResult};
use cas_solver::command_api::solve::{display_solution_set, is_pure_residual_otherwise};

#[test]
fn pure_residual_otherwise_detection_matches_contract() {
    let mut ctx = Context::new();
    let x = ctx.var("x");

    let residual_case = Case::with_result(
        ConditionSet::empty(),
        SolveResult::solved(SolutionSet::Residual(x)),
    );
    assert!(is_pure_residual_otherwise(&residual_case));

    let non_residual_case = Case::new(
        ConditionSet::single(ConditionPredicate::NonZero(x)),
        SolutionSet::Discrete(vec![x]),
    );
    assert!(!is_pure_residual_otherwise(&non_residual_case));
}

#[test]
fn display_solution_set_skips_pure_residual_otherwise_case() {
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let one = ctx.num(1);

    let guarded = Case::new(
        ConditionSet::single(ConditionPredicate::NonZero(x)),
        SolutionSet::Discrete(vec![one]),
    );
    let otherwise_residual = Case::with_result(
        ConditionSet::empty(),
        SolveResult::solved(SolutionSet::Residual(x)),
    );

    let rendered = display_solution_set(
        &ctx,
        &SolutionSet::Conditional(vec![guarded, otherwise_residual]),
    );
    assert!(rendered.contains("if "));
    assert!(!rendered.contains("otherwise:"));
}

#[test]
fn display_solution_set_formats_interval_and_discrete_sets() {
    let mut ctx = Context::new();
    let zero = ctx.num(0);
    let one = ctx.num(1);
    let two = ctx.num(2);

    let discrete = display_solution_set(&ctx, &SolutionSet::Discrete(vec![one, two]));
    assert_eq!(discrete, "{ 1, 2 }");

    let interval = display_solution_set(
        &ctx,
        &SolutionSet::Continuous(cas_ast::Interval::open(zero, one)),
    );
    assert_eq!(interval, "(0, 1)");
}
