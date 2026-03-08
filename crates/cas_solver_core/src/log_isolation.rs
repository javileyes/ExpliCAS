use crate::isolation_utils::contains_var;
use cas_ast::{Context, Equation, Expr, ExprId, RelOp};

/// Didactic payload for one logarithm isolation rewrite.
#[derive(Debug, Clone, PartialEq)]
pub struct LogIsolationStep {
    pub description: String,
    pub equation_after: Equation,
}

/// Planned log-isolation rewrite with equation + didactic payload.
#[derive(Debug, Clone, PartialEq)]
pub struct LogIsolationRewritePlan {
    pub equation: Equation,
    pub items: Vec<LogIsolationExecutionItem>,
}

/// Solved payload for one logarithm-isolation rewrite.
#[derive(Debug, Clone, PartialEq)]
pub struct LogIsolationSolved<T> {
    pub rewrite: LogIsolationRewritePlan,
    pub solved: T,
}

/// One executable log-isolation item aligned with didactic payload.
#[derive(Debug, Clone, PartialEq)]
pub struct LogIsolationExecutionItem {
    pub equation: Equation,
    pub description: String,
}

impl LogIsolationExecutionItem {
    /// User-facing narration for this execution item.
    pub fn description(&self) -> &str {
        &self.description
    }
}

/// Collect log-isolation didactic steps in display order.
pub fn collect_log_isolation_didactic_steps(
    plan: &LogIsolationRewritePlan,
) -> Vec<LogIsolationStep> {
    plan.items
        .iter()
        .cloned()
        .map(|item| LogIsolationStep {
            description: item.description,
            equation_after: item.equation,
        })
        .collect()
}

/// Collect log-isolation execution items in display order.
pub fn collect_log_isolation_execution_items(
    plan: &LogIsolationRewritePlan,
) -> Vec<LogIsolationExecutionItem> {
    plan.items.clone()
}

/// Return the first log-isolation execution item, if any.
pub fn first_log_isolation_execution_item(
    plan: &LogIsolationRewritePlan,
) -> Option<LogIsolationExecutionItem> {
    collect_log_isolation_execution_items(plan)
        .into_iter()
        .next()
}

/// Solve one planned logarithm-isolation rewrite with a caller-provided solver.
pub fn solve_log_isolation_rewrite_with<E, T, FSolve>(
    rewrite: LogIsolationRewritePlan,
    mut solve_rewrite: FSolve,
) -> Result<LogIsolationSolved<T>, E>
where
    FSolve: FnMut(&Equation) -> Result<T, E>,
{
    let solved = solve_rewrite(&rewrite.equation)?;
    Ok(LogIsolationSolved { rewrite, solved })
}

/// Solve one planned logarithm-isolation rewrite while passing the aligned
/// optional execution item to the solver callback.
pub fn solve_log_isolation_rewrite_with_item<E, T, FSolve>(
    rewrite: LogIsolationRewritePlan,
    mut solve_rewrite: FSolve,
) -> Result<LogIsolationSolved<T>, E>
where
    FSolve: FnMut(Option<LogIsolationExecutionItem>, &Equation) -> Result<T, E>,
{
    let item = first_log_isolation_execution_item(&rewrite);
    let solved = solve_rewrite(item, &rewrite.equation)?;
    Ok(LogIsolationSolved { rewrite, solved })
}

/// Solve one planned logarithm-isolation rewrite end-to-end while optionally
/// collecting the first didactic execution item.
pub fn solve_log_isolation_rewrite_pipeline_with_item<E, S, FSolve, FMap>(
    rewrite: LogIsolationRewritePlan,
    include_item: bool,
    mut solve_rewritten: FSolve,
    mut map_item_to_step: FMap,
) -> Result<LogIsolationSolved<(cas_ast::SolutionSet, Vec<S>)>, E>
where
    FSolve: FnMut(&Equation) -> Result<(cas_ast::SolutionSet, Vec<S>), E>,
    FMap: FnMut(LogIsolationExecutionItem) -> S,
{
    let first_item = first_log_isolation_execution_item(&rewrite);
    let (solution_set, mut substeps) = solve_rewritten(&rewrite.equation)?;
    let mut steps = Vec::with_capacity(substeps.len() + usize::from(include_item));
    if include_item {
        if let Some(item) = first_item {
            steps.push(map_item_to_step(item));
        }
    }
    steps.append(&mut substeps);
    Ok(LogIsolationSolved {
        rewrite,
        solved: (solution_set, steps),
    })
}

/// Execute log-isolation planning + solve pipeline while optionally collecting
/// the first didactic execution item.
///
/// Returns `None` when log isolation cannot be planned for the given variable.
#[allow(clippy::type_complexity)]
pub fn execute_log_isolation_pipeline_with_item_with<E, S, FPlan, FSolve, FMap>(
    include_item: bool,
    mut plan_rewrite: FPlan,
    solve_rewritten: FSolve,
    map_item_to_step: FMap,
) -> Option<Result<LogIsolationSolved<(cas_ast::SolutionSet, Vec<S>)>, E>>
where
    FPlan: FnMut() -> Option<LogIsolationRewritePlan>,
    FSolve: FnMut(&Equation) -> Result<(cas_ast::SolutionSet, Vec<S>), E>,
    FMap: FnMut(LogIsolationExecutionItem) -> S,
{
    let rewrite = plan_rewrite()?;
    Some(solve_log_isolation_rewrite_pipeline_with_item(
        rewrite,
        include_item,
        solve_rewritten,
        map_item_to_step,
    ))
}

/// Execute log-isolation planning + solve pipeline returning plain strategy
/// output `(SolutionSet, steps)`.
///
/// Returns `None` when log isolation cannot be planned for the given variable.
#[allow(clippy::type_complexity)]
pub fn execute_log_isolation_result_pipeline_with_item_with<E, S, FPlan, FSolve, FMap>(
    include_item: bool,
    plan_rewrite: FPlan,
    solve_rewritten: FSolve,
    map_item_to_step: FMap,
) -> Option<Result<(cas_ast::SolutionSet, Vec<S>), E>>
where
    FPlan: FnMut() -> Option<LogIsolationRewritePlan>,
    FSolve: FnMut(&Equation) -> Result<(cas_ast::SolutionSet, Vec<S>), E>,
    FMap: FnMut(LogIsolationExecutionItem) -> S,
{
    let solved = execute_log_isolation_pipeline_with_item_with(
        include_item,
        plan_rewrite,
        solve_rewritten,
        map_item_to_step,
    )?;
    Some(solved.map(|payload| payload.solved))
}

/// Execute log-isolation planning + solve pipeline returning plain strategy
/// output `(SolutionSet, steps)`, or map non-plannable cases into caller
/// error type via callback.
#[allow(clippy::type_complexity)]
pub fn execute_log_isolation_result_pipeline_or_else_with<E, S, FPlan, FSolve, FMap, FError>(
    include_item: bool,
    plan_rewrite: FPlan,
    solve_rewritten: FSolve,
    map_item_to_step: FMap,
    not_plannable_error: FError,
) -> Result<(cas_ast::SolutionSet, Vec<S>), E>
where
    FPlan: FnMut() -> Option<LogIsolationRewritePlan>,
    FSolve: FnMut(&Equation) -> Result<(cas_ast::SolutionSet, Vec<S>), E>,
    FMap: FnMut(LogIsolationExecutionItem) -> S,
    FError: FnOnce() -> E,
{
    match execute_log_isolation_result_pipeline_with_item_with(
        include_item,
        plan_rewrite,
        solve_rewritten,
        map_item_to_step,
    ) {
        Some(result) => result,
        None => Err(not_plannable_error()),
    }
}

/// Planned transformation for isolating a logarithmic equation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogIsolationPlan {
    /// `log(base, arg) = rhs` -> `arg = base^rhs`
    SolveArgument { lhs: ExprId, rhs: ExprId },
    /// `log(base, arg) = rhs` -> `base = arg^(1/rhs)`
    SolveBase { lhs: ExprId, rhs: ExprId },
}

impl LogIsolationPlan {
    /// Convert a plan into the target equation preserving the incoming relation.
    pub fn into_equation(self, op: RelOp) -> Equation {
        match self {
            Self::SolveArgument { lhs, rhs } | Self::SolveBase { lhs, rhs } => {
                Equation { lhs, rhs, op }
            }
        }
    }

    /// Build didactic narration for the selected log-isolation rewrite.
    pub fn step_description(self, base_display: &str) -> String {
        match self {
            Self::SolveArgument { .. } => {
                format!("Exponentiate both sides with base {}", base_display)
            }
            Self::SolveBase { .. } => "Isolate base of logarithm".to_string(),
        }
    }
}

/// Build display-ready step payload for a logarithm isolation plan.
pub fn build_log_isolation_step_with<F>(
    plan: LogIsolationPlan,
    ctx: &Context,
    base: ExprId,
    op: RelOp,
    mut render_expr: F,
) -> LogIsolationStep
where
    F: FnMut(&Context, ExprId) -> String,
{
    let base_desc = render_expr(ctx, base);
    LogIsolationStep {
        description: plan.step_description(&base_desc),
        equation_after: plan.into_equation(op),
    }
}

/// Plan logarithm isolation and build its didactic step in one call.
pub fn plan_log_isolation_step(
    ctx: &mut Context,
    base: ExprId,
    arg: ExprId,
    rhs: ExprId,
    var: &str,
    op: RelOp,
    base_display: &str,
) -> Option<LogIsolationRewritePlan> {
    plan_log_isolation_step_with(ctx, base, arg, rhs, var, op, |_, _| {
        base_display.to_string()
    })
}

/// Plan logarithm isolation and build its didactic step by rendering the base
/// expression with a caller-provided formatter.
pub fn plan_log_isolation_step_with<F>(
    ctx: &mut Context,
    base: ExprId,
    arg: ExprId,
    rhs: ExprId,
    var: &str,
    op: RelOp,
    mut render_expr: F,
) -> Option<LogIsolationRewritePlan>
where
    F: FnMut(&Context, ExprId) -> String,
{
    let plan = plan_log_isolation(ctx, base, arg, rhs, var)?;
    let step = build_log_isolation_step_with(plan, ctx, base, op, |core_ctx, id| {
        render_expr(core_ctx, id)
    });
    let equation = step.equation_after;
    let items = vec![LogIsolationExecutionItem {
        equation: equation.clone(),
        description: step.description,
    }];
    Some(LogIsolationRewritePlan { equation, items })
}

/// Build the transformed equation target for `log(base, arg) = rhs`.
///
/// Returns `None` when:
/// - both `base` and `arg` contain `var`, or
/// - neither contains `var`.
pub fn plan_log_isolation(
    ctx: &mut Context,
    base: ExprId,
    arg: ExprId,
    rhs: ExprId,
    var: &str,
) -> Option<LogIsolationPlan> {
    let arg_has_var = contains_var(ctx, arg, var);
    let base_has_var = contains_var(ctx, base, var);

    match (arg_has_var, base_has_var) {
        (true, false) => {
            let new_rhs = ctx.add(Expr::Pow(base, rhs));
            Some(LogIsolationPlan::SolveArgument {
                lhs: arg,
                rhs: new_rhs,
            })
        }
        (false, true) => {
            let one = ctx.num(1);
            let inv_rhs = ctx.add(Expr::Div(one, rhs));
            let new_rhs = ctx.add(Expr::Pow(arg, inv_rhs));
            Some(LogIsolationPlan::SolveBase {
                lhs: base,
                rhs: new_rhs,
            })
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::SolutionSet;

    #[test]
    fn plan_log_isolation_solves_argument_when_var_in_arg_only() {
        let mut ctx = Context::new();
        let b = ctx.num(2);
        let x = ctx.var("x");
        let rhs = ctx.num(3);

        let plan = plan_log_isolation(&mut ctx, b, x, rhs, "x")
            .expect("should isolate logarithm argument");
        match plan {
            LogIsolationPlan::SolveArgument {
                lhs,
                rhs: planned_rhs,
            } => {
                assert_eq!(lhs, x);
                assert!(
                    matches!(ctx.get(planned_rhs), Expr::Pow(base, exp) if *base == b && *exp == rhs)
                );
            }
            LogIsolationPlan::SolveBase { .. } => panic!("expected SolveArgument plan"),
        }
    }

    #[test]
    fn plan_log_isolation_solves_base_when_var_in_base_only() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let arg = ctx.num(5);
        let rhs = ctx.num(2);

        let plan =
            plan_log_isolation(&mut ctx, x, arg, rhs, "x").expect("should isolate logarithm base");
        match plan {
            LogIsolationPlan::SolveBase {
                lhs,
                rhs: planned_rhs,
            } => {
                assert_eq!(lhs, x);
                assert!(matches!(ctx.get(planned_rhs), Expr::Pow(base, _) if *base == arg));
            }
            LogIsolationPlan::SolveArgument { .. } => panic!("expected SolveBase plan"),
        }
    }

    #[test]
    fn plan_log_isolation_rejects_ambiguous_or_var_free_cases() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let three = ctx.num(3);

        assert!(plan_log_isolation(&mut ctx, x, x, three, "x").is_none());
        assert!(plan_log_isolation(&mut ctx, y, three, three, "x").is_none());
    }

    #[test]
    fn into_equation_preserves_relation_and_sides() {
        let mut ctx = Context::new();
        let lhs = ctx.var("x");
        let rhs = ctx.var("y");
        let eq = LogIsolationPlan::SolveArgument { lhs, rhs }.into_equation(RelOp::Geq);
        assert_eq!(eq.lhs, lhs);
        assert_eq!(eq.rhs, rhs);
        assert_eq!(eq.op, RelOp::Geq);
    }

    #[test]
    fn step_description_for_solve_argument_uses_base_display() {
        let mut ctx = Context::new();
        let lhs = ctx.var("x");
        let rhs = ctx.var("y");
        let msg = LogIsolationPlan::SolveArgument { lhs, rhs }.step_description("2");
        assert_eq!(msg, "Exponentiate both sides with base 2");
    }

    #[test]
    fn step_description_for_solve_base_is_constant_text() {
        let mut ctx = Context::new();
        let lhs = ctx.var("x");
        let rhs = ctx.var("y");
        let msg = LogIsolationPlan::SolveBase { lhs, rhs }.step_description("ignored");
        assert_eq!(msg, "Isolate base of logarithm");
    }

    #[test]
    fn build_log_isolation_step_with_uses_rendered_base_and_equation() {
        let mut ctx = Context::new();
        let lhs = ctx.var("x");
        let rhs = ctx.var("y");
        let base = ctx.var("b");
        let plan = LogIsolationPlan::SolveArgument { lhs, rhs };
        let step =
            build_log_isolation_step_with(plan, &ctx, base, RelOp::Eq, |_, _| "b".to_string());
        assert_eq!(step.description, "Exponentiate both sides with base b");
        assert_eq!(step.equation_after.lhs, lhs);
        assert_eq!(step.equation_after.rhs, rhs);
        assert_eq!(step.equation_after.op, RelOp::Eq);
    }

    #[test]
    fn plan_log_isolation_step_builds_rewrite_and_step() {
        let mut ctx = Context::new();
        let base = ctx.num(2);
        let arg = ctx.var("x");
        let rhs = ctx.num(3);

        let out = plan_log_isolation_step(&mut ctx, base, arg, rhs, "x", RelOp::Eq, "2")
            .expect("log isolation should apply");
        assert_eq!(out.equation.lhs, arg);
        assert_eq!(out.items.len(), 1);
        assert_eq!(
            out.items[0].description,
            "Exponentiate both sides with base 2"
        );
        assert_eq!(out.items[0].equation, out.equation);
    }

    #[test]
    fn plan_log_isolation_step_with_uses_renderer() {
        let mut ctx = Context::new();
        let base = ctx.var("b");
        let arg = ctx.var("x");
        let rhs = ctx.num(3);

        let out = plan_log_isolation_step_with(&mut ctx, base, arg, rhs, "x", RelOp::Eq, |_, _| {
            "rendered(b)".to_string()
        })
        .expect("log isolation should apply");

        assert_eq!(
            out.items[0].description,
            "Exponentiate both sides with base rendered(b)"
        );
        assert_eq!(out.items[0].equation, out.equation);
    }

    #[test]
    fn collect_log_isolation_didactic_steps_returns_single_step() {
        let mut ctx = Context::new();
        let base = ctx.var("b");
        let arg = ctx.var("x");
        let rhs = ctx.num(3);
        let out = plan_log_isolation_step_with(&mut ctx, base, arg, rhs, "x", RelOp::Eq, |_, _| {
            "rendered(b)".to_string()
        })
        .expect("log isolation should apply");

        let didactic = collect_log_isolation_didactic_steps(&out);
        assert_eq!(didactic.len(), 1);
        assert_eq!(didactic[0].description, out.items[0].description);
        assert_eq!(didactic[0].equation_after, out.items[0].equation);
    }

    #[test]
    fn collect_log_isolation_execution_items_returns_single_item() {
        let mut ctx = Context::new();
        let base = ctx.var("b");
        let arg = ctx.var("x");
        let rhs = ctx.num(3);
        let out = plan_log_isolation_step_with(&mut ctx, base, arg, rhs, "x", RelOp::Eq, |_, _| {
            "rendered(b)".to_string()
        })
        .expect("log isolation should apply");

        let items = collect_log_isolation_execution_items(&out);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].equation, out.equation);
        assert_eq!(items[0].description, out.items[0].description);
    }

    #[test]
    fn first_log_isolation_execution_item_returns_single_item() {
        let mut ctx = Context::new();
        let base = ctx.var("b");
        let arg = ctx.var("x");
        let rhs = ctx.num(3);
        let out = plan_log_isolation_step_with(&mut ctx, base, arg, rhs, "x", RelOp::Eq, |_, _| {
            "rendered(b)".to_string()
        })
        .expect("log isolation should apply");

        let item = first_log_isolation_execution_item(&out).expect("expected one execution item");
        assert_eq!(item.equation, out.equation);
        assert_eq!(item.description, out.items[0].description);
    }

    #[test]
    fn solve_log_isolation_rewrite_with_runs_solver_once_and_preserves_rewrite() {
        let mut ctx = Context::new();
        let base = ctx.var("b");
        let arg = ctx.var("x");
        let rhs = ctx.num(3);
        let rewrite =
            plan_log_isolation_step_with(&mut ctx, base, arg, rhs, "x", RelOp::Eq, |_, _| {
                "rendered(b)".to_string()
            })
            .expect("log isolation should apply");
        let expected_equation = rewrite.equation.clone();
        let mut calls = 0usize;
        let solved = solve_log_isolation_rewrite_with(rewrite, |_eq| {
            calls += 1;
            Ok::<_, ()>(Equation {
                lhs: arg,
                rhs,
                op: RelOp::Eq,
            })
        })
        .expect("solver callback should succeed");

        assert_eq!(calls, 1);
        assert_eq!(solved.rewrite.equation, expected_equation);
        assert_eq!(
            solved.solved,
            Equation {
                lhs: arg,
                rhs,
                op: RelOp::Eq
            }
        );
    }

    #[test]
    fn solve_log_isolation_rewrite_with_item_passes_item_to_solver() {
        let mut ctx = Context::new();
        let base = ctx.var("b");
        let arg = ctx.var("x");
        let rhs = ctx.num(3);
        let rewrite =
            plan_log_isolation_step_with(&mut ctx, base, arg, rhs, "x", RelOp::Eq, |_, _| {
                "rendered(b)".to_string()
            })
            .expect("log isolation should apply");
        let expected_equation = rewrite.equation.clone();

        let mut seen_item_desc = None;
        let solved = solve_log_isolation_rewrite_with_item(rewrite, |item, equation| {
            seen_item_desc = item.map(|entry| entry.description);
            Ok::<_, ()>(equation.clone())
        })
        .expect("solver callback should succeed");

        assert_eq!(
            seen_item_desc,
            Some("Exponentiate both sides with base rendered(b)".to_string())
        );
        assert_eq!(solved.solved, expected_equation);
    }

    #[test]
    fn solve_log_isolation_rewrite_pipeline_with_item_includes_item_then_substeps() {
        let mut ctx = Context::new();
        let base = ctx.var("b");
        let arg = ctx.var("x");
        let rhs = ctx.num(3);
        let rewrite =
            plan_log_isolation_step_with(&mut ctx, base, arg, rhs, "x", RelOp::Eq, |_, _| {
                "rendered(b)".to_string()
            })
            .expect("log isolation should apply");

        let expected_equation = rewrite.equation.clone();
        let mut solve_calls = 0usize;
        let solved = solve_log_isolation_rewrite_pipeline_with_item(
            rewrite,
            true,
            |equation| {
                solve_calls += 1;
                assert_eq!(equation, &expected_equation);
                Ok::<_, ()>((
                    SolutionSet::AllReals,
                    vec!["sub-1".to_string(), "sub-2".to_string()],
                ))
            },
            |item| format!("ITEM: {}", item.description),
        )
        .expect("pipeline solve should succeed");

        assert_eq!(solve_calls, 1);
        assert_eq!(solved.solved.0, SolutionSet::AllReals);
        assert_eq!(solved.solved.1.len(), 3);
        assert!(solved.solved.1[0].starts_with("ITEM: "));
        assert_eq!(solved.solved.1[1], "sub-1");
        assert_eq!(solved.solved.1[2], "sub-2");
    }

    #[test]
    fn solve_log_isolation_rewrite_pipeline_with_item_omits_item_when_disabled() {
        let mut ctx = Context::new();
        let base = ctx.var("b");
        let arg = ctx.var("x");
        let rhs = ctx.num(3);
        let rewrite =
            plan_log_isolation_step_with(&mut ctx, base, arg, rhs, "x", RelOp::Eq, |_, _| {
                "rendered(b)".to_string()
            })
            .expect("log isolation should apply");

        let expected_equation = rewrite.equation.clone();
        let mut solve_calls = 0usize;
        let solved = solve_log_isolation_rewrite_pipeline_with_item(
            rewrite,
            false,
            |equation| {
                solve_calls += 1;
                assert_eq!(equation, &expected_equation);
                Ok::<_, ()>((
                    SolutionSet::Discrete(vec![rhs]),
                    vec!["sub-only".to_string()],
                ))
            },
            |item| format!("ITEM: {}", item.description),
        )
        .expect("pipeline solve should succeed");

        assert_eq!(solve_calls, 1);
        assert_eq!(solved.solved.0, SolutionSet::Discrete(vec![rhs]));
        assert_eq!(solved.solved.1, vec!["sub-only".to_string()]);
    }

    #[test]
    fn execute_log_isolation_pipeline_with_item_with_runs_supported_pipeline() {
        let mut ctx = Context::new();
        let base = ctx.var("b");
        let arg = ctx.var("x");
        let rhs = ctx.num(3);

        let mut solve_calls = 0usize;
        let solved = execute_log_isolation_pipeline_with_item_with(
            true,
            || {
                plan_log_isolation_step_with(&mut ctx, base, arg, rhs, "x", RelOp::Eq, |_, _| {
                    "rendered(b)".to_string()
                })
            },
            |_equation| {
                solve_calls += 1;
                Ok::<_, ()>((SolutionSet::AllReals, vec!["sub".to_string()]))
            },
            |item| item.description,
        )
        .expect("log isolation should be supported")
        .expect("pipeline solve should succeed");

        assert_eq!(solve_calls, 1);
        assert_eq!(solved.solved.0, SolutionSet::AllReals);
        assert_eq!(solved.solved.1.len(), 2);
        assert_eq!(solved.solved.1[1], "sub");
    }

    #[test]
    fn execute_log_isolation_result_pipeline_with_item_with_returns_plain_tuple() {
        let mut ctx = Context::new();
        let base = ctx.var("b");
        let arg = ctx.var("x");
        let rhs = ctx.num(3);

        let solved = execute_log_isolation_result_pipeline_with_item_with(
            true,
            || {
                plan_log_isolation_step_with(&mut ctx, base, arg, rhs, "x", RelOp::Eq, |_, _| {
                    "rendered(b)".to_string()
                })
            },
            |_equation| Ok::<_, ()>((SolutionSet::AllReals, vec!["sub".to_string()])),
            |item| item.description,
        )
        .expect("log isolation should be supported")
        .expect("pipeline solve should succeed");

        assert_eq!(solved.0, SolutionSet::AllReals);
        assert_eq!(solved.1.len(), 2);
        assert_eq!(solved.1[1], "sub");
    }

    #[test]
    fn execute_log_isolation_result_pipeline_or_else_with_returns_result_for_supported() {
        let mut ctx = Context::new();
        let base = ctx.var("b");
        let arg = ctx.var("x");
        let rhs = ctx.num(3);

        let solved = execute_log_isolation_result_pipeline_or_else_with(
            true,
            || {
                plan_log_isolation_step_with(&mut ctx, base, arg, rhs, "x", RelOp::Eq, |_, _| {
                    "rendered(b)".to_string()
                })
            },
            |_equation| Ok::<_, &'static str>((SolutionSet::AllReals, vec!["sub".to_string()])),
            |item| item.description,
            || "not-plannable",
        )
        .expect("supported log isolation should produce result");

        assert_eq!(solved.0, SolutionSet::AllReals);
        assert_eq!(solved.1.len(), 2);
    }

    #[test]
    fn execute_log_isolation_result_pipeline_or_else_with_maps_not_plannable_to_error() {
        let mut ctx = Context::new();
        let base = ctx.var("x");
        let arg = ctx.var("x");
        let rhs = ctx.num(3);

        let err = execute_log_isolation_result_pipeline_or_else_with(
            true,
            || {
                plan_log_isolation_step_with(&mut ctx, base, arg, rhs, "x", RelOp::Eq, |_, _| {
                    "rendered".to_string()
                })
            },
            |_equation| Ok::<_, &'static str>((SolutionSet::Empty, vec!["unexpected".to_string()])),
            |item| item.description,
            || "not-plannable",
        )
        .expect_err("not-plannable route should map to error");

        assert_eq!(err, "not-plannable");
    }

    #[test]
    fn execute_log_isolation_pipeline_with_item_with_returns_none_when_not_plannable() {
        let mut ctx = Context::new();
        let base = ctx.var("x");
        let arg = ctx.var("x");
        let rhs = ctx.num(3);

        let mut solve_calls = 0usize;
        let out = execute_log_isolation_pipeline_with_item_with(
            true,
            || {
                plan_log_isolation_step_with(&mut ctx, base, arg, rhs, "x", RelOp::Eq, |_, _| {
                    "rendered".to_string()
                })
            },
            |_equation| {
                solve_calls += 1;
                Ok::<_, ()>((SolutionSet::Empty, vec!["unexpected".to_string()]))
            },
            |item| item.description,
        );

        assert!(out.is_none());
        assert_eq!(solve_calls, 0);
    }
}
