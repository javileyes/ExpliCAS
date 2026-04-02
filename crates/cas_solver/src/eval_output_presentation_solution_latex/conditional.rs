use cas_ast::{Case, Context};

pub(super) fn render_conditional_solution_set(ctx: &Context, cases: &[Case]) -> String {
    let filtered_cases: Vec<_> = cases
        .iter()
        .filter(|case| !crate::is_pure_residual_otherwise(case))
        .collect();

    if filtered_cases.is_empty() {
        return cases
            .first()
            .map(|case| super::solution_set_to_output_latex(ctx, &case.then.solutions))
            .unwrap_or_else(|| r"\emptyset".to_string());
    }

    let mut latex_parts = Vec::new();
    for case in &filtered_cases {
        let inner_latex = super::solution_set_to_output_latex(ctx, &case.then.solutions);
        if case.when.is_empty() {
            latex_parts.push(format!(r"{} & \text{{otherwise}}", inner_latex));
        } else {
            let cond_latex = cas_formatter::condition_set_to_latex(&case.when, ctx);
            latex_parts.push(format!(r"{} & \text{{if }} {}", inner_latex, cond_latex));
        }
    }

    if filtered_cases.len() == 1 {
        let case = filtered_cases[0];
        if case.when.is_empty() {
            return super::solution_set_to_output_latex(ctx, &case.then.solutions);
        }
    }

    format!(
        r"\begin{{cases}} {} \end{{cases}}",
        latex_parts.join(r" \\ ")
    )
}

#[cfg(test)]
mod tests {
    use super::render_conditional_solution_set;
    use cas_ast::{Case, ConditionPredicate, ConditionSet, Context, SolutionSet, SolveResult};

    #[test]
    fn conditional_latex_skips_pure_residual_otherwise_but_keeps_guard() {
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

        let rendered = render_conditional_solution_set(&ctx, &[guarded, otherwise_residual]);

        assert!(rendered.contains(r"\text{if }"), "rendered: {rendered}");
        assert!(
            !rendered.contains(r"\text{otherwise}"),
            "rendered: {rendered}"
        );
        assert!(
            !rendered.contains(r"\text{Solve: }"),
            "rendered: {rendered}"
        );
    }
}
