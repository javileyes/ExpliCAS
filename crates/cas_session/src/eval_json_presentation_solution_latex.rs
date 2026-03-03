use cas_ast::{Context, SolutionSet};
use cas_formatter::LaTeXExpr;

pub(crate) fn solution_set_to_latex_eval_json(ctx: &Context, solution_set: &SolutionSet) -> String {
    match solution_set {
        SolutionSet::Empty => r"\emptyset".to_string(),
        SolutionSet::AllReals => r"\mathbb{R}".to_string(),
        SolutionSet::Discrete(exprs) => {
            if exprs.is_empty() {
                r"\emptyset".to_string()
            } else {
                let solutions: Vec<String> = exprs
                    .iter()
                    .map(|e| {
                        LaTeXExpr {
                            context: ctx,
                            id: *e,
                        }
                        .to_latex()
                    })
                    .collect();
                format!(r"\left\{{ {} \right\}}", solutions.join(", "))
            }
        }
        SolutionSet::Conditional(cases) => {
            let mut latex_parts = Vec::new();
            for case in cases {
                let cond_latex = cas_formatter::condition_set_to_latex(&case.when, ctx);
                let inner_latex = solution_set_to_latex_eval_json(ctx, &case.then.solutions);
                if case.when.is_empty() {
                    latex_parts.push(format!(r"{} & \text{{otherwise}}", inner_latex));
                } else {
                    latex_parts.push(format!(r"{} & \text{{if }} {}", inner_latex, cond_latex));
                }
            }
            if latex_parts.len() == 1 {
                let single = &latex_parts[0];
                if let Some(idx) = single.find(r" & \text{if}") {
                    return single[..idx].to_string();
                }
            }
            format!(
                r"\begin{{cases}} {} \end{{cases}}",
                latex_parts.join(r" \\ ")
            )
        }
        SolutionSet::Continuous(interval) => {
            let min_latex = LaTeXExpr {
                context: ctx,
                id: interval.min,
            }
            .to_latex();
            let max_latex = LaTeXExpr {
                context: ctx,
                id: interval.max,
            }
            .to_latex();
            format!(r"\left[{}, {}\right]", min_latex, max_latex)
        }
        SolutionSet::Union(intervals) => {
            let parts: Vec<String> = intervals
                .iter()
                .map(|int| {
                    let min = LaTeXExpr {
                        context: ctx,
                        id: int.min,
                    }
                    .to_latex();
                    let max = LaTeXExpr {
                        context: ctx,
                        id: int.max,
                    }
                    .to_latex();
                    format!(r"\left[{}, {}\right]", min, max)
                })
                .collect();
            parts.join(r" \cup ")
        }
        SolutionSet::Residual(expr) => {
            let expr_latex = LaTeXExpr {
                context: ctx,
                id: *expr,
            }
            .to_latex();
            format!(r"\text{{Solve: }} {} = 0", expr_latex)
        }
    }
}
