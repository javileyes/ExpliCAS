use cas_ast::{BoundType, Context, Interval, SolutionSet};
use cas_formatter::LaTeXExpr;

pub(super) fn render_solution_set_to_latex(
    context: &Context,
    solution_set: &SolutionSet,
) -> String {
    match solution_set {
        SolutionSet::Empty => r"\emptyset".to_string(),
        SolutionSet::AllReals => r"\mathbb{R}".to_string(),
        SolutionSet::Discrete(exprs) => {
            let elements: Vec<String> = exprs
                .iter()
                .map(|e| LaTeXExpr { context, id: *e }.to_latex())
                .collect();
            format!(r"\left\{{ {} \right\}}", elements.join(", "))
        }
        SolutionSet::Continuous(interval) => render_interval_to_latex(context, interval),
        SolutionSet::Union(intervals) => {
            let parts: Vec<String> = intervals
                .iter()
                .map(|i| render_interval_to_latex(context, i))
                .collect();
            parts.join(r" \cup ")
        }
        SolutionSet::Residual(expr) => LaTeXExpr { context, id: *expr }.to_latex(),
        SolutionSet::Conditional(cases) => {
            let case_strs: Vec<String> = cases
                .iter()
                .filter_map(|case| {
                    if case.when.is_otherwise()
                        && matches!(&case.then.solutions, SolutionSet::Residual(_))
                    {
                        return None;
                    }
                    let sol_latex =
                        render_inner_solution_set_to_latex(context, &case.then.solutions);
                    if case.when.is_otherwise() {
                        Some(format!("{} & \\text{{otherwise}}", sol_latex))
                    } else {
                        let cond_latex = cas_formatter::condition_set_to_latex(&case.when, context);
                        Some(format!("{} & \\text{{if }} {}", sol_latex, cond_latex))
                    }
                })
                .collect();
            if case_strs.len() == 1 {
                let single = &case_strs[0];
                if let Some(idx) = single.find(r" & \text{if}") {
                    return single[..idx].to_string();
                }
            }
            format!(r"\begin{{cases}} {} \end{{cases}}", case_strs.join(r" \\ "))
        }
    }
}

fn render_inner_solution_set_to_latex(context: &Context, solution_set: &SolutionSet) -> String {
    match solution_set {
        SolutionSet::Empty => r"\emptyset".to_string(),
        SolutionSet::AllReals => r"\mathbb{R}".to_string(),
        SolutionSet::Discrete(exprs) => {
            let elements: Vec<String> = exprs
                .iter()
                .map(|e| LaTeXExpr { context, id: *e }.to_latex())
                .collect();
            format!(r"\left\{{ {} \right\}}", elements.join(", "))
        }
        SolutionSet::Continuous(interval) => render_interval_to_latex(context, interval),
        SolutionSet::Union(intervals) => {
            let parts: Vec<String> = intervals
                .iter()
                .map(|i| render_interval_to_latex(context, i))
                .collect();
            parts.join(r" \cup ")
        }
        SolutionSet::Residual(expr) => LaTeXExpr { context, id: *expr }.to_latex(),
        SolutionSet::Conditional(_) => r"\text{(nested conditional)}".to_string(),
    }
}

fn render_interval_to_latex(context: &Context, interval: &Interval) -> String {
    let left = match interval.min_type {
        BoundType::Open => "(",
        BoundType::Closed => "[",
    };
    let right = match interval.max_type {
        BoundType::Open => ")",
        BoundType::Closed => "]",
    };
    let min_latex = LaTeXExpr {
        context,
        id: interval.min,
    }
    .to_latex();
    let max_latex = LaTeXExpr {
        context,
        id: interval.max,
    }
    .to_latex();
    format!(r"{}{}, {}{}", left, min_latex, max_latex, right)
}
