use cas_ast::{Case, Context};

pub(super) fn render_conditional_solution_set(ctx: &Context, cases: &[Case]) -> String {
    let mut latex_parts = Vec::new();
    for case in cases {
        let cond_latex = cas_formatter::condition_set_to_latex(&case.when, ctx);
        let inner_latex = super::solution_set_to_latex_eval_json(ctx, &case.then.solutions);
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
