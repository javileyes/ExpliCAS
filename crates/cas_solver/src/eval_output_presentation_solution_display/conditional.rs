use cas_ast::Context;

pub(super) fn format_conditional_output_solution_set(
    ctx: &Context,
    cases: &[cas_ast::Case],
) -> String {
    let mut parts = Vec::new();
    for case in cases {
        if crate::is_pure_residual_otherwise(case) {
            continue;
        }
        let cond_str = cas_formatter::condition_set_to_display(&case.when, ctx);
        let inner_str = super::render::format_output_solution_set(ctx, &case.then.solutions);
        if case.when.is_empty() {
            parts.push(format!("{inner_str} otherwise"));
        } else {
            parts.push(format!("{inner_str} if {cond_str}"));
        }
    }

    if parts.is_empty() && !cases.is_empty() {
        for case in cases {
            if !case.when.is_empty() {
                let inner_str =
                    super::render::format_output_solution_set(ctx, &case.then.solutions);
                let cond_str = cas_formatter::condition_set_to_display(&case.when, ctx);
                return format!("{inner_str} if {cond_str}");
            }
        }
    }

    parts.join("; ")
}
