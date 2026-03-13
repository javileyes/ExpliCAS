use crate::rationalize_command::RationalizeCommandOutcome;

pub(crate) fn format_rationalize_eval_lines(
    context: &cas_ast::Context,
    normalized_expr: cas_ast::ExprId,
    outcome: RationalizeCommandOutcome,
) -> Vec<String> {
    let user_style = cas_formatter::root_style::detect_root_style(context, normalized_expr);
    let parsed = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context,
            id: normalized_expr
        }
    );

    let line = match outcome {
        RationalizeCommandOutcome::Success(simplified_expr) => {
            let style = cas_formatter::root_style::StylePreferences::with_root_style(user_style);
            let rendered = cas_formatter::DisplayExprStyled::new(context, simplified_expr, &style);
            format!("Parsed: {}\nRationalized: {}", parsed, rendered)
        }
        RationalizeCommandOutcome::NotApplicable => format!(
            "Parsed: {}\n\
             Cannot rationalize: denominator is not a sum of surds\n\
             (Supported: 1/(a + b√n + c√m) where a,b,c are rational and n,m are positive integers)",
            parsed
        ),
        RationalizeCommandOutcome::BudgetExceeded => format!(
            "Parsed: {}\n\
             Rationalization aborted: expression became too complex",
            parsed
        ),
    };

    vec![line]
}
