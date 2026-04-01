use cas_api_models::{parse_eval_special_command, EvalLimitApproach, EvalSpecialCommand};
use cas_ast::{Context, ExprId};
use cas_formatter::LaTeXExpr;

pub(crate) fn format_output_input_latex(ctx: &Context, raw_input: &str, parsed: ExprId) -> String {
    if let Some(EvalSpecialCommand::Limit { var, approach, .. }) =
        parse_eval_special_command(raw_input)
    {
        let expr_latex = LaTeXExpr {
            context: ctx,
            id: parsed,
        }
        .to_latex();
        let approach_latex = match approach {
            EvalLimitApproach::PosInfinity => "\\infty",
            EvalLimitApproach::NegInfinity => "-\\infty",
        };
        return format!("\\lim_{{{var} \\to {approach_latex}}} {expr_latex}");
    }

    if let Some((lhs, rhs)) = cas_ast::eq::unwrap_eq(ctx, parsed) {
        let lhs_latex = LaTeXExpr {
            context: ctx,
            id: lhs,
        }
        .to_latex();
        let rhs_latex = LaTeXExpr {
            context: ctx,
            id: rhs,
        }
        .to_latex();
        format!("{lhs_latex} = {rhs_latex}")
    } else {
        LaTeXExpr {
            context: ctx,
            id: parsed,
        }
        .to_latex()
    }
}
