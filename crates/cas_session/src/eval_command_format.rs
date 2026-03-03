use cas_ast::{Expr, ExprId};
use cas_formatter::root_style::ParseStyleSignals;

use crate::eval_command_types::{EvalCommandEvalView, EvalMetadataLines, EvalResultLine};

pub(crate) fn format_eval_result_line(
    context: &cas_ast::Context,
    parsed_expr: ExprId,
    result: &cas_solver::EvalResult,
    style_signals: &ParseStyleSignals,
) -> Option<EvalResultLine> {
    let style_prefs = cas_formatter::StylePreferences::from_expression_with_signals(
        context,
        parsed_expr,
        Some(style_signals),
    );

    match result {
        cas_solver::EvalResult::Expr(res) => {
            if let Expr::Function(name, args) = context.get(*res) {
                if context.is_builtin(*name, cas_ast::BuiltinFn::Equal) && args.len() == 2 {
                    return Some(EvalResultLine {
                        line: format!(
                            "Result: {} = {}",
                            cas_formatter::clean_display_string(&format!(
                                "{}",
                                cas_formatter::DisplayExprStyled::new(
                                    context,
                                    args[0],
                                    &style_prefs
                                )
                            )),
                            cas_formatter::clean_display_string(&format!(
                                "{}",
                                cas_formatter::DisplayExprStyled::new(
                                    context,
                                    args[1],
                                    &style_prefs
                                )
                            )),
                        ),
                        terminal: true,
                    });
                }
            }

            Some(EvalResultLine {
                line: format!(
                    "Result: {}",
                    cas_solver::display_expr_or_poly(context, *res)
                ),
                terminal: false,
            })
        }
        cas_solver::EvalResult::SolutionSet(solution_set) => Some(EvalResultLine {
            line: format!(
                "Result: {}",
                cas_solver::display_solution_set(context, solution_set)
            ),
            terminal: false,
        }),
        cas_solver::EvalResult::Set(_sols) => Some(EvalResultLine {
            line: "Result: Set(...)".to_string(),
            terminal: false,
        }),
        cas_solver::EvalResult::Bool(value) => Some(EvalResultLine {
            line: format!("Result: {}", value),
            terminal: false,
        }),
        cas_solver::EvalResult::None => None,
    }
}

pub(crate) fn format_eval_stored_entry_line(
    context: &cas_ast::Context,
    output: &EvalCommandEvalView,
) -> Option<String> {
    output.stored_id.map(|id| {
        format!(
            "#{id}: {}",
            cas_formatter::DisplayExpr {
                context,
                id: output.parsed
            }
        )
    })
}

pub(crate) fn format_eval_metadata_lines(
    context: &mut cas_ast::Context,
    output: &EvalCommandEvalView,
    requires_display: cas_solver::RequiresDisplayLevel,
    debug_mode: bool,
    hints_enabled: bool,
    domain_mode: cas_solver::DomainMode,
    assumption_reporting: cas_solver::AssumptionReporting,
) -> EvalMetadataLines {
    let warning_lines = crate::format_domain_warning_lines(&output.domain_warnings, true, "⚠ ");

    let result_expr = match &output.result {
        cas_solver::EvalResult::Expr(expr_id) => Some(*expr_id),
        _ => None,
    };
    let mut requires_lines = Vec::new();
    if !output.diagnostics.requires.is_empty() {
        let rendered = crate::format_diagnostics_requires_lines(
            context,
            &output.diagnostics,
            result_expr,
            requires_display,
            debug_mode,
        );
        if !rendered.is_empty() {
            requires_lines.push("ℹ️ Requires:".to_string());
            requires_lines.extend(rendered);
        }
    }

    let hint_lines = if hints_enabled {
        let hints =
            crate::filter_blocked_hints_for_eval(context, output.resolved, &output.blocked_hints);
        if hints.is_empty() {
            Vec::new()
        } else {
            crate::format_eval_blocked_hints_lines(context, &hints, domain_mode)
        }
    } else {
        Vec::new()
    };

    let assumption_lines = if assumption_reporting != cas_solver::AssumptionReporting::Off {
        let assumed_conditions = crate::collect_assumed_conditions_from_steps(&output.steps);
        if assumed_conditions.is_empty() {
            Vec::new()
        } else {
            crate::format_assumed_conditions_report_lines(&assumed_conditions)
        }
    } else {
        Vec::new()
    };

    EvalMetadataLines {
        warning_lines,
        requires_lines,
        hint_lines,
        assumption_lines,
    }
}

pub(crate) fn format_eval_result_text(
    ctx: &cas_ast::Context,
    result: &cas_solver::EvalResult,
) -> String {
    match result {
        cas_solver::EvalResult::Expr(expr) => {
            if let Some(poly_str) = cas_solver::try_render_poly_result(ctx, *expr) {
                poly_str
            } else {
                format!(
                    "{}",
                    cas_formatter::DisplayExpr {
                        context: ctx,
                        id: *expr
                    }
                )
            }
        }
        cas_solver::EvalResult::Set(values) if !values.is_empty() => format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: ctx,
                id: values[0]
            }
        ),
        cas_solver::EvalResult::Bool(value) => value.to_string(),
        _ => "(no result)".to_string(),
    }
}
