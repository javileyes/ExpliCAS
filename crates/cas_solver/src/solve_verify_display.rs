use cas_ast::Context;

/// Format verification summary lines after solve result checking.
pub fn format_verify_summary_lines(
    ctx: &Context,
    var: &str,
    verify_result: &crate::VerifyResult,
    detail_prefix: &str,
) -> Vec<String> {
    format_verify_summary_lines_with_hints(ctx, var, verify_result, detail_prefix, true)
}

/// Format verification summary lines with explicit hint visibility control.
///
/// This keeps session/UI clients aligned with a runtime `hints_enabled` toggle
/// without reimplementing verification rendering policy.
pub fn format_verify_summary_lines_with_hints(
    ctx: &Context,
    var: &str,
    verify_result: &crate::VerifyResult,
    detail_prefix: &str,
    hints_enabled: bool,
) -> Vec<String> {
    let mut lines = Vec::new();

    match verify_result.summary {
        crate::VerifySummary::AllVerified => {
            lines.push("✓ All solutions verified".to_string());
        }
        crate::VerifySummary::PartiallyVerified => {
            lines.push("⚠ Some solutions verified".to_string());
            push_solution_status_lines(
                &mut lines,
                ctx,
                var,
                &verify_result.solutions,
                detail_prefix,
                hints_enabled,
            );
            if let Some(desc) = &verify_result.guard_description {
                lines.push(format!("{detail_prefix}ℹ {desc}"));
            }
        }
        crate::VerifySummary::NoneVerified => {
            lines.push("⚠ No solutions could be verified".to_string());
            push_solution_status_lines(
                &mut lines,
                ctx,
                var,
                &verify_result.solutions,
                detail_prefix,
                hints_enabled,
            );
        }
        crate::VerifySummary::VerifiedUnderGuard => {
            if let Some(desc) = &verify_result.guard_description {
                lines.push(format!("✓ {desc}"));
            } else {
                lines.push("✓ Verified symbolically under guard".to_string());
            }
        }
        crate::VerifySummary::NeedsSampling => {
            if let Some(desc) = &verify_result.guard_description {
                lines.push(format!("ℹ {desc}"));
            } else {
                lines.push("ℹ Verification requires numeric sampling".to_string());
            }
        }
        crate::VerifySummary::NotCheckable => {
            if let Some(desc) = &verify_result.guard_description {
                lines.push(format!("ℹ {desc}"));
            } else {
                lines.push("ℹ Solution type not checkable".to_string());
            }
        }
        crate::VerifySummary::Empty => {}
    }

    lines
}

fn push_solution_status_lines(
    lines: &mut Vec<String>,
    ctx: &Context,
    var: &str,
    solutions: &[(cas_ast::ExprId, crate::VerifyStatus)],
    detail_prefix: &str,
    hints_enabled: bool,
) {
    for (sol_id, status) in solutions {
        let sol_str = cas_formatter::render_expr(ctx, *sol_id);
        match status {
            crate::VerifyStatus::Verified => {
                lines.push(format!("{detail_prefix}✓ {var} = {sol_str} verified"));
            }
            crate::VerifyStatus::Unverifiable {
                reason,
                counterexample_hint,
                ..
            } => {
                lines.push(format!("{detail_prefix}⚠ {var} = {sol_str}: {reason}"));
                if hints_enabled {
                    if let Some(hint) = counterexample_hint {
                        lines.push(format!("{detail_prefix}↳ {hint}"));
                    }
                }
            }
            crate::VerifyStatus::NotCheckable { reason } => {
                lines.push(format!("{detail_prefix}ℹ {var} = {sol_str}: {reason}"));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::Context;

    #[test]
    fn renders_needs_sampling_description_when_present() {
        let ctx = Context::new();
        let verify_result = crate::VerifyResult {
            solutions: vec![],
            summary: crate::VerifySummary::NeedsSampling,
            guard_description: Some(
                "verification requires numeric sampling (continuous interval)".to_string(),
            ),
        };

        let lines = format_verify_summary_lines(&ctx, "x", &verify_result, "  ");
        assert_eq!(
            lines,
            vec!["ℹ verification requires numeric sampling (continuous interval)".to_string()]
        );
    }

    #[test]
    fn renders_generic_needs_sampling_message_without_description() {
        let ctx = Context::new();
        let verify_result = crate::VerifyResult {
            solutions: vec![],
            summary: crate::VerifySummary::NeedsSampling,
            guard_description: None,
        };

        let lines = format_verify_summary_lines(&ctx, "x", &verify_result, "  ");
        assert_eq!(
            lines,
            vec!["ℹ Verification requires numeric sampling".to_string()]
        );
    }

    #[test]
    fn renders_verified_under_guard_description_when_present() {
        let ctx = Context::new();
        let verify_result = crate::VerifyResult {
            solutions: vec![],
            summary: crate::VerifySummary::VerifiedUnderGuard,
            guard_description: Some(
                "verified symbolically under guard (1 guarded non-discrete branch)".to_string(),
            ),
        };

        let lines = format_verify_summary_lines(&ctx, "x", &verify_result, "  ");
        assert_eq!(
            lines,
            vec!["✓ verified symbolically under guard (1 guarded non-discrete branch)".to_string()]
        );
    }

    #[test]
    fn renders_none_verified_details_with_counterexample_hint() {
        let mut ctx = Context::new();
        let one = ctx.num(1);
        let verify_result = crate::VerifyResult {
            solutions: vec![(
                one,
                crate::VerifyStatus::Unverifiable {
                    residual: one,
                    reason: "residual: a - 1".to_string(),
                    counterexample_hint: Some(
                        "counterexample hint: a=0 gives residual -1".to_string(),
                    ),
                },
            )],
            summary: crate::VerifySummary::NoneVerified,
            guard_description: None,
        };

        let lines = format_verify_summary_lines(&ctx, "x", &verify_result, "  ");
        assert_eq!(
            lines,
            vec![
                "⚠ No solutions could be verified".to_string(),
                "  ⚠ x = 1: residual: a - 1".to_string(),
                "  ↳ counterexample hint: a=0 gives residual -1".to_string(),
            ]
        );
    }

    #[test]
    fn renders_none_verified_details_without_counterexample_hint_when_disabled() {
        let mut ctx = Context::new();
        let one = ctx.num(1);
        let verify_result = crate::VerifyResult {
            solutions: vec![(
                one,
                crate::VerifyStatus::Unverifiable {
                    residual: one,
                    reason: "residual: a - 1".to_string(),
                    counterexample_hint: Some(
                        "counterexample hint: a=0 gives residual -1".to_string(),
                    ),
                },
            )],
            summary: crate::VerifySummary::NoneVerified,
            guard_description: None,
        };

        let lines = format_verify_summary_lines_with_hints(&ctx, "x", &verify_result, "  ", false);
        assert_eq!(
            lines,
            vec![
                "⚠ No solutions could be verified".to_string(),
                "  ⚠ x = 1: residual: a - 1".to_string(),
            ]
        );
    }

    #[test]
    fn renders_partially_verified_with_non_discrete_note() {
        let mut ctx = Context::new();
        let one = ctx.num(1);
        let verify_result = crate::VerifyResult {
            solutions: vec![(one, crate::VerifyStatus::Verified)],
            summary: crate::VerifySummary::PartiallyVerified,
            guard_description: Some(
                "some non-discrete branches require numeric sampling".to_string(),
            ),
        };

        let lines = format_verify_summary_lines(&ctx, "x", &verify_result, "  ");
        assert_eq!(
            lines,
            vec![
                "⚠ Some solutions verified".to_string(),
                "  ✓ x = 1 verified".to_string(),
                "  ℹ some non-discrete branches require numeric sampling".to_string(),
            ]
        );
    }
}
