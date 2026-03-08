use cas_ast::Context;

/// Format verification summary lines after solve result checking.
pub fn format_verify_summary_lines(
    ctx: &Context,
    var: &str,
    verify_result: &crate::VerifyResult,
    detail_prefix: &str,
) -> Vec<String> {
    let mut lines = Vec::new();

    match verify_result.summary {
        crate::VerifySummary::AllVerified => {
            lines.push("✓ All solutions verified".to_string());
        }
        crate::VerifySummary::PartiallyVerified => {
            lines.push("⚠ Some solutions verified".to_string());
            for (sol_id, status) in &verify_result.solutions {
                let sol_str = cas_formatter::render_expr(ctx, *sol_id);
                match status {
                    crate::VerifyStatus::Verified => {
                        lines.push(format!("{detail_prefix}✓ {var} = {sol_str} verified"));
                    }
                    crate::VerifyStatus::Unverifiable { reason, .. } => {
                        lines.push(format!("{detail_prefix}⚠ {var} = {sol_str}: {reason}"));
                    }
                    crate::VerifyStatus::NotCheckable { reason } => {
                        lines.push(format!("{detail_prefix}ℹ {var} = {sol_str}: {reason}"));
                    }
                }
            }
        }
        crate::VerifySummary::NoneVerified => {
            lines.push("⚠ No solutions could be verified".to_string());
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
