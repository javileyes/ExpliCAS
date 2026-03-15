use cas_ast::Context;
use cas_solver::api::{VerifyResult, VerifyStatus, VerifySummary};
use cas_solver::session_api::solve::{
    format_verify_summary_lines, format_verify_summary_lines_with_hints,
};

#[test]
fn session_api_verify_formatter_surfaces_counterexample_hint_by_default() {
    let mut ctx = Context::new();
    let one = ctx.num(1);
    let verify_result = VerifyResult {
        solutions: vec![(
            one,
            VerifyStatus::Unverifiable {
                residual: one,
                reason: "residual: a - 1".to_string(),
                counterexample_hint: Some("counterexample hint: a=0 gives residual -1".to_string()),
            },
        )],
        summary: VerifySummary::NoneVerified,
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
fn session_api_verify_formatter_hides_counterexample_hint_when_disabled() {
    let mut ctx = Context::new();
    let one = ctx.num(1);
    let verify_result = VerifyResult {
        solutions: vec![(
            one,
            VerifyStatus::Unverifiable {
                residual: one,
                reason: "residual: a - 1".to_string(),
                counterexample_hint: Some("counterexample hint: a=0 gives residual -1".to_string()),
            },
        )],
        summary: VerifySummary::NoneVerified,
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
