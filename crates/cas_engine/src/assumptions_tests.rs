use crate::{
    collect_assumption_records, collect_assumption_records_from_iter, AssumptionCollector,
    AssumptionEvent, AssumptionReporting,
};
use cas_ast::Context;

#[test]
fn test_dedup_same_expr() {
    let mut ctx = Context::new();
    let x = ctx.var("x");

    let mut collector = AssumptionCollector::new();

    // Note the same assumption 3 times
    collector.note(AssumptionEvent::nonzero(&ctx, x));
    collector.note(AssumptionEvent::nonzero(&ctx, x));
    collector.note(AssumptionEvent::nonzero(&ctx, x));

    let records = collector.finish();

    assert_eq!(records.len(), 1, "Should dedup to single record");
    assert_eq!(records[0].count, 3, "Count should be 3");
    assert_eq!(records[0].kind, "nonzero");
    assert_eq!(records[0].expr, "x");
}

#[test]
fn test_different_exprs_separate() {
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let y = ctx.var("y");

    let mut collector = AssumptionCollector::new();

    collector.note(AssumptionEvent::nonzero(&ctx, x));
    collector.note(AssumptionEvent::nonzero(&ctx, y));

    let records = collector.finish();

    assert_eq!(records.len(), 2, "Different exprs should be separate");
}

#[test]
fn test_different_kinds_separate() {
    let mut ctx = Context::new();
    let x = ctx.var("x");

    let mut collector = AssumptionCollector::new();

    collector.note(AssumptionEvent::nonzero(&ctx, x));
    collector.note(AssumptionEvent::positive(&ctx, x));

    let records = collector.finish();

    assert_eq!(records.len(), 2, "Different kinds should be separate");
}

#[test]
fn test_stable_order() {
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let y = ctx.var("y");
    let z = ctx.var("z");

    let mut collector = AssumptionCollector::new();

    // Add in arbitrary order
    collector.note(AssumptionEvent::positive(&ctx, z));
    collector.note(AssumptionEvent::nonzero(&ctx, y));
    collector.note(AssumptionEvent::nonzero(&ctx, x));

    let records = collector.finish();

    // Should be sorted: nonzero before positive, then by expr
    assert_eq!(records[0].kind, "nonzero");
    assert_eq!(records[0].expr, "x");
    assert_eq!(records[1].kind, "nonzero");
    assert_eq!(records[1].expr, "y");
    assert_eq!(records[2].kind, "positive");
    assert_eq!(records[2].expr, "z");
}

#[test]
fn test_summary_line() {
    let mut ctx = Context::new();
    let x = ctx.var("x");

    let mut collector = AssumptionCollector::new();
    collector.note(AssumptionEvent::nonzero(&ctx, x));
    collector.note(AssumptionEvent::nonzero(&ctx, x));

    let summary = collector.summary_line();

    assert!(summary.is_some());
    let line = summary.expect("summary line should exist");
    assert!(line.contains("nonzero(x)"));
    assert!(line.contains("×2"));
}

#[test]
fn test_empty_collector() {
    let collector = AssumptionCollector::new();

    assert!(collector.is_empty());
    assert_eq!(collector.len(), 0);
    assert!(collector.summary_line().is_none());

    let records = collector.finish();
    assert!(records.is_empty());
}

#[test]
fn test_collect_assumption_records_dedups_events() {
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let y = ctx.var("y");
    let events = vec![
        AssumptionEvent::nonzero(&ctx, x),
        AssumptionEvent::nonzero(&ctx, x),
        AssumptionEvent::positive(&ctx, y),
    ];

    let records = collect_assumption_records(&events);
    assert_eq!(records.len(), 2);
    assert!(records.iter().any(|r| r.kind == "nonzero" && r.count == 2));
    assert!(records.iter().any(|r| r.kind == "positive" && r.count == 1));
}

#[test]
fn test_collect_assumption_records_from_iter_dedups_events() {
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let y = ctx.var("y");

    let records = collect_assumption_records_from_iter([
        AssumptionEvent::nonzero(&ctx, x),
        AssumptionEvent::positive(&ctx, y),
        AssumptionEvent::nonzero(&ctx, x),
    ]);
    assert_eq!(records.len(), 2);
    assert!(records.iter().any(|r| r.kind == "nonzero" && r.count == 2));
    assert!(records.iter().any(|r| r.kind == "positive" && r.count == 1));
}

#[test]
fn test_assumption_event_from_log_assumption_targets_base_and_rhs() {
    let mut ctx = Context::new();
    let base = ctx.var("b");
    let rhs = ctx.var("r");

    let base_event = cas_solver_core::assumption_model::assumption_event_from_log_assumption(
        &ctx,
        cas_solver_core::log_domain::LogAssumption::PositiveBase,
        base,
        rhs,
    );
    let rhs_event = cas_solver_core::assumption_model::assumption_event_from_log_assumption(
        &ctx,
        cas_solver_core::log_domain::LogAssumption::PositiveRhs,
        base,
        rhs,
    );

    assert_eq!(base_event.expr_id, Some(base));
    assert_eq!(rhs_event.expr_id, Some(rhs));
}

#[test]
fn test_map_log_blocked_hint_to_domain_hint_preserves_payload() {
    let mut ctx = Context::new();
    let base = ctx.var("b");
    let hint = cas_solver_core::solve_outcome::LogBlockedHintRecord {
        assumption: cas_solver_core::log_domain::LogAssumption::PositiveBase,
        expr_id: base,
        rule: "Take log of both sides",
        suggestion: "use `semantics set domain assume`",
    };

    let blocked = cas_solver_core::assumption_model::map_log_blocked_hint(&ctx, hint);
    let expected = AssumptionEvent::positive(&ctx, base);

    assert_eq!(blocked.key, expected.key);
    assert_eq!(blocked.expr_id, base);
    assert_eq!(blocked.rule, "Take log of both sides");
    assert_eq!(blocked.suggestion, "use `semantics set domain assume`");
}

#[test]
fn test_reporting_from_str() {
    assert_eq!(
        AssumptionReporting::parse("off"),
        Some(AssumptionReporting::Off)
    );
    assert_eq!(
        AssumptionReporting::parse("summary"),
        Some(AssumptionReporting::Summary)
    );
    assert_eq!(
        AssumptionReporting::parse("trace"),
        Some(AssumptionReporting::Trace)
    );
    assert_eq!(AssumptionReporting::parse("invalid"), None);
}
