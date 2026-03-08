use super::{
    apply_post_dispatch_store_updates, apply_pre_dispatch_store_updates, build_cache_hit_step_with,
    collect_step_items_with_display_dedup, collect_warnings_with, format_cache_hit_summary,
    map_resolve_error_to_anyhow, resolve_and_prepare_dispatch, resolve_eval_input, store_raw_input,
    touch_cache_hits, EvalSession, EvalStore, ResolvePrepareConfig, SimplifiedUpdate,
};
use cas_ast::{Context, ExprId};
use std::sync::Arc;

#[derive(Default)]
struct TestStore {
    push_expr_calls: Vec<(cas_ast::ExprId, String)>,
    push_eq_calls: Vec<(cas_ast::ExprId, cas_ast::ExprId, String)>,
    touched: Vec<u64>,
    diagnostics_updates: Vec<u64>,
    simplified_updates: Vec<(u64, cas_ast::ExprId)>,
}

#[derive(Default)]
struct TestSession {
    store: TestStore,
    options: (),
}

impl EvalStore for TestStore {
    type DomainMode = ();
    type RequiredItem = ();
    type Step = ();
    type Diagnostics = ();

    fn push_raw_expr(&mut self, expr: cas_ast::ExprId, raw_input: String) -> u64 {
        self.push_expr_calls.push((expr, raw_input));
        11
    }

    fn push_raw_equation(
        &mut self,
        lhs: cas_ast::ExprId,
        rhs: cas_ast::ExprId,
        raw_input: String,
    ) -> u64 {
        self.push_eq_calls.push((lhs, rhs, raw_input));
        22
    }

    fn touch_cached(&mut self, entry_id: u64) {
        self.touched.push(entry_id);
    }

    fn update_diagnostics(&mut self, id: u64, _diagnostics: Self::Diagnostics) {
        self.diagnostics_updates.push(id);
    }

    fn update_simplified(
        &mut self,
        id: u64,
        _domain: Self::DomainMode,
        expr: cas_ast::ExprId,
        _requires: Vec<Self::RequiredItem>,
        _steps: Option<Arc<Vec<Self::Step>>>,
    ) {
        self.simplified_updates.push((id, expr));
    }
}

impl EvalSession for TestSession {
    type Store = TestStore;
    type Options = ();
    type Diagnostics = ();

    fn store_mut(&mut self) -> &mut Self::Store {
        &mut self.store
    }

    fn options(&self) -> &Self::Options {
        &self.options
    }

    fn options_mut(&mut self) -> &mut Self::Options {
        &mut self.options
    }

    fn resolve_all_with_diagnostics(
        &self,
        _ctx: &mut Context,
        expr: ExprId,
    ) -> anyhow::Result<(ExprId, Self::Diagnostics, Vec<u64>)> {
        Ok((expr, (), vec![1]))
    }
}

#[test]
fn format_cache_hit_summary_sorts_and_formats_ids() {
    let summary = format_cache_hit_summary(&[7, 2, 5], 6).expect("summary");
    assert_eq!(summary, "Used cached simplified result from #2, #5, #7");
}

#[test]
fn format_cache_hit_summary_truncates_with_suffix() {
    let summary = format_cache_hit_summary(&[1, 2, 3, 4], 2).expect("summary");
    assert_eq!(summary, "Used cached simplified result from #1, #2 (+2)");
}

#[test]
fn map_resolve_error_messages_are_stable() {
    let not_found =
        map_resolve_error_to_anyhow(crate::types::ResolveError::NotFound(9)).to_string();
    let cycle =
        map_resolve_error_to_anyhow(crate::types::ResolveError::CircularReference(3)).to_string();
    assert_eq!(not_found, "Session reference #9 not found");
    assert_eq!(cycle, "Circular reference detected involving #3");
}

#[test]
fn store_raw_input_handles_expr_and_equation() {
    let mut ctx = Context::new();
    let lhs = ctx.var("x");
    let rhs = ctx.num(0);
    let equation = cas_ast::eq::wrap_eq(&mut ctx, lhs, rhs);
    let expr = ctx.add_raw(cas_ast::Expr::Add(lhs, rhs));

    let mut store = TestStore::default();
    let id_eq = store_raw_input(&mut store, &ctx, equation, "x=0".to_string(), true);
    let id_expr = store_raw_input(&mut store, &ctx, expr, "x+0".to_string(), true);
    let id_none = store_raw_input(&mut store, &ctx, expr, "ignored".to_string(), false);

    assert_eq!(id_eq, Some(22));
    assert_eq!(id_expr, Some(11));
    assert_eq!(id_none, None);
    assert_eq!(store.push_eq_calls.len(), 1);
    assert_eq!(store.push_expr_calls.len(), 1);
}

#[test]
fn touch_cache_hits_forwards_all_ids() {
    let mut store = TestStore::default();
    touch_cache_hits(&mut store, &[8, 2, 8]);
    assert_eq!(store.touched, vec![8, 2, 8]);
}

#[test]
fn resolve_eval_input_resolves_main_and_other() {
    let mut ctx = Context::new();
    let a = ctx.var("a");
    let b = ctx.var("b");
    let session = TestSession::default();

    let out = resolve_eval_input(&session, &mut ctx, a, Some(b)).expect("resolved");
    assert_eq!(out.resolved, a);
    assert_eq!(out.resolved_equiv_other, Some(b));
    assert_eq!(out.cache_hits, vec![1]);
}

#[test]
fn apply_pre_dispatch_store_updates_stores_and_touches() {
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let mut store = TestStore::default();
    let out = apply_pre_dispatch_store_updates(&mut store, &ctx, x, "x".to_string(), true, &[4, 9]);
    assert_eq!(out, Some(11));
    assert_eq!(store.push_expr_calls.len(), 1);
    assert_eq!(store.touched, vec![4, 9]);
}

#[test]
fn build_cache_hit_step_with_builds_payload_when_hits_exist() {
    let mut ctx = Context::new();
    let original = ctx.var("x");
    let resolved = ctx.var("y");

    let step = build_cache_hit_step_with(
        &[3, 1],
        6,
        original,
        resolved,
        |description, before, after| (description, before, after),
    )
    .expect("step");

    assert_eq!(
        step.0,
        "Used cached simplified result from #1, #3".to_string()
    );
    assert_eq!(step.1, original);
    assert_eq!(step.2, resolved);
}

#[test]
fn build_cache_hit_step_with_returns_none_without_hits() {
    let mut ctx = Context::new();
    let original = ctx.var("x");
    let resolved = ctx.var("y");
    let step =
        build_cache_hit_step_with::<(), _>(&[], 6, original, resolved, |_description, _, _| ());
    assert!(step.is_none());
}

#[test]
fn apply_post_dispatch_store_updates_noops_without_stored_id() {
    let mut ctx = Context::new();
    let expr = ctx.var("x");
    let mut store = TestStore::default();
    apply_post_dispatch_store_updates(
        &mut store,
        None,
        (),
        Some(SimplifiedUpdate {
            domain: (),
            expr,
            requires: vec![],
            steps: None,
        }),
    );
    assert!(store.diagnostics_updates.is_empty());
    assert!(store.simplified_updates.is_empty());
}

#[test]
fn apply_post_dispatch_store_updates_writes_diagnostics_and_simplified() {
    let mut ctx = Context::new();
    let expr = ctx.var("z");
    let mut store = TestStore::default();
    apply_post_dispatch_store_updates(
        &mut store,
        Some(42),
        (),
        Some(SimplifiedUpdate {
            domain: (),
            expr,
            requires: vec![],
            steps: None,
        }),
    );
    assert_eq!(store.diagnostics_updates, vec![42]);
    assert_eq!(store.simplified_updates, vec![(42, expr)]);
}

#[test]
fn apply_post_dispatch_store_updates_writes_only_diagnostics_without_simplified() {
    let mut store = TestStore::default();
    apply_post_dispatch_store_updates(&mut store, Some(7), (), None);
    assert_eq!(store.diagnostics_updates, vec![7]);
    assert!(store.simplified_updates.is_empty());
}

#[derive(Clone)]
struct WarningStep {
    rule: &'static str,
    events: Vec<(&'static str, bool)>, // (message, skip)
}

#[test]
fn collect_warnings_with_dedupes_by_message() {
    let steps = vec![
        WarningStep {
            rule: "r1",
            events: vec![("a", false), ("b", false)],
        },
        WarningStep {
            rule: "r2",
            events: vec![("b", false), ("c", false)],
        },
    ];

    let out = collect_warnings_with(
        &steps,
        |s| s.events.clone(),
        |e| e.1,
        |e| e.0.to_string(),
        |s| s.rule.to_string(),
        |message, rule| (message, rule),
    );

    assert_eq!(
        out,
        vec![
            ("a".to_string(), "r1".to_string()),
            ("b".to_string(), "r1".to_string()),
            ("c".to_string(), "r2".to_string())
        ]
    );
}

#[test]
fn collect_warnings_with_applies_skip_filter() {
    let steps = vec![WarningStep {
        rule: "r1",
        events: vec![("a", true), ("b", false)],
    }];

    let out = collect_warnings_with(
        &steps,
        |s| s.events.clone(),
        |e| e.1,
        |e| e.0.to_string(),
        |s| s.rule.to_string(),
        |message, rule| (message, rule),
    );

    assert_eq!(out, vec![("b".to_string(), "r1".to_string())]);
}

#[test]
fn collect_step_items_with_display_dedup_keeps_first_item_per_key() {
    #[derive(Clone)]
    struct Step {
        items: Vec<&'static str>,
    }

    let steps = vec![
        Step {
            items: vec!["a", "b"],
        },
        Step {
            items: vec!["b", "c"],
        },
    ];

    let out = collect_step_items_with_display_dedup(
        &steps,
        |s| s.items.clone(),
        |item| item.to_string(),
        |item| item.to_string(),
    );

    assert_eq!(out, vec!["a".to_string(), "b".to_string(), "c".to_string()]);
}

#[test]
fn resolve_and_prepare_dispatch_resolves_stores_and_builds_cache_step() {
    let mut ctx = Context::new();
    let parsed = ctx.var("x");
    let mut session = TestSession::default();

    let out = resolve_and_prepare_dispatch(
        &mut session,
        &mut ctx,
        ResolvePrepareConfig {
            parsed,
            raw_input: "x".to_string(),
            auto_store: true,
            equiv_other: None,
            cache_step_max_shown: 6,
        },
        |_, description, before, after| (description, before, after),
    )
    .expect("prepared");

    assert_eq!(out.stored_id, Some(11));
    assert_eq!(out.resolved, parsed);
    assert_eq!(out.resolved_equiv_other, None);
    assert_eq!(out.inherited_diagnostics, ());
    assert_eq!(
        out.cache_hit_step,
        Some((
            "Used cached simplified result from #1".to_string(),
            parsed,
            parsed
        ))
    );
    assert_eq!(session.store.push_expr_calls.len(), 1);
    assert_eq!(session.store.touched, vec![1]);
}
