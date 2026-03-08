use super::*;
use crate::types::EntryKind;

fn contains_integer(ctx: &Context, root: ExprId, value: i64) -> bool {
    cas_ast::traversal::count_nodes_matching(ctx, root, |node| match node {
        Expr::Number(n) => n.is_integer() && n.to_integer() == value.into(),
        _ => false,
    }) > 0
}

fn has_session_ref(ctx: &Context, root: ExprId) -> bool {
    cas_ast::traversal::count_nodes_matching(ctx, root, |node| matches!(node, Expr::SessionRef(_)))
        > 0
}

#[test]
fn parse_legacy_session_ref_valid() {
    assert_eq!(parse_legacy_session_ref("#1"), Some(1));
    assert_eq!(parse_legacy_session_ref("#42"), Some(42));
}

#[test]
fn first_session_ref_finds_nested_reference() {
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let ref7 = ctx.add(Expr::SessionRef(7));
    let nested = ctx.add(Expr::Mul(ref7, x));
    let expr = ctx.add(Expr::Add(x, nested));
    assert_eq!(first_session_ref(&ctx, expr), Some(7));
}

#[test]
fn first_session_ref_returns_none_without_references() {
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let one = ctx.num(1);
    let expr = ctx.add(Expr::Add(x, one));
    assert_eq!(first_session_ref(&ctx, expr), None);
}

#[test]
fn parse_legacy_session_ref_invalid() {
    assert_eq!(parse_legacy_session_ref("x"), None);
    assert_eq!(parse_legacy_session_ref("#"), None);
    assert_eq!(parse_legacy_session_ref("#abc"), None);
}

#[test]
fn rewrite_session_refs_replaces_explicit_ref() {
    let mut ctx = Context::new();
    let one = ctx.num(1);
    let ref1 = ctx.add(Expr::SessionRef(1));
    let input = ctx.add(Expr::Add(ref1, one));

    let out = rewrite_session_refs(&mut ctx, input, &mut |ctx,
                                                          _node,
                                                          id|
     -> Result<ExprId, ()> {
        Ok(ctx.num(id as i64))
    })
    .unwrap();
    assert!(contains_integer(&ctx, out, 1));
    assert!(cas_ast::traversal::collect_variables(&ctx, out).is_empty());
    assert!(!has_session_ref(&ctx, out));
}

#[test]
fn rewrite_session_refs_replaces_legacy_variable() {
    let mut ctx = Context::new();
    let legacy = ctx.var("#2");
    let input = ctx.add(Expr::Neg(legacy));

    let out = rewrite_session_refs(&mut ctx, input, &mut |ctx,
                                                          _node,
                                                          id|
     -> Result<ExprId, ()> {
        Ok(ctx.num(id as i64))
    })
    .unwrap();
    assert!(!cas_ast::traversal::collect_variables(&ctx, out).contains("#2"));
    assert!(!has_session_ref(&ctx, out));
}

#[test]
fn resolve_with_lookup_basic_expr() {
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let one = ctx.num(1);
    let x_plus_1 = ctx.add(Expr::Add(x, one));

    let ref1 = ctx.add(Expr::SessionRef(1));
    let input = ctx.add(Expr::Add(ref1, one));
    let mut lookup = |id: EntryId| match id {
        1 => Some(EntryKind::Expr(x_plus_1)),
        _ => None,
    };

    let out = resolve_session_refs_with_lookup(&mut ctx, input, &mut lookup).unwrap();
    assert!(cas_ast::traversal::collect_variables(&ctx, out).contains("x"));
    assert!(!has_session_ref(&ctx, out));
}

#[test]
fn resolve_with_lookup_not_found() {
    let mut ctx = Context::new();
    let input = ctx.add(Expr::SessionRef(99));
    let mut lookup = |_id: EntryId| None;

    let err = resolve_session_refs_with_lookup(&mut ctx, input, &mut lookup).unwrap_err();
    assert_eq!(err, ResolveError::NotFound(99));
}

#[test]
fn resolve_all_with_lookup_and_env_applies_bindings() {
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let one = ctx.num(1);
    let five = ctx.num(5);

    let x_plus_one = ctx.add(Expr::Add(x, one));
    let ref1 = ctx.add(Expr::SessionRef(1));
    let input = ctx.add(Expr::Mul(ref1, x));

    let mut lookup = |id: EntryId| match id {
        1 => Some(EntryKind::Expr(x_plus_one)),
        _ => None,
    };
    let mut env = crate::env::Environment::new();
    env.set("x".to_string(), five);

    let out = resolve_all_with_lookup_and_env(&mut ctx, input, &mut lookup, &env).unwrap();
    let vars = cas_ast::traversal::collect_variables(&ctx, out);
    assert!(vars.is_empty());
}

#[test]
fn resolve_with_lookup_cycle() {
    let mut ctx = Context::new();
    let ref1 = ctx.add(Expr::SessionRef(1));
    let ref2 = ctx.add(Expr::SessionRef(2));
    let input = ref1;

    let mut lookup = |id: EntryId| match id {
        1 => Some(EntryKind::Expr(ref2)),
        2 => Some(EntryKind::Expr(ref1)),
        _ => None,
    };

    let err = resolve_session_refs_with_lookup(&mut ctx, input, &mut lookup).unwrap_err();
    assert_eq!(err, ResolveError::CircularReference(1));
}

#[test]
fn resolve_with_lookup_accumulator_collects_visit_order_metadata() {
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let one = ctx.num(1);
    let ref1 = ctx.add(Expr::SessionRef(1));
    let input = ctx.add(Expr::Add(ref1, one));

    let mut lookup = |id: EntryId| match id {
        1 => Some(EntryKind::Expr(x)),
        _ => None,
    };

    let (resolved, visits) = resolve_session_refs_with_lookup_accumulator(
        &mut ctx,
        input,
        &mut lookup,
        Vec::<EntryId>::new(),
        |acc, id| acc.push(id),
    )
    .unwrap();

    assert_eq!(visits, vec![1]);
    assert!(!has_session_ref(&ctx, resolved));
}

#[test]
fn resolve_all_with_mode_lookup_and_env_applies_bindings() {
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let one = ctx.num(1);
    let five = ctx.num(5);
    let x_plus_one = ctx.add(Expr::Add(x, one));
    let ref1 = ctx.add(Expr::SessionRef(1));
    let input = ctx.add(Expr::Mul(ref1, x));

    let mut lookup = |id: EntryId| match id {
        1 => Some(ModeEntry {
            kind: EntryKind::Expr(x_plus_one),
            requires: vec![],
            cache: None,
        }),
        _ => None,
    };
    let mut same_requirement = |_lhs: &(), _rhs: &()| true;
    let mut mark_session_propagated = |_item: &mut ()| {};
    let mut env = crate::env::Environment::new();
    env.set("x".to_string(), five);

    let resolved = resolve_all_with_mode_lookup_and_env(
        &mut ctx,
        input,
        ModeResolveConfig {
            mode: crate::types::RefMode::Raw,
            cache_key: &0u8,
            env: &env,
        },
        &mut lookup,
        &mut same_requirement,
        &mut mark_session_propagated,
    )
    .unwrap();

    let vars = cas_ast::traversal::collect_variables(&ctx, resolved.expr);
    assert!(vars.is_empty());
    assert!(resolved.cache_hits.is_empty());
}

#[test]
fn cache_hit_entry_ids_preserves_order() {
    let mut ctx = Context::new();
    let r1 = ctx.add(Expr::SessionRef(1));
    let r2 = ctx.add(Expr::SessionRef(2));
    let a = ctx.num(10);
    let b = ctx.num(20);

    let hits = vec![
        CacheHitTrace {
            entry_id: 7,
            before_ref_expr: r1,
            after_expr: a,
            requires: vec![()],
        },
        CacheHitTrace {
            entry_id: 2,
            before_ref_expr: r2,
            after_expr: b,
            requires: vec![()],
        },
    ];

    assert_eq!(cache_hit_entry_ids(&hits), vec![7, 2]);
}

#[test]
fn inherited_diagnostics_from_requires_applies_all_items() {
    #[derive(Default)]
    struct Diag {
        values: Vec<i32>,
    }

    let out = inherited_diagnostics_from_requires(vec![1, 2, 3], Diag::default(), |diag, item| {
        diag.values.push(item * 2);
    });

    assert_eq!(out.values, vec![2, 4, 6]);
}

#[test]
fn resolve_mode_with_env_and_diagnostics_collects_requires_and_cache_hits() {
    #[derive(Default)]
    struct Diag {
        requires: Vec<i32>,
    }

    let mut ctx = Context::new();
    let x = ctx.var("x");
    let one = ctx.num(1);
    let five = ctx.num(5);
    let x_plus_one = ctx.add(Expr::Add(x, one));
    let ref1 = ctx.add(Expr::SessionRef(1));

    let mut lookup = |id: EntryId| match id {
        1 => Some(ModeEntry {
            kind: EntryKind::Expr(x),
            requires: vec![10],
            cache: Some(ModeCacheEntry {
                key: 42u8,
                expr: x_plus_one,
                requires: vec![20],
            }),
        }),
        _ => None,
    };
    let mut same_requirement = |lhs: &i32, rhs: &i32| lhs == rhs;
    let mut mark_session_propagated = |_item: &mut i32| {};
    let mut env = crate::env::Environment::new();
    env.set("x".to_string(), five);

    let (resolved_expr, diagnostics, cache_hits) = resolve_mode_with_env_and_diagnostics(
        &mut ctx,
        ref1,
        ModeResolveConfig {
            mode: crate::types::RefMode::PreferSimplified,
            cache_key: &42u8,
            env: &env,
        },
        &mut lookup,
        &mut same_requirement,
        &mut mark_session_propagated,
        Diag::default(),
        |diag, item| diag.requires.push(item),
    )
    .unwrap();

    let vars = cas_ast::traversal::collect_variables(&ctx, resolved_expr);
    assert!(vars.is_empty());
    assert_eq!(diagnostics.requires, vec![20]);
    assert_eq!(cache_hits, vec![1]);
}
