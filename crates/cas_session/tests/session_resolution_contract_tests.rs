//! Tests for session store and reference resolution.

use cas_ast::ExprId;
use cas_session::*;

#[test]
fn test_push_and_get() {
    let mut store = SessionStore::new();

    // Create a dummy ExprId (in real usage this comes from Context)
    let expr_id = ExprId::from_raw(0);

    let id1 = store.push(EntryKind::Expr(expr_id), "x + 1".to_string());
    let id2 = store.push(EntryKind::Expr(expr_id), "x^2".to_string());

    assert_eq!(id1, 1);
    assert_eq!(id2, 2);

    let entry1 = store.get(1).unwrap();
    assert_eq!(entry1.raw_text, "x + 1");
    assert!(entry1.is_expr());

    let entry2 = store.get(2).unwrap();
    assert_eq!(entry2.raw_text, "x^2");
}

#[test]
fn test_ids_not_reused_after_delete() {
    let mut store = SessionStore::new();
    let expr_id = ExprId::from_raw(0);

    let id1 = store.push(EntryKind::Expr(expr_id), "a".to_string());
    let id2 = store.push(EntryKind::Expr(expr_id), "b".to_string());

    // Delete id1
    store.remove(&[id1]);

    // Next ID should be 3, not 1
    let id3 = store.push(EntryKind::Expr(expr_id), "c".to_string());
    assert_eq!(id3, 3);
    assert!(!store.contains(id1));
    assert!(store.contains(id2));
    assert!(store.contains(id3));
}

#[test]
fn test_remove_multiple() {
    let mut store = SessionStore::new();
    let expr_id = ExprId::from_raw(0);

    store.push(EntryKind::Expr(expr_id), "a".to_string());
    store.push(EntryKind::Expr(expr_id), "b".to_string());
    store.push(EntryKind::Expr(expr_id), "c".to_string());

    store.remove(&[1, 3]);
    assert_eq!(store.len(), 1);
    assert!(store.contains(2));
}

#[test]
fn test_clear() {
    let mut store = SessionStore::new();
    let expr_id = ExprId::from_raw(0);

    store.push(EntryKind::Expr(expr_id), "a".to_string());
    store.push(EntryKind::Expr(expr_id), "b".to_string());

    store.clear();
    assert!(store.is_empty());

    // Next ID should still be 3
    let id = store.push(EntryKind::Expr(expr_id), "c".to_string());
    assert_eq!(id, 3);
}

#[test]
fn test_equation_entry() {
    let mut store = SessionStore::new();
    let lhs = ExprId::from_raw(0);
    let rhs = ExprId::from_raw(1);

    let id = store.push(EntryKind::Eq { lhs, rhs }, "x + 1 = 5".to_string());

    let entry = store.get(id).unwrap();
    assert!(entry.is_eq());
    assert_eq!(entry.type_str(), "Eq");
}

// ========== resolve_session_refs Tests ==========

#[test]
fn test_resolve_simple_ref() {
    use cas_ast::{Context, Expr};
    use cas_formatter::DisplayExpr;

    let mut ctx = Context::new();
    let mut store = SessionStore::new();

    // Store x + 1 as #1
    let x = ctx.var("x");
    let one = ctx.num(1);
    let expr1 = ctx.add(Expr::Add(x, one));
    store.push(EntryKind::Expr(expr1), "x + 1".to_string());

    // Create #1 * 2
    let ref1 = ctx.add(Expr::SessionRef(1));
    let two = ctx.num(2);
    let input = ctx.add(Expr::Mul(ref1, two));

    // Resolve
    let resolved = resolve_session_refs(&mut ctx, input, &store).unwrap();

    // Check using DisplayExpr - should contain (x + 1) and 2
    let display = format!(
        "{}",
        DisplayExpr {
            context: &ctx,
            id: resolved
        }
    );
    // Resolved should not contain "#" anymore
    assert!(
        !display.contains('#'),
        "Resolved should not contain session refs: {}",
        display
    );
    // Should contain x and 2
    assert!(display.contains('x'), "Should contain x: {}", display);
    assert!(display.contains('2'), "Should contain 2: {}", display);
    // Should be a multiplication
    assert!(
        display.contains('*'),
        "Should contain multiplication: {}",
        display
    );
}

#[test]
fn test_resolve_not_found() {
    use cas_ast::{Context, Expr};

    let mut ctx = Context::new();
    let store = SessionStore::new();

    // Reference to non-existent #99
    let ref99 = ctx.add(Expr::SessionRef(99));

    let result = resolve_session_refs(&mut ctx, ref99, &store);
    assert!(matches!(result, Err(ResolveError::NotFound(99))));
}

#[test]
fn test_resolve_equation_as_residue() {
    use cas_ast::{Context, Expr};

    let mut ctx = Context::new();
    let mut store = SessionStore::new();

    // Store equation: x + 1 = 5 as #1
    let x = ctx.var("x");
    let one = ctx.num(1);
    let five = ctx.num(5);
    let lhs = ctx.add(Expr::Add(x, one));
    store.push(EntryKind::Eq { lhs, rhs: five }, "x + 1 = 5".to_string());

    // Create just #1
    let ref1 = ctx.add(Expr::SessionRef(1));

    // Resolve - should get (x + 1) - 5
    let resolved = resolve_session_refs(&mut ctx, ref1, &store).unwrap();

    // Should be Sub
    if let Expr::Sub(l, r) = ctx.get(resolved) {
        // Left should be (x + 1)
        assert!(matches!(ctx.get(*l), Expr::Add(_, _)));
        // Right should be 5
        if let Expr::Number(n) = ctx.get(*r) {
            assert_eq!(n.to_integer(), 5.into());
        } else {
            panic!("Expected Number(5)");
        }
    } else {
        panic!("Expected Sub for equation residue");
    }
}

#[test]
fn test_resolve_no_refs() {
    use cas_ast::{Context, Expr};

    let mut ctx = Context::new();
    let store = SessionStore::new();

    // Expression without refs: x + 1
    let x = ctx.var("x");
    let one = ctx.num(1);
    let input = ctx.add(Expr::Add(x, one));

    // Should return same expression
    let resolved = resolve_session_refs(&mut ctx, input, &store).unwrap();
    assert_eq!(resolved, input);
}

#[test]
fn test_resolve_chained_refs() {
    use cas_ast::{Context, Expr};
    use cas_formatter::DisplayExpr;

    let mut ctx = Context::new();
    let mut store = SessionStore::new();

    // #1 = x
    let x = ctx.var("x");
    store.push(EntryKind::Expr(x), "x".to_string());

    // #2 = #1 + 1 (references #1)
    let ref1 = ctx.add(Expr::SessionRef(1));
    let one = ctx.num(1);
    let expr2 = ctx.add(Expr::Add(ref1, one));
    store.push(EntryKind::Expr(expr2), "#1 + 1".to_string());

    // Input: #2 * 2
    let ref2 = ctx.add(Expr::SessionRef(2));
    let two = ctx.num(2);
    let input = ctx.add(Expr::Mul(ref2, two));

    // Resolve - should get (x + 1) * 2
    let resolved = resolve_session_refs(&mut ctx, input, &store).unwrap();

    // Check using DisplayExpr
    let display = format!(
        "{}",
        DisplayExpr {
            context: &ctx,
            id: resolved
        }
    );
    // Should not contain any # refs
    assert!(
        !display.contains('#'),
        "Resolved should not contain session refs: {}",
        display
    );
    // Should contain x, 1, 2 and be a multiplication
    assert!(display.contains('x'), "Should contain x: {}", display);
    assert!(display.contains('2'), "Should contain 2: {}", display);
    assert!(
        display.contains('*'),
        "Should contain multiplication: {}",
        display
    );
}

// ========== Phase 2: resolve_session_refs_with_mode Tests ==========

#[test]
fn test_resolve_with_mode_cache_hit() {
    use cas_ast::{Context, Expr};

    let mut ctx = Context::new();
    let mut store = SessionStore::new();

    // Store raw expr: 5*sqrt(x)/sqrt(x) as #1
    let x = ctx.var("x");
    let sqrt_x = ctx.call_builtin(cas_ast::BuiltinFn::Sqrt, vec![x]);
    let five = ctx.num(5);
    let mul = ctx.add(Expr::Mul(five, sqrt_x));
    let raw_expr = ctx.add(Expr::Div(mul, sqrt_x));
    store.push(EntryKind::Expr(raw_expr), "5*sqrt(x)/sqrt(x)".to_string());

    // Inject simplified cache: simplified = 5
    let simplified_five = ctx.num(5);
    let cache_key = SimplifyCacheKey::from_context(cas_engine::DomainMode::Generic);
    let cache = SimplifiedCache {
        key: cache_key.clone(),
        expr: simplified_five,
        requires: vec![],
        steps: Some(std::sync::Arc::new(vec![])),
    };
    store.update_simplified(1, cache);

    // Resolve #1 + 3
    let ref1 = ctx.add(Expr::SessionRef(1));
    let three = ctx.num(3);
    let input = ctx.add(Expr::Add(ref1, three));

    let result = resolve_session_refs_with_mode(
        &mut ctx,
        input,
        &store,
        RefMode::PreferSimplified,
        &cache_key,
    )
    .unwrap();

    // Should use cache
    assert!(result.used_cache, "Should have used cache");

    // Result should be 5 + 3 (order may vary), not the raw fraction
    if let Expr::Add(l, r) = ctx.get(result.expr) {
        // Extract both operands as numbers
        let left_num = match ctx.get(*l) {
            Expr::Number(n) => n.to_integer(),
            _ => panic!("Left should be Number"),
        };
        let right_num = match ctx.get(*r) {
            Expr::Number(n) => n.to_integer(),
            _ => panic!("Right should be Number"),
        };
        // Should contain 5 (from cache) and 3, order doesn't matter
        let has_five = left_num == 5.into() || right_num == 5.into();
        let has_three = left_num == 3.into() || right_num == 3.into();
        assert!(
            has_five,
            "Should contain 5 from cache, got {} and {}",
            left_num, right_num
        );
        assert!(
            has_three,
            "Should contain 3, got {} and {}",
            left_num, right_num
        );
    } else {
        panic!("Expected Add, got {:?}", ctx.get(result.expr));
    }
}

#[test]
fn test_resolve_with_mode_cache_miss_key_mismatch() {
    use cas_ast::{Context, Expr};

    let mut ctx = Context::new();
    let mut store = SessionStore::new();

    // Store raw expr as #1
    let x = ctx.var("x");
    store.push(EntryKind::Expr(x), "x".to_string());

    // Inject cache with different domain mode
    let simplified = ctx.num(5);
    let cache_key_strict = SimplifyCacheKey::from_context(cas_engine::DomainMode::Strict);
    let cache = SimplifiedCache {
        key: cache_key_strict,
        expr: simplified,
        requires: vec![],
        steps: Some(std::sync::Arc::new(vec![])),
    };
    store.update_simplified(1, cache);

    // Resolve with Generic mode (different from Strict cache)
    let ref1 = ctx.add(Expr::SessionRef(1));
    let cache_key_generic = SimplifyCacheKey::from_context(cas_engine::DomainMode::Generic);

    let result = resolve_session_refs_with_mode(
        &mut ctx,
        ref1,
        &store,
        RefMode::PreferSimplified,
        &cache_key_generic,
    )
    .unwrap();

    // Should NOT use cache (key mismatch)
    assert!(
        !result.used_cache,
        "Should NOT have used cache due to key mismatch"
    );

    // Result should be raw x, not 5
    assert!(matches!(ctx.get(result.expr), Expr::Variable(name) if ctx.sym_name(*name) == "x"));
}

#[test]
fn test_resolve_with_mode_raw_mode() {
    use cas_ast::{Context, Expr};

    let mut ctx = Context::new();
    let mut store = SessionStore::new();

    // Store raw expr as #1
    let x = ctx.var("x");
    store.push(EntryKind::Expr(x), "x".to_string());

    // Inject cache
    let simplified = ctx.num(5);
    let cache_key = SimplifyCacheKey::from_context(cas_engine::DomainMode::Generic);
    let cache = SimplifiedCache {
        key: cache_key.clone(),
        expr: simplified,
        requires: vec![],
        steps: Some(std::sync::Arc::new(vec![])),
    };
    store.update_simplified(1, cache);

    // Resolve with Raw mode - should ignore cache
    let ref1 = ctx.add(Expr::SessionRef(1));

    let result = resolve_session_refs_with_mode(
        &mut ctx,
        ref1,
        &store,
        RefMode::Raw, // Force raw mode
        &cache_key,
    )
    .unwrap();

    // Should NOT use cache (Raw mode)
    assert!(!result.used_cache, "Should NOT have used cache in Raw mode");

    // Result should be raw x
    assert!(matches!(ctx.get(result.expr), Expr::Variable(name) if ctx.sym_name(*name) == "x"));
}

#[test]
fn test_resolve_with_mode_tracks_ref_chain() {
    use cas_ast::{Context, Expr};

    let mut ctx = Context::new();
    let mut store = SessionStore::new();

    // #1 = x
    let x = ctx.var("x");
    store.push(EntryKind::Expr(x), "x".to_string());

    // #2 = #1 + 1
    let ref1 = ctx.add(Expr::SessionRef(1));
    let one = ctx.num(1);
    let expr2 = ctx.add(Expr::Add(ref1, one));
    store.push(EntryKind::Expr(expr2), "#1 + 1".to_string());

    // Resolve #2
    let ref2 = ctx.add(Expr::SessionRef(2));
    let cache_key = SimplifyCacheKey::from_context(cas_engine::DomainMode::Generic);

    let result = resolve_session_refs_with_mode(
        &mut ctx,
        ref2,
        &store,
        RefMode::PreferSimplified,
        &cache_key,
    )
    .unwrap();

    // Should track both #2 and #1 in ref chain
    assert!(result.ref_chain.contains(&1), "Should track #1 in chain");
    assert!(result.ref_chain.contains(&2), "Should track #2 in chain");
}
