#[cfg(test)]
mod tests {
    use cas_ast::{Context, Expr};

    use crate::resolve_refs::{resolve_session_refs, resolve_session_refs_with_env};
    use crate::{env::Environment, CacheConfig, EntryKind, SessionStore};

    fn store_with_expr_entry(expr: cas_ast::ExprId, raw: &str) -> (SessionStore, u64) {
        let mut store =
            crate::store_cache_policy::session_store_with_cache_config(CacheConfig::default());
        let id = store.push(EntryKind::Expr(expr), raw.to_string());
        (store, id)
    }

    #[test]
    fn resolve_session_refs_replaces_direct_session_ref() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let (store, id) = store_with_expr_entry(x, "x");
        let expr = ctx.add(Expr::SessionRef(id));

        let resolved = resolve_session_refs(&mut ctx, expr, &store)
            .expect("session ref should resolve against store");
        assert_eq!(resolved, x);
    }

    #[test]
    fn resolve_session_refs_with_env_substitutes_binding_after_resolution() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let (store, id) = store_with_expr_entry(x, "x");
        let expr = ctx.add(Expr::SessionRef(id));
        let mut env = Environment::new();
        env.set("x".to_string(), two);

        let resolved = resolve_session_refs_with_env(&mut ctx, expr, &store, &env)
            .expect("session ref + env should resolve");
        assert_eq!(resolved, two);
    }
}
