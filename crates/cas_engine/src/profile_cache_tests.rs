#[cfg(test)]
mod tests {
    use crate::options::{ContextMode, EvalOptions};
    use crate::profile_cache::ProfileCache;
    use std::sync::Arc;

    #[test]
    fn test_profile_reuse() {
        let mut cache = ProfileCache::new();
        let opts = EvalOptions::default();

        let p1 = cache.get_or_build(&opts);
        let p2 = cache.get_or_build(&opts);

        assert!(Arc::ptr_eq(&p1, &p2), "Profiles should be reused");
        assert_eq!(cache.len(), 1, "Should only have one cached profile");
    }

    #[test]
    fn test_different_modes_different_profiles() {
        let mut cache = ProfileCache::new();

        let standard_opts = EvalOptions::default();
        let integrate_opts = EvalOptions {
            shared: crate::phase::SharedSemanticConfig {
                context_mode: ContextMode::IntegratePrep,
                ..Default::default()
            },
            ..Default::default()
        };

        let p1 = cache.get_or_build(&standard_opts);
        let p2 = cache.get_or_build(&integrate_opts);

        assert!(
            !Arc::ptr_eq(&p1, &p2),
            "Different modes should have different profiles"
        );
        assert_eq!(
            cache.len(),
            2,
            "Should have two cached profiles for different modes"
        );
    }

    #[test]
    fn test_no_rebuild_multiple_calls() {
        let mut cache = ProfileCache::new();
        let opts = EvalOptions::default();

        for _ in 0..100 {
            let _ = cache.get_or_build(&opts);
        }

        assert_eq!(cache.len(), 1, "Should not rebuild on each call");
    }

    #[test]
    fn test_from_profile_end_to_end() {
        use crate::Simplifier;
        use cas_formatter::DisplayExpr;

        let mut cache = ProfileCache::new();
        let opts = EvalOptions::default();
        let profile = cache.get_or_build(&opts);

        let mut simplifier = Simplifier::from_profile(profile);

        let x = simplifier.context.var("x");
        let zero = simplifier.context.num(0);
        let expr = simplifier.context.add(cas_ast::Expr::Add(x, zero));

        let (result, steps) = simplifier.simplify(expr);
        let result_str = format!(
            "{}",
            DisplayExpr {
                context: &simplifier.context,
                id: result
            }
        );
        assert_eq!(result_str, "x", "Should simplify x + 0 to x");
        assert!(!steps.is_empty(), "Should have simplification steps");
    }
}
