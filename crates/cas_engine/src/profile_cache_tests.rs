#[cfg(test)]
mod tests {
    use crate::options::{ContextMode, EvalOptions};
    use crate::profile_cache::ProfileCache;
    use std::sync::Arc;

    struct TestVariableToYRule;

    impl crate::rule::Rule for TestVariableToYRule {
        fn name(&self) -> &str {
            "Test Variable To Y"
        }

        fn apply(
            &self,
            ctx: &mut cas_ast::Context,
            expr: cas_ast::ExprId,
            _parent_ctx: &crate::parent_context::ParentContext,
        ) -> Option<crate::rule::Rewrite> {
            if !matches!(ctx.get(expr), cas_ast::Expr::Variable(_)) {
                return None;
            }

            Some(crate::rule::Rewrite::new(ctx.var("y")).desc("x -> y"))
        }

        fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
            Some(crate::target_kind::TargetKindSet::VARIABLE)
        }

        fn priority(&self) -> i32 {
            1_000
        }
    }

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

    #[test]
    fn test_from_profile_with_context_uses_existing_context() {
        use crate::Simplifier;

        let mut cache = ProfileCache::new();
        let opts = EvalOptions::default();
        let profile = cache.get_or_build(&opts);

        let mut ctx = cas_ast::Context::new();
        let x = ctx.var("x");

        let simplifier = Simplifier::from_profile_with_context(profile, ctx);

        assert!(matches!(
            simplifier.context.get(x),
            cas_ast::Expr::Variable(_)
        ));
    }

    #[test]
    fn test_phase_prefiltered_rules_match_allowed_phases() {
        let mut cache = ProfileCache::new();
        let opts = EvalOptions::default();
        let profile = cache.get_or_build(&opts);

        for &phase in crate::phase::SimplifyPhase::all() {
            let idx = crate::profile_cache::phase_index(phase);
            let mask = phase.mask();

            for bucket in profile.phase_rules[idx].values() {
                assert!(bucket
                    .iter()
                    .all(|rule| rule.allowed_phases().contains(mask)));
            }

            assert!(profile.phase_global_rules[idx]
                .iter()
                .all(|rule| rule.allowed_phases().contains(mask)));
        }
    }

    #[test]
    fn test_phase_prefiltered_rules_exclude_profile_disabled_rules() {
        let mut cache = ProfileCache::new();
        let opts = EvalOptions {
            shared: crate::phase::SharedSemanticConfig {
                context_mode: ContextMode::Solve,
                ..Default::default()
            },
            ..Default::default()
        };
        let profile = cache.get_or_build(&opts);

        for &phase in crate::phase::SimplifyPhase::all() {
            let idx = crate::profile_cache::phase_index(phase);

            for bucket in profile.phase_rules[idx].values() {
                assert!(bucket
                    .iter()
                    .all(|rule| !profile.disabled_rules.contains(rule.name())));
            }

            assert!(profile.phase_global_rules[idx]
                .iter()
                .all(|rule| !profile.disabled_rules.contains(rule.name())));
        }
    }

    #[test]
    fn test_from_profile_disable_rule_materializes_cached_rules() {
        use crate::Simplifier;

        let mut cache = ProfileCache::new();
        let opts = EvalOptions::default();
        let profile = cache.get_or_build(&opts);

        let mut simplifier = Simplifier::from_profile(profile);
        simplifier.disable_rule("Identity Property of Multiplication");

        let x = simplifier.context.var("x");
        let one = simplifier.context.num(1);
        let expr = simplifier.context.add(cas_ast::Expr::Mul(one, x));

        let (result, _steps) = simplifier.simplify(expr);
        assert!(matches!(
            simplifier.context.get(result),
            cas_ast::Expr::Mul(_, _)
        ));
    }

    #[test]
    fn test_from_profile_uses_profile_disabled_rules_without_clone() {
        use crate::Simplifier;

        let mut cache = ProfileCache::new();
        let opts = EvalOptions {
            shared: crate::phase::SharedSemanticConfig {
                context_mode: ContextMode::Solve,
                ..Default::default()
            },
            ..Default::default()
        };
        let profile = cache.get_or_build(&opts);

        let simplifier = Simplifier::from_profile(profile.clone());

        assert_eq!(
            simplifier.get_disabled_rules_clone(),
            profile.disabled_rules
        );
    }

    #[test]
    fn test_from_profile_add_rule_materializes_cached_rules() {
        use crate::Simplifier;
        use cas_formatter::DisplayExpr;

        let mut cache = ProfileCache::new();
        let opts = EvalOptions::default();
        let profile = cache.get_or_build(&opts);

        let mut simplifier = Simplifier::from_profile(profile);
        simplifier.add_rule(Box::new(TestVariableToYRule));

        let x = simplifier.context.var("x");
        let (result, _) = simplifier.simplify(x);
        let result_str = format!(
            "{}",
            DisplayExpr {
                context: &simplifier.context,
                id: result
            }
        );

        assert_eq!(result_str, "y");
    }
}
