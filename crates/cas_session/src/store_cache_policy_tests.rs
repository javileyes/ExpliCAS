#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use cas_ast::{Context, Expr};

    use crate::{CacheDomainMode, SimplifiedCache, SimplifyCacheKey, Step};

    fn sample_step() -> Step {
        let mut ctx = Context::new();
        let before = ctx.num(1);
        let after = ctx.num(2);
        Step::new_compact("test", "test", before, after)
    }

    fn sample_cache_with_steps(step_count: usize) -> SimplifiedCache {
        let mut ctx = Context::new();
        let one = ctx.num(1);
        let two = ctx.num(2);
        let expr = ctx.add(Expr::Add(one, two));
        let steps = (0..step_count).map(|_| sample_step()).collect::<Vec<_>>();
        SimplifiedCache {
            key: SimplifyCacheKey {
                domain: CacheDomainMode::Generic,
                ruleset_rev: 1,
            },
            expr,
            requires: vec![],
            steps: Some(Arc::new(steps)),
        }
    }

    #[test]
    fn simplify_cache_steps_len_counts_steps() {
        let cache = sample_cache_with_steps(3);
        assert_eq!(
            crate::store_cache_policy::simplify_cache_steps_len(&cache),
            3
        );
    }

    #[test]
    fn apply_simplified_light_cache_drops_steps_above_threshold() {
        let cache = sample_cache_with_steps(2);
        let reduced = crate::store_cache_policy::apply_simplified_light_cache(cache, Some(1));
        assert!(reduced.steps.is_none());
    }

    #[test]
    fn apply_simplified_light_cache_keeps_steps_when_under_threshold() {
        let cache = sample_cache_with_steps(1);
        let reduced = crate::store_cache_policy::apply_simplified_light_cache(cache, Some(2));
        assert_eq!(reduced.steps.as_ref().map(|s| s.len()), Some(1));
    }
}
