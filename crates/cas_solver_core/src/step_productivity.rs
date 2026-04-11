//! Generic step-productivity filtering helpers.
//!
//! Runtime crates can reuse this to remove no-op or semantically redundant
//! steps while preserving didactically important ones.

use cas_ast::{Context, ExprId};
use cas_math::expr_semantic_hash::semantic_hash;
use cas_math::semantic_equality::SemanticEqualityChecker;

/// Filter out non-productive steps using caller-provided adapters.
///
/// A step is kept when:
/// - caller marks it as always-keep, or
/// - it changes local expression semantically and materially changes the
///   reconstructed global expression.
#[allow(clippy::too_many_arguments)]
pub fn filter_non_productive_steps_with<
    T,
    FAlwaysKeep,
    FBefore,
    FAfter,
    FPath,
    FDisplayNoop,
    FRewrite,
>(
    ctx: &mut Context,
    original: ExprId,
    steps: Vec<T>,
    mut always_keep: FAlwaysKeep,
    mut before_of: FBefore,
    mut after_of: FAfter,
    mut path_of: FPath,
    mut is_display_noop: FDisplayNoop,
    mut rewrite_global: FRewrite,
) -> Vec<T>
where
    FAlwaysKeep: FnMut(&T) -> bool,
    FBefore: FnMut(&T) -> ExprId,
    FAfter: FnMut(&T) -> ExprId,
    FPath: FnMut(&T) -> Vec<u8>,
    FDisplayNoop: FnMut(&Context, ExprId, ExprId) -> bool,
    FRewrite: FnMut(&mut Context, ExprId, &[u8], ExprId) -> ExprId,
{
    let mut filtered = Vec::new();
    let mut current_global = original;
    let mut seen_states = vec![original];
    let mut seen_hashes = vec![semantic_hash(ctx, original)];

    for step in steps {
        let before = before_of(&step);
        let after = after_of(&step);
        let path = path_of(&step);

        if always_keep(&step) {
            current_global = rewrite_global(ctx, current_global, &path, after);
            filtered.push(step);
            continue;
        }

        let checker = SemanticEqualityChecker::new(ctx);
        if checker.are_equal(before, after) {
            continue;
        }

        if is_display_noop(ctx, before, after) {
            continue;
        }

        let global_after = rewrite_global(ctx, current_global, &path, after);
        let global_after_hash = semantic_hash(ctx, global_after);
        let checker = SemanticEqualityChecker::new(ctx);
        if checker.are_equal(current_global, global_after) {
            continue;
        }

        let repeated_state_idx = seen_hashes
            .iter()
            .enumerate()
            .rev()
            .find_map(|(idx, hash)| {
                (*hash == global_after_hash && checker.are_equal(seen_states[idx], global_after))
                    .then_some(idx)
            });

        if let Some(idx) = repeated_state_idx {
            filtered.truncate(idx);
            seen_states.truncate(idx + 1);
            seen_hashes.truncate(idx + 1);
            current_global = *seen_states
                .last()
                .expect("original state must remain when trimming a cycle");
            continue;
        }

        filtered.push(step);
        current_global = global_after;
        seen_states.push(global_after);
        seen_hashes.push(global_after_hash);
    }

    filtered
}

#[cfg(test)]
mod tests {
    use super::filter_non_productive_steps_with;
    use cas_ast::{Context, Expr, ExprId};

    #[derive(Clone)]
    struct DummyStep {
        before: ExprId,
        after: ExprId,
        path: Vec<u8>,
        keep: bool,
    }

    #[test]
    fn removes_local_semantic_noops() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let sum = ctx.add(Expr::Add(x, one));

        let steps = vec![DummyStep {
            before: sum,
            after: sum,
            path: vec![],
            keep: false,
        }];

        let out = filter_non_productive_steps_with(
            &mut ctx,
            sum,
            steps,
            |s| s.keep,
            |s| s.before,
            |s| s.after,
            |s| s.path.clone(),
            |_ctx, _before, _after| false,
            |ctx, root, path, replacement| {
                cas_math::expr_path_rewrite::rewrite_at_expr_path(ctx, root, path, replacement)
            },
        );
        assert!(out.is_empty());
    }

    #[test]
    fn keeps_steps_marked_as_always_keep() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let sum = ctx.add(Expr::Add(x, one));

        let steps = vec![DummyStep {
            before: sum,
            after: sum,
            path: vec![],
            keep: true,
        }];

        let out = filter_non_productive_steps_with(
            &mut ctx,
            sum,
            steps,
            |s| s.keep,
            |s| s.before,
            |s| s.after,
            |s| s.path.clone(),
            |_ctx, _before, _after| false,
            |ctx, root, path, replacement| {
                cas_math::expr_path_rewrite::rewrite_at_expr_path(ctx, root, path, replacement)
            },
        );
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn removes_repeated_state_cycles_entirely() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let z = ctx.var("z");

        let steps = vec![
            DummyStep {
                before: x,
                after: y,
                path: vec![],
                keep: false,
            },
            DummyStep {
                before: y,
                after: x,
                path: vec![],
                keep: false,
            },
            DummyStep {
                before: x,
                after: z,
                path: vec![],
                keep: false,
            },
        ];

        let out = filter_non_productive_steps_with(
            &mut ctx,
            x,
            steps,
            |s| s.keep,
            |s| s.before,
            |s| s.after,
            |s| s.path.clone(),
            |_ctx, _before, _after| false,
            |ctx, root, path, replacement| {
                cas_math::expr_path_rewrite::rewrite_at_expr_path(ctx, root, path, replacement)
            },
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].before, x);
        assert_eq!(out[0].after, z);
    }
}
