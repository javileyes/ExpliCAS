//! Generic eval-step cleanup pipeline shared across crates.

use cas_ast::ExprId;

/// Clean raw eval steps for display:
/// 1) Remove no-op steps (no global or focused local change),
/// 2) Repair global-before/global-after chain coherence.
#[allow(clippy::too_many_arguments)]
pub fn clean_eval_steps<
    T,
    FBefore,
    FAfter,
    FBeforeLocal,
    FAfterLocal,
    FGlobalAfter,
    FSetGlobalBefore,
>(
    raw_steps: Vec<T>,
    mut before_of: FBefore,
    mut after_of: FAfter,
    mut before_local_of: FBeforeLocal,
    mut after_local_of: FAfterLocal,
    mut global_after_of: FGlobalAfter,
    mut set_global_before: FSetGlobalBefore,
) -> Vec<T>
where
    FBefore: FnMut(&T) -> ExprId,
    FAfter: FnMut(&T) -> ExprId,
    FBeforeLocal: FnMut(&T) -> Option<ExprId>,
    FAfterLocal: FnMut(&T) -> Option<ExprId>,
    FGlobalAfter: FnMut(&T) -> Option<ExprId>,
    FSetGlobalBefore: FnMut(&mut T, ExprId),
{
    let mut cleaned: Vec<T> = raw_steps
        .into_iter()
        .filter(|step| {
            let global_changed = before_of(step) != after_of(step);
            let local_changed = match (before_local_of(step), after_local_of(step)) {
                (Some(bl), Some(al)) => bl != al,
                _ => false,
            };
            global_changed || local_changed
        })
        .collect();

    for i in 0..cleaned.len().saturating_sub(1) {
        if let Some(after_i) = global_after_of(&cleaned[i]) {
            set_global_before(&mut cleaned[i + 1], after_i);
        }
    }

    cleaned
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::Context;

    #[derive(Clone, Debug)]
    struct TestStep {
        before: ExprId,
        after: ExprId,
        before_local: Option<ExprId>,
        after_local: Option<ExprId>,
        global_before: Option<ExprId>,
        global_after: Option<ExprId>,
    }

    #[test]
    fn removes_global_noops_without_local_change() {
        let mut ctx = Context::new();
        let a = ctx.num(1);
        let b = ctx.num(2);

        let raw = vec![
            TestStep {
                before: a,
                after: a,
                before_local: None,
                after_local: None,
                global_before: None,
                global_after: Some(a),
            },
            TestStep {
                before: a,
                after: b,
                before_local: None,
                after_local: None,
                global_before: None,
                global_after: Some(b),
            },
        ];

        let cleaned = clean_eval_steps(
            raw,
            |s| s.before,
            |s| s.after,
            |s| s.before_local,
            |s| s.after_local,
            |s| s.global_after,
            |s, gb| s.global_before = Some(gb),
        );

        assert_eq!(cleaned.len(), 1);
        assert_eq!(cleaned[0].before, a);
        assert_eq!(cleaned[0].after, b);
    }

    #[test]
    fn preserves_local_changes_when_global_is_equal() {
        let mut ctx = Context::new();
        let a = ctx.num(1);
        let b = ctx.num(2);

        let raw = vec![TestStep {
            before: a,
            after: a,
            before_local: Some(a),
            after_local: Some(b),
            global_before: None,
            global_after: Some(a),
        }];

        let cleaned = clean_eval_steps(
            raw,
            |s| s.before,
            |s| s.after,
            |s| s.before_local,
            |s| s.after_local,
            |s| s.global_after,
            |s, gb| s.global_before = Some(gb),
        );

        assert_eq!(cleaned.len(), 1);
    }

    #[test]
    fn repairs_global_before_chain_from_previous_after() {
        let mut ctx = Context::new();
        let a = ctx.num(1);
        let b = ctx.num(2);
        let c = ctx.num(3);

        let raw = vec![
            TestStep {
                before: a,
                after: b,
                before_local: None,
                after_local: None,
                global_before: None,
                global_after: Some(c),
            },
            TestStep {
                before: b,
                after: c,
                before_local: None,
                after_local: None,
                global_before: None,
                global_after: Some(b),
            },
        ];

        let cleaned = clean_eval_steps(
            raw,
            |s| s.before,
            |s| s.after,
            |s| s.before_local,
            |s| s.after_local,
            |s| s.global_after,
            |s, gb| s.global_before = Some(gb),
        );

        assert_eq!(cleaned.len(), 2);
        assert_eq!(cleaned[1].global_before, Some(c));
    }
}
