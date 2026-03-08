use std::cell::RefCell;

thread_local! {
    /// Thread-local storage for blocked hints produced during one solve/simplify pass.
    static BLOCKED_HINTS: RefCell<Vec<crate::blocked_hint::BlockedHint>> =
        const { RefCell::new(Vec::new()) };
}

/// Register a blocked hint.
///
/// Hints are deduplicated by `(rule, key)` to avoid cascaded duplicates.
pub fn register_blocked_hint(hint: crate::blocked_hint::BlockedHint) {
    BLOCKED_HINTS.with(|hints| {
        let mut hints = hints.borrow_mut();
        let exists = hints
            .iter()
            .any(|existing| existing.rule == hint.rule && existing.key == hint.key);
        if !exists {
            hints.push(hint);
        }
    });
}

/// Take all blocked hints, clearing the collector.
pub fn take_blocked_hints() -> Vec<crate::blocked_hint::BlockedHint> {
    BLOCKED_HINTS.with(|hints| std::mem::take(&mut *hints.borrow_mut()))
}

/// Clear blocked hints without returning them.
pub fn clear_blocked_hints() {
    BLOCKED_HINTS.with(|hints| hints.borrow_mut().clear());
}

#[cfg(test)]
mod tests {
    use super::{clear_blocked_hints, register_blocked_hint, take_blocked_hints};
    use crate::assumption_model::AssumptionKey;
    use crate::blocked_hint::BlockedHint;

    fn hint(rule: &str, key: AssumptionKey) -> BlockedHint {
        let mut ctx = cas_ast::Context::default();
        BlockedHint {
            key,
            expr_id: ctx.num(1),
            rule: rule.to_string(),
            suggestion: "tip",
        }
    }

    #[test]
    fn register_blocked_hint_dedups_by_rule_and_key() {
        clear_blocked_hints();

        let key = AssumptionKey::NonZero {
            expr_fingerprint: 42,
        };
        register_blocked_hint(hint("SimplifyFraction", key.clone()));
        register_blocked_hint(hint("SimplifyFraction", key));

        let hints = take_blocked_hints();
        assert_eq!(hints.len(), 1);
        assert!(take_blocked_hints().is_empty());
    }

    #[test]
    fn register_blocked_hint_keeps_distinct_rule_or_key() {
        clear_blocked_hints();

        register_blocked_hint(hint(
            "RuleA",
            AssumptionKey::NonZero {
                expr_fingerprint: 10,
            },
        ));
        register_blocked_hint(hint(
            "RuleB",
            AssumptionKey::NonZero {
                expr_fingerprint: 10,
            },
        ));
        register_blocked_hint(hint(
            "RuleA",
            AssumptionKey::Positive {
                expr_fingerprint: 10,
            },
        ));

        let hints = take_blocked_hints();
        assert_eq!(hints.len(), 3);
        clear_blocked_hints();
    }
}
