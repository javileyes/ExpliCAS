/// Result of a domain-aware cancellation decision.
///
/// Used by cancellation-style rules to determine whether a transform is
/// allowed and whether assumptions should be recorded.
#[derive(Debug, Clone)]
pub struct CancelDecision {
    /// Whether the transformation is allowed.
    pub allow: bool,
    /// Optional domain assumption message for steps/warnings.
    pub assumption: Option<&'static str>,
    /// Hint when blocked due to domain policy.
    pub blocked_hint: Option<crate::blocked_hint::BlockedHint>,
    /// Assumption keys made when allowing with unknown proof.
    pub assumed_keys: smallvec::SmallVec<[crate::assumption_model::AssumptionKey; 2]>,
}

impl CancelDecision {
    /// Create a decision that allows with no assumption (proven).
    pub fn allow() -> Self {
        Self {
            allow: true,
            assumption: None,
            blocked_hint: None,
            assumed_keys: smallvec::SmallVec::new(),
        }
    }

    /// Create a decision that blocks with no pedagogical hint.
    pub fn deny() -> Self {
        Self {
            allow: false,
            assumption: None,
            blocked_hint: None,
            assumed_keys: smallvec::SmallVec::new(),
        }
    }

    /// Create a decision that blocks with a pedagogical hint.
    pub fn deny_with_hint(
        key: crate::assumption_model::AssumptionKey,
        expr_id: cas_ast::ExprId,
        rule: &'static str,
    ) -> Self {
        Self {
            allow: false,
            assumption: None,
            blocked_hint: Some(crate::blocked_hint::BlockedHint {
                key,
                expr_id,
                rule: rule.to_string(),
                suggestion: "use `semantics set domain assume` to allow analytic assumptions",
            }),
            assumed_keys: smallvec::SmallVec::new(),
        }
    }

    /// Create a decision that allows with an assumption warning.
    pub fn allow_with_assumption(msg: &'static str) -> Self {
        Self {
            allow: true,
            assumption: Some(msg),
            blocked_hint: None,
            assumed_keys: smallvec::SmallVec::new(),
        }
    }

    /// Create a decision that allows with tracked assumption keys.
    pub fn allow_with_keys(
        msg: &'static str,
        keys: smallvec::SmallVec<[crate::assumption_model::AssumptionKey; 2]>,
    ) -> Self {
        Self {
            allow: true,
            assumption: Some(msg),
            blocked_hint: None,
            assumed_keys: keys,
        }
    }

    /// Convert assumed keys to assumption events for rewrite propagation.
    pub fn assumption_events(
        &self,
        ctx: &cas_ast::Context,
        expr_id: cas_ast::ExprId,
    ) -> smallvec::SmallVec<[crate::assumption_model::AssumptionEvent; 1]> {
        self.assumed_keys
            .iter()
            .map(|key| {
                let expr_display = cas_formatter::DisplayExpr {
                    context: ctx,
                    id: expr_id,
                }
                .to_string();
                let message = match key {
                    crate::assumption_model::AssumptionKey::NonZero { .. } => {
                        format!("Assumed {} ≠ 0", expr_display)
                    }
                    crate::assumption_model::AssumptionKey::Positive { .. } => {
                        format!("Assumed {} > 0", expr_display)
                    }
                    crate::assumption_model::AssumptionKey::NonNegative { .. } => {
                        format!("Assumed {} ≥ 0", expr_display)
                    }
                    crate::assumption_model::AssumptionKey::Defined { .. } => {
                        format!("Assumed {} is defined", expr_display)
                    }
                    crate::assumption_model::AssumptionKey::InvTrigPrincipalRange {
                        func, ..
                    } => {
                        format!("Assumed {} in {} principal range", expr_display, func)
                    }
                    crate::assumption_model::AssumptionKey::ComplexPrincipalBranch {
                        func, ..
                    } => {
                        format!("Assumed {}({}) principal branch", func, expr_display)
                    }
                };
                crate::assumption_model::AssumptionEvent {
                    key: key.clone(),
                    expr_display,
                    message,
                    kind: crate::assumption_model::AssumptionKind::DerivedFromRequires,
                    expr_id: Some(expr_id),
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::CancelDecision;
    use crate::assumption_model::AssumptionKey;

    #[test]
    fn allow_and_deny_builders_set_expected_flags() {
        let allow = CancelDecision::allow();
        assert!(allow.allow);
        assert!(allow.assumption.is_none());
        assert!(allow.blocked_hint.is_none());
        assert!(allow.assumed_keys.is_empty());

        let deny = CancelDecision::deny();
        assert!(!deny.allow);
        assert!(deny.assumption.is_none());
        assert!(deny.blocked_hint.is_none());
        assert!(deny.assumed_keys.is_empty());
    }

    #[test]
    fn allow_with_keys_emits_assumption_events() {
        let mut ctx = cas_ast::Context::default();
        let x = ctx.var("x");
        let key = AssumptionKey::nonzero_key(&ctx, x);
        let decision = CancelDecision::allow_with_keys(
            "cancelled factor assumed nonzero",
            smallvec::smallvec![key],
        );

        let events = decision.assumption_events(&ctx, x);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].expr_id, Some(x));
        assert!(events[0].message.contains("Assumed"));
    }
}
