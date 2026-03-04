//! Unified diagnostics container shared across engine/session layers.
//!
//! Consolidates all metadata about an evaluation/solve result:
//! - required domain conditions
//! - assumed conditions
//! - blocked rule hints

use crate::assumption_model::AssumptionEvent;
use crate::blocked_hint::BlockedHint;
use crate::domain_condition::{ImplicitCondition, RequiresDisplayLevel};
use cas_ast::{Context, ExprId};
use smallvec::SmallVec;
use std::collections::HashMap;

/// Origin of a required condition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RequireOrigin {
    /// Implicit in equation structure.
    EquationImplicit,
    /// Derived via equation equality.
    EquationDerived,
    /// Implicit in input expression.
    InputImplicit,
    /// Implicit in output/result expression.
    OutputImplicit,
    /// Added by rewrite airbag checks.
    RewriteAirbag,
    /// Propagated from session cache/reference.
    SessionPropagated,
}

impl RequireOrigin {
    /// Human-readable description for explain/debug views.
    pub fn description(&self) -> &'static str {
        match self {
            RequireOrigin::EquationImplicit => "equation implicit",
            RequireOrigin::EquationDerived => "equation derived",
            RequireOrigin::InputImplicit => "input implicit",
            RequireOrigin::OutputImplicit => "output implicit",
            RequireOrigin::RewriteAirbag => "rewrite airbag",
            RequireOrigin::SessionPropagated => "session propagated",
        }
    }
}

/// One required condition plus all known origins.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequiredItem {
    pub cond: ImplicitCondition,
    pub origins: SmallVec<[RequireOrigin; 2]>,
}

impl RequiredItem {
    pub fn new(cond: ImplicitCondition, origin: RequireOrigin) -> Self {
        let mut origins = SmallVec::new();
        origins.push(origin);
        Self { cond, origins }
    }

    /// Merge a new origin if it is not present yet.
    pub fn merge_origin(&mut self, origin: RequireOrigin) {
        if !self.origins.contains(&origin) {
            self.origins.push(origin);
        }
    }

    /// Display condition plus origin metadata.
    pub fn display_with_origin(&self, ctx: &Context) -> String {
        let cond_str = self.cond.display(ctx);
        if self.origins.len() == 1 {
            format!("{} (from: {})", cond_str, self.origins[0].description())
        } else {
            let origins_str: Vec<_> = self.origins.iter().map(|o| o.description()).collect();
            format!("{} (from: {})", cond_str, origins_str.join(", "))
        }
    }
}

/// Unified diagnostics payload.
#[derive(Debug, Default, Clone)]
pub struct Diagnostics {
    /// Required conditions for correctness.
    pub requires: Vec<RequiredItem>,
    /// Assumptions accepted by policy.
    pub assumed: Vec<AssumptionEvent>,
    /// Blocked rule hints.
    pub blocked: Vec<BlockedHint>,
}

impl Diagnostics {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add one required condition with origin deduplication.
    pub fn push_required(&mut self, cond: ImplicitCondition, origin: RequireOrigin) {
        for item in &mut self.requires {
            if item.cond == cond {
                item.merge_origin(origin);
                return;
            }
        }
        self.requires.push(RequiredItem::new(cond, origin));
    }

    /// Add multiple required conditions under the same origin.
    pub fn extend_required<I>(&mut self, conds: I, origin: RequireOrigin)
    where
        I: IntoIterator<Item = ImplicitCondition>,
    {
        for cond in conds {
            self.push_required(cond, origin);
        }
    }

    /// Add one assumption event.
    pub fn push_assumed(&mut self, event: AssumptionEvent) {
        self.assumed.push(event);
    }

    /// Add one blocked hint.
    pub fn push_blocked(&mut self, hint: BlockedHint) {
        self.blocked.push(hint);
    }

    /// Remove trivial/duplicate entries and stabilize ordering.
    pub fn dedup_and_sort(&mut self, ctx: &Context) {
        self.requires.retain(|item| !item.cond.is_trivial(ctx));

        let mut seen: HashMap<String, usize> = HashMap::new();
        let mut merged: Vec<RequiredItem> = Vec::new();
        for item in std::mem::take(&mut self.requires) {
            let key = item.cond.display(ctx);
            if let Some(&idx) = seen.get(&key) {
                for origin in item.origins {
                    merged[idx].merge_origin(origin);
                }
            } else {
                seen.insert(key, merged.len());
                merged.push(item);
            }
        }
        merged.sort_by(|a, b| a.cond.display(ctx).cmp(&b.cond.display(ctx)));
        self.requires = merged;

        let mut seen_assumed: std::collections::HashSet<String> = std::collections::HashSet::new();
        self.assumed
            .retain(|event| seen_assumed.insert(event.message.clone()));

        let mut seen_blocked: std::collections::HashSet<String> = std::collections::HashSet::new();
        self.blocked
            .retain(|hint| seen_blocked.insert(hint.rule.to_string()));
    }

    /// True when there is no diagnostic payload to show.
    pub fn is_empty(&self) -> bool {
        self.requires.is_empty() && self.assumed.is_empty() && self.blocked.is_empty()
    }

    /// Backward-compatible list view containing only conditions.
    pub fn required_conditions(&self) -> Vec<ImplicitCondition> {
        self.requires.iter().map(|r| r.cond.clone()).collect()
    }

    /// Filter required conditions for display.
    ///
    /// `Essential` keeps:
    /// - strong-origin conditions (`EquationDerived`, `RewriteAirbag`)
    /// - conditions whose witness is no longer visible in the result
    /// - all `Positive(_)` when there are >=2 positives (dominance UX rule)
    ///
    /// `All` keeps all conditions.
    pub fn filter_requires_for_display<'a>(
        &'a self,
        ctx: &Context,
        result: ExprId,
        level: RequiresDisplayLevel,
    ) -> Vec<&'a RequiredItem> {
        if level == RequiresDisplayLevel::All {
            return self.requires.iter().collect();
        }

        let positive_count = self
            .requires
            .iter()
            .filter(|item| matches!(item.cond, ImplicitCondition::Positive(_)))
            .count();
        let has_multiple_log_witnesses = positive_count >= 2;

        self.requires
            .iter()
            .filter(|item| {
                let has_strong_origin = item.origins.iter().any(|origin| {
                    matches!(
                        origin,
                        RequireOrigin::EquationDerived | RequireOrigin::RewriteAirbag
                    )
                });

                if has_strong_origin {
                    return true;
                }

                if has_multiple_log_witnesses && matches!(item.cond, ImplicitCondition::Positive(_))
                {
                    return true;
                }

                !item.cond.witness_survives_in(ctx, result)
            })
            .collect()
    }

    /// Inherit required conditions from another diagnostics payload.
    pub fn inherit_requires_from(&mut self, other: &Diagnostics) {
        for item in &other.requires {
            for &origin in &item.origins {
                self.push_required(item.cond.clone(), origin);
            }
            self.push_required(item.cond.clone(), RequireOrigin::SessionPropagated);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Diagnostics, RequireOrigin};
    use crate::domain_condition::ImplicitCondition;
    use cas_ast::Context;

    #[test]
    fn merges_origins_for_same_condition() {
        let mut ctx = Context::new();
        let x = ctx.var("x");

        let mut diag = Diagnostics::new();
        diag.push_required(
            ImplicitCondition::Positive(x),
            RequireOrigin::EquationImplicit,
        );
        diag.push_required(
            ImplicitCondition::Positive(x),
            RequireOrigin::EquationDerived,
        );

        assert_eq!(diag.requires.len(), 1);
        assert_eq!(diag.requires[0].origins.len(), 2);
    }

    #[test]
    fn filters_trivial_requirements() {
        let mut ctx = Context::new();
        let two = ctx.num(2);
        let x = ctx.var("x");

        let mut diag = Diagnostics::new();
        diag.push_required(
            ImplicitCondition::Positive(two),
            RequireOrigin::EquationImplicit,
        );
        diag.push_required(
            ImplicitCondition::Positive(x),
            RequireOrigin::OutputImplicit,
        );

        diag.dedup_and_sort(&ctx);
        assert_eq!(diag.requires.len(), 1);
        assert!(matches!(diag.requires[0].cond, ImplicitCondition::Positive(id) if id == x));
    }
}
