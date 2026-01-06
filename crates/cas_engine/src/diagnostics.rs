//! Unified diagnostics container for engine output.
//!
//! This module consolidates all "meta" information about an evaluation result:
//! - Requires: domain conditions necessary for validity
//! - Assumed: conditions adopted by policy (not proven)
//! - Blocked: hints about rules that couldn't fire
//!
//! # Origin Tracking
//!
//! Each required condition tracks its origin(s), enabling:
//! - Clear explanation of where conditions come from
//! - Deduplication when the same condition is derived multiple ways
//! - Pedagogical transparency in REPL and timeline

use crate::assumptions::AssumptionEvent;
use crate::domain::BlockedHint;
use crate::implicit_domain::ImplicitCondition;
use cas_ast::Context;
use smallvec::SmallVec;
use std::collections::HashMap;

/// Origin of a required condition - explains where it came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RequireOrigin {
    /// Implicit in the equation structure (e.g., sqrt(y) in equation → y ≥ 0)
    EquationImplicit,
    /// Derived via equation equality (e.g., 2^x = sqrt(y) → 2^x > 0 → sqrt(y) > 0 → y > 0)
    EquationDerived,
    /// Implicit in the output/result (e.g., ln(y) appears in solution)
    OutputImplicit,
    /// Detected by rewrite airbag (witness survival check)
    RewriteAirbag,
    /// From session entry reuse (propagated via #id)
    SessionPropagated,
}

impl RequireOrigin {
    /// Human-readable description for explain mode
    pub fn description(&self) -> &'static str {
        match self {
            RequireOrigin::EquationImplicit => "equation implicit",
            RequireOrigin::EquationDerived => "equation derived",
            RequireOrigin::OutputImplicit => "output implicit",
            RequireOrigin::RewriteAirbag => "rewrite airbag",
            RequireOrigin::SessionPropagated => "session propagated",
        }
    }
}

/// A required condition with its origin(s).
///
/// Multiple origins are possible when the same condition is derived
/// through different paths (e.g., both equation implicit AND derived).
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

    /// Merge another origin into this item (for dedup)
    pub fn merge_origin(&mut self, origin: RequireOrigin) {
        if !self.origins.contains(&origin) {
            self.origins.push(origin);
        }
    }

    /// Display with origin info (for explain mode)
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

/// Unified diagnostics container for evaluation output.
///
/// Consolidates all metadata about conditions, assumptions, and blocked rules.
#[derive(Debug, Default, Clone)]
pub struct Diagnostics {
    /// Required conditions for validity (domain constraints)
    pub requires: Vec<RequiredItem>,
    /// Assumptions made by policy (not proven, just accepted)
    pub assumed: Vec<AssumptionEvent>,
    /// Blocked rules with pedagogical hints
    pub blocked: Vec<BlockedHint>,
}

impl Diagnostics {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a required condition with its origin.
    /// If the condition already exists, merges the origin.
    pub fn push_required(&mut self, cond: ImplicitCondition, origin: RequireOrigin) {
        // Check if condition already exists (by variant and expr)
        for item in &mut self.requires {
            if item.cond == cond {
                item.merge_origin(origin);
                return;
            }
        }
        // New condition
        self.requires.push(RequiredItem::new(cond, origin));
    }

    /// Add multiple required conditions with the same origin
    pub fn extend_required<I>(&mut self, conds: I, origin: RequireOrigin)
    where
        I: IntoIterator<Item = ImplicitCondition>,
    {
        for cond in conds {
            self.push_required(cond, origin);
        }
    }

    /// Add an assumption event
    pub fn push_assumed(&mut self, event: AssumptionEvent) {
        self.assumed.push(event);
    }

    /// Add a blocked hint
    pub fn push_blocked(&mut self, hint: BlockedHint) {
        self.blocked.push(hint);
    }

    /// Deduplicate and sort for stable output.
    /// Also filters out trivial conditions (constants like "2 > 0").
    pub fn dedup_and_sort(&mut self, ctx: &Context) {
        // Filter trivial requires
        self.requires.retain(|item| !item.cond.is_trivial(ctx));

        // Deduplicate by display string (handles structural equivalents)
        let mut seen: HashMap<String, usize> = HashMap::new();
        let mut merged: Vec<RequiredItem> = Vec::new();

        for item in std::mem::take(&mut self.requires) {
            let key = item.cond.display(ctx);
            if let Some(&idx) = seen.get(&key) {
                // Merge origins
                for origin in item.origins {
                    merged[idx].merge_origin(origin);
                }
            } else {
                seen.insert(key.clone(), merged.len());
                merged.push(item);
            }
        }

        // Sort by display string for deterministic output
        merged.sort_by(|a, b| a.cond.display(ctx).cmp(&b.cond.display(ctx)));
        self.requires = merged;

        // Deduplicate assumed (by message)
        let mut seen_assumed: std::collections::HashSet<String> = std::collections::HashSet::new();
        self.assumed
            .retain(|e| seen_assumed.insert(e.message.clone()));

        // Deduplicate blocked (by rule name)
        let mut seen_blocked: std::collections::HashSet<String> = std::collections::HashSet::new();
        self.blocked
            .retain(|h| seen_blocked.insert(h.rule.to_string()));
    }

    /// Check if there are any diagnostics to report
    pub fn is_empty(&self) -> bool {
        self.requires.is_empty() && self.assumed.is_empty() && self.blocked.is_empty()
    }

    /// Get just the conditions (for backward compatibility)
    pub fn required_conditions(&self) -> Vec<ImplicitCondition> {
        self.requires.iter().map(|r| r.cond.clone()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::Context;

    #[test]
    fn test_merge_origins() {
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

        assert_eq!(diag.requires.len(), 1, "Should merge same condition");
        assert_eq!(
            diag.requires[0].origins.len(),
            2,
            "Should have both origins"
        );
    }

    #[test]
    fn test_trivial_filtered() {
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

        assert_eq!(diag.requires.len(), 1, "Trivial 2 > 0 should be filtered");
        assert!(
            matches!(&diag.requires[0].cond, ImplicitCondition::Positive(e) if *e == x),
            "Only x > 0 should remain"
        );
    }
}
