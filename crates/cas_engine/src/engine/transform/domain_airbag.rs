//! Domain Delta Airbag: safety check for analytic domain expansion.
//!
//! Determines whether a rewrite would expand the analytic domain beyond
//! the original expression's implicit domain, and handles the result
//! according to the current DomainMode (Strict/Generic/Assume).

use super::*;

impl<'a> LocalSimplificationTransformer<'a> {
    /// Check the Domain Delta Airbag: whether a rewrite expands the analytic domain.
    ///
    /// Returns `true` if the rewrite should be **skipped** (blocked in Strict mode).
    /// In Generic mode, attaches `required_conditions` to the rewrite.
    /// In Assume mode, attaches `assumption_events` to the rewrite.
    ///
    /// `skip_unless_analytic`: if `true`, only check rules with `NeedsCondition(Analytic)`.
    /// Specific rules use `true` (gated), global rules use `false` (unconditional).
    #[inline(never)]
    pub(super) fn check_domain_airbag(
        &mut self,
        rule: &dyn Rule,
        parent_ctx: &crate::parent_context::ParentContext,
        expr_id: ExprId,
        rewrite: &mut crate::rule::Rewrite,
        skip_unless_analytic: bool,
    ) -> bool {
        // Determine which parent context to read domain info from
        let vd = parent_ctx.value_domain();
        let mode = parent_ctx.domain_mode();

        // Optionally gate on NeedsCondition(Analytic) solve_safety
        if skip_unless_analytic {
            let needs_analytic_check =
                cas_solver_core::solve_safety_policy::requires_introduced_analytic_condition(
                    rule.solve_safety(),
                );
            if !needs_analytic_check {
                return false; // Not blocked
            }
        }

        if parent_ctx.implicit_domain().is_none() {
            return false; // No domain to check
        }

        use crate::{check_analytic_expansion, AnalyticExpansionResult, ImplicitCondition};

        let expansion =
            check_analytic_expansion(self.context, self.root_expr, expr_id, rewrite.new_expr, vd);

        if let AnalyticExpansionResult::WouldExpand { dropped, sources } = expansion {
            let outcome = cas_solver_core::domain_airbag_policy::apply_domain_airbag_policy(
                mode,
                dropped,
                |cond| match cond {
                    ImplicitCondition::NonNegative(t) => {
                        Some(crate::AssumptionEvent::nonnegative(self.context, *t))
                    }
                    ImplicitCondition::Positive(t) => {
                        Some(crate::AssumptionEvent::positive(self.context, *t))
                    }
                    ImplicitCondition::NonZero(_) => None, // Skip definability in Assume mode
                },
            );

            if outcome.blocked {
                debug!(
                    "{}[DEBUG] Rule '{}' would expand analytic domain ({}), blocked in Strict mode",
                    self.indent(),
                    rule.name(),
                    sources.join(", ")
                );
                return true;
            }

            if !outcome.required.is_empty() {
                rewrite.required_conditions.extend(outcome.required);
                debug!(
                    "{}[DEBUG] Rule '{}' expands analytic domain, allowed in Generic mode with required conditions: {}",
                    self.indent(),
                    rule.name(),
                    sources.join(", ")
                );
            }

            if !outcome.assumptions.is_empty() {
                rewrite.assumption_events.extend(outcome.assumptions);
                debug!(
                    "{}[DEBUG] Rule '{}' expands analytic domain, allowed in Assume mode with assumptions: {}",
                    self.indent(),
                    rule.name(),
                    sources.join(", ")
                );
            }
        }

        false // Not blocked
    }
}
