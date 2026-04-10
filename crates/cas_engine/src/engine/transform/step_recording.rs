//! Step recording and path reconstruction for the rewrite trace.
//!
//! Handles recording rewrite steps (main + chained) and reconstructing
//! global expressions at the current path position.

use super::*;

impl<'a> LocalSimplificationTransformer<'a> {
    #[inline]
    fn emit_rule_applied_event(
        &mut self,
        rule_name: &str,
        before: ExprId,
        after: ExprId,
        global_before: Option<ExprId>,
        global_after: Option<ExprId>,
        is_chained: bool,
    ) {
        if let Some(listener) = self.event_listener.as_mut() {
            listener.on_event(&cas_solver_core::engine_events::EngineEvent::RuleApplied {
                rule_name: rule_name.to_string(),
                before,
                after,
                global_before,
                global_after,
                is_chained,
            });
        }
    }

    /// Record a step without inflating the recursive frame.
    /// Using #[inline(never)] to ensure Step construction stays out of transform_expr_recursive.
    #[inline(never)]
    pub(super) fn record_step(
        &mut self,
        name: &'static str,
        description: &'static str,
        before: ExprId,
        after: ExprId,
    ) {
        if self.collect_steps_enabled() {
            let global_before = self.root_expr;
            let global_after = self.reconstruct_at_path(after);

            let step = crate::step::Step::with_snapshots(
                name,
                description,
                before,
                after,
                self.current_path.clone(),
                Some(self.context),
                global_before,
                global_after,
            );
            self.steps.push(step);
            self.emit_rule_applied_event(
                description,
                before,
                after,
                Some(global_before),
                Some(global_after),
                false,
            );
        }
    }

    /// Reconstruct the global expression by substituting `replacement` at the given path
    pub(super) fn reconstruct_at_path(&mut self, replacement: ExprId) -> ExprId {
        if self.current_path.is_empty() {
            self.root_expr = replacement;
            return replacement;
        }

        let path = crate::pathsteps_to_expr_path(&self.current_path);
        let new_root = cas_math::expr_path_rewrite::rewrite_at_expr_path_raw(
            self.context,
            self.root_expr,
            &path,
            replacement,
        );
        self.root_expr = new_root; // Update root for next step
        new_root
    }

    /// Promote freshly recorded local preorder steps into coherent global snapshots.
    ///
    /// Some preorder helpers append steps directly to `self.steps` because they
    /// need to emit multi-step narratives before normal recursive descent.
    /// Those helpers only know the local subtree, so we reconstruct their
    /// `global_before/global_after` here using the transformer's current path
    /// and root expression.
    #[inline(never)]
    pub(super) fn rebuild_recent_preorder_steps_at_current_path(&mut self, start_index: usize) {
        if !self.collect_steps_enabled() || start_index >= self.steps.len() {
            return;
        }

        let path = crate::pathsteps_to_expr_path(&self.current_path);
        let mut current_global = self.root_expr;

        for step in self.steps.iter_mut().skip(start_index) {
            let global_before = current_global;
            let global_after = if path.is_empty() {
                step.after
            } else {
                cas_math::expr_path_rewrite::rewrite_at_expr_path_raw(
                    self.context,
                    current_global,
                    &path,
                    step.after,
                )
            };

            step.global_before = Some(global_before);
            step.global_after = Some(global_after);
            current_global = global_after;
        }

        self.root_expr = current_global;
    }

    /// Record a rewrite as one or more Steps (main + chained), handling `steps_mode` gating.
    ///
    /// Returns the `final_result` ExprId (last of chained, or main `new_expr`).
    /// When `steps_mode == Off`, skips all Step construction for performance.
    #[inline(never)]
    pub(super) fn record_rewrite_step(
        &mut self,
        rule: &dyn Rule,
        expr_id: ExprId,
        rewrite: crate::rule::Rewrite,
    ) -> ExprId {
        if self.collect_steps_enabled() {
            let crate::rule::Rewrite {
                new_expr: main_new_expr,
                description: main_description,
                before_local: main_before_local,
                after_local: main_after_local,
                assumption_events: main_assumptions,
                required_conditions: main_required,
                poly_proof: main_poly_proof,
                chained: chained_rewrites,
                substeps: main_substeps,
                budget_exempt: _,
            } = rewrite;

            // Determine final result (last of chained, or main rewrite)
            let final_result = chained_rewrites
                .last()
                .map(|c| c.after)
                .unwrap_or(main_new_expr);

            let global_before = self.root_expr;
            let main_global_after = self.reconstruct_at_path(main_new_expr);

            // Main step
            let mut step = Step::with_snapshots(
                main_description.as_ref(),
                rule.name(),
                expr_id,
                main_new_expr,
                self.current_path.clone(),
                Some(self.context),
                global_before,
                main_global_after,
            );
            step.importance = rule.importance();
            {
                let meta = step.meta_mut();
                meta.before_local = main_before_local;
                meta.after_local = main_after_local;
                meta.assumption_events = main_assumptions;
                meta.required_conditions = main_required;
                meta.poly_proof = main_poly_proof;
                meta.substeps = main_substeps;
            }
            self.steps.push(step);
            self.emit_rule_applied_event(
                rule.name(),
                expr_id,
                main_new_expr,
                Some(global_before),
                Some(main_global_after),
                false,
            );

            // Trace coherence verification
            debug_assert_eq!(
                main_global_after,
                self.root_expr,
                "[Trace Coherence] Step global_after doesn't match updated root_expr. \
                 Rule: {}, This will cause trace mismatch for next step.",
                rule.name()
            );

            // Process chained rewrites sequentially
            let mut current = main_new_expr;
            for chain_rw in chained_rewrites {
                let crate::rule::ChainedRewrite {
                    after: chain_after,
                    description: chain_description,
                    before_local: chain_before_local,
                    after_local: chain_after_local,
                    required_conditions: chain_required_conditions,
                    assumption_events: chain_assumption_events,
                    poly_proof: chain_poly_proof,
                    importance: chain_importance,
                } = chain_rw;
                let chain_global_before = self.root_expr;
                let chain_global_after = self.reconstruct_at_path(chain_after);

                let mut chain_step = Step::with_snapshots(
                    chain_description.as_ref(),
                    rule.name(),
                    current,
                    chain_after,
                    self.current_path.clone(),
                    Some(self.context),
                    chain_global_before,
                    chain_global_after,
                );
                chain_step.importance = chain_importance.unwrap_or_else(|| rule.importance());
                {
                    let meta = chain_step.meta_mut();
                    meta.before_local = chain_before_local;
                    meta.after_local = chain_after_local;
                    meta.assumption_events = chain_assumption_events;
                    meta.required_conditions = chain_required_conditions;
                    meta.poly_proof = chain_poly_proof;
                    meta.is_chained = true;
                }
                self.steps.push(chain_step);
                self.emit_rule_applied_event(
                    rule.name(),
                    current,
                    chain_after,
                    Some(chain_global_before),
                    Some(chain_global_after),
                    true,
                );

                current = chain_after;
            }

            final_result
        } else {
            // Without steps, keep emitting local rule events when a listener is attached.
            let mut current_before = expr_id;
            self.emit_rule_applied_event(
                rule.name(),
                current_before,
                rewrite.new_expr,
                None,
                None,
                false,
            );
            current_before = rewrite.new_expr;
            for chained in rewrite.chained {
                self.emit_rule_applied_event(
                    rule.name(),
                    current_before,
                    chained.after,
                    None,
                    None,
                    true,
                );
                current_before = chained.after;
            }

            // Without steps, just compute final result.
            current_before
        }
    }
}
