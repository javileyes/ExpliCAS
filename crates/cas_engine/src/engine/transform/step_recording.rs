//! Step recording and path reconstruction for the rewrite trace.
//!
//! Handles recording rewrite steps (main + chained) and reconstructing
//! global expressions at the current path position.

use super::*;

impl<'a> LocalSimplificationTransformer<'a> {
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
        if self.steps_mode != StepsMode::Off {
            let step = crate::step::Step::new(
                name,
                description,
                before,
                after,
                self.current_path.clone(),
                Some(self.context),
            );
            self.steps.push(step);
        }
    }

    /// Reconstruct the global expression by substituting `replacement` at the given path
    pub(super) fn reconstruct_at_path(&mut self, replacement: ExprId) -> ExprId {
        use crate::step::PathStep;

        fn reconstruct_recursive(
            context: &mut Context,
            root: ExprId,
            path: &[PathStep],
            replacement: ExprId,
        ) -> ExprId {
            if path.is_empty() {
                return replacement;
            }

            let current_step = &path[0];
            let remaining_path = &path[1..];
            let expr = context.get(root).clone();

            match (expr, current_step) {
                (Expr::Add(l, r), PathStep::Left) => {
                    let new_l = reconstruct_recursive(context, l, remaining_path, replacement);
                    context.add_raw(Expr::Add(new_l, r)) // Use add_raw to preserve structure
                }
                (Expr::Add(l, r), PathStep::Right) => {
                    // Follow AST literally - don't do magic Neg unwrapping.
                    // If we need to modify inside a Neg, the path should include PathStep::Inner.
                    let new_r = reconstruct_recursive(context, r, remaining_path, replacement);
                    context.add_raw(Expr::Add(l, new_r)) // Use add_raw to preserve structure
                }
                (Expr::Sub(l, r), PathStep::Left) => {
                    let new_l = reconstruct_recursive(context, l, remaining_path, replacement);
                    context.add_raw(Expr::Sub(new_l, r)) // Use add_raw to preserve structure
                }
                (Expr::Sub(l, r), PathStep::Right) => {
                    let new_r = reconstruct_recursive(context, r, remaining_path, replacement);
                    context.add_raw(Expr::Sub(l, new_r)) // Use add_raw to preserve structure
                }
                (Expr::Mul(l, r), PathStep::Left) => {
                    let new_l = reconstruct_recursive(context, l, remaining_path, replacement);
                    context.add_raw(Expr::Mul(new_l, r)) // Use add_raw to preserve structure
                }
                (Expr::Mul(l, r), PathStep::Right) => {
                    let new_r = reconstruct_recursive(context, r, remaining_path, replacement);
                    context.add_raw(Expr::Mul(l, new_r)) // Use add_raw to preserve structure
                }
                (Expr::Div(l, r), PathStep::Left) => {
                    let new_l = reconstruct_recursive(context, l, remaining_path, replacement);
                    context.add_raw(Expr::Div(new_l, r)) // Use add_raw to preserve structure
                }
                (Expr::Div(l, r), PathStep::Right) => {
                    let new_r = reconstruct_recursive(context, r, remaining_path, replacement);
                    context.add_raw(Expr::Div(l, new_r)) // Use add_raw to preserve structure
                }
                (Expr::Pow(b, e), PathStep::Base) => {
                    let new_b = reconstruct_recursive(context, b, remaining_path, replacement);
                    context.add_raw(Expr::Pow(new_b, e)) // Use add_raw to preserve structure
                }
                (Expr::Pow(b, e), PathStep::Exponent) => {
                    let new_e = reconstruct_recursive(context, e, remaining_path, replacement);
                    context.add_raw(Expr::Pow(b, new_e)) // Use add_raw to preserve structure
                }
                (Expr::Neg(e), PathStep::Inner) => {
                    let new_e = reconstruct_recursive(context, e, remaining_path, replacement);
                    context.add_raw(Expr::Neg(new_e)) // Use add_raw to preserve structure
                }
                (Expr::Function(name, args), PathStep::Arg(idx)) => {
                    let mut new_args = args;
                    if *idx < new_args.len() {
                        new_args[*idx] = reconstruct_recursive(
                            context,
                            new_args[*idx],
                            remaining_path,
                            replacement,
                        );
                        context.add_raw(Expr::Function(name, new_args)) // Use add_raw to preserve structure
                    } else {
                        root
                    }
                }
                (Expr::Hold(inner), PathStep::Inner) => {
                    let new_inner =
                        reconstruct_recursive(context, inner, remaining_path, replacement);
                    context.add_raw(Expr::Hold(new_inner))
                }
                // Leaves — no children, path cannot descend further
                (Expr::Number(_), _)
                | (Expr::Constant(_), _)
                | (Expr::Variable(_), _)
                | (Expr::SessionRef(_), _)
                | (Expr::Matrix { .. }, _) => root,
                // Path mismatch: valid expr but wrong PathStep direction
                _ => root,
            }
        }

        let new_root = reconstruct_recursive(
            self.context,
            self.root_expr,
            &self.current_path,
            replacement,
        );
        self.root_expr = new_root; // Update root for next step
        new_root
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
        rewrite: &crate::rule::Rewrite,
    ) -> ExprId {
        if self.steps_mode != StepsMode::Off {
            let main_new_expr = rewrite.new_expr;
            let main_description = &rewrite.description;
            let main_before_local = rewrite.before_local;
            let main_after_local = rewrite.after_local;
            let main_assumptions = rewrite.assumption_events.clone();
            let main_required = rewrite.required_conditions.clone();
            let main_poly_proof = rewrite.poly_proof.clone();
            let main_substeps = rewrite.substeps.clone();
            let chained_rewrites = rewrite.chained.clone();

            // Determine final result (last of chained, or main rewrite)
            let final_result = chained_rewrites
                .last()
                .map(|c| c.after)
                .unwrap_or(main_new_expr);

            let global_before = self.root_expr;
            let main_global_after = self.reconstruct_at_path(main_new_expr);

            // Main step
            let mut step = Step::with_snapshots(
                main_description,
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
                let chain_global_before = self.reconstruct_at_path(current);
                let chain_global_after = self.reconstruct_at_path(chain_rw.after);

                let mut chain_step = Step::with_snapshots(
                    &chain_rw.description,
                    rule.name(),
                    current,
                    chain_rw.after,
                    self.current_path.clone(),
                    Some(self.context),
                    chain_global_before,
                    chain_global_after,
                );
                chain_step.importance = chain_rw.importance.unwrap_or_else(|| rule.importance());
                {
                    let meta = chain_step.meta_mut();
                    meta.before_local = chain_rw.before_local;
                    meta.after_local = chain_rw.after_local;
                    meta.assumption_events = chain_rw.assumption_events;
                    meta.required_conditions = chain_rw.required_conditions;
                    meta.poly_proof = chain_rw.poly_proof;
                    meta.is_chained = true;
                }
                self.steps.push(chain_step);

                current = chain_rw.after;
            }

            final_result
        } else {
            // Without steps, just compute final result
            rewrite
                .chained
                .last()
                .map(|c| c.after)
                .unwrap_or(rewrite.new_expr)
        }
    }
}
