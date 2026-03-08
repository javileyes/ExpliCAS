//! Expression transform helpers: binary, pow, div, and function simplification.
//!
//! These methods are extracted with `#[inline(never)]` to reduce the stack frame
//! size of `transform_expr_recursive`.

use super::*;
use cas_math::factoring_support::{
    try_rewrite_difference_of_squares_product_expr, DifferenceOfSquaresProductRewriteKind,
};
use cas_math::pow_preorder_support::try_plan_sqrt_square_pow_rewrite;

fn format_difference_of_squares_product_desc(
    kind: DifferenceOfSquaresProductRewriteKind,
) -> &'static str {
    match kind {
        DifferenceOfSquaresProductRewriteKind::Basic => "(a-b)(a+b) = a² - b²",
        DifferenceOfSquaresProductRewriteKind::NaryConjugateProduct => {
            "(U+V)(U-V) = U² - V² (conjugate product)"
        }
        DifferenceOfSquaresProductRewriteKind::NaryScan => "(a-b)(a+b)·… = (a²-b²)·… (n-ary scan)",
    }
}

impl<'a> LocalSimplificationTransformer<'a> {
    /// Transform binary expression (Add/Sub/Mul) by simplifying children.
    /// Extracted to reduce stack frame size in transform_expr_recursive.
    #[inline(never)]
    pub(super) fn transform_binary(
        &mut self,
        id: ExprId,
        l: ExprId,
        r: ExprId,
        op: BinaryOp,
    ) -> ExprId {
        // PRE-ORDER: For Mul, detect conjugate pairs in the factor chain BEFORE
        // child simplification. This prevents canonicalization (sqrt→Pow) from
        // breaking structural matching, and prevents DistributeRule from splitting
        // the conjugate pair across inner Mul nodes after factor reordering.
        // Pattern: (a+b)*(a-b)*... → (a²-b²)*...
        if matches!(op, BinaryOp::Mul) {
            if let Some(result) = self.try_conjugate_pair_contraction(id) {
                return result;
            }
        }

        let new_l = self.transform_child_at(id, crate::step::PathStep::Left, l);
        let new_r = self.transform_child_at(id, crate::step::PathStep::Right, r);

        if new_l != l || new_r != r {
            let expr = match op {
                BinaryOp::Add => Expr::Add(new_l, new_r),
                BinaryOp::Sub => Expr::Sub(new_l, new_r),
                BinaryOp::Mul => Expr::Mul(new_l, new_r),
                BinaryOp::Div => Expr::Div(new_l, new_r),
            };
            self.context.add(expr)
        } else {
            id
        }
    }

    /// PRE-ORDER: Flatten a Mul chain and detect conjugate factor pairs.
    ///
    /// If found, contracts (a+b)*(a-b) → (a²-b²), rebuilds the product with
    /// remaining factors, records a step, and re-enters simplification.
    /// Returns None if no conjugate pair is found.
    #[inline(never)]
    fn try_conjugate_pair_contraction(&mut self, id: ExprId) -> Option<ExprId> {
        let rewrite = try_rewrite_difference_of_squares_product_expr(self.context, id)?;
        self.record_step(
            format_difference_of_squares_product_desc(rewrite.kind),
            "Difference of Squares",
            id,
            rewrite.rewritten,
        );
        Some(self.transform_expr_recursive(rewrite.rewritten))
    }

    /// Transform Pow expression with early detection for sqrt-of-square patterns.
    /// Extracted with #[inline(never)] to reduce stack frame size.
    #[inline(never)]
    pub(super) fn transform_pow(&mut self, id: ExprId, base: ExprId, exp: ExprId) -> ExprId {
        // EARLY DETECTION: sqrt-of-square pattern (u^2)^(1/2) -> |u|
        // Must check BEFORE recursing into children to prevent binomial expansion
        if let Some(plan) = try_plan_sqrt_square_pow_rewrite(self.context, base, exp) {
            self.record_step(plan.identity_desc, plan.rule_name, id, plan.rewritten);
            return self.transform_expr_recursive(plan.rewritten);
        }

        // Check if this Pow is canonical before recursing into children
        if crate::canonical_forms::is_canonical_form(self.context, id) {
            debug!(
                "Skipping simplification of canonical Pow: {:?}",
                self.context.get(id)
            );
            return id;
        }

        // Simplify children
        let new_b = self.transform_child_at(id, crate::step::PathStep::Base, base);
        let new_e = self.transform_child_at(id, crate::step::PathStep::Exponent, exp);

        if new_b != base || new_e != exp {
            self.context.add(Expr::Pow(new_b, new_e))
        } else {
            id
        }
    }

    /// Transform Function expression by simplifying children.
    /// Extracted with #[inline(never)] to reduce stack frame size.
    #[inline(never)]
    pub(super) fn transform_function(
        &mut self,
        id: ExprId,
        fn_id: SymbolId,
        args: Vec<ExprId>,
    ) -> ExprId {
        let name = self.context.sym_name(fn_id);
        // Check if this function is canonical before recursing into children
        if (name == "sqrt" || name == "abs")
            && crate::canonical_forms::is_canonical_form(self.context, id)
        {
            debug!(
                "Skipping simplification of canonical Function: {:?}",
                self.context.get(id)
            );
            return id;
        }

        // HoldAll semantics: do NOT simplify arguments for these functions
        if is_hold_all_function(name) {
            debug!(
                "HoldAll function, skipping child simplification: {:?}",
                self.context.get(id)
            );
            return id;
        }

        // Simplify children
        let mut new_args = Vec::with_capacity(args.len());
        let mut changed = false;
        for (i, arg) in args.iter().enumerate() {
            let new_arg = self.transform_child_at(id, crate::step::PathStep::Arg(i), *arg);

            if new_arg != *arg {
                changed = true;
            }
            new_args.push(new_arg);
        }

        if changed {
            self.context.add(Expr::Function(fn_id, new_args))
        } else {
            id
        }
    }

    /// Transform Div expression with early detection for difference-of-squares pattern.
    /// Extracted with #[inline(never)] to reduce stack frame size.
    #[inline(never)]
    pub(super) fn transform_div(&mut self, id: ExprId, l: ExprId, r: ExprId) -> ExprId {
        // EARLY DETECTION: (A² - B²) / (A ± B) pattern
        if let Some(early_result) = crate::rules::algebra::try_difference_of_squares_preorder(
            self.context,
            id,
            l,
            r,
            self.collect_steps_enabled(),
            &mut self.steps,
            &self.current_path,
        ) {
            // Note: don't decrement depth here - transform_expr_recursive manages it
            return self.transform_expr_recursive(early_result);
        }

        // Simplify children
        let new_l = self.transform_child_at(id, crate::step::PathStep::Left, l);
        let new_r = self.transform_child_at(id, crate::step::PathStep::Right, r);

        if new_l != l || new_r != r {
            self.context.add(Expr::Div(new_l, new_r))
        } else {
            id
        }
    }
}
