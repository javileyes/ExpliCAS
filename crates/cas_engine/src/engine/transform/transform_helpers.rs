//! Expression transform helpers: binary, pow, div, and function simplification.
//!
//! These methods are extracted with `#[inline(never)]` to reduce the stack frame
//! size of `transform_expr_recursive`.

use super::*;
use cas_math::factoring_support::{
    try_rewrite_difference_of_squares_product_expr, DifferenceOfSquaresProductRewriteKind,
};
use cas_math::logarithm_inverse_support::try_rewrite_exponential_log_inverse_expr;
use cas_math::numeric::as_i64;
use cas_math::pow_preorder_support::{try_plan_sqrt_square_pow_rewrite, SqrtSquarePowRewriteKind};
use smallvec::SmallVec;

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

fn format_sqrt_square_pow_plan(kind: SqrtSquarePowRewriteKind) -> (&'static str, &'static str) {
    match kind {
        SqrtSquarePowRewriteKind::PowSquare => {
            ("sqrt(u^2) = |u|", "Simplify Square Root of Square")
        }
        SqrtSquarePowRewriteKind::RepeatedMul => {
            ("sqrt(u * u) = |u|", "Simplify Square Root of Product")
        }
    }
}

fn polynomial_identity_preorder_desc(
    kind: crate::polynomial_identity_support::PolynomialIdentityProofKind,
) -> &'static str {
    match kind {
        crate::polynomial_identity_support::PolynomialIdentityProofKind::Direct => {
            "Polynomial identity: normalize and cancel to 0"
        }
        crate::polynomial_identity_support::PolynomialIdentityProofKind::OpaqueSubstitution => {
            "Polynomial identity (opaque substitution): cancel to 0"
        }
        crate::polynomial_identity_support::PolynomialIdentityProofKind::OpaqueRootRelation => {
            "Polynomial identity (opaque root relation): cancel to 0"
        }
    }
}

fn is_symbolic_atom(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Variable(_) | Expr::Constant(_))
}

fn is_plain_symbolic_binomial(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Add(left, right) | Expr::Sub(left, right) => {
            is_symbolic_atom(ctx, *left) && is_symbolic_atom(ctx, *right)
        }
        Expr::Neg(inner) => is_plain_symbolic_binomial(ctx, *inner),
        _ => false,
    }
}

fn is_same_symbolic_atom_fraction(ctx: &Context, left: ExprId, right: ExprId) -> bool {
    is_symbolic_atom(ctx, left)
        && is_symbolic_atom(ctx, right)
        && cas_ast::ordering::compare_expr(ctx, left, right) == std::cmp::Ordering::Equal
}

fn is_symbolic_power_over_same_atom_noop(ctx: &Context, left: ExprId, right: ExprId) -> bool {
    let Expr::Pow(base, exp) = ctx.get(left) else {
        return false;
    };

    is_symbolic_atom(ctx, *base)
        && is_symbolic_atom(ctx, *exp)
        && is_symbolic_atom(ctx, right)
        && cas_ast::ordering::compare_expr(ctx, *base, right) == std::cmp::Ordering::Equal
}

fn collect_positive_add_terms_3(
    ctx: &Context,
    expr: ExprId,
    out: &mut SmallVec<[ExprId; 3]>,
) -> bool {
    match ctx.get(expr) {
        Expr::Add(left, right) => {
            collect_positive_add_terms_3(ctx, *left, out)
                && collect_positive_add_terms_3(ctx, *right, out)
        }
        Expr::Sub(_, _) | Expr::Neg(_) => false,
        _ => {
            if out.len() == 3 {
                return false;
            }
            out.push(expr);
            true
        }
    }
}

fn collect_signed_add_terms_3(
    ctx: &Context,
    expr: ExprId,
    positive: bool,
    out: &mut SmallVec<[(ExprId, bool); 3]>,
) -> bool {
    match ctx.get(expr) {
        Expr::Add(left, right) => {
            collect_signed_add_terms_3(ctx, *left, positive, out)
                && collect_signed_add_terms_3(ctx, *right, positive, out)
        }
        Expr::Sub(left, right) => {
            collect_signed_add_terms_3(ctx, *left, positive, out)
                && collect_signed_add_terms_3(ctx, *right, !positive, out)
        }
        Expr::Neg(inner) => collect_signed_add_terms_3(ctx, *inner, !positive, out),
        _ => {
            if out.len() == 3 {
                return false;
            }
            out.push((expr, positive));
            true
        }
    }
}

fn collect_mul_factors_3(ctx: &Context, expr: ExprId, out: &mut SmallVec<[ExprId; 3]>) -> bool {
    match ctx.get(expr) {
        Expr::Mul(left, right) => {
            collect_mul_factors_3(ctx, *left, out) && collect_mul_factors_3(ctx, *right, out)
        }
        _ => {
            if out.len() == 3 {
                return false;
            }
            out.push(expr);
            true
        }
    }
}

fn multiset_matches_exact(ctx: &Context, actual: &[ExprId], expected: &[ExprId]) -> bool {
    if actual.len() != expected.len() {
        return false;
    }

    let mut used = [false; 3];
    for wanted in expected {
        let mut matched = false;
        for (idx, candidate) in actual.iter().enumerate() {
            if used[idx] {
                continue;
            }
            if *candidate == *wanted
                || cas_ast::ordering::compare_expr(ctx, *candidate, *wanted)
                    == std::cmp::Ordering::Equal
            {
                used[idx] = true;
                matched = true;
                break;
            }
        }
        if !matched {
            return false;
        }
    }

    true
}

fn is_exact_two_ab_product(ctx: &mut Context, expr: ExprId, a: ExprId, b: ExprId) -> bool {
    let mut factors = SmallVec::<[ExprId; 3]>::new();
    if !collect_mul_factors_3(ctx, expr, &mut factors) || factors.len() != 3 {
        return false;
    }

    let two = ctx.num(2);
    multiset_matches_exact(ctx, &factors, &[two, a, b])
}

fn is_exact_binomial_square_fraction_preorder(ctx: &mut Context, num: ExprId, den: ExprId) -> bool {
    let Expr::Pow(base, exp) = ctx.get(den) else {
        return false;
    };
    if as_i64(ctx, *exp) != Some(2) {
        return false;
    }
    let Expr::Add(a, b) = ctx.get(*base) else {
        return false;
    };
    let a = *a;
    let b = *b;

    let exp_two = ctx.num(2);
    let a_sq = ctx.add(Expr::Pow(a, exp_two));
    let exp_two_b = ctx.num(2);
    let b_sq = ctx.add(Expr::Pow(b, exp_two_b));

    let mut terms = SmallVec::<[ExprId; 3]>::new();
    if !collect_positive_add_terms_3(ctx, num, &mut terms) || terms.len() != 3 {
        return false;
    }

    let mut squares = SmallVec::<[ExprId; 2]>::new();
    let mut middle = None;
    for term in terms {
        if term == a_sq
            || term == b_sq
            || cas_ast::ordering::compare_expr(ctx, term, a_sq) == std::cmp::Ordering::Equal
            || cas_ast::ordering::compare_expr(ctx, term, b_sq) == std::cmp::Ordering::Equal
        {
            squares.push(term);
        } else if middle.is_none() {
            middle = Some(term);
        } else {
            return false;
        }
    }

    squares.len() == 2
        && multiset_matches_exact(ctx, &squares, &[a_sq, b_sq])
        && middle.is_some_and(|term| is_exact_two_ab_product(ctx, term, a, b))
}

fn try_exact_perfect_square_minus_fraction_preorder(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
) -> Option<ExprId> {
    let Expr::Sub(a, b) = ctx.get(den) else {
        return None;
    };
    let a = *a;
    let b = *b;

    let exp_two = ctx.num(2);
    let a_sq = ctx.add(Expr::Pow(a, exp_two));
    let exp_two_b = ctx.num(2);
    let b_sq = ctx.add(Expr::Pow(b, exp_two_b));

    let mut terms = SmallVec::<[(ExprId, bool); 3]>::new();
    if !collect_signed_add_terms_3(ctx, num, true, &mut terms) || terms.len() != 3 {
        return None;
    }

    let mut positives = SmallVec::<[ExprId; 2]>::new();
    let mut negative = None;
    for (term, positive) in terms {
        if positive {
            if positives.len() == 2 {
                return None;
            }
            positives.push(term);
        } else if negative.is_none() {
            negative = Some(term);
        } else {
            return None;
        }
    }

    if positives.len() != 2
        || !multiset_matches_exact(ctx, &positives, &[a_sq, b_sq])
        || !negative.is_some_and(|term| is_exact_two_ab_product(ctx, term, a, b))
    {
        return None;
    }

    Some(den)
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
        // PRE-ORDER: For Add/Sub, try exact polynomial-identity closure before
        // children are simplified. The bottom-up pipeline can otherwise
        // normalize opaque rational/root atoms into bulky residual fractions
        // that the same proof helper would have closed immediately.
        if matches!(op, BinaryOp::Add | BinaryOp::Sub)
            && matches!(
                self.current_phase,
                crate::phase::SimplifyPhase::Core
                    | crate::phase::SimplifyPhase::Transform
                    | crate::phase::SimplifyPhase::PostCleanup
            )
            && !self.initial_parent_ctx.is_solve_context()
        {
            if let Some(plan) =
                crate::polynomial_identity_support::try_prove_polynomial_identity_zero_expr(
                    self.context,
                    id,
                )
            {
                let zero = self.context.num(0);
                self.record_step(
                    polynomial_identity_preorder_desc(plan.kind),
                    "Polynomial Identity",
                    id,
                    zero,
                );
                return zero;
            }
        }

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
        let allow_hidden_solve_pow_preorder = !self.collect_steps_enabled()
            && self.event_listener.is_none()
            && self.current_phase == crate::SimplifyPhase::Core
            && self.initial_parent_ctx.is_solve_context()
            && !self.initial_parent_ctx.domain_mode().is_strict();
        if allow_hidden_solve_pow_preorder {
            if let Some(cas_math::power_identity_support::PowerIdentityPolicyPattern::PowZero {
                base,
                base_is_literal_zero: false,
            }) = cas_math::power_identity_support::classify_power_identity_policy_pattern(
                self.context,
                id,
            ) {
                if is_symbolic_atom(self.context, base) {
                    return self.context.num(1);
                }
            }

            if let Some(rewrite) = try_rewrite_exponential_log_inverse_expr(self.context, id) {
                if is_symbolic_atom(self.context, rewrite.rewritten) {
                    return rewrite.rewritten;
                }
            }
        }

        // EARLY DETECTION: sqrt-of-square pattern (u^2)^(1/2) -> |u|
        // Must check BEFORE recursing into children to prevent binomial expansion
        if let Some(plan) = try_plan_sqrt_square_pow_rewrite(self.context, base, exp) {
            let (identity_desc, rule_name) = format_sqrt_square_pow_plan(plan.kind);
            self.record_step(identity_desc, rule_name, id, plan.rewritten);
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
        let allow_identical_atom_fraction_preorder = !self.collect_steps_enabled()
            && self.event_listener.is_none()
            && self.current_phase == crate::SimplifyPhase::Core
            && self.initial_parent_ctx.is_solve_context()
            && !self.initial_parent_ctx.domain_mode().is_strict()
            && is_same_symbolic_atom_fraction(self.context, l, r);
        if allow_identical_atom_fraction_preorder {
            return self.context.num(1);
        }

        let allow_same_atom_power_noop_preorder = !self.collect_steps_enabled()
            && self.event_listener.is_none()
            && self.current_phase == crate::SimplifyPhase::Core
            && self.initial_parent_ctx.is_solve_context()
            && !self.initial_parent_ctx.domain_mode().is_strict()
            && is_symbolic_power_over_same_atom_noop(self.context, l, r);
        if allow_same_atom_power_noop_preorder {
            return id;
        }

        let allow_scalar_multiple_preorder = !self.collect_steps_enabled()
            && self.event_listener.is_none()
            && self.current_phase == crate::SimplifyPhase::Core
            && self.initial_parent_ctx.is_solve_context()
            && match self.initial_parent_ctx.simplify_purpose() {
                crate::SimplifyPurpose::Eval => {
                    self.initial_parent_ctx.context_mode() == crate::options::ContextMode::Solve
                }
                crate::SimplifyPurpose::SolvePrepass => {
                    cas_solver_core::solve_safety_policy::safe_for_prepass(
                        crate::SolveSafety::NeedsCondition(crate::ConditionClass::Definability),
                    )
                }
                crate::SimplifyPurpose::SolveTactic => {
                    let domain_mode = self.initial_parent_ctx.domain_mode();
                    cas_solver_core::solve_safety_policy::safe_for_tactic_with_domain_flags(
                        crate::SolveSafety::NeedsCondition(crate::ConditionClass::Definability),
                        matches!(domain_mode, crate::DomainMode::Assume),
                        matches!(domain_mode, crate::DomainMode::Strict),
                    )
                }
            };
        if allow_scalar_multiple_preorder {
            if let Some(early_result) =
                crate::rules::algebra::try_structural_scalar_multiple_preorder(self.context, l, r)
            {
                return early_result;
            }
        }

        let allow_exact_binomial_square_preorder = !self.collect_steps_enabled()
            && self.event_listener.is_none()
            && self.current_phase == crate::SimplifyPhase::Core
            && self.initial_parent_ctx.is_solve_context()
            && !self.initial_parent_ctx.domain_mode().is_strict();
        if allow_exact_binomial_square_preorder
            && is_exact_binomial_square_fraction_preorder(self.context, l, r)
        {
            return self.context.num(1);
        }

        // EARLY DETECTION: (A² - B²) / (A ± B) pattern
        let allow_difference_of_squares_preorder = match self.initial_parent_ctx.simplify_purpose()
        {
            crate::SimplifyPurpose::Eval => true,
            crate::SimplifyPurpose::SolvePrepass => false,
            crate::SimplifyPurpose::SolveTactic => {
                let domain_mode = self.initial_parent_ctx.domain_mode();
                cas_solver_core::solve_safety_policy::safe_for_tactic_with_domain_flags(
                    crate::SolveSafety::NeedsCondition(crate::ConditionClass::Definability),
                    matches!(domain_mode, crate::DomainMode::Assume),
                    matches!(domain_mode, crate::DomainMode::Strict),
                )
            }
        };
        if allow_difference_of_squares_preorder
            && self.event_listener.is_none()
            && !self.initial_parent_ctx.domain_mode().is_strict()
        {
            if let Some(early_result) =
                crate::rules::algebra::try_exact_common_factor_mul_fraction_preorder(
                    self.context,
                    id,
                    l,
                    r,
                    self.collect_steps_enabled(),
                    &mut self.steps,
                    &self.current_path,
                )
            {
                if !self.collect_steps_enabled()
                    && self.current_phase == crate::SimplifyPhase::Core
                    && self.initial_parent_ctx.is_solve_context()
                {
                    return early_result;
                }
                return self.transform_expr_recursive(early_result);
            }
        }
        if allow_difference_of_squares_preorder {
            if let Some(early_result) = crate::rules::algebra::try_difference_of_squares_preorder(
                self.context,
                id,
                l,
                r,
                self.collect_steps_enabled(),
                &mut self.steps,
                &self.current_path,
            ) {
                if !self.collect_steps_enabled()
                    && self.event_listener.is_none()
                    && self.current_phase == crate::SimplifyPhase::Core
                    && self.initial_parent_ctx.is_solve_context()
                    && is_plain_symbolic_binomial(self.context, early_result)
                {
                    return early_result;
                }
                // Note: don't decrement depth here - transform_expr_recursive manages it
                return self.transform_expr_recursive(early_result);
            }
        }

        // Similar pre-order fast path for perfect-square-minus fractions, but only
        // when no listener is attached so we don't widen the existing event-gap
        // behavior beyond the hidden hot path.
        if allow_difference_of_squares_preorder && self.event_listener.is_none() {
            if !self.collect_steps_enabled()
                && self.current_phase == crate::SimplifyPhase::Core
                && self.initial_parent_ctx.is_solve_context()
                && !self.initial_parent_ctx.domain_mode().is_strict()
            {
                if let Some(early_result) =
                    try_exact_perfect_square_minus_fraction_preorder(self.context, l, r)
                {
                    return early_result;
                }
            }

            if let Some(early_result) = crate::rules::algebra::try_perfect_square_minus_preorder(
                self.context,
                id,
                l,
                r,
                self.collect_steps_enabled(),
                &mut self.steps,
                &self.current_path,
            ) {
                if !self.collect_steps_enabled()
                    && self.current_phase == crate::SimplifyPhase::Core
                    && self.initial_parent_ctx.is_solve_context()
                    && is_plain_symbolic_binomial(self.context, early_result)
                {
                    return early_result;
                }
                return self.transform_expr_recursive(early_result);
            }
        }

        // Exact-shape hidden fast path for `(a^3 ± b^3)/(a±b)`. This avoids the
        // child-recursion sign/canonicalization churn on the raw hotspot inputs
        // without paying the broader planner cost that regressed earlier.
        if allow_difference_of_squares_preorder
            && !self.collect_steps_enabled()
            && self.event_listener.is_none()
            && self.current_phase == crate::SimplifyPhase::Core
            && self.initial_parent_ctx.is_solve_context()
        {
            if let Some(early_result) =
                crate::rules::algebra::try_exact_sum_diff_of_cubes_preorder(self.context, l, r)
            {
                return early_result;
            }
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
