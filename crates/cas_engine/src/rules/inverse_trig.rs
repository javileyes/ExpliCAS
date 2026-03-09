use crate::define_rule;
use crate::rule::Rewrite;
use cas_math::inverse_trig_composition_support::{
    InverseTrigCompositionKind, InverseTrigReciprocalRewriteKind, InverseTrigUnaryRewriteKind,
    PrincipalBranchInverseTrigKind,
};

fn format_inverse_trig_composition_desc(
    kind: InverseTrigCompositionKind,
    assume_defined: bool,
) -> &'static str {
    match (kind, assume_defined) {
        (InverseTrigCompositionKind::SinArcsin, false) => "sin(arcsin(x)) = x",
        (InverseTrigCompositionKind::SinArcsin, true) => {
            "sin(arcsin(x)) = x (assuming x ∈ [-1, 1])"
        }
        (InverseTrigCompositionKind::CosArccos, false) => "cos(arccos(x)) = x",
        (InverseTrigCompositionKind::CosArccos, true) => {
            "cos(arccos(x)) = x (assuming x ∈ [-1, 1])"
        }
        (InverseTrigCompositionKind::TanArctan, _) => "tan(arctan(x)) = x",
        (InverseTrigCompositionKind::ArctanTanArctan, _) => {
            "arctan(tan(arctan(u))) = arctan(u) (principal branch)"
        }
    }
}

fn format_inverse_trig_reciprocal_desc(kind: InverseTrigReciprocalRewriteKind) -> &'static str {
    match kind {
        InverseTrigReciprocalRewriteKind::ArcsecToArccos => "arcsec(x) → arccos(1/x)",
        InverseTrigReciprocalRewriteKind::ArccscToArcsin => "arccsc(x) → arcsin(1/x)",
        InverseTrigReciprocalRewriteKind::ArccotToArctan => "arccot(x) → arctan(1/x)",
    }
}

fn format_inverse_trig_unary_desc(kind: InverseTrigUnaryRewriteKind) -> &'static str {
    match kind {
        InverseTrigUnaryRewriteKind::ArcsinNegative => "arcsin(-x) = -arcsin(x)",
        InverseTrigUnaryRewriteKind::ArctanNegative => "arctan(-x) = -arctan(x)",
        InverseTrigUnaryRewriteKind::ArccosNegative => "arccos(-x) = π - arccos(x)",
    }
}

fn format_principal_branch_inverse_trig_desc(kind: PrincipalBranchInverseTrigKind) -> &'static str {
    match kind {
        PrincipalBranchInverseTrigKind::ArcsinSin => "arcsin(sin(u)) → u (principal branch)",
        PrincipalBranchInverseTrigKind::ArccosCos => "arccos(cos(u)) → u (principal branch)",
        PrincipalBranchInverseTrigKind::ArctanTan => "arctan(tan(u)) → u (principal branch)",
        PrincipalBranchInverseTrigKind::ArctanSinOverCos => {
            "arctan(sin(u)/cos(u)) → u (principal branch)"
        }
    }
}
use cas_ast::{Context, Expr, ExprId};
use cas_math::inverse_trig_composition_support::{
    try_plan_atan_rational_add_expr, try_plan_inverse_atan_reciprocal_add_expr,
    try_plan_inverse_trig_composition_expr, try_plan_inverse_trig_sum_add_expr,
    try_plan_principal_branch_inverse_trig_expr, try_rewrite_arccot_to_arctan_expr,
    try_rewrite_arccsc_to_arcsin_expr, try_rewrite_arcsec_to_arccos_expr,
    try_rewrite_inverse_trig_negative_expr,
};

// ==================== Inverse Trig Identity Rules ====================

// Rule 1: Composition Identities - sin(arcsin(x)) = x, etc.
// sin(arcsin(x)) and cos(arccos(x)) require x ∈ [-1, 1] (domain of inverse)
// tan(arctan(x)) is always valid (arctan has domain R)
pub struct InverseTrigCompositionRule;

impl crate::rule::Rule for InverseTrigCompositionRule {
    fn name(&self) -> &str {
        "Inverse Trig Composition"
    }

    fn apply(
        &self,
        ctx: &mut Context,
        expr: ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        let mode = parent_ctx.domain_mode();
        let plan = try_plan_inverse_trig_composition_expr(
            ctx,
            expr,
            matches!(mode, crate::DomainMode::Assume),
            matches!(mode, crate::DomainMode::Strict),
        )?;
        let mut rewrite = Rewrite::new(plan.rewritten).desc(format_inverse_trig_composition_desc(
            plan.kind,
            plan.assume_defined_expr.is_some(),
        ));
        if let Some(defined_expr) = plan.assume_defined_expr {
            rewrite = rewrite.assume(crate::AssumptionEvent::defined(ctx, defined_expr));
        }
        Some(rewrite)
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::FUNCTION)
    }
}

// Rule 2: arcsin(x) + arccos(x) = π/2
// Enhanced to search across all additive terms (n-ary matching)
// and handle negated pairs: -arcsin(x) - arccos(x) = -π/2
define_rule!(
    InverseTrigSumRule,
    "Inverse Trig Sum Identity",
    Some(crate::target_kind::TargetKindSet::ADD),
    |ctx, expr| {
        let plan = try_plan_inverse_trig_sum_add_expr(ctx, expr)?;
        Some(
            Rewrite::new(plan.final_result)
                .desc(plan.desc)
                .local(plan.local_before, plan.local_after),
        )
    }
);

// Rule 3: arctan(x) + arctan(1/x) = π/2 (for x > 0)
// Enhanced to search across all additive terms (n-ary matching)
// Only applies at root Add level (when parent is NOT Add) to ensure all pairs are visible
pub struct InverseTrigAtanRule;

impl crate::rule::Rule for InverseTrigAtanRule {
    fn name(&self) -> &str {
        "Inverse Tan Relations"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: cas_ast::ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<crate::rule::Rewrite> {
        let parent_is_add = parent_ctx
            .immediate_parent()
            .is_some_and(|p| matches!(ctx.get(p), Expr::Add(_, _)));
        let plan = try_plan_inverse_atan_reciprocal_add_expr(ctx, expr, parent_is_add)?;
        Some(
            Rewrite::new(plan.final_result)
                .desc(plan.desc)
                .local(plan.local_before, plan.local_after),
        )
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::ADD)
    }

    fn priority(&self) -> i32 {
        10 // Higher priority than Machin rule (-10) to run first
    }
}

// Rule: arctan(a) + arctan(b) = arctan((a+b)/(1-a*b)) when a,b are rational and 1-a*b > 0
// This is Machin's identity (simplified form) - enables atan(1/2)+atan(1/3) = π/4
//
// CRITICAL: Uses manual Rule impl to access parent_ctx for sub-sum guard.
// Only applies at "root sum" level - skips when parent is also Add to avoid
// interfering with reciprocal pair detection in larger sums.
pub struct AtanAddRationalRule;

impl crate::rule::Rule for AtanAddRationalRule {
    fn name(&self) -> &str {
        "Arctan Addition (Machin)"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: cas_ast::ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<crate::rule::Rewrite> {
        let parent_is_add = parent_ctx
            .immediate_parent()
            .is_some_and(|p| matches!(ctx.get(p), Expr::Add(_, _)));
        let plan = try_plan_atan_rational_add_expr(ctx, expr, parent_is_add)?;
        Some(
            crate::rule::Rewrite::new(plan.final_result)
                .desc(plan.desc)
                .local(plan.local_before, plan.local_after),
        )
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::ADD)
    }

    fn priority(&self) -> i32 {
        -10 // Lower priority than reciprocal pair detection (InverseTrigAtanRule)
    }
}

// Rule 4: Negative argument handling for inverse trig
define_rule!(
    InverseTrigNegativeRule,
    "Inverse Trig Negative Argument",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    |ctx, expr| {
        let rewrite = try_rewrite_inverse_trig_negative_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(format_inverse_trig_unary_desc(rewrite.kind)))
    }
);

// ==================== Phase 5: Inverse Function Relations ====================
// Unify inverse trig functions by converting arcsec/arccsc/arccot to arccos/arcsin/arctan

// arcsec(x) → arccos(1/x)
define_rule!(
    ArcsecToArccosRule,
    "arcsec(x) → arccos(1/x)",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    |ctx, expr| {
        let rewrite = try_rewrite_arcsec_to_arccos_expr(ctx, expr)?;
        Some(
            Rewrite::new(rewrite.rewritten).desc(format_inverse_trig_reciprocal_desc(rewrite.kind)),
        )
    }
);

// arccsc(x) → arcsin(1/x)
define_rule!(
    ArccscToArcsinRule,
    "arccsc(x) → arcsin(1/x)",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    |ctx, expr| {
        let rewrite = try_rewrite_arccsc_to_arcsin_expr(ctx, expr)?;
        Some(
            Rewrite::new(rewrite.rewritten).desc(format_inverse_trig_reciprocal_desc(rewrite.kind)),
        )
    }
);

// arccot(x) → arctan(1/x)
// Simplified version - works for all x ≠ 0 on principal branch
define_rule!(
    ArccotToArctanRule,
    "arccot(x) → arctan(1/x)",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    |ctx, expr| {
        let rewrite = try_rewrite_arccot_to_arctan_expr(ctx, expr)?;
        Some(
            Rewrite::new(rewrite.rewritten).desc(format_inverse_trig_reciprocal_desc(rewrite.kind)),
        )
    }
);

// ==================== Registration ====================

pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(InverseTrigCompositionRule));
    simplifier.add_rule(Box::new(InverseTrigSumRule));
    simplifier.add_rule(Box::new(InverseTrigAtanRule));
    // AtanAddRationalRule: Uses sub-sum guard (skips when parent is Add) to avoid
    // interfering with reciprocal pairs in larger sums.
    simplifier.add_rule(Box::new(AtanAddRationalRule));
    simplifier.add_rule(Box::new(InverseTrigNegativeRule));
    simplifier.add_rule(Box::new(ArcsecToArccosRule));
    simplifier.add_rule(Box::new(ArccscToArcsinRule));
    simplifier.add_rule(Box::new(ArccotToArctanRule));
    // PrincipalBranchInverseTrigRule: Self-gated by parent_ctx.inv_trig_policy().
    // Always registered; only applies when inv_trig == PrincipalValue.
    simplifier.add_rule(Box::new(PrincipalBranchInverseTrigRule));
}

// ==================== Principal Branch Rules (Educational) ====================
//
// These rules simplify inverse∘function compositions like arctan(tan(u)) → u.
// They are ONLY valid when u is in the principal domain, so they emit warnings.
//
// GATED BY: parent_ctx.inv_trig_policy() == InverseTrigPolicy::PrincipalValue
// This ensures these rules only fire when explicitly enabled via --inv-trig=principal

pub struct PrincipalBranchInverseTrigRule;

impl crate::rule::Rule for PrincipalBranchInverseTrigRule {
    fn name(&self) -> &str {
        "Principal Branch Inverse Trig"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: cas_ast::ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<crate::rule::Rewrite> {
        // GATE: Only apply when inv_trig policy is PrincipalValue
        if parent_ctx.inv_trig_policy() != crate::semantics::InverseTrigPolicy::PrincipalValue {
            return None;
        }

        let plan = try_plan_principal_branch_inverse_trig_expr(ctx, expr)?;
        Some(
            Rewrite::new(plan.rewritten)
                .desc(format_principal_branch_inverse_trig_desc(plan.kind))
                .local(plan.local_before, plan.local_after)
                .assume(crate::AssumptionEvent::inv_trig_principal_range(
                    ctx,
                    plan.assumption_fn,
                    plan.local_after,
                )),
        )
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::FUNCTION)
    }

    fn priority(&self) -> i32 {
        0 // Default priority
    }
}
