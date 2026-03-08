//! Identity zero rules: pattern-driven cancellation for trig identities.
//!
//! Extracted from `half_angle_phase_rules.rs` to reduce file size.
//! Contains:
//! - WeierstrassSinIdentityZeroRule: sin(x) - 2*tan(x/2)/(1+tan²(x/2)) → 0
//! - WeierstrassCosIdentityZeroRule: cos(x) - (1-tan²(x/2))/(1+tan²(x/2)) → 0
//! - Sin4xIdentityZeroRule: sin(4t) - 4*sin(t)*cos(t)*(cos²(t)-sin²(t)) → 0
//! - TanDifferenceIdentityZeroRule: tan(a-b) - (tan(a)-tan(b))/(1+tan(a)*tan(b)) → 0

use crate::rule::Rewrite;
use cas_ast::ExprId;
use cas_math::trig_identity_zero_support::{
    try_rewrite_sin4x_identity_zero_expr, try_rewrite_tan_difference_identity_zero_expr,
    IdentityZeroRewriteKind,
};
use cas_math::trig_weierstrass_support::{
    try_rewrite_weierstrass_cos_identity_zero_expr, try_rewrite_weierstrass_sin_identity_zero_expr,
};

fn format_identity_zero_desc(kind: IdentityZeroRewriteKind) -> &'static str {
    match kind {
        IdentityZeroRewriteKind::WeierstrassSin => {
            "sin(x) = 2·tan(x/2)/(1 + tan²(x/2)) [Weierstrass]"
        }
        IdentityZeroRewriteKind::WeierstrassCos => {
            "cos(x) = (1 - tan²(x/2))/(1 + tan²(x/2)) [Weierstrass]"
        }
        IdentityZeroRewriteKind::TanDifference => "tan(a-b) = (tan(a)-tan(b))/(1+tan(a)·tan(b))",
        IdentityZeroRewriteKind::Sin4x => "sin(4t) = 4·sin(t)·cos(t)·(cos²(t)-sin²(t))",
    }
}

// =============================================================================
// WEIERSTRASS IDENTITY ZERO RULES (Pattern-Driven Cancellation)
// =============================================================================
// These rules detect the complete Weierstrass identity patterns and cancel to 0
// directly, avoiding explosive expansion through tan→sin/cos conversion.
//
// sin(x) - 2*tan(x/2)/(1 + tan(x/2)²) → 0
// cos(x) - (1 - tan(x/2)²)/(1 + tan(x/2)²) → 0

// WeierstrassSinIdentityZeroRule: sin(x) - 2*tan(x/2)/(1+tan²(x/2)) → 0
// Pattern-driven cancellation, no expansion.
pub struct WeierstrassSinIdentityZeroRule;

impl crate::rule::Rule for WeierstrassSinIdentityZeroRule {
    fn name(&self) -> &str {
        "Weierstrass Sin Identity Zero"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
        _parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        let rewrite = try_rewrite_weierstrass_sin_identity_zero_expr(ctx, expr)?;
        Some(Rewrite::new(ctx.num(0)).desc(format_identity_zero_desc(rewrite.kind)))
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::ADD_SUB)
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::High
    }

    fn priority(&self) -> i32 {
        200 // Very high - must run BEFORE Pythagorean 1+tan²→sec²
    }
}

// WeierstrassCosIdentityZeroRule: cos(x) - (1-tan²(x/2))/(1+tan²(x/2)) → 0
pub struct WeierstrassCosIdentityZeroRule;

impl crate::rule::Rule for WeierstrassCosIdentityZeroRule {
    fn name(&self) -> &str {
        "Weierstrass Cos Identity Zero"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
        _parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        let rewrite = try_rewrite_weierstrass_cos_identity_zero_expr(ctx, expr)?;
        Some(Rewrite::new(ctx.num(0)).desc(format_identity_zero_desc(rewrite.kind)))
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::ADD_SUB)
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::High
    }

    fn priority(&self) -> i32 {
        200 // Very high - must run BEFORE Pythagorean 1+tan²→sec²
    }
}

// =============================================================================
// Sin4xIdentityZeroRule: sin(4t) - 4*sin(t)*cos(t)*(cos²(t)-sin²(t)) → 0
// =============================================================================
// Recognizes the sin(4x) expansion identity directly in cancellation context.

pub struct Sin4xIdentityZeroRule;

impl crate::rule::Rule for Sin4xIdentityZeroRule {
    fn name(&self) -> &str {
        "Sin 4x Identity Zero"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
        _parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        let rewrite = try_rewrite_sin4x_identity_zero_expr(ctx, expr)?;
        Some(Rewrite::new(ctx.num(0)).desc(format_identity_zero_desc(rewrite.kind)))
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::ADD_SUB)
    }

    fn priority(&self) -> i32 {
        200 // Run before expansion
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::High
    }
}

// =============================================================================
// TanDifferenceIdentityZeroRule: tan(a-b) - (tan(a)-tan(b))/(1+tan(a)*tan(b)) → 0
// =============================================================================
// Recognizes the tangent difference identity directly in cancellation context.

pub struct TanDifferenceIdentityZeroRule;

impl crate::rule::Rule for TanDifferenceIdentityZeroRule {
    fn name(&self) -> &str {
        "Tangent Difference Identity Zero"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
        _parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        let rewrite = try_rewrite_tan_difference_identity_zero_expr(ctx, expr)?;
        Some(Rewrite::new(ctx.num(0)).desc(format_identity_zero_desc(rewrite.kind)))
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::ADD_SUB)
    }

    fn priority(&self) -> i32 {
        200 // Run before tan→sin/cos expansion
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::High
    }
}
