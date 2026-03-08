//! Algebraic constant rules - simplification rules for mathematical constants like φ (phi)
//!
//! φ (phi) is the golden ratio, defined as (1+√5)/2, and satisfies:
//! - φ² = φ + 1 (characteristic equation)
//! - 1/φ = φ - 1 (reciprocal identity)
//!
//! These rules normalize φ expressions to simpler forms.

use crate::define_rule;
use crate::rule::Rewrite;
use cas_math::constants_support::{
    try_rewrite_phi_reciprocal_expr, try_rewrite_phi_squared_expr, try_rewrite_recognize_phi_expr,
    ConstantRewriteKind,
};

fn format_constant_rewrite_desc(kind: ConstantRewriteKind) -> &'static str {
    match kind {
        ConstantRewriteKind::RecognizePhi => "(1 + √5)/2 = φ",
        ConstantRewriteKind::PhiSquared => "φ² = φ + 1",
        ConstantRewriteKind::PhiReciprocal => "1/φ = φ - 1",
    }
}

// Rule 1: Recognize (1 + √5)/2 as φ
// Matches both:
// - Div(Add(1, Sqrt(5)), 2)
// - Mul(1/2, Add(1, Sqrt(5)))
define_rule!(
    RecognizePhiRule,
    "Recognize Phi",
    importance: crate::step::ImportanceLevel::Low,
    |ctx, expr| {
        let rewrite = try_rewrite_recognize_phi_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(format_constant_rewrite_desc(rewrite.kind)))
    }
);

// Rule 2: φ² → φ + 1 (from characteristic equation φ² - φ - 1 = 0)
define_rule!(
    PhiSquaredRule,
    "Phi Squared",
    importance: crate::step::ImportanceLevel::Medium,
    |ctx, expr| {
        let rewrite = try_rewrite_phi_squared_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(format_constant_rewrite_desc(rewrite.kind)))
    }
);

// Rule 3: 1/φ → φ - 1 (reciprocal identity)
define_rule!(
    PhiReciprocalRule,
    "Phi Reciprocal",
    importance: crate::step::ImportanceLevel::Medium,
    |ctx, expr| {
        let rewrite = try_rewrite_phi_reciprocal_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(format_constant_rewrite_desc(rewrite.kind)))
    }
);

// Note: is_one checks now route directly to cas_math::expr_predicates::is_one_expr.

pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(RecognizePhiRule));
    simplifier.add_rule(Box::new(PhiSquaredRule));
    simplifier.add_rule(Box::new(PhiReciprocalRule));
}
