//! Angle expansion and product-to-sum identities.
//!
//! This module contains rules for:
//! - Product-to-sum: 2·sin(a)·cos(b) → sin(a+b) + sin(a-b)
//! - Trig phase shifts: sin(x + π/2) → cos(x)

use crate::define_rule;
use crate::rule::Rewrite;
use cas_math::trig_phase_shift_support::{
    try_rewrite_trig_phase_shift_function_expr, TrigPhaseShiftFunctionKind, TrigPhaseShiftKind,
};
use cas_math::trig_sum_product_support::{
    try_rewrite_product_to_sum_expr, TrigProductToSumRewriteKind,
};

fn format_product_to_sum_desc(kind: TrigProductToSumRewriteKind) -> &'static str {
    match kind {
        TrigProductToSumRewriteKind::SinCos => "2·sin(a)·cos(b) → sin(a+b) + sin(a-b)",
        TrigProductToSumRewriteKind::CosSin => "2·cos(a)·sin(b) → sin(a+b) - sin(a-b)",
        TrigProductToSumRewriteKind::CosCos => "2·cos(a)·cos(b) → cos(a+b) + cos(a-b)",
        TrigProductToSumRewriteKind::SinSin => "2·sin(a)·sin(b) → cos(a-b) - cos(a+b)",
    }
}

fn format_trig_phase_shift_desc(
    function: TrigPhaseShiftFunctionKind,
    shift: TrigPhaseShiftKind,
) -> &'static str {
    match (function, shift) {
        (TrigPhaseShiftFunctionKind::Sin, TrigPhaseShiftKind::PiOver2) => {
            "sin(x + π/2) phase shift"
        }
        (TrigPhaseShiftFunctionKind::Sin, TrigPhaseShiftKind::NegPiOver2) => {
            "sin(x + -π/2) phase shift"
        }
        (TrigPhaseShiftFunctionKind::Sin, TrigPhaseShiftKind::Pi) => "sin(x + π) phase shift",
        (TrigPhaseShiftFunctionKind::Sin, TrigPhaseShiftKind::NegPi) => "sin(x + -π) phase shift",
        (TrigPhaseShiftFunctionKind::Sin, TrigPhaseShiftKind::ThreePiOver2) => {
            "sin(x + 3π/2) phase shift"
        }
        (TrigPhaseShiftFunctionKind::Sin, TrigPhaseShiftKind::NegThreePiOver2) => {
            "sin(x + -3π/2) phase shift"
        }
        (TrigPhaseShiftFunctionKind::Sin, TrigPhaseShiftKind::KPiOver2) => {
            "sin(x + kπ/2) phase shift"
        }
        (TrigPhaseShiftFunctionKind::Cos, TrigPhaseShiftKind::PiOver2) => {
            "cos(x + π/2) phase shift"
        }
        (TrigPhaseShiftFunctionKind::Cos, TrigPhaseShiftKind::NegPiOver2) => {
            "cos(x + -π/2) phase shift"
        }
        (TrigPhaseShiftFunctionKind::Cos, TrigPhaseShiftKind::Pi) => "cos(x + π) phase shift",
        (TrigPhaseShiftFunctionKind::Cos, TrigPhaseShiftKind::NegPi) => "cos(x + -π) phase shift",
        (TrigPhaseShiftFunctionKind::Cos, TrigPhaseShiftKind::ThreePiOver2) => {
            "cos(x + 3π/2) phase shift"
        }
        (TrigPhaseShiftFunctionKind::Cos, TrigPhaseShiftKind::NegThreePiOver2) => {
            "cos(x + -3π/2) phase shift"
        }
        (TrigPhaseShiftFunctionKind::Cos, TrigPhaseShiftKind::KPiOver2) => {
            "cos(x + kπ/2) phase shift"
        }
    }
}

// =============================================================================
// PRODUCT-TO-SUM IDENTITIES
// =============================================================================
// 2*sin(a)*cos(b) → sin(a+b) + sin(a-b)
// 2*cos(a)*sin(b) → sin(a+b) - sin(a-b)
// 2*cos(a)*cos(b) → cos(a+b) + cos(a-b)
// 2*sin(a)*sin(b) → cos(a-b) - cos(a+b)

define_rule!(ProductToSumRule, "Product to Sum", |ctx, expr| {
    let rewrite = try_rewrite_product_to_sum_expr(ctx, expr)?;
    Some(Rewrite::new(rewrite.rewritten).desc(format_product_to_sum_desc(rewrite.kind)))
});
// ============================================================================
// Trig Phase Shift Rule
// ============================================================================
// sin(x + π/2) → cos(x)
// sin(x - π/2) → -cos(x)
// sin(x + π) → -sin(x)
// cos(x + π/2) → -sin(x)
// cos(x - π/2) → sin(x)
// cos(x + π) → -cos(x)
//
// Also handles canonical form: sin((2*x + π)/2) where arg = (2*x + π)/2

define_rule!(TrigPhaseShiftRule, "Trig Phase Shift", |ctx, expr| {
    let rewrite = try_rewrite_trig_phase_shift_function_expr(ctx, expr)?;
    Some(
        Rewrite::new(rewrite.rewritten).desc(format_trig_phase_shift_desc(
            rewrite.function,
            rewrite.shift,
        )),
    )
});
