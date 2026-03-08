use crate::define_rule;
use crate::rule::Rewrite;
use cas_math::trig_canonicalization_support::{
    is_trig_of_inverse_trig, try_rewrite_cot_to_csc_pythagorean_identity_expr,
    try_rewrite_csc_cot_minus_one_identity_zero_expr,
    try_rewrite_csc_cot_pythagorean_identity_expr, try_rewrite_mixed_fraction_to_sincos_plan_expr,
    try_rewrite_reciprocal_product_identity_expr, try_rewrite_sec_tan_minus_one_identity_zero_expr,
    try_rewrite_sec_tan_pythagorean_identity_expr,
    try_rewrite_tan_to_sec_pythagorean_identity_expr,
    try_rewrite_trig_function_name_canonicalization_expr, TrigCanonicalIdentityKind,
};

fn format_trig_canonical_identity_desc(kind: TrigCanonicalIdentityKind) -> &'static str {
    match kind {
        TrigCanonicalIdentityKind::SecToRecipCos => "sec(x) = 1/cos(x)",
        TrigCanonicalIdentityKind::CscToRecipSin => "csc(x) = 1/sin(x)",
        TrigCanonicalIdentityKind::CotToCosSin => "cot(x) = cos(x)/sin(x)",
        TrigCanonicalIdentityKind::SecTanPythagorean => "sec²(x) - tan²(x) = 1",
        TrigCanonicalIdentityKind::CscCotPythagorean => "csc²(x) - cot²(x) = 1",
        TrigCanonicalIdentityKind::TanToSecPythagorean => "1 + tan²(x) = sec²(x)",
        TrigCanonicalIdentityKind::CotToCscPythagorean => "1 + cot²(x) = csc²(x)",
        TrigCanonicalIdentityKind::SecTanMinusOneIdentityZero => "sec²(x) - tan²(x) - 1 = 0",
        TrigCanonicalIdentityKind::CscCotMinusOneIdentityZero => "csc²(x) - cot²(x) - 1 = 0",
        TrigCanonicalIdentityKind::ReciprocalProductIdentity => "Reciprocal trig product = 1",
        TrigCanonicalIdentityKind::MixedFractionToSinCos => {
            "Convert mixed trig fraction to sin/cos"
        }
    }
}

// ==================== Sophisticated Context-Aware Canonicalization ====================
// STRATEGY: Only convert when it demonstrably helps simplification
// Three-tier approach:
// 1. Never convert: compositions like tan(arctan(x))
// 2. Always convert: known patterns like sec²-tan², mixed fractions
// 3. Selective: on-demand for complex cases

// ============================== Function Name Canonicalization ==============================

// Canonicalize trig function names: asin→arcsin, acos→arccos, atan→arctan
// This prevents bugs from mixed naming like "arccos(x) - acos(x)" not simplifying
define_rule!(
    TrigFunctionNameCanonicalizationRule,
    "Canonicalize Trig Function Names",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    crate::phase::PhaseMask::CORE | crate::phase::PhaseMask::POST,
    importance: crate::step::ImportanceLevel::Low,
    |ctx, expr| {
        let plan = try_rewrite_trig_function_name_canonicalization_expr(ctx, expr)?;
        Some(Rewrite::new(plan.rewritten).desc(plan.desc))
    }
);

// ==================== Tier 1: Preserve Compositions (Negative Rule) ====================

// NEVER convert reciprocal trig if it's a composition with inverse trig
// This preserves tan(arctan(x)) → x simplifications
// Priority: HIGHEST (register first)
define_rule!(
    PreserveCompositionRule,
    "Preserve trig-inverse compositions",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    |ctx, expr| {
        // If this is tan(arctan(...)), cot(arcsin(...)), etc.
        // return None to prevent any conversion
        if is_trig_of_inverse_trig(ctx, expr) {
            // Explicitly return None - this is a "negative rule"
            // It blocks other rules from converting
            return None;
        }
        None
    }
);

// =================================================================================
// Direct Pythagorean Identity Rules (No Conversion)
// =================================================================================
// Instead of converting to sin/cos (which creates complex intermediate forms),
// directly apply the Pythagorean identities:
// - sec²(x) - tan²(x) = 1
// - csc²(x) - cot²(x) = 1
// - 1 + tan²(x) = sec²(x)
// - 1 + cot²(x) = csc²(x)

// sec²(x) - tan²(x) → 1
define_rule!(
    SecTanPythagoreanRule,
    "sec²(x) - tan²(x) = 1",
    Some(crate::target_kind::TargetKindSet::SUB),
    |ctx, expr| {
        let rewrite = try_rewrite_sec_tan_pythagorean_identity_expr(ctx, expr)?;
        Some(
            Rewrite::new(rewrite.rewritten).desc(format_trig_canonical_identity_desc(rewrite.kind)),
        )
    }
);

// csc²(x) - cot²(x) → 1
define_rule!(
    CscCotPythagoreanRule,
    "csc²(x) - cot²(x) = 1",
    Some(crate::target_kind::TargetKindSet::SUB),
    |ctx, expr| {
        let rewrite = try_rewrite_csc_cot_pythagorean_identity_expr(ctx, expr)?;
        Some(
            Rewrite::new(rewrite.rewritten).desc(format_trig_canonical_identity_desc(rewrite.kind)),
        )
    }
);

// 1 + tan²(x) → sec²(x)
define_rule!(
    TanToSecPythagoreanRule,
    "1 + tan²(x) = sec²(x)",
    Some(crate::target_kind::TargetKindSet::ADD),
    |ctx, expr| {
        let rewrite = try_rewrite_tan_to_sec_pythagorean_identity_expr(ctx, expr)?;
        Some(
            Rewrite::new(rewrite.rewritten).desc(format_trig_canonical_identity_desc(rewrite.kind)),
        )
    }
);

// 1 + cot²(x) → csc²(x)
define_rule!(
    CotToCscPythagoreanRule,
    "1 + cot²(x) = csc²(x)",
    Some(crate::target_kind::TargetKindSet::ADD),
    |ctx, expr| {
        let rewrite = try_rewrite_cot_to_csc_pythagorean_identity_expr(ctx, expr)?;
        Some(
            Rewrite::new(rewrite.rewritten).desc(format_trig_canonical_identity_desc(rewrite.kind)),
        )
    }
);

// ==================== Pythagorean Identity Variants with Constants ====================

// sec²(x) - tan²(x) - 1 → 0
// This handles the variant where we have the full identity minus 1
define_rule!(
    SecTanMinusOneIdentityRule,
    "sec²(x) - tan²(x) - 1 = 0",
    Some(crate::target_kind::TargetKindSet::SUB),
    |ctx, expr| {
        let rewrite = try_rewrite_sec_tan_minus_one_identity_zero_expr(ctx, expr)?;
        Some(
            Rewrite::new(rewrite.rewritten).desc(format_trig_canonical_identity_desc(rewrite.kind)),
        )
    }
);

// csc²(x) - cot²(x) - 1 → 0
define_rule!(
    CscCotMinusOneIdentityRule,
    "csc²(x) - cot²(x) - 1 = 0",
    Some(crate::target_kind::TargetKindSet::SUB),
    |ctx, expr| {
        let rewrite = try_rewrite_csc_cot_minus_one_identity_zero_expr(ctx, expr)?;
        Some(
            Rewrite::new(rewrite.rewritten).desc(format_trig_canonical_identity_desc(rewrite.kind)),
        )
    }
);

// Convert reciprocal products like tan(x)*cot(x) → 1
define_rule!(
    ConvertReciprocalProductRule,
    "Simplify reciprocal trig products",
    Some(crate::target_kind::TargetKindSet::MUL),
    |ctx, expr| {
        let rewrite = try_rewrite_reciprocal_product_identity_expr(ctx, expr)?;
        Some(
            Rewrite::new(rewrite.rewritten).desc(format_trig_canonical_identity_desc(rewrite.kind)),
        )
    }
);

// ==================== Phase 4: Mixed Fraction Conversion ====================

// Convert mixed trig fractions to sin/cos for better algebraic simplification
define_rule!(
    ConvertForMixedFractionRule,
    "Convert Mixed Trig Fraction to sin/cos",
    Some(crate::target_kind::TargetKindSet::DIV),
    |ctx, expr| {
        let rewrite = try_rewrite_mixed_fraction_to_sincos_plan_expr(ctx, expr)?;
        Some(
            Rewrite::new(rewrite.rewritten).desc(format_trig_canonical_identity_desc(rewrite.kind)),
        )
    }
);

// ==================== Registration ====================

// Register ONLY direct Pythagorean identity rules
// CRITICAL: Must be called BEFORE any conversion rules to preserve patterns
// sec²-tan²-1 must match BEFORE tan² becomes sin²/cos²
pub fn register_pythagorean_identities(simplifier: &mut crate::engine::Simplifier) {
    // These are the HIGHEST PRIORITY rules that must fire first
    simplifier.add_rule(Box::new(SecTanPythagoreanRule));
    simplifier.add_rule(Box::new(CscCotPythagoreanRule));
    simplifier.add_rule(Box::new(TanToSecPythagoreanRule));
    simplifier.add_rule(Box::new(CotToCscPythagoreanRule));

    // Pythagorean variants with constants
    simplifier.add_rule(Box::new(SecTanMinusOneIdentityRule));
    simplifier.add_rule(Box::new(CscCotMinusOneIdentityRule));
}

// Register sophisticated canonicalization rules
// CRITICAL: These rules are applied AFTER compositions resolve
// so that tan(arctan(x)) → x happens before any conversion attempts
pub fn register(simplifier: &mut crate::engine::Simplifier) {
    // Function name canonicalization - MUST run first
    simplifier.add_rule(Box::new(TrigFunctionNameCanonicalizationRule));

    // Reciprocal product simplification
    simplifier.add_rule(Box::new(ConvertReciprocalProductRule));

    // Mixed fraction conversion
    simplifier.add_rule(Box::new(ConvertForMixedFractionRule));
}
