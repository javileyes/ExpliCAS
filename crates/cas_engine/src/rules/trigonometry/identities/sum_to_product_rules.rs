//! Sum-to-product and dyadic cosine product transformations.

use crate::define_rule;
use crate::rule::Rewrite;
use crate::rules::trigonometry::{evaluation, pythagorean, pythagorean_secondary};
use cas_ast::{BuiltinFn, Expr, ExprId};
use cas_math::expr_rewrite::smart_mul;
use cas_math::trig_multi_angle_support::{
    collect_trig_args_recursive, expand_trig_angle, is_double_angle_relation,
    verify_dyadic_pi_sequence,
};
use num_traits::One;

// Import rules from parent module (still defined in include!() files)
use super::{
    AngleIdentityRule, AngleSumFractionToTanRule, Cos2xAdditiveContractionRule,
    CotHalfAngleDifferenceRule, CscCotPythagoreanRule, DoubleAngleContractionRule, DoubleAngleRule,
    HalfAngleTangentRule, HyperbolicTanhPythRule, PythagoreanIdentityRule, QuintupleAngleRule,
    RecursiveTrigExpansionRule, SecTanPythagoreanRule, Sin4xIdentityZeroRule, SinCosIntegerPiRule,
    SinCosQuarticSumRule, SinCosSumQuotientRule, SinSupplementaryAngleRule,
    TanDifferenceIdentityZeroRule, TanDifferenceRule, TanDoubleAngleContractionRule,
    TanToSinCosRule, TanTripleProductRule, TrigHiddenCubicIdentityRule, TrigOddEvenParityRule,
    TrigQuotientRule, TrigSumToProductRule, TripleAngleRule, WeierstrassContractionRule,
    WeierstrassCosIdentityZeroRule, WeierstrassSinIdentityZeroRule,
};
// Import migration Phase 1-3 rules
use super::{GeneralizedSinCosContractionRule, HyperbolicHalfAngleSquaresRule, TrigPhaseShiftRule};

// =============================================================================
// DyadicCosProductToSinRule: 2^n · ∏_{k=0}^{n-1} cos(2^k·θ) → sin(2^n·θ)/sin(θ)
// =============================================================================
//
// This identity simplifies products like:
// - 2·cos(θ) → sin(2θ)/sin(θ)
// - 4·cos(θ)·cos(2θ) → sin(4θ)/sin(θ)
// - 8·cos(θ)·cos(2θ)·cos(4θ) → sin(8θ)/sin(θ)
//
// Combined with sin supplementary angle (sin(π-x)=sin(x)) and cancellation,
// this solves problems like: 8·cos(π/9)·cos(2π/9)·cos(4π/9) = 1.
//
// Domain: Requires sin(θ) ≠ 0. In Generic mode, this is only allowed when
// θ is a rational multiple of π that is NOT an integer (proven case).

/// DyadicCosProductToSinRule: Recognizes products 2^n · ∏cos(2^k·θ)
pub struct DyadicCosProductToSinRule;

impl crate::rule::Rule for DyadicCosProductToSinRule {
    fn name(&self) -> &str {
        "Dyadic Cos Product"
    }

    fn priority(&self) -> i32 {
        // Run BEFORE DoubleAngleRule and RecursiveTrigExpansionRule
        95
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<crate::rule::Rewrite> {
        use crate::rule::Rewrite;
        use cas_math::numeric::as_number;
        use cas_math::pi_helpers::is_provably_sin_nonzero;
        use num_bigint::BigInt;
        use num_rational::BigRational;

        // Only match Mul expressions
        if !matches!(ctx.get(expr), Expr::Mul(_, _)) {
            return None;
        }

        // Flatten the multiplication
        let factors = crate::nary::mul_leaves(ctx, expr);

        // Separate numeric coefficient from cos factors
        let mut numeric_coeff = BigRational::one();
        let mut cos_args: Vec<ExprId> = Vec::new();
        let mut other_factors: Vec<ExprId> = Vec::new();

        for &factor in &factors {
            if let Some(n) = as_number(ctx, factor) {
                numeric_coeff *= n.clone();
            } else if let Expr::Function(fn_id, args) = ctx.get(factor) {
                if matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Cos)) && args.len() == 1 {
                    cos_args.push(args[0]);
                } else {
                    other_factors.push(factor);
                }
            } else {
                other_factors.push(factor);
            }
        }

        // Must have no other factors and at least 1 cos
        if !other_factors.is_empty() || cos_args.is_empty() {
            return None;
        }

        let n = cos_args.len() as u32;

        // Numeric coefficient must be exactly 2^n
        let expected_coeff = BigRational::from_integer(BigInt::from(1u64 << n));
        if numeric_coeff != expected_coeff {
            return None;
        }

        // Find θ by trying each cos_arg as base and verifying dyadic sequence
        let mut theta: Option<ExprId> = None;

        for candidate in &cos_args {
            if verify_dyadic_pi_sequence(ctx, *candidate, &cos_args) {
                theta = Some(*candidate);
                break;
            }
        }

        let theta = theta?;

        // Domain check: sin(θ) ≠ 0
        let domain_mode = parent_ctx.domain_mode();

        if !is_provably_sin_nonzero(ctx, theta) {
            match domain_mode {
                crate::domain::DomainMode::Generic | crate::domain::DomainMode::Strict => {
                    // Block with hint
                    let sin_theta = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![theta]);
                    crate::domain::register_blocked_hint(crate::domain::BlockedHint {
                        rule: "Dyadic Cos Product".to_string(),
                        expr_id: sin_theta,
                        key: crate::assumptions::AssumptionKey::nonzero_key(ctx, sin_theta),
                        suggestion: "use `domain assume` to allow this transformation",
                    });
                    return None;
                }
                crate::domain::DomainMode::Assume => {
                    // Allow but will record assumption in result
                }
            }
        }

        // Build result: sin(2^n · θ) / sin(θ)
        let two_pow_n = ctx.num((1u64 << n) as i64);
        let scaled_theta = smart_mul(ctx, two_pow_n, theta);
        let sin_scaled = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![scaled_theta]);
        let sin_theta = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![theta]);
        let result = ctx.add(Expr::Div(sin_scaled, sin_theta));

        // Build description
        let desc = format!("2^{n}·∏cos(2^k·θ) = sin(2^{n}·θ)/sin(θ)", n = n);

        let mut rewrite = Rewrite::new(result).desc(desc).local(expr, result);

        // Add assumption if in Assume mode and sin(θ)≠0 not proven
        if domain_mode == crate::domain::DomainMode::Assume && !is_provably_sin_nonzero(ctx, theta)
        {
            // Create NonZero assumption with HeuristicAssumption kind
            let mut event = crate::assumptions::AssumptionEvent::nonzero(ctx, sin_theta);
            event.kind = crate::assumptions::AssumptionKind::HeuristicAssumption;
            rewrite = rewrite.assume(event);
        }

        Some(rewrite)
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::MUL)
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::High
    }

    fn soundness(&self) -> crate::rule::SoundnessLabel {
        // The transformation requires sin(θ) ≠ 0, which is either proven or assumed
        crate::rule::SoundnessLabel::EquivalenceUnderIntroducedRequires
    }
}

pub fn register(simplifier: &mut crate::Simplifier) {
    // ABSOLUTE FIRST: Identity Zero Rules that must run before ANY expansion
    // These recognize exact identity forms and short-circuit to 0
    simplifier.add_rule(Box::new(Sin4xIdentityZeroRule));
    simplifier.add_rule(Box::new(TanDifferenceIdentityZeroRule));

    // PRE-ORDER: Evaluate sin(n·π) = 0 and cos(n·π) = (-1)^n BEFORE any expansion
    // This prevents unnecessary triple/double angle expansions on integer multiples of π
    simplifier.add_rule(Box::new(SinCosIntegerPiRule));

    // PRE-ORDER: Trig parity (odd/even functions)
    // sin(-u) = -sin(u), cos(-u) = cos(u), tan(-u) = -tan(u)
    simplifier.add_rule(Box::new(TrigOddEvenParityRule));

    // NOTE: TrigQuotientToNamedRule is defined but NOT registered.
    // It causes numeric-only regression due to conflict with SecToRecipCosRule (ping-pong).
    // TanDoubleAngleContractionRule is now enabled BEFORE TanToSinCosRule so contraction
    // has priority over expansion: 2·tan(t)/(1-tan²(t)) → tan(2t) directly.
    simplifier.add_rule(Box::new(TanDoubleAngleContractionRule));

    // Use the new data-driven EvaluateTrigTableRule instead of deprecated EvaluateTrigRule
    simplifier.add_rule(Box::new(evaluation::EvaluateTrigTableRule));
    simplifier.add_rule(Box::new(PythagoreanIdentityRule));
    simplifier.add_rule(Box::new(SecTanPythagoreanRule));
    simplifier.add_rule(Box::new(CscCotPythagoreanRule));

    // Hidden Cubic Identity: sin^6 + cos^6 + 3sin^2cos^2 = (sin^2+cos^2)^3
    // Should run in TRANSFORM phase before power expansions
    simplifier.add_rule(Box::new(TrigHiddenCubicIdentityRule));

    // Quartic Pythagorean: sin^4(x) + cos^4(x) = 1 − 2·sin²(x)·cos²(x)
    simplifier.add_rule(Box::new(SinCosQuarticSumRule));

    simplifier.add_rule(Box::new(AngleIdentityRule));
    // Triple tangent product: tan(u)·tan(π/3+u)·tan(π/3-u) → tan(3u)
    // Must run BEFORE TanToSinCosRule to prevent expansion
    simplifier.add_rule(Box::new(TanTripleProductRule));
    // Weierstrass Identity Zero Rules: MUST run BEFORE TanToSinCosRule
    // Pattern-driven cancellation: sin(x) - 2t/(1+t²) → 0, cos(x) - (1-t²)/(1+t²) → 0
    simplifier.add_rule(Box::new(WeierstrassSinIdentityZeroRule));
    simplifier.add_rule(Box::new(WeierstrassCosIdentityZeroRule));
    simplifier.add_rule(Box::new(TanToSinCosRule));
    // Dyadic Cos Product: 2^n·∏cos(2^k·θ) → sin(2^n·θ)/sin(θ)
    // Must run BEFORE DoubleAngleRule to recognize the pattern
    simplifier.add_rule(Box::new(DyadicCosProductToSinRule));
    simplifier.add_rule(Box::new(DoubleAngleRule));
    simplifier.add_rule(Box::new(DoubleAngleContractionRule)); // 2sin·cos→sin(2t), cos²-sin²→cos(2t)
    simplifier.add_rule(Box::new(AngleSumFractionToTanRule)); // sin(a)cos(b)±cos(a)sin(b) / cos·cos∓sin·sin → tan(a±b)
                                                              // Cos2xAdditiveContractionRule: 1-2sin²→cos(2t), 2cos²-1→cos(2t).
                                                              // Previously disabled due to +153 numeric-only regression. Re-enabled under Fix #5
                                                              // (ExtractCommonMulFactorRule → POST phase separation) which shifts the NF landscape
                                                              // and may benefit from tighter cos(2t) normal forms for cross-product trig expressions.
    simplifier.add_rule(Box::new(Cos2xAdditiveContractionRule));
    simplifier.add_rule(Box::new(SinCosSumQuotientRule));
    // Standalone Sum-to-Product: sin(A)+sin(B), cos(A)+cos(B) etc. when args are k*π
    simplifier.add_rule(Box::new(TrigSumToProductRule));
    simplifier.add_rule(Box::new(TripleAngleRule)); // Shortcut: sin(3x), cos(3x), tan(3x)
    simplifier.add_rule(Box::new(super::TripleAngleContractionRule)); // Reverse: 3sin(θ)−4sin³(θ) → sin(3θ)
    simplifier.add_rule(Box::new(QuintupleAngleRule)); // Shortcut: sin(5x), cos(5x)
    simplifier.add_rule(Box::new(RecursiveTrigExpansionRule));
    // Trig Quotient: sin(x)/cos(x) → tan(x) - runs after sum-to-product
    simplifier.add_rule(Box::new(TrigQuotientRule));
    // DISABLED: TrigSumToProductContractionRule conflicts with SinCosSumQuotientRule
    // The quotient rule (sin(a)+sin(b))/(cos(a)+cos(b)) → tan((a+b)/2) is more didactic
    // and should be preferred. This rule is defined but NOT registered.
    // simplifier.add_rule(Box::new(TrigSumToProductContractionRule));
    // Half-Angle Tangent: (1-cos(2x))/sin(2x) → tan(x), sin(2x)/(1+cos(2x)) → tan(x)
    simplifier.add_rule(Box::new(HalfAngleTangentRule));
    // Weierstrass Contraction: 2*tan(x/2)/(1+tan²) → sin(x), (1-tan²)/(1+tan²) → cos(x)
    simplifier.add_rule(Box::new(WeierstrassContractionRule));

    // DISABLED: ProductToSumRule conflicts with AngleIdentityRule creating infinite loops
    // ProductToSumRule: 2*sin(a)*cos(b) → sin(a+b) + sin(a-b)
    // AngleIdentityRule: sin(a+b) → sin(a)*cos(b) + cos(a)*sin(b)
    // When combined, they create cycles. Use manually for specific cases like Dirichlet kernel.
    // simplifier.add_rule(Box::new(ProductToSumRule));

    // DISABLED: Conflicts with Pythagorean identity rules causing infinite loops
    // This rule converts cos²(x) → 1-sin²(x) which interacts badly with:
    // - Pythagorean identities (sec²-tan²=1)
    // - Reciprocal trig canonicalization
    // Creating transformation cycles like: sec² → 1/cos² → 1/(1-sin²) → ...
    // See: debug_sec_tan.rs test and GitHub issue #X
    // simplifier.add_rule(Box::new(CanonicalizeTrigSquareRule));

    // Pythagorean Identity simplification: k - k*sin² → k*cos², k - k*cos² → k*sin²
    // This rule was extracted from CancelCommonFactorsRule for pedagogical clarity
    simplifier.add_rule(Box::new(pythagorean::TrigPythagoreanSimplifyRule));
    // N-ary Pythagorean: sin²(t) + cos²(t) → 1 in chains of any length
    simplifier.add_rule(Box::new(pythagorean::TrigPythagoreanChainRule));
    // Generic coefficient Pythagorean: A*sin²(x) + A*cos²(x) → A for any expression A
    simplifier.add_rule(Box::new(pythagorean::TrigPythagoreanGenericCoefficientRule));
    // Linear fold: a·sin²(t) + b·cos²(t) + c → (a-b)·sin²(t) + (b+c) using sin²+cos²=1
    simplifier.add_rule(Box::new(pythagorean::TrigPythagoreanLinearFoldRule));
    // Local collect fold: k·R·sin²(t) + R·cos²(t) - R → (k-1)·R·sin²(t) for residual R
    simplifier.add_rule(Box::new(pythagorean::TrigPythagoreanLocalCollectFoldRule));
    // High-power factor: R − R·trig²(x) → R·other²(x) when trig² is embedded in trig^n
    simplifier.add_rule(Box::new(pythagorean::TrigPythagoreanHighPowerRule));
    // Contraction: 1 + tan²(x) → sec²(x), 1 + cot²(x) → csc²(x)
    simplifier.add_rule(Box::new(pythagorean::RecognizeSecSquaredRule));
    simplifier.add_rule(Box::new(pythagorean::RecognizeCscSquaredRule));
    // Expansion: sec(x) → 1/cos(x), csc(x) → 1/sin(x) for canonical unification
    simplifier.add_rule(Box::new(pythagorean_secondary::SecToRecipCosRule));
    simplifier.add_rule(Box::new(pythagorean_secondary::CscToRecipSinRule));
    // Expansion: cot(x) → cos(x)/sin(x) for canonical unification
    simplifier.add_rule(Box::new(pythagorean_secondary::CotToCosSinRule));

    simplifier.add_rule(Box::new(AngleConsistencyRule));

    // Phase shift: sin(x + π/2) → cos(x), cos(x + π/2) → -sin(x), etc.
    simplifier.add_rule(Box::new(TrigPhaseShiftRule));

    // Supplementary angle: sin(8π/9) = sin(π - π/9) = sin(π/9)
    simplifier.add_rule(Box::new(SinSupplementaryAngleRule));

    // Fourth power difference: sin⁴(x) - cos⁴(x) → sin²(x) - cos²(x)
    simplifier.add_rule(Box::new(pythagorean_secondary::TrigEvenPowerDifferenceRule));

    // Fourth power sum: k·sin⁴(x) + k·cos⁴(x) → k·(1 - 2·sin²(x)·cos²(x))
    simplifier.add_rule(Box::new(pythagorean_secondary::TrigEvenPowerSumRule));

    // Cotangent half-angle difference: cot(u/2) - cot(u) = 1/sin(u)
    simplifier.add_rule(Box::new(CotHalfAngleDifferenceRule));
    // Tangent difference: tan(a-b) → (tan(a)-tan(b))/(1+tan(a)*tan(b))
    simplifier.add_rule(Box::new(TanDifferenceRule));
    // Hyperbolic Pythagorean: 1 - tanh²(x) → 1/cosh²(x)
    simplifier.add_rule(Box::new(HyperbolicTanhPythRule));
    // Hyperbolic half-angle: cosh²(x/2), sinh²(x/2) → cosh form
    simplifier.add_rule(Box::new(HyperbolicHalfAngleSquaresRule));
    // Trig half-angle squares: sin²(x/2) → (1-cos x)/2, cos²(x/2) → (1+cos x)/2
    simplifier.add_rule(Box::new(super::TrigHalfAngleSquaresRule));
    // Generalized sin*cos contraction: k*sin(t)*cos(t) → (k/2)*sin(2t) for even k≥4
    simplifier.add_rule(Box::new(GeneralizedSinCosContractionRule));
}

define_rule!(
    AngleConsistencyRule,
    "Angle Consistency (Half-Angle)",
    |ctx, expr| {
        // Only run on Add/Sub/Mul/Div to capture context
        match ctx.get(expr) {
            Expr::Add(_, _) | Expr::Sub(_, _) | Expr::Mul(_, _) | Expr::Div(_, _) => {}
            _ => return None,
        }

        // 1. Collect all trig arguments
        let mut trig_args = Vec::new();
        collect_trig_args_recursive(ctx, expr, &mut trig_args);

        if trig_args.is_empty() {
            return None;
        }

        // 2. Check for half-angle relationship
        // We look for pair (A, B) such that A = 2*B.
        // Then we expand trig(A) into trig(B).

        let mut target_expansion: Option<(ExprId, ExprId)> = None; // (A, B) where A=2B

        for i in 0..trig_args.len() {
            for j in 0..trig_args.len() {
                if i == j {
                    continue;
                }
                let a = trig_args[i];
                let b = trig_args[j];

                if is_double_angle_relation(ctx, a, b) {
                    target_expansion = Some((a, b));
                    break;
                }
            }
            if target_expansion.is_some() {
                break;
            }
        }

        if let Some((large_angle, small_angle)) = target_expansion {
            // Expand all occurrences of trig(large_angle) in expr
            // We need a recursive replacement helper
            let new_expr = expand_trig_angle(ctx, expr, large_angle, small_angle);
            if new_expr != expr {
                return Some(Rewrite::new(new_expr).desc("Half-Angle Expansion"));
            }
        }

        None
    }
);
