//! Support for power identity rewrites that are always sound.

use crate::expr_destructure::as_pow;
use crate::tri_proof::TriProof;
use cas_ast::{Context, Expr, ExprId};
use num_traits::One;
use num_traits::{Signed, Zero};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowerIdentityPolicyPattern {
    /// `x^0` with optional literal-zero base detection (`0^0`).
    PowZero {
        base: ExprId,
        base_is_literal_zero: bool,
    },
    /// `0^x`; callers should apply domain policy to the exponent.
    ZeroPow {
        exp: ExprId,
        exp_is_numeric_positive: bool,
        exp_is_numeric_non_positive: bool,
    },
}

/// Domain mode abstraction for power-identity policy decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowerIdentityDomainMode {
    Strict,
    Generic,
    Assume,
}

/// Derive [`PowerIdentityDomainMode`] from generic mode flags.
///
/// This helps adapter crates avoid local enum mapping boilerplate.
pub fn power_identity_mode_from_flags(
    assume_mode: bool,
    strict_mode: bool,
) -> PowerIdentityDomainMode {
    if assume_mode {
        PowerIdentityDomainMode::Assume
    } else if strict_mode {
        PowerIdentityDomainMode::Strict
    } else {
        PowerIdentityDomainMode::Generic
    }
}

/// Decision for the `x^0` branch after applying mode/proof policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowZeroPolicyDecision {
    /// `0^0` is undefined.
    RewriteToUndefined,
    /// Rewrite to 1; caller may need to emit assumptions.
    RewriteToOne { assume_nonzero: bool },
    /// Keep expression as-is.
    NoRewrite,
}

/// Planned rewrite action for the `x^0` branch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowZeroPolicyAction {
    /// `0^0` case.
    RewriteToUndefined,
    /// Rewrite to `1`; caller may emit assumptions.
    RewriteToOne { assume_nonzero: bool },
    /// Keep expression as-is.
    NoRewrite,
}

/// Decision for numeric pre-classification of `0^x`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZeroPowNumericPolicyDecision {
    /// Numeric exponent is provably positive (`0^n -> 0`).
    RewriteToZero,
    /// Numeric exponent is non-positive (`0^0`, `0^-1`, ...).
    NoRewrite,
    /// Exponent is symbolic; caller should apply domain/oracle policy.
    NeedsPositiveCondition,
}

/// Planned rewrite action for the `0^x` branch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZeroPowPolicyAction {
    RewriteToZero,
    NoRewrite,
    NeedsPositiveCondition,
}

/// Try always-safe identities:
/// - `x^1 -> x`
/// - `1^x -> 1`
pub fn try_rewrite_pow_one_or_one_pow_expr(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let (base, exp) = as_pow(ctx, expr)?;
    if let Expr::Number(n) = ctx.get(exp) {
        if n.is_one() {
            return Some(base);
        }
    }

    if let Expr::Number(n) = ctx.get(base) {
        if n.is_one() {
            return Some(ctx.num(1));
        }
    }

    None
}

/// Classify domain-sensitive power-identity patterns:
/// - `x^0`
/// - `0^x`
pub fn classify_power_identity_policy_pattern(
    ctx: &Context,
    expr: ExprId,
) -> Option<PowerIdentityPolicyPattern> {
    let (base, exp) = as_pow(ctx, expr)?;

    if let Expr::Number(n) = ctx.get(exp) {
        if n.is_zero() {
            let base_is_literal_zero = matches!(ctx.get(base), Expr::Number(b) if b.is_zero());
            return Some(PowerIdentityPolicyPattern::PowZero {
                base,
                base_is_literal_zero,
            });
        }
    }

    if let Expr::Number(n) = ctx.get(base) {
        if n.is_zero() {
            let (exp_is_numeric_positive, exp_is_numeric_non_positive) = match ctx.get(exp) {
                Expr::Number(e) => (e.is_positive(), !e.is_positive()),
                _ => (false, false),
            };
            return Some(PowerIdentityPolicyPattern::ZeroPow {
                exp,
                exp_is_numeric_positive,
                exp_is_numeric_non_positive,
            });
        }
    }

    None
}

/// Decide whether `x^0` can rewrite to `1` in the given mode.
pub fn decide_pow_zero_policy(
    mode: PowerIdentityDomainMode,
    base_is_literal_zero: bool,
    base_is_numeric_literal: bool,
    base_nonzero_proof: TriProof,
) -> PowZeroPolicyDecision {
    if base_is_literal_zero {
        return PowZeroPolicyDecision::RewriteToUndefined;
    }

    match mode {
        PowerIdentityDomainMode::Generic => PowZeroPolicyDecision::RewriteToOne {
            assume_nonzero: !base_is_numeric_literal,
        },
        PowerIdentityDomainMode::Strict => {
            if matches!(base_nonzero_proof, TriProof::Proven) {
                PowZeroPolicyDecision::RewriteToOne {
                    assume_nonzero: false,
                }
            } else {
                PowZeroPolicyDecision::NoRewrite
            }
        }
        PowerIdentityDomainMode::Assume => PowZeroPolicyDecision::RewriteToOne {
            assume_nonzero: true,
        },
    }
}

/// Plan the full `x^0` rewrite action.
pub fn plan_pow_zero_policy_action(
    mode: PowerIdentityDomainMode,
    base_is_literal_zero: bool,
    base_is_numeric_literal: bool,
    base_nonzero_proof: TriProof,
) -> PowZeroPolicyAction {
    match decide_pow_zero_policy(
        mode,
        base_is_literal_zero,
        base_is_numeric_literal,
        base_nonzero_proof,
    ) {
        PowZeroPolicyDecision::RewriteToUndefined => PowZeroPolicyAction::RewriteToUndefined,
        PowZeroPolicyDecision::RewriteToOne { assume_nonzero } => {
            PowZeroPolicyAction::RewriteToOne { assume_nonzero }
        }
        PowZeroPolicyDecision::NoRewrite => PowZeroPolicyAction::NoRewrite,
    }
}

/// Plan `x^0` policy action from generic mode flags.
pub fn plan_pow_zero_policy_action_with_mode_flags(
    assume_mode: bool,
    strict_mode: bool,
    base_is_literal_zero: bool,
    base_is_numeric_literal: bool,
    base_nonzero_proof: TriProof,
) -> PowZeroPolicyAction {
    let mode = power_identity_mode_from_flags(assume_mode, strict_mode);
    plan_pow_zero_policy_action(
        mode,
        base_is_literal_zero,
        base_is_numeric_literal,
        base_nonzero_proof,
    )
}

/// Decide the numeric fast-path for `0^x`.
pub fn decide_zero_pow_numeric_policy(
    exp_is_numeric_positive: bool,
    exp_is_numeric_non_positive: bool,
) -> ZeroPowNumericPolicyDecision {
    if exp_is_numeric_positive {
        return ZeroPowNumericPolicyDecision::RewriteToZero;
    }
    if exp_is_numeric_non_positive {
        return ZeroPowNumericPolicyDecision::NoRewrite;
    }
    ZeroPowNumericPolicyDecision::NeedsPositiveCondition
}

/// Plan the numeric pre-action for `0^x`.
pub fn plan_zero_pow_policy_action(
    exp_is_numeric_positive: bool,
    exp_is_numeric_non_positive: bool,
) -> ZeroPowPolicyAction {
    match decide_zero_pow_numeric_policy(exp_is_numeric_positive, exp_is_numeric_non_positive) {
        ZeroPowNumericPolicyDecision::RewriteToZero => ZeroPowPolicyAction::RewriteToZero,
        ZeroPowNumericPolicyDecision::NoRewrite => ZeroPowPolicyAction::NoRewrite,
        ZeroPowNumericPolicyDecision::NeedsPositiveCondition => {
            ZeroPowPolicyAction::NeedsPositiveCondition
        }
    }
}

#[cfg(test)]
mod tests {
    use super::PowerIdentityPolicyPattern;
    use super::{classify_power_identity_policy_pattern, try_rewrite_pow_one_or_one_pow_expr};
    use super::{
        decide_pow_zero_policy, decide_zero_pow_numeric_policy, plan_pow_zero_policy_action,
        plan_pow_zero_policy_action_with_mode_flags, plan_zero_pow_policy_action,
        power_identity_mode_from_flags, PowZeroPolicyAction, PowZeroPolicyDecision,
        PowerIdentityDomainMode, ZeroPowNumericPolicyDecision, ZeroPowPolicyAction,
    };
    use crate::tri_proof::TriProof;
    use cas_ast::Context;
    use cas_parser::parse;

    #[test]
    fn rewrites_pow_one() {
        let mut ctx = Context::new();
        let expr = parse("x^1", &mut ctx).expect("parse");
        assert!(try_rewrite_pow_one_or_one_pow_expr(&mut ctx, expr).is_some());
    }

    #[test]
    fn rewrites_one_pow_any() {
        let mut ctx = Context::new();
        let expr = parse("1^x", &mut ctx).expect("parse");
        assert!(try_rewrite_pow_one_or_one_pow_expr(&mut ctx, expr).is_some());
    }

    #[test]
    fn classifies_pow_zero_with_literal_zero_base() {
        let mut ctx = Context::new();
        let expr = parse("0^0", &mut ctx).expect("parse");
        let pattern = classify_power_identity_policy_pattern(&ctx, expr).expect("pattern");
        assert!(matches!(
            pattern,
            PowerIdentityPolicyPattern::PowZero {
                base_is_literal_zero: true,
                ..
            }
        ));
    }

    #[test]
    fn classifies_zero_pow_with_numeric_positive_exp() {
        let mut ctx = Context::new();
        let expr = parse("0^2", &mut ctx).expect("parse");
        let pattern = classify_power_identity_policy_pattern(&ctx, expr).expect("pattern");
        assert!(matches!(
            pattern,
            PowerIdentityPolicyPattern::ZeroPow {
                exp_is_numeric_positive: true,
                ..
            }
        ));
    }

    #[test]
    fn decide_pow_zero_policy_generic_symbolic_requires_assumption() {
        let out = decide_pow_zero_policy(
            PowerIdentityDomainMode::Generic,
            false,
            false,
            TriProof::Unknown,
        );
        assert_eq!(
            out,
            PowZeroPolicyDecision::RewriteToOne {
                assume_nonzero: true
            }
        );
    }

    #[test]
    fn decide_pow_zero_policy_strict_requires_proven_nonzero() {
        let unknown = decide_pow_zero_policy(
            PowerIdentityDomainMode::Strict,
            false,
            false,
            TriProof::Unknown,
        );
        assert_eq!(unknown, PowZeroPolicyDecision::NoRewrite);

        let proven = decide_pow_zero_policy(
            PowerIdentityDomainMode::Strict,
            false,
            false,
            TriProof::Proven,
        );
        assert_eq!(
            proven,
            PowZeroPolicyDecision::RewriteToOne {
                assume_nonzero: false
            }
        );
    }

    #[test]
    fn decide_pow_zero_policy_zero_pow_zero_is_undefined() {
        let out = decide_pow_zero_policy(
            PowerIdentityDomainMode::Assume,
            true,
            true,
            TriProof::Unknown,
        );
        assert_eq!(out, PowZeroPolicyDecision::RewriteToUndefined);
    }

    #[test]
    fn decide_zero_pow_numeric_policy_routes() {
        assert_eq!(
            decide_zero_pow_numeric_policy(true, false),
            ZeroPowNumericPolicyDecision::RewriteToZero
        );
        assert_eq!(
            decide_zero_pow_numeric_policy(false, true),
            ZeroPowNumericPolicyDecision::NoRewrite
        );
        assert_eq!(
            decide_zero_pow_numeric_policy(false, false),
            ZeroPowNumericPolicyDecision::NeedsPositiveCondition
        );
    }

    #[test]
    fn plan_zero_pow_policy_action_maps_numeric_routes() {
        assert_eq!(
            plan_zero_pow_policy_action(true, false),
            ZeroPowPolicyAction::RewriteToZero
        );
        assert_eq!(
            plan_zero_pow_policy_action(false, true),
            ZeroPowPolicyAction::NoRewrite
        );
        assert_eq!(
            plan_zero_pow_policy_action(false, false),
            ZeroPowPolicyAction::NeedsPositiveCondition
        );
    }

    #[test]
    fn plan_pow_zero_policy_action_generic_symbolic_maps_desc_and_assumption() {
        let action = plan_pow_zero_policy_action(
            PowerIdentityDomainMode::Generic,
            false,
            false,
            TriProof::Unknown,
        );
        assert_eq!(
            action,
            PowZeroPolicyAction::RewriteToOne {
                assume_nonzero: true
            }
        );
    }

    #[test]
    fn plan_pow_zero_policy_action_handles_zero_pow_zero() {
        let action = plan_pow_zero_policy_action(
            PowerIdentityDomainMode::Assume,
            true,
            true,
            TriProof::Unknown,
        );
        assert_eq!(action, PowZeroPolicyAction::RewriteToUndefined);
    }

    #[test]
    fn mode_from_flags_prioritizes_assume_then_strict() {
        assert_eq!(
            power_identity_mode_from_flags(true, true),
            PowerIdentityDomainMode::Assume
        );
        assert_eq!(
            power_identity_mode_from_flags(false, true),
            PowerIdentityDomainMode::Strict
        );
        assert_eq!(
            power_identity_mode_from_flags(false, false),
            PowerIdentityDomainMode::Generic
        );
    }

    #[test]
    fn plan_pow_zero_policy_action_with_mode_flags_matches_direct_mode_planning() {
        let direct = plan_pow_zero_policy_action(
            PowerIdentityDomainMode::Strict,
            false,
            true,
            TriProof::Proven,
        );
        let via_flags =
            plan_pow_zero_policy_action_with_mode_flags(false, true, false, true, TriProof::Proven);
        assert_eq!(via_flags, direct);
    }
}
