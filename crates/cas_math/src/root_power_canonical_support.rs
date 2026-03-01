//! Support for root/power canonical rewrites with injectable proof policies.

use crate::root_forms::extract_square_root_base;
use crate::tri_proof::TriProof;
use crate::{exponents_support::mul_exp, pi_helpers::is_half};
use cas_ast::ordering::compare_expr;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_integer::Integer;
use num_traits::{One, Zero};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RootMergeMulRewrite {
    pub rewritten: ExprId,
    pub left_base: ExprId,
    pub right_base: ExprId,
    pub assume_left_nonnegative: bool,
    pub assume_right_nonnegative: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RootMergeDivRewrite {
    pub rewritten: ExprId,
    pub num_base: ExprId,
    pub den_base: ExprId,
    pub assume_num_nonnegative: bool,
    pub assume_den_positive: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PowPowCancelReciprocalRewrite {
    pub rewritten: ExprId,
    pub base: ExprId,
    pub inner_exp: ExprId,
    pub assume_base_positive: bool,
    pub assume_exp_nonzero: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RootPowCancelKind {
    NumericOdd,
    NumericEven,
    Symbolic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RootPowCancelRewrite {
    pub rewritten: ExprId,
    pub inner_base: ExprId,
    pub inner_exp: ExprId,
    pub kind: RootPowCancelKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RootPowCancelPattern {
    NumericEven {
        rewritten: ExprId,
    },
    NumericOdd {
        rewritten: ExprId,
    },
    SymbolicCandidate {
        rewritten: ExprId,
        inner_base: ExprId,
        inner_exp: ExprId,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PowerPowerEvenRootAbsRewrite {
    pub rewritten: ExprId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PowerPowerAnalysis {
    pub inner_base: ExprId,
    pub inner_exp: ExprId,
    pub outer_exp: ExprId,
    pub multiplied_expr: ExprId,
    pub inner_is_even_root: bool,
    pub symbolic_root_cancel: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowerPowerPattern {
    EvenRootAbs {
        rewritten: ExprId,
    },
    EvenRootNeedsNonNegative {
        rewritten: ExprId,
        inner_base: ExprId,
    },
    SymbolicRootCancelCandidate {
        rewritten: ExprId,
        inner_base: ExprId,
        inner_exp: ExprId,
    },
    MultiplyExponents {
        rewritten: ExprId,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowerPowerEvenRootGuardDecision {
    ProvenRewrite {
        rewritten: ExprId,
    },
    NeedsNonNegativeCondition {
        rewritten: ExprId,
        inner_base: ExprId,
    },
}

/// Planned action for the `(x^a)^b` even-root branch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowerPowerEvenRootAction {
    Apply {
        rewritten: ExprId,
    },
    NeedsNonNegativeCondition {
        rewritten: ExprId,
        inner_base: ExprId,
    },
}

/// Combine an explicit non-negativity proof with an implicit-domain witness.
///
/// If explicit proof is already known (`Proven`/`Disproven`), it is preserved.
/// If explicit proof is `Unknown`, an implicit witness can upgrade it to `Proven`.
pub fn merge_nonnegative_proof_with_witness(
    explicit: TriProof,
    implicit_contains_nonnegative: bool,
    witness_survives: bool,
) -> TriProof {
    if !matches!(explicit, TriProof::Unknown) {
        return explicit;
    }
    if implicit_contains_nonnegative && witness_survives {
        TriProof::Proven
    } else {
        TriProof::Unknown
    }
}

/// Domain mode abstraction for symbolic root-cancel policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolicRootCancelDomainMode {
    Strict,
    Generic,
    Assume,
}

/// Derive symbolic-root-cancel mode from generic mode flags.
pub fn symbolic_root_cancel_mode_from_flags(
    assume_mode: bool,
    strict_mode: bool,
) -> SymbolicRootCancelDomainMode {
    if assume_mode {
        SymbolicRootCancelDomainMode::Assume
    } else if strict_mode {
        SymbolicRootCancelDomainMode::Strict
    } else {
        SymbolicRootCancelDomainMode::Generic
    }
}

/// Policy decision for symbolic `(x^n)^(1/n)` cancellation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolicRootCancelPolicy {
    /// Apply rewrite without extra assumptions (e.g. complex-enabled value domain).
    ApplyUnconditionally,
    /// Apply rewrite but require positivity/non-zero assumptions.
    ApplyWithAssumptions,
    /// Block rewrite and ask caller to surface assume-mode guidance.
    BlockedNeedsAssumeMode,
}

/// Planned action for symbolic root cancellation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolicRootCancelAction {
    ApplyUnconditionally {
        rewritten: ExprId,
    },
    ApplyWithAssumptions {
        rewritten: ExprId,
        inner_base: ExprId,
        inner_exp: ExprId,
    },
    BlockedNeedsAssumeMode {
        inner_base: ExprId,
        inner_exp: ExprId,
    },
}

/// Try `sqrt(a) * sqrt(b) -> sqrt(a*b)`.
pub fn try_rewrite_root_merge_mul_expr_with<FProveNonnegative>(
    ctx: &mut Context,
    expr: ExprId,
    strict_mode: bool,
    mut prove_nonnegative: FProveNonnegative,
) -> Option<RootMergeMulRewrite>
where
    FProveNonnegative: FnMut(&Context, ExprId) -> TriProof,
{
    let (left, right) = match ctx.get(expr) {
        Expr::Mul(l, r) => (*l, *r),
        _ => return None,
    };

    let a = extract_square_root_base(ctx, left)?;
    let b = extract_square_root_base(ctx, right)?;

    let proof_a = prove_nonnegative(ctx, a);
    let proof_b = prove_nonnegative(ctx, b);
    if strict_mode && (!proof_a.is_proven() || !proof_b.is_proven()) {
        return None;
    }

    let product = ctx.add(Expr::Mul(a, b));
    let half = ctx.rational(1, 2);
    let rewritten = ctx.add(Expr::Pow(product, half));

    Some(RootMergeMulRewrite {
        rewritten,
        left_base: a,
        right_base: b,
        assume_left_nonnegative: !proof_a.is_proven(),
        assume_right_nonnegative: !proof_b.is_proven(),
    })
}

/// Try `sqrt(a) / sqrt(b) -> sqrt(a/b)`.
pub fn try_rewrite_root_merge_div_expr_with<FProveNonnegative, FProvePositive>(
    ctx: &mut Context,
    expr: ExprId,
    strict_mode: bool,
    mut prove_nonnegative: FProveNonnegative,
    mut prove_positive: FProvePositive,
) -> Option<RootMergeDivRewrite>
where
    FProveNonnegative: FnMut(&Context, ExprId) -> TriProof,
    FProvePositive: FnMut(&Context, ExprId) -> TriProof,
{
    let (num, den) = match ctx.get(expr) {
        Expr::Div(n, d) => (*n, *d),
        _ => return None,
    };

    let a = extract_square_root_base(ctx, num)?;
    let b = extract_square_root_base(ctx, den)?;

    let proof_a = prove_nonnegative(ctx, a);
    let proof_b = prove_positive(ctx, b);
    if strict_mode && (!proof_a.is_proven() || !proof_b.is_proven()) {
        return None;
    }

    let quotient = ctx.add(Expr::Div(a, b));
    let half = ctx.rational(1, 2);
    let rewritten = ctx.add(Expr::Pow(quotient, half));

    Some(RootMergeDivRewrite {
        rewritten,
        num_base: a,
        den_base: b,
        assume_num_nonnegative: !proof_a.is_proven(),
        assume_den_positive: !proof_b.is_proven(),
    })
}

fn has_reciprocal_outer_exponent(ctx: &Context, outer_exp: ExprId, inner_exp: ExprId) -> bool {
    match ctx.get(outer_exp) {
        Expr::Div(num, den) => {
            let is_one = matches!(ctx.get(*num), Expr::Number(n) if n.is_one());
            is_one && compare_expr(ctx, *den, inner_exp) == std::cmp::Ordering::Equal
        }
        Expr::Number(outer_n) => {
            let outer_n = outer_n.clone();
            if let Expr::Number(y_n) = ctx.get(inner_exp) {
                !y_n.is_zero() && outer_n == y_n.recip()
            } else {
                false
            }
        }
        _ => false,
    }
}

/// Try `(x^n)^(1/n) -> x` (odd `n`), `|x|` (even `n`) or symbolic cancel candidate.
///
/// This function is policy-free: it does not apply domain-mode assumptions or
/// blocked hints. For symbolic `n`, callers should decide whether cancellation
/// is allowed in the active domain policy.
pub fn try_rewrite_root_pow_cancel_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<RootPowCancelRewrite> {
    let (base, outer_exp) = match ctx.get(expr) {
        Expr::Pow(base, exp) => (*base, *exp),
        _ => return None,
    };
    let (inner_base, inner_exp) = match ctx.get(base) {
        Expr::Pow(inner_base, inner_exp) => (*inner_base, *inner_exp),
        _ => return None,
    };

    let combined_is_one = match (ctx.get(outer_exp), ctx.get(inner_exp)) {
        (Expr::Number(o), Expr::Number(i)) => (o * i).is_one(),
        _ => has_reciprocal_outer_exponent(ctx, outer_exp, inner_exp),
    };
    if !combined_is_one {
        return None;
    }

    if let Expr::Number(n) = ctx.get(inner_exp) {
        if !n.is_integer() {
            return None;
        }
        if n.to_integer().is_even() {
            let rewritten = ctx.call_builtin(BuiltinFn::Abs, vec![inner_base]);
            return Some(RootPowCancelRewrite {
                rewritten,
                inner_base,
                inner_exp,
                kind: RootPowCancelKind::NumericEven,
            });
        }
        return Some(RootPowCancelRewrite {
            rewritten: inner_base,
            inner_base,
            inner_exp,
            kind: RootPowCancelKind::NumericOdd,
        });
    }

    Some(RootPowCancelRewrite {
        rewritten: inner_base,
        inner_base,
        inner_exp,
        kind: RootPowCancelKind::Symbolic,
    })
}

/// Classify root-power cancellation pattern for `(x^n)^(1/n)`.
pub fn classify_root_pow_cancel_pattern(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<RootPowCancelPattern> {
    let rewrite = try_rewrite_root_pow_cancel_expr(ctx, expr)?;
    match rewrite.kind {
        RootPowCancelKind::NumericEven => Some(RootPowCancelPattern::NumericEven {
            rewritten: rewrite.rewritten,
        }),
        RootPowCancelKind::NumericOdd => Some(RootPowCancelPattern::NumericOdd {
            rewritten: rewrite.rewritten,
        }),
        RootPowCancelKind::Symbolic => Some(RootPowCancelPattern::SymbolicCandidate {
            rewritten: rewrite.rewritten,
            inner_base: rewrite.inner_base,
            inner_exp: rewrite.inner_exp,
        }),
    }
}

/// Try `(x^(2k))^(1/2) -> |x|^k`.
pub fn try_rewrite_power_power_even_root_abs_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<PowerPowerEvenRootAbsRewrite> {
    let (base, outer_exp) = match ctx.get(expr) {
        Expr::Pow(base, exp) => (*base, *exp),
        _ => return None,
    };
    let (inner_base, inner_exp) = match ctx.get(base) {
        Expr::Pow(inner_base, inner_exp) => (*inner_base, *inner_exp),
        _ => return None,
    };

    let inner_even_int = match ctx.get(inner_exp) {
        Expr::Number(n) => n.is_integer() && n.to_integer().is_even(),
        _ => false,
    };
    if !inner_even_int || !is_half(ctx, outer_exp) {
        return None;
    }

    let prod_exp = mul_exp(ctx, inner_exp, outer_exp);
    let abs_base = ctx.call_builtin(BuiltinFn::Abs, vec![inner_base]);
    let rewritten = ctx.add(Expr::Pow(abs_base, prod_exp));
    Some(PowerPowerEvenRootAbsRewrite { rewritten })
}

/// Analyze `(x^a)^b` shape and derive reusable structural facts.
pub fn analyze_power_power_expr(ctx: &mut Context, expr: ExprId) -> Option<PowerPowerAnalysis> {
    let (base, outer_exp) = match ctx.get(expr) {
        Expr::Pow(base, exp) => (*base, *exp),
        _ => return None,
    };
    let (inner_base, inner_exp) = match ctx.get(base) {
        Expr::Pow(inner_base, inner_exp) => (*inner_base, *inner_exp),
        _ => return None,
    };

    let inner_is_even_root = match ctx.get(inner_exp) {
        Expr::Number(n) => n.denom().is_even(),
        _ => false,
    };

    // Preserve current engine semantics: only denominator match is checked.
    let symbolic_root_cancel = match ctx.get(outer_exp) {
        Expr::Div(_, denom) => compare_expr(ctx, *denom, inner_exp) == std::cmp::Ordering::Equal,
        _ => false,
    };

    let multiplied_exp = mul_exp(ctx, inner_exp, outer_exp);
    let multiplied_expr = ctx.add(Expr::Pow(inner_base, multiplied_exp));

    Some(PowerPowerAnalysis {
        inner_base,
        inner_exp,
        outer_exp,
        multiplied_expr,
        inner_is_even_root,
        symbolic_root_cancel,
    })
}

/// Classify `(x^a)^b` into a rewrite shape, without domain policy decisions.
pub fn classify_power_power_pattern(ctx: &mut Context, expr: ExprId) -> Option<PowerPowerPattern> {
    let analysis = analyze_power_power_expr(ctx, expr)?;
    if let Some(rewrite) = try_rewrite_power_power_even_root_abs_expr(ctx, expr) {
        return Some(PowerPowerPattern::EvenRootAbs {
            rewritten: rewrite.rewritten,
        });
    }
    if analysis.inner_is_even_root {
        return Some(PowerPowerPattern::EvenRootNeedsNonNegative {
            rewritten: analysis.multiplied_expr,
            inner_base: analysis.inner_base,
        });
    }
    if analysis.symbolic_root_cancel {
        return Some(PowerPowerPattern::SymbolicRootCancelCandidate {
            rewritten: analysis.multiplied_expr,
            inner_base: analysis.inner_base,
            inner_exp: analysis.inner_exp,
        });
    }
    Some(PowerPowerPattern::MultiplyExponents {
        rewritten: analysis.multiplied_expr,
    })
}

/// Prepare rewrite decision for the `(x^a)^b` even-root branch using an injected
/// non-negativity prover.
pub fn decide_power_power_even_root_guard_with<FProveNonnegative>(
    ctx: &mut Context,
    expr: ExprId,
    mut prove_nonnegative: FProveNonnegative,
) -> Option<PowerPowerEvenRootGuardDecision>
where
    FProveNonnegative: FnMut(&Context, ExprId) -> TriProof,
{
    let pattern = classify_power_power_pattern(ctx, expr)?;
    let (rewritten, inner_base) = match pattern {
        PowerPowerPattern::EvenRootNeedsNonNegative {
            rewritten,
            inner_base,
        } => (rewritten, inner_base),
        _ => return None,
    };

    let proof = prove_nonnegative(ctx, inner_base);
    if proof.is_proven() {
        Some(PowerPowerEvenRootGuardDecision::ProvenRewrite { rewritten })
    } else {
        Some(PowerPowerEvenRootGuardDecision::NeedsNonNegativeCondition {
            rewritten,
            inner_base,
        })
    }
}

/// Plan the even-root action directly from structure + prover callback.
pub fn plan_power_power_even_root_action_with<FProveNonnegative>(
    ctx: &mut Context,
    expr: ExprId,
    prove_nonnegative: FProveNonnegative,
) -> Option<PowerPowerEvenRootAction>
where
    FProveNonnegative: FnMut(&Context, ExprId) -> TriProof,
{
    match decide_power_power_even_root_guard_with(ctx, expr, prove_nonnegative)? {
        PowerPowerEvenRootGuardDecision::ProvenRewrite { rewritten } => {
            Some(PowerPowerEvenRootAction::Apply { rewritten })
        }
        PowerPowerEvenRootGuardDecision::NeedsNonNegativeCondition {
            rewritten,
            inner_base,
        } => Some(PowerPowerEvenRootAction::NeedsNonNegativeCondition {
            rewritten,
            inner_base,
        }),
    }
}

/// Decide policy for symbolic root cancellation in `(x^n)^(1/n)` style rewrites.
pub fn decide_symbolic_root_cancel_policy(
    value_domain_is_real_only: bool,
    mode: SymbolicRootCancelDomainMode,
) -> SymbolicRootCancelPolicy {
    if !value_domain_is_real_only {
        return SymbolicRootCancelPolicy::ApplyUnconditionally;
    }

    match mode {
        SymbolicRootCancelDomainMode::Assume => SymbolicRootCancelPolicy::ApplyWithAssumptions,
        SymbolicRootCancelDomainMode::Strict | SymbolicRootCancelDomainMode::Generic => {
            SymbolicRootCancelPolicy::BlockedNeedsAssumeMode
        }
    }
}

/// Plan symbolic root-cancel handling for a caller-provided rewrite candidate.
pub fn plan_symbolic_root_cancel_action(
    value_domain_is_real_only: bool,
    mode: SymbolicRootCancelDomainMode,
    rewritten: ExprId,
    inner_base: ExprId,
    inner_exp: ExprId,
) -> SymbolicRootCancelAction {
    match decide_symbolic_root_cancel_policy(value_domain_is_real_only, mode) {
        SymbolicRootCancelPolicy::ApplyUnconditionally => {
            SymbolicRootCancelAction::ApplyUnconditionally { rewritten }
        }
        SymbolicRootCancelPolicy::ApplyWithAssumptions => {
            SymbolicRootCancelAction::ApplyWithAssumptions {
                rewritten,
                inner_base,
                inner_exp,
            }
        }
        SymbolicRootCancelPolicy::BlockedNeedsAssumeMode => {
            SymbolicRootCancelAction::BlockedNeedsAssumeMode {
                inner_base,
                inner_exp,
            }
        }
    }
}

/// Plan symbolic root-cancel action from generic mode flags.
pub fn plan_symbolic_root_cancel_action_with_mode_flags(
    value_domain_is_real_only: bool,
    assume_mode: bool,
    strict_mode: bool,
    rewritten: ExprId,
    inner_base: ExprId,
    inner_exp: ExprId,
) -> SymbolicRootCancelAction {
    let mode = symbolic_root_cancel_mode_from_flags(assume_mode, strict_mode);
    plan_symbolic_root_cancel_action(
        value_domain_is_real_only,
        mode,
        rewritten,
        inner_base,
        inner_exp,
    )
}

/// Try `(u^y)^(1/y) -> u` with strict-mode gating delegated by callbacks.
pub fn try_rewrite_powpow_cancel_reciprocal_expr_with<FProvePositive, FProveNonzero>(
    ctx: &mut Context,
    expr: ExprId,
    strict_mode: bool,
    mut prove_positive: FProvePositive,
    mut prove_nonzero: FProveNonzero,
) -> Option<PowPowCancelReciprocalRewrite>
where
    FProvePositive: FnMut(&Context, ExprId) -> TriProof,
    FProveNonzero: FnMut(&Context, ExprId) -> TriProof,
{
    let (inner_pow, outer_exp) = match ctx.get(expr) {
        Expr::Pow(base, exp) => (*base, *exp),
        _ => return None,
    };
    let (u, y) = match ctx.get(inner_pow) {
        Expr::Pow(base, exp) => (*base, *exp),
        _ => return None,
    };
    if !has_reciprocal_outer_exponent(ctx, outer_exp, y) {
        return None;
    }

    let proof_u_pos = prove_positive(ctx, u);
    let proof_y_nonzero = prove_nonzero(ctx, y);
    if strict_mode && (!proof_u_pos.is_proven() || !proof_y_nonzero.is_proven()) {
        return None;
    }

    Some(PowPowCancelReciprocalRewrite {
        rewritten: u,
        base: u,
        inner_exp: y,
        assume_base_positive: !proof_u_pos.is_proven(),
        assume_exp_nonzero: !proof_y_nonzero.is_proven(),
    })
}

#[cfg(test)]
mod tests {
    use super::{
        analyze_power_power_expr, classify_power_power_pattern, classify_root_pow_cancel_pattern,
        decide_power_power_even_root_guard_with, decide_symbolic_root_cancel_policy,
        merge_nonnegative_proof_with_witness, plan_power_power_even_root_action_with,
        plan_symbolic_root_cancel_action, plan_symbolic_root_cancel_action_with_mode_flags,
        symbolic_root_cancel_mode_from_flags, try_rewrite_power_power_even_root_abs_expr,
        try_rewrite_powpow_cancel_reciprocal_expr_with, try_rewrite_root_merge_div_expr_with,
        try_rewrite_root_merge_mul_expr_with, try_rewrite_root_pow_cancel_expr,
        PowerPowerEvenRootAction, PowerPowerEvenRootGuardDecision, PowerPowerPattern,
        RootPowCancelKind, RootPowCancelPattern, SymbolicRootCancelAction,
        SymbolicRootCancelDomainMode, SymbolicRootCancelPolicy,
    };
    use crate::tri_proof::TriProof;
    use cas_ast::Context;
    use cas_parser::parse;

    #[test]
    fn root_merge_mul_allows_unknown_when_not_strict() {
        let mut ctx = Context::new();
        let expr = parse("sqrt(x)*sqrt(y)", &mut ctx).expect("parse");
        let rewrite = try_rewrite_root_merge_mul_expr_with(&mut ctx, expr, false, |_ctx, _e| {
            TriProof::Unknown
        })
        .expect("rewrite");
        assert!(rewrite.assume_left_nonnegative);
        assert!(rewrite.assume_right_nonnegative);
    }

    #[test]
    fn root_merge_div_rejects_unknown_in_strict_mode() {
        let mut ctx = Context::new();
        let expr = parse("sqrt(x)/sqrt(y)", &mut ctx).expect("parse");
        let rewrite = try_rewrite_root_merge_div_expr_with(
            &mut ctx,
            expr,
            true,
            |_ctx, _e| TriProof::Unknown,
            |_ctx, _e| TriProof::Unknown,
        );
        assert!(rewrite.is_none());
    }

    #[test]
    fn powpow_cancel_recognizes_reciprocal_exponent() {
        let mut ctx = Context::new();
        let expr = parse("(x^y)^(1/y)", &mut ctx).expect("parse");
        let rewrite = try_rewrite_powpow_cancel_reciprocal_expr_with(
            &mut ctx,
            expr,
            false,
            |_ctx, _e| TriProof::Unknown,
            |_ctx, _e| TriProof::Unknown,
        );
        assert!(rewrite.is_some());
    }

    #[test]
    fn powpow_cancel_requires_proofs_in_strict_mode() {
        let mut ctx = Context::new();
        let expr = parse("(x^y)^(1/y)", &mut ctx).expect("parse");
        let rewrite = try_rewrite_powpow_cancel_reciprocal_expr_with(
            &mut ctx,
            expr,
            true,
            |_ctx, _e| TriProof::Unknown,
            |_ctx, _e| TriProof::Unknown,
        );
        assert!(rewrite.is_none());
    }

    #[test]
    fn root_pow_cancel_even_goes_to_abs() {
        let mut ctx = Context::new();
        let expr = parse("(x^2)^(1/2)", &mut ctx).expect("parse");
        let rewrite = try_rewrite_root_pow_cancel_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(rewrite.kind, RootPowCancelKind::NumericEven);
    }

    #[test]
    fn root_pow_cancel_symbolic_detects_candidate() {
        let mut ctx = Context::new();
        let expr = parse("(x^n)^(1/n)", &mut ctx).expect("parse");
        let rewrite = try_rewrite_root_pow_cancel_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(rewrite.kind, RootPowCancelKind::Symbolic);
        assert_eq!(rewrite.rewritten, rewrite.inner_base);
    }

    #[test]
    fn power_power_even_root_abs_rewrite_detects_pattern() {
        let mut ctx = Context::new();
        let expr = parse("(x^4)^0.5", &mut ctx).expect("parse");
        let rewrite = try_rewrite_power_power_even_root_abs_expr(&mut ctx, expr);
        assert!(rewrite.is_some());
    }

    #[test]
    fn analyze_power_power_extracts_flags() {
        let mut ctx = Context::new();
        let expr = parse("(x^(3/2))^(1/(3/2))", &mut ctx).expect("parse");
        let analysis = analyze_power_power_expr(&mut ctx, expr).expect("analysis");
        assert!(!analysis.inner_is_even_root);
        assert!(analysis.symbolic_root_cancel);
    }

    #[test]
    fn classify_power_power_reports_symbolic_root_cancel() {
        let mut ctx = Context::new();
        let expr = parse("(x^n)^(1/n)", &mut ctx).expect("parse");
        let pat = classify_power_power_pattern(&mut ctx, expr).expect("pattern");
        assert!(matches!(
            pat,
            PowerPowerPattern::SymbolicRootCancelCandidate { .. }
        ));
    }

    #[test]
    fn classify_root_pow_cancel_reports_symbolic_candidate() {
        let mut ctx = Context::new();
        let expr = parse("(x^n)^(1/n)", &mut ctx).expect("parse");
        let pat = classify_root_pow_cancel_pattern(&mut ctx, expr).expect("pattern");
        assert!(matches!(
            pat,
            RootPowCancelPattern::SymbolicCandidate { .. }
        ));
    }

    #[test]
    fn decide_even_root_guard_reports_needs_condition_when_unknown() {
        let mut ctx = Context::new();
        let expr = parse("(x^1.5)^y", &mut ctx).expect("parse");
        let decision = decide_power_power_even_root_guard_with(&mut ctx, expr, |_ctx, _base| {
            TriProof::Unknown
        })
        .expect("decision");
        assert!(matches!(
            decision,
            PowerPowerEvenRootGuardDecision::NeedsNonNegativeCondition { .. }
        ));
    }

    #[test]
    fn plan_even_root_action_maps_guard_variants() {
        let mut ctx = Context::new();
        let expr = parse("(x^1.5)^y", &mut ctx).expect("parse");

        let needs =
            plan_power_power_even_root_action_with(&mut ctx, expr, |_ctx, _base| TriProof::Unknown)
                .expect("action");
        assert!(matches!(
            needs,
            PowerPowerEvenRootAction::NeedsNonNegativeCondition { .. }
        ));

        let apply =
            plan_power_power_even_root_action_with(&mut ctx, expr, |_ctx, _base| TriProof::Proven)
                .expect("action");
        assert!(matches!(apply, PowerPowerEvenRootAction::Apply { .. }));
    }

    #[test]
    fn symbolic_root_cancel_policy_real_generic_blocks() {
        let decision =
            decide_symbolic_root_cancel_policy(true, SymbolicRootCancelDomainMode::Generic);
        assert_eq!(decision, SymbolicRootCancelPolicy::BlockedNeedsAssumeMode);
    }

    #[test]
    fn symbolic_root_cancel_policy_real_assume_requires_assumptions() {
        let decision =
            decide_symbolic_root_cancel_policy(true, SymbolicRootCancelDomainMode::Assume);
        assert_eq!(decision, SymbolicRootCancelPolicy::ApplyWithAssumptions);
    }

    #[test]
    fn symbolic_root_cancel_policy_complex_is_unconditional() {
        let decision =
            decide_symbolic_root_cancel_policy(false, SymbolicRootCancelDomainMode::Strict);
        assert_eq!(decision, SymbolicRootCancelPolicy::ApplyUnconditionally);
    }

    #[test]
    fn symbolic_root_cancel_mode_from_flags_prioritizes_assume_then_strict() {
        assert_eq!(
            symbolic_root_cancel_mode_from_flags(true, true),
            SymbolicRootCancelDomainMode::Assume
        );
        assert_eq!(
            symbolic_root_cancel_mode_from_flags(false, true),
            SymbolicRootCancelDomainMode::Strict
        );
        assert_eq!(
            symbolic_root_cancel_mode_from_flags(false, false),
            SymbolicRootCancelDomainMode::Generic
        );
    }

    #[test]
    fn symbolic_root_cancel_action_contains_expected_payload() {
        let mut ctx = Context::new();
        let rewritten = parse("x", &mut ctx).expect("parse");
        let inner_base = parse("x", &mut ctx).expect("parse");
        let inner_exp = parse("n", &mut ctx).expect("parse");

        let blocked = plan_symbolic_root_cancel_action(
            true,
            SymbolicRootCancelDomainMode::Generic,
            rewritten,
            inner_base,
            inner_exp,
        );
        assert!(matches!(
            blocked,
            SymbolicRootCancelAction::BlockedNeedsAssumeMode { .. }
        ));

        let assumed = plan_symbolic_root_cancel_action(
            true,
            SymbolicRootCancelDomainMode::Assume,
            rewritten,
            inner_base,
            inner_exp,
        );
        assert!(matches!(
            assumed,
            SymbolicRootCancelAction::ApplyWithAssumptions { .. }
        ));

        let unconditional = plan_symbolic_root_cancel_action(
            false,
            SymbolicRootCancelDomainMode::Strict,
            rewritten,
            inner_base,
            inner_exp,
        );
        assert!(matches!(
            unconditional,
            SymbolicRootCancelAction::ApplyUnconditionally { .. }
        ));
    }

    #[test]
    fn symbolic_root_cancel_action_with_mode_flags_matches_direct_mode() {
        let mut ctx = Context::new();
        let rewritten = parse("x", &mut ctx).expect("parse");
        let inner_base = parse("x", &mut ctx).expect("parse");
        let inner_exp = parse("n", &mut ctx).expect("parse");

        let via_mode = plan_symbolic_root_cancel_action(
            true,
            SymbolicRootCancelDomainMode::Assume,
            rewritten,
            inner_base,
            inner_exp,
        );
        let via_flags = plan_symbolic_root_cancel_action_with_mode_flags(
            true, true, false, rewritten, inner_base, inner_exp,
        );
        assert_eq!(via_flags, via_mode);
    }

    #[test]
    fn merge_nonnegative_proof_with_witness_respects_explicit_proof() {
        assert_eq!(
            merge_nonnegative_proof_with_witness(TriProof::Proven, false, false),
            TriProof::Proven
        );
        assert_eq!(
            merge_nonnegative_proof_with_witness(TriProof::Disproven, true, true),
            TriProof::Disproven
        );
    }

    #[test]
    fn merge_nonnegative_proof_with_witness_upgrades_unknown_only_when_both_flags_true() {
        assert_eq!(
            merge_nonnegative_proof_with_witness(TriProof::Unknown, true, true),
            TriProof::Proven
        );
        assert_eq!(
            merge_nonnegative_proof_with_witness(TriProof::Unknown, true, false),
            TriProof::Unknown
        );
        assert_eq!(
            merge_nonnegative_proof_with_witness(TriProof::Unknown, false, true),
            TriProof::Unknown
        );
    }
}
