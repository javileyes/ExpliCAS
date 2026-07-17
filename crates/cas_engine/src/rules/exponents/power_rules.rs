use crate::define_rule;
use crate::phase::PhaseMask;
use crate::rule::Rewrite;
use cas_ast::Context;
use cas_ast::Expr;
use cas_ast::ExprId;
use cas_solver_core::rule_names::RULE_EVALUATE_NUMERIC_POWER;

fn format_power_eval_static_desc(
    kind: cas_math::power_eval_support::PowerEvalStaticRewriteKind,
) -> &'static str {
    match kind {
        cas_math::power_eval_support::PowerEvalStaticRewriteKind::NegativeExponentNormalization => {
            "x^(-n) -> 1/x^n"
        }
        cas_math::power_eval_support::PowerEvalStaticRewriteKind::NegativeBaseEven => {
            "(-x)^even -> x^even"
        }
        cas_math::power_eval_support::PowerEvalStaticRewriteKind::NegativeBaseOdd => {
            "(-x)^odd -> -(x^odd)"
        }
    }
}

fn format_power_product_desc(
    kind: cas_math::power_product_support::PowerProductRewriteKind,
) -> &'static str {
    match kind {
        cas_math::power_product_support::PowerProductRewriteKind::SameBase => {
            "Combine powers with same base"
        }
        cas_math::power_product_support::PowerProductRewriteKind::PowerAndBase => {
            "Combine power and base"
        }
        cas_math::power_product_support::PowerProductRewriteKind::BaseAndPower => {
            "Combine base and power"
        }
        cas_math::power_product_support::PowerProductRewriteKind::MultiplyIdenticalTerms => {
            "Multiply identical terms"
        }
        cas_math::power_product_support::PowerProductRewriteKind::NestedIdenticalTerms => {
            "Combine nested identical terms"
        }
        cas_math::power_product_support::PowerProductRewriteKind::NestedPowers => {
            "Combine nested powers"
        }
        cas_math::power_product_support::PowerProductRewriteKind::BaseAndNestedPower => {
            "Combine base and nested power"
        }
        cas_math::power_product_support::PowerProductRewriteKind::PowerAndNestedBase => {
            "Combine power and nested base"
        }
        cas_math::power_product_support::PowerProductRewriteKind::CoeffPowerAndPower => {
            "Combine coeff-power and power"
        }
        cas_math::power_product_support::PowerProductRewriteKind::CoeffPowerAndBase => {
            "Combine coeff-power and base"
        }
        cas_math::power_product_support::PowerProductRewriteKind::CoeffBaseAndPower => {
            "Combine coeff-base and power"
        }
        cas_math::power_product_support::PowerProductRewriteKind::NestedBaseAndPower => {
            "Combine nested base and power"
        }
        cas_math::power_product_support::PowerProductRewriteKind::SameExponent => {
            "Combine powers with same exponent"
        }
        cas_math::power_product_support::PowerProductRewriteKind::NestedSameExponent => {
            "Combine nested powers with same exponent"
        }
        cas_math::power_product_support::PowerProductRewriteKind::QuotientSameExponent => {
            "a^n / b^n = (a/b)^n"
        }
        _ => "Power product rewrite",
    }
}

const ROOT_CANCEL_ASSUME_SUGGESTION: &str =
    "Use 'semantics set domain assume' to simplify (x^n)^(1/n) → x.";

fn register_symbolic_root_cancel_hints(
    ctx: &cas_ast::Context,
    rule: &str,
    inner_base: ExprId,
    inner_exp: ExprId,
) {
    let hint1 = crate::BlockedHint {
        key: crate::AssumptionKey::positive_key(ctx, inner_base),
        expr_id: inner_base,
        rule: rule.to_string(),
        suggestion: ROOT_CANCEL_ASSUME_SUGGESTION,
    };
    let hint2 = crate::BlockedHint {
        key: crate::AssumptionKey::nonzero_key(ctx, inner_exp),
        expr_id: inner_exp,
        rule: rule.to_string(),
        suggestion: ROOT_CANCEL_ASSUME_SUGGESTION,
    };
    crate::register_blocked_hint(hint1);
    crate::register_blocked_hint(hint2);
}

fn assumed_symbolic_root_cancel_rewrite(
    ctx: &Context,
    rewritten: ExprId,
    inner_base: ExprId,
    inner_exp: ExprId,
) -> Rewrite {
    use crate::ImplicitCondition;
    Rewrite::new(rewritten)
        .desc("(x^n)^(1/n) = x (assuming x > 0, n ≠ 0)")
        .requires(ImplicitCondition::Positive(inner_base))
        .requires(ImplicitCondition::NonZero(inner_exp))
        .assume(crate::AssumptionEvent::positive_assumed(ctx, inner_base))
}

fn apply_symbolic_root_cancel_action(
    ctx: &Context,
    rule_name: &str,
    action: cas_math::root_power_canonical_support::SymbolicRootCancelAction,
    unconditional_desc: &'static str,
) -> Option<Rewrite> {
    match action {
        cas_math::root_power_canonical_support::SymbolicRootCancelAction::BlockedNeedsAssumeMode {
            inner_base,
            inner_exp,
        } => {
            register_symbolic_root_cancel_hints(ctx, rule_name, inner_base, inner_exp);
            None
        }
        cas_math::root_power_canonical_support::SymbolicRootCancelAction::ApplyWithAssumptions {
            rewritten,
            inner_base,
            inner_exp,
        } => Some(assumed_symbolic_root_cancel_rewrite(
            ctx, rewritten, inner_base, inner_exp,
        )),
        cas_math::root_power_canonical_support::SymbolicRootCancelAction::ApplyUnconditionally {
            rewritten,
        } => Some(Rewrite::new(rewritten).desc(unconditional_desc)),
    }
}

fn symbolic_root_cancel_action_for_parent(
    parent_ctx: &crate::parent_context::ParentContext,
    rewritten: ExprId,
    inner_base: ExprId,
    inner_exp: ExprId,
) -> cas_math::root_power_canonical_support::SymbolicRootCancelAction {
    use crate::semantics::ValueDomain;

    let domain_mode = parent_ctx.domain_mode();
    cas_math::root_power_canonical_support::plan_symbolic_root_cancel_action_with_mode_flags(
        parent_ctx.value_domain() == ValueDomain::RealOnly,
        matches!(domain_mode, crate::DomainMode::Assume),
        matches!(domain_mode, crate::DomainMode::Strict),
        rewritten,
        inner_base,
        inner_exp,
    )
}

fn power_power_nonnegative_proof_with_witness(
    core_ctx: &Context,
    base: ExprId,
    full_expr: ExprId,
    parent_ctx: &crate::parent_context::ParentContext,
    value_domain: crate::semantics::ValueDomain,
) -> cas_math::tri_proof::TriProof {
    use crate::{witness_survives_in_context, WitnessKind};

    let explicit = cas_solver_core::predicate_proofs::prove_nonnegative_core_with(
        core_ctx,
        base,
        value_domain,
        crate::helpers::prove_nonnegative,
    );
    let (implicit_contains_nonnegative, witness_survives) = if let (Some(implicit), Some(root)) =
        (parent_ctx.implicit_domain(), parent_ctx.root_expr())
    {
        let contains = implicit.contains_nonnegative(base);
        let survives = contains
            && witness_survives_in_context(
                core_ctx,
                base,
                root,
                full_expr,
                Some(base),
                WitnessKind::Sqrt,
            );
        (contains, survives)
    } else {
        (false, false)
    };

    cas_math::root_power_canonical_support::merge_nonnegative_proof_with_witness(
        explicit,
        implicit_contains_nonnegative,
        witness_survives,
    )
}

/// Split `(base^a)^b` into `(base, a, b)`.
fn power_power_parts(ctx: &Context, expr: ExprId) -> Option<(ExprId, ExprId, ExprId)> {
    if let Expr::Pow(inner, b) = ctx.get(expr) {
        let (inner, b) = (*inner, *b);
        if let Expr::Pow(base, a) = ctx.get(inner) {
            return Some((*base, *a, b));
        }
    }
    None
}

fn is_number_const(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Number(_))
}

/// Decide the `(x^a)^b → x^(a·b)` (`MultiplyExponents`) rewrite, threading the base's sign domain.
///
/// When both exponents are CONCRETE rationals the upstream classifier has already split off the
/// sign-unsafe shapes (`(x^2)^(1/2) -> |x|` via the even-root branch), so what reaches here is
/// sign-safe and folds unconditionally (`(x^2)^(1/3) -> x^(2/3)`, `(x^4)^(3/2) -> x^6`). The danger is
/// a SYMBOLIC exponent: `(x^a)^b` can hide an even `a` whose `x^a >= 0` the fold would silently drop
/// the sign of (`((-3)^2)^(1/2) = 3`, but folding to `(-3)^1 = -3`), and a negative base never
/// satisfies the identity. So when either exponent is symbolic, gate the fold through the same domain
/// oracle as the even-root branch: a provably-negative base (or strict mode) declines, an unknown-sign
/// base folds only under a recorded non-negativity assumption (e.g. `--domain assume`), and a
/// provably-non-negative base folds unconditionally.
fn decide_multiply_exponents_rewrite(
    ctx: &Context,
    parent_ctx: &crate::parent_context::ParentContext,
    expr: ExprId,
    rewritten: ExprId,
) -> Option<Rewrite> {
    let unconditional = || Rewrite::new(rewritten).desc("Multiply exponents");
    let Some((base, a, b)) = power_power_parts(ctx, expr) else {
        return Some(unconditional());
    };
    if is_number_const(ctx, a) && is_number_const(ctx, b) {
        return Some(unconditional());
    }
    let decision = crate::oracle_allows_with_hint(
        ctx,
        parent_ctx.domain_mode(),
        parent_ctx.value_domain(),
        &crate::Predicate::NonNegative(base),
        "Power of a Power",
    );
    if !decision.allow {
        return None;
    }
    let mut rewrite = unconditional();
    if decision.assumption.is_some() {
        rewrite = rewrite.assume(crate::AssumptionEvent::nonnegative(ctx, base));
    }
    Some(rewrite)
}

fn apply_power_power_even_root_action(
    ctx: &Context,
    parent_ctx: &crate::parent_context::ParentContext,
    action: cas_math::root_power_canonical_support::PowerPowerEvenRootAction,
) -> Option<Rewrite> {
    match action {
        cas_math::root_power_canonical_support::PowerPowerEvenRootAction::Apply { rewritten } => {
            Some(Rewrite::new(rewritten).desc("Multiply exponents"))
        }
        cas_math::root_power_canonical_support::PowerPowerEvenRootAction::NeedsNonNegativeCondition {
            rewritten,
            inner_base,
        } => {
            let mode = parent_ctx.domain_mode();
            let vd = parent_ctx.value_domain();
            let decision = crate::oracle_allows_with_hint(
                ctx,
                mode,
                vd,
                &crate::Predicate::NonNegative(inner_base),
                "Power of a Power",
            );
            if !decision.allow {
                return None;
            }

            let mut rewrite = Rewrite::new(rewritten).desc("Multiply exponents");
            if decision.assumption.is_some() {
                rewrite =
                    rewrite.assume(crate::AssumptionEvent::nonnegative(ctx, inner_base));
            }
            Some(rewrite)
        }
    }
}

define_rule!(
    ProductPowerRule,
    "Product of Powers",
    Some(crate::target_kind::TargetKindSet::MUL),
    |ctx, expr| {
        let rewrite = cas_math::power_product_support::try_rewrite_product_power_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(format_power_product_desc(rewrite.kind)))
    }
);

// a^n * b^n = (ab)^n - combines products of powers with same exponent
// Guard: at least one base must contain a numeric factor to avoid infinite loop with PowerProductRule
define_rule!(
    ProductSameExponentRule,
    "Product Same Exponent",
    Some(crate::target_kind::TargetKindSet::MUL),
    PhaseMask::CORE | PhaseMask::TRANSFORM | PhaseMask::RATIONALIZE,
    |ctx, expr| {
        let rewrite =
            cas_math::power_product_support::try_rewrite_product_same_exponent_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(format_power_product_desc(rewrite.kind)))
    }
);

// a^n / b^n = (a/b)^n
define_rule!(
    QuotientSameExponentRule,
    "Quotient Same Exponent",
    None,
    PhaseMask::CORE | PhaseMask::TRANSFORM,
    |ctx, expr| {
        let rewrite =
            cas_math::power_product_support::try_rewrite_quotient_same_exponent_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(format_power_product_desc(rewrite.kind)))
    }
);

// ============================================================================
// RootPowCancelRule: (x^n)^(1/n) → x (odd n) or |x| (even n)
// ============================================================================
pub struct RootPowCancelRule;

impl crate::rule::Rule for RootPowCancelRule {
    fn name(&self) -> &str {
        "Root Power Cancel"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<crate::rule::Rewrite> {
        use crate::semantics::ValueDomain;

        let pattern =
            cas_math::root_power_canonical_support::classify_root_pow_cancel_pattern(ctx, expr)?;

        let vd = parent_ctx.value_domain();
        if vd == ValueDomain::ComplexEnabled {
            return None;
        }

        match pattern {
            cas_math::root_power_canonical_support::RootPowCancelPattern::NumericEven {
                rewritten,
            } => Some(crate::rule::Rewrite::new(rewritten).desc("(x^n)^(1/n) = |x| for even n")),
            cas_math::root_power_canonical_support::RootPowCancelPattern::NumericOdd {
                rewritten,
            } => Some(crate::rule::Rewrite::new(rewritten).desc("(x^n)^(1/n) = x for odd n")),
            cas_math::root_power_canonical_support::RootPowCancelPattern::SymbolicCandidate {
                rewritten,
                inner_base,
                inner_exp,
            } => {
                let action = symbolic_root_cancel_action_for_parent(
                    parent_ctx, rewritten, inner_base, inner_exp,
                );
                apply_symbolic_root_cancel_action(
                    ctx,
                    "Root Power Cancel",
                    action,
                    "(x^n)^(1/n) = x",
                )
            }
        }
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::POW)
    }

    fn priority(&self) -> i32 {
        15
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::High
    }
}

// ============================================================================
// ComplexNegativeBaseRootRule: in COMPLEX mode, `(-r)^(p/q)` with ODD denominator `q` is the
// PRINCIPAL value `r^(p/q)·(cos(π·p/q) + i·sin(π·p/q))`, NOT the real odd root. `(-1)^(1/3)` is
// `1/2 + (√3/2)i`, not `-1`. (`EvaluatePowerRule` would otherwise leak the real-odd-root literal
// value into complex mode; even denominators are handled by the `sqrt(-n) → i·sqrt(n)` rewrite.)
// ============================================================================
pub struct ComplexNegativeBaseRootRule;

impl crate::rule::Rule for ComplexNegativeBaseRootRule {
    fn name(&self) -> &str {
        "Complex Negative Base Root"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<crate::rule::Rewrite> {
        use crate::semantics::ValueDomain;
        use num_integer::Integer;
        use num_rational::BigRational;
        use num_traits::Zero;

        if parent_ctx.value_domain() != ValueDomain::ComplexEnabled {
            return None;
        }
        let (base, exp) = match ctx.get(expr) {
            Expr::Pow(b, e) => (*b, *e),
            _ => return None,
        };
        let base_val = cas_math::numeric_eval::as_rational_const(ctx, base)?;
        if base_val >= BigRational::zero() {
            return None; // only a strictly-negative base
        }
        let exp_val = cas_math::numeric_eval::as_rational_const(ctx, exp)?;
        if exp_val.is_integer() || exp_val.denom().is_even() {
            return None; // integer or even-denominator (the latter handled by sqrt(-n) → i·sqrt(n))
        }
        // Principal value: (-r)^(p/q) = r^(p/q)·(cos(π·p/q) + i·sin(π·p/q)), with r = |base|.
        let r_node = ctx.add(Expr::Number(-base_val));
        let magnitude = ctx.add(Expr::Pow(r_node, exp));
        let pi = ctx.add(Expr::Constant(cas_ast::Constant::Pi));
        let angle = ctx.add(Expr::Mul(pi, exp));
        let cos_a = ctx.call("cos", vec![angle]);
        let sin_a = ctx.call("sin", vec![angle]);
        let imaginary = ctx.add(Expr::Constant(cas_ast::Constant::I));
        let i_sin = ctx.add(Expr::Mul(imaginary, sin_a));
        let trig = ctx.add(Expr::Add(cos_a, i_sin));
        let result = ctx.add(Expr::Mul(magnitude, trig));
        Some(
            crate::rule::Rewrite::new(result)
                .desc("(-r)^(p/q) = r^(p/q)·(cos(πp/q) + i·sin(πp/q)) (principal branch)"),
        )
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::POW)
    }

    fn priority(&self) -> i32 {
        16 // fire before EvaluatePowerRule (which would emit the real odd root)
    }
}

define_rule!(
    PowerPowerRule,
    "Power of a Power",
    solve_safety: crate::SolveSafety::NeedsCondition(
        crate::ConditionClass::Analytic
    ),
    |ctx, expr, parent_ctx| {
    // (x^a)^b -> x^(a*b)
    // COMPLEX GATE: over ℂ only an INTEGER outer exponent folds soundly
    // ((z^a)^n = z^(a·n)); every fractional outer exponent risks a
    // principal-branch collapse — (i²)^(1/2) = i ≠ |i| = 1 and
    // (i⁴)^(1/2) = 1 ≠ i² = -1 — so all |·|-emitting and multiply arms
    // decline and the expression stays an honest residual.
    if parent_ctx.value_domain() != crate::semantics::ValueDomain::RealOnly {
        let outer_integer = matches!(ctx.get(expr), cas_ast::Expr::Pow(_, b)
            if cas_math::numeric_eval::as_rational_const(ctx, *b).is_some_and(|r| r.is_integer()));
        if !outer_integer {
            return None;
        }
    }
    if let Some(pattern) = cas_math::root_power_canonical_support::classify_power_power_pattern(ctx, expr) {
        match pattern {
            cas_math::root_power_canonical_support::PowerPowerPattern::EvenRootAbs { rewritten } => {
                // REAL-ONLY: `(x^2k)^(1/2) = |x|^k` is a real-domain identity —
                // over ℂ, `(i²)^(1/2) = i ≠ 1 = |i|`.
                if parent_ctx.value_domain() != crate::semantics::ValueDomain::RealOnly {
                    return None;
                }
                return Some(Rewrite::new(rewritten).desc(
                    "Power of power with even root: (x^2k)^(1/2) -> |x|^k",
                ));
            }
            cas_math::root_power_canonical_support::PowerPowerPattern::EvenRootNeedsNonNegative { .. } => {
                let vd = parent_ctx.value_domain();

                let action = cas_math::root_power_canonical_support::plan_power_power_even_root_action_with(
                    ctx,
                    expr,
                    |core_ctx, base| {
                        power_power_nonnegative_proof_with_witness(
                            core_ctx, base, expr, parent_ctx, vd,
                        )
                    },
                )?;
                return apply_power_power_even_root_action(ctx, parent_ctx, action);
            }
            cas_math::root_power_canonical_support::PowerPowerPattern::SymbolicRootCancelCandidate {
                rewritten,
                inner_base,
                inner_exp,
            } => {
                let action =
                    symbolic_root_cancel_action_for_parent(parent_ctx, rewritten, inner_base, inner_exp);
                if let Some(rewrite) = apply_symbolic_root_cancel_action(
                    ctx,
                    "Power of a Power",
                    action,
                    "Multiply exponents",
                ) {
                    return Some(rewrite);
                }
                return None;
            }
            cas_math::root_power_canonical_support::PowerPowerPattern::MultiplyExponents {
                rewritten,
            } => {
                return decide_multiply_exponents_rewrite(ctx, parent_ctx, expr, rewritten);
            }
        }
    }
    None
});

// ============================================================================
// NegativeExponentNormalizationRule: x^(-n) → 1/x^n
// ============================================================================
define_rule!(
    NegativeExponentNormalizationRule,
    "Normalize Negative Exponent",
    Some(crate::target_kind::TargetKindSet::POW),
    PhaseMask::CORE | PhaseMask::POST,
    importance: crate::step::ImportanceLevel::Low,
    |ctx, expr| {
        let rewrite =
            cas_math::power_eval_support::try_rewrite_negative_exponent_normalization_expr(
                ctx, expr,
            )?;
        Some(Rewrite::new(rewrite.rewritten).desc(format_power_eval_static_desc(rewrite.kind)))
    }
);

define_rule!(
    EvaluatePowerRule,
    RULE_EVALUATE_NUMERIC_POWER,
    Some(crate::target_kind::TargetKindSet::POW),
    PhaseMask::CORE | PhaseMask::POST,
    importance: crate::step::ImportanceLevel::Low,
    |ctx, expr| {
        if let Some(rewrite) =
            cas_math::power_eval_support::try_rewrite_literal_power_eval_expr(ctx, expr)
        {
            return Some(Rewrite::new(rewrite.rewritten).desc("Evaluate literal power"));
        }
        let rewrite = cas_math::power_eval_support::try_rewrite_evaluate_power_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);
