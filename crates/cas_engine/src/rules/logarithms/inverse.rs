use crate::define_rule;
use crate::helpers::{as_add, as_div, as_mul, as_pow};
use crate::ordering::compare_expr;
use crate::rule::Rewrite;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::expr_extract::{
    extract_log_base_argument, extract_log_base_argument_relaxed_view,
    extract_log_base_argument_view,
};
use cas_math::expr_predicates::is_e_constant_expr;
use cas_math::expr_rewrite::smart_mul;
use cas_math::logarithm_inverse_support::{
    collect_mul_factors, estimate_log_terms, is_log, normalize_to_power, simplify_exp_log,
};
use num_traits::{One, Zero};
use std::cmp::Ordering;

// Re-use helpers from parent module
use super::make_log;

/// Domain-aware rule for b^log(b, x) → x.
/// Requires x > 0 (domain of log). Respects domain_mode.
pub struct ExponentialLogRule;

impl crate::rule::Rule for ExponentialLogRule {
    fn name(&self) -> &str {
        "Exponential-Log Inverse"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: cas_ast::ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<crate::rule::Rewrite> {
        let (base, exp) = as_pow(ctx, expr)?;
        {
            // Helper to get log base and arg
            let get_log_parts = |ctx: &mut cas_ast::Context,
                                 e_id: cas_ast::ExprId|
             -> Option<(cas_ast::ExprId, cas_ast::ExprId)> {
                extract_log_base_argument(ctx, e_id)
            };

            // Case 1: b^log(b, x) → x
            // The condition x > 0 is IMPLICIT from ln(x)/log(b,x) being defined.
            // This is NOT a new assumption - it's already required by the expression.
            if let Some((log_base, log_arg)) = get_log_parts(ctx, exp) {
                if compare_expr(ctx, log_base, base) == Ordering::Equal {
                    let mode = parent_ctx.domain_mode();
                    let vd = parent_ctx.value_domain();

                    // Use prove_positive with ValueDomain
                    let arg_positive = crate::helpers::prove_positive(ctx, log_arg, vd);

                    // In Strict mode: only allow if proven
                    if mode == crate::domain::DomainMode::Strict
                        && arg_positive != crate::domain::Proof::Proven
                    {
                        return None;
                    }

                    // In Generic/Assume: allow with implicit requires
                    // The condition x > 0 is ALREADY implied by log(b, x) existing.
                    // This is like sqrt(x)^2 → x with requires x ≥ 0.
                    use crate::implicit_domain::ImplicitCondition;

                    if arg_positive == crate::domain::Proof::Proven {
                        // Already proven positive, no requires needed
                        return Some(crate::rule::Rewrite::new(log_arg).desc("b^log(b, x) = x"));
                    }

                    // Emit implicit requires (like sqrt(x)^2 → x)
                    return Some(
                        crate::rule::Rewrite::new(log_arg)
                            .desc("b^log(b, x) = x")
                            .requires(ImplicitCondition::Positive(log_arg)),
                    );
                }
            }

            // Case 2: b^(-log(b, x)) → x^(-1) = 1/x
            // Handles Neg(ln(x)) pattern that Mul(-1, ln(x)) might not match
            if let Expr::Neg(neg_inner) = ctx.get(exp) {
                let neg_inner = *neg_inner;
                if let Some((log_base, log_arg)) = get_log_parts(ctx, neg_inner) {
                    if compare_expr(ctx, log_base, base) == Ordering::Equal {
                        let mode = parent_ctx.domain_mode();
                        let vd = parent_ctx.value_domain();
                        let arg_positive = crate::helpers::prove_positive(ctx, log_arg, vd);

                        if mode == crate::domain::DomainMode::Strict
                            && arg_positive != crate::domain::Proof::Proven
                        {
                            return None;
                        }

                        let neg_one = ctx.num(-1);
                        let result = ctx.add(Expr::Pow(log_arg, neg_one));

                        use crate::implicit_domain::ImplicitCondition;
                        if arg_positive == crate::domain::Proof::Proven {
                            return Some(
                                crate::rule::Rewrite::new(result).desc("b^(-log(b, x)) = 1/x"),
                            );
                        }
                        return Some(
                            crate::rule::Rewrite::new(result)
                                .desc("b^(-log(b, x)) = 1/x")
                                .requires(ImplicitCondition::Positive(log_arg)),
                        );
                    }
                }
            }

            // Case 3: b^(c * log(b, x)) → x^c
            // Same logic as Case 1: x > 0 is IMPLICIT from log(b, x) existing.
            if let Some((lhs, rhs)) = as_mul(ctx, exp) {
                let vd = parent_ctx.value_domain();
                let mode = parent_ctx.domain_mode();

                let mut check_log = |target: cas_ast::ExprId,
                                     coeff: cas_ast::ExprId|
                 -> Option<crate::rule::Rewrite> {
                    if let Some((log_base, log_arg)) = get_log_parts(ctx, target) {
                        if compare_expr(ctx, log_base, base) == Ordering::Equal {
                            // Use prove_positive with ValueDomain
                            let arg_positive = crate::helpers::prove_positive(ctx, log_arg, vd);

                            // In Strict mode: only allow if proven
                            if mode == crate::domain::DomainMode::Strict
                                && arg_positive != crate::domain::Proof::Proven
                            {
                                return None;
                            }

                            let new_expr = ctx.add(Expr::Pow(log_arg, coeff));

                            // In Generic/Assume: allow with implicit requires
                            use crate::implicit_domain::ImplicitCondition;

                            if arg_positive == crate::domain::Proof::Proven {
                                return Some(
                                    crate::rule::Rewrite::new(new_expr)
                                        .desc("b^(c*log(b, x)) = x^c"),
                                );
                            }

                            return Some(
                                crate::rule::Rewrite::new(new_expr)
                                    .desc("b^(c*log(b, x)) = x^c")
                                    .requires(ImplicitCondition::Positive(log_arg)),
                            );
                        }
                    }
                    None
                };

                if let Some(rw) = check_log(lhs, rhs) {
                    return Some(rw);
                }
                if let Some(rw) = check_log(rhs, lhs) {
                    return Some(rw);
                }
            }
        }
        None
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::POW)
    }

    fn solve_safety(&self) -> crate::solve_safety::SolveSafety {
        // Intrinsic: the condition x > 0 is already guaranteed by ln(x)/log(b,x)
        // being present in the input expression. This is inherited, not introduced.
        crate::solve_safety::SolveSafety::IntrinsicCondition(
            crate::assumptions::ConditionClass::Analytic,
        )
    }
}

define_rule!(
    SplitLogExponentsRule,
    "Split Log Exponents",
    solve_safety: crate::solve_safety::SolveSafety::NeedsCondition(
        crate::assumptions::ConditionClass::Analytic
    ),
    |ctx, expr, _parent_ctx| {
    // e^(a + b) -> e^a * e^b IF a or b is a log
    let (base, exp) = as_pow(ctx, expr)?;
    {
        let base_is_e = is_e_constant_expr(ctx, base);
        if base_is_e {
            if let Some((lhs, rhs)) = as_add(ctx, exp) {
                let lhs_is_log = is_log(ctx, lhs);
                let rhs_is_log = is_log(ctx, rhs);

                if lhs_is_log || rhs_is_log {
                    let term1 = simplify_exp_log(ctx, base, lhs);
                    let term2 = simplify_exp_log(ctx, base, rhs);
                    let new_expr = smart_mul(ctx, term1, term2);
                    return Some(Rewrite::new(new_expr).desc("e^(a+b) -> e^a * e^b (log cancellation)"));
                }
            }
        }
    }
    None
});

define_rule!(
    LogInversePowerRule,
    "Log Inverse Power",
    solve_safety: crate::solve_safety::SolveSafety::NeedsCondition(
        crate::assumptions::ConditionClass::Analytic
    ),
    |ctx, expr, _parent_ctx| {
    let (base, exp) = as_pow(ctx, expr)?;
    {
        // Check for x^(c / log(b, x))
        // exp could be Div(c, log(b, x)) or Mul(c, Pow(log(b, x), -1))

        // Returns Some(Some(base)) for log(b, x), Some(None) for ln(x) -> base e
        let check_log_denom =
            |ctx: &Context, denom: cas_ast::ExprId| -> Option<Option<cas_ast::ExprId>> {
                if let Some((base_opt, log_arg)) = extract_log_base_argument_view(ctx, denom) {
                    if compare_expr(ctx, log_arg, base) == Ordering::Equal {
                        return Some(base_opt);
                    }
                }
                None
            };

        let mut target_b_opt: Option<Option<cas_ast::ExprId>> = None;
        let mut coeff: Option<cas_ast::ExprId> = None;

        if let Some((num, den)) = as_div(ctx, exp) {
            if let Some(b_opt) = check_log_denom(ctx, den) {
                target_b_opt = Some(b_opt);
                coeff = Some(num);
            }
        } else if let Some((l, r)) = as_mul(ctx, exp) {
                // Check l * r^-1
                if let Expr::Pow(b, e) = ctx.get(r) {
                    if let Expr::Number(n) = ctx.get(*e) {
                        if n.is_integer()
                            && *n == num_rational::BigRational::from_integer((-1).into())
                        {
                            if let Some(b_opt) = check_log_denom(ctx, *b) {
                                target_b_opt = Some(b_opt);
                                coeff = Some(l);
                            }
                        }
                    }
                }
                // Check r * l^-1
                if target_b_opt.is_none() {
                    if let Expr::Pow(b, e) = ctx.get(l) {
                        if let Expr::Number(n) = ctx.get(*e) {
                            if n.is_integer()
                                && *n == num_rational::BigRational::from_integer((-1).into())
                            {
                                if let Some(b_opt) = check_log_denom(ctx, *b) {
                                    target_b_opt = Some(b_opt);
                                    coeff = Some(r);
                                }
                            }
                        }
                    }
                }
        } else if let Some((b, e)) = as_pow(ctx, exp) {
            // Check if it's log(b, x)^-1
            if let Expr::Number(n) = ctx.get(e) {
                if n.is_integer() && *n == num_rational::BigRational::from_integer((-1).into())
                {
                    if let Some(b_opt) = check_log_denom(ctx, b) {
                        target_b_opt = Some(b_opt);
                        coeff = Some(ctx.num(1));
                    }
                }
            }
        }

        if let (Some(b_opt), Some(c)) = (target_b_opt, coeff) {
            // Result is b^c
            let b = b_opt.unwrap_or_else(|| ctx.add(Expr::Constant(cas_ast::Constant::E)));
            let new_expr = ctx.add(Expr::Pow(b, c));
            return Some(Rewrite::new(new_expr).desc("x^(c/log(b, x)) = b^c"));
        }
    }
    None
});

/// Domain-aware rule for log(b, b^x) → x.
/// Variable exponents only simplify when domain_mode is NOT strict.
/// Numeric exponents (like log(x, x^2) → 2) always apply.
/// This is controlled by domain_mode because it's a domain assumption (x is real),
/// not an inverse trig composition.
pub struct LogExpInverseRule;

impl crate::rule::Rule for LogExpInverseRule {
    fn name(&self) -> &str {
        "Log-Exp Inverse"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: cas_ast::ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<crate::rule::Rewrite> {
        let fn_id = match ctx.get(expr) {
            Expr::Function(fn_id, _) => *fn_id,
            _ => return None,
        };
        {
            // Handle ln(x) as log(e, x), or log(b, x)
            let (base, arg) = match ctx.builtin_of(fn_id) {
                Some(BuiltinFn::Ln) | Some(BuiltinFn::Log) => extract_log_base_argument(ctx, expr)?,
                _ => return None,
            };

            let (p_base, p_exp) = if let Expr::Pow(b, e) = ctx.get(arg) {
                (*b, *e)
            } else {
                return None;
            };

            // log(b, b^x) → x (when b matches)
            {
                if p_base == base || ctx.get(p_base) == ctx.get(base) {
                    // For numeric exponents like log(x, x^2) → 2, always simplify
                    let is_numeric_exponent = matches!(ctx.get(p_exp), Expr::Number(_));

                    if is_numeric_exponent {
                        // Always safe: log(b, b^n) = n for any numeric n
                        return Some(crate::rule::Rewrite::new(p_exp).desc("log(b, b^n) = n"));
                    } else {
                        // For variable exponents like log(e, e^x) → x
                        //
                        // NEW CONTRACT (RealOnly = symbols are real):
                        // - RealOnly: e^x > 0 for all x ∈ ℝ, so ln(e^x) = x ALWAYS.
                        //   This applies even in Strict mode (no assumption needed).
                        // - ComplexEnabled: ln is multivalued. ln(e^x) = x + 2πik.
                        //   NEVER simplify for symbolic exponents (would require principal branch).
                        //
                        // GATE: For bases other than e, require prove_positive(base) and base ≠ 1
                        // log(b, b^x) = x only when b > 0 AND b ≠ 1
                        //
                        use crate::domain::Proof;
                        use crate::helpers::prove_positive;
                        use crate::semantics::ValueDomain;
                        let vd = parent_ctx.value_domain();

                        if vd == ValueDomain::ComplexEnabled {
                            // ComplexEnabled: Never simplify symbolic exponents
                            // (ln is multivalued, can't assume principal branch)
                            return None;
                        }

                        // RealOnly: Check if base is provably valid (>0 and ≠1)
                        let is_e_base = is_e_constant_expr(ctx, base);

                        if !is_e_base {
                            // For non-e bases, require prove_positive(base) == Proven
                            let base_positive = prove_positive(ctx, base, vd);
                            if base_positive != Proof::Proven {
                                // Cannot prove base > 0
                                let dm = parent_ctx.domain_mode();
                                match dm {
                                    crate::domain::DomainMode::Strict
                                    | crate::domain::DomainMode::Generic => {
                                        // Don't simplify if can't prove base > 0
                                        return None;
                                    }
                                    crate::domain::DomainMode::Assume => {
                                        // Allow with assumption warning
                                        // Require base > 0 and base ≠ 1 (use base - 1 ≠ 0)
                                        let one = ctx.num(1);
                                        let base_minus_1 = ctx.add(Expr::Sub(base, one));
                                        return Some(
                                            crate::rule::Rewrite::new(p_exp)
                                                .desc("log(b, b^x) → x")
                                                .assume(
                                                    crate::assumptions::AssumptionEvent::positive_assumed(
                                                        ctx, base,
                                                    ),
                                                )
                                                .requires(crate::implicit_domain::ImplicitCondition::NonZero(base_minus_1)),
                                        );
                                    }
                                }
                            }
                            // Check base ≠ 1 (log_1 is undefined)
                            if let Expr::Number(n) = ctx.get(base) {
                                if *n == num_rational::BigRational::from_integer(1.into()) {
                                    return None; // log base 1 is undefined
                                }
                            }
                        }

                        // RealOnly with valid base (proven positive): Always simplify
                        // Still need to require base ≠ 1 for log to be defined (use base - 1 ≠ 0)
                        let one = ctx.num(1);
                        let base_minus_1 = ctx.add(Expr::Sub(base, one));
                        return Some(
                            crate::rule::Rewrite::new(p_exp)
                                .desc("log(b, b^x) → x")
                                .requires(crate::implicit_domain::ImplicitCondition::NonZero(
                                    base_minus_1,
                                )),
                        );
                    }
                }
            }
        }
        None
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::FUNCTION)
    }
}

/// Rule for log(a^m, a^n) → n/m
///
/// Handles cases like:
/// - log(x^2, x^6) → 6/2 = 3
/// - log(1/x, x) → log(x^(-1), x^1) → 1/(-1) = -1
///
/// Normalizes bases and arguments to power form:
/// - a → (a, 1)
/// - a^m → (a, m)
/// - 1/a → (a, -1)
pub struct LogPowerBaseRule;

impl crate::rule::Rule for LogPowerBaseRule {
    fn name(&self) -> &str {
        "Log Power Base"
    }

    fn soundness(&self) -> crate::rule::SoundnessLabel {
        crate::rule::SoundnessLabel::EquivalenceUnderIntroducedRequires
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: cas_ast::ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<crate::rule::Rewrite> {
        let (fn_id, args) = match ctx.get(expr) {
            Expr::Function(fn_id, args) => (*fn_id, args.clone()),
            _ => return None,
        };
        {
            // Match log(base, arg) - not ln (which has implicit base e)
            if ctx.builtin_of(fn_id) != Some(BuiltinFn::Log) || args.len() != 2 {
                return None;
            }
            let base = args[0];
            let arg = args[1];

            // Normalize base to (core, exponent) form
            let (base_core, base_exp) = normalize_to_power(ctx, base);
            // Normalize arg to (core, exponent) form
            let (arg_core, arg_exp) = normalize_to_power(ctx, arg);

            // Both must have the same core, and base_exp must not be 0 or 1
            // (if base_exp = 0, base = a^0 = 1, undefined log)
            // (if base_exp = 1, this is just log(a, a^n) → n, handled by LogExpInverseRule)
            if base_core == arg_core || compare_expr(ctx, base_core, arg_core) == Ordering::Equal {
                // Check base_exp is not 0 or 1 (to avoid overlapping with other rules)
                let base_exp_is_one = matches!(ctx.get(base_exp), Expr::Number(n) if n.is_one());
                if base_exp_is_one {
                    // log(a, a^n) → n is handled by LogExpInverseRule
                    return None;
                }

                // Check both exponents are numeric (for now, start conservative)
                let base_exp_num = match ctx.get(base_exp) {
                    Expr::Number(n) => Some(n.clone()),
                    _ => None,
                };
                let arg_exp_num = match ctx.get(arg_exp) {
                    Expr::Number(n) => Some(n.clone()),
                    _ => None,
                };

                if let (Some(m), Some(n)) = (base_exp_num, arg_exp_num) {
                    // Check m ≠ 0 (log base a^0 = 1 is undefined)
                    if m.is_zero() {
                        return None;
                    }

                    // Result: n/m  (clone for description building)
                    let m_disp = m.clone();
                    let n_disp = n.clone();
                    let result_ratio = n / m;
                    let result = ctx.add(Expr::Number(result_ratio.clone()));

                    // Domain requires: a > 0, a ≠ 1
                    use crate::implicit_domain::ImplicitCondition;
                    let one = ctx.num(1);

                    // Gate by domain mode
                    use crate::domain::{DomainMode, Proof};
                    use crate::helpers::prove_positive;
                    use crate::semantics::ValueDomain;

                    let vd = parent_ctx.value_domain();
                    if vd == ValueDomain::ComplexEnabled {
                        // Complex domain: don't simplify
                        return None;
                    }

                    let dm = parent_ctx.domain_mode();
                    let base_positive = prove_positive(ctx, base_core, vd);

                    // For numeric exponents, the identity log(a^m, a^n) = n/m is ALGEBRAICALLY VALID
                    // The domain restrictions (a > 0, a^m ≠ 1) are already implied by the log being defined.
                    // In Generic mode, we can apply without proving a > 0, since the input expression
                    // already requires those conditions to be meaningful.
                    // We only block if we're in Strict mode and can't prove positivity.
                    match dm {
                        DomainMode::Strict => {
                            if base_positive != Proof::Proven {
                                // Cannot prove a > 0, block in Strict
                                return None;
                            }
                            // Also check a ≠ 1
                            if compare_expr(ctx, base_core, one) == Ordering::Equal {
                                return None;
                            }
                        }
                        DomainMode::Generic | DomainMode::Assume => {
                            // For numeric exponents: algebraically valid, proceed
                            // The log existence already implies the domain conditions
                        }
                    }

                    // Build description using cloned exponents
                    let desc = format!(
                        "log(a^{}, a^{}) = {}/{} = {}",
                        m_disp, n_disp, n_disp, m_disp, result_ratio
                    );

                    let mut rewrite = crate::rule::Rewrite::new(result).desc(desc);

                    // Add requires in Assume mode (or always to be explicit)
                    if dm == DomainMode::Assume && base_positive != Proof::Proven {
                        rewrite = rewrite.requires(ImplicitCondition::Positive(base_core));
                    }
                    // base ≠ 1 (log base 1 is undefined) - use base - 1 ≠ 0
                    let base_minus_1 = ctx.add(Expr::Sub(base, one));
                    rewrite = rewrite.requires(ImplicitCondition::NonZero(base_minus_1));

                    return Some(rewrite);
                }
            }
        }
        None
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::FUNCTION)
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::Medium
    }

    fn priority(&self) -> i32 {
        // Higher than LogEvenPowerWithChainedAbsRule (10) to match log(x^2, x^6) first
        // Otherwise LogEvenPower would expand to 6·log(x², |x|) before we can simplify to 3
        15
    }
}

// ============================================================================
// Auto Expand Log Rule with ExpandBudget Integration
// ============================================================================

// NOTE: Local is_provably_positive was removed in V2.15.9.
// Use crate::helpers::prove_positive instead, which handles:
// - base > 0 → base^(p/q) > 0 (RealOnly)
// - sqrt(x) > 0 when x > 0
// - exp(x) > 0 in RealOnly
// - etc.

/// AutoExpandLogRule: Automatically expand log(a*b) -> log(a) + log(b) during simplify
/// when log_expand_policy = Auto and the expansion passes budget checks.
///
/// This rule uses domain gating:
/// - Assume mode: expands with HeuristicAssumption (⚠️) for a>0, b>0
/// - Generic mode: blocks and registers hint if positivity not proven
/// - Strict mode: blocks without hint
pub struct AutoExpandLogRule;

impl crate::rule::Rule for AutoExpandLogRule {
    fn name(&self) -> &'static str {
        "AutoExpandLogRule"
    }

    fn apply(
        &self,
        ctx: &mut Context,
        expr: ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        // GATE: Expand if global auto-expand mode OR inside a marked cancellation context
        // This mirrors AutoExpandPowSumRule behavior exactly
        let in_expand_context = parent_ctx.in_auto_expand_context();
        if !(parent_ctx.is_auto_expand() || in_expand_context) {
            return None;
        }

        // Match log(arg), log(base, arg), or ln(arg)
        let (_, arg) = extract_log_base_argument_relaxed_view(ctx, expr)?;

        // Check if expandable and get term estimates
        let (base_terms, gen_terms, pow_exp) = estimate_log_terms(ctx, arg)?;

        // Get budget - use default if in context but no explicit budget set
        let default_budget = crate::phase::ExpandBudget::default();
        let budget = parent_ctx.auto_expand_budget().unwrap_or(&default_budget);

        // Budget check
        if !budget.allows_log_expansion(base_terms, gen_terms, pow_exp) {
            return None;
        }

        // Don't expand if it wouldn't help (gen_terms <= 1)
        if gen_terms <= 1 {
            return None;
        }

        // Get domain mode from parent context
        let domain_mode = parent_ctx.domain_mode();

        // For Generic/Strict mode, we need to check if factors are provably positive
        // For Assume mode, we proceed and emit HeuristicAssumption events
        match domain_mode {
            crate::domain::DomainMode::Strict => {
                // In Strict, never auto-expand unless proven
                let factors = collect_mul_factors(ctx, arg);
                let vd = parent_ctx.value_domain();
                let all_positive = factors
                    .iter()
                    .all(|&f| crate::helpers::prove_positive(ctx, f, vd).is_proven());
                if !all_positive {
                    return None; // Block silently in Strict
                }
                // Expand without assumption events (proven)
                expand_log_for_rule(ctx, expr, arg, &[])
            }
            crate::domain::DomainMode::Generic => {
                // In Generic, block if not proven AND not implied by global requires
                let factors = collect_mul_factors(ctx, arg);

                // V2.14.21: Before blocking, check if each factor's positivity is
                // implied by global requires (e.g., b^3 > 0 is implied by b > 0)
                // V2.15: Use cached implicit_domain if available, fallback to computation
                let vd = parent_ctx.value_domain();
                let implicit_domain: Option<crate::implicit_domain::ImplicitDomain> =
                    parent_ctx.implicit_domain().cloned().or_else(|| {
                        parent_ctx.root_expr().map(|root| {
                            crate::implicit_domain::infer_implicit_domain(ctx, root, vd)
                        })
                    });

                let mut unproven_factor: Option<ExprId> = None;
                for &factor in &factors {
                    // V2.15.9: Use canonical prove_positive which handles:
                    // - base > 0 → base^(p/q) > 0 (RealOnly)
                    // - sqrt(x) > 0 when x > 0
                    // - etc.
                    let vd = parent_ctx.value_domain();
                    if crate::helpers::prove_positive(ctx, factor, vd).is_proven() {
                        continue; // Algebraically proven
                    }

                    // Check if Positive(factor) is implied by global requires
                    let cond = crate::implicit_domain::ImplicitCondition::Positive(factor);
                    let is_implied = implicit_domain.as_ref().is_some_and(|id| {
                        // Create a temporary DomainContext to use is_condition_implied
                        let dc = crate::implicit_domain::DomainContext::new(
                            id.conditions().iter().cloned().collect(),
                        );
                        dc.is_condition_implied(ctx, &cond)
                    });

                    if !is_implied {
                        unproven_factor = Some(factor);
                        break;
                    }
                }

                if let Some(factor) = unproven_factor {
                    // Register blocked hint for user feedback
                    let hint = crate::domain::BlockedHint {
                        key: crate::assumptions::AssumptionKey::Positive {
                            expr_fingerprint: crate::assumptions::expr_fingerprint(ctx, factor),
                        },
                        expr_id: factor,
                        rule: "AutoExpandLogRule".to_string(),
                        suggestion: "Use 'semantics set domain assume' to enable log expansion.",
                    };
                    crate::domain::register_blocked_hint(hint);
                    return None;
                }

                // All factors proven or implied positive, expand without events
                expand_log_for_rule(ctx, expr, arg, &[])
            }
            crate::domain::DomainMode::Assume => {
                // In Assume mode, expand and emit HeuristicAssumption events
                let factors = collect_mul_factors(ctx, arg);
                let vd = parent_ctx.value_domain();
                let mut events = Vec::new();
                for &factor in &factors {
                    if !crate::helpers::prove_positive(ctx, factor, vd).is_proven() {
                        events.push(crate::assumptions::AssumptionEvent::positive_assumed(
                            ctx, factor,
                        ));
                    }
                }
                expand_log_for_rule(ctx, expr, arg, &events)
            }
        }
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::FUNCTION)
    }

    fn allowed_phases(&self) -> crate::phase::PhaseMask {
        // Same as AutoExpandPowSumRule: CORE, TRANSFORM, RATIONALIZE
        crate::phase::PhaseMask::CORE
            | crate::phase::PhaseMask::TRANSFORM
            | crate::phase::PhaseMask::RATIONALIZE
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        // Didactically important: users should see log expansions
        crate::step::ImportanceLevel::Medium
    }
}

/// Perform the log expansion for AutoExpandLogRule.
fn expand_log_for_rule(
    ctx: &mut Context,
    _original: ExprId,
    arg: ExprId,
    events: &[crate::assumptions::AssumptionEvent],
) -> Option<Rewrite> {
    // Get base (ln = natural log, log with 1 arg = base 10 sentinel).
    let (base_opt, _) = extract_log_base_argument_relaxed_view(ctx, _original)?;
    let base = match base_opt {
        Some(base) => base,
        None => match ctx.get(_original) {
            Expr::Function(fn_id, _) if ctx.is_builtin(*fn_id, BuiltinFn::Ln) => {
                ctx.add(Expr::Constant(cas_ast::Constant::E))
            }
            Expr::Function(fn_id, _) if ctx.is_builtin(*fn_id, BuiltinFn::Log) => {
                // log with 1 arg = base 10, use sentinel
                ExprId::from_raw(u32::MAX - 1)
            }
            _ => return None,
        },
    };

    match ctx.get(arg) {
        Expr::Mul(_, _) => {
            // Expand log(a*b*c) -> log(a) + log(b) + log(c)
            let factors = collect_mul_factors(ctx, arg);
            if factors.len() <= 1 {
                return None;
            }

            let mut sum = make_log(ctx, base, factors[0]);
            for &factor in &factors[1..] {
                let log_f = make_log(ctx, base, factor);
                sum = ctx.add(Expr::Add(sum, log_f));
            }

            let mut rewrite = Rewrite::new(sum).desc("Auto-expand log product");
            for event in events {
                rewrite = rewrite.assume(event.clone());
            }
            Some(rewrite)
        }
        Expr::Div(num, den) => {
            let (num, den) = (*num, *den);
            // Expand log(a/b) -> log(a) - log(b)
            let log_num = make_log(ctx, base, num);
            let log_den = make_log(ctx, base, den);
            let result = ctx.add(Expr::Sub(log_num, log_den));

            let mut rewrite = Rewrite::new(result).desc("Auto-expand log quotient");
            for event in events {
                rewrite = rewrite.assume(event.clone());
            }
            Some(rewrite)
        }
        Expr::Pow(pow_base, exp) => {
            let (pow_base, exp) = (*pow_base, *exp);
            // Expand log(u^n) -> n * log(u)
            let log_base = make_log(ctx, base, pow_base);
            let result = smart_mul(ctx, exp, log_base);

            let mut rewrite = Rewrite::new(result).desc("Auto-expand log power");
            for event in events {
                rewrite = rewrite.assume(event.clone());
            }
            Some(rewrite)
        }
        _ => None,
    }
}
