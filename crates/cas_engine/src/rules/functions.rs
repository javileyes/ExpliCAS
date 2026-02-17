use crate::define_rule;
use crate::helpers::as_pow;
use crate::phase::PhaseMask;
use crate::rule::Rewrite;
use cas_ast::{BuiltinFn, Expr};
use num_integer::Integer;
use num_traits::Signed;

/// Extract the inner argument of `abs(x)`, returning `Some(x)` if the expression
/// is an absolute value call, `None` otherwise.
fn try_unwrap_abs(ctx: &cas_ast::Context, id: cas_ast::ExprId) -> Option<cas_ast::ExprId> {
    if let Expr::Function(fn_id, args) = ctx.get(id) {
        if ctx.is_builtin(*fn_id, BuiltinFn::Abs) && args.len() == 1 {
            return Some(args[0]);
        }
    }
    None
}

define_rule!(EvaluateAbsRule, "Evaluate Absolute Value", |ctx, expr| {
    if let Expr::Function(fn_id, args) = ctx.get(expr) {
        if ctx.is_builtin(*fn_id, BuiltinFn::Abs) && args.len() == 1 {
            let arg = args[0];

            // Case 1: abs(number)
            if let Expr::Number(n) = ctx.get(arg) {
                let n_abs = n.abs();
                let desc = format!("abs({}) = {}", n, n_abs);
                let abs_val = ctx.add(Expr::Number(n_abs));
                return Some(Rewrite::new(abs_val).desc(desc));
            }

            // Case 2: abs(-x) -> abs(x)
            if let Expr::Neg(inner) = ctx.get(arg) {
                // If inner is a number, we can simplify fully: abs(-5) -> 5
                if let Expr::Number(n) = ctx.get(*inner) {
                    let n_clone = n.clone();
                    let desc = format!("abs(-{}) = {}", n, n);
                    let abs_val = ctx.add(Expr::Number(n_clone));
                    return Some(Rewrite::new(abs_val).desc(desc));
                }

                let abs_inner = ctx.call_builtin(cas_ast::BuiltinFn::Abs, vec![*inner]);
                return Some(Rewrite::new(abs_inner).desc("abs(-x) = abs(x)"));
            }
        }
    }
    None
});

/// V2.14.20: Simplify absolute value under positivity
/// |x| → x when x > 0 is proven or assumed (depending on DomainMode)
pub struct AbsPositiveSimplifyRule;

impl crate::rule::Rule for AbsPositiveSimplifyRule {
    fn name(&self) -> &str {
        "Abs Under Positivity"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: cas_ast::ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<crate::rule::Rewrite> {
        use crate::domain::{DomainMode, Proof};
        use crate::helpers::prove_positive;

        // Match abs(inner)
        let inner = match ctx.get(expr) {
            Expr::Function(fn_id, args)
                if ctx.is_builtin(*fn_id, BuiltinFn::Abs) && args.len() == 1 =>
            {
                args[0]
            }
            _ => return None,
        };

        let vd = parent_ctx.value_domain();
        let dm = parent_ctx.domain_mode();
        let pos = prove_positive(ctx, inner, vd);

        match dm {
            DomainMode::Strict | DomainMode::Generic => {
                // Only simplify if proven positive or implied by global requires
                let is_implied = if pos != Proof::Proven {
                    // V2.14.21: Check if Positive(inner) is implied by global requires
                    // V2.15: Use cached implicit_domain to avoid stack overflow from redundant computation
                    if let Some(id) = parent_ctx.implicit_domain() {
                        let dc = crate::implicit_domain::DomainContext::new(
                            id.conditions().iter().cloned().collect(),
                        );
                        let cond = crate::implicit_domain::ImplicitCondition::Positive(inner);
                        dc.is_condition_implied(ctx, &cond)
                    } else {
                        false
                    }
                } else {
                    true
                };

                if !is_implied {
                    return None;
                }
                // V2.14.20: .local(abs_id, inner_id) to capture correct step focus
                Some(
                    Rewrite::new(inner)
                        .desc("|x| = x for x > 0")
                        .local(expr, inner),
                )
            }
            DomainMode::Assume => {
                // In Assume mode: if proven, no warning; if not, emit assumption
                if pos == Proof::Proven {
                    Some(
                        Rewrite::new(inner)
                            .desc("|x| = x for x > 0")
                            .local(expr, inner),
                    )
                } else {
                    // Emit positive_assumed warning
                    // V2.14.20: .local(abs_id, inner_id) to capture correct step focus
                    Some(
                        Rewrite::new(inner)
                            .desc("|x| = x (assuming x > 0)")
                            .local(expr, inner)
                            .assume(crate::assumptions::AssumptionEvent::positive_assumed(
                                ctx, inner,
                            )),
                    )
                }
            }
        }
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::FUNCTION)
    }

    // V2.14.20: Run in POST phase only so |a| created by LogPowerRule exists first
    fn allowed_phases(&self) -> crate::phase::PhaseMask {
        crate::phase::PhaseMask::POST
    }

    // V2.14.21: Ensure step is visible - domain simplification is didactically important
    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::High
    }
}

/// Simplify absolute value under non-negativity
/// |x| → x when x ≥ 0 is proven or implied (e.g., from sqrt(x) requirements)
/// This complements AbsPositiveSimplifyRule for the non-strict case
pub struct AbsNonNegativeSimplifyRule;

impl crate::rule::Rule for AbsNonNegativeSimplifyRule {
    fn name(&self) -> &str {
        "Abs Under Non-Negativity"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: cas_ast::ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<crate::rule::Rewrite> {
        use crate::domain::{DomainMode, Proof};
        use crate::helpers::prove_nonnegative;

        // Match abs(inner)
        let inner = match ctx.get(expr) {
            Expr::Function(fn_id, args)
                if ctx.is_builtin(*fn_id, BuiltinFn::Abs) && args.len() == 1 =>
            {
                args[0]
            }
            _ => return None,
        };

        let vd = parent_ctx.value_domain();
        let dm = parent_ctx.domain_mode();
        let nonneg = prove_nonnegative(ctx, inner, vd);

        match dm {
            DomainMode::Strict | DomainMode::Generic => {
                // Check if proven non-negative directly
                let mut is_implied = nonneg == Proof::Proven;

                // If not proven directly, check implicit domain (from sqrt requirements, etc.)
                // V2.15.8: Uses sticky domain from parent_ctx, set by orchestrator pipeline
                if !is_implied {
                    if let Some(id) = parent_ctx.implicit_domain() {
                        let dc = crate::implicit_domain::DomainContext::new(
                            id.conditions().iter().cloned().collect(),
                        );
                        let cond = crate::implicit_domain::ImplicitCondition::NonNegative(inner);
                        is_implied = dc.is_condition_implied(ctx, &cond);
                    }
                }

                if !is_implied {
                    return None;
                }

                Some(
                    Rewrite::new(inner)
                        .desc("|x| = x for x ≥ 0")
                        .local(expr, inner),
                )
            }
            DomainMode::Assume => {
                // In Assume mode: if proven, no warning; if not, emit assumption
                if nonneg == Proof::Proven {
                    Some(
                        Rewrite::new(inner)
                            .desc("|x| = x for x ≥ 0")
                            .local(expr, inner),
                    )
                } else {
                    // Emit non_negative assumed warning
                    Some(
                        Rewrite::new(inner)
                            .desc("|x| = x (assuming x ≥ 0)")
                            .local(expr, inner)
                            .assume(crate::assumptions::AssumptionEvent::nonnegative(ctx, inner)),
                    )
                }
            }
        }
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::FUNCTION)
    }

    // Run in POST phase only, after abs values from other rules exist
    fn allowed_phases(&self) -> crate::phase::PhaseMask {
        crate::phase::PhaseMask::POST
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::High
    }
}

/// AbsSquaredRule: |x|^(2k) → x^(2k) for even integer k
///
/// This rule simplifies absolute values raised to even powers since the result
/// is always non-negative. However, we SKIP this transformation when the parent
/// is a logarithm (ln, log) because it would prevent the more educational
/// transformation ln(|x|^n) → n·ln(|x|).
///
/// V2.15.9: Converted from define_rule! to structured Rule to access parent_ctx.
pub struct AbsSquaredRule;

impl crate::rule::Rule for AbsSquaredRule {
    fn name(&self) -> &str {
        "Abs Squared Identity"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: cas_ast::ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<crate::rule::Rewrite> {
        // V2.15.9: Skip if parent is ln or log to allow LogAbsPowerRule to apply first
        if let Some(parent_id) = parent_ctx.immediate_parent() {
            if let Expr::Function(name, _) = ctx.get(parent_id) {
                if ctx.is_builtin(*name, BuiltinFn::Ln) || ctx.is_builtin(*name, BuiltinFn::Log) {
                    return None;
                }
            }
        }

        let (base, exp) = as_pow(ctx, expr)?;
        {
            let (fn_id, args) = match ctx.get(base) {
                Expr::Function(fn_id, args) => (*fn_id, args.clone()),
                _ => return None,
            };
            if ctx.is_builtin(fn_id, BuiltinFn::Abs) && args.len() == 1 {
                let inner = args[0];

                // Check if exponent is an even integer
                if let Expr::Number(n) = ctx.get(exp) {
                    if n.is_integer() && n.to_integer().is_even() {
                        // abs(x)^even -> x^even
                        let desc = format!("|x|^{} = x^{}", n, n);
                        let new_expr = ctx.add(Expr::Pow(inner, exp));
                        return Some(Rewrite::new(new_expr).desc(desc));
                    }
                }
            }
        }
        None
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::POW)
    }

    fn allowed_phases(&self) -> PhaseMask {
        PhaseMask::CORE | PhaseMask::TRANSFORM | PhaseMask::RATIONALIZE
    }
}

define_rule!(
    SimplifySqrtSquareRule,
    "Simplify Square Root of Square",
    |ctx, expr| {
        // sqrt(x^2) -> |x|
        // Also handles Pow(x, 1/2) of Pow(x, 2)

        let inner = if let Expr::Function(fn_id, args) = ctx.get(expr) {
            if ctx.is_builtin(*fn_id, BuiltinFn::Sqrt) && args.len() == 1 {
                Some(args[0])
            } else {
                None
            }
        } else if let Expr::Pow(b, e) = ctx.get(expr) {
            // Check if exponent is 1/2
            if let Expr::Number(n) = ctx.get(*e) {
                if *n.numer() == 1.into() && *n.denom() == 2.into() {
                    Some(*b)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        if let Some(inner) = inner {
            // Check if inner is x^2 (or x^even_integer)
            if let Expr::Pow(base, exp) = ctx.get(inner) {
                if let Expr::Number(n) = ctx.get(*exp) {
                    // Check if exponent is 2
                    if n.is_integer() && *n == num_rational::BigRational::from_integer(2.into()) {
                        // sqrt(base^2) -> |base|
                        let abs_base = ctx.call_builtin(cas_ast::BuiltinFn::Abs, vec![*base]);
                        return Some(Rewrite::new(abs_base).desc("sqrt(x^2) = |x|"));
                    }
                }
            }
        }
        None
    }
);

// SimplifySqrtOddPowerRule: x^(n/2) -> |x|^k * sqrt(x) where n = 2k+1 (odd >= 3)
// Works on canonicalized form: sqrt(x^3) becomes x^(3/2) before reaching this rule
// Examples:
//   x^(3/2) -> |x| * sqrt(x)     (n=3, k=1)
//   x^(5/2) -> |x|^2 * sqrt(x)   (n=5, k=2)
//   x^(7/2) -> |x|^3 * sqrt(x)   (n=7, k=3)
define_rule!(
    SimplifySqrtOddPowerRule,
    "Simplify Odd Half-Integer Power",
    Some(crate::target_kind::TargetKindSet::POW), // Only match Pow expressions
    PhaseMask::POST, // Run in POST phase after canonicalization is done
    |ctx, expr| {
        use num_traits::ToPrimitive;

        // Match Pow(base, exp) where exp = n/2 with n odd >= 3
        let (base, k) = if let Expr::Pow(b, e) = ctx.get(expr) {
            let base = *b;
            if let Expr::Number(exp) = ctx.get(*e) {
                // Check if exp = n/2 where n is odd integer >= 3
                // That means denom = 2 and numer is odd >= 3
                let numer = exp.numer().to_i64()?;
                let denom = exp.denom().to_i64()?;

                if denom == 2 && numer >= 3 && numer % 2 == 1 {
                    // n = numer, k = (n-1)/2
                    let k = (numer - 1) / 2;
                    (Some(base), Some((numer, k)))
                } else {
                    (None, None)
                }
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };

        if let (Some(base), Some((n, k))) = (base, k) {
            // Build: |x|^k * sqrt(x)
            let abs_base = ctx.call_builtin(cas_ast::BuiltinFn::Abs, vec![base]);
            let sqrt_base = ctx.call_builtin(cas_ast::BuiltinFn::Sqrt, vec![base]);

            let result = if k == 1 {
                // |x| * sqrt(x)
                ctx.add(Expr::Mul(abs_base, sqrt_base))
            } else {
                // |x|^k * sqrt(x)
                let k_expr = ctx.num(k);
                let abs_pow_k = ctx.add(Expr::Pow(abs_base, k_expr));
                ctx.add(Expr::Mul(abs_pow_k, sqrt_base))
            };

            return Some(
                Rewrite::new(result).desc_lazy(|| format!("x^({}/2) = |x|^{} * √x", n, k)),
            );
        }

        None
    }
);

// ============================================================================
// SymbolicRootCancelRule: sqrt(x^n, n) → x when n is symbolic (Assume mode only)
// ============================================================================
//
// V2.14.45: When the index is symbolic (not a numeric literal), we can't
// determine parity to decide between x and |x|. In Assume mode, we simplify
// to x with the assumption x ≥ 0 (which makes both even and odd cases equivalent).
//
// CONTRACT: sqrt(x, n) / root(x, n) semantics assume n is a POSITIVE INTEGER.
// This is the standard mathematical definition of n-th root where n ∈ ℤ⁺.
// We do NOT emit requires for n ≠ 0 or n > 0 because this is implicit in the
// root function's domain definition.
//
// - Generic/Strict: block (handled by keeping sqrt form in CanonicalizeRootRule)
// - Assume: sqrt(x^n, n) → x with Requires: x ≥ 0
// ============================================================================
pub struct SymbolicRootCancelRule;

impl crate::rule::Rule for SymbolicRootCancelRule {
    fn name(&self) -> &str {
        "Symbolic Root Cancel"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: cas_ast::ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<crate::rule::Rewrite> {
        use crate::domain::DomainMode;
        use crate::semantics::ValueDomain;

        // Only apply in Assume mode (Generic keeps sqrt form)
        if parent_ctx.domain_mode() != DomainMode::Assume {
            return None;
        }

        // Only apply in RealOnly (complex has different semantics)
        if parent_ctx.value_domain() != ValueDomain::RealOnly {
            return None;
        }

        // Match sqrt(arg, index) where arg = Pow(base, exp) and exp == index
        let (fn_id, args) = match ctx.get(expr) {
            Expr::Function(fn_id, args) => (*fn_id, args.clone()),
            _ => return None,
        };

        if !ctx.is_builtin(fn_id, BuiltinFn::Sqrt) || args.len() != 2 {
            return None;
        }

        let arg = args[0];
        let index = args[1];

        // Index must be symbolic (non-numeric) - numeric cases handled elsewhere
        if matches!(ctx.get(index), Expr::Number(_)) {
            return None;
        }

        // Arg must be Pow(base, exp)
        let (base, exp) = as_pow(ctx, arg)?;

        // Check if exp == index (structural equality)
        if crate::ordering::compare_expr(ctx, exp, index) != std::cmp::Ordering::Equal {
            return None;
        }

        // Pattern matched: sqrt(base^n, n) with symbolic n
        // In Assume mode: return base with x ≥ 0 assumption
        use crate::implicit_domain::ImplicitCondition;
        Some(
            crate::rule::Rewrite::new(base)
                .desc("sqrt(x^n, n) = x (assuming x ≥ 0)")
                .requires(ImplicitCondition::NonNegative(base))
                .assume(crate::assumptions::AssumptionEvent::nonnegative(ctx, base)),
        )
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::FUNCTION)
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::High
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rule::Rule;
    use cas_ast::Context;
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    #[test]
    fn test_evaluate_abs() {
        let mut ctx = Context::new();
        let rule = EvaluateAbsRule;

        // abs(-5) -> 5
        // Note: Parser might produce Number(-5) or Neg(Number(5)).
        // Our parser likely produces Number(-5) for literals.
        let expr1 = parse("abs(-5)", &mut ctx).expect("Failed to parse abs(-5)");
        let rewrite1 = rule
            .apply(
                &mut ctx,
                expr1,
                &crate::parent_context::ParentContext::root(),
            )
            .expect("Rule failed to apply");
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite1.new_expr
                }
            ),
            "5"
        );

        // abs(5) -> 5
        let expr2 = parse("abs(5)", &mut ctx).expect("Failed to parse abs(5)");
        let rewrite2 = rule
            .apply(
                &mut ctx,
                expr2,
                &crate::parent_context::ParentContext::root(),
            )
            .expect("Rule failed to apply");
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite2.new_expr
                }
            ),
            "5"
        );

        // abs(-x) -> abs(x)
        let expr3 = parse("abs(-x)", &mut ctx).expect("Failed to parse abs(-x)");
        let rewrite3 = rule
            .apply(
                &mut ctx,
                expr3,
                &crate::parent_context::ParentContext::root(),
            )
            .expect("Rule failed to apply");
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite3.new_expr
                }
            ),
            "|x|"
        );
    }
}

// EvaluateMetaFunctionsRule: Handles meta functions that operate on expressions
// - simplify(expr) → expr (already simplified by bottom-up processing)
// - factor(expr) → expr (factoring is done by other rules during simplification)
// - expand(expr) → expanded version (calls actual expand logic)
define_rule!(
    EvaluateMetaFunctionsRule,
    "Evaluate Meta Functions",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    |ctx, expr| {
        if let Expr::Function(fn_id, args) = ctx.get(expr) {
            let (fn_id, args) = (*fn_id, args.clone());
            if args.len() == 1 {
                let arg = args[0];
                let fn_name = ctx.sym_name(fn_id);
                match fn_name {
                    // simplify() is transparent - argument already processed
                    "simplify" => {
                        return Some(Rewrite::new(arg).desc("simplify(x) = x (already processed)"));
                    }
                    // factor() calls actual factorization logic
                    "factor" => {
                        let factored = crate::factor::factor(ctx, arg);
                        if factored != arg {
                            return Some(Rewrite::new(factored).desc("factor(x) → factored form"));
                        }
                        // No change - return as-is (irreducible)
                        return Some(Rewrite::new(arg).desc("factor(x) = x (irreducible)"));
                    }
                    // expand() needs to call actual expansion logic
                    "expand" => {
                        let expanded = crate::expand::expand(ctx, arg);
                        return Some(Rewrite::new(expanded).desc("expand(x) → expanded form"));
                    }
                    // expand_log is handled in eval.rs BEFORE simplification to ensure
                    // goal=ExpandedLog is set before any rules run
                    _ => {}
                }
            }
        }
        None
    }
);

// =============================================================================
// Abs Idempotent Rule: ||x|| → |x|
// Absolute value of absolute value is just absolute value
// =============================================================================
define_rule!(AbsIdempotentRule, "Abs Idempotent", |ctx, expr| {
    // Match abs(abs(inner))
    if let Expr::Function(fn_id, args) = ctx.get(expr) {
        if ctx.is_builtin(*fn_id, BuiltinFn::Abs) && args.len() == 1 {
            let arg = args[0];
            if let Expr::Function(inner_name, inner_args) = ctx.get(arg) {
                if ctx.is_builtin(*inner_name, BuiltinFn::Abs) && inner_args.len() == 1 {
                    // ||x|| → |x|
                    return Some(Rewrite::new(arg).desc("||x|| = |x|"));
                }
            }
        }
    }
    None
});

// =============================================================================
// Abs Of Even Power Rule: |x^(2k)| → x^(2k)
// Absolute value of even power is just the even power (always non-negative)
// =============================================================================
define_rule!(AbsOfEvenPowerRule, "Abs Of Even Power", |ctx, expr| {
    // Match abs(x^n) where n is even integer
    if let Expr::Function(fn_id, args) = ctx.get(expr) {
        if ctx.is_builtin(*fn_id, BuiltinFn::Abs) && args.len() == 1 {
            let arg = args[0];
            if let Expr::Pow(_base, exp) = ctx.get(arg) {
                if let Expr::Number(n) = ctx.get(*exp) {
                    if n.is_integer() && n.to_integer().is_even() {
                        // |x^(2k)| → x^(2k) since x^(2k) ≥ 0 always
                        return Some(
                            Rewrite::new(arg).desc_lazy(|| format!("|x^{}| = x^{}", n, n)),
                        );
                    }
                }
            }
        }
    }
    None
});

// =============================================================================
// Abs Pow Odd Integer Rule: |x^n| → |x|^n for positive odd integer n
// This canonicalizes the abs-power form so that `abs(x^5)` and `abs(x)^5`
// converge to the same AST, enabling structural cancellation in the solver.
// Even powers are handled by AbsOfEvenPowerRule (|x^2k| → x^2k).
// =============================================================================
define_rule!(
    AbsPowOddIntegerRule,
    "Abs Distribute Over Odd Power",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    PhaseMask::CORE | PhaseMask::TRANSFORM,
    |ctx, expr| {
        // Match abs(x^n) where n is a positive odd integer
        if let Expr::Function(fn_id, args) = ctx.get(expr) {
            if ctx.is_builtin(*fn_id, BuiltinFn::Abs) && args.len() == 1 {
                let arg = args[0];
                if let Expr::Pow(base, exp) = ctx.get(arg) {
                    let (base, exp) = (*base, *exp);
                    if let Expr::Number(n) = ctx.get(exp) {
                        if n.is_integer() {
                            let n_int = n.to_integer();
                            // Positive odd integer only (even handled by AbsOfEvenPowerRule)
                            if n_int > num_bigint::BigInt::from(0) && !n_int.is_even() {
                                // abs(x^n) → abs(x)^n
                                let abs_base =
                                    ctx.call_builtin(cas_ast::BuiltinFn::Abs, vec![base]);
                                let result = ctx.add(Expr::Pow(abs_base, exp));
                                return Some(
                                    Rewrite::new(result)
                                        .desc_lazy(|| format!("|x^{}| = |x|^{}", n_int, n_int)),
                                );
                            }
                        }
                    }
                }
            }
        }
        None
    }
);

// =============================================================================
// Abs Product Rule: |x| * |y| → |x * y|
// Multiplicative property of absolute value
// =============================================================================
define_rule!(
    AbsProductRule,
    "Abs Product",
    Some(crate::target_kind::TargetKindSet::MUL),
    PhaseMask::CORE | PhaseMask::TRANSFORM,
    |ctx, expr| {
        use crate::build::mul2_raw;

        // Match Mul(abs(a), abs(b))
        if let Expr::Mul(lhs, rhs) = ctx.get(expr) {
            if let (Some(a), Some(b)) = (try_unwrap_abs(ctx, *lhs), try_unwrap_abs(ctx, *rhs)) {
                // |a| * |b| → |a * b|
                let product = mul2_raw(ctx, a, b);
                let abs_product = ctx.call_builtin(cas_ast::BuiltinFn::Abs, vec![product]);
                return Some(Rewrite::new(abs_product).desc("|x|·|y| = |x·y|"));
            }
        }
        None
    }
);

// =============================================================================
// Abs Quotient Rule: |x| / |y| → |x / y|
// Quotient property of absolute value
// =============================================================================
define_rule!(
    AbsQuotientRule,
    "Abs Quotient",
    Some(crate::target_kind::TargetKindSet::DIV),
    PhaseMask::CORE | PhaseMask::TRANSFORM,
    |ctx, expr| {
        // Match Div(abs(a), abs(b))
        if let Expr::Div(lhs, rhs) = ctx.get(expr) {
            if let (Some(a), Some(b)) = (try_unwrap_abs(ctx, *lhs), try_unwrap_abs(ctx, *rhs)) {
                // |a| / |b| → |a / b|
                let quotient = ctx.add(Expr::Div(a, b));
                let abs_quotient = ctx.call_builtin(cas_ast::BuiltinFn::Abs, vec![quotient]);
                return Some(Rewrite::new(abs_quotient).desc("|x| / |y| = |x / y|"));
            }
        }
        None
    }
);

// =============================================================================
// Abs Sqrt Rule: |sqrt(x)| → sqrt(x)
// Square root is always non-negative (when it exists in reals)
// =============================================================================
define_rule!(AbsSqrtRule, "Abs Of Sqrt", |ctx, expr| {
    // Match abs(sqrt(x)) or abs(x^(1/2))
    if let Expr::Function(fn_id, args) = ctx.get(expr) {
        if ctx.is_builtin(*fn_id, BuiltinFn::Abs) && args.len() == 1 {
            let arg = args[0];

            // Check for sqrt(x) function
            if let Expr::Function(inner_name, _inner_args) = ctx.get(arg) {
                if ctx.is_builtin(*inner_name, BuiltinFn::Sqrt) {
                    // |sqrt(x)| → sqrt(x)
                    return Some(Rewrite::new(arg).desc("|√x| = √x"));
                }
            }

            // Check for x^(1/2) form
            if let Expr::Pow(_base, exp) = ctx.get(arg) {
                if let Expr::Number(n) = ctx.get(*exp) {
                    // Check if exponent is 1/2
                    if n.numer() == &num_bigint::BigInt::from(1)
                        && n.denom() == &num_bigint::BigInt::from(2)
                    {
                        // |x^(1/2)| → x^(1/2)
                        return Some(Rewrite::new(arg).desc("|√x| = √x"));
                    }
                }
            }
        }
    }
    None
});

// =============================================================================
// Abs Exp Rule: |e^x| → e^x
// Exponential is always positive
// =============================================================================
define_rule!(AbsExpRule, "Abs Of Exp", |ctx, expr| {
    // Match abs(exp(x)) or abs(e^x)
    if let Expr::Function(fn_id, args) = ctx.get(expr) {
        if ctx.is_builtin(*fn_id, BuiltinFn::Abs) && args.len() == 1 {
            let arg = args[0];

            // Check for exp(x) function
            if let Expr::Function(inner_name, _inner_args) = ctx.get(arg) {
                if ctx.is_builtin(*inner_name, BuiltinFn::Exp) {
                    // |exp(x)| → exp(x)
                    return Some(Rewrite::new(arg).desc("|e^x| = e^x"));
                }
            }

            // Check for e^x form (Pow with base = Constant(E))
            if let Expr::Pow(base, _exp) = ctx.get(arg) {
                if let Expr::Constant(c) = ctx.get(*base) {
                    if matches!(c, cas_ast::Constant::E) {
                        // |e^x| → e^x
                        return Some(Rewrite::new(arg).desc("|e^x| = e^x"));
                    }
                }
            }
        }
    }
    None
});

// =============================================================================
// Abs Sum Of Squares Rule: |x² + y²| → x² + y²
// Sum of squares is always non-negative
// =============================================================================
define_rule!(AbsSumOfSquaresRule, "Abs Of Sum Of Squares", |ctx, expr| {
    // Match abs(a + b) where both a and b are non-negative (squares, abs, etc.)
    if let Expr::Function(fn_id, args) = ctx.get(expr) {
        if ctx.is_builtin(*fn_id, BuiltinFn::Abs) && args.len() == 1 {
            let arg = args[0];

            // Check if the argument is provably non-negative
            if is_sum_of_nonnegative(ctx, arg) {
                return Some(Rewrite::new(arg).desc("|x² + ...| = x² + ..."));
            }
        }
    }
    None
});

/// Helper: Check if an expression is a sum of non-negative terms
fn is_sum_of_nonnegative(ctx: &cas_ast::Context, expr: cas_ast::ExprId) -> bool {
    match ctx.get(expr) {
        // x² is non-negative
        Expr::Pow(_, exp) => {
            if let Expr::Number(n) = ctx.get(*exp) {
                n.is_integer() && n.to_integer().is_even()
            } else {
                false
            }
        }
        // |x| is non-negative
        Expr::Function(name, _) if ctx.is_builtin(*name, BuiltinFn::Abs) => true,
        // sqrt(x) is non-negative
        Expr::Function(name, _) if ctx.is_builtin(*name, BuiltinFn::Sqrt) => true,
        // exp(x) is positive
        Expr::Function(name, _) if ctx.is_builtin(*name, BuiltinFn::Exp) => true,
        // Positive number is non-negative
        Expr::Number(n) => !n.is_negative(),
        // Sum: both sides must be non-negative
        Expr::Add(l, r) => is_sum_of_nonnegative(ctx, *l) && is_sum_of_nonnegative(ctx, *r),
        // Product of two non-negatives is non-negative
        Expr::Mul(l, r) => is_sum_of_nonnegative(ctx, *l) && is_sum_of_nonnegative(ctx, *r),
        _ => false,
    }
}

// =============================================================================
// Abs Sub Normalize Rule: |a - b| → |b - a|
// Canonicalize the argument of abs(Sub(..)) so that |a-b| and |b-a| produce
// the same normal form, enabling cancellation of |u-1| - |1-u| → 0.
// Uses compare_expr ordering: if a > b in canonical order, swap to |b-a|.
//
// V2.16: Relaxed from atoms-only to compound expressions with dedup node cap.
// This enables convergence for cases like |sin(u)-1| vs |1-sin(u)|.
// Guards: per-operand ≤ 20 dedup nodes, total abs expr ≤ 60 dedup nodes.
// =============================================================================
define_rule!(
    AbsSubNormalizeRule,
    "Abs Sub Normalize",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    |ctx, expr| {
        if let Expr::Function(fn_id, args) = ctx.get(expr) {
            if ctx.is_builtin(*fn_id, BuiltinFn::Abs) && args.len() == 1 {
                let arg = args[0];
                // Try to detect (a - b) in either Sub(a,b) or Add(a,Neg(b))
                // or Add(x, Number(-k)) form
                let pair = match ctx.get(arg) {
                    Expr::Sub(a, b) => Some((*a, *b)),
                    Expr::Add(l, r) => {
                        let (l, r) = (*l, *r);
                        // Add(a, Neg(b)) = a - b
                        if let Expr::Neg(x) = ctx.get(r) {
                            Some((l, *x))
                        // Add(Neg(b), a) = a - b
                        } else if let Expr::Neg(x) = ctx.get(l) {
                            Some((r, *x))
                        // Add(x, Number(-k)) = x - k where k > 0
                        } else if let Expr::Number(n) = ctx.get(r) {
                            if n.is_negative() {
                                let pos_k = ctx.add(Expr::Number(-n.clone()));
                                Some((l, pos_k))
                            } else {
                                None
                            }
                        // Add(Number(-k), x) = x - k where k > 0
                        } else if let Expr::Number(n) = ctx.get(l) {
                            if n.is_negative() {
                                let pos_k = ctx.add(Expr::Number(-n.clone()));
                                Some((r, pos_k))
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    }
                    _ => None,
                };
                if let Some((a, b)) = pair {
                    // Guard: cap operand size to avoid expensive compare_expr
                    // on huge expressions. Uses dedup node count (HashSet<ExprId>)
                    // to avoid over-counting shared DAG nodes.
                    let dedup_count = |id: cas_ast::ExprId| -> usize {
                        let mut seen = std::collections::HashSet::new();
                        let mut stack = vec![id];
                        while let Some(nid) = stack.pop() {
                            if !seen.insert(nid) {
                                continue;
                            }
                            match ctx.get(nid) {
                                Expr::Add(l, r)
                                | Expr::Sub(l, r)
                                | Expr::Mul(l, r)
                                | Expr::Div(l, r)
                                | Expr::Pow(l, r) => {
                                    stack.push(*l);
                                    stack.push(*r);
                                }
                                Expr::Neg(e) | Expr::Hold(e) => stack.push(*e),
                                Expr::Function(_, fargs) => stack.extend(fargs),
                                _ => {}
                            }
                        }
                        seen.len()
                    };

                    let a_nodes = dedup_count(a);
                    let b_nodes = dedup_count(b);

                    // Per-operand cap: ≤ 20 dedup nodes each
                    if a_nodes > 20 || b_nodes > 20 {
                        return None;
                    }

                    // Total abs expression cap: ≤ 60 dedup nodes
                    let total_nodes = dedup_count(expr);
                    if total_nodes > 60 {
                        return None;
                    }

                    // If a > b in canonical ordering, swap to |b - a|
                    if crate::ordering::compare_expr(ctx, a, b) == std::cmp::Ordering::Greater {
                        let swapped = ctx.add(Expr::Sub(b, a));
                        let new_abs = ctx.call_builtin(BuiltinFn::Abs, vec![swapped]);
                        return Some(Rewrite::new(new_abs).desc("|a−b| = |b−a|"));
                    }
                }
            }
        }
        None
    }
);

// =============================================================================
// Abs Numeric Factor Rule: |k·x| → |k|·|x| for any nonzero numeric k
// Extracts numeric factors from absolute value (both positive and negative).
// Examples: |2u| → 2|u|,  |(-3)·sin(x)| → 3·|sin(x)|
// =============================================================================
define_rule!(
    AbsPositiveFactorRule,
    "Abs Positive Factor",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    |ctx, expr| {
        if let Expr::Function(fn_id, args) = ctx.get(expr) {
            if ctx.is_builtin(*fn_id, BuiltinFn::Abs) && args.len() == 1 {
                let arg = args[0];
                if let Expr::Mul(l, r) = ctx.get(arg) {
                    let (l, r) = (*l, *r);
                    // Check if left factor is a number
                    if let Expr::Number(n) = ctx.get(l) {
                        if n.is_positive() {
                            // |k·x| → k·|x| for k > 0
                            let abs_r = ctx.call_builtin(BuiltinFn::Abs, vec![r]);
                            let result = ctx.add(Expr::Mul(l, abs_r));
                            return Some(Rewrite::new(result).desc("|k·x| = k·|x| for k > 0"));
                        } else if n.is_negative() {
                            // |(-k)·x| → |k|·|x| for k < 0
                            let abs_k = ctx.add(Expr::Number(n.abs()));
                            let abs_r = ctx.call_builtin(BuiltinFn::Abs, vec![r]);
                            let result = ctx.add(Expr::Mul(abs_k, abs_r));
                            return Some(Rewrite::new(result).desc("|(-k)·x| = |k|·|x| for k < 0"));
                        }
                    }
                    // Check if right factor is a number
                    if let Expr::Number(n) = ctx.get(r) {
                        if n.is_positive() {
                            // |x·k| → k·|x| for k > 0
                            let abs_l = ctx.call_builtin(BuiltinFn::Abs, vec![l]);
                            let result = ctx.add(Expr::Mul(r, abs_l));
                            return Some(Rewrite::new(result).desc("|x·k| = k·|x| for k > 0"));
                        } else if n.is_negative() {
                            // |x·(-k)| → |k|·|x| for k < 0
                            let abs_k = ctx.add(Expr::Number(n.abs()));
                            let abs_l = ctx.call_builtin(BuiltinFn::Abs, vec![l]);
                            let result = ctx.add(Expr::Mul(abs_k, abs_l));
                            return Some(Rewrite::new(result).desc("|x·(-k)| = |k|·|x| for k < 0"));
                        }
                    }
                }
            }
        }
        None
    }
);

pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(SimplifySqrtSquareRule)); // Must go BEFORE EvaluateAbsRule to catch sqrt(x^2) early
                                                           // V2.14.45: SimplifySqrtOddPowerRule DISABLED - causes split/merge cycle with ProductPowerRule
                                                           // x^(5/2) → |x|²*√x is a "worsening" transformation (increases AST nodes).
                                                           // The canonical form for odd half-integer powers is Pow(x, n/2), NOT the product form.
                                                           // If visual "extracted square" form is desired, it belongs in a renderer or explain-mode.
                                                           // simplifier.add_rule(Box::new(SimplifySqrtOddPowerRule)); // sqrt(x^3) -> |x| * sqrt(x)
    simplifier.add_rule(Box::new(SymbolicRootCancelRule)); // V2.14.45: sqrt(x^n, n) -> x in Assume mode
    simplifier.add_rule(Box::new(EvaluateAbsRule));
    simplifier.add_rule(Box::new(AbsPositiveSimplifyRule)); // V2.14.20: |x| -> x when x > 0
    simplifier.add_rule(Box::new(AbsNonNegativeSimplifyRule)); // |x| -> x when x >= 0 (from sqrt requirements)
    simplifier.add_rule(Box::new(AbsSquaredRule));
    simplifier.add_rule(Box::new(AbsIdempotentRule)); // ||x|| → |x|
    simplifier.add_rule(Box::new(AbsOfEvenPowerRule)); // |x^2k| → x^2k
    simplifier.add_rule(Box::new(AbsPowOddIntegerRule)); // |x^n| → |x|^n (odd n)
    simplifier.add_rule(Box::new(AbsProductRule)); // |x|*|y| → |xy|
    simplifier.add_rule(Box::new(AbsQuotientRule)); // |x|/|y| → |x/y|
    simplifier.add_rule(Box::new(AbsSqrtRule)); // |sqrt(x)| → sqrt(x)
    simplifier.add_rule(Box::new(AbsExpRule)); // |e^x| → e^x
    simplifier.add_rule(Box::new(AbsSumOfSquaresRule)); // |x² + y²| → x² + y²
    simplifier.add_rule(Box::new(AbsSubNormalizeRule)); // |a-b| → |b-a| (canonical)
    simplifier.add_rule(Box::new(AbsPositiveFactorRule)); // |k·x| → k·|x| for k > 0
    simplifier.add_rule(Box::new(EvaluateMetaFunctionsRule)); // Make simplify/factor/expand transparent
}
