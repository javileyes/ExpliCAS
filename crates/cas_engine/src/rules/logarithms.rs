use crate::define_rule;
use crate::ordering::compare_expr;
use crate::rule::Rewrite;
use crate::rules::algebra::helpers::smart_mul;
use cas_ast::{Context, Expr, ExprId};
use num_traits::{One, Zero};
use std::cmp::Ordering;

/// Helper: create log(base, arg) or ln(arg) depending on base.
/// If base is Constant::E, returns ln(arg) to preserve natural log notation.
/// If base is the sentinel for log10 (u32::MAX - 1), returns log(arg) with 1 arg.
fn make_log(ctx: &mut Context, base: ExprId, arg: ExprId) -> ExprId {
    // Check for log10 sentinel first (before accessing ctx.get which would panic)
    let sentinel_log10 = ExprId::from_raw(u32::MAX - 1);
    if base == sentinel_log10 {
        return ctx.call("log", vec![arg]);
    }
    if let Expr::Constant(cas_ast::Constant::E) = ctx.get(base) {
        ctx.call("ln", vec![arg])
    } else {
        ctx.call("log", vec![base, arg])
    }
}

/// Evaluate log_base(val) as a rational number using prime factorization.
///
/// Returns Some(ratio) if both base and val are powers of a common prime base:
/// - log(2, 8) = 3     because 2 = 2^1, 8 = 2^3 → ratio = 3/1
/// - log(8, 2) = 1/3   because 8 = 2^3, 2 = 2^1 → ratio = 1/3
/// - log(16, 8) = 3/4  because 16 = 2^4, 8 = 2^3 → ratio = 3/4
///
/// Returns None if the log cannot be expressed as a rational number.
fn eval_log_rational(
    base: &num_bigint::BigInt,
    val: &num_bigint::BigInt,
) -> Option<num_rational::BigRational> {
    use num_bigint::BigInt;
    use num_rational::BigRational;
    use num_traits::{One, Signed, Zero};

    // Guard: base > 1, val > 0
    let one = BigInt::one();
    if base <= &one || val.is_zero() || val.is_negative() || base.is_negative() {
        return None;
    }

    // Special case: val = 1 → log_b(1) = 0
    if val.is_one() {
        return Some(BigRational::zero());
    }

    // Special case: base == val → log_b(b) = 1
    if base == val {
        return Some(BigRational::one());
    }

    // Factorize both into prime -> exponent maps
    let fb = prime_exponent_map(base);
    let fv = prime_exponent_map(val);

    // Both must have the same set of primes
    if fb.keys().collect::<std::collections::HashSet<_>>()
        != fv.keys().collect::<std::collections::HashSet<_>>()
    {
        return None;
    }

    // The ratio exp_val / exp_base must be identical for all primes
    // log_base(val) = log(val) / log(base) = (sum of exp_val[p] * log(p)) / (sum of exp_base[p] * log(p))
    // This only works if exp_val[p] / exp_base[p] is constant for all p
    let mut ratio: Option<BigRational> = None;
    for (prime, exp_base) in &fb {
        let exp_val = fv.get(prime)?;
        if exp_base.is_zero() {
            return None; // Shouldn't happen, but guard
        }
        let r = BigRational::new(BigInt::from(*exp_val), BigInt::from(*exp_base));
        match &ratio {
            None => ratio = Some(r),
            Some(prev) if *prev == r => {}
            _ => return None, // Different ratios for different primes
        }
    }

    ratio
}

/// Compute prime factorization as a map of prime -> exponent
fn prime_exponent_map(n: &num_bigint::BigInt) -> HashMap<num_bigint::BigInt, u32> {
    use num_bigint::BigInt;
    use num_integer::Integer;
    use num_traits::{One, Signed, Zero};

    let mut result = HashMap::new();
    let mut n = n.abs();
    let one = BigInt::one();

    if n <= one {
        return result;
    }

    // Factor out 2s
    let mut count_2 = 0u32;
    while n.is_even() {
        count_2 += 1;
        n /= 2;
    }
    if count_2 > 0 {
        result.insert(BigInt::from(2), count_2);
    }

    // Factor out odd primes
    let mut d = BigInt::from(3);
    while &d * &d <= n {
        let mut count = 0u32;
        while (&n % &d).is_zero() {
            count += 1;
            n /= &d;
        }
        if count > 0 {
            result.insert(d.clone(), count);
        }
        d += 2;
    }

    // If n > 1 at this point, it's a prime
    if n > one {
        result.insert(n, 1);
    }

    result
}

use std::collections::HashMap;

define_rule!(EvaluateLogRule, "Evaluate Logarithms", |ctx, expr| {
    let expr_data = ctx.get(expr).clone();
    if let Expr::Function(fn_id, args) = expr_data {
        let name = ctx.sym_name(fn_id);
        // Handle ln(x) as log(e, x)
        let (base, arg) = if name == "ln" && args.len() == 1 {
            let e = ctx.add(Expr::Constant(cas_ast::Constant::E));
            (e, args[0])
        } else if name == "log" && args.len() == 2 {
            (args[0], args[1])
        } else {
            return None;
        };

        let arg_data = ctx.get(arg).clone();

        // 1. log(b, 1) = 0, log(b, 0) = -infinity, log(b, neg) = undefined
        if let Expr::Number(n) = &arg_data {
            if n.is_one() {
                let zero = ctx.num(0);
                return Some(Rewrite::new(zero).desc("log(b, 1) = 0"));
            }
            if n.is_zero() {
                let inf = ctx.add(Expr::Constant(cas_ast::Constant::Infinity));
                let neg_inf = ctx.add(Expr::Neg(inf));
                return Some(Rewrite::new(neg_inf).desc("log(b, 0) = -infinity"));
            }
            if *n < num_rational::BigRational::zero() {
                let undef = ctx.add(Expr::Constant(cas_ast::Constant::Undefined));
                return Some(Rewrite::new(undef).desc("log(b, neg) = undefined"));
            }

            // Check if n is a power of base (if base is a number)
            let base_data = ctx.get(base).clone();
            if let Expr::Number(b) = base_data {
                // Try to evaluate log(base, val) as a rational number
                // This handles cases like:
                //   log(2, 8) = 3      (8 = 2^3)
                //   log(8, 2) = 1/3    (2 = 8^(1/3))
                //   log(16, 8) = 3/4   (16 = 2^4, 8 = 2^3, so log_16(8) = 3/4)
                if b.is_integer() && n.is_integer() {
                    let b_int = b.to_integer();
                    let n_int = n.to_integer();
                    if let Some(ratio) = eval_log_rational(&b_int, &n_int) {
                        let new_expr = ctx.add(Expr::Number(ratio.clone()));
                        return Some(
                            Rewrite::new(new_expr).desc(format!("log({}, {}) = {}", b, n, ratio)),
                        );
                    }
                }
            }
        }

        // 2. log(b, b) = 1
        if base == arg || ctx.get(base) == ctx.get(arg) {
            let one = ctx.num(1);
            return Some(Rewrite::new(one).desc("log(b, b) = 1"));
        }

        // 3. log(b, b^x) = x
        // NOTE: This inverse composition is now handled by LogExpInverseRule
        // which respects inv_trig policy (like arctan(tan(x)) → x).
        // Removed from here to avoid unconditional simplification.

        // 4. Expansion: log(b, x^y) = y * log(b, x)
        // Note: This overlaps with rule 3 if x == b. Rule 3 is more specific/simpler, so it should match first.
        // This rule is good for canonicalization.
        // GUARD: When x == b (inverse composition), only simplify if exponent is a number.
        // For variable exponents like log(e, e^x), let LogExpInverseRule handle with policy.
        if let Expr::Pow(p_base, p_exp) = arg_data {
            let is_inverse_composition = p_base == base || ctx.get(p_base) == ctx.get(base);

            if is_inverse_composition {
                // log(b, b^n) where n is a number → n (always safe, like log(x, x^2) → 2)
                if matches!(ctx.get(p_exp), Expr::Number(_)) {
                    return Some(Rewrite::new(p_exp).desc("log(b, b^n) = n"));
                }
                // For variable exponents like log(e, e^x), skip and let LogExpInverseRule handle
                // with inv_trig policy check
            } else {
                // Non-inverse case: log(b, x^y) = y * log(b, x)
                //
                // MATHEMATICAL CORRECTNESS:
                // - For even integer exponents: ln(x^2) = ln(|x|^2) = 2·ln(|x|)
                //   This is valid for all x ≠ 0, no sign assumption needed.
                // - For odd/non-integer exponents: ln(x^y) = y·ln(x) requires x > 0
                //
                // Check if exponent is an even integer
                let is_even_integer = match ctx.get(p_exp) {
                    Expr::Number(n) if n.is_integer() => {
                        let int_val = n.to_integer();
                        let two: num_bigint::BigInt = 2.into();
                        &int_val % &two == 0.into() && int_val != 0.into()
                    }
                    _ => false,
                };

                if is_even_integer {
                    // Even exponent: ln(x^(2k)) = 2k·ln(|x|) - but this introduces abs()
                    //
                    // V2.14.45 ANTI-WORSEN GUARD:
                    // In Generic/Strict: BLOCK - this would introduce abs() without resolution
                    // Let LogEvenPowerWithChainedAbsRule handle this case in Assume mode
                    //
                    // NOTE: EvaluateLogRule uses define_rule! without parent_ctx, so we
                    // delegate to LogEvenPowerWithChainedAbsRule which has proper domain checks.
                    // Just skip this case here - the specialized rule will handle it.
                    return None;
                } else {
                    // Odd or non-integer exponent: requires x > 0
                    // Only allow in Generic/Assume modes, block in Strict
                    let log_inner = make_log(ctx, base, p_base);
                    let new_expr = smart_mul(ctx, p_exp, log_inner);
                    return Some(
                        Rewrite::new(new_expr)
                            .desc("log(b, x^y) = y * log(b, x)")
                            .assume(crate::assumptions::AssumptionEvent::positive_assumed(
                                ctx, p_base,
                            )),
                    );
                }
            }
        }

        // NOTE: Product/quotient expansions (log(xy) = log(x)+log(y), log(x/y) = log(x)-log(y))
        // are moved to LogExpansionRule which has domain_mode + value_domain gates.
        // These expansions require x > 0 and y > 0, and are NOT valid in complex domain with principal branch.
    }
    None
});

// =============================================================================
// LnEProductRule: ln(e*x) → 1 + ln(x)
// =============================================================================
// This is a SAFE, targeted expansion because ln(e) = 1 is a known constant.
// Unlike general LogExpansionRule, this doesn't risk explosion because it only
// extracts the known constant `e` factor.
//
// This enables residuals like `2*ln(e*u) - 2 - 2*ln(u)` to simplify:
// → 2*(1 + ln(u)) - 2 - 2*ln(u) → 2 + 2*ln(u) - 2 - 2*ln(u) → 0
// =============================================================================
define_rule!(LnEProductRule, "Factor e from ln Product", |ctx, expr| {
    // Match ln(arg)
    if let Expr::Function(fn_id, args) = ctx.get(expr).clone() {
        let name = ctx.sym_name(fn_id);
        if name != "ln" || args.len() != 1 {
            return None;
        }
        let arg = args[0];

        // Match Mul(e, x) or Mul(x, e) in the argument
        if let Expr::Mul(l, r) = ctx.get(arg).clone() {
            let l_is_e = matches!(ctx.get(l), Expr::Constant(cas_ast::Constant::E));
            let r_is_e = matches!(ctx.get(r), Expr::Constant(cas_ast::Constant::E));

            // ln(e*x) → 1 + ln(x) or ln(x*e) → 1 + ln(x)
            let other = if l_is_e {
                Some(r)
            } else if r_is_e {
                Some(l)
            } else {
                None
            };

            if let Some(x) = other {
                let one = ctx.num(1);
                let ln_x = ctx.call("ln", vec![x]);
                let result = ctx.add(Expr::Add(one, ln_x));
                return Some(Rewrite::new(result).desc("ln(e*x) = 1 + ln(x)"));
            }
        }
    }
    None
});

// =============================================================================
// LnEDivRule: ln(x/e) → ln(x) - 1, ln(e/x) → 1 - ln(x)
// =============================================================================
// Companion to LnEProductRule. These are SAFE expansions because ln(e) = 1.
// This enables residuals like `2*ln(u/e) - 2*(ln(u)-1)` to simplify to 0.
// =============================================================================
define_rule!(LnEDivRule, "Factor e from ln Quotient", |ctx, expr| {
    // Match ln(arg)
    if let Expr::Function(fn_id, args) = ctx.get(expr).clone() {
        let name = ctx.sym_name(fn_id);
        if name != "ln" || args.len() != 1 {
            return None;
        }
        let arg = args[0];

        // Match Div(x, e) or Div(e, x) in the argument
        if let Expr::Div(num, den) = ctx.get(arg).clone() {
            let num_is_e = matches!(ctx.get(num), Expr::Constant(cas_ast::Constant::E));
            let den_is_e = matches!(ctx.get(den), Expr::Constant(cas_ast::Constant::E));

            if den_is_e && !num_is_e {
                // ln(x/e) → ln(x) - 1
                let ln_x = ctx.call("ln", vec![num]);
                let one = ctx.num(1);
                let result = ctx.add(Expr::Sub(ln_x, one));
                return Some(Rewrite::new(result).desc("ln(x/e) = ln(x) - 1"));
            } else if num_is_e && !den_is_e {
                // ln(e/x) → 1 - ln(x)
                let one = ctx.num(1);
                let ln_x = ctx.call("ln", vec![den]);
                let result = ctx.add(Expr::Sub(one, ln_x));
                return Some(Rewrite::new(result).desc("ln(e/x) = 1 - ln(x)"));
            }
        }
    }
    None
});

/// Domain-aware expansion rule for log products/quotients.
///
/// log(b, x*y) → log(b, x) + log(b, y) and log(b, x/y) → log(b, x) - log(b, y)
///
/// These expansions require:
/// - RealOnly value_domain (complex domain with principal branch: NEVER expand)
/// - Strict: only if prove_positive(x) && prove_positive(y)
/// - Assume: expand with warning if not Disproven
/// - Generic: same as Strict (conservative - no silent assumptions)
pub struct LogExpansionRule;

impl crate::rule::Rule for LogExpansionRule {
    fn name(&self) -> &str {
        "Log Expansion"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: cas_ast::ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<crate::rule::Rewrite> {
        use crate::helpers::prove_positive;
        use crate::semantics::ValueDomain;

        // GATE 1: Never expand in complex domain (principal branch causes 2πi jumps)
        if parent_ctx.value_domain() == ValueDomain::ComplexEnabled {
            return None;
        }

        let expr_data = ctx.get(expr).clone();
        if let Expr::Function(fn_id, args) = expr_data {
            let name = ctx.sym_name(fn_id);
            // Handle ln(x) as log(e, x), or log(b, x)
            let (base, arg) = if name == "ln" && args.len() == 1 {
                let e = ctx.add(Expr::Constant(cas_ast::Constant::E));
                (e, args[0])
            } else if name == "log" && args.len() == 2 {
                (args[0], args[1])
            } else {
                return None;
            };

            let arg_data = ctx.get(arg).clone();
            let mode = parent_ctx.domain_mode();

            // log(b, x*y) → log(b, x) + log(b, y)
            // Requires Positive(x) AND Positive(y) - Analytic class
            if let Expr::Mul(lhs, rhs) = arg_data {
                let vd = parent_ctx.value_domain();
                let lhs_positive = prove_positive(ctx, lhs, vd);
                let rhs_positive = prove_positive(ctx, rhs, vd);

                // Use Analytic gate for each factor
                let lhs_decision = crate::domain::can_apply_analytic(mode, lhs_positive);
                let rhs_decision = crate::domain::can_apply_analytic(mode, rhs_positive);

                // Both must be allowed
                if !lhs_decision.allow || !rhs_decision.allow {
                    return None; // Strict/Generic block if not proven
                }

                let log_lhs = make_log(ctx, base, lhs);
                let log_rhs = make_log(ctx, base, rhs);
                let new_expr = ctx.add(Expr::Add(log_lhs, log_rhs));

                // Build assumption events for unproven factors
                let mut events: smallvec::SmallVec<[crate::assumptions::AssumptionEvent; 2]> =
                    smallvec::SmallVec::new();
                if lhs_decision.assumption.is_some() {
                    events.push(crate::assumptions::AssumptionEvent::positive(ctx, lhs));
                }
                if rhs_decision.assumption.is_some() {
                    events.push(crate::assumptions::AssumptionEvent::positive(ctx, rhs));
                }

                return Some(
                    crate::rule::Rewrite::new(new_expr)
                        .desc("log(b, x*y) = log(b, x) + log(b, y)")
                        .assume_all(events),
                );
            }

            // log(b, x/y) → log(b, x) - log(b, y)
            // Requires Positive(x) AND Positive(y) - Analytic class
            if let Expr::Div(num, den) = arg_data {
                let vd = parent_ctx.value_domain();
                let num_positive = prove_positive(ctx, num, vd);
                let den_positive = prove_positive(ctx, den, vd);

                // Use Analytic gate for each factor
                let num_decision = crate::domain::can_apply_analytic(mode, num_positive);
                let den_decision = crate::domain::can_apply_analytic(mode, den_positive);

                // Both must be allowed
                if !num_decision.allow || !den_decision.allow {
                    return None; // Strict/Generic block if not proven
                }

                let log_num = make_log(ctx, base, num);
                let log_den = make_log(ctx, base, den);
                let new_expr = ctx.add(Expr::Sub(log_num, log_den));

                // Build assumption events for unproven factors
                let mut events: smallvec::SmallVec<[crate::assumptions::AssumptionEvent; 2]> =
                    smallvec::SmallVec::new();
                if num_decision.assumption.is_some() {
                    events.push(crate::assumptions::AssumptionEvent::positive(ctx, num));
                }
                if den_decision.assumption.is_some() {
                    events.push(crate::assumptions::AssumptionEvent::positive(ctx, den));
                }

                return Some(
                    crate::rule::Rewrite::new(new_expr)
                        .desc("log(b, x/y) = log(b, x) - log(b, y)")
                        .assume_all(events),
                );
            }
        }
        None
    }

    fn target_types(&self) -> Option<Vec<&str>> {
        Some(vec!["Function"])
    }

    fn solve_safety(&self) -> crate::solve_safety::SolveSafety {
        crate::solve_safety::SolveSafety::NeedsCondition(
            crate::assumptions::ConditionClass::Analytic,
        )
    }
}

/// Recursively expand logarithms throughout an expression tree.
///
/// This is a specialized expansion function that applies log expansion rules:
/// - log(x*y) → log(x) + log(y)
/// - log(x/y) → log(x) - log(y)
///
/// Returns the expanded expression and any assumption events generated.
/// Used by the `expand_log()` meta-function.
pub fn expand_logs_with_assumptions(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> (cas_ast::ExprId, Vec<crate::assumptions::AssumptionEvent>) {
    let expr_data = ctx.get(expr).clone();
    let mut events = Vec::new();

    let result = match expr_data {
        Expr::Function(name, ref args)
            if {
                let n = ctx.sym_name(name);
                n == "ln" || n == "log"
            } =>
        {
            let name_str = ctx.sym_name(name);
            // Try to expand this log
            // Sentinel: base-10 log uses ExprId::from_raw(u32::MAX - 1) as base indicator
            let sentinel_log10 = cas_ast::ExprId::from_raw(u32::MAX - 1);
            let (base, arg) = if name_str == "ln" && args.len() == 1 {
                let e = ctx.add(Expr::Constant(cas_ast::Constant::E));
                (e, args[0])
            } else if name_str == "log" && args.len() == 2 {
                (args[0], args[1])
            } else if name_str == "log" && args.len() == 1 {
                // log(x) = base-10 log
                (sentinel_log10, args[0])
            } else {
                // Not a recognized log form, recurse into args
                let mut new_args = Vec::with_capacity(args.len());
                for a in args {
                    let (expanded, sub_events) = expand_logs_with_assumptions(ctx, *a);
                    new_args.push(expanded);
                    events.extend(sub_events);
                }
                return (ctx.add(Expr::Function(name, new_args)), events);
            };

            // First expand logs in the argument recursively
            let (expanded_arg, sub_events) = expand_logs_with_assumptions(ctx, arg);
            events.extend(sub_events);
            let arg_data = ctx.get(expanded_arg).clone();

            // Try to expand log(x*y) → log(x) + log(y)
            if let Expr::Mul(lhs, rhs) = arg_data {
                // Add assumption events: lhs > 0 AND rhs > 0
                events.push(crate::assumptions::AssumptionEvent::positive(ctx, lhs));
                events.push(crate::assumptions::AssumptionEvent::positive(ctx, rhs));

                // Create the expanded form
                let log_lhs = make_log(ctx, base, lhs);
                let log_rhs = make_log(ctx, base, rhs);
                let sum = ctx.add(Expr::Add(log_lhs, log_rhs));
                // Recursively expand the result
                let (final_result, more_events) = expand_logs_with_assumptions(ctx, sum);
                events.extend(more_events);
                return (final_result, events);
            }

            // Try to expand log(x/y) → log(x) - log(y)
            if let Expr::Div(num, den) = arg_data {
                // Add assumption events: num > 0 AND den > 0
                events.push(crate::assumptions::AssumptionEvent::positive(ctx, num));
                events.push(crate::assumptions::AssumptionEvent::positive(ctx, den));

                let log_num = make_log(ctx, base, num);
                let log_den = make_log(ctx, base, den);
                let diff = ctx.add(Expr::Sub(log_num, log_den));
                let (final_result, more_events) = expand_logs_with_assumptions(ctx, diff);
                events.extend(more_events);
                return (final_result, events);
            }

            // No expansion possible, return with expanded arg
            make_log(ctx, base, expanded_arg)
        }

        // Recurse through structural nodes
        Expr::Add(l, r) => {
            let (el, le) = expand_logs_with_assumptions(ctx, l);
            let (er, re) = expand_logs_with_assumptions(ctx, r);
            events.extend(le);
            events.extend(re);
            ctx.add(Expr::Add(el, er))
        }
        Expr::Sub(l, r) => {
            let (el, le) = expand_logs_with_assumptions(ctx, l);
            let (er, re) = expand_logs_with_assumptions(ctx, r);
            events.extend(le);
            events.extend(re);
            ctx.add(Expr::Sub(el, er))
        }
        Expr::Mul(l, r) => {
            let (el, le) = expand_logs_with_assumptions(ctx, l);
            let (er, re) = expand_logs_with_assumptions(ctx, r);
            events.extend(le);
            events.extend(re);
            ctx.add(Expr::Mul(el, er))
        }
        Expr::Div(l, r) => {
            let (el, le) = expand_logs_with_assumptions(ctx, l);
            let (er, re) = expand_logs_with_assumptions(ctx, r);
            events.extend(le);
            events.extend(re);
            ctx.add(Expr::Div(el, er))
        }
        Expr::Pow(b, e) => {
            let (eb, be) = expand_logs_with_assumptions(ctx, b);
            let (ee, exp_e) = expand_logs_with_assumptions(ctx, e);
            events.extend(be);
            events.extend(exp_e);
            ctx.add(Expr::Pow(eb, ee))
        }
        Expr::Neg(e) => {
            let (ee, sub_e) = expand_logs_with_assumptions(ctx, e);
            events.extend(sub_e);
            ctx.add(Expr::Neg(ee))
        }
        Expr::Function(name, args) => {
            let mut new_args = Vec::with_capacity(args.len());
            for a in args {
                let (expanded, sub_events) = expand_logs_with_assumptions(ctx, a);
                new_args.push(expanded);
                events.extend(sub_events);
            }
            ctx.add(Expr::Function(name, new_args))
        }
        // Base cases - return as-is
        _ => expr,
    };

    (result, events)
}

/// Simple wrapper that discards assumptions (for backwards compatibility)
pub fn expand_logs(ctx: &mut cas_ast::Context, expr: cas_ast::ExprId) -> cas_ast::ExprId {
    expand_logs_with_assumptions(ctx, expr).0
}
///
/// Domain-aware:
/// - Strict: only if prove_positive(expr) == Proven
/// - Generic: allow (like x/x → 1 in Generic)
/// - Assume: allow with domain_assumption for traceability
///
/// ValueDomain-aware:
/// - ComplexEnabled: only if prove_positive == Proven (no assume for ℂ)
/// - RealOnly: use DomainMode policy
///
/// NOTE: This rule should be registered BEFORE LogContractionRule to catch
/// `ln(|x|) - ln(x)` before it becomes `ln(|x|/x)`.
///
/// V2.14.20: LogEvenPowerWithChainedAbsRule
/// Handles ln(x^even) → even·ln(|x|) with optional ChainedRewrite for |x|→x
/// when x > 0 is provable or in requires.
///
/// This produces TWO contiguous steps:
/// 1. ln(x^even) → even·ln(|x|)
/// 2. |x| → x (if x > 0 provable)
///
/// Priority: higher than EvaluateLogRule to match first.
pub struct LogEvenPowerWithChainedAbsRule;

impl crate::rule::Rule for LogEvenPowerWithChainedAbsRule {
    fn name(&self) -> &str {
        "Log Even Power" // Distinct from EvaluateLogRule for engine registration
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<crate::rule::Rewrite> {
        use crate::domain::{DomainMode, Proof};
        use crate::helpers::prove_positive;
        use crate::rule::ChainedRewrite;

        let expr_data = ctx.get(expr).clone();
        let Expr::Function(name, args) = expr_data else {
            return None;
        };

        // Handle ln(x) or log(base, x)
        let name_str = ctx.sym_name(name);
        let (base, arg) = if name_str == "ln" && args.len() == 1 {
            let e = ctx.add(Expr::Constant(cas_ast::Constant::E));
            (e, args[0])
        } else if name_str == "log" && args.len() == 2 {
            (args[0], args[1])
        } else {
            return None;
        };

        // Match log(base, x^exp) where exp is even integer
        let arg_data = ctx.get(arg).clone();
        let Expr::Pow(p_base, p_exp) = arg_data else {
            return None;
        };

        // Check if base == p_base (inverse composition like log(b, b^x)) - skip
        if p_base == base || ctx.get(p_base) == ctx.get(base) {
            return None;
        }

        // V2.14.20: Also skip if this is ln(exp(...)^n) - let LogExpInverseRule handle first
        // This prevents matching ln(exp(x)) which should simplify to x directly
        if let Expr::Constant(cas_ast::Constant::E) = ctx.get(base) {
            if let Expr::Function(fname, _) = ctx.get(p_base) {
                if ctx.sym_name(*fname) == "exp" {
                    return None;
                }
            }
        }

        // Check if exponent is even integer
        let is_even_integer = match ctx.get(p_exp) {
            Expr::Number(n) if n.is_integer() => {
                let int_val = n.to_integer();
                let two: num_bigint::BigInt = 2.into();
                &int_val % &two == 0.into() && int_val != 0.into()
            }
            _ => false,
        };

        if !is_even_integer {
            return None;
        }

        // Even exponent: ln(x^(2k)) = 2k·ln(|x|)
        let abs_base = ctx.call("abs", vec![p_base]);
        let log_abs = make_log(ctx, base, abs_base);
        let mid_expr = smart_mul(ctx, p_exp, log_abs);

        // Check if we can simplify |x| → x
        let vd = parent_ctx.value_domain();
        let dm = parent_ctx.domain_mode();
        let pos = prove_positive(ctx, p_base, vd);

        // V2.14.21: Check if x > 0 is in global requires using implicit_domain
        // V2.15: Use cached implicit_domain if available, fallback to computation with root_expr
        let implicit_domain: Option<crate::implicit_domain::ImplicitDomain> =
            parent_ctx.implicit_domain().cloned().or_else(|| {
                parent_ctx
                    .root_expr()
                    .map(|root| crate::implicit_domain::infer_implicit_domain(ctx, root, vd))
            });

        let in_requires = implicit_domain.as_ref().is_some_and(|id| {
            let dc = crate::implicit_domain::DomainContext::new(
                id.conditions().iter().cloned().collect(),
            );
            let cond = crate::implicit_domain::ImplicitCondition::Positive(p_base);
            dc.is_condition_implied(ctx, &cond)
        });

        let can_chain = match dm {
            DomainMode::Strict | DomainMode::Generic => pos == Proof::Proven || in_requires,
            DomainMode::Assume => pos != Proof::Disproven, // In Assume: chain unless disproven
        };

        if can_chain {
            // Build the simplified version: even·ln(x) (without abs)
            let log_base = make_log(ctx, base, p_base);
            let final_expr = smart_mul(ctx, p_exp, log_base);

            // V2.14.20: Main rewrite produces mid_expr (with |x|)
            // ChainedRewrite then produces final_expr (without |x|)
            // This ensures engine creates two distinct steps
            let mut rw =
                crate::rule::Rewrite::new(mid_expr).desc("log(b, x^(even)) = even·log(b, |x|)");

            // Add chained step for |x| → x
            // V2.14.21: Use different descriptions:
            // - "for x > 0" when proven or when x > 0 is in requires
            // - "assuming x > 0" only in Assume mode when not proven and not in requires
            let chain_desc = if pos == Proof::Proven || in_requires {
                "|x| = x for x > 0"
            } else {
                "|x| = x (assuming x > 0)"
            };
            let mut chain = ChainedRewrite::new(final_expr)
                .desc(chain_desc)
                .local(abs_base, p_base);

            // Add assumption event only in Assume mode when not proven
            if pos != Proof::Proven && !in_requires && dm == DomainMode::Assume {
                chain = chain.assume(crate::assumptions::AssumptionEvent::positive_assumed(
                    ctx, p_base,
                ));
            }

            rw = rw.chain(chain);
            Some(rw)
        } else {
            // Cannot chain: would produce even·ln(|x|) introducing abs()
            //
            // V2.14.45 ANTI-WORSEN GUARD:
            // In Generic/Strict: BLOCK - introducing abs() without resolution worsens the expression
            // In Assume: allow since user has explicitly opted into assumptions
            match dm {
                DomainMode::Strict | DomainMode::Generic => {
                    // Don't worsen expression by introducing abs() we can't resolve
                    None
                }
                DomainMode::Assume => {
                    // In Assume mode: produce even·ln(|x|) with assumption event
                    let mut rw = crate::rule::Rewrite::new(mid_expr)
                        .desc("log(b, x^(even)) = even·log(b, |x|)");
                    // Add assumption that x > 0 or x < 0 (needed for abs to be meaningful)
                    rw = rw.assume(crate::assumptions::AssumptionEvent::positive_assumed(
                        ctx, p_base,
                    ));
                    Some(rw)
                }
            }
        }
    }

    fn target_types(&self) -> Option<Vec<&str>> {
        Some(vec!["Function"])
    }

    fn priority(&self) -> i32 {
        10 // Higher priority than EvaluateLogRule (default 0)
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::Medium
    }
}

/// LogAbsPowerRule: ln(|u|^n) → n·ln(|u|) for positive integer n
///
/// This rule handles the case where the argument of the log is already wrapped
/// in abs(), so we don't "introduce" a new abs - it's already there.
///
/// Priority: Very high (15) - must apply BEFORE:
/// - AbsSquareRule (|x|^2 → x^2) which would lose the abs
/// - LogEvenPowerWithChainedAbsRule which handles ln(x^n) without abs
///
/// Requires: u ≠ 0 (so ln(|u|) is defined)
pub struct LogAbsPowerRule;

impl crate::rule::Rule for LogAbsPowerRule {
    fn name(&self) -> &str {
        "Log Abs Power"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
        _parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<crate::rule::Rewrite> {
        let expr_data = ctx.get(expr).clone();
        let Expr::Function(name, args) = expr_data else {
            return None;
        };

        // Handle ln(x) or log(base, x)
        let name_str = ctx.sym_name(name);
        let (base, arg) = if name_str == "ln" && args.len() == 1 {
            let e = ctx.add(Expr::Constant(cas_ast::Constant::E));
            (e, args[0])
        } else if name_str == "log" && args.len() == 2 {
            (args[0], args[1])
        } else {
            return None;
        };

        // Match log(base, |u|^n)
        let arg_data = ctx.get(arg).clone();
        let Expr::Pow(p_base, p_exp) = arg_data else {
            return None;
        };

        // Check if base of power is abs(u)
        let Expr::Function(abs_name, abs_args) = ctx.get(p_base).clone() else {
            return None;
        };
        if ctx.sym_name(abs_name) != "abs" || abs_args.len() != 1 {
            return None;
        }
        let inner = abs_args[0]; // This is 'u' in |u|^n

        // Check if exponent is a positive integer
        let is_positive_integer = match ctx.get(p_exp) {
            Expr::Number(n) if n.is_integer() => {
                let int_val = n.to_integer();
                int_val > 0.into()
            }
            _ => false,
        };

        if !is_positive_integer {
            return None;
        }

        // Build n·ln(|u|)
        // Keep the abs - don't remove it
        let log_abs = make_log(ctx, base, p_base); // log(base, |u|)
        let result = smart_mul(ctx, p_exp, log_abs); // n · log(base, |u|)

        // Register hint that u ≠ 0 is required (for ln(|u|) to be defined)
        let key = crate::assumptions::AssumptionKey::nonzero_key(ctx, inner);
        let hint = crate::domain::BlockedHint {
            key,
            expr_id: inner,
            rule: "Log Abs Power".to_string(),
            suggestion: "requires u ≠ 0 for ln(|u|) to be defined",
        };
        crate::domain::register_blocked_hint(hint);

        Some(crate::rule::Rewrite::new(result).desc("ln(|u|^n) = n·ln(|u|)"))
    }

    fn target_types(&self) -> Option<Vec<&str>> {
        Some(vec!["Function"])
    }

    fn priority(&self) -> i32 {
        15 // Higher than LogEvenPowerWithChainedAbsRule (10) and AbsSquareRule
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::Medium
    }
}

pub struct LogAbsSimplifyRule;

impl crate::rule::Rule for LogAbsSimplifyRule {
    fn name(&self) -> &str {
        "Log Abs Simplify"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: cas_ast::ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<crate::rule::Rewrite> {
        use crate::domain::{DomainMode, Proof};
        use crate::helpers::prove_positive;
        use crate::semantics::ValueDomain;

        // Match ln(arg) or log(base, arg)
        let (base_opt, arg) = match ctx.get(expr).clone() {
            Expr::Function(fn_id, args) if ctx.sym_name(fn_id) == "ln" && args.len() == 1 => {
                (None, args[0])
            }
            Expr::Function(fn_id, args) if ctx.sym_name(fn_id) == "log" && args.len() == 2 => {
                (Some(args[0]), args[1])
            }
            _ => return None,
        };

        // Match abs(inner)
        let inner = match ctx.get(arg).clone() {
            Expr::Function(fn_id, args) if ctx.sym_name(fn_id) == "abs" && args.len() == 1 => {
                args[0]
            }
            _ => return None,
        };

        let vd = parent_ctx.value_domain();
        let dm = parent_ctx.domain_mode();
        let pos = prove_positive(ctx, inner, vd);

        // Helper to rebuild ln/log with inner (without abs)
        let mk_log = |ctx: &mut Context| -> ExprId {
            match base_opt {
                Some(base) => ctx.call("log", vec![base, inner]),
                None => ctx.call("ln", vec![inner]),
            }
        };

        // In ComplexEnabled: only allow if Proven (no assume - "positive" not well-defined for ℂ)
        if vd == ValueDomain::ComplexEnabled {
            if pos != Proof::Proven {
                return None;
            }
            return Some(Rewrite::new(mk_log(ctx)).desc("ln(|x|) = ln(x) for x > 0"));
        }

        // RealOnly: DomainMode policy
        //   - Strict: only if proven
        //   - Generic: only if proven (conservative - no silent assumptions)
        //   - Assume: allow with assumption warning
        match dm {
            DomainMode::Strict | DomainMode::Generic => {
                // Only simplify if proven positive (no silent assumptions)
                if pos != Proof::Proven {
                    return None;
                }
                Some(Rewrite::new(mk_log(ctx)).desc("ln(|x|) = ln(x) for x > 0"))
            }
            DomainMode::Assume => {
                // In Assume mode: simplify with warning (assumption traceability)
                Some(
                    Rewrite::new(mk_log(ctx))
                        .desc("ln(|x|) = ln(x) (assuming x > 0)")
                        .assume(crate::assumptions::AssumptionEvent::positive_assumed(
                            ctx, inner,
                        )),
                )
            }
        }
    }

    fn target_types(&self) -> Option<Vec<&str>> {
        Some(vec!["Function"])
    }

    // V2.14.20: Run in POST phase only so ln(|a|) created by LogPowerRule exists first
    fn allowed_phases(&self) -> crate::phase::PhaseMask {
        crate::phase::PhaseMask::POST
    }

    // Ensure step is visible - domain simplification is didactically important
    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::Low
    }
}

// =============================================================================
// LOG CHAIN PRODUCT RULE (LOG TELESCOPING)
// log(base, a) * log(a, c) → log(base, c)
// =============================================================================
// This implements the "change of base" telescoping identity:
//   log_b(a) * log_a(c) = log_b(c)
//
// Using the definition log_b(x) = ln(x)/ln(b):
//   (ln(a)/ln(b)) * (ln(c)/ln(a)) = ln(c)/ln(b) = log_b(c)
//
// The rule scans Mul chains for pairs of logs where:
// - Value of log_i == Base of log_j (or vice versa, since Mul is commutative)
//
// REDUCES log count: 2 logs → 1 log (naturally terminante)
//
// Soundness: EquivalenceUnderIntroducedRequires
// - Requires: both log arguments > 0, bases ≠ 1
// - These are already implied by the logs being defined
pub struct LogChainProductRule;

impl crate::rule::Rule for LogChainProductRule {
    fn name(&self) -> &str {
        "Log Chain (Telescoping)"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: cas_ast::ExprId,
        _parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<crate::rule::Rewrite> {
        use crate::helpers::flatten_mul;

        // Only match Mul nodes
        if !matches!(ctx.get(expr), Expr::Mul(_, _)) {
            return None;
        }

        // Flatten the multiplication chain
        let mut factors: Vec<ExprId> = Vec::new();
        flatten_mul(ctx, expr, &mut factors);

        // Extract log parts from all factors
        // log_parts[i] = Some((base, arg)) if factor[i] is log(base, arg)
        let log_parts: Vec<Option<(ExprId, ExprId)>> =
            factors.iter().map(|&f| extract_log_parts(ctx, f)).collect();

        // Find a pair (i, j) where log_i.arg == log_j.base
        // i.e., log(b1, a) * log(a, c) → log(b1, c)
        for (i, log_i_opt) in log_parts.iter().enumerate() {
            let Some((base_i, arg_i)) = log_i_opt else {
                continue;
            };

            for (j, log_j_opt) in log_parts.iter().enumerate() {
                if i == j {
                    continue;
                }
                let Some((base_j, arg_j)) = log_j_opt else {
                    continue;
                };

                // Check if arg_i == base_j (telescoping condition)
                // arg_i is the "middle" value that cancels
                if !exprs_match(ctx, *arg_i, *base_j) {
                    continue;
                }

                // Found a match! log(base_i, arg_i) * log(arg_i, arg_j) → log(base_i, arg_j)
                // Build the new log
                let new_log = make_log(ctx, *base_i, *arg_j);

                // Build the remaining product (all factors except i and j)
                let remaining: Vec<ExprId> = factors
                    .iter()
                    .enumerate()
                    .filter(|&(idx, _)| idx != i && idx != j)
                    .map(|(_, &f)| f)
                    .collect();

                let result = if remaining.is_empty() {
                    new_log
                } else {
                    // Multiply new_log with remaining factors
                    let mut product = new_log;
                    for r in remaining {
                        product = smart_mul(ctx, product, r);
                    }
                    product
                };

                // Build description showing what was telescoped
                let desc = "log(b, a) * log(a, c) = log(b, c)";

                return Some(crate::rule::Rewrite::new(result).desc(desc));
            }
        }

        None
    }

    fn target_types(&self) -> Option<Vec<&str>> {
        Some(vec!["Mul"])
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::High
    }
}

/// Check if two expressions match (for telescoping condition)
/// Handles ln sentinel and structural equality
fn exprs_match(ctx: &Context, e1: ExprId, e2: ExprId) -> bool {
    let sentinel = ExprId::from_raw(u32::MAX);

    // ln sentinel matches Constant::E
    if e1 == sentinel {
        return matches!(ctx.get(e2), Expr::Constant(cas_ast::Constant::E));
    }
    if e2 == sentinel {
        return matches!(ctx.get(e1), Expr::Constant(cas_ast::Constant::E));
    }

    // Direct ID match
    if e1 == e2 {
        return true;
    }

    // Structural comparison for same expression
    compare_expr(ctx, e1, e2) == Ordering::Equal
}

/// LogContractionRule: Contracts sums/differences of logs into single logs.
/// - ln(a) + ln(b) → ln(a*b)
/// - ln(a) - ln(b) → ln(a/b)
/// - log(b, x) + log(b, y) → log(b, x*y)  (same base required)
/// - log(b, x) - log(b, y) → log(b, x/y)
///
/// This rule REDUCES node count and is a valid simplification.
/// Unlike LogExpansionRule, this is registered by default.
pub struct LogContractionRule;

impl crate::rule::Rule for LogContractionRule {
    fn name(&self) -> &str {
        "Log Contraction"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: cas_ast::ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<crate::rule::Rewrite> {
        use crate::semantics::NormalFormGoal;

        // GATE: Don't contract logs when goal is ExpandedLog
        // This prevents undoing the effect of expand_log command
        if parent_ctx.goal() == NormalFormGoal::ExpandedLog {
            return None;
        }

        // GATE: Don't contract logs when in auto-expand mode
        // This prevents cycle with AutoExpandLogRule (expand→contract→expand→...)
        if parent_ctx.is_auto_expand() || parent_ctx.in_auto_expand_context() {
            return None;
        }

        let expr_data = ctx.get(expr).clone();

        // Case 1: ln(a) + ln(b) → ln(a*b) or log(b,x) + log(b,y) → log(b, x*y)
        if let Expr::Add(lhs, rhs) = expr_data {
            if let (Some((base_l, arg_l)), Some((base_r, arg_r))) =
                (extract_log_parts(ctx, lhs), extract_log_parts(ctx, rhs))
            {
                // Check bases are equal
                if bases_equal(ctx, base_l, base_r) {
                    let product = ctx.add(Expr::Mul(arg_l, arg_r));
                    // If base is sentinel (ln case), create ln(), otherwise log()
                    let sentinel = cas_ast::ExprId::from_raw(u32::MAX);
                    let new_expr = if base_l == sentinel {
                        ctx.call("ln", vec![product])
                    } else {
                        make_log(ctx, base_l, product)
                    };
                    return Some(
                        crate::rule::Rewrite::new(new_expr).desc("ln(a) + ln(b) = ln(a*b)"),
                    );
                }
            }
        }

        // Case 2: ln(a) - ln(b) → ln(a/b) or log(b,x) - log(b,y) → log(b, x/y)
        if let Expr::Sub(lhs, rhs) = expr_data {
            if let (Some((base_l, arg_l)), Some((base_r, arg_r))) =
                (extract_log_parts(ctx, lhs), extract_log_parts(ctx, rhs))
            {
                // Check bases are equal
                if bases_equal(ctx, base_l, base_r) {
                    let quotient = ctx.add(Expr::Div(arg_l, arg_r));
                    // If base is sentinel (ln case), create ln(), otherwise log()
                    let sentinel = cas_ast::ExprId::from_raw(u32::MAX);
                    let new_expr = if base_l == sentinel {
                        ctx.call("ln", vec![quotient])
                    } else {
                        make_log(ctx, base_l, quotient)
                    };
                    return Some(
                        crate::rule::Rewrite::new(new_expr).desc("ln(a) - ln(b) = ln(a/b)"),
                    );
                }
            }
        }

        None
    }

    fn target_types(&self) -> Option<Vec<&str>> {
        Some(vec!["Add", "Sub"])
    }
}

/// Extract (base, argument) from a log expression.
/// Returns (E, arg) for ln(arg), (base, arg) for log(base, arg), None otherwise.
fn extract_log_parts(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId)> {
    if let Expr::Function(fn_id, args) = ctx.get(expr) {
        let name = ctx.sym_name(*fn_id);
        if name == "ln" && args.len() == 1 {
            // For ln(x), base is implicitly e - we need to create it
            // But we can't mutate ctx here. Instead we'll use a sentinel value.
            // Actually, let's handle ln specially in bases_equal
            return Some((cas_ast::ExprId::from_raw(u32::MAX), args[0])); // Sentinel for "e"
        } else if name == "log" && args.len() == 2 {
            return Some((args[0], args[1]));
        }
    }
    None
}

/// Check if two log bases are equal.
/// Handles ln (sentinel u32::MAX) specially.
fn bases_equal(ctx: &cas_ast::Context, base_l: cas_ast::ExprId, base_r: cas_ast::ExprId) -> bool {
    let sentinel = cas_ast::ExprId::from_raw(u32::MAX);

    // Both are ln (sentinel)
    if base_l == sentinel && base_r == sentinel {
        return true;
    }

    // One is ln, other is explicit log(e, ...)
    if base_l == sentinel {
        if let Expr::Constant(cas_ast::Constant::E) = ctx.get(base_r) {
            return true;
        }
        return false;
    }
    if base_r == sentinel {
        if let Expr::Constant(cas_ast::Constant::E) = ctx.get(base_l) {
            return true;
        }
        return false;
    }

    // Both are explicit bases - check if equal by expression equality
    // For simplicity, only match if they're the same ExprId or both are Constant::E
    if base_l == base_r {
        return true;
    }

    // Check if both are e constant
    if let (Expr::Constant(cas_ast::Constant::E), Expr::Constant(cas_ast::Constant::E)) =
        (ctx.get(base_l), ctx.get(base_r))
    {
        return true;
    }

    false
}

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
        let expr_data = ctx.get(expr).clone();
        if let Expr::Pow(base, exp) = expr_data {
            let exp_data = ctx.get(exp).clone();

            // Helper to get log base and arg
            let get_log_parts = |ctx: &mut cas_ast::Context,
                                 e_id: cas_ast::ExprId|
             -> Option<(cas_ast::ExprId, cas_ast::ExprId)> {
                let e_data = ctx.get(e_id).clone();
                if let Expr::Function(name, args) = e_data {
                    let name_str = ctx.sym_name(name);
                    if name_str == "log" && args.len() == 2 {
                        return Some((args[0], args[1]));
                    } else if name_str == "ln" && args.len() == 1 {
                        let e = ctx.add(Expr::Constant(cas_ast::Constant::E));
                        return Some((e, args[0]));
                    }
                }
                None
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

            // Case 2: b^(c * log(b, x)) → x^c
            // Same logic as Case 1: x > 0 is IMPLICIT from log(b, x) existing.
            if let Expr::Mul(lhs, rhs) = &exp_data {
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

                if let Some(rw) = check_log(*lhs, *rhs) {
                    return Some(rw);
                }
                if let Some(rw) = check_log(*rhs, *lhs) {
                    return Some(rw);
                }
            }
        }
        None
    }

    fn target_types(&self) -> Option<Vec<&str>> {
        Some(vec!["Pow"])
    }

    fn solve_safety(&self) -> crate::solve_safety::SolveSafety {
        crate::solve_safety::SolveSafety::NeedsCondition(
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
    let expr_data = ctx.get(expr).clone();
    if let Expr::Pow(base, exp) = expr_data {
        let base_is_e = matches!(ctx.get(base), Expr::Constant(cas_ast::Constant::E));
        if base_is_e {
            let exp_data = ctx.get(exp).clone();
            if let Expr::Add(lhs, rhs) = exp_data {
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

fn simplify_exp_log(context: &mut Context, base: ExprId, exp: ExprId) -> ExprId {
    // Check if exp is log(base, x)
    if let Expr::Function(name, args) = context.get(exp).clone() {
        let name_str = context.sym_name(name);
        if name_str == "log" && args.len() == 2 {
            let log_base = args[0];
            let log_arg = args[1];
            if log_base == base {
                return log_arg;
            }
        }
    }
    // Also check n*log(base, x) -> x^n?
    // Maybe later. For now just direct cancellation.
    context.add(Expr::Pow(base, exp))
}

fn is_log(context: &Context, expr: ExprId) -> bool {
    if let Expr::Function(name, _) = context.get(expr) {
        let name_str = context.sym_name(*name);
        return name_str == "log" || name_str == "ln";
    }
    // Also check for n*log(x)
    if let Expr::Mul(l, r) = context.get(expr) {
        return is_log(context, *l) || is_log(context, *r);
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rule::Rule;
    use cas_ast::{Context, DisplayExpr};
    use cas_parser::parse;

    #[test]
    fn test_log_one() {
        let mut ctx = Context::new();
        let rule = EvaluateLogRule;
        // log(x, 1) -> 0
        let expr = parse("log(x, 1)", &mut ctx).unwrap();
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "0"
        );
    }

    #[test]
    fn test_log_base_base() {
        let mut ctx = Context::new();
        let rule = EvaluateLogRule;
        // log(x, x) -> 1
        let expr = parse("log(x, x)", &mut ctx).unwrap();
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "1"
        );
    }

    #[test]
    fn test_log_inverse() {
        let mut ctx = Context::new();
        let rule = EvaluateLogRule;
        // log(x, x^2) -> 2
        let expr = parse("log(x, x^2)", &mut ctx).unwrap();
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "2"
        );
    }

    #[test]
    fn test_log_expansion() {
        let mut ctx = Context::new();
        let rule = EvaluateLogRule;
        // log(b, x^y) -> y * log(b, x)
        let expr = parse("log(2, x^3)", &mut ctx).unwrap();
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "3 * log(2, x)"
        );
    }

    #[test]
    fn test_log_product() {
        let mut ctx = Context::new();
        let rule = LogExpansionRule;
        // log(b, x*y) -> log(b, x) + log(b, y) (requires Assume mode for variables)
        let expr = parse("log(2, x * y)", &mut ctx).unwrap();
        // Create parent context with Assume mode (allows expansion with warning)
        let parent_ctx = crate::parent_context::ParentContext::root()
            .with_domain_mode(crate::domain::DomainMode::Assume);
        let rewrite = rule.apply(&mut ctx, expr, &parent_ctx).unwrap();
        let res = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        );
        assert!(res.contains("log(2, x)"));
        assert!(res.contains("log(2, y)"));
        assert!(res.contains("+"));
    }

    #[test]
    fn test_log_quotient() {
        let mut ctx = Context::new();
        let rule = LogExpansionRule;
        // log(b, x/y) -> log(b, x) - log(b, y) (requires Assume mode for variables)
        let expr = parse("log(2, x / y)", &mut ctx).unwrap();
        // Create parent context with Assume mode (allows expansion with warning)
        let parent_ctx = crate::parent_context::ParentContext::root()
            .with_domain_mode(crate::domain::DomainMode::Assume);
        let rewrite = rule.apply(&mut ctx, expr, &parent_ctx).unwrap();
        let res = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        );
        assert!(res.contains("log(2, x)"));
        assert!(res.contains("log(2, y)"));
        assert!(res.contains("-"));
    }

    #[test]
    fn test_ln_e() {
        let mut ctx = Context::new();
        let rule = EvaluateLogRule;
        // ln(e) -> 1
        // Note: ln(e) parses to log(e, e)
        let expr = parse("ln(e)", &mut ctx).unwrap();
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "1"
        );
    }
}

define_rule!(
    LogInversePowerRule,
    "Log Inverse Power",
    solve_safety: crate::solve_safety::SolveSafety::NeedsCondition(
        crate::assumptions::ConditionClass::Analytic
    ),
    |ctx, expr, _parent_ctx| {
    // println!("LogInversePowerRule checking {:?}", expr);
    let expr_data = ctx.get(expr).clone();
    if let Expr::Pow(base, exp) = expr_data {
        // Check for x^(c / log(b, x))
        // exp could be Div(c, log(b, x)) or Mul(c, Pow(log(b, x), -1))

        // Returns Some(Some(base)) for log(b, x), Some(None) for ln(x) -> base e
        let check_log_denom =
            |ctx: &Context, denom: cas_ast::ExprId| -> Option<Option<cas_ast::ExprId>> {
                // println!("check_log_denom checking {:?}", denom);
                if let Expr::Function(fn_id, args) = ctx.get(denom) { let name = ctx.sym_name(*fn_id);
                    // Debug: checking log denominator
                    if name == "log" && args.len() == 2 {
                        let log_base = args[0];
                        let log_arg = args[1];
                        // Check if log_arg == base
                        if compare_expr(ctx, log_arg, base) == Ordering::Equal {
                            // Debug: found matching log
                            return Some(Some(log_base));
                        }
                    } else if name == "ln" && args.len() == 1 {
                        let log_arg = args[0];
                        if compare_expr(ctx, log_arg, base) == Ordering::Equal {
                            // Debug: found matching ln
                            return Some(None); // Base e
                        }
                    }
                }
                None
            };

        let mut target_b_opt: Option<Option<cas_ast::ExprId>> = None;
        let mut coeff: Option<cas_ast::ExprId> = None;

        let exp_data = ctx.get(exp).clone();
        match exp_data {
            Expr::Div(num, den) => {
                if let Some(b_opt) = check_log_denom(ctx, den) {
                    target_b_opt = Some(b_opt);
                    coeff = Some(num);
                }
            }
            Expr::Mul(l, r) => {
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
            }
            Expr::Pow(b, e) => {
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
            _ => {}
        }

        if let (Some(b_opt), Some(c)) = (target_b_opt, coeff) {
            // Result is b^c
            let b = b_opt.unwrap_or_else(|| ctx.add(Expr::Constant(cas_ast::Constant::E)));
            // Debug: applying log inverse power rule
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
        let expr_data = ctx.get(expr).clone();
        if let Expr::Function(fn_id, args) = expr_data {
            let name = ctx.sym_name(fn_id);
            // Handle ln(x) as log(e, x), or log(b, x)
            let (base, arg) = if name == "ln" && args.len() == 1 {
                let e = ctx.add(Expr::Constant(cas_ast::Constant::E));
                (e, args[0])
            } else if name == "log" && args.len() == 2 {
                (args[0], args[1])
            } else {
                return None;
            };

            let arg_data = ctx.get(arg).clone();

            // log(b, b^x) → x (when b matches)
            if let Expr::Pow(p_base, p_exp) = arg_data {
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
                        let is_e_base =
                            matches!(ctx.get(base), Expr::Constant(cas_ast::Constant::E));

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

    fn target_types(&self) -> Option<Vec<&str>> {
        Some(vec!["Function"])
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
        let expr_data = ctx.get(expr).clone();
        if let Expr::Function(fn_id, args) = expr_data {
            let name = ctx.sym_name(fn_id);
            // Match log(base, arg) - not ln (which has implicit base e)
            if name != "log" || args.len() != 2 {
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

    fn target_types(&self) -> Option<Vec<&str>> {
        Some(vec!["Function"])
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

/// Normalize an expression to (core, exponent) form:
/// - a → (a, 1)
/// - a^m → (a, m)
/// - 1/a → (a, -1)
/// - a^m/b → not handled, returns original
fn normalize_to_power(ctx: &mut cas_ast::Context, expr: ExprId) -> (ExprId, ExprId) {
    match ctx.get(expr).clone() {
        Expr::Pow(base, exp) => (base, exp),
        Expr::Div(num, den) => {
            // Check if num is 1 (literal 1)
            if matches!(ctx.get(num), Expr::Number(n) if n.is_one()) {
                // 1/a → (a, -1)
                let neg_one = ctx.num(-1);
                (den, neg_one)
            } else {
                // Not a simple reciprocal, return as is
                let one = ctx.num(1);
                (expr, one)
            }
        }
        _ => {
            // Just a → (a, 1)
            let one = ctx.num(1);
            (expr, one)
        }
    }
}

pub fn register(simplifier: &mut crate::Simplifier) {
    // V2.14.45: LogPowerBaseRule MUST come BEFORE LogEvenPowerRule
    // Otherwise log(x^2, x^6) gets expanded to 6·log(x², |x|) before simplifying to 3
    simplifier.add_rule(Box::new(LogPowerBaseRule)); // log(a^m, a^n) → n/m

    // LogAbsPowerRule: ln(|u|^n) → n·ln(|u|) - highest priority (15)
    // MUST come BEFORE AbsSquareRule which would turn |u|^2 → u^2 and lose the abs
    simplifier.add_rule(Box::new(LogAbsPowerRule));

    // V2.14.20: LogEvenPowerWithChainedAbsRule handles ln(x^even) with ChainedRewrite
    // Has higher priority (10) than EvaluateLogRule (0) so matches first
    simplifier.add_rule(Box::new(LogEvenPowerWithChainedAbsRule));
    simplifier.add_rule(Box::new(EvaluateLogRule));

    // LnEProductRule: ln(e*x) → 1 + ln(x) - safe targeted expansion
    // This enables residuals like `2*ln(e*u) - 2 - 2*ln(u)` to simplify to 0
    simplifier.add_rule(Box::new(LnEProductRule));
    // LnEDivRule: ln(x/e) → ln(x) - 1, ln(e/x) → 1 - ln(x)
    // Companion to LnEProductRule for quotient cases
    simplifier.add_rule(Box::new(LnEDivRule));

    // NOTE: LogExpansionRule removed from auto-registration.
    // Log expansion increases node count (ln(xy) → ln(x) + ln(y)) and is not always desirable.
    // Use the `expand_log` command for explicit expansion.
    // simplifier.add_rule(Box::new(LogExpansionRule));

    // LogAbsSimplifyRule: ln(|x|) → ln(x) when x > 0
    // Must be BEFORE LogContractionRule to catch `ln(|x|) - ln(x)` before it becomes `ln(|x|/x)`
    simplifier.add_rule(Box::new(LogAbsSimplifyRule));

    // LogChainProductRule: log(b,a)*log(a,c) → log(b,c) (telescoping)
    // Must be BEFORE LogContractionRule
    simplifier.add_rule(Box::new(LogChainProductRule));

    // LogContractionRule DOES reduce node count (ln(a)+ln(b) → ln(ab)) - valid simplification
    simplifier.add_rule(Box::new(LogContractionRule));

    simplifier.add_rule(Box::new(ExponentialLogRule));
    simplifier.add_rule(Box::new(SplitLogExponentsRule));
    simplifier.add_rule(Box::new(LogInversePowerRule));
    simplifier.add_rule(Box::new(LogExpInverseRule));

    // AutoExpandLogRule: auto-expand log products/quotients when log_expand_policy=Auto
    simplifier.add_rule(Box::new(AutoExpandLogRule));
}

// ============================================================================
// Auto Expand Log Rule with ExpandBudget Integration
// ============================================================================

/// Estimates the number of terms that would result from expanding a log expression.
/// Returns `(base_terms, gen_terms, pow_exp)`:
/// - base_terms: number of factors in the log argument
/// - gen_terms: number of log terms that would be generated
/// - pow_exp: if the argument is u^n, returns Some(n) for integer n
///
/// Returns None if the expression is not expandable (not Mul/Div/Pow).
pub fn estimate_log_terms(ctx: &Context, arg: ExprId) -> Option<(u32, u32, Option<u32>)> {
    match ctx.get(arg) {
        // Mul(a, b) - could be nested, so we flatten
        Expr::Mul(_, _) => {
            let factors = count_mul_factors(ctx, arg);
            if factors <= 1 {
                return None; // No benefit from expanding
            }
            Some((factors, factors, None))
        }
        // Div(num, den) - expands to log(num) - log(den)
        Expr::Div(num, den) => {
            let num_factors = count_mul_factors(ctx, *num);
            let den_factors = count_mul_factors(ctx, *den);
            let total = num_factors + den_factors;
            if total <= 1 {
                return None;
            }
            Some((total, total, None))
        }
        // Pow(base, exp) - expands to exp * log(base) if exp is integer
        Expr::Pow(_, exp) => {
            // Only expand if exponent is a positive integer
            if let Expr::Number(n) = ctx.get(*exp) {
                if n.is_integer() {
                    let exp_i64: i64 = n.to_integer().try_into().ok()?;
                    if exp_i64 > 0 {
                        let exp_u32 = exp_i64 as u32;
                        // log(u^n) -> n*log(u): base_terms=1, gen_terms=1
                        return Some((1, 1, Some(exp_u32)));
                    }
                }
            }
            None
        }
        _ => None,
    }
}

/// Count the number of multiplicative factors in a flattened Mul expression.
fn count_mul_factors(ctx: &Context, expr: ExprId) -> u32 {
    match ctx.get(expr) {
        Expr::Mul(a, b) => count_mul_factors(ctx, *a) + count_mul_factors(ctx, *b),
        _ => 1,
    }
}

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

        // Match log(arg) or ln(arg)
        let arg = match ctx.get(expr) {
            Expr::Function(fn_id, args)
                if (ctx.sym_name(*fn_id) == "log" || ctx.sym_name(*fn_id) == "ln")
                    && args.len() == 1 =>
            {
                args[0]
            }
            Expr::Function(fn_id, args) if ctx.sym_name(*fn_id) == "log" && args.len() == 2 => {
                args[1] // log(base, arg)
            }
            _ => return None,
        };

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

    fn target_types(&self) -> Option<Vec<&str>> {
        Some(vec!["Function"])
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

/// Collect all multiplicative factors from a Mul expression (flattened).
fn collect_mul_factors(ctx: &Context, expr: ExprId) -> Vec<ExprId> {
    match ctx.get(expr) {
        Expr::Mul(a, b) => {
            let mut factors = collect_mul_factors(ctx, *a);
            factors.extend(collect_mul_factors(ctx, *b));
            factors
        }
        _ => vec![expr],
    }
}

/// Perform the log expansion for AutoExpandLogRule.
fn expand_log_for_rule(
    ctx: &mut Context,
    _original: ExprId,
    arg: ExprId,
    events: &[crate::assumptions::AssumptionEvent],
) -> Option<Rewrite> {
    // Get base (ln = natural log, log with 1 arg = base 10)
    let base = match ctx.get(_original).clone() {
        Expr::Function(name, _) if ctx.sym_name(name) == "ln" => {
            ctx.add(Expr::Constant(cas_ast::Constant::E))
        }
        Expr::Function(fn_id, args) if ctx.sym_name(fn_id) == "log" && args.len() == 2 => args[0],
        Expr::Function(_, _) => {
            // log with 1 arg = base 10, use sentinel
            ExprId::from_raw(u32::MAX - 1)
        }
        _ => return None,
    };

    match ctx.get(arg).clone() {
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
