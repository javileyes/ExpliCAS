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
        return ctx.add(Expr::Function("log".to_string(), vec![arg]));
    }
    if let Expr::Constant(cas_ast::Constant::E) = ctx.get(base) {
        ctx.add(Expr::Function("ln".to_string(), vec![arg]))
    } else {
        ctx.add(Expr::Function("log".to_string(), vec![base, arg]))
    }
}

define_rule!(EvaluateLogRule, "Evaluate Logarithms", |ctx, expr| {
    let expr_data = ctx.get(expr).clone();
    if let Expr::Function(name, args) = expr_data {
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
                // Simple check for integer powers for now
                if b.is_integer() && n.is_integer() {
                    let b_int = b.to_integer();
                    let n_int = n.to_integer();
                    if b_int > num_bigint::BigInt::from(1) {
                        let mut temp = b_int.clone();
                        let mut power = 1;
                        while temp < n_int {
                            temp *= &b_int;
                            power += 1;
                        }
                        if temp == n_int {
                            let new_expr = ctx.num(power);
                            return Some(
                                Rewrite::new(new_expr)
                                    .desc(format!("log({}, {}) = {}", b, n, power)),
                            );
                        }
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
                    // Even exponent: ln(x^(2k)) = 2k·ln(|x|) - no assumption needed
                    let abs_base = ctx.add(Expr::Function("abs".to_string(), vec![p_base]));
                    let log_abs = make_log(ctx, base, abs_base);
                    let new_expr = smart_mul(ctx, p_exp, log_abs);
                    return Some(
                        Rewrite::new(new_expr).desc("log(b, x^(even)) = even·log(b, |x|)"),
                    );
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
        if let Expr::Function(name, args) = expr_data {
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
        Expr::Function(ref name, ref args) if (name == "ln" || name == "log") => {
            // Try to expand this log
            // Sentinel: base-10 log uses ExprId::from_raw(u32::MAX - 1) as base indicator
            let sentinel_log10 = cas_ast::ExprId::from_raw(u32::MAX - 1);
            let (base, arg) = if name == "ln" && args.len() == 1 {
                let e = ctx.add(Expr::Constant(cas_ast::Constant::E));
                (e, args[0])
            } else if name == "log" && args.len() == 2 {
                (args[0], args[1])
            } else if name == "log" && args.len() == 1 {
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
                return (ctx.add(Expr::Function(name.clone(), new_args)), events);
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
            Expr::Function(name, args) if name == "ln" && args.len() == 1 => (None, args[0]),
            Expr::Function(name, args) if name == "log" && args.len() == 2 => {
                (Some(args[0]), args[1])
            }
            _ => return None,
        };

        // Match abs(inner)
        let inner = match ctx.get(arg).clone() {
            Expr::Function(name, args) if name == "abs" && args.len() == 1 => args[0],
            _ => return None,
        };

        let vd = parent_ctx.value_domain();
        let dm = parent_ctx.domain_mode();
        let pos = prove_positive(ctx, inner, vd);

        // Helper to rebuild ln/log with inner (without abs)
        let mk_log = |ctx: &mut Context| -> ExprId {
            match base_opt {
                Some(base) => ctx.add(Expr::Function("log".to_string(), vec![base, inner])),
                None => ctx.add(Expr::Function("ln".to_string(), vec![inner])),
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
                        ctx.add(Expr::Function("ln".to_string(), vec![product]))
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
                        ctx.add(Expr::Function("ln".to_string(), vec![quotient]))
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
    if let Expr::Function(name, args) = ctx.get(expr) {
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
                    if name == "log" && args.len() == 2 {
                        return Some((args[0], args[1]));
                    } else if name == "ln" && args.len() == 1 {
                        let e = ctx.add(Expr::Constant(cas_ast::Constant::E));
                        return Some((e, args[0]));
                    }
                }
                None
            };

            // Case 1: b^log(b, x) → x
            // Requires Positive(x) - Analytic class (Generic blocks, only Assume allows)
            if let Some((log_base, log_arg)) = get_log_parts(ctx, exp) {
                if compare_expr(ctx, log_base, base) == Ordering::Equal {
                    let mode = parent_ctx.domain_mode();
                    let vd = parent_ctx.value_domain();

                    // Use prove_positive with ValueDomain
                    let arg_positive = crate::helpers::prove_positive(ctx, log_arg, vd);

                    // Use Analytic gate with hint: Generic blocks with pedagogical message
                    let key = crate::assumptions::AssumptionKey::positive_key(ctx, log_arg);
                    let decision = crate::domain::can_apply_analytic_with_hint(
                        mode,
                        arg_positive,
                        key,
                        log_arg, // expr_id for pretty-printing
                        "Exponential-Log Inverse",
                    );

                    if !decision.allow {
                        return None; // Strict/Generic: block if not proven
                    }

                    // Build assumption events for unproven
                    let assumption_events: smallvec::SmallVec<
                        [crate::assumptions::AssumptionEvent; 1],
                    > = if decision.assumption.is_some() {
                        smallvec::smallvec![crate::assumptions::AssumptionEvent::positive_assumed(
                            ctx, log_arg
                        )]
                    } else {
                        smallvec::SmallVec::new()
                    };

                    return Some(
                        crate::rule::Rewrite::new(log_arg)
                            .desc("b^log(b, x) = x")
                            .assume_all(assumption_events),
                    );
                }
            }

            // Case 2: b^(c * log(b, x)) → x^c
            // Requires Positive(x) - Analytic class (Generic blocks, only Assume allows)
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

                            // Use Analytic gate: Generic blocks, only Assume allows unproven
                            let decision = crate::domain::can_apply_analytic(mode, arg_positive);

                            if !decision.allow {
                                return None; // Strict/Generic: block if not proven
                            }

                            let new_expr = ctx.add(Expr::Pow(log_arg, coeff));

                            // Build assumption events for unproven
                            let assumption_events: smallvec::SmallVec<
                                [crate::assumptions::AssumptionEvent; 1],
                            > = if decision.assumption.is_some() {
                                smallvec::smallvec![
                                    crate::assumptions::AssumptionEvent::positive_assumed(
                                        ctx, log_arg
                                    )
                                ]
                            } else {
                                smallvec::SmallVec::new()
                            };

                            return Some(
                                crate::rule::Rewrite::new(new_expr)
                                    .desc("b^(c*log(b, x)) = x^c")
                                    .assume_all(assumption_events),
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
    if let Expr::Function(name, args) = context.get(exp) {
        if name == "log" && args.len() == 2 {
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
        return name == "log" || name == "ln";
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
                if let Expr::Function(name, args) = ctx.get(denom) {
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
        if let Expr::Function(name, args) = expr_data {
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
                                        return Some(
                                            crate::rule::Rewrite::new(p_exp)
                                                .desc("log(b, b^x) → x")
                                                .assume(
                                                    crate::assumptions::AssumptionEvent::positive_assumed(
                                                        ctx, base,
                                                    ),
                                                ),
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
                        return Some(crate::rule::Rewrite::new(p_exp).desc("log(b, b^x) → x"));
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

pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(EvaluateLogRule));
    // NOTE: LogExpansionRule removed from auto-registration.
    // Log expansion increases node count (ln(xy) → ln(x) + ln(y)) and is not always desirable.
    // Use the `expand_log` command for explicit expansion.
    // simplifier.add_rule(Box::new(LogExpansionRule));

    // LogAbsSimplifyRule: ln(|x|) → ln(x) when x > 0
    // Must be BEFORE LogContractionRule to catch `ln(|x|) - ln(x)` before it becomes `ln(|x|/x)`
    simplifier.add_rule(Box::new(LogAbsSimplifyRule));

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

/// Check if an expression is provably positive (for Generic mode proof).
/// Returns true for positive literals and known-positive expressions.
fn is_provably_positive(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Number(n) => *n > num_rational::BigRational::zero(),
        Expr::Constant(cas_ast::Constant::E) => true,
        Expr::Constant(cas_ast::Constant::Pi) => true,
        Expr::Function(name, args) if name == "exp" && args.len() == 1 => true, // e^x > 0 always
        Expr::Function(name, args) if name == "abs" && args.len() == 1 => {
            // abs(x) >= 0, but only > 0 if x != 0; we'll be conservative
            false
        }
        Expr::Pow(base, exp) => {
            // x^2 > 0 if x != 0 (we'll be conservative here)
            if let Expr::Number(n) = ctx.get(*exp) {
                if n.is_integer() {
                    let exp_int = n.to_integer();
                    if &exp_int % 2 == num_bigint::BigInt::from(0) {
                        // Even power: need base != 0 to be positive
                        return false; // Conservative
                    }
                }
            }
            is_provably_positive(ctx, *base)
        }
        _ => false,
    }
}

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
            Expr::Function(name, args) if (name == "log" || name == "ln") && args.len() == 1 => {
                args[0]
            }
            Expr::Function(name, args) if name == "log" && args.len() == 2 => {
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
                let all_positive = factors.iter().all(|&f| is_provably_positive(ctx, f));
                if !all_positive {
                    return None; // Block silently in Strict
                }
                // Expand without assumption events (proven)
                expand_log_for_rule(ctx, expr, arg, &[])
            }
            crate::domain::DomainMode::Generic => {
                // In Generic, block if not proven, but register hint
                let factors = collect_mul_factors(ctx, arg);
                let all_positive = factors.iter().all(|&f| is_provably_positive(ctx, f));
                if !all_positive {
                    // Register blocked hint for user feedback
                    if let Some(&factor) = factors.iter().find(|&&f| !is_provably_positive(ctx, f))
                    {
                        let hint = crate::domain::BlockedHint {
                            key: crate::assumptions::AssumptionKey::Positive {
                                expr_fingerprint: crate::assumptions::expr_fingerprint(ctx, factor),
                            },
                            expr_id: factor,
                            rule: "AutoExpandLogRule",
                            suggestion:
                                "Use 'semantics set domain assume' to enable log expansion.",
                        };
                        crate::domain::register_blocked_hint(hint);
                    }
                    return None;
                }
                // All factors proven positive, expand without events
                expand_log_for_rule(ctx, expr, arg, &[])
            }
            crate::domain::DomainMode::Assume => {
                // In Assume mode, expand and emit HeuristicAssumption events
                let factors = collect_mul_factors(ctx, arg);
                let mut events = Vec::new();
                for &factor in &factors {
                    if !is_provably_positive(ctx, factor) {
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
    let base = match ctx.get(_original) {
        Expr::Function(name, _) if name == "ln" => ctx.add(Expr::Constant(cas_ast::Constant::E)),
        Expr::Function(name, args) if name == "log" && args.len() == 2 => args[0],
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
