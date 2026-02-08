use crate::rule::Rewrite;
use crate::rules::algebra::helpers::smart_mul;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};

// Re-use helpers from parent module
use super::{extract_log_parts, make_log};

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
            // Handle ln(x) as log(e, x), or log(b, x)
            let (base, arg) = match ctx.builtin_of(fn_id) {
                Some(BuiltinFn::Ln) if args.len() == 1 => {
                    let e = ctx.add(Expr::Constant(cas_ast::Constant::E));
                    (e, args[0])
                }
                Some(BuiltinFn::Log) if args.len() == 2 => (args[0], args[1]),
                _ => return None,
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
            if matches!(
                ctx.builtin_of(name),
                Some(BuiltinFn::Ln) | Some(BuiltinFn::Log)
            ) =>
        {
            // Try to expand this log
            // Sentinel: base-10 log uses ExprId::from_raw(u32::MAX - 1) as base indicator
            let sentinel_log10 = cas_ast::ExprId::from_raw(u32::MAX - 1);
            let builtin = ctx.builtin_of(name);
            let (base, arg) = if builtin == Some(BuiltinFn::Ln) && args.len() == 1 {
                let e = ctx.add(Expr::Constant(cas_ast::Constant::E));
                (e, args[0])
            } else if builtin == Some(BuiltinFn::Log) && args.len() == 2 {
                (args[0], args[1])
            } else if builtin == Some(BuiltinFn::Log) && args.len() == 1 {
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
        let (base, arg) = match ctx.builtin_of(name) {
            Some(BuiltinFn::Ln) if args.len() == 1 => {
                let e = ctx.add(Expr::Constant(cas_ast::Constant::E));
                (e, args[0])
            }
            Some(BuiltinFn::Log) if args.len() == 2 => (args[0], args[1]),
            _ => return None,
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
                if ctx.is_builtin(*fname, BuiltinFn::Exp) {
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
        let (base, arg) = match ctx.builtin_of(name) {
            Some(BuiltinFn::Ln) if args.len() == 1 => {
                let e = ctx.add(Expr::Constant(cas_ast::Constant::E));
                (e, args[0])
            }
            Some(BuiltinFn::Log) if args.len() == 2 => (args[0], args[1]),
            _ => return None,
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
        if !ctx.is_builtin(abs_name, BuiltinFn::Abs) || abs_args.len() != 1 {
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
            Expr::Function(fn_id, args)
                if ctx.is_builtin(fn_id, BuiltinFn::Ln) && args.len() == 1 =>
            {
                (None, args[0])
            }
            Expr::Function(fn_id, args)
                if ctx.is_builtin(fn_id, BuiltinFn::Log) && args.len() == 2 =>
            {
                (Some(args[0]), args[1])
            }
            _ => return None,
        };

        // Match abs(inner)
        let inner = match ctx.get(arg).clone() {
            Expr::Function(fn_id, args)
                if ctx.is_builtin(fn_id, BuiltinFn::Abs) && args.len() == 1 =>
            {
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
                if !super::exprs_match(ctx, *arg_i, *base_j) {
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
