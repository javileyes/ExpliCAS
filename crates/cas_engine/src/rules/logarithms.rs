use crate::define_rule;
use crate::ordering::compare_expr;
use crate::rule::Rewrite;
use crate::rules::algebra::helpers::smart_mul;
use cas_ast::{Context, Expr, ExprId};
use num_traits::{One, Zero};
use std::cmp::Ordering;

/// Helper: create log(base, arg) or ln(arg) depending on base.
/// If base is Constant::E, returns ln(arg) to preserve natural log notation.
fn make_log(ctx: &mut Context, base: ExprId, arg: ExprId) -> ExprId {
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
                return Some(Rewrite {
                    new_expr: zero,
                    description: "log(b, 1) = 0".to_string(),
                    before_local: None,
                    after_local: None,
                    assumption_events: Default::default(),
                });
            }
            if n.is_zero() {
                let inf = ctx.add(Expr::Constant(cas_ast::Constant::Infinity));
                let neg_inf = ctx.add(Expr::Neg(inf));
                return Some(Rewrite {
                    new_expr: neg_inf,
                    description: "log(b, 0) = -infinity".to_string(),
                    before_local: None,
                    after_local: None,
                    assumption_events: Default::default(),
                });
            }
            if *n < num_rational::BigRational::zero() {
                let undef = ctx.add(Expr::Constant(cas_ast::Constant::Undefined));
                return Some(Rewrite {
                    new_expr: undef,
                    description: "log(b, neg) = undefined".to_string(),
                    before_local: None,
                    after_local: None,
                    assumption_events: Default::default(),
                });
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
                            return Some(Rewrite {
                                new_expr,
                                description: format!("log({}, {}) = {}", b, n, power),
                                before_local: None,
                                after_local: None,
                                assumption_events: Default::default(),
                            });
                        }
                    }
                }
            }
        }

        // 2. log(b, b) = 1
        if base == arg || ctx.get(base) == ctx.get(arg) {
            let one = ctx.num(1);
            return Some(Rewrite {
                new_expr: one,
                description: "log(b, b) = 1".to_string(),
                before_local: None,
                after_local: None,
                assumption_events: Default::default(),
            });
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
                    return Some(Rewrite {
                        new_expr: p_exp,
                        description: "log(b, b^n) = n".to_string(),
                        before_local: None,
                        after_local: None,
                        assumption_events: Default::default(),
                    });
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
                    return Some(Rewrite {
                        new_expr,
                        description: "log(b, x^(even)) = even·log(b, |x|)".to_string(),
                        before_local: None,
                        after_local: None,
                        assumption_events: Default::default(), // No assumption needed!
                    });
                } else {
                    // Odd or non-integer exponent: requires x > 0
                    // Only allow in Generic/Assume modes, block in Strict
                    let log_inner = make_log(ctx, base, p_base);
                    let new_expr = smart_mul(ctx, p_exp, log_inner);
                    return Some(Rewrite {
                        new_expr,
                        description: "log(b, x^y) = y * log(b, x)".to_string(),
                        before_local: None,
                        after_local: None,
                        assumption_events: smallvec::smallvec![
                            crate::assumptions::AssumptionEvent::positive(ctx, p_base)
                        ],
                    });
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
                let mut events = smallvec::SmallVec::new();
                if lhs_decision.assumption.is_some() {
                    events.push(crate::assumptions::AssumptionEvent::positive(ctx, lhs));
                }
                if rhs_decision.assumption.is_some() {
                    events.push(crate::assumptions::AssumptionEvent::positive(ctx, rhs));
                }

                return Some(crate::rule::Rewrite {
                    new_expr,
                    description: "log(b, x*y) = log(b, x) + log(b, y)".to_string(),
                    before_local: None,
                    after_local: None,
                    assumption_events: events,
                });
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
                let mut events = smallvec::SmallVec::new();
                if num_decision.assumption.is_some() {
                    events.push(crate::assumptions::AssumptionEvent::positive(ctx, num));
                }
                if den_decision.assumption.is_some() {
                    events.push(crate::assumptions::AssumptionEvent::positive(ctx, den));
                }

                return Some(crate::rule::Rewrite {
                    new_expr,
                    description: "log(b, x/y) = log(b, x) - log(b, y)".to_string(),
                    before_local: None,
                    after_local: None,
                    assumption_events: events,
                });
            }
        }
        None
    }

    fn target_types(&self) -> Option<Vec<&str>> {
        Some(vec!["Function"])
    }
}

/// LogAbsSimplifyRule: Simplifies ln(|expr|) → ln(expr) when expr > 0.
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
            return Some(Rewrite {
                new_expr: mk_log(ctx),
                description: "ln(|x|) = ln(x) for x > 0".to_string(),
                before_local: None,
                after_local: None,
                assumption_events: Default::default(),
            });
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
                Some(Rewrite {
                    new_expr: mk_log(ctx),
                    description: "ln(|x|) = ln(x) for x > 0".to_string(),
                    before_local: None,
                    after_local: None,
                    assumption_events: Default::default(),
                })
            }
            DomainMode::Assume => {
                // In Assume mode: simplify with warning (assumption traceability)
                let events =
                    smallvec::smallvec![crate::assumptions::AssumptionEvent::positive(ctx, inner)];
                Some(Rewrite {
                    new_expr: mk_log(ctx),
                    description: "ln(|x|) = ln(x) (assuming x > 0)".to_string(),
                    before_local: None,
                    after_local: None,
                    assumption_events: events,
                })
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
                    return Some(crate::rule::Rewrite {
                        new_expr,
                        description: "ln(a) + ln(b) = ln(a*b)".to_string(),
                        before_local: None,
                        after_local: None,
                        assumption_events: Default::default(),
                    });
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
                    return Some(crate::rule::Rewrite {
                        new_expr,
                        description: "ln(a) - ln(b) = ln(a/b)".to_string(),
                        before_local: None,
                        after_local: None,
                        assumption_events: Default::default(),
                    });
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
                    let assumption_events = if decision.assumption.is_some() {
                        smallvec::smallvec![crate::assumptions::AssumptionEvent::positive(
                            ctx, log_arg
                        )]
                    } else {
                        Default::default()
                    };

                    return Some(crate::rule::Rewrite {
                        new_expr: log_arg,
                        description: "b^log(b, x) = x".to_string(),
                        before_local: None,
                        after_local: None,
                        assumption_events,
                    });
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
                            let assumption_events = if decision.assumption.is_some() {
                                smallvec::smallvec![crate::assumptions::AssumptionEvent::positive(
                                    ctx, log_arg
                                )]
                            } else {
                                Default::default()
                            };

                            return Some(crate::rule::Rewrite {
                                new_expr,
                                description: "b^(c*log(b, x)) = x^c".to_string(),
                                before_local: None,
                                after_local: None,
                                assumption_events,
                            });
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
}

define_rule!(SplitLogExponentsRule, "Split Log Exponents", |ctx, expr| {
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
                    return Some(Rewrite {
                        new_expr,
                        description: "e^(a+b) -> e^a * e^b (log cancellation)".to_string(),
                        before_local: None,
                        after_local: None,
                        assumption_events: Default::default(),
                    });
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

define_rule!(LogInversePowerRule, "Log Inverse Power", |ctx, expr| {
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
            return Some(Rewrite {
                new_expr,
                description: "x^(c/log(b, x)) = b^c".to_string(),
                before_local: None,
                after_local: None,
                assumption_events: Default::default(),
            });
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
                        return Some(crate::rule::Rewrite {
                            new_expr: p_exp,
                            description: "log(b, b^n) = n".to_string(),
                            before_local: None,
                            after_local: None,
                            assumption_events: Default::default(),
                        });
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
                                        return Some(crate::rule::Rewrite {
                                            new_expr: p_exp,
                                            description: "log(b, b^x) → x".to_string(),
                                            before_local: None,
                                            after_local: None,
                                            assumption_events: smallvec::smallvec![
                                                crate::assumptions::AssumptionEvent::positive(
                                                    ctx, base
                                                )
                                            ],
                                        });
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
                        return Some(crate::rule::Rewrite {
                            new_expr: p_exp,
                            description: "log(b, b^x) → x".to_string(),
                            before_local: None,
                            after_local: None,
                            assumption_events: Default::default(), // No assumption needed
                        });
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
}
