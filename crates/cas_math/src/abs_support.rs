use cas_ast::ordering::compare_expr;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_integer::Integer;
use num_traits::Signed;

/// Extract the inner argument of `abs(x)`, returning `Some(x)` when expression
/// is an absolute value call.
pub fn try_unwrap_abs_arg(ctx: &Context, id: ExprId) -> Option<ExprId> {
    if let Expr::Function(fn_id, args) = ctx.get(id) {
        if ctx.is_builtin(*fn_id, BuiltinFn::Abs) && args.len() == 1 {
            return Some(args[0]);
        }
    }
    None
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AbsRewrite {
    pub rewritten: ExprId,
    pub desc: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbsDomainMode {
    Strict,
    Generic,
    Assume,
}

/// Derive [`AbsDomainMode`] from generic mode flags.
pub fn abs_domain_mode_from_flags(assume_mode: bool, strict_mode: bool) -> AbsDomainMode {
    if assume_mode {
        AbsDomainMode::Assume
    } else if strict_mode {
        AbsDomainMode::Strict
    } else {
        AbsDomainMode::Generic
    }
}

/// In `Strict/Generic`, implicit-domain scans are only needed when direct proof is missing.
pub fn abs_needs_implicit_domain_check(mode: AbsDomainMode, proven: bool) -> bool {
    matches!(mode, AbsDomainMode::Strict | AbsDomainMode::Generic) && !proven
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbsAssumptionKind {
    Positive,
    NonNegative,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbsDomainRewriteKind {
    Positive,
    PositiveAssume,
    NonNegative,
    NonNegativeAssume,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueDomainMode {
    RealOnly,
    ComplexEnabled,
}

/// Derive [`ValueDomainMode`] from a simple `real_only` flag.
pub fn value_domain_mode_from_flag(real_only: bool) -> ValueDomainMode {
    if real_only {
        ValueDomainMode::RealOnly
    } else {
        ValueDomainMode::ComplexEnabled
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AbsDomainRewrite {
    pub rewritten: ExprId,
    pub kind: AbsDomainRewriteKind,
    pub assumption: Option<AbsAssumptionKind>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolicRootCancelRewriteKind {
    AssumeNonNegative,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SymbolicRootCancelRewrite {
    pub rewritten: ExprId,
    pub kind: SymbolicRootCancelRewriteKind,
    pub requires_nonnegative: bool,
    pub assumption: Option<AbsAssumptionKind>,
}

/// Return true when expression is a direct call to `ln(...)` or `log(...)`.
pub fn is_ln_or_log_call(ctx: &Context, expr: ExprId) -> bool {
    let Expr::Function(fn_id, _) = ctx.get(expr) else {
        return false;
    };
    ctx.is_builtin(*fn_id, BuiltinFn::Ln) || ctx.is_builtin(*fn_id, BuiltinFn::Log)
}

/// Decide rewrite plan for `abs(x)` under a positivity proof context.
///
/// This is a pure planning helper: it does not create assumptions/events.
/// Runtime crates decide how to materialize `assumption` into diagnostics.
pub fn try_plan_abs_positive_rewrite(
    ctx: &Context,
    expr: ExprId,
    mode: AbsDomainMode,
    proven_positive: bool,
    implied_positive: bool,
) -> Option<AbsDomainRewrite> {
    let inner = try_unwrap_abs_arg(ctx, expr)?;
    let is_safe = proven_positive || implied_positive;

    match mode {
        AbsDomainMode::Strict | AbsDomainMode::Generic => {
            if !is_safe {
                return None;
            }
            Some(AbsDomainRewrite {
                rewritten: inner,
                kind: AbsDomainRewriteKind::Positive,
                assumption: None,
            })
        }
        AbsDomainMode::Assume => {
            if proven_positive {
                Some(AbsDomainRewrite {
                    rewritten: inner,
                    kind: AbsDomainRewriteKind::Positive,
                    assumption: None,
                })
            } else {
                Some(AbsDomainRewrite {
                    rewritten: inner,
                    kind: AbsDomainRewriteKind::PositiveAssume,
                    assumption: Some(AbsAssumptionKind::Positive),
                })
            }
        }
    }
}

/// Decide rewrite plan for `abs(x)` under a non-negativity proof context.
///
/// This is a pure planning helper: it does not create assumptions/events.
/// Runtime crates decide how to materialize `assumption` into diagnostics.
pub fn try_plan_abs_nonnegative_rewrite(
    ctx: &Context,
    expr: ExprId,
    mode: AbsDomainMode,
    proven_nonnegative: bool,
    implied_nonnegative: bool,
) -> Option<AbsDomainRewrite> {
    let inner = try_unwrap_abs_arg(ctx, expr)?;
    let is_safe = proven_nonnegative || implied_nonnegative;

    match mode {
        AbsDomainMode::Strict | AbsDomainMode::Generic => {
            if !is_safe {
                return None;
            }
            Some(AbsDomainRewrite {
                rewritten: inner,
                kind: AbsDomainRewriteKind::NonNegative,
                assumption: None,
            })
        }
        AbsDomainMode::Assume => {
            if proven_nonnegative {
                Some(AbsDomainRewrite {
                    rewritten: inner,
                    kind: AbsDomainRewriteKind::NonNegative,
                    assumption: None,
                })
            } else {
                Some(AbsDomainRewrite {
                    rewritten: inner,
                    kind: AbsDomainRewriteKind::NonNegativeAssume,
                    assumption: Some(AbsAssumptionKind::NonNegative),
                })
            }
        }
    }
}

/// Plan symbolic root cancel rewrite: `sqrt(x^n, n) -> x` in assume/real mode.
///
/// The rule is intentionally blocked outside `(domain=Assume, value_domain=RealOnly)`.
pub fn try_plan_symbolic_root_cancel_rewrite(
    ctx: &Context,
    expr: ExprId,
    mode: AbsDomainMode,
    value_domain: ValueDomainMode,
) -> Option<SymbolicRootCancelRewrite> {
    if mode != AbsDomainMode::Assume || value_domain != ValueDomainMode::RealOnly {
        return None;
    }

    let rewritten = try_extract_symbolic_root_cancel_base(ctx, expr)?;
    Some(SymbolicRootCancelRewrite {
        rewritten,
        kind: SymbolicRootCancelRewriteKind::AssumeNonNegative,
        requires_nonnegative: true,
        assumption: Some(AbsAssumptionKind::NonNegative),
    })
}

/// Evaluate simple `abs(...)` forms.
///
/// Supported:
/// - `abs(number) -> number_abs`
/// - `abs(-number) -> number_abs`
/// - `abs(-x) -> abs(x)`
pub fn try_rewrite_evaluate_abs_expr(ctx: &mut Context, expr: ExprId) -> Option<AbsRewrite> {
    let abs_arg = try_unwrap_abs_arg(ctx, expr)?;

    // Case 1: abs(number)
    if let Expr::Number(n) = ctx.get(abs_arg) {
        let n_abs = n.abs();
        let desc = format!("abs({}) = {}", n, n_abs);
        let rewritten = ctx.add(Expr::Number(n_abs));
        return Some(AbsRewrite { rewritten, desc });
    }

    // Case 2: abs(-x)
    if let Expr::Neg(inner) = ctx.get(abs_arg) {
        if let Expr::Number(n) = ctx.get(*inner) {
            let n_clone = n.clone();
            let desc = format!("abs(-{}) = {}", n, n);
            let rewritten = ctx.add(Expr::Number(n_clone));
            return Some(AbsRewrite { rewritten, desc });
        }

        let rewritten = ctx.call_builtin(BuiltinFn::Abs, vec![*inner]);
        return Some(AbsRewrite {
            rewritten,
            desc: "abs(-x) = abs(x)".to_string(),
        });
    }

    None
}

/// Rewrite `||x|| -> |x|`.
pub fn try_rewrite_abs_idempotent_expr(ctx: &Context, expr: ExprId) -> Option<AbsRewrite> {
    let outer_arg = try_unwrap_abs_arg(ctx, expr)?;
    let _inner_arg = try_unwrap_abs_arg(ctx, outer_arg)?;
    Some(AbsRewrite {
        rewritten: outer_arg,
        desc: "||x|| = |x|".to_string(),
    })
}

/// Rewrite `|x^n| -> x^n` for even integer `n`.
pub fn try_rewrite_abs_even_power_expr(ctx: &Context, expr: ExprId) -> Option<AbsRewrite> {
    let abs_arg = try_unwrap_abs_arg(ctx, expr)?;
    let Expr::Pow(_base, exp) = ctx.get(abs_arg) else {
        return None;
    };
    let Expr::Number(n) = ctx.get(*exp) else {
        return None;
    };
    if !(n.is_integer() && n.to_integer().is_even()) {
        return None;
    }

    Some(AbsRewrite {
        rewritten: abs_arg,
        desc: format!("|x^{}| = x^{}", n, n),
    })
}

/// Rewrite `|x^n| -> |x|^n` for positive odd integer `n`.
pub fn try_rewrite_abs_odd_power_expr(ctx: &mut Context, expr: ExprId) -> Option<AbsRewrite> {
    let abs_arg = try_unwrap_abs_arg(ctx, expr)?;
    let Expr::Pow(base, exp) = ctx.get(abs_arg) else {
        return None;
    };
    let (base, exp) = (*base, *exp);
    let Expr::Number(n) = ctx.get(exp) else {
        return None;
    };
    if !n.is_integer() {
        return None;
    }
    let n_int = n.to_integer();
    if !n_int.is_positive() || n_int.is_even() {
        return None;
    }

    let abs_base = ctx.call_builtin(BuiltinFn::Abs, vec![base]);
    let rewritten = ctx.add(Expr::Pow(abs_base, exp));
    Some(AbsRewrite {
        rewritten,
        desc: format!("|x^{}| = |x|^{}", n_int, n_int),
    })
}

/// Rewrite `|x|^n -> x^n` for even integer `n`.
pub fn try_rewrite_abs_power_even_expr(ctx: &mut Context, expr: ExprId) -> Option<AbsRewrite> {
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return None;
    };
    let (base, exp) = (*base, *exp);
    let inner = try_unwrap_abs_arg(ctx, base)?;
    let Expr::Number(n) = ctx.get(exp) else {
        return None;
    };
    if !(n.is_integer() && n.to_integer().is_even()) {
        return None;
    }
    let n_text = format!("{n}");

    let rewritten = ctx.add(Expr::Pow(inner, exp));
    Some(AbsRewrite {
        rewritten,
        desc: format!("|x|^{n_text} = x^{n_text}"),
    })
}

/// Rewrite `sqrt(x^2) -> |x|` and `(x^2)^(1/2) -> |x|`.
pub fn try_rewrite_sqrt_square_expr(ctx: &mut Context, expr: ExprId) -> Option<AbsRewrite> {
    let inner = if let Expr::Function(fn_id, args) = ctx.get(expr) {
        if ctx.is_builtin(*fn_id, BuiltinFn::Sqrt) && args.len() == 1 {
            Some(args[0])
        } else {
            None
        }
    } else if let Expr::Pow(base, exp) = ctx.get(expr) {
        if let Expr::Number(n) = ctx.get(*exp) {
            if *n.numer() == 1.into() && *n.denom() == 2.into() {
                Some(*base)
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    }?;

    let Expr::Pow(base, exp) = ctx.get(inner) else {
        return None;
    };
    let (base, exp) = (*base, *exp);
    let Expr::Number(n) = ctx.get(exp) else {
        return None;
    };
    if !(n.is_integer() && *n == num_rational::BigRational::from_integer(2.into())) {
        return None;
    }

    let rewritten = ctx.call_builtin(BuiltinFn::Abs, vec![base]);
    Some(AbsRewrite {
        rewritten,
        desc: "sqrt(x^2) = |x|".to_string(),
    })
}

/// Extract base from symbolic root-cancel pattern:
/// `sqrt(base^n, n)` where `n` is symbolic (non-numeric).
pub fn try_extract_symbolic_root_cancel_base(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if !ctx.is_builtin(*fn_id, BuiltinFn::Sqrt) || args.len() != 2 {
        return None;
    }

    let arg = args[0];
    let index = args[1];
    if matches!(ctx.get(index), Expr::Number(_)) {
        return None;
    }

    let (base, exp) = crate::expr_destructure::as_pow(ctx, arg)?;
    if compare_expr(ctx, exp, index) != std::cmp::Ordering::Equal {
        return None;
    }
    Some(base)
}

/// Rewrite `|e| -> e` when `e` is structurally non-negative according to
/// `is_sum_of_nonnegative`.
pub fn try_rewrite_abs_sum_nonnegative_expr(ctx: &Context, expr: ExprId) -> Option<AbsRewrite> {
    let abs_arg = try_unwrap_abs_arg(ctx, expr)?;
    if !is_sum_of_nonnegative(ctx, abs_arg) {
        return None;
    }
    Some(AbsRewrite {
        rewritten: abs_arg,
        desc: "|x² + ...| = x² + ...".to_string(),
    })
}

/// Rewrite `|a-b| -> |b-a|` for sub-like expressions when canonical ordering
/// prefers `b-a`, with complexity guards.
pub fn try_rewrite_abs_sub_normalize_expr(ctx: &mut Context, expr: ExprId) -> Option<AbsRewrite> {
    let abs_arg = try_unwrap_abs_arg(ctx, expr)?;
    let (a, b) = crate::expr_sub_like::extract_sub_like_pair(ctx, abs_arg)?;

    // Per-operand guard
    if !crate::expr_complexity::dedup_node_count_within(ctx, a, 20)
        || !crate::expr_complexity::dedup_node_count_within(ctx, b, 20)
    {
        return None;
    }
    // Whole expression guard
    if !crate::expr_complexity::dedup_node_count_within(ctx, expr, 60) {
        return None;
    }

    if compare_expr(ctx, a, b) != std::cmp::Ordering::Greater {
        return None;
    }

    let swapped = ctx.add(Expr::Sub(b, a));
    let rewritten = ctx.call_builtin(BuiltinFn::Abs, vec![swapped]);
    Some(AbsRewrite {
        rewritten,
        desc: "|a−b| = |b−a|".to_string(),
    })
}

/// Rewrite `|a| * |b|` into `|a * b|` when the input expression matches.
pub fn try_rewrite_abs_product_expr(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Mul(lhs, rhs) = ctx.get(expr) else {
        return None;
    };
    let lhs = *lhs;
    let rhs = *rhs;

    let (a, b) = (try_unwrap_abs_arg(ctx, lhs)?, try_unwrap_abs_arg(ctx, rhs)?);
    let product = crate::build::mul2_raw(ctx, a, b);
    Some(ctx.call_builtin(BuiltinFn::Abs, vec![product]))
}

/// Rewrite `|a| * |b|` into `|a * b|` with canonical description.
pub fn try_rewrite_abs_product_identity_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<AbsRewrite> {
    let rewritten = try_rewrite_abs_product_expr(ctx, expr)?;
    Some(AbsRewrite {
        rewritten,
        desc: "|x|·|y| = |x·y|".to_string(),
    })
}

/// Rewrite `|a| / |b|` into `|a / b|` when the input expression matches.
pub fn try_rewrite_abs_quotient_expr(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Div(lhs, rhs) = ctx.get(expr) else {
        return None;
    };
    let lhs = *lhs;
    let rhs = *rhs;

    let (a, b) = (try_unwrap_abs_arg(ctx, lhs)?, try_unwrap_abs_arg(ctx, rhs)?);
    let quotient = ctx.add(Expr::Div(a, b));
    Some(ctx.call_builtin(BuiltinFn::Abs, vec![quotient]))
}

/// Rewrite `|a| / |b|` into `|a / b|` with canonical description.
pub fn try_rewrite_abs_quotient_identity_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<AbsRewrite> {
    let rewritten = try_rewrite_abs_quotient_expr(ctx, expr)?;
    Some(AbsRewrite {
        rewritten,
        desc: "|x| / |y| = |x / y|".to_string(),
    })
}

/// Returns the argument when expression matches `abs(sqrt_like(arg))`.
///
/// `sqrt_like` includes:
/// - `sqrt(arg)`
/// - `arg^(1/2)`
pub fn try_extract_abs_sqrt_like_arg(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let abs_arg = try_unwrap_abs_arg(ctx, expr)?;

    if let Expr::Function(fn_id, _) = ctx.get(abs_arg) {
        if ctx.is_builtin(*fn_id, BuiltinFn::Sqrt) {
            return Some(abs_arg);
        }
    }

    if let Expr::Pow(_, exp) = ctx.get(abs_arg) {
        if let Expr::Number(n) = ctx.get(*exp) {
            if n.numer() == &1.into() && n.denom() == &2.into() {
                return Some(abs_arg);
            }
        }
    }

    None
}

/// Rewrite `|sqrt_like(x)| -> sqrt_like(x)` with canonical description.
pub fn try_rewrite_abs_sqrt_identity_expr(ctx: &Context, expr: ExprId) -> Option<AbsRewrite> {
    let rewritten = try_extract_abs_sqrt_like_arg(ctx, expr)?;
    Some(AbsRewrite {
        rewritten,
        desc: "|√x| = √x".to_string(),
    })
}

/// Returns the argument when expression matches `abs(exp_like(arg))`.
///
/// `exp_like` includes:
/// - `exp(arg)`
/// - `e^arg`
pub fn try_extract_abs_exp_like_arg(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let abs_arg = try_unwrap_abs_arg(ctx, expr)?;

    if let Expr::Function(fn_id, _) = ctx.get(abs_arg) {
        if ctx.is_builtin(*fn_id, BuiltinFn::Exp) {
            return Some(abs_arg);
        }
    }

    if let Expr::Pow(base, _) = ctx.get(abs_arg) {
        if let Expr::Constant(c) = ctx.get(*base) {
            if matches!(c, cas_ast::Constant::E) {
                return Some(abs_arg);
            }
        }
    }

    None
}

/// Rewrite `|exp_like(x)| -> exp_like(x)` with canonical description.
pub fn try_rewrite_abs_exp_identity_expr(ctx: &Context, expr: ExprId) -> Option<AbsRewrite> {
    let rewritten = try_extract_abs_exp_like_arg(ctx, expr)?;
    Some(AbsRewrite {
        rewritten,
        desc: "|e^x| = e^x".to_string(),
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AbsNumericFactorRewrite {
    pub rewritten: ExprId,
    pub desc: &'static str,
}

/// Rewrite `|k*x|`/`|x*k|` by extracting numeric factors:
/// - `|k*x| -> k*|x|` for `k > 0`
/// - `|(-k)*x| -> |k|*|x|` for `k < 0`
pub fn try_rewrite_abs_numeric_factor_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<AbsNumericFactorRewrite> {
    let abs_arg = try_unwrap_abs_arg(ctx, expr)?;
    let Expr::Mul(l, r) = ctx.get(abs_arg) else {
        return None;
    };
    let (l, r) = (*l, *r);

    if let Expr::Number(n) = ctx.get(l) {
        if n.is_positive() {
            let abs_r = ctx.call_builtin(BuiltinFn::Abs, vec![r]);
            let result = ctx.add(Expr::Mul(l, abs_r));
            return Some(AbsNumericFactorRewrite {
                rewritten: result,
                desc: "|k·x| = k·|x| for k > 0",
            });
        }
        if n.is_negative() {
            let abs_k = ctx.add(Expr::Number(n.abs()));
            let abs_r = ctx.call_builtin(BuiltinFn::Abs, vec![r]);
            let result = ctx.add(Expr::Mul(abs_k, abs_r));
            return Some(AbsNumericFactorRewrite {
                rewritten: result,
                desc: "|(-k)·x| = |k|·|x| for k < 0",
            });
        }
    }

    if let Expr::Number(n) = ctx.get(r) {
        if n.is_positive() {
            let abs_l = ctx.call_builtin(BuiltinFn::Abs, vec![l]);
            let result = ctx.add(Expr::Mul(r, abs_l));
            return Some(AbsNumericFactorRewrite {
                rewritten: result,
                desc: "|x·k| = k·|x| for k > 0",
            });
        }
        if n.is_negative() {
            let abs_k = ctx.add(Expr::Number(n.abs()));
            let abs_l = ctx.call_builtin(BuiltinFn::Abs, vec![l]);
            let result = ctx.add(Expr::Mul(abs_k, abs_l));
            return Some(AbsNumericFactorRewrite {
                rewritten: result,
                desc: "|x·(-k)| = |k|·|x| for k < 0",
            });
        }
    }

    None
}

/// Returns true when an expression is provably non-negative via structural rules.
///
/// This is intentionally conservative and syntactic:
/// - even powers
/// - `abs`, `sqrt`, `exp`
/// - non-negative numeric literals
/// - sums/products of non-negative expressions
pub fn is_sum_of_nonnegative(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        // x^(2k) is non-negative
        Expr::Pow(_, exp) => {
            if let Expr::Number(n) = ctx.get(*exp) {
                n.is_integer() && n.to_integer().is_even()
            } else {
                false
            }
        }
        // abs(x), sqrt(x), exp(x) are non-negative / positive on reals.
        Expr::Function(name, _) if ctx.is_builtin(*name, BuiltinFn::Abs) => true,
        Expr::Function(name, _) if ctx.is_builtin(*name, BuiltinFn::Sqrt) => true,
        Expr::Function(name, _) if ctx.is_builtin(*name, BuiltinFn::Exp) => true,
        Expr::Number(n) => !n.is_negative(),
        Expr::Add(l, r) => is_sum_of_nonnegative(ctx, *l) && is_sum_of_nonnegative(ctx, *r),
        Expr::Mul(l, r) => is_sum_of_nonnegative(ctx, *l) && is_sum_of_nonnegative(ctx, *r),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        abs_domain_mode_from_flags, abs_needs_implicit_domain_check, is_ln_or_log_call,
        is_sum_of_nonnegative, try_extract_abs_exp_like_arg, try_extract_abs_sqrt_like_arg,
        try_extract_symbolic_root_cancel_base, try_plan_symbolic_root_cancel_rewrite,
        try_rewrite_abs_even_power_expr, try_rewrite_abs_exp_identity_expr,
        try_rewrite_abs_idempotent_expr, try_rewrite_abs_numeric_factor_expr,
        try_rewrite_abs_odd_power_expr, try_rewrite_abs_power_even_expr,
        try_rewrite_abs_product_expr, try_rewrite_abs_product_identity_expr,
        try_rewrite_abs_quotient_expr, try_rewrite_abs_quotient_identity_expr,
        try_rewrite_abs_sqrt_identity_expr, try_rewrite_abs_sub_normalize_expr,
        try_rewrite_abs_sum_nonnegative_expr, try_rewrite_evaluate_abs_expr,
        try_rewrite_sqrt_square_expr, try_unwrap_abs_arg, value_domain_mode_from_flag,
        AbsDomainMode, ValueDomainMode,
    };
    use cas_ast::{BuiltinFn, Context, Expr};

    #[test]
    fn classifies_basic_nonnegative_forms() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let square = ctx.add(Expr::Pow(x, two));
        let abs_x = ctx.call_builtin(BuiltinFn::Abs, vec![x]);
        let sqrt_x = ctx.call_builtin(BuiltinFn::Sqrt, vec![x]);
        let exp_x = ctx.call_builtin(BuiltinFn::Exp, vec![x]);

        assert!(is_sum_of_nonnegative(&ctx, square));
        assert!(is_sum_of_nonnegative(&ctx, abs_x));
        assert!(is_sum_of_nonnegative(&ctx, sqrt_x));
        assert!(is_sum_of_nonnegative(&ctx, exp_x));
    }

    #[test]
    fn classifies_composed_forms() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let two = ctx.num(2);
        let xsq = ctx.add(Expr::Pow(x, two));
        let ysq = ctx.add(Expr::Pow(y, two));
        let sum = ctx.add(Expr::Add(xsq, ysq));
        let prod = ctx.add(Expr::Mul(xsq, ysq));

        assert!(is_sum_of_nonnegative(&ctx, sum));
        assert!(is_sum_of_nonnegative(&ctx, prod));
    }

    #[test]
    fn rejects_unknown_or_negative_forms() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let neg = ctx.num(-1);
        let one = ctx.num(1);
        let linear = ctx.add(Expr::Add(x, one));

        assert!(!is_sum_of_nonnegative(&ctx, x));
        assert!(!is_sum_of_nonnegative(&ctx, neg));
        assert!(!is_sum_of_nonnegative(&ctx, linear));
    }

    #[test]
    fn unwrap_abs_arg_for_abs_calls_only() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let abs_x = ctx.call_builtin(BuiltinFn::Abs, vec![x]);
        let not_abs = ctx.call_builtin(BuiltinFn::Sqrt, vec![x]);

        assert_eq!(try_unwrap_abs_arg(&ctx, abs_x), Some(x));
        assert_eq!(try_unwrap_abs_arg(&ctx, not_abs), None);
    }

    #[test]
    fn rewrites_abs_product_shape() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let abs_x = ctx.call_builtin(BuiltinFn::Abs, vec![x]);
        let abs_y = ctx.call_builtin(BuiltinFn::Abs, vec![y]);
        let expr = ctx.add(Expr::Mul(abs_x, abs_y));

        let rewritten = try_rewrite_abs_product_expr(&mut ctx, expr).expect("rewrite");
        let Expr::Function(fn_id, args) = ctx.get(rewritten) else {
            panic!("expected abs call");
        };
        assert!(ctx.is_builtin(*fn_id, BuiltinFn::Abs));
        assert_eq!(args.len(), 1);
    }

    #[test]
    fn rewrites_abs_quotient_shape() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let abs_x = ctx.call_builtin(BuiltinFn::Abs, vec![x]);
        let abs_y = ctx.call_builtin(BuiltinFn::Abs, vec![y]);
        let expr = ctx.add(Expr::Div(abs_x, abs_y));

        let rewritten = try_rewrite_abs_quotient_expr(&mut ctx, expr).expect("rewrite");
        let Expr::Function(fn_id, args) = ctx.get(rewritten) else {
            panic!("expected abs call");
        };
        assert!(ctx.is_builtin(*fn_id, BuiltinFn::Abs));
        assert_eq!(args.len(), 1);
    }

    #[test]
    fn rewrites_abs_product_identity_with_desc() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let abs_x = ctx.call_builtin(BuiltinFn::Abs, vec![x]);
        let abs_y = ctx.call_builtin(BuiltinFn::Abs, vec![y]);
        let expr = ctx.add(Expr::Mul(abs_x, abs_y));
        let rewrite = try_rewrite_abs_product_identity_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(rewrite.desc, "|x|·|y| = |x·y|");
    }

    #[test]
    fn rewrites_abs_quotient_identity_with_desc() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let abs_x = ctx.call_builtin(BuiltinFn::Abs, vec![x]);
        let abs_y = ctx.call_builtin(BuiltinFn::Abs, vec![y]);
        let expr = ctx.add(Expr::Div(abs_x, abs_y));
        let rewrite = try_rewrite_abs_quotient_identity_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(rewrite.desc, "|x| / |y| = |x / y|");
    }

    #[test]
    fn extracts_abs_sqrt_like_for_sqrt_and_pow_half() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let sqrt_x = ctx.call_builtin(BuiltinFn::Sqrt, vec![x]);
        let abs_sqrt = ctx.call_builtin(BuiltinFn::Abs, vec![sqrt_x]);
        let half = ctx.add(Expr::Number(num_rational::BigRational::new(
            1.into(),
            2.into(),
        )));
        let pow_half = ctx.add(Expr::Pow(x, half));
        let abs_pow_half = ctx.call_builtin(BuiltinFn::Abs, vec![pow_half]);

        assert_eq!(try_extract_abs_sqrt_like_arg(&ctx, abs_sqrt), Some(sqrt_x));
        assert_eq!(
            try_extract_abs_sqrt_like_arg(&ctx, abs_pow_half),
            Some(pow_half)
        );
    }

    #[test]
    fn rewrites_abs_sqrt_identity_with_desc() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let sqrt_x = ctx.call_builtin(BuiltinFn::Sqrt, vec![x]);
        let abs_sqrt = ctx.call_builtin(BuiltinFn::Abs, vec![sqrt_x]);
        let rewrite = try_rewrite_abs_sqrt_identity_expr(&ctx, abs_sqrt).expect("rewrite");
        assert_eq!(rewrite.desc, "|√x| = √x");
        assert_eq!(rewrite.rewritten, sqrt_x);
    }

    #[test]
    fn extracts_abs_exp_like_for_exp_and_e_pow() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let exp_x = ctx.call_builtin(BuiltinFn::Exp, vec![x]);
        let abs_exp = ctx.call_builtin(BuiltinFn::Abs, vec![exp_x]);
        let e = ctx.add(Expr::Constant(cas_ast::Constant::E));
        let e_pow_x = ctx.add(Expr::Pow(e, x));
        let abs_e_pow = ctx.call_builtin(BuiltinFn::Abs, vec![e_pow_x]);

        assert_eq!(try_extract_abs_exp_like_arg(&ctx, abs_exp), Some(exp_x));
        assert_eq!(try_extract_abs_exp_like_arg(&ctx, abs_e_pow), Some(e_pow_x));
    }

    #[test]
    fn rewrites_abs_exp_identity_with_desc() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let exp_x = ctx.call_builtin(BuiltinFn::Exp, vec![x]);
        let abs_exp = ctx.call_builtin(BuiltinFn::Abs, vec![exp_x]);
        let rewrite = try_rewrite_abs_exp_identity_expr(&ctx, abs_exp).expect("rewrite");
        assert_eq!(rewrite.desc, "|e^x| = e^x");
        assert_eq!(rewrite.rewritten, exp_x);
    }

    #[test]
    fn rewrites_abs_numeric_factor_left_and_right() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let minus_three = ctx.num(-3);
        let left_pos_inner = ctx.add(Expr::Mul(two, x));
        let right_pos_inner = ctx.add(Expr::Mul(x, two));
        let left_neg_inner = ctx.add(Expr::Mul(minus_three, x));
        let right_neg_inner = ctx.add(Expr::Mul(x, minus_three));

        let left_pos = ctx.call_builtin(BuiltinFn::Abs, vec![left_pos_inner]);
        let right_pos = ctx.call_builtin(BuiltinFn::Abs, vec![right_pos_inner]);
        let left_neg = ctx.call_builtin(BuiltinFn::Abs, vec![left_neg_inner]);
        let right_neg = ctx.call_builtin(BuiltinFn::Abs, vec![right_neg_inner]);

        let desc_left_pos = try_rewrite_abs_numeric_factor_expr(&mut ctx, left_pos)
            .expect("left_pos")
            .desc;
        let desc_right_pos = try_rewrite_abs_numeric_factor_expr(&mut ctx, right_pos)
            .expect("right_pos")
            .desc;
        let desc_left_neg = try_rewrite_abs_numeric_factor_expr(&mut ctx, left_neg)
            .expect("left_neg")
            .desc;
        let desc_right_neg = try_rewrite_abs_numeric_factor_expr(&mut ctx, right_neg)
            .expect("right_neg")
            .desc;

        assert!(
            desc_left_pos == "|k·x| = k·|x| for k > 0"
                || desc_left_pos == "|x·k| = k·|x| for k > 0"
        );
        assert!(
            desc_right_pos == "|k·x| = k·|x| for k > 0"
                || desc_right_pos == "|x·k| = k·|x| for k > 0"
        );
        assert!(
            desc_left_neg == "|(-k)·x| = |k|·|x| for k < 0"
                || desc_left_neg == "|x·(-k)| = |k|·|x| for k < 0"
        );
        assert!(
            desc_right_neg == "|(-k)·x| = |k|·|x| for k < 0"
                || desc_right_neg == "|x·(-k)| = |k|·|x| for k < 0"
        );
    }

    #[test]
    fn evaluates_abs_number_and_negation() {
        let mut ctx = Context::new();
        let minus_five = ctx.num(-5);
        let abs_num = ctx.call_builtin(BuiltinFn::Abs, vec![minus_five]);
        let rewrite = try_rewrite_evaluate_abs_expr(&mut ctx, abs_num).expect("rewrite");
        assert!(rewrite.desc.contains("abs("));

        let x = ctx.var("x");
        let neg_x = ctx.add(Expr::Neg(x));
        let abs_neg_x = ctx.call_builtin(BuiltinFn::Abs, vec![neg_x]);
        let rewrite2 = try_rewrite_evaluate_abs_expr(&mut ctx, abs_neg_x).expect("rewrite2");
        let Expr::Function(fn_id, args) = ctx.get(rewrite2.rewritten) else {
            panic!("expected abs call");
        };
        assert!(ctx.is_builtin(*fn_id, BuiltinFn::Abs));
        assert_eq!(args, &vec![x]);
    }

    #[test]
    fn rewrites_abs_idempotent() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let abs_x = ctx.call_builtin(BuiltinFn::Abs, vec![x]);
        let abs_abs_x = ctx.call_builtin(BuiltinFn::Abs, vec![abs_x]);
        let rewrite = try_rewrite_abs_idempotent_expr(&ctx, abs_abs_x).expect("rewrite");
        assert_eq!(rewrite.rewritten, abs_x);
    }

    #[test]
    fn rewrites_abs_even_and_odd_power() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let three = ctx.num(3);
        let x2 = ctx.add(Expr::Pow(x, two));
        let x3 = ctx.add(Expr::Pow(x, three));
        let abs_x2 = ctx.call_builtin(BuiltinFn::Abs, vec![x2]);
        let abs_x3 = ctx.call_builtin(BuiltinFn::Abs, vec![x3]);

        let even = try_rewrite_abs_even_power_expr(&ctx, abs_x2).expect("even");
        assert_eq!(even.rewritten, x2);

        let odd = try_rewrite_abs_odd_power_expr(&mut ctx, abs_x3).expect("odd");
        let Expr::Pow(base, exp) = ctx.get(odd.rewritten) else {
            panic!("expected power");
        };
        let Expr::Function(fn_id, args) = ctx.get(*base) else {
            panic!("expected abs base");
        };
        assert!(ctx.is_builtin(*fn_id, BuiltinFn::Abs));
        assert_eq!(args, &vec![x]);
        assert_eq!(*exp, three);
    }

    #[test]
    fn rewrites_abs_power_even() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let abs_x = ctx.call_builtin(BuiltinFn::Abs, vec![x]);
        let four = ctx.num(4);
        let expr = ctx.add(Expr::Pow(abs_x, four));

        let rewrite = try_rewrite_abs_power_even_expr(&mut ctx, expr).expect("rewrite");
        let Expr::Pow(base, exp) = ctx.get(rewrite.rewritten) else {
            panic!("expected power");
        };
        assert_eq!(*base, x);
        assert_eq!(*exp, four);
    }

    #[test]
    fn rewrites_sqrt_square_forms() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let x2 = ctx.add(Expr::Pow(x, two));
        let sqrt_x2 = ctx.call_builtin(BuiltinFn::Sqrt, vec![x2]);
        let half = ctx.add(Expr::Number(num_rational::BigRational::new(
            1.into(),
            2.into(),
        )));
        let pow_half = ctx.add(Expr::Pow(x2, half));

        let rewrite1 = try_rewrite_sqrt_square_expr(&mut ctx, sqrt_x2).expect("sqrt rewrite");
        let rewrite2 = try_rewrite_sqrt_square_expr(&mut ctx, pow_half).expect("pow rewrite");

        for rewritten in [rewrite1.rewritten, rewrite2.rewritten] {
            let Expr::Function(fn_id, args) = ctx.get(rewritten) else {
                panic!("expected abs call");
            };
            assert!(ctx.is_builtin(*fn_id, BuiltinFn::Abs));
            assert_eq!(args, &vec![x]);
        }
    }

    #[test]
    fn rewrites_abs_sum_nonnegative_and_sub_normalize() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let two = ctx.num(2);
        let x2 = ctx.add(Expr::Pow(x, two));
        let y2 = ctx.add(Expr::Pow(y, two));
        let sum = ctx.add(Expr::Add(x2, y2));
        let abs_sum = ctx.call_builtin(BuiltinFn::Abs, vec![sum]);
        let rewrite = try_rewrite_abs_sum_nonnegative_expr(&ctx, abs_sum).expect("sum rewrite");
        assert_eq!(rewrite.rewritten, sum);

        let a = ctx.var("z");
        let b = ctx.var("a");
        let sub = ctx.add(Expr::Sub(a, b));
        let abs_sub = ctx.call_builtin(BuiltinFn::Abs, vec![sub]);
        let rewrite2 = try_rewrite_abs_sub_normalize_expr(&mut ctx, abs_sub).expect("sub rewrite");
        let Expr::Function(fn_id, args) = ctx.get(rewrite2.rewritten) else {
            panic!("expected abs");
        };
        assert!(ctx.is_builtin(*fn_id, BuiltinFn::Abs));
        assert_eq!(args.len(), 1);
    }

    #[test]
    fn extracts_symbolic_root_cancel_base() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let n = ctx.var("n");
        let x_pow_n = ctx.add(Expr::Pow(x, n));
        let symbolic = ctx.call_builtin(BuiltinFn::Sqrt, vec![x_pow_n, n]);

        let base = try_extract_symbolic_root_cancel_base(&ctx, symbolic).expect("pattern");
        assert_eq!(base, x);

        let two = ctx.num(2);
        let numeric = ctx.call_builtin(BuiltinFn::Sqrt, vec![x_pow_n, two]);
        assert_eq!(try_extract_symbolic_root_cancel_base(&ctx, numeric), None);
    }

    #[test]
    fn detects_ln_or_log_parent_calls() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let ln_x = ctx.call_builtin(BuiltinFn::Ln, vec![x]);
        let log_x = ctx.call_builtin(BuiltinFn::Log, vec![x]);
        let sqrt_x = ctx.call_builtin(BuiltinFn::Sqrt, vec![x]);

        assert!(is_ln_or_log_call(&ctx, ln_x));
        assert!(is_ln_or_log_call(&ctx, log_x));
        assert!(!is_ln_or_log_call(&ctx, sqrt_x));
    }

    #[test]
    fn plans_symbolic_root_cancel_only_in_assume_real_mode() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let n = ctx.var("n");
        let x_pow_n = ctx.add(Expr::Pow(x, n));
        let expr = ctx.call_builtin(BuiltinFn::Sqrt, vec![x_pow_n, n]);

        let plan = try_plan_symbolic_root_cancel_rewrite(
            &ctx,
            expr,
            AbsDomainMode::Assume,
            ValueDomainMode::RealOnly,
        )
        .expect("plan");
        assert_eq!(plan.rewritten, x);
        assert!(plan.requires_nonnegative);

        assert!(try_plan_symbolic_root_cancel_rewrite(
            &ctx,
            expr,
            AbsDomainMode::Generic,
            ValueDomainMode::RealOnly
        )
        .is_none());
        assert!(try_plan_symbolic_root_cancel_rewrite(
            &ctx,
            expr,
            AbsDomainMode::Assume,
            ValueDomainMode::ComplexEnabled
        )
        .is_none());
    }

    #[test]
    fn abs_domain_mode_from_flags_prioritizes_assume_then_strict() {
        assert_eq!(
            abs_domain_mode_from_flags(true, true),
            AbsDomainMode::Assume
        );
        assert_eq!(
            abs_domain_mode_from_flags(false, true),
            AbsDomainMode::Strict
        );
        assert_eq!(
            abs_domain_mode_from_flags(false, false),
            AbsDomainMode::Generic
        );
    }

    #[test]
    fn abs_needs_implicit_domain_check_matches_policy() {
        assert!(abs_needs_implicit_domain_check(
            AbsDomainMode::Strict,
            false
        ));
        assert!(abs_needs_implicit_domain_check(
            AbsDomainMode::Generic,
            false
        ));
        assert!(!abs_needs_implicit_domain_check(
            AbsDomainMode::Assume,
            false
        ));
        assert!(!abs_needs_implicit_domain_check(
            AbsDomainMode::Strict,
            true
        ));
    }

    #[test]
    fn value_domain_mode_from_flag_maps_expected_modes() {
        assert_eq!(value_domain_mode_from_flag(true), ValueDomainMode::RealOnly);
        assert_eq!(
            value_domain_mode_from_flag(false),
            ValueDomainMode::ComplexEnabled
        );
    }
}
