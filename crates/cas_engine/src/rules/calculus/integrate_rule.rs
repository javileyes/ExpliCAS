use crate::define_rule;
use crate::symbolic_calculus_call_support::{try_extract_integrate_call, NamedVarCall};
use cas_ast::{Context, Expr, ExprId};

use super::integration_derivative_cofactor_routes::polynomial_trig_reciprocal_derivative_root_gate_rewrite;
use super::integration_result_pipeline::standard_integration_rewrite;

define_rule!(IntegrateRule, "Symbolic Integration", |ctx, expr| {
    if let Some(mut definite_call) =
        crate::symbolic_calculus_call_support::try_extract_definite_integrate_call(ctx, expr)
    {
        if let Some(folded) =
            fold_var_power_quotient(ctx, definite_call.target, &definite_call.var_name)
        {
            definite_call.target = folded;
        }
        return super::definite_integration::definite_integration_rewrite(ctx, &definite_call);
    }
    let mut call = try_extract_integrate_call(ctx, expr)?;
    // Vector/matrix target: integrate componentwise (Fase 2 V7b), ALL-OR-NOTHING and
    // conditions-conservative: a component whose antiderivative carries required
    // conditions, still contains an integrate node (residual fallback), or fails
    // outright declines the WHOLE call — never a half-integrated matrix, and never a
    // silently dropped domain condition.
    if matches!(ctx.get(call.target), cas_ast::Expr::Matrix { .. }) {
        let var = call.var_name.clone();
        let rewritten =
            crate::matrix_rule_support::map_matrix_components(ctx, call.target, |ctx, entry| {
                let outcome = super::integration::integrate_with_trace(ctx, entry, &var)?;
                if !outcome.required_conditions.is_empty()
                    || contains_integrate_call(ctx, outcome.result)
                {
                    return None;
                }
                Some(outcome.result)
            })?;
        return Some(
            crate::rule::Rewrite::new(rewritten)
                .desc("Integrar cada componente del vector")
                .budget_exempt(),
        );
    }
    // The simplifier RATIONALIZES a fractional reciprocal power — `1/x^(1/3)`
    // becomes `x^(2/3)/x`, not `x^(-1/3)` — so the power-rule integrand matcher
    // (which recognizes `x^n`) missed it and `∫1/x^(1/3)` leaked. Fold a
    // `(c·)x^a/x^b` integrand back to `c·x^(a-b)` so the power rule applies.
    if let Some(folded) = fold_var_power_quotient(ctx, call.target, &call.var_name) {
        call = NamedVarCall {
            target: folded,
            var_name: call.var_name,
        };
    }
    if let Some(rewrite) = polynomial_trig_reciprocal_derivative_root_gate_rewrite(ctx, &call) {
        return Some(rewrite);
    }

    standard_integration_rewrite(ctx, &call)
});

/// Exponent of a PURE power of `var` (`x` → 1, `x^k` → k), else None.
fn pure_var_power(ctx: &Context, expr: ExprId, var: &str) -> Option<num_rational::BigRational> {
    match ctx.get(expr) {
        Expr::Variable(sym) if ctx.sym_name(*sym) == var => {
            Some(num_rational::BigRational::from_integer(1.into()))
        }
        Expr::Pow(base, exp) => {
            let (base, exp) = (*base, *exp);
            match ctx.get(base) {
                Expr::Variable(sym) if ctx.sym_name(*sym) == var => {
                    cas_math::numeric_eval::as_rational_const(ctx, exp)
                }
                _ => None,
            }
        }
        _ => None,
    }
}

/// Fold a quotient of `var`-powers with a rational-constant scale, `c·x^a / x^b`
/// (or `1/x^b`, `x^a/x^b`), to `c·x^(a-b)`. The simplifier rationalizes a
/// fractional reciprocal power into this shape (`1/x^(1/3) → x^(2/3)/x`), which
/// the power-rule matcher misses. The numerator may be a pure power OR a rational
/// constant (`1/x^(2/5)`); the denominator must be a pure power. Returns None for
/// any other integrand, so nothing else is disturbed.
fn fold_var_power_quotient(ctx: &mut Context, target: ExprId, var: &str) -> Option<ExprId> {
    use num_rational::BigRational;
    use num_traits::{One, Zero};
    let Expr::Div(num, den) = ctx.get(target) else {
        return None;
    };
    let (num, den) = (*num, *den);
    let b = pure_var_power(ctx, den, var)?;
    let (scale, a) = if let Some(a) = pure_var_power(ctx, num, var) {
        (BigRational::one(), a)
    } else if let Some(c) = cas_math::numeric_eval::as_rational_const(ctx, num) {
        if c.is_zero() {
            return None;
        }
        (c, BigRational::zero())
    } else {
        return None;
    };
    let e = a - b;
    // Only the FRACTIONAL-exponent forms are the bug (`1/x^(1/3) → x^(2/3)/x`).
    // Integer-exponent quotients (`1/x`, `1/x^2`, `x^3/x`) already integrate on
    // their existing path; leave them untouched to avoid any huella churn. (This
    // also excludes `e = 0`, so `x^a/x^a` is left alone.)
    if e.is_integer() {
        return None;
    }
    let power = {
        let var_expr = ctx.var(var);
        let exp = ctx.add(Expr::Number(e));
        ctx.add(Expr::Pow(var_expr, exp))
    };
    if scale.is_one() {
        Some(power)
    } else {
        let scale_expr = ctx.add(Expr::Number(scale));
        Some(ctx.add(Expr::Mul(scale_expr, power)))
    }
}

/// True when any node of `expr` is an `integrate(...)` call — the all-or-nothing
/// gate of the componentwise arm (a residual-fallback antiderivative must decline,
/// never ship inside a half-integrated matrix).
fn contains_integrate_call(ctx: &cas_ast::Context, expr: cas_ast::ExprId) -> bool {
    use cas_ast::Expr;
    let mut stack = vec![expr];
    while let Some(id) = stack.pop() {
        match ctx.get(id) {
            Expr::Function(f, args) => {
                let name = ctx.sym_name(*f);
                if name == "integrate" {
                    return true;
                }
                stack.extend(args.iter().copied());
            }
            Expr::Add(a, b)
            | Expr::Sub(a, b)
            | Expr::Mul(a, b)
            | Expr::Div(a, b)
            | Expr::Pow(a, b) => {
                stack.push(*a);
                stack.push(*b);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(*inner),
            Expr::Matrix { data, .. } => stack.extend(data.iter().copied()),
            Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => {}
        }
    }
    false
}
