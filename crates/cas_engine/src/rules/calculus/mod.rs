//! Calculus rules: differentiation, integration, summation, and products.
//!
//! This module is split into submodules:
//! - `differentiation`: symbolic derivative computation
//! - `integration`: symbolic integral computation + helpers
//! - `summation`: finite sum/product evaluation (SumRule, ProductRule)

mod differentiation;
mod integration;
mod summation;

use crate::define_rule;
use crate::rule::Rewrite;
use crate::symbolic_calculus_call_support::{
    render_diff_desc_with, render_integrate_desc_with, try_extract_diff_call,
    try_extract_integrate_call,
};
use cas_ast::{BuiltinFn, Context, Expr, ExprId};

use differentiation::differentiate;
use integration::{
    integrate, integrate_required_nonzero_conditions, integrate_required_positive_conditions,
};

fn atanh_diff_required_conditions(
    ctx: &mut Context,
    target: ExprId,
) -> Vec<crate::ImplicitCondition> {
    let (fn_id, args) = match ctx.get(target).clone() {
        Expr::Function(fn_id, args) => (fn_id, args),
        _ => return vec![],
    };

    if ctx.builtin_of(fn_id) != Some(BuiltinFn::Atanh) || args.len() != 1 {
        return vec![];
    }

    let arg = args[0];
    let one = ctx.num(1);
    let two = ctx.num(2);
    let arg_sq = ctx.add(Expr::Pow(arg, two));
    let open_interval = ctx.add(Expr::Sub(one, arg_sq));
    vec![crate::ImplicitCondition::Positive(open_interval)]
}

fn collect_atanh_open_interval_conditions(ctx: &mut Context, root: ExprId) -> Vec<ExprId> {
    let mut out = Vec::new();
    let mut stack = vec![root];

    while let Some(expr) = stack.pop() {
        match ctx.get(expr).clone() {
            Expr::Function(fn_id, args) => {
                if ctx.builtin_of(fn_id) == Some(BuiltinFn::Atanh) && args.len() == 1 {
                    let arg = args[0];
                    let one = ctx.num(1);
                    let two = ctx.num(2);
                    let arg_sq = ctx.add(Expr::Pow(arg, two));
                    out.push(ctx.add(Expr::Sub(one, arg_sq)));
                }
                stack.extend(args);
            }
            Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) => {
                stack.push(l);
                stack.push(r);
            }
            Expr::Pow(base, exp) => {
                stack.push(base);
                stack.push(exp);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(inner),
            Expr::Matrix { data, .. } => stack.extend(data),
            Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => {}
        }
    }

    out
}

define_rule!(IntegrateRule, "Symbolic Integration", |ctx, expr| {
    let call = try_extract_integrate_call(ctx, expr)?;
    let required_nonzero = integrate_required_nonzero_conditions(ctx, call.target, &call.var_name);
    let mut required_positive =
        integrate_required_positive_conditions(ctx, call.target, &call.var_name);
    let result = integrate(ctx, call.target, &call.var_name)?;
    if required_positive.is_empty() {
        required_positive.extend(collect_atanh_open_interval_conditions(ctx, result));
    }
    let desc = render_integrate_desc_with(&call, |id| {
        format!("{}", cas_formatter::DisplayExpr { context: ctx, id })
    });
    let required_conditions = required_nonzero
        .into_iter()
        .map(crate::ImplicitCondition::NonZero)
        .chain(
            required_positive
                .into_iter()
                .map(crate::ImplicitCondition::Positive),
        );
    Some(
        Rewrite::new(result)
            .desc(desc)
            .requires_all(required_conditions),
    )
});

define_rule!(DiffRule, "Symbolic Differentiation", |ctx, expr| {
    let call = try_extract_diff_call(ctx, expr)?;
    let result = differentiate(ctx, call.target, &call.var_name)?;
    let required_conditions = atanh_diff_required_conditions(ctx, call.target);
    let desc = render_diff_desc_with(&call, |id| {
        format!("{}", cas_formatter::DisplayExpr { context: ctx, id })
    });
    Some(
        Rewrite::new(result)
            .desc(desc)
            .requires_all(required_conditions),
    )
});

pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(IntegrateRule));
    simplifier.add_rule(Box::new(DiffRule));
    simplifier.add_rule(Box::new(summation::SumRule));
    simplifier.add_rule(Box::new(summation::ProductRule));
}
