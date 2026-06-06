use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_traits::One;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrigLogSourceConditionAlias {
    pub source_pole: ExprId,
    pub denominator: ExprId,
    pub factored_pole: ExprId,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TrigLogPoleOrientation {
    RatioMinusOffset,
    OffsetMinusRatio,
}

pub fn integrate_reciprocal_trig_log_source_condition_aliases(
    ctx: &mut Context,
    resolved: ExprId,
) -> Vec<TrigLogSourceConditionAlias> {
    let Expr::Function(fn_id, args) = ctx.get(resolved).clone() else {
        return Vec::new();
    };
    if ctx.sym_name(fn_id) != "integrate" || args.len() != 2 {
        return Vec::new();
    }

    let target = cas_ast::hold::strip_all_holds(ctx, args[0]);
    let Expr::Div(num, den) = ctx.get(target).clone() else {
        return Vec::new();
    };
    if cas_ast::views::as_rational_const(ctx, num, 8).is_none_or(|value| !value.is_one()) {
        return Vec::new();
    }

    let factors = crate::expr_nary::mul_leaves(ctx, den);
    if factors.len() != 2 {
        return Vec::new();
    }

    let mut aliases = Vec::new();
    for (ln_index, ln_factor) in factors.iter().enumerate() {
        let ln_factor = cas_ast::hold::strip_all_holds(ctx, *ln_factor);
        let shifted_factor = cas_ast::hold::strip_all_holds(ctx, factors[1 - ln_index]);
        for (ratio_builtin, ratio_num_builtin, ratio_den_builtin) in [
            (BuiltinFn::Tan, BuiltinFn::Sin, BuiltinFn::Cos),
            (BuiltinFn::Cot, BuiltinFn::Cos, BuiltinFn::Sin),
        ] {
            let Some(arg) = ln_unary_builtin_arg(ctx, ln_factor, ratio_builtin) else {
                continue;
            };
            let Some((source_pole, factored_pole)) = source_and_factored_trig_poles(
                ctx,
                shifted_factor,
                ratio_builtin,
                ratio_num_builtin,
                ratio_den_builtin,
                arg,
            ) else {
                continue;
            };
            aliases.push(TrigLogSourceConditionAlias {
                source_pole,
                denominator: ctx.call_builtin(ratio_den_builtin, vec![arg]),
                factored_pole,
            });
        }
    }
    aliases
}

pub fn reciprocal_trig_log_factored_pole_matches_source(
    ctx: &Context,
    factored_pole: ExprId,
    source_pole: ExprId,
) -> bool {
    let Some((builtin, arg, offset, orientation)) =
        factored_reciprocal_trig_log_pole(ctx, factored_pole)
    else {
        return false;
    };

    source_reciprocal_trig_log_pole_matches(ctx, source_pole, builtin, arg, offset, orientation)
}

fn ln_unary_builtin_arg(ctx: &Context, expr: ExprId, arg_builtin: BuiltinFn) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_hold(ctx, expr);
    let Expr::Function(ln_id, ln_args) = ctx.get(expr) else {
        return None;
    };
    if ctx.builtin_of(*ln_id) != Some(BuiltinFn::Ln) || ln_args.len() != 1 {
        return None;
    }
    let Expr::Function(arg_id, arg_args) = ctx.get(ln_args[0]) else {
        return None;
    };
    (ctx.builtin_of(*arg_id) == Some(arg_builtin) && arg_args.len() == 1).then_some(arg_args[0])
}

fn source_and_factored_trig_poles(
    ctx: &mut Context,
    shifted_factor: ExprId,
    ratio_builtin: BuiltinFn,
    ratio_num_builtin: BuiltinFn,
    ratio_den_builtin: BuiltinFn,
    arg: ExprId,
) -> Option<(ExprId, ExprId)> {
    let Expr::Sub(lhs, rhs) = ctx.get(shifted_factor).clone() else {
        return None;
    };

    let lhs_ratio_arg = unary_builtin_arg(ctx, lhs, ratio_builtin);
    let rhs_ratio_arg = unary_builtin_arg(ctx, rhs, ratio_builtin);
    let ratio_num = ctx.call_builtin(ratio_num_builtin, vec![arg]);
    let ratio_den = ctx.call_builtin(ratio_den_builtin, vec![arg]);

    if lhs_ratio_arg == Some(arg) && rhs_ratio_arg.is_none() {
        let scaled_offset = ctx.add(Expr::Mul(rhs, ratio_den));
        let factored = ctx.add(Expr::Sub(ratio_num, scaled_offset));
        return Some((shifted_factor, factored));
    }
    if rhs_ratio_arg == Some(arg) && lhs_ratio_arg.is_none() {
        let scaled_offset = ctx.add(Expr::Mul(lhs, ratio_den));
        let factored = ctx.add(Expr::Sub(scaled_offset, ratio_num));
        return Some((shifted_factor, factored));
    }

    None
}

fn factored_reciprocal_trig_log_pole(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BuiltinFn, ExprId, ExprId, TrigLogPoleOrientation)> {
    let expr = cas_ast::hold::unwrap_hold(ctx, expr);
    let Expr::Sub(left, right) = ctx.get(expr) else {
        return None;
    };

    if let Some((offset, arg)) = mul_by_unary_builtin_arg(ctx, *left, BuiltinFn::Cos) {
        if unary_builtin_arg(ctx, *right, BuiltinFn::Sin) == Some(arg) {
            return Some((
                BuiltinFn::Tan,
                arg,
                offset,
                TrigLogPoleOrientation::OffsetMinusRatio,
            ));
        }
    }
    if let Some(arg) = unary_builtin_arg(ctx, *left, BuiltinFn::Sin) {
        if let Some((offset, rhs_arg)) = mul_by_unary_builtin_arg(ctx, *right, BuiltinFn::Cos) {
            if crate::expr_domain::exprs_equivalent(ctx, arg, rhs_arg) {
                return Some((
                    BuiltinFn::Tan,
                    arg,
                    offset,
                    TrigLogPoleOrientation::RatioMinusOffset,
                ));
            }
        }
    }
    if let Some((offset, arg)) = mul_by_unary_builtin_arg(ctx, *left, BuiltinFn::Sin) {
        if unary_builtin_arg(ctx, *right, BuiltinFn::Cos) == Some(arg) {
            return Some((
                BuiltinFn::Cot,
                arg,
                offset,
                TrigLogPoleOrientation::OffsetMinusRatio,
            ));
        }
    }
    if let Some(arg) = unary_builtin_arg(ctx, *left, BuiltinFn::Cos) {
        if let Some((offset, rhs_arg)) = mul_by_unary_builtin_arg(ctx, *right, BuiltinFn::Sin) {
            if crate::expr_domain::exprs_equivalent(ctx, arg, rhs_arg) {
                return Some((
                    BuiltinFn::Cot,
                    arg,
                    offset,
                    TrigLogPoleOrientation::RatioMinusOffset,
                ));
            }
        }
    }

    None
}

fn mul_by_unary_builtin_arg(
    ctx: &Context,
    expr: ExprId,
    builtin: BuiltinFn,
) -> Option<(ExprId, ExprId)> {
    let expr = cas_ast::hold::unwrap_hold(ctx, expr);
    let Expr::Mul(left, right) = ctx.get(expr) else {
        return None;
    };
    if let Some(arg) = unary_builtin_arg(ctx, *right, builtin) {
        return Some((*left, arg));
    }
    unary_builtin_arg(ctx, *left, builtin).map(|arg| (*right, arg))
}

fn source_reciprocal_trig_log_pole_matches(
    ctx: &Context,
    source_pole: ExprId,
    builtin: BuiltinFn,
    arg: ExprId,
    offset: ExprId,
    orientation: TrigLogPoleOrientation,
) -> bool {
    let source_pole = cas_ast::hold::unwrap_hold(ctx, source_pole);
    let Expr::Sub(left, right) = ctx.get(source_pole) else {
        return false;
    };

    match orientation {
        TrigLogPoleOrientation::RatioMinusOffset => unary_builtin_arg(ctx, *left, builtin)
            .is_some_and(|source_arg| {
                crate::expr_domain::exprs_equivalent(ctx, source_arg, arg)
                    && crate::expr_domain::exprs_equivalent(ctx, *right, offset)
            }),
        TrigLogPoleOrientation::OffsetMinusRatio => unary_builtin_arg(ctx, *right, builtin)
            .is_some_and(|source_arg| {
                crate::expr_domain::exprs_equivalent(ctx, source_arg, arg)
                    && crate::expr_domain::exprs_equivalent(ctx, *left, offset)
            }),
    }
}

fn unary_builtin_arg(ctx: &Context, expr: ExprId, expected_builtin: BuiltinFn) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_hold(ctx, expr);
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    (args.len() == 1 && ctx.builtin_of(*fn_id) == Some(expected_builtin)).then_some(args[0])
}

#[cfg(test)]
mod tests {
    use cas_ast::{Context, ExprId};
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    use super::{
        integrate_reciprocal_trig_log_source_condition_aliases,
        reciprocal_trig_log_factored_pole_matches_source,
    };

    fn parse_expr(ctx: &mut Context, input: &str) -> ExprId {
        parse(input, ctx).unwrap()
    }

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn source_condition_aliases_cover_tan_and_cot_orientations() {
        for (input, source_display, factored_display) in [
            (
                "integrate(1/((tan(x)-a)*ln(tan(x))), x)",
                "tan(x) - a",
                "sin(x) - a * cos(x)",
            ),
            (
                "integrate(1/((a-tan(x))*ln(tan(x))), x)",
                "a - tan(x)",
                "a * cos(x) - sin(x)",
            ),
            (
                "integrate(1/((cot(x)-a)*ln(cot(x))), x)",
                "cot(x) - a",
                "cos(x) - a * sin(x)",
            ),
            (
                "integrate(1/((a-cot(x))*ln(cot(x))), x)",
                "a - cot(x)",
                "a * sin(x) - cos(x)",
            ),
        ] {
            let mut ctx = Context::new();
            let resolved = parse_expr(&mut ctx, input);
            let aliases =
                integrate_reciprocal_trig_log_source_condition_aliases(&mut ctx, resolved);

            assert_eq!(aliases.len(), 1, "{input}");
            let alias = aliases[0];
            assert_eq!(rendered(&ctx, alias.source_pole), source_display);
            assert_eq!(rendered(&ctx, alias.factored_pole), factored_display);
            assert!(reciprocal_trig_log_factored_pole_matches_source(
                &ctx,
                alias.factored_pole,
                alias.source_pole
            ));
        }
    }
}
