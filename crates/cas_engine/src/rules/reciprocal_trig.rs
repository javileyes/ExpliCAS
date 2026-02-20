use crate::define_rule;
use crate::rule::Rewrite;
use cas_ast::Expr;
use cas_math::trig_reciprocal_eval_support::{
    eval_reciprocal_trig_value, is_reciprocal_trig_composition,
    rewrite_negative_reciprocal_trig_argument,
};

define_rule!(
    EvaluateReciprocalTrigRule,
    "Evaluate Reciprocal Trig Functions",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    |ctx, expr| {
        if let Expr::Function(fn_id, args) = ctx.get(expr) {
            let name = match ctx.builtin_of(*fn_id) {
                Some(b) => b.name(),
                None => return None,
            };
            if args.len() != 1 {
                return None;
            }
            let arg = args[0];
            if let Some((new_expr, desc)) = eval_reciprocal_trig_value(ctx, name, arg) {
                return Some(Rewrite::new(new_expr).desc(desc));
            }
        }
        None
    }
);

define_rule!(
    ReciprocalTrigCompositionRule,
    "Reciprocal Trig Composition",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    |ctx, expr| {
        if let Expr::Function(outer_fn_id, outer_args) = ctx.get(expr) {
            if outer_args.len() != 1 {
                return None;
            }
            let inner_expr = outer_args[0];
            if let Expr::Function(inner_fn_id, inner_args) = ctx.get(inner_expr) {
                if inner_args.len() != 1 {
                    return None;
                }
                let x = inner_args[0];
                let outer_name = match ctx.builtin_of(*outer_fn_id) {
                    Some(b) => b.name(),
                    None => return None,
                };
                let inner_name = match ctx.builtin_of(*inner_fn_id) {
                    Some(b) => b.name(),
                    None => return None,
                };
                if is_reciprocal_trig_composition(outer_name, inner_name) {
                    return Some(
                        Rewrite::new(x)
                            .desc_lazy(|| format!("{}({}(x)) = x", outer_name, inner_name)),
                    );
                }
            }
        }
        None
    }
);

define_rule!(
    ReciprocalTrigNegativeRule,
    "Reciprocal Trig Negative Argument",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    |ctx, expr| {
        if let Expr::Function(fn_id, args) = ctx.get(expr) {
            let name = match ctx.builtin_of(*fn_id) {
                Some(b) => b.name(),
                None => return None,
            };
            if args.len() != 1 {
                return None;
            }
            let arg = args[0];
            if let Some((new_expr, desc)) =
                rewrite_negative_reciprocal_trig_argument(ctx, name, arg)
            {
                return Some(Rewrite::new(new_expr).desc(desc));
            }
        }
        None
    }
);

// ==================== Registration ====================

pub fn register(simplifier: &mut crate::engine::Simplifier) {
    simplifier.add_rule(Box::new(EvaluateReciprocalTrigRule));
    simplifier.add_rule(Box::new(ReciprocalTrigCompositionRule));
    simplifier.add_rule(Box::new(ReciprocalTrigNegativeRule));
}
