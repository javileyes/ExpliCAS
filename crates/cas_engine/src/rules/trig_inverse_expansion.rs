use crate::define_rule;
use crate::rule::Rewrite;
use cas_ast::Expr;
use cas_math::trig_inverse_expansion_support::expand_trig_inverse_composition;

// ========== Unified Rule ==========

define_rule!(
    TrigInverseExpansionRule,
    "Trig of Inverse Trig Expansion",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    solve_safety: crate::solve_safety::SolveSafety::NeedsCondition(
        crate::assumptions::ConditionClass::Analytic
    ),
    |ctx, expr, _parent_ctx| {
        if let Expr::Function(outer_name, outer_args) = ctx.get(expr) {
            if outer_args.len() != 1 {
                return None;
            }
            let inner = outer_args[0];

            if let Expr::Function(inner_name, inner_args) = ctx.get(inner) {
                if inner_args.len() != 1 {
                    return None;
                }
                let x = inner_args[0];

                // Look up in expansion table
                let outer = match ctx.builtin_of(*outer_name) {
                    Some(b) => b.name(),
                    None => return None,
                };
                let inner = match ctx.builtin_of(*inner_name) {
                    Some(b) => b.name(),
                    None => return None,
                };
                if let Some((result, description)) =
                    expand_trig_inverse_composition(ctx, outer, inner, x)
                {
                    return Some(Rewrite::new(result).desc(description));
                }
            }
        }
        None
    }
);

// ========== Registration ==========

pub fn register(simplifier: &mut crate::engine::Simplifier) {
    simplifier.add_rule(Box::new(TrigInverseExpansionRule));
}
