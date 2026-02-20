//! Data-driven trigonometric evaluation rule.
//!
//! This replaces the verbose ~360-line `EvaluateTrigRule` with a compact
//! table-lookup approach.

use crate::define_rule;
use crate::rule::Rewrite;
use cas_ast::Expr;
use cas_math::trig_eval_table_support::{lookup_trig_or_inverse, rewrite_negative_trig_argument};

define_rule!(
    EvaluateTrigTableRule,
    "Evaluate Trigonometric Functions (Table)",
    |ctx, expr| {
        let (fn_id, args) = if let Expr::Function(fn_id, args) = ctx.get(expr) {
            (*fn_id, args.clone())
        } else {
            return None;
        };
        {
            let builtin = ctx.builtin_of(fn_id);
            // Only process known trig/inverse trig functions
            let name = match builtin {
                Some(b) => b.name().to_string(),
                None => return None,
            };
            if args.len() == 1 {
                let arg = args[0];

                if let Some(hit) = lookup_trig_or_inverse(ctx, &name, arg) {
                    let new_expr = hit.value.to_expr(ctx);
                    let key_display = hit.key_display;
                    let value_display = hit.value.display();
                    return Some(
                        Rewrite::new(new_expr)
                            .desc_lazy(|| format!("{}({}) = {}", name, key_display, value_display)),
                    );
                }

                if let Some((new_expr, desc)) = rewrite_negative_trig_argument(ctx, &name, arg) {
                    return Some(Rewrite::new(new_expr).desc(desc));
                }
            }
        }
        None
    }
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parent_context::ParentContext;
    use crate::rule::Rule;
    use cas_ast::Context;
    use num_traits::Zero;

    #[test]
    fn test_sin_zero() {
        let mut ctx = Context::new();
        let zero = ctx.num(0);
        let sin_zero = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![zero]);

        let rule = EvaluateTrigTableRule;
        let parent_ctx = ParentContext::root();
        let result = rule.apply(&mut ctx, sin_zero, &parent_ctx);

        assert!(result.is_some());
        let rewrite = result.unwrap();
        if let Expr::Number(n) = ctx.get(rewrite.new_expr) {
            assert!(n.is_zero());
        } else {
            panic!("Expected Number(0)");
        }
    }

    #[test]
    fn test_cos_pi() {
        let mut ctx = Context::new();
        let pi = ctx.add(Expr::Constant(cas_ast::Constant::Pi));
        let cos_pi = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![pi]);

        let rule = EvaluateTrigTableRule;
        let parent_ctx = ParentContext::root();
        let result = rule.apply(&mut ctx, cos_pi, &parent_ctx);

        assert!(result.is_some());
        let rewrite = result.unwrap();
        if let Expr::Number(n) = ctx.get(rewrite.new_expr) {
            assert_eq!(*n, num_rational::BigRational::from_integer((-1).into()));
        } else {
            panic!("Expected Number(-1)");
        }
    }
}
