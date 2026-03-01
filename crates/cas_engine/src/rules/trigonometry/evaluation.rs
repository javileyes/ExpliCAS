//! Data-driven trigonometric evaluation rule.
//!
//! This replaces the verbose ~360-line `EvaluateTrigRule` with a compact
//! table-lookup approach.

use crate::define_rule;
use crate::rule::Rewrite;
use cas_math::trig_eval_table_support::try_rewrite_trig_eval_table_expr;

define_rule!(
    EvaluateTrigTableRule,
    "Evaluate Trigonometric Functions (Table)",
    |ctx, expr| {
        let rewrite = try_rewrite_trig_eval_table_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parent_context::ParentContext;
    use crate::rule::Rule;
    use cas_ast::{Context, Expr};
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
