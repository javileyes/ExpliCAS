//! Data-driven trigonometric evaluation rule.
//!
//! This replaces the verbose ~360-line `EvaluateTrigRule` with a compact
//! table-lookup approach.

use crate::define_rule;
use crate::rule::Rewrite;
use cas_ast::{Expr, ExprId};

use super::values::{
    detect_inverse_trig_input, detect_special_angle, lookup_inverse_trig_value, lookup_trig_value,
};

define_rule!(
    EvaluateTrigTableRule,
    "Evaluate Trigonometric Functions (Table)",
    |ctx, expr| {
        let expr_data = ctx.get(expr).clone();
        if let Expr::Function(name, args) = expr_data {
            if args.len() == 1 {
                let arg = args[0];

                // Check if argument is a special angle (for sin, cos, tan)
                if let Some(angle) = detect_special_angle(ctx, arg) {
                    // Look up the value in our static table
                    if let Some(trig_value) = lookup_trig_value(&name, angle) {
                        let new_expr = trig_value.to_expr(ctx);
                        return Some(Rewrite {
                            new_expr,
                            description: format!(
                                "{}({}) = {}",
                                name,
                                angle.display(),
                                trig_value.display()
                            ),
                            before_local: None,
                            after_local: None,
                            domain_assumption: None,
                            assumption_events: Default::default(),
                        });
                    }
                }

                // Check for inverse trig functions at special inputs (0, 1, 1/2)
                if let Some(input) = detect_inverse_trig_input(ctx, arg) {
                    if let Some(trig_value) = lookup_inverse_trig_value(&name, input) {
                        let new_expr = trig_value.to_expr(ctx);
                        return Some(Rewrite {
                            new_expr,
                            description: format!(
                                "{}({}) = {}",
                                name,
                                input.display(),
                                trig_value.display()
                            ),
                            before_local: None,
                            after_local: None,
                            domain_assumption: None,
                            assumption_events: Default::default(),
                        });
                    }
                }

                // Handle negative arguments: sin(-x) = -sin(x), cos(-x) = cos(x), tan(-x) = -tan(x)
                let inner_opt = extract_negated_inner(ctx, arg);
                if let Some(inner) = inner_opt {
                    match name.as_str() {
                        "sin" => {
                            let sin_inner = ctx.add(Expr::Function("sin".to_string(), vec![inner]));
                            let new_expr = ctx.add(Expr::Neg(sin_inner));
                            return Some(Rewrite {
                                new_expr,
                                description: "sin(-x) = -sin(x)".to_string(),
                                before_local: None,
                                after_local: None,
                                domain_assumption: None,
                                assumption_events: Default::default(),
                            });
                        }
                        "cos" => {
                            let new_expr = ctx.add(Expr::Function("cos".to_string(), vec![inner]));
                            return Some(Rewrite {
                                new_expr,
                                description: "cos(-x) = cos(x)".to_string(),
                                before_local: None,
                                after_local: None,
                                domain_assumption: None,
                                assumption_events: Default::default(),
                            });
                        }
                        "tan" => {
                            let tan_inner = ctx.add(Expr::Function("tan".to_string(), vec![inner]));
                            let new_expr = ctx.add(Expr::Neg(tan_inner));
                            return Some(Rewrite {
                                new_expr,
                                description: "tan(-x) = -tan(x)".to_string(),
                                before_local: None,
                                after_local: None,
                                domain_assumption: None,
                                assumption_events: Default::default(),
                            });
                        }
                        _ => {}
                    }
                }
            }
        }
        None
    }
);

/// Extract the inner expression from Neg(x) or Mul(-1, x)
fn extract_negated_inner(ctx: &cas_ast::Context, arg: ExprId) -> Option<ExprId> {
    match ctx.get(arg) {
        Expr::Neg(inner) => Some(*inner),
        Expr::Mul(l, r) => {
            if let Expr::Number(n) = ctx.get(*l) {
                if *n == num_rational::BigRational::from_integer((-1).into()) {
                    return Some(*r);
                }
            }
            if let Expr::Number(n) = ctx.get(*r) {
                if *n == num_rational::BigRational::from_integer((-1).into()) {
                    return Some(*l);
                }
            }
            None
        }
        _ => None,
    }
}

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
        let sin_zero = ctx.add(Expr::Function("sin".to_string(), vec![zero]));

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
        let cos_pi = ctx.add(Expr::Function("cos".to_string(), vec![pi]));

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
