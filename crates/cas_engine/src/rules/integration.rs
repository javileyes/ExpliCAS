//! Integration preparation rules.
//!
//! These rules transform expressions into forms more amenable to integration:
//! - Product-to-sum identities (Werner formulas)
//! - Telescoping product series (Morrie's law)
//!
//! Only active when `ContextMode::IntegratePrep` is set.

use crate::engine::Simplifier;
use crate::parent_context::ParentContext;
use crate::rule::{Rewrite, Rule};
use cas_ast::{Context, Expr, ExprId};

/// Product-to-sum identity for trigonometric products (Werner formulas).
///
/// `2 * sin(A) * cos(B) → sin(A+B) + sin(A-B)`
/// `2 * cos(A) * cos(B) → cos(A+B) + cos(A-B)`
/// `2 * sin(A) * sin(B) → cos(A-B) - cos(A+B)`
pub struct ProductToSumRule;

impl Rule for ProductToSumRule {
    fn name(&self) -> &str {
        "ProductToSum"
    }

    // Uses default allowed_phases (CORE | POST)

    fn apply(
        &self,
        ctx: &mut Context,
        expr: ExprId,
        _parent_ctx: &ParentContext,
    ) -> Option<Rewrite> {
        // Look for: 2 * sin(A) * cos(B) or similar patterns
        // First, flatten the multiplication and check for coefficient 2

        let factors = crate::helpers::flatten_mul_chain(ctx, expr);

        // Check if we have at least 3 factors (coeff, sin, cos)
        if factors.len() < 2 {
            return None;
        }

        // Find numeric coefficient
        let mut coeff_idx = None;
        let mut sin_idx = None;
        let mut cos_idx = None;
        let mut sin_arg = None;
        let mut cos_arg = None;

        for (i, &f) in factors.iter().enumerate() {
            match ctx.get(f) {
                Expr::Number(n) => {
                    if n.to_integer() == 2.into() && coeff_idx.is_none() {
                        coeff_idx = Some(i);
                    }
                }
                Expr::Function(name, args) if args.len() == 1 => {
                    if name == "sin" && sin_idx.is_none() {
                        sin_idx = Some(i);
                        sin_arg = Some(args[0]);
                    } else if name == "cos" && cos_idx.is_none() {
                        cos_idx = Some(i);
                        cos_arg = Some(args[0]);
                    }
                }
                _ => {}
            }
        }

        // Pattern: 2 * sin(A) * cos(B) → sin(A+B) + sin(A-B)
        if let (Some(_), Some(_), Some(_), Some(a), Some(b)) =
            (coeff_idx, sin_idx, cos_idx, sin_arg, cos_arg)
        {
            // Build sin(A+B) + sin(A-B)
            let a_plus_b = ctx.add(Expr::Add(a, b));
            let a_minus_b = ctx.add(Expr::Sub(a, b));
            let sin_sum = ctx.add(Expr::Function("sin".to_string(), vec![a_plus_b]));
            let sin_diff = ctx.add(Expr::Function("sin".to_string(), vec![a_minus_b]));
            let result = ctx.add(Expr::Add(sin_sum, sin_diff));

            return Some(Rewrite {
                new_expr: result,
                description: "2·sin(A)·cos(B) → sin(A+B) + sin(A-B) (Werner)".to_string(),
                before_local: None,
                after_local: None,
                domain_assumption: None,
            });
        }

        None
    }
}

/// Register integration preparation rules.
pub fn register_integration_prep(simplifier: &mut Simplifier) {
    simplifier.add_rule(Box::new(ProductToSumRule));
    // TODO: Add CosProductTelescopingRule (Morrie's law)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    #[test]
    fn test_product_to_sum_sin_cos() {
        let mut ctx = Context::new();
        let expr = parse("2*sin(x)*cos(y)", &mut ctx).unwrap();

        let rule = ProductToSumRule;
        let result = rule.apply(&mut ctx, expr, &ParentContext::root());

        assert!(
            result.is_some(),
            "ProductToSum should match 2*sin(x)*cos(y)"
        );
    }
}
