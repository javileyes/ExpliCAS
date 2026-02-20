//! Limit rules for V1: x → ±∞.
//!
//! Rules are applied in order:
//! 1. ConstantRule - expression independent of variable
//! 2. VariableRule - expression is the variable itself  
//! 3. PowerRule - x^n with integer n
//! 4. RationalPolyRule - P(x)/Q(x) polynomial division

use cas_ast::{Context, ExprId};
use cas_math::infinity_support::InfSign;
use cas_math::limits_support;

use crate::Budget;

use super::types::Approach;

fn approach_sign(approach: Approach) -> InfSign {
    match approach {
        Approach::PosInfinity => InfSign::Pos,
        Approach::NegInfinity => InfSign::Neg,
    }
}

/// Try all limit rules in order.
///
/// Returns Some(result) if a rule applies, None if no rule applies.
pub fn try_limit_rules(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: Approach,
    _budget: &mut Budget,
) -> Option<ExprId> {
    limits_support::try_limit_rules_at_infinity(ctx, expr, var, approach_sign(approach))
}
