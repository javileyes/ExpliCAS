//! Infinity arithmetic rules for the extended real line ℝ ∪ {+∞, −∞}.
//!
//! Implements safe, conservative rules for infinity operations.
//! Only collapses when all non-infinity terms are "finite literal" (numbers, known constants).
//!
//! # Covered operations
//! - `finite + ∞ → ∞` (absorption)
//! - `finite / ∞ → 0`
//! - `∞ + (-∞) → Undefined` (indeterminate)
//! - `0 · ∞ → Undefined` (indeterminate)
//! - `undefined` propagates through `+`, `-`, unary `-`, `*`, `/`, and `^`

use crate::rule::Rewrite;
use cas_ast::{Constant, Context, Expr, ExprId};
use cas_math::infinity_support::{
    mk_undefined, try_rewrite_add_infinity_absorption_expr, try_rewrite_div_by_infinity_expr,
    try_rewrite_inf_div_finite_expr, try_rewrite_inf_div_inf_expr,
    try_rewrite_mul_finite_infinity_expr, try_rewrite_mul_zero_infinity_expr,
};

// ============================================================
// RULES
// ============================================================

/// Rule: Infinity absorption in addition.
///
/// - `finite + ∞ → ∞`
/// - `finite + (-∞) → -∞`
/// - `∞ + (-∞) → Undefined` (indeterminate)
///
/// Only applies when ALL non-infinity terms are finite literals (conservative).
pub fn add_infinity_absorption(ctx: &mut Context, expr: ExprId) -> Option<Rewrite> {
    let plan = try_rewrite_add_infinity_absorption_expr(ctx, expr)?;
    Some(Rewrite::new(plan.rewritten).desc(plan.description))
}

/// Rule: Division by infinity.
///
/// `finite / ∞ → 0`
/// `finite / (-∞) → 0`
///
/// Only applies when numerator is a finite literal.
pub fn div_by_infinity(ctx: &mut Context, expr: ExprId) -> Option<Rewrite> {
    let plan = try_rewrite_div_by_infinity_expr(ctx, expr)?;
    Some(Rewrite::new(plan.rewritten).desc(plan.description))
}

/// Rule: Zero times infinity is indeterminate.
///
/// `0 · ∞ → Undefined`
/// `∞ · 0 → Undefined`
pub fn mul_zero_infinity(ctx: &mut Context, expr: ExprId) -> Option<Rewrite> {
    let plan = try_rewrite_mul_zero_infinity_expr(ctx, expr)?;
    Some(Rewrite::new(plan.rewritten).desc(plan.description))
}

/// Rule: Finite (non-zero) times infinity.
///
/// `finite * ∞ → ±∞` (sign depends on finite's sign)
/// - `3 * infinity → infinity`
/// - `(-2) * infinity → -infinity`
/// - `x * infinity → no simplification` (conservative)
pub fn mul_finite_infinity(ctx: &mut Context, expr: ExprId) -> Option<Rewrite> {
    let plan = try_rewrite_mul_finite_infinity_expr(ctx, expr)?;
    Some(Rewrite::new(plan.rewritten).desc(plan.description))
}

/// Rule: Infinity divided by finite (non-zero).
///
/// `∞ / finite → ±∞` (sign depends on finite's sign)
/// - `infinity / 2 → infinity`
/// - `infinity / (-3) → -infinity`
/// - `-infinity / 2 → -infinity`
pub fn inf_div_finite(ctx: &mut Context, expr: ExprId) -> Option<Rewrite> {
    let plan = try_rewrite_inf_div_finite_expr(ctx, expr)?;
    Some(Rewrite::new(plan.rewritten).desc(plan.description))
}

/// Rule: Infinity divided by infinity is indeterminate.
///
/// `∞ / ∞ → Undefined` (including finite-scaled forms `(2·∞)/(5·∞)`, `(-∞)/∞`).
pub fn inf_div_inf(ctx: &mut Context, expr: ExprId) -> Option<Rewrite> {
    let plan = try_rewrite_inf_div_inf_expr(ctx, expr)?;
    Some(Rewrite::new(plan.rewritten).desc(plan.description))
}

fn contains_undefined(ctx: &Context, expr: ExprId) -> bool {
    let mut stack = vec![expr];
    while let Some(current) = stack.pop() {
        match ctx.get(current) {
            Expr::Constant(Constant::Undefined) => return true,
            Expr::Add(left, right)
            | Expr::Sub(left, right)
            | Expr::Mul(left, right)
            | Expr::Div(left, right)
            | Expr::Pow(left, right) => {
                stack.push(*left);
                stack.push(*right);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(*inner),
            Expr::Function(_, args) => stack.extend(args.iter().copied()),
            Expr::Matrix { data, .. } => stack.extend(data.iter().copied()),
            Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => {}
        }
    }
    false
}

/// Rule: Any addition containing `undefined` is `undefined`.
pub fn add_undefined(ctx: &mut Context, expr: ExprId) -> Option<Rewrite> {
    let Expr::Add(left, right) = ctx.get(expr) else {
        return None;
    };
    if !contains_undefined(ctx, *left) && !contains_undefined(ctx, *right) {
        return None;
    }
    Some(Rewrite::new(mk_undefined(ctx)).desc("addition with undefined is undefined"))
}

/// Rule: Any subtraction containing `undefined` is `undefined`.
pub fn sub_undefined(ctx: &mut Context, expr: ExprId) -> Option<Rewrite> {
    let Expr::Sub(left, right) = ctx.get(expr) else {
        return None;
    };
    if !contains_undefined(ctx, *left) && !contains_undefined(ctx, *right) {
        return None;
    }
    Some(Rewrite::new(mk_undefined(ctx)).desc("subtraction with undefined is undefined"))
}

/// Rule: Any product containing `undefined` is `undefined`.
pub fn mul_undefined(ctx: &mut Context, expr: ExprId) -> Option<Rewrite> {
    let Expr::Mul(_, _) = ctx.get(expr) else {
        return None;
    };
    if !contains_undefined(ctx, expr) {
        return None;
    }
    Some(Rewrite::new(mk_undefined(ctx)).desc("undefined factor makes product undefined"))
}

/// Rule: Negating `undefined` keeps it `undefined`.
pub fn neg_undefined(ctx: &mut Context, expr: ExprId) -> Option<Rewrite> {
    let Expr::Neg(inner) = ctx.get(expr) else {
        return None;
    };
    if !contains_undefined(ctx, *inner) {
        return None;
    }
    Some(Rewrite::new(mk_undefined(ctx)).desc("negation of undefined is undefined"))
}

/// Rule: Any division containing `undefined` is `undefined`.
pub fn div_undefined(ctx: &mut Context, expr: ExprId) -> Option<Rewrite> {
    let Expr::Div(num, den) = ctx.get(expr) else {
        return None;
    };
    if !contains_undefined(ctx, *num) && !contains_undefined(ctx, *den) {
        return None;
    }
    Some(Rewrite::new(mk_undefined(ctx)).desc("division with undefined is undefined"))
}

/// Rule: Any power containing `undefined` is `undefined`.
pub fn pow_undefined(ctx: &mut Context, expr: ExprId) -> Option<Rewrite> {
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return None;
    };
    if !contains_undefined(ctx, *base) && !contains_undefined(ctx, *exp) {
        return None;
    }
    Some(Rewrite::new(mk_undefined(ctx)).desc("power with undefined is undefined"))
}

/// Rule: Any function call containing `undefined` is `undefined`.
pub fn function_undefined(ctx: &mut Context, expr: ExprId) -> Option<Rewrite> {
    let Expr::Function(_, args) = ctx.get(expr) else {
        return None;
    };
    if !args.iter().copied().any(|arg| contains_undefined(ctx, arg)) {
        return None;
    }
    Some(Rewrite::new(mk_undefined(ctx)).desc("function with undefined argument is undefined"))
}

// ============================================================
// RULE STRUCTS (for pipeline registration)
// ============================================================

use crate::define_rule;

define_rule!(
    AddUndefinedRule,
    "Undefined Plus Anything",
    Some(crate::target_kind::TargetKindSet::ADD),
    |ctx, expr| { add_undefined(ctx, expr) }
);

define_rule!(
    SubUndefinedRule,
    "Undefined Minus Anything",
    Some(crate::target_kind::TargetKindSet::SUB),
    |ctx, expr| { sub_undefined(ctx, expr) }
);

define_rule!(
    AddInfinityRule,
    "Infinity Absorption in Addition",
    Some(crate::target_kind::TargetKindSet::ADD),
    |ctx, expr| { add_infinity_absorption(ctx, expr) }
);

define_rule!(
    DivByInfinityRule,
    "Division by Infinity",
    Some(crate::target_kind::TargetKindSet::DIV),
    |ctx, expr| { div_by_infinity(ctx, expr) }
);

define_rule!(
    MulUndefinedRule,
    "Undefined Times Anything",
    Some(crate::target_kind::TargetKindSet::MUL),
    |ctx, expr| { mul_undefined(ctx, expr) }
);

define_rule!(
    NegUndefinedRule,
    "Negation of Undefined",
    Some(crate::target_kind::TargetKindSet::NEG),
    |ctx, expr| { neg_undefined(ctx, expr) }
);

define_rule!(
    MulZeroInfinityRule,
    "Zero Times Infinity Indeterminate",
    Some(crate::target_kind::TargetKindSet::MUL),
    |ctx, expr| { mul_zero_infinity(ctx, expr) }
);

define_rule!(
    MulInfinityRule,
    "Finite Times Infinity",
    Some(crate::target_kind::TargetKindSet::MUL),
    |ctx, expr| { mul_finite_infinity(ctx, expr) }
);

define_rule!(
    DivUndefinedRule,
    "Undefined Divided by Anything",
    Some(crate::target_kind::TargetKindSet::DIV),
    |ctx, expr| { div_undefined(ctx, expr) }
);

define_rule!(
    PowUndefinedRule,
    "Undefined Raised to Anything",
    Some(crate::target_kind::TargetKindSet::POW),
    |ctx, expr| { pow_undefined(ctx, expr) }
);

define_rule!(
    FunctionUndefinedRule,
    "Function of Undefined",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    |ctx, expr| { function_undefined(ctx, expr) }
);

define_rule!(
    InfDivFiniteRule,
    "Infinity Divided by Finite",
    Some(crate::target_kind::TargetKindSet::DIV),
    |ctx, expr| { inf_div_finite(ctx, expr) }
);

define_rule!(
    InfDivInfRule,
    "Infinity Divided by Infinity Indeterminate",
    Some(crate::target_kind::TargetKindSet::DIV),
    |ctx, expr| { inf_div_inf(ctx, expr) }
);

/// Register infinity arithmetic rules with the simplifier.
///
/// These rules should be registered early in the pipeline (with CORE rules)
/// to handle infinity operations before other simplifications.
pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(AddUndefinedRule));
    simplifier.add_rule(Box::new(SubUndefinedRule));
    simplifier.add_rule(Box::new(NegUndefinedRule));
    simplifier.add_rule(Box::new(MulUndefinedRule));
    // Indeterminate forms first (highest priority)
    simplifier.add_rule(Box::new(MulZeroInfinityRule));
    simplifier.add_rule(Box::new(InfDivInfRule));
    // Then absorption/computation rules
    simplifier.add_rule(Box::new(MulInfinityRule));
    simplifier.add_rule(Box::new(AddInfinityRule));
    simplifier.add_rule(Box::new(DivUndefinedRule));
    simplifier.add_rule(Box::new(PowUndefinedRule));
    simplifier.add_rule(Box::new(FunctionUndefinedRule));
    simplifier.add_rule(Box::new(DivByInfinityRule));
    simplifier.add_rule(Box::new(InfDivFiniteRule));
}
