use crate::infinity_support::{mk_infinity, InfSign};
use crate::limit_types::{Approach, LimitEvalOutcome, LimitOptions, PreSimplifyMode};
use crate::perfect_square_support::rational_sqrt;
use crate::polynomial::Polynomial;
use crate::root_forms::{extract_square_root_base, rational_cbrt_exact};
use cas_ast::{BuiltinFn, Constant, Context, Expr, ExprId};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

/// Check if an expression depends on a specific variable id.
///
/// Uses iterative traversal to avoid recursion limits on deep trees.
pub fn depends_on(ctx: &Context, expr: ExprId, var: ExprId) -> bool {
    let mut stack = vec![expr];

    while let Some(current) = stack.pop() {
        if current == var {
            return true;
        }

        match ctx.get(current) {
            Expr::Add(l, r)
            | Expr::Sub(l, r)
            | Expr::Mul(l, r)
            | Expr::Div(l, r)
            | Expr::Pow(l, r) => {
                stack.push(*l);
                stack.push(*r);
            }
            Expr::Neg(inner) => stack.push(*inner),
            Expr::Hold(inner) => stack.push(*inner),
            Expr::Function(_, args) => {
                for arg in args {
                    stack.push(*arg);
                }
            }
            Expr::Variable(_) | Expr::Number(_) | Expr::Constant(_) => {}
            Expr::Matrix { .. } | Expr::SessionRef(_) => {}
        }
    }

    false
}

/// Parse a power expression with integer exponent.
///
/// Returns `(base, n)` if `expr` is `base^n` where `n` is an integer literal.
pub fn parse_pow_int(ctx: &Context, expr: ExprId) -> Option<(ExprId, i64)> {
    match ctx.get(expr) {
        Expr::Pow(base, exp) => {
            let n = crate::expr_extract::extract_i64_integer(ctx, *exp)?;
            Some((*base, n))
        }
        _ => None,
    }
}

/// Create a residual limit expression: `limit(expr, var, approach_symbol)`.
pub fn mk_limit(ctx: &mut Context, expr: ExprId, var: ExprId, approach: InfSign) -> ExprId {
    let approach_sym = match approach {
        InfSign::Pos => ctx.add(Expr::Constant(Constant::Infinity)),
        InfSign::Neg => {
            let inf = ctx.add(Expr::Constant(Constant::Infinity));
            ctx.add(Expr::Neg(inf))
        }
    };
    ctx.call("limit", vec![expr, var, approach_sym])
}

/// Create a residual limit expression from a typed approach.
pub fn mk_limit_for_approach(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: Approach,
) -> ExprId {
    match approach {
        Approach::PosInfinity => mk_limit(ctx, expr, var, InfSign::Pos),
        Approach::NegInfinity => mk_limit(ctx, expr, var, InfSign::Neg),
        Approach::Finite(point) => ctx.call("limit", vec![expr, var, point]),
    }
}

/// Determine resulting infinity sign from approach sign and exponent parity.
pub fn limit_sign(approach: InfSign, power: i64) -> InfSign {
    match approach {
        InfSign::Pos => InfSign::Pos,
        InfSign::Neg => {
            if power % 2 == 0 {
                InfSign::Pos // (-∞)^even = +∞
            } else {
                InfSign::Neg // (-∞)^odd = -∞
            }
        }
    }
}

/// Create infinity with appropriate sign.
pub fn mk_inf(ctx: &mut Context, sign: InfSign) -> ExprId {
    mk_infinity(ctx, sign)
}

/// Rule 1: Constant - lim c = c (if `expr` doesn't depend on `var`).
pub fn apply_constant_rule(ctx: &Context, expr: ExprId, var: ExprId) -> Option<ExprId> {
    if !depends_on(ctx, expr, var) {
        Some(expr)
    } else {
        None
    }
}

fn apply_finite_polynomial_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    if depends_on(ctx, point, var) {
        return None;
    }
    let Expr::Variable(var_symbol) = ctx.get(var) else {
        return None;
    };
    let var_name = ctx.sym_name(*var_symbol);
    let Expr::Number(point_value) = ctx.get(point) else {
        return None;
    };
    let point_value = point_value.clone();

    let poly = Polynomial::from_expr(ctx, expr, var_name).ok()?;
    let value = poly.eval(&point_value);
    Some(ctx.add(Expr::Number(value)))
}

fn apply_finite_rational_polynomial_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    if depends_on(ctx, point, var) {
        return None;
    }
    let Expr::Variable(var_symbol) = ctx.get(var) else {
        return None;
    };
    let var_name = ctx.sym_name(*var_symbol);
    let Expr::Number(point_value) = ctx.get(point) else {
        return None;
    };
    let point_value = point_value.clone();
    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };

    let numerator = Polynomial::from_expr(ctx, num, var_name).ok()?;
    let denominator = Polynomial::from_expr(ctx, den, var_name).ok()?;
    let denominator_value = denominator.eval(&point_value);
    if denominator_value.is_zero() {
        return None;
    }
    let value = numerator.eval(&point_value) / denominator_value;
    Some(ctx.add(Expr::Number(value)))
}

fn is_finite_total_real_unary_builtin(builtin: BuiltinFn) -> bool {
    matches!(
        builtin,
        BuiltinFn::Exp
            | BuiltinFn::Sin
            | BuiltinFn::Cos
            | BuiltinFn::Sinh
            | BuiltinFn::Cosh
            | BuiltinFn::Tanh
            | BuiltinFn::Atan
            | BuiltinFn::Arctan
            | BuiltinFn::Asinh
            | BuiltinFn::Cbrt
            | BuiltinFn::Abs
    )
}

fn is_finite_positive_domain_unary_builtin(builtin: BuiltinFn) -> bool {
    matches!(
        builtin,
        BuiltinFn::Ln | BuiltinFn::Log2 | BuiltinFn::Log10 | BuiltinFn::Sqrt
    )
}

fn finite_total_real_unary_result(
    ctx: &mut Context,
    builtin: BuiltinFn,
    argument_limit: ExprId,
) -> ExprId {
    if let Some(argument_value) = numeric_limit_value(ctx, argument_limit) {
        if matches!(builtin, BuiltinFn::Cbrt) {
            if let Some(root) = rational_cbrt_exact(&argument_value) {
                return ctx.add(Expr::Number(root));
            }
            let value_expr = ctx.add(Expr::Number(argument_value));
            return ctx.call_builtin(BuiltinFn::Cbrt, vec![value_expr]);
        }
        if matches!(builtin, BuiltinFn::Abs) {
            return ctx.add(Expr::Number(argument_value.abs()));
        }
        if matches!(
            builtin,
            BuiltinFn::Sin
                | BuiltinFn::Sinh
                | BuiltinFn::Tanh
                | BuiltinFn::Atan
                | BuiltinFn::Arctan
                | BuiltinFn::Asinh
        ) && argument_value.is_zero()
        {
            return ctx.num(0);
        }
        if matches!(builtin, BuiltinFn::Exp | BuiltinFn::Cos | BuiltinFn::Cosh)
            && argument_value.is_zero()
        {
            return ctx.num(1);
        }

        let value_expr = ctx.add(Expr::Number(argument_value));
        return ctx.call_builtin(builtin, vec![value_expr]);
    }

    if let Some(exact_result) =
        finite_total_real_unary_exact_expr_result(ctx, builtin, argument_limit)
    {
        return exact_result;
    }

    ctx.call_builtin(builtin, vec![argument_limit])
}

fn finite_total_real_unary_exact_expr_result(
    ctx: &mut Context,
    builtin: BuiltinFn,
    argument_limit: ExprId,
) -> Option<ExprId> {
    match builtin {
        BuiltinFn::Abs => finite_abs_exact_expr_result(ctx, argument_limit),
        BuiltinFn::Exp => finite_exp_exact_expr_result(ctx, argument_limit),
        _ => None,
    }
}

fn finite_abs_exact_expr_result(ctx: &mut Context, argument_limit: ExprId) -> Option<ExprId> {
    if finite_expr_proven_positive(ctx, argument_limit) {
        return Some(argument_limit);
    }

    let Expr::Neg(inner) = ctx.get(argument_limit).clone() else {
        return None;
    };
    finite_expr_proven_positive(ctx, inner).then_some(inner)
}

fn finite_exp_exact_expr_result(ctx: &mut Context, argument_limit: ExprId) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(argument_limit).clone() else {
        return None;
    };
    if !ctx.is_builtin(fn_id, BuiltinFn::Ln) || args.len() != 1 {
        return None;
    }

    let inner = args[0];
    finite_expr_proven_positive(ctx, inner).then_some(inner)
}

fn finite_positive_domain_unary_result(
    ctx: &mut Context,
    builtin: BuiltinFn,
    argument_limit: ExprId,
) -> Option<ExprId> {
    if let Some(argument_value) = numeric_limit_value(ctx, argument_limit) {
        if !argument_value.is_positive() {
            return None;
        }
        if let Some(exact_result) =
            finite_positive_domain_unary_exact_numeric_result(ctx, builtin, &argument_value)
        {
            return Some(exact_result);
        }
        let value_expr = ctx.add(Expr::Number(argument_value));
        return Some(ctx.call_builtin(builtin, vec![value_expr]));
    }

    if let Some(exact_result) =
        finite_positive_domain_unary_exact_expr_result(ctx, builtin, argument_limit)
    {
        return Some(exact_result);
    }

    finite_expr_proven_positive(ctx, argument_limit)
        .then(|| ctx.call_builtin(builtin, vec![argument_limit]))
}

fn finite_positive_domain_unary_exact_numeric_result(
    ctx: &mut Context,
    builtin: BuiltinFn,
    argument_value: &BigRational,
) -> Option<ExprId> {
    match builtin {
        BuiltinFn::Sqrt => rational_sqrt(argument_value).map(|root| ctx.add(Expr::Number(root))),
        BuiltinFn::Ln | BuiltinFn::Log2 | BuiltinFn::Log10 if argument_value.is_one() => {
            Some(ctx.num(0))
        }
        _ => None,
    }
}

fn finite_positive_domain_unary_exact_expr_result(
    ctx: &mut Context,
    builtin: BuiltinFn,
    argument_limit: ExprId,
) -> Option<ExprId> {
    match builtin {
        BuiltinFn::Ln => finite_ln_exact_expr_result(ctx, argument_limit),
        _ => None,
    }
}

fn finite_ln_exact_expr_result(ctx: &mut Context, argument_limit: ExprId) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(argument_limit).clone() else {
        return None;
    };
    if ctx.is_builtin(fn_id, BuiltinFn::Exp) && args.len() == 1 {
        Some(args[0])
    } else {
        None
    }
}

fn finite_log_base_limit_is_valid(ctx: &Context, base_limit: ExprId) -> bool {
    let Some(base_value) = numeric_limit_value(ctx, base_limit) else {
        return false;
    };
    base_value.is_positive() && base_value != rational_one()
}

fn finite_log_result(
    ctx: &mut Context,
    base_limit: ExprId,
    argument_limit: ExprId,
) -> Option<ExprId> {
    if !finite_log_base_limit_is_valid(ctx, base_limit) {
        return None;
    }
    if let Some(argument_value) = numeric_limit_value(ctx, argument_limit) {
        if !argument_value.is_positive() {
            return None;
        }
        if let Some(exact_result) =
            finite_log_exact_numeric_result(ctx, base_limit, &argument_value)
        {
            return Some(exact_result);
        }
        let value_expr = ctx.add(Expr::Number(argument_value));
        return Some(ctx.call_builtin(BuiltinFn::Log, vec![base_limit, value_expr]));
    }

    finite_expr_proven_positive(ctx, argument_limit)
        .then(|| ctx.call_builtin(BuiltinFn::Log, vec![base_limit, argument_limit]))
}

fn finite_log_exact_numeric_result(
    ctx: &mut Context,
    base_limit: ExprId,
    argument_value: &BigRational,
) -> Option<ExprId> {
    if argument_value.is_one() {
        return Some(ctx.num(0));
    }

    let base_value = numeric_limit_value(ctx, base_limit)?;
    if base_value == *argument_value {
        return Some(ctx.num(1));
    }

    None
}

const FINITE_INTEGER_POWER_EXACT_FOLD_LIMIT: u64 = 32;

fn rational_pow_nonnegative(base: &BigRational, exponent: u64) -> BigRational {
    let mut result = BigRational::one();
    let mut factor = base.clone();
    let mut remaining = exponent;

    while remaining > 0 {
        if remaining % 2 == 1 {
            result *= factor.clone();
        }
        remaining /= 2;
        if remaining > 0 {
            factor = factor.clone() * factor;
        }
    }

    result
}

fn finite_sqrt_even_power_result(
    ctx: &mut Context,
    base_limit: ExprId,
    exponent: i64,
) -> Option<ExprId> {
    if exponent % 2 != 0 {
        return None;
    }

    let radicand = extract_square_root_base(ctx, base_limit)?;
    let radicand_value = numeric_limit_value(ctx, radicand)?;
    if !radicand_value.is_positive() {
        return None;
    }

    let half_exponent = exponent.unsigned_abs() / 2;
    if half_exponent > FINITE_INTEGER_POWER_EXACT_FOLD_LIMIT {
        return None;
    }

    let mut value = rational_pow_nonnegative(&radicand_value, half_exponent);
    if exponent < 0 {
        if value.is_zero() {
            return None;
        }
        value = BigRational::one() / value;
    }

    Some(ctx.add(Expr::Number(value)))
}

fn finite_cbrt_multiple_power_result(
    ctx: &mut Context,
    base_limit: ExprId,
    exponent: i64,
) -> Option<ExprId> {
    if exponent % 3 != 0 {
        return None;
    }

    let Expr::Function(fn_id, args) = ctx.get(base_limit).clone() else {
        return None;
    };
    if !ctx.is_builtin(fn_id, BuiltinFn::Cbrt) || args.len() != 1 {
        return None;
    }

    let radicand_value = numeric_limit_value(ctx, args[0])?;
    if exponent <= 0 && radicand_value.is_zero() {
        return None;
    }

    let reduced_exponent = exponent.unsigned_abs() / 3;
    if reduced_exponent > FINITE_INTEGER_POWER_EXACT_FOLD_LIMIT {
        return None;
    }

    let mut value = rational_pow_nonnegative(&radicand_value, reduced_exponent);
    if exponent < 0 {
        if value.is_zero() {
            return None;
        }
        value = BigRational::one() / value;
    }

    Some(ctx.add(Expr::Number(value)))
}

fn finite_integer_power_result(
    ctx: &mut Context,
    base_limit: ExprId,
    exponent: i64,
) -> Option<ExprId> {
    if let Some(result) = finite_cbrt_multiple_power_result(ctx, base_limit, exponent) {
        return Some(result);
    }

    let base_nonzero = finite_denominator_proven_nonzero(ctx, base_limit);
    if exponent <= 0 && !base_nonzero {
        return None;
    }

    if exponent == 0 {
        return Some(ctx.num(1));
    }

    if let Some(result) = finite_sqrt_even_power_result(ctx, base_limit, exponent) {
        return Some(result);
    }

    if let Some(base_value) = numeric_limit_value(ctx, base_limit) {
        let abs_exponent = exponent.unsigned_abs();
        if abs_exponent <= FINITE_INTEGER_POWER_EXACT_FOLD_LIMIT {
            let mut value = rational_pow_nonnegative(&base_value, abs_exponent);
            if exponent < 0 {
                if value.is_zero() {
                    return None;
                }
                value = BigRational::one() / value;
            }
            return Some(ctx.add(Expr::Number(value)));
        }
    }

    let exponent_expr = if exponent > 0 {
        ctx.add(Expr::Number(BigRational::from_integer(BigInt::from(
            exponent,
        ))))
    } else {
        let positive_exponent = exponent.checked_neg()?;
        let positive_exponent_expr = ctx.add(Expr::Number(BigRational::from_integer(
            BigInt::from(positive_exponent),
        )));
        let denominator = if positive_exponent == 1 {
            base_limit
        } else {
            ctx.add(Expr::Pow(base_limit, positive_exponent_expr))
        };
        let one = ctx.num(1);
        return Some(ctx.add(Expr::Div(one, denominator)));
    };

    Some(ctx.add(Expr::Pow(base_limit, exponent_expr)))
}

fn apply_finite_integer_power_composition_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    let (base_expr, exponent) = parse_pow_int(ctx, expr)?;
    let base_limit = try_limit_rules_at_finite(ctx, base_expr, var, point)?;
    finite_integer_power_result(ctx, base_limit, exponent)
}

fn apply_finite_elementary_polynomial_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    if depends_on(ctx, point, var) {
        return None;
    }
    let Expr::Variable(var_symbol) = ctx.get(var) else {
        return None;
    };
    let var_name = ctx.sym_name(*var_symbol);
    let Expr::Number(point_value) = ctx.get(point) else {
        return None;
    };
    let point_value = point_value.clone();
    let (builtin, argument_expr) = match ctx.get(expr).clone() {
        Expr::Function(fn_id, args) => {
            if args.len() != 1 {
                return None;
            }
            (ctx.builtin_of(fn_id)?, args[0])
        }
        Expr::Pow(base, exp) if matches!(ctx.get(base), Expr::Constant(Constant::E)) => {
            (BuiltinFn::Exp, exp)
        }
        _ => return None,
    };
    if !matches!(
        builtin,
        BuiltinFn::Exp
            | BuiltinFn::Sin
            | BuiltinFn::Cos
            | BuiltinFn::Sinh
            | BuiltinFn::Cosh
            | BuiltinFn::Tanh
            | BuiltinFn::Atan
            | BuiltinFn::Arctan
            | BuiltinFn::Asinh
            | BuiltinFn::Cbrt
            | BuiltinFn::Abs
            | BuiltinFn::Ln
            | BuiltinFn::Log2
            | BuiltinFn::Log10
            | BuiltinFn::Sqrt
    ) {
        return None;
    }

    let argument = Polynomial::from_expr(ctx, argument_expr, var_name).ok()?;
    let argument_value = argument.eval(&point_value);
    if is_finite_total_real_unary_builtin(builtin) {
        let argument_limit = ctx.add(Expr::Number(argument_value));
        return Some(finite_total_real_unary_result(ctx, builtin, argument_limit));
    }
    if is_finite_positive_domain_unary_builtin(builtin) {
        let argument_limit = ctx.add(Expr::Number(argument_value));
        return finite_positive_domain_unary_result(ctx, builtin, argument_limit);
    }

    None
}

fn pow_one_third_argument(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return None;
    };
    match ctx.get(*exp) {
        Expr::Number(value) if *value.numer() == 1.into() && *value.denom() == 3.into() => {
            Some(*base)
        }
        Expr::Div(num, den) => {
            let (Expr::Number(num_value), Expr::Number(den_value)) = (ctx.get(*num), ctx.get(*den))
            else {
                return None;
            };
            if num_value.is_one() && den_value.is_integer() && *den_value.numer() == 3.into() {
                return Some(*base);
            }
            None
        }
        _ => None,
    }
}

fn apply_finite_cube_root_power_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    let argument_expr = pow_one_third_argument(ctx, expr)?;
    let argument_limit = try_limit_rules_at_finite(ctx, argument_expr, var, point)?;
    Some(finite_total_real_unary_result(
        ctx,
        BuiltinFn::Cbrt,
        argument_limit,
    ))
}

fn apply_finite_total_real_unary_composition_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    let (builtin, argument_expr) = match ctx.get(expr).clone() {
        Expr::Function(fn_id, args) => {
            if args.len() != 1 {
                return None;
            }
            (ctx.builtin_of(fn_id)?, args[0])
        }
        Expr::Pow(base, exp) if matches!(ctx.get(base), Expr::Constant(Constant::E)) => {
            (BuiltinFn::Exp, exp)
        }
        _ => return None,
    };
    if !is_finite_total_real_unary_builtin(builtin) {
        return None;
    }

    let argument_limit = try_limit_rules_at_finite(ctx, argument_expr, var, point)?;
    Some(finite_total_real_unary_result(ctx, builtin, argument_limit))
}

fn apply_finite_positive_domain_unary_composition_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(expr).clone() else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }
    let builtin = ctx.builtin_of(fn_id)?;
    if !is_finite_positive_domain_unary_builtin(builtin) {
        return None;
    }

    let argument_limit = try_limit_rules_at_finite(ctx, args[0], var, point)?;
    finite_positive_domain_unary_result(ctx, builtin, argument_limit)
}

fn apply_finite_binary_log_composition_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(expr).clone() else {
        return None;
    };
    if !ctx.is_builtin(fn_id, BuiltinFn::Log) || args.len() != 2 {
        return None;
    }

    let base_limit = try_limit_rules_at_finite(ctx, args[0], var, point)?;
    let argument_limit = try_limit_rules_at_finite(ctx, args[1], var, point)?;
    finite_log_result(ctx, base_limit, argument_limit)
}

fn try_limit_rules_at_finite(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    if let Some(result) = apply_constant_rule(ctx, expr, var) {
        return Some(result);
    }
    if expr == var && !depends_on(ctx, point, var) {
        return Some(point);
    }
    if let Some(result) = apply_finite_polynomial_rule(ctx, expr, var, point) {
        return Some(result);
    }
    if let Some(result) = apply_finite_rational_polynomial_rule(ctx, expr, var, point) {
        return Some(result);
    }
    if let Some(result) = apply_finite_elementary_polynomial_rule(ctx, expr, var, point) {
        return Some(result);
    }
    if let Some(result) = apply_finite_cube_root_power_rule(ctx, expr, var, point) {
        return Some(result);
    }
    if let Some(result) = apply_finite_integer_power_composition_rule(ctx, expr, var, point) {
        return Some(result);
    }
    if let Some(result) = apply_finite_positive_domain_unary_composition_rule(ctx, expr, var, point)
    {
        return Some(result);
    }
    if let Some(result) = apply_finite_binary_log_composition_rule(ctx, expr, var, point) {
        return Some(result);
    }
    if let Some(result) = apply_finite_total_real_unary_composition_rule(ctx, expr, var, point) {
        return Some(result);
    }

    match ctx.get(expr).clone() {
        Expr::Add(lhs, rhs) => {
            let lhs_limit = try_limit_rules_at_finite(ctx, lhs, var, point)?;
            let rhs_limit = try_limit_rules_at_finite(ctx, rhs, var, point)?;
            Some(finite_add_result(ctx, lhs_limit, rhs_limit))
        }
        Expr::Sub(lhs, rhs) => {
            let lhs_limit = try_limit_rules_at_finite(ctx, lhs, var, point)?;
            let rhs_limit = try_limit_rules_at_finite(ctx, rhs, var, point)?;
            Some(finite_sub_result(ctx, lhs_limit, rhs_limit))
        }
        Expr::Mul(lhs, rhs) => {
            let lhs_limit = try_limit_rules_at_finite(ctx, lhs, var, point)?;
            let rhs_limit = try_limit_rules_at_finite(ctx, rhs, var, point)?;
            Some(finite_mul_result(ctx, lhs_limit, rhs_limit))
        }
        Expr::Div(num, den) => {
            let num_limit = try_limit_rules_at_finite(ctx, num, var, point)?;
            let den_limit = try_limit_rules_at_finite(ctx, den, var, point)?;
            finite_div_result(ctx, num_limit, den_limit)
        }
        Expr::Neg(inner) => {
            let inner_limit = try_limit_rules_at_finite(ctx, inner, var, point)?;
            Some(finite_neg_result(ctx, inner_limit))
        }
        _ => None,
    }
}

/// Rule 2: Variable - lim x = ±∞ based on approach sign.
pub fn apply_variable_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    if expr != var {
        return None;
    }
    Some(mk_infinity(ctx, approach))
}

/// Rule 3: Power - lim x^n for integer n.
///
/// - n > 0: ±∞ (sign depends on approach and parity)
/// - n = 0: 1
/// - n < 0: 0
pub fn apply_power_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    let (base, n) = parse_pow_int(ctx, expr)?;

    // Base must be exactly the limit variable
    if base != var {
        return None;
    }

    if n == 0 {
        return Some(ctx.num(1));
    }
    if n < 0 {
        return Some(ctx.num(0));
    }

    let sign = limit_sign(approach, n);
    Some(mk_infinity(ctx, sign))
}

/// Rule 4: Reciprocal power - lim c/x^n = 0 for n > 0 and c independent of x.
pub fn apply_reciprocal_power_rule(ctx: &mut Context, expr: ExprId, var: ExprId) -> Option<ExprId> {
    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };

    // Numerator must be constant wrt variable.
    if depends_on(ctx, num, var) {
        return None;
    }

    // Denominator must be x^n with n > 0, or plain x.
    let power = if den == var {
        1
    } else if let Some((base, n)) = parse_pow_int(ctx, den) {
        if base != var || n <= 0 {
            return None;
        }
        n
    } else {
        return None;
    };

    if power > 0 {
        Some(ctx.num(0))
    } else {
        None
    }
}

fn infinity_sign_of_expr(ctx: &Context, expr: ExprId) -> Option<InfSign> {
    match ctx.get(expr) {
        Expr::Constant(Constant::Infinity) => Some(InfSign::Pos),
        Expr::Neg(inner) if matches!(ctx.get(*inner), Expr::Constant(Constant::Infinity)) => {
            Some(InfSign::Neg)
        }
        _ => None,
    }
}

fn neg_inf_sign(sign: InfSign) -> InfSign {
    match sign {
        InfSign::Pos => InfSign::Neg,
        InfSign::Neg => InfSign::Pos,
    }
}

fn negate_limit_result(ctx: &mut Context, expr: ExprId) -> ExprId {
    if let Some(sign) = infinity_sign_of_expr(ctx, expr) {
        return mk_infinity(ctx, neg_inf_sign(sign));
    }

    match ctx.get(expr).clone() {
        Expr::Number(value) => ctx.add(Expr::Number(-value)),
        _ => ctx.add(Expr::Neg(expr)),
    }
}

fn numeric_limit_value(ctx: &Context, expr: ExprId) -> Option<BigRational> {
    match ctx.get(expr) {
        Expr::Number(value) => Some(value.clone()),
        Expr::Neg(inner) => numeric_limit_value(ctx, *inner).map(|value| -value),
        Expr::Add(lhs, rhs) => {
            Some(numeric_limit_value(ctx, *lhs)? + numeric_limit_value(ctx, *rhs)?)
        }
        Expr::Sub(lhs, rhs) => {
            Some(numeric_limit_value(ctx, *lhs)? - numeric_limit_value(ctx, *rhs)?)
        }
        Expr::Mul(lhs, rhs) => {
            Some(numeric_limit_value(ctx, *lhs)? * numeric_limit_value(ctx, *rhs)?)
        }
        Expr::Div(num, den) => {
            let den_value = numeric_limit_value(ctx, *den)?;
            if den_value.is_zero() {
                return None;
            }
            Some(numeric_limit_value(ctx, *num)? / den_value)
        }
        _ => None,
    }
}

fn finite_numeric_expr(ctx: &mut Context, value: BigRational) -> ExprId {
    ctx.add(Expr::Number(value))
}

fn finite_limit_is_numeric_zero(ctx: &Context, expr: ExprId) -> bool {
    numeric_limit_value(ctx, expr).is_some_and(|value| value.is_zero())
}

fn finite_limit_is_numeric_one(ctx: &Context, expr: ExprId) -> bool {
    numeric_limit_value(ctx, expr).is_some_and(|value| value.is_one())
}

fn finite_add_result(ctx: &mut Context, lhs: ExprId, rhs: ExprId) -> ExprId {
    if let (Some(lhs_value), Some(rhs_value)) =
        (numeric_limit_value(ctx, lhs), numeric_limit_value(ctx, rhs))
    {
        return finite_numeric_expr(ctx, lhs_value + rhs_value);
    }
    if finite_limit_is_numeric_zero(ctx, lhs) {
        return rhs;
    }
    if finite_limit_is_numeric_zero(ctx, rhs) {
        return lhs;
    }
    ctx.add(Expr::Add(lhs, rhs))
}

fn finite_sub_result(ctx: &mut Context, lhs: ExprId, rhs: ExprId) -> ExprId {
    if lhs == rhs {
        return ctx.num(0);
    }
    if let (Some(lhs_value), Some(rhs_value)) =
        (numeric_limit_value(ctx, lhs), numeric_limit_value(ctx, rhs))
    {
        return finite_numeric_expr(ctx, lhs_value - rhs_value);
    }
    if finite_limit_is_numeric_zero(ctx, rhs) {
        return lhs;
    }
    if finite_limit_is_numeric_zero(ctx, lhs) {
        return negate_limit_result(ctx, rhs);
    }
    ctx.add(Expr::Sub(lhs, rhs))
}

fn finite_mul_result(ctx: &mut Context, lhs: ExprId, rhs: ExprId) -> ExprId {
    if let (Some(lhs_value), Some(rhs_value)) =
        (numeric_limit_value(ctx, lhs), numeric_limit_value(ctx, rhs))
    {
        return finite_numeric_expr(ctx, lhs_value * rhs_value);
    }
    if finite_limit_is_numeric_zero(ctx, lhs) || finite_limit_is_numeric_zero(ctx, rhs) {
        return ctx.num(0);
    }
    if finite_limit_is_numeric_one(ctx, lhs) {
        return rhs;
    }
    if finite_limit_is_numeric_one(ctx, rhs) {
        return lhs;
    }
    ctx.add(Expr::Mul(lhs, rhs))
}

fn finite_div_result(ctx: &mut Context, num: ExprId, den: ExprId) -> Option<ExprId> {
    if !finite_denominator_proven_nonzero(ctx, den) {
        return None;
    }
    if num == den {
        return Some(ctx.num(1));
    }
    if let (Some(num_value), Some(den_value)) =
        (numeric_limit_value(ctx, num), numeric_limit_value(ctx, den))
    {
        if den_value.is_zero() {
            return None;
        }
        return Some(finite_numeric_expr(ctx, num_value / den_value));
    }
    if finite_limit_is_numeric_zero(ctx, num) {
        return Some(ctx.num(0));
    }
    if finite_limit_is_numeric_one(ctx, den) {
        return Some(num);
    }
    Some(ctx.add(Expr::Div(num, den)))
}

fn finite_neg_result(ctx: &mut Context, inner: ExprId) -> ExprId {
    if let Some(value) = numeric_limit_value(ctx, inner) {
        return finite_numeric_expr(ctx, -value);
    }
    negate_limit_result(ctx, inner)
}

fn finite_expr_proven_positive(ctx: &Context, expr: ExprId) -> bool {
    crate::prove_sign::prove_positive_depth_with(ctx, expr, 4, true, |_, _, _| {
        crate::tri_proof::TriProof::Unknown
    })
    .is_proven()
}

fn finite_denominator_proven_nonzero(ctx: &Context, expr: ExprId) -> bool {
    if numeric_limit_value(ctx, expr).is_some_and(|value| !value.is_zero()) {
        return true;
    }
    if finite_expr_proven_positive(ctx, expr) {
        return true;
    }
    match ctx.get(expr) {
        Expr::Neg(inner) => finite_expr_proven_positive(ctx, *inner),
        _ => false,
    }
}

fn scale_infinity(ctx: &mut Context, scale: &BigRational, sign: InfSign) -> Option<ExprId> {
    if scale.is_zero() {
        return None;
    }

    let result_sign = if scale.is_positive() {
        sign
    } else {
        neg_inf_sign(sign)
    };
    Some(mk_infinity(ctx, result_sign))
}

#[derive(Debug, Clone)]
struct PolynomialGrowthInfo {
    degree: u32,
    leading_coeff: BigRational,
}

fn polynomial_growth_info(
    ctx: &Context,
    expr: ExprId,
    var: ExprId,
) -> Option<PolynomialGrowthInfo> {
    use crate::multipoly::{multipoly_from_expr, PolyBudget};

    let Expr::Variable(var_sym_id) = ctx.get(var).clone() else {
        return None;
    };
    let var_name = ctx.sym_name(var_sym_id);

    let budget = PolyBudget {
        max_terms: 100,
        max_total_degree: 20,
        max_pow_exp: 4,
    };

    let poly = multipoly_from_expr(ctx, expr, &budget).ok()?;
    if poly.is_zero() {
        return None;
    }

    let var_idx = poly.var_index(var_name)?;
    let degree = poly.degree_in(var_idx);
    if degree == 0 {
        return None;
    }

    let leading_coeff = poly.leading_coeff_in(var_idx).constant_value()?;
    Some(PolynomialGrowthInfo {
        degree,
        leading_coeff,
    })
}

fn negate_polynomial_growth_info(mut growth: PolynomialGrowthInfo) -> PolynomialGrowthInfo {
    growth.leading_coeff = -growth.leading_coeff;
    growth
}

fn scale_polynomial_growth_info(
    mut growth: PolynomialGrowthInfo,
    scale: BigRational,
) -> Option<PolynomialGrowthInfo> {
    if scale.is_zero() {
        return None;
    }
    growth.leading_coeff *= scale;
    Some(growth)
}

fn polynomial_growth_info_with_bounded_additive_noise(
    ctx: &Context,
    expr: ExprId,
    var: ExprId,
) -> Option<PolynomialGrowthInfo> {
    if let Some(growth) = polynomial_growth_info(ctx, expr, var) {
        return Some(growth);
    }

    match ctx.get(expr).clone() {
        Expr::Add(lhs, rhs) => {
            if let Some(growth) = polynomial_growth_info_with_bounded_additive_noise(ctx, lhs, var)
            {
                if is_bounded_elementary_expr_at_infinity(ctx, rhs, var) {
                    return Some(growth);
                }
            }
            if let Some(growth) = polynomial_growth_info_with_bounded_additive_noise(ctx, rhs, var)
            {
                if is_bounded_elementary_expr_at_infinity(ctx, lhs, var) {
                    return Some(growth);
                }
            }
            None
        }
        Expr::Sub(lhs, rhs) => {
            if let Some(growth) = polynomial_growth_info_with_bounded_additive_noise(ctx, lhs, var)
            {
                if is_bounded_elementary_expr_at_infinity(ctx, rhs, var) {
                    return Some(growth);
                }
            }
            if let Some(growth) = polynomial_growth_info_with_bounded_additive_noise(ctx, rhs, var)
            {
                if is_bounded_elementary_expr_at_infinity(ctx, lhs, var) {
                    return Some(negate_polynomial_growth_info(growth));
                }
            }
            None
        }
        Expr::Mul(lhs, rhs) => {
            if let Some(scale) = numeric_limit_value(ctx, lhs) {
                return scale_polynomial_growth_info(
                    polynomial_growth_info_with_bounded_additive_noise(ctx, rhs, var)?,
                    scale,
                );
            }
            if let Some(scale) = numeric_limit_value(ctx, rhs) {
                return scale_polynomial_growth_info(
                    polynomial_growth_info_with_bounded_additive_noise(ctx, lhs, var)?,
                    scale,
                );
            }
            None
        }
        Expr::Neg(inner) => Some(negate_polynomial_growth_info(
            polynomial_growth_info_with_bounded_additive_noise(ctx, inner, var)?,
        )),
        _ => None,
    }
}

fn scaled_square_root_base(ctx: &Context, expr: ExprId) -> Option<(BigRational, ExprId)> {
    if let Some(radicand) = extract_square_root_base(ctx, expr) {
        return Some((BigRational::from_integer(BigInt::from(1)), radicand));
    }

    match ctx.get(expr).clone() {
        Expr::Mul(lhs, rhs) => {
            if let Some(scale) = numeric_limit_value(ctx, lhs) {
                if scale.is_zero() {
                    return None;
                }
                return extract_square_root_base(ctx, rhs).map(|radicand| (scale, radicand));
            }
            if let Some(scale) = numeric_limit_value(ctx, rhs) {
                if scale.is_zero() {
                    return None;
                }
                return extract_square_root_base(ctx, lhs).map(|radicand| (scale, radicand));
            }
            None
        }
        Expr::Neg(inner) => {
            scaled_square_root_base(ctx, inner).map(|(scale, radicand)| (-scale, radicand))
        }
        _ => None,
    }
}

fn sqrt_polynomial_ratio_limit_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };

    let (sqrt_scale, radicand) = scaled_square_root_base(ctx, num)?;
    let radicand_growth = polynomial_growth_info_with_bounded_additive_noise(ctx, radicand, var)?;
    let den_growth = polynomial_growth_info_with_bounded_additive_noise(ctx, den, var)?;

    if radicand_growth.degree == 0 || radicand_growth.degree % 2 != 0 {
        return None;
    }
    if !radicand_growth.leading_coeff.is_positive() || den_growth.leading_coeff.is_zero() {
        return None;
    }

    let sqrt_degree = radicand_growth.degree / 2;
    if den_growth.degree != sqrt_degree {
        return None;
    }

    if let Some(sqrt_leading_coeff) = rational_sqrt(&radicand_growth.leading_coeff) {
        let mut ratio = sqrt_scale * sqrt_leading_coeff / den_growth.leading_coeff;
        if approach == InfSign::Neg && sqrt_degree % 2 == 1 {
            ratio = -ratio;
        }
        return Some(ctx.add(Expr::Number(ratio)));
    }

    let leading_coeff = ctx.add(Expr::Number(radicand_growth.leading_coeff));
    let sqrt_leading_coeff = ctx.call_builtin(BuiltinFn::Sqrt, vec![leading_coeff]);
    let denominator_abs = den_growth.leading_coeff.abs();
    let scale_abs = sqrt_scale.abs();
    let one = BigRational::from_integer(BigInt::from(1));
    let scaled_sqrt = if scale_abs == one {
        sqrt_leading_coeff
    } else {
        let multiplier = ctx.add(Expr::Number(scale_abs));
        ctx.add(Expr::Mul(multiplier, sqrt_leading_coeff))
    };
    let unsigned_result = if denominator_abs == one {
        scaled_sqrt
    } else {
        let denominator = ctx.add(Expr::Number(denominator_abs));
        ctx.add(Expr::Div(scaled_sqrt, denominator))
    };

    let flips_at_neg_infinity = approach == InfSign::Neg && sqrt_degree % 2 == 1;
    let needs_negation =
        sqrt_scale.is_negative() ^ den_growth.leading_coeff.is_negative() ^ flips_at_neg_infinity;
    if needs_negation {
        Some(ctx.add(Expr::Neg(unsigned_result)))
    } else {
        Some(unsigned_result)
    }
}

fn rationalized_surd_product(
    ctx: &mut Context,
    coeff: BigRational,
    radicand: BigRational,
) -> ExprId {
    if coeff.is_zero() {
        return ctx.add(Expr::Number(coeff));
    }

    let sqrt_radicand = ctx.add(Expr::Number(radicand));
    let sqrt_expr = ctx.call_builtin(BuiltinFn::Sqrt, vec![sqrt_radicand]);
    let abs_coeff = coeff.abs();
    let one_int = BigInt::from(1);

    let numerator = if abs_coeff.numer() == &one_int {
        sqrt_expr
    } else {
        let multiplier = ctx.add(Expr::Number(BigRational::from_integer(
            abs_coeff.numer().clone(),
        )));
        ctx.add(Expr::Mul(multiplier, sqrt_expr))
    };

    let unsigned = if abs_coeff.denom() == &one_int {
        numerator
    } else {
        let denominator = ctx.add(Expr::Number(BigRational::from_integer(
            abs_coeff.denom().clone(),
        )));
        ctx.add(Expr::Div(numerator, denominator))
    };

    if coeff.is_negative() {
        ctx.add(Expr::Neg(unsigned))
    } else {
        unsigned
    }
}

fn polynomial_sqrt_ratio_limit_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };

    let radicand = extract_square_root_base(ctx, den)?;
    let num_growth = polynomial_growth_info_with_bounded_additive_noise(ctx, num, var)?;
    let radicand_growth = polynomial_growth_info_with_bounded_additive_noise(ctx, radicand, var)?;

    if radicand_growth.degree == 0 || radicand_growth.degree % 2 != 0 {
        return None;
    }
    if !radicand_growth.leading_coeff.is_positive() || num_growth.leading_coeff.is_zero() {
        return None;
    }

    let sqrt_degree = radicand_growth.degree / 2;
    if num_growth.degree != sqrt_degree {
        return None;
    }

    let mut signed_num_lc = num_growth.leading_coeff;
    if approach == InfSign::Neg && sqrt_degree % 2 == 1 {
        signed_num_lc = -signed_num_lc;
    }

    if let Some(sqrt_leading_coeff) = rational_sqrt(&radicand_growth.leading_coeff) {
        return Some(ctx.add(Expr::Number(signed_num_lc / sqrt_leading_coeff)));
    }

    let coeff = signed_num_lc / radicand_growth.leading_coeff.clone();
    Some(rationalized_surd_product(
        ctx,
        coeff,
        radicand_growth.leading_coeff,
    ))
}

fn combine_add_limit_results(ctx: &mut Context, lhs: ExprId, rhs: ExprId) -> Option<ExprId> {
    match (
        infinity_sign_of_expr(ctx, lhs),
        infinity_sign_of_expr(ctx, rhs),
    ) {
        (Some(left), Some(right)) if left == right => Some(mk_infinity(ctx, left)),
        (Some(_), Some(_)) => None,
        (Some(sign), None) | (None, Some(sign)) => Some(mk_infinity(ctx, sign)),
        (None, None) => {
            if let (Some(lhs_value), Some(rhs_value)) =
                (numeric_limit_value(ctx, lhs), numeric_limit_value(ctx, rhs))
            {
                return Some(ctx.add(Expr::Number(lhs_value + rhs_value)));
            }
            Some(ctx.add(Expr::Add(lhs, rhs)))
        }
    }
}

fn combine_sub_limit_results(ctx: &mut Context, lhs: ExprId, rhs: ExprId) -> Option<ExprId> {
    match (
        infinity_sign_of_expr(ctx, lhs),
        infinity_sign_of_expr(ctx, rhs),
    ) {
        (Some(left), Some(right)) if left == right => None,
        (Some(left), Some(right)) if left != right => Some(mk_infinity(ctx, left)),
        (Some(sign), None) => Some(mk_infinity(ctx, sign)),
        (None, Some(sign)) => Some(mk_infinity(ctx, neg_inf_sign(sign))),
        (None, None) => {
            if let (Some(lhs_value), Some(rhs_value)) =
                (numeric_limit_value(ctx, lhs), numeric_limit_value(ctx, rhs))
            {
                return Some(ctx.add(Expr::Number(lhs_value - rhs_value)));
            }
            Some(ctx.add(Expr::Sub(lhs, rhs)))
        }
        _ => None,
    }
}

fn combine_mul_limit_results(ctx: &mut Context, lhs: ExprId, rhs: ExprId) -> Option<ExprId> {
    let lhs_inf = infinity_sign_of_expr(ctx, lhs);
    let rhs_inf = infinity_sign_of_expr(ctx, rhs);

    match (lhs_inf, rhs_inf) {
        (Some(left), Some(right)) => {
            let sign = if left == right {
                InfSign::Pos
            } else {
                InfSign::Neg
            };
            return Some(mk_infinity(ctx, sign));
        }
        (Some(sign), None) => {
            let scale = numeric_limit_value(ctx, rhs)?;
            return scale_infinity(ctx, &scale, sign);
        }
        (None, Some(sign)) => {
            let scale = numeric_limit_value(ctx, lhs)?;
            return scale_infinity(ctx, &scale, sign);
        }
        (None, None) => {}
    }

    let lhs_value = numeric_limit_value(ctx, lhs)?;
    let rhs_value = numeric_limit_value(ctx, rhs)?;
    Some(ctx.add(Expr::Number(lhs_value * rhs_value)))
}

fn combine_div_limit_results(ctx: &mut Context, num: ExprId, den: ExprId) -> Option<ExprId> {
    let num_inf = infinity_sign_of_expr(ctx, num);
    let den_inf = infinity_sign_of_expr(ctx, den);

    match (num_inf, den_inf) {
        (Some(_), Some(_)) => return None,
        (Some(sign), None) => {
            let den_value = numeric_limit_value(ctx, den)?;
            if den_value.is_zero() {
                return None;
            }
            return scale_infinity(
                ctx,
                &(BigRational::from_integer(BigInt::from(1)) / den_value),
                sign,
            );
        }
        (None, Some(_)) => {
            numeric_limit_value(ctx, num)?;
            return Some(ctx.num(0));
        }
        (None, None) => {}
    }

    let num_value = numeric_limit_value(ctx, num)?;
    let den_value = numeric_limit_value(ctx, den)?;
    if den_value.is_zero() {
        return None;
    }
    Some(ctx.add(Expr::Number(num_value / den_value)))
}

fn limit_growth_sign(leading_coeff: &BigRational, degree: u32, approach: InfSign) -> InfSign {
    let coeff_positive = leading_coeff.is_positive();
    let power_positive = match approach {
        InfSign::Pos => true,
        InfSign::Neg => degree.is_multiple_of(2),
    };

    if coeff_positive == power_positive {
        InfSign::Pos
    } else {
        InfSign::Neg
    }
}

fn linear_argument_tail_sign(
    ctx: &Context,
    arg: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<InfSign> {
    let growth = polynomial_growth_info(ctx, arg, var)?;
    if growth.degree != 1 {
        return None;
    }
    Some(limit_growth_sign(
        &growth.leading_coeff,
        growth.degree,
        approach,
    ))
}

#[derive(Debug, Clone)]
struct ScaledLinearExpTailInfo {
    coeff: BigRational,
    tail: InfSign,
}

fn linear_exp_tail_sign(
    ctx: &Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<InfSign> {
    match ctx.get(expr).clone() {
        Expr::Function(fn_id, args)
            if args.len() == 1 && matches!(ctx.builtin_of(fn_id), Some(BuiltinFn::Exp)) =>
        {
            linear_argument_tail_sign(ctx, args[0], var, approach)
        }
        Expr::Pow(base, exp) if matches!(ctx.get(base), Expr::Constant(Constant::E)) => {
            linear_argument_tail_sign(ctx, exp, var, approach)
        }
        _ => None,
    }
}

fn scaled_linear_exp_tail_info(
    ctx: &Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ScaledLinearExpTailInfo> {
    if let Some(tail) = linear_exp_tail_sign(ctx, expr, var, approach) {
        return Some(ScaledLinearExpTailInfo {
            coeff: BigRational::from_integer(BigInt::from(1)),
            tail,
        });
    }

    match ctx.get(expr).clone() {
        Expr::Neg(inner) => {
            let mut info = scaled_linear_exp_tail_info(ctx, inner, var, approach)?;
            info.coeff = -info.coeff;
            Some(info)
        }
        Expr::Mul(lhs, rhs) => {
            if let Some(lhs_scale) = numeric_limit_value(ctx, lhs) {
                if let Some(mut rhs_info) = scaled_linear_exp_tail_info(ctx, rhs, var, approach) {
                    rhs_info.coeff *= lhs_scale;
                    return Some(rhs_info);
                }
            }
            if let Some(rhs_scale) = numeric_limit_value(ctx, rhs) {
                if let Some(mut lhs_info) = scaled_linear_exp_tail_info(ctx, lhs, var, approach) {
                    lhs_info.coeff *= rhs_scale;
                    return Some(lhs_info);
                }
            }
            None
        }
        _ => None,
    }
}

fn nonzero_scaled_linear_exp_tail_info(
    ctx: &Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ScaledLinearExpTailInfo> {
    let info = scaled_linear_exp_tail_info(ctx, expr, var, approach)?;
    if info.coeff.is_zero() {
        None
    } else {
        Some(info)
    }
}

fn polynomial_or_numeric_tail_sign(
    ctx: &Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<InfSign> {
    if let Some(growth) = polynomial_growth_info(ctx, expr, var) {
        return Some(limit_growth_sign(
            &growth.leading_coeff,
            growth.degree,
            approach,
        ));
    }

    let value = numeric_limit_value(ctx, expr)?;
    if value.is_zero() {
        None
    } else if value.is_positive() {
        Some(InfSign::Pos)
    } else {
        Some(InfSign::Neg)
    }
}

fn rational_one() -> BigRational {
    BigRational::from_integer(BigInt::from(1))
}

fn constant_rational_value(ctx: &Context, expr: ExprId) -> Option<BigRational> {
    match ctx.get(expr).clone() {
        Expr::Number(value) => Some(value),
        Expr::Neg(inner) => Some(-constant_rational_value(ctx, inner)?),
        Expr::Div(num, den) => {
            let den_value = constant_rational_value(ctx, den)?;
            if den_value.is_zero() {
                return None;
            }
            Some(constant_rational_value(ctx, num)? / den_value)
        }
        _ => None,
    }
}

fn is_exact_named_gt_one_base(ctx: &Context, expr: ExprId) -> bool {
    matches!(
        ctx.get(expr),
        Expr::Constant(Constant::E | Constant::Pi | Constant::Phi)
    )
}

fn is_rational_one(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Number(value) if value == &rational_one())
}

fn exact_named_log_base_tail_sign(ctx: &Context, base: ExprId) -> Option<InfSign> {
    if is_exact_named_gt_one_base(ctx, base) {
        return Some(InfSign::Pos);
    }

    if let Expr::Div(num, den) = ctx.get(base).clone() {
        if is_rational_one(ctx, num) {
            return match exact_named_log_base_tail_sign(ctx, den)? {
                InfSign::Pos => Some(InfSign::Neg),
                InfSign::Neg => Some(InfSign::Pos),
            };
        }
    }

    if let Expr::Pow(pow_base, exp) = ctx.get(base).clone() {
        let base_sign = exact_named_log_base_tail_sign(ctx, pow_base)?;
        let exponent = crate::expr_extract::extract_i64_integer(ctx, exp)?;
        if exponent == 0 {
            return None;
        }
        return match (base_sign, exponent.is_positive()) {
            (InfSign::Pos, true) | (InfSign::Neg, false) => Some(InfSign::Pos),
            (InfSign::Pos, false) | (InfSign::Neg, true) => Some(InfSign::Neg),
        };
    }

    None
}

fn log_base_tail_coeff_from_sign(sign: InfSign) -> BigRational {
    match sign {
        InfSign::Pos => rational_one(),
        InfSign::Neg => -rational_one(),
    }
}

fn positive_log_base_tail_coeff(ctx: &Context, base: ExprId) -> Option<BigRational> {
    if let Some(sign) = exact_named_log_base_tail_sign(ctx, base) {
        return Some(log_base_tail_coeff_from_sign(sign));
    }

    let base_value = constant_rational_value(ctx, base)?;
    let one = rational_one();
    if !base_value.is_positive() || base_value == one {
        return None;
    }
    if base_value > one {
        Some(one)
    } else {
        Some(-one)
    }
}

fn linear_subpolynomial_tail_coeff(
    ctx: &Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<BigRational> {
    match ctx.get(expr).clone() {
        Expr::Function(fn_id, args) => {
            let builtin = ctx.builtin_of(fn_id)?;
            match (builtin, args.as_slice()) {
                (BuiltinFn::Ln | BuiltinFn::Sqrt | BuiltinFn::Log2 | BuiltinFn::Log10, [arg])
                    if linear_argument_tail_sign(ctx, *arg, var, approach)? == InfSign::Pos =>
                {
                    Some(rational_one())
                }
                (BuiltinFn::Log, [base, arg])
                    if linear_argument_tail_sign(ctx, *arg, var, approach)? == InfSign::Pos =>
                {
                    positive_log_base_tail_coeff(ctx, *base)
                }
                _ => None,
            }
        }
        _ => None,
    }
}

#[derive(Debug, Clone)]
struct ScaledSubpolynomialTailInfo {
    coeff: BigRational,
}

fn scaled_linear_subpolynomial_tail_info(
    ctx: &Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ScaledSubpolynomialTailInfo> {
    if let Some(coeff) = linear_subpolynomial_tail_coeff(ctx, expr, var, approach) {
        return Some(ScaledSubpolynomialTailInfo { coeff });
    }

    match ctx.get(expr).clone() {
        Expr::Neg(inner) => {
            let mut info = scaled_linear_subpolynomial_tail_info(ctx, inner, var, approach)?;
            info.coeff = -info.coeff;
            Some(info)
        }
        Expr::Mul(lhs, rhs) => {
            if let Some(lhs_scale) = numeric_limit_value(ctx, lhs) {
                if let Some(mut rhs_info) =
                    scaled_linear_subpolynomial_tail_info(ctx, rhs, var, approach)
                {
                    rhs_info.coeff *= lhs_scale;
                    return Some(rhs_info);
                }
            }
            if let Some(rhs_scale) = numeric_limit_value(ctx, rhs) {
                if let Some(mut lhs_info) =
                    scaled_linear_subpolynomial_tail_info(ctx, lhs, var, approach)
                {
                    lhs_info.coeff *= rhs_scale;
                    return Some(lhs_info);
                }
            }
            None
        }
        _ => None,
    }
}

fn nonzero_scaled_linear_subpolynomial_tail_info(
    ctx: &Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ScaledSubpolynomialTailInfo> {
    let info = scaled_linear_subpolynomial_tail_info(ctx, expr, var, approach)?;
    if info.coeff.is_zero() {
        None
    } else {
        Some(info)
    }
}

fn subpolynomial_tail_sign(info: &ScaledSubpolynomialTailInfo) -> Option<InfSign> {
    if info.coeff.is_zero() {
        None
    } else if info.coeff.is_positive() {
        Some(InfSign::Pos)
    } else {
        Some(InfSign::Neg)
    }
}

fn elementary_linear_argument_limit_at_infinity(
    ctx: &mut Context,
    builtin: BuiltinFn,
    arg: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    let arg_tail = linear_argument_tail_sign(ctx, arg, var, approach)?;
    match (builtin, arg_tail) {
        (BuiltinFn::Sqrt | BuiltinFn::Ln, InfSign::Pos) => Some(mk_infinity(ctx, InfSign::Pos)),
        (BuiltinFn::Exp, InfSign::Pos) => Some(mk_infinity(ctx, InfSign::Pos)),
        (BuiltinFn::Exp, InfSign::Neg) => Some(ctx.num(0)),
        _ => None,
    }
}

/// Elementary linear-argument limits as `x -> ±∞`.
///
/// This intentionally accepts only degree-one polynomial arguments with numeric
/// leading coefficient. Higher-degree or non-polynomial compositions stay
/// residual until their domain and dominance policy are represented explicitly.
pub fn elementary_function_limit_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    match ctx.get(expr).clone() {
        Expr::Function(fn_id, args) if args.len() == 1 => {
            let builtin = ctx.builtin_of(fn_id)?;
            elementary_linear_argument_limit_at_infinity(ctx, builtin, args[0], var, approach)
        }
        Expr::Pow(base, exp) if matches!(ctx.get(base), Expr::Constant(Constant::E)) => {
            elementary_linear_argument_limit_at_infinity(ctx, BuiltinFn::Exp, exp, var, approach)
        }
        _ => None,
    }
}

pub fn additive_limit_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    match ctx.get(expr).clone() {
        Expr::Add(lhs, rhs) => {
            let lhs_limit = try_limit_rules_at_infinity(ctx, lhs, var, approach)?;
            let rhs_limit = try_limit_rules_at_infinity(ctx, rhs, var, approach)?;
            combine_add_limit_results(ctx, lhs_limit, rhs_limit)
        }
        Expr::Sub(lhs, rhs) => {
            let lhs_limit = try_limit_rules_at_infinity(ctx, lhs, var, approach)?;
            let rhs_limit = try_limit_rules_at_infinity(ctx, rhs, var, approach)?;
            combine_sub_limit_results(ctx, lhs_limit, rhs_limit)
        }
        _ => None,
    }
}

pub fn multiplicative_limit_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    match ctx.get(expr).clone() {
        Expr::Neg(inner) => {
            let inner_limit = try_limit_rules_at_infinity(ctx, inner, var, approach)?;
            Some(negate_limit_result(ctx, inner_limit))
        }
        Expr::Mul(lhs, rhs) => {
            let lhs_limit = try_limit_rules_at_infinity(ctx, lhs, var, approach)?;
            let rhs_limit = try_limit_rules_at_infinity(ctx, rhs, var, approach)?;
            combine_mul_limit_results(ctx, lhs_limit, rhs_limit)
        }
        Expr::Div(num, den) => {
            let num_limit = try_limit_rules_at_infinity(ctx, num, var, approach)?;
            let den_limit = try_limit_rules_at_infinity(ctx, den, var, approach)?;
            combine_div_limit_results(ctx, num_limit, den_limit)
        }
        _ => None,
    }
}

fn is_bounded_elementary_expr_at_infinity(ctx: &Context, expr: ExprId, var: ExprId) -> bool {
    if !depends_on(ctx, expr, var) {
        return true;
    }

    match ctx.get(expr) {
        Expr::Function(fn_id, args) if args.len() == 1 => {
            matches!(
                ctx.builtin_of(*fn_id),
                Some(
                    BuiltinFn::Sin
                        | BuiltinFn::Cos
                        | BuiltinFn::Atan
                        | BuiltinFn::Arctan
                        | BuiltinFn::Tanh,
                )
            )
        }
        Expr::Neg(inner) => is_bounded_elementary_expr_at_infinity(ctx, *inner, var),
        Expr::Add(lhs, rhs) | Expr::Sub(lhs, rhs) | Expr::Mul(lhs, rhs) => {
            is_bounded_elementary_expr_at_infinity(ctx, *lhs, var)
                && is_bounded_elementary_expr_at_infinity(ctx, *rhs, var)
        }
        Expr::Div(num, den) => {
            is_bounded_elementary_expr_at_infinity(ctx, *num, var)
                && !depends_on(ctx, *den, var)
                && constant_rational_value(ctx, *den).is_some_and(|value| !value.is_zero())
        }
        _ => false,
    }
}

/// Conservative bounded-over-divergent rule as `x -> ±∞`.
///
/// This is intentionally narrow: only real-domain globally bounded elementary
/// numerators (`sin`/`cos`/`arctan`/`tanh`, plus finite arithmetic combinations
/// of bounded pieces and constants) are accepted, and the denominator must
/// already be proven divergent by the existing infinity rules.
pub fn bounded_elementary_over_divergent_limit_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };

    if !depends_on(ctx, num, var) || !is_bounded_elementary_expr_at_infinity(ctx, num, var) {
        return None;
    }

    let den_limit = try_limit_rules_at_infinity(ctx, den, var, approach)?;
    infinity_sign_of_expr(ctx, den_limit)?;
    Some(ctx.num(0))
}

fn bounded_elementary_times_decaying_exp_limit_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    let Expr::Mul(lhs, rhs) = ctx.get(expr).clone() else {
        return None;
    };

    if let Some(exp_info) = scaled_linear_exp_tail_info(ctx, lhs, var, approach) {
        if (exp_info.coeff.is_zero() || exp_info.tail == InfSign::Neg)
            && is_bounded_elementary_expr_at_infinity(ctx, rhs, var)
        {
            return Some(ctx.num(0));
        }
    }

    if let Some(exp_info) = scaled_linear_exp_tail_info(ctx, rhs, var, approach) {
        if (exp_info.coeff.is_zero() || exp_info.tail == InfSign::Neg)
            && is_bounded_elementary_expr_at_infinity(ctx, lhs, var)
        {
            return Some(ctx.num(0));
        }
    }

    None
}

/// Dominance rule for linear-argument exponentials against polynomial growth as `x -> ±∞`.
///
/// This is intentionally narrow: only `exp(linear)`/`e^(linear)` with optional
/// numeric scaling is compared with polynomials whose relevant leading
/// coefficient is numeric. Nonlinear exponentials such as `exp(x^2)` remain
/// residual.
pub fn exponential_polynomial_dominance_limit_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    match ctx.get(expr).clone() {
        Expr::Add(lhs, rhs) => {
            if let Some(info) = nonzero_scaled_linear_exp_tail_info(ctx, lhs, var, approach) {
                if info.tail != InfSign::Pos {
                    return None;
                }
                polynomial_growth_info(ctx, rhs, var)?;
                return scale_infinity(ctx, &info.coeff, InfSign::Pos);
            }
            if let Some(info) = nonzero_scaled_linear_exp_tail_info(ctx, rhs, var, approach) {
                if info.tail != InfSign::Pos {
                    return None;
                }
                polynomial_growth_info(ctx, lhs, var)?;
                return scale_infinity(ctx, &info.coeff, InfSign::Pos);
            }
            None
        }
        Expr::Sub(lhs, rhs) => {
            if let Some(info) = nonzero_scaled_linear_exp_tail_info(ctx, lhs, var, approach) {
                if info.tail != InfSign::Pos {
                    return None;
                }
                polynomial_growth_info(ctx, rhs, var)?;
                return scale_infinity(ctx, &info.coeff, InfSign::Pos);
            }
            if let Some(info) = nonzero_scaled_linear_exp_tail_info(ctx, rhs, var, approach) {
                if info.tail != InfSign::Pos {
                    return None;
                }
                polynomial_growth_info(ctx, lhs, var)?;
                return scale_infinity(ctx, &(-info.coeff), InfSign::Pos);
            }
            None
        }
        Expr::Div(num, den) => {
            if let Some(den_info) = nonzero_scaled_linear_exp_tail_info(ctx, den, var, approach) {
                if den_info.tail == InfSign::Pos {
                    polynomial_or_numeric_tail_sign(ctx, num, var, approach)?;
                    return Some(ctx.num(0));
                }

                if let Some(num_sign) = polynomial_or_numeric_tail_sign(ctx, num, var, approach) {
                    return scale_infinity(
                        ctx,
                        &(BigRational::from_integer(BigInt::from(1)) / den_info.coeff),
                        num_sign,
                    );
                }
            }

            if let Some(num_info) = nonzero_scaled_linear_exp_tail_info(ctx, num, var, approach) {
                if num_info.tail != InfSign::Pos {
                    return None;
                }
                let den_sign = polynomial_or_numeric_tail_sign(ctx, den, var, approach)?;
                return scale_infinity(ctx, &num_info.coeff, den_sign);
            }
            None
        }
        Expr::Mul(lhs, rhs) => {
            if let Some(info) = scaled_linear_exp_tail_info(ctx, lhs, var, approach) {
                if info.coeff.is_zero() || info.tail == InfSign::Neg {
                    polynomial_growth_info(ctx, rhs, var)?;
                    return Some(ctx.num(0));
                }
            }
            if let Some(info) = scaled_linear_exp_tail_info(ctx, rhs, var, approach) {
                if info.coeff.is_zero() || info.tail == InfSign::Neg {
                    polynomial_growth_info(ctx, lhs, var)?;
                    return Some(ctx.num(0));
                }
            }
            None
        }
        _ => None,
    }
}

/// Dominance rule for `ln(linear)`/`sqrt(linear)` against polynomial growth.
///
/// The subpolynomial side is accepted only when its linear argument tends to
/// `+∞`, which makes the real-domain tail explicit. Nonlinear arguments and
/// subpolynomial-vs-subpolynomial comparisons remain residual.
pub fn subpolynomial_polynomial_dominance_limit_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    match ctx.get(expr).clone() {
        Expr::Add(lhs, rhs) => {
            if scaled_linear_subpolynomial_tail_info(ctx, lhs, var, approach).is_some() {
                let growth = polynomial_growth_info(ctx, rhs, var)?;
                let sign = limit_growth_sign(&growth.leading_coeff, growth.degree, approach);
                return Some(mk_infinity(ctx, sign));
            }
            if scaled_linear_subpolynomial_tail_info(ctx, rhs, var, approach).is_some() {
                let growth = polynomial_growth_info(ctx, lhs, var)?;
                let sign = limit_growth_sign(&growth.leading_coeff, growth.degree, approach);
                return Some(mk_infinity(ctx, sign));
            }
            None
        }
        Expr::Sub(lhs, rhs) => {
            if scaled_linear_subpolynomial_tail_info(ctx, lhs, var, approach).is_some() {
                let growth = polynomial_growth_info(ctx, rhs, var)?;
                let sign = limit_growth_sign(&growth.leading_coeff, growth.degree, approach);
                return Some(mk_infinity(ctx, neg_inf_sign(sign)));
            }
            if scaled_linear_subpolynomial_tail_info(ctx, rhs, var, approach).is_some() {
                let growth = polynomial_growth_info(ctx, lhs, var)?;
                let sign = limit_growth_sign(&growth.leading_coeff, growth.degree, approach);
                return Some(mk_infinity(ctx, sign));
            }
            None
        }
        Expr::Div(num, den) => {
            if scaled_linear_subpolynomial_tail_info(ctx, num, var, approach).is_some() {
                polynomial_growth_info(ctx, den, var)?;
                return Some(ctx.num(0));
            }

            if let Some(den_info) =
                nonzero_scaled_linear_subpolynomial_tail_info(ctx, den, var, approach)
            {
                let growth = polynomial_growth_info(ctx, num, var)?;
                let num_sign = limit_growth_sign(&growth.leading_coeff, growth.degree, approach);
                return scale_infinity(
                    ctx,
                    &(BigRational::from_integer(BigInt::from(1)) / den_info.coeff),
                    num_sign,
                );
            }
            None
        }
        _ => None,
    }
}

fn scaled_exp_subpoly_product_limit(
    ctx: &mut Context,
    exp_info: ScaledLinearExpTailInfo,
    subpoly_info: ScaledSubpolynomialTailInfo,
) -> Option<ExprId> {
    if exp_info.coeff.is_zero() || subpoly_info.coeff.is_zero() || exp_info.tail == InfSign::Neg {
        return Some(ctx.num(0));
    }

    scale_infinity(ctx, &(exp_info.coeff * subpoly_info.coeff), InfSign::Pos)
}

/// Dominance rule for linear-argument exponentials against `ln/sqrt(linear)`.
///
/// Both sides are accepted only through existing tail analyzers: exponential
/// exponents must be linear, and logarithm/radical arguments must tend to
/// `+∞` on the real tail. Nonlinear exponentials such as `exp(x^2)` remain
/// residual.
pub fn exponential_subpolynomial_dominance_limit_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    match ctx.get(expr).clone() {
        Expr::Add(lhs, rhs) => {
            if let Some(exp_info) = nonzero_scaled_linear_exp_tail_info(ctx, lhs, var, approach) {
                if exp_info.tail == InfSign::Pos
                    && scaled_linear_subpolynomial_tail_info(ctx, rhs, var, approach).is_some()
                {
                    return scale_infinity(ctx, &exp_info.coeff, InfSign::Pos);
                }
            }
            if let Some(exp_info) = nonzero_scaled_linear_exp_tail_info(ctx, rhs, var, approach) {
                if exp_info.tail == InfSign::Pos
                    && scaled_linear_subpolynomial_tail_info(ctx, lhs, var, approach).is_some()
                {
                    return scale_infinity(ctx, &exp_info.coeff, InfSign::Pos);
                }
            }
            None
        }
        Expr::Sub(lhs, rhs) => {
            if let Some(exp_info) = nonzero_scaled_linear_exp_tail_info(ctx, lhs, var, approach) {
                if exp_info.tail == InfSign::Pos
                    && scaled_linear_subpolynomial_tail_info(ctx, rhs, var, approach).is_some()
                {
                    return scale_infinity(ctx, &exp_info.coeff, InfSign::Pos);
                }
            }
            if let Some(exp_info) = nonzero_scaled_linear_exp_tail_info(ctx, rhs, var, approach) {
                if exp_info.tail == InfSign::Pos
                    && scaled_linear_subpolynomial_tail_info(ctx, lhs, var, approach).is_some()
                {
                    return scale_infinity(ctx, &(-exp_info.coeff), InfSign::Pos);
                }
            }
            None
        }
        Expr::Mul(lhs, rhs) => {
            if let Some(exp_info) = scaled_linear_exp_tail_info(ctx, lhs, var, approach) {
                if let Some(subpoly_info) =
                    scaled_linear_subpolynomial_tail_info(ctx, rhs, var, approach)
                {
                    return scaled_exp_subpoly_product_limit(ctx, exp_info, subpoly_info);
                }
            }
            if let Some(exp_info) = scaled_linear_exp_tail_info(ctx, rhs, var, approach) {
                if let Some(subpoly_info) =
                    scaled_linear_subpolynomial_tail_info(ctx, lhs, var, approach)
                {
                    return scaled_exp_subpoly_product_limit(ctx, exp_info, subpoly_info);
                }
            }
            None
        }
        Expr::Div(num, den) => {
            if let Some(den_info) = nonzero_scaled_linear_exp_tail_info(ctx, den, var, approach) {
                if let Some(num_info) =
                    scaled_linear_subpolynomial_tail_info(ctx, num, var, approach)
                {
                    let Some(num_sign) = subpolynomial_tail_sign(&num_info) else {
                        return Some(ctx.num(0));
                    };
                    if den_info.tail == InfSign::Pos {
                        return Some(ctx.num(0));
                    }
                    return scale_infinity(
                        ctx,
                        &(BigRational::from_integer(BigInt::from(1)) / den_info.coeff),
                        num_sign,
                    );
                }
            }

            if let Some(num_info) = scaled_linear_exp_tail_info(ctx, num, var, approach) {
                if let Some(den_info) =
                    nonzero_scaled_linear_subpolynomial_tail_info(ctx, den, var, approach)
                {
                    if num_info.coeff.is_zero() || num_info.tail == InfSign::Neg {
                        return Some(ctx.num(0));
                    }
                    return scale_infinity(ctx, &(num_info.coeff / den_info.coeff), InfSign::Pos);
                }
            }
            None
        }
        _ => None,
    }
}

/// Polynomial limit rule for `P(x)` as `x -> ±∞`.
///
/// This handles polynomial expressions whose leading coefficient in `var` is a
/// numeric constant. Symbolic leading coefficients remain unresolved because
/// their sign is not known under the conservative limit policy.
pub fn polynomial_limit_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    use crate::multipoly::{multipoly_from_expr, PolyBudget};

    let Expr::Variable(var_sym_id) = ctx.get(var).clone() else {
        return None;
    };
    let var_name = ctx.sym_name(var_sym_id);

    let budget = PolyBudget {
        max_terms: 100,
        max_total_degree: 20,
        max_pow_exp: 4,
    };

    let poly = multipoly_from_expr(ctx, expr, &budget).ok()?;
    if poly.is_zero() {
        return Some(ctx.num(0));
    }

    let var_idx = poly.var_index(var_name)?;
    let degree = poly.degree_in(var_idx);
    if degree == 0 {
        return None;
    }

    let leading_coeff = poly.leading_coeff_in(var_idx);
    let leading_value = leading_coeff.constant_value()?;
    let sign = limit_growth_sign(&leading_value, degree, approach);
    Some(mk_infinity(ctx, sign))
}

/// Rational polynomial limit rule for `P(x)/Q(x)` as `x -> ±∞`.
///
/// Compares polynomial degrees in `var`:
/// - `deg(P) < deg(Q) -> 0`
/// - `deg(P) = deg(Q) -> lc(P)/lc(Q)` when both leading coefficients are numeric
/// - `deg(P) > deg(Q) -> ±∞` according to leading coefficient sign and parity
pub fn rational_poly_limit(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    use crate::multipoly::{multipoly_from_expr, PolyBudget};

    // Match Div(num, den)
    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };

    // Get variable name
    let Expr::Variable(var_sym_id) = ctx.get(var).clone() else {
        return None;
    };
    let var_name = ctx.sym_name(var_sym_id);

    // Conservative budget for polynomial conversion
    let budget = PolyBudget {
        max_terms: 100,
        max_total_degree: 20,
        max_pow_exp: 4,
    };

    // Convert numerator and denominator to polynomials
    let p_num = multipoly_from_expr(ctx, num, &budget).ok()?;
    let p_den = multipoly_from_expr(ctx, den, &budget).ok()?;

    // Get variable index in polynomial
    // If var not in poly, it's constant wrt var (degree 0)
    let var_idx_num = p_num.var_index(var_name);
    let var_idx_den = p_den.var_index(var_name);

    // If neither contains the variable, constant rule handles it
    if var_idx_num.is_none() && var_idx_den.is_none() {
        return None; // Let constant rule handle it
    }

    // Check for zero denominator polynomial
    if p_den.is_zero() {
        return None; // Division by zero - don't handle here
    }

    // Get degrees
    let deg_p = var_idx_num.map(|idx| p_num.degree_in(idx)).unwrap_or(0);
    let deg_q = var_idx_den.map(|idx| p_den.degree_in(idx)).unwrap_or(0);

    // Get leading coefficients
    let lc_p = var_idx_num
        .map(|idx| p_num.leading_coeff_in(idx))
        .unwrap_or_else(|| p_num.clone());
    let lc_q = var_idx_den
        .map(|idx| p_den.leading_coeff_in(idx))
        .unwrap_or_else(|| p_den.clone());

    // Both leading coefficients must be numeric constants
    let lc_p_val = lc_p.constant_value()?;
    let lc_q_val = lc_q.constant_value()?;

    // Case 1: deg(P) < deg(Q) -> 0
    if deg_p < deg_q {
        return Some(ctx.add(Expr::Number(BigRational::from_integer(BigInt::from(0)))));
    }

    // Case 2: deg(P) = deg(Q) -> lc(P)/lc(Q)
    if deg_p == deg_q {
        let ratio = &lc_p_val / &lc_q_val;
        return Some(ctx.add(Expr::Number(ratio)));
    }

    // Case 3: deg(P) > deg(Q) -> ±∞
    // Sign = sign(lc_p/lc_q) * sign(x^k) where k = deg_p - deg_q
    let k = deg_p - deg_q;
    let ratio = &lc_p_val / &lc_q_val;
    let sign = limit_growth_sign(&ratio, k, approach);
    Some(mk_infinity(ctx, sign))
}

/// Try all limit-at-infinity rules in conservative order.
///
/// Order:
/// 1. Constant
/// 2. Variable
/// 3. Power
/// 4. Reciprocal power
/// 5. Elementary exact-argument functions
/// 6. Additive combination
/// 7. Determinate multiplicative combination
/// 8. Bounded trig over divergent denominator
/// 9. Square-root polynomial ratio with matching growth
/// 10. Polynomial over square-root polynomial with matching growth
/// 11. Exact exponential-vs-polynomial dominance
/// 12. Exact subpolynomial-vs-polynomial dominance
/// 13. Exact exponential-vs-subpolynomial dominance
/// 14. Polynomial
/// 15. Rational polynomial
pub fn try_limit_rules_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    if let Some(r) = apply_constant_rule(ctx, expr, var) {
        return Some(r);
    }
    if let Some(r) = apply_variable_rule(ctx, expr, var, approach) {
        return Some(r);
    }
    if let Some(r) = apply_power_rule(ctx, expr, var, approach) {
        return Some(r);
    }
    if let Some(r) = apply_reciprocal_power_rule(ctx, expr, var) {
        return Some(r);
    }
    if let Some(r) = elementary_function_limit_at_infinity(ctx, expr, var, approach) {
        return Some(r);
    }
    if let Some(r) = additive_limit_at_infinity(ctx, expr, var, approach) {
        return Some(r);
    }
    if let Some(r) = multiplicative_limit_at_infinity(ctx, expr, var, approach) {
        return Some(r);
    }
    if let Some(r) = bounded_elementary_over_divergent_limit_at_infinity(ctx, expr, var, approach) {
        return Some(r);
    }
    if let Some(r) =
        bounded_elementary_times_decaying_exp_limit_at_infinity(ctx, expr, var, approach)
    {
        return Some(r);
    }
    if let Some(r) = sqrt_polynomial_ratio_limit_at_infinity(ctx, expr, var, approach) {
        return Some(r);
    }
    if let Some(r) = polynomial_sqrt_ratio_limit_at_infinity(ctx, expr, var, approach) {
        return Some(r);
    }
    if let Some(r) = exponential_polynomial_dominance_limit_at_infinity(ctx, expr, var, approach) {
        return Some(r);
    }
    if let Some(r) = subpolynomial_polynomial_dominance_limit_at_infinity(ctx, expr, var, approach)
    {
        return Some(r);
    }
    if let Some(r) = exponential_subpolynomial_dominance_limit_at_infinity(ctx, expr, var, approach)
    {
        return Some(r);
    }
    if let Some(r) = polynomial_limit_at_infinity(ctx, expr, var, approach) {
        return Some(r);
    }
    rational_poly_limit(ctx, expr, var, approach)
}

/// Evaluate a limit at infinity using conservative rules.
///
/// This runs optional safe pre-simplification, applies direct rules in order,
/// and otherwise returns a residual `limit(...)` expression.
pub fn eval_limit_at_infinity(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: Approach,
    opts: &LimitOptions,
) -> LimitEvalOutcome {
    let simplified_expr = match opts.presimplify {
        PreSimplifyMode::Off => expr,
        PreSimplifyMode::Safe => presimplify_safe_for_limit(ctx, expr),
    };

    if let Approach::Finite(point) = approach {
        if let Some(result_expr) = try_limit_rules_at_finite(ctx, simplified_expr, var, point) {
            return LimitEvalOutcome {
                expr: result_expr,
                warning: None,
            };
        }
    }

    if let Some(sign) = approach.inf_sign() {
        if let Some(result_expr) = try_limit_rules_at_infinity(ctx, simplified_expr, var, sign) {
            return LimitEvalOutcome {
                expr: result_expr,
                warning: None,
            };
        }
    }

    let residual = mk_limit_for_approach(ctx, simplified_expr, var, approach);
    let warning = match approach {
        Approach::Finite(_) => "Finite point limits are not supported safely yet".to_string(),
        Approach::PosInfinity | Approach::NegInfinity => {
            "Could not determine limit safely".to_string()
        }
    };
    LimitEvalOutcome {
        expr: residual,
        warning: Some(warning),
    }
}

const PRESIMPLIFY_MAX_DEPTH: usize = 500;

fn expr_is_zero(ctx: &Context, expr: ExprId) -> bool {
    use num_traits::Zero;
    matches!(ctx.get(expr), Expr::Number(n) if n.is_zero())
}

fn expr_is_one(ctx: &Context, expr: ExprId) -> bool {
    use num_traits::One;
    matches!(ctx.get(expr), Expr::Number(n) if n.is_one())
}

fn apply_safe_add_rules(ctx: &mut Context, a: ExprId, b: ExprId) -> ExprId {
    if expr_is_zero(ctx, b) {
        return a;
    }
    if expr_is_zero(ctx, a) {
        return b;
    }

    if let Expr::Neg(neg_inner) = ctx.get(b) {
        if *neg_inner == a {
            return ctx.num(0);
        }
    }
    if let Expr::Neg(neg_inner) = ctx.get(a) {
        if *neg_inner == b {
            return ctx.num(0);
        }
    }

    ctx.add(Expr::Add(a, b))
}

fn apply_safe_sub_rules(ctx: &mut Context, a: ExprId, b: ExprId) -> ExprId {
    if expr_is_zero(ctx, b) {
        return a;
    }
    if a == b {
        return ctx.num(0);
    }
    ctx.add(Expr::Sub(a, b))
}

fn apply_safe_mul_rules(ctx: &mut Context, a: ExprId, b: ExprId) -> ExprId {
    if expr_is_zero(ctx, a) || expr_is_zero(ctx, b) {
        return ctx.num(0);
    }
    if expr_is_one(ctx, b) {
        return a;
    }
    if expr_is_one(ctx, a) {
        return b;
    }
    ctx.add(Expr::Mul(a, b))
}

fn presimplify_recursive(ctx: &mut Context, expr: ExprId, depth: usize) -> ExprId {
    if depth > PRESIMPLIFY_MAX_DEPTH {
        return expr;
    }

    match ctx.get(expr).clone() {
        Expr::Add(a, b) => {
            let a2 = presimplify_recursive(ctx, a, depth + 1);
            let b2 = presimplify_recursive(ctx, b, depth + 1);
            apply_safe_add_rules(ctx, a2, b2)
        }
        Expr::Sub(a, b) => {
            let a2 = presimplify_recursive(ctx, a, depth + 1);
            let b2 = presimplify_recursive(ctx, b, depth + 1);
            apply_safe_sub_rules(ctx, a2, b2)
        }
        Expr::Mul(a, b) => {
            let a2 = presimplify_recursive(ctx, a, depth + 1);
            let b2 = presimplify_recursive(ctx, b, depth + 1);
            apply_safe_mul_rules(ctx, a2, b2)
        }
        Expr::Neg(a) => {
            let a2 = presimplify_recursive(ctx, a, depth + 1);
            if let Expr::Neg(inner) = ctx.get(a2) {
                return *inner;
            }
            ctx.add(Expr::Neg(a2))
        }
        Expr::Div(num, den) => {
            let num2 = presimplify_recursive(ctx, num, depth + 1);
            let den2 = presimplify_recursive(ctx, den, depth + 1);
            ctx.add(Expr::Div(num2, den2))
        }
        Expr::Pow(base, exp) => {
            let base2 = presimplify_recursive(ctx, base, depth + 1);
            let exp2 = presimplify_recursive(ctx, exp, depth + 1);
            ctx.add(Expr::Pow(base2, exp2))
        }
        Expr::Function(name, args) => {
            let mut new_args = Vec::with_capacity(args.len());
            for arg in args {
                new_args.push(presimplify_recursive(ctx, arg, depth + 1));
            }
            ctx.add(Expr::Function(name, new_args))
        }
        Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) => expr,
        Expr::Hold(inner) => {
            let inner2 = presimplify_recursive(ctx, inner, depth + 1);
            ctx.add(Expr::Hold(inner2))
        }
        Expr::Matrix { .. } | Expr::SessionRef(_) => expr,
    }
}

/// Safe pre-simplification for limit evaluation.
///
/// This is an allowlist-only pass and intentionally excludes transforms that
/// require domain assumptions (for example, `a/a -> 1` or `a^0 -> 1`).
pub fn presimplify_safe_for_limit(ctx: &mut Context, expr: ExprId) -> ExprId {
    presimplify_recursive(ctx, expr, 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    fn parse_expr(ctx: &mut Context, s: &str) -> ExprId {
        parse(s, ctx).expect("parse failed")
    }

    fn display_expr(ctx: &Context, expr: ExprId) -> String {
        DisplayExpr {
            context: ctx,
            id: expr,
        }
        .to_string()
    }

    #[test]
    fn depends_on_detects_simple_variable() {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, "x + 1");
        let x = parse_expr(&mut ctx, "x");
        assert!(depends_on(&ctx, expr, x));
    }

    #[test]
    fn depends_on_rejects_constant_expression() {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, "5 + pi");
        let x = parse_expr(&mut ctx, "x");
        assert!(!depends_on(&ctx, expr, x));
    }

    #[test]
    fn parse_pow_int_extracts_integer_exponent() {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, "x^3");
        let (_, n) = parse_pow_int(&ctx, expr).expect("power");
        assert_eq!(n, 3);
    }

    #[test]
    fn limit_sign_handles_neg_infinity_parity() {
        assert_eq!(limit_sign(InfSign::Pos, 7), InfSign::Pos);
        assert_eq!(limit_sign(InfSign::Neg, 2), InfSign::Pos);
        assert_eq!(limit_sign(InfSign::Neg, 3), InfSign::Neg);
    }

    #[test]
    fn mk_limit_builds_limit_call_with_signed_infinity_symbol() {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, "x^2");
        let var = parse_expr(&mut ctx, "x");
        let lim = mk_limit(&mut ctx, expr, var, InfSign::Neg);

        let Expr::Function(_fn_id, args) = ctx.get(lim) else {
            panic!("expected limit function call");
        };
        assert_eq!(args.len(), 3);
        assert_eq!(args[0], expr);
        assert_eq!(args[1], var);

        let approach = args[2];
        match ctx.get(approach) {
            Expr::Neg(inner) => {
                assert!(matches!(
                    ctx.get(*inner),
                    Expr::Constant(Constant::Infinity)
                ));
            }
            _ => panic!("expected negative infinity argument"),
        }
    }

    #[test]
    fn finite_elementary_polynomial_limit_handles_total_real_functions() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let point = parse_expr(&mut ctx, "-1");

        let cases = [
            ("exp(x^2 + 1)", BuiltinFn::Exp),
            ("sin(x^2 + 1)", BuiltinFn::Sin),
            ("cos(x^2 + 1)", BuiltinFn::Cos),
            ("sinh(x^2 + 1)", BuiltinFn::Sinh),
            ("cosh(x^2 + 1)", BuiltinFn::Cosh),
            ("tanh(x^2 + 1)", BuiltinFn::Tanh),
            ("atan(x^2 + 1)", BuiltinFn::Atan),
            ("arctan(x^2 + 1)", BuiltinFn::Arctan),
            ("asinh(x^2 + 1)", BuiltinFn::Asinh),
            ("cbrt(x^2 + 1)", BuiltinFn::Cbrt),
        ];

        for (input, expected_builtin) in cases {
            let expr = parse_expr(&mut ctx, input);
            let out = apply_finite_elementary_polynomial_rule(&mut ctx, expr, x, point)
                .unwrap_or_else(|| panic!("expected finite elementary limit for {input}"));

            let Expr::Function(fn_id, args) = ctx.get(out).clone() else {
                panic!("expected function output for {input}");
            };
            assert_eq!(ctx.builtin_of(fn_id), Some(expected_builtin));
            assert_eq!(args.len(), 1);

            let Expr::Number(value) = ctx.get(args[0]) else {
                panic!("expected numeric function argument for {input}");
            };
            assert_eq!(value, &BigRational::from_integer(2.into()));
        }
    }

    #[test]
    fn finite_elementary_polynomial_limit_evaluates_zero_special_values() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let point = parse_expr(&mut ctx, "0");

        let cases = [
            ("exp(x)", 1),
            ("sin(x)", 0),
            ("cos(x)", 1),
            ("sinh(x)", 0),
            ("cosh(x)", 1),
            ("tanh(x)", 0),
            ("atan(x)", 0),
            ("arctan(x)", 0),
            ("asinh(x)", 0),
            ("cbrt(x)", 0),
            ("abs(x)", 0),
        ];

        for (input, expected) in cases {
            let expr = parse_expr(&mut ctx, input);
            let out = apply_finite_elementary_polynomial_rule(&mut ctx, expr, x, point)
                .unwrap_or_else(|| panic!("expected finite elementary limit for {input}"));

            let Expr::Number(value) = ctx.get(out) else {
                panic!("expected numeric special value for {input}");
            };
            assert_eq!(value, &BigRational::from_integer(expected.into()));
        }
    }

    #[test]
    fn finite_abs_polynomial_limit_evaluates_exact_rational_absolute_value() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let point = parse_expr(&mut ctx, "0");

        let cases = [("abs(x^2 - 1)", 1), ("abs(x - 2)", 2)];

        for (input, expected) in cases {
            let expr = parse_expr(&mut ctx, input);
            let out = apply_finite_elementary_polynomial_rule(&mut ctx, expr, x, point)
                .unwrap_or_else(|| panic!("expected finite abs polynomial limit for {input}"));

            let Expr::Number(value) = ctx.get(out) else {
                panic!("expected numeric absolute value for {input}");
            };
            assert_eq!(value, &BigRational::from_integer(expected.into()));
        }
    }

    #[test]
    fn finite_real_cube_root_limit_evaluates_exact_and_symbolic_values() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");

        let point_neg_two = parse_expr(&mut ctx, "-2");
        let exact_builtin = parse_expr(&mut ctx, "cbrt(x^3)");
        let exact_builtin_out =
            try_limit_rules_at_finite(&mut ctx, exact_builtin, x, point_neg_two)
                .expect("expected exact finite cbrt limit");
        let Expr::Number(value) = ctx.get(exact_builtin_out) else {
            panic!("expected exact cbrt limit to collapse to a number");
        };
        assert_eq!(value, &BigRational::from_integer((-2).into()));

        let point_one = parse_expr(&mut ctx, "1");
        let exact_power = parse_expr(&mut ctx, "(x^2 - 9)^(1/3)");
        let exact_power_out = try_limit_rules_at_finite(&mut ctx, exact_power, x, point_one)
            .expect("expected exact finite one-third power limit");
        let Expr::Number(value) = ctx.get(exact_power_out) else {
            panic!("expected exact one-third power limit to collapse to a number");
        };
        assert_eq!(value, &BigRational::from_integer((-2).into()));

        let point_neg_one = parse_expr(&mut ctx, "-1");
        let symbolic_builtin = parse_expr(&mut ctx, "cbrt(x^2 + 1)");
        let symbolic_builtin_out =
            try_limit_rules_at_finite(&mut ctx, symbolic_builtin, x, point_neg_one)
                .expect("expected symbolic finite cbrt limit");
        assert_eq!(display_expr(&ctx, symbolic_builtin_out), "cbrt(2)");
    }

    #[test]
    fn finite_total_real_unary_composition_limit_reuses_resolved_sublimits() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");

        let point_zero = parse_expr(&mut ctx, "0");
        let nested_trig = parse_expr(&mut ctx, "cos(sin(x))");
        let nested_trig_out = try_limit_rules_at_finite(&mut ctx, nested_trig, x, point_zero)
            .expect("expected nested total-real trig finite limit");
        let Expr::Number(value) = ctx.get(nested_trig_out) else {
            panic!("expected cos(sin(x)) at 0 to collapse to a number");
        };
        assert_eq!(value, &BigRational::from_integer(1.into()));

        let point_neg_two = parse_expr(&mut ctx, "-2");
        let sin_sqrt = parse_expr(&mut ctx, "sin(sqrt(x^2 + 1))");
        let sin_sqrt_out = try_limit_rules_at_finite(&mut ctx, sin_sqrt, x, point_neg_two)
            .expect("expected total-real unary composition over safe sqrt sublimit");
        assert_eq!(display_expr(&ctx, sin_sqrt_out), "sin(sqrt(5))");

        let exp_abs = parse_expr(&mut ctx, "exp(abs(x))");
        let exp_abs_out = try_limit_rules_at_finite(&mut ctx, exp_abs, x, point_neg_two)
            .expect("expected exp over resolved abs sublimit");
        let Expr::Function(fn_id, args) = ctx.get(exp_abs_out).clone() else {
            panic!("expected exp(abs(x)) finite limit to remain an exp function");
        };
        assert_eq!(ctx.builtin_of(fn_id), Some(BuiltinFn::Exp));
        assert_eq!(args.len(), 1);
        let Expr::Number(value) = ctx.get(args[0]) else {
            panic!("expected exp argument to be exact numeric absolute value");
        };
        assert_eq!(value, &BigRational::from_integer(2.into()));

        let point_eight = parse_expr(&mut ctx, "8");
        let sin_cbrt = parse_expr(&mut ctx, "sin(cbrt(x))");
        let sin_cbrt_out = try_limit_rules_at_finite(&mut ctx, sin_cbrt, x, point_eight)
            .expect("expected total-real unary composition over exact cbrt sublimit");
        assert_eq!(display_expr(&ctx, sin_cbrt_out), "sin(2)");
    }

    #[test]
    fn finite_arithmetic_composition_folds_safe_numeric_and_structural_results() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let point_neg_two = parse_expr(&mut ctx, "-2");

        let numeric_sum = parse_expr(&mut ctx, "abs(x) + 1");
        let numeric_sum_out = try_limit_rules_at_finite(&mut ctx, numeric_sum, x, point_neg_two)
            .expect("expected safe numeric finite sum");
        let Expr::Number(value) = ctx.get(numeric_sum_out) else {
            panic!("expected exact numeric finite sum");
        };
        assert_eq!(value, &BigRational::from_integer(3.into()));

        let structural_zero = parse_expr(&mut ctx, "sqrt(x^2 + 1) - sqrt(x^2 + 1)");
        let structural_zero_out =
            try_limit_rules_at_finite(&mut ctx, structural_zero, x, point_neg_two)
                .expect("expected safe structural zero finite difference");
        let Expr::Number(value) = ctx.get(structural_zero_out) else {
            panic!("expected structural zero finite difference to fold");
        };
        assert_eq!(value, &BigRational::zero());

        let zero_quotient = parse_expr(&mut ctx, "(sqrt(x^2 + 1) - sqrt(x^2 + 1))/(abs(x) + 1)");
        let zero_quotient_out =
            try_limit_rules_at_finite(&mut ctx, zero_quotient, x, point_neg_two)
                .expect("expected safe zero quotient finite limit");
        let Expr::Number(value) = ctx.get(zero_quotient_out) else {
            panic!("expected safe zero quotient to fold");
        };
        assert_eq!(value, &BigRational::zero());

        let symbolic_sum = parse_expr(&mut ctx, "sqrt(x^2 + 1) + ln(x + 5)");
        let symbolic_sum_out = try_limit_rules_at_finite(&mut ctx, symbolic_sum, x, point_neg_two)
            .expect("expected safe symbolic finite sum");
        assert_eq!(display_expr(&ctx, symbolic_sum_out), "ln(3) + sqrt(5)");

        let unsafe_zero_product = parse_expr(&mut ctx, "0 * sqrt(x)");
        let point_zero = parse_expr(&mut ctx, "0");
        assert!(
            try_limit_rules_at_finite(&mut ctx, unsafe_zero_product, x, point_zero).is_none(),
            "zero product must not hide an unresolved finite sublimit"
        );
    }

    #[test]
    fn finite_positive_domain_unary_composition_requires_positive_sublimit() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let point_neg_two = parse_expr(&mut ctx, "-2");

        let ln_sqrt = parse_expr(&mut ctx, "ln(sqrt(x^2 + 1))");
        let ln_sqrt_out = try_limit_rules_at_finite(&mut ctx, ln_sqrt, x, point_neg_two)
            .expect("expected ln over proven-positive sqrt sublimit");
        assert_eq!(display_expr(&ctx, ln_sqrt_out), "ln(sqrt(5))");

        let sqrt_abs_shift = parse_expr(&mut ctx, "sqrt(abs(x) + 1)");
        let sqrt_abs_shift_out =
            try_limit_rules_at_finite(&mut ctx, sqrt_abs_shift, x, point_neg_two)
                .expect("expected sqrt over positive arithmetic sublimit");
        assert_eq!(display_expr(&ctx, sqrt_abs_shift_out), "sqrt(3)");

        let ln_abs = parse_expr(&mut ctx, "ln(abs(x))");
        let ln_abs_out = try_limit_rules_at_finite(&mut ctx, ln_abs, x, point_neg_two)
            .expect("expected ln over positive abs sublimit");
        assert_eq!(display_expr(&ctx, ln_abs_out), "ln(2)");

        let log2_poly = parse_expr(&mut ctx, "log2(x^2 + 1)");
        let log2_poly_out = try_limit_rules_at_finite(&mut ctx, log2_poly, x, point_neg_two)
            .expect("expected log2 over positive polynomial argument");
        assert_eq!(display_expr(&ctx, log2_poly_out), "log2(5)");

        let log10_sqrt = parse_expr(&mut ctx, "log10(sqrt(x^2 + 1))");
        let log10_sqrt_out = try_limit_rules_at_finite(&mut ctx, log10_sqrt, x, point_neg_two)
            .expect("expected log10 over proven-positive sqrt sublimit");
        assert_eq!(display_expr(&ctx, log10_sqrt_out), "log10(sqrt(5))");

        let log2_abs = parse_expr(&mut ctx, "log2(abs(x))");
        let log2_abs_out = try_limit_rules_at_finite(&mut ctx, log2_abs, x, point_neg_two)
            .expect("expected log2 over positive abs sublimit");
        assert_eq!(display_expr(&ctx, log2_abs_out), "log2(2)");

        let point_zero = parse_expr(&mut ctx, "0");
        let sqrt_perfect_square_poly = parse_expr(&mut ctx, "sqrt(x^2 + 4*x + 4)");
        let sqrt_perfect_square_poly_out =
            try_limit_rules_at_finite(&mut ctx, sqrt_perfect_square_poly, x, point_zero)
                .expect("expected exact sqrt over positive rational square sublimit");
        assert_eq!(display_expr(&ctx, sqrt_perfect_square_poly_out), "2");

        let ln_one = parse_expr(&mut ctx, "ln(x^2 + 1)");
        let ln_one_out = try_limit_rules_at_finite(&mut ctx, ln_one, x, point_zero)
            .expect("expected exact ln(1) finite limit");
        assert_eq!(display_expr(&ctx, ln_one_out), "0");

        let log2_one = parse_expr(&mut ctx, "log2(x^2 + 1)");
        let log2_one_out = try_limit_rules_at_finite(&mut ctx, log2_one, x, point_zero)
            .expect("expected exact log2(1) finite limit");
        assert_eq!(display_expr(&ctx, log2_one_out), "0");

        let log10_one = parse_expr(&mut ctx, "log10(x^2 + 1)");
        let log10_one_out = try_limit_rules_at_finite(&mut ctx, log10_one, x, point_zero)
            .expect("expected exact log10(1) finite limit");
        assert_eq!(display_expr(&ctx, log10_one_out), "0");

        let exp_ln_abs = parse_expr(&mut ctx, "exp(ln(abs(x)))");
        let exp_ln_abs_out = try_limit_rules_at_finite(&mut ctx, exp_ln_abs, x, point_neg_two)
            .expect("expected exact exp(ln(g)) finite limit when g is positive");
        assert_eq!(display_expr(&ctx, exp_ln_abs_out), "2");

        let ln_exp_abs = parse_expr(&mut ctx, "ln(exp(abs(x)))");
        let ln_exp_abs_out = try_limit_rules_at_finite(&mut ctx, ln_exp_abs, x, point_neg_two)
            .expect("expected exact ln(exp(g)) finite limit");
        assert_eq!(display_expr(&ctx, ln_exp_abs_out), "2");

        let abs_sqrt = parse_expr(&mut ctx, "abs(sqrt(x^2 + 1))");
        let abs_sqrt_out = try_limit_rules_at_finite(&mut ctx, abs_sqrt, x, point_neg_two)
            .expect("expected exact abs over positive sqrt finite limit");
        assert_eq!(display_expr(&ctx, abs_sqrt_out), "sqrt(5)");

        let abs_neg_sqrt = parse_expr(&mut ctx, "abs(-sqrt(x^2 + 1))");
        let abs_neg_sqrt_out = try_limit_rules_at_finite(&mut ctx, abs_neg_sqrt, x, point_neg_two)
            .expect("expected exact abs over negative positive-sqrt finite limit");
        assert_eq!(display_expr(&ctx, abs_neg_sqrt_out), "sqrt(5)");

        let exp_ln_abs_zero = parse_expr(&mut ctx, "exp(ln(abs(x)))");
        assert!(
            try_limit_rules_at_finite(&mut ctx, exp_ln_abs_zero, x, point_zero).is_none(),
            "exp(ln(abs(x))) at zero must remain residual"
        );

        let sqrt_abs_zero = parse_expr(&mut ctx, "sqrt(abs(x))");
        assert!(
            try_limit_rules_at_finite(&mut ctx, sqrt_abs_zero, x, point_zero).is_none(),
            "sqrt over zero sublimit must remain residual"
        );

        let ln_sin_zero = parse_expr(&mut ctx, "ln(sin(x))");
        assert!(
            try_limit_rules_at_finite(&mut ctx, ln_sin_zero, x, point_zero).is_none(),
            "ln over zero sublimit must remain residual"
        );

        let log10_abs_zero = parse_expr(&mut ctx, "log10(abs(x))");
        assert!(
            try_limit_rules_at_finite(&mut ctx, log10_abs_zero, x, point_zero).is_none(),
            "log10 over zero sublimit must remain residual"
        );
    }

    #[test]
    fn finite_binary_log_composition_requires_valid_base_and_positive_sublimits() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let point_neg_two = parse_expr(&mut ctx, "-2");

        let binary_log_poly = parse_expr(&mut ctx, "log(2, x^2 + 1)");
        let binary_log_poly_out =
            try_limit_rules_at_finite(&mut ctx, binary_log_poly, x, point_neg_two)
                .expect("expected constant-base log over positive polynomial argument");
        assert_eq!(display_expr(&ctx, binary_log_poly_out), "log(2, 5)");

        let binary_log_sqrt = parse_expr(&mut ctx, "log(1/2, sqrt(x^2 + 1))");
        let binary_log_sqrt_out =
            try_limit_rules_at_finite(&mut ctx, binary_log_sqrt, x, point_neg_two)
                .expect("expected constant-base log over proven-positive sqrt sublimit");
        assert_eq!(
            display_expr(&ctx, binary_log_sqrt_out),
            "log(1 / 2, sqrt(5))"
        );

        let binary_log_abs = parse_expr(&mut ctx, "log(2, abs(x))");
        let binary_log_abs_out =
            try_limit_rules_at_finite(&mut ctx, binary_log_abs, x, point_neg_two)
                .expect("expected constant-base log over positive abs sublimit");
        assert_eq!(display_expr(&ctx, binary_log_abs_out), "1");

        let point_zero = parse_expr(&mut ctx, "0");
        let binary_log_arg_one = parse_expr(&mut ctx, "log(2, x^2 + 1)");
        let binary_log_arg_one_out =
            try_limit_rules_at_finite(&mut ctx, binary_log_arg_one, x, point_zero)
                .expect("expected exact binary log of one finite limit");
        assert_eq!(display_expr(&ctx, binary_log_arg_one_out), "0");

        let variable_base_log_poly = parse_expr(&mut ctx, "log(x^2 + 3, x^2 + 1)");
        let variable_base_log_poly_out =
            try_limit_rules_at_finite(&mut ctx, variable_base_log_poly, x, point_neg_two)
                .expect("expected log over safe finite base and argument sublimits");
        assert_eq!(display_expr(&ctx, variable_base_log_poly_out), "log(7, 5)");

        let variable_base_log_sqrt = parse_expr(&mut ctx, "log(x^2 + 3, sqrt(x^2 + 1))");
        let variable_base_log_sqrt_out =
            try_limit_rules_at_finite(&mut ctx, variable_base_log_sqrt, x, point_neg_two)
                .expect("expected log over safe finite base and positive sqrt argument sublimit");
        assert_eq!(
            display_expr(&ctx, variable_base_log_sqrt_out),
            "log(7, sqrt(5))"
        );

        let point_neg_one = parse_expr(&mut ctx, "-1");
        let variable_base_log_same = parse_expr(&mut ctx, "log(x^2 + 3, x^2 + 3)");
        let variable_base_log_same_out =
            try_limit_rules_at_finite(&mut ctx, variable_base_log_same, x, point_neg_one)
                .expect("expected exact binary log with equal finite base and argument");
        assert_eq!(display_expr(&ctx, variable_base_log_same_out), "1");

        let binary_log_abs_zero = parse_expr(&mut ctx, "log(2, abs(x))");
        assert!(
            try_limit_rules_at_finite(&mut ctx, binary_log_abs_zero, x, point_zero).is_none(),
            "constant-base log over zero sublimit must remain residual"
        );

        let log_base_one = parse_expr(&mut ctx, "log(1, x^2 + 1)");
        assert!(
            try_limit_rules_at_finite(&mut ctx, log_base_one, x, point_neg_two).is_none(),
            "constant-base log with base one must remain residual"
        );

        let log_negative_base = parse_expr(&mut ctx, "log(-2, x^2 + 1)");
        assert!(
            try_limit_rules_at_finite(&mut ctx, log_negative_base, x, point_neg_two).is_none(),
            "constant-base log with negative base must remain residual"
        );

        let log_variable_base_one = parse_expr(&mut ctx, "log(x^2 - 3, x^2 + 1)");
        assert!(
            try_limit_rules_at_finite(&mut ctx, log_variable_base_one, x, point_neg_two).is_none(),
            "variable-base log with base sublimit one must remain residual"
        );

        let log_variable_base_zero = parse_expr(&mut ctx, "log(x^2 - 4, x^2 + 1)");
        assert!(
            try_limit_rules_at_finite(&mut ctx, log_variable_base_zero, x, point_neg_two).is_none(),
            "variable-base log with zero base sublimit must remain residual"
        );
    }

    #[test]
    fn finite_integer_power_composition_requires_safe_sublimit_and_nonzero_base_when_needed() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let point_neg_two = parse_expr(&mut ctx, "-2");

        let numeric_positive_power = parse_expr(&mut ctx, "(abs(x) + 1)^2");
        let numeric_positive_power_out =
            try_limit_rules_at_finite(&mut ctx, numeric_positive_power, x, point_neg_two)
                .expect("expected integer power over exact numeric sublimit");
        let Expr::Number(value) = ctx.get(numeric_positive_power_out) else {
            panic!("expected exact integer power to fold to a number");
        };
        assert_eq!(value, &BigRational::from_integer(9.into()));

        let symbolic_positive_power = parse_expr(&mut ctx, "(sqrt(x^2 + 1))^2");
        let symbolic_positive_power_out =
            try_limit_rules_at_finite(&mut ctx, symbolic_positive_power, x, point_neg_two)
                .expect("expected integer power over safe symbolic sublimit");
        let Expr::Number(value) = ctx.get(symbolic_positive_power_out) else {
            panic!("expected even power over exact sqrt sublimit to fold to a number");
        };
        assert_eq!(value, &BigRational::from_integer(5.into()));

        let symbolic_odd_power = parse_expr(&mut ctx, "(sqrt(x^2 + 1))^3");
        let symbolic_odd_power_out =
            try_limit_rules_at_finite(&mut ctx, symbolic_odd_power, x, point_neg_two)
                .expect("expected odd integer power over safe symbolic sublimit");
        assert_eq!(display_expr(&ctx, symbolic_odd_power_out), "sqrt(5)^3");

        let numeric_negative_power = parse_expr(&mut ctx, "(abs(x) + 1)^(-2)");
        let numeric_negative_power_out =
            try_limit_rules_at_finite(&mut ctx, numeric_negative_power, x, point_neg_two)
                .expect("expected negative integer power over nonzero numeric sublimit");
        let Expr::Number(value) = ctx.get(numeric_negative_power_out) else {
            panic!("expected exact negative integer power to fold to a number");
        };
        assert_eq!(value, &BigRational::new(BigInt::from(1), BigInt::from(9)));

        let symbolic_negative_power = parse_expr(&mut ctx, "(sqrt(x^2 + 1))^(-1)");
        let symbolic_negative_power_out =
            try_limit_rules_at_finite(&mut ctx, symbolic_negative_power, x, point_neg_two)
                .expect("expected negative integer power over proven nonzero symbolic sublimit");
        assert_eq!(
            display_expr(&ctx, symbolic_negative_power_out),
            "1 / sqrt(5)"
        );

        let symbolic_negative_square_power = parse_expr(&mut ctx, "(sqrt(x^2 + 1))^(-2)");
        let symbolic_negative_square_power_out =
            try_limit_rules_at_finite(&mut ctx, symbolic_negative_square_power, x, point_neg_two)
                .expect("expected negative even power over exact sqrt sublimit to fold");
        let Expr::Number(value) = ctx.get(symbolic_negative_square_power_out) else {
            panic!("expected negative even power over exact sqrt sublimit to fold to a number");
        };
        assert_eq!(value, &BigRational::new(BigInt::from(1), BigInt::from(5)));

        let point_neg_one = parse_expr(&mut ctx, "-1");
        let cbrt_cube_power = parse_expr(&mut ctx, "(cbrt(x^2 + 1))^3");
        let cbrt_cube_power_out =
            try_limit_rules_at_finite(&mut ctx, cbrt_cube_power, x, point_neg_one)
                .expect("expected cube power over exact cbrt sublimit to fold");
        let Expr::Number(value) = ctx.get(cbrt_cube_power_out) else {
            panic!("expected cube power over exact cbrt sublimit to fold to a number");
        };
        assert_eq!(value, &BigRational::from_integer(2.into()));

        let cbrt_square_power = parse_expr(&mut ctx, "(cbrt(x^2 + 1))^2");
        let cbrt_square_power_out =
            try_limit_rules_at_finite(&mut ctx, cbrt_square_power, x, point_neg_one)
                .expect("expected non-multiple cbrt power to remain explicit");
        assert_eq!(display_expr(&ctx, cbrt_square_power_out), "cbrt(2)^2");

        let cbrt_negative_cube_power = parse_expr(&mut ctx, "(cbrt(x^2 + 1))^(-3)");
        let cbrt_negative_cube_power_out =
            try_limit_rules_at_finite(&mut ctx, cbrt_negative_cube_power, x, point_neg_one)
                .expect("expected negative cube power over exact nonzero cbrt sublimit to fold");
        let Expr::Number(value) = ctx.get(cbrt_negative_cube_power_out) else {
            panic!("expected negative cube power over exact cbrt sublimit to fold to a number");
        };
        assert_eq!(value, &BigRational::new(BigInt::from(1), BigInt::from(2)));

        let cbrt_zero_power = parse_expr(&mut ctx, "(cbrt(x^2 + 1))^0");
        let cbrt_zero_power_out =
            try_limit_rules_at_finite(&mut ctx, cbrt_zero_power, x, point_neg_one)
                .expect("expected zero power over nonzero cbrt sublimit to fold");
        let Expr::Number(value) = ctx.get(cbrt_zero_power_out) else {
            panic!("expected zero power over nonzero cbrt sublimit to fold to one");
        };
        assert_eq!(value, &BigRational::one());

        let numeric_zero_power = parse_expr(&mut ctx, "(abs(x) + 1)^0");
        let numeric_zero_power_out =
            try_limit_rules_at_finite(&mut ctx, numeric_zero_power, x, point_neg_two)
                .expect("expected zero power over nonzero sublimit");
        let Expr::Number(value) = ctx.get(numeric_zero_power_out) else {
            panic!("expected safe zero power to fold to one");
        };
        assert_eq!(value, &BigRational::one());

        let point_zero = parse_expr(&mut ctx, "0");
        let zero_base_negative_power = parse_expr(&mut ctx, "(abs(x) - 2)^(-1)");
        assert!(
            try_limit_rules_at_finite(&mut ctx, zero_base_negative_power, x, point_neg_two)
                .is_none(),
            "negative integer power over zero sublimit must remain residual"
        );

        let zero_base_zero_power = parse_expr(&mut ctx, "abs(x)^0");
        assert!(
            try_limit_rules_at_finite(&mut ctx, zero_base_zero_power, x, point_zero).is_none(),
            "zero power over zero sublimit must remain residual"
        );

        let zero_cbrt_base_negative_power = parse_expr(&mut ctx, "cbrt(x)^(-3)");
        assert!(
            try_limit_rules_at_finite(&mut ctx, zero_cbrt_base_negative_power, x, point_zero)
                .is_none(),
            "negative cube power over zero cbrt sublimit must remain residual"
        );

        let unresolved_base_power = parse_expr(&mut ctx, "sqrt(x)^2");
        assert!(
            try_limit_rules_at_finite(&mut ctx, unresolved_base_power, x, point_zero).is_none(),
            "integer power must not hide an unresolved finite base sublimit"
        );
    }

    #[test]
    fn finite_total_real_unary_composition_rejects_unresolved_inner_limit() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let point = parse_expr(&mut ctx, "0");
        let expr = parse_expr(&mut ctx, "sin(sign(x))");

        assert!(
            try_limit_rules_at_finite(&mut ctx, expr, x, point).is_none(),
            "outer total-real function must not hide unresolved discontinuous inner limit"
        );
    }

    #[test]
    fn rational_poly_limit_handles_equal_and_higher_degree_cases() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");

        let equal = parse_expr(&mut ctx, "(3*x^2 + 1)/(6*x^2 - 5)");
        let higher = parse_expr(&mut ctx, "(2*x^3)/(x^2+1)");

        let equal_out = rational_poly_limit(&mut ctx, equal, x, InfSign::Pos).expect("equal");
        let higher_out = rational_poly_limit(&mut ctx, higher, x, InfSign::Neg).expect("higher");

        assert!(matches!(ctx.get(equal_out), Expr::Number(_)));
        assert!(matches!(ctx.get(higher_out), Expr::Neg(_)));
    }

    #[test]
    fn rational_poly_limit_rejects_non_polynomial_and_symbolic_leading_coeff() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let non_poly = parse_expr(&mut ctx, "sin(x)/x");
        let symbolic_lc = parse_expr(&mut ctx, "(y*x^2)/x^2");

        let out1 = rational_poly_limit(&mut ctx, non_poly, x, InfSign::Pos);
        let out2 = rational_poly_limit(&mut ctx, symbolic_lc, x, InfSign::Pos);

        assert!(out1.is_none());
        assert!(out2.is_none());
    }

    #[test]
    fn sqrt_polynomial_ratio_limit_at_infinity_handles_matching_growth() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");

        let pos = parse_expr(&mut ctx, "sqrt(x^2 + 1)/x");
        let neg = parse_expr(&mut ctx, "sqrt(x^2 + 1)/x");
        let scaled = parse_expr(&mut ctx, "sqrt(4*x^2 + 1)/(2*x)");
        let even_den = parse_expr(&mut ctx, "sqrt(x^4 + 1)/x^2");
        let irrational_coeff = parse_expr(&mut ctx, "sqrt(2*x^2 + 1)/x");
        let scaled_surd_den = parse_expr(&mut ctx, "sqrt(2*x^2 + 1)/(3*x)");
        let neg_scaled_surd_den = parse_expr(&mut ctx, "sqrt(2*x^2 + 1)/(-3*x)");
        let noisy_scaled_surd_den = parse_expr(&mut ctx, "sqrt(2*x^2 + x + 1)/(3*x + 1)");
        let bounded_noise_surd_den = parse_expr(&mut ctx, "sqrt((3*x + 1)^2 + sin(x))/(2*x + 1)");
        let bounded_noise_surd_noisy_den =
            parse_expr(&mut ctx, "sqrt((3*x + 1)^2 + sin(x))/(2*x + 1 + cos(x))");
        let scaled_bounded_noise_surd_noisy_den =
            parse_expr(&mut ctx, "5*sqrt((3*x + 1)^2 + sin(x))/(2*x + 1 + cos(x))");
        let bounded_noise_surd_scaled_noisy_den = parse_expr(
            &mut ctx,
            "sqrt((3*x + 1)^2 + sin(x))/(2*(2*x + 1 + cos(x)))",
        );

        let pos_out =
            sqrt_polynomial_ratio_limit_at_infinity(&mut ctx, pos, x, InfSign::Pos).expect("+inf");
        let neg_out =
            sqrt_polynomial_ratio_limit_at_infinity(&mut ctx, neg, x, InfSign::Neg).expect("-inf");
        let scaled_out = sqrt_polynomial_ratio_limit_at_infinity(&mut ctx, scaled, x, InfSign::Pos)
            .expect("scaled");
        let even_den_out =
            sqrt_polynomial_ratio_limit_at_infinity(&mut ctx, even_den, x, InfSign::Neg)
                .expect("even denominator degree");
        let irrational_coeff_out =
            sqrt_polynomial_ratio_limit_at_infinity(&mut ctx, irrational_coeff, x, InfSign::Pos)
                .expect("irrational leading coefficient");
        let scaled_surd_den_out =
            sqrt_polynomial_ratio_limit_at_infinity(&mut ctx, scaled_surd_den, x, InfSign::Pos)
                .expect("scaled surd denominator");
        let neg_scaled_surd_den_out =
            sqrt_polynomial_ratio_limit_at_infinity(&mut ctx, neg_scaled_surd_den, x, InfSign::Neg)
                .expect("negative scaled surd denominator");
        let noisy_scaled_surd_den_pos_out = sqrt_polynomial_ratio_limit_at_infinity(
            &mut ctx,
            noisy_scaled_surd_den,
            x,
            InfSign::Pos,
        )
        .expect("noisy scaled surd denominator at +inf");
        let noisy_scaled_surd_den_neg_out = sqrt_polynomial_ratio_limit_at_infinity(
            &mut ctx,
            noisy_scaled_surd_den,
            x,
            InfSign::Neg,
        )
        .expect("noisy scaled surd denominator at -inf");
        let bounded_noise_surd_den_pos_out = sqrt_polynomial_ratio_limit_at_infinity(
            &mut ctx,
            bounded_noise_surd_den,
            x,
            InfSign::Pos,
        )
        .expect("bounded radicand noise at +inf");
        let bounded_noise_surd_den_neg_out = sqrt_polynomial_ratio_limit_at_infinity(
            &mut ctx,
            bounded_noise_surd_den,
            x,
            InfSign::Neg,
        )
        .expect("bounded radicand noise at -inf");
        let bounded_noise_surd_noisy_den_pos_out = sqrt_polynomial_ratio_limit_at_infinity(
            &mut ctx,
            bounded_noise_surd_noisy_den,
            x,
            InfSign::Pos,
        )
        .expect("bounded radicand and denominator noise at +inf");
        let bounded_noise_surd_noisy_den_neg_out = sqrt_polynomial_ratio_limit_at_infinity(
            &mut ctx,
            bounded_noise_surd_noisy_den,
            x,
            InfSign::Neg,
        )
        .expect("bounded radicand and denominator noise at -inf");
        let scaled_bounded_noise_surd_noisy_den_pos_out = sqrt_polynomial_ratio_limit_at_infinity(
            &mut ctx,
            scaled_bounded_noise_surd_noisy_den,
            x,
            InfSign::Pos,
        )
        .expect("scaled bounded radicand and denominator noise at +inf");
        let bounded_noise_surd_scaled_noisy_den_pos_out = sqrt_polynomial_ratio_limit_at_infinity(
            &mut ctx,
            bounded_noise_surd_scaled_noisy_den,
            x,
            InfSign::Pos,
        )
        .expect("bounded radicand and scaled denominator noise at +inf");

        let one = BigRational::from_integer(BigInt::from(1));
        let minus_one = -one.clone();
        let two = BigRational::from_integer(BigInt::from(2));
        let three = BigRational::from_integer(BigInt::from(3));
        let three_halves = BigRational::new(BigInt::from(3), BigInt::from(2));
        let minus_three_halves = -three_halves.clone();
        let fifteen_halves = BigRational::new(BigInt::from(15), BigInt::from(2));
        let three_quarters = BigRational::new(BigInt::from(3), BigInt::from(4));
        assert!(matches!(ctx.get(pos_out), Expr::Number(n) if n == &one));
        assert!(matches!(ctx.get(neg_out), Expr::Number(n) if n == &minus_one));
        assert!(matches!(ctx.get(scaled_out), Expr::Number(n) if n == &one));
        assert!(matches!(ctx.get(even_den_out), Expr::Number(n) if n == &one));
        assert!(matches!(
            ctx.get(irrational_coeff_out),
            Expr::Function(fn_id, args)
                if ctx.is_builtin(*fn_id, BuiltinFn::Sqrt)
                    && matches!(args.as_slice(), [arg] if matches!(ctx.get(*arg), Expr::Number(n) if n == &two))
        ));
        assert!(matches!(
            ctx.get(scaled_surd_den_out),
            Expr::Div(num, den)
                if matches!(ctx.get(*num), Expr::Function(fn_id, _) if ctx.is_builtin(*fn_id, BuiltinFn::Sqrt))
                    && matches!(ctx.get(*den), Expr::Number(n) if n == &three)
        ));
        assert!(matches!(
            ctx.get(neg_scaled_surd_den_out),
            Expr::Div(num, den)
                if matches!(ctx.get(*num), Expr::Function(fn_id, _) if ctx.is_builtin(*fn_id, BuiltinFn::Sqrt))
                    && matches!(ctx.get(*den), Expr::Number(n) if n == &three)
        ));
        assert!(matches!(
            ctx.get(noisy_scaled_surd_den_pos_out),
            Expr::Div(num, den)
                if matches!(ctx.get(*num), Expr::Function(fn_id, args)
                    if ctx.is_builtin(*fn_id, BuiltinFn::Sqrt)
                        && matches!(args.as_slice(), [arg] if matches!(ctx.get(*arg), Expr::Number(n) if n == &two)))
                    && matches!(ctx.get(*den), Expr::Number(n) if n == &three)
        ));
        assert!(matches!(
            ctx.get(noisy_scaled_surd_den_neg_out),
            Expr::Neg(inner)
                if matches!(ctx.get(*inner), Expr::Div(num, den)
                    if matches!(ctx.get(*num), Expr::Function(fn_id, args)
                        if ctx.is_builtin(*fn_id, BuiltinFn::Sqrt)
                            && matches!(args.as_slice(), [arg] if matches!(ctx.get(*arg), Expr::Number(n) if n == &two)))
                        && matches!(ctx.get(*den), Expr::Number(n) if n == &three))
        ));
        assert!(
            matches!(ctx.get(bounded_noise_surd_den_pos_out), Expr::Number(n) if n == &three_halves)
        );
        assert!(
            matches!(ctx.get(bounded_noise_surd_den_neg_out), Expr::Number(n) if n == &minus_three_halves)
        );
        assert!(
            matches!(ctx.get(bounded_noise_surd_noisy_den_pos_out), Expr::Number(n) if n == &three_halves)
        );
        assert!(
            matches!(ctx.get(bounded_noise_surd_noisy_den_neg_out), Expr::Number(n) if n == &minus_three_halves)
        );
        assert!(
            matches!(ctx.get(scaled_bounded_noise_surd_noisy_den_pos_out), Expr::Number(n) if n == &fifteen_halves)
        );
        assert!(
            matches!(ctx.get(bounded_noise_surd_scaled_noisy_den_pos_out), Expr::Number(n) if n == &three_quarters)
        );
    }

    #[test]
    fn sqrt_polynomial_ratio_limit_at_infinity_rejects_unsafe_shapes() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");

        let negative_leading_coeff = parse_expr(&mut ctx, "sqrt(1 - 2*x^2)/x");
        let odd_radicand_degree = parse_expr(&mut ctx, "sqrt(x^3 + 1)/x");
        let mismatched_growth = parse_expr(&mut ctx, "sqrt(x^2 + 1)/x^2");
        let unbounded_noise = parse_expr(&mut ctx, "sqrt((3*x + 1)^2 + x*sin(x))/(2*x + 1)");
        let unbounded_den_noise =
            parse_expr(&mut ctx, "sqrt((3*x + 1)^2 + sin(x))/(2*x + 1 + x*cos(x))");

        assert!(sqrt_polynomial_ratio_limit_at_infinity(
            &mut ctx,
            negative_leading_coeff,
            x,
            InfSign::Pos
        )
        .is_none());
        assert!(sqrt_polynomial_ratio_limit_at_infinity(
            &mut ctx,
            odd_radicand_degree,
            x,
            InfSign::Pos
        )
        .is_none());
        assert!(sqrt_polynomial_ratio_limit_at_infinity(
            &mut ctx,
            mismatched_growth,
            x,
            InfSign::Pos
        )
        .is_none());
        assert!(sqrt_polynomial_ratio_limit_at_infinity(
            &mut ctx,
            unbounded_noise,
            x,
            InfSign::Pos
        )
        .is_none());
        assert!(sqrt_polynomial_ratio_limit_at_infinity(
            &mut ctx,
            unbounded_den_noise,
            x,
            InfSign::Pos
        )
        .is_none());
    }

    #[test]
    fn polynomial_sqrt_ratio_limit_at_infinity_handles_matching_growth() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");

        let pos = parse_expr(&mut ctx, "x/sqrt(2*x^2 + 1)");
        let neg = parse_expr(&mut ctx, "x/sqrt(2*x^2 + 1)");
        let even_degree = parse_expr(&mut ctx, "x^2/sqrt(2*x^4 + 1)");
        let rational_coeff = parse_expr(&mut ctx, "x/sqrt(4*x^2 + 1)");
        let noisy = parse_expr(&mut ctx, "(3*x + 1)/sqrt(2*x^2 + x + 1)");
        let bounded_noise_num =
            parse_expr(&mut ctx, "(2*x + 1 + cos(x))/sqrt((3*x + 1)^2 + sin(x))");

        let pos_out =
            polynomial_sqrt_ratio_limit_at_infinity(&mut ctx, pos, x, InfSign::Pos).expect("+inf");
        let neg_out =
            polynomial_sqrt_ratio_limit_at_infinity(&mut ctx, neg, x, InfSign::Neg).expect("-inf");
        let even_degree_out =
            polynomial_sqrt_ratio_limit_at_infinity(&mut ctx, even_degree, x, InfSign::Neg)
                .expect("even degree");
        let rational_coeff_out =
            polynomial_sqrt_ratio_limit_at_infinity(&mut ctx, rational_coeff, x, InfSign::Pos)
                .expect("rational sqrt coefficient");
        let noisy_out = polynomial_sqrt_ratio_limit_at_infinity(&mut ctx, noisy, x, InfSign::Pos)
            .expect("lower-order polynomial noise");
        let bounded_noise_num_pos_out =
            polynomial_sqrt_ratio_limit_at_infinity(&mut ctx, bounded_noise_num, x, InfSign::Pos)
                .expect("bounded numerator and radicand noise at +inf");
        let bounded_noise_num_neg_out =
            polynomial_sqrt_ratio_limit_at_infinity(&mut ctx, bounded_noise_num, x, InfSign::Neg)
                .expect("bounded numerator and radicand noise at -inf");

        let two = BigRational::from_integer(BigInt::from(2));
        let three = BigRational::from_integer(BigInt::from(3));
        let two_thirds = BigRational::new(BigInt::from(2), BigInt::from(3));
        let minus_two_thirds = -two_thirds.clone();
        assert!(matches!(
            ctx.get(pos_out),
            Expr::Div(num, den)
                if matches!(ctx.get(*num), Expr::Function(fn_id, args)
                    if ctx.is_builtin(*fn_id, BuiltinFn::Sqrt)
                        && matches!(args.as_slice(), [arg] if matches!(ctx.get(*arg), Expr::Number(n) if n == &two)))
                    && matches!(ctx.get(*den), Expr::Number(n) if n == &two)
        ));
        assert!(matches!(ctx.get(neg_out), Expr::Neg(_)));
        assert!(matches!(
            ctx.get(even_degree_out),
            Expr::Div(num, den)
                if matches!(ctx.get(*num), Expr::Function(fn_id, _) if ctx.is_builtin(*fn_id, BuiltinFn::Sqrt))
                    && matches!(ctx.get(*den), Expr::Number(n) if n == &two)
        ));
        assert!(matches!(
            ctx.get(rational_coeff_out),
            Expr::Number(n) if n == &BigRational::new(BigInt::from(1), BigInt::from(2))
        ));
        assert!(matches!(
            ctx.get(noisy_out),
            Expr::Div(num, den)
                if matches!(ctx.get(*num), Expr::Mul(coeff, sqrt)
                    if matches!(ctx.get(*coeff), Expr::Number(n) if n == &three)
                        && matches!(ctx.get(*sqrt), Expr::Function(fn_id, args)
                            if ctx.is_builtin(*fn_id, BuiltinFn::Sqrt)
                                && matches!(args.as_slice(), [arg] if matches!(ctx.get(*arg), Expr::Number(n) if n == &two))))
                    && matches!(ctx.get(*den), Expr::Number(n) if n == &two)
        ));
        assert!(matches!(ctx.get(bounded_noise_num_pos_out), Expr::Number(n) if n == &two_thirds));
        assert!(
            matches!(ctx.get(bounded_noise_num_neg_out), Expr::Number(n) if n == &minus_two_thirds)
        );
    }

    #[test]
    fn polynomial_sqrt_ratio_limit_at_infinity_rejects_unsafe_shapes() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");

        let negative_leading_coeff = parse_expr(&mut ctx, "x/sqrt(1 - 2*x^2)");
        let odd_radicand_degree = parse_expr(&mut ctx, "x/sqrt(x^3 + 1)");
        let mismatched_growth = parse_expr(&mut ctx, "x/sqrt(x^4 + 1)");
        let unbounded_num_noise =
            parse_expr(&mut ctx, "(2*x + 1 + x*cos(x))/sqrt((3*x + 1)^2 + sin(x))");

        assert!(polynomial_sqrt_ratio_limit_at_infinity(
            &mut ctx,
            negative_leading_coeff,
            x,
            InfSign::Pos
        )
        .is_none());
        assert!(polynomial_sqrt_ratio_limit_at_infinity(
            &mut ctx,
            odd_radicand_degree,
            x,
            InfSign::Pos
        )
        .is_none());
        assert!(polynomial_sqrt_ratio_limit_at_infinity(
            &mut ctx,
            mismatched_growth,
            x,
            InfSign::Pos
        )
        .is_none());
        assert!(polynomial_sqrt_ratio_limit_at_infinity(
            &mut ctx,
            unbounded_num_noise,
            x,
            InfSign::Pos
        )
        .is_none());
    }

    #[test]
    fn polynomial_limit_at_infinity_handles_numeric_leading_coeff_and_parity() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let positive_even = parse_expr(&mut ctx, "x^2 + 1");
        let negative_odd = parse_expr(&mut ctx, "x - 2*x^3");

        let pos_even_out = polynomial_limit_at_infinity(&mut ctx, positive_even, x, InfSign::Neg)
            .expect("positive even polynomial");
        let neg_odd_pos_out = polynomial_limit_at_infinity(&mut ctx, negative_odd, x, InfSign::Pos)
            .expect("negative odd at +inf");
        let neg_odd_neg_out = polynomial_limit_at_infinity(&mut ctx, negative_odd, x, InfSign::Neg)
            .expect("negative odd at -inf");

        assert!(matches!(
            ctx.get(pos_even_out),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(matches!(ctx.get(neg_odd_pos_out), Expr::Neg(_)));
        assert!(matches!(
            ctx.get(neg_odd_neg_out),
            Expr::Constant(Constant::Infinity)
        ));
    }

    #[test]
    fn polynomial_limit_at_infinity_rejects_symbolic_leading_coeff() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let symbolic_lc = parse_expr(&mut ctx, "y*x^2 + 1");

        let out = polynomial_limit_at_infinity(&mut ctx, symbolic_lc, x, InfSign::Pos);

        assert!(out.is_none());
    }

    #[test]
    fn elementary_function_limit_at_infinity_handles_exact_growth_cases() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let sqrt_x = parse_expr(&mut ctx, "sqrt(x)");
        let ln_x = parse_expr(&mut ctx, "ln(x)");
        let exp_x = parse_expr(&mut ctx, "exp(x)");
        let exp_neg_x = parse_expr(&mut ctx, "exp(-x)");
        let exp_two_x = parse_expr(&mut ctx, "exp(2*x)");
        let ln_neg_linear = parse_expr(&mut ctx, "ln(-x + 1)");
        let sqrt_neg_linear = parse_expr(&mut ctx, "sqrt(1 - x)");
        let exp_quadratic = parse_expr(&mut ctx, "exp(x^2)");

        let sqrt_pos = elementary_function_limit_at_infinity(&mut ctx, sqrt_x, x, InfSign::Pos)
            .expect("sqrt at +inf");
        let ln_pos = elementary_function_limit_at_infinity(&mut ctx, ln_x, x, InfSign::Pos)
            .expect("ln at +inf");
        let exp_pos = elementary_function_limit_at_infinity(&mut ctx, exp_x, x, InfSign::Pos)
            .expect("exp at +inf");
        let exp_neg = elementary_function_limit_at_infinity(&mut ctx, exp_x, x, InfSign::Neg)
            .expect("exp at -inf");
        let exp_neg_x_pos =
            elementary_function_limit_at_infinity(&mut ctx, exp_neg_x, x, InfSign::Pos)
                .expect("exp(-x) at +inf");
        let exp_two_x_neg =
            elementary_function_limit_at_infinity(&mut ctx, exp_two_x, x, InfSign::Neg)
                .expect("exp(2*x) at -inf");
        let ln_neg_linear_neg =
            elementary_function_limit_at_infinity(&mut ctx, ln_neg_linear, x, InfSign::Neg)
                .expect("ln(-x + 1) at -inf");
        let sqrt_neg_linear_neg =
            elementary_function_limit_at_infinity(&mut ctx, sqrt_neg_linear, x, InfSign::Neg)
                .expect("sqrt(1 - x) at -inf");

        assert!(matches!(
            ctx.get(sqrt_pos),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(matches!(
            ctx.get(ln_pos),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(matches!(
            ctx.get(exp_pos),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(
            matches!(ctx.get(exp_neg), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert!(
            matches!(ctx.get(exp_neg_x_pos), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert!(
            matches!(ctx.get(exp_two_x_neg), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert!(matches!(
            ctx.get(ln_neg_linear_neg),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(matches!(
            ctx.get(sqrt_neg_linear_neg),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(elementary_function_limit_at_infinity(&mut ctx, sqrt_x, x, InfSign::Neg).is_none());
        assert!(elementary_function_limit_at_infinity(&mut ctx, ln_x, x, InfSign::Neg).is_none());
        assert!(
            elementary_function_limit_at_infinity(&mut ctx, exp_quadratic, x, InfSign::Pos)
                .is_none()
        );
    }

    #[test]
    fn additive_limit_at_infinity_combines_finite_and_infinite_terms_conservatively() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let sqrt_plus_one = parse_expr(&mut ctx, "sqrt(x) + 1");
        let decaying_exp_plus_one = parse_expr(&mut ctx, "exp(-x) + 1");
        let exp_minus_poly = parse_expr(&mut ctx, "exp(x) - x^2");
        let poly_cancel = parse_expr(&mut ctx, "x^2 - x^2");

        let sqrt_plus_one_out =
            try_limit_rules_at_infinity(&mut ctx, sqrt_plus_one, x, InfSign::Pos)
                .expect("sqrt plus one");
        let decaying_exp_plus_one_out =
            try_limit_rules_at_infinity(&mut ctx, decaying_exp_plus_one, x, InfSign::Pos)
                .expect("decaying exp plus one");
        let poly_cancel_out = try_limit_rules_at_infinity(&mut ctx, poly_cancel, x, InfSign::Pos)
            .expect("polynomial cancellation");

        assert!(matches!(
            ctx.get(sqrt_plus_one_out),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(
            matches!(ctx.get(decaying_exp_plus_one_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(1)))
        );
        let exp_minus_poly_out =
            try_limit_rules_at_infinity(&mut ctx, exp_minus_poly, x, InfSign::Pos)
                .expect("exp dominates polynomial");
        assert!(matches!(
            ctx.get(exp_minus_poly_out),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(
            matches!(ctx.get(poly_cancel_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
    }

    #[test]
    fn multiplicative_limit_at_infinity_combines_only_determined_products_and_quotients() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let scaled_sqrt = parse_expr(&mut ctx, "2*sqrt(x)");
        let neg_sqrt = parse_expr(&mut ctx, "-sqrt(x)");
        let reciprocal_exp = parse_expr(&mut ctx, "1/exp(x)");
        let indeterminate_exp_difference = parse_expr(&mut ctx, "exp(x)-exp(x)");

        let scaled_sqrt_out = try_limit_rules_at_infinity(&mut ctx, scaled_sqrt, x, InfSign::Pos)
            .expect("scaled sqrt");
        let neg_sqrt_out =
            try_limit_rules_at_infinity(&mut ctx, neg_sqrt, x, InfSign::Pos).expect("neg sqrt");
        let reciprocal_exp_out =
            try_limit_rules_at_infinity(&mut ctx, reciprocal_exp, x, InfSign::Pos)
                .expect("reciprocal exp");

        assert!(matches!(
            ctx.get(scaled_sqrt_out),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(matches!(ctx.get(neg_sqrt_out), Expr::Neg(_)));
        assert!(
            matches!(ctx.get(reciprocal_exp_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert!(try_limit_rules_at_infinity(
            &mut ctx,
            indeterminate_exp_difference,
            x,
            InfSign::Pos
        )
        .is_none());
    }

    #[test]
    fn exponential_polynomial_dominance_handles_only_exact_safe_shapes() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let exp_minus_poly = parse_expr(&mut ctx, "exp(x) - x^2");
        let poly_minus_exp = parse_expr(&mut ctx, "x^2 - exp(x)");
        let poly_over_exp = parse_expr(&mut ctx, "x^2/exp(x)");
        let exp_over_poly = parse_expr(&mut ctx, "exp(x)/x^2");
        let poly_times_decaying_exp = parse_expr(&mut ctx, "x*exp(x)");
        let poly_over_linear_exp = parse_expr(&mut ctx, "x^2/exp(2*x)");
        let linear_exp_over_poly = parse_expr(&mut ctx, "exp(2*x)/x^2");
        let poly_times_decaying_linear_exp = parse_expr(&mut ctx, "x^2*exp(2*x)");
        let poly_over_decaying_linear_exp = parse_expr(&mut ctx, "x^2/exp(-2*x)");
        let constant_over_decaying_linear_exp = parse_expr(&mut ctx, "1/exp(-2*x)");
        let composed_exp_over_poly = parse_expr(&mut ctx, "exp(x^2)/x^2");
        let even_poly_over_exp_neg = parse_expr(&mut ctx, "x^2/exp(x)");
        let odd_poly_over_exp_neg = parse_expr(&mut ctx, "x/exp(x)");
        let neg_scaled_exp_den_neg = parse_expr(&mut ctx, "x^2/(-2*exp(x))");
        let zero_scaled_exp_den = parse_expr(&mut ctx, "x^2/(0*exp(x))");
        let zero_scaled_linear_exp_den = parse_expr(&mut ctx, "x^2/(0*exp(2*x))");

        let exp_minus_poly_out =
            try_limit_rules_at_infinity(&mut ctx, exp_minus_poly, x, InfSign::Pos)
                .expect("exp dominates polynomial difference");
        let poly_minus_exp_out =
            try_limit_rules_at_infinity(&mut ctx, poly_minus_exp, x, InfSign::Pos)
                .expect("negative exponential dominance");
        let poly_over_exp_out =
            try_limit_rules_at_infinity(&mut ctx, poly_over_exp, x, InfSign::Pos)
                .expect("polynomial over exp");
        let exp_over_poly_out =
            try_limit_rules_at_infinity(&mut ctx, exp_over_poly, x, InfSign::Pos)
                .expect("exp over polynomial");
        let poly_times_decaying_exp_out =
            try_limit_rules_at_infinity(&mut ctx, poly_times_decaying_exp, x, InfSign::Neg)
                .expect("polynomial times decaying exp");
        let poly_over_linear_exp_out =
            try_limit_rules_at_infinity(&mut ctx, poly_over_linear_exp, x, InfSign::Pos)
                .expect("polynomial over linear exp");
        let linear_exp_over_poly_out =
            try_limit_rules_at_infinity(&mut ctx, linear_exp_over_poly, x, InfSign::Pos)
                .expect("linear exp over polynomial");
        let poly_times_decaying_linear_exp_out =
            try_limit_rules_at_infinity(&mut ctx, poly_times_decaying_linear_exp, x, InfSign::Neg)
                .expect("polynomial times decaying linear exp");
        let poly_over_decaying_linear_exp_out =
            try_limit_rules_at_infinity(&mut ctx, poly_over_decaying_linear_exp, x, InfSign::Pos)
                .expect("polynomial over decaying linear exp");
        let constant_over_decaying_linear_exp_out = try_limit_rules_at_infinity(
            &mut ctx,
            constant_over_decaying_linear_exp,
            x,
            InfSign::Pos,
        )
        .expect("constant over decaying linear exp");
        let even_poly_over_exp_neg_out =
            try_limit_rules_at_infinity(&mut ctx, even_poly_over_exp_neg, x, InfSign::Neg)
                .expect("even polynomial over decaying exp");
        let odd_poly_over_exp_neg_out =
            try_limit_rules_at_infinity(&mut ctx, odd_poly_over_exp_neg, x, InfSign::Neg)
                .expect("odd polynomial over decaying exp");
        let neg_scaled_exp_den_neg_out =
            try_limit_rules_at_infinity(&mut ctx, neg_scaled_exp_den_neg, x, InfSign::Neg)
                .expect("negative scaled exp denominator");

        assert!(matches!(
            ctx.get(exp_minus_poly_out),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(matches!(ctx.get(poly_minus_exp_out), Expr::Neg(_)));
        assert!(
            matches!(ctx.get(poly_over_exp_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert!(matches!(
            ctx.get(exp_over_poly_out),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(
            matches!(ctx.get(poly_times_decaying_exp_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert!(
            matches!(ctx.get(poly_over_linear_exp_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert!(matches!(
            ctx.get(linear_exp_over_poly_out),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(
            matches!(ctx.get(poly_times_decaying_linear_exp_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert!(matches!(
            ctx.get(poly_over_decaying_linear_exp_out),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(matches!(
            ctx.get(constant_over_decaying_linear_exp_out),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(matches!(
            ctx.get(even_poly_over_exp_neg_out),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(matches!(ctx.get(odd_poly_over_exp_neg_out), Expr::Neg(_)));
        assert!(matches!(ctx.get(neg_scaled_exp_den_neg_out), Expr::Neg(_)));
        assert!(
            try_limit_rules_at_infinity(&mut ctx, composed_exp_over_poly, x, InfSign::Pos)
                .is_none()
        );
        assert!(
            try_limit_rules_at_infinity(&mut ctx, zero_scaled_exp_den, x, InfSign::Pos).is_none()
        );
        assert!(
            try_limit_rules_at_infinity(&mut ctx, zero_scaled_linear_exp_den, x, InfSign::Pos)
                .is_none()
        );
    }

    #[test]
    fn subpolynomial_polynomial_dominance_handles_only_domain_safe_shapes() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let log_over_poly = parse_expr(&mut ctx, "ln(x)/x");
        let poly_over_log = parse_expr(&mut ctx, "x/ln(x)");
        let root_over_poly = parse_expr(&mut ctx, "sqrt(x)/x");
        let poly_over_root = parse_expr(&mut ctx, "x/sqrt(x)");
        let base_log_over_poly = parse_expr(&mut ctx, "log(2, x)/x");
        let unary_log10_over_poly = parse_expr(&mut ctx, "log10(x)/x");
        let poly_over_half_base_log = parse_expr(&mut ctx, "x/log(1/2, x)");
        let e_base_log_over_poly = parse_expr(&mut ctx, "log(e, x)/x");
        let pi_base_log_over_poly = parse_expr(&mut ctx, "log(pi, x)/x");
        let phi_base_log_over_poly = parse_expr(&mut ctx, "log(phi, x)/x");
        let poly_over_reciprocal_e_base_log = parse_expr(&mut ctx, "x/log(1/e, x)");
        let powered_e_base_log_over_poly = parse_expr(&mut ctx, "log(e^2, x)/x");
        let powered_phi_base_log_over_poly = parse_expr(&mut ctx, "log(phi^3, x)/x");
        let poly_over_negative_power_e_base_log = parse_expr(&mut ctx, "x/log(e^-2, x)");
        let poly_over_reciprocal_power_pi_base_log = parse_expr(&mut ctx, "x/log((1/pi)^2, x)");
        let neg_tail_log_over_poly = parse_expr(&mut ctx, "ln(1 - x)/x^2");
        let neg_tail_poly_over_log = parse_expr(&mut ctx, "x/ln(1 - x)");
        let log_minus_poly = parse_expr(&mut ctx, "ln(x) - x");
        let poly_minus_root = parse_expr(&mut ctx, "x - sqrt(x)");
        let bad_domain_log = parse_expr(&mut ctx, "ln(x)/x");
        let bad_domain_base_log = parse_expr(&mut ctx, "log(2, x)/x");
        let invalid_base_log = parse_expr(&mut ctx, "log(1, x)/x");
        let negative_named_base_log = parse_expr(&mut ctx, "log(-e, x)/x");
        let zero_power_named_base_log = parse_expr(&mut ctx, "log(e^0, x)/x");
        let nonlinear_log = parse_expr(&mut ctx, "ln(x^2)/x");
        let subpoly_over_subpoly = parse_expr(&mut ctx, "ln(x)/sqrt(x)");
        let zero_scaled_log_den = parse_expr(&mut ctx, "x/(0*ln(x))");

        let log_over_poly_out =
            try_limit_rules_at_infinity(&mut ctx, log_over_poly, x, InfSign::Pos)
                .expect("log over polynomial");
        let poly_over_log_out =
            try_limit_rules_at_infinity(&mut ctx, poly_over_log, x, InfSign::Pos)
                .expect("polynomial over log");
        let root_over_poly_out =
            try_limit_rules_at_infinity(&mut ctx, root_over_poly, x, InfSign::Pos)
                .expect("root over polynomial");
        let poly_over_root_out =
            try_limit_rules_at_infinity(&mut ctx, poly_over_root, x, InfSign::Pos)
                .expect("polynomial over root");
        let base_log_over_poly_out =
            try_limit_rules_at_infinity(&mut ctx, base_log_over_poly, x, InfSign::Pos)
                .expect("general-base log over polynomial");
        let unary_log10_over_poly_out =
            try_limit_rules_at_infinity(&mut ctx, unary_log10_over_poly, x, InfSign::Pos)
                .expect("log10 over polynomial");
        let poly_over_half_base_log_out =
            try_limit_rules_at_infinity(&mut ctx, poly_over_half_base_log, x, InfSign::Pos)
                .expect("polynomial over base < 1 log");
        let e_base_log_over_poly_out =
            try_limit_rules_at_infinity(&mut ctx, e_base_log_over_poly, x, InfSign::Pos)
                .expect("e-base log over polynomial");
        let pi_base_log_over_poly_out =
            try_limit_rules_at_infinity(&mut ctx, pi_base_log_over_poly, x, InfSign::Pos)
                .expect("pi-base log over polynomial");
        let phi_base_log_over_poly_out =
            try_limit_rules_at_infinity(&mut ctx, phi_base_log_over_poly, x, InfSign::Pos)
                .expect("phi-base log over polynomial");
        let poly_over_reciprocal_e_base_log_out =
            try_limit_rules_at_infinity(&mut ctx, poly_over_reciprocal_e_base_log, x, InfSign::Pos)
                .expect("polynomial over reciprocal e-base log");
        let powered_e_base_log_over_poly_out =
            try_limit_rules_at_infinity(&mut ctx, powered_e_base_log_over_poly, x, InfSign::Pos)
                .expect("powered e-base log over polynomial");
        let powered_phi_base_log_over_poly_out =
            try_limit_rules_at_infinity(&mut ctx, powered_phi_base_log_over_poly, x, InfSign::Pos)
                .expect("powered phi-base log over polynomial");
        let poly_over_negative_power_e_base_log_out = try_limit_rules_at_infinity(
            &mut ctx,
            poly_over_negative_power_e_base_log,
            x,
            InfSign::Pos,
        )
        .expect("polynomial over negative powered e-base log");
        let poly_over_reciprocal_power_pi_base_log_out = try_limit_rules_at_infinity(
            &mut ctx,
            poly_over_reciprocal_power_pi_base_log,
            x,
            InfSign::Pos,
        )
        .expect("polynomial over reciprocal power pi-base log");
        let neg_tail_log_over_poly_out =
            try_limit_rules_at_infinity(&mut ctx, neg_tail_log_over_poly, x, InfSign::Neg)
                .expect("negative-tail log over polynomial");
        let neg_tail_poly_over_log_out =
            try_limit_rules_at_infinity(&mut ctx, neg_tail_poly_over_log, x, InfSign::Neg)
                .expect("negative-tail polynomial over log");
        let log_minus_poly_out =
            try_limit_rules_at_infinity(&mut ctx, log_minus_poly, x, InfSign::Pos)
                .expect("log minus polynomial");
        let poly_minus_root_out =
            try_limit_rules_at_infinity(&mut ctx, poly_minus_root, x, InfSign::Pos)
                .expect("polynomial minus root");

        assert!(
            matches!(ctx.get(log_over_poly_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert!(matches!(
            ctx.get(poly_over_log_out),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(
            matches!(ctx.get(root_over_poly_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert!(matches!(
            ctx.get(poly_over_root_out),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(
            matches!(ctx.get(base_log_over_poly_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert!(
            matches!(ctx.get(unary_log10_over_poly_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert!(matches!(ctx.get(poly_over_half_base_log_out), Expr::Neg(_)));
        assert!(
            matches!(ctx.get(e_base_log_over_poly_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert!(
            matches!(ctx.get(pi_base_log_over_poly_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert!(
            matches!(ctx.get(phi_base_log_over_poly_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert!(matches!(
            ctx.get(poly_over_reciprocal_e_base_log_out),
            Expr::Neg(_)
        ));
        assert!(
            matches!(ctx.get(powered_e_base_log_over_poly_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert!(
            matches!(ctx.get(powered_phi_base_log_over_poly_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert!(matches!(
            ctx.get(poly_over_negative_power_e_base_log_out),
            Expr::Neg(_)
        ));
        assert!(matches!(
            ctx.get(poly_over_reciprocal_power_pi_base_log_out),
            Expr::Neg(_)
        ));
        assert!(
            matches!(ctx.get(neg_tail_log_over_poly_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert!(matches!(ctx.get(neg_tail_poly_over_log_out), Expr::Neg(_)));
        assert!(matches!(ctx.get(log_minus_poly_out), Expr::Neg(_)));
        assert!(matches!(
            ctx.get(poly_minus_root_out),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(try_limit_rules_at_infinity(&mut ctx, bad_domain_log, x, InfSign::Neg).is_none());
        assert!(
            try_limit_rules_at_infinity(&mut ctx, bad_domain_base_log, x, InfSign::Neg).is_none()
        );
        assert!(try_limit_rules_at_infinity(&mut ctx, invalid_base_log, x, InfSign::Pos).is_none());
        assert!(
            try_limit_rules_at_infinity(&mut ctx, negative_named_base_log, x, InfSign::Pos)
                .is_none()
        );
        assert!(
            try_limit_rules_at_infinity(&mut ctx, zero_power_named_base_log, x, InfSign::Pos)
                .is_none()
        );
        assert!(try_limit_rules_at_infinity(&mut ctx, nonlinear_log, x, InfSign::Pos).is_none());
        assert!(
            try_limit_rules_at_infinity(&mut ctx, subpoly_over_subpoly, x, InfSign::Pos).is_none()
        );
        assert!(
            try_limit_rules_at_infinity(&mut ctx, zero_scaled_log_den, x, InfSign::Pos).is_none()
        );
    }

    #[test]
    fn exponential_subpolynomial_dominance_handles_only_domain_safe_shapes() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let log_over_exp = parse_expr(&mut ctx, "ln(x)/exp(x)");
        let exp_over_log = parse_expr(&mut ctx, "exp(x)/ln(x)");
        let root_times_decaying_exp = parse_expr(&mut ctx, "sqrt(x)*exp(-x)");
        let log_over_decaying_exp = parse_expr(&mut ctx, "ln(x)/exp(-x)");
        let negative_log_over_decaying_exp = parse_expr(&mut ctx, "-ln(x)/exp(-x)");
        let exp_over_negative_root = parse_expr(&mut ctx, "exp(x)/(-sqrt(x))");
        let base_log_over_exp = parse_expr(&mut ctx, "log(2, x)/exp(x)");
        let exp_over_half_base_log = parse_expr(&mut ctx, "exp(x)/log(1/2, x)");
        let exp_over_unary_log2 = parse_expr(&mut ctx, "exp(x)/log2(x)");
        let exp_over_e_base_log = parse_expr(&mut ctx, "exp(x)/log(e, x)");
        let exp_over_reciprocal_e_base_log = parse_expr(&mut ctx, "exp(x)/log(1/e, x)");
        let exp_over_powered_e_base_log = parse_expr(&mut ctx, "exp(x)/log(e^2, x)");
        let exp_over_negative_power_e_base_log = parse_expr(&mut ctx, "exp(x)/log(e^-2, x)");
        let exp_minus_log = parse_expr(&mut ctx, "exp(x) - ln(x)");
        let log_minus_exp = parse_expr(&mut ctx, "ln(x) - exp(x)");
        let neg_tail_log_times_decaying_exp = parse_expr(&mut ctx, "ln(1 - x)*exp(x)");
        let bad_domain_log_over_exp = parse_expr(&mut ctx, "ln(x)/exp(-x)");
        let invalid_base_log_over_exp = parse_expr(&mut ctx, "log(1, x)/exp(x)");
        let negative_named_base_log_over_exp = parse_expr(&mut ctx, "log(-e, x)/exp(x)");
        let nonlinear_exp_over_log = parse_expr(&mut ctx, "exp(x^2)/ln(x)");
        let zero_exp_denominator = parse_expr(&mut ctx, "ln(x)/(0*exp(x))");

        let log_over_exp_out = try_limit_rules_at_infinity(&mut ctx, log_over_exp, x, InfSign::Pos)
            .expect("log over growing exp");
        let exp_over_log_out = try_limit_rules_at_infinity(&mut ctx, exp_over_log, x, InfSign::Pos)
            .expect("growing exp over log");
        let root_times_decaying_exp_out =
            try_limit_rules_at_infinity(&mut ctx, root_times_decaying_exp, x, InfSign::Pos)
                .expect("root times decaying exp");
        let log_over_decaying_exp_out =
            try_limit_rules_at_infinity(&mut ctx, log_over_decaying_exp, x, InfSign::Pos)
                .expect("log over decaying exp");
        let negative_log_over_decaying_exp_out =
            try_limit_rules_at_infinity(&mut ctx, negative_log_over_decaying_exp, x, InfSign::Pos)
                .expect("negative log over decaying exp");
        let exp_over_negative_root_out =
            try_limit_rules_at_infinity(&mut ctx, exp_over_negative_root, x, InfSign::Pos)
                .expect("exp over negative root");
        let base_log_over_exp_out =
            try_limit_rules_at_infinity(&mut ctx, base_log_over_exp, x, InfSign::Pos)
                .expect("general-base log over growing exp");
        let exp_over_half_base_log_out =
            try_limit_rules_at_infinity(&mut ctx, exp_over_half_base_log, x, InfSign::Pos)
                .expect("growing exp over base < 1 log");
        let exp_over_unary_log2_out =
            try_limit_rules_at_infinity(&mut ctx, exp_over_unary_log2, x, InfSign::Pos)
                .expect("growing exp over log2");
        let exp_over_e_base_log_out =
            try_limit_rules_at_infinity(&mut ctx, exp_over_e_base_log, x, InfSign::Pos)
                .expect("growing exp over e-base log");
        let exp_over_reciprocal_e_base_log_out =
            try_limit_rules_at_infinity(&mut ctx, exp_over_reciprocal_e_base_log, x, InfSign::Pos)
                .expect("growing exp over reciprocal e-base log");
        let exp_over_powered_e_base_log_out =
            try_limit_rules_at_infinity(&mut ctx, exp_over_powered_e_base_log, x, InfSign::Pos)
                .expect("growing exp over powered e-base log");
        let exp_over_negative_power_e_base_log_out = try_limit_rules_at_infinity(
            &mut ctx,
            exp_over_negative_power_e_base_log,
            x,
            InfSign::Pos,
        )
        .expect("growing exp over negative powered e-base log");
        let exp_minus_log_out =
            try_limit_rules_at_infinity(&mut ctx, exp_minus_log, x, InfSign::Pos)
                .expect("exp minus log");
        let log_minus_exp_out =
            try_limit_rules_at_infinity(&mut ctx, log_minus_exp, x, InfSign::Pos)
                .expect("log minus exp");
        let neg_tail_log_times_decaying_exp_out =
            try_limit_rules_at_infinity(&mut ctx, neg_tail_log_times_decaying_exp, x, InfSign::Neg)
                .expect("negative-tail log times decaying exp");

        assert!(
            matches!(ctx.get(log_over_exp_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert!(matches!(
            ctx.get(exp_over_log_out),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(
            matches!(ctx.get(root_times_decaying_exp_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert!(matches!(
            ctx.get(log_over_decaying_exp_out),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(matches!(
            ctx.get(negative_log_over_decaying_exp_out),
            Expr::Neg(_)
        ));
        assert!(matches!(ctx.get(exp_over_negative_root_out), Expr::Neg(_)));
        assert!(
            matches!(ctx.get(base_log_over_exp_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert!(matches!(ctx.get(exp_over_half_base_log_out), Expr::Neg(_)));
        assert!(matches!(
            ctx.get(exp_over_unary_log2_out),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(matches!(
            ctx.get(exp_over_e_base_log_out),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(matches!(
            ctx.get(exp_over_reciprocal_e_base_log_out),
            Expr::Neg(_)
        ));
        assert!(matches!(
            ctx.get(exp_over_powered_e_base_log_out),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(matches!(
            ctx.get(exp_over_negative_power_e_base_log_out),
            Expr::Neg(_)
        ));
        assert!(matches!(
            ctx.get(exp_minus_log_out),
            Expr::Constant(Constant::Infinity)
        ));
        assert!(matches!(ctx.get(log_minus_exp_out), Expr::Neg(_)));
        assert!(
            matches!(ctx.get(neg_tail_log_times_decaying_exp_out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert!(
            try_limit_rules_at_infinity(&mut ctx, bad_domain_log_over_exp, x, InfSign::Neg)
                .is_none()
        );
        assert!(
            try_limit_rules_at_infinity(&mut ctx, invalid_base_log_over_exp, x, InfSign::Pos)
                .is_none()
        );
        assert!(try_limit_rules_at_infinity(
            &mut ctx,
            negative_named_base_log_over_exp,
            x,
            InfSign::Pos,
        )
        .is_none());
        assert!(
            try_limit_rules_at_infinity(&mut ctx, nonlinear_exp_over_log, x, InfSign::Pos)
                .is_none()
        );
        assert!(
            try_limit_rules_at_infinity(&mut ctx, zero_exp_denominator, x, InfSign::Pos).is_none()
        );
    }

    #[test]
    fn apply_power_rule_handles_zero_and_negative_exponents() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let x0 = parse_expr(&mut ctx, "x^0");
        let xneg = parse_expr(&mut ctx, "x^-3");

        let out0 = apply_power_rule(&mut ctx, x0, x, InfSign::Pos).expect("x^0");
        let out_neg = apply_power_rule(&mut ctx, xneg, x, InfSign::Neg).expect("x^-3");

        assert!(
            matches!(ctx.get(out0), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(1)))
        );
        assert!(
            matches!(ctx.get(out_neg), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
    }

    #[test]
    fn apply_reciprocal_power_rule_handles_one_over_xn() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let expr1 = parse_expr(&mut ctx, "1/x");
        let expr2 = parse_expr(&mut ctx, "5/x^3");

        let out1 = apply_reciprocal_power_rule(&mut ctx, expr1, x).expect("1/x");
        let out2 = apply_reciprocal_power_rule(&mut ctx, expr2, x).expect("5/x^3");

        assert!(
            matches!(ctx.get(out1), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
        assert!(
            matches!(ctx.get(out2), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
    }

    #[test]
    fn try_limit_rules_at_infinity_resolves_constant_and_variable() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let c = parse_expr(&mut ctx, "7");

        let c_out = try_limit_rules_at_infinity(&mut ctx, c, x, InfSign::Pos).expect("constant");
        let x_out = try_limit_rules_at_infinity(&mut ctx, x, x, InfSign::Neg).expect("variable");

        assert_eq!(c_out, c);
        assert!(matches!(ctx.get(x_out), Expr::Neg(_)));
    }

    #[test]
    fn try_limit_rules_at_infinity_uses_rational_poly_fallback() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let expr = parse_expr(&mut ctx, "x^2/x^3");

        let out = try_limit_rules_at_infinity(&mut ctx, expr, x, InfSign::Pos).expect("rational");
        assert!(
            matches!(ctx.get(out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0)))
        );
    }

    #[test]
    fn try_limit_rules_at_infinity_uses_polynomial_growth_before_residual() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let expr = parse_expr(&mut ctx, "2*x^3 + x");

        let out = try_limit_rules_at_infinity(&mut ctx, expr, x, InfSign::Pos).expect("polynomial");

        assert!(matches!(ctx.get(out), Expr::Constant(Constant::Infinity)));
    }

    #[test]
    fn bounded_elementary_over_divergent_limit_at_infinity_resolves_to_zero() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let cases = [
            "sin(x)/x",
            "cos(2*x + 1)/(x^2 + 1)",
            "(2*sin(x) - cos(x))/(-x)",
            "sin(x)*cos(x)/exp(x)",
            "arctan(x)/x",
            "atan(x^2 + 1)/(x^2 + 1)",
            "(arctan(x) + sin(x))/(0 - x)",
            "tanh(x)/x",
            "tanh(x^2 + 1)/(x^2 + 1)",
            "(tanh(x) - cos(x))/exp(x)",
        ];

        for expr in cases {
            let parsed = parse_expr(&mut ctx, expr);
            let out = try_limit_rules_at_infinity(&mut ctx, parsed, x, InfSign::Pos)
                .unwrap_or_else(|| panic!("expected bounded-over-divergent zero for {expr}"));
            assert!(
                matches!(ctx.get(out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0))),
                "expected zero for {expr}, got {:?}",
                ctx.get(out)
            );
        }
    }

    #[test]
    fn bounded_elementary_over_divergent_limit_at_infinity_rejects_unbounded_or_nondominant_shapes()
    {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let unbounded_num = parse_expr(&mut ctx, "x*sin(x)/x");
        let nondominant_den = parse_expr(&mut ctx, "sin(x)/cos(x)");
        let arctan_nondominant_den = parse_expr(&mut ctx, "arctan(x)/cos(x)");
        let tanh_nondominant_den = parse_expr(&mut ctx, "tanh(x)/cos(x)");

        assert!(bounded_elementary_over_divergent_limit_at_infinity(
            &mut ctx,
            unbounded_num,
            x,
            InfSign::Pos,
        )
        .is_none());
        assert!(bounded_elementary_over_divergent_limit_at_infinity(
            &mut ctx,
            nondominant_den,
            x,
            InfSign::Pos,
        )
        .is_none());
        assert!(bounded_elementary_over_divergent_limit_at_infinity(
            &mut ctx,
            arctan_nondominant_den,
            x,
            InfSign::Pos,
        )
        .is_none());
        assert!(bounded_elementary_over_divergent_limit_at_infinity(
            &mut ctx,
            tanh_nondominant_den,
            x,
            InfSign::Pos,
        )
        .is_none());
    }

    #[test]
    fn bounded_elementary_times_decaying_exp_limit_at_infinity_resolves_to_zero() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let cases = [
            "sin(x)*exp(-x)",
            "exp(-2*x)*cos(x)",
            "(sin(x) + cos(x))*exp(-x)",
            "arctan(x)*exp(-x)",
            "-tanh(x)*exp(-x)",
        ];

        for expr in cases {
            let parsed = parse_expr(&mut ctx, expr);
            let out = try_limit_rules_at_infinity(&mut ctx, parsed, x, InfSign::Pos)
                .unwrap_or_else(|| panic!("expected bounded-times-decaying-exp zero for {expr}"));
            assert!(
                matches!(ctx.get(out), Expr::Number(n) if n == &BigRational::from_integer(BigInt::from(0))),
                "expected zero for {expr}, got {:?}",
                ctx.get(out)
            );
        }

        let tan_product = parse_expr(&mut ctx, "tan(x)*exp(-x)");
        assert!(bounded_elementary_times_decaying_exp_limit_at_infinity(
            &mut ctx,
            tan_product,
            x,
            InfSign::Pos,
        )
        .is_none());
    }

    #[test]
    fn presimplify_safe_for_limit_applies_allowlisted_rewrites() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");
        let expr = parse_expr(&mut ctx, "x + 0");
        let out = presimplify_safe_for_limit(&mut ctx, expr);
        assert_eq!(out, x);
    }

    #[test]
    fn presimplify_safe_for_limit_does_not_apply_domain_sensitive_rewrites() {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, "x/x");
        let out = presimplify_safe_for_limit(&mut ctx, expr);
        assert!(matches!(ctx.get(out), Expr::Div(_, _)));
    }
}
