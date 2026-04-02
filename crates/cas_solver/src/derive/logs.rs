use cas_ast::ordering::compare_expr;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::expr_nary::{add_terms_signed, Sign};
use cas_math::expr_rewrite::smart_mul;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};
use std::cmp::Ordering;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LogFamily {
    Ln,
    Log10,
    LogBase(ExprId),
}

#[derive(Debug, Clone)]
struct ScaledLogTerm {
    family: LogFamily,
    arg: ExprId,
    coeff: BigRational,
}

pub(crate) fn try_rewrite_log_contraction_target_aware(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let terms = add_terms_signed(ctx, expr);
    if terms.len() < 2 {
        return None;
    }

    let mut parsed_terms = Vec::with_capacity(terms.len());
    for (term, sign) in terms {
        parsed_terms.push(extract_scaled_log_term(ctx, term, sign)?);
    }

    let first_family = parsed_terms.first()?.family;
    if !parsed_terms
        .iter()
        .all(|term| same_log_family(ctx, term.family, first_family))
    {
        return None;
    }

    let mut numerator_factors = Vec::new();
    let mut denominator_factors = Vec::new();

    for term in parsed_terms {
        if term.coeff.is_zero() {
            continue;
        }
        let magnitude = term.coeff.abs();
        let factor = build_powered_log_factor(ctx, term.arg, &magnitude);
        if term.coeff.is_negative() {
            denominator_factors.push(factor);
        } else {
            numerator_factors.push(factor);
        }
    }

    if numerator_factors.is_empty() && denominator_factors.is_empty() {
        return None;
    }

    let rewritten = match (numerator_factors.is_empty(), denominator_factors.is_empty()) {
        (false, true) => {
            let product = build_product(ctx, &numerator_factors);
            make_log_expr(ctx, first_family, product)
        }
        (true, false) => {
            let product = build_product(ctx, &denominator_factors);
            let inner = make_log_expr(ctx, first_family, product);
            ctx.add(Expr::Neg(inner))
        }
        (false, false) => {
            let numerator = build_product(ctx, &numerator_factors);
            let denominator = build_product(ctx, &denominator_factors);
            let combined_arg = ctx.add(Expr::Div(numerator, denominator));
            make_log_expr(ctx, first_family, combined_arg)
        }
        (true, true) => return None,
    };

    Some(rewritten)
}

pub(crate) fn try_rewrite_log_expansion_target_aware(
    ctx: &mut Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> Option<ExprId> {
    let contracted = try_rewrite_log_contraction_target_aware(ctx, target_expr)?;
    if super::strong_target_match(ctx, contracted, source_expr) {
        Some(target_expr)
    } else {
        None
    }
}

fn extract_scaled_log_term(ctx: &Context, expr: ExprId, sign: Sign) -> Option<ScaledLogTerm> {
    let sign_coeff = match sign {
        Sign::Pos => BigRational::one(),
        Sign::Neg => -BigRational::one(),
    };

    match ctx.get(expr) {
        Expr::Neg(inner) => {
            let mut parsed = extract_scaled_log_term(ctx, *inner, Sign::Pos)?;
            parsed.coeff = -parsed.coeff;
            parsed.coeff *= sign_coeff;
            Some(parsed)
        }
        Expr::Mul(left, right) => {
            let (numeric, log_expr) = match (ctx.get(*left), ctx.get(*right)) {
                (Expr::Number(n), _) => (n.clone(), *right),
                (_, Expr::Number(n)) => (n.clone(), *left),
                _ => return extract_plain_log_term(ctx, expr, sign_coeff),
            };
            let mut parsed = extract_plain_log_term(ctx, log_expr, BigRational::one())?;
            parsed.coeff = sign_coeff * numeric;
            Some(parsed)
        }
        _ => extract_plain_log_term(ctx, expr, sign_coeff),
    }
}

fn extract_plain_log_term(
    ctx: &Context,
    expr: ExprId,
    coeff: BigRational,
) -> Option<ScaledLogTerm> {
    let (family, arg) = match ctx.get(expr) {
        Expr::Function(fn_id, args) if ctx.is_builtin(*fn_id, BuiltinFn::Ln) && args.len() == 1 => {
            (LogFamily::Ln, args[0])
        }
        Expr::Function(fn_id, args) if ctx.is_builtin(*fn_id, BuiltinFn::Log) => {
            match args.as_slice() {
                [arg] => (LogFamily::Log10, *arg),
                [base, arg] => (LogFamily::LogBase(*base), *arg),
                _ => return None,
            }
        }
        _ => return None,
    };

    Some(ScaledLogTerm { family, arg, coeff })
}

fn same_log_family(ctx: &Context, lhs: LogFamily, rhs: LogFamily) -> bool {
    match (lhs, rhs) {
        (LogFamily::Ln, LogFamily::Ln) => true,
        (LogFamily::Log10, LogFamily::Log10) => true,
        (LogFamily::LogBase(lhs), LogFamily::LogBase(rhs)) => {
            compare_expr(ctx, lhs, rhs) == Ordering::Equal
        }
        _ => false,
    }
}

fn make_log_expr(ctx: &mut Context, family: LogFamily, arg: ExprId) -> ExprId {
    match family {
        LogFamily::Ln => ctx.call_builtin(BuiltinFn::Ln, vec![arg]),
        LogFamily::Log10 => ctx.call_builtin(BuiltinFn::Log, vec![arg]),
        LogFamily::LogBase(base) => ctx.call_builtin(BuiltinFn::Log, vec![base, arg]),
    }
}

fn build_product(ctx: &mut Context, factors: &[ExprId]) -> ExprId {
    let mut iter = factors.iter().copied();
    let Some(first) = iter.next() else {
        return ctx.num(1);
    };
    iter.fold(first, |acc, factor| smart_mul(ctx, acc, factor))
}

fn build_powered_log_factor(ctx: &mut Context, arg: ExprId, exponent: &BigRational) -> ExprId {
    if exponent.is_one() {
        return arg;
    }

    if let Some(rewritten) = rewrite_even_abs_power(ctx, arg, exponent) {
        return rewritten;
    }

    if let Some(rewritten) = multiply_existing_power_exponent(ctx, arg, exponent) {
        return rewritten;
    }

    let exponent_id = ctx.add(Expr::Number(exponent.clone()));
    ctx.add(Expr::Pow(arg, exponent_id))
}

fn rewrite_even_abs_power(
    ctx: &mut Context,
    arg: ExprId,
    exponent: &BigRational,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(arg) else {
        return None;
    };
    let fn_id = *fn_id;
    let inner = *args.first()?;
    if !ctx.is_builtin(fn_id, BuiltinFn::Abs) || args.len() != 1 || !exponent.is_integer() {
        return None;
    }

    let integer = exponent.to_integer();
    if integer <= 0.into() || (&integer % 2) != 0.into() {
        return None;
    }

    let exponent_id = ctx.add(Expr::Number(exponent.clone()));
    Some(ctx.add(Expr::Pow(inner, exponent_id)))
}

fn multiply_existing_power_exponent(
    ctx: &mut Context,
    arg: ExprId,
    exponent: &BigRational,
) -> Option<ExprId> {
    let Expr::Pow(base, inner_exp) = ctx.get(arg) else {
        return None;
    };
    let base = *base;
    let inner_exp = *inner_exp;
    let Expr::Number(inner_number) = ctx.get(inner_exp) else {
        return None;
    };

    let multiplied = inner_number.clone() * exponent.clone();
    let exponent_id = ctx.add(Expr::Number(multiplied));
    Some(ctx.add(Expr::Pow(base, exponent_id)))
}

#[cfg(test)]
mod tests {
    use super::{try_rewrite_log_contraction_target_aware, try_rewrite_log_expansion_target_aware};
    use cas_ast::Context;
    use cas_math::semantic_equality::SemanticEqualityChecker;
    use cas_parser::parse;

    #[test]
    fn contracts_log_sum_with_power_arguments() {
        let mut ctx = Context::new();
        let expr = parse("ln(x^3) + ln(y^2)", &mut ctx).expect("expr");
        let rewritten = try_rewrite_log_contraction_target_aware(&mut ctx, expr).expect("rewrite");
        let expected = parse("ln(x^3*y^2)", &mut ctx).expect("expected");
        let checker = SemanticEqualityChecker::new(&ctx);

        assert!(checker.are_equal(rewritten, expected));
    }

    #[test]
    fn contracts_scaled_logs_back_into_powered_argument() {
        let mut ctx = Context::new();
        let expr = parse("3*ln(x) + 2*ln(abs(y))", &mut ctx).expect("expr");
        let rewritten = try_rewrite_log_contraction_target_aware(&mut ctx, expr).expect("rewrite");
        let expected = parse("ln(x^3*y^2)", &mut ctx).expect("expected");
        let checker = SemanticEqualityChecker::new(&ctx);

        assert!(checker.are_equal(rewritten, expected));
    }

    #[test]
    fn contracts_general_base_logs() {
        let mut ctx = Context::new();
        let expr = parse("log(2, x^3) + log(2, y^2)", &mut ctx).expect("expr");
        let rewritten = try_rewrite_log_contraction_target_aware(&mut ctx, expr).expect("rewrite");
        let expected = parse("log(2, x^3*y^2)", &mut ctx).expect("expected");
        let checker = SemanticEqualityChecker::new(&ctx);

        assert!(checker.are_equal(rewritten, expected));
    }

    #[test]
    fn expands_logs_to_power_preserving_target_when_target_contracts_back() {
        let mut ctx = Context::new();
        let source = parse("ln(x^3*y^2)", &mut ctx).expect("source");
        let target = parse("ln(x^3) + ln(y^2)", &mut ctx).expect("target");
        let rewritten =
            try_rewrite_log_expansion_target_aware(&mut ctx, source, target).expect("rewrite");

        assert_eq!(rewritten, target);
    }

    #[test]
    fn rejects_power_preserving_target_that_does_not_contract_to_source() {
        let mut ctx = Context::new();
        let source = parse("ln(x^3*y^2)", &mut ctx).expect("source");
        let target = parse("ln(x^3) + ln(y)", &mut ctx).expect("target");

        assert!(try_rewrite_log_expansion_target_aware(&mut ctx, source, target).is_none());
    }
}
