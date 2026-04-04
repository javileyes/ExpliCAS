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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DeriveLogSimplifyRewriteKind {
    EvenPower,
    Power,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DeriveLogSimplifyRewrite {
    pub(crate) rewritten: ExprId,
    pub(crate) kind: DeriveLogSimplifyRewriteKind,
}

impl DeriveLogSimplifyRewriteKind {
    pub(crate) fn description(self) -> &'static str {
        match self {
            Self::EvenPower => "Recognize an even power inside the logarithm",
            Self::Power => "log(b, x^y) = y * log(b, x)",
        }
    }

    pub(crate) fn rule_name(self) -> &'static str {
        match self {
            Self::EvenPower => "Factor Perfect Square in Logarithm",
            Self::Power => "Evaluate Logarithms",
        }
    }
}

pub(crate) fn try_rewrite_log_simplify_target_aware(
    ctx: &mut Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveLogSimplifyRewrite> {
    if let Some(plan) =
        cas_math::logarithm_inverse_support::try_plan_log_even_power_abs_expr(ctx, expr)
    {
        if super::strong_target_match(ctx, plan.with_abs_rewrite, target_expr) {
            return Some(DeriveLogSimplifyRewrite {
                rewritten: target_expr,
                kind: DeriveLogSimplifyRewriteKind::EvenPower,
            });
        }
    }

    if try_rewrite_log_power_target_aware(ctx, expr, target_expr).is_some() {
        return Some(DeriveLogSimplifyRewrite {
            rewritten: target_expr,
            kind: DeriveLogSimplifyRewriteKind::Power,
        });
    }

    None
}

fn try_rewrite_log_power_target_aware(
    ctx: &mut Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<ExprId> {
    let source = extract_plain_log_term(ctx, expr, BigRational::one())?;
    let target = extract_scaled_log_term(ctx, target_expr, Sign::Pos)?;
    if !same_log_family(ctx, source.family, target.family) {
        return None;
    }

    if !target.coeff.is_integer() || target.coeff <= BigRational::one() {
        return None;
    }

    let Expr::Pow(base, exp) = ctx.get(source.arg) else {
        return None;
    };
    let base = *base;
    let exp = *exp;
    let Expr::Number(source_coeff) = ctx.get(exp) else {
        return None;
    };

    if compare_expr(ctx, base, target.arg) != Ordering::Equal || source_coeff != &target.coeff {
        return None;
    }

    Some(target_expr)
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
    use super::{
        try_rewrite_log_contraction_target_aware, try_rewrite_log_expansion_target_aware,
        try_rewrite_log_simplify_target_aware, DeriveLogSimplifyRewriteKind,
    };
    use cas_ast::Context;
    use cas_math::semantic_equality::SemanticEqualityChecker;
    use cas_parser::parse;

    #[test]
    fn contracts_tabulated_log_targets_aware() {
        let cases = [
            ("ln(x) + ln(y)", "ln(x*y)"),
            ("ln(x) - ln(y)", "ln(x/y)"),
            ("ln(x) + ln(y) - ln(z)", "ln((x*y)/z)"),
            ("2*ln(abs(x)) + ln(y) - ln(z) - ln(t)", "ln((x^2*y)/(z*t))"),
            ("3*ln(x) + 2*ln(abs(y))", "ln(x^3*y^2)"),
            ("3*ln(x) - 2*ln(y)", "ln(x^3/y^2)"),
            ("log(2, x) - log(2, y)", "log(2, x/y)"),
            (
                "2*log(b, x) + 3*log(b, y) - 2*log(b, z) - log(b, t)",
                "log(b, (x^2*y^3)/(z^2*t))",
            ),
            ("3*log(2, x) - 2*log(2, y)", "log(2, x^3/y^2)"),
        ];

        for (expr_text, expected_text) in cases {
            let mut ctx = Context::new();
            let expr = parse(expr_text, &mut ctx).expect("expr");
            let rewritten =
                try_rewrite_log_contraction_target_aware(&mut ctx, expr).expect("rewrite");
            let expected = parse(expected_text, &mut ctx).expect("expected");
            let checker = SemanticEqualityChecker::new(&ctx);

            assert!(
                checker.are_equal(rewritten, expected),
                "expected `{expr_text}` to contract to `{expected_text}`"
            );
        }
    }

    #[test]
    fn expands_tabulated_log_targets_aware() {
        let cases = [
            ("ln(x*y)", "ln(x) + ln(y)"),
            ("ln(x/y)", "ln(x) - ln(y)"),
            ("ln((x*y)/z)", "ln(x) + ln(y) - ln(z)"),
            ("ln((x^2*y)/(z*t))", "2*ln(abs(x)) + ln(y) - ln(z) - ln(t)"),
            ("log(b, (x*y)/z)", "log(b, x) + log(b, y) - log(b, z)"),
            (
                "log(b, (x^2*y^3)/(z^2*t))",
                "2*log(b, x) + 3*log(b, y) - 2*log(b, z) - log(b, t)",
            ),
            ("ln(x^3*y^2)", "ln(x^3) + ln(y^2)"),
        ];

        for (source, target) in cases {
            let mut ctx = Context::new();
            let source = parse(source, &mut ctx).expect("source");
            let target = parse(target, &mut ctx).expect("target");
            let rewritten =
                try_rewrite_log_expansion_target_aware(&mut ctx, source, target).expect("rewrite");

            assert_eq!(rewritten, target);
        }
    }

    #[test]
    fn rejects_log_expansion_target_that_does_not_contract_back() {
        let mut ctx = Context::new();
        let source = parse("ln(x^3*y^2)", &mut ctx).expect("source");
        let target = parse("ln(x^3) + ln(y)", &mut ctx).expect("target");

        assert!(try_rewrite_log_expansion_target_aware(&mut ctx, source, target).is_none());
    }

    #[test]
    fn rewrites_even_log_power_to_abs_target_aware() {
        let mut ctx = Context::new();
        let source = parse("ln(x^4)", &mut ctx).expect("source");
        let target = parse("4*ln(abs(x))", &mut ctx).expect("target");

        let rewrite =
            try_rewrite_log_simplify_target_aware(&mut ctx, source, target).expect("rewrite");

        assert_eq!(rewrite.rewritten, target);
        assert_eq!(rewrite.kind, DeriveLogSimplifyRewriteKind::EvenPower);
    }

    #[test]
    fn rewrites_log_power_to_scaled_log_target_aware() {
        let mut ctx = Context::new();
        let source = parse("log(b, x^3)", &mut ctx).expect("source");
        let target = parse("3*log(b, x)", &mut ctx).expect("target");

        let rewrite =
            try_rewrite_log_simplify_target_aware(&mut ctx, source, target).expect("rewrite");

        assert_eq!(rewrite.rewritten, target);
        assert_eq!(rewrite.kind, DeriveLogSimplifyRewriteKind::Power);
    }
}
