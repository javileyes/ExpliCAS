use cas_ast::{Context, Expr, ExprId};
use cas_math::expr_domain::exprs_equivalent;
use cas_math::polynomial::Polynomial;
use cas_math::root_forms::extract_square_root_base;

const MAX_FRACTION_RESIDUAL_NODES: usize = 96;
const MAX_EXTENDED_FRACTION_RESIDUAL_NODES: usize = 176;
const MAX_EXTENDED_FRACTION_COMPONENT_NODES: usize = 72;
const MAX_DENOMINATOR_POLY_DEGREE: usize = 8;
const MAX_MULTI_FRACTION_RESIDUAL_TERMS: usize = 4;
const MAX_MULTI_FRACTION_LCM_DEGREE: usize = 8;
const MAX_SYMBOLIC_FRACTION_RESIDUAL_NODES: usize = 128;
const MAX_SYMBOLIC_FRACTION_COMBINED_NUMERATOR_NODES: usize = 180;

fn expr_eq(ctx: &Context, left: ExprId, right: ExprId) -> bool {
    cas_ast::ordering::compare_expr(ctx, left, right) == std::cmp::Ordering::Equal
}

fn signed_fraction_term(
    ctx: &mut Context,
    expr: ExprId,
    positive: bool,
) -> Option<(ExprId, ExprId)> {
    let expr = cas_ast::hold::strip_all_holds(ctx, expr);
    match ctx.get(expr).clone() {
        Expr::Neg(inner) => signed_fraction_term(ctx, inner, !positive),
        Expr::Div(num, den) if positive => Some((num, den)),
        Expr::Div(num, den) => {
            let signed_num = ctx.add(Expr::Neg(num));
            Some((signed_num, den))
        }
        _ => None,
    }
}

fn signed_fraction_pair(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<((ExprId, ExprId), (ExprId, ExprId))> {
    match ctx.get(expr).clone() {
        Expr::Add(left, right) => Some((
            signed_fraction_term(ctx, left, true)?,
            signed_fraction_term(ctx, right, true)?,
        )),
        Expr::Sub(left, right) => Some((
            signed_fraction_term(ctx, left, true)?,
            signed_fraction_term(ctx, right, false)?,
        )),
        _ => None,
    }
}

fn collect_signed_fraction_terms(
    ctx: &mut Context,
    expr: ExprId,
    positive: bool,
    terms: &mut Vec<(ExprId, ExprId)>,
) -> Option<()> {
    if terms.len() >= MAX_MULTI_FRACTION_RESIDUAL_TERMS {
        return None;
    }

    let expr = cas_ast::hold::strip_all_holds(ctx, expr);
    match ctx.get(expr).clone() {
        Expr::Add(left, right) => {
            collect_signed_fraction_terms(ctx, left, positive, terms)?;
            collect_signed_fraction_terms(ctx, right, positive, terms)
        }
        Expr::Sub(left, right) => {
            collect_signed_fraction_terms(ctx, left, positive, terms)?;
            collect_signed_fraction_terms(ctx, right, !positive, terms)
        }
        Expr::Neg(inner) => collect_signed_fraction_terms(ctx, inner, !positive, terms),
        Expr::Div(num, den) if positive => {
            terms.push((num, den));
            Some(())
        }
        Expr::Div(num, den) => {
            let signed_num = ctx.add(Expr::Neg(num));
            terms.push((signed_num, den));
            Some(())
        }
        _ => None,
    }
}

fn collect_signed_rational_terms(
    ctx: &mut Context,
    expr: ExprId,
    positive: bool,
    terms: &mut Vec<(ExprId, ExprId)>,
) -> Option<()> {
    if terms.len() >= MAX_MULTI_FRACTION_RESIDUAL_TERMS {
        return None;
    }

    let expr = cas_ast::hold::strip_all_holds(ctx, expr);
    match ctx.get(expr).clone() {
        Expr::Add(left, right) => {
            collect_signed_rational_terms(ctx, left, positive, terms)?;
            collect_signed_rational_terms(ctx, right, positive, terms)
        }
        Expr::Sub(left, right) => {
            collect_signed_rational_terms(ctx, left, positive, terms)?;
            collect_signed_rational_terms(ctx, right, !positive, terms)
        }
        Expr::Neg(inner) => collect_signed_rational_terms(ctx, inner, !positive, terms),
        Expr::Div(num, den) if positive => {
            terms.push((num, den));
            Some(())
        }
        Expr::Div(num, den) => {
            let signed_num = ctx.add(Expr::Neg(num));
            terms.push((signed_num, den));
            Some(())
        }
        _ if positive => {
            let one = ctx.num(1);
            terms.push((expr, one));
            Some(())
        }
        _ => {
            let one = ctx.num(1);
            let signed_num = ctx.add(Expr::Neg(expr));
            terms.push((signed_num, one));
            Some(())
        }
    }
}

fn fraction_pair_within_budget(
    ctx: &Context,
    expr: ExprId,
    pair: &((ExprId, ExprId), (ExprId, ExprId)),
) -> bool {
    let total_nodes = cas_ast::count_nodes(ctx, expr);
    if total_nodes <= MAX_FRACTION_RESIDUAL_NODES {
        return true;
    }
    if total_nodes > MAX_EXTENDED_FRACTION_RESIDUAL_NODES {
        return false;
    }

    let ((left_num, left_den), (right_num, right_den)) = *pair;
    [left_num, left_den, right_num, right_den]
        .into_iter()
        .all(|part| cas_ast::count_nodes(ctx, part) <= MAX_EXTENDED_FRACTION_COMPONENT_NODES)
}

fn single_shared_variable(ctx: &Context, left: ExprId, right: ExprId) -> Option<String> {
    let left_vars = cas_ast::collect_variables(ctx, left);
    let right_vars = cas_ast::collect_variables(ctx, right);
    if left_vars.len() != 1 || left_vars != right_vars {
        return None;
    }
    left_vars.into_iter().next()
}

fn polynomial_denominators_match(ctx: &Context, left: ExprId, right: ExprId) -> bool {
    if expr_eq(ctx, left, right) {
        return true;
    }

    let Some(var) = single_shared_variable(ctx, left, right) else {
        return false;
    };
    let Ok(left_poly) = Polynomial::from_expr(ctx, left, &var) else {
        return false;
    };
    let Ok(right_poly) = Polynomial::from_expr(ctx, right, &var) else {
        return false;
    };

    left_poly.degree() <= MAX_DENOMINATOR_POLY_DEGREE
        && right_poly.degree() <= MAX_DENOMINATOR_POLY_DEGREE
        && left_poly == right_poly
}

fn signed_numerators_cancel(ctx: &mut Context, left: ExprId, right: ExprId) -> bool {
    let neg_right = ctx.add(Expr::Neg(right));
    if expr_eq(ctx, left, neg_right) || exprs_equivalent(ctx, left, neg_right) {
        return true;
    }

    let sum = ctx.add(Expr::Add(left, right));
    if crate::polynomial_identity_support::try_prove_polynomial_identity_zero_expr(ctx, sum)
        .is_some()
    {
        return true;
    }

    polynomial_sqrt_scaled_terms_equivalent(ctx, left, neg_right)
}

fn signed_fraction_cross_products_cancel(
    ctx: &mut Context,
    left_num: ExprId,
    left_den: ExprId,
    right_num: ExprId,
    right_den: ExprId,
) -> bool {
    let vars = [
        cas_ast::collect_variables(ctx, left_num),
        cas_ast::collect_variables(ctx, left_den),
        cas_ast::collect_variables(ctx, right_num),
        cas_ast::collect_variables(ctx, right_den),
    ]
    .into_iter()
    .flatten()
    .collect::<std::collections::BTreeSet<_>>();
    if vars.len() != 1 {
        return false;
    }
    let Some(var) = vars.into_iter().next() else {
        return false;
    };

    let Ok(left_num_poly) = Polynomial::from_expr(ctx, left_num, &var) else {
        return false;
    };
    let Ok(left_den_poly) = Polynomial::from_expr(ctx, left_den, &var) else {
        return false;
    };
    let Ok(right_num_poly) = Polynomial::from_expr(ctx, right_num, &var) else {
        return false;
    };
    let Ok(right_den_poly) = Polynomial::from_expr(ctx, right_den, &var) else {
        return false;
    };

    if left_den_poly.is_zero()
        || right_den_poly.is_zero()
        || left_den_poly.degree() > MAX_DENOMINATOR_POLY_DEGREE
        || right_den_poly.degree() > MAX_DENOMINATOR_POLY_DEGREE
    {
        return false;
    }

    let left_cross = left_num_poly.mul(&right_den_poly);
    let right_cross = right_num_poly.mul(&left_den_poly);
    let combined = left_cross.add(&right_cross);
    combined.degree() <= MAX_MULTI_FRACTION_LCM_DEGREE && combined.is_zero()
}

fn polynomial_fraction_sum_residual_zero(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    if cas_ast::count_nodes(ctx, expr) > MAX_EXTENDED_FRACTION_RESIDUAL_NODES {
        return None;
    }

    let mut terms = Vec::new();
    collect_signed_fraction_terms(ctx, expr, true, &mut terms)?;
    if terms.len() < 3 || terms.len() > MAX_MULTI_FRACTION_RESIDUAL_TERMS {
        return None;
    }
    if !terms.iter().all(|(num, den)| {
        cas_ast::count_nodes(ctx, *num) <= MAX_EXTENDED_FRACTION_COMPONENT_NODES
            && cas_ast::count_nodes(ctx, *den) <= MAX_EXTENDED_FRACTION_COMPONENT_NODES
    }) {
        return None;
    }

    let vars = cas_ast::collect_variables(ctx, expr);
    if vars.len() != 1 {
        return None;
    }
    let var = vars.into_iter().next()?;

    let mut parsed_terms = Vec::with_capacity(terms.len());
    for (num, den) in terms {
        let numerator = Polynomial::from_expr(ctx, num, &var).ok()?;
        let denominator = Polynomial::from_expr(ctx, den, &var).ok()?;
        if denominator.is_zero() || denominator.degree() > MAX_DENOMINATOR_POLY_DEGREE {
            return None;
        }
        parsed_terms.push((numerator, denominator));
    }

    let mut lcm = Polynomial::one(var.clone());
    for (_, denominator) in &parsed_terms {
        let gcd = lcm.gcd(denominator);
        let product = lcm.mul(denominator);
        let (quotient, remainder) = product.div_rem(&gcd).ok()?;
        if !remainder.is_zero() || quotient.degree() > MAX_MULTI_FRACTION_LCM_DEGREE {
            return None;
        }
        lcm = quotient;
    }

    let mut combined = Polynomial::zero(var);
    for (numerator, denominator) in parsed_terms {
        let (scale, remainder) = lcm.div_rem(&denominator).ok()?;
        if !remainder.is_zero() {
            return None;
        }
        combined = combined.add(&numerator.mul(&scale));
    }

    combined.is_zero().then(|| ctx.num(0))
}

fn symbolic_polynomial_denominator_sum_residual_zero(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    if cas_ast::count_nodes(ctx, expr) > MAX_SYMBOLIC_FRACTION_RESIDUAL_NODES {
        return None;
    }

    let mut terms = Vec::new();
    collect_signed_rational_terms(ctx, expr, true, &mut terms)?;
    if terms.len() < 2 || terms.len() > MAX_MULTI_FRACTION_RESIDUAL_TERMS {
        return None;
    }
    if !terms.iter().all(|(num, den)| {
        cas_ast::count_nodes(ctx, *num) <= MAX_EXTENDED_FRACTION_COMPONENT_NODES
            && cas_ast::count_nodes(ctx, *den) <= MAX_EXTENDED_FRACTION_COMPONENT_NODES
    }) {
        return None;
    }

    let vars = cas_ast::collect_variables(ctx, expr);
    if vars.len() != 1 {
        return None;
    }
    let var = vars.into_iter().next()?;

    let mut has_non_constant_denominator = false;
    let mut parsed_terms = Vec::with_capacity(terms.len());
    for (num, den) in terms {
        let denominator = Polynomial::from_expr(ctx, den, &var).ok()?;
        if denominator.is_zero() || denominator.degree() > MAX_DENOMINATOR_POLY_DEGREE {
            return None;
        }
        has_non_constant_denominator |= denominator.degree() > 0;
        parsed_terms.push((num, denominator));
    }
    if !has_non_constant_denominator {
        return None;
    }

    let mut lcm = Polynomial::one(var.clone());
    for (_, denominator) in &parsed_terms {
        let gcd = lcm.gcd(denominator);
        let product = lcm.mul(denominator);
        let (quotient, remainder) = product.div_rem(&gcd).ok()?;
        if !remainder.is_zero() || quotient.degree() > MAX_MULTI_FRACTION_LCM_DEGREE {
            return None;
        }
        lcm = quotient;
    }

    let mut combined_terms = Vec::with_capacity(parsed_terms.len());
    for (numerator, denominator) in parsed_terms {
        let (scale, remainder) = lcm.div_rem(&denominator).ok()?;
        if !remainder.is_zero() {
            return None;
        }
        let scale_expr = scale.to_expr(ctx);
        combined_terms.push(ctx.add(Expr::Mul(numerator, scale_expr)));
    }

    let combined = cas_math::expr_nary::build_balanced_add(ctx, &combined_terms);
    if cas_ast::count_nodes(ctx, combined) > MAX_SYMBOLIC_FRACTION_COMBINED_NUMERATOR_NODES {
        return None;
    }
    crate::polynomial_identity_support::try_prove_polynomial_identity_zero_expr(ctx, combined)
        .map(|_| ctx.num(0))
}

fn polynomial_sqrt_scaled_terms_equivalent(ctx: &mut Context, left: ExprId, right: ExprId) -> bool {
    let (left_negated, left) = strip_outer_neg(ctx, left);
    let (right_negated, right) = strip_outer_neg(ctx, right);
    let left_factors = cas_math::expr_nary::mul_factors(ctx, left);
    let right_factors = cas_math::expr_nary::mul_factors(ctx, right);
    if left_factors.len() > 4 || right_factors.len() > 4 {
        return false;
    }

    let Some((left_radicand, left_rest)) = split_single_sqrt_factor(ctx, left_factors.as_slice())
    else {
        return false;
    };
    let Some((right_radicand, right_rest)) =
        split_single_sqrt_factor(ctx, right_factors.as_slice())
    else {
        return false;
    };
    if !polynomial_denominators_match(ctx, left_radicand, right_radicand) {
        return false;
    }

    let left_rest_expr = build_signed_product_or_one(ctx, &left_rest, left_negated);
    let right_rest_expr = build_signed_product_or_one(ctx, &right_rest, right_negated);
    if expr_eq(ctx, left_rest_expr, right_rest_expr)
        || exprs_equivalent(ctx, left_rest_expr, right_rest_expr)
    {
        return true;
    }

    let diff = ctx.add(Expr::Sub(left_rest_expr, right_rest_expr));
    crate::polynomial_identity_support::try_prove_polynomial_identity_zero_expr(ctx, diff).is_some()
}

fn strip_outer_neg(ctx: &Context, expr: ExprId) -> (bool, ExprId) {
    match ctx.get(expr) {
        Expr::Neg(inner) => (true, *inner),
        _ => (false, expr),
    }
}

fn split_single_sqrt_factor(ctx: &Context, factors: &[ExprId]) -> Option<(ExprId, Vec<ExprId>)> {
    let mut sqrt_radicand = None;
    let mut rest = Vec::with_capacity(factors.len().saturating_sub(1));
    for &factor in factors {
        if let Some(radicand) = extract_square_root_base(ctx, factor) {
            if sqrt_radicand.is_some() {
                return None;
            }
            sqrt_radicand = Some(radicand);
        } else {
            rest.push(factor);
        }
    }

    Some((sqrt_radicand?, rest))
}

fn build_signed_product_or_one(ctx: &mut Context, factors: &[ExprId], negated: bool) -> ExprId {
    let product = if factors.is_empty() {
        ctx.num(1)
    } else {
        cas_math::expr_nary::build_balanced_mul(ctx, factors)
    };
    if negated {
        ctx.add(Expr::Neg(product))
    } else {
        product
    }
}

pub(crate) fn try_polynomial_denominator_fraction_residual_zero(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    if let Expr::Div(num, den) = ctx.get(expr).clone() {
        let denominator = cas_ast::hold::strip_all_holds(ctx, den);
        if matches!(ctx.get(denominator), Expr::Number(_))
            && try_polynomial_denominator_fraction_residual_zero(ctx, num).is_some()
        {
            return Some(ctx.num(0));
        }
    }

    let Some(pair) = signed_fraction_pair(ctx, expr) else {
        return polynomial_fraction_sum_residual_zero(ctx, expr)
            .or_else(|| symbolic_polynomial_denominator_sum_residual_zero(ctx, expr));
    };
    if !fraction_pair_within_budget(ctx, expr, &pair) {
        return None;
    }

    let ((left_num, left_den), (right_num, right_den)) = pair;
    if polynomial_denominators_match(ctx, left_den, right_den) {
        if signed_numerators_cancel(ctx, left_num, right_num) {
            return Some(ctx.num(0));
        }
    } else if signed_fraction_cross_products_cancel(ctx, left_num, left_den, right_num, right_den) {
        return Some(ctx.num(0));
    }

    symbolic_polynomial_denominator_sum_residual_zero(ctx, expr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    fn residual_result(input: &str) -> Option<String> {
        let mut ctx = Context::new();
        let expr = parse(input, &mut ctx)
            .unwrap_or_else(|err| panic!("parse failed for {input}: {err:?}"));
        try_polynomial_denominator_fraction_residual_zero(&mut ctx, expr).map(|result| {
            DisplayExpr {
                context: &ctx,
                id: result,
            }
            .to_string()
        })
    }

    fn simplified_result(input: &str) -> String {
        let mut simplifier = crate::Simplifier::with_default_rules();
        simplifier.steps_mode = crate::options::StepsMode::Off;
        let expr = parse(input, &mut simplifier.context)
            .unwrap_or_else(|err| panic!("parse failed for {input}: {err:?}"));
        let (result, _) = simplifier.simplify(expr);
        DisplayExpr {
            context: &simplifier.context,
            id: result,
        }
        .to_string()
    }

    #[test]
    fn polynomial_equivalent_denominator_fraction_residual_cancels() {
        assert_eq!(
            residual_result(
                "sqrt(5)*(-2*x-1)/(x^4+2*x^3+3*x^2+2*x+6) + sqrt(5)*(2*x+1)/((x^2+x+1)^2+5)"
            ),
            Some("0".to_string())
        );
    }

    #[test]
    fn polynomial_equivalent_denominator_fraction_residual_rejects_nonzero_pair() {
        assert_eq!(
            residual_result(
                "sqrt(5)*(-2*x-1)/(x^4+2*x^3+3*x^2+2*x+6) + sqrt(5)*(2*x+2)/((x^2+x+1)^2+5)"
            ),
            None
        );
    }

    #[test]
    fn polynomial_equivalent_expanded_radical_denominator_fraction_residual_cancels() {
        assert_eq!(
            residual_result(
                "sqrt(x^4+2*x*x^2+x^2+6*x^2+6*x+9-1)*(2*x+1)/((x^4+2*x*x^2+x^2+6*x^2+6*x+9-1)*(x^2+x+3)) - sqrt(x^4+2*x^3+7*x^2+6*x+8)*(2*x+1)/((x^2+x+3)*(x^4+2*x^3+7*x^2+6*x+8))"
            ),
            Some("0".to_string())
        );
    }

    #[test]
    fn polynomial_equivalent_expanded_radical_denominator_fraction_residual_cancels_negated_orientation(
    ) {
        assert_eq!(
            residual_result(
                "sqrt(x^4+2*x*x^2+x^2+6*x^2+6*x+9-1)*(-2*x-1)/((x^4+2*x*x^2+x^2+6*x^2+6*x+9-1)*(x^2+x+3)) + sqrt(x^4+2*x^3+7*x^2+6*x+8)*(2*x+1)/((x^2+x+3)*(x^4+2*x^3+7*x^2+6*x+8))"
            ),
            Some("0".to_string())
        );
    }

    #[test]
    fn polynomial_equivalent_expanded_radical_denominator_fraction_residual_rejects_nonzero_pair() {
        assert_eq!(
            residual_result(
                "sqrt(x^4+2*x*x^2+x^2+6*x^2+6*x+9-1)*(2*x+1)/((x^4+2*x*x^2+x^2+6*x^2+6*x+9-1)*(x^2+x+3)) - sqrt(x^4+2*x^3+7*x^2+6*x+8)*(2*x+2)/((x^2+x+3)*(x^4+2*x^3+7*x^2+6*x+8))"
            ),
            None
        );
    }

    #[test]
    fn polynomial_fraction_sum_residual_cancels_scaled_quadratic_square_derivative() {
        assert_eq!(
            residual_result("2/(4*(4*x^2+1)) + (2 - 8*x^2)/(8*x^2 + 2)^2 - 1/(4*x^2 + 1)^2"),
            Some("0".to_string())
        );
    }

    #[test]
    fn polynomial_fraction_sum_residual_cancels_shifted_quadratic_cubic_repeated_pole() {
        let input = "1/(x+1)^3 + (x+1)/(x^2+2*x+2) + (-x^4-4*x^3-7*x^2-6*x-3)/(x^5+5*x^4+11*x^3+13*x^2+8*x+2)";
        assert_eq!(residual_result(input), Some("0".to_string()));
        assert_eq!(simplified_result(input), "0");
    }

    #[test]
    fn polynomial_fraction_sum_residual_cancels_quadratic_fourth_power_derivative() {
        let input = "3/(8*(x^2+1)) + (72*x^6+184*x^4+152*x^2+40 - 16*x*(2*x^2+2)*(3*x^3+5*x))/(64*(x^2+1)^4) - x^4/(x^2+1)^3 - (1-x^2)/(x^2+1)^2";
        assert_eq!(residual_result(input), Some("0".to_string()));
        assert_eq!(simplified_result(input), "0");
    }

    #[test]
    fn symbolic_polynomial_denominator_residual_cancels_arctan_coefficient() {
        let input = "(arctan(x)*(84*x^10 + 168*x^8 + 84*x^6) + x^2*(12*x^3 + 12*x) + 12*x^3 + 12*x)/(x^4 + 2*x^2 + 1) - 84*arctan(x)*x^6 - 12*x";
        assert_eq!(residual_result(input), Some("0".to_string()));
    }

    #[test]
    fn polynomial_fraction_sum_residual_rejects_nonzero_scaled_quadratic_square_derivative() {
        assert_eq!(
            residual_result("2/(4*(4*x^2+1)) + (3 - 8*x^2)/(8*x^2 + 2)^2 - 1/(4*x^2 + 1)^2"),
            None
        );
    }

    #[test]
    fn polynomial_fraction_pair_residual_cancels_scaled_denominator_cross_product() {
        assert_eq!(
            residual_result(
                "(81/128*x + 135/128)/(27/128*x^3 + 27/128 - 27/128*x^2 - 27/128*x) - (3*x+5)/(x^3-x^2-x+1)"
            ),
            Some("0".to_string())
        );
    }

    #[test]
    fn polynomial_fraction_pair_residual_rejects_nonzero_scaled_denominator_cross_product() {
        assert_eq!(
            residual_result(
                "(81/128*x + 136/128)/(27/128*x^3 + 27/128 - 27/128*x^2 - 27/128*x) - (3*x+5)/(x^3-x^2-x+1)"
            ),
            None
        );
    }
}
