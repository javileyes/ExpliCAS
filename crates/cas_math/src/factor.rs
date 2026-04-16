use crate::build::mul2_raw;
use crate::poly_compare::poly_eq;
use crate::polynomial::Polynomial;
use crate::trig_roots_flatten::{get_square_root, get_trig_arg, is_trig_pow};
use cas_ast::ordering::compare_expr;
use cas_ast::{Context, Expr, ExprId};
use num_traits::{One, Signed, ToPrimitive};
// use num_rational::BigRational;
use std::cmp::Ordering;

/// Factors an expression.
/// This is the main entry point for factorization.
pub fn factor(ctx: &mut Context, expr: ExprId) -> ExprId {
    // 1. Try polynomial factorization
    if let Some(res) = factor_polynomial(ctx, expr) {
        return res;
    }

    // 2. Try the alternating cubic Vandermonde identity:
    // a^3(b-c) + b^3(c-a) + c^3(a-b) = (a-b)(a-c)(b-c)(a+b+c)
    if let Some(res) = factor_alternating_cubic_vandermonde(ctx, expr) {
        return res;
    }

    // 3. Try difference of squares
    if let Some(res) = factor_difference_squares(ctx, expr) {
        return res;
    }

    // 4. Try Sophie Germain identity: a^4 + 4b^4 = (a² + 2ab + 2b²)(a² - 2ab + 2b²)
    if let Some(res) = factor_sophie_germain(ctx, expr) {
        return res;
    }

    // 5. Try perfect square trinomial: a² ± 2ab + b² = (a ± b)²
    if let Some(res) = factor_perfect_square_trinomial(ctx, expr) {
        return res;
    }

    // 5. Recursive factorization?
    // For now, just return original if no top-level factorization applies.
    // Ideally we should factor sub-expressions too.
    // But `factor` usually means "factor this polynomial".
    // Let's stick to top-level for now, or maybe recurse if it's a product/sum?

    expr
}

/// Factors the alternating cubic Vandermonde identity:
/// `a^3(b-c) + b^3(c-a) + c^3(a-b) = (a-b)(a-c)(b-c)(a+b+c)`.
///
/// This matcher is intentionally narrow: it recognizes the 3-variable
/// alternating quartic up to algebraic expansion/reordering, without trying to
/// be a full multivariate factorizer.
pub fn factor_alternating_cubic_vandermonde(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let vars = cas_ast::collect_variables(ctx, expr);
    if vars.len() != 3 {
        return None;
    }

    let mut vars: Vec<_> = vars.into_iter().collect();
    vars.sort();

    let a = ctx.var(&vars[0]);
    let b = ctx.var(&vars[1]);
    let c = ctx.var(&vars[2]);

    let a_minus_b = ctx.add(Expr::Sub(a, b));
    let a_minus_c = ctx.add(Expr::Sub(a, c));
    let b_minus_c = ctx.add(Expr::Sub(b, c));
    let a_plus_b = ctx.add(Expr::Add(a, b));
    let a_plus_b_plus_c = ctx.add(Expr::Add(a_plus_b, c));

    let left = mul2_raw(ctx, a_minus_b, a_minus_c);
    let right = mul2_raw(ctx, b_minus_c, a_plus_b_plus_c);
    let factored = mul2_raw(ctx, left, right);

    if poly_eq(ctx, expr, factored) {
        Some(factored)
    } else {
        None
    }
}

/// Factors a polynomial expression using rational roots.
pub fn factor_polynomial(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let vars = cas_ast::collect_variables(ctx, expr);
    if vars.len() != 1 {
        return None;
    }
    let var = vars.iter().next()?;

    if let Ok(poly) = Polynomial::from_expr(ctx, expr, var) {
        if poly.is_zero() {
            return None;
        }

        // 1. Extract content (common constant factor)
        let factors = poly.factor_rational_roots();

        if factors.len() == 1 {
            // Irreducible (over rationals) or just trivial
            let content = poly.content();
            let min_deg = poly.min_degree();
            if content.is_one() && min_deg == 0 {
                return None; // No change
            }
        }

        // Group identical factors into powers
        let mut counts: Vec<(Polynomial, u32)> = Vec::new();
        for f in factors {
            if let Some((_, count)) = counts.iter_mut().find(|(p, _)| p == &f) {
                *count += 1;
            } else {
                counts.push((f, 1));
            }
        }

        // Construct expression
        let mut terms = Vec::new();
        for (p, count) in counts {
            let base = p.to_expr(ctx);
            if count == 1 {
                terms.push(base);
            } else {
                let exp = ctx.num(count as i64);
                terms.push(ctx.add(Expr::Pow(base, exp)));
            }
        }

        if terms.is_empty() {
            return None;
        }

        let mut res = terms[0];
        for t in terms.iter().skip(1) {
            res = mul2_raw(ctx, res, *t);
        }

        // println!("factor_polynomial: {} -> {}", cas_formatter::DisplayExpr { context: ctx, id: expr }, cas_formatter::DisplayExpr { context: ctx, id: res });

        return Some(res);
    }
    None
}

/// Factors difference of squares: a^2 - b^2 -> (a-b)(a+b)
pub fn factor_difference_squares(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let expr_data = ctx.get(expr).clone();
    let (l, r) = match expr_data {
        Expr::Sub(l, r) => (l, r),
        Expr::Add(a, b) => {
            // Check if one is negative
            if is_negative_term(ctx, b) {
                (a, negate_term(ctx, b))
            } else if is_negative_term(ctx, a) {
                (b, negate_term(ctx, a))
            } else {
                return None;
            }
        }
        _ => return None,
    };

    // println!("factor_diff_squares checking: {:?} - {:?}", l, r);
    // println!("Left structure: {:?}", ctx.get(l));
    // if let Expr::Mul(a, b) = ctx.get(l) {
    //    println!("  Mul children: {:?} and {:?}", ctx.get(*a), ctx.get(*b));
    // }
    let root_l_opt = get_square_root(ctx, l);
    let root_r_opt = get_square_root(ctx, r);
    // println!("Roots: {:?}, {:?}", root_l_opt, root_r_opt);

    if let (Some(root_l), Some(root_r)) = (root_l_opt, root_r_opt) {
        // a^2 - b^2 = (a - b)(a + b)
        let term1 = ctx.add(Expr::Sub(root_l, root_r));

        // Check for Pythagorean identity in term2 (a + b)
        // sin^2 + cos^2 = 1
        let mut term2 = ctx.add(Expr::Add(root_l, root_r));
        let mut is_pythagorean = false;

        if is_sin_cos_pair(ctx, root_l, root_r) {
            term2 = ctx.num(1);
            is_pythagorean = true;
        }

        let new_expr = if is_pythagorean {
            term1
        } else {
            mul2_raw(ctx, term1, term2)
        };

        return Some(new_expr);
    }
    None
}

/// Factors Sophie Germain identity: a^4 + 4b^4 = (a² + 2ab + 2b²)(a² - 2ab + 2b²)
/// Example: x^4 + 64 = x^4 + 4·16 = x^4 + 4·2^4 → (x² + 4x + 8)(x² - 4x + 8)
pub fn factor_sophie_germain(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    fn pow_four_base(ctx: &Context, term: ExprId) -> Option<ExprId> {
        match ctx.get(term) {
            Expr::Pow(base, exp) => match ctx.get(*exp) {
                Expr::Number(n) if n.is_integer() && *n.numer() == 4.into() => Some(*base),
                _ => None,
            },
            _ => None,
        }
    }

    fn four_times_fourth_power_base(ctx: &mut Context, term: ExprId) -> Option<ExprId> {
        match ctx.get(term).clone() {
            Expr::Mul(l, r) => {
                let left_is_four = matches!(ctx.get(l), Expr::Number(n) if n.is_integer() && *n.numer() == 4.into());
                let right_is_four = matches!(ctx.get(r), Expr::Number(n) if n.is_integer() && *n.numer() == 4.into());
                if left_is_four {
                    return pow_four_base(ctx, r);
                }
                if right_is_four {
                    return pow_four_base(ctx, l);
                }
                None
            }
            Expr::Number(n) => {
                if !n.is_integer() || !n.is_positive() {
                    return None;
                }
                let k = n.numer().to_i64()?;
                if k % 4 != 0 {
                    return None;
                }
                let b4 = k / 4;
                let b_f64 = (b4 as f64).powf(0.25);
                let b_int = b_f64.round() as i64;
                if b_int.pow(4) != b4 {
                    return None;
                }
                Some(ctx.num(b_int))
            }
            _ => None,
        }
    }

    fn build_sophie_germain_factors(ctx: &mut Context, a: ExprId, b: ExprId) -> ExprId {
        let two = ctx.num(2);
        let a_sq = ctx.add(Expr::Pow(a, two));

        let (two_ab, two_b_sq) = match ctx.get(b).clone() {
            Expr::Number(n) if n.is_integer() && n.is_positive() => {
                if let Some(b_val) = n.numer().to_i64() {
                    let two_b = ctx.num(2 * b_val);
                    let two_b_sq = ctx.num(2 * b_val * b_val);
                    (mul2_raw(ctx, two_b, a), two_b_sq)
                } else {
                    let b_sq = ctx.add(Expr::Pow(b, two));
                    let ab = mul2_raw(ctx, a, b);
                    (mul2_raw(ctx, two, ab), mul2_raw(ctx, two, b_sq))
                }
            }
            _ => {
                let b_sq = ctx.add(Expr::Pow(b, two));
                let ab = mul2_raw(ctx, a, b);
                (mul2_raw(ctx, two, ab), mul2_raw(ctx, two, b_sq))
            }
        };

        let common = ctx.add(Expr::Add(a_sq, two_b_sq));
        let factor1 = ctx.add(Expr::Add(common, two_ab));
        let factor2 = ctx.add(Expr::Sub(common, two_ab));
        mul2_raw(ctx, factor1, factor2)
    }

    let expr_data = ctx.get(expr).clone();

    let (left, right) = match expr_data {
        Expr::Add(l, r) => (l, r),
        _ => return None,
    };

    fn try_match(
        ctx: &mut Context,
        term_pow: ExprId,
        term_other: ExprId,
    ) -> Option<(ExprId, ExprId)> {
        Some((
            pow_four_base(ctx, term_pow)?,
            four_times_fourth_power_base(ctx, term_other)?,
        ))
    }

    let (a, b) = try_match(ctx, left, right).or_else(|| try_match(ctx, right, left))?;
    Some(build_sophie_germain_factors(ctx, a, b))
}

/// Factors perfect square trinomials: a² ± 2ab + b² = (a ± b)²
/// Examples:
///   x² + 2xy + y² = (x + y)²
///   x² - 2xy + y² = (x - y)²
pub fn factor_perfect_square_trinomial(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    use crate::expr_nary::{AddView, Sign};

    if let Some((a, b, is_sub)) =
        crate::perfect_square_support::try_match_perfect_square_trinomial(ctx, expr)
    {
        return Some(build_canonical_perfect_square(ctx, a, b, is_sub));
    }

    // Collect all additive terms
    let add_view = AddView::from_expr(ctx, expr);

    if add_view.terms.len() != 3 {
        return None;
    }

    // We need to find: a², b², and ±2ab
    // Strategy: try each pair as (a², b²) and see if third term is ±2ab

    for i in 0..3 {
        for j in 0..3 {
            if i == j {
                continue;
            }
            let k = 3 - i - j; // the remaining index

            // Get terms with their signs
            let (term_i, sign_i) = add_view.terms[i];
            let (term_j, sign_j) = add_view.terms[j];
            let (term_k, sign_k) = add_view.terms[k];

            // Both squared terms should be positive
            if sign_i != Sign::Pos || sign_j != Sign::Pos {
                continue;
            }

            // Extract square roots if they exist
            let a = match get_square_root_base(ctx, term_i) {
                Some(a) => a,
                None => continue,
            };
            let b = match get_square_root_base(ctx, term_j) {
                Some(b) => b,
                None => continue,
            };

            // Check if term_k is ±2ab (structurally) and get the embedded sign
            let embedded_positive = match is_2ab_term(ctx, term_k, a, b) {
                Some(positive) => positive,
                None => continue,
            };

            // Determine final sign: combine AddView sign with embedded coefficient sign
            // - If sign_k is Neg AND embedded is positive (+2): result is negative → (a-b)²
            // - If sign_k is Pos AND embedded is positive (+2): result is positive → (a+b)²
            // - If sign_k is Pos AND embedded is negative (-2): result is negative → (a-b)²
            // - If sign_k is Neg AND embedded is negative (-2): result is positive → (a+b)²
            let is_positive_term = match (sign_k, embedded_positive) {
                (Sign::Pos, true) => true,   // +2ab
                (Sign::Neg, true) => false,  // -(+2ab) = -2ab
                (Sign::Pos, false) => false, // -2ab
                (Sign::Neg, false) => true,  // -(-2ab) = +2ab
            };

            // Found! Build (a ± b)²
            return Some(build_canonical_perfect_square(ctx, a, b, !is_positive_term));
        }
    }

    None
}

fn build_canonical_perfect_square(ctx: &mut Context, a: ExprId, b: ExprId, is_sub: bool) -> ExprId {
    let two = ctx.num(2);
    let binomial = if is_sub {
        match compare_expr(ctx, a, b) {
            Ordering::Less => ctx.add(Expr::Sub(b, a)),
            _ => ctx.add(Expr::Sub(a, b)),
        }
    } else {
        ctx.add(Expr::Add(a, b))
    };
    ctx.add(Expr::Pow(binomial, two))
}

/// Helper: if expr is x^2, return x; otherwise None
fn get_square_root_base(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Pow(base, exp) => {
            if let Expr::Number(n) = ctx.get(*exp) {
                if n.is_integer() && *n.numer() == 2.into() {
                    return Some(*base);
                }
            }
            None
        }
        _ => None,
    }
}

/// Helper: check if expr is ±2*a*b OR a*b (when 2 is the coefficient in AddView)
/// Returns Some(true) for positive coef (+2), Some(false) for negative coef (-2), None if no match
fn is_2ab_term(ctx: &Context, expr: ExprId, a: ExprId, b: ExprId) -> Option<bool> {
    use crate::expr_nary::MulView;

    let mul_view = MulView::from_expr(ctx, expr);
    let factors = &mul_view.factors;

    // Case 1: 3 factors - ±2, a, b (all combined)
    if factors.len() == 3 {
        let mut coef_sign: Option<bool> = None;
        let mut has_a = false;
        let mut has_b = false;

        for &f in factors.iter() {
            if let Expr::Number(n) = ctx.get(f) {
                if n.is_integer() {
                    if *n.numer() == 2.into() {
                        coef_sign = Some(true); // positive
                        continue;
                    } else if *n.numer() == (-2).into() {
                        coef_sign = Some(false); // negative
                        continue;
                    }
                }
            }
            if compare_expr(ctx, f, a) == std::cmp::Ordering::Equal {
                has_a = true;
            } else if compare_expr(ctx, f, b) == std::cmp::Ordering::Equal {
                has_b = true;
            }
        }

        if coef_sign.is_some() && has_a && has_b {
            return coef_sign;
        }
    }

    None
}

fn is_sin_cos_pair(ctx: &Context, a: ExprId, b: ExprId) -> bool {
    let (Some(a_val), Some(b_val)) = (get_trig_arg(ctx, a), get_trig_arg(ctx, b)) else {
        return false;
    };

    if a_val != b_val && compare_expr(ctx, a_val, b_val) != Ordering::Equal {
        return false;
    }

    let is_sin_a = is_trig_pow(ctx, a, "sin", 2);
    let is_cos_b = is_trig_pow(ctx, b, "cos", 2);
    let is_cos_a = is_trig_pow(ctx, a, "cos", 2);
    let is_sin_b = is_trig_pow(ctx, b, "sin", 2);

    (is_sin_a && is_cos_b) || (is_cos_a && is_sin_b)
}

fn is_negative_term(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Neg(_) => true,
        Expr::Mul(l, _) => {
            if let Expr::Number(n) = ctx.get(*l) {
                n.is_negative()
            } else {
                false
            }
        }
        Expr::Number(n) => n.is_negative(),
        _ => false,
    }
}

fn negate_term(ctx: &mut Context, expr: ExprId) -> ExprId {
    let expr_data = ctx.get(expr).clone();
    match expr_data {
        Expr::Neg(inner) => inner,
        Expr::Mul(l, r) => {
            if let Expr::Number(n) = ctx.get(l) {
                if n.is_negative() {
                    let new_n = -n.clone();
                    if new_n == num_rational::BigRational::one() {
                        return r;
                    }
                    let num_expr = ctx.add(Expr::Number(new_n));
                    return mul2_raw(ctx, num_expr, r);
                }
            }
            ctx.add(Expr::Neg(expr))
        }
        Expr::Number(n) => ctx.add(Expr::Number(-n)),
        _ => ctx.add(Expr::Neg(expr)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    fn s(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn test_factor_poly_diff_squares() {
        let mut ctx = Context::new();
        let expr = parse("x^2 - 1", &mut ctx).unwrap();
        let res = factor(&mut ctx, expr);
        // factor_polynomial should catch this first as (x-1)(x+1)
        let str_res = s(&ctx, res);
        assert!(
            str_res.contains("x - 1") || str_res.contains("-1 + x") || str_res.contains("x + -1")
        );
        assert!(str_res.contains("x + 1") || str_res.contains("1 + x"));
    }

    #[test]
    fn test_factor_poly_perfect_square() {
        let mut ctx = Context::new();
        let expr = parse("x^2 + 2*x + 1", &mut ctx).unwrap();
        let res = factor(&mut ctx, expr);
        let str_res = s(&ctx, res);
        // (x+1)^2
        assert!(str_res.contains("1 + x") || str_res.contains("x + 1")); // Canonical: 1 before x
        assert!(str_res.contains("^ 2") || str_res.contains("^2"));
    }

    #[test]
    fn test_factor_poly_perfect_square_with_coefficients() {
        let mut ctx = Context::new();
        let expr = parse("9*x^2 - 6*x + 1", &mut ctx).unwrap();
        let res = factor(&mut ctx, expr);
        let expected = parse("(3*x - 1)^2", &mut ctx).unwrap();
        assert!(poly_eq(&ctx, res, expected));
    }

    #[test]
    fn test_factor_perfect_square_trinomial_with_coefficients() {
        let mut ctx = Context::new();
        let expr = parse("9*x^2 - 6*x + 1", &mut ctx).unwrap();
        let res = factor_perfect_square_trinomial(&mut ctx, expr).expect("factor");
        let expected = parse("(3*x - 1)^2", &mut ctx).unwrap();
        assert!(poly_eq(&ctx, res, expected));
    }

    #[test]
    fn test_factor_perfect_square_trinomial_with_fractional_coefficients() {
        let mut ctx = Context::new();
        let expr = parse("u^2 + u + 1/4", &mut ctx).unwrap();
        let res = factor_perfect_square_trinomial(&mut ctx, expr).expect("factor");
        let expected = parse("(u + 1/2)^2", &mut ctx).unwrap();
        assert!(poly_eq(&ctx, res, expected));
    }

    #[test]
    fn test_factor_perfect_square_trinomial_rejects_missing_middle_coefficient() {
        let mut ctx = Context::new();
        let expr = parse("a^2 + a*b + b^2", &mut ctx).unwrap();
        assert!(factor_perfect_square_trinomial(&mut ctx, expr).is_none());
    }

    #[test]
    fn test_factor_diff_squares_structural() {
        let mut ctx = Context::new();
        // sin(x)^2 - cos(x)^2 -> (sin(x) - cos(x))(sin(x) + cos(x))
        // This is NOT a polynomial in x, so factor_polynomial fails.
        // factor_difference_squares should pick it up.
        let expr = parse("sin(x)^2 - cos(x)^2", &mut ctx).unwrap();
        let res = factor(&mut ctx, expr);
        let str_res = s(&ctx, res);
        // Canonical ordering may reorder the terms, accept various forms
        assert!(str_res.contains("sin(x)") && str_res.contains("cos(x)"));
        assert!(str_res.matches("-").count() >= 1 && str_res.matches("+").count() >= 1);
    }

    #[test]
    fn test_factor_alternating_cubic_vandermonde() {
        let mut ctx = Context::new();
        let expr = parse("a^3*(b-c) + b^3*(c-a) + c^3*(a-b)", &mut ctx).unwrap();
        let res = factor(&mut ctx, expr);
        let expected = parse("(a-b)*(a-c)*(b-c)*(a+b+c)", &mut ctx).unwrap();
        assert!(poly_eq(&ctx, res, expected));
    }

    #[test]
    fn test_factor_sophie_germain_symbolic() {
        let mut ctx = Context::new();
        let expr = parse("x^4 + 4*y^4", &mut ctx).unwrap();
        let res = factor(&mut ctx, expr);
        let expected = parse("(x^2 - 2*x*y + 2*y^2)*(x^2 + 2*x*y + 2*y^2)", &mut ctx).unwrap();
        assert!(poly_eq(&ctx, res, expected));
    }

    #[test]
    fn test_factor_sophie_germain_numeric_still_simplifies_coefficients() {
        let mut ctx = Context::new();
        let expr = parse("x^4 + 64", &mut ctx).unwrap();
        let res = factor(&mut ctx, expr);
        let expected = parse("(x^2 - 4*x + 8)*(x^2 + 4*x + 8)", &mut ctx).unwrap();
        assert!(poly_eq(&ctx, res, expected));
    }
}
