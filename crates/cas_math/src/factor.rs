use crate::build::mul2_raw;
use crate::multipoly::{multipoly_from_expr, multipoly_to_expr, MultiPoly, PolyBudget};
use crate::poly_compare::poly_eq;
use crate::polynomial::Polynomial;
use crate::trig_roots_flatten::{get_square_root, get_trig_arg, is_trig_pow};
use cas_ast::ordering::compare_expr;
use cas_ast::{Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{One, Signed, ToPrimitive};
use std::cmp::Ordering;

/// Factors an expression.
/// This is the main entry point for factorization.
pub fn factor(ctx: &mut Context, expr: ExprId) -> ExprId {
    // 1. Try polynomial factorization
    if let Some(res) = factor_polynomial(ctx, expr) {
        return res;
    }

    // 2. Extract common multivariate content/monomial, then reuse existing
    // factorization on the residual.
    if let Some(res) = factor_common_content_then_residual(ctx, expr) {
        return res;
    }

    // 3. Try the alternating cubic Vandermonde identity:
    // a^3(b-c) + b^3(c-a) + c^3(a-b) = (a-b)(a-c)(b-c)(a+b+c)
    if let Some(res) = factor_alternating_cubic_vandermonde(ctx, expr) {
        return res;
    }

    // 4. Try difference of squares
    if let Some(res) = factor_difference_squares(ctx, expr) {
        return res;
    }

    // 4b. Try sum/difference of cubes: a³ ± b³ = (a ± b)(a² ∓ ab + b²)
    if let Some(res) = factor_sum_difference_of_cubes(ctx, expr) {
        return res;
    }

    // 5. Try Sophie Germain identity: a^4 + 4b^4 = (a² + 2ab + 2b²)(a² - 2ab + 2b²)
    if let Some(res) = factor_sophie_germain(ctx, expr) {
        return res;
    }

    // 6. Try perfect square trinomial: a² ± 2ab + b² = (a ± b)²
    if let Some(res) = factor_perfect_square_trinomial(ctx, expr) {
        return res;
    }

    // 7. Try binomial cube identity: a^3 ± 3a^2b + 3ab^2 ± b^3 = (a ± b)^3
    if let Some(res) = factor_binomial_cube_identity(ctx, expr) {
        return res;
    }

    // Recursive factorization?
    // For now, just return original if no top-level factorization applies.
    // Ideally we should factor sub-expressions too.
    // But `factor` usually means "factor this polynomial".
    // Let's stick to top-level for now, or maybe recurse if it's a product/sum?

    expr
}

fn factor_common_content_then_residual(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    if !matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _)) {
        return None;
    }

    let budget = PolyBudget {
        max_terms: 24,
        max_total_degree: 16,
        max_pow_exp: 4,
    };
    let poly = multipoly_from_expr(ctx, expr, &budget).ok()?;
    if poly.vars.len() < 2 || poly.num_terms() < 2 {
        return None;
    }

    let content = poly.content();
    let monomial_gcd = poly.monomial_gcd();
    let has_content = content > BigRational::one();
    let has_monomial = monomial_gcd.iter().any(|&exp| exp > 0);
    if !has_content && !has_monomial {
        return None;
    }

    let content_reduced = if has_content {
        poly.div_scalar_exact(&content)?
    } else {
        poly.clone()
    };
    let residual_poly = if has_monomial {
        content_reduced.div_monomial_exact(&monomial_gcd)?
    } else {
        content_reduced
    };
    if residual_poly.is_zero() {
        return None;
    }

    let common_poly = MultiPoly {
        vars: poly.vars.clone(),
        terms: vec![(content, monomial_gcd)],
    };
    let common_expr = multipoly_to_expr(&common_poly, ctx);
    let residual_expr = multipoly_to_expr(&residual_poly, ctx);
    let residual_factored = factor(ctx, residual_expr);
    let factored = mul2_raw(ctx, common_expr, residual_factored);

    if compare_expr(ctx, factored, expr) == Ordering::Equal {
        return None;
    }

    let old_nodes = cas_ast::count_nodes(ctx, expr);
    let new_nodes = cas_ast::count_nodes(ctx, factored);
    if new_nodes > old_nodes + 2 {
        return None;
    }

    poly_eq(ctx, expr, factored).then_some(factored)
}

pub fn factor_binomial_cube_identity(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    use crate::expr_nary::{AddView, Sign};

    let add_view = AddView::from_expr(ctx, expr);
    if add_view.terms.len() != 4 {
        return None;
    }

    let cube_terms: Vec<_> = add_view
        .terms
        .iter()
        .enumerate()
        .filter_map(|(index, (term, sign))| {
            get_cube_root_base(ctx, *term).map(|base| (index, base, *sign))
        })
        .collect();
    if cube_terms.len() != 2 {
        return None;
    }

    let ((first_index, first_base, first_sign), (second_index, second_base, second_sign)) =
        (cube_terms[0], cube_terms[1]);
    if compare_expr(ctx, first_base, second_base) == Ordering::Equal {
        return None;
    }

    let remaining_terms: Vec<_> = add_view
        .terms
        .iter()
        .enumerate()
        .filter_map(|(index, (term, sign))| {
            (![first_index, second_index].contains(&index)).then_some((*term, *sign))
        })
        .collect();
    if remaining_terms.len() != 2 {
        return None;
    }

    if first_sign == Sign::Pos && second_sign == Sign::Pos {
        for (left_base, right_base) in [(first_base, second_base), (second_base, first_base)] {
            if remaining_terms_match_binomial_cube(
                ctx,
                &remaining_terms,
                left_base,
                right_base,
                false,
            ) {
                return Some(build_binomial_cube(ctx, left_base, right_base, false));
            }
        }
    }

    let (positive_base, negative_base) = match (first_sign, second_sign) {
        (Sign::Pos, Sign::Neg) => (first_base, second_base),
        (Sign::Neg, Sign::Pos) => (second_base, first_base),
        _ => return None,
    };
    remaining_terms_match_binomial_cube(ctx, &remaining_terms, positive_base, negative_base, true)
        .then(|| build_binomial_cube(ctx, positive_base, negative_base, true))
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

        // 1. Factor over ℚ: rational (linear) roots plus any reducible even quartic.
        let factors = factor_over_rationals(&poly);

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

/// Factor a univariate polynomial over ℚ as far as this module supports: peel the
/// rational (linear) roots via [`Polynomial::factor_rational_roots`], then split each
/// residual reducible even quartic `a·x⁴+b·x²+c` into two quadratics, and any reducible
/// even polynomial of degree ≥ 6 (`x⁶+1`, `x⁶+x⁴+x²+1`) via the `t = x²` substitution.
/// Used by both the `factor` command and the rational-integration partial-fraction path
/// so a degree-6+ denominator like `x⁶-1` decomposes fully into linear × quadratic factors.
pub(crate) fn factor_over_rationals(poly: &Polynomial) -> Vec<Polynomial> {
    poly.factor_rational_roots()
        .into_iter()
        .flat_map(|f| {
            if let Some((g, h)) = split_reducible_even_quartic(&f) {
                return vec![g, h];
            }
            if let Some(factors) = split_reducible_even_poly(&f) {
                return factors;
            }
            vec![f]
        })
        .collect()
}

/// Factor a reducible **even** polynomial `f(x) = g(x²)` of degree ≥ 6 (only even-power
/// terms) by factoring `g(t)` over ℚ (`t = x²`) and back-substituting each factor, then
/// re-factoring it (so a back-substituted even quartic gets the Sophie-Germain split).
/// Returns `None` for a non-even polynomial, degree < 6, or an `g(t)` irreducible over ℚ
/// (no progress) — e.g. `x⁶+1 → (x²+1)(x⁴-x²+1)`, `x⁶+x⁴+x²+1 → (x²+1)(x⁴+1)`.
///
/// Even quartics are handled directly by [`split_reducible_even_quartic`] (which also
/// covers the Sophie-Germain case `t²+t+1` irreducible), so this only fires for degree ≥ 6.
fn split_reducible_even_poly(poly: &Polynomial) -> Option<Vec<Polynomial>> {
    use num_traits::Zero;
    let d = poly.degree();
    if d < 6 || !d.is_multiple_of(2) {
        return None;
    }
    // Every odd-power coefficient must be zero (a pure even polynomial).
    for i in (1..=d).step_by(2) {
        if !poly.coeffs.get(i).is_none_or(|c| c.is_zero()) {
            return None;
        }
    }
    // g(t) with g.coeffs[k] = f.coeffs[2k].
    let g_coeffs: Vec<BigRational> = (0..=d / 2)
        .map(|k| {
            poly.coeffs
                .get(2 * k)
                .cloned()
                .unwrap_or_else(BigRational::zero)
        })
        .collect();
    let g = Polynomial::new(g_coeffs, poly.var.clone());
    let g_factors = factor_over_rationals(&g);
    if g_factors.len() <= 1 {
        return None; // g irreducible over ℚ ⇒ f = g(x²) does not factor this way
    }
    // Back-substitute each gᵢ(t) → gᵢ(x²) and re-factor (handles `x²-r` linears and even
    // quartics from a quadratic-in-t factor).
    let mut result = Vec::new();
    for gi in g_factors {
        let mut bs_coeffs = vec![BigRational::zero(); 2 * gi.degree() + 1];
        for (k, c) in gi.coeffs.iter().enumerate() {
            bs_coeffs[2 * k] = c.clone();
        }
        let bs = Polynomial::new(bs_coeffs, poly.var.clone());
        result.extend(factor_over_rationals(&bs));
    }
    Some(result)
}

/// Exact rational cube root: `∛(p/q)` is rational iff `p` and `q` are both perfect cubes
/// (cube root preserves sign, so negatives are fine). Returns `None` otherwise.
fn rational_cbrt(r: &BigRational) -> Option<BigRational> {
    use num_integer::Roots;
    let (n, d) = (r.numer(), r.denom());
    let (cn, cd) = (Roots::cbrt(n), Roots::cbrt(d));
    if &(&cn * &cn * &cn) == n && &(&cd * &cd * &cd) == d {
        Some(BigRational::new(cn, cd))
    } else {
        None
    }
}

/// The exact cube root of a perfect-cube expression, or `None`. Handles a perfect-cube
/// rational (`8 → 2`, `-27 → -3`), the `Pow(b, 3k)` form (`x³ → x`, `x⁶ → x²`), a product
/// of perfect cubes (`8·x³ → 2·x`), and a negated cube. Used (under a `poly_eq` guard) by
/// the sum/difference-of-cubes matcher so `8x³+27y³` factors too.
fn perfect_cube_root(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr).clone() {
        Expr::Number(n) => rational_cbrt(&n).map(|c| ctx.add(Expr::Number(c))),
        Expr::Pow(base, exp) => {
            let e = match ctx.get(exp) {
                Expr::Number(n) if n.is_integer() => n.to_integer().to_i64()?,
                _ => return None,
            };
            if e <= 0 || e % 3 != 0 {
                return None;
            }
            if e == 3 {
                Some(base)
            } else {
                let k = ctx.num(e / 3);
                Some(ctx.add(Expr::Pow(base, k)))
            }
        }
        Expr::Mul(a, b) => {
            let ra = perfect_cube_root(ctx, a)?;
            let rb = perfect_cube_root(ctx, b)?;
            Some(mul2_raw(ctx, ra, rb))
        }
        Expr::Neg(inner) => perfect_cube_root(ctx, inner).map(|r| ctx.add(Expr::Neg(r))),
        _ => None,
    }
}

/// Exact rational square root: `√(p/q)` for a non-negative reduced rational is rational
/// iff both `p` and `q` are perfect squares. Returns `None` for negatives or non-squares.
fn rational_sqrt(r: &BigRational) -> Option<BigRational> {
    use num_integer::Roots;
    if r.is_negative() {
        return None;
    }
    let (n, d) = (r.numer(), r.denom());
    let (sn, sd) = (Roots::sqrt(n), Roots::sqrt(d));
    if &(&sn * &sn) == n && &(&sd * &sd) == d {
        Some(BigRational::new(sn, sd))
    } else {
        None
    }
}

/// Split a reducible **even** quartic `a·x⁴ + b·x² + c` (no odd-power terms) into two
/// quadratic factors over ℚ, or `None` if it is irreducible (e.g. `x⁴+1`, `x⁴-x²+1`).
///
/// Two reducible forms are covered:
/// - **Biquadratic** `a(x²-r)(x²-s)`, when `a·t²+b·t+c` (t = x²) has rational roots r, s
///   (i.e. `b²-4ac` is a rational square). Emitted as `(x²-r)·(a·x²-a·s)` so both factors
///   have rational coefficients. Example: `x⁴+3x²+2 → (x²+1)(x²+2)`.
/// - **Sophie-Germain** `(α x²+β x+γ)(α x²-β x+γ) = α²x⁴ + (2αγ-β²)x² + γ²`, when
///   `α=√a`, `γ=±√c` and `β²=2αγ-b` are all rational squares. Example:
///   `x⁴+x²+1 → (x²+x+1)(x²-x+1)`, `4x⁴+1 → (2x²+2x+1)(2x²-2x+1)`.
///
/// Only applied to factors that survived [`Polynomial::factor_rational_roots`], so the
/// quartic has no rational root and the resulting quadratics are irreducible over ℚ.
fn split_reducible_even_quartic(poly: &Polynomial) -> Option<(Polynomial, Polynomial)> {
    use num_traits::Zero;
    if poly.degree() != 4 {
        return None;
    }
    let coeff = |i: usize| {
        poly.coeffs
            .get(i)
            .cloned()
            .unwrap_or_else(BigRational::zero)
    };
    let (c, c1, b, c3, a) = (coeff(0), coeff(1), coeff(2), coeff(3), coeff(4));
    // Must be a pure even quartic with non-zero leading coefficient.
    if !c1.is_zero() || !c3.is_zero() || a.is_zero() {
        return None;
    }
    let var = poly.var.clone();
    let quad = |c0: BigRational, c1: BigRational, c2: BigRational| {
        Polynomial::new(vec![c0, c1, c2], var.clone())
    };
    let two = BigRational::from_integer(2.into());
    let four = BigRational::from_integer(4.into());

    // Biquadratic: rational roots of a·t²+b·t+c (t = x²).
    let disc = &b * &b - &four * &a * &c;
    if let Some(sq) = rational_sqrt(&disc) {
        let two_a = &two * &a;
        let r = (-&b + &sq) / &two_a;
        let s = (-&b - &sq) / &two_a;
        // a·x⁴+b·x²+c = (x²-r)·(a·x²-a·s).
        let f1 = quad(-r, BigRational::zero(), BigRational::one());
        let f2 = quad(-(&a * &s), BigRational::zero(), a.clone());
        return Some((f1, f2));
    }

    // Sophie-Germain symmetric split: α=√a, γ=±√c, β=√(2αγ-b).
    let alpha = rational_sqrt(&a)?;
    let root_c = rational_sqrt(&c)?;
    for gamma in [root_c.clone(), -root_c] {
        let beta_sq = &two * &alpha * &gamma - &b;
        if let Some(beta) = rational_sqrt(&beta_sq) {
            if beta.is_zero() {
                continue; // β=0 is the biquadratic case, already handled above
            }
            let f1 = quad(gamma.clone(), beta.clone(), alpha.clone());
            let f2 = quad(gamma.clone(), -beta, alpha.clone());
            return Some((f1, f2));
        }
    }
    None
}

/// Factors a sum or difference of cubes: `a³ + b³ = (a+b)(a²-ab+b²)`,
/// `a³ - b³ = (a-b)(a²+ab+b²)`. Recognises the `Pow(_, 3)` cube form (`x³±y³`,
/// `a³-b³`); the candidate factorization is checked exact via `poly_eq`, so only a
/// genuine identity is returned. (Univariate numeric cubes like `x³-8` are already
/// factored by `factor_polynomial` via rational roots; coefficient cubes such as
/// `8x³+27y³` need a richer cube-root extractor and stay whole here.)
pub(crate) fn factor_sum_difference_of_cubes(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let (l, r, is_difference) = match ctx.get(expr).clone() {
        Expr::Sub(l, r) => (l, r, true),
        Expr::Add(a, b) => {
            if is_negative_term(ctx, b) {
                (a, negate_term(ctx, b), true)
            } else if is_negative_term(ctx, a) {
                (b, negate_term(ctx, a), true)
            } else {
                (a, b, false)
            }
        }
        _ => return None,
    };
    let a = perfect_cube_root(ctx, l)?;
    let b = perfect_cube_root(ctx, r)?;

    let two = ctx.num(2);
    let a2 = ctx.add(Expr::Pow(a, two));
    let b2 = ctx.add(Expr::Pow(b, two));
    let ab = mul2_raw(ctx, a, b);

    let (linear, quadratic) = if is_difference {
        // (a - b)(a² + ab + b²)
        let lin = ctx.add(Expr::Sub(a, b));
        let inner = ctx.add(Expr::Add(a2, ab));
        (lin, ctx.add(Expr::Add(inner, b2)))
    } else {
        // (a + b)(a² - ab + b²)
        let lin = ctx.add(Expr::Add(a, b));
        let inner = ctx.add(Expr::Sub(a2, ab));
        (lin, ctx.add(Expr::Add(inner, b2)))
    };
    // Expand the quadratic factor so a coefficient cube reads `4x²-6xy+9y²`, not
    // `(2x)²-(2x)(3y)+(3y)²`. Value-preserving, so the `poly_eq` guard still holds.
    let budget = PolyBudget {
        max_terms: 16,
        max_total_degree: 8,
        max_pow_exp: 4,
    };
    let quadratic = multipoly_from_expr(ctx, quadratic, &budget)
        .map(|p| multipoly_to_expr(&p, ctx))
        .unwrap_or(quadratic);
    let candidate = mul2_raw(ctx, linear, quadratic);
    poly_eq(ctx, expr, candidate).then_some(candidate)
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

fn build_binomial_cube(ctx: &mut Context, a: ExprId, b: ExprId, is_sub: bool) -> ExprId {
    let three = ctx.num(3);
    let binomial = if is_sub {
        ctx.add(Expr::Sub(a, b))
    } else {
        ctx.add(Expr::Add(a, b))
    };
    ctx.add(Expr::Pow(binomial, three))
}

fn remaining_terms_match_binomial_cube(
    ctx: &Context,
    remaining_terms: &[(ExprId, crate::expr_nary::Sign)],
    a: ExprId,
    b: ExprId,
    is_sub: bool,
) -> bool {
    let mut has_a_sq_b = false;
    let mut has_a_b_sq = false;

    for (term, outer_sign) in remaining_terms {
        if let Some(embedded_positive) = is_3a2b_term(ctx, *term, a, b) {
            let overall_positive = combine_add_view_sign(*outer_sign, embedded_positive);
            if overall_positive != is_sub {
                has_a_sq_b = true;
                continue;
            }
        }
        if let Some(embedded_positive) = is_3ab2_term(ctx, *term, a, b) {
            let overall_positive = combine_add_view_sign(*outer_sign, embedded_positive);
            if overall_positive {
                has_a_b_sq = true;
                continue;
            }
        }
        return false;
    }

    has_a_sq_b && has_a_b_sq
}

fn combine_add_view_sign(outer_sign: crate::expr_nary::Sign, embedded_positive: bool) -> bool {
    use crate::expr_nary::Sign;

    match (outer_sign, embedded_positive) {
        (Sign::Pos, true) => true,
        (Sign::Neg, true) => false,
        (Sign::Pos, false) => false,
        (Sign::Neg, false) => true,
    }
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

fn get_cube_root_base(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Pow(base, exp) => {
            if let Expr::Number(n) = ctx.get(*exp) {
                if n.is_integer() && *n.numer() == 3.into() {
                    return Some(*base);
                }
            }
            None
        }
        _ => None,
    }
}

fn is_3a2b_term(ctx: &Context, expr: ExprId, a: ExprId, b: ExprId) -> Option<bool> {
    is_3_mixed_term(ctx, expr, a, b, true)
}

fn is_3ab2_term(ctx: &Context, expr: ExprId, a: ExprId, b: ExprId) -> Option<bool> {
    is_3_mixed_term(ctx, expr, a, b, false)
}

fn is_3_mixed_term(
    ctx: &Context,
    expr: ExprId,
    a: ExprId,
    b: ExprId,
    a_is_squared: bool,
) -> Option<bool> {
    use crate::expr_nary::MulView;

    let mul_view = MulView::from_expr(ctx, expr);
    if mul_view.factors.len() != 3 {
        return None;
    }

    let mut coef_sign: Option<bool> = None;
    let mut has_first = false;
    let mut has_second = false;

    for &factor in mul_view.factors.iter() {
        if let Expr::Number(n) = ctx.get(factor) {
            if n.is_integer() {
                if *n.numer() == 3.into() {
                    coef_sign = Some(true);
                    continue;
                }
                if *n.numer() == (-3).into() {
                    coef_sign = Some(false);
                    continue;
                }
            }
        }
        if factor_matches_squared_base(ctx, factor, if a_is_squared { a } else { b }) {
            has_first = true;
        } else if compare_expr(ctx, factor, if a_is_squared { b } else { a }) == Ordering::Equal {
            has_second = true;
        }
    }

    (coef_sign.is_some() && has_first && has_second).then_some(coef_sign?)
}

fn factor_matches_squared_base(ctx: &Context, factor: ExprId, base: ExprId) -> bool {
    match ctx.get(factor) {
        Expr::Pow(inner_base, exp) => {
            matches!(ctx.get(*exp), Expr::Number(n) if n.is_integer() && *n.numer() == 2.into())
                && compare_expr(ctx, *inner_base, base) == Ordering::Equal
        }
        _ => false,
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
    fn test_factor_reducible_even_quartics() {
        // Reducible even quartics `a·x⁴+b·x²+c` split into two quadratics over ℚ.
        // (source, expected factorization). Display orders a quadratic's constant
        // before the `-x` term, e.g. `x^2 + 1 - x`.
        for (src, expect) in [
            ("x^4 + x^2 + 1", "(x^2 + x + 1) * (x^2 + 1 - x)"), // Sophie-Germain
            ("x^4 + 3*x^2 + 2", "(x^2 + 1) * (x^2 + 2)"),       // biquadratic
            ("x^4 - 3*x^2 + 1", "(x^2 + x - 1) * (x^2 - x - 1)"),
            ("4*x^4 + 1", "(2 * x^2 + 2 * x + 1) * (2 * x^2 + 1 - 2 * x)"), // non-monic Sophie-Germain
            (
                "x^6 - 1",
                "(x - 1) * (x + 1) * (x^2 + x + 1) * (x^2 + 1 - x)",
            ), // transitive
        ] {
            let mut ctx = Context::new();
            let expr = parse(src, &mut ctx).unwrap();
            let factored = factor(&mut ctx, expr);
            assert_eq!(s(&ctx, factored), expect, "factor({src})");
        }
    }

    #[test]
    fn test_factor_sum_difference_of_cubes() {
        for (src, expect) in [
            ("x^3 + y^3", "(x + y) * (x^2 + y^2 - x * y)"),
            ("x^3 - y^3", "(x - y) * (x^2 + y^2 + x * y)"),
            ("a^3 - b^3", "(a - b) * (a^2 + b^2 + a * b)"),
            ("a^3 + b^3", "(a + b) * (a^2 + b^2 - a * b)"),
            // Coefficient cubes (perfect-cube coefficients).
            (
                "8*x^3 + 27*y^3",
                "(2 * x + 3 * y) * (4 * x^2 + 9 * y^2 - 6 * x * y)",
            ),
            ("x^3 - 8*y^3", "(x - 2 * y) * (x^2 + 2 * x * y + 4 * y^2)"),
        ] {
            let mut ctx = Context::new();
            let expr = parse(src, &mut ctx).unwrap();
            let factored = factor(&mut ctx, expr);
            assert_eq!(s(&ctx, factored), expect, "factor({src})");
            // Exact: the factorization equals the input as a polynomial.
            let reparsed = parse(&s(&ctx, factored), &mut ctx).unwrap();
            assert!(
                crate::poly_compare::poly_eq(&ctx, reparsed, expr),
                "factor({src}) must be exact"
            );
        }
    }

    #[test]
    fn test_cube_matcher_does_not_misfire() {
        // Non-cube binomials and squares must NOT be factored by the cube matcher.
        for src in ["x^2 + y^2", "x^3 + y^2", "x + y", "x^4 - y^4"] {
            let mut ctx = Context::new();
            let expr = parse(src, &mut ctx).unwrap();
            // The cube matcher specifically declines (returns None) for these.
            assert!(
                super::factor_sum_difference_of_cubes(&mut ctx, expr).is_none(),
                "cube matcher must decline {src}"
            );
        }
    }

    #[test]
    fn test_factor_reducible_even_polynomials() {
        // Reducible even polynomials of degree ≥ 6 factor via t = x² substitution.
        for (src, expect) in [
            ("x^6 + 1", "(x^2 + 1) * (x^4 + 1 - x^2)"),
            ("x^6 + x^4 + x^2 + 1", "(x^2 + 1) * (x^4 + 1)"),
            ("x^8 - 1", "(x - 1) * (x + 1) * (x^2 + 1) * (x^4 + 1)"),
            // Degree 8 with Sophie-Germain on the back-substituted quartic-in-t.
            (
                "x^8 + x^4 + 1",
                "(x^2 + x + 1) * (x^2 + 1 - x) * (x^4 + 1 - x^2)",
            ),
        ] {
            let mut ctx = Context::new();
            let expr = parse(src, &mut ctx).unwrap();
            let factored = factor(&mut ctx, expr);
            assert_eq!(s(&ctx, factored), expect, "factor({src})");
            // The factorization is exact: it expands back to the input.
            let reparsed = parse(&s(&ctx, factored), &mut ctx).unwrap();
            assert!(
                crate::poly_compare::poly_eq(&ctx, reparsed, expr),
                "factor({src}) must be exact"
            );
        }
    }

    #[test]
    fn test_irreducible_even_polynomials_stay_whole() {
        // Irreducible-over-ℚ even polys and polys with odd terms must NOT be split.
        for src in ["x^6 + x^3 + 1", "x^6 + x^5 + 1", "x^2 + 1"] {
            let mut ctx = Context::new();
            let expr = parse(src, &mut ctx).unwrap();
            let factored = factor(&mut ctx, expr);
            assert!(
                crate::poly_compare::poly_eq(&ctx, factored, expr),
                "factor({src}) must stay whole, got {}",
                s(&ctx, factored)
            );
        }
    }

    #[test]
    fn test_irreducible_quartics_stay_whole() {
        // Quartics irreducible over ℚ must NOT be split (soundness).
        for src in ["x^4 + 1", "x^4 - x^2 + 1", "x^4 + x^3 + x^2 + x + 1"] {
            let mut ctx = Context::new();
            let expr = parse(src, &mut ctx).unwrap();
            let factored = factor(&mut ctx, expr);
            // `factor` returns the input unchanged (no factorization found).
            assert!(
                crate::poly_compare::poly_eq(&ctx, factored, expr),
                "factor({src}) must stay whole, got {}",
                s(&ctx, factored)
            );
        }
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
    fn test_factor_symbolic_binomial_cube_sum() {
        let mut ctx = Context::new();
        let expr = parse("a^3 + 3*a^2*b + 3*a*b^2 + b^3", &mut ctx).unwrap();
        let res = factor(&mut ctx, expr);
        let expected = parse("(a + b)^3", &mut ctx).unwrap();
        assert!(poly_eq(&ctx, res, expected));
    }

    #[test]
    fn test_factor_symbolic_binomial_cube_difference() {
        let mut ctx = Context::new();
        let expr = parse("a^3 - 3*a^2*b + 3*a*b^2 - b^3", &mut ctx).unwrap();
        let res = factor(&mut ctx, expr);
        let expected = parse("(a - b)^3", &mut ctx).unwrap();
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

    #[test]
    fn test_factor_multivar_common_monomial_then_residual_square() {
        let mut ctx = Context::new();
        let expr = parse("y^2*z^2 + 2*y^2*z + y^2", &mut ctx).unwrap();
        let res = factor(&mut ctx, expr);
        let expected = parse("y^2*(z+1)^2", &mut ctx).unwrap();

        assert!(poly_eq(&ctx, res, expected));
        let rendered = s(&ctx, res);
        assert!(
            rendered.contains("y^2") && rendered.contains("(z + 1)^2"),
            "unexpected factor shape: {rendered}"
        );
    }

    #[test]
    fn test_factor_multivar_common_numeric_content() {
        let mut ctx = Context::new();
        let expr = parse("2*x + 4*y", &mut ctx).unwrap();
        let res = factor(&mut ctx, expr);
        let expected = parse("2*(x + 2*y)", &mut ctx).unwrap();

        assert!(poly_eq(&ctx, res, expected));
        let rendered = s(&ctx, res);
        assert!(
            rendered.contains("2 * (x + 2 * y)"),
            "unexpected factor shape: {rendered}"
        );
    }

    #[test]
    fn test_factor_multivar_common_content_and_monomial() {
        let mut ctx = Context::new();
        let expr = parse("2*x*y + 4*x*z", &mut ctx).unwrap();
        let res = factor(&mut ctx, expr);
        let expected = parse("2*x*(y + 2*z)", &mut ctx).unwrap();

        assert!(poly_eq(&ctx, res, expected));
        let rendered = s(&ctx, res);
        assert!(
            rendered.contains("2 * x * (y + 2 * z)"),
            "unexpected factor shape: {rendered}"
        );
    }
}
