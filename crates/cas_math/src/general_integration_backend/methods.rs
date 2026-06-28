//! Method probes (rational, Hermite, heurisch) and the public backend entry point.

use super::verification_normalization::*;
use super::*;

use crate::expr_nary::{add_terms_signed, Sign};
use crate::expr_predicates::contains_named_var;
use crate::semantic_equality::SemanticEqualityChecker;
use cas_ast::{BuiltinFn, ConditionPredicate, Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

fn try_rational_reciprocal_affine_probe(
    ctx: &mut Context,
    integrand: ExprId,
    variable: &str,
    probe_runner: &mut AlgorithmicIntegrationProbeRunner,
) -> AlgorithmicIntegrationProbeResult {
    let parts = match affine_denominator_linear_numerator_parts(ctx, integrand, variable) {
        Ok(parts) => parts,
        Err(reason) => {
            if let Some(antiderivative) =
                multi_quadratic_partial_fraction_antiderivative(ctx, integrand, variable)
            {
                // Distinct irreducible numeric quadratics are strictly
                // positive, so the antiderivative is unconditional.
                let mut candidate = AlgorithmicIntegrationCandidate::unverified(
                    integrand,
                    variable,
                    antiderivative,
                    AlgorithmicIntegrationMethod::Rational,
                );
                if !probe_runner.try_verification_check() {
                    candidate.mark_budget_exceeded();
                    return AlgorithmicIntegrationProbeResult::Candidate(candidate);
                }
                verify_antiderivative_by_differentiation(ctx, &mut candidate);
                return AlgorithmicIntegrationProbeResult::Candidate(candidate);
            }
            if let Some(parts) =
                general_rational_partial_fraction_antiderivative(ctx, integrand, variable)
            {
                let mut candidate = AlgorithmicIntegrationCandidate::unverified(
                    integrand,
                    variable,
                    parts.antiderivative,
                    AlgorithmicIntegrationMethod::Rational,
                );
                candidate.required_conditions = parts.pole_conditions;
                if !probe_runner.try_verification_check() {
                    candidate.mark_budget_exceeded();
                    return AlgorithmicIntegrationProbeResult::Candidate(candidate);
                }
                verify_antiderivative_by_differentiation(ctx, &mut candidate);
                return AlgorithmicIntegrationProbeResult::Candidate(candidate);
            }
            if let Some(antiderivative) =
                symmetric_surd_even_quartic_antiderivative(ctx, integrand, variable)
            {
                // The two emitted quadratics are strictly positive (irreducible),
                // so the antiderivative is unconditional.
                let mut candidate = AlgorithmicIntegrationCandidate::unverified(
                    integrand,
                    variable,
                    antiderivative,
                    AlgorithmicIntegrationMethod::Rational,
                );
                if !probe_runner.try_verification_check() {
                    candidate.mark_budget_exceeded();
                    return AlgorithmicIntegrationProbeResult::Candidate(candidate);
                }
                verify_antiderivative_by_differentiation(ctx, &mut candidate);
                return AlgorithmicIntegrationProbeResult::Candidate(candidate);
            }
            return AlgorithmicIntegrationProbeResult::NoMatch(reason);
        }
    };

    let variable_expr = ctx.var(variable);
    let quotient_antiderivative =
        build_backend_product(ctx, parts.quotient_coefficient, variable_expr);
    let log_antiderivative = build_affine_denominator_remainder_antiderivative(
        ctx,
        parts.remainder,
        parts.denominator,
        &parts.denominator_slope,
    );
    let antiderivative = build_backend_sum(ctx, quotient_antiderivative, log_antiderivative);
    let mut candidate = AlgorithmicIntegrationCandidate::unverified(
        integrand,
        variable,
        antiderivative,
        AlgorithmicIntegrationMethod::Rational,
    );
    candidate
        .required_conditions
        .push(ConditionPredicate::NonZero(parts.denominator));
    if let Some(condition) = parts.denominator_slope.required_condition() {
        candidate.required_conditions.push(condition);
    }
    if !probe_runner.try_verification_check() {
        candidate.mark_budget_exceeded();
        return AlgorithmicIntegrationProbeResult::Candidate(candidate);
    }
    verify_antiderivative_by_differentiation(ctx, &mut candidate);

    AlgorithmicIntegrationProbeResult::Candidate(candidate)
}

pub(super) struct AffineDenominatorLinearNumeratorParts {
    pub(super) quotient_coefficient: ExprId,
    pub(super) remainder: ExprId,
    pub(super) denominator: ExprId,
    pub(super) denominator_slope: BackendAffineSlope,
}

const MULTI_QUADRATIC_MAX_FACTORS: usize = 3;

pub(super) struct MultiQuadraticFactorTerm {
    factor_expr: ExprId,
    pub(super) linear_b: BigRational,
    pub(super) constant_c: BigRational,
    pub(super) alpha: BigRational,
    pub(super) beta: BigRational,
}

/// Partial fractions over a product of 2..=3 DISTINCT monic irreducible
/// quadratics with numeric coefficients: N(x)/prod(x^2+b_i*x+c_i) with
/// deg(N) < 2k decomposes as sum (alpha_i*x+beta_i)/q_i via a 2k x 2k
/// rational linear system (the same shared solver the educational
/// partial-fraction families use). Returns the assembled antiderivative
/// sum of (alpha/2)*ln(q_i) + gamma_i-scaled arctan terms; the quadratics
/// are strictly positive so no conditions are required.
pub(super) fn multi_quadratic_partial_fraction_antiderivative(
    ctx: &mut Context,
    integrand: ExprId,
    variable: &str,
) -> Option<ExprId> {
    let terms = multi_quadratic_partial_fraction_terms(ctx, integrand, variable)?;
    let mut antiderivative = ctx.num(0);
    for term in terms {
        let piece = build_multi_quadratic_term_antiderivative(ctx, &term, variable);
        antiderivative = build_backend_sum(ctx, antiderivative, piece);
    }
    Some(antiderivative)
}

pub(super) fn multi_quadratic_partial_fraction_terms(
    ctx: &mut Context,
    integrand: ExprId,
    variable: &str,
) -> Option<Vec<MultiQuadraticFactorTerm>> {
    let (numerator_expr, denominator_expr) = match ctx.get(integrand) {
        Expr::Div(numerator, denominator) => (*numerator, *denominator),
        _ => return None,
    };
    let factor_exprs = backend_mul_factors(ctx, denominator_expr);
    if factor_exprs.len() < 2 || factor_exprs.len() > MULTI_QUADRATIC_MAX_FACTORS {
        return None;
    }

    let mut factors: Vec<(ExprId, crate::polynomial::Polynomial)> = Vec::new();
    for factor_expr in factor_exprs {
        let poly = crate::polynomial::Polynomial::from_expr(ctx, factor_expr, variable).ok()?;
        if poly.degree() != 2 || !poly.leading_coeff().is_one() {
            return None;
        }
        let b = poly.coeffs[1].clone();
        let c = poly.coeffs[0].clone();
        // Irreducible over Q (and hence strictly positive): b^2 - 4c < 0.
        let four = BigRational::from_integer(4.into());
        if &b * &b - four * &c >= BigRational::zero() {
            return None;
        }
        if factors.iter().any(|(_, known)| known == &poly) {
            return None;
        }
        factors.push((factor_expr, poly));
    }

    let numerator = crate::polynomial::Polynomial::from_expr(ctx, numerator_expr, variable).ok()?;
    let unknowns = 2 * factors.len();
    if numerator.is_zero() || numerator.degree() >= unknowns {
        return None;
    }

    // Denominator product and per-factor cofactors prod_{j != i} q_j.
    let mut denominator_poly = crate::polynomial::Polynomial::one(variable.to_string());
    for (_, poly) in &factors {
        denominator_poly = denominator_poly.mul(poly);
    }
    let mut basis: Vec<crate::polynomial::Polynomial> = Vec::new();
    let x_poly = crate::polynomial::Polynomial::new(
        vec![BigRational::zero(), BigRational::one()],
        variable.to_string(),
    );
    for (_, poly) in &factors {
        let (cofactor, remainder) = denominator_poly.div_rem(poly).ok()?;
        if !remainder.is_zero() {
            return None;
        }
        basis.push(cofactor.mul(&x_poly));
        basis.push(cofactor);
    }

    let mut matrix = vec![vec![BigRational::zero(); unknowns]; unknowns];
    let mut rhs = vec![BigRational::zero(); unknowns];
    for row in 0..unknowns {
        for (column, column_poly) in basis.iter().enumerate() {
            matrix[row][column] = column_poly
                .coeffs
                .get(row)
                .cloned()
                .unwrap_or_else(BigRational::zero);
        }
        rhs[row] = numerator
            .coeffs
            .get(row)
            .cloned()
            .unwrap_or_else(BigRational::zero);
    }
    let solution = crate::symbolic_integration_support::solve_rational_linear_system(matrix, rhs)?;

    let mut terms = Vec::with_capacity(factors.len());
    for (index, (factor_expr, poly)) in factors.into_iter().enumerate() {
        terms.push(MultiQuadraticFactorTerm {
            factor_expr,
            linear_b: poly.coeffs[1].clone(),
            constant_c: poly.coeffs[0].clone(),
            alpha: solution[2 * index].clone(),
            beta: solution[2 * index + 1].clone(),
        });
    }
    Some(terms)
}

/// Build the partial-fraction decomposition expression for the
/// multi-quadratic family - the sum of (alpha_i*x + beta_i)/q_i with the
/// user's syntactic factors as denominators. Didactic narration support:
/// returns Some exactly when the multi-quadratic probe would match.
pub fn multi_quadratic_partial_fraction_decomposition_expr(
    ctx: &mut Context,
    integrand: ExprId,
    variable: &str,
) -> Option<ExprId> {
    let terms = multi_quadratic_partial_fraction_terms(ctx, integrand, variable)?;
    let variable_expr = ctx.var(variable);
    let mut sum = ctx.num(0);
    for term in &terms {
        let alpha_expr = ctx.add(Expr::Number(term.alpha.clone()));
        let linear = build_backend_product(ctx, alpha_expr, variable_expr);
        let beta_expr = ctx.add(Expr::Number(term.beta.clone()));
        let numerator = build_backend_sum(ctx, linear, beta_expr);
        let piece = ctx.add(Expr::Div(numerator, term.factor_expr));
        sum = build_backend_sum(ctx, sum, piece);
    }
    Some(sum)
}

/// One partial-fraction piece (alpha*x + beta)/(x^2 + b*x + c):
/// (alpha/2)*ln(q) + gamma-scaled arctan with gamma = beta - alpha*b/2.
/// Presentation picks the half-center form (x + b/2) when b is even (so
/// x^2+1 renders arctan(x), not arctan(2x/2)) and the doubled form
/// (2x + b)/sqrt(4c - b^2) otherwise, matching the engine's own style for
/// odd linear coefficients.
fn build_multi_quadratic_term_antiderivative(
    ctx: &mut Context,
    term: &MultiQuadraticFactorTerm,
    variable: &str,
) -> ExprId {
    let two = BigRational::from_integer(2.into());
    let four = BigRational::from_integer(4.into());
    let gamma = &term.beta - &term.alpha * &term.linear_b / &two;

    let log_part = if term.alpha.is_zero() {
        ctx.num(0)
    } else {
        let alpha_expr = ctx.add(Expr::Number(term.alpha.clone()));
        build_positive_quadratic_log_derivative_antiderivative(
            ctx,
            alpha_expr,
            &BackendAffineSlope::Numeric(BigRational::one()),
            term.factor_expr,
        )
    };

    let arctan_part = if gamma.is_zero() {
        ctx.num(0)
    } else {
        let half_b = &term.linear_b / &two;
        let variable_expr = ctx.var(variable);
        let (center, slope, radius_square) = if half_b.is_integer() {
            let center = build_numeric_shifted_center(ctx, variable_expr, &half_b);
            let radius_square = &term.constant_c - &half_b * &half_b;
            (center, BigRational::one(), radius_square)
        } else {
            let doubled = ctx.add(Expr::Number(two.clone()));
            let doubled_variable = build_backend_product(ctx, doubled, variable_expr);
            let center = build_numeric_shifted_center(ctx, doubled_variable, &term.linear_b);
            let radius_square = &four * &term.constant_c - &term.linear_b * &term.linear_b;
            (center, two.clone(), radius_square)
        };
        // q = ((s*x + h)^2 + radius_square)/s^2, so gamma/q contributes
        // s^2*gamma/((s*x+h)^2 + r^2); the builder then divides once by the
        // slope and once by the radius: (s^2*gamma/s/r)*arctan(center/r),
        // which is exactly the integral for both the half (s=1) and the
        // doubled (s=2) presentation forms.
        let radius = build_numeric_radius_expr(ctx, &radius_square);
        let gamma_scale = &gamma * &slope * &slope;
        let gamma_expr = ctx.add(Expr::Number(gamma_scale));
        build_positive_quadratic_constant_numerator_antiderivative(
            ctx,
            gamma_expr,
            center,
            &BackendAffineSlope::Numeric(slope),
            radius,
        )
    };

    build_backend_sum(ctx, log_part, arctan_part)
}

/// Degree window for the general rational pipeline: degree <= 2 stays
/// owned by the existing routes (affine probe, educational quadratics,
/// the observability lane fixtures), and degree > 8 exceeds the exact
/// linear-algebra budget.
const GENERAL_RATIONAL_MIN_DENOMINATOR_DEGREE: usize = 3;
const GENERAL_RATIONAL_MAX_DENOMINATOR_DEGREE: usize = 8;
/// factor_rational_roots trial-divides up to sqrt of the constant term;
/// cap it so expanded high-degree denominators cannot stall the probe.
const GENERAL_RATIONAL_MAX_ROOT_SEARCH_CONSTANT: u64 = 100_000_000;

pub(super) struct GeneralRationalParts {
    pub(super) antiderivative: ExprId,
    pub(super) pole_conditions: Vec<ConditionPredicate>,
}

enum SquarefreeFactor {
    Linear {
        root: BigRational,
    },
    Quadratic {
        linear_b: BigRational,
        constant_c: BigRational,
    },
}

/// General rational integration for numeric-coefficient integrands whose
/// denominator splitting needs only rational roots and even-substitution
/// (biquadratic) resolvents: Ostrogradsky-Horowitz reduction extracts the
/// rational part P/D1 without factoring (D1 = gcd(D, D'), one exact
/// linear system), and the remaining squarefree integral is decomposed by
/// mixed linear/quadratic partial fractions. Quartics that only factor
/// into quadratics with linear terms (x^4+4, x^4+x^2+1) stay residual.
pub(super) fn general_rational_partial_fraction_antiderivative(
    ctx: &mut Context,
    integrand: ExprId,
    variable: &str,
) -> Option<GeneralRationalParts> {
    let (numerator_expr, denominator_expr) = match ctx.get(integrand) {
        Expr::Div(numerator, denominator) => (*numerator, *denominator),
        _ => return None,
    };
    let denominator =
        crate::polynomial::Polynomial::from_expr(ctx, denominator_expr, variable).ok()?;
    let degree = denominator.degree();
    if !(GENERAL_RATIONAL_MIN_DENOMINATOR_DEGREE..=GENERAL_RATIONAL_MAX_DENOMINATOR_DEGREE)
        .contains(&degree)
    {
        return None;
    }
    let numerator = crate::polynomial::Polynomial::from_expr(ctx, numerator_expr, variable).ok()?;
    if numerator.is_zero() || numerator.degree() >= degree {
        return None;
    }

    // Normalize the denominator monic; fold its leading coefficient into
    // the numerator so every later system works over monic polynomials.
    let leading = denominator.leading_coeff();
    let denominator = denominator.div_scalar(&leading);
    let numerator = numerator.div_scalar(&leading);

    let (repeated_part, squarefree_part) = squarefree_split(&denominator)?;
    let (rational_part, squarefree_numerator) = if repeated_part.degree() == 0 {
        (None, numerator)
    } else {
        let (p, q) = ostrogradsky_reduce(&numerator, &repeated_part, &squarefree_part, variable)?;
        (Some((p, repeated_part)), q)
    };

    let factors = split_squarefree_factors(&squarefree_part)?;
    let terms = mixed_partial_fraction_terms(ctx, &squarefree_numerator, &factors, variable)?;

    let mut antiderivative = match &rational_part {
        Some((p, d1)) if !p.is_zero() => {
            let p_expr = p.to_expr(ctx);
            let d1_expr = d1.to_expr(ctx);
            ctx.add(Expr::Div(p_expr, d1_expr))
        }
        _ => ctx.num(0),
    };
    let mut pole_conditions = Vec::new();
    for (factor, term) in factors.iter().zip(terms) {
        match factor {
            SquarefreeFactor::Linear { root } => {
                let coefficient = term.alpha;
                // A zero-residue pole emits no ln term, but if the factor
                // also lives in the repeated part the pole survives inside
                // P/D1 and its condition is still required.
                let pole_in_rational_part = rational_part
                    .as_ref()
                    .is_some_and(|(_, d1)| d1.eval(root).is_zero());
                if coefficient.is_zero() && !pole_in_rational_part {
                    continue;
                }
                let variable_expr = ctx.var(variable);
                let shift = -root.clone();
                let pole = build_numeric_shifted_center(ctx, variable_expr, &shift);
                pole_conditions.push(ConditionPredicate::NonZero(pole));
                if coefficient.is_zero() {
                    continue;
                }
                let abs_pole = ctx.call_builtin(BuiltinFn::Abs, vec![pole]);
                let log_pole = ctx.call_builtin(BuiltinFn::Ln, vec![abs_pole]);
                let coefficient_expr = ctx.add(Expr::Number(coefficient));
                let piece = build_backend_product(ctx, coefficient_expr, log_pole);
                antiderivative = build_backend_sum(ctx, antiderivative, piece);
            }
            SquarefreeFactor::Quadratic { .. } => {
                let piece = build_multi_quadratic_term_antiderivative(ctx, &term, variable);
                antiderivative = build_backend_sum(ctx, antiderivative, piece);
            }
        }
    }

    Some(GeneralRationalParts {
        antiderivative,
        pole_conditions,
    })
}

/// Real intermediate forms of the general rational pipeline, for the
/// didactic narration: the Ostrogradsky split (P/D1 plus the remaining
/// squarefree integrand Q/D2), the factored squarefree denominator, and
/// the partial-fraction decomposition. Returns Some exactly when the
/// general rational probe would match.
pub struct GeneralRationalNarrationParts {
    pub rational_part: Option<(ExprId, ExprId)>,
    pub factored_denominator: ExprId,
    pub decomposition: ExprId,
}

pub fn general_rational_partial_fraction_narration_parts(
    ctx: &mut Context,
    integrand: ExprId,
    variable: &str,
) -> Option<GeneralRationalNarrationParts> {
    let (numerator_expr, denominator_expr) = match ctx.get(integrand) {
        Expr::Div(numerator, denominator) => (*numerator, *denominator),
        _ => return None,
    };
    let denominator =
        crate::polynomial::Polynomial::from_expr(ctx, denominator_expr, variable).ok()?;
    let degree = denominator.degree();
    if !(GENERAL_RATIONAL_MIN_DENOMINATOR_DEGREE..=GENERAL_RATIONAL_MAX_DENOMINATOR_DEGREE)
        .contains(&degree)
    {
        return None;
    }
    let numerator = crate::polynomial::Polynomial::from_expr(ctx, numerator_expr, variable).ok()?;
    if numerator.is_zero() || numerator.degree() >= degree {
        return None;
    }
    let leading = denominator.leading_coeff();
    let denominator = denominator.div_scalar(&leading);
    let numerator = numerator.div_scalar(&leading);

    let (repeated_part, squarefree_part) = squarefree_split(&denominator)?;
    let (rational_part, squarefree_numerator) = if repeated_part.degree() == 0 {
        (None, numerator)
    } else {
        let (p, q) = ostrogradsky_reduce(&numerator, &repeated_part, &squarefree_part, variable)?;
        let p_expr = p.to_expr(ctx);
        let d1_expr = repeated_part.to_expr(ctx);
        let rational = ctx.add(Expr::Div(p_expr, d1_expr));
        let q_expr = q.to_expr(ctx);
        let d2_expr = squarefree_part.to_expr(ctx);
        let remaining = ctx.add(Expr::Div(q_expr, d2_expr));
        (Some((rational, remaining)), q)
    };

    let factors = split_squarefree_factors(&squarefree_part)?;
    let terms = mixed_partial_fraction_terms(ctx, &squarefree_numerator, &factors, variable)?;

    let variable_expr = ctx.var(variable);
    let mut factored = ctx.num(1);
    let mut decomposition = ctx.num(0);
    for (factor, term) in factors.iter().zip(&terms) {
        factored = build_backend_product(ctx, factored, term.factor_expr);
        let piece_numerator = match factor {
            SquarefreeFactor::Linear { .. } => ctx.add(Expr::Number(term.alpha.clone())),
            SquarefreeFactor::Quadratic { .. } => {
                let alpha_expr = ctx.add(Expr::Number(term.alpha.clone()));
                let linear = build_backend_product(ctx, alpha_expr, variable_expr);
                let beta_expr = ctx.add(Expr::Number(term.beta.clone()));
                build_backend_sum(ctx, linear, beta_expr)
            }
        };
        let piece = ctx.add(Expr::Div(piece_numerator, term.factor_expr));
        decomposition = build_backend_sum(ctx, decomposition, piece);
    }

    Some(GeneralRationalNarrationParts {
        rational_part,
        factored_denominator: factored,
        decomposition,
    })
}

/// Partial-fraction decomposition for the user-facing `apart(p/q, x)` operation.
/// Mirrors the integration narration but starts at denominator degree 2 (so the
/// canonical `1/(x²−1) → 1/2/(x−1) − 1/2/(x+1)` is covered) WITHOUT lowering the
/// integration path's own degree-3 floor — keeping that backend's observability
/// lane byte-identical. PROPER fractions only (deg p < deg q); a denominator that
/// does not split into rational linear/quadratic factors declines to a residual.
pub fn apart_decomposition_expr(
    ctx: &mut Context,
    integrand: ExprId,
    variable: &str,
) -> Option<ExprId> {
    const APART_MIN_DENOMINATOR_DEGREE: usize = 2;

    let (numerator_expr, denominator_expr) = match ctx.get(integrand) {
        Expr::Div(numerator, denominator) => (*numerator, *denominator),
        _ => return None,
    };
    let denominator =
        crate::polynomial::Polynomial::from_expr(ctx, denominator_expr, variable).ok()?;
    let degree = denominator.degree();
    if !(APART_MIN_DENOMINATOR_DEGREE..=GENERAL_RATIONAL_MAX_DENOMINATOR_DEGREE).contains(&degree) {
        return None;
    }
    let numerator = crate::polynomial::Polynomial::from_expr(ctx, numerator_expr, variable).ok()?;
    if numerator.is_zero() {
        return None;
    }
    let leading = denominator.leading_coeff();
    let denominator = denominator.div_scalar(&leading);
    let numerator = numerator.div_scalar(&leading);

    // Improper fraction (deg num >= deg den): polynomial-divide first so the classical ladder runs on
    // the PROPER remainder, then prepend the polynomial quotient — `num/den = q + r/den`. A proper
    // fraction keeps the previous behaviour (zero quotient, remainder = numerator).
    let (quotient, remainder) = if numerator.degree() >= degree {
        numerator.div_rem(&denominator).ok()?
    } else {
        (
            crate::polynomial::Polynomial::new(vec![BigRational::zero()], variable.to_string()),
            numerator,
        )
    };

    let proper_expr = if remainder.is_zero() {
        None
    } else {
        Some(apart_classical_ladder_decomposition(
            ctx,
            &remainder,
            &denominator,
            variable,
        )?)
    };
    let quotient_expr = (!quotient.is_zero()).then(|| quotient.to_expr(ctx));
    match (quotient_expr, proper_expr) {
        (Some(q), Some(p)) => Some(ctx.add(Expr::Add(q, p))),
        (Some(q), None) => Some(q),
        (None, Some(p)) => Some(p),
        (None, None) => None,
    }
}

/// Classical partial-fraction decomposition for the `apart` command.
///
/// For each distinct irreducible factor `P` of the monic denominator `D` with
/// multiplicity `m`, emit the full tower
///   `A_1/P + A_2/P^2 + ... + A_m/P^m`               (linear `P = x - r`)
///   `(B_1 x + C_1)/P + ... + (B_m x + C_m)/P^m`     (irreducible quadratic `P`)
/// solving every coefficient at once by undetermined coefficients
/// (`sum_k column_k * coeff_k = numerator`, exact rational linear solve with the
/// shared `solve_rational_linear_system`).
///
/// This is the decomposition of the INTEGRAND, *not* the Ostrogradsky/Hermite
/// rational part of its integral. A repeated root `(x - r)^m` keeps its proper
/// `1/(x - r)^k` tower instead of being collapsed to a single `1/(x - r)` plus a
/// reduced rational part: the latter is correct for `integrate`, but a
/// non-equivalent answer for `apart` (it drops e.g. the `1/(2(x-1)^2)` term of
/// `1/((x-1)^2 (x+1))`). Squarefree denominators reduce to one column per factor,
/// reproducing the previous behaviour exactly.
fn apart_classical_ladder_decomposition(
    ctx: &mut Context,
    numerator: &crate::polynomial::Polynomial,
    denominator: &crate::polynomial::Polynomial,
    variable: &str,
) -> Option<ExprId> {
    let (_repeated_part, squarefree_part) = squarefree_split(denominator)?;
    let factors = split_squarefree_factors(&squarefree_part)?;
    if factors.is_empty() {
        return None;
    }

    let unknowns = denominator.degree();
    if numerator.degree() >= unknowns {
        return None;
    }
    let x_poly = crate::polynomial::Polynomial::new(
        vec![BigRational::zero(), BigRational::one()],
        variable.to_string(),
    );

    // For each distinct factor, record its monic polynomial and its multiplicity
    // in D, and append the ladder basis columns D/P^k (one for a linear factor,
    // an `x*` and a plain copy for a quadratic) in tower order. The solved
    // coefficients are read back in this very order below.
    let mut factor_polys: Vec<crate::polynomial::Polynomial> = Vec::with_capacity(factors.len());
    let mut multiplicities: Vec<usize> = Vec::with_capacity(factors.len());
    let mut basis: Vec<crate::polynomial::Polynomial> = Vec::new();
    for factor in &factors {
        let poly = squarefree_factor_poly(factor, variable);
        let multiplicity = polynomial_factor_multiplicity(denominator, &poly)?;
        let mut power = crate::polynomial::Polynomial::one(variable.to_string());
        for _ in 1..=multiplicity {
            power = power.mul(&poly);
            let (cofactor, remainder) = denominator.div_rem(&power).ok()?;
            if !remainder.is_zero() {
                return None;
            }
            match factor {
                SquarefreeFactor::Linear { .. } => basis.push(cofactor),
                SquarefreeFactor::Quadratic { .. } => {
                    basis.push(cofactor.mul(&x_poly));
                    basis.push(cofactor);
                }
            }
        }
        factor_polys.push(poly);
        multiplicities.push(multiplicity);
    }

    if basis.len() != unknowns {
        return None;
    }

    let mut matrix = vec![vec![BigRational::zero(); unknowns]; unknowns];
    let mut rhs = vec![BigRational::zero(); unknowns];
    for row in 0..unknowns {
        for (column, column_poly) in basis.iter().enumerate() {
            matrix[row][column] = column_poly
                .coeffs
                .get(row)
                .cloned()
                .unwrap_or_else(BigRational::zero);
        }
        rhs[row] = numerator
            .coeffs
            .get(row)
            .cloned()
            .unwrap_or_else(BigRational::zero);
    }
    let solution = crate::symbolic_integration_support::solve_rational_linear_system(matrix, rhs)?;

    // Emit terms in the same order the basis columns were built. Zero
    // coefficients are dropped so e.g. `1/(x-1)^2` renders without a `0/(x-1)`.
    let variable_expr = ctx.var(variable);
    let mut decomposition: Option<ExprId> = None;
    let mut column = 0usize;
    for (factor, (poly, multiplicity)) in
        factors.iter().zip(factor_polys.iter().zip(&multiplicities))
    {
        let factor_expr = poly.to_expr(ctx);
        for power in 1..=*multiplicity {
            let piece_numerator = match factor {
                SquarefreeFactor::Linear { .. } => {
                    let alpha = solution[column].clone();
                    column += 1;
                    if alpha.is_zero() {
                        continue;
                    }
                    ctx.add(Expr::Number(alpha))
                }
                SquarefreeFactor::Quadratic { .. } => {
                    let alpha = solution[column].clone();
                    let beta = solution[column + 1].clone();
                    column += 2;
                    match (alpha.is_zero(), beta.is_zero()) {
                        (true, true) => continue,
                        (true, false) => ctx.add(Expr::Number(beta)),
                        (false, true) => {
                            let alpha_expr = ctx.add(Expr::Number(alpha));
                            build_backend_product(ctx, alpha_expr, variable_expr)
                        }
                        (false, false) => {
                            let alpha_expr = ctx.add(Expr::Number(alpha));
                            let linear = build_backend_product(ctx, alpha_expr, variable_expr);
                            let beta_expr = ctx.add(Expr::Number(beta));
                            build_backend_sum(ctx, linear, beta_expr)
                        }
                    }
                }
            };
            let denom_expr = if power == 1 {
                factor_expr
            } else {
                let exponent = ctx.num(power as i64);
                ctx.add(Expr::Pow(factor_expr, exponent))
            };
            let piece = ctx.add(Expr::Div(piece_numerator, denom_expr));
            decomposition = Some(match decomposition {
                None => piece,
                Some(acc) => build_backend_sum(ctx, acc, piece),
            });
        }
    }

    decomposition
}

/// The monic polynomial of a squarefree factor: `x - r` (linear) or
/// `x^2 + b x + c` (irreducible quadratic).
fn squarefree_factor_poly(
    factor: &SquarefreeFactor,
    variable: &str,
) -> crate::polynomial::Polynomial {
    match factor {
        SquarefreeFactor::Linear { root } => crate::polynomial::Polynomial::new(
            vec![-root.clone(), BigRational::one()],
            variable.to_string(),
        ),
        SquarefreeFactor::Quadratic {
            linear_b,
            constant_c,
        } => crate::polynomial::Polynomial::new(
            vec![constant_c.clone(), linear_b.clone(), BigRational::one()],
            variable.to_string(),
        ),
    }
}

/// Multiplicity of the monic irreducible factor `P` in `D`: the largest `m`
/// with `P^m | D`. Returns `None` if `P` does not divide `D` (a broken caller
/// invariant), bounded by `deg D` so it can never loop unboundedly.
fn polynomial_factor_multiplicity(
    denominator: &crate::polynomial::Polynomial,
    factor: &crate::polynomial::Polynomial,
) -> Option<usize> {
    let mut remaining = denominator.clone();
    let mut multiplicity = 0usize;
    while multiplicity <= denominator.degree() {
        let (quotient, remainder) = remaining.div_rem(factor).ok()?;
        if !remainder.is_zero() {
            break;
        }
        remaining = quotient;
        multiplicity += 1;
    }
    (multiplicity > 0).then_some(multiplicity)
}

/// D1 = gcd(D, D') (the repeated part), D2 = D/D1 (squarefree, carrying
/// every distinct irreducible factor of D exactly once); both monic.
fn squarefree_split(
    denominator: &crate::polynomial::Polynomial,
) -> Option<(crate::polynomial::Polynomial, crate::polynomial::Polynomial)> {
    let derivative = denominator.derivative();
    let gcd = denominator.gcd(&derivative);
    let repeated = gcd.div_scalar(&gcd.leading_coeff());
    let (squarefree, remainder) = denominator.div_rem(&repeated).ok()?;
    if !remainder.is_zero() {
        return None;
    }
    let squarefree = squarefree.div_scalar(&squarefree.leading_coeff());
    Some((repeated, squarefree))
}

/// Horowitz-Ostrogradsky: solve N = P'*D2 - P*T + Q*D1 with
/// T = D1'*D2/D1 (a polynomial), deg P < deg D1, deg Q < deg D2.
/// The integral is then N/D = (P/D1)' + Q/D2 exactly.
fn ostrogradsky_reduce(
    numerator: &crate::polynomial::Polynomial,
    repeated: &crate::polynomial::Polynomial,
    squarefree: &crate::polynomial::Polynomial,
    variable: &str,
) -> Option<(crate::polynomial::Polynomial, crate::polynomial::Polynomial)> {
    let repeated_derivative = repeated.derivative();
    let (transfer, transfer_remainder) =
        repeated_derivative.mul(squarefree).div_rem(repeated).ok()?;
    if !transfer_remainder.is_zero() {
        return None;
    }

    let p_unknowns = repeated.degree();
    let q_unknowns = squarefree.degree();
    let unknowns = p_unknowns + q_unknowns;
    let mut columns: Vec<crate::polynomial::Polynomial> = Vec::with_capacity(unknowns);
    for power in 0..p_unknowns {
        // Column for the coefficient of x^power in P: derivative part
        // minus the transfer part.
        let mut basis_coeffs = vec![BigRational::zero(); power + 1];
        basis_coeffs[power] = BigRational::one();
        let basis = crate::polynomial::Polynomial::new(basis_coeffs, variable.to_string());
        let column = basis
            .derivative()
            .mul(squarefree)
            .sub(&basis.mul(&transfer));
        columns.push(column);
    }
    for power in 0..q_unknowns {
        let mut basis_coeffs = vec![BigRational::zero(); power + 1];
        basis_coeffs[power] = BigRational::one();
        let basis = crate::polynomial::Polynomial::new(basis_coeffs, variable.to_string());
        columns.push(basis.mul(repeated));
    }

    let mut matrix = vec![vec![BigRational::zero(); unknowns]; unknowns];
    let mut rhs = vec![BigRational::zero(); unknowns];
    for row in 0..unknowns {
        for (column_index, column) in columns.iter().enumerate() {
            matrix[row][column_index] = column
                .coeffs
                .get(row)
                .cloned()
                .unwrap_or_else(BigRational::zero);
        }
        rhs[row] = numerator
            .coeffs
            .get(row)
            .cloned()
            .unwrap_or_else(BigRational::zero);
    }
    let solution = crate::symbolic_integration_support::solve_rational_linear_system(matrix, rhs)?;

    let p =
        crate::polynomial::Polynomial::new(solution[..p_unknowns].to_vec(), variable.to_string());
    let q =
        crate::polynomial::Polynomial::new(solution[p_unknowns..].to_vec(), variable.to_string());
    Some((p, q))
}

/// Split a monic squarefree polynomial into rational-root linear factors
/// plus irreducible numeric quadratics, using only rational roots and the
/// even-substitution resolvent. Anything else (irrational real poles,
/// quartics that need general factorization) returns None.
fn split_squarefree_factors(
    squarefree: &crate::polynomial::Polynomial,
) -> Option<Vec<SquarefreeFactor>> {
    if root_search_constant_too_large(squarefree) {
        return None;
    }
    let mut factors = Vec::new();
    for piece in squarefree.factor_rational_roots() {
        match piece.degree() {
            0 => continue,
            1 => {
                let root = -&piece.coeffs[0] / &piece.coeffs[1];
                factors.push(SquarefreeFactor::Linear { root });
            }
            2 => {
                let monic = piece.div_scalar(&piece.leading_coeff());
                push_quadratic_or_bail(&monic, &mut factors)?;
            }
            _ => {
                let monic = piece.div_scalar(&piece.leading_coeff());
                if monic.even_substitution().is_some() {
                    split_even_residual(&monic, &mut factors)?;
                } else if monic.degree() == 4 {
                    split_general_quartic(&monic, &mut factors)?;
                } else {
                    return None;
                }
            }
        }
    }
    Some(factors)
}

/// Factor a monic squarefree quartic with no rational roots and a
/// nonzero odd part over Q via the resolvent cubic: depress with
/// y = x + c3/4 to y^4 + p*y^2 + q*y + r, find a rational root t0 = a^2
/// of t^3 + 2p*t^2 + (p^2 - 4r)*t - q^2 that is a perfect square, and
/// recover (y^2 + a*y + b)(y^2 - a*y + c) with b, c from coefficient
/// matching, un-shifting back to x. Both factors must be irreducible.
fn split_general_quartic(
    quartic: &crate::polynomial::Polynomial,
    factors: &mut Vec<SquarefreeFactor>,
) -> Option<()> {
    let two = BigRational::from_integer(2.into());
    let four = BigRational::from_integer(4.into());
    let c3 = quartic.coeffs[3].clone();
    let c2 = quartic.coeffs[2].clone();
    let c1 = quartic.coeffs[1].clone();
    let c0 = quartic.coeffs[0].clone();
    let shift = &c3 / &four;
    let shift2 = &shift * &shift;
    let p = &c2 - BigRational::from_integer(6.into()) * &shift2;
    let q = BigRational::from_integer(8.into()) * &shift2 * &shift - &two * &c2 * &shift + &c1;
    let r = -BigRational::from_integer(3.into()) * &shift2 * &shift2 + &c2 * &shift2 - &c1 * &shift
        + &c0;
    if q.is_zero() {
        // Depressed-even quartic with a nonzero original odd part is rare;
        // out of scope for this arm (the even path owns even inputs).
        return None;
    }

    let resolvent = crate::polynomial::Polynomial::new(
        vec![
            -(&q * &q),
            &p * &p - &four * &r,
            &two * &p,
            BigRational::one(),
        ],
        quartic.var.clone(),
    );
    if root_search_constant_too_large(&resolvent) {
        return None;
    }
    for piece in resolvent.factor_rational_roots() {
        if piece.degree() != 1 {
            continue;
        }
        let root = -&piece.coeffs[0] / &piece.coeffs[1];
        if !root.is_positive() {
            continue;
        }
        let Some(a) = rational_positive_square_root(&root) else {
            continue;
        };
        let q_over_a = &q / &a;
        let b = (&p + &root - &q_over_a) / &two;
        let c = (&p + &root + &q_over_a) / &two;
        if &b * &c != r {
            continue;
        }
        // Un-shift y = x + s: y^2 +- a*y + k -> x^2 + (2s +- a)x + (s^2 +- a*s + k).
        let candidates = [
            (&two * &shift + &a, &shift2 + &a * &shift + &b),
            (&two * &shift - &a, &shift2 - &a * &shift + &c),
        ];
        if candidates
            .iter()
            .any(|(lin, con)| lin * lin - &four * con >= BigRational::zero())
        {
            // A reducible factor here means irrational real poles.
            return None;
        }
        for (lin, con) in candidates {
            factors.push(SquarefreeFactor::Quadratic {
                linear_b: lin,
                constant_c: con,
            });
        }
        return Some(());
    }
    None
}

/// Resolve a monic residual of degree >= 3 through u = x^2: rational
/// roots u0 < 0 give irreducible quadratics x^2 - u0, perfect-square
/// roots u0 = s^2 give the linear pair x -+ s.
fn split_even_residual(
    residual: &crate::polynomial::Polynomial,
    factors: &mut Vec<SquarefreeFactor>,
) -> Option<()> {
    let resolvent = residual.even_substitution()?;
    if root_search_constant_too_large(&resolvent) {
        return None;
    }
    for piece in resolvent.factor_rational_roots() {
        match piece.degree() {
            0 => continue,
            1 => {
                let root = -&piece.coeffs[0] / &piece.coeffs[1];
                if root.is_negative() {
                    factors.push(SquarefreeFactor::Quadratic {
                        linear_b: BigRational::zero(),
                        constant_c: -root,
                    });
                } else if let Some(square_root) = rational_positive_square_root(&root) {
                    factors.push(SquarefreeFactor::Linear {
                        root: square_root.clone(),
                    });
                    factors.push(SquarefreeFactor::Linear { root: -square_root });
                } else {
                    return None;
                }
            }
            2 => {
                let monic = piece.div_scalar(&piece.leading_coeff());
                even_quartic_descent(&monic, factors)?;
            }
            _ => return None,
        }
    }
    Some(())
}

/// Factor x^4 + p*x^2 + r (given as the irreducible-over-rational-roots
/// resolvent u^2 + p*u + r) into the symmetric quadratic pair
/// (x^2 + a*x + b)(x^2 - a*x + b): matching coefficients with the odd
/// term zero forces b^2 = r and a^2 = 2b - p, so the descent succeeds
/// exactly when r is a perfect rational square and 2b - p is a positive
/// perfect rational square (Sophie Germain x^4+4, cyclotomic-style
/// x^4+x^2+1). Both emitted quadratics must be irreducible.
fn even_quartic_descent(
    resolvent_piece: &crate::polynomial::Polynomial,
    factors: &mut Vec<SquarefreeFactor>,
) -> Option<()> {
    let p = resolvent_piece.coeffs[1].clone();
    let r = resolvent_piece.coeffs[0].clone();
    let two = BigRational::from_integer(2.into());
    let four = BigRational::from_integer(4.into());
    let b_magnitude = rational_positive_square_root(&r)?;
    for b in [b_magnitude.clone(), -b_magnitude] {
        let a_square = &two * &b - &p;
        if !a_square.is_positive() {
            continue;
        }
        let Some(a) = rational_positive_square_root(&a_square) else {
            continue;
        };
        // Each factor x^2 +- a*x + b must be irreducible: a^2 - 4b < 0.
        if &a_square - &four * &b >= BigRational::zero() {
            continue;
        }
        factors.push(SquarefreeFactor::Quadratic {
            linear_b: a.clone(),
            constant_c: b.clone(),
        });
        factors.push(SquarefreeFactor::Quadratic {
            linear_b: -a,
            constant_c: b,
        });
        return Some(());
    }
    None
}

/// Integrate `c / (x^4 + p*x^2 + r)` when the even quartic factors over the
/// reals into the SYMMETRIC SURD pair `(x^2 + a*x + s)(x^2 - a*x + s)` with
/// `s = sqrt(r)` rational but `a = sqrt(2*s - p)` IRRATIONAL — exactly the
/// case `even_quartic_descent` declines, because `SquarefreeFactor` carries
/// only rational coefficients (e.g. `1/(x^4 - x^2 + 1)` needs `a = sqrt(3)`,
/// `1/(x^4 - 3*x^2 + 4)` needs `a = sqrt(7)`). With a constant numerator the
/// integrand is even, so the partial fraction collapses to the closed form
///   F = c*[ (1/(4*a*s))*(ln(x^2+a*x+s) - ln(x^2-a*x+s))
///         + (1/(2*s*D))*(arctan((2x+a)/D) + arctan((2x-a)/D)) ],  D = sqrt(2s+p),
/// which the downstream differentiation oracle verifies (so a bad surd match
/// degrades to an honest residual, never a wrong answer).
///
/// Every keep/drop decision is EXACT `BigRational`: `r` a positive perfect
/// square (so `s` is rational), `a^2 = 2s - p` a positive NON-square (a perfect
/// square is owned by `even_quartic_descent`; excluding it keeps that lane
/// byte-identical), and `a^2 - 4s < 0` so both quadratics are irreducible —
/// which also forces `D^2 = 2s + p = 4s - a^2 > 0`, keeping the arctan real and
/// the logs argument strictly positive (no `|.|` needed).
pub(super) fn symmetric_surd_even_quartic_antiderivative(
    ctx: &mut Context,
    integrand: ExprId,
    variable: &str,
) -> Option<ExprId> {
    let (numerator_expr, denominator_expr) = match ctx.get(integrand) {
        Expr::Div(numerator, denominator) => (*numerator, *denominator),
        _ => return None,
    };
    let denominator =
        crate::polynomial::Polynomial::from_expr(ctx, denominator_expr, variable).ok()?;
    if denominator.degree() != 4 {
        return None;
    }
    let numerator = crate::polynomial::Polynomial::from_expr(ctx, numerator_expr, variable).ok()?;
    // Constant numerator only: the symmetric closed form relies on an even integrand.
    if numerator.is_zero() || numerator.degree() != 0 {
        return None;
    }

    // Normalize the denominator monic; fold its leading coefficient into the
    // numerator constant so the match works over `x^4 + p*x^2 + r`.
    let leading = denominator.leading_coeff();
    let denominator = denominator.div_scalar(&leading);
    if !denominator.coeffs[3].is_zero() || !denominator.coeffs[1].is_zero() {
        return None;
    }
    let p = denominator.coeffs[2].clone();
    let r = denominator.coeffs[0].clone();
    let c = &numerator.coeffs[0] / &leading;
    if c.is_zero() {
        return None;
    }

    let two = BigRational::from_integer(2.into());
    let four = BigRational::from_integer(4.into());
    // s = sqrt(r) must be rational.
    let s = rational_positive_square_root(&r)?;
    // a^2 = 2s - p must be a positive NON-square (square => even_quartic_descent owns it).
    let a_square = &two * &s - &p;
    if !a_square.is_positive() || rational_positive_square_root(&a_square).is_some() {
        return None;
    }
    // Both factors irreducible: a^2 - 4s < 0 (also forces D^2 = 2s + p > 0).
    if &a_square - &four * &s >= BigRational::zero() {
        return None;
    }
    let d_square = &two * &s + &p;

    let a = build_numeric_radius_expr(ctx, &a_square);
    let d = build_numeric_radius_expr(ctx, &d_square);
    let s_expr = ctx.add(Expr::Number(s.clone()));
    let variable_expr = ctx.var(variable);
    let two_expr = ctx.add(Expr::Number(two.clone()));
    let x_square = ctx.add(Expr::Pow(variable_expr, two_expr));
    let ax = build_backend_product(ctx, a, variable_expr);

    // q_plus = x^2 + a*x + s, q_minus = x^2 - a*x + s.
    let q_plus = {
        let head = build_backend_sum(ctx, x_square, ax);
        build_backend_sum(ctx, head, s_expr)
    };
    let q_minus = {
        let head = build_backend_difference(ctx, x_square, ax);
        build_backend_sum(ctx, head, s_expr)
    };

    // Divide by a surd radius, skipping the no-op `/1` when it is a perfect square.
    let divide_by_radius = |ctx: &mut Context, body: ExprId, radius: ExprId| -> ExprId {
        if is_one(ctx, radius) {
            body
        } else {
            ctx.add(Expr::Div(body, radius))
        }
    };

    // Log part: (c/(4*s)) * (ln(q_plus) - ln(q_minus)) / a   (a is always irrational here).
    let log_piece = {
        let ln_plus = ctx.call_builtin(BuiltinFn::Ln, vec![q_plus]);
        let ln_minus = ctx.call_builtin(BuiltinFn::Ln, vec![q_minus]);
        let log_diff = build_backend_difference(ctx, ln_plus, ln_minus);
        let scalar = ctx.add(Expr::Number(&c / (&four * &s)));
        let scaled = build_backend_product(ctx, scalar, log_diff);
        divide_by_radius(ctx, scaled, a)
    };

    // Arctan part: (c/(2*s)) * (arctan((2x+a)/D) + arctan((2x-a)/D)) / D.
    let arctan_piece = {
        let two_x = build_backend_product(ctx, two_expr, variable_expr);
        let center_plus = build_backend_sum(ctx, two_x, a);
        let center_minus = build_backend_difference(ctx, two_x, a);
        let arg_plus = divide_by_radius(ctx, center_plus, d);
        let arg_minus = divide_by_radius(ctx, center_minus, d);
        let arctan_plus = ctx.call_builtin(BuiltinFn::Arctan, vec![arg_plus]);
        let arctan_minus = ctx.call_builtin(BuiltinFn::Arctan, vec![arg_minus]);
        let arctan_sum = build_backend_sum(ctx, arctan_plus, arctan_minus);
        let scalar = ctx.add(Expr::Number(&c / (&two * &s)));
        let scaled = build_backend_product(ctx, scalar, arctan_sum);
        divide_by_radius(ctx, scaled, d)
    };

    Some(build_backend_sum(ctx, log_piece, arctan_piece))
}

fn push_quadratic_or_bail(
    monic: &crate::polynomial::Polynomial,
    factors: &mut Vec<SquarefreeFactor>,
) -> Option<()> {
    let b = monic.coeffs[1].clone();
    let c = monic.coeffs[0].clone();
    let four = BigRational::from_integer(4.into());
    if &b * &b - four * &c >= BigRational::zero() {
        // Real irrational poles: out of scope for exact partial fractions.
        return None;
    }
    factors.push(SquarefreeFactor::Quadratic {
        linear_b: b,
        constant_c: c,
    });
    Some(())
}

fn root_search_constant_too_large(poly: &crate::polynomial::Polynomial) -> bool {
    use num_traits::Signed;
    let bound = num_bigint::BigInt::from(GENERAL_RATIONAL_MAX_ROOT_SEARCH_CONSTANT);
    poly.coeffs
        .first()
        .is_some_and(|constant| constant.numer().abs() > bound || constant.denom().abs() > bound)
}

/// Mixed partial fractions over the split factors: one coefficient per
/// linear factor (stored in `alpha`), an (alpha, beta) pair per quadratic.
/// Every result is returned as a MultiQuadraticFactorTerm so quadratic
/// pieces reuse the cycle-14 arctan/log assembler.
fn mixed_partial_fraction_terms(
    ctx: &mut Context,
    numerator: &crate::polynomial::Polynomial,
    factors: &[SquarefreeFactor],
    variable: &str,
) -> Option<Vec<MultiQuadraticFactorTerm>> {
    let mut factor_polys: Vec<crate::polynomial::Polynomial> = Vec::with_capacity(factors.len());
    for factor in factors {
        match factor {
            SquarefreeFactor::Linear { root } => {
                factor_polys.push(crate::polynomial::Polynomial::new(
                    vec![-root.clone(), BigRational::one()],
                    variable.to_string(),
                ));
            }
            SquarefreeFactor::Quadratic {
                linear_b,
                constant_c,
            } => {
                factor_polys.push(crate::polynomial::Polynomial::new(
                    vec![constant_c.clone(), linear_b.clone(), BigRational::one()],
                    variable.to_string(),
                ));
            }
        }
    }
    let mut denominator = crate::polynomial::Polynomial::one(variable.to_string());
    for poly in &factor_polys {
        denominator = denominator.mul(poly);
    }
    let unknowns = denominator.degree();
    if numerator.degree() >= unknowns {
        return None;
    }

    let x_poly = crate::polynomial::Polynomial::new(
        vec![BigRational::zero(), BigRational::one()],
        variable.to_string(),
    );
    let mut columns: Vec<crate::polynomial::Polynomial> = Vec::new();
    for (factor, poly) in factors.iter().zip(&factor_polys) {
        let (cofactor, remainder) = denominator.div_rem(poly).ok()?;
        if !remainder.is_zero() {
            return None;
        }
        match factor {
            SquarefreeFactor::Linear { .. } => columns.push(cofactor),
            SquarefreeFactor::Quadratic { .. } => {
                columns.push(cofactor.mul(&x_poly));
                columns.push(cofactor);
            }
        }
    }
    if columns.len() != unknowns {
        return None;
    }

    let mut matrix = vec![vec![BigRational::zero(); unknowns]; unknowns];
    let mut rhs = vec![BigRational::zero(); unknowns];
    for row in 0..unknowns {
        for (column_index, column) in columns.iter().enumerate() {
            matrix[row][column_index] = column
                .coeffs
                .get(row)
                .cloned()
                .unwrap_or_else(BigRational::zero);
        }
        rhs[row] = numerator
            .coeffs
            .get(row)
            .cloned()
            .unwrap_or_else(BigRational::zero);
    }
    let solution = crate::symbolic_integration_support::solve_rational_linear_system(matrix, rhs)?;

    let mut terms = Vec::with_capacity(factors.len());
    let mut cursor = 0;
    for (factor, poly) in factors.iter().zip(&factor_polys) {
        match factor {
            SquarefreeFactor::Linear { .. } => {
                terms.push(MultiQuadraticFactorTerm {
                    factor_expr: poly.to_expr(ctx),
                    linear_b: BigRational::zero(),
                    constant_c: BigRational::zero(),
                    alpha: solution[cursor].clone(),
                    beta: BigRational::zero(),
                });
                cursor += 1;
            }
            SquarefreeFactor::Quadratic {
                linear_b,
                constant_c,
            } => {
                terms.push(MultiQuadraticFactorTerm {
                    factor_expr: poly.to_expr(ctx),
                    linear_b: linear_b.clone(),
                    constant_c: constant_c.clone(),
                    alpha: solution[cursor].clone(),
                    beta: solution[cursor + 1].clone(),
                });
                cursor += 2;
            }
        }
    }
    Some(terms)
}

fn build_numeric_shifted_center(
    ctx: &mut Context,
    variable_term: ExprId,
    shift: &BigRational,
) -> ExprId {
    if shift.is_zero() {
        variable_term
    } else if shift.is_negative() {
        let magnitude = ctx.add(Expr::Number(-shift.clone()));
        build_backend_difference(ctx, variable_term, magnitude)
    } else {
        let shift_expr = ctx.add(Expr::Number(shift.clone()));
        build_backend_sum(ctx, variable_term, shift_expr)
    }
}

fn build_numeric_radius_expr(ctx: &mut Context, radius_square: &BigRational) -> ExprId {
    if let Some(root) = rational_positive_square_root(radius_square) {
        ctx.add(Expr::Number(root))
    } else {
        let square_expr = ctx.add(Expr::Number(radius_square.clone()));
        ctx.call_builtin(BuiltinFn::Sqrt, vec![square_expr])
    }
}

fn try_hermite_positive_quadratic_log_derivative_probe(
    ctx: &mut Context,
    integrand: ExprId,
    variable: &str,
    probe_runner: &mut AlgorithmicIntegrationProbeRunner,
) -> AlgorithmicIntegrationProbeResult {
    let mut pole_conditions = Vec::new();
    let mut constant_policy = IntegrationConstantPolicy::ArbitraryConstantOmitted;
    let (antiderivative, slope_condition, radius_condition) =
        if let Some((coefficient, variable_slope, denominator, required_condition)) =
            positive_quadratic_log_derivative_parts(ctx, integrand, variable)
        {
            (
                build_positive_quadratic_log_derivative_antiderivative(
                    ctx,
                    coefficient,
                    &variable_slope,
                    denominator,
                ),
                variable_slope.required_condition(),
                required_condition,
            )
        } else if let Some(parts) =
            positive_quadratic_linear_numerator_parts(ctx, integrand, variable)
        {
            (
                build_positive_quadratic_linear_numerator_antiderivative(
                    ctx,
                    parts.variable_coefficient,
                    parts.constant_term,
                    parts.variable_expr,
                    &parts.variable_slope,
                    parts.denominator,
                    parts.radius,
                ),
                parts.variable_slope.required_condition(),
                parts.required_condition,
            )
        } else if let Some(parts) =
            indefinite_square_denominator_reciprocal_parts(ctx, integrand, variable)
        {
            pole_conditions.push(ConditionPredicate::NonZero(parts.left_pole));
            pole_conditions.push(ConditionPredicate::NonZero(parts.right_pole));
            constant_policy = IntegrationConstantPolicy::ComponentLocalConstant;
            (
                build_indefinite_square_denominator_linear_numerator_antiderivative(
                    ctx,
                    parts.variable_coefficient,
                    parts.constant_term,
                    parts.variable_expr,
                    &parts.variable_slope,
                    parts.denominator,
                    parts.radius,
                ),
                parts.variable_slope.required_condition(),
                parts.radius_condition,
            )
        } else {
            return AlgorithmicIntegrationProbeResult::NoMatch(
                positive_quadratic_log_derivative_no_match_reason(ctx, integrand, variable),
            );
        };

    let mut candidate = AlgorithmicIntegrationCandidate::unverified(
        integrand,
        variable,
        antiderivative,
        AlgorithmicIntegrationMethod::Hermite,
    );
    candidate.constant_policy = constant_policy;
    if let Some(condition) = radius_condition {
        candidate.required_conditions.push(condition);
    }
    candidate.required_conditions.extend(pole_conditions);
    if let Some(condition) = slope_condition {
        candidate.required_conditions.push(condition);
    }
    if !probe_runner.try_verification_check() {
        candidate.mark_budget_exceeded();
        return AlgorithmicIntegrationProbeResult::Candidate(candidate);
    }
    verify_antiderivative_by_differentiation(ctx, &mut candidate);

    AlgorithmicIntegrationProbeResult::Candidate(candidate)
}

fn try_heurisch_sine_log_derivative_probe(
    ctx: &mut Context,
    integrand: ExprId,
    variable: &str,
    probe_runner: &mut AlgorithmicIntegrationProbeRunner,
) -> AlgorithmicIntegrationProbeResult {
    let Some(denominator) = sine_log_derivative_denominator(ctx, integrand, variable) else {
        return AlgorithmicIntegrationProbeResult::NoMatch(
            AlgorithmicIntegrationProbeNoMatchReason::ShapeMismatch,
        );
    };

    let abs_denominator = ctx.call_builtin(BuiltinFn::Abs, vec![denominator]);
    let antiderivative = ctx.call_builtin(BuiltinFn::Ln, vec![abs_denominator]);
    let mut candidate = AlgorithmicIntegrationCandidate::unverified(
        integrand,
        variable,
        antiderivative,
        AlgorithmicIntegrationMethod::HeurischProbe,
    );
    candidate
        .required_conditions
        .push(ConditionPredicate::NonZero(denominator));
    if !probe_runner.try_verification_check() {
        candidate.mark_budget_exceeded();
        return AlgorithmicIntegrationProbeResult::Candidate(candidate);
    }
    verify_antiderivative_by_differentiation(ctx, &mut candidate);

    AlgorithmicIntegrationProbeResult::Candidate(candidate)
}

pub(super) fn affine_denominator_linear_numerator_parts(
    ctx: &mut Context,
    expr: ExprId,
    variable: &str,
) -> Result<AffineDenominatorLinearNumeratorParts, AlgorithmicIntegrationProbeNoMatchReason> {
    match ctx.get(expr).clone() {
        Expr::Div(numerator, denominator) => {
            affine_denominator_linear_numerator_div_parts(ctx, numerator, denominator, variable)
        }
        Expr::Mul(left, right) => {
            if let Some(parts) =
                scaled_affine_denominator_linear_numerator_parts(ctx, left, right, variable)
                    .or_else(|| {
                        scaled_affine_denominator_linear_numerator_parts(ctx, right, left, variable)
                    })
            {
                return Ok(parts);
            }
            if matches!(ctx.get(left), Expr::Div(_, _)) || matches!(ctx.get(right), Expr::Div(_, _))
            {
                Err(AlgorithmicIntegrationProbeNoMatchReason::NumeratorPolicyMismatch)
            } else {
                Err(AlgorithmicIntegrationProbeNoMatchReason::ShapeMismatch)
            }
        }
        Expr::Neg(inner) => {
            let negative_one = ctx.num(-1);
            scaled_affine_denominator_linear_numerator_parts(ctx, negative_one, inner, variable)
                .ok_or(AlgorithmicIntegrationProbeNoMatchReason::ShapeMismatch)
        }
        _ => Err(AlgorithmicIntegrationProbeNoMatchReason::ShapeMismatch),
    }
}

fn affine_denominator_linear_numerator_div_parts(
    ctx: &mut Context,
    numerator: ExprId,
    denominator: ExprId,
    variable: &str,
) -> Result<AffineDenominatorLinearNumeratorParts, AlgorithmicIntegrationProbeNoMatchReason> {
    if is_supported_scaled_affine_reciprocal_numerator(ctx, numerator, variable) {
        let Some(denominator_slope) = affine_denominator_slope(ctx, denominator, variable) else {
            return Err(AlgorithmicIntegrationProbeNoMatchReason::DenominatorPolicyMismatch);
        };
        return Ok(AffineDenominatorLinearNumeratorParts {
            quotient_coefficient: ctx.num(0),
            remainder: numerator,
            denominator,
            denominator_slope,
        });
    }

    let Some(denominator_slope) = affine_denominator_slope(ctx, denominator, variable) else {
        return Err(AlgorithmicIntegrationProbeNoMatchReason::NumeratorPolicyMismatch);
    };
    let Some((quotient_coefficient, remainder)) =
        linear_numerator_decomposition_terms(ctx, numerator, denominator, variable).or_else(|| {
            affine_quotient_remainder_from_linear_terms(
                ctx,
                numerator,
                denominator,
                &denominator_slope,
                variable,
            )
        })
    else {
        return Err(AlgorithmicIntegrationProbeNoMatchReason::NumeratorPolicyMismatch);
    };
    if (is_zero(ctx, quotient_coefficient) && is_zero(ctx, remainder))
        || !is_supported_backend_linear_coefficient_for_affine_slope(
            ctx,
            quotient_coefficient,
            variable,
            &denominator_slope,
        )
        || !is_supported_backend_linear_coefficient_for_affine_slope(
            ctx,
            remainder,
            variable,
            &denominator_slope,
        )
    {
        return Err(AlgorithmicIntegrationProbeNoMatchReason::NumeratorPolicyMismatch);
    }
    Ok(AffineDenominatorLinearNumeratorParts {
        quotient_coefficient,
        remainder,
        denominator,
        denominator_slope,
    })
}

fn scaled_affine_denominator_linear_numerator_parts(
    ctx: &mut Context,
    scale: ExprId,
    quotient: ExprId,
    variable: &str,
) -> Option<AffineDenominatorLinearNumeratorParts> {
    if contains_named_var(ctx, scale, variable)
        || !is_supported_backend_linear_coefficient(ctx, scale, variable)
    {
        return None;
    }
    let Expr::Div(numerator, denominator) = ctx.get(quotient).clone() else {
        return None;
    };
    let scaled_numerator = build_backend_product(ctx, scale, numerator);
    affine_denominator_linear_numerator_div_parts(ctx, scaled_numerator, denominator, variable).ok()
}

fn affine_quotient_remainder_from_linear_terms(
    ctx: &mut Context,
    numerator: ExprId,
    denominator: ExprId,
    denominator_slope: &BackendAffineSlope,
    variable: &str,
) -> Option<(ExprId, ExprId)> {
    let (numerator_slope, numerator_intercept) =
        backend_affine_linear_terms(ctx, numerator, variable)?;
    let (_, denominator_intercept) = backend_affine_linear_terms(ctx, denominator, variable)?;
    if is_zero(ctx, numerator_slope) {
        return None;
    }

    let quotient_coefficient =
        divide_backend_coefficient_by_slope(ctx, numerator_slope, denominator_slope);
    let scaled_denominator_intercept =
        build_backend_product(ctx, quotient_coefficient, denominator_intercept);
    let remainder =
        build_backend_difference(ctx, numerator_intercept, scaled_denominator_intercept);
    Some((quotient_coefficient, remainder))
}

pub(super) fn backend_affine_linear_terms(
    ctx: &mut Context,
    expr: ExprId,
    variable: &str,
) -> Option<(ExprId, ExprId)> {
    if is_variable(ctx, expr, variable) {
        let one = ctx.num(1);
        let zero = ctx.num(0);
        return Some((one, zero));
    }
    if is_supported_backend_linear_coefficient(ctx, expr, variable) {
        let zero = ctx.num(0);
        return Some((zero, expr));
    }

    match ctx.get(expr).clone() {
        Expr::Add(left, right) => {
            let (left_slope, left_intercept) = backend_affine_linear_terms(ctx, left, variable)?;
            let (right_slope, right_intercept) = backend_affine_linear_terms(ctx, right, variable)?;
            Some((
                build_backend_sum(ctx, left_slope, right_slope),
                build_backend_sum(ctx, left_intercept, right_intercept),
            ))
        }
        Expr::Sub(left, right) => {
            let (left_slope, left_intercept) = backend_affine_linear_terms(ctx, left, variable)?;
            let (right_slope, right_intercept) = backend_affine_linear_terms(ctx, right, variable)?;
            Some((
                build_backend_difference(ctx, left_slope, right_slope),
                build_backend_difference(ctx, left_intercept, right_intercept),
            ))
        }
        Expr::Neg(inner) => {
            let (slope, intercept) = backend_affine_linear_terms(ctx, inner, variable)?;
            Some((
                negate_backend_expr(ctx, slope),
                negate_backend_expr(ctx, intercept),
            ))
        }
        Expr::Mul(left, right) => {
            if is_supported_backend_linear_coefficient(ctx, left, variable)
                && !contains_named_var(ctx, left, variable)
            {
                let (slope, intercept) = backend_affine_linear_terms(ctx, right, variable)?;
                return Some((
                    build_backend_product(ctx, left, slope),
                    build_backend_product(ctx, left, intercept),
                ));
            }
            if is_supported_backend_linear_coefficient(ctx, right, variable)
                && !contains_named_var(ctx, right, variable)
            {
                let (slope, intercept) = backend_affine_linear_terms(ctx, left, variable)?;
                return Some((
                    build_backend_product(ctx, right, slope),
                    build_backend_product(ctx, right, intercept),
                ));
            }
            None
        }
        _ => None,
    }
}

fn build_affine_denominator_remainder_antiderivative(
    ctx: &mut Context,
    remainder: ExprId,
    denominator: ExprId,
    denominator_slope: &BackendAffineSlope,
) -> ExprId {
    if is_zero(ctx, remainder) {
        return ctx.num(0);
    }

    let abs_denominator = ctx.call_builtin(BuiltinFn::Abs, vec![denominator]);
    let log_denominator = ctx.call_builtin(BuiltinFn::Ln, vec![abs_denominator]);
    let antiderivative_scale =
        divide_backend_coefficient_by_slope(ctx, remainder, denominator_slope);
    if is_one(ctx, antiderivative_scale) {
        log_denominator
    } else {
        build_backend_product(ctx, antiderivative_scale, log_denominator)
    }
}

fn is_supported_scaled_affine_reciprocal_numerator(
    ctx: &Context,
    expr: ExprId,
    variable: &str,
) -> bool {
    numeric_value(ctx, expr)
        .map(|value| !value.is_zero())
        .unwrap_or(false)
        || is_supported_external_coefficient(ctx, expr, variable)
}

fn is_supported_backend_linear_coefficient(ctx: &Context, expr: ExprId, variable: &str) -> bool {
    is_supported_backend_linear_coefficient_inner(ctx, expr, variable, 0, None)
}

pub(super) fn is_supported_backend_linear_coefficient_for_affine_slope(
    ctx: &Context,
    expr: ExprId,
    variable: &str,
    affine_slope: &BackendAffineSlope,
) -> bool {
    is_supported_backend_linear_coefficient_inner(ctx, expr, variable, 0, Some(affine_slope))
}

fn is_supported_backend_linear_coefficient_inner(
    ctx: &Context,
    expr: ExprId,
    variable: &str,
    depth: usize,
    allowed_symbolic_divisor: Option<&BackendAffineSlope>,
) -> bool {
    if depth >= BACKEND_EXTERNAL_COEFFICIENT_DEPTH {
        return false;
    }
    if is_zero(ctx, expr) {
        return true;
    }
    if numeric_value(ctx, expr)
        .map(|value| !value.is_zero())
        .unwrap_or(false)
    {
        return true;
    }
    if is_supported_external_coefficient(ctx, expr, variable) {
        return true;
    }
    if contains_named_var(ctx, expr, variable) {
        return false;
    }

    match ctx.get(expr) {
        Expr::Add(left, right) | Expr::Sub(left, right) | Expr::Mul(left, right) => {
            is_supported_backend_linear_coefficient_inner(
                ctx,
                *left,
                variable,
                depth + 1,
                allowed_symbolic_divisor,
            ) && is_supported_backend_linear_coefficient_inner(
                ctx,
                *right,
                variable,
                depth + 1,
                allowed_symbolic_divisor,
            )
        }
        Expr::Div(numerator, denominator) => {
            if let Some(denominator_value) = numeric_value(ctx, *denominator) {
                return !denominator_value.is_zero()
                    && is_supported_backend_linear_coefficient_inner(
                        ctx,
                        *numerator,
                        variable,
                        depth + 1,
                        allowed_symbolic_divisor,
                    );
            }
            backend_affine_slope_allows_divisor(ctx, *denominator, allowed_symbolic_divisor)
                && is_supported_backend_linear_coefficient_inner(
                    ctx,
                    *numerator,
                    variable,
                    depth + 1,
                    allowed_symbolic_divisor,
                )
        }
        Expr::Neg(inner) => is_supported_backend_linear_coefficient_inner(
            ctx,
            *inner,
            variable,
            depth + 1,
            allowed_symbolic_divisor,
        ),
        _ => false,
    }
}

fn backend_affine_slope_allows_divisor(
    ctx: &Context,
    divisor: ExprId,
    allowed_symbolic_divisor: Option<&BackendAffineSlope>,
) -> bool {
    let Some(BackendAffineSlope::Symbolic(allowed_divisor)) = allowed_symbolic_divisor else {
        return false;
    };

    divisor == *allowed_divisor
        || SemanticEqualityChecker::new(ctx).are_equal(divisor, *allowed_divisor)
}

fn is_supported_nonzero_backend_coefficient(ctx: &Context, expr: ExprId, variable: &str) -> bool {
    numeric_value(ctx, expr)
        .map(|value| !value.is_zero())
        .unwrap_or(false)
        || is_supported_external_coefficient(ctx, expr, variable)
}

fn positive_quadratic_log_derivative_parts(
    ctx: &mut Context,
    expr: ExprId,
    variable: &str,
) -> Option<(
    ExprId,
    BackendAffineSlope,
    ExprId,
    Option<ConditionPredicate>,
)> {
    match ctx.get(expr).clone() {
        Expr::Div(numerator, denominator) => {
            let (variable_expr, variable_slope, radius_square, required_condition) =
                positive_shifted_quadratic_denominator_parts(ctx, denominator, variable)?;
            let coefficient =
                affine_variable_coefficient_expr(ctx, numerator, variable_expr, variable).or_else(
                    || {
                        derivative_multiple_numerator_coefficient(
                            ctx,
                            numerator,
                            variable_expr,
                            &variable_slope,
                            variable,
                        )
                    },
                )?;
            let denominator =
                build_positive_quadratic_denominator(ctx, variable_expr, radius_square);
            Some((coefficient, variable_slope, denominator, required_condition))
        }
        _ => None,
    }
}

/// Recognizes distributed derivative-multiple numerators against the
/// reconstructed affine center, such as `m*s*x + b*m` over a denominator
/// with center `s*x + b`. Accepts only decompositions whose constant
/// component is exactly zero; mixed numerators stay owned by the
/// linear-numerator route.
fn derivative_multiple_numerator_coefficient(
    ctx: &mut Context,
    numerator: ExprId,
    variable_expr: ExprId,
    variable_slope: &BackendAffineSlope,
    variable: &str,
) -> Option<ExprId> {
    let (coefficient, constant_term) =
        linear_numerator_decomposition_terms(ctx, numerator, variable_expr, variable)?;
    if !is_zero(ctx, constant_term) || is_zero(ctx, coefficient) {
        return None;
    }
    if !is_supported_backend_linear_coefficient_for_affine_slope(
        ctx,
        coefficient,
        variable,
        variable_slope,
    ) {
        return None;
    }
    Some(coefficient)
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct PositiveQuadraticLinearNumeratorParts {
    variable_coefficient: ExprId,
    constant_term: ExprId,
    variable_expr: ExprId,
    variable_slope: BackendAffineSlope,
    denominator: ExprId,
    radius: ExprId,
    required_condition: Option<ConditionPredicate>,
}

fn positive_quadratic_linear_numerator_parts(
    ctx: &mut Context,
    expr: ExprId,
    variable: &str,
) -> Option<PositiveQuadraticLinearNumeratorParts> {
    match ctx.get(expr).clone() {
        Expr::Div(numerator, denominator) => {
            let (variable_expr, variable_slope, radius_square, required_condition) =
                positive_shifted_quadratic_denominator_parts(ctx, denominator, variable)?;
            let (variable_coefficient, constant_term) =
                linear_numerator_decomposition_terms(ctx, numerator, variable_expr, variable)?;
            if is_zero(ctx, constant_term) {
                return None;
            }
            if !is_supported_backend_linear_coefficient_for_affine_slope(
                ctx,
                variable_coefficient,
                variable,
                &variable_slope,
            ) || !is_supported_backend_linear_coefficient_for_affine_slope(
                ctx,
                constant_term,
                variable,
                &variable_slope,
            ) {
                return None;
            }
            let radius = positive_radius_expr(ctx, radius_square, &required_condition)?;
            let denominator =
                build_positive_quadratic_denominator(ctx, variable_expr, radius_square);
            Some(PositiveQuadraticLinearNumeratorParts {
                variable_coefficient,
                constant_term,
                variable_expr,
                variable_slope,
                denominator,
                radius,
                required_condition,
            })
        }
        _ => None,
    }
}

fn build_positive_quadratic_denominator(
    ctx: &mut Context,
    variable_expr: ExprId,
    radius_square: ExprId,
) -> ExprId {
    let two = ctx.num(2);
    let variable_square = ctx.add(Expr::Pow(variable_expr, two));
    build_backend_sum(ctx, variable_square, radius_square)
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct IndefiniteSquareDenominatorReciprocalParts {
    variable_coefficient: ExprId,
    constant_term: ExprId,
    variable_expr: ExprId,
    variable_slope: BackendAffineSlope,
    denominator: ExprId,
    radius: ExprId,
    radius_condition: Option<ConditionPredicate>,
    left_pole: ExprId,
    right_pole: ExprId,
}

fn indefinite_square_denominator_reciprocal_parts(
    ctx: &mut Context,
    expr: ExprId,
    variable: &str,
) -> Option<IndefiniteSquareDenominatorReciprocalParts> {
    match ctx.get(expr).clone() {
        Expr::Div(numerator, denominator) => {
            let (variable_expr, variable_slope, radius, radius_condition) =
                indefinite_square_denominator_parts(ctx, denominator, variable)?;
            let (variable_coefficient, constant_term) =
                linear_numerator_decomposition_terms(ctx, numerator, variable_expr, variable)?;
            if is_zero(ctx, variable_coefficient) && is_zero(ctx, constant_term) {
                return None;
            }
            if !is_supported_backend_linear_coefficient(ctx, variable_coefficient, variable)
                || !is_supported_backend_linear_coefficient(ctx, constant_term, variable)
            {
                return None;
            }
            let left_pole = build_backend_difference(ctx, variable_expr, radius);
            let right_pole = build_backend_sum(ctx, variable_expr, radius);
            Some(IndefiniteSquareDenominatorReciprocalParts {
                variable_coefficient,
                constant_term,
                variable_expr,
                variable_slope,
                denominator,
                radius,
                radius_condition,
                left_pole,
                right_pole,
            })
        }
        _ => None,
    }
}

fn indefinite_square_denominator_parts(
    ctx: &mut Context,
    expr: ExprId,
    variable: &str,
) -> Option<(
    ExprId,
    BackendAffineSlope,
    ExprId,
    Option<ConditionPredicate>,
)> {
    match ctx.get(expr).clone() {
        Expr::Sub(left, right) => {
            let (variable_expr, variable_slope) = affine_variable_from_square(ctx, left, variable)?;
            let radius_condition = positive_radius_square_required_condition(ctx, right, variable)?;
            let radius = positive_radius_expr(ctx, right, &radius_condition)?;
            Some((variable_expr, variable_slope, radius, radius_condition))
        }
        _ => None,
    }
}

fn positive_quadratic_log_derivative_no_match_reason(
    ctx: &mut Context,
    expr: ExprId,
    variable: &str,
) -> AlgorithmicIntegrationProbeNoMatchReason {
    match ctx.get(expr).clone() {
        Expr::Div(numerator, denominator) => {
            let Some((variable_expr, _, radius_square, required_condition)) =
                positive_shifted_quadratic_denominator_parts(ctx, denominator, variable)
            else {
                return positive_quadratic_denominator_no_match_reason(ctx, denominator, variable);
            };
            if affine_variable_coefficient_expr(ctx, numerator, variable_expr, variable).is_none() {
                if linear_numerator_decomposition_terms(ctx, numerator, variable_expr, variable)
                    .is_some()
                    && positive_radius_expr(ctx, radius_square, &required_condition).is_none()
                {
                    return AlgorithmicIntegrationProbeNoMatchReason::RadiusPolicyMismatch;
                }
                return AlgorithmicIntegrationProbeNoMatchReason::NumeratorDerivativeMismatch;
            }
            AlgorithmicIntegrationProbeNoMatchReason::ShapeMismatch
        }
        _ => AlgorithmicIntegrationProbeNoMatchReason::ShapeMismatch,
    }
}

fn positive_quadratic_denominator_no_match_reason(
    ctx: &mut Context,
    expr: ExprId,
    variable: &str,
) -> AlgorithmicIntegrationProbeNoMatchReason {
    if positive_quadratic_radius_policy_mismatch(ctx, expr, variable) {
        AlgorithmicIntegrationProbeNoMatchReason::RadiusPolicyMismatch
    } else {
        AlgorithmicIntegrationProbeNoMatchReason::DenominatorPolicyMismatch
    }
}

fn positive_quadratic_radius_policy_mismatch(
    ctx: &mut Context,
    expr: ExprId,
    variable: &str,
) -> bool {
    match ctx.get(expr).clone() {
        Expr::Add(left, right) => {
            positive_quadratic_radius_policy_mismatch_pair(ctx, left, right, variable)
                || positive_quadratic_radius_policy_mismatch_pair(ctx, right, left, variable)
        }
        Expr::Sub(left, right) => {
            affine_variable_from_square(ctx, left, variable).is_some()
                && backend_radius_policy_candidate(ctx, right, variable)
        }
        _ => false,
    }
}

fn positive_quadratic_radius_policy_mismatch_pair(
    ctx: &mut Context,
    square_candidate: ExprId,
    radius_candidate: ExprId,
    variable: &str,
) -> bool {
    affine_variable_from_square(ctx, square_candidate, variable).is_some()
        && backend_radius_policy_candidate(ctx, radius_candidate, variable)
        && positive_radius_square_required_condition(ctx, radius_candidate, variable).is_none()
}

fn backend_radius_policy_candidate(ctx: &Context, expr: ExprId, variable: &str) -> bool {
    !contains_named_var(ctx, expr, variable)
}

fn build_positive_quadratic_log_derivative_antiderivative(
    ctx: &mut Context,
    numerator_coefficient: ExprId,
    variable_slope: &BackendAffineSlope,
    denominator: ExprId,
) -> ExprId {
    let log_denominator = ctx.call_builtin(BuiltinFn::Ln, vec![denominator]);
    let halved_coefficient = halve_backend_coefficient(ctx, numerator_coefficient);
    let antiderivative_coefficient =
        divide_backend_coefficient_by_slope(ctx, halved_coefficient, variable_slope);
    if is_one(ctx, antiderivative_coefficient) {
        log_denominator
    } else {
        ctx.add(Expr::Mul(antiderivative_coefficient, log_denominator))
    }
}

fn build_positive_quadratic_linear_numerator_antiderivative(
    ctx: &mut Context,
    variable_coefficient: ExprId,
    constant_term: ExprId,
    variable_expr: ExprId,
    variable_slope: &BackendAffineSlope,
    denominator: ExprId,
    radius: ExprId,
) -> ExprId {
    let log_part = if is_zero(ctx, variable_coefficient) {
        ctx.num(0)
    } else {
        build_positive_quadratic_log_derivative_antiderivative(
            ctx,
            variable_coefficient,
            variable_slope,
            denominator,
        )
    };
    let arctan_part = if is_zero(ctx, constant_term) {
        ctx.num(0)
    } else {
        build_positive_quadratic_constant_numerator_antiderivative(
            ctx,
            constant_term,
            variable_expr,
            variable_slope,
            radius,
        )
    };
    build_backend_sum(ctx, log_part, arctan_part)
}

fn build_positive_quadratic_constant_numerator_antiderivative(
    ctx: &mut Context,
    constant_term: ExprId,
    variable_expr: ExprId,
    variable_slope: &BackendAffineSlope,
    radius: ExprId,
) -> ExprId {
    let slope_scaled_constant =
        divide_backend_coefficient_by_slope(ctx, constant_term, variable_slope);
    if is_one(ctx, radius) {
        let arctan_variable = ctx.call_builtin(BuiltinFn::Arctan, vec![variable_expr]);
        return build_backend_product(ctx, slope_scaled_constant, arctan_variable);
    }

    let scaled_variable = ctx.add(Expr::Div(variable_expr, radius));
    let arctan_scaled_variable = ctx.call_builtin(BuiltinFn::Arctan, vec![scaled_variable]);
    let scaled_constant =
        divide_backend_coefficient_by_symbolic(ctx, slope_scaled_constant, radius);
    build_backend_product(ctx, scaled_constant, arctan_scaled_variable)
}

fn build_indefinite_square_denominator_linear_numerator_antiderivative(
    ctx: &mut Context,
    variable_coefficient: ExprId,
    constant_term: ExprId,
    variable_expr: ExprId,
    variable_slope: &BackendAffineSlope,
    denominator: ExprId,
    radius: ExprId,
) -> ExprId {
    let log_derivative_part = if is_zero(ctx, variable_coefficient) {
        ctx.num(0)
    } else {
        let abs_denominator = ctx.call_builtin(BuiltinFn::Abs, vec![denominator]);
        let log_denominator = ctx.call_builtin(BuiltinFn::Ln, vec![abs_denominator]);
        let halved_coefficient = halve_backend_coefficient(ctx, variable_coefficient);
        let antiderivative_scale =
            divide_backend_coefficient_by_slope(ctx, halved_coefficient, variable_slope);
        build_backend_product(ctx, antiderivative_scale, log_denominator)
    };

    let reciprocal_part = if is_zero(ctx, constant_term) {
        ctx.num(0)
    } else {
        build_indefinite_square_denominator_reciprocal_antiderivative(
            ctx,
            constant_term,
            variable_expr,
            variable_slope,
            radius,
        )
    };

    build_backend_sum(ctx, log_derivative_part, reciprocal_part)
}

fn build_indefinite_square_denominator_reciprocal_antiderivative(
    ctx: &mut Context,
    numerator: ExprId,
    variable_expr: ExprId,
    variable_slope: &BackendAffineSlope,
    radius: ExprId,
) -> ExprId {
    let left_pole = build_backend_difference(ctx, variable_expr, radius);
    let right_pole = build_backend_sum(ctx, variable_expr, radius);
    let abs_left = ctx.call_builtin(BuiltinFn::Abs, vec![left_pole]);
    let abs_right = ctx.call_builtin(BuiltinFn::Abs, vec![right_pole]);
    let log_left = ctx.call_builtin(BuiltinFn::Ln, vec![abs_left]);
    let log_right = ctx.call_builtin(BuiltinFn::Ln, vec![abs_right]);
    let log_difference = build_backend_difference(ctx, log_left, log_right);
    let slope_scaled_numerator =
        divide_backend_coefficient_by_slope(ctx, numerator, variable_slope);
    let radius_scaled_numerator =
        divide_backend_coefficient_by_symbolic(ctx, slope_scaled_numerator, radius);
    let antiderivative_scale = divide_backend_coefficient_by_numeric(
        ctx,
        radius_scaled_numerator,
        BigRational::from_integer(2.into()),
    );
    build_backend_product(ctx, antiderivative_scale, log_difference)
}

fn sine_log_derivative_denominator(ctx: &Context, expr: ExprId, variable: &str) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Div(numerator, denominator)
            if is_builtin_of_variable(ctx, *numerator, BuiltinFn::Cos, variable)
                && is_builtin_of_variable(ctx, *denominator, BuiltinFn::Sin, variable) =>
        {
            Some(*denominator)
        }
        _ => None,
    }
}

/// Public recognizer for denominators the Hermite positive-quadratic method
/// accepts: compact `(s*x + b)^2 + a` or expanded
/// `s^2*x^2 + 2*b*s*x + b^2 + a` with a variable-free radius. Returns the
/// radius expression so condition presentation can drop source-denominator
/// conditions that are already implied by the displayed `Positive(radius)`
/// backend condition, without duplicating center reconstruction outside the
/// backend.
pub fn backend_positive_quadratic_denominator_radius(
    ctx: &mut Context,
    denominator: ExprId,
    variable: &str,
) -> Option<ExprId> {
    positive_shifted_quadratic_denominator_parts(ctx, denominator, variable)
        .map(|(_, _, radius, _)| radius)
}

pub(super) fn positive_shifted_quadratic_denominator_parts(
    ctx: &mut Context,
    expr: ExprId,
    variable: &str,
) -> Option<(
    ExprId,
    BackendAffineSlope,
    ExprId,
    Option<ConditionPredicate>,
)> {
    match ctx.get(expr).clone() {
        Expr::Add(left, right) => {
            if let Some(required_condition) =
                positive_radius_square_required_condition(ctx, right, variable)
            {
                let (variable_expr, variable_slope) =
                    affine_variable_from_square(ctx, left, variable)?;
                return Some((variable_expr, variable_slope, right, required_condition));
            }
            if let Some(required_condition) =
                positive_radius_square_required_condition(ctx, left, variable)
            {
                let (variable_expr, variable_slope) =
                    affine_variable_from_square(ctx, right, variable)?;
                return Some((variable_expr, variable_slope, left, required_condition));
            }
            expanded_positive_shifted_quadratic_denominator_parts(ctx, expr, variable)
        }
        _ => expanded_positive_shifted_quadratic_denominator_parts(ctx, expr, variable),
    }
}

fn expanded_positive_shifted_quadratic_denominator_parts(
    ctx: &mut Context,
    expr: ExprId,
    variable: &str,
) -> Option<(
    ExprId,
    BackendAffineSlope,
    ExprId,
    Option<ConditionPredicate>,
)> {
    let terms = add_terms_signed(ctx, expr);
    if terms.len() != 4 {
        return expanded_numeric_positive_shifted_quadratic_denominator_parts(ctx, expr, variable);
    }

    for (radius_index, (radius_candidate, radius_sign)) in terms.iter().copied().enumerate() {
        if radius_sign != Sign::Pos {
            continue;
        }
        let Some(required_condition) =
            positive_radius_square_required_condition(ctx, radius_candidate, variable)
        else {
            continue;
        };

        for (quadratic_index, (quadratic_candidate, quadratic_sign)) in
            terms.iter().copied().enumerate()
        {
            if quadratic_index == radius_index || quadratic_sign != Sign::Pos {
                continue;
            }
            let Some(variable_slope) =
                expanded_affine_square_quadratic_slope(ctx, quadratic_candidate, variable)
            else {
                continue;
            };

            for (intercept_index, (intercept_square_candidate, intercept_sign)) in
                terms.iter().copied().enumerate()
            {
                if intercept_index == radius_index
                    || intercept_index == quadratic_index
                    || intercept_sign != Sign::Pos
                {
                    continue;
                }
                let Some(intercept) =
                    squared_external_radius_base(ctx, intercept_square_candidate, variable)
                else {
                    continue;
                };

                let Some((cross_candidate, cross_sign)) = terms
                    .iter()
                    .copied()
                    .enumerate()
                    .find(|(index, _)| {
                        *index != radius_index
                            && *index != quadratic_index
                            && *index != intercept_index
                    })
                    .map(|(_, term)| term)
                else {
                    continue;
                };
                if !expanded_affine_square_cross_term_matches(
                    ctx,
                    cross_candidate,
                    &variable_slope,
                    intercept,
                    variable,
                ) {
                    continue;
                }

                let variable_expr = build_expanded_affine_square_variable_expr(
                    ctx,
                    &variable_slope,
                    intercept,
                    cross_sign,
                    variable,
                );
                return Some((
                    variable_expr,
                    variable_slope,
                    radius_candidate,
                    required_condition,
                ));
            }
        }
    }

    expanded_numeric_positive_shifted_quadratic_denominator_parts(ctx, expr, variable)
}

/// Numeric fallback for expanded positive-quadratic denominators whose folded
/// coefficients defeat the structural symbolic pattern, such as
/// `x^2 + 4*x + 4 + a` (center `x + 2`, radius `a`) or
/// `4*x^2 + 16*x + 16 + a` (center `2*x + 4`). Coefficients are compared as
/// rationals: the quadratic coefficient must be the square of a positive
/// rational slope, the folded numeric constant must equal the squared
/// intercept exactly, and exactly one plus-signed variable-free term remains
/// as the radius under the family's existing radius policy. Fully numeric
/// denominators (no symbolic radius term) are rejected here and stay owned by
/// the educational route.
fn expanded_numeric_positive_shifted_quadratic_denominator_parts(
    ctx: &mut Context,
    expr: ExprId,
    variable: &str,
) -> Option<(
    ExprId,
    BackendAffineSlope,
    ExprId,
    Option<ConditionPredicate>,
)> {
    let terms = add_terms_signed(ctx, expr);
    let mut quadratic_coefficient = BigRational::zero();
    let mut linear_coefficient = BigRational::zero();
    let mut numeric_constant = BigRational::zero();
    let mut radius: Option<(ExprId, Option<ConditionPredicate>)> = None;
    for (term, sign) in terms {
        let signed = |value: BigRational| match sign {
            Sign::Pos => value,
            Sign::Neg => -value,
        };
        if let Some(coefficient) = numeric_named_variable_square_coefficient(ctx, term, variable) {
            quadratic_coefficient += signed(coefficient);
            continue;
        }
        if let Some(coefficient) = numeric_named_variable_linear_coefficient(ctx, term, variable) {
            linear_coefficient += signed(coefficient);
            continue;
        }
        if let Some(value) = backend_numeric_constant_value(ctx, term, 0) {
            numeric_constant += signed(value);
            continue;
        }
        if sign == Sign::Pos && radius.is_none() {
            if let Some(required_condition) =
                positive_radius_square_required_condition(ctx, term, variable)
            {
                radius = Some((term, required_condition));
                continue;
            }
        }
        return None;
    }
    let (radius, required_condition) = radius?;
    let slope = rational_positive_square_root(&quadratic_coefficient)?;
    if linear_coefficient.is_zero() {
        return None;
    }
    let two = BigRational::from_integer(2.into());
    let intercept = linear_coefficient / (two * &slope);
    if numeric_constant != &intercept * &intercept {
        return None;
    }

    let variable_expr = ctx.var(variable);
    let variable_term = if slope.is_one() {
        variable_expr
    } else {
        let slope_expr = ctx.add(Expr::Number(slope.clone()));
        build_backend_product(ctx, slope_expr, variable_expr)
    };
    let center = if intercept.is_negative() {
        let magnitude = ctx.add(Expr::Number(-intercept.clone()));
        build_backend_difference(ctx, variable_term, magnitude)
    } else {
        let intercept_expr = ctx.add(Expr::Number(intercept.clone()));
        build_backend_sum(ctx, variable_term, intercept_expr)
    };
    Some((
        center,
        BackendAffineSlope::Numeric(slope),
        radius,
        required_condition,
    ))
}

fn numeric_named_variable_square_coefficient(
    ctx: &Context,
    expr: ExprId,
    variable: &str,
) -> Option<BigRational> {
    if is_named_variable_square_factor(ctx, expr, variable) {
        return Some(BigRational::one());
    }
    let factors = backend_mul_factors(ctx, expr);
    let mut coefficient = BigRational::one();
    let mut variable_square_seen = false;
    for factor in factors {
        if is_named_variable_square_factor(ctx, factor, variable) {
            if variable_square_seen {
                return None;
            }
            variable_square_seen = true;
            continue;
        }
        coefficient *= numeric_value(ctx, factor)?;
    }
    variable_square_seen.then_some(coefficient)
}

fn numeric_named_variable_linear_coefficient(
    ctx: &Context,
    expr: ExprId,
    variable: &str,
) -> Option<BigRational> {
    if is_variable(ctx, expr, variable) {
        return Some(BigRational::one());
    }
    let factors = backend_mul_factors(ctx, expr);
    let mut coefficient = BigRational::one();
    let mut variable_seen = false;
    for factor in factors {
        if is_variable(ctx, factor, variable) {
            if variable_seen {
                return None;
            }
            variable_seen = true;
            continue;
        }
        coefficient *= numeric_value(ctx, factor)?;
    }
    variable_seen.then_some(coefficient)
}

fn rational_positive_square_root(value: &BigRational) -> Option<BigRational> {
    if !value.is_positive() {
        return None;
    }
    let numerator_root = value.numer().sqrt();
    if &(&numerator_root * &numerator_root) != value.numer() {
        return None;
    }
    let denominator_root = value.denom().sqrt();
    if &(&denominator_root * &denominator_root) != value.denom() {
        return None;
    }
    Some(BigRational::new(numerator_root, denominator_root))
}

fn expanded_affine_square_quadratic_slope(
    ctx: &mut Context,
    expr: ExprId,
    variable: &str,
) -> Option<BackendAffineSlope> {
    if is_named_variable_square_factor(ctx, expr, variable) {
        return Some(BackendAffineSlope::Numeric(BigRational::one()));
    }

    let mut factors = backend_mul_factors(ctx, expr);
    let variable_square_index = factors
        .iter()
        .position(|factor| is_named_variable_square_factor(ctx, *factor, variable))?;
    factors.remove(variable_square_index);
    if factors.is_empty() {
        return Some(BackendAffineSlope::Numeric(BigRational::one()));
    }
    if factors.len() != 1 {
        return None;
    }
    let slope_square_candidate = factors[0];
    let slope = squared_external_radius_base(ctx, slope_square_candidate, variable)?;
    affine_slope_coefficient(ctx, slope, variable)
}

fn is_named_variable_square_factor(ctx: &Context, expr: ExprId, variable: &str) -> bool {
    match ctx.get(expr) {
        Expr::Pow(base, exponent) if is_two(ctx, *exponent) => is_variable(ctx, *base, variable),
        _ => false,
    }
}

fn expanded_affine_square_cross_term_matches(
    ctx: &mut Context,
    expr: ExprId,
    variable_slope: &BackendAffineSlope,
    intercept: ExprId,
    variable: &str,
) -> bool {
    let two = ctx.num(2);
    let slope = backend_affine_slope_expr(ctx, variable_slope);
    let variable_expr = ctx.var(variable);
    let expected = build_backend_factor_product(ctx, vec![two, slope, intercept, variable_expr]);
    expr == expected || SemanticEqualityChecker::new(ctx).are_equal(expr, expected)
}

fn build_expanded_affine_square_variable_expr(
    ctx: &mut Context,
    variable_slope: &BackendAffineSlope,
    intercept: ExprId,
    cross_sign: Sign,
    variable: &str,
) -> ExprId {
    let slope = backend_affine_slope_expr(ctx, variable_slope);
    let variable_expr = ctx.var(variable);
    let variable_term = build_backend_product(ctx, slope, variable_expr);
    let signed_intercept = match cross_sign {
        Sign::Pos => intercept,
        Sign::Neg => negate_backend_expr(ctx, intercept),
    };
    build_backend_sum(ctx, variable_term, signed_intercept)
}

fn backend_affine_slope_expr(ctx: &mut Context, slope: &BackendAffineSlope) -> ExprId {
    match slope {
        BackendAffineSlope::Numeric(value) => ctx.add(Expr::Number(value.clone())),
        BackendAffineSlope::Symbolic(expr) => *expr,
    }
}

fn positive_radius_square_required_condition(
    ctx: &Context,
    expr: ExprId,
    variable: &str,
) -> Option<Option<ConditionPredicate>> {
    if let Some(value) = numeric_value(ctx, expr) {
        return value.is_positive().then_some(None);
    }
    if let Some(required_condition) =
        squared_external_radius_required_condition(ctx, expr, variable)
    {
        return Some(required_condition);
    }
    is_supported_external_coefficient(ctx, expr, variable)
        .then_some(Some(ConditionPredicate::Positive(expr)))
}

fn squared_external_radius_required_condition(
    ctx: &Context,
    expr: ExprId,
    variable: &str,
) -> Option<Option<ConditionPredicate>> {
    let radius = squared_external_radius_base(ctx, expr, variable)?;
    if let Some(value) = numeric_value(ctx, radius) {
        return (!value.is_zero()).then_some(None);
    }
    Some(Some(ConditionPredicate::NonZero(radius)))
}

fn squared_external_radius_base(ctx: &Context, expr: ExprId, variable: &str) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Pow(base, exponent)
            if is_two(ctx, *exponent)
                && is_supported_external_coefficient(ctx, *base, variable) =>
        {
            Some(*base)
        }
        _ => None,
    }
}

fn affine_variable_from_square(
    ctx: &mut Context,
    expr: ExprId,
    variable: &str,
) -> Option<(ExprId, BackendAffineSlope)> {
    match ctx.get(expr) {
        Expr::Pow(base, exponent) if is_two(ctx, *exponent) => {
            affine_variable_expr(ctx, *base, variable)
        }
        _ => None,
    }
}

pub(super) fn affine_variable_expr(
    ctx: &mut Context,
    expr: ExprId,
    variable: &str,
) -> Option<(ExprId, BackendAffineSlope)> {
    affine_denominator_slope(ctx, expr, variable).map(|slope| (expr, slope))
}

fn affine_variable_coefficient_expr(
    ctx: &mut Context,
    expr: ExprId,
    variable_expr: ExprId,
    variable: &str,
) -> Option<ExprId> {
    if expr == variable_expr || SemanticEqualityChecker::new(ctx).are_equal(expr, variable_expr) {
        return Some(ctx.num(1));
    }

    match ctx.get(expr).clone() {
        Expr::Mul(left, right) => {
            if (right == variable_expr
                || SemanticEqualityChecker::new(ctx).are_equal(right, variable_expr))
                && is_supported_nonzero_backend_coefficient(ctx, left, variable)
            {
                return Some(left);
            }
            if (left == variable_expr
                || SemanticEqualityChecker::new(ctx).are_equal(left, variable_expr))
                && is_supported_nonzero_backend_coefficient(ctx, right, variable)
            {
                return Some(right);
            }
            None
        }
        _ => None,
    }
}

pub(super) fn linear_numerator_decomposition_terms(
    ctx: &mut Context,
    expr: ExprId,
    variable_expr: ExprId,
    variable: &str,
) -> Option<(ExprId, ExprId)> {
    if let Some(coefficient) = affine_variable_coefficient_expr(ctx, expr, variable_expr, variable)
    {
        let zero = ctx.num(0);
        return Some((coefficient, zero));
    }
    if is_supported_external_coefficient(ctx, expr, variable) {
        let zero = ctx.num(0);
        return Some((zero, expr));
    }
    if let Some(decomposition) =
        affine_linear_numerator_decomposition_terms(ctx, expr, variable_expr, variable)
    {
        return Some(decomposition);
    }

    let direct = match ctx.get(expr).clone() {
        Expr::Add(left, right) => {
            let (left_coefficient, left_constant) =
                linear_numerator_decomposition_terms(ctx, left, variable_expr, variable)?;
            let (right_coefficient, right_constant) =
                linear_numerator_decomposition_terms(ctx, right, variable_expr, variable)?;
            let coefficient = build_backend_sum(ctx, left_coefficient, right_coefficient);
            let constant = build_backend_sum(ctx, left_constant, right_constant);
            Some((coefficient, constant))
        }
        Expr::Sub(left, right) => {
            let (left_coefficient, left_constant) =
                linear_numerator_decomposition_terms(ctx, left, variable_expr, variable)?;
            let (right_coefficient, right_constant) =
                linear_numerator_decomposition_terms(ctx, right, variable_expr, variable)?;
            let coefficient = build_backend_difference(ctx, left_coefficient, right_coefficient);
            let constant = build_backend_difference(ctx, left_constant, right_constant);
            Some((coefficient, constant))
        }
        Expr::Neg(inner) => {
            let (coefficient, constant) =
                linear_numerator_decomposition_terms(ctx, inner, variable_expr, variable)?;
            Some((
                negate_backend_expr(ctx, coefficient),
                negate_backend_expr(ctx, constant),
            ))
        }
        _ => None,
    };
    direct
}

fn affine_linear_numerator_decomposition_terms(
    ctx: &mut Context,
    expr: ExprId,
    variable_expr: ExprId,
    variable: &str,
) -> Option<(ExprId, ExprId)> {
    let variable_slope = affine_denominator_slope(ctx, variable_expr, variable)?;
    let (_, variable_intercept) = backend_affine_linear_terms(ctx, variable_expr, variable)?;
    let (numerator_slope, numerator_intercept) = backend_affine_linear_terms(ctx, expr, variable)?;
    if is_zero(ctx, numerator_slope) {
        return None;
    }

    let coefficient = divide_backend_coefficient_by_slope(ctx, numerator_slope, &variable_slope);
    let scaled_variable_intercept = build_backend_product(ctx, coefficient, variable_intercept);
    let constant = build_backend_difference_canceling_sum_term(
        ctx,
        numerator_intercept,
        scaled_variable_intercept,
    );
    Some((coefficient, constant))
}

fn halve_backend_coefficient(ctx: &mut Context, coefficient: ExprId) -> ExprId {
    if let Some(value) = numeric_value(ctx, coefficient) {
        let half = value / BigRational::from_integer(2.into());
        return ctx.add(Expr::Number(half));
    }

    let two = ctx.add(Expr::Number(BigRational::from_integer(2.into())));
    ctx.add(Expr::Div(coefficient, two))
}

fn divide_backend_coefficient_by_numeric(
    ctx: &mut Context,
    coefficient: ExprId,
    divisor: BigRational,
) -> ExprId {
    multiply_backend_numeric_coefficient(ctx, BigRational::one() / divisor, coefficient)
}

fn divide_backend_coefficient_by_slope(
    ctx: &mut Context,
    coefficient: ExprId,
    slope: &BackendAffineSlope,
) -> ExprId {
    match slope {
        BackendAffineSlope::Numeric(value) => {
            divide_backend_coefficient_by_numeric(ctx, coefficient, value.clone())
        }
        BackendAffineSlope::Symbolic(divisor) => {
            divide_backend_coefficient_by_symbolic(ctx, coefficient, *divisor)
        }
    }
}

fn divide_backend_coefficient_by_symbolic(
    ctx: &mut Context,
    coefficient: ExprId,
    divisor: ExprId,
) -> ExprId {
    if coefficient == divisor || SemanticEqualityChecker::new(ctx).are_equal(coefficient, divisor) {
        return ctx.num(1);
    }
    if let Some(stripped) = strip_backend_exact_factor(ctx, coefficient, divisor, "") {
        return stripped;
    }
    ctx.add(Expr::Div(coefficient, divisor))
}

fn is_symbolic_external_coefficient(ctx: &Context, expr: ExprId, variable: &str) -> bool {
    matches!(ctx.get(expr), Expr::Variable(sym_id) if ctx.sym_name(*sym_id) != variable)
        && !contains_named_var(ctx, expr, variable)
}

pub(super) fn is_supported_external_coefficient(
    ctx: &Context,
    expr: ExprId,
    variable: &str,
) -> bool {
    !contains_named_var(ctx, expr, variable)
        && is_supported_external_coefficient_inner(ctx, expr, variable, 0)
}

fn is_supported_external_coefficient_inner(
    ctx: &Context,
    expr: ExprId,
    variable: &str,
    depth: usize,
) -> bool {
    if depth >= BACKEND_EXTERNAL_COEFFICIENT_DEPTH {
        return false;
    }

    match ctx.get(expr) {
        Expr::Number(value) => !value.is_zero(),
        Expr::Variable(sym_id) => ctx.sym_name(*sym_id) != variable,
        Expr::Mul(left, right) => {
            is_supported_external_coefficient_inner(ctx, *left, variable, depth + 1)
                && is_supported_external_coefficient_inner(ctx, *right, variable, depth + 1)
        }
        Expr::Div(numerator, denominator) => {
            let Some(denominator_value) = numeric_value(ctx, *denominator) else {
                return false;
            };
            !denominator_value.is_zero()
                && is_supported_external_coefficient_inner(ctx, *numerator, variable, depth + 1)
        }
        Expr::Neg(inner) => {
            is_supported_external_coefficient_inner(ctx, *inner, variable, depth + 1)
        }
        _ => false,
    }
}

pub(super) fn affine_denominator_slope(
    ctx: &mut Context,
    expr: ExprId,
    variable: &str,
) -> Option<BackendAffineSlope> {
    match ctx.get(expr).clone() {
        Expr::Variable(sym_id) if ctx.sym_name(sym_id) == variable => {
            Some(BackendAffineSlope::Numeric(BigRational::one()))
        }
        Expr::Mul(left, right) => affine_linear_term_coefficient(ctx, left, right, variable),
        Expr::Neg(inner) => {
            let slope = affine_denominator_slope(ctx, inner, variable)?;
            negate_affine_slope(ctx, slope)
        }
        Expr::Add(left, right) => {
            if is_affine_intercept(ctx, right, variable) {
                return affine_denominator_slope(ctx, left, variable);
            }
            if is_affine_intercept(ctx, left, variable) {
                return affine_denominator_slope(ctx, right, variable);
            }
            None
        }
        Expr::Sub(left, right) => {
            if is_affine_intercept(ctx, right, variable) {
                return affine_denominator_slope(ctx, left, variable);
            }
            if is_affine_intercept(ctx, left, variable) {
                let slope = affine_denominator_slope(ctx, right, variable)?;
                return negate_affine_slope(ctx, slope);
            }
            None
        }
        _ => None,
    }
}

fn affine_linear_term_coefficient(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
    variable: &str,
) -> Option<BackendAffineSlope> {
    if is_variable(ctx, right, variable) {
        return affine_slope_coefficient(ctx, left, variable);
    }
    if is_variable(ctx, left, variable) {
        return affine_slope_coefficient(ctx, right, variable);
    }
    if is_supported_external_coefficient(ctx, left, variable) {
        if let Some(slope) = affine_denominator_slope(ctx, right, variable) {
            return multiply_affine_slope(ctx, left, slope);
        }
    }
    if is_supported_external_coefficient(ctx, right, variable) {
        if let Some(slope) = affine_denominator_slope(ctx, left, variable) {
            return multiply_affine_slope(ctx, right, slope);
        }
    }
    None
}

fn affine_slope_coefficient(
    ctx: &Context,
    expr: ExprId,
    variable: &str,
) -> Option<BackendAffineSlope> {
    if let Some(value) = numeric_value(ctx, expr) {
        return (!value.is_zero()).then_some(BackendAffineSlope::Numeric(value));
    }
    if is_supported_external_coefficient(ctx, expr, variable) {
        return Some(BackendAffineSlope::Symbolic(expr));
    }
    None
}

fn multiply_affine_slope(
    ctx: &mut Context,
    coefficient: ExprId,
    slope: BackendAffineSlope,
) -> Option<BackendAffineSlope> {
    match slope {
        BackendAffineSlope::Numeric(value) => {
            if let Some(coefficient_value) = numeric_value(ctx, coefficient) {
                let product = coefficient_value * value;
                return (!product.is_zero()).then_some(BackendAffineSlope::Numeric(product));
            }
            let value_expr = ctx.add(Expr::Number(value));
            Some(BackendAffineSlope::Symbolic(build_backend_product(
                ctx,
                coefficient,
                value_expr,
            )))
        }
        BackendAffineSlope::Symbolic(slope_expr) => Some(BackendAffineSlope::Symbolic(
            build_backend_product(ctx, coefficient, slope_expr),
        )),
    }
}

fn negate_affine_slope(ctx: &mut Context, slope: BackendAffineSlope) -> Option<BackendAffineSlope> {
    match slope {
        BackendAffineSlope::Numeric(value) => Some(BackendAffineSlope::Numeric(-value)),
        BackendAffineSlope::Symbolic(expr) => {
            Some(BackendAffineSlope::Symbolic(ctx.add(Expr::Neg(expr))))
        }
    }
}

fn is_affine_intercept(ctx: &Context, expr: ExprId, variable: &str) -> bool {
    is_numeric_constant(ctx, expr)
        || (is_symbolic_external_coefficient(ctx, expr, variable)
            && !contains_named_var(ctx, expr, variable))
}

fn is_builtin_of_variable(ctx: &Context, expr: ExprId, builtin: BuiltinFn, variable: &str) -> bool {
    match ctx.get(expr) {
        Expr::Function(fn_id, args) if ctx.is_builtin(*fn_id, builtin) && args.len() == 1 => {
            is_variable(ctx, args[0], variable)
        }
        _ => false,
    }
}

pub(super) fn is_variable(ctx: &Context, expr: ExprId, variable: &str) -> bool {
    matches!(ctx.get(expr), Expr::Variable(sym_id) if ctx.sym_name(*sym_id) == variable)
}

fn is_numeric_constant(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Number(_))
}

pub(super) fn numeric_value(ctx: &Context, expr: ExprId) -> Option<BigRational> {
    match ctx.get(expr) {
        Expr::Number(value) => Some(value.clone()),
        _ => None,
    }
}

pub(super) fn backend_numeric_constant_value(
    ctx: &Context,
    expr: ExprId,
    depth: usize,
) -> Option<BigRational> {
    if depth > 4 {
        return None;
    }
    match ctx.get(expr) {
        Expr::Number(value) => Some(value.clone()),
        Expr::Add(left, right) => Some(
            backend_numeric_constant_value(ctx, *left, depth + 1)?
                + backend_numeric_constant_value(ctx, *right, depth + 1)?,
        ),
        Expr::Sub(left, right) => Some(
            backend_numeric_constant_value(ctx, *left, depth + 1)?
                - backend_numeric_constant_value(ctx, *right, depth + 1)?,
        ),
        Expr::Mul(left, right) => Some(
            backend_numeric_constant_value(ctx, *left, depth + 1)?
                * backend_numeric_constant_value(ctx, *right, depth + 1)?,
        ),
        Expr::Div(left, right) => {
            let numerator = backend_numeric_constant_value(ctx, *left, depth + 1)?;
            let denominator = backend_numeric_constant_value(ctx, *right, depth + 1)?;
            (!denominator.is_zero()).then(|| numerator / denominator)
        }
        Expr::Neg(inner) => Some(-backend_numeric_constant_value(ctx, *inner, depth + 1)?),
        _ => None,
    }
}

fn positive_rational_radius_expr(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let value = numeric_value(ctx, expr)?;
    if !value.is_positive() {
        return None;
    }
    if let Some(exact_radius) = exact_positive_rational_sqrt_expr(ctx, expr) {
        Some(exact_radius)
    } else {
        Some(ctx.call_builtin(BuiltinFn::Sqrt, vec![expr]))
    }
}

pub(super) fn positive_radius_expr(
    ctx: &mut Context,
    expr: ExprId,
    required_condition: &Option<ConditionPredicate>,
) -> Option<ExprId> {
    if let Some(radius) = positive_rational_radius_expr(ctx, expr) {
        return Some(radius);
    }
    if let Some(radius) = squared_radius_expr(ctx, expr, required_condition) {
        return Some(radius);
    }
    matches!(required_condition, Some(ConditionPredicate::Positive(condition_expr)) if *condition_expr == expr || SemanticEqualityChecker::new(ctx).are_equal(*condition_expr, expr))
        .then(|| ctx.call_builtin(BuiltinFn::Sqrt, vec![expr]))
}

fn squared_radius_expr(
    ctx: &Context,
    expr: ExprId,
    required_condition: &Option<ConditionPredicate>,
) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Pow(base, exponent) if is_two(ctx, *exponent) => {
            if numeric_value(ctx, *base)
                .map(|value| !value.is_zero())
                .unwrap_or(false)
            {
                return Some(*base);
            }
            matches!(required_condition, Some(ConditionPredicate::NonZero(condition_expr)) if *condition_expr == *base || SemanticEqualityChecker::new(ctx).are_equal(*condition_expr, *base))
                .then_some(*base)
        }
        _ => None,
    }
}

pub(super) fn positive_radius_square_value(
    ctx: &mut Context,
    expr: ExprId,
    variable: &str,
    required_conditions: &[ConditionPredicate],
) -> Option<BackendRadiusSquareValue> {
    if let Some(radius_value) = numeric_value(ctx, expr) {
        return radius_value
            .is_positive()
            .then_some(BackendRadiusSquareValue::Numeric(
                radius_value.clone() * radius_value,
            ));
    }

    if let Some(radicand) = positive_numeric_sqrt_radicand(ctx, expr) {
        return Some(BackendRadiusSquareValue::Numeric(radicand));
    }

    if let Some(radicand_expr) = crate::root_forms::extract_square_root_base(ctx, expr) {
        if required_positive_condition_matches(ctx, radicand_expr, required_conditions) {
            return Some(BackendRadiusSquareValue::ConditionalSymbolic(radicand_expr));
        }
    }

    if required_nonzero_condition_matches(ctx, expr, required_conditions)
        && !contains_named_var(ctx, expr, variable)
    {
        let two = ctx.add(Expr::Number(BigRational::from_integer(2.into())));
        return Some(BackendRadiusSquareValue::ConditionalSymbolic(
            ctx.add(Expr::Pow(expr, two)),
        ));
    }

    None
}

fn positive_numeric_sqrt_radicand(ctx: &Context, expr: ExprId) -> Option<BigRational> {
    let radicand_expr = crate::root_forms::extract_square_root_base(ctx, expr)?;
    let radicand = numeric_value(ctx, radicand_expr)?;
    radicand.is_positive().then_some(radicand)
}

fn required_positive_condition_matches(
    ctx: &Context,
    expr: ExprId,
    required_conditions: &[ConditionPredicate],
) -> bool {
    required_conditions.iter().any(|condition| {
        let ConditionPredicate::Positive(condition_expr) = condition else {
            return false;
        };
        *condition_expr == expr
            || SemanticEqualityChecker::new(ctx).are_equal(*condition_expr, expr)
    })
}

fn required_nonzero_condition_matches(
    ctx: &Context,
    expr: ExprId,
    required_conditions: &[ConditionPredicate],
) -> bool {
    required_conditions.iter().any(|condition| {
        let ConditionPredicate::NonZero(condition_expr) = condition else {
            return false;
        };
        *condition_expr == expr
            || SemanticEqualityChecker::new(ctx).are_equal(*condition_expr, expr)
    })
}

fn exact_positive_rational_sqrt_expr(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let value = numeric_value(ctx, expr)?;
    if !value.is_positive() {
        return None;
    }
    let sqrt_num = value.numer().sqrt();
    let sqrt_den = value.denom().sqrt();
    if &sqrt_num * &sqrt_num == value.numer().clone()
        && &sqrt_den * &sqrt_den == value.denom().clone()
    {
        Some(ctx.add(Expr::Number(BigRational::new(sqrt_num, sqrt_den))))
    } else {
        None
    }
}

pub(super) fn is_one(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Number(value) if value.is_one())
}

pub(super) fn is_zero(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Number(value) if value.is_zero())
}

pub(super) fn is_two(ctx: &Context, expr: ExprId) -> bool {
    matches!(
        ctx.get(expr),
        Expr::Number(value) if *value == BigRational::from_integer(2.into())
    )
}

pub fn try_algorithmic_integration_backend(
    ctx: &mut Context,
    integrand: ExprId,
    variable: &str,
    config: AlgorithmicIntegrationBackendConfig,
) -> AlgorithmicIntegrationCandidate {
    if !config.mode.attempts_backend() {
        return AlgorithmicIntegrationCandidate::disabled(integrand, variable);
    }

    let mut probe_runner = AlgorithmicIntegrationProbeRunner::new(config.budget);
    if let Some(candidate) = probe_runner
        .try_method_probe(AlgorithmicIntegrationMethod::Rational, |probe_runner| {
            try_rational_reciprocal_affine_probe(ctx, integrand, variable, probe_runner)
        })
    {
        let mut candidate = candidate;
        candidate.record_probe_usage(&probe_runner);
        return candidate;
    }
    if let Some(candidate) =
        probe_runner.try_method_probe(AlgorithmicIntegrationMethod::Hermite, |probe_runner| {
            try_hermite_positive_quadratic_log_derivative_probe(
                ctx,
                integrand,
                variable,
                probe_runner,
            )
        })
    {
        let mut candidate = candidate;
        candidate.record_probe_usage(&probe_runner);
        return candidate;
    }
    if let Some(candidate) = probe_runner.try_method_probe(
        AlgorithmicIntegrationMethod::HeurischProbe,
        |probe_runner| {
            try_heurisch_sine_log_derivative_probe(ctx, integrand, variable, probe_runner)
        },
    ) {
        let mut candidate = candidate;
        candidate.record_probe_usage(&probe_runner);
        return candidate;
    }

    let mut candidate = if probe_runner.method_budget_exhausted() {
        AlgorithmicIntegrationCandidate::budget_exceeded(integrand, variable)
    } else {
        AlgorithmicIntegrationCandidate::unsupported(integrand, variable)
    };
    candidate.record_probe_usage(&probe_runner);
    candidate
}
