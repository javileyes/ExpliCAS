use crate::isolation_utils::contains_var;
use crate::quadratic_formula::{discriminant, roots_from_a_b_delta};
use cas_ast::{Context, Equation, Expr, ExprId, RelOp, SolutionSet};
use num_bigint::BigInt;
use num_integer::Integer;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

/// Extract polynomial coefficients from an expression in `var`.
///
/// Returns `[a0, a1, ..., an]` where
/// `expr = a0 + a1*var + ... + an*var^n`.
///
/// Returns `None` if the expression is not polynomial in `var`
/// or its degree exceeds `max_degree`.
pub fn extract_poly_coefficients(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
    max_degree: usize,
) -> Option<Vec<ExprId>> {
    let mut terms: Vec<(ExprId, usize)> = Vec::new();
    let mut stack: Vec<(ExprId, bool)> = vec![(expr, true)];

    while let Some((curr, positive)) = stack.pop() {
        let curr_data = ctx.get(curr).clone();
        match curr_data {
            Expr::Add(l, r) => {
                stack.push((r, positive));
                stack.push((l, positive));
            }
            Expr::Sub(l, r) => {
                stack.push((r, !positive));
                stack.push((l, positive));
            }
            _ => {
                let (coeff, degree) = analyze_term_for_poly(ctx, curr, var)?;
                let term_coeff = if positive {
                    coeff
                } else {
                    ctx.add(Expr::Neg(coeff))
                };
                terms.push((term_coeff, degree as usize));
            }
        }
    }

    if terms.is_empty() {
        return None;
    }

    let max_seen_degree = terms.iter().map(|(_, d)| *d).max().unwrap_or(0);
    if max_seen_degree > max_degree {
        return None;
    }

    let zero = ctx.num(0);
    let mut coeffs = vec![zero; max_seen_degree + 1];
    for (coeff, degree) in terms {
        coeffs[degree] = ctx.add(Expr::Add(coeffs[degree], coeff));
    }

    Some(coeffs)
}

fn analyze_term_for_poly(ctx: &mut Context, term: ExprId, var: &str) -> Option<(ExprId, i32)> {
    if !contains_var(ctx, term, var) {
        return Some((term, 0));
    }

    let term_data = ctx.get(term).clone();
    match term_data {
        Expr::Variable(sym_id) if ctx.sym_name(sym_id) == var => Some((ctx.num(1), 1)),
        Expr::Pow(base, exp) => {
            if let Expr::Variable(sym_id) = ctx.get(base) {
                if ctx.sym_name(*sym_id) == var && !contains_var(ctx, exp, var) {
                    if let Expr::Number(n) = ctx.get(exp) {
                        if n.is_integer() && n.is_positive() {
                            let d: i32 = n.to_integer().try_into().ok()?;
                            return Some((ctx.num(1), d));
                        }
                    }
                }
            }
            None
        }
        Expr::Mul(l, r) => {
            let l_has = contains_var(ctx, l, var);
            let r_has = contains_var(ctx, r, var);

            if l_has && r_has {
                let (c1, d1) = analyze_term_for_poly(ctx, l, var)?;
                let (c2, d2) = analyze_term_for_poly(ctx, r, var)?;
                let new_coeff = cas_math::build::mul2_raw(ctx, c1, c2);
                Some((new_coeff, d1 + d2))
            } else if l_has {
                let (c, d) = analyze_term_for_poly(ctx, l, var)?;
                let new_coeff = cas_math::build::mul2_raw(ctx, c, r);
                Some((new_coeff, d))
            } else {
                let (c, d) = analyze_term_for_poly(ctx, r, var)?;
                let new_coeff = cas_math::build::mul2_raw(ctx, l, c);
                Some((new_coeff, d))
            }
        }
        Expr::Div(l, r) => {
            if contains_var(ctx, r, var) {
                return None;
            }
            let (c, d) = analyze_term_for_poly(ctx, l, var)?;
            let new_coeff = ctx.add(Expr::Div(c, r));
            Some((new_coeff, d))
        }
        Expr::Neg(inner) => {
            let (c, d) = analyze_term_for_poly(ctx, inner, var)?;
            let new_coeff = ctx.add(Expr::Neg(c));
            Some((new_coeff, d))
        }
        _ => None,
    }
}

/// Try to extract a rational number from an expression.
pub fn get_rational(ctx: &Context, expr: ExprId) -> Option<BigRational> {
    match ctx.get(expr) {
        Expr::Number(n) => Some(n.clone()),
        Expr::Neg(inner) => get_rational(ctx, *inner).map(|n| -n),
        Expr::Div(l, r) => {
            let ln = get_rational(ctx, *l)?;
            let rn = get_rational(ctx, *r)?;
            if rn.is_zero() {
                None
            } else {
                Some(ln / rn)
            }
        }
        Expr::Add(l, r) => {
            let ln = get_rational(ctx, *l)?;
            let rn = get_rational(ctx, *r)?;
            Some(ln + rn)
        }
        Expr::Sub(l, r) => {
            let ln = get_rational(ctx, *l)?;
            let rn = get_rational(ctx, *r)?;
            Some(ln - rn)
        }
        Expr::Mul(l, r) => {
            let ln = get_rational(ctx, *l)?;
            let rn = get_rational(ctx, *r)?;
            Some(ln * rn)
        }
        _ => None,
    }
}

/// Convert a rational to expression.
pub fn rational_to_expr(ctx: &mut Context, r: &BigRational) -> ExprId {
    ctx.add(Expr::Number(r.clone()))
}

/// Normalize rational coefficients to integers by multiplying by LCM of denominators.
pub fn normalize_to_integers(coeffs: &[BigRational]) -> Vec<BigInt> {
    let mut lcm = BigInt::one();
    for c in coeffs {
        if !c.is_zero() {
            let d = c.denom().clone();
            lcm = lcm.clone() / lcm.clone().gcd(&d) * d;
        }
    }

    coeffs
        .iter()
        .map(|c| (c * BigRational::from_integer(lcm.clone())).to_integer())
        .collect()
}

/// Generate candidate rational roots using Rational Root Theorem.
pub fn rational_root_candidates(int_coeffs: &[BigInt], max_candidates: usize) -> Vec<BigRational> {
    let a0 = &int_coeffs[0];
    let an = int_coeffs
        .last()
        .expect("coeff vector for polynomial must not be empty");

    if an.is_zero() {
        return vec![];
    }
    if a0.is_zero() {
        return vec![BigRational::zero()];
    }

    let divisors_a0 = small_divisors(&a0.abs());
    let divisors_an = small_divisors(&an.abs());
    if divisors_a0.is_empty() || divisors_an.is_empty() {
        return vec![];
    }

    let candidate_count = divisors_a0.len() * divisors_an.len() * 2;
    if candidate_count > max_candidates {
        return vec![];
    }

    let mut candidates = Vec::with_capacity(candidate_count);
    let mut seen = std::collections::HashSet::new();

    for p in &divisors_a0 {
        for q in &divisors_an {
            let candidate = BigRational::new(p.clone(), q.clone());
            let key = (candidate.numer().clone(), candidate.denom().clone());
            if seen.insert(key.clone()) {
                candidates.push(candidate.clone());
                let neg_key = (-key.0.clone(), key.1);
                if seen.insert(neg_key) {
                    candidates.push(-candidate);
                }
            }
        }
    }

    candidates
}

fn small_divisors(n: &BigInt) -> Vec<BigInt> {
    if n.is_zero() {
        return vec![];
    }

    let n_u64: u64 = match n.abs().try_into() {
        Ok(v) => v,
        Err(_) => return vec![],
    };
    if n_u64 == 0 {
        return vec![];
    }

    let mut divs = Vec::new();
    let sqrt_n = (n_u64 as f64).sqrt() as u64;
    for i in 1..=sqrt_n {
        if n_u64.is_multiple_of(i) {
            divs.push(BigInt::from(i));
            if i != n_u64 / i {
                divs.push(BigInt::from(n_u64 / i));
            }
        }
    }

    if divs.len() > 50 {
        return vec![];
    }

    divs
}

/// Evaluate polynomial at `x` using Horner's method.
pub fn horner_eval(coeffs: &[BigRational], x: &BigRational) -> BigRational {
    let mut result = BigRational::zero();
    for c in coeffs.iter().rev() {
        result = result * x.clone() + c.clone();
    }
    result
}

/// Divide polynomial by `(x - root)` using synthetic division.
///
/// Coeff order is `[a0, a1, ..., an]` (low-to-high degree).
pub fn synthetic_division(coeffs: &[BigRational], root: &BigRational) -> Vec<BigRational> {
    let n = coeffs.len();
    if n <= 1 {
        return vec![];
    }

    let mut quotient = vec![BigRational::zero(); n - 1];
    quotient[n - 2] = coeffs[n - 1].clone();

    for i in (0..n - 2).rev() {
        quotient[i] = coeffs[i + 1].clone() + root.clone() * quotient[i + 1].clone();
    }

    quotient
}

/// Extract rational roots by repeated candidate testing + synthetic deflation.
///
/// Returns `(roots, residual_coeffs)` where:
/// - `roots` are found rational roots (possibly with multiplicity),
/// - `residual_coeffs` is the remaining polynomial coefficient vector.
///
/// The routine stops deflation once residual degree is <= 2.
pub fn find_rational_roots(
    mut coeffs: Vec<BigRational>,
    max_candidates: usize,
) -> (Vec<BigRational>, Vec<BigRational>) {
    let mut roots = Vec::new();

    loop {
        if coeffs.len() <= 1 {
            break;
        }

        // a0 == 0 => root x=0; remove constant term and continue.
        while coeffs.len() > 1 && coeffs[0].is_zero() {
            coeffs.remove(0);
            roots.push(BigRational::zero());
        }

        if coeffs.len() <= 1 {
            break;
        }

        let degree = coeffs.len() - 1;
        if degree <= 2 {
            break;
        }

        let int_coeffs = normalize_to_integers(&coeffs);
        let candidates = rational_root_candidates(&int_coeffs, max_candidates);
        if candidates.is_empty() {
            break;
        }

        let mut found = false;
        for candidate in &candidates {
            if horner_eval(&coeffs, candidate).is_zero() {
                roots.push(candidate.clone());
                coeffs = synthetic_division(&coeffs, candidate);
                found = true;
                break;
            }
        }

        if !found {
            break;
        }
    }

    (roots, coeffs)
}

/// Outcome of solving a numeric-coefficient univariate polynomial.
pub enum NumericPolynomialSolveOutcome {
    /// All coefficients are zero, so equation `0 = 0` holds for every real value.
    AllReals,
    /// Candidate roots found for an in-range polynomial degree.
    CandidateRoots { degree: usize, roots: Vec<ExprId> },
}

/// Didactic step payload for Rational Root strategy activation.
#[derive(Debug, Clone, PartialEq)]
pub struct RationalRootsDidacticStep {
    pub description: String,
    pub equation_after: Equation,
}

/// One executable rational-roots item aligned with a didactic payload.
#[derive(Debug, Clone, PartialEq)]
pub struct RationalRootsExecutionItem {
    pub equation: Equation,
    pub description: String,
}

impl RationalRootsExecutionItem {
    /// User-facing narration for this execution item.
    pub fn description(&self) -> &str {
        &self.description
    }
}

/// Build narration for Rational Root Theorem strategy activation.
pub fn rational_roots_strategy_message(degree: usize) -> String {
    format!(
        "Applied Rational Root Theorem to degree-{} polynomial",
        degree
    )
}

/// Build the full didactic step payload for strategy activation.
pub fn build_rational_roots_strategy_step(
    equation_after: Equation,
    degree: usize,
) -> RationalRootsDidacticStep {
    RationalRootsDidacticStep {
        description: rational_roots_strategy_message(degree),
        equation_after,
    }
}

/// Build the full step payload for rational-roots strategy activation.
pub fn build_rational_roots_step(
    equation_after: Equation,
    degree: usize,
) -> RationalRootsDidacticStep {
    build_rational_roots_strategy_step(equation_after, degree)
}

/// Collect Rational Root strategy didactic steps in display order.
pub fn collect_rational_roots_didactic_steps(
    step: &RationalRootsDidacticStep,
) -> Vec<RationalRootsDidacticStep> {
    vec![step.clone()]
}

/// Collect Rational Root strategy execution items in display order.
pub fn collect_rational_roots_execution_items(
    step: &RationalRootsDidacticStep,
) -> Vec<RationalRootsExecutionItem> {
    collect_rational_roots_didactic_steps(step)
        .into_iter()
        .map(|didactic| RationalRootsExecutionItem {
            equation: didactic.equation_after.clone(),
            description: didactic.description,
        })
        .collect()
}

/// Return the first Rational Root strategy execution item, if present.
pub fn first_rational_roots_execution_item(
    step: &RationalRootsDidacticStep,
) -> Option<RationalRootsExecutionItem> {
    collect_rational_roots_execution_items(step)
        .into_iter()
        .next()
}

/// Solve rational-roots strategy step while optionally mapping the execution
/// item into caller-owned step payloads.
pub fn solve_rational_roots_step_pipeline_with_item<S, FStep>(
    step: RationalRootsDidacticStep,
    include_item: bool,
    mut map_item_to_step: FStep,
) -> Vec<S>
where
    FStep: FnMut(RationalRootsExecutionItem) -> S,
{
    if include_item {
        if let Some(item) = first_rational_roots_execution_item(&step) {
            return vec![map_item_to_step(item)];
        }
    }
    vec![]
}

/// Solved payload for rational-roots strategy execution.
#[derive(Debug, Clone, PartialEq)]
pub struct RationalRootsStrategySolved<TStep> {
    pub solution_set: SolutionSet,
    pub steps: Vec<TStep>,
}

/// Solve rational-roots strategy with closure hooks and optional didactic step mapping.
#[allow(clippy::too_many_arguments)]
pub fn solve_rational_roots_strategy_with_and_item<
    S,
    FBuildDiff,
    FSimplifyExpr,
    FExpandExpr,
    FExtractCoefficients,
    FSolveNumericPolynomial,
    FSortAndDedupRoots,
    FPlanStep,
    FStep,
>(
    lhs: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    min_degree: usize,
    max_degree: usize,
    max_candidates: usize,
    include_item: bool,
    mut build_diff: FBuildDiff,
    mut simplify_expr: FSimplifyExpr,
    mut expand_expr: FExpandExpr,
    mut extract_coefficients: FExtractCoefficients,
    mut solve_numeric_polynomial: FSolveNumericPolynomial,
    mut sort_and_dedup_roots: FSortAndDedupRoots,
    mut plan_step: FPlanStep,
    map_item_to_step: FStep,
) -> Option<RationalRootsStrategySolved<S>>
where
    FBuildDiff: FnMut(ExprId, ExprId) -> ExprId,
    FSimplifyExpr: FnMut(ExprId) -> ExprId,
    FExpandExpr: FnMut(ExprId) -> ExprId,
    FExtractCoefficients: FnMut(ExprId, &str, usize) -> Option<Vec<ExprId>>,
    FSolveNumericPolynomial:
        FnMut(&[ExprId], usize, usize, usize) -> Option<NumericPolynomialSolveOutcome>,
    FSortAndDedupRoots: FnMut(&mut Vec<ExprId>),
    FPlanStep: FnMut(ExprId, usize) -> RationalRootsDidacticStep,
    FStep: FnMut(RationalRootsExecutionItem) -> S,
{
    if op != RelOp::Eq {
        return None;
    }

    let diff = build_diff(lhs, rhs);
    let diff = simplify_expr(diff);
    let expanded = expand_expr(diff);
    let coeffs = extract_coefficients(expanded, var, max_degree)?;
    let outcome = solve_numeric_polynomial(&coeffs, min_degree, max_degree, max_candidates)?;

    let (degree, mut roots) = match outcome {
        NumericPolynomialSolveOutcome::AllReals => {
            return Some(RationalRootsStrategySolved {
                solution_set: SolutionSet::AllReals,
                steps: vec![],
            });
        }
        NumericPolynomialSolveOutcome::CandidateRoots { degree, roots } => (
            degree,
            roots
                .into_iter()
                .map(&mut simplify_expr)
                .collect::<Vec<_>>(),
        ),
    };

    if roots.is_empty() {
        return None;
    }

    sort_and_dedup_roots(&mut roots);
    let step = plan_step(expanded, degree);
    let steps = solve_rational_roots_step_pipeline_with_item(step, include_item, map_item_to_step);

    Some(RationalRootsStrategySolved {
        solution_set: SolutionSet::Discrete(roots),
        steps,
    })
}

/// Solve rational-roots strategy returning the plain `(SolutionSet, steps)` tuple
/// used by engine strategy surfaces.
#[allow(clippy::too_many_arguments)]
pub fn solve_rational_roots_strategy_result_with_and_item<
    S,
    FBuildDiff,
    FSimplifyExpr,
    FExpandExpr,
    FExtractCoefficients,
    FSolveNumericPolynomial,
    FSortAndDedupRoots,
    FPlanStep,
    FStep,
>(
    lhs: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    min_degree: usize,
    max_degree: usize,
    max_candidates: usize,
    include_item: bool,
    build_diff: FBuildDiff,
    simplify_expr: FSimplifyExpr,
    expand_expr: FExpandExpr,
    extract_coefficients: FExtractCoefficients,
    solve_numeric_polynomial: FSolveNumericPolynomial,
    sort_and_dedup_roots: FSortAndDedupRoots,
    plan_step: FPlanStep,
    map_item_to_step: FStep,
) -> Option<(SolutionSet, Vec<S>)>
where
    FBuildDiff: FnMut(ExprId, ExprId) -> ExprId,
    FSimplifyExpr: FnMut(ExprId) -> ExprId,
    FExpandExpr: FnMut(ExprId) -> ExprId,
    FExtractCoefficients: FnMut(ExprId, &str, usize) -> Option<Vec<ExprId>>,
    FSolveNumericPolynomial:
        FnMut(&[ExprId], usize, usize, usize) -> Option<NumericPolynomialSolveOutcome>,
    FSortAndDedupRoots: FnMut(&mut Vec<ExprId>),
    FPlanStep: FnMut(ExprId, usize) -> RationalRootsDidacticStep,
    FStep: FnMut(RationalRootsExecutionItem) -> S,
{
    let solved = solve_rational_roots_strategy_with_and_item(
        lhs,
        rhs,
        op,
        var,
        min_degree,
        max_degree,
        max_candidates,
        include_item,
        build_diff,
        simplify_expr,
        expand_expr,
        extract_coefficients,
        solve_numeric_polynomial,
        sort_and_dedup_roots,
        plan_step,
        map_item_to_step,
    )?;
    Some((solved.solution_set, solved.steps))
}

/// Solve rational-roots strategy for a concrete equation returning the plain
/// `(SolutionSet, steps)` tuple used by engine strategy surfaces.
#[allow(clippy::too_many_arguments)]
pub fn solve_rational_roots_strategy_result_for_equation_with_and_item<
    S,
    FBuildDiff,
    FSimplifyExpr,
    FExpandExpr,
    FExtractCoefficients,
    FSolveNumericPolynomial,
    FSortAndDedupRoots,
    FPlanStep,
    FStep,
>(
    equation: &Equation,
    var: &str,
    min_degree: usize,
    max_degree: usize,
    max_candidates: usize,
    include_item: bool,
    build_diff: FBuildDiff,
    simplify_expr: FSimplifyExpr,
    expand_expr: FExpandExpr,
    extract_coefficients: FExtractCoefficients,
    solve_numeric_polynomial: FSolveNumericPolynomial,
    sort_and_dedup_roots: FSortAndDedupRoots,
    plan_step: FPlanStep,
    map_item_to_step: FStep,
) -> Option<(SolutionSet, Vec<S>)>
where
    FBuildDiff: FnMut(ExprId, ExprId) -> ExprId,
    FSimplifyExpr: FnMut(ExprId) -> ExprId,
    FExpandExpr: FnMut(ExprId) -> ExprId,
    FExtractCoefficients: FnMut(ExprId, &str, usize) -> Option<Vec<ExprId>>,
    FSolveNumericPolynomial:
        FnMut(&[ExprId], usize, usize, usize) -> Option<NumericPolynomialSolveOutcome>,
    FSortAndDedupRoots: FnMut(&mut Vec<ExprId>),
    FPlanStep: FnMut(ExprId, usize) -> RationalRootsDidacticStep,
    FStep: FnMut(RationalRootsExecutionItem) -> S,
{
    solve_rational_roots_strategy_result_with_and_item(
        equation.lhs,
        equation.rhs,
        equation.op.clone(),
        var,
        min_degree,
        max_degree,
        max_candidates,
        include_item,
        build_diff,
        simplify_expr,
        expand_expr,
        extract_coefficients,
        solve_numeric_polynomial,
        sort_and_dedup_roots,
        plan_step,
        map_item_to_step,
    )
}

/// Plan Rational Root strategy didactic step for equation `expanded_expr = 0`.
pub fn plan_rational_roots_strategy_step(
    ctx: &mut Context,
    expanded_expr: ExprId,
    degree: usize,
) -> RationalRootsDidacticStep {
    build_rational_roots_strategy_step(
        Equation {
            lhs: expanded_expr,
            rhs: ctx.num(0),
            op: RelOp::Eq,
        },
        degree,
    )
}

/// Plan Rational Root strategy didactic step for equation `expanded_expr = 0`
/// with a precomputed zero expression id.
pub fn plan_rational_roots_strategy_step_with_zero_rhs(
    expanded_expr: ExprId,
    degree: usize,
    zero_rhs: ExprId,
) -> RationalRootsDidacticStep {
    build_rational_roots_strategy_step(
        Equation {
            lhs: expanded_expr,
            rhs: zero_rhs,
            op: RelOp::Eq,
        },
        degree,
    )
}

/// Plan rational-roots strategy step for equation `expanded_expr = 0`.
pub fn plan_rational_roots_step(
    ctx: &mut Context,
    expanded_expr: ExprId,
    degree: usize,
) -> RationalRootsDidacticStep {
    plan_rational_roots_strategy_step(ctx, expanded_expr, degree)
}

/// Solve a polynomial represented by coefficient expressions `[a0, a1, ..., an]`.
///
/// Returns `None` when:
/// - degree is outside `[min_degree, max_degree]`, or
/// - any coefficient is non-numeric (non-rational).
pub fn solve_numeric_coeff_polynomial(
    ctx: &mut Context,
    coeff_exprs: &[ExprId],
    min_degree: usize,
    max_degree: usize,
    max_candidates: usize,
) -> Option<NumericPolynomialSolveOutcome> {
    if coeff_exprs.is_empty() {
        return None;
    }

    let degree = coeff_exprs.len().saturating_sub(1);
    if !(min_degree..=max_degree).contains(&degree) {
        return None;
    }

    let rat_coeffs: Vec<BigRational> = coeff_exprs
        .iter()
        .map(|&c| get_rational(ctx, c))
        .collect::<Option<Vec<_>>>()?;

    if rat_coeffs.iter().all(|c| c.is_zero()) {
        return Some(NumericPolynomialSolveOutcome::AllReals);
    }

    let roots = extract_candidate_roots(ctx, rat_coeffs, max_candidates);
    Some(NumericPolynomialSolveOutcome::CandidateRoots { degree, roots })
}

/// Extract all discrete candidate roots for a degree>=3 polynomial with
/// rational coefficients.
///
/// This combines:
/// - rational-root deflation for high-degree components, and
/// - residual solving for the remaining degree<=2 polynomial.
pub fn extract_candidate_roots(
    ctx: &mut Context,
    coeffs: Vec<BigRational>,
    max_candidates: usize,
) -> Vec<ExprId> {
    let mut roots = Vec::new();
    let (rational_roots, residual_coeffs) = find_rational_roots(coeffs, max_candidates);

    for r in &rational_roots {
        roots.push(rational_to_expr(ctx, r));
    }

    if residual_coeffs.len() == 3 || residual_coeffs.len() == 2 {
        roots.extend(solve_residual_degree_leq_two(ctx, &residual_coeffs));
    }

    roots
}

/// Solve a residual polynomial of degree <= 2 with rational coefficients.
///
/// Coeff order is `[a0, a1, ..., an]` (low-to-high degree).
/// Returns zero, one, or two root expressions.
pub fn solve_residual_degree_leq_two(ctx: &mut Context, coeffs: &[BigRational]) -> Vec<ExprId> {
    if coeffs.len() == 3 {
        // Quadratic: a*x^2 + b*x + c = 0
        let a = &coeffs[2];
        let b = &coeffs[1];
        let c = &coeffs[0];

        if a.is_zero() {
            // Degenerated quadratic -> linear
            if b.is_zero() {
                return vec![];
            }
            let root = -c.clone() / b.clone();
            return vec![rational_to_expr(ctx, &root)];
        }

        let discriminant = discriminant(a, b, c);

        if discriminant.is_zero() {
            let root = -b.clone() / (BigRational::from_integer(2.into()) * a.clone());
            vec![rational_to_expr(ctx, &root)]
        } else if discriminant.is_positive() {
            let a_expr = rational_to_expr(ctx, a);
            let b_expr = rational_to_expr(ctx, b);
            let disc_expr = rational_to_expr(ctx, &discriminant);
            let (r1, r2) = roots_from_a_b_delta(ctx, a_expr, b_expr, disc_expr);
            vec![r1, r2]
        } else {
            vec![]
        }
    } else if coeffs.len() == 2 {
        // Linear: a*x + b = 0
        let a = &coeffs[1];
        let b = &coeffs[0];
        if a.is_zero() {
            vec![]
        } else {
            let root = -b.clone() / a.clone();
            vec![rational_to_expr(ctx, &root)]
        }
    } else {
        vec![]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solve_rational_roots_strategy_with_and_item_solves_cubic_and_emits_step() {
        let mut context = Context::new();
        let x = context.var("x");
        let two = context.num(2);
        let x2 = context.add(Expr::Pow(x, two));
        let x3 = context.add(Expr::Mul(x2, x));
        let lhs = context.add(Expr::Sub(x3, x)); // x^3 - x
        let zero = context.num(0);
        let context_cell = std::cell::RefCell::new(context);

        let solved = solve_rational_roots_strategy_with_and_item(
            lhs,
            zero,
            RelOp::Eq,
            "x",
            3,
            10,
            200,
            true,
            |left, right| {
                let mut context_ref = context_cell.borrow_mut();
                context_ref.add(Expr::Sub(left, right))
            },
            |expr| expr,
            |expr| expr,
            |expanded, var, max_degree| {
                let mut context_ref = context_cell.borrow_mut();
                extract_poly_coefficients(&mut context_ref, expanded, var, max_degree)
            },
            |coeffs, min_degree, max_degree, max_candidates| {
                let mut context_ref = context_cell.borrow_mut();
                solve_numeric_coeff_polynomial(
                    &mut context_ref,
                    coeffs,
                    min_degree,
                    max_degree,
                    max_candidates,
                )
            },
            |roots| {
                let context_ref = context_cell.borrow();
                crate::solution_set::sort_and_dedup_exprs(&context_ref, roots);
            },
            |expanded, degree| {
                let mut context_ref = context_cell.borrow_mut();
                plan_rational_roots_step(&mut context_ref, expanded, degree)
            },
            |item| item.description,
        )
        .expect("strategy should solve cubic");

        match solved.solution_set {
            SolutionSet::Discrete(roots) => assert_eq!(roots.len(), 3),
            other => panic!("expected discrete roots, got {:?}", other),
        }
        assert_eq!(solved.steps.len(), 1);
        assert!(solved.steps[0].contains("degree-3"));
    }

    #[test]
    fn solve_rational_roots_strategy_result_with_and_item_returns_plain_tuple() {
        let mut context = Context::new();
        let x = context.var("x");
        let two = context.num(2);
        let x2 = context.add(Expr::Pow(x, two));
        let x3 = context.add(Expr::Mul(x2, x));
        let lhs = context.add(Expr::Sub(x3, x)); // x^3 - x
        let zero = context.num(0);
        let context_cell = std::cell::RefCell::new(context);

        let solved = solve_rational_roots_strategy_result_with_and_item(
            lhs,
            zero,
            RelOp::Eq,
            "x",
            3,
            10,
            200,
            true,
            |left, right| {
                let mut context_ref = context_cell.borrow_mut();
                context_ref.add(Expr::Sub(left, right))
            },
            |expr| expr,
            |expr| expr,
            |expanded, var, max_degree| {
                let mut context_ref = context_cell.borrow_mut();
                extract_poly_coefficients(&mut context_ref, expanded, var, max_degree)
            },
            |coeffs, min_degree, max_degree, max_candidates| {
                let mut context_ref = context_cell.borrow_mut();
                solve_numeric_coeff_polynomial(
                    &mut context_ref,
                    coeffs,
                    min_degree,
                    max_degree,
                    max_candidates,
                )
            },
            |roots| {
                let context_ref = context_cell.borrow();
                crate::solution_set::sort_and_dedup_exprs(&context_ref, roots);
            },
            |expanded, degree| {
                let mut context_ref = context_cell.borrow_mut();
                plan_rational_roots_step(&mut context_ref, expanded, degree)
            },
            |item| item.description,
        )
        .expect("strategy should solve cubic");

        assert!(matches!(solved.0, SolutionSet::Discrete(_)));
        assert_eq!(solved.1.len(), 1);
    }

    #[test]
    fn solve_rational_roots_strategy_result_for_equation_with_and_item_returns_plain_tuple() {
        let mut context = Context::new();
        let x = context.var("x");
        let two = context.num(2);
        let x2 = context.add(Expr::Pow(x, two));
        let x3 = context.add(Expr::Mul(x2, x));
        let lhs = context.add(Expr::Sub(x3, x)); // x^3 - x
        let zero = context.num(0);
        let equation = Equation {
            lhs,
            rhs: zero,
            op: RelOp::Eq,
        };
        let context_cell = std::cell::RefCell::new(context);

        let solved = solve_rational_roots_strategy_result_for_equation_with_and_item(
            &equation,
            "x",
            3,
            10,
            200,
            true,
            |left, right| {
                let mut context_ref = context_cell.borrow_mut();
                context_ref.add(Expr::Sub(left, right))
            },
            |expr| expr,
            |expr| expr,
            |expanded, var, max_degree| {
                let mut context_ref = context_cell.borrow_mut();
                extract_poly_coefficients(&mut context_ref, expanded, var, max_degree)
            },
            |coeffs, min_degree, max_degree, max_candidates| {
                let mut context_ref = context_cell.borrow_mut();
                solve_numeric_coeff_polynomial(
                    &mut context_ref,
                    coeffs,
                    min_degree,
                    max_degree,
                    max_candidates,
                )
            },
            |roots| {
                let context_ref = context_cell.borrow();
                crate::solution_set::sort_and_dedup_exprs(&context_ref, roots);
            },
            |expanded, degree| {
                let mut context_ref = context_cell.borrow_mut();
                plan_rational_roots_step(&mut context_ref, expanded, degree)
            },
            |item| item.description,
        )
        .expect("strategy should solve cubic");

        assert!(matches!(solved.0, SolutionSet::Discrete(_)));
        assert_eq!(solved.1.len(), 1);
    }

    #[test]
    fn solve_rational_roots_strategy_result_for_equation_with_and_item_returns_none_for_inequality()
    {
        let mut context = Context::new();
        let x = context.var("x");
        let one = context.num(1);
        let equation = Equation {
            lhs: x,
            rhs: one,
            op: RelOp::Gt,
        };
        let mut extract_calls = 0usize;

        let out = solve_rational_roots_strategy_result_for_equation_with_and_item(
            &equation,
            "x",
            3,
            10,
            200,
            true,
            |left, right| {
                let mut ctx = Context::new();
                ctx.add(Expr::Sub(left, right))
            },
            |expr| expr,
            |expr| expr,
            |_expanded, _var, _max_degree| {
                extract_calls += 1;
                None
            },
            |_coeffs, _min_degree, _max_degree, _max_candidates| None,
            |_roots| {},
            |_expanded, _degree| panic!("didactic step must not be planned for inequalities"),
            |item| item.description,
        );

        assert!(out.is_none());
        assert_eq!(extract_calls, 0);
    }

    #[test]
    fn solve_rational_roots_strategy_with_and_item_rejects_non_equality() {
        let mut context = Context::new();
        let x = context.var("x");
        let zero = context.num(0);
        let context_cell = std::cell::RefCell::new(context);

        let solved = solve_rational_roots_strategy_with_and_item(
            x,
            zero,
            RelOp::Gt,
            "x",
            3,
            10,
            200,
            true,
            |left, right| {
                let mut context_ref = context_cell.borrow_mut();
                context_ref.add(Expr::Sub(left, right))
            },
            |expr| expr,
            |expr| expr,
            |_expanded, _var, _max_degree| {
                panic!("must short-circuit before coefficient extraction")
            },
            |_coeffs, _min_degree, _max_degree, _max_candidates| {
                panic!("must short-circuit before numeric solve")
            },
            |_roots| panic!("must short-circuit before root post-processing"),
            |_expanded, _degree| panic!("must short-circuit before didactic planning"),
            |item| item.description,
        );
        assert!(solved.is_none());
    }

    #[test]
    fn solve_rational_roots_strategy_with_and_item_returns_all_reals_for_zero_poly() {
        let mut context = Context::new();
        let x = context.var("x");
        let zero = context.num(0);
        let two = context.num(2);
        let x2 = context.add(Expr::Pow(x, two));
        let x3 = context.add(Expr::Mul(x2, x));
        let t3 = context.add(Expr::Mul(zero, x3));
        let t2 = context.add(Expr::Mul(zero, x2));
        let t1 = context.add(Expr::Mul(zero, x));
        let sum1 = context.add(Expr::Add(t3, t2));
        let sum2 = context.add(Expr::Add(t1, zero));
        let lhs = context.add(Expr::Add(sum1, sum2));
        let zero = context.num(0);
        let context_cell = std::cell::RefCell::new(context);

        let solved = solve_rational_roots_strategy_with_and_item(
            lhs,
            zero,
            RelOp::Eq,
            "x",
            3,
            10,
            200,
            true,
            |left, right| {
                let mut context_ref = context_cell.borrow_mut();
                context_ref.add(Expr::Sub(left, right))
            },
            |expr| expr,
            |expr| expr,
            |expanded, var, max_degree| {
                let mut context_ref = context_cell.borrow_mut();
                extract_poly_coefficients(&mut context_ref, expanded, var, max_degree)
            },
            |coeffs, min_degree, max_degree, max_candidates| {
                let mut context_ref = context_cell.borrow_mut();
                solve_numeric_coeff_polynomial(
                    &mut context_ref,
                    coeffs,
                    min_degree,
                    max_degree,
                    max_candidates,
                )
            },
            |roots| {
                let context_ref = context_cell.borrow();
                crate::solution_set::sort_and_dedup_exprs(&context_ref, roots);
            },
            |expanded, degree| {
                let mut context_ref = context_cell.borrow_mut();
                plan_rational_roots_step(&mut context_ref, expanded, degree)
            },
            |item| item.description,
        )
        .expect("strategy should match 0 = 0");

        assert!(matches!(solved.solution_set, SolutionSet::AllReals));
        assert!(solved.steps.is_empty());
    }

    #[test]
    fn horner_eval_works() {
        let coeffs = vec![
            BigRational::zero(),
            BigRational::from_integer((-1).into()),
            BigRational::zero(),
            BigRational::one(),
        ];
        assert!(horner_eval(&coeffs, &BigRational::zero()).is_zero());
        assert!(horner_eval(&coeffs, &BigRational::one()).is_zero());
        assert!(horner_eval(&coeffs, &BigRational::from_integer((-1).into())).is_zero());
    }

    #[test]
    fn synthetic_division_works() {
        let coeffs = vec![
            BigRational::zero(),
            BigRational::from_integer((-1).into()),
            BigRational::zero(),
            BigRational::one(),
        ];
        let quotient = synthetic_division(&coeffs, &BigRational::zero());
        assert_eq!(quotient.len(), 3);
        assert_eq!(quotient[0], BigRational::from_integer((-1).into()));
        assert!(quotient[1].is_zero());
        assert_eq!(quotient[2], BigRational::one());
    }

    #[test]
    fn small_divisors_for_12() {
        let divs = small_divisors(&BigInt::from(12));
        let mut vals: Vec<u64> = divs.iter().map(|d| d.try_into().unwrap_or(0)).collect();
        vals.sort();
        assert_eq!(vals, vec![1, 2, 3, 4, 6, 12]);
    }

    #[test]
    fn solve_residual_linear_works() {
        let mut ctx = Context::new();
        // 2x + 4 = 0 -> x = -2
        let coeffs = vec![
            BigRational::from_integer(4.into()),
            BigRational::from_integer(2.into()),
        ];
        let roots = solve_residual_degree_leq_two(&mut ctx, &coeffs);
        assert_eq!(roots.len(), 1);
        assert_eq!(
            get_rational(&ctx, roots[0]).expect("root should be numeric"),
            BigRational::from_integer((-2).into())
        );
    }

    #[test]
    fn solve_residual_quadratic_positive_discriminant_returns_two() {
        let mut ctx = Context::new();
        // x^2 - 1 = 0
        let coeffs = vec![
            BigRational::from_integer((-1).into()),
            BigRational::zero(),
            BigRational::one(),
        ];
        let roots = solve_residual_degree_leq_two(&mut ctx, &coeffs);
        assert_eq!(roots.len(), 2);
    }

    #[test]
    fn solve_residual_quadratic_negative_discriminant_returns_none() {
        let mut ctx = Context::new();
        // x^2 + 1 = 0 (no real roots)
        let coeffs = vec![BigRational::one(), BigRational::zero(), BigRational::one()];
        let roots = solve_residual_degree_leq_two(&mut ctx, &coeffs);
        assert!(roots.is_empty());
    }

    #[test]
    fn find_rational_roots_stops_at_quadratic_residual() {
        // x^3 - x = x*(x^2 - 1)
        let coeffs = vec![
            BigRational::zero(),
            BigRational::from_integer((-1).into()),
            BigRational::zero(),
            BigRational::one(),
        ];
        let (roots, residual) = find_rational_roots(coeffs, 200);
        assert_eq!(roots, vec![BigRational::zero()]);
        assert_eq!(residual.len(), 3); // quadratic residual
    }

    #[test]
    fn extract_candidate_roots_combines_deflation_and_residual() {
        // x^3 - x = x*(x^2-1), roots: -1, 0, 1
        let coeffs = vec![
            BigRational::zero(),
            BigRational::from_integer((-1).into()),
            BigRational::zero(),
            BigRational::one(),
        ];
        let mut ctx = Context::new();
        let roots = extract_candidate_roots(&mut ctx, coeffs, 200);
        assert_eq!(roots.len(), 3);
    }

    #[test]
    fn solve_numeric_coeff_polynomial_all_reals_when_all_zero() {
        let mut ctx = Context::new();
        let coeffs = vec![ctx.num(0), ctx.num(0), ctx.num(0), ctx.num(0)];
        let out = solve_numeric_coeff_polynomial(&mut ctx, &coeffs, 3, 10, 200)
            .expect("in-range degree with numeric coeffs should produce outcome");
        assert!(matches!(out, NumericPolynomialSolveOutcome::AllReals));
    }

    #[test]
    fn solve_numeric_coeff_polynomial_returns_candidates_for_cubic() {
        let mut ctx = Context::new();
        // x^3 - x
        let coeffs = vec![ctx.num(0), ctx.num(-1), ctx.num(0), ctx.num(1)];
        let out = solve_numeric_coeff_polynomial(&mut ctx, &coeffs, 3, 10, 200)
            .expect("in-range degree with numeric coeffs should produce outcome");
        match out {
            NumericPolynomialSolveOutcome::CandidateRoots { degree, roots } => {
                assert_eq!(degree, 3);
                assert_eq!(roots.len(), 3);
            }
            NumericPolynomialSolveOutcome::AllReals => {
                panic!("expected candidate roots for non-zero polynomial")
            }
        }
    }

    #[test]
    fn solve_numeric_coeff_polynomial_rejects_degree_out_of_range() {
        let mut ctx = Context::new();
        let coeffs = vec![ctx.num(1), ctx.num(2), ctx.num(3)];
        assert!(solve_numeric_coeff_polynomial(&mut ctx, &coeffs, 3, 10, 200).is_none());
    }

    #[test]
    fn solve_numeric_coeff_polynomial_rejects_non_numeric_coefficients() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let coeffs = vec![ctx.num(0), x, ctx.num(0), ctx.num(1)];
        assert!(solve_numeric_coeff_polynomial(&mut ctx, &coeffs, 3, 10, 200).is_none());
    }

    #[test]
    fn rational_roots_strategy_message_formats_expected_text() {
        assert_eq!(
            rational_roots_strategy_message(4),
            "Applied Rational Root Theorem to degree-4 polynomial"
        );
    }

    #[test]
    fn build_rational_roots_strategy_step_builds_payload() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let zero = ctx.num(0);
        let eq = Equation {
            lhs: x,
            rhs: zero,
            op: cas_ast::RelOp::Eq,
        };

        let step = build_rational_roots_strategy_step(eq.clone(), 3);
        assert_eq!(
            step.description,
            "Applied Rational Root Theorem to degree-3 polynomial"
        );
        assert_eq!(step.equation_after, eq);
    }

    #[test]
    fn build_rational_roots_step_alias_matches_strategy_builder() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let zero = ctx.num(0);
        let eq = Equation {
            lhs: x,
            rhs: zero,
            op: cas_ast::RelOp::Eq,
        };

        let step = build_rational_roots_step(eq.clone(), 4);
        assert_eq!(
            step.description,
            "Applied Rational Root Theorem to degree-4 polynomial"
        );
        assert_eq!(step.equation_after, eq);
    }

    #[test]
    fn plan_rational_roots_strategy_step_builds_zero_rhs_equation() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let step = plan_rational_roots_strategy_step(&mut ctx, x, 5);

        assert_eq!(
            step.description,
            "Applied Rational Root Theorem to degree-5 polynomial"
        );
        assert_eq!(step.equation_after.lhs, x);
        assert!(matches!(ctx.get(step.equation_after.rhs), Expr::Number(n) if n.is_zero()));
        assert_eq!(step.equation_after.op, cas_ast::RelOp::Eq);
    }

    #[test]
    fn plan_rational_roots_strategy_step_with_zero_rhs_uses_given_rhs() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let zero = ctx.num(0);
        let step = plan_rational_roots_strategy_step_with_zero_rhs(x, 5, zero);

        assert_eq!(
            step.description,
            "Applied Rational Root Theorem to degree-5 polynomial"
        );
        assert_eq!(step.equation_after.lhs, x);
        assert_eq!(step.equation_after.rhs, zero);
        assert_eq!(step.equation_after.op, cas_ast::RelOp::Eq);
    }

    #[test]
    fn plan_rational_roots_step_alias_builds_zero_rhs_equation() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let step = plan_rational_roots_step(&mut ctx, x, 6);

        assert_eq!(
            step.description,
            "Applied Rational Root Theorem to degree-6 polynomial"
        );
        assert_eq!(step.equation_after.lhs, x);
        assert!(matches!(ctx.get(step.equation_after.rhs), Expr::Number(n) if n.is_zero()));
        assert_eq!(step.equation_after.op, cas_ast::RelOp::Eq);
    }

    #[test]
    fn collect_rational_roots_didactic_steps_returns_single_step() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let step = plan_rational_roots_strategy_step(&mut ctx, x, 3);
        let didactic = collect_rational_roots_didactic_steps(&step);
        assert_eq!(didactic.len(), 1);
        assert_eq!(didactic[0], step);
    }

    #[test]
    fn collect_rational_roots_execution_items_returns_single_item() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let step = plan_rational_roots_strategy_step(&mut ctx, x, 3);
        let items = collect_rational_roots_execution_items(&step);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].equation, step.equation_after);
        assert_eq!(items[0].description, step.description);
    }

    #[test]
    fn first_rational_roots_execution_item_returns_single_item() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let step = plan_rational_roots_strategy_step(&mut ctx, x, 3);
        let item =
            first_rational_roots_execution_item(&step).expect("expected one rational-roots item");
        assert_eq!(item.equation, step.equation_after);
        assert_eq!(item.description, step.description);
    }

    #[test]
    fn solve_rational_roots_step_pipeline_with_item_maps_item_when_enabled() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let step = plan_rational_roots_strategy_step(&mut ctx, x, 4);

        let steps =
            solve_rational_roots_step_pipeline_with_item(step, true, |item| item.description);
        assert_eq!(steps.len(), 1);
        assert_eq!(
            steps[0],
            "Applied Rational Root Theorem to degree-4 polynomial"
        );
    }

    #[test]
    fn solve_rational_roots_step_pipeline_with_item_omits_item_when_disabled() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let step = plan_rational_roots_strategy_step(&mut ctx, x, 4);

        let steps = solve_rational_roots_step_pipeline_with_item(step, false, |_item| 1u8);
        assert!(steps.is_empty());
    }
}
