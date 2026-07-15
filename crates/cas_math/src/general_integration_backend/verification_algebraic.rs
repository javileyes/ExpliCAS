//! Algebraic zero-test for rational verification residuals (the Phase 4
//! workstream: graduate rational-candidate verification from case-by-case
//! structural normalization to a multipoly decision procedure).
//!
//! Decides `derivative == integrand` exactly for expressions that are
//! rational in the integration variable, free symbols, and square/cube roots
//! of variable-free radicands. Radicals are mapped to fresh atoms `t` and
//! the comparison is reduced by the quotient relation `t^d = radicand`
//! (`d = 2` for square roots, `d = 3` for cube roots — the G1 Cap. D
//! extension; an odd-degree relation needs NO sign condition, since the real
//! cube root exists for every real radicand), so
//! identities such as `1/(sqrt(a)*sqrt(a)*(1+u^2/a)) == 1/(a+u^2)` are
//! decided without bespoke normalization cases. NESTED radicals form a
//! relation TOWER: an outer radicand like `(5 − √5)/2` is rewritten in terms
//! of the inner radical's atom (`(5 − t₂)/2`), so its relation is a genuine
//! polynomial and the reduction descends the tower (`t₁² → (5 − t₂)/2`, then
//! `t₂² → 5`). Returns `None` when the shapes are out of scope or a budget is
//! exceeded, so callers keep today's conservative behavior.

use super::methods::*;

use crate::expr_predicates::contains_named_var;
use crate::multipoly::{multipoly_from_expr, MultiPoly, PolyBudget};
use crate::semantic_equality::SemanticEqualityChecker;
use crate::substitute::{substitute_power_aware, SubstituteOptions};
use cas_ast::{BuiltinFn, ConditionPredicate, Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::Signed;

// Sized for the doubly-even octic residual (G1 R3, `1/(x^8+1)`): its
// differentiate-back residual reaches ~370 monomials over 4 poly vars and the
// single-monomial reduction takes up to 8916 measured steps (x^8+5x^4+16, the
// widest family member; 16384 gives ~1.8x headroom). Raising both is sound:
// the node cap only gates INPUT size and the step cap only bounds work before
// an honest `None` — neither can turn a non-zero residual into `Some(true)`;
// termination is guaranteed by the tower's lexicographic descent, the cap is
// a safety valve.
const ALGEBRAIC_ZERO_TEST_MAX_NODES: usize = 1024;
const ALGEBRAIC_ZERO_TEST_MAX_VARS: usize = 6;
// 3 relations cover the G1 Cap. C tower `s = √5, t₁ = √((5−s)/2), t₂ = √((5+s)/2)`
// and the R3 octic tower `t = √(2S−P), u = √(2s−t), v = √(2s+t)`.
const ALGEBRAIC_ZERO_TEST_MAX_RELATIONS: usize = 3;
const ALGEBRAIC_ZERO_TEST_REDUCTION_STEPS: usize = 16384;

fn algebraic_zero_test_budget() -> PolyBudget {
    // Sized so a degree-≤8 rational integrand whose squarefree denominator splits into
    // linears + irreducible quadratics (e.g. `1/(x^6-1)`) reduces: the residual over the common
    // denominator reaches ~degree 12 in the variable plus the sqrt atom. Raising this is sound —
    // the zero test stays an exact decision procedure; a larger budget only handles bigger inputs.
    PolyBudget {
        max_terms: 2048,
        max_total_degree: 48,
        max_pow_exp: 24,
    }
}

/// Decide whether `derivative` and `integrand` are equal as rational
/// expressions (over the reals, away from poles, under the candidate's own
/// conditions for the square-root atoms). `Some(true)` is a proof of
/// equality; `Some(false)` is a proof of inequality and is only emitted in
/// the pure rational case (no square-root atoms); `None` means undecidable
/// by this procedure.
///
/// The quotient relation `t^2 = radicand` presupposes the radicand is
/// non-negative, so every radicand must either have an exactly decidable
/// non-negative sign (a numeric constant, or a constant surd like the nested
/// `(5 − √5)/2` proved by the surd-sign kernel) or be covered by an explicit
/// `Positive`/`NonNegative` required condition — no implicit domain
/// assumptions (this mirrors the structural verifier's contract pinned by
/// `verification_report_requires_positive_condition_for_symbolic_radius_square`).
pub(super) fn algebraic_rational_zero_test(
    ctx: &Context,
    derivative: ExprId,
    integrand: ExprId,
    variable: &str,
    required_conditions: &[ConditionPredicate],
) -> Option<bool> {
    let mut tmp = ctx.clone();
    let residual = tmp.add(Expr::Sub(derivative, integrand));
    if expr_node_count(&tmp, residual) > ALGEBRAIC_ZERO_TEST_MAX_NODES {
        return None;
    }

    let mut radicands = collect_variable_free_radicands(&tmp, residual, variable)?;
    if radicands.len() > ALGEBRAIC_ZERO_TEST_MAX_RELATIONS {
        return None;
    }
    for (radicand, degree) in &radicands {
        // `t² = radicand` presupposes a non-negative radicand; an ODD-degree
        // relation (`t³ = radicand`) is sound unconditionally — the real cube
        // root exists for every real radicand.
        if *degree == 2
            && !radicand_nonnegativity_is_represented(&mut tmp, *radicand, required_conditions)
        {
            return None;
        }
    }
    // Substitute OUTERMOST radicals first: replacing an inner `√5` first would
    // rewrite the tree UNDER an outer `√((5−√5)/2)` node, and the outer radical's
    // original form would no longer occur to be substituted. Strict containment
    // implies strictly more nodes, so a descending node count is a valid
    // topological order of the containment partial order (exact, no heuristics;
    // the order between unrelated radicands is irrelevant).
    radicands.sort_by_key(|(radicand, _)| std::cmp::Reverse(expr_node_count(&tmp, *radicand)));

    // Phase A: map each radical occurrence to a fresh atom variable.
    let mut expr = residual;
    let mut atoms: Vec<(String, ExprId, u32)> = Vec::new();
    for (index, (radicand, degree)) in radicands.iter().enumerate() {
        let name = fresh_atom_name(&tmp, residual, index);
        let atom = tmp.var(&name);
        expr = substitute_radical_atom(&mut tmp, expr, *radicand, atom, *degree);
        atoms.push((name, *radicand, *degree));
    }
    // Each relation's radicand gets every OTHER radical replaced by its atom
    // (outermost-first again), so a NESTED radicand like `(5 − √5)/2` becomes
    // the genuine polynomial `(5 − t₂)/2` and no sqrt node ever reaches the
    // multipoly layer. Containment is strict and acyclic, so the references
    // between atoms are triangular by construction.
    let mut relations: Vec<(String, ExprId, u32)> = Vec::new();
    for (index, (name, radicand, degree)) in atoms.iter().enumerate() {
        let mut substituted = *radicand;
        for (other_index, (other_name, other_radicand, other_degree)) in atoms.iter().enumerate() {
            if other_index == index {
                continue;
            }
            let atom = tmp.var(other_name);
            substituted = substitute_radical_atom(
                &mut tmp,
                substituted,
                *other_radicand,
                atom,
                *other_degree,
            );
        }
        relations.push((name.clone(), substituted, *degree));
    }

    // Phase B: build the residual as a single rational function over the
    // shared variable universe. The universe must also include every
    // variable of every radicand: a parameter that occurs only inside a
    // radicand is absent from the substituted expression, and align_vars
    // silently projects missing variables out, which would degenerate the
    // relation t^2 = radicand into t^2 = 1.
    let mut universe_set = crate::multipoly::collect_poly_vars(&tmp, expr);
    for (_, radicand, _) in &relations {
        universe_set.extend(crate::multipoly::collect_poly_vars(&tmp, *radicand));
    }
    let universe: Vec<String> = universe_set.into_iter().collect();
    if universe.len() > ALGEBRAIC_ZERO_TEST_MAX_VARS {
        return None;
    }
    let budget = algebraic_zero_test_budget();
    let (numerator, denominator) = expr_to_rational(&tmp, expr, &universe, &budget)?;

    // Phase C: reduce both sides by every relation t^2 = radicand, then
    // decide. A vanishing denominator means the construction was degenerate.
    let relation_polys = relation_polys(&tmp, &relations, &universe, &budget)?;
    let numerator = reduce_by_relations(numerator, &relation_polys, &budget)?;
    let denominator = reduce_by_relations(denominator, &relation_polys, &budget)?;
    if denominator.is_zero() {
        return None;
    }
    if numerator.is_zero() {
        return Some(true);
    }
    // A nonzero reduced numerator proves inequality only in the pure
    // rational case. With square-root atoms the quotient ring may be
    // reducible or the atoms algebraically dependent (sqrt(a^2) vs a,
    // sqrt(4*a) vs 2*sqrt(a)), so a nonzero form is not a refutation.
    if relation_polys.is_empty() {
        Some(false)
    } else {
        None
    }
}

fn expr_node_count(ctx: &Context, root: ExprId) -> usize {
    let mut stack = vec![root];
    let mut count = 0usize;
    while let Some(expr) = stack.pop() {
        count += 1;
        if count > ALGEBRAIC_ZERO_TEST_MAX_NODES {
            return count;
        }
        match ctx.get(expr) {
            Expr::Add(a, b)
            | Expr::Sub(a, b)
            | Expr::Mul(a, b)
            | Expr::Div(a, b)
            | Expr::Pow(a, b) => {
                stack.push(*a);
                stack.push(*b);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(*inner),
            Expr::Function(_, args) => stack.extend(args.iter().copied()),
            _ => {}
        }
    }
    count
}

/// Collect the distinct variable-free `(radicand, degree)` pairs of every
/// radical subexpression (`sqrt(r)`/`r^(1/2)` → degree 2, `cbrt(r)`/`r^(1/3)`
/// → degree 3). Returns `None` when a radicand depends on the integration
/// variable (out of scope for the quotient reduction) so the caller bails
/// instead of mis-translating.
fn collect_variable_free_radicands(
    ctx: &Context,
    root: ExprId,
    variable: &str,
) -> Option<Vec<(ExprId, u32)>> {
    let mut stack = vec![root];
    let mut radicands: Vec<(ExprId, u32)> = Vec::new();
    while let Some(expr) = stack.pop() {
        if let Some((radicand, degree)) = radical_like_radicand(ctx, expr) {
            if contains_named_var(ctx, radicand, variable) {
                return None;
            }
            if !radicands.iter().any(|(known, known_degree)| {
                *known_degree == degree && exprs_match_structurally(ctx, *known, radicand)
            }) {
                radicands.push((radicand, degree));
            }
            stack.push(radicand);
            continue;
        }
        match ctx.get(expr) {
            Expr::Add(a, b)
            | Expr::Sub(a, b)
            | Expr::Mul(a, b)
            | Expr::Div(a, b)
            | Expr::Pow(a, b) => {
                stack.push(*a);
                stack.push(*b);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(*inner),
            Expr::Function(_, args) => stack.extend(args.iter().copied()),
            _ => {}
        }
    }
    Some(radicands)
}

/// The `(radicand, degree)` of a radical node: `sqrt(r)`/`r^(1/2)` (degree 2)
/// or `cbrt(r)`/`r^(1/3)` (degree 3). `None` for anything else.
fn radical_like_radicand(ctx: &Context, expr: ExprId) -> Option<(ExprId, u32)> {
    match ctx.get(expr) {
        Expr::Function(fn_id, args) if args.len() == 1 => {
            if ctx.is_builtin(*fn_id, BuiltinFn::Sqrt) {
                Some((args[0], 2))
            } else if ctx.is_builtin(*fn_id, BuiltinFn::Cbrt) {
                Some((args[0], 3))
            } else {
                None
            }
        }
        Expr::Pow(base, exponent) => {
            let value = crate::numeric_eval::as_rational_const(ctx, *exponent)?;
            if value == BigRational::new(1.into(), 2.into()) {
                Some((*base, 2))
            } else if value == BigRational::new(1.into(), 3.into()) {
                Some((*base, 3))
            } else {
                None
            }
        }
        _ => None,
    }
}

fn exprs_match_structurally(ctx: &Context, left: ExprId, right: ExprId) -> bool {
    left == right || SemanticEqualityChecker::new(ctx).are_equal(left, right)
}

/// Replace both spellings of the degree-`degree` radical over `radicand`
/// (`radicand^(1/degree)` and the `sqrt`/`cbrt` call form) by `atom`
/// throughout `expr`.
fn substitute_radical_atom(
    tmp: &mut Context,
    expr: ExprId,
    radicand: ExprId,
    atom: ExprId,
    degree: u32,
) -> ExprId {
    let half = tmp.add(Expr::Number(BigRational::new(
        1.into(),
        i64::from(degree).into(),
    )));
    let pow_form = tmp.add(Expr::Pow(radicand, half));
    let expr = substitute_power_aware(
        tmp,
        expr,
        pow_form,
        atom,
        SubstituteOptions {
            power_aware: true,
            ..Default::default()
        },
    );
    let builtin = if degree == 3 {
        BuiltinFn::Cbrt
    } else {
        BuiltinFn::Sqrt
    };
    let call_form = tmp.call_builtin(builtin, vec![radicand]);
    substitute_power_aware(
        tmp,
        expr,
        call_form,
        atom,
        SubstituteOptions {
            power_aware: true,
            ..Default::default()
        },
    )
}

/// The `t^2 = radicand` relation is only sound when the radicand is known
/// non-negative; require that knowledge to be represented in the candidate's
/// conditions (or be a positive numeric constant).
fn radicand_nonnegativity_is_represented(
    ctx: &mut Context,
    radicand: ExprId,
    required_conditions: &[ConditionPredicate],
) -> bool {
    if let Some(value) = numeric_value(ctx, radicand) {
        return !value.is_negative();
    }
    // A variable-free surd radicand — e.g. the nested `(5 − √5)/2` — has an
    // exactly decidable sign: the surd-sign kernel is a PROOF over the reals,
    // never a float estimate (soundness gates must be exact).
    if let Some(sign) = crate::root_forms::provable_sign_vs_zero(ctx, radicand) {
        return sign != std::cmp::Ordering::Less;
    }
    required_conditions.iter().any(|condition| match condition {
        ConditionPredicate::Positive(expr) | ConditionPredicate::NonNegative(expr) => {
            crate::expr_domain::exprs_equivalent(ctx, *expr, radicand)
        }
        _ => false,
    })
}

fn fresh_atom_name(ctx: &Context, root: ExprId, index: usize) -> String {
    let existing = cas_ast::collect_variables(ctx, root);
    let mut suffix = index;
    loop {
        let candidate = format!("__alg{suffix}");
        if !existing.contains(&candidate) {
            return candidate;
        }
        suffix += ALGEBRAIC_ZERO_TEST_MAX_RELATIONS + 1;
    }
}

fn relation_polys(
    ctx: &Context,
    relations: &[(String, ExprId, u32)],
    universe: &[String],
    budget: &PolyBudget,
) -> Option<Vec<(usize, u32, MultiPoly)>> {
    let mut polys = Vec::with_capacity(relations.len());
    for (name, radicand, degree) in relations {
        let atom_index = universe.iter().position(|var| var == name)?;
        let radicand_poly = multipoly_from_expr(ctx, *radicand, budget)
            .ok()?
            .align_vars(universe);
        // TRIANGULAR tower: a radicand may reference OTHER atoms (only radicals
        // strictly contained in it, by construction of the substitution), but
        // never its own — rewriting `t²` then lowers the degree at that atom's
        // nesting height and can only raise degrees of strictly-inner atoms, a
        // lexicographic descent that terminates. Self-reference is structurally
        // impossible (a finite tree cannot contain its own sqrt); this check is
        // a soundness backstop, and the step cap in `reduce_by_relations`
        // bounds any pathological residue to an honest `None`.
        if radicand_poly.degree_in(atom_index) != 0 {
            return None;
        }
        polys.push((atom_index, *degree, radicand_poly));
    }
    Some(polys)
}

/// Build `expr` as a `(numerator, denominator)` pair of multivariate
/// polynomials over `universe`. Returns `None` on any non-rational node or
/// budget overflow.
fn expr_to_rational(
    ctx: &Context,
    expr: ExprId,
    universe: &[String],
    budget: &PolyBudget,
) -> Option<(MultiPoly, MultiPoly)> {
    match ctx.get(expr).clone() {
        Expr::Add(a, b) => {
            let (na, da) = expr_to_rational(ctx, a, universe, budget)?;
            let (nb, db) = expr_to_rational(ctx, b, universe, budget)?;
            let left = na.mul(&db, budget).ok()?;
            let right = nb.mul(&da, budget).ok()?;
            Some((left.add(&right).ok()?, da.mul(&db, budget).ok()?))
        }
        Expr::Sub(a, b) => {
            let (na, da) = expr_to_rational(ctx, a, universe, budget)?;
            let (nb, db) = expr_to_rational(ctx, b, universe, budget)?;
            let left = na.mul(&db, budget).ok()?;
            let right = nb.mul(&da, budget).ok()?;
            Some((left.sub(&right).ok()?, da.mul(&db, budget).ok()?))
        }
        Expr::Mul(a, b) => {
            let (na, da) = expr_to_rational(ctx, a, universe, budget)?;
            let (nb, db) = expr_to_rational(ctx, b, universe, budget)?;
            Some((na.mul(&nb, budget).ok()?, da.mul(&db, budget).ok()?))
        }
        Expr::Div(a, b) => {
            let (na, da) = expr_to_rational(ctx, a, universe, budget)?;
            let (nb, db) = expr_to_rational(ctx, b, universe, budget)?;
            if nb.is_zero() {
                return None;
            }
            Some((na.mul(&db, budget).ok()?, da.mul(&nb, budget).ok()?))
        }
        Expr::Neg(inner) => {
            let (n, d) = expr_to_rational(ctx, inner, universe, budget)?;
            Some((n.neg(), d))
        }
        Expr::Hold(inner) => expr_to_rational(ctx, inner, universe, budget),
        Expr::Pow(base, exponent) => {
            // Fold the exponent (derivatives produce shapes like x^(2 - 1)):
            // literal-only matching would silently bail on decidable inputs.
            let value = crate::numeric_eval::as_rational_const(ctx, exponent)?;
            if !value.is_integer() {
                return None;
            }
            let power = i64::try_from(value.to_integer()).ok()?;
            let (n, d) = expr_to_rational(ctx, base, universe, budget)?;
            let magnitude = u32::try_from(power.unsigned_abs()).ok()?;
            if magnitude > budget.max_total_degree {
                return None;
            }
            let n_pow = poly_pow(&n, magnitude, budget)?;
            let d_pow = poly_pow(&d, magnitude, budget)?;
            if power >= 0 {
                Some((n_pow, d_pow))
            } else if n_pow.is_zero() {
                None
            } else {
                Some((d_pow, n_pow))
            }
        }
        Expr::Number(_) | Expr::Variable(_) => {
            let poly = multipoly_from_expr(ctx, expr, budget)
                .ok()?
                .align_vars(universe);
            let one = MultiPoly::one(universe.to_vec());
            Some((poly, one))
        }
        _ => None,
    }
}

fn poly_pow(poly: &MultiPoly, exponent: u32, budget: &PolyBudget) -> Option<MultiPoly> {
    let mut result = MultiPoly::one(poly.vars.clone());
    let mut base = poly.clone();
    let mut remaining = exponent;
    while remaining > 0 {
        if remaining & 1 == 1 {
            result = result.mul(&base, budget).ok()?;
        }
        remaining >>= 1;
        if remaining > 0 {
            base = base.mul(&base, budget).ok()?;
        }
    }
    Some(result)
}

/// Reduce `poly` modulo every relation `t^d = radicand` (`d = 2` for square
/// roots, `d = 3` for cube roots): while any term carries `t^e` with `e >= d`,
/// replace `t^e` by `t^(e-d) * radicand`. Each step strictly lowers the atom
/// degree at that atom's nesting height, so the loop terminates; the step cap
/// guards budget blowups from large radicands.
fn reduce_by_relations(
    mut poly: MultiPoly,
    relations: &[(usize, u32, MultiPoly)],
    budget: &PolyBudget,
) -> Option<MultiPoly> {
    for _ in 0..ALGEBRAIC_ZERO_TEST_REDUCTION_STEPS {
        let Some((coeff, mono, atom_index, degree, radicand)) =
            relations.iter().find_map(|(atom_index, degree, radicand)| {
                poly.terms
                    .iter()
                    .find(|(_, mono)| mono[*atom_index] >= *degree)
                    .map(|(coeff, mono)| {
                        (coeff.clone(), mono.clone(), *atom_index, *degree, radicand)
                    })
            })
        else {
            return Some(poly);
        };

        let mut single_terms = std::collections::BTreeMap::new();
        single_terms.insert(mono.clone(), coeff.clone());
        let single = MultiPoly::from_map(poly.vars.clone(), single_terms);

        let mut reduced_mono = mono;
        reduced_mono[atom_index] -= degree;
        let replacement = radicand
            .mul_scalar(&coeff)
            .mul_monomial(&reduced_mono)
            .ok()?;

        poly = poly.sub(&single).ok()?.add(&replacement).ok()?;
        if poly.num_terms() > budget.max_terms {
            return None;
        }
    }
    None
}
