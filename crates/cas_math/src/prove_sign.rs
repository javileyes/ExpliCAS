//! Generic sign-proof helpers (`> 0` and `>= 0`) shared by runtime crates.
//!
//! Runtime crates provide a non-zero prover callback so this module stays
//! independent from runtime domain/predicate stacks.

use crate::expr_extract::{extract_abs_argument_view, extract_sqrt_argument_view};
use crate::expr_predicates::is_zero_expr as is_zero;
use crate::polynomial::Polynomial;
use crate::tri_proof::TriProof;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

fn univariate_quadratic_shape(ctx: &Context, expr: ExprId) -> Option<(BigRational, BigRational)> {
    let vars = cas_ast::collect_variables(ctx, expr);
    if vars.len() != 1 {
        return None;
    }

    let var = vars.iter().next()?;
    let poly = Polynomial::from_expr(ctx, expr, var).ok()?;
    if poly.degree() != 2 || poly.coeffs.len() < 3 {
        return None;
    }

    let a = poly.coeffs.get(2)?.clone();
    if a.is_zero() {
        return None;
    }

    let b = poly
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let c = poly
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let four = BigRational::from_integer(4.into());
    let discriminant = b.clone() * b - four * a.clone() * c;

    Some((a, discriminant))
}

fn is_strictly_positive_univariate_quadratic(ctx: &Context, expr: ExprId) -> bool {
    if univariate_quadratic_shape(ctx, expr)
        .is_some_and(|(a, discriminant)| a.is_positive() && discriminant.is_negative())
    {
        return true;
    }
    constant_coeff_quadratic_discriminant_bounds(ctx, expr)
        .is_some_and(|(a_lo, disc_hi)| a_lo.is_positive() && disc_hi.is_negative())
}

fn is_nonnegative_univariate_quadratic(ctx: &Context, expr: ExprId) -> bool {
    if univariate_quadratic_shape(ctx, expr)
        .is_some_and(|(a, discriminant)| a.is_positive() && !discriminant.is_positive())
    {
        return true;
    }
    constant_coeff_quadratic_discriminant_bounds(ctx, expr)
        .is_some_and(|(a_lo, disc_hi)| a_lo.is_positive() && !disc_hi.is_positive())
}

/// Exact interval bounds `(a_lo, disc_hi)` — a lower bound on the leading
/// coefficient and an upper bound on the discriminant `b² − 4ac` — for a
/// univariate quadratic whose coefficients are CONSTANT (variable-free)
/// expressions decidable by the interval oracle, e.g. the nested surd
/// `x² + √(2−√2)·x + 1` that `Polynomial::from_expr` cannot represent (its
/// coefficients are not rational). All arithmetic is outward-safe exact
/// BigRational interval arithmetic over `const_value_bounds`, so
/// `a_lo > 0 ∧ disc_hi < 0` is a PROOF of strict positivity for every real
/// value of the variable — never a numeric estimate.
fn constant_coeff_quadratic_discriminant_bounds(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BigRational, BigRational)> {
    let vars = cas_ast::collect_variables(ctx, expr);
    if vars.len() != 1 {
        return None;
    }
    let var = vars.iter().next()?.clone();

    // Interval accumulators for the degree-0/1/2 coefficients.
    let mut coeff_bounds: [(BigRational, BigRational); 3] = [
        (BigRational::zero(), BigRational::zero()),
        (BigRational::zero(), BigRational::zero()),
        (BigRational::zero(), BigRational::zero()),
    ];

    // Walk the additive spine; classify each term as coeff * var^degree with
    // a variable-free coefficient. Anything else is out of scope.
    let mut stack: Vec<(ExprId, bool)> = vec![(expr, false)];
    while let Some((term, negated)) = stack.pop() {
        match ctx.get(term) {
            Expr::Add(a, b) => {
                stack.push((*a, negated));
                stack.push((*b, negated));
                continue;
            }
            Expr::Sub(a, b) => {
                stack.push((*a, negated));
                stack.push((*b, !negated));
                continue;
            }
            Expr::Neg(inner) => {
                stack.push((*inner, !negated));
                continue;
            }
            _ => {}
        }
        let (degree, coeff) = classify_constant_coeff_monomial(ctx, term, &var)?;
        let (mut lo, mut hi) = coeff;
        if negated {
            std::mem::swap(&mut lo, &mut hi);
            lo = -lo;
            hi = -hi;
        }
        coeff_bounds[degree].0 += lo;
        coeff_bounds[degree].1 += hi;
    }

    let [(c_lo, c_hi), (b_lo, b_hi), (a_lo, a_hi)] = coeff_bounds;
    // b² upper/lower bounds from the b interval.
    let b_lo_sq = &b_lo * &b_lo;
    let b_hi_sq = &b_hi * &b_hi;
    let b_square_hi = b_lo_sq.clone().max(b_hi_sq.clone());
    // 4ac lower bound: minimum over the four interval products.
    let four = BigRational::from_integer(4.into());
    let ac_lo = [&a_lo * &c_lo, &a_lo * &c_hi, &a_hi * &c_lo, &a_hi * &c_hi]
        .into_iter()
        .min()?;
    let disc_hi = b_square_hi - four * ac_lo;
    Some((a_lo, disc_hi))
}

/// `(degree, (lo, hi))` of a monomial `coeff * var^degree` (degree ≤ 2) whose
/// coefficient is variable-free with decidable exact bounds.
fn classify_constant_coeff_monomial(
    ctx: &Context,
    term: ExprId,
    var: &str,
) -> Option<(usize, (BigRational, BigRational))> {
    let one = || (BigRational::one(), BigRational::one());
    match ctx.get(term) {
        Expr::Variable(name) if ctx.sym_name(*name) == var => Some((1, one())),
        Expr::Pow(base, exp) => {
            if !matches!(ctx.get(*base), Expr::Variable(n) if ctx.sym_name(*n) == var) {
                return constant_term_bounds(ctx, term).map(|b| (0, b));
            }
            match crate::numeric_eval::as_rational_const(ctx, *exp) {
                Some(e) if e == BigRational::from_integer(2.into()) => Some((2, one())),
                Some(e) if e.is_one() => Some((1, one())),
                _ => None,
            }
        }
        Expr::Mul(a, b) => {
            let (var_side, const_side) = if contains_named_var_local(ctx, *a, var) {
                (*a, *b)
            } else {
                (*b, *a)
            };
            let (degree, inner) = classify_constant_coeff_monomial(ctx, var_side, var)?;
            if degree == 0 {
                // Both sides variable-free: treat the whole product as a constant.
                return constant_term_bounds(ctx, term).map(|b| (0, b));
            }
            // Interval product: the var side may itself carry a coefficient
            // (`(2·x)·√c` has total coefficient `2·√c`).
            let outer = constant_term_bounds(ctx, const_side)?;
            let products = [
                &inner.0 * &outer.0,
                &inner.0 * &outer.1,
                &inner.1 * &outer.0,
                &inner.1 * &outer.1,
            ];
            let lo = products.iter().min()?.clone();
            let hi = products.iter().max()?.clone();
            Some((degree, (lo, hi)))
        }
        _ => constant_term_bounds(ctx, term).map(|b| (0, b)),
    }
}

fn constant_term_bounds(ctx: &Context, expr: ExprId) -> Option<(BigRational, BigRational)> {
    if !crate::expr_predicates::contains_variable(ctx, expr) {
        if let Expr::Number(n) = ctx.get(expr) {
            return Some((n.clone(), n.clone()));
        }
        return crate::const_sign::const_value_bounds(ctx, expr);
    }
    None
}

fn contains_named_var_local(ctx: &Context, expr: ExprId, var: &str) -> bool {
    crate::expr_predicates::contains_named_var(ctx, expr, var)
}

fn square_coeff(coeffs: &[BigRational], degree: usize) -> BigRational {
    let mut out = BigRational::zero();
    for i in 0..=degree {
        let Some(left) = coeffs.get(i) else {
            continue;
        };
        let Some(right) = coeffs.get(degree - i) else {
            continue;
        };
        out += left * right;
    }
    out
}

fn scaled_monic_square_root_and_constant_residual(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BigRational, Polynomial, BigRational)> {
    let vars = cas_ast::collect_variables(ctx, expr);
    if vars.len() != 1 {
        return None;
    }

    let var = vars.iter().next()?;
    let poly = Polynomial::from_expr(ctx, expr, var).ok()?;
    let degree = poly.degree();
    let scale = poly.leading_coeff();
    if degree < 2 || degree % 2 != 0 || !scale.is_positive() {
        return None;
    }

    let half = degree / 2;
    let mut root_coeffs = vec![BigRational::zero(); half + 1];
    root_coeffs[half] = BigRational::one();
    let two = BigRational::from_integer(2.into());

    for root_degree in (0..half).rev() {
        let target_degree = half + root_degree;
        let target_coeff = poly
            .coeffs
            .get(target_degree)
            .cloned()
            .unwrap_or_else(BigRational::zero)
            / scale.clone();
        let known = square_coeff(&root_coeffs, target_degree);
        root_coeffs[root_degree] = (target_coeff - known) / two.clone();
    }

    let root = Polynomial::new(root_coeffs, var.clone());
    let scaled_square = Polynomial::new(
        root.mul(&root)
            .coeffs
            .into_iter()
            .map(|coeff| coeff * scale.clone())
            .collect(),
        var.clone(),
    );
    let residual = poly.sub(&scaled_square);
    if residual.is_zero() {
        return Some((scale, root, BigRational::zero()));
    }
    if residual.degree() != 0 {
        return None;
    }

    residual
        .coeffs
        .first()
        .cloned()
        .map(|constant| (scale, root, constant))
}

fn positive_quadratic_minimum(poly: &Polynomial) -> Option<BigRational> {
    if poly.degree() != 2 || poly.coeffs.len() < 3 {
        return None;
    }

    let a = poly.coeffs.get(2)?.clone();
    if !a.is_positive() {
        return None;
    }

    let b = poly
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let c = poly
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let four = BigRational::from_integer(4.into());
    let minimum = c - (b.clone() * b) / (four * a);
    minimum.is_positive().then_some(minimum)
}

fn is_strictly_positive_monic_square_with_constant_offset(ctx: &Context, expr: ExprId) -> bool {
    let Some((scale, root, constant)) = scaled_monic_square_root_and_constant_residual(ctx, expr)
    else {
        return false;
    };

    if constant.is_positive() {
        return true;
    }

    if !constant.is_negative() {
        return false;
    }

    positive_quadratic_minimum(&root)
        .map(|minimum| scale * minimum.clone() * minimum + constant)
        .is_some_and(|lower_bound| lower_bound.is_positive())
}

fn scaled_bounded_sin_cos_term(ctx: &Context, expr: ExprId) -> Option<BigRational> {
    match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if args.len() == 1
                && (ctx.is_builtin(*fn_id, BuiltinFn::Sin)
                    || ctx.is_builtin(*fn_id, BuiltinFn::Cos)) =>
        {
            Some(BigRational::one())
        }
        Expr::Neg(inner) => scaled_bounded_sin_cos_term(ctx, *inner).map(|value| -value),
        Expr::Div(num, den) => {
            let num_scale = scaled_bounded_sin_cos_term(ctx, *num)?;
            let den_scale = crate::numeric_eval::as_rational_const(ctx, *den)?;
            if den_scale.is_zero() {
                None
            } else {
                Some(num_scale / den_scale)
            }
        }
        Expr::Mul(_, _) => {
            let mut numeric_scale = BigRational::one();
            let mut trig_scale = None;
            for factor in crate::expr_nary::mul_leaves(ctx, expr) {
                if let Some(value) = crate::numeric_eval::as_rational_const(ctx, factor) {
                    numeric_scale *= value;
                    continue;
                }

                let factor_scale = scaled_bounded_sin_cos_term(ctx, factor)?;
                if trig_scale.is_some() {
                    return None;
                }
                trig_scale = Some(factor_scale);
            }

            trig_scale.map(|factor_scale| numeric_scale * factor_scale)
        }
        _ => None,
    }
}

fn bounded_sin_cos_affine_margin(ctx: &Context, expr: ExprId) -> Option<BigRational> {
    let mut constant = BigRational::zero();
    let mut trig_bound = BigRational::zero();
    let mut has_bounded_trig = false;

    for (term, sign) in crate::expr_nary::add_terms_signed(ctx, expr) {
        let signed = |value: BigRational| match sign {
            crate::expr_nary::Sign::Pos => value,
            crate::expr_nary::Sign::Neg => -value,
        };

        if let Some(value) = crate::numeric_eval::as_rational_const(ctx, term) {
            constant += signed(value);
            continue;
        }

        let value = signed(scaled_bounded_sin_cos_term(ctx, term)?);
        trig_bound += value.abs();
        has_bounded_trig = true;
    }

    has_bounded_trig.then_some(constant - trig_bound)
}

fn is_strictly_positive_bounded_sin_cos_affine(ctx: &Context, expr: ExprId) -> bool {
    bounded_sin_cos_affine_margin(ctx, expr).is_some_and(|margin| margin.is_positive())
}

fn is_nonnegative_bounded_sin_cos_affine(ctx: &Context, expr: ExprId) -> bool {
    bounded_sin_cos_affine_margin(ctx, expr).is_some_and(|margin| !margin.is_negative())
}

/// Prove whether an expression is strictly positive (`> 0`).
///
/// `real_only = true` models a real-only value domain. `false` models a
/// complex-enabled domain where positivity is only provable in fewer cases.
pub fn prove_positive_depth_with<FProveNonzero>(
    ctx: &Context,
    expr: ExprId,
    depth: usize,
    real_only: bool,
    mut prove_nonzero: FProveNonzero,
) -> TriProof
where
    FProveNonzero: FnMut(&Context, ExprId, usize) -> TriProof,
{
    prove_positive_depth_inner(ctx, expr, depth, real_only, &mut prove_nonzero)
}

/// Prove whether an expression is non-negative (`>= 0`).
pub fn prove_nonnegative_depth_with<FProveNonzero>(
    ctx: &Context,
    expr: ExprId,
    depth: usize,
    real_only: bool,
    mut prove_nonzero: FProveNonzero,
) -> TriProof
where
    FProveNonzero: FnMut(&Context, ExprId, usize) -> TriProof,
{
    prove_nonnegative_depth_inner(ctx, expr, depth, real_only, &mut prove_nonzero)
}

fn prove_positive_depth_inner<FProveNonzero>(
    ctx: &Context,
    expr: ExprId,
    depth: usize,
    real_only: bool,
    prove_nonzero: &mut FProveNonzero,
) -> TriProof
where
    FProveNonzero: FnMut(&Context, ExprId, usize) -> TriProof,
{
    use num_traits::Zero;

    if depth == 0 {
        return TriProof::Unknown;
    }

    // The two exact constant oracles below only ever decide a VARIABLE-FREE expression (a linear
    // surd / rational / transcendental constant); on any variable-bearing subtree they walk part of
    // the tree only to bail to `None`. Since they run at EVERY recursion node, gate them behind one
    // cheap bail-at-first-variable check — behavior-identical (both return `None` for variable-bearing
    // exprs), but it replaces two allocating walks (incl. the 50-digit `const_value_bounds` interval
    // arithmetic) with a single boolean walk on the common case (measured ~449 ns/node of waste
    // removed per variable-bearing node; the `provable_const_sign` fallback was added in 1250a156e).
    if !crate::expr_predicates::contains_variable(ctx, expr) {
        // A constant LINEAR SURD `A + B·√n` (e.g. `2 − √3`, `1 − √2`): decide its sign EXACTLY. The
        // structural rules below cannot compare `2` against `√3`, leaving such offsets `Unknown` —
        // which makes a solver attach a spurious `2 − √3 > 0` domain condition (dropping a valid
        // branch).
        match crate::root_forms::provable_sign_vs_zero(ctx, expr) {
            Some(std::cmp::Ordering::Greater) => return TriProof::Proven,
            Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal) => {
                return TriProof::Disproven
            }
            None => {}
        }
        // General constant fallback (`1 − e^(1/3)`, `π − 4`, `ln(2)`): the exact value-bounds oracle
        // decides transcendental constants the linear-surd form misses, so a linear solve with such a
        // coefficient returns its solution DIRECTLY instead of a vacuous conditional
        // (`All reals if e^(1/3) = 0 and …`). `None`/straddling ⇒ fall through unchanged.
        match crate::const_sign::provable_const_sign(ctx, expr) {
            Some(crate::const_sign::ConstSign::Positive) => return TriProof::Proven,
            Some(crate::const_sign::ConstSign::Negative | crate::const_sign::ConstSign::Zero) => {
                return TriProof::Disproven
            }
            None => {}
        }
    }

    match ctx.get(expr) {
        Expr::Number(n) => {
            if *n > num_rational::BigRational::zero() {
                TriProof::Proven
            } else {
                TriProof::Disproven
            }
        }
        Expr::Constant(c) => {
            if matches!(c, cas_ast::Constant::Pi | cas_ast::Constant::E) {
                TriProof::Proven
            } else {
                TriProof::Unknown
            }
        }
        Expr::Function(fn_id, args)
            if ctx.is_builtin(*fn_id, BuiltinFn::Hold) && args.len() == 1 =>
        {
            prove_positive_depth_inner(ctx, args[0], depth - 1, real_only, prove_nonzero)
        }
        Expr::Add(_, _) | Expr::Sub(_, _)
            if real_only && is_strictly_positive_univariate_quadratic(ctx, expr) =>
        {
            TriProof::Proven
        }
        Expr::Add(_, _) | Expr::Sub(_, _)
            if real_only && is_strictly_positive_monic_square_with_constant_offset(ctx, expr) =>
        {
            TriProof::Proven
        }
        Expr::Add(_, _) | Expr::Sub(_, _)
            if real_only && is_strictly_positive_bounded_sin_cos_affine(ctx, expr) =>
        {
            TriProof::Proven
        }
        Expr::Add(a, b) => {
            let proof_a_pos =
                prove_positive_depth_inner(ctx, *a, depth - 1, real_only, prove_nonzero);
            let proof_b_pos =
                prove_positive_depth_inner(ctx, *b, depth - 1, real_only, prove_nonzero);

            if proof_a_pos == TriProof::Proven && proof_b_pos == TriProof::Proven {
                return TriProof::Proven;
            }

            if proof_a_pos == TriProof::Proven {
                let b_nonneg =
                    prove_nonnegative_depth_inner(ctx, *b, depth - 1, real_only, prove_nonzero);
                if b_nonneg == TriProof::Proven {
                    return TriProof::Proven;
                }
            }
            if proof_b_pos == TriProof::Proven {
                let a_nonneg =
                    prove_nonnegative_depth_inner(ctx, *a, depth - 1, real_only, prove_nonzero);
                if a_nonneg == TriProof::Proven {
                    return TriProof::Proven;
                }
            }

            TriProof::Unknown
        }
        Expr::Mul(a, b) => {
            if is_zero(ctx, *a) || is_zero(ctx, *b) {
                return TriProof::Disproven;
            }

            let proof_a = prove_positive_depth_inner(ctx, *a, depth - 1, real_only, prove_nonzero);
            let proof_b = prove_positive_depth_inner(ctx, *b, depth - 1, real_only, prove_nonzero);

            match (proof_a, proof_b) {
                (TriProof::Proven, TriProof::Proven) => TriProof::Proven,
                _ => TriProof::Unknown,
            }
        }
        Expr::Div(a, b) => {
            let proof_a = prove_positive_depth_inner(ctx, *a, depth - 1, real_only, prove_nonzero);
            let proof_b = prove_positive_depth_inner(ctx, *b, depth - 1, real_only, prove_nonzero);

            match (proof_a, proof_b) {
                (TriProof::Proven, TriProof::Proven) => TriProof::Proven,
                _ => TriProof::Unknown,
            }
        }
        Expr::Pow(base, exp) => {
            let base_positive =
                prove_positive_depth_inner(ctx, *base, depth - 1, real_only, prove_nonzero);

            if real_only {
                if base_positive == TriProof::Proven {
                    return TriProof::Proven;
                }
            } else {
                let exp_is_real_numeric = matches!(ctx.get(*exp), Expr::Number(_));
                if base_positive == TriProof::Proven && exp_is_real_numeric {
                    return TriProof::Proven;
                }
            }

            if let Expr::Number(n) = ctx.get(*exp) {
                if n.is_integer() {
                    let int_val = n.to_integer();
                    let two: num_bigint::BigInt = 2.into();
                    if &int_val % &two == 0.into() {
                        let base_nonzero = prove_nonzero(ctx, *base, depth - 1);
                        if base_nonzero == TriProof::Proven {
                            return TriProof::Proven;
                        }
                    }
                }
            }
            TriProof::Unknown
        }
        Expr::Function(_, _) if extract_abs_argument_view(ctx, expr).is_some() => {
            let Some(arg) = extract_abs_argument_view(ctx, expr) else {
                return TriProof::Unknown;
            };
            let inner_nonzero = prove_nonzero(ctx, arg, depth - 1);
            if inner_nonzero == TriProof::Proven {
                TriProof::Proven
            } else if inner_nonzero == TriProof::Disproven {
                TriProof::Disproven
            } else {
                TriProof::Unknown
            }
        }
        Expr::Function(fn_id, args)
            if real_only && ctx.is_builtin(*fn_id, BuiltinFn::Cosh) && args.len() == 1 =>
        {
            TriProof::Proven
        }
        Expr::Function(fn_id, args)
            if ctx.is_builtin(*fn_id, BuiltinFn::Exp) && args.len() == 1 =>
        {
            if real_only {
                TriProof::Proven
            } else {
                match ctx.get(args[0]) {
                    Expr::Number(_)
                    | Expr::Constant(cas_ast::Constant::Pi)
                    | Expr::Constant(cas_ast::Constant::E) => TriProof::Proven,
                    _ => TriProof::Unknown,
                }
            }
        }
        Expr::Function(fn_id, args) if ctx.is_builtin(*fn_id, BuiltinFn::Ln) && args.len() == 1 => {
            let Some(n) = crate::numeric_eval::as_rational_const(ctx, args[0]) else {
                return TriProof::Unknown;
            };
            let zero = num_rational::BigRational::from_integer(0.into());
            let one = num_rational::BigRational::from_integer(1.into());
            if n > one {
                TriProof::Proven
            } else if n > zero {
                TriProof::Disproven
            } else {
                TriProof::Unknown
            }
        }
        Expr::Function(_, _) if extract_sqrt_argument_view(ctx, expr).is_some() => {
            let Some(arg) = extract_sqrt_argument_view(ctx, expr) else {
                return TriProof::Unknown;
            };
            prove_positive_depth_inner(ctx, arg, depth - 1, real_only, prove_nonzero)
        }
        Expr::Neg(inner) => {
            let inner_proof =
                prove_positive_depth_inner(ctx, *inner, depth - 1, real_only, prove_nonzero);
            match inner_proof {
                TriProof::Proven => TriProof::Disproven,
                TriProof::Disproven => {
                    if let Expr::Number(n) = ctx.get(*inner) {
                        if n.is_negative() {
                            return TriProof::Proven;
                        }
                    }
                    TriProof::Unknown
                }
                _ => TriProof::Unknown,
            }
        }
        Expr::Hold(inner) => {
            prove_positive_depth_inner(ctx, *inner, depth - 1, real_only, prove_nonzero)
        }
        _ => TriProof::Unknown,
    }
}

fn prove_nonnegative_depth_inner<FProveNonzero>(
    ctx: &Context,
    expr: ExprId,
    depth: usize,
    real_only: bool,
    prove_nonzero: &mut FProveNonzero,
) -> TriProof
where
    FProveNonzero: FnMut(&Context, ExprId, usize) -> TriProof,
{
    use num_traits::Zero;

    if depth == 0 {
        return TriProof::Unknown;
    }

    // Gate both exact constant oracles behind one cheap variable-free check (see the positive prover
    // for the rationale): both return `None` on variable-bearing subtrees anyway, so this is
    // behavior-identical and removes the two per-node walks on the common case.
    if !crate::expr_predicates::contains_variable(ctx, expr) {
        // A constant LINEAR SURD `A + B·√n`: decide `≥ 0` exactly (`> 0` or `= 0` ⇒ Proven, `< 0` ⇒
        // Disproven).
        match crate::root_forms::provable_sign_vs_zero(ctx, expr) {
            Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal) => {
                return TriProof::Proven
            }
            Some(std::cmp::Ordering::Less) => return TriProof::Disproven,
            None => {}
        }
        // General constant fallback via exact value bounds (same rationale as the positive prover).
        match crate::const_sign::provable_const_sign(ctx, expr) {
            Some(crate::const_sign::ConstSign::Positive | crate::const_sign::ConstSign::Zero) => {
                return TriProof::Proven
            }
            Some(crate::const_sign::ConstSign::Negative) => return TriProof::Disproven,
            None => {}
        }
    }

    match ctx.get(expr) {
        Expr::Number(n) => {
            if *n >= num_rational::BigRational::zero() {
                TriProof::Proven
            } else {
                TriProof::Disproven
            }
        }
        Expr::Constant(c) => {
            if matches!(c, cas_ast::Constant::Pi | cas_ast::Constant::E) {
                TriProof::Proven
            } else {
                TriProof::Unknown
            }
        }
        Expr::Function(fn_id, args)
            if ctx.is_builtin(*fn_id, BuiltinFn::Hold) && args.len() == 1 =>
        {
            prove_nonnegative_depth_inner(ctx, args[0], depth - 1, real_only, prove_nonzero)
        }
        Expr::Add(_, _) | Expr::Sub(_, _)
            if real_only && is_nonnegative_univariate_quadratic(ctx, expr) =>
        {
            TriProof::Proven
        }
        Expr::Add(_, _) | Expr::Sub(_, _)
            if real_only && is_nonnegative_bounded_sin_cos_affine(ctx, expr) =>
        {
            TriProof::Proven
        }
        Expr::Pow(base, exp) => {
            if let Expr::Number(n) = ctx.get(*exp) {
                if n.is_integer() {
                    let int_val = n.to_integer();
                    let two: num_bigint::BigInt = 2.into();
                    if &int_val % &two == 0.into() && int_val > 0.into() {
                        return TriProof::Proven;
                    }
                }
            }

            if real_only {
                let base_positive =
                    prove_positive_depth_inner(ctx, *base, depth - 1, real_only, prove_nonzero);
                if base_positive == TriProof::Proven {
                    return TriProof::Proven;
                }
            }

            TriProof::Unknown
        }
        Expr::Function(_, _) if extract_abs_argument_view(ctx, expr).is_some() => TriProof::Proven,
        Expr::Function(_, _) if extract_sqrt_argument_view(ctx, expr).is_some() => {
            let Some(arg) = extract_sqrt_argument_view(ctx, expr) else {
                return TriProof::Unknown;
            };
            prove_nonnegative_depth_inner(ctx, arg, depth - 1, real_only, prove_nonzero)
        }
        Expr::Function(fn_id, args)
            if real_only && ctx.is_builtin(*fn_id, BuiltinFn::Cosh) && args.len() == 1 =>
        {
            TriProof::Proven
        }
        Expr::Function(fn_id, args)
            if ctx.is_builtin(*fn_id, BuiltinFn::Exp) && args.len() == 1 =>
        {
            if real_only {
                TriProof::Proven
            } else {
                match ctx.get(args[0]) {
                    Expr::Number(_)
                    | Expr::Constant(cas_ast::Constant::Pi)
                    | Expr::Constant(cas_ast::Constant::E) => TriProof::Proven,
                    _ => TriProof::Unknown,
                }
            }
        }
        Expr::Function(fn_id, args) if ctx.is_builtin(*fn_id, BuiltinFn::Ln) && args.len() == 1 => {
            let Some(n) = crate::numeric_eval::as_rational_const(ctx, args[0]) else {
                return TriProof::Unknown;
            };
            let zero = num_rational::BigRational::from_integer(0.into());
            let one = num_rational::BigRational::from_integer(1.into());
            if n >= one {
                TriProof::Proven
            } else if n > zero {
                TriProof::Disproven
            } else {
                TriProof::Unknown
            }
        }
        Expr::Mul(a, b) => {
            let proof_a =
                prove_nonnegative_depth_inner(ctx, *a, depth - 1, real_only, prove_nonzero);
            let proof_b =
                prove_nonnegative_depth_inner(ctx, *b, depth - 1, real_only, prove_nonzero);

            match (proof_a, proof_b) {
                (TriProof::Proven, TriProof::Proven) => TriProof::Proven,
                _ => TriProof::Unknown,
            }
        }
        Expr::Add(a, b) => {
            let proof_a =
                prove_nonnegative_depth_inner(ctx, *a, depth - 1, real_only, prove_nonzero);
            let proof_b =
                prove_nonnegative_depth_inner(ctx, *b, depth - 1, real_only, prove_nonzero);

            match (proof_a, proof_b) {
                (TriProof::Proven, TriProof::Proven) => TriProof::Proven,
                _ => TriProof::Unknown,
            }
        }
        Expr::Div(a, b) => {
            let proof_a =
                prove_nonnegative_depth_inner(ctx, *a, depth - 1, real_only, prove_nonzero);
            let proof_b = prove_positive_depth_inner(ctx, *b, depth - 1, real_only, prove_nonzero);

            match (proof_a, proof_b) {
                (TriProof::Proven, TriProof::Proven) => TriProof::Proven,
                _ => TriProof::Unknown,
            }
        }
        Expr::Neg(inner) => {
            let inner_proof =
                prove_nonnegative_depth_inner(ctx, *inner, depth - 1, real_only, prove_nonzero);
            match inner_proof {
                TriProof::Disproven => TriProof::Proven,
                _ => TriProof::Unknown,
            }
        }
        Expr::Hold(inner) => {
            prove_nonnegative_depth_inner(ctx, *inner, depth - 1, real_only, prove_nonzero)
        }
        _ => TriProof::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::{prove_nonnegative_depth_with, prove_positive_depth_with};
    use crate::tri_proof::TriProof;
    use cas_parser::parse;

    #[test]
    fn positive_proves_numeric_literal() {
        let mut ctx = cas_ast::Context::new();
        let expr = parse("2", &mut ctx).expect("parse");
        let out = prove_positive_depth_with(&ctx, expr, 20, true, |_ctx, _expr, _depth| {
            TriProof::Unknown
        });
        assert_eq!(out, TriProof::Proven);
    }

    #[test]
    fn nonnegative_proves_even_power() {
        let mut ctx = cas_ast::Context::new();
        let expr = parse("x^2", &mut ctx).expect("parse");
        let out = prove_nonnegative_depth_with(&ctx, expr, 20, true, |_ctx, _expr, _depth| {
            TriProof::Unknown
        });
        assert_eq!(out, TriProof::Proven);
    }

    #[test]
    fn internal_hold_is_transparent_to_sign_proofs() {
        let mut ctx = cas_ast::Context::new();
        let positive = parse("(2*x+1)^2 + 1", &mut ctx).expect("parse");
        let held_positive = cas_ast::hold::wrap_hold(&mut ctx, positive);
        let out =
            prove_positive_depth_with(&ctx, held_positive, 20, true, |_ctx, _expr, _depth| {
                TriProof::Unknown
            });
        assert_eq!(out, TriProof::Proven);

        let nonnegative = parse("(x+1)^2", &mut ctx).expect("parse");
        let held_nonnegative = cas_ast::hold::wrap_hold(&mut ctx, nonnegative);
        let out = prove_nonnegative_depth_with(
            &ctx,
            held_nonnegative,
            20,
            true,
            |_ctx, _expr, _depth| TriProof::Unknown,
        );
        assert_eq!(out, TriProof::Proven);
    }

    #[test]
    fn positive_proves_positive_definite_quadratic() {
        let mut ctx = cas_ast::Context::new();
        let expr = parse("2*x^2 + 2*x + 1", &mut ctx).expect("parse");
        let out = prove_positive_depth_with(&ctx, expr, 20, true, |_ctx, _expr, _depth| {
            TriProof::Unknown
        });
        assert_eq!(out, TriProof::Proven);
    }

    #[test]
    fn positive_does_not_prove_perfect_square_quadratic() {
        let mut ctx = cas_ast::Context::new();
        let expr = parse("x^2 + 2*x + 1", &mut ctx).expect("parse");
        let out = prove_positive_depth_with(&ctx, expr, 20, true, |_ctx, _expr, _depth| {
            TriProof::Unknown
        });
        assert_eq!(out, TriProof::Unknown);
    }

    #[test]
    fn positive_proves_expanded_monic_square_plus_constant() {
        let mut ctx = cas_ast::Context::new();
        let expr = parse("x^4 + 2*x^3 + 3*x^2 + 2*x + 8", &mut ctx).expect("parse");
        let out = prove_positive_depth_with(&ctx, expr, 20, true, |_ctx, _expr, _depth| {
            TriProof::Unknown
        });
        assert_eq!(out, TriProof::Proven);
    }

    #[test]
    fn positive_proves_expanded_positive_quadratic_square_minus_small_constant() {
        let mut ctx = cas_ast::Context::new();
        let expr = parse("x^4 + 2*x^3 + 7*x^2 + 6*x + 7", &mut ctx).expect("parse");
        let out = prove_positive_depth_with(&ctx, expr, 20, true, |_ctx, _expr, _depth| {
            TriProof::Unknown
        });
        assert_eq!(out, TriProof::Proven);
    }

    #[test]
    fn positive_proves_expanded_scaled_positive_quadratic_square_minus_small_constant() {
        let mut ctx = cas_ast::Context::new();
        let expr = parse("2*x^4 + 4*x^3 + 14*x^2 + 12*x + 17", &mut ctx).expect("parse");
        let out = prove_positive_depth_with(&ctx, expr, 20, true, |_ctx, _expr, _depth| {
            TriProof::Unknown
        });
        assert_eq!(out, TriProof::Proven);
    }

    #[test]
    fn positive_does_not_prove_expanded_monic_perfect_square_without_offset() {
        let mut ctx = cas_ast::Context::new();
        let expr = parse("x^4 + 2*x^3 + 3*x^2 + 2*x + 1", &mut ctx).expect("parse");
        let out = prove_positive_depth_with(&ctx, expr, 20, true, |_ctx, _expr, _depth| {
            TriProof::Unknown
        });
        assert_eq!(out, TriProof::Unknown);
    }

    #[test]
    fn positive_does_not_prove_expanded_quadratic_square_minus_large_constant() {
        let mut ctx = cas_ast::Context::new();
        let expr = parse("x^4 + 2*x^3 + 7*x^2 + 6*x - 1", &mut ctx).expect("parse");
        let out = prove_positive_depth_with(&ctx, expr, 20, true, |_ctx, _expr, _depth| {
            TriProof::Unknown
        });
        assert_eq!(out, TriProof::Unknown);
    }

    #[test]
    fn positive_does_not_prove_expanded_scaled_quadratic_square_minus_large_constant() {
        let mut ctx = cas_ast::Context::new();
        let expr = parse("2*x^4 + 4*x^3 + 14*x^2 + 12*x", &mut ctx).expect("parse");
        let out = prove_positive_depth_with(&ctx, expr, 20, true, |_ctx, _expr, _depth| {
            TriProof::Unknown
        });
        assert_eq!(out, TriProof::Unknown);
    }

    #[test]
    fn nonnegative_proves_expanded_perfect_square_quadratic() {
        let mut ctx = cas_ast::Context::new();
        let expr = parse("x^2 + 2*x + 1", &mut ctx).expect("parse");
        let out = prove_nonnegative_depth_with(&ctx, expr, 20, true, |_ctx, _expr, _depth| {
            TriProof::Unknown
        });
        assert_eq!(out, TriProof::Proven);
    }

    #[test]
    fn real_positive_proves_bounded_trig_affine_with_strict_margin() {
        let mut ctx = cas_ast::Context::new();
        let sin_shift = parse("sin(x)+2", &mut ctx).expect("parse");
        let cos_shift = parse("2-cos(x)", &mut ctx).expect("parse");
        let multi_shift = parse("2*sin(x)+cos(y)+4", &mut ctx).expect("parse");

        let sin_out =
            prove_positive_depth_with(&ctx, sin_shift, 20, true, |_ctx, _expr, _depth| {
                TriProof::Unknown
            });
        let cos_out =
            prove_positive_depth_with(&ctx, cos_shift, 20, true, |_ctx, _expr, _depth| {
                TriProof::Unknown
            });
        let multi_out =
            prove_positive_depth_with(&ctx, multi_shift, 20, true, |_ctx, _expr, _depth| {
                TriProof::Unknown
            });

        assert_eq!(sin_out, TriProof::Proven);
        assert_eq!(cos_out, TriProof::Proven);
        assert_eq!(multi_out, TriProof::Proven);
    }

    #[test]
    fn positive_bounded_trig_affine_rejects_boundary_and_complex_domain() {
        let mut ctx = cas_ast::Context::new();
        let boundary = parse("sin(x)+1", &mut ctx).expect("parse");
        let multi_boundary = parse("sin(x)+cos(x)+2", &mut ctx).expect("parse");
        let strict = parse("sin(x)+2", &mut ctx).expect("parse");

        let boundary_out =
            prove_positive_depth_with(&ctx, boundary, 20, true, |_ctx, _expr, _depth| {
                TriProof::Unknown
            });
        let multi_boundary_out =
            prove_positive_depth_with(&ctx, multi_boundary, 20, true, |_ctx, _expr, _depth| {
                TriProof::Unknown
            });
        let complex_out =
            prove_positive_depth_with(&ctx, strict, 20, false, |_ctx, _expr, _depth| {
                TriProof::Unknown
            });

        assert_eq!(boundary_out, TriProof::Unknown);
        assert_eq!(multi_boundary_out, TriProof::Unknown);
        assert_eq!(complex_out, TriProof::Unknown);
    }

    #[test]
    fn real_nonnegative_proves_bounded_trig_affine_boundary() {
        let mut ctx = cas_ast::Context::new();
        let expr = parse("sin(x)+cos(y)+2", &mut ctx).expect("parse");
        let out = prove_nonnegative_depth_with(&ctx, expr, 20, true, |_ctx, _expr, _depth| {
            TriProof::Unknown
        });
        assert_eq!(out, TriProof::Proven);
    }

    #[test]
    fn positive_abs_uses_nonzero_callback() {
        let mut ctx = cas_ast::Context::new();
        let expr = parse("abs(x)", &mut ctx).expect("parse");
        let x = parse("x", &mut ctx).expect("parse");
        let out = prove_positive_depth_with(&ctx, expr, 20, true, |_ctx, candidate, _depth| {
            if candidate == x {
                TriProof::Proven
            } else {
                TriProof::Unknown
            }
        });
        assert_eq!(out, TriProof::Proven);
    }

    #[test]
    fn complex_domain_exp_symbolic_is_unknown() {
        let mut ctx = cas_ast::Context::new();
        let expr = parse("exp(x)", &mut ctx).expect("parse");
        let out = prove_positive_depth_with(&ctx, expr, 20, false, |_ctx, _expr, _depth| {
            TriProof::Unknown
        });
        assert_eq!(out, TriProof::Unknown);
    }

    #[test]
    fn real_cosh_is_strictly_positive() {
        let mut ctx = cas_ast::Context::new();
        let expr = parse("cosh(sqrt(x))", &mut ctx).expect("parse");

        let real_positive =
            prove_positive_depth_with(&ctx, expr, 20, true, |_ctx, _expr, _depth| {
                TriProof::Unknown
            });
        let complex_positive =
            prove_positive_depth_with(&ctx, expr, 20, false, |_ctx, _expr, _depth| {
                TriProof::Unknown
            });

        assert_eq!(real_positive, TriProof::Proven);
        assert_eq!(complex_positive, TriProof::Unknown);
    }

    #[test]
    fn proves_ln_of_rational_constant_sign() {
        let mut ctx = cas_ast::Context::new();
        let ln_two = parse("ln(2)", &mut ctx).expect("parse");
        let ln_half = parse("ln(1/2)", &mut ctx).expect("parse");
        let ln_one = parse("ln(1)", &mut ctx).expect("parse");

        let positive_two =
            prove_positive_depth_with(&ctx, ln_two, 20, true, |_ctx, _expr, _depth| {
                TriProof::Unknown
            });
        let positive_half =
            prove_positive_depth_with(&ctx, ln_half, 20, true, |_ctx, _expr, _depth| {
                TriProof::Unknown
            });
        let nonnegative_one =
            prove_nonnegative_depth_with(&ctx, ln_one, 20, true, |_ctx, _expr, _depth| {
                TriProof::Unknown
            });

        assert_eq!(positive_two, TriProof::Proven);
        assert_eq!(positive_half, TriProof::Disproven);
        assert_eq!(nonnegative_one, TriProof::Proven);
    }

    #[test]
    fn proves_positive_quadratic_with_constant_surd_coefficients() {
        // The rational extractor cannot represent surd coefficients; the exact
        // interval path decides the discriminant sign instead (G1 R3: the
        // doubly-even octic render's `ln` arguments carry NESTED surd linear
        // coefficients like `√(2−√2)`).
        let mut ctx = cas_ast::Context::new();
        let unknown =
            |_ctx: &cas_ast::Context, _expr: cas_ast::ExprId, _depth: usize| TriProof::Unknown;
        for src in [
            "x^2 + sqrt(2 - sqrt(2))*x + 1", // R3 family +: disc = (2−√2) − 4 < 0
            "x^2 - sqrt(2 - sqrt(2))*x + 1", // negated linear term
            "x^2 + sqrt(sqrt(2) + 2)*x + 1", // R3 family −: disc = (2+√2) − 4 < 0
            "x^2 - sqrt(3)*x + 1",           // flat surd: disc = 3 − 4 < 0
            "x^2 + sqrt(2 - sqrt(2))*x + 2", // scaled constant term
        ] {
            let expr = parse(src, &mut ctx).expect(src);
            let proof = prove_positive_depth_with(&ctx, expr, 20, true, unknown);
            assert_eq!(proof, TriProof::Proven, "{src} is positive-definite");
        }

        // Honest declines: a genuinely sign-changing quadratic must NOT prove
        // (disc > 0: real roots), and a zero-discriminant one must not prove
        // STRICT positivity (the interval cannot certify equality either —
        // conservative Unknown, never a false Proven).
        for src in [
            "x^2 + sqrt(5)*x + 1",           // disc = 5 − 4 > 0: has real roots
            "x^2 + sqrt(2)*x + 1/2",         // disc = 0: touches zero
            "x^2 + sqrt(2 - sqrt(2))*x - 1", // negative constant term
            "-x^2 + sqrt(3)*x - 1",          // negative leading coefficient
        ] {
            let expr = parse(src, &mut ctx).expect(src);
            let proof = prove_positive_depth_with(&ctx, expr, 20, true, unknown);
            assert_ne!(proof, TriProof::Proven, "{src} must not prove positive");
        }
    }
}
