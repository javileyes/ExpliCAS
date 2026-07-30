// ========== Numeric Evaluation ==========

use cas_ast::{Context, Expr, ExprId};

/// Default depth limit for numeric evaluation.
/// Prevents stack overflow on deeply nested expressions.
pub const DEFAULT_NUMERIC_EVAL_DEPTH: usize = 50;

/// Extract a rational constant from an expression, handling multiple representations.
/// Uses default depth limit (50) to prevent stack overflow.
///
/// Supports (all must be purely numeric - returns None if any variable/function present):
/// - `Number(n)` - direct rational
/// - `Div(a, b)` - fraction (recursive)
/// - `Neg(a)` - negation (recursive)
/// - `Mul(a, b)` - product (recursive)
/// - `Add(a, b)` - sum (recursive)
/// - `Sub(a, b)` - difference (recursive)
///
/// This is the canonical helper for numeric evaluation. Used by:
/// - `SemanticEqualityChecker::try_evaluate_numeric`
/// - `EvaluatePowerRule` for exponent matching
pub fn as_rational_const(ctx: &Context, expr: ExprId) -> Option<num_rational::BigRational> {
    as_rational_const_depth(ctx, expr, DEFAULT_NUMERIC_EVAL_DEPTH)
}

/// Extract a rational constant with explicit depth limit.
/// Returns None if depth is exhausted (prevents stack overflow on deep expressions).
pub fn as_rational_const_depth(
    ctx: &Context,
    expr: ExprId,
    depth: usize,
) -> Option<num_rational::BigRational> {
    use num_traits::Zero;

    if depth == 0 {
        return None; // Depth budget exhausted
    }

    match ctx.get(expr) {
        Expr::Number(n) => Some(n.clone()),

        Expr::Div(num, den) => {
            let n = as_rational_const_depth(ctx, *num, depth - 1)?;
            let d = as_rational_const_depth(ctx, *den, depth - 1)?;
            if !d.is_zero() {
                Some(n / d)
            } else {
                None
            }
        }

        Expr::Neg(inner) => {
            let val = as_rational_const_depth(ctx, *inner, depth - 1)?;
            Some(-val)
        }

        Expr::Mul(l, r) => {
            let lv = as_rational_const_depth(ctx, *l, depth - 1)?;
            let rv = as_rational_const_depth(ctx, *r, depth - 1)?;
            Some(lv * rv)
        }

        Expr::Add(l, r) => {
            let lv = as_rational_const_depth(ctx, *l, depth - 1)?;
            let rv = as_rational_const_depth(ctx, *r, depth - 1)?;
            Some(lv + rv)
        }

        Expr::Sub(l, r) => {
            let lv = as_rational_const_depth(ctx, *l, depth - 1)?;
            let rv = as_rational_const_depth(ctx, *r, depth - 1)?;
            Some(lv - rv)
        }

        // Variables, Constants, Functions, Pow, Matrix -> not purely numeric
        _ => None,
    }
}

/// Check if an expression contains an integral (for auto-context detection).
///
/// Searches the expression tree for `integrate(...)` function calls.
/// Uses iterative traversal to avoid stack overflow on deep expressions.
pub fn contains_integral(ctx: &Context, root: ExprId) -> bool {
    let mut stack = vec![root];

    while let Some(e) = stack.pop() {
        match ctx.get(e) {
            Expr::Function(name, args) => {
                let fn_name = ctx.sym_name(*name);
                if fn_name == "integrate" || fn_name == "int" {
                    return true;
                }
                for arg in args {
                    stack.push(*arg);
                }
            }
            Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Pow(l, r) => {
                stack.push(*l);
                stack.push(*r);
            }
            Expr::Neg(inner) => {
                stack.push(*inner);
            }
            Expr::Matrix { data, .. } => {
                for elem in data {
                    stack.push(*elem);
                }
            }
            Expr::Div(num, den) => {
                stack.push(*num);
                stack.push(*den);
            }
            // Leaf nodes: nothing to push
            Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => {}
            // Hold is transparent - traverse inner
            Expr::Hold(inner) => stack.push(*inner),
        }
    }

    false
}

/// Check if an expression contains the imaginary unit `i` anywhere.
/// Check if an expression contains the imaginary unit `i` or imaginary-producing expressions.
/// Detects: Constant::I, sqrt(-1), (-1)^(1/2), and similar patterns.
/// Uses iterative traversal to avoid stack overflow on deep expressions.
pub fn contains_i(ctx: &Context, root: ExprId) -> bool {
    let mut stack = vec![root];

    while let Some(e) = stack.pop() {
        match ctx.get(e) {
            Expr::Constant(c) if *c == cas_ast::Constant::I => {
                return true;
            }
            // Check for sqrt(-1) pattern
            Expr::Function(fn_id, args)
                if ctx.is_builtin(*fn_id, cas_ast::BuiltinFn::Sqrt) && args.len() == 1 =>
            {
                if is_negative_one(ctx, args[0]) {
                    return true;
                }
                // Still need to traverse the arg for nested i
                stack.push(args[0]);
            }
            // Check for (-1)^(1/2) pattern
            Expr::Pow(base, exp) => {
                if is_negative_one(ctx, *base) && is_one_half(ctx, *exp) {
                    return true;
                }
                stack.push(*base);
                stack.push(*exp);
            }
            Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) => {
                stack.push(*l);
                stack.push(*r);
            }
            Expr::Neg(inner) => {
                stack.push(*inner);
            }
            Expr::Function(_, args) => {
                for arg in args {
                    stack.push(*arg);
                }
            }
            Expr::Matrix { data, .. } => {
                for elem in data {
                    stack.push(*elem);
                }
            }
            Expr::Div(num, den) => {
                stack.push(*num);
                stack.push(*den);
            }
            // Leaf nodes: nothing to push
            Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => {}
            // Hold is transparent - traverse inner
            Expr::Hold(inner) => stack.push(*inner),
        }
    }

    false
}

/// Like [`contains_i`], but also recognizes an EVEN ROOT OF A PROVABLY-NEGATIVE
/// value — `(-4)^(1/2)`, `sqrt(-9)`, `(-2)^(3/2)`, `(-16)^(1/4)` — which the engine
/// stores as `(-n)^(1/2)` (and renders as the imaginary `k·(-1)^(1/2)`). Used to
/// emit the imaginary-usage caveat on a RESULT that folded to an imaginary value
/// even when the INPUT had no literal `i` (Round-4 Cluster H: `sqrt(-4)+sqrt(-9)`
/// → `5·(-1)^(1/2)`). Kept SEPARATE from `contains_i` so the input-gated
/// complex-mode resolution is unaffected — only the result-level caveat changes.
pub fn expr_contains_imaginary(ctx: &Context, root: ExprId) -> bool {
    use num_traits::Signed;

    // Negativity must be decided EXACTLY but not only for rational literals:
    // surd/transcendental constants are decidable (`provable_const_sign`), and
    // missing them lets an even root like `sqrt(-pi^2)` (= i·π) pass as "real"
    // — the F0 adversarial sweep used exactly that spelling to reach a pole.
    let is_neg_const = |e: ExprId| {
        as_rational_const(ctx, e).is_some_and(|v| v.is_negative())
            || matches!(
                crate::const_sign::provable_const_sign(ctx, e),
                Some(crate::const_sign::ConstSign::Negative)
            )
    };
    let mut stack = vec![root];
    while let Some(e) = stack.pop() {
        match ctx.get(e) {
            Expr::Constant(c) if *c == cas_ast::Constant::I => return true,
            // `sqrt` of a provably-negative value (covers `sqrt(-1)`).
            Expr::Function(fn_id, args)
                if ctx.is_builtin(*fn_id, cas_ast::BuiltinFn::Sqrt) && args.len() == 1 =>
            {
                if is_neg_const(args[0]) {
                    return true;
                }
                stack.push(args[0]);
            }
            // An EVEN root (`exp` with even denominator) of a provably-negative base
            // is imaginary; an odd root (`(-8)^(1/3) = -2`) is real and is NOT flagged.
            Expr::Pow(base, exp) => {
                let (base, exp) = (*base, *exp);
                if as_rational_const(ctx, exp)
                    .is_some_and(|n| crate::expr_predicates::is_even_root_exponent(&n))
                    && is_neg_const(base)
                {
                    return true;
                }
                stack.push(base);
                stack.push(exp);
            }
            Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) => {
                stack.push(*l);
                stack.push(*r);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(*inner),
            Expr::Function(_, args) => {
                for a in args {
                    stack.push(*a);
                }
            }
            Expr::Matrix { data, .. } => {
                for el in data {
                    stack.push(*el);
                }
            }
            _ => {}
        }
    }
    false
}

/// Check if an expression represents -1
fn is_negative_one(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Number(n) => *n == num_rational::BigRational::from_integer((-1).into()),
        Expr::Neg(inner) => {
            matches!(
                ctx.get(*inner),
                Expr::Number(n) if *n == num_rational::BigRational::from_integer(1.into())
            )
        }
        _ => false,
    }
}

/// Check if an expression represents 1/2
fn is_one_half(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Number(n) => *n == num_rational::BigRational::new(1.into(), 2.into()),
        Expr::Div(num, den) => {
            matches!((ctx.get(*num), ctx.get(*den)),
                (Expr::Number(n), Expr::Number(d))
                if *n == num_rational::BigRational::from_integer(1.into())
                && *d == num_rational::BigRational::from_integer(2.into())
            )
        }
        _ => false,
    }
}

/// Exact polynomial zero check: probes may REFUTE, only algebra CONFIRMS.
///
/// Decides whether an expression is identically zero. Handles cases where
/// `expand()` produces raw unsimplified AST (e.g., `u·u + u·1 - u² - u`)
/// that structural comparison can't match.
///
/// Contract (R2: soundness gates are exact, never probabilistic):
/// - Exact rational probe evaluations can only return `false` (a nonzero
///   value at an exact rational point PROVES non-zero). They are a fast
///   pre-filter, never a source of `true`.
/// - `true` comes exclusively from exact polynomial normalization
///   (`poly_compare::poly_is_zero`, BigRational coefficients, budgeted).
/// - Anything the normalizer can't express (surds, π/e, trig, |·|,
///   oversized) yields a conservative `false`: "cannot prove zero".
///
/// History (auditoría 2026-07-30, fichas S2a-001/002/003, S2b-002, Q1b-001):
/// the previous version CONFIRMED zero from 3 FIXED public probes — any
/// nonzero polynomial vanishing at x=2/3, 3/5, 5/3 collapsed to 0
/// (`((x-2/3)·(x-3/5)·(x-5/3)+1)/y - 1/y` → `0`) — rationalized π→355/113
/// (`pi*x/y - (355/113)*x/y` → `0`) and kept an f64 `1e-10` gate for
/// surd/transcendental forms (`(sin(x/10^11)+1)/y - 1/y` → `0`). All three
/// confirmation channels are gone; only the refutation power remains.
pub(crate) fn numeric_poly_zero_check(ctx: &mut Context, expr: ExprId) -> bool {
    use num_rational::BigRational;
    use num_traits::Zero;

    // Collect variables in the expression
    let vars = cas_ast::collect_variables(ctx, expr);
    if vars.is_empty() {
        // No variables: decide zero EXACTLY. An f64 near-zero is NEVER a sufficient
        // condition for "is zero" — it falsely collapses a nonzero IRRATIONAL constant
        // that merely f64-matches a decimal literal (e.g. `ln(2)/ln(3) - 0.6309297535714574`,
        // whose true residual is ~3.7e-17 but folds to 0 under a `1e-10` gate, asserting a
        // false identity). Soundness gates must be exact (no `f64` drop/keep decisions).
        if let Some(v) = as_rational_const(ctx, expr) {
            return v.is_zero();
        }
        // Exact algebraic-zero check (rationals, perfect-square surds like `sqrt(4)-2`,
        // and anything the exact rational sign oracle pins to zero); bails to `false`
        // when it cannot PROVE zero, never guessing from a float.
        return matches!(
            crate::const_sign::provable_const_sign(ctx, expr),
            Some(crate::const_sign::ConstSign::Zero)
        );
    }

    // Limit to reasonable number of variables (avoid combinatorial explosion)
    if vars.len() > 5 {
        return false;
    }

    // Probe points: distinct rationals unlikely to be roots of spurious polynomials
    let probes: Vec<Vec<BigRational>> = zero_probe_sets();

    let var_list: Vec<String> = vars.into_iter().collect();

    // Exact BigRational probes: REFUTATION-ONLY fast path. A nonzero value at
    // an exact rational point proves non-zero; an all-zeros run proves
    // nothing (any polynomial with those probes among its roots vanishes on
    // every probe set — that was the S2a-001 exploit).
    for probe_set in &probes {
        match eval_with_substitution(ctx, expr, &var_list, probe_set) {
            Some(val) => {
                if !val.is_zero() {
                    return false; // Non-zero at this point => not identically zero
                }
            }
            // Not exactly evaluable (surd/π/function): the probes can't
            // refute; confirmation below decides.
            None => break,
        }
    }

    // CONFIRMATION: exact polynomial normalization or nothing. Opaque atoms
    // (e^x, sin(u), π…) become independent indeterminates — sound for
    // confirming, useless for refuting, which is exactly the split we need.
    crate::poly_compare::poly_is_zero_opaque(ctx, expr)
}

/// Weaker sibling for DENOMINATOR-equivalence grouping: exact probes refute,
/// plain rational-polynomial normalization confirms, and the opaque-atom /
/// Pythagorean closure is deliberately NOT applied. Making the grouping
/// STRONGER than it historically was reorders the fraction-combine flow and
/// breaks downstream pattern rules (R5: rewrite eligibility is order): with
/// the full closure, `1 − tanh²` and `1/cosh²` suddenly group as equal
/// denominators and the engine's own hyperbolic-Pythagoras rules never see
/// their expected shapes (the tanh⁴/⁶/⁸ verification residues stopped
/// closing). Equivalence here answers "may these fractions merge?", and the
/// honest answer set must stay what the rules were built around.
pub(crate) fn numeric_poly_zero_check_structural(ctx: &Context, expr: ExprId) -> bool {
    use num_rational::BigRational;
    use num_traits::Zero;

    let vars = cas_ast::collect_variables(ctx, expr);
    if vars.is_empty() {
        if let Some(v) = as_rational_const(ctx, expr) {
            return v.is_zero();
        }
        return matches!(
            crate::const_sign::provable_const_sign(ctx, expr),
            Some(crate::const_sign::ConstSign::Zero)
        );
    }
    if vars.len() > 5 {
        return false;
    }
    let probes: Vec<Vec<BigRational>> = zero_probe_sets();
    let var_list: Vec<String> = vars.into_iter().collect();
    for probe_set in &probes {
        match eval_with_substitution(ctx, expr, &var_list, probe_set) {
            Some(val) => {
                if !val.is_zero() {
                    return false;
                }
            }
            None => break,
        }
    }
    crate::poly_compare::poly_is_zero(ctx, expr)
}

/// Shared exact probe grid for the zero-check pair (full + structural).
fn zero_probe_sets() -> Vec<Vec<num_rational::BigRational>> {
    use num_rational::BigRational;
    vec![
        vec![
            BigRational::new(2.into(), 3.into()),
            BigRational::new(5.into(), 7.into()),
            BigRational::new(3.into(), 11.into()),
            BigRational::new(7.into(), 13.into()),
            BigRational::new(11.into(), 17.into()),
        ],
        vec![
            BigRational::new(3.into(), 5.into()),
            BigRational::new(7.into(), 11.into()),
            BigRational::new(11.into(), 13.into()),
            BigRational::new(13.into(), 17.into()),
            BigRational::new(17.into(), 19.into()),
        ],
        vec![
            BigRational::new(5.into(), 3.into()),
            BigRational::new(11.into(), 7.into()),
            BigRational::new(13.into(), 11.into()),
            BigRational::new(17.into(), 13.into()),
            BigRational::new(19.into(), 17.into()),
        ],
    ]
}

/// Evaluate an expression by substituting rational values for variables.
/// Returns None if evaluation fails (e.g., division by zero, unsupported operation).
fn eval_with_substitution(
    ctx: &Context,
    expr: ExprId,
    var_names: &[String],
    values: &[num_rational::BigRational],
) -> Option<num_rational::BigRational> {
    use num_rational::BigRational;
    use num_traits::Zero;

    match ctx.get(expr) {
        Expr::Number(n) => Some(n.clone()),

        Expr::Variable(v) => {
            // Resolve variable name and look up in substitution map
            let name = ctx.sym_name(*v);
            var_names
                .iter()
                .position(|var_name| var_name == name)
                .and_then(|idx| values.get(idx).cloned())
        }

        // NO constant gets a rational stand-in. π→355/113 / e→193/71 made
        // `pi*x/y - (355/113)*x/y` evaluate to exactly 0 at every probe and
        // collapse to 0 (auditoría 2026-07-30, ficha S2a-002). An exact
        // evaluator that silently rationalizes a transcendental is not exact.
        Expr::Constant(_) => None,

        Expr::Add(l, r) => {
            let lv = eval_with_substitution(ctx, *l, var_names, values)?;
            let rv = eval_with_substitution(ctx, *r, var_names, values)?;
            Some(lv + rv)
        }

        Expr::Sub(l, r) => {
            let lv = eval_with_substitution(ctx, *l, var_names, values)?;
            let rv = eval_with_substitution(ctx, *r, var_names, values)?;
            Some(lv - rv)
        }

        Expr::Mul(l, r) => {
            let lv = eval_with_substitution(ctx, *l, var_names, values)?;
            let rv = eval_with_substitution(ctx, *r, var_names, values)?;
            Some(lv * rv)
        }

        Expr::Div(n, d) => {
            let nv = eval_with_substitution(ctx, *n, var_names, values)?;
            let dv = eval_with_substitution(ctx, *d, var_names, values)?;
            if dv.is_zero() {
                None // Division by zero
            } else {
                Some(nv / dv)
            }
        }

        Expr::Neg(inner) => {
            let v = eval_with_substitution(ctx, *inner, var_names, values)?;
            Some(-v)
        }

        Expr::Pow(base, exp) => {
            let bv = eval_with_substitution(ctx, *base, var_names, values)?;
            let ev = eval_with_substitution(ctx, *exp, var_names, values)?;
            // Only handle integer exponents for exact computation
            if ev.is_integer() {
                let n: i64 = ev.to_integer().try_into().ok()?;
                if (0..=20).contains(&n) {
                    let mut result = BigRational::from_integer(1.into());
                    for _ in 0..n {
                        result *= &bv;
                    }
                    Some(result)
                } else if (-20..0).contains(&n) && !bv.is_zero() {
                    let mut result = BigRational::from_integer(1.into());
                    for _ in 0..(-n) {
                        result *= &bv;
                    }
                    Some(BigRational::from_integer(1.into()) / result)
                } else {
                    None
                }
            } else {
                None // Non-integer exponent: can't compute exactly
            }
        }

        // Functions, Hold, Matrix, SessionRef: bail out
        _ => None,
    }
}

/// Evaluate an expression by substituting f64 values for variables.
/// Supports fractional exponents via `f64::powf()`.
/// Returns None if evaluation fails (division by zero, NaN, unsupported operations).
pub fn eval_f64_with_substitution(
    ctx: &Context,
    expr: ExprId,
    var_names: &[String],
    values: &[f64],
) -> Option<f64> {
    match ctx.get(expr) {
        Expr::Number(n) => {
            use num_traits::ToPrimitive;
            let f = n.numer().to_f64()? / n.denom().to_f64()?;
            if f.is_finite() {
                Some(f)
            } else {
                None
            }
        }

        Expr::Variable(v) => {
            let name = ctx.sym_name(*v);
            var_names
                .iter()
                .position(|var_name| var_name == name)
                .and_then(|idx| values.get(idx).copied())
        }

        Expr::Constant(c) => match c {
            cas_ast::Constant::Pi => Some(std::f64::consts::PI),
            cas_ast::Constant::E => Some(std::f64::consts::E),
            cas_ast::Constant::Phi => Some(1.618033988749895),
            _ => None,
        },

        Expr::Add(l, r) => {
            let lv = eval_f64_with_substitution(ctx, *l, var_names, values)?;
            let rv = eval_f64_with_substitution(ctx, *r, var_names, values)?;
            let result = lv + rv;
            if result.is_finite() {
                Some(result)
            } else {
                None
            }
        }

        Expr::Sub(l, r) => {
            let lv = eval_f64_with_substitution(ctx, *l, var_names, values)?;
            let rv = eval_f64_with_substitution(ctx, *r, var_names, values)?;
            let result = lv - rv;
            if result.is_finite() {
                Some(result)
            } else {
                None
            }
        }

        Expr::Mul(l, r) => {
            let lv = eval_f64_with_substitution(ctx, *l, var_names, values)?;
            let rv = eval_f64_with_substitution(ctx, *r, var_names, values)?;
            let result = lv * rv;
            if result.is_finite() {
                Some(result)
            } else {
                None
            }
        }

        Expr::Div(n, d) => {
            let nv = eval_f64_with_substitution(ctx, *n, var_names, values)?;
            let dv = eval_f64_with_substitution(ctx, *d, var_names, values)?;
            if dv.abs() < 1e-15 {
                return None;
            } // Avoid division by near-zero
            let result = nv / dv;
            if result.is_finite() {
                Some(result)
            } else {
                None
            }
        }

        Expr::Neg(inner) => {
            let v = eval_f64_with_substitution(ctx, *inner, var_names, values)?;
            Some(-v)
        }

        Expr::Pow(base, exp) => {
            let bv = eval_f64_with_substitution(ctx, *base, var_names, values)?;
            let ev = eval_f64_with_substitution(ctx, *exp, var_names, values)?;
            let result = bv.powf(ev);
            if result.is_finite() {
                Some(result)
            } else {
                None
            }
        }

        Expr::Function(fn_id, args) => {
            let name = ctx.sym_name(*fn_id);
            match name {
                "sqrt" if args.len() == 1 => {
                    let av = eval_f64_with_substitution(ctx, args[0], var_names, values)?;
                    if av >= 0.0 {
                        let result = av.sqrt();
                        if result.is_finite() {
                            Some(result)
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                }
                "abs" if args.len() == 1 => {
                    let av = eval_f64_with_substitution(ctx, args[0], var_names, values)?;
                    Some(av.abs())
                }
                "sin" if args.len() == 1 => {
                    let av = eval_f64_with_substitution(ctx, args[0], var_names, values)?;
                    Some(av.sin())
                }
                "cos" if args.len() == 1 => {
                    let av = eval_f64_with_substitution(ctx, args[0], var_names, values)?;
                    Some(av.cos())
                }
                "tan" if args.len() == 1 => {
                    let av = eval_f64_with_substitution(ctx, args[0], var_names, values)?;
                    let result = av.tan();
                    if result.is_finite() {
                        Some(result)
                    } else {
                        None
                    }
                }
                "ln" if args.len() == 1 => {
                    let av = eval_f64_with_substitution(ctx, args[0], var_names, values)?;
                    if av > 0.0 {
                        let result = av.ln();
                        if result.is_finite() {
                            Some(result)
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                }
                "log" if args.len() == 1 => {
                    let av = eval_f64_with_substitution(ctx, args[0], var_names, values)?;
                    if av > 0.0 {
                        let result = av.ln();
                        if result.is_finite() {
                            Some(result)
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                }
                "log10" if args.len() == 1 => {
                    let av = eval_f64_with_substitution(ctx, args[0], var_names, values)?;
                    if av > 0.0 {
                        let result = av.log10();
                        if result.is_finite() {
                            Some(result)
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                }
                "exp" if args.len() == 1 => {
                    let av = eval_f64_with_substitution(ctx, args[0], var_names, values)?;
                    let result = av.exp();
                    if result.is_finite() {
                        Some(result)
                    } else {
                        None
                    }
                }
                "sinh" if args.len() == 1 => {
                    let av = eval_f64_with_substitution(ctx, args[0], var_names, values)?;
                    let result = av.sinh();
                    if result.is_finite() {
                        Some(result)
                    } else {
                        None
                    }
                }
                "cosh" if args.len() == 1 => {
                    let av = eval_f64_with_substitution(ctx, args[0], var_names, values)?;
                    let result = av.cosh();
                    if result.is_finite() {
                        Some(result)
                    } else {
                        None
                    }
                }
                "tanh" if args.len() == 1 => {
                    let av = eval_f64_with_substitution(ctx, args[0], var_names, values)?;
                    Some(av.tanh())
                }
                "cot" if args.len() == 1 => {
                    let av = eval_f64_with_substitution(ctx, args[0], var_names, values)?;
                    let s = av.sin();
                    if s.abs() < 1e-15 {
                        return None;
                    }
                    let result = av.cos() / s;
                    if result.is_finite() {
                        Some(result)
                    } else {
                        None
                    }
                }
                "sec" if args.len() == 1 => {
                    let av = eval_f64_with_substitution(ctx, args[0], var_names, values)?;
                    let c = av.cos();
                    if c.abs() < 1e-15 {
                        return None;
                    }
                    let result = 1.0 / c;
                    if result.is_finite() {
                        Some(result)
                    } else {
                        None
                    }
                }
                "csc" if args.len() == 1 => {
                    let av = eval_f64_with_substitution(ctx, args[0], var_names, values)?;
                    let s = av.sin();
                    if s.abs() < 1e-15 {
                        return None;
                    }
                    let result = 1.0 / s;
                    if result.is_finite() {
                        Some(result)
                    } else {
                        None
                    }
                }
                _ => None, // Unsupported function
            }
        }

        Expr::Hold(inner) => eval_f64_with_substitution(ctx, *inner, var_names, values),

        // Matrix, SessionRef: bail out
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{expr_contains_imaginary, numeric_poly_zero_check};
    use cas_ast::Context;
    use cas_parser::parse;

    #[test]
    fn numeric_poly_zero_check_rejects_variable_abs_piecewise_identity() {
        let mut ctx = Context::new();
        let expr = parse("a + 1 - abs(a + 1)", &mut ctx).expect("parse");

        assert!(!numeric_poly_zero_check(&mut ctx, expr));
    }

    #[test]
    fn numeric_poly_zero_check_is_exact_not_f64_for_constants() {
        // SOUNDNESS (H1): a nonzero IRRATIONAL constant that merely f64-matches a
        // decimal literal must NOT be deemed zero. `ln(2)/ln(3) - 0.6309297535714574`
        // has true residual ~3.7e-17 (a transcendental ratio cannot equal a finite
        // decimal); the old `eval_f64(...).abs() < 1e-10` gate folded it to a false 0.
        let mut ctx = Context::new();
        let near = parse("log(2)/log(3) - 0.6309297535714574", &mut ctx).expect("parse");
        assert!(
            !numeric_poly_zero_check(&mut ctx, near),
            "f64-close transcendental must not be deemed exactly zero"
        );

        // Still detects EXACT zeros (rational + perfect-square surd via the exact
        // sign oracle), so the fix does not lose sound zero-detection.
        let rat = parse("1/2 - 0.5", &mut ctx).expect("parse");
        assert!(numeric_poly_zero_check(&mut ctx, rat));
        let surd = parse("sqrt(4) - 2", &mut ctx).expect("parse");
        assert!(numeric_poly_zero_check(&mut ctx, surd));
    }

    #[test]
    fn probes_never_confirm_polynomial_vanishing_on_all_probe_points() {
        // SOUNDNESS (auditoría 2026-07-30, ficha S2a-001): the probe values are
        // fixed and public; a nonzero polynomial with all of them among its
        // roots evaluates to 0 on every probe set. The old version CONFIRMED
        // zero from exactly that ("((x-2/3)*(x-3/5)*(x-5/3)+1)/y - 1/y" → 0).
        // Probes may refute; only the exact normalizer may confirm.
        let mut ctx = Context::new();
        let poison = parse("(x-2/3)*(x-3/5)*(x-5/3)", &mut ctx).expect("parse");
        assert!(
            !numeric_poly_zero_check(&mut ctx, poison),
            "polynomial vanishing on the probe grid must not be deemed zero"
        );
    }

    #[test]
    fn pi_is_not_355_over_113() {
        // SOUNDNESS (ficha S2a-002): the exact evaluator rationalized π→355/113
        // and e→193/71, so `pi*x - (355/113)*x` probed to exact 0 everywhere
        // and collapsed (`pi*x/y - (355/113)*x/y` → 0 at the CLI).
        let mut ctx = Context::new();
        let pi_diff = parse("pi*x - (355/113)*x", &mut ctx).expect("parse");
        assert!(!numeric_poly_zero_check(&mut ctx, pi_diff));
        let e_diff = parse("e*x - (193/71)*x", &mut ctx).expect("parse");
        assert!(!numeric_poly_zero_check(&mut ctx, e_diff));
    }

    #[test]
    fn f64_near_zero_with_variables_is_not_zero() {
        // SOUNDNESS (fichas S2a-003 / S2b-002 / Q1b-001): the f64 fallback
        // declared zero anything under 1e-10 on 3 probes —
        // `(sin(x/10^11)+1)/y - 1/y` → 0. The fallback is gone; forms the
        // exact machinery can't express yield a conservative `false`.
        let mut ctx = Context::new();
        let tiny = parse("sin(x/100000000000)", &mut ctx).expect("parse");
        assert!(!numeric_poly_zero_check(&mut ctx, tiny));
    }

    #[test]
    fn rational_zero_with_embedded_divisions_confirms() {
        // Capability pin (Bernoulli-IVP Gate 1, dsolve `y(0)=2`): the
        // semi-combined verification numerator carries embedded fractions AND
        // mixes e^x with e^(2x)/(e^x)². Zero is only visible through (a) the
        // rational normalization num/den and (b) the faithful fold
        // e^(k·g) = (e^g)^k. The old f64 fallback confirmed this unsoundly;
        // the exact channel must keep the capability.
        let mut ctx = Context::new();
        let gate1 =
            parse("(2-e^x)^2 * (2*e^x/(2-e^x)^2 + 2/(2-e^x)) - 4", &mut ctx).expect("parse");
        assert!(numeric_poly_zero_check(&mut ctx, gate1));

        // e^(2x) vs (e^x)^2: same identity spelled with an explicit product
        // exponent.
        let fold = parse("e^(2*x) - (e^x)^2", &mut ctx).expect("parse");
        assert!(numeric_poly_zero_check(&mut ctx, fold));

        // And the refutation direction stays exact: a NON-zero rational
        // function with embedded divisions must not confirm.
        let nonzero =
            parse("(2-e^x)^2 * (2*e^x/(2-e^x)^2 + 2/(2-e^x)) - 5", &mut ctx).expect("parse");
        assert!(!numeric_poly_zero_check(&mut ctx, nonzero));
    }

    #[test]
    fn affine_related_radicals_close_via_defining_relations() {
        // Capability pin (diff(arccos(1/√(x+1))) / arccot(√(4/x)) residues):
        // radicals over ALGEBRAICALLY RELATED bases (x and x+1; 4/x) need the
        // defining relations s_B^d = B and the gated product/quotient split —
        // and the exponent extractor must see through Neg(Div(3,2)) shapes
        // (as_rational_const, not bare Number matching).
        let mut ctx = Context::new();
        let e = parse(
            "(x/(x+1))^(-1/2) * (x+1)^(-3/2) * (sqrt(x) + x*sqrt(x)) - 1",
            &mut ctx,
        )
        .expect("parse");
        assert!(
            numeric_poly_zero_check(&mut ctx, e),
            "affine-related radicals"
        );

        // Refutation stays exact: perturbed sibling must NOT confirm.
        let bad = parse(
            "(x/(x+1))^(-1/2) * (x+1)^(-3/2) * (sqrt(x) + x*sqrt(x)) - 2",
            &mut ctx,
        )
        .expect("parse");
        assert!(!numeric_poly_zero_check(&mut ctx, bad));
    }

    #[test]
    fn exact_confirmation_still_covers_the_expand_clientele() {
        // The function's raison d'être (raw expand() output) must keep
        // confirming — now via exact normalization instead of probes.
        let mut ctx = Context::new();
        let raw = parse("u*u + u*1 - u^2 - u", &mut ctx).expect("parse");
        assert!(numeric_poly_zero_check(&mut ctx, raw));

        // Unexpanded products within budget are confirmed directly.
        let prod = parse("(x+1)*(x-1) - (x^2 - 1)", &mut ctx).expect("parse");
        assert!(numeric_poly_zero_check(&mut ctx, prod));

        // Degree above compare_budget's 10 but within the zero-check budget:
        // documents why poly_is_zero carries its own (wider) caps.
        let deg12 = parse("2*x^12*y - x^12*y - y*x^12", &mut ctx).expect("parse");
        assert!(numeric_poly_zero_check(&mut ctx, deg12));
    }

    #[test]
    fn expr_contains_imaginary_detects_even_root_of_negative() {
        // Round-4 Cluster H: an even root of a provably-negative value is imaginary,
        // even when the input has no literal `i` (the engine stores `(-n)^(1/2)`).
        let mut ctx = Context::new();
        for src in [
            "(-1)^(1/2)",
            "(-4)^(1/2)",
            "(-25)^(1/2)",
            "(-2)^(1/2)",
            "(-8)^(3/2)",
            "(-16)^(1/4)",
            "(-4)^(1/2) + (-9)^(1/2)",
            "5*(-1)^(1/2)",
            "i",
            "sqrt(-9)",
            // F0b: NON-rational provably-negative radicands (the adversarial
            // sweep reached a tanh pole via `sqrt(-pi^2)` = i·pi).
            "sqrt(-pi^2)",
            "(-pi)^(1/2)",
            "sqrt(1 - e)",
        ] {
            let e = parse(src, &mut ctx).expect("parse");
            assert!(expr_contains_imaginary(&ctx, e), "`{src}` is imaginary");
        }
        // Real values must NOT be flagged — including ODD roots of a negative.
        for src in [
            "(-8)^(1/3)",  // = -2, real
            "(-27)^(1/3)", // = -3, real
            "4^(1/2)",     // = 2
            "sqrt(2)",
            "5",
            "x^2",
            "(-1)^2",
            "2 + 3",
            // Provably-POSITIVE non-rational radicands stay real.
            "sqrt(pi^2)",
            "sqrt(e - 1)",
        ] {
            let e = parse(src, &mut ctx).expect("parse");
            assert!(
                !expr_contains_imaginary(&ctx, e),
                "`{src}` is a real value, must NOT be flagged imaginary"
            );
        }
    }
}
