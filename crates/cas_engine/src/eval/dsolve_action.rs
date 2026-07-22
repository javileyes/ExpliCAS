//! `EvalAction::Dsolve` handler — elementary ODE solving (Fase 4).
//!
//! O0 substrate: separable first-order equations `y' = f(x)·g(y)`. The ODE
//! tree arrives RAW (dispatch resolves session refs but never simplifies), and
//! this handler extracts its structure before any simplify pass touches it:
//! a plain-eval pass over a subtree holding `diff(y,·)` with a bare `y`
//! silently collapses the derivative to `0` (that is diff's contract for an
//! independent variable, not dsolve's). Emission is verification-gated (D5):
//! the candidate is substituted into the ODE and must reduce to an exact
//! symbolic `Number(0)` under the FULL evaluator with numeric verification
//! disabled — otherwise the command declines to an honest residual.

use super::*;
use cas_ast::eq::wrap_eq;
use cas_ast::traversal::collect_variables;
use cas_ast::{Equation, RelOp, SolutionSet};
use cas_formatter::render_expr;
use cas_math::substitute::{substitute_power_aware, SubstituteOptions};
use cas_solver_core::step_types::ImportanceLevel;
use num_traits::{One, Signed, Zero};

/// Push every direct child of `id` onto `stack` (local traversal helper).
fn push_children(ctx: &Context, id: ExprId, stack: &mut Vec<ExprId>) {
    match ctx.get(id) {
        cas_ast::Expr::Add(a, b)
        | cas_ast::Expr::Sub(a, b)
        | cas_ast::Expr::Mul(a, b)
        | cas_ast::Expr::Div(a, b)
        | cas_ast::Expr::Pow(a, b) => {
            stack.push(*a);
            stack.push(*b);
        }
        cas_ast::Expr::Neg(a) | cas_ast::Expr::Hold(a) => stack.push(*a),
        cas_ast::Expr::Function(_, args) => stack.extend(args.iter().copied()),
        cas_ast::Expr::Matrix { data, .. } => stack.extend(data.iter().copied()),
        _ => {}
    }
}

const DSOLVE_RULE: &str = "dsolve";
/// Wall-clock cap for the verification gate only. The counter Budget has no
/// clock; a hostile candidate (the known expand↔factor oscillation family) must
/// degrade to an honest "unverified → residual" decline, never block the eval.
/// Time may only ever turn "would verify" into "declined" — a conservative
/// under-answer — so it stays outside the exact-soundness rule for drop/keep.
/// Sized as a TERMINATION net, not a performance bound: every legitimate
/// candidate of the phase verifies far below it on both debug and release
/// profiles (a 3s cap made L11's emission depend on the compile profile —
/// debug verified slower than the cap), while the known genuine hang (the O23
/// constant-attached oscillation) exceeds it on every profile and O4 dodges it
/// by design (linearity-split verification).
const VERIFY_TIME_BUDGET_MS: u64 = 30_000;

/// Product-factor decomposition entry: (factor, lives_in_denominator).
type Factors = Vec<(ExprId, bool)>;

fn collect_product_factors(ctx: &Context, e: ExprId, invert: bool, out: &mut Factors) {
    match ctx.get(e) {
        cas_ast::Expr::Mul(a, b) => {
            let (a, b) = (*a, *b);
            collect_product_factors(ctx, a, invert, out);
            collect_product_factors(ctx, b, invert, out);
        }
        cas_ast::Expr::Div(a, b) => {
            let (a, b) = (*a, *b);
            collect_product_factors(ctx, a, invert, out);
            collect_product_factors(ctx, b, !invert, out);
        }
        cas_ast::Expr::Neg(a) => {
            let a = *a;
            // The sign is a free factor: park it as a marker the caller folds
            // into f(x). Represented by pushing the inner factors plus a flag
            // via a sentinel handled in `split_separable`.
            out.push((e, invert));
            let _ = a; // handled structurally below (Neg kept whole)
        }
        _ => out.push((e, invert)),
    }
}

struct SeparableSplit {
    /// f(x): product of factors free of the unknown (includes free parameters).
    fx: ExprId,
    /// g(y): product of factors in the unknown only. `None` ⇒ g ≡ 1 (direct integration).
    gy: Option<ExprId>,
}

/// Split `rhs = f(x)·g(y)` by multiplicative factor classification. Every
/// factor free of the unknown goes to `f(x)` (free parameters like `k`
/// included — declared spec); a factor mixing both variables means the RHS is
/// not separable. The RAW tree is read, never rewritten (D4).
fn split_separable(
    ctx: &mut Context,
    rhs: ExprId,
    func: &str,
    var: &str,
) -> Option<SeparableSplit> {
    // Neg heads: peel them counting sign so `-x/y` splits as `(-x)·(1/y)`.
    let mut negs = 0usize;
    let mut core = rhs;
    while let cas_ast::Expr::Neg(inner) = ctx.get(core) {
        negs += 1;
        core = *inner;
    }
    let mut factors: Factors = Vec::new();
    collect_product_factors(ctx, core, false, &mut factors);

    let mut fx_num: Vec<ExprId> = Vec::new();
    let mut fx_den: Vec<ExprId> = Vec::new();
    let mut gy_num: Vec<ExprId> = Vec::new();
    let mut gy_den: Vec<ExprId> = Vec::new();
    for (f, is_den) in factors {
        // Nested Neg inside a product: peel here too.
        let mut sub_negs = 0usize;
        let mut g = f;
        while let cas_ast::Expr::Neg(inner) = ctx.get(g) {
            sub_negs += 1;
            g = *inner;
        }
        negs += sub_negs;
        let vars = collect_variables(ctx, g);
        let has_y = vars.contains(func);
        let has_x = vars.contains(var);
        if has_y && has_x {
            return None;
        }
        if has_y {
            if is_den {
                gy_den.push(g)
            } else {
                gy_num.push(g)
            }
        } else if is_den {
            fx_den.push(g)
        } else {
            fx_num.push(g)
        }
    }

    let build_product = |ctx: &mut Context, parts: &[ExprId]| -> Option<ExprId> {
        let mut it = parts.iter();
        let first = *it.next()?;
        Some(it.fold(first, |acc, &p| ctx.add(cas_ast::Expr::Mul(acc, p))))
    };
    let build_ratio = |ctx: &mut Context, num: &[ExprId], den: &[ExprId]| -> Option<ExprId> {
        let n = build_product(ctx, num).unwrap_or_else(|| ctx.num(1));
        match build_product(ctx, den) {
            Some(d) => Some(ctx.add(cas_ast::Expr::Div(n, d))),
            None if num.is_empty() => None,
            None => Some(n),
        }
    };

    let mut fx = build_ratio(ctx, &fx_num, &fx_den).unwrap_or_else(|| ctx.num(1));
    if negs % 2 == 1 {
        fx = ctx.add(cas_ast::Expr::Neg(fx));
    }
    let gy = build_ratio(ctx, &gy_num, &gy_den);
    Some(SeparableSplit { fx, gy })
}

/// True when `e` is the literal number `1`.
fn is_literal_one(ctx: &Context, e: ExprId) -> bool {
    matches!(ctx.get(e), cas_ast::Expr::Number(n) if n.is_one())
}

/// Walk the tree looking for `diff(<func>, ...)` calls; report the maximum
/// arity found (2 = first order, 3+ = higher order sugar `diff(y,x,2)`).
/// `diff_sym` is the interned symbol for "diff" (interned once by the caller —
/// SymbolId comparison, never per-node string compares).
fn scan_diff_calls_of(
    ctx: &Context,
    root: ExprId,
    diff_sym: cas_ast::symbol::SymbolId,
    func: &str,
) -> Option<usize> {
    let mut max_arity: Option<usize> = None;
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        match ctx.get(id) {
            cas_ast::Expr::Function(fn_id, args) => {
                if *fn_id == diff_sym && !args.is_empty() {
                    if let cas_ast::Expr::Variable(s) = ctx.get(args[0]) {
                        if ctx.sym_name(*s) == func {
                            let a = args.len();
                            max_arity = Some(max_arity.map_or(a, |m: usize| m.max(a)));
                        }
                    }
                }
                stack.extend(args.iter().copied());
            }
            _ => push_children(ctx, id, &mut stack),
        }
    }
    max_arity
}

/// Differentiation order of `id` when it is a diff-call on `func` w.r.t.
/// `var`: `diff(y,x)` → 1, `diff(y,x,n)` (literal integer n ≥ 1) → n,
/// nested `diff(diff(y,x),x)` → inner + 1. `usize::MAX` encodes a symbolic
/// (non-literal) order. `None` when it is not a diff-of-func call at all.
fn diff_call_order(
    ctx: &Context,
    id: ExprId,
    diff_sym: cas_ast::symbol::SymbolId,
    func: &str,
    var: &str,
) -> Option<usize> {
    let cas_ast::Expr::Function(fn_id, args) = ctx.get(id) else {
        return None;
    };
    if *fn_id != diff_sym || args.len() < 2 {
        return None;
    }
    // Variable check on the differentiation variable.
    let var_ok = matches!(ctx.get(args[1]), cas_ast::Expr::Variable(v) if ctx.sym_name(*v) == var);
    if !var_ok {
        return None;
    }
    // Target: the bare unknown, or a nested diff-of-func (one order deeper).
    let base_order = match ctx.get(args[0]) {
        cas_ast::Expr::Variable(s) if ctx.sym_name(*s) == func => 0usize,
        _ => diff_call_order(ctx, args[0], diff_sym, func, var)?,
    };
    match args.len() {
        2 => Some(base_order + 1),
        3 => {
            let n: Option<usize> = cas_math::numeric_eval::as_rational_const(ctx, args[2])
                .filter(|n| {
                    n.is_integer() && *n >= num_rational::BigRational::from_integer(1.into())
                })
                .and_then(|n| n.to_integer().try_into().ok());
            match n {
                Some(count) => Some(base_order + count),
                None => Some(usize::MAX), // symbolic order
            }
        }
        _ => Some(usize::MAX),
    }
}

/// Maximum differentiation order of `func` found anywhere in the tree.
/// `None` = no diff-of-func at all; `usize::MAX` = a symbolic order appears.
fn scan_max_diff_order(
    ctx: &Context,
    root: ExprId,
    diff_sym: cas_ast::symbol::SymbolId,
    func: &str,
    var: &str,
) -> Option<usize> {
    let mut max_order: Option<usize> = None;
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        if let Some(order) = diff_call_order(ctx, id, diff_sym, func, var) {
            max_order = Some(max_order.map_or(order, |m| m.max(order)));
            // Do not descend into a recognized diff-call (its target is part
            // of the call itself).
            continue;
        }
        push_children(ctx, id, &mut stack);
    }
    max_order
}

/// Constant-coefficient second-order match `a·y'' + b·y' + c·y = rhs`
/// (a, b, c exact rationals; `rhs` is the unknown-free forcing term moved to
/// the right side — `None` for the homogeneous case).
struct SecondOrderOde {
    a: num_rational::BigRational,
    b: num_rational::BigRational,
    c: num_rational::BigRational,
    /// Unknown-free forcing term (already sign-flipped to the RHS). `None`
    /// when the equation is homogeneous.
    rhs: Option<ExprId>,
}

/// Match `a·y'' + b·y' + c·y = rhs` on the RAW tree with RATIONAL constant
/// coefficients. Declines (None) on any variable coefficient (`x·y''` —
/// Cauchy-Euler/Bessel stay honest residuals), nonlinear unknown (`y²`,
/// `sin(y)`), or non-product diff placement.
fn try_match_second_order_constant(
    ctx: &mut Context,
    eq: &Equation,
    diff_sym: cas_ast::symbol::SymbolId,
    func: &str,
    var: &str,
) -> Option<SecondOrderOde> {
    let mut terms: Vec<(ExprId, bool)> = Vec::new();
    collect_signed_terms(ctx, eq.lhs, true, &mut terms);
    collect_signed_terms(ctx, eq.rhs, false, &mut terms);

    let zero = num_rational::BigRational::from_integer(0.into());
    let mut a = zero.clone();
    let mut b = zero.clone();
    let mut c = zero.clone();
    let mut free_sum = zero.clone();
    let mut free_terms: Vec<ExprId> = Vec::new();
    for (term, positive) in terms {
        let mut sign = positive;
        let mut core = term;
        while let cas_ast::Expr::Neg(inner) = ctx.get(core) {
            sign = !sign;
            core = *inner;
        }
        let mut factors: Factors = Vec::new();
        collect_product_factors(ctx, core, false, &mut factors);

        // Sign at the TERM level (before factor-level Neg peeling): the free
        // branch re-uses the whole `core` tree, whose inner Negs must NOT be
        // double-counted through `sign`.
        let term_sign = sign;
        let mut order2 = 0usize;
        let mut order1 = 0usize;
        let mut y_count = 0usize;
        let mut coef = num_rational::BigRational::from_integer(1.into());
        let mut rational_coef = true;
        let mut ok = true;
        for (f, is_den) in factors {
            let mut g = f;
            while let cas_ast::Expr::Neg(inner) = ctx.get(g) {
                sign = !sign;
                g = *inner;
            }
            match diff_call_order(ctx, g, diff_sym, func, var) {
                Some(2) if !is_den => {
                    order2 += 1;
                    continue;
                }
                Some(1) if !is_den => {
                    order1 += 1;
                    continue;
                }
                Some(_) => {
                    ok = false;
                    break;
                }
                None => {}
            }
            if matches!(ctx.get(g), cas_ast::Expr::Variable(s) if ctx.sym_name(*s) == func) {
                if is_den {
                    ok = false;
                    break;
                }
                y_count += 1;
                continue;
            }
            let vars = collect_variables(ctx, g);
            if vars.contains(func) {
                ok = false; // nonlinear/nested unknown
                break;
            }
            // Coefficient factor: must fold to an exact rational (constant
            // coefficients only — variable coefficients decline).
            match cas_math::numeric_eval::as_rational_const(ctx, g) {
                Some(r) if is_den && !r.is_zero() => coef /= r,
                Some(r) if !is_den => coef *= r,
                _ => rational_coef = false,
            }
        }
        if !ok || order2 + order1 + y_count > 1 {
            return None;
        }
        if !rational_coef {
            // A y'/y''/y term with a NON-rational coefficient is a variable
            // coefficient — decline. A free term that is a FUNCTION of x
            // (`y'' + y = x`) is forcing: collect it (sign-flipped) as RHS.
            if order2 + order1 + y_count > 0 {
                return None;
            }
            let flipped = if term_sign {
                ctx.add(cas_ast::Expr::Neg(core))
            } else {
                core
            };
            free_terms.push(flipped);
            continue;
        }
        let signed = if sign { coef } else { -coef };
        if order2 == 1 {
            a += signed;
        } else if order1 == 1 {
            b += signed;
        } else if y_count == 1 {
            c += signed;
        } else {
            free_sum += signed;
        }
    }
    if a.is_zero() {
        return None;
    }
    // Assemble the forcing RHS: functional free terms plus any nonzero
    // rational free sum, all sign-flipped to the right side.
    if !free_sum.is_zero() {
        let n = ctx.add(cas_ast::Expr::Number(-free_sum.clone()));
        free_terms.push(n);
    }
    let rhs = free_terms
        .into_iter()
        .reduce(|acc, t| ctx.add(cas_ast::Expr::Add(acc, t)));
    Some(SecondOrderOde { a, b, c, rhs })
}

/// True when `id` is exactly the 2-arg call `diff(func, var)`.
fn is_first_order_diff_call(
    ctx: &Context,
    id: ExprId,
    diff_sym: cas_ast::symbol::SymbolId,
    func: &str,
    var: &str,
) -> bool {
    if let cas_ast::Expr::Function(fn_id, args) = ctx.get(id) {
        if *fn_id == diff_sym && args.len() == 2 {
            if let (cas_ast::Expr::Variable(f), cas_ast::Expr::Variable(v)) =
                (ctx.get(args[0]), ctx.get(args[1]))
            {
                return ctx.sym_name(*f) == func && ctx.sym_name(*v) == var;
            }
        }
    }
    false
}

/// `Some(rhs)` when one side of the equation is exactly `diff(func, var)` and
/// the other side is the RHS `f(x, y)` (O0 shape). Linear/exact rearrangements
/// (`y' + p·y = q`, `M + N·y' = 0`) are later cycles.
fn match_isolated_first_order(
    ctx: &Context,
    eq: &Equation,
    diff_sym: cas_ast::symbol::SymbolId,
    func: &str,
    var: &str,
) -> Option<ExprId> {
    if is_first_order_diff_call(ctx, eq.lhs, diff_sym, func, var) {
        return Some(eq.rhs);
    }
    if is_first_order_diff_call(ctx, eq.rhs, diff_sym, func, var) {
        return Some(eq.lhs);
    }
    None
}

/// Split `e` into additive terms with a sign flag (`true` = positive).
fn collect_signed_terms(ctx: &Context, e: ExprId, positive: bool, out: &mut Vec<(ExprId, bool)>) {
    match ctx.get(e) {
        cas_ast::Expr::Add(a, b) => {
            let (a, b) = (*a, *b);
            collect_signed_terms(ctx, a, positive, out);
            collect_signed_terms(ctx, b, positive, out);
        }
        cas_ast::Expr::Sub(a, b) => {
            let (a, b) = (*a, *b);
            collect_signed_terms(ctx, a, positive, out);
            collect_signed_terms(ctx, b, !positive, out);
        }
        cas_ast::Expr::Neg(a) => {
            let a = *a;
            collect_signed_terms(ctx, a, !positive, out);
        }
        _ => out.push((e, positive)),
    }
}

/// Normalized linear first-order form `y' + p(x)·y = q(x)`.
struct LinearOde {
    p: ExprId,
    q: ExprId,
}

/// Match `a(x)·y' + b(x)·y = c(x)` on the RAW tree (additive terms of
/// `LHS − RHS`, product factors per term) and normalize to `y' + p·y = q`.
/// Declines on any nonlinear appearance of the unknown (`y²`, `sin(y)`, `y` in
/// a denominator, `y·y'`, a diff nested deeper than a plain product factor).
fn try_match_linear_first_order(
    ctx: &mut Context,
    eq: &Equation,
    diff_sym: cas_ast::symbol::SymbolId,
    func: &str,
    var: &str,
) -> Option<LinearOde> {
    let mut terms: Vec<(ExprId, bool)> = Vec::new();
    collect_signed_terms(ctx, eq.lhs, true, &mut terms);
    collect_signed_terms(ctx, eq.rhs, false, &mut terms);

    let mut a_parts: Vec<ExprId> = Vec::new(); // Σ coef(y')
    let mut b_parts: Vec<ExprId> = Vec::new(); // Σ coef(y)
    let mut c_parts: Vec<ExprId> = Vec::new(); // Σ free terms moved to the RHS
    for (term, positive) in terms {
        // Peel outer Negs folded into the sign, then factor the product.
        let mut sign = positive;
        let mut core = term;
        while let cas_ast::Expr::Neg(inner) = ctx.get(core) {
            sign = !sign;
            core = *inner;
        }
        let mut factors: Factors = Vec::new();
        collect_product_factors(ctx, core, false, &mut factors);

        let mut diff_count = 0usize;
        let mut y_count = 0usize;
        let mut coef_num: Vec<ExprId> = Vec::new();
        let mut coef_den: Vec<ExprId> = Vec::new();
        let mut ok = true;
        for (f, is_den) in factors {
            let mut g = f;
            while let cas_ast::Expr::Neg(inner) = ctx.get(g) {
                sign = !sign;
                g = *inner;
            }
            if is_first_order_diff_call(ctx, g, diff_sym, func, var) {
                if is_den {
                    ok = false; // y' in a denominator: not linear
                    break;
                }
                diff_count += 1;
                continue;
            }
            let vars = collect_variables(ctx, g);
            if vars.contains(func) {
                if let cas_ast::Expr::Variable(s) = ctx.get(g) {
                    if ctx.sym_name(*s) == func && !is_den {
                        y_count += 1;
                        continue;
                    }
                }
                ok = false; // nonlinear/nested unknown (y², sin(y), y in denom)
                break;
            }
            if is_den {
                coef_den.push(g)
            } else {
                coef_num.push(g)
            }
        }
        if !ok || diff_count + y_count > 1 {
            return None;
        }

        let build_product = |ctx: &mut Context, parts: &[ExprId]| -> Option<ExprId> {
            let mut it = parts.iter();
            let first = *it.next()?;
            Some(it.fold(first, |acc, &p| ctx.add(cas_ast::Expr::Mul(acc, p))))
        };
        let num = build_product(ctx, &coef_num).unwrap_or_else(|| ctx.num(1));
        let mut coef = match build_product(ctx, &coef_den) {
            Some(d) => ctx.add(cas_ast::Expr::Div(num, d)),
            None => num,
        };
        if !sign {
            coef = ctx.add(cas_ast::Expr::Neg(coef));
        }
        if diff_count == 1 {
            a_parts.push(coef);
        } else if y_count == 1 {
            b_parts.push(coef);
        } else {
            // Free term: moving it to the RHS flips its sign.
            let negated = ctx.add(cas_ast::Expr::Neg(coef));
            c_parts.push(negated);
        }
    }
    if a_parts.is_empty() {
        return None;
    }
    let sum = |ctx: &mut Context, parts: &[ExprId]| -> Option<ExprId> {
        let mut it = parts.iter();
        let first = *it.next()?;
        Some(it.fold(first, |acc, &p| ctx.add(cas_ast::Expr::Add(acc, p))))
    };
    let a = sum(ctx, &a_parts).expect("nonempty");
    let b = sum(ctx, &b_parts).unwrap_or_else(|| ctx.num(0));
    let c = sum(ctx, &c_parts).unwrap_or_else(|| ctx.num(0));
    let p = if is_literal_one(ctx, a) {
        b
    } else {
        ctx.add(cas_ast::Expr::Div(b, a))
    };
    let q = if is_literal_one(ctx, a) {
        c
    } else {
        ctx.add(cas_ast::Expr::Div(c, a))
    };
    Some(LinearOde { p, q })
}

/// Decompose `e` into additive terms with rational multipliers, distributing
/// division/multiplication by numeric literals (`(x² + 2·C)/2` → `[(x², 1/2),
/// (C, 1)]`). Used to find and strip the `C` term of an exponent wherever the
/// solver's canonical form parked it.
fn collect_linear_terms(
    ctx: &Context,
    e: ExprId,
    mult: num_rational::BigRational,
    out: &mut Vec<(ExprId, num_rational::BigRational)>,
) {
    match ctx.get(e) {
        cas_ast::Expr::Add(a, b) => {
            let (a, b) = (*a, *b);
            collect_linear_terms(ctx, a, mult.clone(), out);
            collect_linear_terms(ctx, b, mult, out);
        }
        cas_ast::Expr::Sub(a, b) => {
            let (a, b) = (*a, *b);
            collect_linear_terms(ctx, a, mult.clone(), out);
            collect_linear_terms(ctx, b, -mult, out);
        }
        cas_ast::Expr::Neg(a) => {
            let a = *a;
            collect_linear_terms(ctx, a, -mult, out);
        }
        cas_ast::Expr::Div(a, b) => {
            let (a, b) = (*a, *b);
            if let cas_ast::Expr::Number(d) = ctx.get(b) {
                if !d.is_zero() {
                    collect_linear_terms(ctx, a, mult / d, out);
                    return;
                }
            }
            out.push((e, mult));
        }
        cas_ast::Expr::Mul(a, b) => {
            let (a, b) = (*a, *b);
            if let cas_ast::Expr::Number(k) = ctx.get(a) {
                collect_linear_terms(ctx, b, mult * k, out);
                return;
            }
            if let cas_ast::Expr::Number(k) = ctx.get(b) {
                collect_linear_terms(ctx, a, mult * k, out);
                return;
            }
            out.push((e, mult));
        }
        _ => out.push((e, mult)),
    }
}

/// Remove the `r·C` linear term (rational `r ≠ 0`) from an exponent: `H + r·C
/// → H`. Sound for absorption because `e^(H + r·C) = e^H·(e^C)^r` and `(e^C)^r`
/// ranges over `(0, ∞)` exactly as `e^C` does.
fn strip_additive_c(ctx: &mut Context, e: ExprId, c: ExprId) -> Option<ExprId> {
    let one = num_rational::BigRational::from_integer(1.into());
    let mut terms: Vec<(ExprId, num_rational::BigRational)> = Vec::new();
    collect_linear_terms(ctx, e, one, &mut terms);
    let c_pos = terms.iter().position(|(t, r)| *t == c && !r.is_zero())?;
    terms.remove(c_pos);
    // C must not appear anywhere else (a nonlinear occurrence blocks absorption).
    if terms.iter().any(|(t, _)| {
        let mut stack = vec![*t];
        while let Some(id) = stack.pop() {
            if id == c {
                return true;
            }
            push_children(ctx, id, &mut stack);
        }
        false
    }) {
        return None;
    }
    let mut rest: Option<ExprId> = None;
    for (t, r) in terms {
        let coef = ctx.add(cas_ast::Expr::Number(r));
        let scaled = ctx.add(cas_ast::Expr::Mul(coef, t));
        rest = Some(match rest {
            None => scaled,
            Some(acc) => ctx.add(cas_ast::Expr::Add(acc, scaled)),
        });
    }
    Some(rest.unwrap_or_else(|| ctx.num(0)))
}

/// Rewrite `e^(C + H)` (anywhere in the tree) as `C·e^H` — the textbook
/// constant-absorption for the ± branch pair of `ln|y| = ∫f + C` (D12).
fn absorb_exp_constant(ctx: &mut Context, root: ExprId, c: ExprId) -> Option<ExprId> {
    match ctx.get(root) {
        cas_ast::Expr::Pow(base, exp) => {
            let (base, exp) = (*base, *exp);
            let stripped = strip_additive_c(ctx, exp, c)?;
            let new_pow = ctx.add(cas_ast::Expr::Pow(base, stripped));
            Some(ctx.add(cas_ast::Expr::Mul(c, new_pow)))
        }
        cas_ast::Expr::Mul(a, b) => {
            let (a, b) = (*a, *b);
            if let Some(a2) = absorb_exp_constant(ctx, a, c) {
                return Some(ctx.add(cas_ast::Expr::Mul(a2, b)));
            }
            if let Some(b2) = absorb_exp_constant(ctx, b, c) {
                return Some(ctx.add(cas_ast::Expr::Mul(a, b2)));
            }
            None
        }
        cas_ast::Expr::Div(a, b) => {
            let (a, b) = (*a, *b);
            if let Some(a2) = absorb_exp_constant(ctx, a, c) {
                return Some(ctx.add(cas_ast::Expr::Div(a2, b)));
            }
            None
        }
        cas_ast::Expr::Neg(a) => {
            let a = *a;
            absorb_exp_constant(ctx, a, c)
        }
        _ => None,
    }
}

/// After a ± absorption, `C` legitimately swallows absolute values of
/// unknown-free arguments (`±e^C·|x| ≡ C·x` as C ranges over ℝ∖{0}). Strip
/// `abs(u)` calls with `func`-free arguments.
fn strip_free_abs(ctx: &mut Context, root: ExprId, func: &str) -> ExprId {
    let expr = ctx.get(root).clone();
    match expr {
        cas_ast::Expr::Function(fn_id, args) => {
            if fn_id == ctx.builtin_id(cas_ast::BuiltinFn::Abs) && args.len() == 1 {
                let vars = collect_variables(ctx, args[0]);
                if !vars.contains(func) {
                    return strip_free_abs(ctx, args[0], func);
                }
            }
            let new_args: Vec<ExprId> =
                args.iter().map(|a| strip_free_abs(ctx, *a, func)).collect();
            let name = ctx.sym_name(fn_id).to_string();
            ctx.call(&name, new_args)
        }
        cas_ast::Expr::Add(a, b) => {
            let (a2, b2) = (strip_free_abs(ctx, a, func), strip_free_abs(ctx, b, func));
            ctx.add(cas_ast::Expr::Add(a2, b2))
        }
        cas_ast::Expr::Sub(a, b) => {
            let (a2, b2) = (strip_free_abs(ctx, a, func), strip_free_abs(ctx, b, func));
            ctx.add(cas_ast::Expr::Sub(a2, b2))
        }
        cas_ast::Expr::Mul(a, b) => {
            let (a2, b2) = (strip_free_abs(ctx, a, func), strip_free_abs(ctx, b, func));
            ctx.add(cas_ast::Expr::Mul(a2, b2))
        }
        cas_ast::Expr::Div(a, b) => {
            let (a2, b2) = (strip_free_abs(ctx, a, func), strip_free_abs(ctx, b, func));
            ctx.add(cas_ast::Expr::Div(a2, b2))
        }
        cas_ast::Expr::Pow(a, b) => {
            let (a2, b2) = (strip_free_abs(ctx, a, func), strip_free_abs(ctx, b, func));
            ctx.add(cas_ast::Expr::Pow(a2, b2))
        }
        cas_ast::Expr::Neg(a) => {
            let a2 = strip_free_abs(ctx, a, func);
            ctx.add(cas_ast::Expr::Neg(a2))
        }
        _ => root,
    }
}

/// True when the tree contains a square root whose radicand involves `c` —
/// the D6 "surd over C" criterion that prefers the implicit form.
fn contains_surd_over_c(ctx: &Context, root: ExprId, c: ExprId) -> bool {
    let contains_c = |ctx: &Context, e: ExprId| -> bool {
        let mut stack = vec![e];
        while let Some(id) = stack.pop() {
            if id == c {
                return true;
            }
            push_children(ctx, id, &mut stack);
        }
        false
    };
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        match ctx.get(id) {
            cas_ast::Expr::Pow(base, exp) => {
                if let cas_ast::Expr::Number(n) = ctx.get(*exp) {
                    if !n.is_integer() && contains_c(ctx, *base) {
                        return true;
                    }
                }
                stack.push(*base);
                stack.push(*exp);
            }
            cas_ast::Expr::Function(fn_id, args) => {
                if *fn_id == ctx.builtin_id(cas_ast::BuiltinFn::Sqrt)
                    && args.len() == 1
                    && contains_c(ctx, args[0])
                {
                    return true;
                }
                stack.extend(args.iter().copied());
            }
            _ => push_children(ctx, id, &mut stack),
        }
    }
    false
}

/// Match the exact-equation shape `M(x,y) + N(x,y)·y' = 0` on the RAW tree:
/// additive terms of `LHS − RHS`; terms carrying `diff(func,var)` as a plain
/// numerator product factor contribute their cofactor to `N` (which MAY
/// involve the unknown — unlike the linear matcher), every other term goes to
/// `M`. Declines when the diff is nested, inverted, or appears in several
/// factors of one term.
fn try_extract_exact_form(
    ctx: &mut Context,
    eq: &Equation,
    diff_sym: cas_ast::symbol::SymbolId,
    func: &str,
    var: &str,
) -> Option<(ExprId, ExprId)> {
    let mut terms: Vec<(ExprId, bool)> = Vec::new();
    collect_signed_terms(ctx, eq.lhs, true, &mut terms);
    collect_signed_terms(ctx, eq.rhs, false, &mut terms);

    let mut m_parts: Vec<ExprId> = Vec::new();
    let mut n_parts: Vec<ExprId> = Vec::new();
    for (term, positive) in terms {
        let mut sign = positive;
        let mut core = term;
        while let cas_ast::Expr::Neg(inner) = ctx.get(core) {
            sign = !sign;
            core = *inner;
        }
        let mut factors: Factors = Vec::new();
        collect_product_factors(ctx, core, false, &mut factors);

        let mut diff_count = 0usize;
        let mut cof_num: Vec<ExprId> = Vec::new();
        let mut cof_den: Vec<ExprId> = Vec::new();
        for (f, is_den) in factors {
            let mut g = f;
            while let cas_ast::Expr::Neg(inner) = ctx.get(g) {
                sign = !sign;
                g = *inner;
            }
            if is_first_order_diff_call(ctx, g, diff_sym, func, var) {
                if is_den {
                    return None; // y' inverted: not the exact shape
                }
                diff_count += 1;
                continue;
            }
            // A diff nested anywhere deeper than a plain factor declines.
            if scan_diff_calls_of(ctx, g, diff_sym, func).is_some() {
                return None;
            }
            if is_den {
                cof_den.push(g)
            } else {
                cof_num.push(g)
            }
        }
        if diff_count > 1 {
            return None;
        }
        let build_product = |ctx: &mut Context, parts: &[ExprId]| -> Option<ExprId> {
            let mut it = parts.iter();
            let first = *it.next()?;
            Some(it.fold(first, |acc, &p| ctx.add(cas_ast::Expr::Mul(acc, p))))
        };
        let num = build_product(ctx, &cof_num).unwrap_or_else(|| ctx.num(1));
        let mut piece = match build_product(ctx, &cof_den) {
            Some(d) => ctx.add(cas_ast::Expr::Div(num, d)),
            None => num,
        };
        if !sign {
            piece = ctx.add(cas_ast::Expr::Neg(piece));
        }
        if diff_count == 1 {
            n_parts.push(piece);
        } else {
            m_parts.push(piece);
        }
    }
    if n_parts.is_empty() {
        return None;
    }
    let sum = |ctx: &mut Context, parts: &[ExprId]| -> Option<ExprId> {
        let mut it = parts.iter();
        let first = *it.next()?;
        Some(it.fold(first, |acc, &p| ctx.add(cas_ast::Expr::Add(acc, p))))
    };
    let n = sum(ctx, &n_parts).expect("nonempty");
    let m = sum(ctx, &m_parts).unwrap_or_else(|| ctx.num(0));
    Some((m, n))
}

/// `Some(numerator/f with one literal factor `f` removed)` when `f` appears as
/// an exact numerator product factor of `e`. Structural cancellation only —
/// used to peel `μ` out of `∫μq` before building the display quotient.
fn cancel_literal_factor(ctx: &mut Context, e: ExprId, f: ExprId) -> Option<ExprId> {
    if e == f {
        return Some(ctx.num(1));
    }
    let mut factors: Factors = Vec::new();
    collect_product_factors(ctx, e, false, &mut factors);
    let pos = factors.iter().position(|(g, is_den)| !is_den && *g == f)?;
    factors.remove(pos);
    let mut num: Vec<ExprId> = Vec::new();
    let mut den: Vec<ExprId> = Vec::new();
    for (g, is_den) in factors {
        if is_den {
            den.push(g)
        } else {
            num.push(g)
        }
    }
    let build_product = |ctx: &mut Context, parts: &[ExprId]| -> Option<ExprId> {
        let mut it = parts.iter();
        let first = *it.next()?;
        Some(it.fold(first, |acc, &p| ctx.add(cas_ast::Expr::Mul(acc, p))))
    };
    let n = build_product(ctx, &num).unwrap_or_else(|| ctx.num(1));
    Some(match build_product(ctx, &den) {
        Some(d) => ctx.add(cas_ast::Expr::Div(n, d)),
        None => n,
    })
}

/// One initial condition `y(point) = value` (order 0) or `y'(point) = value`
/// (order 1 — declines until O4), already parsed by the solver layer (the
/// engine never parses text — D1 keeps the heads out of cas_parser entirely).
type InitialCondition = cas_solver_core::eval_models::DsolveCondition;

/// Honest residual (D8): a re-emittable `dsolve(...)` echo plus the warning
/// naming why the command declined and which cycle owns the missing method.
fn residual_action_result(
    ctx: &mut Context,
    resolved: ExprId,
    y_var: ExprId,
    x_var: ExprId,
    reason: &str,
) -> ActionResult {
    let eco = ctx.call(DSOLVE_RULE, vec![resolved, y_var, x_var]);
    (
        EvalResult::SolutionSet(SolutionSet::Residual(eco)),
        vec![DomainWarning {
            message: reason.to_string(),
            rule_name: DSOLVE_RULE.to_string(),
        }],
        vec![],
        vec![],
        vec![],
        vec![],
        vec![],
        vec![],
    )
}

/// Normalize an implicit potential for display: strip a global rational
/// denominator (`(x²+y²)/2 = C ⇒ x²+y² = C`, the constant absorbs it).
fn clear_global_rational_factor(ctx: &mut Context, e: ExprId) -> ExprId {
    match ctx.get(e) {
        cas_ast::Expr::Div(n, d) => {
            let (n, d) = (*n, *d);
            if matches!(ctx.get(d), cas_ast::Expr::Number(_)) {
                return n;
            }
            e
        }
        cas_ast::Expr::Mul(a, b) => {
            let (a, b) = (*a, *b);
            if matches!(ctx.get(a), cas_ast::Expr::Number(_)) {
                return b;
            }
            if matches!(ctx.get(b), cas_ast::Expr::Number(_)) {
                return a;
            }
            e
        }
        _ => e,
    }
}

impl Engine {
    /// Handle `EvalAction::Dsolve` (Fase 4 · O0: separables).
    pub(super) fn eval_dsolve(
        &mut self,
        options: &crate::options::EvalOptions,
        resolved: ExprId,
        func: &str,
        var: &str,
        conditions: &[InitialCondition],
    ) -> Result<ActionResult, anyhow::Error> {
        let ctx = &mut self.simplifier.context;
        let y_var = ctx.var(func);
        let x_var = ctx.var(var);
        let diff_sym = ctx.intern_symbol("diff");

        // The wire guarantees a diff(func,·) exists textually; re-check on the
        // tree so envelope/JSON callers get the same honest contract. The
        // ORDER (not the arity) drives the dispatch: `diff(y,x,2)` and nested
        // `diff(diff(y,x),x)` are both order 2.
        let Some(max_diff_order) = scan_max_diff_order(ctx, resolved, diff_sym, func, var) else {
            return Err(anyhow::anyhow!(
                "dsolve: the equation contains no diff({func}, ...) — not an ODE in `{func}`"
            ));
        };

        let ode_eq = cas_solver_core::solve_entry::equation_from_expr_or_zero(ctx, resolved);
        if ode_eq.op != RelOp::Eq {
            return Err(anyhow::anyhow!(
                "dsolve: expected an equation (lhs = rhs), got a relation `{}`",
                ode_eq.op
            ));
        }

        // Honest residual scaffolding shared by every decline path (D8).
        let mk_residual = |ctx: &mut Context, reason: &str| -> ActionResult {
            residual_action_result(ctx, resolved, y_var, x_var, reason)
        };

        // Order ≥3 (or a symbolic order) stays an honest residual.
        if max_diff_order > 2 {
            let r = mk_residual(
                ctx,
                "EDO de orden ≥3 (o de orden simbólico): fuera del alcance elemental; se declina honesto",
            );
            return Ok(r);
        }

        // Initial conditions (O3/O4): already parsed by the solver layer.
        let parsed_conditions: Vec<InitialCondition> = conditions.to_vec();
        if parsed_conditions.iter().any(|c| c.order >= max_diff_order) {
            let r = mk_residual(
                ctx,
                "Una condición inicial solo puede fijar derivadas de orden MENOR que el de la EDO (y(x0) para 1er orden; y(x0)/y'(x0) para 2º)",
            );
            return Ok(r);
        }
        if parsed_conditions.len() > max_diff_order {
            let r = mk_residual(
                ctx,
                "Más condiciones iniciales que constantes libres (sobredeterminado); se declina honesto",
            );
            return Ok(r);
        }

        // O4: second order with constant coefficients; O9: Cauchy-Euler.
        if max_diff_order == 2 {
            if let Some(second) = try_match_second_order_constant(ctx, &ode_eq, diff_sym, func, var)
            {
                return self.eval_dsolve_second_order(
                    options,
                    resolved,
                    func,
                    var,
                    &ode_eq,
                    second,
                    &parsed_conditions,
                );
            }
            if let Some(ce) = try_match_cauchy_euler(ctx, &ode_eq, diff_sym, func, var) {
                return self.eval_dsolve_cauchy_euler(
                    options,
                    resolved,
                    func,
                    var,
                    &ode_eq,
                    ce,
                    &parsed_conditions,
                );
            }
            let r = mk_residual(
                &mut self.simplifier.context,
                "2º orden con coeficientes variables o forma no-lineal (Bessel/no-homogénea con RHS funcional: ciclos futuros); se declina honesto",
            );
            return Ok(r);
        }
        let initial_condition = parsed_conditions.into_iter().next();

        // Method dispatch: separable first (O0 — the S1-S9 pins stay
        // byte-identical), then linear first-order (O1), then exact (O2). A
        // form that fits none declines honestly naming the owner cycles (O8
        // Bernoulli/homogeneous). The general solution is computed first; an
        // initial condition then pins the constant (O3) over the SAME result.
        let isolated_rhs = match_isolated_first_order(ctx, &ode_eq, diff_sym, func, var);
        let split = isolated_rhs
            .and_then(|rhs| split_separable(ctx, rhs, func, var).map(|split| (rhs, split)));
        let Some((rhs, split)) = split else {
            // O1: linear first-order via integrating factor.
            if let Some(linear) = try_match_linear_first_order(ctx, &ode_eq, diff_sym, func, var) {
                let general =
                    self.eval_dsolve_linear(options, resolved, func, var, &ode_eq, linear)?;
                return Ok(self.maybe_apply_initial_condition(
                    options,
                    general,
                    initial_condition.as_ref(),
                    resolved,
                    func,
                    var,
                    &ode_eq,
                ));
            }
            // O2: exact equation M + N·y' = 0 via the F6 potential machinery.
            if let Some((m, n)) = try_extract_exact_form(ctx, &ode_eq, diff_sym, func, var) {
                if let Some(general) = self.eval_dsolve_exact(options, resolved, func, var, m, n)? {
                    return Ok(self.maybe_apply_initial_condition(
                        options,
                        general,
                        initial_condition.as_ref(),
                        resolved,
                        func,
                        var,
                        &ode_eq,
                    ));
                }
            }
            // O8a: Bernoulli y' + p·y = q·yⁿ.
            if let Some(bern) =
                try_match_bernoulli(&mut self.simplifier.context, &ode_eq, diff_sym, func, var)
            {
                let general =
                    self.eval_dsolve_bernoulli(options, resolved, func, var, &ode_eq, bern)?;
                return Ok(self.maybe_apply_initial_condition(
                    options,
                    general,
                    initial_condition.as_ref(),
                    resolved,
                    func,
                    var,
                    &ode_eq,
                ));
            }
            // O8b: homogeneous y' = F(y/x) on the isolated RHS.
            if let Some(iso_rhs) = isolated_rhs {
                let mut verify_options = options.clone();
                verify_options.steps_mode = cas_solver_core::eval_options::StepsMode::Off;
                if let Some((v_var, f_v)) =
                    try_match_homogeneous_rhs(self, &verify_options, iso_rhs, func, var)
                {
                    if let Some(general) = self.eval_dsolve_homogeneous(
                        options, resolved, func, var, &ode_eq, iso_rhs, v_var, f_v,
                    )? {
                        return Ok(self.maybe_apply_initial_condition(
                            options,
                            general,
                            initial_condition.as_ref(),
                            resolved,
                            func,
                            var,
                            &ode_eq,
                        ));
                    }
                }
            }
            let r = mk_residual(
                &mut self.simplifier.context,
                "La EDO de 1er orden no casa ningún método clásico (separable/lineal/exacta/Bernoulli/homogénea); Riccati y formas sin método clásico son residuales honestos permanentes",
            );
            return Ok(r);
        };

        // Integrate both sides: ∫ dy/g(y) = ∫ f(x) dx.
        let lhs_int = match split.gy {
            None => y_var,
            Some(gy) if is_literal_one(ctx, gy) => y_var,
            Some(gy) => {
                let one = ctx.num(1);
                let integrand = ctx.add(cas_ast::Expr::Div(one, gy));
                match crate::rules::calculus::integrate_with_trace(ctx, integrand, func) {
                    Some(outcome) if outcome.required_conditions.is_empty() => outcome.result,
                    _ => {
                        let r = mk_residual(
                            ctx,
                            "La integral ∫ dy/g(y) no cierra en forma elemental sin condiciones extra; se declina honesto",
                        );
                        return Ok(r);
                    }
                }
            }
        };
        let rhs_int = if is_literal_one(ctx, split.fx) {
            x_var
        } else {
            match crate::rules::calculus::integrate_with_trace(ctx, split.fx, var) {
                Some(outcome) if outcome.required_conditions.is_empty() => outcome.result,
                _ => {
                    let r = mk_residual(
                        ctx,
                        "La integral ∫ f(x) dx no cierra en forma elemental sin condiciones extra; se declina honesto",
                    );
                    return Ok(r);
                }
            }
        };

        // Arbitrary constant (D7): fresh when the input already uses `C`.
        let input_vars = collect_variables(ctx, resolved);
        let c_name = if input_vars.contains("C") { "K1" } else { "C" };
        let c_var = ctx.var(c_name);
        let mut warnings: Vec<DomainWarning> = Vec::new();
        if c_name != "C" {
            warnings.push(DomainWarning {
                message: "La entrada ya usa el nombre C; la constante arbitraria se emite como K1"
                    .to_string(),
                rule_name: DSOLVE_RULE.to_string(),
            });
        }

        // Singular solutions (D12): roots of g(y) = 0 are constant solutions
        // dropped when dividing by g.
        let mut singular_notes: Vec<String> = Vec::new();
        if let Some(gy) = split.gy {
            if !is_literal_one(ctx, gy) {
                let zero = ctx.num(0);
                let g_eq = Equation {
                    lhs: gy,
                    rhs: zero,
                    op: RelOp::Eq,
                };
                let solver_opts = cas_solver_core::solver_options::SolverOptions::from_eval_config(
                    options.shared.semantics,
                    options.budget,
                );
                if let Ok((SolutionSet::Discrete(roots), _, _)) =
                    crate::api::solve_with_display_steps(
                        &g_eq,
                        func,
                        &mut self.simplifier,
                        solver_opts,
                    )
                {
                    for r in roots {
                        singular_notes.push(format!(
                            "{func} = {} es solución singular (descartada al dividir por g({func}))",
                            render_expr(&self.simplifier.context, r)
                        ));
                    }
                }
            }
        }
        let ctx = &mut self.simplifier.context;

        // Integrated equation G(y) = F(x) + C.
        let rhs_with_c = ctx.add(cas_ast::Expr::Add(rhs_int, c_var));
        let integrated_eq = Equation {
            lhs: lhs_int,
            rhs: rhs_with_c,
            op: RelOp::Eq,
        };

        // Solve inverse (D6): try the explicit form first.
        let solver_opts = cas_solver_core::solver_options::SolverOptions::from_eval_config(
            options.shared.semantics,
            options.budget,
        );
        let solve_outcome = crate::api::solve_with_display_steps(
            &integrated_eq,
            func,
            &mut self.simplifier,
            solver_opts,
        );

        enum Emission {
            Explicit(Vec<ExprId>),
            Implicit,
        }
        let ctx_ref = &self.simplifier.context;
        let emission = match &solve_outcome {
            Ok((SolutionSet::Discrete(roots), _, _))
                if !roots.is_empty()
                    && roots.len() <= 2
                    && !roots
                        .iter()
                        .any(|r| contains_surd_over_c(ctx_ref, *r, c_var)) =>
            {
                Emission::Explicit(roots.clone())
            }
            _ => Emission::Implicit,
        };

        // Assemble candidates, absorb ± into C where legitimate (D12).
        let mut absorbed_pm = false;
        let candidates: Vec<ExprId> = match emission {
            Emission::Explicit(mut roots) => {
                if roots.len() == 2 {
                    let ctx = &mut self.simplifier.context;
                    let sum = ctx.add(cas_ast::Expr::Add(roots[0], roots[1]));
                    if self.reduces_to_zero_exact(options, sum) {
                        // ± pair: keep the branch without a leading Neg, absorb.
                        let ctx = &mut self.simplifier.context;
                        let principal = if matches!(ctx.get(roots[0]), cas_ast::Expr::Neg(_)) {
                            roots[1]
                        } else {
                            roots[0]
                        };
                        if let Some(absorbed) = absorb_exp_constant(ctx, principal, c_var) {
                            absorbed_pm = true;
                            roots = vec![absorbed];
                        }
                    }
                }
                roots
            }
            Emission::Implicit => Vec::new(),
        };

        // Build the final result form + its verification residue.
        let ctx = &mut self.simplifier.context;
        let ode_residue_raw = ctx.add(cas_ast::Expr::Sub(ode_eq.lhs, ode_eq.rhs));

        let mut verify_options = options.clone();
        verify_options.steps_mode = cas_solver_core::eval_options::StepsMode::Off;
        verify_options.time_budget_ms = Some(
            verify_options
                .time_budget_ms
                .map_or(VERIFY_TIME_BUDGET_MS, |t| t.min(VERIFY_TIME_BUDGET_MS)),
        );

        if !candidates.is_empty() {
            // Explicit path: verify EVERY branch by substitution before emitting.
            let mut verified: Vec<ExprId> = Vec::new();
            for cand in &candidates {
                let cand = if absorbed_pm {
                    // Canonize the hand-built absorbed tree through the full
                    // pipeline (F10 doctrine: no branch-hop forms), then let C
                    // swallow unknown-free absolute values (D12).
                    let folded = match self.eval_simplify(&verify_options, *cand) {
                        Ok((EvalResult::Expr(s), ..)) => s,
                        _ => *cand,
                    };
                    let ctx = &mut self.simplifier.context;
                    strip_free_abs(ctx, folded, func)
                } else {
                    *cand
                };
                let ctx = &mut self.simplifier.context;
                let substituted = substitute_power_aware(
                    ctx,
                    ode_residue_raw,
                    y_var,
                    cand,
                    SubstituteOptions::exact(),
                );
                if self.reduces_to_zero_exact(&verify_options, substituted) {
                    verified.push(cand);
                } else {
                    let ctx = &mut self.simplifier.context;
                    let r = mk_residual(
                        ctx,
                        "La candidata no verificó (el residuo LHS−RHS no se redujo a 0 exacto); se declina honesto en lugar de emitir sin red",
                    );
                    return Ok(r);
                }
            }

            let ctx = &mut self.simplifier.context;
            warnings.push(DomainWarning {
                message: format!("Solución general: {c_name} es una constante arbitraria"),
                rule_name: DSOLVE_RULE.to_string(),
            });
            if absorbed_pm {
                let mut msg =
                    format!("El doble signo ± de e^{c_name} se absorbe en {c_name} ({c_name} ≠ 0)");
                if let Some(first) = singular_notes.first() {
                    msg.push_str("; ");
                    msg.push_str(first);
                }
                warnings.push(DomainWarning {
                    message: msg,
                    rule_name: DSOLVE_RULE.to_string(),
                });
            } else {
                for note in &singular_notes {
                    warnings.push(DomainWarning {
                        message: note.clone(),
                        rule_name: DSOLVE_RULE.to_string(),
                    });
                }
            }

            let solve_steps = build_separable_steps(
                ctx,
                &ode_eq,
                split.fx,
                split.gy,
                &integrated_eq,
                y_var,
                &verified,
                None,
                c_var,
            );

            let result = if verified.len() == 1 {
                EvalResult::Expr(wrap_eq(ctx, y_var, verified[0]))
            } else {
                let eqs: Vec<ExprId> = verified.iter().map(|r| wrap_eq(ctx, y_var, *r)).collect();
                EvalResult::SolutionSet(SolutionSet::Discrete(eqs))
            };
            let general = (
                result,
                warnings,
                vec![],
                solve_steps,
                vec![],
                vec![],
                vec![],
                vec![],
            );
            return Ok(self.maybe_apply_initial_condition(
                options,
                general,
                initial_condition.as_ref(),
                resolved,
                func,
                var,
                &ode_eq,
            ));
        }

        // Implicit path: φ(x,y) = C, verified by implicit differentiation
        // (residue ∂φ/∂x + ∂φ/∂y·f reduces to 0 — D5).
        let phi_raw = ctx.add(cas_ast::Expr::Sub(lhs_int, rhs_int));
        let phi = match self.eval_simplify(&verify_options, phi_raw) {
            Ok((EvalResult::Expr(simplified), ..)) => simplified,
            _ => phi_raw,
        };
        let ctx = &mut self.simplifier.context;
        let phi = clear_global_rational_factor(ctx, phi);

        let dphi_dx = ctx.call("diff", vec![phi, x_var]);
        let dphi_dy = ctx.call("diff", vec![phi, y_var]);
        let dy_term = ctx.add(cas_ast::Expr::Mul(dphi_dy, rhs));
        let implicit_residue = ctx.add(cas_ast::Expr::Add(dphi_dx, dy_term));
        if !self.reduces_to_zero_exact(&verify_options, implicit_residue) {
            let ctx = &mut self.simplifier.context;
            let r = mk_residual(
                ctx,
                "La solución implícita no verificó por diferenciación implícita; se declina honesto",
            );
            return Ok(r);
        }

        let ctx = &mut self.simplifier.context;
        warnings.push(DomainWarning {
            message: format!(
                "Solución implícita: se emite φ({var},{func}) = {c_name} porque el despeje explícito no cierra limpio"
            ),
            rule_name: DSOLVE_RULE.to_string(),
        });
        warnings.push(DomainWarning {
            message: format!("Solución general: {c_name} es una constante arbitraria"),
            rule_name: DSOLVE_RULE.to_string(),
        });
        for note in &singular_notes {
            warnings.push(DomainWarning {
                message: note.clone(),
                rule_name: DSOLVE_RULE.to_string(),
            });
        }

        let solve_steps = build_separable_steps(
            ctx,
            &ode_eq,
            split.fx,
            split.gy,
            &integrated_eq,
            y_var,
            &[],
            Some(phi),
            c_var,
        );
        let result = EvalResult::Expr(wrap_eq(ctx, phi, c_var));
        let general = (
            result,
            warnings,
            vec![],
            solve_steps,
            vec![],
            vec![],
            vec![],
            vec![],
        );
        Ok(self.maybe_apply_initial_condition(
            options,
            general,
            initial_condition.as_ref(),
            resolved,
            func,
            var,
            &ode_eq,
        ))
    }

    /// True when the FULL evaluator reduces `e` to exactly `Number(0)`, with
    /// numeric verification disabled (the D5 ritual: a probe never confirms).
    /// Runs up to TWO passes as defense in depth: the root fixpoint gap that
    /// motivated this (`(1/e^0 + 0 - 1) - 0` stranded at `1/e^0 - 1` by the
    /// additive-pair shortcut) is CLOSED in the orchestrator (constant
    /// residuals always re-pass), but iterating the exact evaluator never
    /// changes semantics and the second pass only runs when the first fails.
    fn reduces_to_zero_exact(&mut self, options: &crate::options::EvalOptions, e: ExprId) -> bool {
        let saved_numeric = self.simplifier.allow_numerical_verification;
        self.simplifier.allow_numerical_verification = false;
        let mut reduced = false;
        let mut current = e;
        for _pass in 0..2 {
            match self.eval_simplify(options, current) {
                Ok((EvalResult::Expr(result), ..)) => {
                    if matches!(self.simplifier.context.get(result), cas_ast::Expr::Number(n) if n.is_zero())
                    {
                        reduced = true;
                        break;
                    }
                    if result == current {
                        break; // true fixpoint, nonzero
                    }
                    current = result;
                }
                _ => break,
            }
        }
        self.simplifier.allow_numerical_verification = saved_numeric;
        reduced
    }

    /// Full simplify of a y-free subtree (coefficients, μ, candidates): the
    /// D4 invariant allows the pipeline only on trees without `diff(y,·)`.
    fn fold_free_subtree(&mut self, options: &crate::options::EvalOptions, e: ExprId) -> ExprId {
        match self.eval_simplify(options, e) {
            Ok((EvalResult::Expr(simplified), ..)) => simplified,
            _ => e,
        }
    }

    /// O1: linear first-order `y' + p·y = q` via the integrating factor
    /// μ = e^(∫p dx); emission stays verification-gated (D5).
    /// Core of the integrating-factor method: μ = e^(∫p dx) (abs stripped —
    /// D12), candidate = ∫μq/μ + C/μ in the SPLIT per-term-folded form (the
    /// O1 lesson: the joined quotient defeats the verifier and the full fold
    /// re-joins it). Shared by the linear path (O1) and the Bernoulli
    /// reduction (O8). `Err` = the honest-decline reason.
    fn linear_general_candidate(
        &mut self,
        verify_options: &crate::options::EvalOptions,
        p: ExprId,
        q: ExprId,
        func: &str,
        var: &str,
        c_var: ExprId,
    ) -> Result<(ExprId, ExprId, ExprId), &'static str> {
        let ctx = &mut self.simplifier.context;
        let p_int = if matches!(ctx.get(p), cas_ast::Expr::Number(n) if n.is_zero()) {
            ctx.num(0)
        } else {
            match crate::rules::calculus::integrate_with_trace(ctx, p, var) {
                Some(outcome) if outcome.required_conditions.is_empty() => outcome.result,
                _ => return Err(
                    "La integral ∫ p(x) dx del factor integrante no cierra en forma elemental; se declina honesto",
                ),
            }
        };
        let e_const = ctx.add(cas_ast::Expr::Constant(cas_ast::Constant::E));
        let mu_raw = ctx.add(cas_ast::Expr::Pow(e_const, p_int));
        let mu_folded = self.fold_free_subtree(verify_options, mu_raw);
        // D12 (pin L9): μ = e^(ln|x|) folds to |x|; any functional μ works, so
        // presentation takes the textbook μ = x (strip unknown-free abs).
        let ctx = &mut self.simplifier.context;
        let mu = strip_free_abs(ctx, mu_folded, func);

        let mu_q_raw = ctx.add(cas_ast::Expr::Mul(mu, q));
        let mu_q = self.fold_free_subtree(verify_options, mu_q_raw);
        let ctx = &mut self.simplifier.context;
        let g_int = if matches!(ctx.get(mu_q), cas_ast::Expr::Number(n) if n.is_zero()) {
            ctx.num(0)
        } else {
            match crate::rules::calculus::integrate_with_trace(ctx, mu_q, var) {
                Some(outcome) if outcome.required_conditions.is_empty() => outcome.result,
                _ => {
                    return Err(
                        "La integral ∫ μ·q dx no cierra en forma elemental; se declina honesto",
                    )
                }
            }
        };
        // D12 (same doctrine as μ): ∫μq may carry `ln|x|` — textbook display
        // takes ln(x); the exact verification gate remains the judge.
        let ctx = &mut self.simplifier.context;
        let g_int = strip_free_abs(ctx, g_int, func);
        let ctx = &mut self.simplifier.context;

        let part_g = match cancel_literal_factor(ctx, g_int, mu) {
            Some(cancelled) => cancelled,
            None => ctx.add(cas_ast::Expr::Div(g_int, mu)),
        };
        let part_c = ctx.add(cas_ast::Expr::Div(c_var, mu));
        let part_g = self.fold_free_subtree(verify_options, part_g);
        let part_c = self.fold_free_subtree(verify_options, part_c);
        let ctx = &mut self.simplifier.context;
        let candidate = if matches!(ctx.get(part_g), cas_ast::Expr::Number(n) if n.is_zero()) {
            part_c
        } else {
            ctx.add(cas_ast::Expr::Add(part_g, part_c))
        };
        Ok((mu, g_int, candidate))
    }

    fn eval_dsolve_linear(
        &mut self,
        options: &crate::options::EvalOptions,
        resolved: ExprId,
        func: &str,
        var: &str,
        ode_eq: &Equation,
        linear: LinearOde,
    ) -> Result<ActionResult, anyhow::Error> {
        let mut verify_options = options.clone();
        verify_options.steps_mode = cas_solver_core::eval_options::StepsMode::Off;
        verify_options.time_budget_ms = Some(
            verify_options
                .time_budget_ms
                .map_or(VERIFY_TIME_BUDGET_MS, |t| t.min(VERIFY_TIME_BUDGET_MS)),
        );

        // Normalize p, q (y-free subtrees — safe to fold under D4).
        let p = self.fold_free_subtree(&verify_options, linear.p);
        let q = self.fold_free_subtree(&verify_options, linear.q);
        let ctx = &mut self.simplifier.context;
        let y_var = ctx.var(func);
        let x_var = ctx.var(var);

        // Arbitrary constant (D7).
        let input_vars = collect_variables(ctx, resolved);
        let c_name = if input_vars.contains("C") { "K1" } else { "C" };
        let c_var = ctx.var(c_name);
        let mut warnings: Vec<DomainWarning> = Vec::new();
        if c_name != "C" {
            warnings.push(DomainWarning {
                message: "La entrada ya usa el nombre C; la constante arbitraria se emite como K1"
                    .to_string(),
                rule_name: DSOLVE_RULE.to_string(),
            });
        }
        let (mu, g_int, candidate) =
            match self.linear_general_candidate(&verify_options, p, q, func, var, c_var) {
                Ok(core) => core,
                Err(reason) => {
                    let ctx = &mut self.simplifier.context;
                    return Ok(residual_action_result(ctx, resolved, y_var, x_var, reason));
                }
            };

        // D5: substitute into the RAW ODE; the residue must reduce to exact 0.
        let ctx = &mut self.simplifier.context;
        let ode_residue_raw = ctx.add(cas_ast::Expr::Sub(ode_eq.lhs, ode_eq.rhs));
        let substituted = substitute_power_aware(
            ctx,
            ode_residue_raw,
            y_var,
            candidate,
            SubstituteOptions::exact(),
        );
        if !self.reduces_to_zero_exact(&verify_options, substituted) {
            let ctx = &mut self.simplifier.context;
            let r = residual_action_result(
                ctx,
                resolved,
                y_var,
                x_var,
                "La candidata no verificó (el residuo LHS−RHS no se redujo a 0 exacto); se declina honesto en lugar de emitir sin red",
            );
            return Ok(r);
        }

        let ctx = &mut self.simplifier.context;
        warnings.push(DomainWarning {
            message: format!("Solución general: {c_name} es una constante arbitraria"),
            rule_name: DSOLVE_RULE.to_string(),
        });

        let solve_steps =
            build_linear_steps(ctx, ode_eq, p, q, mu, g_int, y_var, x_var, candidate, c_var);
        let result = EvalResult::Expr(wrap_eq(ctx, y_var, candidate));
        Ok((
            result,
            warnings,
            vec![],
            solve_steps,
            vec![],
            vec![],
            vec![],
            vec![],
        ))
    }

    /// O2: exact equation `M + N·y' = 0` — the F6 potential machinery IS the
    /// heart. Level 1 delegates to `try_potential_expr` (poly_eq-verified
    /// reconstruction); level 2 (D11) re-runs the same reconstruction in the
    /// caller with the FULL evaluator folding the pieces, so transcendental
    /// exact fields (`e^y + (x·e^y+2y)·y' = 0`) graduate too. Emission is
    /// gated per component (D5): `∂φ/∂x − M → 0` AND `∂φ/∂y − N → 0` under
    /// `reduces_to_zero_exact` — a non-conservative field can never verify,
    /// so exactness detection is free. `Ok(None)` = this path declines (the
    /// dispatcher falls to the honest residual).
    fn eval_dsolve_exact(
        &mut self,
        options: &crate::options::EvalOptions,
        resolved: ExprId,
        func: &str,
        var: &str,
        m: ExprId,
        n: ExprId,
    ) -> Result<Option<ActionResult>, anyhow::Error> {
        let mut verify_options = options.clone();
        verify_options.steps_mode = cas_solver_core::eval_options::StepsMode::Off;
        verify_options.time_budget_ms = Some(
            verify_options
                .time_budget_ms
                .map_or(VERIFY_TIME_BUDGET_MS, |t| t.min(VERIFY_TIME_BUDGET_MS)),
        );

        let ctx = &mut self.simplifier.context;
        let y_var = ctx.var(func);
        let x_var = ctx.var(var);

        // Level 1: the F6 reconstruction with its exact poly_eq gate.
        let vars_spec = vec![var.to_string(), func.to_string()];
        let phi =
            crate::rules::calculus::try_potential_expr(ctx, &[m, n], &vars_spec).or_else(|| {
                // Level 2 (D11): same reconstruction, FULL-evaluator folds.
                self.reconstruct_potential_full_eval(&verify_options, m, n, func, var)
            });
        let Some(phi) = phi else {
            return Ok(None);
        };

        // Per-component verification gate (D5) — for BOTH levels.
        let ctx = &mut self.simplifier.context;
        let dphi_dx = ctx.call("diff", vec![phi, x_var]);
        let res_x = ctx.add(cas_ast::Expr::Sub(dphi_dx, m));
        if !self.reduces_to_zero_exact(&verify_options, res_x) {
            return Ok(None);
        }
        let ctx = &mut self.simplifier.context;
        let dphi_dy = ctx.call("diff", vec![phi, y_var]);
        let res_y = ctx.add(cas_ast::Expr::Sub(dphi_dy, n));
        if !self.reduces_to_zero_exact(&verify_options, res_y) {
            return Ok(None);
        }

        // Display normalization + arbitrary constant (D7).
        let phi = self.fold_free_subtree(&verify_options, phi);
        let ctx = &mut self.simplifier.context;
        let phi = clear_global_rational_factor(ctx, phi);
        let input_vars = collect_variables(ctx, resolved);
        let c_name = if input_vars.contains("C") { "K1" } else { "C" };
        let c_var = ctx.var(c_name);
        let mut warnings: Vec<DomainWarning> = Vec::new();
        if c_name != "C" {
            warnings.push(DomainWarning {
                message: "La entrada ya usa el nombre C; la constante arbitraria se emite como K1"
                    .to_string(),
                rule_name: DSOLVE_RULE.to_string(),
            });
        }
        warnings.push(DomainWarning {
            message: format!(
                "Solución implícita: se emite φ({var},{func}) = {c_name} porque la EDO es exacta (φ es el potencial del campo (M, N))"
            ),
            rule_name: DSOLVE_RULE.to_string(),
        });
        warnings.push(DomainWarning {
            message: format!("Solución general: {c_name} es una constante arbitraria"),
            rule_name: DSOLVE_RULE.to_string(),
        });

        let solve_steps = build_exact_steps(ctx, m, n, phi, y_var, x_var, c_var);
        let result = EvalResult::Expr(wrap_eq(ctx, phi, c_var));
        Ok(Some((
            result,
            warnings,
            vec![],
            solve_steps,
            vec![],
            vec![],
            vec![],
            vec![],
        )))
    }

    /// O4: second-order homogeneous with constant coefficients — the
    /// characteristic equation is solved by an INTERNAL exact discriminant
    /// (D9: never a bare `solve`, whose solution SET collapses multiplicity
    /// and whose complex roots depend on the session value-domain). The three
    /// branches assemble the textbook basis; emission is gated by LINEARITY
    /// (D5): each basis function verifies against the ODE separately — the
    /// C1/C2-attached combination is NEVER substituted (the known O23
    /// expand↔factor hang lives exactly there).
    #[allow(clippy::too_many_arguments)]
    fn eval_dsolve_second_order(
        &mut self,
        options: &crate::options::EvalOptions,
        resolved: ExprId,
        func: &str,
        var: &str,
        ode_eq: &Equation,
        second: SecondOrderOde,
        conditions: &[InitialCondition],
    ) -> Result<ActionResult, anyhow::Error> {
        let mut verify_options = options.clone();
        verify_options.steps_mode = cas_solver_core::eval_options::StepsMode::Off;
        verify_options.time_budget_ms = Some(
            verify_options
                .time_budget_ms
                .map_or(VERIFY_TIME_BUDGET_MS, |t| t.min(VERIFY_TIME_BUDGET_MS)),
        );

        let ctx = &mut self.simplifier.context;
        let x_var = ctx.var(var);

        // Normalized characteristic r² + p·r + q = 0 and its exact discriminant.
        let p = &second.b / &second.a;
        let q = &second.c / &second.a;
        let disc = &p * &p - num_rational::BigRational::from_integer(4.into()) * &q;

        let two = num_rational::BigRational::from_integer(2.into());
        let e_const = ctx.add(cas_ast::Expr::Constant(cas_ast::Constant::E));

        // exp(r·x) with a RATIONAL rate, folded per-basis for clean display.
        fn exp_rate(
            ctx: &mut Context,
            e_const: ExprId,
            x_var: ExprId,
            rate: &num_rational::BigRational,
        ) -> ExprId {
            let r_num = ctx.add(cas_ast::Expr::Number(rate.clone()));
            let arg = ctx.add(cas_ast::Expr::Mul(r_num, x_var));
            ctx.add(cas_ast::Expr::Pow(e_const, arg))
        }

        enum Basis {
            DistinctReal {
                u1: ExprId,
                u2: ExprId,
            },
            Double {
                rate: num_rational::BigRational,
                u1: ExprId,
                u2: ExprId,
            },
            Complex {
                alpha: num_rational::BigRational,
                u1: ExprId,
                u2: ExprId,
                trig_cos: ExprId,
                trig_sin: ExprId,
            },
        }

        let basis = if disc.is_positive() {
            let (r1, r2) = match cas_math::perfect_square_support::rational_sqrt(&disc) {
                Some(s) => ((-&p + &s) / &two, (-&p - &s) / &two),
                None => {
                    // Surd roots r = (−p ± √disc)/2: build the exact trees.
                    let d_num = ctx.add(cas_ast::Expr::Number(disc.clone()));
                    let sqrt_d = ctx.call("sqrt", vec![d_num]);
                    let neg_p = ctx.add(cas_ast::Expr::Number(-&p));
                    let two_num = ctx.add(cas_ast::Expr::Number(two.clone()));
                    let sum1 = ctx.add(cas_ast::Expr::Add(neg_p, sqrt_d));
                    let r1e = ctx.add(cas_ast::Expr::Div(sum1, two_num));
                    let diff1 = ctx.add(cas_ast::Expr::Sub(neg_p, sqrt_d));
                    let r2e = ctx.add(cas_ast::Expr::Div(diff1, two_num));
                    let a1 = ctx.add(cas_ast::Expr::Mul(r1e, x_var));
                    let u1 = ctx.add(cas_ast::Expr::Pow(e_const, a1));
                    let a2 = ctx.add(cas_ast::Expr::Mul(r2e, x_var));
                    let u2 = ctx.add(cas_ast::Expr::Pow(e_const, a2));
                    let u1 = self.fold_free_subtree(&verify_options, u1);
                    let u2 = self.fold_free_subtree(&verify_options, u2);
                    return self.finish_second_order(
                        options,
                        &verify_options,
                        resolved,
                        func,
                        var,
                        ode_eq,
                        &second,
                        &disc,
                        SecondOrderShape::DistinctReal,
                        u1,
                        u2,
                        None,
                        conditions,
                    );
                }
            };
            let u1_raw = exp_rate(ctx, e_const, x_var, &r1);
            let u2_raw = exp_rate(ctx, e_const, x_var, &r2);
            let u1 = self.fold_free_subtree(&verify_options, u1_raw);
            let u2 = self.fold_free_subtree(&verify_options, u2_raw);
            Basis::DistinctReal { u1, u2 }
        } else if disc.is_zero() {
            let rate = -&p / &two;
            let ctx = &mut self.simplifier.context;
            let u1_raw = exp_rate(ctx, e_const, x_var, &rate);
            let u1 = self.fold_free_subtree(&verify_options, u1_raw);
            let ctx = &mut self.simplifier.context;
            let u2_raw = ctx.add(cas_ast::Expr::Mul(x_var, u1));
            let u2 = self.fold_free_subtree(&verify_options, u2_raw);
            Basis::Double { rate, u1, u2 }
        } else {
            let alpha = -&p / &two;
            let neg_disc = -&disc;
            let ctx = &mut self.simplifier.context;
            let beta_arg = match cas_math::perfect_square_support::rational_sqrt(&neg_disc) {
                Some(s) => {
                    let beta = &s / &two;
                    let b_num = ctx.add(cas_ast::Expr::Number(beta));
                    ctx.add(cas_ast::Expr::Mul(b_num, x_var))
                }
                None => {
                    let d_num = ctx.add(cas_ast::Expr::Number(neg_disc));
                    let sqrt_d = ctx.call("sqrt", vec![d_num]);
                    let two_num = ctx.add(cas_ast::Expr::Number(two.clone()));
                    let beta = ctx.add(cas_ast::Expr::Div(sqrt_d, two_num));
                    ctx.add(cas_ast::Expr::Mul(beta, x_var))
                }
            };
            let trig_cos_raw = ctx.call("cos", vec![beta_arg]);
            let trig_sin_raw = ctx.call("sin", vec![beta_arg]);
            let trig_cos = self.fold_free_subtree(&verify_options, trig_cos_raw);
            let trig_sin = self.fold_free_subtree(&verify_options, trig_sin_raw);
            let ctx = &mut self.simplifier.context;
            let (u1, u2) = if alpha.is_zero() {
                (trig_cos, trig_sin)
            } else {
                let envelope = exp_rate(ctx, e_const, x_var, &alpha);
                let u1 = ctx.add(cas_ast::Expr::Mul(envelope, trig_cos));
                let u2 = ctx.add(cas_ast::Expr::Mul(envelope, trig_sin));
                (u1, u2)
            };
            Basis::Complex {
                alpha,
                u1,
                u2,
                trig_cos,
                trig_sin,
            }
        };

        match basis {
            Basis::DistinctReal { u1, u2 } => self.finish_second_order(
                options,
                &verify_options,
                resolved,
                func,
                var,
                ode_eq,
                &second,
                &disc,
                SecondOrderShape::DistinctReal,
                u1,
                u2,
                None,
                conditions,
            ),
            Basis::Double { rate, u1, u2 } => self.finish_second_order(
                options,
                &verify_options,
                resolved,
                func,
                var,
                ode_eq,
                &second,
                &disc,
                SecondOrderShape::DoubleRoot { rate },
                u1,
                u2,
                None,
                conditions,
            ),
            Basis::Complex {
                alpha,
                u1,
                u2,
                trig_cos,
                trig_sin,
            } => self.finish_second_order(
                options,
                &verify_options,
                resolved,
                func,
                var,
                ode_eq,
                &second,
                &disc,
                SecondOrderShape::ComplexPair { alpha },
                u1,
                u2,
                Some((trig_cos, trig_sin)),
                conditions,
            ),
        }
    }

    /// Verification, assembly, narration, and (optional) IVP application for
    /// the second-order path. Every basis function must reduce the ODE residue
    /// to exact 0 before anything is emitted.
    #[allow(clippy::too_many_arguments)]
    fn finish_second_order(
        &mut self,
        options: &crate::options::EvalOptions,
        verify_options: &crate::options::EvalOptions,
        resolved: ExprId,
        func: &str,
        var: &str,
        ode_eq: &Equation,
        second: &SecondOrderOde,
        disc: &num_rational::BigRational,
        shape: SecondOrderShape,
        u1: ExprId,
        u2: ExprId,
        trig_parts: Option<(ExprId, ExprId)>,
        conditions: &[InitialCondition],
    ) -> Result<ActionResult, anyhow::Error> {
        let ctx = &mut self.simplifier.context;
        let y_var = ctx.var(func);
        let x_var = ctx.var(var);
        let _ = ode_eq;

        // Canonical residue of the ASSOCIATED HOMOGENEOUS operator L[y] =
        // a·y'' + b·y' + c·y, built from the exact coefficients: the per-basis
        // gate must check L[u_i] ≡ 0 — substituting into the user's full
        // equation would wrongly include the forcing RHS (D5: bases verify
        // against the homogeneous; y_p verifies against the complete).
        let a_num = ctx.add(cas_ast::Expr::Number(second.a.clone()));
        let b_num = ctx.add(cas_ast::Expr::Number(second.b.clone()));
        let c_num = ctx.add(cas_ast::Expr::Number(second.c.clone()));
        let two_lit = ctx.num(2);
        let d2_call = ctx.call("diff", vec![y_var, x_var, two_lit]);
        let d1_call = ctx.call("diff", vec![y_var, x_var]);
        let t2 = ctx.add(cas_ast::Expr::Mul(a_num, d2_call));
        let t1 = ctx.add(cas_ast::Expr::Mul(b_num, d1_call));
        let t0 = ctx.add(cas_ast::Expr::Mul(c_num, y_var));
        let sum21 = ctx.add(cas_ast::Expr::Add(t2, t1));
        let homog_residue = ctx.add(cas_ast::Expr::Add(sum21, t0));

        // D5 linearity gate: each basis function alone must annihilate L.
        for u in [u1, u2] {
            let ctx = &mut self.simplifier.context;
            let substituted =
                substitute_power_aware(ctx, homog_residue, y_var, u, SubstituteOptions::exact());
            if !self.reduces_to_zero_exact(verify_options, substituted) {
                let ctx = &mut self.simplifier.context;
                return Ok(residual_action_result(
                    ctx,
                    resolved,
                    y_var,
                    x_var,
                    "Una función de la base no verificó contra la homogénea asociada (residuo ≠ 0 exacto); se declina honesto",
                ));
            }
        }

        // Fresh constants (D7): C1/C2, or K1/K2 when the input already uses them.
        let ctx = &mut self.simplifier.context;
        let input_vars = collect_variables(ctx, resolved);
        let (c1_name, c2_name) = if input_vars.contains("C1") || input_vars.contains("C2") {
            ("K1", "K2")
        } else {
            ("C1", "C2")
        };
        let c1 = ctx.var(c1_name);
        let c2 = ctx.var(c2_name);

        // Textbook display form per branch. A reciprocal basis (`e^(-x)`
        // canonizes to `1/e^x`) multiplies as a clean quotient — `C2/e^x`,
        // never `(1·C2)/e^x` (structural, no fold: folding the product would
        // distribute and break the textbook shape).
        let mul_basis = |ctx: &mut Context, coef: ExprId, u: ExprId| -> ExprId {
            if let cas_ast::Expr::Div(num, den) = ctx.get(u) {
                let (num, den) = (*num, *den);
                if is_literal_one(ctx, num) {
                    return ctx.add(cas_ast::Expr::Div(coef, den));
                }
            }
            ctx.add(cas_ast::Expr::Mul(coef, u))
        };
        let general = match &shape {
            SecondOrderShape::DistinctReal => {
                let t1 = mul_basis(ctx, c1, u1);
                let t2 = mul_basis(ctx, c2, u2);
                ctx.add(cas_ast::Expr::Add(t1, t2))
            }
            SecondOrderShape::DoubleRoot { rate } => {
                let c2x = ctx.add(cas_ast::Expr::Mul(c2, x_var));
                let head = ctx.add(cas_ast::Expr::Add(c1, c2x));
                if rate.is_zero() {
                    head
                } else {
                    mul_basis(ctx, head, u1)
                }
            }
            SecondOrderShape::ComplexPair { alpha } => {
                let (trig_cos, trig_sin) = trig_parts.expect("complex branch carries trig parts");
                let t1 = ctx.add(cas_ast::Expr::Mul(c1, trig_sin));
                let t2 = ctx.add(cas_ast::Expr::Mul(c2, trig_cos));
                let combo = ctx.add(cas_ast::Expr::Add(t1, t2));
                if alpha.is_zero() {
                    combo
                } else {
                    let e_const = ctx.add(cas_ast::Expr::Constant(cas_ast::Constant::E));
                    let a_num = ctx.add(cas_ast::Expr::Number(alpha.clone()));
                    let arg = ctx.add(cas_ast::Expr::Mul(a_num, x_var));
                    let envelope = ctx.add(cas_ast::Expr::Pow(e_const, arg));
                    ctx.add(cas_ast::Expr::Mul(envelope, combo))
                }
            }
        };

        // O5: non-homogeneous forcing → particular solution by undetermined
        // coefficients, appended to the homogeneous general (superposition).
        let y_particular = match second.rhs {
            None => None,
            Some(rhs_expr) => {
                match self.uc_particular(verify_options, second, rhs_expr, func, var, resolved) {
                    Some(y_p) => Some(y_p),
                    None => {
                        let ctx = &mut self.simplifier.context;
                        return Ok(residual_action_result(
                            ctx,
                            resolved,
                            y_var,
                            x_var,
                            "RHS fuera de la tabla UC (polinomio·e^(kx)·sin/cos con coeficientes racionales): variación de parámetros queda fuera de fase; se declina honesto",
                        ));
                    }
                }
            }
        };
        let ctx = &mut self.simplifier.context;
        let general = match y_particular {
            None => general,
            Some(y_p) => ctx.add(cas_ast::Expr::Add(y_p, general)),
        };

        let mut warnings: Vec<DomainWarning> = Vec::new();
        if c1_name != "C1" {
            warnings.push(DomainWarning {
                message:
                    "La entrada ya usa C1/C2; las constantes arbitrarias se emiten como K1, K2"
                        .to_string(),
                rule_name: DSOLVE_RULE.to_string(),
            });
        }
        warnings.push(DomainWarning {
            message: format!("Solución general: {c1_name} y {c2_name} son constantes arbitrarias"),
            rule_name: DSOLVE_RULE.to_string(),
        });

        let mut solve_steps =
            build_second_order_steps(ctx, func, var, second, disc, &shape, u1, u2, y_var, general);
        if let Some(y_p) = y_particular {
            solve_steps.push(crate::api::SolveStep {
                description:
                    "Proponer y resolver la solución particular por coeficientes indeterminados (tabla UC con corrección de resonancia x^s)"
                        .to_string(),
                equation_after: Equation {
                    lhs: y_var,
                    rhs: y_p,
                    op: RelOp::Eq,
                },
                importance: ImportanceLevel::High,
                substeps: vec![],
            });
        }

        // No conditions: emit the general solution.
        if conditions.is_empty() {
            let ctx = &mut self.simplifier.context;
            let result = EvalResult::Expr(wrap_eq(ctx, y_var, general));
            return Ok((
                result,
                warnings,
                vec![],
                solve_steps,
                vec![],
                vec![],
                vec![],
                vec![],
            ));
        }
        if conditions.len() != 2 {
            let ctx = &mut self.simplifier.context;
            return Ok(residual_action_result(
                ctx,
                resolved,
                y_var,
                x_var,
                "Un IVP de 2º orden necesita DOS condiciones (y(x0)=y0 y y'(x0)=v0) para fijar C1 y C2; se declina honesto",
            ));
        }

        // Build one linear equation in (C1, C2) per condition FROM THE BASES:
        // y = C1·b1 + C2·b2 ⇒ y^(k)(x0) = C1·b1^(k)(x0) + C2·b2^(k)(x0). The
        // general WITH attached constants is never differentiated nor folded —
        // that is exactly the constant×exp×trig oscillation family (C5) the
        // linearity doctrine (D5) exists to dodge; each bare basis folds fast.
        let (b1, b2) = match &shape {
            SecondOrderShape::DistinctReal | SecondOrderShape::DoubleRoot { .. } => (u1, u2),
            // Complex display is C1·(e^{αx}·sin) + C2·(e^{αx}·cos).
            SecondOrderShape::ComplexPair { .. } => (u2, u1),
        };
        let mut cond_equations: Vec<Equation> = Vec::new();
        for cond in conditions {
            let (f1, f2) = if cond.order == 0 {
                (b1, b2)
            } else {
                let ctx = &mut self.simplifier.context;
                let d1_call = ctx.call("diff", vec![b1, x_var]);
                let d1 = self.fold_free_subtree(verify_options, d1_call);
                let ctx = &mut self.simplifier.context;
                let d2_call = ctx.call("diff", vec![b2, x_var]);
                let d2 = self.fold_free_subtree(verify_options, d2_call);
                (d1, d2)
            };
            let ctx = &mut self.simplifier.context;
            let f1_at =
                substitute_power_aware(ctx, f1, x_var, cond.point, SubstituteOptions::exact());
            let a1 = self.fold_free_subtree(verify_options, f1_at);
            let ctx = &mut self.simplifier.context;
            let f2_at =
                substitute_power_aware(ctx, f2, x_var, cond.point, SubstituteOptions::exact());
            let a2 = self.fold_free_subtree(verify_options, f2_at);
            let ctx = &mut self.simplifier.context;
            let t1 = ctx.add(cas_ast::Expr::Mul(c1, a1));
            let t2 = ctx.add(cas_ast::Expr::Mul(c2, a2));
            let mut lhs = ctx.add(cas_ast::Expr::Add(t1, t2));
            // Non-homogeneous: the particular contributes y_p^(k)(x0) as a
            // plain constant term (y_p is fresh-free and numeric-coefficient,
            // so its derivative and point-fold are C5-safe).
            if let Some(y_p) = y_particular {
                let f = if cond.order == 0 {
                    y_p
                } else {
                    let ctx = &mut self.simplifier.context;
                    let d_call = ctx.call("diff", vec![y_p, x_var]);
                    self.fold_free_subtree(verify_options, d_call)
                };
                let ctx = &mut self.simplifier.context;
                let at =
                    substitute_power_aware(ctx, f, x_var, cond.point, SubstituteOptions::exact());
                let at_folded = self.fold_free_subtree(verify_options, at);
                let ctx = &mut self.simplifier.context;
                lhs = ctx.add(cas_ast::Expr::Add(lhs, at_folded));
            }
            let ctx = &mut self.simplifier.context;
            let _ = ctx;
            cond_equations.push(Equation {
                lhs,
                rhs: cond.value,
                op: RelOp::Eq,
            });
        }

        // Solve the 2×2 by univariate elimination: C1 from eq1 (C2 symbolic),
        // substitute into eq2, solve C2, back-substitute.
        let solver_opts = cas_solver_core::solver_options::SolverOptions::from_eval_config(
            options.shared.semantics,
            options.budget,
        );
        let mut solved: Option<(ExprId, ExprId)> = None;
        'orders: for (first_name, second_name, first_var, second_var) in
            [(c1_name, c2_name, c1, c2), (c2_name, c1_name, c2, c1)]
        {
            let Ok((SolutionSet::Discrete(first_roots), _, _)) =
                crate::api::solve_with_display_steps(
                    &cond_equations[0],
                    first_name,
                    &mut self.simplifier,
                    solver_opts,
                )
            else {
                continue;
            };
            for first_expr in first_roots {
                let ctx = &mut self.simplifier.context;
                let eq2_lhs = substitute_power_aware(
                    ctx,
                    cond_equations[1].lhs,
                    first_var,
                    first_expr,
                    SubstituteOptions::exact(),
                );
                let eq2 = Equation {
                    lhs: eq2_lhs,
                    rhs: cond_equations[1].rhs,
                    op: RelOp::Eq,
                };
                let Ok((SolutionSet::Discrete(second_roots), _, _)) =
                    crate::api::solve_with_display_steps(
                        &eq2,
                        second_name,
                        &mut self.simplifier,
                        solver_opts,
                    )
                else {
                    continue;
                };
                if let Some(second_expr) = second_roots.into_iter().next() {
                    let ctx = &mut self.simplifier.context;
                    let first_back = substitute_power_aware(
                        ctx,
                        first_expr,
                        second_var,
                        second_expr,
                        SubstituteOptions::exact(),
                    );
                    let first_val = self.fold_free_subtree(verify_options, first_back);
                    let second_val = self.fold_free_subtree(verify_options, second_expr);
                    // Map back to (C1, C2) order.
                    let (c1_val, c2_val) = if first_name == c1_name {
                        (first_val, second_val)
                    } else {
                        (second_val, first_val)
                    };
                    solved = Some((c1_val, c2_val));
                    break 'orders;
                }
            }
        }
        let Some((c1_val, c2_val)) = solved else {
            let ctx = &mut self.simplifier.context;
            return Ok(residual_action_result(
                ctx,
                resolved,
                y_var,
                x_var,
                "Las condiciones iniciales no determinaron C1 y C2 (sistema inconsistente o degenerado); se declina honesto",
            ));
        };

        // Particular solution. The ODE itself needs NO re-verification for
        // y_p: both basis functions already reduced L[b_i] to exact 0, and L
        // is linear with constant coefficients, so L[c1·b1 + c2·b2] ≡ 0 by
        // THEOREM for any constants — substituting the combination back walks
        // straight into the constant×exp×trig oscillation family the D5
        // linearity doctrine exists to dodge (the complex-envelope IVP burned
        // its whole verify budget exactly there). The CONDITIONS are what the
        // constants must satisfy — those verify directly below.
        let ctx = &mut self.simplifier.context;
        let with_c1 = substitute_power_aware(ctx, general, c1, c1_val, SubstituteOptions::exact());
        let y_p_raw = substitute_power_aware(ctx, with_c1, c2, c2_val, SubstituteOptions::exact());
        let y_p = self.fold_free_subtree(verify_options, y_p_raw);

        for cond in conditions {
            let lhs_fn = if cond.order == 0 {
                y_p
            } else {
                let ctx = &mut self.simplifier.context;
                let d_call = ctx.call("diff", vec![y_p, x_var]);
                self.fold_free_subtree(verify_options, d_call)
            };
            let ctx = &mut self.simplifier.context;
            let at_point =
                substitute_power_aware(ctx, lhs_fn, x_var, cond.point, SubstituteOptions::exact());
            let residue = ctx.add(cas_ast::Expr::Sub(at_point, cond.value));
            if !self.reduces_to_zero_exact(verify_options, residue) {
                let ctx = &mut self.simplifier.context;
                return Ok(residual_action_result(
                    ctx,
                    resolved,
                    y_var,
                    x_var,
                    "La solución particular no verificó una condición inicial; se declina honesto",
                ));
            }
        }

        // Drop the free-constant warnings (both constants got pinned).
        let warnings: Vec<DomainWarning> = warnings
            .into_iter()
            .filter(|w| !w.message.contains("constantes arbitrarias"))
            .collect();
        let ctx = &mut self.simplifier.context;
        let point_str = render_expr(ctx, conditions[0].point);
        solve_steps.push(crate::api::SolveStep {
            description: format!(
                "Aplicar las condiciones iniciales en {var} = {point_str}: resolver el sistema 2×2 en {c1_name}, {c2_name}"
            ),
            equation_after: Equation {
                lhs: c1,
                rhs: c1_val,
                op: RelOp::Eq,
            },
            importance: ImportanceLevel::High,
            substeps: vec![],
        });
        solve_steps.push(crate::api::SolveStep {
            description: "Solución particular con la condición aplicada".to_string(),
            equation_after: Equation {
                lhs: y_var,
                rhs: y_p,
                op: RelOp::Eq,
            },
            importance: ImportanceLevel::High,
            substeps: vec![],
        });
        let result = EvalResult::Expr(wrap_eq(ctx, y_var, y_p));
        Ok((
            result,
            warnings,
            vec![],
            solve_steps,
            vec![],
            vec![],
            vec![],
            vec![],
        ))
    }

    /// O9 (pre-approved optional): Cauchy-Euler `x²y'' + a·x·y' + b·y = 0` by
    /// the indicial equation `r(r−1) + (c1/c2)·r + (c0/c2) = 0` — the same
    /// exact-discriminant mould as D9 with `x^r`/`ln x` bases instead of
    /// exponentials. Emission stays basis-gated (D5).
    #[allow(clippy::too_many_arguments)]
    fn eval_dsolve_cauchy_euler(
        &mut self,
        options: &crate::options::EvalOptions,
        resolved: ExprId,
        func: &str,
        var: &str,
        ode_eq: &Equation,
        ce: CauchyEulerOde,
        conditions: &[InitialCondition],
    ) -> Result<ActionResult, anyhow::Error> {
        let mut verify_options = options.clone();
        verify_options.steps_mode = cas_solver_core::eval_options::StepsMode::Off;
        verify_options.time_budget_ms = Some(
            verify_options
                .time_budget_ms
                .map_or(VERIFY_TIME_BUDGET_MS, |t| t.min(VERIFY_TIME_BUDGET_MS)),
        );
        let ctx = &mut self.simplifier.context;
        let y_var = ctx.var(func);
        let x_var = ctx.var(var);
        if !conditions.is_empty() {
            let r = residual_action_result(
                ctx,
                resolved,
                y_var,
                x_var,
                "IVP de Cauchy-Euler: peldaño futuro (x0 = 0 es singular); se declina honesto",
            );
            return Ok(r);
        }

        // Indicial r² + (a−1)·r + b with a = c1/c2, b = c0/c2.
        let one = num_rational::BigRational::from_integer(1.into());
        let p = &ce.c1 / &ce.c2 - &one;
        let q = &ce.c0 / &ce.c2;
        let disc = &p * &p - num_rational::BigRational::from_integer(4.into()) * &q;
        let two = num_rational::BigRational::from_integer(2.into());

        let x_pow_rat = |ctx: &mut Context, r: &num_rational::BigRational| -> ExprId {
            if r.is_zero() {
                return ctx.num(1);
            }
            if *r == one {
                return ctx.var(var);
            }
            let x = ctx.var(var);
            let e = ctx.add(cas_ast::Expr::Number(r.clone()));
            ctx.add(cas_ast::Expr::Pow(x, e))
        };

        enum CeShape {
            Distinct,
            Double,
            Complex,
        }
        let (u1, u2, shape) = if disc.is_positive() {
            let Some(s) = cas_math::perfect_square_support::rational_sqrt(&disc) else {
                let ctx = &mut self.simplifier.context;
                let r = residual_action_result(
                    ctx,
                    resolved,
                    y_var,
                    x_var,
                    "Raíces indiciales irracionales: peldaño futuro; se declina honesto",
                );
                return Ok(r);
            };
            let r1 = (-&p + &s) / &two;
            let r2 = (-&p - &s) / &two;
            let ctx = &mut self.simplifier.context;
            let u1 = x_pow_rat(ctx, &r1);
            let u2 = x_pow_rat(ctx, &r2);
            (u1, u2, CeShape::Distinct)
        } else if disc.is_zero() {
            let r = -&p / &two;
            let ctx = &mut self.simplifier.context;
            let xr = x_pow_rat(ctx, &r);
            let ln_x = {
                let x = ctx.var(var);
                ctx.call("ln", vec![x])
            };
            let u2 = ctx.add(cas_ast::Expr::Mul(xr, ln_x));
            (xr, u2, CeShape::Double)
        } else {
            let neg_disc = -&disc;
            let Some(s) = cas_math::perfect_square_support::rational_sqrt(&neg_disc) else {
                let ctx = &mut self.simplifier.context;
                let r = residual_action_result(
                    ctx,
                    resolved,
                    y_var,
                    x_var,
                    "Parte imaginaria indicial irracional: peldaño futuro; se declina honesto",
                );
                return Ok(r);
            };
            let alpha = -&p / &two;
            let beta = &s / &two;
            let ctx = &mut self.simplifier.context;
            let x = ctx.var(var);
            let ln_x = ctx.call("ln", vec![x]);
            let b_num = ctx.add(cas_ast::Expr::Number(beta));
            let arg = ctx.add(cas_ast::Expr::Mul(b_num, ln_x));
            let cos_part = ctx.call("cos", vec![arg]);
            let sin_part = ctx.call("sin", vec![arg]);
            let (u1, u2) = if alpha.is_zero() {
                (cos_part, sin_part)
            } else {
                let xa = x_pow_rat(ctx, &alpha);
                let u1 = ctx.add(cas_ast::Expr::Mul(xa, cos_part));
                let u2 = ctx.add(cas_ast::Expr::Mul(xa, sin_part));
                (u1, u2)
            };
            (u1, u2, CeShape::Complex)
        };
        let u1 = self.fold_free_subtree(&verify_options, u1);
        let u2 = self.fold_free_subtree(&verify_options, u2);

        // D5 basis gate against the raw ODE residue.
        let ctx = &mut self.simplifier.context;
        let ode_residue = ctx.add(cas_ast::Expr::Sub(ode_eq.lhs, ode_eq.rhs));
        for u in [u1, u2] {
            let ctx = &mut self.simplifier.context;
            let substituted =
                substitute_power_aware(ctx, ode_residue, y_var, u, SubstituteOptions::exact());
            if !self.reduces_to_zero_exact(&verify_options, substituted) {
                let ctx = &mut self.simplifier.context;
                return Ok(residual_action_result(
                    ctx,
                    resolved,
                    y_var,
                    x_var,
                    "Una base de Cauchy-Euler no verificó contra la EDO; se declina honesto",
                ));
            }
        }

        // Fresh constants + textbook display (mul_basis rules).
        let ctx = &mut self.simplifier.context;
        let input_vars = collect_variables(ctx, resolved);
        let (c1_name, c2_name) = if input_vars.contains("C1") || input_vars.contains("C2") {
            ("K1", "K2")
        } else {
            ("C1", "C2")
        };
        let c1 = ctx.var(c1_name);
        let c2 = ctx.var(c2_name);
        let mul_basis = |ctx: &mut Context, coef: ExprId, u: ExprId| -> ExprId {
            if is_literal_one(ctx, u) {
                return coef;
            }
            if let cas_ast::Expr::Div(num, den) = ctx.get(u) {
                let (num, den) = (*num, *den);
                if is_literal_one(ctx, num) {
                    return ctx.add(cas_ast::Expr::Div(coef, den));
                }
            }
            ctx.add(cas_ast::Expr::Mul(coef, u))
        };
        let general = match shape {
            CeShape::Double => {
                // x^r·(C1 + C2·ln x).
                let c2ln = {
                    let x = ctx.var(var);
                    let ln_x = ctx.call("ln", vec![x]);
                    ctx.add(cas_ast::Expr::Mul(c2, ln_x))
                };
                let head = ctx.add(cas_ast::Expr::Add(c1, c2ln));
                if is_literal_one(ctx, u1) {
                    head
                } else {
                    mul_basis(ctx, head, u1)
                }
            }
            _ => {
                let t1 = mul_basis(ctx, c1, u1);
                let t2 = mul_basis(ctx, c2, u2);
                ctx.add(cas_ast::Expr::Add(t1, t2))
            }
        };

        let mut warnings: Vec<DomainWarning> = Vec::new();
        if c1_name != "C1" {
            warnings.push(DomainWarning {
                message:
                    "La entrada ya usa C1/C2; las constantes arbitrarias se emiten como K1, K2"
                        .to_string(),
                rule_name: DSOLVE_RULE.to_string(),
            });
        }
        warnings.push(DomainWarning {
            message: format!(
                "Solución general de Cauchy-Euler (dominio x > 0): {c1_name} y {c2_name} son constantes arbitrarias"
            ),
            rule_name: DSOLVE_RULE.to_string(),
        });

        let branch_desc = match shape {
            CeShape::Distinct => "Raíces indiciales reales distintas: base {x^r1, x^r2}",
            CeShape::Double => "Raíz indicial doble: base {x^r, x^r·ln(x)}",
            CeShape::Complex => {
                "Raíces indiciales complejas: base {x^α·cos(β·ln x), x^α·sin(β·ln x)}"
            }
        };
        let mut solve_steps: Vec<crate::api::SolveStep> = Vec::new();
        solve_steps.push(crate::api::SolveStep {
            description: format!(
                "Identificar ecuación de Cauchy-Euler: x²·y'' + a·x·y' + b·y = 0 con a = {}, b = {}",
                &ce.c1 / &ce.c2,
                q
            ),
            equation_after: ode_eq.clone(),
            importance: ImportanceLevel::High,
            substeps: vec![],
        });
        solve_steps.push(crate::api::SolveStep {
            description: format!(
                "Plantear la ecuación indicial: r·(r−1) + a·r + b = 0 con discriminante Δ = {disc}"
            ),
            equation_after: ode_eq.clone(),
            importance: ImportanceLevel::High,
            substeps: vec![],
        });
        solve_steps.push(crate::api::SolveStep {
            description: branch_desc.to_string(),
            equation_after: Equation {
                lhs: y_var,
                rhs: general,
                op: RelOp::Eq,
            },
            importance: ImportanceLevel::High,
            substeps: vec![],
        });
        solve_steps.push(crate::api::SolveStep {
            description: "Verificar por sustitución: cada función de la base anula la EDO"
                .to_string(),
            equation_after: Equation {
                lhs: y_var,
                rhs: general,
                op: RelOp::Eq,
            },
            importance: ImportanceLevel::Medium,
            substeps: vec![],
        });
        let result = EvalResult::Expr(wrap_eq(ctx, y_var, general));
        Ok((
            result,
            warnings,
            vec![],
            solve_steps,
            vec![],
            vec![],
            vec![],
            vec![],
        ))
    }

    /// O6: first-order 2×2 linear system `X' = A·X` by the INTERNAL exact
    /// eigen route (D17: the `eigenvalues`/`eigenvectors` verbs are NEVER
    /// touched — their declines are their own contract). Characteristic
    /// λ² − tr·λ + det over BigRational, eigenvectors by hand for 2×2,
    /// complex pairs emitted as REAL solutions via Re/Im, defective double
    /// roots via a generalized vector. Every basis solution verifies against
    /// BOTH equations (D5 per component) before anything is emitted.
    pub(super) fn eval_dsolve_system(
        &mut self,
        options: &crate::options::EvalOptions,
        resolved: ExprId,
        second_equation: ExprId,
        funcs: &[String],
        var: &str,
        conditions: &[InitialCondition],
    ) -> Result<ActionResult, anyhow::Error> {
        let mut verify_options = options.clone();
        verify_options.steps_mode = cas_solver_core::eval_options::StepsMode::Off;
        verify_options.time_budget_ms = Some(
            verify_options
                .time_budget_ms
                .map_or(VERIFY_TIME_BUDGET_MS, |t| t.min(VERIFY_TIME_BUDGET_MS)),
        );

        let ctx = &mut self.simplifier.context;
        let diff_sym = ctx.intern_symbol("diff");
        let f0_var = ctx.var(&funcs[0]);
        let f1_var = ctx.var(&funcs[1]);
        let t_var = ctx.var(var);
        // Residual echo carries the list form.
        let mk_residual = |ctx: &mut Context, reason: &str| -> ActionResult {
            let eco = ctx.call(DSOLVE_RULE, vec![resolved, second_equation, t_var]);
            (
                EvalResult::SolutionSet(SolutionSet::Residual(eco)),
                vec![DomainWarning {
                    message: reason.to_string(),
                    rule_name: DSOLVE_RULE.to_string(),
                }],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
            )
        };

        if !conditions.is_empty() {
            let r = mk_residual(
                ctx,
                "Condiciones iniciales de sistemas: ciclo futuro; se declina honesto",
            );
            return Ok(r);
        }

        // Extract A row by row (order fixed by which unknown is differentiated).
        let eq1 = cas_solver_core::solve_entry::equation_from_expr_or_zero(ctx, resolved);
        let eq2 = cas_solver_core::solve_entry::equation_from_expr_or_zero(ctx, second_equation);
        let zero = num_rational::BigRational::from_integer(0.into());
        let mut a = [[zero.clone(), zero.clone()], [zero.clone(), zero.clone()]];
        let mut seen = [false, false];
        for eq in [&eq1, &eq2] {
            let Some((which, row)) = try_extract_system_row(ctx, eq, diff_sym, funcs, var) else {
                let r = mk_residual(
                    ctx,
                    "El sistema no es lineal homogéneo de coeficientes constantes X' = A·X (o no es 2×2); se declina honesto",
                );
                return Ok(r);
            };
            if seen[which] {
                let r = mk_residual(
                    ctx,
                    "Las dos ecuaciones derivan la misma incógnita; se declina honesto",
                );
                return Ok(r);
            }
            seen[which] = true;
            a[which] = row;
        }

        // Characteristic λ² − tr·λ + det, exact.
        let tr = &a[0][0] + &a[1][1];
        let det = &a[0][0] * &a[1][1] - &a[0][1] * &a[1][0];
        let disc = &tr * &tr - num_rational::BigRational::from_integer(4.into()) * &det;
        let two = num_rational::BigRational::from_integer(2.into());

        // Eigenvector for a RATIONAL eigenvalue λ (2×2 by hand).
        let eigvec = |a: &[[num_rational::BigRational; 2]; 2],
                      lam: &num_rational::BigRational|
         -> [num_rational::BigRational; 2] {
            if !a[0][1].is_zero() {
                [a[0][1].clone(), lam - &a[0][0]]
            } else if !a[1][0].is_zero() {
                [lam - &a[1][1], a[1][0].clone()]
            } else if *lam == a[0][0] {
                [
                    num_rational::BigRational::from_integer(1.into()),
                    num_rational::BigRational::from_integer(0.into()),
                ]
            } else {
                [
                    num_rational::BigRational::from_integer(0.into()),
                    num_rational::BigRational::from_integer(1.into()),
                ]
            }
        };
        let rat_expr = |ctx: &mut Context, r: &num_rational::BigRational| -> ExprId {
            ctx.add(cas_ast::Expr::Number(r.clone()))
        };
        // vec_scale: component expression coef·shape (folded later per basis).
        let scale = |ctx: &mut Context, c: &num_rational::BigRational, e: ExprId| -> ExprId {
            let cn = rat_expr(ctx, c);
            ctx.add(cas_ast::Expr::Mul(cn, e))
        };

        enum BranchKind {
            Distinct,
            Defective,
            Diagonal,
            Complex,
        }
        let (bases, branch): ([SystemBasis; 2], BranchKind) = if disc.is_positive() {
            let Some(s) = cas_math::perfect_square_support::rational_sqrt(&disc) else {
                let ctx = &mut self.simplifier.context;
                let r = mk_residual(
                    ctx,
                    "Autovalores irracionales (discriminante no cuadrado): peldaño futuro; se declina honesto",
                );
                return Ok(r);
            };
            let l1 = (&tr + &s) / &two;
            let l2 = (&tr - &s) / &two;
            let v1 = eigvec(&a, &l1);
            let v2 = eigvec(&a, &l2);
            let ctx = &mut self.simplifier.context;
            let e1 = build_exp_rate(ctx, &l1, t_var);
            let e2 = build_exp_rate(ctx, &l2, t_var);
            (
                [
                    SystemBasis {
                        comp: [scale(ctx, &v1[0], e1), scale(ctx, &v1[1], e1)],
                    },
                    SystemBasis {
                        comp: [scale(ctx, &v2[0], e2), scale(ctx, &v2[1], e2)],
                    },
                ],
                BranchKind::Distinct,
            )
        } else if disc.is_zero() {
            let lam = &tr / &two;
            let a_minus_diag_zero =
                a[0][1].is_zero() && a[1][0].is_zero() && a[0][0] == lam && a[1][1] == lam;
            let ctx = &mut self.simplifier.context;
            let e_l = build_exp_rate(ctx, &lam, t_var);
            if a_minus_diag_zero {
                // A = λI: decoupled, two independent eigenvectors.
                let zero_e = ctx.num(0);
                (
                    [
                        SystemBasis {
                            comp: [e_l, zero_e],
                        },
                        SystemBasis {
                            comp: [zero_e, e_l],
                        },
                    ],
                    BranchKind::Diagonal,
                )
            } else {
                // Defective: v eigenvector, w generalized with (A−λI)w = v.
                let v = eigvec(&a, &lam);
                let m = [
                    [&a[0][0] - &lam, a[0][1].clone()],
                    [a[1][0].clone(), &a[1][1] - &lam],
                ];
                // Rank-1 solve: pick a nonzero row.
                let w = if !m[0][0].is_zero() {
                    [&v[0] / &m[0][0], zero.clone()]
                } else if !m[0][1].is_zero() {
                    [zero.clone(), &v[0] / &m[0][1]]
                } else if !m[1][0].is_zero() {
                    [&v[1] / &m[1][0], zero.clone()]
                } else if !m[1][1].is_zero() {
                    [zero.clone(), &v[1] / &m[1][1]]
                } else {
                    let ctx = &mut self.simplifier.context;
                    let r = mk_residual(ctx, "Sistema defectivo degenerado; se declina honesto");
                    return Ok(r);
                };
                // X2 = (v·t + w)·e^(λt) per component.
                let mut comp2 = [t_var, t_var];
                for i in 0..2 {
                    let ctx = &mut self.simplifier.context;
                    let vt = scale(ctx, &v[i], t_var);
                    let wn = rat_expr(ctx, &w[i]);
                    let sum = ctx.add(cas_ast::Expr::Add(vt, wn));
                    comp2[i] = ctx.add(cas_ast::Expr::Mul(sum, e_l));
                }
                let ctx = &mut self.simplifier.context;
                (
                    [
                        SystemBasis {
                            comp: [scale(ctx, &v[0], e_l), scale(ctx, &v[1], e_l)],
                        },
                        SystemBasis { comp: comp2 },
                    ],
                    BranchKind::Defective,
                )
            }
        } else {
            let neg_disc = -&disc;
            let Some(s) = cas_math::perfect_square_support::rational_sqrt(&neg_disc) else {
                let ctx = &mut self.simplifier.context;
                let r = mk_residual(
                    ctx,
                    "Autovalores complejos con parte imaginaria irracional: peldaño futuro; se declina honesto",
                );
                return Ok(r);
            };
            let alpha = &tr / &two;
            let beta = &s / &two;
            // Complex eigenvector for λ = α + iβ: v = vr + i·vi.
            let (vr, vi) = if !a[0][1].is_zero() {
                (
                    [a[0][1].clone(), &alpha - &a[0][0]],
                    [zero.clone(), beta.clone()],
                )
            } else if !a[1][0].is_zero() {
                (
                    [&alpha - &a[1][1], a[1][0].clone()],
                    [beta.clone(), zero.clone()],
                )
            } else {
                let ctx = &mut self.simplifier.context;
                let r = mk_residual(ctx, "Sistema complejo degenerado; se declina honesto");
                return Ok(r);
            };
            let ctx = &mut self.simplifier.context;
            let b_num = rat_expr(ctx, &beta);
            let bt = ctx.add(cas_ast::Expr::Mul(b_num, t_var));
            let cos_bt = ctx.call("cos", vec![bt]);
            let sin_bt = ctx.call("sin", vec![bt]);
            // X1 = e^(αt)(vr·cos − vi·sin); X2 = e^(αt)(vr·sin + vi·cos).
            let mut comp1 = [t_var, t_var];
            let mut comp2 = [t_var, t_var];
            for i in 0..2 {
                let ctx = &mut self.simplifier.context;
                let rc = scale(ctx, &vr[i], cos_bt);
                let is_ = scale(ctx, &vi[i], sin_bt);
                let x1 = ctx.add(cas_ast::Expr::Sub(rc, is_));
                let rs = scale(ctx, &vr[i], sin_bt);
                let ic = scale(ctx, &vi[i], cos_bt);
                let x2 = ctx.add(cas_ast::Expr::Add(rs, ic));
                if alpha.is_zero() {
                    comp1[i] = x1;
                    comp2[i] = x2;
                } else {
                    let env = build_exp_rate(ctx, &alpha, t_var);
                    comp1[i] = ctx.add(cas_ast::Expr::Mul(env, x1));
                    comp2[i] = ctx.add(cas_ast::Expr::Mul(env, x2));
                }
            }
            (
                [SystemBasis { comp: comp1 }, SystemBasis { comp: comp2 }],
                BranchKind::Complex,
            )
        };

        // Fold each basis component (numeric coefficients — no hostility).
        let mut folded_bases: Vec<[ExprId; 2]> = Vec::new();
        for basis in &bases {
            let c0 = self.fold_free_subtree(&verify_options, basis.comp[0]);
            let c1 = self.fold_free_subtree(&verify_options, basis.comp[1]);
            folded_bases.push([c0, c1]);
        }

        // D5 per-component gate: each basis solution must annihilate BOTH
        // equations (substitute both unknowns simultaneously via two chained
        // exact substitutions on the raw residues).
        for basis in &folded_bases {
            for eq in [&eq1, &eq2] {
                let ctx = &mut self.simplifier.context;
                let residue = ctx.add(cas_ast::Expr::Sub(eq.lhs, eq.rhs));
                let sub0 = substitute_power_aware(
                    ctx,
                    residue,
                    f0_var,
                    basis[0],
                    SubstituteOptions::exact(),
                );
                let sub01 =
                    substitute_power_aware(ctx, sub0, f1_var, basis[1], SubstituteOptions::exact());
                if !self.reduces_to_zero_exact(&verify_options, sub01) {
                    let ctx = &mut self.simplifier.context;
                    let r = mk_residual(
                        ctx,
                        "Una solución base del sistema no verificó contra ambas ecuaciones; se declina honesto",
                    );
                    return Ok(r);
                }
            }
        }

        // Fresh constants (D7).
        let ctx = &mut self.simplifier.context;
        let mut input_vars = collect_variables(ctx, resolved);
        input_vars.extend(collect_variables(ctx, second_equation));
        let (c1_name, c2_name) = if input_vars.contains("C1") || input_vars.contains("C2") {
            ("K1", "K2")
        } else {
            ("C1", "C2")
        };
        let c1 = ctx.var(c1_name);
        let c2 = ctx.var(c2_name);

        // General per component: f_i = C1·basis1_i + C2·basis2_i. Structural
        // display rules (the O4 lesson): a reciprocal basis multiplies as a
        // clean quotient, and a ZERO basis component drops its term entirely
        // (never emit `0·C1`).
        let mul_basis = |ctx: &mut Context, coef: ExprId, u: ExprId| -> Option<ExprId> {
            if matches!(ctx.get(u), cas_ast::Expr::Number(n) if n.is_zero()) {
                return None;
            }
            if let cas_ast::Expr::Div(num, den) = ctx.get(u) {
                let (num, den) = (*num, *den);
                if is_literal_one(ctx, num) {
                    return Some(ctx.add(cas_ast::Expr::Div(coef, den)));
                }
            }
            if let cas_ast::Expr::Neg(inner) = ctx.get(u) {
                let inner = *inner;
                if let cas_ast::Expr::Div(num, den) = ctx.get(inner) {
                    let (num, den) = (*num, *den);
                    if is_literal_one(ctx, num) {
                        let q = ctx.add(cas_ast::Expr::Div(coef, den));
                        return Some(ctx.add(cas_ast::Expr::Neg(q)));
                    }
                }
            }
            Some(ctx.add(cas_ast::Expr::Mul(coef, u)))
        };
        let mut general = [t_var, t_var];
        for i in 0..2 {
            let ctx = &mut self.simplifier.context;
            let t1 = mul_basis(ctx, c1, folded_bases[0][i]);
            let t2 = mul_basis(ctx, c2, folded_bases[1][i]);
            general[i] = match (t1, t2) {
                (Some(a_), Some(b_)) => ctx.add(cas_ast::Expr::Add(a_, b_)),
                (Some(a_), None) => a_,
                (None, Some(b_)) => b_,
                (None, None) => ctx.num(0),
            };
        }

        let ctx = &mut self.simplifier.context;
        let mut warnings: Vec<DomainWarning> = Vec::new();
        if c1_name != "C1" {
            warnings.push(DomainWarning {
                message:
                    "La entrada ya usa C1/C2; las constantes arbitrarias se emiten como K1, K2"
                        .to_string(),
                rule_name: DSOLVE_RULE.to_string(),
            });
        }
        warnings.push(DomainWarning {
            message: format!(
                "Solución general del sistema: {c1_name} y {c2_name} son constantes arbitrarias"
            ),
            rule_name: DSOLVE_RULE.to_string(),
        });

        let branch_desc = match branch {
            BranchKind::Distinct => "Autovalores reales distintos: base {v1·e^(λ1·t), v2·e^(λ2·t)}",
            BranchKind::Diagonal => "Autovalor doble con A = λI: sistema desacoplado",
            BranchKind::Defective => {
                "Autovalor doble defectivo: base {v·e^(λ·t), (v·t + w)·e^(λ·t)} con (A−λI)w = v"
            }
            BranchKind::Complex => {
                "Autovalores complejos conjugados: soluciones REALES por partes real/imaginaria"
            }
        };
        let mut solve_steps: Vec<crate::api::SolveStep> = Vec::new();
        solve_steps.push(crate::api::SolveStep {
            description: format!(
                "Identificar sistema lineal X' = A·X con A = [[{}, {}], [{}, {}]]",
                a[0][0], a[0][1], a[1][0], a[1][1]
            ),
            equation_after: eq1.clone(),
            importance: ImportanceLevel::High,
            substeps: vec![],
        });
        solve_steps.push(crate::api::SolveStep {
            description: format!(
                "Característica del sistema: λ² − tr(A)·λ + det(A) = 0 con tr = {tr}, det = {det}"
            ),
            equation_after: eq1.clone(),
            importance: ImportanceLevel::High,
            substeps: vec![],
        });
        solve_steps.push(crate::api::SolveStep {
            description: branch_desc.to_string(),
            equation_after: Equation {
                lhs: f0_var,
                rhs: general[0],
                op: RelOp::Eq,
            },
            importance: ImportanceLevel::High,
            substeps: vec![],
        });
        solve_steps.push(crate::api::SolveStep {
            description:
                "Verificar por sustitución: cada solución base anula AMBAS ecuaciones del sistema"
                    .to_string(),
            equation_after: Equation {
                lhs: f1_var,
                rhs: general[1],
                op: RelOp::Eq,
            },
            importance: ImportanceLevel::Medium,
            substeps: vec![],
        });

        let eq_x = wrap_eq(ctx, f0_var, general[0]);
        let eq_y = wrap_eq(ctx, f1_var, general[1]);
        let result = EvalResult::SolutionSet(SolutionSet::Discrete(vec![eq_x, eq_y]));
        Ok((
            result,
            warnings,
            vec![],
            solve_steps,
            vec![],
            vec![],
            vec![],
            vec![],
        ))
    }

    /// D11 level 2: reconstruct φ = ∫M dx + ∫(N − ∂y∫M) dy with the FULL
    /// evaluator folding the intermediate pieces (the internal F6 path only
    /// canonicalizes polynomial rests). All pieces are diff-free ordinary
    /// expressions in (x, y), so the D4 invariant allows the pipeline.
    /// O8a: Bernoulli `y' + p·y = q·yⁿ` — substitute `v = y^(1−n)` to get the
    /// LINEAR `v' + (1−n)p·v = (1−n)q`, solve with the shared integrating-
    /// factor core, and back-substitute. `n = 2` emits the explicit textbook
    /// `y = 1/v`; other integer n emit the branch-complete IMPLICIT relation
    /// `y^(1−n) = v(x)` (a fractional back-power would silently drop the ±
    /// branch). Emission stays verification-gated (D5).
    fn eval_dsolve_bernoulli(
        &mut self,
        options: &crate::options::EvalOptions,
        resolved: ExprId,
        func: &str,
        var: &str,
        ode_eq: &Equation,
        bern: BernoulliOde,
    ) -> Result<ActionResult, anyhow::Error> {
        let mut verify_options = options.clone();
        verify_options.steps_mode = cas_solver_core::eval_options::StepsMode::Off;
        verify_options.time_budget_ms = Some(
            verify_options
                .time_budget_ms
                .map_or(VERIFY_TIME_BUDGET_MS, |t| t.min(VERIFY_TIME_BUDGET_MS)),
        );

        let one = num_rational::BigRational::from_integer(1.into());
        let one_minus_n = &one - &bern.n;
        let p = self.fold_free_subtree(&verify_options, bern.p);
        let q = self.fold_free_subtree(&verify_options, bern.q);
        let ctx = &mut self.simplifier.context;
        let y_var = ctx.var(func);
        let x_var = ctx.var(var);

        // v' + (1−n)·p·v = (1−n)·q.
        let factor = ctx.add(cas_ast::Expr::Number(one_minus_n.clone()));
        let p_v_raw = ctx.add(cas_ast::Expr::Mul(factor, p));
        let q_v_raw = ctx.add(cas_ast::Expr::Mul(factor, q));
        let p_v = self.fold_free_subtree(&verify_options, p_v_raw);
        let q_v = self.fold_free_subtree(&verify_options, q_v_raw);

        // Arbitrary constant (D7).
        let ctx = &mut self.simplifier.context;
        let input_vars = collect_variables(ctx, resolved);
        let c_name = if input_vars.contains("C") { "K1" } else { "C" };
        let c_var = ctx.var(c_name);
        let mut warnings: Vec<DomainWarning> = Vec::new();
        if c_name != "C" {
            warnings.push(DomainWarning {
                message: "La entrada ya usa el nombre C; la constante arbitraria se emite como K1"
                    .to_string(),
                rule_name: DSOLVE_RULE.to_string(),
            });
        }

        let (_mu, _g_int, v_general) =
            match self.linear_general_candidate(&verify_options, p_v, q_v, func, var, c_var) {
                Ok(core) => core,
                Err(reason) => {
                    let ctx = &mut self.simplifier.context;
                    return Ok(residual_action_result(ctx, resolved, y_var, x_var, reason));
                }
            };

        // Back-substitution and verification.
        let ctx = &mut self.simplifier.context;
        let neg_one = -&one;
        let ode_residue = ctx.add(cas_ast::Expr::Sub(ode_eq.lhs, ode_eq.rhs));
        let (result_expr, verified_form) = if one_minus_n == neg_one {
            // n = 2: y = 1/v, the explicit textbook form.
            let one_lit = ctx.num(1);
            let y_cand = ctx.add(cas_ast::Expr::Div(one_lit, v_general));
            let substituted =
                substitute_power_aware(ctx, ode_residue, y_var, y_cand, SubstituteOptions::exact());
            if !self.reduces_to_zero_exact(&verify_options, substituted) {
                let ctx = &mut self.simplifier.context;
                return Ok(residual_action_result(
                    ctx,
                    resolved,
                    y_var,
                    x_var,
                    "La candidata Bernoulli no verificó (residuo ≠ 0 exacto); se declina honesto",
                ));
            }
            let ctx = &mut self.simplifier.context;
            (wrap_eq(ctx, y_var, y_cand), y_cand)
        } else {
            // n ≠ 2: the back-power 1/(1−n) is fractional (branch-dropping)
            // and the implicit relation y^(1−n) = v(x) is NOT a free-variable
            // identity (its residue vanishes only ON the solution curve), so
            // the D5 gate cannot certify it as-is. Honest decline with the
            // step named — never an unverified emission.
            let _ = ode_residue;
            let ctx = &mut self.simplifier.context;
            return Ok(residual_action_result(
                ctx,
                resolved,
                y_var,
                x_var,
                "Bernoulli con n ≠ 2: la relación y^(1−n) = v(x) requiere verificación por rama (peldaño futuro); se declina honesto",
            ));
        };

        let ctx = &mut self.simplifier.context;
        warnings.push(DomainWarning {
            message: format!("Solución general: {c_name} es una constante arbitraria"),
            rule_name: DSOLVE_RULE.to_string(),
        });
        if bern.n.is_positive() {
            warnings.push(DomainWarning {
                message: format!(
                    "{func} = 0 es solución singular (descartada al dividir por {func}^n)"
                ),
                rule_name: DSOLVE_RULE.to_string(),
            });
        }

        let solve_steps = build_bernoulli_steps(
            ctx,
            ode_eq,
            p,
            q,
            &bern.n,
            &one_minus_n,
            y_var,
            v_general,
            verified_form,
        );
        Ok((
            EvalResult::Expr(result_expr),
            warnings,
            vec![],
            solve_steps,
            vec![],
            vec![],
            vec![],
            vec![],
        ))
    }

    /// O8b: first-order homogeneous `y' = F(y/x)` — substitute `v = y/x` to
    /// reduce to the separable `dv/(F(v)−v) = dx/x`, integrate both sides,
    /// solve back for `v`, and emit `y = x·v(x)` (or the implicit
    /// `G(y/x) − ln x = C` when the inverse does not close cleanly — which
    /// also dodges the known H19 surd-residue problem entirely). Emission
    /// stays verification-gated (D5).
    #[allow(clippy::too_many_arguments)]
    fn eval_dsolve_homogeneous(
        &mut self,
        options: &crate::options::EvalOptions,
        resolved: ExprId,
        func: &str,
        var: &str,
        ode_eq: &Equation,
        rhs: ExprId,
        v_var: ExprId,
        f_v: ExprId,
    ) -> Result<Option<ActionResult>, anyhow::Error> {
        let mut verify_options = options.clone();
        verify_options.steps_mode = cas_solver_core::eval_options::StepsMode::Off;
        verify_options.time_budget_ms = Some(
            verify_options
                .time_budget_ms
                .map_or(VERIFY_TIME_BUDGET_MS, |t| t.min(VERIFY_TIME_BUDGET_MS)),
        );

        let ctx = &mut self.simplifier.context;
        let y_var = ctx.var(func);
        let x_var = ctx.var(var);
        let v_name = match ctx.get(v_var) {
            cas_ast::Expr::Variable(s) => ctx.sym_name(*s).to_string(),
            _ => return Ok(None),
        };

        // F(v) − v: zero would mean y' = y/x (the separable path owns it).
        let fv_minus_v_raw = ctx.add(cas_ast::Expr::Sub(f_v, v_var));
        let fv_minus_v = self.fold_free_subtree(&verify_options, fv_minus_v_raw);
        let ctx = &mut self.simplifier.context;
        if matches!(ctx.get(fv_minus_v), cas_ast::Expr::Number(n) if n.is_zero()) {
            return Ok(None);
        }

        // G(v) = ∫ dv/(F(v)−v); honest residual when it does not close.
        let one_lit = ctx.num(1);
        let integrand = ctx.add(cas_ast::Expr::Div(one_lit, fv_minus_v));
        let g_int = match crate::rules::calculus::integrate_with_trace(ctx, integrand, &v_name) {
            Some(outcome) if outcome.required_conditions.is_empty() => outcome.result,
            _ => {
                let r = residual_action_result(
                    ctx,
                    resolved,
                    y_var,
                    x_var,
                    "La integral ∫ dv/(F(v)−v) de la reducción homogénea no cierra en forma elemental; se declina honesto",
                );
                return Ok(Some(r));
            }
        };
        // ln(x) side (abs stripped for textbook display — D12).
        let one_lit = ctx.num(1);
        let inv_x = ctx.add(cas_ast::Expr::Div(one_lit, x_var));
        let ln_side_raw = match crate::rules::calculus::integrate_with_trace(ctx, inv_x, var) {
            Some(outcome) if outcome.required_conditions.is_empty() => outcome.result,
            _ => return Ok(None),
        };
        let ln_folded = self.fold_free_subtree(&verify_options, ln_side_raw);
        let ctx = &mut self.simplifier.context;
        let ln_side = strip_free_abs(ctx, ln_folded, func);

        // Arbitrary constant (D7).
        let input_vars = collect_variables(ctx, resolved);
        let c_name = if input_vars.contains("C") { "K1" } else { "C" };
        let c_var = ctx.var(c_name);
        let mut warnings: Vec<DomainWarning> = Vec::new();
        if c_name != "C" {
            warnings.push(DomainWarning {
                message: "La entrada ya usa el nombre C; la constante arbitraria se emite como K1"
                    .to_string(),
                rule_name: DSOLVE_RULE.to_string(),
            });
        }

        // Singular rays: roots of F(v) − v = 0 give y = v0·x (dropped when
        // dividing).
        let mut singular_notes: Vec<String> = Vec::new();
        {
            let zero = ctx.num(0);
            let sing_eq = Equation {
                lhs: fv_minus_v,
                rhs: zero,
                op: RelOp::Eq,
            };
            let solver_opts = cas_solver_core::solver_options::SolverOptions::from_eval_config(
                options.shared.semantics,
                options.budget,
            );
            if let Ok((SolutionSet::Discrete(roots), _, _)) = crate::api::solve_with_display_steps(
                &sing_eq,
                &v_name,
                &mut self.simplifier,
                solver_opts,
            ) {
                let ctx = &self.simplifier.context;
                for r in roots {
                    singular_notes.push(format!(
                        "{func} = {}·{var} es solución singular (recta descartada al dividir por F(v)−v)",
                        render_expr(ctx, r)
                    ));
                }
            }
        }
        let ctx = &mut self.simplifier.context;

        // Relation G(v) = ln(x) + C; try the explicit solve for v.
        let rhs_with_c = ctx.add(cas_ast::Expr::Add(ln_side, c_var));
        let rel_eq = Equation {
            lhs: g_int,
            rhs: rhs_with_c,
            op: RelOp::Eq,
        };
        let solver_opts = cas_solver_core::solver_options::SolverOptions::from_eval_config(
            options.shared.semantics,
            options.budget,
        );
        let solve_outcome = crate::api::solve_with_display_steps(
            &rel_eq,
            &v_name,
            &mut self.simplifier,
            solver_opts,
        );
        let ctx_ref = &self.simplifier.context;
        let explicit_roots = match &solve_outcome {
            Ok((SolutionSet::Discrete(roots), _, _))
                if !roots.is_empty()
                    && roots.len() <= 2
                    && !roots
                        .iter()
                        .any(|r| contains_surd_over_c(ctx_ref, *r, c_var)) =>
            {
                Some(roots.clone())
            }
            _ => None,
        };

        let ode_residue = {
            let ctx = &mut self.simplifier.context;
            ctx.add(cas_ast::Expr::Sub(ode_eq.lhs, ode_eq.rhs))
        };

        if let Some(roots) = explicit_roots {
            // y = x·v(x), verified per branch.
            let mut verified: Vec<ExprId> = Vec::new();
            for v_sol in &roots {
                let ctx = &mut self.simplifier.context;
                let y_raw = ctx.add(cas_ast::Expr::Mul(x_var, *v_sol));
                let y_cand = self.fold_free_subtree(&verify_options, y_raw);
                let ctx = &mut self.simplifier.context;
                let substituted = substitute_power_aware(
                    ctx,
                    ode_residue,
                    y_var,
                    y_cand,
                    SubstituteOptions::exact(),
                );
                if self.reduces_to_zero_exact(&verify_options, substituted) {
                    verified.push(y_cand);
                }
            }
            if verified.len() == roots.len() {
                let ctx = &mut self.simplifier.context;
                warnings.push(DomainWarning {
                    message: format!("Solución general: {c_name} es una constante arbitraria"),
                    rule_name: DSOLVE_RULE.to_string(),
                });
                for note in &singular_notes {
                    warnings.push(DomainWarning {
                        message: note.clone(),
                        rule_name: DSOLVE_RULE.to_string(),
                    });
                }
                let solve_steps = build_homogeneous_steps(
                    ctx, ode_eq, f_v, v_var, g_int, ln_side, y_var, &verified, None, c_var,
                );
                let result = if verified.len() == 1 {
                    EvalResult::Expr(wrap_eq(ctx, y_var, verified[0]))
                } else {
                    let eqs: Vec<ExprId> =
                        verified.iter().map(|r| wrap_eq(ctx, y_var, *r)).collect();
                    EvalResult::SolutionSet(SolutionSet::Discrete(eqs))
                };
                return Ok(Some((
                    result,
                    warnings,
                    vec![],
                    solve_steps,
                    vec![],
                    vec![],
                    vec![],
                    vec![],
                )));
            }
        }

        // Implicit fallback: φ(x,y) = G(y/x) − ln(x), verified by implicit
        // differentiation (this rational route dodges the H19 surd residue).
        let ctx = &mut self.simplifier.context;
        let y_over_x = ctx.add(cas_ast::Expr::Div(y_var, x_var));
        let g_subbed =
            substitute_power_aware(ctx, g_int, v_var, y_over_x, SubstituteOptions::exact());
        let phi_raw = ctx.add(cas_ast::Expr::Sub(g_subbed, ln_side));
        let phi = self.fold_free_subtree(&verify_options, phi_raw);
        let ctx = &mut self.simplifier.context;
        let phi = clear_global_rational_factor(ctx, phi);

        let dphi_dx = ctx.call("diff", vec![phi, x_var]);
        let dphi_dy = ctx.call("diff", vec![phi, y_var]);
        let dy_term = ctx.add(cas_ast::Expr::Mul(dphi_dy, rhs));
        let implicit_residue = ctx.add(cas_ast::Expr::Add(dphi_dx, dy_term));
        if !self.reduces_to_zero_exact(&verify_options, implicit_residue) {
            let ctx = &mut self.simplifier.context;
            let r = residual_action_result(
                ctx,
                resolved,
                y_var,
                x_var,
                "La reducción homogénea no verificó ni explícita ni implícita; se declina honesto",
            );
            return Ok(Some(r));
        }
        let ctx = &mut self.simplifier.context;
        warnings.push(DomainWarning {
            message: format!(
                "Solución implícita: se emite φ({var},{func}) = {c_name} porque el despeje explícito no cierra limpio"
            ),
            rule_name: DSOLVE_RULE.to_string(),
        });
        warnings.push(DomainWarning {
            message: format!("Solución general: {c_name} es una constante arbitraria"),
            rule_name: DSOLVE_RULE.to_string(),
        });
        for note in &singular_notes {
            warnings.push(DomainWarning {
                message: note.clone(),
                rule_name: DSOLVE_RULE.to_string(),
            });
        }
        let solve_steps = build_homogeneous_steps(
            ctx,
            ode_eq,
            f_v,
            v_var,
            g_int,
            ln_side,
            y_var,
            &[],
            Some(phi),
            c_var,
        );
        let result = EvalResult::Expr(wrap_eq(ctx, phi, c_var));
        Ok(Some((
            result,
            warnings,
            vec![],
            solve_steps,
            vec![],
            vec![],
            vec![],
            vec![],
        )))
    }

    /// O5: particular solution by undetermined coefficients. The RHS is
    /// classified into UC families `P_d(x)·e^(kx)·{1|sin(bx)|cos(bx)}`, one
    /// trial block per (k, b) family with the resonance shift `x^s` (s = the
    /// EXACT characteristic multiplicity — all-rational, no root extraction),
    /// `L[trial]` is differentiated symbolically and flattened STRUCTURALLY
    /// (never through the simplifier — the fresh-constant×exp×trig family is
    /// C5-hostile), coefficients are collected per basis atom, and the linear
    /// system solves by exact rational Gauss. `None` = outside the UC table.
    fn uc_particular(
        &mut self,
        verify_options: &crate::options::EvalOptions,
        second: &SecondOrderOde,
        rhs: ExprId,
        func: &str,
        var: &str,
        resolved: ExprId,
    ) -> Option<ExprId> {
        let p = &second.b / &second.a;
        let q = &second.c / &second.a;
        // Normalize the forcing to the monic operator: y'' + p·y' + q·y = rhs/a.
        let inv_a = num_rational::BigRational::from_integer(1.into()) / &second.a;

        // Classify the RHS.
        let ctx = &mut self.simplifier.context;
        let rhs_dist = distribute_structural(ctx, rhs);
        let mut rhs_map: std::collections::HashMap<UcBase, UcComb> =
            std::collections::HashMap::new();
        if !uc_decompose(ctx, rhs_dist, var, &[], &inv_a, &mut rhs_map) {
            return None;
        }
        if rhs_map.is_empty() {
            return None;
        }

        // Group by (k, |b|) family; track max polynomial degree per family.
        #[derive(Clone, PartialEq, Eq, Hash)]
        struct Family {
            k: num_rational::BigRational,
            b: Option<num_rational::BigRational>,
        }
        let mut families: std::collections::HashMap<Family, usize> =
            std::collections::HashMap::new();
        for base in rhs_map.keys() {
            let fam = Family {
                k: base.k.clone(),
                b: match &base.trig {
                    UcTrig::None => None,
                    UcTrig::Sin(b) | UcTrig::Cos(b) => Some(b.clone()),
                },
            };
            let entry = families.entry(fam).or_insert(0);
            *entry = (*entry).max(base.x_pow);
        }

        // Fresh coefficient symbols (D7 freshness against the ODE's vars).
        let ctx = &mut self.simplifier.context;
        let input_vars = collect_variables(ctx, resolved);
        let prefix = if input_vars.iter().any(|v| v.starts_with("uc")) {
            "ucoef"
        } else {
            "uc"
        };
        let mut fresh_syms: Vec<cas_ast::symbol::SymbolId> = Vec::new();
        let mut fresh_vars: Vec<ExprId> = Vec::new();
        let next_fresh = |ctx: &mut Context,
                          fresh_syms: &mut Vec<cas_ast::symbol::SymbolId>,
                          fresh_vars: &mut Vec<ExprId>|
         -> ExprId {
            let name = format!("{prefix}{}", fresh_syms.len());
            let sym = ctx.intern_symbol(&name);
            let v = ctx.var(&name);
            fresh_syms.push(sym);
            fresh_vars.push(v);
            v
        };

        // Build the trial: Σ_families x^s · (Σ_j u_j·x^j) · e^(kx) · trig-part.
        let x_var = ctx.var(var);
        let e_const = ctx.add(cas_ast::Expr::Constant(cas_ast::Constant::E));
        let mut trial: Option<ExprId> = None;
        // Deterministic family order for reproducible fresh-symbol numbering.
        let mut fam_list: Vec<(Family, usize)> = families.into_iter().collect();
        fam_list.sort_by_key(|(f, _)| {
            (
                f.k.clone(),
                f.b.clone()
                    .unwrap_or_else(|| num_rational::BigRational::from_integer(0.into())),
                f.b.is_some(),
            )
        });
        for (fam, degree) in fam_list {
            let s = match &fam.b {
                None => real_char_multiplicity(&p, &q, &fam.k),
                Some(b) => {
                    if complex_char_is_root(&p, &q, &fam.k, b) {
                        1
                    } else {
                        0
                    }
                }
            };
            // Polynomial block(s): one for the no-trig case, two (cos+sin)
            // when b is present.
            let blocks: Vec<Option<UcTrig>> = match &fam.b {
                None => vec![None],
                Some(b) => vec![Some(UcTrig::Cos(b.clone())), Some(UcTrig::Sin(b.clone()))],
            };
            for block_trig in blocks {
                let mut poly: Option<ExprId> = None;
                for j in 0..=degree {
                    let u = next_fresh(
                        &mut self.simplifier.context,
                        &mut fresh_syms,
                        &mut fresh_vars,
                    );
                    let ctx = &mut self.simplifier.context;
                    let total_pow = j + s;
                    let term = if total_pow == 0 {
                        u
                    } else {
                        let xp = if total_pow == 1 {
                            x_var
                        } else {
                            let n = ctx.num(total_pow as i64);
                            ctx.add(cas_ast::Expr::Pow(x_var, n))
                        };
                        ctx.add(cas_ast::Expr::Mul(u, xp))
                    };
                    poly = Some(match poly {
                        None => term,
                        Some(acc) => {
                            let ctx = &mut self.simplifier.context;
                            ctx.add(cas_ast::Expr::Add(acc, term))
                        }
                    });
                }
                let ctx = &mut self.simplifier.context;
                let mut block = poly.expect("degree ≥ 0 always yields a term");
                if !fam.k.is_zero() {
                    let k_num = ctx.add(cas_ast::Expr::Number(fam.k.clone()));
                    let arg = ctx.add(cas_ast::Expr::Mul(k_num, x_var));
                    let envelope = ctx.add(cas_ast::Expr::Pow(e_const, arg));
                    block = ctx.add(cas_ast::Expr::Mul(block, envelope));
                }
                if let Some(trig) = block_trig {
                    let (name, b) = match trig {
                        UcTrig::Cos(b) => ("cos", b),
                        UcTrig::Sin(b) => ("sin", b),
                        UcTrig::None => unreachable!(),
                    };
                    let b_num = ctx.add(cas_ast::Expr::Number(b));
                    let arg = ctx.add(cas_ast::Expr::Mul(b_num, x_var));
                    let t = ctx.call(name, vec![arg]);
                    block = ctx.add(cas_ast::Expr::Mul(block, t));
                }
                trial = Some(match trial {
                    None => block,
                    Some(acc) => {
                        let ctx = &mut self.simplifier.context;
                        ctx.add(cas_ast::Expr::Add(acc, block))
                    }
                });
            }
        }
        let trial = trial?;

        // L[trial]/a = trial'' + p·trial' + q·trial, differentiated by the
        // PURE symbolic differ and flattened structurally.
        let ctx = &mut self.simplifier.context;
        let d1 = cas_math::symbolic_differentiation_support::differentiate_symbolic_expr(
            ctx, trial, var,
        )?;
        let d2 =
            cas_math::symbolic_differentiation_support::differentiate_symbolic_expr(ctx, d1, var)?;
        let one = num_rational::BigRational::from_integer(1.into());
        let mut lhs_map: std::collections::HashMap<UcBase, UcComb> =
            std::collections::HashMap::new();
        let d2_dist = distribute_structural(ctx, d2);
        if !uc_decompose(ctx, d2_dist, var, &fresh_syms, &one, &mut lhs_map) {
            return None;
        }
        let d1_dist = distribute_structural(ctx, d1);
        if !uc_decompose(ctx, d1_dist, var, &fresh_syms, &p, &mut lhs_map) {
            return None;
        }
        let trial_dist = distribute_structural(ctx, trial);
        if !uc_decompose(ctx, trial_dist, var, &fresh_syms, &q, &mut lhs_map) {
            return None;
        }

        // Assemble the linear system over every basis atom seen on either side.
        let mut all_bases: Vec<UcBase> = lhs_map
            .keys()
            .chain(rhs_map.keys())
            .cloned()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        all_bases.sort_by_key(|b| {
            (
                b.k.clone(),
                match &b.trig {
                    UcTrig::None => (0u8, num_rational::BigRational::from_integer(0.into())),
                    UcTrig::Cos(v) => (1u8, v.clone()),
                    UcTrig::Sin(v) => (2u8, v.clone()),
                },
                b.x_pow,
            )
        });
        let n = fresh_syms.len();
        let mut matrix: Vec<Vec<num_rational::BigRational>> = Vec::new();
        let mut rhs_vec: Vec<num_rational::BigRational> = Vec::new();
        for base in &all_bases {
            let row = lhs_map
                .get(base)
                .map(|c| c.fresh.clone())
                .unwrap_or_else(|| UcComb::zero(n).fresh);
            let mut target = rhs_map
                .get(base)
                .map(|c| c.constant.clone())
                .unwrap_or_else(|| num_rational::BigRational::from_integer(0.into()));
            if let Some(c) = lhs_map.get(base) {
                // L[trial] is linear-homogeneous in the fresh symbols; a
                // nonzero constant here would be a collector bug — decline.
                if !c.constant.is_zero() {
                    return None;
                }
            }
            target = target.clone();
            matrix.push(row);
            rhs_vec.push(target);
        }
        let solution = solve_rational_system(matrix, rhs_vec)?;

        // Substitute the solved coefficients into the trial and fold (numeric
        // constants only — no symbolic-constant hostility).
        let mut y_p = trial;
        for (idx, value) in solution.iter().enumerate() {
            let ctx = &mut self.simplifier.context;
            let v_num = ctx.add(cas_ast::Expr::Number(value.clone()));
            y_p = substitute_power_aware(
                ctx,
                y_p,
                fresh_vars[idx],
                v_num,
                SubstituteOptions::exact(),
            );
        }
        let y_p = self.fold_free_subtree(verify_options, y_p);

        // Affine verification gate (D5): L[y_p] − rhs must reduce to exact 0
        // — the final net over the whole collector/Gauss pipeline.
        let ctx = &mut self.simplifier.context;
        let a_num = ctx.add(cas_ast::Expr::Number(second.a.clone()));
        let b_num = ctx.add(cas_ast::Expr::Number(second.b.clone()));
        let c_num = ctx.add(cas_ast::Expr::Number(second.c.clone()));
        let d1_call = ctx.call("diff", vec![y_p, x_var]);
        let two_lit = ctx.num(2);
        let d2_call = ctx.call("diff", vec![y_p, x_var, two_lit]);
        let t2 = ctx.add(cas_ast::Expr::Mul(a_num, d2_call));
        let t1 = ctx.add(cas_ast::Expr::Mul(b_num, d1_call));
        let t0 = ctx.add(cas_ast::Expr::Mul(c_num, y_p));
        let sum01 = ctx.add(cas_ast::Expr::Add(t2, t1));
        let l_yp = ctx.add(cas_ast::Expr::Add(sum01, t0));
        let residue = ctx.add(cas_ast::Expr::Sub(l_yp, rhs));
        if !self.reduces_to_zero_exact(verify_options, residue) {
            return None;
        }
        let _ = func;
        Some(y_p)
    }

    fn reconstruct_potential_full_eval(
        &mut self,
        verify_options: &crate::options::EvalOptions,
        m: ExprId,
        n: ExprId,
        func: &str,
        var: &str,
    ) -> Option<ExprId> {
        let ctx = &mut self.simplifier.context;
        let y_var = ctx.var(func);
        let phi_x = match crate::rules::calculus::integrate_with_trace(ctx, m, var) {
            Some(outcome) if outcome.required_conditions.is_empty() => outcome.result,
            _ => return None,
        };
        let dphi_dy_call = ctx.call("diff", vec![phi_x, y_var]);
        let dphi_dy = self.fold_free_subtree(verify_options, dphi_dy_call);
        let ctx = &mut self.simplifier.context;
        let rest_raw = ctx.add(cas_ast::Expr::Sub(n, dphi_dy));
        let rest = self.fold_free_subtree(verify_options, rest_raw);
        let ctx = &mut self.simplifier.context;
        // For an exact field the rest is a function of y alone; an x left
        // over means non-exact → decline.
        if collect_variables(ctx, rest).contains(var) {
            return None;
        }
        let piece = if matches!(ctx.get(rest), cas_ast::Expr::Number(v) if v.is_zero()) {
            None
        } else {
            match crate::rules::calculus::integrate_with_trace(ctx, rest, func) {
                Some(outcome) if outcome.required_conditions.is_empty() => Some(outcome.result),
                _ => return None,
            }
        };
        let ctx = &mut self.simplifier.context;
        Some(match piece {
            Some(p) => ctx.add(cas_ast::Expr::Add(phi_x, p)),
            None => phi_x,
        })
    }

    /// O3: pin the arbitrary constant of a general solution with one initial
    /// condition `y(x0) = y0`. Explicit forms substitute the point and solve
    /// for the constant (each root is verified against BOTH the ODE and the
    /// condition before emission — D5 twice); the implicit form `φ(x,y) = C`
    /// evaluates `C = φ(x0, y0)` directly. Any failure declines to the honest
    /// residual — a general solution is NEVER emitted as if the condition had
    /// been applied.
    #[allow(clippy::too_many_arguments)]
    fn maybe_apply_initial_condition(
        &mut self,
        options: &crate::options::EvalOptions,
        general: ActionResult,
        cond: Option<&InitialCondition>,
        resolved: ExprId,
        func: &str,
        var: &str,
        ode_eq: &Equation,
    ) -> ActionResult {
        let Some(cond) = cond else { return general };
        let (result, warnings, steps, mut solve_steps, a4, a5, a6, a7) = general;
        if matches!(result, EvalResult::SolutionSet(SolutionSet::Residual(_))) {
            // The general solve already declined; keep the honest residual.
            return (result, warnings, steps, solve_steps, a4, a5, a6, a7);
        }

        let mut verify_options = options.clone();
        verify_options.steps_mode = cas_solver_core::eval_options::StepsMode::Off;
        verify_options.time_budget_ms = Some(
            verify_options
                .time_budget_ms
                .map_or(VERIFY_TIME_BUDGET_MS, |t| t.min(VERIFY_TIME_BUDGET_MS)),
        );

        let ctx = &mut self.simplifier.context;
        let y_var = ctx.var(func);
        let x_var = ctx.var(var);
        let mk_residual = |ctx: &mut Context, reason: &str| -> ActionResult {
            residual_action_result(ctx, resolved, y_var, x_var, reason)
        };

        enum Shape {
            Explicit(Vec<ExprId>),
            Implicit { phi: ExprId },
        }
        let shape = match &result {
            EvalResult::Expr(eq_id) => match cas_ast::eq::unwrap_eq(ctx, *eq_id) {
                Some((lhs, rhs)) if lhs == y_var => Shape::Explicit(vec![rhs]),
                Some((lhs, _)) => Shape::Implicit { phi: lhs },
                None => {
                    return mk_residual(
                        ctx,
                        "La condición inicial no pudo aplicarse sobre la forma emitida; se declina honesto",
                    )
                }
            },
            EvalResult::SolutionSet(SolutionSet::Discrete(eqs)) => {
                let mut hs = Vec::new();
                for e in eqs {
                    match cas_ast::eq::unwrap_eq(ctx, *e) {
                        Some((lhs, rhs)) if lhs == y_var => hs.push(rhs),
                        _ => {
                            return mk_residual(
                                ctx,
                                "La condición inicial no pudo aplicarse sobre la forma emitida; se declina honesto",
                            )
                        }
                    }
                }
                Shape::Explicit(hs)
            }
            _ => {
                return mk_residual(
                    ctx,
                    "La condición inicial no pudo aplicarse sobre la forma emitida; se declina honesto",
                )
            }
        };

        // Drop the "arbitrary constant" warnings: the constant gets pinned.
        let warnings: Vec<DomainWarning> = warnings
            .into_iter()
            .filter(|w| {
                !w.message.contains("constante arbitraria")
                    && !w.message.contains("se absorbe en C")
            })
            .collect();

        match shape {
            Shape::Implicit { phi } => {
                // C = φ(x0, y0), evaluated exactly.
                let ctx = &mut self.simplifier.context;
                let at_x =
                    substitute_power_aware(ctx, phi, x_var, cond.point, SubstituteOptions::exact());
                let at_xy = substitute_power_aware(
                    ctx,
                    at_x,
                    y_var,
                    cond.value,
                    SubstituteOptions::exact(),
                );
                let c_star = self.fold_free_subtree(&verify_options, at_xy);
                let ctx = &mut self.simplifier.context;
                let c_vars = collect_variables(ctx, c_star);
                if c_vars.contains(func) || c_vars.contains(var) {
                    return mk_residual(
                        ctx,
                        "La condición inicial no fijó la constante de la solución implícita; se declina honesto",
                    );
                }
                let point_str = render_expr(ctx, cond.point);
                let value_str = render_expr(ctx, cond.value);
                solve_steps.push(crate::api::SolveStep {
                    description: format!(
                        "Aplicar la condición inicial {func}({point_str}) = {value_str}: sustituir el punto y fijar la constante"
                    ),
                    equation_after: Equation {
                        lhs: phi,
                        rhs: c_star,
                        op: RelOp::Eq,
                    },
                    importance: ImportanceLevel::High,
                    substeps: vec![],
                });
                let result = EvalResult::Expr(wrap_eq(ctx, phi, c_star));
                (result, warnings, steps, solve_steps, a4, a5, a6, a7)
            }
            Shape::Explicit(hs) => {
                // The constant is the ONE variable of the solution that is
                // neither the unknown, the independent variable, nor a symbol
                // of the ODE itself.
                let ctx = &mut self.simplifier.context;
                let ode_vars = collect_variables(ctx, resolved);
                let mut const_names: Vec<String> = Vec::new();
                for h in &hs {
                    for v in collect_variables(ctx, *h) {
                        if v != func
                            && v != var
                            && !ode_vars.contains(&v)
                            && !const_names.contains(&v)
                        {
                            const_names.push(v);
                        }
                    }
                }
                if const_names.len() != 1 {
                    return mk_residual(
                        ctx,
                        "La condición inicial no pudo aislar la constante de la solución general; se declina honesto",
                    );
                }
                let c_name = const_names.remove(0);
                let c_var = ctx.var(&c_name);

                let mut winners: Vec<(ExprId, ExprId)> = Vec::new(); // (y_p, c_star)
                for h in &hs {
                    let ctx = &mut self.simplifier.context;
                    let h_at = substitute_power_aware(
                        ctx,
                        *h,
                        x_var,
                        cond.point,
                        SubstituteOptions::exact(),
                    );
                    let solve_eq = Equation {
                        lhs: h_at,
                        rhs: cond.value,
                        op: RelOp::Eq,
                    };
                    let solver_opts =
                        cas_solver_core::solver_options::SolverOptions::from_eval_config(
                            options.shared.semantics,
                            options.budget,
                        );
                    let Ok((SolutionSet::Discrete(c_roots), _, _)) =
                        crate::api::solve_with_display_steps(
                            &solve_eq,
                            &c_name,
                            &mut self.simplifier,
                            solver_opts,
                        )
                    else {
                        continue;
                    };
                    for c_star in c_roots {
                        let ctx = &mut self.simplifier.context;
                        let y_p_raw = substitute_power_aware(
                            ctx,
                            *h,
                            c_var,
                            c_star,
                            SubstituteOptions::exact(),
                        );
                        // Selective fold (the O1 lesson): folding an Add
                        // re-joins the split fractions; fold only non-Add
                        // shapes so `1/e^x + x - 1` keeps its textbook form.
                        let y_p = if matches!(
                            self.simplifier.context.get(y_p_raw),
                            cas_ast::Expr::Add(..)
                        ) {
                            y_p_raw
                        } else {
                            self.fold_free_subtree(&verify_options, y_p_raw)
                        };
                        // Gate 1: the particular solution verifies the ODE.
                        let ctx = &mut self.simplifier.context;
                        let ode_residue = ctx.add(cas_ast::Expr::Sub(ode_eq.lhs, ode_eq.rhs));
                        let substituted = substitute_power_aware(
                            ctx,
                            ode_residue,
                            y_var,
                            y_p,
                            SubstituteOptions::exact(),
                        );
                        if !self.reduces_to_zero_exact(&verify_options, substituted) {
                            continue;
                        }
                        // Gate 2: the condition holds exactly.
                        let ctx = &mut self.simplifier.context;
                        let y_p_at = substitute_power_aware(
                            ctx,
                            y_p,
                            x_var,
                            cond.point,
                            SubstituteOptions::exact(),
                        );
                        let cond_residue = ctx.add(cas_ast::Expr::Sub(y_p_at, cond.value));
                        if !self.reduces_to_zero_exact(&verify_options, cond_residue) {
                            continue;
                        }
                        winners.push((y_p, c_star));
                        break;
                    }
                }
                if winners.is_empty() {
                    let ctx = &mut self.simplifier.context;
                    return mk_residual(
                        ctx,
                        "La condición inicial es inconsistente con la familia general (o apunta a una solución singular); se declina honesto",
                    );
                }
                let ctx = &mut self.simplifier.context;
                let point_str = render_expr(ctx, cond.point);
                let value_str = render_expr(ctx, cond.value);
                let (first_yp, first_c) = winners[0];
                solve_steps.push(crate::api::SolveStep {
                    description: format!(
                        "Aplicar la condición inicial {func}({point_str}) = {value_str}: sustituir el punto y fijar la constante"
                    ),
                    equation_after: Equation {
                        lhs: c_var,
                        rhs: first_c,
                        op: RelOp::Eq,
                    },
                    importance: ImportanceLevel::High,
                    substeps: vec![],
                });
                solve_steps.push(crate::api::SolveStep {
                    description: "Solución particular con la condición aplicada".to_string(),
                    equation_after: Equation {
                        lhs: y_var,
                        rhs: first_yp,
                        op: RelOp::Eq,
                    },
                    importance: ImportanceLevel::High,
                    substeps: vec![],
                });
                let result = if winners.len() == 1 {
                    EvalResult::Expr(wrap_eq(ctx, y_var, first_yp))
                } else {
                    let eqs: Vec<ExprId> = winners
                        .iter()
                        .map(|(y_p, _)| wrap_eq(ctx, y_var, *y_p))
                        .collect();
                    EvalResult::SolutionSet(SolutionSet::Discrete(eqs))
                };
                (result, warnings, steps, solve_steps, a4, a5, a6, a7)
            }
        }
    }
}

/// Narrated solve steps for the separable method (D13). Every description
/// template must have an es/en entry in `SOLVE_DESCRIPTIONS`.
#[allow(clippy::too_many_arguments)]
fn build_separable_steps(
    ctx: &mut Context,
    ode_eq: &Equation,
    fx: ExprId,
    gy: Option<ExprId>,
    integrated_eq: &Equation,
    y_var: ExprId,
    explicit: &[ExprId],
    implicit_phi: Option<ExprId>,
    c_var: ExprId,
) -> Vec<crate::api::SolveStep> {
    let one = ctx.num(1);
    let g_shown = gy.unwrap_or(one);
    let mut steps: Vec<crate::api::SolveStep> = Vec::new();
    steps.push(crate::api::SolveStep {
        description: format!(
            "Identificar EDO separable: y' = f(x)·g(y) con f = {}, g = {}",
            render_expr(ctx, fx),
            render_expr(ctx, g_shown)
        ),
        equation_after: ode_eq.clone(),
        importance: ImportanceLevel::High,
        substeps: vec![],
    });
    let sep_lhs = ctx.add(cas_ast::Expr::Div(one, g_shown));
    steps.push(crate::api::SolveStep {
        description: "Separar las variables: dy/g(y) = f(x)·dx".to_string(),
        equation_after: Equation {
            lhs: sep_lhs,
            rhs: fx,
            op: RelOp::Eq,
        },
        importance: ImportanceLevel::Medium,
        substeps: vec![],
    });
    steps.push(crate::api::SolveStep {
        description: "Integrar ambos lados de la ecuación separada".to_string(),
        equation_after: integrated_eq.clone(),
        importance: ImportanceLevel::High,
        substeps: vec![],
    });
    if let Some(phi) = implicit_phi {
        steps.push(crate::api::SolveStep {
            description: "Combinar en una solución implícita φ(x,y) = C".to_string(),
            equation_after: Equation {
                lhs: phi,
                rhs: c_var,
                op: RelOp::Eq,
            },
            importance: ImportanceLevel::High,
            substeps: vec![],
        });
    } else if let Some(first) = explicit.first() {
        steps.push(crate::api::SolveStep {
            description: "Despejar la incógnita de la relación integrada".to_string(),
            equation_after: Equation {
                lhs: y_var,
                rhs: *first,
                op: RelOp::Eq,
            },
            importance: ImportanceLevel::High,
            substeps: vec![],
        });
    }
    steps.push(crate::api::SolveStep {
        description: "Verificar por sustitución: el residuo de la EDO se reduce a 0".to_string(),
        equation_after: if let Some(phi) = implicit_phi {
            Equation {
                lhs: phi,
                rhs: c_var,
                op: RelOp::Eq,
            }
        } else {
            Equation {
                lhs: y_var,
                rhs: explicit.first().copied().unwrap_or(y_var),
                op: RelOp::Eq,
            }
        },
        importance: ImportanceLevel::Medium,
        substeps: vec![],
    });
    steps
}

/// Narrated solve steps for the linear first-order method (D13). Every
/// description template must have an es/en entry in `SOLVE_DESCRIPTIONS`.
#[allow(clippy::too_many_arguments)]
fn build_linear_steps(
    ctx: &mut Context,
    ode_eq: &Equation,
    p: ExprId,
    q: ExprId,
    mu: ExprId,
    g_int: ExprId,
    y_var: ExprId,
    x_var: ExprId,
    candidate: ExprId,
    c_var: ExprId,
) -> Vec<crate::api::SolveStep> {
    let mut steps: Vec<crate::api::SolveStep> = Vec::new();
    steps.push(crate::api::SolveStep {
        description: format!(
            "Identificar forma lineal: y' + p·y = q con p = {}, q = {}",
            render_expr(ctx, p),
            render_expr(ctx, q)
        ),
        equation_after: ode_eq.clone(),
        importance: ImportanceLevel::High,
        substeps: vec![],
    });
    steps.push(crate::api::SolveStep {
        description: format!(
            "Calcular el factor integrante: μ = e^(∫p dx) = {}",
            render_expr(ctx, mu)
        ),
        equation_after: ode_eq.clone(),
        importance: ImportanceLevel::High,
        substeps: vec![],
    });
    // (μ·y)' = μ·q, shown with a raw diff call (readable equation state).
    let mu_y = ctx.add(cas_ast::Expr::Mul(mu, y_var));
    let d_mu_y = ctx.call("diff", vec![mu_y, x_var]);
    let mu_q = ctx.add(cas_ast::Expr::Mul(mu, q));
    steps.push(crate::api::SolveStep {
        description: "Multiplicar por μ: el lado izquierdo se vuelve la derivada del producto μ·y"
            .to_string(),
        equation_after: Equation {
            lhs: d_mu_y,
            rhs: mu_q,
            op: RelOp::Eq,
        },
        importance: ImportanceLevel::Medium,
        substeps: vec![],
    });
    let rhs_int = ctx.add(cas_ast::Expr::Add(g_int, c_var));
    steps.push(crate::api::SolveStep {
        description: "Integrar ambos lados: μ·y = ∫ μ·q dx + C".to_string(),
        equation_after: Equation {
            lhs: mu_y,
            rhs: rhs_int,
            op: RelOp::Eq,
        },
        importance: ImportanceLevel::High,
        substeps: vec![],
    });
    steps.push(crate::api::SolveStep {
        description: "Despejar la incógnita de la relación integrada".to_string(),
        equation_after: Equation {
            lhs: y_var,
            rhs: candidate,
            op: RelOp::Eq,
        },
        importance: ImportanceLevel::High,
        substeps: vec![],
    });
    steps.push(crate::api::SolveStep {
        description: "Verificar por sustitución: el residuo de la EDO se reduce a 0".to_string(),
        equation_after: Equation {
            lhs: y_var,
            rhs: candidate,
            op: RelOp::Eq,
        },
        importance: ImportanceLevel::Medium,
        substeps: vec![],
    });
    steps
}

/// Narrated solve steps for the exact method (D13). Every description
/// template must have an es/en entry in `SOLVE_DESCRIPTIONS`.
fn build_exact_steps(
    ctx: &mut Context,
    m: ExprId,
    n: ExprId,
    phi: ExprId,
    y_var: ExprId,
    x_var: ExprId,
    c_var: ExprId,
) -> Vec<crate::api::SolveStep> {
    let mut steps: Vec<crate::api::SolveStep> = Vec::new();
    let diff_call =
        |ctx: &mut Context, target: ExprId, v: ExprId| ctx.call("diff", vec![target, v]);
    let n_yprime = {
        let dy = diff_call(ctx, y_var, x_var);
        ctx.add(cas_ast::Expr::Mul(n, dy))
    };
    let ode_shown = ctx.add(cas_ast::Expr::Add(m, n_yprime));
    let zero = ctx.num(0);
    steps.push(crate::api::SolveStep {
        description: format!(
            "Identificar forma exacta: M + N·y' = 0 con M = {}, N = {}",
            render_expr(ctx, m),
            render_expr(ctx, n)
        ),
        equation_after: Equation {
            lhs: ode_shown,
            rhs: zero,
            op: RelOp::Eq,
        },
        importance: ImportanceLevel::High,
        substeps: vec![],
    });
    let dm_dy = diff_call(ctx, m, y_var);
    let dn_dx = diff_call(ctx, n, x_var);
    steps.push(crate::api::SolveStep {
        description: "Comprobar exactitud: ∂M/∂y = ∂N/∂x (el campo (M, N) es conservativo)"
            .to_string(),
        equation_after: Equation {
            lhs: dm_dy,
            rhs: dn_dx,
            op: RelOp::Eq,
        },
        importance: ImportanceLevel::Medium,
        substeps: vec![],
    });
    steps.push(crate::api::SolveStep {
        description: "Reconstruir el potencial: φ = ∫M dx + h(y) con h'(y) ajustando ∂φ/∂y = N"
            .to_string(),
        equation_after: Equation {
            lhs: phi,
            rhs: phi,
            op: RelOp::Eq,
        },
        importance: ImportanceLevel::High,
        substeps: vec![],
    });
    steps.push(crate::api::SolveStep {
        description: "Combinar en una solución implícita φ(x,y) = C".to_string(),
        equation_after: Equation {
            lhs: phi,
            rhs: c_var,
            op: RelOp::Eq,
        },
        importance: ImportanceLevel::High,
        substeps: vec![],
    });
    steps.push(crate::api::SolveStep {
        description: "Verificar el potencial: ∂φ/∂x = M y ∂φ/∂y = N (residuos exactos a 0)"
            .to_string(),
        equation_after: Equation {
            lhs: phi,
            rhs: c_var,
            op: RelOp::Eq,
        },
        importance: ImportanceLevel::Medium,
        substeps: vec![],
    });
    steps
}

/// Which characteristic-discriminant branch produced the basis.
enum SecondOrderShape {
    DistinctReal,
    DoubleRoot { rate: num_rational::BigRational },
    ComplexPair { alpha: num_rational::BigRational },
}

/// Narrated solve steps for the second-order constant-coefficient method
/// (D13). Every template must have an es/en entry in `SOLVE_DESCRIPTIONS`.
#[allow(clippy::too_many_arguments)]
fn build_second_order_steps(
    ctx: &mut Context,
    func: &str,
    var: &str,
    second: &SecondOrderOde,
    disc: &num_rational::BigRational,
    shape: &SecondOrderShape,
    u1: ExprId,
    u2: ExprId,
    y_var: ExprId,
    general: ExprId,
) -> Vec<crate::api::SolveStep> {
    // Characteristic polynomial in a fresh symbol (r, or s when taken).
    let r_name = if func == "r" || var == "r" { "s" } else { "r" };
    let r_var = ctx.var(r_name);
    let a_num = ctx.add(cas_ast::Expr::Number(second.a.clone()));
    let b_num = ctx.add(cas_ast::Expr::Number(second.b.clone()));
    let c_num = ctx.add(cas_ast::Expr::Number(second.c.clone()));
    let two = ctx.num(2);
    let r_sq = ctx.add(cas_ast::Expr::Pow(r_var, two));
    let ar2 = ctx.add(cas_ast::Expr::Mul(a_num, r_sq));
    let br = ctx.add(cas_ast::Expr::Mul(b_num, r_var));
    let sum1 = ctx.add(cas_ast::Expr::Add(ar2, br));
    let poly = ctx.add(cas_ast::Expr::Add(sum1, c_num));
    let zero = ctx.num(0);
    let char_eq = Equation {
        lhs: poly,
        rhs: zero,
        op: RelOp::Eq,
    };

    let mut steps: Vec<crate::api::SolveStep> = Vec::new();
    steps.push(crate::api::SolveStep {
        description: format!(
            "Plantear la ecuación característica: a·r² + b·r + c = 0 con a = {}, b = {}, c = {}",
            second.a, second.b, second.c
        ),
        equation_after: char_eq.clone(),
        importance: ImportanceLevel::High,
        substeps: vec![],
    });
    let disc_num = ctx.add(cas_ast::Expr::Number(disc.clone()));
    let disc_var = ctx.var("Δ");
    steps.push(crate::api::SolveStep {
        description: format!("Calcular el discriminante de la característica: Δ = {disc}"),
        equation_after: Equation {
            lhs: disc_var,
            rhs: disc_num,
            op: RelOp::Eq,
        },
        importance: ImportanceLevel::Medium,
        substeps: vec![],
    });
    let branch_desc = match shape {
        SecondOrderShape::DistinctReal => {
            "Raíces reales distintas (Δ > 0): la base es {e^(r1·x), e^(r2·x)}"
        }
        SecondOrderShape::DoubleRoot { .. } => {
            "Raíz real doble (Δ = 0): la base es {e^(r·x), x·e^(r·x)}"
        }
        SecondOrderShape::ComplexPair { .. } => {
            "Raíces complejas conjugadas (Δ < 0): la base es {e^(α·x)·cos(β·x), e^(α·x)·sin(β·x)}"
        }
    };
    let u_sum = ctx.add(cas_ast::Expr::Add(u1, u2));
    steps.push(crate::api::SolveStep {
        description: branch_desc.to_string(),
        equation_after: Equation {
            lhs: y_var,
            rhs: u_sum,
            op: RelOp::Eq,
        },
        importance: ImportanceLevel::High,
        substeps: vec![],
    });
    steps.push(crate::api::SolveStep {
        description: "Solución general: combinación lineal de la base con C1 y C2".to_string(),
        equation_after: Equation {
            lhs: y_var,
            rhs: general,
            op: RelOp::Eq,
        },
        importance: ImportanceLevel::High,
        substeps: vec![],
    });
    steps.push(crate::api::SolveStep {
        description: "Verificar por sustitución: cada función de la base anula la EDO".to_string(),
        equation_after: Equation {
            lhs: y_var,
            rhs: general,
            op: RelOp::Eq,
        },
        importance: ImportanceLevel::Medium,
        substeps: vec![],
    });
    steps
}

// ===================== O5: undetermined coefficients =====================

/// Structural distribution of products over sums (`a·(b+c) → a·b + a·c`,
/// recursive, Sub/Neg normalized into Add-of-Neg). Purely structural — never
/// touches the simplifier, so trial derivatives (fresh-constant × exp × trig,
/// the C5-hostile family) flatten deterministically without oscillation risk.
fn distribute_structural(ctx: &mut Context, e: ExprId) -> ExprId {
    match ctx.get(e).clone() {
        cas_ast::Expr::Add(a, b) => {
            let da = distribute_structural(ctx, a);
            let db = distribute_structural(ctx, b);
            ctx.add(cas_ast::Expr::Add(da, db))
        }
        cas_ast::Expr::Sub(a, b) => {
            let da = distribute_structural(ctx, a);
            let db = distribute_structural(ctx, b);
            let nb = ctx.add(cas_ast::Expr::Neg(db));
            ctx.add(cas_ast::Expr::Add(da, nb))
        }
        cas_ast::Expr::Neg(a) => {
            let da = distribute_structural(ctx, a);
            match ctx.get(da).clone() {
                cas_ast::Expr::Add(x, y) => {
                    let nx = ctx.add(cas_ast::Expr::Neg(x));
                    let ny = ctx.add(cas_ast::Expr::Neg(y));
                    let inner = ctx.add(cas_ast::Expr::Add(nx, ny));
                    distribute_structural(ctx, inner)
                }
                _ => ctx.add(cas_ast::Expr::Neg(da)),
            }
        }
        cas_ast::Expr::Mul(a, b) => {
            let da = distribute_structural(ctx, a);
            let db = distribute_structural(ctx, b);
            if let cas_ast::Expr::Add(x, y) = ctx.get(da).clone() {
                let mx = ctx.add(cas_ast::Expr::Mul(x, db));
                let my = ctx.add(cas_ast::Expr::Mul(y, db));
                let sum = ctx.add(cas_ast::Expr::Add(mx, my));
                return distribute_structural(ctx, sum);
            }
            if let cas_ast::Expr::Add(x, y) = ctx.get(db).clone() {
                let mx = ctx.add(cas_ast::Expr::Mul(da, x));
                let my = ctx.add(cas_ast::Expr::Mul(da, y));
                let sum = ctx.add(cas_ast::Expr::Add(mx, my));
                return distribute_structural(ctx, sum);
            }
            ctx.add(cas_ast::Expr::Mul(da, db))
        }
        cas_ast::Expr::Div(a, b) => {
            let da = distribute_structural(ctx, a);
            let db = distribute_structural(ctx, b);
            if let cas_ast::Expr::Add(x, y) = ctx.get(da).clone() {
                let dx = ctx.add(cas_ast::Expr::Div(x, db));
                let dy = ctx.add(cas_ast::Expr::Div(y, db));
                let sum = ctx.add(cas_ast::Expr::Add(dx, dy));
                return distribute_structural(ctx, sum);
            }
            ctx.add(cas_ast::Expr::Div(da, db))
        }
        _ => e,
    }
}

/// One UC basis atom `x^pow · e^(k·x) · {1 | sin(b·x) | cos(b·x)}`.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
enum UcTrig {
    None,
    Sin(num_rational::BigRational),
    Cos(num_rational::BigRational),
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct UcBase {
    x_pow: usize,
    k: num_rational::BigRational,
    trig: UcTrig,
}

/// Linear combination `const + Σ coef_j · fresh_j` attached to one UC base.
#[derive(Clone, Debug)]
struct UcComb {
    constant: num_rational::BigRational,
    fresh: Vec<num_rational::BigRational>,
}

impl UcComb {
    fn zero(n: usize) -> Self {
        let z = num_rational::BigRational::from_integer(0.into());
        UcComb {
            constant: z.clone(),
            fresh: vec![z; n],
        }
    }
}

/// Extract the rational slope `m` of a linear-in-`var` argument `m·x` (no
/// additive constant tolerated — UC arguments are pure rates).
fn linear_rate(ctx: &Context, e: ExprId, var: &str) -> Option<num_rational::BigRational> {
    let one = num_rational::BigRational::from_integer(1.into());
    let mut terms: Vec<(ExprId, num_rational::BigRational)> = Vec::new();
    collect_linear_terms(ctx, e, one, &mut terms);
    let mut rate: Option<num_rational::BigRational> = None;
    for (t, m) in terms {
        match ctx.get(t) {
            cas_ast::Expr::Variable(s) if ctx.sym_name(*s) == var => {
                let acc = rate
                    .take()
                    .unwrap_or_else(|| num_rational::BigRational::from_integer(0.into()));
                rate = Some(acc + m);
            }
            _ => return None,
        }
    }
    rate.filter(|r| !r.is_zero())
}

/// Decompose a STRUCTURALLY DISTRIBUTED expression into UC bases with linear
/// coefficients over the fresh symbols. `None` = a factor falls outside the
/// UC table (polynomial · e^(kx) · sin/cos(bx) with at most one fresh symbol
/// per term).
fn uc_decompose(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
    fresh: &[cas_ast::symbol::SymbolId],
    mult: &num_rational::BigRational,
    out: &mut std::collections::HashMap<UcBase, UcComb>,
) -> bool {
    let mut terms: Vec<(ExprId, bool)> = Vec::new();
    collect_signed_terms(ctx, expr, true, &mut terms);
    'term: for (term, positive) in terms {
        let mut sign = positive;
        let mut core = term;
        while let cas_ast::Expr::Neg(inner) = ctx.get(core) {
            sign = !sign;
            core = *inner;
        }
        let mut factors: Factors = Vec::new();
        collect_product_factors(ctx, core, false, &mut factors);

        let mut coef = mult.clone();
        let mut fresh_idx: Option<usize> = None;
        let mut x_pow = 0usize;
        let mut k = num_rational::BigRational::from_integer(0.into());
        let mut trig = UcTrig::None;
        // Worklist: peeling a Neg can expose a product (`−(x·2)` from the
        // differ) whose sub-factors must be re-decomposed.
        let mut work: Vec<(ExprId, bool)> = factors;
        work.reverse();
        while let Some((f, is_den)) = work.pop() {
            let mut g = f;
            while let cas_ast::Expr::Neg(inner) = ctx.get(g) {
                sign = !sign;
                g = *inner;
            }
            if matches!(ctx.get(g), cas_ast::Expr::Mul(..) | cas_ast::Expr::Div(..)) {
                let mut sub: Factors = Vec::new();
                collect_product_factors(ctx, g, is_den, &mut sub);
                sub.reverse();
                work.extend(sub);
                continue;
            }
            // Rational constant (also den). The pure differ leaves foldable
            // constant litter (`ln(e)` from d/dx aᵘ = aᵘ·ln(a)·u') — fold
            // variable-free factors through the pure constant folder before
            // giving up on them.
            let g_const = cas_math::numeric_eval::as_rational_const(ctx, g).or_else(|| {
                if collect_variables(ctx, g).is_empty() {
                    let folded = cas_math::limits_support::fold_constant_subexprs(ctx, g);
                    cas_math::numeric_eval::as_rational_const(ctx, folded)
                } else {
                    None
                }
            });
            if let Some(r) = g_const {
                if is_den {
                    if r.is_zero() {
                        return false;
                    }
                    coef /= r;
                } else {
                    coef *= r;
                }
                continue;
            }
            match ctx.get(g).clone() {
                cas_ast::Expr::Variable(s) => {
                    let name = ctx.sym_name(s).to_string();
                    if name == var {
                        if is_den {
                            return false;
                        }
                        x_pow += 1;
                    } else if let Some(idx) = fresh.iter().position(|fs| *fs == s) {
                        if is_den || fresh_idx.is_some() {
                            return false;
                        }
                        fresh_idx = Some(idx);
                    } else {
                        return false;
                    }
                }
                cas_ast::Expr::Pow(base, exp) => match ctx.get(base) {
                    cas_ast::Expr::Constant(cas_ast::Constant::E) => {
                        let Some(rate) = linear_rate(ctx, exp, var) else {
                            return false;
                        };
                        k += if is_den { -rate } else { rate };
                    }
                    cas_ast::Expr::Variable(s) if ctx.sym_name(*s) == var => {
                        if is_den {
                            return false;
                        }
                        let Some(n) = cas_math::numeric_eval::as_rational_const(ctx, exp)
                            .filter(|n| n.is_integer() && !n.is_negative())
                            .and_then(|n| usize::try_from(n.to_integer()).ok())
                        else {
                            return false;
                        };
                        x_pow += n;
                    }
                    _ => return false,
                },
                cas_ast::Expr::Function(fn_id, args) if args.len() == 1 => {
                    let name = ctx.sym_name(fn_id).to_string();
                    // Differ litter: d/dx aᵘ emits a `ln(a)` factor unfolded —
                    // for the UC table a = e, so ln(e) ≡ 1 (skip the factor).
                    if name == "ln"
                        && matches!(
                            ctx.get(args[0]),
                            cas_ast::Expr::Constant(cas_ast::Constant::E)
                        )
                    {
                        continue;
                    }
                    if is_den || trig != UcTrig::None {
                        return false;
                    }
                    let Some(rate) = linear_rate(ctx, args[0], var) else {
                        return false;
                    };
                    // Normalize sin(−bx) = −sin(bx), cos(−bx) = cos(bx).
                    let (rate, flip) = if rate.is_negative() {
                        (-rate, true)
                    } else {
                        (rate, false)
                    };
                    match name.as_str() {
                        "sin" => {
                            if flip {
                                sign = !sign;
                            }
                            trig = UcTrig::Sin(rate);
                        }
                        "cos" => trig = UcTrig::Cos(rate),
                        _ => return false,
                    }
                }
                _ => return false,
            }
        }
        if !sign {
            coef = -coef;
        }
        if coef.is_zero() {
            continue 'term;
        }
        let base = UcBase { x_pow, k, trig };
        let entry = out.entry(base).or_insert_with(|| UcComb::zero(fresh.len()));
        match fresh_idx {
            Some(idx) => entry.fresh[idx] += coef,
            None => entry.constant += coef,
        }
    }
    true
}

/// Multiplicity of the REAL rate `k` as a characteristic root of
/// `r² + p·r + q` (0, 1, or 2) — all-rational, no root extraction.
fn real_char_multiplicity(
    p: &num_rational::BigRational,
    q: &num_rational::BigRational,
    k: &num_rational::BigRational,
) -> usize {
    let value = k * k + p * k + q;
    if !value.is_zero() {
        return 0;
    }
    let derivative = num_rational::BigRational::from_integer(2.into()) * k + p;
    if derivative.is_zero() {
        2
    } else {
        1
    }
}

/// True when `k ± i·b` is a characteristic root of `r² + p·r + q`
/// (multiplicity is at most 1 for a real quadratic).
fn complex_char_is_root(
    p: &num_rational::BigRational,
    q: &num_rational::BigRational,
    k: &num_rational::BigRational,
    b: &num_rational::BigRational,
) -> bool {
    let re = k * k - b * b + p * k + q;
    let im = b * (num_rational::BigRational::from_integer(2.into()) * k + p);
    re.is_zero() && im.is_zero()
}

/// Exact Gaussian elimination over BigRational: solve `m · u = d`. `None` =
/// singular/inconsistent (the trial cannot match — decline).
#[allow(clippy::needless_range_loop)] // Gaussian pivoting is clearest with indices
fn solve_rational_system(
    mut m: Vec<Vec<num_rational::BigRational>>,
    mut d: Vec<num_rational::BigRational>,
) -> Option<Vec<num_rational::BigRational>> {
    let rows = m.len();
    let cols = if rows == 0 { 0 } else { m[0].len() };
    let mut pivot_row = 0usize;
    let mut pivot_cols: Vec<usize> = Vec::new();
    for col in 0..cols {
        let Some(sel) = (pivot_row..rows).find(|&r| !m[r][col].is_zero()) else {
            continue;
        };
        m.swap(pivot_row, sel);
        d.swap(pivot_row, sel);
        let inv = m[pivot_row][col].clone();
        for c in col..cols {
            let v = m[pivot_row][c].clone() / inv.clone();
            m[pivot_row][c] = v;
        }
        d[pivot_row] = d[pivot_row].clone() / inv;
        for r in 0..rows {
            if r != pivot_row && !m[r][col].is_zero() {
                let factor = m[r][col].clone();
                for c in col..cols {
                    let v = m[r][c].clone() - factor.clone() * m[pivot_row][c].clone();
                    m[r][c] = v;
                }
                d[r] = d[r].clone() - factor * d[pivot_row].clone();
            }
        }
        pivot_cols.push(col);
        pivot_row += 1;
        if pivot_row == rows {
            break;
        }
    }
    // Consistency: zero rows must have zero rhs.
    for r in pivot_row..rows {
        if !d[r].is_zero() {
            return None;
        }
    }
    // Unique solution required: every column pivoted.
    if pivot_cols.len() != cols {
        return None;
    }
    let mut u = vec![num_rational::BigRational::from_integer(0.into()); cols];
    for (row, &col) in pivot_cols.iter().enumerate() {
        u[col] = d[row].clone();
    }
    Some(u)
}

// ===================== O8: Bernoulli + homogeneous substitution =====================

/// Bernoulli form `y' + p(x)·y = q(x)·yⁿ` with a literal integer n ∉ {0, 1}.
struct BernoulliOde {
    p: ExprId,
    q: ExprId,
    n: num_rational::BigRational,
}

/// Match `a(x)·y' + b(x)·y + c(x)·yⁿ = 0` on the RAW tree (additive terms of
/// `LHS − RHS`): exactly ONE distinct power n ∉ {0,1}, coefficients free of
/// the unknown, no free forcing term. Normalizes to `y' + p·y = q·yⁿ` with
/// `p = b/a`, `q = −c/a`.
fn try_match_bernoulli(
    ctx: &mut Context,
    eq: &Equation,
    diff_sym: cas_ast::symbol::SymbolId,
    func: &str,
    var: &str,
) -> Option<BernoulliOde> {
    let mut terms: Vec<(ExprId, bool)> = Vec::new();
    collect_signed_terms(ctx, eq.lhs, true, &mut terms);
    collect_signed_terms(ctx, eq.rhs, false, &mut terms);

    let mut a_parts: Vec<ExprId> = Vec::new();
    let mut b_parts: Vec<ExprId> = Vec::new();
    let mut c_parts: Vec<ExprId> = Vec::new();
    let mut n_seen: Option<num_rational::BigRational> = None;
    for (term, positive) in terms {
        let mut sign = positive;
        let mut core = term;
        while let cas_ast::Expr::Neg(inner) = ctx.get(core) {
            sign = !sign;
            core = *inner;
        }
        // The moved-over RHS leaves a literal `0` term (`… = 0`): a zero
        // constant is NOT a forcing term — discard it explicitly (recorded
        // repo lesson: structural collectors must drop the parked zero).
        if matches!(cas_math::numeric_eval::as_rational_const(ctx, core), Some(r) if r.is_zero()) {
            continue;
        }
        let mut factors: Factors = Vec::new();
        collect_product_factors(ctx, core, false, &mut factors);

        let mut diff_count = 0usize;
        let mut y_lin = 0usize;
        let mut y_pow: Option<num_rational::BigRational> = None;
        let mut coef_num: Vec<ExprId> = Vec::new();
        let mut coef_den: Vec<ExprId> = Vec::new();
        for (f, is_den) in factors {
            let mut g = f;
            while let cas_ast::Expr::Neg(inner) = ctx.get(g) {
                sign = !sign;
                g = *inner;
            }
            if is_first_order_diff_call(ctx, g, diff_sym, func, var) {
                if is_den {
                    return None;
                }
                diff_count += 1;
                continue;
            }
            if matches!(ctx.get(g), cas_ast::Expr::Variable(s) if ctx.sym_name(*s) == func) {
                if is_den {
                    // y in a denominator is y^(−1): a Bernoulli power.
                    let neg_one = -num_rational::BigRational::from_integer(1.into());
                    if y_pow.is_some() {
                        return None;
                    }
                    y_pow = Some(neg_one);
                    continue;
                }
                y_lin += 1;
                continue;
            }
            if let cas_ast::Expr::Pow(base, exp) = ctx.get(g) {
                let (base, exp) = (*base, *exp);
                if matches!(ctx.get(base), cas_ast::Expr::Variable(s) if ctx.sym_name(*s) == func) {
                    let n = cas_math::numeric_eval::as_rational_const(ctx, exp)
                        .filter(|n| n.is_integer())?;
                    let n = if is_den { -n } else { n };
                    if y_pow.is_some() {
                        return None;
                    }
                    y_pow = Some(n);
                    continue;
                }
            }
            let vars = collect_variables(ctx, g);
            if vars.contains(func) {
                return None;
            }
            if is_den {
                coef_den.push(g)
            } else {
                coef_num.push(g)
            }
        }
        if diff_count + y_lin + usize::from(y_pow.is_some()) > 1 {
            return None;
        }
        let build_product = |ctx: &mut Context, parts: &[ExprId]| -> Option<ExprId> {
            let mut it = parts.iter();
            let first = *it.next()?;
            Some(it.fold(first, |acc, &p| ctx.add(cas_ast::Expr::Mul(acc, p))))
        };
        let num = build_product(ctx, &coef_num).unwrap_or_else(|| ctx.num(1));
        let mut coef = match build_product(ctx, &coef_den) {
            Some(d) => ctx.add(cas_ast::Expr::Div(num, d)),
            None => num,
        };
        if !sign {
            coef = ctx.add(cas_ast::Expr::Neg(coef));
        }
        if diff_count == 1 {
            a_parts.push(coef);
        } else if y_lin == 1 {
            b_parts.push(coef);
        } else if let Some(n) = y_pow {
            let one = num_rational::BigRational::from_integer(1.into());
            if n.is_zero() || n == one {
                return None;
            }
            match &n_seen {
                None => n_seen = Some(n),
                Some(seen) if *seen == n => {}
                Some(_) => return None, // two distinct powers: not Bernoulli
            }
            c_parts.push(coef);
        } else {
            // Free forcing term: not the pure Bernoulli shape.
            return None;
        }
    }
    let n = n_seen?;
    if a_parts.is_empty() || c_parts.is_empty() {
        return None;
    }
    let sum = |ctx: &mut Context, parts: &[ExprId]| -> Option<ExprId> {
        let mut it = parts.iter();
        let first = *it.next()?;
        Some(it.fold(first, |acc, &p| ctx.add(cas_ast::Expr::Add(acc, p))))
    };
    let a = sum(ctx, &a_parts).expect("nonempty");
    let b = sum(ctx, &b_parts).unwrap_or_else(|| ctx.num(0));
    let c = sum(ctx, &c_parts).expect("nonempty");
    let p = if is_literal_one(ctx, a) {
        b
    } else {
        ctx.add(cas_ast::Expr::Div(b, a))
    };
    let neg_c = ctx.add(cas_ast::Expr::Neg(c));
    let q = if is_literal_one(ctx, a) {
        neg_c
    } else {
        ctx.add(cas_ast::Expr::Div(neg_c, a))
    };
    Some(BernoulliOde { p, q, n })
}

/// Homogeneous-in-degree form `y' = F(y/x)`: substitute `y → v·x` in the
/// isolated RHS and check the fold is x-free. Returns `(v_var, F(v))`.
fn try_match_homogeneous_rhs(
    engine: &mut Engine,
    verify_options: &crate::options::EvalOptions,
    rhs: ExprId,
    func: &str,
    var: &str,
) -> Option<(ExprId, ExprId)> {
    let ctx = &mut engine.simplifier.context;
    let input_vars = collect_variables(ctx, rhs);
    let v_name = if input_vars.contains("v") { "vh" } else { "v" };
    let v_var = ctx.var(v_name);
    let x_var = ctx.var(var);
    let y_var = ctx.var(func);
    let vx = ctx.add(cas_ast::Expr::Mul(v_var, x_var));
    let substituted = substitute_power_aware(ctx, rhs, y_var, vx, SubstituteOptions::exact());
    let folded = engine.fold_free_subtree(verify_options, substituted);
    let ctx = &engine.simplifier.context;
    let vars = collect_variables(ctx, folded);
    if vars.contains(var) {
        return None;
    }
    Some((v_var, folded))
}

/// Narrated solve steps for the Bernoulli method (D13).
#[allow(clippy::too_many_arguments)]
fn build_bernoulli_steps(
    ctx: &mut Context,
    ode_eq: &Equation,
    p: ExprId,
    q: ExprId,
    n: &num_rational::BigRational,
    one_minus_n: &num_rational::BigRational,
    y_var: ExprId,
    v_general: ExprId,
    final_form: ExprId,
) -> Vec<crate::api::SolveStep> {
    let mut steps: Vec<crate::api::SolveStep> = Vec::new();
    steps.push(crate::api::SolveStep {
        description: format!(
            "Identificar forma de Bernoulli: y' + p·y = q·y^n con p = {}, q = {}, n = {}",
            render_expr(ctx, p),
            render_expr(ctx, q),
            n
        ),
        equation_after: ode_eq.clone(),
        importance: ImportanceLevel::High,
        substeps: vec![],
    });
    let v_name = ctx.var("v");
    let exp_num = ctx.add(cas_ast::Expr::Number(one_minus_n.clone()));
    let y_pow = ctx.add(cas_ast::Expr::Pow(y_var, exp_num));
    steps.push(crate::api::SolveStep {
        description: format!(
            "Sustituir v = y^(1−n) = y^({one_minus_n}): la EDO se vuelve lineal v' + (1−n)·p·v = (1−n)·q"
        ),
        equation_after: Equation {
            lhs: v_name,
            rhs: y_pow,
            op: RelOp::Eq,
        },
        importance: ImportanceLevel::High,
        substeps: vec![],
    });
    steps.push(crate::api::SolveStep {
        description: "Resolver la lineal en v por factor integrante".to_string(),
        equation_after: Equation {
            lhs: v_name,
            rhs: v_general,
            op: RelOp::Eq,
        },
        importance: ImportanceLevel::High,
        substeps: vec![],
    });
    steps.push(crate::api::SolveStep {
        description: "Deshacer la sustitución y verificar por sustitución en la EDO".to_string(),
        equation_after: Equation {
            lhs: y_var,
            rhs: final_form,
            op: RelOp::Eq,
        },
        importance: ImportanceLevel::Medium,
        substeps: vec![],
    });
    steps
}

/// Narrated solve steps for the homogeneous-substitution method (D13).
#[allow(clippy::too_many_arguments)]
fn build_homogeneous_steps(
    ctx: &mut Context,
    ode_eq: &Equation,
    f_v: ExprId,
    v_var: ExprId,
    g_int: ExprId,
    ln_side: ExprId,
    y_var: ExprId,
    explicit: &[ExprId],
    implicit_phi: Option<ExprId>,
    c_var: ExprId,
) -> Vec<crate::api::SolveStep> {
    let mut steps: Vec<crate::api::SolveStep> = Vec::new();
    steps.push(crate::api::SolveStep {
        description: format!(
            "Identificar EDO homogénea: y' = F(y/x) con F(v) = {}",
            render_expr(ctx, f_v)
        ),
        equation_after: ode_eq.clone(),
        importance: ImportanceLevel::High,
        substeps: vec![],
    });
    steps.push(crate::api::SolveStep {
        description: "Sustituir v = y/x: la EDO se vuelve separable dv/(F(v)−v) = dx/x".to_string(),
        equation_after: ode_eq.clone(),
        importance: ImportanceLevel::High,
        substeps: vec![],
    });
    let rhs_c = ctx.add(cas_ast::Expr::Add(ln_side, c_var));
    steps.push(crate::api::SolveStep {
        description: "Integrar ambos lados de la reducción separable".to_string(),
        equation_after: Equation {
            lhs: g_int,
            rhs: rhs_c,
            op: RelOp::Eq,
        },
        importance: ImportanceLevel::High,
        substeps: vec![],
    });
    let _ = v_var;
    if let Some(phi) = implicit_phi {
        steps.push(crate::api::SolveStep {
            description: "Combinar en una solución implícita φ(x,y) = C".to_string(),
            equation_after: Equation {
                lhs: phi,
                rhs: c_var,
                op: RelOp::Eq,
            },
            importance: ImportanceLevel::High,
            substeps: vec![],
        });
    } else if let Some(first) = explicit.first() {
        steps.push(crate::api::SolveStep {
            description: "Deshacer la sustitución: y = x·v(x)".to_string(),
            equation_after: Equation {
                lhs: y_var,
                rhs: *first,
                op: RelOp::Eq,
            },
            importance: ImportanceLevel::High,
            substeps: vec![],
        });
    }
    steps.push(crate::api::SolveStep {
        description: "Verificar por sustitución: el residuo de la EDO se reduce a 0".to_string(),
        equation_after: ode_eq.clone(),
        importance: ImportanceLevel::Medium,
        substeps: vec![],
    });
    steps
}

// ===================== O6: 2×2 linear systems X' = A·X =====================

/// exp(r·t) with a rational rate (free helper shared by the system path).
fn build_exp_rate(ctx: &mut Context, rate: &num_rational::BigRational, t_var: ExprId) -> ExprId {
    let e_const = ctx.add(cas_ast::Expr::Constant(cas_ast::Constant::E));
    let r_num = ctx.add(cas_ast::Expr::Number(rate.clone()));
    let arg = ctx.add(cas_ast::Expr::Mul(r_num, t_var));
    ctx.add(cas_ast::Expr::Pow(e_const, arg))
}

/// Extract one row of `X' = A·X` from `a·diff(f_i, t) + Σ b_j·f_j = 0`:
/// which unknown is differentiated, and the RATIONAL row `A[i][·] = −b/a`.
/// `None` on variable coefficients, nonlinear unknowns, nested diffs, or a
/// nonzero forcing term (non-homogeneous systems are future work).
fn try_extract_system_row(
    ctx: &mut Context,
    eq: &Equation,
    diff_sym: cas_ast::symbol::SymbolId,
    funcs: &[String],
    var: &str,
) -> Option<(usize, [num_rational::BigRational; 2])> {
    let mut terms: Vec<(ExprId, bool)> = Vec::new();
    collect_signed_terms(ctx, eq.lhs, true, &mut terms);
    collect_signed_terms(ctx, eq.rhs, false, &mut terms);

    let zero = num_rational::BigRational::from_integer(0.into());
    let mut a_coef = zero.clone();
    let mut which_diff: Option<usize> = None;
    let mut b = [zero.clone(), zero.clone()];
    for (term, positive) in terms {
        let mut sign = positive;
        let mut core = term;
        while let cas_ast::Expr::Neg(inner) = ctx.get(core) {
            sign = !sign;
            core = *inner;
        }
        if matches!(cas_math::numeric_eval::as_rational_const(ctx, core), Some(r) if r.is_zero()) {
            continue; // the parked zero from the moved RHS
        }
        let mut factors: Factors = Vec::new();
        collect_product_factors(ctx, core, false, &mut factors);

        let mut diff_of: Option<usize> = None;
        let mut func_of: Option<usize> = None;
        let mut coef = num_rational::BigRational::from_integer(1.into());
        for (f, is_den) in factors {
            let mut g = f;
            while let cas_ast::Expr::Neg(inner) = ctx.get(g) {
                sign = !sign;
                g = *inner;
            }
            let mut classified = false;
            for (idx, fname) in funcs.iter().enumerate() {
                if is_first_order_diff_call(ctx, g, diff_sym, fname, var) {
                    if is_den || diff_of.is_some() || func_of.is_some() {
                        return None;
                    }
                    diff_of = Some(idx);
                    classified = true;
                    break;
                }
                if matches!(ctx.get(g), cas_ast::Expr::Variable(s) if ctx.sym_name(*s) == fname) {
                    if is_den || diff_of.is_some() || func_of.is_some() {
                        return None;
                    }
                    func_of = Some(idx);
                    classified = true;
                    break;
                }
            }
            if classified {
                continue;
            }
            // Any other appearance of an unknown (nested, higher-order) declines.
            let vars = collect_variables(ctx, g);
            if funcs.iter().any(|f| vars.contains(f)) {
                return None;
            }
            match cas_math::numeric_eval::as_rational_const(ctx, g) {
                Some(r) if is_den && !r.is_zero() => coef /= r,
                Some(r) if !is_den => coef *= r,
                _ => return None, // variable coefficient
            }
        }
        let signed = if sign { coef } else { -coef };
        if let Some(idx) = diff_of {
            match which_diff {
                None => which_diff = Some(idx),
                Some(seen) if seen == idx => {}
                Some(_) => return None,
            }
            a_coef += signed;
        } else if let Some(idx) = func_of {
            b[idx] += signed;
        } else {
            return None; // nonzero free term: non-homogeneous system
        }
    }
    let which = which_diff?;
    if a_coef.is_zero() {
        return None;
    }
    Some((which, [-&b[0] / &a_coef, -&b[1] / &a_coef]))
}

/// One basis solution of the 2×2 system: component expressions (x(t), y(t)).
struct SystemBasis {
    comp: [ExprId; 2],
}

// ===================== O9: Cauchy-Euler x²y'' + a·x·y' + b·y = 0 =====================

/// Cauchy-Euler match `c2·x²·y'' + c1·x·y' + c0·y = 0` (exact rational
/// c-coefficients; the x-power must match the derivative order EXACTLY —
/// Bessel's `(x²−1)·y` term declines because y carries x_pow 2).
struct CauchyEulerOde {
    c2: num_rational::BigRational,
    c1: num_rational::BigRational,
    c0: num_rational::BigRational,
}

fn try_match_cauchy_euler(
    ctx: &mut Context,
    eq: &Equation,
    diff_sym: cas_ast::symbol::SymbolId,
    func: &str,
    var: &str,
) -> Option<CauchyEulerOde> {
    let mut terms: Vec<(ExprId, bool)> = Vec::new();
    collect_signed_terms(ctx, eq.lhs, true, &mut terms);
    collect_signed_terms(ctx, eq.rhs, false, &mut terms);

    let zero = num_rational::BigRational::from_integer(0.into());
    let mut c2 = zero.clone();
    let mut c1 = zero.clone();
    let mut c0 = zero.clone();
    for (term, positive) in terms {
        let mut sign = positive;
        let mut core = term;
        while let cas_ast::Expr::Neg(inner) = ctx.get(core) {
            sign = !sign;
            core = *inner;
        }
        if matches!(cas_math::numeric_eval::as_rational_const(ctx, core), Some(r) if r.is_zero()) {
            continue; // the parked zero
        }
        let mut factors: Factors = Vec::new();
        collect_product_factors(ctx, core, false, &mut factors);

        let mut order: Option<usize> = None; // 0 = bare y, 1 = y', 2 = y''
        let mut x_pow = 0usize;
        let mut coef = num_rational::BigRational::from_integer(1.into());
        for (f, is_den) in factors {
            let mut g = f;
            while let cas_ast::Expr::Neg(inner) = ctx.get(g) {
                sign = !sign;
                g = *inner;
            }
            match diff_call_order(ctx, g, diff_sym, func, var) {
                Some(n @ (1 | 2)) if !is_den => {
                    if order.is_some() {
                        return None;
                    }
                    order = Some(n);
                    continue;
                }
                Some(_) => return None,
                None => {}
            }
            if matches!(ctx.get(g), cas_ast::Expr::Variable(s) if ctx.sym_name(*s) == func) {
                if is_den || order.is_some() {
                    return None;
                }
                order = Some(0);
                continue;
            }
            if matches!(ctx.get(g), cas_ast::Expr::Variable(s) if ctx.sym_name(*s) == var) {
                if is_den {
                    return None;
                }
                x_pow += 1;
                continue;
            }
            if let cas_ast::Expr::Pow(base, exp) = ctx.get(g) {
                let (base, exp) = (*base, *exp);
                if matches!(ctx.get(base), cas_ast::Expr::Variable(s) if ctx.sym_name(*s) == var)
                    && !is_den
                {
                    let n: Option<usize> = cas_math::numeric_eval::as_rational_const(ctx, exp)
                        .filter(|n| n.is_integer() && !n.is_negative())
                        .and_then(|n| usize::try_from(n.to_integer()).ok());
                    if let Some(n) = n {
                        x_pow += n;
                        continue;
                    }
                    return None;
                }
            }
            let vars = collect_variables(ctx, g);
            if vars.contains(func) || vars.contains(var) {
                return None;
            }
            match cas_math::numeric_eval::as_rational_const(ctx, g) {
                Some(r) if is_den && !r.is_zero() => coef /= r,
                Some(r) if !is_den => coef *= r,
                _ => return None,
            }
        }
        let Some(ord) = order else {
            return None; // a free forcing term: not the homogeneous CE shape
        };
        // The Euler structure: x-power EQUALS the derivative order.
        if x_pow != ord {
            return None;
        }
        let signed = if sign { coef } else { -coef };
        match ord {
            2 => c2 += signed,
            1 => c1 += signed,
            _ => c0 += signed,
        }
    }
    if c2.is_zero() {
        return None;
    }
    Some(CauchyEulerOde { c2, c1, c0 })
}
