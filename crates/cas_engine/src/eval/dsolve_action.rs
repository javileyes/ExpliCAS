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
use num_traits::{One, Zero};

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
        // tree so envelope/JSON callers get the same honest contract.
        let Some(max_diff_arity) = scan_diff_calls_of(ctx, resolved, diff_sym, func) else {
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

        // Initial conditions (O3): already parsed by the solver layer.
        let parsed_conditions: Vec<InitialCondition> = conditions.to_vec();
        if parsed_conditions.iter().any(|c| c.order > 0) {
            let r = mk_residual(
                ctx,
                "Condiciones sobre la derivada (y'(x0)=v0) llegan con el 2º orden (ciclo O4); se declina honesto",
            );
            return Ok(r);
        }
        if parsed_conditions.len() > 1 {
            let r = mk_residual(
                ctx,
                "Una EDO de primer orden admite UNA condición inicial; el sistema de constantes de 2º orden llega en O4",
            );
            return Ok(r);
        }
        let initial_condition = parsed_conditions.into_iter().next();

        // Higher-order ODEs are future cycles (O4+).
        if max_diff_arity > 2 {
            let r = mk_residual(
                ctx,
                "EDO de orden superior: la característica de 2º orden llega en el ciclo O4; se declina honesto",
            );
            return Ok(r);
        }

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
            let r = mk_residual(
                &mut self.simplifier.context,
                "La EDO no es separable, lineal ni exacta; Bernoulli/homogéneas llegan en el ciclo O8",
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
    /// Runs up to TWO passes: the pipeline can stop one step short of the
    /// fixpoint (`(1/e^0 + 0 - 1) - 0` folds to `1/e^0 - 1` in one pass and
    /// to `0` only on the next — a named engine-coverage candidate); iterating
    /// the exact evaluator never changes semantics, only reaches the fixpoint.
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

        // μ = e^(∫p dx). A p whose antiderivative does not close declines.
        let p_int = if matches!(ctx.get(p), cas_ast::Expr::Number(n) if n.is_zero()) {
            ctx.num(0)
        } else {
            match crate::rules::calculus::integrate_with_trace(ctx, p, var) {
                Some(outcome) if outcome.required_conditions.is_empty() => outcome.result,
                _ => {
                    let r = residual_action_result(
                        ctx,
                        resolved,
                        y_var,
                        x_var,
                        "La integral ∫ p(x) dx del factor integrante no cierra en forma elemental; se declina honesto",
                    );
                    return Ok(r);
                }
            }
        };
        let e_const = ctx.add(cas_ast::Expr::Constant(cas_ast::Constant::E));
        let mu_raw = ctx.add(cas_ast::Expr::Pow(e_const, p_int));
        let mu_folded = self.fold_free_subtree(&verify_options, mu_raw);
        // D12 (pin L9): μ = e^(ln|x|) folds to |x|; any functional μ works, so
        // presentation takes the textbook μ = x (strip unknown-free abs).
        let ctx = &mut self.simplifier.context;
        let mu = strip_free_abs(ctx, mu_folded, func);

        // ∫ μ·q dx.
        let mu_q_raw = ctx.add(cas_ast::Expr::Mul(mu, q));
        let mu_q = self.fold_free_subtree(&verify_options, mu_q_raw);
        let ctx = &mut self.simplifier.context;
        let g_int = if matches!(ctx.get(mu_q), cas_ast::Expr::Number(n) if n.is_zero()) {
            ctx.num(0)
        } else {
            match crate::rules::calculus::integrate_with_trace(ctx, mu_q, var) {
                Some(outcome) if outcome.required_conditions.is_empty() => outcome.result,
                _ => {
                    let r = residual_action_result(
                        ctx,
                        resolved,
                        y_var,
                        x_var,
                        "La integral ∫ μ·q dx no cierra en forma elemental; se declina honesto",
                    );
                    return Ok(r);
                }
            }
        };

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

        // y = ∫μq/μ + C/μ — the SPLIT form: each term folds canonically
        // (`C/e^(2x)`, `x³/4`) and the verifier reduces the residue by
        // linearity; the joined `(G+C)/μ` quotient leaves an uncancelled
        // double-denominator combination the evaluator cannot always close.
        // The exponential cancellation `e^x·(…)/e^x` is peeled STRUCTURALLY
        // (μ as a literal product factor of ∫μq) so the simplifier never sees
        // the depth-hostile quotient.
        let part_g = match cancel_literal_factor(ctx, g_int, mu) {
            Some(cancelled) => cancelled,
            None => ctx.add(cas_ast::Expr::Div(g_int, mu)),
        };
        let part_c = ctx.add(cas_ast::Expr::Div(c_var, mu));
        // Fold each part SEPARATELY and sum without a final fold: the full
        // pipeline would re-join the fractions over μ (AddFractions), which
        // both un-does the textbook split form and rebuilds the quotient the
        // verifier cannot reduce.
        let part_g = self.fold_free_subtree(&verify_options, part_g);
        let part_c = self.fold_free_subtree(&verify_options, part_c);
        let ctx = &mut self.simplifier.context;
        let candidate = if matches!(ctx.get(part_g), cas_ast::Expr::Number(n) if n.is_zero()) {
            part_c
        } else {
            ctx.add(cas_ast::Expr::Add(part_g, part_c))
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

    /// D11 level 2: reconstruct φ = ∫M dx + ∫(N − ∂y∫M) dy with the FULL
    /// evaluator folding the intermediate pieces (the internal F6 path only
    /// canonicalizes polynomial rests). All pieces are diff-free ordinary
    /// expressions in (x, y), so the D4 invariant allows the pipeline.
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
