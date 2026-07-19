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
const VERIFY_TIME_BUDGET_MS: u64 = 3_000;

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
    let is_diff_call = |id: ExprId| -> bool {
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
    };
    if is_diff_call(eq.lhs) {
        return Some(eq.rhs);
    }
    if is_diff_call(eq.rhs) {
        return Some(eq.lhs);
    }
    None
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
        conditions: &[String],
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
        };

        // Initial conditions parse but resolve in a future cycle (O3).
        if !conditions.is_empty() {
            let r = mk_residual(
                ctx,
                "Condiciones iniciales aún no soportadas (ciclo O3): se declina en lugar de emitir la solución general sin aplicarlas",
            );
            return Ok(r);
        }
        // Higher-order ODEs are future cycles (O4+).
        if max_diff_arity > 2 {
            let r = mk_residual(
                ctx,
                "EDO de orden superior: la característica de 2º orden llega en el ciclo O4; se declina honesto",
            );
            return Ok(r);
        }

        // O0 method: separable with the derivative isolated on one side.
        let Some(rhs) = match_isolated_first_order(ctx, &ode_eq, diff_sym, func, var) else {
            let r = mk_residual(
                ctx,
                "Forma no soportada todavía: la EDO no tiene diff aislado en un lado (lineal/exactas llegan en O1/O2); se declina honesto",
            );
            return Ok(r);
        };

        let Some(split) = split_separable(ctx, rhs, func, var) else {
            let r = mk_residual(
                ctx,
                "La EDO no es separable (f(x)·g(y)); los métodos lineal/exactas/Bernoulli llegan en ciclos futuros",
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

    /// True when the FULL evaluator reduces `e` to exactly `Number(0)`, with
    /// numeric verification disabled (the D5 ritual: a probe never confirms).
    fn reduces_to_zero_exact(&mut self, options: &crate::options::EvalOptions, e: ExprId) -> bool {
        let saved_numeric = self.simplifier.allow_numerical_verification;
        self.simplifier.allow_numerical_verification = false;
        let reduced = matches!(
            self.eval_simplify(options, e),
            Ok((EvalResult::Expr(result), ..))
                if matches!(self.simplifier.context.get(result), cas_ast::Expr::Number(n) if n.is_zero())
        );
        self.simplifier.allow_numerical_verification = saved_numeric;
        reduced
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
