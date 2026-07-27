//! The instance↔template MATCHER (C1.8, the half with teeth).
//!
//! The census settled what each template ASSERTS; this module settles whether
//! the pair in front of the reader is an INSTANCE of it. The audit's named
//! lies are exactly this failure: «Usar tan(u)·cot(u) = 1» emitted over a pair
//! that is not `tan(σu)·cot(σu) ⇒ 1` for any σ, and «Usar tan(u) = …» emitted
//! by the branch that recognized no variant at all. The census cannot see
//! those — the template is TRUE; what lies is its application.
//!
//! ## Semantics
//!
//! A template parses in its own scratch [`Context`], where EVERY variable is a
//! metavariable by construction — templates are source constants and contain
//! no user symbols. Matching walks template and instance in their two contexts
//! simultaneously:
//!
//! - a metavariable binds any instance subtree, CONSISTENTLY (`sin(u)·cos(u)`
//!   never matches `sin(x)·cos(y)`);
//! - additive and multiplicative chains match as MULTISETS through the n-ary
//!   views, with bounded backtracking — the canonical order of `tan(x)·cot(x)`
//!   need not be the canonical order of `tan(u)·cot(u)` under σ;
//! - everything else is rigid: numbers by value, constants by variant,
//!   functions by builtin identity (falling back to name), structure by shape.
//!
//! Two modes:
//!
//! - [`match_instance`]: the WHOLE pair `(before, after)` is one instance of
//!   `(lhs, rhs)` under one shared σ. This is what a «Usar L = R» sub-step
//!   over its own two sides asserts.
//! - [`match_rewrite`]: `after` is `before` with σ(lhs) rewritten to σ(rhs)
//!   at some subterm — the relation the CHAIN emitters state when a template
//!   fires inside a larger expression.
//!
//! ## What a failed match means
//!
//! Nothing, on its own — this matcher is INCOMPLETE by design (bounded
//! backtracking, no sign absorption, no modulo-simplification matching), so a
//! miss is only evidence when the shapes involved are ones the unit tests pin
//! as covered. That is why enforcement lands emitter by emitter, each with its
//! own decline test, instead of as a blanket sweep — the lesson of the
//! assume-equality prototype that deleted 51 legitimate sub-steps.

use cas_ast::ordering::compare_expr;
use cas_ast::{Context, Expr, ExprId};
use cas_math::expr_nary::{AddView, MulView, Sign};
use std::cmp::Ordering;
use std::collections::BTreeMap;

/// A template pair parsed into its own scratch context.
pub struct ParsedTemplate {
    pub ctx: Context,
    pub lhs: ExprId,
    pub rhs: ExprId,
}

/// Parse the two display-form sides of a template (`sin(2u)^2 / 4`, `·`, `²`
/// and friends included) into a scratch context. `None` means the template is
/// display notation the parser does not accept — the `DisplayNotation` class,
/// which no matcher can serve.
pub fn parse_template(lhs: &str, rhs: &str) -> Option<ParsedTemplate> {
    let mut ctx = Context::new();
    let lhs_id = cas_parser::parse(&super::schema::to_parser_input(lhs), &mut ctx).ok()?;
    let rhs_id = cas_parser::parse(&super::schema::to_parser_input(rhs), &mut ctx).ok()?;
    Some(ParsedTemplate {
        ctx,
        lhs: lhs_id,
        rhs: rhs_id,
    })
}

/// Metavariable name → the instance subtree it binds.
pub type Bindings = BTreeMap<String, ExprId>;

/// Additive/multiplicative chains longer than this decline instead of
/// backtracking: the guard that keeps the multiset search bounded.
const MAX_CHAIN_TERMS: usize = 8;

/// The STRUCTURAL-ONLY mode: both sides must bind under one σ, no directed
/// fallback. Emitters with SEVERAL equivalent templates need it as a first
/// pass — the directed mode decides by the ENGINE's equality, so it happily
/// verifies `2·cos(u)²−1` against the `1 − 2·sin(u)²` template (they are equal
/// by Pythagoras), and the published title then names a form the reader is
/// not looking at. Structural-first keeps the cited formula the one on screen;
/// the directed pass stays available for instances whose shape folded away.
pub fn match_instance_structural(
    template: &ParsedTemplate,
    ictx: &Context,
    before: ExprId,
    after: ExprId,
) -> Option<Bindings> {
    let mut bindings = Bindings::new();
    if match_expr(&template.ctx, template.lhs, ictx, before, &mut bindings).is_some()
        && match_expr(&template.ctx, template.rhs, ictx, after, &mut bindings).is_some()
    {
        return Some(bindings);
    }
    None
}

/// Decide whether `(before, after)` is ONE instance of the template pair under
/// a single shared σ. Returns the bindings as the positive witness.
pub fn match_instance(
    template: &ParsedTemplate,
    ictx: &Context,
    before: ExprId,
    after: ExprId,
) -> Option<Bindings> {
    // Fast path: both sides match structurally under one σ.
    if let Some(bindings) = match_instance_structural(template, ictx, before, after) {
        return Some(bindings);
    }
    // DIRECTED fallback, and it is load-bearing, not a convenience: with
    // `u = x/2` the genuine instance of `(1 - cos(2u))/sin(2u)` reads
    // `(1 - cos(x))/sin(x)` — the doubling CANONICALIZED away when the
    // instance was built, so no structural walk can find it. Instead, take σ
    // from whichever side binds structurally, INSTANTIATE the other side in
    // the instance context — where `Context::add` folds `2·(x/2)` exactly the
    // way the original fold happened — and compare canonically.
    match_directed(template, template.lhs, template.rhs, ictx, before, after)
        .or_else(|| match_directed(template, template.rhs, template.lhs, ictx, after, before))
}

/// σ from `anchor_template → anchor_instance` (structural), then σ(other) is
/// BUILT in the instance context and compared canonically with the other
/// instance side.
fn match_directed(
    template: &ParsedTemplate,
    anchor_template: ExprId,
    other_template: ExprId,
    ictx: &Context,
    anchor_instance: ExprId,
    other_instance: ExprId,
) -> Option<Bindings> {
    let mut bindings = Bindings::new();
    match_expr(
        &template.ctx,
        anchor_template,
        ictx,
        anchor_instance,
        &mut bindings,
    )?;
    let mut scratch = ictx.clone();
    let built = instantiate(&template.ctx, other_template, &mut scratch, &bindings)?;
    if compare_expr(&scratch, built, other_instance) == Ordering::Equal {
        return Some(bindings);
    }
    // `Context::add` builds raw nodes — it does not fold `2·(x/2)` to `x` the
    // way the original instance was folded on its way through the engine. The
    // comparison that owns that judgement is the engine's own: fold the
    // difference, and only a PROVEN zero is a match. `Undecided` declines —
    // for these anchored templates the genuine instances fold trivially, and
    // an abstention publishing nothing is the conservative side.
    (super::claim::decide_equality(&mut scratch, built, other_instance)
        == super::claim::ClaimVerdict::Verified)
        .then_some(bindings)
}

/// Decide whether `after` equals `before` with σ(lhs) → σ(rhs) rewritten at
/// some subterm. Hash-consing note: replacing a subterm replaces EVERY
/// occurrence of that subtree, which is also how the engine's own rewrites
/// land — a narration that rewrote only one of two identical occurrences
/// would not match, and should not.
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "rewrite mode: exercised by its tests today, consumed by the \
                               shadow-measurement pass over the ~30 remaining «Usar L = R» \
                               emitters before any of them migrates"
    )
)]
pub fn match_rewrite(
    template: &ParsedTemplate,
    ictx: &Context,
    before: ExprId,
    after: ExprId,
) -> Option<Bindings> {
    // The whole pair is the degenerate rewrite at the root.
    if let Some(bindings) = match_instance(template, ictx, before, after) {
        return Some(bindings);
    }
    let mut sites = Vec::new();
    collect_subterms(ictx, before, &mut sites);
    for site in sites {
        let mut bindings = Bindings::new();
        if match_expr(&template.ctx, template.lhs, ictx, site, &mut bindings).is_none() {
            continue;
        }
        let mut scratch = ictx.clone();
        let Some(replacement) = instantiate(&template.ctx, template.rhs, &mut scratch, &bindings)
        else {
            continue;
        };
        let rewritten = cas_ast::substitute_expr_by_id(&mut scratch, before, site, replacement);
        if compare_expr(&scratch, rewritten, after) == Ordering::Equal {
            return Some(bindings);
        }
    }
    None
}

fn collect_subterms(ctx: &Context, expr: ExprId, out: &mut Vec<ExprId>) {
    if out.contains(&expr) {
        return;
    }
    out.push(expr);
    match ctx.get(expr) {
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) | Expr::Pow(l, r) => {
            let (l, r) = (*l, *r);
            collect_subterms(ctx, l, out);
            collect_subterms(ctx, r, out);
        }
        Expr::Neg(inner) | Expr::Hold(inner) => {
            let inner = *inner;
            collect_subterms(ctx, inner, out);
        }
        Expr::Function(_, args) => {
            for arg in args.clone() {
                collect_subterms(ctx, arg, out);
            }
        }
        _ => {}
    }
}

/// Build σ(template) inside the instance context. `None` when the template
/// contains a metavariable σ never bound (an rhs-only metavariable cannot be
/// invented) or a node kind this walker does not build.
fn instantiate(
    tctx: &Context,
    template: ExprId,
    ictx: &mut Context,
    bindings: &Bindings,
) -> Option<ExprId> {
    match tctx.get(template).clone() {
        Expr::Variable(sym) => bindings.get(tctx.sym_name(sym)).copied(),
        Expr::Number(n) => Some(ictx.add(Expr::Number(n))),
        Expr::Constant(c) => Some(ictx.add(Expr::Constant(c))),
        Expr::Add(l, r) => {
            let (l, r) = (
                instantiate(tctx, l, ictx, bindings)?,
                instantiate(tctx, r, ictx, bindings)?,
            );
            Some(ictx.add(Expr::Add(l, r)))
        }
        Expr::Sub(l, r) => {
            let (l, r) = (
                instantiate(tctx, l, ictx, bindings)?,
                instantiate(tctx, r, ictx, bindings)?,
            );
            Some(ictx.add(Expr::Sub(l, r)))
        }
        Expr::Mul(l, r) => {
            let (l, r) = (
                instantiate(tctx, l, ictx, bindings)?,
                instantiate(tctx, r, ictx, bindings)?,
            );
            Some(ictx.add(Expr::Mul(l, r)))
        }
        Expr::Div(l, r) => {
            let (l, r) = (
                instantiate(tctx, l, ictx, bindings)?,
                instantiate(tctx, r, ictx, bindings)?,
            );
            Some(ictx.add(Expr::Div(l, r)))
        }
        Expr::Pow(l, r) => {
            let (l, r) = (
                instantiate(tctx, l, ictx, bindings)?,
                instantiate(tctx, r, ictx, bindings)?,
            );
            Some(ictx.add(Expr::Pow(l, r)))
        }
        Expr::Neg(inner) => {
            let inner = instantiate(tctx, inner, ictx, bindings)?;
            Some(ictx.add(Expr::Neg(inner)))
        }
        Expr::Function(fn_id, args) => {
            let name = tctx.sym_name(fn_id).to_string();
            let mut built = Vec::with_capacity(args.len());
            for arg in args {
                built.push(instantiate(tctx, arg, ictx, bindings)?);
            }
            let sym = ictx.intern_symbol(&name);
            Some(ictx.add(Expr::Function(sym, built)))
        }
        _ => None,
    }
}

fn unwrap_hold(ctx: &Context, mut expr: ExprId) -> ExprId {
    loop {
        let unwrapped = cas_ast::hold::unwrap_internal_hold(ctx, expr);
        if unwrapped == expr {
            return expr;
        }
        expr = unwrapped;
    }
}

/// The core walk. `None` = no match; `Some(())` = matched, with `bindings`
/// extended. On failure the caller must discard the bindings it passed in
/// (partial extensions are not rolled back — every caller clones or retries
/// with a fresh map).
fn match_expr(
    tctx: &Context,
    template: ExprId,
    ictx: &Context,
    instance: ExprId,
    bindings: &mut Bindings,
) -> Option<()> {
    let template = unwrap_hold(tctx, template);
    let instance = unwrap_hold(ictx, instance);

    // Metavariable: bind consistently.
    if let Expr::Variable(sym) = tctx.get(template) {
        let name = tctx.sym_name(*sym).to_string();
        return match bindings.get(&name) {
            Some(&bound) => (compare_expr(ictx, bound, instance) == Ordering::Equal).then_some(()),
            None => {
                bindings.insert(name, instance);
                Some(())
            }
        };
    }

    let is_additive = |ctx: &Context, e: ExprId| {
        matches!(ctx.get(e), Expr::Add(_, _) | Expr::Sub(_, _) | Expr::Neg(_))
    };
    // Additive chains: multiset match through AddView, so canonical-order
    // differences between the two contexts cannot produce a false miss.
    if is_additive(tctx, template) || is_additive(ictx, instance) {
        let t_view = AddView::from_expr(tctx, template);
        let i_view = AddView::from_expr(ictx, instance);
        return match_multiset_signed(tctx, &t_view.terms, ictx, &i_view.terms, bindings);
    }

    let is_mul = |ctx: &Context, e: ExprId| matches!(ctx.get(e), Expr::Mul(_, _));
    if is_mul(tctx, template) || is_mul(ictx, instance) {
        let t_view = MulView::from_expr(tctx, template);
        let i_view = MulView::from_expr(ictx, instance);
        let t_terms: Vec<(ExprId, Sign)> = t_view.factors.iter().map(|&f| (f, Sign::Pos)).collect();
        let i_terms: Vec<(ExprId, Sign)> = i_view.factors.iter().map(|&f| (f, Sign::Pos)).collect();
        return match_multiset_signed(tctx, &t_terms, ictx, &i_terms, bindings);
    }

    match (tctx.get(template).clone(), ictx.get(instance).clone()) {
        (Expr::Number(a), Expr::Number(b)) => (a == b).then_some(()),
        (Expr::Constant(a), Expr::Constant(b)) => (a == b).then_some(()),
        (Expr::Div(tl, tr), Expr::Div(il, ir)) => {
            match_expr(tctx, tl, ictx, il, bindings)?;
            match_expr(tctx, tr, ictx, ir, bindings)
        }
        (Expr::Pow(tl, tr), Expr::Pow(il, ir)) => {
            match_expr(tctx, tl, ictx, il, bindings)?;
            match_expr(tctx, tr, ictx, ir, bindings)
        }
        (Expr::Function(t_fn, t_args), Expr::Function(i_fn, i_args)) => {
            let same_builtin = match (tctx.builtin_of(t_fn), ictx.builtin_of(i_fn)) {
                (Some(a), Some(b)) => a == b,
                _ => tctx.sym_name(t_fn) == ictx.sym_name(i_fn),
            };
            if !same_builtin || t_args.len() != i_args.len() {
                return None;
            }
            for (&t_arg, &i_arg) in t_args.iter().zip(i_args.iter()) {
                match_expr(tctx, t_arg, ictx, i_arg, bindings)?;
            }
            Some(())
        }
        _ => None,
    }
}

/// Multiset matching with bounded backtracking: assign every template term to
/// a distinct instance term of the same sign. Terms are tried in template
/// order with bare metavariables LAST — they match anything, so committing
/// them early throws away the pruning the rigid terms provide.
fn match_multiset_signed(
    tctx: &Context,
    t_terms: &[(ExprId, Sign)],
    ictx: &Context,
    i_terms: &[(ExprId, Sign)],
    bindings: &mut Bindings,
) -> Option<()> {
    if t_terms.len() != i_terms.len() || t_terms.len() > MAX_CHAIN_TERMS {
        return None;
    }
    let mut order: Vec<usize> = (0..t_terms.len()).collect();
    order.sort_by_key(|&k| matches!(tctx.get(t_terms[k].0), Expr::Variable(_)));
    let mut used = vec![false; i_terms.len()];
    backtrack(tctx, t_terms, &order, 0, ictx, i_terms, &mut used, bindings)
}

#[allow(clippy::too_many_arguments)]
fn backtrack(
    tctx: &Context,
    t_terms: &[(ExprId, Sign)],
    order: &[usize],
    depth: usize,
    ictx: &Context,
    i_terms: &[(ExprId, Sign)],
    used: &mut [bool],
    bindings: &mut Bindings,
) -> Option<()> {
    if depth == order.len() {
        return Some(());
    }
    let (t_term, t_sign) = t_terms[order[depth]];
    for i in 0..i_terms.len() {
        if used[i] {
            continue;
        }
        let (i_term, i_sign) = i_terms[i];
        // Literal constant terms compare by SIGNED VALUE: the engine
        // canonicalizes `X − 1` into `X + (−1)`, so the template's `(1, Neg)`
        // must accept the instance's `(−1, Pos)`. Measured miss: the wire's
        // `2·cos(x)² − 1` failed the structural pass of its OWN template and
        // fell to the directed mode, which picked the Pythagoras-equivalent
        // sine form — a true title about a shape the reader was not seeing.
        let literal_verdict = match (tctx.get(t_term), ictx.get(i_term)) {
            (Expr::Number(a), Expr::Number(b)) => {
                let a_signed = if t_sign == Sign::Neg {
                    -a.clone()
                } else {
                    a.clone()
                };
                let b_signed = if i_sign == Sign::Neg {
                    -b.clone()
                } else {
                    b.clone()
                };
                Some(a_signed == b_signed)
            }
            _ => None,
        };
        if let Some(matches_value) = literal_verdict {
            if !matches_value {
                continue;
            }
            used[i] = true;
            let mut trial = bindings.clone();
            if backtrack(
                tctx,
                t_terms,
                order,
                depth + 1,
                ictx,
                i_terms,
                used,
                &mut trial,
            )
            .is_some()
            {
                *bindings = trial;
                return Some(());
            }
            used[i] = false;
            continue;
        }
        if i_sign != t_sign {
            continue;
        }
        let mut trial = bindings.clone();
        if match_expr(tctx, t_term, ictx, i_term, &mut trial).is_some() {
            used[i] = true;
            if backtrack(
                tctx,
                t_terms,
                order,
                depth + 1,
                ictx,
                i_terms,
                used,
                &mut trial,
            )
            .is_some()
            {
                *bindings = trial;
                return Some(());
            }
            used[i] = false;
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn instance_ctx(before: &str, after: &str) -> (Context, ExprId, ExprId) {
        let mut ctx = Context::new();
        let b = cas_parser::parse(before, &mut ctx).expect("parse before");
        let a = cas_parser::parse(after, &mut ctx).expect("parse after");
        (ctx, b, a)
    }

    /// The audit's first named lie, both faces. The TRUE application matches —
    /// with the binding as the positive witness — and the pair that is no
    /// instance of the template is refused.
    #[test]
    fn tan_cot_matches_its_instances_and_refuses_impostors() {
        let template = parse_template("tan(u) · cot(u)", "1").expect("template parses");
        let (ctx, b, a) = instance_ctx("tan(x^2) * cot(x^2)", "1");
        let bindings = match_instance(&template, &ctx, b, a).expect("true instance");
        assert_eq!(bindings.len(), 1, "one metavariable, one binding");

        // Commuted factors must still match: multiset, not sequence.
        let (ctx, b, a) = instance_ctx("cot(3*x) * tan(3*x)", "1");
        assert!(match_instance(&template, &ctx, b, a).is_some());

        // A pair that is NOT an application of the identity is refused.
        let (ctx, b, a) = instance_ctx("sin(x)", "1");
        assert!(match_instance(&template, &ctx, b, a).is_none());
    }

    /// Shared σ: `sin(u)·cos(u)` must never match mixed arguments.
    #[test]
    fn bindings_are_shared_across_the_whole_pair() {
        let template = parse_template("sin(u) · cos(u)", "sin(2u) / 2").expect("template");
        let (ctx, b, a) = instance_ctx("sin(x) * cos(x)", "sin(2*x) / 2");
        assert!(match_instance(&template, &ctx, b, a).is_some());
        let (ctx, b, a) = instance_ctx("sin(x) * cos(y)", "sin(2*x) / 2");
        assert!(
            match_instance(&template, &ctx, b, a).is_none(),
            "mixed arguments are not an instance"
        );
    }

    /// The audit's second named lie: the half-angle branch that recognized no
    /// variant still emitted «Usar tan(u) = (1 - cos(2u)) / sin(2u)». With the
    /// matcher, that title only survives over a genuine tan(σu) pair.
    #[test]
    fn half_angle_none_branch_shape_is_checkable() {
        let template =
            parse_template("tan(u)", "(1 - cos(2u)) / sin(2u)").expect("template parses");
        let (ctx, b, a) = instance_ctx("tan(x/2)", "(1 - cos(2*(x/2))) / sin(2*(x/2))");
        assert!(match_instance(&template, &ctx, b, a).is_some());
        // The unrecognized-variant pair the audit saw: not tan(anything).
        let (ctx, b, a) = instance_ctx("(1 - cos(x)) / sin(x)", "tan(x/2)");
        assert!(
            match_instance(&template, &ctx, b, a).is_none(),
            "the None branch's pair is not an instance of the title it published"
        );
    }

    /// The rewrite mode: the pythagorean-chain shape, where the template fires
    /// INSIDE a product and the whole pair is not one instance.
    #[test]
    fn rewrite_mode_finds_the_identity_inside_a_product() {
        let template = parse_template("1 - sin(u)^2", "cos(u)^2").expect("template parses");
        let (ctx, b, a) = instance_ctx("4*sin(x)*(1 - sin(x)^2)", "4*sin(x)*cos(x)^2");
        assert!(
            match_instance(&template, &ctx, b, a).is_none(),
            "the whole pair is not one instance…"
        );
        assert!(
            match_rewrite(&template, &ctx, b, a).is_some(),
            "…but it IS a rewrite at the inner subterm"
        );
        // And the hardcoded-coefficient lie: before says 6, narration says 4.
        let (ctx, b, a) = instance_ctx("6*sin(x)*(1 - sin(x)^2)", "4*sin(x)*cos(x)^2");
        assert!(match_rewrite(&template, &ctx, b, a).is_none());
    }

    /// Numbers are rigid: a template constant matches only itself.
    #[test]
    fn constants_do_not_bind() {
        let template = parse_template("2 · sin(u) · cos(u)", "sin(2u)").expect("template");
        let (ctx, b, a) = instance_ctx("2*sin(x)*cos(x)", "sin(2*x)");
        assert!(match_instance(&template, &ctx, b, a).is_some());
        let (ctx, b, a) = instance_ctx("3*sin(x)*cos(x)", "sin(2*x)");
        assert!(match_instance(&template, &ctx, b, a).is_none());
    }
}
