//! Numeric-presentation walker shared by `approx()` and the session-level
//! `--numeric-display` output boundary: map CLOSED numeric subtrees to
//! `decimal(...)` display nodes, keeping symbolic structure and exact
//! semantics (payloads are rounded exact rationals — never f64 state).

use cas_ast::{Context, Expr, ExprId};

/// Is this node the `decimal(Number)` display wrapper?
pub fn is_decimal_node(ctx: &Context, id: ExprId) -> bool {
    matches!(
        ctx.get(id),
        Expr::Function(fn_id, args)
            if args.len() == 1 && ctx.sym_name(*fn_id) == "decimal"
    )
}

/// Evaluate a CLOSED subexpression through the complex walker and emit the
/// cartesian `decimal(re) + decimal(im)·i` (pure-imaginary and finite-only;
/// a real result means the real f64 surface owns it — decline).
pub fn eval_closed_complex_to_decimal(ctx: &mut Context, id: ExprId) -> Option<ExprId> {
    let z = crate::evaluator_complex::eval_complex(ctx, id, &std::collections::HashMap::new())?;
    if !z.re.is_finite() || !z.im.is_finite() || z.im == 0.0 {
        return None;
    }
    let decimal_sym = ctx.intern_symbol("decimal");
    let im_rational = crate::decimal_display::approx_display_rational(z.im)?;
    let im_number = ctx.add(Expr::Number(im_rational));
    let im_decimal = ctx.add(Expr::Function(decimal_sym, vec![im_number]));
    let i = ctx.add(Expr::Constant(cas_ast::Constant::I));
    let im_part = ctx.add(Expr::Mul(im_decimal, i));
    if z.re == 0.0 {
        return Some(im_part);
    }
    let re_rational = crate::decimal_display::approx_display_rational(z.re)?;
    let re_number = ctx.add(Expr::Number(re_rational));
    let re_decimal = ctx.add(Expr::Function(decimal_sym, vec![re_number]));
    Some(ctx.add(Expr::Add(re_decimal, im_part)))
}

/// D5 triviality gate: a closed subtree is worth approximating only when it
/// contains a numeric leaf that is not integer-valued (a non-integer
/// rational, `pi`/`e`, a function call like `sqrt(2)`, a fractional power,
/// a non-exact quotient). Bare integers, bare `i`, and integer·`i` stay
/// symbolic — no `2 -> 2.0` churn, and `approx(x + i)` keeps declining.
pub fn closed_subtree_wants_decimal(ctx: &Context, id: ExprId) -> bool {
    match ctx.get(id) {
        Expr::Number(n) => !n.is_integer(),
        Expr::Constant(cas_ast::Constant::Pi | cas_ast::Constant::E | cas_ast::Constant::Phi) => {
            true
        }
        Expr::Constant(_) => false,
        Expr::Function(fn_id, _) => ctx.sym_name(*fn_id) != "decimal",
        Expr::Pow(b, e) => {
            closed_subtree_wants_decimal(ctx, *b) || closed_subtree_wants_decimal(ctx, *e)
        }
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) => {
            closed_subtree_wants_decimal(ctx, *l) || closed_subtree_wants_decimal(ctx, *r)
        }
        Expr::Div(l, r) => {
            if closed_subtree_wants_decimal(ctx, *l) || closed_subtree_wants_decimal(ctx, *r) {
                return true;
            }
            // Integer/integer with a non-exact quotient wants a decimal.
            if let (Expr::Number(a), Expr::Number(b)) = (ctx.get(*l), ctx.get(*r)) {
                use num_traits::Zero as _;
                return !b.is_zero() && !(a / b).is_integer();
            }
            false
        }
        Expr::Neg(inner) => closed_subtree_wants_decimal(ctx, *inner),
        _ => false,
    }
}

/// Evaluate a CLOSED numeric subexpression to its decimal form: the real
/// f64 surface first, the complex cartesian form when enabled.
pub fn eval_closed_to_decimal(
    ctx: &mut Context,
    id: ExprId,
    complex_enabled: bool,
) -> Option<ExprId> {
    if let Some(value) = crate::rootsum_numeric::numeric_eval_with_rootsum(ctx, id) {
        let rational = crate::decimal_display::approx_display_rational(value)?;
        let number = ctx.add(Expr::Number(rational));
        let decimal_sym = ctx.intern_symbol("decimal");
        return Some(ctx.add(Expr::Function(decimal_sym, vec![number])));
    }
    if complex_enabled {
        return eval_closed_complex_to_decimal(ctx, id);
    }
    None
}

/// Map the MAXIMAL closed numeric subtrees of an open expression to decimal
/// coefficients (`None` = no change). Structure-only recursion: Add/Sub and
/// Mul spines partition through the n-ary views (canonical ordering
/// interleaves closed and open operands — `pi·e·x·sqrt(2)` must yield ONE
/// coefficient), Div and Pow-bases recurse, and `Expr::Function` args and
/// matrices are NEVER entered: closed pointwise calls are caught whole at
/// their own node, and recursing into args would decimal-ize inert
/// `integrate`/`root_sum`/`solve` bounds whose exact re-evaluation the
/// soundness gates depend on. Exponents stay exact in v1.
pub fn approx_closed_subtrees(
    ctx: &mut Context,
    id: ExprId,
    complex_enabled: bool,
) -> Option<ExprId> {
    use crate::expr_predicates::contains_variable;

    // A closed node: evaluate whole (D5-gated); trivial closed stays.
    if !contains_variable(ctx, id) {
        if closed_subtree_wants_decimal(ctx, id) {
            return eval_closed_to_decimal(ctx, id, complex_enabled);
        }
        return None;
    }

    match ctx.get(id).clone() {
        Expr::Add(_, _) | Expr::Sub(_, _) | Expr::Neg(_) => {
            let view = crate::expr_nary::AddView::from_expr(ctx, id);
            let mut closed: Vec<(ExprId, crate::expr_nary::Sign)> = Vec::new();
            let mut open: Vec<(ExprId, crate::expr_nary::Sign)> = Vec::new();
            for (term, sign) in view.terms.iter().copied() {
                if contains_variable(ctx, term) {
                    open.push((term, sign));
                } else {
                    closed.push((term, sign));
                }
            }
            let closed_wants = closed
                .iter()
                .any(|(t, _)| closed_subtree_wants_decimal(ctx, *t));
            let mut changed = false;
            let mut new_terms: smallvec::SmallVec<[(ExprId, crate::expr_nary::Sign); 8]> =
                smallvec::SmallVec::new();
            if closed_wants && !closed.is_empty() {
                let closed_view = crate::expr_nary::AddView {
                    root: id,
                    terms: closed.iter().copied().collect(),
                };
                let closed_sum = closed_view.rebuild(ctx);
                if let Some(decimal) = eval_closed_to_decimal(ctx, closed_sum, complex_enabled) {
                    new_terms.push((decimal, crate::expr_nary::Sign::Pos));
                    changed = true;
                } else {
                    new_terms.extend(closed.iter().copied());
                }
            } else {
                new_terms.extend(closed.iter().copied());
            }
            for (term, sign) in open {
                match approx_closed_subtrees(ctx, term, complex_enabled) {
                    Some(new_term) => {
                        new_terms.push((new_term, sign));
                        changed = true;
                    }
                    None => new_terms.push((term, sign)),
                }
            }
            if !changed {
                return None;
            }
            let rebuilt = crate::expr_nary::AddView {
                root: id,
                terms: new_terms,
            };
            Some(rebuilt.rebuild(ctx))
        }
        Expr::Mul(_, _) => {
            let view = crate::expr_nary::MulView::from_expr(ctx, id);
            let mut closed: Vec<ExprId> = Vec::new();
            let mut open: Vec<ExprId> = Vec::new();
            for factor in view.factors.iter().copied() {
                if contains_variable(ctx, factor) {
                    open.push(factor);
                } else {
                    closed.push(factor);
                }
            }
            let closed_wants = closed.iter().any(|t| closed_subtree_wants_decimal(ctx, *t));
            let mut changed = false;
            let mut new_factors: smallvec::SmallVec<[ExprId; 8]> = smallvec::SmallVec::new();
            if closed_wants && !closed.is_empty() {
                let closed_view = crate::expr_nary::MulView {
                    root: id,
                    factors: closed.iter().copied().collect(),
                    commutative: true,
                };
                let closed_prod = closed_view.rebuild(ctx);
                if let Some(decimal) = eval_closed_to_decimal(ctx, closed_prod, complex_enabled) {
                    new_factors.push(decimal);
                    changed = true;
                } else {
                    new_factors.extend(closed.iter().copied());
                }
            } else {
                new_factors.extend(closed.iter().copied());
            }
            for factor in open {
                match approx_closed_subtrees(ctx, factor, complex_enabled) {
                    Some(new_factor) => {
                        new_factors.push(new_factor);
                        changed = true;
                    }
                    None => new_factors.push(factor),
                }
            }
            if !changed {
                return None;
            }
            let rebuilt = crate::expr_nary::MulView {
                root: id,
                factors: new_factors,
                commutative: true,
            };
            Some(rebuilt.rebuild(ctx))
        }
        Expr::Div(num, den) => {
            let new_num = approx_closed_subtrees(ctx, num, complex_enabled);
            let new_den = approx_closed_subtrees(ctx, den, complex_enabled);
            if new_num.is_none() && new_den.is_none() {
                return None;
            }
            let n = new_num.unwrap_or(num);
            let d = new_den.unwrap_or(den);
            Some(ctx.add(Expr::Div(n, d)))
        }
        Expr::Pow(base, exp) => {
            // Exponents stay exact (x^(1/3), not x^0.333…): only the base
            // recurses in v1.
            let new_base = approx_closed_subtrees(ctx, base, complex_enabled)?;
            Some(ctx.add(Expr::Pow(new_base, exp)))
        }
        _ => None,
    }
}

/// Output-boundary presentation for the `--numeric-display decimal` axis:
/// the whole-result form of the walker. A fully closed result evaluates
/// whole (D5-gated); open results map their maximal closed subtrees.
/// `None` = the result already has its final presentation.
pub fn present_numeric(ctx: &mut Context, id: ExprId, complex_enabled: bool) -> Option<ExprId> {
    use crate::expr_predicates::contains_variable;
    if is_decimal_node(ctx, id) {
        return None;
    }
    if !contains_variable(ctx, id) {
        if closed_subtree_wants_decimal(ctx, id) {
            return eval_closed_to_decimal(ctx, id, complex_enabled);
        }
        return None;
    }
    approx_closed_subtrees(ctx, id, complex_enabled)
}
