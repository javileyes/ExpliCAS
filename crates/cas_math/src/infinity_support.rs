use crate::expr_predicates::is_zero_expr;
use cas_ast::{Constant, Context, Expr, ExprId};
use num_traits::Signed;

/// Sign of infinity: positive (+∞) or negative (−∞).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InfSign {
    Pos,
    Neg,
}

/// Classification of an expression's finiteness.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Finiteness {
    /// Expression is a known finite value (number, π, e, i)
    FiniteLiteral,
    /// Expression is ±∞ with known sign
    Infinity(InfSign),
    /// Unknown finiteness - could be finite, infinite, or undefined
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InfinityRewritePlan {
    pub rewritten: ExprId,
    pub description: String,
}

/// Detect if an expression is ±∞.
pub fn inf_sign(ctx: &Context, id: ExprId) -> Option<InfSign> {
    match ctx.get(id) {
        Expr::Constant(Constant::Infinity) => Some(InfSign::Pos),
        Expr::Neg(inner) => match ctx.get(*inner) {
            Expr::Constant(Constant::Infinity) => Some(InfSign::Neg),
            _ => None,
        },
        _ => None,
    }
}

/// Construct ±∞.
pub fn mk_infinity(ctx: &mut Context, sign: InfSign) -> ExprId {
    let inf = ctx.add(Expr::Constant(Constant::Infinity));
    match sign {
        InfSign::Pos => inf,
        InfSign::Neg => ctx.add(Expr::Neg(inf)),
    }
}

/// Construct Undefined (for indeterminate forms).
pub fn mk_undefined(ctx: &mut Context) -> ExprId {
    ctx.add(Expr::Constant(Constant::Undefined))
}

/// Check if an expression is a known finite literal.
///
/// Conservative policy: only true for expressions we KNOW are finite:
/// - Numbers (BigRational)
/// - Constants that are not Infinity or Undefined (π, e, i)
pub fn is_finite_literal(ctx: &Context, id: ExprId) -> bool {
    match ctx.get(id) {
        Expr::Number(_) => true,
        Expr::Constant(c) => !matches!(c, Constant::Infinity | Constant::Undefined),
        _ => false,
    }
}

/// Classify an expression's finiteness.
pub fn classify_finiteness(ctx: &Context, id: ExprId) -> Finiteness {
    if let Some(sign) = inf_sign(ctx, id) {
        return Finiteness::Infinity(sign);
    }

    if is_finite_literal(ctx, id) {
        return Finiteness::FiniteLiteral;
    }

    Finiteness::Unknown
}

/// Check if expression is a positive finite literal (for sign determination).
pub fn is_positive_literal(ctx: &Context, id: ExprId) -> bool {
    match ctx.get(id) {
        Expr::Number(n) => n.is_positive(),
        Expr::Constant(c) => matches!(c, Constant::Pi | Constant::E), // π and e are positive
        _ => false,
    }
}

/// Check if expression is a negative finite literal.
pub fn is_negative_literal(ctx: &Context, id: ExprId) -> bool {
    match ctx.get(id) {
        Expr::Number(n) => n.is_negative(),
        Expr::Neg(inner) => is_positive_literal(ctx, *inner),
        _ => false,
    }
}

/// Collect additive terms with their signs (iterative, handles Sub).
pub fn collect_add_terms_with_sign(
    ctx: &Context,
    id: ExprId,
    is_positive: bool,
    terms: &mut Vec<(ExprId, bool)>,
) {
    let mut stack = vec![(id, is_positive)];

    while let Some((current, sign)) = stack.pop() {
        match ctx.get(current) {
            Expr::Add(l, r) => {
                stack.push((*r, sign));
                stack.push((*l, sign));
            }
            Expr::Sub(l, r) => {
                stack.push((*r, !sign)); // Right side gets inverted sign
                stack.push((*l, sign));
            }
            Expr::Neg(inner) => {
                stack.push((*inner, !sign));
            }
            _ => terms.push((current, sign)),
        }
    }
}

/// Plan infinity absorption in addition.
///
/// - `finite + ∞ -> ∞`
/// - `finite + (-∞) -> -∞`
/// - `∞ + (-∞) -> Undefined`
pub fn try_rewrite_add_infinity_absorption_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<InfinityRewritePlan> {
    let mut terms = Vec::new();
    collect_add_terms_with_sign(ctx, expr, true, &mut terms);

    let mut has_pos_inf = false;
    let mut has_neg_inf = false;

    for &(term, is_positive) in &terms {
        if let Some(s) = inf_sign(ctx, term) {
            let total = match (is_positive, s) {
                (true, InfSign::Pos) => InfSign::Pos,
                (false, InfSign::Pos) => InfSign::Neg,
                (true, InfSign::Neg) => InfSign::Neg,
                (false, InfSign::Neg) => InfSign::Pos,
            };
            match total {
                InfSign::Pos => has_pos_inf = true,
                InfSign::Neg => has_neg_inf = true,
            }
        } else if !is_finite_literal(ctx, term) {
            return None;
        }
    }

    let (rewritten, description) = match (has_pos_inf, has_neg_inf) {
        (true, true) => (mk_undefined(ctx), "∞ + (-∞) is indeterminate".to_string()),
        (true, false) => (
            mk_infinity(ctx, InfSign::Pos),
            "finite + ∞ -> ∞".to_string(),
        ),
        (false, true) => (
            mk_infinity(ctx, InfSign::Neg),
            "finite + (-∞) -> -∞".to_string(),
        ),
        (false, false) => return None,
    };

    Some(InfinityRewritePlan {
        rewritten,
        description,
    })
}

/// Plan division by infinity.
///
/// `finite / ±∞ -> 0` only when numerator is finite literal.
pub fn try_rewrite_div_by_infinity_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<InfinityRewritePlan> {
    let (num, den) = if let Expr::Div(num, den) = ctx.get(expr) {
        (*num, *den)
    } else {
        return None;
    };

    if inf_sign(ctx, den).is_some() && is_finite_literal(ctx, num) {
        Some(InfinityRewritePlan {
            rewritten: ctx.num(0),
            description: "finite / ∞ -> 0".to_string(),
        })
    } else {
        None
    }
}

/// Plan indeterminate form `0 * ∞ -> Undefined`.
pub fn try_rewrite_mul_zero_infinity_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<InfinityRewritePlan> {
    let (a, b) = if let Expr::Mul(a, b) = ctx.get(expr) {
        (*a, *b)
    } else {
        return None;
    };

    let a_inf = inf_sign(ctx, a).is_some();
    let b_inf = inf_sign(ctx, b).is_some();
    let a_zero = is_zero_expr(ctx, a);
    let b_zero = is_zero_expr(ctx, b);

    if (a_zero && b_inf) || (b_zero && a_inf) {
        Some(InfinityRewritePlan {
            rewritten: mk_undefined(ctx),
            description: "0 · ∞ is indeterminate".to_string(),
        })
    } else {
        None
    }
}

/// Plan finite-times-infinity multiplication.
///
/// `finite(non-zero) * ±∞ -> ±∞`.
pub fn try_rewrite_mul_finite_infinity_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<InfinityRewritePlan> {
    let (a, b) = if let Expr::Mul(a, b) = ctx.get(expr) {
        (*a, *b)
    } else {
        return None;
    };

    if let Some(inf_s) = inf_sign(ctx, a) {
        if is_finite_literal(ctx, b) && !is_zero_expr(ctx, b) {
            let b_negative = is_negative_literal(ctx, b);
            let result_sign = if b_negative {
                match inf_s {
                    InfSign::Pos => InfSign::Neg,
                    InfSign::Neg => InfSign::Pos,
                }
            } else {
                inf_s
            };
            return Some(InfinityRewritePlan {
                rewritten: mk_infinity(ctx, result_sign),
                description: format!("finite * ∞ -> {:?}∞", result_sign),
            });
        }
    }

    if let Some(inf_s) = inf_sign(ctx, b) {
        if is_finite_literal(ctx, a) && !is_zero_expr(ctx, a) {
            let a_negative = is_negative_literal(ctx, a);
            let result_sign = if a_negative {
                match inf_s {
                    InfSign::Pos => InfSign::Neg,
                    InfSign::Neg => InfSign::Pos,
                }
            } else {
                inf_s
            };
            return Some(InfinityRewritePlan {
                rewritten: mk_infinity(ctx, result_sign),
                description: format!("finite * ∞ -> {:?}∞", result_sign),
            });
        }
    }

    None
}

/// Plan `±∞ / finite(non-zero) -> ±∞`.
pub fn try_rewrite_inf_div_finite_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<InfinityRewritePlan> {
    let (num, den) = if let Expr::Div(num, den) = ctx.get(expr) {
        (*num, *den)
    } else {
        return None;
    };

    let inf_s = inf_sign(ctx, num)?;
    if !is_finite_literal(ctx, den) || is_zero_expr(ctx, den) {
        return None;
    }

    let den_negative = is_negative_literal(ctx, den);
    let result_sign = if den_negative {
        match inf_s {
            InfSign::Pos => InfSign::Neg,
            InfSign::Neg => InfSign::Pos,
        }
    } else {
        inf_s
    };

    Some(InfinityRewritePlan {
        rewritten: mk_infinity(ctx, result_sign),
        description: format!("∞ / finite -> {:?}∞", result_sign),
    })
}

/// True if `expr` evaluates to `±∞` — bare `±∞`, or a finite NON-ZERO scalar multiple of it
/// (`2·∞`, `(-∞)`, `∞·5`, …). Used to recognise the indeterminate `∞/∞` regardless of finite scaling.
pub fn is_infinite_valued(ctx: &Context, id: ExprId) -> bool {
    match ctx.get(id) {
        Expr::Constant(Constant::Infinity) => true,
        Expr::Neg(inner) => is_infinite_valued(ctx, *inner),
        Expr::Mul(a, b) => {
            let (a, b) = (*a, *b);
            (is_infinite_valued(ctx, a) && is_finite_literal(ctx, b) && !is_zero_expr(ctx, b))
                || (is_finite_literal(ctx, a)
                    && !is_zero_expr(ctx, a)
                    && is_infinite_valued(ctx, b))
        }
        _ => false,
    }
}

/// Whether `id` carries the `∞` constant as a (possibly nested) MULTIPLICATIVE factor, making its
/// value non-finite (`±∞`) or indeterminate (`0·∞`) regardless of the other, finite cofactors.
///
/// This is a GUARD predicate, deliberately broader than [`is_infinite_valued`] (which stays precise
/// — only `±∞` and `c·∞` for a finite NONZERO literal `c` — because it drives the `→ Undefined`
/// rewrite trigger). `contains_unbounded_factor` additionally flags `x·∞` (symbolic cofactor) and
/// multi-factor products like `2·∞·sin(x)`, since a quotient of two such expressions is `∞/∞`,
/// indeterminate, for every value of the cofactors (`x·∞ / (2·x·∞)` is `∞/∞` when `x≠0` and
/// `undefined/undefined` when `x=0` — never `1/2`). It does NOT recurse into `Div`/`Pow`/`Add`, so a
/// finite expression that merely mentions `∞` (`1/∞ = 0`, `∞+1`, `∞^0`) is correctly NOT flagged.
pub fn contains_unbounded_factor(ctx: &Context, id: ExprId) -> bool {
    match ctx.get(id) {
        Expr::Constant(Constant::Infinity) => true,
        Expr::Neg(inner) => contains_unbounded_factor(ctx, *inner),
        Expr::Mul(a, b) => contains_unbounded_factor(ctx, *a) || contains_unbounded_factor(ctx, *b),
        _ => false,
    }
}

/// Plan the indeterminate form `±∞ / ±∞ -> Undefined` (including finite-scaled infinities like
/// `(2·∞)/(5·∞)`, symbolic-scaled `(x·∞)/(2·x·∞)`, and multi-factor `(2·∞·sin x)/(5·∞·sin x)`).
/// Without this, the generic common-factor cancellation `(a·X)/(b·X) -> a/b` treats `∞` as a
/// cancellable factor and fabricates a finite value (`∞/∞ -> 1`, `(2·∞)/(5·∞) -> 2/5`,
/// `(x·∞)/(2·x·∞) -> 1`). Uses [`contains_unbounded_factor`] (not [`is_infinite_valued`]) so the
/// symbolic and multi-factor shapes are covered: a quotient of two expressions that each carry an
/// `∞` factor is `∞/∞`, indeterminate, regardless of the finite cofactors.
pub fn try_rewrite_inf_div_inf_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<InfinityRewritePlan> {
    let (num, den) = if let Expr::Div(num, den) = ctx.get(expr) {
        (*num, *den)
    } else {
        return None;
    };

    if contains_unbounded_factor(ctx, num) && contains_unbounded_factor(ctx, den) {
        Some(InfinityRewritePlan {
            rewritten: mk_undefined(ctx),
            description: "∞ / ∞ is indeterminate".to_string(),
        })
    } else {
        None
    }
}

/// Saturating / monotone functions evaluated at ±∞ on the extended
/// real line. Handles both the function form `f(±∞)` and the
/// exponential power form `e^(±∞)`:
/// - `arctan(±∞) -> ±π/2`, `tanh(±∞) -> ±1`
/// - `exp(+∞) -> +∞`, `exp(-∞) -> 0` (function and `e^x` power form)
/// - `ln(+∞) -> +∞`, `sqrt(+∞) -> +∞`
/// - `sinh(±∞) -> ±∞`, `cosh(±∞) -> +∞`
///
/// Conservative: only fires when the single argument is exactly ±∞.
/// Oscillating functions (sin, cos, tan) and domain-bounded inverses
/// (arcsin, arccos) are intentionally excluded - their limit at ∞ does
/// not exist or is undefined.
///
/// NOT a global simplification rule: it is applied ONLY to limit
/// outputs (via [`fold_infinity_saturation`]), where the limit engine
/// has already committed to a single signed infinity. Folding it
/// globally would materialize a bare `infinity` literal into raw
/// arithmetic, where pre-existing cancellation rules (a-a=0, a/a=1,
/// x^0=1) mishandle it - e.g. `sinh(∞) - cosh(∞)` would wrongly
/// collapse to 0 instead of staying an honest ∞ - ∞ form.
pub fn try_rewrite_function_at_infinity_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<InfinityRewritePlan> {
    use cas_ast::BuiltinFn;

    // e^(±∞) power form: Pow(E, ±∞). Also sqrt as Pow(±∞, 1/2).
    if let Expr::Pow(base, exponent) = ctx.get(expr).clone() {
        if matches!(ctx.get(base), Expr::Constant(Constant::E)) {
            return match inf_sign(ctx, exponent)? {
                InfSign::Pos => Some(InfinityRewritePlan {
                    rewritten: mk_infinity(ctx, InfSign::Pos),
                    description: "e^(+∞) -> +∞".to_string(),
                }),
                InfSign::Neg => Some(InfinityRewritePlan {
                    rewritten: ctx.add(Expr::Number(num_rational::BigRational::from_integer(
                        0.into(),
                    ))),
                    description: "e^(-∞) -> 0".to_string(),
                }),
            };
        }
        // (+∞)^positive_literal -> +∞ (covers sqrt(∞) = ∞^(1/2)).
        if inf_sign(ctx, base) == Some(InfSign::Pos) && is_positive_literal(ctx, exponent) {
            return Some(InfinityRewritePlan {
                rewritten: mk_infinity(ctx, InfSign::Pos),
                description: "(+∞)^positive -> +∞".to_string(),
            });
        }
    }

    let Expr::Function(fn_id, args) = ctx.get(expr).clone() else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }
    let builtin = ctx.builtin_of(fn_id)?;
    let sign = inf_sign(ctx, args[0])?;

    let half_pi = |ctx: &mut Context, negative: bool| -> ExprId {
        let pi = ctx.add(Expr::Constant(Constant::Pi));
        let two = ctx.add(Expr::Number(num_rational::BigRational::from_integer(
            2.into(),
        )));
        let half = ctx.add(Expr::Div(pi, two));
        if negative {
            ctx.add(Expr::Neg(half))
        } else {
            half
        }
    };
    let signed_one = |ctx: &mut Context, negative: bool| -> ExprId {
        let one = ctx.add(Expr::Number(num_rational::BigRational::from_integer(
            1.into(),
        )));
        if negative {
            ctx.add(Expr::Neg(one))
        } else {
            one
        }
    };
    let zero = |ctx: &mut Context| {
        ctx.add(Expr::Number(num_rational::BigRational::from_integer(
            0.into(),
        )))
    };

    let (rewritten, description) = match (builtin, sign) {
        // Arctan and its `atan` alias (the limit machinery substitutes
        // before atan -> arctan canonicalization).
        (BuiltinFn::Arctan | BuiltinFn::Atan, InfSign::Pos) => {
            (half_pi(ctx, false), "arctan(+∞) -> π/2")
        }
        (BuiltinFn::Arctan | BuiltinFn::Atan, InfSign::Neg) => {
            (half_pi(ctx, true), "arctan(-∞) -> -π/2")
        }
        (BuiltinFn::Tanh, InfSign::Pos) => (signed_one(ctx, false), "tanh(+∞) -> 1"),
        (BuiltinFn::Tanh, InfSign::Neg) => (signed_one(ctx, true), "tanh(-∞) -> -1"),
        (BuiltinFn::Exp, InfSign::Pos) => (mk_infinity(ctx, InfSign::Pos), "exp(+∞) -> +∞"),
        (BuiltinFn::Exp, InfSign::Neg) => (zero(ctx), "exp(-∞) -> 0"),
        (BuiltinFn::Ln, InfSign::Pos) => (mk_infinity(ctx, InfSign::Pos), "ln(+∞) -> +∞"),
        (BuiltinFn::Sinh, InfSign::Pos) => (mk_infinity(ctx, InfSign::Pos), "sinh(+∞) -> +∞"),
        (BuiltinFn::Sinh, InfSign::Neg) => (mk_infinity(ctx, InfSign::Neg), "sinh(-∞) -> -∞"),
        (BuiltinFn::Cosh, InfSign::Pos) | (BuiltinFn::Cosh, InfSign::Neg) => {
            (mk_infinity(ctx, InfSign::Pos), "cosh(±∞) -> +∞")
        }
        (BuiltinFn::Sqrt, InfSign::Pos) => (mk_infinity(ctx, InfSign::Pos), "sqrt(+∞) -> +∞"),
        _ => return None,
    };
    Some(InfinityRewritePlan {
        rewritten,
        description: description.to_string(),
    })
}

/// Recursively fold saturating functions at ±∞ bottom-up. Used to
/// clean limit outputs that substitute an infinite inner value into a
/// saturating outer function (e.g. `exp(-∞) -> 0`, `atan(∞) -> π/2`)
/// without running the full simplifier.
pub fn fold_infinity_saturation(ctx: &mut Context, expr: ExprId) -> ExprId {
    let node = ctx.get(expr).clone();
    let rebuilt = match node {
        Expr::Add(l, r) => {
            let l = fold_infinity_saturation(ctx, l);
            let r = fold_infinity_saturation(ctx, r);
            ctx.add(Expr::Add(l, r))
        }
        Expr::Sub(l, r) => {
            let l = fold_infinity_saturation(ctx, l);
            let r = fold_infinity_saturation(ctx, r);
            ctx.add(Expr::Sub(l, r))
        }
        Expr::Mul(l, r) => {
            let l = fold_infinity_saturation(ctx, l);
            let r = fold_infinity_saturation(ctx, r);
            ctx.add(Expr::Mul(l, r))
        }
        Expr::Div(l, r) => {
            let l = fold_infinity_saturation(ctx, l);
            let r = fold_infinity_saturation(ctx, r);
            ctx.add(Expr::Div(l, r))
        }
        Expr::Neg(inner) => {
            let inner = fold_infinity_saturation(ctx, inner);
            ctx.add(Expr::Neg(inner))
        }
        Expr::Pow(base, exponent) => {
            let base = fold_infinity_saturation(ctx, base);
            let exponent = fold_infinity_saturation(ctx, exponent);
            ctx.add(Expr::Pow(base, exponent))
        }
        Expr::Function(fn_id, args) => {
            let args: Vec<_> = args
                .iter()
                .map(|arg| fold_infinity_saturation(ctx, *arg))
                .collect();
            ctx.add(Expr::Function(fn_id, args))
        }
        _ => expr,
    };
    if let Some(plan) = try_rewrite_function_at_infinity_expr(ctx, rebuilt) {
        return plan.rewritten;
    }
    rebuilt
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;
    use num_traits::Zero;

    fn parse_expr(ctx: &mut Context, s: &str) -> ExprId {
        parse(s, ctx).expect("parse failed")
    }

    #[test]
    fn inf_sign_detects_both_orientations() {
        let mut ctx = Context::new();
        let pos = parse_expr(&mut ctx, "infinity");
        let neg = parse_expr(&mut ctx, "-infinity");
        let finite = parse_expr(&mut ctx, "7");

        assert_eq!(inf_sign(&ctx, pos), Some(InfSign::Pos));
        assert_eq!(inf_sign(&ctx, neg), Some(InfSign::Neg));
        assert_eq!(inf_sign(&ctx, finite), None);
    }

    #[test]
    fn classify_finiteness_keeps_unknown_for_undefined_and_vars() {
        let mut ctx = Context::new();
        let n = parse_expr(&mut ctx, "42");
        let inf = parse_expr(&mut ctx, "infinity");
        let x = parse_expr(&mut ctx, "x");
        let undef = parse_expr(&mut ctx, "undefined");

        assert_eq!(classify_finiteness(&ctx, n), Finiteness::FiniteLiteral);
        assert_eq!(
            classify_finiteness(&ctx, inf),
            Finiteness::Infinity(InfSign::Pos)
        );
        assert_eq!(classify_finiteness(&ctx, x), Finiteness::Unknown);
        assert_eq!(classify_finiteness(&ctx, undef), Finiteness::Unknown);
    }

    #[test]
    fn add_term_collection_tracks_signs_through_sub_and_neg() {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, "a - (b + -c)");
        let mut terms = Vec::new();
        collect_add_terms_with_sign(&ctx, expr, true, &mut terms);

        let mut rendered: Vec<(String, bool)> = terms
            .into_iter()
            .map(|(id, sign)| {
                let name = match ctx.get(id) {
                    Expr::Variable(sym) => ctx.sym_name(*sym).to_string(),
                    _ => panic!("expected variable term"),
                };
                (name, sign)
            })
            .collect();
        rendered.sort();

        assert_eq!(
            rendered,
            vec![
                ("a".to_string(), true),
                ("b".to_string(), false),
                ("c".to_string(), true)
            ]
        );
    }

    #[test]
    fn literal_sign_helpers_handle_numbers_and_constants() {
        let mut ctx = Context::new();
        let pos = parse_expr(&mut ctx, "pi");
        let neg = parse_expr(&mut ctx, "-2");
        let imag = parse_expr(&mut ctx, "i");

        assert!(is_positive_literal(&ctx, pos));
        assert!(is_negative_literal(&ctx, neg));
        assert!(!is_positive_literal(&ctx, imag));
        assert!(!is_negative_literal(&ctx, imag));
    }

    #[test]
    fn add_infinity_absorption_rewrites_finite_plus_infinity() {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, "1000 + infinity");
        let plan =
            try_rewrite_add_infinity_absorption_expr(&mut ctx, expr).expect("should rewrite");
        assert!(matches!(
            ctx.get(plan.rewritten),
            Expr::Constant(Constant::Infinity)
        ));
    }

    #[test]
    fn div_by_infinity_rewrites_to_zero() {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, "5 / infinity");
        let plan = try_rewrite_div_by_infinity_expr(&mut ctx, expr).expect("should rewrite");
        match ctx.get(plan.rewritten) {
            Expr::Number(n) => assert!(n.is_zero()),
            _ => panic!("expected numeric zero"),
        }
    }

    #[test]
    fn mul_zero_infinity_rewrites_to_undefined() {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, "0 * infinity");
        let plan = try_rewrite_mul_zero_infinity_expr(&mut ctx, expr).expect("should rewrite");
        assert!(matches!(
            ctx.get(plan.rewritten),
            Expr::Constant(Constant::Undefined)
        ));
    }

    #[test]
    fn mul_finite_infinity_preserves_sign() {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, "(-2) * infinity");
        let plan = try_rewrite_mul_finite_infinity_expr(&mut ctx, expr).expect("should rewrite");
        match ctx.get(plan.rewritten) {
            Expr::Neg(inner) => assert!(matches!(
                ctx.get(*inner),
                Expr::Constant(Constant::Infinity)
            )),
            _ => panic!("expected negative infinity"),
        }
    }

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", cas_formatter::DisplayExpr { context: ctx, id })
    }

    #[test]
    fn function_at_infinity_folds_saturating_functions() {
        let cases = [
            ("arctan(infinity)", "pi / 2"),
            ("arctan(-infinity)", "-pi / 2"),
            ("tanh(infinity)", "1"),
            ("tanh(-infinity)", "-1"),
            ("exp(infinity)", "infinity"),
            ("exp(-infinity)", "0"),
            ("ln(infinity)", "infinity"),
            ("sinh(infinity)", "infinity"),
            ("sinh(-infinity)", "-infinity"),
            ("cosh(-infinity)", "infinity"),
        ];
        for (source, expected) in cases {
            let mut ctx = Context::new();
            let expr = parse_expr(&mut ctx, source);
            let plan = try_rewrite_function_at_infinity_expr(&mut ctx, expr).expect(source);
            assert_eq!(rendered(&ctx, plan.rewritten), expected, "{source}");
        }
    }

    #[test]
    fn function_at_infinity_folds_exponential_power_form() {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, "e^infinity");
        let plan = try_rewrite_function_at_infinity_expr(&mut ctx, expr).expect("e^inf");
        assert_eq!(rendered(&ctx, plan.rewritten), "infinity");

        let expr = parse_expr(&mut ctx, "e^(-infinity)");
        let plan = try_rewrite_function_at_infinity_expr(&mut ctx, expr).expect("e^-inf");
        assert_eq!(rendered(&ctx, plan.rewritten), "0");

        // sqrt(infinity) = infinity^(1/2).
        let expr = parse_expr(&mut ctx, "sqrt(infinity)");
        let plan = try_rewrite_function_at_infinity_expr(&mut ctx, expr).expect("sqrt inf");
        assert_eq!(rendered(&ctx, plan.rewritten), "infinity");
    }

    #[test]
    fn function_at_infinity_declines_oscillating_and_finite_args() {
        // Oscillating functions and finite arguments must NOT fold:
        // their limit at infinity does not exist.
        for source in [
            "sin(infinity)",
            "cos(infinity)",
            "tan(infinity)",
            "arctan(2)",
            "exp(3)",
        ] {
            let mut ctx = Context::new();
            let expr = parse_expr(&mut ctx, source);
            assert!(
                try_rewrite_function_at_infinity_expr(&mut ctx, expr).is_none(),
                "must decline: {source}"
            );
        }
    }

    #[test]
    fn fold_infinity_saturation_handles_atan_alias_and_nesting() {
        // The limit machinery emits the `atan` alias (BuiltinFn::Atan)
        // before canonicalization; the fold must still resolve it.
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, "atan(infinity)");
        let folded = fold_infinity_saturation(&mut ctx, expr);
        assert_eq!(rendered(&ctx, folded), "pi / 2");

        // Nested: exp(-infinity) inside a sum folds bottom-up.
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, "exp(-infinity) + 5");
        let folded = fold_infinity_saturation(&mut ctx, expr);
        assert_eq!(rendered(&ctx, folded), "0 + 5");
    }

    #[test]
    fn inf_div_finite_preserves_sign() {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, "infinity / (-3)");
        let plan = try_rewrite_inf_div_finite_expr(&mut ctx, expr).expect("should rewrite");
        match ctx.get(plan.rewritten) {
            Expr::Neg(inner) => assert!(matches!(
                ctx.get(*inner),
                Expr::Constant(Constant::Infinity)
            )),
            _ => panic!("expected negative infinity"),
        }
    }
}
