use crate::expr_predicates::{is_one_expr, is_zero_expr};
use crate::expr_relations::extract_negated_inner;
use crate::pi_helpers::{build_pi_over_n, is_pi_over_n};
use cas_ast::{Context, Expr, ExprId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReciprocalTrigEvalKind {
    CotPiOver4,
    CotPiOver2,
    SecZero,
    CscPiOver2,
    ArccotOne,
    ArccotZero,
    ArcsecOne,
    ArccscOne,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReciprocalTrigEvalRewrite {
    pub rewritten: ExprId,
    pub kind: ReciprocalTrigEvalKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReciprocalTrigNegativeKind {
    Cot,
    Sec,
    Csc,
    Arccot,
    Arcsec,
    Arccsc,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReciprocalTrigNegativeRewrite {
    pub rewritten: ExprId,
    pub kind: ReciprocalTrigNegativeKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReciprocalTrigCompositionKind {
    CotArccot,
    SecArcsec,
    CscArccsc,
    ArccotCot,
    ArcsecSec,
    ArccscCsc,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReciprocalTrigCompositionRewrite {
    pub rewritten: ExprId,
    pub kind: ReciprocalTrigCompositionKind,
}

type EvalCheck = fn(&Context, ExprId) -> bool;
type ResultBuilder = fn(&mut Context) -> ExprId;

fn build_zero(ctx: &mut Context) -> ExprId {
    ctx.num(0)
}

fn build_one(ctx: &mut Context) -> ExprId {
    ctx.num(1)
}

fn build_pi_over_2(ctx: &mut Context) -> ExprId {
    build_pi_over_n(ctx, 2)
}

fn build_pi_over_4(ctx: &mut Context) -> ExprId {
    build_pi_over_n(ctx, 4)
}

const EVAL_RULES: &[(&str, EvalCheck, ResultBuilder, ReciprocalTrigEvalKind)] = &[
    (
        "cot",
        |ctx, e| is_pi_over_n(ctx, e, 4),
        build_one,
        ReciprocalTrigEvalKind::CotPiOver4,
    ),
    (
        "cot",
        |ctx, e| is_pi_over_n(ctx, e, 2),
        build_zero,
        ReciprocalTrigEvalKind::CotPiOver2,
    ),
    (
        "sec",
        |ctx, e| is_zero_expr(ctx, e),
        build_one,
        ReciprocalTrigEvalKind::SecZero,
    ),
    (
        "csc",
        |ctx, e| is_pi_over_n(ctx, e, 2),
        build_one,
        ReciprocalTrigEvalKind::CscPiOver2,
    ),
    (
        "arccot",
        |ctx, e| is_one_expr(ctx, e),
        build_pi_over_4,
        ReciprocalTrigEvalKind::ArccotOne,
    ),
    (
        "arccot",
        |ctx, e| is_zero_expr(ctx, e),
        build_pi_over_2,
        ReciprocalTrigEvalKind::ArccotZero,
    ),
    (
        "arcsec",
        |ctx, e| is_one_expr(ctx, e),
        build_zero,
        ReciprocalTrigEvalKind::ArcsecOne,
    ),
    (
        "arccsc",
        |ctx, e| is_one_expr(ctx, e),
        build_pi_over_2,
        ReciprocalTrigEvalKind::ArccscOne,
    ),
];

/// Evaluate exact reciprocal trig and inverse-reciprocal trig table values.
pub fn eval_reciprocal_trig_value(
    ctx: &mut Context,
    fn_name: &str,
    arg: ExprId,
) -> Option<(ExprId, ReciprocalTrigEvalKind)> {
    for (func, check, build, desc) in EVAL_RULES {
        if fn_name == *func && check(ctx, arg) {
            return Some((build(ctx), *desc));
        }
    }
    None
}

/// Match and rewrite known table values for reciprocal trig calls.
pub fn try_rewrite_eval_reciprocal_trig_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ReciprocalTrigEvalRewrite> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }
    let fn_name = ctx.builtin_of(*fn_id)?.name();
    let (rewritten, desc) = eval_reciprocal_trig_value(ctx, fn_name, args[0])?;
    Some(ReciprocalTrigEvalRewrite {
        rewritten,
        kind: desc,
    })
}

const COMPOSITION_PAIRS: &[(&str, &str)] = &[
    ("cot", "arccot"),
    ("sec", "arcsec"),
    ("csc", "arccsc"),
    ("arccot", "cot"),
    ("arcsec", "sec"),
    ("arccsc", "csc"),
];

/// Returns true for identities of form `f(g(x)) = x` in reciprocal trig pairs.
pub fn is_reciprocal_trig_composition(outer_name: &str, inner_name: &str) -> bool {
    COMPOSITION_PAIRS
        .iter()
        .any(|(outer, inner)| outer_name == *outer && inner_name == *inner)
}

/// Match and rewrite reciprocal-trig compositions of form `f(g(x))`.
pub fn try_rewrite_reciprocal_trig_composition_expr(
    ctx: &Context,
    expr: ExprId,
) -> Option<ReciprocalTrigCompositionRewrite> {
    let Expr::Function(outer_fn_id, outer_args) = ctx.get(expr) else {
        return None;
    };
    if outer_args.len() != 1 {
        return None;
    }
    let inner_expr = outer_args[0];
    let Expr::Function(inner_fn_id, inner_args) = ctx.get(inner_expr) else {
        return None;
    };
    if inner_args.len() != 1 {
        return None;
    }

    let outer_name = ctx.builtin_of(*outer_fn_id)?.name();
    let inner_name = ctx.builtin_of(*inner_fn_id)?.name();
    if !is_reciprocal_trig_composition(outer_name, inner_name) {
        return None;
    }

    let kind = match (outer_name, inner_name) {
        ("cot", "arccot") => ReciprocalTrigCompositionKind::CotArccot,
        ("sec", "arcsec") => ReciprocalTrigCompositionKind::SecArcsec,
        ("csc", "arccsc") => ReciprocalTrigCompositionKind::CscArccsc,
        ("arccot", "cot") => ReciprocalTrigCompositionKind::ArccotCot,
        ("arcsec", "sec") => ReciprocalTrigCompositionKind::ArcsecSec,
        ("arccsc", "csc") => ReciprocalTrigCompositionKind::ArccscCsc,
        _ => return None,
    };

    Some(ReciprocalTrigCompositionRewrite {
        rewritten: inner_args[0],
        kind,
    })
}

#[derive(Clone, Copy)]
enum NegBehavior {
    Odd,
    Even,
    PiMinus,
}

const NEG_BEHAVIORS: &[(&str, NegBehavior)] = &[
    ("cot", NegBehavior::Odd),
    ("sec", NegBehavior::Even),
    ("csc", NegBehavior::Odd),
    ("arccot", NegBehavior::Odd),
    ("arcsec", NegBehavior::PiMinus),
    ("arccsc", NegBehavior::Odd),
];

/// Rewrite reciprocal trig negative arguments using parity/principal-branch identities.
pub fn rewrite_negative_reciprocal_trig_argument(
    ctx: &mut Context,
    fn_name: &str,
    arg: ExprId,
) -> Option<(ExprId, ReciprocalTrigNegativeKind)> {
    let inner = extract_negated_inner(ctx, arg)?;
    for (func, behavior) in NEG_BEHAVIORS {
        if fn_name != *func {
            continue;
        }
        let f_inner = ctx.call(func, vec![inner]);
        let out = match behavior {
            NegBehavior::Odd => (
                ctx.add(Expr::Neg(f_inner)),
                match fn_name {
                    "cot" => ReciprocalTrigNegativeKind::Cot,
                    "csc" => ReciprocalTrigNegativeKind::Csc,
                    "arccot" => ReciprocalTrigNegativeKind::Arccot,
                    "arccsc" => ReciprocalTrigNegativeKind::Arccsc,
                    _ => return None,
                },
            ),
            NegBehavior::Even => (
                f_inner,
                match fn_name {
                    "sec" => ReciprocalTrigNegativeKind::Sec,
                    _ => return None,
                },
            ),
            NegBehavior::PiMinus => {
                let pi = ctx.add(Expr::Constant(cas_ast::Constant::Pi));
                (
                    ctx.add(Expr::Sub(pi, f_inner)),
                    match fn_name {
                        "arcsec" => ReciprocalTrigNegativeKind::Arcsec,
                        _ => return None,
                    },
                )
            }
        };
        return Some(out);
    }
    None
}

/// Match and rewrite reciprocal trig negative-argument identities.
pub fn try_rewrite_negative_reciprocal_trig_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ReciprocalTrigNegativeRewrite> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }
    let fn_name = ctx.builtin_of(*fn_id)?.name();
    let (rewritten, desc) = rewrite_negative_reciprocal_trig_argument(ctx, fn_name, args[0])?;
    Some(ReciprocalTrigNegativeRewrite {
        rewritten,
        kind: desc,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    #[test]
    fn evaluates_known_reciprocal_trig_values() {
        let mut ctx = Context::new();
        let arg = parse("pi/4", &mut ctx).expect("arg");
        let (out, kind) = eval_reciprocal_trig_value(&mut ctx, "cot", arg).expect("eval");
        assert_eq!(kind, ReciprocalTrigEvalKind::CotPiOver4);
        assert!(
            matches!(ctx.get(out), Expr::Number(n) if *n == num_rational::BigRational::from_integer(1.into()))
        );
    }

    #[test]
    fn composition_pairs_detected() {
        assert!(is_reciprocal_trig_composition("cot", "arccot"));
        assert!(is_reciprocal_trig_composition("arcsec", "sec"));
        assert!(!is_reciprocal_trig_composition("cot", "arcsec"));
    }

    #[test]
    fn rewrites_negative_arcsec_argument() {
        let mut ctx = Context::new();
        let x = parse("x", &mut ctx).expect("x");
        let neg_x = ctx.add(Expr::Neg(x));
        let (out, kind) =
            rewrite_negative_reciprocal_trig_argument(&mut ctx, "arcsec", neg_x).expect("rw");
        assert_eq!(kind, ReciprocalTrigNegativeKind::Arcsec);
        match ctx.get(out) {
            Expr::Sub(l, r) => {
                assert!(matches!(ctx.get(*l), Expr::Constant(cas_ast::Constant::Pi)));
                match ctx.get(*r) {
                    Expr::Function(_, args) => assert_eq!(args.as_slice(), &[x]),
                    _ => panic!("expected arcsec(x)"),
                }
            }
            _ => panic!("expected subtraction expression"),
        }
    }

    #[test]
    fn rewrites_eval_expression_shape() {
        let mut ctx = Context::new();
        let expr = parse("cot(pi/4)", &mut ctx).expect("expr");
        let rewrite = try_rewrite_eval_reciprocal_trig_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(rewrite.kind, ReciprocalTrigEvalKind::CotPiOver4);
        assert!(
            matches!(ctx.get(rewrite.rewritten), Expr::Number(n) if *n == num_rational::BigRational::from_integer(1.into()))
        );
    }

    #[test]
    fn rewrites_composition_expression_shape() {
        let mut ctx = Context::new();
        let expr = parse("sec(arcsec(x))", &mut ctx).expect("expr");
        let rewrite = try_rewrite_reciprocal_trig_composition_expr(&ctx, expr).expect("rewrite");
        assert_eq!(rewrite.kind, ReciprocalTrigCompositionKind::SecArcsec);
        assert_eq!(rewrite.rewritten, parse("x", &mut ctx).expect("x"));
    }

    #[test]
    fn rewrites_negative_expression_shape() {
        let mut ctx = Context::new();
        let expr = parse("arcsec(-x)", &mut ctx).expect("expr");
        let rewrite = try_rewrite_negative_reciprocal_trig_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(rewrite.kind, ReciprocalTrigNegativeKind::Arcsec);
    }
}
