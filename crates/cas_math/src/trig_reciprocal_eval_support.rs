use crate::expr_predicates::{is_one_expr, is_zero_expr};
use crate::expr_relations::extract_negated_inner;
use crate::pi_helpers::{build_pi_over_n, is_pi_over_n};
use cas_ast::{Context, Expr, ExprId};

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

const EVAL_RULES: &[(&str, EvalCheck, ResultBuilder, &str)] = &[
    (
        "cot",
        |ctx, e| is_pi_over_n(ctx, e, 4),
        build_one,
        "cot(π/4) = 1",
    ),
    (
        "cot",
        |ctx, e| is_pi_over_n(ctx, e, 2),
        build_zero,
        "cot(π/2) = 0",
    ),
    (
        "sec",
        |ctx, e| is_zero_expr(ctx, e),
        build_one,
        "sec(0) = 1",
    ),
    (
        "csc",
        |ctx, e| is_pi_over_n(ctx, e, 2),
        build_one,
        "csc(π/2) = 1",
    ),
    (
        "arccot",
        |ctx, e| is_one_expr(ctx, e),
        build_pi_over_4,
        "arccot(1) = π/4",
    ),
    (
        "arccot",
        |ctx, e| is_zero_expr(ctx, e),
        build_pi_over_2,
        "arccot(0) = π/2",
    ),
    (
        "arcsec",
        |ctx, e| is_one_expr(ctx, e),
        build_zero,
        "arcsec(1) = 0",
    ),
    (
        "arccsc",
        |ctx, e| is_one_expr(ctx, e),
        build_pi_over_2,
        "arccsc(1) = π/2",
    ),
];

/// Evaluate exact reciprocal trig and inverse-reciprocal trig table values.
pub fn eval_reciprocal_trig_value(
    ctx: &mut Context,
    fn_name: &str,
    arg: ExprId,
) -> Option<(ExprId, &'static str)> {
    for (func, check, build, desc) in EVAL_RULES {
        if fn_name == *func && check(ctx, arg) {
            return Some((build(ctx), desc));
        }
    }
    None
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
) -> Option<(ExprId, &'static str)> {
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
                    "cot" => "cot(-x) = -cot(x)",
                    "csc" => "csc(-x) = -csc(x)",
                    "arccot" => "arccot(-x) = -arccot(x)",
                    "arccsc" => "arccsc(-x) = -arccsc(x)",
                    _ => return None,
                },
            ),
            NegBehavior::Even => (
                f_inner,
                match fn_name {
                    "sec" => "sec(-x) = sec(x)",
                    _ => return None,
                },
            ),
            NegBehavior::PiMinus => {
                let pi = ctx.add(Expr::Constant(cas_ast::Constant::Pi));
                (
                    ctx.add(Expr::Sub(pi, f_inner)),
                    match fn_name {
                        "arcsec" => "arcsec(-x) = π - arcsec(x)",
                        _ => return None,
                    },
                )
            }
        };
        return Some(out);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    #[test]
    fn evaluates_known_reciprocal_trig_values() {
        let mut ctx = Context::new();
        let arg = parse("pi/4", &mut ctx).expect("arg");
        let (out, desc) = eval_reciprocal_trig_value(&mut ctx, "cot", arg).expect("eval");
        assert_eq!(desc, "cot(π/4) = 1");
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
        let (out, desc) =
            rewrite_negative_reciprocal_trig_argument(&mut ctx, "arcsec", neg_x).expect("rw");
        assert_eq!(desc, "arcsec(-x) = π - arcsec(x)");
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
}
