use crate::define_rule;
use crate::helpers::{build_pi_over_n, is_one, is_pi_over_n, is_zero};
use crate::rule::Rewrite;
use cas_ast::{Context, Expr, ExprId};

// ==================== Helper Functions ====================
// is_zero and is_one are now imported from crate::helpers==================

// ==================== Evaluation Table ====================

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
    ("sec", |ctx, e| is_zero(ctx, e), build_one, "sec(0) = 1"),
    (
        "csc",
        |ctx, e| is_pi_over_n(ctx, e, 2),
        build_one,
        "csc(π/2) = 1",
    ),
    (
        "arccot",
        |ctx, e| is_one(ctx, e),
        build_pi_over_4,
        "arccot(1) = π/4",
    ),
    (
        "arccot",
        |ctx, e| is_zero(ctx, e),
        build_pi_over_2,
        "arccot(0) = π/2",
    ),
    (
        "arcsec",
        |ctx, e| is_one(ctx, e),
        build_zero,
        "arcsec(1) = 0",
    ),
    (
        "arccsc",
        |ctx, e| is_one(ctx, e),
        build_pi_over_2,
        "arccsc(1) = π/2",
    ),
];

define_rule!(
    EvaluateReciprocalTrigRule,
    "Evaluate Reciprocal Trig Functions",
    Some(vec!["Function"]),
    |ctx, expr| {
        if let Expr::Function(name, args) = ctx.get(expr) {
            if args.len() == 1 {
                let arg = args[0];
                for (func, check, build, desc) in EVAL_RULES {
                    if name == *func && check(ctx, arg) {
                        return Some(Rewrite {
                            new_expr: build(ctx),
                            description: desc.to_string(),
                            before_local: None,
                            after_local: None,
                            assumption_events: Default::default(),
                            required_conditions: vec![],
                            poly_proof: None,
                        });
                    }
                }
            }
        }
        None
    }
);

// ==================== Composition Pairs ====================

const COMPOSITION_PAIRS: &[(&str, &str)] = &[
    ("cot", "arccot"),
    ("sec", "arcsec"),
    ("csc", "arccsc"),
    ("arccot", "cot"),
    ("arcsec", "sec"),
    ("arccsc", "csc"),
];

define_rule!(
    ReciprocalTrigCompositionRule,
    "Reciprocal Trig Composition",
    Some(vec!["Function"]),
    |ctx, expr| {
        if let Expr::Function(outer_name, outer_args) = ctx.get(expr) {
            if outer_args.len() == 1 {
                let inner_expr = outer_args[0];
                if let Expr::Function(inner_name, inner_args) = ctx.get(inner_expr) {
                    if inner_args.len() == 1 {
                        let x = inner_args[0];
                        for (outer, inner) in COMPOSITION_PAIRS {
                            if outer_name == *outer && inner_name == *inner {
                                return Some(Rewrite {
                                    new_expr: x,
                                    description: format!("{}({}(x)) = x", outer, inner),
                                    before_local: None,
                                    after_local: None,
                                    assumption_events: Default::default(),
                                    required_conditions: vec![],
                                    poly_proof: None,
                                });
                            }
                        }
                    }
                }
            }
        }
        None
    }
);

// ==================== Negative Argument Table ====================

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

define_rule!(
    ReciprocalTrigNegativeRule,
    "Reciprocal Trig Negative Argument",
    Some(vec!["Function"]),
    |ctx, expr| {
        if let Expr::Function(name, args) = ctx.get(expr) {
            if args.len() == 1 {
                let arg = args[0];
                if let Expr::Neg(inner) = ctx.get(arg) {
                    let inner = *inner;
                    let name_str = name.as_str();

                    for (func, behavior) in NEG_BEHAVIORS {
                        if name_str == *func {
                            let f_inner = ctx.add(Expr::Function(func.to_string(), vec![inner]));
                            let (new_expr, desc) = match behavior {
                                NegBehavior::Odd => (
                                    ctx.add(Expr::Neg(f_inner)),
                                    format!("{}(-x) = -{}(x)", func, func),
                                ),
                                NegBehavior::Even => {
                                    (f_inner, format!("{}(-x) = {}(x)", func, func))
                                }
                                NegBehavior::PiMinus => {
                                    let pi = ctx.add(Expr::Constant(cas_ast::Constant::Pi));
                                    (
                                        ctx.add(Expr::Sub(pi, f_inner)),
                                        format!("{}(-x) = π - {}(x)", func, func),
                                    )
                                }
                            };
                            return Some(Rewrite {
                                new_expr,
                                description: desc,
                                before_local: None,
                                after_local: None,
                                assumption_events: Default::default(),
                                required_conditions: vec![],
                                poly_proof: None,
                            });
                        }
                    }
                }
            }
        }
        None
    }
);

// ==================== Registration ====================

pub fn register(simplifier: &mut crate::engine::Simplifier) {
    simplifier.add_rule(Box::new(EvaluateReciprocalTrigRule));
    simplifier.add_rule(Box::new(ReciprocalTrigCompositionRule));
    simplifier.add_rule(Box::new(ReciprocalTrigNegativeRule));
}
