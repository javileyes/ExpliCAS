use crate::expr_relations::extract_negated_inner;
use crate::trig_values::{
    detect_inverse_trig_input, detect_special_angle, lookup_inverse_trig_value, lookup_trig_value,
    TrigValue,
};
use cas_ast::{BuiltinFn, Context, Expr, ExprId};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrigEvalRewrite {
    pub rewritten: ExprId,
    pub kind: TrigEvalRewriteKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrigEvalRewriteKind {
    Table(String),
    NegativeParity(TrigNegativeParityKind),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrigNegativeParityKind {
    Sin,
    Cos,
    Tan,
    Arcsin,
    Arccos,
    Arctan,
}

/// Lookup result for trig/inverse-trig exact-value tables.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrigLookupResult {
    pub key_display: &'static str,
    pub value: &'static TrigValue,
}

fn canonical_table_fn_name(fn_name: &str) -> &str {
    match fn_name {
        "asin" => "arcsin",
        "acos" => "arccos",
        "atan" => "arctan",
        _ => fn_name,
    }
}

/// Resolve an exact trig or inverse-trig value from table data.
///
/// Returns `None` if `arg` is not recognized as a special angle/input.
pub fn lookup_trig_or_inverse(
    ctx: &Context,
    fn_name: &str,
    arg: ExprId,
) -> Option<TrigLookupResult> {
    let table_fn_name = canonical_table_fn_name(fn_name);

    if let Some(angle) = detect_special_angle(ctx, arg) {
        if let Some(value) = lookup_trig_value(table_fn_name, angle) {
            return Some(TrigLookupResult {
                key_display: angle.display(),
                value,
            });
        }
    }

    if let Some(input) = detect_inverse_trig_input(ctx, arg) {
        if let Some(value) = lookup_inverse_trig_value(table_fn_name, input) {
            return Some(TrigLookupResult {
                key_display: input.display(),
                value,
            });
        }
    }

    None
}

/// Rewrite trig calls with negative arguments according to parity:
/// `sin(-x)=-sin(x)`, `cos(-x)=cos(x)`, `tan(-x)=-tan(x)`.
pub fn rewrite_negative_trig_argument(
    ctx: &mut Context,
    fn_name: &str,
    arg: ExprId,
) -> Option<(ExprId, TrigNegativeParityKind)> {
    use num_traits::Signed;
    let canonical = canonical_table_fn_name(fn_name);
    // The positive counterpart of a negative argument: a `Neg(...)` / `(-1)·x` wrapper, or — for
    // inverse trig only — a bare negative numeric literal (`arccos(-1/2)` arrives as `Number(-1/2)`,
    // which `extract_negated_inner` does not treat as a negation). Forward trig keeps its prior
    // behavior, since a bare negative literal there has no special-angle value to unlock.
    let inner = match extract_negated_inner(ctx, arg) {
        Some(i) => i,
        None => {
            let neg_literal = match ctx.get(arg) {
                Expr::Number(n) if n.is_negative() => Some(-n.clone()),
                _ => None,
            };
            match (canonical, neg_literal) {
                ("arcsin" | "arccos" | "arctan", Some(pos)) => ctx.add(Expr::Number(pos)),
                _ => return None,
            }
        }
    };
    match canonical {
        "sin" => {
            let sin_inner = ctx.call_builtin(BuiltinFn::Sin, vec![inner]);
            Some((ctx.add(Expr::Neg(sin_inner)), TrigNegativeParityKind::Sin))
        }
        "cos" => {
            let cos_inner = ctx.call_builtin(BuiltinFn::Cos, vec![inner]);
            Some((cos_inner, TrigNegativeParityKind::Cos))
        }
        "tan" => {
            let tan_inner = ctx.call_builtin(BuiltinFn::Tan, vec![inner]);
            Some((ctx.add(Expr::Neg(tan_inner)), TrigNegativeParityKind::Tan))
        }
        // arcsin(-x) = -arcsin(x), arctan(-x) = -arctan(x) (both odd).
        "arcsin" => {
            let call = ctx.call_builtin(BuiltinFn::Arcsin, vec![inner]);
            Some((ctx.add(Expr::Neg(call)), TrigNegativeParityKind::Arcsin))
        }
        "arctan" => {
            let call = ctx.call_builtin(BuiltinFn::Arctan, vec![inner]);
            Some((ctx.add(Expr::Neg(call)), TrigNegativeParityKind::Arctan))
        }
        // arccos(-x) = π − arccos(x).
        "arccos" => {
            let call = ctx.call_builtin(BuiltinFn::Arccos, vec![inner]);
            let pi = ctx.add(Expr::Constant(cas_ast::Constant::Pi));
            Some((ctx.add(Expr::Sub(pi, call)), TrigNegativeParityKind::Arccos))
        }
        _ => None,
    }
}

/// Match and rewrite trig/inverse-trig evaluation table hits and parity-on-negative forms.
pub fn try_rewrite_trig_eval_table_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<TrigEvalRewrite> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }
    let fn_name = ctx.builtin_of(*fn_id)?.name().to_string();
    let arg = args[0];

    if let Some(hit) = lookup_trig_or_inverse(ctx, &fn_name, arg) {
        return Some(TrigEvalRewrite {
            rewritten: hit.value.to_expr(ctx),
            kind: TrigEvalRewriteKind::Table(format!(
                "{}({}) = {}",
                fn_name,
                hit.key_display,
                hit.value.display()
            )),
        });
    }

    let (rewritten, kind) = rewrite_negative_trig_argument(ctx, &fn_name, arg)?;
    Some(TrigEvalRewrite {
        rewritten,
        kind: TrigEvalRewriteKind::NegativeParity(kind),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::{Constant, Expr};

    #[test]
    fn lookup_sin_pi_over_6() {
        let mut ctx = Context::new();
        let pi = ctx.add(Expr::Constant(Constant::Pi));
        let six = ctx.num(6);
        let pi_over_six = ctx.add(Expr::Div(pi, six));

        let out = lookup_trig_or_inverse(&ctx, "sin", pi_over_six).expect("expected table hit");
        assert_eq!(out.key_display, "π/6");
        assert_eq!(out.value, &TrigValue::Fraction(1, 2));
    }

    #[test]
    fn lookup_arctan_one() {
        let mut ctx = Context::new();
        let one = ctx.num(1);

        let out = lookup_trig_or_inverse(&ctx, "arctan", one).expect("expected inverse table hit");
        assert_eq!(out.key_display, "1");
        assert_eq!(out.value, &TrigValue::PiDiv(4));
    }

    #[test]
    fn lookup_arctan_direct_sqrt_three() {
        let mut ctx = Context::new();
        let sqrt_three = cas_parser::parse("sqrt(3)", &mut ctx).expect("parse sqrt(3)");

        let out =
            lookup_trig_or_inverse(&ctx, "arctan", sqrt_three).expect("expected inverse table hit");
        assert_eq!(out.key_display, "√3");
        assert_eq!(out.value, &TrigValue::PiDiv(3));
    }

    #[test]
    fn lookup_inverse_trig_sqrt_rational_forms() {
        let cases = [
            ("arcsin", "sqrt(1/2)", "√2/2", TrigValue::PiDiv(4)),
            ("arccos", "sqrt(1/2)", "√2/2", TrigValue::PiDiv(4)),
            ("arcsin", "sqrt(3/4)", "√3/2", TrigValue::PiDiv(3)),
            ("arccos", "sqrt(3/4)", "√3/2", TrigValue::PiDiv(6)),
            ("arctan", "sqrt(1/3)", "√3/3", TrigValue::PiDiv(6)),
            ("arcsin", "2^(-1/2)", "√2/2", TrigValue::PiDiv(4)),
            ("arccos", "2^(-1/2)", "√2/2", TrigValue::PiDiv(4)),
            ("arcsin", "3/2 * 3^(-1/2)", "√3/2", TrigValue::PiDiv(3)),
            ("arccos", "3/2 * 3^(-1/2)", "√3/2", TrigValue::PiDiv(6)),
            ("arctan", "3^(-1/2)", "√3/3", TrigValue::PiDiv(6)),
        ];

        for (function, input, key_display, expected) in cases {
            let mut ctx = Context::new();
            let expr = cas_parser::parse(input, &mut ctx).expect("parse inverse trig input");
            let out =
                lookup_trig_or_inverse(&ctx, function, expr).expect("expected inverse table hit");
            assert_eq!(out.key_display, key_display, "input: {input}");
            assert_eq!(out.value, &expected, "input: {input}");
        }
    }

    #[test]
    fn lookup_inverse_trig_aliases() {
        let mut ctx = Context::new();
        let zero = ctx.num(0);
        let one = ctx.num(1);

        let asin = lookup_trig_or_inverse(&ctx, "asin", zero).expect("asin alias");
        assert_eq!(asin.value, &TrigValue::Zero);

        let acos = lookup_trig_or_inverse(&ctx, "acos", one).expect("acos alias");
        assert_eq!(acos.value, &TrigValue::Zero);

        let atan = lookup_trig_or_inverse(&ctx, "atan", one).expect("atan alias");
        assert_eq!(atan.value, &TrigValue::PiDiv(4));
    }

    #[test]
    fn lookup_reciprocal_trig_values() {
        let mut ctx = Context::new();
        let expr = cas_parser::parse("pi/4", &mut ctx).expect("parse");

        let sec = lookup_trig_or_inverse(&ctx, "sec", expr).expect("sec table hit");
        assert_eq!(sec.value, &TrigValue::Sqrt(2));

        let csc = lookup_trig_or_inverse(&ctx, "csc", expr).expect("csc table hit");
        assert_eq!(csc.value, &TrigValue::Sqrt(2));

        let cot = lookup_trig_or_inverse(&ctx, "cot", expr).expect("cot table hit");
        assert_eq!(cot.value, &TrigValue::One);
    }

    #[test]
    fn lookup_miss_for_non_special_input() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        assert!(lookup_trig_or_inverse(&ctx, "sin", x).is_none());
    }

    #[test]
    fn rewrite_negative_sin() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let neg_x = ctx.add(Expr::Neg(x));
        let (new_expr, kind) =
            rewrite_negative_trig_argument(&mut ctx, "sin", neg_x).expect("expected rewrite");
        assert_eq!(kind, TrigNegativeParityKind::Sin);
        match ctx.get(new_expr) {
            Expr::Neg(inner) => match ctx.get(*inner) {
                Expr::Function(fn_id, args) => {
                    assert_eq!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Sin));
                    assert_eq!(args.as_slice(), &[x]);
                }
                _ => panic!("expected sin(x)"),
            },
            _ => panic!("expected negated sin"),
        }
    }

    #[test]
    fn rewrite_negative_cos() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let neg_x = ctx.add(Expr::Neg(x));
        let (new_expr, kind) =
            rewrite_negative_trig_argument(&mut ctx, "cos", neg_x).expect("expected rewrite");
        assert_eq!(kind, TrigNegativeParityKind::Cos);
        match ctx.get(new_expr) {
            Expr::Function(fn_id, args) => {
                assert_eq!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Cos));
                assert_eq!(args.as_slice(), &[x]);
            }
            _ => panic!("expected cos(x)"),
        }
    }

    #[test]
    fn rewrite_negative_inverse_trig_via_symmetry_end_to_end() {
        // arccos(-1/2) = 2π/3, arcsin(-1/2) = -π/6, arctan(-1) = -π/4 — driven by the inverse-trig
        // symmetries on a bare negative numeric literal (not a `Neg(...)` wrapper).
        // The rule sees the SIMPLIFIED argument, a single `Number(-1/2)` (the simplifier folds the
        // literal before the trig rule runs), not the raw `Div(-1, 2)` parse tree.
        let neg_half_val = num_rational::BigRational::new((-1).into(), 2.into());
        for (src, want_kind) in [
            ("arccos", TrigNegativeParityKind::Arccos),
            ("arcsin", TrigNegativeParityKind::Arcsin),
            ("arctan", TrigNegativeParityKind::Arctan),
        ] {
            let mut ctx = Context::new();
            let neg_half = ctx.add(Expr::Number(neg_half_val.clone()));
            let (_rewritten, kind) = rewrite_negative_trig_argument(&mut ctx, src, neg_half)
                .expect("inverse-trig negative literal should rewrite");
            assert_eq!(kind, want_kind);
        }
        // The `a*` spelling canonicalizes to the same rewrite.
        let mut ctx = Context::new();
        let neg_one = ctx.add(Expr::Number(num_rational::BigRational::from_integer(
            (-1).into(),
        )));
        let (_e, kind) =
            rewrite_negative_trig_argument(&mut ctx, "asin", neg_one).expect("asin alias rewrite");
        assert_eq!(kind, TrigNegativeParityKind::Arcsin);
    }

    #[test]
    fn rewrite_expr_table_hit() {
        let mut ctx = Context::new();
        let expr = cas_parser::parse("sin(pi/6)", &mut ctx).expect("parse");
        let rewrite = try_rewrite_trig_eval_table_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(
            rewrite.kind,
            TrigEvalRewriteKind::Table("sin(π/6) = 1/2".to_string())
        );
    }

    #[test]
    fn rewrite_expr_negative_parity() {
        let mut ctx = Context::new();
        let expr = cas_parser::parse("cos(-x)", &mut ctx).expect("parse");
        let rewrite = try_rewrite_trig_eval_table_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(
            rewrite.kind,
            TrigEvalRewriteKind::NegativeParity(TrigNegativeParityKind::Cos)
        );
        match ctx.get(rewrite.rewritten) {
            Expr::Function(fn_id, _) => assert_eq!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Cos)),
            _ => panic!("expected cosine call"),
        }
    }
}
