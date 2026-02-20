use crate::expr_relations::extract_negated_inner;
use crate::trig_values::{
    detect_inverse_trig_input, detect_special_angle, lookup_inverse_trig_value, lookup_trig_value,
    TrigValue,
};
use cas_ast::{BuiltinFn, Context, Expr, ExprId};

/// Lookup result for trig/inverse-trig exact-value tables.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrigLookupResult {
    pub key_display: &'static str,
    pub value: &'static TrigValue,
}

/// Resolve an exact trig or inverse-trig value from table data.
///
/// Returns `None` if `arg` is not recognized as a special angle/input.
pub fn lookup_trig_or_inverse(
    ctx: &Context,
    fn_name: &str,
    arg: ExprId,
) -> Option<TrigLookupResult> {
    if let Some(angle) = detect_special_angle(ctx, arg) {
        if let Some(value) = lookup_trig_value(fn_name, angle) {
            return Some(TrigLookupResult {
                key_display: angle.display(),
                value,
            });
        }
    }

    if let Some(input) = detect_inverse_trig_input(ctx, arg) {
        if let Some(value) = lookup_inverse_trig_value(fn_name, input) {
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
) -> Option<(ExprId, &'static str)> {
    let inner = extract_negated_inner(ctx, arg)?;
    match fn_name {
        "sin" => {
            let sin_inner = ctx.call_builtin(BuiltinFn::Sin, vec![inner]);
            Some((ctx.add(Expr::Neg(sin_inner)), "sin(-x) = -sin(x)"))
        }
        "cos" => {
            let cos_inner = ctx.call_builtin(BuiltinFn::Cos, vec![inner]);
            Some((cos_inner, "cos(-x) = cos(x)"))
        }
        "tan" => {
            let tan_inner = ctx.call_builtin(BuiltinFn::Tan, vec![inner]);
            Some((ctx.add(Expr::Neg(tan_inner)), "tan(-x) = -tan(x)"))
        }
        _ => None,
    }
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
        let (new_expr, desc) =
            rewrite_negative_trig_argument(&mut ctx, "sin", neg_x).expect("expected rewrite");
        assert_eq!(desc, "sin(-x) = -sin(x)");
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
        let (new_expr, desc) =
            rewrite_negative_trig_argument(&mut ctx, "cos", neg_x).expect("expected rewrite");
        assert_eq!(desc, "cos(-x) = cos(x)");
        match ctx.get(new_expr) {
            Expr::Function(fn_id, args) => {
                assert_eq!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Cos));
                assert_eq!(args.as_slice(), &[x]);
            }
            _ => panic!("expected cos(x)"),
        }
    }
}
