use cas_ast::{Context, Expr, ExprId};
use cas_math::expand_call_support::expand_explicit_arg_with_post_compaction;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetaFunctionRewrite {
    pub rewritten: ExprId,
    pub desc: &'static str,
}

/// Evaluate meta helper functions that operate on expression arguments.
///
/// Supported calls:
/// - `simplify(expr)` -> `expr`
/// - `factor(expr)` -> factored `expr` (or unchanged if irreducible)
/// - `expand(expr)` -> expanded `expr`
/// - `approx(expr)` / `evalf(expr)` -> f64 decimal presentation
///
/// `complex_enabled` opens the complex `approx()` fallback (an `a + b·i`
/// decimal result) when the real f64 path declines; `false` is the exact
/// pre-existing real surface.
pub(crate) fn try_rewrite_meta_function_expr_in_domain(
    ctx: &mut Context,
    expr: ExprId,
    complex_enabled: bool,
) -> Option<MetaFunctionRewrite> {
    let (fn_id, args) = if let Expr::Function(fn_id, args) = ctx.get(expr) {
        (*fn_id, args.clone())
    } else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }

    let arg = args[0];
    match ctx.sym_name(fn_id) {
        "simplify" => Some(MetaFunctionRewrite {
            rewritten: arg,
            desc: "simplify(x) = x (already processed)",
        }),
        "factor" => {
            let factored = cas_math::factor::factor(ctx, arg);
            let desc = if factored != arg {
                "factor(x) -> factored form"
            } else {
                "factor(x) = x (irreducible)"
            };
            Some(MetaFunctionRewrite {
                rewritten: factored,
                desc,
            })
        }
        "expand" => Some(MetaFunctionRewrite {
            rewritten: expand_explicit_arg_with_post_compaction(ctx, arg),
            desc: "expand(x) -> expanded form",
        }),
        // approx(x) / evalf(x): the explicit numeric-presentation surface.
        // Exact-only everywhere else; here the value is evaluated in f64
        // (root_sum-aware: sums over the numeric roots of the resultant) and
        // wrapped in the `decimal` display node. Free variables or
        // out-of-scope shapes leave the call unevaluated (no rewrite).
        "approx" | "evalf" => {
            // Definite-integral composition: `approx(integrate(f, x, a, b))`
            // over a RATIONAL integrand whose antiderivative comes from the
            // algorithmic backend (e.g. a root_sum). Gated on the EXACT
            // pole-in-interval check — a pole inside [a, b] means the
            // integral diverges and F(b) − F(a) would be a WRONG value, so
            // the Sturm count must be zero and both endpoints non-poles.
            if let Some(rewrite) = try_approx_definite_integral(ctx, arg) {
                return Some(rewrite);
            }
            if let Some(value) = cas_math::rootsum_numeric::numeric_eval_with_rootsum(ctx, arg) {
                let rational = cas_math::decimal_display::approx_display_rational(value)?;
                let number = ctx.add(Expr::Number(rational));
                let decimal_sym = ctx.intern_symbol("decimal");
                let node = ctx.add(Expr::Function(decimal_sym, vec![number]));
                return Some(MetaFunctionRewrite {
                    rewritten: node,
                    desc: "approx(x) -> numeric value (12 significant digits)",
                });
            }
            // Complex fallback (ComplexEnabled only): closed values the real
            // path rejects (`approx(ln(i))`, `approx(2^i)`) evaluate through
            // the refute-net walker into a cartesian `a + b·i` decimal.
            // Presentation surface only — approx is f64 BY CONTRACT; no
            // keep/drop decision rides on this value.
            if !complex_enabled {
                return None;
            }
            let z = cas_math::evaluator_complex::eval_complex(
                ctx,
                arg,
                &std::collections::HashMap::new(),
            )?;
            if !z.re.is_finite() || !z.im.is_finite() || z.im == 0.0 {
                // A real result here means the real path SHOULD have taken
                // it (or it did and we never reach this): decline rather
                // than second-guess the real surface.
                return None;
            }
            let decimal_sym = ctx.intern_symbol("decimal");
            let im_rational = cas_math::decimal_display::approx_display_rational(z.im)?;
            let im_number = ctx.add(Expr::Number(im_rational));
            let im_decimal = ctx.add(Expr::Function(decimal_sym, vec![im_number]));
            let i = ctx.add(Expr::Constant(cas_ast::Constant::I));
            let im_part = ctx.add(Expr::Mul(im_decimal, i));
            let rewritten = if z.re == 0.0 {
                im_part
            } else {
                let re_rational = cas_math::decimal_display::approx_display_rational(z.re)?;
                let re_number = ctx.add(Expr::Number(re_rational));
                let re_decimal = ctx.add(Expr::Function(decimal_sym, vec![re_number]));
                ctx.add(Expr::Add(re_decimal, im_part))
            };
            Some(MetaFunctionRewrite {
                rewritten,
                desc: "approx(z) -> complex numeric value a + b·i (12 significant digits)",
            })
        }
        _ => None,
    }
}

/// `approx(integrate(N/D, x, a, b))`: numeric value of a definite integral
/// whose indefinite antiderivative the algorithmic backend can produce (the
/// root_sum universal closure included). Declines honestly — leaving the
/// call unevaluated — unless the integrand is rational with rational bounds
/// AND the denominator provably has NO real root in `[a, b]` (exact interval
/// Sturm; floats never decide).
fn try_approx_definite_integral(ctx: &mut Context, arg: ExprId) -> Option<MetaFunctionRewrite> {
    use num_traits::Zero;
    let (fn_id, args) = match ctx.get(arg) {
        Expr::Function(fn_id, args) => (*fn_id, args.clone()),
        _ => return None,
    };
    if ctx.sym_name(fn_id) != "integrate" || args.len() != 4 {
        return None;
    }
    let variable = match ctx.get(args[1]) {
        Expr::Variable(sym) => ctx.sym_name(*sym).to_string(),
        _ => return None,
    };
    let (lower_raw, upper_raw) = (args[2], args[3]);
    let as_rational = |ctx: &Context, e: ExprId| -> Option<num_rational::BigRational> {
        match ctx.get(e) {
            Expr::Number(n) => Some(n.clone()),
            Expr::Neg(inner) => match ctx.get(*inner) {
                Expr::Number(n) => Some(-n.clone()),
                _ => None,
            },
            _ => None,
        }
    };
    let a = as_rational(ctx, lower_raw)?;
    let b = as_rational(ctx, upper_raw)?;
    if a == b {
        let zero = ctx.add(Expr::Number(num_rational::BigRational::zero()));
        let decimal_sym = ctx.intern_symbol("decimal");
        let node = ctx.add(Expr::Function(decimal_sym, vec![zero]));
        return Some(MetaFunctionRewrite {
            rewritten: node,
            desc: "approx(definite integral) -> numeric value",
        });
    }
    let (lo, hi, sign) = if a < b { (a, b, 1.0) } else { (b, a, -1.0) };

    // Rational integrand only: the exact pole gate needs D as a polynomial.
    let (num_expr, den_expr) = match ctx.get(args[0]) {
        Expr::Div(n, d) => (*n, *d),
        _ => return None,
    };
    let denominator = cas_math::polynomial::Polynomial::from_expr(ctx, den_expr, &variable).ok()?;
    if cas_math::polynomial::Polynomial::from_expr(ctx, num_expr, &variable).is_err() {
        return None;
    }
    // THE soundness gate: no pole at the endpoints or inside the interval.
    if denominator.eval(&lo).is_zero()
        || denominator.eval(&hi).is_zero()
        || denominator.count_real_roots_in_interval(&lo, &hi) != 0
    {
        return None;
    }

    // Antiderivative via the public algorithmic backend route.
    let candidate = cas_math::general_integration_backend::try_algorithmic_integration_backend(
        ctx,
        args[0],
        &variable,
        cas_math::general_integration_backend::AlgorithmicIntegrationBackendConfig::residual_fallback(),
    );
    let antiderivative = candidate.public_antiderivative()?;

    use num_traits::ToPrimitive;
    let mut var_map = std::collections::HashMap::new();
    var_map.insert(variable.clone(), hi.to_f64()?);
    let at_hi =
        cas_math::rootsum_numeric::numeric_eval_with_rootsum_at(ctx, antiderivative, &var_map)?;
    var_map.insert(variable, lo.to_f64()?);
    let at_lo =
        cas_math::rootsum_numeric::numeric_eval_with_rootsum_at(ctx, antiderivative, &var_map)?;
    let value = sign * (at_hi - at_lo);
    if !value.is_finite() {
        return None;
    }
    let rational = cas_math::decimal_display::approx_display_rational(value)?;
    let number = ctx.add(Expr::Number(rational));
    let decimal_sym = ctx.intern_symbol("decimal");
    let node = ctx.add(Expr::Function(decimal_sym, vec![number]));
    Some(MetaFunctionRewrite {
        rewritten: node,
        desc: "approx(definite integral) -> numeric value",
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    #[test]
    fn approx_payload_is_the_displayed_rounded_rational() {
        // WYSIWYG contract: decimal() wraps the ROUNDED 12-significant-digit
        // rational — exactly the value the formatter shows and exactly the
        // rational that typing the displayed string back would parse to.
        // Without this, approx carried the raw f64 binary expansion with
        // invisible phantom digits.
        let mut ctx = Context::new();
        let expr = parse("approx(3/7)", &mut ctx).expect("parse");
        let rewrite =
            try_rewrite_meta_function_expr_in_domain(&mut ctx, expr, false).expect("evaluates");
        let payload = match ctx.get(rewrite.rewritten) {
            Expr::Function(_, args) => match ctx.get(args[0]) {
                Expr::Number(n) => n.clone(),
                other => panic!("payload not a Number: {other:?}"),
            },
            other => panic!("not a decimal node: {other:?}"),
        };
        let typed_back =
            num_rational::BigRational::new(428571428571i64.into(), 1_000_000_000_000i64.into());
        assert_eq!(payload, typed_back);

        // Exactly-representable values stay exact.
        let expr = parse("approx(1/2)", &mut ctx).expect("parse");
        let rewrite =
            try_rewrite_meta_function_expr_in_domain(&mut ctx, expr, false).expect("evaluates");
        let payload = match ctx.get(rewrite.rewritten) {
            Expr::Function(_, args) => match ctx.get(args[0]) {
                Expr::Number(n) => n.clone(),
                _ => panic!("payload not a Number"),
            },
            _ => panic!("not a decimal node"),
        };
        assert_eq!(payload, num_rational::BigRational::new(1.into(), 2.into()));
    }

    #[test]
    fn approx_complex_fallback_emits_cartesian_decimal() {
        // ComplexEnabled: closed values the real f64 path rejects evaluate
        // through the complex walker into `a + b·i` decimals.
        for (src, expected) in [
            ("approx(ln(i))", "i * 1.57079632679"),
            ("approx(ln(-2))", "0.69314718056 + i * 3.14159265359"),
        ] {
            let mut ctx = Context::new();
            let expr = parse(src, &mut ctx).expect("parse");
            let rewrite = try_rewrite_meta_function_expr_in_domain(&mut ctx, expr, true)
                .unwrap_or_else(|| panic!("{src} should evaluate"));
            let rendered = format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.rewritten
                }
            );
            assert_eq!(rendered, expected, "{src}");
        }

        // Real surface (complex_enabled = false): byte-identical decline —
        // the complex fallback must never leak into the real contract.
        let mut ctx = Context::new();
        let expr = parse("approx(ln(i))", &mut ctx).expect("parse");
        assert!(try_rewrite_meta_function_expr_in_domain(&mut ctx, expr, false).is_none());
        let expr = parse("approx(ln(i))", &mut ctx).expect("parse");
        assert!(try_rewrite_meta_function_expr_in_domain(&mut ctx, expr, false).is_none());

        // Free variables decline in every domain (honest residual).
        let mut ctx = Context::new();
        let expr = parse("approx(x + i)", &mut ctx).expect("parse");
        assert!(try_rewrite_meta_function_expr_in_domain(&mut ctx, expr, true).is_none());
    }

    #[test]
    fn approx_definite_integral_evaluates_and_gates_poles() {
        let mut ctx = Context::new();
        // Happy path: no pole in [2,3]; reference = mpmath 30-digit quadrature.
        let ok = parse("approx(integrate(1/(x^3-x-1), x, 2, 3))", &mut ctx).expect("parse");
        let rewrite =
            try_rewrite_meta_function_expr_in_domain(&mut ctx, ok, false).expect("must evaluate");
        let rendered = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.rewritten
            }
        );
        assert_eq!(rendered, "0.0944265690584");

        // Pole inside (1,2) (the plastic-number root of x^3-x-1 at ~1.3247):
        // the integral diverges, so the call must stay honestly unevaluated.
        let pole = parse("approx(integrate(1/(x^3-x-1), x, 1, 2))", &mut ctx).expect("parse");
        assert!(try_rewrite_meta_function_expr_in_domain(&mut ctx, pole, false).is_none());

        // Reversed bounds negate.
        let rev = parse("approx(integrate(1/(x^3-x-1), x, 3, 2))", &mut ctx).expect("parse");
        let rewrite =
            try_rewrite_meta_function_expr_in_domain(&mut ctx, rev, false).expect("must evaluate");
        let rendered = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.rewritten
            }
        );
        assert_eq!(rendered, "-0.0944265690584");
    }

    #[test]
    fn rewrites_simplify_transparently() {
        let mut ctx = Context::new();
        let expr = parse("simplify(x+1)", &mut ctx).expect("parse");
        let rewrite =
            try_rewrite_meta_function_expr_in_domain(&mut ctx, expr, false).expect("rewrite");
        assert_eq!(rewrite.desc, "simplify(x) = x (already processed)");
        let rendered = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.rewritten
            }
        );
        assert_eq!(rendered, "x + 1");
    }

    #[test]
    fn rewrites_expand_call() {
        let mut ctx = Context::new();
        let expr = parse("expand((x+1)^2)", &mut ctx).expect("parse");
        let rewrite =
            try_rewrite_meta_function_expr_in_domain(&mut ctx, expr, false).expect("rewrite");
        assert_eq!(rewrite.desc, "expand(x) -> expanded form");
        let rendered = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.rewritten
            }
        );
        assert!(rendered.contains("x^2"));
    }

    #[test]
    fn rewrites_expand_call_with_compact_univariate_polynomial_terms() {
        let mut ctx = Context::new();
        let expr = parse("expand(3-(x^2+2*x+1)^2)", &mut ctx).expect("parse");
        let rewrite =
            try_rewrite_meta_function_expr_in_domain(&mut ctx, expr, false).expect("rewrite");
        let rendered = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.rewritten
            }
        );
        assert_eq!(rendered, "2 - x^4 - 4 * x^3 - 6 * x^2 - 4 * x");
    }

    #[test]
    fn rewrites_factor_call_with_multivar_common_monomial() {
        let mut ctx = Context::new();
        let expr = parse("factor(y^2*z^2 + 2*y^2*z + y^2)", &mut ctx).expect("parse");
        let rewrite =
            try_rewrite_meta_function_expr_in_domain(&mut ctx, expr, false).expect("rewrite");
        assert_eq!(rewrite.desc, "factor(x) -> factored form");
        let rendered = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.rewritten
            }
        );
        assert!(
            rendered.contains("y^2") && rendered.contains("(z + 1)^2"),
            "unexpected factor shape: {rendered}"
        );
    }

    #[test]
    fn rewrites_factor_call_with_multivar_common_numeric_content() {
        let mut ctx = Context::new();
        let expr = parse("factor(2*x + 4*y)", &mut ctx).expect("parse");
        let rewrite =
            try_rewrite_meta_function_expr_in_domain(&mut ctx, expr, false).expect("rewrite");
        assert_eq!(rewrite.desc, "factor(x) -> factored form");
        let rendered = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.rewritten
            }
        );
        assert!(
            rendered.contains("2 * (x + 2 * y)"),
            "unexpected factor shape: {rendered}"
        );
    }
}
