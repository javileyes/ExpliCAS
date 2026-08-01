//! `focused_rule_substeps`: familia `calculus`.
//!
//! Ver la cabecera de `focused_rule_substeps.rs` para el contexto.

use super::*;

/// `∂f/∂x_i`, simplified, on a scratch context — the state row `i` of the
/// Hessian actually starts from. Returns `None` (and the caller falls back to
/// `f`) when the derivative cannot be rebuilt, rather than inventing one.
pub(super) fn hessian_row_first_derivative(
    ctx: &Context,
    target: ExprId,
    var_name: &str,
) -> Option<(Context, ExprId)> {
    let mut scratch = ctx.clone();
    let derivative = cas_math::symbolic_differentiation_support::differentiate_symbolic_expr(
        &mut scratch,
        target,
        var_name,
    )?;
    // The raw derivative carries the machinery's exponent arithmetic
    // (`x^(2 - 1)`); simplifying is what gives the state the student would
    // write. Same treatment as the cells, so the two sides of the substep are
    // in the same form.
    let folded = simplify_expr_in_context(&mut scratch, derivative);
    Some((scratch, folded))
}

/// Formula-level narration for the line-integral verb (F4, Fase 3 — same
/// formula-level precedent as divergence/laplacian/taylor).
pub(super) fn generate_lineintegral_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    if !matches!(
        step.rule_name.as_str(),
        "Line Integral" | "Calcular la integral de línea"
    ) {
        return Vec::new();
    }
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Expr::Function(fn_id, args) = ctx.get(before) else {
        return Vec::new();
    };
    let fn_name = ctx.sym_name(*fn_id);
    if fn_name != "lineintegral" || args.len() != 6 {
        return Vec::new();
    }
    let field = args[0];
    let key = if matches!(ctx.get(field), Expr::Matrix { .. }) {
        "lineintegral.formula_vector"
    } else {
        "lineintegral.formula_scalar"
    };
    let lower = display_expr(ctx, args[4]);
    let upper = display_expr(ctx, args[5]);
    vec![SubStep::keyed(
        key,
        vec![lower, upper],
        display_expr(ctx, field),
        display_expr(ctx, after),
    )]
}

/// Formula-level narration for the surface-integral verb (F5, Fase 3).
pub(super) fn generate_surface_integral_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    if !matches!(
        step.rule_name.as_str(),
        "Surface Integral" | "Calcular la integral de superficie"
    ) {
        return Vec::new();
    }
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Expr::Function(fn_id, args) = ctx.get(before) else {
        return Vec::new();
    };
    let fn_name = ctx.sym_name(*fn_id);
    if fn_name != "surface_integral" || args.len() != 6 {
        return Vec::new();
    }
    let field = args[0];
    let key = if matches!(ctx.get(field), Expr::Matrix { .. }) {
        "surfaceintegral.formula_vector"
    } else {
        "surfaceintegral.formula_scalar"
    };
    vec![SubStep::keyed(
        key,
        vec![],
        display_expr(ctx, field),
        display_expr(ctx, after),
    )]
}

/// Formula-level narration for the MULTIVARIATE Taylor verb (F2, Fase 3 —
/// same precedent as the divergence/laplacian formula substeps: per-term
/// narration would need the assembler's internals threaded through the wire,
/// which `Rewrite::substep()` metadata does not reach).
pub(super) fn generate_taylor_multivar_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    if !matches!(
        step.rule_name.as_str(),
        "Taylor Series" | "Desarrollar en serie de Taylor"
    ) {
        return Vec::new();
    }
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Expr::Function(fn_id, args) = ctx.get(before) else {
        return Vec::new();
    };
    let fn_name = ctx.sym_name(*fn_id);
    if (fn_name != "taylor" && fn_name != "series") || !(2..=4).contains(&args.len()) {
        return Vec::new();
    }
    let target = args[0];
    // Multivariate form only: the 2nd argument is the variable LIST.
    let Expr::Matrix { data: vars, .. } = ctx.get(args[1]) else {
        return Vec::new();
    };
    let var_count = vars.len();
    let order = match args.len() {
        4 => integer_literal(ctx, args[3]),
        3 => integer_literal(ctx, args[2]),
        _ => Some(2),
    };
    let Some(order) = order else {
        return Vec::new();
    };
    vec![SubStep::keyed(
        "taylor.formula",
        vec![order.to_string(), var_count.to_string()],
        display_expr(ctx, target),
        display_expr(ctx, after),
    )]
}

pub(super) fn differentiation_chain_inner_derivative(
    ctx: &Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, String, String)> {
    let chain_target = match ctx.get(target) {
        Expr::Neg(inner) => *inner,
        _ => target,
    };
    let chain_target = differentiation_constant_multiple_inner(ctx, chain_target, var_name)
        .unwrap_or(chain_target);
    let inner = match ctx.get(chain_target) {
        Expr::Pow(base, exponent)
            if contains_named_var(ctx, *base, var_name)
                && !contains_named_var(ctx, *exponent, var_name)
                && !is_named_var(ctx, *base, var_name) =>
        {
            *base
        }
        Expr::Pow(base, exponent)
            if !contains_named_var(ctx, *base, var_name)
                && contains_named_var(ctx, *exponent, var_name)
                && !is_named_var(ctx, *exponent, var_name) =>
        {
            *exponent
        }
        Expr::Function(_, args)
            if args.len() == 1
                && contains_named_var(ctx, args[0], var_name)
                && !is_named_var(ctx, args[0], var_name) =>
        {
            args[0]
        }
        _ => return None,
    };

    let mut scratch = ctx.clone();
    let derivative = cas_math::symbolic_differentiation_support::differentiate_symbolic_expr(
        &mut scratch,
        inner,
        var_name,
    )?;
    let derivative = if contains_trig_function_for_didactic_derivative(ctx, inner) {
        derivative
    } else {
        simplify_expr_in_context(&mut scratch, derivative)
    };
    Some((
        inner,
        display_expr(&scratch, derivative),
        latex_expr(&scratch, derivative),
    ))
}

fn contains_trig_function_for_didactic_derivative(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Function(fn_id, args) => {
            matches!(
                ctx.builtin_of(*fn_id),
                Some(
                    BuiltinFn::Sin
                        | BuiltinFn::Cos
                        | BuiltinFn::Tan
                        | BuiltinFn::Cot
                        | BuiltinFn::Sec
                        | BuiltinFn::Csc
                        | BuiltinFn::Sinh
                        | BuiltinFn::Cosh
                        | BuiltinFn::Tanh
                )
            ) || args
                .iter()
                .copied()
                .any(|arg| contains_trig_function_for_didactic_derivative(ctx, arg))
        }
        Expr::Add(left, right)
        | Expr::Sub(left, right)
        | Expr::Mul(left, right)
        | Expr::Div(left, right)
        | Expr::Pow(left, right) => {
            contains_trig_function_for_didactic_derivative(ctx, *left)
                || contains_trig_function_for_didactic_derivative(ctx, *right)
        }
        Expr::Neg(inner) | Expr::Hold(inner) => {
            contains_trig_function_for_didactic_derivative(ctx, *inner)
        }
        _ => false,
    }
}

pub(super) fn differentiation_component_derivative_substeps(
    ctx: &Context,
    target: ExprId,
    var_name: &str,
) -> Vec<SubStep> {
    if differentiation_constant_multiple_inner(ctx, target, var_name).is_some() {
        return Vec::new();
    }

    let components: Vec<(&'static str, ExprId)> = match ctx.get(target) {
        Expr::Mul(left, right) => vec![
            ("derivative.differentiate_first_factor", *left),
            ("derivative.differentiate_second_factor", *right),
        ],
        Expr::Div(numerator, denominator) => vec![
            ("derivative.differentiate_numerator", *numerator),
            ("derivative.differentiate_denominator", *denominator),
        ],
        _ => return Vec::new(),
    };

    components
        .into_iter()
        .filter_map(|(title, component)| {
            differentiation_component_derivative_substep(ctx, component, var_name, title)
        })
        .collect()
}

fn differentiation_component_derivative_substep(
    ctx: &Context,
    component: ExprId,
    var_name: &str,
    title: &'static str,
) -> Option<SubStep> {
    if !contains_named_var(ctx, component, var_name) {
        return None;
    }

    let mut scratch = ctx.clone();
    let derivative = cas_math::symbolic_differentiation_support::differentiate_symbolic_expr(
        &mut scratch,
        component,
        var_name,
    )?;
    let derivative = simplify_expr_in_context(&mut scratch, derivative);
    Some(
        SubStep::keyed(
            title,
            vec![],
            display_expr(ctx, component),
            display_expr(&scratch, derivative),
        )
        .with_before_latex(latex_expr(ctx, component))
        .with_after_latex(latex_expr(&scratch, derivative)),
    )
}

pub(super) fn integral_sum_display(
    ctx: &Context,
    terms: &[(ExprId, Sign)],
    var_name: &str,
) -> String {
    join_signed_terms(terms.iter().map(|(term, sign)| {
        (
            format!("integrate({}, {})", display_expr(ctx, *term), var_name),
            *sign,
        )
    }))
}

pub(super) fn integral_sum_latex(
    ctx: &Context,
    terms: &[(ExprId, Sign)],
    var_name: &str,
) -> String {
    join_signed_terms(terms.iter().map(|(term, sign)| {
        (
            format!("\\int {}\\,d{}", latex_expr(ctx, *term), var_name),
            *sign,
        )
    }))
}

pub(super) fn polynomial_antiderivative_display(
    ctx: &Context,
    polynomial: ExprId,
    var_name: &str,
) -> Option<(String, String)> {
    let poly = Polynomial::from_expr(ctx, polynomial, var_name).ok()?;
    let mut v_coeffs = vec![BigRational::zero(); poly.coeffs.len() + 1];
    let mut has_nonzero_term = false;
    for (degree, coefficient) in poly.coeffs.iter().enumerate() {
        if coefficient.is_zero() {
            continue;
        }
        has_nonzero_term = true;
        let denominator = BigRational::from_integer(((degree + 1) as i64).into());
        v_coeffs[degree + 1] = coefficient.clone() / denominator;
    }
    if !has_nonzero_term {
        return None;
    }
    let v_poly = Polynomial::new(v_coeffs, var_name.to_string());
    let mut scratch = ctx.clone();
    let v_expr = v_poly.to_expr(&mut scratch);
    Some((display_expr(&scratch, v_expr), latex_expr(&scratch, v_expr)))
}

pub(super) fn log_argument_derivative_fraction_display(
    ctx: &Context,
    log_arg: ExprId,
    var_name: &str,
) -> Option<(String, String, String, String)> {
    let mut scratch = ctx.clone();
    let derivative = cas_math::symbolic_differentiation_support::differentiate_symbolic_expr(
        &mut scratch,
        log_arg,
        var_name,
    )?;
    let derivative = simplify_expr_in_context(&mut scratch, derivative);
    let derivative_display = display_expr(&scratch, derivative);
    let derivative_latex = latex_expr(&scratch, derivative);
    let arg_display = display_expr(ctx, log_arg);
    let arg_latex = latex_expr(ctx, log_arg);
    let factor_display = if derivative_display == "1" {
        format!("1/{}", group_display_for_quotient_denominator(&arg_display))
    } else {
        format!(
            "{}/{}",
            group_display_for_quotient_numerator(&derivative_display),
            group_display_for_quotient_denominator(&arg_display)
        )
    };
    let factor_latex = format!("\\frac{{{}}}{{{}}}", derivative_latex, arg_latex);
    Some((
        format!("{} dx", factor_display),
        factor_display,
        format!("{}\\,dx", factor_latex),
        factor_latex,
    ))
}

/// Narrate the fundamental-theorem story for definite integrals
/// (block 13): find the antiderivative (rebuilt on a scratch context via
/// the educational route, falling back to the verified algorithmic
/// backend), evaluate it at the bounds, or - for undefined results -
/// explain the pole inside the integration interval.
/// Narrate `int_a^b |c x + d| dx` (split at the root). This route has no single
/// elementary antiderivative, so the FTC narration below would produce nothing;
/// instead we explain the structural shortcut, mirroring
/// `abs_linear_definite_integral_rewrite` in cas_engine: G(x) = c x^2/2 + d x is
/// the per-piece antiderivative, the inner has constant sign on each side of the
/// root r = -d/c, and the value sums the absolute area of each piece.
fn generate_abs_linear_definite_integral_substeps(
    ctx: &Context,
    integrand: ExprId,
    var_name: &str,
    lower: ExprId,
    upper: ExprId,
    result: ExprId,
) -> Option<Vec<SubStep>> {
    // Integrand must be |g(x)| with g a nonzero-slope linear polynomial.
    let Expr::Function(fn_id, args) = ctx.get(integrand) else {
        return None;
    };
    if args.len() != 1 || ctx.builtin_of(*fn_id) != Some(BuiltinFn::Abs) {
        return None;
    }
    let inner = args[0];
    let poly = Polynomial::from_expr(ctx, inner, var_name).ok()?;
    if poly.degree() != 1 {
        return None;
    }
    let slope = poly.coeffs.get(1)?.clone();
    let intercept = poly.coeffs.first()?.clone();
    if slope.is_zero() {
        return None;
    }

    // Pure rational bounds only (matches the rewrite's scope; a pi/e component
    // is out of range and as_rational_const declines it).
    let lo = as_rational_const(ctx, lower, 8)?;
    let hi = as_rational_const(ctx, upper, 8)?;

    let root = -&intercept / &slope;
    let (left, right) = if lo <= hi {
        (lo.clone(), hi.clone())
    } else {
        (hi.clone(), lo.clone())
    };
    let root_inside = left < root && right > root;

    let rat = |r: &BigRational| -> String {
        if r.denom().is_one() {
            r.numer().to_string()
        } else {
            format!("{}/{}", r.numer(), r.denom())
        }
    };
    let inner_str = display_expr(ctx, inner);
    let result_str = display_expr(ctx, result);

    let mut substeps = vec![SubStep::new(
        "Localizar la raíz del valor absoluto",
        format!("|{inner_str}|"),
        format!("{var_name} = {}", rat(&root)),
    )];
    if root_inside {
        substeps.push(SubStep::new(
            "Partir el intervalo en la raíz: el interior tiene signo constante en cada tramo",
            format!("[{}, {}]", rat(&left), rat(&right)),
            format!(
                "[{}, {}] ∪ [{}, {}]",
                rat(&left),
                rat(&root),
                rat(&root),
                rat(&right)
            ),
        ));
    } else {
        // Constant sign on the whole interval: read it off the midpoint.
        let two = BigRational::from_integer(2.into());
        let mid = (&left + &right) / &two;
        let inner_at_mid = &slope * &mid + &intercept;
        let signed = if inner_at_mid.is_negative() {
            format!("-({inner_str})")
        } else {
            format!("({inner_str})")
        };
        substeps.push(SubStep::new(
            "El interior mantiene signo constante en el intervalo",
            format!("|{inner_str}|"),
            signed,
        ));
    }
    // The closing line used to publish `∫ |2·x - 1| dx ⇒ 5/2`: an INDEFINITE
    // integral equated to a number. It was false as written and unfixable as
    // written, because the sub-step never carried the bounds it was evaluating
    // at. Now it shows the per-piece work the title describes, which is true
    // AND checkable: the left side is rebuilt here from the coefficients, the
    // right side is the engine's value, and `Equality` decides between them by
    // pure rational arithmetic.
    //
    // The two branches show different things because the informative thing IS
    // different. With the root inside there are two areas and the SUM is the
    // news, so the pieces go in as their values: `1/4 + 9/4`. Writing them as
    // `|G(b) − G(a)|` differences instead would cost more than it buys — a
    // subtrahend that is negative renders `2 - -1/4` in plain text and
    // `2 + \frac{1}{4}` in LaTeX (one sub-step, two different-looking claims),
    // and the commutative reordering both renderers apply to `Add` puts the
    // pieces out of narrative order.
    //
    // With a single piece there is no sum to show, and the area value alone
    // would just restate the result the parent step already carries. There the
    // EVALUATION is the news, so the difference goes in explicitly — with the
    // negative subtrahend folded into an addition, which is what keeps the two
    // renderers saying the same thing.
    //
    // The orientation is part of the claim. `∫_2^0 |2x−1| dx` is −5/2 while the
    // sum of areas is +5/2, so a reversed interval carries the sign; without it
    // the sub-step would assert `5/2 = −5/2` and be refuted, which is exactly
    // what the check is there for.
    let mut work = ctx.clone();
    let g_at = |t: &BigRational| -> BigRational {
        &slope * t * t / BigRational::from_integer(2.into()) + &intercept * t
    };
    let mut area_sum = if root_inside {
        let first = (g_at(&root) - g_at(&left)).abs();
        let second = (g_at(&right) - g_at(&root)).abs();
        let first = work.add(Expr::Number(first));
        let second = work.add(Expr::Number(second));
        work.add_raw(Expr::Add(first, second))
    } else {
        let from = g_at(&left);
        let to = work.add(Expr::Number(g_at(&right)));
        let difference = if from.is_negative() {
            let addend = work.add(Expr::Number(-from));
            work.add_raw(Expr::Add(to, addend))
        } else {
            let from = work.add(Expr::Number(from));
            work.add_raw(Expr::Sub(to, from))
        };
        work.call_builtin(BuiltinFn::Abs, vec![difference])
    };
    if lo > hi {
        area_sum = work.add_raw(Expr::Neg(area_sum));
    }
    // Backstop for the degenerate case both branches can still reach (an empty
    // or single-point interval): a line whose sides read the same narrates
    // nothing, and the parent step already carries that value.
    let sum_display = display_expr(&work, area_sum);
    if sum_display == result_str {
        return Some(substeps);
    }
    if let Some(substep) = SubStep::checked_new(
        &work,
        crate::didactic::substep::Claim::Equality,
        area_sum,
        result,
        "Integrar por tramos con G(x) = c·x²/2 + d·x y sumar las áreas",
        sum_display,
        result_str,
    ) {
        substeps.push(
            substep
                .with_before_latex(latex_expr(&work, area_sum))
                .with_after_latex(latex_expr(ctx, result)),
        );
    }
    Some(substeps)
}

pub(super) fn generate_definite_integral_substeps(
    ctx: &Context,
    step: &Step,
    depth: usize,
) -> Vec<SubStep> {
    if step.rule_name != "Symbolic Integration" {
        return Vec::new();
    }

    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Expr::Function(fn_id, args) = ctx.get(before) else {
        return Vec::new();
    };
    if ctx.sym_name(*fn_id) != "integrate" || args.len() != 4 {
        return Vec::new();
    }
    let Expr::Variable(var_sym) = ctx.get(args[1]) else {
        return Vec::new();
    };
    let var_name = ctx.sym_name(*var_sym).to_string();
    let integrand = args[0];
    let var_expr = args[1];
    let lower = args[2];
    let upper = args[3];

    let mut result = after;
    loop {
        let unwrapped = cas_ast::hold::unwrap_internal_hold(ctx, result);
        if unwrapped == result {
            break;
        }
        result = unwrapped;
    }
    if expr_contains_integrate_call(ctx, result) {
        return Vec::new();
    }

    if matches!(
        ctx.get(result),
        Expr::Constant(cas_ast::Constant::Undefined)
    ) {
        return vec![SubStep::new(
            "Detectar un polo dentro del intervalo de integración",
            display_expr(ctx, integrand),
            display_expr(ctx, result),
        )
        .with_before_latex(latex_expr(ctx, integrand))
        .with_after_latex(latex_expr(ctx, result))];
    }

    // |linear| has no single elementary antiderivative, so the FTC narration
    // below would produce nothing; narrate the root-split route instead.
    if let Some(substeps) = generate_abs_linear_definite_integral_substeps(
        ctx, integrand, &var_name, lower, upper, result,
    ) {
        return substeps;
    }

    let mut scratch = ctx.clone();
    let antiderivative = cas_math::symbolic_integration_support::integrate_symbolic_expr(
        &mut scratch,
        integrand,
        &var_name,
    )
    .or_else(|| {
        let candidate = cas_math::general_integration_backend::try_algorithmic_integration_backend(
            &mut scratch,
            integrand,
            &var_name,
            cas_math::general_integration_backend::AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );
        match candidate.verification_status {
            cas_math::general_integration_backend::AlgorithmicIntegrationVerificationStatus::Verified
            | cas_math::general_integration_backend::AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions => {
                candidate.antiderivative
            }
            _ => None,
        }
    });
    let Some(antiderivative) = antiderivative else {
        return Vec::new();
    };

    // At infinite bounds the boundary value is a LIMIT, not a
    // substitution: narrate lim_{x -> +-inf} F instead of quoting an
    // infinity constant inside F.
    let bound_is_infinite = |ctx: &Context, bound: ExprId| -> Option<&'static str> {
        match ctx.get(bound) {
            Expr::Constant(cas_ast::Constant::Infinity) => Some("∞"),
            Expr::Neg(inner)
                if matches!(ctx.get(*inner), Expr::Constant(cas_ast::Constant::Infinity)) =>
            {
                Some("-∞")
            }
            _ => None,
        }
    };
    // A substituted boundary form containing ln(0), x/0 or 0^negative is
    // a boundary-touched endpoint: narrate the one-sided limit instead of
    // quoting the undefined form.
    fn substituted_form_is_undefined(ctx: &Context, expr: ExprId) -> bool {
        let is_zero = |ctx: &Context, candidate: ExprId| -> bool {
            matches!(ctx.get(candidate), Expr::Number(value) if value == &num_rational::BigRational::from_integer(0.into()))
        };
        match ctx.get(expr) {
            Expr::Function(fn_id, args)
                if args.len() == 1
                    && matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Ln))
                    && is_zero(ctx, args[0]) =>
            {
                true
            }
            Expr::Function(_, args) => args
                .iter()
                .any(|arg| substituted_form_is_undefined(ctx, *arg)),
            Expr::Div(l, r) => {
                is_zero(ctx, *r)
                    || substituted_form_is_undefined(ctx, *l)
                    || substituted_form_is_undefined(ctx, *r)
            }
            Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Pow(l, r) => {
                substituted_form_is_undefined(ctx, *l) || substituted_form_is_undefined(ctx, *r)
            }
            Expr::Neg(inner) | Expr::Hold(inner) => substituted_form_is_undefined(ctx, *inner),
            _ => false,
        }
    }

    // A sum or difference must be delimited wherever it lands under an operator
    // that binds tighter than its own terms: as the operand of a `lim`, and as
    // the subtrahend of `F(b) - F(a)`. Without this the minus (or the limit)
    // only reaches the first term and the substep publishes a false identity.
    fn is_additive_chain(ctx: &Context, expr: ExprId) -> bool {
        // The antiderivative can arrive wrapped in `Hold` (the integration
        // pipeline wraps backend results); `Hold` is transparent to the
        // renderers, so it must be transparent to this check too.
        let mut current = expr;
        loop {
            let unwrapped = cas_ast::hold::unwrap_internal_hold(ctx, current);
            if unwrapped == current {
                break;
            }
            current = unwrapped;
        }
        matches!(
            ctx.get(current),
            Expr::Add(_, _) | Expr::Sub(_, _) | Expr::Neg(_)
        )
    }
    let delimit_display = |ctx: &Context, expr: ExprId| -> String {
        if is_additive_chain(ctx, expr) {
            format!("({})", display_expr(ctx, expr))
        } else {
            display_expr(ctx, expr)
        }
    };
    let delimit_latex = |ctx: &Context, expr: ExprId| -> String {
        if is_additive_chain(ctx, expr) {
            format!("\\left({}\\right)", latex_expr(ctx, expr))
        } else {
            latex_expr(ctx, expr)
        }
    };

    // The third element is the substituted endpoint value when the bound is
    // finite and defined; `None` means the string is limit notation, which has
    // no expression node to subtract from.
    let boundary_strings =
        |scratch: &mut Context, bound: ExprId| -> (String, String, Option<ExprId>) {
            if let Some(sign) = bound_is_infinite(scratch, bound) {
                let latex_sign = if sign == "∞" {
                    "\\infty".to_string()
                } else {
                    "-\\infty".to_string()
                };
                return (
                    format!(
                        "lim_{{{} → {}}} {}",
                        var_name,
                        sign,
                        delimit_display(scratch, antiderivative)
                    ),
                    format!(
                        "\\lim_{{{} \\to {}}} {}",
                        var_name,
                        latex_sign,
                        delimit_latex(scratch, antiderivative)
                    ),
                    None,
                );
            }
            {
                let substituted =
                    cas_ast::substitute_expr_by_id(scratch, antiderivative, var_expr, bound);
                if substituted_form_is_undefined(scratch, substituted) {
                    // Touched endpoint: one-sided limit notation. The side is
                    // a presentation choice; the lower bound approaches from
                    // the right and the upper from the left in the common
                    // oriented case.
                    let arrow = format!("{} → {}", var_name, display_expr(scratch, bound));
                    return (
                        format!(
                            "lim_{{{}}} {}",
                            arrow,
                            delimit_display(scratch, antiderivative)
                        ),
                        format!(
                            "\\lim_{{{} \\to {}}} {}",
                            var_name,
                            latex_expr(scratch, bound),
                            delimit_latex(scratch, antiderivative)
                        ),
                        None,
                    );
                }
                (
                    display_expr(scratch, substituted),
                    latex_expr(scratch, substituted),
                    Some(substituted),
                )
            }
        };
    let (upper_display, upper_latex, upper_value) = boundary_strings(&mut scratch, upper);
    let (lower_display, lower_latex, lower_value) = boundary_strings(&mut scratch, lower);
    // When both endpoints are plain values, F(b) - F(a) is an EXPRESSION: build
    // the `Sub` node and let the renderers place the parentheses they already
    // know how to place (`cas_formatter::latex::test_latex_sub_with_add_rhs`).
    // Concatenating the two strings is what made the minus reach only the first
    // term of F(a). `add_raw` keeps the didactic shape: `add` would canonicalize
    // the difference away.
    let (difference_display, difference_latex, difference_node) = match (upper_value, lower_value) {
        (Some(upper_id), Some(lower_id)) => {
            let difference = scratch.add_raw(Expr::Sub(upper_id, lower_id));
            (
                display_expr(&scratch, difference),
                latex_expr(&scratch, difference),
                Some(difference),
            )
        }
        // At least one endpoint is limit notation, which has no node to
        // subtract from. Delimit the subtrahend by hand so the minus still
        // reaches all of it.
        (_, lower_value) => {
            let needs_parens = lower_value.is_none_or(|id| is_additive_chain(&scratch, id));
            if needs_parens {
                (
                    format!("{} - ({})", upper_display, lower_display),
                    format!("{} - \\left({}\\right)", upper_latex, lower_latex),
                    None,
                )
            } else {
                (
                    format!("{} - {}", upper_display, lower_display),
                    format!("{} - {}", upper_latex, lower_latex),
                    None,
                )
            }
        }
    };

    // C1.8: this sub-step ASSERTS that `after` is an antiderivative of `before`.
    // Declared and checked (differentiate the after, compare to the before)
    // instead of trusted. A refuted claim is not published; an undecided one is
    // (a surd the simplifier cannot fold is not evidence of a lie).
    let mut substeps = Vec::new();
    if let Some(step) = SubStep::checked(
        &scratch,
        crate::didactic::substep::Claim::Antiderivative {
            var: var_name.clone(),
        },
        integrand,
        antiderivative,
        "integral.find_antiderivative",
        vec![],
        display_expr(ctx, integrand),
        display_expr(&scratch, antiderivative),
    ) {
        substeps.push(
            step.with_before_latex(latex_expr(ctx, integrand))
                .with_after_latex(latex_expr(&scratch, antiderivative)),
        );
    }

    // "Find the antiderivative" states WHAT was obtained and never HOW: on
    // `∫dx/(x^5-x-1)` a 200-character `root_sum` appears out of nowhere. The
    // rest of the chain already knows how to narrate the INDEFINITE integral, so
    // hand it a synthetic 2-arg step and splice its narration in between.
    // Recursion is safe by construction: this narrator requires `args.len() == 4`
    // and the synthetic step has 2, so it declines itself.
    if depth < MAX_NARRATION_RECURSION_DEPTH {
        let mut method_scratch = scratch.clone();
        let indefinite = method_scratch.add(Expr::Function(*fn_id, vec![integrand, var_expr]));
        let synthetic = Step::new_compact(
            step.description.as_str(),
            "Symbolic Integration",
            indefinite,
            antiderivative,
        );
        substeps.extend(generate_focused_rule_substeps_at_depth(
            &method_scratch,
            &synthetic,
            depth + 1,
        ));
    }

    // C1.8: this sub-step ASSERTS `F(upper) − F(lower)`, so it must CARRY the
    // bounds — the type takes them as data precisely because a sub-step that
    // evaluates at bounds it does not hold cannot render them, which is how
    // `∫|2x−1|dx ⇒ 5/2` reached the page.
    //
    // Only the both-endpoints-finite branch declares the relation. When an
    // endpoint is infinite or the substituted form is undefined, the after is
    // LIMIT NOTATION with no node behind it: the honest answer is that this
    // layer cannot check it yet (the `Limit` arm is a cycle of its own), not a
    // relation asserted over a string.
    let evaluation = match difference_node {
        Some(difference) => SubStep::checked(
            &scratch,
            crate::didactic::substep::Claim::DefiniteEval {
                var: var_name.clone(),
                lower,
                upper,
            },
            antiderivative,
            difference,
            "integral.evaluate_antiderivative_at_bounds",
            vec![],
            display_expr(&scratch, antiderivative),
            difference_display,
        ),
        None => Some(SubStep::keyed(
            "integral.evaluate_antiderivative_at_bounds",
            vec![],
            display_expr(&scratch, antiderivative),
            difference_display,
        )),
    };
    if let Some(evaluation) = evaluation {
        substeps.push(
            evaluation
                .with_before_latex(latex_expr(&scratch, antiderivative))
                .with_after_latex(difference_latex),
        );
    }
    substeps
}

pub(super) fn affine_internal_derivative_display(
    ctx: &Context,
    arg: ExprId,
    var_name: &str,
    slope: &BigRational,
) -> String {
    format!(
        "d/d{}({}) = {}",
        var_name,
        display_expr(ctx, arg),
        rational_display(slope)
    )
}

pub(super) fn affine_internal_derivative_latex(
    ctx: &Context,
    arg: ExprId,
    var_name: &str,
    slope: &BigRational,
) -> String {
    format!(
        "\\frac{{d}}{{d{}}}\\left({}\\right) = {}",
        var_name,
        latex_expr(ctx, arg),
        rational_latex(slope)
    )
}

pub(super) fn linear_elementary_integrand_arg(
    ctx: &Context,
    integrand: ExprId,
) -> Option<(BuiltinFn, ExprId)> {
    if let Some(arg) = extract_exp_argument(ctx, integrand) {
        return Some((BuiltinFn::Exp, arg));
    }

    let Expr::Function(fn_id, args) = ctx.get(integrand) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }
    let builtin = ctx.builtin_of(*fn_id)?;
    match builtin {
        BuiltinFn::Sin | BuiltinFn::Cos | BuiltinFn::Sinh | BuiltinFn::Cosh => {
            Some((builtin, args[0]))
        }
        _ => None,
    }
}

pub(super) fn trig_log_table_integrand_arg(
    ctx: &Context,
    integrand: ExprId,
    var_name: &str,
) -> Option<TrigLogTableMatch> {
    if let Expr::Function(fn_id, args) = ctx.get(integrand) {
        if args.len() == 1 {
            let kind = match ctx.builtin_of(*fn_id)? {
                BuiltinFn::Tan => TrigLogTableKind::Tangent,
                BuiltinFn::Cot => TrigLogTableKind::Cotangent,
                BuiltinFn::Sec => TrigLogTableKind::Secant,
                BuiltinFn::Csc => TrigLogTableKind::Cosecant,
                _ => return None,
            };
            return nontrivial_affine_argument(ctx, args[0], var_name).then_some(
                TrigLogTableMatch {
                    kind,
                    arg: args[0],
                    trace: TrigLogTableTrace::AffineArgument,
                },
            );
        }
    }

    if let Some(table_match) = trig_log_polynomial_reciprocal_integrand(ctx, integrand, var_name) {
        return Some(table_match);
    }

    if let Some(table_match) = trig_log_sqrt_chain_integrand(ctx, integrand, var_name) {
        return Some(table_match);
    }

    if let Some(table_match) = trig_log_constant_multiple_affine_integrand(ctx, integrand, var_name)
    {
        return Some(table_match);
    }

    if let Some(table_match) = trig_log_symbolic_scaled_quotient_integrand(ctx, integrand, var_name)
    {
        return Some(table_match);
    }

    let (num, den) = as_div(ctx, integrand)?;
    if let Some(table_match) = trig_log_polynomial_quotient_integrand(ctx, num, den, var_name) {
        return Some(table_match);
    }

    if let Some(coefficient) = as_rational_const(ctx, num, 8) {
        if coefficient != BigRational::one() {
            return None;
        }
        let (den_builtin, den_arg) = unary_builtin_arg(ctx, den)?;
        if !nontrivial_affine_argument(ctx, den_arg, var_name) {
            return None;
        }
        let kind = match den_builtin {
            BuiltinFn::Cos => TrigLogTableKind::Secant,
            BuiltinFn::Sin => TrigLogTableKind::Cosecant,
            _ => return None,
        };
        return Some(TrigLogTableMatch {
            kind,
            arg: den_arg,
            trace: TrigLogTableTrace::AffineArgument,
        });
    }

    let (num_builtin, num_arg) = unary_builtin_arg(ctx, num)?;
    let (den_builtin, den_arg) = unary_builtin_arg(ctx, den)?;
    if compare_expr(ctx, num_arg, den_arg) != Ordering::Equal
        || !nontrivial_affine_argument(ctx, num_arg, var_name)
    {
        return None;
    }

    match (num_builtin, den_builtin) {
        (BuiltinFn::Sin, BuiltinFn::Cos) => Some(TrigLogTableMatch {
            kind: TrigLogTableKind::Tangent,
            arg: num_arg,
            trace: TrigLogTableTrace::AffineArgument,
        }),
        (BuiltinFn::Cos, BuiltinFn::Sin) => Some(TrigLogTableMatch {
            kind: TrigLogTableKind::Cotangent,
            arg: num_arg,
            trace: TrigLogTableTrace::AffineArgument,
        }),
        _ => None,
    }
}

fn trig_log_constant_multiple_affine_integrand(
    ctx: &Context,
    integrand: ExprId,
    var_name: &str,
) -> Option<TrigLogTableMatch> {
    if let Some(table_match) =
        trig_log_denominator_scaled_affine_integrand(ctx, integrand, var_name)
    {
        return Some(table_match);
    }

    let (negative, factors) = signed_mul_factors(ctx, integrand);
    let mut coefficient = if negative {
        -BigRational::one()
    } else {
        BigRational::one()
    };
    let mut matched_trig: Option<(TrigLogTableKind, ExprId)> = None;
    for factor in factors {
        if let Some(value) = as_rational_const(ctx, factor, 8) {
            coefficient *= value;
            continue;
        }
        let (kind, arg) = trig_log_table_factor_kind_and_arg(ctx, factor)?;
        if matched_trig.is_some() || !nontrivial_affine_argument(ctx, arg, var_name) {
            return None;
        }
        matched_trig = Some((kind, arg));
    }
    if coefficient.is_zero() || coefficient.is_one() {
        return None;
    }
    let (kind, arg) = matched_trig?;
    Some(TrigLogTableMatch {
        kind,
        arg,
        trace: TrigLogTableTrace::ConstantMultipleAffineArgument {
            cofactor_display: rational_display(&coefficient),
            cofactor_latex: rational_latex(&coefficient),
            coefficient: coefficient.clone(),
            slope: affine_argument_slope(ctx, arg, var_name)?,
        },
    })
}

fn trig_log_denominator_scaled_affine_integrand(
    ctx: &Context,
    integrand: ExprId,
    var_name: &str,
) -> Option<TrigLogTableMatch> {
    let (num, den) = as_div(ctx, integrand)?;
    let numerator_scale = as_rational_const(ctx, num, 8)?;
    let (denominator_scale, kind, arg) = trig_log_denominator_factor_kind_and_arg(ctx, den)?;
    let coefficient = numerator_scale / denominator_scale;
    if coefficient.is_zero()
        || coefficient.is_one()
        || !nontrivial_affine_argument(ctx, arg, var_name)
    {
        return None;
    }
    Some(TrigLogTableMatch {
        kind,
        arg,
        trace: TrigLogTableTrace::ConstantMultipleAffineArgument {
            cofactor_display: rational_display(&coefficient),
            cofactor_latex: rational_latex(&coefficient),
            coefficient: coefficient.clone(),
            slope: affine_argument_slope(ctx, arg, var_name)?,
        },
    })
}

fn trig_log_sqrt_chain_integrand(
    ctx: &Context,
    integrand: ExprId,
    var_name: &str,
) -> Option<TrigLogTableMatch> {
    match ctx.get(integrand) {
        Expr::Div(num, den) => {
            let (numerator_negative, numerator_factors) = signed_mul_factors(ctx, *num);
            trig_log_sqrt_chain_from_factors(
                ctx,
                numerator_negative,
                &numerator_factors,
                *den,
                var_name,
            )
        }
        Expr::Mul(left, right) => trig_log_sqrt_chain_scaled_div(ctx, *left, *right, var_name)
            .or_else(|| trig_log_sqrt_chain_scaled_div(ctx, *right, *left, var_name)),
        Expr::Neg(inner) => {
            let Expr::Div(num, den) = ctx.get(*inner) else {
                return None;
            };
            let (numerator_negative, numerator_factors) = signed_mul_factors(ctx, *num);
            trig_log_sqrt_chain_from_factors(
                ctx,
                !numerator_negative,
                &numerator_factors,
                *den,
                var_name,
            )
        }
        _ => None,
    }
}

fn trig_log_polynomial_quotient_integrand(
    ctx: &Context,
    num: ExprId,
    den: ExprId,
    var_name: &str,
) -> Option<TrigLogTableMatch> {
    let cofactor = trig_factor_with_polynomial_cofactor(ctx, num, var_name)?;
    let (den_builtin, den_arg) = unary_builtin_arg(ctx, den)?;
    if compare_expr(ctx, cofactor.arg, den_arg) != Ordering::Equal {
        return None;
    }

    let kind = match (cofactor.builtin, den_builtin) {
        (BuiltinFn::Sin, BuiltinFn::Cos) => TrigLogTableKind::Tangent,
        (BuiltinFn::Cos, BuiltinFn::Sin) => TrigLogTableKind::Cotangent,
        _ => return None,
    };

    let arg_poly = Polynomial::from_expr(ctx, cofactor.arg, var_name).ok()?;
    if arg_poly.degree() <= 1 {
        return None;
    }
    let derivative = arg_poly.derivative();
    let scale = constant_polynomial_ratio(&cofactor.polynomial, &derivative)?;
    if scale.is_zero() {
        return None;
    }

    let (derivative_display, derivative_latex) = polynomial_display_and_latex(ctx, &derivative);
    Some(TrigLogTableMatch {
        kind,
        arg: cofactor.arg,
        trace: TrigLogTableTrace::PolynomialCofactor(Box::new(TrigLogPolynomialCofactorTrace {
            cofactor_display: display_expr(ctx, cofactor.expr),
            cofactor_latex: latex_expr(ctx, cofactor.expr),
            derivative_display,
            derivative_latex,
            scale,
            symbolic_scale_display: None,
            symbolic_scale_latex: None,
        })),
    })
}

fn trig_log_symbolic_scaled_quotient_integrand(
    ctx: &Context,
    integrand: ExprId,
    var_name: &str,
) -> Option<TrigLogTableMatch> {
    match ctx.get(integrand) {
        Expr::Div(num, den) => {
            let (negative, numerator_factors) = signed_mul_factors(ctx, *num);
            trig_log_symbolic_scaled_quotient_from_factors(
                ctx,
                negative,
                &numerator_factors,
                *den,
                var_name,
            )
        }
        Expr::Mul(left, right) => trig_log_symbolic_scaled_quotient_scaled_div(
            ctx, *left, *right, var_name,
        )
        .or_else(|| trig_log_symbolic_scaled_quotient_scaled_div(ctx, *right, *left, var_name)),
        Expr::Neg(inner) => {
            let Expr::Div(num, den) = ctx.get(*inner) else {
                return None;
            };
            let (negative, numerator_factors) = signed_mul_factors(ctx, *num);
            trig_log_symbolic_scaled_quotient_from_factors(
                ctx,
                !negative,
                &numerator_factors,
                *den,
                var_name,
            )
        }
        _ => None,
    }
}

fn trig_log_polynomial_reciprocal_integrand(
    ctx: &Context,
    integrand: ExprId,
    var_name: &str,
) -> Option<TrigLogTableMatch> {
    match ctx.get(integrand) {
        Expr::Div(num, den) => {
            if let Ok(numerator) = Polynomial::from_expr(ctx, *num, var_name) {
                return trig_log_polynomial_reciprocal_arg(ctx, numerator, *den, var_name).map(
                    |(kind, arg, derivative, scale)| {
                        let (derivative_display, derivative_latex) =
                            polynomial_display_and_latex(ctx, &derivative);
                        TrigLogTableMatch {
                            kind,
                            arg,
                            trace: TrigLogTableTrace::PolynomialCofactor(Box::new(
                                TrigLogPolynomialCofactorTrace {
                                    cofactor_display: display_expr(ctx, *num),
                                    cofactor_latex: latex_expr(ctx, *num),
                                    derivative_display,
                                    derivative_latex,
                                    scale,
                                    symbolic_scale_display: None,
                                    symbolic_scale_latex: None,
                                },
                            )),
                        }
                    },
                );
            }

            let (negative, numerator_factors) = signed_mul_factors(ctx, *num);
            trig_log_symbolic_scaled_polynomial_reciprocal_arg(
                ctx,
                negative,
                &numerator_factors,
                *den,
                var_name,
            )
        }
        Expr::Mul(left, right) => trig_log_scaled_polynomial_reciprocal_integrand(
            ctx, *left, *right, var_name,
        )
        .or_else(|| trig_log_scaled_polynomial_reciprocal_integrand(ctx, *right, *left, var_name)),
        _ => None,
    }
}

fn trig_log_scaled_polynomial_reciprocal_integrand(
    ctx: &Context,
    scale_expr: ExprId,
    div_expr: ExprId,
    var_name: &str,
) -> Option<TrigLogTableMatch> {
    let scale = as_rational_const(ctx, scale_expr, 8)?;
    if scale.is_zero() {
        return None;
    }
    let Expr::Div(num, den) = ctx.get(div_expr) else {
        return None;
    };
    if let Ok(numerator) = Polynomial::from_expr(ctx, *num, var_name) {
        let numerator = scale_polynomial(&numerator, &scale);
        return trig_log_polynomial_reciprocal_arg(ctx, numerator, *den, var_name).map(
            |(kind, arg, derivative, scale)| {
                let (derivative_display, derivative_latex) =
                    polynomial_display_and_latex(ctx, &derivative);
                TrigLogTableMatch {
                    kind,
                    arg,
                    trace: TrigLogTableTrace::PolynomialCofactor(Box::new(
                        TrigLogPolynomialCofactorTrace {
                            cofactor_display: format!(
                                "{} · {}",
                                display_expr(ctx, scale_expr),
                                display_expr(ctx, *num)
                            ),
                            cofactor_latex: format!(
                                "{}\\cdot {}",
                                latex_expr(ctx, scale_expr),
                                latex_expr(ctx, *num)
                            ),
                            derivative_display,
                            derivative_latex,
                            scale,
                            symbolic_scale_display: None,
                            symbolic_scale_latex: None,
                        },
                    )),
                }
            },
        );
    }

    let (scale_negative, mut numerator_factors) = signed_mul_factors(ctx, scale_expr);
    let (num_negative, num_factors) = signed_mul_factors(ctx, *num);
    numerator_factors.extend(num_factors);
    trig_log_symbolic_scaled_polynomial_reciprocal_arg(
        ctx,
        scale_negative != num_negative,
        &numerator_factors,
        *den,
        var_name,
    )
}

pub(super) fn hyperbolic_log_sqrt_chain_integrand(
    ctx: &Context,
    integrand: ExprId,
    var_name: &str,
) -> Option<HyperbolicLogTableMatch> {
    match ctx.get(integrand) {
        Expr::Div(num, den) => {
            let (numerator_negative, numerator_factors) = signed_mul_factors(ctx, *num);
            hyperbolic_log_sqrt_chain_from_factors(
                ctx,
                numerator_negative,
                &numerator_factors,
                *den,
                var_name,
            )
        }
        Expr::Mul(left, right) => {
            hyperbolic_log_sqrt_chain_scaled_div(ctx, *left, *right, var_name)
                .or_else(|| hyperbolic_log_sqrt_chain_scaled_div(ctx, *right, *left, var_name))
        }
        Expr::Neg(inner) => {
            let Expr::Div(num, den) = ctx.get(*inner) else {
                return None;
            };
            let (numerator_negative, numerator_factors) = signed_mul_factors(ctx, *num);
            hyperbolic_log_sqrt_chain_from_factors(
                ctx,
                !numerator_negative,
                &numerator_factors,
                *den,
                var_name,
            )
        }
        _ => None,
    }
}

pub(super) fn hyperbolic_reciprocal_sqrt_chain_integrand(
    ctx: &Context,
    integrand: ExprId,
    var_name: &str,
) -> Option<HyperbolicReciprocalTableMatch> {
    match ctx.get(integrand) {
        Expr::Div(num, den) => {
            let (numerator_negative, numerator_factors) = signed_mul_factors(ctx, *num);
            hyperbolic_reciprocal_sqrt_chain_from_factors(
                ctx,
                numerator_negative,
                &numerator_factors,
                *den,
                var_name,
            )
        }
        Expr::Mul(left, right) => hyperbolic_reciprocal_sqrt_chain_scaled_div(
            ctx, *left, *right, var_name,
        )
        .or_else(|| hyperbolic_reciprocal_sqrt_chain_scaled_div(ctx, *right, *left, var_name)),
        Expr::Neg(inner) => {
            let Expr::Div(num, den) = ctx.get(*inner) else {
                return None;
            };
            let (numerator_negative, numerator_factors) = signed_mul_factors(ctx, *num);
            hyperbolic_reciprocal_sqrt_chain_from_factors(
                ctx,
                !numerator_negative,
                &numerator_factors,
                *den,
                var_name,
            )
        }
        _ => None,
    }
}

pub(super) fn hyperbolic_reciprocal_derivative_match(
    ctx: &Context,
    numerator_negative: bool,
    numerator_factors: &[ExprId],
    denominator_factors: &[ExprId],
    var_name: &str,
) -> Option<HyperbolicReciprocalTableMatch> {
    for (denominator_index, factor) in denominator_factors.iter().enumerate() {
        let Some((den_builtin, arg)) = hyperbolic_square_denominator_arg(ctx, *factor) else {
            continue;
        };
        let (kind, numerator_builtin) = match den_builtin {
            BuiltinFn::Cosh => (
                HyperbolicReciprocalTableKind::SinhOverCoshSquare,
                BuiltinFn::Sinh,
            ),
            BuiltinFn::Sinh => (
                HyperbolicReciprocalTableKind::CoshOverSinhSquare,
                BuiltinFn::Cosh,
            ),
            _ => continue,
        };

        for (numerator_index, factor) in numerator_factors.iter().enumerate() {
            let Some((candidate_builtin, candidate_arg)) = unary_builtin_arg(ctx, *factor) else {
                continue;
            };
            if candidate_builtin != numerator_builtin
                || !same_sqrt_chain_arg(ctx, candidate_arg, arg)
            {
                continue;
            }

            let remaining_numerator = numerator_factors
                .iter()
                .enumerate()
                .filter_map(|(idx, factor)| (idx != numerator_index).then_some(*factor))
                .collect::<Vec<_>>();
            let remaining_denominator = denominator_factors
                .iter()
                .enumerate()
                .filter_map(|(idx, factor)| (idx != denominator_index).then_some(*factor))
                .collect::<Vec<_>>();

            if let Some(table_match) = hyperbolic_reciprocal_match_from_cofactor(
                ctx,
                kind,
                arg,
                numerator_negative,
                &remaining_numerator,
                &remaining_denominator,
                var_name,
            ) {
                return Some(table_match);
            }
        }
    }

    None
}

pub(super) fn generate_polynomial_derivative_table_integration_substeps(
    ctx: &Context,
    step: &Step,
) -> Vec<SubStep> {
    if step.rule_name != "Symbolic Integration" {
        return Vec::new();
    }

    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Expr::Function(fn_id, args) = ctx.get(before) else {
        return Vec::new();
    };
    if ctx.sym_name(*fn_id) != "integrate" || args.len() != 2 {
        return Vec::new();
    }
    if let Expr::Function(after_fn_id, _) = ctx.get(after) {
        if ctx.sym_name(*after_fn_id) == "integrate" {
            return Vec::new();
        }
    }

    let Expr::Variable(var_sym) = ctx.get(args[1]) else {
        return Vec::new();
    };
    let var_name = ctx.sym_name(*var_sym);
    let Some(table_match) = polynomial_derivative_table_integrand(ctx, args[0], var_name) else {
        return Vec::new();
    };

    let title = match table_match.builtin {
        BuiltinFn::Exp => "Usar la regla de exp(u) -> exp(u)",
        BuiltinFn::Cos => "Usar la regla de cos(u) -> sin(u)",
        BuiltinFn::Sin => "Usar la regla de sin(u) -> -cos(u)",
        BuiltinFn::Sinh => "Usar la regla de sinh(u) -> cosh(u)",
        BuiltinFn::Cosh => "Usar la regla de cosh(u) -> sinh(u)",
        BuiltinFn::Tanh => "Usar la regla de tanh(u) -> ln(cosh(u))",
        _ => return Vec::new(),
    };

    let mut substeps = Vec::new();
    if let Some(step) = checked_antiderivative_substep(ctx, title, args[0], after, var_name) {
        substeps.push(step);
    }
    substeps.push(
        SubStep::keyed(
            "usub.identify_u_du",
            vec![],
            format!("u = {}", display_expr(ctx, table_match.arg)),
            format!("du = {} dx", table_match.derivative_display),
        )
        .with_before_latex(format!("u = {}", latex_expr(ctx, table_match.arg)))
        .with_after_latex(format!("du = {}\\,dx", table_match.derivative_latex)),
    );
    push_integration_constant_factor_adjustment_substep(
        &mut substeps,
        IntegrationConstantFactorAdjustment {
            cofactor_display: &table_match.cofactor_display,
            cofactor_latex: &table_match.cofactor_latex,
            derivative_display: &table_match.derivative_display,
            derivative_latex: &table_match.derivative_latex,
            scale: &table_match.scale,
            symbolic_scale_display: table_match.symbolic_scale_display.as_deref(),
            symbolic_scale_latex: table_match.symbolic_scale_latex.as_deref(),
        },
    );

    substeps
}

fn polynomial_derivative_table_integrand(
    ctx: &Context,
    integrand: ExprId,
    var_name: &str,
) -> Option<PolynomialDerivativeTableMatch> {
    match ctx.get(integrand) {
        Expr::Mul(_, _) | Expr::Neg(_) => {
            let (negative, factors) = signed_mul_factors(ctx, integrand);
            polynomial_derivative_table_from_factors(ctx, negative, &factors, var_name)
        }
        _ => None,
    }
}

fn polynomial_derivative_table_from_factors(
    ctx: &Context,
    negative: bool,
    factors: &[ExprId],
    var_name: &str,
) -> Option<PolynomialDerivativeTableMatch> {
    for (kernel_index, factor) in factors.iter().enumerate() {
        let Some((builtin, arg)) = polynomial_derivative_kernel_arg(ctx, *factor) else {
            continue;
        };
        if !contains_named_var(ctx, arg, var_name) {
            continue;
        }

        let remaining_factors = factors
            .iter()
            .enumerate()
            .filter_map(|(idx, factor)| (idx != kernel_index).then_some(*factor))
            .collect::<Vec<_>>();
        if let Some(trace) =
            polynomial_derivative_cofactor_trace(ctx, negative, &remaining_factors, arg, var_name)
        {
            return Some(PolynomialDerivativeTableMatch {
                builtin,
                arg,
                cofactor_display: trace.cofactor_display,
                cofactor_latex: trace.cofactor_latex,
                derivative_display: trace.derivative_display,
                derivative_latex: trace.derivative_latex,
                scale: trace.scale,
                symbolic_scale_display: trace.symbolic_scale_display,
                symbolic_scale_latex: trace.symbolic_scale_latex,
            });
        }

        if let Some(trace) = polynomial_derivative_cofactor_trace_with_symbolic_scale(
            ctx,
            negative,
            &remaining_factors,
            arg,
            var_name,
        ) {
            return Some(PolynomialDerivativeTableMatch {
                builtin,
                arg,
                cofactor_display: trace.cofactor_display,
                cofactor_latex: trace.cofactor_latex,
                derivative_display: trace.derivative_display,
                derivative_latex: trace.derivative_latex,
                scale: trace.scale,
                symbolic_scale_display: trace.symbolic_scale_display,
                symbolic_scale_latex: trace.symbolic_scale_latex,
            });
        }
    }

    None
}

fn polynomial_derivative_kernel_arg(ctx: &Context, expr: ExprId) -> Option<(BuiltinFn, ExprId)> {
    if let Some(arg) = extract_exp_argument(ctx, expr) {
        return Some((BuiltinFn::Exp, arg));
    }

    let (builtin, arg) = unary_builtin_arg(ctx, expr)?;
    match builtin {
        BuiltinFn::Sin | BuiltinFn::Cos | BuiltinFn::Sinh | BuiltinFn::Cosh | BuiltinFn::Tanh => {
            Some((builtin, arg))
        }
        _ => None,
    }
}

pub(super) fn polynomial_derivative_cofactor_trace_with_symbolic_scale(
    ctx: &Context,
    negative: bool,
    cofactor_factors: &[ExprId],
    arg: ExprId,
    var_name: &str,
) -> Option<PolynomialDerivativeCofactorTrace> {
    let arg_poly = polynomial_trace_arg_ignoring_independent_addends(ctx, arg, var_name)?;
    if arg_poly.degree() == 0 {
        return None;
    }
    let derivative_poly = arg_poly.derivative();
    if derivative_poly.is_zero() {
        return None;
    }

    let mut scratch = ctx.clone();
    let mut signed_cofactor_factors = Vec::new();
    if negative {
        signed_cofactor_factors
            .push(scratch.add(Expr::Number(BigRational::from_integer((-1).into()))));
    }
    signed_cofactor_factors.extend_from_slice(cofactor_factors);
    let cofactor_expr = build_quotient_from_factors(&mut scratch, &signed_cofactor_factors, &[]);
    let cofactor_simplified = simplify_expr_in_context(&mut scratch, cofactor_expr);

    let mut derivative_factors = Vec::new();
    let mut scale_factors = Vec::new();
    for factor in &signed_cofactor_factors {
        if contains_named_var(&scratch, *factor, var_name) {
            derivative_factors.push(*factor);
        } else {
            scale_factors.push(*factor);
        }
    }
    if derivative_factors.is_empty()
        || scale_factors.is_empty()
        || !scale_factors
            .iter()
            .any(|factor| as_rational_const(&scratch, *factor, 8).is_none())
    {
        return None;
    }

    let derivative_cofactor_expr = build_mul_expr_from_factors(&mut scratch, &derivative_factors);
    let derivative_cofactor_poly =
        Polynomial::from_expr(&scratch, derivative_cofactor_expr, var_name).ok()?;
    let rational_scale = constant_polynomial_ratio(&derivative_cofactor_poly, &derivative_poly)?;
    if rational_scale.is_zero() {
        return None;
    }

    let symbolic_scale = build_mul_expr_from_factors(&mut scratch, &scale_factors);
    let scaled_symbolic_scale = if rational_scale.is_one() {
        symbolic_scale
    } else {
        let rational_expr = scratch.add(Expr::Number(rational_scale));
        scratch.add(Expr::Mul(rational_expr, symbolic_scale))
    };
    let scaled_symbolic_scale = simplify_expr_in_context(&mut scratch, scaled_symbolic_scale);
    if contains_named_var(&scratch, scaled_symbolic_scale, var_name) {
        return None;
    }

    let (derivative_display, derivative_latex) =
        polynomial_display_and_latex(ctx, &derivative_poly);
    Some(PolynomialDerivativeCofactorTrace {
        derivative_display,
        derivative_latex,
        scale: BigRational::one(),
        cofactor_display: display_expr(&scratch, cofactor_simplified),
        cofactor_latex: latex_expr(&scratch, cofactor_simplified),
        symbolic_scale_display: Some(display_expr(&scratch, scaled_symbolic_scale)),
        symbolic_scale_latex: Some(latex_expr(&scratch, scaled_symbolic_scale)),
    })
}

pub(super) fn symbolic_linear_exact_derivative_cofactor_trace(
    ctx: &Context,
    numerator_negative: bool,
    numerator_factors: &[ExprId],
    arg: ExprId,
    var_name: &str,
) -> Option<LinearDerivativeCofactorTrace> {
    let mut scratch = ctx.clone();
    let derivative = extract_non_unit_affine_var_coeff_with_sign(&scratch, arg, var_name)?;

    let mut cofactor_factors = Vec::new();
    if numerator_negative {
        cofactor_factors.push(scratch.add(Expr::Number(BigRational::from_integer((-1).into()))));
    }
    cofactor_factors.extend_from_slice(numerator_factors);
    let cofactor_expr = build_quotient_from_factors(&mut scratch, &cofactor_factors, &[]);
    let cofactor_simplified = simplify_expr_in_context(&mut scratch, cofactor_expr);
    let cofactor_is_negative =
        expr_matches_signed_affine_coeff(&scratch, cofactor_simplified, derivative)?;
    let scale = if cofactor_is_negative == derivative.is_negative {
        BigRational::one()
    } else {
        -BigRational::one()
    };

    Some(LinearDerivativeCofactorTrace {
        derivative_display: signed_affine_coeff_display(&scratch, derivative),
        derivative_latex: signed_affine_coeff_latex(&scratch, derivative),
        scale,
        cofactor_display: display_expr(&scratch, cofactor_simplified),
        cofactor_latex: latex_expr(&scratch, cofactor_simplified),
    })
}

pub(super) fn symbolic_linear_scaled_derivative_cofactor_trace(
    ctx: &Context,
    numerator_negative: bool,
    numerator_factors: &[ExprId],
    arg: ExprId,
    var_name: &str,
) -> Option<LinearDerivativeCofactorTrace> {
    let mut scratch = ctx.clone();
    let derivative = extract_non_unit_affine_var_coeff_with_sign(&scratch, arg, var_name)?;

    let mut cofactor_factors = Vec::new();
    if numerator_negative {
        cofactor_factors.push(scratch.add(Expr::Number(BigRational::from_integer((-1).into()))));
    }
    cofactor_factors.extend_from_slice(numerator_factors);
    let cofactor_expr = build_quotient_from_factors(&mut scratch, &cofactor_factors, &[]);
    let cofactor_simplified = simplify_expr_in_context(&mut scratch, cofactor_expr);
    let (_cofactor_negative, factors) = signed_mul_factors(&scratch, cofactor_simplified);

    for (idx, factor) in factors.iter().enumerate() {
        if expr_matches_signed_affine_coeff(&scratch, *factor, derivative).is_none() {
            continue;
        }

        let scale_factors = factors
            .iter()
            .enumerate()
            .filter_map(|(factor_idx, factor)| (factor_idx != idx).then_some(*factor))
            .collect::<Vec<_>>();
        if scale_factors.is_empty()
            || scale_factors
                .iter()
                .any(|factor| contains_named_var(&scratch, *factor, var_name))
        {
            return None;
        }

        return Some(LinearDerivativeCofactorTrace {
            derivative_display: signed_affine_coeff_display(&scratch, derivative),
            derivative_latex: signed_affine_coeff_latex(&scratch, derivative),
            scale: BigRational::one(),
            cofactor_display: display_expr(&scratch, cofactor_simplified),
            cofactor_latex: latex_expr(&scratch, cofactor_simplified),
        });
    }

    None
}

pub(super) fn log_power_product_table_integrand(
    ctx: &Context,
    integrand: ExprId,
    var_name: &str,
) -> Option<LogPowerProductTableMatch> {
    match ctx.get(integrand) {
        Expr::Mul(_, _) | Expr::Neg(_) => {
            let (negative, factors) = signed_mul_factors(ctx, integrand);
            log_power_product_table_from_factors(ctx, negative, &factors, var_name)
        }
        _ => None,
    }
}

pub(super) fn polynomial_base_table_integrand(
    ctx: &Context,
    integrand: ExprId,
    var_name: &str,
) -> Option<PolynomialBaseTableMatch> {
    match ctx.get(integrand) {
        Expr::Div(num, den) => polynomial_base_table_from_div(ctx, *num, *den, var_name),
        Expr::Mul(_, _) | Expr::Neg(_) => {
            let (negative, factors) = signed_mul_factors(ctx, integrand);
            polynomial_base_table_from_product(ctx, negative, &factors, var_name)
        }
        _ => None,
    }
}

pub(super) fn scaled_sqrt_affine_derivative_display_and_latex(
    ctx: &Context,
    scale: &BigRational,
    radicand: ExprId,
    var_name: &str,
) -> Option<(String, String)> {
    let radicand_poly = Polynomial::from_expr(ctx, radicand, var_name).ok()?;
    let radicand_slope = radicand_poly.coeffs.get(1).cloned()?;
    if radicand_slope.is_zero() {
        return None;
    }
    let coeff = scale.clone() * radicand_slope / BigRational::from_integer(2.into());
    let numerator = coeff.numer().to_string();
    let denominator = coeff.denom().to_string();
    let radicand_display = display_expr(ctx, radicand);
    let radicand_latex = latex_expr(ctx, radicand);
    Some(match (numerator.as_str(), denominator.as_str()) {
        ("1", "1") => (
            format!("1 / sqrt({radicand_display})"),
            format!("\\frac{{1}}{{\\sqrt{{{radicand_latex}}}}}"),
        ),
        ("-1", "1") => (
            format!("-1 / sqrt({radicand_display})"),
            format!("-\\frac{{1}}{{\\sqrt{{{radicand_latex}}}}}"),
        ),
        ("1", denominator) => (
            format!("1 / ({denominator}·sqrt({radicand_display}))"),
            format!("\\frac{{1}}{{{denominator}\\sqrt{{{radicand_latex}}}}}"),
        ),
        ("-1", denominator) => (
            format!("-1 / ({denominator}·sqrt({radicand_display}))"),
            format!("-\\frac{{1}}{{{denominator}\\sqrt{{{radicand_latex}}}}}"),
        ),
        (numerator, "1") => (
            format!("{numerator} / sqrt({radicand_display})"),
            format!("\\frac{{{numerator}}}{{\\sqrt{{{radicand_latex}}}}}"),
        ),
        (numerator, denominator) => (
            format!("{numerator} / ({denominator}·sqrt({radicand_display}))"),
            format!("\\frac{{{numerator}}}{{{denominator}\\sqrt{{{radicand_latex}}}}}"),
        ),
    })
}

pub(super) fn symbolic_denominator_sqrt_affine_derivative_display_and_latex(
    ctx: &Context,
    scale: &BigRational,
    radicand: ExprId,
    parameter: ExprId,
    var_name: &str,
) -> Option<(String, String)> {
    let radicand_poly = Polynomial::from_expr(ctx, radicand, var_name).ok()?;
    let radicand_slope = radicand_poly.coeffs.get(1).cloned()?;
    if !radicand_slope.is_one() {
        return None;
    }

    let coeff = scale.clone() / BigRational::from_integer(2.into());
    let parameter_display = display_expr(ctx, parameter);
    let radicand_display = display_expr(ctx, radicand);
    let parameter_latex = latex_expr(ctx, parameter);
    let radicand_latex = latex_expr(ctx, radicand);
    Some(if coeff.is_one() {
        (
            format!("1 / ({}·sqrt({}))", parameter_display, radicand_display),
            format!(
                "\\frac{{1}}{{{}\\cdot \\sqrt{{{}}}}}",
                parameter_latex, radicand_latex
            ),
        )
    } else if coeff.numer().is_one() {
        let denominator = coeff.denom();
        (
            format!(
                "1 / ({}·{}·sqrt({}))",
                denominator, parameter_display, radicand_display
            ),
            format!(
                "\\frac{{1}}{{{}\\cdot {}\\cdot \\sqrt{{{}}}}}",
                denominator, parameter_latex, radicand_latex
            ),
        )
    } else {
        let numerator = coeff.numer();
        let denominator = coeff.denom();
        (
            format!(
                "{} / ({}·{}·sqrt({}))",
                numerator, denominator, parameter_display, radicand_display
            ),
            format!(
                "\\frac{{{}}}{{{}\\cdot {}\\cdot \\sqrt{{{}}}}}",
                numerator, denominator, parameter_latex, radicand_latex
            ),
        )
    })
}

pub(super) fn symbolic_multiplier_sqrt_affine_derivative_display_and_latex(
    ctx: &Context,
    scale: &BigRational,
    radicand: ExprId,
    parameter: ExprId,
    var_name: &str,
) -> Option<(String, String)> {
    if !scale.is_one() {
        return None;
    }
    let radicand_poly = Polynomial::from_expr(ctx, radicand, var_name).ok()?;
    let radicand_slope = radicand_poly.coeffs.get(1).cloned()?;
    if !radicand_slope.is_one() {
        return None;
    }

    let parameter_display = display_expr(ctx, parameter);
    let radicand_display = display_expr(ctx, radicand);
    let parameter_latex = latex_expr(ctx, parameter);
    let radicand_latex = latex_expr(ctx, radicand);
    Some((
        format!("{} / (2·sqrt({}))", parameter_display, radicand_display),
        format!(
            "\\frac{{{}}}{{2\\cdot \\sqrt{{{}}}}}",
            parameter_latex, radicand_latex
        ),
    ))
}

pub(super) fn trig_quotient_table_integrand(
    ctx: &Context,
    integrand: ExprId,
    var_name: &str,
) -> Option<TrigQuotientTableMatch> {
    match ctx.get(integrand) {
        Expr::Div(num, den) => trig_quotient_table_from_div(ctx, *num, *den, var_name),
        Expr::Mul(_, _) | Expr::Neg(_) => {
            let (negative, factors) = signed_mul_factors(ctx, integrand);
            trig_quotient_table_from_product(ctx, negative, &factors, var_name)
        }
        _ => None,
    }
}

pub(super) fn generate_reciprocal_trig_derivative_product_integration_substeps(
    ctx: &Context,
    step: &Step,
) -> Vec<SubStep> {
    if step.rule_name != "Symbolic Integration" {
        return Vec::new();
    }

    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Expr::Function(fn_id, args) = ctx.get(before) else {
        return Vec::new();
    };
    if ctx.sym_name(*fn_id) != "integrate" || args.len() != 2 {
        return Vec::new();
    }
    if let Expr::Function(after_fn_id, _) = ctx.get(after) {
        if ctx.sym_name(*after_fn_id) == "integrate" {
            return Vec::new();
        }
    }

    let Expr::Variable(var_sym) = ctx.get(args[1]) else {
        return Vec::new();
    };
    let var_name = ctx.sym_name(*var_sym);
    let Some(table_match) = reciprocal_trig_derivative_product_integrand(ctx, args[0], var_name)
    else {
        return Vec::new();
    };

    let title = match table_match.kind {
        ReciprocalTrigDerivativeProductKind::SecantTangent => {
            "Usar la regla de sec(u)·tan(u) -> sec(u)"
        }
        ReciprocalTrigDerivativeProductKind::CosecantCotangent => {
            "Usar la regla de csc(u)·cot(u) -> -csc(u)"
        }
    };

    let mut substeps = Vec::new();
    if let Some(step) = checked_antiderivative_substep(ctx, title, args[0], after, var_name) {
        substeps.push(step);
    }
    substeps.push(
        SubStep::keyed(
            "usub.identify_u_du",
            vec![],
            format!("u = {}", display_expr(ctx, table_match.arg)),
            format!("du = {} dx", table_match.derivative_display),
        )
        .with_before_latex(format!("u = {}", latex_expr(ctx, table_match.arg)))
        .with_after_latex(format!("du = {}\\,dx", table_match.derivative_latex)),
    );
    push_integration_constant_factor_adjustment_substep(
        &mut substeps,
        IntegrationConstantFactorAdjustment {
            cofactor_display: &table_match.cofactor_display,
            cofactor_latex: &table_match.cofactor_latex,
            derivative_display: &table_match.derivative_display,
            derivative_latex: &table_match.derivative_latex,
            scale: &table_match.scale,
            symbolic_scale_display: table_match.symbolic_scale_display.as_deref(),
            symbolic_scale_latex: table_match.symbolic_scale_latex.as_deref(),
        },
    );

    substeps
}

fn reciprocal_trig_derivative_product_integrand(
    ctx: &Context,
    integrand: ExprId,
    var_name: &str,
) -> Option<ReciprocalTrigDerivativeProductMatch> {
    if let Some(table_match) =
        sqrt_chain_reciprocal_trig_derivative_product_integrand(ctx, integrand, var_name)
    {
        return Some(table_match);
    }

    match ctx.get(integrand) {
        Expr::Div(num, den) => {
            reciprocal_trig_derivative_product_div(ctx, *num, *den, var_name, None)
        }
        Expr::Mul(left, right) => reciprocal_trig_derivative_product_scaled_div(
            ctx, *left, *right, var_name,
        )
        .or_else(|| reciprocal_trig_derivative_product_scaled_div(ctx, *right, *left, var_name)),
        Expr::Neg(inner) => reciprocal_trig_derivative_product_integrand(ctx, *inner, var_name)
            .map(negate_reciprocal_trig_derivative_product_match),
        _ => None,
    }
}

fn negate_reciprocal_trig_derivative_product_match(
    mut table_match: ReciprocalTrigDerivativeProductMatch,
) -> ReciprocalTrigDerivativeProductMatch {
    table_match.scale = -table_match.scale;
    table_match.cofactor_display = format!("-{}", table_match.cofactor_display);
    table_match.cofactor_latex = format!("-{}", table_match.cofactor_latex);
    if let Some(scale_display) = table_match.symbolic_scale_display.as_mut() {
        *scale_display = format!("-{}", scale_display);
    }
    if let Some(scale_latex) = table_match.symbolic_scale_latex.as_mut() {
        *scale_latex = format!("-{}", scale_latex);
    }
    table_match
}

fn reciprocal_trig_derivative_product_scaled_div(
    ctx: &Context,
    scale_expr: ExprId,
    div_expr: ExprId,
    var_name: &str,
) -> Option<ReciprocalTrigDerivativeProductMatch> {
    let scale = as_rational_const(ctx, scale_expr, 8)?;
    if scale.is_zero() {
        return None;
    }
    let Expr::Div(num, den) = ctx.get(div_expr) else {
        return None;
    };
    reciprocal_trig_derivative_product_div(ctx, *num, *den, var_name, Some((scale_expr, scale)))
}

fn reciprocal_trig_derivative_product_div(
    ctx: &Context,
    num: ExprId,
    den: ExprId,
    var_name: &str,
    outer_scale: Option<(ExprId, BigRational)>,
) -> Option<ReciprocalTrigDerivativeProductMatch> {
    if outer_scale.is_none() {
        if let Some(table_match) =
            reciprocal_trig_derivative_product_direct_quotient(ctx, num, den, var_name)
        {
            return Some(table_match);
        }
    }

    if outer_scale.is_none() {
        if let Some(table_match) =
            reciprocal_trig_derivative_product_symbolic_linear_exact_div(ctx, num, den, var_name)
        {
            return Some(table_match);
        }
    }

    let mut cofactor = reciprocal_trig_derivative_product_numerator_cofactor(ctx, num, var_name)?;
    let (den_builtin, den_arg) = trig_square_denominator_arg(ctx, den)?;
    if compare_expr(ctx, cofactor.arg, den_arg) != Ordering::Equal {
        return None;
    }

    let kind = match (cofactor.builtin, den_builtin) {
        (BuiltinFn::Sin, BuiltinFn::Cos) => ReciprocalTrigDerivativeProductKind::SecantTangent,
        (BuiltinFn::Cos, BuiltinFn::Sin) => ReciprocalTrigDerivativeProductKind::CosecantCotangent,
        _ => return None,
    };

    if let Some((scale_expr, scale)) = outer_scale {
        cofactor.polynomial = scale_polynomial(&cofactor.polynomial, &scale);
        cofactor.display = format!("{} · {}", display_expr(ctx, scale_expr), cofactor.display);
        cofactor.latex = format!("{}\\cdot {}", latex_expr(ctx, scale_expr), cofactor.latex);
    }

    reciprocal_trig_derivative_product_match_from_cofactor(
        ctx,
        kind,
        cofactor.arg,
        cofactor.display,
        cofactor.latex,
        cofactor.polynomial,
        var_name,
    )
}

fn reciprocal_trig_derivative_product_direct_quotient(
    ctx: &Context,
    num: ExprId,
    den: ExprId,
    var_name: &str,
) -> Option<ReciprocalTrigDerivativeProductMatch> {
    let (num_builtin, num_arg) = unary_builtin_arg(ctx, num)?;
    let (den_builtin, den_arg) = unary_builtin_arg(ctx, den)?;
    if compare_expr(ctx, num_arg, den_arg) != Ordering::Equal {
        return None;
    }

    let kind = match (num_builtin, den_builtin) {
        (BuiltinFn::Tan, BuiltinFn::Cos) => ReciprocalTrigDerivativeProductKind::SecantTangent,
        (BuiltinFn::Cot, BuiltinFn::Sin) => ReciprocalTrigDerivativeProductKind::CosecantCotangent,
        _ => return None,
    };

    reciprocal_trig_derivative_product_match_from_cofactor(
        ctx,
        kind,
        num_arg,
        "1".to_string(),
        "1".to_string(),
        Polynomial::one(var_name.to_string()),
        var_name,
    )
}

fn reciprocal_trig_derivative_product_symbolic_linear_exact_div(
    ctx: &Context,
    num: ExprId,
    den: ExprId,
    var_name: &str,
) -> Option<ReciprocalTrigDerivativeProductMatch> {
    let (num_builtin, num_arg, cofactor_negative, cofactor_factors) =
        reciprocal_trig_derivative_product_numerator_expr_cofactor(ctx, num)?;
    let (den_builtin, den_arg) = trig_square_denominator_arg(ctx, den)?;
    if compare_expr(ctx, num_arg, den_arg) != Ordering::Equal {
        return None;
    }

    let kind = match (num_builtin, den_builtin) {
        (BuiltinFn::Sin, BuiltinFn::Cos) => ReciprocalTrigDerivativeProductKind::SecantTangent,
        (BuiltinFn::Cos, BuiltinFn::Sin) => ReciprocalTrigDerivativeProductKind::CosecantCotangent,
        _ => return None,
    };

    if let Some(trace) = symbolic_linear_exact_derivative_cofactor_trace(
        ctx,
        cofactor_negative,
        &cofactor_factors,
        num_arg,
        var_name,
    ) {
        return Some(ReciprocalTrigDerivativeProductMatch {
            kind,
            arg: num_arg,
            cofactor_display: trace.cofactor_display,
            cofactor_latex: trace.cofactor_latex,
            derivative_display: trace.derivative_display,
            derivative_latex: trace.derivative_latex,
            scale: trace.scale,
            symbolic_scale_display: None,
            symbolic_scale_latex: None,
        });
    }

    let trace = symbolic_linear_scaled_derivative_cofactor_trace(
        ctx,
        cofactor_negative,
        &cofactor_factors,
        num_arg,
        var_name,
    )?;

    Some(ReciprocalTrigDerivativeProductMatch {
        kind,
        arg: num_arg,
        cofactor_display: trace.cofactor_display,
        cofactor_latex: trace.cofactor_latex,
        derivative_display: trace.derivative_display,
        derivative_latex: trace.derivative_latex,
        scale: trace.scale,
        symbolic_scale_display: None,
        symbolic_scale_latex: None,
    })
}

fn reciprocal_trig_derivative_product_numerator_expr_cofactor(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BuiltinFn, ExprId, bool, Vec<ExprId>)> {
    let (negative, factors) = signed_mul_factors(ctx, expr);
    let mut trig_factor = None;

    for (idx, factor) in factors.iter().enumerate() {
        let Some((builtin, arg)) = unary_builtin_arg(ctx, *factor) else {
            continue;
        };
        if !matches!(builtin, BuiltinFn::Sin | BuiltinFn::Cos) {
            continue;
        }
        if trig_factor.replace((idx, builtin, arg)).is_some() {
            return None;
        }
    }

    let (trig_index, builtin, arg) = trig_factor?;
    let cofactor_factors = factors
        .iter()
        .enumerate()
        .filter_map(|(idx, factor)| (idx != trig_index).then_some(*factor))
        .collect::<Vec<_>>();
    if cofactor_factors.is_empty() {
        return None;
    }
    Some((builtin, arg, negative, cofactor_factors))
}

fn reciprocal_trig_derivative_product_numerator_cofactor(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
) -> Option<ReciprocalTrigNumeratorCofactor> {
    if let Some((builtin, arg)) = unary_builtin_arg(ctx, expr) {
        if matches!(builtin, BuiltinFn::Sin | BuiltinFn::Cos) {
            return Some(ReciprocalTrigNumeratorCofactor {
                builtin,
                arg,
                display: "1".to_string(),
                latex: "1".to_string(),
                polynomial: Polynomial::one(var_name.to_string()),
            });
        }
    }

    let cofactor = trig_factor_with_polynomial_cofactor(ctx, expr, var_name)?;
    Some(ReciprocalTrigNumeratorCofactor {
        builtin: cofactor.builtin,
        arg: cofactor.arg,
        display: display_expr(ctx, cofactor.expr),
        latex: latex_expr(ctx, cofactor.expr),
        polynomial: cofactor.polynomial,
    })
}

fn reciprocal_trig_derivative_product_match_from_cofactor(
    ctx: &Context,
    kind: ReciprocalTrigDerivativeProductKind,
    arg: ExprId,
    cofactor_display: String,
    cofactor_latex: String,
    cofactor_poly: Polynomial,
    var_name: &str,
) -> Option<ReciprocalTrigDerivativeProductMatch> {
    let arg_poly = polynomial_trace_arg_ignoring_independent_addends(ctx, arg, var_name)?;
    if arg_poly.degree() == 0 {
        return None;
    }
    let derivative = arg_poly.derivative();
    let scale = constant_polynomial_ratio(&cofactor_poly, &derivative)?;
    if scale.is_zero() {
        return None;
    }
    let (derivative_display, derivative_latex) = polynomial_display_and_latex(ctx, &derivative);

    Some(ReciprocalTrigDerivativeProductMatch {
        kind,
        arg,
        cofactor_display,
        cofactor_latex,
        derivative_display,
        derivative_latex,
        scale,
        symbolic_scale_display: None,
        symbolic_scale_latex: None,
    })
}

fn sqrt_chain_reciprocal_trig_derivative_product_integrand(
    ctx: &Context,
    integrand: ExprId,
    var_name: &str,
) -> Option<ReciprocalTrigDerivativeProductMatch> {
    match ctx.get(integrand) {
        Expr::Div(num, den) => {
            sqrt_chain_reciprocal_trig_derivative_product_div(ctx, *num, *den, var_name)
        }
        Expr::Mul(left, right) => {
            sqrt_chain_reciprocal_trig_derivative_product_scaled_div(ctx, *left, *right, var_name)
                .or_else(|| {
                    sqrt_chain_reciprocal_trig_derivative_product_scaled_div(
                        ctx, *right, *left, var_name,
                    )
                })
        }
        Expr::Neg(inner) => {
            sqrt_chain_reciprocal_trig_derivative_product_negated(ctx, *inner, var_name)
        }
        _ => None,
    }
}

fn sqrt_chain_reciprocal_trig_derivative_product_negated(
    ctx: &Context,
    inner: ExprId,
    var_name: &str,
) -> Option<ReciprocalTrigDerivativeProductMatch> {
    let Expr::Div(num, den) = ctx.get(inner) else {
        return None;
    };
    let (num_negative, numerator_factors) = signed_mul_factors(ctx, *num);
    sqrt_chain_reciprocal_trig_derivative_product_from_factors(
        ctx,
        !num_negative,
        &numerator_factors,
        *den,
        var_name,
    )
}

fn sqrt_chain_reciprocal_trig_derivative_product_scaled_div(
    ctx: &Context,
    scale_expr: ExprId,
    div_expr: ExprId,
    var_name: &str,
) -> Option<ReciprocalTrigDerivativeProductMatch> {
    if contains_named_var(ctx, scale_expr, var_name) {
        return None;
    }
    if as_rational_const(ctx, scale_expr, 8).is_some_and(|scale| scale.is_zero()) {
        return None;
    }
    let Expr::Div(num, den) = ctx.get(div_expr) else {
        return None;
    };

    let (scale_negative, mut numerator_factors) = signed_mul_factors(ctx, scale_expr);
    let (num_negative, num_factors) = signed_mul_factors(ctx, *num);
    numerator_factors.extend(num_factors);
    sqrt_chain_reciprocal_trig_derivative_product_from_factors(
        ctx,
        scale_negative != num_negative,
        &numerator_factors,
        *den,
        var_name,
    )
}

fn sqrt_chain_reciprocal_trig_derivative_product_div(
    ctx: &Context,
    num: ExprId,
    den: ExprId,
    var_name: &str,
) -> Option<ReciprocalTrigDerivativeProductMatch> {
    let (numerator_negative, numerator_factors) = signed_mul_factors(ctx, num);
    sqrt_chain_reciprocal_trig_derivative_product_from_factors(
        ctx,
        numerator_negative,
        &numerator_factors,
        den,
        var_name,
    )
}

fn sqrt_chain_reciprocal_trig_derivative_product_from_factors(
    ctx: &Context,
    numerator_negative: bool,
    numerator_factors: &[ExprId],
    den: ExprId,
    var_name: &str,
) -> Option<ReciprocalTrigDerivativeProductMatch> {
    let denominator_factors = collect_mul_chain_factors_readonly(ctx, den);

    sqrt_chain_reciprocal_trig_direct_product(
        ctx,
        numerator_negative,
        numerator_factors,
        &denominator_factors,
        var_name,
    )
    .or_else(|| {
        sqrt_chain_reciprocal_trig_raw_quotient(
            ctx,
            numerator_negative,
            numerator_factors,
            &denominator_factors,
            var_name,
        )
    })
}

pub(super) fn sqrt_chain_cofactor_derivative_trace(
    ctx: &Context,
    arg: ExprId,
    numerator_negative: bool,
    numerator_factors: &[ExprId],
    denominator_factors: &[ExprId],
    var_name: &str,
) -> Option<SqrtChainCofactorTrace> {
    let trace = sqrt_chain_cofactor_derivative_trace_with_symbolic_scale(
        ctx,
        arg,
        numerator_negative,
        numerator_factors,
        denominator_factors,
        var_name,
    )?;
    if trace.symbolic_scale_display.is_some() || trace.symbolic_scale_latex.is_some() {
        return None;
    }
    Some(trace)
}

pub(super) fn sqrt_chain_cofactor_derivative_trace_with_symbolic_scale(
    ctx: &Context,
    arg: ExprId,
    numerator_negative: bool,
    numerator_factors: &[ExprId],
    denominator_factors: &[ExprId],
    var_name: &str,
) -> Option<SqrtChainCofactorTrace> {
    let (radicand, derivative_sign) = sqrt_chain_arg_radicand_and_sign(ctx, arg, var_name)?;
    let radicand_poly = Polynomial::from_expr(ctx, radicand, var_name).ok()?;
    if radicand_poly.degree() != 1 {
        return None;
    }
    let radicand_derivative = scale_polynomial(&radicand_poly.derivative(), &derivative_sign);
    if radicand_derivative.is_zero() {
        return None;
    }

    let mut scratch = ctx.clone();
    let mut cofactor_numerator_factors = Vec::new();
    if numerator_negative {
        cofactor_numerator_factors
            .push(scratch.add(Expr::Number(BigRational::from_integer((-1).into()))));
    }
    cofactor_numerator_factors.extend_from_slice(numerator_factors);
    let cofactor_expr = build_quotient_from_factors(
        &mut scratch,
        &cofactor_numerator_factors,
        denominator_factors,
    );
    let cofactor_simplified = simplify_expr_in_context(&mut scratch, cofactor_expr);

    let derivative_expr = sqrt_chain_derivative_expr(&mut scratch, radicand, &radicand_derivative);
    let derivative_simplified = simplify_expr_in_context(&mut scratch, derivative_expr);
    let has_symbolic_scale = numerator_factors.iter().any(|factor| {
        !contains_named_var(ctx, *factor, var_name) && as_rational_const(ctx, *factor, 8).is_none()
    });
    let (scale, symbolic_scale_display, symbolic_scale_latex) = if has_symbolic_scale {
        if let Some((display, latex)) = sqrt_chain_symbolic_scale_trace(
            &mut scratch,
            SqrtChainSymbolicScaleTraceInput {
                numerator_negative,
                numerator_factors,
                denominator_factors,
                radicand,
                radicand_derivative: &radicand_derivative,
                derivative_simplified,
                var_name,
            },
        ) {
            (BigRational::one(), Some(display), Some(latex))
        } else {
            let ratio = scratch.add(Expr::Div(cofactor_simplified, derivative_simplified));
            let ratio_simplified = simplify_expr_in_context(&mut scratch, ratio);
            if !contains_named_var(&scratch, ratio_simplified, var_name) {
                (
                    BigRational::one(),
                    Some(display_expr(&scratch, ratio_simplified)),
                    Some(latex_expr(&scratch, ratio_simplified)),
                )
            } else {
                return None;
            }
        }
    } else {
        let ratio = scratch.add(Expr::Div(cofactor_simplified, derivative_simplified));
        let ratio_simplified = simplify_expr_in_context(&mut scratch, ratio);
        if let Some(scale) = as_rational_const(&scratch, ratio_simplified, 8) {
            (scale, None, None)
        } else if !contains_named_var(&scratch, ratio_simplified, var_name) {
            (
                BigRational::one(),
                Some(display_expr(&scratch, ratio_simplified)),
                Some(latex_expr(&scratch, ratio_simplified)),
            )
        } else {
            return None;
        }
    };
    if scale.is_zero() {
        return None;
    }

    Some(SqrtChainCofactorTrace {
        derivative_display: display_expr(&scratch, derivative_expr),
        derivative_latex: latex_expr(&scratch, derivative_expr),
        scale,
        cofactor_display: display_expr(&scratch, cofactor_simplified),
        cofactor_latex: latex_expr(&scratch, cofactor_simplified),
        symbolic_scale_display,
        symbolic_scale_latex,
    })
}

fn sqrt_chain_derivative_expr(
    ctx: &mut Context,
    radicand: ExprId,
    radicand_derivative: &Polynomial,
) -> ExprId {
    let derivative = radicand_derivative.to_expr(ctx);
    let two = ctx.add(Expr::Number(BigRational::from_integer(2.into())));
    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let denominator = ctx.add(Expr::Mul(two, sqrt_radicand));
    ctx.add(Expr::Div(derivative, denominator))
}

pub(super) fn nested_trig_log_derivative_substitution_substeps(
    ctx: &Context,
    integrand: ExprId,
    after: ExprId,
    var_name: &str,
) -> Option<Vec<SubStep>> {
    let plan = nested_trig_log_derivative_substitution_plan(ctx, integrand, var_name)?;
    Some(vec![
        SubStep::new(
            "Usar la regla de u'/u -> ln|u|",
            display_expr(ctx, integrand),
            display_expr(ctx, after),
        )
        .with_before_latex(latex_expr(ctx, integrand))
        .with_after_latex(latex_expr(ctx, after)),
        SubStep::keyed(
            "usub.identify_u_du",
            vec![],
            format!("u = {}", display_expr(ctx, plan.log_expr)),
            format!("du = {} dx", plan.derivative_display),
        )
        .with_before_latex(format!("u = {}", latex_expr(ctx, plan.log_expr)))
        .with_after_latex(format!("du = {}\\,dx", plan.derivative_latex)),
    ])
}

fn nested_trig_log_derivative_substitution_plan(
    ctx: &Context,
    integrand: ExprId,
    var_name: &str,
) -> Option<NestedTrigLogDerivativeSubstitutionPlan> {
    let Expr::Div(_, den) = ctx.get(integrand) else {
        return None;
    };
    let denominator_factors = cas_math::expr_nary::mul_factors(ctx, *den);
    let (log_expr, arg) = denominator_factors
        .iter()
        .find_map(|factor| nested_trig_log_factor_arg(ctx, *factor))?;
    if !contains_named_var(ctx, arg, var_name) {
        return None;
    }

    let has_sin = denominator_factors.iter().any(|factor| {
        matches!(
            unary_builtin_arg(ctx, *factor),
            Some((BuiltinFn::Sin, factor_arg))
                if compare_expr(ctx, factor_arg, arg) == Ordering::Equal
        )
    });
    let has_cos = denominator_factors.iter().any(|factor| {
        matches!(
            unary_builtin_arg(ctx, *factor),
            Some((BuiltinFn::Cos, factor_arg))
                if compare_expr(ctx, factor_arg, arg) == Ordering::Equal
        )
    });
    if !has_sin || !has_cos {
        return None;
    }

    let mut scratch = ctx.clone();
    let derivative = cas_math::symbolic_differentiation_support::differentiate_symbolic_expr(
        &mut scratch,
        log_expr,
        var_name,
    )?;
    let derivative = simplify_expr_in_context(&mut scratch, derivative);
    Some(NestedTrigLogDerivativeSubstitutionPlan {
        log_expr,
        derivative_display: display_expr(&scratch, derivative),
        derivative_latex: latex_expr(&scratch, derivative),
    })
}

pub(super) fn generate_integral_residual_policy_substeps(
    ctx: &Context,
    step: &Step,
) -> Vec<SubStep> {
    if step.rule_name != RULE_CONSERVAR_INTEGRAL_RESIDUAL
        && step.description != RULE_CONSERVAR_INTEGRAL_RESIDUAL
    {
        return Vec::new();
    }

    let mut substeps = Vec::new();
    collect_integral_residual_policy_substeps(ctx, step.before, &mut substeps);
    collect_integral_residual_required_condition_substeps(ctx, step, &mut substeps);
    substeps
}

fn collect_integral_residual_policy_substeps(
    ctx: &Context,
    expr: ExprId,
    substeps: &mut Vec<SubStep>,
) {
    if substeps.len() >= 2 {
        return;
    }

    if let Some(substep) = integral_residual_policy_substep_for_expr(ctx, expr) {
        push_unique_integral_residual_substep(substeps, substep);
    }

    match ctx.get(expr) {
        Expr::Add(left, right)
        | Expr::Sub(left, right)
        | Expr::Mul(left, right)
        | Expr::Div(left, right)
        | Expr::Pow(left, right) => {
            collect_integral_residual_policy_substeps(ctx, *left, substeps);
            collect_integral_residual_policy_substeps(ctx, *right, substeps);
        }
        Expr::Neg(inner) | Expr::Hold(inner) => {
            collect_integral_residual_policy_substeps(ctx, *inner, substeps);
        }
        Expr::Function(_, args) => {
            for arg in args {
                collect_integral_residual_policy_substeps(ctx, *arg, substeps);
                if substeps.len() >= 2 {
                    break;
                }
            }
        }
        Expr::Matrix { data, .. } => {
            for child in data {
                collect_integral_residual_policy_substeps(ctx, *child, substeps);
                if substeps.len() >= 2 {
                    break;
                }
            }
        }
        Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => {}
    }
}

fn collect_integral_residual_required_condition_substeps(
    ctx: &Context,
    step: &Step,
    substeps: &mut Vec<SubStep>,
) {
    for condition in step.required_conditions() {
        if substeps.len() >= 3 {
            break;
        }

        if let Some(substep) = integral_residual_required_condition_substep(ctx, condition) {
            push_unique_integral_residual_substep(substeps, substep);
        }
    }
}

fn push_unique_integral_residual_substep(substeps: &mut Vec<SubStep>, substep: SubStep) {
    let after_latex = substep.after_latex.clone();
    let already_present = substeps
        .iter()
        .any(|existing| existing.after_latex == after_latex);
    if !already_present {
        substeps.push(substep);
    }
}

fn integral_residual_required_condition_substep(
    ctx: &Context,
    condition: &ImplicitCondition,
) -> Option<SubStep> {
    let (title, witness, after_latex) = match condition {
        // Branch annotations belong to the complex frontier, not to the
        // real-domain integral residual narration.
        ImplicitCondition::PrincipalBranch { .. } => return None,
        ImplicitCondition::NonNegative(expr) => (
            "Registrar dominio real del residual",
            *expr,
            format!("{} \\ge 0", latex_expr(ctx, *expr)),
        ),
        ImplicitCondition::LowerBound(expr, lower) => (
            "Registrar dominio real del residual",
            *expr,
            format!("{} \\ge {}", latex_expr(ctx, *expr), rational_latex(lower)),
        ),
        ImplicitCondition::Positive(expr) => (
            "Registrar condición de dominio del residual",
            *expr,
            format!("{} > 0", latex_expr(ctx, *expr)),
        ),
        ImplicitCondition::NonZero(expr) => (
            "Registrar condición no nula del residual",
            *expr,
            format!("{} \\ne 0", latex_expr(ctx, *expr)),
        ),
    };

    Some(
        SubStep::new(title, display_expr(ctx, witness), condition.display(ctx))
            .with_before_latex(latex_expr(ctx, witness))
            .with_after_latex(after_latex),
    )
}

fn integral_residual_policy_substep_for_expr(ctx: &Context, expr: ExprId) -> Option<SubStep> {
    let (builtin, arg) = unary_builtin_arg(ctx, expr)?;
    let arg_display = display_expr(ctx, arg);
    let arg_latex = latex_expr(ctx, arg);

    match builtin {
        BuiltinFn::Tan | BuiltinFn::Sec => Some(
            SubStep::new(
                "Registrar polo del integrando",
                display_expr(ctx, expr),
                format!("cos({arg_display}) ≠ 0"),
            )
            .with_before_latex(latex_expr(ctx, expr))
            .with_after_latex(format!("\\cos({arg_latex}) \\ne 0")),
        ),
        BuiltinFn::Cot | BuiltinFn::Csc => Some(
            SubStep::new(
                "Registrar polo del integrando",
                display_expr(ctx, expr),
                format!("sin({arg_display}) ≠ 0"),
            )
            .with_before_latex(latex_expr(ctx, expr))
            .with_after_latex(format!("\\sin({arg_latex}) \\ne 0")),
        ),
        BuiltinFn::Ln | BuiltinFn::Log | BuiltinFn::Log2 | BuiltinFn::Log10 => Some(
            SubStep::new(
                "Registrar dominio del logaritmo",
                display_expr(ctx, expr),
                format!("{arg_display} > 0"),
            )
            .with_before_latex(latex_expr(ctx, expr))
            .with_after_latex(format!("{arg_latex} > 0")),
        ),
        _ => None,
    }
}

/// Narración rica de la vía u-du SIMBÓLICA
/// (`symbolic_power_substitution_from_base` en cas_math): integrando
/// `s·u'·uⁿ` con `u` NO polinómica (`∫cos·(sin+1)² = (sin+1)³/3`).
///
/// Doble cerrojo para no robar narraciones ya fijadas: (a) la base NO baja a
/// `Polynomial` (las polinómicas conservan sus narradores), y (b) el `after`
/// lleva la huella exacta de la ruta — `c·uⁿ⁺¹` sobre la MISMA base. Si
/// cualquiera falla, traza vacía y el siguiente narrador decide.
pub(super) fn generate_symbolic_power_substitution_substeps(
    ctx: &Context,
    step: &Step,
) -> Vec<SubStep> {
    if step.rule_name != "Symbolic Integration" {
        return Vec::new();
    }
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Expr::Function(fn_id, args) = ctx.get(before) else {
        return Vec::new();
    };
    if ctx.sym_name(*fn_id) != "integrate" || args.len() != 2 {
        return Vec::new();
    }
    if matches!(
        ctx.get(after),
        Expr::Function(after_fn, after_args)
            if ctx.sym_name(*after_fn) == "integrate" && after_args.len() == 2
    ) {
        return Vec::new();
    }
    let Expr::Variable(var_sym) = ctx.get(args[1]) else {
        return Vec::new();
    };
    let var_name = ctx.sym_name(*var_sym).to_string();

    // (base, n) del integrando: Mul con Pow(base, n racional) — incluido n<0 —
    // o Div cuyo denominador es Pow(base, m>0) o la base directa (n = −1).
    // Base con la variable, NO polinómica y, en el brazo Div, no una función
    // desnuda (esas tienen dueños con narración propia).
    let mut found: Option<(ExprId, num_rational::BigRational)> = None;
    let factors = cas_math::expr_nary::mul_leaves(ctx, args[0]);
    if factors.len() >= 2 {
        for factor in &factors {
            let Expr::Pow(base, exp) = ctx.get(*factor) else {
                continue;
            };
            let Some(exponent) = cas_ast::views::as_rational_const(ctx, *exp, 8) else {
                continue;
            };
            if !contains_named_var(ctx, *base, &var_name) {
                continue;
            }
            if Polynomial::from_expr(ctx, *base, &var_name).is_ok() {
                continue;
            }
            found = Some((*base, exponent));
            break;
        }
    }
    if found.is_none() {
        if let Expr::Div(_, den) = ctx.get(args[0]) {
            let (cand, m) = match ctx.get(*den) {
                Expr::Pow(b, e) => match cas_ast::views::as_rational_const(ctx, *e, 8) {
                    Some(m) if m > num_rational::BigRational::from_integer(0.into()) => (*b, m),
                    _ => (*den, num_rational::BigRational::from_integer(1.into())),
                },
                _ => (*den, num_rational::BigRational::from_integer(1.into())),
            };
            if !matches!(ctx.get(cand), Expr::Function(_, _))
                && contains_named_var(ctx, cand, &var_name)
                && Polynomial::from_expr(ctx, cand, &var_name).is_err()
            {
                found = Some((cand, -m));
            }
        }
    }
    let Some((base, exponent)) = found else {
        return Vec::new();
    };

    // Huella del after sobre la MISMA base: `c·base^(n+1)` (n≠−1, con la
    // potencia en factor directo o en denominador recíproco) o `c·ln(|base|)`
    // para n=−1. Sin huella no se narra: es lo que garantiza que el paso lo
    // produjo esta ruta y no un dueño ajeno.
    let minus_one = num_rational::BigRational::from_integer((-1).into());
    let is_log_case = exponent == minus_one;
    let expected_exp = exponent + num_rational::BigRational::from_integer(1.into());

    fn strip_neg(ctx: &Context, e: ExprId) -> ExprId {
        match ctx.get(e) {
            Expr::Neg(i) => *i,
            _ => e,
        }
    }
    fn has_power_of(
        ctx: &Context,
        e: ExprId,
        base: ExprId,
        wanted: &num_rational::BigRational,
    ) -> bool {
        for factor in cas_math::expr_nary::mul_leaves(ctx, e) {
            if wanted == &num_rational::BigRational::from_integer(1.into())
                && cas_ast::ordering::compare_expr(ctx, factor, base) == std::cmp::Ordering::Equal
            {
                return true;
            }
            if let Expr::Pow(b, x) = ctx.get(factor) {
                if let Some(v) = cas_ast::views::as_rational_const(ctx, *x, 8) {
                    if &v == wanted
                        && cas_ast::ordering::compare_expr(ctx, *b, base)
                            == std::cmp::Ordering::Equal
                    {
                        return true;
                    }
                }
            }
        }
        false
    }

    let core = strip_neg(ctx, after);
    let fingerprint = if is_log_case {
        // c·ln(base) o c·ln(|base|)
        cas_math::expr_nary::mul_leaves(ctx, core)
            .into_iter()
            .any(|f| {
                let Expr::Function(fn_id2, ln_args) = ctx.get(f) else {
                    return false;
                };
                if ctx.sym_name(*fn_id2) != "ln" || ln_args.len() != 1 {
                    return false;
                }
                let mut arg = ln_args[0];
                if let Expr::Function(abs_id, abs_args) = ctx.get(arg) {
                    if ctx.sym_name(*abs_id) == "abs" && abs_args.len() == 1 {
                        arg = abs_args[0];
                    }
                }
                cas_ast::ordering::compare_expr(ctx, arg, base) == std::cmp::Ordering::Equal
            })
    } else if expected_exp > num_rational::BigRational::from_integer(0.into()) {
        has_power_of(ctx, core, base, &expected_exp)
    } else {
        // negativo: potencia en factor directo (Pow con exponente negativo) o
        // en el denominador de un Div
        has_power_of(ctx, core, base, &expected_exp)
            || match ctx.get(core) {
                Expr::Div(_, d) => has_power_of(ctx, *d, base, &(-expected_exp.clone())),
                _ => false,
            }
    };
    if !fingerprint {
        return Vec::new();
    }

    // u y du, con la derivada plegada a la forma que escribiría el estudiante.
    let mut scratch = ctx.clone();
    let Some(derivative) = cas_math::symbolic_differentiation_support::differentiate_symbolic_expr(
        &mut scratch,
        base,
        &var_name,
    ) else {
        return Vec::new();
    };
    let derivative = simplify_expr_in_context(&mut scratch, derivative);

    vec![
        SubStep::keyed(
            "usub.identify_u_du",
            vec![],
            format!("u = {}", display_expr(ctx, base)),
            format!("du = {} dx", display_expr(&scratch, derivative)),
        )
        .with_before_latex(format!("u = {}", latex_expr(ctx, base)))
        .with_after_latex(format!("du = {}\\,dx", latex_expr(&scratch, derivative))),
        if is_log_case {
            SubStep::keyed(
                "usub.rule_ln_abs_inner_derivative",
                vec![],
                display_expr(ctx, args[0]),
                display_expr(ctx, after),
            )
            .with_before_latex(latex_expr(ctx, args[0]))
            .with_after_latex(latex_expr(ctx, after))
        } else {
            SubStep::new(
                "Usar regla de potencia para integrales",
                display_expr(ctx, args[0]),
                display_expr(ctx, after),
            )
            .with_before_latex(latex_expr(ctx, args[0]))
            .with_after_latex(latex_expr(ctx, after))
        },
    ]
}

/// Narración de la TABLA u-du simbólica (`symbolic_derivative_table_antiderivative`):
/// integrando `s·u′·F(u)` con F ∈ {exp, sin, cos, sinh, cosh} y `u` compuesta
/// no polinómica (`∫cos(x)·cos(sin(x)) = sin(sin(x))`). Mismos cerrojos que la
/// ruta y huella del after (`c·G(u)`, con Neg opcional) para no narrar pasos
/// de dueños ajenos. Los kernels POLINÓMICOS conservan su narrador de tabla.
pub(super) fn generate_symbolic_table_substitution_substeps(
    ctx: &Context,
    step: &Step,
) -> Vec<SubStep> {
    if step.rule_name != "Symbolic Integration" {
        return Vec::new();
    }
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Expr::Function(fn_id, args) = ctx.get(before) else {
        return Vec::new();
    };
    if ctx.sym_name(*fn_id) != "integrate" || args.len() != 2 {
        return Vec::new();
    }
    if matches!(
        ctx.get(after),
        Expr::Function(after_fn, after_args)
            if ctx.sym_name(*after_fn) == "integrate" && after_args.len() == 2
    ) {
        return Vec::new();
    }
    let Expr::Variable(var_sym) = ctx.get(args[1]) else {
        return Vec::new();
    };
    let var_name = ctx.sym_name(*var_sym).to_string();

    let table = |b: BuiltinFn| -> Option<(&'static str, BuiltinFn, bool)> {
        match b {
            BuiltinFn::Exp => Some(("Usar la regla de exp(u) -> exp(u)", BuiltinFn::Exp, false)),
            BuiltinFn::Cos => Some(("Usar la regla de cos(u) -> sin(u)", BuiltinFn::Sin, false)),
            BuiltinFn::Sin => Some(("Usar la regla de sin(u) -> -cos(u)", BuiltinFn::Cos, true)),
            BuiltinFn::Sinh => Some((
                "Usar la regla de sinh(u) -> cosh(u)",
                BuiltinFn::Cosh,
                false,
            )),
            BuiltinFn::Cosh => Some((
                "Usar la regla de cosh(u) -> sinh(u)",
                BuiltinFn::Sinh,
                false,
            )),
            _ => None,
        }
    };

    let mut found: Option<(ExprId, &'static str, BuiltinFn)> = None;
    for factor in cas_math::expr_nary::mul_leaves(ctx, args[0]) {
        // exp(u) canonizado como Pow(E, u) cuenta como exterior Exp
        let outer_inner: Option<(BuiltinFn, ExprId)> = match ctx.get(factor) {
            Expr::Function(f_id, f_args) if f_args.len() == 1 => {
                ctx.builtin_of(*f_id).map(|b| (b, f_args[0]))
            }
            Expr::Pow(base, exp)
                if matches!(ctx.get(*base), Expr::Constant(cas_ast::Constant::E)) =>
            {
                Some((BuiltinFn::Exp, *exp))
            }
            _ => None,
        };
        let Some((builtin, inner)) = outer_inner else {
            continue;
        };
        let Some((title, anti, _)) = table(builtin) else {
            continue;
        };
        if !contains_named_var(ctx, inner, &var_name) {
            continue;
        }
        if Polynomial::from_expr(ctx, inner, &var_name).is_ok() {
            continue;
        }
        found = Some((inner, title, anti));
        break;
    }
    let Some((inner, title, anti_builtin)) = found else {
        return Vec::new();
    };

    // Huella: el after contiene G(inner) (bajo Neg/coeficiente opcionales).
    let core = match ctx.get(after) {
        Expr::Neg(i) => *i,
        _ => after,
    };
    let fingerprint = cas_math::expr_nary::mul_leaves(ctx, core)
        .into_iter()
        .any(|f| {
            // el motor canoniza exp(u) como Pow(E, u): las dos formas cuentan
            if anti_builtin == BuiltinFn::Exp {
                if let Expr::Pow(base, exp) = ctx.get(f) {
                    if matches!(ctx.get(*base), Expr::Constant(cas_ast::Constant::E))
                        && cas_ast::ordering::compare_expr(ctx, *exp, inner)
                            == std::cmp::Ordering::Equal
                    {
                        return true;
                    }
                }
            }
            matches!(
                ctx.get(f),
                Expr::Function(g_id, g_args)
                    if g_args.len() == 1
                        && ctx.builtin_of(*g_id) == Some(anti_builtin)
                        && cas_ast::ordering::compare_expr(ctx, g_args[0], inner)
                            == std::cmp::Ordering::Equal
            )
        });
    if !fingerprint {
        return Vec::new();
    }

    let mut scratch = ctx.clone();
    let Some(derivative) = cas_math::symbolic_differentiation_support::differentiate_symbolic_expr(
        &mut scratch,
        inner,
        &var_name,
    ) else {
        return Vec::new();
    };
    let derivative = simplify_expr_in_context(&mut scratch, derivative);

    vec![
        SubStep::keyed(
            "usub.identify_u_du",
            vec![],
            format!("u = {}", display_expr(ctx, inner)),
            format!("du = {} dx", display_expr(&scratch, derivative)),
        )
        .with_before_latex(format!("u = {}", latex_expr(ctx, inner)))
        .with_after_latex(format!("du = {}\\,dx", latex_expr(&scratch, derivative))),
        SubStep::new(title, display_expr(ctx, args[0]), display_expr(ctx, after))
            .with_before_latex(latex_expr(ctx, args[0]))
            .with_after_latex(latex_expr(ctx, after)),
    ]
}
