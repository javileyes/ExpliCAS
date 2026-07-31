//! `focused_rule_substeps`: familia `support`.
//!
//! Ver la cabecera de `focused_rule_substeps.rs` para el contexto.

use super::*;

pub(super) fn collect_mul_chain_factors_readonly(ctx: &Context, expr: ExprId) -> Vec<ExprId> {
    let mut out = Vec::new();
    collect_mul_chain_factors_readonly_into(ctx, expr, &mut out);
    out
}

pub(super) fn simplify_expr_in_context(ctx: &mut Context, expr: ExprId) -> ExprId {
    let mut simplifier = cas_solver::runtime::Simplifier::with_default_rules();
    std::mem::swap(&mut simplifier.context, ctx);
    let (rewritten, _steps, _stats) = cas_engine::with_suppressed_depth_overflow_warnings(|| {
        simplifier.simplify_with_stats(expr, cas_solver::runtime::SimplifyOptions::default())
    });
    std::mem::swap(&mut simplifier.context, ctx);
    rewritten
}

pub(super) fn contains_named_var(ctx: &Context, expr: ExprId, var: &str) -> bool {
    let mut stack = vec![expr];
    while let Some(current) = stack.pop() {
        match ctx.get(current) {
            Expr::Variable(sym_id) if ctx.sym_name(*sym_id) == var => return true,
            Expr::Add(left, right)
            | Expr::Sub(left, right)
            | Expr::Mul(left, right)
            | Expr::Div(left, right)
            | Expr::Pow(left, right) => {
                stack.push(*left);
                stack.push(*right);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(*inner),
            Expr::Function(_, args) => stack.extend(args.iter().copied()),
            Expr::Matrix { data, .. } => stack.extend(data.iter().copied()),
            Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => {}
        }
    }
    false
}

pub(super) fn is_named_var(ctx: &Context, expr: ExprId, var: &str) -> bool {
    matches!(ctx.get(expr), Expr::Variable(sym_id) if ctx.sym_name(*sym_id) == var)
}

pub(super) fn human_formula_title_plain(plain: &str) -> String {
    plain.replace(" * ", "·").replace('*', "·")
}

pub(super) fn temp_ctx_substep(
    title: impl Into<String>,
    ctx: &Context,
    before: ExprId,
    after: ExprId,
) -> SubStep {
    SubStep::new(title, human_expr(ctx, before), human_expr(ctx, after))
        .with_before_latex(latex_expr(ctx, before))
        .with_after_latex(latex_expr(ctx, after))
}

pub(super) fn render_temp_expr(ctx: &Context, expr: ExprId) -> (String, String) {
    (
        cas_formatter::clean_display_string(&format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: ctx,
                id: expr
            }
        )),
        cas_formatter::LaTeXExpr {
            context: ctx,
            id: expr,
        }
        .to_latex(),
    )
}

/// Emit a sub-step whose two sides are a SCHEMA: an identity written with
/// metavariables (`sin²(u)·cos²(u) = sin²(2u)/4`), not the expression in front
/// of the reader.
///
/// C1.8: the pair is DECLARED as `SchematicIdentity` and adjudicated by the
/// census in `substep::schema`, which proves the 133 provable templates once in
/// a test instead of re-deciding each of them on every emission. Publishing is
/// unaffected — the classified exceptions are declared gaps, and each has its
/// own owner — but an emitter can no longer state a schema that nothing has
/// ever looked at.
/// Emit a sub-step from four INDEPENDENT rendered strings. This is the
/// instance emitter: its sides are built per-emission from the reader's
/// expression, so no static census can adjudicate them. The template twin is
/// [`schema_substep`], whose `&'static str` sides are what makes the schema
/// census possible — the COMPILER partitions the two families, not a heuristic.
pub(super) fn formula_substep(
    description: impl Into<String>,
    before_expr: &str,
    after_expr: &str,
    before_latex: &str,
    after_latex: &str,
) -> SubStep {
    SubStep::new(description, before_expr, after_expr)
        .with_before_latex(before_latex)
        .with_after_latex(after_latex)
}

/// Emit a sub-step whose two sides are a SCHEMA: an identity written with
/// metavariables (`sin²(u)·cos²(u) = sin²(2u)/4`), not the reader's
/// expression.
///
/// C1.8 (`SchematicIdentity`): the sides are `&'static str` ON PURPOSE. A
/// template is a constant of the source; requiring `'static` makes the type
/// system separate template sites from instance sites — the earlier textual
/// detector confused a user symbol named `a` with a metavariable, and no
/// string heuristic can tell them apart. The pair is declared as a claim and
/// adjudicated by the census in `substep::schema`, which proves every provable
/// template ONCE in a test instead of once per emission.
/// Emit a «Usar LHS = RHS» sub-step over a CONCRETE pair, publishing ONLY if
/// the pair is an instance of the template (C1.8, instance↔template matcher).
///
/// This is the constructor that kills the audit's named lies by construction:
/// «Usar tan(u)·cot(u) = 1» emitted unconditionally, and the half-angle branch
/// that recognized no variant yet still cited the identity. The template's own
/// truth is the census's job; whether it APPLIES to this pair is decided here,
/// and a pair that instantiates nothing publishes nothing.
///
/// Migration policy: emitter by emitter, each with a decline test — the
/// matcher is incomplete by design, and a blanket sweep would delete correct
/// narration wherever a true application outruns the matcher's coverage
/// (measured precedent: the assume-equality prototype deleted 51 legitimate
/// sub-steps).
/// Structural-FIRST selection over a table of equivalent templates: pass 1
/// accepts only structural instances (the cited formula is the one on
/// screen), pass 2 falls back to the directed mode (for instances whose shape
/// folded away). Without the split, the directed mode picks whichever
/// equivalent template comes first — measured: `2·cos(u)²−1` narrated as
/// «1 − 2·sin(u)² = cos(2u)», true by Pythagoras and useless to the reader.
pub(super) fn named_identity_from_table(
    ctx: &Context,
    templates: &[(&'static str, &'static str)],
    before: ExprId,
    after: ExprId,
) -> Option<SubStep> {
    for (lhs, rhs) in templates {
        let _ = crate::didactic::substep::claim::verify_schematic_identity(lhs, rhs);
        let Some(template) = crate::didactic::substep::matching::parse_template(lhs, rhs) else {
            continue;
        };
        let structural = crate::didactic::substep::matching::match_instance_structural(
            &template, ctx, before, after,
        )
        .or_else(|| {
            crate::didactic::substep::matching::match_instance_structural(
                &template, ctx, after, before,
            )
        });
        if structural.is_some() {
            return Some(concrete_expr_substep(
                ctx,
                format!("Usar {lhs} = {rhs}"),
                before,
                after,
            ));
        }
    }
    for (lhs, rhs) in templates {
        if let Some(substep) = named_identity_substep(ctx, lhs, rhs, before, after) {
            return Some(substep);
        }
    }
    // Pass 3 — SCALED instances: the pair applies the identity inside a
    // linear combination, both sides carrying the same numeric factor
    // (`4·cos(u)² − 2 ⟹ 2·cos(2u)`). Peel and match structurally.
    for (lhs, rhs) in templates {
        let _ = crate::didactic::substep::claim::verify_schematic_identity(lhs, rhs);
        let Some(template) = crate::didactic::substep::matching::parse_template(lhs, rhs) else {
            continue;
        };
        let scaled = crate::didactic::substep::matching::match_instance_scaled(
            &template, ctx, before, after,
        )
        .or_else(|| {
            crate::didactic::substep::matching::match_instance_scaled(&template, ctx, after, before)
        });
        if scaled.is_some() {
            return Some(concrete_expr_substep(
                ctx,
                format!("Usar {lhs} = {rhs}"),
                before,
                after,
            ));
        }
    }
    None
}

pub(super) fn named_identity_substep(
    ctx: &Context,
    lhs: &'static str,
    rhs: &'static str,
    before: ExprId,
    after: ExprId,
) -> Option<SubStep> {
    // The template itself must be adjudicated by the census (debug-asserts on
    // a pair nothing has measured).
    let _ = crate::didactic::substep::claim::verify_schematic_identity(lhs, rhs);
    let template = crate::didactic::substep::matching::parse_template(lhs, rhs)?;
    // «Usar L = R» names an IDENTITY, which has no direction: the pair may
    // apply it left-to-right (contraction) or right-to-left (expansion), and
    // the half-angle emitter genuinely produces both. Either orientation is a
    // valid application; neither is a licence for a non-instance.
    crate::didactic::substep::matching::match_instance(&template, ctx, before, after).or_else(
        || crate::didactic::substep::matching::match_instance(&template, ctx, after, before),
    )?;
    Some(concrete_expr_substep(
        ctx,
        format!("Usar {lhs} = {rhs}"),
        before,
        after,
    ))
}

pub(super) fn schema_substep(
    description: impl Into<String>,
    lhs: &'static str,
    rhs: &'static str,
    lhs_latex: &'static str,
    rhs_latex: &'static str,
) -> SubStep {
    let (substep, _verdict) = SubStep::checked_schema(lhs, rhs, description);
    substep
        .with_before_latex(lhs_latex)
        .with_after_latex(rhs_latex)
}

pub(super) fn concrete_expr_substep(
    ctx: &Context,
    description: impl Into<String>,
    before: ExprId,
    after: ExprId,
) -> SubStep {
    SubStep::new(
        description,
        display_expr(ctx, before),
        display_expr(ctx, after),
    )
    .with_before_latex(latex_expr(ctx, before))
    .with_after_latex(latex_expr(ctx, after))
}

pub(super) fn is_small_positive_integer(ctx: &Context, expr: ExprId, value: i64) -> bool {
    matches!(ctx.get(expr), Expr::Number(n) if n.is_integer() && n.to_integer() == value.into())
}

/// A table-rule integration sub-step ASSERTS that `after` is an antiderivative
/// of the integrand — the single most repeated shape in this file. Declared and
/// checked (C1.8): a refuted pair is not published, an undecided one is (a surd
/// the simplifier cannot fold is not evidence of a lie).
pub(super) fn checked_antiderivative_substep(
    ctx: &Context,
    title: &str,
    integrand: ExprId,
    after: ExprId,
    var_name: &str,
) -> Option<SubStep> {
    SubStep::checked_new(
        ctx,
        crate::didactic::substep::Claim::Antiderivative {
            var: var_name.to_string(),
        },
        integrand,
        after,
        title,
        display_expr(ctx, integrand),
        display_expr(ctx, after),
    )
    .map(|step| {
        step.with_before_latex(latex_expr(ctx, integrand))
            .with_after_latex(latex_expr(ctx, after))
    })
}

pub(super) fn expr_contains_integrate_call(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Function(fn_id, args) => {
            ctx.sym_name(*fn_id) == "integrate"
                || args
                    .iter()
                    .any(|arg| expr_contains_integrate_call(ctx, *arg))
        }
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) | Expr::Pow(l, r) => {
            expr_contains_integrate_call(ctx, *l) || expr_contains_integrate_call(ctx, *r)
        }
        Expr::Neg(inner) | Expr::Hold(inner) => expr_contains_integrate_call(ctx, *inner),
        _ => false,
    }
}

pub(super) fn constant_polynomial_ratio(
    numerator: &Polynomial,
    denominator: &Polynomial,
) -> Option<BigRational> {
    if denominator.is_zero() {
        return None;
    }

    let pivot = denominator
        .coeffs
        .iter()
        .position(|coeff| !coeff.is_zero())?;
    let numerator_pivot = numerator
        .coeffs
        .get(pivot)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let scale = numerator_pivot / denominator.coeffs[pivot].clone();
    let len = numerator.coeffs.len().max(denominator.coeffs.len());

    for idx in 0..len {
        let left = numerator
            .coeffs
            .get(idx)
            .cloned()
            .unwrap_or_else(BigRational::zero);
        let right = denominator
            .coeffs
            .get(idx)
            .cloned()
            .unwrap_or_else(BigRational::zero)
            * scale.clone();
        if left != right {
            return None;
        }
    }

    Some(scale)
}

pub(super) fn polynomial_derivative_cofactor_trace(
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
    let cofactor_poly = Polynomial::from_expr(&scratch, cofactor_simplified, var_name).ok()?;
    let scale = constant_polynomial_ratio(&cofactor_poly, &derivative_poly)?;
    if scale.is_zero() {
        return None;
    }

    let (derivative_display, derivative_latex) =
        polynomial_display_and_latex(ctx, &derivative_poly);
    Some(PolynomialDerivativeCofactorTrace {
        derivative_display,
        derivative_latex,
        scale,
        cofactor_display: display_expr(&scratch, cofactor_simplified),
        cofactor_latex: latex_expr(&scratch, cofactor_simplified),
        symbolic_scale_display: None,
        symbolic_scale_latex: None,
    })
}

pub(super) fn rational_display(value: &BigRational) -> String {
    if value.denom().is_one() {
        value.numer().to_string()
    } else {
        format!("{}/{}", value.numer(), value.denom())
    }
}

pub(super) fn rational_latex(value: &BigRational) -> String {
    if value.denom().is_one() {
        value.numer().to_string()
    } else {
        format!("\\frac{{{}}}{{{}}}", value.numer(), value.denom())
    }
}

pub(super) fn unary_builtin_arg(ctx: &Context, expr: ExprId) -> Option<(BuiltinFn, ExprId)> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }
    Some((ctx.builtin_of(*fn_id)?, args[0]))
}

pub(super) fn signed_mul_factors(ctx: &Context, expr: ExprId) -> (bool, Vec<ExprId>) {
    let mut factors = Vec::new();
    let mut negative = false;
    signed_mul_factors_into(ctx, expr, &mut negative, &mut factors);
    (negative, factors)
}

pub(super) fn display_expr(ctx: &Context, expr: ExprId) -> String {
    format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: ctx,
            id: expr,
        }
    )
}

pub(super) fn latex_expr(ctx: &Context, expr: ExprId) -> String {
    cas_formatter::LaTeXExpr {
        context: ctx,
        id: expr,
    }
    .to_latex()
}

pub(super) fn human_expr(ctx: &Context, expr: ExprId) -> String {
    cas_formatter::clean_display_string(&crate::didactic::latex_to_plain_text(&latex_expr(
        ctx, expr,
    )))
}

pub(super) fn build_quotient_from_factors(
    ctx: &mut Context,
    numerator_factors: &[ExprId],
    denominator_factors: &[ExprId],
) -> ExprId {
    let numerator = build_mul_expr_from_factors(ctx, numerator_factors);
    let denominator = build_mul_expr_from_factors(ctx, denominator_factors);

    if is_one_expr(ctx, denominator) {
        numerator
    } else {
        ctx.add(Expr::Div(numerator, denominator))
    }
}

pub(super) fn build_mul_expr_from_factors(ctx: &mut Context, factors: &[ExprId]) -> ExprId {
    match factors {
        [] => ctx.add(Expr::Number(BigRational::from_integer(1.into()))),
        [only] => *only,
        _ => {
            let mut iter = factors.iter().copied();
            let first = iter.next().expect("non-empty factors");
            iter.fold(first, |acc, factor| ctx.add(Expr::Mul(acc, factor)))
        }
    }
}

pub(super) fn same_presentational_expr(
    left_ctx: &Context,
    left_expr: ExprId,
    right_ctx: &Context,
    right_expr: ExprId,
) -> bool {
    display_expr(left_ctx, left_expr) == display_expr(right_ctx, right_expr)
}

pub(super) fn is_one(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Number(value) if value.numer() == &1.into() && value.denom() == &1.into())
}

pub(super) fn is_zero(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Number(value) if value.numer() == &0.into() && value.denom() == &1.into())
}

pub(super) fn is_integer_literal(ctx: &Context, expr: ExprId, expected: i64) -> bool {
    matches!(
        ctx.get(expr),
        Expr::Number(value) if value.numer() == &expected.into() && value.denom() == &1.into()
    )
}
