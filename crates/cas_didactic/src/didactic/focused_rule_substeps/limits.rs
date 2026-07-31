//! `focused_rule_substeps`: familia `limits`.
//!
//! Ver la cabecera de `focused_rule_substeps.rs` para el contexto.

use super::*;

/// Path-counterexample narration for the multivariate-limit DNE verdict (F8,
/// Fase 3): the witnesses travel in the step's assumption event.
pub(super) fn generate_limit_path_counterexample_substeps(
    ctx: &Context,
    step: &Step,
) -> Vec<SubStep> {
    if !matches!(
        step.rule_name.as_str(),
        "Multivariate Limit" | "Evaluar el límite multivariable por continuidad"
    ) {
        return Vec::new();
    }
    let after = step.after_local().unwrap_or(step.after);
    if !matches!(ctx.get(after), Expr::Constant(cas_ast::Constant::Undefined)) {
        return Vec::new();
    }
    step.assumption_events()
        .iter()
        .filter(|event| event.message.contains("no existe"))
        .map(|event| {
            SubStep::keyed(
                "limit.path_counterexample",
                vec![event.message.clone()],
                display_expr(ctx, step.before_local().unwrap_or(step.before)),
                display_expr(ctx, after),
            )
        })
        .collect()
}

/// The approach a limit step is about, sign included.
///
/// Read from the step's metadata, which the eval path now records. The rule
/// name only says "en infinito" — it cannot tell `+∞` from `−∞`, and reading
/// the direction off that substring is what let the `−∞` narration cite the
/// `x→+∞` theorem. The fallback keeps a producer that forgets the field from
/// silencing the narration, but it is a fallback, not the source of truth.
fn limit_step_approach(step: &Step) -> Option<Approach> {
    step.meta.as_ref().and_then(|m| m.limit_approach)
}

/// True when the sub-step's `−∞` phrasing applies. Only certain when the
/// approach is recorded; an unrecorded step keeps the old behaviour.
fn limit_approaches_negative_infinity(step: &Step) -> bool {
    matches!(limit_step_approach(step), Some(Approach::NegInfinity))
}

pub(crate) fn generate_limit_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let approach = limit_step_approach(step);
    let at_infinity = match approach {
        Some(approach) => matches!(approach, Approach::PosInfinity | Approach::NegInfinity),
        None => step.rule_name.contains("infinito"),
    };
    let point = step.meta.as_ref().and_then(|m| m.limit_point);
    let Some(description) = notable_limit_name(
        ctx,
        step.before,
        step.after,
        at_infinity,
        limit_approaches_negative_infinity(step),
        point,
    ) else {
        return Vec::new();
    };
    // Deepened narratives that SHOW the work (factor → cancel → substitute, …).
    // `notable_limit_name` stays the single technique oracle; each deepening is
    // keyed on the recognized technique, and a builder that declines falls back to
    // the one-line technique-name substep below.
    if description == LIMIT_FACTOR_CANCEL_TITLE {
        if let Some(substeps) = generate_limit_factor_cancel_substeps(ctx, step, point) {
            return substeps;
        }
    }
    // Every FINITE notable-limit quotient (sin u/u, (e^u−1)/u, ln(1+u)/u, the
    // scaled/cross/reciprocal forms, (1−cos u)/u²) is a 0/0 indeterminate form;
    // show that direct substitution gives 0/0 before applying the standard limit.
    // The `… = e` forms are 1^∞, not 0/0, and are excluded.
    if !at_infinity
        && description.starts_with(LIMIT_NOTABLE_PREFIX)
        && !description.ends_with("= e")
    {
        return generate_limit_notable_zero_over_zero_substeps(ctx, step, description);
    }
    // The `… = e` notables (`(1+1/x)^x`, `(1+u)^(1/u)`) are the 1^∞ indeterminate
    // form (base → 1, exponent → ±∞); show that before citing the definition of e.
    if description.starts_with(LIMIT_NOTABLE_PREFIX) && description.ends_with("= e") {
        return generate_limit_e_form_substeps(ctx, step, description);
    }
    if description.starts_with(LIMIT_LHOPITAL_DESC_PREFIX) {
        return generate_limit_lhopital_substeps(ctx, step, point, description);
    }
    if description == LIMIT_SQUEEZE_TITLE {
        if let Some(substeps) = generate_limit_squeeze_substeps(ctx, step, approach) {
            return substeps;
        }
    }
    if description.starts_with(LIMIT_DOMINANCE_PREFIX) {
        if let Some(substeps) = generate_limit_dominance_substeps(ctx, step, &description) {
            return substeps;
        }
    }
    if description == LIMIT_CONJUGATE_TITLE {
        if let Some(substeps) = generate_limit_conjugate_substeps(ctx, step, approach) {
            return substeps;
        }
    }
    if description == LIMIT_COMMON_DENOM_TITLE {
        if let Some(p) = point {
            if let Some(substeps) = generate_limit_common_denom_substeps(ctx, step, p) {
                return substeps;
            }
        }
    }
    vec![SubStep::new(
        description,
        display_expr(ctx, step.before),
        display_expr(ctx, step.after),
    )]
}

/// Narrate a "Conservar límite residual" step (a finite-point limit the safe
/// policy does not decide) with an HONEST method hint: compute the one-sided
/// limits to investigate.
///
/// SOUNDNESS: the residual is a CONSERVATIVE under-answer, not a proof of
/// non-existence — the engine declines every undecided finite-point limit the
/// same way, lumping together genuine DNE cases (`1/x` at 0: left −∞, right +∞)
/// and limits that actually EXIST (`1/|x|` at 0 = +∞). So we must NOT claim the
/// limit does not exist; we only state the correct general METHOD (one-sided
/// limits decide it), which is sound for every case. Gated to finite-point
/// residuals (`limit_point` set); infinity / one-sided residuals get nothing.
pub(crate) fn generate_limit_residual_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let Some(point) = step.meta.as_ref().and_then(|m| m.limit_point) else {
        return Vec::new();
    };
    let var = limit_single_var_name(ctx, step.before).unwrap_or_else(|| "x".to_string());
    let point_disp = display_expr(ctx, point);
    let limit_disp = display_expr(ctx, step.after);
    let limit_latex = latex_expr(ctx, step.after);
    vec![SubStep::new(
        format!(
            "La política segura no decide este límite. Para investigarlo, calcula los límites \
             laterales en {var} = {point_disp} (por la izquierda y por la derecha): si coinciden, \
             ese es el valor del límite; si difieren, el límite no existe"
        ),
        limit_disp.clone(),
        limit_disp,
    )
    .with_before_latex(limit_latex.clone())
    .with_after_latex(limit_latex)]
}

/// Deepen factor-and-cancel into the explicit chain
/// `num/den → (g·cofn)/(g·cofd) → cofn/cofd → value`: extract the monic common
/// polynomial factor `g = gcd(num, den)`, show it pulled out of both sides, cancel
/// it, then substitute the limit point. Returns `None` (caller falls back to the
/// one-line name) unless `before` is a rational with a genuine shared factor and
/// the limit point is known.
fn generate_limit_factor_cancel_substeps(
    ctx: &Context,
    step: &Step,
    point: Option<ExprId>,
) -> Option<Vec<SubStep>> {
    use num_traits::{One, Zero};
    let point = point?;
    let (num, den) = as_div(ctx, step.before)?;
    let var = limit_single_var_name(ctx, num).or_else(|| limit_single_var_name(ctx, den))?;
    let num_poly = Polynomial::from_expr(ctx, num, &var).ok()?;
    let den_poly = Polynomial::from_expr(ctx, den, &var).ok()?;
    let g = num_poly.gcd(&den_poly);
    if g.degree() < 1 {
        return None;
    }
    // gcd returns an arbitrary rational scale; renormalize to monic so the pulled
    // factor reads naturally (`x - 1`, not `2·x - 2`).
    let lead = g.leading_coeff();
    if lead.is_zero() {
        return None;
    }
    let g = g.div_scalar(&lead);
    let (cofn, rn) = num_poly.div_rem(&g).ok()?;
    let (cofd, rd) = den_poly.div_rem(&g).ok()?;
    if !rn.is_zero() || !rd.is_zero() {
        return None;
    }
    let num_cofactor_is_one = cofn.degree() == 0 && cofn.leading_coeff().is_one();
    let den_cofactor_is_one = cofd.degree() == 0 && cofd.leading_coeff().is_one();

    let mut scratch = ctx.clone();
    let g_expr = g.to_expr(&mut scratch);
    let cofn_expr = cofn.to_expr(&mut scratch);
    let cofd_expr = cofd.to_expr(&mut scratch);

    // Factored sides: pull `g` out of each; a cofactor of 1 is not displayed.
    let factored_num = if num_cofactor_is_one {
        g_expr
    } else {
        scratch.add(Expr::Mul(g_expr, cofn_expr))
    };
    let factored_den = if den_cofactor_is_one {
        g_expr
    } else {
        scratch.add(Expr::Mul(g_expr, cofd_expr))
    };
    let factored = scratch.add(Expr::Div(factored_num, factored_den));
    // Cancelled form: cofn / cofd (just cofn when the denominator cancels away).
    let cancelled = if den_cofactor_is_one {
        cofn_expr
    } else {
        scratch.add(Expr::Div(cofn_expr, cofd_expr))
    };

    let g_disp = display_expr(&scratch, g_expr);
    let point_disp = display_expr(ctx, point);

    // C1.8: the closing line ASSERTS that substituting the point into the
    // cancelled form lands on the engine's answer. Unlike the emitter-built
    // pairs above it, the two sides come from different places — the cancelled
    // form was rebuilt here, `step.after` is the limit oracle — so the check has
    // something real to disagree about.
    let substitution = SubStep::checked_new(
        &scratch,
        crate::didactic::substep::Claim::EvalAt {
            var: var.clone(),
            point,
        },
        cancelled,
        step.after,
        format!("Sustituye {var} = {point_disp} en la expresión simplificada"),
        display_expr(&scratch, cancelled),
        display_expr(ctx, step.after),
    )?
    .with_before_latex(latex_expr(&scratch, cancelled))
    .with_after_latex(latex_expr(ctx, step.after));

    Some(vec![
        SubStep::keyed(
            "limit.factor_numerator_denominator",
            vec![],
            display_expr(ctx, step.before),
            display_expr(&scratch, factored),
        )
        .with_before_latex(latex_expr(ctx, step.before))
        .with_after_latex(latex_expr(&scratch, factored)),
        SubStep::new(
            format!("Cancela el factor común ({g_disp})"),
            display_expr(&scratch, factored),
            display_expr(&scratch, cancelled),
        )
        .with_before_latex(latex_expr(&scratch, factored))
        .with_after_latex(latex_expr(&scratch, cancelled)),
        substitution,
    ])
}

/// Deepen a 0/0 quotient notable limit into two substeps: first show that direct
/// substitution gives the indeterminate form 0/0 (so you cannot just plug in),
/// then apply the standard limit. Used for every FINITE notable-prefixed quotient
/// (`sin u/u`, `(e^u−1)/u`, `ln(1+u)/u`, the scaled/cross/reciprocal forms,
/// `(1−cos u)/u²`); the 1^∞ `… = e` forms are excluded by the caller.
fn generate_limit_notable_zero_over_zero_substeps(
    ctx: &Context,
    step: &Step,
    description: String,
) -> Vec<SubStep> {
    let before_disp = display_expr(ctx, step.before);
    let before_latex = latex_expr(ctx, step.before);
    vec![
        SubStep::keyed(
            "limit.direct_substitution_0_0",
            vec![],
            before_disp.clone(),
            "0 / 0",
        )
        .with_before_latex(before_latex.clone())
        .with_after_latex(r"\frac{0}{0}"),
        SubStep::new(description, before_disp, display_expr(ctx, step.after))
            .with_before_latex(before_latex)
            .with_after_latex(latex_expr(ctx, step.after)),
    ]
}

/// Deepen the generic 0/0 (L'Hôpital) narration into the explicit iteration, or
/// fall back to the one-line technique name when it cannot be reconstructed
/// soundly (transcendental denominator, irrational point, etc.).
fn generate_limit_lhopital_substeps(
    ctx: &Context,
    step: &Step,
    point: Option<ExprId>,
    description: String,
) -> Vec<SubStep> {
    generate_limit_lhopital_iteration(ctx, step, point).unwrap_or_else(|| {
        vec![SubStep::new(
            description,
            display_expr(ctx, step.before),
            display_expr(ctx, step.after),
        )]
    })
}

/// Reconstruct the L'Hôpital iteration: while the denominator vanishes at the
/// point, differentiate numerator and denominator (the limit is preserved by
/// L'Hôpital), then substitute once the denominator no longer vanishes.
///
/// SOUNDNESS: gated to a POLYNOMIAL denominator, so the number of steps is the
/// EXACT multiplicity of the root (computed via `Polynomial::derivative`/`eval`
/// over `BigRational`). Each intermediate `numᵏ/denᵏ` (k < m) is provably 0/0:
/// the polynomial denominator vanishes there, and a finite result forces the
/// numerator to vanish too (the same argument the one-line narration uses, and
/// L'Hôpital preserves the limit at every level). The final value is the engine's
/// result (`step.after`, the oracle) — never re-derived from the transcendental
/// numerator. A transcendental denominator (`sin x`, …) is not a polynomial here
/// and declines to the one-line name.
fn generate_limit_lhopital_iteration(
    ctx: &Context,
    step: &Step,
    point: Option<ExprId>,
) -> Option<Vec<SubStep>> {
    use cas_math::symbolic_differentiation_support::differentiate_symbolic_expr;
    use num_traits::{One, Zero};
    let point = point?;
    let p = as_rational_const(ctx, point, 8)?;
    let (num, den) = as_div(ctx, step.before)?;
    let var = limit_single_var_name(ctx, step.before)?;
    let den_poly = Polynomial::from_expr(ctx, den, &var).ok()?;
    // Steps needed = multiplicity of `p` as a root of the polynomial denominator.
    let mut d = den_poly.clone();
    let mut steps_needed = 0usize;
    while d.eval(&p).is_zero() {
        steps_needed += 1;
        if steps_needed > 8 {
            return None; // unrealistically high order; keep the narrative bounded
        }
        d = d.derivative();
    }
    if steps_needed == 0 {
        return None;
    }

    let point_disp = display_expr(ctx, point);
    let mut scratch = ctx.clone();
    let mut substeps: Vec<SubStep> = Vec::new();
    let mut cur_num = num;
    let mut cur_den = den;
    // Kept as a NODE, not only as strings: the closing sub-step declares a
    // relation about it, and a claim cannot be checked against a render.
    let mut cur_form = step.before;
    let mut cur_disp = display_expr(ctx, step.before);
    let mut cur_latex = latex_expr(ctx, step.before);
    for k in 0..steps_needed {
        // The symbolic differentiator emits UNFOLDED arithmetic (`3·x^(3-1)`,
        // `e^x·ln(e)`); simplify numerator and denominator independently so each
        // step reads cleanly (`3·x²`, `e^x`). Simplifying between steps also keeps
        // the next derivative clean.
        let raw_num = differentiate_symbolic_expr(&mut scratch, cur_num, &var)?;
        let raw_den = differentiate_symbolic_expr(&mut scratch, cur_den, &var)?;
        let next_num = simplify_expr_in_context(&mut scratch, raw_num);
        let next_den = simplify_expr_in_context(&mut scratch, raw_den);
        // `num'/1` reads as just `num'` (the derivative of a linear factor).
        let next_den_is_one = matches!(scratch.get(next_den), Expr::Number(n) if n.is_one());
        let next_form = if next_den_is_one {
            next_num
        } else {
            scratch.add(Expr::Div(next_num, next_den))
        };
        let next_disp = display_expr(&scratch, next_form);
        let next_latex = latex_expr(&scratch, next_form);
        let substep = if k == 0 {
            SubStep::keyed(
                "limit.lhopital_first_iteration",
                vec![format!("{var}"), format!("{point_disp}")],
                cur_disp.clone(),
                next_disp.clone(),
            )
        } else {
            SubStep::keyed(
                "limit.lhopital_still_0_0_again",
                vec![],
                cur_disp.clone(),
                next_disp.clone(),
            )
        };
        substeps.push(
            substep
                .with_before_latex(cur_latex.clone())
                .with_after_latex(next_latex.clone()),
        );
        cur_num = next_num;
        cur_den = next_den;
        cur_form = next_form;
        cur_disp = next_disp;
        cur_latex = next_latex;
    }
    // C1.8: same shape as the factor-and-cancel closing line — the iterated
    // quotient was rebuilt here, `step.after` is the engine's answer, and the
    // sub-step asserts that one substituted into the other gives the other. A
    // refutation means the reconstruction diverged from the oracle somewhere in
    // the iteration, so the WHOLE deepened narration declines and the caller
    // falls back to the one-line technique name.
    substeps.push(
        SubStep::checked(
            &scratch,
            crate::didactic::substep::Claim::EvalAt {
                var: var.clone(),
                point,
            },
            cur_form,
            step.after,
            "limit.lhopital_denominator_nonzero_substitute",
            vec![format!("{var}"), format!("{point_disp}")],
            cur_disp,
            display_expr(ctx, step.after),
        )?
        .with_before_latex(cur_latex)
        .with_after_latex(latex_expr(ctx, step.after)),
    );
    Some(substeps)
}

/// Extract the `(power, oscillator)` factors of a squeeze product
/// `u^k · sin/cos(…/u)`, in whichever order they appear.
fn limit_squeeze_parts(ctx: &Context, before: ExprId) -> Option<(ExprId, ExprId)> {
    let Expr::Mul(left, right) = ctx.get(before) else {
        return None;
    };
    let (left, right) = (*left, *right);
    [(left, right), (right, left)]
        .into_iter()
        .find_map(|(power, bounded)| {
            limit_power_of_var(ctx, power)
                .filter(|u| limit_is_bounded_reciprocal_oscillator(ctx, bounded, *u))
                .map(|_| (power, bounded))
        })
}

/// Deepen the squeeze theorem into the bounding argument: the oscillator is bounded
/// (`|sin/cos| ≤ 1`), so `|uᵏ · osc| ≤ |uᵏ|`, and `|uᵏ| → 0`, hence the product → 0.
fn generate_limit_squeeze_substeps(
    ctx: &Context,
    step: &Step,
    approach: Option<Approach>,
) -> Option<Vec<SubStep>> {
    let (power, osc) = limit_squeeze_parts(ctx, step.before)?;
    let var = limit_single_var_name(ctx, step.before)?;
    let mut scratch = ctx.clone();
    let abs_id = scratch.builtin_id(BuiltinFn::Abs);
    let abs_power = scratch.add(Expr::Function(abs_id, vec![power]));
    let osc_disp = display_expr(ctx, osc);
    let before_disp = display_expr(ctx, step.before);
    let abs_disp = display_expr(&scratch, abs_power);
    // C1.8: the closing line ASSERTS that the bounding infinitesimal has the
    // step's limit. `|uᵏ|` is rebuilt HERE and `step.after` is the engine's
    // answer for a different expression (the product), so the engine's own
    // limit oracle has something real to disagree with.
    let conclusion = SubStep::checked_new(
        &scratch,
        crate::didactic::substep::Claim::Limit {
            var,
            approach: approach?,
        },
        abs_power,
        step.after,
        format!(
            "El infinitésimo {abs_disp} → 0, así que por el teorema del sándwich el límite es 0"
        ),
        abs_disp.clone(),
        display_expr(ctx, step.after),
    )?
    .with_before_latex(latex_expr(&scratch, abs_power))
    .with_after_latex(latex_expr(ctx, step.after));
    Some(vec![
        SubStep::new(
            format!(
                "Acota el factor oscilante: |{osc_disp}| ≤ 1, luego |{before_disp}| ≤ {abs_disp}"
            ),
            before_disp,
            abs_disp,
        )
        .with_before_latex(latex_expr(ctx, step.before))
        .with_after_latex(latex_expr(&scratch, abs_power)),
        conclusion,
    ])
}

/// Deepen an infinity-dominance quotient into the ∞/∞ form, then the dominance
/// conclusion — but only when the quotient is GENUINELY ∞/∞: the growth-class
/// cases (ln ≪ power ≪ exp) always are, and a rational quotient is iff both
/// numerator and denominator are polynomials of degree ≥ 1 (so `1/x`, a `1/∞`,
/// declines, as do the bare-polynomial and product-decay cases that are not
/// quotients).
fn generate_limit_dominance_substeps(
    ctx: &Context,
    step: &Step,
    description: &str,
) -> Option<Vec<SubStep>> {
    let (num, den) = as_div(ctx, step.before)?;
    let var = limit_single_var_name(ctx, step.before)?;
    let poly_unbounded =
        |e: ExprId| Polynomial::from_expr(ctx, e, &var).is_ok_and(|p| p.degree() >= 1);
    // Growth-class titles carry the hierarchy phrase; both sides tend to ∞ there.
    let is_inf_over_inf =
        description.contains("jerarquía ln") || (poly_unbounded(num) && poly_unbounded(den));
    if !is_inf_over_inf {
        return None;
    }
    let before_disp = display_expr(ctx, step.before);
    let before_latex = latex_expr(ctx, step.before);
    // At `x→−∞` an odd-degree numerator tends to −∞, so "numerator and
    // denominator → ∞" is simply false there. The form is still ∞/∞; the
    // recorded approach is what lets the line say so without lying.
    let indeterminate_key = if limit_approaches_negative_infinity(step) {
        "limit.numerator_denominator_inf_over_inf_negative"
    } else {
        "limit.numerator_denominator_inf_over_inf"
    };
    Some(vec![
        SubStep::keyed(indeterminate_key, vec![], before_disp.clone(), "∞/∞")
            .with_before_latex(before_latex.clone())
            .with_after_latex(r"\frac{\infty}{\infty}"),
        SubStep::new(
            description.to_string(),
            before_disp,
            display_expr(ctx, step.after),
        )
        .with_before_latex(before_latex)
        .with_after_latex(latex_expr(ctx, step.after)),
    ])
}

/// Split an `∞−∞` difference into `(√P, linear, surd_is_first)`: `√P − L` gives
/// `surd_is_first = true`, `L − √P` gives `false`. `P` must be degree 2 and `L`
/// degree 1 with `leading(P) = leading(L)²`, so the leading `±∞` terms cancel and
/// the difference is the genuine `∞−∞` that conjugate rationalization resolves.
fn limit_conjugate_parts(ctx: &Context, before: ExprId) -> Option<(ExprId, ExprId, bool)> {
    let Expr::Sub(l, r) = ctx.get(before) else {
        return None;
    };
    let (l, r) = (*l, *r);
    let (surd, linear, surd_first) = if as_sqrt_radicand(ctx, l).is_some() {
        (l, r, true)
    } else if as_sqrt_radicand(ctx, r).is_some() {
        (r, l, false)
    } else {
        return None;
    };
    let radicand = as_sqrt_radicand(ctx, surd)?;
    let var = limit_single_var_name(ctx, before)?;
    let p = Polynomial::from_expr(ctx, radicand, &var).ok()?;
    let q = Polynomial::from_expr(ctx, linear, &var).ok()?;
    if p.degree() != 2 || q.degree() != 1 {
        return None;
    }
    let d = q.leading_coeff();
    if p.leading_coeff() != &d * &d {
        return None;
    }
    Some((surd, linear, surd_first))
}

/// Recognize an `∞−∞` limit at infinity of the form `√(a·x²+b·x+c) − d·x` (or the
/// reverse) with `a = d²`, resolved by multiplying and dividing by the conjugate.
/// `after` (the true limit, required finite) is the soundness oracle.
fn limit_infinity_conjugate_radical(
    ctx: &Context,
    before: ExprId,
    after: ExprId,
) -> Option<String> {
    as_rational_const(ctx, after, 8)?;
    limit_conjugate_parts(ctx, before)?;
    Some(LIMIT_CONJUGATE_TITLE.to_string())
}

/// Deepen the conjugate-rationalization `∞−∞` limit into three substeps: show the
/// indeterminate `∞−∞`, multiply/divide by the conjugate to reach the rational
/// form `(P − L²)/(√P + L)`, then divide by the dominant power and evaluate.
///
/// SOUNDNESS: the rationalized numerator is `P − L²`, which — since `(√P)² = P` —
/// equals `(√P − L)(√P + L)`. A self-check re-multiplies `before · conjugate` and
/// requires it to fold to that (signed) numerator, so any construction error
/// DECLINES to the one-line technique name rather than narrating false algebra.
fn generate_limit_conjugate_substeps(
    ctx: &Context,
    step: &Step,
    approach: Option<Approach>,
) -> Option<Vec<SubStep>> {
    let before = step.before;
    let (surd, linear, surd_first) = limit_conjugate_parts(ctx, before)?;
    let radicand = as_sqrt_radicand(ctx, surd)?;
    let var = limit_single_var_name(ctx, before)?;

    let mut scratch = ctx.clone();
    // Conjugate `√P + L`.
    let conjugate = scratch.add(Expr::Add(surd, linear));
    // Rationalized numerator `P − L²`, simplified (folds to `b·x + c` when `a = d²`).
    let two = scratch.add(Expr::Number(BigRational::from_integer(2.into())));
    let linear_sq = scratch.add(Expr::Pow(linear, two));
    let num_raw = scratch.add(Expr::Sub(radicand, linear_sq));
    let num = simplify_expr_in_context(&mut scratch, num_raw);
    // `√P − L` equals `num/conjugate`; `L − √P` equals `−num/conjugate`. Pick the
    // signed numerator so the quotient equals `before`.
    let signed_num = if surd_first {
        num
    } else {
        let neg = scratch.add(Expr::Neg(num));
        simplify_expr_in_context(&mut scratch, neg)
    };
    // Self-check: `before · conjugate` must fold to `signed_num` ((√P)² = P).
    let check_raw = scratch.add(Expr::Mul(before, conjugate));
    let check = simplify_expr_in_context(&mut scratch, check_raw);
    if check != signed_num {
        return None;
    }
    let rationalized = scratch.add(Expr::Div(signed_num, conjugate));

    let conj_disp = display_expr(&scratch, conjugate);
    let surd_disp = display_expr(ctx, surd);
    let after_disp = display_expr(ctx, step.after);
    let rationalized_disp = display_expr(&scratch, rationalized);

    Some(vec![
        SubStep::keyed(
            "limit.inf_minus_inf_indeterminate",
            vec![],
            display_expr(ctx, before),
            "∞ − ∞",
        )
        .with_before_latex(latex_expr(ctx, before))
        .with_after_latex(r"\infty - \infty"),
        SubStep::keyed(
            "limit.multiply_divide_by_conjugate",
            vec![conj_disp],
            display_expr(ctx, before),
            rationalized_disp.clone(),
        )
        .with_before_latex(latex_expr(ctx, before))
        .with_after_latex(latex_expr(&scratch, rationalized)),
        // C1.8: the closing line ASSERTS that the RATIONALIZED form — rebuilt
        // here — has the step's limit. Declared and handed to the engine's own
        // oracle. MEASURED: the oracle's conservative policy decides the
        // original `√P − L` but declines the rationalized quotient, so this one
        // lands on `Undecided` and publishes. That is the honest outcome, not a
        // silent pass: an abstention by the oracle is not evidence of a lie.
        SubStep::checked(
            &scratch,
            crate::didactic::substep::Claim::Limit {
                var: var.clone(),
                approach: approach?,
            },
            rationalized,
            step.after,
            "limit.divide_by_dominant_power_evaluate",
            vec![surd_disp, var, after_disp.clone()],
            rationalized_disp,
            after_disp,
        )?
        .with_before_latex(latex_expr(&scratch, rationalized))
        .with_after_latex(latex_expr(ctx, step.after)),
    ])
}

/// Recognize a reciprocal-difference `∞−∞` limit at a FINITE point: `c/f − d/g`
/// where `c,d` are nonzero rational constants and both `f,g → 0` at the point (so
/// each term blows up), with a finite `after` (the value oracle). Resolved by
/// combining over a common denominator. Gated to the finite branch (the caller is
/// past `if at_infinity`), so the at-infinity conjugate form never reaches here.
fn limit_reciprocal_difference_common_denom(
    ctx: &Context,
    before: ExprId,
    after: ExprId,
    point: Option<ExprId>,
) -> Option<String> {
    let point = point?;
    as_rational_const(ctx, after, 8)?; // finite result is the oracle
    let (a, b, is_subtraction) = extract_fraction_add_sub_operands(ctx, before)?;
    if !is_subtraction {
        return None; // ∞−∞ needs a subtraction (∞+∞ is not indeterminate)
    }
    for term in [a, b] {
        let (num, den) = as_div(ctx, term)?;
        let k = as_rational_const(ctx, num, 8)?;
        if k.is_zero() {
            return None;
        }
        if !limit_denominator_vanishes_at(ctx, den, point) {
            return None; // this term does not blow up → not ∞−∞
        }
    }
    Some(LIMIT_COMMON_DENOM_TITLE.to_string())
}

/// Deepen the reciprocal-difference `∞−∞` limit into substeps: show the
/// indeterminate `∞−∞`, combine over a common denominator to reach a single
/// fraction, then resolve the resulting `0/0` — recursing the limit-substep
/// machinery on the combined fraction (rational cases get factor-cancel / a full
/// L'Hôpital iteration) and, when that declines (transcendental product
/// denominators such as `x·sin(x)`), closing with the honest `0/0 → L'Hôpital /
/// Taylor` one-liner.
///
/// SOUNDNESS: the common denominator is `f·g`, and the recognizer already certified
/// both `f→0` and `g→0` at the point, so the denominator vanishes; a finite `after`
/// then forces the numerator to vanish too, making the `0/0` claim honest. The
/// combiner `(c₁/f₁) − (c₂/f₂) = (c₁·f₂ − c₂·f₁)/(f₁·f₂)` is an algebraic IDENTITY
/// for the `c/f` terms the recognizer certified, so its output always equals
/// `before` by construction — no simplifier-proof self-check is possible or needed
/// (the simplifier cannot fold the tangent identities to literal 0 anyway).
fn generate_limit_common_denom_substeps(
    ctx: &Context,
    step: &Step,
    point: ExprId,
) -> Option<Vec<SubStep>> {
    let before = step.before;
    let var = limit_single_var_name(ctx, before)?;

    let mut scratch = ctx.clone();
    // Combine `A − B` over a common denominator via the shared cross-multiply combiner.
    let combined_raw = build_two_fraction_common_denominator_intermediate(&mut scratch, before)?;
    // Clean the `1·…` factors into a single readable fraction for display + recursion.
    let combined = simplify_expr_in_context(&mut scratch, combined_raw);
    let combined_disp = display_expr(&scratch, combined);
    let combined_latex = latex_expr(&scratch, combined);
    let after_disp = display_expr(ctx, step.after);

    let mut substeps = vec![
        SubStep::keyed(
            "limit.inf_minus_inf_indeterminate",
            vec![],
            display_expr(ctx, before),
            "∞ − ∞",
        )
        .with_before_latex(latex_expr(ctx, before))
        .with_after_latex(r"\infty - \infty"),
        SubStep::keyed(
            "limit.combine_over_common_denominator",
            vec![],
            display_expr(ctx, before),
            combined_disp.clone(),
        )
        .with_before_latex(latex_expr(ctx, before))
        .with_after_latex(combined_latex.clone()),
    ];

    // Resolve the resulting 0/0. Recurse on the combined fraction; the combined form
    // is a `Div`, never a `Sub`-of-reciprocals, so this recognizer cannot re-fire.
    let mut synthetic = Step::new_compact("desc", "Evaluar límite finito", combined, step.after);
    synthetic.meta_mut().limit_point = Some(point);
    // The recursion inherits the approach too. A synthetic step that carries the
    // point but not the direction would make every claim downstream decline for
    // want of a datum this function already has.
    synthetic.meta_mut().limit_approach = Some(Approach::Finite(point));
    let tail = generate_limit_substeps(&scratch, &synthetic);
    if tail.is_empty() {
        // C1.8: this closing line ASSERTS that the COMBINED fraction — rebuilt
        // here by the cross-multiply combiner — has the step's limit. The
        // engine's own oracle decides it (measured), so the reconstruction is
        // checked against the value rather than trusted.
        substeps.push(
            SubStep::checked(
                &scratch,
                crate::didactic::substep::Claim::Limit {
                    var: var.clone(),
                    approach: Approach::Finite(point),
                },
                combined,
                step.after,
                "limit.generic_0_0_lhopital_or_taylor",
                vec![var, display_expr(ctx, point)],
                combined_disp,
                after_disp,
            )?
            .with_before_latex(combined_latex)
            .with_after_latex(latex_expr(ctx, step.after)),
        );
    } else {
        substeps.extend(tail);
    }
    Some(substeps)
}

/// Deepen a `1^∞` notable (`(1+1/x)^x → e`, `(1+u)^(1/u) → e`) into two substeps:
/// show that the base tends to 1 and the exponent to ±∞ (the indeterminate form
/// `1^∞`, so you cannot just take `1^∞ = 1`), then cite the definition of `e`.
fn generate_limit_e_form_substeps(ctx: &Context, step: &Step, description: String) -> Vec<SubStep> {
    let before_disp = display_expr(ctx, step.before);
    let before_latex = latex_expr(ctx, step.before);
    // The exponent is the VARIABLE, so at `x→−∞` it tends to `−∞`, not `∞`.
    let indeterminate_key = if limit_approaches_negative_infinity(step) {
        "limit.base_to_1_exponent_to_inf_1_pow_inf_negative"
    } else {
        "limit.base_to_1_exponent_to_inf_1_pow_inf"
    };
    vec![
        SubStep::keyed(indeterminate_key, vec![], before_disp.clone(), "1^∞")
            .with_before_latex(before_latex.clone())
            .with_after_latex(r"1^{\infty}"),
        SubStep::new(description, before_disp, display_expr(ctx, step.after))
            .with_before_latex(before_latex)
            .with_after_latex(latex_expr(ctx, step.after)),
    ]
}

/// Full didactic description of the standard ("notable") limit / theorem / method a `before/after`
/// limit step realises, matching the structural form of `before` AND requiring `after` to be the
/// matching value (the result acts as the soundness oracle — a structural match with the wrong
/// value is rejected). Returns `None` when no standard form is recognised.
pub(super) fn notable_limit_name(
    ctx: &Context,
    before: ExprId,
    after: ExprId,
    at_infinity: bool,
    negative_infinity: bool,
    point: Option<ExprId>,
) -> Option<String> {
    // Limits at infinity have their own dominance methods; the finite forms below (notable
    // limits at 0, continuity, factor-and-cancel) do not apply there. The one infinity-side
    // NOTABLE is the definition of e — narrated only when the result is exactly e.
    if at_infinity {
        if matches!(ctx.get(after), Expr::Constant(Constant::E))
            && limit_is_one_plus_reciprocal_power(ctx, before)
        {
            // NAME THE DIRECTION. `(1 + 1/x)^x → e` holds at both infinities,
            // but the sub-step used to cite the `x→∞` theorem while narrating a
            // limit at `x→−∞` — a true value justified by a statement about a
            // different limit. The step's recorded approach settles it; without
            // that datum this line had no way to be right.
            let arrow = if negative_infinity {
                "x→−∞"
            } else {
                "x→∞"
            };
            return Some(format!(
                "Aplicar el límite notable: lím({arrow}) (1 + 1/x)^x = e"
            ));
        }
        if let Some(desc) = limit_infinity_conjugate_radical(ctx, before, after) {
            return Some(desc);
        }
        return limit_infinity_dominance(ctx, before, after);
    }

    let prefix = LIMIT_NOTABLE_PREFIX;
    let after_value = as_rational_const(ctx, after, 8);
    let after_is = |target: BigRational| after_value.as_ref() == Some(&target);

    // Reciprocal-difference `∞−∞` at a finite point: `c/f − d/g` where both f,g → 0
    // (each term blows up) and the result is finite → combine over a common
    // denominator. `before` is a `Sub`, disjoint from the `Div` forms below.
    if let Some(desc) = limit_reciprocal_difference_common_denom(ctx, before, after, point) {
        return Some(desc);
    }

    if let Some((num, den)) = as_div(ctx, before) {
        // den is the bare variable u.
        if matches!(ctx.get(den), Expr::Variable(_)) {
            // f(a·u)/u → a for the first-order equivalents (sin, tan, arcsin, arctan, sinh, ...):
            // the bare notable f(u)/u = 1 is the a = 1 case, and the scaled `sin(3x)/x → 3` follows
            // from sin(au)/u = a·sin(au)/(au) → a. SOUNDNESS: narrate only when the result equals
            // the scale a exactly, so `sin(3x)/x → 2` (fabricated) and `sin(x)/x → sin(5)/5` decline.
            if let Some((arg, builtin)) = limit_unary_builtin(ctx, num) {
                if let Some(name) = first_order_equivalent_name(builtin) {
                    if let Some(scale) = limit_linear_scale(ctx, arg, den) {
                        if !scale.is_zero() && after_is(scale.clone()) {
                            return Some(if scale.is_one() {
                                format!("{prefix}lím(u→0) {name}(u)/u = 1")
                            } else {
                                format!("{prefix}lím(u→0) {name}({scale}·u)/u = {scale}")
                            });
                        }
                    }
                }
            }
            if after_is(BigRational::one()) {
                if let Some((arg, builtin)) = limit_unary_builtin(ctx, num) {
                    if builtin == BuiltinFn::Ln && limit_is_one_plus(ctx, arg, den) {
                        return Some(format!("{prefix}lím(u→0) ln(1+u)/u = 1"));
                    }
                }
                if limit_is_exp_minus_one(ctx, num, den) {
                    return Some(format!("{prefix}lím(u→0) (e^u − 1)/u = 1"));
                }
            }
            // (a^u − 1)/u → ln(a) for a positive rational base a ≠ 1 (result must be ln(a)).
            if let Some(base) = limit_rational_base_pow_minus_one(ctx, num, den) {
                if limit_after_is_ln_of(ctx, after, base) {
                    return Some(format!("{prefix}lím(u→0) (aᵘ − 1)/u = ln(a)"));
                }
            }
            // ((1+u)^a − 1)/u → a, the binomial / root first-order equivalent
            // `(1+u)^a ~ 1 + a·u` (a = 1/2 is `(√(1+u) − 1)/u → 1/2`). The result must equal a.
            if let Some(a) = limit_one_plus_power_minus_one_exponent(ctx, num, den) {
                if !a.is_zero() && after_is(a.clone()) {
                    return Some(format!(
                        "{prefix}lím(u→0) ((1+u)^({a}) − 1)/u = {a}  (equivalente de primer orden (1+u)^a ~ 1 + a·u)"
                    ));
                }
            }
        }
        // num is the bare variable u: the RECIPROCAL notables u/f(u) → 1.
        if matches!(ctx.get(num), Expr::Variable(_)) && after_is(BigRational::one()) {
            if let Some((arg, builtin)) = limit_unary_builtin(ctx, den) {
                if compare_expr(ctx, arg, num) == Ordering::Equal {
                    if let Some(name) = first_order_equivalent_name(builtin) {
                        return Some(format!("{prefix}lím(u→0) u/{name}(u) = 1"));
                    }
                }
            }
            if limit_is_exp_minus_one(ctx, den, num) {
                return Some(format!("{prefix}lím(u→0) u/(e^u − 1) = 1"));
            }
        }
        // den is u²: (1 − cos(u))/u² → 1/2.
        if after_is(BigRational::new(1.into(), 2.into())) {
            if let Some(u) = limit_square_of_var(ctx, den) {
                if limit_is_one_minus_cos(ctx, num, u) {
                    return Some(format!("{prefix}lím(u→0) (1 − cos(u))/u² = 1/2"));
                }
            }
        }
        // f(a·u)/g(b·u) → a/b for first-order equivalents on one or both sides (the bare side is
        // just `b·u`): `sin(3x)/(2x) → 3/2`, `tan(3x)/sin(2x) → 3/2`, `sin(x)/(2x) → 1/2`. Each side
        // contributes its linear scale (a notable `g(b·u) ~ b·u`, so scale b). At least one side
        // must be a genuine notable function (else it is plain `a·u/(b·u)` cancellation). SOUNDNESS:
        // narrate only when the result equals a/b exactly (the result is the oracle, as elsewhere).
        if let Some(u_name) = limit_single_var_name(ctx, before) {
            if let (Some((a, num_fn)), Some((b, den_fn))) = (
                limit_first_order_factor(ctx, num, &u_name),
                limit_first_order_factor(ctx, den, &u_name),
            ) {
                if (num_fn.is_some() || den_fn.is_some()) && !b.is_zero() {
                    let ratio = &a / &b;
                    if after_is(ratio.clone()) {
                        // `f(a·u)` is self-delimited; a bare `a·u` (a ≠ 1) is parenthesised so the
                        // quotient reads unambiguously (`(2·u)`, not `2·u`).
                        let side = |scale: &BigRational, name: Option<&str>| -> String {
                            match name {
                                Some(f) if scale.is_one() => format!("{f}(u)"),
                                Some(f) => format!("{f}({scale}·u)"),
                                None if scale.is_one() => "u".to_string(),
                                None => format!("({scale}·u)"),
                            }
                        };
                        return Some(format!(
                            "{prefix}lím(u→0) {}/{} = {ratio}",
                            side(&a, num_fn),
                            side(&b, den_fn),
                        ));
                    }
                }
            }
        }
        // Rational with a shared polynomial factor (gcd ≠ 1) and a finite result:
        // factor-and-cancel — `(x²−1)/(x−1) = x+1`. The cancellation is a valid algebraic
        // step at ANY point (no "0/0" claim, which would need the limit point).
        if after_value.is_some() && limit_share_polynomial_factor(ctx, num, den) {
            return Some(LIMIT_FACTOR_CANCEL_TITLE.to_string());
        }
        // Generic 0/0 at the limit point, not captured by a specific notable above. SOUNDNESS: the
        // denominator is a polynomial that vanishes AT THE LIMIT POINT (evaluated exactly), so it
        // tends to 0 there; a FINITE result then forces the numerator to vanish too
        // (lím num = result · lím den = result · 0 = 0), making the form provably 0/0 — no Taylor
        // re-derivation needed and no need to substitute the (possibly transcendental) numerator.
        // The point is checked at the ACTUAL approached value, so `ln(x)/(x−1)` narrates at 1 but
        // declines at 0 (where x−1 → −1 ≠ 0), and `(x+1)/x → x→2` / `sin(πx)/x → x→1` decline
        // (their denominator does not vanish at the point). Fires only as a fallback (every specific
        // notable / factor-cancel returns earlier) and only when the numerator involves the variable
        // (a constant numerator gives ±∞, never a finite result). Covers `(x − sin x)/x³ → 1/6` at 0,
        // `(1 − cos(x−1))/(x−1)² → 1/2` and `ln(x)/(x−1) → 1` at 1.
        if let Some(point) = point {
            if after_value.is_some()
                && limit_single_var_name(ctx, num).is_some()
                && limit_denominator_vanishes_at(ctx, den, point)
            {
                let var = limit_single_var_name(ctx, before).unwrap_or_else(|| "x".to_string());
                return Some(format!(
                    "{LIMIT_LHOPITAL_DESC_PREFIX} {var}={}: aplica la regla de L'Hôpital (deriva \
                     numerador y denominador) o el desarrollo de Taylor",
                    display_expr(ctx, point)
                ));
            }
        }
    }

    // Continuous polynomial: the limit is the value at the point (direct
    // substitution). A polynomial is an atomic, single-step evaluation — there is
    // no intermediate worth showing (the substituted-but-unevaluated arithmetic
    // form, e.g. `2² + 3·2 + 1`, only renders messily through the canonical
    // formatter, which reorders terms and folds `(-2)·(-1)` to `1·2`), so we keep
    // ONE substep but name the specific point when it is known.
    if after_value.is_some() && as_div(ctx, before).is_none() && limit_is_polynomial(ctx, before) {
        if let Some(point) = point {
            let var = limit_single_var_name(ctx, before).unwrap_or_else(|| "x".to_string());
            return Some(format!(
                "Sustitución directa: el polinomio es continuo, así que el límite es su valor en {var} = {}",
                display_expr(ctx, point)
            ));
        }
        return Some(LIMIT_DIRECT_SUBSTITUTION_TITLE.to_string());
    }

    // Squeeze theorem: (power of u) · (bounded sin/cos of a reciprocal in u) → 0.
    if after_value.as_ref() == Some(&BigRational::zero()) && limit_is_squeeze_product(ctx, before) {
        return Some(LIMIT_SQUEEZE_TITLE.to_string());
    }

    // (1 + u)^(1/u) → e.
    if matches!(ctx.get(after), Expr::Constant(Constant::E))
        && limit_is_one_plus_to_reciprocal(ctx, before)
    {
        return Some(format!("{prefix}lím(u→0) (1 + u)^(1/u) = e"));
    }

    None
}

/// If `arg` is `a·u` for a nonzero rational `a` and the bare variable `u` (no constant offset),
/// return `a`. This is the scale of the notable argument `f(a·u)/u → a` (`a = 1` is the bare
/// `f(u)/u`); `1 + u` and `a·u + b` with `b ≠ 0` return `None` so only the pure linear form matches.
fn limit_linear_scale(ctx: &Context, arg: ExprId, u: ExprId) -> Option<BigRational> {
    let Expr::Variable(sym) = ctx.get(u) else {
        return None;
    };
    let name = ctx.sym_name(*sym).to_string();
    linear_scale_of(ctx, arg, &name)
}

/// The linear scale a side of a `num/den` notable quotient contributes at `u → 0`: a bare `a·u`
/// gives `(a, None)`, and a first-order-equivalent notable `f(a·u)` gives `(a, Some("f"))` since
/// `f(a·u) ~ a·u`. A unary function without a first-order equivalent (cos, ln, …) returns `None`.
fn limit_first_order_factor(
    ctx: &Context,
    expr: ExprId,
    u_name: &str,
) -> Option<(BigRational, Option<&'static str>)> {
    if let Some((arg, builtin)) = limit_unary_builtin(ctx, expr) {
        let name = first_order_equivalent_name(builtin)?;
        return Some((linear_scale_of(ctx, arg, u_name)?, Some(name)));
    }
    Some((linear_scale_of(ctx, expr, u_name)?, None))
}

fn limit_unary_builtin(ctx: &Context, expr: ExprId) -> Option<(ExprId, BuiltinFn)> {
    if let Expr::Function(fn_id, args) = ctx.get(expr) {
        if args.len() == 1 {
            if let Some(builtin) = ctx.builtin_of(*fn_id) {
                return Some((args[0], builtin));
            }
        }
    }
    None
}

fn limit_is_number(ctx: &Context, expr: ExprId, value: i64) -> bool {
    as_rational_const(ctx, expr, 8).as_ref() == Some(&BigRational::from_integer(value.into()))
}

fn limit_square_of_var(ctx: &Context, den: ExprId) -> Option<ExprId> {
    let (base, exponent) = as_pow(ctx, den)?;
    (matches!(ctx.get(base), Expr::Variable(_)) && limit_is_number(ctx, exponent, 2))
        .then_some(base)
}

/// `den` provably tends to 0 as the variable approaches the limit `point`, certifying — paired with
/// a finite result — a genuine `0/0` indeterminate form there. Three sound, structural ways (the
/// last two recurse on the inner expression, so `sin(x)³`, `tan(x²)`, … are covered):
///
/// (1) a single-variable polynomial that vanishes at `point` (evaluated EXACTLY at the rational
/// point): `x³` at 0, `x−1` and `(x−1)²` at 1.
///
/// (2) `f(g)` where `f` is a zero-at-the-origin first-order function (sin/tan/sinh/tanh/arcsin/
/// arctan/…, all `f(0)=0`) and `g` itself tends to 0 at the point, so `f(g) → f(0) = 0`: `sin(x)`,
/// `tan(x)` at 0.
///
/// (3) `d^k` for an integer `k ≥ 1` whose base `d` tends to 0 at the point: `sin(x)³` at 0.
fn limit_denominator_vanishes_at(ctx: &Context, den: ExprId, point: ExprId) -> bool {
    // (1) polynomial vanishing exactly at the (rational) point.
    if let Some(p) = as_rational_const(ctx, point, 8) {
        if let Some(var) = limit_single_var_name(ctx, den) {
            if Polynomial::from_expr(ctx, den, &var).is_ok_and(|poly| poly.eval(&p).is_zero()) {
                return true;
            }
        }
    }
    // (2) f(g) with f(0)=0 and g → 0 at the point.
    if let Some((arg, builtin)) = limit_unary_builtin(ctx, den) {
        if first_order_equivalent_name(builtin).is_some()
            && limit_denominator_vanishes_at(ctx, arg, point)
        {
            return true;
        }
    }
    // (3) d^k with integer k ≥ 1 and d → 0 at the point.
    if let Some((base, exponent)) = as_pow(ctx, den) {
        if as_rational_const(ctx, exponent, 4)
            .is_some_and(|k| k.is_integer() && k >= BigRational::one())
            && limit_denominator_vanishes_at(ctx, base, point)
        {
            return true;
        }
    }
    false
}

fn limit_is_one_plus(ctx: &Context, arg: ExprId, u: ExprId) -> bool {
    if let Expr::Add(left, right) = ctx.get(arg) {
        let (left, right) = (*left, *right);
        (limit_is_number(ctx, left, 1) && compare_expr(ctx, right, u) == Ordering::Equal)
            || (limit_is_number(ctx, right, 1) && compare_expr(ctx, left, u) == Ordering::Equal)
    } else {
        false
    }
}

/// The rational exponent `a` for which `num == (1+u)^a − 1`, including the `√(1+u) − 1` spelling
/// (a = 1/2). `None` if `num` is not that shape. The first-order equivalent gives `num/u → a`.
fn limit_one_plus_power_minus_one_exponent(
    ctx: &Context,
    num: ExprId,
    u: ExprId,
) -> Option<BigRational> {
    let Expr::Sub(left, right) = *ctx.get(num) else {
        return None;
    };
    if !limit_is_number(ctx, right, 1) {
        return None;
    }
    if let Some((base, exponent)) = as_pow(ctx, left) {
        return limit_is_one_plus(ctx, base, u)
            .then(|| as_rational_const(ctx, exponent, 8))
            .flatten();
    }
    if let Some((arg, BuiltinFn::Sqrt)) = limit_unary_builtin(ctx, left) {
        return limit_is_one_plus(ctx, arg, u).then(|| BigRational::new(1.into(), 2.into()));
    }
    None
}

fn limit_is_exp_minus_one(ctx: &Context, num: ExprId, u: ExprId) -> bool {
    if let Expr::Sub(left, right) = ctx.get(num) {
        let (left, right) = (*left, *right);
        if limit_is_number(ctx, right, 1) {
            if let Some(arg) = extract_exp_argument(ctx, left) {
                return compare_expr(ctx, arg, u) == Ordering::Equal;
            }
        }
    }
    false
}

fn limit_is_one_minus_cos(ctx: &Context, num: ExprId, u: ExprId) -> bool {
    if let Expr::Sub(left, right) = ctx.get(num) {
        let (left, right) = (*left, *right);
        if limit_is_number(ctx, left, 1) {
            if let Some((arg, BuiltinFn::Cos)) = limit_unary_builtin(ctx, right) {
                return compare_expr(ctx, arg, u) == Ordering::Equal;
            }
        }
    }
    false
}

/// `num = a^u − 1` with `a` a positive rational ≠ 1 and the exponent equal to `u`; returns `a`.
fn limit_rational_base_pow_minus_one(ctx: &Context, num: ExprId, u: ExprId) -> Option<ExprId> {
    let Expr::Sub(left, right) = ctx.get(num) else {
        return None;
    };
    let (left, right) = (*left, *right);
    if !limit_is_number(ctx, right, 1) {
        return None;
    }
    let (base, exponent) = as_pow(ctx, left)?;
    if compare_expr(ctx, exponent, u) != Ordering::Equal {
        return None;
    }
    let base_value = as_rational_const(ctx, base, 8)?;
    (base_value.is_positive() && !base_value.is_one()).then_some(base)
}

/// `after == ln(base)` (structurally), confirming the `(a^u − 1)/u → ln(a)` result.
fn limit_after_is_ln_of(ctx: &Context, after: ExprId, base: ExprId) -> bool {
    matches!(limit_unary_builtin(ctx, after), Some((arg, BuiltinFn::Ln))
        if compare_expr(ctx, arg, base) == Ordering::Equal)
}

/// `before = (power of a variable u) · (bounded sin/cos whose argument is a reciprocal in u)`:
/// the squeeze (sandwich) shape `u^k · sin(1/u) → 0`.
fn limit_is_squeeze_product(ctx: &Context, before: ExprId) -> bool {
    let Expr::Mul(left, right) = ctx.get(before) else {
        return false;
    };
    let (left, right) = (*left, *right);
    [(left, right), (right, left)]
        .into_iter()
        .any(|(power, bounded)| {
            limit_power_of_var(ctx, power)
                .is_some_and(|u| limit_is_bounded_reciprocal_oscillator(ctx, bounded, u))
        })
}

/// `expr` is `u` or `u^k` (k a positive integer) for a variable `u`; returns that variable.
fn limit_power_of_var(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    if matches!(ctx.get(expr), Expr::Variable(_)) {
        return Some(expr);
    }
    let (base, exponent) = as_pow(ctx, expr)?;
    let exp_value = as_rational_const(ctx, exponent, 8)?;
    (matches!(ctx.get(base), Expr::Variable(_))
        && exp_value.is_integer()
        && exp_value.is_positive())
    .then_some(base)
}

/// `expr = sin(arg)` or `cos(arg)` where `arg` contains `u` inside a denominator (so the
/// oscillator does not converge at the point — the genuine squeeze case, not mere continuity).
fn limit_is_bounded_reciprocal_oscillator(ctx: &Context, expr: ExprId, u: ExprId) -> bool {
    let Some((arg, builtin)) = limit_unary_builtin(ctx, expr) else {
        return false;
    };
    matches!(builtin, BuiltinFn::Sin | BuiltinFn::Cos) && limit_var_in_denominator(ctx, arg, u)
}

/// True if `u` appears inside a denominator of `expr` (a `Div(_, d)` with `u` in `d`, or a
/// negative power of `u`).
fn limit_var_in_denominator(ctx: &Context, expr: ExprId, u: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Div(numerator, denominator) => {
            let (numerator, denominator) = (*numerator, *denominator);
            limit_expr_contains(ctx, denominator, u) || limit_var_in_denominator(ctx, numerator, u)
        }
        Expr::Pow(base, exponent) => {
            let (base, exponent) = (*base, *exponent);
            as_rational_const(ctx, exponent, 8).is_some_and(|v| v.is_negative())
                && limit_expr_contains(ctx, base, u)
        }
        Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) => {
            let (a, b) = (*a, *b);
            limit_var_in_denominator(ctx, a, u) || limit_var_in_denominator(ctx, b, u)
        }
        Expr::Neg(inner) => limit_var_in_denominator(ctx, *inner, u),
        _ => false,
    }
}

/// True if `expr` references the variable `u` anywhere.
fn limit_expr_contains(ctx: &Context, expr: ExprId, u: ExprId) -> bool {
    if compare_expr(ctx, expr, u) == Ordering::Equal {
        return true;
    }
    match ctx.get(expr) {
        Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::Div(a, b) | Expr::Pow(a, b) => {
            let (a, b) = (*a, *b);
            limit_expr_contains(ctx, a, u) || limit_expr_contains(ctx, b, u)
        }
        Expr::Neg(inner) => limit_expr_contains(ctx, *inner, u),
        Expr::Function(_, args) => args.clone().iter().any(|&a| limit_expr_contains(ctx, a, u)),
        _ => false,
    }
}

/// Narrate a limit at infinity by leading-term DOMINANCE. For a rational `P/Q` as the variable
/// → ∞: lower-degree numerator → 0, equal degrees → ratio of leading coefficients, higher-degree
/// numerator → ±∞. Bare polynomials of degree ≥ 1 → ±∞. Sound by matching `after` to each regime.
fn limit_infinity_dominance(ctx: &Context, before: ExprId, after: ExprId) -> Option<String> {
    let after_value = as_rational_const(ctx, after, 8);
    let after_is_inf = limit_is_infinite(ctx, after);

    if let Some((num, den)) = as_div(ctx, before) {
        let var = limit_single_var_name(ctx, num).or_else(|| limit_single_var_name(ctx, den))?;
        // Cross-class growth dominance `ln(x) ≪ x^a ≪ e^x`: the higher class wins. Tried before the
        // polynomial degree comparison so non-polynomial sides (ln(x), e^x) are classified; same-class
        // quotients (power/power) fall through to the degree comparison below.
        if let (Some(num_class), Some(den_class)) = (
            limit_growth_class(ctx, num, &var),
            limit_growth_class(ctx, den, &var),
        ) {
            if num_class < den_class && after_value.as_ref() == Some(&BigRational::zero()) {
                return Some(format!(
                    "Dominancia: {} crece más despacio que {} (jerarquía ln ≪ potencia ≪ exp), así que el cociente → 0",
                    num_class.name(),
                    den_class.name()
                ));
            }
            if num_class > den_class && after_is_inf {
                return Some(format!(
                    "Dominancia: {} crece más rápido que {} (jerarquía ln ≪ potencia ≪ exp), así que el cociente → ±∞",
                    num_class.name(),
                    den_class.name()
                ));
            }
        }
        let p = Polynomial::from_expr(ctx, num, &var).ok()?;
        let q = Polynomial::from_expr(ctx, den, &var).ok()?;
        return match p.degree().cmp(&q.degree()) {
            Ordering::Less if after_value.as_ref() == Some(&BigRational::zero()) => Some(
                "Dominancia: el denominador tiene mayor grado, así que el cociente → 0".to_string(),
            ),
            Ordering::Equal
                if after_value.as_ref() == Some(&(p.leading_coeff() / q.leading_coeff())) =>
            {
                Some(
                    "Dominancia: grados iguales, el límite es el cociente de los coeficientes líderes"
                        .to_string(),
                )
            }
            Ordering::Greater if after_is_inf => Some(
                "Dominancia: el numerador tiene mayor grado, así que el cociente → ±∞".to_string(),
            ),
            _ => None,
        };
    }

    // A bare polynomial of degree ≥ 1 diverges to ±∞.
    if after_is_inf && limit_is_polynomial(ctx, before) {
        return Some("Dominancia: un polinomio de grado ≥ 1 tiende a ±∞".to_string());
    }

    // Product form `p(x)·e^{q(x)}` with a DECAYING exponential factor (q → −∞) and every other
    // factor sub-exponential (Power/Log/constant): the decay beats polynomial growth → 0. This is
    // the product spelling of the quotient `x²/e^x` handled above.
    if after_value.as_ref() == Some(&BigRational::zero()) {
        if let Some(var) = limit_single_var_name(ctx, before) {
            let factors = expr_nary::mul_leaves(ctx, before);
            let mut has_decaying_exp = false;
            let mut others_sub_exponential = true;
            for &factor in &factors {
                if limit_is_decaying_exponential(ctx, factor, &var) {
                    has_decaying_exp = true;
                } else if as_rational_const(ctx, factor, 8).is_some() {
                    // a constant factor does not change the growth class
                } else if !matches!(
                    limit_growth_class(ctx, factor, &var),
                    Some(LimitGrowthClass::Power | LimitGrowthClass::Log)
                ) {
                    others_sub_exponential = false;
                }
            }
            if has_decaying_exp && others_sub_exponential && factors.len() >= 2 {
                return Some(
                    "Dominancia: la exponencial decae más rápido de lo que crece la potencia, así que el producto → 0"
                        .to_string(),
                );
            }
        }
    }

    None
}

/// `expr` is a DECAYING exponential at `var → ∞`: `e^{q(var)}` with `q(var) → −∞` (a polynomial of
/// degree ≥ 1 with a negative leading coefficient, e.g. `e^{-x}`, `e^{-x²}`).
fn limit_is_decaying_exponential(ctx: &Context, expr: ExprId, var: &str) -> bool {
    let Some(arg) = extract_exp_argument(ctx, expr) else {
        return false;
    };
    Polynomial::from_expr(ctx, arg, var)
        .is_ok_and(|p| p.degree() >= 1 && p.leading_coeff() < BigRational::zero())
}

fn limit_growth_class(ctx: &Context, expr: ExprId, var: &str) -> Option<LimitGrowthClass> {
    // Exponential `e^{p(var)}` with `p(var) → +∞` (polynomial of degree ≥ 1, positive leading
    // coefficient) — the fastest class; checked first since `e^x` matches no other case.
    if let Some(arg) = extract_exp_argument(ctx, expr) {
        let p = Polynomial::from_expr(ctx, arg, var).ok()?;
        return (p.degree() >= 1 && p.leading_coeff() > BigRational::zero())
            .then_some(LimitGrowthClass::Exp);
    }
    // Logarithmic `ln(var)` or a positive-integer power of it — the slowest class.
    if limit_is_log_power(ctx, expr, var) {
        return Some(LimitGrowthClass::Log);
    }
    // Power: a polynomial of degree ≥ 1, or `var^a` with `a > 0` rational (covers `√x`).
    if Polynomial::from_expr(ctx, expr, var).is_ok_and(|p| p.degree() >= 1) {
        return Some(LimitGrowthClass::Power);
    }
    if let Some((base, exponent)) = as_pow(ctx, expr) {
        if is_named_var(ctx, base, var) {
            if let Some(a) = as_rational_const(ctx, exponent, 8) {
                return (a > BigRational::zero()).then_some(LimitGrowthClass::Power);
            }
        }
    }
    // `sqrt(var)` left in function form (the simplifier does not always rewrite it to `var^(1/2)`)
    // is also the Power class.
    if let Some((arg, BuiltinFn::Sqrt)) = limit_unary_builtin(ctx, expr) {
        return is_named_var(ctx, arg, var).then_some(LimitGrowthClass::Power);
    }
    None
}

/// `expr` is `ln(var)` or a positive-integer power of it (`ln(var)^k`, `k ≥ 1`).
fn limit_is_log_power(ctx: &Context, expr: ExprId, var: &str) -> bool {
    let base = match ctx.get(expr) {
        Expr::Pow(b, e) => {
            let positive_integer = as_rational_const(ctx, *e, 8)
                .is_some_and(|k| k.is_integer() && k > BigRational::zero());
            if !positive_integer {
                return false;
            }
            *b
        }
        _ => expr,
    };
    matches!(limit_unary_builtin(ctx, base), Some((arg, BuiltinFn::Ln)) if is_named_var(ctx, arg, var))
}

/// `expr` is `±∞` (`Constant::Infinity` or its negation).
fn limit_is_infinite(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Constant(Constant::Infinity) => true,
        Expr::Neg(inner) => matches!(ctx.get(*inner), Expr::Constant(Constant::Infinity)),
        _ => false,
    }
}

/// The single variable name occurring in `expr`, or `None` if there is not exactly one.
fn limit_single_var_name(ctx: &Context, expr: ExprId) -> Option<String> {
    let mut names = std::collections::BTreeSet::new();
    limit_collect_var_names(ctx, expr, &mut names);
    (names.len() == 1).then(|| names.into_iter().next().unwrap())
}

fn limit_collect_var_names(
    ctx: &Context,
    expr: ExprId,
    names: &mut std::collections::BTreeSet<String>,
) {
    match ctx.get(expr) {
        Expr::Variable(sym) => {
            names.insert(ctx.sym_name(*sym).to_string());
        }
        Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::Div(a, b) | Expr::Pow(a, b) => {
            let (a, b) = (*a, *b);
            limit_collect_var_names(ctx, a, names);
            limit_collect_var_names(ctx, b, names);
        }
        Expr::Neg(inner) => limit_collect_var_names(ctx, *inner, names),
        Expr::Function(_, args) => {
            for arg in args.clone() {
                limit_collect_var_names(ctx, arg, names);
            }
        }
        _ => {}
    }
}

/// `num` and `den` are polynomials in the (single) variable sharing a factor of degree ≥ 1.
fn limit_share_polynomial_factor(ctx: &Context, num: ExprId, den: ExprId) -> bool {
    let Some(var) = limit_single_var_name(ctx, num).or_else(|| limit_single_var_name(ctx, den))
    else {
        return false;
    };
    let (Ok(p), Ok(q)) = (
        Polynomial::from_expr(ctx, num, &var),
        Polynomial::from_expr(ctx, den, &var),
    ) else {
        return false;
    };
    p.gcd(&q).degree() >= 1
}

/// `expr` is a polynomial of degree ≥ 1 in its single variable.
fn limit_is_polynomial(ctx: &Context, expr: ExprId) -> bool {
    let Some(var) = limit_single_var_name(ctx, expr) else {
        return false;
    };
    Polynomial::from_expr(ctx, expr, &var).is_ok_and(|p| p.degree() >= 1)
}

/// `before = (1 + u)^(1/u)` for a variable `u` (the Euler-number limit shape).
fn limit_is_one_plus_to_reciprocal(ctx: &Context, before: ExprId) -> bool {
    let Some((base, exponent)) = as_pow(ctx, before) else {
        return false;
    };
    // exponent = 1/u with u a variable.
    let Some((one, u)) = as_div(ctx, exponent) else {
        return false;
    };
    if !limit_is_number(ctx, one, 1) || !matches!(ctx.get(u), Expr::Variable(_)) {
        return false;
    }
    limit_is_one_plus(ctx, base, u)
}

/// `before` is `(1 + 1/x)^x` with `x` the bare variable (→ ∞): the infinity-side form of the e
/// limit. The exponent must be the bare variable and the base exactly `1 + 1/x` (numerator 1), so
/// `(1 + 2/x)^x → e²` and `(1 + 1/x)^(2x) → e²` decline structurally (and by the result check).
fn limit_is_one_plus_reciprocal_power(ctx: &Context, before: ExprId) -> bool {
    let Some((base, exponent)) = as_pow(ctx, before) else {
        return false;
    };
    if !matches!(ctx.get(exponent), Expr::Variable(_)) {
        return false;
    }
    let Expr::Add(left, right) = *ctx.get(base) else {
        return false;
    };
    let is_one_plus_reciprocal = |constant: ExprId, reciprocal: ExprId| -> bool {
        limit_is_number(ctx, constant, 1)
            && as_div(ctx, reciprocal).is_some_and(|(one, u)| {
                limit_is_number(ctx, one, 1) && compare_expr(ctx, u, exponent) == Ordering::Equal
            })
    };
    is_one_plus_reciprocal(left, right) || is_one_plus_reciprocal(right, left)
}
