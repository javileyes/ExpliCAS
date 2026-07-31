//! `focused_rule_substeps`: familia `polynomial`.
//!
//! Ver la cabecera de `focused_rule_substeps.rs` para el contexto.

use super::*;

pub(super) fn matches_linear_difference(
    ctx: &Context,
    expr: ExprId,
    left_name: &str,
    right_name: &str,
) -> bool {
    match ctx.get(expr) {
        Expr::Sub(left, right) => {
            matches_var_name(ctx, *left, left_name) && matches_var_name(ctx, *right, right_name)
        }
        _ => false,
    }
}

pub(super) fn detect_affine_consecutive_telescoping_sum_pattern(
    ctx: &Context,
    factor1: ExprId,
    factor2: ExprId,
    var: &str,
) -> Option<(ExprId, ExprId, ExprId)> {
    for (base_candidate, other_factor) in [(factor1, factor2), (factor2, factor1)] {
        let coeff = extract_non_unit_affine_var_coeff(ctx, base_candidate, var)?;
        if additive_gap_relation_holds(ctx, base_candidate, coeff, other_factor) {
            return Some((base_candidate, other_factor, coeff));
        }
    }
    None
}

fn extract_non_unit_affine_var_coeff(ctx: &Context, expr: ExprId, var: &str) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Add(left, right) => {
            if !contains_named_var(ctx, *left, var) {
                return extract_non_unit_affine_var_coeff(ctx, *right, var);
            }
            if !contains_named_var(ctx, *right, var) {
                return extract_non_unit_affine_var_coeff(ctx, *left, var);
            }
            None
        }
        Expr::Sub(left, right) => {
            if contains_named_var(ctx, *right, var) {
                return None;
            }
            extract_non_unit_affine_var_coeff(ctx, *left, var)
        }
        _ => extract_non_unit_affine_linear_coeff(ctx, expr, var),
    }
}

fn extract_non_unit_affine_linear_coeff(ctx: &Context, expr: ExprId, var: &str) -> Option<ExprId> {
    if is_named_var(ctx, expr, var) {
        return None;
    }

    let factors = expr_nary::mul_leaves(ctx, expr);
    let mut saw_var = false;
    let mut coeff_factors = Vec::new();

    for factor in factors {
        if is_named_var(ctx, factor, var) {
            if saw_var {
                return None;
            }
            saw_var = true;
        } else if contains_named_var(ctx, factor, var) {
            return None;
        } else {
            coeff_factors.push(factor);
        }
    }

    if !saw_var {
        return None;
    }

    match coeff_factors.as_slice() {
        [] => None,
        [single] => Some(*single),
        _ => None,
    }
}

pub(super) fn generate_polynomial_product_normalize_substeps(
    ctx: &Context,
    step: &Step,
) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Some(plan) = polynomial_product_didactic_plan(ctx, before) else {
        return Vec::new();
    };

    let before_display = display_expr(ctx, before);
    let before_latex = latex_expr(ctx, before);
    let after_display = display_expr(ctx, after);
    let after_latex = latex_expr(ctx, after);

    if plan.expanded_terms > MAX_FULL_POLY_PRODUCT_SUBSTEP_TERMS {
        let summary = if plan.cancelled_degree_groups > 0 {
            "Multiplicar y reagrupar por grados para cancelar términos intermedios"
        } else {
            "Multiplicar y reagrupar por grados"
        };
        let grouped_display = plan
            .grouped_display
            .clone()
            .unwrap_or(plan.expanded_display.clone());
        let grouped_latex = plan
            .grouped_latex
            .clone()
            .unwrap_or(plan.expanded_latex.clone());
        return vec![SubStep::new(summary, grouped_display, after_display)
            .with_before_latex(grouped_latex)
            .with_after_latex(after_latex)];
    }

    let mut out = vec![SubStep::new(
        "Distribuir cada término del producto",
        before_display,
        plan.expanded_display.clone(),
    )
    .with_before_latex(before_latex)
    .with_after_latex(plan.expanded_latex.clone())];

    match (plan.grouped_display.clone(), plan.grouped_latex.clone()) {
        (Some(grouped_display), Some(grouped_latex))
            if grouped_display != plan.expanded_display =>
        {
            out.push(
                SubStep::new(
                    "Agrupar los términos del mismo grado",
                    plan.expanded_display.clone(),
                    grouped_display.clone(),
                )
                .with_before_latex(plan.expanded_latex)
                .with_after_latex(grouped_latex.clone()),
            );

            let finish_title = if plan.cancelled_degree_groups >= 2 {
                "Los términos intermedios se cancelan por parejas"
            } else if plan.cancelled_degree_groups == 1 {
                "Al combinar esos términos, se cancelan"
            } else if plan.repeated_degree_groups > 0 {
                "Sumar los términos del mismo grado"
            } else {
                "Escribir el resultado ya ordenado por grados"
            };

            out.push(
                SubStep::new(finish_title, grouped_display, after_display)
                    .with_before_latex(grouped_latex)
                    .with_after_latex(after_latex),
            );
        }
        _ => {
            let finish_title = if plan.repeated_degree_groups > 0 {
                "Sumar los términos del mismo grado"
            } else {
                "Escribir el resultado ya ordenado por grados"
            };
            out.push(
                SubStep::new(finish_title, plan.expanded_display, after_display)
                    .with_before_latex(plan.expanded_latex)
                    .with_after_latex(after_latex),
            );
        }
    }

    out
}

pub(super) fn polynomial_product_didactic_plan(
    ctx: &Context,
    before: ExprId,
) -> Option<PolyProductDidacticPlan> {
    let Expr::Mul(_, _) = ctx.get(before) else {
        return None;
    };

    let vars = cas_ast::collect_variables(ctx, before);
    if vars.len() != 1 {
        return None;
    }
    let var = vars.iter().next()?.to_string();

    let factors = cas_math::expr_nary::mul_leaves(ctx, before);
    if factors.len() < 2 {
        return None;
    }

    let factor_terms = factors
        .iter()
        .map(|factor| factor_polynomial_terms(ctx, *factor, &var))
        .collect::<Option<Vec<_>>>()?;

    let expanded_terms = expand_polynomial_term_products(&factor_terms);
    if expanded_terms.len() < 2 {
        return None;
    }

    let (expanded_display, expanded_latex) = render_contribution_sum(&var, &expanded_terms);
    let grouped_by_degree = group_contributions_by_degree(&expanded_terms);
    let repeated_degree_groups = grouped_by_degree
        .values()
        .filter(|group| group.len() > 1)
        .count();
    let cancelled_degree_groups = grouped_by_degree
        .values()
        .filter(|group| group.len() > 1 && contribution_group_sum(group).is_zero())
        .count();

    let (grouped_display, grouped_latex) = if repeated_degree_groups > 0 {
        let (display, latex) = render_grouped_contributions(&var, &grouped_by_degree);
        (Some(display), Some(latex))
    } else {
        (None, None)
    };

    Some(PolyProductDidacticPlan {
        expanded_display,
        expanded_latex,
        grouped_display,
        grouped_latex,
        expanded_terms: expanded_terms.len(),
        repeated_degree_groups,
        cancelled_degree_groups,
    })
}

fn group_contributions_by_degree(
    contributions: &[PolyContribution],
) -> BTreeMap<usize, Vec<PolyContribution>> {
    let mut out: BTreeMap<usize, Vec<PolyContribution>> = BTreeMap::new();
    for contribution in contributions {
        out.entry(contribution.degree)
            .or_default()
            .push(contribution.clone());
    }
    out
}

pub(super) fn build_signed_monomial_expr(
    ctx: &mut Context,
    var: &str,
    term: &PolyContribution,
) -> ExprId {
    let abs = term.coeff.abs();
    let unsigned = build_unsigned_monomial_expr(ctx, var, &abs, term.degree);
    if term.coeff.is_negative() {
        ctx.add_raw(Expr::Neg(unsigned))
    } else {
        unsigned
    }
}

pub(super) fn build_unsigned_monomial_expr(
    ctx: &mut Context,
    var: &str,
    coeff: &BigRational,
    degree: usize,
) -> ExprId {
    if degree == 0 {
        return ctx.add(Expr::Number(coeff.clone()));
    }

    let var_expr = ctx.var(var);
    let power_expr = if degree == 1 {
        var_expr
    } else {
        let exp = ctx.num(degree as i64);
        ctx.add(Expr::Pow(var_expr, exp))
    };

    if coeff == &BigRational::from_integer(1.into()) {
        power_expr
    } else {
        let coeff_expr = ctx.add(Expr::Number(coeff.clone()));
        ctx.add(Expr::Mul(coeff_expr, power_expr))
    }
}

pub(super) fn generate_polynomial_identity_exact_cancel_substeps(
    ctx: &Context,
    step: &Step,
) -> Vec<SubStep> {
    if step
        .description
        .to_ascii_lowercase()
        .contains("opaque substitution")
        && is_zero(ctx, step.after_local().unwrap_or(step.after))
    {
        let before = step.before_local().unwrap_or(step.before);
        let after = step.after_local().unwrap_or(step.after);
        return vec![SubStep::new(
            "Las dos partes se compensan exactamente",
            display_expr(ctx, before),
            display_expr(ctx, after),
        )
        .with_before_latex(latex_expr(ctx, before))
        .with_after_latex(latex_expr(ctx, after))];
    }

    if let Some(proof) = step.poly_proof() {
        if !proof.opaque_substitutions.is_empty()
            && is_zero(ctx, step.after_local().unwrap_or(step.after))
        {
            let before = step.before_local().unwrap_or(step.before);
            let after = step.after_local().unwrap_or(step.after);
            return vec![SubStep::new(
                "Las dos partes se compensan exactamente",
                display_expr(ctx, before),
                display_expr(ctx, after),
            )
            .with_before_latex(latex_expr(ctx, before))
            .with_after_latex(latex_expr(ctx, after))];
        }
        return Vec::new();
    }

    let before = step.before_local().unwrap_or(step.before);
    let Some((left, right)) =
        difference_like_terms(ctx, before).or_else(|| difference_like_terms(ctx, step.before))
    else {
        return Vec::new();
    };

    let identity_substeps = generate_identity_equivalence_substeps(ctx, left, right);
    if !identity_substeps.is_empty() {
        return identity_substeps;
    }

    vec![SubStep::new(
        "Las dos partes representan la misma cantidad",
        display_expr(ctx, left),
        display_expr(ctx, right),
    )
    .with_before_latex(latex_expr(ctx, left))
    .with_after_latex(latex_expr(ctx, right))]
}

pub(super) fn generate_basic_polynomial_integration_substeps(
    ctx: &Context,
    step: &Step,
) -> Vec<SubStep> {
    if step.rule_name != "Symbolic Integration" {
        return Vec::new();
    }

    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    if let Some(substep) = nonfinite_or_undefined_integration_substep(ctx, before, after) {
        return vec![substep];
    }

    let Expr::Function(fn_id, args) = ctx.get(before) else {
        return Vec::new();
    };
    if ctx.sym_name(*fn_id) != "integrate" || args.len() != 2 {
        return Vec::new();
    }
    if matches!(
        ctx.get(after),
        Expr::Function(after_fn_id, after_args)
            if ctx.sym_name(*after_fn_id) == "integrate" && after_args.len() == 2
    ) {
        return Vec::new();
    }

    let Expr::Variable(var_sym) = ctx.get(args[1]) else {
        return Vec::new();
    };
    let var_name = ctx.sym_name(*var_sym);
    if Polynomial::from_expr(ctx, args[0], var_name).is_err() {
        return Vec::new();
    }

    let terms = AddView::from_expr(ctx, args[0]).terms;
    let mut substeps = Vec::new();
    if terms.len() > 1 {
        let linearity_display = integral_sum_display(ctx, terms.as_slice(), var_name);
        let linearity_latex = integral_sum_latex(ctx, terms.as_slice(), var_name);
        substeps.push(
            SubStep::keyed(
                "integral.use_linearity",
                vec![],
                display_expr(ctx, args[0]),
                linearity_display.clone(),
            )
            .with_before_latex(latex_expr(ctx, args[0]))
            .with_after_latex(linearity_latex.clone()),
        );
        substeps.push(
            SubStep::keyed(
                "integral.integrate_each_term",
                vec![],
                linearity_display,
                display_expr(ctx, after),
            )
            .with_before_latex(linearity_latex)
            .with_after_latex(latex_expr(ctx, after)),
        );
    } else {
        substeps.push(
            SubStep::new(
                polynomial_integration_rule_title(ctx, args[0], var_name),
                display_expr(ctx, args[0]),
                display_expr(ctx, after),
            )
            .with_before_latex(latex_expr(ctx, args[0]))
            .with_after_latex(latex_expr(ctx, after)),
        );
    }

    substeps
}

fn polynomial_integration_rule_title(
    ctx: &Context,
    integrand: ExprId,
    var_name: &str,
) -> &'static str {
    if !contains_named_var(ctx, integrand, var_name) {
        "Integrar una constante"
    } else {
        "Usar regla de potencia para integrales"
    }
}

/// Narrate one integration-by-parts application for the
/// `polynomial(linear) * {exp, sin, cos, sinh, cosh}(affine)` family, mirroring
/// `generate_polynomial_affine_log_by_parts_substeps` but with the OTHER u/dv
/// assignment: `u = polynomial`, `dv = elementary factor` (the log narrator
/// keeps `u = ln`). `v` is the antiderivative of the elementary factor and
/// `du = p'(x)`. Presentation only -- the integration result is untouched.
/// Returns an empty trace (graceful no-op) for the ln family (owned by the log
/// narrator), the repeated degree>=2 case (kept title-only), and anything whose
/// antiderivative or derivative is unavailable, so a trace is never corrupted.
pub(super) fn generate_polynomial_elementary_by_parts_substeps(
    ctx: &Context,
    integrand: ExprId,
    after: ExprId,
    var_name: &str,
) -> Vec<SubStep> {
    let Some((left, right)) = as_mul(ctx, integrand) else {
        return Vec::new();
    };
    let Some((u_factor, dv_factor)) = linear_times_elementary_factors(ctx, left, right, var_name)
    else {
        return Vec::new();
    };

    let mut scratch = ctx.clone();
    let Some(v_expr) = cas_math::symbolic_integration_support::integrate_symbolic_expr(
        &mut scratch,
        dv_factor,
        var_name,
    ) else {
        return Vec::new();
    };
    let v_expr = simplify_expr_in_context(&mut scratch, v_expr);
    let Some(du_expr) = cas_math::symbolic_differentiation_support::differentiate_symbolic_expr(
        &mut scratch,
        u_factor,
        var_name,
    ) else {
        return Vec::new();
    };
    let du_expr = simplify_expr_in_context(&mut scratch, du_expr);

    let u_display = display_expr(&scratch, u_factor);
    let u_latex = latex_expr(&scratch, u_factor);
    let v_display = display_expr(&scratch, v_expr);
    let v_latex = latex_expr(&scratch, v_expr);
    let du_display = display_expr(&scratch, du_expr);
    let du_latex = latex_expr(&scratch, du_expr);
    let dv_display = format!("{} dx", display_expr(&scratch, dv_factor));
    let dv_latex = format!("{}\\,dx", latex_expr(&scratch, dv_factor));

    let u_display_factor = group_display_for_product(&u_display);
    let u_latex_factor = group_latex_for_product(&u_latex);
    let v_display_factor = group_display_for_product(&v_display);
    let v_latex_factor = group_latex_for_product(&v_latex);
    let choice_display = format!("u = {}, dv = {}", u_display, dv_display);
    let choice_latex = format!("u = {},\\; dv = {}", u_latex, dv_latex);

    // The remaining integral is `int v du`; drop the redundant `* 1` when du = 1.
    let (remaining_display, remaining_latex) = if du_display == "1" {
        (v_display_factor.clone(), v_latex_factor.clone())
    } else {
        (
            format!(
                "{}·{}",
                v_display_factor,
                group_display_for_product(&du_display)
            ),
            format!(
                "{}\\cdot {}",
                v_latex_factor,
                group_latex_for_product(&du_latex)
            ),
        )
    };

    vec![
        SubStep::keyed(
            "by_parts.choose_u_dv",
            vec![],
            display_expr(&scratch, integrand),
            choice_display.clone(),
        )
        .with_before_latex(latex_expr(&scratch, integrand))
        .with_after_latex(choice_latex.clone()),
        SubStep::keyed(
            "by_parts.compute_du_v",
            vec![],
            choice_display,
            format!("du = {} dx, v = {}", du_display, v_display),
        )
        .with_before_latex(choice_latex)
        .with_after_latex(format!("du = {}\\,dx,\\; v = {}", du_latex, v_latex)),
        SubStep::keyed(
            "by_parts.apply_formula",
            vec![],
            format!(
                "{}·{} - integrate({}, {})",
                u_display_factor, v_display_factor, remaining_display, var_name
            ),
            display_expr(&scratch, after),
        )
        .with_before_latex(format!(
            "{}\\cdot {} - \\int {}\\,d{}",
            u_latex_factor, v_latex_factor, remaining_latex, var_name
        ))
        .with_after_latex(latex_expr(&scratch, after)),
    ]
}

/// Narrate the REPEATED integration-by-parts reductions for
/// `p(x) * {exp, sin, cos, sinh, cosh}` with `deg p >= 2` (e.g. `x^2 e^x`),
/// which the engine integrates by the closed-form tabular method (iterated
/// derivatives with alternating signs -- exactly the result of applying by-parts
/// `deg p` times). The deg>=2 case previously stayed title-only ("Usar
/// integración por partes repetida"); this unrolls each application, mirroring
/// `generate_polynomial_elementary_by_parts_substeps`: at level k it chooses
/// `u = p_k`, `dv = e_k dx`, computes `du = p_k'`, `v = integral e_k`, and shows
/// `integral p_k e_k = p_k v - integral v p_k'`, where the remaining integral is
/// the next level's integrand. The polynomial degree drops by one each level
/// until it is a constant, whose remaining elementary integral closes into the
/// final antiderivative. Presentation only -- the integration RESULT is
/// untouched. Empty trace (graceful no-op) for degree <= 1 (owned by the linear
/// narrator), the ln family, or when an intermediate antiderivative/derivative
/// is unavailable, so a trace is never corrupted.
pub(super) fn generate_repeated_polynomial_elementary_by_parts_substeps(
    ctx: &Context,
    integrand: ExprId,
    after: ExprId,
    var_name: &str,
) -> Vec<SubStep> {
    let Some((poly_factor, elem_factor)) =
        repeated_polynomial_times_elementary_factors(ctx, integrand, var_name)
    else {
        return Vec::new();
    };

    let mut scratch = ctx.clone();
    let mut substeps: Vec<SubStep> = Vec::new();

    let mut current_poly = poly_factor;
    let mut current_elem = elem_factor;
    let mut core_display = display_expr(&scratch, integrand);
    let mut core_latex = latex_expr(&scratch, integrand);
    // One entry per by-parts application: the boundary piece `u_k·v_k` as
    // (display, latex, node). The recomposition needs them because the chain's
    // identity is `∫p·e = u0·v0 − u1·v1 + u2·v2 − … ± ∫(last core)` — each
    // level's apply_formula narrates only its LOCAL identity, so without the
    // accumulated pieces the closing sub-step cannot state anything about the
    // engine's final answer without lying (which is exactly what it used to do:
    // the audit's `∫−2·sin(x)dx ⟹ 2x·sin(x) + (2−x²)·cos(x)`).
    let mut outer_terms: Vec<(String, String, ExprId)> = Vec::new();

    // The degree strictly decreases each iteration, so this terminates; the cap
    // is a defensive backstop matching the engine's maximum supported degree.
    for _ in 0..=(MAX_REPEATED_BY_PARTS_NARRATION_LEVELS) {
        let Ok(poly) = Polynomial::from_expr(&scratch, current_poly, var_name) else {
            return Vec::new();
        };
        if poly.degree() == 0 {
            // current_poly is a constant: the remaining elementary integral is
            // `∫ current_elem · current_poly`. The closer integrates ITS OWN
            // integrand — the claim is checked, so a wrong table answer
            // declines instead of publishing — and a separate RECOMPOSITION
            // sub-step assembles the boundary pieces into the engine's answer,
            // gated on exact equality. On any decline the per-level narration
            // above stays (it is true on its own); only the closers are
            // withheld.
            let remaining_node = scratch.add(Expr::Mul(current_elem, current_poly));
            let Some(remaining_antiderivative) =
                cas_math::symbolic_integration_support::integrate_symbolic_expr(
                    &mut scratch,
                    remaining_node,
                    var_name,
                )
            else {
                return substeps;
            };
            let remaining_antiderivative =
                simplify_expr_in_context(&mut scratch, remaining_antiderivative);
            let Some(closer) = SubStep::checked(
                &scratch,
                crate::didactic::substep::claim::Claim::Antiderivative {
                    var: var_name.to_string(),
                },
                remaining_node,
                remaining_antiderivative,
                "by_parts.integrate_remaining",
                vec![],
                format!("integrate({}, {})", core_display, var_name),
                display_expr(&scratch, remaining_antiderivative),
            ) else {
                return substeps;
            };
            substeps.push(
                closer
                    .with_before_latex(format!("\\int {}\\,d{}", core_latex, var_name))
                    .with_after_latex(latex_expr(&scratch, remaining_antiderivative)),
            );

            if outer_terms.is_empty() {
                return substeps;
            }
            // `∫p·e = u0·v0 − u1·v1 + u2·v2 − … + (−1)^n·F`, n = applications.
            let mut assembled_node = outer_terms[0].2;
            let mut assembled_display = outer_terms[0].0.clone();
            let mut assembled_latex = outer_terms[0].1.clone();
            for (k, (term_display, term_latex, term_node)) in outer_terms.iter().enumerate().skip(1)
            {
                assembled_node = if k % 2 == 1 {
                    scratch.add(Expr::Sub(assembled_node, *term_node))
                } else {
                    scratch.add(Expr::Add(assembled_node, *term_node))
                };
                let sign = if k % 2 == 1 { " - " } else { " + " };
                assembled_display = format!("{assembled_display}{sign}{term_display}");
                assembled_latex = format!("{assembled_latex}{sign}{term_latex}");
            }
            let last_negated = outer_terms.len() % 2 == 1;
            assembled_node = if last_negated {
                scratch.add(Expr::Sub(assembled_node, remaining_antiderivative))
            } else {
                scratch.add(Expr::Add(assembled_node, remaining_antiderivative))
            };
            let closing_sign = if last_negated { " - " } else { " + " };
            let closing_display =
                group_display_for_product(&display_expr(&scratch, remaining_antiderivative));
            let closing_latex =
                group_latex_for_product(&latex_expr(&scratch, remaining_antiderivative));
            assembled_display = format!("{assembled_display}{closing_sign}{closing_display}");
            assembled_latex = format!("{assembled_latex}{closing_sign}{closing_latex}");

            // STRICTLY gated on a proved equality, like the C3.1 linearity
            // narrator: this sub-step claims to reproduce the ENGINE's answer
            // from the assembled pieces, and an unproved "A = B" about the
            // final answer is not narration this closer is entitled to. The
            // difference is EXPANDED before folding — the engine publishes its
            // antiderivative factored (`(2 − x²)·cos(x)`), and the default
            // fold does not distribute, so a plain `decide_equality` leaves
            // true recompositions undecided. Exact ZERO is required, not just
            // a constant: two antiderivatives may differ by a constant, but
            // the displayed pair asserts equality, and a non-zero offset would
            // make it a lie.
            // The engine's `after` arrives `__hold`-wrapped, and a hold is
            // frozen for every rewriter — expand and simplify both leave
            // `Sub(x, hold(y))` intact, so the proof must run on the unwrapped
            // node (same dance as the linearity narrator above).
            let mut compare_after = after;
            loop {
                let unwrapped = cas_ast::hold::unwrap_internal_hold(&scratch, compare_after);
                if unwrapped == compare_after {
                    break;
                }
                compare_after = unwrapped;
            }
            let difference = scratch.add(Expr::Sub(assembled_node, compare_after));
            let expanded = cas_math::expand_ops::expand(&mut scratch, difference);
            let expanded = deep_readd_canonical(&mut scratch, expanded);
            let residual = simplify_expr_in_context(&mut scratch, expanded);
            let recomposition_proved = matches!(
                scratch.get(residual),
                Expr::Number(n) if num_traits::Zero::is_zero(n)
            );
            if recomposition_proved {
                substeps.push(
                    SubStep::keyed(
                        "by_parts.recompose",
                        vec![],
                        assembled_display,
                        display_expr(&scratch, after),
                    )
                    .with_before_latex(assembled_latex)
                    .with_after_latex(latex_expr(&scratch, after)),
                );
            }
            return substeps;
        }

        let Some(v_expr) = cas_math::symbolic_integration_support::integrate_symbolic_expr(
            &mut scratch,
            current_elem,
            var_name,
        ) else {
            return Vec::new();
        };
        let v_expr = simplify_expr_in_context(&mut scratch, v_expr);
        let Some(du_expr) = cas_math::symbolic_differentiation_support::differentiate_symbolic_expr(
            &mut scratch,
            current_poly,
            var_name,
        ) else {
            return Vec::new();
        };
        let du_expr = simplify_expr_in_context(&mut scratch, du_expr);

        let u_display = display_expr(&scratch, current_poly);
        let u_latex = latex_expr(&scratch, current_poly);
        let elem_display = display_expr(&scratch, current_elem);
        let elem_latex = latex_expr(&scratch, current_elem);
        let v_display = display_expr(&scratch, v_expr);
        let v_latex = latex_expr(&scratch, v_expr);
        let du_display = display_expr(&scratch, du_expr);
        let du_latex = latex_expr(&scratch, du_expr);

        let u_factor = group_display_for_product(&u_display);
        let u_latex_factor = group_latex_for_product(&u_latex);
        let v_factor = group_display_for_product(&v_display);
        let v_latex_factor = group_latex_for_product(&v_latex);
        let du_factor = group_display_for_product(&du_display);
        let du_latex_factor = group_latex_for_product(&du_latex);

        let choice_display = format!("u = {}, dv = {} dx", u_display, elem_display);
        let choice_latex = format!("u = {},\\; dv = {}\\,dx", u_latex, elem_latex);

        // Remaining integral after this application: integral v * p_k'.
        let remaining_display = format!("{}·{}", v_factor, du_factor);
        let remaining_latex = format!("{}\\cdot {}", v_latex_factor, du_latex_factor);

        // The boundary piece `u_k·v_k` this application leaves behind, for the
        // closing recomposition.
        outer_terms.push((
            format!("{}·{}", u_factor, v_factor),
            format!("{}\\cdot {}", u_latex_factor, v_latex_factor),
            scratch.add(Expr::Mul(current_poly, v_expr)),
        ));

        substeps.push(
            SubStep::keyed(
                "by_parts.choose_u_dv",
                vec![],
                core_display.clone(),
                choice_display.clone(),
            )
            .with_before_latex(core_latex.clone())
            .with_after_latex(choice_latex.clone()),
        );
        substeps.push(
            SubStep::keyed(
                "by_parts.compute_du_v",
                vec![],
                choice_display,
                // Group du so a multi-term derivative (e.g. 4x - 3) reads as
                // `(4·x - 3) dx`, not the ambiguous `4·x - 3 dx`. A single-term
                // du (constant or monomial) stays bare.
                format!("du = {} dx, v = {}", du_factor, v_display),
            )
            .with_before_latex(choice_latex)
            .with_after_latex(format!("du = {}\\,dx,\\; v = {}", du_latex_factor, v_latex)),
        );
        substeps.push(
            SubStep::keyed(
                "by_parts.apply_formula",
                vec![],
                format!("integrate({}, {})", core_display, var_name),
                format!(
                    "{}·{} - integrate({}, {})",
                    u_factor, v_factor, remaining_display, var_name
                ),
            )
            .with_before_latex(format!("\\int {}\\,d{}", core_latex, var_name))
            .with_after_latex(format!(
                "{}\\cdot {} - \\int {}\\,d{}",
                u_latex_factor, v_latex_factor, remaining_latex, var_name
            )),
        );

        // Descend: the remaining integral integral v * p_k' is the next level's
        // integrand, with u <- p_k' (one degree lower) and dv <- v.
        core_display = remaining_display;
        core_latex = remaining_latex;
        current_poly = du_expr;
        current_elem = v_expr;
    }

    // Degree never collapsed within the cap (should be unreachable for the gated
    // family): abandon the partial trace rather than emit a truncated one.
    Vec::new()
}

pub(super) fn oriented_linear_times_elementary(
    ctx: &Context,
    poly_candidate: ExprId,
    elem_candidate: ExprId,
    var_name: &str,
) -> Option<(ExprId, ExprId)> {
    // u must be a genuine degree-1 polynomial (defer the repeated degree>=2 case).
    let poly = Polynomial::from_expr(ctx, poly_candidate, var_name).ok()?;
    if poly.degree() != 1 {
        return None;
    }
    // dv must be the transcendental factor: not a polynomial, and not a logarithm.
    if Polynomial::from_expr(ctx, elem_candidate, var_name).is_ok() {
        return None;
    }
    if let Expr::Function(fn_id, _) = ctx.get(elem_candidate) {
        if ctx.is_builtin(*fn_id, BuiltinFn::Ln) {
            return None;
        }
    }
    Some((poly_candidate, elem_candidate))
}

pub(super) fn oriented_repeated_polynomial_times_elementary(
    ctx: &Context,
    poly_candidate: ExprId,
    elem_candidate: ExprId,
    var_name: &str,
) -> Option<(ExprId, ExprId)> {
    // u must be a genuine degree >= 2 polynomial (the degree-1 case is owned by
    // generate_polynomial_elementary_by_parts_substeps).
    let poly = Polynomial::from_expr(ctx, poly_candidate, var_name).ok()?;
    if poly.degree() < 2 {
        return None;
    }
    // dv must be the transcendental factor: not a polynomial, and not a logarithm.
    if Polynomial::from_expr(ctx, elem_candidate, var_name).is_ok() {
        return None;
    }
    if let Expr::Function(fn_id, _) = ctx.get(elem_candidate) {
        if ctx.is_builtin(*fn_id, BuiltinFn::Ln) {
            return None;
        }
    }
    Some((poly_candidate, elem_candidate))
}

pub(super) fn is_affine_in_var(ctx: &Context, expr: ExprId, var_name: &str) -> bool {
    let Ok(poly) = Polynomial::from_expr(ctx, expr, var_name) else {
        return false;
    };
    let mut has_linear_term = false;
    for (degree, coefficient) in poly.coeffs.iter().enumerate() {
        if coefficient.is_zero() {
            continue;
        }
        if degree > 1 {
            return false;
        }
        if degree == 1 {
            has_linear_term = true;
        }
    }
    has_linear_term
}

fn is_symbolic_affine_in_var(ctx: &Context, expr: ExprId, var_name: &str) -> bool {
    if is_affine_in_var(ctx, expr, var_name) {
        return true;
    }
    symbolic_linear_contains_var(ctx, expr, var_name).unwrap_or(false)
}

fn symbolic_linear_contains_var(ctx: &Context, expr: ExprId, var_name: &str) -> Option<bool> {
    if !contains_named_var(ctx, expr, var_name) {
        return Some(false);
    }

    match ctx.get(expr) {
        Expr::Variable(sym_id) if ctx.sym_name(*sym_id) == var_name => Some(true),
        Expr::Add(left, right) | Expr::Sub(left, right) => {
            let left_has_var = symbolic_linear_contains_var(ctx, *left, var_name)?;
            let right_has_var = symbolic_linear_contains_var(ctx, *right, var_name)?;
            Some(left_has_var || right_has_var)
        }
        Expr::Mul(left, right) => {
            let left_has_var = contains_named_var(ctx, *left, var_name);
            let right_has_var = contains_named_var(ctx, *right, var_name);
            match (left_has_var, right_has_var) {
                (true, true) => None,
                (true, false) => symbolic_linear_contains_var(ctx, *left, var_name),
                (false, true) => symbolic_linear_contains_var(ctx, *right, var_name),
                (false, false) => Some(false),
            }
        }
        Expr::Div(num, den) => {
            if contains_named_var(ctx, *den, var_name) {
                return None;
            }
            symbolic_linear_contains_var(ctx, *num, var_name)
        }
        Expr::Neg(inner) | Expr::Hold(inner) => symbolic_linear_contains_var(ctx, *inner, var_name),
        Expr::Function(_, _) | Expr::Pow(_, _) | Expr::Matrix { .. } => None,
        Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => None,
    }
}

pub(super) fn contains_linear_integration_by_parts_target(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
) -> bool {
    let mut scratch = ctx.clone();
    if cas_math::symbolic_integration_support::integrate_symbolic_is_linear_times_exp_linear_target(
        &mut scratch,
        expr,
        var_name,
    ) || cas_math::symbolic_integration_support::integrate_symbolic_is_linear_times_trig_linear_target(
        &mut scratch,
        expr,
        var_name,
    ) || cas_math::symbolic_integration_support::integrate_symbolic_is_linear_times_hyperbolic_linear_target(
        &mut scratch,
        expr,
        var_name,
    ) || cas_math::symbolic_integration_support::integrate_symbolic_is_polynomial_times_hyperbolic_linear_target(
        &mut scratch,
        expr,
        var_name,
    ) || cas_math::symbolic_integration_support::integrate_symbolic_is_monomial_times_ln_var_by_parts_target(
        &mut scratch,
        expr,
        var_name,
    ) || cas_math::symbolic_integration_support::integrate_symbolic_is_linear_times_affine_ln_by_parts_target(
        &mut scratch,
        expr,
        var_name,
    ) || cas_math::symbolic_integration_support::integrate_symbolic_is_quadratic_times_affine_ln_by_parts_target(
        &mut scratch,
        expr,
        var_name,
    ) || (!(cas_math::symbolic_integration_support::integrate_symbolic_is_log_product_substitution_target(
        &mut scratch,
        expr,
        var_name,
    ) || cas_math::symbolic_integration_support::integrate_symbolic_is_log_power_product_substitution_target(
        &mut scratch,
        expr,
        var_name,
    )) && cas_math::symbolic_integration_support::integrate_symbolic_is_quadratic_times_positive_quadratic_ln_by_parts_target(
        &mut scratch,
        expr,
        var_name,
    )) || cas_math::symbolic_integration_support::integrate_symbolic_is_bounded_inverse_trig_variable_target(
        &mut scratch,
        expr,
        var_name,
    ) || cas_math::symbolic_integration_support::integrate_symbolic_is_arctan_scaled_variable_target(
        &mut scratch,
        expr,
        var_name,
    ) || cas_math::symbolic_integration_support::integrate_symbolic_is_arctan_reciprocal_affine_variable_target(
        &mut scratch,
        expr,
        var_name,
    ) || cas_math::symbolic_integration_support::integrate_symbolic_is_asinh_affine_variable_target(
        &mut scratch,
        expr,
        var_name,
    ) || cas_math::symbolic_integration_support::integrate_symbolic_is_atanh_affine_variable_target(
        &mut scratch,
        expr,
        var_name,
    ) || cas_math::symbolic_integration_support::integrate_symbolic_is_acosh_affine_variable_target(
        &mut scratch,
        expr,
        var_name,
    ) {
        return true;
    }

    match ctx.get(expr) {
        // A bare `ln(affine)` integrates by parts with u = ln, dv = dx; the
        // single-inverse narrator owns the u/dv trace. (A polynomial * ln is
        // already claimed by the log-by-parts target predicates above.)
        Expr::Function(fn_id, args)
            if args.len() == 1
                && ctx.is_builtin(*fn_id, BuiltinFn::Ln)
                && is_affine_in_var(ctx, args[0], var_name) =>
        {
            true
        }
        Expr::Add(left, right) | Expr::Sub(left, right) => {
            contains_linear_integration_by_parts_target(ctx, *left, var_name)
                || contains_linear_integration_by_parts_target(ctx, *right, var_name)
        }
        Expr::Neg(inner) => contains_linear_integration_by_parts_target(ctx, *inner, var_name),
        _ => false,
    }
}

pub(super) fn generate_linear_inverse_table_integration_substeps(
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
    let Some((builtin, arg, rational_scale, has_symbolic_external_scale)) =
        linear_inverse_table_result_arg(ctx, after, var_name)
    else {
        return Vec::new();
    };

    let title = match builtin {
        BuiltinFn::Arcsin | BuiltinFn::Asin => "Usar la regla de arcsin con derivada interna",
        BuiltinFn::Arctan | BuiltinFn::Atan => "Usar la regla de arctan con derivada interna",
        BuiltinFn::Asinh => "Usar la regla de asinh con derivada interna",
        BuiltinFn::Acosh => "Usar la regla de acosh con derivada interna",
        BuiltinFn::Atanh => "Usar la regla de atanh con derivada interna",
        _ => return Vec::new(),
    };

    let mut substeps = Vec::new();
    if let Some(step) = checked_antiderivative_substep(ctx, title, args[0], after, var_name) {
        substeps.push(step);
    }

    if nontrivial_affine_argument(ctx, arg, var_name) {
        substeps.push(
            SubStep::keyed(
                "usub.identify_affine_argument",
                vec![],
                display_expr(ctx, arg),
                display_expr(ctx, after),
            )
            .with_before_latex(latex_expr(ctx, arg))
            .with_after_latex(latex_expr(ctx, after)),
        );
    }

    if has_symbolic_external_scale || rational_scale != BigRational::one() {
        substeps.push(
            SubStep::keyed(
                "usub.adjust_constant_factor",
                vec![],
                inverse_table_function_display(ctx, builtin, arg),
                display_expr(ctx, after),
            )
            .with_before_latex(inverse_table_function_latex(ctx, builtin, arg))
            .with_after_latex(latex_expr(ctx, after)),
        );
    }

    substeps
}

fn linear_inverse_table_result_arg(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(BuiltinFn, ExprId, BigRational, bool)> {
    // Unwrap holds to a fixpoint: algorithmic-backend results arrive
    // double-held (result preservation plus the backend summary wrap), and
    // the second level is in Function(__hold) form, which a single unwrap
    // misses - silencing the educational substeps for those families.
    let mut expr = expr;
    loop {
        let unwrapped = cas_ast::hold::unwrap_internal_hold(ctx, expr);
        if unwrapped == expr {
            break;
        }
        expr = unwrapped;
    }
    match ctx.get(expr) {
        Expr::Function(fn_id, args) if args.len() == 1 => {
            let builtin = ctx.builtin_of(*fn_id)?;
            if !matches!(
                builtin,
                BuiltinFn::Arctan
                    | BuiltinFn::Atan
                    | BuiltinFn::Arcsin
                    | BuiltinFn::Asin
                    | BuiltinFn::Asinh
                    | BuiltinFn::Acosh
                    | BuiltinFn::Atanh
            ) {
                return None;
            }
            is_scaled_affine_inverse_table_arg(ctx, args[0], var_name).then_some((
                builtin,
                args[0],
                BigRational::one(),
                false,
            ))
        }
        Expr::Neg(inner) => linear_inverse_table_result_arg(ctx, *inner, var_name).map(
            |(builtin, arg, rational_scale, has_symbolic_external_scale)| {
                (builtin, arg, -rational_scale, has_symbolic_external_scale)
            },
        ),
        Expr::Hold(inner) => linear_inverse_table_result_arg(ctx, *inner, var_name),
        Expr::Mul(left, right) => {
            if let Some(scale) = as_rational_const(ctx, *left, 8) {
                return linear_inverse_table_result_arg(ctx, *right, var_name).map(
                    |(builtin, arg, rational_scale, has_symbolic_external_scale)| {
                        (
                            builtin,
                            arg,
                            scale * rational_scale,
                            has_symbolic_external_scale,
                        )
                    },
                );
            }
            if let Some(scale) = as_rational_const(ctx, *right, 8) {
                return linear_inverse_table_result_arg(ctx, *left, var_name).map(
                    |(builtin, arg, rational_scale, has_symbolic_external_scale)| {
                        (
                            builtin,
                            arg,
                            scale * rational_scale,
                            has_symbolic_external_scale,
                        )
                    },
                );
            }
            if !contains_named_var(ctx, *left, var_name) {
                return linear_inverse_table_result_arg(ctx, *right, var_name)
                    .map(|(builtin, arg, rational_scale, _)| (builtin, arg, rational_scale, true));
            }
            if !contains_named_var(ctx, *right, var_name) {
                return linear_inverse_table_result_arg(ctx, *left, var_name)
                    .map(|(builtin, arg, rational_scale, _)| (builtin, arg, rational_scale, true));
            }
            None
        }
        Expr::Div(num, den) => {
            if let Some(denominator) = as_rational_const(ctx, *den, 8) {
                if denominator.is_zero() {
                    return None;
                }
                return linear_inverse_table_result_arg(ctx, *num, var_name).map(
                    |(builtin, arg, rational_scale, has_symbolic_external_scale)| {
                        (
                            builtin,
                            arg,
                            rational_scale / denominator,
                            has_symbolic_external_scale,
                        )
                    },
                );
            }
            if contains_named_var(ctx, *den, var_name) {
                return None;
            }
            linear_inverse_table_result_arg(ctx, *num, var_name)
                .map(|(builtin, arg, rational_scale, _)| (builtin, arg, rational_scale, true))
        }
        _ => None,
    }
}

fn is_scaled_affine_inverse_table_arg(ctx: &Context, expr: ExprId, var_name: &str) -> bool {
    if is_symbolic_affine_in_var(ctx, expr, var_name) {
        return true;
    }

    let Expr::Div(num, den) = ctx.get(expr) else {
        return false;
    };
    is_symbolic_affine_in_var(ctx, *num, var_name) && !contains_named_var(ctx, *den, var_name)
}

pub(super) fn nontrivial_affine_argument(ctx: &Context, arg: ExprId, var_name: &str) -> bool {
    if !is_named_var_expr(ctx, arg, var_name) && is_symbolic_affine_in_var(ctx, arg, var_name) {
        return true;
    }

    let Ok(poly) = Polynomial::from_expr(ctx, arg, var_name) else {
        return false;
    };
    if poly.degree() != 1 {
        return false;
    }

    let slope = poly
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let offset = poly
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    !slope.is_zero() && (slope != BigRational::one() || !offset.is_zero())
}

pub(super) fn generate_linear_elementary_table_integration_substeps(
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
    let Some((builtin, arg)) = linear_elementary_integrand_arg(ctx, args[0]) else {
        return Vec::new();
    };
    let Ok(arg_poly) = Polynomial::from_expr(ctx, arg, var_name) else {
        return Vec::new();
    };
    let slope = arg_poly
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let offset = arg_poly
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    if slope.is_zero() || (slope.is_one() && offset.is_zero()) {
        return Vec::new();
    }

    let title = match builtin {
        BuiltinFn::Exp => "Usar la regla de exp con derivada interna",
        BuiltinFn::Sin => "Usar la regla de sin con derivada interna",
        BuiltinFn::Cos => "Usar la regla de cos con derivada interna",
        BuiltinFn::Sinh => "Usar la regla de sinh con derivada interna",
        BuiltinFn::Cosh => "Usar la regla de cosh con derivada interna",
        _ => return Vec::new(),
    };

    let mut substeps: Vec<SubStep> =
        checked_antiderivative_substep(ctx, title, args[0], after, var_name)
            .into_iter()
            .collect();
    substeps.extend([SubStep::keyed(
        "usub.identify_affine_argument",
        vec![],
        display_expr(ctx, arg),
        display_expr(ctx, after),
    )
    .with_before_latex(latex_expr(ctx, arg))
    .with_after_latex(latex_expr(ctx, after))]);

    if !slope.is_one() {
        substeps.push(
            SubStep::keyed(
                "usub.adjust_constant_factor",
                vec![],
                affine_internal_derivative_display(ctx, arg, var_name, &slope),
                display_expr(ctx, after),
            )
            .with_before_latex(affine_internal_derivative_latex(ctx, arg, var_name, &slope))
            .with_after_latex(latex_expr(ctx, after)),
        );
    }

    substeps
}

pub(super) fn affine_argument_slope(
    ctx: &Context,
    arg: ExprId,
    var_name: &str,
) -> Option<BigRational> {
    let Ok(poly) = Polynomial::from_expr(ctx, arg, var_name) else {
        return None;
    };
    if poly.degree() != 1 {
        return None;
    }
    let slope = poly
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    (!slope.is_zero()).then_some(slope)
}

pub(super) fn scale_polynomial(poly: &Polynomial, scale: &BigRational) -> Polynomial {
    Polynomial::new(
        poly.coeffs.iter().map(|coeff| coeff * scale).collect(),
        poly.var.clone(),
    )
}

pub(super) fn polynomial_display_and_latex(ctx: &Context, poly: &Polynomial) -> (String, String) {
    let mut scratch = ctx.clone();
    let expr = poly.to_expr(&mut scratch);
    (display_expr(&scratch, expr), latex_expr(&scratch, expr))
}

pub(super) fn generate_nested_inverse_polynomial_table_integration_substeps(
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
    let mut scratch = ctx.clone();
    if !cas_math::symbolic_integration_support::integrate_symbolic_is_nested_inverse_polynomial_substitution_target(
        &mut scratch,
        args[0],
        var_name,
    ) {
        return Vec::new();
    }

    let Some(table_match) = nested_inverse_polynomial_result(ctx, after, var_name) else {
        return Vec::new();
    };
    let title = match table_match.builtin {
        BuiltinFn::Arcsin | BuiltinFn::Asin => "Usar la regla de u'/sqrt(1-u^2) -> arcsin(u)",
        BuiltinFn::Asinh => "Usar la regla de u'/sqrt(1+u^2) -> asinh(u)",
        BuiltinFn::Acosh => "Usar la regla de u'/sqrt(u^2-1) -> acosh(u)",
        BuiltinFn::Arctan | BuiltinFn::Atan => "Usar la regla de u'/(1+u^2) -> arctan(u)",
        BuiltinFn::Atanh => "Usar la regla de u'/(1-u^2) -> atanh(u)",
        _ => return Vec::new(),
    };

    let mut substeps: Vec<SubStep> =
        checked_antiderivative_substep(ctx, title, args[0], after, var_name)
            .into_iter()
            .collect();
    substeps.push(
        SubStep::keyed(
            "usub.identify_u_du",
            vec![],
            display_expr(ctx, table_match.arg),
            table_match.derivative_display,
        )
        .with_before_latex(latex_expr(ctx, table_match.arg))
        .with_after_latex(table_match.derivative_latex),
    );
    substeps
}

fn nested_inverse_polynomial_result(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
) -> Option<NestedInversePolynomialTableMatch> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    match ctx.get(expr) {
        Expr::Function(fn_id, args) if args.len() == 1 => {
            let builtin = ctx.builtin_of(*fn_id)?;
            if !matches!(
                builtin,
                BuiltinFn::Arcsin
                    | BuiltinFn::Asin
                    | BuiltinFn::Asinh
                    | BuiltinFn::Acosh
                    | BuiltinFn::Arctan
                    | BuiltinFn::Atan
                    | BuiltinFn::Atanh
            ) {
                return None;
            }
            nested_inverse_polynomial_arg(ctx, builtin, args[0], var_name)
        }
        Expr::Neg(inner) => nested_inverse_polynomial_result(ctx, *inner, var_name),
        Expr::Hold(inner) => nested_inverse_polynomial_result(ctx, *inner, var_name),
        Expr::Mul(left, right) => {
            if as_rational_const(ctx, *left, 8).is_some() {
                return nested_inverse_polynomial_result(ctx, *right, var_name);
            }
            if as_rational_const(ctx, *right, 8).is_some() {
                return nested_inverse_polynomial_result(ctx, *left, var_name);
            }
            None
        }
        Expr::Div(num, den) => {
            if as_rational_const(ctx, *den, 8).is_some() {
                return nested_inverse_polynomial_result(ctx, *num, var_name);
            }
            None
        }
        _ => None,
    }
}

fn nested_inverse_polynomial_arg(
    ctx: &Context,
    builtin: BuiltinFn,
    arg: ExprId,
    var_name: &str,
) -> Option<NestedInversePolynomialTableMatch> {
    let arg_poly = Polynomial::from_expr(ctx, arg, var_name).ok()?;
    if arg_poly.degree() <= 1 {
        return None;
    }
    let derivative = arg_poly.derivative();
    if derivative.is_zero() {
        return None;
    }
    let (derivative_display, derivative_latex) = polynomial_display_and_latex(ctx, &derivative);
    Some(NestedInversePolynomialTableMatch {
        builtin,
        arg,
        derivative_display,
        derivative_latex,
    })
}

pub(super) fn extract_non_unit_affine_var_coeff_with_sign(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
) -> Option<SignedAffineCoeff> {
    match ctx.get(expr) {
        Expr::Add(left, right) => {
            if !contains_named_var(ctx, *left, var_name) {
                return extract_non_unit_affine_var_coeff_with_sign(ctx, *right, var_name);
            }
            if !contains_named_var(ctx, *right, var_name) {
                return extract_non_unit_affine_var_coeff_with_sign(ctx, *left, var_name);
            }
            None
        }
        Expr::Sub(left, right) => {
            if !contains_named_var(ctx, *left, var_name)
                && contains_named_var(ctx, *right, var_name)
            {
                let mut coeff = extract_non_unit_affine_var_coeff_with_sign(ctx, *right, var_name)?;
                coeff.is_negative = !coeff.is_negative;
                return Some(coeff);
            }
            if contains_named_var(ctx, *left, var_name)
                && !contains_named_var(ctx, *right, var_name)
            {
                return extract_non_unit_affine_var_coeff_with_sign(ctx, *left, var_name);
            }
            None
        }
        Expr::Neg(inner) => {
            let mut coeff = extract_non_unit_affine_var_coeff_with_sign(ctx, *inner, var_name)?;
            coeff.is_negative = !coeff.is_negative;
            Some(coeff)
        }
        _ => extract_non_unit_affine_linear_coeff(ctx, expr, var_name).map(|coeff| {
            SignedAffineCoeff {
                coeff,
                is_negative: false,
            }
        }),
    }
}

pub(super) fn signed_affine_coeff_display(ctx: &Context, coeff: SignedAffineCoeff) -> String {
    let display = display_expr(ctx, coeff.coeff);
    if coeff.is_negative {
        format!("-{}", display)
    } else {
        display
    }
}

pub(super) fn signed_affine_coeff_latex(ctx: &Context, coeff: SignedAffineCoeff) -> String {
    let latex = latex_expr(ctx, coeff.coeff);
    if coeff.is_negative {
        format!("-{}", latex)
    } else {
        latex
    }
}

pub(super) fn expr_matches_signed_affine_coeff(
    ctx: &Context,
    expr: ExprId,
    coeff: SignedAffineCoeff,
) -> Option<bool> {
    if compare_expr(ctx, expr, coeff.coeff) == Ordering::Equal {
        return Some(false);
    }
    if let Expr::Neg(inner) = ctx.get(expr) {
        if compare_expr(ctx, *inner, coeff.coeff) == Ordering::Equal {
            return Some(true);
        }
    }
    None
}

pub(super) fn polynomial_trace_arg_ignoring_independent_addends(
    ctx: &Context,
    arg: ExprId,
    var_name: &str,
) -> Option<Polynomial> {
    if let Ok(poly) = Polynomial::from_expr(ctx, arg, var_name) {
        return Some(poly);
    }

    let terms = expr_nary::add_terms_signed(ctx, arg);
    if terms.len() <= 1 {
        return None;
    }

    let mut removed_independent_term = false;
    let mut saw_dependent_term = false;
    let mut poly = Polynomial::zero(var_name.to_string());
    for (term, sign) in terms {
        if contains_named_var(ctx, term, var_name) {
            let term_poly = Polynomial::from_expr(ctx, term, var_name).ok()?;
            saw_dependent_term = true;
            poly = match sign {
                Sign::Pos => poly.add(&term_poly),
                Sign::Neg => poly.sub(&term_poly),
            };
        } else {
            removed_independent_term = true;
        }
    }

    (removed_independent_term && saw_dependent_term).then_some(poly)
}

/// The nonzero rational `a` for which `expr == a·u` (`u` the variable named `u_name`, no constant
/// offset), or `None` if `expr` is not that pure linear form.
pub(super) fn linear_scale_of(ctx: &Context, expr: ExprId, u_name: &str) -> Option<BigRational> {
    let poly = Polynomial::from_expr(ctx, expr, u_name).ok()?;
    if poly.degree() != 1 {
        return None;
    }
    if poly
        .coeffs
        .first()
        .is_some_and(|constant| !constant.is_zero())
    {
        return None;
    }
    poly.coeffs.get(1).cloned().filter(|scale| !scale.is_zero())
}
