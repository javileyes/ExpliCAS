//! Orquestador: familia `general` (troceo P1).
//!
//! Ver la cabecera de `orchestrator.rs` para el contexto.

use super::*;

pub(super) fn to_math_auto_expand_budget(
    budget: &crate::phase::ExpandBudget,
) -> cas_math::auto_expand_scan::ExpandBudget {
    cas_math::auto_expand_scan::ExpandBudget {
        max_pow_exp: budget.max_pow_exp,
        max_base_terms: budget.max_base_terms,
        max_generated_terms: budget.max_generated_terms,
        max_vars: budget.max_vars,
    }
}

fn poly_lower_step_message(kind: cas_math::poly_lowering::PolyLowerStepKind) -> &'static str {
    match kind {
        cas_math::poly_lowering::PolyLowerStepKind::Direct { op } => match op {
            cas_math::poly_lowering_ops::PolyBinaryOp::Add => {
                "Poly lowering: combined poly_result + poly_result"
            }
            cas_math::poly_lowering_ops::PolyBinaryOp::Sub => {
                "Poly lowering: combined poly_result - poly_result"
            }
            cas_math::poly_lowering_ops::PolyBinaryOp::Mul => {
                "Poly lowering: combined poly_result * poly_result"
            }
        },
        cas_math::poly_lowering::PolyLowerStepKind::Promoted => {
            "Poly lowering: promoted and combined expressions"
        }
    }
}

pub(super) fn run_poly_lower_pass(
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> (ExprId, Vec<Step>) {
    let out =
        poly_lowering::poly_lower_pass_with_items(ctx, expr, collect_steps, |core_ctx, step| {
            Step::new(
                poly_lower_step_message(step.kind),
                "Polynomial Combination",
                step.before,
                step.after,
                Vec::new(),
                Some(core_ctx),
            )
        });
    (out.expr, out.items)
}

pub(super) fn run_poly_gcd_modp_eager_pass(
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> (ExprId, Vec<Step>) {
    cas_math::poly_modp_calls::eager_eval_poly_gcd_calls_with(
        ctx,
        expr,
        collect_steps,
        |core_ctx, before, after| {
            Step::new(
                "Eager eval poly_gcd_modp (bypass simplifier)",
                "Polynomial GCD mod p",
                before,
                after,
                Vec::new(),
                Some(core_ctx),
            )
        },
    )
}

pub(super) fn run_profiled_orchestrator_section<T>(
    name: &'static str,
    sample: Option<String>,
    run: impl FnOnce() -> T,
) -> T {
    if !crate::orchestrator_shortcut_profiler::should_profile_orchestrator_shortcut(name) {
        return run();
    }

    if let Some(sample) = sample {
        crate::orchestrator_shortcut_profiler::record_orchestrator_shortcut_sample(name, sample);
    }

    let start = web_time::Instant::now();
    let result = run();
    crate::orchestrator_shortcut_profiler::record_orchestrator_shortcut_attempt(
        name,
        true,
        start.elapsed(),
    );
    result
}

pub(super) fn run_profiled_orchestrator_bool_section(
    name: &'static str,
    run: impl FnOnce() -> bool,
) -> bool {
    if !crate::orchestrator_shortcut_profiler::should_profile_orchestrator_shortcut(name) {
        return run();
    }

    let start = web_time::Instant::now();
    let result = run();
    crate::orchestrator_shortcut_profiler::record_orchestrator_shortcut_attempt(
        name,
        result,
        start.elapsed(),
    );
    result
}

pub(super) fn run_profiled_orchestrator_bool_section_with_sample(
    name: &'static str,
    sample: Option<String>,
    run: impl FnOnce() -> bool,
) -> bool {
    if !crate::orchestrator_shortcut_profiler::should_profile_orchestrator_shortcut(name) {
        return run();
    }

    let start = web_time::Instant::now();
    let result = run();
    crate::orchestrator_shortcut_profiler::record_orchestrator_shortcut_attempt(
        name,
        result,
        start.elapsed(),
    );
    if let Some(sample) = sample {
        crate::orchestrator_shortcut_profiler::record_orchestrator_shortcut_outcome_sample(
            name, result, sample,
        );
    }
    result
}

pub(super) fn record_profiled_orchestrator_route_hit(
    ctx: &mut Context,
    expr: ExprId,
    name: &'static str,
) {
    if !crate::orchestrator_shortcut_profiler::should_profile_orchestrator_shortcut(name) {
        return;
    }
    run_profiled_orchestrator_section(
        name,
        Some(render_expr_for_orchestrator_profile(ctx, expr)),
        || (),
    );
}

pub(super) fn record_orchestrator_shortcut_profile_sample(
    ctx: &Context,
    expr: ExprId,
    name: &'static str,
) {
    if !crate::orchestrator_shortcut_profiler::should_profile_orchestrator_shortcut(name) {
        return;
    }
    crate::orchestrator_shortcut_profiler::record_orchestrator_shortcut_sample(
        name,
        render_direct_small_zero_profile_sample_root(ctx, expr),
    );
}

pub(super) fn pipeline_phase_profile_label(phase: SimplifyPhase) -> &'static str {
    match phase {
        SimplifyPhase::Core => "pipeline.phase.core",
        SimplifyPhase::Transform => "pipeline.phase.transform",
        SimplifyPhase::Rationalize => "pipeline.phase.rationalize",
        SimplifyPhase::PostCleanup => "pipeline.phase.post_cleanup",
    }
}

fn pipeline_phase_pass_profile_label(phase: SimplifyPhase, changed: bool) -> &'static str {
    match (phase, changed) {
        (SimplifyPhase::Core, true) => "pipeline.phase.core.pass.changed",
        (SimplifyPhase::Core, false) => "pipeline.phase.core.pass.fixed",
        (SimplifyPhase::Transform, true) => "pipeline.phase.transform.pass.changed",
        (SimplifyPhase::Transform, false) => "pipeline.phase.transform.pass.fixed",
        (SimplifyPhase::Rationalize, true) => "pipeline.phase.rationalize.pass.changed",
        (SimplifyPhase::Rationalize, false) => "pipeline.phase.rationalize.pass.fixed",
        (SimplifyPhase::PostCleanup, true) => "pipeline.phase.post_cleanup.pass.changed",
        (SimplifyPhase::PostCleanup, false) => "pipeline.phase.post_cleanup.pass.fixed",
    }
}

pub(super) fn active_pipeline_phase_pass_profile_labels(
    phase: SimplifyPhase,
) -> Option<(&'static str, &'static str)> {
    if !crate::orchestrator_shortcut_profiler::orchestrator_shortcut_profiling_enabled() {
        return None;
    }

    let changed_label = pipeline_phase_pass_profile_label(phase, true);
    let fixed_label = pipeline_phase_pass_profile_label(phase, false);
    (crate::orchestrator_shortcut_profiler::should_profile_orchestrator_shortcut(changed_label)
        || crate::orchestrator_shortcut_profiler::should_profile_orchestrator_shortcut(fixed_label))
    .then_some((changed_label, fixed_label))
}

pub(super) fn is_terminal_after_core(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) => true,
        Expr::Div(num, den) => {
            matches!(ctx.get(*num), Expr::Number(_)) && matches!(ctx.get(*den), Expr::Number(_))
        }
        _ => false,
    }
}

pub(super) fn is_plain_symbolic_binomial_after_core(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Add(left, right) | Expr::Sub(left, right) => {
            is_symbolic_atom(ctx, *left) && is_symbolic_atom(ctx, *right)
        }
        Expr::Neg(inner) => is_plain_symbolic_binomial_after_core(ctx, *inner),
        _ => false,
    }
}

pub(super) fn expr_contains_builtin_function_local(ctx: &Context, root: ExprId) -> bool {
    let mut stack = vec![root];

    while let Some(expr) = stack.pop() {
        match ctx.get(expr) {
            Expr::Function(_, _) => return true,
            Expr::Pow(lhs, rhs)
            | Expr::Add(lhs, rhs)
            | Expr::Sub(lhs, rhs)
            | Expr::Mul(lhs, rhs)
            | Expr::Div(lhs, rhs) => {
                stack.push(*lhs);
                stack.push(*rhs);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(*inner),
            Expr::Matrix { data, .. } => stack.extend(data.iter().copied()),
            Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => {}
        }
    }

    false
}

pub(super) fn expr_contains_named_function_local(ctx: &Context, root: ExprId, name: &str) -> bool {
    let mut stack = vec![root];

    while let Some(expr) = stack.pop() {
        match ctx.get(expr) {
            Expr::Function(fn_id, args) => {
                if ctx.sym_name(*fn_id) == name {
                    return true;
                }
                stack.extend(args.iter().copied());
            }
            Expr::Pow(lhs, rhs)
            | Expr::Add(lhs, rhs)
            | Expr::Sub(lhs, rhs)
            | Expr::Mul(lhs, rhs)
            | Expr::Div(lhs, rhs) => {
                stack.push(*lhs);
                stack.push(*rhs);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(*inner),
            Expr::Matrix { data, .. } => stack.extend(data.iter().copied()),
            Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => {}
        }
    }

    false
}

pub(super) fn expr_contains_factorial_call_local(ctx: &Context, root: ExprId) -> bool {
    let mut stack = vec![root];
    while let Some(expr) = stack.pop() {
        match ctx.get(expr) {
            Expr::Function(fn_id, args)
                if args.len() == 1 && matches!(ctx.sym_name(*fn_id), "fact" | "factorial") =>
            {
                return true;
            }
            Expr::Function(_, args) => stack.extend(args.iter().copied()),
            Expr::Pow(base, exp) => {
                stack.push(*base);
                stack.push(*exp);
            }
            Expr::Add(lhs, rhs)
            | Expr::Sub(lhs, rhs)
            | Expr::Mul(lhs, rhs)
            | Expr::Div(lhs, rhs) => {
                stack.push(*lhs);
                stack.push(*rhs);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(*inner),
            Expr::Matrix { data, .. } => stack.extend(data.iter().copied()),
            Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => {}
        }
    }
    false
}

pub(super) fn expr_contains_pi_constant_local(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Constant(Constant::Pi) => true,
        Expr::Add(lhs, rhs)
        | Expr::Sub(lhs, rhs)
        | Expr::Mul(lhs, rhs)
        | Expr::Div(lhs, rhs)
        | Expr::Pow(lhs, rhs) => {
            expr_contains_pi_constant_local(ctx, *lhs) || expr_contains_pi_constant_local(ctx, *rhs)
        }
        Expr::Neg(inner) | Expr::Hold(inner) => expr_contains_pi_constant_local(ctx, *inner),
        Expr::Function(_, args) => args
            .iter()
            .copied()
            .any(|arg| expr_contains_pi_constant_local(ctx, arg)),
        Expr::Matrix { data, .. } => data
            .iter()
            .copied()
            .any(|arg| expr_contains_pi_constant_local(ctx, arg)),
        Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => false,
    }
}

fn symbolic_cross_term_atoms(ctx: &Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    let view = MulView::from_expr(ctx, expr);
    if view.factors.len() != 2 {
        return None;
    }
    let left = view.factors[0];
    let right = view.factors[1];
    if is_symbolic_atom(ctx, left) && is_symbolic_atom(ctx, right) {
        Some((left, right))
    } else {
        None
    }
}

pub(super) fn expr_eq(ctx: &Context, left: ExprId, right: ExprId) -> bool {
    cas_ast::ordering::compare_expr(ctx, left, right) == std::cmp::Ordering::Equal
}

pub(super) fn build_root_shortcut_step_from_rewrite(
    ctx: &Context,
    before: ExprId,
    rewrite: &crate::rule::Rewrite,
    rule_name: &'static str,
) -> Step {
    let mut step = Step::with_snapshots(
        &rewrite.description,
        rule_name,
        before,
        rewrite.new_expr,
        smallvec::SmallVec::<[crate::step::PathStep; 8]>::new(),
        Some(ctx),
        before,
        rewrite.new_expr,
    );
    step.importance = crate::step::ImportanceLevel::High;
    {
        let meta = step.meta_mut();
        meta.before_local = rewrite.before_local;
        meta.after_local = rewrite.after_local;
        meta.assumption_events = rewrite.assumption_events.clone();
        meta.required_conditions = rewrite.required_conditions.clone();
        meta.poly_proof = rewrite.poly_proof.clone();
        meta.substeps = rewrite.substeps.clone();
    }
    step
}

/// SOUNDNESS veto shared by both root-shortcut dispatch macros: returns `true` when a shortcut
/// `result` for `expr` is unsound and must be skipped so the honest rule pipeline runs instead.
/// (1) A collapse to `0` of an expression with an EXACT non-zero value at a generic rational point
/// (`1/(x²-1) - 1/(x-1)`). (2) An `∞/∞` quotient cancelled to anything other than `undefined`
/// (`(2·∞)/(5·∞) -> 2/5`).
pub(super) fn root_shortcut_result_is_unsound(
    ctx: &mut Context,
    expr: ExprId,
    result: ExprId,
) -> bool {
    let collapses_to_zero = {
        let zero = ctx.num(0);
        compare_expr(ctx, result, zero) == Ordering::Equal
    };
    if collapses_to_zero
        && crate::rules::arithmetic::common_scaled_difference_has_exact_nonzero_witness(ctx, expr)
    {
        return true;
    }
    if let Expr::Div(num, den) = *ctx.get(expr) {
        if cas_math::infinity_support::is_infinite_valued(ctx, num)
            && cas_math::infinity_support::is_infinite_valued(ctx, den)
            && !matches!(
                ctx.get(result),
                Expr::Constant(cas_ast::Constant::Undefined)
            )
        {
            return true;
        }
    }
    false
}

pub(super) fn flip_add_sign_root(sign: Sign) -> Sign {
    match sign {
        Sign::Pos => Sign::Neg,
        Sign::Neg => Sign::Pos,
    }
}

fn combine_add_signs_root(lhs: Sign, rhs: Sign) -> Sign {
    if lhs == rhs {
        Sign::Pos
    } else {
        Sign::Neg
    }
}

pub(super) fn extract_normalized_signed_terms_with_outer_sign_root(
    ctx: &mut Context,
    expr: ExprId,
    outer_sign: Sign,
) -> smallvec::SmallVec<[(ExprId, Sign); 8]> {
    AddView::from_expr(ctx, expr)
        .terms
        .into_iter()
        .map(|(term_expr, term_sign)| {
            normalize_signed_add_term_root(
                ctx,
                term_expr,
                combine_add_signs_root(outer_sign, term_sign),
            )
        })
        .collect()
}

pub(super) fn signed_term_multiset_matches_root(
    ctx: &mut Context,
    lhs_terms: &[(ExprId, Sign)],
    rhs_terms: &[(ExprId, Sign)],
) -> bool {
    if lhs_terms.len() != rhs_terms.len() {
        return false;
    }

    let mut rhs_used = vec![false; rhs_terms.len()];
    for (lhs_expr, lhs_sign) in lhs_terms.iter().copied() {
        let Some(rhs_index) =
            rhs_terms
                .iter()
                .copied()
                .enumerate()
                .find_map(|(rhs_index, (rhs_expr, rhs_sign))| {
                    (!rhs_used[rhs_index]
                        && lhs_sign == rhs_sign
                        && compare_expr(ctx, lhs_expr, rhs_expr) == Ordering::Equal)
                        .then_some(rhs_index)
                })
        else {
            return false;
        };
        rhs_used[rhs_index] = true;
    }

    true
}

pub(super) fn flipped_signed_terms_root(
    terms: &[(ExprId, Sign)],
) -> smallvec::SmallVec<[(ExprId, Sign); 8]> {
    terms
        .iter()
        .copied()
        .map(|(expr, sign)| (expr, flip_add_sign_root(sign)))
        .collect()
}

pub(super) fn strip_positive_one_passthrough_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let view = AddView::from_expr(ctx, expr);
    let mut stripped = false;
    let mut residual_terms = Vec::new();

    for (term_expr, term_sign) in view.terms {
        let is_positive_one = term_sign == Sign::Pos
            && matches!(
                ctx.get(term_expr),
                Expr::Number(n) if *n == num_rational::BigRational::from_integer(1.into())
            );

        if is_positive_one && !stripped {
            stripped = true;
            continue;
        }
        residual_terms.push((term_expr, term_sign));
    }

    if !stripped || residual_terms.is_empty() {
        return None;
    }

    Some(build_signed_sum_expr_root(ctx, &residual_terms))
}

pub(super) fn extract_half_scaled_base_root(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    if let Expr::Div(numerator, denominator) = ctx.get(expr) {
        if extract_i64_integer(ctx, *denominator) == Some(2) {
            return Some(*numerator);
        }
    }

    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() != 2 {
        return None;
    }

    let mut saw_half = false;
    let mut base = None;
    for factor in factors {
        match ctx.get(factor) {
            Expr::Number(n) if *n == BigRational::new(1.into(), 2.into()) => {
                if saw_half {
                    return None;
                }
                saw_half = true;
            }
            _ => {
                if base.is_some() {
                    return None;
                }
                base = Some(factor);
            }
        }
    }

    saw_half.then_some(base?).or(None)
}

pub(super) fn matches_expr_or_negation_root(ctx: &Context, lhs: ExprId, rhs: ExprId) -> bool {
    if compare_expr(ctx, lhs, rhs) == Ordering::Equal {
        return true;
    }

    let (lhs_coeff, lhs_base) = extract_coef_and_base(ctx, lhs);
    let (rhs_coeff, rhs_base) = extract_coef_and_base(ctx, rhs);
    compare_expr(ctx, lhs_base, rhs_base) == Ordering::Equal && lhs_coeff == -rhs_coeff
}

pub(super) fn extract_positive_one_plus_other_term_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 2 {
        return None;
    }

    let mut saw_positive_one = false;
    let mut other_term = None;
    for (term_expr, term_sign) in view.terms {
        if term_sign != Sign::Pos {
            return None;
        }

        if let Expr::Number(n) = ctx.get(term_expr) {
            if n.is_one() && !saw_positive_one {
                saw_positive_one = true;
                continue;
            }
        }

        if other_term.replace(term_expr).is_some() {
            return None;
        }
    }

    saw_positive_one.then_some(other_term?)
}

pub(super) fn build_half_expr_root(ctx: &mut Context, expr: ExprId) -> ExprId {
    let two = ctx.num(2);
    ctx.add(Expr::Div(expr, two))
}

pub(super) fn extract_plain_pow2_base_root(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Pow(base, exponent) = ctx.get(expr) else {
        return None;
    };
    let Expr::Number(n) = ctx.get(*exponent) else {
        return None;
    };
    (*n == BigRational::from_integer(2.into())).then_some(*base)
}

fn extract_plain_pow_base_root(ctx: &Context, expr: ExprId, expected_power: i64) -> Option<ExprId> {
    let Expr::Pow(base, exponent) = ctx.get(expr) else {
        return None;
    };
    if extract_i64_integer(ctx, *exponent)? != expected_power {
        return None;
    }
    Some(*base)
}

pub(super) fn extract_base_plus_constant_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, BigRational)> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 2 {
        return None;
    }

    let mut base_term = None;
    let mut constant = None;
    for (term_expr, term_sign) in view.terms {
        if let Expr::Number(n) = ctx.get(term_expr) {
            if constant.is_some() {
                return None;
            }
            constant = Some(if term_sign == Sign::Neg {
                -n.clone()
            } else {
                n.clone()
            });
            continue;
        }

        let (mut coeff, base) = extract_coef_and_base(ctx, term_expr);
        if term_sign == Sign::Neg {
            coeff = -coeff;
        }
        if coeff != BigRational::one() || base_term.is_some() {
            return None;
        }
        base_term = Some(base);
    }

    let (Some(base), Some(constant)) = (base_term, constant) else {
        return None;
    };
    (!constant.is_zero()).then_some((base, constant))
}

pub(super) fn extract_direct_two_linear_shift_product_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<BigRational>)> {
    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() != 2 {
        return None;
    }

    let (lhs_base, lhs_constant) = extract_base_plus_constant_root(ctx, factors[0])?;
    let (rhs_base, rhs_constant) = extract_base_plus_constant_root(ctx, factors[1])?;
    if compare_expr(ctx, lhs_base, rhs_base) != Ordering::Equal {
        return None;
    }

    let mut constants = vec![lhs_constant, rhs_constant];
    constants.sort();
    Some((lhs_base, constants))
}

pub(super) fn extract_direct_three_linear_shift_product_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<BigRational>)> {
    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() != 3 {
        return None;
    }

    let mut base = None;
    let mut constants = Vec::with_capacity(3);
    for factor in factors {
        let (factor_base, constant) = extract_base_plus_constant_root(ctx, factor)?;
        if let Some(expected_base) = base {
            if compare_expr(ctx, expected_base, factor_base) != Ordering::Equal {
                return None;
            }
        } else {
            base = Some(factor_base);
        }
        constants.push(constant);
    }

    constants.sort();
    Some((base?, constants))
}

pub(super) fn extract_direct_quartic_gcf_base_expanded_root(
    ctx: &Context,
    expr: ExprId,
) -> Option<ExprId> {
    let Expr::Sub(lhs, rhs) = ctx.get(expr) else {
        return None;
    };
    let fourth_base = extract_plain_pow_base_root(ctx, *lhs, 4)?;
    let squared_base = extract_plain_pow_base_root(ctx, *rhs, 2)?;
    (compare_expr(ctx, fourth_base, squared_base) == Ordering::Equal).then_some(fourth_base)
}

pub(super) fn extract_direct_quartic_gcf_base_factored_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() != 3 {
        return None;
    }

    let mut squared_base = None;
    let mut plus_base = None;
    let mut minus_base = None;
    for factor in factors {
        if let Some(base) = extract_plain_pow2_base_root(ctx, factor) {
            if squared_base.replace(base).is_some() {
                return None;
            }
            continue;
        }

        let (base, constant) = extract_base_plus_constant_root(ctx, factor)?;
        if constant == BigRational::one() {
            if plus_base.replace(base).is_some() {
                return None;
            }
        } else if constant == -BigRational::one() {
            if minus_base.replace(base).is_some() {
                return None;
            }
        } else {
            return None;
        }
    }

    let squared_base = squared_base?;
    let plus_base = plus_base?;
    let minus_base = minus_base?;
    (compare_expr(ctx, squared_base, plus_base) == Ordering::Equal
        && compare_expr(ctx, squared_base, minus_base) == Ordering::Equal)
        .then_some(squared_base)
}

pub(super) fn extract_numeric_atan_ratio_arg_root(
    ctx: &Context,
    expr: ExprId,
) -> Option<BigRational> {
    let ratio_arg = extract_unary_builtin_arg_root(ctx, expr, BuiltinFn::Atan)
        .or_else(|| extract_unary_builtin_arg_root(ctx, expr, BuiltinFn::Arctan))?;
    extract_literal_rational_root(ctx, ratio_arg)
}

pub(super) fn ground_exact_constant_key_root(ctx: &Context, expr: ExprId) -> Option<String> {
    match ctx.get(expr) {
        Expr::Number(n) => Some(format!("N({}/{})", n.numer(), n.denom())),
        Expr::Constant(c) => Some(format!("C({c:?})")),
        Expr::Neg(inner) => Some(format!(
            "Neg({})",
            ground_exact_constant_key_root(ctx, *inner)?
        )),
        Expr::Add(lhs, rhs) => {
            let mut parts = [
                ground_exact_constant_key_root(ctx, *lhs)?,
                ground_exact_constant_key_root(ctx, *rhs)?,
            ];
            parts.sort_unstable();
            Some(format!("Add({},{})", parts[0], parts[1]))
        }
        Expr::Sub(lhs, rhs) => {
            let mut parts = [
                ground_exact_constant_key_root(ctx, *lhs)?,
                format!("Neg({})", ground_exact_constant_key_root(ctx, *rhs)?),
            ];
            parts.sort_unstable();
            Some(format!("Add({},{})", parts[0], parts[1]))
        }
        Expr::Mul(lhs, rhs) => {
            let lhs_key = ground_exact_constant_key_root(ctx, *lhs)?;
            let rhs_key = ground_exact_constant_key_root(ctx, *rhs)?;
            if lhs_key == "N(-1/1)" {
                return Some(format!("Neg({rhs_key})"));
            }
            if rhs_key == "N(-1/1)" {
                return Some(format!("Neg({lhs_key})"));
            }
            let mut parts = [lhs_key, rhs_key];
            parts.sort_unstable();
            Some(format!("Mul({},{})", parts[0], parts[1]))
        }
        Expr::Div(lhs, rhs) => {
            if let (Expr::Number(ln), Expr::Number(rn)) = (ctx.get(*lhs), ctx.get(*rhs)) {
                let value = ln / rn.clone();
                return Some(format!("N({}/{})", value.numer(), value.denom()));
            }
            Some(format!(
                "Div({},{})",
                ground_exact_constant_key_root(ctx, *lhs)?,
                ground_exact_constant_key_root(ctx, *rhs)?
            ))
        }
        Expr::Pow(base, exp) => Some(format!(
            "Pow({},{})",
            ground_exact_constant_key_root(ctx, *base)?,
            ground_exact_constant_key_root(ctx, *exp)?
        )),
        Expr::Function(fn_id, args)
            if args.len() == 1 && ctx.is_builtin(*fn_id, BuiltinFn::Sqrt) =>
        {
            Some(format!(
                "Pow({},N(1/2))",
                ground_exact_constant_key_root(ctx, args[0])?
            ))
        }
        _ => None,
    }
}

pub(super) fn extract_plus_one_expr_target_root(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 2 {
        return None;
    }

    let mut base = None;
    let mut saw_one = false;
    for (term_expr, term_sign) in view.terms {
        if term_sign != Sign::Pos {
            return None;
        }
        if extract_i64_integer(ctx, term_expr) == Some(1) {
            if saw_one {
                return None;
            }
            saw_one = true;
        } else if base.is_none() {
            base = Some(term_expr);
        } else {
            return None;
        }
    }

    saw_one.then_some(base?).or(None)
}

pub(super) fn extract_sophie_germain_bases_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId)> {
    fn pow_four_base(ctx: &Context, term: ExprId) -> Option<ExprId> {
        let Expr::Pow(base, exp) = ctx.get(term) else {
            return None;
        };
        (extract_i64_integer(ctx, *exp) == Some(4)).then_some(*base)
    }

    fn four_times_fourth_power_base(ctx: &mut Context, term: ExprId) -> Option<ExprId> {
        match ctx.get(term).clone() {
            Expr::Mul(lhs, rhs) => {
                if extract_i64_integer(ctx, lhs) == Some(4) {
                    return pow_four_base(ctx, rhs);
                }
                if extract_i64_integer(ctx, rhs) == Some(4) {
                    return pow_four_base(ctx, lhs);
                }
                None
            }
            Expr::Number(n) if n == BigRational::from_integer(4.into()) => Some(ctx.num(1)),
            _ => None,
        }
    }

    let view = AddView::from_expr(ctx, expr).terms;
    if view.len() != 2 || !view.iter().all(|(_, sign)| *sign == Sign::Pos) {
        return None;
    }

    for (pow_four_term, fourth_term) in [(view[0].0, view[1].0), (view[1].0, view[0].0)] {
        let Some(a) = pow_four_base(ctx, pow_four_term) else {
            continue;
        };
        let Some(b) = four_times_fourth_power_base(ctx, fourth_term) else {
            continue;
        };
        return Some((a, b));
    }

    None
}

pub(super) fn matches_sophie_germain_quadratic_root(
    ctx: &mut Context,
    expr: ExprId,
    a: ExprId,
    b: ExprId,
    positive_linear: bool,
) -> bool {
    let terms = AddView::from_expr(ctx, expr).terms;
    if terms.len() != 3 {
        return false;
    }

    let two = ctx.num(2);
    let a_sq = ctx.add(Expr::Pow(a, two));
    let b_is_one = extract_i64_integer(ctx, b) == Some(1);
    let two_rational = BigRational::from_integer(2.into());
    let b_sq = if b_is_one {
        None
    } else {
        Some(ctx.add(Expr::Pow(b, two)))
    };
    let ab = if b_is_one {
        None
    } else {
        Some(smart_mul(ctx, a, b))
    };

    let mut found_a_sq = false;
    let mut found_two_b_sq = false;
    let mut found_two_ab = false;
    for (term, sign) in terms {
        let (coeff, base) = extract_coef_and_base(ctx, term);
        if sign == Sign::Pos
            && !found_a_sq
            && coeff.is_one()
            && compare_expr(ctx, base, a_sq) == Ordering::Equal
        {
            found_a_sq = true;
            continue;
        }
        if sign == Sign::Pos && !found_two_b_sq {
            if b_is_one {
                if matches!(ctx.get(term), Expr::Number(n) if *n == two_rational) {
                    found_two_b_sq = true;
                    continue;
                }
            } else if let Some(b_sq) = b_sq {
                if coeff == two_rational && compare_expr(ctx, base, b_sq) == Ordering::Equal {
                    found_two_b_sq = true;
                    continue;
                }
            }
        }
        if sign
            == (if positive_linear {
                Sign::Pos
            } else {
                Sign::Neg
            })
        {
            if found_two_ab {
                continue;
            }
            if b_is_one {
                if coeff == two_rational && compare_expr(ctx, base, a) == Ordering::Equal {
                    found_two_ab = true;
                    continue;
                }
            } else if let Some(ab) = ab {
                if coeff == two_rational && compare_expr(ctx, base, ab) == Ordering::Equal {
                    found_two_ab = true;
                    continue;
                }
            }
        }
    }

    found_a_sq && found_two_b_sq && found_two_ab
}

pub(super) fn build_sophie_germain_quadratic_expr_root(
    ctx: &mut Context,
    a: ExprId,
    b: ExprId,
    positive_linear: bool,
) -> ExprId {
    let two = ctx.num(2);
    let a_sq = ctx.add(Expr::Pow(a, two));
    let b_is_one = extract_i64_integer(ctx, b) == Some(1);
    let two_b_sq = if b_is_one {
        ctx.num(2)
    } else {
        let b_sq = ctx.add(Expr::Pow(b, two));
        mul2_raw(ctx, two, b_sq)
    };
    let two_ab = if b_is_one {
        mul2_raw(ctx, two, a)
    } else {
        let ab = smart_mul(ctx, a, b);
        mul2_raw(ctx, two, ab)
    };
    build_signed_sum_expr_root(
        ctx,
        &[
            (a_sq, Sign::Pos),
            (two_b_sq, Sign::Pos),
            (
                two_ab,
                if positive_linear {
                    Sign::Pos
                } else {
                    Sign::Neg
                },
            ),
        ],
    )
}

pub(super) fn extract_direct_tangent_addition_target_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId)> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 2 {
        return None;
    }

    let mut first_arg = None;
    let mut second_arg = None;
    for (term_expr, term_sign) in view.terms {
        if term_sign != Sign::Pos {
            return None;
        }
        let arg = extract_unary_builtin_arg_root(ctx, term_expr, BuiltinFn::Tan)?;
        if first_arg.is_none() {
            first_arg = Some(arg);
        } else if second_arg.is_none() {
            second_arg = Some(arg);
        } else {
            return None;
        }
    }

    Some((first_arg?, second_arg?))
}

pub(super) fn extract_scaled_plain_sine_term_arg_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let factors = flatten_mul_chain(ctx, expr);
    let mut numeric_coeff = BigRational::one();
    let mut sin_arg = None;

    for factor in factors {
        if let Expr::Number(n) = ctx.get(factor) {
            numeric_coeff *= n.clone();
            continue;
        }
        let Some((BuiltinFn::Sin, arg)) = extract_plain_sin_or_cos_arg_root(ctx, factor) else {
            return None;
        };
        if sin_arg.is_some() {
            return None;
        }
        sin_arg = Some(arg);
    }

    (numeric_coeff == BigRational::from_integer(2.into()))
        .then_some(sin_arg)
        .flatten()
}

pub(super) fn extract_consecutive_product_core_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() != 2 {
        return None;
    }

    for &candidate_u in &factors {
        let one = ctx.num(1);
        let candidate_u_plus_one = ctx.add(Expr::Add(candidate_u, one));
        if factors
            .iter()
            .any(|factor| compare_expr(ctx, *factor, candidate_u_plus_one) == Ordering::Equal)
        {
            return Some(candidate_u);
        }
    }

    None
}

pub(super) fn positive_two_term_sum_matches_terms_root(
    ctx: &mut Context,
    expr: ExprId,
    lhs: ExprId,
    rhs: ExprId,
) -> bool {
    let terms = AddView::from_expr(ctx, expr).terms;
    if terms.len() != 2 || terms.iter().any(|(_, sign)| *sign != Sign::Pos) {
        return false;
    }

    let first = terms[0].0;
    let second = terms[1].0;
    (compare_expr(ctx, first, lhs) == Ordering::Equal
        && compare_expr(ctx, second, rhs) == Ordering::Equal)
        || (compare_expr(ctx, first, rhs) == Ordering::Equal
            && compare_expr(ctx, second, lhs) == Ordering::Equal)
}

pub(super) fn extract_plain_cube_base_root(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    if let Expr::Number(n) = ctx.get(expr) {
        if let Some(root) = cas_math::root_forms::rational_cbrt_exact(n) {
            return Some(ctx.add(Expr::Number(root)));
        }
    }

    let Expr::Pow(base, exponent) = ctx.get(expr) else {
        return None;
    };
    let Expr::Number(n) = ctx.get(*exponent) else {
        return None;
    };
    (*n == BigRational::from_integer(3.into())).then_some(*base)
}

fn extract_base_minus_one_factor_root(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Sub(lhs, rhs) if extract_i64_integer(ctx, *rhs) == Some(1) => Some(*lhs),
        Expr::Add(lhs, rhs) if extract_i64_integer(ctx, *rhs) == Some(-1) => Some(*lhs),
        Expr::Add(lhs, rhs) if extract_i64_integer(ctx, *lhs) == Some(-1) => Some(*rhs),
        _ => None,
    }
}

pub(super) fn matches_geometric_series_sum_root(
    ctx: &mut Context,
    expr: ExprId,
    base: ExprId,
    max_exponent: i64,
) -> bool {
    let terms = AddView::from_expr(ctx, expr).terms;
    if max_exponent < 1 || terms.len() != (max_exponent as usize + 1) {
        return false;
    }

    let mut seen = HashSet::new();
    for (term_expr, term_sign) in terms {
        if term_sign != Sign::Pos {
            return false;
        }
        let exponent = if extract_i64_integer(ctx, term_expr) == Some(1) {
            0
        } else if compare_expr(ctx, term_expr, base) == Ordering::Equal {
            1
        } else {
            let Some(exponent) = extract_power_of_base_exponent_root(ctx, term_expr, base) else {
                return false;
            };
            exponent
        };
        if exponent < 0 || exponent > max_exponent || !seen.insert(exponent) {
            return false;
        }
    }

    seen.len() == (max_exponent as usize + 1)
}

pub(super) fn matches_geometric_difference_terms_root(
    ctx: &mut Context,
    terms: &[(ExprId, Sign)],
) -> bool {
    if terms.len() != 3 {
        return false;
    }

    let normalized_terms: Vec<(ExprId, Sign)> = terms
        .iter()
        .copied()
        .map(|(term_expr, term_sign)| normalize_signed_add_term_root(ctx, term_expr, term_sign))
        .collect();

    for (power_index, (power_expr, power_sign)) in normalized_terms.iter().copied().enumerate() {
        let (base, exponent_expr) = match ctx.get(power_expr).clone() {
            Expr::Pow(base, exponent) => (base, exponent),
            _ => continue,
        };
        let Some(exponent) = extract_i64_integer(ctx, exponent_expr) else {
            continue;
        };
        if exponent < 2 {
            continue;
        }

        let mut saw_one = false;
        let mut saw_product = false;
        for (index, (term_expr, term_sign)) in normalized_terms.iter().copied().enumerate() {
            if index == power_index {
                continue;
            }

            if extract_i64_integer(ctx, term_expr) == Some(1)
                && term_sign == power_sign.negate()
                && !saw_one
            {
                saw_one = true;
                continue;
            }

            let factors = flatten_mul_chain(ctx, term_expr);
            if factors.len() != 2 || term_sign != power_sign.negate() || saw_product {
                saw_one = false;
                saw_product = false;
                break;
            }

            let mut matched_product = false;
            for (first, second) in [(factors[0], factors[1]), (factors[1], factors[0])] {
                let Some(factor_base) = extract_base_minus_one_factor_root(ctx, first) else {
                    continue;
                };
                if compare_expr(ctx, factor_base, base) != Ordering::Equal {
                    continue;
                }
                if matches_geometric_series_sum_root(ctx, second, base, exponent - 1) {
                    matched_product = true;
                    break;
                }
            }

            if !matched_product {
                saw_one = false;
                saw_product = false;
                break;
            }
            saw_product = true;
        }

        if saw_one && saw_product {
            return true;
        }
    }

    false
}

pub(super) fn build_two_group_factorizations_root(
    ctx: &mut Context,
    factors: &[ExprId],
) -> Vec<(ExprId, ExprId)> {
    if factors.len() < 2 || factors.len() > 6 {
        return Vec::new();
    }

    if factors.len() == 2 {
        return vec![(factors[0], factors[1])];
    }

    let mut partitions = Vec::new();
    let total_masks = 1usize << factors.len();
    for mask in 1..(total_masks - 1) {
        let mut first = Vec::new();
        let mut second = Vec::new();
        for (index, factor) in factors.iter().copied().enumerate() {
            if ((mask >> index) & 1) == 1 {
                first.push(factor);
            } else {
                second.push(factor);
            }
        }
        if first.is_empty() || second.is_empty() {
            continue;
        }

        partitions.push((
            build_mul_expr_from_factors_root(ctx, &first),
            build_mul_expr_from_factors_root(ctx, &second),
        ));
    }

    partitions
}

fn build_smart_mul_expr_from_factors_root(ctx: &mut Context, factors: &[ExprId]) -> ExprId {
    let Some((first, rest)) = factors.split_first() else {
        return ctx.num(1);
    };

    let mut acc = *first;
    for factor in rest {
        acc = smart_mul(ctx, acc, *factor);
    }
    acc
}

pub(super) fn build_locally_simplified_mul_expr_from_factors_root(
    ctx: &mut Context,
    factors: &[ExprId],
) -> ExprId {
    let mut saw_zero = false;
    let mut saw_nonfinite = false;
    let mut filtered = Vec::new();

    for factor in factors.iter().copied() {
        match ctx.get(factor) {
            Expr::Number(n) if n.is_zero() => {
                saw_zero = true;
            }
            Expr::Number(n) if n.is_one() => {}
            Expr::Constant(Constant::Undefined | Constant::Infinity) => {
                saw_nonfinite = true;
                filtered.push(factor);
            }
            Expr::Neg(inner) if matches!(ctx.get(*inner), Expr::Constant(Constant::Infinity)) => {
                saw_nonfinite = true;
                filtered.push(factor);
            }
            _ => filtered.push(factor),
        }
    }

    if saw_zero {
        return if saw_nonfinite {
            ctx.add(Expr::Constant(Constant::Undefined))
        } else {
            ctx.num(0)
        };
    }

    build_smart_mul_expr_from_factors_root(ctx, &filtered)
}

pub(super) fn build_nonexpanding_locally_simplified_mul_expr_from_factors_root(
    ctx: &mut Context,
    factors: &[ExprId],
) -> ExprId {
    let mut saw_zero = false;
    let mut saw_nonfinite = false;
    let mut filtered = Vec::new();

    for factor in factors.iter().copied() {
        match ctx.get(factor) {
            Expr::Number(n) if n.is_zero() => {
                saw_zero = true;
            }
            Expr::Number(n) if n.is_one() => {}
            Expr::Constant(Constant::Undefined | Constant::Infinity) => {
                saw_nonfinite = true;
                filtered.push(factor);
            }
            Expr::Neg(inner) if matches!(ctx.get(*inner), Expr::Constant(Constant::Infinity)) => {
                saw_nonfinite = true;
                filtered.push(factor);
            }
            _ => filtered.push(factor),
        }
    }

    if saw_zero {
        return if saw_nonfinite {
            ctx.add(Expr::Constant(Constant::Undefined))
        } else {
            ctx.num(0)
        };
    }

    build_mul_expr_from_factors_root(ctx, &filtered)
}

pub(super) fn extract_common_multiplicative_residual_sum_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId)> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 2 {
        return None;
    }

    let factor_lists: Vec<Vec<_>> = view
        .terms
        .iter()
        .map(|(term_expr, _)| flatten_mul_chain(ctx, *term_expr))
        .collect();
    if factor_lists.iter().any(Vec::is_empty) {
        return None;
    }

    let mut used_by_term = factor_lists
        .iter()
        .map(|factors| vec![false; factors.len()])
        .collect::<Vec<_>>();
    let mut common = Vec::new();

    for (first_index, first_factor) in factor_lists[0].iter().copied().enumerate() {
        let Some(second_index) =
            factor_lists[1]
                .iter()
                .enumerate()
                .find_map(|(candidate_index, factor)| {
                    (!used_by_term[1][candidate_index]
                        && compare_expr(ctx, *factor, first_factor) == Ordering::Equal)
                        .then_some(candidate_index)
                })
        else {
            continue;
        };

        common.push(first_factor);
        used_by_term[0][first_index] = true;
        used_by_term[1][second_index] = true;
    }

    if common.is_empty() {
        return None;
    }

    let residual_terms: Vec<_> = view
        .terms
        .iter()
        .copied()
        .enumerate()
        .map(|(term_index, (_term_expr, term_sign))| {
            let residual_factors = factor_lists[term_index]
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(factor_index, factor)| {
                    (!used_by_term[term_index][factor_index]).then_some(factor)
                })
                .collect::<Vec<_>>();
            (
                build_mul_expr_from_factors_root(ctx, &residual_factors),
                term_sign,
            )
        })
        .collect();

    let common_factor = build_mul_expr_from_factors_root(ctx, &common);
    let residual_expr = build_signed_sum_expr_root(ctx, &residual_terms);
    let one = ctx.num(1);
    if compare_expr(ctx, common_factor, one) == Ordering::Equal
        || compare_expr(ctx, residual_expr, expr) == Ordering::Equal
    {
        return None;
    }
    Some((common_factor, residual_expr))
}

pub(super) fn try_standard_abs_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if !ctx.is_builtin(*fn_id, BuiltinFn::Abs) {
        return None;
    }
    let inner = args.first().copied();

    let parent_ctx = build_root_shortcut_parent_ctx(options, ctx, expr);

    // A NON-RATIONAL constant argument with a provable sign (π, e, φ, surds,
    // `π+1`, `2π`, and their negations) is owned by the sign rules of the
    // full pipeline: EvaluateAbs here only strips the Neg (`|−π| → |π|`) and
    // this single-shot chain returns a form nobody revisits, so `abs(-pi)`
    // stayed `|π|` without steps while steps mode (Core + PostCleanup's
    // Abs Under Positivity) folded it to `π`. Rational-foldable arguments
    // stay on the shortcut — EvaluateAbs resolves them completely
    // (`|−3| → 3`) with no divergence. Mirrors AbsSubNormalize's guard.
    if let Some(inner) = inner {
        if !cas_math::expr_predicates::contains_variable(ctx, inner)
            && cas_math::numeric_eval::as_rational_const(ctx, inner).is_none()
        {
            let vd = parent_ctx.value_domain();
            if crate::helpers::prove_positive(ctx, inner, vd) == crate::Proof::Proven {
                return None;
            }
            // Fold `−(−u)` to `u` by hand: the prover does not normalize a
            // double Neg, so `|−π|` would slip through as Unknown.
            let neg_inner = match ctx.get(inner) {
                Expr::Neg(u) => *u,
                _ => ctx.add(Expr::Neg(inner)),
            };
            if crate::helpers::prove_positive(ctx, neg_inner, vd) == crate::Proof::Proven {
                return None;
            }
        }
    }

    let evaluate = crate::rules::functions::EvaluateAbsRule;
    if let Some(rewrite) = crate::rule::Rule::apply(&evaluate, ctx, expr, &parent_ctx) {
        return Some(finish_standard_root_shortcut(
            ctx,
            expr,
            rewrite,
            "Evaluate Absolute Value",
            collect_steps,
        ));
    }

    let numeric_factor = crate::rules::functions::AbsPositiveFactorRule;
    if let Some(rewrite) = crate::rule::Rule::apply(&numeric_factor, ctx, expr, &parent_ctx) {
        return Some(finish_standard_root_shortcut(
            ctx,
            expr,
            rewrite,
            "Abs Positive Factor",
            collect_steps,
        ));
    }

    let sub_normalize = crate::rules::functions::AbsSubNormalizeRule;
    if let Some(rewrite) = crate::rule::Rule::apply(&sub_normalize, ctx, expr, &parent_ctx) {
        return Some(finish_standard_root_shortcut(
            ctx,
            expr,
            rewrite,
            "Abs Sub Normalize",
            collect_steps,
        ));
    }

    let quotient_sub_normalize = crate::rules::functions::AbsQuotientSubNormalizeRule;
    if let Some(rewrite) = crate::rule::Rule::apply(&quotient_sub_normalize, ctx, expr, &parent_ctx)
    {
        return Some(finish_standard_root_shortcut(
            ctx,
            expr,
            rewrite,
            "Abs Quotient Sub Normalize",
            collect_steps,
        ));
    }

    None
}

pub(super) fn run_common_scale_rebuilt_root_shortcut_simplify(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    before: ExprId,
    rewritten: ExprId,
    collect_steps: bool,
) -> (ExprId, Vec<Step>) {
    run_named_rebuilt_root_shortcut_simplify(
        options,
        ctx,
        before,
        rewritten,
        "Collapse Common-Scale Equivalent Difference",
        "Collapse Common-Scale Equivalent Difference",
        collect_steps,
    )
}

pub(super) fn transplant_expr_subtree(src: &Context, id: ExprId, dst: &mut Context) -> ExprId {
    match src.get(id) {
        Expr::Number(n) => dst.add(Expr::Number(n.clone())),
        Expr::Constant(c) => dst.add(Expr::Constant(c.clone())),
        Expr::Variable(sym) => dst.var(src.sym_name(*sym)),
        Expr::SessionRef(r) => dst.add(Expr::SessionRef(*r)),
        Expr::Add(lhs, rhs) => {
            let lhs = transplant_expr_subtree(src, *lhs, dst);
            let rhs = transplant_expr_subtree(src, *rhs, dst);
            dst.add(Expr::Add(lhs, rhs))
        }
        Expr::Sub(lhs, rhs) => {
            let lhs = transplant_expr_subtree(src, *lhs, dst);
            let rhs = transplant_expr_subtree(src, *rhs, dst);
            dst.add(Expr::Sub(lhs, rhs))
        }
        Expr::Mul(lhs, rhs) => {
            let lhs = transplant_expr_subtree(src, *lhs, dst);
            let rhs = transplant_expr_subtree(src, *rhs, dst);
            dst.add(Expr::Mul(lhs, rhs))
        }
        Expr::Div(lhs, rhs) => {
            let lhs = transplant_expr_subtree(src, *lhs, dst);
            let rhs = transplant_expr_subtree(src, *rhs, dst);
            dst.add(Expr::Div(lhs, rhs))
        }
        Expr::Pow(lhs, rhs) => {
            let lhs = transplant_expr_subtree(src, *lhs, dst);
            let rhs = transplant_expr_subtree(src, *rhs, dst);
            dst.add(Expr::Pow(lhs, rhs))
        }
        Expr::Neg(inner) => {
            let inner = transplant_expr_subtree(src, *inner, dst);
            dst.add(Expr::Neg(inner))
        }
        Expr::Function(name, args) => {
            let args = args
                .iter()
                .map(|&arg| transplant_expr_subtree(src, arg, dst))
                .collect();
            let name = dst.intern_symbol(src.sym_name(*name));
            dst.add(Expr::Function(name, args))
        }
        Expr::Matrix { rows, cols, data } => {
            let data = data
                .iter()
                .map(|&elem| transplant_expr_subtree(src, elem, dst))
                .collect();
            dst.add(Expr::Matrix {
                rows: *rows,
                cols: *cols,
                data,
            })
        }
        Expr::Hold(inner) => {
            let inner = transplant_expr_subtree(src, *inner, dst);
            dst.add(Expr::Hold(inner))
        }
    }
}

pub(super) fn enter_isolated_simplify_probe() -> Option<IsolatedSimplifyNestingGuard> {
    ISOLATED_SIMPLIFY_NESTING.with(|depth| {
        if depth.get() >= ISOLATED_SIMPLIFY_MAX_NESTING {
            return None;
        }
        depth.set(depth.get() + 1);
        Some(IsolatedSimplifyNestingGuard)
    })
}

/// Fingerprint of every `SimplifyOptions` axis that can change a probe's
/// simplification outcome. Payload-free enums without `Hash` go through
/// `std::mem::discriminant`; time/phase budgets are deliberately excluded
/// (stable within an eval, and a budget-weakened `false` is the same
/// conservative answer the repeated probe would produce today).
pub(super) fn isolated_probe_options_fingerprint(options: &crate::phase::SimplifyOptions) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = rustc_hash::FxHasher::default();
    options.enable_transform.hash(&mut hasher);
    options.expand_mode.hash(&mut hasher);
    options.goal.hash(&mut hasher);
    std::mem::discriminant(&options.simplify_purpose).hash(&mut hasher);
    std::mem::discriminant(&options.rationalize.auto_level).hash(&mut hasher);
    options.shared.semantics.hash(&mut hasher);
    options.shared.context_mode.hash(&mut hasher);
    std::mem::discriminant(&options.shared.expand_policy).hash(&mut hasher);
    std::mem::discriminant(&options.shared.log_expand_policy).hash(&mut hasher);
    std::mem::discriminant(&options.shared.autoexpand_binomials).hash(&mut hasher);
    std::mem::discriminant(&options.shared.heuristic_poly).hash(&mut hasher);
    hasher.finish()
}

pub(super) fn expr_contains_explicit_negative_marker_local(ctx: &Context, expr: ExprId) -> bool {
    let mut stack = vec![expr];
    while let Some(current) = stack.pop() {
        match ctx.get(current) {
            Expr::Number(n) => {
                if n.is_negative() {
                    return true;
                }
            }
            Expr::Neg(_) => return true,
            Expr::Add(lhs, rhs)
            | Expr::Sub(lhs, rhs)
            | Expr::Mul(lhs, rhs)
            | Expr::Div(lhs, rhs)
            | Expr::Pow(lhs, rhs) => {
                stack.push(*lhs);
                stack.push(*rhs);
            }
            Expr::Function(_, args) => stack.extend(args.iter().copied()),
            Expr::Matrix { data, .. } => stack.extend(data.iter().copied()),
            Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) | Expr::Hold(_) => {}
        }
    }

    false
}

pub(super) fn is_plain_division_difference_root(ctx: &Context, expr: ExprId) -> bool {
    matches!(
        ctx.get(expr),
        Expr::Sub(lhs, rhs)
            if matches!(ctx.get(*lhs), Expr::Div(_, _))
                && matches!(ctx.get(*rhs), Expr::Div(_, _))
    )
}

/// Options for an ISOLATED simplify launched from a shortcut path: the
/// defaults plus the ambient pipeline value domain. Without the axis, a
/// default-armed isolated run is a covert domain translator — its RealOnly
/// verdicts get adopted by a complex session (audit 2026-07-30, causa C2;
/// same doctrine as `run_default_simplify`'s probe options).
pub(super) fn isolated_probe_options() -> crate::phase::SimplifyOptions {
    let mut options = crate::phase::SimplifyOptions::default();
    options.shared.semantics.value_domain =
        crate::rules::arithmetic::ambient_pipeline_value_domain();
    options
}

pub(super) fn try_standard_tangent_addition_factor_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() < 2 {
        return None;
    }

    for (index, factor) in factors.iter().copied().enumerate() {
        let Some((lhs_arg, rhs_arg)) = extract_direct_tangent_addition_target_root(ctx, factor)
        else {
            continue;
        };

        let replacement = build_tangent_addition_fraction_root(ctx, lhs_arg, rhs_arg);
        let mut rewritten_factors = factors.clone();
        rewritten_factors[index] = replacement;
        let rewritten = build_mul_expr_from_factors_root(ctx, &rewritten_factors);
        return Some(run_named_rebuilt_root_shortcut_simplify(
            options,
            ctx,
            expr,
            rewritten,
            "Tangent Addition",
            "Tangent Addition",
            collect_steps,
        ));
    }

    None
}

pub(super) fn build_direct_three_linear_shift_expanded_target_root(
    ctx: &mut Context,
    base: ExprId,
    constants: &[BigRational],
) -> Option<ExprId> {
    let one = BigRational::from_integer(1.into());
    let two = BigRational::from_integer(2.into());
    let three = BigRational::from_integer(3.into());
    if constants != [one, two, three] {
        return None;
    }

    let two_expr = ctx.num(2);
    let three_expr = ctx.num(3);
    let six_expr = ctx.num(6);
    let eleven_expr = ctx.num(11);
    let base_sq = ctx.add(Expr::Pow(base, two_expr));
    let base_cu = ctx.add(Expr::Pow(base, three_expr));
    let six_base_sq = smart_mul(ctx, six_expr, base_sq);
    let eleven_base = smart_mul(ctx, eleven_expr, base);
    Some(build_balanced_add(
        ctx,
        &[base_cu, six_base_sq, eleven_base, six_expr],
    ))
}

pub(super) fn build_direct_two_linear_shift_expanded_target_root(
    ctx: &mut Context,
    base: ExprId,
    constants: &[BigRational],
) -> Option<ExprId> {
    if constants.len() != 2 {
        return None;
    }

    let two = ctx.num(2);
    let base_sq = ctx.add(Expr::Pow(base, two));
    let linear_coeff = constants[0].clone() + constants[1].clone();
    let constant_term = constants[0].clone() * constants[1].clone();

    let mut terms = vec![base_sq];
    if !linear_coeff.is_zero() {
        let coeff_expr = ctx.add(Expr::Number(linear_coeff));
        terms.push(smart_mul(ctx, coeff_expr, base));
    }
    if !constant_term.is_zero() {
        terms.push(ctx.add(Expr::Number(constant_term)));
    }

    Some(build_balanced_add(ctx, &terms))
}

pub(super) fn is_pure_arithmetic_constant_expr_root(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Number(_) => true,
        Expr::Neg(inner) => is_pure_arithmetic_constant_expr_root(ctx, *inner),
        Expr::Add(lhs, rhs)
        | Expr::Sub(lhs, rhs)
        | Expr::Mul(lhs, rhs)
        | Expr::Div(lhs, rhs)
        | Expr::Pow(lhs, rhs) => {
            is_pure_arithmetic_constant_expr_root(ctx, *lhs)
                && is_pure_arithmetic_constant_expr_root(ctx, *rhs)
        }
        Expr::Constant(_)
        | Expr::Function(_, _)
        | Expr::Variable(_)
        | Expr::Matrix { .. }
        | Expr::SessionRef(_)
        | Expr::Hold(_) => false,
    }
}

pub(super) fn extract_direct_short_geometric_product_base_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() != 2 {
        return None;
    }

    for (linear, quadratic) in [(factors[0], factors[1]), (factors[1], factors[0])] {
        let (base, constant) = extract_base_plus_constant_root(ctx, linear)?;
        if constant != BigRational::one() {
            continue;
        }
        let quadratic_base = extract_square_plus_one_base_root(ctx, quadratic)?;
        if compare_expr(ctx, base, quadratic_base) == Ordering::Equal {
            return Some(base);
        }
    }

    None
}

pub(super) fn build_direct_short_geometric_sum_expanded_target_root(
    ctx: &mut Context,
    base: ExprId,
) -> ExprId {
    let one = ctx.num(1);
    let two = ctx.num(2);
    let three = ctx.num(3);
    let base_sq = ctx.add(Expr::Pow(base, two));
    let base_cu = ctx.add(Expr::Pow(base, three));
    build_balanced_add(ctx, &[base_cu, base_sq, base, one])
}

pub(super) fn extract_signed_two_factor_product_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(BigRational, [ExprId; 2])> {
    let factors = flatten_mul_chain(ctx, expr);
    let mut numeric_coeff = BigRational::one();
    let mut non_numeric_factors = Vec::with_capacity(2);

    for factor in factors {
        match ctx.get(factor) {
            Expr::Number(n) => numeric_coeff *= n.clone(),
            Expr::Neg(inner) => {
                numeric_coeff = -numeric_coeff;
                non_numeric_factors.push(*inner);
            }
            _ => non_numeric_factors.push(factor),
        }
    }

    if non_numeric_factors.len() != 2
        || (numeric_coeff != BigRational::one()
            && numeric_coeff != BigRational::from_integer((-1).into()))
    {
        return None;
    }

    Some((
        numeric_coeff,
        [non_numeric_factors[0], non_numeric_factors[1]],
    ))
}

pub(super) fn simplify_shallow_small_constant_expr_root(ctx: &mut Context, expr: ExprId) -> ExprId {
    let mut current = expr;

    loop {
        let mut changed = false;

        if let Expr::Neg(inner) = ctx.get(current).clone() {
            if let Expr::Number(n) = ctx.get(inner) {
                current = ctx.add(Expr::Number(-n.clone()));
                changed = true;
            }
        }
        if changed {
            continue;
        }

        if let Some(rewrite) = try_rewrite_combine_constants_expr(ctx, current) {
            if compare_expr(ctx, rewrite.rewritten, current) != Ordering::Equal {
                current = rewrite.rewritten;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    strip_multiplicative_one_root(ctx, current)
}

fn extract_small_exact_constant_function_value_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    extract_special_angle_exact_value_root(ctx, expr)
        .map(|rewritten| strip_multiplicative_one_root(ctx, rewritten))
}

pub(super) fn try_rewrite_small_constant_function_wrapper_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    match ctx.get(expr).clone() {
        Expr::Neg(inner) => {
            let rewritten_inner = extract_small_exact_constant_function_value_root(ctx, inner)?;
            let rewritten = ctx.add(Expr::Neg(rewritten_inner));
            Some(simplify_shallow_small_constant_expr_root(ctx, rewritten))
        }
        Expr::Add(lhs, rhs) => {
            if matches!(ctx.get(lhs), Expr::Number(_)) {
                let rewritten_rhs = extract_small_exact_constant_function_value_root(ctx, rhs)?;
                let rewritten = ctx.add(Expr::Add(lhs, rewritten_rhs));
                return Some(simplify_shallow_small_constant_expr_root(ctx, rewritten));
            }
            if matches!(ctx.get(rhs), Expr::Number(_)) {
                let rewritten_lhs = extract_small_exact_constant_function_value_root(ctx, lhs)?;
                let rewritten = ctx.add(Expr::Add(rewritten_lhs, rhs));
                return Some(simplify_shallow_small_constant_expr_root(ctx, rewritten));
            }
            None
        }
        Expr::Sub(lhs, rhs) => {
            if matches!(ctx.get(lhs), Expr::Number(_)) {
                let rewritten_rhs = extract_small_exact_constant_function_value_root(ctx, rhs)?;
                let rewritten = ctx.add(Expr::Sub(lhs, rewritten_rhs));
                return Some(simplify_shallow_small_constant_expr_root(ctx, rewritten));
            }
            if matches!(ctx.get(rhs), Expr::Number(_)) {
                let rewritten_lhs = extract_small_exact_constant_function_value_root(ctx, lhs)?;
                let rewritten = ctx.add(Expr::Sub(rewritten_lhs, rhs));
                return Some(simplify_shallow_small_constant_expr_root(ctx, rewritten));
            }
            None
        }
        Expr::Mul(lhs, rhs) => {
            if matches!(ctx.get(lhs), Expr::Number(_)) {
                let rewritten_rhs = extract_small_exact_constant_function_value_root(ctx, rhs)?;
                let rewritten = smart_mul(ctx, lhs, rewritten_rhs);
                return Some(simplify_shallow_small_constant_expr_root(ctx, rewritten));
            }
            if matches!(ctx.get(rhs), Expr::Number(_)) {
                let rewritten_lhs = extract_small_exact_constant_function_value_root(ctx, lhs)?;
                let rewritten = smart_mul(ctx, rewritten_lhs, rhs);
                return Some(simplify_shallow_small_constant_expr_root(ctx, rewritten));
            }
            None
        }
        Expr::Div(lhs, rhs) => {
            if matches!(ctx.get(rhs), Expr::Number(_)) {
                let rewritten_lhs = extract_small_exact_constant_function_value_root(ctx, lhs)?;
                let rewritten = ctx.add(Expr::Div(rewritten_lhs, rhs));
                return Some(simplify_shallow_small_constant_expr_root(ctx, rewritten));
            }
            if matches!(ctx.get(lhs), Expr::Number(_)) {
                let rewritten_rhs = extract_small_exact_constant_function_value_root(ctx, rhs)?;
                let rewritten = ctx.add(Expr::Div(lhs, rewritten_rhs));
                return Some(simplify_shallow_small_constant_expr_root(ctx, rewritten));
            }
            None
        }
        _ => None,
    }
}

pub(super) fn constant_like_isolated_simplify_profile_label_root(
    ctx: &Context,
    expr: ExprId,
) -> &'static str {
    if is_function_free_arithmetic_expr_root(ctx, expr)
        && expr_contains_sqrt_or_half_power_local(ctx, expr)
    {
        "root.mul.14c6a.constant_like.isolated_simplify.function_free_radical"
    } else if is_function_free_arithmetic_expr_root(ctx, expr) {
        "root.mul.14c6b.constant_like.isolated_simplify.function_free_symbolic"
    } else if expr_contains_builtin_function_local(ctx, expr) {
        "root.mul.14c6c.constant_like.isolated_simplify.constant_function"
    } else {
        "root.mul.14c6d.constant_like.isolated_simplify.other"
    }
}

pub(super) fn build_small_two_chunk_additive_partitions_root(
    ctx: &mut Context,
    terms: &[(ExprId, Sign)],
) -> Vec<(ExprId, ExprId)> {
    if !(2..=8).contains(&terms.len()) {
        return Vec::new();
    }

    let mut partitions = Vec::new();
    let full_mask = (1usize << terms.len()) - 1;
    for left_mask in 1..full_mask {
        let right_mask = full_mask ^ left_mask;
        if right_mask == 0 || (left_mask & 1) == 0 {
            continue;
        }

        let mut left_terms = Vec::new();
        let mut right_terms = Vec::new();
        for (index, term) in terms.iter().copied().enumerate() {
            if ((left_mask >> index) & 1) == 1 {
                left_terms.push(term);
            } else {
                right_terms.push(term);
            }
        }

        if left_terms.is_empty() || right_terms.is_empty() {
            continue;
        }

        let left_expr = build_signed_sum_expr_root(ctx, &left_terms);
        let right_expr = build_signed_sum_expr_root(ctx, &right_terms);
        partitions.push((left_expr, right_expr));
    }

    partitions
}

pub(super) fn extend_additive_terms_from_expr_root(
    ctx: &mut Context,
    expr: ExprId,
    out: &mut smallvec::SmallVec<[(ExprId, Sign); 8]>,
) {
    if matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _)) {
        out.extend(AddView::from_expr(ctx, expr).terms);
    } else {
        let zero = ctx.num(0);
        if compare_expr(ctx, expr, zero) != Ordering::Equal {
            out.push((expr, Sign::Pos));
        }
    }
}

pub(super) fn try_finalize_trivial_additive_closure_root(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    if !matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _)) {
        return None;
    }

    let mut current = expr;
    let mut shortcut_steps = Vec::new();
    let mut changed = false;

    loop {
        if let Some((rewritten, mut exact_steps)) =
            try_standard_exact_additive_pair_chain_shortcut(options, ctx, current, collect_steps)
        {
            if compare_expr(ctx, rewritten, current) != Ordering::Equal {
                current = rewritten;
                changed = true;
                if collect_steps {
                    shortcut_steps.append(&mut exact_steps);
                }
                continue;
            }
        }

        let parent_ctx = build_root_shortcut_parent_ctx(options, ctx, current);
        let late_rewrite = crate::rules::arithmetic::SubSelfToZeroRule
            .apply(ctx, current, &parent_ctx)
            .map(|rewrite| (rewrite, "Subtract Self"))
            .or_else(|| {
                crate::rules::arithmetic::AddInverseRule
                    .apply(ctx, current, &parent_ctx)
                    .map(|rewrite| (rewrite, "Add Inverse"))
            })
            .or_else(|| {
                crate::rules::arithmetic::CombineConstantsRule
                    .apply(ctx, current, &parent_ctx)
                    .map(|rewrite| (rewrite, "Combine Constants"))
            })
            .or_else(|| {
                crate::rules::arithmetic::AddZeroRule
                    .apply(ctx, current, &parent_ctx)
                    .map(|rewrite| (rewrite, "Identity Property of Addition"))
            });

        if let Some((rewrite, rule_name)) = late_rewrite {
            if compare_expr(ctx, rewrite.new_expr, current) != Ordering::Equal {
                let before = current;
                current = rewrite.new_expr;
                changed = true;
                if collect_steps {
                    shortcut_steps.push(build_root_shortcut_step_from_rewrite(
                        ctx, before, &rewrite, rule_name,
                    ));
                }
                continue;
            }
        }

        break;
    }

    if !changed {
        return None;
    }

    if collect_steps {
        if let Some(first) = shortcut_steps.first_mut() {
            first.global_before = Some(expr);
        }
        if let Some(last) = shortcut_steps.last_mut() {
            last.global_after = Some(current);
        }
    }

    Some((current, shortcut_steps))
}

pub(super) fn try_standard_shared_passthrough_small_pow_expansion_shortcut(
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let (lhs_core, rhs_core) = extract_shared_additive_passthrough_sub_cores_root(ctx, expr)?;
    if !matches_direct_small_pow_expansion_pair_root(ctx, lhs_core, rhs_core) {
        return None;
    }

    let zero = ctx.num(0);
    let residual_expr = ctx.add(Expr::Sub(lhs_core, rhs_core));
    Some(finish_standard_root_shortcut(
        ctx,
        expr,
        crate::rule::Rewrite::with_local(
            zero,
            "Collapse Exact Zero Additive Subexpression",
            residual_expr,
            zero,
        ),
        "Collapse Exact Zero Additive Subexpression",
        collect_steps,
    ))
}

/// True when `expr` carries the imaginary unit anywhere.
fn expr_contains_imaginary_unit(ctx: &Context, root: ExprId) -> bool {
    let mut stack = vec![root];
    while let Some(expr) = stack.pop() {
        match ctx.get(expr) {
            Expr::Constant(cas_ast::Constant::I) => return true,
            Expr::Constant(_) | Expr::Number(_) | Expr::Variable(_) | Expr::SessionRef(_) => {}
            Expr::Add(l, r)
            | Expr::Sub(l, r)
            | Expr::Mul(l, r)
            | Expr::Div(l, r)
            | Expr::Pow(l, r) => {
                stack.push(*l);
                stack.push(*r);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(*inner),
            Expr::Function(_, args) => stack.extend(args.iter().copied()),
            Expr::Matrix { data, .. } => stack.extend(data.iter().copied()),
        }
    }
    false
}

/// Complex-mode guard for the perfect-power extraction shortcuts: extraction
/// of a positive real scale is complex-valid (`√18 → 3·√2` keeps working),
/// but the shortcut's collapse of an `i`-bearing radicand runs real-only
/// reasoning on a complex literal — `sqrt(16·i⁴)` published `4·i²` (= −4,
/// true value 4) with steps=off while steps=on folded `i⁴ → 1` first and got
/// `4` (audit 2026-07-30, ficha S4-002). Declining hands the literal to the
/// pipeline's complex-aware power folding, so both modes converge. Symbolic
/// radicands keep today's behavior (pinned under `assume_scope: real`).
pub(super) fn extract_shortcut_declines_for_value_domain(
    options: &crate::phase::SimplifyOptions,
    ctx: &Context,
    expr: ExprId,
) -> bool {
    options.shared.semantics.value_domain != crate::semantics::ValueDomain::RealOnly
        && expr_contains_imaginary_unit(ctx, expr)
}

pub(super) fn try_standard_numeric_add_chain_shortcut(
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let Expr::Add(left, right) = ctx.get(expr) else {
        return None;
    };
    let (path, number_side, reducible_side) = match (ctx.get(*left), ctx.get(*right)) {
        (Expr::Number(_), _) => (crate::step::PathStep::Right, *left, *right),
        (_, Expr::Number(_)) => (crate::step::PathStep::Left, *right, *left),
        _ => return None,
    };

    let inner = try_rewrite_combine_constants_expr(ctx, reducible_side)?;
    if !matches!(ctx.get(inner.rewritten), Expr::Number(_)) {
        return None;
    }

    let after_inner = match path {
        crate::step::PathStep::Left => ctx.add(Expr::Add(inner.rewritten, number_side)),
        crate::step::PathStep::Right => ctx.add(Expr::Add(number_side, inner.rewritten)),
        _ => unreachable!(),
    };
    let outer = try_rewrite_combine_constants_expr(ctx, after_inner)?;

    let mut shortcut_steps = Vec::new();
    if collect_steps {
        let mut inner_step = Step::new(
            &inner.description,
            "Combine Constants",
            reducible_side,
            inner.rewritten,
            vec![path.clone()],
            Some(ctx),
        );
        inner_step.before = reducible_side;
        inner_step.after = inner.rewritten;
        inner_step.global_before = Some(expr);
        inner_step.global_after = Some(after_inner);
        shortcut_steps.push(inner_step);

        let outer_step = Step::new_compact(
            &outer.description,
            "Combine Constants",
            after_inner,
            outer.rewritten,
        );
        let mut outer_step = outer_step;
        outer_step.global_before = Some(after_inner);
        outer_step.global_after = Some(outer.rewritten);
        shortcut_steps.push(outer_step);
    }

    Some((outer.rewritten, shortcut_steps))
}

pub(super) fn multiset_matches_exact(
    ctx: &Context,
    actual: &[ExprId],
    expected: &[ExprId],
) -> bool {
    if actual.len() != expected.len() {
        return false;
    }

    let mut used = [false; 3];
    for wanted in expected {
        let mut matched = false;
        for (idx, candidate) in actual.iter().enumerate() {
            if used[idx] {
                continue;
            }
            if expr_eq(ctx, *candidate, *wanted) {
                used[idx] = true;
                matched = true;
                break;
            }
        }
        if !matched {
            return false;
        }
    }

    true
}

pub(super) fn is_exact_two_ab_product(
    ctx: &mut Context,
    expr: ExprId,
    a: ExprId,
    b: ExprId,
) -> bool {
    let view = MulView::from_expr(ctx, expr);
    if view.factors.len() != 3 {
        return false;
    }

    let two = ctx.num(2);
    multiset_matches_exact(ctx, &view.factors, &[two, a, b])
}

pub(super) fn allow_hidden_solve_root_scalar_multiple_shortcut(opts: &SimplifyOptions) -> bool {
    match opts.simplify_purpose {
        crate::SimplifyPurpose::Eval => {
            opts.shared.context_mode == crate::options::ContextMode::Solve
        }
        crate::SimplifyPurpose::SolvePrepass => {
            cas_solver_core::solve_safety_policy::safe_for_prepass(
                crate::SolveSafety::NeedsCondition(crate::ConditionClass::Definability),
            )
        }
        crate::SimplifyPurpose::SolveTactic => {
            let domain_mode = opts.shared.semantics.domain_mode;
            cas_solver_core::solve_safety_policy::safe_for_tactic_with_domain_flags(
                crate::SolveSafety::NeedsCondition(crate::ConditionClass::Definability),
                matches!(domain_mode, crate::DomainMode::Assume),
                matches!(domain_mode, crate::DomainMode::Strict),
            )
        }
    }
}

pub(super) fn allow_definability_root_shortcuts(opts: &SimplifyOptions) -> bool {
    match opts.simplify_purpose {
        crate::SimplifyPurpose::Eval => true,
        crate::SimplifyPurpose::SolvePrepass => {
            cas_solver_core::solve_safety_policy::safe_for_prepass(
                crate::SolveSafety::NeedsCondition(crate::ConditionClass::Definability),
            )
        }
        crate::SimplifyPurpose::SolveTactic => {
            let domain_mode = opts.shared.semantics.domain_mode;
            cas_solver_core::solve_safety_policy::safe_for_tactic_with_domain_flags(
                crate::SolveSafety::NeedsCondition(crate::ConditionClass::Definability),
                matches!(domain_mode, crate::DomainMode::Assume),
                matches!(domain_mode, crate::DomainMode::Strict),
            )
        }
    }
}

pub(super) fn prove_positive_literal_fast(ctx: &Context, expr: ExprId) -> Option<crate::Proof> {
    use crate::Proof;

    if is_positive_literal(ctx, expr) {
        return Some(Proof::Proven);
    }
    if is_negative_literal(ctx, expr) {
        return Some(Proof::Disproven);
    }

    match ctx.get(expr) {
        Expr::Number(n) if n.is_zero() => Some(Proof::Disproven),
        _ => None,
    }
}

pub(super) fn try_finish_dirichlet_kernel_root_shortcut(
    simplifier: &mut crate::Simplifier,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>, crate::phase::PipelineStats)> {
    let result = crate::try_dirichlet_kernel_identity_pub(&mut simplifier.context, expr)?;
    let zero = simplifier.context.num(0);
    let steps = if collect_steps {
        vec![Step::new(
            &format!(
                "Dirichlet Kernel Identity: 1 + 2Σcos(kx) = sin((n+½)x)/sin(x/2) for n={}",
                result.n
            ),
            "Trig Summation Identity",
            expr,
            zero,
            Vec::new(),
            Some(&simplifier.context),
        )]
    } else {
        Vec::new()
    };
    simplifier.clear_sticky_implicit_domain();
    Some((zero, steps, crate::phase::PipelineStats::default()))
}

pub(super) fn try_hidden_solve_root_exact_two_term_scalar_multiple_shortcut(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let Expr::Div(num, den) = ctx.get(expr) else {
        return None;
    };
    let (num_left, num_right) = match ctx.get(*num) {
        Expr::Add(left, right) => (*left, *right),
        _ => return None,
    };
    let (den_left, den_right) = match ctx.get(*den) {
        Expr::Add(left, right) => (*left, *right),
        _ => return None,
    };

    let (num_l_coeff, num_l_base) = extract_coef_and_base(ctx, num_left);
    let (num_r_coeff, num_r_base) = extract_coef_and_base(ctx, num_right);
    let (den_l_coeff, den_l_base) = extract_coef_and_base(ctx, den_left);
    let (den_r_coeff, den_r_base) = extract_coef_and_base(ctx, den_right);

    if num_l_coeff.is_zero()
        || num_r_coeff.is_zero()
        || den_l_coeff.is_zero()
        || den_r_coeff.is_zero()
    {
        return None;
    }

    let ratio = if expr_eq(ctx, num_l_base, den_l_base) && expr_eq(ctx, num_r_base, den_r_base) {
        let left_ratio = den_l_coeff / num_l_coeff;
        let right_ratio = den_r_coeff / num_r_coeff;
        if left_ratio != right_ratio || left_ratio.is_zero() {
            return None;
        }
        left_ratio
    } else if expr_eq(ctx, num_l_base, den_r_base) && expr_eq(ctx, num_r_base, den_l_base) {
        let left_ratio = den_r_coeff / num_l_coeff;
        let right_ratio = den_l_coeff / num_r_coeff;
        if left_ratio != right_ratio || left_ratio.is_zero() {
            return None;
        }
        left_ratio
    } else {
        return None;
    };

    let result_ratio = BigRational::from_integer(1.into()) / ratio;
    Some(ctx.add(Expr::Number(result_ratio)))
}

pub(super) fn try_standard_exact_two_term_scalar_multiple_shortcut(
    ctx: &mut Context,
    expr: ExprId,
    domain_mode: crate::DomainMode,
    value_domain: crate::semantics::ValueDomain,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    use crate::{ImplicitCondition, Predicate};

    let Expr::Div(num, den) = ctx.get(expr) else {
        return None;
    };
    let (num_left, num_right) = match ctx.get(*num) {
        Expr::Add(left, right) => (*left, *right),
        _ => return None,
    };
    let (den_left, den_right) = match ctx.get(*den) {
        Expr::Add(left, right) => (*left, *right),
        _ => return None,
    };

    let (num_l_coeff, num_l_base) = extract_coef_and_base(ctx, num_left);
    let (num_r_coeff, num_r_base) = extract_coef_and_base(ctx, num_right);
    let (den_l_coeff, den_l_base) = extract_coef_and_base(ctx, den_left);
    let (den_r_coeff, den_r_base) = extract_coef_and_base(ctx, den_right);

    if num_l_coeff.is_zero()
        || num_r_coeff.is_zero()
        || den_l_coeff.is_zero()
        || den_r_coeff.is_zero()
    {
        return None;
    }

    let ratio = if expr_eq(ctx, num_l_base, den_l_base) && expr_eq(ctx, num_r_base, den_r_base) {
        let left_ratio = den_l_coeff / num_l_coeff.clone();
        let right_ratio = den_r_coeff / num_r_coeff.clone();
        if left_ratio != right_ratio || left_ratio.is_zero() {
            return None;
        }
        left_ratio
    } else if expr_eq(ctx, num_l_base, den_r_base) && expr_eq(ctx, num_r_base, den_l_base) {
        let left_ratio = den_r_coeff / num_l_coeff.clone();
        let right_ratio = den_l_coeff / num_r_coeff.clone();
        if left_ratio != right_ratio || left_ratio.is_zero() {
            return None;
        }
        left_ratio
    } else {
        return None;
    };

    let common = ctx.add(Expr::Add(num_l_base, num_r_base));
    let decision = crate::oracle_allows_with_hint(
        ctx,
        domain_mode,
        value_domain,
        &Predicate::NonZero(common),
        "Simplify Nested Fraction",
    );
    if !decision.allow {
        return None;
    }

    let num_coeff_expr = ctx.add(Expr::Number(num_l_coeff.clone()));
    let den_coeff_expr = ctx.add(Expr::Number((num_l_coeff * ratio.clone()).clone()));
    let factored_num = mul2_raw(ctx, num_coeff_expr, common);
    let factored_den = mul2_raw(ctx, den_coeff_expr, common);
    let factored_form = ctx.add(Expr::Div(factored_num, factored_den));
    let result = ctx.add(Expr::Number(BigRational::from_integer(1.into()) / ratio));

    let mut shortcut_steps = Vec::new();
    if collect_steps {
        let mut factor_step = Step::new_compact(
            &format!(
                "Factor by GCD: {}",
                cas_formatter::DisplayExpr {
                    context: ctx,
                    id: common
                }
            ),
            "Simplify Nested Fraction",
            expr,
            factored_form,
        );
        factor_step.global_before = Some(expr);
        factor_step.global_after = Some(factored_form);
        factor_step.importance = crate::step::ImportanceLevel::High;
        shortcut_steps.push(factor_step);

        let mut cancel_step = Step::new_compact(
            "Cancel common factor",
            "Simplify Nested Fraction",
            factored_form,
            result,
        );
        cancel_step.global_before = Some(factored_form);
        cancel_step.global_after = Some(result);
        cancel_step.importance = crate::step::ImportanceLevel::High;
        let meta = cancel_step.meta_mut();
        meta.assumption_events = decision.assumption_events(ctx, common);
        meta.required_conditions
            .push(ImplicitCondition::NonZero(common));
        shortcut_steps.push(cancel_step);
    }

    Some((result, shortcut_steps))
}

pub(super) fn is_plain_symbolic_cube_trinomial_after_core(ctx: &Context, expr: ExprId) -> bool {
    let terms = AddView::from_expr(ctx, expr).terms;
    if terms.len() != 3 {
        return false;
    }

    let mut square_bases = [None, None];
    let mut square_count = 0usize;
    let mut cross_atoms = None;

    for (term, sign) in terms {
        if sign == Sign::Pos {
            if let Some(base) = square_of_symbolic_atom(ctx, term) {
                if square_count >= square_bases.len() {
                    return false;
                }
                square_bases[square_count] = Some(base);
                square_count += 1;
                continue;
            }
        }

        if cross_atoms.is_none() {
            if let Some((left, right)) = symbolic_cross_term_atoms(ctx, term) {
                cross_atoms = Some((left, right));
                continue;
            }
        }

        return false;
    }

    let Some(left_square) = square_bases[0] else {
        return false;
    };
    let Some(right_square) = square_bases[1] else {
        return false;
    };
    let Some((cross_left, cross_right)) = cross_atoms else {
        return false;
    };

    !expr_eq(ctx, left_square, right_square)
        && ((expr_eq(ctx, left_square, cross_left) && expr_eq(ctx, right_square, cross_right))
            || (expr_eq(ctx, left_square, cross_right) && expr_eq(ctx, right_square, cross_left)))
}

fn is_exact_gaussian_noop_component(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Number(_) | Expr::Constant(cas_ast::Constant::I) => true,
        Expr::Neg(inner) => is_exact_gaussian_noop_component(ctx, *inner),
        Expr::Mul(left, right) => {
            (matches!(ctx.get(*left), Expr::Number(_))
                && matches!(ctx.get(*right), Expr::Constant(cas_ast::Constant::I)))
                || (matches!(ctx.get(*left), Expr::Constant(cas_ast::Constant::I))
                    && matches!(ctx.get(*right), Expr::Number(_)))
        }
        Expr::Add(left, right) | Expr::Sub(left, right) => {
            is_exact_gaussian_noop_component(ctx, *left)
                && is_exact_gaussian_noop_component(ctx, *right)
        }
        _ => false,
    }
}

/// Whether a Gaussian-noop component actually carries the imaginary unit `i`.
/// A bare real `Number` (or a sum/product of reals) returns `false`: it is a
/// genuinely real quantity, not a complex one.
fn gaussian_noop_component_has_imaginary_unit(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Constant(cas_ast::Constant::I) => true,
        Expr::Number(_) => false,
        Expr::Neg(inner) => gaussian_noop_component_has_imaginary_unit(ctx, *inner),
        Expr::Mul(left, right)
        | Expr::Add(left, right)
        | Expr::Sub(left, right)
        | Expr::Div(left, right) => {
            gaussian_noop_component_has_imaginary_unit(ctx, *left)
                || gaussian_noop_component_has_imaginary_unit(ctx, *right)
        }
        _ => false,
    }
}

pub(super) fn is_real_domain_complex_noop_root(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Pow(base, exp) => {
            matches!(ctx.get(*base), Expr::Constant(cas_ast::Constant::I))
                && matches!(ctx.get(*exp), Expr::Number(n) if n.is_integer())
        }
        // SOUNDNESS/CONSISTENCY: only a quotient that genuinely involves the
        // imaginary unit `i` is a "complex noop" to be returned unchanged in the
        // RealOnly fast path. `is_exact_gaussian_noop_component` accepts a bare
        // real `Number`, so without the `i` requirement a pure-real `Number/Number`
        // (`6/3`, `7/2`, and critically `1/0`) matched here and was returned
        // UNEVALUATED in plain mode — diverging from `--steps` (which folds them to
        // `2`, `7/2`, `undefined`) and, for `1/0`, reporting a division by zero as a
        // valid result with `ok:true`. Require an actual `i` so real quotients fall
        // through to the folding pipeline.
        Expr::Div(num, den) => {
            is_exact_gaussian_noop_component(ctx, *num)
                && is_exact_gaussian_noop_component(ctx, *den)
                && (gaussian_noop_component_has_imaginary_unit(ctx, *num)
                    || gaussian_noop_component_has_imaginary_unit(ctx, *den))
        }
        _ => false,
    }
}
