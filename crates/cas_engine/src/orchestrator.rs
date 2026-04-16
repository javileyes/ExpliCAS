use crate::best_so_far::{BestSoFar, BestSoFarBudget};
use crate::expand::eager_eval_expand_calls;
use crate::phase::{SimplifyOptions, SimplifyPhase};
use crate::rule::Rule;
use crate::{Simplifier, Step};
use cas_ast::ordering::compare_expr;
use cas_ast::{BuiltinFn, Constant, Context, Expr, ExprId};
use cas_formatter::render_expr;
use cas_math::abs_support::try_unwrap_abs_arg;
use cas_math::arithmetic_rule_support::try_rewrite_combine_constants_expr;
use cas_math::build::mul2_raw;
use cas_math::expansion_rule_support::{try_expand_small_pow_sum_expr, SmallPowExpandPolicy};
use cas_math::expr_extract::{extract_exp_argument, extract_i64_integer};
use cas_math::expr_nary::{build_balanced_add, build_balanced_mul, AddView, MulView, Sign};
use cas_math::expr_rewrite::smart_mul;
use cas_math::factoring_support::try_rewrite_automatic_factor_expr;
use cas_math::fraction_power_cancel_support::try_rewrite_cancel_same_base_powers_div_expr;
use cas_math::hyperbolic_identity_support::{
    try_rewrite_hyperbolic_double_angle_sum, try_rewrite_hyperbolic_triple_angle,
    try_rewrite_recognize_hyperbolic_from_exp, try_rewrite_tanh_double_angle_expansion,
    try_rewrite_tanh_to_sinh_cosh,
};
use cas_math::infinity_support::{is_negative_literal, is_positive_literal};
use cas_math::logarithm_inverse_support::{
    expand_logs_collect_positive_assumptions, log_exp_inverse_policy_mode_from_flags,
    plan_log_power_base_numeric_policy, try_rewrite_exponential_log_inverse_expr,
    try_rewrite_log_power_base_numeric_expr,
};
use cas_math::poly_lowering;
use cas_math::poly_store::clear_thread_local_store;
use cas_math::reciprocal_sqrt_canon_support::try_rewrite_reciprocal_sqrt_canon_expr;
use cas_math::root_forms::{
    extract_square_root_base, try_rewrite_canonical_root_expr,
    try_rewrite_extract_perfect_power_from_radicand_expr, try_rewrite_simplify_square_root_expr,
    SimplifySquareRootRewriteKind,
};
use cas_math::semantic_equality::SemanticEqualityChecker;
use cas_math::trig_canonicalization_support::{
    try_rewrite_cot_to_csc_pythagorean_identity_expr,
    try_rewrite_csc_cot_pythagorean_identity_expr, try_rewrite_sec_tan_pythagorean_identity_expr,
    try_rewrite_tan_to_sec_pythagorean_identity_expr, try_rewrite_tan_to_sin_cos_function_expr,
};
use cas_math::trig_contraction_support::try_rewrite_angle_sum_fraction_to_tan_expr;
use cas_math::trig_core_identity_support::{
    try_rewrite_legacy_evaluate_trig_expr, try_rewrite_pythagorean_identity_add_expr,
};
use cas_math::trig_eval_table_support::lookup_trig_or_inverse;
use cas_math::trig_half_angle_support::{
    extract_trig_half_angle, try_rewrite_hyperbolic_half_angle_squares_expr,
};
use cas_math::trig_identity_zero_support::try_rewrite_sin_sum_triple_identity_zero_expr;
use cas_math::trig_inverse_expansion_support::try_rewrite_trig_inverse_composition_expr;
use cas_math::trig_linear_support::{
    build_coef_times_base, extract_coef_and_base, extract_linear_coefficients,
};
use cas_math::trig_multi_angle_support::{
    try_rewrite_double_angle_function_expr, try_rewrite_quintuple_angle_expr,
    try_rewrite_triple_angle_expr,
};
use cas_math::trig_phase_shift_support::try_rewrite_trig_phase_shift_function_expr;
use cas_math::trig_power_identity_support::{
    extract_coeff_trig_pow2, extract_coeff_trig_pow4, try_rewrite_pythagorean_chain_add_expr,
    try_rewrite_pythagorean_factor_form_add_expr,
    try_rewrite_pythagorean_generic_coefficient_add_expr,
    try_rewrite_reciprocal_product_pythagorean_zero_add_expr,
    try_rewrite_trig_fourth_power_difference_add_expr,
};
use cas_math::trig_roots_flatten::extract_double_angle_arg_relaxed;
use cas_math::trig_roots_flatten::flatten_mul_chain;
use cas_math::trig_sum_product_support::{
    try_rewrite_product_to_sum_expr, try_rewrite_sum_to_product_contraction_expr,
};
use cas_math::trig_value_detection_support::detect_special_angle;
use cas_math::trig_values::lookup_trig_value;
use cas_math::trig_weierstrass_support::try_rewrite_weierstrass_contraction_div_expr;
use cas_solver_core::rationalize_policy::AutoRationalizeLevel;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};
use std::cmp::Ordering;
use std::collections::HashSet;

fn to_math_auto_expand_budget(
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

fn run_poly_lower_pass(
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

fn run_poly_gcd_modp_eager_pass(
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

fn is_terminal_after_core(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) => true,
        Expr::Div(num, den) => {
            matches!(ctx.get(*num), Expr::Number(_)) && matches!(ctx.get(*den), Expr::Number(_))
        }
        _ => false,
    }
}

fn is_symbolic_atom(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Variable(_) | Expr::Constant(_))
}

fn is_plain_symbolic_binomial_after_core(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Add(left, right) | Expr::Sub(left, right) => {
            is_symbolic_atom(ctx, *left) && is_symbolic_atom(ctx, *right)
        }
        Expr::Neg(inner) => is_plain_symbolic_binomial_after_core(ctx, *inner),
        _ => false,
    }
}

fn is_plain_symbolic_power_after_core(ctx: &Context, expr: ExprId) -> bool {
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return false;
    };
    is_symbolic_atom(ctx, *base)
        && matches!(
            ctx.get(*exp),
            Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_)
        )
}

fn is_symbolic_power_over_same_atom_noop_root(ctx: &Context, expr: ExprId) -> bool {
    let Expr::Div(left, right) = ctx.get(expr) else {
        return false;
    };
    let Expr::Pow(base, exp) = ctx.get(*left) else {
        return false;
    };

    is_symbolic_atom(ctx, *base)
        && matches!(ctx.get(*exp), Expr::Variable(_) | Expr::Constant(_))
        && is_symbolic_atom(ctx, *right)
        && expr_eq(ctx, *base, *right)
}

fn is_symbolic_pow_zero_root(ctx: &Context, expr: ExprId) -> bool {
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return false;
    };

    is_symbolic_atom(ctx, *base) && matches!(ctx.get(*exp), Expr::Number(n) if n.is_zero())
}

fn is_symbolic_atom_plus_nonzero_literal_root(ctx: &Context, expr: ExprId) -> bool {
    let Expr::Add(left, right) = ctx.get(expr) else {
        return false;
    };
    (is_symbolic_atom(ctx, *left) && matches!(ctx.get(*right), Expr::Number(n) if !n.is_zero()))
        || (matches!(ctx.get(*left), Expr::Number(n) if !n.is_zero())
            && is_symbolic_atom(ctx, *right))
}

fn is_same_denominator_difference_root(ctx: &mut Context, expr: ExprId) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 2 {
        return false;
    }

    let mut denominator = None;
    for (term_expr, _) in view.terms {
        let Expr::Div(_, den) = ctx.get(term_expr) else {
            return false;
        };

        if let Some(existing_den) = denominator {
            if compare_expr(ctx, *den, existing_den) != Ordering::Equal {
                return false;
            }
        } else {
            denominator = Some(*den);
        }
    }

    denominator.is_some()
}

fn extract_same_denominator_direct_pair_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId, ExprId)> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 2 {
        return None;
    }

    let first = view.terms[0];
    let second = view.terms[1];
    let (first_num, first_den) = match ctx.get(first.0) {
        Expr::Div(num, den) => (*num, *den),
        _ => return None,
    };
    let (second_num, second_den) = match ctx.get(second.0) {
        Expr::Div(num, den) => (*num, *den),
        _ => return None,
    };
    if compare_expr(ctx, first_den, second_den) != Ordering::Equal {
        return None;
    }

    match (first.1, second.1) {
        (Sign::Pos, Sign::Neg) => Some((first_den, first_num, second_num)),
        (Sign::Neg, Sign::Pos) => Some((first_den, second_num, first_num)),
        _ => None,
    }
}

fn is_nested_additive_pair_root(ctx: &Context, expr: ExprId) -> bool {
    let (lhs, rhs) = match ctx.get(expr) {
        Expr::Add(lhs, rhs) | Expr::Sub(lhs, rhs) => (*lhs, *rhs),
        _ => return false,
    };

    matches!(ctx.get(lhs), Expr::Add(_, _) | Expr::Sub(_, _))
        && matches!(ctx.get(rhs), Expr::Add(_, _) | Expr::Sub(_, _))
}

fn expr_contains_any_builtin_local(ctx: &Context, root: ExprId, builtins: &[BuiltinFn]) -> bool {
    let mut stack = vec![root];
    while let Some(expr) = stack.pop() {
        match ctx.get(expr) {
            Expr::Function(fn_id, args) => {
                if let Some(builtin) = ctx.builtin_of(*fn_id) {
                    if builtins.contains(&builtin) {
                        return true;
                    }
                }
                stack.extend(args.iter().copied());
            }
            Expr::Add(lhs, rhs)
            | Expr::Sub(lhs, rhs)
            | Expr::Mul(lhs, rhs)
            | Expr::Div(lhs, rhs)
            | Expr::Pow(lhs, rhs) => {
                stack.push(*lhs);
                stack.push(*rhs);
            }
            Expr::Neg(inner) => stack.push(*inner),
            _ => {}
        }
    }
    false
}

fn expr_contains_division_node_local(ctx: &Context, root: ExprId) -> bool {
    let mut stack = vec![root];
    while let Some(expr) = stack.pop() {
        match ctx.get(expr) {
            Expr::Div(_, _) => return true,
            Expr::Add(lhs, rhs)
            | Expr::Sub(lhs, rhs)
            | Expr::Mul(lhs, rhs)
            | Expr::Pow(lhs, rhs) => {
                stack.push(*lhs);
                stack.push(*rhs);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(*inner),
            Expr::Function(_, args) => stack.extend(args.iter().copied()),
            Expr::Matrix { data, .. } => stack.extend(data.iter().copied()),
            Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => {}
        }
    }
    false
}

fn expr_contains_sqrt_or_half_power_local(ctx: &Context, root: ExprId) -> bool {
    let mut stack = vec![root];
    let half = BigRational::new(1.into(), 2.into());

    while let Some(expr) = stack.pop() {
        match ctx.get(expr) {
            Expr::Function(fn_id, args)
                if ctx.is_builtin(*fn_id, BuiltinFn::Sqrt) && args.len() == 1 =>
            {
                return true;
            }
            Expr::Function(_, args) => stack.extend(args.iter().copied()),
            Expr::Pow(base, exp) => {
                if matches!(ctx.get(*exp), Expr::Number(n) if *n == half) {
                    return true;
                }
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

fn expr_contains_factorial_call_local(ctx: &Context, root: ExprId) -> bool {
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

fn expr_contains_guarded_small_zero_family_local(ctx: &Context, expr: ExprId) -> bool {
    expr_contains_division_node_local(ctx, expr)
        || expr_contains_sqrt_or_half_power_local(ctx, expr)
        || expr_contains_factorial_call_local(ctx, expr)
}

fn matches_guarded_small_zero_pair_root(ctx: &Context, lhs: ExprId, rhs: ExprId) -> bool {
    (expr_contains_trig_or_hyperbolic_builtin_local(ctx, lhs)
        && expr_contains_guarded_small_zero_family_local(ctx, rhs))
        || (expr_contains_trig_or_hyperbolic_builtin_local(ctx, rhs)
            && expr_contains_guarded_small_zero_family_local(ctx, lhs))
}

fn matches_direct_small_zero_pair_root(ctx: &mut Context, lhs: ExprId, rhs: ExprId) -> bool {
    matches_direct_small_zero_or_known_pair_base_root(ctx, lhs)
        && matches_direct_small_zero_or_known_pair_base_root(ctx, rhs)
}

fn is_direct_small_zero_composition_candidate_root(ctx: &mut Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Add(lhs, rhs) | Expr::Sub(lhs, rhs) | Expr::Mul(lhs, rhs) => {
            matches_direct_small_zero_pair_root(ctx, *lhs, *rhs)
        }
        _ => false,
    }
}

fn is_guarded_small_zero_composition_candidate_root(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Add(lhs, rhs) | Expr::Sub(lhs, rhs) | Expr::Mul(lhs, rhs) => {
            matches_guarded_small_zero_pair_root(ctx, *lhs, *rhs)
        }
        _ => false,
    }
}

fn is_guarded_small_zero_shifted_quotient_candidate_root(ctx: &mut Context, expr: ExprId) -> bool {
    let (numerator, denominator) = match ctx.get(expr).clone() {
        Expr::Div(numerator, denominator) => (numerator, denominator),
        _ => return false,
    };
    let Some(numerator_core) = strip_positive_one_passthrough_root(ctx, numerator) else {
        return false;
    };
    let Some(denominator_core) = strip_positive_one_passthrough_root(ctx, denominator) else {
        return false;
    };

    matches_guarded_small_zero_pair_root(ctx, numerator_core, denominator_core)
}

fn is_direct_small_zero_shifted_quotient_candidate_root(ctx: &mut Context, expr: ExprId) -> bool {
    let (numerator, denominator) = match ctx.get(expr).clone() {
        Expr::Div(numerator, denominator) => (numerator, denominator),
        _ => return false,
    };
    let Some(numerator_core) = strip_positive_one_passthrough_root(ctx, numerator) else {
        return false;
    };
    let Some(denominator_core) = strip_positive_one_passthrough_root(ctx, denominator) else {
        return false;
    };

    matches_direct_small_zero_pair_root(ctx, numerator_core, denominator_core)
}

fn is_nested_additive_log_residual_pair_root(ctx: &mut Context, lhs: ExprId, rhs: ExprId) -> bool {
    let residual_difference = ctx.add(Expr::Sub(lhs, rhs));
    is_nested_additive_pair_root(ctx, residual_difference)
        && expr_contains_any_builtin_local(
            ctx,
            residual_difference,
            &[BuiltinFn::Ln, BuiltinFn::Log, BuiltinFn::Abs],
        )
}

fn try_hidden_solve_root_exp_ln_shortcut(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let rewrite = try_rewrite_exponential_log_inverse_expr(ctx, expr)?;
    if is_symbolic_atom(ctx, rewrite.rewritten) {
        Some(rewrite.rewritten)
    } else {
        None
    }
}

fn square_of_symbolic_atom(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return None;
    };
    if !is_symbolic_atom(ctx, *base) {
        return None;
    }
    match ctx.get(*exp) {
        Expr::Number(n) if *n == BigRational::from_integer(2.into()) => Some(*base),
        _ => None,
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

fn expr_eq(ctx: &Context, left: ExprId, right: ExprId) -> bool {
    cas_ast::ordering::compare_expr(ctx, left, right) == std::cmp::Ordering::Equal
}

fn build_root_shortcut_parent_ctx(
    options: &crate::phase::SimplifyOptions,
    ctx: &Context,
    expr: ExprId,
) -> crate::parent_context::ParentContext {
    crate::parent_context::ParentContext::root()
        .with_domain_mode(options.shared.semantics.domain_mode)
        .with_value_domain(options.shared.semantics.value_domain)
        .with_inv_trig(options.shared.semantics.inv_trig)
        .with_goal(options.goal)
        .with_context_mode(options.shared.context_mode)
        .with_simplify_purpose(options.simplify_purpose)
        .with_autoexpand_binomials(options.shared.autoexpand_binomials)
        .with_heuristic_poly(options.shared.heuristic_poly)
        .with_expand_mode_flag(options.expand_mode)
        .with_root_expr(ctx, expr)
}

fn finish_standard_root_shortcut(
    _ctx: &Context,
    before: ExprId,
    rewrite: crate::rule::Rewrite,
    rule_name: &'static str,
    collect_steps: bool,
) -> (ExprId, Vec<Step>) {
    let result = rewrite.final_expr();
    let mut shortcut_steps = Vec::new();
    if collect_steps {
        let mut step = Step::new_compact(&rewrite.description, rule_name, before, rewrite.new_expr);
        step.global_before = Some(before);
        step.global_after = Some(rewrite.new_expr);
        step.importance = crate::step::ImportanceLevel::High;
        shortcut_steps.push(step);

        let mut current = rewrite.new_expr;
        for chained in rewrite.chained {
            let mut chain_step = Step::new_compact(
                chained.description.as_ref(),
                rule_name,
                current,
                chained.after,
            );
            chain_step.global_before = Some(current);
            chain_step.global_after = Some(chained.after);
            chain_step.importance = chained
                .importance
                .unwrap_or(crate::step::ImportanceLevel::High);
            shortcut_steps.push(chain_step);
            current = chained.after;
        }
    }
    (result, shortcut_steps)
}

fn build_root_shortcut_step_from_rewrite(
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

fn build_root_shortcut_compact_step(
    before: ExprId,
    after: ExprId,
    description: &'static str,
    rule_name: &'static str,
) -> Step {
    let mut step = Step::new_compact(description, rule_name, before, after);
    step.global_before = Some(before);
    step.global_after = Some(after);
    step.importance = crate::step::ImportanceLevel::High;
    step
}

fn finish_root_shortcut_with_rewrite_meta(
    ctx: &Context,
    before: ExprId,
    rewrite: crate::rule::Rewrite,
    rule_name: &'static str,
    collect_steps: bool,
) -> (ExprId, Vec<Step>) {
    let result = rewrite.final_expr();
    let mut shortcut_steps = Vec::new();
    if collect_steps {
        shortcut_steps.push(build_root_shortcut_step_from_rewrite(
            ctx, before, &rewrite, rule_name,
        ));

        let mut current = rewrite.new_expr;
        for chained in &rewrite.chained {
            let mut chain_step = Step::with_snapshots(
                chained.description.as_ref(),
                rule_name,
                current,
                chained.after,
                smallvec::SmallVec::<[crate::step::PathStep; 8]>::new(),
                Some(ctx),
                current,
                chained.after,
            );
            chain_step.importance = chained
                .importance
                .unwrap_or(crate::step::ImportanceLevel::High);
            {
                let meta = chain_step.meta_mut();
                meta.before_local = chained.before_local;
                meta.after_local = chained.after_local;
                meta.assumption_events = chained.assumption_events.clone();
                meta.required_conditions = chained.required_conditions.clone();
                meta.poly_proof = chained.poly_proof.clone();
                meta.is_chained = true;
            }
            shortcut_steps.push(chain_step);
            current = chained.after;
        }
    }
    (result, shortcut_steps)
}

fn build_signed_sum_expr_root(ctx: &mut Context, terms: &[(ExprId, Sign)]) -> ExprId {
    let Some((first_expr, first_sign)) = terms.first().copied() else {
        return ctx.num(0);
    };
    let mut acc = if first_sign == Sign::Neg {
        ctx.add(Expr::Neg(first_expr))
    } else {
        first_expr
    };
    for (expr, sign) in terms.iter().copied().skip(1) {
        let term = if sign == Sign::Neg {
            ctx.add(Expr::Neg(expr))
        } else {
            expr
        };
        acc = ctx.add(Expr::Add(acc, term));
    }
    acc
}

fn flip_add_sign_root(sign: Sign) -> Sign {
    match sign {
        Sign::Pos => Sign::Neg,
        Sign::Neg => Sign::Pos,
    }
}

fn normalize_signed_add_term_root(
    ctx: &mut Context,
    term_expr: ExprId,
    term_sign: Sign,
) -> (ExprId, Sign) {
    match ctx.get(term_expr).clone() {
        Expr::Neg(inner) => (inner, flip_add_sign_root(term_sign)),
        Expr::Number(n) if n < BigRational::zero() => {
            (ctx.add(Expr::Number(-n)), flip_add_sign_root(term_sign))
        }
        _ => {
            let (coeff, base) = extract_coef_and_base(ctx, term_expr);
            if coeff < BigRational::zero() {
                let normalized = if coeff == BigRational::from_integer((-1).into()) {
                    base
                } else {
                    let positive_coeff = ctx.add(Expr::Number(-coeff));
                    smart_mul(ctx, positive_coeff, base)
                };
                (normalized, flip_add_sign_root(term_sign))
            } else {
                (term_expr, term_sign)
            }
        }
    }
}

fn strip_positive_one_passthrough_root(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
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

fn extract_plain_sin_or_cos_arg_root(ctx: &Context, expr: ExprId) -> Option<(BuiltinFn, ExprId)> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }
    if ctx.is_builtin(*fn_id, BuiltinFn::Sin) {
        Some((BuiltinFn::Sin, args[0]))
    } else if ctx.is_builtin(*fn_id, BuiltinFn::Cos) {
        Some((BuiltinFn::Cos, args[0]))
    } else {
        None
    }
}

fn extract_plain_sinh_or_cosh_arg_root(ctx: &Context, expr: ExprId) -> Option<(BuiltinFn, ExprId)> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }
    if ctx.is_builtin(*fn_id, BuiltinFn::Sinh) {
        Some((BuiltinFn::Sinh, args[0]))
    } else if ctx.is_builtin(*fn_id, BuiltinFn::Cosh) {
        Some((BuiltinFn::Cosh, args[0]))
    } else {
        None
    }
}

fn extract_plain_sinh_or_cosh_pow2_arg_root(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BuiltinFn, ExprId)> {
    let Expr::Pow(base, exponent) = ctx.get(expr) else {
        return None;
    };
    if extract_i64_integer(ctx, *exponent)? != 2 {
        return None;
    }
    extract_plain_sinh_or_cosh_arg_root(ctx, *base)
}

fn build_tanh_pythagorean_target_root(ctx: &mut Context, arg: ExprId) -> ExprId {
    let one = ctx.num(1);
    let cosh_arg = ctx.call_builtin(BuiltinFn::Cosh, vec![arg]);
    let two = ctx.num(2);
    let denominator = ctx.add(Expr::Pow(cosh_arg, two));
    ctx.add(Expr::Div(one, denominator))
}

fn extract_direct_tanh_pythagorean_identity_arg_root(
    ctx: &Context,
    expr: ExprId,
) -> Option<ExprId> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 2 {
        return None;
    }

    let mut saw_positive_one = false;
    let mut tanh_sq_arg = None;

    for (term_expr, term_sign) in view.terms {
        if extract_i64_integer(ctx, term_expr) == Some(1) {
            if term_sign != Sign::Pos || saw_positive_one {
                return None;
            }
            saw_positive_one = true;
            continue;
        }

        if term_sign != Sign::Neg {
            return None;
        }
        let Expr::Pow(base, exponent) = ctx.get(term_expr) else {
            return None;
        };
        if extract_i64_integer(ctx, *exponent)? != 2 {
            return None;
        }
        let Expr::Function(fn_id, args) = ctx.get(*base) else {
            return None;
        };
        if !ctx.is_builtin(*fn_id, BuiltinFn::Tanh) || args.len() != 1 || tanh_sq_arg.is_some() {
            return None;
        }
        tanh_sq_arg = Some(args[0]);
    }

    saw_positive_one.then_some(tanh_sq_arg?).or(None)
}

fn extract_direct_tanh_pythagorean_target_root(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Div(numerator, denominator) = ctx.get(expr) else {
        return None;
    };
    if extract_i64_integer(ctx, *numerator)? != 1 {
        return None;
    }
    let Some((BuiltinFn::Cosh, arg)) = extract_plain_sinh_or_cosh_pow2_arg_root(ctx, *denominator)
    else {
        return None;
    };
    Some(arg)
}

fn extract_trig_binomial_square_identity_data_root(
    ctx: &Context,
    expr: ExprId,
) -> Option<(ExprId, bool)> {
    let Expr::Pow(base, exponent) = ctx.get(expr) else {
        return None;
    };
    if extract_i64_integer(ctx, *exponent)? != 2 {
        return None;
    }

    let (left, right, is_sum) = match ctx.get(*base) {
        Expr::Add(left, right) => (*left, *right, true),
        Expr::Sub(left, right) => (*left, *right, false),
        _ => return None,
    };

    let (lhs_fn, lhs_arg) = extract_plain_sin_or_cos_arg_root(ctx, left)?;
    let (rhs_fn, rhs_arg) = extract_plain_sin_or_cos_arg_root(ctx, right)?;
    if compare_expr(ctx, lhs_arg, rhs_arg) != Ordering::Equal {
        return None;
    }
    let trig_kinds = [lhs_fn, rhs_fn];
    if !trig_kinds.contains(&BuiltinFn::Sin) || !trig_kinds.contains(&BuiltinFn::Cos) {
        return None;
    }

    Some((lhs_arg, is_sum))
}

fn build_trig_square_double_angle_term_root(ctx: &mut Context, arg: ExprId) -> ExprId {
    let two = ctx.num(2);
    let doubled_arg = smart_mul(ctx, two, arg);
    ctx.call_builtin(BuiltinFn::Sin, vec![doubled_arg])
}

fn matches_trig_square_double_angle_term_root(
    ctx: &mut Context,
    expr: ExprId,
    arg: ExprId,
) -> bool {
    let target = build_trig_square_double_angle_term_root(ctx, arg);
    compare_expr(ctx, expr, target) == Ordering::Equal
}

fn build_half_angle_square_target_root(
    ctx: &mut Context,
    trig_fn: BuiltinFn,
    arg: ExprId,
) -> ExprId {
    let one = ctx.num(1);
    let two = ctx.num(2);
    let double_arg = smart_mul(ctx, two, arg);
    let cos_double_arg = ctx.call_builtin(BuiltinFn::Cos, vec![double_arg]);
    let numerator = match trig_fn {
        BuiltinFn::Sin => ctx.add(Expr::Sub(one, cos_double_arg)),
        BuiltinFn::Cos => ctx.add(Expr::Add(one, cos_double_arg)),
        _ => return ctx.num(0),
    };
    ctx.add(Expr::Div(numerator, two))
}

fn extract_direct_half_angle_square_target_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(BuiltinFn, ExprId)> {
    let Expr::Div(numerator, denominator) = ctx.get(expr) else {
        return None;
    };
    if extract_i64_integer(ctx, *denominator)? != 2 {
        return None;
    }

    let view = AddView::from_expr(ctx, *numerator);
    if view.terms.len() != 2 {
        return None;
    }

    let mut saw_positive_one = false;
    let mut cos_term = None;

    for (term_expr, term_sign) in view.terms {
        if extract_i64_integer(ctx, term_expr) == Some(1) {
            if term_sign != Sign::Pos || saw_positive_one {
                return None;
            }
            saw_positive_one = true;
            continue;
        }

        let arg = extract_positive_cos_double_angle_arg_root(ctx, term_expr)?;
        if cos_term.is_some() {
            return None;
        }
        cos_term = Some((arg, term_sign));
    }

    let (arg, cos_sign) = cos_term?;
    if !saw_positive_one {
        return None;
    }

    let trig_fn = match cos_sign {
        Sign::Pos => BuiltinFn::Cos,
        Sign::Neg => BuiltinFn::Sin,
    };
    Some((trig_fn, arg))
}

fn matches_direct_half_angle_square_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (half_angle_expr, trig_square_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some((trig_fn, half_angle_arg)) =
            extract_direct_half_angle_square_target_root(ctx, half_angle_expr)
        else {
            continue;
        };

        let Some((coeff, trig_name, trig_arg, effective_sign)) =
            extract_signed_numeric_trig_pow2(ctx, trig_square_expr, Sign::Pos)
        else {
            continue;
        };
        if effective_sign != Sign::Pos || coeff != BigRational::one() {
            continue;
        }

        let expected_trig_name = match trig_fn {
            BuiltinFn::Sin => "sin",
            BuiltinFn::Cos => "cos",
            _ => continue,
        };
        if trig_name == expected_trig_name
            && compare_expr(ctx, half_angle_arg, trig_arg) == Ordering::Equal
        {
            return true;
        }
    }

    false
}

fn extract_direct_scaled_half_angle_square_target_root(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BuiltinFn, ExprId)> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 2 {
        return None;
    }

    let mut saw_positive_one = false;
    let mut cos_arg = None;

    for (term_expr, term_sign) in view.terms {
        if extract_i64_integer(ctx, term_expr) == Some(1) {
            if term_sign != Sign::Pos || saw_positive_one {
                return None;
            }
            saw_positive_one = true;
            continue;
        }

        let Some((BuiltinFn::Cos, arg)) = extract_plain_sin_or_cos_arg_root(ctx, term_expr) else {
            return None;
        };
        if cos_arg.is_some() {
            return None;
        }
        cos_arg = Some((arg, term_sign));
    }

    let (arg, cos_sign) = cos_arg?;
    if !saw_positive_one {
        return None;
    }

    let trig_fn = match cos_sign {
        Sign::Pos => BuiltinFn::Cos,
        Sign::Neg => BuiltinFn::Sin,
    };
    Some((trig_fn, arg))
}

fn matches_direct_scaled_half_angle_square_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (scaled_target_expr, trig_square_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some((trig_fn, full_arg)) =
            extract_direct_scaled_half_angle_square_target_root(ctx, scaled_target_expr)
        else {
            continue;
        };

        let Some((coeff, trig_name, trig_arg, effective_sign)) =
            extract_signed_numeric_trig_pow2(ctx, trig_square_expr, Sign::Pos)
        else {
            continue;
        };
        if effective_sign != Sign::Pos || coeff != BigRational::from_integer(2.into()) {
            continue;
        }

        let expected_trig_name = match trig_fn {
            BuiltinFn::Sin => "sin",
            BuiltinFn::Cos => "cos",
            _ => continue,
        };
        if trig_name != expected_trig_name {
            continue;
        }

        let Some(base_arg) = extract_half_scaled_base_root(ctx, trig_arg) else {
            continue;
        };
        if compare_expr(ctx, base_arg, full_arg) == Ordering::Equal {
            return true;
        }
    }

    false
}

fn extract_direct_abs_trig_half_angle_target_root(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BuiltinFn, ExprId)> {
    let abs_inner = try_unwrap_abs_arg(ctx, expr)?;
    let (full_angle, is_sin) = extract_trig_half_angle(ctx, abs_inner)?;
    let trig_fn = if is_sin {
        BuiltinFn::Sin
    } else {
        BuiltinFn::Cos
    };
    Some((trig_fn, full_angle))
}

fn extract_direct_sqrt_abs_trig_half_angle_target_root(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BuiltinFn, ExprId)> {
    let radicand = extract_unary_builtin_arg_root(ctx, expr, BuiltinFn::Sqrt)?;
    let Expr::Div(numerator, denominator) = ctx.get(radicand) else {
        return None;
    };
    if extract_i64_integer(ctx, *denominator)? != 2 {
        return None;
    }

    let view = AddView::from_expr(ctx, *numerator);
    if view.terms.len() != 2 {
        return None;
    }

    let mut saw_positive_one = false;
    let mut cos_arg = None;
    let mut cos_sign = None;

    for (term_expr, term_sign) in view.terms {
        if extract_i64_integer(ctx, term_expr) == Some(1) {
            if term_sign != Sign::Pos || saw_positive_one {
                return None;
            }
            saw_positive_one = true;
            continue;
        }

        let Some((BuiltinFn::Cos, arg)) = extract_plain_sin_or_cos_arg_root(ctx, term_expr) else {
            return None;
        };
        if cos_arg.is_some() {
            return None;
        }
        cos_arg = Some(arg);
        cos_sign = Some(term_sign);
    }

    let trig_fn = match cos_sign? {
        Sign::Neg => BuiltinFn::Sin,
        Sign::Pos => BuiltinFn::Cos,
    };

    saw_positive_one.then_some((trig_fn, cos_arg?))
}

fn build_direct_sqrt_abs_trig_half_angle_target_root(
    ctx: &mut Context,
    trig_fn: BuiltinFn,
    full_arg: ExprId,
) -> ExprId {
    let one = ctx.num(1);
    let two = ctx.num(2);
    let cos_expr = ctx.call_builtin(BuiltinFn::Cos, vec![full_arg]);
    let numerator = match trig_fn {
        BuiltinFn::Sin => ctx.add(Expr::Sub(one, cos_expr)),
        BuiltinFn::Cos => ctx.add(Expr::Add(one, cos_expr)),
        _ => unreachable!("only sin/cos half-angle absolutes are supported"),
    };
    let radicand = ctx.add(Expr::Div(numerator, two));
    ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand])
}

fn matches_direct_abs_trig_half_angle_pair_root(
    ctx: &Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (abs_expr, sqrt_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some((abs_fn, abs_arg)) = extract_direct_abs_trig_half_angle_target_root(ctx, abs_expr)
        else {
            continue;
        };
        let Some((sqrt_fn, sqrt_arg)) =
            extract_direct_sqrt_abs_trig_half_angle_target_root(ctx, sqrt_expr)
        else {
            continue;
        };
        if abs_fn == sqrt_fn && compare_expr(ctx, abs_arg, sqrt_arg) == Ordering::Equal {
            return true;
        }
    }

    false
}

fn build_plain_trig_pow2_root(ctx: &mut Context, trig_fn: BuiltinFn, arg: ExprId) -> ExprId {
    let trig_expr = ctx.call_builtin(trig_fn, vec![arg]);
    let two = ctx.num(2);
    ctx.add(Expr::Pow(trig_expr, two))
}

fn extract_positive_cos_quadruple_angle_arg_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let Some((BuiltinFn::Cos, arg)) = extract_plain_sin_or_cos_arg_root(ctx, expr) else {
        return None;
    };
    let double_angle_arg = extract_double_angle_arg_relaxed(ctx, arg)?;
    extract_double_angle_arg_relaxed(ctx, double_angle_arg)
}

fn extract_direct_cos_fourth_power_reduction_target_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let Expr::Div(numerator, denominator) = ctx.get(expr) else {
        return None;
    };
    if extract_i64_integer(ctx, *denominator)? != 8 {
        return None;
    }

    let view = AddView::from_expr(ctx, *numerator);
    if view.terms.len() != 3 {
        return None;
    }

    let mut saw_positive_three = false;
    let mut double_angle_arg = None;
    let mut quadruple_angle_arg = None;

    for (term_expr, term_sign) in view.terms {
        if extract_i64_integer(ctx, term_expr) == Some(3) {
            if term_sign != Sign::Pos || saw_positive_three {
                return None;
            }
            saw_positive_three = true;
            continue;
        }

        let (coeff, base) = extract_coef_and_base(ctx, term_expr);
        let signed_coeff = if term_sign == Sign::Neg {
            -coeff
        } else {
            coeff
        };
        if signed_coeff == BigRational::from_integer(4.into()) {
            if double_angle_arg.is_some() {
                return None;
            }
            double_angle_arg = extract_positive_cos_double_angle_arg_root(ctx, base);
            double_angle_arg?;
            continue;
        }

        if signed_coeff.is_one() {
            if quadruple_angle_arg.is_some() {
                return None;
            }
            quadruple_angle_arg = extract_positive_cos_quadruple_angle_arg_root(ctx, base);
            quadruple_angle_arg?;
            continue;
        }

        return None;
    }

    let double_angle_arg = double_angle_arg?;
    let quadruple_angle_arg = quadruple_angle_arg?;
    if !saw_positive_three
        || compare_expr(ctx, double_angle_arg, quadruple_angle_arg) != Ordering::Equal
    {
        return None;
    }

    Some(double_angle_arg)
}

fn extract_direct_sin_cos_square_product_reduction_target_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let Expr::Div(numerator, denominator) = ctx.get(expr) else {
        return None;
    };
    if extract_i64_integer(ctx, *denominator)? != 8 {
        return None;
    }

    let view = AddView::from_expr(ctx, *numerator);
    if view.terms.len() != 2 {
        return None;
    }

    let mut saw_positive_one = false;
    let mut quadruple_angle_arg = None;

    for (term_expr, term_sign) in view.terms {
        if extract_i64_integer(ctx, term_expr) == Some(1) {
            if term_sign != Sign::Pos || saw_positive_one {
                return None;
            }
            saw_positive_one = true;
            continue;
        }

        let (coeff, base) = extract_coef_and_base(ctx, term_expr);
        let signed_coeff = if term_sign == Sign::Neg {
            -coeff
        } else {
            coeff
        };
        if signed_coeff != BigRational::from_integer((-1).into()) {
            return None;
        }

        if quadruple_angle_arg.is_some() {
            return None;
        }
        quadruple_angle_arg = extract_positive_cos_quadruple_angle_arg_root(ctx, base);
        quadruple_angle_arg?;
    }

    if !saw_positive_one {
        return None;
    }
    quadruple_angle_arg
}

fn build_plain_trig_pow4_root(ctx: &mut Context, trig_fn: BuiltinFn, arg: ExprId) -> ExprId {
    let trig_expr = ctx.call_builtin(trig_fn, vec![arg]);
    let four = ctx.num(4);
    ctx.add(Expr::Pow(trig_expr, four))
}

fn extract_scaled_sin_fourth_power_target_root(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let (coeff, trig_name, arg) = extract_coeff_trig_pow4(ctx, expr)?;
    (trig_name == "sin" && coeff == BigRational::from_integer(8.into())).then_some(arg)
}

fn extract_scaled_sin_fourth_power_reduction_target_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 3 {
        return None;
    }

    let mut saw_positive_three = false;
    let mut double_angle_arg = None;
    let mut quadruple_angle_arg = None;

    for (term_expr, term_sign) in view.terms {
        if extract_i64_integer(ctx, term_expr) == Some(3) {
            if term_sign != Sign::Pos || saw_positive_three {
                return None;
            }
            saw_positive_three = true;
            continue;
        }

        let (coeff, base) = extract_coef_and_base(ctx, term_expr);
        let signed_coeff = if term_sign == Sign::Neg {
            -coeff
        } else {
            coeff
        };

        if signed_coeff == BigRational::from_integer((-4).into()) {
            if double_angle_arg.is_some() {
                return None;
            }
            double_angle_arg = extract_positive_cos_double_angle_arg_root(ctx, base);
            double_angle_arg?;
            continue;
        }

        if signed_coeff.is_one() {
            if quadruple_angle_arg.is_some() {
                return None;
            }
            quadruple_angle_arg = extract_positive_cos_quadruple_angle_arg_root(ctx, base);
            quadruple_angle_arg?;
            continue;
        }

        return None;
    }

    let double_angle_arg = double_angle_arg?;
    let quadruple_angle_arg = quadruple_angle_arg?;
    if !saw_positive_three
        || compare_expr(ctx, double_angle_arg, quadruple_angle_arg) != Ordering::Equal
    {
        return None;
    }

    Some(double_angle_arg)
}

fn build_plain_sin_cos_square_product_root(ctx: &mut Context, arg: ExprId) -> ExprId {
    let two = ctx.num(2);
    let sin_expr = ctx.call_builtin(BuiltinFn::Sin, vec![arg]);
    let cos_expr = ctx.call_builtin(BuiltinFn::Cos, vec![arg]);
    let sin_sq = ctx.add(Expr::Pow(sin_expr, two));
    let cos_sq = ctx.add(Expr::Pow(cos_expr, two));
    build_mul_expr_from_factors_root(ctx, &[sin_sq, cos_sq])
}

fn build_scaled_double_angle_sin_square_root(ctx: &mut Context, arg: ExprId) -> ExprId {
    let two = ctx.num(2);
    let four = ctx.num(4);
    let doubled_arg = smart_mul(ctx, two, arg);
    let sin_double = ctx.call_builtin(BuiltinFn::Sin, vec![doubled_arg]);
    let sin_sq = ctx.add(Expr::Pow(sin_double, two));
    ctx.add(Expr::Div(sin_sq, four))
}

fn matches_direct_half_angle_square_zero_identity_root(ctx: &mut Context, expr: ExprId) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 2 {
        return false;
    }

    for index in 0..view.terms.len() {
        let (term_expr, term_sign) = view.terms[index];
        let Some((coeff, trig_name, arg, effective_sign)) =
            extract_signed_numeric_trig_pow2(ctx, term_expr, term_sign)
        else {
            continue;
        };
        if !coeff.is_one() {
            continue;
        }
        let trig_fn = match trig_name {
            "sin" => BuiltinFn::Sin,
            "cos" => BuiltinFn::Cos,
            _ => continue,
        };
        let other_terms: Vec<_> = view
            .terms
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(other_index, term)| (other_index != index).then_some(term))
            .collect();
        if (other_terms.len() != 1 || other_terms[0].1 != Sign::Neg)
            && (other_terms.len() != 1 || other_terms[0].1 != Sign::Pos)
        {
            continue;
        }
        let target = build_half_angle_square_target_root(ctx, trig_fn, arg);
        let matches_target = compare_expr(ctx, other_terms[0].0, target) == Ordering::Equal;
        if matches_target
            && ((effective_sign == Sign::Pos && other_terms[0].1 == Sign::Neg)
                || (effective_sign == Sign::Neg && other_terms[0].1 == Sign::Pos))
        {
            return true;
        }
    }

    false
}

fn matches_direct_trig_binomial_square_zero_identity_root(ctx: &mut Context, expr: ExprId) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() < 2 {
        return false;
    }

    for index in 0..view.terms.len() {
        let (term_expr, term_sign) = view.terms[index];
        if term_sign != Sign::Pos {
            continue;
        }
        let Some((arg, is_sum)) = extract_trig_binomial_square_identity_data_root(ctx, term_expr)
        else {
            continue;
        };
        let mut saw_negative_one = false;
        let mut saw_double_angle = false;
        let mut bad_term = false;
        for (other_index, (other_expr, other_sign)) in view.terms.iter().copied().enumerate() {
            if other_index == index {
                continue;
            }
            let is_negative_one = extract_i64_integer(ctx, other_expr).is_some_and(|value| {
                matches!((value, other_sign), (1, Sign::Neg) | (-1, Sign::Pos))
            });
            let matches_double_angle =
                matches_trig_square_double_angle_term_root(ctx, other_expr, arg)
                    && other_sign == if is_sum { Sign::Neg } else { Sign::Pos };
            if is_negative_one {
                saw_negative_one = true;
            } else if matches_double_angle {
                saw_double_angle = true;
            } else {
                bad_term = true;
                break;
            }
        }
        if !bad_term && saw_negative_one && saw_double_angle {
            return true;
        }
    }

    false
}

fn matches_direct_half_angle_binomial_square_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    (matches_direct_half_angle_square_zero_identity_root(ctx, lhs_core)
        && matches_direct_trig_binomial_square_zero_identity_root(ctx, rhs_core))
        || (matches_direct_half_angle_square_zero_identity_root(ctx, rhs_core)
            && matches_direct_trig_binomial_square_zero_identity_root(ctx, lhs_core))
}

fn matches_direct_trig_binomial_square_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    let lhs_minus_rhs = ctx.add(Expr::Sub(lhs_core, rhs_core));
    if matches_direct_trig_binomial_square_zero_identity_root(ctx, lhs_minus_rhs) {
        return true;
    }

    let rhs_minus_lhs = ctx.add(Expr::Sub(rhs_core, lhs_core));
    matches_direct_trig_binomial_square_zero_identity_root(ctx, rhs_minus_lhs)
}

fn matches_direct_trig_product_to_sum_sin_sin_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    let lhs_rewrite = try_rewrite_product_to_sum_expr(ctx, lhs_core);
    if lhs_rewrite.is_some_and(|rewrite| {
        rewrite.kind == cas_math::trig_sum_product_support::TrigProductToSumRewriteKind::SinSin
            && render_expr(ctx, rewrite.rewritten) == render_expr(ctx, rhs_core)
    }) {
        return true;
    }

    let rhs_rewrite = try_rewrite_product_to_sum_expr(ctx, rhs_core);
    rhs_rewrite.is_some_and(|rewrite| {
        rewrite.kind == cas_math::trig_sum_product_support::TrigProductToSumRewriteKind::SinSin
            && render_expr(ctx, rewrite.rewritten) == render_expr(ctx, lhs_core)
    })
}

fn extract_scaled_trig_sin_sin_product_args_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId)> {
    let factors = flatten_mul_chain(ctx, expr);
    let mut numeric_coeff = BigRational::one();
    let mut first_sin_arg = None;
    let mut second_sin_arg = None;

    for factor in factors {
        if let Expr::Number(n) = ctx.get(factor) {
            numeric_coeff *= n.clone();
            continue;
        }

        let Some((BuiltinFn::Sin, arg)) = extract_plain_sin_or_cos_arg_root(ctx, factor) else {
            return None;
        };
        if first_sin_arg.is_none() {
            first_sin_arg = Some(arg);
        } else if second_sin_arg.is_none() {
            second_sin_arg = Some(arg);
        } else {
            return None;
        }
    }

    if numeric_coeff != BigRational::from_integer(2.into()) {
        return None;
    }

    Some((first_sin_arg?, second_sin_arg?))
}

fn matches_direct_trig_product_to_sum_sin_sin_zero_identity_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 3 {
        return false;
    }

    let mut product_args = None;
    let mut cos_sum_arg = None;
    let mut cos_diff_arg = None;

    for (term_expr, term_sign) in view.terms {
        if term_sign == Sign::Pos {
            if let Some(args) = extract_scaled_trig_sin_sin_product_args_root(ctx, term_expr) {
                if product_args.is_some() {
                    return false;
                }
                product_args = Some(args);
                continue;
            }

            let Some((BuiltinFn::Cos, arg)) = extract_plain_sin_or_cos_arg_root(ctx, term_expr)
            else {
                return false;
            };
            if cos_sum_arg.is_some() {
                return false;
            }
            cos_sum_arg = Some(arg);
            continue;
        }

        let Some((BuiltinFn::Cos, arg)) = extract_plain_sin_or_cos_arg_root(ctx, term_expr) else {
            return false;
        };
        if cos_diff_arg.is_some() {
            return false;
        }
        cos_diff_arg = Some(arg);
    }

    let Some((lhs_arg, rhs_arg)) = product_args else {
        return false;
    };
    let Some(cos_sum_arg) = cos_sum_arg else {
        return false;
    };
    let Some(cos_diff_arg) = cos_diff_arg else {
        return false;
    };

    matches_angle_sum_or_diff_arg_root(ctx, cos_sum_arg, lhs_arg, rhs_arg, true)
        && matches_angle_sum_or_diff_arg_root(ctx, cos_diff_arg, lhs_arg, rhs_arg, false)
}

fn matches_direct_trig_product_to_sum_cos_cos_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    let lhs_rewrite = try_rewrite_product_to_sum_expr(ctx, lhs_core);
    if lhs_rewrite.is_some_and(|rewrite| {
        rewrite.kind == cas_math::trig_sum_product_support::TrigProductToSumRewriteKind::CosCos
            && render_expr(ctx, rewrite.rewritten) == render_expr(ctx, rhs_core)
    }) {
        return true;
    }

    let rhs_rewrite = try_rewrite_product_to_sum_expr(ctx, rhs_core);
    rhs_rewrite.is_some_and(|rewrite| {
        rewrite.kind == cas_math::trig_sum_product_support::TrigProductToSumRewriteKind::CosCos
            && render_expr(ctx, rewrite.rewritten) == render_expr(ctx, lhs_core)
    })
}

fn matches_direct_trig_product_to_sum_sin_cos_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    fn matches_sum_target(ctx: &mut Context, product_expr: ExprId, sum_expr: ExprId) -> bool {
        let Some((sin_arg, cos_arg)) =
            extract_scaled_trig_sin_cos_product_args_root(ctx, product_expr)
        else {
            return false;
        };
        let view = AddView::from_expr(ctx, sum_expr);
        if view.terms.len() != 2 {
            return false;
        }

        let mut saw_sum = false;
        let mut saw_diff = false;
        for (term_expr, term_sign) in view.terms {
            let Some((BuiltinFn::Sin, arg)) = extract_plain_sin_or_cos_arg_root(ctx, term_expr)
            else {
                return false;
            };
            match term_sign {
                Sign::Pos
                    if matches_angle_sum_or_diff_arg_root(ctx, arg, sin_arg, cos_arg, true) =>
                {
                    if saw_sum {
                        return false;
                    }
                    saw_sum = true;
                }
                Sign::Pos
                    if matches_angle_sum_or_diff_arg_root(ctx, arg, sin_arg, cos_arg, false) =>
                {
                    if saw_diff {
                        return false;
                    }
                    saw_diff = true;
                }
                Sign::Neg
                    if matches_angle_sum_or_diff_arg_root(ctx, arg, cos_arg, sin_arg, false) =>
                {
                    if saw_diff {
                        return false;
                    }
                    saw_diff = true;
                }
                _ => return false,
            }
        }

        saw_sum && saw_diff
    }

    matches_sum_target(ctx, lhs_core, rhs_core) || matches_sum_target(ctx, rhs_core, lhs_core)
}

fn extract_div_by_two_numerator_root(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Div(num, den) = ctx.get(expr) else {
        return None;
    };
    matches!(ctx.get(*den), Expr::Number(n) if n.is_integer() && n.to_integer() == 2.into())
        .then_some(*num)
}

fn matches_direct_normalized_trig_product_to_sum_sin_cos_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (product_expr, averaged_sum_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(sum_numerator) = extract_div_by_two_numerator_root(ctx, averaged_sum_expr) else {
            continue;
        };
        let two = ctx.num(2);
        let doubled_product = smart_mul(ctx, two, product_expr);
        if matches_direct_trig_product_to_sum_sin_cos_pair_root(ctx, doubled_product, sum_numerator)
        {
            return true;
        }
    }

    false
}

fn extract_direct_trig_power_mixed_square_target_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() != 2 {
        return None;
    }

    let mut sin_arg = None;
    let mut cos_arg = None;
    for factor in factors {
        let (coeff, trig_name, arg, effective_sign) =
            extract_signed_numeric_trig_pow2(ctx, factor, Sign::Pos)?;
        if effective_sign != Sign::Pos || coeff != BigRational::one() {
            return None;
        }
        match trig_name {
            "sin" if sin_arg.is_none() => sin_arg = Some(arg),
            "cos" if cos_arg.is_none() => cos_arg = Some(arg),
            _ => return None,
        }
    }

    let sin_arg = sin_arg?;
    let cos_arg = cos_arg?;
    (compare_expr(ctx, sin_arg, cos_arg) == Ordering::Equal).then_some(sin_arg)
}

fn extract_scaled_double_angle_sin_square_target_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    if let Expr::Div(num, den) = ctx.get(expr) {
        if matches!(ctx.get(*den), Expr::Number(n) if *n == BigRational::new(4.into(), 1.into())) {
            let (coeff, trig_name, arg, effective_sign) =
                extract_signed_numeric_trig_pow2(ctx, *num, Sign::Pos)?;
            if effective_sign == Sign::Pos && coeff == BigRational::one() && trig_name == "sin" {
                return extract_double_angle_arg_relaxed(ctx, arg);
            }
        }
    }

    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() != 2 {
        return None;
    }

    let mut saw_quarter = false;
    let mut sin_sq_arg = None;
    for factor in factors {
        match ctx.get(factor) {
            Expr::Number(n) if *n == BigRational::new(1.into(), 4.into()) => {
                if saw_quarter {
                    return None;
                }
                saw_quarter = true;
            }
            _ => {
                let (coeff, trig_name, arg, effective_sign) =
                    extract_signed_numeric_trig_pow2(ctx, factor, Sign::Pos)?;
                if effective_sign != Sign::Pos
                    || coeff != BigRational::one()
                    || trig_name != "sin"
                    || sin_sq_arg.is_some()
                {
                    return None;
                }
                let double_angle_arg = extract_double_angle_arg_relaxed(ctx, arg)?;
                sin_sq_arg = Some(double_angle_arg);
            }
        }
    }

    saw_quarter.then_some(sin_sq_arg?).or(None)
}

fn matches_direct_trig_power_mixed_square_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (mixed_square, doubled_square) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(base_arg) = extract_direct_trig_power_mixed_square_target_root(ctx, mixed_square)
        else {
            continue;
        };
        let Some(double_arg) =
            extract_scaled_double_angle_sin_square_target_root(ctx, doubled_square)
        else {
            continue;
        };
        if compare_expr(ctx, base_arg, double_arg) == Ordering::Equal {
            return true;
        }
    }

    false
}

fn matches_direct_cos_fourth_power_reduction_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (reduced_expr, pow4_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(arg) = extract_direct_cos_fourth_power_reduction_target_root(ctx, reduced_expr)
        else {
            continue;
        };
        let expected = build_plain_trig_pow4_root(ctx, BuiltinFn::Cos, arg);
        if compare_expr(ctx, pow4_expr, expected) == Ordering::Equal {
            return true;
        }
    }

    false
}

fn extract_half_scaled_base_root(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
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

fn extract_direct_hyperbolic_half_angle_square_target_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(BuiltinFn, ExprId)> {
    let base = extract_half_scaled_base_root(ctx, expr)?;
    let view = AddView::from_expr(ctx, base);
    if view.terms.len() != 2 {
        return None;
    }

    let mut cosh_arg = None;
    let mut one_sign = None;

    for (term_expr, term_sign) in view.terms {
        if extract_i64_integer(ctx, term_expr) == Some(1) {
            if one_sign.is_some() {
                return None;
            }
            one_sign = Some(term_sign);
            continue;
        }

        let Some((BuiltinFn::Cosh, arg)) = extract_plain_sinh_or_cosh_arg_root(ctx, term_expr)
        else {
            return None;
        };
        if term_sign != Sign::Pos || cosh_arg.is_some() {
            return None;
        }
        cosh_arg = Some(arg);
    }

    let hyperbolic_fn = match one_sign? {
        Sign::Pos => BuiltinFn::Cosh,
        Sign::Neg => BuiltinFn::Sinh,
    };
    Some((hyperbolic_fn, cosh_arg?))
}

fn matches_expr_or_negation_root(ctx: &Context, lhs: ExprId, rhs: ExprId) -> bool {
    if compare_expr(ctx, lhs, rhs) == Ordering::Equal {
        return true;
    }

    let (lhs_coeff, lhs_base) = extract_coef_and_base(ctx, lhs);
    let (rhs_coeff, rhs_base) = extract_coef_and_base(ctx, rhs);
    compare_expr(ctx, lhs_base, rhs_base) == Ordering::Equal && lhs_coeff == -rhs_coeff
}

fn matches_unordered_cos_arg_pair_up_to_sign_root(
    ctx: &Context,
    lhs_a: ExprId,
    lhs_b: ExprId,
    rhs_a: ExprId,
    rhs_b: ExprId,
) -> bool {
    (matches_expr_or_negation_root(ctx, lhs_a, rhs_a)
        && matches_expr_or_negation_root(ctx, lhs_b, rhs_b))
        || (matches_expr_or_negation_root(ctx, lhs_a, rhs_b)
            && matches_expr_or_negation_root(ctx, lhs_b, rhs_a))
}

fn canonicalize_even_cos_arg_root(ctx: &mut Context, arg: ExprId) -> ExprId {
    let (coeff, base) = extract_coef_and_base(ctx, arg);
    if coeff < BigRational::zero() {
        build_coef_times_base(ctx, &(-coeff), base)
    } else {
        arg
    }
}

fn matches_direct_hyperbolic_half_angle_square_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (source, target) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(rewrite) = try_rewrite_hyperbolic_half_angle_squares_expr(ctx, source) else {
            continue;
        };
        let Some((rewritten_fn, rewritten_arg)) =
            extract_direct_hyperbolic_half_angle_square_target_root(ctx, rewrite.rewritten)
        else {
            continue;
        };
        let Some((target_fn, target_arg)) =
            extract_direct_hyperbolic_half_angle_square_target_root(ctx, target)
        else {
            continue;
        };
        if rewritten_fn == target_fn
            && compare_expr(ctx, rewritten_arg, target_arg) == Ordering::Equal
        {
            return true;
        }
    }

    false
}

fn build_plain_hyperbolic_half_angle_pow2_root(
    ctx: &mut Context,
    hyperbolic_fn: BuiltinFn,
    arg: ExprId,
) -> ExprId {
    let two = ctx.num(2);
    let half_arg = ctx.add(Expr::Div(arg, two));
    let hyperbolic_expr = ctx.call_builtin(hyperbolic_fn, vec![half_arg]);
    ctx.add(Expr::Pow(hyperbolic_expr, two))
}

fn build_direct_hyperbolic_half_angle_square_target_root(
    ctx: &mut Context,
    hyperbolic_fn: BuiltinFn,
    arg: ExprId,
) -> ExprId {
    let one = ctx.num(1);
    let two = ctx.num(2);
    let cosh_arg = ctx.call_builtin(BuiltinFn::Cosh, vec![arg]);
    let numerator = match hyperbolic_fn {
        BuiltinFn::Sinh => ctx.add(Expr::Sub(cosh_arg, one)),
        BuiltinFn::Cosh => ctx.add(Expr::Add(cosh_arg, one)),
        _ => unreachable!("only sinh/cosh half-angle squares are supported"),
    };
    ctx.add(Expr::Div(numerator, two))
}

fn matches_direct_sum_to_product_contraction_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (sum_expr, product_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(rewrite) = try_rewrite_sum_to_product_contraction_expr(ctx, sum_expr) else {
            continue;
        };
        match rewrite.kind {
            cas_math::trig_sum_product_support::TrigSumToProductContractionRewriteKind::SinSum
            | cas_math::trig_sum_product_support::TrigSumToProductContractionRewriteKind::SinDiff =>
            {
                let Some((rewritten_sin_arg, rewritten_cos_arg)) =
                    extract_scaled_trig_sin_cos_product_args_root(ctx, rewrite.rewritten)
                else {
                    continue;
                };
                let Some((product_sin_arg, product_cos_arg)) =
                    extract_scaled_trig_sin_cos_product_args_root(ctx, product_expr)
                else {
                    continue;
                };
                if compare_expr(ctx, rewritten_sin_arg, product_sin_arg) == Ordering::Equal
                    && matches_expr_or_negation_root(ctx, rewritten_cos_arg, product_cos_arg)
                {
                    return true;
                }
            }
            cas_math::trig_sum_product_support::TrigSumToProductContractionRewriteKind::CosSum => {
                let Some((rewritten_arg_a, rewritten_arg_b)) =
                    extract_scaled_trig_cos_cos_product_args_root(ctx, rewrite.rewritten)
                else {
                    continue;
                };
                let Some((product_arg_a, product_arg_b)) =
                    extract_scaled_trig_cos_cos_product_args_root(ctx, product_expr)
                else {
                    continue;
                };
                if matches_unordered_cos_arg_pair_up_to_sign_root(
                    ctx,
                    rewritten_arg_a,
                    rewritten_arg_b,
                    product_arg_a,
                    product_arg_b,
                ) {
                    return true;
                }
            }
            _ => {
                if compare_expr(ctx, rewrite.rewritten, product_expr) == Ordering::Equal {
                    return true;
                }
            }
        }
    }

    false
}

fn build_plain_trig_sin_cos_product_root(
    ctx: &mut Context,
    sin_arg: ExprId,
    cos_arg: ExprId,
) -> ExprId {
    let sin_expr = ctx.call_builtin(BuiltinFn::Sin, vec![sin_arg]);
    let cos_expr = ctx.call_builtin(BuiltinFn::Cos, vec![cos_arg]);
    build_mul_expr_from_factors_root(ctx, &[sin_expr, cos_expr])
}

fn build_scaled_trig_sin_cos_product_root(
    ctx: &mut Context,
    sin_arg: ExprId,
    cos_arg: ExprId,
) -> ExprId {
    let two = ctx.num(2);
    let product = build_plain_trig_sin_cos_product_root(ctx, sin_arg, cos_arg);
    smart_mul(ctx, two, product)
}

fn build_trig_product_to_sum_double_angle_difference_target_root(
    ctx: &mut Context,
    arg: ExprId,
) -> ExprId {
    let three = ctx.num(3);
    let triple_arg = smart_mul(ctx, three, arg);
    let sin_triple = ctx.call_builtin(BuiltinFn::Sin, vec![triple_arg]);
    let sin_arg = ctx.call_builtin(BuiltinFn::Sin, vec![arg]);
    ctx.add(Expr::Sub(sin_triple, sin_arg))
}

fn build_trig_product_to_sum_double_angle_sum_target_root(
    ctx: &mut Context,
    arg: ExprId,
) -> ExprId {
    let three = ctx.num(3);
    let triple_arg = smart_mul(ctx, three, arg);
    let sin_triple = ctx.call_builtin(BuiltinFn::Sin, vec![triple_arg]);
    let sin_arg = ctx.call_builtin(BuiltinFn::Sin, vec![arg]);
    ctx.add(Expr::Add(sin_triple, sin_arg))
}

fn rewrite_direct_trig_product_to_sum_double_angle_target_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let (sin_arg, cos_arg) = extract_scaled_trig_sin_cos_product_args_root(ctx, expr)?;
    let canonical_cos_arg = canonicalize_even_cos_arg_root(ctx, cos_arg);

    if let Some(base_arg) = extract_double_angle_arg_relaxed(ctx, canonical_cos_arg) {
        if compare_expr(ctx, sin_arg, base_arg) == Ordering::Equal {
            return Some(
                build_trig_product_to_sum_double_angle_difference_target_root(ctx, base_arg),
            );
        }
    }

    if let Some(base_arg) = extract_double_angle_arg_relaxed(ctx, sin_arg) {
        if compare_expr(ctx, canonical_cos_arg, base_arg) == Ordering::Equal {
            return Some(build_trig_product_to_sum_double_angle_sum_target_root(
                ctx, base_arg,
            ));
        }
    }

    None
}

fn build_collapsed_successive_unit_fractions_expr_root(ctx: &mut Context, base: ExprId) -> ExprId {
    let one = ctx.num(1);
    let two = ctx.num(2);
    let doubled_base = smart_mul(ctx, two, base);
    let numerator = ctx.add(Expr::Add(doubled_base, one));
    let plus_one = ctx.add(Expr::Add(base, one));
    let denominator = build_mul_expr_from_factors_root(ctx, &[base, plus_one]);
    ctx.add(Expr::Div(numerator, denominator))
}

fn build_consecutive_telescoping_fraction_difference_expr_root(
    ctx: &mut Context,
    base: ExprId,
) -> ExprId {
    let one = ctx.num(1);
    let plus_one = ctx.add(Expr::Add(base, one));
    let lhs = ctx.add(Expr::Div(one, base));
    let rhs = ctx.add(Expr::Div(one, plus_one));
    ctx.add(Expr::Sub(lhs, rhs))
}

fn extract_scaled_trig_cos_cos_product_args_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId)> {
    let factors = flatten_mul_chain(ctx, expr);
    let mut numeric_coeff = BigRational::one();
    let mut first_cos_arg = None;
    let mut second_cos_arg = None;

    for factor in factors {
        if let Expr::Number(n) = ctx.get(factor) {
            numeric_coeff *= n.clone();
            continue;
        }

        let Some((BuiltinFn::Cos, arg)) = extract_plain_sin_or_cos_arg_root(ctx, factor) else {
            return None;
        };
        if first_cos_arg.is_none() {
            first_cos_arg = Some(arg);
        } else if second_cos_arg.is_none() {
            second_cos_arg = Some(arg);
        } else {
            return None;
        }
    }

    if numeric_coeff != BigRational::from_integer(2.into()) {
        return None;
    }

    Some((first_cos_arg?, second_cos_arg?))
}

fn matches_direct_trig_product_to_sum_cos_cos_zero_identity_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 3 {
        return false;
    }

    let mut product_args = None;
    let mut plain_cos_args: smallvec::SmallVec<[ExprId; 2]> = smallvec::SmallVec::new();

    for (term_expr, term_sign) in view.terms {
        if term_sign == Sign::Pos {
            let Some(args) = extract_scaled_trig_cos_cos_product_args_root(ctx, term_expr) else {
                return false;
            };
            if product_args.is_some() {
                return false;
            }
            product_args = Some(args);
            continue;
        }

        let Some((BuiltinFn::Cos, arg)) = extract_plain_sin_or_cos_arg_root(ctx, term_expr) else {
            return false;
        };
        plain_cos_args.push(arg);
    }

    let Some((lhs_arg, rhs_arg)) = product_args else {
        return false;
    };
    if plain_cos_args.len() != 2 {
        return false;
    }

    (matches_angle_sum_or_diff_arg_root(ctx, plain_cos_args[0], lhs_arg, rhs_arg, true)
        && matches_angle_sum_or_diff_arg_root(ctx, plain_cos_args[1], lhs_arg, rhs_arg, false))
        || (matches_angle_sum_or_diff_arg_root(ctx, plain_cos_args[1], lhs_arg, rhs_arg, true)
            && matches_angle_sum_or_diff_arg_root(ctx, plain_cos_args[0], lhs_arg, rhs_arg, false))
}

fn extract_scaled_trig_sin_cos_product_args_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId)> {
    let factors = flatten_mul_chain(ctx, expr);
    let mut numeric_coeff = BigRational::one();
    let mut sin_arg = None;
    let mut cos_arg = None;

    for factor in factors {
        if let Expr::Number(n) = ctx.get(factor) {
            numeric_coeff *= n.clone();
            continue;
        }

        match extract_plain_sin_or_cos_arg_root(ctx, factor) {
            Some((BuiltinFn::Sin, arg)) => {
                if sin_arg.is_some() {
                    return None;
                }
                sin_arg = Some(arg);
            }
            Some((BuiltinFn::Cos, arg)) => {
                if cos_arg.is_some() {
                    return None;
                }
                cos_arg = Some(arg);
            }
            _ => return None,
        }
    }

    if numeric_coeff != BigRational::from_integer(2.into()) {
        return None;
    }

    Some((sin_arg?, cos_arg?))
}

fn matches_direct_trig_product_to_sum_sin_cos_zero_identity_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 3 {
        return false;
    }

    let mut product_args = None;
    let mut plain_sin_args: smallvec::SmallVec<[ExprId; 2]> = smallvec::SmallVec::new();

    for (term_expr, term_sign) in view.terms {
        if term_sign == Sign::Pos {
            let Some(args) = extract_scaled_trig_sin_cos_product_args_root(ctx, term_expr) else {
                return false;
            };
            if product_args.is_some() {
                return false;
            }
            product_args = Some(args);
            continue;
        }

        let Some((BuiltinFn::Sin, arg)) = extract_plain_sin_or_cos_arg_root(ctx, term_expr) else {
            return false;
        };
        plain_sin_args.push(arg);
    }

    let Some((sin_arg, cos_arg)) = product_args else {
        return false;
    };
    if plain_sin_args.len() != 2 {
        return false;
    }

    let expected_sum = ctx.add(Expr::Add(sin_arg, cos_arg));
    let expected_diff = ctx.add(Expr::Sub(sin_arg, cos_arg));
    (compare_expr(ctx, plain_sin_args[0], expected_sum) == Ordering::Equal
        && compare_expr(ctx, plain_sin_args[1], expected_diff) == Ordering::Equal)
        || (compare_expr(ctx, plain_sin_args[1], expected_sum) == Ordering::Equal
            && compare_expr(ctx, plain_sin_args[0], expected_diff) == Ordering::Equal)
}

fn matches_direct_nested_fraction_simplified_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (continued_fraction_expr, rational_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(arg) =
            extract_depth_three_unit_continued_fraction_arg_root(ctx, continued_fraction_expr)
        else {
            continue;
        };
        if matches_depth_three_unit_continued_fraction_target_root(ctx, rational_expr, arg) {
            return true;
        }
    }

    false
}

fn matches_direct_nested_fraction_simplified_zero_identity_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() < 2 {
        return false;
    }

    for (index, (term_expr, term_sign)) in view.terms.iter().copied().enumerate() {
        if term_sign != Sign::Neg {
            continue;
        }

        let remaining_terms: smallvec::SmallVec<[(ExprId, Sign); 8]> = view
            .terms
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(other_index, term)| (other_index != index).then_some(term))
            .collect();
        if remaining_terms.is_empty() {
            continue;
        }

        let remaining_expr = AddView {
            root: expr,
            terms: remaining_terms,
        }
        .rebuild(ctx);
        if matches_direct_nested_fraction_simplified_pair_root(ctx, remaining_expr, term_expr) {
            return true;
        }
    }

    false
}

fn extract_depth_three_unit_continued_fraction_arg_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let mut current = expr;

    for _ in 0..3 {
        let view = AddView::from_expr(ctx, current);
        if view.terms.len() != 2 {
            return None;
        }

        let mut saw_positive_one = false;
        let mut next = None;
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

            if next.is_some() {
                return None;
            }
            next = extract_unit_fraction_denominator_root(ctx, term_expr);
        }

        if !saw_positive_one {
            return None;
        }
        current = next?;
    }

    Some(current)
}

fn matches_depth_three_unit_continued_fraction_target_root(
    ctx: &mut Context,
    expr: ExprId,
    arg: ExprId,
) -> bool {
    let (numerator, denominator) = match ctx.get(expr) {
        Expr::Div(numerator, denominator) => (*numerator, *denominator),
        _ => return false,
    };

    let one = ctx.num(1);
    let two = ctx.num(2);
    let three = ctx.num(3);
    let three_times_arg = smart_mul(ctx, three, arg);
    let expected_numerator = ctx.add(Expr::Add(three_times_arg, two));
    let two_times_arg = smart_mul(ctx, two, arg);
    let expected_denominator = ctx.add(Expr::Add(two_times_arg, one));

    compare_expr(ctx, numerator, expected_numerator) == Ordering::Equal
        && compare_expr(ctx, denominator, expected_denominator) == Ordering::Equal
}

fn extract_unit_trig_pow2_root(ctx: &Context, expr: ExprId) -> Option<(&'static str, ExprId)> {
    let (coeff, name, arg) = extract_coeff_trig_pow2(ctx, expr)?;
    (coeff == BigRational::one()).then_some((name, arg))
}

fn extract_unit_pythagorean_complement_pow2_root(
    ctx: &Context,
    expr: ExprId,
) -> Option<(&'static str, ExprId)> {
    let terms = AddView::from_expr(ctx, expr).terms;
    if terms.len() != 2 {
        return None;
    }

    let mut saw_one = false;
    let mut trig_term = None;
    for (term_expr, term_sign) in terms {
        if term_sign == Sign::Pos
            && matches!(
                ctx.get(term_expr),
                Expr::Number(n) if *n == BigRational::from_integer(1.into())
            )
        {
            saw_one = true;
            continue;
        }
        if term_sign == Sign::Neg {
            trig_term = extract_unit_trig_pow2_root(ctx, term_expr);
            continue;
        }
        return None;
    }

    saw_one.then_some(())?;
    trig_term
}

fn matches_direct_pythagorean_factor_form_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (pow_side, complement_side) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some((pow_name, pow_arg)) = extract_unit_trig_pow2_root(ctx, pow_side) else {
            continue;
        };
        let Some((complement_name, complement_arg)) =
            extract_unit_pythagorean_complement_pow2_root(ctx, complement_side)
        else {
            continue;
        };
        let expected_complement = match pow_name {
            "sin" => "cos",
            "cos" => "sin",
            _ => continue,
        };
        if complement_name == expected_complement
            && compare_expr(ctx, pow_arg, complement_arg) == Ordering::Equal
        {
            return true;
        }
    }
    false
}

fn matches_direct_pythagorean_identity_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (sum_side, one_side) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Expr::Number(n) = ctx.get(one_side) else {
            continue;
        };
        if !n.is_one() {
            continue;
        }

        let terms = AddView::from_expr(ctx, sum_side).terms;
        if terms.len() != 2 {
            continue;
        }

        let mut sin_arg = None;
        let mut cos_arg = None;
        let mut valid = true;
        for (term_expr, term_sign) in terms {
            if term_sign != Sign::Pos {
                valid = false;
                break;
            }

            let Some((name, arg)) = extract_unit_trig_pow2_root(ctx, term_expr) else {
                valid = false;
                break;
            };

            match name {
                "sin" if sin_arg.is_none() => sin_arg = Some(arg),
                "cos" if cos_arg.is_none() => cos_arg = Some(arg),
                _ => {
                    valid = false;
                    break;
                }
            }
        }

        if valid {
            if let (Some(sin_arg), Some(cos_arg)) = (sin_arg, cos_arg) {
                if compare_expr(ctx, sin_arg, cos_arg) == Ordering::Equal {
                    return true;
                }
            }
        }
    }

    false
}

fn extract_unit_trig_pow_root(
    ctx: &Context,
    expr: ExprId,
    expected_fn: BuiltinFn,
    expected_power: i64,
) -> Option<ExprId> {
    let Expr::Pow(base, exponent) = ctx.get(expr) else {
        return None;
    };
    if extract_i64_integer(ctx, *exponent) != Some(expected_power) {
        return None;
    }
    let (trig_fn, arg) = extract_plain_sin_or_cos_arg_root(ctx, *base)?;
    (trig_fn == expected_fn).then_some(arg)
}

fn extract_direct_pythagorean_extended_lhs_arg_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 2 || !view.terms.iter().all(|(_, sign)| *sign == Sign::Pos) {
        return None;
    }

    let mut sin_arg = None;
    let mut cos_arg = None;
    for (term_expr, _) in view.terms {
        if let Some(arg) = extract_unit_trig_pow_root(ctx, term_expr, BuiltinFn::Sin, 4) {
            if sin_arg.is_some() {
                return None;
            }
            sin_arg = Some(arg);
            continue;
        }
        if let Some(arg) = extract_unit_trig_pow_root(ctx, term_expr, BuiltinFn::Cos, 4) {
            if cos_arg.is_some() {
                return None;
            }
            cos_arg = Some(arg);
            continue;
        }
        return None;
    }

    let (Some(sin_arg), Some(cos_arg)) = (sin_arg, cos_arg) else {
        return None;
    };
    (compare_expr(ctx, sin_arg, cos_arg) == Ordering::Equal).then_some(sin_arg)
}

fn extract_direct_pythagorean_extended_rhs_arg_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 2 {
        return None;
    }

    let mut saw_one = false;
    let mut product_arg = None;
    for (term_expr, term_sign) in view.terms {
        if extract_i64_integer(ctx, term_expr) == Some(1) {
            if term_sign != Sign::Pos || saw_one {
                return None;
            }
            saw_one = true;
            continue;
        }

        if term_sign != Sign::Neg || product_arg.is_some() {
            return None;
        }

        let factors = flatten_mul_chain(ctx, term_expr);
        let mut numeric_coeff = BigRational::one();
        let mut sin_arg = None;
        let mut cos_arg = None;
        for factor in factors {
            if let Expr::Number(n) = ctx.get(factor) {
                numeric_coeff *= n.clone();
                continue;
            }
            if let Some(arg) = extract_unit_trig_pow_root(ctx, factor, BuiltinFn::Sin, 2) {
                if sin_arg.is_some() {
                    return None;
                }
                sin_arg = Some(arg);
                continue;
            }
            if let Some(arg) = extract_unit_trig_pow_root(ctx, factor, BuiltinFn::Cos, 2) {
                if cos_arg.is_some() {
                    return None;
                }
                cos_arg = Some(arg);
                continue;
            }
            return None;
        }

        let (Some(sin_arg), Some(cos_arg)) = (sin_arg, cos_arg) else {
            return None;
        };
        if numeric_coeff != BigRational::from_integer(2.into())
            || compare_expr(ctx, sin_arg, cos_arg) != Ordering::Equal
        {
            return None;
        }
        product_arg = Some(sin_arg);
    }

    saw_one.then_some(product_arg?).or(None)
}

fn matches_direct_pythagorean_extended_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (quartic_side, reduced_side) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(quartic_arg) = extract_direct_pythagorean_extended_lhs_arg_root(ctx, quartic_side)
        else {
            continue;
        };
        let Some(reduced_arg) = extract_direct_pythagorean_extended_rhs_arg_root(ctx, reduced_side)
        else {
            continue;
        };
        if compare_expr(ctx, quartic_arg, reduced_arg) == Ordering::Equal {
            return true;
        }
    }

    false
}

fn extract_scaled_hyperbolic_sinh_cosh_product_half_args_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId)> {
    let factors = flatten_mul_chain(ctx, expr);
    let mut numeric_coeff = BigRational::one();
    let mut sinh_arg = None;
    let mut cosh_arg = None;

    for factor in factors {
        if let Expr::Number(n) = ctx.get(factor) {
            numeric_coeff *= n.clone();
            continue;
        }

        match extract_plain_sinh_or_cosh_arg_root(ctx, factor) {
            Some((BuiltinFn::Sinh, arg)) => {
                if sinh_arg.is_some() {
                    return None;
                }
                sinh_arg = Some(arg);
            }
            Some((BuiltinFn::Cosh, arg)) => {
                if cosh_arg.is_some() {
                    return None;
                }
                cosh_arg = Some(arg);
            }
            _ => return None,
        }
    }

    if numeric_coeff != BigRational::from_integer(2.into()) {
        return None;
    }

    Some((sinh_arg?, cosh_arg?))
}

fn extract_scaled_hyperbolic_cosh_cosh_product_half_args_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId)> {
    let factors = flatten_mul_chain(ctx, expr);
    let mut numeric_coeff = BigRational::one();
    let mut first_cosh_arg = None;
    let mut second_cosh_arg = None;

    for factor in factors {
        if let Expr::Number(n) = ctx.get(factor) {
            numeric_coeff *= n.clone();
            continue;
        }

        let Some((BuiltinFn::Cosh, arg)) = extract_plain_sinh_or_cosh_arg_root(ctx, factor) else {
            return None;
        };
        if first_cosh_arg.is_none() {
            first_cosh_arg = Some(arg);
        } else if second_cosh_arg.is_none() {
            second_cosh_arg = Some(arg);
        } else {
            return None;
        }
    }

    if numeric_coeff != BigRational::from_integer(2.into()) {
        return None;
    }

    Some((first_cosh_arg?, second_cosh_arg?))
}

fn extract_scaled_hyperbolic_sinh_sinh_product_half_args_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId)> {
    let factors = flatten_mul_chain(ctx, expr);
    let mut numeric_coeff = BigRational::one();
    let mut first_sinh_arg = None;
    let mut second_sinh_arg = None;

    for factor in factors {
        if let Expr::Number(n) = ctx.get(factor) {
            numeric_coeff *= n.clone();
            continue;
        }

        let Some((BuiltinFn::Sinh, arg)) = extract_plain_sinh_or_cosh_arg_root(ctx, factor) else {
            return None;
        };
        if first_sinh_arg.is_none() {
            first_sinh_arg = Some(arg);
        } else if second_sinh_arg.is_none() {
            second_sinh_arg = Some(arg);
        } else {
            return None;
        }
    }

    if numeric_coeff != BigRational::from_integer(2.into()) {
        return None;
    }

    Some((first_sinh_arg?, second_sinh_arg?))
}

fn extract_plain_hyperbolic_product_pair_args_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<((BuiltinFn, ExprId), (BuiltinFn, ExprId))> {
    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() != 2 {
        return None;
    }

    let lhs = extract_plain_sinh_or_cosh_arg_root(ctx, factors[0])?;
    let rhs = extract_plain_sinh_or_cosh_arg_root(ctx, factors[1])?;
    Some((lhs, rhs))
}

fn build_half_expr_root(ctx: &mut Context, expr: ExprId) -> ExprId {
    let two = ctx.num(2);
    ctx.add(Expr::Div(expr, two))
}

fn matches_direct_hyperbolic_sinh_sum_to_product_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (sum_expr, product_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let sum_view = AddView::from_expr(ctx, sum_expr);
        if sum_view.terms.len() != 2 || !sum_view.terms.iter().all(|(_, sign)| *sign == Sign::Pos) {
            continue;
        }

        let mut sum_args: smallvec::SmallVec<[ExprId; 2]> = smallvec::SmallVec::new();
        let mut bad_sum = false;
        for (term_expr, _term_sign) in sum_view.terms {
            let Some((BuiltinFn::Sinh, arg)) = extract_plain_sinh_or_cosh_arg_root(ctx, term_expr)
            else {
                bad_sum = true;
                break;
            };
            sum_args.push(arg);
        }
        if bad_sum || sum_args.len() != 2 {
            continue;
        }

        let Some((sinh_half_sum_arg, cosh_half_diff_arg)) =
            extract_scaled_hyperbolic_sinh_cosh_product_half_args_root(ctx, product_expr)
        else {
            continue;
        };

        let sum_expr = ctx.add(Expr::Add(sum_args[0], sum_args[1]));
        let half_sum = build_half_expr_root(ctx, sum_expr);
        if compare_expr(ctx, sinh_half_sum_arg, half_sum) != Ordering::Equal {
            continue;
        }

        let diff_ab_expr = ctx.add(Expr::Sub(sum_args[0], sum_args[1]));
        let half_diff_ab = build_half_expr_root(ctx, diff_ab_expr);
        let diff_ba_expr = ctx.add(Expr::Sub(sum_args[1], sum_args[0]));
        let half_diff_ba = build_half_expr_root(ctx, diff_ba_expr);
        if compare_expr(ctx, cosh_half_diff_arg, half_diff_ab) == Ordering::Equal
            || compare_expr(ctx, cosh_half_diff_arg, half_diff_ba) == Ordering::Equal
        {
            return true;
        }
    }

    false
}

fn matches_direct_hyperbolic_cosh_sum_to_product_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (sum_expr, product_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let sum_view = AddView::from_expr(ctx, sum_expr);
        if sum_view.terms.len() != 2 || !sum_view.terms.iter().all(|(_, sign)| *sign == Sign::Pos) {
            continue;
        }

        let mut sum_args: smallvec::SmallVec<[ExprId; 2]> = smallvec::SmallVec::new();
        let mut bad_sum = false;
        for (term_expr, _term_sign) in sum_view.terms {
            let Some((BuiltinFn::Cosh, arg)) = extract_plain_sinh_or_cosh_arg_root(ctx, term_expr)
            else {
                bad_sum = true;
                break;
            };
            sum_args.push(arg);
        }
        if bad_sum || sum_args.len() != 2 {
            continue;
        }

        let Some((product_arg_a, product_arg_b)) =
            extract_scaled_hyperbolic_cosh_cosh_product_half_args_root(ctx, product_expr)
        else {
            continue;
        };

        let sum_expr = ctx.add(Expr::Add(sum_args[0], sum_args[1]));
        let half_sum = build_half_expr_root(ctx, sum_expr);
        let diff_ab_expr = ctx.add(Expr::Sub(sum_args[0], sum_args[1]));
        let half_diff_ab = build_half_expr_root(ctx, diff_ab_expr);
        let diff_ba_expr = ctx.add(Expr::Sub(sum_args[1], sum_args[0]));
        let half_diff_ba = build_half_expr_root(ctx, diff_ba_expr);

        let direct_order = compare_expr(ctx, product_arg_a, half_sum) == Ordering::Equal
            && (compare_expr(ctx, product_arg_b, half_diff_ab) == Ordering::Equal
                || compare_expr(ctx, product_arg_b, half_diff_ba) == Ordering::Equal);
        let swapped_order = compare_expr(ctx, product_arg_b, half_sum) == Ordering::Equal
            && (compare_expr(ctx, product_arg_a, half_diff_ab) == Ordering::Equal
                || compare_expr(ctx, product_arg_a, half_diff_ba) == Ordering::Equal);
        if direct_order || swapped_order {
            return true;
        }
    }

    false
}

fn matches_direct_hyperbolic_cosh_difference_to_product_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (difference_expr, product_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let diff_view = AddView::from_expr(ctx, difference_expr);
        if diff_view.terms.len() != 2 {
            continue;
        }

        let mut positive_cosh_arg = None;
        let mut negative_cosh_arg = None;
        let mut bad_diff = false;
        for (term_expr, term_sign) in diff_view.terms {
            let Some((BuiltinFn::Cosh, arg)) = extract_plain_sinh_or_cosh_arg_root(ctx, term_expr)
            else {
                bad_diff = true;
                break;
            };
            match term_sign {
                Sign::Pos if positive_cosh_arg.is_none() => positive_cosh_arg = Some(arg),
                Sign::Neg if negative_cosh_arg.is_none() => negative_cosh_arg = Some(arg),
                _ => {
                    bad_diff = true;
                    break;
                }
            }
        }
        if bad_diff {
            continue;
        }

        let (Some(lhs_arg), Some(rhs_arg)) = (positive_cosh_arg, negative_cosh_arg) else {
            continue;
        };
        let Some((product_arg_a, product_arg_b)) =
            extract_scaled_hyperbolic_sinh_sinh_product_half_args_root(ctx, product_expr)
        else {
            continue;
        };

        let sum_expr = ctx.add(Expr::Add(lhs_arg, rhs_arg));
        let half_sum = build_half_expr_root(ctx, sum_expr);
        let diff_ab_expr = ctx.add(Expr::Sub(lhs_arg, rhs_arg));
        let half_diff_ab = build_half_expr_root(ctx, diff_ab_expr);
        let diff_ba_expr = ctx.add(Expr::Sub(rhs_arg, lhs_arg));
        let half_diff_ba = build_half_expr_root(ctx, diff_ba_expr);

        let direct_order = compare_expr(ctx, product_arg_a, half_sum) == Ordering::Equal
            && (compare_expr(ctx, product_arg_b, half_diff_ab) == Ordering::Equal
                || compare_expr(ctx, product_arg_b, half_diff_ba) == Ordering::Equal);
        let swapped_order = compare_expr(ctx, product_arg_b, half_sum) == Ordering::Equal
            && (compare_expr(ctx, product_arg_a, half_diff_ab) == Ordering::Equal
                || compare_expr(ctx, product_arg_a, half_diff_ba) == Ordering::Equal);
        if direct_order || swapped_order {
            return true;
        }
    }

    false
}

fn matches_direct_recursive_hyperbolic_sinh_sum_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (single_expr, expanded_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some((BuiltinFn::Sinh, angle_arg)) =
            extract_plain_sinh_or_cosh_arg_root(ctx, single_expr)
        else {
            continue;
        };

        let view = AddView::from_expr(ctx, expanded_expr);
        if view.terms.len() != 2 || !view.terms.iter().all(|(_, sign)| *sign == Sign::Pos) {
            continue;
        }

        let Some(((lhs_fn_a, lhs_arg_a), (lhs_fn_b, lhs_arg_b))) =
            extract_plain_hyperbolic_product_pair_args_root(ctx, view.terms[0].0)
        else {
            continue;
        };
        let Some(((rhs_fn_a, rhs_arg_a), (rhs_fn_b, rhs_arg_b))) =
            extract_plain_hyperbolic_product_pair_args_root(ctx, view.terms[1].0)
        else {
            continue;
        };

        let lhs_is_sinh_cosh = matches!(
            (lhs_fn_a, lhs_fn_b),
            (BuiltinFn::Sinh, BuiltinFn::Cosh) | (BuiltinFn::Cosh, BuiltinFn::Sinh)
        );
        let rhs_is_sinh_cosh = matches!(
            (rhs_fn_a, rhs_fn_b),
            (BuiltinFn::Sinh, BuiltinFn::Cosh) | (BuiltinFn::Cosh, BuiltinFn::Sinh)
        );
        if !lhs_is_sinh_cosh || !rhs_is_sinh_cosh {
            continue;
        }

        let lhs_sinh_arg = if lhs_fn_a == BuiltinFn::Sinh {
            lhs_arg_a
        } else {
            lhs_arg_b
        };
        let lhs_cosh_arg = if lhs_fn_a == BuiltinFn::Cosh {
            lhs_arg_a
        } else {
            lhs_arg_b
        };
        let rhs_sinh_arg = if rhs_fn_a == BuiltinFn::Sinh {
            rhs_arg_a
        } else {
            rhs_arg_b
        };
        let rhs_cosh_arg = if rhs_fn_a == BuiltinFn::Cosh {
            rhs_arg_a
        } else {
            rhs_arg_b
        };

        if compare_expr(ctx, lhs_sinh_arg, rhs_cosh_arg) != Ordering::Equal
            || compare_expr(ctx, lhs_cosh_arg, rhs_sinh_arg) != Ordering::Equal
        {
            continue;
        }

        if matches_angle_sum_or_diff_arg_root(ctx, angle_arg, lhs_sinh_arg, lhs_cosh_arg, true) {
            return true;
        }
    }

    false
}

fn matches_direct_recursive_hyperbolic_cosh_sum_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (single_expr, expanded_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some((BuiltinFn::Cosh, angle_arg)) =
            extract_plain_sinh_or_cosh_arg_root(ctx, single_expr)
        else {
            continue;
        };

        let view = AddView::from_expr(ctx, expanded_expr);
        if view.terms.len() != 2 || !view.terms.iter().all(|(_, sign)| *sign == Sign::Pos) {
            continue;
        }

        let Some(((lhs_fn_a, lhs_arg_a), (lhs_fn_b, lhs_arg_b))) =
            extract_plain_hyperbolic_product_pair_args_root(ctx, view.terms[0].0)
        else {
            continue;
        };
        let Some(((rhs_fn_a, rhs_arg_a), (rhs_fn_b, rhs_arg_b))) =
            extract_plain_hyperbolic_product_pair_args_root(ctx, view.terms[1].0)
        else {
            continue;
        };

        let lhs_is_cosh_cosh = lhs_fn_a == BuiltinFn::Cosh && lhs_fn_b == BuiltinFn::Cosh;
        let rhs_is_sinh_sinh = rhs_fn_a == BuiltinFn::Sinh && rhs_fn_b == BuiltinFn::Sinh;
        let lhs_is_sinh_sinh = lhs_fn_a == BuiltinFn::Sinh && lhs_fn_b == BuiltinFn::Sinh;
        let rhs_is_cosh_cosh = rhs_fn_a == BuiltinFn::Cosh && rhs_fn_b == BuiltinFn::Cosh;

        let (arg_u, arg_v) = if lhs_is_cosh_cosh && rhs_is_sinh_sinh {
            if !matches_unordered_expr_pair_root(ctx, lhs_arg_a, lhs_arg_b, rhs_arg_a, rhs_arg_b) {
                continue;
            }
            (lhs_arg_a, lhs_arg_b)
        } else if lhs_is_sinh_sinh && rhs_is_cosh_cosh {
            if !matches_unordered_expr_pair_root(ctx, lhs_arg_a, lhs_arg_b, rhs_arg_a, rhs_arg_b) {
                continue;
            }
            (rhs_arg_a, rhs_arg_b)
        } else {
            continue;
        };

        if matches_angle_sum_or_diff_arg_root(ctx, angle_arg, arg_u, arg_v, true) {
            return true;
        }
    }

    false
}

fn matches_direct_negative_double_cos_square_diff_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (square_diff, negative_cos) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(square_arg) = extract_mixed_sign_trig_square_difference_arg_root(ctx, square_diff)
        else {
            continue;
        };
        let Some(double_angle_arg) = extract_negative_cos_double_angle_arg_root(ctx, negative_cos)
        else {
            continue;
        };
        if compare_expr(ctx, square_arg, double_angle_arg) == Ordering::Equal {
            return true;
        }
    }

    false
}

fn extract_positive_cos_double_angle_arg_root(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let Some((BuiltinFn::Cos, arg)) = extract_plain_sin_or_cos_arg_root(ctx, expr) else {
        return None;
    };
    extract_double_angle_arg_relaxed(ctx, arg)
}

fn extract_direct_cos_minus_sin_square_diff_target_root(
    ctx: &Context,
    expr: ExprId,
) -> Option<ExprId> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 2 {
        return None;
    }

    let mut positive_cos_arg = None;
    let mut negative_sin_arg = None;

    for (term_expr, term_sign) in view.terms {
        let (coeff, trig_name, arg, effective_sign) =
            extract_signed_numeric_trig_pow2(ctx, term_expr, term_sign)?;
        if coeff != BigRational::one() {
            return None;
        }
        match (trig_name, effective_sign) {
            ("cos", Sign::Pos) => {
                if positive_cos_arg.is_some() {
                    return None;
                }
                positive_cos_arg = Some(arg);
            }
            ("sin", Sign::Neg) => {
                if negative_sin_arg.is_some() {
                    return None;
                }
                negative_sin_arg = Some(arg);
            }
            _ => return None,
        }
    }

    let positive_cos_arg = positive_cos_arg?;
    let negative_sin_arg = negative_sin_arg?;
    (compare_expr(ctx, positive_cos_arg, negative_sin_arg) == Ordering::Equal)
        .then_some(positive_cos_arg)
}

fn build_positive_cos_double_angle_expr_root(ctx: &mut Context, arg: ExprId) -> ExprId {
    let two = ctx.num(2);
    let doubled_arg = smart_mul(ctx, two, arg);
    ctx.call_builtin(BuiltinFn::Cos, vec![doubled_arg])
}

fn extract_direct_positive_double_cos_square_diff_target_root(
    ctx: &Context,
    expr: ExprId,
) -> Option<ExprId> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 2 {
        return None;
    }

    let mut cos_sq_arg = None;
    let mut saw_negative_one = false;

    for (term_expr, term_sign) in view.terms {
        if extract_i64_integer(ctx, term_expr) == Some(1) {
            if term_sign != Sign::Neg || saw_negative_one {
                return None;
            }
            saw_negative_one = true;
            continue;
        }

        let (coeff, trig_name, arg, effective_sign) =
            extract_signed_numeric_trig_pow2(ctx, term_expr, term_sign)?;
        if trig_name != "cos"
            || effective_sign != Sign::Pos
            || coeff != BigRational::from_integer(2.into())
            || cos_sq_arg.is_some()
        {
            return None;
        }
        cos_sq_arg = Some(arg);
    }

    saw_negative_one.then_some(cos_sq_arg?).or(None)
}

fn matches_direct_positive_double_cos_square_diff_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (square_diff, positive_cos) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(square_arg) =
            extract_direct_positive_double_cos_square_diff_target_root(ctx, square_diff)
        else {
            continue;
        };
        let Some(double_angle_arg) = extract_positive_cos_double_angle_arg_root(ctx, positive_cos)
        else {
            continue;
        };
        if compare_expr(ctx, square_arg, double_angle_arg) == Ordering::Equal {
            return true;
        }
    }

    false
}

fn matches_direct_cos_minus_sin_square_diff_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (square_diff, positive_cos) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(square_arg) =
            extract_direct_cos_minus_sin_square_diff_target_root(ctx, square_diff)
        else {
            continue;
        };
        let Some(double_angle_arg) = extract_positive_cos_double_angle_arg_root(ctx, positive_cos)
        else {
            continue;
        };
        if compare_expr(ctx, square_arg, double_angle_arg) == Ordering::Equal {
            return true;
        }
    }

    false
}

fn matches_direct_cos_square_diff_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    matches_direct_negative_double_cos_square_diff_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_cos_minus_sin_square_diff_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_positive_double_cos_square_diff_pair_root(ctx, lhs_core, rhs_core)
}

fn matches_direct_abs_square_pair_root(ctx: &Context, lhs_core: ExprId, rhs_core: ExprId) -> bool {
    for (abs_square, plain_square) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Expr::Pow(abs_base, abs_exp) = ctx.get(abs_square) else {
            continue;
        };
        if extract_i64_integer(ctx, *abs_exp) != Some(2) {
            continue;
        }
        let Some(abs_inner) = try_unwrap_abs_arg(ctx, *abs_base) else {
            continue;
        };

        let Expr::Pow(plain_base, plain_exp) = ctx.get(plain_square) else {
            continue;
        };
        if extract_i64_integer(ctx, *plain_exp) != Some(2) {
            continue;
        }

        if compare_expr(ctx, abs_inner, *plain_base) == Ordering::Equal {
            return true;
        }
    }

    false
}

fn canonicalize_direct_reciprocal_sqrt_pair_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    try_rewrite_reciprocal_sqrt_canon_expr(ctx, expr).map(|rewrite| rewrite.rewritten)
}

fn matches_direct_reciprocal_sqrt_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    let Some(lhs_canon) = canonicalize_direct_reciprocal_sqrt_pair_root(ctx, lhs_core) else {
        return false;
    };
    let Some(rhs_canon) = canonicalize_direct_reciprocal_sqrt_pair_root(ctx, rhs_core) else {
        return false;
    };
    compare_expr(ctx, lhs_canon, rhs_canon) == Ordering::Equal
}

fn sort_direct_pair_args_root(ctx: &mut Context, lhs: ExprId, rhs: ExprId) -> (ExprId, ExprId) {
    if compare_expr(ctx, lhs, rhs) == Ordering::Greater {
        (rhs, lhs)
    } else {
        (lhs, rhs)
    }
}

fn extract_direct_exponential_product_pair_args_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId)> {
    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() != 2 {
        return None;
    }
    let lhs = extract_exp_argument(ctx, factors[0])?;
    let rhs = extract_exp_argument(ctx, factors[1])?;
    Some(sort_direct_pair_args_root(ctx, lhs, rhs))
}

fn extract_direct_exponential_sum_pair_args_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId)> {
    let sum_expr = extract_exp_argument(ctx, expr)?;
    let view = AddView::from_expr(ctx, sum_expr);
    if view.terms.len() != 2 {
        return None;
    }
    let mut args = Vec::with_capacity(2);
    for (term_expr, term_sign) in view.terms {
        if term_sign != Sign::Pos {
            return None;
        }
        args.push(term_expr);
    }
    Some(sort_direct_pair_args_root(ctx, args[0], args[1]))
}

fn matches_direct_exponential_combination_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (product_expr, combined_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some((prod_lhs, prod_rhs)) =
            extract_direct_exponential_product_pair_args_root(ctx, product_expr)
        else {
            continue;
        };
        let Some((sum_lhs, sum_rhs)) =
            extract_direct_exponential_sum_pair_args_root(ctx, combined_expr)
        else {
            continue;
        };
        if compare_expr(ctx, prod_lhs, sum_lhs) == Ordering::Equal
            && compare_expr(ctx, prod_rhs, sum_rhs) == Ordering::Equal
        {
            return true;
        }
    }

    false
}

fn extract_direct_hyperbolic_exp_sum_target_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, bool)> {
    use cas_math::expr_nary::{AddView, Sign};

    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 2 {
        return None;
    }

    let mut cosh_arg = None;
    let mut sinh_arg = None;
    let mut sinh_positive = None;

    for (term_expr, term_sign) in view.terms {
        match extract_plain_sinh_or_cosh_arg_root(ctx, term_expr) {
            Some((BuiltinFn::Cosh, arg)) => {
                if term_sign != Sign::Pos || cosh_arg.is_some() {
                    return None;
                }
                cosh_arg = Some(arg);
            }
            Some((BuiltinFn::Sinh, arg)) => {
                if sinh_arg.is_some() {
                    return None;
                }
                sinh_arg = Some(arg);
                sinh_positive = Some(term_sign == Sign::Pos);
            }
            _ => return None,
        }
    }

    let cosh_arg = cosh_arg?;
    let sinh_arg = sinh_arg?;
    if compare_expr(ctx, cosh_arg, sinh_arg) != Ordering::Equal {
        return None;
    }

    Some((cosh_arg, sinh_positive?))
}

fn build_direct_hyperbolic_exp_sum_target_root(
    ctx: &mut Context,
    arg: ExprId,
    is_sum: bool,
) -> ExprId {
    let exp_arg = if is_sum { arg } else { ctx.add(Expr::Neg(arg)) };
    ctx.call_builtin(BuiltinFn::Exp, vec![exp_arg])
}

fn matches_direct_hyperbolic_exp_sum_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (hyper_expr, exp_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some((arg, is_sum)) = extract_direct_hyperbolic_exp_sum_target_root(ctx, hyper_expr)
        else {
            continue;
        };
        let Some(exp_arg) = extract_exp_argument(ctx, exp_expr) else {
            continue;
        };
        let expected_arg = if is_sum { arg } else { ctx.add(Expr::Neg(arg)) };
        if compare_expr(ctx, exp_arg, expected_arg) == Ordering::Equal {
            return true;
        }
    }

    false
}

fn matches_direct_quintuple_angle_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    let lhs_rewrite = try_rewrite_quintuple_angle_expr(ctx, lhs_core);
    if lhs_rewrite
        .is_some_and(|rewrite| render_expr(ctx, rewrite.rewritten) == render_expr(ctx, rhs_core))
    {
        return true;
    }

    let rhs_rewrite = try_rewrite_quintuple_angle_expr(ctx, rhs_core);
    rhs_rewrite
        .is_some_and(|rewrite| render_expr(ctx, rewrite.rewritten) == render_expr(ctx, lhs_core))
}

fn matches_direct_trig_cubic_cosine_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    crate::rules::arithmetic::try_build_direct_trig_cos_double_angle_polynomial_equivalence_rewrite(
        ctx, lhs_core, rhs_core,
    )
    .is_some()
        || crate::rules::arithmetic::try_build_direct_trig_sine_product_cubic_equivalence_rewrite(
            ctx, lhs_core, rhs_core,
        )
        .is_some()
}

fn matches_direct_trig_mixed_double_angle_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    let lhs_minus_rhs = ctx.add(Expr::Sub(lhs_core, rhs_core));
    if matches_direct_trig_mixed_double_angle_zero_identity_root(ctx, lhs_minus_rhs) {
        return true;
    }

    let rhs_minus_lhs = ctx.add(Expr::Sub(rhs_core, lhs_core));
    matches_direct_trig_mixed_double_angle_zero_identity_root(ctx, rhs_minus_lhs)
}

fn matches_direct_small_pow_expansion_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    if expr_contains_trig_or_hyperbolic_builtin_local(ctx, lhs_core)
        || expr_contains_trig_or_hyperbolic_builtin_local(ctx, rhs_core)
    {
        return false;
    }

    for (source, target) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(expanded) = try_expand_small_pow_sum_expr(
            ctx,
            source,
            SmallPowExpandPolicy {
                max_vars: 3,
                ..SmallPowExpandPolicy::default()
            },
        ) else {
            continue;
        };
        let expanded = cas_ast::hold::unwrap_hold(ctx, expanded);
        if compare_expr(ctx, expanded, target) == Ordering::Equal {
            return true;
        }
        if cas_ast::count_nodes(ctx, expanded) <= 24 && cas_ast::count_nodes(ctx, target) <= 24 {
            if isolated_simplify_rewrites_to_target(
                &crate::phase::SimplifyOptions::default(),
                ctx,
                expanded,
                target,
            ) {
                return true;
            }

            let difference = ctx.add(Expr::Sub(expanded, target));
            if isolated_simplify_rewrites_to_zero(
                &crate::phase::SimplifyOptions::default(),
                ctx,
                difference,
            ) {
                return true;
            }
        }
    }

    false
}

fn extract_plain_pow2_base_root(ctx: &Context, expr: ExprId) -> Option<ExprId> {
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

fn extract_base_plus_constant_root(
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

fn extract_direct_factored_linear_shift_pair_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, BigRational)> {
    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() != 2 {
        return None;
    }

    for (base_index, shifted_index) in [(0usize, 1usize), (1usize, 0usize)] {
        let base = factors[base_index];
        let Some((shifted_base, constant)) =
            extract_base_plus_constant_root(ctx, factors[shifted_index])
        else {
            continue;
        };
        if compare_expr(ctx, base, shifted_base) == Ordering::Equal {
            return Some((base, constant));
        }
    }

    None
}

fn extract_direct_expanded_linear_shift_pair_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, BigRational)> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 2 {
        return None;
    }

    let mut square_base = None;
    let mut linear_term = None;
    for (term_expr, term_sign) in view.terms {
        if let Some(base) = extract_plain_pow2_base_root(ctx, term_expr) {
            if term_sign == Sign::Neg || square_base.is_some() {
                return None;
            }
            square_base = Some(base);
            continue;
        }

        let (mut coeff, base) = extract_coef_and_base(ctx, term_expr);
        if term_sign == Sign::Neg {
            coeff = -coeff;
        }
        if linear_term.is_some() || coeff.is_zero() {
            return None;
        }
        linear_term = Some((base, coeff));
    }

    let (Some(square_base), Some((linear_base, constant))) = (square_base, linear_term) else {
        return None;
    };
    (compare_expr(ctx, square_base, linear_base) == Ordering::Equal)
        .then_some((square_base, constant))
}

fn matches_direct_linear_factoring_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    let lhs = extract_direct_expanded_linear_shift_pair_root(ctx, lhs_core)
        .or_else(|| extract_direct_factored_linear_shift_pair_root(ctx, lhs_core));
    let rhs = extract_direct_expanded_linear_shift_pair_root(ctx, rhs_core)
        .or_else(|| extract_direct_factored_linear_shift_pair_root(ctx, rhs_core));

    let (Some((lhs_base, lhs_constant)), Some((rhs_base, rhs_constant))) = (lhs, rhs) else {
        return false;
    };
    compare_expr(ctx, lhs_base, rhs_base) == Ordering::Equal && lhs_constant == rhs_constant
}

fn extract_direct_two_linear_shift_product_root(
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

fn matches_direct_two_linear_shift_product_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (product_expr, expanded_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some((product_base, product_constants)) =
            extract_direct_two_linear_shift_product_root(ctx, product_expr)
        else {
            continue;
        };

        let Some(rewrite) = try_rewrite_automatic_factor_expr(ctx, expanded_expr) else {
            continue;
        };
        let factored = strip_multiplicative_one_root(ctx, rewrite.rewritten);
        if compare_expr(ctx, factored, product_expr) == Ordering::Equal {
            return true;
        }

        let Some((factored_base, factored_constants)) =
            extract_direct_two_linear_shift_product_root(ctx, factored)
        else {
            continue;
        };

        if compare_expr(ctx, product_base, factored_base) == Ordering::Equal
            && product_constants == factored_constants
        {
            return true;
        }
    }

    false
}

fn extract_direct_three_linear_shift_product_root(
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

fn matches_direct_three_linear_shift_product_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (product_expr, expanded_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some((product_base, product_constants)) =
            extract_direct_three_linear_shift_product_root(ctx, product_expr)
        else {
            continue;
        };

        let Some(rewrite) = try_rewrite_automatic_factor_expr(ctx, expanded_expr) else {
            continue;
        };
        let factored = strip_multiplicative_one_root(ctx, rewrite.rewritten);
        if compare_expr(ctx, factored, product_expr) == Ordering::Equal {
            return true;
        }

        let Some((factored_base, factored_constants)) =
            extract_direct_three_linear_shift_product_root(ctx, factored)
        else {
            continue;
        };

        if compare_expr(ctx, product_base, factored_base) == Ordering::Equal
            && product_constants == factored_constants
        {
            return true;
        }
    }

    false
}

fn matches_direct_difference_of_squares_quotient_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (quotient_expr, target_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Expr::Div(numerator, denominator) = ctx.get(quotient_expr) else {
            continue;
        };
        let Some(plan) =
            cas_math::difference_of_squares_support::try_plan_difference_of_squares_division_expr(
                ctx,
                *numerator,
                *denominator,
                cas_math::difference_of_squares_support::DifferenceOfSquaresDivisionPolicy::default(
                ),
            )
        else {
            continue;
        };

        if compare_expr(ctx, plan.final_result, target_expr) == Ordering::Equal {
            return true;
        }
        if cas_ast::count_nodes(ctx, plan.final_result) <= 24
            && cas_ast::count_nodes(ctx, target_expr) <= 24
            && isolated_simplify_rewrites_to_target(
                &crate::phase::SimplifyOptions::default(),
                ctx,
                plan.final_result,
                target_expr,
            )
        {
            return true;
        }
    }

    false
}

fn build_sum_diff_cubes_quotient_expansion_root(
    ctx: &mut Context,
    lhs: ExprId,
    rhs: ExprId,
) -> ExprId {
    let build_square = |ctx: &mut Context, expr: ExprId| -> ExprId {
        match ctx.get(expr) {
            Expr::Number(n) => ctx.add(Expr::Number(n.clone() * n.clone())),
            _ => {
                let two = ctx.num(2);
                ctx.add(Expr::Pow(expr, two))
            }
        }
    };
    let lhs_sq = build_square(ctx, lhs);
    let rhs_sq = build_square(ctx, rhs);
    let mixed = build_mul_expr_from_factors_root(ctx, &[lhs, rhs]);
    build_balanced_add(ctx, &[lhs_sq, mixed, rhs_sq])
}

fn matches_direct_sum_diff_cubes_quotient_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (quotient_expr, target_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some((lhs, rhs)) = extract_sum_diff_cubes_quotient_bases_root(ctx, quotient_expr)
        else {
            continue;
        };
        let expanded = build_sum_diff_cubes_quotient_expansion_root(ctx, lhs, rhs);
        if compare_expr(ctx, expanded, target_expr) == Ordering::Equal {
            return true;
        }
        if cas_ast::count_nodes(ctx, expanded) <= 24
            && cas_ast::count_nodes(ctx, target_expr) <= 24
            && isolated_simplify_rewrites_to_target(
                &crate::phase::SimplifyOptions::default(),
                ctx,
                expanded,
                target_expr,
            )
        {
            return true;
        }
    }

    false
}

fn matches_direct_sec_tan_pythagorean_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (source_expr, target_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(rewrite) = try_rewrite_sec_tan_pythagorean_identity_expr(ctx, source_expr) else {
            continue;
        };
        if compare_expr(ctx, rewrite.rewritten, target_expr) == Ordering::Equal {
            return true;
        }
        if cas_ast::count_nodes(ctx, rewrite.rewritten) <= 24
            && cas_ast::count_nodes(ctx, target_expr) <= 24
            && isolated_simplify_rewrites_to_target(
                &crate::phase::SimplifyOptions::default(),
                ctx,
                rewrite.rewritten,
                target_expr,
            )
        {
            return true;
        }
    }

    false
}

fn matches_direct_tan_to_sec_pythagorean_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (source_expr, target_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(rewrite) = try_rewrite_tan_to_sec_pythagorean_identity_expr(ctx, source_expr)
        else {
            continue;
        };
        if compare_expr(ctx, rewrite.rewritten, target_expr) == Ordering::Equal {
            return true;
        }
        if cas_ast::count_nodes(ctx, rewrite.rewritten) <= 24
            && cas_ast::count_nodes(ctx, target_expr) <= 24
            && isolated_simplify_rewrites_to_target(
                &crate::phase::SimplifyOptions::default(),
                ctx,
                rewrite.rewritten,
                target_expr,
            )
        {
            return true;
        }
    }

    false
}

fn extract_direct_quartic_gcf_base_expanded_root(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Sub(lhs, rhs) = ctx.get(expr) else {
        return None;
    };
    let fourth_base = extract_plain_pow_base_root(ctx, *lhs, 4)?;
    let squared_base = extract_plain_pow_base_root(ctx, *rhs, 2)?;
    (compare_expr(ctx, fourth_base, squared_base) == Ordering::Equal).then_some(fourth_base)
}

fn extract_direct_quartic_gcf_base_factored_root(
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

fn matches_direct_quartic_gcf_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (expanded_expr, factored_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(expanded_base) = extract_direct_quartic_gcf_base_expanded_root(ctx, expanded_expr)
        else {
            continue;
        };
        let Some(factored_base) = extract_direct_quartic_gcf_base_factored_root(ctx, factored_expr)
        else {
            continue;
        };
        if compare_expr(ctx, expanded_base, factored_base) == Ordering::Equal {
            return true;
        }
    }

    false
}

fn matches_direct_csc_cot_pythagorean_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (source_expr, target_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(rewrite) = try_rewrite_csc_cot_pythagorean_identity_expr(ctx, source_expr) else {
            continue;
        };
        if compare_expr(ctx, rewrite.rewritten, target_expr) == Ordering::Equal {
            return true;
        }
        if cas_ast::count_nodes(ctx, rewrite.rewritten) <= 24
            && cas_ast::count_nodes(ctx, target_expr) <= 24
            && isolated_simplify_rewrites_to_target(
                &crate::phase::SimplifyOptions::default(),
                ctx,
                rewrite.rewritten,
                target_expr,
            )
        {
            return true;
        }
    }

    false
}

fn matches_direct_cot_to_csc_pythagorean_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (source_expr, target_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(rewrite) = try_rewrite_cot_to_csc_pythagorean_identity_expr(ctx, source_expr)
        else {
            continue;
        };
        if compare_expr(ctx, rewrite.rewritten, target_expr) == Ordering::Equal {
            return true;
        }
        if cas_ast::count_nodes(ctx, rewrite.rewritten) <= 24
            && cas_ast::count_nodes(ctx, target_expr) <= 24
            && isolated_simplify_rewrites_to_target(
                &crate::phase::SimplifyOptions::default(),
                ctx,
                rewrite.rewritten,
                target_expr,
            )
        {
            return true;
        }
    }

    false
}

fn matches_direct_hyperbolic_pythagorean_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (identity_expr, target_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        if !matches!(ctx.get(target_expr), Expr::Number(n) if n.is_one()) {
            continue;
        }

        let view = AddView::from_expr(ctx, identity_expr);
        if view.terms.len() != 2 {
            continue;
        }

        let mut cosh_arg = None;
        let mut sinh_arg = None;
        let mut valid = true;
        for (term_expr, sign) in view.terms {
            match (
                extract_plain_sinh_or_cosh_pow2_arg_root(ctx, term_expr),
                sign,
            ) {
                (Some((BuiltinFn::Cosh, arg)), Sign::Pos) => {
                    if cosh_arg.replace(arg).is_some() {
                        valid = false;
                        break;
                    }
                }
                (Some((BuiltinFn::Sinh, arg)), Sign::Neg) => {
                    if sinh_arg.replace(arg).is_some() {
                        valid = false;
                        break;
                    }
                }
                _ => {
                    valid = false;
                    break;
                }
            }
        }

        let (Some(cosh_arg), Some(sinh_arg)) = (cosh_arg, sinh_arg) else {
            continue;
        };
        if valid && compare_expr(ctx, cosh_arg, sinh_arg) == Ordering::Equal {
            return true;
        }
    }

    false
}

fn extract_direct_reciprocal_trig_product_one_arg_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() != 2 {
        return None;
    }

    for (plain_factor, reciprocal_factor) in [(factors[0], factors[1]), (factors[1], factors[0])] {
        let Some((plain_fn, plain_arg)) = extract_plain_sin_or_cos_arg_root(ctx, plain_factor)
        else {
            continue;
        };
        let Expr::Function(fn_id, args) = ctx.get(reciprocal_factor) else {
            continue;
        };
        let reciprocal_fn = if ctx.is_builtin(*fn_id, BuiltinFn::Csc) {
            BuiltinFn::Csc
        } else if ctx.is_builtin(*fn_id, BuiltinFn::Sec) {
            BuiltinFn::Sec
        } else if ctx.is_builtin(*fn_id, BuiltinFn::Cot) {
            BuiltinFn::Cot
        } else {
            continue;
        };
        if args.len() != 1 {
            continue;
        }
        if compare_expr(ctx, plain_arg, args[0]) != Ordering::Equal {
            continue;
        }

        let matches = matches!(
            (plain_fn, reciprocal_fn),
            (BuiltinFn::Sin, BuiltinFn::Csc)
                | (BuiltinFn::Cos, BuiltinFn::Sec)
                | (BuiltinFn::Tan, BuiltinFn::Cot)
        );
        if matches {
            return Some(plain_arg);
        }
    }

    None
}

fn matches_direct_reciprocal_trig_product_one_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (product_expr, target_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        if !matches!(ctx.get(target_expr), Expr::Number(n) if n.is_one()) {
            continue;
        }
        if extract_direct_reciprocal_trig_product_one_arg_root(ctx, product_expr).is_some() {
            return true;
        }
    }

    false
}

fn matches_direct_hyperbolic_triple_angle_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (source_expr, target_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(rewrite) = try_rewrite_hyperbolic_triple_angle(ctx, source_expr) else {
            continue;
        };
        if compare_expr(ctx, rewrite.rewritten, target_expr) == Ordering::Equal {
            return true;
        }
        if cas_ast::count_nodes(ctx, rewrite.rewritten) <= 32
            && cas_ast::count_nodes(ctx, target_expr) <= 32
            && isolated_simplify_rewrites_to_target(
                &crate::phase::SimplifyOptions::default(),
                ctx,
                rewrite.rewritten,
                target_expr,
            )
        {
            return true;
        }
    }

    false
}

fn extract_special_angle_exact_value_root(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }
    let builtin = ctx.builtin_of(*fn_id)?;
    if let Some(hit) = lookup_trig_or_inverse(ctx, builtin.name(), args[0]) {
        return Some(hit.value.to_expr(ctx));
    }
    if let Some(angle) = detect_special_angle(ctx, args[0]) {
        if let Some(value) = lookup_trig_value(builtin.name(), angle) {
            return Some(value.to_expr(ctx));
        }
    }
    try_rewrite_legacy_evaluate_trig_expr(ctx, expr).map(|rewrite| rewrite.rewritten)
}

fn ground_exact_constant_key_root(ctx: &Context, expr: ExprId) -> Option<String> {
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

fn matches_direct_special_angle_exact_value_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (source_expr, target_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(expected) = extract_special_angle_exact_value_root(ctx, source_expr) else {
            continue;
        };
        let normalized_expected = if cas_ast::count_nodes(ctx, expected) <= 20 {
            isolated_simplify_expr_if_changed(
                &crate::phase::SimplifyOptions::default(),
                ctx,
                expected,
            )
            .unwrap_or(expected)
        } else {
            expected
        };
        let normalized_target = if cas_ast::count_nodes(ctx, target_expr) <= 20 {
            isolated_simplify_expr_if_changed(
                &crate::phase::SimplifyOptions::default(),
                ctx,
                target_expr,
            )
            .unwrap_or(target_expr)
        } else {
            target_expr
        };

        if compare_expr(ctx, normalized_expected, normalized_target) == Ordering::Equal {
            return true;
        }
        if ground_exact_constant_key_root(ctx, normalized_expected).is_some()
            && ground_exact_constant_key_root(ctx, normalized_expected)
                == ground_exact_constant_key_root(ctx, normalized_target)
        {
            return true;
        }
        if cas_ast::count_nodes(ctx, normalized_expected) <= 20
            && cas_ast::count_nodes(ctx, normalized_target) <= 20
            && isolated_simplify_rewrites_to_target(
                &crate::phase::SimplifyOptions::default(),
                ctx,
                normalized_expected,
                normalized_target,
            )
        {
            return true;
        }
    }

    false
}

fn matches_direct_trig_inverse_composition_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (source_expr, target_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(rewrite) = try_rewrite_trig_inverse_composition_expr(ctx, source_expr) else {
            continue;
        };
        let rewritten = strip_multiplicative_one_root(ctx, rewrite.rewritten);
        let target = strip_multiplicative_one_root(ctx, target_expr);
        if compare_expr(ctx, rewritten, target) == Ordering::Equal {
            return true;
        }
        if cas_ast::count_nodes(ctx, rewritten) <= 24
            && cas_ast::count_nodes(ctx, target) <= 24
            && isolated_simplify_rewrites_to_target(
                &crate::phase::SimplifyOptions::default(),
                ctx,
                rewritten,
                target,
            )
        {
            return true;
        }
        let difference = ctx.add(Expr::Sub(rewritten, target));
        if cas_ast::count_nodes(ctx, difference) <= 36
            && isolated_simplify_rewrites_to_zero(
                &crate::phase::SimplifyOptions::default(),
                ctx,
                difference,
            )
        {
            return true;
        }
    }

    false
}

fn rewrite_direct_double_angle_inverse_trig_target_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let Some((BuiltinFn::Sin, doubled_arg)) = extract_plain_sin_or_cos_arg_root(ctx, expr) else {
        return None;
    };
    let inner_arg = extract_double_angle_arg_relaxed(ctx, doubled_arg)?;

    let sin_inner = ctx.call_builtin(BuiltinFn::Sin, vec![inner_arg]);
    let cos_inner = ctx.call_builtin(BuiltinFn::Cos, vec![inner_arg]);

    let rewritten_sin = if let Some(plan) =
        cas_math::inverse_trig_composition_support::try_plan_inverse_trig_composition_expr(
            ctx, sin_inner, false, false,
        ) {
        strip_multiplicative_one_root(ctx, plan.rewritten)
    } else if let Some(rewrite) = try_rewrite_trig_inverse_composition_expr(ctx, sin_inner) {
        strip_multiplicative_one_root(ctx, rewrite.rewritten)
    } else {
        return None;
    };

    let rewritten_cos = if let Some(plan) =
        cas_math::inverse_trig_composition_support::try_plan_inverse_trig_composition_expr(
            ctx, cos_inner, false, false,
        ) {
        strip_multiplicative_one_root(ctx, plan.rewritten)
    } else if let Some(rewrite) = try_rewrite_trig_inverse_composition_expr(ctx, cos_inner) {
        strip_multiplicative_one_root(ctx, rewrite.rewritten)
    } else {
        return None;
    };

    let two = ctx.num(2);
    Some(build_mul_expr_from_factors_root(
        ctx,
        &[two, rewritten_sin, rewritten_cos],
    ))
}

fn matches_direct_double_angle_inverse_trig_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (source_expr, target_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(rewritten) =
            rewrite_direct_double_angle_inverse_trig_target_root(ctx, source_expr)
        else {
            continue;
        };
        let rewritten = strip_multiplicative_one_root(ctx, rewritten);
        let target = strip_multiplicative_one_root(ctx, target_expr);
        if compare_expr(ctx, rewritten, target) == Ordering::Equal {
            return true;
        }
        if cas_ast::count_nodes(ctx, rewritten) <= 32
            && cas_ast::count_nodes(ctx, target) <= 32
            && isolated_simplify_rewrites_to_target(
                &crate::phase::SimplifyOptions::default(),
                ctx,
                rewritten,
                target,
            )
        {
            return true;
        }
        let difference = ctx.add(Expr::Sub(rewritten, target));
        if cas_ast::count_nodes(ctx, difference) <= 48
            && isolated_simplify_rewrites_to_zero(
                &crate::phase::SimplifyOptions::default(),
                ctx,
                difference,
            )
        {
            return true;
        }
    }

    false
}

fn matches_direct_weierstrass_contraction_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (source_expr, target_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(rewrite) = try_rewrite_weierstrass_contraction_div_expr(ctx, source_expr) else {
            continue;
        };
        let rewritten = strip_multiplicative_one_root(ctx, rewrite.rewritten);
        let target = strip_multiplicative_one_root(ctx, target_expr);
        if compare_expr(ctx, rewritten, target) == Ordering::Equal {
            return true;
        }
        if cas_ast::count_nodes(ctx, rewritten) <= 24
            && cas_ast::count_nodes(ctx, target) <= 24
            && isolated_simplify_rewrites_to_target(
                &crate::phase::SimplifyOptions::default(),
                ctx,
                rewritten,
                target,
            )
        {
            return true;
        }
    }

    false
}

fn matches_direct_pure_double_angle_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (double_angle_expr, product_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some((BuiltinFn::Sin, doubled_arg)) =
            extract_plain_sin_or_cos_arg_root(ctx, double_angle_expr)
        else {
            continue;
        };
        let Some(base_arg) = extract_double_angle_arg_relaxed(ctx, doubled_arg) else {
            continue;
        };
        let Some((sin_arg, cos_arg)) =
            extract_scaled_trig_sin_cos_product_args_root(ctx, product_expr)
        else {
            continue;
        };
        let canonical_cos_arg = canonicalize_even_cos_arg_root(ctx, cos_arg);
        if compare_expr(ctx, base_arg, sin_arg) == Ordering::Equal
            && compare_expr(ctx, base_arg, canonical_cos_arg) == Ordering::Equal
        {
            return true;
        }
    }

    false
}

fn matches_direct_hyperbolic_double_angle_sum_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (source_expr, target_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(rewrite) = try_rewrite_hyperbolic_double_angle_sum(ctx, source_expr) else {
            continue;
        };
        if compare_expr(ctx, rewrite.rewritten, target_expr) == Ordering::Equal {
            return true;
        }
    }

    false
}

fn matches_direct_hyperbolic_from_exp_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (source_expr, target_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(rewrite) = try_rewrite_recognize_hyperbolic_from_exp(ctx, source_expr) else {
            continue;
        };
        if compare_expr(ctx, rewrite.rewritten, target_expr) == Ordering::Equal {
            return true;
        }
    }

    false
}

fn matches_direct_tanh_to_sinh_cosh_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (source_expr, target_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(rewrite) = try_rewrite_tanh_to_sinh_cosh(ctx, source_expr) else {
            continue;
        };
        if compare_expr(ctx, rewrite, target_expr) == Ordering::Equal {
            return true;
        }
    }

    false
}

fn matches_direct_cube_root_rationalization_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (source_expr, target_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(rewrite) =
            cas_math::root_den_rationalize_support::try_rewrite_rationalize_cube_root_den_expr(
                ctx,
                source_expr,
            )
        else {
            continue;
        };
        let normalized_rewritten = strip_multiplicative_one_root(ctx, rewrite.rewritten);
        let normalized_target = strip_multiplicative_one_root(ctx, target_expr);
        if compare_expr(ctx, normalized_rewritten, normalized_target) == Ordering::Equal {
            return true;
        }
        if cas_ast::count_nodes(ctx, normalized_rewritten) <= 20
            && cas_ast::count_nodes(ctx, normalized_target) <= 20
            && isolated_simplify_rewrites_to_target(
                &crate::phase::SimplifyOptions::default(),
                ctx,
                normalized_rewritten,
                normalized_target,
            )
        {
            return true;
        }
        let difference = ctx.add(Expr::Sub(normalized_rewritten, normalized_target));
        if cas_ast::count_nodes(ctx, difference) <= 32
            && isolated_simplify_rewrites_to_zero(
                &crate::phase::SimplifyOptions::default(),
                ctx,
                difference,
            )
        {
            return true;
        }
    }

    false
}

fn matches_direct_tanh_pythagorean_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (identity_expr, target_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(identity_arg) =
            extract_direct_tanh_pythagorean_identity_arg_root(ctx, identity_expr)
        else {
            continue;
        };
        let Some(target_arg) = extract_direct_tanh_pythagorean_target_root(ctx, target_expr) else {
            continue;
        };
        if compare_expr(ctx, identity_arg, target_arg) == Ordering::Equal {
            return true;
        }
    }

    false
}

fn matches_direct_small_exact_constant_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (source_expr, target_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        if !matches!(ctx.get(target_expr), Expr::Number(_)) {
            continue;
        }
        if cas_ast::count_nodes(ctx, source_expr) > 16 {
            continue;
        }
        if isolated_simplify_rewrites_to_target(
            &crate::phase::SimplifyOptions::default(),
            ctx,
            source_expr,
            target_expr,
        ) {
            return true;
        }
    }

    false
}

fn matches_direct_trig_phase_shift_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    fn matches_cos_even_target(ctx: &mut Context, lhs: ExprId, rhs: ExprId) -> bool {
        let Some((BuiltinFn::Cos, lhs_arg)) = extract_plain_sin_or_cos_arg_root(ctx, lhs) else {
            return false;
        };
        let Some((BuiltinFn::Cos, rhs_arg)) = extract_plain_sin_or_cos_arg_root(ctx, rhs) else {
            return false;
        };
        let neg_lhs_arg = ctx.add(Expr::Neg(lhs_arg));
        let neg_rhs_arg = ctx.add(Expr::Neg(rhs_arg));

        compare_expr(ctx, lhs_arg, rhs_arg) == Ordering::Equal
            || compare_expr(ctx, neg_lhs_arg, rhs_arg) == Ordering::Equal
            || compare_expr(ctx, lhs_arg, neg_rhs_arg) == Ordering::Equal
    }

    fn matches_negated_cos_even_target(ctx: &mut Context, lhs: ExprId, rhs: ExprId) -> bool {
        let (Expr::Neg(lhs_inner), Expr::Neg(rhs_inner)) = (ctx.get(lhs), ctx.get(rhs)) else {
            return false;
        };
        matches_cos_even_target(ctx, *lhs_inner, *rhs_inner)
    }

    for (source_expr, target_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(rewrite) = try_rewrite_trig_phase_shift_function_expr(ctx, source_expr) else {
            continue;
        };
        if compare_expr(ctx, rewrite.rewritten, target_expr) == Ordering::Equal
            || matches_cos_even_target(ctx, rewrite.rewritten, target_expr)
            || matches_negated_cos_even_target(ctx, rewrite.rewritten, target_expr)
        {
            return true;
        }
    }

    false
}

fn matches_direct_trig_triple_angle_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (source_expr, target_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(rewrite) = try_rewrite_triple_angle_expr(ctx, source_expr) else {
            continue;
        };
        if compare_expr(ctx, rewrite.rewritten, target_expr) == Ordering::Equal {
            return true;
        }
    }

    false
}

fn matches_direct_perfect_square_trinomial_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (source_expr, target_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        if let Some(plan) =
            cas_math::expansion_rule_support::try_expand_binomial_pow_expr(ctx, source_expr, 2, 2)
        {
            if cas_math::poly_compare::poly_eq(ctx, plan.expanded, target_expr) {
                return true;
            }
        }
    }

    let lhs_minus_rhs = ctx.add(Expr::Sub(lhs_core, rhs_core));
    if matches_direct_perfect_square_trinomial_zero_identity_root(ctx, lhs_minus_rhs) {
        return true;
    }

    let rhs_minus_lhs = ctx.add(Expr::Sub(rhs_core, lhs_core));
    matches_direct_perfect_square_trinomial_zero_identity_root(ctx, rhs_minus_lhs)
}

fn extract_plus_or_minus_one_denominator_arg_root(
    ctx: &Context,
    expr: ExprId,
) -> Option<(ExprId, bool)> {
    match ctx.get(expr) {
        Expr::Sub(lhs, rhs) if matches!(ctx.get(*rhs), Expr::Number(n) if n.is_one()) => {
            Some((*lhs, false))
        }
        Expr::Add(lhs, rhs) if matches!(ctx.get(*rhs), Expr::Number(n) if n.is_one()) => {
            Some((*lhs, true))
        }
        Expr::Add(lhs, rhs) if matches!(ctx.get(*lhs), Expr::Number(n) if n.is_one()) => {
            Some((*rhs, true))
        }
        _ => None,
    }
}

fn extract_rational_plus_minus_one_sum_arg_root(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let terms = AddView::from_expr(ctx, expr).terms;
    if terms.len() != 2 || !terms.iter().all(|(_, sign)| *sign == Sign::Pos) {
        return None;
    }

    let mut minus_arg = None;
    let mut plus_arg = None;
    for (term_expr, _) in terms {
        let Expr::Div(num, den) = ctx.get(term_expr) else {
            return None;
        };
        if !matches!(ctx.get(*num), Expr::Number(n) if n.is_one()) {
            return None;
        }
        let (arg, is_plus) = extract_plus_or_minus_one_denominator_arg_root(ctx, *den)?;
        if is_plus {
            plus_arg = Some(arg);
        } else {
            minus_arg = Some(arg);
        }
    }

    let minus_arg = minus_arg?;
    let plus_arg = plus_arg?;
    (compare_expr(ctx, minus_arg, plus_arg) == Ordering::Equal).then_some(minus_arg)
}

fn matches_rational_plus_minus_one_target_root(
    ctx: &mut Context,
    expr: ExprId,
    arg: ExprId,
) -> bool {
    let two = ctx.num(2);
    let one = ctx.num(1);
    let numerator = smart_mul(ctx, two, arg);
    let squared = ctx.add(Expr::Pow(arg, two));
    let den_poly = ctx.add(Expr::Sub(squared, one));
    let minus_one_den = ctx.add(Expr::Sub(arg, one));
    let plus_one_den = ctx.add(Expr::Add(arg, one));
    let den_factored = ctx.add(Expr::Mul(minus_one_den, plus_one_den));
    let poly_target = ctx.add(Expr::Div(numerator, den_poly));
    let factored_target = ctx.add(Expr::Div(numerator, den_factored));

    compare_expr(ctx, expr, poly_target) == Ordering::Equal
        || compare_expr(ctx, expr, factored_target) == Ordering::Equal
}

fn matches_direct_rational_plus_minus_one_sum_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (sum_expr, rational_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(arg) = extract_rational_plus_minus_one_sum_arg_root(ctx, sum_expr) else {
            continue;
        };
        if matches_rational_plus_minus_one_target_root(ctx, rational_expr, arg) {
            return true;
        }
    }

    false
}

fn extract_plus_one_expr_target_root(ctx: &Context, expr: ExprId) -> Option<ExprId> {
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

fn extract_addition_of_successive_unit_fractions_arg_root(
    ctx: &Context,
    expr: ExprId,
) -> Option<ExprId> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 2 || !view.terms.iter().all(|(_, sign)| *sign == Sign::Pos) {
        return None;
    }

    let mut denominators = Vec::with_capacity(2);
    for (term_expr, _) in view.terms {
        let Expr::Div(num, den) = ctx.get(term_expr) else {
            return None;
        };
        if !matches!(ctx.get(*num), Expr::Number(n) if n.is_one()) {
            return None;
        }
        denominators.push(*den);
    }

    let first = denominators[0];
    let second = denominators[1];
    if let Some(base) = extract_plus_one_expr_target_root(ctx, first) {
        if compare_expr(ctx, base, second) == Ordering::Equal {
            return Some(second);
        }
    }
    if let Some(base) = extract_plus_one_expr_target_root(ctx, second) {
        if compare_expr(ctx, base, first) == Ordering::Equal {
            return Some(first);
        }
    }

    None
}

fn extract_collapsed_successive_unit_fractions_arg_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let (num, den) = match ctx.get(expr) {
        Expr::Div(num, den) => (*num, *den),
        _ => return None,
    };

    let one = ctx.num(1);
    let two = ctx.num(2);
    let view = AddView::from_expr(ctx, num);
    if view.terms.len() != 2 || !view.terms.iter().all(|(_, sign)| *sign == Sign::Pos) {
        return None;
    }

    let mut base = None;
    let mut saw_one = false;
    for (term_expr, _) in view.terms {
        if extract_i64_integer(ctx, term_expr) == Some(1) {
            if saw_one {
                return None;
            }
            saw_one = true;
            continue;
        }

        let Expr::Mul(lhs, rhs) = ctx.get(term_expr) else {
            return None;
        };
        let doubled_base = if extract_i64_integer(ctx, *lhs) == Some(2) {
            *rhs
        } else if extract_i64_integer(ctx, *rhs) == Some(2) {
            *lhs
        } else {
            return None;
        };
        if base.replace(doubled_base).is_some() {
            return None;
        }
    }

    let base = base?;
    if !saw_one {
        return None;
    }

    let plus_one = ctx.add(Expr::Add(base, one));
    let doubled_base = smart_mul(ctx, two, base);
    let expected_num = ctx.add(Expr::Add(doubled_base, one));
    let expected_den = ctx.add(Expr::Mul(base, plus_one));
    let squared_base = ctx.add(Expr::Pow(base, two));
    let expanded_den = ctx.add(Expr::Add(squared_base, base));
    if compare_expr(ctx, num, expected_num) == Ordering::Equal
        && (compare_expr(ctx, den, expected_den) == Ordering::Equal
            || compare_expr(ctx, den, expanded_den) == Ordering::Equal
            || matches_direct_linear_factoring_pair_root(ctx, den, expected_den))
    {
        return Some(base);
    }

    None
}

fn extract_consecutive_telescoping_fraction_difference_arg_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 2 {
        return None;
    }

    let mut positive_den = None;
    let mut negative_den = None;
    for (term_expr, sign) in view.terms {
        let denominator = extract_unit_fraction_denominator_root(ctx, term_expr)?;
        match sign {
            Sign::Pos => {
                if positive_den.replace(denominator).is_some() {
                    return None;
                }
            }
            Sign::Neg => {
                if negative_den.replace(denominator).is_some() {
                    return None;
                }
            }
        }
    }

    let positive_den = positive_den?;
    let negative_den = negative_den?;
    let one = ctx.num(1);
    let positive_plus_one = ctx.add(Expr::Add(positive_den, one));
    (compare_expr(ctx, positive_plus_one, negative_den) == Ordering::Equal).then_some(positive_den)
}

fn matches_direct_addition_of_successive_unit_fractions_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (sum_expr, collapsed_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(base) = extract_addition_of_successive_unit_fractions_arg_root(ctx, sum_expr)
        else {
            continue;
        };
        let Some(collapsed_base) =
            extract_collapsed_successive_unit_fractions_arg_root(ctx, collapsed_expr)
        else {
            continue;
        };
        if compare_expr(ctx, base, collapsed_base) == Ordering::Equal {
            return true;
        }
    }

    false
}

fn matches_direct_tanh_double_angle_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (source, target) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(rewritten) = try_rewrite_tanh_double_angle_expansion(ctx, source) else {
            continue;
        };
        if compare_expr(ctx, rewritten, target) == Ordering::Equal {
            return true;
        }
    }

    false
}

fn extract_sum_of_two_squared_atoms_root(ctx: &Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    let terms = AddView::from_expr(ctx, expr).terms;
    if terms.len() != 2 || !terms.iter().all(|(_, sign)| *sign == Sign::Pos) {
        return None;
    }

    let mut first_atom = None;
    let mut second_atom = None;
    for (index, (term_expr, _)) in terms.into_iter().enumerate() {
        let Expr::Pow(base, exp) = ctx.get(term_expr) else {
            return None;
        };
        if !matches!(ctx.get(*exp), Expr::Number(n) if n.is_integer() && n.to_integer() == 2.into())
        {
            return None;
        }
        if index == 0 {
            first_atom = Some(*base);
        } else {
            second_atom = Some(*base);
        }
    }
    Some((first_atom?, second_atom?))
}

fn build_sum_of_squares_product_target_root(
    ctx: &mut Context,
    p: ExprId,
    q: ExprId,
    r: ExprId,
    s: ExprId,
) -> ExprId {
    let first_product = smart_mul(ctx, p, r);
    let second_product = smart_mul(ctx, q, s);
    let first_sum = ctx.add(Expr::Add(first_product, second_product));
    let third_product = smart_mul(ctx, p, s);
    let fourth_product = smart_mul(ctx, q, r);
    let second_diff = ctx.add(Expr::Sub(third_product, fourth_product));
    let two = ctx.num(2);
    let first_square = ctx.add(Expr::Pow(first_sum, two));
    let second_square = ctx.add(Expr::Pow(second_diff, two));
    ctx.add(Expr::Add(first_square, second_square))
}

fn extract_squared_binomial_terms_root(ctx: &Context, expr: ExprId) -> Option<[(ExprId, Sign); 2]> {
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return None;
    };
    if !matches!(ctx.get(*exp), Expr::Number(n) if n.is_integer() && n.to_integer() == 2.into()) {
        return None;
    }

    let base = match ctx.get(*base) {
        Expr::Neg(inner) => *inner,
        _ => *base,
    };
    let terms = AddView::from_expr(ctx, base).terms;
    if terms.len() != 2 {
        return None;
    }
    Some([terms[0], terms[1]])
}

fn matches_squared_sum_pair_root(
    ctx: &mut Context,
    expr: ExprId,
    lhs: ExprId,
    rhs: ExprId,
) -> bool {
    let Some([(first_term, first_sign), (second_term, second_sign)]) =
        extract_squared_binomial_terms_root(ctx, expr)
    else {
        return false;
    };
    if first_sign != Sign::Pos || second_sign != Sign::Pos {
        return false;
    }
    matches_unordered_expr_pair_root(ctx, first_term, second_term, lhs, rhs)
}

fn matches_squared_difference_pair_root(
    ctx: &mut Context,
    expr: ExprId,
    lhs: ExprId,
    rhs: ExprId,
) -> bool {
    let Some([(first_term, first_sign), (second_term, second_sign)]) =
        extract_squared_binomial_terms_root(ctx, expr)
    else {
        return false;
    };
    if first_sign == second_sign {
        return false;
    }

    (first_sign == Sign::Pos
        && second_sign == Sign::Neg
        && ((compare_expr(ctx, first_term, lhs) == Ordering::Equal
            && compare_expr(ctx, second_term, rhs) == Ordering::Equal)
            || (compare_expr(ctx, first_term, rhs) == Ordering::Equal
                && compare_expr(ctx, second_term, lhs) == Ordering::Equal)))
        || (first_sign == Sign::Neg
            && second_sign == Sign::Pos
            && ((compare_expr(ctx, second_term, lhs) == Ordering::Equal
                && compare_expr(ctx, first_term, rhs) == Ordering::Equal)
                || (compare_expr(ctx, second_term, rhs) == Ordering::Equal
                    && compare_expr(ctx, first_term, lhs) == Ordering::Equal)))
}

fn rewrite_sum_of_squares_product_root(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Mul(left, right) = ctx.get(expr) else {
        return None;
    };
    let (p, q) = extract_sum_of_two_squared_atoms_root(ctx, *left)?;
    let (r, s) = extract_sum_of_two_squared_atoms_root(ctx, *right)?;
    Some(build_sum_of_squares_product_target_root(ctx, p, q, r, s))
}

fn matches_direct_sum_of_squares_product_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (product_expr, square_sum_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Expr::Mul(left, right) = ctx.get(product_expr) else {
            continue;
        };
        let Some((p, q)) = extract_sum_of_two_squared_atoms_root(ctx, *left) else {
            continue;
        };
        let Some((r, s)) = extract_sum_of_two_squared_atoms_root(ctx, *right) else {
            continue;
        };

        let sum_view = AddView::from_expr(ctx, square_sum_expr).terms;
        if sum_view.len() == 2 && sum_view.iter().all(|(_, sign)| *sign == Sign::Pos) {
            let pr = smart_mul(ctx, p, r);
            let qs = smart_mul(ctx, q, s);
            let ps = smart_mul(ctx, p, s);
            let qr = smart_mul(ctx, q, r);

            let square_terms = [sum_view[0].0, sum_view[1].0];
            if (matches_squared_sum_pair_root(ctx, square_terms[0], pr, qs)
                && matches_squared_difference_pair_root(ctx, square_terms[1], ps, qr))
                || (matches_squared_sum_pair_root(ctx, square_terms[1], pr, qs)
                    && matches_squared_difference_pair_root(ctx, square_terms[0], ps, qr))
            {
                return true;
            }
        }

        for (first_a, first_b) in [(p, q), (q, p)] {
            for (second_a, second_b) in [(r, s), (s, r)] {
                let expected = build_sum_of_squares_product_target_root(
                    ctx, first_a, first_b, second_a, second_b,
                );
                if compare_expr(ctx, square_sum_expr, expected) == Ordering::Equal {
                    return true;
                }
                if SemanticEqualityChecker::new(ctx).are_equal(square_sum_expr, expected) {
                    return true;
                }
                if cas_ast::count_nodes(ctx, expected) <= 48
                    && cas_ast::count_nodes(ctx, square_sum_expr) <= 48
                    && isolated_simplify_rewrites_to_target(
                        &crate::phase::SimplifyOptions::default(),
                        ctx,
                        expected,
                        square_sum_expr,
                    )
                {
                    return true;
                }
            }
        }
    }

    false
}
fn matches_direct_sum_diff_cubes_product_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    fn extract_sum_diff_cubes_compact_bases_root(
        ctx: &mut Context,
        expr: ExprId,
    ) -> Option<(ExprId, ExprId, bool)> {
        let view = AddView::from_expr(ctx, expr);
        if view.terms.len() != 2 {
            return None;
        }

        let mut positive_bases = Vec::with_capacity(2);
        let mut negative_bases = Vec::with_capacity(1);
        for (term_expr, term_sign) in view.terms {
            let base = extract_plain_cube_base_root(ctx, term_expr)?;
            match term_sign {
                Sign::Pos => positive_bases.push(base),
                Sign::Neg => negative_bases.push(base),
            }
        }

        match (&positive_bases[..], &negative_bases[..]) {
            ([a, b], []) => Some((*a, *b, false)),
            ([a], [b]) => Some((*a, *b, true)),
            _ => None,
        }
    }

    fn matches_sum_diff_cubes_binomial_root(
        ctx: &mut Context,
        expr: ExprId,
        lhs: ExprId,
        rhs: ExprId,
        is_difference: bool,
    ) -> bool {
        let view = AddView::from_expr(ctx, expr);
        if view.terms.len() != 2 {
            return false;
        }

        if is_difference {
            let mut positive = None;
            let mut negative = None;
            for (term, sign) in view.terms {
                match sign {
                    Sign::Pos => positive = Some(term),
                    Sign::Neg => negative = Some(term),
                }
            }
            return positive.is_some_and(|term| compare_expr(ctx, term, lhs) == Ordering::Equal)
                && negative.is_some_and(|term| compare_expr(ctx, term, rhs) == Ordering::Equal);
        }

        let lhs_matches = compare_expr(ctx, view.terms[0].0, lhs) == Ordering::Equal
            && compare_expr(ctx, view.terms[1].0, rhs) == Ordering::Equal;
        let rhs_matches = compare_expr(ctx, view.terms[0].0, rhs) == Ordering::Equal
            && compare_expr(ctx, view.terms[1].0, lhs) == Ordering::Equal;
        view.terms.iter().all(|(_, sign)| *sign == Sign::Pos) && (lhs_matches || rhs_matches)
    }

    fn matches_sum_diff_cubes_trinomial_root(
        ctx: &mut Context,
        expr: ExprId,
        lhs: ExprId,
        rhs: ExprId,
        is_difference: bool,
    ) -> bool {
        let terms = AddView::from_expr(ctx, expr).terms;
        if terms.len() != 3 {
            return false;
        }

        let two = ctx.num(2);
        let lhs_sq = ctx.add(Expr::Pow(lhs, two));
        let rhs_sq = ctx.add(Expr::Pow(rhs, two));
        let middle = smart_mul(ctx, lhs, rhs);
        let expected_middle_sign = if is_difference { Sign::Pos } else { Sign::Neg };

        let mut found_lhs_sq = false;
        let mut found_rhs_sq = false;
        let mut found_middle = false;
        for (term, sign) in terms {
            if sign == Sign::Pos
                && !found_lhs_sq
                && compare_expr(ctx, term, lhs_sq) == Ordering::Equal
            {
                found_lhs_sq = true;
                continue;
            }
            if sign == Sign::Pos
                && !found_rhs_sq
                && compare_expr(ctx, term, rhs_sq) == Ordering::Equal
            {
                found_rhs_sq = true;
                continue;
            }
            if sign == expected_middle_sign
                && !found_middle
                && compare_expr(ctx, term, middle) == Ordering::Equal
            {
                found_middle = true;
            }
        }

        found_lhs_sq && found_rhs_sq && found_middle
    }

    fn matches_sum_diff_cubes_product_from_compact_root(
        ctx: &mut Context,
        product_expr: ExprId,
        lhs: ExprId,
        rhs: ExprId,
        is_difference: bool,
    ) -> bool {
        let factors = flatten_mul_chain(ctx, product_expr);
        if factors.len() != 2 {
            return false;
        }

        [(factors[0], factors[1]), (factors[1], factors[0])]
            .into_iter()
            .any(|(binomial, trinomial)| {
                matches_sum_diff_cubes_binomial_root(ctx, binomial, lhs, rhs, is_difference)
                    && matches_sum_diff_cubes_trinomial_root(
                        ctx,
                        trinomial,
                        lhs,
                        rhs,
                        is_difference,
                    )
            })
    }

    for (product_expr, compact_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some((lhs, rhs, is_difference)) =
            extract_sum_diff_cubes_compact_bases_root(ctx, compact_expr)
        else {
            continue;
        };

        if matches_sum_diff_cubes_product_from_compact_root(
            ctx,
            product_expr,
            lhs,
            rhs,
            is_difference,
        ) {
            return true;
        }
    }

    false
}

fn matches_direct_higher_degree_difference_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    fn matches_geometric_quadratic_factor_root(
        ctx: &mut Context,
        expr: ExprId,
        base: ExprId,
        positive_linear: bool,
    ) -> bool {
        let terms = AddView::from_expr(ctx, expr).terms;
        if terms.len() != 3 {
            return false;
        }

        let two = ctx.num(2);
        let base_sq = ctx.add(Expr::Pow(base, two));
        let mut found_base_sq = false;
        let mut found_linear = false;
        let mut found_one = false;
        for (term, sign) in terms {
            if sign == Sign::Pos
                && !found_base_sq
                && compare_expr(ctx, term, base_sq) == Ordering::Equal
            {
                found_base_sq = true;
                continue;
            }
            if sign == Sign::Pos && !found_one && extract_i64_integer(ctx, term) == Some(1) {
                found_one = true;
                continue;
            }
            if sign
                == if positive_linear {
                    Sign::Pos
                } else {
                    Sign::Neg
                }
                && !found_linear
                && compare_expr(ctx, term, base) == Ordering::Equal
            {
                found_linear = true;
                continue;
            }
        }

        found_base_sq && found_linear && found_one
    }

    fn extract_base_from_sixth_power_minus_one_root(ctx: &Context, expr: ExprId) -> Option<ExprId> {
        let Expr::Sub(lhs, rhs) = ctx.get(expr) else {
            return None;
        };
        if extract_i64_integer(ctx, *rhs) != Some(1) {
            return None;
        }
        let Expr::Pow(base, exponent) = ctx.get(*lhs) else {
            return None;
        };
        (extract_i64_integer(ctx, *exponent) == Some(6)).then_some(*base)
    }

    for (compact_expr, product_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(base) = extract_base_from_sixth_power_minus_one_root(ctx, compact_expr) else {
            continue;
        };
        let product_factors = flatten_mul_chain(ctx, product_expr);
        if product_factors.len() != 4 {
            continue;
        }

        let mut saw_plus_one = false;
        let mut saw_minus_one = false;
        let mut saw_positive_quadratic = false;
        let mut saw_negative_quadratic = false;
        let mut invalid = false;
        for factor in product_factors {
            if let Some((factor_base, constant)) = extract_base_plus_constant_root(ctx, factor) {
                if compare_expr(ctx, factor_base, base) == Ordering::Equal {
                    if constant == BigRational::one() && !saw_plus_one {
                        saw_plus_one = true;
                        continue;
                    }
                    if constant == -BigRational::one() && !saw_minus_one {
                        saw_minus_one = true;
                        continue;
                    }
                }
            }
            if !saw_positive_quadratic
                && matches_geometric_quadratic_factor_root(ctx, factor, base, true)
            {
                saw_positive_quadratic = true;
                continue;
            }
            if !saw_negative_quadratic
                && matches_geometric_quadratic_factor_root(ctx, factor, base, false)
            {
                saw_negative_quadratic = true;
                continue;
            }
            invalid = true;
            break;
        }

        if !invalid
            && saw_plus_one
            && saw_minus_one
            && saw_positive_quadratic
            && saw_negative_quadratic
        {
            return true;
        }
    }

    false
}

fn matches_direct_sophie_germain_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    fn extract_sophie_germain_bases_root(
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

    fn matches_sophie_germain_quadratic_root(
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

        let mut found_a_sq = false;
        let mut found_two_b_sq = false;
        let mut found_two_ab = false;
        for (term, sign) in terms {
            if sign == Sign::Pos && !found_a_sq && compare_expr(ctx, term, a_sq) == Ordering::Equal
            {
                found_a_sq = true;
                continue;
            }
            if sign == Sign::Pos
                && !found_two_b_sq
                && compare_expr(ctx, term, two_b_sq) == Ordering::Equal
            {
                found_two_b_sq = true;
                continue;
            }
            if sign
                == if positive_linear {
                    Sign::Pos
                } else {
                    Sign::Neg
                }
                && !found_two_ab
                && compare_expr(ctx, term, two_ab) == Ordering::Equal
            {
                found_two_ab = true;
                continue;
            }
        }

        found_a_sq && found_two_b_sq && found_two_ab
    }

    for (compact_expr, product_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some((a, b)) = extract_sophie_germain_bases_root(ctx, compact_expr) else {
            continue;
        };
        let product_factors = flatten_mul_chain(ctx, product_expr);
        if product_factors.len() != 2 {
            continue;
        }

        if (matches_sophie_germain_quadratic_root(ctx, product_factors[0], a, b, true)
            && matches_sophie_germain_quadratic_root(ctx, product_factors[1], a, b, false))
            || (matches_sophie_germain_quadratic_root(ctx, product_factors[0], a, b, false)
                && matches_sophie_germain_quadratic_root(ctx, product_factors[1], a, b, true))
        {
            return true;
        }
    }

    false
}

fn matches_known_direct_pair_root(ctx: &mut Context, lhs_core: ExprId, rhs_core: ExprId) -> bool {
    matches_direct_addition_of_successive_unit_fractions_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_reciprocal_sqrt_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_exponential_combination_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_hyperbolic_exp_sum_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_normalized_trig_product_to_sum_sin_cos_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_trig_phase_shift_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_tangent_addition_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_tan_angle_sum_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_weierstrass_contraction_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_sum_to_product_contraction_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_trig_power_mixed_square_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_half_angle_square_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_scaled_half_angle_square_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_cos_fourth_power_reduction_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_hyperbolic_half_angle_square_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_trig_product_to_sum_sin_sin_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_trig_product_to_sum_sin_cos_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_trig_product_to_sum_cos_cos_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_pythagorean_identity_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_pythagorean_extended_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_nested_fraction_simplified_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_hyperbolic_sinh_sum_to_product_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_hyperbolic_cosh_sum_to_product_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_hyperbolic_cosh_difference_to_product_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_recursive_hyperbolic_sinh_sum_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_recursive_hyperbolic_cosh_sum_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_hyperbolic_double_angle_sum_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_pure_double_angle_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_double_angle_inverse_trig_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_trig_mixed_double_angle_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_quintuple_angle_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_trig_cubic_cosine_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_trig_binomial_square_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_cos_square_diff_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_abs_square_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_abs_trig_half_angle_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_positive_double_cos_square_diff_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_angle_sum_diff_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_small_pow_expansion_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_quartic_gcf_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_perfect_square_trinomial_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_linear_factoring_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_two_linear_shift_product_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_three_linear_shift_product_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_difference_of_squares_quotient_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_sum_diff_cubes_quotient_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_tan_to_sec_pythagorean_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_sec_tan_pythagorean_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_cot_to_csc_pythagorean_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_csc_cot_pythagorean_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_hyperbolic_pythagorean_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_reciprocal_trig_product_one_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_trig_triple_angle_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_hyperbolic_triple_angle_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_special_angle_exact_value_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_trig_inverse_composition_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_hyperbolic_from_exp_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_tanh_to_sinh_cosh_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_cube_root_rationalization_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_sum_diff_cubes_product_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_higher_degree_difference_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_sophie_germain_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_tanh_pythagorean_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_small_exact_constant_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_rational_plus_minus_one_sum_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_tanh_double_angle_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_sum_of_squares_product_pair_root(ctx, lhs_core, rhs_core)
}

fn extract_plain_trig_product_pair_args_root(
    ctx: &mut Context,
    expr: ExprId,
    trig_fn: BuiltinFn,
) -> Option<(ExprId, ExprId)> {
    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() != 2 {
        return None;
    }

    let lhs_arg = extract_unary_builtin_arg_root(ctx, factors[0], trig_fn)?;
    let rhs_arg = extract_unary_builtin_arg_root(ctx, factors[1], trig_fn)?;
    Some((lhs_arg, rhs_arg))
}

fn extract_plain_mixed_sin_cos_product_pair_args_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId)> {
    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() != 2 {
        return None;
    }

    let (lhs_fn, lhs_arg) = extract_plain_sin_or_cos_arg_root(ctx, factors[0])?;
    let (rhs_fn, rhs_arg) = extract_plain_sin_or_cos_arg_root(ctx, factors[1])?;
    match (lhs_fn, rhs_fn) {
        (BuiltinFn::Sin, BuiltinFn::Cos) => Some((lhs_arg, rhs_arg)),
        (BuiltinFn::Cos, BuiltinFn::Sin) => Some((rhs_arg, lhs_arg)),
        _ => None,
    }
}

fn matches_unordered_expr_pair_root(
    ctx: &Context,
    lhs_a: ExprId,
    lhs_b: ExprId,
    rhs_a: ExprId,
    rhs_b: ExprId,
) -> bool {
    (compare_expr(ctx, lhs_a, rhs_a) == Ordering::Equal
        && compare_expr(ctx, lhs_b, rhs_b) == Ordering::Equal)
        || (compare_expr(ctx, lhs_a, rhs_b) == Ordering::Equal
            && compare_expr(ctx, lhs_b, rhs_a) == Ordering::Equal)
}

fn matches_angle_sum_or_diff_arg_root(
    ctx: &mut Context,
    angle_arg: ExprId,
    lhs_arg: ExprId,
    rhs_arg: ExprId,
    is_sum: bool,
) -> bool {
    let direct_candidate = if is_sum {
        ctx.add(Expr::Add(lhs_arg, rhs_arg))
    } else {
        ctx.add(Expr::Sub(lhs_arg, rhs_arg))
    };
    if compare_expr(ctx, angle_arg, direct_candidate) == Ordering::Equal {
        return true;
    }

    if is_sum {
        let reversed_candidate = ctx.add(Expr::Add(rhs_arg, lhs_arg));
        if compare_expr(ctx, angle_arg, reversed_candidate) == Ordering::Equal {
            return true;
        }
    } else {
        let reversed_candidate = ctx.add(Expr::Sub(rhs_arg, lhs_arg));
        if compare_expr(ctx, angle_arg, reversed_candidate) == Ordering::Equal {
            return true;
        }
    }

    let Some((base, lhs_coeff, rhs_coeff)) = extract_linear_coefficients(ctx, lhs_arg, rhs_arg)
    else {
        return false;
    };
    let expected_coeff = if is_sum {
        lhs_coeff.clone() + rhs_coeff.clone()
    } else {
        lhs_coeff.clone() - rhs_coeff.clone()
    };
    let expected_arg = build_coef_times_base(ctx, &expected_coeff, base);
    if compare_expr(ctx, angle_arg, expected_arg) == Ordering::Equal {
        return true;
    }

    if is_sum {
        return false;
    }

    let reversed_coeff = rhs_coeff - lhs_coeff;
    let reversed_arg = build_coef_times_base(ctx, &reversed_coeff, base);
    compare_expr(ctx, angle_arg, reversed_arg) == Ordering::Equal
}

fn matches_direct_angle_sum_diff_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (angle_expr, product_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some((angle_fn, angle_arg)) = extract_plain_sin_or_cos_arg_root(ctx, angle_expr) else {
            continue;
        };

        let view = AddView::from_expr(ctx, product_expr);
        if view.terms.len() != 2 {
            continue;
        }

        if angle_fn == BuiltinFn::Sin {
            let mut first_pair = None;
            let mut first_sign = None;
            let mut second_pair = None;
            let mut second_sign = None;
            let mut bad_term = false;

            for (term_expr, term_sign) in view.terms {
                let Some(pair) = extract_plain_mixed_sin_cos_product_pair_args_root(ctx, term_expr)
                else {
                    bad_term = true;
                    break;
                };
                if first_pair.is_none() {
                    first_pair = Some(pair);
                    first_sign = Some(term_sign);
                } else if second_pair.is_none() {
                    second_pair = Some(pair);
                    second_sign = Some(term_sign);
                } else {
                    bad_term = true;
                    break;
                }
            }

            let (
                Some((first_sin_arg, first_cos_arg)),
                Some(first_sign),
                Some((second_sin_arg, second_cos_arg)),
                Some(second_sign),
            ) = (first_pair, first_sign, second_pair, second_sign)
            else {
                continue;
            };
            if bad_term
                || !matches_unordered_expr_pair_root(
                    ctx,
                    first_sin_arg,
                    first_cos_arg,
                    second_sin_arg,
                    second_cos_arg,
                )
            {
                continue;
            }

            let is_sum = match (first_sign, second_sign) {
                (Sign::Pos, Sign::Pos) => true,
                (Sign::Pos, Sign::Neg) | (Sign::Neg, Sign::Pos) => false,
                _ => continue,
            };
            if matches_angle_sum_or_diff_arg_root(
                ctx,
                angle_arg,
                first_sin_arg,
                first_cos_arg,
                is_sum,
            ) {
                return true;
            }

            continue;
        }

        if angle_fn != BuiltinFn::Cos {
            continue;
        }

        let mut cos_pair = None;
        let mut sin_pair = None;
        let mut sin_sign = None;
        let mut bad_term = false;

        for (term_expr, term_sign) in view.terms {
            if let Some(pair) =
                extract_plain_trig_product_pair_args_root(ctx, term_expr, BuiltinFn::Cos)
            {
                if cos_pair.is_some() || term_sign != Sign::Pos {
                    bad_term = true;
                    break;
                }
                cos_pair = Some(pair);
                continue;
            }

            if let Some(pair) =
                extract_plain_trig_product_pair_args_root(ctx, term_expr, BuiltinFn::Sin)
            {
                if sin_pair.is_some() {
                    bad_term = true;
                    break;
                }
                sin_pair = Some(pair);
                sin_sign = Some(term_sign);
                continue;
            }

            bad_term = true;
            break;
        }

        let (Some((cos_lhs, cos_rhs)), Some((sin_lhs, sin_rhs)), Some(sin_sign)) =
            (cos_pair, sin_pair, sin_sign)
        else {
            continue;
        };
        if bad_term || !matches_unordered_expr_pair_root(ctx, cos_lhs, cos_rhs, sin_lhs, sin_rhs) {
            continue;
        }

        let is_sum = sin_sign == Sign::Neg;
        if matches_angle_sum_or_diff_arg_root(ctx, angle_arg, cos_lhs, cos_rhs, is_sum) {
            return true;
        }
    }

    false
}

fn extract_direct_tangent_addition_target_root(
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

fn build_tangent_addition_fraction_root(
    ctx: &mut Context,
    lhs_arg: ExprId,
    rhs_arg: ExprId,
) -> ExprId {
    let sum_arg = ctx.add(Expr::Add(lhs_arg, rhs_arg));
    let numerator = ctx.call_builtin(BuiltinFn::Sin, vec![sum_arg]);
    let lhs_cos = ctx.call_builtin(BuiltinFn::Cos, vec![lhs_arg]);
    let rhs_cos = ctx.call_builtin(BuiltinFn::Cos, vec![rhs_arg]);
    let denominator = build_mul_expr_from_factors_root(ctx, &[lhs_cos, rhs_cos]);
    ctx.add(Expr::Div(numerator, denominator))
}

fn extract_direct_tan_angle_sum_target_root(
    ctx: &Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId)> {
    let sum_arg = extract_unary_builtin_arg_root(ctx, expr, BuiltinFn::Tan)?;
    let Expr::Add(lhs_arg, rhs_arg) = ctx.get(sum_arg) else {
        return None;
    };
    Some((*lhs_arg, *rhs_arg))
}

fn build_tan_angle_sum_fraction_root(
    ctx: &mut Context,
    lhs_arg: ExprId,
    rhs_arg: ExprId,
) -> ExprId {
    let tan_lhs = ctx.call_builtin(BuiltinFn::Tan, vec![lhs_arg]);
    let tan_rhs = ctx.call_builtin(BuiltinFn::Tan, vec![rhs_arg]);
    let numerator = ctx.add(Expr::Add(tan_lhs, tan_rhs));
    let one = ctx.num(1);
    let tan_product = build_mul_expr_from_factors_root(ctx, &[tan_lhs, tan_rhs]);
    let denominator = ctx.add(Expr::Sub(one, tan_product));
    ctx.add(Expr::Div(numerator, denominator))
}

fn extract_direct_tan_angle_sum_fraction_target_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId)> {
    let Expr::Div(numerator, denominator) = ctx.get(expr) else {
        return None;
    };
    let numerator = *numerator;
    let denominator = *denominator;
    let (num_lhs, num_rhs) = extract_direct_tangent_addition_target_root(ctx, numerator)?;

    let view = AddView::from_expr(ctx, denominator);
    if view.terms.len() != 2 {
        return None;
    }

    let mut saw_positive_one = false;
    let mut product_args = None;
    for (term_expr, term_sign) in view.terms {
        if extract_i64_integer(ctx, term_expr) == Some(1) {
            if term_sign != Sign::Pos || saw_positive_one {
                return None;
            }
            saw_positive_one = true;
            continue;
        }
        if term_sign != Sign::Neg || product_args.is_some() {
            return None;
        }
        product_args = extract_plain_trig_product_pair_args_root(ctx, term_expr, BuiltinFn::Tan);
        product_args?;
    }

    let (prod_lhs, prod_rhs) = product_args?;
    if !saw_positive_one
        || !matches_unordered_expr_pair_root(ctx, num_lhs, num_rhs, prod_lhs, prod_rhs)
    {
        return None;
    }

    Some((num_lhs, num_rhs))
}

fn extract_direct_tangent_addition_fraction_target_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId)> {
    let Expr::Div(numerator, denominator) = ctx.get(expr) else {
        return None;
    };
    let numerator = *numerator;
    let denominator = *denominator;
    let Some((BuiltinFn::Sin, sum_arg)) = extract_plain_sin_or_cos_arg_root(ctx, numerator) else {
        return None;
    };
    let Expr::Add(sum_lhs, sum_rhs) = ctx.get(sum_arg) else {
        return None;
    };
    let sum_lhs = *sum_lhs;
    let sum_rhs = *sum_rhs;
    let (den_lhs, den_rhs) =
        extract_plain_trig_product_pair_args_root(ctx, denominator, BuiltinFn::Cos)?;
    matches_unordered_expr_pair_root(ctx, sum_lhs, sum_rhs, den_lhs, den_rhs)
        .then_some((sum_lhs, sum_rhs))
}

fn matches_direct_tangent_addition_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (tan_sum_expr, fraction_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some((lhs_arg, rhs_arg)) =
            extract_direct_tangent_addition_target_root(ctx, tan_sum_expr)
        else {
            continue;
        };
        let Some((fraction_lhs, fraction_rhs)) =
            extract_direct_tangent_addition_fraction_target_root(ctx, fraction_expr)
        else {
            continue;
        };
        if matches_unordered_expr_pair_root(ctx, lhs_arg, rhs_arg, fraction_lhs, fraction_rhs) {
            return true;
        }
    }

    false
}

fn matches_direct_tan_angle_sum_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (tan_expr, fraction_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some((tan_lhs, tan_rhs)) = extract_direct_tan_angle_sum_target_root(ctx, tan_expr)
        else {
            continue;
        };
        let Some((fraction_lhs, fraction_rhs)) =
            extract_direct_tan_angle_sum_fraction_target_root(ctx, fraction_expr)
        else {
            continue;
        };
        if matches_unordered_expr_pair_root(ctx, tan_lhs, tan_rhs, fraction_lhs, fraction_rhs) {
            return true;
        }
    }

    false
}

fn matches_direct_recursive_hyperbolic_sinh_sum_zero_identity_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 3 {
        return false;
    }

    for single_index in 0..view.terms.len() {
        let (single_expr, single_sign) = view.terms[single_index];
        if single_sign != Sign::Pos {
            continue;
        }
        let Some((BuiltinFn::Sinh, _)) = extract_plain_sinh_or_cosh_arg_root(ctx, single_expr)
        else {
            continue;
        };

        let expanded_terms: smallvec::SmallVec<[(ExprId, Sign); 2]> = view
            .terms
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(index, (term_expr, term_sign))| {
                (index != single_index).then_some((term_expr, term_sign))
            })
            .collect();
        if expanded_terms.len() != 2 || expanded_terms.iter().any(|(_, sign)| *sign != Sign::Neg) {
            continue;
        }

        let expanded_expr = build_signed_sum_expr_root(
            ctx,
            &[
                (expanded_terms[0].0, Sign::Pos),
                (expanded_terms[1].0, Sign::Pos),
            ],
        );
        if matches_direct_recursive_hyperbolic_sinh_sum_pair_root(ctx, single_expr, expanded_expr) {
            return true;
        }
    }

    false
}

fn matches_direct_recursive_hyperbolic_cosh_sum_zero_identity_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 3 {
        return false;
    }

    for single_index in 0..view.terms.len() {
        let (single_expr, single_sign) = view.terms[single_index];
        if single_sign != Sign::Pos {
            continue;
        }
        let Some((BuiltinFn::Cosh, _)) = extract_plain_sinh_or_cosh_arg_root(ctx, single_expr)
        else {
            continue;
        };

        let expanded_terms: smallvec::SmallVec<[(ExprId, Sign); 2]> = view
            .terms
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(index, (term_expr, term_sign))| {
                (index != single_index).then_some((term_expr, term_sign))
            })
            .collect();
        if expanded_terms.len() != 2 || expanded_terms.iter().any(|(_, sign)| *sign != Sign::Neg) {
            continue;
        }

        let expanded_expr = build_signed_sum_expr_root(
            ctx,
            &[
                (expanded_terms[0].0, Sign::Pos),
                (expanded_terms[1].0, Sign::Pos),
            ],
        );
        if matches_direct_recursive_hyperbolic_cosh_sum_pair_root(ctx, single_expr, expanded_expr) {
            return true;
        }
    }

    false
}

fn matches_direct_hyperbolic_exp_sum_zero_identity_root(ctx: &mut Context, expr: ExprId) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 3 {
        return false;
    }

    let mut cosh_arg = None;
    let mut sinh_arg = None;
    let mut exp_arg = None;

    for (term_expr, term_sign) in view.terms {
        if term_sign == Sign::Pos {
            match extract_plain_sinh_or_cosh_arg_root(ctx, term_expr) {
                Some((BuiltinFn::Cosh, arg)) => cosh_arg = Some(arg),
                Some((BuiltinFn::Sinh, arg)) => sinh_arg = Some(arg),
                _ => return false,
            }
            continue;
        }

        if term_sign == Sign::Neg {
            if let Some(arg) = extract_exp_argument(ctx, term_expr) {
                exp_arg = Some(arg);
                continue;
            }
        }

        return false;
    }

    match (cosh_arg, sinh_arg, exp_arg) {
        (Some(cosh_arg), Some(sinh_arg), Some(exp_arg)) => {
            compare_expr(ctx, cosh_arg, sinh_arg) == Ordering::Equal
                && compare_expr(ctx, cosh_arg, exp_arg) == Ordering::Equal
        }
        _ => false,
    }
}

fn matches_direct_hyperbolic_pythagorean_zero_identity_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 3 {
        return false;
    }

    let mut cosh_pos = None;
    let mut cosh_neg = None;
    let mut sinh_pos = None;
    let mut sinh_neg = None;
    let mut saw_pos_one = false;
    let mut saw_neg_one = false;

    for (term_expr, term_sign) in view.terms {
        if extract_i64_integer(ctx, term_expr)
            .is_some_and(|value| matches!((value, term_sign), (1, Sign::Pos) | (-1, Sign::Neg)))
        {
            saw_pos_one = true;
            continue;
        }
        if extract_i64_integer(ctx, term_expr)
            .is_some_and(|value| matches!((value, term_sign), (1, Sign::Neg) | (-1, Sign::Pos)))
        {
            saw_neg_one = true;
            continue;
        }

        match extract_plain_sinh_or_cosh_pow2_arg_root(ctx, term_expr) {
            Some((BuiltinFn::Cosh, arg)) if term_sign == Sign::Pos => cosh_pos = Some(arg),
            Some((BuiltinFn::Cosh, arg)) if term_sign == Sign::Neg => cosh_neg = Some(arg),
            Some((BuiltinFn::Sinh, arg)) if term_sign == Sign::Pos => sinh_pos = Some(arg),
            Some((BuiltinFn::Sinh, arg)) if term_sign == Sign::Neg => sinh_neg = Some(arg),
            _ => return false,
        }
    }

    match (
        cosh_pos,
        sinh_neg,
        saw_neg_one,
        cosh_neg,
        sinh_pos,
        saw_pos_one,
    ) {
        (Some(cosh_arg), Some(sinh_arg), true, _, _, _) => {
            compare_expr(ctx, cosh_arg, sinh_arg) == Ordering::Equal
        }
        (_, _, _, Some(cosh_arg), Some(sinh_arg), true) => {
            compare_expr(ctx, cosh_arg, sinh_arg) == Ordering::Equal
        }
        _ => false,
    }
}

fn extract_scaled_cos_double_angle_sine_term_arg_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let factors = flatten_mul_chain(ctx, expr);
    let mut numeric_coeff = BigRational::one();
    let mut sin_arg = None;
    let mut cos_arg = None;

    for factor in factors {
        if let Expr::Number(n) = ctx.get(factor) {
            numeric_coeff *= n.clone();
            continue;
        }
        match extract_plain_sin_or_cos_arg_root(ctx, factor) {
            Some((BuiltinFn::Sin, arg)) => sin_arg = Some(arg),
            Some((BuiltinFn::Cos, arg)) => cos_arg = Some(arg),
            _ => return None,
        }
    }

    if numeric_coeff != BigRational::from_integer(2.into()) {
        return None;
    }
    let sin_arg = sin_arg?;
    let cos_arg = cos_arg?;
    let two = ctx.num(2);
    let doubled_sin_arg = smart_mul(ctx, two, sin_arg);
    (compare_expr(ctx, cos_arg, doubled_sin_arg) == Ordering::Equal).then_some(sin_arg)
}

fn extract_scaled_plain_sine_term_arg_root(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
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

fn extract_scaled_cos_square_sine_term_arg_root(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let factors = flatten_mul_chain(ctx, expr);
    let mut numeric_coeff = BigRational::one();
    let mut sin_arg = None;
    let mut cos_arg = None;

    for factor in factors {
        if let Expr::Number(n) = ctx.get(factor) {
            numeric_coeff *= n.clone();
            continue;
        }
        if let Some((BuiltinFn::Sin, arg)) = extract_plain_sin_or_cos_arg_root(ctx, factor) {
            if sin_arg.is_some() {
                return None;
            }
            sin_arg = Some(arg);
            continue;
        }

        let Expr::Pow(base, exponent) = ctx.get(factor) else {
            return None;
        };
        let Expr::Number(n) = ctx.get(*exponent) else {
            return None;
        };
        if *n != BigRational::from_integer(2.into()) {
            return None;
        }
        let Some((BuiltinFn::Cos, arg)) = extract_plain_sin_or_cos_arg_root(ctx, *base) else {
            return None;
        };
        if cos_arg.is_some() {
            return None;
        }
        cos_arg = Some(arg);
    }

    let (Some(sin_arg), Some(cos_arg)) = (sin_arg, cos_arg) else {
        return None;
    };
    (numeric_coeff == BigRational::from_integer(4.into())
        && compare_expr(ctx, sin_arg, cos_arg) == Ordering::Equal)
        .then_some(sin_arg)
}

fn extract_scaled_sin_double_angle_sine_term_arg_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let factors = flatten_mul_chain(ctx, expr);
    let mut numeric_coeff = BigRational::one();
    let mut sin_args: smallvec::SmallVec<[ExprId; 2]> = smallvec::SmallVec::new();

    for factor in factors {
        if let Expr::Number(n) = ctx.get(factor) {
            numeric_coeff *= n.clone();
            continue;
        }
        let Some((BuiltinFn::Sin, arg)) = extract_plain_sin_or_cos_arg_root(ctx, factor) else {
            return None;
        };
        sin_args.push(arg);
    }

    if numeric_coeff != BigRational::from_integer(2.into()) || sin_args.len() != 2 {
        return None;
    }

    for &candidate_u in &sin_args {
        let two = ctx.num(2);
        let doubled_candidate_u = smart_mul(ctx, two, candidate_u);
        if sin_args
            .iter()
            .any(|arg| compare_expr(ctx, *arg, doubled_candidate_u) == Ordering::Equal)
        {
            return Some(candidate_u);
        }
    }

    None
}

fn extract_scaled_plain_cosh_term_arg_root(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let factors = flatten_mul_chain(ctx, expr);
    let mut numeric_coeff = BigRational::one();
    let mut cosh_arg = None;

    for factor in factors {
        if let Expr::Number(n) = ctx.get(factor) {
            numeric_coeff *= n.clone();
            continue;
        }
        let Some((BuiltinFn::Cosh, arg)) = extract_plain_sinh_or_cosh_arg_root(ctx, factor) else {
            return None;
        };
        if cosh_arg.is_some() {
            return None;
        }
        cosh_arg = Some(arg);
    }

    (numeric_coeff == BigRational::from_integer(4.into()))
        .then_some(cosh_arg)
        .flatten()
}

fn extract_scaled_cosh_cubic_term_arg_root(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let factors = flatten_mul_chain(ctx, expr);
    let mut numeric_coeff = BigRational::one();
    let mut cosh_arg = None;

    for factor in factors {
        if let Expr::Number(n) = ctx.get(factor) {
            numeric_coeff *= n.clone();
            continue;
        }

        let Expr::Pow(base, exponent) = ctx.get(factor) else {
            return None;
        };
        if extract_i64_integer(ctx, *exponent)? != 3 {
            return None;
        }
        let Some((BuiltinFn::Cosh, arg)) = extract_plain_sinh_or_cosh_arg_root(ctx, *base) else {
            return None;
        };
        if cosh_arg.is_some() {
            return None;
        }
        cosh_arg = Some(arg);
    }

    (numeric_coeff == BigRational::from_integer(4.into()))
        .then_some(cosh_arg)
        .flatten()
}

fn extract_scaled_sinh_double_angle_sinh_term_arg_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let factors = flatten_mul_chain(ctx, expr);
    let mut numeric_coeff = BigRational::one();
    let mut sinh_args: smallvec::SmallVec<[ExprId; 2]> = smallvec::SmallVec::new();

    for factor in factors {
        if let Expr::Number(n) = ctx.get(factor) {
            numeric_coeff *= n.clone();
            continue;
        }
        let Some((BuiltinFn::Sinh, arg)) = extract_plain_sinh_or_cosh_arg_root(ctx, factor) else {
            return None;
        };
        sinh_args.push(arg);
    }

    if numeric_coeff != BigRational::from_integer(2.into()) || sinh_args.len() != 2 {
        return None;
    }

    for &candidate_u in &sinh_args {
        let two = ctx.num(2);
        let doubled_candidate_u = smart_mul(ctx, two, candidate_u);
        if sinh_args
            .iter()
            .any(|arg| compare_expr(ctx, *arg, doubled_candidate_u) == Ordering::Equal)
        {
            return Some(candidate_u);
        }
    }

    None
}

fn matches_narrow_trig_mixed_double_angle_zero_candidate_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    view.terms.len() == 3
        && view.terms.iter().any(|(term_expr, _)| {
            extract_scaled_cos_double_angle_sine_term_arg_root(ctx, *term_expr).is_some()
        })
}

fn matches_direct_trig_mixed_double_angle_zero_identity_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 3 {
        return false;
    }

    let mut double_angle = None;
    let mut linear_sine = None;
    let mut cos_square_sine = None;

    for (term_expr, term_sign) in view.terms {
        if let Some(arg) = extract_scaled_cos_double_angle_sine_term_arg_root(ctx, term_expr) {
            if double_angle.is_some() {
                return false;
            }
            double_angle = Some((arg, term_sign));
            continue;
        }

        if let Some(arg) = extract_scaled_plain_sine_term_arg_root(ctx, term_expr) {
            if linear_sine.is_some() {
                return false;
            }
            linear_sine = Some((arg, term_sign));
            continue;
        }

        if let Some(arg) = extract_scaled_cos_square_sine_term_arg_root(ctx, term_expr) {
            if cos_square_sine.is_some() {
                return false;
            }
            cos_square_sine = Some((arg, term_sign));
            continue;
        }

        return false;
    }

    let (
        Some((double_arg, double_sign)),
        Some((linear_arg, linear_sign)),
        Some((square_arg, square_sign)),
    ) = (double_angle, linear_sine, cos_square_sine)
    else {
        return false;
    };

    compare_expr(ctx, double_arg, linear_arg) == Ordering::Equal
        && compare_expr(ctx, double_arg, square_arg) == Ordering::Equal
        && double_sign == linear_sign
        && square_sign == double_sign.negate()
}

fn matches_direct_trig_cubic_cosine_zero_identity_root(ctx: &mut Context, expr: ExprId) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 3 {
        return false;
    }

    let mut sin_product_arg = None;
    let mut cos_linear = None;
    let mut cos_cubic = None;

    for (term_expr, term_sign) in view.terms {
        if let Some(arg) = extract_scaled_sin_double_angle_sine_term_arg_root(ctx, term_expr) {
            if sin_product_arg.is_some() {
                return false;
            }
            sin_product_arg = Some(arg);
            continue;
        }

        let (mut coeff, base) = extract_coef_and_base(ctx, term_expr);
        if term_sign == Sign::Neg {
            coeff = -coeff;
        }

        if let Some((BuiltinFn::Cos, arg)) = extract_plain_sin_or_cos_arg_root(ctx, base) {
            if cos_linear.is_some() {
                return false;
            }
            cos_linear = Some((coeff, arg));
            continue;
        }

        let Expr::Pow(power_base, exponent) = ctx.get(base) else {
            return false;
        };
        let Expr::Number(n) = ctx.get(*exponent) else {
            return false;
        };
        if *n != BigRational::from_integer(3.into()) {
            return false;
        }
        let Some((BuiltinFn::Cos, arg)) = extract_plain_sin_or_cos_arg_root(ctx, *power_base)
        else {
            return false;
        };
        if cos_cubic.is_some() {
            return false;
        }
        cos_cubic = Some((coeff, arg));
    }

    let (
        Some(sin_product_arg),
        Some((cos_linear_coeff, cos_linear_arg)),
        Some((cos_cubic_coeff, cos_cubic_arg)),
    ) = (sin_product_arg, cos_linear, cos_cubic)
    else {
        return false;
    };

    compare_expr(ctx, sin_product_arg, cos_linear_arg) == Ordering::Equal
        && compare_expr(ctx, sin_product_arg, cos_cubic_arg) == Ordering::Equal
        && cos_linear_coeff == BigRational::from_integer((-4).into())
        && cos_cubic_coeff == BigRational::from_integer(4.into())
}

fn extract_unit_fraction_denominator_root(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Div(numerator, denominator) = ctx.get(expr) else {
        return None;
    };
    let numerator = *numerator;
    let denominator = *denominator;
    let one = ctx.num(1);
    (compare_expr(ctx, numerator, one) == Ordering::Equal).then_some(denominator)
}

fn extract_consecutive_product_core_root(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
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

fn matches_direct_consecutive_telescoping_fraction_zero_identity_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 3 {
        return false;
    }

    let mut product_core = None;
    let mut product_sign = None;
    let mut single_terms: smallvec::SmallVec<[(ExprId, Sign); 2]> = smallvec::SmallVec::new();

    for (term_expr, term_sign) in view.terms {
        let Some(denominator) = extract_unit_fraction_denominator_root(ctx, term_expr) else {
            return false;
        };

        if let Some(candidate_u) = extract_consecutive_product_core_root(ctx, denominator) {
            if product_core.is_some() {
                return false;
            }
            product_core = Some(candidate_u);
            product_sign = Some(term_sign);
        } else {
            single_terms.push((denominator, term_sign));
        }
    }

    let (u, product_sign) = match (product_core, product_sign) {
        (Some(u), Some(sign)) => (u, sign),
        _ => return false,
    };
    if single_terms.len() != 2 {
        return false;
    }

    let one = ctx.num(1);
    let u_plus_one = ctx.add(Expr::Add(u, one));
    let mut saw_u = false;
    let mut saw_u_plus_one = false;

    for (denominator, sign) in single_terms {
        if compare_expr(ctx, denominator, u) == Ordering::Equal {
            if sign == product_sign {
                return false;
            }
            saw_u = true;
            continue;
        }
        if compare_expr(ctx, denominator, u_plus_one) == Ordering::Equal {
            if sign != product_sign {
                return false;
            }
            saw_u_plus_one = true;
            continue;
        }
        return false;
    }

    saw_u && saw_u_plus_one
}

fn build_square_preserving_one_root(ctx: &mut Context, expr: ExprId) -> ExprId {
    if extract_i64_integer(ctx, expr) == Some(1) {
        expr
    } else {
        let two = ctx.num(2);
        ctx.add(Expr::Pow(expr, two))
    }
}

fn numerator_matches_two_times_shift_root(
    ctx: &mut Context,
    numerator: ExprId,
    shift: ExprId,
) -> bool {
    if extract_i64_integer(ctx, numerator) == Some(2) && extract_i64_integer(ctx, shift) == Some(1)
    {
        return true;
    }

    let factors = flatten_mul_chain(ctx, numerator);
    if factors.len() != 2 {
        return false;
    }

    let two = ctx.num(2);
    (compare_expr(ctx, factors[0], two) == Ordering::Equal
        && compare_expr(ctx, factors[1], shift) == Ordering::Equal)
        || (compare_expr(ctx, factors[1], two) == Ordering::Equal
            && compare_expr(ctx, factors[0], shift) == Ordering::Equal)
}

fn positive_two_term_sum_matches_terms_root(
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

fn matches_direct_symmetric_partial_fraction_zero_identity_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 3 || !expr_contains_division_node_local(ctx, expr) {
        return false;
    }

    let positive_terms: smallvec::SmallVec<[(ExprId, Sign); 1]> = view
        .terms
        .iter()
        .copied()
        .filter(|(_, sign)| *sign == Sign::Pos)
        .collect();
    let negative_terms: smallvec::SmallVec<[(ExprId, Sign); 2]> = view
        .terms
        .iter()
        .copied()
        .filter(|(_, sign)| *sign == Sign::Neg)
        .collect();
    if positive_terms.len() != 1 || negative_terms.len() != 2 {
        return false;
    }

    let Some(positive_denominator) =
        extract_unit_fraction_denominator_root(ctx, positive_terms[0].0)
    else {
        return false;
    };
    let Expr::Sub(base, shift) = ctx.get(positive_denominator).clone() else {
        return false;
    };

    for (negative_unit, target_fraction) in [
        (negative_terms[0].0, negative_terms[1].0),
        (negative_terms[1].0, negative_terms[0].0),
    ] {
        let Some(negative_denominator) = extract_unit_fraction_denominator_root(ctx, negative_unit)
        else {
            continue;
        };
        if !positive_two_term_sum_matches_terms_root(ctx, negative_denominator, base, shift) {
            continue;
        }

        let Expr::Div(target_numerator, target_denominator) = ctx.get(target_fraction).clone()
        else {
            continue;
        };
        if !numerator_matches_two_times_shift_root(ctx, target_numerator, shift) {
            continue;
        }

        let two = ctx.num(2);
        let base_squared = ctx.add(Expr::Pow(base, two));
        let shift_squared = build_square_preserving_one_root(ctx, shift);
        let expected_denominator = ctx.add(Expr::Sub(base_squared, shift_squared));
        if compare_expr(ctx, target_denominator, expected_denominator) != Ordering::Equal {
            continue;
        }

        return true;
    }

    false
}

fn matches_direct_small_rational_zero_identity_root(ctx: &mut Context, expr: ExprId) -> bool {
    if matches_direct_symmetric_partial_fraction_zero_identity_root(ctx, expr) {
        return true;
    }

    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 3
        || !expr_contains_division_node_local(ctx, expr)
        || expr_contains_trig_or_hyperbolic_builtin_local(ctx, expr)
    {
        return false;
    }

    let parent_ctx = crate::ParentContext::root().with_domain_mode(crate::DomainMode::Generic);
    let rule = crate::rules::arithmetic::CollapseExactZeroThreeTermSubsetRule;
    let Some(rewrite) = crate::rule::Rule::apply(&rule, ctx, expr, &parent_ctx) else {
        return false;
    };
    let zero = ctx.num(0);
    compare_expr(ctx, rewrite.final_expr(), zero) == Ordering::Equal
}

fn matches_direct_exp_hyperbolic_double_identity_root(ctx: &mut Context, expr: ExprId) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 3 {
        return false;
    }

    let mut exp_terms: smallvec::SmallVec<[(BigRational, ExprId); 2]> = smallvec::SmallVec::new();
    let mut hyper_term: Option<(BigRational, BuiltinFn, ExprId)> = None;

    for (term_expr, term_sign) in view.terms {
        let (mut coeff, base) = extract_coef_and_base(ctx, term_expr);
        if term_sign == Sign::Neg {
            coeff = -coeff;
        }

        if let Some(arg) = extract_exp_argument(ctx, base) {
            exp_terms.push((coeff, arg));
            continue;
        }

        match extract_plain_sinh_or_cosh_arg_root(ctx, base) {
            Some((BuiltinFn::Cosh, arg)) => hyper_term = Some((coeff, BuiltinFn::Cosh, arg)),
            Some((BuiltinFn::Sinh, arg)) => hyper_term = Some((coeff, BuiltinFn::Sinh, arg)),
            _ => return false,
        }
    }

    let Some((hyper_coeff, hyper_fn, hyper_arg)) = hyper_term else {
        return false;
    };
    if exp_terms.len() != 2 {
        return false;
    }

    let two = BigRational::from_integer(2.into());
    for first_index in 0..exp_terms.len() {
        let (first_coeff, first_arg) = &exp_terms[first_index];
        if compare_expr(ctx, *first_arg, hyper_arg) != Ordering::Equal {
            continue;
        }
        let second_index = 1 - first_index;
        let (second_coeff, second_arg) = &exp_terms[second_index];
        let neg_hyper_arg = ctx.add(Expr::Neg(hyper_arg));
        if compare_expr(ctx, *second_arg, neg_hyper_arg) != Ordering::Equal {
            continue;
        }

        let matches_cosh = hyper_fn == BuiltinFn::Cosh
            && *first_coeff == *second_coeff
            && hyper_coeff == -(&two * first_coeff);
        let matches_sinh = hyper_fn == BuiltinFn::Sinh
            && *second_coeff == -first_coeff.clone()
            && hyper_coeff == -(&two * first_coeff);
        if matches_cosh || matches_sinh {
            return true;
        }
    }

    false
}

fn matches_direct_hyperbolic_cosh_cubic_zero_identity_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 3 {
        return false;
    }

    let mut double_angle = None;
    let mut linear_cosh = None;
    let mut cubic_cosh = None;

    for (term_expr, term_sign) in view.terms {
        if let Some(arg) = extract_scaled_sinh_double_angle_sinh_term_arg_root(ctx, term_expr) {
            if double_angle.is_some() || term_sign != Sign::Pos {
                return false;
            }
            double_angle = Some(arg);
            continue;
        }

        if let Some(arg) = extract_scaled_plain_cosh_term_arg_root(ctx, term_expr) {
            if linear_cosh.is_some() || term_sign != Sign::Pos {
                return false;
            }
            linear_cosh = Some(arg);
            continue;
        }

        if let Some(arg) = extract_scaled_cosh_cubic_term_arg_root(ctx, term_expr) {
            if cubic_cosh.is_some() || term_sign != Sign::Neg {
                return false;
            }
            cubic_cosh = Some(arg);
            continue;
        }

        return false;
    }

    match (double_angle, linear_cosh, cubic_cosh) {
        (Some(double_arg), Some(linear_arg), Some(cubic_arg)) => {
            compare_expr(ctx, double_arg, linear_arg) == Ordering::Equal
                && compare_expr(ctx, double_arg, cubic_arg) == Ordering::Equal
        }
        _ => false,
    }
}

fn matches_direct_general_phase_shift_zero_identity_root(ctx: &mut Context, expr: ExprId) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 3 || !expr_contains_trig_builtin_local(ctx, expr) {
        return false;
    }

    let extract_shifted_base_arg = |ctx: &mut Context, expr: ExprId| {
        let (_trig_fn, arg) = extract_plain_sin_or_cos_arg_root(ctx, expr)?;
        let arg_terms = AddView::from_expr(ctx, arg).terms;
        if arg_terms.len() != 2 {
            return None;
        }
        let positive_terms: smallvec::SmallVec<[ExprId; 2]> = arg_terms
            .iter()
            .filter_map(|(term, sign)| (*sign == Sign::Pos).then_some(*term))
            .collect();
        (positive_terms.len() == 1).then_some(positive_terms[0])
    };

    let mut plain_sin_arg = None;
    let mut plain_cos_arg = None;
    let mut shifted_base_arg = None;

    for (term_expr, _term_sign) in view.terms {
        let (_coeff, base) = extract_coef_and_base(ctx, term_expr);
        if let Some((trig_fn, arg)) = extract_plain_sin_or_cos_arg_root(ctx, base) {
            if let Some(candidate_base_arg) = extract_shifted_base_arg(ctx, base) {
                shifted_base_arg = Some(candidate_base_arg);
            } else if trig_fn == BuiltinFn::Sin {
                plain_sin_arg = Some(arg);
            } else if trig_fn == BuiltinFn::Cos {
                plain_cos_arg = Some(arg);
            }
        }
    }

    let (Some(plain_sin_arg), Some(plain_cos_arg), Some(shifted_base_arg)) =
        (plain_sin_arg, plain_cos_arg, shifted_base_arg)
    else {
        return false;
    };
    if compare_expr(ctx, plain_sin_arg, plain_cos_arg) != Ordering::Equal
        || compare_expr(ctx, plain_sin_arg, shifted_base_arg) != Ordering::Equal
    {
        return false;
    }

    let parent_ctx = crate::ParentContext::root().with_domain_mode(crate::DomainMode::Generic);
    let rule = crate::rules::arithmetic::ExpandTrigPhaseShiftToEnableCancellationRule;
    let Some(rewrite) = crate::rule::Rule::apply(&rule, ctx, expr, &parent_ctx) else {
        return false;
    };
    let zero = ctx.num(0);
    compare_expr(ctx, rewrite.final_expr(), zero) == Ordering::Equal
}

fn matches_direct_sum_diff_cubes_quotient_zero_identity_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() == 4 {
        for candidate_index in 0..view.terms.len() {
            let (quotient_term, quotient_sign) = view.terms[candidate_index];
            let Some((a, b)) = extract_sum_diff_cubes_quotient_bases_root(ctx, quotient_term)
            else {
                continue;
            };

            let remaining_terms = view
                .terms
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(index, term)| (index != candidate_index).then_some(term))
                .collect::<Vec<_>>();

            let normalized_remaining_terms = if quotient_sign == Sign::Pos {
                remaining_terms
                    .into_iter()
                    .map(|(term_expr, term_sign)| {
                        let flipped_sign = match term_sign {
                            Sign::Pos => Sign::Neg,
                            Sign::Neg => Sign::Pos,
                        };
                        (term_expr, flipped_sign)
                    })
                    .collect::<Vec<_>>()
            } else {
                remaining_terms
            };

            let expected_poly = {
                let two = ctx.num(2);
                let a_sq = ctx.add(Expr::Pow(a, two));
                let ab = smart_mul(ctx, a, b);
                let b_sq = ctx.add(Expr::Pow(b, two));
                let ab_plus_b_sq = ctx.add(Expr::Add(ab, b_sq));
                ctx.add(Expr::Add(a_sq, ab_plus_b_sq))
            };
            let remaining_expr = build_signed_sum_expr_root(ctx, &normalized_remaining_terms);
            if compare_expr(ctx, remaining_expr, expected_poly) == Ordering::Equal {
                return true;
            }
        }
    }

    let parent_ctx = build_root_shortcut_parent_ctx(&SimplifyOptions::default(), ctx, expr);
    let rule = crate::rules::arithmetic::SubtractExpandedSumDiffCubesQuotientRule;
    let Some(rewrite) = crate::rule::Rule::apply(&rule, ctx, expr, &parent_ctx) else {
        return false;
    };
    let zero = ctx.num(0);
    compare_expr(ctx, rewrite.final_expr(), zero) == Ordering::Equal
}

fn matches_direct_sqrt_perfect_square_abs_zero_identity_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 2 {
        return false;
    }

    for candidate_index in 0..view.terms.len() {
        let (sqrt_like_term, sqrt_like_sign) = view.terms[candidate_index];
        let Some(rewrite) = cas_math::perfect_square_support::try_rewrite_sqrt_perfect_square_expr(
            ctx,
            sqrt_like_term,
        ) else {
            continue;
        };

        let (other_term, other_sign) = view.terms[1 - candidate_index];
        if sqrt_like_sign == other_sign {
            continue;
        }
        if compare_expr(ctx, rewrite.rewritten, other_term) == Ordering::Equal {
            return true;
        }
    }

    false
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OddHalfPowerProductFormRoot {
    base: ExprId,
    outside_power: i64,
}

fn extract_odd_half_power_outer_factor_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, i64)> {
    if let Some(inner) = extract_unary_builtin_arg_root(ctx, expr, BuiltinFn::Abs) {
        return Some((inner, 1));
    }

    match ctx.get(expr) {
        Expr::Pow(base, exponent) => {
            let power = extract_i64_integer(ctx, *exponent)?;
            if power < 1 {
                return None;
            }
            if let Some(inner) = extract_unary_builtin_arg_root(ctx, *base, BuiltinFn::Abs) {
                Some((inner, power))
            } else {
                Some((*base, power))
            }
        }
        _ => Some((expr, 1)),
    }
}

fn extract_odd_half_power_product_form_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<OddHalfPowerProductFormRoot> {
    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() != 2 {
        return None;
    }

    for (sqrt_index, sqrt_factor) in factors.iter().copied().enumerate() {
        let Some(base) = extract_unary_builtin_arg_root(ctx, sqrt_factor, BuiltinFn::Sqrt) else {
            continue;
        };
        let outer_factor = factors[1 - sqrt_index];
        let Some((outer_base, outside_power)) =
            extract_odd_half_power_outer_factor_root(ctx, outer_factor)
        else {
            continue;
        };
        if compare_expr(ctx, outer_base, base) == Ordering::Equal {
            return Some(OddHalfPowerProductFormRoot {
                base,
                outside_power,
            });
        }
    }

    None
}

fn extract_odd_half_power_radical_form_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<OddHalfPowerProductFormRoot> {
    let radicand = extract_unary_builtin_arg_root(ctx, expr, BuiltinFn::Sqrt)?;
    let Expr::Pow(base, exponent) = ctx.get(radicand) else {
        return None;
    };
    let power = extract_i64_integer(ctx, *exponent)?;
    if power < 3 || power % 2 == 0 {
        return None;
    }

    Some(OddHalfPowerProductFormRoot {
        base: *base,
        outside_power: (power - 1) / 2,
    })
}

fn odd_half_power_domain_equivalent_target_match_root(
    ctx: &mut Context,
    rewritten: ExprId,
    target_expr: ExprId,
) -> bool {
    let Some(rewritten_form) = extract_odd_half_power_product_form_root(ctx, rewritten) else {
        return false;
    };
    let Some(target_form) = extract_odd_half_power_product_form_root(ctx, target_expr) else {
        return false;
    };

    rewritten_form.outside_power == target_form.outside_power
        && compare_expr(ctx, rewritten_form.base, target_form.base) == Ordering::Equal
}

fn matches_direct_odd_half_power_zero_identity_root(ctx: &mut Context, expr: ExprId) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 2 {
        return false;
    }

    let parent_ctx = crate::ParentContext::root().with_domain_mode(crate::DomainMode::Generic);
    let rule = crate::rules::arithmetic::ExpandOddHalfPowerToEnableCancellationRule;

    for candidate_index in 0..view.terms.len() {
        let (focus_expr, focus_sign) = normalize_signed_add_term_root(
            ctx,
            view.terms[candidate_index].0,
            view.terms[candidate_index].1,
        );
        let Some(rewrite) = crate::rule::Rule::apply(&rule, ctx, focus_expr, &parent_ctx) else {
            continue;
        };

        let (other_expr, other_sign) = normalize_signed_add_term_root(
            ctx,
            view.terms[1 - candidate_index].0,
            view.terms[1 - candidate_index].1,
        );
        if focus_sign == other_sign {
            continue;
        }

        if compare_expr(ctx, rewrite.new_expr, other_expr) == Ordering::Equal
            || odd_half_power_domain_equivalent_target_match_root(ctx, rewrite.new_expr, other_expr)
        {
            return true;
        }
    }

    false
}

fn matches_direct_odd_half_power_zero_scope_root(ctx: &mut Context, expr: ExprId) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() == 2 {
        let (lhs_expr, lhs_sign) =
            normalize_signed_add_term_root(ctx, view.terms[0].0, view.terms[0].1);
        let (rhs_expr, rhs_sign) =
            normalize_signed_add_term_root(ctx, view.terms[1].0, view.terms[1].1);
        if lhs_sign != rhs_sign {
            let lhs_matches_rhs = extract_odd_half_power_radical_form_root(ctx, lhs_expr)
                .zip(extract_odd_half_power_product_form_root(ctx, rhs_expr))
                .map(|(lhs_form, rhs_form)| {
                    lhs_form.outside_power == rhs_form.outside_power
                        && compare_expr(ctx, lhs_form.base, rhs_form.base) == Ordering::Equal
                })
                .unwrap_or(false);
            let rhs_matches_lhs = extract_odd_half_power_radical_form_root(ctx, rhs_expr)
                .zip(extract_odd_half_power_product_form_root(ctx, lhs_expr))
                .map(|(rhs_form, lhs_form)| {
                    rhs_form.outside_power == lhs_form.outside_power
                        && compare_expr(ctx, rhs_form.base, lhs_form.base) == Ordering::Equal
                })
                .unwrap_or(false);
            if lhs_matches_rhs || rhs_matches_lhs {
                return true;
            }
        }
    }

    if view.terms.len() != 2 || !expr_contains_sqrt_or_half_power_local(ctx, expr) {
        return false;
    }

    let parent_ctx = crate::ParentContext::root().with_domain_mode(crate::DomainMode::Generic);
    let expand_rule = crate::rules::arithmetic::ExpandOddHalfPowerToEnableCancellationRule;
    let Some(expanded) = crate::rule::Rule::apply(&expand_rule, ctx, expr, &parent_ctx) else {
        return false;
    };
    let cancel_rule = crate::rules::arithmetic::SubSelfToZeroRule;
    let Some(cancelled) =
        crate::rule::Rule::apply(&cancel_rule, ctx, expanded.new_expr, &parent_ctx)
    else {
        return false;
    };
    let zero = ctx.num(0);
    compare_expr(ctx, cancelled.final_expr(), zero) == Ordering::Equal
}

fn is_square_power_root(ctx: &Context, expr: ExprId) -> bool {
    let Expr::Pow(_, exponent) = ctx.get(expr) else {
        return false;
    };
    let Expr::Number(n) = ctx.get(*exponent) else {
        return false;
    };
    *n == BigRational::from_integer(2.into())
}

fn extract_plain_cube_base_root(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
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

fn extract_sum_diff_cubes_quotient_bases_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId)> {
    let Expr::Div(numerator, denominator) = ctx.get(expr) else {
        return None;
    };
    let numerator_view = AddView::from_expr(ctx, *numerator);
    let denominator_view = AddView::from_expr(ctx, *denominator);
    if numerator_view.terms.len() != 2 || denominator_view.terms.len() != 2 {
        return None;
    }

    let mut denominator_pos = None;
    let mut denominator_neg = None;
    for (term_expr, term_sign) in denominator_view.terms {
        match term_sign {
            Sign::Pos => denominator_pos = Some(term_expr),
            Sign::Neg => denominator_neg = Some(term_expr),
        }
    }

    let mut numerator_pos = None;
    let mut numerator_neg = None;
    for (term_expr, term_sign) in numerator_view.terms {
        let base = extract_plain_cube_base_root(ctx, term_expr)?;
        match term_sign {
            Sign::Pos => numerator_pos = Some(base),
            Sign::Neg => numerator_neg = Some(base),
        }
    }

    let (Some(denominator_pos), Some(denominator_neg), Some(numerator_pos), Some(numerator_neg)) = (
        denominator_pos,
        denominator_neg,
        numerator_pos,
        numerator_neg,
    ) else {
        return None;
    };

    (compare_expr(ctx, denominator_pos, numerator_pos) == Ordering::Equal
        && compare_expr(ctx, denominator_neg, numerator_neg) == Ordering::Equal)
        .then_some((denominator_pos, denominator_neg))
}

fn extract_square_power_base_root(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Pow(base, exponent) = ctx.get(expr) else {
        return None;
    };
    let Expr::Number(n) = ctx.get(*exponent) else {
        return None;
    };
    (*n == BigRational::from_integer(2.into())).then_some(*base)
}

fn extract_mul_pair_root(ctx: &mut Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() != 2 {
        return None;
    }

    Some((factors[0], factors[1]))
}

fn build_direct_perfect_square_from_terms_root(
    ctx: &mut Context,
    terms: &[(ExprId, Sign)],
) -> Option<ExprId> {
    if terms.len() != 3 {
        return None;
    }

    let expr = build_signed_sum_expr_root(ctx, terms);
    cas_math::factor::factor_perfect_square_trinomial(ctx, expr)
}

fn matches_direct_perfect_square_trinomial_zero_identity_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 4 {
        return false;
    }

    for candidate_index in 0..view.terms.len() {
        let (square_term, square_sign) = view.terms[candidate_index];
        if !is_square_power_root(ctx, square_term) {
            continue;
        }

        let remaining_terms = view
            .terms
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(index, term)| (index != candidate_index).then_some(term))
            .collect::<Vec<_>>();

        let normalized_remaining_terms = if square_sign == Sign::Neg {
            remaining_terms
        } else {
            remaining_terms
                .into_iter()
                .map(|(term_expr, term_sign)| {
                    let flipped_sign = match term_sign {
                        Sign::Pos => Sign::Neg,
                        Sign::Neg => Sign::Pos,
                    };
                    (term_expr, flipped_sign)
                })
                .collect()
        };

        let remaining_expr = build_signed_sum_expr_root(ctx, &normalized_remaining_terms);
        let Some(factored_square) =
            build_direct_perfect_square_from_terms_root(ctx, &normalized_remaining_terms)
                .or_else(|| cas_math::factor::factor_perfect_square_trinomial(ctx, remaining_expr))
        else {
            continue;
        };

        if compare_expr(ctx, factored_square, square_term) == Ordering::Equal {
            return true;
        }
    }

    false
}

fn matches_direct_log_square_product_split_zero_identity_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 3 {
        return false;
    }

    for candidate_index in 0..view.terms.len() {
        let (candidate_expr, candidate_sign) = view.terms[candidate_index];
        let Some((candidate_base_opt, candidate_arg)) =
            cas_math::expr_extract::extract_log_base_argument_view(ctx, candidate_expr)
        else {
            continue;
        };
        let Some(candidate_product_base) = extract_square_power_base_root(ctx, candidate_arg)
        else {
            continue;
        };
        let Some((factor_a, factor_b)) = extract_mul_pair_root(ctx, candidate_product_base) else {
            continue;
        };

        let mut saw_factor_a = false;
        let mut saw_factor_b = false;
        let mut ok = true;

        for (other_index, (other_expr, other_sign)) in view.terms.iter().copied().enumerate() {
            if other_index == candidate_index {
                continue;
            }
            if other_sign == candidate_sign {
                ok = false;
                break;
            }

            let Some((other_base_opt, other_arg)) =
                cas_math::expr_extract::extract_log_base_argument_view(ctx, other_expr)
            else {
                ok = false;
                break;
            };
            if other_base_opt != candidate_base_opt {
                ok = false;
                break;
            }

            let Some(other_base) = extract_square_power_base_root(ctx, other_arg) else {
                ok = false;
                break;
            };

            if !saw_factor_a && compare_expr(ctx, other_base, factor_a) == Ordering::Equal {
                saw_factor_a = true;
                continue;
            }
            if !saw_factor_b && compare_expr(ctx, other_base, factor_b) == Ordering::Equal {
                saw_factor_b = true;
                continue;
            }

            ok = false;
            break;
        }

        if ok && saw_factor_a && saw_factor_b {
            return true;
        }
    }

    false
}

fn matches_direct_ln_abs_product_split_zero_identity_root(ctx: &mut Context, expr: ExprId) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 3 {
        return false;
    }

    for candidate_index in 0..view.terms.len() {
        let (candidate_expr, candidate_sign) = view.terms[candidate_index];
        let (candidate_coeff, candidate_base) = extract_coef_and_base(ctx, candidate_expr);
        if candidate_coeff != BigRational::from_integer(2.into()) {
            continue;
        }
        let Some((candidate_base_opt, candidate_log_arg)) =
            cas_math::expr_extract::extract_log_base_argument_view(ctx, candidate_base)
        else {
            continue;
        };
        if candidate_base_opt.is_some() {
            continue;
        }
        let Some(candidate_abs_arg) =
            cas_math::expr_extract::extract_abs_argument_view(ctx, candidate_log_arg)
        else {
            continue;
        };
        let Some((factor_a, factor_b)) = extract_mul_pair_root(ctx, candidate_abs_arg) else {
            continue;
        };

        let mut saw_factor_a = false;
        let mut saw_factor_b = false;
        let mut ok = true;

        for (other_index, (other_expr, other_sign)) in view.terms.iter().copied().enumerate() {
            if other_index == candidate_index {
                continue;
            }
            if other_sign == candidate_sign {
                ok = false;
                break;
            }

            let (other_coeff, other_base) = extract_coef_and_base(ctx, other_expr);
            if other_coeff != BigRational::from_integer(2.into()) {
                ok = false;
                break;
            }

            let Some((other_base_opt, other_log_arg)) =
                cas_math::expr_extract::extract_log_base_argument_view(ctx, other_base)
            else {
                ok = false;
                break;
            };
            if other_base_opt.is_some() {
                ok = false;
                break;
            }
            let Some(other_abs_arg) =
                cas_math::expr_extract::extract_abs_argument_view(ctx, other_log_arg)
            else {
                ok = false;
                break;
            };

            if !saw_factor_a && compare_expr(ctx, other_abs_arg, factor_a) == Ordering::Equal {
                saw_factor_a = true;
                continue;
            }
            if !saw_factor_b && compare_expr(ctx, other_abs_arg, factor_b) == Ordering::Equal {
                saw_factor_b = true;
                continue;
            }

            ok = false;
            break;
        }

        if ok && saw_factor_a && saw_factor_b {
            return true;
        }
    }

    false
}

fn extract_unary_builtin_arg_root(
    ctx: &Context,
    expr: ExprId,
    builtin: BuiltinFn,
) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Function(name, args) if ctx.is_builtin(*name, builtin) && args.len() == 1 => {
            Some(args[0])
        }
        _ => None,
    }
}

fn matches_direct_tan_cot_product_zero_identity_root(ctx: &mut Context, expr: ExprId) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 2 {
        return false;
    }

    for product_index in 0..view.terms.len() {
        let (product_expr, product_sign) = normalize_signed_add_term_root(
            ctx,
            view.terms[product_index].0,
            view.terms[product_index].1,
        );
        let (other_expr, other_sign) = normalize_signed_add_term_root(
            ctx,
            view.terms[1 - product_index].0,
            view.terms[1 - product_index].1,
        );
        if product_sign == other_sign || extract_i64_integer(ctx, other_expr) != Some(1) {
            continue;
        }

        let factors = flatten_mul_chain(ctx, product_expr);
        if factors.len() != 2 {
            continue;
        }

        for (first, second) in [(factors[0], factors[1]), (factors[1], factors[0])] {
            let Some(tan_arg) = extract_unary_builtin_arg_root(ctx, first, BuiltinFn::Tan) else {
                continue;
            };
            let Some(cot_arg) = extract_unary_builtin_arg_root(ctx, second, BuiltinFn::Cot) else {
                continue;
            };
            if compare_expr(ctx, tan_arg, cot_arg) == Ordering::Equal {
                return true;
            }
        }
    }

    false
}

fn matches_direct_tan_cot_sec_csc_zero_identity_root(ctx: &mut Context, expr: ExprId) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 3 {
        return false;
    }

    let mut tan_arg = None;
    let mut cot_arg = None;
    let mut product_arg = None;
    let mut product_sign = None;

    for (term_expr, term_sign) in view.terms {
        let (term_expr, term_sign) = normalize_signed_add_term_root(ctx, term_expr, term_sign);
        if let Some(arg) = extract_unary_builtin_arg_root(ctx, term_expr, BuiltinFn::Tan) {
            if tan_arg.is_some() || term_sign != Sign::Pos {
                return false;
            }
            tan_arg = Some(arg);
            continue;
        }
        if let Some(arg) = extract_unary_builtin_arg_root(ctx, term_expr, BuiltinFn::Cot) {
            if cot_arg.is_some() || term_sign != Sign::Pos {
                return false;
            }
            cot_arg = Some(arg);
            continue;
        }

        let factors = flatten_mul_chain(ctx, term_expr);
        if factors.len() != 2 || product_arg.is_some() || term_sign != Sign::Neg {
            return false;
        }
        for (first, second) in [(factors[0], factors[1]), (factors[1], factors[0])] {
            let Some(sec_arg) = extract_unary_builtin_arg_root(ctx, first, BuiltinFn::Sec) else {
                continue;
            };
            let Some(csc_arg) = extract_unary_builtin_arg_root(ctx, second, BuiltinFn::Csc) else {
                continue;
            };
            if compare_expr(ctx, sec_arg, csc_arg) == Ordering::Equal {
                product_arg = Some(sec_arg);
                product_sign = Some(term_sign);
                break;
            }
        }
        if product_arg.is_none() {
            return false;
        }
    }

    let (Some(tan_arg), Some(cot_arg), Some(product_arg), Some(Sign::Neg)) =
        (tan_arg, cot_arg, product_arg, product_sign)
    else {
        return false;
    };

    compare_expr(ctx, tan_arg, cot_arg) == Ordering::Equal
        && compare_expr(ctx, tan_arg, product_arg) == Ordering::Equal
}

fn matches_direct_positive_double_cos_square_diff_zero_identity_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 3 {
        return false;
    }

    let mut cos_sq_arg = None;
    let mut sin_sq_arg = None;
    let mut cos_double_arg = None;

    for (term_expr, term_sign) in view.terms {
        if let Some(arg) = extract_positive_cos_double_angle_arg_root(ctx, term_expr) {
            if term_sign != Sign::Neg || cos_double_arg.is_some() {
                return false;
            }
            cos_double_arg = Some(arg);
            continue;
        }

        let Some((coeff, trig_name, arg, effective_sign)) =
            extract_signed_numeric_trig_pow2(ctx, term_expr, term_sign)
        else {
            return false;
        };
        if coeff != BigRational::one() {
            return false;
        }

        match (trig_name, effective_sign) {
            ("cos", Sign::Pos) if cos_sq_arg.is_none() => cos_sq_arg = Some(arg),
            ("sin", Sign::Neg) if sin_sq_arg.is_none() => sin_sq_arg = Some(arg),
            _ => return false,
        }
    }

    let (Some(cos_sq_arg), Some(sin_sq_arg), Some(cos_double_arg)) =
        (cos_sq_arg, sin_sq_arg, cos_double_arg)
    else {
        return false;
    };

    compare_expr(ctx, cos_sq_arg, sin_sq_arg) == Ordering::Equal
        && compare_expr(ctx, cos_sq_arg, cos_double_arg) == Ordering::Equal
}

fn matches_direct_sec_tan_pythagorean_zero_identity_root(ctx: &mut Context, expr: ExprId) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 3 {
        return false;
    }

    for first_index in 0..view.terms.len().saturating_sub(1) {
        for second_index in (first_index + 1)..view.terms.len() {
            let focus_terms: Vec<_> = view
                .terms
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(index, term)| {
                    (index == first_index || index == second_index).then_some(term)
                })
                .collect();
            let remaining_terms: Vec<_> = view
                .terms
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(index, term)| {
                    (index != first_index && index != second_index).then_some(term)
                })
                .collect();
            if focus_terms.len() != 2 || remaining_terms.len() != 1 {
                continue;
            }

            let focus_expr = build_signed_sum_expr_root(ctx, &focus_terms);
            let Some(rewrite) = try_rewrite_sec_tan_pythagorean_identity_expr(ctx, focus_expr)
            else {
                continue;
            };
            let one = ctx.num(1);
            if compare_expr(ctx, rewrite.rewritten, one) != Ordering::Equal {
                continue;
            }

            let (remaining_expr, remaining_sign) =
                normalize_signed_add_term_root(ctx, remaining_terms[0].0, remaining_terms[0].1);
            if remaining_sign == Sign::Neg && extract_i64_integer(ctx, remaining_expr) == Some(1) {
                return true;
            }
        }
    }

    false
}

fn matches_direct_csc_cot_pythagorean_zero_identity_root(ctx: &mut Context, expr: ExprId) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 3 {
        return false;
    }

    for first_index in 0..view.terms.len().saturating_sub(1) {
        for second_index in (first_index + 1)..view.terms.len() {
            let focus_terms: Vec<_> = view
                .terms
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(index, term)| {
                    (index == first_index || index == second_index).then_some(term)
                })
                .collect();
            let remaining_terms: Vec<_> = view
                .terms
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(index, term)| {
                    (index != first_index && index != second_index).then_some(term)
                })
                .collect();
            if focus_terms.len() != 2 || remaining_terms.len() != 1 {
                continue;
            }

            let focus_expr = build_signed_sum_expr_root(ctx, &focus_terms);
            let Some(rewrite) = try_rewrite_csc_cot_pythagorean_identity_expr(ctx, focus_expr)
            else {
                continue;
            };
            let one = ctx.num(1);
            if compare_expr(ctx, rewrite.rewritten, one) != Ordering::Equal {
                continue;
            }

            let (remaining_expr, remaining_sign) =
                normalize_signed_add_term_root(ctx, remaining_terms[0].0, remaining_terms[0].1);
            if remaining_sign == Sign::Neg && extract_i64_integer(ctx, remaining_expr) == Some(1) {
                return true;
            }
        }
    }

    false
}

fn matches_direct_small_polynomial_zero_identity_root(ctx: &mut Context, expr: ExprId) -> bool {
    if !matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _))
        || cas_ast::count_nodes(ctx, expr) > 24
        || expr_contains_trig_or_hyperbolic_builtin_local(ctx, expr)
        || expr_contains_log_builtin_local(ctx, expr)
    {
        return false;
    }

    let policy = crate::polynomial_identity_support::PolynomialIdentityPolicy {
        max_nodes: 24,
        max_vars: 4,
        max_atoms: 0,
        var_limit: 4,
        max_scan_depth: 12,
        max_pow_exp_scan: 8,
        poly_budget: cas_math::multipoly::PolyBudget {
            max_terms: 24,
            max_total_degree: 8,
            max_pow_exp: 8,
        },
    };

    crate::polynomial_identity_support::try_prove_polynomial_identity_zero_with_policy(
        ctx, expr, &policy,
    )
    .is_some()
}

fn extract_power_of_base_exponent_root(
    ctx: &mut Context,
    expr: ExprId,
    base: ExprId,
) -> Option<i64> {
    if compare_expr(ctx, expr, base) == Ordering::Equal {
        return Some(1);
    }

    let Expr::Pow(pow_base, exponent) = ctx.get(expr) else {
        return None;
    };
    if compare_expr(ctx, *pow_base, base) != Ordering::Equal {
        return None;
    }

    let exponent = extract_i64_integer(ctx, *exponent)?;
    (exponent >= 1).then_some(exponent)
}

fn extract_base_minus_one_factor_root(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Sub(lhs, rhs) if extract_i64_integer(ctx, *rhs) == Some(1) => Some(*lhs),
        Expr::Add(lhs, rhs) if extract_i64_integer(ctx, *rhs) == Some(-1) => Some(*lhs),
        Expr::Add(lhs, rhs) if extract_i64_integer(ctx, *lhs) == Some(-1) => Some(*rhs),
        _ => None,
    }
}

fn matches_geometric_series_sum_root(
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

fn matches_geometric_difference_terms_root(ctx: &mut Context, terms: &[(ExprId, Sign)]) -> bool {
    if terms.len() != 3 {
        return false;
    }

    for (power_index, (power_expr, power_sign)) in terms.iter().copied().enumerate() {
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
        for (index, (term_expr, term_sign)) in terms.iter().copied().enumerate() {
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

#[cfg(test)]
fn matches_direct_geometric_difference_zero_identity_root(ctx: &mut Context, expr: ExprId) -> bool {
    let terms = AddView::from_expr(ctx, expr).terms;
    matches_geometric_difference_terms_root(ctx, &terms)
}

fn matches_direct_symbolic_trig_sum_to_product_zero_identity_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 3 || !expr_contains_trig_builtin_local(ctx, expr) {
        return false;
    }

    let mut plain_term_count = 0usize;
    let mut product_term_count = 0usize;
    for (term_expr, _term_sign) in view.terms {
        let (_coeff, base) = extract_coef_and_base(ctx, term_expr);
        if extract_plain_sin_or_cos_arg_root(ctx, base).is_some() {
            plain_term_count += 1;
            continue;
        }

        let trig_factor_count = flatten_mul_chain(ctx, base)
            .into_iter()
            .filter(|factor| extract_plain_sin_or_cos_arg_root(ctx, *factor).is_some())
            .count();
        if trig_factor_count >= 2 {
            product_term_count += 1;
            continue;
        }

        return false;
    }

    if plain_term_count != 2 || product_term_count != 1 {
        return false;
    }

    let parent_ctx = crate::ParentContext::root().with_domain_mode(crate::DomainMode::Generic);
    let rule = crate::rules::arithmetic::ExpandTrigSumToProductToEnableCancellationRule;
    let Some(rewrite) = crate::rule::Rule::apply(&rule, ctx, expr, &parent_ctx) else {
        return false;
    };
    let zero = ctx.num(0);
    compare_expr(ctx, rewrite.final_expr(), zero) == Ordering::Equal
}

fn matches_direct_small_zero_identity_root(ctx: &mut Context, expr: ExprId) -> bool {
    matches_direct_quotient_pair_zero_difference_root(ctx, expr)
        || matches_direct_half_angle_square_zero_identity_root(ctx, expr)
        || matches_direct_trig_binomial_square_zero_identity_root(ctx, expr)
        || matches_direct_positive_double_cos_square_diff_zero_identity_root(ctx, expr)
        || matches_direct_tan_cot_product_zero_identity_root(ctx, expr)
        || matches_direct_tan_cot_sec_csc_zero_identity_root(ctx, expr)
        || matches_direct_sec_tan_pythagorean_zero_identity_root(ctx, expr)
        || matches_direct_csc_cot_pythagorean_zero_identity_root(ctx, expr)
        || matches_direct_trig_product_to_sum_sin_sin_zero_identity_root(ctx, expr)
        || matches_direct_trig_product_to_sum_sin_cos_zero_identity_root(ctx, expr)
        || matches_direct_trig_product_to_sum_cos_cos_zero_identity_root(ctx, expr)
        || matches_direct_trig_mixed_double_angle_zero_identity_root(ctx, expr)
        || matches_direct_nested_fraction_simplified_zero_identity_root(ctx, expr)
        || matches_direct_trig_cubic_cosine_zero_identity_root(ctx, expr)
        || matches_direct_sum_diff_cubes_quotient_zero_identity_root(ctx, expr)
        || matches_direct_sqrt_perfect_square_abs_zero_identity_root(ctx, expr)
        || matches_direct_odd_half_power_zero_scope_root(ctx, expr)
        || matches_direct_odd_half_power_zero_identity_root(ctx, expr)
        || matches_direct_perfect_square_trinomial_zero_identity_root(ctx, expr)
        || matches_direct_log_square_product_split_zero_identity_root(ctx, expr)
        || matches_direct_ln_abs_product_split_zero_identity_root(ctx, expr)
        || matches_direct_small_polynomial_zero_identity_root(ctx, expr)
        || matches_direct_consecutive_telescoping_fraction_zero_identity_root(ctx, expr)
        || matches_direct_small_rational_zero_identity_root(ctx, expr)
        || matches_direct_symbolic_trig_sum_to_product_zero_identity_root(ctx, expr)
        || matches_direct_general_phase_shift_zero_identity_root(ctx, expr)
        || matches_direct_hyperbolic_exp_sum_zero_identity_root(ctx, expr)
        || matches_direct_recursive_hyperbolic_sinh_sum_zero_identity_root(ctx, expr)
        || matches_direct_recursive_hyperbolic_cosh_sum_zero_identity_root(ctx, expr)
        || matches_direct_hyperbolic_cosh_cubic_zero_identity_root(ctx, expr)
        || matches_direct_hyperbolic_pythagorean_zero_identity_root(ctx, expr)
        || matches_direct_exp_hyperbolic_double_identity_root(ctx, expr)
}

fn matches_direct_small_zero_or_known_pair_base_root(ctx: &mut Context, expr: ExprId) -> bool {
    if matches_direct_small_zero_identity_root(ctx, expr) {
        return true;
    }

    match ctx.get(expr).clone() {
        Expr::Sub(lhs, rhs) => {
            matches_known_direct_pair_root(ctx, lhs, rhs)
                || matches_direct_half_angle_binomial_square_pair_root(ctx, lhs, rhs)
        }
        Expr::Add(lhs, rhs) => {
            let Some((pos, neg)) = (match (ctx.get(lhs), ctx.get(rhs)) {
                (Expr::Neg(inner), _) => Some((rhs, *inner)),
                (_, Expr::Neg(inner)) => Some((lhs, *inner)),
                _ => None,
            }) else {
                return false;
            };
            matches_known_direct_pair_root(ctx, pos, neg)
                || matches_direct_half_angle_binomial_square_pair_root(ctx, pos, neg)
        }
        _ => false,
    }
}

fn matches_direct_small_zero_or_known_pair_residual_root(ctx: &mut Context, expr: ExprId) -> bool {
    if matches_direct_small_zero_or_known_pair_base_root(ctx, expr)
        || matches_partitioned_direct_small_zero_sum_root(ctx, expr)
        || extract_partitioned_phase_shift_zero_chunks_root(ctx, expr).is_some()
    {
        return true;
    }

    false
}

fn should_defer_guarded_small_zero_additive_shortcut(ctx: &mut Context, expr: ExprId) -> bool {
    let (lhs, rhs) = match ctx.get(expr).clone() {
        Expr::Add(lhs, rhs) | Expr::Sub(lhs, rhs) => (lhs, rhs),
        _ => return false,
    };

    for (trig_side, guarded_side) in [(lhs, rhs), (rhs, lhs)] {
        if !expr_contains_trig_or_hyperbolic_builtin_local(ctx, trig_side)
            || !expr_contains_guarded_small_zero_family_local(ctx, guarded_side)
            || !matches!(ctx.get(trig_side), Expr::Add(_, _) | Expr::Sub(_, _))
        {
            continue;
        }

        if !matches_direct_small_zero_or_known_pair_base_root(ctx, trig_side) {
            return true;
        }
    }

    false
}

fn factors_match_by_equality_or_direct_pair_root(
    ctx: &mut Context,
    lhs: ExprId,
    rhs: ExprId,
) -> bool {
    fn normalize_negative_unit_product_root(ctx: &mut Context, expr: ExprId) -> ExprId {
        let factors = flatten_mul_chain(ctx, expr);
        if factors.len() != 2 {
            return expr;
        }

        if extract_i64_integer(ctx, factors[0]) == Some(-1) {
            return ctx.add(Expr::Neg(factors[1]));
        }
        if extract_i64_integer(ctx, factors[1]) == Some(-1) {
            return ctx.add(Expr::Neg(factors[0]));
        }

        expr
    }

    let lhs = normalize_negative_unit_product_root(ctx, lhs);
    let rhs = normalize_negative_unit_product_root(ctx, rhs);

    if compare_expr(ctx, lhs, rhs) == Ordering::Equal
        || matches_direct_addition_of_successive_unit_fractions_pair_root(ctx, lhs, rhs)
        || matches_known_direct_pair_root(ctx, lhs, rhs)
        || matches_direct_half_angle_binomial_square_pair_root(ctx, lhs, rhs)
    {
        return true;
    }

    if expr_contains_trig_or_hyperbolic_builtin_local(ctx, lhs)
        || expr_contains_trig_or_hyperbolic_builtin_local(ctx, rhs)
        || cas_ast::count_nodes(ctx, lhs) > 48
        || cas_ast::count_nodes(ctx, rhs) > 48
        || (matches!(ctx.get(lhs), Expr::Mul(_, _)) && matches!(ctx.get(rhs), Expr::Mul(_, _)))
    {
        return false;
    }

    let difference = ctx.add(Expr::Sub(lhs, rhs));
    cas_ast::count_nodes(ctx, difference) <= 96
        && isolated_simplify_rewrites_to_zero(
            &crate::phase::SimplifyOptions::default(),
            ctx,
            difference,
        )
}

fn matches_product_to_sum_sin_cos_factor_pair_direction_root(
    ctx: &mut Context,
    source_factors: &[ExprId],
    target_factors: &[ExprId],
) -> bool {
    if source_factors.len() < 4 || target_factors.is_empty() {
        return false;
    }

    for first in 0..source_factors.len().saturating_sub(2) {
        for second in (first + 1)..source_factors.len().saturating_sub(1) {
            for third in (second + 1)..source_factors.len() {
                let trig_subset = build_mul_expr_from_factors_root(
                    ctx,
                    &[
                        source_factors[first],
                        source_factors[second],
                        source_factors[third],
                    ],
                );

                for (target_index, target_factor) in target_factors.iter().copied().enumerate() {
                    if !matches_direct_trig_product_to_sum_sin_cos_pair_root(
                        ctx,
                        trig_subset,
                        target_factor,
                    ) {
                        continue;
                    }

                    let remaining_source: Vec<_> = source_factors
                        .iter()
                        .copied()
                        .enumerate()
                        .filter_map(|(index, factor)| {
                            (index != first && index != second && index != third).then_some(factor)
                        })
                        .collect();
                    let remaining_target: Vec<_> = target_factors
                        .iter()
                        .copied()
                        .enumerate()
                        .filter_map(|(index, factor)| (index != target_index).then_some(factor))
                        .collect();

                    if remaining_source.is_empty() || remaining_target.is_empty() {
                        continue;
                    }

                    let source_partner = build_mul_expr_from_factors_root(ctx, &remaining_source);
                    let target_partner = build_mul_expr_from_factors_root(ctx, &remaining_target);
                    if factors_match_by_equality_or_direct_pair_root(
                        ctx,
                        source_partner,
                        target_partner,
                    ) {
                        return true;
                    }
                }
            }
        }
    }

    false
}

fn matches_direct_product_to_sum_sin_cos_factor_pair_zero_difference_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 2 {
        return false;
    }

    let (lhs_term, rhs_term) = match (view.terms[0].1, view.terms[1].1) {
        (Sign::Pos, Sign::Neg) => (view.terms[0].0, view.terms[1].0),
        (Sign::Neg, Sign::Pos) => (view.terms[1].0, view.terms[0].0),
        _ => return false,
    };

    let lhs_factors = flatten_mul_chain(ctx, lhs_term);
    let rhs_factors = flatten_mul_chain(ctx, rhs_term);
    matches_product_to_sum_sin_cos_factor_pair_direction_root(ctx, &lhs_factors, &rhs_factors)
        || matches_product_to_sum_sin_cos_factor_pair_direction_root(
            ctx,
            &rhs_factors,
            &lhs_factors,
        )
}

fn strip_multiplicative_one_root(ctx: &mut Context, expr: ExprId) -> ExprId {
    match ctx.get(expr).clone() {
        Expr::Mul(_, _) => {
            let factors = flatten_mul_chain(ctx, expr);
            let original_len = factors.len();
            let filtered: smallvec::SmallVec<[ExprId; 4]> = factors
                .into_iter()
                .filter(|factor| extract_i64_integer(ctx, *factor) != Some(1))
                .collect();
            if filtered.len() == 1 {
                filtered[0]
            } else if !filtered.is_empty() && filtered.len() < original_len {
                build_mul_expr_from_factors_root(ctx, &filtered)
            } else {
                expr
            }
        }
        Expr::Div(numerator, denominator) => {
            let normalized_numerator = strip_multiplicative_one_root(ctx, numerator);
            if compare_expr(ctx, normalized_numerator, numerator) == Ordering::Equal {
                expr
            } else {
                ctx.add(Expr::Div(normalized_numerator, denominator))
            }
        }
        _ => expr,
    }
}

fn build_two_group_factorizations_root(
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

fn matches_direct_two_factor_product_pair_zero_difference_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    fn perfect_square_pairwise_matches(
        ctx: &mut Context,
        left_a: ExprId,
        left_b: ExprId,
        right_a: ExprId,
        right_b: ExprId,
    ) -> bool {
        (matches_direct_perfect_square_trinomial_pair_root(ctx, left_a, right_a)
            && factors_match_by_equality_or_direct_pair_root(ctx, left_b, right_b))
            || (matches_direct_perfect_square_trinomial_pair_root(ctx, left_b, right_b)
                && factors_match_by_equality_or_direct_pair_root(ctx, left_a, right_a))
            || (matches_direct_perfect_square_trinomial_pair_root(ctx, left_a, right_b)
                && factors_match_by_equality_or_direct_pair_root(ctx, left_b, right_a))
            || (matches_direct_perfect_square_trinomial_pair_root(ctx, left_b, right_a)
                && factors_match_by_equality_or_direct_pair_root(ctx, left_a, right_b))
    }

    fn perfect_square_with_grouped_sum_diff_cubes_matches(
        ctx: &mut Context,
        two_factor_side: &[ExprId],
        three_factor_side: &[ExprId],
    ) -> bool {
        if two_factor_side.len() != 2 || three_factor_side.len() != 3 {
            return false;
        }

        for grouped_anchor_index in 0..three_factor_side.len() {
            let grouped_anchor = three_factor_side[grouped_anchor_index];
            let grouped_partner_factors: Vec<_> = three_factor_side
                .iter()
                .enumerate()
                .filter_map(|(index, factor)| (index != grouped_anchor_index).then_some(*factor))
                .collect();
            let grouped_partner = build_mul_expr_from_factors_root(ctx, &grouped_partner_factors);

            if (matches_direct_perfect_square_trinomial_pair_root(
                ctx,
                two_factor_side[0],
                grouped_anchor,
            ) && matches_direct_sum_diff_cubes_product_pair_root(
                ctx,
                two_factor_side[1],
                grouped_partner,
            )) || (matches_direct_perfect_square_trinomial_pair_root(
                ctx,
                two_factor_side[1],
                grouped_anchor,
            ) && matches_direct_sum_diff_cubes_product_pair_root(
                ctx,
                two_factor_side[0],
                grouped_partner,
            )) {
                return true;
            }
        }

        false
    }

    fn perfect_square_with_grouped_sophie_germain_matches(
        ctx: &mut Context,
        two_factor_side: &[ExprId],
        three_factor_side: &[ExprId],
    ) -> bool {
        if two_factor_side.len() != 2 || three_factor_side.len() != 3 {
            return false;
        }

        for grouped_anchor_index in 0..three_factor_side.len() {
            let grouped_anchor = three_factor_side[grouped_anchor_index];
            let grouped_partner_factors: Vec<_> = three_factor_side
                .iter()
                .enumerate()
                .filter_map(|(index, factor)| (index != grouped_anchor_index).then_some(*factor))
                .collect();
            let grouped_partner = build_mul_expr_from_factors_root(ctx, &grouped_partner_factors);

            if (matches_direct_perfect_square_trinomial_pair_root(
                ctx,
                two_factor_side[0],
                grouped_anchor,
            ) && matches_direct_sophie_germain_pair_root(
                ctx,
                two_factor_side[1],
                grouped_partner,
            )) || (matches_direct_perfect_square_trinomial_pair_root(
                ctx,
                two_factor_side[1],
                grouped_anchor,
            ) && matches_direct_sophie_germain_pair_root(
                ctx,
                two_factor_side[0],
                grouped_partner,
            )) {
                return true;
            }
        }

        false
    }

    fn perfect_square_with_grouped_higher_degree_difference_matches(
        ctx: &mut Context,
        two_factor_side: &[ExprId],
        five_factor_side: &[ExprId],
    ) -> bool {
        if two_factor_side.len() != 2 || five_factor_side.len() != 5 {
            return false;
        }

        for grouped_anchor_index in 0..five_factor_side.len() {
            let grouped_anchor = five_factor_side[grouped_anchor_index];
            let grouped_partner_factors: Vec<_> = five_factor_side
                .iter()
                .enumerate()
                .filter_map(|(index, factor)| (index != grouped_anchor_index).then_some(*factor))
                .collect();
            let grouped_partner = build_mul_expr_from_factors_root(ctx, &grouped_partner_factors);

            if (matches_direct_perfect_square_trinomial_pair_root(
                ctx,
                two_factor_side[0],
                grouped_anchor,
            ) && matches_direct_higher_degree_difference_pair_root(
                ctx,
                two_factor_side[1],
                grouped_partner,
            )) || (matches_direct_perfect_square_trinomial_pair_root(
                ctx,
                two_factor_side[1],
                grouped_anchor,
            ) && matches_direct_higher_degree_difference_pair_root(
                ctx,
                two_factor_side[0],
                grouped_partner,
            )) {
                return true;
            }
        }

        false
    }

    fn pairwise_matches(
        ctx: &mut Context,
        left_a: ExprId,
        left_b: ExprId,
        right_a: ExprId,
        right_b: ExprId,
    ) -> bool {
        (factors_match_by_equality_or_direct_pair_root(ctx, left_a, right_a)
            && factors_match_by_equality_or_direct_pair_root(ctx, left_b, right_b))
            || (factors_match_by_equality_or_direct_pair_root(ctx, left_a, right_b)
                && factors_match_by_equality_or_direct_pair_root(ctx, left_b, right_a))
    }

    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 2 {
        return false;
    }

    let (lhs_term, rhs_term) = match (view.terms[0].1, view.terms[1].1) {
        (Sign::Pos, Sign::Neg) => (view.terms[0].0, view.terms[1].0),
        (Sign::Neg, Sign::Pos) => (view.terms[1].0, view.terms[0].0),
        _ => return false,
    };

    let lhs_factors = flatten_mul_chain(ctx, lhs_term);
    let rhs_factors = flatten_mul_chain(ctx, rhs_term);

    if lhs_factors.len() == 2
        && rhs_factors.len() == 2
        && perfect_square_pairwise_matches(
            ctx,
            lhs_factors[0],
            lhs_factors[1],
            rhs_factors[0],
            rhs_factors[1],
        )
    {
        return true;
    }

    if (lhs_factors.len() == 2
        && rhs_factors.len() == 3
        && perfect_square_with_grouped_sum_diff_cubes_matches(ctx, &lhs_factors, &rhs_factors))
        || (lhs_factors.len() == 3
            && rhs_factors.len() == 2
            && perfect_square_with_grouped_sum_diff_cubes_matches(ctx, &rhs_factors, &lhs_factors))
    {
        return true;
    }

    if (lhs_factors.len() == 2
        && rhs_factors.len() == 3
        && perfect_square_with_grouped_sophie_germain_matches(ctx, &lhs_factors, &rhs_factors))
        || (lhs_factors.len() == 3
            && rhs_factors.len() == 2
            && perfect_square_with_grouped_sophie_germain_matches(ctx, &rhs_factors, &lhs_factors))
    {
        return true;
    }

    if (lhs_factors.len() == 2
        && rhs_factors.len() == 5
        && perfect_square_with_grouped_higher_degree_difference_matches(
            ctx,
            &lhs_factors,
            &rhs_factors,
        ))
        || (lhs_factors.len() == 5
            && rhs_factors.len() == 2
            && perfect_square_with_grouped_higher_degree_difference_matches(
                ctx,
                &rhs_factors,
                &lhs_factors,
            ))
    {
        return true;
    }

    if matches_direct_product_to_sum_sin_cos_factor_pair_zero_difference_root(ctx, expr) {
        return true;
    }

    let lhs_groupings = build_two_group_factorizations_root(ctx, &lhs_factors);
    let rhs_groupings = build_two_group_factorizations_root(ctx, &rhs_factors);
    if lhs_groupings.is_empty() || rhs_groupings.is_empty() {
        return false;
    }

    for (lhs_a, lhs_b) in lhs_groupings {
        for (rhs_a, rhs_b) in rhs_groupings.iter().copied() {
            if pairwise_matches(ctx, lhs_a, lhs_b, rhs_a, rhs_b) {
                return true;
            }
        }
    }

    false
}

fn try_standard_two_factor_product_pair_zero_shortcut(
    _options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let has_common_scale = extract_common_multiplicative_residual_sum_root(ctx, expr).is_some();
    if let Some((_common_factor, residual_expr)) =
        extract_common_multiplicative_residual_sum_root(ctx, expr)
    {
        if matches_direct_small_zero_or_known_pair_residual_root(ctx, residual_expr)
            || has_structural_numeric_pythagorean_pair(ctx, residual_expr)
        {
            return None;
        }
    }

    if !matches_direct_two_factor_product_pair_zero_difference_root(ctx, expr) {
        return None;
    }

    let zero = ctx.num(0);
    let rewrite =
        crate::rule::Rewrite::with_local(zero, "Equivalent Product Pair Cancellation", expr, zero);
    Some(finish_root_shortcut_with_rewrite_meta(
        ctx,
        expr,
        rewrite,
        if has_common_scale {
            "Collapse Common-Scale Equivalent Difference"
        } else {
            "Collapse Product of Equivalent Factors Difference"
        },
        collect_steps,
    ))
}

fn matches_direct_quotient_pair_zero_difference_root(ctx: &mut Context, expr: ExprId) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 2 {
        return false;
    }

    let (lhs_term, rhs_term) = match (view.terms[0].1, view.terms[1].1) {
        (Sign::Pos, Sign::Neg) => (view.terms[0].0, view.terms[1].0),
        (Sign::Neg, Sign::Pos) => (view.terms[1].0, view.terms[0].0),
        _ => return false,
    };

    let (lhs_num, lhs_den) = match ctx.get(lhs_term) {
        Expr::Div(num, den) => (*num, *den),
        _ => return false,
    };
    let (rhs_num, rhs_den) = match ctx.get(rhs_term) {
        Expr::Div(num, den) => (*num, *den),
        _ => return false,
    };

    factors_match_by_equality_or_direct_pair_root(ctx, lhs_num, rhs_num)
        && factors_match_by_equality_or_direct_pair_root(ctx, lhs_den, rhs_den)
}

fn matches_direct_or_isolated_quotient_pair_zero_difference_root(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 2 {
        return false;
    }

    let (lhs_term, rhs_term) = match (view.terms[0].1, view.terms[1].1) {
        (Sign::Pos, Sign::Neg) => (view.terms[0].0, view.terms[1].0),
        (Sign::Neg, Sign::Pos) => (view.terms[1].0, view.terms[0].0),
        _ => return false,
    };

    let (lhs_num, lhs_den) = match ctx.get(lhs_term) {
        Expr::Div(num, den) => (*num, *den),
        _ => return false,
    };
    let (rhs_num, rhs_den) = match ctx.get(rhs_term) {
        Expr::Div(num, den) => (*num, *den),
        _ => return false,
    };

    if !factors_match_by_equality_or_direct_pair_root(ctx, lhs_num, rhs_num) {
        return false;
    }
    if factors_match_by_equality_or_direct_pair_root(ctx, lhs_den, rhs_den) {
        return true;
    }

    let denominator_difference = ctx.add(Expr::Sub(lhs_den, rhs_den));
    if matches_direct_small_zero_or_known_pair_residual_root(ctx, denominator_difference) {
        return true;
    }

    isolated_simplify_rewrites_to_zero(options, ctx, denominator_difference)
}

fn build_mul_expr_from_factors_root(ctx: &mut Context, factors: &[ExprId]) -> ExprId {
    match factors {
        [] => ctx.num(1),
        [single] => *single,
        _ => build_balanced_mul(ctx, factors),
    }
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

fn build_locally_simplified_mul_expr_from_factors_root(
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

fn build_nonexpanding_locally_simplified_mul_expr_from_factors_root(
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

fn extract_common_multiplicative_residual_sum_root(
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

fn try_standard_common_scale_exact_zero_shortcut_fallback(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let (_common_factor, residual_expr) =
        extract_common_multiplicative_residual_sum_root(ctx, expr)?;

    if matches_direct_small_zero_or_known_pair_residual_root(ctx, residual_expr) {
        let zero = ctx.num(0);
        let rewrite =
            crate::rule::Rewrite::with_local(zero, "Equivalent Residual Cancellation", expr, zero);
        return Some(finish_root_shortcut_with_rewrite_meta(
            ctx,
            expr,
            rewrite,
            "Collapse Common-Scale Equivalent Difference",
            collect_steps,
        ));
    }

    if try_standard_exact_zero_equivalence_shortcut(options, ctx, residual_expr, false).is_some() {
        let zero = ctx.num(0);
        let rewrite =
            crate::rule::Rewrite::with_local(zero, "Equivalent Residual Cancellation", expr, zero);
        return Some(finish_root_shortcut_with_rewrite_meta(
            ctx,
            expr,
            rewrite,
            "Collapse Common-Scale Equivalent Difference",
            collect_steps,
        ));
    }

    if matches_direct_two_factor_product_pair_zero_difference_root(ctx, expr)
        || matches_direct_quotient_pair_zero_difference_root(ctx, expr)
    {
        let zero = ctx.num(0);
        let rewrite =
            crate::rule::Rewrite::with_local(zero, "Equivalent Residual Cancellation", expr, zero);
        return Some(finish_root_shortcut_with_rewrite_meta(
            ctx,
            expr,
            rewrite,
            "Collapse Common-Scale Equivalent Difference",
            collect_steps,
        ));
    }

    let mut residual_simplifier = crate::Simplifier::with_default_rules();
    std::mem::swap(&mut residual_simplifier.context, ctx);
    let mut residual_orchestrator = Orchestrator::new();
    residual_orchestrator.options = SimplifyOptions {
        collect_steps: false,
        suppress_depth_overflow_warnings: true,
        ..options.clone()
    };
    let (residual_result, _residual_steps, _stats) =
        residual_orchestrator.simplify_pipeline(residual_expr, &mut residual_simplifier);
    std::mem::swap(&mut residual_simplifier.context, ctx);

    let zero = ctx.num(0);
    if compare_expr(ctx, residual_result, zero) != Ordering::Equal {
        return None;
    }

    let rewrite =
        crate::rule::Rewrite::with_local(zero, "Equivalent Residual Cancellation", expr, zero);
    Some(finish_root_shortcut_with_rewrite_meta(
        ctx,
        expr,
        rewrite,
        "Collapse Common-Scale Equivalent Difference",
        collect_steps,
    ))
}

fn try_standard_common_scale_known_pair_shortcut(
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let (_common_factor, residual_expr) =
        extract_common_multiplicative_residual_sum_root(ctx, expr)?;
    if !matches_direct_small_zero_or_known_pair_residual_root(ctx, residual_expr) {
        return None;
    }

    let zero = ctx.num(0);
    let rewrite =
        crate::rule::Rewrite::with_local(zero, "Equivalent Residual Cancellation", expr, zero);
    Some(finish_root_shortcut_with_rewrite_meta(
        ctx,
        expr,
        rewrite,
        "Collapse Common-Scale Equivalent Difference",
        collect_steps,
    ))
}

fn format_standard_simplify_square_root_shortcut_desc(
    kind: SimplifySquareRootRewriteKind,
) -> &'static str {
    match kind {
        SimplifySquareRootRewriteKind::PerfectSquare => "Simplify perfect square root",
        SimplifySquareRootRewriteKind::SquareRootFactors => "Simplify square root factors",
        SimplifySquareRootRewriteKind::AdditiveCommonFactor => {
            "Extract common square factor from additive radicand"
        }
        SimplifySquareRootRewriteKind::QuotientOfSquares => {
            "Simplify square root of quotient of squares"
        }
    }
}

fn try_standard_simplify_square_root_shortcut(
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let rewrite = try_rewrite_simplify_square_root_expr(ctx, expr)?;
    let rewrite = crate::rule::Rewrite::new(rewrite.rewritten).desc(
        format_standard_simplify_square_root_shortcut_desc(rewrite.kind),
    );
    Some(finish_standard_root_shortcut(
        ctx,
        expr,
        rewrite,
        "Simplify Square Root",
        collect_steps,
    ))
}

fn try_standard_abs_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let Expr::Function(fn_id, _) = ctx.get(expr) else {
        return None;
    };
    if !ctx.is_builtin(*fn_id, BuiltinFn::Abs) {
        return None;
    }

    let parent_ctx = build_root_shortcut_parent_ctx(options, ctx, expr);

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

fn try_standard_assumed_dyadic_cos_product_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let Expr::Mul(_, _) = ctx.get(expr) else {
        return None;
    };

    let plan = cas_math::trig_multi_angle_support::try_plan_dyadic_cos_product_with_policy(
        ctx,
        expr,
        matches!(
            options.shared.semantics.domain_mode,
            crate::DomainMode::Assume
        ),
        matches!(
            options.shared.semantics.domain_mode,
            crate::DomainMode::Strict
        ),
    )?;
    if !matches!(
        plan.policy,
        cas_math::trig_dyadic_policy_support::DyadicSinNonzeroPolicyDecision::Apply {
            assume_sin_nonzero: true
        }
    ) {
        return None;
    }

    let parent_ctx = build_root_shortcut_parent_ctx(options, ctx, expr);
    let rule = crate::rules::trigonometry::DyadicCosProductToSinRule;
    let rewrite = crate::rule::Rule::apply(&rule, ctx, expr, &parent_ctx)?;
    Some(finish_root_shortcut_with_rewrite_meta(
        ctx,
        expr,
        rewrite,
        "Dyadic Cos Product",
        collect_steps,
    ))
}

fn try_standard_sum_diff_cubes_fraction_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let Expr::Div(_, _) = ctx.get(expr) else {
        return None;
    };

    let parent_ctx = build_root_shortcut_parent_ctx(options, ctx, expr);
    let rule = crate::rules::algebra::CancelSumDiffCubesFractionRule;
    let rewrite = crate::rule::Rule::apply(&rule, ctx, expr, &parent_ctx)?;
    Some(finish_root_shortcut_with_rewrite_meta(
        ctx,
        expr,
        rewrite,
        "Cancel Sum/Difference of Cubes Fraction",
        collect_steps,
    ))
}

fn try_standard_sub_self_cancel_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let parent_ctx = build_root_shortcut_parent_ctx(options, ctx, expr);
    let odd_half_power_rule = crate::rules::arithmetic::ExpandOddHalfPowerToEnableCancellationRule;
    if let Some(rewrite) = crate::rule::Rule::apply(&odd_half_power_rule, ctx, expr, &parent_ctx) {
        let cancel_rule = crate::rules::arithmetic::SubSelfToZeroRule;
        let cancel_rewrite =
            crate::rule::Rule::apply(&cancel_rule, ctx, rewrite.new_expr, &parent_ctx)?;

        let result = cancel_rewrite.new_expr;
        let mut shortcut_steps = Vec::new();
        if collect_steps {
            let mut first_step = Step::with_snapshots(
                &rewrite.description,
                "Expand Odd Half Power",
                expr,
                rewrite.new_expr,
                smallvec::SmallVec::<[crate::step::PathStep; 8]>::new(),
                Some(ctx),
                expr,
                rewrite.new_expr,
            );
            first_step.importance = crate::step::ImportanceLevel::High;
            {
                let meta = first_step.meta_mut();
                meta.before_local = rewrite.before_local;
                meta.after_local = rewrite.after_local;
                meta.assumption_events = rewrite.assumption_events.clone();
                meta.required_conditions = rewrite.required_conditions.clone();
                meta.poly_proof = rewrite.poly_proof.clone();
                meta.substeps = rewrite.substeps.clone();
            }
            shortcut_steps.push(first_step);

            let mut second_step = Step::with_snapshots(
                &cancel_rewrite.description,
                "Subtraction Self-Cancel",
                rewrite.new_expr,
                result,
                smallvec::SmallVec::<[crate::step::PathStep; 8]>::new(),
                Some(ctx),
                rewrite.new_expr,
                result,
            );
            second_step.importance = crate::step::ImportanceLevel::High;
            {
                let meta = second_step.meta_mut();
                meta.before_local = cancel_rewrite.before_local;
                meta.after_local = cancel_rewrite.after_local;
                meta.assumption_events = cancel_rewrite.assumption_events.clone();
                meta.required_conditions = cancel_rewrite.required_conditions.clone();
                meta.poly_proof = cancel_rewrite.poly_proof.clone();
                meta.substeps = cancel_rewrite.substeps.clone();
            }
            shortcut_steps.push(second_step);
        }

        return Some((result, shortcut_steps));
    }

    let log_abs_rule = crate::rules::arithmetic::ExpandLogAbsMulDivToEnableCancellationRule;
    if let Some(rewrite) = crate::rule::Rule::apply(&log_abs_rule, ctx, expr, &parent_ctx) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        std::mem::swap(&mut simplifier.context, ctx);
        let (result, inner_steps, _stats) = simplifier.simplify_with_stats(
            rewrite.new_expr,
            crate::SimplifyOptions {
                suppress_depth_overflow_warnings: true,
                ..crate::SimplifyOptions::default()
            },
        );
        std::mem::swap(&mut simplifier.context, ctx);

        let zero = ctx.num(0);
        if compare_expr(ctx, result, zero) == Ordering::Equal && !inner_steps.is_empty() {
            let mut shortcut_steps = Vec::new();
            if collect_steps {
                let mut first_step = Step::with_snapshots(
                    &rewrite.description,
                    "Expand Log Abs Mul/Div",
                    expr,
                    rewrite.new_expr,
                    smallvec::SmallVec::<[crate::step::PathStep; 8]>::new(),
                    Some(ctx),
                    expr,
                    rewrite.new_expr,
                );
                first_step.importance = crate::step::ImportanceLevel::High;
                {
                    let meta = first_step.meta_mut();
                    meta.before_local = rewrite.before_local;
                    meta.after_local = rewrite.after_local;
                    meta.assumption_events = rewrite.assumption_events.clone();
                    meta.required_conditions = rewrite.required_conditions.clone();
                    meta.poly_proof = rewrite.poly_proof.clone();
                    meta.substeps = rewrite.substeps.clone();
                }
                shortcut_steps.push(first_step);
                shortcut_steps.extend(inner_steps);
            }

            return Some((result, shortcut_steps));
        }

        if compare_expr(ctx, result, zero) == Ordering::Equal {
            let mut shortcut_steps = Vec::new();
            if collect_steps {
                let mut first_step = Step::with_snapshots(
                    &rewrite.description,
                    "Expand Log Abs Mul/Div",
                    expr,
                    rewrite.new_expr,
                    smallvec::SmallVec::<[crate::step::PathStep; 8]>::new(),
                    Some(ctx),
                    expr,
                    rewrite.new_expr,
                );
                first_step.importance = crate::step::ImportanceLevel::High;
                {
                    let meta = first_step.meta_mut();
                    meta.before_local = rewrite.before_local;
                    meta.after_local = rewrite.after_local;
                    meta.assumption_events = rewrite.assumption_events.clone();
                    meta.required_conditions = rewrite.required_conditions.clone();
                    meta.poly_proof = rewrite.poly_proof.clone();
                    meta.substeps = rewrite.substeps.clone();
                }
                shortcut_steps.push(first_step);

                let mut second_step = Step::with_snapshots(
                    "Exact identity cancellation",
                    "Polynomial Identity",
                    rewrite.new_expr,
                    result,
                    smallvec::SmallVec::<[crate::step::PathStep; 8]>::new(),
                    Some(ctx),
                    rewrite.new_expr,
                    result,
                );
                second_step.importance = crate::step::ImportanceLevel::High;
                shortcut_steps.push(second_step);
            }

            return Some((result, shortcut_steps));
        }
    }

    let rule = crate::rules::arithmetic::SubSelfToZeroRule;
    let rewrite = crate::rule::Rule::apply(&rule, ctx, expr, &parent_ctx)?;
    Some(finish_standard_root_shortcut(
        ctx,
        expr,
        rewrite,
        "Subtraction Self-Cancel",
        collect_steps,
    ))
}

fn try_standard_exact_zero_equivalence_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    if is_guarded_small_zero_composition_candidate_root(ctx, expr) {
        let parent_ctx = build_root_shortcut_parent_ctx(options, ctx, expr);
        match ctx.get(expr) {
            Expr::Mul(_, _) => {
                let rule = crate::rules::arithmetic::CollapseExactZeroProductFactorRule;
                if let Some(rewrite) = crate::rule::Rule::apply(&rule, ctx, expr, &parent_ctx) {
                    return Some(finish_root_shortcut_with_rewrite_meta(
                        ctx,
                        expr,
                        rewrite,
                        "Collapse Zero Product via Exact Residual",
                        collect_steps,
                    ));
                }
            }
            Expr::Add(lhs, rhs) | Expr::Sub(lhs, rhs) => {
                if matches_direct_trig_cubic_cosine_pair_root(ctx, *lhs, *rhs) {
                    let zero = ctx.num(0);
                    return Some(run_named_rebuilt_root_shortcut_simplify(
                        options,
                        ctx,
                        expr,
                        zero,
                        "Collapse Exact Zero Additive Subexpression",
                        "Collapse Exact Zero Additive Subexpression",
                        collect_steps,
                    ));
                }
                let rule = crate::rules::arithmetic::CollapseExactZeroThreeTermSubsetRule;
                if let Some(rewrite) = crate::rule::Rule::apply(&rule, ctx, expr, &parent_ctx) {
                    return Some(finish_root_shortcut_with_rewrite_meta(
                        ctx,
                        expr,
                        rewrite,
                        "Collapse Exact Zero Additive Subexpression",
                        collect_steps,
                    ));
                }
            }
            _ => {}
        }
    }

    let binary_zero_pair = match ctx.get(expr) {
        Expr::Add(lhs, rhs) | Expr::Sub(lhs, rhs) | Expr::Mul(lhs, rhs) => Some((*lhs, *rhs)),
        _ => None,
    };
    if let Some((lhs, rhs)) = binary_zero_pair {
        if matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _))
            && matches_direct_trig_cubic_cosine_pair_root(ctx, lhs, rhs)
        {
            let zero = ctx.num(0);
            return Some(run_rebuilt_root_shortcut_simplify(
                options,
                ctx,
                expr,
                zero,
                collect_steps,
            ));
        }

        if matches_direct_cos_square_diff_pair_root(ctx, lhs, rhs) {
            let zero = ctx.num(0);
            return Some(run_rebuilt_root_shortcut_simplify(
                options,
                ctx,
                expr,
                zero,
                collect_steps,
            ));
        }

        let lhs_is_direct_zero = matches_direct_small_zero_identity_root(ctx, lhs);
        let rhs_is_direct_zero = matches_direct_small_zero_identity_root(ctx, rhs);
        let lhs_is_small_trig_zero = is_small_trig_or_hyperbolic_zero_child(options, ctx, lhs)
            && child_isolated_exact_zero(options, ctx, lhs);
        let rhs_is_small_trig_zero = is_small_trig_or_hyperbolic_zero_child(options, ctx, rhs)
            && child_isolated_exact_zero(options, ctx, rhs);

        if lhs_is_direct_zero && rhs_is_direct_zero {
            let zero = ctx.num(0);
            let rewrite =
                crate::rule::Rewrite::with_local(zero, "Exact Zero Core Composition", expr, zero);
            return Some(finish_root_shortcut_with_rewrite_meta(
                ctx,
                expr,
                rewrite,
                "Collapse Exact Zero Additive Subexpression",
                collect_steps,
            ));
        }

        if (lhs_is_direct_zero && rhs_is_small_trig_zero)
            || (rhs_is_direct_zero && lhs_is_small_trig_zero)
        {
            let zero = ctx.num(0);
            return Some(run_rebuilt_root_shortcut_simplify(
                options,
                ctx,
                expr,
                zero,
                collect_steps,
            ));
        }
    }

    let parent_ctx = build_root_shortcut_parent_ctx(options, ctx, expr);
    let common_scale_rule = crate::rules::arithmetic::CollapseExactZeroCommonScaledDifferenceRule;

    if let Some((_common_factor, residual_expr)) =
        extract_common_multiplicative_residual_sum_root(ctx, expr)
    {
        if matches_direct_small_zero_or_known_pair_residual_root(ctx, residual_expr) {
            let zero = ctx.num(0);
            return Some(run_common_scale_rebuilt_root_shortcut_simplify(
                options,
                ctx,
                expr,
                zero,
                collect_steps,
            ));
        }
    }

    if is_same_denominator_difference_root(ctx, expr) {
        if let Some((_den, lhs_core, rhs_core)) =
            extract_same_denominator_direct_pair_root(ctx, expr)
        {
            if matches_known_direct_pair_root(ctx, lhs_core, rhs_core)
                || matches_direct_half_angle_binomial_square_pair_root(ctx, lhs_core, rhs_core)
            {
                let zero = ctx.num(0);
                return Some(run_common_scale_rebuilt_root_shortcut_simplify(
                    options,
                    ctx,
                    expr,
                    zero,
                    collect_steps,
                ));
            }

            if let Some((lhs_residual, rhs_residual)) =
                extract_shared_additive_passthrough_pair_cores_root(ctx, lhs_core, rhs_core)
            {
                if matches_known_direct_pair_root(ctx, lhs_residual, rhs_residual)
                    || matches_direct_half_angle_binomial_square_pair_root(
                        ctx,
                        lhs_residual,
                        rhs_residual,
                    )
                {
                    let zero = ctx.num(0);
                    return Some(run_common_scale_rebuilt_root_shortcut_simplify(
                        options,
                        ctx,
                        expr,
                        zero,
                        collect_steps,
                    ));
                }
            }
        }

        if let Some(rewrite) = crate::rule::Rule::apply(&common_scale_rule, ctx, expr, &parent_ctx)
        {
            let zero = ctx.num(0);
            if compare_expr(ctx, rewrite.final_expr(), zero) == Ordering::Equal {
                return Some(finish_root_shortcut_with_rewrite_meta(
                    ctx,
                    expr,
                    rewrite,
                    "Collapse Common-Scale Equivalent Difference",
                    collect_steps,
                ));
            }
        }
    }

    if matches_direct_two_factor_product_pair_zero_difference_root(ctx, expr)
        || matches_direct_quotient_pair_zero_difference_root(ctx, expr)
    {
        let zero = ctx.num(0);
        let rewrite =
            crate::rule::Rewrite::with_local(zero, "Equivalent Residual Cancellation", expr, zero);
        return Some(finish_root_shortcut_with_rewrite_meta(
            ctx,
            expr,
            rewrite,
            "Collapse Common-Scale Equivalent Difference",
            collect_steps,
        ));
    }

    if let Some((result, shortcut_steps)) =
        try_standard_subtract_expanded_sum_diff_cubes_quotient_shortcut(
            options,
            ctx,
            expr,
            collect_steps,
        )
    {
        return Some((result, shortcut_steps));
    }

    if let Some((lhs_core, rhs_core)) =
        extract_shared_additive_passthrough_sub_cores_root(ctx, expr)
    {
        if matches_direct_small_pow_expansion_pair_root(ctx, lhs_core, rhs_core) {
            let zero = ctx.num(0);
            return Some(run_rebuilt_root_shortcut_simplify(
                options,
                ctx,
                expr,
                zero,
                collect_steps,
            ));
        }
    }

    let direct_rule = crate::rules::arithmetic::CollapseExactZeroThreeTermSubsetRule;
    if let Some(rewrite) = crate::rule::Rule::apply(&direct_rule, ctx, expr, &parent_ctx) {
        let zero = ctx.num(0);
        if compare_expr(ctx, rewrite.final_expr(), zero) == Ordering::Equal {
            return Some(finish_root_shortcut_with_rewrite_meta(
                ctx,
                expr,
                rewrite,
                "Collapse Exact Zero Additive Subexpression",
                collect_steps,
            ));
        }
    }

    if let Some(rewrite) = crate::rule::Rule::apply(&common_scale_rule, ctx, expr, &parent_ctx) {
        let zero = ctx.num(0);
        if compare_expr(ctx, rewrite.final_expr(), zero) == Ordering::Equal {
            return Some(finish_root_shortcut_with_rewrite_meta(
                ctx,
                expr,
                rewrite,
                "Collapse Common-Scale Equivalent Difference",
                collect_steps,
            ));
        }
    }

    if let Some(result) =
        try_standard_common_scale_exact_zero_shortcut_fallback(options, ctx, expr, collect_steps)
    {
        return Some(result);
    }

    None
}

fn run_rebuilt_root_shortcut_simplify(
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
        "Collapse Exact Zero Additive Subexpression",
        "Collapse Exact Zero Additive Subexpression",
        collect_steps,
    )
}

fn run_shifted_quotient_rebuilt_root_shortcut_simplify(
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
        "Collapse Shifted Quotient of Equivalent Expressions",
        "Collapse Shifted Quotient of Equivalent Expressions",
        collect_steps,
    )
}

fn run_common_scale_rebuilt_root_shortcut_simplify(
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

fn run_named_rebuilt_root_shortcut_simplify(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    before: ExprId,
    rewritten: ExprId,
    local_desc: &'static str,
    rule_name: &'static str,
    collect_steps: bool,
) -> (ExprId, Vec<Step>) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    std::mem::swap(&mut simplifier.context, ctx);
    let (result, inner_steps, _stats) = simplifier.simplify_with_stats(
        rewritten,
        crate::SimplifyOptions {
            suppress_depth_overflow_warnings: true,
            ..options.clone()
        },
    );
    std::mem::swap(&mut simplifier.context, ctx);

    let mut shortcut_steps = Vec::new();
    if collect_steps {
        let mut step = Step::new_compact(local_desc, rule_name, before, rewritten);
        step.global_before = Some(before);
        step.global_after = Some(rewritten);
        step.importance = crate::step::ImportanceLevel::High;
        shortcut_steps.push(step);
        shortcut_steps.extend(inner_steps);
    }

    (result, shortcut_steps)
}

fn isolated_simplify_expr_if_changed(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let mut simplifier = crate::Simplifier::with_default_rules();
    std::mem::swap(&mut simplifier.context, ctx);
    let (rewritten, _steps, _stats) = simplifier.simplify_with_stats(
        expr,
        crate::SimplifyOptions {
            collect_steps: false,
            suppress_depth_overflow_warnings: true,
            ..options.clone()
        },
    );
    std::mem::swap(&mut simplifier.context, ctx);
    (compare_expr(ctx, rewritten, expr) != Ordering::Equal).then_some(rewritten)
}

fn isolated_simplify_rewrites_to_target(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    target: ExprId,
) -> bool {
    let mut simplifier = crate::Simplifier::with_default_rules();
    std::mem::swap(&mut simplifier.context, ctx);
    let mut orchestrator = Orchestrator::new();
    orchestrator.options = SimplifyOptions {
        collect_steps: false,
        suppress_depth_overflow_warnings: true,
        ..options.clone()
    };
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    std::mem::swap(&mut simplifier.context, ctx);
    compare_expr(ctx, rewritten, target) == Ordering::Equal
}

fn isolated_simplify_rewrites_to_zero(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let zero = ctx.num(0);
    isolated_simplify_rewrites_to_target(options, ctx, expr, zero)
}

fn child_isolated_exact_zero(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    child: ExprId,
) -> bool {
    if !matches!(ctx.get(child), Expr::Add(_, _) | Expr::Sub(_, _)) {
        return false;
    }

    if expr_contains_trig_or_hyperbolic_builtin_local(ctx, child) {
        let term_count = AddView::from_expr(ctx, child).terms.len();
        if term_count > 4 {
            return false;
        }

        return isolated_simplify_rewrites_to_zero(options, ctx, child);
    }

    try_standard_exact_zero_equivalence_shortcut(options, ctx, child, false).is_some()
}

fn child_matches_direct_or_isolated_exact_zero(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    child: ExprId,
) -> bool {
    matches_direct_small_zero_identity_root(ctx, child)
        || matches_direct_hyperbolic_cosh_cubic_zero_identity_root(ctx, child)
        || child_isolated_exact_zero(options, ctx, child)
}

fn expr_contains_hyperbolic_builtin_local(ctx: &Context, expr: ExprId) -> bool {
    expr_contains_any_builtin_local(
        ctx,
        expr,
        &[BuiltinFn::Sinh, BuiltinFn::Cosh, BuiltinFn::Tanh],
    )
}

fn expr_contains_trig_builtin_local(ctx: &Context, expr: ExprId) -> bool {
    expr_contains_any_builtin_local(
        ctx,
        expr,
        &[
            BuiltinFn::Sin,
            BuiltinFn::Cos,
            BuiltinFn::Tan,
            BuiltinFn::Cot,
            BuiltinFn::Sec,
            BuiltinFn::Csc,
        ],
    )
}

fn expr_contains_reciprocal_trig_builtin_local(ctx: &Context, expr: ExprId) -> bool {
    expr_contains_any_builtin_local(
        ctx,
        expr,
        &[
            BuiltinFn::Tan,
            BuiltinFn::Cot,
            BuiltinFn::Sec,
            BuiltinFn::Csc,
        ],
    )
}

fn expr_contains_trig_or_hyperbolic_builtin_local(ctx: &Context, expr: ExprId) -> bool {
    expr_contains_trig_builtin_local(ctx, expr) || expr_contains_hyperbolic_builtin_local(ctx, expr)
}

fn expr_contains_log_builtin_local(ctx: &Context, expr: ExprId) -> bool {
    expr_contains_any_builtin_local(ctx, expr, &[BuiltinFn::Ln, BuiltinFn::Log, BuiltinFn::Abs])
}

fn is_supported_nested_zero_child_partner(ctx: &Context, expr: ExprId) -> bool {
    expr_contains_log_builtin_local(ctx, expr)
        || (matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _))
            && !expr_contains_trig_or_hyperbolic_builtin_local(ctx, expr)
            && !expr_contains_log_builtin_local(ctx, expr))
}

fn supported_nested_zero_partner_rewrites_to_zero(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    is_supported_nested_zero_child_partner(ctx, expr)
        && (matches_direct_small_zero_identity_root(ctx, expr)
            || isolated_simplify_rewrites_to_zero(options, ctx, expr))
}

fn is_small_trig_or_hyperbolic_zero_child(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    if !matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _))
        || !expr_contains_trig_or_hyperbolic_builtin_local(ctx, expr)
    {
        return false;
    }

    let terms = AddView::from_expr(ctx, expr).terms;
    let has_positive = terms.iter().any(|(_, sign)| *sign == Sign::Pos);
    let has_negative = terms.iter().any(|(_, sign)| *sign == Sign::Neg);
    if terms.len() > 4
        || !has_positive
        || !has_negative
        || !terms.iter().all(|(term, _)| {
            expr_contains_trig_or_hyperbolic_builtin_local(ctx, *term)
                || matches!(ctx.get(*term), Expr::Number(_))
        })
    {
        return false;
    }

    let mut isolated_ctx = ctx.clone();
    isolated_simplify_rewrites_to_zero(options, &mut isolated_ctx, expr)
}

fn try_standard_reciprocal_trig_zero_pair_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let matches_zero_pair = |ctx: &mut Context, lhs: ExprId, rhs: ExprId| {
        expr_contains_reciprocal_trig_builtin_local(ctx, lhs)
            && is_small_trig_or_hyperbolic_zero_child(options, ctx, lhs)
            && is_small_trig_or_hyperbolic_zero_child(options, ctx, rhs)
    };

    let matched = match ctx.get(expr) {
        Expr::Add(lhs, rhs) | Expr::Sub(lhs, rhs) | Expr::Mul(lhs, rhs) => {
            let lhs = *lhs;
            let rhs = *rhs;
            matches_zero_pair(ctx, lhs, rhs) || matches_zero_pair(ctx, rhs, lhs)
        }
        _ => false,
    };

    if !matched {
        return None;
    }

    let zero = ctx.num(0);
    Some(run_rebuilt_root_shortcut_simplify(
        options,
        ctx,
        expr,
        zero,
        collect_steps,
    ))
}

fn try_standard_small_trig_zero_pair_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let matches_small_zero_identity = |ctx: &mut Context, child: ExprId| {
        matches_direct_small_zero_identity_root(ctx, child)
            || matches_direct_hyperbolic_cosh_cubic_zero_identity_root(ctx, child)
    };

    let matches_zero_pair = |ctx: &mut Context, lhs: ExprId, rhs: ExprId| {
        matches_small_zero_identity(ctx, lhs)
            && (matches_small_zero_identity(ctx, rhs)
                || child_isolated_exact_zero(options, ctx, rhs))
    };

    let matched = match ctx.get(expr) {
        Expr::Add(lhs, rhs) | Expr::Sub(lhs, rhs) | Expr::Mul(lhs, rhs) => {
            let lhs = *lhs;
            let rhs = *rhs;
            matches_zero_pair(ctx, lhs, rhs) || matches_zero_pair(ctx, rhs, lhs)
        }
        _ => false,
    };

    if !matched {
        return None;
    }

    let zero = ctx.num(0);
    Some(run_rebuilt_root_shortcut_simplify(
        options,
        ctx,
        expr,
        zero,
        collect_steps,
    ))
}

fn try_standard_direct_trig_mixed_zero_pair_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let (lhs, rhs) = match ctx.get(expr) {
        Expr::Add(lhs, rhs) | Expr::Sub(lhs, rhs) | Expr::Mul(lhs, rhs) => (*lhs, *rhs),
        _ => return None,
    };

    let matches_zero_pair = |ctx: &mut Context, trig_side: ExprId, other_side: ExprId| {
        matches_narrow_trig_mixed_double_angle_zero_candidate_root(ctx, trig_side)
            && child_isolated_exact_zero(options, ctx, trig_side)
            && (matches_direct_small_zero_identity_root(ctx, other_side)
                || (is_small_trig_or_hyperbolic_zero_child(options, ctx, other_side)
                    && child_isolated_exact_zero(options, ctx, other_side)))
    };

    if !(matches_zero_pair(ctx, lhs, rhs) || matches_zero_pair(ctx, rhs, lhs)) {
        return None;
    }

    let zero = ctx.num(0);
    Some(run_rebuilt_root_shortcut_simplify(
        options,
        ctx,
        expr,
        zero,
        collect_steps,
    ))
}

fn try_standard_direct_half_angle_square_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let (trig_fn, arg) = extract_direct_half_angle_square_target_root(ctx, expr)?;
    let rewritten = build_plain_trig_pow2_root(ctx, trig_fn, arg);
    let rule_name = match trig_fn {
        BuiltinFn::Sin => "sin²(x/2) = (1 - cos(x))/2",
        BuiltinFn::Cos => "cos²(x/2) = (1 + cos(x))/2",
        _ => unreachable!("only sin/cos half-angle squares are supported"),
    };

    Some(run_named_rebuilt_root_shortcut_simplify(
        options,
        ctx,
        expr,
        rewritten,
        "Trig Half-Angle Squares",
        rule_name,
        collect_steps,
    ))
}

fn try_standard_direct_trig_power_reduction_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let (rewritten, rule_name) = if let Some(arg) =
        extract_direct_cos_fourth_power_reduction_target_root(ctx, expr)
    {
        (
            build_plain_trig_pow4_root(ctx, BuiltinFn::Cos, arg),
            "Power Reduction Identity",
        )
    } else if let Some(arg) = extract_direct_sin_cos_square_product_reduction_target_root(ctx, expr)
    {
        (
            build_plain_sin_cos_square_product_root(ctx, arg),
            "Power Reduction Identity",
        )
    } else {
        return None;
    };

    Some(run_named_rebuilt_root_shortcut_simplify(
        options,
        ctx,
        expr,
        rewritten,
        "Power Reduction Identity",
        rule_name,
        collect_steps,
    ))
}

fn try_standard_scaled_sin_fourth_power_reduction_zero_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let (lhs_core, rhs_core) =
        crate::rules::arithmetic::extract_two_term_core_difference(ctx, expr)?;

    for (scaled_term, reduced_term) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(scaled_arg) = extract_scaled_sin_fourth_power_target_root(ctx, scaled_term) else {
            continue;
        };
        let Some(reduced_arg) =
            extract_scaled_sin_fourth_power_reduction_target_root(ctx, reduced_term)
        else {
            continue;
        };
        if compare_expr(ctx, scaled_arg, reduced_arg) != Ordering::Equal {
            continue;
        }

        let zero = ctx.num(0);
        return Some(run_named_rebuilt_root_shortcut_simplify(
            options,
            ctx,
            expr,
            zero,
            "Power Reduction Identity",
            "Power Reduction Identity",
            collect_steps,
        ));
    }

    None
}

fn try_standard_direct_positive_double_cos_square_diff_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let arg = extract_direct_positive_double_cos_square_diff_target_root(ctx, expr)?;
    let rewritten = build_positive_cos_double_angle_expr_root(ctx, arg);

    Some(run_named_rebuilt_root_shortcut_simplify(
        options,
        ctx,
        expr,
        rewritten,
        "Double Angle Expansion",
        "Double Angle Expansion",
        collect_steps,
    ))
}

fn try_standard_direct_sum_to_product_root_shortcut(
    _options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    if !matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _)) {
        return None;
    }

    let rewrite = try_rewrite_sum_to_product_contraction_expr(ctx, expr)?;
    let rewritten = match rewrite.kind {
        cas_math::trig_sum_product_support::TrigSumToProductContractionRewriteKind::SinSum
        | cas_math::trig_sum_product_support::TrigSumToProductContractionRewriteKind::SinDiff => {
            if let Some((sin_arg, cos_arg)) =
                extract_scaled_trig_sin_cos_product_args_root(ctx, rewrite.rewritten)
            {
                let canonical_cos_arg = canonicalize_even_cos_arg_root(ctx, cos_arg);
                build_scaled_trig_sin_cos_product_root(ctx, sin_arg, canonical_cos_arg)
            } else {
                rewrite.rewritten
            }
        }
        _ => rewrite.rewritten,
    };
    let mut shortcut_steps = Vec::new();
    if collect_steps {
        shortcut_steps.push(build_root_shortcut_compact_step(
            expr,
            rewritten,
            "Aplicar suma a producto",
            "Sum-to-Product Identity",
        ));
    }
    Some((rewritten, shortcut_steps))
}

fn canonicalize_sum_to_product_contraction_target_root(
    ctx: &mut Context,
    rewrite: cas_math::trig_sum_product_support::TrigSumToProductContractionRewrite,
) -> ExprId {
    match rewrite.kind {
        cas_math::trig_sum_product_support::TrigSumToProductContractionRewriteKind::SinSum
        | cas_math::trig_sum_product_support::TrigSumToProductContractionRewriteKind::SinDiff => {
            if let Some((sin_arg, cos_arg)) =
                extract_scaled_trig_sin_cos_product_args_root(ctx, rewrite.rewritten)
            {
                let canonical_cos_arg = canonicalize_even_cos_arg_root(ctx, cos_arg);
                build_scaled_trig_sin_cos_product_root(ctx, sin_arg, canonical_cos_arg)
            } else {
                rewrite.rewritten
            }
        }
        _ => rewrite.rewritten,
    }
}

fn rewrites_sum_to_product_target_root(
    ctx: &mut Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> bool {
    let Some(rewrite) = try_rewrite_sum_to_product_contraction_expr(ctx, source_expr) else {
        return false;
    };
    let rewritten = canonicalize_sum_to_product_contraction_target_root(ctx, rewrite);
    compare_expr(ctx, rewritten, target_expr) == Ordering::Equal
}

fn rewrites_product_to_sum_target_root(
    ctx: &mut Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> bool {
    let Some(rewrite) = try_rewrite_product_to_sum_expr(ctx, source_expr) else {
        return false;
    };
    let rewritten = rewrite_direct_trig_product_to_sum_double_angle_target_root(ctx, source_expr)
        .unwrap_or(rewrite.rewritten);
    compare_expr(ctx, rewritten, target_expr) == Ordering::Equal
}

fn extract_direct_trig_sum_product_zero_cores_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId)> {
    if let Some((lhs_core, rhs_core)) =
        extract_shared_additive_passthrough_sub_cores_root(ctx, expr)
    {
        return Some((lhs_core, rhs_core));
    }

    if let Some((lhs_core, rhs_core)) =
        crate::rules::arithmetic::extract_two_term_core_difference(ctx, expr)
    {
        return Some((lhs_core, rhs_core));
    }

    let terms = AddView::from_expr(ctx, expr).terms;
    if !(3..=6).contains(&terms.len()) {
        return None;
    }

    let full_mask = (1usize << terms.len()) - 1;
    let mut product_to_sum_fallback = None;
    for left_mask in 1..full_mask {
        let right_mask = full_mask ^ left_mask;
        if right_mask == 0 {
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

        let lhs_chunk = build_signed_sum_expr_root(ctx, &left_terms);
        let rhs_chunk = build_signed_sum_expr_root(ctx, &right_terms);
        if is_plain_two_term_sin_cos_sum_or_diff_root(ctx, lhs_chunk)
            && is_trig_sum_product_candidate_root(ctx, rhs_chunk)
        {
            return Some((lhs_chunk, rhs_chunk));
        }
        if product_to_sum_fallback.is_none()
            && is_trig_sum_product_candidate_root(ctx, lhs_chunk)
            && is_plain_two_term_sin_cos_sum_or_diff_root(ctx, rhs_chunk)
        {
            product_to_sum_fallback = Some((lhs_chunk, rhs_chunk));
        }
    }

    product_to_sum_fallback
}

fn is_plain_two_term_sin_cos_sum_or_diff_root(ctx: &mut Context, expr: ExprId) -> bool {
    let terms = AddView::from_expr(ctx, expr).terms;
    terms.len() == 2
        && terms.iter().all(|(term_expr, _)| {
            let (_coeff, base) = extract_coef_and_base(ctx, *term_expr);
            extract_plain_sin_or_cos_arg_root(ctx, base).is_some()
        })
}

fn is_trig_sum_product_candidate_root(ctx: &mut Context, expr: ExprId) -> bool {
    let (_coeff, base) = extract_coef_and_base(ctx, expr);
    let mut trig_factor_count = 0usize;
    for factor in flatten_mul_chain(ctx, base) {
        if extract_plain_sin_or_cos_arg_root(ctx, factor).is_some() {
            trig_factor_count += 1;
            continue;
        }
        if matches!(ctx.get(factor), Expr::Number(_)) {
            continue;
        }
        return false;
    }

    trig_factor_count >= 2
}

fn try_standard_direct_trig_sum_product_zero_shortcut(
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let (lhs_core, rhs_core) = extract_direct_trig_sum_product_zero_cores_root(ctx, expr)?;
    if is_plain_two_term_sin_cos_sum_or_diff_root(ctx, lhs_core)
        && is_trig_sum_product_candidate_root(ctx, rhs_core)
    {
        let zero = ctx.num(0);
        return Some(finish_standard_root_shortcut(
            ctx,
            expr,
            crate::rule::Rewrite::with_local(zero, "Aplicar suma a producto", expr, zero),
            "Aplicar suma a producto",
            collect_steps,
        ));
    }
    if is_trig_sum_product_candidate_root(ctx, lhs_core)
        && is_plain_two_term_sin_cos_sum_or_diff_root(ctx, rhs_core)
    {
        let zero = ctx.num(0);
        return Some(finish_standard_root_shortcut(
            ctx,
            expr,
            crate::rule::Rewrite::with_local(zero, "Aplicar producto a suma", expr, zero),
            "Aplicar producto a suma",
            collect_steps,
        ));
    }

    let whole_expr_is_direct_trig_sum_product_zero =
        matches_direct_symbolic_trig_sum_to_product_zero_identity_root(ctx, expr)
            || matches_direct_trig_product_to_sum_zero_identity_root(ctx, expr);
    let lhs_has_sum_to_product_rewrite =
        try_rewrite_sum_to_product_contraction_expr(ctx, lhs_core).is_some();
    let lhs_has_product_to_sum_rewrite = try_rewrite_product_to_sum_expr(ctx, lhs_core).is_some()
        || rewrite_direct_trig_product_to_sum_double_angle_target_root(ctx, lhs_core).is_some();

    let rule_name = if rewrites_sum_to_product_target_root(ctx, lhs_core, rhs_core)
        || (whole_expr_is_direct_trig_sum_product_zero && lhs_has_sum_to_product_rewrite)
    {
        "Aplicar suma a producto"
    } else if rewrites_product_to_sum_target_root(ctx, lhs_core, rhs_core)
        || (whole_expr_is_direct_trig_sum_product_zero && lhs_has_product_to_sum_rewrite)
    {
        "Aplicar producto a suma"
    } else {
        return None;
    };

    let zero = ctx.num(0);
    Some(finish_standard_root_shortcut(
        ctx,
        expr,
        crate::rule::Rewrite::with_local(zero, rule_name, expr, zero),
        rule_name,
        collect_steps,
    ))
}

fn try_standard_half_angle_square_factor_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let _ = options;
    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() < 2 {
        return None;
    }

    for (index, factor) in factors.iter().copied().enumerate() {
        let Some((trig_fn, arg)) = extract_direct_half_angle_square_target_root(ctx, factor) else {
            continue;
        };

        let mut rewritten_factors = factors.clone();
        rewritten_factors[index] = build_plain_trig_pow2_root(ctx, trig_fn, arg);
        let rewritten = build_mul_expr_from_factors_root(ctx, &rewritten_factors);
        let rule_name = match trig_fn {
            BuiltinFn::Sin => "sin²(x/2) = (1 - cos(x))/2",
            BuiltinFn::Cos => "cos²(x/2) = (1 + cos(x))/2",
            _ => unreachable!("only sin/cos half-angle squares are supported"),
        };

        return Some(run_named_rebuilt_root_shortcut_simplify(
            options,
            ctx,
            expr,
            rewritten,
            "Trig Half-Angle Squares",
            rule_name,
            collect_steps,
        ));
    }

    None
}

fn canonicalize_direct_pair_factor_root(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    if let Some(rewritten) = extract_special_angle_exact_value_root(ctx, expr) {
        return Some(rewritten);
    }
    if cas_ast::collect_variables(ctx, expr).is_empty() && cas_ast::count_nodes(ctx, expr) <= 16 {
        if let Some(rewritten) =
            isolated_simplify_expr_if_changed(&crate::phase::SimplifyOptions::default(), ctx, expr)
        {
            return Some(strip_multiplicative_one_root(ctx, rewritten));
        }
    }
    if let Some(rewrite) = try_rewrite_trig_phase_shift_function_expr(ctx, expr) {
        let normalized = strip_multiplicative_one_root(ctx, rewrite.rewritten);
        return Some(canonicalize_even_cos_in_simple_expr_root(ctx, normalized));
    }
    if let Some(plan) =
        cas_math::inverse_trig_composition_support::try_plan_inverse_trig_composition_expr(
            ctx, expr, false, false,
        )
    {
        return Some(strip_multiplicative_one_root(ctx, plan.rewritten));
    }
    if let Some(rewrite) = try_rewrite_trig_inverse_composition_expr(ctx, expr) {
        return Some(strip_multiplicative_one_root(ctx, rewrite.rewritten));
    }
    if let Some(rewrite) = try_rewrite_exponential_log_inverse_expr(ctx, expr) {
        return Some(strip_multiplicative_one_root(ctx, rewrite.rewritten));
    }
    if let Some((trig_fn, full_arg)) = extract_direct_abs_trig_half_angle_target_root(ctx, expr) {
        return Some(build_direct_sqrt_abs_trig_half_angle_target_root(
            ctx, trig_fn, full_arg,
        ));
    }
    if let Some(log_inverse_match) =
        cas_math::logarithm_inverse_support::try_match_log_exp_inverse_expr(ctx, expr)
    {
        match log_inverse_match {
            cas_math::logarithm_inverse_support::LogExpInverseMatch::Numeric {
                rewritten, ..
            } => {
                return Some(strip_multiplicative_one_root(ctx, rewritten));
            }
            cas_math::logarithm_inverse_support::LogExpInverseMatch::Symbolic {
                base,
                exponent,
            } => {
                let e = ctx.add(Expr::Constant(Constant::E));
                if compare_expr(ctx, base, e) == Ordering::Equal {
                    return Some(strip_multiplicative_one_root(ctx, exponent));
                }
            }
        }
    }
    if let Some(base) = extract_addition_of_successive_unit_fractions_arg_root(ctx, expr) {
        return Some(build_collapsed_successive_unit_fractions_expr_root(
            ctx, base,
        ));
    }
    if let Some(base) = extract_consecutive_telescoping_fraction_difference_arg_root(ctx, expr) {
        return Some(build_consecutive_telescoping_fraction_difference_expr_root(
            ctx, base,
        ));
    }
    if let Some(base) = extract_collapsed_successive_unit_fractions_arg_root(ctx, expr) {
        return Some(build_collapsed_successive_unit_fractions_expr_root(
            ctx, base,
        ));
    }
    if let Some(denominator) = extract_unit_fraction_denominator_root(ctx, expr) {
        if let Some(base) = extract_consecutive_product_core_root(ctx, denominator) {
            return Some(build_consecutive_telescoping_fraction_difference_expr_root(
                ctx, base,
            ));
        }
    }
    if let Some(factored) = cas_math::factor::factor_perfect_square_trinomial(ctx, expr) {
        return Some(factored);
    }
    if let Some(rewrite) = try_rewrite_simplify_square_root_expr(ctx, expr) {
        return Some(strip_multiplicative_one_root(ctx, rewrite.rewritten));
    }
    if let Some(canonical) = try_rewrite_canonical_root_expr(ctx, expr) {
        if let Some(extract) =
            try_rewrite_extract_perfect_power_from_radicand_expr(ctx, canonical.rewritten)
        {
            return Some(strip_multiplicative_one_root(ctx, extract.rewritten));
        }
        return Some(strip_multiplicative_one_root(ctx, canonical.rewritten));
    }
    if let Some(numerator) = extract_div_by_two_numerator_root(ctx, expr) {
        if let Some(rewrite) = try_rewrite_sum_to_product_contraction_expr(ctx, numerator) {
            if rewrite.kind
                == cas_math::trig_sum_product_support::TrigSumToProductContractionRewriteKind::SinSum
            {
                if let Some((sin_arg, cos_arg)) =
                    extract_scaled_trig_sin_cos_product_args_root(ctx, rewrite.rewritten)
                {
                    let canonical_cos_arg = canonicalize_even_cos_arg_root(ctx, cos_arg);
                    return Some(build_plain_trig_sin_cos_product_root(
                        ctx,
                        sin_arg,
                        canonical_cos_arg,
                    ));
                }
            }
        }
    }
    if let Some(rewrite) = try_rewrite_sum_to_product_contraction_expr(ctx, expr) {
        match rewrite.kind {
            cas_math::trig_sum_product_support::TrigSumToProductContractionRewriteKind::SinSum
            | cas_math::trig_sum_product_support::TrigSumToProductContractionRewriteKind::SinDiff => {
                if let Some((sin_arg, cos_arg)) =
                    extract_scaled_trig_sin_cos_product_args_root(ctx, rewrite.rewritten)
                {
                    let canonical_cos_arg = canonicalize_even_cos_arg_root(ctx, cos_arg);
                    return Some(build_scaled_trig_sin_cos_product_root(
                        ctx,
                        sin_arg,
                        canonical_cos_arg,
                    ));
                }
            }
            _ => {}
        }
    }
    if let Some((sin_arg, cos_arg)) = extract_scaled_trig_sin_cos_product_args_root(ctx, expr) {
        let canonical_cos_arg = canonicalize_even_cos_arg_root(ctx, cos_arg);
        return Some(build_scaled_trig_sin_cos_product_root(
            ctx,
            sin_arg,
            canonical_cos_arg,
        ));
    }
    if let Some(rewritten) = rewrite_direct_trig_product_to_sum_double_angle_target_root(ctx, expr)
    {
        return Some(rewritten);
    }
    if let Some(rewrite) = try_rewrite_product_to_sum_expr(ctx, expr) {
        return Some(rewrite.rewritten);
    }
    if let Some(rewrite) = try_rewrite_angle_sum_fraction_to_tan_expr(ctx, expr) {
        return Some(rewrite.rewritten);
    }
    if let Some((lhs_arg, rhs_arg)) = extract_direct_tan_angle_sum_target_root(ctx, expr) {
        return Some(build_tan_angle_sum_fraction_root(ctx, lhs_arg, rhs_arg));
    }
    if let Some(rewrite) = try_rewrite_tan_to_sin_cos_function_expr(ctx, expr) {
        return Some(rewrite.rewritten);
    }
    if let Some(rewrite) = try_rewrite_double_angle_function_expr(ctx, expr) {
        return Some(rewrite.rewritten);
    }
    if let Some((lhs_arg, rhs_arg)) = extract_direct_tangent_addition_target_root(ctx, expr) {
        return Some(build_tangent_addition_fraction_root(ctx, lhs_arg, rhs_arg));
    }
    if let Some(rewrite) = try_rewrite_triple_angle_expr(ctx, expr) {
        return Some(rewrite.rewritten);
    }
    if let Some(rewrite) = try_rewrite_hyperbolic_triple_angle(ctx, expr) {
        return Some(rewrite.rewritten);
    }
    if let Some(rewrite) = try_rewrite_hyperbolic_double_angle_sum(ctx, expr) {
        return Some(rewrite.rewritten);
    }
    if let Some((arg, is_sum)) = extract_direct_hyperbolic_exp_sum_target_root(ctx, expr) {
        return Some(build_direct_hyperbolic_exp_sum_target_root(
            ctx, arg, is_sum,
        ));
    }
    if let Some(rewrite) = try_rewrite_recognize_hyperbolic_from_exp(ctx, expr) {
        return Some(rewrite.rewritten);
    }
    if let Some(arg) = extract_direct_tanh_pythagorean_identity_arg_root(ctx, expr) {
        return Some(build_tanh_pythagorean_target_root(ctx, arg));
    }
    if let Some(rewritten) = try_rewrite_tanh_double_angle_expansion(ctx, expr) {
        return Some(rewritten);
    }
    if let Some(rewritten) = try_rewrite_tanh_to_sinh_cosh(ctx, expr) {
        return Some(rewritten);
    }
    if let Some(rewrite) =
        cas_math::root_den_rationalize_support::try_rewrite_rationalize_cube_root_den_expr(
            ctx, expr,
        )
    {
        return Some(strip_multiplicative_one_root(ctx, rewrite.rewritten));
    }
    if let Some((hyperbolic_fn, arg)) =
        extract_direct_hyperbolic_half_angle_square_target_root(ctx, expr)
    {
        return Some(build_plain_hyperbolic_half_angle_pow2_root(
            ctx,
            hyperbolic_fn,
            arg,
        ));
    }
    if let Some(arg) = extract_scaled_double_angle_sin_square_target_root(ctx, expr) {
        return Some(build_plain_sin_cos_square_product_root(ctx, arg));
    }
    if let Some(arg) = extract_direct_positive_double_cos_square_diff_target_root(ctx, expr) {
        return Some(build_positive_cos_double_angle_expr_root(ctx, arg));
    }
    if let Some(arg) = extract_direct_cos_fourth_power_reduction_target_root(ctx, expr) {
        return Some(build_plain_trig_pow4_root(ctx, BuiltinFn::Cos, arg));
    }
    if let Some(rewritten) = rewrite_sum_of_squares_product_root(ctx, expr) {
        return Some(rewritten);
    }
    None
}

fn canonicalize_even_cos_in_simple_expr_root(ctx: &mut Context, expr: ExprId) -> ExprId {
    match ctx.get(expr).clone() {
        Expr::Neg(inner) => {
            let normalized_inner = canonicalize_even_cos_in_simple_expr_root(ctx, inner);
            if compare_expr(ctx, normalized_inner, inner) == Ordering::Equal {
                expr
            } else {
                ctx.add(Expr::Neg(normalized_inner))
            }
        }
        _ => {
            let Some((BuiltinFn::Cos, arg)) = extract_plain_sin_or_cos_arg_root(ctx, expr) else {
                return expr;
            };
            let normalized_arg = canonicalize_even_cos_arg_root(ctx, arg);
            if compare_expr(ctx, normalized_arg, arg) == Ordering::Equal {
                expr
            } else {
                ctx.call_builtin(BuiltinFn::Cos, vec![normalized_arg])
            }
        }
    }
}

fn try_standard_collapsed_fraction_direct_pair_factor_shortcut(
    _options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let factors = flatten_mul_chain(ctx, expr);
    if !(2..=4).contains(&factors.len()) {
        return None;
    }

    for partner_index in 0..factors.len() {
        let Some(partner_canonical) =
            canonicalize_direct_pair_factor_root(ctx, factors[partner_index])
        else {
            continue;
        };
        if extract_collapsed_successive_unit_fractions_arg_root(ctx, partner_canonical).is_none() {
            continue;
        }
        let remaining_factors: Vec<_> = factors
            .iter()
            .enumerate()
            .filter_map(|(index, factor)| (index != partner_index).then_some(*factor))
            .collect();
        let combined_factor = build_mul_expr_from_factors_root(ctx, &remaining_factors);
        let Some(factor_canonical) = canonicalize_direct_pair_factor_root(ctx, combined_factor)
        else {
            continue;
        };

        let partner_changed =
            compare_expr(ctx, partner_canonical, factors[partner_index]) != Ordering::Equal;
        let factor_changed =
            compare_expr(ctx, factor_canonical, combined_factor) != Ordering::Equal;
        if !partner_changed && !factor_changed {
            continue;
        }

        let rewritten =
            build_mul_expr_from_factors_root(ctx, &[partner_canonical, factor_canonical]);
        let rewrite = crate::rule::Rewrite::new(rewritten).desc("Canonical Direct Pair Product");
        return Some(finish_standard_root_shortcut(
            ctx,
            expr,
            rewrite,
            "Canonical Direct Pair Product",
            collect_steps,
        ));
    }

    None
}

fn factor_sum_diff_cubes_partner_root(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 2 {
        return None;
    }

    let mut positive_bases = Vec::with_capacity(2);
    let mut negative_bases = Vec::with_capacity(1);
    for (term_expr, term_sign) in view.terms {
        let base = extract_plain_cube_base_root(ctx, term_expr)?;
        match term_sign {
            Sign::Pos => positive_bases.push(base),
            Sign::Neg => negative_bases.push(base),
        }
    }

    let (a, b, is_difference) = match (&positive_bases[..], &negative_bases[..]) {
        ([a, b], []) => (*a, *b, false),
        ([a], [b]) => (*a, *b, true),
        _ => return None,
    };

    let two = ctx.num(2);
    let a_sq = ctx.add(Expr::Pow(a, two));
    let b_sq = ctx.add(Expr::Pow(b, two));
    let ab = smart_mul(ctx, a, b);
    let trinomial = if is_difference {
        let inner = ctx.add(Expr::Add(ab, b_sq));
        ctx.add(Expr::Add(a_sq, inner))
    } else {
        let neg_ab = ctx.add(Expr::Neg(ab));
        let inner = ctx.add(Expr::Add(neg_ab, b_sq));
        ctx.add(Expr::Add(a_sq, inner))
    };
    let binomial = if is_difference {
        ctx.add(Expr::Sub(a, b))
    } else {
        ctx.add(Expr::Add(a, b))
    };

    Some(build_mul_expr_from_factors_root(
        ctx,
        &[binomial, trinomial],
    ))
}

fn factor_higher_degree_difference_partner_root(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let (lhs, rhs) = match ctx.get(expr).clone() {
        Expr::Sub(lhs, rhs) => (lhs, rhs),
        _ => return None,
    };
    if extract_i64_integer(ctx, rhs) != Some(1) {
        return None;
    }
    let (base, exponent) = match ctx.get(lhs).clone() {
        Expr::Pow(base, exponent) => (base, exponent),
        _ => return None,
    };
    if extract_i64_integer(ctx, exponent) != Some(6) {
        return None;
    }

    let one = ctx.num(1);
    let plus_one = ctx.add(Expr::Add(base, one));
    let minus_one = ctx.add(Expr::Sub(base, one));
    let two = ctx.num(2);
    let base_sq = ctx.add(Expr::Pow(base, two));
    let positive_quad = build_balanced_add(ctx, &[base_sq, base, one]);
    let negative_base = ctx.add(Expr::Neg(base));
    let negative_quad = build_balanced_add(ctx, &[base_sq, negative_base, one]);
    Some(build_mul_expr_from_factors_root(
        ctx,
        &[positive_quad, negative_quad, plus_one, minus_one],
    ))
}

fn factor_sophie_germain_partner_root(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let factored = cas_math::factor::factor_sophie_germain(ctx, expr)?;
    (compare_expr(ctx, factored, expr) != Ordering::Equal).then_some(factored)
}

fn try_standard_collapsed_fraction_factored_numerator_shortcut(
    _options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() != 2 {
        return None;
    }

    for fraction_index in 0..2 {
        let partner_index = 1 - fraction_index;
        let Some(denominator) =
            extract_unit_fraction_denominator_root(ctx, factors[fraction_index])
        else {
            continue;
        };
        if extract_consecutive_product_core_root(ctx, denominator).is_none() {
            continue;
        }

        let partner = factors[partner_index];
        let partner_factored = factor_sum_diff_cubes_partner_root(ctx, partner)
            .or_else(|| factor_small_linear_shift_product_partner_root(ctx, partner))
            .or_else(|| factor_known_small_polynomial_partner_root(ctx, partner));
        let Some(partner_factored) = partner_factored else {
            continue;
        };
        if compare_expr(ctx, partner_factored, partner) == Ordering::Equal {
            continue;
        }

        let rewritten = ctx.add(Expr::Div(partner_factored, denominator));
        let rewrite = crate::rule::Rewrite::new(rewritten)
            .desc("Canonizar numerador factorizable sobre fracción consecutiva colapsada");
        return Some(finish_standard_root_shortcut(
            ctx,
            expr,
            rewrite,
            "Collapsed Fraction Factored Numerator",
            collect_steps,
        ));
    }

    None
}

fn try_standard_collapsed_fraction_partner_canonicalization_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() != 2 {
        return None;
    }

    for fraction_index in 0..2 {
        let partner_index = 1 - fraction_index;
        let Some(denominator) =
            extract_unit_fraction_denominator_root(ctx, factors[fraction_index])
        else {
            continue;
        };
        if extract_consecutive_product_core_root(ctx, denominator).is_none() {
            continue;
        }

        let partner = factors[partner_index];
        let partner_canonical = canonicalize_direct_pair_factor_root(ctx, partner)
            .or_else(|| factor_small_linear_shift_product_partner_root(ctx, partner))
            .or_else(|| factor_known_small_polynomial_partner_root(ctx, partner))
            .or_else(|| {
                let rewrite = try_rewrite_automatic_factor_expr(ctx, partner)?;
                let factored = strip_multiplicative_one_root(ctx, rewrite.rewritten);
                (compare_expr(ctx, factored, partner) != Ordering::Equal).then_some(factored)
            });

        let Some(partner_canonical) = partner_canonical else {
            continue;
        };
        if compare_expr(ctx, partner_canonical, partner) == Ordering::Equal {
            continue;
        }

        let rewritten = if fraction_index == 0 {
            ctx.add(Expr::Mul(factors[fraction_index], partner_canonical))
        } else {
            ctx.add(Expr::Mul(partner_canonical, factors[fraction_index]))
        };
        return Some(run_named_rebuilt_root_shortcut_simplify(
            options,
            ctx,
            expr,
            rewritten,
            "Canonical Collapsed Fraction Partner",
            "Canonical Collapsed Fraction Partner",
            collect_steps,
        ));
    }

    None
}

fn try_standard_collapsed_fraction_hyperbolic_half_angle_factor_shortcut(
    _options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() != 2 {
        return None;
    }

    for (partner_index, factor_index) in [(0usize, 1usize), (1usize, 0usize)] {
        let partner = factors[partner_index];
        let partner_collapsed = if let Some(base) =
            extract_addition_of_successive_unit_fractions_arg_root(ctx, partner)
        {
            build_collapsed_successive_unit_fractions_expr_root(ctx, base)
        } else if let Some(base) =
            extract_collapsed_successive_unit_fractions_arg_root(ctx, partner)
        {
            build_collapsed_successive_unit_fractions_expr_root(ctx, base)
        } else {
            continue;
        };

        let Some((hyperbolic_fn, half_arg)) =
            extract_plain_sinh_or_cosh_pow2_arg_root(ctx, factors[factor_index])
        else {
            continue;
        };
        let Some(arg) = extract_half_scaled_base_root(ctx, half_arg) else {
            continue;
        };
        let target_factor =
            build_direct_hyperbolic_half_angle_square_target_root(ctx, hyperbolic_fn, arg);

        let rewritten = if partner_index == 0 {
            build_mul_expr_from_factors_root(ctx, &[partner_collapsed, target_factor])
        } else {
            build_mul_expr_from_factors_root(ctx, &[target_factor, partner_collapsed])
        };
        let rewrite = crate::rule::Rewrite::new(rewritten)
            .desc("Canonizar producto de fracción consecutiva con media-ángulo hiperbólico");
        return Some(finish_standard_root_shortcut(
            ctx,
            expr,
            rewrite,
            "Canonical Hyperbolic Half-Angle Product",
            collect_steps,
        ));
    }

    None
}

fn try_standard_tangent_addition_factor_shortcut(
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

fn try_standard_tangent_addition_fraction_product_shortcut(
    _options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() < 2 {
        return None;
    }

    for (index, factor) in factors.iter().copied().enumerate() {
        if extract_direct_tangent_addition_fraction_target_root(ctx, factor).is_none() {
            continue;
        }
        let Expr::Div(numerator, denominator) = ctx.get(factor) else {
            continue;
        };
        let numerator = *numerator;
        let denominator = *denominator;

        let rewritten_numerator_factors = factors
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(factor_index, other_factor)| {
                (factor_index != index).then_some(other_factor)
            })
            .chain(std::iter::once(numerator))
            .collect::<Vec<_>>();
        let rewritten_numerator =
            build_mul_expr_from_factors_root(ctx, &rewritten_numerator_factors);
        let rewritten = ctx.add(Expr::Div(rewritten_numerator, denominator));
        let rewrite = crate::rule::Rewrite::new(rewritten)
            .desc("Colapsar producto sobre fracción de suma de tangentes");
        return Some(finish_standard_root_shortcut(
            ctx,
            expr,
            rewrite,
            "Tangent Addition Fraction Product",
            collect_steps,
        ));
    }

    None
}

fn try_standard_trig_product_to_sum_subset_factor_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() < 4 {
        return None;
    }

    for i in 0..factors.len() {
        for j in (i + 1)..factors.len() {
            for k in (j + 1)..factors.len() {
                let subset =
                    build_mul_expr_from_factors_root(ctx, &[factors[i], factors[j], factors[k]]);
                let Some(rewrite) = try_rewrite_product_to_sum_expr(ctx, subset) else {
                    continue;
                };
                if !matches!(
                    rewrite.kind,
                    cas_math::trig_sum_product_support::TrigProductToSumRewriteKind::SinCos
                        | cas_math::trig_sum_product_support::TrigProductToSumRewriteKind::CosSin
                ) {
                    continue;
                }
                let rewritten_subset =
                    rewrite_direct_trig_product_to_sum_double_angle_target_root(ctx, subset)
                        .unwrap_or(rewrite.rewritten);

                let mut remaining_indices = Vec::with_capacity(factors.len() - 3);
                for index in 0..factors.len() {
                    if index != i && index != j && index != k {
                        remaining_indices.push(index);
                    }
                }

                let remaining_factors: Vec<_> = remaining_indices
                    .iter()
                    .map(|&index| factors[index])
                    .collect();
                let combined_partner = build_mul_expr_from_factors_root(ctx, &remaining_factors);

                if let Some(partner_canonical) =
                    canonicalize_direct_pair_factor_root(ctx, combined_partner)
                {
                    if compare_expr(ctx, partner_canonical, combined_partner) != Ordering::Equal {
                        let rewritten =
                            build_nonexpanding_locally_simplified_mul_expr_from_factors_root(
                                ctx,
                                &[rewritten_subset, partner_canonical],
                            );
                        let rewrite = crate::rule::Rewrite::with_local(
                            rewritten,
                            "Product-to-Sum Combined Partner",
                            expr,
                            rewritten,
                        );
                        return Some(finish_standard_root_shortcut(
                            ctx,
                            expr,
                            rewrite,
                            "Product-to-Sum Combined Partner",
                            collect_steps,
                        ));
                    }
                }

                if remaining_indices.len() == 1 {
                    if let Some(partner_simplified) =
                        isolated_simplify_expr_if_changed(options, ctx, combined_partner)
                    {
                        let rewritten =
                            build_nonexpanding_locally_simplified_mul_expr_from_factors_root(
                                ctx,
                                &[rewritten_subset, partner_simplified],
                            );
                        let rewrite = crate::rule::Rewrite::with_local(
                            rewritten,
                            "Product-to-Sum Simplified Partner",
                            expr,
                            rewritten,
                        );
                        return Some(finish_standard_root_shortcut(
                            ctx,
                            expr,
                            rewrite,
                            "Product-to-Sum Simplified Partner",
                            collect_steps,
                        ));
                    }
                }

                for partner_index in remaining_indices.iter().copied() {
                    let Some(partner_canonical) =
                        canonicalize_direct_pair_factor_root(ctx, factors[partner_index])
                    else {
                        continue;
                    };
                    if compare_expr(ctx, partner_canonical, factors[partner_index])
                        == Ordering::Equal
                    {
                        continue;
                    }

                    let mut rewritten_factors = Vec::with_capacity(factors.len() - 2);
                    for (index, factor) in factors.iter().copied().enumerate() {
                        if index == i {
                            rewritten_factors.push(rewritten_subset);
                        } else if index == j || index == k {
                            continue;
                        } else if index == partner_index {
                            rewritten_factors.push(partner_canonical);
                        } else {
                            rewritten_factors.push(factor);
                        }
                    }

                    let rewritten =
                        build_nonexpanding_locally_simplified_mul_expr_from_factors_root(
                            ctx,
                            &rewritten_factors,
                        );
                    let rewrite = crate::rule::Rewrite::with_local(
                        rewritten,
                        "Product-to-Sum Direct Pair Factor",
                        expr,
                        rewritten,
                    );
                    return Some(finish_standard_root_shortcut(
                        ctx,
                        expr,
                        rewrite,
                        "Product-to-Sum Direct Pair Factor",
                        collect_steps,
                    ));
                }

                let mut rewritten_factors = Vec::with_capacity(factors.len() - 2);
                for (index, factor) in factors.iter().copied().enumerate() {
                    if index == i {
                        rewritten_factors.push(rewritten_subset);
                    } else if index == j || index == k {
                        continue;
                    } else {
                        rewritten_factors.push(factor);
                    }
                }

                let rewritten = build_nonexpanding_locally_simplified_mul_expr_from_factors_root(
                    ctx,
                    &rewritten_factors,
                );
                let rewrite = crate::rule::Rewrite::with_local(
                    rewritten,
                    "Product-to-Sum Factor",
                    expr,
                    rewritten,
                );
                return Some(finish_standard_root_shortcut(
                    ctx,
                    expr,
                    rewrite,
                    "Product-to-Sum Factor",
                    collect_steps,
                ));
            }
        }
    }

    None
}

fn try_standard_sum_of_squares_product_subset_factor_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() < 3 {
        return None;
    }

    for i in 0..factors.len() {
        for j in (i + 1)..factors.len() {
            let subset = build_mul_expr_from_factors_root(ctx, &[factors[i], factors[j]]);
            let Some(rewritten_subset) = rewrite_sum_of_squares_product_root(ctx, subset) else {
                continue;
            };
            if compare_expr(ctx, subset, rewritten_subset) == Ordering::Equal {
                continue;
            }
            let mut rewritten_factors = Vec::with_capacity(factors.len() - 1);
            rewritten_factors.push(rewritten_subset);
            for (index, factor) in factors.iter().copied().enumerate() {
                if index != i && index != j {
                    rewritten_factors.push(factor);
                }
            }
            let rewritten = build_mul_expr_from_factors_root(ctx, &rewritten_factors);
            return Some(run_named_rebuilt_root_shortcut_simplify(
                options,
                ctx,
                expr,
                rewritten,
                "Sum of Squares Product Factor",
                "Sum of Squares Product Factor",
                collect_steps,
            ));
        }
    }

    None
}

fn try_standard_perfect_square_trinomial_factor_shortcut(
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
        let Some(squared_factor) = cas_math::factor::factor_perfect_square_trinomial(ctx, factor)
            .or_else(|| {
                let view = AddView::from_expr(ctx, factor);
                build_direct_perfect_square_from_terms_root(ctx, &view.terms)
            })
        else {
            continue;
        };
        if squared_factor == factor {
            continue;
        }

        let remaining_factors: Vec<_> = factors
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(other_index, factor_expr)| (other_index != index).then_some(factor_expr))
            .collect();
        let combined_partner = build_mul_expr_from_factors_root(ctx, &remaining_factors);

        if extract_square_power_base_root(ctx, squared_factor).is_some() {
            if let Some(partner_linear_shift) =
                factor_small_linear_shift_product_partner_root(ctx, combined_partner)
            {
                let rewritten = ctx.add(Expr::Mul(squared_factor, partner_linear_shift));
                let rewrite = crate::rule::Rewrite::new(rewritten)
                    .desc("Canonizar cuadrado perfecto con partner lineal pequeño");
                return Some(finish_standard_root_shortcut(
                    ctx,
                    expr,
                    rewrite,
                    "Perfect Square Trinomial Factor",
                    collect_steps,
                ));
            }
        }

        if let Some(partner_canonical) = canonicalize_direct_pair_factor_root(ctx, combined_partner)
        {
            if partner_canonical != combined_partner {
                let rewritten =
                    build_mul_expr_from_factors_root(ctx, &[squared_factor, partner_canonical]);
                return Some(run_named_rebuilt_root_shortcut_simplify(
                    options,
                    ctx,
                    expr,
                    rewritten,
                    "Perfect Square Trinomial Factor",
                    "Perfect Square Trinomial Factor",
                    collect_steps,
                ));
            }
        }

        if remaining_factors.len() == 1 {
            if let Some(partner_simplified) =
                isolated_simplify_expr_if_changed(options, ctx, combined_partner)
            {
                let rewritten =
                    build_mul_expr_from_factors_root(ctx, &[squared_factor, partner_simplified]);
                return Some(run_named_rebuilt_root_shortcut_simplify(
                    options,
                    ctx,
                    expr,
                    rewritten,
                    "Perfect Square Trinomial Factor",
                    "Perfect Square Trinomial Factor",
                    collect_steps,
                ));
            }
        }

        let rewritten = build_mul_expr_from_factors_root(ctx, &[squared_factor, combined_partner]);
        return Some(run_named_rebuilt_root_shortcut_simplify(
            options,
            ctx,
            expr,
            rewritten,
            "Perfect Square Trinomial Factor",
            "Perfect Square Trinomial Factor",
            collect_steps,
        ));
    }

    None
}

fn try_standard_special_angle_exact_value_factor_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    fn canonicalize_special_angle_partner_root(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
        factor_sum_diff_cubes_partner_root(ctx, expr)
            .or_else(|| factor_higher_degree_difference_partner_root(ctx, expr))
            .or_else(|| factor_sophie_germain_partner_root(ctx, expr))
            .or_else(|| factor_known_small_polynomial_partner_root(ctx, expr))
            .or_else(|| canonicalize_direct_pair_factor_root(ctx, expr))
    }

    fn split_fractional_constant_factor_root(
        ctx: &mut Context,
        expr: ExprId,
    ) -> Option<Vec<ExprId>> {
        let Expr::Div(numerator, denominator) = ctx.get(expr).clone() else {
            return None;
        };
        if !is_pure_arithmetic_constant_expr_root(ctx, denominator) {
            return None;
        }
        let one = ctx.num(1);
        let reciprocal = ctx.add(Expr::Div(one, denominator));
        Some(vec![reciprocal, numerator])
    }

    fn build_remaining_partner_root(
        ctx: &mut Context,
        factors: &[ExprId],
        excluded_index: usize,
    ) -> Option<ExprId> {
        let remaining_factors: Vec<_> = factors
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(index, factor)| (index != excluded_index).then_some(factor))
            .collect();
        (!remaining_factors.is_empty())
            .then(|| build_mul_expr_from_factors_root(ctx, &remaining_factors))
    }

    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() < 2 {
        return None;
    }

    for special_index in 0..factors.len() {
        let Some(special_value) =
            extract_special_angle_exact_value_root(ctx, factors[special_index])
        else {
            continue;
        };
        let Some(combined_partner) = build_remaining_partner_root(ctx, &factors, special_index)
        else {
            continue;
        };
        let Some(double_angle_arg) =
            extract_positive_cos_double_angle_arg_root(ctx, combined_partner)
        else {
            continue;
        };
        if cas_ast::collect_variables(ctx, double_angle_arg).is_empty() {
            continue;
        }

        let rewritten_factors = [special_value, combined_partner];
        let rewritten = build_nonexpanding_locally_simplified_mul_expr_from_factors_root(
            ctx,
            &rewritten_factors,
        );
        let shortcut_steps = if collect_steps {
            vec![build_root_shortcut_compact_step(
                expr,
                rewritten,
                "Canonizar producto con factor trigonométrico de ángulo especial y coseno de doble ángulo",
                "Special Angle Direct Pair Product",
            )]
        } else {
            Vec::new()
        };
        return Some((rewritten, shortcut_steps));
    }

    for special_index in 0..factors.len() {
        let Some(special_value) =
            extract_special_angle_exact_value_root(ctx, factors[special_index])
        else {
            continue;
        };
        let Some(combined_partner) = build_remaining_partner_root(ctx, &factors, special_index)
        else {
            continue;
        };
        let Some((trig_fn, full_arg)) =
            extract_direct_scaled_half_angle_pow2_source_root(ctx, combined_partner)
        else {
            continue;
        };

        let partner_target = build_scaled_half_angle_target_root(ctx, trig_fn, full_arg);
        let rewritten_factors = [special_value, partner_target];
        let rewritten = build_nonexpanding_locally_simplified_mul_expr_from_factors_root(
            ctx,
            &rewritten_factors,
        );
        let shortcut_steps = if collect_steps {
            vec![build_root_shortcut_compact_step(
                expr,
                rewritten,
                "Canonizar producto con factor exacto de ángulo especial y partner de medio ángulo escalado",
                "Special Angle Direct Pair Product",
            )]
        } else {
            Vec::new()
        };
        return Some((rewritten, shortcut_steps));
    }

    for special_index in 0..factors.len() {
        let Some(special_value) =
            extract_special_angle_exact_value_root(ctx, factors[special_index])
        else {
            continue;
        };
        let Some(combined_partner) = build_remaining_partner_root(ctx, &factors, special_index)
        else {
            continue;
        };
        let partner_canonical = canonicalize_special_angle_partner_root(ctx, combined_partner)
            .unwrap_or(combined_partner);
        let partner_canonical = if cas_ast::count_nodes(ctx, partner_canonical) <= 20
            && extract_direct_scaled_half_angle_pow2_source_root(ctx, partner_canonical).is_none()
        {
            isolated_simplify_expr_if_changed(options, ctx, partner_canonical)
                .unwrap_or(partner_canonical)
        } else {
            partner_canonical
        };
        let mut rewritten_factors = if matches!(
            ctx.get(partner_canonical),
            Expr::Add(_, _) | Expr::Sub(_, _)
        ) {
            vec![special_value]
        } else {
            let Some(split_factors) = split_fractional_constant_factor_root(ctx, special_value)
            else {
                continue;
            };
            split_factors
        };
        rewritten_factors.push(partner_canonical);
        let rewritten_raw = build_nonexpanding_locally_simplified_mul_expr_from_factors_root(
            ctx,
            &rewritten_factors,
        );
        let rewritten_raw = strip_multiplicative_one_root(ctx, rewritten_raw);
        let defer_nested_simplify = rewritten_factors.iter().copied().any(|factor| {
            extract_positive_cos_double_angle_arg_root(ctx, factor)
                .is_some_and(|arg| !cas_ast::collect_variables(ctx, arg).is_empty())
                || extract_direct_positive_double_cos_square_diff_target_root(ctx, factor)
                    .is_some_and(|arg| !cas_ast::collect_variables(ctx, arg).is_empty())
        });
        let rewritten = if defer_nested_simplify {
            rewritten_raw
        } else {
            isolated_simplify_expr_if_changed(options, ctx, rewritten_raw).unwrap_or(rewritten_raw)
        };
        let shortcut_steps = if collect_steps {
            vec![build_root_shortcut_compact_step(
                expr,
                rewritten,
                "Canonizar producto con factor exacto fraccional de ángulo especial y partner equivalente",
                "Special Angle Fractional Exact Product",
            )]
        } else {
            Vec::new()
        };
        return Some((rewritten, shortcut_steps));
    }

    for special_index in 0..factors.len() {
        let Some(special_value) =
            extract_special_angle_exact_value_root(ctx, factors[special_index])
        else {
            continue;
        };
        let Some(combined_partner) = build_remaining_partner_root(ctx, &factors, special_index)
        else {
            continue;
        };
        let Some(partner_factored) =
            factor_known_small_polynomial_partner_root(ctx, combined_partner)
        else {
            continue;
        };

        let rewritten = build_mul_expr_from_factors_root(ctx, &[special_value, partner_factored]);
        let shortcut_steps = if collect_steps {
            vec![build_root_shortcut_compact_step(
                expr,
                rewritten,
                "Canonizar producto con factor trigonométrico de ángulo especial y partner polinómico pequeño",
                "Special Angle Direct Pair Product",
            )]
        } else {
            Vec::new()
        };
        return Some((rewritten, shortcut_steps));
    }

    let mut saw_special_angle = false;
    let mut any_changed = false;
    let mut rewritten_factors = factors.clone();

    for (index, factor) in factors.iter().copied().enumerate() {
        let replacement =
            if let Some(special_value) = extract_special_angle_exact_value_root(ctx, factor) {
                saw_special_angle = true;
                Some(special_value)
            } else {
                canonicalize_direct_pair_factor_root(ctx, factor)
            };

        let Some(replacement) = replacement else {
            continue;
        };
        if compare_expr(ctx, replacement, factor) == Ordering::Equal {
            continue;
        }
        rewritten_factors[index] = replacement;
        any_changed = true;
    }

    if saw_special_angle && any_changed {
        let rewritten_raw = build_nonexpanding_locally_simplified_mul_expr_from_factors_root(
            ctx,
            &rewritten_factors,
        );
        let defer_nested_simplify = rewritten_factors.iter().copied().any(|factor| {
            extract_positive_cos_double_angle_arg_root(ctx, factor)
                .is_some_and(|arg| !cas_ast::collect_variables(ctx, arg).is_empty())
                || extract_direct_positive_double_cos_square_diff_target_root(ctx, factor)
                    .is_some_and(|arg| !cas_ast::collect_variables(ctx, arg).is_empty())
                || matches!(ctx.get(factor), Expr::Add(_, _) | Expr::Sub(_, _))
        });
        let rewritten = if defer_nested_simplify {
            rewritten_raw
        } else {
            isolated_simplify_expr_if_changed(options, ctx, rewritten_raw).unwrap_or(rewritten_raw)
        };
        let shortcut_steps = if collect_steps {
            vec![build_root_shortcut_compact_step(
                expr,
                rewritten,
                "Canonizar producto con factor trigonométrico de ángulo especial y partner directo",
                "Special Angle Direct Pair Product",
            )]
        } else {
            Vec::new()
        };
        return Some((rewritten, shortcut_steps));
    }

    None
}

fn factor_known_small_polynomial_partner_root(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    if let Some(factored) = factor_short_geometric_sum_partner_root(ctx, expr) {
        if compare_expr(ctx, factored, expr) != Ordering::Equal {
            return Some(factored);
        }
    }

    if let Some(factored) =
        cas_math::factor::factor_perfect_square_trinomial(ctx, expr).or_else(|| {
            let view = AddView::from_expr(ctx, expr);
            build_direct_perfect_square_from_terms_root(ctx, &view.terms)
        })
    {
        if compare_expr(ctx, factored, expr) != Ordering::Equal {
            return Some(factored);
        }
    }

    let rewrite = try_rewrite_automatic_factor_expr(ctx, expr)?;
    let factored = strip_multiplicative_one_root(ctx, rewrite.rewritten);
    if compare_expr(ctx, factored, expr) == Ordering::Equal {
        return None;
    }

    if extract_direct_two_linear_shift_product_root(ctx, factored).is_some()
        || extract_direct_three_linear_shift_product_root(ctx, factored).is_some()
        || matches_direct_small_pow_expansion_pair_root(ctx, expr, factored)
    {
        return Some(factored);
    }

    None
}

fn factor_short_geometric_sum_partner_root(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let terms = AddView::from_expr(ctx, expr).terms;
    if terms.len() != 4 || terms.iter().any(|(_, sign)| *sign != Sign::Pos) {
        return None;
    }

    for (term_expr, _) in &terms {
        let base = match ctx.get(*term_expr).clone() {
            Expr::Pow(base, exponent) => {
                let exponent = extract_i64_integer(ctx, exponent)?;
                if (1..=3).contains(&exponent) {
                    base
                } else {
                    continue;
                }
            }
            _ if extract_i64_integer(ctx, *term_expr) != Some(1) => *term_expr,
            _ => continue,
        };

        if !matches_geometric_series_sum_root(ctx, expr, base, 3) {
            continue;
        }

        let one = ctx.num(1);
        let two = ctx.num(2);
        let linear = ctx.add(Expr::Add(base, one));
        let squared = ctx.add(Expr::Pow(base, two));
        let quadratic = ctx.add(Expr::Add(squared, one));
        return Some(ctx.add(Expr::Mul(linear, quadratic)));
    }

    None
}

fn build_scaled_half_angle_pow2_target_root(
    ctx: &mut Context,
    trig_fn: BuiltinFn,
    full_arg: ExprId,
) -> ExprId {
    let half = ctx.add(Expr::Number(BigRational::new(1.into(), 2.into())));
    let half_arg = smart_mul(ctx, half, full_arg);
    let trig_expr = ctx.call_builtin(trig_fn, vec![half_arg]);
    let two = ctx.num(2);
    let trig_sq = ctx.add(Expr::Pow(trig_expr, two));
    smart_mul(ctx, two, trig_sq)
}

fn build_scaled_half_angle_target_root(
    ctx: &mut Context,
    trig_fn: BuiltinFn,
    full_arg: ExprId,
) -> ExprId {
    let one = ctx.num(1);
    let cos_expr = ctx.call_builtin(BuiltinFn::Cos, vec![full_arg]);
    match trig_fn {
        BuiltinFn::Sin => ctx.add(Expr::Sub(one, cos_expr)),
        BuiltinFn::Cos => ctx.add(Expr::Add(one, cos_expr)),
        _ => ctx.num(0),
    }
}

fn extract_direct_scaled_half_angle_pow2_source_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(BuiltinFn, ExprId)> {
    let (coeff, trig_name, trig_arg, effective_sign) =
        extract_signed_numeric_trig_pow2(ctx, expr, Sign::Pos)?;
    if effective_sign != Sign::Pos || coeff != BigRational::from_integer(2.into()) {
        return None;
    }

    let trig_fn = match trig_name {
        "sin" => BuiltinFn::Sin,
        "cos" => BuiltinFn::Cos,
        _ => return None,
    };
    let full_arg = extract_half_scaled_base_root(ctx, trig_arg)?;
    Some((trig_fn, full_arg))
}

fn factor_small_linear_shift_product_partner_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    if extract_direct_two_linear_shift_product_root(ctx, expr).is_some()
        || extract_direct_three_linear_shift_product_root(ctx, expr).is_some()
    {
        return Some(expr);
    }

    let rewrite = try_rewrite_automatic_factor_expr(ctx, expr)?;
    let factored = strip_multiplicative_one_root(ctx, rewrite.rewritten);
    if extract_direct_two_linear_shift_product_root(ctx, factored).is_some()
        || extract_direct_three_linear_shift_product_root(ctx, factored).is_some()
    {
        return Some(factored);
    }

    None
}

fn build_direct_three_linear_shift_expanded_target_root(
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

fn build_direct_two_linear_shift_expanded_target_root(
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

fn is_safe_direct_pair_anchor_target_root(ctx: &mut Context, expr: ExprId) -> bool {
    (cas_ast::collect_variables(ctx, expr).is_empty() && cas_ast::count_nodes(ctx, expr) <= 16)
        || extract_plain_sinh_or_cosh_arg_root(ctx, expr).is_some()
        || extract_unary_builtin_arg_root(ctx, expr, BuiltinFn::Tanh).is_some()
        || extract_direct_tangent_addition_fraction_target_root(ctx, expr).is_some()
}

fn is_pure_arithmetic_constant_expr_root(ctx: &Context, expr: ExprId) -> bool {
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

fn rewrite_small_exp_product_root(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() != 2 {
        return None;
    }

    let e = ctx.add(Expr::Constant(Constant::E));
    for (left, right) in [(factors[0], factors[1]), (factors[1], factors[0])] {
        if compare_expr(ctx, left, e) != Ordering::Equal {
            continue;
        }
        let exp_arg = extract_exp_argument(ctx, right)?;
        let one = ctx.num(1);
        let shifted_arg = ctx.add(Expr::Add(exp_arg, one));
        return Some(ctx.call_builtin(BuiltinFn::Exp, vec![shifted_arg]));
    }

    None
}

fn canonicalize_safe_anchor_direct_partner_root(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    if let Some((trig_fn, full_arg)) =
        extract_direct_scaled_half_angle_square_target_root(ctx, expr)
    {
        if trig_fn == BuiltinFn::Cos {
            return Some(build_scaled_half_angle_pow2_target_root(
                ctx, trig_fn, full_arg,
            ));
        }
    }

    if let Some((trig_fn, full_arg)) = extract_direct_abs_trig_half_angle_target_root(ctx, expr) {
        return Some(build_direct_sqrt_abs_trig_half_angle_target_root(
            ctx, trig_fn, full_arg,
        ));
    }

    if let Some(base) = extract_addition_of_successive_unit_fractions_arg_root(ctx, expr) {
        return Some(build_collapsed_successive_unit_fractions_expr_root(
            ctx, base,
        ));
    }

    if matches!(ctx.get(expr), Expr::Function(_, _)) {
        let expanded = expand_logs_collect_positive_assumptions(ctx, expr).rewritten;
        if compare_expr(ctx, expanded, expr) != Ordering::Equal {
            return Some(strip_multiplicative_one_root(ctx, expanded));
        }
    }

    if let Some(rewritten) = rewrite_small_exp_product_root(ctx, expr) {
        if compare_expr(ctx, rewritten, expr) != Ordering::Equal {
            return Some(strip_multiplicative_one_root(ctx, rewritten));
        }
    }

    None
}

fn try_standard_direct_scaled_half_angle_square_shortcut(
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let (trig_fn, full_arg) = extract_direct_scaled_half_angle_pow2_source_root(ctx, expr)?;
    if !expr_contains_division_node_local(ctx, full_arg) {
        return None;
    }

    let canonical_arg =
        isolated_simplify_expr_if_changed(&crate::phase::SimplifyOptions::default(), ctx, full_arg)
            .unwrap_or(full_arg);
    let rewritten = build_scaled_half_angle_target_root(ctx, trig_fn, canonical_arg);
    let rule_name = match trig_fn {
        BuiltinFn::Sin => "2·sin²(x/2) = 1 - cos(x)",
        BuiltinFn::Cos => "2·cos²(x/2) = 1 + cos(x)",
        _ => unreachable!("only sin/cos scaled half-angle squares are supported"),
    };

    Some(finish_standard_root_shortcut(
        ctx,
        expr,
        crate::rule::Rewrite::with_local(rewritten, rule_name, expr, rewritten),
        rule_name,
        collect_steps,
    ))
}

fn try_standard_rational_half_angle_target_passthrough_shortcut(
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let (trig_fn, full_arg) = extract_direct_scaled_half_angle_square_target_root(ctx, expr)?;
    if !expr_contains_division_node_local(ctx, full_arg) {
        return None;
    }

    let canonical_arg =
        isolated_simplify_expr_if_changed(&crate::phase::SimplifyOptions::default(), ctx, full_arg)
            .unwrap_or(full_arg);
    let rewritten = build_scaled_half_angle_target_root(ctx, trig_fn, canonical_arg);
    if compare_expr(ctx, rewritten, expr) == Ordering::Equal {
        return Some((expr, Vec::new()));
    }

    let rule_name = match trig_fn {
        BuiltinFn::Sin => "2·sin²(x/2) = 1 - cos(x)",
        BuiltinFn::Cos => "2·cos²(x/2) = 1 + cos(x)",
        _ => unreachable!("only sin/cos rational half-angle targets are supported"),
    };

    Some(finish_standard_root_shortcut(
        ctx,
        expr,
        crate::rule::Rewrite::with_local(rewritten, rule_name, expr, rewritten),
        rule_name,
        collect_steps,
    ))
}

fn extract_sin_arctan_source_arg_root(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Function(outer_fn, outer_args) = ctx.get(expr) else {
        return None;
    };
    if outer_args.len() != 1 || !ctx.is_builtin(*outer_fn, BuiltinFn::Sin) {
        return None;
    }

    let Expr::Function(inner_fn, inner_args) = ctx.get(outer_args[0]) else {
        return None;
    };
    if inner_args.len() != 1 {
        return None;
    }
    if ctx.is_builtin(*inner_fn, BuiltinFn::Arctan) || ctx.is_builtin(*inner_fn, BuiltinFn::Atan) {
        return Some(inner_args[0]);
    }

    None
}

fn extract_inverse_trig_ratio_anchor_base_root(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    if let Some(base) = extract_sin_arctan_source_arg_root(ctx, expr) {
        return Some(base);
    }

    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };
    let radicand = extract_square_root_base(ctx, den)?;
    let one = ctx.num(1);
    let two = ctx.num(2);
    let num_sq = ctx.add(Expr::Pow(num, two));
    let expected_radicand = ctx.add(Expr::Add(num_sq, one));
    (compare_expr(ctx, radicand, expected_radicand) == Ordering::Equal).then_some(num)
}

fn build_inverse_trig_ratio_anchor_product_root(
    ctx: &mut Context,
    base: ExprId,
    partner: ExprId,
) -> ExprId {
    let one = ctx.num(1);
    let two = ctx.num(2);
    let half = ctx.add(Expr::Number(BigRational::new(1.into(), 2.into())));
    let base_sq = ctx.add(Expr::Pow(base, two));
    let radicand = ctx.add(Expr::Add(base_sq, one));
    let sqrt = ctx.add(Expr::Pow(radicand, half));
    let numerator = build_mul_expr_from_factors_root(ctx, &[base, sqrt, partner]);
    ctx.add(Expr::Div(numerator, radicand))
}

fn extract_square_plus_one_base_root(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let terms = AddView::from_expr(ctx, expr).terms;
    if terms.len() != 2 {
        return None;
    }

    let mut base = None;
    let mut saw_one = false;
    let two = ctx.num(2);
    for (term_expr, sign) in terms {
        if sign != Sign::Pos {
            return None;
        }
        if extract_i64_integer(ctx, term_expr) == Some(1) {
            if saw_one {
                return None;
            }
            saw_one = true;
            continue;
        }
        let Expr::Pow(candidate_base, exponent) = ctx.get(term_expr).clone() else {
            return None;
        };
        if compare_expr(ctx, exponent, two) != Ordering::Equal
            || base.replace(candidate_base).is_some()
        {
            return None;
        }
    }

    saw_one.then_some(base?)
}

fn extract_direct_short_geometric_product_base_root(
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

fn build_direct_short_geometric_sum_expanded_target_root(
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

fn canonicalize_inverse_trig_ratio_small_polynomial_partner_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    if factor_short_geometric_sum_partner_root(ctx, expr).is_some() {
        return Some(expr);
    }
    if let Some(base) = extract_direct_short_geometric_product_base_root(ctx, expr) {
        return Some(build_direct_short_geometric_sum_expanded_target_root(
            ctx, base,
        ));
    }
    if let Some((base, constants)) = extract_direct_two_linear_shift_product_root(ctx, expr) {
        return build_direct_two_linear_shift_expanded_target_root(ctx, base, &constants);
    }
    if let Some(factored) = factor_known_small_polynomial_partner_root(ctx, expr) {
        if let Some((base, constants)) = extract_direct_two_linear_shift_product_root(ctx, factored)
        {
            return build_direct_two_linear_shift_expanded_target_root(ctx, base, &constants);
        }
    }

    None
}

fn extract_signed_two_factor_product_root(
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

fn try_standard_two_factor_small_partner_canonicalization_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let (numeric_coeff, factors) = extract_signed_two_factor_product_root(ctx, expr)?;

    for partner_index in 0..2 {
        let other_index = 1 - partner_index;
        let Some(partner_factored) =
            factor_known_small_polynomial_partner_root(ctx, factors[partner_index])
        else {
            continue;
        };
        if extract_special_angle_exact_value_root(ctx, factors[other_index]).is_some() {
            continue;
        }
        if factor_known_small_polynomial_partner_root(ctx, factors[other_index]).is_some() {
            continue;
        }

        let other_canonical = canonicalize_direct_pair_factor_root(ctx, factors[other_index])
            .unwrap_or(factors[other_index]);
        if compare_expr(ctx, partner_factored, factors[partner_index]) == Ordering::Equal
            && compare_expr(ctx, other_canonical, factors[other_index]) == Ordering::Equal
        {
            continue;
        }

        let base_factors = if partner_index == 0 {
            vec![partner_factored, other_canonical]
        } else {
            vec![other_canonical, partner_factored]
        };
        let rewritten_factors = if numeric_coeff == BigRational::from_integer((-1).into()) {
            let minus_one = ctx.num(-1);
            let mut factors_with_sign = Vec::with_capacity(3);
            factors_with_sign.push(minus_one);
            factors_with_sign.extend(base_factors);
            factors_with_sign
        } else {
            base_factors
        };
        let rewritten =
            build_locally_simplified_mul_expr_from_factors_root(ctx, &rewritten_factors);
        return Some(run_named_rebuilt_root_shortcut_simplify(
            options,
            ctx,
            expr,
            rewritten,
            "Canonizar producto binario firmado con partner polinómico pequeño",
            "Canonical Two-Factor Partner",
            collect_steps,
        ));
    }

    None
}

fn try_standard_three_linear_shift_anchor_direct_partner_shortcut(
    _options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() < 4 {
        return None;
    }

    for i in 0..factors.len() {
        for j in (i + 1)..factors.len() {
            for k in (j + 1)..factors.len() {
                let anchor_subset = [factors[i], factors[j], factors[k]];
                let anchor_subset_expr = build_mul_expr_from_factors_root(ctx, &anchor_subset);
                let Some((base, constants)) =
                    extract_direct_three_linear_shift_product_root(ctx, anchor_subset_expr)
                else {
                    continue;
                };
                let Some(anchor_expanded) =
                    build_direct_three_linear_shift_expanded_target_root(ctx, base, &constants)
                else {
                    continue;
                };

                let remaining_factors = factors
                    .iter()
                    .copied()
                    .enumerate()
                    .filter_map(|(index, factor)| {
                        (index != i && index != j && index != k).then_some(factor)
                    })
                    .collect::<Vec<_>>();
                if remaining_factors.is_empty() {
                    continue;
                }

                let partner_expr = build_mul_expr_from_factors_root(ctx, &remaining_factors);
                let partner_canonical = if let Some(rewritten) =
                    rewrite_direct_double_angle_inverse_trig_target_root(ctx, partner_expr)
                {
                    strip_multiplicative_one_root(ctx, rewritten)
                } else if let Some(rewrite) =
                    try_rewrite_trig_inverse_composition_expr(ctx, partner_expr)
                {
                    strip_multiplicative_one_root(ctx, rewrite.rewritten)
                } else if let Some((lhs_arg, rhs_arg)) =
                    extract_direct_tangent_addition_target_root(ctx, partner_expr)
                {
                    build_tangent_addition_fraction_root(ctx, lhs_arg, rhs_arg)
                } else if expr_contains_sqrt_or_half_power_local(ctx, partner_expr)
                    && cas_ast::count_nodes(ctx, partner_expr) <= 16
                {
                    isolated_simplify_expr_if_changed(
                        &crate::phase::SimplifyOptions::default(),
                        ctx,
                        partner_expr,
                    )
                    .map(|rewritten| strip_multiplicative_one_root(ctx, rewritten))
                    .unwrap_or(partner_expr)
                } else {
                    continue;
                };

                let rewritten =
                    build_mul_expr_from_factors_root(ctx, &[anchor_expanded, partner_canonical]);
                let rewrite = crate::rule::Rewrite::new(rewritten).desc(
                    "Canonizar producto de tres desplazamientos lineales con partner directo pequeño",
                );
                return Some(finish_standard_root_shortcut(
                    ctx,
                    expr,
                    rewrite,
                    "Three Linear Shift Anchor Product",
                    collect_steps,
                ));
            }
        }
    }

    None
}

fn try_standard_two_factor_direct_pair_anchor_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    fn canonicalize_two_factor_partner_root(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
        factor_sum_diff_cubes_partner_root(ctx, expr)
            .or_else(|| factor_higher_degree_difference_partner_root(ctx, expr))
            .or_else(|| factor_sophie_germain_partner_root(ctx, expr))
            .or_else(|| factor_known_small_polynomial_partner_root(ctx, expr))
            .or_else(|| canonicalize_direct_pair_factor_root(ctx, expr))
    }

    let (numeric_coeff, factors) = extract_signed_two_factor_product_root(ctx, expr)?;

    for anchor_index in 0..2 {
        let partner_index = 1 - anchor_index;
        if extract_special_angle_exact_value_root(ctx, factors[anchor_index]).is_some() {
            continue;
        }
        if is_pure_arithmetic_constant_expr_root(ctx, factors[anchor_index])
            || is_pure_arithmetic_constant_expr_root(ctx, factors[partner_index])
        {
            continue;
        }
        let anchor_canonical = canonicalize_direct_pair_factor_root(ctx, factors[anchor_index])
            .unwrap_or(factors[anchor_index]);
        let anchor_changed =
            compare_expr(ctx, anchor_canonical, factors[anchor_index]) != Ordering::Equal;
        if !anchor_changed || !is_safe_direct_pair_anchor_target_root(ctx, anchor_canonical) {
            continue;
        }

        let partner_canonical = canonicalize_two_factor_partner_root(ctx, factors[partner_index])
            .unwrap_or(factors[partner_index]);

        let base_factors = if anchor_index == 0 {
            vec![anchor_canonical, partner_canonical]
        } else {
            vec![partner_canonical, anchor_canonical]
        };
        let rewritten_factors = if numeric_coeff == BigRational::from_integer((-1).into()) {
            let minus_one = ctx.num(-1);
            let mut factors_with_sign = Vec::with_capacity(3);
            factors_with_sign.push(minus_one);
            factors_with_sign.extend(base_factors);
            factors_with_sign
        } else {
            base_factors
        };
        let rewritten_raw =
            build_locally_simplified_mul_expr_from_factors_root(ctx, &rewritten_factors);
        let rewritten =
            isolated_simplify_expr_if_changed(options, ctx, rewritten_raw).unwrap_or(rewritten_raw);
        if compare_expr(ctx, rewritten, expr) == Ordering::Equal {
            continue;
        }

        let shortcut_steps = if collect_steps {
            vec![build_root_shortcut_compact_step(
                expr,
                rewritten,
                "Canonizar producto binario con ancla directa y partner equivalente",
                "Direct Pair Anchor Product",
            )]
        } else {
            Vec::new()
        };
        return Some((rewritten, shortcut_steps));
    }

    None
}

fn try_standard_safe_anchor_small_polynomial_partner_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let factors = flatten_mul_chain(ctx, expr);
    if !(2..=3).contains(&factors.len()) {
        return None;
    }

    for partner_index in 0..factors.len() {
        let Some(partner_canonical) =
            factor_known_small_polynomial_partner_root(ctx, factors[partner_index])
        else {
            continue;
        };

        let anchor_factors: Vec<_> = factors
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(index, factor)| (index != partner_index).then_some(factor))
            .collect();
        if anchor_factors
            .iter()
            .copied()
            .any(|factor| extract_special_angle_exact_value_root(ctx, factor).is_some())
        {
            continue;
        }
        let anchor_expr = build_mul_expr_from_factors_root(ctx, &anchor_factors);
        let anchor_canonical = canonicalize_direct_pair_factor_root(ctx, anchor_expr)
            .or_else(|| isolated_simplify_expr_if_changed(options, ctx, anchor_expr))
            .unwrap_or(anchor_expr);
        if !is_safe_direct_pair_anchor_target_root(ctx, anchor_canonical) {
            continue;
        }

        let anchor_changed = compare_expr(ctx, anchor_canonical, anchor_expr) != Ordering::Equal;
        let partner_changed =
            compare_expr(ctx, partner_canonical, factors[partner_index]) != Ordering::Equal;
        if !anchor_changed && !partner_changed {
            continue;
        }

        let rewritten_raw = build_locally_simplified_mul_expr_from_factors_root(
            ctx,
            &[anchor_canonical, partner_canonical],
        );
        let rewritten =
            isolated_simplify_expr_if_changed(options, ctx, rewritten_raw).unwrap_or(rewritten_raw);
        if compare_expr(ctx, rewritten, expr) == Ordering::Equal {
            continue;
        }

        let shortcut_steps = if collect_steps {
            vec![build_root_shortcut_compact_step(
                expr,
                rewritten,
                "Canonizar producto con ancla segura y partner polinómico pequeño",
                "Safe Anchor Small Polynomial Partner",
            )]
        } else {
            Vec::new()
        };
        return Some((rewritten, shortcut_steps));
    }

    None
}

fn try_standard_inverse_trig_anchor_small_polynomial_partner_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let factors = flatten_mul_chain(ctx, expr);
    if !(2..=4).contains(&factors.len()) {
        return None;
    }

    for anchor_index in 0..factors.len() {
        let anchor_expr = factors[anchor_index];
        let Some(anchor_base) = extract_inverse_trig_ratio_anchor_base_root(ctx, anchor_expr)
        else {
            continue;
        };
        let partner_factors: Vec<_> = factors
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(index, factor)| (index != anchor_index).then_some(factor))
            .collect();
        let partner_expr = build_mul_expr_from_factors_root(ctx, &partner_factors);
        let Some(partner_canonical) =
            canonicalize_inverse_trig_ratio_small_polynomial_partner_root(ctx, partner_expr)
        else {
            continue;
        };
        let rewritten_raw =
            build_inverse_trig_ratio_anchor_product_root(ctx, anchor_base, partner_canonical);
        let rewritten =
            isolated_simplify_expr_if_changed(options, ctx, rewritten_raw).unwrap_or(rewritten_raw);
        if compare_expr(ctx, rewritten, expr) == Ordering::Equal
            && compare_expr(ctx, partner_canonical, partner_expr) == Ordering::Equal
        {
            continue;
        }

        let shortcut_steps = if collect_steps {
            vec![build_root_shortcut_compact_step(
                expr,
                rewritten,
                "Canonizar producto con ancla trig-inversa cociente y partner polinómico pequeño",
                "Inverse Trig Ratio Anchor Small Polynomial Partner",
            )]
        } else {
            Vec::new()
        };
        return Some((rewritten, shortcut_steps));
    }

    None
}

fn try_standard_safe_anchor_direct_partner_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let factors = flatten_mul_chain(ctx, expr);
    if !(2..=3).contains(&factors.len()) {
        return None;
    }

    for partner_index in 0..factors.len() {
        let Some(partner_canonical) =
            canonicalize_safe_anchor_direct_partner_root(ctx, factors[partner_index])
        else {
            continue;
        };

        let anchor_factors: Vec<_> = factors
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(index, factor)| (index != partner_index).then_some(factor))
            .collect();
        if anchor_factors
            .iter()
            .copied()
            .any(|factor| extract_special_angle_exact_value_root(ctx, factor).is_some())
        {
            continue;
        }
        let anchor_expr = build_mul_expr_from_factors_root(ctx, &anchor_factors);
        let anchor_canonical = canonicalize_direct_pair_factor_root(ctx, anchor_expr)
            .or_else(|| isolated_simplify_expr_if_changed(options, ctx, anchor_expr))
            .unwrap_or(anchor_expr);
        if !is_safe_direct_pair_anchor_target_root(ctx, anchor_canonical) {
            continue;
        }

        let anchor_changed = compare_expr(ctx, anchor_canonical, anchor_expr) != Ordering::Equal;
        let partner_changed =
            compare_expr(ctx, partner_canonical, factors[partner_index]) != Ordering::Equal;
        if !anchor_changed && !partner_changed {
            continue;
        }

        let rewritten_raw = build_locally_simplified_mul_expr_from_factors_root(
            ctx,
            &[anchor_canonical, partner_canonical],
        );
        let rewritten =
            isolated_simplify_expr_if_changed(options, ctx, rewritten_raw).unwrap_or(rewritten_raw);
        if compare_expr(ctx, rewritten, expr) == Ordering::Equal {
            continue;
        }

        let shortcut_steps = if collect_steps {
            vec![build_root_shortcut_compact_step(
                expr,
                rewritten,
                "Canonizar producto con ancla segura y partner directo",
                "Safe Anchor Direct Partner",
            )]
        } else {
            Vec::new()
        };
        return Some((rewritten, shortcut_steps));
    }

    None
}

fn try_standard_scaled_half_angle_anchor_direct_partner_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    fn canonicalize_scaled_half_angle_partner_root(
        ctx: &mut Context,
        expr: ExprId,
    ) -> Option<ExprId> {
        factor_sum_diff_cubes_partner_root(ctx, expr)
            .or_else(|| factor_higher_degree_difference_partner_root(ctx, expr))
            .or_else(|| factor_sophie_germain_partner_root(ctx, expr))
            .or_else(|| factor_known_small_polynomial_partner_root(ctx, expr))
            .or_else(|| canonicalize_direct_pair_factor_root(ctx, expr))
    }

    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() < 3 {
        return None;
    }

    for left_index in 0..factors.len() {
        for right_index in (left_index + 1)..factors.len() {
            let anchor_source =
                build_mul_expr_from_factors_root(ctx, &[factors[left_index], factors[right_index]]);
            let Some((trig_fn, full_arg)) =
                extract_direct_scaled_half_angle_pow2_source_root(ctx, anchor_source)
            else {
                continue;
            };

            let partner_factors: Vec<_> = factors
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(index, factor)| {
                    (index != left_index && index != right_index).then_some(factor)
                })
                .collect();
            if partner_factors.is_empty() {
                continue;
            }

            let partner_expr = build_mul_expr_from_factors_root(ctx, &partner_factors);
            if cas_ast::collect_variables(ctx, partner_expr).is_empty() {
                continue;
            }
            if is_safe_direct_pair_anchor_target_root(ctx, partner_expr) {
                continue;
            }
            let partner_canonical = canonicalize_scaled_half_angle_partner_root(ctx, partner_expr)
                .unwrap_or(partner_expr);
            if is_safe_direct_pair_anchor_target_root(ctx, partner_canonical) {
                continue;
            }
            let canonical_arg =
                isolated_simplify_expr_if_changed(options, ctx, full_arg).unwrap_or(full_arg);
            let anchor_target = build_scaled_half_angle_target_root(ctx, trig_fn, canonical_arg);
            let rewritten =
                build_mul_expr_from_factors_root(ctx, &[anchor_target, partner_canonical]);
            if compare_expr(ctx, rewritten, expr) == Ordering::Equal
                && compare_expr(ctx, partner_canonical, partner_expr) == Ordering::Equal
            {
                continue;
            }

            return Some(run_named_rebuilt_root_shortcut_simplify(
                options,
                ctx,
                expr,
                rewritten,
                "Scaled Half-Angle Anchor Direct Partner",
                "Scaled Half-Angle Anchor Direct Partner",
                collect_steps,
            ));
        }
    }

    None
}

fn try_standard_square_anchor_linear_shift_partner_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let factors = flatten_mul_chain(ctx, expr);
    if !(2..=4).contains(&factors.len()) {
        return None;
    }

    for anchor_index in 0..factors.len() {
        let anchor_canonical = canonicalize_direct_pair_factor_root(ctx, factors[anchor_index])
            .or_else(|| factor_known_small_polynomial_partner_root(ctx, factors[anchor_index]))
            .or_else(|| isolated_simplify_expr_if_changed(options, ctx, factors[anchor_index]))
            .unwrap_or(factors[anchor_index]);
        if extract_square_power_base_root(ctx, anchor_canonical).is_none() {
            continue;
        }

        let remaining_factors: Vec<_> = factors
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(index, factor)| (index != anchor_index).then_some(factor))
            .collect();
        let partner_expr = build_mul_expr_from_factors_root(ctx, &remaining_factors);
        let Some(partner_canonical) =
            factor_small_linear_shift_product_partner_root(ctx, partner_expr)
        else {
            continue;
        };

        let anchor_changed =
            compare_expr(ctx, anchor_canonical, factors[anchor_index]) != Ordering::Equal;
        let partner_changed = compare_expr(ctx, partner_canonical, partner_expr) != Ordering::Equal;
        if !anchor_changed && !partner_changed {
            continue;
        }

        let rewritten = ctx.add(Expr::Mul(anchor_canonical, partner_canonical));
        let rewrite = crate::rule::Rewrite::new(rewritten)
            .desc("Canonizar producto con ancla cuadrada y partner lineal pequeño");
        return Some(finish_standard_root_shortcut(
            ctx,
            expr,
            rewrite,
            "Square Anchor Linear Shift Product",
            collect_steps,
        ));
    }

    None
}

fn try_standard_positive_double_cos_square_diff_factor_shortcut(
    _options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() < 2 {
        return None;
    }

    for (index, factor) in factors.iter().copied().enumerate() {
        let Some(arg) = extract_direct_positive_double_cos_square_diff_target_root(ctx, factor)
        else {
            continue;
        };

        let mut rewritten_factors = factors.clone();
        rewritten_factors[index] = build_positive_cos_double_angle_expr_root(ctx, arg);
        let rewritten = build_nonexpanding_locally_simplified_mul_expr_from_factors_root(
            ctx,
            &rewritten_factors,
        );
        let rewrite =
            crate::rule::Rewrite::with_local(rewritten, "Double Angle Expansion", expr, rewritten);
        return Some(finish_standard_root_shortcut(
            ctx,
            expr,
            rewrite,
            "Double Angle Expansion",
            collect_steps,
        ));
    }

    None
}

fn try_standard_trig_power_reduction_factor_shortcut(
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
        let replacement = if let Some(arg) =
            extract_direct_trig_power_mixed_square_target_root(ctx, factor)
        {
            Some(build_scaled_double_angle_sin_square_root(ctx, arg))
        } else if let Some(arg) = extract_direct_cos_fourth_power_reduction_target_root(ctx, factor)
        {
            Some(build_plain_trig_pow4_root(ctx, BuiltinFn::Cos, arg))
        } else {
            extract_direct_sin_cos_square_product_reduction_target_root(ctx, factor)
                .map(|arg| build_plain_sin_cos_square_product_root(ctx, arg))
        };

        let Some(replacement) = replacement else {
            continue;
        };

        let mut rewritten_factors = factors.clone();
        rewritten_factors[index] = replacement;
        let rewritten = build_mul_expr_from_factors_root(ctx, &rewritten_factors);

        return Some(run_named_rebuilt_root_shortcut_simplify(
            options,
            ctx,
            expr,
            rewritten,
            "Power Reduction Identity",
            "Power Reduction Identity",
            collect_steps,
        ));
    }

    for first_index in 0..factors.len() {
        for second_index in (first_index + 1)..factors.len() {
            let pair_factor = build_mul_expr_from_factors_root(
                ctx,
                &[factors[first_index], factors[second_index]],
            );
            let Some(arg) = extract_direct_trig_power_mixed_square_target_root(ctx, pair_factor)
            else {
                continue;
            };

            let replacement = build_scaled_double_angle_sin_square_root(ctx, arg);
            let rewritten_factors = factors
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(index, factor)| {
                    if index == first_index {
                        Some(replacement)
                    } else if index == second_index {
                        None
                    } else {
                        Some(factor)
                    }
                })
                .collect::<Vec<_>>();
            let rewritten = build_mul_expr_from_factors_root(ctx, &rewritten_factors);

            return Some(run_named_rebuilt_root_shortcut_simplify(
                options,
                ctx,
                expr,
                rewritten,
                "Power Reduction Identity",
                "Power Reduction Identity",
                collect_steps,
            ));
        }
    }

    None
}

fn try_standard_guarded_small_zero_pair_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    if !is_guarded_small_zero_composition_candidate_root(ctx, expr) {
        return None;
    }
    if should_defer_guarded_small_zero_additive_shortcut(ctx, expr) {
        return None;
    }

    let parent_ctx = build_root_shortcut_parent_ctx(options, ctx, expr);
    match ctx.get(expr) {
        Expr::Mul(_, _) => {
            let rule = crate::rules::arithmetic::CollapseExactZeroProductFactorRule;
            let rewrite = crate::rule::Rule::apply(&rule, ctx, expr, &parent_ctx)?;
            Some(finish_root_shortcut_with_rewrite_meta(
                ctx,
                expr,
                rewrite,
                "Collapse Zero Product via Exact Residual",
                collect_steps,
            ))
        }
        Expr::Add(_, _) | Expr::Sub(_, _) => {
            let rule = crate::rules::arithmetic::CollapseExactZeroThreeTermSubsetRule;
            let rewrite = crate::rule::Rule::apply(&rule, ctx, expr, &parent_ctx)?;
            Some(finish_root_shortcut_with_rewrite_meta(
                ctx,
                expr,
                rewrite,
                "Collapse Exact Zero Additive Subexpression",
                collect_steps,
            ))
        }
        _ => None,
    }
}

fn try_standard_direct_small_zero_pair_shortcut(
    _options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    if !is_direct_small_zero_composition_candidate_root(ctx, expr) {
        return None;
    }

    match ctx.get(expr) {
        Expr::Mul(_, _) | Expr::Add(_, _) | Expr::Sub(_, _) => {
            let zero = ctx.num(0);
            let rewrite =
                crate::rule::Rewrite::with_local(zero, "Exact Zero Core Composition", expr, zero);
            Some(finish_root_shortcut_with_rewrite_meta(
                ctx,
                expr,
                rewrite,
                "Collapse Exact Zero Additive Subexpression",
                collect_steps,
            ))
        }
        _ => None,
    }
}

fn try_standard_trig_power_reduction_zero_shortcut(
    _options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    if !expr_contains_any_builtin_local(ctx, expr, &[BuiltinFn::Sin, BuiltinFn::Cos]) {
        return None;
    }

    let (lhs_core, rhs_core) =
        crate::rules::arithmetic::extract_two_term_core_difference(ctx, expr)?;
    let rewrite =
        crate::rules::arithmetic::try_build_direct_trig_power_reduction_equivalence_rewrite(
            ctx, lhs_core, rhs_core,
        )?;
    Some(finish_root_shortcut_with_rewrite_meta(
        ctx,
        expr,
        rewrite,
        "Power Reduction Identity",
        collect_steps,
    ))
}

fn try_standard_hyperbolic_cosh_cubic_subset_zero_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    if !matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _)) {
        return None;
    }

    let view = AddView::from_expr(ctx, expr);
    if !(5..=6).contains(&view.terms.len()) {
        return None;
    }

    let matches_small_hyperbolic_identity_subset = |ctx: &mut Context, subset_expr: ExprId| {
        let subset_view = AddView::from_expr(ctx, subset_expr);
        let has_positive = subset_view.terms.iter().any(|(_, sign)| *sign == Sign::Pos);
        let has_negative = subset_view.terms.iter().any(|(_, sign)| *sign == Sign::Neg);

        subset_view.terms.len() <= 3
            && has_positive
            && has_negative
            && expr_contains_hyperbolic_builtin_local(ctx, subset_expr)
            && !expr_contains_trig_builtin_local(ctx, subset_expr)
            && subset_view.terms.iter().all(|(term, _)| {
                expr_contains_hyperbolic_builtin_local(ctx, *term)
                    || matches!(ctx.get(*term), Expr::Number(_))
            })
            && (matches_direct_hyperbolic_cosh_cubic_zero_identity_root(ctx, subset_expr)
                || isolated_simplify_rewrites_to_zero(options, ctx, subset_expr))
    };

    for subset_size in [2usize, 3usize] {
        for first_index in 0..view.terms.len() {
            for second_index in (first_index + 1)..view.terms.len() {
                if subset_size == 2 {
                    let subset_terms = [view.terms[first_index], view.terms[second_index]];
                    let subset_expr = build_signed_sum_expr_root(ctx, &subset_terms);
                    if !matches_small_hyperbolic_identity_subset(ctx, subset_expr) {
                        continue;
                    }

                    let remaining_terms: smallvec::SmallVec<[(ExprId, Sign); 8]> = view
                        .terms
                        .iter()
                        .copied()
                        .enumerate()
                        .filter_map(|(index, term)| {
                            (index != first_index && index != second_index).then_some(term)
                        })
                        .collect();
                    if !(2..=4).contains(&remaining_terms.len()) {
                        continue;
                    }

                    let remaining_expr = AddView {
                        root: expr,
                        terms: remaining_terms,
                    }
                    .rebuild(ctx);
                    if try_standard_exact_zero_equivalence_shortcut(
                        options,
                        ctx,
                        remaining_expr,
                        false,
                    )
                    .is_some()
                        || isolated_simplify_rewrites_to_zero(options, ctx, remaining_expr)
                    {
                        let zero = ctx.num(0);
                        return Some(run_rebuilt_root_shortcut_simplify(
                            options,
                            ctx,
                            expr,
                            zero,
                            collect_steps,
                        ));
                    }
                    continue;
                }

                for third_index in (second_index + 1)..view.terms.len() {
                    let subset_terms = [
                        view.terms[first_index],
                        view.terms[second_index],
                        view.terms[third_index],
                    ];
                    let subset_expr = build_signed_sum_expr_root(ctx, &subset_terms);
                    if !matches_small_hyperbolic_identity_subset(ctx, subset_expr) {
                        continue;
                    }

                    let remaining_terms: smallvec::SmallVec<[(ExprId, Sign); 8]> = view
                        .terms
                        .iter()
                        .copied()
                        .enumerate()
                        .filter_map(|(index, term)| {
                            (index != first_index && index != second_index && index != third_index)
                                .then_some(term)
                        })
                        .collect();
                    if !(2..=4).contains(&remaining_terms.len()) {
                        continue;
                    }

                    let remaining_expr = AddView {
                        root: expr,
                        terms: remaining_terms,
                    }
                    .rebuild(ctx);
                    if try_standard_exact_zero_equivalence_shortcut(
                        options,
                        ctx,
                        remaining_expr,
                        false,
                    )
                    .is_some()
                        || isolated_simplify_rewrites_to_zero(options, ctx, remaining_expr)
                    {
                        let zero = ctx.num(0);
                        return Some(run_rebuilt_root_shortcut_simplify(
                            options,
                            ctx,
                            expr,
                            zero,
                            collect_steps,
                        ));
                    }
                }
            }
        }
    }

    None
}

fn try_standard_half_angle_subset_zero_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    if !matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _)) {
        return None;
    }

    let view = AddView::from_expr(ctx, expr);
    if !(5..=6).contains(&view.terms.len()) {
        return None;
    }

    let matches_trig_identity_subset = |ctx: &mut Context, subset_expr: ExprId| {
        expr_contains_trig_builtin_local(ctx, subset_expr)
            && !expr_contains_hyperbolic_builtin_local(ctx, subset_expr)
            && (matches_direct_small_zero_identity_root(ctx, subset_expr)
                || matches_direct_trig_mixed_double_angle_zero_identity_root(ctx, subset_expr))
    };

    for subset_size in [2usize, 3usize] {
        for first_index in 0..view.terms.len() {
            for second_index in (first_index + 1)..view.terms.len() {
                if subset_size == 2 {
                    let subset_terms = [view.terms[first_index], view.terms[second_index]];
                    let subset_expr = build_signed_sum_expr_root(ctx, &subset_terms);
                    if !matches_trig_identity_subset(ctx, subset_expr) {
                        continue;
                    }

                    let remaining_terms: smallvec::SmallVec<[(ExprId, Sign); 8]> = view
                        .terms
                        .iter()
                        .copied()
                        .enumerate()
                        .filter_map(|(index, term)| {
                            (index != first_index && index != second_index).then_some(term)
                        })
                        .collect();
                    if remaining_terms.len() != 3 {
                        continue;
                    }

                    let remaining_expr = AddView {
                        root: expr,
                        terms: remaining_terms,
                    }
                    .rebuild(ctx);
                    if try_standard_exact_zero_equivalence_shortcut(
                        options,
                        ctx,
                        remaining_expr,
                        false,
                    )
                    .is_some()
                        || isolated_simplify_rewrites_to_zero(options, ctx, remaining_expr)
                    {
                        let zero = ctx.num(0);
                        return Some(run_rebuilt_root_shortcut_simplify(
                            options,
                            ctx,
                            expr,
                            zero,
                            collect_steps,
                        ));
                    }
                    continue;
                }

                for third_index in (second_index + 1)..view.terms.len() {
                    let subset_terms = [
                        view.terms[first_index],
                        view.terms[second_index],
                        view.terms[third_index],
                    ];
                    let subset_expr = build_signed_sum_expr_root(ctx, &subset_terms);
                    if !matches_trig_identity_subset(ctx, subset_expr) {
                        continue;
                    }

                    let remaining_terms: smallvec::SmallVec<[(ExprId, Sign); 8]> = view
                        .terms
                        .iter()
                        .copied()
                        .enumerate()
                        .filter_map(|(index, term)| {
                            (index != first_index && index != second_index && index != third_index)
                                .then_some(term)
                        })
                        .collect();
                    if remaining_terms.len() != 3 {
                        continue;
                    }

                    let remaining_expr = AddView {
                        root: expr,
                        terms: remaining_terms,
                    }
                    .rebuild(ctx);
                    if try_standard_exact_zero_equivalence_shortcut(
                        options,
                        ctx,
                        remaining_expr,
                        false,
                    )
                    .is_some()
                        || isolated_simplify_rewrites_to_zero(options, ctx, remaining_expr)
                    {
                        let zero = ctx.num(0);
                        return Some(run_rebuilt_root_shortcut_simplify(
                            options,
                            ctx,
                            expr,
                            zero,
                            collect_steps,
                        ));
                    }
                }
            }
        }
    }

    None
}

fn try_standard_nested_exact_zero_child_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let rewritten = match ctx.get(expr) {
        Expr::Add(lhs, rhs) => {
            let lhs = *lhs;
            let rhs = *rhs;
            if expr_contains_trig_or_hyperbolic_builtin_local(ctx, lhs)
                && is_supported_nested_zero_child_partner(ctx, rhs)
                && child_matches_direct_or_isolated_exact_zero(options, ctx, lhs)
            {
                Some(rhs)
            } else if expr_contains_trig_or_hyperbolic_builtin_local(ctx, rhs)
                && is_supported_nested_zero_child_partner(ctx, lhs)
                && child_matches_direct_or_isolated_exact_zero(options, ctx, rhs)
            {
                Some(lhs)
            } else {
                None
            }
        }
        Expr::Sub(lhs, rhs) => {
            let lhs = *lhs;
            let rhs = *rhs;
            if expr_contains_trig_or_hyperbolic_builtin_local(ctx, rhs)
                && is_supported_nested_zero_child_partner(ctx, lhs)
                && child_matches_direct_or_isolated_exact_zero(options, ctx, rhs)
            {
                Some(lhs)
            } else if expr_contains_trig_or_hyperbolic_builtin_local(ctx, lhs)
                && is_supported_nested_zero_child_partner(ctx, rhs)
                && child_matches_direct_or_isolated_exact_zero(options, ctx, lhs)
            {
                Some(ctx.add(Expr::Neg(rhs)))
            } else {
                None
            }
        }
        _ => None,
    }?;

    Some(run_rebuilt_root_shortcut_simplify(
        options,
        ctx,
        expr,
        rewritten,
        collect_steps,
    ))
}

fn try_standard_zero_product_with_exact_zero_child_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let Expr::Mul(lhs, rhs) = ctx.get(expr) else {
        return None;
    };
    let lhs = *lhs;
    let rhs = *rhs;

    let zero_factor = if expr_contains_trig_or_hyperbolic_builtin_local(ctx, lhs)
        && is_supported_nested_zero_child_partner(ctx, rhs)
        && child_matches_direct_or_isolated_exact_zero(options, ctx, lhs)
    {
        Some(lhs)
    } else if expr_contains_trig_or_hyperbolic_builtin_local(ctx, rhs)
        && is_supported_nested_zero_child_partner(ctx, lhs)
        && child_matches_direct_or_isolated_exact_zero(options, ctx, rhs)
    {
        Some(rhs)
    } else {
        None
    }?;

    let zero = ctx.num(0);
    let mut shortcut_steps = Vec::new();
    if collect_steps {
        let mut step = Step::new_compact(
            "Collapse Exact Zero Additive Subexpression",
            "Collapse Exact Zero Additive Subexpression",
            expr,
            zero,
        );
        step.global_before = Some(expr);
        step.global_after = Some(zero);
        step.importance = crate::step::ImportanceLevel::High;
        shortcut_steps.push(step);
        let sibling = if compare_expr(ctx, zero_factor, lhs) == Ordering::Equal {
            rhs
        } else {
            lhs
        };
        shortcut_steps.push(build_root_shortcut_compact_step(
            ctx.add(Expr::Mul(zero, sibling)),
            zero,
            "Cualquier producto con un factor 0 vale 0",
            "Producto por cero",
        ));
    }

    Some((zero, shortcut_steps))
}

fn is_supported_nested_direct_equivalence_partner(ctx: &Context, expr: ExprId) -> bool {
    !expr_contains_trig_or_hyperbolic_builtin_local(ctx, expr)
}

fn try_standard_embedded_trig_product_to_sum_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let rewritten = match ctx.get(expr) {
        Expr::Mul(_, _) => {
            let factor_count = flatten_mul_chain(ctx, expr).len();
            if factor_count > 3 {
                try_rewrite_product_to_sum_expr(ctx, expr).map(|rewrite| rewrite.rewritten)
            } else if let Expr::Mul(lhs, rhs) = ctx.get(expr) {
                let lhs = *lhs;
                let rhs = *rhs;
                if expr_contains_trig_builtin_local(ctx, lhs)
                    && is_supported_nested_direct_equivalence_partner(ctx, rhs)
                {
                    try_rewrite_product_to_sum_expr(ctx, lhs)
                        .map(|rewrite| smart_mul(ctx, rewrite.rewritten, rhs))
                } else if expr_contains_trig_builtin_local(ctx, rhs)
                    && is_supported_nested_direct_equivalence_partner(ctx, lhs)
                {
                    try_rewrite_product_to_sum_expr(ctx, rhs)
                        .map(|rewrite| smart_mul(ctx, lhs, rewrite.rewritten))
                } else {
                    None
                }
            } else {
                None
            }
        }
        Expr::Div(num, den) => {
            let num = *num;
            let den = *den;
            if expr_contains_trig_builtin_local(ctx, num)
                && is_supported_nested_direct_equivalence_partner(ctx, den)
            {
                try_rewrite_product_to_sum_expr(ctx, num)
                    .map(|rewrite| ctx.add(Expr::Div(rewrite.rewritten, den)))
            } else if expr_contains_trig_builtin_local(ctx, den)
                && is_supported_nested_direct_equivalence_partner(ctx, num)
            {
                try_rewrite_product_to_sum_expr(ctx, den)
                    .map(|rewrite| ctx.add(Expr::Div(num, rewrite.rewritten)))
            } else {
                None
            }
        }
        _ => None,
    }?;

    Some(run_named_rebuilt_root_shortcut_simplify(
        options,
        ctx,
        expr,
        rewritten,
        "Aplicar producto a suma en el factor trigonométrico",
        "Product-to-Sum Identity",
        collect_steps,
    ))
}

fn try_standard_pythagorean_additive_shortcut(
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let mut current = AddView::from_expr(ctx, expr).rebuild(ctx);
    let mut shortcut_steps = Vec::new();
    let mut changed = false;

    loop {
        let normalized = AddView::from_expr(ctx, current).rebuild(ctx);
        if let Some(rewritten) =
            try_rewrite_structural_numeric_pythagorean_add_pair(ctx, normalized)
        {
            let before = current;
            current = rewritten;
            changed = true;
            if collect_steps {
                let mut step = Step::new_compact(
                    "Pythagorean Identity",
                    "Pythagorean Identity",
                    before,
                    current,
                );
                step.importance = crate::step::ImportanceLevel::High;
                shortcut_steps.push(step);
            }
            continue;
        }
        if let Some(rewrite) = try_rewrite_pythagorean_identity_add_expr(ctx, normalized) {
            let before = current;
            current = rewrite.rewritten;
            changed = true;
            if collect_steps {
                let mut step = Step::new_compact(
                    "Pythagorean Identity",
                    "Pythagorean Identity",
                    before,
                    current,
                );
                step.importance = crate::step::ImportanceLevel::High;
                shortcut_steps.push(step);
            }
            continue;
        }

        let Some(rewrite) = try_rewrite_combine_constants_expr(ctx, current) else {
            break;
        };
        if compare_expr(ctx, current, rewrite.rewritten) == Ordering::Equal {
            break;
        }

        let before = current;
        current = rewrite.rewritten;
        changed = true;
        if collect_steps {
            let mut step =
                Step::new_compact(&rewrite.description, "Combine Constants", before, current);
            step.importance = crate::step::ImportanceLevel::High;
            shortcut_steps.push(step);
        }
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

fn try_standard_pythagorean_generic_coefficient_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let rewrite = try_rewrite_pythagorean_generic_coefficient_add_expr(ctx, expr)?;
    let mut simplifier = crate::Simplifier::with_default_rules();
    std::mem::swap(&mut simplifier.context, ctx);
    let (result, inner_steps, _stats) = simplifier.simplify_with_stats(
        rewrite.rewritten,
        crate::SimplifyOptions {
            suppress_depth_overflow_warnings: true,
            ..options.clone()
        },
    );
    std::mem::swap(&mut simplifier.context, ctx);

    let mut shortcut_steps = Vec::new();
    if collect_steps {
        let mut step = Step::new_compact(
            &rewrite.desc,
            "Pythagorean with Generic Coefficient",
            expr,
            rewrite.rewritten,
        );
        step.global_before = Some(expr);
        step.global_after = Some(rewrite.rewritten);
        step.importance = crate::step::ImportanceLevel::High;
        shortcut_steps.push(step);
        shortcut_steps.extend(inner_steps);
    }
    Some((result, shortcut_steps))
}

fn extract_shared_additive_passthrough_pair_cores_root(
    ctx: &mut Context,
    lhs: ExprId,
    rhs: ExprId,
) -> Option<(ExprId, ExprId)> {
    let lhs_terms = AddView::from_expr(ctx, lhs).terms;
    let rhs_terms = AddView::from_expr(ctx, rhs).terms;
    if lhs_terms.is_empty() || rhs_terms.is_empty() {
        return None;
    }

    let mut lhs_used = vec![false; lhs_terms.len()];
    let mut rhs_used = vec![false; rhs_terms.len()];
    let mut matched_any = false;

    for (lhs_index, (lhs_term, lhs_sign)) in lhs_terms.iter().copied().enumerate() {
        let Some(rhs_index) =
            rhs_terms
                .iter()
                .copied()
                .enumerate()
                .find_map(|(rhs_index, (rhs_term, rhs_sign))| {
                    (!rhs_used[rhs_index]
                        && lhs_sign == rhs_sign
                        && compare_expr(ctx, lhs_term, rhs_term) == Ordering::Equal)
                        .then_some(rhs_index)
                })
        else {
            continue;
        };
        lhs_used[lhs_index] = true;
        rhs_used[rhs_index] = true;
        matched_any = true;
    }

    if !matched_any {
        return None;
    }

    let remaining_lhs_terms: smallvec::SmallVec<[(ExprId, Sign); 8]> = lhs_terms
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(index, term)| (!lhs_used[index]).then_some(term))
        .collect();
    let remaining_rhs_terms: smallvec::SmallVec<[(ExprId, Sign); 8]> = rhs_terms
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(index, term)| (!rhs_used[index]).then_some(term))
        .collect();

    if remaining_lhs_terms.is_empty() || remaining_rhs_terms.is_empty() {
        return None;
    }

    let lhs_core = AddView {
        root: lhs,
        terms: remaining_lhs_terms,
    }
    .rebuild(ctx);
    let rhs_core = AddView {
        root: rhs,
        terms: remaining_rhs_terms,
    }
    .rebuild(ctx);
    Some((lhs_core, rhs_core))
}

fn extract_shared_additive_passthrough_sub_cores_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId)> {
    let (lhs, rhs) = match ctx.get(expr) {
        Expr::Sub(lhs, rhs) => (*lhs, *rhs),
        Expr::Add(lhs, rhs) => match ctx.get(*rhs) {
            Expr::Neg(inner) => (*lhs, *inner),
            _ => return None,
        },
        _ => return None,
    };
    extract_shared_additive_passthrough_pair_cores_root(ctx, lhs, rhs)
}

fn term_pair_is_small_exact_equivalent_root(ctx: &mut Context, lhs: ExprId, rhs: ExprId) -> bool {
    compare_expr(ctx, lhs, rhs) == Ordering::Equal || matches_known_direct_pair_root(ctx, lhs, rhs)
}

fn build_small_two_chunk_additive_partitions_root(
    ctx: &mut Context,
    terms: &[(ExprId, Sign)],
) -> Vec<(ExprId, ExprId)> {
    if !(2..=6).contains(&terms.len()) {
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

fn matches_partitioned_direct_small_zero_sum_root(ctx: &mut Context, expr: ExprId) -> bool {
    let terms = AddView::from_expr(ctx, expr).terms;
    if !(3..=6).contains(&terms.len()) {
        return false;
    }

    build_small_two_chunk_additive_partitions_root(ctx, &terms)
        .into_iter()
        .any(|(lhs_chunk, rhs_chunk)| {
            matches_direct_small_zero_or_known_pair_base_root(ctx, lhs_chunk)
                && matches_direct_small_zero_or_known_pair_base_root(ctx, rhs_chunk)
        })
}

fn extract_partitioned_phase_shift_zero_chunks_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId)> {
    crate::rules::arithmetic::extract_repeated_trig_phase_shift_pair_zero_chunks(ctx, expr)
}

fn matches_direct_trig_product_to_sum_zero_identity_root(ctx: &mut Context, expr: ExprId) -> bool {
    matches_direct_trig_product_to_sum_sin_sin_zero_identity_root(ctx, expr)
        || matches_direct_trig_product_to_sum_sin_cos_zero_identity_root(ctx, expr)
        || matches_direct_trig_product_to_sum_cos_cos_zero_identity_root(ctx, expr)
}

fn matches_direct_trig_product_to_sum_and_odd_half_partition_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let terms = AddView::from_expr(ctx, expr).terms;
    if !(5..=6).contains(&terms.len()) {
        return false;
    }

    for first_index in 0..terms.len().saturating_sub(1) {
        for second_index in (first_index + 1)..terms.len() {
            let odd_terms: Vec<_> = terms
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(index, term)| {
                    (index == first_index || index == second_index).then_some(term)
                })
                .collect();
            if odd_terms.len() != 2 {
                continue;
            }

            let trig_terms: Vec<_> = terms
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(index, term)| {
                    (index != first_index && index != second_index).then_some(term)
                })
                .collect();
            if trig_terms.len() != terms.len() - 2 {
                continue;
            }

            let odd_expr = build_signed_sum_expr_root(ctx, &odd_terms);
            if !matches_direct_odd_half_power_zero_scope_root(ctx, odd_expr) {
                continue;
            }

            let trig_expr = build_signed_sum_expr_root(ctx, &trig_terms);
            if matches_direct_trig_product_to_sum_zero_identity_root(ctx, trig_expr) {
                return true;
            }
        }
    }

    false
}

fn matches_direct_trig_product_to_sum_and_geometric_difference_partition_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let terms = AddView::from_expr(ctx, expr).terms;
    if terms.len() != 6 {
        return false;
    }

    let mut stack = vec![(0usize, Vec::<usize>::new())];
    while let Some((next_index, chosen)) = stack.pop() {
        if chosen.len() == 3 {
            let trig_terms: Vec<_> = terms
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(index, term)| chosen.contains(&index).then_some(term))
                .collect();
            let residual_terms: Vec<_> = terms
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(index, term)| (!chosen.contains(&index)).then_some(term))
                .collect();
            if residual_terms.len() != 3 {
                continue;
            }

            let trig_expr = build_signed_sum_expr_root(ctx, &trig_terms);
            if matches_direct_trig_product_to_sum_zero_identity_root(ctx, trig_expr)
                && matches_geometric_difference_terms_root(ctx, &residual_terms)
            {
                return true;
            }
            continue;
        }

        for index in next_index..terms.len() {
            let mut next = chosen.clone();
            next.push(index);
            stack.push((index + 1, next));
        }
    }

    false
}

fn matches_composed_small_additive_pair_root(ctx: &mut Context, lhs: ExprId, rhs: ExprId) -> bool {
    let lhs_terms = AddView::from_expr(ctx, lhs).terms;
    let rhs_terms = AddView::from_expr(ctx, rhs).terms;
    if !(2..=4).contains(&lhs_terms.len()) || !(2..=4).contains(&rhs_terms.len()) {
        return false;
    }

    let lhs_partitions = build_small_two_chunk_additive_partitions_root(ctx, &lhs_terms);
    let rhs_partitions = build_small_two_chunk_additive_partitions_root(ctx, &rhs_terms);
    for (lhs_a, lhs_b) in lhs_partitions {
        for (rhs_a, rhs_b) in rhs_partitions.iter().copied() {
            if (term_pair_is_small_exact_equivalent_root(ctx, lhs_a, rhs_a)
                && term_pair_is_small_exact_equivalent_root(ctx, lhs_b, rhs_b))
                || (term_pair_is_small_exact_equivalent_root(ctx, lhs_a, rhs_b)
                    && term_pair_is_small_exact_equivalent_root(ctx, lhs_b, rhs_a))
            {
                return true;
            }
        }
    }

    false
}

fn try_standard_small_composed_additive_pair_shortcut(
    _options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    if extract_shared_additive_passthrough_sub_cores_root(ctx, expr).is_some() {
        return None;
    }

    let (lhs, rhs) = match ctx.get(expr) {
        Expr::Sub(lhs, rhs) => (*lhs, *rhs),
        Expr::Add(lhs, rhs) => match ctx.get(*rhs) {
            Expr::Neg(inner) => (*lhs, *inner),
            _ => return None,
        },
        _ => return None,
    };

    if !matches_composed_small_additive_pair_root(ctx, lhs, rhs) {
        return None;
    }

    let zero = ctx.num(0);
    Some(finish_standard_root_shortcut(
        ctx,
        expr,
        crate::rule::Rewrite::new(zero).desc("Parallel additive equivalence composition"),
        "Parallel additive equivalence composition",
        collect_steps,
    ))
}

fn try_standard_partitioned_direct_small_zero_sum_shortcut(
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    if !matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _))
        || extract_shared_additive_passthrough_sub_cores_root(ctx, expr).is_some()
        || !matches_partitioned_direct_small_zero_sum_root(ctx, expr)
    {
        return None;
    }

    let zero = ctx.num(0);
    Some(finish_standard_root_shortcut(
        ctx,
        expr,
        crate::rule::Rewrite::with_local(zero, "Exact Zero Core Composition", expr, zero),
        "Collapse Exact Zero Additive Subexpression",
        collect_steps,
    ))
}

fn try_standard_repeated_phase_shift_pair_shortcut(
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    if let Some((first_chunk, second_chunk)) =
        crate::rules::arithmetic::extract_repeated_trig_phase_shift_pair_zero_chunks(ctx, expr)
    {
        let zero = ctx.num(0);
        let mut shortcut_steps = Vec::new();
        if collect_steps {
            let mut first_step = Step::new_compact(
                "Aplicar identidad de desfase",
                "Aplicar identidad de desfase",
                first_chunk,
                zero,
            );
            first_step.global_before = Some(expr);
            first_step.global_after = Some(second_chunk);
            first_step.importance = crate::step::ImportanceLevel::High;
            shortcut_steps.push(first_step);

            let mut second_step = Step::new_compact(
                "Aplicar identidad de desfase",
                "Aplicar identidad de desfase",
                second_chunk,
                zero,
            );
            second_step.global_before = Some(second_chunk);
            second_step.global_after = Some(zero);
            second_step.importance = crate::step::ImportanceLevel::High;
            shortcut_steps.push(second_step);
        }

        return Some((zero, shortcut_steps));
    }

    if let Some((_common_factor, residual_expr)) =
        extract_common_multiplicative_residual_sum_root(ctx, expr)
    {
        if crate::rules::arithmetic::extract_repeated_trig_phase_shift_pair_zero_chunks(
            ctx,
            residual_expr,
        )
        .is_some()
        {
            let zero = ctx.num(0);
            let rewrite = crate::rule::Rewrite::with_local(
                zero,
                "Equivalent Residual Cancellation",
                expr,
                zero,
            );
            return Some(finish_root_shortcut_with_rewrite_meta(
                ctx,
                expr,
                rewrite,
                "Collapse Common-Scale Equivalent Difference",
                collect_steps,
            ));
        }
    }

    None
}

fn try_standard_direct_small_zero_identity_shortcut(
    _options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    if !matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _)) {
        return None;
    }

    if let Some((first_chunk, second_chunk)) =
        extract_partitioned_phase_shift_zero_chunks_root(ctx, expr)
    {
        let zero = ctx.num(0);
        let mut shortcut_steps = Vec::new();
        if collect_steps {
            let mut first_step = Step::new_compact(
                "Aplicar identidad de desfase",
                "Aplicar identidad de desfase",
                first_chunk,
                zero,
            );
            first_step.global_before = Some(expr);
            first_step.global_after = Some(second_chunk);
            first_step.importance = crate::step::ImportanceLevel::High;
            shortcut_steps.push(first_step);

            let mut second_step = Step::new_compact(
                "Aplicar identidad de desfase",
                "Aplicar identidad de desfase",
                second_chunk,
                zero,
            );
            second_step.global_before = Some(second_chunk);
            second_step.global_after = Some(zero);
            second_step.importance = crate::step::ImportanceLevel::High;
            shortcut_steps.push(second_step);
        }

        return Some((zero, shortcut_steps));
    }

    if let Some((_common_factor, residual_expr)) =
        extract_common_multiplicative_residual_sum_root(ctx, expr)
    {
        if extract_partitioned_phase_shift_zero_chunks_root(ctx, residual_expr).is_some() {
            let zero = ctx.num(0);
            let rewrite = crate::rule::Rewrite::with_local(
                zero,
                "Equivalent Residual Cancellation",
                expr,
                zero,
            );
            return Some(finish_root_shortcut_with_rewrite_meta(
                ctx,
                expr,
                rewrite,
                "Collapse Common-Scale Equivalent Difference",
                collect_steps,
            ));
        }
    }

    if let Some((lhs_core, rhs_core)) =
        crate::rules::arithmetic::extract_two_term_core_difference(ctx, expr)
    {
        if collect_steps
            && expr_contains_division_node_local(ctx, lhs_core)
            && expr_contains_division_node_local(ctx, rhs_core)
        {
            return None;
        }

        if matches_direct_sum_diff_cubes_quotient_pair_root(ctx, lhs_core, rhs_core) {
            return None;
        }

        if matches_direct_half_angle_square_pair_root(ctx, lhs_core, rhs_core) {
            let zero = ctx.num(0);
            let rewrite = crate::rule::Rewrite::with_local(
                zero,
                "Aplicar identidad de ángulo mitad",
                expr,
                zero,
            );
            return Some(finish_root_shortcut_with_rewrite_meta(
                ctx,
                expr,
                rewrite,
                "Aplicar identidad de ángulo mitad",
                collect_steps,
            ));
        }

        if matches_direct_pythagorean_identity_pair_root(ctx, lhs_core, rhs_core) {
            let zero = ctx.num(0);
            let rewrite = crate::rule::Rewrite::with_local(
                zero,
                "Collapse Exact Zero Additive Subexpression",
                expr,
                zero,
            );
            return Some(finish_root_shortcut_with_rewrite_meta(
                ctx,
                expr,
                rewrite,
                "Collapse Exact Zero Additive Subexpression",
                collect_steps,
            ));
        }

        if matches_direct_angle_sum_diff_pair_root(ctx, lhs_core, rhs_core) {
            let zero = ctx.num(0);
            let rewrite =
                crate::rule::Rewrite::with_local(zero, "Angle Sum/Diff Identity", expr, zero);
            return Some(finish_root_shortcut_with_rewrite_meta(
                ctx,
                expr,
                rewrite,
                "Angle Sum/Diff Identity",
                collect_steps,
            ));
        }
    }

    if collect_steps
        && (matches_direct_nested_fraction_simplified_zero_identity_root(ctx, expr)
            || matches_direct_consecutive_telescoping_fraction_zero_identity_root(ctx, expr))
    {
        return None;
    }

    if collect_steps {
        let rule = crate::rules::algebra::fractions::SubFractionsRule;
        if crate::rule::Rule::apply(&rule, ctx, expr, &crate::ParentContext::root()).is_some() {
            return None;
        }
    }

    if let Some((lhs_core, rhs_core)) = extract_direct_trig_sum_product_zero_cores_root(ctx, expr) {
        if is_plain_two_term_sin_cos_sum_or_diff_root(ctx, lhs_core)
            && is_trig_sum_product_candidate_root(ctx, rhs_core)
        {
            let zero = ctx.num(0);
            let rewrite =
                crate::rule::Rewrite::with_local(zero, "Aplicar suma a producto", expr, zero);
            return Some(finish_root_shortcut_with_rewrite_meta(
                ctx,
                expr,
                rewrite,
                "Aplicar suma a producto",
                collect_steps,
            ));
        }

        if is_trig_sum_product_candidate_root(ctx, lhs_core)
            && is_plain_two_term_sin_cos_sum_or_diff_root(ctx, rhs_core)
        {
            let zero = ctx.num(0);
            let rewrite =
                crate::rule::Rewrite::with_local(zero, "Aplicar producto a suma", expr, zero);
            return Some(finish_root_shortcut_with_rewrite_meta(
                ctx,
                expr,
                rewrite,
                "Aplicar producto a suma",
                collect_steps,
            ));
        }
    }

    if AddView::from_expr(ctx, expr).terms.len() == 2
        && (matches_direct_odd_half_power_zero_scope_root(ctx, expr)
            || matches_direct_odd_half_power_zero_identity_root(ctx, expr))
    {
        return None;
    }

    if let Some((lhs_core, rhs_core)) =
        extract_shared_additive_passthrough_sub_cores_root(ctx, expr)
    {
        if matches_direct_half_angle_square_pair_root(ctx, lhs_core, rhs_core) {
            let zero = ctx.num(0);
            let rewrite = crate::rule::Rewrite::with_local(
                zero,
                "Aplicar identidad de ángulo mitad",
                expr,
                zero,
            );
            return Some(finish_root_shortcut_with_rewrite_meta(
                ctx,
                expr,
                rewrite,
                "Aplicar identidad de ángulo mitad",
                collect_steps,
            ));
        }

        if matches_direct_pythagorean_identity_pair_root(ctx, lhs_core, rhs_core)
            || matches_direct_pythagorean_factor_form_pair_root(ctx, lhs_core, rhs_core)
        {
            let zero = ctx.num(0);
            let rewrite =
                crate::rule::Rewrite::with_local(zero, "Pythagorean Identity", expr, zero);
            return Some(finish_root_shortcut_with_rewrite_meta(
                ctx,
                expr,
                rewrite,
                "Pythagorean Identity",
                collect_steps,
            ));
        }

        if matches_direct_cos_square_diff_pair_root(ctx, lhs_core, rhs_core) {
            let zero = ctx.num(0);
            let rewrite = crate::rule::Rewrite::with_local(
                zero,
                "Collapse Exact Zero Additive Subexpression",
                expr,
                zero,
            );
            return Some(finish_root_shortcut_with_rewrite_meta(
                ctx,
                expr,
                rewrite,
                "Collapse Exact Zero Additive Subexpression",
                collect_steps,
            ));
        }

        if matches_direct_angle_sum_diff_pair_root(ctx, lhs_core, rhs_core) {
            let zero = ctx.num(0);
            let rewrite =
                crate::rule::Rewrite::with_local(zero, "Angle Sum/Diff Identity", expr, zero);
            return Some(finish_root_shortcut_with_rewrite_meta(
                ctx,
                expr,
                rewrite,
                "Angle Sum/Diff Identity",
                collect_steps,
            ));
        }

        if matches_direct_trig_cubic_cosine_pair_root(ctx, lhs_core, rhs_core) {
            let zero = ctx.num(0);
            let rewrite = crate::rule::Rewrite::with_local(
                zero,
                "Collapse Exact Zero Additive Subexpression",
                expr,
                zero,
            );
            return Some(finish_root_shortcut_with_rewrite_meta(
                ctx,
                expr,
                rewrite,
                "Collapse Exact Zero Additive Subexpression",
                collect_steps,
            ));
        }
    }

    let view = AddView::from_expr(ctx, expr);
    if !(2..=6).contains(&view.terms.len())
        || !view.terms.iter().any(|(_, sign)| *sign == Sign::Neg)
    {
        return None;
    }

    let has_supported_shape = expr_contains_any_builtin_local(
        ctx,
        expr,
        &[
            BuiltinFn::Sin,
            BuiltinFn::Cos,
            BuiltinFn::Tan,
            BuiltinFn::Cot,
            BuiltinFn::Sec,
            BuiltinFn::Csc,
            BuiltinFn::Sinh,
            BuiltinFn::Cosh,
            BuiltinFn::Tanh,
        ],
    ) || expr_contains_division_node_local(ctx, expr)
        || expr_contains_sqrt_or_half_power_local(ctx, expr)
        || expr_contains_factorial_call_local(ctx, expr);
    if !has_supported_shape {
        return None;
    }

    let direct_child_zero_composition = match ctx.get(expr).clone() {
        Expr::Add(lhs, rhs) | Expr::Sub(lhs, rhs) => {
            matches_direct_small_zero_or_known_pair_base_root(ctx, lhs)
                && matches_direct_small_zero_or_known_pair_base_root(ctx, rhs)
        }
        _ => false,
    };
    let direct_trig_odd_half_partition =
        matches_direct_trig_product_to_sum_and_odd_half_partition_root(ctx, expr);
    let direct_trig_geometric_partition =
        matches_direct_trig_product_to_sum_and_geometric_difference_partition_root(ctx, expr);
    let direct_pair_match = match ctx.get(expr) {
        Expr::Add(lhs, rhs) | Expr::Sub(lhs, rhs) => {
            matches_direct_small_zero_pair_root(ctx, *lhs, *rhs)
        }
        _ => false,
    };
    if !direct_child_zero_composition
        && !direct_trig_odd_half_partition
        && !direct_trig_geometric_partition
        && !matches_direct_small_zero_identity_root(ctx, expr)
        && !direct_pair_match
        && !matches_partitioned_direct_small_zero_sum_root(ctx, expr)
    {
        return None;
    }

    let zero = ctx.num(0);
    let rewrite = crate::rule::Rewrite::with_local(zero, "Exact Zero Core Composition", expr, zero);
    Some(finish_root_shortcut_with_rewrite_meta(
        ctx,
        expr,
        rewrite,
        "Collapse Exact Zero Additive Subexpression",
        collect_steps,
    ))
}

fn try_standard_direct_known_pair_zero_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    if let Some((_common_factor, residual_expr)) =
        extract_common_multiplicative_residual_sum_root(ctx, expr)
    {
        if matches_direct_small_zero_or_known_pair_residual_root(ctx, residual_expr) {
            return None;
        }
    }

    if matches_direct_trig_product_to_sum_and_odd_half_partition_root(ctx, expr)
        || matches_direct_trig_product_to_sum_and_geometric_difference_partition_root(ctx, expr)
        || matches_partitioned_direct_small_zero_sum_root(ctx, expr)
    {
        return None;
    }

    if let Some((lhs_core, rhs_core)) =
        crate::rules::arithmetic::extract_two_term_core_difference(ctx, expr)
    {
        if matches_direct_pythagorean_identity_pair_root(ctx, lhs_core, rhs_core)
            || matches_direct_angle_sum_diff_pair_root(ctx, lhs_core, rhs_core)
        {
            return None;
        }
    }

    if matches_direct_two_factor_product_pair_zero_difference_root(ctx, expr)
        || matches_direct_or_isolated_quotient_pair_zero_difference_root(options, ctx, expr)
    {
        let zero = ctx.num(0);
        let rewrite =
            crate::rule::Rewrite::with_local(zero, "Equivalent Residual Cancellation", expr, zero);
        return Some(finish_root_shortcut_with_rewrite_meta(
            ctx,
            expr,
            rewrite,
            "Collapse Common-Scale Equivalent Difference",
            collect_steps,
        ));
    }

    None
}

fn try_standard_direct_small_zero_additive_combination_shortcut(
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    if !matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _)) {
        return None;
    }

    let term_count = AddView::from_expr(ctx, expr).terms.len();
    if !(4..=6).contains(&term_count) {
        return None;
    }

    if matches_direct_trig_product_to_sum_and_odd_half_partition_root(ctx, expr) {
        let zero = ctx.num(0);
        let rewrite =
            crate::rule::Rewrite::with_local(zero, "Exact Zero Core Composition", expr, zero);
        return Some(finish_root_shortcut_with_rewrite_meta(
            ctx,
            expr,
            rewrite,
            "Collapse Exact Zero Additive Subexpression",
            collect_steps,
        ));
    }

    if matches_direct_trig_product_to_sum_and_geometric_difference_partition_root(ctx, expr) {
        let zero = ctx.num(0);
        let rewrite =
            crate::rule::Rewrite::with_local(zero, "Exact Zero Core Composition", expr, zero);
        return Some(finish_root_shortcut_with_rewrite_meta(
            ctx,
            expr,
            rewrite,
            "Collapse Exact Zero Additive Subexpression",
            collect_steps,
        ));
    }

    if let Expr::Add(lhs, rhs) | Expr::Sub(lhs, rhs) = ctx.get(expr).clone() {
        if matches_direct_small_zero_or_known_pair_base_root(ctx, lhs)
            && matches_direct_small_zero_or_known_pair_base_root(ctx, rhs)
        {
            let zero = ctx.num(0);
            let rewrite =
                crate::rule::Rewrite::with_local(zero, "Exact Zero Core Composition", expr, zero);
            return Some(finish_root_shortcut_with_rewrite_meta(
                ctx,
                expr,
                rewrite,
                "Collapse Exact Zero Additive Subexpression",
                collect_steps,
            ));
        }
    }

    if matches_partitioned_direct_small_zero_sum_root(ctx, expr) {
        let zero = ctx.num(0);
        let rewrite =
            crate::rule::Rewrite::with_local(zero, "Exact Zero Core Composition", expr, zero);
        return Some(finish_root_shortcut_with_rewrite_meta(
            ctx,
            expr,
            rewrite,
            "Collapse Exact Zero Additive Subexpression",
            collect_steps,
        ));
    }

    if let Some((first_chunk, second_chunk)) =
        extract_partitioned_phase_shift_zero_chunks_root(ctx, expr)
    {
        let zero = ctx.num(0);
        let mut shortcut_steps = Vec::new();
        if collect_steps {
            let mut first_step = Step::new_compact(
                "Aplicar identidad de desfase",
                "Aplicar identidad de desfase",
                first_chunk,
                zero,
            );
            first_step.global_before = Some(expr);
            first_step.global_after = Some(second_chunk);
            first_step.importance = crate::step::ImportanceLevel::High;
            shortcut_steps.push(first_step);

            let mut second_step = Step::new_compact(
                "Aplicar identidad de desfase",
                "Aplicar identidad de desfase",
                second_chunk,
                zero,
            );
            second_step.global_before = Some(second_chunk);
            second_step.global_after = Some(zero);
            second_step.importance = crate::step::ImportanceLevel::High;
            shortcut_steps.push(second_step);
        }

        return Some((zero, shortcut_steps));
    }

    let rewrite =
        crate::rules::arithmetic::try_build_direct_small_zero_additive_combination_rewrite(
            ctx, expr,
        )?;
    Some(finish_root_shortcut_with_rewrite_meta(
        ctx,
        expr,
        rewrite,
        "Collapse Exact Zero Additive Subexpression",
        collect_steps,
    ))
}

fn passthrough_direct_pair_rule_name_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> Option<&'static str> {
    if matches_direct_trig_product_to_sum_sin_sin_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_trig_product_to_sum_sin_cos_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_trig_product_to_sum_cos_cos_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_pythagorean_identity_pair_root(ctx, lhs_core, rhs_core)
    {
        return Some("Aplicar suma a producto");
    }

    if matches_direct_angle_sum_diff_pair_root(ctx, lhs_core, rhs_core) {
        return Some("Angle Sum/Diff Identity");
    }

    if matches_direct_nested_fraction_simplified_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_hyperbolic_sinh_sum_to_product_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_hyperbolic_cosh_sum_to_product_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_hyperbolic_cosh_difference_to_product_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_recursive_hyperbolic_sinh_sum_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_recursive_hyperbolic_cosh_sum_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_trig_mixed_double_angle_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_trig_cubic_cosine_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_trig_binomial_square_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_cos_square_diff_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_small_pow_expansion_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_pythagorean_identity_pair_root(ctx, lhs_core, rhs_core)
    {
        return Some("Collapse Exact Zero Additive Subexpression");
    }

    None
}

fn passthrough_residual_zero_rule_name_root(
    ctx: &mut Context,
    residual_expr: ExprId,
) -> Option<&'static str> {
    if matches_direct_symbolic_trig_sum_to_product_zero_identity_root(ctx, residual_expr) {
        return Some("Aplicar suma a producto");
    }

    if matches_direct_half_angle_square_zero_identity_root(ctx, residual_expr) {
        return Some("Aplicar identidad de ángulo mitad");
    }

    if matches_direct_general_phase_shift_zero_identity_root(ctx, residual_expr) {
        return Some("Aplicar identidad de desfase");
    }

    if matches_direct_log_square_product_split_zero_identity_root(ctx, residual_expr)
        || matches_direct_ln_abs_product_split_zero_identity_root(ctx, residual_expr)
    {
        return Some("Expandir logaritmos y cancelar términos iguales");
    }

    if matches_direct_small_zero_identity_root(ctx, residual_expr) {
        return Some("Collapse Exact Zero Additive Subexpression");
    }

    None
}

fn try_standard_shared_passthrough_small_pow_expansion_shortcut(
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

fn try_standard_shared_passthrough_pythagorean_factor_form_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let (lhs_core, rhs_core) = extract_shared_additive_passthrough_sub_cores_root(ctx, expr)?;
    if matches_direct_trig_product_to_sum_sin_sin_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_trig_product_to_sum_sin_cos_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_trig_product_to_sum_cos_cos_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_nested_fraction_simplified_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_hyperbolic_sinh_sum_to_product_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_hyperbolic_cosh_sum_to_product_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_hyperbolic_cosh_difference_to_product_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_recursive_hyperbolic_sinh_sum_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_recursive_hyperbolic_cosh_sum_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_cos_square_diff_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_angle_sum_diff_pair_root(ctx, lhs_core, rhs_core)
    {
        let zero = ctx.num(0);
        let residual_expr = ctx.add(Expr::Sub(lhs_core, rhs_core));
        return Some(finish_standard_root_shortcut(
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
        ));
    }

    for (source, target) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(rewrite) = try_rewrite_pythagorean_factor_form_add_expr(ctx, source) else {
            continue;
        };
        if compare_expr(ctx, rewrite.rewritten, target) == Ordering::Equal {
            let zero = ctx.num(0);
            return Some(finish_standard_root_shortcut(
                ctx,
                expr,
                crate::rule::Rewrite::new(zero).desc("Pythagorean Identity"),
                "Pythagorean Identity",
                collect_steps,
            ));
        }
    }
    let residual_expr = ctx.add(Expr::Sub(lhs_core, rhs_core));
    if expr_contains_trig_or_hyperbolic_builtin_local(ctx, residual_expr)
        && try_standard_exact_zero_equivalence_shortcut(options, ctx, residual_expr, false)
            .is_some()
    {
        let zero = ctx.num(0);
        return Some(finish_standard_root_shortcut(
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
        ));
    }
    None
}

fn try_standard_shared_passthrough_direct_pair_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    if matches_direct_product_to_sum_sin_cos_factor_pair_zero_difference_root(ctx, expr) {
        return None;
    }

    let (lhs_core, rhs_core) = extract_shared_additive_passthrough_sub_cores_root(ctx, expr)?;
    let zero = ctx.num(0);
    let residual_expr = ctx.add(Expr::Sub(lhs_core, rhs_core));

    if let Some(rule_name) = passthrough_direct_pair_rule_name_root(ctx, lhs_core, rhs_core) {
        return Some(finish_standard_root_shortcut(
            ctx,
            expr,
            crate::rule::Rewrite::with_local(zero, rule_name, residual_expr, zero),
            rule_name,
            collect_steps,
        ));
    }

    if expr_contains_any_builtin_local(ctx, residual_expr, &[BuiltinFn::Ln, BuiltinFn::Log])
        && try_standard_exact_zero_equivalence_shortcut(options, ctx, residual_expr, false)
            .is_some()
    {
        return Some(run_named_rebuilt_root_shortcut_simplify(
            options,
            ctx,
            expr,
            zero,
            "Expandir logaritmos y cancelar términos iguales",
            "Expandir logaritmos y cancelar términos iguales",
            collect_steps,
        ));
    }

    if let Some(rule_name) = passthrough_residual_zero_rule_name_root(ctx, residual_expr) {
        return Some(run_named_rebuilt_root_shortcut_simplify(
            options,
            ctx,
            expr,
            zero,
            rule_name,
            rule_name,
            collect_steps,
        ));
    }

    if try_standard_exact_zero_equivalence_shortcut(options, ctx, residual_expr, false).is_some() {
        return Some(run_named_rebuilt_root_shortcut_simplify(
            options,
            ctx,
            expr,
            zero,
            "Collapse Exact Zero Additive Subexpression",
            "Collapse Exact Zero Additive Subexpression",
            collect_steps,
        ));
    }

    if expr_contains_division_node_local(ctx, residual_expr)
        && cas_ast::count_nodes(ctx, residual_expr) <= 48
        && isolated_simplify_rewrites_to_zero(options, ctx, residual_expr)
    {
        return Some(run_named_rebuilt_root_shortcut_simplify(
            options,
            ctx,
            expr,
            zero,
            "Collapse Exact Zero Additive Subexpression",
            "Collapse Exact Zero Additive Subexpression",
            collect_steps,
        ));
    }

    None
}

fn try_standard_reciprocal_product_pythagorean_zero_shortcut(
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let plan = try_rewrite_reciprocal_product_pythagorean_zero_add_expr(ctx, expr)?;
    Some(finish_standard_root_shortcut(
        ctx,
        expr,
        crate::rule::Rewrite::new(plan.rewritten).desc(plan.desc),
        "Pythagorean Identity",
        collect_steps,
    ))
}

fn try_standard_reciprocal_pythagorean_pair_shortcut(
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let rewritten = if let Some(rewrite) = try_rewrite_sec_tan_pythagorean_identity_expr(ctx, expr)
    {
        rewrite.rewritten
    } else if let Some(rewrite) = try_rewrite_csc_cot_pythagorean_identity_expr(ctx, expr) {
        rewrite.rewritten
    } else {
        return None;
    };

    Some(finish_standard_root_shortcut(
        ctx,
        expr,
        crate::rule::Rewrite::new(rewritten).desc("Pythagorean Identity"),
        "Pythagorean Identity",
        collect_steps,
    ))
}

fn try_standard_reciprocal_pythagorean_zero_shortcut(
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    if !matches_direct_sec_tan_pythagorean_zero_identity_root(ctx, expr)
        && !matches_direct_csc_cot_pythagorean_zero_identity_root(ctx, expr)
    {
        return None;
    }

    let zero = ctx.num(0);
    Some(finish_standard_root_shortcut(
        ctx,
        expr,
        crate::rule::Rewrite::new(zero).desc("Pythagorean Identity"),
        "Pythagorean Identity",
        collect_steps,
    ))
}

fn try_standard_sin_sum_triple_identity_zero_shortcut(
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    try_rewrite_sin_sum_triple_identity_zero_expr(ctx, expr)?;
    let zero = ctx.num(0);
    Some(finish_standard_root_shortcut(
        ctx,
        expr,
        crate::rule::Rewrite::new(zero).desc("sin(t) + sin(3t) = 2·sin(2t)·cos(t)"),
        "Sin Sum Triple Identity Zero",
        collect_steps,
    ))
}

fn extract_standard_trig_binomial_square_data(
    ctx: &Context,
    expr: ExprId,
) -> Option<(ExprId, bool)> {
    let Expr::Pow(base, exponent) = ctx.get(expr) else {
        return None;
    };
    if extract_i64_integer(ctx, *exponent)? != 2 {
        return None;
    }

    let (left, right, is_sum) = match ctx.get(*base) {
        Expr::Add(left, right) => (*left, *right, true),
        Expr::Sub(left, right) => (*left, *right, false),
        _ => return None,
    };

    let extract_trig_arg = |term: ExprId| -> Option<(bool, ExprId)> {
        let Expr::Function(fn_id, args) = ctx.get(term) else {
            return None;
        };
        let [arg] = args.as_slice() else {
            return None;
        };
        if ctx.is_builtin(*fn_id, BuiltinFn::Sin) {
            return Some((true, *arg));
        }
        if ctx.is_builtin(*fn_id, BuiltinFn::Cos) {
            return Some((false, *arg));
        }
        None
    };

    let (lhs_kind, lhs_arg) = extract_trig_arg(left)?;
    let (rhs_kind, rhs_arg) = extract_trig_arg(right)?;
    if lhs_kind == rhs_kind || compare_expr(ctx, lhs_arg, rhs_arg) != Ordering::Equal {
        return None;
    }

    Some((lhs_arg, is_sum))
}

fn build_standard_trig_square_double_angle_term(ctx: &mut Context, arg: ExprId) -> ExprId {
    let two = ctx.num(2);
    let doubled_arg = smart_mul(ctx, two, arg);
    ctx.call_builtin(BuiltinFn::Sin, vec![doubled_arg])
}

fn try_rewrite_standard_trig_binomial_square_double_angle_pair(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<crate::rule::Rewrite> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() < 2 {
        return None;
    }

    for square_idx in 0..view.terms.len() {
        let (square_term, square_sign) = view.terms[square_idx];
        let Some((arg, is_sum)) = extract_standard_trig_binomial_square_data(ctx, square_term)
        else {
            continue;
        };
        let double_angle = build_standard_trig_square_double_angle_term(ctx, arg);
        let required_sign = match (square_sign, is_sum) {
            (Sign::Pos, true) => Sign::Neg,
            (Sign::Pos, false) => Sign::Pos,
            (Sign::Neg, true) => Sign::Pos,
            (Sign::Neg, false) => Sign::Neg,
        };

        for angle_idx in 0..view.terms.len() {
            if angle_idx == square_idx {
                continue;
            }
            let (angle_term, angle_sign) = view.terms[angle_idx];
            if angle_sign != required_sign
                || compare_expr(ctx, angle_term, double_angle) != Ordering::Equal
            {
                continue;
            }

            let mut new_terms = smallvec::SmallVec::<[(ExprId, Sign); 8]>::new();
            for (idx, term) in view.terms.iter().copied().enumerate() {
                if idx != square_idx && idx != angle_idx {
                    new_terms.push(term);
                }
            }
            if new_terms.iter().any(
                |(term, _sign)| matches!(ctx.get(*term), Expr::Number(value) if value.is_one()),
            ) {
                continue;
            }
            new_terms.push((ctx.num(1), square_sign));
            let rewritten = AddView {
                root: expr,
                terms: new_terms,
            }
            .rebuild(ctx);
            let desc = if is_sum {
                "(sin(u)+cos(u))^2 = 1 + sin(2u)"
            } else {
                "(sin(u)-cos(u))^2 = 1 - sin(2u)"
            };
            return Some(crate::rule::Rewrite::new(rewritten).desc(desc));
        }
    }

    None
}

fn try_standard_trig_binomial_square_double_angle_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let rewrite = try_rewrite_standard_trig_binomial_square_double_angle_pair(ctx, expr)?;
    let mut simplifier = crate::Simplifier::with_default_rules();
    std::mem::swap(&mut simplifier.context, ctx);
    let (result, inner_steps, _stats) = simplifier.simplify_with_stats(
        rewrite.new_expr,
        crate::SimplifyOptions {
            suppress_depth_overflow_warnings: true,
            ..options.clone()
        },
    );
    std::mem::swap(&mut simplifier.context, ctx);

    if compare_expr(ctx, result, rewrite.new_expr) == Ordering::Equal {
        return Some(finish_standard_root_shortcut(
            ctx,
            expr,
            rewrite,
            "Trig Square Identity",
            collect_steps,
        ));
    }

    let mut shortcut_steps = Vec::new();
    if collect_steps {
        shortcut_steps.push(build_root_shortcut_step_from_rewrite(
            ctx,
            expr,
            &rewrite,
            "Trig Square Identity",
        ));
        shortcut_steps.extend(inner_steps);
    }
    Some((result, shortcut_steps))
}

fn try_standard_trig_fourth_power_difference_shortcut(
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let normalized = AddView::from_expr(ctx, expr).rebuild(ctx);
    let rewrite = try_rewrite_trig_fourth_power_difference_add_expr(ctx, normalized)?;
    let mut current = rewrite.rewritten;
    let mut shortcut_steps = Vec::new();

    if collect_steps {
        shortcut_steps.push(build_root_shortcut_compact_step(
            expr,
            current,
            "sin⁴(x) - cos⁴(x) = sin²(x) - cos²(x)",
            "Trig Fourth Power Difference",
        ));
    }

    if let Some(rewrite) = try_rewrite_pythagorean_generic_coefficient_add_expr(ctx, current) {
        let before = current;
        current = rewrite.rewritten;
        if collect_steps {
            shortcut_steps.push(build_root_shortcut_compact_step(
                before,
                current,
                "A·sin²(x) + A·cos²(x) = A",
                "Pythagorean with Generic Coefficient",
            ));
        }
    }

    if let Some((result, mut extra_steps)) =
        try_standard_pythagorean_additive_shortcut(ctx, current, collect_steps)
    {
        current = result;
        if collect_steps {
            shortcut_steps.append(&mut extra_steps);
        }
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

fn extract_signed_numeric_trig_pow2(
    ctx: &Context,
    term: ExprId,
    outer_sign: Sign,
) -> Option<(BigRational, &'static str, ExprId, Sign)> {
    let (mut coeff, name, arg) = extract_coeff_trig_pow2(ctx, term)?;
    let mut effective_sign = outer_sign;
    if coeff.is_negative() {
        coeff = -coeff;
        effective_sign = effective_sign.negate();
    }
    Some((coeff, name, arg, effective_sign))
}

fn try_rewrite_structural_numeric_pythagorean_add_pair(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() < 2 {
        return None;
    }

    for i in 0..view.terms.len() {
        for j in (i + 1)..view.terms.len() {
            let (lhs_term, lhs_sign) = view.terms[i];
            let (rhs_term, rhs_sign) = view.terms[j];
            let Some((lhs_coeff, lhs_name, lhs_arg, lhs_effective_sign)) =
                extract_signed_numeric_trig_pow2(ctx, lhs_term, lhs_sign)
            else {
                continue;
            };
            let Some((rhs_coeff, rhs_name, rhs_arg, rhs_effective_sign)) =
                extract_signed_numeric_trig_pow2(ctx, rhs_term, rhs_sign)
            else {
                continue;
            };
            if lhs_name == rhs_name
                || lhs_coeff != rhs_coeff
                || lhs_effective_sign != rhs_effective_sign
                || compare_expr(ctx, lhs_arg, rhs_arg) != Ordering::Equal
            {
                continue;
            }

            let mut remaining_terms = smallvec::SmallVec::<[(ExprId, Sign); 8]>::new();
            for (idx, term) in view.terms.iter().copied().enumerate() {
                if idx != i && idx != j {
                    remaining_terms.push(term);
                }
            }
            remaining_terms.push((ctx.add(Expr::Number(lhs_coeff)), lhs_effective_sign));
            return Some(
                AddView {
                    root: expr,
                    terms: remaining_terms,
                }
                .rebuild(ctx),
            );
        }
    }

    None
}

fn is_mixed_sign_trig_square_difference_root(ctx: &Context, expr: ExprId) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 2 {
        return false;
    }

    let (lhs_term, lhs_sign) = view.terms[0];
    let (rhs_term, rhs_sign) = view.terms[1];
    let Some((lhs_coeff, lhs_name, lhs_arg, lhs_effective_sign)) =
        extract_signed_numeric_trig_pow2(ctx, lhs_term, lhs_sign)
    else {
        return false;
    };
    let Some((rhs_coeff, rhs_name, rhs_arg, rhs_effective_sign)) =
        extract_signed_numeric_trig_pow2(ctx, rhs_term, rhs_sign)
    else {
        return false;
    };

    lhs_name != rhs_name
        && lhs_effective_sign != rhs_effective_sign
        && lhs_coeff == rhs_coeff
        && compare_expr(ctx, lhs_arg, rhs_arg) == Ordering::Equal
}

fn extract_mixed_sign_trig_square_difference_arg_root(
    ctx: &Context,
    expr: ExprId,
) -> Option<ExprId> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 2 {
        return None;
    }

    let (lhs_term, lhs_sign) = view.terms[0];
    let (rhs_term, rhs_sign) = view.terms[1];
    let (lhs_coeff, lhs_name, lhs_arg, lhs_effective_sign) =
        extract_signed_numeric_trig_pow2(ctx, lhs_term, lhs_sign)?;
    let (rhs_coeff, rhs_name, rhs_arg, rhs_effective_sign) =
        extract_signed_numeric_trig_pow2(ctx, rhs_term, rhs_sign)?;

    (lhs_name != rhs_name
        && lhs_effective_sign != rhs_effective_sign
        && lhs_coeff == rhs_coeff
        && compare_expr(ctx, lhs_arg, rhs_arg) == Ordering::Equal)
        .then_some(lhs_arg)
}

fn extract_negative_cos_double_angle_arg_root(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let (coeff, base) = extract_coef_and_base(ctx, expr);
    if coeff != BigRational::from_integer((-1).into()) {
        return None;
    }
    let Some((BuiltinFn::Cos, arg)) = extract_plain_sin_or_cos_arg_root(ctx, base) else {
        return None;
    };
    extract_double_angle_arg_relaxed(ctx, arg)
}

fn has_negative_numeric_pythagorean_pair(ctx: &Context, expr: ExprId) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() < 2 {
        return false;
    }

    for i in 0..view.terms.len() {
        for j in (i + 1)..view.terms.len() {
            let (lhs_term, lhs_sign) = view.terms[i];
            let (rhs_term, rhs_sign) = view.terms[j];
            let Some((lhs_coeff, lhs_name, lhs_arg, lhs_effective_sign)) =
                extract_signed_numeric_trig_pow2(ctx, lhs_term, lhs_sign)
            else {
                continue;
            };
            let Some((rhs_coeff, rhs_name, rhs_arg, rhs_effective_sign)) =
                extract_signed_numeric_trig_pow2(ctx, rhs_term, rhs_sign)
            else {
                continue;
            };
            if lhs_name != rhs_name
                && lhs_coeff == rhs_coeff
                && lhs_effective_sign == Sign::Neg
                && rhs_effective_sign == Sign::Neg
                && compare_expr(ctx, lhs_arg, rhs_arg) == Ordering::Equal
            {
                return true;
            }
        }
    }

    false
}

fn has_structural_numeric_pythagorean_pair(ctx: &Context, expr: ExprId) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() < 2 {
        return false;
    }

    for i in 0..view.terms.len() {
        for j in (i + 1)..view.terms.len() {
            let (lhs_term, lhs_sign) = view.terms[i];
            let (rhs_term, rhs_sign) = view.terms[j];
            let Some((lhs_coeff, lhs_name, lhs_arg, lhs_effective_sign)) =
                extract_signed_numeric_trig_pow2(ctx, lhs_term, lhs_sign)
            else {
                continue;
            };
            let Some((rhs_coeff, rhs_name, rhs_arg, rhs_effective_sign)) =
                extract_signed_numeric_trig_pow2(ctx, rhs_term, rhs_sign)
            else {
                continue;
            };
            if lhs_name != rhs_name
                && lhs_coeff == rhs_coeff
                && lhs_effective_sign == rhs_effective_sign
                && compare_expr(ctx, lhs_arg, rhs_arg) == Ordering::Equal
            {
                return true;
            }
        }
    }

    false
}

fn has_numeric_pythagorean_complement_pair(ctx: &Context, expr: ExprId) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 2 {
        return false;
    }

    let mut constant_is_positive = None;
    let mut trig_sign = None;
    let mut trig_coeff = None;

    for (term, sign) in view.terms.iter().copied() {
        match ctx.get(term) {
            Expr::Number(n) if n.is_one() => {
                constant_is_positive = Some(sign == Sign::Pos);
            }
            _ => {
                let Some((coeff, _name, _arg, effective_sign)) =
                    extract_signed_numeric_trig_pow2(ctx, term, sign)
                else {
                    return false;
                };
                trig_coeff = Some(coeff);
                trig_sign = Some(effective_sign);
            }
        }
    }

    matches!(trig_coeff, Some(coeff) if coeff.is_one())
        && matches!(
            (constant_is_positive, trig_sign),
            (Some(true), Some(Sign::Neg)) | (Some(false), Some(Sign::Pos))
        )
}

fn try_standard_shifted_quotient_exact_one_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    if is_direct_small_zero_shifted_quotient_candidate_root(ctx, expr) {
        let one = ctx.num(1);
        let rewrite =
            crate::rule::Rewrite::with_local(one, "Exact Zero Core Quotient Identity", expr, one);
        return Some(finish_root_shortcut_with_rewrite_meta(
            ctx,
            expr,
            rewrite,
            "Collapse Shifted Quotient of Equivalent Expressions",
            collect_steps,
        ));
    }

    if is_guarded_small_zero_shifted_quotient_candidate_root(ctx, expr) {
        let parent_ctx = build_root_shortcut_parent_ctx(options, ctx, expr);
        let rule = crate::rules::arithmetic::CollapseExactOneShiftedQuotientRule;
        if let Some(rewrite) = crate::rule::Rule::apply(&rule, ctx, expr, &parent_ctx) {
            return Some(finish_root_shortcut_with_rewrite_meta(
                ctx,
                expr,
                rewrite,
                "Collapse Shifted Quotient of Equivalent Expressions",
                collect_steps,
            ));
        }
    }

    let Expr::Div(numerator, denominator) = ctx.get(expr) else {
        return None;
    };
    let numerator = *numerator;
    let denominator = *denominator;
    let passthrough_cores = strip_positive_one_passthrough_root(ctx, numerator)
        .zip(strip_positive_one_passthrough_root(ctx, denominator));
    if let Some((numerator_core, denominator_core)) = passthrough_cores {
        if matches_known_direct_pair_root(ctx, numerator_core, denominator_core)
            || matches_direct_trig_product_to_sum_sin_sin_pair_root(
                ctx,
                numerator_core,
                denominator_core,
            )
            || matches_direct_trig_product_to_sum_sin_cos_pair_root(
                ctx,
                numerator_core,
                denominator_core,
            )
            || matches_direct_trig_product_to_sum_cos_cos_pair_root(
                ctx,
                numerator_core,
                denominator_core,
            )
            || matches_direct_nested_fraction_simplified_pair_root(
                ctx,
                numerator_core,
                denominator_core,
            )
            || matches_direct_hyperbolic_sinh_sum_to_product_pair_root(
                ctx,
                numerator_core,
                denominator_core,
            )
            || matches_direct_hyperbolic_cosh_sum_to_product_pair_root(
                ctx,
                numerator_core,
                denominator_core,
            )
            || matches_direct_hyperbolic_cosh_difference_to_product_pair_root(
                ctx,
                numerator_core,
                denominator_core,
            )
            || matches_direct_recursive_hyperbolic_sinh_sum_pair_root(
                ctx,
                numerator_core,
                denominator_core,
            )
            || matches_direct_recursive_hyperbolic_cosh_sum_pair_root(
                ctx,
                numerator_core,
                denominator_core,
            )
            || matches_direct_trig_mixed_double_angle_pair_root(
                ctx,
                numerator_core,
                denominator_core,
            )
            || matches_direct_trig_cubic_cosine_pair_root(ctx, numerator_core, denominator_core)
            || matches_direct_trig_binomial_square_pair_root(ctx, numerator_core, denominator_core)
            || matches_direct_cos_square_diff_pair_root(ctx, numerator_core, denominator_core)
            || matches_direct_angle_sum_diff_pair_root(ctx, numerator_core, denominator_core)
        {
            let one = ctx.num(1);
            return Some(run_shifted_quotient_rebuilt_root_shortcut_simplify(
                options,
                ctx,
                expr,
                one,
                collect_steps,
            ));
        }
        if let Some((numerator_residual, denominator_residual)) =
            extract_shared_additive_passthrough_pair_cores_root(
                ctx,
                numerator_core,
                denominator_core,
            )
        {
            if matches_known_direct_pair_root(ctx, numerator_residual, denominator_residual)
                || matches_direct_trig_product_to_sum_sin_sin_pair_root(
                    ctx,
                    numerator_residual,
                    denominator_residual,
                )
                || matches_direct_trig_product_to_sum_sin_cos_pair_root(
                    ctx,
                    numerator_residual,
                    denominator_residual,
                )
                || matches_direct_trig_product_to_sum_cos_cos_pair_root(
                    ctx,
                    numerator_residual,
                    denominator_residual,
                )
                || matches_direct_nested_fraction_simplified_pair_root(
                    ctx,
                    numerator_residual,
                    denominator_residual,
                )
                || matches_direct_hyperbolic_sinh_sum_to_product_pair_root(
                    ctx,
                    numerator_residual,
                    denominator_residual,
                )
                || matches_direct_hyperbolic_cosh_sum_to_product_pair_root(
                    ctx,
                    numerator_residual,
                    denominator_residual,
                )
                || matches_direct_hyperbolic_cosh_difference_to_product_pair_root(
                    ctx,
                    numerator_residual,
                    denominator_residual,
                )
                || matches_direct_recursive_hyperbolic_sinh_sum_pair_root(
                    ctx,
                    numerator_residual,
                    denominator_residual,
                )
                || matches_direct_recursive_hyperbolic_cosh_sum_pair_root(
                    ctx,
                    numerator_residual,
                    denominator_residual,
                )
                || matches_direct_trig_mixed_double_angle_pair_root(
                    ctx,
                    numerator_residual,
                    denominator_residual,
                )
                || matches_direct_trig_cubic_cosine_pair_root(
                    ctx,
                    numerator_residual,
                    denominator_residual,
                )
                || matches_direct_trig_binomial_square_pair_root(
                    ctx,
                    numerator_residual,
                    denominator_residual,
                )
                || matches_direct_cos_square_diff_pair_root(
                    ctx,
                    numerator_residual,
                    denominator_residual,
                )
                || matches_direct_angle_sum_diff_pair_root(
                    ctx,
                    numerator_residual,
                    denominator_residual,
                )
            {
                let one = ctx.num(1);
                return Some(run_shifted_quotient_rebuilt_root_shortcut_simplify(
                    options,
                    ctx,
                    expr,
                    one,
                    collect_steps,
                ));
            }
        }
        if ((expr_contains_reciprocal_trig_builtin_local(ctx, numerator_core)
            && expr_contains_trig_or_hyperbolic_builtin_local(ctx, denominator_core))
            || (expr_contains_reciprocal_trig_builtin_local(ctx, denominator_core)
                && expr_contains_trig_or_hyperbolic_builtin_local(ctx, numerator_core)))
            && child_isolated_exact_zero(options, ctx, numerator_core)
            && child_isolated_exact_zero(options, ctx, denominator_core)
        {
            let one = ctx.num(1);
            return Some(run_shifted_quotient_rebuilt_root_shortcut_simplify(
                options,
                ctx,
                expr,
                one,
                collect_steps,
            ));
        }
        if is_nested_additive_log_residual_pair_root(ctx, numerator_core, denominator_core) {
            return None;
        }
    }

    let parent_ctx = build_root_shortcut_parent_ctx(options, ctx, expr);
    let rule = crate::rules::arithmetic::CollapseExactOneShiftedQuotientRule;
    let rewrite = if let Some(rewrite) = crate::rule::Rule::apply(&rule, ctx, expr, &parent_ctx) {
        rewrite
    } else {
        let numerator_core = strip_positive_one_passthrough_root(ctx, numerator)?;
        let denominator_core = strip_positive_one_passthrough_root(ctx, denominator)?;
        let residual_difference = ctx.add(Expr::Sub(numerator_core, denominator_core));

        let mut residual_simplifier = crate::Simplifier::with_default_rules();
        std::mem::swap(&mut residual_simplifier.context, ctx);
        let mut residual_orchestrator = Orchestrator::new();
        residual_orchestrator.options = SimplifyOptions {
            collect_steps: false,
            suppress_depth_overflow_warnings: true,
            ..options.clone()
        };
        let (residual_result, _residual_steps, _stats) =
            residual_orchestrator.simplify_pipeline(residual_difference, &mut residual_simplifier);
        std::mem::swap(&mut residual_simplifier.context, ctx);

        let zero = ctx.num(0);
        if compare_expr(ctx, residual_result, zero) != Ordering::Equal {
            return None;
        }

        crate::rule::Rewrite::with_local(
            ctx.add(Expr::Div(denominator, denominator)),
            "Equivalent Residual Cancellation",
            numerator,
            denominator,
        )
    };

    if let Expr::Div(numerator, denominator) = ctx.get(rewrite.new_expr) {
        if compare_expr(ctx, *numerator, *denominator) == Ordering::Equal {
            let one = ctx.num(1);
            let mut shortcut_steps = Vec::new();
            if collect_steps {
                shortcut_steps.push(build_root_shortcut_step_from_rewrite(
                    ctx,
                    expr,
                    &rewrite,
                    "Collapse Shifted Quotient of Equivalent Expressions",
                ));
                shortcut_steps.push(build_root_shortcut_compact_step(
                    rewrite.new_expr,
                    one,
                    "Cancelar numerador y denominador iguales",
                    "Simplificar fracción",
                ));
            }
            return Some((one, shortcut_steps));
        }
    }

    let mut simplifier = crate::Simplifier::with_default_rules();
    std::mem::swap(&mut simplifier.context, ctx);
    let (result, inner_steps, _stats) = simplifier.simplify_with_stats(
        rewrite.new_expr,
        crate::SimplifyOptions {
            suppress_depth_overflow_warnings: true,
            ..crate::SimplifyOptions::default()
        },
    );
    std::mem::swap(&mut simplifier.context, ctx);

    let one = ctx.num(1);
    if compare_expr(ctx, result, one) != Ordering::Equal {
        return None;
    }

    let mut shortcut_steps = Vec::new();
    if collect_steps {
        shortcut_steps.push(build_root_shortcut_step_from_rewrite(
            ctx,
            expr,
            &rewrite,
            "Collapse Shifted Quotient of Equivalent Expressions",
        ));
        shortcut_steps.extend(inner_steps);
    }

    Some((result, shortcut_steps))
}

fn try_standard_shifted_quotient_nested_zero_core_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let Expr::Div(numerator, denominator) = ctx.get(expr) else {
        return None;
    };
    let numerator = *numerator;
    let denominator = *denominator;
    let one = ctx.num(1);

    let numerator_core = strip_positive_one_passthrough_root(ctx, numerator);
    let denominator_core = strip_positive_one_passthrough_root(ctx, denominator);

    if let (Some(numerator_core), Some(denominator_core)) = (numerator_core, denominator_core) {
        if matches_direct_trig_product_to_sum_sin_sin_pair_root(
            ctx,
            numerator_core,
            denominator_core,
        ) || matches_direct_trig_product_to_sum_sin_cos_pair_root(
            ctx,
            numerator_core,
            denominator_core,
        ) || matches_direct_trig_product_to_sum_cos_cos_pair_root(
            ctx,
            numerator_core,
            denominator_core,
        ) || matches_direct_nested_fraction_simplified_pair_root(
            ctx,
            numerator_core,
            denominator_core,
        ) || matches_direct_hyperbolic_sinh_sum_to_product_pair_root(
            ctx,
            numerator_core,
            denominator_core,
        ) || matches_direct_hyperbolic_cosh_sum_to_product_pair_root(
            ctx,
            numerator_core,
            denominator_core,
        ) || matches_direct_hyperbolic_cosh_difference_to_product_pair_root(
            ctx,
            numerator_core,
            denominator_core,
        ) || matches_direct_recursive_hyperbolic_sinh_sum_pair_root(
            ctx,
            numerator_core,
            denominator_core,
        ) || matches_direct_recursive_hyperbolic_cosh_sum_pair_root(
            ctx,
            numerator_core,
            denominator_core,
        ) || matches_direct_cos_square_diff_pair_root(ctx, numerator_core, denominator_core)
            || matches_direct_trig_binomial_square_pair_root(ctx, numerator_core, denominator_core)
            || matches_direct_angle_sum_diff_pair_root(ctx, numerator_core, denominator_core)
        {
            return Some(run_shifted_quotient_rebuilt_root_shortcut_simplify(
                options,
                ctx,
                expr,
                one,
                collect_steps,
            ));
        }

        if matches_direct_small_zero_identity_root(ctx, numerator_core)
            && matches_direct_small_zero_identity_root(ctx, denominator_core)
        {
            return Some(run_shifted_quotient_rebuilt_root_shortcut_simplify(
                options,
                ctx,
                expr,
                one,
                collect_steps,
            ));
        }

        if ((expr_contains_reciprocal_trig_builtin_local(ctx, numerator_core)
            && expr_contains_trig_or_hyperbolic_builtin_local(ctx, denominator_core))
            || (expr_contains_reciprocal_trig_builtin_local(ctx, denominator_core)
                && expr_contains_trig_or_hyperbolic_builtin_local(ctx, numerator_core)))
            && child_isolated_exact_zero(options, ctx, numerator_core)
            && child_isolated_exact_zero(options, ctx, denominator_core)
        {
            return Some(run_shifted_quotient_rebuilt_root_shortcut_simplify(
                options,
                ctx,
                expr,
                one,
                collect_steps,
            ));
        }

        if (expr_contains_trig_or_hyperbolic_builtin_local(ctx, numerator_core)
            && child_matches_direct_or_isolated_exact_zero(options, ctx, numerator_core)
            && supported_nested_zero_partner_rewrites_to_zero(options, ctx, denominator_core))
            || (expr_contains_trig_or_hyperbolic_builtin_local(ctx, denominator_core)
                && child_matches_direct_or_isolated_exact_zero(options, ctx, denominator_core)
                && supported_nested_zero_partner_rewrites_to_zero(options, ctx, numerator_core))
        {
            return Some(run_shifted_quotient_rebuilt_root_shortcut_simplify(
                options,
                ctx,
                expr,
                one,
                collect_steps,
            ));
        }

        if (matches_narrow_trig_mixed_double_angle_zero_candidate_root(ctx, numerator_core)
            && child_isolated_exact_zero(options, ctx, numerator_core)
            && (matches_direct_small_zero_identity_root(ctx, denominator_core)
                || (is_small_trig_or_hyperbolic_zero_child(options, ctx, denominator_core)
                    && child_isolated_exact_zero(options, ctx, denominator_core))))
            || (matches_narrow_trig_mixed_double_angle_zero_candidate_root(ctx, denominator_core)
                && child_isolated_exact_zero(options, ctx, denominator_core)
                && (matches_direct_small_zero_identity_root(ctx, numerator_core)
                    || (is_small_trig_or_hyperbolic_zero_child(options, ctx, numerator_core)
                        && child_isolated_exact_zero(options, ctx, numerator_core))))
        {
            return Some(run_shifted_quotient_rebuilt_root_shortcut_simplify(
                options,
                ctx,
                expr,
                one,
                collect_steps,
            ));
        }

        let narrow_small_trig_zero_pair =
            (matches_direct_small_zero_identity_root(ctx, numerator_core)
                && (matches_direct_small_zero_identity_root(ctx, denominator_core)
                    || (is_small_trig_or_hyperbolic_zero_child(options, ctx, denominator_core)
                        && child_isolated_exact_zero(options, ctx, denominator_core))))
                || (matches_direct_small_zero_identity_root(ctx, denominator_core)
                    && (matches_direct_small_zero_identity_root(ctx, numerator_core)
                        || (is_small_trig_or_hyperbolic_zero_child(options, ctx, numerator_core)
                            && child_isolated_exact_zero(options, ctx, numerator_core))));
        if narrow_small_trig_zero_pair {
            return Some(run_shifted_quotient_rebuilt_root_shortcut_simplify(
                options,
                ctx,
                expr,
                one,
                collect_steps,
            ));
        }

        if expr_contains_trig_or_hyperbolic_builtin_local(ctx, numerator_core)
            && expr_contains_trig_or_hyperbolic_builtin_local(ctx, denominator_core)
        {
            let residual_difference = ctx.add(Expr::Sub(numerator_core, denominator_core));
            if child_isolated_exact_zero(options, ctx, residual_difference) {
                return Some(run_shifted_quotient_rebuilt_root_shortcut_simplify(
                    options,
                    ctx,
                    expr,
                    one,
                    collect_steps,
                ));
            }
        }
    }

    let new_numerator = numerator_core.and_then(|core| {
        (expr_contains_trig_or_hyperbolic_builtin_local(ctx, core)
            && denominator_core
                .is_some_and(|other| is_supported_nested_zero_child_partner(ctx, other))
            && child_isolated_exact_zero(options, ctx, core))
        .then_some(one)
    });
    let new_denominator = denominator_core.and_then(|core| {
        (expr_contains_trig_or_hyperbolic_builtin_local(ctx, core)
            && numerator_core
                .is_some_and(|other| is_supported_nested_zero_child_partner(ctx, other))
            && child_isolated_exact_zero(options, ctx, core))
        .then_some(one)
    });

    if new_numerator.is_none() && new_denominator.is_none() {
        return None;
    }

    let rewritten = match (new_numerator, new_denominator) {
        (Some(num), Some(den))
            if compare_expr(ctx, num, one) == Ordering::Equal
                && compare_expr(ctx, den, one) == Ordering::Equal =>
        {
            one
        }
        (Some(num), Some(den)) => ctx.add(Expr::Div(num, den)),
        (Some(num), None) => ctx.add(Expr::Div(num, denominator)),
        (None, Some(den)) => ctx.add(Expr::Div(numerator, den)),
        (None, None) => return None,
    };

    Some(run_rebuilt_root_shortcut_simplify(
        options,
        ctx,
        expr,
        rewritten,
        collect_steps,
    ))
}

fn try_standard_subtract_expanded_sum_diff_cubes_quotient_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let parent_ctx = build_root_shortcut_parent_ctx(options, ctx, expr);
    let rule = crate::rules::arithmetic::SubtractExpandedSumDiffCubesQuotientRule;
    let rewrite = crate::rule::Rule::apply(&rule, ctx, expr, &parent_ctx)?;
    Some(finish_standard_root_shortcut(
        ctx,
        expr,
        rewrite,
        "Subtract Expanded Sum/Difference of Cubes Quotient",
        collect_steps,
    ))
}

fn try_standard_extract_perfect_square_root_shortcut(
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let canonical = try_rewrite_canonical_root_expr(ctx, expr)?;
    let extract = try_rewrite_extract_perfect_power_from_radicand_expr(ctx, canonical.rewritten)?;

    let rewrite = crate::rule::Rewrite::new(extract.rewritten)
        .desc("Extract perfect square from under radical");
    Some(finish_standard_root_shortcut(
        ctx,
        expr,
        rewrite,
        "Extract Perfect Square from Radicand",
        collect_steps,
    ))
}

fn try_standard_numeric_add_chain_shortcut(
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

fn multiset_matches_exact(ctx: &Context, actual: &[ExprId], expected: &[ExprId]) -> bool {
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

fn is_exact_two_ab_product(ctx: &mut Context, expr: ExprId, a: ExprId, b: ExprId) -> bool {
    let view = MulView::from_expr(ctx, expr);
    if view.factors.len() != 3 {
        return false;
    }

    let two = ctx.num(2);
    multiset_matches_exact(ctx, &view.factors, &[two, a, b])
}

fn try_hidden_solve_root_binomial_square_shortcut(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let Expr::Div(num, den) = ctx.get(expr) else {
        return None;
    };
    let num = *num;
    let den = *den;

    let Expr::Pow(base, exp) = ctx.get(den) else {
        return None;
    };
    if !matches!(ctx.get(*exp), Expr::Number(n) if *n == BigRational::from_integer(2.into())) {
        return None;
    }
    let Expr::Add(a, b) = ctx.get(*base) else {
        return None;
    };
    let a = *a;
    let b = *b;

    let exp_two = ctx.num(2);
    let a_sq = ctx.add(Expr::Pow(a, exp_two));
    let exp_two_b = ctx.num(2);
    let b_sq = ctx.add(Expr::Pow(b, exp_two_b));

    let terms = AddView::from_expr(ctx, num).terms;
    if terms.len() != 3 {
        return None;
    }

    let mut squares = [None, None];
    let mut squares_len = 0usize;
    let mut middle = None;

    for (term, sign) in terms {
        if sign != Sign::Pos {
            return None;
        }

        if expr_eq(ctx, term, a_sq) || expr_eq(ctx, term, b_sq) {
            if squares_len >= squares.len() {
                return None;
            }
            squares[squares_len] = Some(term);
            squares_len += 1;
        } else if middle.is_none() {
            middle = Some(term);
        } else {
            return None;
        }
    }

    let (Some(left_sq), Some(right_sq), Some(middle_term)) = (squares[0], squares[1], middle)
    else {
        return None;
    };

    if !multiset_matches_exact(ctx, &[left_sq, right_sq], &[a_sq, b_sq])
        || !is_exact_two_ab_product(ctx, middle_term, a, b)
    {
        return None;
    }

    Some(ctx.num(1))
}

fn try_hidden_solve_root_perfect_square_minus_shortcut(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let Expr::Div(num, den) = ctx.get(expr) else {
        return None;
    };
    let num = *num;
    let den = *den;

    let Expr::Sub(a, b) = ctx.get(den) else {
        return None;
    };
    let a = *a;
    let b = *b;

    let terms = AddView::from_expr(ctx, num).terms;
    if terms.len() != 3 {
        return None;
    }

    let exp_two = ctx.num(2);
    let a_sq = ctx.add(Expr::Pow(a, exp_two));
    let exp_two_b = ctx.num(2);
    let b_sq = ctx.add(Expr::Pow(b, exp_two_b));

    let mut positives = [None, None];
    let mut positive_count = 0usize;
    let mut negative = None;

    for (term, sign) in terms {
        match sign {
            Sign::Pos => {
                if positive_count >= positives.len() {
                    return None;
                }
                positives[positive_count] = Some(term);
                positive_count += 1;
            }
            Sign::Neg => {
                if negative.is_some() {
                    return None;
                }
                negative = Some(term);
            }
        }
    }

    let (Some(left_pos), Some(right_pos), Some(negative_term)) =
        (positives[0], positives[1], negative)
    else {
        return None;
    };

    if !multiset_matches_exact(ctx, &[left_pos, right_pos], &[a_sq, b_sq])
        || !is_exact_two_ab_product(ctx, negative_term, a, b)
    {
        return None;
    }

    Some(den)
}

fn try_hidden_solve_root_difference_of_squares_shortcut(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let Expr::Div(num, den) = ctx.get(expr) else {
        return None;
    };
    let num = *num;
    let den = *den;

    let Expr::Sub(left, right) = ctx.get(num) else {
        return None;
    };
    let left = *left;
    let right = *right;
    let a = square_of_symbolic_atom(ctx, left)?;
    let b = square_of_symbolic_atom(ctx, right)?;

    if expr_eq(ctx, a, b) {
        return None;
    }

    match ctx.get(den) {
        Expr::Sub(dl, dr) if expr_eq(ctx, *dl, a) && expr_eq(ctx, *dr, b) => {
            Some(ctx.add(Expr::Add(a, b)))
        }
        Expr::Add(dl, dr) if expr_eq(ctx, *dl, a) && expr_eq(ctx, *dr, b) => {
            Some(ctx.add(Expr::Sub(a, b)))
        }
        _ => None,
    }
}

fn allow_hidden_solve_root_scalar_multiple_shortcut(opts: &SimplifyOptions) -> bool {
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

fn allow_definability_root_shortcuts(opts: &SimplifyOptions) -> bool {
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

fn prove_positive_literal_fast(ctx: &Context, expr: ExprId) -> Option<crate::Proof> {
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

fn try_hidden_solve_root_power_quotient_shortcut(
    ctx: &mut Context,
    expr: ExprId,
    _domain_mode: crate::DomainMode,
) -> Option<ExprId> {
    let plan = try_rewrite_cancel_same_base_powers_div_expr(ctx, expr)?;
    Some(plan.rewritten)
}

fn try_hidden_solve_root_identical_atom_fraction_shortcut(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let Expr::Div(num, den) = ctx.get(expr) else {
        return None;
    };
    let num = *num;
    let den = *den;

    if !is_symbolic_atom(ctx, num) || !is_symbolic_atom(ctx, den) || !expr_eq(ctx, num, den) {
        return None;
    }

    Some(ctx.num(1))
}

fn try_hidden_solve_root_exact_two_term_scalar_multiple_shortcut(
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

fn try_standard_exact_two_term_scalar_multiple_shortcut(
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

fn try_standard_small_polynomial_denominator_factor_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let Expr::Div(num, den) = ctx.get(expr) else {
        return None;
    };
    let num = *num;
    let den = *den;

    if !matches!(ctx.get(den), Expr::Add(_, _) | Expr::Sub(_, _)) {
        return None;
    }
    if cas_ast::count_nodes(ctx, den) > 15 {
        return None;
    }
    if cas_ast::collect_variables(ctx, den).len() != 1 {
        return None;
    }

    let rewrite = try_rewrite_automatic_factor_expr(ctx, den)?;
    let rewritten = ctx.add(Expr::Div(num, rewrite.rewritten));
    Some(run_named_rebuilt_root_shortcut_simplify(
        options,
        ctx,
        expr,
        rewritten,
        "Factorizar denominador polinómico",
        "Factor Polynomial Denominator",
        collect_steps,
    ))
}

fn try_hidden_solve_root_log_power_base_shortcut(
    ctx: &mut Context,
    expr: ExprId,
    domain_mode: crate::DomainMode,
    value_domain: crate::ValueDomain,
) -> Option<(ExprId, Vec<Step>)> {
    use crate::{ImplicitCondition, Proof};

    let planned = try_rewrite_log_power_base_numeric_expr(ctx, expr)?;
    let mode = log_exp_inverse_policy_mode_from_flags(
        matches!(domain_mode, crate::DomainMode::Assume),
        matches!(domain_mode, crate::DomainMode::Strict),
    );
    let one = ctx.num(1);
    let base_positive_proven = if domain_mode.is_generic() {
        matches!(
            prove_positive_literal_fast(ctx, planned.base_core),
            Some(Proof::Proven)
        )
    } else {
        crate::prove_positive(ctx, planned.base_core, value_domain) == Proof::Proven
    };
    let policy = plan_log_power_base_numeric_policy(
        mode,
        value_domain == crate::ValueDomain::ComplexEnabled,
        base_positive_proven,
        cas_ast::ordering::compare_expr(ctx, planned.base_core, one) == std::cmp::Ordering::Equal,
    );

    let cas_math::logarithm_inverse_support::LogPowerBasePolicyPlan::Rewrite {
        require_positive_base,
        require_nonzero_base_minus_one: _,
    } = policy
    else {
        return None;
    };

    if !require_positive_base {
        return Some((planned.rewritten, Vec::new()));
    }

    let mut step = Step::new_compact(
        "log(a^m, a^n) = n/m",
        "Log Power Base",
        expr,
        planned.rewritten,
    );
    step.soundness = crate::SoundnessLabel::EquivalenceUnderIntroducedRequires;
    {
        let meta = step.meta_mut();
        if require_positive_base {
            meta.required_conditions
                .push(ImplicitCondition::Positive(planned.base_core));
        }
    }

    Some((planned.rewritten, vec![step]))
}

fn is_plain_symbolic_cube_trinomial_after_core(ctx: &Context, expr: ExprId) -> bool {
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

fn is_symbolic_power_over_same_atom_noop_after_core(ctx: &Context, expr: ExprId) -> bool {
    let Expr::Div(num, den) = ctx.get(expr) else {
        return false;
    };
    let Expr::Pow(base, exp) = ctx.get(*num) else {
        return false;
    };
    if cas_ast::ordering::compare_expr(ctx, *base, *den) != std::cmp::Ordering::Equal {
        return false;
    }

    matches!(ctx.get(*base), Expr::Variable(_) | Expr::Constant(_))
        && matches!(ctx.get(*exp), Expr::Variable(_) | Expr::Constant(_))
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

fn is_real_domain_complex_noop_root(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Pow(base, exp) => {
            matches!(ctx.get(*base), Expr::Constant(cas_ast::Constant::I))
                && matches!(ctx.get(*exp), Expr::Number(n) if n.is_integer())
        }
        Expr::Div(num, den) => {
            is_exact_gaussian_noop_component(ctx, *num)
                && is_exact_gaussian_noop_component(ctx, *den)
        }
        _ => false,
    }
}

pub struct Orchestrator {
    // Configuration for the pipeline
    pub max_iterations: usize,
    pub enable_polynomial_strategy: bool,
    /// Pre-scanned pattern marks for context-aware guards
    pub pattern_marks: crate::pattern_marks::PatternMarks,
    /// Expr these marks were last computed for; reused when the tree is unchanged.
    pub pattern_marks_expr: Option<ExprId>,
    /// Pipeline options (budgets, transform/rationalize control)
    pub options: SimplifyOptions,
}

impl Default for Orchestrator {
    fn default() -> Self {
        Self::new()
    }
}

impl Orchestrator {
    pub fn new() -> Self {
        Self {
            max_iterations: 10,
            enable_polynomial_strategy: true,
            pattern_marks: crate::pattern_marks::PatternMarks::new(),
            pattern_marks_expr: None,
            options: SimplifyOptions::default(),
        }
    }

    /// Create orchestrator for expand() command (no rationalization)
    pub fn for_expand() -> Self {
        let mut o = Self::new();
        o.options = SimplifyOptions::for_expand();
        o
    }

    /// Run a single phase of the pipeline until fixed point or budget exhausted.
    ///
    /// Returns the simplified expression, steps, and phase statistics.
    fn run_phase(
        &mut self,
        simplifier: &mut Simplifier,
        start: ExprId,
        phase: SimplifyPhase,
        max_iters: usize,
    ) -> (ExprId, Vec<Step>, crate::phase::PhaseStats) {
        use crate::phase::PhaseStats;

        let mut current = start;
        let mut all_steps = Vec::new();
        let mut seen_hashes: HashSet<u64> = HashSet::new();
        let mut stats = PhaseStats::new(phase);

        tracing::debug!(
            target: "simplify",
            phase = %phase,
            budget = max_iters,
            "phase_start"
        );

        for iter in 0..max_iters {
            let is_solve_mode =
                self.options.shared.context_mode == crate::options::ContextMode::Solve;
            if self.pattern_marks_expr != Some(current) {
                self.pattern_marks = crate::pattern_marks::PatternMarks::new();
                crate::pattern_scanner::scan_and_mark_patterns(
                    &simplifier.context,
                    current,
                    &mut self.pattern_marks,
                );

                // Auto-expand scanner: mark cancellation contexts (difference quotients)
                // Only skip in Solve mode (which should never auto-expand to preserve structure)
                // The scanner has its own strict budgets (n=2, base_terms<=3) so it's safe to always run
                if !is_solve_mode {
                    let math_budget =
                        to_math_auto_expand_budget(&self.options.shared.expand_budget);
                    cas_math::auto_expand_scan::mark_auto_expand_candidates(
                        &simplifier.context,
                        current,
                        &math_budget,
                        &mut self.pattern_marks,
                    );
                }
                self.pattern_marks_expr = Some(current);
            }
            let global_auto_expand = self.options.shared.expand_policy
                == crate::phase::ExpandPolicy::Auto
                && !is_solve_mode;
            let config = crate::engine::LoopConfig {
                phase,
                expand_mode: self.options.expand_mode,
                auto_expand: global_auto_expand,
                expand_budget: self.options.shared.expand_budget,
                domain_mode: self.options.shared.semantics.domain_mode,
                inv_trig: self.options.shared.semantics.inv_trig,
                value_domain: self.options.shared.semantics.value_domain,
                goal: self.options.goal,
                simplify_purpose: self.options.simplify_purpose,
                context_mode: self.options.shared.context_mode,
                autoexpand_binomials: self.options.shared.autoexpand_binomials,
                heuristic_poly: self.options.shared.heuristic_poly,
                suppress_depth_overflow_warnings: self.options.suppress_depth_overflow_warnings,
            };
            let (next, steps, pass_stats) =
                simplifier.apply_rules_loop_with_config(current, &self.pattern_marks, &config);

            // Log budget stats for this iteration (actual charging done by caller if Budget provided)
            if pass_stats.rewrite_count > 0 || pass_stats.nodes_delta > 0 {
                tracing::trace!(
                    target: "budget",
                    op = %pass_stats.op,
                    rewrites = pass_stats.rewrite_count,
                    nodes_delta = pass_stats.nodes_delta,
                    "pass_budget_stats"
                );
            }

            // Warn user when budget limit was reached (best-effort mode)
            if let Some(ref exceeded) = pass_stats.stop_reason {
                tracing::warn!(
                    target: "budget",
                    op = %exceeded.op,
                    metric = %exceeded.metric,
                    used = exceeded.used,
                    limit = exceeded.limit,
                    "Budget limit reached: {}/{} (used {}, limit {}). Returned partial result.",
                    exceeded.op,
                    exceeded.metric,
                    exceeded.used,
                    exceeded.limit
                );
            }

            stats.rewrites_used += steps.len();
            all_steps.extend(steps);

            // Hidden solve fast path: once Core collapses to a terminal value or a
            // plain symbolic closed form, another full Core pass is only paying the
            // fixed-point check. Later pipeline decisions are still made by the
            // caller after this phase returns.
            if phase == SimplifyPhase::Core
                && !self.options.collect_steps
                && is_solve_mode
                && next != current
                && (is_terminal_after_core(&simplifier.context, next)
                    || is_plain_symbolic_binomial_after_core(&simplifier.context, next)
                    || is_plain_symbolic_cube_trinomial_after_core(&simplifier.context, next)
                    || (!self.options.shared.semantics.domain_mode.is_strict()
                        && matches!(simplifier.context.get(current), Expr::Div(_, _))
                        && is_plain_symbolic_power_after_core(&simplifier.context, next)))
            {
                current = next;
                stats.iters_used = iter + 1;
                tracing::debug!(
                    target: "simplify",
                    phase = %phase,
                    iters = stats.iters_used,
                    rewrites = stats.rewrites_used,
                    "phase_early_exit_after_closed_form"
                );
                break;
            }

            // Fixed point check
            if next == current {
                stats.iters_used = iter + 1;
                tracing::debug!(
                    target: "simplify",
                    phase = %phase,
                    iters = stats.iters_used,
                    rewrites = stats.rewrites_used,
                    "phase_fixed_point"
                );
                break;
            }

            // Cycle detection: HashSet catches cycles of any period
            let hash = cas_math::expr_semantic_hash::semantic_hash(&simplifier.context, current);
            if !seen_hashes.insert(hash) {
                // Emit cycle event for the registry
                cas_solver_core::cycle_event_registry::register_cycle_event_for_expr(
                    &simplifier.context,
                    current,
                    phase,
                    0, // unknown period at inter-iteration level
                    cas_solver_core::cycle_models::CycleLevel::InterIteration,
                    "(inter-iteration)",
                    hash,
                    iter,
                );
                stats.iters_used = iter + 1;
                tracing::warn!(
                    target: "simplify",
                    phase = %phase,
                    iters = stats.iters_used,
                    "cycle_detected"
                );
                break;
            }

            current = next;
            stats.iters_used = iter + 1;
        }

        stats.changed = current != start;

        tracing::debug!(
            target: "simplify",
            phase = %phase,
            iters = stats.iters_used,
            rewrites = stats.rewrites_used,
            changed = stats.changed,
            "phase_end"
        );

        (current, all_steps, stats)
    }

    /// Simplify using explicit phase pipeline.
    ///
    /// Pipeline order: Core → Transform → Rationalize → PostCleanup
    ///
    /// Key invariant: Transform never runs after Rationalize.
    pub fn simplify_pipeline(
        &mut self,
        expr: ExprId,
        simplifier: &mut Simplifier,
    ) -> (ExprId, Vec<Step>, crate::phase::PipelineStats) {
        // Extract collect_steps early so pre-passes can skip Step construction
        let collect_steps = self.options.collect_steps;
        let is_solve_mode = self.options.shared.context_mode == crate::options::ContextMode::Solve;
        self.pattern_marks_expr = None;

        // Narrow hidden solve root shortcuts. Keep them limited to the
        // no-steps, no-listener solve path and dispatch by root kind so we do
        // not pay unrelated matchers on every expression.
        if self.options.shared.context_mode == crate::options::ContextMode::Standard
            && self.options.shared.semantics.value_domain == crate::semantics::ValueDomain::RealOnly
            && !simplifier.has_step_listener()
            && is_real_domain_complex_noop_root(&simplifier.context, expr)
        {
            return (expr, Vec::new(), crate::phase::PipelineStats::default());
        }

        if matches!(
            self.options.shared.context_mode,
            crate::options::ContextMode::Standard | crate::options::ContextMode::Auto
        ) {
            let add_root = matches!(simplifier.context.get(expr), Expr::Add(_, _));
            let sub_root = matches!(simplifier.context.get(expr), Expr::Sub(_, _));
            let div_root = matches!(simplifier.context.get(expr), Expr::Div(_, _));
            let mul_root = matches!(simplifier.context.get(expr), Expr::Mul(_, _));

            // These exact-equivalence shortcuts emit proper didactic steps, so keep
            // them available even when the caller requested step collection.
            if mul_root {
                if let Some((result, shortcut_steps)) =
                    try_standard_direct_scaled_half_angle_square_shortcut(
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_collapsed_fraction_direct_pair_factor_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_collapsed_fraction_factored_numerator_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_collapsed_fraction_partner_canonicalization_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_embedded_trig_product_to_sum_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_collapsed_fraction_hyperbolic_half_angle_factor_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_tangent_addition_fraction_product_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_trig_product_to_sum_subset_factor_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_perfect_square_trinomial_factor_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_sum_of_squares_product_subset_factor_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_square_anchor_linear_shift_partner_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_three_linear_shift_anchor_direct_partner_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_two_factor_small_partner_canonicalization_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_two_factor_direct_pair_anchor_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_inverse_trig_anchor_small_polynomial_partner_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_safe_anchor_small_polynomial_partner_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_scaled_half_angle_anchor_direct_partner_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_safe_anchor_direct_partner_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_special_angle_exact_value_factor_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_tangent_addition_factor_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_direct_scaled_half_angle_square_shortcut(
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_half_angle_square_factor_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_trig_power_reduction_factor_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_positive_double_cos_square_diff_factor_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) = try_standard_direct_small_zero_pair_shortcut(
                    &self.options,
                    &mut simplifier.context,
                    expr,
                    collect_steps,
                ) {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_reciprocal_trig_zero_pair_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) = try_standard_small_trig_zero_pair_shortcut(
                    &self.options,
                    &mut simplifier.context,
                    expr,
                    collect_steps,
                ) {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_direct_trig_mixed_zero_pair_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_zero_product_with_exact_zero_child_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_embedded_trig_product_to_sum_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_assumed_dyadic_cos_product_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
            }
            if add_root || sub_root {
                if let Some((result, shortcut_steps)) =
                    try_standard_rational_half_angle_target_passthrough_shortcut(
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_scaled_sin_fourth_power_reduction_zero_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_trig_fourth_power_difference_shortcut(
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_reciprocal_pythagorean_pair_shortcut(
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_shared_passthrough_direct_pair_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_two_factor_product_pair_zero_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                let add_term_count = AddView::from_expr(&simplifier.context, expr).terms.len();
                let raw_binary_pythagorean_identity =
                    crate::rules::arithmetic::extract_two_term_core_difference(
                        &mut simplifier.context,
                        expr,
                    )
                    .is_some_and(|(lhs_core, rhs_core)| {
                        matches_direct_pythagorean_identity_pair_root(
                            &mut simplifier.context,
                            lhs_core,
                            rhs_core,
                        )
                    });
                if add_term_count > 2
                    && !raw_binary_pythagorean_identity
                    && has_structural_numeric_pythagorean_pair(&simplifier.context, expr)
                {
                    if let Some((result, shortcut_steps)) =
                        try_standard_pythagorean_additive_shortcut(
                            &mut simplifier.context,
                            expr,
                            collect_steps,
                        )
                    {
                        return (
                            result,
                            shortcut_steps,
                            crate::phase::PipelineStats::default(),
                        );
                    }
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_repeated_phase_shift_pair_shortcut(
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) = try_standard_direct_known_pair_zero_shortcut(
                    &self.options,
                    &mut simplifier.context,
                    expr,
                    collect_steps,
                ) {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_common_scale_known_pair_shortcut(
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_direct_sum_to_product_root_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_direct_trig_sum_product_zero_shortcut(
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_direct_small_zero_identity_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_nested_exact_zero_child_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_direct_small_zero_additive_combination_shortcut(
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_partitioned_direct_small_zero_sum_shortcut(
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) = try_standard_direct_small_zero_pair_shortcut(
                    &self.options,
                    &mut simplifier.context,
                    expr,
                    collect_steps,
                ) {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_reciprocal_pythagorean_zero_shortcut(
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_trig_power_reduction_zero_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_subtract_expanded_sum_diff_cubes_quotient_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_shared_passthrough_small_pow_expansion_shortcut(
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_small_composed_additive_pair_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_shared_passthrough_direct_pair_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_common_scale_known_pair_shortcut(
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_reciprocal_trig_zero_pair_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) = try_standard_small_trig_zero_pair_shortcut(
                    &self.options,
                    &mut simplifier.context,
                    expr,
                    collect_steps,
                ) {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_direct_trig_mixed_zero_pair_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_guarded_small_zero_pair_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_hyperbolic_cosh_cubic_subset_zero_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) = try_standard_half_angle_subset_zero_shortcut(
                    &self.options,
                    &mut simplifier.context,
                    expr,
                    collect_steps,
                ) {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_shared_passthrough_pythagorean_factor_form_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if !is_nested_additive_pair_root(&simplifier.context, expr) {
                    if let Some((result, shortcut_steps)) =
                        try_standard_pythagorean_generic_coefficient_shortcut(
                            &self.options,
                            &mut simplifier.context,
                            expr,
                            collect_steps,
                        )
                    {
                        return (
                            result,
                            shortcut_steps,
                            crate::phase::PipelineStats::default(),
                        );
                    }
                }
                if is_mixed_sign_trig_square_difference_root(&simplifier.context, expr) {
                    return (expr, Vec::new(), crate::phase::PipelineStats::default());
                }
                if has_negative_numeric_pythagorean_pair(&simplifier.context, expr) {
                    if let Some((result, shortcut_steps)) =
                        try_standard_pythagorean_additive_shortcut(
                            &mut simplifier.context,
                            expr,
                            collect_steps,
                        )
                    {
                        return (
                            result,
                            shortcut_steps,
                            crate::phase::PipelineStats::default(),
                        );
                    }
                }
                if has_numeric_pythagorean_complement_pair(&simplifier.context, expr) {
                    if let Some((result, shortcut_steps)) =
                        try_standard_pythagorean_additive_shortcut(
                            &mut simplifier.context,
                            expr,
                            collect_steps,
                        )
                    {
                        return (
                            result,
                            shortcut_steps,
                            crate::phase::PipelineStats::default(),
                        );
                    }
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_reciprocal_product_pythagorean_zero_shortcut(
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_trig_binomial_square_double_angle_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_sin_sum_triple_identity_zero_shortcut(
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_trig_fourth_power_difference_shortcut(
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if !is_nested_additive_pair_root(&simplifier.context, expr) {
                    if let Some((result, shortcut_steps)) =
                        try_standard_exact_zero_equivalence_shortcut(
                            &self.options,
                            &mut simplifier.context,
                            expr,
                            collect_steps,
                        )
                    {
                        return (
                            result,
                            shortcut_steps,
                            crate::phase::PipelineStats::default(),
                        );
                    }
                }
            }
            if div_root {
                if let Some((result, shortcut_steps)) =
                    try_standard_direct_positive_double_cos_square_diff_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_direct_trig_power_reduction_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_direct_half_angle_square_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                let div_pair = match simplifier.context.get(expr) {
                    Expr::Div(numerator, denominator) => Some((*numerator, *denominator)),
                    _ => None,
                };
                let try_exact_one_first = div_pair.is_some_and(|(numerator, denominator)| {
                    strip_positive_one_passthrough_root(&mut simplifier.context, numerator)
                        .zip(strip_positive_one_passthrough_root(
                            &mut simplifier.context,
                            denominator,
                        ))
                        .is_some_and(|(numerator_core, denominator_core)| {
                            let shared_passthrough_direct_pair =
                                extract_shared_additive_passthrough_pair_cores_root(
                                    &mut simplifier.context,
                                    numerator_core,
                                    denominator_core,
                                )
                                .is_some_and(|(numerator_residual, denominator_residual)| {
                                    matches_direct_trig_product_to_sum_sin_sin_pair_root(
                                        &mut simplifier.context,
                                        numerator_residual,
                                        denominator_residual,
                                    ) || matches_direct_trig_product_to_sum_sin_cos_pair_root(
                                        &mut simplifier.context,
                                        numerator_residual,
                                        denominator_residual,
                                    ) || matches_direct_trig_product_to_sum_cos_cos_pair_root(
                                        &mut simplifier.context,
                                        numerator_residual,
                                        denominator_residual,
                                    ) || matches_direct_nested_fraction_simplified_pair_root(
                                        &mut simplifier.context,
                                        numerator_residual,
                                        denominator_residual,
                                    ) || matches_direct_hyperbolic_sinh_sum_to_product_pair_root(
                                        &mut simplifier.context,
                                        numerator_residual,
                                        denominator_residual,
                                    ) || matches_direct_hyperbolic_cosh_sum_to_product_pair_root(
                                        &mut simplifier.context,
                                        numerator_residual,
                                        denominator_residual,
                                    ) || matches_direct_hyperbolic_cosh_difference_to_product_pair_root(
                                        &mut simplifier.context,
                                        numerator_residual,
                                        denominator_residual,
                                    ) || matches_direct_recursive_hyperbolic_sinh_sum_pair_root(
                                        &mut simplifier.context,
                                        numerator_residual,
                                        denominator_residual,
                                    ) || matches_direct_recursive_hyperbolic_cosh_sum_pair_root(
                                        &mut simplifier.context,
                                        numerator_residual,
                                        denominator_residual,
                                    ) || matches_direct_trig_mixed_double_angle_pair_root(
                                        &mut simplifier.context,
                                        numerator_residual,
                                        denominator_residual,
                                    ) || matches_direct_trig_cubic_cosine_pair_root(
                                        &mut simplifier.context,
                                        numerator_residual,
                                        denominator_residual,
                                    ) || matches_direct_cos_square_diff_pair_root(
                                        &mut simplifier.context,
                                        numerator_residual,
                                        denominator_residual,
                                    ) || matches_direct_angle_sum_diff_pair_root(
                                        &mut simplifier.context,
                                        numerator_residual,
                                        denominator_residual,
                                    )
                                });
                            matches_direct_half_angle_binomial_square_pair_root(
                                &mut simplifier.context,
                                numerator_core,
                                denominator_core,
                            ) || matches_direct_pythagorean_factor_form_pair_root(
                                &mut simplifier.context,
                                numerator_core,
                                denominator_core,
                            ) || matches_direct_trig_product_to_sum_sin_sin_pair_root(
                                &mut simplifier.context,
                                numerator_core,
                                denominator_core,
                            ) || matches_direct_trig_product_to_sum_sin_cos_pair_root(
                                &mut simplifier.context,
                                numerator_core,
                                denominator_core,
                            ) || matches_direct_trig_product_to_sum_cos_cos_pair_root(
                                &mut simplifier.context,
                                numerator_core,
                                denominator_core,
                            ) || matches_direct_nested_fraction_simplified_pair_root(
                                &mut simplifier.context,
                                numerator_core,
                                denominator_core,
                            ) || matches_direct_hyperbolic_sinh_sum_to_product_pair_root(
                                &mut simplifier.context,
                                numerator_core,
                                denominator_core,
                            ) || matches_direct_hyperbolic_cosh_sum_to_product_pair_root(
                                &mut simplifier.context,
                                numerator_core,
                                denominator_core,
                            ) || matches_direct_hyperbolic_cosh_difference_to_product_pair_root(
                                &mut simplifier.context,
                                numerator_core,
                                denominator_core,
                            ) || matches_direct_recursive_hyperbolic_sinh_sum_pair_root(
                                &mut simplifier.context,
                                numerator_core,
                                denominator_core,
                            ) || matches_direct_recursive_hyperbolic_cosh_sum_pair_root(
                                &mut simplifier.context,
                                numerator_core,
                                denominator_core,
                            ) || matches_direct_trig_mixed_double_angle_pair_root(
                                &mut simplifier.context,
                                numerator_core,
                                denominator_core,
                            ) || matches_direct_trig_cubic_cosine_pair_root(
                                &mut simplifier.context,
                                numerator_core,
                                denominator_core,
                            ) || matches_direct_cos_square_diff_pair_root(
                                &mut simplifier.context,
                                numerator_core,
                                denominator_core,
                            ) || matches_direct_angle_sum_diff_pair_root(
                                &mut simplifier.context,
                                numerator_core,
                                denominator_core,
                            ) || shared_passthrough_direct_pair
                        })
                });
                if try_exact_one_first {
                    if let Some((result, shortcut_steps)) =
                        try_standard_shifted_quotient_exact_one_shortcut(
                            &self.options,
                            &mut simplifier.context,
                            expr,
                            collect_steps,
                        )
                    {
                        return (
                            result,
                            shortcut_steps,
                            crate::phase::PipelineStats::default(),
                        );
                    }
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_shifted_quotient_nested_zero_core_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_sum_diff_cubes_fraction_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_shifted_quotient_exact_one_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
            }
        }

        if matches!(
            self.options.shared.context_mode,
            crate::options::ContextMode::Standard | crate::options::ContextMode::Auto
        ) && !simplifier.has_step_listener()
        {
            let mut shortcut_steps = Vec::new();
            let allow_definability_shortcuts = allow_definability_root_shortcuts(&self.options);
            let add_root = matches!(simplifier.context.get(expr), Expr::Add(_, _));
            let sub_root = matches!(simplifier.context.get(expr), Expr::Sub(_, _));
            let div_root = matches!(simplifier.context.get(expr), Expr::Div(_, _));
            let pow_root = matches!(simplifier.context.get(expr), Expr::Pow(_, _));
            if !is_nested_additive_pair_root(&simplifier.context, expr) {
                if let Some((result, shortcut_steps)) =
                    try_standard_pythagorean_generic_coefficient_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
            }
            if div_root {
                if let Some((result, shortcut_steps)) =
                    try_standard_small_polynomial_denominator_factor_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
            }
            if add_root || sub_root {
                if let Some((result, shortcut_steps)) =
                    try_standard_direct_small_zero_identity_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_direct_small_zero_additive_combination_shortcut(
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_partitioned_direct_small_zero_sum_shortcut(
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) = try_standard_direct_small_zero_pair_shortcut(
                    &self.options,
                    &mut simplifier.context,
                    expr,
                    collect_steps,
                ) {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_reciprocal_pythagorean_zero_shortcut(
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_subtract_expanded_sum_diff_cubes_quotient_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
            }
            if sub_root {
                if let Some((result, shortcut_steps)) = try_standard_sub_self_cancel_shortcut(
                    &self.options,
                    &mut simplifier.context,
                    expr,
                    collect_steps,
                ) {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
            }
            if add_root {
                if is_symbolic_atom_plus_nonzero_literal_root(&simplifier.context, expr) {
                    return (expr, Vec::new(), crate::phase::PipelineStats::default());
                }
                if let Some((result, shortcut_steps)) = try_standard_numeric_add_chain_shortcut(
                    &mut simplifier.context,
                    expr,
                    collect_steps,
                ) {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_reciprocal_product_pythagorean_zero_shortcut(
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_trig_binomial_square_double_angle_shortcut(
                        &self.options,
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_sin_sum_triple_identity_zero_shortcut(
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_trig_fourth_power_difference_shortcut(
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if try_rewrite_pythagorean_chain_add_expr(&mut simplifier.context, expr).is_some() {
                    if let Some((result, shortcut_steps)) =
                        try_standard_pythagorean_additive_shortcut(
                            &mut simplifier.context,
                            expr,
                            collect_steps,
                        )
                    {
                        return (
                            result,
                            shortcut_steps,
                            crate::phase::PipelineStats::default(),
                        );
                    }
                }
            }

            if matches!(simplifier.context.get(expr), Expr::Function(_, _)) {
                if let Some((result, shortcut_steps)) = try_standard_abs_shortcut(
                    &self.options,
                    &mut simplifier.context,
                    expr,
                    collect_steps,
                ) {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) = try_standard_simplify_square_root_shortcut(
                    &mut simplifier.context,
                    expr,
                    collect_steps,
                ) {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
                if let Some((result, shortcut_steps)) =
                    try_standard_extract_perfect_square_root_shortcut(
                        &mut simplifier.context,
                        expr,
                        collect_steps,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
            }

            if pow_root {
                let parent_ctx =
                    build_root_shortcut_parent_ctx(&self.options, &simplifier.context, expr);
                let root_pow_cancel = crate::rules::exponents::RootPowCancelRule;
                if let Some(rewrite) = crate::rule::Rule::apply(
                    &root_pow_cancel,
                    &mut simplifier.context,
                    expr,
                    &parent_ctx,
                ) {
                    let (result, shortcut_steps) = finish_standard_root_shortcut(
                        &simplifier.context,
                        expr,
                        rewrite,
                        "Root Power Cancel",
                        collect_steps,
                    );
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
            }

            let div_parts = match simplifier.context.get(expr) {
                Expr::Div(num, den) => Some((*num, *den)),
                _ => None,
            };
            if let Some((num, den)) = div_parts {
                if allow_definability_shortcuts {
                    if let Some(result) = crate::rules::algebra::try_difference_of_squares_preorder(
                        &mut simplifier.context,
                        expr,
                        num,
                        den,
                        self.options.shared.semantics.domain_mode,
                        self.options.shared.semantics.value_domain,
                        self.options.shared.semantics.value_domain
                            == crate::semantics::ValueDomain::RealOnly,
                        collect_steps,
                        &mut shortcut_steps,
                        &[],
                    ) {
                        return (
                            result,
                            shortcut_steps,
                            crate::phase::PipelineStats::default(),
                        );
                    }
                    if let Some(result) = crate::rules::algebra::try_sum_diff_of_cubes_preorder(
                        &mut simplifier.context,
                        expr,
                        num,
                        den,
                        collect_steps,
                        &mut shortcut_steps,
                        &[],
                    ) {
                        return (
                            result,
                            shortcut_steps,
                            crate::phase::PipelineStats::default(),
                        );
                    }
                    if let Some(result) =
                        crate::rules::algebra::try_exact_common_factor_mul_fraction_preorder(
                            &mut simplifier.context,
                            expr,
                            num,
                            den,
                            self.options.shared.semantics.domain_mode,
                            self.options.shared.semantics.value_domain,
                            collect_steps,
                            &mut shortcut_steps,
                            &[],
                        )
                    {
                        return (
                            result,
                            shortcut_steps,
                            crate::phase::PipelineStats::default(),
                        );
                    }
                    if let Some((result, shortcut_steps)) =
                        try_standard_exact_two_term_scalar_multiple_shortcut(
                            &mut simplifier.context,
                            expr,
                            self.options.shared.semantics.domain_mode,
                            self.options.shared.semantics.value_domain,
                            collect_steps,
                        )
                    {
                        return (
                            result,
                            shortcut_steps,
                            crate::phase::PipelineStats::default(),
                        );
                    }
                    if let Some(result) =
                        crate::rules::algebra::try_exact_scalar_multiple_fraction_preorder(
                            &mut simplifier.context,
                            expr,
                            num,
                            den,
                            self.options.shared.semantics.domain_mode,
                            self.options.shared.semantics.value_domain,
                            collect_steps,
                            &mut shortcut_steps,
                            &[],
                        )
                    {
                        return (
                            result,
                            shortcut_steps,
                            crate::phase::PipelineStats::default(),
                        );
                    }
                }
            }
        }

        if !collect_steps && is_solve_mode && !simplifier.has_step_listener() {
            let domain_is_strict = self.options.shared.semantics.domain_mode.is_strict();
            let allow_scalar_root = allow_hidden_solve_root_scalar_multiple_shortcut(&self.options);
            let allow_definability_shortcuts = allow_definability_root_shortcuts(&self.options);
            let (is_pow_root, is_function_root, div_parts) = match simplifier.context.get(expr) {
                Expr::Pow(_, _) => (true, false, None),
                Expr::Function(_, _) => (false, true, None),
                Expr::Div(num, den) => (false, false, Some((*num, *den))),
                _ => (false, false, None),
            };

            if is_function_root && !domain_is_strict {
                if let Some((result, shortcut_steps)) =
                    try_hidden_solve_root_log_power_base_shortcut(
                        &mut simplifier.context,
                        expr,
                        self.options.shared.semantics.domain_mode,
                        self.options.shared.semantics.value_domain,
                    )
                {
                    return (
                        result,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
                }
            }

            if is_pow_root && !domain_is_strict {
                if let Some(result) =
                    try_hidden_solve_root_exp_ln_shortcut(&mut simplifier.context, expr)
                {
                    return (result, Vec::new(), crate::phase::PipelineStats::default());
                }
                if is_symbolic_pow_zero_root(&simplifier.context, expr) {
                    return (
                        simplifier.context.num(1),
                        Vec::new(),
                        crate::phase::PipelineStats::default(),
                    );
                }
            }

            if let Some((num, den)) = div_parts {
                if !domain_is_strict {
                    if is_symbolic_power_over_same_atom_noop_root(&simplifier.context, expr) {
                        return (expr, Vec::new(), crate::phase::PipelineStats::default());
                    }
                    match simplifier.context.get(den) {
                        Expr::Variable(_) | Expr::Constant(_) => {
                            if allow_scalar_root {
                                if let Some(result) =
                                    try_hidden_solve_root_identical_atom_fraction_shortcut(
                                        &mut simplifier.context,
                                        expr,
                                    )
                                {
                                    return (
                                        result,
                                        Vec::new(),
                                        crate::phase::PipelineStats::default(),
                                    );
                                }
                            }
                        }
                        Expr::Pow(_, _) => {
                            if let Some(result) = try_hidden_solve_root_binomial_square_shortcut(
                                &mut simplifier.context,
                                expr,
                            ) {
                                return (
                                    result,
                                    Vec::new(),
                                    crate::phase::PipelineStats::default(),
                                );
                            }
                            if allow_scalar_root {
                                if let Some(result) = try_hidden_solve_root_power_quotient_shortcut(
                                    &mut simplifier.context,
                                    expr,
                                    self.options.shared.semantics.domain_mode,
                                ) {
                                    return (
                                        result,
                                        Vec::new(),
                                        crate::phase::PipelineStats::default(),
                                    );
                                }
                            }
                        }
                        Expr::Add(_, _) => {
                            if allow_definability_shortcuts {
                                if let Some(result) =
                                    crate::rules::algebra::try_exact_sum_diff_of_cubes_preorder(
                                        &mut simplifier.context,
                                        num,
                                        den,
                                    )
                                {
                                    return (
                                        result,
                                        Vec::new(),
                                        crate::phase::PipelineStats::default(),
                                    );
                                }
                            }
                            if allow_scalar_root {
                                if let Some(result) =
                                    try_hidden_solve_root_exact_two_term_scalar_multiple_shortcut(
                                        &mut simplifier.context,
                                        expr,
                                    )
                                {
                                    return (
                                        result,
                                        Vec::new(),
                                        crate::phase::PipelineStats::default(),
                                    );
                                }
                                if let Some(result) =
                                    crate::rules::algebra::try_structural_scalar_multiple_preorder(
                                        &mut simplifier.context,
                                        num,
                                        den,
                                        self.options.shared.semantics.domain_mode,
                                        self.options.shared.semantics.value_domain,
                                    )
                                {
                                    return (
                                        result,
                                        Vec::new(),
                                        crate::phase::PipelineStats::default(),
                                    );
                                }
                            }
                        }
                        Expr::Sub(_, _) => {
                            if allow_definability_shortcuts {
                                if let Some(result) =
                                    crate::rules::algebra::try_exact_sum_diff_of_cubes_preorder(
                                        &mut simplifier.context,
                                        num,
                                        den,
                                    )
                                {
                                    return (
                                        result,
                                        Vec::new(),
                                        crate::phase::PipelineStats::default(),
                                    );
                                }
                                if let Some(result) =
                                    try_hidden_solve_root_difference_of_squares_shortcut(
                                        &mut simplifier.context,
                                        expr,
                                    )
                                {
                                    return (
                                        result,
                                        Vec::new(),
                                        crate::phase::PipelineStats::default(),
                                    );
                                }
                                if let Some(result) =
                                    try_hidden_solve_root_perfect_square_minus_shortcut(
                                        &mut simplifier.context,
                                        expr,
                                    )
                                {
                                    return (
                                        result,
                                        Vec::new(),
                                        crate::phase::PipelineStats::default(),
                                    );
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        // Clear cycle events only when we are about to enter the heavy phase
        // pipeline. Hidden root shortcuts above do not register or consume
        // cycle events, so clearing here avoids fixed overhead on the hot early
        // return paths without changing final stats on full runs.
        cas_solver_core::cycle_event_registry::clear_cycle_events();

        // Clear thread-local PolyStore before evaluation
        clear_thread_local_store();

        // V2.15.8: Set sticky implicit domain from original input to propagate inherited
        // requires across the phase pipeline. Hidden solve root shortcuts above do not need it,
        // because final diagnostics re-derive implicit conditions from input/result.
        simplifier.set_sticky_implicit_domain(expr, self.options.shared.semantics.value_domain);

        // PRE-PASS 1: Eager eval for expand() calls using fast mod-p path
        // This runs BEFORE any simplification to avoid budget exhaustion on huge arguments
        let (current, expand_steps) =
            eager_eval_expand_calls(&mut simplifier.context, expr, collect_steps);
        let mut all_steps = expand_steps;

        // PRE-PASS 2: Eager eval for special functions (poly_gcd_modp)
        let (current, eager_steps) =
            run_poly_gcd_modp_eager_pass(&mut simplifier.context, current, collect_steps);
        all_steps.extend(eager_steps);

        // PRE-PASS 3: Poly lowering - combine poly_result operations before simplification
        // This handles poly_result(0) + poly_result(1) → poly_result(2) internally
        let (current, lower_steps) =
            run_poly_lower_pass(&mut simplifier.context, current, collect_steps);
        all_steps.extend(lower_steps);

        // Check for specialized strategies first
        if let Some(result) =
            crate::try_dirichlet_kernel_identity_pub(&mut simplifier.context, current)
        {
            let zero = simplifier.context.num(0);
            if self.options.collect_steps {
                all_steps.push(Step::new(
                    &format!(
                        "Dirichlet Kernel Identity: 1 + 2Σcos(kx) = sin((n+½)x)/sin(x/2) for n={}",
                        result.n
                    ),
                    "Trig Summation Identity",
                    current,
                    zero,
                    Vec::new(),
                    Some(&simplifier.context),
                ));
            }
            simplifier.clear_sticky_implicit_domain();
            return (zero, all_steps, crate::phase::PipelineStats::default());
        }

        let mut pipeline_stats = crate::phase::PipelineStats::default();

        // Copy values to avoid borrow conflicts with &mut self in run_phase
        let budgets = self.options.budgets;
        let enable_transform = self.options.enable_transform;
        let auto_level = self.options.rationalize.auto_level;

        // V2.15.25: Best-So-Far tracking to prevent returning worse expressions
        // Initialize BSF AFTER Core phase (not from raw input) to preserve Phase 1 canonicalizations
        // This prevents reverting beneficial transformations like tan→sin/cos, arcsec→arccos, etc.
        let budget = BestSoFarBudget::default();

        // Phase 1: Core - Safe simplifications (canonicalizations, basic identities)
        let (next, steps, stats) =
            self.run_phase(simplifier, current, SimplifyPhase::Core, budgets.core_iters);
        let mut current = next;
        all_steps.extend(steps);
        pipeline_stats.core = stats;
        pipeline_stats.total_rewrites += pipeline_stats.core.rewrites_used;
        // Fast path: when Core already collapses to a terminal exact value and the
        // caller is not collecting steps, later phases are pure fixed-cost noise.
        if !collect_steps && is_terminal_after_core(&simplifier.context, current) {
            pipeline_stats.rationalize_level = Some(auto_level);
            pipeline_stats.rationalize_outcome = Some(if auto_level != AutoRationalizeLevel::Off {
                cas_solver_core::rationalize_policy::RationalizeOutcome::NotApplied(
                    cas_solver_core::rationalize_policy::RationalizeReason::NoBinomialFound,
                )
            } else {
                cas_solver_core::rationalize_policy::RationalizeOutcome::NotApplied(
                    cas_solver_core::rationalize_policy::RationalizeReason::PolicyDisabled,
                )
            });
            pipeline_stats.cycle_events =
                cas_solver_core::cycle_event_registry::take_cycle_events();
            simplifier.clear_sticky_implicit_domain();
            return (current, all_steps, pipeline_stats);
        }

        if !collect_steps
            && is_solve_mode
            && !self.options.shared.semantics.domain_mode.is_strict()
            && matches!(simplifier.context.get(expr), Expr::Div(_, _))
            && !self.pattern_marks.has_root_in_denominator()
            && !self.pattern_marks.has_auto_expand_contexts()
            && is_plain_symbolic_power_after_core(&simplifier.context, current)
        {
            pipeline_stats.rationalize_level = Some(auto_level);
            pipeline_stats.rationalize_outcome = Some(if auto_level != AutoRationalizeLevel::Off {
                cas_solver_core::rationalize_policy::RationalizeOutcome::NotApplied(
                    cas_solver_core::rationalize_policy::RationalizeReason::NoBinomialFound,
                )
            } else {
                cas_solver_core::rationalize_policy::RationalizeOutcome::NotApplied(
                    cas_solver_core::rationalize_policy::RationalizeReason::PolicyDisabled,
                )
            });
            pipeline_stats.cycle_events =
                cas_solver_core::cycle_event_registry::take_cycle_events();
            simplifier.clear_sticky_implicit_domain();
            return (current, all_steps, pipeline_stats);
        }

        // Narrow solve fast path: symbolic atom^x / atom with no didactic work.
        // Current solve generic/assume behavior leaves this unchanged, and the
        // later phases are pure overhead on the plain result-only path.
        if !collect_steps
            && is_solve_mode
            && !self.pattern_marks.has_root_in_denominator()
            && !self.pattern_marks.has_auto_expand_contexts()
            && is_symbolic_power_over_same_atom_noop_after_core(&simplifier.context, current)
        {
            pipeline_stats.rationalize_level = Some(auto_level);
            pipeline_stats.rationalize_outcome = Some(if auto_level != AutoRationalizeLevel::Off {
                cas_solver_core::rationalize_policy::RationalizeOutcome::NotApplied(
                    cas_solver_core::rationalize_policy::RationalizeReason::NoBinomialFound,
                )
            } else {
                cas_solver_core::rationalize_policy::RationalizeOutcome::NotApplied(
                    cas_solver_core::rationalize_policy::RationalizeReason::PolicyDisabled,
                )
            });
            pipeline_stats.cycle_events =
                cas_solver_core::cycle_event_registry::take_cycle_events();
            simplifier.clear_sticky_implicit_domain();
            return (current, all_steps, pipeline_stats);
        }

        // Another narrow solve fast path: after Core, symbolic sums like
        // `x + y` do not benefit from later phases on the hidden
        // result-only path.
        if !collect_steps
            && is_solve_mode
            && !self.pattern_marks.has_root_in_denominator()
            && !self.pattern_marks.has_auto_expand_contexts()
            && is_plain_symbolic_binomial_after_core(&simplifier.context, current)
        {
            pipeline_stats.rationalize_level = Some(auto_level);
            pipeline_stats.rationalize_outcome = Some(if auto_level != AutoRationalizeLevel::Off {
                cas_solver_core::rationalize_policy::RationalizeOutcome::NotApplied(
                    cas_solver_core::rationalize_policy::RationalizeReason::NoBinomialFound,
                )
            } else {
                cas_solver_core::rationalize_policy::RationalizeOutcome::NotApplied(
                    cas_solver_core::rationalize_policy::RationalizeReason::PolicyDisabled,
                )
            });
            pipeline_stats.cycle_events =
                cas_solver_core::cycle_event_registry::take_cycle_events();
            simplifier.clear_sticky_implicit_domain();
            return (current, all_steps, pipeline_stats);
        }

        // Same hidden solve fast path for exact cube outputs like
        // `x^2 + y^2 +/- x*y`, which are already in their plain final form
        // after Core and only pay late-phase overhead.
        if !collect_steps
            && is_solve_mode
            && !self.pattern_marks.has_root_in_denominator()
            && !self.pattern_marks.has_auto_expand_contexts()
            && is_plain_symbolic_cube_trinomial_after_core(&simplifier.context, current)
        {
            pipeline_stats.rationalize_level = Some(auto_level);
            pipeline_stats.rationalize_outcome = Some(if auto_level != AutoRationalizeLevel::Off {
                cas_solver_core::rationalize_policy::RationalizeOutcome::NotApplied(
                    cas_solver_core::rationalize_policy::RationalizeReason::NoBinomialFound,
                )
            } else {
                cas_solver_core::rationalize_policy::RationalizeOutcome::NotApplied(
                    cas_solver_core::rationalize_policy::RationalizeReason::PolicyDisabled,
                )
            });
            pipeline_stats.cycle_events =
                cas_solver_core::cycle_event_registry::take_cycle_events();
            simplifier.clear_sticky_implicit_domain();
            return (current, all_steps, pipeline_stats);
        }

        // Initialize BSF lazily from the post-Core baseline.
        // Many solve hot paths stop changing after Core; deferring the score work
        // avoids paying BSF overhead when later phases are pure no-ops.
        let best_baseline_expr = current;
        let best_baseline_steps_len = all_steps.len();
        let mut best: Option<BestSoFar> = None;

        // Phase 2: Transform - Distribution, expansion (if enabled)
        if enable_transform {
            let (next, steps, stats) = self.run_phase(
                simplifier,
                current,
                SimplifyPhase::Transform,
                budgets.transform_iters,
            );
            current = next;
            all_steps.extend(steps);
            pipeline_stats.transform = stats;
            pipeline_stats.total_rewrites += pipeline_stats.transform.rewrites_used;
            if pipeline_stats.transform.changed {
                let best_ref = best.get_or_insert_with(|| {
                    BestSoFar::new(
                        best_baseline_expr,
                        &all_steps[..best_baseline_steps_len],
                        &simplifier.context,
                        budget,
                    )
                });
                best_ref.consider(current, &all_steps, &simplifier.context);
            }
        }

        // Narrow hidden solve fast path: if Transform lands on a plain symbolic
        // binomial, later phases are fixed-cost overhead on the result-only path.
        if !collect_steps
            && is_solve_mode
            && pipeline_stats.transform.changed
            && !self.pattern_marks.has_root_in_denominator()
            && !self.pattern_marks.has_auto_expand_contexts()
            && is_plain_symbolic_binomial_after_core(&simplifier.context, current)
        {
            pipeline_stats.rationalize_level = Some(auto_level);
            pipeline_stats.rationalize_outcome = Some(if auto_level != AutoRationalizeLevel::Off {
                cas_solver_core::rationalize_policy::RationalizeOutcome::NotApplied(
                    cas_solver_core::rationalize_policy::RationalizeReason::NoBinomialFound,
                )
            } else {
                cas_solver_core::rationalize_policy::RationalizeOutcome::NotApplied(
                    cas_solver_core::rationalize_policy::RationalizeReason::PolicyDisabled,
                )
            });
            pipeline_stats.cycle_events =
                cas_solver_core::cycle_event_registry::take_cycle_events();
            simplifier.clear_sticky_implicit_domain();
            return (current, all_steps, pipeline_stats);
        }

        // Phase 3: Rationalize - Auto-rationalization per policy
        // Skip the whole phase when the pre-scan proves there is no root-like
        // form anywhere inside a denominator subtree.
        let should_run_rationalize =
            auto_level != AutoRationalizeLevel::Off && self.pattern_marks.has_root_in_denominator();
        if should_run_rationalize {
            let (next, steps, stats) = self.run_phase(
                simplifier,
                current,
                SimplifyPhase::Rationalize,
                budgets.rationalize_iters,
            );

            // Track rationalization outcome
            pipeline_stats.rationalize_level = Some(auto_level);
            if stats.changed {
                pipeline_stats.rationalize_outcome =
                    Some(cas_solver_core::rationalize_policy::RationalizeOutcome::Applied);
            } else {
                // If enabled but didn't change, it was blocked for some reason
                // We don't have detailed reason here; would need deeper integration
                pipeline_stats.rationalize_outcome = Some(
                    cas_solver_core::rationalize_policy::RationalizeOutcome::NotApplied(
                        cas_solver_core::rationalize_policy::RationalizeReason::NoBinomialFound,
                    ),
                );
            }

            current = next;
            all_steps.extend(steps);
            pipeline_stats.rationalize = stats;
            pipeline_stats.total_rewrites += pipeline_stats.rationalize.rewrites_used;
            if pipeline_stats.rationalize.changed {
                let best_ref = best.get_or_insert_with(|| {
                    BestSoFar::new(
                        best_baseline_expr,
                        &all_steps[..best_baseline_steps_len],
                        &simplifier.context,
                        budget,
                    )
                });
                best_ref.consider(current, &all_steps, &simplifier.context);
            }
        } else {
            pipeline_stats.rationalize_level = Some(auto_level);
            pipeline_stats.rationalize_outcome = Some(if auto_level == AutoRationalizeLevel::Off {
                cas_solver_core::rationalize_policy::RationalizeOutcome::NotApplied(
                    cas_solver_core::rationalize_policy::RationalizeReason::PolicyDisabled,
                )
            } else {
                cas_solver_core::rationalize_policy::RationalizeOutcome::NotApplied(
                    cas_solver_core::rationalize_policy::RationalizeReason::NoBinomialFound,
                )
            });
        }

        // Phase 4: PostCleanup - Final cleanup
        let (next, steps, stats) = self.run_phase(
            simplifier,
            current,
            SimplifyPhase::PostCleanup,
            budgets.post_iters,
        );
        current = next;
        all_steps.extend(steps);
        pipeline_stats.post_cleanup = stats;
        pipeline_stats.total_rewrites += pipeline_stats.post_cleanup.rewrites_used;
        if pipeline_stats.post_cleanup.changed {
            let best_ref = best.get_or_insert_with(|| {
                BestSoFar::new(
                    best_baseline_expr,
                    &all_steps[..best_baseline_steps_len],
                    &simplifier.context,
                    budget,
                )
            });
            best_ref.consider(current, &all_steps, &simplifier.context);
        }

        // Log pipeline summary
        tracing::info!(
            target: "simplify",
            core_iters = pipeline_stats.core.iters_used,
            transform_iters = pipeline_stats.transform.iters_used,
            rationalize_iters = pipeline_stats.rationalize.iters_used,
            post_iters = pipeline_stats.post_cleanup.iters_used,
            total_rewrites = pipeline_stats.total_rewrites,
            "pipeline_complete"
        );

        // Final collection for canonical form - RESPECTS domain mode
        // Use collect_with_semantics to preserve Strict definedness invariant
        let final_parent_ctx = crate::parent_context::ParentContext::root()
            .with_domain_mode(self.options.shared.semantics.domain_mode);
        let final_collected = match crate::collect::collect_with_semantics(
            &mut simplifier.context,
            current,
            &final_parent_ctx,
        ) {
            Some(result) => result.new_expr,
            None => current, // No change (blocked by Strict mode or same result)
        };
        if final_collected != current {
            if crate::ordering::compare_expr(&simplifier.context, final_collected, current)
                != std::cmp::Ordering::Equal
                && collect_steps
            {
                all_steps.push(Step::new(
                    "Final Collection",
                    "Collect",
                    current,
                    final_collected,
                    Vec::new(),
                    Some(&simplifier.context),
                ));
            }
            current = final_collected;
        }

        let late_log_zero = simplifier.context.num(0);
        let late_log_parent_ctx =
            build_root_shortcut_parent_ctx(&self.options, &simplifier.context, current);
        let late_log_rule = crate::rules::arithmetic::ExpandLogAbsMulDivToEnableCancellationRule;
        if let Some(rewrite) = crate::rule::Rule::apply(
            &late_log_rule,
            &mut simplifier.context,
            current,
            &late_log_parent_ctx,
        ) {
            if compare_expr(&simplifier.context, rewrite.new_expr, late_log_zero) == Ordering::Equal
            {
                if collect_steps {
                    let mut step = Step::with_snapshots(
                        &rewrite.description,
                        late_log_rule.name(),
                        current,
                        rewrite.new_expr,
                        smallvec::SmallVec::<[crate::step::PathStep; 8]>::new(),
                        Some(&simplifier.context),
                        current,
                        rewrite.new_expr,
                    );
                    step.importance = late_log_rule.importance();
                    {
                        let meta = step.meta_mut();
                        meta.before_local = rewrite.before_local;
                        meta.after_local = rewrite.after_local;
                        meta.assumption_events = rewrite.assumption_events.clone();
                        meta.required_conditions = rewrite.required_conditions.clone();
                        meta.poly_proof = rewrite.poly_proof.clone();
                        meta.substeps = rewrite.substeps.clone();
                    }
                    all_steps.push(step);
                }
                current = rewrite.new_expr;
            }
        }

        // Filter and optimize steps
        let filtered_steps = if collect_steps {
            cas_solver_core::step_productivity_runtime::filter_non_productive_solver_steps_with_runtime_recompose_mul(
                &mut simplifier.context,
                expr,
                all_steps,
                crate::build::mul2_raw,
            )
        } else {
            all_steps
        };

        let optimized_steps = if collect_steps {
            match cas_solver_core::step_optimization_runtime::optimize_steps_semantic(
                filtered_steps,
                &simplifier.context,
                expr,
                current,
            ) {
                cas_solver_core::step_optimization_runtime::StepOptimizationResult::Steps(steps) => {
                    steps
                }
                cas_solver_core::step_optimization_runtime::StepOptimizationResult::NoSimplificationNeeded => vec![],
            }
        } else {
            filtered_steps
        };

        // Collect assumptions from steps if reporting is enabled
        // Priority: 1) structured assumption_events, 2) legacy domain_assumption string parsing
        if self.options.shared.assumption_reporting != crate::AssumptionReporting::Off {
            pipeline_stats.assumptions = crate::collect_assumption_records_from_iter(
                optimized_steps
                    .iter()
                    .flat_map(|step| step.assumption_events().iter().cloned()),
            );
        }

        // Collect cycle events detected during this pipeline run
        pipeline_stats.cycle_events = cas_solver_core::cycle_event_registry::take_cycle_events();

        // V2.15.8: Clear sticky domain when pipeline completes
        simplifier.clear_sticky_implicit_domain();

        // V2.15.25: Best-So-Far guard - use best if current is worse
        // After all processing, compare current to best seen during phases
        let Some(best) = best else {
            return (current, optimized_steps, pipeline_stats);
        };
        let best_expr = best.best_expr();
        let current_score = crate::best_so_far::score_expr(&simplifier.context, current);
        let best_score = best.best_score();

        // V2.15.35: Skip rollback for explicit expand() calls
        // When user explicitly calls expand(), they want the expanded form even if "worse"
        let has_explicit_expand =
            if let cas_ast::Expr::Function(name, _) = simplifier.context.get(expr) {
                simplifier.context.is_builtin(*name, BuiltinFn::Expand)
            } else {
                false
            };

        // Only rollback if:
        // 1. Best is strictly better AND
        // 2. Current has significantly more nodes (> 12 extra) to avoid reverting expansions
        // 3. NOT an explicit expand() call (user wants expansion)
        // Moderate-to-large increases (1-12 nodes) are allowed to preserve:
        // - Canonicalizations (tan→sin/cos, arcsec→arccos)
        // - Deliberate expansions (AutoExpandBinomials::On)
        let significant_increase = current_score.nodes > best_score.nodes + 12;

        if best_score < current_score && significant_increase && !has_explicit_expand {
            // The best seen during phases is better than final result
            // This can happen when expansion rules don't close with cancellation
            tracing::debug!(
                target: "simplify",
                best_nodes = best_score.nodes,
                current_nodes = current_score.nodes,
                "best_so_far_rollback"
            );
            // Use best expression but keep optimized steps for now
            // TODO: In phase 2, also use best_steps for consistency
            (best_expr, optimized_steps, pipeline_stats)
        } else {
            (current, optimized_steps, pipeline_stats)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    fn render(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn standard_pythagorean_additive_shortcut_handles_negated_numeric_pair() {
        let mut ctx = Context::new();
        let expr = parse("-3*sin(x)^2 - 3*cos(x)^2", &mut ctx)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let (rewritten, _) = try_standard_pythagorean_additive_shortcut(&mut ctx, expr, false)
            .unwrap_or_else(|| panic!("shortcut should match negated numeric pythagorean pair"));
        assert_eq!(render(&ctx, rewritten), "-3");
    }

    #[test]
    fn standard_pythagorean_additive_shortcut_combines_positive_pair_with_constant() {
        let mut ctx = Context::new();
        let expr = parse("sin(x)^2 + cos(x)^2 + 5", &mut ctx)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let (rewritten, _) = try_standard_pythagorean_additive_shortcut(&mut ctx, expr, false)
            .unwrap_or_else(|| {
                panic!("shortcut should match positive numeric pythagorean pair with constant")
            });
        assert_eq!(render(&ctx, rewritten), "6");
    }

    #[test]
    fn standard_pythagorean_additive_shortcut_combines_two_positive_pairs() {
        let mut ctx = Context::new();
        let expr = parse("sin(x)^2 + cos(x)^2 + sin(y)^2 + cos(y)^2", &mut ctx)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let (rewritten, _) = try_standard_pythagorean_additive_shortcut(&mut ctx, expr, false)
            .unwrap_or_else(|| {
                panic!("shortcut should match two positive numeric pythagorean pairs")
            });
        assert_eq!(render(&ctx, rewritten), "2");
    }

    #[test]
    fn mixed_sign_trig_square_difference_root_guard_matches_two_term_difference() {
        let mut ctx = Context::new();
        let expr = parse("-sin(x)^2 + cos(x)^2", &mut ctx)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(is_mixed_sign_trig_square_difference_root(&ctx, expr));
    }

    #[test]
    fn standard_trig_fourth_power_difference_shortcut_finishes_hidden_zero_identity() {
        let mut ctx = Context::new();
        let expr = parse("sin(x)^4 - cos(x)^4 - (sin(x)^2 - cos(x)^2)", &mut ctx)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let (rewritten, _) =
            try_standard_trig_fourth_power_difference_shortcut(&mut ctx, expr, false)
                .unwrap_or_else(|| panic!("shortcut should match hidden quartic identity"));
        assert_eq!(render(&ctx, rewritten), "0");
    }

    #[test]
    fn standard_sin_sum_triple_identity_zero_shortcut_handles_nested_scaled_argument() {
        let mut ctx = Context::new();
        let expr = parse(
            "sin(2*u) + sin(3*(2*u)) - 2*sin(2*(2*u))*cos(2*u)",
            &mut ctx,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let (rewritten, _) =
            try_standard_sin_sum_triple_identity_zero_shortcut(&mut ctx, expr, false)
                .unwrap_or_else(|| panic!("shortcut should match nested scaled triple identity"));
        assert_eq!(render(&ctx, rewritten), "0");
    }

    #[test]
    fn standard_trig_binomial_square_double_angle_shortcut_reduces_to_one() {
        let mut ctx = Context::new();
        let expr = parse("(sin(x) + cos(x))^2 - sin(2*x)", &mut ctx)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let (rewritten, _) = try_standard_trig_binomial_square_double_angle_shortcut(
            &crate::SimplifyOptions::default(),
            &mut ctx,
            expr,
            false,
        )
        .unwrap_or_else(|| panic!("shortcut should reduce trig square plus double-angle pair"));
        assert_eq!(render(&ctx, rewritten), "1");
    }

    #[test]
    fn simplify_pipeline_finishes_pythagorean_passthrough_regression_to_zero() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((1 - sin(x)^2) + m) - ((cos(x)^2) + m)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_finishes_pythagorean_passthrough_from_sin_sq_regression_to_zero() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((sin(x)^2) + m) - ((1-cos(x)^2) + m)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_nested_additive_zero_sum_case21_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + (sin(x)^2 - (1 - cos(2*x))/2)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_nested_additive_shifted_quotient_case24_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + 1)/((sin(x)^2 - (1 - cos(2*x))/2) + 1)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "1");
    }

    #[test]
    fn simplify_pipeline_handles_nested_additive_hyperbolic_cubic_sum_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x)))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_nested_additive_hyperbolic_cubic_difference_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) - (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x)))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_nested_additive_hyperbolic_cubic_shifted_quotient_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + 1)/((2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x))) + 1)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "1");
    }

    #[test]
    fn trig_log_zero_product_direct_shortcut_returns_zero() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(tan(x) + cot(x) - sec(x)*csc(x)) * (2*ln(abs(x*y)) - 2*ln(abs(x)) - 2*ln(abs(y)))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let options = SimplifyOptions::default();
        let (rewritten, _steps) = try_standard_zero_product_with_exact_zero_child_shortcut(
            &options,
            &mut simplifier.context,
            expr,
            false,
        )
        .unwrap_or_else(|| panic!("expected zero-product shortcut to match"));
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_trig_log_zero_product_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(tan(x) + cot(x) - sec(x)*csc(x)) * (2*ln(abs(x*y)) - 2*ln(abs(x)) - 2*ln(abs(y)))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn child_isolated_exact_zero_handles_small_trig_identity_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse("tan(x) + cot(x) - sec(x)*csc(x)", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let options = SimplifyOptions::default();
        assert!(child_isolated_exact_zero(
            &options,
            &mut simplifier.context,
            expr
        ));
    }

    #[test]
    fn child_isolated_exact_zero_handles_small_log_identity_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse("ln(x^3) + ln(y^2) - ln(x^3 * y^2)", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let options = SimplifyOptions::default();
        assert!(child_isolated_exact_zero(
            &options,
            &mut simplifier.context,
            expr
        ));
    }

    #[test]
    fn child_isolated_exact_zero_handles_trig_product_sum_identity_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "2*cos(2*x)*sin(x) - (4*cos(x)^2*sin(x) - 2*sin(x))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let options = SimplifyOptions::default();
        assert!(child_isolated_exact_zero(
            &options,
            &mut simplifier.context,
            expr
        ));
    }

    #[test]
    fn shifted_trig_identity_case336_strips_passthrough_and_proves_both_cores_zero() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((tan(x) + cot(x) - sec(x)*csc(x)) + 1)/((sin(x)^2 - (1 - cos(2*x))/2) + 1)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let (numerator, denominator) = match simplifier.context.get(expr) {
            Expr::Div(numerator, denominator) => (*numerator, *denominator),
            _ => panic!("expected division root"),
        };
        let numerator_core =
            strip_positive_one_passthrough_root(&mut simplifier.context, numerator)
                .unwrap_or_else(|| panic!("expected numerator passthrough core"));
        let denominator_core =
            strip_positive_one_passthrough_root(&mut simplifier.context, denominator)
                .unwrap_or_else(|| panic!("expected denominator passthrough core"));
        let options = SimplifyOptions::default();
        assert!(child_isolated_exact_zero(
            &options,
            &mut simplifier.context,
            numerator_core
        ));
        assert!(child_isolated_exact_zero(
            &options,
            &mut simplifier.context,
            denominator_core
        ));
    }

    #[test]
    fn shifted_trig_identity_case336_direct_div_shortcut_returns_one() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((tan(x) + cot(x) - sec(x)*csc(x)) + 1)/((sin(x)^2 - (1 - cos(2*x))/2) + 1)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let options = SimplifyOptions::default();
        let (rewritten, _steps) = try_standard_shifted_quotient_nested_zero_core_shortcut(
            &options,
            &mut simplifier.context,
            expr,
            false,
        )
        .unwrap_or_else(|| panic!("expected shifted quotient shortcut to match"));
        assert_eq!(render(&simplifier.context, rewritten), "1");
    }

    #[test]
    fn simplify_pipeline_handles_nested_additive_shifted_trig_identity_case336_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((tan(x) + cot(x) - sec(x)*csc(x)) + 1)/((sin(x)^2 - (1 - cos(2*x))/2) + 1)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "1");
    }

    #[test]
    fn simplify_pipeline_handles_reciprocal_trig_plus_product_to_sum_zero_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(tan(x) + cot(x) - sec(x)*csc(x)) + (2*sin(x)*sin(y) - cos(x-y) + cos(x+y))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_reciprocal_trig_minus_product_to_sum_zero_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(tan(x) + cot(x) - sec(x)*csc(x)) - (2*sin(x)*sin(y) - cos(x-y) + cos(x+y))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_rational_factor_times_product_to_sum_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let raw = parse(
            "((1/x + 1/(x+1)) * (2*sin(x)*cos(2*x)))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rewritten_target = parse(
            "((1/x + 1/(x+1)) * (sin(3*x) - sin(x)))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (raw_result, _steps, _stats) = orchestrator.simplify_pipeline(raw, &mut simplifier);
        let (target_result, _steps, _stats) =
            orchestrator.simplify_pipeline(rewritten_target, &mut simplifier);
        assert_eq!(
            render(&simplifier.context, raw_result),
            render(&simplifier.context, target_result)
        );
    }

    #[test]
    fn embedded_trig_product_to_sum_shortcut_matches_rational_factor_regression() {
        let mut ctx = Context::new();
        let expr = parse("((1/x + 1/(x+1)) * (2*sin(x)*cos(2*x)))", &mut ctx)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

        let result = super::try_standard_embedded_trig_product_to_sum_shortcut(
            &crate::phase::SimplifyOptions::default(),
            &mut ctx,
            expr,
            false,
        );

        assert!(
            result.is_some(),
            "embedded product-to-sum shortcut should match"
        );
    }

    #[test]
    fn collapsed_fraction_direct_pair_factor_shortcut_matches_sum_to_product_regression() {
        let mut ctx = Context::new();
        let expr = parse("((1/x + 1/(x+1)) * (sin(x) + sin(3*x)))", &mut ctx)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

        let result = super::try_standard_collapsed_fraction_direct_pair_factor_shortcut(
            &crate::phase::SimplifyOptions::default(),
            &mut ctx,
            expr,
            false,
        );

        assert!(
            result.is_some(),
            "collapsed-fraction direct-pair factor shortcut should match"
        );
    }

    #[test]
    fn collapsed_fraction_direct_pair_factor_shortcut_matches_flattened_product_to_sum_regression()
    {
        let mut ctx = Context::new();
        let expr = parse("((1/x + 1/(x+1)) * (2*sin(x)*cos(2*x)))", &mut ctx)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

        let result = super::try_standard_collapsed_fraction_direct_pair_factor_shortcut(
            &crate::phase::SimplifyOptions::default(),
            &mut ctx,
            expr,
            false,
        );

        assert!(
            result.is_some(),
            "collapsed-fraction direct-pair factor shortcut should match flattened product-to-sum"
        );
    }

    #[test]
    fn simplify_pipeline_handles_collapsed_fraction_times_geometric_sum_partner_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(1/(x*(x+1))) * (u^3 + u^2 + u + 1)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let expected = parse("((u+1)*(u^2+1))/(x*(x+1))", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        let diff_expr = simplifier.context.add(Expr::Sub(rewritten, expected));
        let (diff, _steps, _stats) = orchestrator.simplify_pipeline(diff_expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, diff), "0");
    }

    #[test]
    fn simplify_pipeline_handles_collapsed_fraction_times_sum_of_cubes_partner_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse("(1/(x*(x+1))) * (u^3 + v^3)", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let expected = parse("((u+v)*(u^2-u*v+v^2))/(x*(x+1))", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        let diff_expr = simplifier.context.add(Expr::Sub(rewritten, expected));
        let (diff, _steps, _stats) = orchestrator.simplify_pipeline(diff_expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, diff), "0");
    }

    #[test]
    fn simplify_pipeline_canonicalizes_collapsed_fraction_times_sum_of_cubes_partner_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse("(1/(x*(x+1))) * (u^3 + v^3)", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(
            render(&simplifier.context, rewritten),
            "(u + v) * (u^2 + v^2 - u * v) / (x * (x + 1))"
        );
    }

    #[test]
    fn simplify_pipeline_handles_square_anchor_times_three_linear_shift_partner_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((sqrt(x))^4) * ((u+1)*(u+2)*(u+3))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let expected = parse("(x^2) * ((u+1)*(u+2)*(u+3))", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        let diff_expr = simplifier.context.add(Expr::Sub(rewritten, expected));
        let (diff, _steps, _stats) = orchestrator.simplify_pipeline(diff_expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, diff), "0");
    }

    #[test]
    fn simplify_pipeline_handles_square_anchor_times_expanded_three_linear_shift_partner_regression(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse("(x^2) * (u^3 + 6*u^2 + 11*u + 6)", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let expected = parse("(x^2) * ((u+1)*(u+2)*(u+3))", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        let diff_expr = simplifier.context.add(Expr::Sub(rewritten, expected));
        let (diff, _steps, _stats) = orchestrator.simplify_pipeline(diff_expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, diff), "0");
    }

    #[test]
    fn simplify_pipeline_handles_quartic_square_anchor_times_three_linear_shift_partner_regression()
    {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(x^4 + 4*x^2 + 4) * ((u+1)*(u+2)*(u+3))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let expected = parse(
            "((x^2 + 2)^2) * ((u+1)*(u+2)*(u+3))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        let diff_expr = simplifier.context.add(Expr::Sub(rewritten, expected));
        let (diff, _steps, _stats) = orchestrator.simplify_pipeline(diff_expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, diff), "0");
    }

    #[test]
    fn collapsed_fraction_hyperbolic_half_angle_factor_shortcut_matches_regression() {
        let mut ctx = Context::new();
        let expr = parse("((1/x + 1/(x+1)) * (sinh(x/2)^2))", &mut ctx)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

        let result = super::try_standard_collapsed_fraction_hyperbolic_half_angle_factor_shortcut(
            &crate::phase::SimplifyOptions::default(),
            &mut ctx,
            expr,
            false,
        );

        assert!(
            result.is_some(),
            "collapsed-fraction hyperbolic half-angle shortcut should match"
        );
    }

    #[test]
    fn trig_power_reduction_factor_shortcut_matches_collapsed_fraction_regression() {
        let mut ctx = Context::new();
        let expr = parse("((1/x + 1/(x+1)) * (sin(x)^2*cos(x)^2))", &mut ctx)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

        let result = super::try_standard_trig_power_reduction_factor_shortcut(
            &crate::phase::SimplifyOptions::default(),
            &mut ctx,
            expr,
            false,
        );

        assert!(
            result.is_some(),
            "trig-power reduction factor shortcut should match collapsed-fraction mixed-square products"
        );
    }

    #[test]
    fn simplify_pipeline_handles_collapsed_fraction_times_hyperbolic_half_angle_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let raw = parse("((1/x + 1/(x+1)) * (sinh(x/2)^2))", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (raw_result, _steps, _stats) = orchestrator.simplify_pipeline(raw, &mut simplifier);
        assert_eq!(
            render(&simplifier.context, raw_result),
            "((2 * x + 1) * (cosh(x) - 1))/(x * (x + 1) * 2)"
        );
    }

    #[test]
    fn simplify_pipeline_handles_sum_to_product_root_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let raw = parse("(sin(x) + sin(3*x))", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (raw_result, _steps, _stats) = orchestrator.simplify_pipeline(raw, &mut simplifier);
        assert_eq!(
            render(&simplifier.context, raw_result),
            "2 * sin(2 * x) * cos(x)"
        );
    }

    #[test]
    fn simplify_pipeline_handles_collapsed_fraction_times_trig_power_reduction_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let raw = parse(
            "((1/x + 1/(x+1)) * (sin(x)^2*cos(x)^2))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let expected = parse(
            "(((2*x+1)/(x*(x+1))) * ((sin(2*x)^2)/4))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (raw_result, _steps, _stats) = orchestrator.simplify_pipeline(raw, &mut simplifier);
        let (expected_result, _steps, _stats) =
            orchestrator.simplify_pipeline(expected, &mut simplifier);
        assert_eq!(
            render(&simplifier.context, raw_result),
            render(&simplifier.context, expected_result)
        );
    }

    #[test]
    fn simplify_pipeline_handles_reciprocal_trig_product_with_product_to_sum_zero_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(tan(x) + cot(x) - sec(x)*csc(x)) * (2*sin(x)*sin(y) - cos(x-y) + cos(x+y))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_reciprocal_trig_shifted_quotient_with_product_to_sum_zero_regression(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((tan(x) + cot(x) - sec(x)*csc(x)) + 1)/((2*sin(x)*sin(y) - cos(x-y) + cos(x+y)) + 1)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "1");
    }

    #[test]
    fn detects_direct_trig_product_to_sum_sin_sin_zero_identity_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lhs = parse("2*sin(x)*sin(y)", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs = parse("cos(x-y) - cos(x+y)", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(matches_direct_trig_product_to_sum_sin_sin_pair_root(
            &mut simplifier.context,
            lhs,
            rhs
        ));
    }

    #[test]
    fn detects_direct_trig_product_to_sum_sin_sin_raw_zero_identity_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "2*sin(x)*sin(y) - cos(x-y) + cos(x+y)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(matches_direct_small_zero_identity_root(
            &mut simplifier.context,
            expr
        ));
    }

    #[test]
    fn detects_direct_trig_product_to_sum_sin_sin_raw_zero_identity_reordered_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "cos(x+y) + 2*sin(x)*sin(y) - cos(x-y)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(matches_direct_small_zero_identity_root(
            &mut simplifier.context,
            expr
        ));
    }

    #[test]
    fn simplify_pipeline_handles_trig_product_to_sum_sin_sin_shifted_quotient_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((2*sin(x)*sin(y)) + 1)/((cos(x-y) - cos(x+y)) + 1)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "1");
    }

    #[test]
    fn simplify_pipeline_handles_trig_product_to_sum_sin_sin_raw_zero_identity_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "2*sin(x)*sin(y) - cos(x-y) + cos(x+y)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_tan_cot_product_plus_trig_product_to_sum_sin_sin_zero_regression()
    {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(tan(x)*cot(x) - 1) + (2*sin(x)*sin(y) - cos(x-y) + cos(x+y))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_trig_product_to_sum_sin_sin_plus_odd_half_power_zero_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(2*sin(x)*sin(y) - cos(x-y) + cos(x+y)) + (sqrt(x^5) - x^2*sqrt(x))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn detects_direct_odd_half_power_zero_scope_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse("sqrt(x^5) - x^2*sqrt(x)", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(matches_direct_small_zero_identity_root(
            &mut simplifier.context,
            expr
        ));
    }

    #[test]
    fn detects_direct_trig_product_to_sum_and_odd_half_partition_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(2*sin(x)*sin(y) - cos(x-y) + cos(x+y)) + (sqrt(x^5) - x^2*sqrt(x))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let terms = AddView::from_expr(&simplifier.context, expr).terms;
        let odd_expr = build_signed_sum_expr_root(&mut simplifier.context, &[terms[1], terms[2]]);
        let trig_expr =
            build_signed_sum_expr_root(&mut simplifier.context, &[terms[0], terms[3], terms[4]]);
        assert!(
            matches_direct_odd_half_power_zero_scope_root(&mut simplifier.context, odd_expr),
            "odd_expr={}",
            render(&simplifier.context, odd_expr)
        );
        assert!(
            matches_direct_trig_product_to_sum_zero_identity_root(
                &mut simplifier.context,
                trig_expr
            ),
            "trig_expr={}",
            render(&simplifier.context, trig_expr)
        );
        assert!(
            matches_direct_trig_product_to_sum_and_odd_half_partition_root(
                &mut simplifier.context,
                expr
            )
        );
    }

    #[test]
    fn detects_direct_geometric_difference_zero_identity_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "x^6 - 1 - (x-1)*(x^5+x^4+x^3+x^2+x+1)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(
            matches_direct_geometric_difference_zero_identity_root(&mut simplifier.context, expr),
            "expr={}",
            render(&simplifier.context, expr)
        );
    }

    #[test]
    fn simplify_pipeline_handles_trig_product_to_sum_sin_sin_plus_small_polynomial_zero_regression()
    {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(2*sin(x)*sin(y) - cos(x-y) + cos(x+y)) + (x^6 - 1 - (x-1)*(x^5+x^4+x^3+x^2+x+1))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_trig_product_to_sum_sin_sin_minus_small_polynomial_zero_regression(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(2*sin(x)*sin(y) - cos(x-y) + cos(x+y)) - (x^6 - 1 - (x-1)*(x^5+x^4+x^3+x^2+x+1))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn detects_trig_product_to_sum_sin_sin_shifted_quotient_pair_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((2*sin(x)*sin(y)) + 1)/((cos(x-y) - cos(x+y)) + 1)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let (numerator, denominator) = match simplifier.context.get(expr) {
            Expr::Div(numerator, denominator) => (*numerator, *denominator),
            _ => panic!("expected division root"),
        };
        let numerator_core =
            strip_positive_one_passthrough_root(&mut simplifier.context, numerator)
                .unwrap_or_else(|| panic!("expected numerator passthrough core"));
        let denominator_core =
            strip_positive_one_passthrough_root(&mut simplifier.context, denominator)
                .unwrap_or_else(|| panic!("expected denominator passthrough core"));

        let numerator_rewrite =
            try_rewrite_product_to_sum_expr(&mut simplifier.context, numerator_core)
                .map(|rewrite| render(&simplifier.context, rewrite.rewritten))
                .unwrap_or_else(|| "<none>".to_string());
        assert!(
            matches_direct_trig_product_to_sum_sin_sin_pair_root(
                &mut simplifier.context,
                numerator_core,
                denominator_core
            ),
            "numerator_core={}, denominator_core={}, numerator_rewrite={}",
            render(&simplifier.context, numerator_core),
            render(&simplifier.context, denominator_core),
            numerator_rewrite,
        );
    }

    #[test]
    fn detects_direct_trig_product_to_sum_cos_cos_pair_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lhs = parse("2*cos(x)*cos(y)", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs = parse("cos(x+y) + cos(x-y)", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(matches_direct_trig_product_to_sum_cos_cos_pair_root(
            &mut simplifier.context,
            lhs,
            rhs
        ));
    }

    #[test]
    fn detects_direct_trig_product_to_sum_cos_cos_zero_identity_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "2*cos(x)*cos(y) - cos(x+y) - cos(x-y)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(matches_direct_small_zero_identity_root(
            &mut simplifier.context,
            expr
        ));
    }

    #[test]
    fn simplify_pipeline_handles_trig_product_to_sum_cos_cos_shifted_quotient_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((2*cos(x)*cos(y) - cos(x+y) - cos(x-y)) + 1)/((sinh(x+y) - (sinh(x)*cosh(y) + cosh(x)*sinh(y))) + 1)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "1");
    }

    #[test]
    fn detects_direct_trig_product_to_sum_sin_cos_pair_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lhs = parse("2*sin(x)*cos(y)", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs = parse("sin(x+y) + sin(x-y)", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(matches_direct_trig_product_to_sum_sin_cos_pair_root(
            &mut simplifier.context,
            lhs,
            rhs
        ));
    }

    #[test]
    fn detects_direct_trig_product_to_sum_sin_cos_odd_difference_pair_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lhs = parse("2*sin(x)*cos(2*x)", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs = parse("sin(3*x) - sin(x)", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(matches_direct_trig_product_to_sum_sin_cos_pair_root(
            &mut simplifier.context,
            lhs,
            rhs
        ));
    }

    #[test]
    fn detects_direct_trig_product_to_sum_sin_cos_zero_identity_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "2*sin(x)*cos(y) - sin(x+y) - sin(x-y)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(matches_direct_small_zero_identity_root(
            &mut simplifier.context,
            expr
        ));
    }

    #[test]
    fn simplify_pipeline_handles_trig_product_to_sum_sin_cos_shifted_quotient_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((2*sin(x)*cos(y) - sin(x+y) - sin(x-y)) + 1)/((tan(x) + cot(x) - sec(x)*csc(x)) + 1)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "1");
    }

    #[test]
    fn detects_direct_nested_fraction_simplified_pair_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lhs = parse("1 + 1/(1 + 1/(1 + 1/x))", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs = parse("(3*x + 2)/(2*x + 1)", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(matches_direct_nested_fraction_simplified_pair_root(
            &mut simplifier.context,
            lhs,
            rhs
        ));
    }

    #[test]
    fn detects_direct_nested_fraction_simplified_zero_identity_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(matches_direct_small_zero_identity_root(
            &mut simplifier.context,
            expr
        ));
    }

    #[test]
    fn simplify_pipeline_handles_nested_fraction_shifted_quotient_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((3*sin(x) + 4*cos(x) - 5*sin(x + arctan(4/3))) + 1)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "1");
    }

    #[test]
    fn detects_direct_hyperbolic_sinh_sum_to_product_pair_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lhs = parse("sinh(x) + sinh(y)", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs = parse("2*sinh((x+y)/2)*cosh((x-y)/2)", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(matches_direct_hyperbolic_sinh_sum_to_product_pair_root(
            &mut simplifier.context,
            lhs,
            rhs
        ));
    }

    #[test]
    fn simplify_pipeline_handles_hyperbolic_sinh_sum_to_product_passthrough_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((sinh(x)+sinh(y)) + m) - ((2*sinh((x+y)/2)*cosh((x-y)/2)) + m)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_hyperbolic_sinh_sum_to_product_shifted_quotient_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((sinh(x)+sinh(y)) + 1)/((2*sinh((x+y)/2)*cosh((x-y)/2)) + 1)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "1");
    }

    #[test]
    fn detects_direct_hyperbolic_cosh_sum_to_product_pair_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lhs = parse("cosh(x) + cosh(y)", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs = parse("2*cosh((x+y)/2)*cosh((x-y)/2)", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(matches_direct_hyperbolic_cosh_sum_to_product_pair_root(
            &mut simplifier.context,
            lhs,
            rhs
        ));
    }

    #[test]
    fn simplify_pipeline_handles_hyperbolic_cosh_sum_to_product_passthrough_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((cosh(x)+cosh(y)) + m) - ((2*cosh((x+y)/2)*cosh((x-y)/2)) + m)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_hyperbolic_cosh_sum_to_product_shifted_quotient_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((cosh(x)+cosh(y)) + 1)/((2*cosh((x+y)/2)*cosh((x-y)/2)) + 1)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "1");
    }

    #[test]
    fn detects_direct_hyperbolic_cosh_difference_to_product_pair_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lhs = parse("cosh(x) - cosh(y)", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs = parse("2*sinh((x+y)/2)*sinh((x-y)/2)", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(
            matches_direct_hyperbolic_cosh_difference_to_product_pair_root(
                &mut simplifier.context,
                lhs,
                rhs
            )
        );
    }

    #[test]
    fn simplify_pipeline_handles_hyperbolic_cosh_difference_to_product_passthrough_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((cosh(x)-cosh(y)) + m) - ((2*sinh((x+y)/2)*sinh((x-y)/2)) + m)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn detects_direct_recursive_hyperbolic_sinh_sum_pair_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lhs = parse("sinh(6*x)", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs = parse(
            "sinh(5*x)*cosh(x)+cosh(5*x)*sinh(x)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(matches_direct_recursive_hyperbolic_sinh_sum_pair_root(
            &mut simplifier.context,
            lhs,
            rhs
        ));
    }

    #[test]
    fn simplify_pipeline_handles_recursive_hyperbolic_sinh_sum_passthrough_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((sinh(6*x)) + m) - ((sinh(5*x)*cosh(x)+cosh(5*x)*sinh(x)) + m)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_recursive_hyperbolic_sinh_sum_shifted_quotient_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((sinh(6*x)) + 1)/((sinh(5*x)*cosh(x)+cosh(5*x)*sinh(x)) + 1)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "1");
    }

    #[test]
    fn detects_direct_recursive_hyperbolic_cosh_sum_pair_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lhs = parse("cosh(6*x)", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs = parse(
            "cosh(5*x)*cosh(x)+sinh(5*x)*sinh(x)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(matches_direct_recursive_hyperbolic_cosh_sum_pair_root(
            &mut simplifier.context,
            lhs,
            rhs
        ));
    }

    #[test]
    fn simplify_pipeline_handles_recursive_hyperbolic_cosh_sum_passthrough_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((cosh(6*x)) + m) - ((cosh(5*x)*cosh(x)+sinh(5*x)*sinh(x)) + m)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_recursive_hyperbolic_cosh_sum_shifted_quotient_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((cosh(6*x)) + 1)/((cosh(5*x)*cosh(x)+sinh(5*x)*sinh(x)) + 1)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "1");
    }

    #[test]
    fn detects_direct_trig_mixed_double_angle_zero_identity_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "2*cos(2*x)*sin(x) - (4*cos(x)^2*sin(x) - 2*sin(x))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(matches_direct_trig_mixed_double_angle_zero_identity_root(
            &mut simplifier.context,
            expr
        ));
    }

    #[test]
    fn detects_negative_double_cos_square_diff_shifted_quotient_pair_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((sin(x)^2 - cos(x)^2) + 1)/((-cos(2*x)) + 1)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let (numerator, denominator) = match simplifier.context.get(expr) {
            Expr::Div(numerator, denominator) => (*numerator, *denominator),
            _ => panic!("expected division root"),
        };
        let numerator_core =
            strip_positive_one_passthrough_root(&mut simplifier.context, numerator)
                .unwrap_or_else(|| panic!("expected numerator passthrough core"));
        let denominator_core =
            strip_positive_one_passthrough_root(&mut simplifier.context, denominator)
                .unwrap_or_else(|| panic!("expected denominator passthrough core"));

        assert!(matches_direct_negative_double_cos_square_diff_pair_root(
            &mut simplifier.context,
            numerator_core,
            denominator_core
        ));
    }

    #[test]
    fn simplify_pipeline_handles_negative_double_cos_square_diff_shifted_quotient_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((sin(x)^2 - cos(x)^2) + 1)/((-cos(2*x)) + 1)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "1");
    }

    #[test]
    fn simplify_pipeline_handles_positive_double_cos_square_diff_direct_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse("cos(x)^2 - sin(x)^2 - cos(2*x)", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_positive_double_cos_square_diff_nested_arg_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "cos(sin(u))^2 - sin(sin(u))^2 - cos(2*sin(u))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn detects_direct_pythagorean_extended_pair_nested_arg_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lhs = parse("sin(sin(u))^4 + cos(sin(u))^4", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs = parse("1 - 2*sin(sin(u))^2*cos(sin(u))^2", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

        assert!(matches_direct_pythagorean_extended_pair_root(
            &mut simplifier.context,
            lhs,
            rhs
        ));
    }

    #[test]
    fn simplify_pipeline_handles_pythagorean_extended_nested_arg_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(sin(sin(u))^4 + cos(sin(u))^4) - (1 - 2*sin(sin(u))^2*cos(sin(u))^2)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_positive_double_cos_square_diff_shifted_quotient_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((cos(x)^2 - sin(x)^2) + 1)/((cos(2*x)) + 1)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "1");
    }

    #[test]
    fn simplify_pipeline_handles_negative_double_cos_square_diff_passthrough_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((-cos(2*x)) + m) - ((sin(x)^2 - cos(x)^2) + m)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_negative_double_cos_square_diff_passthrough_forward_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((sin(x)^2 - cos(x)^2) + m) - ((-cos(2*x)) + m)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_negative_double_cos_square_diff_scaled_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "k*(-cos(2*x)) - k*(sin(x)^2 - cos(x)^2)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_negative_double_cos_square_diff_common_denominator_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((sin(x)^2 - cos(x)^2)/q) - ((-cos(2*x))/q)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn detects_angle_sum_diff_shifted_quotient_pair_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((cos(6*x)) + 1)/((cos(5*x)*cos(x)-sin(5*x)*sin(x)) + 1)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let (numerator, denominator) = match simplifier.context.get(expr) {
            Expr::Div(numerator, denominator) => (*numerator, *denominator),
            _ => panic!("expected division root"),
        };
        let numerator_core =
            strip_positive_one_passthrough_root(&mut simplifier.context, numerator)
                .unwrap_or_else(|| panic!("expected numerator passthrough core"));
        let denominator_core =
            strip_positive_one_passthrough_root(&mut simplifier.context, denominator)
                .unwrap_or_else(|| panic!("expected denominator passthrough core"));

        assert!(
            matches_direct_angle_sum_diff_pair_root(
                &mut simplifier.context,
                numerator_core,
                denominator_core
            ),
            "numerator_core={}, denominator_core={}",
            render(&simplifier.context, numerator_core),
            render(&simplifier.context, denominator_core),
        );
    }

    #[test]
    fn simplify_pipeline_handles_angle_sum_diff_shifted_quotient_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((cos(6*x)) + 1)/((cos(5*x)*cos(x)-sin(5*x)*sin(x)) + 1)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "1");
    }

    #[test]
    fn simplify_pipeline_handles_angle_sum_diff_passthrough_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((cos(6*x)) + m) - ((cos(5*x)*cos(x)-sin(5*x)*sin(x)) + m)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_angle_sum_diff_passthrough_reverse_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((cos(5*x)*cos(x)-sin(5*x)*sin(x)) + m) - ((cos(6*x)) + m)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_trig_cubic_cosine_passthrough_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((2*cos(2*x)*cos(x)) + m) - ((4*cos(x)^3-2*cos(x)) + m)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn detects_recursive_sine_shifted_quotient_pair_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((sin(6*x)) + 1)/((sin(5*x)*cos(x)+cos(5*x)*sin(x)) + 1)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let (numerator, denominator) = match simplifier.context.get(expr) {
            Expr::Div(numerator, denominator) => (*numerator, *denominator),
            _ => panic!("expected division root"),
        };
        let numerator_core =
            strip_positive_one_passthrough_root(&mut simplifier.context, numerator)
                .unwrap_or_else(|| panic!("expected numerator passthrough core"));
        let denominator_core =
            strip_positive_one_passthrough_root(&mut simplifier.context, denominator)
                .unwrap_or_else(|| panic!("expected denominator passthrough core"));

        assert!(
            matches_direct_angle_sum_diff_pair_root(
                &mut simplifier.context,
                numerator_core,
                denominator_core
            ),
            "numerator_core={}, denominator_core={}",
            render(&simplifier.context, numerator_core),
            render(&simplifier.context, denominator_core),
        );
    }

    #[test]
    fn simplify_pipeline_handles_recursive_sine_shifted_quotient_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((sin(6*x)) + 1)/((sin(5*x)*cos(x)+cos(5*x)*sin(x)) + 1)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "1");
    }

    #[test]
    fn simplify_pipeline_handles_recursive_sine_shifted_quotient_reverse_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((sin(5*x)*cos(x)+cos(5*x)*sin(x)) + 1)/((sin(6*x)) + 1)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "1");
    }

    #[test]
    fn simplify_pipeline_handles_hyperbolic_cosh_cubic_passthrough_shifted_quotient_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((2*sinh(2*x)*sinh(x)+a) + 1)/((4*cosh(x)^3-4*cosh(x)+a) + 1)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "1");
    }

    #[test]
    fn simplify_pipeline_handles_shifted_quotient_with_reversed_reciprocal_trig_zero_pair_regression(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((sin(x)^2 - (1 - cos(2*x))/2) + 1)/((tan(x) + cot(x) - sec(x)*csc(x)) + 1)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "1");
    }

    #[test]
    fn detects_half_angle_binomial_square_shifted_quotient_pair_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((sin(x)^2 - (1 - cos(2*x))/2) + 1)/(((sin(x) + cos(x))^2 - (1 + sin(2*x))) + 1)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let (numerator, denominator) = match simplifier.context.get(expr) {
            Expr::Div(numerator, denominator) => (*numerator, *denominator),
            _ => panic!("expected division root"),
        };
        let numerator_core =
            strip_positive_one_passthrough_root(&mut simplifier.context, numerator)
                .unwrap_or_else(|| panic!("expected numerator passthrough core"));
        let denominator_core =
            strip_positive_one_passthrough_root(&mut simplifier.context, denominator)
                .unwrap_or_else(|| panic!("expected denominator passthrough core"));

        assert!(matches_direct_half_angle_square_zero_identity_root(
            &mut simplifier.context,
            numerator_core,
        ));
        assert!(matches_direct_trig_binomial_square_zero_identity_root(
            &mut simplifier.context,
            denominator_core,
        ));
        assert!(matches_direct_half_angle_binomial_square_pair_root(
            &mut simplifier.context,
            numerator_core,
            denominator_core,
        ));
    }

    #[test]
    fn detects_direct_half_angle_square_pair_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lhs = parse("cos(x)^2", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs = parse("((1 + cos(2*x))/2)", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

        assert!(matches_direct_half_angle_square_pair_root(
            &mut simplifier.context,
            lhs,
            rhs,
        ));
    }

    #[test]
    fn detects_direct_trig_binomial_square_pair_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lhs = parse("(sin(x) + cos(x))^2", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs = parse("1 + sin(2*x)", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

        assert!(matches_direct_trig_binomial_square_pair_root(
            &mut simplifier.context,
            lhs,
            rhs,
        ));
    }

    #[test]
    fn detects_direct_pythagorean_identity_pair_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lhs = parse("sin(x)^2 + cos(x)^2", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs =
            parse("1", &mut simplifier.context).unwrap_or_else(|e| panic!("parse failed: {e:?}"));

        assert!(matches_direct_pythagorean_identity_pair_root(
            &mut simplifier.context,
            lhs,
            rhs,
        ));
    }

    #[test]
    fn simplify_pipeline_handles_pythagorean_identity_passthrough_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((sin(x)^2 + cos(x)^2) + m) - ((1) + m)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn direct_pair_shortcut_handles_pythagorean_identity_passthrough_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((sin(x)^2 + cos(x)^2) + m) - ((1) + m)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let options = SimplifyOptions::default();
        let (rewritten, _steps) = try_standard_shared_passthrough_direct_pair_shortcut(
            &options,
            &mut simplifier.context,
            expr,
            false,
        )
        .unwrap_or_else(|| panic!("expected direct pair shortcut to match"));
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_pythagorean_identity_shifted_quotient_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((sin(x)^2 + cos(x)^2) + 1)/((1) + 1)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "1");
    }

    #[test]
    fn simplify_pipeline_handles_embedded_positive_pythagorean_pair_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse("sin(x)^2 + cos(x)^2 - sin(y)^2", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "cos(y)^2");
    }

    #[test]
    fn simplify_pipeline_handles_mixed_positive_negative_pythagorean_pairs_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "sin(x)^2 + cos(x)^2 - sin(y)^2 - cos(y)^2",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_trig_binomial_square_passthrough_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(((sin(x)+cos(x))^2) + m) - ((1+sin(2*x)) + m)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_trig_binomial_square_passthrough_without_steps_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(((sin(x)+cos(x))^2) + m) - ((1+sin(2*x)) + m)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        orchestrator.options.collect_steps = false;
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn detects_trig_binomial_square_passthrough_direct_pair_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(((sin(x)+cos(x))^2) + m) - ((1+sin(2*x)) + m)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let expected_lhs = parse("(sin(x)+cos(x))^2", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let expected_rhs = parse("1+sin(2*x)", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let (lhs_core, rhs_core) =
            extract_shared_additive_passthrough_sub_cores_root(&mut simplifier.context, expr)
                .unwrap_or_else(|| panic!("expected passthrough cores"));

        assert_eq!(
            compare_expr(&simplifier.context, lhs_core, expected_lhs),
            Ordering::Equal
        );
        assert_eq!(
            compare_expr(&simplifier.context, rhs_core, expected_rhs),
            Ordering::Equal
        );
        assert_eq!(
            passthrough_direct_pair_rule_name_root(&mut simplifier.context, lhs_core, rhs_core),
            Some("Collapse Exact Zero Additive Subexpression"),
        );
    }

    #[test]
    fn direct_pair_shortcut_handles_trig_binomial_square_passthrough_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(((sin(x)+cos(x))^2) + m) - ((1+sin(2*x)) + m)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let options = SimplifyOptions::default();
        let (rewritten, _steps) = try_standard_shared_passthrough_direct_pair_shortcut(
            &options,
            &mut simplifier.context,
            expr,
            false,
        )
        .unwrap_or_else(|| panic!("expected direct pair shortcut to match"));
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn direct_pair_shortcut_handles_trig_binomial_square_passthrough_with_steps_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(((sin(x)+cos(x))^2) + m) - ((1+sin(2*x)) + m)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let options = SimplifyOptions::default();
        let (rewritten, steps) = try_standard_shared_passthrough_direct_pair_shortcut(
            &options,
            &mut simplifier.context,
            expr,
            true,
        )
        .unwrap_or_else(|| panic!("expected direct pair shortcut to match"));
        assert_eq!(render(&simplifier.context, rewritten), "0");
        assert!(!steps.is_empty());
    }

    #[test]
    fn simplify_pipeline_handles_trig_binomial_square_shifted_quotient_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((((sin(x)+cos(x))^2) + 1))/(((1+sin(2*x)) + 1))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "1");
    }

    #[test]
    fn exact_one_shortcut_handles_half_angle_binomial_square_shifted_quotient_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((sin(x)^2 - (1 - cos(2*x))/2) + 1)/(((sin(x) + cos(x))^2 - (1 + sin(2*x))) + 1)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let options = SimplifyOptions::default();
        let (rewritten, _steps) = try_standard_shifted_quotient_exact_one_shortcut(
            &options,
            &mut simplifier.context,
            expr,
            false,
        )
        .unwrap_or_else(|| panic!("expected exact-one shortcut to match"));
        assert_eq!(render(&simplifier.context, rewritten), "1");
    }

    #[test]
    fn simplify_pipeline_contracts_direct_half_angle_cos_square_root_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse("((1+cos(2*x))/2)", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "cos(x)^2");
    }

    #[test]
    fn simplify_pipeline_handles_scaled_direct_half_angle_cos_square_root_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse("8*((1+cos(2*x))/2)", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "8 * cos(x)^2");
    }

    #[test]
    fn simplify_pipeline_handles_fraction_times_direct_half_angle_cos_square_root_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((2*x+1)/(x*(x+1))) * ((1+cos(2*x))/2)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(
            render(&simplifier.context, rewritten),
            "(cos(x)^2 + 2 * x * cos(x)^2) / (x * (x + 1))"
        );
    }

    #[test]
    fn simplify_pipeline_contracts_direct_cos_fourth_power_reduction_root_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse("((3+4*cos(2*x)+cos(4*x))/8)", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "cos(x)^4");
    }

    #[test]
    fn simplify_pipeline_handles_scaled_direct_cos_fourth_power_reduction_root_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse("8*((3+4*cos(2*x)+cos(4*x))/8)", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "8 * cos(x)^4");
    }

    #[test]
    fn simplify_pipeline_handles_scaled_sine_fourth_power_reduction_zero_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "8*sin(x)^4 - (3 - 4*cos(2*x) + cos(4*x))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_fraction_times_direct_cos_fourth_power_reduction_root_regression()
    {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((2*x+1)/(x*(x+1))) * ((3+4*cos(2*x)+cos(4*x))/8)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(
            render(&simplifier.context, rewritten),
            "(cos(x)^4 * (2 * x + 1))/(x * (x + 1))"
        );
    }

    #[test]
    fn simplify_pipeline_contracts_direct_sin_cos_square_product_reduction_root_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse("((1-cos(4*x))/8)", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "1/4 * sin(2 * x)^2");
    }

    #[test]
    fn simplify_pipeline_handles_scaled_positive_double_cos_square_diff_factor_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse("8*(2*cos(x)^2 - 1)", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "8 * cos(2 * x)");
    }

    #[test]
    fn simplify_pipeline_handles_fraction_times_positive_double_cos_square_diff_factor_regression()
    {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((2*x+1)/(x*(x+1))) * (2*cos(x)^2 - 1)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(
            render(&simplifier.context, rewritten),
            "(cos(2 * x) * (2 * x + 1))/(x * (x + 1))"
        );
    }

    #[test]
    fn tangent_addition_factor_shortcut_matches_multiple_angle_regression() {
        let mut ctx = Context::new();
        let expr = parse("(sin(5*x)) * (tan(x) + tan(y))", &mut ctx)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

        let result = super::try_standard_tangent_addition_factor_shortcut(
            &crate::phase::SimplifyOptions::default(),
            &mut ctx,
            expr,
            false,
        );

        assert!(
            result.is_some(),
            "tangent-addition factor shortcut should match multiple-angle products"
        );
    }

    #[test]
    fn special_angle_exact_value_factor_shortcut_matches_tangent_addition_regression() {
        let mut ctx = Context::new();
        let expr = parse("(cot(5*pi/12)) * (tan(x) + tan(y))", &mut ctx)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

        let result = super::try_standard_special_angle_exact_value_factor_shortcut(
            &crate::phase::SimplifyOptions::default(),
            &mut ctx,
            expr,
            false,
        );

        assert!(
            result.is_some(),
            "special-angle exact-value factor shortcut should match cot multiple-angle products"
        );
    }

    #[test]
    fn special_angle_exact_value_factor_shortcut_matches_tan_angle_sum_regression() {
        let mut ctx = Context::new();
        let expr = parse("(cot(5*pi/12)) * (tan(x+y))", &mut ctx)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

        let result = super::try_standard_special_angle_exact_value_factor_shortcut(
            &crate::phase::SimplifyOptions::default(),
            &mut ctx,
            expr,
            false,
        );

        assert!(
            result.is_some(),
            "special-angle exact-value factor shortcut should match tan-angle-sum products"
        );
    }

    #[test]
    fn special_angle_exact_value_factor_shortcut_matches_sum_of_squares_product_regression() {
        let mut ctx = Context::new();
        let expr = parse("(cot(5*pi/12)) * ((w^2 + p^2)*(u^2 + v^2))", &mut ctx)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

        let result = super::try_standard_special_angle_exact_value_factor_shortcut(
            &crate::phase::SimplifyOptions::default(),
            &mut ctx,
            expr,
            false,
        );

        assert!(
            result.is_some(),
            "special-angle exact-value factor shortcut should match sum-of-squares products"
        );
    }

    #[test]
    fn sum_of_squares_product_subset_factor_shortcut_matches_special_angle_regression() {
        let mut ctx = Context::new();
        let expr = parse("(cot(5*pi/12)) * ((w^2 + p^2)*(u^2 + v^2))", &mut ctx)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

        let result = super::try_standard_sum_of_squares_product_subset_factor_shortcut(
            &crate::phase::SimplifyOptions::default(),
            &mut ctx,
            expr,
            false,
        );

        assert!(
            result.is_some(),
            "sum-of-squares subset factor shortcut should match special-angle products"
        );
    }

    #[test]
    fn special_angle_exact_value_factor_shortcut_matches_product_to_sum_regression() {
        let mut ctx = Context::new();
        let expr = parse("(sin(5*pi/6)) * (2*sin(x)*cos(2*x))", &mut ctx)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

        let result = super::try_standard_special_angle_exact_value_factor_shortcut(
            &crate::phase::SimplifyOptions::default(),
            &mut ctx,
            expr,
            false,
        );

        assert!(
            result.is_some(),
            "special-angle exact-value factor shortcut should match product-to-sum products"
        );
    }

    #[test]
    fn trig_product_to_sum_subset_factor_shortcut_matches_special_angle_regression() {
        let mut ctx = Context::new();
        let expr = parse("(sin(5*pi/6)) * (2*sin(x)*cos(2*x))", &mut ctx)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

        let result = super::try_standard_trig_product_to_sum_subset_factor_shortcut(
            &crate::phase::SimplifyOptions::default(),
            &mut ctx,
            expr,
            false,
        );

        assert!(
            result.is_some(),
            "product-to-sum subset factor shortcut should match special-angle products"
        );
    }

    #[test]
    fn trig_product_to_sum_subset_factor_shortcut_matches_external_partner_regression() {
        let mut ctx = Context::new();
        let expr = parse("(2*sin(x)*cos(2*x)) * (cos(pi - x))", &mut ctx)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

        let result = super::try_standard_trig_product_to_sum_subset_factor_shortcut(
            &crate::phase::SimplifyOptions::default(),
            &mut ctx,
            expr,
            false,
        );

        assert!(
            result.is_some(),
            "product-to-sum subset factor shortcut should match products with an external partner"
        );
    }

    #[test]
    fn trig_product_to_sum_subset_factor_shortcut_simplifies_reflection_partner_regression() {
        let mut ctx = Context::new();
        let expr = parse("(2*sin(x)*cos(2*x)) * (cos(pi - u))", &mut ctx)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

        let Some((result, _)) = super::try_standard_trig_product_to_sum_subset_factor_shortcut(
            &crate::phase::SimplifyOptions::default(),
            &mut ctx,
            expr,
            false,
        ) else {
            panic!("product-to-sum subset factor shortcut should simplify reflection partners");
        };

        let rendered = render_expr(&ctx, result);
        assert!(rendered.contains("sin(3 * x) - sin(x)"));
        assert!(rendered.contains("cos(u)"));
    }

    #[test]
    fn trig_product_to_sum_subset_factor_shortcut_canonicalizes_double_angle_partner_regression() {
        let mut ctx = Context::new();
        let expr = parse("(2*sin(x)*cos(2*x)) * (sin(2*u))", &mut ctx)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

        let Some((result, _)) = super::try_standard_trig_product_to_sum_subset_factor_shortcut(
            &crate::phase::SimplifyOptions::default(),
            &mut ctx,
            expr,
            false,
        ) else {
            panic!("product-to-sum subset factor shortcut should rewrite direct-pair partners");
        };

        let rendered = render_expr(&ctx, result);
        assert!(rendered.contains("sin(3 * x) - sin(x)"));
        assert!(rendered.contains("2 * sin(u) * cos(u)"));
    }

    #[test]
    fn trig_product_to_sum_subset_factor_shortcut_simplifies_sqrt_partner_regression() {
        let mut ctx = Context::new();
        let expr = parse("(2*sin(x)*cos(2*x)) * (sqrt(18) - sqrt(2))", &mut ctx)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

        let Some((result, _)) = super::try_standard_trig_product_to_sum_subset_factor_shortcut(
            &crate::phase::SimplifyOptions::default(),
            &mut ctx,
            expr,
            false,
        ) else {
            panic!("product-to-sum subset factor shortcut should simplify sqrt partners");
        };

        let rendered = render_expr(&ctx, result);
        assert!(rendered.contains("sin(3 * x) - sin(x)"));
    }

    #[test]
    fn perfect_square_trinomial_factor_shortcut_matches_fundamental_exp_partner_regression() {
        let mut ctx = Context::new();
        let expr = parse("(x^2 + 2*x + 1) * (cosh(u) - sinh(u))", &mut ctx)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

        let Some((result, _)) = super::try_standard_perfect_square_trinomial_factor_shortcut(
            &crate::phase::SimplifyOptions::default(),
            &mut ctx,
            expr,
            false,
        ) else {
            panic!("perfect-square factor shortcut should match exp-decomposition partners");
        };

        let rendered = render_expr(&ctx, result);
        assert!(rendered.contains("(x + 1)^2"));
        assert!(rendered.contains("e^u") || rendered.contains("exp(-u)"));
    }

    #[test]
    fn perfect_square_trinomial_factor_shortcut_matches_tanh_partner_regression() {
        let mut ctx = Context::new();
        let expr = parse("(9*x^2 - 6*x + 1) * tanh(u)", &mut ctx)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

        let Some((result, _)) = super::try_standard_perfect_square_trinomial_factor_shortcut(
            &crate::phase::SimplifyOptions::default(),
            &mut ctx,
            expr,
            false,
        ) else {
            panic!("perfect-square factor shortcut should match tanh partners");
        };

        let rendered = render_expr(&ctx, result);
        assert!(rendered.contains("(3 * x - 1)^2") || rendered.contains("(1 - 3 * x)^2"));
        assert!(
            rendered.contains("tanh(u)")
                || rendered.contains("sinh(u)")
                || rendered.contains("cosh(u)")
        );
    }

    #[test]
    fn trig_product_to_sum_subset_factor_shortcut_canonicalizes_sum_to_product_partner_regression()
    {
        let mut ctx = Context::new();
        let expr = parse("(2*sin(x)*cos(2*x)) * (sin(u) + sin(3*u))", &mut ctx)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

        let Some((result, _)) = super::try_standard_trig_product_to_sum_subset_factor_shortcut(
            &crate::phase::SimplifyOptions::default(),
            &mut ctx,
            expr,
            false,
        ) else {
            panic!("product-to-sum subset factor shortcut should rewrite sum-to-product partners");
        };

        let rendered = render_expr(&ctx, result);
        assert!(rendered.contains("sin(3 * x) - sin(x)"));
        assert!(rendered.contains("2 * sin(2 * u) * cos(u)"));
    }

    #[test]
    fn special_angle_exact_value_factor_shortcut_matches_hyperbolic_exp_ratio_regression() {
        let mut ctx = Context::new();
        let expr = parse(
            "(cot(5*pi/12)) * ((exp(x)-exp(-x))/(exp(x)+exp(-x)))",
            &mut ctx,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

        let result = super::try_standard_special_angle_exact_value_factor_shortcut(
            &crate::phase::SimplifyOptions::default(),
            &mut ctx,
            expr,
            false,
        );

        assert!(
            result.is_some(),
            "special-angle exact-value factor shortcut should match hyperbolic exp-ratio products"
        );
    }

    #[test]
    fn special_angle_exact_value_factor_shortcut_matches_double_angle_regression() {
        let mut ctx = Context::new();
        let expr = parse("(cot(5*pi/12)) * (sin(2*x))", &mut ctx)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

        let result = super::try_standard_special_angle_exact_value_factor_shortcut(
            &crate::phase::SimplifyOptions::default(),
            &mut ctx,
            expr,
            false,
        );

        assert!(
            result.is_some(),
            "special-angle exact-value factor shortcut should match double-angle factors"
        );
    }

    #[test]
    fn special_angle_exact_value_factor_shortcut_matches_positive_double_cos_square_diff_regression(
    ) {
        let mut ctx = Context::new();
        let expr = parse("(tan(5*pi/12)) * (cos(2*u))", &mut ctx)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

        let result = super::try_standard_special_angle_exact_value_factor_shortcut(
            &crate::phase::SimplifyOptions::default(),
            &mut ctx,
            expr,
            false,
        );

        assert!(
            result.is_some(),
            "special-angle exact-value factor shortcut should match positive double-angle cosine factors"
        );
    }

    #[test]
    fn special_angle_exact_value_factor_shortcut_matches_small_exact_constant_partner_regression() {
        let mut ctx = Context::new();
        let expr = parse("(tan(5*pi/12)) * (cos(2*pi/3))", &mut ctx)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

        let result = super::try_standard_special_angle_exact_value_factor_shortcut(
            &crate::phase::SimplifyOptions::default(),
            &mut ctx,
            expr,
            false,
        );

        assert!(
            result.is_some(),
            "special-angle exact-value factor shortcut should match small exact constant partners"
        );
    }

    #[test]
    fn special_angle_exact_value_factor_shortcut_matches_cos_fourth_power_regression() {
        let mut ctx = Context::new();
        let expr = parse("(cot(5*pi/12)) * ((3+4*cos(2*x)+cos(4*x))/8)", &mut ctx)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

        let result = super::try_standard_special_angle_exact_value_factor_shortcut(
            &crate::phase::SimplifyOptions::default(),
            &mut ctx,
            expr,
            false,
        );

        assert!(
            result.is_some(),
            "special-angle exact-value factor shortcut should match cos-fourth-power reduction"
        );
    }

    #[test]
    fn special_angle_exact_value_factor_shortcut_matches_angle_sum_fraction_regression() {
        let mut ctx = Context::new();
        let expr = parse(
            "(cot(5*pi/12)) * ((sin(x)*cos(y)+cos(x)*sin(y))/(cos(x)*cos(y)-sin(x)*sin(y)))",
            &mut ctx,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

        let result = super::try_standard_special_angle_exact_value_factor_shortcut(
            &crate::phase::SimplifyOptions::default(),
            &mut ctx,
            expr,
            false,
        );

        assert!(
            result.is_some(),
            "special-angle exact-value factor shortcut should match angle-sum tangent fractions"
        );
    }

    #[test]
    fn special_angle_exact_value_factor_shortcut_matches_phase_shift_regression() {
        let mut ctx = Context::new();
        let expr = parse("(tan(5*pi/12)) * (cos(pi-u))", &mut ctx)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

        let Some((result, _)) = super::try_standard_special_angle_exact_value_factor_shortcut(
            &crate::phase::SimplifyOptions::default(),
            &mut ctx,
            expr,
            false,
        ) else {
            panic!("special-angle exact-value factor shortcut should match phase-shift partners");
        };

        let rendered = render_expr(&ctx, result);
        assert!(rendered.contains("cos(u)"));
        assert!(rendered.contains("-"));
    }

    #[test]
    fn special_angle_exact_value_factor_shortcut_matches_log_exp_inverse_regression() {
        let mut ctx = Context::new();
        let expr = parse("(tan(5*pi/12)) * (ln(exp(exp(u))))", &mut ctx)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

        let Some((result, _)) = super::try_standard_special_angle_exact_value_factor_shortcut(
            &crate::phase::SimplifyOptions::default(),
            &mut ctx,
            expr,
            false,
        ) else {
            panic!(
                "special-angle exact-value factor shortcut should match log-exp inverse partners"
            );
        };

        let rendered = render_expr(&ctx, result);
        assert!(rendered.contains("e^u") || rendered.contains("exp(u)"));
        assert!(!rendered.contains("ln("));
    }

    #[test]
    fn special_angle_exact_value_factor_shortcut_matches_inverse_trig_plan_regression() {
        let mut ctx = Context::new();
        let expr = parse("(tan(5*pi/12)) * (sin(arcsin(u)))", &mut ctx)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

        let Some((result, _)) = super::try_standard_special_angle_exact_value_factor_shortcut(
            &crate::phase::SimplifyOptions::default(),
            &mut ctx,
            expr,
            false,
        ) else {
            panic!(
                "special-angle exact-value factor shortcut should match direct inverse-trig partners"
            );
        };

        let rendered = render_expr(&ctx, result);
        assert!(rendered.contains("3^(1 / 2) + 2"));
        assert!(rendered.contains("u"));
        assert!(!rendered.contains("arcsin"));
    }

    #[test]
    fn special_angle_exact_value_factor_shortcut_matches_telescoping_fraction_regression() {
        let mut ctx = Context::new();
        let expr = parse("(tan(5*pi/12)) * (1/(u*(u+1)))", &mut ctx)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

        let Some((result, _)) = super::try_standard_special_angle_exact_value_factor_shortcut(
            &crate::phase::SimplifyOptions::default(),
            &mut ctx,
            expr,
            false,
        ) else {
            panic!(
                "special-angle exact-value factor shortcut should match telescoping-fraction partners"
            );
        };

        let rendered = render_expr(&ctx, result);
        assert!(rendered.contains("u + 1"));
        assert!(rendered.contains("/"));
    }

    #[test]
    fn special_angle_exact_value_factor_shortcut_matches_sqrt_abs_partner_regression() {
        let mut ctx = Context::new();
        let expr = parse("(tan(5*pi/12)) * (sqrt((u+1)^2))", &mut ctx)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

        let Some((result, _)) = super::try_standard_special_angle_exact_value_factor_shortcut(
            &crate::phase::SimplifyOptions::default(),
            &mut ctx,
            expr,
            false,
        ) else {
            panic!("special-angle exact-value factor shortcut should match sqrt-abs partners");
        };

        let rendered = render_expr(&ctx, result);
        assert!(rendered.contains("abs(u + 1)") || rendered.contains("|u + 1|"));
    }

    #[test]
    fn special_angle_exact_value_factor_shortcut_matches_perfect_square_polynomial_regression() {
        let mut ctx = Context::new();
        let expr = parse("(tan(5*pi/12)) * (u^4 + 4*u^2 + 4)", &mut ctx)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

        let Some((result, _)) = super::try_standard_special_angle_exact_value_factor_shortcut(
            &crate::phase::SimplifyOptions::default(),
            &mut ctx,
            expr,
            false,
        ) else {
            panic!(
                "special-angle exact-value factor shortcut should match perfect-square polynomial partners"
            );
        };

        let rendered = render_expr(&ctx, result);
        assert!(rendered.contains("(u^2 + 2)^2"));
    }

    #[test]
    fn tangent_addition_fraction_product_shortcut_matches_multiple_angle_regression() {
        let mut ctx = Context::new();
        let expr = parse("(sin(5*x)) * (sin(x+y)/(cos(x)*cos(y)))", &mut ctx)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

        let result = super::try_standard_tangent_addition_fraction_product_shortcut(
            &crate::phase::SimplifyOptions::default(),
            &mut ctx,
            expr,
            false,
        );

        assert!(
            result.is_some(),
            "tangent-addition fraction product shortcut should match explicit fraction products"
        );
    }

    #[test]
    fn simplify_pipeline_handles_multiple_angle_times_tangent_addition_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse("(sin(5*x)) * (tan(x) + tan(y))", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let expected = parse(
            "(sin(5*x)) * (sin(x+y)/(cos(x)*cos(y)))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        let difference = simplifier.context.add(Expr::Sub(rewritten, expected));
        assert!(super::isolated_simplify_rewrites_to_zero(
            &crate::phase::SimplifyOptions::default(),
            &mut simplifier.context,
            difference
        ));
    }

    #[test]
    fn detects_two_factor_product_pair_zero_difference_special_angle_product_to_sum_subset_regression(
    ) {
        let mut ctx = Context::new();
        let expr = parse(
            "((sin(5*pi/6)) * (2*sin(x)*cos(2*x))) - ((1/2) * (sin(3*x) - sin(x)))",
            &mut ctx,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

        assert!(
            matches_direct_two_factor_product_pair_zero_difference_root(&mut ctx, expr),
            "two-factor product matcher should recognize special-angle times product-to-sum residuals"
        );
    }

    #[test]
    fn simplify_pipeline_handles_multiple_angle_times_tangent_addition_fraction_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(sin(5*x)) * (sin(x+y)/(cos(x)*cos(y)))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let expected = parse(
            "(sin(5*x)*sin(x+y))/(cos(x)*cos(y))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        let difference = simplifier.context.add(Expr::Sub(rewritten, expected));
        assert!(super::isolated_simplify_rewrites_to_zero(
            &crate::phase::SimplifyOptions::default(),
            &mut simplifier.context,
            difference
        ));
    }

    #[test]
    fn simplify_pipeline_handles_multiple_angle_times_positive_double_cos_square_diff_factor_regression(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse("(sin(5*x)) * (2*cos(x)^2 - 1)", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(
            render(&simplifier.context, rewritten),
            "sin(5 * x) * cos(2 * x)"
        );
    }

    #[test]
    fn simplify_pipeline_handles_successive_unit_fraction_times_positive_double_cos_square_diff_zero_regression(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((1/x + 1/(x+1)) * cos(2*x)) - (((2*x+1)/(x*(x+1))) * (2*cos(x)^2 - 1))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_successive_unit_fraction_times_sin_cos_product_to_sum_zero_regression(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((1/x + 1/(x+1)) * (sin(x)*cos(y))) - (((2*x+1)/(x*(x+1))) * ((sin(x+y)+sin(x-y))/2))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_successive_unit_fraction_times_trig_power_zero_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((1/x + 1/(x+1)) * (sin(x)^2*cos(x)^2)) - (((2*x+1)/(x*(x+1))) * ((sin(2*x)^2)/4))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn detects_direct_successive_unit_fraction_pair_regression() {
        let mut ctx = Context::new();
        let lhs =
            parse("1/x + 1/(x+1)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs =
            parse("(2*x+1)/(x*(x+1))", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(matches_direct_addition_of_successive_unit_fractions_pair_root(&mut ctx, lhs, rhs));
    }

    #[test]
    fn detects_direct_reciprocal_sqrt_pair_regression() {
        let mut ctx = Context::new();
        let lhs = parse("1/sqrt(x)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs = parse("sqrt(x)/x", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(matches_direct_reciprocal_sqrt_pair_root(&mut ctx, lhs, rhs));
    }

    #[test]
    fn detects_direct_cos_fourth_power_reduction_pair_regression() {
        let mut ctx = Context::new();
        let lhs = parse("cos(x)^4", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs = parse("((3+4*cos(2*x)+cos(4*x))/8)", &mut ctx)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(matches_direct_cos_fourth_power_reduction_pair_root(
            &mut ctx, lhs, rhs
        ));
    }

    #[test]
    fn detects_direct_scaled_half_angle_square_pair_regression() {
        let mut ctx = Context::new();
        let lhs = parse("2*cos(u/2)^2", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs = parse("1 + cos(u)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(matches_direct_scaled_half_angle_square_pair_root(
            &mut ctx, lhs, rhs
        ));
    }

    #[test]
    fn detects_direct_abs_square_pair_regression() {
        let mut ctx = Context::new();
        let lhs =
            parse("abs(cos(x))^2", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs = parse("cos(x)^2", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(matches_direct_abs_square_pair_root(&ctx, lhs, rhs));
    }

    #[test]
    fn detects_direct_abs_trig_half_angle_pair_regression() {
        let mut ctx = Context::new();
        let lhs =
            parse("abs(sin(x/2))", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs =
            parse("sqrt((1-cos(x))/2)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(matches_direct_abs_trig_half_angle_pair_root(&ctx, lhs, rhs));
    }

    #[test]
    fn detects_direct_exponential_combination_pair_regression() {
        let mut ctx = Context::new();
        let lhs =
            parse("exp(a)*exp(b)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs = parse("exp(a+b)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(matches_direct_exponential_combination_pair_root(
            &mut ctx, lhs, rhs
        ));
    }

    #[test]
    fn detects_direct_hyperbolic_exp_sum_pair_regression() {
        let mut ctx = Context::new();
        let lhs =
            parse("cosh(u) - sinh(u)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs = parse("exp(-u)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(matches_direct_hyperbolic_exp_sum_pair_root(
            &mut ctx, lhs, rhs
        ));
    }

    #[test]
    fn detects_direct_quintuple_angle_pair_regression() {
        let mut ctx = Context::new();
        let lhs = parse("sin(5*x)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs = parse("16*sin(x)^5 - 20*sin(x)^3 + 5*sin(x)", &mut ctx)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(matches_direct_quintuple_angle_pair_root(&mut ctx, lhs, rhs));
    }

    #[test]
    fn detects_direct_hyperbolic_half_angle_square_pair_regression() {
        let mut ctx = Context::new();
        let lhs = parse("sinh(x/2)^2", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs =
            parse("(cosh(x)-1)/2", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(matches_direct_hyperbolic_half_angle_square_pair_root(
            &mut ctx, lhs, rhs
        ));
    }

    #[test]
    fn detects_direct_sum_to_product_contraction_pair_regression() {
        let mut ctx = Context::new();
        let lhs =
            parse("sin(x) + sin(3*x)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs =
            parse("2*sin(2*x)*cos(x)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(matches_direct_sum_to_product_contraction_pair_root(
            &mut ctx, lhs, rhs
        ));
    }

    #[test]
    fn detects_direct_tangent_addition_pair_regression() {
        let mut ctx = Context::new();
        let lhs =
            parse("tan(x) + tan(y)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs = parse("sin(x+y)/(cos(x)*cos(y))", &mut ctx)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(matches_direct_tangent_addition_pair_root(
            &mut ctx, lhs, rhs
        ));
    }

    #[test]
    fn detects_direct_tan_angle_sum_pair_regression() {
        let mut ctx = Context::new();
        let lhs = parse("tan(x+y)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs = parse("(tan(x)+tan(y))/(1 - tan(x)*tan(y))", &mut ctx)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(matches_direct_tan_angle_sum_pair_root(&mut ctx, lhs, rhs));
    }

    #[test]
    fn detects_two_factor_product_pair_zero_difference_regression() {
        let mut ctx = Context::new();
        let expr = parse(
            "((1/x + 1/(x+1)) * cos(2*x)) - (((2*x+1)/(x*(x+1))) * (2*cos(x)^2 - 1))",
            &mut ctx,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let view = AddView::from_expr(&ctx, expr);
        let lhs_factors = flatten_mul_chain(&mut ctx, view.terms[0].0)
            .into_iter()
            .map(|factor| render(&ctx, factor))
            .collect::<Vec<_>>();
        let rhs_factors = flatten_mul_chain(&mut ctx, view.terms[1].0)
            .into_iter()
            .map(|factor| render(&ctx, factor))
            .collect::<Vec<_>>();
        let lhs_term = view.terms[0].0;
        let rhs_term = view.terms[1].0;
        let lhs_factor_ids = flatten_mul_chain(&mut ctx, lhs_term);
        let rhs_factor_ids = flatten_mul_chain(&mut ctx, rhs_term);
        let pair_00 = factors_match_by_equality_or_direct_pair_root(
            &mut ctx,
            lhs_factor_ids[0],
            rhs_factor_ids[0],
        );
        let pair_01 = factors_match_by_equality_or_direct_pair_root(
            &mut ctx,
            lhs_factor_ids[0],
            rhs_factor_ids[1],
        );
        let pair_10 = factors_match_by_equality_or_direct_pair_root(
            &mut ctx,
            lhs_factor_ids[1],
            rhs_factor_ids[0],
        );
        let pair_11 = factors_match_by_equality_or_direct_pair_root(
            &mut ctx,
            lhs_factor_ids[1],
            rhs_factor_ids[1],
        );
        assert!(
            matches_direct_two_factor_product_pair_zero_difference_root(&mut ctx, expr),
            "lhs factors = {:?}, rhs factors = {:?}, pair00 = {}, pair01 = {}, pair10 = {}, pair11 = {}",
            lhs_factors,
            rhs_factors,
            pair_00,
            pair_01,
            pair_10,
            pair_11
        );
    }

    #[test]
    fn detects_two_factor_product_pair_zero_difference_sin_cos_regression() {
        let mut ctx = Context::new();
        let expr = parse(
            "((1/x + 1/(x+1)) * (sin(x)*cos(y))) - (((2*x+1)/(x*(x+1))) * ((sin(x+y)+sin(x-y))/2))",
            &mut ctx,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let view = AddView::from_expr(&ctx, expr);
        let lhs_factor_ids = flatten_mul_chain(&mut ctx, view.terms[0].0);
        let rhs_factor_ids = flatten_mul_chain(&mut ctx, view.terms[1].0);
        let lhs_factors = lhs_factor_ids
            .iter()
            .copied()
            .map(|factor| render(&ctx, factor))
            .collect::<Vec<_>>();
        let rhs_factors = rhs_factor_ids
            .iter()
            .copied()
            .map(|factor| render(&ctx, factor))
            .collect::<Vec<_>>();
        let pair_00 = factors_match_by_equality_or_direct_pair_root(
            &mut ctx,
            lhs_factor_ids[0],
            rhs_factor_ids[0],
        );
        let pair_01 = factors_match_by_equality_or_direct_pair_root(
            &mut ctx,
            lhs_factor_ids[0],
            rhs_factor_ids[1],
        );
        let pair_10 = factors_match_by_equality_or_direct_pair_root(
            &mut ctx,
            lhs_factor_ids[1],
            rhs_factor_ids[0],
        );
        let pair_11 = factors_match_by_equality_or_direct_pair_root(
            &mut ctx,
            lhs_factor_ids[1],
            rhs_factor_ids[1],
        );
        assert!(
            matches_direct_two_factor_product_pair_zero_difference_root(&mut ctx, expr),
            "lhs factors = {:?}, rhs factors = {:?}, pair00 = {}, pair01 = {}, pair10 = {}, pair11 = {}",
            lhs_factors,
            rhs_factors,
            pair_00,
            pair_01,
            pair_10,
            pair_11
        );
    }

    #[test]
    fn detects_two_factor_product_pair_zero_difference_tangent_addition_regression() {
        let mut ctx = Context::new();
        let expr = parse(
            "((sin(5*x)) * (tan(x) + tan(y))) - ((sin(5*x)) * (sin(x+y)/(cos(x)*cos(y))))",
            &mut ctx,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(matches_direct_two_factor_product_pair_zero_difference_root(
            &mut ctx, expr
        ));
    }

    #[test]
    fn detects_two_factor_product_pair_zero_difference_special_angle_tan_angle_sum_regression() {
        let mut ctx = Context::new();
        let expr = parse(
            "((cot(5*pi/12)) * (tan(x+y))) - (((2 - sqrt(3))) * ((tan(x)+tan(y))/(1 - tan(x)*tan(y))))",
            &mut ctx,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(matches_direct_two_factor_product_pair_zero_difference_root(
            &mut ctx, expr
        ));
    }

    #[test]
    fn detects_two_factor_product_pair_zero_difference_hyperbolic_half_angle_regression() {
        let mut ctx = Context::new();
        let expr = parse(
            "((1/x + 1/(x+1)) * (sinh(x/2)^2)) - (((2*x+1)/(x*(x+1))) * ((cosh(x)-1)/2))",
            &mut ctx,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let view = AddView::from_expr(&ctx, expr);
        let lhs_factor_ids = flatten_mul_chain(&mut ctx, view.terms[0].0);
        let rhs_factor_ids = flatten_mul_chain(&mut ctx, view.terms[1].0);
        let lhs_factors = lhs_factor_ids
            .iter()
            .copied()
            .map(|factor| render(&ctx, factor))
            .collect::<Vec<_>>();
        let rhs_factors = rhs_factor_ids
            .iter()
            .copied()
            .map(|factor| render(&ctx, factor))
            .collect::<Vec<_>>();
        assert!(
            matches_direct_two_factor_product_pair_zero_difference_root(&mut ctx, expr),
            "lhs factors = {:?}, rhs factors = {:?}",
            lhs_factors,
            rhs_factors
        );
    }

    #[test]
    fn detects_two_factor_product_pair_zero_difference_sum_to_product_regression() {
        let mut ctx = Context::new();
        let expr = parse(
            "((1/x + 1/(x+1)) * (sin(x) + sin(3*x))) - (((2*x+1)/(x*(x+1))) * (2*sin(2*x)*cos(x)))",
            &mut ctx,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let view = AddView::from_expr(&ctx, expr);
        let lhs_factor_ids = flatten_mul_chain(&mut ctx, view.terms[0].0);
        let rhs_factor_ids = flatten_mul_chain(&mut ctx, view.terms[1].0);
        let lhs_factors = lhs_factor_ids
            .iter()
            .copied()
            .map(|factor| render(&ctx, factor))
            .collect::<Vec<_>>();
        let rhs_factors = rhs_factor_ids
            .iter()
            .copied()
            .map(|factor| render(&ctx, factor))
            .collect::<Vec<_>>();
        assert!(
            matches_direct_two_factor_product_pair_zero_difference_root(&mut ctx, expr),
            "lhs factors = {:?}, rhs factors = {:?}",
            lhs_factors,
            rhs_factors
        );
    }

    #[test]
    fn detects_two_factor_product_pair_zero_difference_product_to_sum_pure_double_angle_regression()
    {
        let mut ctx = Context::new();
        let expr = parse(
            "((2*sin(x)*cos(2*x)) * (sin(2*u))) - ((sin(3*x) - sin(x)) * (2*sin(u)*cos(u)))",
            &mut ctx,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let view = AddView::from_expr(&ctx, expr);
        let lhs_factor_ids = flatten_mul_chain(&mut ctx, view.terms[0].0);
        let rhs_factor_ids = flatten_mul_chain(&mut ctx, view.terms[1].0);
        let lhs_factors = lhs_factor_ids
            .iter()
            .copied()
            .map(|factor| render(&ctx, factor))
            .collect::<Vec<_>>();
        let rhs_factors = rhs_factor_ids
            .iter()
            .copied()
            .map(|factor| render(&ctx, factor))
            .collect::<Vec<_>>();
        assert!(
            matches_direct_two_factor_product_pair_zero_difference_root(&mut ctx, expr),
            "lhs factors = {:?}, rhs factors = {:?}",
            lhs_factors,
            rhs_factors
        );
    }

    #[test]
    fn detects_two_factor_product_pair_zero_difference_product_to_sum_sum_to_product_regression() {
        let mut ctx = Context::new();
        let expr = parse(
            "((2*sin(x)*cos(2*x)) * (sin(u) + sin(3*u))) - ((sin(3*x) - sin(x)) * (2*sin(2*u)*cos(u)))",
            &mut ctx,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let view = AddView::from_expr(&ctx, expr);
        let lhs_factor_ids = flatten_mul_chain(&mut ctx, view.terms[0].0);
        let rhs_factor_ids = flatten_mul_chain(&mut ctx, view.terms[1].0);
        let lhs_factors = lhs_factor_ids
            .iter()
            .copied()
            .map(|factor| render(&ctx, factor))
            .collect::<Vec<_>>();
        let rhs_factors = rhs_factor_ids
            .iter()
            .copied()
            .map(|factor| render(&ctx, factor))
            .collect::<Vec<_>>();
        assert!(
            matches_direct_two_factor_product_pair_zero_difference_root(&mut ctx, expr),
            "lhs factors = {:?}, rhs factors = {:?}",
            lhs_factors,
            rhs_factors
        );
    }

    #[test]
    fn detects_direct_linear_factoring_pair_regression() {
        let mut ctx = Context::new();
        let lhs = parse("z^2 + 2*z", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs = parse("z*(z+2)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(super::matches_direct_linear_factoring_pair_root(
            &mut ctx, lhs, rhs
        ));
    }

    #[test]
    fn detects_direct_quartic_gcf_pair_regression() {
        let mut ctx = Context::new();
        let lhs = parse("z^4 - z^2", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs =
            parse("z^2*(z-1)*(z+1)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(super::matches_direct_quartic_gcf_pair_root(
            &mut ctx, lhs, rhs
        ));
    }

    #[test]
    fn detects_direct_difference_of_squares_quotient_pair_regression() {
        let mut ctx = Context::new();
        let lhs =
            parse("(z^2 - 9)/(z + 3)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs = parse("z - 3", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(super::matches_direct_difference_of_squares_quotient_pair_root(&mut ctx, lhs, rhs));
    }

    #[test]
    fn detects_direct_sum_diff_cubes_quotient_pair_regression() {
        let mut ctx = Context::new();
        let lhs =
            parse("(z^3 - 8)/(z - 2)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs =
            parse("z^2 + 2*z + 4", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(super::matches_direct_sum_diff_cubes_quotient_pair_root(
            &mut ctx, lhs, rhs
        ));
    }

    #[test]
    fn detects_direct_trig_phase_shift_pair_regression() {
        let mut ctx = Context::new();
        let lhs =
            parse("sin(pi/2 - z)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs = parse("cos(z)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(super::matches_direct_trig_phase_shift_pair_root(
            &mut ctx, lhs, rhs
        ));
    }

    #[test]
    fn detects_direct_trig_phase_shift_reflection_pair_regression() {
        let mut ctx = Context::new();
        let lhs = parse("cos(pi - u)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs = parse("-cos(u)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(super::matches_direct_trig_phase_shift_pair_root(
            &mut ctx, lhs, rhs
        ));
    }

    #[test]
    fn detects_direct_trig_triple_angle_pair_regression() {
        let mut ctx = Context::new();
        let lhs = parse("sin(3*z)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs = parse("3*sin(z) - 4*sin(z)^3", &mut ctx)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(super::matches_direct_trig_triple_angle_pair_root(
            &mut ctx, lhs, rhs
        ));
    }

    #[test]
    fn detects_direct_perfect_square_trinomial_pair_regression() {
        let mut ctx = Context::new();
        let lhs =
            parse("z^2 + 2*z + 1", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs = parse("(z+1)^2", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(super::matches_direct_perfect_square_trinomial_pair_root(
            &mut ctx, lhs, rhs
        ));
    }

    #[test]
    fn detects_direct_perfect_square_trinomial_fractional_pair_regression() {
        let mut ctx = Context::new();
        let lhs =
            parse("u^2 + u + 1/4", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs = parse("(u+1/2)^2", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(super::matches_direct_perfect_square_trinomial_pair_root(
            &mut ctx, lhs, rhs
        ));
    }

    #[test]
    fn detects_direct_successive_unit_fractions_pair_with_expanded_denominator_regression() {
        let mut ctx = Context::new();
        let lhs =
            parse("(1/z) + (1/(z+1))", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs =
            parse("(2*z+1)/(z^2+z)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(
            super::matches_direct_addition_of_successive_unit_fractions_pair_root(
                &mut ctx, lhs, rhs
            )
        );
    }

    #[test]
    fn detects_two_factor_product_pair_zero_difference_factoring_tangent_addition_regression() {
        let mut ctx = Context::new();
        let expr = parse(
            "((z^2 + 2*z) * (tan(u) + tan(v))) - (((z*(z+2)) * (sin(u+v)/(cos(u)*cos(v)))))",
            &mut ctx,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(matches_direct_two_factor_product_pair_zero_difference_root(
            &mut ctx, expr
        ));
    }

    #[test]
    fn detects_direct_sec_tan_pythagorean_pair_regression() {
        let mut ctx = Context::new();
        let lhs = parse("sec(z)^2 - tan(z)^2", &mut ctx)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs = parse("1", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(super::matches_direct_sec_tan_pythagorean_pair_root(
            &mut ctx, lhs, rhs
        ));
    }

    #[test]
    fn detects_direct_tan_to_sec_pythagorean_pair_regression() {
        let mut ctx = Context::new();
        let lhs = parse("1 + tan(z)^2", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs = parse("sec(z)^2", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(super::matches_direct_tan_to_sec_pythagorean_pair_root(
            &mut ctx, lhs, rhs
        ));
    }

    #[test]
    fn detects_direct_csc_cot_pythagorean_pair_regression() {
        let mut ctx = Context::new();
        let lhs = parse("csc(z)^2 - cot(z)^2", &mut ctx)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs = parse("1", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(super::matches_direct_csc_cot_pythagorean_pair_root(
            &mut ctx, lhs, rhs
        ));
    }

    #[test]
    fn detects_direct_cot_to_csc_pythagorean_pair_regression() {
        let mut ctx = Context::new();
        let lhs = parse("1 + cot(z)^2", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs = parse("csc(z)^2", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(super::matches_direct_cot_to_csc_pythagorean_pair_root(
            &mut ctx, lhs, rhs
        ));
    }

    #[test]
    fn detects_direct_hyperbolic_pythagorean_pair_regression() {
        let mut ctx = Context::new();
        let lhs = parse("cosh(z)^2 - sinh(z)^2", &mut ctx)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs = parse("1", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(super::matches_direct_hyperbolic_pythagorean_pair_root(
            &mut ctx, lhs, rhs
        ));
    }

    #[test]
    fn detects_direct_reciprocal_trig_product_one_pair_regression() {
        let mut ctx = Context::new();
        let lhs =
            parse("sin(z)*csc(z)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs = parse("1", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(super::matches_direct_reciprocal_trig_product_one_pair_root(
            &mut ctx, lhs, rhs
        ));
    }

    #[test]
    fn detects_direct_hyperbolic_triple_angle_pair_regression() {
        let mut ctx = Context::new();
        let lhs = parse("sinh(3*z)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs = parse("3*sinh(z) + 4*sinh(z)^3", &mut ctx)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(super::matches_direct_hyperbolic_triple_angle_pair_root(
            &mut ctx, lhs, rhs
        ));
    }

    #[test]
    fn detects_direct_small_exact_constant_pair_regression() {
        let mut ctx = Context::new();
        let lhs = parse("sec(pi)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs = parse("-1", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(super::matches_direct_small_exact_constant_pair_root(
            &mut ctx, lhs, rhs
        ));
    }

    #[test]
    fn detects_direct_special_angle_exact_value_pair_regression() {
        let mut ctx = Context::new();
        let lhs = parse("cot(5*pi/12)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs = parse("2 - 3^(1/2)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(super::matches_direct_special_angle_exact_value_pair_root(
            &mut ctx, lhs, rhs
        ));
    }

    #[test]
    fn detects_direct_special_angle_exact_value_pair_sqrt_form_regression() {
        let mut ctx = Context::new();
        let lhs = parse("cot(5*pi/12)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs = parse("2 - sqrt(3)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(super::matches_direct_special_angle_exact_value_pair_root(
            &mut ctx, lhs, rhs
        ));
    }

    #[test]
    fn detects_direct_special_angle_exact_value_half_regression() {
        let mut ctx = Context::new();
        let lhs = parse("sin(5*pi/6)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs = parse("1/2", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(super::matches_direct_special_angle_exact_value_pair_root(
            &mut ctx, lhs, rhs
        ));
    }

    #[test]
    fn detects_direct_special_angle_exact_value_negative_fraction_regression() {
        let mut ctx = Context::new();
        let lhs = parse("cos(2*pi/3)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs = parse("-1/2", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(super::matches_direct_special_angle_exact_value_pair_root(
            &mut ctx, lhs, rhs
        ));
    }

    #[test]
    fn detects_direct_inverse_trig_exact_value_pair_regression() {
        let mut ctx = Context::new();
        let lhs = parse("arcsin(1)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs = parse("pi/2", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(super::matches_direct_special_angle_exact_value_pair_root(
            &mut ctx, lhs, rhs
        ));
    }

    #[test]
    fn detects_direct_trig_inverse_composition_pair_regression() {
        let mut ctx = Context::new();
        let lhs =
            parse("sin(arctan(u))", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs =
            parse("u/sqrt(1 + u^2)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(super::matches_direct_trig_inverse_composition_pair_root(
            &mut ctx, lhs, rhs
        ));
    }

    #[test]
    fn detects_direct_hyperbolic_from_exp_pair_regression() {
        let mut ctx = Context::new();
        let lhs = parse("(exp(z)-exp(-z))/(exp(z)+exp(-z))", &mut ctx)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs = parse("tanh(z)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(super::matches_direct_hyperbolic_from_exp_pair_root(
            &mut ctx, lhs, rhs
        ));
    }

    #[test]
    fn detects_direct_tanh_to_sinh_cosh_pair_regression() {
        let mut ctx = Context::new();
        let lhs = parse("tanh(z)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs =
            parse("sinh(z)/cosh(z)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(super::matches_direct_tanh_to_sinh_cosh_pair_root(
            &mut ctx, lhs, rhs
        ));
    }

    #[test]
    fn detects_direct_cube_root_rationalization_pair_regression() {
        let mut ctx = Context::new();
        let lhs =
            parse("1/(1+z^(1/3))", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs = parse("(1-z^(1/3)+z^(2/3))/(1+z)", &mut ctx)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(super::matches_direct_cube_root_rationalization_pair_root(
            &mut ctx, lhs, rhs
        ));
    }

    #[test]
    fn detects_direct_hyperbolic_double_angle_sum_pair_regression() {
        let mut ctx = Context::new();
        let lhs = parse("cosh(2*z)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs = parse("cosh(z)^2 + sinh(z)^2", &mut ctx)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(super::matches_direct_hyperbolic_double_angle_sum_pair_root(
            &mut ctx, lhs, rhs
        ));
    }

    #[test]
    fn detects_direct_pure_double_angle_pair_regression() {
        let mut ctx = Context::new();
        let lhs = parse("sin(2*z)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs =
            parse("2*sin(z)*cos(z)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(super::matches_direct_pure_double_angle_pair_root(
            &mut ctx, lhs, rhs
        ));
    }

    #[test]
    fn detects_direct_double_angle_inverse_trig_pair_regression() {
        let mut ctx = Context::new();
        let lhs =
            parse("sin(2*arcsin(z))", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs =
            parse("2*z*sqrt(1-z^2)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(super::matches_direct_double_angle_inverse_trig_pair_root(
            &mut ctx, lhs, rhs
        ));
    }

    #[test]
    fn detects_direct_weierstrass_contraction_pair_regression() {
        let mut ctx = Context::new();
        let lhs = parse("2*tan(z/2)/(1 + tan(z/2)^2)", &mut ctx)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs = parse("sin(z)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(super::matches_direct_weierstrass_contraction_pair_root(
            &mut ctx, lhs, rhs
        ));
    }

    #[test]
    fn detects_direct_tanh_pythagorean_pair_regression() {
        let mut ctx = Context::new();
        let lhs =
            parse("1 - tanh(z)^2", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs = parse("1/cosh(z)^2", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(super::matches_direct_tanh_pythagorean_pair_root(
            &mut ctx, lhs, rhs
        ));
    }

    #[test]
    fn simplify_pipeline_handles_negative_exact_constant_factor_times_chebyshev_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((sec(pi)) * (cos(2*u))) - (((-1)) * (2*cos(u)^2 - 1))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_special_angle_cot_times_tangent_addition_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((cot(5*pi/12)) * (tan(x) + tan(y))) - (((2 - 3^(1/2))) * (sin(x+y)/(cos(x)*cos(y))))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_special_angle_cot_times_tan_angle_sum_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((cot(5*pi/12)) * (tan(x+y))) - (((2 - sqrt(3))) * ((tan(x)+tan(y))/(1 - tan(x)*tan(y))))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn detects_two_factor_product_pair_zero_difference_special_angle_hyperbolic_exp_ratio_regression(
    ) {
        let mut ctx = Context::new();
        let expr = parse(
            "((cot(5*pi/12)) * ((exp(x)-exp(-x))/(exp(x)+exp(-x)))) - (((2 - 3^(1/2))) * tanh(x))",
            &mut ctx,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(matches_direct_two_factor_product_pair_zero_difference_root(
            &mut ctx, expr
        ));
    }

    #[test]
    fn detects_two_factor_product_pair_zero_difference_special_angle_hyperbolic_double_angle_regression(
    ) {
        let mut ctx = Context::new();
        let expr = parse(
            "((cot(5*pi/12)) * (cosh(2*x))) - (((2 - 3^(1/2))) * (cosh(x)^2 + sinh(x)^2))",
            &mut ctx,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(matches_direct_two_factor_product_pair_zero_difference_root(
            &mut ctx, expr
        ));
    }

    #[test]
    fn detects_two_factor_product_pair_zero_difference_special_angle_hyperbolic_triple_angle_sqrt_form_regression(
    ) {
        let mut ctx = Context::new();
        let expr = parse(
            "((cot(5*pi/12)) * (sinh(3*x))) - (((2 - sqrt(3))) * (3*sinh(x) + 4*sinh(x)^3))",
            &mut ctx,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(matches_direct_two_factor_product_pair_zero_difference_root(
            &mut ctx, expr
        ));
    }

    #[test]
    fn simplify_pipeline_handles_special_angle_cot_times_hyperbolic_triple_angle_sqrt_form_regression(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((cot(5*pi/12)) * (sinh(3*x))) - (((2 - sqrt(3))) * (3*sinh(x) + 4*sinh(x)^3))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_special_angle_tan_times_positive_double_cos_square_diff_regression(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse("(tan(5*pi/12)) * (cos(2*u))", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let expected = parse("((2 + 3^(1/2))) * (cos(2*u))", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        let difference = simplifier.context.add(Expr::Sub(rewritten, expected));
        assert!(super::isolated_simplify_rewrites_to_zero(
            &crate::phase::SimplifyOptions::default(),
            &mut simplifier.context,
            difference
        ));
    }

    #[test]
    fn simplify_pipeline_handles_special_angle_tan_times_small_exact_constant_partner_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse("(tan(5*pi/12)) * (cos(2*pi/3))", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let expected = parse("(-1/2) * (3^(1/2) + 2)", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        let diff_expr = simplifier.context.add(Expr::Sub(rewritten, expected));
        let mut diff_orchestrator = Orchestrator::new();
        let (diff, _steps, _stats) =
            diff_orchestrator.simplify_pipeline(diff_expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, diff), "0");
    }

    #[test]
    fn simplify_pipeline_aligns_fractional_special_angle_with_short_geometric_partner_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = parse(
            "(cos(3*pi/8)) * (u^3 + u^2 + u + 1)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let target = parse(
            "((sqrt(2 - sqrt(2))/2) * ((u+1)*(u^2+1)))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (source_nf, _steps, _stats) = orchestrator.simplify_pipeline(source, &mut simplifier);
        let (target_nf, _steps, _stats) = orchestrator.simplify_pipeline(target, &mut simplifier);
        assert_eq!(
            render(&simplifier.context, source_nf),
            render(&simplifier.context, target_nf)
        );
    }

    #[test]
    fn simplify_pipeline_aligns_fractional_special_angle_with_shifted_square_partner_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = parse("(cos(3*pi/8)) * (u^2 + 2*u)", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let target = parse(
            "((sqrt(2 - sqrt(2))/2) * ((u+1)^2 - 1))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (source_nf, _steps, _stats) = orchestrator.simplify_pipeline(source, &mut simplifier);
        let (target_nf, _steps, _stats) = orchestrator.simplify_pipeline(target, &mut simplifier);
        assert_eq!(
            render(&simplifier.context, source_nf),
            render(&simplifier.context, target_nf)
        );
    }

    #[test]
    fn simplify_pipeline_aligns_fractional_special_angle_with_difference_of_squares_partner_regression(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = parse("(cos(3*pi/8)) * (u^2 - 4)", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let target = parse(
            "((sqrt(2 - sqrt(2))/2) * ((u-2)*(u+2)))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (source_nf, _steps, _stats) = orchestrator.simplify_pipeline(source, &mut simplifier);
        let (target_nf, _steps, _stats) = orchestrator.simplify_pipeline(target, &mut simplifier);
        assert_eq!(
            render(&simplifier.context, source_nf),
            render(&simplifier.context, target_nf)
        );
    }

    #[test]
    fn simplify_pipeline_aligns_fractional_special_angle_with_difference_of_cubes_partner_regression(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = parse("(cos(3*pi/8)) * (u^3 - 1)", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let target = parse(
            "((sqrt(2 - sqrt(2))/2) * ((u-1)*(u^2 + u + 1)))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (source_nf, _steps, _stats) = orchestrator.simplify_pipeline(source, &mut simplifier);
        let (target_nf, _steps, _stats) = orchestrator.simplify_pipeline(target, &mut simplifier);
        assert_eq!(
            render(&simplifier.context, source_nf),
            render(&simplifier.context, target_nf)
        );
    }

    #[test]
    fn simplify_pipeline_aligns_fractional_special_angle_with_sum_of_squares_partner_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = parse("(cos(3*pi/8)) * (u^2 + v^2)", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let target = parse(
            "((sqrt(2 - sqrt(2))/2) * ((u+v)^2 - 2*u*v))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (source_nf, _steps, _stats) = orchestrator.simplify_pipeline(source, &mut simplifier);
        let (target_nf, _steps, _stats) = orchestrator.simplify_pipeline(target, &mut simplifier);
        assert_eq!(
            render(&simplifier.context, source_nf),
            render(&simplifier.context, target_nf)
        );
    }

    #[test]
    fn simplify_pipeline_aligns_fractional_special_angle_with_abs_half_angle_partner_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = parse("(cos(3*pi/8)) * (abs(sin(u/2)))", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let target = parse(
            "((sqrt(2 - sqrt(2))/2) * (sqrt((1-cos(u))/2)))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (source_nf, _steps, _stats) = orchestrator.simplify_pipeline(source, &mut simplifier);
        let (target_nf, _steps, _stats) = orchestrator.simplify_pipeline(target, &mut simplifier);
        assert_eq!(
            render(&simplifier.context, source_nf),
            render(&simplifier.context, target_nf)
        );
    }

    #[test]
    fn simplify_pipeline_aligns_phi_with_abs_half_angle_partner_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = parse("(phi^2) * (abs(sin(u/2)))", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let target = parse(
            "((phi + 1) * (sqrt((1-cos(u))/2)))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (source_nf, _steps, _stats) = orchestrator.simplify_pipeline(source, &mut simplifier);
        let (target_nf, _steps, _stats) = orchestrator.simplify_pipeline(target, &mut simplifier);
        assert_eq!(
            render(&simplifier.context, source_nf),
            render(&simplifier.context, target_nf)
        );
    }

    #[test]
    fn simplify_pipeline_preserves_scaled_half_angle_partner_inside_fractional_special_angle_regression(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = parse("(cos(3*pi/8)) * (2*cos(u/2)^2)", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let target = parse(
            "((sqrt(2 - sqrt(2))/2) * (1 + cos(u)))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (source_nf, _steps, _stats) = orchestrator.simplify_pipeline(source, &mut simplifier);
        let (target_nf, _steps, _stats) = orchestrator.simplify_pipeline(target, &mut simplifier);
        let rendered = render(&simplifier.context, source_nf);
        assert!(rendered.contains("cos(u)"));
        let diff = simplifier.context.add(Expr::Sub(source_nf, target_nf));
        assert!(super::isolated_simplify_rewrites_to_zero(
            &crate::phase::SimplifyOptions::default(),
            &mut simplifier.context,
            diff
        ));
    }

    #[test]
    fn simplify_pipeline_aligns_fractional_special_angle_with_duplicate_sum_partner_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = parse("(cos(3*pi/8)) * (u+u)", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let target = parse("((sqrt(2 - sqrt(2))/2) * (2*u))", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (source_nf, _steps, _stats) = orchestrator.simplify_pipeline(source, &mut simplifier);
        let (target_nf, _steps, _stats) = orchestrator.simplify_pipeline(target, &mut simplifier);
        let diff = simplifier.context.add(Expr::Sub(source_nf, target_nf));
        assert!(super::isolated_simplify_rewrites_to_zero(
            &crate::phase::SimplifyOptions::default(),
            &mut simplifier.context,
            diff
        ));
    }

    #[test]
    fn simplify_pipeline_aligns_fractional_special_angle_with_partition_of_unity_partner_regression(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = parse(
            "(cos(3*pi/8)) * (exp(u)/(exp(u) + 1) + 1/(exp(u) + 1))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let target = parse("((sqrt(2 - sqrt(2))/2) * 1)", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (source_nf, _steps, _stats) = orchestrator.simplify_pipeline(source, &mut simplifier);
        let (target_nf, _steps, _stats) = orchestrator.simplify_pipeline(target, &mut simplifier);
        let diff = simplifier.context.add(Expr::Sub(source_nf, target_nf));
        assert!(super::isolated_simplify_rewrites_to_zero(
            &crate::phase::SimplifyOptions::default(),
            &mut simplifier.context,
            diff
        ));
    }

    #[test]
    fn simplify_pipeline_handles_special_angle_tan_times_two_linear_shift_partner_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse("(tan(5*pi/12)) * (u^2 + 5*u + 6)", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        let rendered = render_expr(&simplifier.context, rewritten);
        assert!(rendered.contains("3^(1 / 2) + 2"));
        assert!(rendered.contains("u + 2"));
        assert!(rendered.contains("u + 3"));
    }

    #[test]
    fn simplify_pipeline_handles_phase_shift_times_two_linear_shift_partner_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse("(cos(pi - x)) * (u^2 + 5*u + 6)", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        let rendered = render_expr(&simplifier.context, rewritten);
        assert!(rendered.starts_with("-"));
        assert!(rendered.contains("cos(x)"));
        assert!(rendered.contains("u + 2"));
        assert!(rendered.contains("u + 3"));
    }

    #[test]
    fn simplify_pipeline_handles_negative_cos_times_two_linear_shift_partner_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse("(-cos(x)) * (u^2 + 5*u + 6)", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        let rendered = render_expr(&simplifier.context, rewritten);
        assert!(rendered.starts_with("-"));
        assert!(rendered.contains("cos(x)"));
        assert!(rendered.contains("u + 2"));
        assert!(rendered.contains("u + 3"));
    }

    #[test]
    fn simplify_pipeline_handles_safe_anchor_times_geometric_sum_partner_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(sqrt(18) - sqrt(2)) * (u^3 + u^2 + u + 1)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let expected = parse("(2*sqrt(2)) * ((u+1)*(u^2+1))", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        let diff_expr = simplifier.context.add(Expr::Sub(rewritten, expected));
        let (diff, _steps, _stats) = orchestrator.simplify_pipeline(diff_expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, diff), "0");
    }

    #[test]
    fn simplify_pipeline_handles_safe_anchor_times_two_linear_shift_partner_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(sqrt(18) - sqrt(2)) * (u^2 + 5*u + 6)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let expected = parse("(2*sqrt(2)) * ((u+2)*(u+3))", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        let diff_expr = simplifier.context.add(Expr::Sub(rewritten, expected));
        let (diff, _steps, _stats) = orchestrator.simplify_pipeline(diff_expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, diff), "0");
    }

    #[test]
    fn simplify_pipeline_handles_hyperbolic_anchor_times_geometric_sum_partner_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((exp(x) - exp(-x))/2) * (u^3 + u^2 + u + 1)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let expected = parse("(sinh(x)) * ((u+1)*(u^2+1))", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        let diff_expr = simplifier.context.add(Expr::Sub(rewritten, expected));
        let (diff, _steps, _stats) = orchestrator.simplify_pipeline(diff_expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, diff), "0");
    }

    #[test]
    fn simplify_pipeline_handles_ratio_anchor_times_geometric_sum_partner_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(sqrt(8*x)/sqrt(2*x)) * (u^3 + u^2 + u + 1)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let expected = parse("2 * ((u+1)*(u^2+1))", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        let diff_expr = simplifier.context.add(Expr::Sub(rewritten, expected));
        let (diff, _steps, _stats) = orchestrator.simplify_pipeline(diff_expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, diff), "0");
    }

    #[test]
    fn simplify_pipeline_handles_safe_anchor_times_log_split_partner_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((exp(x) - exp(-x))/2) * (ln(sqrt(u)*v))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let expected = parse("(sinh(x)) * (ln(u)/2 + ln(v))", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        let diff_expr = simplifier.context.add(Expr::Sub(rewritten, expected));
        let (diff, _steps, _stats) = orchestrator.simplify_pipeline(diff_expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, diff), "0");
    }

    #[test]
    fn simplify_pipeline_handles_safe_anchor_times_exp_linear_shift_partner_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((exp(x) - exp(-x))/2) * (e*exp(u))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let expected = parse("(sinh(x)) * (exp(u+1))", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        let diff_expr = simplifier.context.add(Expr::Sub(rewritten, expected));
        let (diff, _steps, _stats) = orchestrator.simplify_pipeline(diff_expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, diff), "0");
    }

    #[test]
    fn simplify_pipeline_aligns_safe_anchor_with_successive_unit_fraction_partner_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = parse("(phi + 1) * (1/u + 1/(u+1))", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let target = parse(
            "((phi + 1) * ((2*u + 1)/(u*(u+1))))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (source_nf, _steps, _stats) = orchestrator.simplify_pipeline(source, &mut simplifier);
        let (target_nf, _steps, _stats) = orchestrator.simplify_pipeline(target, &mut simplifier);
        assert_eq!(
            render(&simplifier.context, source_nf),
            render(&simplifier.context, target_nf)
        );
    }

    #[test]
    fn simplify_pipeline_aligns_safe_anchor_with_abs_half_angle_partner_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = parse("(phi + 1) * (abs(cos(u/2)))", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let target = parse(
            "((phi + 1) * (sqrt((1+cos(u))/2)))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (source_nf, _steps, _stats) = orchestrator.simplify_pipeline(source, &mut simplifier);
        let (target_nf, _steps, _stats) = orchestrator.simplify_pipeline(target, &mut simplifier);
        assert_eq!(
            render(&simplifier.context, source_nf),
            render(&simplifier.context, target_nf)
        );
    }

    #[test]
    fn simplify_pipeline_handles_safe_anchor_times_positive_scaled_half_angle_partner_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse("(2*sqrt(2)) * (1 + cos(u))", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let expected = parse("(2*sqrt(2)) * (2*cos(u/2)^2)", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        let diff_expr = simplifier.context.add(Expr::Sub(rewritten, expected));
        let (diff, _steps, _stats) = orchestrator.simplify_pipeline(diff_expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, diff), "0");
    }

    #[test]
    fn simplify_pipeline_aligns_scaled_half_angle_anchor_with_sum_diff_cubes_partner_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = parse("(2*cos(x/2)^2) * (u^3 + v^3)", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let target = parse(
            "(1 + cos(x)) * ((u+v)*(u^2 - u*v + v^2))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (source_nf, _steps, _stats) = orchestrator.simplify_pipeline(source, &mut simplifier);
        let (target_nf, _steps, _stats) = orchestrator.simplify_pipeline(target, &mut simplifier);
        assert_eq!(
            render(&simplifier.context, source_nf),
            render(&simplifier.context, target_nf)
        );
    }

    #[test]
    fn simplify_pipeline_aligns_scaled_half_angle_anchor_with_higher_degree_difference_partner_regression(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = parse("(2*cos(x/2)^2) * (u^6 - 1)", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let target = parse(
            "(1 + cos(x)) * ((u^2+u+1)*(u^2-u+1)*(u+1)*(u-1))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (source_nf, _steps, _stats) = orchestrator.simplify_pipeline(source, &mut simplifier);
        let (target_nf, _steps, _stats) = orchestrator.simplify_pipeline(target, &mut simplifier);
        assert_eq!(
            render(&simplifier.context, source_nf),
            render(&simplifier.context, target_nf)
        );
    }

    #[test]
    fn simplify_pipeline_avoids_scaled_half_angle_anchor_loop_with_safe_anchor_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse("(2*sqrt(2)) * (1 + cos(u))", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let expected = parse("(2*sqrt(2)) * (2*cos(u/2)^2)", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        let diff_expr = simplifier.context.add(Expr::Sub(rewritten, expected));
        let (diff, _steps, _stats) = orchestrator.simplify_pipeline(diff_expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, diff), "0");
    }

    #[test]
    fn simplify_pipeline_avoids_scaled_half_angle_anchor_loop_with_constant_partner_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse("(2*cos(x/2)^2) * 2", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let expected = parse("(1 + cos(x)) * 2", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        let diff_expr = simplifier.context.add(Expr::Sub(rewritten, expected));
        let (diff, _steps, _stats) = orchestrator.simplify_pipeline(diff_expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, diff), "0");
    }

    #[test]
    fn simplify_pipeline_handles_two_factor_fractional_perfect_square_partner_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse("((x^2 + 2)^2) * (u^2 + u + 1/4)", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        let rendered = render_expr(&simplifier.context, rewritten);
        assert!(rendered.contains("(u + 1/2)^2"));
        assert!(rendered.contains("(x^2 + 2)^2"));
    }

    #[test]
    fn detects_two_factor_product_pair_zero_difference_special_angle_pure_double_angle_regression()
    {
        let mut ctx = Context::new();
        let expr = parse(
            "((cot(5*pi/12)) * (sin(2*x))) - (((2 - 3^(1/2))) * (2*sin(x)*cos(x)))",
            &mut ctx,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(matches_direct_two_factor_product_pair_zero_difference_root(
            &mut ctx, expr
        ));
    }

    #[test]
    fn detects_two_factor_product_pair_zero_difference_special_angle_tanh_pythagorean_regression() {
        let mut ctx = Context::new();
        let expr = parse(
            "((cot(5*pi/12)) * (1/cosh(x)^2)) - (((2 - 3^(1/2))) * (1 - tanh(x)^2))",
            &mut ctx,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(matches_direct_two_factor_product_pair_zero_difference_root(
            &mut ctx, expr
        ));
    }

    #[test]
    fn detects_two_factor_product_pair_zero_difference_perfect_square_tanh_fraction_regression() {
        let mut ctx = Context::new();
        let expr = parse(
            "((9*x^2 - 6*x + 1) * tanh(u)) - (((3*x - 1)^2) * (sinh(u)/cosh(u)))",
            &mut ctx,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(matches_direct_two_factor_product_pair_zero_difference_root(
            &mut ctx, expr
        ));
    }

    #[test]
    fn detects_two_factor_product_pair_zero_difference_perfect_square_cube_rationalization_regression(
    ) {
        let mut ctx = Context::new();
        let expr = parse(
            "((x^2 + 2*x + 1) * (1/(1+u^(1/3)))) - (((x+1)^2) * ((1-u^(1/3)+u^(2/3))/(1+u)))",
            &mut ctx,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(matches_direct_two_factor_product_pair_zero_difference_root(
            &mut ctx, expr
        ));
    }

    #[test]
    fn simplify_pipeline_handles_sum_diff_cubes_quotient_times_chebyshev_zero_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(((z^3 - 8)/(z - 2)) * (cos(2*u))) - (((z^2 + 2*z + 4)) * (2*cos(u)^2 - 1))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_quartic_gcf_times_chebyshev_zero_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((z^4 - z^2) * (cos(2*u))) - (((z^2*(z-1)*(z+1))) * (2*cos(u)^2 - 1))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_difference_of_squares_quotient_times_chebyshev_zero_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(((z^2 - 9)/(z + 3)) * (cos(2*u))) - (((z - 3)) * (2*cos(u)^2 - 1))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_quartic_gcf_times_hyperbolic_triple_angle_zero_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((z^4 - z^2) * (sinh(3*u))) - (((z^2*(z-1)*(z+1))) * (3*sinh(u) + 4*sinh(u)^3))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_sec_tan_pythagorean_times_chebyshev_zero_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((sec(u)^2 - tan(u)^2) * (cos(2*u))) - ((1) * (2*cos(u)^2 - 1))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn detects_two_factor_product_pair_zero_difference_cot_to_csc_chebyshev_regression() {
        let mut ctx = Context::new();
        let expr = parse(
            "((1 + cot(u)^2) * (cos(2*u))) - (((csc(u)^2)) * (2*cos(u)^2 - 1))",
            &mut ctx,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(matches_direct_two_factor_product_pair_zero_difference_root(
            &mut ctx, expr
        ));
    }

    #[test]
    fn simplify_pipeline_handles_cot_to_csc_pythagorean_times_chebyshev_zero_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((1 + cot(u)^2) * (cos(2*u))) - (((csc(u)^2)) * (2*cos(u)^2 - 1))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn detects_two_factor_product_pair_zero_difference_reciprocal_sqrt_chebyshev_regression() {
        let mut ctx = Context::new();
        let expr = parse(
            "((1/sqrt(x)) * (cos(2*x))) - (((sqrt(x)/x) * (2*cos(x)^2 - 1)))",
            &mut ctx,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(matches_direct_two_factor_product_pair_zero_difference_root(
            &mut ctx, expr
        ));
    }

    #[test]
    fn detects_two_factor_product_pair_zero_difference_exp_chebyshev_regression() {
        let mut ctx = Context::new();
        let expr = parse(
            "((exp(a)*exp(b)) * (cos(2*x))) - ((exp(a+b)) * (2*cos(x)^2 - 1))",
            &mut ctx,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(matches_direct_two_factor_product_pair_zero_difference_root(
            &mut ctx, expr
        ));
    }

    #[test]
    fn detects_two_factor_product_pair_zero_difference_quintuple_product_to_sum_regression() {
        let mut ctx = Context::new();
        let expr = parse(
            "((sin(5*x)) * (2*sin(x)*cos(2*x))) - (((16*sin(x)^5 - 20*sin(x)^3 + 5*sin(x)) * (sin(3*x) - sin(x))))",
            &mut ctx,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(matches_direct_two_factor_product_pair_zero_difference_root(
            &mut ctx, expr
        ));
    }

    #[test]
    fn detects_two_factor_product_pair_zero_difference_quintuple_chebyshev_regression() {
        let mut ctx = Context::new();
        let expr = parse(
            "((sin(5*x)) * (cos(2*x))) - (((16*sin(x)^5 - 20*sin(x)^3 + 5*sin(x)) * (2*cos(x)^2 - 1)))",
            &mut ctx,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(matches_direct_two_factor_product_pair_zero_difference_root(
            &mut ctx, expr
        ));
    }

    #[test]
    fn detects_two_factor_product_pair_zero_difference_quartic_gcf_power_reduction_regression() {
        let mut ctx = Context::new();
        let expr = parse(
            "((x^4 - x^2) * (cos(x)^2)) - (((x^2*(x-1)*(x+1)) * ((1 + cos(2*x))/2)))",
            &mut ctx,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(matches_direct_two_factor_product_pair_zero_difference_root(
            &mut ctx, expr
        ));
    }

    #[test]
    fn detects_two_factor_product_pair_zero_difference_quartic_gcf_sum_to_product_regression() {
        let mut ctx = Context::new();
        let expr = parse(
            "((x^4 - x^2) * (sin(x) + sin(3*x))) - (((x^2*(x-1)*(x+1)) * (2*sin(2*x)*cos(x))))",
            &mut ctx,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(matches_direct_two_factor_product_pair_zero_difference_root(
            &mut ctx, expr
        ));
    }

    #[test]
    fn detects_two_factor_product_pair_zero_difference_fractional_perfect_square_regression() {
        let mut ctx = Context::new();
        let expr = parse(
            "((x^2 + 2*x + 1) * ((u+1/2)^2)) - (((x+1)^2) * (u^2 + u + 1/4))",
            &mut ctx,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(matches_direct_two_factor_product_pair_zero_difference_root(
            &mut ctx, expr
        ));
    }

    #[test]
    fn detects_two_factor_product_pair_zero_difference_fractional_square_sum_diff_cubes_regression()
    {
        let mut ctx = Context::new();
        let expr = parse(
            "(((x+1/2)^2) * (u^3 + v^3)) - (((x^2 + x + 1/4)) * ((u+v)*(u^2-u*v+v^2)))",
            &mut ctx,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(matches_direct_two_factor_product_pair_zero_difference_root(
            &mut ctx, expr
        ));
    }

    #[test]
    fn detects_two_factor_product_pair_zero_difference_fractional_square_higher_degree_difference_regression(
    ) {
        let mut ctx = Context::new();
        let expr = parse(
            "(((x+1/2)^2) * (u^6 - 1)) - (((x^2 + x + 1/4)) * ((u^2+u+1)*(u^2-u+1)*(u+1)*(u-1)))",
            &mut ctx,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(matches_direct_two_factor_product_pair_zero_difference_root(
            &mut ctx, expr
        ));
    }

    #[test]
    fn detects_two_factor_product_pair_zero_difference_fractional_square_sophie_germain_regression()
    {
        let mut ctx = Context::new();
        let expr = parse(
            "(((x+1/2)^2) * (u^4 + 4)) - (((x^2 + x + 1/4)) * ((u^2 + 2*u + 2)*(u^2 - 2*u + 2)))",
            &mut ctx,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(matches_direct_two_factor_product_pair_zero_difference_root(
            &mut ctx, expr
        ));
    }

    #[test]
    fn detects_two_factor_product_pair_zero_difference_perfect_square_exp_sum_regression() {
        let mut ctx = Context::new();
        let expr = parse(
            "((x^2 + 2*x + 1) * (cosh(u) - sinh(u))) - (((x+1)^2) * exp(-u))",
            &mut ctx,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(matches_direct_two_factor_product_pair_zero_difference_root(
            &mut ctx, expr
        ));
    }

    #[test]
    fn simplify_pipeline_handles_reciprocal_sqrt_times_positive_double_cos_square_diff_zero_regression(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((1/sqrt(x)) * (cos(2*x))) - (((sqrt(x)/x) * (2*cos(x)^2 - 1)))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_quartic_gcf_times_power_reduction_zero_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((x^4 - x^2) * (cos(x)^2)) - (((x^2*(x-1)*(x+1)) * ((1 + cos(2*x))/2)))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_quartic_gcf_times_sum_to_product_zero_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((x^4 - x^2) * (sin(x) + sin(3*x))) - (((x^2*(x-1)*(x+1)) * (2*sin(2*x)*cos(x))))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_cos_fourth_over_chebyshev_zero_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(cos(x)^4)/(cos(2*x)) - (((3+4*cos(2*x)+cos(4*x))/8)/(2*cos(x)^2 - 1))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_fraction_over_chebyshev_zero_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(1/x + 1/(x+1))/(cos(2*x)) - (((2*x+1)/(x*(x+1)))/(2*cos(x)^2 - 1))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_fraction_over_abs_square_zero_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(1/x + 1/(x+1))/(abs(cos(x))^2) - (((2*x+1)/(x*(x+1)))/(cos(x)^2))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn detects_division_factor_pairs_for_cos_fourth_over_chebyshev_regression() {
        let mut ctx = Context::new();
        let lhs_num = parse("cos(x)^4", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs_num = parse("((3+4*cos(2*x)+cos(4*x))/8)", &mut ctx)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let lhs_den = parse("cos(2*x)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs_den =
            parse("2*cos(x)^2 - 1", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(factors_match_by_equality_or_direct_pair_root(
            &mut ctx, lhs_num, rhs_num
        ));
        assert!(factors_match_by_equality_or_direct_pair_root(
            &mut ctx, lhs_den, rhs_den
        ));
    }

    #[test]
    fn detects_direct_quotient_pair_zero_difference_cos_fourth_chebyshev_regression() {
        let mut ctx = Context::new();
        let expr = parse(
            "(cos(x)^4)/(cos(2*x)) - (((3+4*cos(2*x)+cos(4*x))/8)/(2*cos(x)^2 - 1))",
            &mut ctx,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(matches_direct_quotient_pair_zero_difference_root(
            &mut ctx, expr
        ));
    }

    #[test]
    fn detects_direct_quotient_pair_zero_difference_fraction_chebyshev_regression() {
        let mut ctx = Context::new();
        let expr = parse(
            "(1/x + 1/(x+1))/(cos(2*x)) - (((2*x+1)/(x*(x+1)))/(2*cos(x)^2 - 1))",
            &mut ctx,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(matches_direct_quotient_pair_zero_difference_root(
            &mut ctx, expr
        ));
    }

    #[test]
    fn detects_direct_quotient_pair_zero_difference_fraction_abs_square_regression() {
        let mut ctx = Context::new();
        let expr = parse(
            "(1/x + 1/(x+1))/(abs(cos(x))^2) - (((2*x+1)/(x*(x+1)))/(cos(x)^2))",
            &mut ctx,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(matches_direct_quotient_pair_zero_difference_root(
            &mut ctx, expr
        ));
    }

    #[test]
    fn detects_direct_quotient_pair_zero_difference_tanh_half_angle_regression() {
        let mut ctx = Context::new();
        let expr = parse(
            "(tanh(2*x))/(abs(sin(x/2))) - ((2*tanh(x)/(1+tanh(x)^2))/(sqrt((1-cos(x))/2)) )",
            &mut ctx,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(matches_direct_quotient_pair_zero_difference_root(
            &mut ctx, expr
        ));
    }

    #[test]
    fn simplify_pipeline_handles_cos_fourth_over_known_angle_zero_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(cos(x)^4)/(cos(2*pi/5)) - (((3+4*cos(2*x)+cos(4*x))/8)/((sqrt(5)-1)/4))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_cos_fourth_over_exp_log_zero_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(cos(x)^4)/(ln(exp(x)^2)) - (((3+4*cos(2*x)+cos(4*x))/8)/(2*x))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_cos_fourth_over_completing_square_zero_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(cos(x)^4)/(x^2 + 2*x) - (((3+4*cos(2*x)+cos(4*x))/8)/(x*(x+2)))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_exp_combination_times_positive_double_cos_square_diff_zero_regression(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((exp(a)*exp(b)) * (cos(2*x))) - ((exp(a+b)) * (2*cos(x)^2 - 1))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_quintuple_angle_times_product_to_sum_zero_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((sin(5*x)) * (2*sin(x)*cos(2*x))) - (((16*sin(x)^5 - 20*sin(x)^3 + 5*sin(x)) * (sin(3*x) - sin(x))))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_quintuple_angle_times_positive_double_cos_square_diff_zero_regression(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((sin(5*x)) * (cos(2*x))) - (((16*sin(x)^5 - 20*sin(x)^3 + 5*sin(x)) * (2*cos(x)^2 - 1)))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_half_angle_against_small_trig_zero_shifted_quotient_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((sin(x)^2 - (1 - cos(2*x))/2) + 1)/((2*cos(2*x)*sin(x) - (4*cos(x)^2*sin(x) - 2*sin(x))) + 1)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "1");
    }

    #[test]
    fn simplify_pipeline_handles_half_angle_against_hyperbolic_sinh_cubic_sum_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(sin(x)^2 - (1 - cos(2*x))/2) + (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x)))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_trig_binomial_square_against_exp_hyperbolic_sum_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(cosh(x) + sinh(x) - e^x) + ((sin(x) + cos(x))^2 - (1 + sin(2*x)))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_trig_binomial_square_against_exp_hyperbolic_product_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(cosh(x) + sinh(x) - e^x) * ((sin(x) + cos(x))^2 - (1 + sin(2*x)))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_trig_binomial_square_against_exp_hyperbolic_shifted_quotient_regression(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((cosh(x) + sinh(x) - e^x) + 1)/(((sin(x) + cos(x))^2 - (1 + sin(2*x))) + 1)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "1");
    }

    #[test]
    fn simplify_pipeline_handles_exp_hyperbolic_against_hyperbolic_sinh_cubic_sum_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(cosh(x) + sinh(x) - e^x) + (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x)))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_exp_hyperbolic_against_hyperbolic_sinh_cubic_difference_regression(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(cosh(x) + sinh(x) - e^x) - (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x)))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_phase_shift_against_hyperbolic_cosh_cubic_sum_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(3*sin(x) + 4*cos(x) - 5*sin(x + arctan(4/3))) + (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x)))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_phase_shift_against_hyperbolic_cosh_cubic_difference_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(3*sin(x) + 4*cos(x) - 5*sin(x + arctan(4/3))) - (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x)))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_hyperbolic_sum_against_hyperbolic_cosh_cubic_sum_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x))) + (sinh(x+y) - (sinh(x)*cosh(y) + cosh(x)*sinh(y)))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_hyperbolic_sum_against_hyperbolic_cosh_cubic_difference_regression(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x))) - (sinh(x+y) - (sinh(x)*cosh(y) + cosh(x)*sinh(y)))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_hyperbolic_sum_against_reciprocal_trig_shifted_quotient_regression(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((sinh(x+y) - (sinh(x)*cosh(y) + cosh(x)*sinh(y))) + 1)/((tan(x) + cot(x) - sec(x)*csc(x)) + 1)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "1");
    }

    #[test]
    fn simplify_pipeline_handles_hyperbolic_cosh_cubic_against_telescoping_sum_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x))) + (1/(u*(u+1)) - 1/u + 1/(u+1))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_hyperbolic_sum_against_telescoping_sum_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(sinh(x+y) - (sinh(x)*cosh(y) + cosh(x)*sinh(y))) + (1/(u*(u+1)) - 1/u + 1/(u+1))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_trig_binomial_square_against_hyperbolic_pythagorean_product_regression(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((sin(x) + cos(x))^2 - (1 + sin(2*x))) * (cosh(x)^2 - sinh(x)^2 - 1)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_trig_binomial_square_against_hyperbolic_pythagorean_shifted_quotient_regression(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(((sin(x) + cos(x))^2 - (1 + sin(2*x))) + 1)/((cosh(x)^2 - sinh(x)^2 - 1) + 1)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "1");
    }

    #[test]
    fn simplify_pipeline_handles_trig_binomial_square_against_exp_cosh_product_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((sin(x) + cos(x))^2 - (1 + sin(2*x))) * (exp(x) + exp(-x) - 2*cosh(x))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_trig_binomial_square_against_exp_cosh_shifted_quotient_regression()
    {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(((sin(x) + cos(x))^2 - (1 + sin(2*x))) + 1)/((exp(x) + exp(-x) - 2*cosh(x)) + 1)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "1");
    }

    #[test]
    fn simplify_pipeline_handles_small_trig_zero_pair_product_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(2*cos(2*x)*sin(x) - (4*cos(x)^2*sin(x) - 2*sin(x))) * (2*sin(2*x)*sin(x) - (4*cos(x) - 4*cos(x)^3))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_small_trig_zero_pair_shifted_quotient_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((2*cos(2*x)*sin(x) - (4*cos(x)^2*sin(x) - 2*sin(x))) + 1)/((2*sin(2*x)*sin(x) - (4*cos(x) - 4*cos(x)^3)) + 1)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "1");
    }

    #[test]
    fn simplify_pipeline_handles_trig_cubic_passthrough_shifted_quotient_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((2*sin(2*x)*sin(x)+a) + 1)/((4*cos(x)-4*cos(x)^3+a) + 1)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "1");
    }

    #[test]
    fn simplify_pipeline_handles_trig_cubic_scaled_difference_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "k*(2*sin(2*x)*sin(x)) - k*(4*cos(x)-4*cos(x)^3)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_trig_cubic_common_denominator_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((2*sin(2*x)*sin(x))/q) - ((4*cos(x)-4*cos(x)^3)/q)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_trig_cubic_passthrough_common_denominator_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((2*sin(2*x)*sin(x)+a)/q) - ((4*cos(x)-4*cos(x)^3+a)/q)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn detects_direct_small_zero_telescoping_vs_half_angle_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(1/(x - 1) - 1/(x + 1) - 2/(x^2 - 1)) - (sin(x)^2 - (1 - cos(2*x))/2)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let lhs = parse(
            "1/(x - 1) - 1/(x + 1) - 2/(x^2 - 1)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs = parse("sin(x)^2 - (1 - cos(2*x))/2", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(matches_direct_small_zero_identity_root(
            &mut simplifier.context,
            lhs,
        ));
        assert!(matches_direct_small_zero_identity_root(
            &mut simplifier.context,
            rhs,
        ));
        assert!(is_direct_small_zero_composition_candidate_root(
            &mut simplifier.context,
            expr,
        ));
    }

    #[test]
    fn direct_small_zero_pair_shortcut_handles_telescoping_vs_half_angle_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(1/(x - 1) - 1/(x + 1) - 2/(x^2 - 1)) - (sin(x)^2 - (1 - cos(2*x))/2)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let options = SimplifyOptions::default();
        let (rewritten, _steps) = try_standard_direct_small_zero_pair_shortcut(
            &options,
            &mut simplifier.context,
            expr,
            options.collect_steps,
        )
        .unwrap_or_else(|| panic!("expected direct small zero shortcut to match"));
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_telescoping_vs_half_angle_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(1/(x - 1) - 1/(x + 1) - 2/(x^2 - 1)) - (sin(x)^2 - (1 - cos(2*x))/2)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_trig_mixed_scaled_difference_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "k*(2*cos(2*x)*sin(x)) - k*(4*cos(x)^2*sin(x)-2*sin(x))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn common_scale_residual_extracts_trig_product_to_sum_sin_sin_scaled_difference_regression() {
        let mut ctx = Context::new();
        let expr = parse("k*(2*sin(x)*sin(y)) - k*(cos(x-y) - cos(x+y))", &mut ctx)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let (common_factor, residual_expr) =
            extract_common_multiplicative_residual_sum_root(&mut ctx, expr)
                .unwrap_or_else(|| panic!("expected common multiplicative residual"));
        assert_eq!(render(&ctx, common_factor), "k");
        assert_eq!(
            render(&ctx, residual_expr),
            "2 * sin(x) * sin(y) - (cos(x - y) - cos(x + y))"
        );
    }

    #[test]
    fn common_scale_residual_matches_trig_product_to_sum_sin_sin_scaled_difference_regression() {
        let mut ctx = Context::new();
        let residual_expr = parse("(2*sin(x)*sin(y)) - (cos(x-y) - cos(x+y))", &mut ctx)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(matches_direct_small_zero_or_known_pair_residual_root(
            &mut ctx,
            residual_expr
        ));
    }

    #[test]
    fn common_scale_fallback_matches_trig_product_to_sum_sin_sin_scaled_difference_regression() {
        let mut ctx = Context::new();
        let expr = parse("k*(2*sin(x)*sin(y)) - k*(cos(x-y) - cos(x+y))", &mut ctx)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let options = SimplifyOptions::default();
        let (rewritten, _steps) =
            try_standard_common_scale_exact_zero_shortcut_fallback(&options, &mut ctx, expr, false)
                .unwrap_or_else(|| panic!("expected common-scale fallback to match"));
        assert_eq!(render(&ctx, rewritten), "0");
    }

    #[test]
    fn common_scale_known_pair_shortcut_matches_trig_product_to_sum_sin_sin_scaled_difference_regression(
    ) {
        let mut ctx = Context::new();
        let expr = parse("k*(2*sin(x)*sin(y)) - k*(cos(x-y) - cos(x+y))", &mut ctx)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let (rewritten, _steps) =
            try_standard_common_scale_known_pair_shortcut(&mut ctx, expr, false)
                .unwrap_or_else(|| panic!("expected common-scale known-pair shortcut to match"));
        assert_eq!(render(&ctx, rewritten), "0");
    }

    #[test]
    fn direct_known_pair_zero_shortcut_skips_common_scale_trig_product_to_sum_regression() {
        let mut ctx = Context::new();
        let expr = parse("k*(2*sin(x)*sin(y)) - k*(cos(x-y) - cos(x+y))", &mut ctx)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let options = SimplifyOptions::default();
        assert!(
            try_standard_direct_known_pair_zero_shortcut(&options, &mut ctx, expr, false).is_none()
        );
    }

    #[test]
    fn two_factor_product_pair_zero_shortcut_skips_common_scale_trig_product_to_sum_regression() {
        let mut ctx = Context::new();
        let expr = parse("k*(2*sin(x)*sin(y)) - k*(cos(x-y) - cos(x+y))", &mut ctx)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let options = SimplifyOptions::default();
        assert!(try_standard_two_factor_product_pair_zero_shortcut(
            &options, &mut ctx, expr, false
        )
        .is_none());
    }

    #[test]
    fn exact_zero_equivalence_shortcut_matches_trig_product_to_sum_sin_sin_scaled_difference_regression(
    ) {
        let mut ctx = Context::new();
        let expr = parse("k*(2*sin(x)*sin(y)) - k*(cos(x-y) - cos(x+y))", &mut ctx)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let options = SimplifyOptions::default();
        let (rewritten, _steps) =
            try_standard_exact_zero_equivalence_shortcut(&options, &mut ctx, expr, false)
                .unwrap_or_else(|| panic!("expected exact-zero shortcut to match"));
        assert_eq!(render(&ctx, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_trig_product_to_sum_sin_sin_scaled_difference_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "k*(2*sin(x)*sin(y)) - k*(cos(x-y) - cos(x+y))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_trig_mixed_common_denominator_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((2*cos(2*x)*sin(x))/q) - ((4*cos(x)^2*sin(x)-2*sin(x))/q)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_trig_mixed_passthrough_shifted_quotient_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((2*cos(2*x)*sin(x)+a) + 1)/((4*cos(x)^2*sin(x)-2*sin(x)+a) + 1)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "1");
    }

    #[test]
    fn simplify_pipeline_handles_small_mixed_trig_hyperbolic_zero_sum_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(2*cos(2*x)*sin(x) - (4*cos(x)^2*sin(x) - 2*sin(x))) + (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x)))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_small_mixed_trig_hyperbolic_zero_difference_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(2*cos(2*x)*sin(x) - (4*cos(x)^2*sin(x) - 2*sin(x))) - (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x)))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_log_product_split_against_trig_mixed_sum_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + (2*cos(2*x)*sin(x) - (4*cos(x)^2*sin(x) - 2*sin(x)))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn nested_exact_zero_child_shortcut_handles_log_product_split_against_trig_mixed_sum_regression(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + (2*cos(2*x)*sin(x) - (4*cos(x)^2*sin(x) - 2*sin(x)))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let options = SimplifyOptions::default();
        let (rewritten, _steps) = try_standard_nested_exact_zero_child_shortcut(
            &options,
            &mut simplifier.context,
            expr,
            false,
        )
        .unwrap_or_else(|| panic!("expected nested exact-zero shortcut to match"));
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn nested_exact_zero_child_shortcut_handles_log_product_split_against_trig_mixed_sum_with_steps_regression(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + (2*cos(2*x)*sin(x) - (4*cos(x)^2*sin(x) - 2*sin(x)))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let options = SimplifyOptions::default();
        let (rewritten, _steps) = try_standard_nested_exact_zero_child_shortcut(
            &options,
            &mut simplifier.context,
            expr,
            true,
        )
        .unwrap_or_else(|| panic!("expected nested exact-zero shortcut to match"));
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn nested_exact_zero_child_shortcut_handles_log_product_split_against_sin_sin_sum_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + (2*sin(x)*sin(y) - cos(x-y) + cos(x+y))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let options = SimplifyOptions::default();
        let (rewritten, _steps) = try_standard_nested_exact_zero_child_shortcut(
            &options,
            &mut simplifier.context,
            expr,
            true,
        )
        .unwrap_or_else(|| panic!("expected nested exact-zero shortcut to match"));
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn shifted_quotient_shortcut_handles_log_product_split_against_sin_cos_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + 1)/((2*sin(x)*cos(y) - sin(x+y) - sin(x-y)) + 1)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let options = SimplifyOptions::default();
        let (rewritten, _steps) = try_standard_shifted_quotient_nested_zero_core_shortcut(
            &options,
            &mut simplifier.context,
            expr,
            true,
        )
        .unwrap_or_else(|| panic!("expected shifted quotient nested-zero shortcut to match"));
        assert_eq!(render(&simplifier.context, rewritten), "1");
    }

    #[test]
    fn shifted_quotient_shortcut_handles_pythagorean_factor_form_from_sin_sq_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((sin(x)^2) + 1)/((1-cos(x)^2) + 1)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let options = SimplifyOptions::default();
        let (rewritten, _steps) = try_standard_shifted_quotient_exact_one_shortcut(
            &options,
            &mut simplifier.context,
            expr,
            true,
        )
        .unwrap_or_else(|| panic!("expected shifted quotient exact-one shortcut to match"));
        assert_eq!(render(&simplifier.context, rewritten), "1");
    }

    #[test]
    fn simplify_pipeline_handles_pythagorean_factor_form_from_sin_sq_shifted_quotient_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((sin(x)^2) + 1)/((1-cos(x)^2) + 1)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "1");
    }

    #[test]
    fn shifted_quotient_shortcut_handles_trig_mixed_against_exp_sinh_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((2*cos(2*x)*sin(x) - (4*cos(x)^2*sin(x) - 2*sin(x))) + 1)/((exp(x) - exp(-x) - 2*sinh(x)) + 1)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let options = SimplifyOptions::default();
        let (rewritten, _steps) = try_standard_shifted_quotient_nested_zero_core_shortcut(
            &options,
            &mut simplifier.context,
            expr,
            false,
        )
        .unwrap_or_else(|| panic!("expected shifted quotient shortcut to match"));
        assert_eq!(render(&simplifier.context, rewritten), "1");
    }

    #[test]
    fn simplify_pipeline_handles_half_angle_against_telescoping_fraction_sum_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(sin(x)^2 - (1 - cos(2*x))/2) + (1/(u*(u+1)) - 1/u + 1/(u+1))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_trig_mixed_against_telescoping_fraction_sum_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(2*cos(2*x)*sin(x) - (4*cos(x)^2*sin(x) - 2*sin(x))) + (1/(u*(u+1)) - 1/u + 1/(u+1))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_contextual_rational_square_composition_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((1/(u - 1) + 1/(u + 1)) + ((v+1)^2)) - ((2*u/(u^2 - 1)) + (v^2 + 2*v + 1))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_contextual_tanh_square_composition_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((9*u^2 - 6*u + 1) + tanh(2*v)) - (((3*u - 1)^2) + (2*tanh(v)/(1 + tanh(v)^2)))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_contextual_multivariate_tanh_composition_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(((x^2 + y^2)*(a^2 + b^2)) + tanh(2*u)) - (((x*a + y*b)^2 + (x*b - y*a)^2) + (2*tanh(u)/(1 + tanh(u)^2)))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn detects_direct_small_pow_expansion_pair_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lhs = parse("(v+1)^2", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs = parse("v^2 + 2*v + 1", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(super::matches_direct_small_pow_expansion_pair_root(
            &mut simplifier.context,
            lhs,
            rhs
        ));
    }

    #[test]
    fn detects_direct_small_pow_expansion_pair_subtractive_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lhs = parse("(3*u - 1)^2", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs = parse("9*u^2 - 6*u + 1", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(super::matches_direct_small_pow_expansion_pair_root(
            &mut simplifier.context,
            lhs,
            rhs
        ));
    }

    #[test]
    fn detects_direct_small_pow_expansion_pair_trinomial_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lhs = parse("(a + b + c)^2", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs = parse(
            "a^2 + b^2 + c^2 + 2*a*b + 2*a*c + 2*b*c",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(super::matches_direct_small_pow_expansion_pair_root(
            &mut simplifier.context,
            lhs,
            rhs
        ));
    }

    #[test]
    fn rejects_trig_binomial_square_in_small_pow_expansion_pair_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lhs = parse("(sin(x)+cos(x))^2", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs = parse("1+sin(2*x)", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(!super::matches_direct_small_pow_expansion_pair_root(
            &mut simplifier.context,
            lhs,
            rhs
        ));
    }

    #[test]
    fn shared_passthrough_small_pow_expansion_shortcut_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((a + b + c)^2 + m) - ((a^2 + b^2 + c^2 + 2*a*b + 2*a*c + 2*b*c) + m)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let (rewritten, _steps) =
            super::try_standard_shared_passthrough_small_pow_expansion_shortcut(
                &mut simplifier.context,
                expr,
                true,
            )
            .unwrap_or_else(|| panic!("expected passthrough small-pow shortcut"));
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn detects_direct_rational_plus_minus_one_sum_pair_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lhs = parse("1/(u - 1) + 1/(u + 1)", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs = parse("2*u/(u^2 - 1)", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(super::matches_direct_rational_plus_minus_one_sum_pair_root(
            &mut simplifier.context,
            lhs,
            rhs
        ));
    }

    #[test]
    fn detects_direct_tanh_double_angle_pair_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lhs = parse("tanh(2*v)", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs = parse("2*tanh(v)/(1 + tanh(v)^2)", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(super::matches_direct_tanh_double_angle_pair_root(
            &mut simplifier.context,
            lhs,
            rhs
        ));
    }

    #[test]
    fn detects_direct_sum_of_squares_product_pair_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lhs = parse("(x^2 + y^2)*(a^2 + b^2)", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs = parse("(x*a + y*b)^2 + (x*b - y*a)^2", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(super::matches_direct_sum_of_squares_product_pair_root(
            &mut simplifier.context,
            lhs,
            rhs
        ));
    }

    #[test]
    fn detects_direct_sum_diff_cubes_product_pair_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lhs = parse("(u+v)*(u^2-u*v+v^2)", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs = parse("u^3 + v^3", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(super::matches_direct_sum_diff_cubes_product_pair_root(
            &mut simplifier.context,
            lhs,
            rhs
        ));
    }

    #[test]
    fn detects_direct_higher_degree_difference_pair_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lhs = parse("u^6 - 1", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs = parse("((u^2+u+1)*(u^2-u+1)*(u+1)*(u-1))", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(super::matches_direct_higher_degree_difference_pair_root(
            &mut simplifier.context,
            lhs,
            rhs
        ));
    }

    #[test]
    fn detects_direct_sophie_germain_pair_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lhs = parse("u^4 + 4", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs = parse("((u^2 + 2*u + 2)*(u^2 - 2*u + 2))", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(super::matches_direct_sophie_germain_pair_root(
            &mut simplifier.context,
            lhs,
            rhs
        ));
    }

    #[test]
    fn detects_direct_three_linear_shift_product_pair_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lhs = parse("(u+1)*(u+2)*(u+3)", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs = parse("u^3 + 6*u^2 + 11*u + 6", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(super::matches_direct_three_linear_shift_product_pair_root(
            &mut simplifier.context,
            lhs,
            rhs
        ));
    }

    #[test]
    fn detects_direct_two_linear_shift_product_pair_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lhs = parse("(u+2)*(u+3)", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs = parse("u^2 + 5*u + 6", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(super::matches_direct_two_linear_shift_product_pair_root(
            &mut simplifier.context,
            lhs,
            rhs
        ));
    }

    #[test]
    fn detects_two_factor_product_pair_zero_difference_product_to_sum_reflection_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((2*sin(x)*cos(2*x)) * (cos(pi - u))) - ((sin(3*x) - sin(x)) * (-cos(u)))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(
            super::matches_direct_two_factor_product_pair_zero_difference_root(
                &mut simplifier.context,
                expr
            )
        );
    }

    #[test]
    fn detects_two_factor_product_pair_zero_difference_product_to_sum_two_linear_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((2*sin(x)*cos(2*x)) * ((u+2)*(u+3))) - ((sin(3*x) - sin(x)) * (u^2 + 5*u + 6))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(
            super::matches_direct_two_factor_product_pair_zero_difference_root(
                &mut simplifier.context,
                expr
            )
        );
    }

    #[test]
    fn detects_two_factor_product_pair_zero_difference_product_to_sum_three_linear_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((2*sin(x)*cos(2*x)) * ((u+1)*(u+2)*(u+3))) - ((sin(3*x) - sin(x)) * (u^3 + 6*u^2 + 11*u + 6))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(
            super::matches_direct_two_factor_product_pair_zero_difference_root(
                &mut simplifier.context,
                expr
            )
        );
    }

    #[test]
    fn detects_two_factor_product_pair_zero_difference_product_to_sum_inverse_trig_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((2*sin(x)*cos(2*x)) * (sin(arctan(u)))) - ((sin(3*x) - sin(x)) * (u/sqrt(1 + u^2)))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(
            super::matches_direct_two_factor_product_pair_zero_difference_root(
                &mut simplifier.context,
                expr
            )
        );
    }

    #[test]
    fn detects_two_factor_product_pair_zero_difference_product_to_sum_double_angle_inverse_trig_regression(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((2*sin(x)*cos(2*x)) * (sin(2*arcsin(u)))) - ((sin(3*x) - sin(x)) * (2*u*sqrt(1-u^2)))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(
            super::matches_direct_two_factor_product_pair_zero_difference_root(
                &mut simplifier.context,
                expr
            )
        );
    }

    #[test]
    fn detects_two_factor_product_pair_zero_difference_product_to_sum_weierstrass_sin_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((2*sin(x)*cos(2*x)) * (sin(u))) - ((sin(3*x) - sin(x)) * (2*tan(u/2)/(1 + tan(u/2)^2)))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(
            super::matches_direct_two_factor_product_pair_zero_difference_root(
                &mut simplifier.context,
                expr
            )
        );
    }

    #[test]
    fn detects_two_factor_product_pair_zero_difference_product_to_sum_higher_binomial_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((2*sin(x)*cos(2*x)) * ((u-1)^5)) - ((sin(3*x) - sin(x)) * (u^5 - 5*u^4 + 10*u^3 - 10*u^2 + 5*u - 1))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(
            super::matches_direct_two_factor_product_pair_zero_difference_root(
                &mut simplifier.context,
                expr
            )
        );
    }

    #[test]
    fn detects_two_factor_product_pair_zero_difference_product_to_sum_log_split_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((2*sin(x)*cos(2*x)) * (ln(sqrt(u)*v))) - ((sin(3*x) - sin(x)) * (ln(u)/2 + ln(v)))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(
            super::matches_direct_two_factor_product_pair_zero_difference_root(
                &mut simplifier.context,
                expr
            )
        );
    }

    #[test]
    fn detects_two_factor_product_pair_zero_difference_product_to_sum_higher_degree_difference_regression(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((2*sin(x)*cos(2*x)) * (u^6 - 1)) - ((sin(3*x) - sin(x)) * ((u^2+u+1)*(u^2-u+1)*(u+1)*(u-1)))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(
            super::matches_direct_two_factor_product_pair_zero_difference_root(
                &mut simplifier.context,
                expr
            )
        );
    }

    #[test]
    fn detects_two_factor_product_pair_zero_difference_product_to_sum_cauchy_schwarz_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((2*sin(x)*cos(2*x)) * ((w^2 + p^2)*(u^2 + v^2))) - ((sin(3*x) - sin(x)) * ((w*u + p*v)^2 + (w*v - p*u)^2))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(
            super::matches_direct_two_factor_product_pair_zero_difference_root(
                &mut simplifier.context,
                expr
            )
        );
    }

    #[test]
    fn direct_product_to_sum_factor_partner_matches_cauchy_schwarz_regression() {
        let mut ctx = Context::new();
        let lhs = parse("(2*sin(x)*cos(2*x))", &mut ctx)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs =
            parse("sin(3*x) - sin(x)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(super::matches_direct_trig_product_to_sum_sin_cos_pair_root(
            &mut ctx, lhs, rhs
        ));

        let partner_lhs = parse("((w^2 + p^2)*(u^2 + v^2))", &mut ctx)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let partner_rhs = parse("((w*u + p*v)^2 + (w*v - p*u)^2)", &mut ctx)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(super::factors_match_by_equality_or_direct_pair_root(
            &mut ctx,
            partner_lhs,
            partner_rhs
        ));
    }

    #[test]
    fn detects_two_factor_product_pair_zero_difference_product_to_sum_inverse_trig_constant_regression(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((2*sin(x)*cos(2*x)) * (arcsin(1))) - ((sin(3*x) - sin(x)) * (pi/2))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(
            super::matches_direct_two_factor_product_pair_zero_difference_root(
                &mut simplifier.context,
                expr
            )
        );
    }

    #[test]
    fn detects_two_factor_product_pair_zero_difference_product_to_sum_special_angle_constant_regression(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((2*sin(x)*cos(2*x)) * (cos(2*pi/3))) - ((sin(3*x) - sin(x)) * (-1/2))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(
            super::matches_direct_two_factor_product_pair_zero_difference_root(
                &mut simplifier.context,
                expr
            )
        );
    }

    #[test]
    fn simplify_pipeline_handles_product_to_sum_three_linear_zero_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((2*sin(x)*cos(2*x)) * ((u+1)*(u+2)*(u+3))) - ((sin(3*x) - sin(x)) * (u^3 + 6*u^2 + 11*u + 6))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_product_to_sum_two_linear_zero_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((2*sin(x)*cos(2*x)) * ((u+2)*(u+3))) - ((sin(3*x) - sin(x)) * (u^2 + 5*u + 6))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_product_to_sum_pure_double_angle_zero_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((2*sin(x)*cos(2*x)) * (sin(2*u))) - ((sin(3*x) - sin(x)) * (2*sin(u)*cos(u)))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_product_to_sum_sum_to_product_zero_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((2*sin(x)*cos(2*x)) * (sin(u) + sin(3*u))) - ((sin(3*x) - sin(x)) * (2*sin(2*u)*cos(u)))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_product_to_sum_inverse_trig_zero_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((2*sin(x)*cos(2*x)) * (sin(arctan(u)))) - ((sin(3*x) - sin(x)) * (u/sqrt(1 + u^2)))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_product_to_sum_double_angle_inverse_trig_zero_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((2*sin(x)*cos(2*x)) * (sin(2*arcsin(u)))) - ((sin(3*x) - sin(x)) * (2*u*sqrt(1-u^2)))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_three_linear_shift_anchor_times_double_angle_inverse_trig_partner_regression(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((u+1)*(u+2)*(u+3)) * (sin(2*arcsin(x)))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let expected = parse(
            "(u^3 + 6*u^2 + 11*u + 6) * 2*x*(1-x^2)^(1/2)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        let difference = simplifier.context.add(Expr::Sub(rewritten, expected));
        assert!(super::isolated_simplify_rewrites_to_zero(
            &crate::phase::SimplifyOptions::default(),
            &mut simplifier.context,
            difference
        ));
    }

    #[test]
    fn simplify_pipeline_handles_three_linear_shift_anchor_times_radical_product_partner_regression(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((u+1)*(u+2)*(u+3)) * (sqrt(x)*sqrt(4*x))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let expected = parse("(u^3 + 6*u^2 + 11*u + 6) * 2*x", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        let difference = simplifier.context.add(Expr::Sub(rewritten, expected));
        assert!(super::isolated_simplify_rewrites_to_zero(
            &crate::phase::SimplifyOptions::default(),
            &mut simplifier.context,
            difference
        ));
    }

    #[test]
    fn simplify_pipeline_handles_three_linear_shift_anchor_times_inverse_trig_partner_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((u+1)*(u+2)*(u+3)) * (sin(arctan(x)))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let expected = parse(
            "(u^3 + 6*u^2 + 11*u + 6) * (x/sqrt(1+x^2))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        let difference = simplifier.context.add(Expr::Sub(rewritten, expected));
        assert!(super::isolated_simplify_rewrites_to_zero(
            &crate::phase::SimplifyOptions::default(),
            &mut simplifier.context,
            difference
        ));
    }

    #[test]
    fn simplify_pipeline_handles_three_linear_shift_anchor_times_tangent_addition_partner_regression(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((u+1)*(u+2)*(u+3)) * (tan(x) + tan(y))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let expected = parse(
            "(u^3 + 6*u^2 + 11*u + 6) * (sin(x+y)/(cos(x)*cos(y)))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        let difference = simplifier.context.add(Expr::Sub(rewritten, expected));
        assert!(super::isolated_simplify_rewrites_to_zero(
            &crate::phase::SimplifyOptions::default(),
            &mut simplifier.context,
            difference
        ));
    }

    #[test]
    fn simplify_pipeline_handles_tangent_addition_anchor_times_sum_diff_cubes_partner_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse("(tan(x) + tan(y)) * (u^3 - 1)", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let expected = parse(
            "(sin(x+y)/(cos(x)*cos(y))) * ((u-1)*(u^2 + u + 1))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        let difference = simplifier.context.add(Expr::Sub(rewritten, expected));
        assert!(super::isolated_simplify_rewrites_to_zero(
            &crate::phase::SimplifyOptions::default(),
            &mut simplifier.context,
            difference
        ));
    }

    #[test]
    fn simplify_pipeline_handles_tangent_addition_anchor_times_log_split_partner_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(tan(x) + tan(y)) * (ln(sqrt(u)*v))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let expected = parse(
            "(sin(x+y)/(cos(x)*cos(y))) * (ln(u)/2 + ln(v))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        let difference = simplifier.context.add(Expr::Sub(rewritten, expected));
        assert!(super::isolated_simplify_rewrites_to_zero(
            &crate::phase::SimplifyOptions::default(),
            &mut simplifier.context,
            difference
        ));
    }

    #[test]
    fn simplify_pipeline_aligns_inverse_trig_anchor_with_short_geometric_partner_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = parse(
            "(sin(arctan(x))) * (u^3 + u^2 + u + 1)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let target = parse(
            "(x/sqrt(1 + x^2)) * ((u+1)*(u^2 + 1))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (target_nf, _steps, _stats) = orchestrator.simplify_pipeline(target, &mut simplifier);
        let (source_nf, _steps, _stats) = orchestrator.simplify_pipeline(source, &mut simplifier);
        let expected = "x * (x^2 + 1)^(1/2) * (u^3 + u^2 + u + 1) / (x^2 + 1)";
        assert_eq!(render(&simplifier.context, source_nf), expected);
        assert_eq!(render(&simplifier.context, target_nf), expected);
    }

    #[test]
    fn simplify_pipeline_aligns_inverse_trig_anchor_with_two_linear_shift_partner_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let source = parse(
            "(sin(arctan(x))) * (u^2 + 5*u + 6)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let target = parse("(x/sqrt(1 + x^2)) * ((u+2)*(u+3))", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (target_nf, _steps, _stats) = orchestrator.simplify_pipeline(target, &mut simplifier);
        let (source_nf, _steps, _stats) = orchestrator.simplify_pipeline(source, &mut simplifier);
        let expected = "x * (x^2 + 1)^(1/2) * (u^2 + 5 * u + 6) / (x^2 + 1)";
        assert_eq!(render(&simplifier.context, source_nf), expected);
        assert_eq!(render(&simplifier.context, target_nf), expected);
    }

    #[test]
    fn simplify_pipeline_handles_product_to_sum_weierstrass_sin_zero_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((2*sin(x)*cos(2*x)) * (sin(u))) - ((sin(3*x) - sin(x)) * (2*tan(u/2)/(1 + tan(u/2)^2)))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_product_to_sum_higher_binomial_zero_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((2*sin(x)*cos(2*x)) * ((u-1)^5)) - ((sin(3*x) - sin(x)) * (u^5 - 5*u^4 + 10*u^3 - 10*u^2 + 5*u - 1))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_product_to_sum_cauchy_schwarz_zero_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((2*sin(x)*cos(2*x)) * ((w^2 + p^2)*(u^2 + v^2))) - ((sin(3*x) - sin(x)) * ((w*u + p*v)^2 + (w*v - p*u)^2))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_product_to_sum_inverse_trig_constant_zero_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((2*sin(x)*cos(2*x)) * (arcsin(1))) - ((sin(3*x) - sin(x)) * (pi/2))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_product_to_sum_special_angle_constant_zero_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((2*sin(x)*cos(2*x)) * (cos(2*pi/3))) - ((sin(3*x) - sin(x)) * (-1/2))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn detects_composed_small_additive_pair_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lhs = parse(
            "(1/(u - 1) + 1/(u + 1)) + ((v+1)^2)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs = parse("(2*u/(u^2 - 1)) + (v^2 + 2*v + 1)", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(super::matches_composed_small_additive_pair_root(
            &mut simplifier.context,
            lhs,
            rhs
        ));
    }

    #[test]
    fn detects_composed_small_additive_tanh_square_pair_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lhs = parse("(9*u^2 - 6*u + 1) + tanh(2*v)", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let rhs = parse(
            "((3*u - 1)^2) + (2*tanh(v)/(1 + tanh(v)^2))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(super::matches_composed_small_additive_pair_root(
            &mut simplifier.context,
            lhs,
            rhs
        ));
    }

    #[test]
    fn simplify_pipeline_handles_trig_cubic_against_general_phase_shift_sum_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(2*sin(2*x)*sin(x) - (4*cos(x) - 4*cos(x)^3)) + (3*sin(x) + 4*cos(x) - 5*sin(x + arctan(4/3)))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_trig_cubic_against_hyperbolic_cubic_sum_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(2*sin(2*x)*sin(x) - (4*cos(x) - 4*cos(x)^3)) + (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x)))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_cubes_quotient_against_common_factor_sum_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((a^3-b^3)/(a-b) - (a^2 + a*b + b^2)) + (x*y + x*z - x*(y+z))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_cubes_quotient_against_common_factor_shifted_quotient_regression()
    {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(((a^3-b^3)/(a-b) - (a^2 + a*b + b^2)) + 1)/((x*y + x*z - x*(y+z)) + 1)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "1");
    }

    #[test]
    fn simplify_pipeline_handles_cubes_quotient_against_binomial_square_shifted_quotient_regression(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(((a^3-b^3)/(a-b) - (a^2 + a*b + b^2)) + 1)/((x^2 + 2*x + 1 - (x+1)^2) + 1)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "1");
    }

    #[test]
    fn simplify_pipeline_factors_small_polynomial_denominator_binomial_square_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse("1/(u^2 + 2*u + 1)", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "1 / (u + 1)^2");
    }

    #[test]
    fn simplify_pipeline_factors_small_polynomial_denominator_cubic_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse("1/(u^3 + u^2 + u + 1)", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(
            render(&simplifier.context, rewritten),
            "1 / ((u + 1) * (u^2 + 1))"
        );
    }

    #[test]
    fn detects_direct_perfect_square_trinomial_zero_identity_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse("x^2 + 2*x + 1 - (x+1)^2", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(matches_direct_perfect_square_trinomial_zero_identity_root(
            &mut simplifier.context,
            expr
        ));
    }

    #[test]
    fn shifted_quotient_passthrough_cores_match_direct_small_zero_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(((a^3-b^3)/(a-b) - (a^2 + a*b + b^2)) + 1)/((x^2 + 2*x + 1 - (x+1)^2) + 1)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let (numerator, denominator) = match simplifier.context.get(expr).clone() {
            Expr::Div(numerator, denominator) => (numerator, denominator),
            _ => panic!("expected division root"),
        };
        let numerator_core =
            strip_positive_one_passthrough_root(&mut simplifier.context, numerator)
                .unwrap_or_else(|| panic!("expected numerator core"));
        let denominator_core =
            strip_positive_one_passthrough_root(&mut simplifier.context, denominator)
                .unwrap_or_else(|| panic!("expected denominator core"));

        assert!(matches_direct_small_zero_identity_root(
            &mut simplifier.context,
            numerator_core
        ));
        assert!(matches_direct_small_zero_identity_root(
            &mut simplifier.context,
            denominator_core
        ));
    }

    #[test]
    fn detects_direct_sqrt_perfect_square_abs_zero_identity_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "sqrt(a^2 + 2*a*b + b^2) - abs(a+b)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(matches_direct_sqrt_perfect_square_abs_zero_identity_root(
            &mut simplifier.context,
            expr
        ));
    }

    #[test]
    fn simplify_pipeline_handles_sqrt_perfect_square_against_trig_product_to_sum_sum_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(sqrt(a^2 + 2*a*b + b^2) - abs(a+b)) + (2*sin(x)*sin(y) - cos(x-y) + cos(x+y))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn partitioned_direct_small_zero_sum_shortcut_handles_sqrt_perfect_square_against_trig_product_to_sum_sum_regression(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(sqrt(a^2 + 2*a*b + b^2) - abs(a+b)) + (2*sin(x)*sin(y) - cos(x-y) + cos(x+y))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let result = super::try_standard_partitioned_direct_small_zero_sum_shortcut(
            &mut simplifier.context,
            expr,
            true,
        );
        assert!(
            result.is_some(),
            "expected direct small-zero identity shortcut"
        );
    }

    #[test]
    fn partitioned_direct_small_zero_sum_shortcut_handles_trig_binomial_square_against_telescoping_sum_regression(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((sin(x) + cos(x))^2 - (1 + sin(2*x))) + (1/(u*(u+1)) - 1/u + 1/(u+1))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let result = super::try_standard_partitioned_direct_small_zero_sum_shortcut(
            &mut simplifier.context,
            expr,
            true,
        );
        assert!(
            result.is_some(),
            "expected partitioned direct small-zero sum shortcut"
        );
    }

    #[test]
    fn direct_small_zero_identity_shortcut_handles_tan_cot_product_against_trig_product_to_sum_sum_regression(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(tan(x)*cot(x) - 1) + (2*sin(x)*sin(y) - cos(x-y) + cos(x+y))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let result = super::try_standard_direct_small_zero_identity_shortcut(
            &crate::phase::SimplifyOptions::default(),
            &mut simplifier.context,
            expr,
            true,
        );
        assert!(
            result.is_some(),
            "expected direct small-zero identity shortcut"
        );
    }

    #[test]
    fn direct_small_zero_additive_combination_shortcut_handles_trig_product_to_sum_against_odd_half_power_sum_regression(
    ) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(2*sin(x)*sin(y) - cos(x-y) + cos(x+y)) + (sqrt(x^5) - x^2*sqrt(x))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let result = super::try_standard_direct_small_zero_additive_combination_shortcut(
            &mut simplifier.context,
            expr,
            true,
        );
        assert!(
            result.is_some(),
            "expected direct small-zero additive combination shortcut"
        );
    }

    #[test]
    fn detects_tan_cot_plus_trig_product_to_sum_sum_structure_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(tan(x)*cot(x) - 1) + (2*sin(x)*sin(y) - cos(x-y) + cos(x+y))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let view = AddView::from_expr(&simplifier.context, expr);
        let rendered_terms: Vec<_> = view
            .terms
            .iter()
            .map(|(term, sign)| format!("{sign:?}:{}", render(&simplifier.context, *term)))
            .collect();
        let trig_chunk = super::build_signed_sum_expr_root(
            &mut simplifier.context,
            &[view.terms[0], view.terms[3], view.terms[4]],
        );
        let tan_chunk = super::build_signed_sum_expr_root(
            &mut simplifier.context,
            &[view.terms[1], view.terms[2]],
        );
        let tan_chunk_terms: Vec<_> = AddView::from_expr(&simplifier.context, tan_chunk)
            .terms
            .iter()
            .map(|(term, sign)| format!("{sign:?}:{}", render(&simplifier.context, *term)))
            .collect();
        assert!(
            super::matches_direct_small_zero_identity_root(&mut simplifier.context, trig_chunk),
            "trig_chunk={} terms={rendered_terms:?} rendered={}",
            render(&simplifier.context, trig_chunk),
            render(&simplifier.context, expr),
        );
        assert!(
            super::matches_direct_small_zero_identity_root(&mut simplifier.context, tan_chunk),
            "tan_chunk={} tan_terms={tan_chunk_terms:?} terms={rendered_terms:?} rendered={}",
            render(&simplifier.context, tan_chunk),
            render(&simplifier.context, expr),
        );
    }

    #[test]
    fn simplify_pipeline_handles_trig_binomial_square_against_telescoping_sum_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "((sin(x) + cos(x))^2 - (1 + sin(2*x))) + (1/(u*(u+1)) - 1/u + 1/(u+1))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn detects_direct_tan_cot_product_zero_identity_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse("tan(x)*cot(x) - 1", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(matches_direct_tan_cot_product_zero_identity_root(
            &mut simplifier.context,
            expr
        ));
        assert!(matches_direct_small_zero_identity_root(
            &mut simplifier.context,
            expr
        ));
    }

    #[test]
    fn detects_direct_tan_cot_sec_csc_zero_identity_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse("tan(x) + cot(x) - sec(x)*csc(x)", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(matches_direct_tan_cot_sec_csc_zero_identity_root(
            &mut simplifier.context,
            expr
        ));
        assert!(matches_direct_small_zero_identity_root(
            &mut simplifier.context,
            expr
        ));
    }

    #[test]
    fn detects_direct_sec_tan_pythagorean_zero_identity_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse("sec(x)^2 - tan(x)^2 - 1", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(matches_direct_sec_tan_pythagorean_zero_identity_root(
            &mut simplifier.context,
            expr
        ));
        assert!(matches_direct_small_zero_identity_root(
            &mut simplifier.context,
            expr
        ));
    }

    #[test]
    fn detects_direct_csc_cot_pythagorean_zero_identity_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse("csc(x)^2 - cot(x)^2 - 1", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(matches_direct_csc_cot_pythagorean_zero_identity_root(
            &mut simplifier.context,
            expr
        ));
        assert!(matches_direct_small_zero_identity_root(
            &mut simplifier.context,
            expr
        ));
    }

    #[test]
    fn simplify_pipeline_handles_csc_cot_pythagorean_zero_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse("csc(x)^2 - cot(x)^2 - 1", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn simplify_pipeline_handles_csc_cot_pythagorean_pair_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse("csc(x)^2 - cot(x)^2", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "1");
    }

    #[test]
    fn detects_direct_log_square_product_split_zero_identity_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "log((x*y)^2) - log(x^2) - log(y^2)",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(matches_direct_log_square_product_split_zero_identity_root(
            &mut simplifier.context,
            expr
        ));
    }

    #[test]
    fn detects_direct_ln_abs_product_split_zero_identity_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "2*ln(abs(x*y)) - 2*ln(abs(x)) - 2*ln(abs(y))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(matches_direct_ln_abs_product_split_zero_identity_root(
            &mut simplifier.context,
            expr
        ));
    }

    #[test]
    fn simplify_pipeline_handles_log_square_vs_ln_abs_difference_regression() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(log((x*y)^2) - log(x^2) - log(y^2)) - (2*ln(abs(x*y)) - 2*ln(abs(x)) - 2*ln(abs(y)))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }

    #[test]
    fn small_trig_zero_child_gate_matches_half_angle_sine_core() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse("sin(x)^2 - (1 - cos(2*x))/2", &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let options = SimplifyOptions::default();
        assert!(is_small_trig_or_hyperbolic_zero_child(
            &options,
            &mut simplifier.context,
            expr
        ));
    }

    #[test]
    fn small_trig_zero_child_gate_matches_binomial_square_core() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "(sin(x) + cos(x))^2 - (1 + sin(2*x))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let options = SimplifyOptions::default();
        assert!(is_small_trig_or_hyperbolic_zero_child(
            &options,
            &mut simplifier.context,
            expr
        ));
    }

    #[test]
    fn small_trig_zero_child_gate_matches_product_sum_core() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "2*cos(2*x)*sin(x) - (4*cos(x)^2*sin(x) - 2*sin(x))",
            &mut simplifier.context,
        )
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let options = SimplifyOptions::default();
        assert!(is_small_trig_or_hyperbolic_zero_child(
            &options,
            &mut simplifier.context,
            expr
        ));
    }
}
