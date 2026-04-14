use crate::best_so_far::{BestSoFar, BestSoFarBudget};
use crate::expand::eager_eval_expand_calls;
use crate::phase::{SimplifyOptions, SimplifyPhase};
use crate::rule::Rule;
use crate::{Simplifier, Step};
use cas_ast::ordering::compare_expr;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_formatter::render_expr;
use cas_math::arithmetic_rule_support::try_rewrite_combine_constants_expr;
use cas_math::build::mul2_raw;
use cas_math::expansion_rule_support::{try_expand_small_pow_sum_expr, SmallPowExpandPolicy};
use cas_math::expr_extract::{extract_exp_argument, extract_i64_integer};
use cas_math::expr_nary::{build_balanced_mul, AddView, MulView, Sign};
use cas_math::expr_rewrite::smart_mul;
use cas_math::factoring_support::try_rewrite_automatic_factor_expr;
use cas_math::fraction_power_cancel_support::try_rewrite_cancel_same_base_powers_div_expr;
use cas_math::hyperbolic_identity_support::try_rewrite_tanh_double_angle_expansion;
use cas_math::infinity_support::{is_negative_literal, is_positive_literal};
use cas_math::logarithm_inverse_support::{
    log_exp_inverse_policy_mode_from_flags, plan_log_power_base_numeric_policy,
    try_rewrite_exponential_log_inverse_expr, try_rewrite_log_power_base_numeric_expr,
};
use cas_math::poly_lowering;
use cas_math::poly_store::clear_thread_local_store;
use cas_math::root_forms::{
    try_rewrite_canonical_root_expr, try_rewrite_extract_perfect_power_from_radicand_expr,
    try_rewrite_simplify_square_root_expr, SimplifySquareRootRewriteKind,
};
use cas_math::trig_canonicalization_support::{
    try_rewrite_csc_cot_pythagorean_identity_expr, try_rewrite_sec_tan_pythagorean_identity_expr,
};
use cas_math::trig_core_identity_support::try_rewrite_pythagorean_identity_add_expr;
use cas_math::trig_identity_zero_support::try_rewrite_sin_sum_triple_identity_zero_expr;
use cas_math::trig_linear_support::{
    build_coef_times_base, extract_coef_and_base, extract_linear_coefficients,
};
use cas_math::trig_power_identity_support::{
    extract_coeff_trig_pow2, try_rewrite_pythagorean_chain_add_expr,
    try_rewrite_pythagorean_factor_form_add_expr,
    try_rewrite_pythagorean_generic_coefficient_add_expr,
    try_rewrite_reciprocal_product_pythagorean_zero_add_expr,
    try_rewrite_trig_fourth_power_difference_add_expr,
};
use cas_math::trig_roots_flatten::extract_double_angle_arg_relaxed;
use cas_math::trig_roots_flatten::flatten_mul_chain;
use cas_math::trig_sum_product_support::try_rewrite_product_to_sum_expr;
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
    matches_direct_small_zero_identity_root(ctx, lhs)
        && matches_direct_small_zero_identity_root(ctx, rhs)
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

fn additive_scope_has_numeric_term_root(ctx: &mut Context, expr: ExprId) -> bool {
    AddView::from_expr(ctx, expr)
        .terms
        .iter()
        .any(|(term, _)| matches!(ctx.get(*term), Expr::Number(_)))
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
    let lhs_rewrite = try_rewrite_product_to_sum_expr(ctx, lhs_core);
    if lhs_rewrite.is_some_and(|rewrite| {
        rewrite.kind == cas_math::trig_sum_product_support::TrigProductToSumRewriteKind::SinCos
            && render_expr(ctx, rewrite.rewritten) == render_expr(ctx, rhs_core)
    }) {
        return true;
    }

    let rhs_rewrite = try_rewrite_product_to_sum_expr(ctx, rhs_core);
    rhs_rewrite.is_some_and(|rewrite| {
        rewrite.kind == cas_math::trig_sum_product_support::TrigProductToSumRewriteKind::SinCos
            && render_expr(ctx, rewrite.rewritten) == render_expr(ctx, lhs_core)
    })
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

fn matches_direct_positive_double_cos_square_diff_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (square_diff, positive_cos) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(square_arg) = extract_mixed_sign_trig_square_difference_arg_root(ctx, square_diff)
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
        || matches_direct_positive_double_cos_square_diff_pair_root(ctx, lhs_core, rhs_core)
}

fn matches_direct_trig_cubic_cosine_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    let lhs_minus_rhs = ctx.add(Expr::Sub(lhs_core, rhs_core));
    if matches_direct_trig_cubic_cosine_zero_identity_root(ctx, lhs_minus_rhs) {
        return true;
    }

    let rhs_minus_lhs = ctx.add(Expr::Sub(rhs_core, lhs_core));
    matches_direct_trig_cubic_cosine_zero_identity_root(ctx, rhs_minus_lhs)
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

        for (first_a, first_b) in [(p, q), (q, p)] {
            for (second_a, second_b) in [(r, s), (s, r)] {
                let first_product = smart_mul(ctx, first_a, second_a);
                let second_product = smart_mul(ctx, first_b, second_b);
                let first_sum = ctx.add(Expr::Add(first_product, second_product));
                let third_product = smart_mul(ctx, first_a, second_b);
                let fourth_product = smart_mul(ctx, first_b, second_a);
                let second_diff = ctx.add(Expr::Sub(third_product, fourth_product));
                let two = ctx.num(2);
                let first_square = ctx.add(Expr::Pow(first_sum, two));
                let second_square = ctx.add(Expr::Pow(second_diff, two));
                let expected = ctx.add(Expr::Add(first_square, second_square));
                if compare_expr(ctx, square_sum_expr, expected) == Ordering::Equal {
                    return true;
                }
            }
        }
    }

    false
}

fn matches_known_direct_pair_root(ctx: &mut Context, lhs_core: ExprId, rhs_core: ExprId) -> bool {
    matches_direct_trig_product_to_sum_sin_sin_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_trig_product_to_sum_sin_cos_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_trig_product_to_sum_cos_cos_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_nested_fraction_simplified_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_hyperbolic_sinh_sum_to_product_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_hyperbolic_cosh_sum_to_product_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_hyperbolic_cosh_difference_to_product_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_recursive_hyperbolic_sinh_sum_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_recursive_hyperbolic_cosh_sum_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_trig_mixed_double_angle_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_trig_cubic_cosine_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_cos_square_diff_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_angle_sum_diff_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_small_pow_expansion_pair_root(ctx, lhs_core, rhs_core)
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

    let (lhs_fn, lhs_arg) = extract_plain_sin_or_cos_arg_root(ctx, factors[0])?;
    let (rhs_fn, rhs_arg) = extract_plain_sin_or_cos_arg_root(ctx, factors[1])?;
    (lhs_fn == trig_fn && rhs_fn == trig_fn).then_some((lhs_arg, rhs_arg))
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
        let Some((BuiltinFn::Cos, angle_arg)) = extract_plain_sin_or_cos_arg_root(ctx, angle_expr)
        else {
            continue;
        };

        let view = AddView::from_expr(ctx, product_expr);
        if view.terms.len() != 2 {
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

fn is_square_power_root(ctx: &Context, expr: ExprId) -> bool {
    let Expr::Pow(_, exponent) = ctx.get(expr) else {
        return false;
    };
    let Expr::Number(n) = ctx.get(*exponent) else {
        return false;
    };
    *n == BigRational::from_integer(2.into())
}

fn extract_plain_cube_base_root(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    if matches!(ctx.get(expr), Expr::Number(n) if n.is_one()) {
        return Some(expr);
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

fn extract_square_root_or_unit_root(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    if is_square_power_root(ctx, expr) {
        let Expr::Pow(base, _) = ctx.get(expr) else {
            unreachable!();
        };
        return Some(*base);
    }

    extract_i64_integer(ctx, expr)
        .is_some_and(|value| value == 1)
        .then(|| ctx.num(1))
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

    for i in 0..terms.len() {
        for j in (i + 1)..terms.len() {
            let k = 3 - i - j;
            let (term_i, sign_i) = terms[i];
            let (term_j, sign_j) = terms[j];
            let (term_k, sign_k) = terms[k];

            if sign_i != Sign::Pos || sign_j != Sign::Pos {
                continue;
            }

            let Some(a) = extract_square_root_or_unit_root(ctx, term_i) else {
                continue;
            };
            let Some(b) = extract_square_root_or_unit_root(ctx, term_j) else {
                continue;
            };

            let (coeff, base) = extract_coef_and_base(ctx, term_k);
            let signed_coeff = if sign_k == Sign::Pos {
                coeff.clone()
            } else {
                -coeff.clone()
            };
            if signed_coeff != BigRational::from_integer(2.into())
                && signed_coeff != BigRational::from_integer((-2).into())
            {
                continue;
            }

            let expected_cross_base = smart_mul(ctx, a, b);
            if compare_expr(ctx, base, expected_cross_base) != Ordering::Equal {
                continue;
            }

            let two = ctx.num(2);
            let binomial = if signed_coeff.is_positive() {
                ctx.add(Expr::Add(a, b))
            } else {
                ctx.add(Expr::Sub(a, b))
            };
            return Some(ctx.add(Expr::Pow(binomial, two)));
        }
    }

    None
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
        let (product_expr, product_sign) = view.terms[product_index];
        let (other_expr, other_sign) = view.terms[1 - product_index];
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

            let (remaining_expr, remaining_sign) = remaining_terms[0];
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

            let (remaining_expr, remaining_sign) = remaining_terms[0];
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
    matches_direct_half_angle_square_zero_identity_root(ctx, expr)
        || matches_direct_trig_binomial_square_zero_identity_root(ctx, expr)
        || matches_direct_tan_cot_product_zero_identity_root(ctx, expr)
        || matches_direct_tan_cot_sec_csc_zero_identity_root(ctx, expr)
        || matches_direct_sec_tan_pythagorean_zero_identity_root(ctx, expr)
        || matches_direct_csc_cot_pythagorean_zero_identity_root(ctx, expr)
        || matches_direct_trig_product_to_sum_sin_sin_zero_identity_root(ctx, expr)
        || matches_direct_trig_product_to_sum_sin_cos_zero_identity_root(ctx, expr)
        || matches_direct_trig_product_to_sum_cos_cos_zero_identity_root(ctx, expr)
        || matches_direct_nested_fraction_simplified_zero_identity_root(ctx, expr)
        || matches_direct_trig_cubic_cosine_zero_identity_root(ctx, expr)
        || matches_direct_sum_diff_cubes_quotient_zero_identity_root(ctx, expr)
        || matches_direct_sqrt_perfect_square_abs_zero_identity_root(ctx, expr)
        || matches_direct_perfect_square_trinomial_zero_identity_root(ctx, expr)
        || matches_direct_log_square_product_split_zero_identity_root(ctx, expr)
        || matches_direct_ln_abs_product_split_zero_identity_root(ctx, expr)
        || matches_direct_small_polynomial_zero_identity_root(ctx, expr)
        || matches_direct_consecutive_telescoping_fraction_zero_identity_root(ctx, expr)
        || matches_direct_symbolic_trig_sum_to_product_zero_identity_root(ctx, expr)
        || matches_direct_general_phase_shift_zero_identity_root(ctx, expr)
        || matches_direct_hyperbolic_exp_sum_zero_identity_root(ctx, expr)
        || matches_direct_hyperbolic_pythagorean_zero_identity_root(ctx, expr)
        || matches_direct_exp_hyperbolic_double_identity_root(ctx, expr)
}

fn build_mul_expr_from_factors_root(ctx: &mut Context, factors: &[ExprId]) -> ExprId {
    match factors {
        [] => ctx.num(1),
        [single] => *single,
        _ => build_balanced_mul(ctx, factors),
    }
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

    if is_same_denominator_difference_root(ctx, expr) {
        if let Some((_den, lhs_core, rhs_core)) =
            extract_same_denominator_direct_pair_root(ctx, expr)
        {
            if matches_direct_trig_product_to_sum_sin_sin_pair_root(ctx, lhs_core, rhs_core)
                || matches_direct_trig_product_to_sum_sin_cos_pair_root(ctx, lhs_core, rhs_core)
                || matches_direct_trig_product_to_sum_cos_cos_pair_root(ctx, lhs_core, rhs_core)
                || matches_direct_nested_fraction_simplified_pair_root(ctx, lhs_core, rhs_core)
                || matches_direct_hyperbolic_sinh_sum_to_product_pair_root(ctx, lhs_core, rhs_core)
                || matches_direct_hyperbolic_cosh_sum_to_product_pair_root(ctx, lhs_core, rhs_core)
                || matches_direct_hyperbolic_cosh_difference_to_product_pair_root(
                    ctx, lhs_core, rhs_core,
                )
                || matches_direct_recursive_hyperbolic_sinh_sum_pair_root(ctx, lhs_core, rhs_core)
                || matches_direct_recursive_hyperbolic_cosh_sum_pair_root(ctx, lhs_core, rhs_core)
                || matches_direct_cos_square_diff_pair_root(ctx, lhs_core, rhs_core)
                || matches_direct_angle_sum_diff_pair_root(ctx, lhs_core, rhs_core)
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

    if try_standard_exact_zero_equivalence_shortcut(options, ctx, child, false).is_some() {
        return true;
    }

    let term_count = AddView::from_expr(ctx, child).terms.len();
    if term_count > 4 || !expr_contains_trig_or_hyperbolic_builtin_local(ctx, child) {
        return false;
    }

    isolated_simplify_rewrites_to_zero(options, ctx, child)
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

fn try_standard_guarded_small_zero_pair_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    if !is_guarded_small_zero_composition_candidate_root(ctx, expr) {
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
                && child_isolated_exact_zero(options, ctx, lhs)
            {
                Some(rhs)
            } else if expr_contains_trig_or_hyperbolic_builtin_local(ctx, rhs)
                && is_supported_nested_zero_child_partner(ctx, lhs)
                && child_isolated_exact_zero(options, ctx, rhs)
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
                && child_isolated_exact_zero(options, ctx, rhs)
            {
                Some(lhs)
            } else if expr_contains_trig_or_hyperbolic_builtin_local(ctx, lhs)
                && is_supported_nested_zero_child_partner(ctx, rhs)
                && child_isolated_exact_zero(options, ctx, lhs)
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
        && child_isolated_exact_zero(options, ctx, lhs)
    {
        Some(lhs)
    } else if expr_contains_trig_or_hyperbolic_builtin_local(ctx, rhs)
        && is_supported_nested_zero_child_partner(ctx, lhs)
        && child_isolated_exact_zero(options, ctx, rhs)
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
    if !(2..=5).contains(&terms.len()) {
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
    if !(3..=5).contains(&terms.len()) {
        return false;
    }

    build_small_two_chunk_additive_partitions_root(ctx, &terms)
        .into_iter()
        .any(|(lhs_chunk, rhs_chunk)| {
            matches_direct_small_zero_identity_root(ctx, lhs_chunk)
                && matches_direct_small_zero_identity_root(ctx, rhs_chunk)
        })
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

fn passthrough_direct_pair_rule_name_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> Option<&'static str> {
    if matches_direct_trig_product_to_sum_sin_sin_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_trig_product_to_sum_sin_cos_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_trig_product_to_sum_cos_cos_pair_root(ctx, lhs_core, rhs_core)
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
        || matches_direct_cos_square_diff_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_small_pow_expansion_pair_root(ctx, lhs_core, rhs_core)
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
        ) || matches_direct_trig_mixed_double_angle_pair_root(
            ctx,
            numerator_core,
            denominator_core,
        ) || matches_direct_trig_cubic_cosine_pair_root(ctx, numerator_core, denominator_core)
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
            if matches_direct_trig_product_to_sum_sin_sin_pair_root(
                ctx,
                numerator_residual,
                denominator_residual,
            ) || matches_direct_trig_product_to_sum_sin_cos_pair_root(
                ctx,
                numerator_residual,
                denominator_residual,
            ) || matches_direct_trig_product_to_sum_cos_cos_pair_root(
                ctx,
                numerator_residual,
                denominator_residual,
            ) || matches_direct_nested_fraction_simplified_pair_root(
                ctx,
                numerator_residual,
                denominator_residual,
            ) || matches_direct_hyperbolic_sinh_sum_to_product_pair_root(
                ctx,
                numerator_residual,
                denominator_residual,
            ) || matches_direct_hyperbolic_cosh_sum_to_product_pair_root(
                ctx,
                numerator_residual,
                denominator_residual,
            ) || matches_direct_hyperbolic_cosh_difference_to_product_pair_root(
                ctx,
                numerator_residual,
                denominator_residual,
            ) || matches_direct_recursive_hyperbolic_sinh_sum_pair_root(
                ctx,
                numerator_residual,
                denominator_residual,
            ) || matches_direct_recursive_hyperbolic_cosh_sum_pair_root(
                ctx,
                numerator_residual,
                denominator_residual,
            ) || matches_direct_trig_mixed_double_angle_pair_root(
                ctx,
                numerator_residual,
                denominator_residual,
            ) || matches_direct_trig_cubic_cosine_pair_root(
                ctx,
                numerator_residual,
                denominator_residual,
            ) || matches_direct_cos_square_diff_pair_root(
                ctx,
                numerator_residual,
                denominator_residual,
            ) || matches_direct_angle_sum_diff_pair_root(
                ctx,
                numerator_residual,
                denominator_residual,
            ) {
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
                                || (expr_contains_trig_or_hyperbolic_builtin_local(
                                &simplifier.context,
                                numerator_core,
                            ) && expr_contains_trig_or_hyperbolic_builtin_local(
                                &simplifier.context,
                                denominator_core,
                            ) && !(matches_direct_small_zero_identity_root(
                                &mut simplifier.context,
                                numerator_core,
                            ) && matches_direct_small_zero_identity_root(
                                &mut simplifier.context,
                                denominator_core,
                            )) && additive_scope_has_numeric_term_root(
                                &mut simplifier.context,
                                numerator_core,
                            ) && additive_scope_has_numeric_term_root(
                                &mut simplifier.context,
                                denominator_core,
                            ))
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
            "expected partitioned direct small-zero sum shortcut"
        );
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
