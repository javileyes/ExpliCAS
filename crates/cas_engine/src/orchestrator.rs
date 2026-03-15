use crate::best_so_far::{BestSoFar, BestSoFarBudget};
use crate::expand::eager_eval_expand_calls;
use crate::phase::{SimplifyOptions, SimplifyPhase};
use crate::{Simplifier, Step};
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::arithmetic_rule_support::try_rewrite_combine_constants_expr;
use cas_math::build::mul2_raw;
use cas_math::expr_nary::{AddView, MulView, Sign};
use cas_math::fraction_power_cancel_support::try_rewrite_cancel_same_base_powers_div_expr;
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
use cas_math::trig_linear_support::extract_coef_and_base;
use cas_math::trig_power_identity_support::try_rewrite_pythagorean_chain_add_expr;
use cas_solver_core::rationalize_policy::AutoRationalizeLevel;
use num_rational::BigRational;
use num_traits::Zero;
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
    let result = rewrite.new_expr;
    let mut shortcut_steps = Vec::new();
    if collect_steps {
        let mut step = Step::new_compact(&rewrite.description, rule_name, before, result);
        step.global_before = Some(before);
        step.global_after = Some(result);
        step.importance = crate::step::ImportanceLevel::High;
        shortcut_steps.push(step);
    }
    (result, shortcut_steps)
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

fn try_standard_sub_self_cancel_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let parent_ctx = build_root_shortcut_parent_ctx(options, ctx, expr);
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
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
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
        shortcut_steps.push(cancel_step);
    }

    Some((result, shortcut_steps))
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

        if self.options.shared.context_mode == crate::options::ContextMode::Standard
            && !simplifier.has_step_listener()
        {
            let mut shortcut_steps = Vec::new();
            let add_root = matches!(simplifier.context.get(expr), Expr::Add(_, _));
            let sub_root = matches!(simplifier.context.get(expr), Expr::Sub(_, _));
            let pow_root = matches!(simplifier.context.get(expr), Expr::Pow(_, _));
            if add_root || sub_root {
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
                if let Some(rewrite) =
                    try_rewrite_pythagorean_chain_add_expr(&mut simplifier.context, expr)
                {
                    if collect_steps {
                        let mut step = Step::new_compact(
                            &rewrite.desc,
                            "Pythagorean Chain Identity",
                            expr,
                            rewrite.rewritten,
                        );
                        step.global_before = Some(expr);
                        step.global_after = Some(rewrite.rewritten);
                        step.importance = crate::step::ImportanceLevel::High;
                        shortcut_steps.push(step);
                    }
                    return (
                        rewrite.rewritten,
                        shortcut_steps,
                        crate::phase::PipelineStats::default(),
                    );
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
                if let Some(result) = crate::rules::algebra::try_difference_of_squares_preorder(
                    &mut simplifier.context,
                    expr,
                    num,
                    den,
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

        if !collect_steps && is_solve_mode && !simplifier.has_step_listener() {
            let domain_is_strict = self.options.shared.semantics.domain_mode.is_strict();
            let allow_scalar_root = allow_hidden_solve_root_scalar_multiple_shortcut(&self.options);
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
        if let Some(result) = crate::try_dirichlet_kernel_identity_pub(&simplifier.context, current)
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
