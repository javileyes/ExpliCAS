//! Polynomial manipulation rules: distribution, annihilation, combining like terms,
//! expansion, and factoring.
//!
//! This module is split into submodules:
//! - `expansion`: Binomial/multinomial expansion, auto-expand, polynomial identity detection
//! - `factoring`: Heuristic common factor extraction

mod expansion;
mod expansion_normalize;
mod factoring;

pub use expansion::{
    AutoExpandPowSumRule, AutoExpandSubCancelRule, BinomialExpansionRule,
    SmallMultinomialExpansionRule,
};
pub use expansion_normalize::{
    ExpandSmallBinomialPowRule, HeuristicPolyNormalizeAddRule, PolynomialIdentityZeroRule,
    PolynomialProductNormalizeRule,
};
pub use factoring::{ExtractCommonMulFactorRule, HeuristicExtractCommonFactorAddRule};

use crate::define_rule;
use crate::phase::PhaseMask;
use crate::rule::Rewrite;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::annihilation_support::{
    should_rewrite_annihilation_to_zero_with_mode_flags, AnnihilationRewriteKind,
};
use cas_math::cube_identity_support::try_rewrite_sum_diff_cubes_product_expr;
use cas_math::distribution_division_support::try_rewrite_div_distribution_simplifying_expr;
use cas_math::distribution_rule_support::try_rewrite_mul_distribution_legacy_expr;
use cas_math::expr_destructure::as_mul;
use std::cmp::Ordering;

// ── Sum/Difference of Cubes Contraction Rule ────────────────────────────
//
// Pre-order rule: (X + c)·(X² - c·X + c²) → X³ + c³
//                 (X - c)·(X² + c·X + c²) → X³ - c³
//
// This fires BEFORE DistributeRule to prevent suboptimal splitting of the
// trinomial factor. Works for any base X (polynomial, transcendental, etc.)
define_rule!(
    SumDiffCubesContractionRule,
    "Sum/Difference of Cubes Contraction",
    None,
    PhaseMask::CORE,
    |ctx, expr| {
        let rewrite = try_rewrite_sum_diff_cubes_product_expr(ctx, expr)?;
        Some(
            Rewrite::new(rewrite.rewritten)
                .desc("Sum/Difference of cubes")
                .local(expr, rewrite.rewritten),
        )
    }
);

// ── Sqrt Perfect-Square Trinomial Rule ───────────────────────────────────
//
// sqrt(A² + 2·A·B + B²) → |A + B|
//
// Detects perfect-square trinomials inside sqrt and simplifies directly.
// Works for any sub-expressions A, B (polynomial, transcendental, etc.)
//
// Example: sqrt(sin²(u) + 2·sin(u) + 1) → |sin(u) + 1|
//
// We support two forms:
//   (a) A² + 2·A·c + c²  where c is a Number (most common from CSV)
//   (b) Fully symbolic: both A² and B² are Pow(_, 2) nodes

define_rule!(
    SqrtPerfectSquareRule,
    "Sqrt Perfect Square",
    None,
    PhaseMask::CORE,
    |ctx, expr| {
        let rewrite =
            cas_math::perfect_square_support::try_rewrite_sqrt_perfect_square_expr(ctx, expr)?;
        Some(
            Rewrite::new(rewrite.rewritten)
                .desc("sqrt(A^2 ± 2AB + B^2) = |A ± B|")
                .local(expr, rewrite.rewritten),
        )
    }
);

// DistributeRule: Runs in CORE, TRANSFORM, RATIONALIZE but NOT in POST
// This prevents Factor↔Distribute infinite loops (FactorCommonIntegerFromAdd runs in POST)
define_rule!(
    DistributeRule,
    "Distributive Property",
    None,
    // NO POST: evita ciclo con FactorCommonIntegerFromAdd (ver test_factor_distribute_no_loop)
    PhaseMask::CORE | PhaseMask::TRANSFORM | PhaseMask::RATIONALIZE,
    |ctx, expr, parent_ctx| {
        use crate::semantics::NormalFormGoal;

        // GATE: Don't distribute when goal is Collected or Factored
        // This prevents undoing the effect of collect() or factor() commands
        match parent_ctx.goal() {
            NormalFormGoal::Collected | NormalFormGoal::Factored => return None,
            _ => {}
        }

        // Don't distribute if expression is in canonical form (e.g., inside abs() or sqrt())
        // This protects patterns like abs((x-2)(x+2)) from expanding
        if crate::canonical_forms::is_canonical_form(ctx, expr) {
            return None;
        }

        // GUARD: Block distribution when sin(4x) identity pattern is detected
        // This allows Sin4xIdentityZeroRule to see 4*sin(t)*cos(t)*(cos²-sin²) as a single product
        if let Some(marks) = parent_ctx.pattern_marks() {
            if marks.has_sin4x_identity_pattern {
                return None;
            }
        }
        // Use zero-clone destructuring pattern
        let (l, r) = as_mul(ctx, expr)?;

        // GUARD: Skip distribution when a factor is 1.
        // 1*(a+b) -> 1*a + 1*b is a visual no-op (MulOne is applied in rendering),
        // and produces confusing "Before/After identical" steps.
        if cas_math::expr_predicates::is_one_expr(ctx, l)
            || cas_math::expr_predicates::is_one_expr(ctx, r)
        {
            return None;
        }

        if should_preserve_compact_power_product_with_factorable_integer_add(ctx, l, r, parent_ctx)
        {
            return None;
        }

        if should_preserve_compact_asinh_self_product(ctx, l, r) {
            return None;
        }

        if should_preserve_compact_hyperbolic_additive_product(ctx, l, r) {
            return None;
        }

        if should_preserve_compact_polynomial_function_integrate_target(ctx, expr, parent_ctx) {
            return None;
        }

        // Multiplicative distribution uses cas_math helper that preserves
        // historical guard ordering and semantics.
        let parent_mul_terms =
            parent_ctx
                .immediate_parent()
                .and_then(|parent_id| match ctx.get(parent_id) {
                    Expr::Mul(pl, pr) => Some((*pl, *pr)),
                    _ => None,
                });
        if let Some(rewrite) = try_rewrite_mul_distribution_legacy_expr(ctx, expr, parent_mul_terms)
        {
            return Some(
                Rewrite::new(rewrite.rewritten)
                    .desc("Distribute")
                    .local(expr, rewrite.rewritten),
            );
        }

        if let Some(rewrite) = try_rewrite_div_distribution_simplifying_expr(ctx, expr) {
            return Some(
                Rewrite::new(rewrite.rewritten)
                    .desc("Distribute division (simplifying)")
                    .local(expr, rewrite.rewritten),
            );
        }
        None
    }
);

fn should_preserve_compact_power_product_with_factorable_integer_add(
    ctx: &cas_ast::Context,
    left: cas_ast::ExprId,
    right: cas_ast::ExprId,
    parent_ctx: &crate::parent_context::ParentContext,
) -> bool {
    if parent_ctx.is_expand_mode() || parent_ctx.is_auto_expand() {
        return false;
    }

    (side_has_compact_low_degree_polynomial_power(ctx, left)
        && binary_add_has_variable_common_integer_factor(ctx, right))
        || (side_has_compact_low_degree_polynomial_power(ctx, right)
            && binary_add_has_variable_common_integer_factor(ctx, left))
}

fn should_preserve_compact_asinh_self_product(ctx: &Context, left: ExprId, right: ExprId) -> bool {
    asinh_arg(ctx, left).is_some_and(|arg| structurally_same(ctx, arg, right))
        || asinh_arg(ctx, right).is_some_and(|arg| structurally_same(ctx, arg, left))
}

fn should_preserve_compact_hyperbolic_additive_product(
    ctx: &Context,
    left: ExprId,
    right: ExprId,
) -> bool {
    let left_additive = matches!(ctx.get(left), Expr::Add(_, _) | Expr::Sub(_, _));
    let right_additive = matches!(ctx.get(right), Expr::Add(_, _) | Expr::Sub(_, _));

    (left_additive && expr_contains_hyperbolic_function(ctx, right))
        || (right_additive && expr_contains_hyperbolic_function(ctx, left))
}

fn integrate_variable_name(ctx: &Context, expr: ExprId) -> Option<String> {
    let Expr::Variable(sym_id) = ctx.get(expr) else {
        return None;
    };
    Some(ctx.sym_name(*sym_id).to_string())
}

fn should_preserve_compact_polynomial_function_integrate_target(
    ctx: &mut Context,
    expr: ExprId,
    parent_ctx: &crate::parent_context::ParentContext,
) -> bool {
    if parent_ctx.context_mode() != crate::options::ContextMode::IntegratePrep {
        return false;
    }

    let integrate_call = parent_ctx.all_ancestors().iter().find_map(|ancestor| {
        let Expr::Function(fn_id, args) = ctx.get(*ancestor) else {
            return None;
        };
        if ctx.sym_name(*fn_id) != "integrate" || args.len() != 2 {
            return None;
        }
        let var_name = integrate_variable_name(ctx, args[1])?;
        Some((args[0], var_name))
    });

    integrate_call.is_some_and(|(target, var_name)| {
        (target == expr
            && (cas_math::symbolic_integration_support::integrate_symbolic_is_polynomial_times_exp_linear_target(
                ctx, expr, &var_name,
            ) || cas_math::symbolic_integration_support::integrate_symbolic_is_polynomial_times_trig_linear_target(
                ctx, expr, &var_name,
            ) || cas_math::symbolic_integration_support::integrate_symbolic_is_quadratic_times_affine_ln_by_parts_target(
                ctx, expr, &var_name,
            ) || cas_math::symbolic_integration_support::integrate_symbolic_is_quadratic_times_positive_quadratic_ln_by_parts_target(
                ctx, expr, &var_name,
            )))
            || cas_math::symbolic_integration_support::integrate_symbolic_is_polynomial_log_reciprocal_derivative_target(
                ctx,
                target,
                &var_name,
            )
    })
}

fn expr_contains_hyperbolic_function(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Function(fn_id, args) => {
            matches!(ctx.sym_name(*fn_id), "sinh" | "cosh")
                || args
                    .iter()
                    .any(|arg| expr_contains_hyperbolic_function(ctx, *arg))
        }
        Expr::Add(left, right)
        | Expr::Sub(left, right)
        | Expr::Mul(left, right)
        | Expr::Div(left, right)
        | Expr::Pow(left, right) => {
            expr_contains_hyperbolic_function(ctx, *left)
                || expr_contains_hyperbolic_function(ctx, *right)
        }
        Expr::Neg(inner) | Expr::Hold(inner) => expr_contains_hyperbolic_function(ctx, *inner),
        Expr::Matrix { data, .. } => data
            .iter()
            .any(|item| expr_contains_hyperbolic_function(ctx, *item)),
        Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => false,
    }
}

fn asinh_arg(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() == 1 && ctx.builtin_of(*fn_id) == Some(BuiltinFn::Asinh) {
        Some(args[0])
    } else {
        None
    }
}

fn structurally_same(ctx: &Context, left: ExprId, right: ExprId) -> bool {
    cas_ast::ordering::compare_expr(ctx, left, right) == Ordering::Equal
}

fn side_has_compact_low_degree_polynomial_power(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> bool {
    is_compact_low_degree_polynomial_power(ctx, expr)
        || cas_math::expr_nary::mul_leaves(ctx, expr)
            .iter()
            .any(|&factor| is_compact_low_degree_polynomial_power(ctx, factor))
}

fn is_compact_low_degree_polynomial_power(ctx: &cas_ast::Context, expr: cas_ast::ExprId) -> bool {
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return false;
    };
    let Some(power) = cas_math::numeric::as_i64(ctx, *exp) else {
        return false;
    };
    if !(2..=8).contains(&power) || cas_ast::count_nodes(ctx, *base) > 25 {
        return false;
    }

    let variables = cas_ast::collect_variables(ctx, *base);
    let Some(var) = variables.iter().next() else {
        return false;
    };
    variables.len() == 1
        && matches!(
            cas_math::polynomial::Polynomial::from_expr(ctx, *base, var.as_str()),
            Ok(poly) if (1..=2).contains(&poly.degree())
        )
}

fn binary_add_has_variable_common_integer_factor(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> bool {
    let Expr::Add(left, right) = ctx.get(expr) else {
        return false;
    };
    if cas_ast::collect_variables(ctx, expr).is_empty() {
        return false;
    }

    let Some(left_coeff) = integer_coefficient_abs(ctx, *left) else {
        return false;
    };
    let Some(right_coeff) = integer_coefficient_abs(ctx, *right) else {
        return false;
    };
    gcd_abs(left_coeff, right_coeff) > 1
}

fn integer_coefficient_abs(ctx: &cas_ast::Context, expr: cas_ast::ExprId) -> Option<i64> {
    match ctx.get(expr) {
        Expr::Number(_) => cas_math::numeric::as_i64(ctx, expr)?.checked_abs(),
        Expr::Mul(left, right) => cas_math::numeric::as_i64(ctx, *left)
            .or_else(|| cas_math::numeric::as_i64(ctx, *right))?
            .checked_abs(),
        Expr::Neg(inner) => integer_coefficient_abs(ctx, *inner),
        _ => None,
    }
}

fn gcd_abs(mut left: i64, mut right: i64) -> i64 {
    while right != 0 {
        let next = left % right;
        left = right;
        right = next;
    }
    left.abs()
}

// AnnihilationRule: Detects and cancels terms like x - x or __hold(sum) - sum
// Domain Mode Policy: Like AddInverseRule, we must respect domain_mode
// because if `x` can be undefined (e.g., a/(a-1) when a=1), then x - x
// is undefined, not 0.
// - Strict: only if no term contains potentially-undefined subexpressions
// - Assume: always apply (educational mode assumption: all expressions are defined)
// - Generic: same as Assume
define_rule!(AnnihilationRule, "Annihilation", |ctx, expr, parent_ctx| {
    if let Some(kind) = should_rewrite_annihilation_to_zero_with_mode_flags(
        ctx,
        expr,
        matches!(parent_ctx.domain_mode(), crate::DomainMode::Assume),
        matches!(parent_ctx.domain_mode(), crate::DomainMode::Strict),
        crate::collect::has_undefined_risk,
    ) {
        let zero = ctx.num(0);
        let desc = match kind {
            AnnihilationRewriteKind::TwoTerm => "x - x = 0",
            AnnihilationRewriteKind::HoldSum => "__hold(sum) - sum = 0",
        };
        return Some(Rewrite::new(zero).desc(desc));
    }
    None
});

// CombineLikeTermsRule: Collects like terms in Add/Mul expressions
// Now uses collect_with_semantics for domain_mode awareness:
// - Strict: refuses to cancel terms with undefined risk (e.g., x/(x+1) - x/(x+1))
// - Assume: cancels with domain_assumption warning
// - Generic: cancels unconditionally
define_rule!(
    CombineLikeTermsRule,
    "Combine Like Terms",
    |ctx, expr, parent_ctx| {
        // Only try to collect if it's an Add or Mul
        if matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Mul(_, _)) {
            let plan = crate::collect::plan_collect_rule_rewrite(ctx, expr, parent_ctx)?;
            let mut rewrite = Rewrite::new(plan.new_expr).desc(plan.description);
            if let (Some(before), Some(after)) = (plan.local_before, plan.local_after) {
                rewrite = rewrite.local(before, after);
            }
            Some(rewrite)
        } else {
            None
        }
    }
);

/// BinomialExpansionRule: (a + b)^n → expanded polynomial
/// ONLY expands true binomials (exactly 2 terms).
/// Multinomial expansion (>2 terms) is NOT done by default to avoid explosion.
/// Use explicit expand() mode for multinomial expansion.
/// Implements Rule directly to access ParentContext
pub fn register(simplifier: &mut crate::Simplifier) {
    // Register cube identity contraction BEFORE distribution to prevent suboptimal splits
    simplifier.add_rule(Box::new(SumDiffCubesContractionRule));
    // Sqrt perfect-square trinomial: sqrt(A²+2AB+B²) → |A+B|
    simplifier.add_rule(Box::new(SqrtPerfectSquareRule));
    simplifier.add_rule(Box::new(DistributeRule));
    simplifier.add_rule(Box::new(AnnihilationRule));
    simplifier.add_rule(Box::new(CombineLikeTermsRule));
    simplifier.add_rule(Box::new(PolynomialProductNormalizeRule));
    simplifier.add_rule(Box::new(BinomialExpansionRule));
    simplifier.add_rule(Box::new(SmallMultinomialExpansionRule));
    // V2.15.8: ExpandSmallBinomialPowRule - controlled by autoexpand_binomials flag
    // Enable via REPL: set autoexpand_binomials on
    simplifier.add_rule(Box::new(ExpandSmallBinomialPowRule));
    simplifier.add_rule(Box::new(AutoExpandPowSumRule));
    simplifier.add_rule(Box::new(AutoExpandSubCancelRule));
    simplifier.add_rule(Box::new(PolynomialIdentityZeroRule));
    // V2.15.8: HeuristicPolyNormalizeAddRule - poly-normalize Add/Sub in Heuristic mode
    // V2.15.9: HeuristicExtractCommonFactorAddRule - extract common factors first (priority 110)
    simplifier.add_rule(Box::new(HeuristicExtractCommonFactorAddRule));
    // V2.16: ExtractCommonMulFactorRule - extract common multiplicative factors from n-ary sums
    // Fixes cross-product NF divergence in metamorphic Mul tests (priority 108)
    simplifier.add_rule(Box::new(ExtractCommonMulFactorRule));
    simplifier.add_rule(Box::new(HeuristicPolyNormalizeAddRule));
}

#[cfg(test)]
mod tests;
