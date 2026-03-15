use crate::define_rule;
use crate::rule::Rewrite;
use cas_math::pull_constant_from_fraction_support::PullConstantFromFractionKind;

fn format_pull_constant_from_fraction_desc(kind: PullConstantFromFractionKind) -> &'static str {
    match kind {
        PullConstantFromFractionKind::PullConstant => "Pull constant from numerator",
        PullConstantFromFractionKind::PullNegation => "Pull negation from numerator",
    }
}

// =============================================================================
// DivAddCommonFactorFromDenRule: Factor out MULTI-FACTOR from Add numerator
// when those factors appear in the denominator, enabling cancellation.
// =============================================================================
//
// V2: Multi-factor version - extracts the MAXIMUM common factor:
// - Intersects factor multisets across ALL Add terms
// - Caps each factor exponent by what's in the denominator
// - Only fires if the common factor is non-trivial
//
// Pattern: Div(Add(f*g*a, f*g*b), f*g*c) → Div(f*g*Add(a, b), f*g*c) → Add(a, b)/c
// =============================================================================
define_rule!(
    DivAddCommonFactorFromDenRule,
    "Factor Common Factors from Add in Div",
    importance: crate::step::ImportanceLevel::Medium,
    |ctx, expr| {
        let rewrite =
            cas_math::div_add_common_factor_from_den_support::try_rewrite_div_add_common_factor_from_den_expr(
                ctx, expr,
            )?;
        Some(Rewrite::new(rewrite.rewritten).desc("Factor common factors from Add in Div"))
    }
);

// =============================================================================
// DivDenFactorOutRule: Factor out common symbolic factors from Add denominator
// =============================================================================
//
// Pattern: Div(num, f*a + f*b + f*c) → Div(num, f*(a+b+c))
//
// This enables CancelCommonFactorsRule to then cancel f from num/den.
// Example: (x+2)³·eᵘ / (eᵘ·x³ + 6eᵘ·x² + 12eᵘ·x + 8eᵘ)
//       → (x+2)³·eᵘ / (eᵘ·(x³ + 6x² + 12x + 8))
//       → (x+2)³ / (x³ + 6x² + 12x + 8)   [via CancelCommonFactorsRule]
//       → 1                                  [via GCD cancel]
// =============================================================================
define_rule!(
    DivDenFactorOutRule,
    "Factor Out Common From Denominator",
    |ctx, expr| {
        let rewrite =
            cas_math::div_den_factor_out_support::try_rewrite_div_den_factor_out_expr(ctx, expr)?;
        Some(
            Rewrite::new(rewrite.rewritten).desc("Factor common symbolic factors from denominator"),
        )
    }
);

// =============================================================================
// DivExpandNumForCancelRule: Distribute Mul in Div numerator to enable factor
// cancellation when both sides share a common symbolic factor.
// =============================================================================
//
// Pattern: Div(Mul(X, Add(f*a, f*b)), Add(f*c, f*d))
//       → Div(Add(X*f*a, X*f*b), Add(f*c, f*d))
//
// This allows DivAddSymmetricFactorRule to then cancel f from both sides.
// Only fires when the expansion would actually expose common factors.
// =============================================================================
define_rule!(
    DivExpandNumForCancelRule,
    "Expand Numerator Product for Div Cancellation",
    |ctx, expr| {
        let rewrite = cas_math::div_expand_num_for_cancel_support::try_rewrite_div_expand_num_for_cancel_expr(
            ctx, expr,
        )?;
        Some(
            Rewrite::new(rewrite.rewritten)
                .desc("Expand numerator product to enable factor cancellation"),
        )
    }
);

// =============================================================================
// DivAddSymmetricFactorRule: Extract common factor from BOTH Add num AND Add den
// =============================================================================
//
// Pattern: Div(Add(f*a, f*b), Add(f*c, f*d)) → Div(Add(a, b), Add(c, d))
//
// This complements DivAddCommonFactorFromDenRule by handling cases where
// both numerator AND denominator are Adds that share a common factor.
// The factor cancels completely, simplifying the fraction.
// =============================================================================
define_rule!(
    DivAddSymmetricFactorRule,
    "Cancel Common Factor from Add/Add Fraction",
    importance: crate::step::ImportanceLevel::Medium,
    |ctx, expr| {
        let rewrite =
            cas_math::div_add_symmetric_factor_support::try_rewrite_div_add_symmetric_factor_expr(
                ctx, expr,
            )?;
        Some(Rewrite::new(rewrite.rewritten).desc("Cancel common factor from Add/Add fraction"))
    }
);

// Atomized rule for quotient of powers: a^n / a^m = a^(n-m)
// This is separated from CancelCommonFactorsRule for pedagogical clarity
fn format_quotient_of_powers_desc(
    kind: cas_math::quotient_of_powers_support::QuotientOfPowersRewriteKind,
) -> &'static str {
    match kind {
        cas_math::quotient_of_powers_support::QuotientOfPowersRewriteKind::EqualPowers => {
            "a^n / a^n = 1"
        }
        cas_math::quotient_of_powers_support::QuotientOfPowersRewriteKind::PowOverPow => {
            "a^n / a^m = a^(n-m)"
        }
        cas_math::quotient_of_powers_support::QuotientOfPowersRewriteKind::PowOverBase => {
            "a^n / a = a^(n-1)"
        }
        cas_math::quotient_of_powers_support::QuotientOfPowersRewriteKind::BaseOverPow => {
            "a / a^m = a^(1-m)"
        }
    }
}

define_rule!(
    QuotientOfPowersRule,
    "Quotient of Powers",
    solve_safety: crate::SolveSafety::NeedsCondition(
        crate::ConditionClass::Definability
    ),
    |ctx, expr, parent_ctx| {
        use crate::Predicate;

        let domain_mode = parent_ctx.domain_mode();
        let value_domain = parent_ctx.value_domain();
        let rewrite = cas_math::quotient_of_powers_support::try_rewrite_quotient_of_powers_expr_with(
            ctx,
            expr,
            |core_ctx, base| {
                let decision = crate::oracle_allows_with_hint(
                    core_ctx,
                    domain_mode,
                    value_domain,
                    &Predicate::NonZero(base),
                    "Quotient of Powers",
                );
                decision.allow
            },
        )?;

        Some(Rewrite::new(rewrite.rewritten).desc(format_quotient_of_powers_desc(rewrite.kind)))
    }
);

define_rule!(
    PullConstantFromFractionRule,
    "Pull Constant From Fraction",
    |ctx, expr| {
        let rewrite =
            cas_math::pull_constant_from_fraction_support::try_rewrite_pull_constant_from_fraction_expr(
                ctx, expr,
            )?;
        Some(
            Rewrite::new(rewrite.rewritten)
                .desc(format_pull_constant_from_fraction_desc(rewrite.kind)),
        )
    }
);

define_rule!(
    FactorBasedLCDRule,
    "Factor-Based LCD",
    Some(crate::target_kind::TargetKindSet::ADD),
    |ctx, expr| {
        let rewrite =
            cas_math::factor_based_lcd_support::try_rewrite_factor_based_lcd_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc("Combine fractions with factor-based LCD"))
    }
);

// =============================================================================
// DivExpandToCancelRule: Cancel fractions by polynomial comparison
// =============================================================================
//
// When structural factor-based cancellation fails (CancelCommonFactorsRule,
// DivAddSymmetricFactorRule, etc.), this rule converts both numerator and
// denominator to canonical polynomial form and checks equality.
//
// This catches cases like:
//   (a² + b²)(x² + y²) / (a²x² + a²y² + b²x² + b²y²)  → 1
//   (1 + x) / ((1 + x^(1/3))(1 - x^(1/3) + x^(2/3)))   → 1
//
// Strategy:
//   1. Try poly_eq (MultiPoly) — handles pure polynomial expressions (no context mutation)
//   2. Fall back to expand() on a CLONED context — handles fractional exponents
//
// Safety:
//   - Budget-guarded: only fires when total node count is small enough
//   - Only fires when the expression contains expandable products (Mul+Add)
//   - poly_eq is read-only; expand fallback uses a cloned context
// =============================================================================
define_rule!(
    DivExpandToCancelRule,
    "Expand to Cancel Fraction",
    |ctx, expr| {
        fn format_div_expand_cancel_desc(
            kind: cas_math::div_expand_cancel_support::DivExpandToCancelKind,
        ) -> &'static str {
            match kind {
                cas_math::div_expand_cancel_support::DivExpandToCancelKind::OpaqueSubstitution => {
                    "Polynomial division with opaque substitution"
                }
                cas_math::div_expand_cancel_support::DivExpandToCancelKind::ExpandedEquality => {
                    "Expanded numerator equals denominator"
                }
            }
        }

        let rewrite = cas_math::div_expand_cancel_support::try_rewrite_div_expand_to_cancel_expr_with_thread_guards(
            ctx,
            expr,
            |base_ctx, sub_frac| {
                let mut simplifier = crate::Simplifier::with_default_rules();
                simplifier.context = base_ctx.clone();
                let (simplified, _) = simplifier.simplify(sub_frac);
                Some((simplifier.context, simplified))
            },
            crate::expand::expand,
            |expanded_ctx, expanded_num, expanded_den| {
                let mut simplifier = crate::Simplifier::with_default_rules();
                simplifier.context = expanded_ctx;
                let (simplified_num, _) = simplifier.simplify(expanded_num);
                let (simplified_den, _) = simplifier.simplify(expanded_den);
                Some((simplifier.context, simplified_num, simplified_den))
            },
        )?;

        Some(Rewrite::new(rewrite.rewritten).desc(format_div_expand_cancel_desc(rewrite.kind)))
    }
);

define_rule!(
    ReciprocalDifferenceOfSquaresRule,
    "Reciprocal Difference of Squares",
    solve_safety: crate::SolveSafety::NeedsCondition(
        crate::ConditionClass::Definability
    ),
    |ctx, expr, parent_ctx| {
        use crate::Predicate;

        let (num, den) = cas_math::expr_destructure::as_div(ctx, expr)?;
        let plan = cas_math::difference_of_squares_support::try_plan_reciprocal_difference_of_squares_expr(
            ctx,
            num,
            den,
        )?;

        let decision = crate::oracle_allows_with_hint(
            ctx,
            parent_ctx.domain_mode(),
            parent_ctx.value_domain(),
            &Predicate::NonZero(den),
            "Reciprocal Difference of Squares",
        );
        if !decision.allow {
            return None;
        }

        Some(
            Rewrite::new(plan.final_result)
                .requires(crate::ImplicitCondition::NonZero(den))
                .desc("(A-B)/(A^2-B^2) = 1/(A+B)"),
        )
    }
);

#[cfg(test)]
mod tests {
    use crate::parent_context::ParentContext;
    use crate::rule::Rule;
    use crate::rules::algebra::ReciprocalDifferenceOfSquaresRule;
    use crate::DomainMode;
    use cas_ast::ordering::compare_expr;
    use cas_ast::Context;
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    #[test]
    fn div_expand_to_cancel_rule_rewrites_collapsed_root_quotient_before_rationalize() {
        let mut ctx = Context::new();
        let expr = parse(
            "(x^2 + 1 + 2*(x^2 + 1)^(1/2))/((x^2 + 1)^(1/2) + 2)",
            &mut ctx,
        )
        .unwrap_or_else(|err| panic!("parse expr: {err}"));
        let rewrite =
            cas_math::div_expand_cancel_support::try_rewrite_div_expand_to_cancel_expr_with_thread_guards(
                &mut ctx,
                expr,
                |base_ctx, sub_frac| {
                    let mut simplifier = crate::Simplifier::with_default_rules();
                    simplifier.context = base_ctx.clone();
                    let (simplified, _) = simplifier.simplify(sub_frac);
                    Some((simplifier.context, simplified))
                },
                crate::expand::expand,
                |expanded_ctx, expanded_num, expanded_den| {
                    let mut simplifier = crate::Simplifier::with_default_rules();
                    simplifier.context = expanded_ctx;
                    let (simplified_num, _) = simplifier.simplify(expanded_num);
                    let (simplified_den, _) = simplifier.simplify(expanded_den);
                    Some((simplifier.context, simplified_num, simplified_den))
                },
            )
            .unwrap_or_else(|| panic!("rewrite"));
        let rendered = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.rewritten
            }
        );
        assert!(
            rendered == "(x^2 + 1)^(1/2)" || rendered == "sqrt(x^2 + 1)",
            "unexpected collapsed-root quotient rewrite: {rendered}"
        );
    }

    #[test]
    fn reciprocal_difference_of_squares_rule_rewrites_arctan_in_generic() {
        let mut ctx = Context::new();
        let expr = parse("(arctan(u) - 1)/(arctan(u)^2 - 1)", &mut ctx)
            .unwrap_or_else(|err| panic!("parse expr: {err}"));
        let rule = ReciprocalDifferenceOfSquaresRule;
        let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
        let rewrite = rule
            .apply(&mut ctx, expr, &parent_ctx)
            .unwrap_or_else(|| panic!("rewrite"));
        let expected =
            parse("1/(arctan(u) + 1)", &mut ctx).unwrap_or_else(|err| panic!("expected: {err}"));
        assert_eq!(
            compare_expr(&ctx, rewrite.new_expr, expected),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn reciprocal_difference_of_squares_rule_rewrites_arcsin_in_generic() {
        let mut ctx = Context::new();
        let expr = parse("(arcsin(u) - 1)/(arcsin(u)^2 - 1)", &mut ctx)
            .unwrap_or_else(|err| panic!("parse expr: {err}"));
        let rule = ReciprocalDifferenceOfSquaresRule;
        let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
        let rewrite = rule
            .apply(&mut ctx, expr, &parent_ctx)
            .unwrap_or_else(|| panic!("rewrite"));
        let expected =
            parse("1/(arcsin(u) + 1)", &mut ctx).unwrap_or_else(|err| panic!("expected: {err}"));
        assert_eq!(
            compare_expr(&ctx, rewrite.new_expr, expected),
            std::cmp::Ordering::Equal
        );
    }
}
