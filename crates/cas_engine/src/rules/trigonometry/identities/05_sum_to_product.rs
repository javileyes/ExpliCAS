// =============================================================================
// DyadicCosProductToSinRule: 2^n · ∏_{k=0}^{n-1} cos(2^k·θ) → sin(2^n·θ)/sin(θ)
// =============================================================================
//
// This identity simplifies products like:
// - 2·cos(θ) → sin(2θ)/sin(θ)
// - 4·cos(θ)·cos(2θ) → sin(4θ)/sin(θ)
// - 8·cos(θ)·cos(2θ)·cos(4θ) → sin(8θ)/sin(θ)
//
// Combined with sin supplementary angle (sin(π-x)=sin(x)) and cancellation,
// this solves problems like: 8·cos(π/9)·cos(2π/9)·cos(4π/9) = 1.
//
// Domain: Requires sin(θ) ≠ 0. In Generic mode, this is only allowed when
// θ is a rational multiple of π that is NOT an integer (proven case).

/// DyadicCosProductToSinRule: Recognizes products 2^n · ∏cos(2^k·θ)
pub struct DyadicCosProductToSinRule;

impl crate::rule::Rule for DyadicCosProductToSinRule {
    fn name(&self) -> &str {
        "Dyadic Cos Product"
    }

    fn priority(&self) -> i32 {
        // Run BEFORE DoubleAngleRule and RecursiveTrigExpansionRule
        95
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<crate::rule::Rewrite> {
        use crate::helpers::{as_number, flatten_mul, is_provably_sin_nonzero};
        use crate::rule::Rewrite;
        use num_bigint::BigInt;
        use num_rational::BigRational;

        // Only match Mul expressions
        if !matches!(ctx.get(expr), Expr::Mul(_, _)) {
            return None;
        }

        // Flatten the multiplication
        let mut factors = Vec::new();
        flatten_mul(ctx, expr, &mut factors);

        // Separate numeric coefficient from cos factors
        let mut numeric_coeff = BigRational::one();
        let mut cos_args: Vec<ExprId> = Vec::new();
        let mut other_factors: Vec<ExprId> = Vec::new();

        for &factor in &factors {
            if let Some(n) = as_number(ctx, factor) {
                numeric_coeff *= n.clone();
            } else if let Expr::Function(name, args) = ctx.get(factor) {
                if name == "cos" && args.len() == 1 {
                    cos_args.push(args[0]);
                } else {
                    other_factors.push(factor);
                }
            } else {
                other_factors.push(factor);
            }
        }

        // Must have no other factors and at least 1 cos
        if !other_factors.is_empty() || cos_args.is_empty() {
            return None;
        }

        let n = cos_args.len() as u32;

        // Numeric coefficient must be exactly 2^n
        let expected_coeff = BigRational::from_integer(BigInt::from(1u64 << n));
        if numeric_coeff != expected_coeff {
            return None;
        }

        // Find θ by trying each cos_arg as base and verifying dyadic sequence
        let mut theta: Option<ExprId> = None;

        for candidate in &cos_args {
            if verify_dyadic_sequence(ctx, *candidate, &cos_args) {
                theta = Some(*candidate);
                break;
            }
        }

        let theta = theta?;

        // Domain check: sin(θ) ≠ 0
        let domain_mode = parent_ctx.domain_mode();

        if !is_provably_sin_nonzero(ctx, theta) {
            match domain_mode {
                crate::domain::DomainMode::Generic | crate::domain::DomainMode::Strict => {
                    // Block with hint
                    let sin_theta = ctx.add(Expr::Function("sin".to_string(), vec![theta]));
                    crate::domain::register_blocked_hint(crate::domain::BlockedHint {
                        rule: "Dyadic Cos Product".to_string(),
                        expr_id: sin_theta,
                        key: crate::assumptions::AssumptionKey::nonzero_key(ctx, sin_theta),
                        suggestion: "use `domain assume` to allow this transformation",
                    });
                    return None;
                }
                crate::domain::DomainMode::Assume => {
                    // Allow but will record assumption in result
                }
            }
        }

        // Build result: sin(2^n · θ) / sin(θ)
        let two_pow_n = ctx.num((1u64 << n) as i64);
        let scaled_theta = smart_mul(ctx, two_pow_n, theta);
        let sin_scaled = ctx.add(Expr::Function("sin".to_string(), vec![scaled_theta]));
        let sin_theta = ctx.add(Expr::Function("sin".to_string(), vec![theta]));
        let result = ctx.add(Expr::Div(sin_scaled, sin_theta));

        // Build description
        let desc = format!("2^{n}·∏cos(2^k·θ) = sin(2^{n}·θ)/sin(θ)", n = n);

        let mut rewrite = Rewrite::new(result).desc(&desc).local(expr, result);

        // Add assumption if in Assume mode and sin(θ)≠0 not proven
        if domain_mode == crate::domain::DomainMode::Assume && !is_provably_sin_nonzero(ctx, theta)
        {
            // Create NonZero assumption with HeuristicAssumption kind
            let mut event = crate::assumptions::AssumptionEvent::nonzero(ctx, sin_theta);
            event.kind = crate::assumptions::AssumptionKind::HeuristicAssumption;
            rewrite = rewrite.assume(event);
        }

        Some(rewrite)
    }

    fn target_types(&self) -> Option<Vec<&str>> {
        Some(vec!["Mul"])
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::High
    }

    fn soundness(&self) -> crate::rule::SoundnessLabel {
        // The transformation requires sin(θ) ≠ 0, which is either proven or assumed
        crate::rule::SoundnessLabel::EquivalenceUnderIntroducedRequires
    }
}

/// Verify that cos_args form a dyadic sequence: θ, 2θ, 4θ, ..., 2^(n-1)θ
///
/// Instead of structural comparison (which fails on normalized forms),
/// we extract the rational coefficient of each arg relative to π and check
/// if they form the sequence k, 2k, 4k, ..., 2^(n-1)k for some base k.
fn verify_dyadic_sequence(ctx: &mut cas_ast::Context, theta: ExprId, cos_args: &[ExprId]) -> bool {
    use crate::helpers::extract_rational_pi_multiple;
    use num_rational::BigRational;

    let n = cos_args.len() as u32;
    if n == 0 {
        return false;
    }

    // Extract the base coefficient from theta
    let base_coeff = match extract_rational_pi_multiple(ctx, theta) {
        Some(k) => k,
        None => return false, // theta must be a rational multiple of π
    };

    // Collect all coefficients from cos_args
    let mut coeffs: Vec<BigRational> = Vec::with_capacity(n as usize);
    for &arg in cos_args {
        match extract_rational_pi_multiple(ctx, arg) {
            Some(k) => coeffs.push(k),
            None => return false, // All args must be rational multiples of π
        }
    }

    // Build expected coefficients: base, 2*base, 4*base, ..., 2^(n-1)*base
    let mut expected: Vec<BigRational> = Vec::with_capacity(n as usize);
    for k in 0..n {
        let multiplier = BigRational::from_integer((1u64 << k).into());
        expected.push(&base_coeff * &multiplier);
    }

    // Check if coeffs matches expected as multiset
    let mut used = vec![false; expected.len()];
    for coeff in &coeffs {
        let mut found = false;
        for (i, exp) in expected.iter().enumerate() {
            if !used[i] && coeff == exp {
                used[i] = true;
                found = true;
                break;
            }
        }
        if !found {
            return false;
        }
    }

    used.iter().all(|&u| u)
}

pub fn register(simplifier: &mut crate::Simplifier) {
    // ABSOLUTE FIRST: Identity Zero Rules that must run before ANY expansion
    // These recognize exact identity forms and short-circuit to 0
    simplifier.add_rule(Box::new(Sin4xIdentityZeroRule));
    simplifier.add_rule(Box::new(TanDifferenceIdentityZeroRule));

    // PRE-ORDER: Evaluate sin(n·π) = 0 and cos(n·π) = (-1)^n BEFORE any expansion
    // This prevents unnecessary triple/double angle expansions on integer multiples of π
    simplifier.add_rule(Box::new(SinCosIntegerPiRule));

    // PRE-ORDER: Trig parity (odd/even functions)
    // sin(-u) = -sin(u), cos(-u) = cos(u), tan(-u) = -tan(u)
    simplifier.add_rule(Box::new(TrigOddEvenParityRule));

    // NOTE: TrigQuotientToNamedRule and TanDoubleAngleContractionRule are defined but NOT registered
    // They cause 9→16 numeric-only regression due to conflict with SecToRecipCosRule (ping-pong).
    // To enable, first disable SecToRecipCosRule, CscToRecipSinRule, and TanToSinCosRule.

    // Use the new data-driven EvaluateTrigTableRule instead of deprecated EvaluateTrigRule
    simplifier.add_rule(Box::new(super::evaluation::EvaluateTrigTableRule));
    simplifier.add_rule(Box::new(PythagoreanIdentityRule));
    simplifier.add_rule(Box::new(SecTanPythagoreanRule));
    simplifier.add_rule(Box::new(CscCotPythagoreanRule));

    // Hidden Cubic Identity: sin^6 + cos^6 + 3sin^2cos^2 = (sin^2+cos^2)^3
    // Should run in TRANSFORM phase before power expansions
    simplifier.add_rule(Box::new(TrigHiddenCubicIdentityRule));

    simplifier.add_rule(Box::new(AngleIdentityRule));
    // Triple tangent product: tan(u)·tan(π/3+u)·tan(π/3-u) → tan(3u)
    // Must run BEFORE TanToSinCosRule to prevent expansion
    simplifier.add_rule(Box::new(TanTripleProductRule));
    // Weierstrass Identity Zero Rules: MUST run BEFORE TanToSinCosRule
    // Pattern-driven cancellation: sin(x) - 2t/(1+t²) → 0, cos(x) - (1-t²)/(1+t²) → 0
    simplifier.add_rule(Box::new(WeierstrassSinIdentityZeroRule));
    simplifier.add_rule(Box::new(WeierstrassCosIdentityZeroRule));
    simplifier.add_rule(Box::new(TanToSinCosRule));
    // Dyadic Cos Product: 2^n·∏cos(2^k·θ) → sin(2^n·θ)/sin(θ)
    // Must run BEFORE DoubleAngleRule to recognize the pattern
    simplifier.add_rule(Box::new(DyadicCosProductToSinRule));
    simplifier.add_rule(Box::new(DoubleAngleRule));
    simplifier.add_rule(Box::new(DoubleAngleContractionRule)); // 2sin·cos→sin(2t), cos²-sin²→cos(2t)
                                                               // Sum-to-Product Quotient: runs BEFORE TripleAngleRule to avoid polynomial explosion
    simplifier.add_rule(Box::new(SinCosSumQuotientRule));
    // Standalone Sum-to-Product: sin(A)+sin(B), cos(A)+cos(B) etc. when args are k*π
    simplifier.add_rule(Box::new(TrigSumToProductRule));
    simplifier.add_rule(Box::new(TripleAngleRule)); // Shortcut: sin(3x), cos(3x), tan(3x)
    simplifier.add_rule(Box::new(QuintupleAngleRule)); // Shortcut: sin(5x), cos(5x)
    simplifier.add_rule(Box::new(RecursiveTrigExpansionRule));
    // Trig Quotient: sin(x)/cos(x) → tan(x) - runs after sum-to-product
    simplifier.add_rule(Box::new(TrigQuotientRule));
    // DISABLED: TrigSumToProductContractionRule conflicts with SinCosSumQuotientRule
    // The quotient rule (sin(a)+sin(b))/(cos(a)+cos(b)) → tan((a+b)/2) is more didactic
    // and should be preferred. This rule is defined but NOT registered.
    // simplifier.add_rule(Box::new(TrigSumToProductContractionRule));
    // Half-Angle Tangent: (1-cos(2x))/sin(2x) → tan(x), sin(2x)/(1+cos(2x)) → tan(x)
    simplifier.add_rule(Box::new(HalfAngleTangentRule));
    // Weierstrass Contraction: 2*tan(x/2)/(1+tan²) → sin(x), (1-tan²)/(1+tan²) → cos(x)
    simplifier.add_rule(Box::new(WeierstrassContractionRule));

    // DISABLED: ProductToSumRule conflicts with AngleIdentityRule creating infinite loops
    // ProductToSumRule: 2*sin(a)*cos(b) → sin(a+b) + sin(a-b)
    // AngleIdentityRule: sin(a+b) → sin(a)*cos(b) + cos(a)*sin(b)
    // When combined, they create cycles. Use manually for specific cases like Dirichlet kernel.
    // simplifier.add_rule(Box::new(ProductToSumRule));

    // DISABLED: Conflicts with Pythagorean identity rules causing infinite loops
    // This rule converts cos²(x) → 1-sin²(x) which interacts badly with:
    // - Pythagorean identities (sec²-tan²=1)
    // - Reciprocal trig canonicalization
    // Creating transformation cycles like: sec² → 1/cos² → 1/(1-sin²) → ...
    // See: debug_sec_tan.rs test and GitHub issue #X
    // simplifier.add_rule(Box::new(CanonicalizeTrigSquareRule));

    // Pythagorean Identity simplification: k - k*sin² → k*cos², k - k*cos² → k*sin²
    // This rule was extracted from CancelCommonFactorsRule for pedagogical clarity
    simplifier.add_rule(Box::new(super::pythagorean::TrigPythagoreanSimplifyRule));
    // N-ary Pythagorean: sin²(t) + cos²(t) → 1 in chains of any length
    simplifier.add_rule(Box::new(super::pythagorean::TrigPythagoreanChainRule));
    // Generic coefficient Pythagorean: A*sin²(x) + A*cos²(x) → A for any expression A
    simplifier.add_rule(Box::new(
        super::pythagorean::TrigPythagoreanGenericCoefficientRule,
    ));
    // Linear fold: a·sin²(t) + b·cos²(t) + c → (a-b)·sin²(t) + (b+c) using sin²+cos²=1
    simplifier.add_rule(Box::new(super::pythagorean::TrigPythagoreanLinearFoldRule));
    // Local collect fold: k·R·sin²(t) + R·cos²(t) - R → (k-1)·R·sin²(t) for residual R
    simplifier.add_rule(Box::new(
        super::pythagorean::TrigPythagoreanLocalCollectFoldRule,
    ));
    // Contraction: 1 + tan²(x) → sec²(x), 1 + cot²(x) → csc²(x)
    simplifier.add_rule(Box::new(super::pythagorean::RecognizeSecSquaredRule));
    simplifier.add_rule(Box::new(super::pythagorean::RecognizeCscSquaredRule));
    // Expansion: sec(x) → 1/cos(x), csc(x) → 1/sin(x) for canonical unification
    simplifier.add_rule(Box::new(super::pythagorean::SecToRecipCosRule));
    simplifier.add_rule(Box::new(super::pythagorean::CscToRecipSinRule));
    // Expansion: cot(x) → cos(x)/sin(x) for canonical unification
    simplifier.add_rule(Box::new(super::pythagorean::CotToCosSinRule));

    simplifier.add_rule(Box::new(AngleConsistencyRule));

    // Phase shift: sin(x + π/2) → cos(x), cos(x + π/2) → -sin(x), etc.
    simplifier.add_rule(Box::new(TrigPhaseShiftRule));

    // Supplementary angle: sin(8π/9) = sin(π - π/9) = sin(π/9)
    simplifier.add_rule(Box::new(SinSupplementaryAngleRule));

    // Fourth power difference: sin⁴(x) - cos⁴(x) → sin²(x) - cos²(x)
    simplifier.add_rule(Box::new(super::pythagorean::TrigEvenPowerDifferenceRule));

    // Cotangent half-angle difference: cot(u/2) - cot(u) = 1/sin(u)
    simplifier.add_rule(Box::new(CotHalfAngleDifferenceRule));
    // Tangent difference: tan(a-b) → (tan(a)-tan(b))/(1+tan(a)*tan(b))
    simplifier.add_rule(Box::new(TanDifferenceRule));
    // Hyperbolic Pythagorean: 1 - tanh²(x) → 1/cosh²(x)
    simplifier.add_rule(Box::new(HyperbolicTanhPythRule));
    // Hyperbolic half-angle: cosh²(x/2), sinh²(x/2) → cosh form
    simplifier.add_rule(Box::new(HyperbolicHalfAngleSquaresRule));
    // Generalized sin*cos contraction: k*sin(t)*cos(t) → (k/2)*sin(2t) for even k≥4
    simplifier.add_rule(Box::new(GeneralizedSinCosContractionRule));
}

define_rule!(
    AngleConsistencyRule,
    "Angle Consistency (Half-Angle)",
    |ctx, expr| {
        // Only run on Add/Sub/Mul/Div to capture context
        match ctx.get(expr) {
            Expr::Add(_, _) | Expr::Sub(_, _) | Expr::Mul(_, _) | Expr::Div(_, _) => {}
            _ => return None,
        }

        // 1. Collect all trig arguments
        let mut trig_args = Vec::new();
        collect_trig_args_recursive(ctx, expr, &mut trig_args);

        if trig_args.is_empty() {
            return None;
        }

        // 2. Check for half-angle relationship
        // We look for pair (A, B) such that A = 2*B.
        // Then we expand trig(A) into trig(B).

        let mut target_expansion: Option<(ExprId, ExprId)> = None; // (A, B) where A=2B

        for i in 0..trig_args.len() {
            for j in 0..trig_args.len() {
                if i == j {
                    continue;
                }
                let a = trig_args[i];
                let b = trig_args[j];

                if is_double(ctx, a, b) {
                    target_expansion = Some((a, b));
                    break;
                }
            }
            if target_expansion.is_some() {
                break;
            }
        }

        if let Some((large_angle, small_angle)) = target_expansion {
            // Expand all occurrences of trig(large_angle) in expr
            // We need a recursive replacement helper
            let new_expr = expand_trig_angle(ctx, expr, large_angle, small_angle);
            if new_expr != expr {
                return Some(Rewrite::new(new_expr).desc("Half-Angle Expansion"));
            }
        }

        None
    }
);

fn collect_trig_args_recursive(ctx: &cas_ast::Context, expr: ExprId, args: &mut Vec<ExprId>) {
    match ctx.get(expr) {
        Expr::Function(name, fargs) => {
            if (name == "sin" || name == "cos" || name == "tan") && fargs.len() == 1 {
                args.push(fargs[0]);
            }
            for arg in fargs {
                collect_trig_args_recursive(ctx, *arg, args);
            }
        }
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) | Expr::Pow(l, r) => {
            collect_trig_args_recursive(ctx, *l, args);
            collect_trig_args_recursive(ctx, *r, args);
        }
        Expr::Neg(e) => collect_trig_args_recursive(ctx, *e, args),
        _ => {}
    }
}

fn is_double(ctx: &cas_ast::Context, large: ExprId, small: ExprId) -> bool {
    // Check if large == 2 * small

    // Case 1: large = 2 * small
    if let Expr::Mul(l, r) = ctx.get(large) {
        if let Expr::Number(n) = ctx.get(*l) {
            if n == &num_rational::BigRational::from_integer(2.into())
                && crate::ordering::compare_expr(ctx, *r, small) == Ordering::Equal
            {
                return true;
            }
        }
        if let Expr::Number(n) = ctx.get(*r) {
            if n == &num_rational::BigRational::from_integer(2.into())
                && crate::ordering::compare_expr(ctx, *l, small) == Ordering::Equal
            {
                return true;
            }
        }
    }

    // Case 2: small = large / 2
    if let Expr::Div(n, d) = ctx.get(small) {
        if let Expr::Number(val) = ctx.get(*d) {
            if val == &num_rational::BigRational::from_integer(2.into())
                && crate::ordering::compare_expr(ctx, *n, large) == Ordering::Equal
            {
                return true;
            }
        }
    }

    // Case 3: small = large * 0.5
    if let Expr::Mul(l, r) = ctx.get(small) {
        if let Expr::Number(n) = ctx.get(*l) {
            if n == &num_rational::BigRational::new(1.into(), 2.into())
                && crate::ordering::compare_expr(ctx, *r, large) == Ordering::Equal
            {
                return true;
            }
        }
        if let Expr::Number(n) = ctx.get(*r) {
            if n == &num_rational::BigRational::new(1.into(), 2.into())
                && crate::ordering::compare_expr(ctx, *l, large) == Ordering::Equal
            {
                return true;
            }
        }
    }

    false
}

fn expand_trig_angle(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    large_angle: ExprId,
    small_angle: ExprId,
) -> ExprId {
    let expr_data = ctx.get(expr).clone();

    // Check if this node is trig(large_angle)
    if let Expr::Function(name, args) = &expr_data {
        if args.len() == 1
            && crate::ordering::compare_expr(ctx, args[0], large_angle) == Ordering::Equal
        {
            match name.as_str() {
                "sin" => {
                    // sin(A) -> 2sin(A/2)cos(A/2)
                    let two = ctx.num(2);
                    let sin_half = ctx.add(Expr::Function("sin".to_string(), vec![small_angle]));
                    let cos_half = ctx.add(Expr::Function("cos".to_string(), vec![small_angle]));
                    let term = smart_mul(ctx, sin_half, cos_half);
                    return smart_mul(ctx, two, term);
                }
                "cos" => {
                    // cos(A) -> 2cos^2(A/2) - 1
                    let two = ctx.num(2);
                    let one = ctx.num(1);
                    let cos_half = ctx.add(Expr::Function("cos".to_string(), vec![small_angle]));
                    let cos_sq = ctx.add(Expr::Pow(cos_half, two));
                    let term = smart_mul(ctx, two, cos_sq);
                    return ctx.add(Expr::Sub(term, one));
                }
                "tan" => {
                    // tan(A) -> 2tan(A/2) / (1 - tan^2(A/2))
                    let two = ctx.num(2);
                    let one = ctx.num(1);
                    let tan_half = ctx.add(Expr::Function("tan".to_string(), vec![small_angle]));
                    let num = smart_mul(ctx, two, tan_half);

                    let tan_sq = ctx.add(Expr::Pow(tan_half, two));
                    let den = ctx.add(Expr::Sub(one, tan_sq));

                    return ctx.add(Expr::Div(num, den));
                }
                _ => {}
            }
        }
    }

    // Recurse
    match expr_data {
        Expr::Add(l, r) => {
            let nl = expand_trig_angle(ctx, l, large_angle, small_angle);
            let nr = expand_trig_angle(ctx, r, large_angle, small_angle);
            if nl != l || nr != r {
                ctx.add(Expr::Add(nl, nr))
            } else {
                expr
            }
        }
        Expr::Sub(l, r) => {
            let nl = expand_trig_angle(ctx, l, large_angle, small_angle);
            let nr = expand_trig_angle(ctx, r, large_angle, small_angle);
            if nl != l || nr != r {
                ctx.add(Expr::Sub(nl, nr))
            } else {
                expr
            }
        }
        Expr::Mul(l, r) => {
            let nl = expand_trig_angle(ctx, l, large_angle, small_angle);
            let nr = expand_trig_angle(ctx, r, large_angle, small_angle);
            if nl != l || nr != r {
                smart_mul(ctx, nl, nr)
            } else {
                expr
            }
        }
        Expr::Div(l, r) => {
            let nl = expand_trig_angle(ctx, l, large_angle, small_angle);
            let nr = expand_trig_angle(ctx, r, large_angle, small_angle);
            if nl != l || nr != r {
                ctx.add(Expr::Div(nl, nr))
            } else {
                expr
            }
        }
        Expr::Pow(b, e) => {
            let nb = expand_trig_angle(ctx, b, large_angle, small_angle);
            let ne = expand_trig_angle(ctx, e, large_angle, small_angle);
            if nb != b || ne != e {
                ctx.add(Expr::Pow(nb, ne))
            } else {
                expr
            }
        }
        Expr::Neg(e) => {
            let ne = expand_trig_angle(ctx, e, large_angle, small_angle);
            if ne != e {
                ctx.add(Expr::Neg(ne))
            } else {
                expr
            }
        }
        Expr::Function(name, args) => {
            let mut new_args = Vec::new();
            let mut changed = false;
            for arg in args {
                let na = expand_trig_angle(ctx, arg, large_angle, small_angle);
                if na != arg {
                    changed = true;
                }
                new_args.push(na);
            }
            if changed {
                ctx.add(Expr::Function(name, new_args))
            } else {
                expr
            }
        }
        _ => expr,
    }
}

