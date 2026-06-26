use super::methods::*;
use super::verification::*;
use super::verification_algebraic::*;
use super::verification_normalization::*;
use super::*;

use crate::semantic_equality::SemanticEqualityChecker;
use crate::symbolic_differentiation_support::differentiate_symbolic_expr;
use cas_ast::{ConditionPredicate, Context, Expr};
use std::collections::BTreeMap;
use std::time::Instant;

#[test]
fn default_config_disables_backend() {
    assert_eq!(
        AlgorithmicIntegrationBackendConfig::default(),
        AlgorithmicIntegrationBackendConfig::disabled()
    );
}

#[test]
fn probe_runner_consumes_method_and_verification_budget() {
    let mut runner =
        AlgorithmicIntegrationProbeRunner::new(AlgorithmicIntegrationBackendBudget::single_probe());
    let mut first_probe_ran = false;

    assert!(runner
        .try_method_probe(AlgorithmicIntegrationMethod::Rational, |probe_runner| {
            first_probe_ran = true;
            assert_eq!(probe_runner.remaining_method_probes(), 0);
            assert_eq!(probe_runner.method_probe_budget_limit(), 1);
            assert_eq!(probe_runner.verification_check_budget_limit(), 1);
            assert!(probe_runner.try_verification_check());
            AlgorithmicIntegrationProbeResult::NoMatch(
                AlgorithmicIntegrationProbeNoMatchReason::ShapeMismatch,
            )
        },)
        .is_none());

    assert!(first_probe_ran);
    assert_eq!(runner.remaining_method_probes(), 0);
    assert_eq!(runner.remaining_verification_checks(), 0);
    assert_eq!(
        runner.method_probe_attempts(),
        &[AlgorithmicIntegrationMethod::Rational]
    );
    assert_eq!(
        runner.method_probe_no_match_reasons(),
        &[(
            AlgorithmicIntegrationMethod::Rational,
            AlgorithmicIntegrationProbeNoMatchReason::ShapeMismatch
        )]
    );
    assert_eq!(runner.method_probes_used(), 1);
    assert_eq!(runner.verification_checks_used(), 1);
    assert!(!runner.method_budget_exhausted());
    assert!(!runner.verification_budget_exhausted());

    let mut second_probe_ran = false;
    assert!(runner
        .try_method_probe(AlgorithmicIntegrationMethod::Hermite, |_| {
            second_probe_ran = true;
            AlgorithmicIntegrationProbeResult::NoMatch(
                AlgorithmicIntegrationProbeNoMatchReason::ShapeMismatch,
            )
        },)
        .is_none());

    assert!(!second_probe_ran);
    assert_eq!(
        runner.method_probe_attempts(),
        &[AlgorithmicIntegrationMethod::Rational]
    );
    assert_eq!(
        runner.method_probe_no_match_reasons(),
        &[(
            AlgorithmicIntegrationMethod::Rational,
            AlgorithmicIntegrationProbeNoMatchReason::ShapeMismatch
        )]
    );
    assert_eq!(runner.method_probes_used(), 1);
    assert_eq!(runner.verification_checks_used(), 1);
    assert!(runner.method_budget_exhausted());
}

#[test]
fn disabled_backend_candidate_is_diagnostic_only() {
    let mut ctx = Context::new();
    let integrand = ctx.num(1);

    let candidate = try_algorithmic_integration_backend(
        &mut ctx,
        integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::default(),
    );

    assert_eq!(candidate.integrand, integrand);
    assert_eq!(candidate.variable, "x");
    assert_eq!(candidate.antiderivative, None);
    assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Unsupported);
    assert_eq!(
        candidate.verification_status,
        AlgorithmicIntegrationVerificationStatus::NotAttempted
    );
    assert_eq!(
        candidate.residual_reason,
        Some(AlgorithmicIntegrationResidualReason::DisabledByMode)
    );
    assert_eq!(
        candidate.trace_level,
        AlgorithmicIntegrationTraceLevel::DiagnosticOnly
    );
    assert_eq!(candidate.method_probes_used, 0);
    assert_eq!(candidate.verification_checks_used, 0);
    assert_eq!(candidate.method_probe_budget_limit, 0);
    assert_eq!(candidate.verification_check_budget_limit, 0);
    assert!(!candidate.is_publicly_acceptable());
    assert_eq!(
        candidate.publication_status(),
        AlgorithmicIntegrationPublicationStatus::RejectedNoAntiderivative
    );
    assert_eq!(
        candidate.fallback_status(AlgorithmicIntegrationBackendConfig::default()),
        AlgorithmicIntegrationFallbackStatus::BlockedByCandidatePolicy
    );
    assert_eq!(candidate.public_antiderivative(), None);
}

#[test]
fn diagnostic_backend_respects_method_probe_budget() {
    let mut ctx = Context::new();
    let integrand = ctx.num(1);
    let config = AlgorithmicIntegrationBackendConfig::diagnostic_only()
        .with_budget(AlgorithmicIntegrationBackendBudget::disabled());

    let candidate = try_algorithmic_integration_backend(&mut ctx, integrand, "x", config);

    assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Unsupported);
    assert_eq!(candidate.method_probe_budget_limit, 0);
    assert_eq!(candidate.verification_check_budget_limit, 0);
    assert_eq!(candidate.antiderivative, None);
    assert_eq!(
        candidate.verification_status,
        AlgorithmicIntegrationVerificationStatus::Inconclusive
    );
    assert_eq!(
        candidate.residual_reason,
        Some(AlgorithmicIntegrationResidualReason::BudgetExceeded)
    );
    assert!(!candidate.is_publicly_acceptable());
    assert_eq!(
        candidate.publication_status(),
        AlgorithmicIntegrationPublicationStatus::RejectedNoAntiderivative
    );
    assert_eq!(
        candidate.fallback_status(config),
        AlgorithmicIntegrationFallbackStatus::BlockedByCandidatePolicy
    );
    assert_eq!(candidate.public_antiderivative(), None);
    assert_eq!(candidate.fallback_antiderivative(config), None);
}

#[test]
fn diagnostic_empty_backend_reports_unsupported_without_public_candidate() {
    let mut ctx = Context::new();
    let integrand = ctx.num(1);

    let candidate = try_algorithmic_integration_backend(
        &mut ctx,
        integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );

    assert_eq!(
        candidate.residual_reason,
        Some(AlgorithmicIntegrationResidualReason::UnsupportedMethod)
    );
    assert!(!candidate.is_publicly_acceptable());
    assert_eq!(candidate.public_antiderivative(), None);
}

#[test]
fn sextic_rational_factoring_fully_over_q_verifies() {
    // 1/(x^6-1) = 1/((x-1)(x+1)(x^2+x+1)(x^2-x+1)): linears + Sophie-Germain quadratics, all
    // over Q. The algebraic zero test (sqrt(3) atom, t^2=3) decides it once the multipoly budget
    // is large enough for the degree-6 residual.
    for src in ["1/(x^6-1)", "1/(x^6-64)"] {
        let mut ctx = Context::new();
        let integrand = cas_parser::parse(src, &mut ctx).expect("integrand");
        let candidate = try_algorithmic_integration_backend(
            &mut ctx,
            integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );
        assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Rational);
        assert!(
            matches!(
                candidate.verification_status,
                AlgorithmicIntegrationVerificationStatus::Verified
                    | AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
            ),
            "{src}: expected verified, got {:?} (blocker {:?})",
            candidate.verification_status,
            candidate.verification_blocker
        );
    }
}

#[test]
fn sextic_rational_irreducible_over_q_stays_residual() {
    // 1/(x^6+1) and 1/(x^8-1) need factoring over R (irreducible-over-Q quartics): they must
    // NOT be claimed — the budget raise does not turn an unfactorable denominator into a result.
    for src in ["1/(x^6+1)", "1/(x^8-1)"] {
        let mut ctx = Context::new();
        let integrand = cas_parser::parse(src, &mut ctx).expect("integrand");
        let candidate = try_algorithmic_integration_backend(
            &mut ctx,
            integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );
        assert_ne!(
            candidate.verification_status,
            AlgorithmicIntegrationVerificationStatus::Verified,
            "{src} should stay residual (needs factoring over R)"
        );
    }
}

#[test]
fn diagnostic_rational_probe_verifies_but_is_not_fallback_consumable() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("1/(x+1)", &mut ctx).expect("integrand");
    let denominator = match ctx.get(integrand) {
        Expr::Div(_, denominator) => *denominator,
        _ => panic!("expected reciprocal integrand"),
    };

    let candidate = try_algorithmic_integration_backend(
        &mut ctx,
        integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );
    assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Rational);
    assert_eq!(
        candidate.verification_status,
        AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions,
        "blocker: {:?}, evidence: {:?}, normalization: {:?}, passes: {:?}, residual: {:?}",
        candidate.verification_blocker,
        candidate.verification_evidence,
        candidate.verification_normalization_reason,
        candidate.verification_normalization_passes_used,
        candidate.residual_reason
    );
    assert_eq!(
        candidate.verification_evidence,
        AlgorithmicIntegrationVerificationEvidence::DirectDifferentiation
    );
    assert_eq!(candidate.residual_reason, None);
    assert_eq!(
        candidate.required_conditions,
        vec![ConditionPredicate::NonZero(denominator)]
    );
    assert_eq!(
        candidate.trace_level,
        AlgorithmicIntegrationTraceLevel::AlgorithmicSummary
    );
    assert_eq!(candidate.method_probes_used, 1);
    assert_eq!(candidate.verification_checks_used, 1);
    assert!(candidate.antiderivative.is_some());
    assert!(candidate.is_publicly_acceptable());
    assert_eq!(
        candidate.publication_status(),
        AlgorithmicIntegrationPublicationStatus::Accepted
    );
    assert_eq!(
        candidate.fallback_status(AlgorithmicIntegrationBackendConfig::diagnostic_only()),
        AlgorithmicIntegrationFallbackStatus::BlockedByMode
    );
    assert!(candidate.public_antiderivative().is_some());
    assert_eq!(
        candidate.fallback_antiderivative(AlgorithmicIntegrationBackendConfig::diagnostic_only()),
        None
    );
}

#[test]
fn residual_fallback_mode_can_consume_verified_conditioned_candidate() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("1/(x+1)", &mut ctx).expect("integrand");
    let config = AlgorithmicIntegrationBackendConfig::residual_fallback();

    let candidate = try_algorithmic_integration_backend(&mut ctx, integrand, "x", config);

    assert_eq!(
        candidate.verification_status,
        AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
    );
    assert_eq!(
        candidate.publication_status(),
        AlgorithmicIntegrationPublicationStatus::Accepted
    );
    assert_eq!(
        candidate.fallback_status(config),
        AlgorithmicIntegrationFallbackStatus::Eligible
    );
    assert!(candidate.fallback_antiderivative(config).is_some());
}

#[test]
fn diagnostic_rational_probe_verifies_numeric_scaled_affine_reciprocal() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("2/(x+1)", &mut ctx).expect("integrand");
    let denominator = match ctx.get(integrand) {
        Expr::Div(_, denominator) => *denominator,
        _ => panic!("expected scaled reciprocal integrand"),
    };

    let candidate = try_algorithmic_integration_backend(
        &mut ctx,
        integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );

    assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Rational);
    assert_eq!(
        candidate.verification_status,
        AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
    );
    assert_eq!(
        candidate.verification_evidence,
        AlgorithmicIntegrationVerificationEvidence::MethodSpecificDifferentiation
    );
    assert_eq!(candidate.residual_reason, None);
    assert_eq!(
        candidate.required_conditions,
        vec![ConditionPredicate::NonZero(denominator)]
    );
    assert_eq!(
        candidate.trace_level,
        AlgorithmicIntegrationTraceLevel::AlgorithmicSummary
    );
    assert_eq!(candidate.method_probes_used, 1);
    assert_eq!(candidate.verification_checks_used, 1);
    assert!(candidate.public_antiderivative().is_some());
}

#[test]
fn diagnostic_rational_probe_verifies_symbolic_scaled_affine_reciprocal() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("a/(x+1)", &mut ctx).expect("integrand");
    let denominator = match ctx.get(integrand) {
        Expr::Div(_, denominator) => *denominator,
        _ => panic!("expected symbolic scaled reciprocal integrand"),
    };

    let candidate = try_algorithmic_integration_backend(
        &mut ctx,
        integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );

    assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Rational);
    assert_eq!(
        candidate.verification_status,
        AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
    );
    assert_eq!(
        candidate.verification_evidence,
        AlgorithmicIntegrationVerificationEvidence::MethodSpecificDifferentiation
    );
    assert_eq!(candidate.residual_reason, None);
    assert_eq!(
        candidate.required_conditions,
        vec![ConditionPredicate::NonZero(denominator)]
    );
    assert_eq!(
        candidate.trace_level,
        AlgorithmicIntegrationTraceLevel::AlgorithmicSummary
    );
    assert_eq!(candidate.method_probes_used, 1);
    assert_eq!(candidate.verification_checks_used, 1);
    assert!(candidate.public_antiderivative().is_some());
}

#[test]
fn diagnostic_rational_probe_verifies_product_symbolic_scaled_affine_reciprocal() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("2*a/(x+1)", &mut ctx).expect("integrand");
    let denominator = match ctx.get(integrand) {
        Expr::Div(_, denominator) => *denominator,
        _ => panic!("expected product symbolic scaled reciprocal integrand"),
    };

    let candidate = try_algorithmic_integration_backend(
        &mut ctx,
        integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );

    assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Rational);
    assert_eq!(
        candidate.verification_status,
        AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
    );
    assert_eq!(
        candidate.verification_evidence,
        AlgorithmicIntegrationVerificationEvidence::MethodSpecificDifferentiation
    );
    assert_eq!(
        candidate.required_conditions,
        vec![ConditionPredicate::NonZero(denominator)]
    );
    assert_eq!(candidate.residual_reason, None);
    assert_eq!(candidate.method_probes_used, 1);
    assert_eq!(candidate.verification_checks_used, 1);
    assert!(candidate.public_antiderivative().is_some());
}

#[test]
fn diagnostic_rational_probe_verifies_numeric_slope_affine_reciprocal() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("1/(2*x+1)", &mut ctx).expect("integrand");
    let denominator = match ctx.get(integrand) {
        Expr::Div(_, denominator) => *denominator,
        _ => panic!("expected numeric slope reciprocal integrand"),
    };

    let candidate = try_algorithmic_integration_backend(
        &mut ctx,
        integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );

    assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Rational);
    assert_eq!(
        candidate.verification_status,
        AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
    );
    assert_eq!(
        candidate.verification_evidence,
        AlgorithmicIntegrationVerificationEvidence::MethodSpecificDifferentiation
    );
    assert_eq!(
        candidate.required_conditions,
        vec![ConditionPredicate::NonZero(denominator)]
    );
    assert_eq!(candidate.residual_reason, None);
    assert_eq!(candidate.method_probes_used, 1);
    assert_eq!(candidate.verification_checks_used, 1);
    assert!(candidate.public_antiderivative().is_some());
}

#[test]
fn diagnostic_rational_probe_verifies_negative_slope_affine_reciprocal() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("1/(1-2*x)", &mut ctx).expect("integrand");

    let candidate = try_algorithmic_integration_backend(
        &mut ctx,
        integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );

    assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Rational);
    assert_eq!(
        candidate.verification_status,
        AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
    );
    assert_eq!(
        candidate.verification_evidence,
        AlgorithmicIntegrationVerificationEvidence::MethodSpecificDifferentiation
    );
    assert_eq!(candidate.residual_reason, None);
    assert!(candidate.public_antiderivative().is_some());
}

#[test]
fn diagnostic_rational_probe_verifies_symbolic_scaled_numeric_slope_affine_reciprocal() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("a/(2*x+1)", &mut ctx).expect("integrand");

    let candidate = try_algorithmic_integration_backend(
        &mut ctx,
        integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );

    assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Rational);
    assert_eq!(
        candidate.verification_status,
        AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
    );
    assert_eq!(
        candidate.verification_evidence,
        AlgorithmicIntegrationVerificationEvidence::MethodSpecificDifferentiation
    );
    assert_eq!(candidate.residual_reason, None);
    assert!(candidate.public_antiderivative().is_some());
}

#[test]
fn diagnostic_rational_probe_verifies_symbolic_slope_affine_reciprocal() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("1/(a*x+b)", &mut ctx).expect("integrand");
    let denominator = match ctx.get(integrand) {
        Expr::Div(_, denominator) => *denominator,
        _ => panic!("expected symbolic slope reciprocal integrand"),
    };
    let slope = match affine_denominator_slope(&mut ctx, denominator, "x")
        .expect("symbolic affine slope")
    {
        BackendAffineSlope::Symbolic(slope) => slope,
        BackendAffineSlope::Numeric(_) => panic!("expected symbolic slope"),
    };

    let candidate = try_algorithmic_integration_backend(
        &mut ctx,
        integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );

    assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Rational);
    assert_eq!(
        candidate.verification_status,
        AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
    );
    assert_eq!(
        candidate.verification_evidence,
        AlgorithmicIntegrationVerificationEvidence::MethodSpecificDifferentiation
    );
    assert_eq!(
        candidate.required_conditions,
        vec![
            ConditionPredicate::NonZero(denominator),
            ConditionPredicate::NonZero(slope),
        ]
    );
    assert_eq!(candidate.residual_reason, None);
    assert_eq!(candidate.method_probes_used, 1);
    assert_eq!(candidate.verification_checks_used, 1);
    assert!(candidate.public_antiderivative().is_some());
}

#[test]
fn diagnostic_rational_probe_verifies_product_symbolic_slope_affine_reciprocal() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("1/(2*a*x+b)", &mut ctx).expect("integrand");
    let denominator = match ctx.get(integrand) {
        Expr::Div(_, denominator) => *denominator,
        _ => panic!("expected product symbolic slope reciprocal integrand"),
    };
    let slope = match affine_denominator_slope(&mut ctx, denominator, "x")
        .expect("product symbolic affine slope")
    {
        BackendAffineSlope::Symbolic(slope) => slope,
        BackendAffineSlope::Numeric(_) => panic!("expected symbolic slope"),
    };

    let candidate = try_algorithmic_integration_backend(
        &mut ctx,
        integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );

    assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Rational);
    assert_eq!(
        candidate.verification_status,
        AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
    );
    assert_eq!(
        candidate.verification_evidence,
        AlgorithmicIntegrationVerificationEvidence::MethodSpecificDifferentiation
    );
    assert_eq!(
        candidate.verification_normalization_reason,
        AlgorithmicIntegrationVerificationNormalizationReason::None
    );
    assert_eq!(
        candidate.required_conditions,
        vec![
            ConditionPredicate::NonZero(denominator),
            ConditionPredicate::NonZero(slope),
        ]
    );
    assert_eq!(candidate.residual_reason, None);
    assert_eq!(candidate.method_probes_used, 1);
    assert_eq!(candidate.verification_checks_used, 1);
    assert!(candidate.public_antiderivative().is_some());
}

#[test]
fn diagnostic_rational_probe_verifies_symbolic_slope_affine_quotient_remainder() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("(m*(a*x+b)+c)/(a*x+b)", &mut ctx).expect("integrand");
    let denominator = match ctx.get(integrand) {
        Expr::Div(_, denominator) => *denominator,
        _ => panic!("expected affine quotient-remainder integrand"),
    };
    let slope = match affine_denominator_slope(&mut ctx, denominator, "x")
        .expect("symbolic affine slope")
    {
        BackendAffineSlope::Symbolic(slope) => slope,
        BackendAffineSlope::Numeric(_) => panic!("expected symbolic slope"),
    };

    let candidate = try_algorithmic_integration_backend(
        &mut ctx,
        integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );

    assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Rational);
    assert_eq!(
        candidate.verification_status,
        AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
    );
    assert_eq!(
        candidate.verification_evidence,
        AlgorithmicIntegrationVerificationEvidence::MethodSpecificDifferentiation
    );
    assert_eq!(
        candidate.verification_normalization_reason,
        AlgorithmicIntegrationVerificationNormalizationReason::None
    );
    assert_eq!(candidate.residual_reason, None);
    assert_eq!(
        candidate.required_conditions,
        vec![
            ConditionPredicate::NonZero(denominator),
            ConditionPredicate::NonZero(slope),
        ]
    );
    assert_eq!(candidate.method_probes_used, 1);
    assert_eq!(candidate.verification_checks_used, 1);
    assert!(candidate.public_antiderivative().is_some());
}

#[test]
fn diagnostic_rational_probe_verifies_external_scaled_symbolic_slope_affine_reciprocal() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("c/(a*x+b)", &mut ctx).expect("integrand");

    let candidate = try_algorithmic_integration_backend(
        &mut ctx,
        integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );

    assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Rational);
    assert_eq!(
        candidate.verification_status,
        AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
    );
    assert_eq!(candidate.residual_reason, None);
    assert_eq!(candidate.required_conditions.len(), 2);
    assert!(candidate.public_antiderivative().is_some());
}

#[test]
fn diagnostic_rational_probe_verifies_external_scaled_product_symbolic_slope_affine_reciprocal() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("c/(2*a*x+b)", &mut ctx).expect("integrand");

    let candidate = try_algorithmic_integration_backend(
        &mut ctx,
        integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );

    assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Rational);
    assert_eq!(
        candidate.verification_status,
        AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
    );
    assert_eq!(candidate.residual_reason, None);
    assert_eq!(candidate.required_conditions.len(), 2);
    assert!(candidate.public_antiderivative().is_some());
}

#[test]
fn diagnostic_rational_probe_verifies_negative_symbolic_slope_affine_reciprocal() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("1/(b-a*x)", &mut ctx).expect("integrand");

    let candidate = try_algorithmic_integration_backend(
        &mut ctx,
        integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );

    assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Rational);
    assert_eq!(
        candidate.verification_status,
        AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
    );
    assert_eq!(candidate.residual_reason, None);
    assert_eq!(candidate.required_conditions.len(), 2);
    assert!(candidate.public_antiderivative().is_some());
}

#[test]
fn diagnostic_rational_probe_verifies_raw_affine_quotient_remainder() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("x/(x+1)", &mut ctx).expect("integrand");
    let denominator = match ctx.get(integrand) {
        Expr::Div(_, denominator) => *denominator,
        _ => panic!("expected raw affine quotient-remainder integrand"),
    };

    let candidate = try_algorithmic_integration_backend(
        &mut ctx,
        integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );

    assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Rational);
    assert_eq!(
        candidate.verification_status,
        AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
    );
    assert_eq!(
        candidate.verification_evidence,
        AlgorithmicIntegrationVerificationEvidence::MethodSpecificDifferentiation
    );
    assert_eq!(
        candidate.verification_normalization_reason,
        AlgorithmicIntegrationVerificationNormalizationReason::None
    );
    assert_eq!(candidate.residual_reason, None);
    assert_eq!(
        candidate.required_conditions,
        vec![ConditionPredicate::NonZero(denominator)]
    );
    assert_eq!(candidate.method_probes_used, 1);
    assert_eq!(candidate.verification_checks_used, 1);
    assert!(candidate.public_antiderivative().is_some());
}

#[test]
fn diagnostic_rational_probe_verifies_external_scaled_zero_intercept_affine_quotient() {
    let mut ctx = Context::new();

    for source in ["3*x/(2*x+b)", "-3*x/(2*x+b)", "a*x/(c*x+d)", "-a*x/(c*x+d)"] {
        let integrand = cas_parser::parse(source, &mut ctx).expect("integrand");
        let candidate = try_algorithmic_integration_backend(
            &mut ctx,
            integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );

        assert_eq!(
            candidate.method,
            AlgorithmicIntegrationMethod::Rational,
            "{source}"
        );
        assert_eq!(
            candidate.verification_status,
            AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions,
            "{source}"
        );
        assert_eq!(candidate.residual_reason, None, "{source}");
        assert_eq!(candidate.method_probes_used, 1, "{source}");
        assert_eq!(candidate.verification_checks_used, 1, "{source}");
        assert!(candidate.public_antiderivative().is_some(), "{source}");
    }

    let integrand = cas_parser::parse("a*x/(c*x+d)", &mut ctx).expect("integrand");
    let candidate = try_algorithmic_integration_backend(
        &mut ctx,
        integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );
    let mut required_displays: Vec<_> = candidate
        .required_conditions
        .iter()
        .map(|condition| match condition {
            ConditionPredicate::NonZero(expr) => format!(
                "{} ≠ 0",
                cas_formatter::DisplayExpr {
                    context: &ctx,
                    id: *expr,
                }
            ),
            _ => condition.display(),
        })
        .collect();
    required_displays.sort();
    assert_eq!(
        required_displays,
        vec!["c * x + d ≠ 0".to_string(), "c ≠ 0".to_string()]
    );
}

#[test]
fn diagnostic_rational_probe_verifies_numeric_slope_affine_symbolic_intercept_quotient_remainder() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("(3*x+c)/(2*x+1)", &mut ctx).expect("integrand");
    let denominator = match ctx.get(integrand) {
        Expr::Div(_, denominator) => *denominator,
        _ => panic!("expected symbolic-intercept affine quotient-remainder integrand"),
    };

    let candidate = try_algorithmic_integration_backend(
        &mut ctx,
        integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );

    assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Rational);
    assert_eq!(
        candidate.verification_status,
        AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
    );
    assert_eq!(candidate.residual_reason, None);
    assert_eq!(
        candidate.required_conditions,
        vec![ConditionPredicate::NonZero(denominator)]
    );
    assert_eq!(candidate.method_probes_used, 1);
    assert_eq!(candidate.verification_checks_used, 1);
    assert!(candidate.public_antiderivative().is_some());
}

#[test]
fn diagnostic_rational_probe_verifies_symbolic_slope_raw_affine_quotient_remainder() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("(3*x+c)/(a*x+b)", &mut ctx).expect("integrand");
    let denominator = match ctx.get(integrand) {
        Expr::Div(_, denominator) => *denominator,
        _ => panic!("expected symbolic-slope raw affine quotient-remainder integrand"),
    };
    let slope = match affine_denominator_slope(&mut ctx, denominator, "x")
        .expect("symbolic affine slope")
    {
        BackendAffineSlope::Symbolic(slope) => slope,
        BackendAffineSlope::Numeric(_) => panic!("expected symbolic slope"),
    };

    let candidate = try_algorithmic_integration_backend(
        &mut ctx,
        integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );

    assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Rational);
    assert_eq!(
        candidate.verification_status,
        AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
    );
    assert_eq!(candidate.residual_reason, None);
    assert_eq!(
        candidate.required_conditions,
        vec![
            ConditionPredicate::NonZero(denominator),
            ConditionPredicate::NonZero(slope),
        ]
    );
    assert_eq!(candidate.method_probes_used, 1);
    assert_eq!(candidate.verification_checks_used, 1);
    assert!(candidate.public_antiderivative().is_some());
}

#[test]
fn diagnostic_rational_probe_verifies_product_symbolic_slope_raw_affine_quotient_remainder() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("(3*x+c)/(2*a*x+b)", &mut ctx).expect("integrand");
    let denominator = match ctx.get(integrand) {
        Expr::Div(_, denominator) => *denominator,
        _ => panic!("expected product-symbolic-slope raw affine quotient-remainder integrand"),
    };
    let slope = match affine_denominator_slope(&mut ctx, denominator, "x")
        .expect("product symbolic affine slope")
    {
        BackendAffineSlope::Symbolic(slope) => slope,
        BackendAffineSlope::Numeric(_) => panic!("expected symbolic slope"),
    };

    let candidate = try_algorithmic_integration_backend(
        &mut ctx,
        integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );

    assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Rational);
    assert_eq!(
        candidate.verification_status,
        AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
    );
    assert_eq!(candidate.residual_reason, None);
    assert_eq!(
        candidate.required_conditions,
        vec![
            ConditionPredicate::NonZero(denominator),
            ConditionPredicate::NonZero(slope),
        ]
    );
    assert_eq!(candidate.method_probes_used, 1);
    assert_eq!(candidate.verification_checks_used, 1);
    assert!(candidate.public_antiderivative().is_some());
}

#[test]
fn diagnostic_rational_probe_rejects_non_linear_reciprocal_numerator() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("x^2/(x+1)", &mut ctx).expect("integrand");

    let candidate = try_algorithmic_integration_backend(
        &mut ctx,
        integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );

    assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Unsupported);
    assert_eq!(
        candidate.method_probe_no_match_reasons.first(),
        Some(&(
            AlgorithmicIntegrationMethod::Rational,
            AlgorithmicIntegrationProbeNoMatchReason::NumeratorPolicyMismatch,
        ))
    );
    assert!(!candidate.is_publicly_acceptable());
    assert_eq!(candidate.public_antiderivative(), None);
}

#[test]
fn diagnostic_rational_probe_rejects_non_linear_symbolic_slope_reciprocal_numerator() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("x^2/(a*x+b)", &mut ctx).expect("integrand");

    let candidate = try_algorithmic_integration_backend(
        &mut ctx,
        integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );

    assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Unsupported);
    assert_eq!(
        candidate.method_probe_no_match_reasons.first(),
        Some(&(
            AlgorithmicIntegrationMethod::Rational,
            AlgorithmicIntegrationProbeNoMatchReason::NumeratorPolicyMismatch,
        ))
    );
    assert!(!candidate.is_publicly_acceptable());
    assert_eq!(candidate.public_antiderivative(), None);
}

#[test]
fn diagnostic_rational_probe_rejects_non_linear_product_symbolic_slope_reciprocal_numerator() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("x^2/(2*a*x+b)", &mut ctx).expect("integrand");

    let candidate = try_algorithmic_integration_backend(
        &mut ctx,
        integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );

    assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Unsupported);
    assert_eq!(
        candidate.method_probe_no_match_reasons.first(),
        Some(&(
            AlgorithmicIntegrationMethod::Rational,
            AlgorithmicIntegrationProbeNoMatchReason::NumeratorPolicyMismatch,
        ))
    );
    assert!(!candidate.is_publicly_acceptable());
    assert_eq!(candidate.public_antiderivative(), None);
}

#[test]
fn diagnostic_rational_probe_rejects_variable_dependent_product_slope() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("1/(2*x*x+b)", &mut ctx).expect("integrand");

    let candidate = try_algorithmic_integration_backend(
        &mut ctx,
        integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );

    assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Unsupported);
    assert_eq!(
        candidate.method_probe_no_match_reasons.first(),
        Some(&(
            AlgorithmicIntegrationMethod::Rational,
            AlgorithmicIntegrationProbeNoMatchReason::DenominatorPolicyMismatch,
        ))
    );
    assert!(!candidate.is_publicly_acceptable());
    assert_eq!(candidate.public_antiderivative(), None);
}

#[test]
fn diagnostic_rational_probe_respects_verification_budget() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("1/(x+1)", &mut ctx).expect("integrand");
    let denominator = match ctx.get(integrand) {
        Expr::Div(_, denominator) => *denominator,
        _ => panic!("expected reciprocal integrand"),
    };
    let config = AlgorithmicIntegrationBackendConfig::diagnostic_only()
        .with_budget(AlgorithmicIntegrationBackendBudget::single_probe_without_verification());

    let candidate = try_algorithmic_integration_backend(&mut ctx, integrand, "x", config);

    assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Rational);
    assert!(candidate.antiderivative.is_some());
    assert_eq!(candidate.method_probe_budget_limit, 1);
    assert_eq!(candidate.verification_check_budget_limit, 0);
    assert_eq!(
        candidate.required_conditions,
        vec![ConditionPredicate::NonZero(denominator)]
    );
    assert_eq!(
        candidate.verification_status,
        AlgorithmicIntegrationVerificationStatus::Inconclusive
    );
    assert_eq!(
        candidate.residual_reason,
        Some(AlgorithmicIntegrationResidualReason::BudgetExceeded)
    );
    assert_eq!(
        candidate.verification_blocker,
        AlgorithmicIntegrationVerificationBlocker::BudgetExceeded
    );
    assert_eq!(candidate.method_probes_used, 1);
    assert_eq!(candidate.verification_checks_used, 0);
    assert!(!candidate.is_publicly_acceptable());
    assert_eq!(
        candidate.publication_status(),
        AlgorithmicIntegrationPublicationStatus::RejectedResidualReason
    );
    assert_eq!(
        candidate.fallback_status(config),
        AlgorithmicIntegrationFallbackStatus::BlockedByCandidatePolicy
    );
    assert_eq!(candidate.public_antiderivative(), None);
    assert_eq!(candidate.fallback_antiderivative(config), None);
}

#[test]
fn diagnostic_hermite_probe_verifies_after_rational_probe_miss() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("2*x/(x^2+1)", &mut ctx).expect("integrand");

    let candidate = try_algorithmic_integration_backend(
        &mut ctx,
        integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );
    assert_eq!(
        candidate.method,
        AlgorithmicIntegrationMethod::Hermite,
        "no-match reasons: {:?}",
        candidate.method_probe_no_match_reasons
    );
    assert_eq!(
        candidate.verification_status,
        AlgorithmicIntegrationVerificationStatus::Verified
    );
    assert_eq!(
        candidate.verification_evidence,
        AlgorithmicIntegrationVerificationEvidence::MethodSpecificDifferentiation
    );
    assert_eq!(candidate.residual_reason, None);
    assert!(candidate.required_conditions.is_empty());
    assert_eq!(
        candidate.trace_level,
        AlgorithmicIntegrationTraceLevel::AlgorithmicSummary
    );
    assert_eq!(candidate.method_probes_used, 2);
    assert_eq!(candidate.verification_checks_used, 1);
    assert!(candidate.public_antiderivative().is_some());
    assert_eq!(
        candidate.fallback_antiderivative(AlgorithmicIntegrationBackendConfig::diagnostic_only()),
        None
    );
}

#[test]
fn diagnostic_hermite_probe_verifies_positive_numeric_quadratic_shift() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("2*x/(x^2+2)", &mut ctx).expect("integrand");

    let candidate = try_algorithmic_integration_backend(
        &mut ctx,
        integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );
    assert_eq!(
        candidate.method,
        AlgorithmicIntegrationMethod::Hermite,
        "no-match reasons: {:?}",
        candidate.method_probe_no_match_reasons
    );
    assert_eq!(
        candidate.verification_status,
        AlgorithmicIntegrationVerificationStatus::Verified
    );
    assert_eq!(
        candidate.verification_evidence,
        AlgorithmicIntegrationVerificationEvidence::MethodSpecificDifferentiation
    );
    assert_eq!(
        candidate.verification_normalization_reason,
        AlgorithmicIntegrationVerificationNormalizationReason::None
    );
    assert_eq!(candidate.residual_reason, None);
    assert!(candidate.required_conditions.is_empty());
    assert_eq!(
        candidate.trace_level,
        AlgorithmicIntegrationTraceLevel::AlgorithmicSummary
    );
    assert!(candidate.public_antiderivative().is_some());
}

#[test]
fn diagnostic_hermite_probe_verifies_scaled_positive_quadratic_log_derivative() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("3*x/(x^2+2)", &mut ctx).expect("integrand");

    let candidate = try_algorithmic_integration_backend(
        &mut ctx,
        integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );
    assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Hermite);
    assert_eq!(
        candidate.verification_status,
        AlgorithmicIntegrationVerificationStatus::Verified
    );
    assert_eq!(
        candidate.verification_evidence,
        AlgorithmicIntegrationVerificationEvidence::MethodSpecificDifferentiation
    );
    assert_eq!(
        candidate.verification_normalization_reason,
        AlgorithmicIntegrationVerificationNormalizationReason::None
    );
    assert_eq!(candidate.residual_reason, None);
    assert!(candidate.required_conditions.is_empty());
    assert_eq!(
        candidate.trace_level,
        AlgorithmicIntegrationTraceLevel::AlgorithmicSummary
    );
    assert!(candidate.public_antiderivative().is_some());
}

#[test]
fn diagnostic_hermite_probe_verifies_symbolic_external_scale_log_derivative() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("a*x/(x^2+2)", &mut ctx).expect("integrand");

    let candidate = try_algorithmic_integration_backend(
        &mut ctx,
        integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );
    assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Hermite);
    assert_eq!(
        candidate.verification_status,
        AlgorithmicIntegrationVerificationStatus::Verified
    );
    assert_eq!(
        candidate.verification_evidence,
        AlgorithmicIntegrationVerificationEvidence::MethodSpecificDifferentiation
    );
    assert_eq!(
        candidate.verification_normalization_reason,
        AlgorithmicIntegrationVerificationNormalizationReason::None
    );
    assert_eq!(candidate.residual_reason, None);
    assert!(candidate.required_conditions.is_empty());
    assert_eq!(
        candidate.trace_level,
        AlgorithmicIntegrationTraceLevel::AlgorithmicSummary
    );
    assert!(candidate.public_antiderivative().is_some());
}

#[test]
fn diagnostic_hermite_probe_verifies_expanded_symbolic_slope_positive_quadratic_mixed_numerator() {
    let mut ctx = Context::new();
    let integrand =
        cas_parser::parse("(m*s*x+b*m+c)/(s^2*x^2+2*b*s*x+b^2+a)", &mut ctx).expect("integrand");
    let (numerator, denominator) = match ctx.get(integrand) {
        Expr::Div(numerator, denominator) => (*numerator, *denominator),
        _ => panic!("expected quotient integrand"),
    };
    let (variable_expr, variable_slope, radius_square, required_condition) =
        positive_shifted_quadratic_denominator_parts(&mut ctx, denominator, "x")
            .expect("expanded positive quadratic denominator");
    assert!(positive_radius_expr(&mut ctx, radius_square, &required_condition).is_some());
    let decomposition =
        linear_numerator_decomposition_terms(&mut ctx, numerator, variable_expr, "x");
    assert!(
        decomposition.is_some(),
        "failed to decompose numerator {} against {}",
        cas_formatter::DisplayExpr {
            context: &ctx,
            id: numerator
        },
        cas_formatter::DisplayExpr {
            context: &ctx,
            id: variable_expr
        }
    );
    let (variable_coefficient, constant_term) = decomposition.expect("decomposition");
    assert!(
        is_supported_backend_linear_coefficient_for_affine_slope(
            &ctx,
            variable_coefficient,
            "x",
            &variable_slope,
        ),
        "unsupported variable coefficient {}",
        cas_formatter::DisplayExpr {
            context: &ctx,
            id: variable_coefficient
        }
    );
    assert!(
        is_supported_backend_linear_coefficient_for_affine_slope(
            &ctx,
            constant_term,
            "x",
            &variable_slope,
        ),
        "unsupported constant term {}",
        cas_formatter::DisplayExpr {
            context: &ctx,
            id: constant_term
        }
    );

    let candidate = try_algorithmic_integration_backend(
        &mut ctx,
        integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );

    assert_eq!(
        candidate.method,
        AlgorithmicIntegrationMethod::Hermite,
        "no-match reasons: {:?}",
        candidate.method_probe_no_match_reasons
    );
    assert_eq!(
        candidate.verification_status,
        AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions,
        "blocker: {:?}, evidence: {:?}, normalization: {:?}, residual: {:?}",
        candidate.verification_blocker,
        candidate.verification_evidence,
        candidate.verification_normalization_reason,
        candidate.residual_reason
    );
    assert_eq!(candidate.residual_reason, None);
    assert_eq!(
        candidate.required_conditions.len(),
        2,
        "expected positive radius and symbolic slope conditions, got {:?}",
        candidate.required_conditions
    );
    assert!(candidate.public_antiderivative().is_some());
}

#[test]
fn diagnostic_hermite_probe_accepts_symbolic_scale_after_variable() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("x*a/(x^2+2)", &mut ctx).expect("integrand");

    let candidate = try_algorithmic_integration_backend(
        &mut ctx,
        integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );
    assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Hermite);
    assert_eq!(
        candidate.verification_status,
        AlgorithmicIntegrationVerificationStatus::Verified
    );
    assert_eq!(candidate.residual_reason, None);
    assert!(candidate.public_antiderivative().is_some());
}

#[test]
fn diagnostic_hermite_probe_verifies_unit_positive_quadratic_mixed_numerator() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("(x+1)/(x^2+1)", &mut ctx).expect("integrand");

    let candidate = try_algorithmic_integration_backend(
        &mut ctx,
        integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );
    assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Hermite);
    assert_eq!(
        candidate.verification_status,
        AlgorithmicIntegrationVerificationStatus::Verified
    );
    assert_eq!(candidate.residual_reason, None);
    assert!(candidate.required_conditions.is_empty());
    assert_eq!(
        candidate.method_probe_no_match_reasons,
        vec![(
            AlgorithmicIntegrationMethod::Rational,
            AlgorithmicIntegrationProbeNoMatchReason::NumeratorPolicyMismatch,
        )]
    );
    assert_eq!(candidate.method_probes_used, 2);
    assert_eq!(candidate.verification_checks_used, 1);
    assert!(candidate.public_antiderivative().is_some());
}

#[test]
fn diagnostic_hermite_probe_verifies_scaled_unit_positive_quadratic_mixed_numerator() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("(2*x+3)/(x^2+1)", &mut ctx).expect("integrand");

    let candidate = try_algorithmic_integration_backend(
        &mut ctx,
        integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );
    assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Hermite);
    assert_eq!(
        candidate.verification_status,
        AlgorithmicIntegrationVerificationStatus::Verified
    );
    assert_eq!(candidate.residual_reason, None);
    assert!(candidate.required_conditions.is_empty());
    assert_eq!(candidate.method_probes_used, 2);
    assert_eq!(candidate.verification_checks_used, 1);
    assert!(candidate.public_antiderivative().is_some());
}

#[test]
fn diagnostic_hermite_probe_verifies_exact_square_radius_positive_quadratic_mixed_numerator() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("(x+1)/(x^2+4)", &mut ctx).expect("integrand");

    let candidate = try_algorithmic_integration_backend(
        &mut ctx,
        integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );
    assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Hermite);
    assert_eq!(
        candidate.verification_status,
        AlgorithmicIntegrationVerificationStatus::Verified
    );
    assert_eq!(candidate.residual_reason, None);
    assert!(candidate.required_conditions.is_empty());
    assert_eq!(
        candidate.method_probe_no_match_reasons,
        vec![(
            AlgorithmicIntegrationMethod::Rational,
            AlgorithmicIntegrationProbeNoMatchReason::NumeratorPolicyMismatch,
        )]
    );
    assert_eq!(candidate.method_probes_used, 2);
    assert_eq!(candidate.verification_checks_used, 1);
    assert!(candidate.public_antiderivative().is_some());
}

#[test]
fn diagnostic_hermite_probe_verifies_scaled_exact_square_radius_mixed_numerator() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("(2*x+3)/(x^2+4)", &mut ctx).expect("integrand");

    let candidate = try_algorithmic_integration_backend(
        &mut ctx,
        integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );

    assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Hermite);
    assert_eq!(
        candidate.verification_status,
        AlgorithmicIntegrationVerificationStatus::Verified
    );
    assert_eq!(candidate.residual_reason, None);
    assert!(candidate.required_conditions.is_empty());
    assert_eq!(candidate.method_probes_used, 2);
    assert_eq!(candidate.verification_checks_used, 1);
    assert!(candidate.public_antiderivative().is_some());
}

#[test]
fn diagnostic_hermite_probe_verifies_symbolic_mixed_coefficients_with_verifier_policy() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("(a*x+b)/(x^2+4)", &mut ctx).expect("integrand");

    let candidate = try_algorithmic_integration_backend(
        &mut ctx,
        integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );

    assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Hermite);
    assert_eq!(
        candidate.verification_status,
        AlgorithmicIntegrationVerificationStatus::Verified
    );
    assert_eq!(
        candidate.verification_evidence,
        AlgorithmicIntegrationVerificationEvidence::MethodSpecificDifferentiation
    );
    assert_eq!(candidate.residual_reason, None);
    assert!(candidate.required_conditions.is_empty());
    assert_eq!(
        candidate.method_probe_no_match_reasons,
        vec![(
            AlgorithmicIntegrationMethod::Rational,
            AlgorithmicIntegrationProbeNoMatchReason::NumeratorPolicyMismatch,
        )]
    );
    assert_eq!(candidate.method_probes_used, 2);
    assert_eq!(candidate.verification_checks_used, 1);
    assert!(candidate.public_antiderivative().is_some());
}

#[test]
fn diagnostic_hermite_probe_verifies_constant_over_exact_square_radius() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("1/(x^2+4)", &mut ctx).expect("integrand");

    let candidate = try_algorithmic_integration_backend(
        &mut ctx,
        integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );

    assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Hermite);
    assert_eq!(
        candidate.verification_status,
        AlgorithmicIntegrationVerificationStatus::Verified
    );
    assert_eq!(
        candidate.verification_evidence,
        AlgorithmicIntegrationVerificationEvidence::MethodSpecificDifferentiation
    );
    assert_eq!(
        candidate.verification_normalization_reason,
        AlgorithmicIntegrationVerificationNormalizationReason::None
    );
    assert_eq!(candidate.residual_reason, None);
    assert!(candidate.required_conditions.is_empty());
    assert_eq!(
        candidate.method_probe_no_match_reasons,
        vec![(
            AlgorithmicIntegrationMethod::Rational,
            AlgorithmicIntegrationProbeNoMatchReason::DenominatorPolicyMismatch,
        )]
    );
    assert_eq!(candidate.method_probes_used, 2);
    assert_eq!(candidate.verification_checks_used, 1);
    assert!(candidate.public_antiderivative().is_some());
}

#[test]
fn diagnostic_hermite_probe_verifies_scaled_constant_over_exact_square_radius() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("3/(x^2+4)", &mut ctx).expect("integrand");

    let candidate = try_algorithmic_integration_backend(
        &mut ctx,
        integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );

    assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Hermite);
    assert_eq!(
        candidate.verification_status,
        AlgorithmicIntegrationVerificationStatus::Verified
    );
    assert_eq!(
        candidate.verification_evidence,
        AlgorithmicIntegrationVerificationEvidence::MethodSpecificDifferentiation
    );
    assert_eq!(
        candidate.verification_normalization_reason,
        AlgorithmicIntegrationVerificationNormalizationReason::None
    );
    assert_eq!(candidate.residual_reason, None);
    assert!(candidate.required_conditions.is_empty());
    assert_eq!(candidate.method_probes_used, 2);
    assert_eq!(candidate.verification_checks_used, 1);
    assert!(candidate.public_antiderivative().is_some());
}

#[test]
fn diagnostic_hermite_probe_verifies_constant_over_nonexact_numeric_radius() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("1/(x^2+2)", &mut ctx).expect("integrand");

    let candidate = try_algorithmic_integration_backend(
        &mut ctx,
        integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );

    assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Hermite);
    assert_eq!(
        candidate.verification_status,
        AlgorithmicIntegrationVerificationStatus::Verified
    );
    assert_eq!(
        candidate.verification_evidence,
        AlgorithmicIntegrationVerificationEvidence::MethodSpecificDifferentiation
    );
    assert_eq!(
        candidate.method_probe_no_match_reasons,
        vec![(
            AlgorithmicIntegrationMethod::Rational,
            AlgorithmicIntegrationProbeNoMatchReason::DenominatorPolicyMismatch,
        )]
    );
    assert_eq!(candidate.residual_reason, None);
    assert!(candidate.required_conditions.is_empty());
    assert_eq!(candidate.method_probes_used, 2);
    assert_eq!(candidate.verification_checks_used, 1);
    assert!(candidate.public_antiderivative().is_some());
}

#[test]
fn diagnostic_hermite_probe_verifies_scaled_constant_over_nonexact_numeric_radius() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("3/(x^2+2)", &mut ctx).expect("integrand");

    let candidate = try_algorithmic_integration_backend(
        &mut ctx,
        integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );

    assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Hermite);
    assert_eq!(
        candidate.verification_status,
        AlgorithmicIntegrationVerificationStatus::Verified
    );
    assert_eq!(
        candidate.verification_evidence,
        AlgorithmicIntegrationVerificationEvidence::MethodSpecificDifferentiation
    );
    assert_eq!(candidate.residual_reason, None);
    assert!(candidate.required_conditions.is_empty());
    assert_eq!(candidate.method_probes_used, 2);
    assert_eq!(candidate.verification_checks_used, 1);
    assert!(candidate.public_antiderivative().is_some());
}

#[test]
fn diagnostic_hermite_probe_verifies_nonexact_radius_mixed_numerator() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("(x+1)/(x^2+2)", &mut ctx).expect("integrand");

    let candidate = try_algorithmic_integration_backend(
        &mut ctx,
        integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );

    assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Hermite);
    assert_eq!(
        candidate.verification_status,
        AlgorithmicIntegrationVerificationStatus::Verified
    );
    assert_eq!(
        candidate.verification_evidence,
        AlgorithmicIntegrationVerificationEvidence::MethodSpecificDifferentiation
    );
    assert_eq!(
        candidate.method_probe_no_match_reasons,
        vec![(
            AlgorithmicIntegrationMethod::Rational,
            AlgorithmicIntegrationProbeNoMatchReason::NumeratorPolicyMismatch,
        )]
    );
    assert_eq!(candidate.residual_reason, None);
    assert!(candidate.required_conditions.is_empty());
    assert_eq!(candidate.method_probes_used, 2);
    assert_eq!(candidate.verification_checks_used, 1);
    assert!(candidate.public_antiderivative().is_some());
}

#[test]
fn diagnostic_hermite_probe_verifies_symbolic_positive_shift_log_derivative() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("2*x/(x^2+a)", &mut ctx).expect("integrand");
    let radius_square = cas_parser::parse("a", &mut ctx).expect("radius square");

    let candidate = try_algorithmic_integration_backend(
        &mut ctx,
        integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );

    assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Hermite);
    assert_eq!(
        candidate.verification_status,
        AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
    );
    assert_eq!(
        candidate.required_conditions,
        vec![ConditionPredicate::Positive(radius_square)]
    );
    assert_eq!(
        candidate.method_probe_no_match_reasons,
        vec![(
            AlgorithmicIntegrationMethod::Rational,
            AlgorithmicIntegrationProbeNoMatchReason::NumeratorPolicyMismatch,
        )]
    );
    assert_eq!(candidate.residual_reason, None);
    assert_eq!(candidate.method_probes_used, 2);
    assert_eq!(candidate.verification_checks_used, 1);
    assert!(candidate.public_antiderivative().is_some());
}

#[test]
fn diagnostic_hermite_probe_verifies_constant_over_symbolic_positive_radius() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("1/(x^2+a)", &mut ctx).expect("integrand");
    let radius_square = cas_parser::parse("a", &mut ctx).expect("radius square");

    let candidate = try_algorithmic_integration_backend(
        &mut ctx,
        integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );

    assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Hermite);
    assert_eq!(
        candidate.verification_status,
        AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
    );
    assert_eq!(
        candidate.verification_evidence,
        AlgorithmicIntegrationVerificationEvidence::MethodSpecificDifferentiation
    );
    assert_eq!(
        candidate.verification_normalization_reason,
        AlgorithmicIntegrationVerificationNormalizationReason::None
    );
    assert_eq!(
        candidate.required_conditions,
        vec![ConditionPredicate::Positive(radius_square)]
    );
    assert_eq!(
        candidate.method_probe_no_match_reasons,
        vec![(
            AlgorithmicIntegrationMethod::Rational,
            AlgorithmicIntegrationProbeNoMatchReason::DenominatorPolicyMismatch,
        )]
    );
    assert_eq!(candidate.residual_reason, None);
    assert_eq!(candidate.method_probes_used, 2);
    assert_eq!(candidate.verification_checks_used, 1);
    assert!(candidate.public_antiderivative().is_some());
}

#[test]
fn diagnostic_hermite_probe_verifies_symbolic_constant_over_symbolic_positive_radius() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("c/(x^2+a)", &mut ctx).expect("integrand");
    let radius_square = cas_parser::parse("a", &mut ctx).expect("radius square");

    let candidate = try_algorithmic_integration_backend(
        &mut ctx,
        integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );

    assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Hermite);
    assert_eq!(
        candidate.verification_status,
        AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
    );
    assert_eq!(
        candidate.verification_evidence,
        AlgorithmicIntegrationVerificationEvidence::MethodSpecificDifferentiation
    );
    assert_eq!(
        candidate.verification_normalization_reason,
        AlgorithmicIntegrationVerificationNormalizationReason::None
    );
    assert_eq!(
        candidate.required_conditions,
        vec![ConditionPredicate::Positive(radius_square)]
    );
    assert_eq!(
        candidate.method_probe_no_match_reasons,
        vec![(
            AlgorithmicIntegrationMethod::Rational,
            AlgorithmicIntegrationProbeNoMatchReason::DenominatorPolicyMismatch,
        )]
    );
    assert_eq!(candidate.residual_reason, None);
    assert_eq!(candidate.method_probes_used, 2);
    assert_eq!(candidate.verification_checks_used, 1);
    assert!(candidate.public_antiderivative().is_some());
}

#[test]
fn diagnostic_hermite_probe_verifies_symbolic_positive_shift_mixed_numerator() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("(x+1)/(x^2+a)", &mut ctx).expect("integrand");
    let radius_square = cas_parser::parse("a", &mut ctx).expect("radius square");

    let candidate = try_algorithmic_integration_backend(
        &mut ctx,
        integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );

    assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Hermite);
    assert_eq!(
        candidate.verification_status,
        AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
    );
    assert_eq!(
        candidate.verification_evidence,
        AlgorithmicIntegrationVerificationEvidence::MethodSpecificDifferentiation
    );
    assert_eq!(
        candidate.verification_normalization_reason,
        AlgorithmicIntegrationVerificationNormalizationReason::None
    );
    assert_eq!(
        candidate.required_conditions,
        vec![ConditionPredicate::Positive(radius_square)]
    );
    assert_eq!(
        candidate.method_probe_no_match_reasons,
        vec![(
            AlgorithmicIntegrationMethod::Rational,
            AlgorithmicIntegrationProbeNoMatchReason::NumeratorPolicyMismatch,
        )]
    );
    assert_eq!(candidate.residual_reason, None);
    assert_eq!(candidate.method_probes_used, 2);
    assert_eq!(candidate.verification_checks_used, 1);
    assert!(candidate.public_antiderivative().is_some());
}

#[test]
fn diagnostic_hermite_probe_verifies_symbolic_positive_shift_external_mixed_numerator() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("(b*x+c)/(x^2+a)", &mut ctx).expect("integrand");
    let radius_square = cas_parser::parse("a", &mut ctx).expect("radius square");

    let candidate = try_algorithmic_integration_backend(
        &mut ctx,
        integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );

    assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Hermite);
    assert_eq!(
        candidate.verification_status,
        AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
    );
    assert_eq!(
        candidate.verification_evidence,
        AlgorithmicIntegrationVerificationEvidence::MethodSpecificDifferentiation
    );
    assert_eq!(
        candidate.required_conditions,
        vec![ConditionPredicate::Positive(radius_square)]
    );
    assert_eq!(
        candidate.method_probe_no_match_reasons,
        vec![(
            AlgorithmicIntegrationMethod::Rational,
            AlgorithmicIntegrationProbeNoMatchReason::NumeratorPolicyMismatch,
        )]
    );
    assert_eq!(candidate.residual_reason, None);
    assert_eq!(candidate.method_probes_used, 2);
    assert_eq!(candidate.verification_checks_used, 1);
    assert!(candidate.public_antiderivative().is_some());
}

#[test]
fn diagnostic_hermite_probe_verifies_shifted_affine_symbolic_constant_over_positive_radius() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("c/((x+b)^2+a)", &mut ctx).expect("integrand");
    let radius_square = cas_parser::parse("a", &mut ctx).expect("radius square");

    let candidate = try_algorithmic_integration_backend(
        &mut ctx,
        integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );

    assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Hermite);
    assert_eq!(
        candidate.verification_status,
        AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
    );
    assert_eq!(
        candidate.verification_evidence,
        AlgorithmicIntegrationVerificationEvidence::MethodSpecificDifferentiation
    );
    assert_eq!(
        candidate.required_conditions,
        vec![ConditionPredicate::Positive(radius_square)]
    );
    assert_eq!(
        candidate.method_probe_no_match_reasons,
        vec![(
            AlgorithmicIntegrationMethod::Rational,
            AlgorithmicIntegrationProbeNoMatchReason::DenominatorPolicyMismatch,
        )]
    );
    assert_eq!(candidate.residual_reason, None);
    assert_eq!(candidate.method_probes_used, 2);
    assert_eq!(candidate.verification_checks_used, 1);
    assert!(candidate.public_antiderivative().is_some());
}

#[test]
fn diagnostic_hermite_probe_verifies_shifted_affine_external_mixed_numerator() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("(m*(x+b)+c)/((x+b)^2+a)", &mut ctx).expect("integrand");
    let radius_square = cas_parser::parse("a", &mut ctx).expect("radius square");

    let candidate = try_algorithmic_integration_backend(
        &mut ctx,
        integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );

    assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Hermite);
    assert_eq!(
        candidate.verification_status,
        AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
    );
    assert_eq!(
        candidate.verification_evidence,
        AlgorithmicIntegrationVerificationEvidence::MethodSpecificDifferentiation
    );
    assert_eq!(
        candidate.required_conditions,
        vec![ConditionPredicate::Positive(radius_square)]
    );
    assert_eq!(
        candidate.method_probe_no_match_reasons,
        vec![(
            AlgorithmicIntegrationMethod::Rational,
            AlgorithmicIntegrationProbeNoMatchReason::NumeratorPolicyMismatch,
        )]
    );
    assert_eq!(candidate.residual_reason, None);
    assert_eq!(candidate.method_probes_used, 2);
    assert_eq!(candidate.verification_checks_used, 1);
    assert!(candidate.public_antiderivative().is_some());
}

#[test]
fn diagnostic_hermite_probe_verifies_numeric_slope_shifted_affine_constant_over_positive_radius() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("1/((2*x+b)^2+a)", &mut ctx).expect("integrand");
    let radius_square = cas_parser::parse("a", &mut ctx).expect("radius square");

    let candidate = try_algorithmic_integration_backend(
        &mut ctx,
        integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );

    assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Hermite);
    assert_eq!(
        candidate.verification_status,
        AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
    );
    assert_eq!(
        candidate.verification_evidence,
        AlgorithmicIntegrationVerificationEvidence::MethodSpecificDifferentiation
    );
    assert_eq!(
        candidate.required_conditions,
        vec![ConditionPredicate::Positive(radius_square)]
    );
    assert_eq!(
        candidate.method_probe_no_match_reasons,
        vec![(
            AlgorithmicIntegrationMethod::Rational,
            AlgorithmicIntegrationProbeNoMatchReason::DenominatorPolicyMismatch,
        )]
    );
    assert_eq!(candidate.residual_reason, None);
    assert_eq!(candidate.method_probes_used, 2);
    assert_eq!(candidate.verification_checks_used, 1);
    assert!(candidate.public_antiderivative().is_some());
}

#[test]
fn diagnostic_hermite_probe_verifies_negative_numeric_slope_shifted_affine_constant() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("1/((b-2*x)^2+a)", &mut ctx).expect("integrand");
    let radius_square = cas_parser::parse("a", &mut ctx).expect("radius square");

    let candidate = try_algorithmic_integration_backend(
        &mut ctx,
        integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );

    assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Hermite);
    assert_eq!(
        candidate.verification_status,
        AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
    );
    assert_eq!(
        candidate.required_conditions,
        vec![ConditionPredicate::Positive(radius_square)]
    );
    assert_eq!(candidate.residual_reason, None);
    assert!(candidate.public_antiderivative().is_some());
}

#[test]
fn diagnostic_hermite_probe_verifies_numeric_slope_shifted_affine_external_mixed_numerator() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("(m*(2*x+b)+c)/((2*x+b)^2+a)", &mut ctx).expect("integrand");
    let radius_square = cas_parser::parse("a", &mut ctx).expect("radius square");

    let candidate = try_algorithmic_integration_backend(
        &mut ctx,
        integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );

    assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Hermite);
    assert_eq!(
        candidate.verification_status,
        AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
    );
    assert_eq!(
        candidate.verification_evidence,
        AlgorithmicIntegrationVerificationEvidence::MethodSpecificDifferentiation
    );
    assert_eq!(
        candidate.required_conditions,
        vec![ConditionPredicate::Positive(radius_square)]
    );
    assert_eq!(
        candidate.method_probe_no_match_reasons,
        vec![(
            AlgorithmicIntegrationMethod::Rational,
            AlgorithmicIntegrationProbeNoMatchReason::NumeratorPolicyMismatch,
        )]
    );
    assert_eq!(candidate.residual_reason, None);
    assert_eq!(candidate.method_probes_used, 2);
    assert_eq!(candidate.verification_checks_used, 1);
    assert!(candidate.public_antiderivative().is_some());
}

#[test]
fn diagnostic_hermite_probe_verifies_symbolic_slope_shifted_affine_mixed_numerator() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("(m*(s*x+b)+c)/((s*x+b)^2+a)", &mut ctx).expect("integrand");
    let radius_square = cas_parser::parse("a", &mut ctx).expect("radius square");
    let slope = cas_parser::parse("s", &mut ctx).expect("slope");

    let candidate = try_algorithmic_integration_backend(
        &mut ctx,
        integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );
    assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Hermite);
    assert_eq!(
        candidate.verification_status,
        AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
    );
    assert_eq!(
        candidate.verification_evidence,
        AlgorithmicIntegrationVerificationEvidence::MethodSpecificDifferentiation
    );
    assert_eq!(
        candidate.verification_normalization_reason,
        AlgorithmicIntegrationVerificationNormalizationReason::None
    );
    assert_eq!(
        candidate.required_conditions,
        vec![
            ConditionPredicate::Positive(radius_square),
            ConditionPredicate::NonZero(slope),
        ]
    );
    assert_eq!(
        candidate.method_probe_no_match_reasons,
        vec![(
            AlgorithmicIntegrationMethod::Rational,
            AlgorithmicIntegrationProbeNoMatchReason::NumeratorPolicyMismatch,
        ),]
    );
    assert_eq!(candidate.residual_reason, None);
    assert_eq!(candidate.method_probes_used, 2);
    assert_eq!(candidate.verification_checks_used, 1);
    assert!(candidate.public_antiderivative().is_some());
}

#[test]
fn backend_verifier_cancels_generated_symbolic_slope_quotient_factors() {
    let mut ctx = Context::new();
    let quotient =
        cas_parser::parse("((s*x+b)*m*s*2)/(s*2)", &mut ctx).expect("generated quotient");
    let expected = cas_parser::parse("m*(s*x+b)", &mut ctx).expect("expected quotient");
    let slope = cas_parser::parse("s", &mut ctx).expect("slope");
    let Expr::Div(numerator, denominator) = ctx.get(quotient).clone() else {
        panic!("expected quotient");
    };

    let normalized = normalize_backend_common_factor_quotient(
        &mut ctx,
        numerator,
        denominator,
        "x",
        &[ConditionPredicate::NonZero(slope)],
    )
    .expect("common factor cancellation");

    assert!(
        normalized == expected
            || SemanticEqualityChecker::new(&ctx).are_equal(normalized, expected),
        "expected {}, got {}",
        cas_formatter::DisplayExpr {
            context: &ctx,
            id: expected
        },
        cas_formatter::DisplayExpr {
            context: &ctx,
            id: normalized
        }
    );
}

#[test]
fn backend_verifier_normalizes_generated_symbolic_slope_recombined_numerator() {
    let mut ctx = Context::new();
    let numerator =
        cas_parser::parse("((s*x+b)*m*s*2)/(s*2)+c", &mut ctx).expect("generated numerator");
    let expected = cas_parser::parse("m*(s*x+b)+c", &mut ctx).expect("expected numerator");
    let slope = cas_parser::parse("s", &mut ctx).expect("slope");

    let normalized = normalize_backend_verification_expr(
        &mut ctx,
        numerator,
        "x",
        &[ConditionPredicate::NonZero(slope)],
    )
    .expect("recombined numerator normalization");

    assert!(
        normalized.expr == expected
            || SemanticEqualityChecker::new(&ctx).are_equal(normalized.expr, expected),
        "expected {}, got {}",
        cas_formatter::DisplayExpr {
            context: &ctx,
            id: expected
        },
        cas_formatter::DisplayExpr {
            context: &ctx,
            id: normalized.expr
        }
    );
}

#[test]
fn backend_verifier_normalizes_generated_symbolic_slope_derivative() {
    let mut ctx = Context::new();
    let derivative = cas_parser::parse(
        "(m*s*(s*x+b)^(2-1)*2)/(s*2*((s*x+b)^2+a)) + (c*s)/(sqrt(a)*s*(((s*x+b)/sqrt(a))^2+1)*sqrt(a))",
        &mut ctx,
    )
    .expect("generated derivative");
    let expected =
        cas_parser::parse("(m*(s*x+b)+c)/((s*x+b)^2+a)", &mut ctx).expect("expected derivative");
    let radius_square = cas_parser::parse("a", &mut ctx).expect("radius square");
    let slope = cas_parser::parse("s", &mut ctx).expect("slope");

    let normalized = normalize_backend_verification_expr(
        &mut ctx,
        derivative,
        "x",
        &[
            ConditionPredicate::Positive(radius_square),
            ConditionPredicate::NonZero(slope),
        ],
    )
    .expect("generated derivative normalization");

    assert!(
        normalized.expr == expected
            || SemanticEqualityChecker::new(&ctx).are_equal(normalized.expr, expected),
        "expected {}, got {} [{:?}]",
        cas_formatter::DisplayExpr {
            context: &ctx,
            id: expected
        },
        cas_formatter::DisplayExpr {
            context: &ctx,
            id: normalized.expr
        },
        normalized.reason
    );
}

#[test]
fn backend_verifier_normalizes_generated_symbolic_slope_intermediate_quotient() {
    let mut ctx = Context::new();
    let intermediate = cas_parser::parse("(((s*x+b)*m*s*2)/(s*2)+c)/((s*x+b)^2+a)", &mut ctx)
        .expect("intermediate quotient");
    let expected =
        cas_parser::parse("(m*(s*x+b)+c)/((s*x+b)^2+a)", &mut ctx).expect("expected quotient");
    let radius_square = cas_parser::parse("a", &mut ctx).expect("radius square");
    let slope = cas_parser::parse("s", &mut ctx).expect("slope");

    let normalized = normalize_backend_verification_expr(
        &mut ctx,
        intermediate,
        "x",
        &[
            ConditionPredicate::Positive(radius_square),
            ConditionPredicate::NonZero(slope),
        ],
    )
    .expect("intermediate quotient normalization");

    assert!(
        normalized.expr == expected
            || SemanticEqualityChecker::new(&ctx).are_equal(normalized.expr, expected),
        "expected {}, got {} [{:?}]",
        cas_formatter::DisplayExpr {
            context: &ctx,
            id: expected
        },
        cas_formatter::DisplayExpr {
            context: &ctx,
            id: normalized.expr
        },
        normalized.reason
    );
}

#[test]
fn backend_verifier_fixed_point_normalizes_generated_symbolic_slope_derivative_ast() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("(m*(s*x+b)+c)/((s*x+b)^2+a)", &mut ctx).expect("integrand");
    let antiderivative = cas_parser::parse(
        "(c*arctan((s*x+b)/sqrt(a)))/(sqrt(a)*s) + (m*ln((s*x+b)^2+a))/(s*2)",
        &mut ctx,
    )
    .expect("antiderivative");
    let radius_square = cas_parser::parse("a", &mut ctx).expect("radius square");
    let slope = cas_parser::parse("s", &mut ctx).expect("slope");
    let derivative =
        differentiate_symbolic_expr(&mut ctx, antiderivative, "x").expect("derivative");

    let matched = normalize_backend_verification_expr_to_match(
        &mut ctx,
        derivative,
        integrand,
        "x",
        &[
            ConditionPredicate::Positive(radius_square),
            ConditionPredicate::NonZero(slope),
        ],
    );

    assert!(matched.matched_reason.is_some());
    assert!(matched.passes_used > 1);
}

#[test]
fn backend_verifier_normalizes_indefinite_square_log_difference_derivative() {
    let mut ctx = Context::new();
    let derivative =
        cas_parser::parse("((1/(x-a)-1/(x+a))*1*1/2)/a", &mut ctx).expect("derivative");
    let expected = cas_parser::parse("1/(x^2-a^2)", &mut ctx).expect("expected");
    let radius = cas_parser::parse("a", &mut ctx).expect("radius");
    let left_pole = cas_parser::parse("x-a", &mut ctx).expect("left pole");
    let right_pole = cas_parser::parse("x+a", &mut ctx).expect("right pole");

    let matched = normalize_backend_verification_expr_to_match(
        &mut ctx,
        derivative,
        expected,
        "x",
        &[
            ConditionPredicate::NonZero(radius),
            ConditionPredicate::NonZero(left_pole),
            ConditionPredicate::NonZero(right_pole),
        ],
    );

    assert!(matched.matched_reason.is_some());
}

#[test]
fn backend_verifier_combines_conjugate_reciprocal_difference() {
    let mut ctx = Context::new();
    let left = cas_parser::parse("1/(x-a)", &mut ctx).expect("left reciprocal");
    let right = cas_parser::parse("1/(a+x)", &mut ctx).expect("right reciprocal");

    let combined = normalize_backend_conjugate_reciprocal_difference(&mut ctx, left, right, "x")
        .expect("conjugate reciprocal difference");
    let expected = cas_parser::parse("2*a/((x-a)*(a+x))", &mut ctx).expect("expected");

    assert!(
        combined == expected || SemanticEqualityChecker::new(&ctx).are_equal(combined, expected),
        "expected {}, got {}",
        cas_formatter::DisplayExpr {
            context: &ctx,
            id: expected
        },
        cas_formatter::DisplayExpr {
            context: &ctx,
            id: combined
        }
    );
}

#[test]
fn backend_verifier_normalizes_generated_indefinite_square_derivative_ast() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("1/(x^2-a^2)", &mut ctx).expect("integrand");
    let generated_candidate = try_algorithmic_integration_backend(
        &mut ctx,
        integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );
    let antiderivative = generated_candidate
        .antiderivative
        .expect("generated antiderivative");
    let derivative =
        differentiate_symbolic_expr(&mut ctx, antiderivative, "x").expect("derivative");
    let matched = normalize_backend_verification_expr_to_match(
        &mut ctx,
        derivative,
        integrand,
        "x",
        &generated_candidate.required_conditions,
    );

    assert!(
        matched.matched_reason.is_some(),
        "expected generated derivative normalization, got {:?} in {} passes",
        matched.matched_reason,
        matched.passes_used
    );
}

#[test]
fn backend_verifier_normalizes_generated_symbolic_slope_indefinite_square_derivative_ast() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("1/((s*x+b)^2-a^2)", &mut ctx).expect("integrand");
    let generated_candidate = try_algorithmic_integration_backend(
        &mut ctx,
        integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );
    let antiderivative = generated_candidate
        .antiderivative
        .expect("generated antiderivative");
    let derivative =
        differentiate_symbolic_expr(&mut ctx, antiderivative, "x").expect("derivative");
    let matched = normalize_backend_verification_expr_to_match(
        &mut ctx,
        derivative,
        integrand,
        "x",
        &generated_candidate.required_conditions,
    );

    assert!(
        matched.matched_reason.is_some(),
        "expected generated symbolic-slope derivative normalization, got {:?} in {} passes",
        matched.matched_reason,
        matched.passes_used
    );
}

#[test]
fn diagnostic_hermite_probe_rejects_variable_dependent_affine_square_slope() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("1/((x*x+b)^2+a)", &mut ctx).expect("integrand");

    let candidate = try_algorithmic_integration_backend(
        &mut ctx,
        integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );

    assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Unsupported);
    assert_eq!(candidate.antiderivative, None);
    assert_eq!(
        candidate.residual_reason,
        Some(AlgorithmicIntegrationResidualReason::UnsupportedMethod)
    );
    assert_eq!(
        candidate.method_probe_no_match_reasons,
        vec![
            (
                AlgorithmicIntegrationMethod::Rational,
                AlgorithmicIntegrationProbeNoMatchReason::DenominatorPolicyMismatch,
            ),
            (
                AlgorithmicIntegrationMethod::Hermite,
                AlgorithmicIntegrationProbeNoMatchReason::DenominatorPolicyMismatch,
            ),
            (
                AlgorithmicIntegrationMethod::HeurischProbe,
                AlgorithmicIntegrationProbeNoMatchReason::ShapeMismatch,
            ),
        ]
    );
    assert!(!candidate.is_publicly_acceptable());
}

#[test]
fn verification_report_requires_positive_condition_for_symbolic_radius_square() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("1/(x^2+a)", &mut ctx).expect("integrand");
    let generated_candidate = try_algorithmic_integration_backend(
        &mut ctx,
        integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );
    let antiderivative = generated_candidate
        .antiderivative
        .expect("generated antiderivative");
    let mut unconditioned_candidate = AlgorithmicIntegrationCandidate::unverified(
        integrand,
        "x",
        antiderivative,
        AlgorithmicIntegrationMethod::Hermite,
    );

    let unconditioned_report =
        antiderivative_verification_report(&mut ctx, &unconditioned_candidate);

    assert_ne!(
        unconditioned_report.status,
        AlgorithmicIntegrationVerificationStatus::Verified
    );
    assert_ne!(
        unconditioned_report.status,
        AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
    );

    unconditioned_candidate.required_conditions = generated_candidate.required_conditions.clone();
    let conditioned_report = antiderivative_verification_report(&mut ctx, &unconditioned_candidate);

    assert_eq!(
        conditioned_report.status,
        AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
    );
    assert_eq!(
        conditioned_report.evidence,
        AlgorithmicIntegrationVerificationEvidence::MethodSpecificDifferentiation
    );
    assert_eq!(
        conditioned_report.normalization_reason,
        AlgorithmicIntegrationVerificationNormalizationReason::None
    );
    assert_eq!(conditioned_report.residual_reason, None);
}

#[test]
fn diagnostic_hermite_probe_rejects_variable_dependent_scale() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("x*x/(x^2+2)", &mut ctx).expect("integrand");

    let candidate = try_algorithmic_integration_backend(
        &mut ctx,
        integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );

    assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Unsupported);
    assert_eq!(candidate.antiderivative, None);
    assert_eq!(
        candidate.residual_reason,
        Some(AlgorithmicIntegrationResidualReason::UnsupportedMethod)
    );
    assert_eq!(candidate.method_probes_used, 3);
    assert_eq!(candidate.verification_checks_used, 0);
    assert!(!candidate.is_publicly_acceptable());
}

#[test]
fn diagnostic_hermite_probe_verifies_indefinite_quadratic_log_derivative() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("2*x/(x^2-1)", &mut ctx).expect("integrand");
    let left_pole = cas_parser::parse("x-1", &mut ctx).expect("left pole");
    let right_pole = cas_parser::parse("x+1", &mut ctx).expect("right pole");

    let candidate = try_algorithmic_integration_backend(
        &mut ctx,
        integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );

    assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Hermite);
    assert!(candidate.antiderivative.is_some());
    assert_eq!(
        candidate.verification_status,
        AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
    );
    assert_eq!(
        candidate.verification_evidence,
        AlgorithmicIntegrationVerificationEvidence::DirectDifferentiation
    );
    assert_eq!(candidate.residual_reason, None);
    assert_eq!(
        candidate.constant_policy,
        IntegrationConstantPolicy::ComponentLocalConstant
    );
    assert_eq!(
        candidate.required_conditions,
        vec![
            ConditionPredicate::NonZero(left_pole),
            ConditionPredicate::NonZero(right_pole),
        ]
    );
    assert_eq!(
        candidate.method_probe_no_match_reasons,
        vec![(
            AlgorithmicIntegrationMethod::Rational,
            AlgorithmicIntegrationProbeNoMatchReason::NumeratorPolicyMismatch,
        ),]
    );
    assert_eq!(candidate.method_probes_used, 2);
    assert_eq!(candidate.verification_checks_used, 1);
    assert_eq!(candidate.method_probe_budget_limit, 3);
    assert_eq!(candidate.verification_check_budget_limit, 1);
    assert!(candidate.is_publicly_acceptable());
}

#[test]
fn diagnostic_hermite_probe_verifies_symbolic_square_radius_log_derivative() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("2*x/(x^2+a^2)", &mut ctx).expect("integrand");

    let candidate = try_algorithmic_integration_backend(
        &mut ctx,
        integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );

    assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Hermite);
    assert!(candidate.antiderivative.is_some());
    assert_eq!(
        candidate.verification_status,
        AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
    );
    assert_eq!(
        candidate.verification_evidence,
        AlgorithmicIntegrationVerificationEvidence::MethodSpecificDifferentiation
    );
    assert_eq!(candidate.residual_reason, None);
    assert_eq!(candidate.required_conditions.len(), 1);
    assert!(matches!(
        candidate.required_conditions[0],
        ConditionPredicate::NonZero(_)
    ));
    assert_eq!(
        candidate.method_probe_no_match_reasons,
        vec![(
            AlgorithmicIntegrationMethod::Rational,
            AlgorithmicIntegrationProbeNoMatchReason::NumeratorPolicyMismatch,
        ),]
    );
    assert_eq!(candidate.method_probes_used, 2);
    assert_eq!(candidate.verification_checks_used, 1);
    assert!(candidate.is_publicly_acceptable());
}

#[test]
fn diagnostic_hermite_probe_verifies_symbolic_square_radius_arctan_branch() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("1/(x^2+a^2)", &mut ctx).expect("integrand");

    let candidate = try_algorithmic_integration_backend(
        &mut ctx,
        integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );

    assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Hermite);
    assert!(candidate.antiderivative.is_some());
    assert_eq!(
        candidate.verification_status,
        AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
    );
    assert_eq!(
        candidate.verification_evidence,
        AlgorithmicIntegrationVerificationEvidence::MethodSpecificDifferentiation
    );
    assert_eq!(
        candidate.verification_normalization_reason,
        AlgorithmicIntegrationVerificationNormalizationReason::None
    );
    assert_eq!(candidate.residual_reason, None);
    assert_eq!(candidate.required_conditions.len(), 1);
    assert!(matches!(
        candidate.required_conditions[0],
        ConditionPredicate::NonZero(_)
    ));
    assert_eq!(
        candidate.method_probe_no_match_reasons,
        vec![(
            AlgorithmicIntegrationMethod::Rational,
            AlgorithmicIntegrationProbeNoMatchReason::DenominatorPolicyMismatch,
        ),]
    );
    assert_eq!(candidate.method_probes_used, 2);
    assert_eq!(candidate.verification_checks_used, 1);
    assert!(candidate.is_publicly_acceptable());
}

#[test]
fn diagnostic_hermite_probe_verifies_symbolic_square_radius_mixed_numerator() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("(x+1)/(x^2+a^2)", &mut ctx).expect("integrand");

    let candidate = try_algorithmic_integration_backend(
        &mut ctx,
        integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );

    assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Hermite);
    assert_eq!(
        candidate.verification_status,
        AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
    );
    assert_eq!(candidate.residual_reason, None);
    assert_eq!(candidate.required_conditions.len(), 1);
    assert!(matches!(
        candidate.required_conditions[0],
        ConditionPredicate::NonZero(_)
    ));
    assert_eq!(candidate.method_probes_used, 2);
    assert_eq!(candidate.verification_checks_used, 1);
    assert!(candidate.is_publicly_acceptable());
}

#[test]
fn diagnostic_hermite_probe_verifies_symbolic_slope_shifted_symbolic_square_radius_mixed_numerator()
{
    let mut ctx = Context::new();
    let integrand =
        cas_parser::parse("(m*(s*x+b)+c)/((s*x+b)^2+a^2)", &mut ctx).expect("integrand");
    let radius = cas_parser::parse("a", &mut ctx).expect("radius");
    let slope = cas_parser::parse("s", &mut ctx).expect("slope");

    let candidate = try_algorithmic_integration_backend(
        &mut ctx,
        integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );

    assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Hermite);
    assert_eq!(
        candidate.verification_status,
        AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
    );
    assert_eq!(
        candidate.verification_evidence,
        AlgorithmicIntegrationVerificationEvidence::MethodSpecificDifferentiation
    );
    assert_eq!(candidate.residual_reason, None);
    assert_eq!(
        candidate.required_conditions,
        vec![
            ConditionPredicate::NonZero(radius),
            ConditionPredicate::NonZero(slope),
        ]
    );
    assert_eq!(
        candidate.method_probe_no_match_reasons,
        vec![(
            AlgorithmicIntegrationMethod::Rational,
            AlgorithmicIntegrationProbeNoMatchReason::NumeratorPolicyMismatch,
        ),]
    );
    assert_eq!(candidate.method_probes_used, 2);
    assert_eq!(candidate.verification_checks_used, 1);
    assert!(candidate.is_publicly_acceptable());
}

#[test]
fn diagnostic_hermite_probe_verifies_indefinite_symbolic_square_denominator() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("1/(x^2-a^2)", &mut ctx).expect("integrand");
    let radius = cas_parser::parse("a", &mut ctx).expect("radius");
    let left_pole = cas_parser::parse("x-a", &mut ctx).expect("left pole");
    let right_pole = cas_parser::parse("x+a", &mut ctx).expect("right pole");

    let candidate = try_algorithmic_integration_backend(
        &mut ctx,
        integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );

    assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Hermite);
    assert!(candidate.antiderivative.is_some());
    assert_eq!(
        candidate.verification_status,
        AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
    );
    assert_eq!(
        candidate.verification_evidence,
        AlgorithmicIntegrationVerificationEvidence::MethodSpecificDifferentiation
    );
    assert_eq!(candidate.residual_reason, None);
    assert_eq!(
        candidate.constant_policy,
        IntegrationConstantPolicy::ComponentLocalConstant
    );
    assert_eq!(
        candidate.required_conditions,
        vec![
            ConditionPredicate::NonZero(radius),
            ConditionPredicate::NonZero(left_pole),
            ConditionPredicate::NonZero(right_pole),
        ]
    );
    assert_eq!(
        candidate.method_probe_no_match_reasons,
        vec![(
            AlgorithmicIntegrationMethod::Rational,
            AlgorithmicIntegrationProbeNoMatchReason::DenominatorPolicyMismatch,
        ),]
    );
    assert_eq!(candidate.method_probes_used, 2);
    assert_eq!(candidate.verification_checks_used, 1);
    assert!(candidate.is_publicly_acceptable());
}

#[test]
fn diagnostic_hermite_probe_verifies_symbolic_slope_indefinite_square_denominator() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("1/((s*x+b)^2-a^2)", &mut ctx).expect("integrand");
    let radius = cas_parser::parse("a", &mut ctx).expect("radius");
    let left_pole = cas_parser::parse("s*x+b-a", &mut ctx).expect("left pole");
    let right_pole = cas_parser::parse("s*x+b+a", &mut ctx).expect("right pole");
    let slope = cas_parser::parse("s", &mut ctx).expect("slope");

    let candidate = try_algorithmic_integration_backend(
        &mut ctx,
        integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );

    assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Hermite);
    assert!(candidate.antiderivative.is_some());
    assert_eq!(
        candidate.verification_status,
        AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
    );
    assert_eq!(
        candidate.verification_evidence,
        AlgorithmicIntegrationVerificationEvidence::MethodSpecificDifferentiation
    );
    assert_eq!(candidate.residual_reason, None);
    assert_eq!(
        candidate.constant_policy,
        IntegrationConstantPolicy::ComponentLocalConstant
    );
    assert_eq!(
        candidate.required_conditions,
        vec![
            ConditionPredicate::NonZero(radius),
            ConditionPredicate::NonZero(left_pole),
            ConditionPredicate::NonZero(right_pole),
            ConditionPredicate::NonZero(slope),
        ]
    );
    assert_eq!(
        candidate.method_probe_no_match_reasons,
        vec![(
            AlgorithmicIntegrationMethod::Rational,
            AlgorithmicIntegrationProbeNoMatchReason::DenominatorPolicyMismatch,
        ),]
    );
    assert_eq!(candidate.method_probes_used, 2);
    assert_eq!(candidate.verification_checks_used, 1);
    assert!(candidate.is_publicly_acceptable());
}

#[test]
fn diagnostic_hermite_probe_verifies_indefinite_square_mixed_numerator() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("(x+1)/(x^2-a^2)", &mut ctx).expect("integrand");
    let radius = cas_parser::parse("a", &mut ctx).expect("radius");
    let left_pole = cas_parser::parse("x-a", &mut ctx).expect("left pole");
    let right_pole = cas_parser::parse("x+a", &mut ctx).expect("right pole");

    let candidate = try_algorithmic_integration_backend(
        &mut ctx,
        integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );

    assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Hermite);
    assert!(candidate.antiderivative.is_some());
    assert_eq!(
        candidate.verification_status,
        AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
    );
    assert_eq!(
        candidate.verification_evidence,
        AlgorithmicIntegrationVerificationEvidence::MethodSpecificDifferentiation
    );
    assert_eq!(candidate.residual_reason, None);
    assert_eq!(
        candidate.constant_policy,
        IntegrationConstantPolicy::ComponentLocalConstant
    );
    assert_eq!(
        candidate.required_conditions,
        vec![
            ConditionPredicate::NonZero(radius),
            ConditionPredicate::NonZero(left_pole),
            ConditionPredicate::NonZero(right_pole),
        ]
    );
    assert_eq!(
        candidate.method_probe_no_match_reasons,
        vec![(
            AlgorithmicIntegrationMethod::Rational,
            AlgorithmicIntegrationProbeNoMatchReason::NumeratorPolicyMismatch,
        ),]
    );
    assert_eq!(candidate.method_probes_used, 2);
    assert_eq!(candidate.verification_checks_used, 1);
    assert!(candidate.is_publicly_acceptable());
}

#[test]
fn diagnostic_hermite_probe_verifies_symbolic_slope_indefinite_square_mixed_numerator() {
    let mut ctx = Context::new();
    let integrand =
        cas_parser::parse("(m*(s*x+b)+c)/((s*x+b)^2-a^2)", &mut ctx).expect("integrand");
    let radius = cas_parser::parse("a", &mut ctx).expect("radius");
    let left_pole = cas_parser::parse("s*x+b-a", &mut ctx).expect("left pole");
    let right_pole = cas_parser::parse("s*x+b+a", &mut ctx).expect("right pole");
    let slope = cas_parser::parse("s", &mut ctx).expect("slope");

    let candidate = try_algorithmic_integration_backend(
        &mut ctx,
        integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );

    assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Hermite);
    assert!(candidate.antiderivative.is_some());
    assert_eq!(
        candidate.verification_status,
        AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
    );
    assert_eq!(
        candidate.verification_evidence,
        AlgorithmicIntegrationVerificationEvidence::MethodSpecificDifferentiation
    );
    assert_eq!(candidate.residual_reason, None);
    assert_eq!(
        candidate.constant_policy,
        IntegrationConstantPolicy::ComponentLocalConstant
    );
    assert_eq!(
        candidate.required_conditions,
        vec![
            ConditionPredicate::NonZero(radius),
            ConditionPredicate::NonZero(left_pole),
            ConditionPredicate::NonZero(right_pole),
            ConditionPredicate::NonZero(slope),
        ]
    );
    assert_eq!(
        candidate.method_probe_no_match_reasons,
        vec![(
            AlgorithmicIntegrationMethod::Rational,
            AlgorithmicIntegrationProbeNoMatchReason::NumeratorPolicyMismatch,
        ),]
    );
    assert_eq!(candidate.method_probes_used, 2);
    assert_eq!(candidate.verification_checks_used, 1);
    assert!(candidate.is_publicly_acceptable());
}

#[test]
fn diagnostic_hermite_probe_requires_second_method_probe_budget() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("2*x/(x^2+1)", &mut ctx).expect("integrand");
    let config = AlgorithmicIntegrationBackendConfig::diagnostic_only()
        .with_budget(AlgorithmicIntegrationBackendBudget::single_probe());

    let candidate = try_algorithmic_integration_backend(&mut ctx, integrand, "x", config);

    assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Unsupported);
    assert_eq!(candidate.antiderivative, None);
    assert_eq!(
        candidate.verification_status,
        AlgorithmicIntegrationVerificationStatus::Inconclusive
    );
    assert_eq!(
        candidate.verification_evidence,
        AlgorithmicIntegrationVerificationEvidence::None
    );
    assert_eq!(
        candidate.residual_reason,
        Some(AlgorithmicIntegrationResidualReason::BudgetExceeded)
    );
    assert_eq!(candidate.method_probes_used, 1);
    assert_eq!(candidate.verification_checks_used, 0);
    assert_eq!(candidate.public_antiderivative(), None);
    assert_eq!(candidate.fallback_antiderivative(config), None);
}

#[test]
fn diagnostic_heurisch_probe_verifies_after_two_probe_misses() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("cos(x)/sin(x)", &mut ctx).expect("integrand");
    let denominator = match ctx.get(integrand) {
        Expr::Div(_, denominator) => *denominator,
        _ => panic!("expected quotient integrand"),
    };

    let candidate = try_algorithmic_integration_backend(
        &mut ctx,
        integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );

    assert_eq!(
        candidate.method,
        AlgorithmicIntegrationMethod::HeurischProbe
    );
    assert_eq!(
        candidate.verification_status,
        AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
    );
    assert_eq!(candidate.residual_reason, None);
    assert_eq!(
        candidate.verification_blocker,
        AlgorithmicIntegrationVerificationBlocker::None
    );
    assert_eq!(
        candidate.required_conditions,
        vec![ConditionPredicate::NonZero(denominator)]
    );
    assert_eq!(
        candidate.trace_level,
        AlgorithmicIntegrationTraceLevel::AlgorithmicSummary
    );
    assert_eq!(candidate.method_probes_used, 3);
    assert_eq!(candidate.verification_checks_used, 1);
    assert_eq!(candidate.method_probe_budget_limit, 3);
    assert_eq!(candidate.verification_check_budget_limit, 1);
    assert!(candidate.public_antiderivative().is_some());
    assert_eq!(
        candidate.fallback_antiderivative(AlgorithmicIntegrationBackendConfig::diagnostic_only()),
        None
    );
}

#[test]
fn diagnostic_heurisch_probe_respects_method_probe_budget() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("cos(x)/sin(x)", &mut ctx).expect("integrand");
    let config = AlgorithmicIntegrationBackendConfig::diagnostic_only()
        .with_budget(AlgorithmicIntegrationBackendBudget::single_probe());

    let candidate = try_algorithmic_integration_backend(&mut ctx, integrand, "x", config);

    assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Unsupported);
    assert_eq!(candidate.antiderivative, None);
    assert_eq!(
        candidate.verification_status,
        AlgorithmicIntegrationVerificationStatus::Inconclusive
    );
    assert_eq!(
        candidate.residual_reason,
        Some(AlgorithmicIntegrationResidualReason::BudgetExceeded)
    );
    assert_eq!(candidate.public_antiderivative(), None);
    assert_eq!(candidate.fallback_antiderivative(config), None);
}

#[test]
fn residual_fallback_mode_can_consume_verified_heurisch_candidate() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("cos(x)/sin(x)", &mut ctx).expect("integrand");
    let config = AlgorithmicIntegrationBackendConfig::residual_fallback();

    let candidate = try_algorithmic_integration_backend(&mut ctx, integrand, "x", config);

    assert_eq!(
        candidate.method,
        AlgorithmicIntegrationMethod::HeurischProbe
    );
    assert_eq!(
        candidate.verification_status,
        AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
    );
    assert!(candidate.fallback_antiderivative(config).is_some());
}

#[test]
fn public_acceptance_requires_verified_antiderivative_without_residual() {
    let mut ctx = Context::new();
    let integrand = ctx.num(1);
    let antiderivative = ctx.var("x");

    let candidate =
        AlgorithmicIntegrationCandidate::verified_table_reused(integrand, "x", antiderivative);

    assert!(candidate.is_publicly_acceptable());
    assert_eq!(
        candidate.verification_evidence,
        AlgorithmicIntegrationVerificationEvidence::Preverified
    );
    assert_eq!(candidate.public_antiderivative(), Some(antiderivative));
}

#[test]
fn public_acceptance_requires_consumable_trace_and_constant_policy() {
    let mut ctx = Context::new();
    let integrand = ctx.num(1);
    let antiderivative = ctx.var("x");
    let config = AlgorithmicIntegrationBackendConfig::residual_fallback();

    let mut diagnostic_trace_candidate =
        AlgorithmicIntegrationCandidate::verified_table_reused(integrand, "x", antiderivative);
    diagnostic_trace_candidate.trace_level = AlgorithmicIntegrationTraceLevel::DiagnosticOnly;

    assert!(!diagnostic_trace_candidate.is_publicly_acceptable());
    assert_eq!(diagnostic_trace_candidate.public_antiderivative(), None);
    assert_eq!(
        diagnostic_trace_candidate.fallback_antiderivative(config),
        None
    );

    let mut unspecified_constant_candidate =
        AlgorithmicIntegrationCandidate::verified_table_reused(integrand, "x", antiderivative);
    unspecified_constant_candidate.constant_policy = IntegrationConstantPolicy::Unspecified;

    assert!(!unspecified_constant_candidate.is_publicly_acceptable());
    assert_eq!(unspecified_constant_candidate.public_antiderivative(), None);
    assert_eq!(
        unspecified_constant_candidate.fallback_antiderivative(config),
        None
    );

    let mut raw_assumption_candidate =
        AlgorithmicIntegrationCandidate::verified_table_reused(integrand, "x", antiderivative);
    raw_assumption_candidate.assumptions.push(ctx.var("a"));

    assert!(!raw_assumption_candidate.is_publicly_acceptable());
    assert_eq!(raw_assumption_candidate.public_antiderivative(), None);
    assert_eq!(
        raw_assumption_candidate.fallback_antiderivative(config),
        None
    );
}

#[test]
fn residual_blocks_public_candidate_even_when_antiderivative_exists() {
    let mut ctx = Context::new();
    let integrand = ctx.num(1);
    let antiderivative = ctx.var("x");
    let mut candidate =
        AlgorithmicIntegrationCandidate::verified_table_reused(integrand, "x", antiderivative);
    candidate.residual_reason = Some(AlgorithmicIntegrationResidualReason::DomainPolicyMissing);

    assert!(!candidate.is_publicly_acceptable());
    assert_eq!(candidate.public_antiderivative(), None);
}

#[test]
fn unverified_candidate_is_blocked_until_direct_diff_verification() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("1/(x^2+1)", &mut ctx).expect("integrand");
    let antiderivative = cas_parser::parse("arctan(x)", &mut ctx).expect("antiderivative");
    let mut candidate =
        AlgorithmicIntegrationCandidate::unverified_table_reused(integrand, "x", antiderivative);

    assert!(!candidate.is_publicly_acceptable());

    let outcome = verify_antiderivative_by_differentiation(&mut ctx, &mut candidate);

    assert!(matches!(
        outcome,
        AlgorithmicIntegrationVerificationOutcome::Verified { .. }
    ));
    assert_eq!(
        candidate.verification_status,
        AlgorithmicIntegrationVerificationStatus::Verified
    );
    assert_eq!(candidate.residual_reason, None);
    assert_eq!(candidate.public_antiderivative(), Some(antiderivative));
}

#[test]
fn verification_report_does_not_mutate_candidate_until_applied() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("1/(x^2+1)", &mut ctx).expect("integrand");
    let antiderivative = cas_parser::parse("arctan(x)", &mut ctx).expect("antiderivative");
    let mut candidate =
        AlgorithmicIntegrationCandidate::unverified_table_reused(integrand, "x", antiderivative);

    let report = antiderivative_verification_report(&mut ctx, &candidate);

    assert_eq!(
        report.status,
        AlgorithmicIntegrationVerificationStatus::Verified
    );
    assert_eq!(
        report.evidence,
        AlgorithmicIntegrationVerificationEvidence::DirectDifferentiation
    );
    assert_eq!(
        report.blocker,
        AlgorithmicIntegrationVerificationBlocker::None
    );
    assert_eq!(report.residual_reason, None);
    assert!(report.derivative.is_some());
    assert_eq!(
        candidate.verification_status,
        AlgorithmicIntegrationVerificationStatus::NotAttempted
    );
    assert_eq!(
        candidate.residual_reason,
        Some(AlgorithmicIntegrationResidualReason::VerificationInconclusive)
    );

    report.apply_to_candidate(&mut candidate);

    assert_eq!(
        candidate.verification_status,
        AlgorithmicIntegrationVerificationStatus::Verified
    );
    assert_eq!(candidate.residual_reason, None);
}

#[test]
fn direct_diff_verification_rejects_mismatched_candidate() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("x", &mut ctx).expect("integrand");
    let antiderivative = cas_parser::parse("x", &mut ctx).expect("antiderivative");
    let mut candidate =
        AlgorithmicIntegrationCandidate::unverified_table_reused(integrand, "x", antiderivative);

    let outcome = verify_antiderivative_by_differentiation(&mut ctx, &mut candidate);

    assert!(matches!(
        outcome,
        AlgorithmicIntegrationVerificationOutcome::Failed { .. }
    ));
    assert_eq!(
        candidate.verification_status,
        AlgorithmicIntegrationVerificationStatus::Failed
    );
    assert_eq!(
        candidate.verification_evidence,
        AlgorithmicIntegrationVerificationEvidence::FailedDifferentiation
    );
    assert_eq!(
        candidate.residual_reason,
        Some(AlgorithmicIntegrationResidualReason::VerificationFailed)
    );
    assert!(candidate.verification_residual.is_some());
    assert_eq!(
        candidate.verification_residual_kind,
        Some(AlgorithmicIntegrationVerificationResidualKind::DependsOnVariable)
    );
    assert_eq!(
        candidate.verification_residual_signature,
        Some(AlgorithmicIntegrationVerificationResidualSignature::AffineInVariable)
    );
    assert_eq!(
        candidate.verification_blocker,
        AlgorithmicIntegrationVerificationBlocker::DerivativeMismatch
    );
    assert_eq!(
        candidate.failure_class(),
        Some(AlgorithmicIntegrationFailureClass::ResidualAffineInVariable)
    );
    assert_eq!(candidate.public_antiderivative(), None);
}

#[test]
fn failure_class_covers_publication_policy_rejections() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("x", &mut ctx).expect("integrand");
    let antiderivative = cas_parser::parse("x^2/2", &mut ctx).expect("antiderivative");
    let assumption = cas_parser::parse("a > 0", &mut ctx).unwrap_or(antiderivative);
    let mut candidate = AlgorithmicIntegrationCandidate::verified(
        integrand,
        "x",
        antiderivative,
        AlgorithmicIntegrationMethod::Rational,
    );

    candidate.assumptions.push(assumption);
    assert_eq!(
        candidate.failure_class(),
        Some(AlgorithmicIntegrationFailureClass::AssumptionPolicyMissing)
    );

    candidate.assumptions.clear();
    candidate.trace_level = AlgorithmicIntegrationTraceLevel::DiagnosticOnly;
    assert_eq!(
        candidate.failure_class(),
        Some(AlgorithmicIntegrationFailureClass::DiagnosticTraceOnly)
    );

    candidate.trace_level = AlgorithmicIntegrationTraceLevel::AlgorithmicSummary;
    candidate.constant_policy = IntegrationConstantPolicy::Unspecified;
    assert_eq!(
        candidate.failure_class(),
        Some(AlgorithmicIntegrationFailureClass::ConstantPolicyMissing)
    );

    candidate.constant_policy = IntegrationConstantPolicy::ArbitraryConstantOmitted;
    candidate.verification_status = AlgorithmicIntegrationVerificationStatus::NotAttempted;
    assert_eq!(
        candidate.failure_class(),
        Some(AlgorithmicIntegrationFailureClass::RejectedUnverified)
    );
}

#[test]
fn candidate_domain_policy_distinguishes_boundary_regimes() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("1/(x+1)", &mut ctx).expect("integrand");
    let antiderivative = cas_parser::parse("ln(abs(x+1))", &mut ctx).expect("antiderivative");
    let mut candidate = AlgorithmicIntegrationCandidate::verified(
        integrand,
        "x",
        antiderivative,
        AlgorithmicIntegrationMethod::Rational,
    );

    assert_eq!(
        candidate.domain_policy(),
        AlgorithmicIntegrationDomainPolicy::UnconditionalReal
    );

    candidate
        .required_conditions
        .push(ConditionPredicate::NonZero(integrand));
    assert_eq!(
        candidate.domain_policy(),
        AlgorithmicIntegrationDomainPolicy::RealWithRequiredConditions
    );

    candidate.residual_reason = Some(AlgorithmicIntegrationResidualReason::DomainPolicyMissing);
    assert_eq!(
        candidate.domain_policy(),
        AlgorithmicIntegrationDomainPolicy::DomainPolicyMissing
    );

    candidate.residual_reason = None;
    candidate.assumptions.push(antiderivative);
    assert_eq!(
        candidate.domain_policy(),
        AlgorithmicIntegrationDomainPolicy::AssumptionPolicyMissing
    );

    let unsupported = AlgorithmicIntegrationCandidate::unsupported(integrand, "x");
    assert_eq!(
        unsupported.domain_policy(),
        AlgorithmicIntegrationDomainPolicy::Unspecified
    );
}

#[test]
fn verification_report_preserves_derivative_mismatch_blocker() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("x", &mut ctx).expect("integrand");
    let antiderivative = cas_parser::parse("x", &mut ctx).expect("antiderivative");
    let candidate =
        AlgorithmicIntegrationCandidate::unverified_table_reused(integrand, "x", antiderivative);

    let report = antiderivative_verification_report(&mut ctx, &candidate);

    assert_eq!(
        report.status,
        AlgorithmicIntegrationVerificationStatus::Failed
    );
    assert_eq!(
        report.evidence,
        AlgorithmicIntegrationVerificationEvidence::FailedDifferentiation
    );
    assert_eq!(
        report.blocker,
        AlgorithmicIntegrationVerificationBlocker::DerivativeMismatch
    );
    assert_eq!(
        report.residual_reason,
        Some(AlgorithmicIntegrationResidualReason::VerificationFailed)
    );
    assert!(report.derivative.is_some());
    assert!(report.verification_residual.is_some());
    assert_eq!(
        report.verification_residual_kind,
        Some(AlgorithmicIntegrationVerificationResidualKind::DependsOnVariable)
    );
    assert_eq!(
        report.verification_residual_signature,
        Some(AlgorithmicIntegrationVerificationResidualSignature::AffineInVariable)
    );
}

#[test]
fn verification_report_classifies_variable_free_residual_mismatch() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("0", &mut ctx).expect("integrand");
    let antiderivative = cas_parser::parse("x", &mut ctx).expect("antiderivative");
    let candidate =
        AlgorithmicIntegrationCandidate::unverified_table_reused(integrand, "x", antiderivative);

    let report = antiderivative_verification_report(&mut ctx, &candidate);

    assert_eq!(
        report.status,
        AlgorithmicIntegrationVerificationStatus::Failed
    );
    assert_eq!(
        report.verification_residual_kind,
        Some(AlgorithmicIntegrationVerificationResidualKind::VariableFree)
    );
    assert_eq!(
        report.verification_residual_signature,
        Some(AlgorithmicIntegrationVerificationResidualSignature::VariableFreeConstant)
    );
}

#[test]
fn verification_report_classifies_function_residual_signature() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("sin(x)", &mut ctx).expect("integrand");
    let antiderivative = cas_parser::parse("x", &mut ctx).expect("antiderivative");
    let candidate =
        AlgorithmicIntegrationCandidate::unverified_table_reused(integrand, "x", antiderivative);

    let report = antiderivative_verification_report(&mut ctx, &candidate);

    assert_eq!(
        report.status,
        AlgorithmicIntegrationVerificationStatus::Failed
    );
    assert_eq!(
        report.verification_residual_kind,
        Some(AlgorithmicIntegrationVerificationResidualKind::DependsOnVariable)
    );
    assert_eq!(
        report.verification_residual_signature,
        Some(AlgorithmicIntegrationVerificationResidualSignature::FunctionOfVariable)
    );
}

#[test]
fn backend_observability_reports_boundary_metrics() {
    let mut ctx = Context::new();
    let disabled_integrand = ctx.num(1);
    let disabled = try_algorithmic_integration_backend(
        &mut ctx,
        disabled_integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::default(),
    );
    let unsupported_integrand = ctx.num(1);
    let unsupported = try_algorithmic_integration_backend(
        &mut ctx,
        unsupported_integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );
    let method_limited_integrand = ctx.num(1);
    let method_limited_config = AlgorithmicIntegrationBackendConfig::diagnostic_only()
        .with_budget(AlgorithmicIntegrationBackendBudget::disabled());
    let method_limited = try_algorithmic_integration_backend(
        &mut ctx,
        method_limited_integrand,
        "x",
        method_limited_config,
    );
    let rational_integrand = cas_parser::parse("1/(x+1)", &mut ctx).expect("rational integrand");
    let diagnostic_rational_probe = try_algorithmic_integration_backend(
        &mut ctx,
        rational_integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );
    let scaled_rational_integrand =
        cas_parser::parse("2/(x+1)", &mut ctx).expect("scaled rational integrand");
    let diagnostic_scaled_rational_probe = try_algorithmic_integration_backend(
        &mut ctx,
        scaled_rational_integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );
    let symbolic_scaled_rational_integrand =
        cas_parser::parse("a/(x+1)", &mut ctx).expect("symbolic scaled rational integrand");
    let diagnostic_symbolic_scaled_rational_probe = try_algorithmic_integration_backend(
        &mut ctx,
        symbolic_scaled_rational_integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );
    let numeric_slope_rational_integrand =
        cas_parser::parse("1/(2*x+1)", &mut ctx).expect("numeric slope rational integrand");
    let diagnostic_numeric_slope_rational_probe = try_algorithmic_integration_backend(
        &mut ctx,
        numeric_slope_rational_integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );
    let symbolic_slope_rational_integrand =
        cas_parser::parse("1/(a*x+b)", &mut ctx).expect("symbolic slope rational integrand");
    let diagnostic_symbolic_slope_rational_probe = try_algorithmic_integration_backend(
        &mut ctx,
        symbolic_slope_rational_integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );
    let product_symbolic_slope_rational_integrand = cas_parser::parse("1/(2*a*x+b)", &mut ctx)
        .expect("product symbolic slope rational integrand");
    let diagnostic_product_symbolic_slope_rational_probe = try_algorithmic_integration_backend(
        &mut ctx,
        product_symbolic_slope_rational_integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );
    let affine_quotient_remainder_rational_integrand =
        cas_parser::parse("(m*(a*x+b)+c)/(a*x+b)", &mut ctx)
            .expect("affine quotient-remainder rational integrand");
    let diagnostic_affine_quotient_remainder_rational_probe = try_algorithmic_integration_backend(
        &mut ctx,
        affine_quotient_remainder_rational_integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );
    let raw_affine_quotient_remainder_rational_integrand = cas_parser::parse("x/(x+1)", &mut ctx)
        .expect("raw affine quotient-remainder rational integrand");
    let diagnostic_raw_affine_quotient_remainder_rational_probe =
        try_algorithmic_integration_backend(
            &mut ctx,
            raw_affine_quotient_remainder_rational_integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );
    let symbolic_intercept_affine_quotient_remainder_rational_integrand =
        cas_parser::parse("(3*x+c)/(2*x+1)", &mut ctx)
            .expect("symbolic intercept affine quotient-remainder rational integrand");
    let diagnostic_symbolic_intercept_affine_quotient_remainder_rational_probe =
        try_algorithmic_integration_backend(
            &mut ctx,
            symbolic_intercept_affine_quotient_remainder_rational_integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );
    let symbolic_slope_raw_affine_quotient_remainder_rational_integrand =
        cas_parser::parse("(3*x+c)/(a*x+b)", &mut ctx)
            .expect("symbolic slope raw affine quotient-remainder rational integrand");
    let diagnostic_symbolic_slope_raw_affine_quotient_remainder_rational_probe =
        try_algorithmic_integration_backend(
            &mut ctx,
            symbolic_slope_raw_affine_quotient_remainder_rational_integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );
    let product_symbolic_slope_raw_affine_quotient_remainder_rational_integrand =
        cas_parser::parse("(3*x+c)/(2*a*x+b)", &mut ctx)
            .expect("product symbolic slope raw affine quotient-remainder rational integrand");
    let diagnostic_product_symbolic_slope_raw_affine_quotient_remainder_rational_probe =
        try_algorithmic_integration_backend(
            &mut ctx,
            product_symbolic_slope_raw_affine_quotient_remainder_rational_integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );
    let hermite_integrand = cas_parser::parse("2*x/(x^2+1)", &mut ctx).expect("hermite integrand");
    let diagnostic_hermite_probe = try_algorithmic_integration_backend(
        &mut ctx,
        hermite_integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );
    let shifted_hermite_integrand =
        cas_parser::parse("2*x/(x^2+2)", &mut ctx).expect("shifted hermite integrand");
    let diagnostic_shifted_hermite_probe = try_algorithmic_integration_backend(
        &mut ctx,
        shifted_hermite_integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );
    let scaled_hermite_integrand =
        cas_parser::parse("3*x/(x^2+2)", &mut ctx).expect("scaled hermite integrand");
    let diagnostic_scaled_hermite_probe = try_algorithmic_integration_backend(
        &mut ctx,
        scaled_hermite_integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );
    let symbolic_scaled_hermite_integrand =
        cas_parser::parse("a*x/(x^2+2)", &mut ctx).expect("symbolic scaled hermite integrand");
    let diagnostic_symbolic_scaled_hermite_probe = try_algorithmic_integration_backend(
        &mut ctx,
        symbolic_scaled_hermite_integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );
    let unit_mixed_numerator_hermite_integrand =
        cas_parser::parse("(x+1)/(x^2+1)", &mut ctx).expect("unit mixed numerator hermite");
    let diagnostic_unit_mixed_numerator_hermite_probe = try_algorithmic_integration_backend(
        &mut ctx,
        unit_mixed_numerator_hermite_integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );
    let square_radius_mixed_numerator_hermite_integrand =
        cas_parser::parse("(x+1)/(x^2+4)", &mut ctx)
            .expect("square radius mixed numerator hermite");
    let diagnostic_square_radius_mixed_numerator_hermite_probe =
        try_algorithmic_integration_backend(
            &mut ctx,
            square_radius_mixed_numerator_hermite_integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );
    let square_radius_constant_hermite_integrand =
        cas_parser::parse("1/(x^2+4)", &mut ctx).expect("square radius constant hermite");
    let diagnostic_square_radius_constant_hermite_probe = try_algorithmic_integration_backend(
        &mut ctx,
        square_radius_constant_hermite_integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );
    let mixed_numerator_hermite_integrand =
        cas_parser::parse("(x+1)/(x^2+2)", &mut ctx).expect("mixed numerator hermite gap");
    let diagnostic_mixed_numerator_hermite_probe = try_algorithmic_integration_backend(
        &mut ctx,
        mixed_numerator_hermite_integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );
    let symbolic_positive_radius_mixed_numerator_hermite_integrand =
        cas_parser::parse("(x+1)/(x^2+a)", &mut ctx)
            .expect("symbolic positive radius mixed numerator hermite");
    let diagnostic_symbolic_positive_radius_mixed_numerator_hermite_probe =
        try_algorithmic_integration_backend(
            &mut ctx,
            symbolic_positive_radius_mixed_numerator_hermite_integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );
    let symbolic_external_positive_radius_mixed_numerator_hermite_integrand =
        cas_parser::parse("(b*x+c)/(x^2+a)", &mut ctx)
            .expect("symbolic external positive radius mixed numerator hermite");
    let diagnostic_symbolic_external_positive_radius_mixed_numerator_hermite_probe =
        try_algorithmic_integration_backend(
            &mut ctx,
            symbolic_external_positive_radius_mixed_numerator_hermite_integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );
    let shifted_affine_symbolic_external_positive_radius_mixed_numerator_hermite_integrand =
        cas_parser::parse("(m*(x+b)+c)/((x+b)^2+a)", &mut ctx)
            .expect("shifted affine symbolic external positive radius mixed numerator hermite");
    let diagnostic_shifted_affine_symbolic_external_positive_radius_mixed_numerator_hermite_probe =
        try_algorithmic_integration_backend(
            &mut ctx,
            shifted_affine_symbolic_external_positive_radius_mixed_numerator_hermite_integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );
    let numeric_slope_shifted_affine_symbolic_external_positive_radius_mixed_numerator_hermite_integrand =
        cas_parser::parse("(m*(2*x+b)+c)/((2*x+b)^2+a)", &mut ctx)
            .expect("numeric slope shifted affine symbolic external positive radius mixed numerator hermite");
    let diagnostic_numeric_slope_shifted_affine_symbolic_external_positive_radius_mixed_numerator_hermite_probe =
        try_algorithmic_integration_backend(
            &mut ctx,
            numeric_slope_shifted_affine_symbolic_external_positive_radius_mixed_numerator_hermite_integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );
    let symbolic_slope_shifted_affine_symbolic_external_positive_radius_mixed_numerator_hermite_integrand =
        cas_parser::parse("(m*(s*x+b)+c)/((s*x+b)^2+a)", &mut ctx)
            .expect("symbolic slope shifted affine symbolic external positive radius mixed numerator hermite");
    let diagnostic_symbolic_slope_shifted_affine_symbolic_external_positive_radius_mixed_numerator_hermite_probe =
        try_algorithmic_integration_backend(
            &mut ctx,
            symbolic_slope_shifted_affine_symbolic_external_positive_radius_mixed_numerator_hermite_integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );
    let symbolic_square_radius_log_derivative_integrand =
        cas_parser::parse("2*x/(x^2+a^2)", &mut ctx)
            .expect("symbolic square radius log derivative");
    let diagnostic_symbolic_square_radius_log_derivative_probe =
        try_algorithmic_integration_backend(
            &mut ctx,
            symbolic_square_radius_log_derivative_integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );
    let symbolic_square_radius_arctan_integrand =
        cas_parser::parse("1/(x^2+a^2)", &mut ctx).expect("symbolic square radius arctan");
    let diagnostic_symbolic_square_radius_arctan_probe = try_algorithmic_integration_backend(
        &mut ctx,
        symbolic_square_radius_arctan_integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );
    let symbolic_slope_shifted_symbolic_square_radius_mixed_numerator_integrand =
        cas_parser::parse("(m*(s*x+b)+c)/((s*x+b)^2+a^2)", &mut ctx)
            .expect("symbolic slope shifted symbolic square radius mixed numerator");
    let diagnostic_symbolic_slope_shifted_symbolic_square_radius_mixed_numerator_probe =
        try_algorithmic_integration_backend(
            &mut ctx,
            symbolic_slope_shifted_symbolic_square_radius_mixed_numerator_integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );
    let indefinite_symbolic_square_denominator_policy_gap_integrand =
        cas_parser::parse("1/(x^2-a^2)", &mut ctx)
            .expect("indefinite symbolic square denominator policy gap");
    let diagnostic_indefinite_symbolic_square_denominator_policy_gap_probe =
        try_algorithmic_integration_backend(
            &mut ctx,
            indefinite_symbolic_square_denominator_policy_gap_integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );
    let symbolic_slope_indefinite_square_denominator_policy_gap_integrand =
        cas_parser::parse("1/((s*x+b)^2-a^2)", &mut ctx)
            .expect("symbolic slope indefinite square denominator policy gap");
    let diagnostic_symbolic_slope_indefinite_square_denominator_policy_gap_probe =
        try_algorithmic_integration_backend(
            &mut ctx,
            symbolic_slope_indefinite_square_denominator_policy_gap_integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );
    let indefinite_square_unit_mixed_numerator_integrand =
        cas_parser::parse("(x+1)/(x^2-a^2)", &mut ctx)
            .expect("indefinite square unit mixed numerator");
    let diagnostic_indefinite_square_unit_mixed_numerator_probe =
        try_algorithmic_integration_backend(
            &mut ctx,
            indefinite_square_unit_mixed_numerator_integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );
    let symbolic_slope_indefinite_square_mixed_numerator_integrand =
        cas_parser::parse("(m*(s*x+b)+c)/((s*x+b)^2-a^2)", &mut ctx)
            .expect("symbolic slope indefinite square mixed numerator");
    let diagnostic_symbolic_slope_indefinite_square_mixed_numerator_probe =
        try_algorithmic_integration_backend(
            &mut ctx,
            symbolic_slope_indefinite_square_mixed_numerator_integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );
    let heurisch_integrand =
        cas_parser::parse("cos(x)/sin(x)", &mut ctx).expect("heurisch integrand");
    let diagnostic_heurisch_probe = try_algorithmic_integration_backend(
        &mut ctx,
        heurisch_integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );
    let fallback_rational_integrand =
        cas_parser::parse("1/(x+1)", &mut ctx).expect("fallback rational integrand");
    let fallback_config = AlgorithmicIntegrationBackendConfig::residual_fallback();
    let fallback_rational_probe = try_algorithmic_integration_backend(
        &mut ctx,
        fallback_rational_integrand,
        "x",
        fallback_config,
    );
    let budget_limited_integrand =
        cas_parser::parse("1/(x+1)", &mut ctx).expect("budget-limited rational integrand");
    let budget_limited_config = AlgorithmicIntegrationBackendConfig::diagnostic_only()
        .with_budget(AlgorithmicIntegrationBackendBudget::single_probe_without_verification());
    let budget_limited_rational_probe = try_algorithmic_integration_backend(
        &mut ctx,
        budget_limited_integrand,
        "x",
        budget_limited_config,
    );

    let arctan_integrand = cas_parser::parse("1/(x^2+1)", &mut ctx).expect("integrand");
    let arctan_antiderivative = cas_parser::parse("arctan(x)", &mut ctx).expect("antiderivative");
    let mut verified_candidate = AlgorithmicIntegrationCandidate::unverified_table_reused(
        arctan_integrand,
        "x",
        arctan_antiderivative,
    );
    let verification_start = Instant::now();
    verify_antiderivative_by_differentiation(&mut ctx, &mut verified_candidate);
    let verified_elapsed = verification_start.elapsed();

    let mismatch_integrand = cas_parser::parse("x", &mut ctx).expect("mismatch integrand");
    let mismatch_antiderivative =
        cas_parser::parse("x", &mut ctx).expect("mismatch antiderivative");
    let mut rejected_candidate = AlgorithmicIntegrationCandidate::unverified_table_reused(
        mismatch_integrand,
        "x",
        mismatch_antiderivative,
    );
    let rejection_start = Instant::now();
    verify_antiderivative_by_differentiation(&mut ctx, &mut rejected_candidate);
    let rejected_elapsed = rejection_start.elapsed();

    let function_mismatch_integrand =
        cas_parser::parse("sin(x)", &mut ctx).expect("function mismatch integrand");
    let function_mismatch_antiderivative =
        cas_parser::parse("x", &mut ctx).expect("function mismatch antiderivative");
    let mut function_rejected_candidate = AlgorithmicIntegrationCandidate::unverified_table_reused(
        function_mismatch_integrand,
        "x",
        function_mismatch_antiderivative,
    );
    let function_rejection_start = Instant::now();
    verify_antiderivative_by_differentiation(&mut ctx, &mut function_rejected_candidate);
    let function_rejected_elapsed = function_rejection_start.elapsed();

    let labeled_observed = vec![
        (
            "disabled",
            AlgorithmicIntegrationBackendConfig::default(),
            disabled,
        ),
        (
            "constant_integrand_backend_unsupported_baseline",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
            unsupported,
        ),
        (
            "method_limited_constant",
            method_limited_config,
            method_limited,
        ),
        (
            "linear_denominator_rational",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
            diagnostic_rational_probe,
        ),
        (
            "scaled_linear_denominator_rational",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
            diagnostic_scaled_rational_probe,
        ),
        (
            "symbolic_scaled_linear_denominator_rational",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
            diagnostic_symbolic_scaled_rational_probe,
        ),
        (
            "numeric_slope_linear_denominator_rational",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
            diagnostic_numeric_slope_rational_probe,
        ),
        (
            "symbolic_slope_linear_denominator_rational",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
            diagnostic_symbolic_slope_rational_probe,
        ),
        (
            "product_symbolic_slope_linear_denominator_rational",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
            diagnostic_product_symbolic_slope_rational_probe,
        ),
        (
            "affine_quotient_remainder_rational",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
            diagnostic_affine_quotient_remainder_rational_probe,
        ),
        (
            "raw_affine_quotient_remainder_rational",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
            diagnostic_raw_affine_quotient_remainder_rational_probe,
        ),
        (
            "symbolic_intercept_affine_quotient_remainder_rational",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
            diagnostic_symbolic_intercept_affine_quotient_remainder_rational_probe,
        ),
        (
            "symbolic_slope_raw_affine_quotient_remainder_rational",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
            diagnostic_symbolic_slope_raw_affine_quotient_remainder_rational_probe,
        ),
        (
            "product_symbolic_slope_raw_affine_quotient_remainder_rational",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
            diagnostic_product_symbolic_slope_raw_affine_quotient_remainder_rational_probe,
        ),
        (
            "positive_quadratic_log_derivative",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
            diagnostic_hermite_probe,
        ),
        (
            "shifted_positive_quadratic_log_derivative",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
            diagnostic_shifted_hermite_probe,
        ),
        (
            "scaled_positive_quadratic_log_derivative",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
            diagnostic_scaled_hermite_probe,
        ),
        (
            "symbolic_scaled_positive_quadratic_log_derivative",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
            diagnostic_symbolic_scaled_hermite_probe,
        ),
        (
            "unit_mixed_numerator_positive_quadratic",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
            diagnostic_unit_mixed_numerator_hermite_probe,
        ),
        (
            "square_radius_mixed_numerator_positive_quadratic",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
            diagnostic_square_radius_mixed_numerator_hermite_probe,
        ),
        (
            "square_radius_constant_positive_quadratic",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
            diagnostic_square_radius_constant_hermite_probe,
        ),
        (
            "positive_quadratic_mixed_numerator_shape_gap",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
            diagnostic_mixed_numerator_hermite_probe,
        ),
        (
            "symbolic_positive_radius_mixed_numerator",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
            diagnostic_symbolic_positive_radius_mixed_numerator_hermite_probe,
        ),
        (
            "symbolic_external_positive_radius_mixed_numerator",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
            diagnostic_symbolic_external_positive_radius_mixed_numerator_hermite_probe,
        ),
        (
            "shifted_affine_symbolic_external_positive_radius_mixed_numerator",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
            diagnostic_shifted_affine_symbolic_external_positive_radius_mixed_numerator_hermite_probe,
        ),
        (
            "numeric_slope_shifted_affine_symbolic_external_positive_radius_mixed_numerator",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
            diagnostic_numeric_slope_shifted_affine_symbolic_external_positive_radius_mixed_numerator_hermite_probe,
        ),
        (
            "symbolic_slope_shifted_affine_symbolic_external_positive_radius_mixed_numerator",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
            diagnostic_symbolic_slope_shifted_affine_symbolic_external_positive_radius_mixed_numerator_hermite_probe,
        ),
        (
            "symbolic_square_radius_log_derivative",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
            diagnostic_symbolic_square_radius_log_derivative_probe,
        ),
        (
            "symbolic_square_radius_arctan",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
            diagnostic_symbolic_square_radius_arctan_probe,
        ),
        (
            "symbolic_slope_shifted_symbolic_square_radius_mixed_numerator",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
            diagnostic_symbolic_slope_shifted_symbolic_square_radius_mixed_numerator_probe,
        ),
        (
            "indefinite_square_denominator_policy_gap",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
            diagnostic_indefinite_symbolic_square_denominator_policy_gap_probe,
        ),
        (
            "indefinite_square_denominator_policy_gap",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
            diagnostic_symbolic_slope_indefinite_square_denominator_policy_gap_probe,
        ),
        (
            "indefinite_square_unit_mixed_numerator",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
            diagnostic_indefinite_square_unit_mixed_numerator_probe,
        ),
        (
            "indefinite_square_mixed_numerator",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
            diagnostic_symbolic_slope_indefinite_square_mixed_numerator_probe,
        ),
        (
            "heurisch_log_derivative",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
            diagnostic_heurisch_probe,
        ),
        (
            "fallback_linear_denominator_rational",
            fallback_config,
            fallback_rational_probe,
        ),
        (
            "verification_limited_linear_denominator_rational",
            budget_limited_config,
            budget_limited_rational_probe,
        ),
        (
            "table_reused_verified_arctan",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
            verified_candidate,
        ),
        (
            "table_reused_affine_residual",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
            rejected_candidate,
        ),
        (
            "table_reused_function_residual",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
            function_rejected_candidate,
        ),
    ];
    let observed: Vec<_> = labeled_observed
        .iter()
        .map(|(_, config, candidate)| (*config, candidate.clone()))
        .collect();

    assert_eq!(
        observed
            .iter()
            .filter(|(_, candidate)| candidate.is_publicly_acceptable())
            .count(),
        34
    );
    assert_eq!(
        observed
            .iter()
            .filter(|(_, candidate)| {
                candidate.is_publicly_acceptable()
                    && !matches!(
                        candidate.verification_status,
                        AlgorithmicIntegrationVerificationStatus::Verified
                            | AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
                    )
            })
            .count(),
        0
    );
    assert_eq!(
        observed
            .iter()
            .filter(|(config, candidate)| candidate.fallback_antiderivative(*config).is_some())
            .count(),
        1
    );
    assert_eq!(
        method_probe_no_match_reason_counts(&observed),
        BTreeMap::from([
            ("hermite/denominator_policy_mismatch".to_string(), 1),
            ("hermite/shape_mismatch".to_string(), 1),
            ("heurisch_probe/shape_mismatch".to_string(), 1),
            ("rational/denominator_policy_mismatch".to_string(), 4),
            ("rational/numerator_policy_mismatch".to_string(), 17),
            ("rational/shape_mismatch".to_string(), 1),
        ])
    );
    assert_eq!(
        method_probe_no_match_class_counts(&observed),
        BTreeMap::from([("policy", 22), ("shape", 3),])
    );
    assert_eq!(
        method_probe_no_match_class_by_method(&observed),
        BTreeMap::from([
            ("hermite/policy".to_string(), 1),
            ("hermite/shape".to_string(), 1),
            ("heurisch_probe/shape".to_string(), 1),
            ("rational/policy".to_string(), 21),
            ("rational/shape".to_string(), 1),
        ])
    );
    assert_eq!(
        method_probe_no_match_final_method_counts(&observed),
        BTreeMap::from([("hermite", 20), ("heurisch_probe", 2), ("unsupported", 3),])
    );
    assert_eq!(
        method_probe_no_match_final_method_by_attempt(&observed),
        BTreeMap::from([
            ("hermite/heurisch_probe".to_string(), 1),
            ("hermite/unsupported".to_string(), 1),
            ("heurisch_probe/unsupported".to_string(), 1),
            ("rational/hermite".to_string(), 20),
            ("rational/heurisch_probe".to_string(), 1),
            ("rational/unsupported".to_string(), 1),
        ])
    );
    assert_eq!(
        method_probe_terminal_no_match_reason_counts(&observed),
        BTreeMap::from([
            ("hermite/shape_mismatch".to_string(), 1),
            ("heurisch_probe/shape_mismatch".to_string(), 1),
            ("rational/shape_mismatch".to_string(), 1),
        ])
    );
    assert_eq!(
        method_probe_terminal_no_match_class_counts(&observed),
        BTreeMap::from([("shape", 3),])
    );
    assert_eq!(
        method_probe_terminal_no_match_class_by_method(&observed),
        BTreeMap::from([
            ("hermite/shape".to_string(), 1),
            ("heurisch_probe/shape".to_string(), 1),
            ("rational/shape".to_string(), 1),
        ])
    );
    assert_eq!(method_probe_terminal_candidate_count(&observed), 1);
    assert_eq!(
        method_probe_terminal_candidate_signature_counts(&labeled_observed),
        BTreeMap::from([("constant_integrand_backend_unsupported_baseline", 1),])
    );
    assert_eq!(
        verification_check_usage_by_publication_status(&observed),
        BTreeMap::from([("accepted", 33),])
    );
    assert_eq!(
        verification_check_usage_by_evidence(&observed),
        BTreeMap::from([
            ("direct_differentiation", 5),
            ("method_specific_differentiation", 28),
        ])
    );
    assert_eq!(
        verification_check_usage_by_method_and_evidence(&observed),
        BTreeMap::from([
            ("hermite/direct_differentiation".to_string(), 2),
            ("hermite/method_specific_differentiation".to_string(), 18),
            ("heurisch_probe/direct_differentiation".to_string(), 1),
            ("rational/direct_differentiation".to_string(), 2),
            ("rational/method_specific_differentiation".to_string(), 10),
        ])
    );
    assert_eq!(
        verification_check_usage_by_method_and_publication_status(&observed),
        BTreeMap::from([
            ("hermite/accepted".to_string(), 20),
            ("heurisch_probe/accepted".to_string(), 1),
            ("rational/accepted".to_string(), 12),
        ])
    );
    assert_eq!(
        failure_class_counts(&observed),
        BTreeMap::from([
            ("budget_exceeded", 2),
            ("disabled_by_mode", 1),
            ("residual_affine_in_variable", 1),
            ("residual_function_of_variable", 1),
            ("unsupported_method", 1),
        ])
    );
    assert_eq!(
        failure_class_by_method(&observed),
        BTreeMap::from([
            ("rational/budget_exceeded".to_string(), 1),
            ("table_reused/residual_affine_in_variable".to_string(), 1,),
            ("table_reused/residual_function_of_variable".to_string(), 1,),
            ("unsupported/budget_exceeded".to_string(), 1),
            ("unsupported/disabled_by_mode".to_string(), 1),
            ("unsupported/unsupported_method".to_string(), 1),
        ])
    );
    assert_eq!(method_probe_budget_exhausted_count(&observed), 1);
    assert_eq!(verification_budget_exceeded_count(&observed), 0);
    assert_eq!(verification_boundary_budget_exceeded_count(&observed), 1);

    let verification_elapsed_ms =
        (verified_elapsed + rejected_elapsed + function_rejected_elapsed).as_secs_f64() * 1000.0;
    println!(
        "algorithmic_backend_observability: {{\"attempts\":{},\"public_accepted\":{},\"unverified_public_acceptances\":{},\"fallback_eligible\":{},\"unverified_fallback_acceptances\":{},\"method_probe_budget_exhausted\":{},\"verification_budget_exceeded\":{},\"verification_boundary_budget_exceeded\":{},\"method_probes_used_total\":{},\"verification_checks_used_total\":{},\"method_probe_budget_limit_total\":{},\"verification_check_budget_limit_total\":{},\"verification_elapsed_ms\":{:.3},\"mode_counts\":{},\"method_counts\":{},\"method_probe_usage_by_method\":{},\"method_probe_attempt_counts\":{},\"method_probe_candidate_counts\":{},\"method_probe_no_match_counts\":{},\"method_probe_no_match_reason_counts\":{},\"method_probe_no_match_class_counts\":{},\"method_probe_no_match_class_by_method\":{},\"method_probe_no_match_final_method_counts\":{},\"method_probe_no_match_final_method_by_attempt\":{},\"method_probe_terminal_no_match_reason_counts\":{},\"method_probe_terminal_no_match_class_counts\":{},\"method_probe_terminal_no_match_class_by_method\":{},\"method_probe_terminal_candidate_count\":{},\"method_probe_terminal_candidate_signature_counts\":{},\"verification_check_usage_by_method\":{},\"verification_check_usage_by_evidence\":{},\"verification_check_usage_by_method_and_evidence\":{},\"verification_check_usage_by_publication_status\":{},\"verification_check_usage_by_method_and_publication_status\":{},\"verification_status_by_method\":{},\"residual_reason_by_method\":{},\"verification_blocker_counts\":{},\"verification_blocker_by_method\":{},\"failure_class_counts\":{},\"failure_class_by_method\":{},\"verification_residual_counts\":{},\"verification_residual_by_method\":{},\"verification_residual_kind_counts\":{},\"verification_residual_kind_by_method\":{},\"verification_residual_signature_counts\":{},\"verification_residual_signature_by_method\":{},\"publication_status_counts\":{},\"publication_status_by_method\":{},\"fallback_status_counts\":{},\"fallback_status_by_method\":{},\"trace_level_counts\":{},\"constant_policy_counts\":{},\"domain_policy_counts\":{},\"domain_policy_by_method\":{},\"public_trace_level_counts\":{},\"public_constant_policy_counts\":{},\"public_domain_policy_counts\":{},\"public_domain_policy_by_method\":{},\"fallback_trace_level_counts\":{},\"fallback_constant_policy_counts\":{},\"fallback_domain_policy_counts\":{},\"fallback_domain_policy_by_method\":{},\"assumption_exprs\":{},\"public_assumption_exprs\":{},\"fallback_assumption_exprs\":{},\"verification_evidence_counts\":{},\"public_verification_evidence_counts\":{},\"fallback_verification_evidence_counts\":{},\"verification_evidence_by_method\":{},\"public_verification_evidence_by_method\":{},\"fallback_verification_evidence_by_method\":{},\"verification_normalization_reason_counts\":{},\"public_verification_normalization_reason_counts\":{},\"fallback_verification_normalization_reason_counts\":{},\"verification_normalization_reason_by_method\":{},\"public_verification_normalization_reason_by_method\":{},\"fallback_verification_normalization_reason_by_method\":{},\"verification_normalization_reason_by_label\":{},\"verification_normalization_pass_count_counts\":{},\"public_verification_normalization_pass_count_counts\":{},\"fallback_verification_normalization_pass_count_counts\":{},\"verification_normalization_pass_count_by_method\":{},\"public_verification_normalization_pass_count_by_method\":{},\"fallback_verification_normalization_pass_count_by_method\":{},\"max_verification_normalization_passes\":{},\"public_max_verification_normalization_passes\":{},\"fallback_max_verification_normalization_passes\":{},\"verification_status_counts\":{},\"residual_reason_counts\":{},\"required_condition_counts\":{}}}",
        observed.len(),
        observed
            .iter()
            .filter(|(_, candidate)| candidate.is_publicly_acceptable())
            .count(),
        observed
            .iter()
            .filter(|(_, candidate)| {
                candidate.is_publicly_acceptable()
                    && !matches!(
                        candidate.verification_status,
                        AlgorithmicIntegrationVerificationStatus::Verified
                            | AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
                    )
            })
            .count(),
        observed
            .iter()
            .filter(|(config, candidate)| candidate.fallback_antiderivative(*config).is_some())
            .count(),
        observed
            .iter()
            .filter(|(config, candidate)| {
                candidate.fallback_antiderivative(*config).is_some()
                    && !matches!(
                        candidate.verification_status,
                        AlgorithmicIntegrationVerificationStatus::Verified
                            | AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
                    )
        })
            .count(),
        method_probe_budget_exhausted_count(&observed),
        verification_budget_exceeded_count(&observed),
        verification_boundary_budget_exceeded_count(&observed),
        method_probes_used_total(&observed),
        verification_checks_used_total(&observed),
        method_probe_budget_limit_total(&observed),
        verification_check_budget_limit_total(&observed),
        verification_elapsed_ms,
        json_count_map(mode_counts(&observed)),
        json_count_map(method_counts(&observed)),
        json_count_map(method_probe_usage_by_method(&observed)),
        json_count_map(method_probe_attempt_counts(&observed)),
        json_count_map(method_probe_candidate_counts(&observed)),
        json_count_map(method_probe_no_match_counts(&observed)),
        json_string_count_map(method_probe_no_match_reason_counts(&observed)),
        json_count_map(method_probe_no_match_class_counts(&observed)),
        json_string_count_map(method_probe_no_match_class_by_method(&observed)),
        json_count_map(method_probe_no_match_final_method_counts(&observed)),
        json_string_count_map(method_probe_no_match_final_method_by_attempt(&observed)),
        json_string_count_map(method_probe_terminal_no_match_reason_counts(&observed)),
        json_count_map(method_probe_terminal_no_match_class_counts(&observed)),
        json_string_count_map(method_probe_terminal_no_match_class_by_method(&observed)),
        method_probe_terminal_candidate_count(&observed),
        json_count_map(method_probe_terminal_candidate_signature_counts(
            &labeled_observed
        )),
        json_count_map(verification_check_usage_by_method(&observed)),
        json_count_map(verification_check_usage_by_evidence(&observed)),
        json_string_count_map(verification_check_usage_by_method_and_evidence(
            &observed
        )),
        json_count_map(verification_check_usage_by_publication_status(&observed)),
        json_string_count_map(verification_check_usage_by_method_and_publication_status(
            &observed
        )),
        json_string_count_map(verification_status_by_method(&observed)),
        json_string_count_map(residual_reason_by_method(&observed)),
        json_count_map(verification_blocker_counts(&observed)),
        json_string_count_map(verification_blocker_by_method(&observed)),
        json_count_map(failure_class_counts(&observed)),
        json_string_count_map(failure_class_by_method(&observed)),
        json_count_map(verification_residual_counts(&observed)),
        json_string_count_map(verification_residual_by_method(&observed)),
        json_count_map(verification_residual_kind_counts(&observed)),
        json_string_count_map(verification_residual_kind_by_method(&observed)),
        json_count_map(verification_residual_signature_counts(&observed)),
        json_string_count_map(verification_residual_signature_by_method(&observed)),
        json_count_map(publication_status_counts(&observed)),
        json_string_count_map(publication_status_by_method(&observed)),
        json_count_map(fallback_status_counts(&observed)),
        json_string_count_map(fallback_status_by_method(&observed)),
        json_count_map(trace_level_counts(&observed)),
        json_count_map(constant_policy_counts(&observed)),
        json_count_map(domain_policy_counts(&observed)),
        json_string_count_map(domain_policy_by_method(&observed)),
        json_count_map(public_trace_level_counts(&observed)),
        json_count_map(public_constant_policy_counts(&observed)),
        json_count_map(public_domain_policy_counts(&observed)),
        json_string_count_map(public_domain_policy_by_method(&observed)),
        json_count_map(fallback_trace_level_counts(&observed)),
        json_count_map(fallback_constant_policy_counts(&observed)),
        json_count_map(fallback_domain_policy_counts(&observed)),
        json_string_count_map(fallback_domain_policy_by_method(&observed)),
        assumption_expr_count(&observed),
        public_assumption_expr_count(&observed),
        fallback_assumption_expr_count(&observed),
        json_count_map(verification_evidence_counts(&observed)),
        json_count_map(public_verification_evidence_counts(&observed)),
        json_count_map(fallback_verification_evidence_counts(&observed)),
        json_string_count_map(verification_evidence_by_method(&observed)),
        json_string_count_map(public_verification_evidence_by_method(&observed)),
        json_string_count_map(fallback_verification_evidence_by_method(&observed)),
        json_count_map(verification_normalization_reason_counts(&observed)),
        json_count_map(public_verification_normalization_reason_counts(
            &observed
        )),
        json_count_map(fallback_verification_normalization_reason_counts(
            &observed
        )),
        json_string_count_map(verification_normalization_reason_by_method(&observed)),
        json_string_count_map(public_verification_normalization_reason_by_method(
            &observed
        )),
        json_string_count_map(fallback_verification_normalization_reason_by_method(
            &observed
        )),
        json_string_count_map(verification_normalization_reason_by_label(
            &labeled_observed
        )),
        json_string_count_map(verification_normalization_pass_count_counts(&observed)),
        json_string_count_map(public_verification_normalization_pass_count_counts(
            &observed
        )),
        json_string_count_map(fallback_verification_normalization_pass_count_counts(
            &observed
        )),
        json_string_count_map(verification_normalization_pass_count_by_method(&observed)),
        json_string_count_map(public_verification_normalization_pass_count_by_method(
            &observed
        )),
        json_string_count_map(fallback_verification_normalization_pass_count_by_method(
            &observed
        )),
        max_verification_normalization_passes(&observed),
        public_max_verification_normalization_passes(&observed),
        fallback_max_verification_normalization_passes(&observed),
        json_count_map(verification_status_counts(&observed)),
        json_count_map(residual_reason_counts(&observed)),
        json_count_map(required_condition_counts(&observed)),
    );
}

fn method_probe_budget_exhausted_count(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> usize {
    observed
        .iter()
        .filter(|(config, candidate)| {
            config.mode.attempts_backend()
                && config.budget.max_method_probes == 0
                && candidate.residual_reason.as_ref()
                    == Some(&AlgorithmicIntegrationResidualReason::BudgetExceeded)
        })
        .count()
}

fn verification_budget_exceeded_count(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> usize {
    observed
        .iter()
        .filter(|(config, candidate)| {
            config.mode.attempts_backend()
                && config.budget.max_method_probes > 0
                && config.budget.max_verification_checks > 0
                && candidate.verification_blocker
                    == AlgorithmicIntegrationVerificationBlocker::BudgetExceeded
                && candidate.residual_reason.as_ref()
                    == Some(&AlgorithmicIntegrationResidualReason::BudgetExceeded)
        })
        .count()
}

fn verification_boundary_budget_exceeded_count(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> usize {
    observed
        .iter()
        .filter(|(config, candidate)| {
            config.mode.attempts_backend()
                && config.budget.max_method_probes > 0
                && config.budget.max_verification_checks == 0
                && candidate.antiderivative.is_some()
                && candidate.residual_reason.as_ref()
                    == Some(&AlgorithmicIntegrationResidualReason::BudgetExceeded)
        })
        .count()
}

fn method_probes_used_total(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> usize {
    observed
        .iter()
        .map(|(_, candidate)| candidate.method_probes_used)
        .sum()
}

fn method_probe_budget_limit_total(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> usize {
    observed
        .iter()
        .map(|(_, candidate)| candidate.method_probe_budget_limit)
        .sum()
}

fn verification_checks_used_total(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> usize {
    observed
        .iter()
        .map(|(_, candidate)| candidate.verification_checks_used)
        .sum()
}

fn verification_check_budget_limit_total(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> usize {
    observed
        .iter()
        .map(|(_, candidate)| candidate.verification_check_budget_limit)
        .sum()
}

fn method_probe_usage_by_method(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> BTreeMap<&'static str, usize> {
    let mut counts = BTreeMap::new();
    for (_, candidate) in observed {
        if candidate.method_probes_used > 0 {
            *counts.entry(candidate.method.metric_label()).or_insert(0) +=
                candidate.method_probes_used;
        }
    }
    counts
}

fn method_probe_attempt_counts(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> BTreeMap<&'static str, usize> {
    let mut counts = BTreeMap::new();
    for (_, candidate) in observed {
        for method in &candidate.method_probe_attempts {
            *counts.entry(method.metric_label()).or_insert(0) += 1;
        }
    }
    counts
}

fn method_probe_candidate_counts(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> BTreeMap<&'static str, usize> {
    let mut counts = BTreeMap::new();
    for (_, candidate) in observed {
        if !candidate.method_probe_attempts.is_empty()
            && !matches!(candidate.method, AlgorithmicIntegrationMethod::Unsupported)
        {
            *counts.entry(candidate.method.metric_label()).or_insert(0) += 1;
        }
    }
    counts
}

fn method_probe_no_match_counts(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> BTreeMap<&'static str, usize> {
    let mut counts = method_probe_attempt_counts(observed);
    for (method, candidate_count) in method_probe_candidate_counts(observed) {
        let attempt_count = counts
            .get_mut(method)
            .expect("candidate method was attempted by a method probe");
        *attempt_count = attempt_count
            .checked_sub(candidate_count)
            .expect("candidate count cannot exceed method attempts");
    }
    counts.retain(|_, count| *count > 0);
    counts
}

fn method_probe_no_match_reason_counts(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for (_, candidate) in observed {
        for (method, reason) in &candidate.method_probe_no_match_reasons {
            let key = format!("{}/{}", method.metric_label(), reason.metric_label());
            *counts.entry(key).or_insert(0) += 1;
        }
    }
    counts
}

fn method_probe_no_match_class_counts(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> BTreeMap<&'static str, usize> {
    let mut counts = BTreeMap::new();
    for (_, candidate) in observed {
        for (_, reason) in &candidate.method_probe_no_match_reasons {
            *counts.entry(reason.class_label()).or_insert(0) += 1;
        }
    }
    counts
}

fn method_probe_no_match_class_by_method(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for (_, candidate) in observed {
        for (method, reason) in &candidate.method_probe_no_match_reasons {
            let key = format!("{}/{}", method.metric_label(), reason.class_label());
            *counts.entry(key).or_insert(0) += 1;
        }
    }
    counts
}

fn method_probe_no_match_final_method_counts(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> BTreeMap<&'static str, usize> {
    let mut counts = BTreeMap::new();
    for (_, candidate) in observed {
        let no_match_count = candidate.method_probe_no_match_reasons.len();
        if no_match_count > 0 {
            *counts.entry(candidate.method.metric_label()).or_insert(0) += no_match_count;
        }
    }
    counts
}

fn method_probe_no_match_final_method_by_attempt(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for (_, candidate) in observed {
        for (attempted_method, _) in &candidate.method_probe_no_match_reasons {
            let key = format!(
                "{}/{}",
                attempted_method.metric_label(),
                candidate.method.metric_label()
            );
            *counts.entry(key).or_insert(0) += 1;
        }
    }
    counts
}

fn method_probe_terminal_no_match_reason_counts(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for (_, candidate) in observed {
        if !matches!(candidate.method, AlgorithmicIntegrationMethod::Unsupported) {
            continue;
        }
        for (method, reason) in &candidate.method_probe_no_match_reasons {
            let key = format!("{}/{}", method.metric_label(), reason.metric_label());
            *counts.entry(key).or_insert(0) += 1;
        }
    }
    counts
}

fn method_probe_terminal_no_match_class_counts(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> BTreeMap<&'static str, usize> {
    let mut counts = BTreeMap::new();
    for (_, candidate) in observed {
        if !matches!(candidate.method, AlgorithmicIntegrationMethod::Unsupported) {
            continue;
        }
        for (_, reason) in &candidate.method_probe_no_match_reasons {
            *counts.entry(reason.class_label()).or_insert(0) += 1;
        }
    }
    counts
}

fn method_probe_terminal_no_match_class_by_method(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for (_, candidate) in observed {
        if !matches!(candidate.method, AlgorithmicIntegrationMethod::Unsupported) {
            continue;
        }
        for (method, reason) in &candidate.method_probe_no_match_reasons {
            let key = format!("{}/{}", method.metric_label(), reason.class_label());
            *counts.entry(key).or_insert(0) += 1;
        }
    }
    counts
}

fn method_probe_terminal_candidate_count(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> usize {
    observed
        .iter()
        .filter(|(_, candidate)| {
            matches!(candidate.method, AlgorithmicIntegrationMethod::Unsupported)
                && !candidate.method_probe_no_match_reasons.is_empty()
        })
        .count()
}

fn method_probe_terminal_candidate_signature_counts(
    labeled_observed: &[(
        &'static str,
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> BTreeMap<&'static str, usize> {
    let mut counts = BTreeMap::new();
    for (signature, _, candidate) in labeled_observed {
        if matches!(candidate.method, AlgorithmicIntegrationMethod::Unsupported)
            && !candidate.method_probe_no_match_reasons.is_empty()
        {
            *counts.entry(*signature).or_insert(0) += 1;
        }
    }
    counts
}

fn verification_check_usage_by_method(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> BTreeMap<&'static str, usize> {
    let mut counts = BTreeMap::new();
    for (_, candidate) in observed {
        if candidate.verification_checks_used > 0 {
            *counts.entry(candidate.method.metric_label()).or_insert(0) +=
                candidate.verification_checks_used;
        }
    }
    counts
}

fn verification_check_usage_by_publication_status(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> BTreeMap<&'static str, usize> {
    let mut counts = BTreeMap::new();
    for (_, candidate) in observed {
        if candidate.verification_checks_used > 0 {
            *counts
                .entry(candidate.publication_status().metric_label())
                .or_insert(0) += candidate.verification_checks_used;
        }
    }
    counts
}

fn verification_check_usage_by_evidence(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> BTreeMap<&'static str, usize> {
    let mut counts = BTreeMap::new();
    for (_, candidate) in observed {
        if candidate.verification_checks_used > 0 {
            *counts
                .entry(candidate.verification_evidence.metric_label())
                .or_insert(0) += candidate.verification_checks_used;
        }
    }
    counts
}

fn verification_check_usage_by_method_and_evidence(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for (_, candidate) in observed {
        if candidate.verification_checks_used > 0 {
            let key = format!(
                "{}/{}",
                candidate.method.metric_label(),
                candidate.verification_evidence.metric_label()
            );
            *counts.entry(key).or_insert(0) += candidate.verification_checks_used;
        }
    }
    counts
}

fn verification_check_usage_by_method_and_publication_status(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for (_, candidate) in observed {
        if candidate.verification_checks_used > 0 {
            let key = format!(
                "{}/{}",
                candidate.method.metric_label(),
                candidate.publication_status().metric_label()
            );
            *counts.entry(key).or_insert(0) += candidate.verification_checks_used;
        }
    }
    counts
}

fn method_counts(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> BTreeMap<&'static str, usize> {
    let mut counts = BTreeMap::new();
    for (_, candidate) in observed {
        *counts.entry(candidate.method.metric_label()).or_insert(0) += 1;
    }
    counts
}

fn verification_status_by_method(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for (_, candidate) in observed {
        let key = format!(
            "{}/{}",
            candidate.method.metric_label(),
            candidate.verification_status.metric_label()
        );
        *counts.entry(key).or_insert(0) += 1;
    }
    counts
}

fn residual_reason_by_method(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for (_, candidate) in observed {
        if let Some(reason) = &candidate.residual_reason {
            let key = format!(
                "{}/{}",
                candidate.method.metric_label(),
                reason.metric_label()
            );
            *counts.entry(key).or_insert(0) += 1;
        }
    }
    counts
}

fn verification_blocker_counts(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> BTreeMap<&'static str, usize> {
    let mut counts = BTreeMap::new();
    for (_, candidate) in observed {
        if candidate.verification_blocker.is_present() {
            *counts
                .entry(candidate.verification_blocker.metric_label())
                .or_insert(0) += 1;
        }
    }
    counts
}

fn verification_blocker_by_method(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for (_, candidate) in observed {
        if candidate.verification_blocker.is_present() {
            let key = format!(
                "{}/{}",
                candidate.method.metric_label(),
                candidate.verification_blocker.metric_label()
            );
            *counts.entry(key).or_insert(0) += 1;
        }
    }
    counts
}

fn failure_class_counts(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> BTreeMap<&'static str, usize> {
    let mut counts = BTreeMap::new();
    for (_, candidate) in observed {
        if let Some(failure_class) = candidate.failure_class() {
            *counts.entry(failure_class.metric_label()).or_insert(0) += 1;
        }
    }
    counts
}

fn failure_class_by_method(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for (_, candidate) in observed {
        if let Some(failure_class) = candidate.failure_class() {
            let key = format!(
                "{}/{}",
                candidate.method.metric_label(),
                failure_class.metric_label()
            );
            *counts.entry(key).or_insert(0) += 1;
        }
    }
    counts
}

fn verification_residual_counts(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> BTreeMap<&'static str, usize> {
    let mut counts = BTreeMap::new();
    for (_, candidate) in observed {
        if candidate.verification_residual.is_some() {
            *counts.entry("derivative_minus_integrand").or_insert(0) += 1;
        }
    }
    counts
}

fn verification_residual_by_method(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for (_, candidate) in observed {
        if candidate.verification_residual.is_some() {
            let key = format!(
                "{}/{}",
                candidate.method.metric_label(),
                "derivative_minus_integrand"
            );
            *counts.entry(key).or_insert(0) += 1;
        }
    }
    counts
}

fn verification_residual_kind_counts(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> BTreeMap<&'static str, usize> {
    let mut counts = BTreeMap::new();
    for (_, candidate) in observed {
        if let Some(kind) = &candidate.verification_residual_kind {
            *counts.entry(kind.metric_label()).or_insert(0) += 1;
        }
    }
    counts
}

fn verification_residual_kind_by_method(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for (_, candidate) in observed {
        if let Some(kind) = &candidate.verification_residual_kind {
            let key = format!(
                "{}/{}",
                candidate.method.metric_label(),
                kind.metric_label()
            );
            *counts.entry(key).or_insert(0) += 1;
        }
    }
    counts
}

fn verification_residual_signature_counts(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> BTreeMap<&'static str, usize> {
    let mut counts = BTreeMap::new();
    for (_, candidate) in observed {
        if let Some(signature) = &candidate.verification_residual_signature {
            *counts.entry(signature.metric_label()).or_insert(0) += 1;
        }
    }
    counts
}

fn verification_residual_signature_by_method(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for (_, candidate) in observed {
        if let Some(signature) = &candidate.verification_residual_signature {
            let key = format!(
                "{}/{}",
                candidate.method.metric_label(),
                signature.metric_label()
            );
            *counts.entry(key).or_insert(0) += 1;
        }
    }
    counts
}

fn publication_status_counts(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> BTreeMap<&'static str, usize> {
    let mut counts = BTreeMap::new();
    for (_, candidate) in observed {
        *counts
            .entry(candidate.publication_status().metric_label())
            .or_insert(0) += 1;
    }
    counts
}

fn publication_status_by_method(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for (_, candidate) in observed {
        let key = format!(
            "{}/{}",
            candidate.method.metric_label(),
            candidate.publication_status().metric_label()
        );
        *counts.entry(key).or_insert(0) += 1;
    }
    counts
}

fn fallback_status_counts(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> BTreeMap<&'static str, usize> {
    let mut counts = BTreeMap::new();
    for (config, candidate) in observed {
        *counts
            .entry(candidate.fallback_status(*config).metric_label())
            .or_insert(0) += 1;
    }
    counts
}

fn fallback_status_by_method(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for (config, candidate) in observed {
        let key = format!(
            "{}/{}",
            candidate.method.metric_label(),
            candidate.fallback_status(*config).metric_label()
        );
        *counts.entry(key).or_insert(0) += 1;
    }
    counts
}

fn trace_level_counts(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> BTreeMap<&'static str, usize> {
    let mut counts = BTreeMap::new();
    for (_, candidate) in observed {
        *counts
            .entry(candidate.trace_level.metric_label())
            .or_insert(0) += 1;
    }
    counts
}

fn constant_policy_counts(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> BTreeMap<&'static str, usize> {
    let mut counts = BTreeMap::new();
    for (_, candidate) in observed {
        *counts
            .entry(candidate.constant_policy.metric_label())
            .or_insert(0) += 1;
    }
    counts
}

fn domain_policy_counts(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> BTreeMap<&'static str, usize> {
    let mut counts = BTreeMap::new();
    for (_, candidate) in observed {
        *counts
            .entry(candidate.domain_policy().metric_label())
            .or_insert(0) += 1;
    }
    counts
}

fn domain_policy_by_method(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for (_, candidate) in observed {
        let key = format!(
            "{}/{}",
            candidate.method.metric_label(),
            candidate.domain_policy().metric_label()
        );
        *counts.entry(key).or_insert(0) += 1;
    }
    counts
}

fn public_trace_level_counts(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> BTreeMap<&'static str, usize> {
    let mut counts = BTreeMap::new();
    for (_, candidate) in observed {
        if candidate.is_publicly_acceptable() {
            *counts
                .entry(candidate.trace_level.metric_label())
                .or_insert(0) += 1;
        }
    }
    counts
}

fn public_constant_policy_counts(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> BTreeMap<&'static str, usize> {
    let mut counts = BTreeMap::new();
    for (_, candidate) in observed {
        if candidate.is_publicly_acceptable() {
            *counts
                .entry(candidate.constant_policy.metric_label())
                .or_insert(0) += 1;
        }
    }
    counts
}

fn public_domain_policy_counts(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> BTreeMap<&'static str, usize> {
    let mut counts = BTreeMap::new();
    for (_, candidate) in observed {
        if candidate.is_publicly_acceptable() {
            *counts
                .entry(candidate.domain_policy().metric_label())
                .or_insert(0) += 1;
        }
    }
    counts
}

fn public_domain_policy_by_method(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for (_, candidate) in observed {
        if candidate.is_publicly_acceptable() {
            let key = format!(
                "{}/{}",
                candidate.method.metric_label(),
                candidate.domain_policy().metric_label()
            );
            *counts.entry(key).or_insert(0) += 1;
        }
    }
    counts
}

fn fallback_trace_level_counts(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> BTreeMap<&'static str, usize> {
    let mut counts = BTreeMap::new();
    for (config, candidate) in observed {
        if candidate.fallback_antiderivative(*config).is_some() {
            *counts
                .entry(candidate.trace_level.metric_label())
                .or_insert(0) += 1;
        }
    }
    counts
}

fn fallback_constant_policy_counts(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> BTreeMap<&'static str, usize> {
    let mut counts = BTreeMap::new();
    for (config, candidate) in observed {
        if candidate.fallback_antiderivative(*config).is_some() {
            *counts
                .entry(candidate.constant_policy.metric_label())
                .or_insert(0) += 1;
        }
    }
    counts
}

fn fallback_domain_policy_counts(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> BTreeMap<&'static str, usize> {
    let mut counts = BTreeMap::new();
    for (config, candidate) in observed {
        if candidate.fallback_antiderivative(*config).is_some() {
            *counts
                .entry(candidate.domain_policy().metric_label())
                .or_insert(0) += 1;
        }
    }
    counts
}

fn fallback_domain_policy_by_method(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for (config, candidate) in observed {
        if candidate.fallback_antiderivative(*config).is_some() {
            let key = format!(
                "{}/{}",
                candidate.method.metric_label(),
                candidate.domain_policy().metric_label()
            );
            *counts.entry(key).or_insert(0) += 1;
        }
    }
    counts
}

fn assumption_expr_count(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> usize {
    observed
        .iter()
        .map(|(_, candidate)| candidate.assumptions.len())
        .sum()
}

fn public_assumption_expr_count(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> usize {
    observed
        .iter()
        .filter(|(_, candidate)| candidate.is_publicly_acceptable())
        .map(|(_, candidate)| candidate.assumptions.len())
        .sum()
}

fn fallback_assumption_expr_count(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> usize {
    observed
        .iter()
        .filter(|(config, candidate)| candidate.fallback_antiderivative(*config).is_some())
        .map(|(_, candidate)| candidate.assumptions.len())
        .sum()
}

fn verification_evidence_counts(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> BTreeMap<&'static str, usize> {
    let mut counts = BTreeMap::new();
    for (_, candidate) in observed {
        *counts
            .entry(candidate.verification_evidence.metric_label())
            .or_insert(0) += 1;
    }
    counts
}

fn public_verification_evidence_counts(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> BTreeMap<&'static str, usize> {
    let mut counts = BTreeMap::new();
    for (_, candidate) in observed {
        if candidate.is_publicly_acceptable() {
            *counts
                .entry(candidate.verification_evidence.metric_label())
                .or_insert(0) += 1;
        }
    }
    counts
}

fn fallback_verification_evidence_counts(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> BTreeMap<&'static str, usize> {
    let mut counts = BTreeMap::new();
    for (config, candidate) in observed {
        if candidate.fallback_antiderivative(*config).is_some() {
            *counts
                .entry(candidate.verification_evidence.metric_label())
                .or_insert(0) += 1;
        }
    }
    counts
}

fn verification_evidence_by_method(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for (_, candidate) in observed {
        let key = format!(
            "{}/{}",
            candidate.method.metric_label(),
            candidate.verification_evidence.metric_label()
        );
        *counts.entry(key).or_insert(0) += 1;
    }
    counts
}

fn public_verification_evidence_by_method(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for (_, candidate) in observed {
        if candidate.is_publicly_acceptable() {
            let key = format!(
                "{}/{}",
                candidate.method.metric_label(),
                candidate.verification_evidence.metric_label()
            );
            *counts.entry(key).or_insert(0) += 1;
        }
    }
    counts
}

fn fallback_verification_evidence_by_method(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for (config, candidate) in observed {
        if candidate.fallback_antiderivative(*config).is_some() {
            let key = format!(
                "{}/{}",
                candidate.method.metric_label(),
                candidate.verification_evidence.metric_label()
            );
            *counts.entry(key).or_insert(0) += 1;
        }
    }
    counts
}

fn verification_normalization_reason_counts(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> BTreeMap<&'static str, usize> {
    let mut counts = BTreeMap::new();
    for (_, candidate) in observed {
        if candidate.verification_normalization_reason.is_present() {
            *counts
                .entry(candidate.verification_normalization_reason.metric_label())
                .or_insert(0) += 1;
        }
    }
    counts
}

fn public_verification_normalization_reason_counts(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> BTreeMap<&'static str, usize> {
    let mut counts = BTreeMap::new();
    for (_, candidate) in observed {
        if candidate.is_publicly_acceptable()
            && candidate.verification_normalization_reason.is_present()
        {
            *counts
                .entry(candidate.verification_normalization_reason.metric_label())
                .or_insert(0) += 1;
        }
    }
    counts
}

fn fallback_verification_normalization_reason_counts(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> BTreeMap<&'static str, usize> {
    let mut counts = BTreeMap::new();
    for (config, candidate) in observed {
        if candidate.fallback_antiderivative(*config).is_some()
            && candidate.verification_normalization_reason.is_present()
        {
            *counts
                .entry(candidate.verification_normalization_reason.metric_label())
                .or_insert(0) += 1;
        }
    }
    counts
}

fn verification_normalization_reason_by_method(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for (_, candidate) in observed {
        if candidate.verification_normalization_reason.is_present() {
            let key = format!(
                "{}/{}",
                candidate.method.metric_label(),
                candidate.verification_normalization_reason.metric_label()
            );
            *counts.entry(key).or_insert(0) += 1;
        }
    }
    counts
}

fn public_verification_normalization_reason_by_method(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for (_, candidate) in observed {
        if candidate.is_publicly_acceptable()
            && candidate.verification_normalization_reason.is_present()
        {
            let key = format!(
                "{}/{}",
                candidate.method.metric_label(),
                candidate.verification_normalization_reason.metric_label()
            );
            *counts.entry(key).or_insert(0) += 1;
        }
    }
    counts
}

fn fallback_verification_normalization_reason_by_method(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for (config, candidate) in observed {
        if candidate.fallback_antiderivative(*config).is_some()
            && candidate.verification_normalization_reason.is_present()
        {
            let key = format!(
                "{}/{}",
                candidate.method.metric_label(),
                candidate.verification_normalization_reason.metric_label()
            );
            *counts.entry(key).or_insert(0) += 1;
        }
    }
    counts
}

fn verification_normalization_reason_by_label(
    labeled_observed: &[(
        &str,
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for (label, _, candidate) in labeled_observed {
        if candidate.verification_normalization_reason.is_present() {
            let key = format!(
                "{}/{}",
                label,
                candidate.verification_normalization_reason.metric_label()
            );
            *counts.entry(key).or_insert(0) += 1;
        }
    }
    counts
}

fn verification_normalization_pass_count_counts(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for (_, candidate) in observed {
        *counts
            .entry(candidate.verification_normalization_passes_used.to_string())
            .or_insert(0) += 1;
    }
    counts
}

fn public_verification_normalization_pass_count_counts(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for (_, candidate) in observed {
        if candidate.is_publicly_acceptable() {
            *counts
                .entry(candidate.verification_normalization_passes_used.to_string())
                .or_insert(0) += 1;
        }
    }
    counts
}

fn fallback_verification_normalization_pass_count_counts(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for (config, candidate) in observed {
        if candidate.fallback_antiderivative(*config).is_some() {
            *counts
                .entry(candidate.verification_normalization_passes_used.to_string())
                .or_insert(0) += 1;
        }
    }
    counts
}

fn verification_normalization_pass_count_by_method(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for (_, candidate) in observed {
        let key = format!(
            "{}/{}",
            candidate.method.metric_label(),
            candidate.verification_normalization_passes_used
        );
        *counts.entry(key).or_insert(0) += 1;
    }
    counts
}

fn public_verification_normalization_pass_count_by_method(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for (_, candidate) in observed {
        if candidate.is_publicly_acceptable() {
            let key = format!(
                "{}/{}",
                candidate.method.metric_label(),
                candidate.verification_normalization_passes_used
            );
            *counts.entry(key).or_insert(0) += 1;
        }
    }
    counts
}

fn fallback_verification_normalization_pass_count_by_method(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for (config, candidate) in observed {
        if candidate.fallback_antiderivative(*config).is_some() {
            let key = format!(
                "{}/{}",
                candidate.method.metric_label(),
                candidate.verification_normalization_passes_used
            );
            *counts.entry(key).or_insert(0) += 1;
        }
    }
    counts
}

fn max_verification_normalization_passes(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> usize {
    observed
        .iter()
        .map(|(_, candidate)| candidate.verification_normalization_passes_used)
        .max()
        .unwrap_or(0)
}

fn public_max_verification_normalization_passes(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> usize {
    observed
        .iter()
        .filter(|(_, candidate)| candidate.is_publicly_acceptable())
        .map(|(_, candidate)| candidate.verification_normalization_passes_used)
        .max()
        .unwrap_or(0)
}

fn fallback_max_verification_normalization_passes(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> usize {
    observed
        .iter()
        .filter(|(config, candidate)| candidate.fallback_antiderivative(*config).is_some())
        .map(|(_, candidate)| candidate.verification_normalization_passes_used)
        .max()
        .unwrap_or(0)
}

fn verification_status_counts(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> BTreeMap<&'static str, usize> {
    let mut counts = BTreeMap::new();
    for (_, candidate) in observed {
        *counts
            .entry(candidate.verification_status.metric_label())
            .or_insert(0) += 1;
    }
    counts
}

fn residual_reason_counts(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> BTreeMap<&'static str, usize> {
    let mut counts = BTreeMap::new();
    for (_, candidate) in observed {
        if let Some(reason) = &candidate.residual_reason {
            *counts.entry(reason.metric_label()).or_insert(0) += 1;
        }
    }
    counts
}

fn mode_counts(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> BTreeMap<&'static str, usize> {
    let mut counts = BTreeMap::new();
    for (config, _) in observed {
        *counts.entry(config.mode.metric_label()).or_insert(0) += 1;
    }
    counts
}

fn required_condition_counts(
    observed: &[(
        AlgorithmicIntegrationBackendConfig,
        AlgorithmicIntegrationCandidate,
    )],
) -> BTreeMap<&'static str, usize> {
    let mut counts = BTreeMap::new();
    for (_, candidate) in observed {
        for condition in &candidate.required_conditions {
            let label = match condition {
                ConditionPredicate::NonZero(_) => "nonzero",
                ConditionPredicate::Positive(_) => "positive",
                ConditionPredicate::NonNegative(_) => "nonnegative",
                ConditionPredicate::LowerBound { .. } => "lower_bound",
                ConditionPredicate::Defined(_) => "defined",
                ConditionPredicate::InvTrigPrincipalRange { .. } => "inv_trig_principal_range",
                ConditionPredicate::EqZero(_) => "eq_zero",
                ConditionPredicate::EqOne(_) => "eq_one",
            };
            *counts.entry(label).or_insert(0) += 1;
        }
    }
    counts
}

fn json_count_map(counts: BTreeMap<&'static str, usize>) -> String {
    let entries = counts
        .into_iter()
        .map(|(key, value)| format!("\"{key}\":{value}"))
        .collect::<Vec<_>>()
        .join(",");
    format!("{{{entries}}}")
}

fn json_string_count_map(counts: BTreeMap<String, usize>) -> String {
    let entries = counts
        .into_iter()
        .map(|(key, value)| format!("\"{key}\":{value}"))
        .collect::<Vec<_>>()
        .join(",");
    format!("{{{entries}}}")
}

#[test]
fn backend_positive_quadratic_denominator_radius_recognizes_compact_and_expanded() {
    let mut ctx = Context::new();
    let compact = cas_parser::parse("(s*x+b)^2+a", &mut ctx).expect("compact denominator");
    let expanded =
        cas_parser::parse("s^2*x^2+2*b*s*x+b^2+a", &mut ctx).expect("expanded denominator");
    let indefinite =
        cas_parser::parse("(s*x+b)^2-a^2", &mut ctx).expect("indefinite square denominator");

    let compact_radius = backend_positive_quadratic_denominator_radius(&mut ctx, compact, "x")
        .expect("compact radius");
    let expanded_radius = backend_positive_quadratic_denominator_radius(&mut ctx, expanded, "x")
        .expect("expanded radius");

    assert!(crate::expr_domain::exprs_equivalent(
        &ctx,
        compact_radius,
        expanded_radius
    ));
    assert!(
        backend_positive_quadratic_denominator_radius(&mut ctx, indefinite, "x").is_none(),
        "indefinite square denominators must not be recognized as positive quadratics"
    );
}

#[test]
fn expanded_scaled_derivative_multiple_numerator_decomposes_to_log_only_coefficient() {
    let mut ctx = Context::new();
    let numerator = cas_parser::parse("m*s*x+b*m", &mut ctx).expect("numerator");
    let denominator = cas_parser::parse("s^2*x^2+2*b*s*x+b^2+a", &mut ctx).expect("denominator");

    let (center, _slope, _radius, _condition) =
        positive_shifted_quadratic_denominator_parts(&mut ctx, denominator, "x")
            .expect("expanded denominator reconstructs the affine center");

    let (coefficient, constant) =
        linear_numerator_decomposition_terms(&mut ctx, numerator, center, "x")
            .expect("distributed derivative-multiple numerator decomposes");

    let m = cas_parser::parse("m", &mut ctx).expect("external coefficient");
    assert!(crate::expr_domain::exprs_equivalent(&ctx, coefficient, m));
    assert!(
        is_zero(&ctx, constant),
        "whole-expression cancellation must yield a structural zero constant"
    );
}

#[test]
fn backend_difference_canceling_sum_term_cancels_whole_commuted_product() {
    let mut ctx = Context::new();
    let left = cas_parser::parse("b*m", &mut ctx).expect("left");
    let right = cas_parser::parse("m*b", &mut ctx).expect("right");

    let difference = build_backend_difference_canceling_sum_term(&mut ctx, left, right);

    assert!(is_zero(&ctx, difference));
}

#[test]
fn expanded_numeric_center_positive_quadratic_denominator_reconstructs() {
    let mut ctx = Context::new();
    let unit = cas_parser::parse("x^2+4*x+4+a", &mut ctx).expect("unit slope denominator");
    let scaled = cas_parser::parse("4*x^2+16*x+16+a", &mut ctx).expect("scaled denominator");
    let negative_center =
        cas_parser::parse("x^2-4*x+4+a", &mut ctx).expect("negative center denominator");

    let (center, slope, radius, condition) =
        positive_shifted_quadratic_denominator_parts(&mut ctx, unit, "x")
            .expect("unit numeric expanded denominator reconstructs");
    let expected_center = cas_parser::parse("x+2", &mut ctx).expect("expected center");
    assert!(crate::expr_domain::exprs_equivalent(
        &ctx,
        center,
        expected_center
    ));
    assert_eq!(
        slope,
        BackendAffineSlope::Numeric(num_rational::BigRational::from_integer(1.into()))
    );
    let a = cas_parser::parse("a", &mut ctx).expect("radius symbol");
    assert!(crate::expr_domain::exprs_equivalent(&ctx, radius, a));
    assert!(matches!(condition, Some(ConditionPredicate::Positive(_))));

    let (scaled_center, scaled_slope, _, _) =
        positive_shifted_quadratic_denominator_parts(&mut ctx, scaled, "x")
            .expect("scaled numeric expanded denominator reconstructs");
    let expected_scaled_center = cas_parser::parse("2*x+4", &mut ctx).expect("scaled center");
    assert!(crate::expr_domain::exprs_equivalent(
        &ctx,
        scaled_center,
        expected_scaled_center
    ));
    assert_eq!(
        scaled_slope,
        BackendAffineSlope::Numeric(num_rational::BigRational::from_integer(2.into()))
    );

    let (negative_center_expr, _, _, _) =
        positive_shifted_quadratic_denominator_parts(&mut ctx, negative_center, "x")
            .expect("negative numeric center reconstructs");
    let expected_negative_center = cas_parser::parse("x-2", &mut ctx).expect("negative center");
    assert!(crate::expr_domain::exprs_equivalent(
        &ctx,
        negative_center_expr,
        expected_negative_center
    ));
}

#[test]
fn expanded_numeric_center_positive_quadratic_denominator_rejects_unsafe_shapes() {
    let mut ctx = Context::new();
    let rejects = [
        "x^2+5*x+4+a",   // numeric constant != squared intercept
        "2*x^2+8*x+8+a", // irrational slope sqrt(2)
        "x^2+4*x+4",     // fully numeric: educational route owns it
        "x^2+4*x+4+1/2", // fully numeric with fractional literal (Div form)
        "x^2+4*x+4-1/2", // fully numeric with negative fractional remainder
        "x^2+4*x+4-a",   // minus-signed radius term
        "x^2+4*x+a",     // missing squared-intercept constant
    ];
    for source in rejects {
        let denominator = cas_parser::parse(source, &mut ctx).expect(source);
        assert!(
            positive_shifted_quadratic_denominator_parts(&mut ctx, denominator, "x").is_none(),
            "must reject {source}"
        );
    }
}

#[test]
fn backend_difference_canceling_sum_term_folds_numeric_operands() {
    let mut ctx = Context::new();
    let six = cas_parser::parse("6", &mut ctx).expect("six");
    let product = cas_parser::parse("3*2", &mut ctx).expect("product");
    let three = cas_parser::parse("3", &mut ctx).expect("three");
    let smaller = cas_parser::parse("1*2", &mut ctx).expect("smaller product");

    let zero = build_backend_difference_canceling_sum_term(&mut ctx, six, product);
    assert!(is_zero(&ctx, zero));

    let one = build_backend_difference_canceling_sum_term(&mut ctx, three, smaller);
    assert_eq!(
        numeric_value(&ctx, one),
        Some(num_rational::BigRational::from_integer(1.into()))
    );
}

#[test]
fn algebraic_zero_test_decides_compact_vs_expanded_rational_identity() {
    let mut ctx = Context::new();
    let derivative = cas_parser::parse("(x+b)/((x+b)^2+a)", &mut ctx).expect("derivative");
    let integrand = cas_parser::parse("(x+b)/(x^2+2*b*x+b^2+a)", &mut ctx).expect("integrand");

    assert_eq!(
        algebraic_rational_zero_test(&ctx, derivative, integrand, "x", &[]),
        Some(true)
    );
}

#[test]
fn algebraic_zero_test_reduces_sqrt_atoms_by_quotient_relation() {
    let mut ctx = Context::new();
    let derivative = cas_parser::parse("c/(sqrt(a)*sqrt(a)*(1+((x+b)/sqrt(a))^2))", &mut ctx)
        .expect("derivative");
    let integrand = cas_parser::parse("c/((x+b)^2+a)", &mut ctx).expect("integrand");
    let radius = cas_parser::parse("a", &mut ctx).expect("radius");
    let conditions = vec![ConditionPredicate::Positive(radius)];

    assert_eq!(
        algebraic_rational_zero_test(&ctx, derivative, integrand, "x", &conditions),
        Some(true)
    );
    assert_eq!(
        algebraic_rational_zero_test(&ctx, derivative, integrand, "x", &[]),
        None,
        "the quotient relation must not run without a represented \
         non-negativity condition for the radicand"
    );
}

#[test]
fn algebraic_zero_test_refutes_genuine_mismatches() {
    let mut ctx = Context::new();
    let derivative = cas_parser::parse("1", &mut ctx).expect("derivative");
    let integrand = cas_parser::parse("x", &mut ctx).expect("integrand");

    assert_eq!(
        algebraic_rational_zero_test(&ctx, derivative, integrand, "x", &[]),
        Some(false)
    );
}

#[test]
fn algebraic_zero_test_bails_out_of_scope_shapes() {
    let mut ctx = Context::new();
    let one = cas_parser::parse("1", &mut ctx).expect("one");
    let sine = cas_parser::parse("sin(x)", &mut ctx).expect("sine");
    let variable_radicand = cas_parser::parse("1/sqrt(x)", &mut ctx).expect("variable radicand");
    let half_power = cas_parser::parse("x/(2*sqrt(x))", &mut ctx).expect("half power");

    assert_eq!(
        algebraic_rational_zero_test(&ctx, one, sine, "x", &[]),
        None
    );
    assert_eq!(
        algebraic_rational_zero_test(&ctx, variable_radicand, half_power, "x", &[]),
        None,
        "radicands containing the integration variable are out of scope"
    );
}

#[test]
fn algebraic_zero_test_keeps_radicand_only_parameters_in_the_universe() {
    // Regression for the silent-projection blocker: `b` occurs only inside
    // the radicand, and dropping it from the variable universe degenerated
    // the relation t^2 = b into t^2 = 1, proving sqrt(b)^2*x == x.
    let mut ctx = Context::new();
    let derivative = cas_parser::parse("sqrt(b)*sqrt(b)*x", &mut ctx).expect("derivative");
    let integrand = cas_parser::parse("x", &mut ctx).expect("integrand");
    let b = cas_parser::parse("b", &mut ctx).expect("b");
    let conditions = vec![ConditionPredicate::Positive(b)];

    assert_ne!(
        algebraic_rational_zero_test(&ctx, derivative, integrand, "x", &conditions),
        Some(true)
    );

    let true_match = cas_parser::parse("b*x", &mut ctx).expect("b*x");
    assert_eq!(
        algebraic_rational_zero_test(&ctx, derivative, true_match, "x", &conditions),
        Some(true),
        "the genuine identity sqrt(b)^2*x == b*x must still verify"
    );
}

#[test]
fn multi_quadratic_partial_fraction_decomposes_distinct_irreducible_quadratics() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("(x^3+x+1)/((x^2+1)*(x^2+4))", &mut ctx).expect("integrand");

    let terms =
        multi_quadratic_partial_fraction_terms(&mut ctx, integrand, "x").expect("decomposition");
    assert_eq!(terms.len(), 2);
    let third = num_rational::BigRational::new(1.into(), 3.into());
    assert!(num_traits::Zero::is_zero(&terms[0].alpha));
    assert_eq!(terms[0].beta, third);
    assert_eq!(
        terms[1].alpha,
        num_rational::BigRational::from_integer(1.into())
    );
    assert_eq!(terms[1].beta, -third);
}

#[test]
fn multi_quadratic_partial_fraction_rejects_out_of_scope_shapes() {
    let mut ctx = Context::new();
    let rejects = [
        "1/((x^2+1)*(x^2+1))",   // repeated factor
        "1/((x^2-1)*(x^2+4))",   // reducible factor
        "x^4/((x^2+1)*(x^2+4))", // improper numerator
        "1/(x^2+1)",             // single factor
        "1/((x^2+a)*(x^2+4))",   // symbolic coefficients
    ];
    for source in rejects {
        let integrand = cas_parser::parse(source, &mut ctx).expect(source);
        assert!(
            multi_quadratic_partial_fraction_terms(&mut ctx, integrand, "x").is_none(),
            "must reject {source}"
        );
    }
}

#[test]
fn multi_quadratic_candidate_is_verified_and_accepted() {
    let mut ctx = Context::new();
    let integrand = cas_parser::parse("(x^3+x+1)/((x^2+1)*(x^2+4))", &mut ctx).expect("integrand");

    let candidate = try_algorithmic_integration_backend(
        &mut ctx,
        integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );

    assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Rational);
    assert_eq!(
        candidate.verification_status,
        AlgorithmicIntegrationVerificationStatus::Verified
    );
    assert!(
        candidate.required_conditions.is_empty(),
        "irreducible numeric quadratics are strictly positive: unconditional"
    );
}

#[test]
fn algebraic_zero_test_folds_unfolded_numeric_exponents() {
    // Regression: derivatives produce shapes like x^(2 - 1); literal-only
    // exponent matching made the zero test bail on decidable rational pairs.
    let mut ctx = Context::new();
    let derivative = cas_parser::parse("(x^(2-1)*2*(1/2))/(x^2+4)", &mut ctx).expect("derivative");
    let integrand = cas_parser::parse("x/(x^2+4)", &mut ctx).expect("integrand");

    assert_eq!(
        algebraic_rational_zero_test(&ctx, derivative, integrand, "x", &[]),
        Some(true)
    );
}

#[test]
fn general_rational_pipeline_reduces_and_verifies_expanded_sextic() {
    let mut ctx = Context::new();
    // (x^2+1)*(x^2+4)^2 expanded: Ostrogradsky extracts P/(x^2+4), the
    // squarefree remainder splits as a rational-root biquadratic.
    let integrand = cas_parser::parse("1/(x^6+9*x^4+24*x^2+16)", &mut ctx).expect("integrand");

    let candidate = try_algorithmic_integration_backend(
        &mut ctx,
        integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );

    assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Rational);
    assert_eq!(
        candidate.verification_status,
        AlgorithmicIntegrationVerificationStatus::Verified
    );
    assert!(candidate.required_conditions.is_empty());
}

#[test]
fn general_rational_pipeline_emits_pole_conditions_for_linear_factors() {
    let mut ctx = Context::new();
    // x*(x^2+1)^2: one rational pole, repeated quadratic.
    let integrand = cas_parser::parse("1/(x^5+2*x^3+x)", &mut ctx).expect("integrand");

    let candidate = try_algorithmic_integration_backend(
        &mut ctx,
        integrand,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    );

    assert_eq!(
        candidate.verification_status,
        AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
    );
    assert_eq!(candidate.required_conditions.len(), 1);
    assert!(matches!(
        candidate.required_conditions[0],
        ConditionPredicate::NonZero(_)
    ));
}

#[test]
fn general_rational_pipeline_rejects_out_of_scope_shapes() {
    let mut ctx = Context::new();
    let rejects = [
        "1/(x^4+2*x^2+3)",     // even quartic needing irrational coefficients
        "1/(x^4-x^2+1)",       // cyclotomic Phi_12: irreducible over Q
        "1/((x^2-2)*(x^2+1))", // irrational real poles
        "1/(x^2+1)",           // degree window: <= 2 owned by existing routes
        "1/(x^4+a)",           // symbolic coefficients
    ];
    for source in rejects {
        let integrand = cas_parser::parse(source, &mut ctx).expect(source);
        assert!(
            general_rational_partial_fraction_antiderivative(&mut ctx, integrand, "x").is_none(),
            "must reject {source}"
        );
    }
}

#[test]
fn general_rational_pipeline_keeps_pole_condition_for_zero_residue_poles() {
    let mut ctx = Context::new();
    // -(x^2+1)/((x-1)^2*(x^2+1)) == -1/(x-1)^2: the ln term vanishes
    // (zero residue) but the antiderivative 1/(x-1) keeps the pole, so
    // the x != 1 condition must survive.
    let integrand = cas_parser::parse("(-(x^2+1))/(x^4 - 2*x^3 + 2*x^2 - 2*x + 1)", &mut ctx)
        .expect("integrand");

    let parts = general_rational_partial_fraction_antiderivative(&mut ctx, integrand, "x");
    if let Some(parts) = parts {
        assert_eq!(
            parts.pole_conditions.len(),
            1,
            "zero-residue pole inside P/D1 must keep its NonZero condition"
        );
    }
}

#[test]
fn even_quartic_descent_splits_sophie_germain_and_cyclotomic_quartics() {
    let mut ctx = Context::new();
    for source in ["1/(x^4+4)", "(x^2+1)/(x^4+x^2+1)"] {
        let integrand = cas_parser::parse(source, &mut ctx).expect(source);
        let candidate = try_algorithmic_integration_backend(
            &mut ctx,
            integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );
        assert_eq!(
            candidate.verification_status,
            AlgorithmicIntegrationVerificationStatus::Verified,
            "{source} must verify"
        );
        assert!(candidate.required_conditions.is_empty(), "{source}");
    }
}

#[test]
fn even_quartic_descent_rejects_irrational_and_irreducible_quartics() {
    let mut ctx = Context::new();
    let rejects = [
        "1/(x^4+2*x^2+3)", // r = 3 is not a perfect square: needs irrational coefficients
        "1/(x^4-x^2+1)",   // cyclotomic Phi_12: irreducible over Q (a^2 = 3 irrational)
    ];
    for source in rejects {
        let integrand = cas_parser::parse(source, &mut ctx).expect(source);
        assert!(
            general_rational_partial_fraction_antiderivative(&mut ctx, integrand, "x").is_none(),
            "must reject {source}"
        );
    }
}

#[test]
fn resolvent_cubic_splits_non_even_quartics() {
    let mut ctx = Context::new();
    // (x^2+x+1)*(x^2+2) and (x^2+1)*(x^2+2x+3), both expanded.
    for source in ["1/(x^4+x^3+3*x^2+2*x+2)", "1/(x^4+2*x^3+4*x^2+2*x+3)"] {
        let integrand = cas_parser::parse(source, &mut ctx).expect(source);
        let candidate = try_algorithmic_integration_backend(
            &mut ctx,
            integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );
        assert_eq!(
            candidate.verification_status,
            AlgorithmicIntegrationVerificationStatus::Verified,
            "{source} must verify"
        );
    }
}

#[test]
fn resolvent_cubic_rejects_irreducible_non_even_quartics() {
    let mut ctx = Context::new();
    // Phi_5 cyclotomic: irreducible over Q (resolvent has no rational
    // perfect-square root).
    let integrand = cas_parser::parse("1/(x^4+x^3+x^2+x+1)", &mut ctx).expect("integrand");
    assert!(
        general_rational_partial_fraction_antiderivative(&mut ctx, integrand, "x").is_none(),
        "must reject Phi_5"
    );
}

#[test]
fn symmetric_surd_even_quartic_splits_phi12_and_scaled_constants() {
    let mut ctx = Context::new();
    // Phi_12 = x^4-x^2+1 (factor sqrt(3)), a scaled numerator, and x^4-3x^2+4
    // (factor sqrt(7), s = 2): the symmetric surd pair is the cyclotomic case
    // even_quartic_descent declines because the linear coefficient is irrational.
    for source in ["1/(x^4-x^2+1)", "2/(x^4-x^2+1)", "1/(x^4-3*x^2+4)"] {
        let integrand = cas_parser::parse(source, &mut ctx).expect(source);
        let candidate = try_algorithmic_integration_backend(
            &mut ctx,
            integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );
        assert_eq!(
            candidate.verification_status,
            AlgorithmicIntegrationVerificationStatus::Verified,
            "{source} must verify by differentiation"
        );
        assert!(
            candidate.required_conditions.is_empty(),
            "{source}: the irreducible quadratics are strictly positive, so no conditions"
        );
    }
}

#[test]
fn symmetric_surd_even_quartic_rejects_out_of_scope_shapes() {
    let mut ctx = Context::new();
    let rejects = [
        "1/(x^4+2*x^2+3)",   // r = 3 not a perfect square: s = sqrt(3) irrational
        "1/(x^4+x^2+1)",     // a^2 = 1 is a perfect square: owned by even_quartic_descent
        "(x+1)/(x^4-x^2+1)", // non-constant numerator: symmetric collapse does not apply
        "1/(x^4+3*x^2+1)", // a^2 = 2 - 3 < 0: real factors carry irrational constants, not a surd linear term
        "1/(x^2+1)",       // not a quartic denominator
    ];
    for source in rejects {
        let integrand = cas_parser::parse(source, &mut ctx).expect(source);
        assert!(
            symmetric_surd_even_quartic_antiderivative(&mut ctx, integrand, "x").is_none(),
            "must reject {source}"
        );
    }
}
