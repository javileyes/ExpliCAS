use crate::rule::Rule;
use crate::rules::calculus::IntegrateRule;
use cas_ast::{ConditionPredicate, Context};
use cas_formatter::DisplayExpr;
use cas_math::general_integration_backend::{
    try_algorithmic_integration_backend, AlgorithmicIntegrationBackendBudget,
    AlgorithmicIntegrationBackendConfig, AlgorithmicIntegrationMethod,
};
use cas_parser::parse;

use super::{
    integrate_with_trace, integrate_with_trace_with_backend_config,
    public_algorithmic_backend_fallback, IntegrationTraceKind,
};

#[test]
fn test_integrate_power() {
    let mut ctx = Context::new();
    let rule = IntegrateRule;
    // integrate(x^2, x) -> x^3/3
    let expr = parse("integrate(x^2, x)", &mut ctx).unwrap();
    let rewrite = rule
        .apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        )
        .unwrap();
    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        ),
        "x^3 / 3"
    );
}

#[test]
fn test_integrate_constant() {
    let mut ctx = Context::new();
    let rule = IntegrateRule;
    // integrate(5, x) -> 5x
    let expr = parse("integrate(5, x)", &mut ctx).unwrap();
    let rewrite = rule
        .apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        )
        .unwrap();
    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        ),
        "5 * x"
    );
}

#[test]
fn test_integrate_trig() {
    let mut ctx = Context::new();
    let rule = IntegrateRule;
    // integrate(sin(x), x) -> -cos(x)
    let expr = parse("integrate(sin(x), x)", &mut ctx).unwrap();
    let rewrite = rule
        .apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        )
        .unwrap();
    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        ),
        "-cos(x)"
    );
}

#[test]
fn test_integrate_linearity() {
    let mut ctx = Context::new();
    let rule = IntegrateRule;
    // integrate(x + 1, x) -> x^2/2 + x
    let expr = parse("integrate(x + 1, x)", &mut ctx).unwrap();
    let rewrite = rule
        .apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        )
        .unwrap();
    let res = format!(
        "{}",
        DisplayExpr {
            context: &ctx,
            id: rewrite.new_expr
        }
    );
    // x^2/2 + 1*x
    assert!(res.contains("x^2 / 2"));
    assert!(res.contains("1 * x") || res.contains("x"));
}

#[test]
fn test_integrate_scaled_denominator_square_carries_required_domain() {
    let mut ctx = Context::new();
    let rule = IntegrateRule;
    let expr = parse("integrate((2*x+1)/(3*(x^2+x-1)^2), x)", &mut ctx).unwrap();
    let rewrite = rule
        .apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        )
        .unwrap();

    let required: Vec<_> = rewrite
        .required_conditions
        .iter()
        .map(|condition| condition.display(&ctx))
        .collect();
    assert_eq!(required, vec!["x^2 + x - 1 ≠ 0".to_string()]);
}

#[test]
fn test_integrate_algorithmic_backend_mode_boundary_controls_public_fallback() {
    let mut ctx = Context::new();
    let educational_target = parse("x^2", &mut ctx).unwrap();
    let disabled_educational = integrate_with_trace_with_backend_config(
        &mut ctx,
        educational_target,
        "x",
        AlgorithmicIntegrationBackendConfig::disabled(),
    )
    .unwrap();

    assert_eq!(
        disabled_educational.trace_kind,
        IntegrationTraceKind::EducationalRule
    );
    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: disabled_educational.result
            }
        ),
        "x^3 / 3"
    );

    let backend_target = parse("1/(x^2-a^2)", &mut ctx).unwrap();
    assert!(integrate_with_trace_with_backend_config(
        &mut ctx,
        backend_target,
        "x",
        AlgorithmicIntegrationBackendConfig::disabled(),
    )
    .is_none());
    assert!(integrate_with_trace_with_backend_config(
        &mut ctx,
        backend_target,
        "x",
        AlgorithmicIntegrationBackendConfig::diagnostic_only(),
    )
    .is_none());

    let fallback_outcome = integrate_with_trace_with_backend_config(
        &mut ctx,
        backend_target,
        "x",
        AlgorithmicIntegrationBackendConfig::residual_fallback(),
    )
    .unwrap();
    assert_eq!(
        fallback_outcome.trace_kind,
        IntegrationTraceKind::AlgorithmicBackendSummary
    );
    assert!(!format!(
        "{}",
        DisplayExpr {
            context: &ctx,
            id: fallback_outcome.result
        }
    )
    .starts_with("integrate("));
    assert_eq!(fallback_outcome.required_conditions.len(), 3);
}

#[test]
fn test_integrate_algorithmic_backend_mode_boundary_clamps_public_budget() {
    let permissive_config = AlgorithmicIntegrationBackendConfig::residual_fallback()
        .with_budget(AlgorithmicIntegrationBackendBudget::new(99, 99));

    let mut direct_ctx = Context::new();
    let direct_target = parse("cos(x)/sin(x)", &mut direct_ctx).unwrap();
    let direct_candidate =
        try_algorithmic_integration_backend(&mut direct_ctx, direct_target, "x", permissive_config);
    assert_eq!(
        direct_candidate.method,
        AlgorithmicIntegrationMethod::HeurischProbe
    );
    assert!(direct_candidate
        .fallback_antiderivative(permissive_config)
        .is_some());

    let mut public_ctx = Context::new();
    let public_target = parse("cos(x)/sin(x)", &mut public_ctx).unwrap();
    assert!(public_algorithmic_backend_fallback(
        &mut public_ctx,
        public_target,
        "x",
        permissive_config,
    )
    .is_none());

    println!(
        "algorithmic_backend_mode_boundary: {{\"direct_high_budget_method\":\"{}\",\"public_budget_clamped\":1,\"public_clamp_rejected\":1}}",
        direct_candidate.method.metric_label()
    );
}

#[test]
fn test_integrate_algorithmic_backend_affine_quotient_carries_required_domain() {
    let mut ctx = Context::new();
    let symbolic_slope_target = parse("(3*x+c)/(2*a*x+b)", &mut ctx).unwrap();
    let symbolic_slope_outcome =
        integrate_with_trace(&mut ctx, symbolic_slope_target, "x").unwrap();

    assert_eq!(
        symbolic_slope_outcome.trace_kind,
        IntegrationTraceKind::AlgorithmicBackendSummary
    );
    let mut symbolic_required: Vec<_> = symbolic_slope_outcome
        .required_conditions
        .iter()
        .map(|condition| match condition {
            ConditionPredicate::NonZero(expr) => format!(
                "{} ≠ 0",
                DisplayExpr {
                    context: &ctx,
                    id: *expr,
                }
            ),
            _ => condition.display(),
        })
        .collect();
    symbolic_required.sort();
    assert_eq!(
        symbolic_required,
        vec!["2 * a * x + b ≠ 0".to_string(), "2 * a ≠ 0".to_string()]
    );
    let symbolic_result = format!(
        "{}",
        DisplayExpr {
            context: &ctx,
            id: symbolic_slope_outcome.result
        }
    );
    assert!(symbolic_result.contains("ln(|2 * a * x + b|)"));
    assert!(!symbolic_result.contains("0 *"));
    assert!(!symbolic_result.starts_with("integrate("));

    let external_scale_target = parse("a*x/(c*x+d)", &mut ctx).unwrap();
    let external_scale_outcome =
        integrate_with_trace(&mut ctx, external_scale_target, "x").unwrap();

    assert_eq!(
        external_scale_outcome.trace_kind,
        IntegrationTraceKind::AlgorithmicBackendSummary
    );
    let mut external_required: Vec<_> = external_scale_outcome
        .required_conditions
        .iter()
        .map(|condition| match condition {
            ConditionPredicate::NonZero(expr) => format!(
                "{} ≠ 0",
                DisplayExpr {
                    context: &ctx,
                    id: *expr,
                }
            ),
            _ => condition.display(),
        })
        .collect();
    external_required.sort();
    assert_eq!(
        external_required,
        vec!["c * x + d ≠ 0".to_string(), "c ≠ 0".to_string()]
    );
    let external_result = format!(
        "{}",
        DisplayExpr {
            context: &ctx,
            id: external_scale_outcome.result
        }
    );
    assert!(external_result.contains("ln(|c * x + d|)"));
    assert!(!external_result.contains("0 *"));
    assert!(!external_result.starts_with("integrate("));

    let target = parse("(3*x+c)/(2*x+b)", &mut ctx).unwrap();
    let outcome = integrate_with_trace(&mut ctx, target, "x").unwrap();

    assert_eq!(
        outcome.trace_kind,
        IntegrationTraceKind::AlgorithmicBackendSummary
    );
    assert_eq!(outcome.required_conditions.len(), 1);
    assert!(!format!(
        "{}",
        DisplayExpr {
            context: &ctx,
            id: outcome.result
        }
    )
    .starts_with("integrate("));

    let rule = IntegrateRule;
    let expr = parse("integrate((3*x+c)/(2*x+b), x)", &mut ctx).unwrap();
    let rewrite = rule
        .apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        )
        .unwrap();
    let mut required: Vec<_> = rewrite
        .required_conditions
        .iter()
        .map(|condition| condition.display(&ctx))
        .collect();
    required.sort();
    assert_eq!(required, vec!["b + 2 * x ≠ 0".to_string()]);
}

#[test]
fn test_integrate_algorithmic_backend_hermite_positive_radius_carries_required_domain() {
    let mut ctx = Context::new();
    let target = parse("(x+1)/(x^2+a)", &mut ctx).unwrap();
    let outcome = integrate_with_trace(&mut ctx, target, "x").unwrap();

    assert_eq!(
        outcome.trace_kind,
        IntegrationTraceKind::AlgorithmicBackendSummary
    );
    let required: Vec<_> = outcome
        .required_conditions
        .iter()
        .map(|condition| match condition {
            ConditionPredicate::Positive(expr) => format!(
                "{} > 0",
                DisplayExpr {
                    context: &ctx,
                    id: *expr,
                }
            ),
            ConditionPredicate::NonZero(expr) => format!(
                "{} ≠ 0",
                DisplayExpr {
                    context: &ctx,
                    id: *expr,
                }
            ),
            _ => condition.display(),
        })
        .collect();
    assert_eq!(required, vec!["a > 0".to_string()]);

    let result = format!(
        "{}",
        DisplayExpr {
            context: &ctx,
            id: outcome.result
        }
    );
    assert!(result.contains("ln(x^2 + a)"));
    assert!(result.contains("arctan(x / sqrt(a))"));
    assert!(!result.starts_with("integrate("));

    let shifted_symbolic_slope = parse("(m*(s*x+b)+c)/((s*x+b)^2+a)", &mut ctx).unwrap();
    let shifted_outcome = integrate_with_trace(&mut ctx, shifted_symbolic_slope, "x").unwrap();

    assert_eq!(
        shifted_outcome.trace_kind,
        IntegrationTraceKind::AlgorithmicBackendSummary
    );
    let mut shifted_required: Vec<_> = shifted_outcome
        .required_conditions
        .iter()
        .map(|condition| match condition {
            ConditionPredicate::Positive(expr) => format!(
                "{} > 0",
                DisplayExpr {
                    context: &ctx,
                    id: *expr,
                }
            ),
            ConditionPredicate::NonZero(expr) => format!(
                "{} ≠ 0",
                DisplayExpr {
                    context: &ctx,
                    id: *expr,
                }
            ),
            _ => condition.display(),
        })
        .collect();
    shifted_required.sort();
    assert_eq!(
        shifted_required,
        vec!["a > 0".to_string(), "s ≠ 0".to_string()]
    );
    let shifted_result = format!(
        "{}",
        DisplayExpr {
            context: &ctx,
            id: shifted_outcome.result
        }
    );
    assert!(shifted_result.contains("ln((s * x + b)^2 + a)"));
    assert!(shifted_result.contains("arctan((s * x + b) / sqrt(a))"));
    assert!(!shifted_result.starts_with("integrate("));

    let expanded_shifted_symbolic_slope =
        parse("(m*s*x+b*m+c)/(s^2*x^2+2*b*s*x+b^2+a)", &mut ctx).unwrap();
    let expanded_shifted_outcome =
        integrate_with_trace(&mut ctx, expanded_shifted_symbolic_slope, "x").unwrap();

    assert_eq!(
        expanded_shifted_outcome.trace_kind,
        IntegrationTraceKind::AlgorithmicBackendSummary
    );
    let mut expanded_shifted_required: Vec<_> = expanded_shifted_outcome
        .required_conditions
        .iter()
        .map(|condition| match condition {
            ConditionPredicate::Positive(expr) => format!(
                "{} > 0",
                DisplayExpr {
                    context: &ctx,
                    id: *expr,
                }
            ),
            ConditionPredicate::NonZero(expr) => format!(
                "{} ≠ 0",
                DisplayExpr {
                    context: &ctx,
                    id: *expr,
                }
            ),
            _ => condition.display(),
        })
        .collect();
    expanded_shifted_required.sort();
    assert_eq!(
        expanded_shifted_required,
        vec!["a > 0".to_string(), "s ≠ 0".to_string()]
    );
    let expanded_shifted_result = format!(
        "{}",
        DisplayExpr {
            context: &ctx,
            id: expanded_shifted_outcome.result
        }
    );
    assert!(!expanded_shifted_result.starts_with("integrate("));
}

#[test]
fn test_integrate_algorithmic_backend_hermite_indefinite_square_carries_required_domain() {
    let mut ctx = Context::new();
    let target = parse("1/(x^2-a^2)", &mut ctx).unwrap();
    let outcome = integrate_with_trace(&mut ctx, target, "x").unwrap();

    assert_eq!(
        outcome.trace_kind,
        IntegrationTraceKind::AlgorithmicBackendSummary
    );
    let mut required: Vec<_> = outcome
        .required_conditions
        .iter()
        .map(|condition| match condition {
            ConditionPredicate::NonZero(expr) => format!(
                "{} ≠ 0",
                DisplayExpr {
                    context: &ctx,
                    id: *expr,
                }
            ),
            _ => condition.display(),
        })
        .collect();
    required.sort();
    assert_eq!(
        required,
        vec![
            "a + x ≠ 0".to_string(),
            "a ≠ 0".to_string(),
            "x - a ≠ 0".to_string(),
        ]
    );

    let result = format!(
        "{}",
        DisplayExpr {
            context: &ctx,
            id: outcome.result
        }
    );
    assert!(result.contains("ln(|x - a|)"));
    assert!(result.contains("ln(|a + x|)"));
    assert!(!result.starts_with("integrate("));

    let affine_mixed = parse("(m*(s*x+b)+c)/((s*x+b)^2-a^2)", &mut ctx).unwrap();
    let affine_outcome = integrate_with_trace(&mut ctx, affine_mixed, "x").unwrap();

    assert_eq!(
        affine_outcome.trace_kind,
        IntegrationTraceKind::AlgorithmicBackendSummary
    );
    let mut affine_required: Vec<_> = affine_outcome
        .required_conditions
        .iter()
        .map(|condition| match condition {
            ConditionPredicate::NonZero(expr) => format!(
                "{} ≠ 0",
                DisplayExpr {
                    context: &ctx,
                    id: *expr,
                }
            ),
            _ => condition.display(),
        })
        .collect();
    affine_required.sort();
    assert_eq!(
        affine_required,
        vec![
            "a ≠ 0".to_string(),
            "s * x + a + b ≠ 0".to_string(),
            "s * x + b - a ≠ 0".to_string(),
            "s ≠ 0".to_string(),
        ]
    );

    let affine_result = format!(
        "{}",
        DisplayExpr {
            context: &ctx,
            id: affine_outcome.result
        }
    );
    assert!(affine_result.contains("ln(|(s * x + b)^2 - a^2|)"));
    assert!(affine_result.contains("ln(|s * x + b - a|)"));
    assert!(affine_result.contains("ln(|s * x + a + b|)"));
    assert!(!affine_result.starts_with("integrate("));
}

#[test]
fn test_integrate_linear_subst_trig() {
    let mut ctx = Context::new();
    let rule = IntegrateRule;
    // integrate(sin(2*x), x) -> -cos(2*x)/2
    let expr = parse("integrate(sin(2*x), x)", &mut ctx).unwrap();
    let rewrite = rule
        .apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        )
        .unwrap();
    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        ),
        "-1/2 * cos(2 * x)"
    );
}

#[test]
fn test_integrate_linear_subst_exp() {
    let mut ctx = Context::new();
    let rule = IntegrateRule;
    // integrate(exp(3*x), x) -> exp(3*x)/3
    let expr = parse("integrate(exp(3*x), x)", &mut ctx).unwrap();
    let rewrite = rule
        .apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        )
        .unwrap();
    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        ),
        "1/3 * e^(3 * x)"
    );
}

#[test]
fn test_integrate_linear_subst_power() {
    let mut ctx = Context::new();
    let rule = IntegrateRule;
    // integrate((2*x + 1)^2, x) -> (2*x + 1)^3 / (2*3) -> (2*x+1)^3 / 6
    let expr = parse("integrate((2*x + 1)^2, x)", &mut ctx).unwrap();
    let rewrite = rule
        .apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        )
        .unwrap();
    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        ),
        "(2 * x + 1)^3 / (2 * 3)"
    );
}

#[test]
fn test_integrate_linear_subst_log() {
    let mut ctx = Context::new();
    let rule = IntegrateRule;
    // integrate(1/(3*x), x) -> ln(abs(3*x))/3
    let expr = parse("integrate(1/(3*x), x)", &mut ctx).unwrap();
    let rewrite = rule
        .apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        )
        .unwrap();
    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        ),
        "1/3 * ln(|3 * x|)"
    );
}

#[test]
fn test_integrate_inverse_function_kernels() {
    let mut ctx = Context::new();
    let rule = IntegrateRule;

    let arctan_expr = parse("integrate(1/(x^2+1), x)", &mut ctx).unwrap();
    let arctan_rewrite = rule
        .apply(
            &mut ctx,
            arctan_expr,
            &crate::parent_context::ParentContext::root(),
        )
        .unwrap();
    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: arctan_rewrite.new_expr
            }
        ),
        "arctan(x)"
    );

    let asinh_expr = parse("integrate((x^2+1)^(-1/2), x)", &mut ctx).unwrap();
    let asinh_rewrite = rule
        .apply(
            &mut ctx,
            asinh_expr,
            &crate::parent_context::ParentContext::root(),
        )
        .unwrap();
    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: asinh_rewrite.new_expr
            }
        ),
        "asinh(x)"
    );
}

#[test]
fn test_integrate_atanh_surd_open_interval_condition_compacts_to_denominator() {
    let mut ctx = Context::new();
    let rule = IntegrateRule;
    let expr = parse("integrate(2*x/(3-x^4), x)", &mut ctx).unwrap();
    let rewrite = rule
        .apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        )
        .unwrap();

    assert_eq!(rewrite.required_conditions.len(), 1);
    assert_eq!(rewrite.required_conditions[0].display(&ctx), "3 - x^4 > 0");
}

#[test]
fn test_integrate_secant_squared_kernel() {
    let mut ctx = Context::new();
    let rule = IntegrateRule;
    let expr = parse("integrate(1/cos(x)^2, x)", &mut ctx).unwrap();
    let rewrite = rule
        .apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        )
        .unwrap();
    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        ),
        "tan(x)"
    );
    assert_eq!(rewrite.required_conditions.len(), 1);
    assert_eq!(rewrite.required_conditions[0].display(&ctx), "cos(x) ≠ 0");
}

#[test]
fn test_integrate_cosecant_squared_kernel() {
    let mut ctx = Context::new();
    let rule = IntegrateRule;
    let expr = parse("integrate(1/sin(x)^2, x)", &mut ctx).unwrap();
    let rewrite = rule
        .apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        )
        .unwrap();
    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        ),
        "-cot(x)"
    );
    assert_eq!(rewrite.required_conditions.len(), 1);
    assert_eq!(rewrite.required_conditions[0].display(&ctx), "sin(x) ≠ 0");
}
