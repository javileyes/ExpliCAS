use cas_solver::rules::arithmetic::{AddZeroRule, CombineConstantsRule, MulOneRule, MulZeroRule};
use cas_solver::rules::calculus::{DiffRule, IntegrateRule};
use cas_solver::rules::exponents::{
    EvaluatePowerRule, IdentityPowerRule, PowerPowerRule, PowerProductRule, PowerQuotientRule,
    ProductPowerRule,
};
use cas_solver::rules::grouping::CollectRule;
use cas_solver::rules::logarithms::{EvaluateLogRule, ExponentialLogRule, SplitLogExponentsRule};
use cas_solver::rules::number_theory::NumberTheoryRule;
use cas_solver::rules::polynomial::{
    AnnihilationRule, BinomialExpansionRule, CombineLikeTermsRule, DistributeRule,
};
use cas_solver::rules::trigonometry::{
    AngleConsistencyRule, AngleIdentityRule, DoubleAngleRule, EvaluateTrigRule,
    PythagoreanIdentityRule, TanToSinCosRule,
};

/// Feature switches for building a configured solver simplifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SimplifierRuleConfig {
    pub distribute: bool,
    pub expand_binomials: bool,
    pub factor_difference_squares: bool,
    pub root_denesting: bool,
    pub trig_double_angle: bool,
    pub trig_angle_sum: bool,
    pub log_split_exponents: bool,
    pub rationalize_denominator: bool,
    pub canonicalize_trig_square: bool,
    pub auto_factor: bool,
}

impl Default for SimplifierRuleConfig {
    fn default() -> Self {
        Self {
            distribute: false,
            expand_binomials: true,
            factor_difference_squares: false,
            root_denesting: true,
            trig_double_angle: true,
            trig_angle_sum: true,
            log_split_exponents: true,
            rationalize_denominator: true,
            canonicalize_trig_square: false,
            auto_factor: false,
        }
    }
}

/// Runtime feature switches used to enable/disable simplifier rules.
///
/// Unlike [`SimplifierRuleConfig`], this is applied to an already-built
/// simplifier and supports dynamic CLI toggles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SimplifierToggleConfig {
    pub distribute: bool,
    pub expand_binomials: bool,
    pub distribute_constants: bool,
    pub factor_difference_squares: bool,
    pub root_denesting: bool,
    pub trig_double_angle: bool,
    pub trig_angle_sum: bool,
    pub log_split_exponents: bool,
    pub rationalize_denominator: bool,
    pub canonicalize_trig_square: bool,
    pub auto_factor: bool,
}

impl Default for SimplifierToggleConfig {
    fn default() -> Self {
        Self {
            distribute: false,
            expand_binomials: true,
            distribute_constants: true,
            factor_difference_squares: false,
            root_denesting: true,
            trig_double_angle: true,
            trig_angle_sum: true,
            log_split_exponents: true,
            rationalize_denominator: true,
            canonicalize_trig_square: false,
            auto_factor: false,
        }
    }
}

impl From<SimplifierRuleConfig> for SimplifierToggleConfig {
    fn from(value: SimplifierRuleConfig) -> Self {
        Self {
            distribute: value.distribute,
            expand_binomials: value.expand_binomials,
            distribute_constants: true,
            factor_difference_squares: value.factor_difference_squares,
            root_denesting: value.root_denesting,
            trig_double_angle: value.trig_double_angle,
            trig_angle_sum: value.trig_angle_sum,
            log_split_exponents: value.log_split_exponents,
            rationalize_denominator: value.rationalize_denominator,
            canonicalize_trig_square: value.canonicalize_trig_square,
            auto_factor: value.auto_factor,
        }
    }
}

/// Update one named toggle rule.
pub fn set_simplifier_toggle_rule(
    config: &mut SimplifierToggleConfig,
    rule: &str,
    enabled: bool,
) -> Result<(), String> {
    match rule {
        "distribute" => config.distribute = enabled,
        "expand_binomials" => config.expand_binomials = enabled,
        "distribute_constants" => config.distribute_constants = enabled,
        "factor_difference_squares" => config.factor_difference_squares = enabled,
        "root_denesting" => config.root_denesting = enabled,
        "trig_double_angle" => config.trig_double_angle = enabled,
        "trig_angle_sum" => config.trig_angle_sum = enabled,
        "log_split_exponents" => config.log_split_exponents = enabled,
        "rationalize_denominator" => config.rationalize_denominator = enabled,
        "canonicalize_trig_square" => config.canonicalize_trig_square = enabled,
        "auto_factor" => config.auto_factor = enabled,
        _ => return Err(format!("Unknown rule: {}", rule)),
    }
    Ok(())
}

/// Build a configured simplifier with the rule portfolio expected by CLI workflows.
///
/// This centralizes rule wiring outside frontends, so REPL/FFI/web can share
/// a consistent initialization path.
pub fn build_simplifier_with_rule_config(config: SimplifierRuleConfig) -> cas_solver::Simplifier {
    let mut simplifier = cas_solver::Simplifier::with_default_rules();

    // Always enabled core rules.
    simplifier.add_rule(Box::new(cas_solver::rules::functions::AbsSquaredRule));
    simplifier.add_rule(Box::new(EvaluateTrigRule));
    simplifier.add_rule(Box::new(PythagoreanIdentityRule));
    if config.trig_angle_sum {
        simplifier.add_rule(Box::new(AngleIdentityRule));
    }
    simplifier.add_rule(Box::new(TanToSinCosRule));
    if config.trig_double_angle {
        simplifier.add_rule(Box::new(DoubleAngleRule));
    }
    if config.canonicalize_trig_square {
        simplifier.add_rule(Box::new(
            cas_solver::rules::trigonometry::CanonicalizeTrigSquareRule,
        ));
    }
    simplifier.add_rule(Box::new(EvaluateLogRule));
    simplifier.add_rule(Box::new(ExponentialLogRule));
    simplifier.add_rule(Box::new(cas_solver::rules::algebra::SimplifyFractionRule));
    simplifier.add_rule(Box::new(cas_solver::rules::algebra::ExpandRule));
    simplifier.add_rule(Box::new(cas_solver::rules::algebra::ConservativeExpandRule));
    simplifier.add_rule(Box::new(cas_solver::rules::algebra::FactorRule));
    simplifier.add_rule(Box::new(CollectRule));
    // Kept duplicated intentionally to preserve current behavior.
    simplifier.add_rule(Box::new(EvaluatePowerRule));
    simplifier.add_rule(Box::new(EvaluatePowerRule));
    if config.log_split_exponents {
        simplifier.add_rule(Box::new(SplitLogExponentsRule));
    }

    // Advanced algebra rules (critical for solver).
    simplifier.add_rule(Box::new(cas_solver::rules::algebra::NestedFractionRule));
    simplifier.add_rule(Box::new(cas_solver::rules::algebra::AddFractionsRule));
    simplifier.add_rule(Box::new(cas_solver::rules::algebra::SimplifyMulDivRule));
    if config.rationalize_denominator {
        simplifier.add_rule(Box::new(
            cas_solver::rules::algebra::RationalizeDenominatorRule,
        ));
    }
    simplifier.add_rule(Box::new(
        cas_solver::rules::algebra::CancelCommonFactorsRule,
    ));

    if config.distribute {
        simplifier.add_rule(Box::new(DistributeRule));
    }
    if config.expand_binomials {
        simplifier.add_rule(Box::new(BinomialExpansionRule));
    }
    if config.factor_difference_squares {
        simplifier.add_rule(Box::new(
            cas_solver::rules::algebra::FactorDifferenceSquaresRule,
        ));
    }
    if config.root_denesting {
        simplifier.add_rule(Box::new(cas_solver::rules::algebra::RootDenestingRule));
    }
    if config.auto_factor {
        simplifier.add_rule(Box::new(cas_solver::rules::algebra::AutomaticFactorRule));
    }

    simplifier.add_rule(Box::new(AngleConsistencyRule));
    // Kept duplicated intentionally to preserve current behavior.
    simplifier.add_rule(Box::new(CombineLikeTermsRule));
    simplifier.add_rule(Box::new(CombineLikeTermsRule));
    simplifier.add_rule(Box::new(AnnihilationRule));
    simplifier.add_rule(Box::new(ProductPowerRule));
    simplifier.add_rule(Box::new(PowerPowerRule));
    simplifier.add_rule(Box::new(PowerProductRule));
    simplifier.add_rule(Box::new(PowerQuotientRule));
    simplifier.add_rule(Box::new(IdentityPowerRule));
    simplifier.add_rule(Box::new(
        cas_solver::rules::exponents::NegativeBasePowerRule,
    ));
    simplifier.add_rule(Box::new(AddZeroRule));
    simplifier.add_rule(Box::new(MulOneRule));
    simplifier.add_rule(Box::new(MulZeroRule));
    simplifier.add_rule(Box::new(cas_solver::rules::arithmetic::DivZeroRule));
    simplifier.add_rule(Box::new(CombineConstantsRule));
    simplifier.add_rule(Box::new(IntegrateRule));
    simplifier.add_rule(Box::new(DiffRule));
    simplifier.add_rule(Box::new(NumberTheoryRule));

    simplifier
}

/// Apply runtime rule toggles to an existing simplifier instance.
pub fn apply_simplifier_toggle_config(
    simplifier: &mut cas_solver::Simplifier,
    config: SimplifierToggleConfig,
) {
    let mut toggle = |name: &str, enabled: bool| {
        if enabled {
            simplifier.enable_rule(name);
        } else {
            simplifier.disable_rule(name);
        }
    };

    toggle("Distributive Property", config.distribute);
    toggle("Binomial Expansion", config.expand_binomials);
    toggle("Distribute Constant", config.distribute_constants);
    toggle(
        "Factor Difference of Squares",
        config.factor_difference_squares,
    );
    toggle("Root Denesting", config.root_denesting);
    toggle("Double Angle Identity", config.trig_double_angle);
    toggle("Angle Sum/Diff Identity", config.trig_angle_sum);
    toggle("Split Log Exponents", config.log_split_exponents);
    toggle("Rationalize Denominator", config.rationalize_denominator);
    toggle("Canonicalize Trig Square", config.canonicalize_trig_square);

    // If auto_factor is on, keep conservative expansion and disable aggressive expansion
    // to avoid rewrite ping-pong.
    if config.auto_factor {
        simplifier.enable_rule("Automatic Factorization");
        simplifier.enable_rule("Conservative Expand");
        simplifier.disable_rule("Expand Polynomial");
        simplifier.disable_rule("Binomial Expansion");
    } else {
        simplifier.disable_rule("Automatic Factorization");
        simplifier.disable_rule("Conservative Expand");
        simplifier.enable_rule("Expand Polynomial");
        if config.expand_binomials {
            simplifier.enable_rule("Binomial Expansion");
        }
    }
}

/// Sync an existing simplifier with toggle values from `CasConfig`.
pub fn sync_simplifier_with_cas_config(
    simplifier: &mut cas_solver::Simplifier,
    config: &crate::CasConfig,
) {
    let toggles = crate::solver_toggle_config_from_cas_config(config);
    apply_simplifier_toggle_config(simplifier, toggles);
}

#[cfg(test)]
mod tests {
    use super::{
        apply_simplifier_toggle_config, build_simplifier_with_rule_config,
        set_simplifier_toggle_rule, sync_simplifier_with_cas_config, SimplifierRuleConfig,
        SimplifierToggleConfig,
    };
    use cas_formatter::DisplayExpr;

    #[test]
    fn build_simplifier_with_rule_config_can_simplify_basic_expression() {
        let mut simplifier = build_simplifier_with_rule_config(SimplifierRuleConfig::default());
        let expr = cas_parser::parse("x + x", &mut simplifier.context).expect("parse");
        let (result, _steps) = simplifier.simplify(expr);
        let rendered = format!(
            "{}",
            DisplayExpr {
                context: &simplifier.context,
                id: result
            }
        );
        assert_eq!(rendered, "2 * x");
    }

    #[test]
    fn apply_simplifier_toggle_config_auto_factor_disables_expand_rules() {
        let mut simplifier = build_simplifier_with_rule_config(SimplifierRuleConfig::default());
        let toggles = SimplifierToggleConfig {
            auto_factor: true,
            expand_binomials: true,
            ..SimplifierToggleConfig::default()
        };
        apply_simplifier_toggle_config(&mut simplifier, toggles);

        let disabled = simplifier.get_disabled_rules_clone();
        assert!(disabled.contains("Expand Polynomial"));
        assert!(disabled.contains("Binomial Expansion"));
        assert!(!disabled.contains("Automatic Factorization"));
        assert!(!disabled.contains("Conservative Expand"));
    }

    #[test]
    fn apply_simplifier_toggle_config_respects_distribute_constant_flag() {
        let mut simplifier = build_simplifier_with_rule_config(SimplifierRuleConfig::default());
        apply_simplifier_toggle_config(
            &mut simplifier,
            SimplifierToggleConfig {
                distribute_constants: false,
                ..SimplifierToggleConfig::default()
            },
        );
        assert!(simplifier
            .get_disabled_rules_clone()
            .contains("Distribute Constant"));
    }

    #[test]
    fn set_simplifier_toggle_rule_updates_known_rule() {
        let mut cfg = SimplifierToggleConfig::default();
        set_simplifier_toggle_rule(&mut cfg, "auto_factor", true).expect("known rule");
        assert!(cfg.auto_factor);
    }

    #[test]
    fn set_simplifier_toggle_rule_rejects_unknown_rule() {
        let mut cfg = SimplifierToggleConfig::default();
        let err = set_simplifier_toggle_rule(&mut cfg, "missing_rule", true).expect_err("error");
        assert!(err.contains("Unknown rule: missing_rule"));
    }

    #[test]
    fn sync_simplifier_with_cas_config_applies_toggles() {
        let mut simplifier = build_simplifier_with_rule_config(SimplifierRuleConfig::default());
        let config = crate::CasConfig {
            distribute: true,
            ..crate::CasConfig::default()
        };
        sync_simplifier_with_cas_config(&mut simplifier, &config);
        assert!(!simplifier
            .get_disabled_rules_clone()
            .contains("Distributive Property"));
    }
}
