#[cfg(test)]
mod tests {
    use crate::config::sync_simplifier_with_cas_config;
    use cas_formatter::DisplayExpr;
    #[allow(unused_imports)]
    use cas_solver::session_api::{assumptions::*, runtime::*, simplifier::*};

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
        let config = crate::config::CasConfig {
            distribute: true,
            ..crate::config::CasConfig::default()
        };
        sync_simplifier_with_cas_config(&mut simplifier, &config);
        assert!(!simplifier
            .get_disabled_rules_clone()
            .contains("Distributive Property"));
    }
}
