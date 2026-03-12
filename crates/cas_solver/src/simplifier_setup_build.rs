mod advanced;
mod core;

use crate::Simplifier;
use cas_solver_core::simplifier_config::SimplifierRuleConfig;
use std::cell::RefCell;

fn default_configured_rule_profile() -> std::sync::Arc<cas_engine::RuleProfile> {
    thread_local! {
        static DEFAULT_CONFIGURED_RULE_PROFILE: RefCell<Option<std::sync::Arc<cas_engine::RuleProfile>>> =
            const { RefCell::new(None) };
    }

    DEFAULT_CONFIGURED_RULE_PROFILE.with(|slot| {
        let mut slot = slot.borrow_mut();
        std::sync::Arc::clone(slot.get_or_insert_with(|| {
            let mut simplifier = Simplifier::from_profile(cas_engine::default_rule_profile());
            core::add_core_rules(&mut simplifier, &SimplifierRuleConfig::default());
            advanced::add_advanced_rules(&mut simplifier, &SimplifierRuleConfig::default());
            cas_engine::rule_profile_from_simplifier(
                &simplifier,
                &cas_engine::EvalOptions::default(),
            )
        }))
    })
}

/// Build a configured simplifier with the rule portfolio expected by CLI workflows.
///
/// This centralizes rule wiring outside frontends, so REPL/FFI/web can share
/// a consistent initialization path.
pub fn build_simplifier_with_rule_config(config: SimplifierRuleConfig) -> Simplifier {
    if config == SimplifierRuleConfig::default() {
        return Simplifier::from_profile(default_configured_rule_profile());
    }

    let mut simplifier = Simplifier::from_profile(cas_engine::default_rule_profile());
    core::add_core_rules(&mut simplifier, &config);
    advanced::add_advanced_rules(&mut simplifier, &config);
    simplifier
}
