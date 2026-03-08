mod advanced;
mod core;

use crate::{Simplifier, SimplifierRuleConfig};

/// Build a configured simplifier with the rule portfolio expected by CLI workflows.
///
/// This centralizes rule wiring outside frontends, so REPL/FFI/web can share
/// a consistent initialization path.
pub fn build_simplifier_with_rule_config(config: SimplifierRuleConfig) -> Simplifier {
    let mut simplifier = Simplifier::with_default_rules();
    core::add_core_rules(&mut simplifier, &config);
    advanced::add_advanced_rules(&mut simplifier, &config);
    simplifier
}
