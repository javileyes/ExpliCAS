//! Runtime adapters shared by polynomial rewrite entry points.
//!
//! These helpers keep rule modules thin and avoid cross-module coupling when
//! `cas_math` APIs require runtime callbacks.

use cas_ast::{Context, ExprId};

/// Evaluate one `expand(...)` call in an isolated, steps-off simplifier context.
pub(crate) fn eval_expand_off(core_ctx: &mut Context, expand_call: ExprId) -> ExprId {
    use crate::options::StepsMode;
    use crate::phase::ExpandPolicy;

    let opts = crate::options::EvalOptions {
        steps_mode: StepsMode::Off,
        shared: crate::phase::SharedSemanticConfig {
            expand_policy: ExpandPolicy::Off,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut simplifier = crate::engine::Simplifier::with_profile(&opts);
    simplifier.set_steps_mode(StepsMode::Off);

    std::mem::swap(&mut simplifier.context, core_ctx);
    let (result, _) = simplifier.expand(expand_call);
    std::mem::swap(&mut simplifier.context, core_ctx);
    result
}
