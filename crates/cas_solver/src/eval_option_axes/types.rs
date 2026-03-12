/// Typed option axes accepted by eval entry points.
#[derive(Debug, Clone, Copy)]
pub(crate) struct EvalOptionAxes {
    pub context: cas_api_models::EvalContextMode,
    pub branch: cas_api_models::EvalBranchMode,
    pub complex: cas_api_models::EvalComplexMode,
    pub autoexpand: cas_api_models::EvalExpandPolicy,
    pub steps: cas_api_models::EvalStepsMode,
    pub domain: cas_api_models::EvalDomainMode,
    pub value_domain: cas_api_models::EvalValueDomain,
    pub inv_trig: cas_api_models::EvalInvTrigPolicy,
    pub complex_branch: cas_api_models::EvalBranchMode,
    pub assume_scope: cas_api_models::EvalAssumeScope,
}
