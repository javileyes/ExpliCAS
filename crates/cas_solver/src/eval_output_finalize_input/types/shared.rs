use cas_api_models::{RequiredConditionWire, SolveStepWire, StepWire, TimingsWire, WarningWire};

pub(crate) struct EvalOutputFinalizeShared<'a> {
    pub(crate) input: &'a str,
    pub(crate) input_latex: Option<String>,
    pub(crate) steps_mode: &'a str,
    pub(crate) steps: Vec<StepWire>,
    pub(crate) solve_steps: Vec<SolveStepWire>,
    pub(crate) warnings: Vec<WarningWire>,
    pub(crate) required_conditions: Vec<RequiredConditionWire>,
    pub(crate) required_display: Vec<String>,
    pub(crate) raw_steps_count: usize,
    pub(crate) raw_solve_steps_count: usize,
    pub(crate) budget_preset: &'a str,
    pub(crate) strict: bool,
    pub(crate) domain: &'a str,
    pub(crate) timings_us: TimingsWire,
    pub(crate) context_mode: &'a str,
    pub(crate) branch_mode: &'a str,
    pub(crate) expand_policy: &'a str,
    pub(crate) complex_mode: &'a str,
    pub(crate) const_fold: &'a str,
    pub(crate) value_domain: &'a str,
    pub(crate) complex_branch: &'a str,
    pub(crate) inv_trig: &'a str,
    pub(crate) assume_scope: &'a str,
}

impl EvalOutputFinalizeShared<'_> {
    pub(crate) fn primary_steps_count(&self) -> usize {
        self.raw_steps_count
    }

    pub(crate) fn combined_steps_count(&self) -> usize {
        self.raw_steps_count + self.raw_solve_steps_count
    }
}
