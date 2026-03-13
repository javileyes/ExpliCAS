use super::{EvalOutputFinalizeContext, EvalOutputFinalizeInput, EvalOutputFinalizeShared};

impl<'a> EvalOutputFinalizeInput<'a> {
    pub(crate) fn split(self) -> (EvalOutputFinalizeContext<'a>, EvalOutputFinalizeShared<'a>) {
        (
            EvalOutputFinalizeContext {
                result: self.result,
                ctx: self.ctx,
                max_chars: self.max_chars,
            },
            EvalOutputFinalizeShared {
                input: self.input,
                input_latex: self.input_latex,
                steps_mode: self.steps_mode,
                steps: self.steps,
                solve_steps: self.solve_steps,
                warnings: self.warnings,
                required_conditions: self.required_conditions,
                required_display: self.required_display,
                raw_steps_count: self.raw_steps_count,
                raw_solve_steps_count: self.raw_solve_steps_count,
                budget_preset: self.budget_preset,
                strict: self.strict,
                domain: self.domain,
                timings_us: self.timings_us,
                context_mode: self.context_mode,
                branch_mode: self.branch_mode,
                expand_policy: self.expand_policy,
                complex_mode: self.complex_mode,
                const_fold: self.const_fold,
                value_domain: self.value_domain,
                complex_branch: self.complex_branch,
                inv_trig: self.inv_trig,
                assume_scope: self.assume_scope,
            },
        )
    }
}
