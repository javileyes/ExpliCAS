use crate::eval_output_finalize::finalize_eval_output;
use crate::eval_output_finalize_input::EvalOutputFinalizeInput;

use super::collect::CollectedEvalArtifacts;
use super::{EvalCommandOutput, EvalCommandRunConfig, PreparedEvalRun};

pub(super) fn finalize_eval_collected(
    engine: &mut crate::Engine,
    config: EvalCommandRunConfig<'_>,
    mut prepared: PreparedEvalRun,
    collected: CollectedEvalArtifacts,
) -> Result<EvalCommandOutput, String> {
    // `--numeric-display decimal`: OUTPUT-BOUNDARY presentation only. The
    // whole pipeline above ran exact and symbolic; here the final result
    // (and solution-set members / interval bounds) maps its maximal closed
    // numeric subtrees to decimal display nodes via the shared walker.
    if config.numeric_display == cas_api_models::EvalNumericDisplay::Decimal {
        let complex_enabled = config.value_domain == cas_api_models::EvalValueDomain::Complex;
        let ctx = &mut engine.simplifier.context;
        let present = |ctx: &mut cas_ast::Context, id: cas_ast::ExprId| {
            cas_math::numeric_presentation::present_numeric(ctx, id, complex_enabled).unwrap_or(id)
        };
        match &mut prepared.output_view.result {
            crate::EvalResult::Expr(id) => {
                *id = present(ctx, *id);
            }
            crate::EvalResult::Set(ids) => {
                for id in ids.iter_mut() {
                    *id = present(ctx, *id);
                }
            }
            crate::EvalResult::SolutionSet(set) => match set {
                cas_ast::SolutionSet::Discrete(ids) => {
                    for id in ids.iter_mut() {
                        *id = present(ctx, *id);
                    }
                }
                cas_ast::SolutionSet::Continuous(interval) => {
                    interval.min = present(ctx, interval.min);
                    interval.max = present(ctx, interval.max);
                }
                cas_ast::SolutionSet::Union(intervals) => {
                    for interval in intervals.iter_mut() {
                        interval.min = present(ctx, interval.min);
                        interval.max = present(ctx, interval.max);
                    }
                }
                // Periodic/Conditional/Residual keep exact presentation in
                // v1 (their members feed downstream exact re-evaluation).
                _ => {}
            },
            _ => {}
        }
    }

    finalize_eval_output(EvalOutputFinalizeInput {
        result: &prepared.output_view.result,
        ctx: &engine.simplifier.context,
        max_chars: config.max_chars,
        input: config.expr,
        input_latex: collected.input_latex,
        style_signals: prepared.style_signals,
        stored_id: prepared.output_view.stored_id,
        strategy: prepared.output_view.strategy.clone(),
        steps_mode: config.steps_mode.as_str(),
        steps: collected.steps,
        solve_steps: collected.solve_steps,
        warnings: collected.warnings,
        required_conditions: collected.required_conditions,
        required_display: collected.required_display,
        assumptions_used: collected.assumptions_used,
        blocked_hints: collected.blocked_hints,
        equivalence_diagnostics: collected.equivalence_diagnostics,
        budget_preset: config.budget_preset.as_str(),
        strict: config.strict,
        domain: config.domain.as_str(),
        timings_us: collected.timings_us,
        context_mode: config.context_mode.as_str(),
        branch_mode: config.branch_mode.as_str(),
        expand_policy: config.expand_policy.as_str(),
        complex_mode: config.complex_mode.as_str(),
        const_fold: config.const_fold.as_str(),
        value_domain: config.value_domain.as_str(),
        complex_branch: config.complex_branch.as_str(),
        inv_trig: config.inv_trig.as_str(),
        assume_scope: config.assume_scope.as_str(),
    })
}
