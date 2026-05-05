use cas_api_models::{
    AssumptionDto, EquivalenceDiagnosticsWire, RequiredConditionWire, SolveStepWire, StepWire,
    TimingsWire, WarningWire,
};
use cas_solver_core::engine_events::EngineEvent;

use crate::eval_output_presentation::{
    collect_output_assumption_display_set, collect_output_assumptions_used,
    collect_output_required_conditions, collect_output_required_display,
    collect_output_solve_steps, collect_output_warnings, format_output_input_latex,
};

use super::PreparedEvalRun;

pub(super) struct CollectedEvalArtifacts {
    pub(super) input_latex: Option<String>,
    pub(super) steps: Vec<StepWire>,
    pub(super) solve_steps: Vec<SolveStepWire>,
    pub(super) warnings: Vec<WarningWire>,
    pub(super) required_conditions: Vec<RequiredConditionWire>,
    pub(super) required_display: Vec<String>,
    pub(super) assumptions_used: Vec<AssumptionDto>,
    pub(super) equivalence_diagnostics: Option<EquivalenceDiagnosticsWire>,
    pub(super) timings_us: TimingsWire,
}

pub(super) fn collect_eval_artifacts<F>(
    ctx: &mut cas_ast::Context,
    raw_input: &str,
    steps_mode: &str,
    domain_mode: &str,
    prepared: &PreparedEvalRun,
    total_us: u64,
    collect_steps: F,
) -> CollectedEvalArtifacts
where
    F: Fn(&[crate::Step], &[EngineEvent], &cas_ast::Context, &str) -> Vec<StepWire>,
{
    let input_latex = Some(format_output_input_latex(
        ctx,
        raw_input,
        prepared.parsed_input,
        prepared.derive_target,
        prepared.equiv_target,
        &prepared.style_signals,
    ));
    let steps_raw = prepared.output_view.steps.as_slice();
    let solve_steps_raw = prepared.output_view.solve_steps.as_slice();
    let solve_special_command = matches!(
        cas_api_models::parse_eval_special_command(raw_input),
        Some(cas_api_models::EvalSpecialCommand::Solve { .. })
    );
    let filtered_primary_steps = if solve_special_command && !solve_steps_raw.is_empty() {
        Some(
            steps_raw
                .iter()
                .filter(|step| should_keep_primary_solve_step(step))
                .cloned()
                .collect::<Vec<_>>(),
        )
    } else {
        None
    };
    let steps = if solve_special_command && !solve_steps_raw.is_empty() {
        Vec::new()
    } else if let Some(filtered) = filtered_primary_steps.as_ref() {
        if filtered.is_empty() {
            Vec::new()
        } else {
            collect_steps(
                filtered.as_slice(),
                prepared.events.as_slice(),
                ctx,
                steps_mode,
            )
        }
    } else {
        collect_steps(steps_raw, prepared.events.as_slice(), ctx, steps_mode)
    };
    let solve_steps = collect_output_solve_steps(
        solve_steps_raw,
        filtered_primary_steps.as_deref().unwrap_or(&[]),
        ctx,
        steps_mode,
    );
    let warnings = collect_output_warnings(&prepared.output_view.domain_warnings);
    let assumptions_used = if domain_mode == "assume" {
        collect_output_assumptions_used(steps_raw)
    } else {
        Vec::new()
    };
    let assumed_display = collect_output_assumption_display_set(&assumptions_used);
    let required_conditions_raw = prepared.output_view.required_conditions.as_slice();
    let required_conditions =
        collect_output_required_conditions(required_conditions_raw, ctx, &assumed_display);
    let required_display =
        collect_output_required_display(required_conditions_raw, ctx, &assumed_display);
    let timings_us = TimingsWire {
        parse_us: prepared.parse_us,
        simplify_us: prepared.simplify_us,
        total_us,
    };
    CollectedEvalArtifacts {
        input_latex,
        steps,
        solve_steps,
        warnings,
        required_conditions,
        required_display,
        assumptions_used,
        equivalence_diagnostics: prepared.equivalence_diagnostics.clone(),
        timings_us,
    }
}

fn should_keep_primary_solve_step(step: &crate::Step) -> bool {
    let importance = step.get_importance();
    if importance < crate::ImportanceLevel::Medium {
        return false;
    }

    if matches!(
        step.category,
        crate::StepCategory::Canonicalize
            | crate::StepCategory::ConstEval
            | crate::StepCategory::ConstFold
    ) {
        return false;
    }

    let rule_name = step.rule_name.as_str();
    if rule_name.starts_with("Canonicalize")
        || rule_name.starts_with("Evaluate Numeric")
        || rule_name == "Combine Constants"
    {
        return false;
    }

    let has_didactic_substeps = step
        .substeps()
        .iter()
        .any(|substep| !substep.title.trim().is_empty() || !substep.lines.is_empty());

    has_didactic_substeps || importance >= crate::ImportanceLevel::High
}

#[cfg(test)]
mod tests {
    use super::should_keep_primary_solve_step;
    use cas_solver_core::step_types::SubStep;

    fn demo_step() -> crate::Step {
        let mut ctx = cas_ast::Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let x_plus_one = ctx.add(cas_ast::Expr::Add(x, one));
        crate::Step::new("demo", "DemoRule", x, x_plus_one, Vec::new(), Some(&ctx))
    }

    #[test]
    fn rejects_medium_template_like_solve_steps_without_visible_intermediate() {
        let mut step = demo_step();
        step.importance = crate::ImportanceLevel::Medium;
        step.category = crate::StepCategory::Simplify;

        assert!(
            !should_keep_primary_solve_step(&step),
            "medium solve-adjacent steps without didactic substeps should stay hidden"
        );
    }

    #[test]
    fn rejects_canonicalization_even_when_marked_medium() {
        let mut step = demo_step();
        step.rule_name = "Canonicalize Division".into();
        step.importance = crate::ImportanceLevel::Medium;
        step.category = crate::StepCategory::Canonicalize;

        assert!(
            !should_keep_primary_solve_step(&step),
            "canonicalization noise must stay hidden in solve traces"
        );
    }

    #[test]
    fn keeps_medium_step_when_didactic_substeps_make_the_intermediate_visible() {
        let mut step = demo_step();
        step.importance = crate::ImportanceLevel::Medium;
        step.category = crate::StepCategory::Simplify;
        step.meta_mut().substeps.push(SubStep::with_importance(
            "Mostrar intermedio",
            vec!["Reescribir el lado izquierdo como producto".to_string()],
            crate::ImportanceLevel::Medium,
        ));

        assert!(
            should_keep_primary_solve_step(&step),
            "medium steps with real didactic substeps should stay visible in solve traces"
        );
    }

    #[test]
    fn keeps_high_importance_structural_steps_even_without_substeps() {
        let mut step = demo_step();
        step.importance = crate::ImportanceLevel::High;
        step.category = crate::StepCategory::Factor;

        assert!(
            should_keep_primary_solve_step(&step),
            "high-importance structural solve prep should remain visible"
        );
    }
}
