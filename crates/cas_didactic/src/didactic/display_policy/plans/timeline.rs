use super::super::classification::SubStepClassification;
use super::TimelineSubstepsRenderPlan;

pub(super) fn build_timeline_substeps_render_plan(
    classification: SubStepClassification,
    should_dedupe_once: fn(SubStepClassification) -> bool,
) -> TimelineSubstepsRenderPlan {
    let header = if classification.has_nested_fraction {
        "Simplificación de fracción compleja"
    } else if classification.has_fraction_sum {
        "Suma de fracciones"
    } else if classification.has_factorization {
        "Factorización de polinomios"
    } else {
        "Pasos intermedios"
    };

    TimelineSubstepsRenderPlan {
        header,
        dedupe_once: should_dedupe_once(classification),
    }
}
