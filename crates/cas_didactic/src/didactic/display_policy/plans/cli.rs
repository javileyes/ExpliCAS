use super::super::classification::SubStepClassification;
use super::CliSubstepsRenderPlan;

pub(super) fn build_cli_substeps_render_plan(
    classification: SubStepClassification,
    should_dedupe_once: fn(SubStepClassification) -> bool,
) -> CliSubstepsRenderPlan {
    let header = if classification.has_polynomial_identity {
        Some("[Normalización polinómica]")
    } else if classification.has_fraction_sum && !classification.has_nested_fraction {
        Some("[Suma de fracciones en exponentes]")
    } else if classification.has_factorization {
        Some("[Factorización de polinomios]")
    } else if classification.has_nested_fraction {
        Some("[Simplificación de fracción compleja]")
    } else {
        None
    };

    CliSubstepsRenderPlan {
        header,
        dedupe_once: should_dedupe_once(classification),
    }
}
