use super::classification::classify_sub_steps;

/// Rendering hints for CLI sub-step blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CliSubstepsRenderPlan {
    /// Optional category header to display before sub-steps.
    pub header: Option<&'static str>,
    /// If true, this block should be shown only once (deduplicated across steps).
    pub dedupe_once: bool,
}

/// Rendering hints for timeline sub-step blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimelineSubstepsRenderPlan {
    /// Category header to display before sub-steps.
    pub header: &'static str,
    /// If true, this block should be shown only once globally.
    pub dedupe_once: bool,
}

/// Build a CLI rendering plan for enriched sub-steps.
pub fn build_cli_substeps_render_plan(
    sub_steps: &[crate::didactic::SubStep],
) -> CliSubstepsRenderPlan {
    let classification = classify_sub_steps(sub_steps);
    let has_fraction_sum = classification.has_fraction_sum;
    let has_factorization = classification.has_factorization;
    let has_nested_fraction = classification.has_nested_fraction;
    let has_polynomial_identity = classification.has_polynomial_identity;

    let dedupe_once =
        has_fraction_sum && !has_nested_fraction && !has_factorization && !has_polynomial_identity;

    let header = if has_polynomial_identity {
        Some("[Normalización polinómica]")
    } else if has_fraction_sum && !has_nested_fraction {
        Some("[Suma de fracciones en exponentes]")
    } else if has_factorization {
        Some("[Factorización de polinomios]")
    } else if has_nested_fraction {
        Some("[Simplificación de fracción compleja]")
    } else {
        None
    };

    CliSubstepsRenderPlan {
        header,
        dedupe_once,
    }
}

/// Build a timeline rendering plan for enriched sub-steps.
pub fn build_timeline_substeps_render_plan(
    sub_steps: &[crate::didactic::SubStep],
) -> TimelineSubstepsRenderPlan {
    let classification = classify_sub_steps(sub_steps);

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
        dedupe_once: classification.has_fraction_sum
            && !classification.has_nested_fraction
            && !classification.has_factorization,
    }
}
