use super::latex_plain_text::latex_to_plain_text;
use super::{EnrichedStep, SubStep};

/// Display verbosity mode for didactic simplification rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepDisplayMode {
    None,
    Succinct,
    Normal,
    Verbose,
}

/// Rendering hints for CLI sub-step blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CliSubstepsRenderPlan {
    /// Optional category header to display before sub-steps.
    pub header: Option<&'static str>,
    /// If true, this block should be shown only once (deduplicated across steps).
    pub dedupe_once: bool,
}

/// Mutable rendering state for CLI enriched sub-step blocks.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CliSubstepsRenderState {
    dedupe_shown: bool,
}

/// Rendering hints for timeline sub-step blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimelineSubstepsRenderPlan {
    /// Category header to display before sub-steps.
    pub header: &'static str,
    /// If true, this block should be shown only once globally.
    pub dedupe_once: bool,
}

/// Stable classification of didactic sub-steps used by CLI and timeline renderers.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SubStepClassification {
    pub has_fraction_sum: bool,
    pub has_factorization: bool,
    pub has_nested_fraction: bool,
    pub has_polynomial_identity: bool,
}

/// Classify a sub-step block by its didactic content.
pub fn classify_sub_steps(sub_steps: &[SubStep]) -> SubStepClassification {
    SubStepClassification {
        has_fraction_sum: sub_steps.iter().any(|s| {
            s.description.contains("common denominator")
                || s.description.contains("Sum the fractions")
        }),
        has_factorization: sub_steps.iter().any(|s| {
            s.description.contains("Cancel common factor") || s.description.contains("Factor")
        }),
        has_nested_fraction: sub_steps.iter().any(|s| {
            s.description.contains("Combinar términos")
                || s.description.contains("Invertir la fracción")
                || s.description.contains("denominadores internos")
                || s.description.contains("Invertir")
                || s.description.contains("denominador")
        }),
        has_polynomial_identity: sub_steps.iter().any(|s| {
            s.description.contains("forma normal polinómica")
                || s.description.contains("Cancelar términos semejantes")
        }),
    }
}

/// Build a CLI rendering plan for enriched sub-steps.
///
/// This preserves the legacy precedence and dedupe behavior from REPL:
/// - Polynomial normalization header has highest priority.
/// - Fraction-sum header comes next (unless nested-fraction classification overrides it).
/// - Factorization header next.
/// - Nested-fraction header next.
/// - Fraction-sum-only blocks are deduplicated (shown once).
pub fn build_cli_substeps_render_plan(sub_steps: &[SubStep]) -> CliSubstepsRenderPlan {
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

/// Render enriched CLI sub-steps, applying the shared header/deduplication policy.
pub(crate) fn render_cli_enriched_substeps_lines(
    enriched_step: &EnrichedStep,
    state: &mut CliSubstepsRenderState,
) -> Vec<String> {
    if enriched_step.sub_steps.is_empty() {
        return Vec::new();
    }

    let render_plan = build_cli_substeps_render_plan(&enriched_step.sub_steps);
    let should_show = if render_plan.dedupe_once {
        !state.dedupe_shown
    } else {
        true
    };

    if !should_show {
        return Vec::new();
    }

    if render_plan.dedupe_once {
        state.dedupe_shown = true;
    }

    let mut lines = Vec::new();
    if let Some(header) = render_plan.header {
        lines.push(format!("   {}", header));
    }
    for sub in &enriched_step.sub_steps {
        lines.push(format!("      → {}", sub.description));
        if !sub.before_expr.is_empty() {
            lines.push(format!(
                "        {} → {}",
                latex_to_plain_text(&sub.before_expr),
                latex_to_plain_text(&sub.after_expr)
            ));
        }
    }

    lines
}

/// Build a timeline rendering plan for enriched sub-steps.
///
/// This preserves the current timeline behavior:
/// - nested-fraction and factorization blocks render per-step
/// - fraction-sum blocks render once globally
/// - default blocks render on every step
pub fn build_timeline_substeps_render_plan(sub_steps: &[SubStep]) -> TimelineSubstepsRenderPlan {
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
