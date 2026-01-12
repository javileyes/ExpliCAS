//! V2.9.9: Unified Eval Step Pipeline
//!
//! This module is the **single point of truth** for converting raw simplification
//! steps into display-ready steps. All cleanup, enrichment, and optimization
//! happens here — not scattered across engine.rs, repl.rs, or timeline.rs.
//!
//! The pipeline enforces the "impossible by construction" principle: raw steps
//! cannot escape to display layers because only `to_display_steps()` produces
//! the required `DisplayEvalSteps` type.

use crate::step::{DisplayEvalSteps, Step};

/// Convert raw steps to display-ready steps.
///
/// This is the **ONLY** function that should produce `DisplayEvalSteps`.
/// All step cleanup/enrichment MUST happen here to ensure consistency
/// across all renderers (Text, HTML, JSON).
///
/// # V2.9.9 Pipeline Stages
///
/// 1. **Remove no-ops**: Steps where `before == after` (no visible change)
/// 2. **Collapse duplicates**: Consecutive steps with identical descriptions (future)
/// 3. **Normalize descriptions**: Consistent formatting (future)
/// 4. **Enrich**: Add narrator text for didactic display (future)
///
/// # Arguments
///
/// * `raw_steps` - The raw steps from simplification, in order of application
///
/// # Returns
///
/// A `DisplayEvalSteps` wrapper guaranteeing cleanup has been applied.
#[must_use = "the result of pipeline processing should be used"]
pub fn to_display_steps(raw_steps: Vec<Step>) -> DisplayEvalSteps {
    // Stage 1: Remove no-op steps
    // V2.14.20: A step is a no-op only if:
    //   - before == after (root level), AND
    //   - before_local == after_local OR no local focus set
    // This preserves steps with local changes (e.g., abs(x) -> x in nested context)
    let mut cleaned: Vec<Step> = raw_steps
        .into_iter()
        .filter(|step| {
            let global_changed = step.before != step.after;
            let local_changed = match (step.before_local, step.after_local) {
                (Some(bl), Some(al)) => bl != al,
                _ => false, // No local focus means rely on global
            };
            global_changed || local_changed
        })
        .collect();

    // V2.14.20: Trace coherence fix
    // Due to recursive step generation, global_before may use stale root_expr.
    // Fix: For consecutive steps, step[i+1].global_before = step[i].global_after
    // This ensures the trace invariant: what we show as "After" in step N
    // is exactly what we show as "Before" in step N+1
    for i in 0..cleaned.len().saturating_sub(1) {
        // Get the global_after from step i (immutably first)
        if let Some(after_i) = cleaned[i].global_after {
            // Then mutate step i+1's global_before
            cleaned[i + 1].global_before = Some(after_i);
        }
    }

    // Stage 2: Future - collapse consecutive duplicates
    // Stage 3: Future - normalize descriptions
    // Stage 4: Future - enrich with narrator text

    DisplayEvalSteps(cleaned)
}

/// Convert raw steps to display-ready steps with semantic context.
///
/// Extended version that can use semantic configuration for smarter cleanup.
/// Currently delegates to `to_display_steps()`, but provides hook for future.
#[allow(dead_code)]
pub fn to_display_steps_with_context(
    raw_steps: Vec<Step>,
    _context: &cas_ast::Context,
) -> DisplayEvalSteps {
    // Future: use context for smart deduplication, witness survival filtering, etc.
    to_display_steps(raw_steps)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::Context;

    #[test]
    fn removes_no_op_steps() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");

        let steps = vec![
            Step::new("Real change", "SomeRule", x, y, vec![], Some(&ctx)),
            Step::new("No-op", "NoOpRule", x, x, vec![], Some(&ctx)), // Should be removed
            Step::new("Another change", "AnotherRule", y, x, vec![], Some(&ctx)),
        ];

        let display = to_display_steps(steps);

        assert_eq!(display.len(), 2, "No-op step should be filtered out");
        assert_eq!(display[0].description, "Real change");
        assert_eq!(display[1].description, "Another change");
    }

    #[test]
    fn empty_input_produces_empty_output() {
        let display = to_display_steps(vec![]);
        assert!(display.is_empty());
    }

    #[test]
    fn all_no_ops_produces_empty_output() {
        let mut ctx = Context::new();
        let x = ctx.var("x");

        let steps = vec![
            Step::new("No-op 1", "Rule1", x, x, vec![], Some(&ctx)),
            Step::new("No-op 2", "Rule2", x, x, vec![], Some(&ctx)),
        ];

        let display = to_display_steps(steps);
        assert!(display.is_empty(), "All no-ops should produce empty result");
    }
}
