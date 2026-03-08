use cas_ast::Context;

pub use cas_solver_core::solve_assumption_types::SolveAssumptionSectionConfig;

/// Render optional solve assumption/blocked sections according to CLI flags.
pub fn format_solve_assumption_and_blocked_sections(
    ctx: &Context,
    assumption_records: &[crate::AssumptionRecord],
    blocked_hints: &[crate::BlockedHint],
    config: SolveAssumptionSectionConfig,
) -> Vec<String> {
    let has_assumptions = !assumption_records.is_empty();
    let has_blocked = !blocked_hints.is_empty();

    if config.debug_mode && (has_assumptions || has_blocked) {
        let mut lines = vec![String::new()];
        if has_assumptions {
            lines.extend(crate::format_assumption_records_section_lines(
                assumption_records,
                "ℹ️ Assumptions used:",
                "  - ",
            ));
        }
        if has_blocked {
            lines.extend(crate::format_blocked_simplifications_section_lines(
                ctx,
                blocked_hints,
                config.domain_mode,
            ));
        }
        return lines;
    }

    if has_blocked && config.hints_enabled {
        let mut lines = vec![String::new()];
        lines.extend(crate::format_blocked_simplifications_section_lines(
            ctx,
            blocked_hints,
            config.domain_mode,
        ));
        return lines;
    }

    Vec::new()
}
