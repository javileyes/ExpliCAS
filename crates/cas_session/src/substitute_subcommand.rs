//! Stateless CLI-subcommand helpers for `substitute`.

/// Substitution mode for subcommand-level evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubstituteCommandMode {
    Exact,
    Power,
}

/// CLI-friendly output contract for `substitute` subcommand.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubstituteSubcommandOutput {
    Json(String),
    TextLines(Vec<String>),
}

/// Evaluate substitute subcommand in canonical JSON mode.
///
/// Uses `cas_solver::substitute_str_to_json` as the canonical serializer.
pub fn evaluate_substitute_subcommand_json_canonical(
    expr: &str,
    target: &str,
    replacement: &str,
    mode: SubstituteCommandMode,
    steps_enabled: bool,
) -> String {
    let mode = match mode {
        SubstituteCommandMode::Exact => "exact",
        SubstituteCommandMode::Power => "power",
    };
    let opts = format!(
        "{{\"mode\":\"{}\",\"steps\":{},\"pretty\":true}}",
        mode, steps_enabled
    );
    cas_solver::substitute_str_to_json(expr, target, replacement, Some(&opts))
}

/// Evaluate substitute subcommand and map solver contracts to session-layer output.
pub fn evaluate_substitute_subcommand(
    expr: &str,
    target: &str,
    replacement: &str,
    mode: SubstituteCommandMode,
    steps_enabled: bool,
    json_output: bool,
) -> Result<SubstituteSubcommandOutput, String> {
    if json_output {
        let out = evaluate_substitute_subcommand_json_canonical(
            expr,
            target,
            replacement,
            mode,
            steps_enabled,
        );
        return Ok(SubstituteSubcommandOutput::Json(out));
    }

    let mode = match mode {
        SubstituteCommandMode::Exact => "exact",
        SubstituteCommandMode::Power => "power",
    };

    let lines = cas_solver::json::evaluate_substitute_subcommand_text_lines_with_mode(
        expr,
        target,
        replacement,
        mode,
        steps_enabled,
    )?;
    Ok(SubstituteSubcommandOutput::TextLines(lines))
}

#[cfg(test)]
mod tests {
    use super::{
        evaluate_substitute_subcommand, SubstituteCommandMode, SubstituteSubcommandOutput,
    };

    #[test]
    fn evaluate_substitute_subcommand_json_contract() {
        let out = evaluate_substitute_subcommand(
            "x^2+1",
            "x",
            "y",
            SubstituteCommandMode::Exact,
            false,
            true,
        )
        .expect("substitute json");

        match out {
            SubstituteSubcommandOutput::Json(payload) => {
                assert!(payload.contains("\"ok\""));
            }
            _ => panic!("expected json output"),
        }
    }
}
