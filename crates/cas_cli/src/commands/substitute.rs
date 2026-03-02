//! substitute subcommand handler.
//!
//! Keeps CLI I/O and delegates canonical JSON/text logic to `cas_solver`.

use crate::{OutputFormat, SubstituteArgs, SubstituteModeArg};

/// Run the substitute command.
pub fn run(args: SubstituteArgs) {
    let mode_str = match args.mode {
        SubstituteModeArg::Exact => "exact",
        SubstituteModeArg::Power => "power",
    };

    if matches!(args.format, OutputFormat::Json) {
        let out = cas_solver::substitute_str_to_json_with_options(
            &args.expr,
            &args.target,
            &args.replacement,
            mode_str,
            args.steps,
            true,
        );
        println!("{}", out);
        return;
    }

    let mode = match args.mode {
        SubstituteModeArg::Exact => cas_solver::json::SubstituteEvalMode::Exact,
        SubstituteModeArg::Power => cas_solver::json::SubstituteEvalMode::Power,
    };
    let output = match cas_solver::json::eval_substitute_from_str(
        &args.expr,
        &args.target,
        &args.replacement,
        mode,
        args.steps,
    ) {
        Ok(output) => output,
        Err(
            cas_solver::json::SubstituteEvalError::ParseExpression(message)
            | cas_solver::json::SubstituteEvalError::ParseTarget(message)
            | cas_solver::json::SubstituteEvalError::ParseReplacement(message),
        ) => {
            eprintln!("{}", message);
            std::process::exit(1);
        }
    };

    if args.steps && !output.steps.is_empty() {
        println!("Steps:");
        for step in &output.steps {
            if let Some(ref note) = step.note {
                println!(
                    "  {} → {} [{}] ({})",
                    step.before, step.after, step.rule, note
                );
            } else {
                println!("  {} → {} [{}]", step.before, step.after, step.rule);
            }
        }
    }
    println!("{}", output.result);
}
