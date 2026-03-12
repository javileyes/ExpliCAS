//! eval (text output) command handler.
//!
//! Handles parse/eval flow for text mode and optional session persistence.

use crate::{DomainArg, EvalArgs};

/// Run eval command in text mode.
pub fn run(args: &EvalArgs) {
    let domain = match args.domain {
        DomainArg::Strict => "strict",
        DomainArg::Generic => "generic",
        DomainArg::Assume => "assume",
    };

    let (result, load_warning, save_warning) =
        cas_session::eval_api::evaluate_eval_text_command_with_session(
            args.session.as_deref(),
            domain,
            &args.expr,
            args.session.is_some(),
        );
    if let Some(warning) = load_warning {
        eprintln!("{}", warning);
    }

    match result {
        Ok(result_str) => println!("{}", result_str),
        Err(message) => {
            eprintln!("{}", message);
            std::process::exit(1);
        }
    }

    if let Some(warning) = save_warning {
        eprintln!("{}", warning);
    }
}
