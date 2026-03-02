//! eval (text output) command handler.
//!
//! Handles parse/eval flow for text mode and optional session persistence.

use crate::{DomainArg, EvalArgs};

/// Run eval command in text mode.
pub fn run(args: &EvalArgs) {
    use cas_parser::parse;
    use cas_session::SimplifyCacheKey;
    use cas_solver::{EvalAction, EvalRequest};

    let domain_mode = match args.domain {
        DomainArg::Strict => cas_solver::DomainMode::Strict,
        DomainArg::Generic => cas_solver::DomainMode::Generic,
        DomainArg::Assume => cas_solver::DomainMode::Assume,
    };
    let cache_key = SimplifyCacheKey::from_context(domain_mode);

    let (mut engine, mut state, load_warning) =
        cas_session::load_or_new_session(args.session.as_deref(), &cache_key);
    if let Some(warning) = load_warning {
        eprintln!("{}", warning);
    }

    let parsed = match parse(&args.expr, &mut engine.simplifier.context) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Parse error: {}", e);
            std::process::exit(1);
        }
    };

    let req = EvalRequest {
        raw_input: args.expr.clone(),
        parsed,
        action: EvalAction::Simplify,
        auto_store: args.session.is_some(),
    };

    match engine.eval(&mut state, req) {
        Ok(output) => {
            let result_str = cas_solver::json::format_eval_result_text(
                &engine.simplifier.context,
                &output.result,
            );
            println!("{}", result_str);

            if let Some(path) = args.session.as_deref() {
                if let Err(e) = cas_session::save_session(&engine, &state, path, &cache_key) {
                    eprintln!("Warning: Failed to save session: {}", e);
                }
            }
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    }
}
