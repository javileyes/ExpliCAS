//! eval subcommand dispatcher.
//!
//! Handles stdin/arg expression sourcing and routes to JSON/text handlers.

use crate::{EvalArgs, OutputFormat};

/// Run the eval command (JSON or text output).
pub fn run(args: EvalArgs) {
    let expr = read_expr_or_stdin(&args.expr);
    if expr.is_empty() {
        eprintln!("Error: No expression provided");
        std::process::exit(1);
    }

    if let Some(n) = args.threads {
        std::env::set_var("RAYON_NUM_THREADS", n.to_string());
    }

    tracing::debug!(
        target: "budget",
        preset = ?args.budget,
        "Using budget preset"
    );

    match args.format {
        OutputFormat::Json => {
            let json_args = crate::commands::eval_json::from_eval_args(expr.clone(), &args);
            crate::commands::eval_json::run(json_args);
        }
        OutputFormat::Text => {
            let text_args = EvalArgs { expr, ..args };
            crate::commands::eval_text::run(&text_args);
        }
    }
}

fn read_expr_or_stdin(expr: &str) -> String {
    if expr == "-" {
        use std::io::{self, BufRead};
        let stdin = io::stdin();
        let mut lines = Vec::new();
        for line in stdin.lock().lines() {
            match line {
                Ok(l) => lines.push(l),
                Err(_) => break,
            }
        }
        lines.join("\n").trim().to_string()
    } else {
        expr.to_string()
    }
}
