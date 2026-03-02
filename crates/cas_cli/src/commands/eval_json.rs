//! eval-json subcommand handler.
//!
//! Evaluates a single expression and returns JSON output.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use clap::Args;

use cas_session::SimplifyCacheKey;

use crate::session_io;
use cas_api_models::{ErrorJsonOutput, EvalJsonOutput, TimingsJson};

/// Arguments for eval-json subcommand
#[derive(Args, Debug)]
pub struct EvalJsonArgs {
    /// Expression to evaluate
    pub expr: String,

    /// Budget preset: "small", "standard", "unlimited"
    #[arg(long, default_value = "standard")]
    pub budget_preset: String,

    /// Strict mode: fail on budget exceeded (default: best-effort)
    #[arg(long, default_value_t = false)]
    pub strict: bool,

    /// Maximum characters in result output (truncates if larger)
    #[arg(long, default_value_t = 2000)]
    pub max_chars: usize,

    /// Steps mode: on, off, compact
    #[arg(long, default_value = "off")]
    pub steps: String,

    /// Context mode: auto, standard, solve, integrate
    #[arg(long, default_value = "auto")]
    pub context: String,

    /// Branch mode: strict, principal
    #[arg(long, default_value = "strict")]
    pub branch: String,

    /// Complex mode: auto, on, off
    #[arg(long, default_value = "auto")]
    pub complex: String,

    /// Expand policy: off, auto
    #[arg(long, default_value = "off")]
    pub autoexpand: String,

    /// Number of threads for parallel processing (sets RAYON_NUM_THREADS)
    #[arg(long)]
    pub threads: Option<usize>,

    /// Domain mode: strict, generic, assume
    #[arg(long, default_value = "generic")]
    pub domain: String,

    /// Value domain: real, complex
    #[arg(long, default_value = "real")]
    pub value_domain: String,

    /// Inverse trig policy: strict, principal
    #[arg(long, default_value = "strict")]
    pub inv_trig: String,

    /// Branch policy for multi-valued functions
    #[arg(long, default_value = "principal")]
    pub complex_branch: String,

    /// Constant folding mode: off, safe
    #[arg(long, default_value = "off")]
    pub const_fold: String,

    /// Assume scope: real, wildcard
    #[arg(long, default_value = "real")]
    pub assume_scope: String,

    /// Path to session file for persistent session across CLI invocations.
    #[arg(long)]
    pub session: Option<PathBuf>,
}

/// Run the eval-json command
pub fn run(args: EvalJsonArgs) {
    // Set thread count if specified
    if let Some(n) = args.threads {
        std::env::set_var("RAYON_NUM_THREADS", n.to_string());
    }

    match run_inner(&args) {
        Ok(output) => {
            println!("{}", output.to_json_pretty());
        }
        Err(e) => {
            // Classify error type based on message prefix
            let err_str = e.to_string();
            let err_output = if err_str.starts_with("Parse error:") {
                ErrorJsonOutput::parse_error(&err_str, Some(args.expr.clone()))
            } else {
                ErrorJsonOutput::with_input(&err_str, &args.expr)
            };
            println!("{}", err_output.to_json_pretty());
        }
    }
}

fn run_inner(args: &EvalJsonArgs) -> Result<EvalJsonOutput> {
    let total_start = Instant::now();

    // Build cache key for snapshot compatibility
    let domain_mode = match args.domain.as_str() {
        "strict" => cas_solver::DomainMode::Strict,
        "assume" => cas_solver::DomainMode::Assume,
        _ => cas_solver::DomainMode::Generic,
    };
    let cache_key = SimplifyCacheKey::from_context(domain_mode);

    // Load or create engine and session state
    let (mut engine, mut state, _) =
        session_io::load_or_new_session(args.session.as_deref(), &cache_key);

    // Configure options from args
    cas_solver::json::apply_eval_json_options(
        state.options_mut(),
        &args.context,
        &args.branch,
        &args.complex,
        &args.autoexpand,
        &args.steps,
        &args.domain,
        &args.value_domain,
        &args.inv_trig,
        &args.complex_branch,
        &args.assume_scope,
    );

    // Check for solve(equation, variable) or limit(expr, var, approach) syntax
    let parse_start = Instant::now();

    let req = cas_solver::json::build_eval_request_for_input(
        &args.expr,
        &mut engine.simplifier.context,
        args.session.is_some(),
    )
    .map_err(anyhow::Error::msg)?;
    let parsed_input = req.parsed;
    let parse_us = parse_start.elapsed().as_micros() as u64;

    // Evaluate
    let simplify_start = Instant::now();
    let output = engine.eval(&mut state, req)?;
    let simplify_us = simplify_start.elapsed().as_micros() as u64;

    let input_latex = Some(cas_solver::json::format_eval_input_latex(
        &engine.simplifier.context,
        parsed_input,
    ));

    // Save session snapshot if using persistent session
    if let Some(path) = args.session.as_deref() {
        let _ = session_io::save_session(&engine, &state, path, &cache_key);
    }

    let steps =
        cas_didactic::collect_eval_json_steps(&output, &engine.simplifier.context, &args.steps);
    let solve_steps = cas_solver::json::collect_solve_steps_eval_json(
        &output,
        &engine.simplifier.context,
        &args.steps,
    );
    let warnings = cas_solver::json::collect_warnings_eval_json(&output);
    let required_conditions = cas_solver::json::collect_required_conditions_eval_json(
        &output,
        &engine.simplifier.context,
    );
    let required_display =
        cas_solver::json::collect_required_display_eval_json(&output, &engine.simplifier.context);
    let timings_us = TimingsJson {
        parse_us,
        simplify_us,
        total_us: total_start.elapsed().as_micros() as u64,
    };

    cas_solver::json::finalize_eval_json_output(cas_solver::json::EvalJsonFinalizeInput {
        result: &output.result,
        ctx: &engine.simplifier.context,
        max_chars: args.max_chars,
        input: &args.expr,
        input_latex,
        steps_mode: &args.steps,
        steps,
        solve_steps,
        warnings,
        required_conditions,
        required_display,
        raw_steps_count: output.steps.len(),
        raw_solve_steps_count: output.solve_steps.len(),
        budget_preset: &args.budget_preset,
        strict: args.strict,
        domain: &args.domain,
        timings_us,
        context_mode: &args.context,
        branch_mode: &args.branch,
        expand_policy: &args.autoexpand,
        complex_mode: &args.complex,
        const_fold: &args.const_fold,
        value_domain: &args.value_domain,
        complex_branch: &args.complex_branch,
        inv_trig: &args.inv_trig,
        assume_scope: &args.assume_scope,
    })
    .map_err(anyhow::Error::msg)
}
