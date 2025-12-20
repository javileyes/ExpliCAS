//! eval-json subcommand handler.
//!
//! Evaluates a single expression and returns JSON output.

use std::time::Instant;

use anyhow::Result;
use clap::Args;

use cas_engine::{Engine, EvalAction, EvalRequest, EvalResult, SessionState};
use cas_parser::parse;

use crate::format::{expr_hash, expr_stats, format_expr_limited};
use crate::json_types::{ErrorJsonOutput, EvalJsonOutput, OptionsJson, TimingsJson, WarningJson};

/// Arguments for eval-json subcommand
#[derive(Args, Debug)]
pub struct EvalJsonArgs {
    /// Expression to evaluate
    pub expr: String,

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
}

/// Run the eval-json command
pub fn run(args: EvalJsonArgs) {
    // Set thread count if specified
    if let Some(n) = args.threads {
        std::env::set_var("RAYON_NUM_THREADS", n.to_string());
    }

    match run_inner(&args) {
        Ok(output) => {
            println!("{}", serde_json::to_string_pretty(&output).unwrap());
        }
        Err(e) => {
            let err_output = ErrorJsonOutput::with_input(e.to_string(), &args.expr);
            println!("{}", serde_json::to_string_pretty(&err_output).unwrap());
        }
    }
}

fn run_inner(args: &EvalJsonArgs) -> Result<EvalJsonOutput> {
    let total_start = Instant::now();

    // Create engine and session state
    let mut engine = Engine::new();
    let mut state = SessionState::new();

    // Configure options from args
    configure_options(&mut state.options, args);

    // Parse expression
    let parse_start = Instant::now();
    let parsed = parse(&args.expr, &mut engine.simplifier.context)
        .map_err(|e| anyhow::anyhow!("Parse error: {}", e))?;
    let parse_us = parse_start.elapsed().as_micros() as u64;

    // Build eval request
    let req = EvalRequest {
        raw_input: args.expr.clone(),
        parsed,
        kind: cas_engine::EntryKind::Expr(parsed),
        action: EvalAction::Simplify,
        auto_store: false,
    };

    // Evaluate
    let simplify_start = Instant::now();
    let output = engine.eval(&mut state, req)?;
    let simplify_us = simplify_start.elapsed().as_micros() as u64;

    // Extract result expression
    let result_expr = match &output.result {
        EvalResult::Expr(e) => *e,
        EvalResult::Set(v) if !v.is_empty() => v[0],
        EvalResult::Bool(b) => {
            // For bool results, format specially
            return Ok(EvalJsonOutput {
                ok: true,
                input: args.expr.clone(),
                result: b.to_string(),
                result_truncated: false,
                result_chars: b.to_string().len(),
                steps_mode: args.steps.clone(),
                steps_count: output.steps.len(),
                warnings: collect_warnings(&output),
                stats: Default::default(),
                hash: None,
                timings_us: TimingsJson {
                    parse_us,
                    simplify_us,
                    total_us: total_start.elapsed().as_micros() as u64,
                },
                options: build_options_json(args),
            });
        }
        _ => {
            return Err(anyhow::anyhow!("No result expression"));
        }
    };

    // Format result with truncation
    let (result_str, truncated, char_count) =
        format_expr_limited(&engine.simplifier.context, result_expr, args.max_chars);

    // Compute stats and hash
    let stats = expr_stats(&engine.simplifier.context, result_expr);
    let hash = if truncated {
        Some(expr_hash(&engine.simplifier.context, result_expr))
    } else {
        None
    };

    let total_us = total_start.elapsed().as_micros() as u64;

    Ok(EvalJsonOutput {
        ok: true,
        input: args.expr.clone(),
        result: result_str,
        result_truncated: truncated,
        result_chars: char_count,
        steps_mode: args.steps.clone(),
        steps_count: output.steps.len(),
        warnings: collect_warnings(&output),
        stats,
        hash,
        timings_us: TimingsJson {
            parse_us,
            simplify_us,
            total_us,
        },
        options: build_options_json(args),
    })
}

fn configure_options(opts: &mut cas_engine::options::EvalOptions, args: &EvalJsonArgs) {
    use cas_engine::options::{BranchMode, ComplexMode, ContextMode, StepsMode};
    use cas_engine::phase::ExpandPolicy;

    // Context mode
    opts.context_mode = match args.context.as_str() {
        "standard" => ContextMode::Standard,
        "solve" => ContextMode::Solve,
        "integrate" => ContextMode::IntegratePrep,
        _ => ContextMode::Auto,
    };

    // Branch mode
    opts.branch_mode = match args.branch.as_str() {
        "principal" => BranchMode::PrincipalBranch,
        _ => BranchMode::Strict,
    };

    // Complex mode
    opts.complex_mode = match args.complex.as_str() {
        "on" => ComplexMode::On,
        "off" => ComplexMode::Off,
        _ => ComplexMode::Auto,
    };

    // Steps mode
    opts.steps_mode = match args.steps.as_str() {
        "on" => StepsMode::On,
        "compact" => StepsMode::Compact,
        _ => StepsMode::Off,
    };

    // Expand policy
    opts.expand_policy = match args.autoexpand.as_str() {
        "auto" => ExpandPolicy::Auto,
        _ => ExpandPolicy::Off,
    };
}

fn collect_warnings(output: &cas_engine::EvalOutput) -> Vec<WarningJson> {
    output
        .domain_warnings
        .iter()
        .map(|w| WarningJson {
            rule: w.rule_name.clone(),
            assumption: w.message.clone(),
        })
        .collect()
}

fn build_options_json(args: &EvalJsonArgs) -> OptionsJson {
    OptionsJson {
        context_mode: args.context.clone(),
        branch_mode: args.branch.clone(),
        expand_policy: args.autoexpand.clone(),
        complex_mode: args.complex.clone(),
        steps_mode: args.steps.clone(),
    }
}
