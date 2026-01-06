//! eval-json subcommand handler.
//!
//! Evaluates a single expression and returns JSON output.

use std::time::Instant;

use anyhow::Result;
use clap::Args;

use cas_engine::{Engine, EvalAction, EvalRequest, EvalResult, SessionState};
use cas_parser::parse;

use crate::format::{expr_hash, expr_stats, format_expr_limited};
use crate::json_types::{
    BudgetJson, DomainJson, ErrorJsonOutput, EvalJsonOutput, OptionsJson, RequiredConditionJson,
    SemanticsJson, TimingsJson, WarningJson,
};

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
            // Classify error type based on message prefix
            let err_str = e.to_string();
            let err_output = if err_str.starts_with("Parse error:") {
                ErrorJsonOutput::parse_error(&err_str, Some(args.expr.clone()))
            } else {
                ErrorJsonOutput::with_input(&err_str, &args.expr)
            };
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
                schema_version: 1,
                ok: true,
                input: args.expr.clone(),
                result: b.to_string(),
                result_truncated: false,
                result_chars: b.to_string().len(),
                steps_mode: args.steps.clone(),
                steps_count: output.steps.len(),
                warnings: collect_warnings(&output),
                required_conditions: collect_required_conditions(
                    &output,
                    &engine.simplifier.context,
                ),
                required_display: collect_required_display(&output, &engine.simplifier.context),
                budget: build_budget_json(args),
                domain: build_domain_json(args),
                stats: Default::default(),
                hash: None,
                timings_us: TimingsJson {
                    parse_us,
                    simplify_us,
                    total_us: total_start.elapsed().as_micros() as u64,
                },
                options: build_options_json(args),
                semantics: build_semantics_json(args),
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
        schema_version: 1,
        ok: true,
        input: args.expr.clone(),
        result: result_str,
        result_truncated: truncated,
        result_chars: char_count,
        steps_mode: args.steps.clone(),
        steps_count: output.steps.len(),
        warnings: collect_warnings(&output),
        required_conditions: collect_required_conditions(&output, &engine.simplifier.context),
        required_display: collect_required_display(&output, &engine.simplifier.context),
        budget: build_budget_json(args),
        domain: build_domain_json(args),
        stats,
        hash,
        timings_us: TimingsJson {
            parse_us,
            simplify_us,
            total_us,
        },
        options: build_options_json(args),
        semantics: build_semantics_json(args),
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

    // Domain mode
    opts.domain_mode = match args.domain.as_str() {
        "strict" => cas_engine::DomainMode::Strict,
        "assume" => cas_engine::DomainMode::Assume,
        _ => cas_engine::DomainMode::Generic,
    };

    // Inverse trig policy
    opts.inv_trig = match args.inv_trig.as_str() {
        "principal" => cas_engine::InverseTrigPolicy::PrincipalValue,
        _ => cas_engine::InverseTrigPolicy::Strict,
    };

    // Value domain
    opts.value_domain = match args.value_domain.as_str() {
        "complex" => cas_engine::ValueDomain::ComplexEnabled,
        _ => cas_engine::ValueDomain::RealOnly,
    };

    // Branch policy (only Principal for now)
    let _ = args.complex_branch.as_str(); // Parse but only one option
    opts.branch = cas_engine::BranchPolicy::Principal;

    // Assume scope
    opts.assume_scope = match args.assume_scope.as_str() {
        "wildcard" => cas_engine::AssumeScope::Wildcard,
        _ => cas_engine::AssumeScope::Real,
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

/// Collect required_conditions as structured JSON objects
fn collect_required_conditions(
    output: &cas_engine::EvalOutput,
    ctx: &cas_ast::Context,
) -> Vec<RequiredConditionJson> {
    use cas_ast::DisplayExpr;
    use cas_engine::implicit_domain::ImplicitCondition;

    output
        .required_conditions
        .iter()
        .map(|cond| {
            let (kind, expr_id) = match cond {
                ImplicitCondition::NonNegative(e) => ("NonNegative", *e),
                ImplicitCondition::Positive(e) => ("Positive", *e),
                ImplicitCondition::NonZero(e) => ("NonZero", *e),
            };
            // For now, canonical = display (no transforms currently applied)
            // When display transforms are added, expr_canonical should render without them
            let expr_str = DisplayExpr {
                context: ctx,
                id: expr_id,
            }
            .to_string();
            RequiredConditionJson {
                kind: kind.to_string(),
                expr_display: expr_str.clone(),
                expr_canonical: expr_str,
            }
        })
        .collect()
}

/// Collect required_conditions as human-readable strings for simple frontends
fn collect_required_display(
    output: &cas_engine::EvalOutput,
    ctx: &cas_ast::Context,
) -> Vec<String> {
    output
        .required_conditions
        .iter()
        .map(|cond| cond.display(ctx))
        .collect()
}

fn build_options_json(args: &EvalJsonArgs) -> OptionsJson {
    OptionsJson {
        context_mode: args.context.clone(),
        branch_mode: args.branch.clone(),
        expand_policy: args.autoexpand.clone(),
        complex_mode: args.complex.clone(),
        steps_mode: args.steps.clone(),
        domain_mode: args.domain.clone(),
        const_fold: args.const_fold.clone(),
    }
}

fn build_budget_json(args: &EvalJsonArgs) -> BudgetJson {
    BudgetJson {
        preset: args.budget_preset.clone(),
        mode: if args.strict {
            "strict".to_string()
        } else {
            "best-effort".to_string()
        },
        exceeded: None,
    }
}

fn build_domain_json(args: &EvalJsonArgs) -> DomainJson {
    DomainJson {
        mode: args.domain.clone(),
    }
}

fn build_semantics_json(args: &EvalJsonArgs) -> SemanticsJson {
    SemanticsJson {
        domain_mode: args.domain.clone(),
        value_domain: args.value_domain.clone(),
        branch: args.complex_branch.clone(),
        inv_trig: args.inv_trig.clone(),
        assume_scope: args.assume_scope.clone(),
    }
}
