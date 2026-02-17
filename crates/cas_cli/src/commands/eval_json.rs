//! eval-json subcommand handler.
//!
//! Evaluates a single expression and returns JSON output.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use clap::Args;

use cas_session::{SessionSnapshot, SimplifyCacheKey};
use cas_solver::{EvalAction, EvalRequest, EvalResult};

// For step filtering (match timeline behavior)
use cas_didactic::{pathsteps_to_expr_path, ImportanceLevel};
// For didactic substeps (like timeline)
use cas_didactic as didactic;

use crate::format::{expr_hash, expr_stats, format_expr_limited};
use crate::json_types::{
    BudgetJsonInfo, DomainJson, ErrorJsonOutput, EvalJsonOutput, OptionsJson,
    RequiredConditionJson, SemanticsJson, SolveStepJson, SolveSubStepJson, StepJson, SubStepJson,
    TimingsJson, WarningJson,
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
            print_pretty_json(&output);
        }
        Err(e) => {
            // Classify error type based on message prefix
            let err_str = e.to_string();
            let err_output = if err_str.starts_with("Parse error:") {
                ErrorJsonOutput::parse_error(&err_str, Some(args.expr.clone()))
            } else {
                ErrorJsonOutput::with_input(&err_str, &args.expr)
            };
            print_pretty_json(&err_output);
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
    let (mut engine, mut state) = load_or_new_session(&args.session, &cache_key);

    // Configure options from args
    configure_options(&mut state.options, args);

    // Check for solve(equation, variable) or limit(expr, var, approach) syntax
    let parse_start = Instant::now();

    let req = if let Some((eq_str, var)) = parse_solve_command(&args.expr) {
        // Explicit solve command: solve(equation, variable)
        let stmt = cas_parser::parse_statement(&eq_str, &mut engine.simplifier.context)
            .map_err(|e| anyhow::anyhow!("Parse error in solve equation: {}", e))?;

        match stmt {
            cas_parser::Statement::Equation(eq) => {
                let eq_expr = engine
                    .simplifier
                    .context
                    .call("Equal", vec![eq.lhs, eq.rhs]);

                EvalRequest {
                    raw_input: args.expr.clone(),
                    parsed: eq_expr,
                    action: EvalAction::Solve { var },
                    auto_store: args.session.is_some(),
                }
            }
            cas_parser::Statement::Expression(expr) => {
                // Expression treated as equation = 0: solve(expr, var) means expr = 0
                let zero = engine.simplifier.context.num(0);
                let eq_expr = engine.simplifier.context.call("Equal", vec![expr, zero]);

                EvalRequest {
                    raw_input: args.expr.clone(),
                    parsed: eq_expr,
                    action: EvalAction::Solve { var },
                    auto_store: args.session.is_some(),
                }
            }
        }
    } else if let Some((expr_str, var, approach)) = parse_limit_command(&args.expr) {
        // Explicit limit command: limit(expression, variable, approach)
        let parsed = cas_parser::parse(&expr_str, &mut engine.simplifier.context)
            .map_err(|e| anyhow::anyhow!("Parse error in limit expression: {}", e))?;

        EvalRequest {
            raw_input: args.expr.clone(),
            parsed,
            action: EvalAction::Limit { var, approach },
            auto_store: args.session.is_some(),
        }
    } else {
        // Standard parsing: detect equations vs expressions
        let stmt = cas_parser::parse_statement(&args.expr, &mut engine.simplifier.context)
            .map_err(|e| anyhow::anyhow!("Parse error: {}", e))?;

        // Build eval request based on statement type
        match stmt {
            cas_parser::Statement::Equation(eq) => {
                // For equations, use Solve action with default variable detection
                let eq_expr = engine
                    .simplifier
                    .context
                    .call("Equal", vec![eq.lhs, eq.rhs]);

                // Detect solve variable from equation (prefer 'x' if present, else first variable found)
                let var = detect_solve_variable(&engine.simplifier.context, eq.lhs, eq.rhs);

                EvalRequest {
                    raw_input: args.expr.clone(),
                    parsed: eq_expr,
                    action: EvalAction::Solve { var },
                    auto_store: args.session.is_some(),
                }
            }
            cas_parser::Statement::Expression(parsed) => {
                // For expressions, use Simplify action
                EvalRequest {
                    raw_input: args.expr.clone(),
                    parsed,
                    action: EvalAction::Simplify,
                    auto_store: args.session.is_some(),
                }
            }
        }
    };
    let parse_us = parse_start.elapsed().as_micros() as u64;

    // Capture parsed info before eval() consumes the request
    let input_kind = cas_ast::eq::unwrap_eq(&engine.simplifier.context, req.parsed)
        .map(|(lhs, rhs)| cas_session::EntryKind::Eq { lhs, rhs })
        .unwrap_or(cas_session::EntryKind::Expr(req.parsed));

    // Evaluate
    let simplify_start = Instant::now();
    let output = engine.eval(&mut state, req)?;
    let simplify_us = simplify_start.elapsed().as_micros() as u64;

    // Generate LaTeX for input expression
    // For equations, render as "lhs = rhs" instead of "Equal(lhs, rhs)"
    let input_latex = Some(match &input_kind {
        cas_session::EntryKind::Eq { lhs, rhs } => {
            let lhs_latex = cas_formatter::LaTeXExpr {
                context: &engine.simplifier.context,
                id: *lhs,
            }
            .to_latex();
            let rhs_latex = cas_formatter::LaTeXExpr {
                context: &engine.simplifier.context,
                id: *rhs,
            }
            .to_latex();
            format!("{} = {}", lhs_latex, rhs_latex)
        }
        cas_session::EntryKind::Expr(id) => cas_formatter::LaTeXExpr {
            context: &engine.simplifier.context,
            id: *id,
        }
        .to_latex(),
    });

    // Save session snapshot if using persistent session
    if let Some(ref path) = args.session {
        save_session(&engine, &state, path, &cache_key);
    }

    // Extract result expression
    let result_expr = match &output.result {
        EvalResult::Expr(e) => *e,
        EvalResult::Set(v) if !v.is_empty() => v[0],
        EvalResult::SolutionSet(solution_set) => {
            // For solution sets, format the solution and return
            let result_str = format_solution_set(&engine.simplifier.context, solution_set);
            let result_latex = solution_set_to_latex(&engine.simplifier.context, solution_set);

            return Ok(EvalJsonOutput {
                schema_version: 1,
                ok: true,
                input: args.expr.clone(),
                input_latex: input_latex.clone(),
                result: result_str.clone(),
                result_truncated: false,
                result_chars: result_str.len(),
                result_latex: Some(result_latex.clone()),
                steps_mode: args.steps.clone(),
                steps_count: output.steps.len() + output.solve_steps.len(),
                steps: collect_steps(&output, &engine.simplifier.context, &args.steps),
                solve_steps: collect_solve_steps(&output, &engine.simplifier.context, &args.steps),
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
                wire: serde_json::to_value(build_wire_reply(
                    &collect_warnings(&output),
                    &collect_required_display(&output, &engine.simplifier.context),
                    &result_str,
                    Some(&result_latex),
                    output.steps.len() + output.solve_steps.len(),
                    &args.steps,
                ))
                .ok(),
            });
        }
        EvalResult::Bool(b) => {
            // For bool results, format specially
            return Ok(EvalJsonOutput {
                schema_version: 1,
                ok: true,
                input: args.expr.clone(),
                input_latex: input_latex.clone(),
                result: b.to_string(),
                result_truncated: false,
                result_chars: b.to_string().len(),
                result_latex: None,
                steps_mode: args.steps.clone(),
                steps_count: output.steps.len(),
                steps: collect_steps(&output, &engine.simplifier.context, &args.steps),
                solve_steps: collect_solve_steps(&output, &engine.simplifier.context, &args.steps),
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
                wire: serde_json::to_value(build_wire_reply(
                    &collect_warnings(&output),
                    &collect_required_display(&output, &engine.simplifier.context),
                    &b.to_string(),
                    None,
                    output.steps.len(),
                    &args.steps,
                ))
                .ok(),
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

    // Generate LaTeX for result
    let result_latex = if !truncated {
        Some(
            cas_formatter::LaTeXExpr {
                context: &engine.simplifier.context,
                id: result_expr,
            }
            .to_latex(),
        )
    } else {
        None
    };

    // Build wire before moving values
    let warnings = collect_warnings(&output);
    let required_display = collect_required_display(&output, &engine.simplifier.context);
    let wire = build_wire_reply(
        &warnings,
        &required_display,
        &result_str,
        result_latex.as_deref(),
        output.steps.len(),
        &args.steps,
    );

    Ok(EvalJsonOutput {
        schema_version: 1,
        ok: true,
        input: args.expr.clone(),
        input_latex,
        result: result_str,
        result_truncated: truncated,
        result_chars: char_count,
        result_latex,
        steps_mode: args.steps.clone(),
        steps_count: output.steps.len(),
        steps: collect_steps(&output, &engine.simplifier.context, &args.steps),
        solve_steps: collect_solve_steps(&output, &engine.simplifier.context, &args.steps),
        warnings,
        required_conditions: collect_required_conditions(&output, &engine.simplifier.context),
        required_display,
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
        wire: serde_json::to_value(&wire).ok(),
    })
}

fn configure_options(opts: &mut cas_solver::EvalOptions, args: &EvalJsonArgs) {
    use cas_solver::{BranchMode, ComplexMode, ContextMode, ExpandPolicy, StepsMode};

    // Context mode
    opts.shared.context_mode = match args.context.as_str() {
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
    opts.shared.expand_policy = match args.autoexpand.as_str() {
        "auto" => ExpandPolicy::Auto,
        _ => ExpandPolicy::Off,
    };

    // Domain mode
    opts.shared.semantics.domain_mode = match args.domain.as_str() {
        "strict" => cas_solver::DomainMode::Strict,
        "assume" => cas_solver::DomainMode::Assume,
        _ => cas_solver::DomainMode::Generic,
    };

    // Inverse trig policy
    opts.shared.semantics.inv_trig = match args.inv_trig.as_str() {
        "principal" => cas_solver::InverseTrigPolicy::PrincipalValue,
        _ => cas_solver::InverseTrigPolicy::Strict,
    };

    // Value domain
    opts.shared.semantics.value_domain = match args.value_domain.as_str() {
        "complex" => cas_solver::ValueDomain::ComplexEnabled,
        _ => cas_solver::ValueDomain::RealOnly,
    };

    // Branch policy (only Principal for now)
    let _ = args.complex_branch.as_str(); // Parse but only one option
    opts.shared.semantics.branch = cas_solver::BranchPolicy::Principal;

    // Assume scope
    opts.shared.semantics.assume_scope = match args.assume_scope.as_str() {
        "wildcard" => cas_solver::AssumeScope::Wildcard,
        _ => cas_solver::AssumeScope::Real,
    };
}

fn collect_warnings(output: &cas_solver::EvalOutput) -> Vec<WarningJson> {
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
    output: &cas_solver::EvalOutput,
    ctx: &cas_ast::Context,
) -> Vec<RequiredConditionJson> {
    use cas_formatter::DisplayExpr;
    use cas_solver::ImplicitCondition;

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
    output: &cas_solver::EvalOutput,
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

fn build_budget_json(args: &EvalJsonArgs) -> BudgetJsonInfo {
    BudgetJsonInfo {
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

/// Build WireReply from evaluation output data.
///
/// Constructs a unified wire format with all messages in order:
/// 1. Warnings (if any)
/// 2. Required conditions (info)
/// 3. Result (output)
/// 4. Steps summary (if enabled)
fn build_wire_reply(
    warnings: &[WarningJson],
    required_display: &[String],
    result: &str,
    result_latex: Option<&str>,
    steps_count: usize,
    steps_mode: &str,
) -> crate::repl::wire::WireReply {
    use crate::repl::wire::{WireKind, WireMsg, WireReply, SCHEMA_VERSION};

    let mut messages = Vec::new();

    // 1. Warnings
    for w in warnings {
        messages.push(WireMsg::new(
            WireKind::Warn,
            format!("⚠ {} ({})", w.assumption, w.rule),
        ));
    }

    // 2. Required conditions
    if !required_display.is_empty() {
        messages.push(WireMsg::new(WireKind::Info, "ℹ️ Requires:".to_string()));
        for cond in required_display {
            messages.push(WireMsg::new(WireKind::Info, format!("  • {}", cond)));
        }
    }

    // 3. Result (main output)
    let result_text = if let Some(latex) = result_latex {
        format!("Result: {} [LaTeX: {}]", result, latex)
    } else {
        format!("Result: {}", result)
    };
    messages.push(WireMsg::new(WireKind::Output, result_text));

    // 4. Steps summary (if enabled and present)
    if steps_mode == "on" && steps_count > 0 {
        messages.push(WireMsg::new(
            WireKind::Steps,
            format!("{} simplification step(s)", steps_count),
        ));
    }

    WireReply {
        schema_version: SCHEMA_VERSION,
        messages,
    }
}

// =============================================================================
// Session Persistence Helpers
// =============================================================================

fn load_or_new_session(
    path: &Option<PathBuf>,
    key: &SimplifyCacheKey,
) -> (cas_solver::Engine, cas_session::SessionState) {
    let Some(path) = path else {
        return (cas_solver::Engine::new(), cas_session::SessionState::new());
    };

    if !path.exists() {
        return (cas_solver::Engine::new(), cas_session::SessionState::new());
    }

    match SessionSnapshot::load(path) {
        Ok(snap) => {
            if snap.is_compatible(key) {
                let (ctx, store) = snap.into_parts();
                let engine = cas_solver::Engine::with_context(ctx);
                let state = cas_session::SessionState::from_store(store);
                (engine, state)
            } else {
                (cas_solver::Engine::new(), cas_session::SessionState::new())
            }
        }
        Err(_) => (cas_solver::Engine::new(), cas_session::SessionState::new()),
    }
}

fn save_session(
    engine: &cas_solver::Engine,
    state: &cas_session::SessionState,
    path: &std::path::Path,
    key: &SimplifyCacheKey,
) {
    let snap = SessionSnapshot::new(&engine.simplifier.context, state.store(), key.clone());
    let _ = snap.save_atomic(path); // Ignore save errors in JSON mode
}

/// Convert engine steps to JSON format
fn collect_steps(
    output: &cas_solver::EvalOutput,
    ctx: &cas_ast::Context,
    steps_mode: &str,
) -> Vec<StepJson> {
    // Only include steps if steps_mode is "on"
    if steps_mode != "on" {
        return vec![];
    }

    // V2.15.36: Filter steps by ImportanceLevel to match timeline behavior
    // Timeline only shows steps with importance >= Medium (VerbosityLevel::Normal)
    let filtered: Vec<_> = output
        .steps
        .iter()
        .filter(|step| step.get_importance() >= ImportanceLevel::Medium)
        .cloned()
        .collect();

    if filtered.is_empty() {
        return vec![];
    }

    // V2.15.36: Enrich steps with didactic substeps (like timeline)
    let first_step = match filtered.first() {
        Some(s) => s,
        None => return vec![],
    };
    let original_expr = first_step.global_before.unwrap_or(first_step.before);
    let enriched_steps = didactic::enrich_steps(ctx, original_expr, filtered.clone());

    // Build the JSON output using enriched steps
    enriched_steps
        .iter()
        .enumerate()
        .map(|(i, enriched)| {
            let step = &enriched.base_step;

            // Format before/after expressions - use global if available
            let before_expr = step.global_before.unwrap_or(step.before);
            let after_expr = step.global_after.unwrap_or(step.after);

            // Focus expressions (the parts that actually changed)
            let focus_before = step.before_local().unwrap_or(step.before);
            let focus_after = step.after_local().unwrap_or(step.after);

            // Plain text format
            let before_str = format!(
                "{}",
                cas_formatter::DisplayExpr {
                    context: ctx,
                    id: before_expr
                }
            );
            let after_str = format!(
                "{}",
                cas_formatter::DisplayExpr {
                    context: ctx,
                    id: after_expr
                }
            );

            // LaTeX with highlighting - use path-based highlighting like timeline
            // V2.15.36: Path-based highlighting ensures exact position in tree is highlighted
            let expr_path = pathsteps_to_expr_path(step.path());

            // Use PathHighlightConfig for path-based highlighting (matches timeline)
            let mut before_config = cas_formatter::PathHighlightConfig::new();
            before_config.add(expr_path.clone(), cas_ast::HighlightColor::Red);
            let before_latex = cas_formatter::PathHighlightedLatexRenderer {
                context: ctx,
                id: before_expr,
                path_highlights: &before_config,
                hints: None,
                style_prefs: None,
            }
            .to_latex();

            let mut after_config = cas_formatter::PathHighlightConfig::new();
            after_config.add(expr_path, cas_ast::HighlightColor::Green);
            let after_latex = cas_formatter::PathHighlightedLatexRenderer {
                context: ctx,
                id: after_expr,
                path_highlights: &after_config,
                hints: None,
                style_prefs: None,
            }
            .to_latex();

            // Generate rule_latex: colored "antecedent → consequent"
            let mut rule_before_config = cas_ast::HighlightConfig::new();
            rule_before_config.add(focus_before, cas_ast::HighlightColor::Red);
            let local_before_colored = cas_formatter::LaTeXExprHighlighted {
                context: ctx,
                id: focus_before,
                highlights: &rule_before_config,
            }
            .to_latex();

            let mut rule_after_config = cas_ast::HighlightConfig::new();
            rule_after_config.add(focus_after, cas_ast::HighlightColor::Green);
            let local_after_colored = cas_formatter::LaTeXExprHighlighted {
                context: ctx,
                id: focus_after,
                highlights: &rule_after_config,
            }
            .to_latex();

            let rule_latex = format!(
                "{} \\rightarrow {}",
                local_before_colored, local_after_colored
            );

            // V2.15.36: Combine engine substeps + didactic substeps
            let mut substeps: Vec<SubStepJson> = Vec::new();

            // Add engine substeps (from step.substeps)
            for ss in step.substeps() {
                substeps.push(SubStepJson {
                    title: ss.title.clone(),
                    lines: ss.lines.clone(),
                    before_latex: None,
                    after_latex: None,
                });
            }

            // Add didactic substeps (from enrich_steps)
            for ss in &enriched.sub_steps {
                // Use explicit LaTeX if available, otherwise wrap plain text
                // in \text{} to prevent MathJax from rendering words as math
                let before_latex = ss
                    .before_latex
                    .clone()
                    .unwrap_or_else(|| format!("\\text{{{}}}", ss.before_expr));
                let after_latex = ss
                    .after_latex
                    .clone()
                    .unwrap_or_else(|| format!("\\text{{{}}}", ss.after_expr));
                substeps.push(SubStepJson {
                    title: ss.description.clone(),
                    lines: vec![],
                    before_latex: Some(before_latex),
                    after_latex: Some(after_latex),
                });
            }

            StepJson {
                index: i + 1,
                rule: step.rule_name.clone(),
                rule_latex,
                before: before_str,
                after: after_str,
                before_latex,
                after_latex,
                substeps,
            }
        })
        .collect()
}

/// Convert solver steps to JSON format (for equation solving)
fn collect_solve_steps(
    output: &cas_solver::EvalOutput,
    ctx: &cas_ast::Context,
    steps_mode: &str,
) -> Vec<SolveStepJson> {
    // Only include solve steps if steps_mode is "on"
    if steps_mode != "on" {
        return vec![];
    }

    // Filter steps by ImportanceLevel to match timeline behavior
    let filtered: Vec<_> = output
        .solve_steps
        .iter()
        .filter(|step| step.importance >= ImportanceLevel::Medium)
        .collect();

    if filtered.is_empty() {
        return vec![];
    }

    filtered
        .iter()
        .enumerate()
        .map(|(i, step)| {
            // Format equation parts
            let lhs_str = format!(
                "{}",
                cas_formatter::DisplayExpr {
                    context: ctx,
                    id: step.equation_after.lhs
                }
            );
            let rhs_str = format!(
                "{}",
                cas_formatter::DisplayExpr {
                    context: ctx,
                    id: step.equation_after.rhs
                }
            );
            let relop_str = format!("{}", step.equation_after.op);
            let equation_str = format!("{} {} {}", lhs_str, relop_str, rhs_str);

            // LaTeX for equation parts
            let lhs_latex = cas_formatter::LaTeXExpr {
                context: ctx,
                id: step.equation_after.lhs,
            }
            .to_latex();
            let rhs_latex = cas_formatter::LaTeXExpr {
                context: ctx,
                id: step.equation_after.rhs,
            }
            .to_latex();
            let relop_latex = relop_to_latex(&step.equation_after.op);

            // Collect substeps (no filter - show all substeps for educational detail)
            let substeps: Vec<SolveSubStepJson> = step
                .substeps
                .iter()
                .enumerate()
                .map(|(j, ss)| {
                    let ss_lhs_str = format!(
                        "{}",
                        cas_formatter::DisplayExpr {
                            context: ctx,
                            id: ss.equation_after.lhs
                        }
                    );
                    let ss_rhs_str = format!(
                        "{}",
                        cas_formatter::DisplayExpr {
                            context: ctx,
                            id: ss.equation_after.rhs
                        }
                    );
                    let ss_relop_str = format!("{}", ss.equation_after.op);
                    let ss_equation_str = format!("{} {} {}", ss_lhs_str, ss_relop_str, ss_rhs_str);

                    let ss_lhs_latex = cas_formatter::LaTeXExpr {
                        context: ctx,
                        id: ss.equation_after.lhs,
                    }
                    .to_latex();
                    let ss_rhs_latex = cas_formatter::LaTeXExpr {
                        context: ctx,
                        id: ss.equation_after.rhs,
                    }
                    .to_latex();
                    let ss_relop_latex = relop_to_latex(&ss.equation_after.op);

                    SolveSubStepJson {
                        index: format!("{}.{}", i + 1, j + 1),
                        description: ss.description.clone(),
                        equation: ss_equation_str,
                        lhs_latex: ss_lhs_latex,
                        relop: ss_relop_latex,
                        rhs_latex: ss_rhs_latex,
                    }
                })
                .collect();

            SolveStepJson {
                index: i + 1,
                description: step.description.clone(),
                equation: equation_str,
                lhs_latex,
                relop: relop_latex,
                rhs_latex,
                substeps,
            }
        })
        .collect()
}

/// Convert RelOp to LaTeX string
fn relop_to_latex(op: &cas_ast::RelOp) -> String {
    match op {
        cas_ast::RelOp::Eq => "=".to_string(),
        cas_ast::RelOp::Lt => "<".to_string(),
        cas_ast::RelOp::Leq => r"\leq".to_string(),
        cas_ast::RelOp::Gt => ">".to_string(),
        cas_ast::RelOp::Geq => r"\geq".to_string(),
        cas_ast::RelOp::Neq => r"\neq".to_string(),
    }
}

/// Detect the variable to solve for in an equation.
/// Prefers 'x' if present, otherwise uses the first variable found.
fn detect_solve_variable(
    ctx: &cas_ast::Context,
    lhs: cas_ast::ExprId,
    rhs: cas_ast::ExprId,
) -> String {
    let mut variables = std::collections::HashSet::new();

    // Helper to collect variables from an expression
    fn collect_vars(
        ctx: &cas_ast::Context,
        expr: cas_ast::ExprId,
        vars: &mut std::collections::HashSet<String>,
    ) {
        match ctx.get(expr) {
            cas_ast::Expr::Variable(sym_id) => {
                // Skip special built-in variables
                let name = ctx.sym_name(*sym_id);
                if !name.starts_with('#') && name != "e" && name != "i" && name != "pi" {
                    vars.insert(name.to_string());
                }
            }
            cas_ast::Expr::Add(a, b)
            | cas_ast::Expr::Sub(a, b)
            | cas_ast::Expr::Mul(a, b)
            | cas_ast::Expr::Div(a, b)
            | cas_ast::Expr::Pow(a, b) => {
                collect_vars(ctx, *a, vars);
                collect_vars(ctx, *b, vars);
            }
            cas_ast::Expr::Neg(e) => {
                collect_vars(ctx, *e, vars);
            }
            cas_ast::Expr::Function(_, args) => {
                for arg in args {
                    collect_vars(ctx, *arg, vars);
                }
            }
            _ => {}
        }
    }

    collect_vars(ctx, lhs, &mut variables);
    collect_vars(ctx, rhs, &mut variables);

    // Prefer 'x' if present
    if variables.contains("x") {
        return "x".to_string();
    }

    // Otherwise prefer common variable names in order
    for preferred in &["y", "z", "t", "n", "a", "b", "c"] {
        if variables.contains(*preferred) {
            return preferred.to_string();
        }
    }

    // Fallback: first variable alphabetically, or "x" if none found
    variables
        .into_iter()
        .min()
        .unwrap_or_else(|| "x".to_string())
}

/// Parse solve(equation, variable) syntax.
/// Returns Some((equation_string, variable_name)) if the input matches solve syntax.
/// Supports:
/// - solve(x + y = 4, y)
/// - solve(x^2 - 4, x)  (treated as x^2 - 4 = 0)
fn parse_solve_command(input: &str) -> Option<(String, String)> {
    let trimmed = input.trim();

    // Check for "solve(" prefix (case-insensitive)
    if !trimmed.to_lowercase().starts_with("solve(") {
        return None;
    }

    // Must end with ')'
    if !trimmed.ends_with(')') {
        return None;
    }

    // Extract content between "solve(" and ")"
    let content = &trimmed[6..trimmed.len() - 1];

    // Find the last comma (the separator before the variable)
    // Must handle nested parentheses, e.g., solve(sin(x) = 0, x)
    let mut paren_depth = 0;
    let mut last_comma_pos = None;

    for (i, ch) in content.char_indices() {
        match ch {
            '(' => paren_depth += 1,
            ')' => paren_depth -= 1,
            ',' if paren_depth == 0 => last_comma_pos = Some(i),
            _ => {}
        }
    }

    let comma_pos = last_comma_pos?;

    let equation_part = content[..comma_pos].trim();
    let variable_part = content[comma_pos + 1..].trim();

    // Validate variable is a valid identifier (alphanumeric starting with letter)
    if variable_part.is_empty() || !variable_part.chars().next()?.is_alphabetic() {
        return None;
    }
    if !variable_part
        .chars()
        .all(|c| c.is_alphanumeric() || c == '_')
    {
        return None;
    }

    Some((equation_part.to_string(), variable_part.to_string()))
}

/// Parse limit(expression, variable, approach) syntax.
/// Returns Some((expr_string, variable_name, approach)) if the input matches limit syntax.
/// Approach can be: inf, infinity, -inf, -infinity, +inf, +infinity
/// If approach is omitted, defaults to +infinity.
///
/// Examples:
/// - limit(1/x, x, inf)
/// - limit(x^2, x, -inf)
/// - limit((x^2 + 3*x)/(2*x^2 - x), x, inf)
fn parse_limit_command(input: &str) -> Option<(String, String, cas_solver::Approach)> {
    use cas_solver::Approach;

    let trimmed = input.trim();

    // Check for "limit(" or "lim(" prefix (case-insensitive)
    let lower = trimmed.to_lowercase();
    let prefix_len = if lower.starts_with("limit(") {
        6
    } else if lower.starts_with("lim(") {
        4
    } else {
        return None;
    };

    // Must end with ')'
    if !trimmed.ends_with(')') {
        return None;
    }

    // Extract content between "limit(" and ")"
    let content = &trimmed[prefix_len..trimmed.len() - 1];

    // Split by commas at depth 0 (handle nested parentheses)
    let parts = split_by_comma_at_depth_0(content);

    if parts.len() < 2 || parts.len() > 3 {
        return None;
    }

    let expr_str = parts[0].trim();
    let var_str = parts[1].trim();

    // Validate variable is a valid identifier
    if var_str.is_empty() || !var_str.chars().next()?.is_alphabetic() {
        return None;
    }
    if !var_str.chars().all(|c| c.is_alphanumeric() || c == '_') {
        return None;
    }

    // Parse approach (default to PosInfinity)
    let approach = if parts.len() == 3 {
        let approach_str = parts[2].trim().to_lowercase();
        match approach_str.as_str() {
            "inf" | "infinity" | "+inf" | "+infinity" => Approach::PosInfinity,
            "-inf" | "-infinity" => Approach::NegInfinity,
            _ => return None, // Invalid approach
        }
    } else {
        Approach::PosInfinity // Default
    };

    Some((expr_str.to_string(), var_str.to_string(), approach))
}

/// Split a string by commas, but only at parenthesis depth 0.
fn split_by_comma_at_depth_0(s: &str) -> Vec<&str> {
    let mut result = Vec::new();
    let mut start = 0;
    let mut depth = 0;

    for (i, ch) in s.char_indices() {
        match ch {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth -= 1,
            ',' if depth == 0 => {
                result.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }

    // Push the last part
    result.push(&s[start..]);

    result
}

/// Format a SolutionSet as a human-readable string
fn format_solution_set(ctx: &cas_ast::Context, solution_set: &cas_ast::SolutionSet) -> String {
    use cas_ast::SolutionSet;

    match solution_set {
        SolutionSet::Empty => "No solution".to_string(),
        SolutionSet::AllReals => "All real numbers".to_string(),
        SolutionSet::Discrete(exprs) => {
            if exprs.is_empty() {
                "No solution".to_string()
            } else {
                let sols: Vec<String> = exprs
                    .iter()
                    .map(|e| {
                        format!(
                            "{}",
                            cas_formatter::DisplayExpr {
                                context: ctx,
                                id: *e
                            }
                        )
                    })
                    .collect();
                format!("{{ {} }}", sols.join(", "))
            }
        }
        SolutionSet::Conditional(cases) => {
            let mut parts = Vec::new();
            for case in cases {
                // Skip "otherwise" cases that only contain Residual - they don't add useful info
                if crate::format::is_pure_residual_otherwise(case) {
                    continue;
                }
                let cond_str = case.when.display_with_context(ctx);
                // Format inner solutions recursively
                let inner_str = format_solution_set(ctx, &case.then.solutions);
                if case.when.is_empty() {
                    parts.push(format!("{} otherwise", inner_str));
                } else {
                    parts.push(format!("{} if {}", inner_str, cond_str));
                }
            }
            // If all cases were filtered out, return the first non-otherwise case's solution
            if parts.is_empty() && !cases.is_empty() {
                // Find the first non-otherwise case
                for case in cases {
                    if !case.when.is_empty() {
                        let inner_str = format_solution_set(ctx, &case.then.solutions);
                        let cond_str = case.when.display_with_context(ctx);
                        return format!("{} if {}", inner_str, cond_str);
                    }
                }
            }
            parts.join("; ")
        }
        SolutionSet::Continuous(interval) => {
            format!(
                "[{}, {}]",
                cas_formatter::DisplayExpr {
                    context: ctx,
                    id: interval.min
                },
                cas_formatter::DisplayExpr {
                    context: ctx,
                    id: interval.max
                }
            )
        }
        SolutionSet::Union(intervals) => {
            let parts: Vec<String> = intervals
                .iter()
                .map(|int| {
                    format!(
                        "[{}, {}]",
                        cas_formatter::DisplayExpr {
                            context: ctx,
                            id: int.min
                        },
                        cas_formatter::DisplayExpr {
                            context: ctx,
                            id: int.max
                        }
                    )
                })
                .collect();
            parts.join(" ∪ ")
        }
        SolutionSet::Residual(expr) => {
            format!(
                "Solve: {} = 0",
                cas_formatter::DisplayExpr {
                    context: ctx,
                    id: *expr
                }
            )
        }
    }
}

/// Format a SolutionSet as LaTeX
fn solution_set_to_latex(ctx: &cas_ast::Context, solution_set: &cas_ast::SolutionSet) -> String {
    use cas_ast::SolutionSet;

    match solution_set {
        SolutionSet::Empty => r"\emptyset".to_string(),
        SolutionSet::AllReals => r"\mathbb{R}".to_string(),
        SolutionSet::Discrete(exprs) => {
            if exprs.is_empty() {
                r"\emptyset".to_string()
            } else {
                let sols: Vec<String> = exprs
                    .iter()
                    .map(|e| {
                        cas_formatter::LaTeXExpr {
                            context: ctx,
                            id: *e,
                        }
                        .to_latex()
                    })
                    .collect();
                format!(r"\left\{{ {} \right\}}", sols.join(", "))
            }
        }
        SolutionSet::Conditional(cases) => {
            let mut latex_parts = Vec::new();
            for case in cases {
                // Skip "otherwise" cases that only contain Residual - they don't add useful info
                if crate::format::is_pure_residual_otherwise(case) {
                    continue;
                }
                let cond_latex = case.when.latex_display_with_context(ctx);
                // Format inner solutions recursively
                let inner_latex = solution_set_to_latex(ctx, &case.then.solutions);
                if case.when.is_empty() {
                    latex_parts.push(format!(r"{} & \text{{otherwise}}", inner_latex));
                } else {
                    latex_parts.push(format!(r"{} & \text{{if }} {}", inner_latex, cond_latex));
                }
            }
            // If only one case remains after filtering, don't use \begin{cases}
            if latex_parts.len() == 1 {
                // Extract just the solution part (before the " & \text{if}")
                let single = &latex_parts[0];
                if let Some(idx) = single.find(r" & \text{if}") {
                    return single[..idx].to_string();
                }
            }
            format!(
                r"\begin{{cases}} {} \end{{cases}}",
                latex_parts.join(r" \\ ")
            )
        }
        SolutionSet::Continuous(interval) => {
            let min_latex = cas_formatter::LaTeXExpr {
                context: ctx,
                id: interval.min,
            }
            .to_latex();
            let max_latex = cas_formatter::LaTeXExpr {
                context: ctx,
                id: interval.max,
            }
            .to_latex();
            format!(r"\left[{}, {}\right]", min_latex, max_latex)
        }
        SolutionSet::Union(intervals) => {
            let parts: Vec<String> = intervals
                .iter()
                .map(|int| {
                    let min = cas_formatter::LaTeXExpr {
                        context: ctx,
                        id: int.min,
                    }
                    .to_latex();
                    let max = cas_formatter::LaTeXExpr {
                        context: ctx,
                        id: int.max,
                    }
                    .to_latex();
                    format!(r"\left[{}, {}\right]", min, max)
                })
                .collect();
            parts.join(r" \cup ")
        }
        SolutionSet::Residual(expr) => {
            let expr_latex = cas_formatter::LaTeXExpr {
                context: ctx,
                id: *expr,
            }
            .to_latex();
            format!(r"\text{{Solve: }} {} = 0", expr_latex)
        }
    }
}

fn print_pretty_json<T: serde::Serialize>(value: &T) {
    match serde_json::to_string_pretty(value) {
        Ok(s) => println!("{}", s),
        Err(e) => {
            eprintln!("JSON serialization error: {}", e);
            match serde_json::to_string(value) {
                Ok(s) => println!("{}", s),
                Err(_) => println!("{{\"ok\":false,\"error\":\"JSON_SERIALIZATION_FAILED\"}}"),
            }
        }
    }
}
