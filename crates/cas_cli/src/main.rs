//! CAS CLI - Computer Algebra System Command Line Interface
//!
//! Supports both interactive REPL mode (default) and subcommands for scripting.

mod commands;
mod completer;
mod config;
mod format;
mod health_suite;
pub mod json_types;
pub mod repl;

use clap::{Parser, Subcommand, ValueEnum};
use repl::Repl;

/// Rust CAS - Computer Algebra System
#[derive(Parser, Debug)]
#[command(name = "expli")]
#[command(about = "A symbolic mathematics engine with step-by-step explanations")]
#[command(version)]
#[command(after_help = "EXAMPLES:
    expli                                   Start interactive REPL
    expli eval \"x^2 + 1\"                    Evaluate expression (text output)
    expli eval \"x^2 + 1\" --format json      Evaluate expression (JSON output)
    expli eval \"expand((x+1)^5)\" --budget small --format json
    expli limit \"(x^2+1)/(2*x^2-3)\" --var x --to infinity  Compute limit")]
struct Cli {
    /// Use ASCII output (*, ^) instead of Unicode (·, ²)
    #[arg(long, global = true)]
    no_pretty: bool,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Evaluate an expression
    Eval(EvalArgs),

    /// Evaluate an expression and return JSON output (alias for: eval --format json)
    #[command(name = "eval-json", hide = true)]
    EvalJson(EvalJsonLegacyArgs),

    /// Evaluate and return stable OutputEnvelope V1 (for Android/FFI)
    #[command(name = "envelope-json")]
    EnvelopeJson(commands::envelope_json::EnvelopeJsonArgs),

    /// Compute the limit of an expression
    Limit(LimitArgs),

    /// Substitute a target expression with a replacement
    Substitute(SubstituteArgs),

    /// Start interactive REPL (default if no subcommand given)
    Repl,
}

/// Output format for eval command
#[derive(ValueEnum, Debug, Clone, Copy, Default)]
pub enum OutputFormat {
    #[default]
    Text,
    Json,
}

/// Budget preset for resource limits
///
/// Presets only control numeric limits, NOT error handling mode.
/// Use `--strict` to control what happens when limits are exceeded.
#[derive(ValueEnum, Debug, Clone, Copy, Default)]
pub enum BudgetPreset {
    /// Conservative limits (5k rewrites, 25k nodes) - for teaching/REPL
    Small,
    /// Standard limits for interactive use (50k rewrites, 250k nodes)
    #[default]
    Standard,
    /// Alias for 'standard' (deprecated, use --budget standard)
    #[value(hide = true)]
    Cli,
    /// No limits (use with caution)
    Unlimited,
}

/// Approach direction for limits
#[derive(ValueEnum, Debug, Clone, Copy, Default)]
pub enum ApproachArg {
    /// x → +∞
    #[default]
    Infinity,
    /// x → -∞
    #[value(name = "-infinity")]
    NegInfinity,
}

/// Pre-simplification mode for limits
#[derive(ValueEnum, Debug, Clone, Copy, Default)]
pub enum PreSimplifyArg {
    /// No pre-simplification (most conservative)
    #[default]
    Off,
    /// Safe pre-simplification (allowlist only, no domain assumptions)
    Safe,
}

/// Domain mode for cancellation rules
#[derive(ValueEnum, Debug, Clone, Copy, Default)]
pub enum DomainArg {
    /// Only cancel factors provably non-zero (safest)
    Strict,
    /// Always cancel, silently (legacy behavior)
    #[default]
    Generic,
    /// Cancel and emit warnings/assumptions
    Assume,
}

/// Value domain for constant evaluation
#[derive(ValueEnum, Debug, Clone, Copy, Default)]
pub enum ValueDomainArg {
    /// Real numbers extended with ±∞ (sqrt(-1) → undefined)
    #[default]
    Real,
    /// Complex numbers with principal branch (sqrt(-1) → i)
    Complex,
}

/// Inverse trig composition policy
#[derive(ValueEnum, Debug, Clone, Copy, Default)]
pub enum InvTrigArg {
    /// Do not simplify arctan(tan(x)) etc.
    #[default]
    Strict,
    /// Simplify with principal domain assumption + warning
    Principal,
}

/// Branch policy for multi-valued functions (only if complex)
#[derive(ValueEnum, Debug, Clone, Copy, Default)]
pub enum BranchArg {
    /// Use principal branch
    #[default]
    Principal,
}

/// Constant folding mode
#[derive(ValueEnum, Debug, Clone, Copy, Default)]
pub enum ConstFoldArg {
    /// No constant folding (default)
    #[default]
    Off,
    /// Safe constant folding (allowlist only)
    Safe,
}

/// Assume scope for domain assumptions
#[derive(ValueEnum, Debug, Clone, Copy, Default)]
pub enum AssumeScopeArg {
    /// Assume for ℝ, error if ℂ needed (default)
    #[default]
    Real,
    /// Assume for ℝ, residual+warning if ℂ needed
    Wildcard,
}

/// Substitute mode
#[derive(ValueEnum, Debug, Clone, Copy, Default)]
pub enum SubstituteModeArg {
    /// Exact structural matching only
    Exact,
    /// Power-aware matching (x^4 with target x^2 → y^2)
    #[default]
    Power,
}

/// Arguments for substitute subcommand
#[derive(clap::Args, Debug)]
pub struct SubstituteArgs {
    /// Expression to substitute in
    pub expr: String,

    /// Target expression to replace
    #[arg(long)]
    pub target: String,

    /// Replacement expression
    #[arg(long = "with")]
    pub replacement: String,

    /// Substitution mode
    #[arg(long, value_enum, default_value_t = SubstituteModeArg::Power)]
    pub mode: SubstituteModeArg,

    /// Output format
    #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
    pub format: OutputFormat,

    /// Include simplification steps in output
    #[arg(long, default_value_t = false)]
    pub steps: bool,
}

/// Arguments for limit subcommand
#[derive(clap::Args, Debug)]
pub struct LimitArgs {
    /// Expression to compute limit of
    pub expr: String,

    /// Variable approaching the limit point
    #[arg(long, default_value = "x")]
    pub var: String,

    /// Direction of approach
    #[arg(long, value_enum, default_value_t = ApproachArg::Infinity)]
    pub to: ApproachArg,

    /// Pre-simplification mode (off = conservative, safe = allowlist transforms)
    #[arg(long, value_enum, default_value_t = PreSimplifyArg::Off)]
    pub presimplify: PreSimplifyArg,

    /// Output format
    #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
    pub format: OutputFormat,

    /// Budget preset for resource limits
    #[arg(long, value_enum, default_value_t = BudgetPreset::Standard)]
    pub budget: BudgetPreset,
}

/// Arguments for eval subcommand
#[derive(clap::Args, Debug)]
pub struct EvalArgs {
    /// Expression to evaluate (use "-" or omit to read from stdin)
    #[arg(default_value = "-")]
    pub expr: String,

    /// Output format
    #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
    pub format: OutputFormat,

    /// Budget preset for resource limits
    #[arg(long, value_enum, default_value_t = BudgetPreset::Cli)]
    pub budget: BudgetPreset,

    /// Strict mode: fail with error on budget exceeded (default: best-effort)
    #[arg(long, default_value_t = false)]
    pub strict: bool,

    /// Maximum characters in result output (JSON only, truncates if larger)
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

    /// Number of threads for parallel processing
    #[arg(long)]
    pub threads: Option<usize>,

    /// Domain mode for cancellation rules (strict, generic, assume)
    #[arg(long, value_enum, default_value_t = DomainArg::Generic)]
    pub domain: DomainArg,

    /// Value domain: real or complex
    #[arg(long, value_enum, default_value_t = ValueDomainArg::Real)]
    pub value_domain: ValueDomainArg,

    /// Inverse trig composition policy: strict or principal
    #[arg(long, value_enum, default_value_t = InvTrigArg::Strict)]
    pub inv_trig: InvTrigArg,

    /// Branch policy for multi-valued functions (if complex)
    #[arg(long, value_enum, default_value_t = BranchArg::Principal)]
    pub complex_branch: BranchArg,

    /// Constant folding mode
    #[arg(long, value_enum, default_value_t = ConstFoldArg::Off)]
    pub const_fold: ConstFoldArg,

    /// Assume scope (only active when domain=assume)
    #[arg(long, value_enum, default_value_t = AssumeScopeArg::Real)]
    pub assume_scope: AssumeScopeArg,

    /// Path to session file for persistent session across CLI invocations.
    /// Enables `#N` references to work across multiple eval calls.
    #[arg(long)]
    pub session: Option<std::path::PathBuf>,
}

/// Legacy eval-json arguments (hidden, for backward compatibility)
#[derive(clap::Args, Debug)]
pub struct EvalJsonLegacyArgs {
    /// Expression to evaluate
    pub expr: String,

    #[arg(long, default_value_t = 2000)]
    pub max_chars: usize,

    #[arg(long, default_value = "off")]
    pub steps: String,

    #[arg(long, default_value = "auto")]
    pub context: String,

    #[arg(long, default_value = "strict")]
    pub branch: String,

    #[arg(long, default_value = "auto")]
    pub complex: String,

    #[arg(long, default_value = "off")]
    pub autoexpand: String,

    #[arg(long)]
    pub threads: Option<usize>,

    #[arg(long, value_enum, default_value_t = DomainArg::Generic)]
    pub domain: DomainArg,

    #[arg(long, value_enum, default_value_t = ValueDomainArg::Real)]
    pub value_domain: ValueDomainArg,

    #[arg(long, value_enum, default_value_t = InvTrigArg::Strict)]
    pub inv_trig: InvTrigArg,

    #[arg(long, value_enum, default_value_t = BranchArg::Principal)]
    pub branch_policy: BranchArg,

    #[arg(long, value_enum, default_value_t = ConstFoldArg::Off)]
    pub const_fold: ConstFoldArg,

    #[arg(long, value_enum, default_value_t = AssumeScopeArg::Real)]
    pub assume_scope: AssumeScopeArg,
}

fn main() -> rustyline::Result<()> {
    // Use clap for all argument parsing (unified, no manual parsing)
    let cli = Cli::parse();

    // Configure pretty output
    if cli.no_pretty {
        cas_formatter::display::disable_pretty_output();
    } else {
        cas_formatter::display::enable_pretty_output();
    }

    // Initialize tracing subscriber with WARN as default level
    // Use RUST_LOG=info or RUST_LOG=debug for more verbose output
    // Use RUST_LOG=budget=warn to see budget limit messages
    // IMPORTANT: Write to stderr to avoid corrupting JSON output on stdout
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::builder()
                .with_default_directive(tracing::Level::WARN.into())
                .from_env_lossy(),
        )
        .init();

    // Handle subcommand or default to REPL
    match cli.command {
        Some(Command::Eval(args)) => {
            run_eval(args);
            Ok(())
        }
        Some(Command::EvalJson(args)) => {
            // Legacy: convert to new format and run as JSON
            let eval_args = commands::eval_json::EvalJsonArgs {
                expr: args.expr,
                budget_preset: "standard".to_string(),
                strict: false,
                max_chars: args.max_chars,
                steps: args.steps,
                context: args.context,
                branch: args.branch,
                complex: args.complex,
                autoexpand: args.autoexpand,
                threads: args.threads,
                domain: domain_arg_to_string(args.domain),
                value_domain: value_domain_arg_to_string(args.value_domain),
                inv_trig: inv_trig_arg_to_string(args.inv_trig),
                complex_branch: branch_arg_to_string(args.branch_policy),
                const_fold: const_fold_arg_to_string(args.const_fold),
                assume_scope: assume_scope_arg_to_string(args.assume_scope),
                session: None, // Legacy command doesn't support sessions
            };
            commands::eval_json::run(eval_args);
            Ok(())
        }
        Some(Command::EnvelopeJson(args)) => {
            commands::envelope_json::run(args);
            Ok(())
        }
        Some(Command::Limit(args)) => {
            run_limit(args);
            Ok(())
        }
        Some(Command::Substitute(args)) => {
            run_substitute(args);
            Ok(())
        }
        Some(Command::Repl) | None => {
            // Default: start REPL
            let mut repl = Repl::new();
            repl.run()
        }
    }
}

/// Run eval command with format and budget support
fn run_eval(args: EvalArgs) {
    // Read expression from stdin if needed
    let expr = read_expr_or_stdin(&args.expr);
    if expr.is_empty() {
        eprintln!("Error: No expression provided");
        std::process::exit(1);
    }

    // Set thread count if specified
    if let Some(n) = args.threads {
        std::env::set_var("RAYON_NUM_THREADS", n.to_string());
    }

    // Log budget preset being used
    tracing::debug!(
        target: "budget",
        preset = ?args.budget,
        "Using budget preset"
    );

    match args.format {
        OutputFormat::Json => {
            // Delegate to existing JSON handler
            let json_args = commands::eval_json::EvalJsonArgs {
                expr: expr.clone(),
                budget_preset: budget_preset_to_string(args.budget),
                strict: args.strict,
                max_chars: args.max_chars,
                steps: args.steps,
                context: args.context,
                branch: args.branch,
                complex: args.complex,
                autoexpand: args.autoexpand,
                threads: args.threads,
                domain: domain_arg_to_string(args.domain),
                value_domain: value_domain_arg_to_string(args.value_domain),
                inv_trig: inv_trig_arg_to_string(args.inv_trig),
                complex_branch: branch_arg_to_string(args.complex_branch),
                const_fold: const_fold_arg_to_string(args.const_fold),
                assume_scope: assume_scope_arg_to_string(args.assume_scope),
                session: args.session.clone(),
            };
            commands::eval_json::run(json_args);
        }
        OutputFormat::Text => {
            // Simple text output - create modified args with resolved expr
            let text_args = EvalArgs { expr, ..args };
            run_eval_text(&text_args);
        }
    }
}

/// Read expression from argument or stdin if "-"
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

/// Run eval with text output
fn run_eval_text(args: &EvalArgs) {
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;
    use cas_session::SimplifyCacheKey;
    use cas_solver::{EvalAction, EvalRequest, EvalResult};

    // Build cache key for snapshot compatibility check
    let domain_mode = match args.domain {
        DomainArg::Strict => cas_solver::DomainMode::Strict,
        DomainArg::Generic => cas_solver::DomainMode::Generic,
        DomainArg::Assume => cas_solver::DomainMode::Assume,
    };
    let cache_key = SimplifyCacheKey::from_context(domain_mode);

    // Load or create session and context
    let (mut engine, mut state) = load_or_new_session(&args.session, &cache_key);

    // Parse expression
    let parsed = match parse(&args.expr, &mut engine.simplifier.context) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Parse error: {}", e);
            std::process::exit(1);
        }
    };

    // Build eval request (auto_store=true to enable #N references)
    let req = EvalRequest {
        raw_input: args.expr.clone(),
        parsed,
        action: EvalAction::Simplify,
        auto_store: args.session.is_some(), // Store if using persistent session
    };

    // Evaluate
    match engine.eval(&mut state, req) {
        Ok(output) => {
            let result_str = match &output.result {
                EvalResult::Expr(e) => {
                    // Try to render as poly_result first (fast path for large polynomials)
                    if let Some(poly_str) =
                        cas_solver::try_render_poly_result(&engine.simplifier.context, *e)
                    {
                        poly_str
                    } else {
                        format!(
                            "{}",
                            DisplayExpr {
                                context: &engine.simplifier.context,
                                id: *e
                            }
                        )
                    }
                }
                EvalResult::Set(v) if !v.is_empty() => {
                    format!(
                        "{}",
                        DisplayExpr {
                            context: &engine.simplifier.context,
                            id: v[0]
                        }
                    )
                }
                EvalResult::Bool(b) => b.to_string(),
                _ => "(no result)".to_string(),
            };
            println!("{}", result_str);

            // Save session snapshot if --session is specified
            if let Some(ref path) = args.session {
                if let Err(e) = save_session(&engine, &state, path, &cache_key) {
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

/// Load session from snapshot file or create new session
fn load_or_new_session(
    path: &Option<std::path::PathBuf>,
    key: &cas_session::SimplifyCacheKey,
) -> (cas_solver::Engine, cas_session::SessionState) {
    let Some(path) = path else {
        return (cas_solver::Engine::new(), cas_session::SessionState::new());
    };

    if !path.exists() {
        return (cas_solver::Engine::new(), cas_session::SessionState::new());
    }

    match cas_session::SessionState::load_compatible_snapshot(path, key) {
        Ok(Some((ctx, state))) => {
            let engine = cas_solver::Engine::with_context(ctx);
            (engine, state)
        }
        Ok(None) => {
            eprintln!("Session snapshot incompatible, starting fresh");
            (cas_solver::Engine::new(), cas_session::SessionState::new())
        }
        Err(e) => {
            eprintln!("Warning: Failed to load session ({}), starting fresh", e);
            (cas_solver::Engine::new(), cas_session::SessionState::new())
        }
    }
}

/// Save session to snapshot file
fn save_session(
    engine: &cas_solver::Engine,
    state: &cas_session::SessionState,
    path: &std::path::Path,
    key: &cas_session::SimplifyCacheKey,
) -> Result<(), cas_session::SnapshotError> {
    state.save_snapshot(&engine.simplifier.context, path, key.clone())
}

/// Convert BudgetPreset enum to string for JSON args
fn budget_preset_to_string(preset: BudgetPreset) -> String {
    match preset {
        BudgetPreset::Small => "small".to_string(),
        BudgetPreset::Standard | BudgetPreset::Cli => "standard".to_string(),
        BudgetPreset::Unlimited => "unlimited".to_string(),
    }
}

/// Convert DomainArg enum to string for JSON args
fn domain_arg_to_string(domain: DomainArg) -> String {
    match domain {
        DomainArg::Strict => "strict".to_string(),
        DomainArg::Generic => "generic".to_string(),
        DomainArg::Assume => "assume".to_string(),
    }
}

/// Convert ValueDomainArg enum to string for JSON args
fn value_domain_arg_to_string(vd: ValueDomainArg) -> String {
    match vd {
        ValueDomainArg::Real => "real".to_string(),
        ValueDomainArg::Complex => "complex".to_string(),
    }
}

/// Convert InvTrigArg enum to string for JSON args
fn inv_trig_arg_to_string(it: InvTrigArg) -> String {
    match it {
        InvTrigArg::Strict => "strict".to_string(),
        InvTrigArg::Principal => "principal".to_string(),
    }
}

/// Convert BranchArg enum to string for JSON args
fn branch_arg_to_string(b: BranchArg) -> String {
    match b {
        BranchArg::Principal => "principal".to_string(),
    }
}

/// Convert ConstFoldArg enum to string for JSON args
fn const_fold_arg_to_string(cf: ConstFoldArg) -> String {
    match cf {
        ConstFoldArg::Off => "off".to_string(),
        ConstFoldArg::Safe => "safe".to_string(),
    }
}

/// Convert AssumeScopeArg enum to string for JSON args
fn assume_scope_arg_to_string(as_: AssumeScopeArg) -> String {
    match as_ {
        AssumeScopeArg::Real => "real".to_string(),
        AssumeScopeArg::Wildcard => "wildcard".to_string(),
    }
}

fn run_limit(args: LimitArgs) {
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;
    use cas_solver::{limit, Approach, Budget, LimitOptions, PreSimplifyMode};

    let mut ctx = cas_ast::Context::new();

    // Parse expression
    let expr = match parse(&args.expr, &mut ctx) {
        Ok(e) => e,
        Err(e) => {
            match args.format {
                OutputFormat::Json => {
                    let json = serde_json::json!({
                        "ok": false,
                        "error": format!("Parse error: {}", e),
                        "code": "PARSE_ERROR"
                    });
                    println!("{}", json);
                }
                OutputFormat::Text => {
                    eprintln!("Parse error: {}", e);
                    std::process::exit(1);
                }
            }
            return;
        }
    };

    // Parse variable
    let var = ctx.var(&args.var);

    // Convert approach
    let approach = match args.to {
        ApproachArg::Infinity => Approach::PosInfinity,
        ApproachArg::NegInfinity => Approach::NegInfinity,
    };

    // Convert presimplify mode
    let presimplify = match args.presimplify {
        PreSimplifyArg::Off => PreSimplifyMode::Off,
        PreSimplifyArg::Safe => PreSimplifyMode::Safe,
    };

    // Run limit
    let mut budget = Budget::new();
    let opts = LimitOptions {
        presimplify,
        ..Default::default()
    };

    let result = limit(&mut ctx, expr, var, approach, &opts, &mut budget);

    match result {
        Ok(limit_result) => {
            let result_str = DisplayExpr {
                context: &ctx,
                id: limit_result.expr,
            }
            .to_string();

            match args.format {
                OutputFormat::Json => {
                    let mut json = serde_json::json!({
                        "ok": true,
                        "result": result_str,
                    });
                    if let Some(warning) = &limit_result.warning {
                        json["warning"] = serde_json::Value::String(warning.clone());
                    }
                    println!("{}", json);
                }
                OutputFormat::Text => {
                    println!("{}", result_str);
                    if let Some(warning) = &limit_result.warning {
                        eprintln!("Warning: {}", warning);
                    }
                }
            }
        }
        Err(e) => match args.format {
            OutputFormat::Json => {
                let json = serde_json::json!({
                    "ok": false,
                    "error": format!("{}", e),
                    "code": "LIMIT_ERROR"
                });
                println!("{}", json);
            }
            OutputFormat::Text => {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        },
    }
}

fn run_substitute(args: SubstituteArgs) {
    // For JSON output, delegate to canonical API (single source of truth)
    if matches!(args.format, OutputFormat::Json) {
        let mode_str = match args.mode {
            SubstituteModeArg::Exact => "exact",
            SubstituteModeArg::Power => "power",
        };
        let opts_json = serde_json::json!({
            "mode": mode_str,
            "steps": args.steps,
            "pretty": true
        });
        let opts_str = match serde_json::to_string(&opts_json) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Internal error: failed to serialize options: {}", e);
                std::process::exit(1);
            }
        };

        let out = cas_solver::substitute_str_to_json(
            &args.expr,
            &args.target,
            &args.replacement,
            Some(&opts_str),
        );
        println!("{}", out);
        return;
    }

    // Text output path (unchanged)
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;
    use cas_solver::{substitute_power_aware, substitute_with_steps, SubstituteOptions};

    let mut ctx = cas_ast::Context::new();

    // Parse expression
    let expr = match parse(&args.expr, &mut ctx) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Parse error in expression: {}", e);
            std::process::exit(1);
        }
    };

    // Parse target
    let target = match parse(&args.target, &mut ctx) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Parse error in target: {}", e);
            std::process::exit(1);
        }
    };

    // Parse replacement
    let replacement = match parse(&args.replacement, &mut ctx) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Parse error in replacement: {}", e);
            std::process::exit(1);
        }
    };

    // Build options
    let mut opts = match args.mode {
        SubstituteModeArg::Exact => SubstituteOptions::exact(),
        SubstituteModeArg::Power => SubstituteOptions::default(),
    };
    if args.steps {
        opts = opts.with_steps();
    }

    // Run substitution with steps if requested
    let (result, steps) = if args.steps {
        let res = substitute_with_steps(&mut ctx, expr, target, replacement, opts);
        (res.expr, res.steps)
    } else {
        (
            substitute_power_aware(&mut ctx, expr, target, replacement, opts),
            vec![],
        )
    };

    // Text output
    if args.steps && !steps.is_empty() {
        println!("Steps:");
        for step in &steps {
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
    println!(
        "{}",
        DisplayExpr {
            context: &ctx,
            id: result
        }
    );
}
