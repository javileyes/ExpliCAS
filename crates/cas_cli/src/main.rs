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
    expli eval \"expand((x+1)^5)\" --budget small --format json")]
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
}

fn main() -> rustyline::Result<()> {
    // Use clap for all argument parsing (unified, no manual parsing)
    let cli = Cli::parse();

    // Configure pretty output
    if cli.no_pretty {
        cas_ast::display::disable_pretty_output();
    } else {
        cas_ast::display::enable_pretty_output();
    }

    // Initialize tracing subscriber with WARN as default level
    // Use RUST_LOG=info or RUST_LOG=debug for more verbose output
    // Use RUST_LOG=budget=warn to see budget limit messages
    tracing_subscriber::fmt()
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
            };
            commands::eval_json::run(eval_args);
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
    use cas_ast::DisplayExpr;
    use cas_engine::{Engine, EvalAction, EvalRequest, EvalResult, SessionState};
    use cas_parser::parse;

    let mut engine = Engine::new();
    let mut state = SessionState::new();

    // Parse expression
    let parsed = match parse(&args.expr, &mut engine.simplifier.context) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Parse error: {}", e);
            std::process::exit(1);
        }
    };

    // Build eval request
    let req = EvalRequest {
        raw_input: args.expr.clone(),
        parsed,
        kind: cas_engine::EntryKind::Expr(parsed),
        action: EvalAction::Simplify,
        auto_store: false,
    };

    // Evaluate
    match engine.eval(&mut state, req) {
        Ok(output) => {
            let result_str = match &output.result {
                EvalResult::Expr(e) => {
                    format!(
                        "{}",
                        DisplayExpr {
                            context: &engine.simplifier.context,
                            id: *e
                        }
                    )
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
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    }
}

/// Convert BudgetPreset enum to string for JSON args
fn budget_preset_to_string(preset: BudgetPreset) -> String {
    match preset {
        BudgetPreset::Small => "small".to_string(),
        BudgetPreset::Standard | BudgetPreset::Cli => "standard".to_string(),
        BudgetPreset::Unlimited => "unlimited".to_string(),
    }
}
