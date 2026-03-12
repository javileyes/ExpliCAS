//! CLI argument and option type definitions.

use clap::{Parser, Subcommand, ValueEnum};

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
    expli envelope \"x/x\" --domain generic   Evaluate and return OutputEnvelope V1
    expli limit \"(x^2+1)/(2*x^2-3)\" --var x --to infinity  Compute limit")]
pub struct Cli {
    /// Use ASCII output (*, ^) instead of Unicode (·, ²)
    #[arg(long, global = true)]
    pub no_pretty: bool,

    #[command(subcommand)]
    pub command: Option<Command>,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Evaluate an expression
    Eval(EvalArgs),

    /// Evaluate and return stable OutputEnvelope V1 (for Android/FFI)
    Envelope(crate::commands::envelope::EnvelopeArgs),

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

/// Steps mode for eval command
#[derive(ValueEnum, Debug, Clone, Copy, Default)]
pub enum StepsArg {
    #[default]
    Off,
    On,
    Compact,
}

/// Context mode for eval command
#[derive(ValueEnum, Debug, Clone, Copy, Default)]
pub enum ContextArg {
    #[default]
    Auto,
    Standard,
    Solve,
    Integrate,
}

/// Branch mode for eval command
#[derive(ValueEnum, Debug, Clone, Copy, Default)]
pub enum EvalBranchArg {
    #[default]
    Strict,
    Principal,
}

/// Complex evaluation mode for eval command
#[derive(ValueEnum, Debug, Clone, Copy, Default)]
pub enum ComplexModeArg {
    #[default]
    Auto,
    On,
    Off,
}

/// Autoexpand policy for eval command
#[derive(ValueEnum, Debug, Clone, Copy, Default)]
pub enum AutoexpandArg {
    #[default]
    Off,
    Auto,
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
    #[arg(long, value_enum, default_value_t = BudgetPreset::Standard)]
    pub budget: BudgetPreset,

    /// Strict mode: fail with error on budget exceeded (default: best-effort)
    #[arg(long, default_value_t = false)]
    pub strict: bool,

    /// Maximum characters in result output (wire output only, truncates if larger)
    #[arg(long, default_value_t = 2000)]
    pub max_chars: usize,

    /// Steps mode: on, off, compact
    #[arg(long, value_enum, default_value_t = StepsArg::Off)]
    pub steps: StepsArg,

    /// Context mode: auto, standard, solve, integrate
    #[arg(long, value_enum, default_value_t = ContextArg::Auto)]
    pub context: ContextArg,

    /// Branch mode: strict, principal
    #[arg(long, value_enum, default_value_t = EvalBranchArg::Strict)]
    pub branch: EvalBranchArg,

    /// Complex mode: auto, on, off
    #[arg(long, value_enum, default_value_t = ComplexModeArg::Auto)]
    pub complex: ComplexModeArg,

    /// Expand policy: off, auto
    #[arg(long, value_enum, default_value_t = AutoexpandArg::Off)]
    pub autoexpand: AutoexpandArg,

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
