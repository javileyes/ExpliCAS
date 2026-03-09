//! eval-json subcommand handler.
//!
//! Evaluates a single expression and returns JSON output.

use std::path::PathBuf;

use clap::Args;

use crate::{
    AssumeScopeArg, BranchArg, BudgetPreset, ConstFoldArg, DomainArg, EvalArgs, EvalJsonLegacyArgs,
    InvTrigArg, ValueDomainArg,
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

/// Build JSON args from the richer CLI `eval` args.
pub fn from_eval_args(expr: String, args: &EvalArgs) -> EvalJsonArgs {
    EvalJsonArgs {
        expr,
        budget_preset: budget_preset_to_string(args.budget),
        strict: args.strict,
        max_chars: args.max_chars,
        steps: args.steps.clone(),
        context: args.context.clone(),
        branch: args.branch.clone(),
        complex: args.complex.clone(),
        autoexpand: args.autoexpand.clone(),
        threads: args.threads,
        domain: domain_arg_to_string(args.domain),
        value_domain: value_domain_arg_to_string(args.value_domain),
        inv_trig: inv_trig_arg_to_string(args.inv_trig),
        complex_branch: branch_arg_to_string(args.complex_branch),
        const_fold: const_fold_arg_to_string(args.const_fold),
        assume_scope: assume_scope_arg_to_string(args.assume_scope),
        session: args.session.clone(),
    }
}

/// Build JSON args from legacy `eval-json` CLI args.
pub fn from_legacy_eval_json_args(args: EvalJsonLegacyArgs) -> EvalJsonArgs {
    EvalJsonArgs {
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
        session: None,
    }
}

fn budget_preset_to_string(preset: BudgetPreset) -> String {
    match preset {
        BudgetPreset::Small => "small".to_string(),
        BudgetPreset::Standard | BudgetPreset::Cli => "standard".to_string(),
        BudgetPreset::Unlimited => "unlimited".to_string(),
    }
}

fn domain_arg_to_string(domain: DomainArg) -> String {
    match domain {
        DomainArg::Strict => "strict".to_string(),
        DomainArg::Generic => "generic".to_string(),
        DomainArg::Assume => "assume".to_string(),
    }
}

fn value_domain_arg_to_string(vd: ValueDomainArg) -> String {
    match vd {
        ValueDomainArg::Real => "real".to_string(),
        ValueDomainArg::Complex => "complex".to_string(),
    }
}

fn inv_trig_arg_to_string(it: InvTrigArg) -> String {
    match it {
        InvTrigArg::Strict => "strict".to_string(),
        InvTrigArg::Principal => "principal".to_string(),
    }
}

fn branch_arg_to_string(b: BranchArg) -> String {
    match b {
        BranchArg::Principal => "principal".to_string(),
    }
}

fn const_fold_arg_to_string(cf: ConstFoldArg) -> String {
    match cf {
        ConstFoldArg::Off => "off".to_string(),
        ConstFoldArg::Safe => "safe".to_string(),
    }
}

fn assume_scope_arg_to_string(as_: AssumeScopeArg) -> String {
    match as_ {
        AssumeScopeArg::Real => "real".to_string(),
        AssumeScopeArg::Wildcard => "wildcard".to_string(),
    }
}

/// Run the eval-json command
pub fn run(args: EvalJsonArgs) {
    // Set thread count if specified
    if let Some(n) = args.threads {
        std::env::set_var("RAYON_NUM_THREADS", n.to_string());
    }

    let output = cas_session::evaluate_eval_json_command_pretty_with_session(
        args.session.as_deref(),
        eval_json_command_config(&args),
        |steps, events, context, steps_mode| {
            cas_didactic::collect_step_payloads_with_events(steps, events, context, steps_mode)
        },
    );
    println!("{}", output);
}

fn eval_json_command_config(args: &EvalJsonArgs) -> cas_session::EvalJsonCommandConfig<'_> {
    cas_session::EvalJsonCommandConfig {
        expr: &args.expr,
        auto_store: args.session.is_some(),
        max_chars: args.max_chars,
        steps_mode: &args.steps,
        budget_preset: &args.budget_preset,
        strict: args.strict,
        domain: &args.domain,
        context_mode: &args.context,
        branch_mode: &args.branch,
        expand_policy: &args.autoexpand,
        complex_mode: &args.complex,
        const_fold: &args.const_fold,
        value_domain: &args.value_domain,
        complex_branch: &args.complex_branch,
        inv_trig: &args.inv_trig,
        assume_scope: &args.assume_scope,
    }
}
