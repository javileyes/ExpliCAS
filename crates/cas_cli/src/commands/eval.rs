//! eval subcommand dispatcher.
//!
//! Handles stdin/arg expression sourcing and routes to wire/text handlers.

use super::output::CommandOutput;
use crate::{
    AssumeScopeArg, AutoexpandArg, BranchArg, BudgetPreset, ComplexModeArg, ConstFoldArg,
    ContextArg, DomainArg, EvalArgs, EvalBranchArg, InvTrigArg, OutputFormat, StepsArg,
    ValueDomainArg,
};
use cas_api_models::{
    EvalAssumeScope, EvalBranchMode, EvalBudgetPreset, EvalComplexMode, EvalConstFoldMode,
    EvalContextMode, EvalDomainMode, EvalExpandPolicy, EvalInvTrigPolicy, EvalStepsMode,
    EvalValueDomain,
};

pub(crate) fn render(mut args: EvalArgs) -> Result<CommandOutput, String> {
    let expr = read_expr_or_stdin(&args.expr);
    if expr.is_empty() {
        return Err("Error: No expression provided".to_string());
    }

    if let Some(n) = args.threads {
        std::env::set_var("RAYON_NUM_THREADS", n.to_string());
    }

    tracing::debug!(
        target: "budget",
        preset = ?args.budget,
        "Using budget preset"
    );

    args.expr = expr;

    match args.format {
        OutputFormat::Json => Ok(render_wire(&args)),
        OutputFormat::Text => crate::commands::eval_text::render(&args),
    }
}

fn render_wire(args: &EvalArgs) -> CommandOutput {
    CommandOutput::from_stdout(
        cas_session::eval::evaluate_eval_command_pretty_with_session(
            args.session.as_deref(),
            eval_command_config(&args.expr, args),
            |steps, events, context, steps_mode| {
                cas_didactic::collect_step_payloads_with_events(steps, events, context, steps_mode)
            },
        ),
    )
}

fn eval_command_config<'a>(
    expr: &'a str,
    args: &'a EvalArgs,
) -> cas_session::eval::EvalCommandConfig<'a> {
    cas_session::eval::EvalCommandConfig {
        expr,
        auto_store: args.session.is_some(),
        max_chars: args.max_chars,
        steps_mode: steps_mode(args.steps),
        budget_preset: budget_preset(args.budget),
        strict: args.strict,
        domain: domain_mode(args.domain),
        context_mode: context_mode(args.context),
        branch_mode: eval_branch_mode(args.branch),
        expand_policy: autoexpand_mode(args.autoexpand),
        complex_mode: complex_mode(args.complex),
        const_fold: const_fold_mode(args.const_fold),
        value_domain: value_domain(args.value_domain),
        complex_branch: complex_branch_mode(args.complex_branch),
        inv_trig: inv_trig_mode(args.inv_trig),
        assume_scope: assume_scope(args.assume_scope),
    }
}

fn budget_preset(preset: BudgetPreset) -> EvalBudgetPreset {
    match preset {
        BudgetPreset::Small => EvalBudgetPreset::Small,
        BudgetPreset::Standard => EvalBudgetPreset::Standard,
        BudgetPreset::Unlimited => EvalBudgetPreset::Unlimited,
    }
}

fn steps_mode(mode: StepsArg) -> EvalStepsMode {
    match mode {
        StepsArg::Off => EvalStepsMode::Off,
        StepsArg::On => EvalStepsMode::On,
        StepsArg::Compact => EvalStepsMode::Compact,
    }
}

fn context_mode(mode: ContextArg) -> EvalContextMode {
    match mode {
        ContextArg::Auto => EvalContextMode::Auto,
        ContextArg::Standard => EvalContextMode::Standard,
        ContextArg::Solve => EvalContextMode::Solve,
        ContextArg::Integrate => EvalContextMode::Integrate,
    }
}

fn eval_branch_mode(mode: EvalBranchArg) -> EvalBranchMode {
    match mode {
        EvalBranchArg::Strict => EvalBranchMode::Strict,
        EvalBranchArg::Principal => EvalBranchMode::Principal,
    }
}

fn autoexpand_mode(mode: AutoexpandArg) -> EvalExpandPolicy {
    match mode {
        AutoexpandArg::Off => EvalExpandPolicy::Off,
        AutoexpandArg::Auto => EvalExpandPolicy::Auto,
    }
}

fn domain_mode(domain: DomainArg) -> EvalDomainMode {
    match domain {
        DomainArg::Strict => EvalDomainMode::Strict,
        DomainArg::Generic => EvalDomainMode::Generic,
        DomainArg::Assume => EvalDomainMode::Assume,
    }
}

fn value_domain(vd: ValueDomainArg) -> EvalValueDomain {
    match vd {
        ValueDomainArg::Real => EvalValueDomain::Real,
        ValueDomainArg::Complex => EvalValueDomain::Complex,
    }
}

fn inv_trig_mode(mode: InvTrigArg) -> EvalInvTrigPolicy {
    match mode {
        InvTrigArg::Strict => EvalInvTrigPolicy::Strict,
        InvTrigArg::Principal => EvalInvTrigPolicy::Principal,
    }
}

fn complex_branch_mode(mode: BranchArg) -> EvalBranchMode {
    match mode {
        BranchArg::Principal => EvalBranchMode::Principal,
    }
}

fn complex_mode(mode: ComplexModeArg) -> EvalComplexMode {
    match mode {
        ComplexModeArg::Auto => EvalComplexMode::Auto,
        ComplexModeArg::On => EvalComplexMode::On,
        ComplexModeArg::Off => EvalComplexMode::Off,
    }
}

fn const_fold_mode(mode: ConstFoldArg) -> EvalConstFoldMode {
    match mode {
        ConstFoldArg::Off => EvalConstFoldMode::Off,
        ConstFoldArg::Safe => EvalConstFoldMode::Safe,
    }
}

fn assume_scope(scope: AssumeScopeArg) -> EvalAssumeScope {
    match scope {
        AssumeScopeArg::Real => EvalAssumeScope::Real,
        AssumeScopeArg::Wildcard => EvalAssumeScope::Wildcard,
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
