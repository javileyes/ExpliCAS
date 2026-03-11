//! eval subcommand dispatcher.
//!
//! Handles stdin/arg expression sourcing and routes to wire/text handlers.

use crate::{
    AssumeScopeArg, BranchArg, BudgetPreset, ConstFoldArg, DomainArg, EvalArgs, InvTrigArg,
    OutputFormat, ValueDomainArg,
};

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
            run_wire(&expr, &args);
        }
        OutputFormat::Text => {
            let text_args = EvalArgs { expr, ..args };
            crate::commands::eval_text::run(&text_args);
        }
    }
}

fn run_wire(expr: &str, args: &EvalArgs) {
    let output = cas_session::evaluate_eval_command_pretty_with_session(
        args.session.as_deref(),
        eval_command_config(expr, args),
        |steps, events, context, steps_mode| {
            cas_didactic::collect_step_payloads_with_events(steps, events, context, steps_mode)
        },
    );
    println!("{}", output);
}

fn eval_command_config<'a>(
    expr: &'a str,
    args: &'a EvalArgs,
) -> cas_session::EvalCommandConfig<'a> {
    cas_session::EvalCommandConfig {
        expr,
        auto_store: args.session.is_some(),
        max_chars: args.max_chars,
        steps_mode: &args.steps,
        budget_preset: budget_preset(args.budget),
        strict: args.strict,
        domain: domain_mode(args.domain),
        context_mode: &args.context,
        branch_mode: &args.branch,
        expand_policy: &args.autoexpand,
        complex_mode: &args.complex,
        const_fold: const_fold_mode(args.const_fold),
        value_domain: value_domain(args.value_domain),
        complex_branch: branch_mode_arg(args.complex_branch),
        inv_trig: inv_trig_mode(args.inv_trig),
        assume_scope: assume_scope(args.assume_scope),
    }
}

fn budget_preset(preset: BudgetPreset) -> &'static str {
    match preset {
        BudgetPreset::Small => "small",
        BudgetPreset::Standard | BudgetPreset::Cli => "standard",
        BudgetPreset::Unlimited => "unlimited",
    }
}

fn domain_mode(domain: DomainArg) -> &'static str {
    match domain {
        DomainArg::Strict => "strict",
        DomainArg::Generic => "generic",
        DomainArg::Assume => "assume",
    }
}

fn value_domain(vd: ValueDomainArg) -> &'static str {
    match vd {
        ValueDomainArg::Real => "real",
        ValueDomainArg::Complex => "complex",
    }
}

fn inv_trig_mode(mode: InvTrigArg) -> &'static str {
    match mode {
        InvTrigArg::Strict => "strict",
        InvTrigArg::Principal => "principal",
    }
}

fn branch_mode_arg(mode: BranchArg) -> &'static str {
    match mode {
        BranchArg::Principal => "principal",
    }
}

fn const_fold_mode(mode: ConstFoldArg) -> &'static str {
    match mode {
        ConstFoldArg::Off => "off",
        ConstFoldArg::Safe => "safe",
    }
}

fn assume_scope(scope: AssumeScopeArg) -> &'static str {
    match scope {
        AssumeScopeArg::Real => "real",
        AssumeScopeArg::Wildcard => "wildcard",
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
