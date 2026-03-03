//! Session-backed eval-json command orchestration.

use std::path::Path;
use std::time::Instant;

use cas_api_models::{
    parse_eval_json_special_command, wire::build_eval_wire_reply, EvalJsonLimitApproach,
    EvalJsonOutput, EvalJsonOutputBuild, EvalJsonSessionRunConfig, EvalJsonSpecialCommand,
    ExprStatsJson, RequiredConditionJson, SolveStepJson, SolveSubStepJson, StepJson, TimingsJson,
    WarningJson,
};
use cas_ast::{Context, Expr, ExprId, RelOp, SolutionSet};
use cas_formatter::{DisplayExpr, LaTeXExpr};

/// Session-backed config for `eval-json` command orchestration.
pub type EvalJsonCommandConfig<'a> = cas_api_models::EvalJsonSessionRunConfig<'a>;

#[allow(clippy::too_many_arguments)]
fn apply_eval_json_options(
    opts: &mut cas_solver::EvalOptions,
    context: &str,
    branch: &str,
    complex: &str,
    autoexpand: &str,
    steps: &str,
    domain: &str,
    value_domain: &str,
    inv_trig: &str,
    complex_branch: &str,
    assume_scope: &str,
) {
    opts.shared.context_mode = match context {
        "standard" => cas_solver::ContextMode::Standard,
        "solve" => cas_solver::ContextMode::Solve,
        "integrate" => cas_solver::ContextMode::IntegratePrep,
        _ => cas_solver::ContextMode::Auto,
    };

    opts.branch_mode = match branch {
        "principal" => cas_solver::BranchMode::PrincipalBranch,
        _ => cas_solver::BranchMode::Strict,
    };

    opts.complex_mode = match complex {
        "on" => cas_solver::ComplexMode::On,
        "off" => cas_solver::ComplexMode::Off,
        _ => cas_solver::ComplexMode::Auto,
    };

    opts.steps_mode = match steps {
        "on" => cas_solver::StepsMode::On,
        "compact" => cas_solver::StepsMode::Compact,
        _ => cas_solver::StepsMode::Off,
    };

    opts.shared.expand_policy = match autoexpand {
        "auto" => cas_solver::ExpandPolicy::Auto,
        _ => cas_solver::ExpandPolicy::Off,
    };

    opts.shared.semantics.domain_mode = match domain {
        "strict" => cas_solver::DomainMode::Strict,
        "assume" => cas_solver::DomainMode::Assume,
        _ => cas_solver::DomainMode::Generic,
    };

    opts.shared.semantics.inv_trig = match inv_trig {
        "principal" => cas_solver::InverseTrigPolicy::PrincipalValue,
        _ => cas_solver::InverseTrigPolicy::Strict,
    };

    opts.shared.semantics.value_domain = match value_domain {
        "complex" => cas_solver::ValueDomain::ComplexEnabled,
        _ => cas_solver::ValueDomain::RealOnly,
    };

    let _ = complex_branch;
    opts.shared.semantics.branch = cas_solver::BranchPolicy::Principal;

    opts.shared.semantics.assume_scope = match assume_scope {
        "wildcard" => cas_solver::AssumeScope::Wildcard,
        _ => cas_solver::AssumeScope::Real,
    };
}

fn parse_statement_or_session_ref(
    ctx: &mut cas_ast::Context,
    input: &str,
) -> Result<cas_parser::Statement, String> {
    if input.starts_with('#') && input[1..].chars().all(char::is_numeric) {
        Ok(cas_parser::Statement::Expression(ctx.var(input)))
    } else {
        cas_parser::parse_statement(input, ctx).map_err(|e| e.to_string())
    }
}

fn detect_solve_variable_eval_json(ctx: &Context, lhs: ExprId, rhs: ExprId) -> String {
    let mut variables = std::collections::HashSet::new();

    fn collect_vars(ctx: &Context, expr: ExprId, vars: &mut std::collections::HashSet<String>) {
        match ctx.get(expr) {
            Expr::Variable(sym_id) => {
                let name = ctx.sym_name(*sym_id);
                if !name.starts_with('#') && name != "e" && name != "i" && name != "pi" {
                    vars.insert(name.to_string());
                }
            }
            Expr::Add(a, b)
            | Expr::Sub(a, b)
            | Expr::Mul(a, b)
            | Expr::Div(a, b)
            | Expr::Pow(a, b) => {
                collect_vars(ctx, *a, vars);
                collect_vars(ctx, *b, vars);
            }
            Expr::Neg(e) => collect_vars(ctx, *e, vars),
            Expr::Function(_, args) => {
                for arg in args {
                    collect_vars(ctx, *arg, vars);
                }
            }
            _ => {}
        }
    }

    collect_vars(ctx, lhs, &mut variables);
    collect_vars(ctx, rhs, &mut variables);

    if variables.contains("x") {
        return "x".to_string();
    }

    for preferred in ["y", "z", "t", "n", "a", "b", "c"] {
        if variables.contains(preferred) {
            return preferred.to_string();
        }
    }

    variables
        .into_iter()
        .min()
        .unwrap_or_else(|| "x".to_string())
}

fn map_limit_approach(approach: EvalJsonLimitApproach) -> cas_solver::Approach {
    match approach {
        EvalJsonLimitApproach::PosInfinity => cas_solver::Approach::PosInfinity,
        EvalJsonLimitApproach::NegInfinity => cas_solver::Approach::NegInfinity,
    }
}

fn parse_solve_input_as_equation_expr(
    ctx: &mut cas_ast::Context,
    input: &str,
) -> Result<ExprId, String> {
    let stmt = parse_statement_or_session_ref(ctx, input)?;
    let parsed = match stmt {
        cas_parser::Statement::Equation(eq) => ctx.call("Equal", vec![eq.lhs, eq.rhs]),
        cas_parser::Statement::Expression(expr) => {
            let zero = ctx.num(0);
            ctx.call("Equal", vec![expr, zero])
        }
    };
    Ok(parsed)
}

fn build_eval_request_for_input(
    raw_input: &str,
    ctx: &mut cas_ast::Context,
    auto_store: bool,
) -> Result<cas_solver::EvalRequest, String> {
    if let Some(command) = parse_eval_json_special_command(raw_input) {
        return match command {
            EvalJsonSpecialCommand::Solve { equation, var } => {
                let parsed = parse_solve_input_as_equation_expr(ctx, &equation)
                    .map_err(|e| format!("Parse error in solve equation: {e}"))?;

                Ok(cas_solver::EvalRequest {
                    raw_input: raw_input.to_string(),
                    parsed,
                    action: cas_solver::EvalAction::Solve { var },
                    auto_store,
                })
            }
            EvalJsonSpecialCommand::Limit {
                expr,
                var,
                approach,
            } => {
                let parsed = cas_parser::parse(&expr, ctx)
                    .map_err(|e| format!("Parse error in limit expression: {e}"))?;

                Ok(cas_solver::EvalRequest {
                    raw_input: raw_input.to_string(),
                    parsed,
                    action: cas_solver::EvalAction::Limit {
                        var,
                        approach: map_limit_approach(approach),
                    },
                    auto_store,
                })
            }
        };
    }

    let stmt =
        parse_statement_or_session_ref(ctx, raw_input).map_err(|e| format!("Parse error: {e}"))?;
    match stmt {
        cas_parser::Statement::Equation(eq) => {
            let parsed = ctx.call("Equal", vec![eq.lhs, eq.rhs]);
            let var = detect_solve_variable_eval_json(ctx, eq.lhs, eq.rhs);
            Ok(cas_solver::EvalRequest {
                raw_input: raw_input.to_string(),
                parsed,
                action: cas_solver::EvalAction::Solve { var },
                auto_store,
            })
        }
        cas_parser::Statement::Expression(parsed) => Ok(cas_solver::EvalRequest {
            raw_input: raw_input.to_string(),
            parsed,
            action: cas_solver::EvalAction::Simplify,
            auto_store,
        }),
    }
}

fn format_eval_input_latex(ctx: &cas_ast::Context, parsed: ExprId) -> String {
    if let Some((lhs, rhs)) = cas_ast::eq::unwrap_eq(ctx, parsed) {
        let lhs_latex = LaTeXExpr {
            context: ctx,
            id: lhs,
        }
        .to_latex();
        let rhs_latex = LaTeXExpr {
            context: ctx,
            id: rhs,
        }
        .to_latex();
        format!("{lhs_latex} = {rhs_latex}")
    } else {
        LaTeXExpr {
            context: ctx,
            id: parsed,
        }
        .to_latex()
    }
}

fn relop_to_latex(op: &RelOp) -> String {
    match op {
        RelOp::Eq => "=".to_string(),
        RelOp::Lt => "<".to_string(),
        RelOp::Leq => r"\leq".to_string(),
        RelOp::Gt => ">".to_string(),
        RelOp::Geq => r"\geq".to_string(),
        RelOp::Neq => r"\neq".to_string(),
    }
}

fn collect_solve_steps_eval_json(
    solve_steps: &[cas_solver::SolveStep],
    ctx: &Context,
    steps_mode: &str,
) -> Vec<SolveStepJson> {
    if steps_mode != "on" {
        return vec![];
    }

    let filtered: Vec<_> = solve_steps
        .iter()
        .filter(|step| step.importance >= cas_solver::ImportanceLevel::Medium)
        .collect();

    if filtered.is_empty() {
        return vec![];
    }

    filtered
        .iter()
        .enumerate()
        .map(|(i, step)| {
            let lhs_str = format!(
                "{}",
                DisplayExpr {
                    context: ctx,
                    id: step.equation_after.lhs
                }
            );
            let rhs_str = format!(
                "{}",
                DisplayExpr {
                    context: ctx,
                    id: step.equation_after.rhs
                }
            );
            let relop_str = format!("{}", step.equation_after.op);
            let equation_str = format!("{lhs_str} {relop_str} {rhs_str}");

            let lhs_latex = LaTeXExpr {
                context: ctx,
                id: step.equation_after.lhs,
            }
            .to_latex();
            let rhs_latex = LaTeXExpr {
                context: ctx,
                id: step.equation_after.rhs,
            }
            .to_latex();
            let relop_latex = relop_to_latex(&step.equation_after.op);

            let substeps: Vec<SolveSubStepJson> = step
                .substeps
                .iter()
                .enumerate()
                .map(|(j, ss)| {
                    let ss_lhs_str = format!(
                        "{}",
                        DisplayExpr {
                            context: ctx,
                            id: ss.equation_after.lhs
                        }
                    );
                    let ss_rhs_str = format!(
                        "{}",
                        DisplayExpr {
                            context: ctx,
                            id: ss.equation_after.rhs
                        }
                    );
                    let ss_relop_str = format!("{}", ss.equation_after.op);
                    let ss_equation_str = format!("{ss_lhs_str} {ss_relop_str} {ss_rhs_str}");

                    let ss_lhs_latex = LaTeXExpr {
                        context: ctx,
                        id: ss.equation_after.lhs,
                    }
                    .to_latex();
                    let ss_rhs_latex = LaTeXExpr {
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

fn format_solution_set_eval_json(ctx: &Context, solution_set: &SolutionSet) -> String {
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
                            DisplayExpr {
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
                if cas_solver::is_pure_residual_otherwise(case) {
                    continue;
                }
                let cond_str = cas_formatter::condition_set_to_display(&case.when, ctx);
                let inner_str = format_solution_set_eval_json(ctx, &case.then.solutions);
                if case.when.is_empty() {
                    parts.push(format!("{inner_str} otherwise"));
                } else {
                    parts.push(format!("{inner_str} if {cond_str}"));
                }
            }

            if parts.is_empty() && !cases.is_empty() {
                for case in cases {
                    if !case.when.is_empty() {
                        let inner_str = format_solution_set_eval_json(ctx, &case.then.solutions);
                        let cond_str = cas_formatter::condition_set_to_display(&case.when, ctx);
                        return format!("{inner_str} if {cond_str}");
                    }
                }
            }
            parts.join("; ")
        }
        SolutionSet::Continuous(interval) => {
            format!(
                "[{}, {}]",
                DisplayExpr {
                    context: ctx,
                    id: interval.min
                },
                DisplayExpr {
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
                        DisplayExpr {
                            context: ctx,
                            id: int.min
                        },
                        DisplayExpr {
                            context: ctx,
                            id: int.max
                        }
                    )
                })
                .collect();
            parts.join(" U ")
        }
        SolutionSet::Residual(expr) => {
            let expr_str = format!(
                "{}",
                DisplayExpr {
                    context: ctx,
                    id: *expr
                }
            );
            format!("Solve: {expr_str} = 0")
        }
    }
}

fn solution_set_to_latex_eval_json(ctx: &Context, solution_set: &SolutionSet) -> String {
    match solution_set {
        SolutionSet::Empty => r"\emptyset".to_string(),
        SolutionSet::AllReals => r"\mathbb{R}".to_string(),
        SolutionSet::Discrete(exprs) => {
            if exprs.is_empty() {
                r"\emptyset".to_string()
            } else {
                let solutions: Vec<String> = exprs
                    .iter()
                    .map(|e| {
                        LaTeXExpr {
                            context: ctx,
                            id: *e,
                        }
                        .to_latex()
                    })
                    .collect();
                format!(r"\left\{{ {} \right\}}", solutions.join(", "))
            }
        }
        SolutionSet::Conditional(cases) => {
            let mut latex_parts = Vec::new();
            for case in cases {
                let cond_latex = cas_formatter::condition_set_to_latex(&case.when, ctx);
                let inner_latex = solution_set_to_latex_eval_json(ctx, &case.then.solutions);
                if case.when.is_empty() {
                    latex_parts.push(format!(r"{} & \text{{otherwise}}", inner_latex));
                } else {
                    latex_parts.push(format!(r"{} & \text{{if }} {}", inner_latex, cond_latex));
                }
            }
            if latex_parts.len() == 1 {
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
            let min_latex = LaTeXExpr {
                context: ctx,
                id: interval.min,
            }
            .to_latex();
            let max_latex = LaTeXExpr {
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
                    let min = LaTeXExpr {
                        context: ctx,
                        id: int.min,
                    }
                    .to_latex();
                    let max = LaTeXExpr {
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
            let expr_latex = LaTeXExpr {
                context: ctx,
                id: *expr,
            }
            .to_latex();
            format!(r"\text{{Solve: }} {} = 0", expr_latex)
        }
    }
}

fn collect_warnings_eval_json(domain_warnings: &[cas_solver::DomainWarning]) -> Vec<WarningJson> {
    domain_warnings
        .iter()
        .map(|w| WarningJson {
            rule: w.rule_name.clone(),
            assumption: w.message.clone(),
        })
        .collect()
}

fn collect_required_conditions_eval_json(
    required_conditions: &[cas_solver::ImplicitCondition],
    ctx: &Context,
) -> Vec<RequiredConditionJson> {
    required_conditions
        .iter()
        .map(|cond| {
            let (kind, expr_id) = match cond {
                cas_solver::ImplicitCondition::NonNegative(e) => ("NonNegative", *e),
                cas_solver::ImplicitCondition::Positive(e) => ("Positive", *e),
                cas_solver::ImplicitCondition::NonZero(e) => ("NonZero", *e),
            };
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

fn collect_required_display_eval_json(
    required_conditions: &[cas_solver::ImplicitCondition],
    ctx: &Context,
) -> Vec<String> {
    required_conditions
        .iter()
        .map(|cond| cond.display(ctx))
        .collect()
}

fn build_eval_wire_value(
    warnings: &[WarningJson],
    required_display: &[String],
    result: &str,
    result_latex: Option<&str>,
    steps_count: usize,
    steps_mode: &str,
) -> Option<serde_json::Value> {
    serde_json::to_value(build_eval_wire_reply(
        warnings,
        required_display,
        result,
        result_latex,
        steps_count,
        steps_mode,
    ))
    .ok()
}

fn format_expr_limited_eval_json(
    ctx: &Context,
    expr: ExprId,
    max_chars: usize,
) -> (String, bool, usize) {
    if let Some(poly_str) = cas_solver::try_render_poly_result(ctx, expr) {
        let len = poly_str.chars().count();
        if len <= max_chars {
            return (poly_str, false, len);
        }
        let truncated: String = poly_str.chars().take(max_chars).collect();
        return (format!("{truncated} … <truncated>"), true, len);
    }

    let full = format!(
        "{}",
        DisplayExpr {
            context: ctx,
            id: expr
        }
    );
    let len = full.chars().count();

    if len <= max_chars {
        return (full, false, len);
    }

    let truncated: String = full.chars().take(max_chars).collect();
    (format!("{truncated} … <truncated>"), true, len)
}

fn count_add_terms(ctx: &Context, expr: ExprId) -> Option<usize> {
    let inner_expr = match ctx.get(expr) {
        Expr::Function(name, args)
            if ctx.is_builtin(*name, cas_ast::BuiltinFn::Hold) && args.len() == 1 =>
        {
            args[0]
        }
        _ => expr,
    };

    if !matches!(ctx.get(inner_expr), Expr::Add(_, _) | Expr::Sub(_, _)) {
        return None;
    }

    let mut count = 0usize;
    let mut stack = vec![inner_expr];

    while let Some(current) = stack.pop() {
        match ctx.get(current) {
            Expr::Add(l, r) | Expr::Sub(l, r) => {
                stack.push(*l);
                stack.push(*r);
            }
            _ => {
                count += 1;
            }
        }
    }

    if count > 1 {
        Some(count)
    } else {
        None
    }
}

fn expr_stats_eval_json(ctx: &Context, expr: ExprId) -> ExprStatsJson {
    let (node_count, depth) = cas_ast::traversal::count_nodes_and_max_depth(ctx, expr);
    let term_count = cas_solver::try_get_poly_result_term_count(ctx, expr)
        .or_else(|| count_add_terms(ctx, expr));

    ExprStatsJson {
        node_count,
        depth,
        term_count,
    }
}

fn hash_expr_recursive<H: std::hash::Hasher>(ctx: &Context, expr: ExprId, hasher: &mut H) {
    use std::hash::Hash;

    match ctx.get(expr) {
        Expr::Number(n) => {
            0u8.hash(hasher);
            n.numer().to_string().hash(hasher);
            n.denom().to_string().hash(hasher);
        }
        Expr::Variable(name) => {
            1u8.hash(hasher);
            name.hash(hasher);
        }
        Expr::Constant(c) => {
            2u8.hash(hasher);
            format!("{:?}", c).hash(hasher);
        }
        Expr::SessionRef(id) => {
            11u8.hash(hasher);
            id.hash(hasher);
        }
        Expr::Hold(inner) => {
            12u8.hash(hasher);
            hash_expr_recursive(ctx, *inner, hasher);
        }
        Expr::Add(l, r) => {
            3u8.hash(hasher);
            hash_expr_recursive(ctx, *l, hasher);
            hash_expr_recursive(ctx, *r, hasher);
        }
        Expr::Sub(l, r) => {
            4u8.hash(hasher);
            hash_expr_recursive(ctx, *l, hasher);
            hash_expr_recursive(ctx, *r, hasher);
        }
        Expr::Mul(l, r) => {
            5u8.hash(hasher);
            hash_expr_recursive(ctx, *l, hasher);
            hash_expr_recursive(ctx, *r, hasher);
        }
        Expr::Div(l, r) => {
            6u8.hash(hasher);
            hash_expr_recursive(ctx, *l, hasher);
            hash_expr_recursive(ctx, *r, hasher);
        }
        Expr::Pow(l, r) => {
            7u8.hash(hasher);
            hash_expr_recursive(ctx, *l, hasher);
            hash_expr_recursive(ctx, *r, hasher);
        }
        Expr::Neg(inner) => {
            8u8.hash(hasher);
            hash_expr_recursive(ctx, *inner, hasher);
        }
        Expr::Function(name, args) => {
            9u8.hash(hasher);
            name.hash(hasher);
            for arg in args {
                hash_expr_recursive(ctx, *arg, hasher);
            }
        }
        Expr::Matrix { rows, cols, data } => {
            10u8.hash(hasher);
            rows.hash(hasher);
            cols.hash(hasher);
            for elem in data {
                hash_expr_recursive(ctx, *elem, hasher);
            }
        }
    }
}

fn expr_hash_eval_json(ctx: &Context, expr: ExprId) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hasher;

    let mut hasher = DefaultHasher::new();
    hash_expr_recursive(ctx, expr, &mut hasher);
    format!("{:016x}", hasher.finish())
}

struct EvalJsonFinalizeInput<'a> {
    result: &'a cas_solver::EvalResult,
    ctx: &'a Context,
    max_chars: usize,
    input: &'a str,
    input_latex: Option<String>,
    steps_mode: &'a str,
    steps: Vec<StepJson>,
    solve_steps: Vec<SolveStepJson>,
    warnings: Vec<WarningJson>,
    required_conditions: Vec<RequiredConditionJson>,
    required_display: Vec<String>,
    raw_steps_count: usize,
    raw_solve_steps_count: usize,
    budget_preset: &'a str,
    strict: bool,
    domain: &'a str,
    timings_us: TimingsJson,
    context_mode: &'a str,
    branch_mode: &'a str,
    expand_policy: &'a str,
    complex_mode: &'a str,
    const_fold: &'a str,
    value_domain: &'a str,
    complex_branch: &'a str,
    inv_trig: &'a str,
    assume_scope: &'a str,
}

fn finalize_eval_json_output(input: EvalJsonFinalizeInput<'_>) -> Result<EvalJsonOutput, String> {
    let EvalJsonFinalizeInput {
        result,
        ctx,
        max_chars,
        input,
        input_latex,
        steps_mode,
        steps,
        solve_steps,
        warnings,
        required_conditions,
        required_display,
        raw_steps_count,
        raw_solve_steps_count,
        budget_preset,
        strict,
        domain,
        timings_us,
        context_mode,
        branch_mode,
        expand_policy,
        complex_mode,
        const_fold,
        value_domain,
        complex_branch,
        inv_trig,
        assume_scope,
    } = input;

    match result {
        cas_solver::EvalResult::SolutionSet(solution_set) => {
            let result_str = format_solution_set_eval_json(ctx, solution_set);
            let result_latex = solution_set_to_latex_eval_json(ctx, solution_set);
            let wire = build_eval_wire_value(
                &warnings,
                &required_display,
                &result_str,
                Some(&result_latex),
                raw_steps_count + raw_solve_steps_count,
                steps_mode,
            );

            Ok(EvalJsonOutput::from_build(EvalJsonOutputBuild {
                input,
                input_latex,
                result_chars: result_str.len(),
                result: result_str,
                result_truncated: false,
                result_latex: Some(result_latex),
                steps_mode,
                steps_count: raw_steps_count + raw_solve_steps_count,
                steps,
                solve_steps,
                warnings,
                required_conditions,
                required_display,
                budget_preset,
                strict,
                domain,
                stats: Default::default(),
                hash: None,
                timings_us,
                context_mode,
                branch_mode,
                expand_policy,
                complex_mode,
                const_fold,
                value_domain,
                complex_branch,
                inv_trig,
                assume_scope,
                wire,
            }))
        }
        cas_solver::EvalResult::Bool(b) => {
            let result_str = b.to_string();
            let wire = build_eval_wire_value(
                &warnings,
                &required_display,
                &result_str,
                None,
                raw_steps_count,
                steps_mode,
            );

            Ok(EvalJsonOutput::from_build(EvalJsonOutputBuild {
                input,
                input_latex,
                result_chars: result_str.len(),
                result: result_str,
                result_truncated: false,
                result_latex: None,
                steps_mode,
                steps_count: raw_steps_count,
                steps,
                solve_steps,
                warnings,
                required_conditions,
                required_display,
                budget_preset,
                strict,
                domain,
                stats: Default::default(),
                hash: None,
                timings_us,
                context_mode,
                branch_mode,
                expand_policy,
                complex_mode,
                const_fold,
                value_domain,
                complex_branch,
                inv_trig,
                assume_scope,
                wire,
            }))
        }
        cas_solver::EvalResult::Expr(e) => finalize_expr_like_eval_json_output(
            ctx,
            *e,
            max_chars,
            input,
            input_latex,
            steps_mode,
            steps,
            solve_steps,
            warnings,
            required_conditions,
            required_display,
            raw_steps_count,
            budget_preset,
            strict,
            domain,
            timings_us,
            context_mode,
            branch_mode,
            expand_policy,
            complex_mode,
            const_fold,
            value_domain,
            complex_branch,
            inv_trig,
            assume_scope,
        ),
        cas_solver::EvalResult::Set(v) if !v.is_empty() => finalize_expr_like_eval_json_output(
            ctx,
            v[0],
            max_chars,
            input,
            input_latex,
            steps_mode,
            steps,
            solve_steps,
            warnings,
            required_conditions,
            required_display,
            raw_steps_count,
            budget_preset,
            strict,
            domain,
            timings_us,
            context_mode,
            branch_mode,
            expand_policy,
            complex_mode,
            const_fold,
            value_domain,
            complex_branch,
            inv_trig,
            assume_scope,
        ),
        _ => Err("No result expression".to_string()),
    }
}

#[allow(clippy::too_many_arguments)]
fn finalize_expr_like_eval_json_output(
    ctx: &Context,
    result_expr: ExprId,
    max_chars: usize,
    input: &str,
    input_latex: Option<String>,
    steps_mode: &str,
    steps: Vec<StepJson>,
    solve_steps: Vec<SolveStepJson>,
    warnings: Vec<WarningJson>,
    required_conditions: Vec<RequiredConditionJson>,
    required_display: Vec<String>,
    raw_steps_count: usize,
    budget_preset: &str,
    strict: bool,
    domain: &str,
    timings_us: TimingsJson,
    context_mode: &str,
    branch_mode: &str,
    expand_policy: &str,
    complex_mode: &str,
    const_fold: &str,
    value_domain: &str,
    complex_branch: &str,
    inv_trig: &str,
    assume_scope: &str,
) -> Result<EvalJsonOutput, String> {
    let (result_str, truncated, char_count) =
        format_expr_limited_eval_json(ctx, result_expr, max_chars);
    let stats = expr_stats_eval_json(ctx, result_expr);
    let hash = if truncated {
        Some(expr_hash_eval_json(ctx, result_expr))
    } else {
        None
    };

    let result_latex = if !truncated {
        Some(
            LaTeXExpr {
                context: ctx,
                id: result_expr,
            }
            .to_latex(),
        )
    } else {
        None
    };

    let wire = build_eval_wire_value(
        &warnings,
        &required_display,
        &result_str,
        result_latex.as_deref(),
        raw_steps_count,
        steps_mode,
    );

    Ok(EvalJsonOutput::from_build(EvalJsonOutputBuild {
        input,
        input_latex,
        result_chars: char_count,
        result: result_str,
        result_truncated: truncated,
        result_latex,
        steps_mode,
        steps_count: raw_steps_count,
        steps,
        solve_steps,
        warnings,
        required_conditions,
        required_display,
        budget_preset,
        strict,
        domain,
        stats,
        hash,
        timings_us,
        context_mode,
        branch_mode,
        expand_policy,
        complex_mode,
        const_fold,
        value_domain,
        complex_branch,
        inv_trig,
        assume_scope,
        wire,
    }))
}

fn evaluate_eval_json_with_session<S, F>(
    engine: &mut cas_solver::Engine,
    session: &mut S,
    config: EvalJsonSessionRunConfig<'_>,
    collect_steps: F,
) -> Result<EvalJsonOutput, String>
where
    S: cas_solver::EvalSession<
        Options = cas_solver::EvalOptions,
        Diagnostics = cas_solver::Diagnostics,
    >,
    S::Store: cas_solver::EvalStore<
        DomainMode = cas_solver::DomainMode,
        RequiredItem = cas_solver::RequiredItem,
        Step = cas_solver::Step,
        Diagnostics = cas_solver::Diagnostics,
    >,
    F: Fn(&[cas_solver::Step], &cas_ast::Context, &str) -> Vec<StepJson>,
{
    let total_start = Instant::now();

    apply_eval_json_options(
        session.options_mut(),
        config.context_mode,
        config.branch_mode,
        config.complex_mode,
        config.expand_policy,
        config.steps_mode,
        config.domain,
        config.value_domain,
        config.inv_trig,
        config.complex_branch,
        config.assume_scope,
    );

    let parse_start = Instant::now();
    let req = build_eval_request_for_input(
        config.expr,
        &mut engine.simplifier.context,
        config.auto_store,
    )?;
    let parsed_input = req.parsed;
    let parse_us = parse_start.elapsed().as_micros() as u64;

    let simplify_start = Instant::now();
    let output = engine.eval(session, req).map_err(|e| e.to_string())?;
    let simplify_us = simplify_start.elapsed().as_micros() as u64;
    let output_view = cas_solver::eval_output_view(&output);

    let input_latex = Some(format_eval_input_latex(
        &engine.simplifier.context,
        parsed_input,
    ));
    let steps_raw = output_view.steps.as_slice();
    let solve_steps_raw = output_view.solve_steps.as_slice();
    let steps = collect_steps(steps_raw, &engine.simplifier.context, config.steps_mode);
    let solve_steps = collect_solve_steps_eval_json(
        solve_steps_raw,
        &engine.simplifier.context,
        config.steps_mode,
    );
    let warnings = collect_warnings_eval_json(&output_view.domain_warnings);
    let required_conditions_raw = output_view.required_conditions.as_slice();
    let required_conditions =
        collect_required_conditions_eval_json(required_conditions_raw, &engine.simplifier.context);
    let required_display =
        collect_required_display_eval_json(required_conditions_raw, &engine.simplifier.context);
    let timings_us = TimingsJson {
        parse_us,
        simplify_us,
        total_us: total_start.elapsed().as_micros() as u64,
    };

    finalize_eval_json_output(EvalJsonFinalizeInput {
        result: &output_view.result,
        ctx: &engine.simplifier.context,
        max_chars: config.max_chars,
        input: config.expr,
        input_latex,
        steps_mode: config.steps_mode,
        steps,
        solve_steps,
        warnings,
        required_conditions,
        required_display,
        raw_steps_count: steps_raw.len(),
        raw_solve_steps_count: solve_steps_raw.len(),
        budget_preset: config.budget_preset,
        strict: config.strict,
        domain: config.domain,
        timings_us,
        context_mode: config.context_mode,
        branch_mode: config.branch_mode,
        expand_policy: config.expand_policy,
        complex_mode: config.complex_mode,
        const_fold: config.const_fold,
        value_domain: config.value_domain,
        complex_branch: config.complex_branch,
        inv_trig: config.inv_trig,
        assume_scope: config.assume_scope,
    })
}

/// Evaluate `eval-json` using optional persisted session state.
///
/// Keeps CLI/frontends thin by centralizing session load/run/save orchestration.
pub fn evaluate_eval_json_command_with_session<F>(
    session_path: Option<&Path>,
    config: EvalJsonCommandConfig<'_>,
    collect_steps: F,
) -> (
    Result<cas_api_models::EvalJsonOutput, String>,
    Option<String>,
    Option<String>,
)
where
    F: Fn(&[cas_solver::Step], &cas_ast::Context, &str) -> Vec<cas_api_models::StepJson>,
{
    crate::run_with_domain_session(session_path, config.domain, |engine, state| {
        evaluate_eval_json_with_session(engine, state, config, |steps, ctx, mode| {
            collect_steps(steps, ctx, mode)
        })
    })
}

/// Evaluate `eval-json` and always return a pretty JSON string.
///
/// Successful runs return canonical JSON payload. Errors are normalized into
/// canonical JSON error output.
pub fn evaluate_eval_json_command_pretty_with_session<F>(
    session_path: Option<&Path>,
    config: EvalJsonCommandConfig<'_>,
    collect_steps: F,
) -> String
where
    F: Fn(&[cas_solver::Step], &cas_ast::Context, &str) -> Vec<cas_api_models::StepJson>,
{
    let input = config.expr.to_string();
    let (output, _, _) =
        evaluate_eval_json_command_with_session(session_path, config, collect_steps);
    match output {
        Ok(payload) => payload.to_json_pretty(),
        Err(error) => cas_api_models::ErrorJsonOutput::from_eval_error_message(&error, &input)
            .to_json_pretty(),
    }
}

#[cfg(test)]
mod tests {
    use super::{evaluate_eval_json_with_session, EvalJsonSessionRunConfig};

    #[test]
    fn evaluate_eval_json_with_session_runs() {
        let mut engine = cas_solver::Engine::new();
        let mut session = crate::SessionState::new();
        let out = evaluate_eval_json_with_session(
            &mut engine,
            &mut session,
            EvalJsonSessionRunConfig {
                expr: "x + x",
                auto_store: false,
                max_chars: 2000,
                steps_mode: "off",
                budget_preset: "standard",
                strict: false,
                domain: "generic",
                context_mode: "auto",
                branch_mode: "strict",
                expand_policy: "off",
                complex_mode: "auto",
                const_fold: "off",
                value_domain: "real",
                complex_branch: "principal",
                inv_trig: "strict",
                assume_scope: "real",
            },
            |_steps, _context, _steps_mode| Vec::new(),
        )
        .expect("eval-json");

        assert!(out.ok);
        assert!(out.result.contains("2 * x"));
    }
}
