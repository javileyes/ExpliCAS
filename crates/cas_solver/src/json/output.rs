use cas_api_models::{
    wire::build_eval_wire_reply, EvalJsonOutput, ExprStatsJson, RequiredConditionJson,
    SolveStepJson, StepJson, TimingsJson, WarningJson, SCHEMA_VERSION,
};
use cas_ast::{Context, Expr, ExprId};
use cas_formatter::{DisplayExpr, LaTeXExpr};

/// Inputs required to build a complete `EvalJsonOutput`.
pub struct EvalJsonOutputBuild<'a> {
    pub input: &'a str,
    pub input_latex: Option<String>,
    pub result: String,
    pub result_truncated: bool,
    pub result_chars: usize,
    pub result_latex: Option<String>,
    pub steps_mode: &'a str,
    pub steps_count: usize,
    pub steps: Vec<StepJson>,
    pub solve_steps: Vec<SolveStepJson>,
    pub warnings: Vec<WarningJson>,
    pub required_conditions: Vec<RequiredConditionJson>,
    pub required_display: Vec<String>,
    pub budget_preset: &'a str,
    pub strict: bool,
    pub domain: &'a str,
    pub stats: ExprStatsJson,
    pub hash: Option<String>,
    pub timings_us: TimingsJson,
    pub context_mode: &'a str,
    pub branch_mode: &'a str,
    pub expand_policy: &'a str,
    pub complex_mode: &'a str,
    pub const_fold: &'a str,
    pub value_domain: &'a str,
    pub complex_branch: &'a str,
    pub inv_trig: &'a str,
    pub assume_scope: &'a str,
    pub wire: Option<serde_json::Value>,
}

/// Inputs used to finalize eval-json output from an already computed `EvalResult`.
pub struct EvalJsonFinalizeInput<'a> {
    pub result: &'a crate::EvalResult,
    pub ctx: &'a Context,
    pub max_chars: usize,
    pub input: &'a str,
    pub input_latex: Option<String>,
    pub steps_mode: &'a str,
    pub steps: Vec<StepJson>,
    pub solve_steps: Vec<SolveStepJson>,
    pub warnings: Vec<WarningJson>,
    pub required_conditions: Vec<RequiredConditionJson>,
    pub required_display: Vec<String>,
    pub raw_steps_count: usize,
    pub raw_solve_steps_count: usize,
    pub budget_preset: &'a str,
    pub strict: bool,
    pub domain: &'a str,
    pub timings_us: TimingsJson,
    pub context_mode: &'a str,
    pub branch_mode: &'a str,
    pub expand_policy: &'a str,
    pub complex_mode: &'a str,
    pub const_fold: &'a str,
    pub value_domain: &'a str,
    pub complex_branch: &'a str,
    pub inv_trig: &'a str,
    pub assume_scope: &'a str,
}

/// Inputs used to finalize eval-json output using engine context.
pub struct EvalJsonFinalizeWithEngineInput<'a> {
    pub result: &'a crate::EvalResult,
    pub max_chars: usize,
    pub input: &'a str,
    pub input_latex: Option<String>,
    pub steps_mode: &'a str,
    pub steps: Vec<StepJson>,
    pub solve_steps: Vec<SolveStepJson>,
    pub warnings: Vec<WarningJson>,
    pub required_conditions: Vec<RequiredConditionJson>,
    pub required_display: Vec<String>,
    pub raw_steps_count: usize,
    pub raw_solve_steps_count: usize,
    pub budget_preset: &'a str,
    pub strict: bool,
    pub domain: &'a str,
    pub timings_us: TimingsJson,
    pub context_mode: &'a str,
    pub branch_mode: &'a str,
    pub expand_policy: &'a str,
    pub complex_mode: &'a str,
    pub const_fold: &'a str,
    pub value_domain: &'a str,
    pub complex_branch: &'a str,
    pub inv_trig: &'a str,
    pub assume_scope: &'a str,
}

/// Build a complete eval-json output payload from precomputed components.
pub fn build_eval_json_output(parts: EvalJsonOutputBuild<'_>) -> EvalJsonOutput {
    EvalJsonOutput {
        schema_version: SCHEMA_VERSION,
        ok: true,
        input: parts.input.to_string(),
        input_latex: parts.input_latex,
        result: parts.result,
        result_truncated: parts.result_truncated,
        result_chars: parts.result_chars,
        result_latex: parts.result_latex,
        steps_mode: parts.steps_mode.to_string(),
        steps_count: parts.steps_count,
        steps: parts.steps,
        solve_steps: parts.solve_steps,
        warnings: parts.warnings,
        required_conditions: parts.required_conditions,
        required_display: parts.required_display,
        budget: super::build_budget_json_eval(parts.budget_preset, parts.strict),
        domain: super::build_domain_json_eval(parts.domain),
        stats: parts.stats,
        hash: parts.hash,
        timings_us: parts.timings_us,
        options: super::build_options_json_eval(
            parts.context_mode,
            parts.branch_mode,
            parts.expand_policy,
            parts.complex_mode,
            parts.steps_mode,
            parts.domain,
            parts.const_fold,
        ),
        semantics: super::build_semantics_json_eval(
            parts.domain,
            parts.value_domain,
            parts.complex_branch,
            parts.inv_trig,
            parts.assume_scope,
        ),
        wire: parts.wire,
    }
}

/// Finalize `EvalJsonOutput` from common precomputed metadata and a computed `EvalResult`.
pub fn finalize_eval_json_output(
    input: EvalJsonFinalizeInput<'_>,
) -> Result<EvalJsonOutput, String> {
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
        crate::EvalResult::SolutionSet(solution_set) => {
            let result_str = super::format_solution_set_eval_json(ctx, solution_set);
            let result_latex = super::solution_set_to_latex_eval_json(ctx, solution_set);
            let wire = build_eval_wire_value(
                &warnings,
                &required_display,
                &result_str,
                Some(&result_latex),
                raw_steps_count + raw_solve_steps_count,
                steps_mode,
            );

            Ok(build_eval_json_output(EvalJsonOutputBuild {
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
        crate::EvalResult::Bool(b) => {
            let result_str = b.to_string();
            let wire = build_eval_wire_value(
                &warnings,
                &required_display,
                &result_str,
                None,
                raw_steps_count,
                steps_mode,
            );

            Ok(build_eval_json_output(EvalJsonOutputBuild {
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
        crate::EvalResult::Expr(e) => finalize_expr_like_eval_json_output(ExprLikeFinalizeInput {
            result_expr: *e,
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
        }),
        crate::EvalResult::Set(v) if !v.is_empty() => {
            finalize_expr_like_eval_json_output(ExprLikeFinalizeInput {
                result_expr: v[0],
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
            })
        }
        _ => Err("No result expression".to_string()),
    }
}

/// Finalize `EvalJsonOutput` using context from engine.
pub fn finalize_eval_json_output_with_engine(
    engine: &crate::Engine,
    input: EvalJsonFinalizeWithEngineInput<'_>,
) -> Result<EvalJsonOutput, String> {
    finalize_eval_json_output(EvalJsonFinalizeInput {
        result: input.result,
        ctx: &engine.simplifier.context,
        max_chars: input.max_chars,
        input: input.input,
        input_latex: input.input_latex,
        steps_mode: input.steps_mode,
        steps: input.steps,
        solve_steps: input.solve_steps,
        warnings: input.warnings,
        required_conditions: input.required_conditions,
        required_display: input.required_display,
        raw_steps_count: input.raw_steps_count,
        raw_solve_steps_count: input.raw_solve_steps_count,
        budget_preset: input.budget_preset,
        strict: input.strict,
        domain: input.domain,
        timings_us: input.timings_us,
        context_mode: input.context_mode,
        branch_mode: input.branch_mode,
        expand_policy: input.expand_policy,
        complex_mode: input.complex_mode,
        const_fold: input.const_fold,
        value_domain: input.value_domain,
        complex_branch: input.complex_branch,
        inv_trig: input.inv_trig,
        assume_scope: input.assume_scope,
    })
}

struct ExprLikeFinalizeInput<'a> {
    result_expr: ExprId,
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

fn finalize_expr_like_eval_json_output(
    input: ExprLikeFinalizeInput<'_>,
) -> Result<EvalJsonOutput, String> {
    let ExprLikeFinalizeInput {
        result_expr,
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

    Ok(build_eval_json_output(EvalJsonOutputBuild {
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

/// Build eval-json `wire` payload as `serde_json::Value`.
pub fn build_eval_wire_value(
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

/// Render an `EvalResult` to plain text for CLI-style output.
pub fn format_eval_result_text(ctx: &Context, result: &crate::EvalResult) -> String {
    match result {
        crate::EvalResult::Expr(expr) => {
            if let Some(poly_str) = crate::try_render_poly_result(ctx, *expr) {
                poly_str
            } else {
                format!(
                    "{}",
                    DisplayExpr {
                        context: ctx,
                        id: *expr
                    }
                )
            }
        }
        crate::EvalResult::Set(values) if !values.is_empty() => format!(
            "{}",
            DisplayExpr {
                context: ctx,
                id: values[0]
            }
        ),
        crate::EvalResult::Bool(value) => value.to_string(),
        _ => "(no result)".to_string(),
    }
}

/// Format an expression with a character limit for eval-json output.
///
/// Returns `(formatted, truncated, original_char_count)`.
pub fn format_expr_limited_eval_json(
    ctx: &Context,
    expr: ExprId,
    max_chars: usize,
) -> (String, bool, usize) {
    if let Some(poly_str) = crate::try_render_poly_result(ctx, expr) {
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

/// Compute expression statistics for eval-json payload.
pub fn expr_stats_eval_json(ctx: &Context, expr: ExprId) -> ExprStatsJson {
    let (node_count, depth) = count_nodes_and_depth(ctx, expr);
    let term_count =
        crate::try_get_poly_result_term_count(ctx, expr).or_else(|| count_add_terms(ctx, expr));

    ExprStatsJson {
        node_count,
        depth,
        term_count,
    }
}

/// Compute a deterministic structural hash for an expression.
pub fn expr_hash_eval_json(ctx: &Context, expr: ExprId) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hasher;

    let mut hasher = DefaultHasher::new();
    hash_expr_recursive(ctx, expr, &mut hasher);
    format!("{:016x}", hasher.finish())
}

fn count_nodes_and_depth(ctx: &Context, expr: ExprId) -> (usize, usize) {
    cas_ast::traversal::count_nodes_and_max_depth(ctx, expr)
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
        Expr::Hold(inner) => {
            12u8.hash(hasher);
            hash_expr_recursive(ctx, *inner, hasher);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_expr_limited_eval_json_no_truncate() {
        let mut ctx = Context::new();
        let expr = cas_parser::parse("x + 1", &mut ctx).expect("parse");
        let (s, trunc, len) = format_expr_limited_eval_json(&ctx, expr, 100);
        assert!(!trunc);
        assert!(len <= 100);
        assert!(s.contains("x"));
    }

    #[test]
    fn format_expr_limited_eval_json_truncate() {
        let mut ctx = Context::new();
        let expr = cas_parser::parse("x + y + z + a + b + c", &mut ctx).expect("parse");
        let (s, trunc, _len) = format_expr_limited_eval_json(&ctx, expr, 5);
        assert!(trunc);
        assert!(s.contains("truncated"));
    }

    #[test]
    fn expr_stats_eval_json_counts_nodes_and_depth() {
        let mut ctx = Context::new();
        let expr = cas_parser::parse("x + 1", &mut ctx).expect("parse");
        let stats = expr_stats_eval_json(&ctx, expr);
        assert!(stats.node_count >= 3);
        assert!(stats.depth >= 1);
    }

    #[test]
    fn expr_hash_eval_json_is_deterministic() {
        let mut ctx = Context::new();
        let e1 = cas_parser::parse("x + 1", &mut ctx).expect("parse");
        let e2 = cas_parser::parse("x + 1", &mut ctx).expect("parse");
        assert_eq!(expr_hash_eval_json(&ctx, e1), expr_hash_eval_json(&ctx, e2));
    }
}
