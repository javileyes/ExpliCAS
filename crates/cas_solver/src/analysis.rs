use crate::rationalize::{rationalize_denominator, RationalizeConfig, RationalizeResult};
use crate::{
    apply_weierstrass_recursive, canonical_forms, expand_log_recursive, parse_expr_pair,
    parse_substitute_args, substitute_auto_with_strategy, EquivalenceResult, ParseExprPairError,
    ParseSubstituteArgsError, Simplifier, Step, SubstituteOptions, SubstituteStrategy,
};
use cas_ast::{Expr, ExprId};
use cas_formatter::DisplayExpr;

/// Error while evaluating unary function commands (e.g. det/trace/transpose).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnaryFunctionEvalError {
    Parse(String),
}

/// Output of evaluating a unary function call over a parsed expression.
#[derive(Debug, Clone)]
pub struct UnaryFunctionEvalOutput {
    pub parsed_expr: ExprId,
    pub result_expr: ExprId,
    pub steps: Vec<Step>,
}

/// Rendering options for unary-function command outputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnaryFunctionRenderConfig<'a> {
    pub function_name: &'a str,
    pub show_steps: bool,
    pub show_step_assumptions: bool,
}

/// Error while evaluating rationalize commands.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RationalizeEvalError {
    Parse(String),
}

/// Error while evaluating symbolic transform commands.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransformEvalError {
    Parse(String),
}

/// Error while evaluating timeline simplification commands.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TimelineEvalError {
    Parse(String),
    Eval(String),
}

/// Error while evaluating full simplify commands.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FullSimplifyEvalError {
    Parse(String),
    Resolve(String),
}

/// Error while evaluating CLI-style limit command strings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LimitCommandEvalError {
    EmptyInput,
    Parse(String),
    Limit(String),
}

/// Rationalize command result classes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RationalizeEvalOutcome {
    Success { simplified_expr: ExprId },
    NotApplicable,
    BudgetExceeded,
}

/// Output of rationalize command evaluation.
#[derive(Debug, Clone)]
pub struct RationalizeEvalOutput {
    pub parsed_expr: ExprId,
    pub normalized_expr: ExprId,
    pub outcome: RationalizeEvalOutcome,
}

/// Output of Weierstrass command evaluation.
#[derive(Debug, Clone)]
pub struct WeierstrassEvalOutput {
    pub parsed_expr: ExprId,
    pub substituted_expr: ExprId,
    pub simplified_expr: ExprId,
}

/// Output of expand_log command evaluation.
#[derive(Debug, Clone)]
pub struct ExpandLogEvalOutput {
    pub parsed_expr: ExprId,
    pub expanded_expr: ExprId,
}

/// Output of telescope command evaluation.
#[derive(Debug, Clone)]
pub struct TelescopeEvalOutput {
    pub parsed_expr: ExprId,
    pub formatted_result: String,
}

/// Error while evaluating explain commands.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExplainEvalError {
    Parse(String),
    NotFunctionCall,
    UnsupportedFunction(String),
    InvalidArity {
        function: String,
        expected: usize,
        actual: usize,
    },
}

/// Output for `explain gcd(a, b)`.
#[derive(Debug, Clone)]
pub struct ExplainGcdEvalOutput {
    pub parsed_expr: ExprId,
    pub steps: Vec<String>,
    pub value: Option<ExprId>,
}

/// Output for AST visualization command.
#[derive(Debug, Clone)]
pub struct VisualizeEvalOutput {
    pub parsed_expr: ExprId,
    pub dot: String,
}

/// Output of substitution command after applying substitution + simplification.
#[derive(Debug, Clone)]
pub struct SubstituteEvalOutput {
    pub substituted_expr: ExprId,
    pub simplified_expr: ExprId,
    pub strategy: SubstituteStrategy,
    pub steps: Vec<Step>,
}

/// Output of timeline simplification evaluation.
#[derive(Debug, Clone)]
pub struct TimelineSimplifyEvalOutput {
    pub parsed_expr: ExprId,
    pub simplified_expr: ExprId,
    pub steps: crate::DisplayEvalSteps,
}

/// Simplify branch payload for `timeline ...` command.
#[derive(Debug, Clone)]
pub struct TimelineSimplifyCommandEvalOutput {
    pub expr_input: String,
    pub use_aggressive: bool,
    pub parsed_expr: ExprId,
    pub simplified_expr: ExprId,
    pub steps: crate::DisplayEvalSteps,
}

/// End-to-end dispatch output for REPL `timeline` command.
#[derive(Debug)]
pub enum TimelineCommandEvalOutput {
    Solve(crate::solve::TimelineSolveEvalOutput),
    Simplify(TimelineSimplifyCommandEvalOutput),
}

/// End-to-end dispatch error for REPL `timeline` command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TimelineCommandEvalError {
    Solve(crate::solve::TimelineSolveEvalError),
    Simplify(TimelineEvalError),
}

/// Output of full simplify command evaluation.
#[derive(Debug, Clone)]
pub struct FullSimplifyEvalOutput {
    pub parsed_expr: ExprId,
    pub resolved_expr: ExprId,
    pub simplified_expr: ExprId,
    pub steps: Vec<Step>,
    pub stats: cas_engine::PipelineStats,
}

/// Output of evaluating a CLI-style `limit` command tail.
#[derive(Debug, Clone)]
pub struct LimitCommandEvalOutput {
    pub var: String,
    pub approach: crate::Approach,
    pub result: String,
    pub warning: Option<String>,
}

/// Rendering mode for substitution command output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubstituteRenderMode {
    None,
    Succinct,
    Normal,
    Verbose,
}

/// Map generic display mode (shared across REPL commands) to substitution rendering mode.
pub fn substitute_render_mode_from_display_mode(
    mode: crate::set_command::SetDisplayMode,
) -> SubstituteRenderMode {
    match mode {
        crate::set_command::SetDisplayMode::None => SubstituteRenderMode::None,
        crate::set_command::SetDisplayMode::Succinct => SubstituteRenderMode::Succinct,
        crate::set_command::SetDisplayMode::Normal => SubstituteRenderMode::Normal,
        crate::set_command::SetDisplayMode::Verbose => SubstituteRenderMode::Verbose,
    }
}

/// Parsed tail for `rationalize` command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RationalizeCommandInput<'a> {
    MissingInput,
    Expr(&'a str),
}

/// Parsed tail for `expand` command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpandCommandInput<'a> {
    MissingInput,
    Expr(&'a str),
}

/// Parsed tail for `expand_log` command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpandLogCommandInput<'a> {
    MissingInput,
    Expr(&'a str),
}

/// Parsed tail for `telescope` command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TelescopeCommandInput<'a> {
    MissingInput,
    Expr(&'a str),
}

/// Parsed tail for `weierstrass` command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeierstrassCommandInput<'a> {
    MissingInput,
    Expr(&'a str),
}

/// Usage message for `subst` command.
pub fn substitute_usage_message() -> &'static str {
    "Usage: subst <expr>, <target>, <replacement>\n\n\
                     Examples:\n\
                       subst x^2 + x, x, 3              → 12\n\
                       subst x^4 + x^2 + 1, x^2, y      → y² + y + 1\n\
                       subst x^3, x^2, y                → y·x"
}

/// Usage message for `weierstrass` command.
pub fn weierstrass_usage_message() -> &'static str {
    "Usage: weierstrass <expression>\n\
                 Description: Apply Weierstrass substitution (t = tan(x/2))\n\
                 Transforms:\n\
                   sin(x) → 2t/(1+t²)\n\
                   cos(x) → (1-t²)/(1+t²)\n\
                   tan(x) → 2t/(1-t²)\n\
                 Example: weierstrass sin(x) + cos(x)"
}

/// Usage message for `limit` command.
pub fn limit_usage_message() -> &'static str {
    "Usage: limit <expr> [, <var> [, <direction> [, safe]]]\n\
                 Examples:\n\
                   limit x^2                      → infinity (default: x → +∞)\n\
                   limit (x^2+1)/(2*x^2-3), x     → 1/2\n\
                   limit x^3/x^2, x, -infinity    → -infinity\n\
                   limit (x-x)/x, x, infinity, safe → 0 (with pre-simplify)"
}

/// Usage message for `rationalize` command.
pub fn rationalize_usage_message() -> &'static str {
    "Usage: rationalize <expr>\n\
                 Example: rationalize 1/(1 + sqrt(2) + sqrt(3))"
}

/// Parse full `rationalize ...` line into command input.
pub fn parse_rationalize_command_input(line: &str) -> RationalizeCommandInput<'_> {
    let rest = line.strip_prefix("rationalize").unwrap_or(line).trim();
    if rest.is_empty() {
        RationalizeCommandInput::MissingInput
    } else {
        RationalizeCommandInput::Expr(rest)
    }
}

/// Usage message for `expand` command.
pub fn expand_usage_message() -> &'static str {
    "Usage: expand <expr>\n\
                 Description: Aggressively expands and distributes polynomials.\n\
                 Example: expand 1/2 * (sqrt(2) - 1) → sqrt(2)/2 - 1/2"
}

/// Parse full `expand ...` line into command input.
pub fn parse_expand_command_input(line: &str) -> ExpandCommandInput<'_> {
    let rest = line.strip_prefix("expand").unwrap_or(line).trim();
    if rest.is_empty() {
        ExpandCommandInput::MissingInput
    } else {
        ExpandCommandInput::Expr(rest)
    }
}

/// Build `expand(<expr>)` function call string for eval path.
pub fn wrap_expand_eval_expression(expr: &str) -> String {
    format!("expand({})", expr)
}

/// Usage message for `expand_log` command.
pub fn expand_log_usage_message() -> &'static str {
    "Usage: expand_log <expr>\n\
                 Description: Expand logarithms using log properties.\n\
                 Transformations:\n\
                   ln(x*y)   → ln(x) + ln(y)\n\
                   ln(x/y)   → ln(x) - ln(y)\n\
                   ln(x^n)   → n * ln(x)\n\
                 Example: expand_log ln(x^2 * y) → 2*ln(x) + ln(y)"
}

/// Parse full `expand_log ...` line into command input.
pub fn parse_expand_log_command_input(line: &str) -> ExpandLogCommandInput<'_> {
    let rest = line.strip_prefix("expand_log").unwrap_or(line).trim();
    if rest.is_empty() {
        ExpandLogCommandInput::MissingInput
    } else {
        ExpandLogCommandInput::Expr(rest)
    }
}

/// Usage message for `telescope` command.
pub fn telescope_usage_message() -> &'static str {
    "Usage: telescope <expression>\n\
                 Example: telescope 1 + 2*cos(x) + 2*cos(2*x) - sin(5*x/2)/sin(x/2)"
}

/// Parse full `telescope ...` line into command input.
pub fn parse_telescope_command_input(line: &str) -> TelescopeCommandInput<'_> {
    let rest = line.strip_prefix("telescope").unwrap_or(line).trim();
    if rest.is_empty() {
        TelescopeCommandInput::MissingInput
    } else {
        TelescopeCommandInput::Expr(rest)
    }
}

/// Parse full `weierstrass ...` line into command input.
pub fn parse_weierstrass_command_input(line: &str) -> WeierstrassCommandInput<'_> {
    let rest = line.strip_prefix("weierstrass").unwrap_or(line).trim();
    if rest.is_empty() {
        WeierstrassCommandInput::MissingInput
    } else {
        WeierstrassCommandInput::Expr(rest)
    }
}

/// Extract command tail from `equiv ...` input.
pub fn extract_equiv_command_tail(line: &str) -> &str {
    line.strip_prefix("equiv").unwrap_or(line).trim()
}

/// Extract command tail from `subst ...` input.
pub fn extract_substitute_command_tail(line: &str) -> &str {
    line.strip_prefix("subst").unwrap_or(line).trim()
}

/// Extract command tail from `timeline ...` input.
pub fn extract_timeline_command_tail(line: &str) -> &str {
    line.strip_prefix("timeline").unwrap_or(line).trim()
}

/// Extract command tail from `explain ...` input.
pub fn extract_explain_command_tail(line: &str) -> &str {
    line.strip_prefix("explain").unwrap_or(line).trim()
}

/// Extract command tail from `solve ...` input.
pub fn extract_solve_command_tail(line: &str) -> &str {
    line.strip_prefix("solve").unwrap_or(line).trim()
}

/// Extract command tail from `simplify ...` input.
pub fn extract_simplify_command_tail(line: &str) -> &str {
    line.strip_prefix("simplify").unwrap_or(line).trim()
}

/// Extract command tail from `limit ...` input.
pub fn extract_limit_command_tail(line: &str) -> &str {
    line.strip_prefix("limit").unwrap_or(line).trim()
}

/// Extract command tail from `visualize ...` / `viz ...` input.
pub fn extract_visualize_command_tail(line: &str) -> &str {
    line.strip_prefix("visualize ")
        .or_else(|| line.strip_prefix("viz "))
        .unwrap_or(line)
        .trim()
}

/// Extract command tail from unary command input (e.g. `det`, `trace`, `transpose`).
pub fn extract_unary_command_tail<'a>(line: &'a str, command: &str) -> &'a str {
    line.strip_prefix(command).unwrap_or(line).trim()
}

/// Format parse errors for `subst` command inputs.
pub fn format_substitute_parse_error_message(error: &ParseSubstituteArgsError) -> String {
    match error {
        ParseSubstituteArgsError::InvalidArity => substitute_usage_message().to_string(),
        ParseSubstituteArgsError::Expression(e) => format!("Error parsing expression: {}", e),
        ParseSubstituteArgsError::Target(e) => format!("Error parsing target: {}", e),
        ParseSubstituteArgsError::Replacement(e) => format!("Error parsing replacement: {}", e),
    }
}

fn should_render_substitute_step(step: &Step, mode: SubstituteRenderMode) -> bool {
    use cas_engine::ImportanceLevel;

    match mode {
        SubstituteRenderMode::None => false,
        SubstituteRenderMode::Verbose => true,
        SubstituteRenderMode::Succinct | SubstituteRenderMode::Normal => {
            if step.get_importance() < ImportanceLevel::Medium {
                return false;
            }
            if let (Some(before), Some(after)) = (step.global_before, step.global_after) {
                if before == after {
                    return false;
                }
            }
            true
        }
    }
}

/// Format substitution eval output for CLI/UI rendering.
pub fn format_substitute_eval_lines(
    context: &cas_ast::Context,
    input: &str,
    output: &SubstituteEvalOutput,
    mode: SubstituteRenderMode,
) -> Vec<String> {
    let display_parts = crate::split_by_comma_ignoring_parens(input);
    let expr_str = display_parts.first().map(|s| s.trim()).unwrap_or_default();
    let target_str = display_parts.get(1).map(|s| s.trim()).unwrap_or_default();
    let replacement_str = display_parts.get(2).map(|s| s.trim()).unwrap_or_default();

    let mut lines = Vec::new();
    if mode != SubstituteRenderMode::None {
        let label = match output.strategy {
            SubstituteStrategy::Variable => "Variable substitution",
            SubstituteStrategy::PowerAware => "Expression substitution",
        };
        lines.push(format!(
            "{label}: {} → {} in {}",
            target_str, replacement_str, expr_str
        ));
    }

    if mode != SubstituteRenderMode::None && !output.steps.is_empty() {
        if mode != SubstituteRenderMode::Succinct {
            lines.push("Steps:".to_string());
        }
        for step in &output.steps {
            if should_render_substitute_step(step, mode) {
                if mode == SubstituteRenderMode::Succinct {
                    lines.push(format!(
                        "-> {}",
                        DisplayExpr {
                            context,
                            id: step.global_after.unwrap_or(step.after)
                        }
                    ));
                } else {
                    lines.push(format!("  {}  [{}]", step.description, step.rule_name));
                }
            }
        }
    }

    lines.push(format!(
        "Result: {}",
        DisplayExpr {
            context,
            id: output.simplified_expr
        }
    ));
    lines
}

/// Format parse errors for pair-style commands (`equiv`).
pub fn format_expr_pair_parse_error_message(error: &ParseExprPairError, command: &str) -> String {
    match error {
        ParseExprPairError::MissingDelimiter => format!("Usage: {} <expr1>, <expr2>", command),
        ParseExprPairError::FirstArg(e) => format!("Error parsing first arg: {}", e),
        ParseExprPairError::SecondArg(e) => format!("Error parsing second arg: {}", e),
    }
}

/// Format equivalence result lines for CLI/UI output.
pub fn format_equivalence_result_lines(result: &EquivalenceResult) -> Vec<String> {
    let mut lines = Vec::new();
    match result {
        EquivalenceResult::True => {
            lines.push("True".to_string());
        }
        EquivalenceResult::ConditionalTrue { requires } => {
            lines.push("True (conditional)".to_string());
            lines.extend(crate::format_text_requires_lines(requires));
        }
        EquivalenceResult::False => {
            lines.push("False".to_string());
        }
        EquivalenceResult::Unknown => {
            lines.push("Unknown (cannot prove equivalence)".to_string());
        }
    }
    lines
}

/// Format timeline simplify error for CLI/UI output.
pub fn format_timeline_eval_error_message(error: &TimelineEvalError) -> String {
    match error {
        TimelineEvalError::Parse(e) => format!("Parse error: {}", e),
        TimelineEvalError::Eval(e) => format!("Simplification error: {}", e),
    }
}

/// Message used when no timeline steps were produced.
pub fn timeline_no_steps_message() -> &'static str {
    "No simplification steps to visualize."
}

/// Shared hint message for timeline HTML output.
pub fn timeline_open_hint_message() -> &'static str {
    "Open in browser to view interactive visualization."
}

/// Format timeline simplify info lines (post file-write).
pub fn format_timeline_simplify_info_lines(use_aggressive: bool) -> Vec<String> {
    let mut lines = Vec::new();
    if use_aggressive {
        lines.push("(Aggressive simplification mode)".to_string());
    }
    lines.push(timeline_open_hint_message().to_string());
    lines
}

/// Default labels for history-entry eval metadata sections.
pub fn history_eval_metadata_section_labels() -> crate::EvalMetadataSectionLabels<'static> {
    crate::EvalMetadataSectionLabels {
        required_header: "  ℹ️ Requires:",
        assumed_header: "  ⚠ Assumed:",
        blocked_header: "  🚫 Blocked:",
        line_prefix: "    - ",
    }
}

/// Hint lines for `visualize` command output.
pub fn visualize_output_hint_lines() -> Vec<String> {
    vec![
        "Render with: dot -Tsvg ast.dot -o ast.svg".to_string(),
        "Or: dot -Tpng ast.dot -o ast.png".to_string(),
    ]
}

/// Format timeline command dispatch errors for CLI/UI output.
pub fn format_timeline_command_error_message(error: &TimelineCommandEvalError) -> String {
    match error {
        TimelineCommandEvalError::Solve(e) => crate::format_timeline_solve_error_message(e),
        TimelineCommandEvalError::Simplify(e) => format_timeline_eval_error_message(e),
    }
}

/// Format explain command errors for CLI/UI output.
pub fn format_explain_error_message(error: &ExplainEvalError) -> String {
    match error {
        ExplainEvalError::Parse(e) => format!("Parse error: {}", e),
        ExplainEvalError::NotFunctionCall => {
            "Explain mode currently only supports function calls\n\
                 Try: explain gcd(48, 18)"
                .to_string()
        }
        ExplainEvalError::UnsupportedFunction(name) => format!(
            "Explain mode not yet implemented for function '{}'\n\
                 Currently supported: gcd",
            name
        ),
        ExplainEvalError::InvalidArity { .. } => "Usage: explain gcd(a, b)".to_string(),
    }
}

/// Format unary-function eval output lines for CLI/UI rendering.
pub fn format_unary_function_eval_lines(
    context: &cas_ast::Context,
    input: &str,
    output: &UnaryFunctionEvalOutput,
    config: UnaryFunctionRenderConfig<'_>,
) -> Vec<String> {
    let mut lines = vec![format!("Parsed: {}({})", config.function_name, input)];

    if config.show_steps && !output.steps.is_empty() {
        lines.push("Steps:".to_string());
        for (i, step) in output.steps.iter().enumerate() {
            lines.push(format!(
                "{}. {}  [{}]",
                i + 1,
                step.description,
                step.rule_name
            ));
            if config.show_step_assumptions {
                for assumption_line in
                    crate::format_displayable_assumption_lines(step.assumption_events())
                {
                    lines.push(format!("   {}", assumption_line));
                }
            }
        }
    }

    lines.push(format!(
        "Result: {}",
        DisplayExpr {
            context,
            id: output.result_expr
        }
    ));
    lines
}

/// Build unary-function render config from shared display mode.
pub fn unary_render_config_for_display_mode<'a>(
    function_name: &'a str,
    mode: crate::set_command::SetDisplayMode,
    show_step_assumptions: bool,
) -> UnaryFunctionRenderConfig<'a> {
    UnaryFunctionRenderConfig {
        function_name,
        show_steps: mode != crate::set_command::SetDisplayMode::None,
        show_step_assumptions,
    }
}

/// Format unary-function eval errors for CLI/UI output.
pub fn format_unary_function_eval_error_message(error: &UnaryFunctionEvalError) -> String {
    match error {
        UnaryFunctionEvalError::Parse(e) => format!("Parse error: {}", e),
    }
}

/// Format rationalize eval output lines for CLI/UI rendering.
pub fn format_rationalize_eval_lines(
    context: &cas_ast::Context,
    output: &RationalizeEvalOutput,
) -> Vec<String> {
    let user_style = cas_formatter::root_style::detect_root_style(context, output.normalized_expr);
    let parsed = format!(
        "{}",
        DisplayExpr {
            context,
            id: output.normalized_expr
        }
    );

    let line = match output.outcome {
        RationalizeEvalOutcome::Success { simplified_expr } => {
            let style = cas_formatter::root_style::StylePreferences::with_root_style(user_style);
            let rendered = cas_formatter::DisplayExprStyled::new(context, simplified_expr, &style);
            format!("Parsed: {}\nRationalized: {}", parsed, rendered)
        }
        RationalizeEvalOutcome::NotApplicable => format!(
            "Parsed: {}\n\
             Cannot rationalize: denominator is not a sum of surds\n\
             (Supported: 1/(a + b√n + c√m) where a,b,c are rational and n,m are positive integers)",
            parsed
        ),
        RationalizeEvalOutcome::BudgetExceeded => format!(
            "Parsed: {}\n\
             Rationalization aborted: expression became too complex",
            parsed
        ),
    };

    vec![line]
}

/// Format rationalize eval errors for CLI/UI output.
pub fn format_rationalize_eval_error_message(error: &RationalizeEvalError) -> String {
    match error {
        RationalizeEvalError::Parse(e) => format!("Parse error: {}", e),
    }
}

/// Format expand-log eval output lines for CLI/UI rendering.
pub fn format_expand_log_eval_lines(
    context: &cas_ast::Context,
    output: &ExpandLogEvalOutput,
) -> Vec<String> {
    vec![
        format!(
            "Parsed: {}",
            DisplayExpr {
                context,
                id: output.parsed_expr
            }
        ),
        format!(
            "Result: {}",
            DisplayExpr {
                context,
                id: output.expanded_expr
            }
        ),
    ]
}

/// Format telescope eval output lines for CLI/UI rendering.
pub fn format_telescope_eval_lines(input: &str, output: &TelescopeEvalOutput) -> Vec<String> {
    vec![format!("Parsed: {}\n\n{}", input, output.formatted_result)]
}

/// Format transform eval errors for CLI/UI output.
pub fn format_transform_eval_error_message(error: &TransformEvalError) -> String {
    match error {
        TransformEvalError::Parse(e) => format!("Parse error: {}", e),
    }
}

/// Format Weierstrass eval output lines for CLI/UI rendering.
pub fn format_weierstrass_eval_lines(
    context: &cas_ast::Context,
    input: &str,
    output: &WeierstrassEvalOutput,
) -> Vec<String> {
    vec![
        format!("Parsed: {}", input),
        String::new(),
        "Weierstrass substitution (t = tan(x/2)):".to_string(),
        format!(
            "  {} → {}",
            input,
            DisplayExpr {
                context,
                id: output.substituted_expr
            }
        ),
        String::new(),
        "Simplifying...".to_string(),
        format!(
            "Result: {}",
            DisplayExpr {
                context,
                id: output.simplified_expr
            }
        ),
    ]
}

/// Format limit eval output lines for CLI/UI rendering.
pub fn format_limit_command_eval_lines(output: &LimitCommandEvalOutput) -> Vec<String> {
    let dir_disp = match output.approach {
        crate::Approach::PosInfinity => "+∞",
        crate::Approach::NegInfinity => "-∞",
    };
    let mut lines = vec![format!(
        "lim_{{{}→{}}} = {}",
        output.var, dir_disp, output.result
    )];
    if let Some(warning) = &output.warning {
        lines.push(format!("Warning: {}", warning));
    }
    lines
}

/// Format limit eval errors for CLI/UI output.
pub fn format_limit_command_error_message(error: &LimitCommandEvalError) -> String {
    match error {
        LimitCommandEvalError::EmptyInput => limit_usage_message().to_string(),
        LimitCommandEvalError::Parse(message) => message.clone(),
        LimitCommandEvalError::Limit(message) => format!("Error computing limit: {}", message),
    }
}

/// Format `explain gcd(...)` output lines for CLI/UI rendering.
pub fn format_explain_gcd_eval_lines(
    context: &cas_ast::Context,
    input: &str,
    output: &ExplainGcdEvalOutput,
) -> Vec<String> {
    let mut lines = Vec::new();
    lines.push(format!("Parsed: {}", input));
    lines.push(String::new());
    lines.push("Educational Steps:".to_string());
    lines.push("─".repeat(60));

    for step in &output.steps {
        lines.push(step.clone());
    }

    lines.push("─".repeat(60));
    lines.push(String::new());

    if let Some(result_expr) = output.value {
        lines.push(format!(
            "Result: {}",
            DisplayExpr {
                context,
                id: result_expr
            }
        ));
    } else {
        lines.push("Could not compute GCD".to_string());
    }
    lines
}

/// Evaluate equivalence from a REPL-style input:
/// `"<expr1>, <expr2>"`.
pub fn evaluate_equiv_input(
    simplifier: &mut Simplifier,
    input: &str,
) -> Result<EquivalenceResult, ParseExprPairError> {
    let (lhs, rhs) = parse_expr_pair(&mut simplifier.context, input)?;
    Ok(simplifier.are_equivalent_extended(lhs, rhs))
}

/// Evaluate substitution from REPL-style input:
/// `"<expr>, <target>, <replacement>"`.
///
/// Returns the substituted expression id and strategy; caller can decide
/// whether to further simplify and how to render steps.
pub fn evaluate_substitute_input(
    simplifier: &mut Simplifier,
    input: &str,
    options: SubstituteOptions,
) -> Result<(ExprId, SubstituteStrategy), ParseSubstituteArgsError> {
    let (expr, target, replacement) = parse_substitute_args(&mut simplifier.context, input)?;
    Ok(substitute_auto_with_strategy(
        &mut simplifier.context,
        expr,
        target,
        replacement,
        options,
    ))
}

/// Evaluate substitution input and immediately simplify the substituted result.
pub fn evaluate_substitute_and_simplify_input(
    simplifier: &mut Simplifier,
    input: &str,
    options: SubstituteOptions,
) -> Result<SubstituteEvalOutput, ParseSubstituteArgsError> {
    let (substituted_expr, strategy) = evaluate_substitute_input(simplifier, input, options)?;
    let (simplified_expr, steps) = simplifier.simplify(substituted_expr);
    Ok(SubstituteEvalOutput {
        substituted_expr,
        simplified_expr,
        strategy,
        steps,
    })
}

/// Evaluate a unary function command:
/// parse `<input>`, build `function_name(input)`, simplify, and return steps.
pub fn evaluate_unary_function_input(
    simplifier: &mut Simplifier,
    function_name: &str,
    input: &str,
) -> Result<UnaryFunctionEvalOutput, UnaryFunctionEvalError> {
    let parsed_expr = cas_parser::parse(input, &mut simplifier.context)
        .map_err(|e| UnaryFunctionEvalError::Parse(e.to_string()))?;
    let call_expr = simplifier.context.call(function_name, vec![parsed_expr]);
    let (result_expr, steps) = simplifier.simplify(call_expr);
    Ok(UnaryFunctionEvalOutput {
        parsed_expr,
        result_expr,
        steps,
    })
}

/// Evaluate rationalization command:
/// parse input, normalize canonical form, rationalize denominator, simplify success case.
pub fn evaluate_rationalize_input(
    simplifier: &mut Simplifier,
    input: &str,
) -> Result<RationalizeEvalOutput, RationalizeEvalError> {
    let parsed_expr = cas_parser::parse(input, &mut simplifier.context)
        .map_err(|e| RationalizeEvalError::Parse(format!("{:?}", e)))?;

    let normalized_expr = canonical_forms::normalize_core(&mut simplifier.context, parsed_expr);
    let config = RationalizeConfig::default();
    let rationalized = rationalize_denominator(&mut simplifier.context, normalized_expr, &config);

    let outcome = match rationalized {
        RationalizeResult::Success(expr) => {
            let (simplified_expr, _) = simplifier.simplify(expr);
            RationalizeEvalOutcome::Success { simplified_expr }
        }
        RationalizeResult::NotApplicable => RationalizeEvalOutcome::NotApplicable,
        RationalizeResult::BudgetExceeded => RationalizeEvalOutcome::BudgetExceeded,
    };

    Ok(RationalizeEvalOutput {
        parsed_expr,
        normalized_expr,
        outcome,
    })
}

/// Evaluate Weierstrass command:
/// parse input, apply recursive substitution, then simplify.
pub fn evaluate_weierstrass_input(
    simplifier: &mut Simplifier,
    input: &str,
) -> Result<WeierstrassEvalOutput, TransformEvalError> {
    let parsed_expr = cas_parser::parse(input, &mut simplifier.context)
        .map_err(|e| TransformEvalError::Parse(e.to_string()))?;
    let substituted_expr = apply_weierstrass_recursive(&mut simplifier.context, parsed_expr);
    let (simplified_expr, _steps) = simplifier.simplify(substituted_expr);
    Ok(WeierstrassEvalOutput {
        parsed_expr,
        substituted_expr,
        simplified_expr,
    })
}

/// Evaluate expand_log command:
/// parse input and recursively apply log expansion.
pub fn evaluate_expand_log_input(
    simplifier: &mut Simplifier,
    input: &str,
) -> Result<ExpandLogEvalOutput, TransformEvalError> {
    let parsed_expr = cas_parser::parse(input, &mut simplifier.context)
        .map_err(|e| TransformEvalError::Parse(e.to_string()))?;
    let expanded_expr = expand_log_recursive(&mut simplifier.context, parsed_expr);
    Ok(ExpandLogEvalOutput {
        parsed_expr,
        expanded_expr,
    })
}

/// Evaluate telescope command:
/// parse input and execute telescoping strategy, returning formatted report.
pub fn evaluate_telescope_input(
    simplifier: &mut Simplifier,
    input: &str,
) -> Result<TelescopeEvalOutput, TransformEvalError> {
    let parsed_expr = cas_parser::parse(input, &mut simplifier.context)
        .map_err(|e| TransformEvalError::Parse(e.to_string()))?;
    let result = crate::telescoping::telescope(&mut simplifier.context, parsed_expr);
    let formatted_result = result.format(&simplifier.context);
    Ok(TelescopeEvalOutput {
        parsed_expr,
        formatted_result,
    })
}

/// Evaluate explain command for `gcd(a, b)`.
pub fn evaluate_explain_gcd_input(
    simplifier: &mut Simplifier,
    input: &str,
) -> Result<ExplainGcdEvalOutput, ExplainEvalError> {
    let parsed_expr = cas_parser::parse(input, &mut simplifier.context)
        .map_err(|e| ExplainEvalError::Parse(e.to_string()))?;

    let expr_data = simplifier.context.get(parsed_expr).clone();
    let Expr::Function(name_id, args) = expr_data else {
        return Err(ExplainEvalError::NotFunctionCall);
    };
    let name = simplifier.context.sym_name(name_id).to_string();
    if name != "gcd" {
        return Err(ExplainEvalError::UnsupportedFunction(name));
    }
    if args.len() != 2 {
        return Err(ExplainEvalError::InvalidArity {
            function: "gcd".to_string(),
            expected: 2,
            actual: args.len(),
        });
    }

    let result = crate::number_theory::explain_gcd(&mut simplifier.context, args[0], args[1]);
    Ok(ExplainGcdEvalOutput {
        parsed_expr,
        steps: result.steps,
        value: result.value,
    })
}

/// Evaluate visualize command and return DOT graph.
pub fn evaluate_visualize_input(
    simplifier: &mut Simplifier,
    input: &str,
) -> Result<VisualizeEvalOutput, TransformEvalError> {
    let parsed_expr = cas_parser::parse(input, &mut simplifier.context)
        .map_err(|e| TransformEvalError::Parse(e.to_string()))?;
    let mut viz = crate::visualizer::AstVisualizer::new(&simplifier.context);
    let dot = viz.to_dot(parsed_expr);
    Ok(VisualizeEvalOutput { parsed_expr, dot })
}

/// Evaluate timeline simplify in "aggressive" mode by using a temporary default simplifier.
/// This mirrors CLI `simplify` aggressive flow while preserving caller context.
pub fn evaluate_timeline_simplify_aggressive_input(
    simplifier: &mut Simplifier,
    input: &str,
) -> Result<TimelineSimplifyEvalOutput, TimelineEvalError> {
    let mut temp_simplifier = Simplifier::with_default_rules();
    temp_simplifier.set_collect_steps(true);

    std::mem::swap(&mut simplifier.context, &mut temp_simplifier.context);
    let result = (|| {
        let parsed_expr = cas_parser::parse(input.trim(), &mut temp_simplifier.context)
            .map_err(|e| TimelineEvalError::Parse(e.to_string()))?;
        let (simplified_expr, steps) = temp_simplifier.simplify(parsed_expr);
        Ok(TimelineSimplifyEvalOutput {
            parsed_expr,
            simplified_expr,
            steps: crate::to_display_steps(steps),
        })
    })();
    std::mem::swap(&mut simplifier.context, &mut temp_simplifier.context);
    result
}

/// Evaluate timeline simplify through `Engine::eval` (stateful path).
pub fn evaluate_timeline_simplify_input<S>(
    engine: &mut cas_engine::Engine,
    session: &mut S,
    input: &str,
) -> Result<TimelineSimplifyEvalOutput, TimelineEvalError>
where
    S: cas_engine::EvalSession<
        Options = cas_engine::EvalOptions,
        Diagnostics = cas_engine::Diagnostics,
    >,
    S::Store: cas_engine::EvalStore<
        DomainMode = cas_engine::DomainMode,
        RequiredItem = cas_engine::RequiredItem,
        Step = cas_engine::Step,
        Diagnostics = cas_engine::Diagnostics,
    >,
{
    let was_collecting = engine.simplifier.collect_steps();
    engine.simplifier.set_collect_steps(true);
    let result = (|| {
        let parsed_expr = cas_parser::parse(input.trim(), &mut engine.simplifier.context)
            .map_err(|e| TimelineEvalError::Parse(e.to_string()))?;
        let req = cas_engine::EvalRequest {
            raw_input: input.to_string(),
            parsed: parsed_expr,
            action: cas_engine::EvalAction::Simplify,
            auto_store: false,
        };
        let output = engine
            .eval(session, req)
            .map_err(|e| TimelineEvalError::Eval(e.to_string()))?;
        let simplified_expr = match output.result {
            cas_engine::EvalResult::Expr(e) => e,
            _ => parsed_expr,
        };
        Ok(TimelineSimplifyEvalOutput {
            parsed_expr,
            simplified_expr,
            steps: output.steps,
        })
    })();
    engine.simplifier.set_collect_steps(was_collecting);
    result
}

/// Evaluate full `timeline` command tail:
/// - `solve ...` => solver timeline branch
/// - otherwise simplify timeline branch (normal or aggressive)
pub fn evaluate_timeline_command_input<S>(
    engine: &mut cas_engine::Engine,
    session: &mut S,
    input: &str,
    eval_options: &crate::EvalOptions,
) -> Result<TimelineCommandEvalOutput, TimelineCommandEvalError>
where
    S: cas_engine::EvalSession<
        Options = cas_engine::EvalOptions,
        Diagnostics = cas_engine::Diagnostics,
    >,
    S::Store: cas_engine::EvalStore<
        DomainMode = cas_engine::DomainMode,
        RequiredItem = cas_engine::RequiredItem,
        Step = cas_engine::Step,
        Diagnostics = cas_engine::Diagnostics,
    >,
{
    match crate::parse_timeline_command_input(input) {
        crate::TimelineCommandInput::Solve(solve_rest) => {
            crate::solve::evaluate_timeline_solve_with_eval_options(
                &mut engine.simplifier,
                &solve_rest,
                eval_options,
            )
            .map(TimelineCommandEvalOutput::Solve)
            .map_err(TimelineCommandEvalError::Solve)
        }
        crate::TimelineCommandInput::Simplify { expr, aggressive } => {
            let out = if aggressive {
                evaluate_timeline_simplify_aggressive_input(&mut engine.simplifier, &expr)
            } else {
                evaluate_timeline_simplify_input(engine, session, &expr)
            }
            .map_err(TimelineCommandEvalError::Simplify)?;

            Ok(TimelineCommandEvalOutput::Simplify(
                TimelineSimplifyCommandEvalOutput {
                    expr_input: expr,
                    use_aggressive: aggressive,
                    parsed_expr: out.parsed_expr,
                    simplified_expr: out.simplified_expr,
                    steps: out.steps,
                },
            ))
        }
    }
}

/// Evaluate full simplify input with aggressive default-rule simplifier.
/// Uses a temporary simplifier with swapped context/profiler and resolves
/// session refs via the provided session.
pub fn evaluate_full_simplify_input<S>(
    engine: &mut cas_engine::Engine,
    session: &S,
    input: &str,
    collect_steps: bool,
) -> Result<FullSimplifyEvalOutput, FullSimplifyEvalError>
where
    S: cas_engine::EvalSession<
        Options = cas_engine::EvalOptions,
        Diagnostics = cas_engine::Diagnostics,
    >,
{
    let mut temp_simplifier = Simplifier::with_default_rules();
    std::mem::swap(&mut engine.simplifier.context, &mut temp_simplifier.context);
    std::mem::swap(
        &mut engine.simplifier.profiler,
        &mut temp_simplifier.profiler,
    );

    let result = (|| {
        let parsed_expr = cas_parser::parse(input, &mut temp_simplifier.context)
            .map_err(|e| FullSimplifyEvalError::Parse(e.to_string()))?;
        let (resolved_expr, _diag, _cache_hits) = session
            .resolve_all_with_diagnostics(&mut temp_simplifier.context, parsed_expr)
            .map_err(|e| FullSimplifyEvalError::Resolve(e.to_string()))?;

        let mut opts = session.options().to_simplify_options();
        opts.collect_steps = collect_steps;
        let (simplified_expr, steps, stats) =
            temp_simplifier.simplify_with_stats(resolved_expr, opts);
        Ok(FullSimplifyEvalOutput {
            parsed_expr,
            resolved_expr,
            simplified_expr,
            steps,
            stats,
        })
    })();

    std::mem::swap(&mut engine.simplifier.context, &mut temp_simplifier.context);
    std::mem::swap(
        &mut engine.simplifier.profiler,
        &mut temp_simplifier.profiler,
    );
    result
}

/// Evaluate CLI `limit` command tail:
/// `<expr> [, <var> [, <direction> [, safe]]]`.
pub fn evaluate_limit_command_input(
    input: &str,
) -> Result<LimitCommandEvalOutput, LimitCommandEvalError> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err(LimitCommandEvalError::EmptyInput);
    }

    let parsed = crate::parse_limit_command_input(trimmed);
    match crate::json::eval_limit_from_str(
        parsed.expr,
        parsed.var,
        parsed.approach,
        parsed.presimplify,
    ) {
        Ok(limit_result) => Ok(LimitCommandEvalOutput {
            var: parsed.var.to_string(),
            approach: parsed.approach,
            result: limit_result.result,
            warning: limit_result.warning,
        }),
        Err(crate::json::LimitEvalError::Parse(message)) => {
            Err(LimitCommandEvalError::Parse(message))
        }
        Err(crate::json::LimitEvalError::Limit(message)) => {
            Err(LimitCommandEvalError::Limit(message))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        evaluate_equiv_input, evaluate_expand_log_input, evaluate_explain_gcd_input,
        evaluate_full_simplify_input, evaluate_limit_command_input, evaluate_rationalize_input,
        evaluate_substitute_and_simplify_input, evaluate_substitute_input,
        evaluate_telescope_input, evaluate_timeline_command_input,
        evaluate_timeline_simplify_input, evaluate_unary_function_input, evaluate_visualize_input,
        evaluate_weierstrass_input, expand_log_usage_message, expand_usage_message,
        extract_equiv_command_tail, extract_explain_command_tail, extract_limit_command_tail,
        extract_simplify_command_tail, extract_solve_command_tail, extract_substitute_command_tail,
        extract_timeline_command_tail, extract_unary_command_tail, extract_visualize_command_tail,
        format_equivalence_result_lines, format_expand_log_eval_lines,
        format_explain_error_message, format_explain_gcd_eval_lines,
        format_expr_pair_parse_error_message, format_limit_command_error_message,
        format_limit_command_eval_lines, format_rationalize_eval_error_message,
        format_rationalize_eval_lines, format_substitute_eval_lines,
        format_substitute_parse_error_message, format_telescope_eval_lines,
        format_timeline_command_error_message, format_timeline_eval_error_message,
        format_timeline_simplify_info_lines, format_transform_eval_error_message,
        format_unary_function_eval_error_message, format_unary_function_eval_lines,
        format_weierstrass_eval_lines, history_eval_metadata_section_labels, limit_usage_message,
        parse_expand_command_input, parse_expand_log_command_input,
        parse_rationalize_command_input, parse_telescope_command_input,
        parse_weierstrass_command_input, rationalize_usage_message,
        substitute_render_mode_from_display_mode, substitute_usage_message,
        telescope_usage_message, timeline_no_steps_message, timeline_open_hint_message,
        unary_render_config_for_display_mode, visualize_output_hint_lines,
        weierstrass_usage_message, wrap_expand_eval_expression, ExpandCommandInput,
        ExpandLogCommandInput, ExplainEvalError, FullSimplifyEvalError, LimitCommandEvalError,
        RationalizeCommandInput, RationalizeEvalOutcome, SubstituteRenderMode,
        TelescopeCommandInput, TimelineCommandEvalError, TimelineCommandEvalOutput,
        TimelineEvalError, TransformEvalError, UnaryFunctionEvalError, UnaryFunctionRenderConfig,
        WeierstrassCommandInput,
    };
    use crate::SubstituteOptions;
    use crate::{Engine, EquivalenceResult};
    use cas_formatter::DisplayExpr;
    use cas_session::SessionState;

    #[test]
    fn evaluate_equiv_input_true_for_basic_identity() {
        let mut s = crate::Simplifier::with_default_rules();
        let result = evaluate_equiv_input(&mut s, "x + x, 2*x").expect("equiv");
        assert!(matches!(
            result,
            EquivalenceResult::True | EquivalenceResult::ConditionalTrue { .. }
        ));
    }

    #[test]
    fn evaluate_equiv_input_reports_missing_delimiter() {
        let mut s = crate::Simplifier::with_default_rules();
        let err = evaluate_equiv_input(&mut s, "x+x").expect_err("missing delimiter");
        assert!(matches!(err, crate::ParseExprPairError::MissingDelimiter));
    }

    #[test]
    fn evaluate_substitute_input_runs_auto_strategy() {
        let mut s = crate::Simplifier::with_default_rules();
        let (subbed, _strategy) =
            evaluate_substitute_input(&mut s, "x^2 + x, x, 3", SubstituteOptions::default())
                .expect("subst");
        let (simplified, _steps) = s.simplify(subbed);
        let out = format!(
            "{}",
            DisplayExpr {
                context: &s.context,
                id: simplified
            }
        );
        assert_eq!(out, "12");
    }

    #[test]
    fn evaluate_substitute_and_simplify_input_runs() {
        let mut s = crate::Simplifier::with_default_rules();
        let out = evaluate_substitute_and_simplify_input(
            &mut s,
            "x^2 + x, x, 3",
            SubstituteOptions::default(),
        )
        .expect("subst");
        let rendered = format!(
            "{}",
            DisplayExpr {
                context: &s.context,
                id: out.simplified_expr
            }
        );
        assert_eq!(rendered, "12");
    }

    #[test]
    fn evaluate_unary_function_input_trace_works() {
        let mut s = crate::Simplifier::with_default_rules();
        let out = evaluate_unary_function_input(&mut s, "trace", "[[1,2],[3,4]]").expect("trace");
        let display = format!(
            "{}",
            DisplayExpr {
                context: &s.context,
                id: out.result_expr
            }
        );
        assert_eq!(display, "5");
    }

    #[test]
    fn evaluate_unary_function_input_parse_error() {
        let mut s = crate::Simplifier::with_default_rules();
        let err = evaluate_unary_function_input(&mut s, "det", "[[1,2]").expect_err("parse");
        assert!(matches!(err, UnaryFunctionEvalError::Parse(_)));
    }

    #[test]
    fn evaluate_rationalize_input_success() {
        let mut s = crate::Simplifier::with_default_rules();
        let out = evaluate_rationalize_input(&mut s, "1/(1 + sqrt(2))").expect("rationalize");
        match out.outcome {
            RationalizeEvalOutcome::Success { simplified_expr } => {
                let display = format!(
                    "{}",
                    DisplayExpr {
                        context: &s.context,
                        id: simplified_expr
                    }
                );
                assert_ne!(display, "1/(1 + sqrt(2))");
            }
            _ => panic!("expected success"),
        }
    }

    #[test]
    fn parse_rationalize_command_input_handles_missing_and_expr() {
        assert_eq!(
            parse_rationalize_command_input("rationalize"),
            RationalizeCommandInput::MissingInput
        );
        assert_eq!(
            parse_rationalize_command_input("rationalize x + 1"),
            RationalizeCommandInput::Expr("x + 1")
        );
    }

    #[test]
    fn format_rationalize_eval_error_message_parse() {
        let message = format_rationalize_eval_error_message(&crate::RationalizeEvalError::Parse(
            "bad input".to_string(),
        ));
        assert_eq!(message, "Parse error: bad input");
    }

    #[test]
    fn rationalize_usage_message_contains_usage() {
        assert!(rationalize_usage_message().contains("Usage: rationalize"));
    }

    #[test]
    fn format_rationalize_eval_lines_not_applicable_message() {
        let mut s = crate::Simplifier::with_default_rules();
        let out = evaluate_rationalize_input(&mut s, "1/x").expect("rationalize");
        let rendered = format_rationalize_eval_lines(&s.context, &out).join("\n");
        assert!(rendered.contains("Cannot rationalize"));
    }

    #[test]
    fn parse_expand_command_input_handles_missing_and_expr() {
        assert_eq!(
            parse_expand_command_input("expand"),
            ExpandCommandInput::MissingInput
        );
        assert_eq!(
            parse_expand_command_input("expand x*(x+1)"),
            ExpandCommandInput::Expr("x*(x+1)")
        );
    }

    #[test]
    fn parse_expand_log_command_input_handles_missing_and_expr() {
        assert_eq!(
            parse_expand_log_command_input("expand_log"),
            ExpandLogCommandInput::MissingInput
        );
        assert_eq!(
            parse_expand_log_command_input("expand_log ln(x*y)"),
            ExpandLogCommandInput::Expr("ln(x*y)")
        );
    }

    #[test]
    fn parse_telescope_command_input_handles_missing_and_expr() {
        assert_eq!(
            parse_telescope_command_input("telescope"),
            TelescopeCommandInput::MissingInput
        );
        assert_eq!(
            parse_telescope_command_input("telescope 1+2*cos(x)"),
            TelescopeCommandInput::Expr("1+2*cos(x)")
        );
    }

    #[test]
    fn parse_weierstrass_command_input_handles_missing_and_expr() {
        assert_eq!(
            parse_weierstrass_command_input("weierstrass"),
            WeierstrassCommandInput::MissingInput
        );
        assert_eq!(
            parse_weierstrass_command_input("weierstrass sin(x)+cos(x)"),
            WeierstrassCommandInput::Expr("sin(x)+cos(x)")
        );
    }

    #[test]
    fn extract_command_tails_match_expected_shapes() {
        assert_eq!(extract_equiv_command_tail("equiv x+1, y+2"), "x+1, y+2");
        assert_eq!(
            extract_substitute_command_tail("subst x^2, x, y"),
            "x^2, x, y"
        );
        assert_eq!(
            extract_timeline_command_tail("timeline solve x+1=2, x"),
            "solve x+1=2, x"
        );
        assert_eq!(
            extract_explain_command_tail("explain gcd(8,12)"),
            "gcd(8,12)"
        );
        assert_eq!(extract_solve_command_tail("solve x+1=2, x"), "x+1=2, x");
        assert_eq!(extract_simplify_command_tail("simplify (x+x)"), "(x+x)");
        assert_eq!(
            extract_limit_command_tail("limit x^2, x, infinity"),
            "x^2, x, infinity"
        );
        assert_eq!(extract_visualize_command_tail("visualize x+1"), "x+1");
        assert_eq!(extract_visualize_command_tail("viz x+1"), "x+1");
        assert_eq!(
            extract_unary_command_tail("trace [[1,2],[3,4]]", "trace"),
            "[[1,2],[3,4]]"
        );
    }

    #[test]
    fn usage_messages_and_expand_wrapper_match_expected_shape() {
        assert!(expand_usage_message().contains("Usage: expand"));
        assert!(expand_log_usage_message().contains("Usage: expand_log"));
        assert!(telescope_usage_message().contains("Usage: telescope"));
        assert_eq!(wrap_expand_eval_expression("x+1"), "expand(x+1)");
    }

    #[test]
    fn substitute_render_mode_from_display_mode_maps_all_variants() {
        assert_eq!(
            substitute_render_mode_from_display_mode(crate::set_command::SetDisplayMode::None),
            SubstituteRenderMode::None
        );
        assert_eq!(
            substitute_render_mode_from_display_mode(crate::set_command::SetDisplayMode::Succinct),
            SubstituteRenderMode::Succinct
        );
        assert_eq!(
            substitute_render_mode_from_display_mode(crate::set_command::SetDisplayMode::Normal),
            SubstituteRenderMode::Normal
        );
        assert_eq!(
            substitute_render_mode_from_display_mode(crate::set_command::SetDisplayMode::Verbose),
            SubstituteRenderMode::Verbose
        );
    }

    #[test]
    fn unary_render_config_for_display_mode_maps_steps_flag() {
        let none = unary_render_config_for_display_mode(
            "det",
            crate::set_command::SetDisplayMode::None,
            true,
        );
        assert_eq!(
            none,
            UnaryFunctionRenderConfig {
                function_name: "det",
                show_steps: false,
                show_step_assumptions: true
            }
        );

        let verbose = unary_render_config_for_display_mode(
            "det",
            crate::set_command::SetDisplayMode::Verbose,
            false,
        );
        assert_eq!(
            verbose,
            UnaryFunctionRenderConfig {
                function_name: "det",
                show_steps: true,
                show_step_assumptions: false
            }
        );
    }

    #[test]
    fn evaluate_weierstrass_input_produces_substitution() {
        let mut s = crate::Simplifier::with_default_rules();
        let out = evaluate_weierstrass_input(&mut s, "sin(x)").expect("weierstrass");
        let parsed = format!(
            "{}",
            DisplayExpr {
                context: &s.context,
                id: out.parsed_expr
            }
        );
        let substituted = format!(
            "{}",
            DisplayExpr {
                context: &s.context,
                id: out.substituted_expr
            }
        );
        assert_eq!(parsed, "sin(x)");
        assert_ne!(substituted, parsed);
    }

    #[test]
    fn evaluate_expand_log_input_expands_simple_log() {
        let mut s = crate::Simplifier::with_default_rules();
        let out = evaluate_expand_log_input(&mut s, "ln(x*y)").expect("expand_log");
        let display = format!(
            "{}",
            DisplayExpr {
                context: &s.context,
                id: out.expanded_expr
            }
        );
        assert!(display.contains("ln(x)") && display.contains("ln(y)"));
    }

    #[test]
    fn evaluate_telescope_input_reports_steps() {
        let mut s = crate::Simplifier::with_default_rules();
        let out = evaluate_telescope_input(&mut s, "1 + 2*cos(x)").expect("telescope");
        assert!(!out.formatted_result.trim().is_empty());
    }

    #[test]
    fn evaluate_transform_inputs_parse_error() {
        let mut s = crate::Simplifier::with_default_rules();
        let err = evaluate_expand_log_input(&mut s, "ln(").expect_err("parse");
        assert!(matches!(err, TransformEvalError::Parse(_)));
    }

    #[test]
    fn evaluate_visualize_input_emits_dot() {
        let mut s = crate::Simplifier::with_default_rules();
        let out = evaluate_visualize_input(&mut s, "x + 1").expect("viz");
        assert!(out.dot.contains("digraph"));
    }

    #[test]
    fn evaluate_timeline_simplify_input_stateful_runs() {
        let mut engine = Engine::new();
        let mut session = SessionState::new();
        let out =
            evaluate_timeline_simplify_input(&mut engine, &mut session, "x + x").expect("timeline");
        let rendered = format!(
            "{}",
            DisplayExpr {
                context: &engine.simplifier.context,
                id: out.simplified_expr
            }
        );
        assert_eq!(rendered, "2 * x");
    }

    #[test]
    fn evaluate_timeline_simplify_input_parse_error() {
        let mut engine = Engine::new();
        let mut session = SessionState::new();
        let err =
            evaluate_timeline_simplify_input(&mut engine, &mut session, "x +").expect_err("parse");
        assert!(matches!(err, TimelineEvalError::Parse(_)));
    }

    #[test]
    fn evaluate_timeline_command_input_dispatches_simplify() {
        let mut engine = Engine::new();
        let mut session = SessionState::new();
        let eval_options = session.options().clone();
        let out =
            evaluate_timeline_command_input(&mut engine, &mut session, "x + x", &eval_options)
                .expect("timeline command");
        match out {
            TimelineCommandEvalOutput::Simplify(simplify_out) => {
                assert_eq!(simplify_out.expr_input, "x + x");
                assert!(!simplify_out.steps.is_empty());
            }
            TimelineCommandEvalOutput::Solve(_) => panic!("expected simplify branch"),
        }
    }

    #[test]
    fn evaluate_timeline_command_input_dispatches_solve() {
        let mut engine = Engine::new();
        let mut session = SessionState::new();
        let eval_options = session.options().clone();
        let out = evaluate_timeline_command_input(
            &mut engine,
            &mut session,
            "solve x + 2 = 5, x",
            &eval_options,
        )
        .expect("timeline solve command");
        assert!(matches!(out, TimelineCommandEvalOutput::Solve(_)));
    }

    #[test]
    fn format_timeline_command_error_message_simplify_parse() {
        let msg = format_timeline_command_error_message(&TimelineCommandEvalError::Simplify(
            TimelineEvalError::Parse("bad".to_string()),
        ));
        assert_eq!(msg, "Parse error: bad");
    }

    #[test]
    fn evaluate_full_simplify_input_runs() {
        let mut engine = Engine::new();
        let session = SessionState::new();
        let out = evaluate_full_simplify_input(&mut engine, &session, "x + x", true).expect("ok");
        let rendered = format!(
            "{}",
            DisplayExpr {
                context: &engine.simplifier.context,
                id: out.simplified_expr
            }
        );
        assert_eq!(rendered, "2 * x");
    }

    #[test]
    fn evaluate_full_simplify_input_parse_error() {
        let mut engine = Engine::new();
        let session = SessionState::new();
        let err =
            evaluate_full_simplify_input(&mut engine, &session, "x +", true).expect_err("parse");
        assert!(matches!(err, FullSimplifyEvalError::Parse(_)));
    }

    #[test]
    fn evaluate_limit_command_input_runs() {
        let out = evaluate_limit_command_input("(x^2+1)/(2*x^2-3), x, infinity").expect("limit");
        assert_eq!(out.var, "x");
        assert!(!out.result.is_empty());
    }

    #[test]
    fn evaluate_limit_command_input_empty_error() {
        let err = evaluate_limit_command_input("  ").expect_err("empty");
        assert_eq!(err, LimitCommandEvalError::EmptyInput);
    }

    #[test]
    fn evaluate_explain_gcd_input_runs() {
        let mut s = crate::Simplifier::with_default_rules();
        let out = evaluate_explain_gcd_input(&mut s, "gcd(48, 18)").expect("explain");
        assert!(!out.steps.is_empty());
    }

    #[test]
    fn format_explain_gcd_eval_lines_contains_sections() {
        let mut s = crate::Simplifier::with_default_rules();
        let out = evaluate_explain_gcd_input(&mut s, "gcd(48, 18)").expect("explain");
        let lines = format_explain_gcd_eval_lines(&s.context, "gcd(48, 18)", &out);
        assert!(lines.iter().any(|line| line == "Educational Steps:"));
        assert!(lines
            .iter()
            .any(|line| line.starts_with("Result: ") || line == "Could not compute GCD"));
    }

    #[test]
    fn evaluate_explain_gcd_input_errors_for_non_function() {
        let mut s = crate::Simplifier::with_default_rules();
        let err = evaluate_explain_gcd_input(&mut s, "x + 1").expect_err("not function");
        assert_eq!(err, ExplainEvalError::NotFunctionCall);
    }

    #[test]
    fn substitute_usage_message_contains_examples() {
        let usage = substitute_usage_message();
        assert!(usage.contains("subst x^2 + x, x, 3"));
    }

    #[test]
    fn weierstrass_usage_message_contains_example() {
        let usage = weierstrass_usage_message();
        assert!(usage.contains("weierstrass sin(x) + cos(x)"));
    }

    #[test]
    fn limit_usage_message_contains_examples() {
        let usage = limit_usage_message();
        assert!(usage.contains("limit (x^2+1)/(2*x^2-3), x"));
    }

    #[test]
    fn format_substitute_eval_lines_supports_succinct_mode() {
        let mut s = crate::Simplifier::with_default_rules();
        let out = evaluate_substitute_and_simplify_input(
            &mut s,
            "x^2 + x, x, 3",
            SubstituteOptions::default(),
        )
        .expect("subst");
        let lines = format_substitute_eval_lines(
            &s.context,
            "x^2 + x, x, 3",
            &out,
            SubstituteRenderMode::Succinct,
        );
        assert!(lines
            .iter()
            .any(|line| line.starts_with("Variable substitution:")));
        assert!(lines.iter().any(|line| line.starts_with("Result: ")));
    }

    #[test]
    fn format_weierstrass_eval_lines_contains_result() {
        let mut s = crate::Simplifier::with_default_rules();
        let out = evaluate_weierstrass_input(&mut s, "sin(x)").expect("weierstrass");
        let lines = format_weierstrass_eval_lines(&s.context, "sin(x)", &out);
        assert!(lines
            .iter()
            .any(|line| line.contains("Weierstrass substitution")));
        assert!(lines.iter().any(|line| line.starts_with("Result: ")));
    }

    #[test]
    fn format_limit_command_eval_lines_includes_warning() {
        let lines = format_limit_command_eval_lines(&super::LimitCommandEvalOutput {
            var: "x".to_string(),
            approach: crate::Approach::PosInfinity,
            result: "1/2".to_string(),
            warning: Some("unstable".to_string()),
        });
        assert_eq!(lines[0], "lim_{x→+∞} = 1/2");
        assert_eq!(lines[1], "Warning: unstable");
    }

    #[test]
    fn format_limit_command_error_message_empty_uses_usage() {
        let message = format_limit_command_error_message(&LimitCommandEvalError::EmptyInput);
        assert!(message.contains("Usage: limit <expr>"));
    }

    #[test]
    fn format_substitute_parse_error_message_maps_expression() {
        let msg = format_substitute_parse_error_message(
            &crate::ParseSubstituteArgsError::Expression("bad".to_string()),
        );
        assert_eq!(msg, "Error parsing expression: bad");
    }

    #[test]
    fn format_expr_pair_parse_error_message_maps_missing_delimiter() {
        let msg = format_expr_pair_parse_error_message(
            &crate::ParseExprPairError::MissingDelimiter,
            "equiv",
        );
        assert_eq!(msg, "Usage: equiv <expr1>, <expr2>");
    }

    #[test]
    fn format_equivalence_result_lines_handles_unknown() {
        let lines = format_equivalence_result_lines(&crate::EquivalenceResult::Unknown);
        assert_eq!(
            lines,
            vec!["Unknown (cannot prove equivalence)".to_string()]
        );
    }

    #[test]
    fn format_timeline_eval_error_message_maps_parse() {
        let msg = format_timeline_eval_error_message(&TimelineEvalError::Parse("x".to_string()));
        assert_eq!(msg, "Parse error: x");
    }

    #[test]
    fn timeline_no_steps_message_is_stable() {
        assert_eq!(
            timeline_no_steps_message(),
            "No simplification steps to visualize."
        );
    }

    #[test]
    fn timeline_open_hint_message_is_stable() {
        assert_eq!(
            timeline_open_hint_message(),
            "Open in browser to view interactive visualization."
        );
    }

    #[test]
    fn format_timeline_simplify_info_lines_aggressive() {
        let lines = format_timeline_simplify_info_lines(true);
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0], "(Aggressive simplification mode)");
        assert_eq!(
            lines[1],
            "Open in browser to view interactive visualization."
        );
    }

    #[test]
    fn history_eval_metadata_section_labels_match_cli_convention() {
        let labels = history_eval_metadata_section_labels();
        assert_eq!(labels.required_header, "  ℹ️ Requires:");
        assert_eq!(labels.assumed_header, "  ⚠ Assumed:");
        assert_eq!(labels.blocked_header, "  🚫 Blocked:");
        assert_eq!(labels.line_prefix, "    - ");
    }

    #[test]
    fn visualize_output_hint_lines_has_two_commands() {
        let lines = visualize_output_hint_lines();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].starts_with("Render with: dot -Tsvg"));
        assert!(lines[1].starts_with("Or: dot -Tpng"));
    }

    #[test]
    fn format_unary_function_eval_lines_include_steps_when_enabled() {
        let mut s = crate::Simplifier::with_default_rules();
        let out = evaluate_unary_function_input(&mut s, "trace", "[[1,2],[3,4]]").expect("trace");
        let lines = format_unary_function_eval_lines(
            &s.context,
            "[[1,2],[3,4]]",
            &out,
            UnaryFunctionRenderConfig {
                function_name: "trace",
                show_steps: true,
                show_step_assumptions: true,
            },
        );
        assert!(lines.iter().any(|line| line.starts_with("Parsed: trace(")));
        assert!(lines.iter().any(|line| line == "Steps:"));
        assert!(lines.iter().any(|line| line.starts_with("Result: ")));
    }

    #[test]
    fn format_unary_function_eval_error_message_parse() {
        let msg = format_unary_function_eval_error_message(&UnaryFunctionEvalError::Parse(
            "bad input".to_string(),
        ));
        assert_eq!(msg, "Parse error: bad input");
    }

    #[test]
    fn format_expand_log_eval_lines_returns_parsed_and_result() {
        let mut s = crate::Simplifier::with_default_rules();
        let out = evaluate_expand_log_input(&mut s, "ln(x*y)").expect("expand_log");
        let lines = format_expand_log_eval_lines(&s.context, &out);
        assert_eq!(lines.len(), 2);
        assert!(lines[0].starts_with("Parsed: "));
        assert!(lines[1].starts_with("Result: "));
    }

    #[test]
    fn format_telescope_eval_lines_returns_single_block() {
        let mut s = crate::Simplifier::with_default_rules();
        let out = evaluate_telescope_input(&mut s, "1 + 2*cos(x)").expect("telescope");
        let lines = format_telescope_eval_lines("1 + 2*cos(x)", &out);
        assert_eq!(lines.len(), 1);
        assert!(lines[0].starts_with("Parsed: 1 + 2*cos(x)"));
    }

    #[test]
    fn format_transform_eval_error_message_parse() {
        let msg =
            format_transform_eval_error_message(&TransformEvalError::Parse("oops".to_string()));
        assert_eq!(msg, "Parse error: oops");
    }

    #[test]
    fn format_explain_error_message_maps_unsupported_function() {
        let msg =
            format_explain_error_message(&ExplainEvalError::UnsupportedFunction("sin".to_string()));
        assert!(msg.contains("not yet implemented for function 'sin'"));
    }
}
