//! JSON DTOs shared by CLI/FFI layers.
//!
//! This crate intentionally keeps transport models independent from engine internals.

use serde::{Deserialize, Serialize};

/// Stable schema version for JSON outputs.
pub const SCHEMA_VERSION: u8 = 1;

fn serialization_fallback(pretty: bool, message: &str) -> String {
    let fallback = serde_json::json!({
        "schema_version": SCHEMA_VERSION,
        "ok": false,
        "error": {
            "kind": "InternalError",
            "code": "E_INTERNAL",
            "message": message,
        }
    });

    if pretty {
        serde_json::to_string_pretty(&fallback)
    } else {
        serde_json::to_string(&fallback)
    }
    .unwrap_or_else(|_| {
        if pretty {
            format!(
                "{{\n  \"schema_version\": {},\n  \"ok\": false,\n  \"error\": {{\"kind\": \"InternalError\", \"code\": \"E_INTERNAL\", \"message\": \"JSON serialization failed\"}}\n}}",
                SCHEMA_VERSION
            )
        } else {
            format!(
                r#"{{"schema_version":{},"ok":false,"error":{{"kind":"InternalError","code":"E_INTERNAL","message":"JSON serialization failed"}}}}"#,
                SCHEMA_VERSION
            )
        }
    })
}

fn serialize_json<T: Serialize>(value: &T, pretty: bool) -> String {
    let encoded = if pretty {
        serde_json::to_string_pretty(value)
    } else {
        serde_json::to_string(value)
    };

    encoded.unwrap_or_else(|e| {
        serialization_fallback(pretty, &format!("JSON serialization failed: {e}"))
    })
}

// =============================================================================
// Shared Eval JSON types
// =============================================================================

/// Expression statistics (node count, depth).
#[derive(Serialize, Debug, Default)]
pub struct ExprStatsJson {
    pub node_count: usize,
    pub depth: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub term_count: Option<usize>,
}

/// Timing breakdown in microseconds.
#[derive(Serialize, Debug, Default)]
pub struct TimingsJson {
    pub parse_us: u64,
    pub simplify_us: u64,
    pub total_us: u64,
}

/// Domain mode information.
#[derive(Serialize, Debug, Default)]
pub struct DomainJson {
    pub mode: String,
}

impl DomainJson {
    pub fn from_mode(mode: impl Into<String>) -> Self {
        Self { mode: mode.into() }
    }
}

/// Options used for evaluation.
#[derive(Serialize, Debug, Default)]
pub struct OptionsJson {
    pub context_mode: String,
    pub branch_mode: String,
    pub expand_policy: String,
    pub complex_mode: String,
    pub steps_mode: String,
    pub domain_mode: String,
    pub const_fold: String,
}

impl OptionsJson {
    #[allow(clippy::too_many_arguments)]
    pub fn from_eval_axes(
        context_mode: impl Into<String>,
        branch_mode: impl Into<String>,
        expand_policy: impl Into<String>,
        complex_mode: impl Into<String>,
        steps_mode: impl Into<String>,
        domain_mode: impl Into<String>,
        const_fold: impl Into<String>,
    ) -> Self {
        Self {
            context_mode: context_mode.into(),
            branch_mode: branch_mode.into(),
            expand_policy: expand_policy.into(),
            complex_mode: complex_mode.into(),
            steps_mode: steps_mode.into(),
            domain_mode: domain_mode.into(),
            const_fold: const_fold.into(),
        }
    }
}

/// Complete semantics configuration in JSON output.
#[derive(Serialize, Debug, Default)]
pub struct SemanticsJson {
    pub domain_mode: String,
    pub value_domain: String,
    pub branch: String,
    pub inv_trig: String,
    pub assume_scope: String,
}

impl SemanticsJson {
    pub fn from_eval_axes(
        domain_mode: impl Into<String>,
        value_domain: impl Into<String>,
        branch: impl Into<String>,
        inv_trig: impl Into<String>,
        assume_scope: impl Into<String>,
    ) -> Self {
        Self {
            domain_mode: domain_mode.into(),
            value_domain: value_domain.into(),
            branch: branch.into(),
            inv_trig: inv_trig.into(),
            assume_scope: assume_scope.into(),
        }
    }
}

/// A required condition (implicit domain constraint) from the input expression.
#[derive(Serialize, Debug, Clone)]
pub struct RequiredConditionJson {
    pub kind: String,
    pub expr_display: String,
    pub expr_canonical: String,
}

/// A simplification step for JSON output.
#[derive(Serialize, Debug, Clone)]
pub struct StepJson {
    pub index: usize,
    pub rule: String,
    pub rule_latex: String,
    pub before: String,
    pub after: String,
    pub before_latex: String,
    pub after_latex: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub substeps: Vec<SubStepJson>,
}

/// A sub-step within a step for detailed explanations.
#[derive(Serialize, Debug, Clone)]
pub struct SubStepJson {
    pub title: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub lines: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub before_latex: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub after_latex: Option<String>,
}

/// A solver step for equation-solving JSON output.
#[derive(Serialize, Debug, Clone)]
pub struct SolveStepJson {
    pub index: usize,
    pub description: String,
    pub equation: String,
    pub lhs_latex: String,
    pub relop: String,
    pub rhs_latex: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub substeps: Vec<SolveSubStepJson>,
}

/// A sub-step within a solve step for detailed derivation.
#[derive(Serialize, Debug, Clone)]
pub struct SolveSubStepJson {
    pub index: String,
    pub description: String,
    pub equation: String,
    pub lhs_latex: String,
    pub relop: String,
    pub rhs_latex: String,
}

/// A domain assumption warning with its source rule.
#[derive(Serialize, Debug, Clone)]
pub struct WarningJson {
    pub rule: String,
    pub assumption: String,
}

/// Warning payload used by engine-style response envelopes.
#[derive(Serialize, Debug, Clone)]
pub struct EngineJsonWarning {
    pub kind: String,
    pub message: String,
}

/// Stable budget information for JSON responses.
#[derive(Serialize, Debug, Default, Clone)]
pub struct BudgetJsonInfo {
    pub preset: String,
    pub mode: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exceeded: Option<BudgetExceededJson>,
}

impl BudgetJsonInfo {
    pub fn new(preset: impl Into<String>, strict: bool) -> Self {
        Self {
            preset: preset.into(),
            mode: if strict {
                "strict".to_string()
            } else {
                "best-effort".to_string()
            },
            exceeded: None,
        }
    }

    pub fn cli(strict: bool) -> Self {
        Self::new("cli", strict)
    }

    pub fn small(strict: bool) -> Self {
        Self::new("small", strict)
    }
}

/// Backward-compatible alias.
pub type BudgetJson = BudgetJsonInfo;

/// Budget exceeded details.
#[derive(Serialize, Debug, Clone)]
pub struct BudgetExceededJson {
    pub op: String,
    pub metric: String,
    pub used: u64,
    pub limit: u64,
}

/// An error result with stable kind/code for API consumers.
#[derive(Serialize, Debug)]
pub struct ErrorJsonOutput {
    pub schema_version: u8,
    pub ok: bool,
    pub kind: String,
    pub code: String,
    pub error: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<String>,
}

impl ErrorJsonOutput {
    pub fn new(error: impl Into<String>) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            ok: false,
            kind: "InternalError".to_string(),
            code: "E_INTERNAL".to_string(),
            error: error.into(),
            input: None,
        }
    }

    pub fn with_input(error: impl Into<String>, input: impl Into<String>) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            ok: false,
            kind: "InternalError".to_string(),
            code: "E_INTERNAL".to_string(),
            error: error.into(),
            input: Some(input.into()),
        }
    }

    pub fn parse_error(message: impl Into<String>, input: Option<String>) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            ok: false,
            kind: "ParseError".to_string(),
            code: "E_PARSE".to_string(),
            error: message.into(),
            input,
        }
    }

    pub fn to_json(&self) -> String {
        serialize_json(self, false)
    }

    pub fn to_json_pretty(&self) -> String {
        serialize_json(self, true)
    }

    pub fn from_eval_error_message(error: &str, input: &str) -> Self {
        if error.starts_with("Parse error:") {
            Self::parse_error(error, Some(input.to_string()))
        } else {
            Self::with_input(error, input)
        }
    }
}

/// Result of evaluating a single expression via eval-json.
#[derive(Serialize, Debug)]
pub struct EvalJsonOutput {
    pub schema_version: u8,
    pub ok: bool,
    pub input: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_latex: Option<String>,
    pub result: String,
    pub result_truncated: bool,
    pub result_chars: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result_latex: Option<String>,
    pub steps_mode: String,
    pub steps_count: usize,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub steps: Vec<StepJson>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub solve_steps: Vec<SolveStepJson>,
    pub warnings: Vec<WarningJson>,
    pub required_conditions: Vec<RequiredConditionJson>,
    pub required_display: Vec<String>,
    pub budget: BudgetJsonInfo,
    pub domain: DomainJson,
    pub stats: ExprStatsJson,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hash: Option<String>,
    pub timings_us: TimingsJson,
    pub options: OptionsJson,
    pub semantics: SemanticsJson,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wire: Option<serde_json::Value>,
}

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

impl EvalJsonOutput {
    pub fn from_build(parts: EvalJsonOutputBuild<'_>) -> Self {
        Self {
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
            budget: BudgetJsonInfo::new(parts.budget_preset, parts.strict),
            domain: DomainJson::from_mode(parts.domain),
            stats: parts.stats,
            hash: parts.hash,
            timings_us: parts.timings_us,
            options: OptionsJson::from_eval_axes(
                parts.context_mode,
                parts.branch_mode,
                parts.expand_policy,
                parts.complex_mode,
                parts.steps_mode,
                parts.domain,
                parts.const_fold,
            ),
            semantics: SemanticsJson::from_eval_axes(
                parts.domain,
                parts.value_domain,
                parts.complex_branch,
                parts.inv_trig,
                parts.assume_scope,
            ),
            wire: parts.wire,
        }
    }

    pub fn to_json(&self) -> String {
        serialize_json(self, false)
    }

    pub fn to_json_pretty(&self) -> String {
        serialize_json(self, true)
    }
}

/// Limit approach used by parsed eval special commands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvalLimitApproach {
    PosInfinity,
    NegInfinity,
}

/// Parsed special command forms accepted by eval input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvalSpecialCommand {
    Solve {
        equation: String,
        var: String,
    },
    Limit {
        expr: String,
        var: String,
        approach: EvalLimitApproach,
    },
}

/// Parse special eval command forms:
/// - `solve(equation, var)`
/// - `limit(expr, var, approach)` or `lim(expr, var, approach)`
pub fn parse_eval_special_command(input: &str) -> Option<EvalSpecialCommand> {
    if let Some((equation, var)) = parse_solve_command(input) {
        return Some(EvalSpecialCommand::Solve { equation, var });
    }
    if let Some((expr, var, approach)) = parse_limit_command(input) {
        return Some(EvalSpecialCommand::Limit {
            expr,
            var,
            approach,
        });
    }
    None
}

fn parse_solve_command(input: &str) -> Option<(String, String)> {
    let trimmed = input.trim();
    if !trimmed.to_lowercase().starts_with("solve(") || !trimmed.ends_with(')') {
        return None;
    }

    let content = &trimmed[6..trimmed.len() - 1];
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

fn parse_limit_command(input: &str) -> Option<(String, String, EvalLimitApproach)> {
    let trimmed = input.trim();
    let lower = trimmed.to_lowercase();
    let prefix_len = if lower.starts_with("limit(") {
        6
    } else if lower.starts_with("lim(") {
        4
    } else {
        return None;
    };

    if !trimmed.ends_with(')') {
        return None;
    }

    let content = &trimmed[prefix_len..trimmed.len() - 1];
    let parts = split_by_comma_at_depth_0(content);
    if parts.len() < 2 || parts.len() > 3 {
        return None;
    }

    let expr_str = parts[0].trim();
    let var_str = parts[1].trim();
    if var_str.is_empty() || !var_str.chars().next()?.is_alphabetic() {
        return None;
    }
    if !var_str.chars().all(|c| c.is_alphanumeric() || c == '_') {
        return None;
    }

    let approach = if parts.len() == 3 {
        match parts[2].trim().to_lowercase().as_str() {
            "inf" | "infinity" | "+inf" | "+infinity" => EvalLimitApproach::PosInfinity,
            "-inf" | "-infinity" => EvalLimitApproach::NegInfinity,
            _ => return None,
        }
    } else {
        EvalLimitApproach::PosInfinity
    };

    Some((expr_str.to_string(), var_str.to_string(), approach))
}

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

    result.push(&s[start..]);
    result
}

/// Configuration for session-backed eval execution.
#[derive(Debug, Clone, Copy)]
pub struct EvalSessionRunConfig<'a> {
    pub expr: &'a str,
    pub auto_store: bool,
    pub max_chars: usize,
    pub steps_mode: &'a str,
    pub budget_preset: &'a str,
    pub strict: bool,
    pub domain: &'a str,
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

// =============================================================================
// Engine-style response DTOs (used by FFI fallback paths)
// =============================================================================

/// Source span for JSON serialization.
#[derive(Serialize, Debug, Clone, Copy)]
pub struct SpanJson {
    pub start: usize,
    pub end: usize,
}

/// A sub-step representing a rewrite in a subexpression.
#[derive(Serialize, Debug, Clone)]
pub struct EngineJsonSubstep {
    pub rule: String,
    pub before: String,
    pub after: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

/// A simplification step in a response.
#[derive(Serialize, Debug, Clone)]
pub struct EngineJsonStep {
    pub phase: String,
    pub rule: String,
    pub before: String,
    pub after: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub substeps: Vec<EngineJsonSubstep>,
}

/// Assumption summary record for response payloads.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssumptionRecord {
    pub kind: String,
    pub expr: String,
    pub message: String,
    pub count: u32,
}

/// Structured error in a response.
#[derive(Serialize, Debug, Clone)]
pub struct EngineJsonError {
    pub kind: &'static str,
    pub code: &'static str,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub span: Option<SpanJson>,
    #[serde(default)]
    pub details: serde_json::Value,
}

impl EngineJsonError {
    pub fn simple(kind: &'static str, code: &'static str, message: impl Into<String>) -> Self {
        Self {
            kind,
            code,
            message: message.into(),
            span: None,
            details: serde_json::Value::Null,
        }
    }

    pub fn parse(message: impl Into<String>, span: Option<SpanJson>) -> Self {
        Self {
            kind: "ParseError",
            code: "E_PARSE",
            message: message.into(),
            span,
            details: serde_json::Value::Null,
        }
    }

    pub fn invalid_input(message: impl Into<String>, details: serde_json::Value) -> Self {
        Self {
            kind: "InvalidInput",
            code: "E_INVALID_INPUT",
            message: message.into(),
            span: None,
            details,
        }
    }

    pub fn session_ref_not_supported_for_stateless_eval() -> Self {
        Self::invalid_input(
            "Session references (#N) are not supported in stateless eval_json mode",
            serde_json::json!({
                "hint": "Use stateful Engine::eval with an EvalSession for #N references"
            }),
        )
    }

    pub fn invalid_options_json(error: impl Into<String>) -> Self {
        let error = error.into();
        Self::invalid_input(
            format!("Invalid options JSON: {error}"),
            serde_json::json!({ "error": error }),
        )
    }

    pub fn from_eval_runtime_error(message: impl Into<String>) -> Self {
        let message = message.into();
        if message.contains("requires stateful eval") {
            Self::session_ref_not_supported_for_stateless_eval()
        } else {
            Self::simple("InternalError", "E_INTERNAL", message)
        }
    }
}

/// Unified JSON response for engine-like operations.
#[derive(Serialize, Debug, Clone)]
pub struct EngineJsonResponse {
    pub schema_version: u8,
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<EngineJsonError>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub steps: Vec<EngineJsonStep>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<EngineJsonWarning>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub assumptions: Vec<AssumptionRecord>,
    pub budget: BudgetJsonInfo,
}

impl EngineJsonResponse {
    pub fn ok(result: String, budget: BudgetJsonInfo) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            ok: true,
            result: Some(result),
            error: None,
            steps: vec![],
            warnings: vec![],
            assumptions: vec![],
            budget,
        }
    }

    pub fn ok_with_steps(
        result: String,
        steps: Vec<EngineJsonStep>,
        budget: BudgetJsonInfo,
    ) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            ok: true,
            result: Some(result),
            error: None,
            steps,
            warnings: vec![],
            assumptions: vec![],
            budget,
        }
    }

    pub fn err(error: EngineJsonError, budget: BudgetJsonInfo) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            ok: false,
            result: None,
            error: Some(error),
            steps: vec![],
            warnings: vec![],
            assumptions: vec![],
            budget,
        }
    }

    pub fn invalid_options_json(error: impl Into<String>) -> Self {
        Self::err(
            EngineJsonError::invalid_options_json(error),
            BudgetJsonInfo::new("unknown", true),
        )
    }

    pub fn with_warning(mut self, warning: EngineJsonWarning) -> Self {
        self.warnings.push(warning);
        self
    }

    pub fn to_json(&self) -> String {
        self.to_json_with_pretty(false)
    }

    pub fn to_json_pretty(&self) -> String {
        self.to_json_with_pretty(true)
    }

    pub fn to_json_with_pretty(&self, pretty: bool) -> String {
        serialize_json(self, pretty)
    }
}

// =============================================================================
// Engine JSON input options
// =============================================================================

/// Options for JSON evaluation input.
#[derive(Deserialize, Debug, Default, Clone)]
pub struct JsonRunOptions {
    /// Budget configuration
    #[serde(default)]
    pub budget: BudgetOpts,
    /// Include simplification steps in output
    #[serde(default)]
    pub steps: bool,
    /// Pretty-print JSON output
    #[serde(default)]
    pub pretty: bool,
}

/// Budget options in JSON input.
#[derive(Deserialize, Debug, Clone)]
pub struct BudgetOpts {
    /// Preset name: "small", "cli", "unlimited"
    #[serde(default = "default_preset")]
    pub preset: String,
    /// Mode: "strict" or "best-effort"
    #[serde(default = "default_mode")]
    pub mode: String,
}

impl Default for BudgetOpts {
    fn default() -> Self {
        Self {
            preset: default_preset(),
            mode: default_mode(),
        }
    }
}

fn default_preset() -> String {
    "cli".to_string()
}

fn default_mode() -> String {
    "best-effort".to_string()
}

impl JsonRunOptions {
    pub fn requested_pretty(opts_json: &str) -> bool {
        opts_json.contains("\"pretty\":true")
    }
}

// =============================================================================
// substitute-json types
// =============================================================================

/// Options for substitute JSON operation.
#[derive(Deserialize, Debug, Clone)]
pub struct SubstituteJsonOptions {
    /// Substitution mode: "exact" or "power" (default: "power")
    #[serde(default = "default_substitute_mode")]
    pub mode: String,
    /// Include substitution steps in output
    #[serde(default)]
    pub steps: bool,
    /// Pretty-print JSON output
    #[serde(default)]
    pub pretty: bool,
}

impl Default for SubstituteJsonOptions {
    fn default() -> Self {
        Self {
            mode: "power".to_string(),
            steps: false,
            pretty: false,
        }
    }
}

fn default_substitute_mode() -> String {
    "power".to_string()
}

impl SubstituteJsonOptions {
    pub fn from_mode_flags(mode: &str, steps: bool, pretty: bool) -> Self {
        Self {
            mode: mode.to_string(),
            steps,
            pretty,
        }
    }

    pub fn parse_optional_json(opts_json: Option<&str>) -> Self {
        match opts_json {
            Some(json) => serde_json::from_str(json).unwrap_or_default(),
            None => Self::default(),
        }
    }
}

/// Request echo for substitute operations.
#[derive(Serialize, Debug, Clone)]
pub struct SubstituteRequestEcho {
    pub expr: String,
    pub target: String,
    #[serde(rename = "with")]
    pub with_expr: String,
}

/// Options echo for substitute operations.
#[derive(Serialize, Debug, Clone)]
pub struct SubstituteOptionsJson {
    pub substitute: SubstituteOptionsInner,
}

#[derive(Serialize, Debug, Clone)]
pub struct SubstituteOptionsInner {
    pub mode: String,
    pub steps: bool,
}

/// Substitute JSON response with request echo and options.
#[derive(Serialize, Debug, Clone)]
pub struct SubstituteJsonResponse {
    /// Schema version for API stability
    pub schema_version: u8,
    /// True if operation succeeded
    pub ok: bool,
    /// Result expression (success only)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<String>,
    /// Error details (failure only)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<EngineJsonError>,
    /// Request echo for reproducibility
    pub request: SubstituteRequestEcho,
    /// Options used
    pub options: SubstituteOptionsJson,
    /// Substitution steps (if requested)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub steps: Vec<EngineJsonSubstep>,
}

impl SubstituteJsonResponse {
    pub fn ok(
        result: String,
        request: SubstituteRequestEcho,
        options: SubstituteOptionsJson,
        steps: Vec<EngineJsonSubstep>,
    ) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            ok: true,
            result: Some(result),
            error: None,
            request,
            options,
            steps,
        }
    }

    pub fn err(
        error: EngineJsonError,
        request: SubstituteRequestEcho,
        options: SubstituteOptionsJson,
    ) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            ok: false,
            result: None,
            error: Some(error),
            request,
            options,
            steps: vec![],
        }
    }

    pub fn to_json(&self) -> String {
        self.to_json_with_pretty(false)
    }

    pub fn to_json_pretty(&self) -> String {
        self.to_json_with_pretty(true)
    }

    pub fn to_json_with_pretty(&self, pretty: bool) -> String {
        serialize_json(self, pretty)
    }
}

/// Substitution mode for typed non-JSON evaluation APIs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SubstituteEvalMode {
    Exact,
    Power,
}

impl SubstituteEvalMode {
    pub fn as_mode_str(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::Power => "power",
        }
    }
}

/// One substitution step for typed non-JSON evaluation APIs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SubstituteEvalStep {
    pub rule: String,
    pub before: String,
    pub after: String,
    pub note: Option<String>,
}

/// Result payload for typed non-JSON substitution evaluation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SubstituteEvalResult {
    pub result: String,
    pub steps: Vec<SubstituteEvalStep>,
}

/// Parse-time errors produced by substitute helpers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SubstituteEvalError {
    ParseExpression(String),
    ParseTarget(String),
    ParseReplacement(String),
}

impl std::fmt::Display for SubstituteEvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ParseExpression(message)
            | Self::ParseTarget(message)
            | Self::ParseReplacement(message) => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for SubstituteEvalError {}

/// Result payload for typed non-JSON limit evaluation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LimitEvalResult {
    pub result: String,
    pub warning: Option<String>,
}

/// Errors produced by typed non-JSON limit evaluation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LimitEvalError {
    Parse(String),
    Limit(String),
}

/// Canonical JSON response for limit evaluation.
#[derive(Serialize, Debug, Clone, PartialEq, Eq)]
pub struct LimitJsonResponse {
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warning: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<&'static str>,
}

impl LimitJsonResponse {
    pub fn ok(result: String, warning: Option<String>) -> Self {
        Self {
            ok: true,
            result: Some(result),
            warning,
            error: None,
            code: None,
        }
    }

    pub fn parse_error(message: impl Into<String>) -> Self {
        Self {
            ok: false,
            result: None,
            warning: None,
            error: Some(message.into()),
            code: Some("PARSE_ERROR"),
        }
    }

    pub fn limit_error(message: impl Into<String>) -> Self {
        Self {
            ok: false,
            result: None,
            warning: None,
            error: Some(message.into()),
            code: Some("LIMIT_ERROR"),
        }
    }

    pub fn internal_error(message: impl Into<String>) -> Self {
        Self {
            ok: false,
            result: None,
            warning: None,
            error: Some(message.into()),
            code: Some("INTERNAL_ERROR"),
        }
    }

    pub fn to_json(&self) -> String {
        self.to_json_with_pretty(false)
    }

    pub fn to_json_pretty(&self) -> String {
        self.to_json_with_pretty(true)
    }

    pub fn to_json_with_pretty(&self, pretty: bool) -> String {
        let serialized = if pretty {
            serde_json::to_string_pretty(self)
        } else {
            serde_json::to_string(self)
        };

        serialized.unwrap_or_else(|e| {
            let fallback = Self::internal_error(format!("JSON serialization failed: {}", e));
            serde_json::to_string(&fallback).unwrap_or_else(|_| {
                "{\"ok\":false,\"error\":\"JSON serialization failed\",\"code\":\"INTERNAL_ERROR\"}"
                    .to_string()
            })
        })
    }
}

// =============================================================================
// script-json types
// =============================================================================

/// Result of processing a script via script-json.
#[derive(Serialize, Debug)]
pub struct ScriptJsonOutput {
    pub ok: bool,
    pub lines: Vec<ScriptLineResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub final_result: Option<EvalJsonOutput>,
    pub total_time_us: u64,
}

/// Result of processing a single line in a script.
#[derive(Serialize, Debug)]
pub struct ScriptLineResult {
    pub line_no: usize,
    pub input: String,
    /// "command" | "let" | "expr" | "empty" | "error"
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<EvalJsonOutput>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

// =============================================================================
// mm-gcd-modp-json types
// =============================================================================

/// Result of running mm_gcd benchmark via mm-gcd-modp-json.
#[derive(Serialize, Debug)]
pub struct MmGcdModpJsonOutput {
    pub ok: bool,
    pub modulus: u64,
    pub a_terms: usize,
    pub b_terms: usize,
    pub g_terms: usize,
    pub ag_terms: usize,
    pub bg_terms: usize,
    pub gcd_terms: usize,
    pub gcd_total_degree: u32,
    pub gcd_matches_g: bool,
    pub timings_ms: MmGcdTimingsMs,
}

/// Timing breakdown for mm_gcd in milliseconds.
#[derive(Serialize, Debug, Default)]
pub struct MmGcdTimingsMs {
    pub build_ms: f64,
    pub mul_ms: f64,
    pub gcd_ms: f64,
    pub full_ms: f64,
}

// =============================================================================
// OutputEnvelope V1
// =============================================================================

/// Root envelope for all API responses (eval & solve).
#[derive(Serialize, Debug)]
pub struct OutputEnvelope {
    pub schema_version: u8,
    pub engine: EngineInfo,
    pub request: RequestInfo,
    pub result: ResultDto,
    pub transparency: TransparencyDto,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub steps: Vec<StepDto>,
}

/// Engine metadata.
#[derive(Serialize, Debug)]
pub struct EngineInfo {
    pub name: String,
    pub version: String,
}

impl Default for EngineInfo {
    fn default() -> Self {
        Self {
            name: "ExpliCAS".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }
}

/// Request information.
#[derive(Serialize, Debug)]
pub struct RequestInfo {
    pub kind: String,
    pub input: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub solve_var: Option<String>,
    pub options: RequestOptions,
}

/// Request options.
#[derive(Serialize, Debug, Default)]
pub struct RequestOptions {
    pub domain_mode: String,
    pub value_domain: String,
    pub hints: bool,
    pub explain: bool,
}

/// Options accepted by envelope eval entrypoints.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EnvelopeEvalOptions {
    pub domain: String,
    pub value_domain: String,
}

impl Default for EnvelopeEvalOptions {
    fn default() -> Self {
        Self {
            domain: "generic".to_string(),
            value_domain: "real".to_string(),
        }
    }
}

/// Expression with dual rendering.
#[derive(Serialize, Debug, Clone)]
pub struct ExprDto {
    pub display: String,
    pub canonical: String,
}

impl ExprDto {
    pub fn from_display(display: impl Into<String>) -> Self {
        let display = display.into();
        Self {
            display: display.clone(),
            canonical: display,
        }
    }
}

/// Result (polymorphic by kind).
#[derive(Serialize, Debug)]
#[serde(tag = "kind")]
pub enum ResultDto {
    #[serde(rename = "eval_result")]
    Eval { value: ExprDto },
    #[serde(rename = "solve_result")]
    Solve {
        solutions: SolutionSetDto,
        #[serde(skip_serializing_if = "Option::is_none")]
        residual: Option<ExprDto>,
    },
    #[serde(rename = "error")]
    Error { message: String },
}

/// Solution set (polymorphic by kind).
#[derive(Serialize, Debug)]
#[serde(tag = "kind")]
pub enum SolutionSetDto {
    #[serde(rename = "finite_set")]
    FiniteSet { elements: Vec<ExprDto> },
    #[serde(rename = "all_reals")]
    AllReals,
    #[serde(rename = "empty_set")]
    EmptySet,
    #[serde(rename = "interval")]
    Interval { lower: BoundDto, upper: BoundDto },
    #[serde(rename = "conditional")]
    Conditional { cases: Vec<CaseDto> },
}

/// Interval bound.
#[derive(Serialize, Debug)]
#[serde(tag = "kind")]
pub enum BoundDto {
    #[serde(rename = "closed")]
    Closed { value: ExprDto },
    #[serde(rename = "open")]
    Open { value: ExprDto },
    #[serde(rename = "neg_infinity")]
    NegInfinity,
    #[serde(rename = "infinity")]
    Infinity,
}

/// Conditional case.
#[derive(Serialize, Debug)]
pub struct CaseDto {
    pub when: WhenDto,
    pub then: ThenDto,
}

/// Predicate set for conditional.
#[derive(Serialize, Debug)]
pub struct WhenDto {
    pub predicates: Vec<ConditionDto>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub is_otherwise: bool,
}

/// Result branch for conditional.
#[derive(Serialize, Debug)]
pub struct ThenDto {
    pub solutions: SolutionSetDto,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub residual: Option<ExprDto>,
}

/// Transparency section (global summary).
#[derive(Serialize, Debug, Default)]
pub struct TransparencyDto {
    pub required_conditions: Vec<ConditionDto>,
    pub assumptions_used: Vec<AssumptionDto>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub blocked_hints: Vec<BlockedHintDto>,
}

/// Domain condition (Requires).
#[derive(Serialize, Debug, Clone)]
pub struct ConditionDto {
    pub kind: String,
    pub display: String,
    pub expr_display: String,
    pub expr_canonical: String,
}

/// Assumption used (Assumed).
#[derive(Serialize, Debug, Clone)]
pub struct AssumptionDto {
    pub kind: String,
    pub display: String,
    pub expr_canonical: String,
    pub rule: String,
}

/// Blocked hint.
#[derive(Serialize, Debug, Clone)]
pub struct BlockedHintDto {
    pub rule: String,
    pub requires: Vec<String>,
    pub tip: String,
}

/// Step DTO for trace.
#[derive(Serialize, Debug)]
pub struct StepDto {
    pub index: usize,
    pub rule: String,
    pub before: ExprDto,
    pub after: ExprDto,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub assumptions_used: Vec<AssumptionDto>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub required_conditions: Vec<ConditionDto>,
}

impl RequestInfo {
    pub fn eval(input: impl Into<String>, options: RequestOptions) -> Self {
        Self {
            kind: "eval".to_string(),
            input: input.into(),
            solve_var: None,
            options,
        }
    }
}

impl OutputEnvelope {
    pub fn eval_success(
        request: RequestInfo,
        value: ExprDto,
        transparency: TransparencyDto,
    ) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            engine: EngineInfo::default(),
            request,
            result: ResultDto::Eval { value },
            transparency,
            steps: vec![],
        }
    }

    pub fn eval_error(request: RequestInfo, message: impl Into<String>) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            engine: EngineInfo::default(),
            request,
            result: ResultDto::Error {
                message: message.into(),
            },
            transparency: TransparencyDto::default(),
            steps: vec![],
        }
    }

    pub fn to_json(&self) -> String {
        serialize_json(self, false)
    }

    pub fn to_json_pretty(&self) -> String {
        serialize_json(self, true)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        parse_eval_special_command, DomainJson, EngineJsonError, EngineJsonResponse,
        ErrorJsonOutput, EvalJsonOutput, EvalJsonOutputBuild, EvalLimitApproach, ExprStatsJson,
        JsonRunOptions, LimitJsonResponse, OptionsJson, SemanticsJson, SubstituteJsonOptions,
        TimingsJson,
    };

    #[test]
    fn domain_json_from_mode_sets_mode() {
        let domain = DomainJson::from_mode("strict");
        assert_eq!(domain.mode, "strict");
    }

    #[test]
    fn options_json_from_eval_axes_sets_all_fields() {
        let options =
            OptionsJson::from_eval_axes("auto", "strict", "off", "auto", "off", "generic", "safe");
        assert_eq!(options.context_mode, "auto");
        assert_eq!(options.branch_mode, "strict");
        assert_eq!(options.expand_policy, "off");
        assert_eq!(options.complex_mode, "auto");
        assert_eq!(options.steps_mode, "off");
        assert_eq!(options.domain_mode, "generic");
        assert_eq!(options.const_fold, "safe");
    }

    #[test]
    fn semantics_json_from_eval_axes_sets_all_fields() {
        let semantics = SemanticsJson::from_eval_axes(
            "assume",
            "complex",
            "principal",
            "principal",
            "wildcard",
        );
        assert_eq!(semantics.domain_mode, "assume");
        assert_eq!(semantics.value_domain, "complex");
        assert_eq!(semantics.branch, "principal");
        assert_eq!(semantics.inv_trig, "principal");
        assert_eq!(semantics.assume_scope, "wildcard");
    }

    #[test]
    fn limit_json_response_ok_omits_error_and_code() {
        let response = LimitJsonResponse::ok("1/2".to_string(), None);
        let value: serde_json::Value = serde_json::from_str(&response.to_json()).expect("json");
        assert_eq!(value["ok"], true);
        assert_eq!(value["result"], "1/2");
        assert!(value.get("warning").is_none());
        assert!(value.get("error").is_none());
        assert!(value.get("code").is_none());
    }

    #[test]
    fn limit_json_response_parse_error_has_code_contract() {
        let response = LimitJsonResponse::parse_error("Parse error: bad input");
        let value: serde_json::Value = serde_json::from_str(&response.to_json()).expect("json");
        assert_eq!(value["ok"], false);
        assert_eq!(value["error"], "Parse error: bad input");
        assert_eq!(value["code"], "PARSE_ERROR");
        assert!(value.get("result").is_none());
    }

    #[test]
    fn engine_json_error_invalid_options_json_sets_contract_fields() {
        let err = EngineJsonError::invalid_options_json("expected value");
        let value = serde_json::to_value(err).expect("serialize");
        assert_eq!(value["kind"], "InvalidInput");
        assert_eq!(value["code"], "E_INVALID_INPUT");
        assert_eq!(value["details"]["error"], "expected value");
    }

    #[test]
    fn engine_json_error_session_ref_not_supported_has_hint() {
        let err = EngineJsonError::session_ref_not_supported_for_stateless_eval();
        let value = serde_json::to_value(err).expect("serialize");
        assert_eq!(value["kind"], "InvalidInput");
        assert_eq!(value["code"], "E_INVALID_INPUT");
        assert!(value["message"]
            .as_str()
            .unwrap_or_default()
            .contains("Session references"));
        assert!(value["details"]["hint"]
            .as_str()
            .unwrap_or_default()
            .contains("stateful Engine::eval"));
    }

    #[test]
    fn engine_json_error_from_eval_runtime_error_maps_stateful_hint() {
        let err = EngineJsonError::from_eval_runtime_error("requires stateful eval for #1");
        assert_eq!(err.kind, "InvalidInput");
        assert_eq!(err.code, "E_INVALID_INPUT");
    }

    #[test]
    fn engine_json_error_from_eval_runtime_error_maps_internal_error() {
        let err = EngineJsonError::from_eval_runtime_error("boom");
        assert_eq!(err.kind, "InternalError");
        assert_eq!(err.code, "E_INTERNAL");
        assert_eq!(err.message, "boom");
    }

    #[test]
    fn error_json_output_from_eval_error_message_maps_parse_errors() {
        let out = ErrorJsonOutput::from_eval_error_message("Parse error: bad token", "x+");
        assert_eq!(out.kind, "ParseError");
        assert_eq!(out.code, "E_PARSE");
        assert_eq!(out.input.as_deref(), Some("x+"));
    }

    #[test]
    fn eval_json_output_from_build_sets_schema_and_budget_mode() {
        let out = EvalJsonOutput::from_build(EvalJsonOutputBuild {
            input: "x+x",
            input_latex: None,
            result: "2*x".to_string(),
            result_truncated: false,
            result_chars: 3,
            result_latex: None,
            steps_mode: "off",
            steps_count: 0,
            steps: vec![],
            solve_steps: vec![],
            warnings: vec![],
            required_conditions: vec![],
            required_display: vec![],
            budget_preset: "cli",
            strict: true,
            domain: "generic",
            stats: ExprStatsJson::default(),
            hash: None,
            timings_us: TimingsJson::default(),
            context_mode: "auto",
            branch_mode: "principal",
            expand_policy: "off",
            complex_mode: "auto",
            const_fold: "safe",
            value_domain: "real",
            complex_branch: "principal",
            inv_trig: "principal",
            assume_scope: "wildcard",
            wire: None,
        });
        assert_eq!(out.schema_version, 1);
        assert_eq!(out.budget.mode, "strict");
        assert_eq!(out.options.domain_mode, "generic");
        assert_eq!(out.semantics.value_domain, "real");
    }

    #[test]
    fn engine_json_response_invalid_options_json_has_expected_contract() {
        let out = EngineJsonResponse::invalid_options_json("bad value");
        let value = serde_json::to_value(out).expect("serialize");
        assert_eq!(value["ok"], false);
        assert_eq!(value["error"]["kind"], "InvalidInput");
        assert_eq!(value["error"]["code"], "E_INVALID_INPUT");
        assert_eq!(value["budget"]["preset"], "unknown");
        assert_eq!(value["budget"]["mode"], "strict");
    }

    #[test]
    fn json_run_options_requested_pretty_detects_true_literal() {
        assert!(JsonRunOptions::requested_pretty("{\"pretty\":true}"));
        assert!(!JsonRunOptions::requested_pretty("{\"pretty\": false}"));
    }

    #[test]
    fn substitute_json_options_parse_optional_json_uses_defaults_on_invalid() {
        let parsed = SubstituteJsonOptions::parse_optional_json(Some("{invalid"));
        assert_eq!(parsed.mode, "power");
        assert!(!parsed.steps);
        assert!(!parsed.pretty);
    }

    #[test]
    fn substitute_json_options_from_mode_flags_sets_fields() {
        let parsed = SubstituteJsonOptions::from_mode_flags("exact", true, true);
        assert_eq!(parsed.mode, "exact");
        assert!(parsed.steps);
        assert!(parsed.pretty);
    }

    #[test]
    fn parse_eval_special_command_parses_solve_and_limit() {
        let solve = parse_eval_special_command("solve((x+1)=0, x)").expect("solve");
        assert_eq!(
            solve,
            super::EvalSpecialCommand::Solve {
                equation: "(x+1)=0".to_string(),
                var: "x".to_string()
            }
        );

        let limit = parse_eval_special_command("limit((x^2+1)/x, x, -inf)").expect("limit");
        assert_eq!(
            limit,
            super::EvalSpecialCommand::Limit {
                expr: "(x^2+1)/x".to_string(),
                var: "x".to_string(),
                approach: EvalLimitApproach::NegInfinity,
            }
        );
    }

    #[test]
    fn parse_eval_special_command_rejects_invalid_forms() {
        assert!(parse_eval_special_command("solve(x+1=0)").is_none());
        assert!(parse_eval_special_command("limit(x, x, sideways)").is_none());
        assert!(parse_eval_special_command("x + 1").is_none());
    }
}
