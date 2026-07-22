//! Wire DTOs shared by CLI/FFI layers.
//!
//! This crate intentionally keeps transport models independent from engine internals.

use serde::{Deserialize, Serialize};

/// Stable schema version for wire outputs.
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
// Shared Eval wire types
// =============================================================================

/// Expression statistics (node count, depth).
#[derive(Serialize, Debug, Default)]
pub struct ExprStatsWire {
    pub node_count: usize,
    pub depth: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub term_count: Option<usize>,
}

/// Timing breakdown in microseconds.
#[derive(Serialize, Debug, Default)]
pub struct TimingsWire {
    pub parse_us: u64,
    pub simplify_us: u64,
    pub total_us: u64,
}

/// Domain mode information.
#[derive(Serialize, Debug, Default)]
pub struct DomainWire {
    pub mode: String,
}

impl DomainWire {
    pub fn from_mode(mode: impl Into<String>) -> Self {
        Self { mode: mode.into() }
    }
}

/// Options used for evaluation.
#[derive(Serialize, Debug, Default)]
pub struct OptionsWire {
    pub context_mode: String,
    pub branch_mode: String,
    pub expand_policy: String,
    pub complex_mode: String,
    pub steps_mode: String,
    pub domain_mode: String,
    pub const_fold: String,
}

impl OptionsWire {
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

/// Complete semantics configuration in wire output.
#[derive(Serialize, Debug, Default)]
pub struct SemanticsWire {
    pub domain_mode: String,
    pub value_domain: String,
    pub branch: String,
    pub inv_trig: String,
    pub assume_scope: String,
}

impl SemanticsWire {
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
pub struct RequiredConditionWire {
    pub kind: String,
    pub expr_display: String,
    pub expr_canonical: String,
}

/// A simplification step for wire output.
#[derive(Serialize, Debug, Clone)]
pub struct StepWire {
    pub index: usize,
    pub rule: String,
    pub rule_latex: String,
    pub before: String,
    pub after: String,
    pub before_latex: String,
    pub after_latex: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub substeps: Vec<SubStepWire>,
}

/// A sub-step within a step for detailed explanations.
#[derive(Serialize, Debug, Clone)]
pub struct SubStepWire {
    pub title: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub lines: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub before_latex: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub after_latex: Option<String>,
}

/// A solver step for equation-solving wire output.
#[derive(Serialize, Debug, Clone)]
pub struct SolveStepWire {
    pub index: usize,
    pub description: String,
    pub equation: String,
    pub lhs_latex: String,
    pub relop: String,
    pub rhs_latex: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub substeps: Vec<SolveSubStepWire>,
}

/// A sub-step within a solve step for detailed derivation.
#[derive(Serialize, Debug, Clone)]
pub struct SolveSubStepWire {
    pub index: String,
    pub description: String,
    pub equation: String,
    pub lhs_latex: String,
    pub relop: String,
    pub rhs_latex: String,
}

/// A domain assumption warning with its source rule.
#[derive(Serialize, Debug, Clone)]
pub struct WarningWire {
    pub rule: String,
    pub assumption: String,
}

/// Diagnostic payload for equivalence checks that do not prove equality.
#[derive(Serialize, Debug, Clone)]
pub struct EquivalenceDiagnosticsWire {
    pub residual: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub residual_latex: Option<String>,
}

/// Warning payload used by engine-style response envelopes.
#[derive(Serialize, Debug, Clone)]
pub struct EngineWireWarning {
    pub kind: String,
    pub message: String,
}

/// Stable budget information for wire responses.
#[derive(Serialize, Debug, Default, Clone)]
pub struct BudgetWireInfo {
    pub preset: String,
    pub mode: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exceeded: Option<BudgetExceededWire>,
}

impl BudgetWireInfo {
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

/// Budget exceeded details.
#[derive(Serialize, Debug, Clone)]
pub struct BudgetExceededWire {
    pub op: String,
    pub metric: String,
    pub used: u64,
    pub limit: u64,
}

/// An error result with stable kind/code for API consumers.
#[derive(Serialize, Debug)]
pub struct ErrorWireOutput {
    pub schema_version: u8,
    pub ok: bool,
    pub kind: String,
    pub code: String,
    pub error: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<String>,
}

impl ErrorWireOutput {
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
        // Chokepoint E: match the span-carrying form ("Parse error at 2..3:")
        // as well as the bare prefix, and route solver errors to their real
        // kind/code instead of corrupting them to E_INTERNAL.
        if error.starts_with("Parse error") {
            Self::parse_error(error, Some(input.to_string()))
        } else if error.starts_with("Solver error:") {
            Self {
                schema_version: SCHEMA_VERSION,
                ok: false,
                kind: "SolverError".to_string(),
                code: "E_SOLVER".to_string(),
                error: error.to_string(),
                input: Some(input.to_string()),
            }
        } else {
            Self::with_input(error, input)
        }
    }
}

/// Result of evaluating a single expression via the eval wire contract.
#[derive(Serialize, Debug)]
pub struct EvalWireOutput {
    pub schema_version: u8,
    pub ok: bool,
    pub input: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_latex: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stored_id: Option<u64>,
    pub result: String,
    pub result_truncated: bool,
    pub result_chars: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result_latex: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strategy: Option<String>,
    pub steps_mode: String,
    pub steps_count: usize,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub steps: Vec<StepWire>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub solve_steps: Vec<SolveStepWire>,
    pub warnings: Vec<WarningWire>,
    pub required_conditions: Vec<RequiredConditionWire>,
    pub required_display: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub assumptions_used: Vec<AssumptionDto>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub blocked_hints: Vec<BlockedHintDto>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub equivalence_diagnostics: Option<EquivalenceDiagnosticsWire>,
    pub budget: BudgetWireInfo,
    pub domain: DomainWire,
    pub stats: ExprStatsWire,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hash: Option<String>,
    pub timings_us: TimingsWire,
    pub options: OptionsWire,
    pub semantics: SemanticsWire,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wire: Option<serde_json::Value>,
}

/// Inputs required to build a complete `EvalWireOutput`.
pub struct EvalOutputBuild<'a> {
    pub input: &'a str,
    pub input_latex: Option<String>,
    pub stored_id: Option<u64>,
    pub result: String,
    pub result_truncated: bool,
    pub result_chars: usize,
    pub result_latex: Option<String>,
    pub strategy: Option<String>,
    pub steps_mode: &'a str,
    pub steps_count: usize,
    pub steps: Vec<StepWire>,
    pub solve_steps: Vec<SolveStepWire>,
    pub warnings: Vec<WarningWire>,
    pub required_conditions: Vec<RequiredConditionWire>,
    pub required_display: Vec<String>,
    pub assumptions_used: Vec<AssumptionDto>,
    pub blocked_hints: Vec<BlockedHintDto>,
    pub equivalence_diagnostics: Option<EquivalenceDiagnosticsWire>,
    pub budget_preset: &'a str,
    pub strict: bool,
    pub domain: &'a str,
    pub stats: ExprStatsWire,
    pub hash: Option<String>,
    pub timings_us: TimingsWire,
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

impl EvalWireOutput {
    pub fn from_build(parts: EvalOutputBuild<'_>) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            ok: true,
            input: parts.input.to_string(),
            input_latex: parts.input_latex,
            stored_id: parts.stored_id,
            result: parts.result,
            result_truncated: parts.result_truncated,
            result_chars: parts.result_chars,
            result_latex: parts.result_latex,
            strategy: parts.strategy,
            steps_mode: parts.steps_mode.to_string(),
            steps_count: parts.steps_count,
            steps: parts.steps,
            solve_steps: parts.solve_steps,
            warnings: parts.warnings,
            required_conditions: parts.required_conditions,
            required_display: parts.required_display,
            assumptions_used: parts.assumptions_used,
            blocked_hints: parts.blocked_hints,
            equivalence_diagnostics: parts.equivalence_diagnostics,
            budget: BudgetWireInfo::new(parts.budget_preset, parts.strict),
            domain: DomainWire::from_mode(parts.domain),
            stats: parts.stats,
            hash: parts.hash,
            timings_us: parts.timings_us,
            options: OptionsWire::from_eval_axes(
                parts.context_mode,
                parts.branch_mode,
                parts.expand_policy,
                parts.complex_mode,
                parts.steps_mode,
                parts.domain,
                parts.const_fold,
            ),
            semantics: SemanticsWire::from_eval_axes(
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvalLimitApproach {
    PosInfinity,
    NegInfinity,
    Finite(String),
    FiniteFromLeft(String),
    FiniteFromRight(String),
}

/// Parsed special command forms accepted by eval input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvalSpecialCommand {
    Solve {
        equation: String,
        var: String,
    },
    SolveSystem {
        input: String,
    },
    Derive {
        input: String,
    },
    Equiv {
        input: String,
    },
    Limit {
        expr: String,
        var: String,
        approach: EvalLimitApproach,
    },
    /// `dsolve(<equation>, <func>[, <var>[, <condition>...]])` — elementary ODE
    /// solving (Fase 4). The equation travels as RAW text: the ODE tree must
    /// never touch the general simplifier before the dsolve action extracts its
    /// structure (`diff(y,x)` collapses to `0` under plain eval). Conditions
    /// (initial values like `y(0)=3`) also stay textual: their heads (`y(0)`,
    /// `y'(0)`) are not parseable expressions.
    Dsolve {
        equation: String,
        func: String,
        var: String,
        conditions: Vec<String>,
    },
    /// `dsolve([eq1, eq2], [x, y], t[, conditions...])` — first-order 2×2
    /// linear system (Fase 4 · O6). Equations travel as RAW text (same
    /// anti-collapse rule as the scalar form).
    DsolveSystem {
        equations: Vec<String>,
        funcs: Vec<String>,
        var: String,
        conditions: Vec<String>,
    },
}

/// Parse special eval command forms:
/// - `solve(equation, var)`
/// - `solve_system(eq1; eq2; ...; var1; var2; ...)`
/// - `derive <expr1>, <expr2>` or `derive(expr1, expr2)`
/// - `equiv <expr1>, <expr2>` or `equiv(expr1, expr2)`
/// - `limit(expr, var, approach)` or `lim(expr, var, approach)`
pub fn parse_eval_special_command(input: &str) -> Option<EvalSpecialCommand> {
    if let Some((equation, var)) = parse_solve_command(input) {
        return Some(EvalSpecialCommand::Solve { equation, var });
    }
    // `solve([eq1, eq2, ...], [x, y, ...])` — the natural list form of a linear system,
    // routed to the same pipeline as `solve_system(...)`. Tried after the single-equation
    // `solve(eq, var)` form (which declines the list shape).
    if let Some(system_input) = parse_solve_system_list_command(input) {
        return Some(EvalSpecialCommand::SolveSystem {
            input: system_input,
        });
    }
    if let Some(system_input) = parse_solve_system_command(input) {
        return Some(EvalSpecialCommand::SolveSystem {
            input: system_input,
        });
    }
    if let Some(derive_input) = parse_derive_command(input) {
        return Some(EvalSpecialCommand::Derive {
            input: derive_input,
        });
    }
    if let Some(equiv_input) = parse_equiv_command(input) {
        return Some(EvalSpecialCommand::Equiv { input: equiv_input });
    }
    if let Some((expr, var, approach)) = parse_limit_command(input) {
        return Some(EvalSpecialCommand::Limit {
            expr,
            var,
            approach,
        });
    }
    if let Some((equations, funcs, var, conditions)) = parse_dsolve_system_command(input) {
        return Some(EvalSpecialCommand::DsolveSystem {
            equations,
            funcs,
            var,
            conditions,
        });
    }
    if let Some((equation, func, var, conditions)) = parse_dsolve_command(input) {
        return Some(EvalSpecialCommand::Dsolve {
            equation,
            func,
            var,
            conditions,
        });
    }
    None
}

/// Parsed 2×2 system parts: (equations, funcs, var, conditions).
type DsolveSystemParts = (Vec<String>, Vec<String>, String, Vec<String>);

/// Parse the 2×2 system list form
/// `dsolve([eq1, eq2], [f1, f2], var[, conditions...])`.
fn parse_dsolve_system_command(input: &str) -> Option<DsolveSystemParts> {
    let trimmed = input.trim();
    if !trimmed.to_lowercase().starts_with("dsolve(") || !trimmed.ends_with(')') {
        return None;
    }
    let content = trimmed["dsolve(".len()..trimmed.len() - 1].trim();
    let parts = split_top_level_commas(content);
    if parts.len() < 3 {
        return None;
    }
    let equations: Vec<String> = split_top_level_commas(bracketed_inner(&parts[0])?)
        .into_iter()
        .map(|e| e.trim().to_string())
        .collect();
    let funcs: Vec<String> = split_top_level_commas(bracketed_inner(&parts[1])?)
        .into_iter()
        .map(|f| f.trim().to_string())
        .collect();
    if equations.len() != 2 || funcs.len() != 2 {
        return None;
    }
    if equations.iter().any(|e| e.is_empty()) || !funcs.iter().all(|f| is_plain_identifier(f)) {
        return None;
    }
    let var = parts[2].trim().to_string();
    if !is_plain_identifier(&var) {
        return None;
    }
    let conditions: Vec<String> = parts[3..].iter().map(|c| c.trim().to_string()).collect();
    if conditions.iter().any(|c| c.is_empty()) {
        return None;
    }
    Some((equations, funcs, var, conditions))
}

/// Scan the raw equation text for `diff(<func>, <var>...)` calls and collect the
/// distinct differentiation variables attached to `func`. Textual on purpose: the
/// ODE must not be parsed/simplified at the wire layer (the tree would collapse).
/// Word-boundary guarded so `mydiff(...)` never matches.
fn scan_dsolve_diff_vars(equation: &str, func: &str) -> Vec<String> {
    let bytes = equation.as_bytes();
    let mut vars: Vec<String> = Vec::new();
    let mut search_from = 0usize;
    while let Some(rel) = equation[search_from..].find("diff") {
        let start = search_from + rel;
        search_from = start + 4;
        // Word boundary: previous char must not be alphanumeric or '_'.
        if start > 0 {
            let prev = bytes[start - 1] as char;
            if prev.is_alphanumeric() || prev == '_' {
                continue;
            }
        }
        // Skip whitespace, expect '('.
        let mut i = start + 4;
        while i < equation.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= equation.len() || bytes[i] as char != '(' {
            continue;
        }
        // Extract the full argument list of THIS call by paren depth.
        let args_start = i + 1;
        let mut depth = 1i32;
        let mut j = args_start;
        while j < equation.len() && depth > 0 {
            match bytes[j] as char {
                '(' | '[' => depth += 1,
                ')' | ']' => depth -= 1,
                _ => {}
            }
            j += 1;
        }
        if depth != 0 {
            continue;
        }
        let args = split_top_level_commas(&equation[args_start..j - 1]);
        if args.len() < 2 {
            continue;
        }
        if args[0].trim() != func {
            continue;
        }
        let var = args[1].trim();
        if !var.is_empty()
            && var.chars().next().is_some_and(|c| c.is_alphabetic())
            && var.chars().all(|c| c.is_alphanumeric() || c == '_')
            && !vars.iter().any(|v| v == var)
        {
            vars.push(var.to_string());
        }
    }
    vars
}

fn is_plain_identifier(s: &str) -> bool {
    !s.is_empty()
        && s.chars().next().is_some_and(|c| c.is_alphabetic())
        && s.chars().all(|c| c.is_alphanumeric() || c == '_')
}

/// Parse `dsolve(<equation>, <func>[, <var>[, <condition>...]])`.
///
/// Accepted shapes (D1 + resolved question #1 of the Fase 4 scoping):
/// - canonical: `dsolve(diff(y,x)=x*y, y, x)` (+ trailing textual conditions);
/// - arity-2 sugar: `dsolve(diff(y,x)=x*y, y)` — the variable is inferred from
///   the unique `diff(y, <var>)` occurrence in the equation text;
/// - a third argument that is not a plain identifier is treated as the first
///   condition with the variable inferred (`dsolve(eq, y, y(0)=3)`).
///
/// Returns `None` for every malformed `dsolve(` form; the companion
/// `parse_eval_dsolve_command_error` pre-pass owns the usage message so the
/// input never falls through to the cryptic statement-parse error.
fn parse_dsolve_command(input: &str) -> Option<(String, String, String, Vec<String>)> {
    let trimmed = input.trim();
    if !trimmed.to_lowercase().starts_with("dsolve(") || !trimmed.ends_with(')') {
        return None;
    }
    let content = trimmed["dsolve(".len()..trimmed.len() - 1].trim();
    let parts = split_top_level_commas(content);
    if parts.len() < 2 {
        return None;
    }
    let equation = parts[0].trim().to_string();
    let func = parts[1].trim().to_string();
    if equation.is_empty() || !is_plain_identifier(&func) {
        return None;
    }
    let diff_vars = scan_dsolve_diff_vars(&equation, &func);
    let (var, conditions_start) = if parts.len() >= 3 && is_plain_identifier(parts[2].trim()) {
        (parts[2].trim().to_string(), 3)
    } else {
        // Arity-2 sugar: infer the variable; ambiguous/missing diff declines to
        // the error pre-pass.
        if diff_vars.len() != 1 {
            return None;
        }
        (diff_vars[0].clone(), 2)
    };
    // The equation must contain diff(func, var): otherwise it is not an ODE in
    // the declared unknown/variable (wrong-variable mismatches decline too).
    if !diff_vars.contains(&var) {
        return None;
    }
    let conditions: Vec<String> = parts[conditions_start..]
        .iter()
        .map(|c| c.trim().to_string())
        .collect();
    if conditions.iter().any(|c| c.is_empty()) {
        return None;
    }
    Some((equation, func, var, conditions))
}

/// Textual split of a dsolve initial-condition string: `y(x0) = y0` →
/// `(x0, y0, 0)`; `y'(x0) = v0` → `(x0, v0, 1)`. The head (`y(0)`, `y'(0)`)
/// is scanned textually and NEVER reaches the expression parser (D1:
/// apostrophes and call-heads on the unknown are not parseable). Point and
/// value come back as separate texts for individual parsing by the caller.
pub fn split_dsolve_initial_condition(cond: &str, func: &str) -> Option<(String, String, usize)> {
    let trimmed = cond.trim();
    let rest = trimmed.strip_prefix(func)?;
    let (order, rest) = match rest.strip_prefix('\'') {
        Some(r) => (1usize, r),
        None => (0usize, rest),
    };
    let rest = rest.trim_start();
    let inner_and_tail = rest.strip_prefix('(')?;
    // Find the matching close paren of the point group.
    let mut depth = 1i32;
    let mut close = None;
    for (i, ch) in inner_and_tail.char_indices() {
        match ch {
            '(' | '[' => depth += 1,
            ')' | ']' => {
                depth -= 1;
                if depth == 0 {
                    close = Some(i);
                    break;
                }
            }
            _ => {}
        }
    }
    let close = close?;
    let point = inner_and_tail[..close].trim();
    let tail = inner_and_tail[close + 1..].trim_start();
    let value = tail.strip_prefix('=')?.trim();
    if point.is_empty() || value.is_empty() {
        return None;
    }
    Some((point.to_string(), value.to_string(), order))
}

fn dsolve_command_usage_error() -> String {
    "Invalid dsolve command. Expected dsolve(diff(y,x) = f(x,y), y, x[, conditions...]), e.g. dsolve(diff(y,x)=x*y, y, x).".to_string()
}

/// Usage-error pre-pass for malformed `dsolve(` inputs (molde: the limit
/// pre-pass). Without it, a malformed dsolve falls to statement-parse and dies
/// with a cryptic parse error on the inner `=`. Returns `None` for well-formed
/// commands (already captured by `parse_dsolve_command`) and for inputs that do
/// not start with `dsolve(` (composed expressions like `1+dsolve(...)` keep
/// their generic-eval decline).
pub fn parse_eval_dsolve_command_error(input: &str) -> Option<String> {
    let trimmed = input.trim();
    if !trimmed.to_lowercase().starts_with("dsolve(") {
        return None;
    }
    if parse_dsolve_command(trimmed).is_some() {
        return None;
    }
    if !trimmed.ends_with(')') {
        // `dsolve(...)+1` and other composed heads: general eval owns them.
        let mut depth = 0i32;
        for (i, ch) in trimmed.char_indices() {
            match ch {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 && i + 1 < trimmed.len() {
                        return None;
                    }
                }
                _ => {}
            }
        }
        return Some(dsolve_command_usage_error());
    }
    let content = trimmed["dsolve(".len()..trimmed.len() - 1].trim();
    let parts = split_top_level_commas(content);
    // List head → the 2×2 system form owns the message.
    if parts
        .first()
        .is_some_and(|p| p.trim_start().starts_with('['))
        && parse_dsolve_system_command(trimmed).is_none()
    {
        return Some(
            "Invalid dsolve system command. Expected dsolve([eq1, eq2], [f1, f2], var), e.g. dsolve([diff(x,t)=y, diff(y,t)=x], [x,y], t).".to_string(),
        );
    }
    // SymPy-style `y(x)` heads: name the fix explicitly (today this dies in the
    // unknown-function gate with no guidance).
    if parts.len() >= 2 {
        let func_part = parts[1].trim();
        let sympy_func = func_part
            .split_once('(')
            .map(|(head, _)| head.trim().to_string())
            .filter(|head| is_plain_identifier(head));
        if let Some(head) = sympy_func {
            return Some(format!(
                "Invalid dsolve unknown `{func_part}`: write the function as a bare name. Use dsolve(diff({head},x)=..., {head}, x), not {head}(x)."
            ));
        }
        if is_plain_identifier(func_part) {
            let diff_vars = scan_dsolve_diff_vars(parts[0].trim(), func_part);
            if diff_vars.is_empty() {
                return Some(format!(
                    "Invalid dsolve equation: it contains no diff({func_part}, ...) — not an ODE in `{func_part}`. Expected dsolve(diff(y,x) = f(x,y), y, x), e.g. dsolve(diff(y,x)=x*y, y, x)."
                ));
            }
            if parts.len() >= 3 && is_plain_identifier(parts[2].trim()) {
                let var = parts[2].trim();
                if !diff_vars.iter().any(|v| v == var) {
                    return Some(format!(
                        "Invalid dsolve variable `{var}`: the equation differentiates {func_part} with respect to {}, not `{var}`.",
                        diff_vars.join(", ")
                    ));
                }
            } else if diff_vars.len() > 1 {
                return Some(format!(
                    "Ambiguous dsolve variable: the equation contains diff({func_part}, ...) in several variables ({}). Pass the variable explicitly: dsolve(equation, {func_part}, var).",
                    diff_vars.join(", ")
                ));
            }
        }
    }
    Some(dsolve_command_usage_error())
}

pub fn parse_eval_limit_command_error(input: &str) -> Option<String> {
    let trimmed = input.trim();
    let lower = trimmed.to_lowercase();
    let prefix_len = if lower.starts_with("limit(") {
        6
    } else if lower.starts_with("lim(") {
        4
    } else {
        return None;
    };

    let content = match classify_limit_wire_shape(trimmed, prefix_len) {
        LimitWireShape::SingleCall(content) => content,
        // Composed expression (`limit(...)+1`, iterated head): general eval
        // owns it — no wire-level error (F9).
        LimitWireShape::ComposedExpression => return None,
        LimitWireShape::Malformed => return Some(limit_command_usage_error()),
    };
    let parts = split_by_comma_at_depth_0(content);
    // Multivariate list form `limit(f, [x,y], [a,b])` (F7, Fase 3): not a
    // wire-level limit command — let the general eval path route it to the
    // engine rule instead of erroring here.
    if parts.iter().any(|p| p.trim().starts_with('[')) {
        return None;
    }
    if parts.len() < 2 {
        return Some(limit_command_usage_error());
    }
    if parts.len() > 3 {
        let extra = parts[3].trim();
        if looks_like_one_sided_limit_marker(extra) {
            if parts.len() == 4
                && looks_like_finite_limit_point(parts[2].trim())
                && parse_one_sided_limit_marker(extra).is_some()
            {
                return None;
            }
            return Some(format!(
                "Unsupported one-sided limit approach `{extra}`. One-sided eval limit syntax expects limit(expr, var, finite_point, left|right)."
            ));
        }
        return Some(
            "Invalid limit command. Expected limit(expr, var, approach); eval limit syntax does not accept a fourth argument."
                .to_string(),
        );
    }

    let var_str = parts[1].trim();
    if var_str.is_empty()
        || !var_str.chars().next().is_some_and(|ch| ch.is_alphabetic())
        || !var_str.chars().all(|c| c.is_alphanumeric() || c == '_')
    {
        return Some("Invalid limit variable. Expected a variable name.".to_string());
    }

    if parts.len() == 3 {
        let dir = parts[2].trim();
        match dir.to_lowercase().as_str() {
            "inf" | "infinity" | "+inf" | "+infinity" | "-inf" | "-infinity" => None,
            _ if parse_one_sided_finite_limit_point(dir).is_some() => None,
            _ if looks_like_finite_limit_point(dir) => None,
            _ => Some(format!(
                "Unsupported limit direction `{dir}`. Use infinity, -infinity, or a finite point."
            )),
        }
    } else {
        None
    }
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

/// Split `s` by top-level commas, ignoring commas nested inside `()` or `[]`.
fn split_top_level_commas(s: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut start = 0usize;
    for (i, ch) in s.char_indices() {
        match ch {
            '(' | '[' => depth += 1,
            ')' | ']' => depth -= 1,
            ',' if depth == 0 => {
                parts.push(s[start..i].trim().to_string());
                start = i + ch.len_utf8();
            }
            _ => {}
        }
    }
    parts.push(s[start..].trim().to_string());
    parts
}

/// Return the inside of a `[ ... ]` group (trimmed), or `None` if `s` is not bracketed.
fn bracketed_inner(s: &str) -> Option<&str> {
    let s = s.trim();
    s.strip_prefix('[')?.strip_suffix(']').map(str::trim)
}

/// True when `s` contains a top-level `=` (not nested inside `()`/`[]`). Used to decide
/// whether an equation part already carries a relation or should be read as `expr = 0`.
fn has_top_level_eq(s: &str) -> bool {
    let mut depth = 0i32;
    for ch in s.chars() {
        match ch {
            '(' | '[' => depth += 1,
            ')' | ']' => depth -= 1,
            '=' if depth == 0 => return true,
            _ => {}
        }
    }
    false
}

/// Parse the natural system-solve list form
/// `solve([eq1, eq2, ...], [var1, var2, ...])` into the canonical `solve_system`
/// semicolon spec `eq1; eq2; ...; var1; var2; ...`, so it reuses the existing linear
/// system pipeline verbatim. An equation part with no top-level `=` is read as
/// `expr = 0` (so `solve([x+y-3, x-y-1], [x, y])` works too). Requires equal, ≥2 counts
/// of equations and variables (a square system); a single-equation solve goes through
/// `solve(eq, var)`. Variables must be plain identifiers.
pub fn parse_solve_system_list_command(input: &str) -> Option<String> {
    let trimmed = input.trim();
    if !trimmed.to_lowercase().starts_with("solve(") || !trimmed.ends_with(')') {
        return None;
    }
    let content = trimmed["solve(".len()..trimmed.len() - 1].trim();
    let groups = split_top_level_commas(content);
    if groups.len() != 2 {
        return None;
    }
    let eqs = split_top_level_commas(bracketed_inner(&groups[0])?);
    let vars = split_top_level_commas(bracketed_inner(&groups[1])?);
    // Square system with ≥2 equations; single solves use `solve(eq, var)`.
    if eqs.len() != vars.len() || eqs.len() < 2 {
        return None;
    }

    let mut parts: Vec<String> = Vec::with_capacity(eqs.len() + vars.len());
    for eq in &eqs {
        let eq = eq.trim();
        if eq.is_empty() {
            return None;
        }
        if has_top_level_eq(eq) {
            parts.push(eq.to_string());
        } else {
            parts.push(format!("{eq} = 0"));
        }
    }
    for var in &vars {
        let var = var.trim();
        if var.is_empty()
            || !var.chars().next()?.is_alphabetic()
            || !var.chars().all(|c| c.is_alphanumeric() || c == '_')
        {
            return None;
        }
        parts.push(var.to_string());
    }
    Some(parts.join("; "))
}

fn parse_derive_command(input: &str) -> Option<String> {
    let trimmed = input.trim();
    let lower = trimmed.to_lowercase();
    if lower == "derive" {
        return Some(String::new());
    }
    if lower.starts_with("derive(") && trimmed.ends_with(')') {
        return Some(
            trimmed["derive(".len()..trimmed.len() - 1]
                .trim()
                .to_string(),
        );
    }
    if !lower.starts_with("derive ") {
        return None;
    }
    Some(trimmed[6..].trim().to_string())
}

fn parse_equiv_command(input: &str) -> Option<String> {
    let trimmed = input.trim();
    let lower = trimmed.to_lowercase();
    if lower == "equiv" {
        return Some(String::new());
    }
    if lower.starts_with("equiv(") && trimmed.ends_with(')') {
        return Some(
            trimmed["equiv(".len()..trimmed.len() - 1]
                .trim()
                .to_string(),
        );
    }
    if !lower.starts_with("equiv ") {
        return None;
    }
    Some(trimmed[5..].trim().to_string())
}

fn parse_solve_system_command(input: &str) -> Option<String> {
    let trimmed = input.trim();
    let lower = trimmed.to_lowercase();
    if lower == "solve_system" {
        return Some(String::new());
    }
    if lower.starts_with("solve_system(") && trimmed.ends_with(')') {
        return Some(
            trimmed["solve_system(".len()..trimmed.len() - 1]
                .trim()
                .to_string(),
        );
    }
    if !lower.starts_with("solve_system ") {
        return None;
    }
    Some(trimmed["solve_system".len()..].trim().to_string())
}

/// Wire-shape verdict for an input starting with `limit(`/`lim(` (F9): is it a
/// SINGLE call (the wire command), a COMPOSED expression (`limit(...)+1`,
/// `limit(limit(...),y,2)` — general eval owns it: the nested-limit rule
/// composes bottom-up), or MALFORMED (unbalanced — wire usage-error)?
enum LimitWireShape<'a> {
    SingleCall(&'a str),
    ComposedExpression,
    Malformed,
}

fn classify_limit_wire_shape(trimmed: &str, prefix_len: usize) -> LimitWireShape<'_> {
    // Find the close paren matching the prefix's opening `(`.
    let mut depth: i32 = 1;
    for (i, ch) in trimmed[prefix_len..].char_indices() {
        match ch {
            '(' | '[' => depth += 1,
            ')' | ']' => {
                depth -= 1;
                if depth == 0 {
                    let end = prefix_len + i;
                    return if end == trimmed.len() - 1 {
                        let content = &trimmed[prefix_len..end];
                        // An iterated head (`limit(limit(...),y,b)`) is ALSO
                        // general-eval territory: the inner call must resolve
                        // through the simplifier before the outer sees it.
                        let first = split_by_comma_at_depth_0(content)
                            .first()
                            .map(|p| p.trim().to_lowercase())
                            .unwrap_or_default();
                        if first.starts_with("limit(") || first.starts_with("lim(") {
                            LimitWireShape::ComposedExpression
                        } else {
                            LimitWireShape::SingleCall(content)
                        }
                    } else {
                        LimitWireShape::ComposedExpression
                    };
                }
            }
            _ => {}
        }
    }
    LimitWireShape::Malformed
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

    let LimitWireShape::SingleCall(content) = classify_limit_wire_shape(trimmed, prefix_len) else {
        return None;
    };
    let parts = split_by_comma_at_depth_0(content);
    if parts.len() < 2 || parts.len() > 4 {
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
        let dir = parts[2].trim();
        match dir.to_lowercase().as_str() {
            "inf" | "infinity" | "+inf" | "+infinity" => EvalLimitApproach::PosInfinity,
            "-inf" | "-infinity" => EvalLimitApproach::NegInfinity,
            _ if parse_one_sided_finite_limit_point(dir).is_some() => {
                let (point, side) = parse_one_sided_finite_limit_point(dir)?;
                match side {
                    OneSidedLimitMarker::Left => EvalLimitApproach::FiniteFromLeft(point),
                    OneSidedLimitMarker::Right => EvalLimitApproach::FiniteFromRight(point),
                }
            }
            _ if looks_like_finite_limit_point(dir) => EvalLimitApproach::Finite(dir.to_string()),
            _ => return None,
        }
    } else if parts.len() == 4 {
        let point = parts[2].trim();
        if !looks_like_finite_limit_point(point) {
            return None;
        }
        match parse_one_sided_limit_marker(parts[3].trim())? {
            OneSidedLimitMarker::Left => EvalLimitApproach::FiniteFromLeft(point.to_string()),
            OneSidedLimitMarker::Right => EvalLimitApproach::FiniteFromRight(point.to_string()),
        }
    } else {
        EvalLimitApproach::PosInfinity
    };

    Some((expr_str.to_string(), var_str.to_string(), approach))
}

fn limit_command_usage_error() -> String {
    "Invalid limit command. Expected limit(expr, var, approach).".to_string()
}

fn looks_like_finite_limit_point(dir: &str) -> bool {
    let trimmed = dir.trim();
    !trimmed.is_empty()
        && has_finite_limit_point_signal(trimmed)
        && !has_one_sided_limit_suffix(trimmed)
}

fn has_finite_limit_point_signal(raw: &str) -> bool {
    raw.chars().any(|ch| ch.is_ascii_digit()) || contains_named_finite_constant_token(raw)
}

fn contains_named_finite_constant_token(raw: &str) -> bool {
    // F11: `i` cuenta como señal de punto finito — el gate deja de ser un
    // colador léxico (rechazaba `i` pero dejaba pasar `2*i`): el MOTOR decide
    // semánticamente (F0 declina con warning en real; el camino selectivo F11
    // evalúa bajo complex).
    raw.split(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '_'))
        .any(|token| {
            token.eq_ignore_ascii_case("pi")
                || token.eq_ignore_ascii_case("e")
                || token.eq_ignore_ascii_case("i")
        })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OneSidedLimitMarker {
    Left,
    Right,
}

fn parse_one_sided_finite_limit_point(dir: &str) -> Option<(String, OneSidedLimitMarker)> {
    let compact: String = dir.chars().filter(|ch| !ch.is_whitespace()).collect();
    let last = compact.chars().last()?;
    let side = match last {
        '-' => OneSidedLimitMarker::Left,
        '+' => OneSidedLimitMarker::Right,
        _ => return None,
    };
    let point = &compact[..compact.len() - last.len_utf8()];
    looks_like_finite_limit_point(point).then(|| (point.to_string(), side))
}

fn has_one_sided_limit_suffix(dir: &str) -> bool {
    let compact: String = dir.chars().filter(|ch| !ch.is_whitespace()).collect();
    compact
        .chars()
        .last()
        .is_some_and(|last| matches!(last, '+' | '-'))
        && compact.len() > 1
        && has_finite_limit_point_signal(&compact[..compact.len() - 1])
}

fn parse_one_sided_limit_marker(raw: &str) -> Option<OneSidedLimitMarker> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "-" | "left" | "from_left" | "left_hand" | "lh" => Some(OneSidedLimitMarker::Left),
        "+" | "right" | "from_right" | "right_hand" | "rh" => Some(OneSidedLimitMarker::Right),
        _ => None,
    }
}

fn looks_like_one_sided_limit_marker(raw: &str) -> bool {
    matches!(
        raw.trim().to_ascii_lowercase().as_str(),
        "+" | "-"
            | "left"
            | "right"
            | "from_left"
            | "from_right"
            | "left_hand"
            | "right_hand"
            | "lh"
            | "rh"
    )
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
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum EvalStepsMode {
    On,
    #[default]
    Off,
    Compact,
}

impl EvalStepsMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::On => "on",
            Self::Off => "off",
            Self::Compact => "compact",
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum EvalContextMode {
    #[default]
    Auto,
    Standard,
    Solve,
    Integrate,
}

impl EvalContextMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Standard => "standard",
            Self::Solve => "solve",
            Self::Integrate => "integrate",
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum EvalBranchMode {
    #[default]
    Strict,
    Principal,
}

impl EvalBranchMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Strict => "strict",
            Self::Principal => "principal",
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum EvalExpandPolicy {
    #[default]
    Off,
    Auto,
}

impl EvalExpandPolicy {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Auto => "auto",
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum EvalComplexMode {
    #[default]
    Auto,
    On,
    Off,
}

impl EvalComplexMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::On => "on",
            Self::Off => "off",
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum EvalBudgetPreset {
    Small,
    #[default]
    Standard,
    Unlimited,
}

impl EvalBudgetPreset {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Small => "small",
            Self::Standard => "standard",
            Self::Unlimited => "unlimited",
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum EvalDomainMode {
    Strict,
    #[default]
    Generic,
    Assume,
}

impl EvalDomainMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Strict => "strict",
            Self::Generic => "generic",
            Self::Assume => "assume",
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum EvalConstFoldMode {
    #[default]
    Off,
    Safe,
}

impl EvalConstFoldMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Safe => "safe",
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum EvalValueDomain {
    #[default]
    Real,
    Complex,
}

impl EvalValueDomain {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Real => "real",
            Self::Complex => "complex",
        }
    }
}

/// Session-level numeric display mode: `Exact` renders results as exact
/// fractions/radicals (the default, byte-identical contract); `Decimal`
/// applies the numeric-presentation walker at the OUTPUT BOUNDARY only —
/// internally everything stays symbolic and exact.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum EvalNumericDisplay {
    #[default]
    Exact,
    Decimal,
}

impl EvalNumericDisplay {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::Decimal => "decimal",
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum EvalInvTrigPolicy {
    #[default]
    Strict,
    Principal,
}

impl EvalInvTrigPolicy {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Strict => "strict",
            Self::Principal => "principal",
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum EvalAssumeScope {
    #[default]
    Real,
    Wildcard,
}

impl EvalAssumeScope {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Real => "real",
            Self::Wildcard => "wildcard",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct EvalSessionRunConfig<'a> {
    pub expr: &'a str,
    pub auto_store: bool,
    pub max_chars: usize,
    pub time_budget_ms: Option<u64>,
    pub steps_mode: EvalStepsMode,
    pub budget_preset: EvalBudgetPreset,
    pub strict: bool,
    pub domain: EvalDomainMode,
    pub context_mode: EvalContextMode,
    pub branch_mode: EvalBranchMode,
    pub expand_policy: EvalExpandPolicy,
    pub complex_mode: EvalComplexMode,
    pub const_fold: EvalConstFoldMode,
    pub value_domain: EvalValueDomain,
    pub complex_branch: EvalBranchMode,
    pub inv_trig: EvalInvTrigPolicy,
    pub assume_scope: EvalAssumeScope,
    pub numeric_display: EvalNumericDisplay,
}

// =============================================================================
// Engine-style response DTOs (used by FFI fallback paths)
// =============================================================================

/// Source span for wire serialization.
#[derive(Serialize, Debug, Clone, Copy)]
pub struct SpanWire {
    pub start: usize,
    pub end: usize,
}

/// A sub-step representing a rewrite in a subexpression.
#[derive(Serialize, Debug, Clone)]
pub struct EngineWireSubstep {
    pub rule: String,
    pub before: String,
    pub after: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

/// A simplification step in a response.
#[derive(Serialize, Debug, Clone)]
pub struct EngineWireStep {
    pub phase: String,
    pub rule: String,
    pub before: String,
    pub after: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub substeps: Vec<EngineWireSubstep>,
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
pub struct EngineWireError {
    pub kind: &'static str,
    pub code: &'static str,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub span: Option<SpanWire>,
    #[serde(default)]
    pub details: serde_json::Value,
}

impl EngineWireError {
    pub fn simple(kind: &'static str, code: &'static str, message: impl Into<String>) -> Self {
        Self {
            kind,
            code,
            message: message.into(),
            span: None,
            details: serde_json::Value::Null,
        }
    }

    pub fn parse(message: impl Into<String>, span: Option<SpanWire>) -> Self {
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
            "Session references (#N) are not supported in stateless eval mode",
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

/// Unified wire response for engine-like operations.
#[derive(Serialize, Debug, Clone)]
pub struct EngineWireResponse {
    pub schema_version: u8,
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<EngineWireError>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub steps: Vec<EngineWireStep>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<EngineWireWarning>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub assumptions: Vec<AssumptionRecord>,
    pub budget: BudgetWireInfo,
}

impl EngineWireResponse {
    pub fn ok(result: String, budget: BudgetWireInfo) -> Self {
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
        steps: Vec<EngineWireStep>,
        budget: BudgetWireInfo,
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

    pub fn err(error: EngineWireError, budget: BudgetWireInfo) -> Self {
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
            EngineWireError::invalid_options_json(error),
            BudgetWireInfo::new("unknown", true),
        )
    }

    pub fn with_warning(mut self, warning: EngineWireWarning) -> Self {
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
// Engine eval input options
// =============================================================================

/// Parsed run options for eval operations.
#[derive(Deserialize, Debug, Default, Clone)]
pub struct EvalRunOptions {
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

impl EvalRunOptions {
    pub fn requested_pretty(opts_json: &str) -> bool {
        opts_json.contains("\"pretty\":true")
    }
}

// =============================================================================
// substitute wire types
// =============================================================================

/// Parsed run options for substitute operations.
#[derive(Deserialize, Debug, Clone)]
pub struct SubstituteRunOptions {
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

impl Default for SubstituteRunOptions {
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

impl SubstituteRunOptions {
    pub fn from_mode_flags(mode: &str, steps: bool, pretty: bool) -> Self {
        Self {
            mode: mode.to_string(),
            steps,
            pretty,
        }
    }

    pub fn parse_optional_json(opts_json: Option<&str>) -> Self {
        match opts_json {
            Some(json) => {
                let trimmed = json.trim();
                parse_substitute_run_options_fast(trimmed)
                    .unwrap_or_else(|| serde_json::from_str(trimmed).unwrap_or_default())
            }
            None => Self::default(),
        }
    }
}

fn parse_substitute_run_options_fast(opts_json: &str) -> Option<SubstituteRunOptions> {
    match opts_json {
        ""
        | "{}"
        | r#"{"mode":"power"}"#
        | r#"{"pretty":false}"#
        | r#"{"steps":false}"#
        | r#"{"steps":false,"pretty":false}"#
        | r#"{"pretty":false,"steps":false}"#
        | r#"{"mode":"power","steps":false}"#
        | r#"{"steps":false,"mode":"power"}"#
        | r#"{"mode":"power","pretty":false}"#
        | r#"{"pretty":false,"mode":"power"}"# => Some(SubstituteRunOptions::default()),
        r#"{"steps":true}"# => Some(SubstituteRunOptions {
            steps: true,
            ..SubstituteRunOptions::default()
        }),
        r#"{"pretty":true}"# => Some(SubstituteRunOptions {
            pretty: true,
            ..SubstituteRunOptions::default()
        }),
        r#"{"mode":"exact"}"# => Some(SubstituteRunOptions {
            mode: "exact".to_string(),
            ..SubstituteRunOptions::default()
        }),
        r#"{"mode":"exact","steps":true}"# | r#"{"steps":true,"mode":"exact"}"# => {
            Some(SubstituteRunOptions {
                mode: "exact".to_string(),
                steps: true,
                ..SubstituteRunOptions::default()
            })
        }
        r#"{"steps":true,"pretty":true}"# | r#"{"pretty":true,"steps":true}"# => {
            Some(SubstituteRunOptions {
                steps: true,
                pretty: true,
                ..SubstituteRunOptions::default()
            })
        }
        _ => None,
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

/// Wire options echo for substitute operations.
#[derive(Serialize, Debug, Clone)]
pub struct SubstituteWireOptions {
    pub substitute: SubstituteWireOptionsInner,
}

#[derive(Serialize, Debug, Clone)]
pub struct SubstituteWireOptionsInner {
    pub mode: String,
    pub steps: bool,
}

/// Substitute wire response with request echo and options.
#[derive(Serialize, Debug, Clone)]
pub struct SubstituteWireResponse {
    /// Schema version for API stability
    pub schema_version: u8,
    /// True if operation succeeded
    pub ok: bool,
    /// Result expression (success only)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<String>,
    /// Error details (failure only)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<EngineWireError>,
    /// Request echo for reproducibility
    pub request: SubstituteRequestEcho,
    /// Options used
    pub options: SubstituteWireOptions,
    /// Substitution steps (if requested)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub steps: Vec<EngineWireSubstep>,
}

impl SubstituteWireResponse {
    pub fn ok(
        result: String,
        request: SubstituteRequestEcho,
        options: SubstituteWireOptions,
        steps: Vec<EngineWireSubstep>,
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
        error: EngineWireError,
        request: SubstituteRequestEcho,
        options: SubstituteWireOptions,
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

/// Substitution mode for typed substitution evaluation APIs.
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

/// One substitution step for typed substitution evaluation APIs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SubstituteEvalStep {
    pub rule: String,
    pub before: String,
    pub after: String,
    pub note: Option<String>,
}

/// Result payload for typed substitution evaluation.
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

/// Result payload for typed limit evaluation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LimitEvalResult {
    pub result: String,
    pub warning: Option<String>,
}

/// Errors produced by typed limit evaluation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LimitEvalError {
    Parse(String),
    Limit(String),
}

/// Canonical wire response for limit evaluation.
#[derive(Serialize, Debug, Clone, PartialEq, Eq)]
pub struct LimitWireResponse {
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

impl LimitWireResponse {
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
// script wire types
// =============================================================================

/// Result of processing a single line in a script.
#[derive(Serialize, Debug)]
pub struct ScriptLineResult {
    pub line_no: usize,
    pub input: String,
    /// "command" | "let" | "expr" | "empty" | "error"
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<EvalWireOutput>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

// =============================================================================
// mm-gcd-modp wire types
// =============================================================================

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
    pub domain: EvalDomainMode,
    pub value_domain: EvalValueDomain,
}

impl Default for EnvelopeEvalOptions {
    fn default() -> Self {
        Self {
            domain: EvalDomainMode::Generic,
            value_domain: EvalValueDomain::Real,
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
        parse_eval_limit_command_error, parse_eval_special_command,
        parse_solve_system_list_command, DomainWire, EngineWireError, EngineWireResponse,
        ErrorWireOutput, EvalLimitApproach, EvalOutputBuild, EvalRunOptions, EvalWireOutput,
        ExprStatsWire, LimitWireResponse, OptionsWire, SemanticsWire, SubstituteRunOptions,
        TimingsWire,
    };

    #[test]
    fn domain_wire_from_mode_sets_mode() {
        let domain = DomainWire::from_mode("strict");
        assert_eq!(domain.mode, "strict");
    }

    #[test]
    fn options_wire_from_eval_axes_sets_all_fields() {
        let options =
            OptionsWire::from_eval_axes("auto", "strict", "off", "auto", "off", "generic", "safe");
        assert_eq!(options.context_mode, "auto");
        assert_eq!(options.branch_mode, "strict");
        assert_eq!(options.expand_policy, "off");
        assert_eq!(options.complex_mode, "auto");
        assert_eq!(options.steps_mode, "off");
        assert_eq!(options.domain_mode, "generic");
        assert_eq!(options.const_fold, "safe");
    }

    #[test]
    fn semantics_wire_from_eval_axes_sets_all_fields() {
        let semantics = SemanticsWire::from_eval_axes(
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
    fn limit_wire_response_ok_omits_error_and_code() {
        let response = LimitWireResponse::ok("1/2".to_string(), None);
        let value: serde_json::Value = serde_json::from_str(&response.to_json()).expect("json");
        assert_eq!(value["ok"], true);
        assert_eq!(value["result"], "1/2");
        assert!(value.get("warning").is_none());
        assert!(value.get("error").is_none());
        assert!(value.get("code").is_none());
    }

    #[test]
    fn limit_wire_response_parse_error_has_code_contract() {
        let response = LimitWireResponse::parse_error("Parse error: bad input");
        let value: serde_json::Value = serde_json::from_str(&response.to_json()).expect("json");
        assert_eq!(value["ok"], false);
        assert_eq!(value["error"], "Parse error: bad input");
        assert_eq!(value["code"], "PARSE_ERROR");
        assert!(value.get("result").is_none());
    }

    #[test]
    fn engine_wire_error_invalid_options_json_sets_contract_fields() {
        let err = EngineWireError::invalid_options_json("expected value");
        let value = serde_json::to_value(err).expect("serialize");
        assert_eq!(value["kind"], "InvalidInput");
        assert_eq!(value["code"], "E_INVALID_INPUT");
        assert_eq!(value["details"]["error"], "expected value");
    }

    #[test]
    fn engine_wire_error_session_ref_not_supported_has_hint() {
        let err = EngineWireError::session_ref_not_supported_for_stateless_eval();
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
    fn engine_wire_error_from_eval_runtime_error_maps_stateful_hint() {
        let err = EngineWireError::from_eval_runtime_error("requires stateful eval for #1");
        assert_eq!(err.kind, "InvalidInput");
        assert_eq!(err.code, "E_INVALID_INPUT");
    }

    #[test]
    fn engine_wire_error_from_eval_runtime_error_maps_internal_error() {
        let err = EngineWireError::from_eval_runtime_error("boom");
        assert_eq!(err.kind, "InternalError");
        assert_eq!(err.code, "E_INTERNAL");
        assert_eq!(err.message, "boom");
    }

    #[test]
    fn error_wire_output_from_eval_error_message_maps_parse_errors() {
        let out = ErrorWireOutput::from_eval_error_message("Parse error: bad token", "x+");
        assert_eq!(out.kind, "ParseError");
        assert_eq!(out.code, "E_PARSE");
        assert_eq!(out.input.as_deref(), Some("x+"));
    }

    #[test]
    fn eval_wire_output_from_build_sets_schema_and_budget_mode() {
        let out = EvalWireOutput::from_build(EvalOutputBuild {
            input: "x+x",
            input_latex: None,
            stored_id: Some(7),
            result: "2*x".to_string(),
            result_truncated: false,
            result_chars: 3,
            result_latex: None,
            strategy: Some("expand".to_string()),
            steps_mode: "off",
            steps_count: 0,
            steps: vec![],
            solve_steps: vec![],
            warnings: vec![],
            required_conditions: vec![],
            required_display: vec![],
            assumptions_used: vec![],
            blocked_hints: vec![],
            equivalence_diagnostics: None,
            budget_preset: "cli",
            strict: true,
            domain: "generic",
            stats: ExprStatsWire::default(),
            hash: None,
            timings_us: TimingsWire::default(),
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
        assert_eq!(out.strategy.as_deref(), Some("expand"));
        assert_eq!(out.stored_id, Some(7));
        assert_eq!(out.budget.mode, "strict");
        assert_eq!(out.options.domain_mode, "generic");
        assert_eq!(out.semantics.value_domain, "real");
    }

    #[test]
    fn engine_wire_response_invalid_options_json_has_expected_contract() {
        let out = EngineWireResponse::invalid_options_json("bad value");
        let value = serde_json::to_value(out).expect("serialize");
        assert_eq!(value["ok"], false);
        assert_eq!(value["error"]["kind"], "InvalidInput");
        assert_eq!(value["error"]["code"], "E_INVALID_INPUT");
        assert_eq!(value["budget"]["preset"], "unknown");
        assert_eq!(value["budget"]["mode"], "strict");
    }

    #[test]
    fn eval_run_options_requested_pretty_detects_true_literal() {
        assert!(EvalRunOptions::requested_pretty("{\"pretty\":true}"));
        assert!(!EvalRunOptions::requested_pretty("{\"pretty\": false}"));
    }

    #[test]
    fn substitute_run_options_parse_optional_json_uses_defaults_on_invalid() {
        let parsed = SubstituteRunOptions::parse_optional_json(Some("{invalid"));
        assert_eq!(parsed.mode, "power");
        assert!(!parsed.steps);
        assert!(!parsed.pretty);
    }

    #[test]
    fn substitute_run_options_parse_optional_json_fast_paths_common_shapes() {
        let parsed = SubstituteRunOptions::parse_optional_json(Some("{}"));
        assert_eq!(parsed.mode, "power");
        assert!(!parsed.steps);
        assert!(!parsed.pretty);

        let parsed = SubstituteRunOptions::parse_optional_json(Some(r#"{"steps":true}"#));
        assert_eq!(parsed.mode, "power");
        assert!(parsed.steps);
        assert!(!parsed.pretty);

        let parsed = SubstituteRunOptions::parse_optional_json(Some(r#"{"mode":"exact"}"#));
        assert_eq!(parsed.mode, "exact");
        assert!(!parsed.steps);
        assert!(!parsed.pretty);
    }

    #[test]
    fn substitute_run_options_from_mode_flags_sets_fields() {
        let parsed = SubstituteRunOptions::from_mode_flags("exact", true, true);
        assert_eq!(parsed.mode, "exact");
        assert!(parsed.steps);
        assert!(parsed.pretty);
    }

    #[test]
    fn parse_eval_special_command_parses_solve_solve_system_derive_equiv_and_limit() {
        let solve = parse_eval_special_command("solve((x+1)=0, x)").expect("solve");
        assert_eq!(
            solve,
            super::EvalSpecialCommand::Solve {
                equation: "(x+1)=0".to_string(),
                var: "x".to_string()
            }
        );

        let solve_system =
            parse_eval_special_command("solve_system(x+y=3; x-y=1; x; y)").expect("solve_system");
        assert_eq!(
            solve_system,
            super::EvalSpecialCommand::SolveSystem {
                input: "x+y=3; x-y=1; x; y".to_string(),
            }
        );

        let derive = parse_eval_special_command("derive x + x, 2*x").expect("derive");
        assert_eq!(
            derive,
            super::EvalSpecialCommand::Derive {
                input: "x + x, 2*x".to_string(),
            }
        );

        let derive_fn = parse_eval_special_command("derive(x + x, 2*x)").expect("derive fn");
        assert_eq!(
            derive_fn,
            super::EvalSpecialCommand::Derive {
                input: "x + x, 2*x".to_string(),
            }
        );

        let equiv = parse_eval_special_command("equiv x + 1, 1 + x").expect("equiv");
        assert_eq!(
            equiv,
            super::EvalSpecialCommand::Equiv {
                input: "x + 1, 1 + x".to_string(),
            }
        );

        let equiv_fn = parse_eval_special_command("equiv(x + 1, 1 + x)").expect("equiv fn");
        assert_eq!(
            equiv_fn,
            super::EvalSpecialCommand::Equiv {
                input: "x + 1, 1 + x".to_string(),
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

        let finite_limit = parse_eval_special_command("limit(ln(x), x, -1)").expect("limit");
        assert_eq!(
            finite_limit,
            super::EvalSpecialCommand::Limit {
                expr: "ln(x)".to_string(),
                var: "x".to_string(),
                approach: EvalLimitApproach::Finite("-1".to_string()),
            }
        );

        let finite_pi_limit = parse_eval_special_command("limit(sin(x), x, pi)").expect("limit");
        assert_eq!(
            finite_pi_limit,
            super::EvalSpecialCommand::Limit {
                expr: "sin(x)".to_string(),
                var: "x".to_string(),
                approach: EvalLimitApproach::Finite("pi".to_string()),
            }
        );

        let finite_e_limit = parse_eval_special_command("limit(ln(x), x, e)").expect("limit");
        assert_eq!(
            finite_e_limit,
            super::EvalSpecialCommand::Limit {
                expr: "ln(x)".to_string(),
                var: "x".to_string(),
                approach: EvalLimitApproach::Finite("e".to_string()),
            }
        );

        let finite_right_limit =
            parse_eval_special_command("limit(abs(x)/x, x, 0+)").expect("limit");
        assert_eq!(
            finite_right_limit,
            super::EvalSpecialCommand::Limit {
                expr: "abs(x)/x".to_string(),
                var: "x".to_string(),
                approach: EvalLimitApproach::FiniteFromRight("0".to_string()),
            }
        );

        let finite_pi_right_limit =
            parse_eval_special_command("limit(1/sin(x), x, pi+)").expect("limit");
        assert_eq!(
            finite_pi_right_limit,
            super::EvalSpecialCommand::Limit {
                expr: "1/sin(x)".to_string(),
                var: "x".to_string(),
                approach: EvalLimitApproach::FiniteFromRight("pi".to_string()),
            }
        );

        let finite_e_right_limit =
            parse_eval_special_command("limit(ln(x), x, e+)").expect("limit");
        assert_eq!(
            finite_e_right_limit,
            super::EvalSpecialCommand::Limit {
                expr: "ln(x)".to_string(),
                var: "x".to_string(),
                approach: EvalLimitApproach::FiniteFromRight("e".to_string()),
            }
        );

        let finite_left_limit =
            parse_eval_special_command("limit(abs(x)/x, x, 0, left)").expect("limit");
        assert_eq!(
            finite_left_limit,
            super::EvalSpecialCommand::Limit {
                expr: "abs(x)/x".to_string(),
                var: "x".to_string(),
                approach: EvalLimitApproach::FiniteFromLeft("0".to_string()),
            }
        );
    }

    #[test]
    fn parse_eval_special_command_rejects_invalid_forms() {
        assert!(parse_eval_special_command("solve(x+1=0)").is_none());
        assert!(parse_eval_special_command("limit(x, x, sideways)").is_none());
        assert!(parse_eval_special_command("limit(abs(x)/x, x, 0++)").is_none());
        assert!(parse_eval_special_command("limit(abs(x)/x, x, 0, safe)").is_none());
        assert!(parse_eval_special_command("x + 1").is_none());
    }

    #[test]
    fn parse_eval_special_command_parses_solve_list_system_form() {
        // `solve([eqs], [vars])` routes to the same SolveSystem pipeline as
        // `solve_system(...)`, normalising to the `eq; ...; var; ...` spec.
        let two_by_two =
            parse_eval_special_command("solve([x+y=3, x-y=1], [x, y])").expect("list system");
        assert_eq!(
            two_by_two,
            super::EvalSpecialCommand::SolveSystem {
                input: "x+y=3; x-y=1; x; y".to_string(),
            }
        );

        // A bare expression (no top-level `=`) is read as `expr = 0`.
        let implicit_zero =
            parse_eval_special_command("solve([x+y-3, x-y-1], [x, y])").expect("implicit zero");
        assert_eq!(
            implicit_zero,
            super::EvalSpecialCommand::SolveSystem {
                input: "x+y-3 = 0; x-y-1 = 0; x; y".to_string(),
            }
        );

        // 3×3 with commas inside the (function-call) terms stays at the right depth.
        let three =
            parse_eval_special_command("solve([x+y+z=6, x-y=0, z=3], [x, y, z])").expect("3x3");
        assert_eq!(
            three,
            super::EvalSpecialCommand::SolveSystem {
                input: "x+y+z=6; x-y=0; z=3; x; y; z".to_string(),
            }
        );
    }

    #[test]
    fn parse_eval_special_command_declines_malformed_list_system() {
        // Single equation/variable -> not a system (use `solve(eq, var)`).
        assert!(parse_solve_system_list_command("solve([x+y=1], [x])").is_none());
        // Mismatched equation/variable counts.
        assert!(parse_solve_system_list_command("solve([x+y=3, x-y=1], [x, y, z])").is_none());
        // Second argument not a bracketed list.
        assert!(parse_solve_system_list_command("solve([x+y=3, x-y=1], x)").is_none());
        // A non-identifier "variable".
        assert!(parse_solve_system_list_command("solve([x+y=3, x-y=1], [x, 2*y])").is_none());
        // Empty groups.
        assert!(parse_solve_system_list_command("solve([], [])").is_none());
        // The single-equation `solve(eq, var)` form is NOT captured by the list parser.
        assert!(parse_solve_system_list_command("solve(x+y=3, x)").is_none());
    }

    #[test]
    fn parse_eval_special_command_parses_dsolve_forms() {
        // Canonical 3-arg form.
        assert_eq!(
            super::parse_eval_special_command("dsolve(diff(y,x)=x*y, y, x)"),
            Some(super::EvalSpecialCommand::Dsolve {
                equation: "diff(y,x)=x*y".to_string(),
                func: "y".to_string(),
                var: "x".to_string(),
                conditions: vec![],
            })
        );
        // Arity-2 sugar: the variable is inferred from diff(y, <var>).
        assert_eq!(
            super::parse_eval_special_command("dsolve(diff(y,x)=x*y, y)"),
            Some(super::EvalSpecialCommand::Dsolve {
                equation: "diff(y,x)=x*y".to_string(),
                func: "y".to_string(),
                var: "x".to_string(),
                conditions: vec![],
            })
        );
        // Conditions stay textual; a non-identifier third arg is a condition.
        assert_eq!(
            super::parse_eval_special_command("dsolve(diff(y,x)=-y, y, x, y(0)=3)"),
            Some(super::EvalSpecialCommand::Dsolve {
                equation: "diff(y,x)=-y".to_string(),
                func: "y".to_string(),
                var: "x".to_string(),
                conditions: vec!["y(0)=3".to_string()],
            })
        );
        assert_eq!(
            super::parse_eval_special_command("dsolve(diff(y,x)=-y, y, y(0)=3)"),
            Some(super::EvalSpecialCommand::Dsolve {
                equation: "diff(y,x)=-y".to_string(),
                func: "y".to_string(),
                var: "x".to_string(),
                conditions: vec!["y(0)=3".to_string()],
            })
        );
        // Second-order raw shape still parses at the wire (the action owns the
        // higher-order decline).
        assert_eq!(
            super::parse_eval_special_command("dsolve(diff(y,x,2)+4*y=0, y, x)"),
            Some(super::EvalSpecialCommand::Dsolve {
                equation: "diff(y,x,2)+4*y=0".to_string(),
                func: "y".to_string(),
                var: "x".to_string(),
                conditions: vec![],
            })
        );
        // No prefix collision in either direction.
        assert!(matches!(
            super::parse_eval_special_command("solve(x^2=4, x)"),
            Some(super::EvalSpecialCommand::Solve { .. })
        ));
    }

    #[test]
    fn parse_eval_dsolve_command_error_reports_usage_and_sympy_style() {
        use super::parse_eval_dsolve_command_error as err_of;
        // Well-formed commands produce no pre-pass error.
        assert!(err_of("dsolve(diff(y,x)=x*y, y, x)").is_none());
        assert!(err_of("dsolve(diff(y,x)=x*y, y)").is_none());
        // Non-dsolve inputs are untouched.
        assert!(err_of("solve(x^2=4, x)").is_none());
        // Composed expressions decline to general eval (no wire error).
        assert!(err_of("dsolve(diff(y,x)=y, y, x)+1").is_none());
        // The D10 prey: `dsolve(y, x)` is now an EXPLICIT usage error (no diff).
        assert!(err_of("dsolve(y, x)")
            .expect("usage error")
            .contains("contains no diff"));
        // SymPy-style `y(x)` unknown names the fix.
        assert!(err_of("dsolve(diff(y(x),x)=y(x), y(x))")
            .expect("usage error")
            .contains("not y(x)"));
        // Wrong explicit variable names the mismatch.
        assert!(err_of("dsolve(diff(y,t)=y, y, x)")
            .expect("usage error")
            .contains("with respect to t"));
        // Ambiguous arity-2 inference asks for the explicit variable.
        assert!(err_of("dsolve(diff(y,x)+diff(y,t)=0, y)")
            .expect("usage error")
            .contains("Ambiguous"));
        // Bare malformed head.
        assert!(err_of("dsolve(y)")
            .expect("usage error")
            .contains("Invalid dsolve"));
    }

    #[test]
    fn scan_dsolve_diff_vars_respects_word_boundaries_and_nesting() {
        assert_eq!(
            super::scan_dsolve_diff_vars("diff(y,x)=x*y", "y"),
            vec!["x"]
        );
        // Nested diff(diff(y,x),x): the inner call carries the variable.
        assert_eq!(
            super::scan_dsolve_diff_vars("diff(diff(y,x),x)=0", "y"),
            vec!["x"]
        );
        // Word boundary: mydiff does not match.
        assert!(super::scan_dsolve_diff_vars("mydiff(y,x)=0", "y").is_empty());
        // Other-function diffs do not attach to y.
        assert!(super::scan_dsolve_diff_vars("diff(z,x)=z", "y").is_empty());
        // Spaces tolerated.
        assert_eq!(
            super::scan_dsolve_diff_vars("diff( y , x ) = y", "y"),
            vec!["x"]
        );
    }

    #[test]
    fn parse_eval_limit_command_error_allows_finite_points_and_reports_bad_directions() {
        assert!(parse_eval_limit_command_error("limit(ln(x), x, -1)").is_none());
        assert_eq!(
            parse_eval_limit_command_error("limit(ln(x), x, sideways)").as_deref(),
            Some("Unsupported limit direction `sideways`. Use infinity, -infinity, or a finite point.")
        );
        assert_eq!(
            parse_eval_limit_command_error("limit(abs(x)/x, x, 0++)").as_deref(),
            Some("Unsupported limit direction `0++`. Use infinity, -infinity, or a finite point.")
        );
        assert!(parse_eval_limit_command_error("limit(abs(x)/x, x, 0+)").is_none());
        assert!(parse_eval_limit_command_error("limit(abs(x)/x, x, 0, right)").is_none());
        assert!(parse_eval_limit_command_error("limit(sin(x), x, pi)").is_none());
        assert!(parse_eval_limit_command_error("limit(1/sin(x), x, pi+)").is_none());
        assert!(parse_eval_limit_command_error("limit(1/sin(x), x, pi, right)").is_none());
        assert!(parse_eval_limit_command_error("limit(ln(x), x, e)").is_none());
        assert!(parse_eval_limit_command_error("limit(ln(x), x, e+)").is_none());
        assert!(parse_eval_limit_command_error("limit(ln(x), x, e, right)").is_none());
        assert_eq!(
            parse_eval_limit_command_error("limit(abs(x)/x, x, 0, upward)").as_deref(),
            Some("Invalid limit command. Expected limit(expr, var, approach); eval limit syntax does not accept a fourth argument.")
        );
        assert_eq!(
            parse_eval_limit_command_error("limit(abs(x)/x, x, 0, safe)").as_deref(),
            Some("Invalid limit command. Expected limit(expr, var, approach); eval limit syntax does not accept a fourth argument.")
        );
        assert!(parse_eval_limit_command_error("limit(ln(x), x, -infinity)").is_none());
        assert!(parse_eval_limit_command_error("x + 1").is_none());
    }
}
