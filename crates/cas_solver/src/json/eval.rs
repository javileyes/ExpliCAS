use cas_api_models::{
    BudgetJsonInfo, EngineJsonError, EngineJsonResponse, EngineJsonStep, JsonRunOptions, SpanJson,
};
use cas_ast::{Expr, ExprId};
use cas_engine::eval::{CacheHitTrace, EvalSession, EvalStore};

/// Evaluate an expression and return JSON response.
///
/// This is the **canonical entry point** for all JSON-returning evaluation.
/// Both CLI and FFI should use this to ensure consistent behavior.
///
/// # Arguments
/// * `expr` - Expression string to evaluate
/// * `opts_json` - Options JSON string (see `JsonRunOptions`)
///
/// # Returns
/// JSON string with `EngineJsonResponse` (schema v1).
/// Always returns valid JSON, even on errors.
///
/// # Example
/// ```
/// use cas_solver::eval_str_to_json;
///
/// let json = eval_str_to_json("x + x", r#"{"budget":{"preset":"cli"}}"#);
/// assert!(json.contains("\"ok\":true"));
/// ```
pub fn eval_str_to_json(expr: &str, opts_json: &str) -> String {
    // Parse options (with defaults)
    let opts: JsonRunOptions = match serde_json::from_str(opts_json) {
        Ok(o) => o,
        Err(e) => {
            // Invalid optsJson -> return error response
            let budget = BudgetJsonInfo::new("unknown", true);
            let error = EngineJsonError::invalid_input(
                format!("Invalid options JSON: {}", e),
                serde_json::json!({ "error": e.to_string() }),
            );
            let resp = EngineJsonResponse::err(error, budget);
            return if opts_json.contains("\"pretty\":true") {
                resp.to_json_pretty()
            } else {
                resp.to_json()
            };
        }
    };

    let strict = opts.budget.mode == "strict";
    let budget_info = BudgetJsonInfo::new(&opts.budget.preset, strict);

    // Create engine and explicit eval components (stateless-friendly API)
    let mut engine = cas_engine::eval::Engine::new();
    let mut session = JsonEvalSession::new(cas_engine::options::EvalOptions::default());

    // Parse expression
    let parsed = match cas_parser::parse(expr, &mut engine.simplifier.context) {
        Ok(id) => id,
        Err(e) => {
            let error = EngineJsonError::parse(
                e.to_string(),
                e.span().map(|s| SpanJson {
                    start: s.start,
                    end: s.end,
                }),
            );
            let resp = EngineJsonResponse::err(error, budget_info);
            return if opts.pretty {
                resp.to_json_pretty()
            } else {
                resp.to_json()
            };
        }
    };

    // Build eval request
    let req = cas_engine::eval::EvalRequest {
        raw_input: expr.to_string(),
        parsed,
        action: cas_engine::eval::EvalAction::Simplify,
        auto_store: false,
    };

    // Evaluate
    let output = match engine.eval(&mut session, req) {
        Ok(o) => o,
        Err(e) => {
            // anyhow::Error - create generic error
            let error = EngineJsonError::simple("InternalError", "E_INTERNAL", e.to_string());
            let resp = EngineJsonResponse::err(error, budget_info);
            return if opts.pretty {
                resp.to_json_pretty()
            } else {
                resp.to_json()
            };
        }
    };

    // Format result
    let result_str = match &output.result {
        cas_engine::eval::EvalResult::Expr(e) => {
            let clean = cas_engine::engine::strip_all_holds(&mut engine.simplifier.context, *e);
            format!(
                "{}",
                cas_formatter::DisplayExpr {
                    context: &engine.simplifier.context,
                    id: clean
                }
            )
        }
        cas_engine::eval::EvalResult::Set(v) if !v.is_empty() => {
            let clean = cas_engine::engine::strip_all_holds(&mut engine.simplifier.context, v[0]);
            format!(
                "{}",
                cas_formatter::DisplayExpr {
                    context: &engine.simplifier.context,
                    id: clean
                }
            )
        }
        cas_engine::eval::EvalResult::Bool(b) => b.to_string(),
        _ => "(no result)".to_string(),
    };

    // Build steps (if requested)
    let steps = if opts.steps {
        output
            .steps
            .iter()
            .map(|s| {
                let before_str = s.global_before.map(|id| {
                    let clean =
                        cas_engine::engine::strip_all_holds(&mut engine.simplifier.context, id);
                    format!(
                        "{}",
                        cas_formatter::DisplayExpr {
                            context: &engine.simplifier.context,
                            id: clean
                        }
                    )
                });
                let after_str = s.global_after.map(|id| {
                    let clean =
                        cas_engine::engine::strip_all_holds(&mut engine.simplifier.context, id);
                    format!(
                        "{}",
                        cas_formatter::DisplayExpr {
                            context: &engine.simplifier.context,
                            id: clean
                        }
                    )
                });
                EngineJsonStep {
                    phase: "Simplify".to_string(),
                    rule: s.rule_name.clone(),
                    before: before_str.unwrap_or_default(),
                    after: after_str.unwrap_or_default(),
                    substeps: vec![],
                }
            })
            .collect()
    } else {
        vec![]
    };

    let resp = EngineJsonResponse::ok_with_steps(result_str, steps, budget_info);
    if opts.pretty {
        resp.to_json_pretty()
    } else {
        resp.to_json()
    }
}

#[derive(Default)]
struct JsonStore {
    next_id: u64,
}

impl JsonStore {
    fn new() -> Self {
        Self::default()
    }
}

impl EvalStore for JsonStore {
    fn push_raw_input(
        &mut self,
        _ctx: &cas_ast::Context,
        _parsed: ExprId,
        _raw_input: String,
    ) -> u64 {
        self.next_id = self.next_id.saturating_add(1);
        self.next_id
    }

    fn touch_cached(&mut self, _entry_id: u64) {}

    fn update_diagnostics(&mut self, _id: u64, _diagnostics: cas_engine::diagnostics::Diagnostics) {
    }

    fn update_simplified(&mut self, _id: u64, _cache: cas_engine::eval::SimplifiedCache) {}
}

fn first_session_ref(ctx: &cas_ast::Context, root: ExprId) -> Option<u64> {
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        match ctx.get(id) {
            Expr::SessionRef(ref_id) => return Some(*ref_id),
            Expr::Add(l, r)
            | Expr::Sub(l, r)
            | Expr::Mul(l, r)
            | Expr::Div(l, r)
            | Expr::Pow(l, r) => {
                stack.push(*l);
                stack.push(*r);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(*inner),
            Expr::Function(_, args) => stack.extend(args.iter().copied()),
            Expr::Matrix { data, .. } => stack.extend(data.iter().copied()),
            Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) => {}
        }
    }
    None
}

struct JsonEvalSession {
    store: JsonStore,
    options: cas_engine::options::EvalOptions,
    profile_cache: cas_engine::profile_cache::ProfileCache,
}

impl JsonEvalSession {
    fn new(options: cas_engine::options::EvalOptions) -> Self {
        Self {
            store: JsonStore::new(),
            options,
            profile_cache: cas_engine::profile_cache::ProfileCache::new(),
        }
    }
}

impl EvalSession for JsonEvalSession {
    type Store = JsonStore;

    fn store_mut(&mut self) -> &mut Self::Store {
        &mut self.store
    }

    fn options(&self) -> &cas_engine::options::EvalOptions {
        &self.options
    }

    fn profile_cache_mut(&mut self) -> &mut cas_engine::profile_cache::ProfileCache {
        &mut self.profile_cache
    }

    fn resolve_all(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
    ) -> Result<ExprId, cas_engine::eval::EvalResolveError> {
        if let Some(id) = first_session_ref(ctx, expr) {
            return Err(cas_engine::eval::EvalResolveError::NotFound(id));
        }
        Ok(expr)
    }

    fn resolve_all_with_diagnostics(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
    ) -> Result<
        (
            ExprId,
            cas_engine::diagnostics::Diagnostics,
            Vec<CacheHitTrace>,
        ),
        cas_engine::eval::EvalResolveError,
    > {
        if let Some(id) = first_session_ref(ctx, expr) {
            return Err(cas_engine::eval::EvalResolveError::NotFound(id));
        }
        Ok((expr, cas_engine::diagnostics::Diagnostics::new(), vec![]))
    }
}
