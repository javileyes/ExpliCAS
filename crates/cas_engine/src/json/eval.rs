use super::response::*;
use crate::eval::{CacheHitTrace, EvalSession, EvalStore};
use cas_ast::ExprId;
use cas_session_core::types::{EntryId, EntryKind, RefMode, ResolveError};

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
/// use cas_engine::json::eval_str_to_json;
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
            let error = EngineJsonError {
                kind: "InvalidInput",
                code: "E_INVALID_INPUT",
                message: format!("Invalid options JSON: {}", e),
                span: None,
                details: serde_json::json!({ "error": e.to_string() }),
            };
            let resp = EngineJsonResponse {
                schema_version: SCHEMA_VERSION,
                ok: false,
                result: None,
                error: Some(error),
                steps: vec![],
                warnings: vec![],
                assumptions: vec![],
                budget,
            };
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
    let mut engine = crate::eval::Engine::new();
    let mut session = JsonEvalSession::new(crate::options::EvalOptions::default());

    // Parse expression
    let parsed = match cas_parser::parse(expr, &mut engine.simplifier.context) {
        Ok(id) => id,
        Err(e) => {
            let error = EngineJsonError {
                kind: "ParseError",
                code: "E_PARSE",
                message: e.to_string(),
                span: e.span().map(SpanJson::from),
                details: serde_json::Value::Null,
            };
            let resp = EngineJsonResponse {
                schema_version: SCHEMA_VERSION,
                ok: false,
                result: None,
                error: Some(error),
                steps: vec![],
                warnings: vec![],
                assumptions: vec![],
                budget: budget_info,
            };
            return if opts.pretty {
                resp.to_json_pretty()
            } else {
                resp.to_json()
            };
        }
    };

    // Build eval request
    let req = crate::eval::EvalRequest {
        raw_input: expr.to_string(),
        parsed,
        action: crate::eval::EvalAction::Simplify,
        auto_store: false,
    };

    // Evaluate
    let output = match engine.eval(&mut session, req) {
        Ok(o) => o,
        Err(e) => {
            // anyhow::Error - create generic error
            let error = EngineJsonError::simple("InternalError", "E_INTERNAL", e.to_string());
            let resp = EngineJsonResponse {
                schema_version: SCHEMA_VERSION,
                ok: false,
                result: None,
                error: Some(error),
                steps: vec![],
                warnings: vec![],
                assumptions: vec![],
                budget: budget_info,
            };
            return if opts.pretty {
                resp.to_json_pretty()
            } else {
                resp.to_json()
            };
        }
    };

    // Format result
    let result_str = match &output.result {
        crate::eval::EvalResult::Expr(e) => {
            let clean = crate::engine::strip_all_holds(&mut engine.simplifier.context, *e);
            format!(
                "{}",
                cas_ast::display::DisplayExpr {
                    context: &engine.simplifier.context,
                    id: clean
                }
            )
        }
        crate::eval::EvalResult::Set(v) if !v.is_empty() => {
            let clean = crate::engine::strip_all_holds(&mut engine.simplifier.context, v[0]);
            format!(
                "{}",
                cas_ast::display::DisplayExpr {
                    context: &engine.simplifier.context,
                    id: clean
                }
            )
        }
        crate::eval::EvalResult::Bool(b) => b.to_string(),
        _ => "(no result)".to_string(),
    };

    // Build steps (if requested)
    let steps = if opts.steps {
        output
            .steps
            .iter()
            .map(|s| {
                let before_str = s.global_before.map(|id| {
                    let clean = crate::engine::strip_all_holds(&mut engine.simplifier.context, id);
                    format!(
                        "{}",
                        cas_ast::display::DisplayExpr {
                            context: &engine.simplifier.context,
                            id: clean
                        }
                    )
                });
                let after_str = s.global_after.map(|id| {
                    let clean = crate::engine::strip_all_holds(&mut engine.simplifier.context, id);
                    format!(
                        "{}",
                        cas_ast::display::DisplayExpr {
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

type JsonStoreInner = cas_session_core::store::SessionStore<
    crate::diagnostics::Diagnostics,
    crate::eval::SimplifiedCache,
>;

struct JsonStore(JsonStoreInner);

impl JsonStore {
    fn new() -> Self {
        Self(JsonStoreInner::new())
    }
}

impl EvalStore for JsonStore {
    fn push(&mut self, kind: EntryKind, raw_input: String) -> EntryId {
        self.0.push(kind, raw_input)
    }

    fn touch_cached(&mut self, entry_id: EntryId) {
        self.0.touch_cached(entry_id);
    }

    fn update_diagnostics(&mut self, id: EntryId, diagnostics: crate::diagnostics::Diagnostics) {
        self.0.update_diagnostics(id, diagnostics);
    }

    fn update_simplified(&mut self, id: EntryId, cache: crate::eval::SimplifiedCache) {
        self.0.update_simplified(id, cache);
    }
}

struct JsonEvalSession {
    store: JsonStore,
    env: cas_session_core::env::Environment,
    options: crate::options::EvalOptions,
    profile_cache: crate::profile_cache::ProfileCache,
}

impl JsonEvalSession {
    fn new(options: crate::options::EvalOptions) -> Self {
        Self {
            store: JsonStore::new(),
            env: cas_session_core::env::Environment::new(),
            options,
            profile_cache: crate::profile_cache::ProfileCache::new(),
        }
    }
}

impl EvalSession for JsonEvalSession {
    type Store = JsonStore;

    fn store_mut(&mut self) -> &mut Self::Store {
        &mut self.store
    }

    fn options(&self) -> &crate::options::EvalOptions {
        &self.options
    }

    fn profile_cache_mut(&mut self) -> &mut crate::profile_cache::ProfileCache {
        &mut self.profile_cache
    }

    fn resolve_all(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
    ) -> Result<ExprId, ResolveError> {
        let mut lookup = |id: EntryId| self.store.0.get(id).map(|entry| entry.kind.clone());
        cas_session_core::resolve::resolve_all_with_lookup_and_env(
            ctx,
            expr,
            &mut lookup,
            &self.env,
        )
    }

    fn resolve_all_with_diagnostics(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
    ) -> Result<(ExprId, crate::diagnostics::Diagnostics, Vec<CacheHitTrace>), ResolveError> {
        let cache_key =
            crate::eval::SimplifyCacheKey::from_context(self.options.shared.semantics.domain_mode);

        let mut lookup = |id: EntryId| {
            let entry = self.store.0.get(id)?;
            Some(cas_session_core::resolve::ModeEntry {
                kind: entry.kind.clone(),
                requires: entry.diagnostics.requires.clone(),
                cache: entry.simplified.as_ref().map(|cache| {
                    cas_session_core::resolve::ModeCacheEntry {
                        key: cache.key.clone(),
                        expr: cache.expr,
                        requires: cache.requires.clone(),
                    }
                }),
            })
        };
        let mut same_requirement =
            |lhs: &crate::diagnostics::RequiredItem, rhs: &crate::diagnostics::RequiredItem| {
                lhs.cond == rhs.cond
            };
        let mut mark_session_propagated = |item: &mut crate::diagnostics::RequiredItem| {
            item.merge_origin(crate::diagnostics::RequireOrigin::SessionPropagated);
        };

        let resolved = cas_session_core::resolve::resolve_session_refs_with_mode_lookup(
            ctx,
            expr,
            RefMode::PreferSimplified,
            &cache_key,
            &mut lookup,
            &mut same_requirement,
            &mut mark_session_propagated,
        )?;

        let mut inherited = crate::diagnostics::Diagnostics::new();
        for item in resolved.requires {
            inherited.push_required(
                item.cond,
                crate::diagnostics::RequireOrigin::SessionPropagated,
            );
        }

        let fully_resolved = cas_session_core::env::substitute(ctx, &self.env, resolved.expr);
        Ok((fully_resolved, inherited, resolved.cache_hits))
    }
}
