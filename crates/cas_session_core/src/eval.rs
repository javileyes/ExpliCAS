//! Session contracts used by engine-level eval orchestration.
//!
//! Kept in `cas_session_core` so stateful session crates can implement
//! them without defining session abstractions inside `cas_engine`.

use std::sync::Arc;

use cas_ast::{Context, ExprId};

/// Store-side operations required by `Engine::eval`.
pub trait EvalStore {
    type DomainMode;
    type RequiredItem;
    type Step;
    type Diagnostics: Clone;

    fn push_raw_expr(&mut self, expr: ExprId, raw_input: String) -> u64;
    fn push_raw_equation(&mut self, lhs: ExprId, rhs: ExprId, raw_input: String) -> u64;
    fn touch_cached(&mut self, entry_id: u64);
    fn update_diagnostics(&mut self, id: u64, diagnostics: Self::Diagnostics);
    fn update_simplified(
        &mut self,
        id: u64,
        domain: Self::DomainMode,
        expr: ExprId,
        requires: Vec<Self::RequiredItem>,
        steps: Option<Arc<Vec<Self::Step>>>,
    );
}

/// Session-side operations required by `Engine::eval`.
pub trait EvalSession {
    type Store: EvalStore;
    type Options;
    type Diagnostics: Clone;

    fn store_mut(&mut self) -> &mut Self::Store;
    fn options(&self) -> &Self::Options;
    fn options_mut(&mut self) -> &mut Self::Options;

    fn resolve_all_with_diagnostics(
        &self,
        ctx: &mut Context,
        expr: ExprId,
    ) -> anyhow::Result<(ExprId, Self::Diagnostics, Vec<u64>)>;

    /// Optional fast path for direct cached eval of a root `#N` reference.
    ///
    /// Default sessions return `None` and fall back to the regular
    /// resolve + engine dispatch path.
    fn try_direct_cached_eval(
        &mut self,
        _ctx: &mut Context,
        _expr: ExprId,
        _auto_store: bool,
    ) -> anyhow::Result<Option<DirectCachedEval<Self::Diagnostics>>> {
        Ok(None)
    }
}

/// Inputs resolved against session/environment for a single eval request.
#[derive(Debug)]
pub struct ResolvedEvalInput<Diagnostics> {
    pub resolved: ExprId,
    pub inherited_diagnostics: Diagnostics,
    pub cache_hits: Vec<u64>,
    pub resolved_equiv_other: Option<ExprId>,
}

/// Optional direct cached-eval payload for pure session-reference lookups.
#[derive(Debug)]
pub struct DirectCachedEval<Diagnostics> {
    pub resolved: ExprId,
    pub inherited_diagnostics: Diagnostics,
}

/// Input options for resolve + pre-dispatch preparation.
#[derive(Debug, Clone)]
pub struct ResolvePrepareConfig {
    pub parsed: ExprId,
    pub raw_input: String,
    pub auto_store: bool,
    pub equiv_other: Option<ExprId>,
    pub cache_step_max_shown: usize,
}

/// Prepared eval input after session resolution and pre-dispatch store updates.
#[derive(Debug)]
pub struct PreparedEvalDispatch<Diagnostics, StepT> {
    pub stored_id: Option<u64>,
    pub resolved: ExprId,
    pub inherited_diagnostics: Diagnostics,
    pub resolved_equiv_other: Option<ExprId>,
    pub cache_hit_step: Option<StepT>,
}

/// Simplified-result cache payload for post-dispatch store updates.
#[derive(Debug, Clone)]
pub struct SimplifiedUpdate<DomainMode, RequiredItem, Step> {
    pub domain: DomainMode,
    pub expr: ExprId,
    pub requires: Vec<RequiredItem>,
    pub steps: Option<Arc<Vec<Step>>>,
}

/// Type-constrained view over [`EvalStore`].
pub trait TypedEvalStore<DomainMode, RequiredItem, Step, Diagnostics>:
    EvalStore<
    DomainMode = DomainMode,
    RequiredItem = RequiredItem,
    Step = Step,
    Diagnostics = Diagnostics,
>
where
    Diagnostics: Clone,
{
}

impl<T, DomainMode, RequiredItem, Step, Diagnostics>
    TypedEvalStore<DomainMode, RequiredItem, Step, Diagnostics> for T
where
    T: EvalStore<
        DomainMode = DomainMode,
        RequiredItem = RequiredItem,
        Step = Step,
        Diagnostics = Diagnostics,
    >,
    Diagnostics: Clone,
{
}

/// Type-constrained view over [`EvalSession`] with fully specified store/output types.
pub trait TypedEvalSession<DomainMode, RequiredItem, Step, Diagnostics, Options>:
    EvalSession<Options = Options, Diagnostics = Diagnostics>
where
    Diagnostics: Clone,
    Self::Store: TypedEvalStore<DomainMode, RequiredItem, Step, Diagnostics>,
{
}

impl<T, DomainMode, RequiredItem, Step, Diagnostics, Options>
    TypedEvalSession<DomainMode, RequiredItem, Step, Diagnostics, Options> for T
where
    T: EvalSession<Options = Options, Diagnostics = Diagnostics>,
    Diagnostics: Clone,
    T::Store: TypedEvalStore<DomainMode, RequiredItem, Step, Diagnostics>,
{
}

/// No-op store for stateless eval sessions.
///
/// Useful in contexts where eval should run without history/session writes.
#[derive(Debug)]
pub struct NoopEvalStore<DomainMode, RequiredItem, Step, Diagnostics> {
    _marker: std::marker::PhantomData<(DomainMode, RequiredItem, Step, Diagnostics)>,
}

impl<DomainMode, RequiredItem, Step, Diagnostics> Default
    for NoopEvalStore<DomainMode, RequiredItem, Step, Diagnostics>
{
    fn default() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}

impl<DomainMode, RequiredItem, Step, Diagnostics> EvalStore
    for NoopEvalStore<DomainMode, RequiredItem, Step, Diagnostics>
where
    Diagnostics: Clone,
{
    type DomainMode = DomainMode;
    type RequiredItem = RequiredItem;
    type Step = Step;
    type Diagnostics = Diagnostics;

    fn push_raw_expr(&mut self, _expr: ExprId, _raw_input: String) -> u64 {
        0
    }

    fn push_raw_equation(&mut self, _lhs: ExprId, _rhs: ExprId, _raw_input: String) -> u64 {
        0
    }

    fn touch_cached(&mut self, _entry_id: u64) {}

    fn update_diagnostics(&mut self, _id: u64, _diagnostics: Self::Diagnostics) {}

    fn update_simplified(
        &mut self,
        _id: u64,
        _domain: Self::DomainMode,
        _expr: ExprId,
        _requires: Vec<Self::RequiredItem>,
        _steps: Option<Arc<Vec<Self::Step>>>,
    ) {
    }
}

/// Default stateless eval session adapter.
///
/// It rejects expressions containing session references (`#N`) and otherwise
/// forwards the input expression unchanged.
#[derive(Debug)]
pub struct StatelessEvalSession<Options, DomainMode, RequiredItem, Step, Diagnostics> {
    store: NoopEvalStore<DomainMode, RequiredItem, Step, Diagnostics>,
    options: Options,
}

impl<Options, DomainMode, RequiredItem, Step, Diagnostics>
    StatelessEvalSession<Options, DomainMode, RequiredItem, Step, Diagnostics>
{
    pub fn new(options: Options) -> Self {
        Self {
            store: NoopEvalStore::default(),
            options,
        }
    }
}

impl<Options, DomainMode, RequiredItem, Step, Diagnostics> EvalSession
    for StatelessEvalSession<Options, DomainMode, RequiredItem, Step, Diagnostics>
where
    Diagnostics: Clone + Default,
{
    type Store = NoopEvalStore<DomainMode, RequiredItem, Step, Diagnostics>;
    type Options = Options;
    type Diagnostics = Diagnostics;

    fn store_mut(&mut self) -> &mut Self::Store {
        &mut self.store
    }

    fn options(&self) -> &Self::Options {
        &self.options
    }

    fn options_mut(&mut self) -> &mut Self::Options {
        &mut self.options
    }

    fn resolve_all_with_diagnostics(
        &self,
        ctx: &mut Context,
        expr: ExprId,
    ) -> anyhow::Result<(ExprId, Self::Diagnostics, Vec<u64>)> {
        if let Some(ref_id) = crate::resolve::first_session_ref(ctx, expr) {
            return Err(stateless_session_ref_error(ref_id));
        }
        Ok((expr, Diagnostics::default(), Vec::new()))
    }
}

/// Build a stable, user-facing cache-hit summary message.
pub fn format_cache_hit_summary(cache_hits: &[u64], max_shown: usize) -> Option<String> {
    if cache_hits.is_empty() {
        return None;
    }

    let mut ids: Vec<u64> = cache_hits.to_vec();
    ids.sort_unstable();

    let shown: Vec<String> = ids
        .iter()
        .take(max_shown)
        .map(|id| format!("#{}", id))
        .collect();
    let suffix = if ids.len() > max_shown {
        format!(" (+{})", ids.len() - max_shown)
    } else {
        String::new()
    };

    Some(format!(
        "Used cached simplified result from {}{}",
        shown.join(", "),
        suffix
    ))
}

/// Build a caller-defined cache-hit step payload.
///
/// Returns `None` when there are no cache hits to report.
pub fn build_cache_hit_step_with<StepT, FBuild>(
    cache_hits: &[u64],
    max_shown: usize,
    original_expr: ExprId,
    resolved_expr: ExprId,
    mut build_step: FBuild,
) -> Option<StepT>
where
    FBuild: FnMut(String, ExprId, ExprId) -> StepT,
{
    let description = format_cache_hit_summary(cache_hits, max_shown)?;
    Some(build_step(description, original_expr, resolved_expr))
}

/// Collect warnings from per-step assumption-like events with message deduplication.
///
/// The first occurrence of each message wins. Rule metadata is captured from the
/// step that emitted that first occurrence.
pub fn collect_warnings_with<
    StepT,
    EventT,
    WarningT,
    RuleT,
    EIter,
    FEvents,
    FSkip,
    FMessage,
    FRule,
    FBuild,
>(
    steps: &[StepT],
    mut events: FEvents,
    mut skip_event: FSkip,
    mut event_message: FMessage,
    mut step_rule: FRule,
    mut build_warning: FBuild,
) -> Vec<WarningT>
where
    EIter: IntoIterator<Item = EventT>,
    FEvents: FnMut(&StepT) -> EIter,
    FSkip: FnMut(&EventT) -> bool,
    FMessage: FnMut(&EventT) -> String,
    FRule: FnMut(&StepT) -> RuleT,
    RuleT: Clone,
    FBuild: FnMut(String, RuleT) -> WarningT,
{
    use std::collections::HashSet;

    let mut seen: HashSet<String> = HashSet::new();
    let mut warnings = Vec::new();

    for step in steps {
        let rule = step_rule(step);
        for event in events(step) {
            if skip_event(&event) {
                continue;
            }
            let message = event_message(&event);
            if seen.insert(message.clone()) {
                warnings.push(build_warning(message, rule.clone()));
            }
        }
    }

    warnings
}

/// Collect per-step items with deduplication by display key.
///
/// The first item for each display string is retained.
pub fn collect_step_items_with_display_dedup<StepT, ItemT, OutT, FItems, FDisplay, FBuild>(
    steps: &[StepT],
    mut items_of: FItems,
    mut display_of: FDisplay,
    mut build_out: FBuild,
) -> Vec<OutT>
where
    FItems: FnMut(&StepT) -> Vec<ItemT>,
    FDisplay: FnMut(&ItemT) -> String,
    FBuild: FnMut(ItemT) -> OutT,
{
    use std::collections::HashSet;

    let mut seen: HashSet<String> = HashSet::new();
    let mut out = Vec::new();
    for step in steps {
        for item in items_of(step) {
            let key = display_of(&item);
            if seen.insert(key) {
                out.push(build_out(item));
            }
        }
    }
    out
}

/// Build canonical error used when stateless eval sees a `#N` reference.
pub fn stateless_session_ref_error(entry_id: crate::types::EntryId) -> anyhow::Error {
    anyhow::anyhow!(
        "Session reference #{} requires stateful eval (Engine::eval with EvalSession)",
        entry_id
    )
}

/// Map session-core resolve failures into user-facing eval errors.
pub fn map_resolve_error_to_anyhow(err: crate::types::ResolveError) -> anyhow::Error {
    match err {
        crate::types::ResolveError::NotFound(id) => {
            anyhow::anyhow!("Session reference #{} not found", id)
        }
        crate::types::ResolveError::CircularReference(id) => {
            anyhow::anyhow!("Circular reference detected involving #{}", id)
        }
    }
}

/// Resolve primary input (and optional equivalence operand) via the session.
pub fn resolve_eval_input<S: EvalSession>(
    session: &S,
    ctx: &mut Context,
    parsed: ExprId,
    equiv_other: Option<ExprId>,
) -> anyhow::Result<ResolvedEvalInput<S::Diagnostics>> {
    let (resolved, inherited_diagnostics, cache_hits) = session
        .resolve_all_with_diagnostics(ctx, parsed)
        .map_err(|e| anyhow::anyhow!("Resolution error: {}", e))?;

    let resolved_equiv_other = if let Some(other) = equiv_other {
        Some(
            session
                .resolve_all_with_diagnostics(ctx, other)
                .map_err(|e| anyhow::anyhow!("Resolution error in other: {}", e))?
                .0,
        )
    } else {
        None
    };

    Ok(ResolvedEvalInput {
        resolved,
        inherited_diagnostics,
        cache_hits,
        resolved_equiv_other,
    })
}

/// Persist raw parsed input into a session store when auto-store is enabled.
///
/// Returns the new entry id when stored, or `None` when `auto_store` is false.
pub fn store_raw_input<StoreT: EvalStore>(
    store: &mut StoreT,
    ctx: &Context,
    parsed: ExprId,
    raw_input: String,
    auto_store: bool,
) -> Option<u64> {
    if !auto_store {
        return None;
    }

    let id = if let Some((lhs, rhs)) = cas_ast::eq::unwrap_eq(ctx, parsed) {
        store.push_raw_equation(lhs, rhs, raw_input)
    } else {
        store.push_raw_expr(parsed, raw_input)
    };
    Some(id)
}

/// Touch all cache-hit entries in store order.
pub fn touch_cache_hits<StoreT: EvalStore>(store: &mut StoreT, cache_hits: &[u64]) {
    for hit in cache_hits {
        store.touch_cached(*hit);
    }
}

/// Apply pre-dispatch store updates for one eval:
/// - optionally persist raw input (`auto_store`)
/// - touch cache-hit entries for LRU freshness
pub fn apply_pre_dispatch_store_updates<StoreT: EvalStore>(
    store: &mut StoreT,
    ctx: &Context,
    parsed: ExprId,
    raw_input: String,
    auto_store: bool,
    cache_hits: &[u64],
) -> Option<u64> {
    let stored_id = store_raw_input(store, ctx, parsed, raw_input, auto_store);
    touch_cache_hits(store, cache_hits);
    stored_id
}

/// Resolve eval input and apply pre-dispatch store/cache updates in one call.
pub fn resolve_and_prepare_dispatch<S, StepT, FBuildStep>(
    session: &mut S,
    ctx: &mut Context,
    config: ResolvePrepareConfig,
    mut build_step: FBuildStep,
) -> anyhow::Result<PreparedEvalDispatch<S::Diagnostics, StepT>>
where
    S: EvalSession,
    FBuildStep: FnMut(&Context, String, ExprId, ExprId) -> StepT,
{
    let ResolvePrepareConfig {
        parsed,
        raw_input,
        auto_store,
        equiv_other,
        cache_step_max_shown,
    } = config;

    let resolved_input = resolve_eval_input(session, ctx, parsed, equiv_other)?;
    let stored_id = apply_pre_dispatch_store_updates(
        session.store_mut(),
        ctx,
        parsed,
        raw_input,
        auto_store,
        &resolved_input.cache_hits,
    );
    let cache_hit_step = build_cache_hit_step_with(
        &resolved_input.cache_hits,
        cache_step_max_shown,
        parsed,
        resolved_input.resolved,
        |description, before, after| build_step(ctx, description, before, after),
    );

    Ok(PreparedEvalDispatch {
        stored_id,
        resolved: resolved_input.resolved,
        inherited_diagnostics: resolved_input.inherited_diagnostics,
        resolved_equiv_other: resolved_input.resolved_equiv_other,
        cache_hit_step,
    })
}

/// Apply post-dispatch store updates for one eval:
/// - persist final diagnostics when entry was auto-stored
/// - optionally persist simplified-cache payload
pub fn apply_post_dispatch_store_updates<StoreT: EvalStore>(
    store: &mut StoreT,
    stored_id: Option<u64>,
    diagnostics: StoreT::Diagnostics,
    simplified: Option<SimplifiedUpdate<StoreT::DomainMode, StoreT::RequiredItem, StoreT::Step>>,
) {
    let Some(id) = stored_id else {
        return;
    };

    store.update_diagnostics(id, diagnostics);

    if let Some(simplified) = simplified {
        store.update_simplified(
            id,
            simplified.domain,
            simplified.expr,
            simplified.requires,
            simplified.steps,
        );
    }
}

#[cfg(test)]
mod tests;
