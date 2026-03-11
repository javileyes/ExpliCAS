use cas_solver_core::diagnostics_model::{Diagnostics, RequireOrigin, RequiredItem};

use crate::{env::Environment, Entry, EntryId, RefMode, SessionStore, SimplifyCacheKey};

fn mode_entry_from_store_entry(
    entry: &Entry,
) -> cas_session_core::resolve::ModeEntry<SimplifyCacheKey, RequiredItem> {
    cas_session_core::resolve::ModeEntry {
        kind: entry.kind.clone(),
        requires: entry.diagnostics.requires.clone(),
        cache: entry
            .simplified
            .as_ref()
            .map(|cache| cas_session_core::resolve::ModeCacheEntry {
                key: cache.key.clone(),
                expr: cache.expr,
                requires: cache.requires.clone(),
            }),
    }
}

fn same_requirement(lhs: &RequiredItem, rhs: &RequiredItem) -> bool {
    lhs.cond == rhs.cond
}

fn mark_session_propagated(item: &mut RequiredItem) {
    item.merge_origin(RequireOrigin::SessionPropagated);
}

pub(super) fn mode_resolve_config<'a>(
    mode: RefMode,
    cache_key: &'a SimplifyCacheKey,
    env: &'a Environment,
) -> cas_session_core::resolve::ModeResolveConfig<'a, SimplifyCacheKey> {
    cas_session_core::resolve::ModeResolveConfig {
        mode,
        cache_key,
        env,
    }
}

pub(super) fn push_session_propagated_requirement(
    diagnostics: &mut Diagnostics,
    item: RequiredItem,
) {
    diagnostics.push_required(item.cond, RequireOrigin::SessionPropagated);
}

pub(super) fn with_mode_resolution_plumbing<T, F>(store: &SessionStore, run: F) -> T
where
    F: FnOnce(
        &mut dyn FnMut(
            EntryId,
        ) -> Option<
            cas_session_core::resolve::ModeEntry<SimplifyCacheKey, RequiredItem>,
        >,
        &mut dyn FnMut(&RequiredItem, &RequiredItem) -> bool,
        &mut dyn FnMut(&mut RequiredItem),
    ) -> T,
{
    let mut lookup = |id: EntryId| store.get(id).map(mode_entry_from_store_entry);
    let mut same = same_requirement;
    let mut mark = mark_session_propagated;
    run(&mut lookup, &mut same, &mut mark)
}
