use cas_ast::{Expr, ExprId};
use cas_engine::eval::{CacheHitEntryId, EvalSession, EvalStore, StoredInputKind};

#[derive(Default)]
pub(crate) struct JsonStore {
    next_id: u64,
}

impl JsonStore {
    fn new() -> Self {
        Self::default()
    }
}

impl EvalStore for JsonStore {
    fn push_raw_input(&mut self, _kind: StoredInputKind, _raw_input: String) -> u64 {
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

pub(crate) struct JsonEvalSession {
    store: JsonStore,
    options: cas_engine::options::EvalOptions,
    profile_cache: cas_engine::profile_cache::ProfileCache,
}

impl JsonEvalSession {
    pub(crate) fn new(options: cas_engine::options::EvalOptions) -> Self {
        Self {
            store: JsonStore::new(),
            options,
            profile_cache: cas_engine::profile_cache::ProfileCache::new(),
        }
    }

    pub(crate) fn options_mut(&mut self) -> &mut cas_engine::options::EvalOptions {
        &mut self.options
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

    fn resolve_all_with_diagnostics(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
    ) -> Result<
        (
            ExprId,
            cas_engine::diagnostics::Diagnostics,
            Vec<CacheHitEntryId>,
        ),
        cas_engine::eval::EvalResolveError,
    > {
        if let Some(id) = first_session_ref(ctx, expr) {
            return Err(cas_engine::eval::EvalResolveError::NotFound(id));
        }
        Ok((expr, cas_engine::diagnostics::Diagnostics::new(), vec![]))
    }
}
