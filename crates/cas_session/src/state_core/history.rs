use crate::state_core::SessionState;

impl SessionState {
    pub fn history_entries(&self) -> &[crate::Entry] {
        self.store.list()
    }

    pub fn history_push<S: Into<String>>(
        &mut self,
        kind: cas_session_core::types::EntryKind,
        raw_text: S,
    ) -> cas_session_core::types::EntryId {
        self.store.push(kind, raw_text.into())
    }

    pub fn history_get(&self, id: cas_session_core::types::EntryId) -> Option<&crate::Entry> {
        self.store.get(id)
    }

    pub fn history_len(&self) -> usize {
        self.store.len()
    }

    pub fn history_remove(&mut self, ids: &[cas_session_core::types::EntryId]) {
        self.store.remove(ids);
    }
}

impl cas_solver_core::history_runtime::HistoryDeleteContext for SessionState {
    fn history_len(&self) -> usize {
        SessionState::history_len(self)
    }

    fn history_remove(&mut self, ids: &[u64]) {
        SessionState::history_remove(self, ids);
    }
}

impl cas_solver_core::history_runtime::HistoryOverviewContext for SessionState {
    fn history_entries_raw(&self) -> Vec<cas_solver_core::history_runtime::HistoryEntryRaw> {
        self.history_entries()
            .iter()
            .map(|entry| {
                let kind = match entry.kind {
                    cas_session_core::types::EntryKind::Expr(expr) => {
                        cas_solver_core::history_runtime::HistoryEntryKindRaw::Expr(expr)
                    }
                    cas_session_core::types::EntryKind::Eq { lhs, rhs } => {
                        cas_solver_core::history_runtime::HistoryEntryKindRaw::Eq { lhs, rhs }
                    }
                };
                cas_solver_core::history_runtime::HistoryEntryRaw { id: entry.id, kind }
            })
            .collect()
    }
}

impl cas_solver::session_api::history::InspectHistoryContext for SessionState {
    fn history_entry_raw(
        &self,
        id: u64,
    ) -> Option<cas_solver_core::history_runtime::HistoryInspectEntryRaw> {
        self.history_get(id).map(|entry| {
            let kind = match entry.kind {
                cas_session_core::types::EntryKind::Expr(expr) => {
                    cas_solver_core::history_runtime::HistoryEntryKindRaw::Expr(expr)
                }
                cas_session_core::types::EntryKind::Eq { lhs, rhs } => {
                    cas_solver_core::history_runtime::HistoryEntryKindRaw::Eq { lhs, rhs }
                }
            };

            cas_solver_core::history_runtime::HistoryInspectEntryRaw {
                id: entry.id,
                type_str: entry.type_str().to_string(),
                raw_text: entry.raw_text.clone(),
                kind,
            }
        })
    }

    fn resolve_state_refs_for_inspect(
        &self,
        ctx: &mut cas_ast::Context,
        expr: cas_ast::ExprId,
    ) -> Result<cas_ast::ExprId, String> {
        self.resolve_state_refs(ctx, expr)
            .map_err(|error| error.to_string())
    }

    fn eval_for_inspect(
        &mut self,
        engine: &mut cas_solver::runtime::Engine,
        request: cas_solver::runtime::EvalRequest,
    ) -> Result<cas_solver::runtime::EvalOutput, String> {
        engine
            .eval(self, request)
            .map_err(|error| error.to_string())
    }
}
