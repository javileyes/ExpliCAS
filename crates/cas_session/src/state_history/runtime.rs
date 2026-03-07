use crate::state_core::SessionState;

impl cas_solver::HistoryDeleteContext for SessionState {
    fn history_len(&self) -> usize {
        SessionState::history_len(self)
    }

    fn history_remove(&mut self, ids: &[u64]) {
        SessionState::history_remove(self, ids);
    }
}

impl cas_solver::HistoryOverviewContext for SessionState {
    fn history_entries_raw(&self) -> Vec<cas_solver::HistoryEntryRaw> {
        self.history_entries()
            .iter()
            .map(|entry| {
                let kind = match entry.kind {
                    crate::EntryKind::Expr(expr) => cas_solver::HistoryEntryKindRaw::Expr(expr),
                    crate::EntryKind::Eq { lhs, rhs } => {
                        cas_solver::HistoryEntryKindRaw::Eq { lhs, rhs }
                    }
                };
                cas_solver::HistoryEntryRaw { id: entry.id, kind }
            })
            .collect()
    }
}

impl cas_solver::InspectHistoryContext for SessionState {
    fn history_entry_raw(&self, id: u64) -> Option<cas_solver::HistoryInspectEntryRaw> {
        self.history_get(id).map(|entry| {
            let kind = match entry.kind {
                crate::EntryKind::Expr(expr) => cas_solver::HistoryEntryKindRaw::Expr(expr),
                crate::EntryKind::Eq { lhs, rhs } => {
                    cas_solver::HistoryEntryKindRaw::Eq { lhs, rhs }
                }
            };

            cas_solver::HistoryInspectEntryRaw {
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
        engine: &mut cas_solver::Engine,
        request: cas_solver::EvalRequest,
    ) -> Result<cas_solver::EvalOutput, String> {
        engine
            .eval(self, request)
            .map_err(|error| error.to_string())
    }
}
