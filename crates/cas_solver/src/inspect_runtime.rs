mod input;
mod inspect;
mod raw;

use cas_ast::ExprId;
use cas_session_core::types::EntryId;

use crate::{Engine, EvalOutput, EvalRequest};

pub use input::inspect_history_entry_input;
pub use inspect::inspect_history_entry;
pub use raw::HistoryInspectEntryRaw;

/// Context needed to inspect a history entry with optional eval metadata.
pub trait InspectHistoryContext {
    fn history_entry_raw(&self, id: EntryId) -> Option<HistoryInspectEntryRaw>;

    fn resolve_state_refs_for_inspect(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
    ) -> Result<ExprId, String>;

    fn eval_for_inspect(
        &mut self,
        engine: &mut Engine,
        request: EvalRequest,
    ) -> Result<EvalOutput, String>;
}
