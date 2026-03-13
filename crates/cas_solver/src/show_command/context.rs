use crate::session_api::history::{
    inspect_history_entry_input, HistoryEntryInspection, InspectHistoryContext,
};
use crate::Engine;

/// Context required to evaluate `show` command from a stateful layer.
pub trait ShowCommandContext {
    fn inspect_history_entry_input(
        &mut self,
        engine: &mut Engine,
        input: &str,
    ) -> Result<HistoryEntryInspection, crate::InspectHistoryEntryInputError>;
}

impl<T> ShowCommandContext for T
where
    T: InspectHistoryContext,
{
    fn inspect_history_entry_input(
        &mut self,
        engine: &mut Engine,
        input: &str,
    ) -> Result<HistoryEntryInspection, crate::InspectHistoryEntryInputError> {
        inspect_history_entry_input(self, engine, input)
    }
}
