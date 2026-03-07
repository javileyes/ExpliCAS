use cas_session_core::types::EntryId;

/// Raw history entry payload needed for `show`/inspection flows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HistoryInspectEntryRaw {
    pub id: EntryId,
    pub type_str: String,
    pub raw_text: String,
    pub kind: crate::HistoryEntryKindRaw,
}
