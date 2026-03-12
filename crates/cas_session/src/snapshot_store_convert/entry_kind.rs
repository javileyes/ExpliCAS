use cas_session_core::store_snapshot::EntryKindSnapshot;
use cas_session_core::types::EntryKind;

pub(crate) fn snapshot_entry_kind(kind: &EntryKind) -> EntryKindSnapshot {
    match kind {
        EntryKind::Expr(id) => EntryKindSnapshot::Expr(id.index() as u32),
        EntryKind::Eq { lhs, rhs } => EntryKindSnapshot::Eq {
            lhs: lhs.index() as u32,
            rhs: rhs.index() as u32,
        },
    }
}

pub(crate) fn restore_entry_kind(kind: EntryKindSnapshot) -> EntryKind {
    use cas_ast::ExprId;

    match kind {
        EntryKindSnapshot::Expr(id) => EntryKind::Expr(ExprId::from_raw(id)),
        EntryKindSnapshot::Eq { lhs, rhs } => EntryKind::Eq {
            lhs: ExprId::from_raw(lhs),
            rhs: ExprId::from_raw(rhs),
        },
    }
}
