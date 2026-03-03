use super::SnapshotHeader;

#[test]
fn snapshot_header_identity_check_works() {
    let header = SnapshotHeader::new(*b"EXPLICAS", 1, 7u32);
    assert!(header.is_valid_with(*b"EXPLICAS", 1));
    assert!(!header.is_valid_with(*b"EXPLICA1", 1));
    assert!(!header.is_valid_with(*b"EXPLICAS", 2));
}
