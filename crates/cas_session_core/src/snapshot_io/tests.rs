use super::{load_bincode, save_bincode_atomic, tmp_path};
use serde::{Deserialize, Serialize};
use tempfile::tempdir;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct Payload {
    a: u32,
    b: String,
}

#[test]
fn roundtrip_bincode_io_is_stable() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("snapshot.bin");
    let value = Payload {
        a: 7,
        b: "ok".to_string(),
    };
    save_bincode_atomic(&value, &path).expect("save");
    let loaded: Payload = load_bincode(&path).expect("load");
    assert_eq!(loaded, value);
}

#[test]
fn tmp_path_appends_tmp_suffix() {
    let path = std::path::Path::new("/tmp/session.snap");
    let tmp = tmp_path(path);
    assert_eq!(
        tmp.file_name().and_then(|n| n.to_str()),
        Some("session.snap.tmp")
    );
}
