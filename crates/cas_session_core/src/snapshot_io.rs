//! Generic bincode snapshot I/O helpers.

use serde::de::DeserializeOwned;
use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};

use crate::snapshot_error::SnapshotError;

/// Load one bincode payload from disk.
pub fn load_bincode<T: DeserializeOwned>(path: &Path) -> Result<T, SnapshotError> {
    let bytes = fs::read(path)?;
    Ok(bincode::deserialize(&bytes)?)
}

/// Save one bincode payload atomically: write temp file then rename.
pub fn save_bincode_atomic<T: Serialize>(value: &T, path: &Path) -> Result<(), SnapshotError> {
    let tmp = tmp_path(path);
    let bytes = bincode::serialize(value)?;
    fs::write(&tmp, bytes)?;
    fs::rename(&tmp, path)?;
    Ok(())
}

/// Compute sidecar temp path used by [`save_bincode_atomic`].
pub fn tmp_path(path: &Path) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(".tmp");
    path.with_file_name(name)
}

#[cfg(test)]
mod tests {
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
}
