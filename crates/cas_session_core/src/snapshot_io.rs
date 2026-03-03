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
mod tests;
