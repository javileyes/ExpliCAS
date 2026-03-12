//! Generic bincode snapshot I/O helpers.

use serde::de::DeserializeOwned;
use serde::Serialize;
use std::fs;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use crate::snapshot_error::SnapshotError;

pub const SNAPSHOT_IO_BUFFER_CAPACITY: usize = 64 * 1024;

/// Open a buffered reader configured for snapshot I/O.
pub fn open_bincode_reader(path: &Path) -> Result<BufReader<fs::File>, SnapshotError> {
    let file = fs::File::open(path)?;
    Ok(BufReader::with_capacity(SNAPSHOT_IO_BUFFER_CAPACITY, file))
}

/// Deserialize one bincode payload from an existing reader.
pub fn load_bincode_from_reader<R: Read, T: DeserializeOwned>(
    reader: &mut R,
) -> Result<T, SnapshotError> {
    Ok(bincode::deserialize_from(reader)?)
}

/// Load one bincode payload from disk.
pub fn load_bincode<T: DeserializeOwned>(path: &Path) -> Result<T, SnapshotError> {
    let mut reader = open_bincode_reader(path)?;
    load_bincode_from_reader(&mut reader)
}

/// Save one bincode payload atomically: write temp file then rename.
pub fn save_bincode_atomic<T: Serialize>(value: &T, path: &Path) -> Result<(), SnapshotError> {
    let tmp = tmp_path(path);
    let file = fs::File::create(&tmp)?;
    let mut writer = BufWriter::with_capacity(SNAPSHOT_IO_BUFFER_CAPACITY, file);
    bincode::serialize_into(&mut writer, value)?;
    writer.flush()?;
    drop(writer);
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
