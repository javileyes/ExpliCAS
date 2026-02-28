//! Shared snapshot error type for session persistence runtimes.

use std::io;

#[derive(Debug)]
pub enum SnapshotError {
    Io(io::Error),
    Bincode(Box<bincode::ErrorKind>),
}

impl std::fmt::Display for SnapshotError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SnapshotError::Io(e) => write!(f, "IO error: {}", e),
            SnapshotError::Bincode(e) => write!(f, "Serialization error: {}", e),
        }
    }
}

impl std::error::Error for SnapshotError {}

impl From<io::Error> for SnapshotError {
    fn from(e: io::Error) -> Self {
        SnapshotError::Io(e)
    }
}

impl From<Box<bincode::ErrorKind>> for SnapshotError {
    fn from(e: Box<bincode::ErrorKind>) -> Self {
        SnapshotError::Bincode(e)
    }
}
