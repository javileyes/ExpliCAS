//! Generic snapshot header primitives shared by session runtimes.

use serde::{Deserialize, Serialize};

/// Header used to validate snapshot format identity and cache compatibility.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SnapshotHeader<CacheKey> {
    /// Magic bytes for file identification.
    pub magic: [u8; 8],
    /// Format version (increment on breaking changes).
    pub version: u32,
    /// Cache key for invalidation (runtime-defined payload).
    pub cache_key: CacheKey,
}

impl<CacheKey> SnapshotHeader<CacheKey> {
    /// Build one header from identity tuple and cache key.
    pub fn new(magic: [u8; 8], version: u32, cache_key: CacheKey) -> Self {
        Self {
            magic,
            version,
            cache_key,
        }
    }

    /// Check identity tuple against expected magic + version.
    pub fn is_valid_with(&self, magic: [u8; 8], version: u32) -> bool {
        self.magic == magic && self.version == version
    }
}

#[cfg(test)]
mod tests;
