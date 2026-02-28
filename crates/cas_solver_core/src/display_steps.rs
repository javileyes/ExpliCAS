//! Generic display-step container used by solver frontends.
//!
//! This wrapper enforces that renderers consume already post-processed steps.

/// Display-ready steps wrapper.
#[derive(Debug, Clone)]
pub struct DisplaySteps<S>(pub Vec<S>);

impl<S> DisplaySteps<S> {
    /// Check if there are no steps.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Get the number of steps.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Iterate over steps.
    pub fn iter(&self) -> std::slice::Iter<'_, S> {
        self.0.iter()
    }

    /// Get inner slice reference.
    pub fn as_slice(&self) -> &[S] {
        &self.0
    }

    /// Consume and return inner vector.
    pub fn into_inner(self) -> Vec<S> {
        self.0
    }
}

impl<S> Default for DisplaySteps<S> {
    fn default() -> Self {
        Self(Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use super::DisplaySteps;

    #[test]
    fn wrapper_exposes_basic_vec_api() {
        let steps = DisplaySteps(vec![1u8, 2, 3]);
        assert!(!steps.is_empty());
        assert_eq!(steps.len(), 3);
        assert_eq!(steps.as_slice(), &[1, 2, 3]);
        assert_eq!(steps.iter().copied().collect::<Vec<_>>(), vec![1, 2, 3]);
        assert_eq!(steps.into_inner(), vec![1, 2, 3]);
    }
}
