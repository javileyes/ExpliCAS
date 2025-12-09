//! Display hints for preserving original expression forms
//!
//! This module provides types for hinting how expressions should be displayed.
//! For example, when sqrt(x) is internally canonicalized to x^(1/2), a hint
//! can be stored to still render it as √x.

use crate::ExprId;
use std::collections::HashMap;

/// Hint for how an expression should be displayed
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DisplayHint {
    /// Display x^(1/n) as nth root: √x or ∛x etc.
    AsRoot { index: u32 },
    // Future extensions:
    // AsFraction,     // x^(-1) as 1/x
    // AsNegation,     // (-1)*x as -x
}

/// Context that maps expressions to their preferred display form
#[derive(Debug, Default, Clone)]
pub struct DisplayContext {
    hints: HashMap<ExprId, DisplayHint>,
}

impl DisplayContext {
    /// Create an empty DisplayContext
    pub fn new() -> Self {
        Self {
            hints: HashMap::new(),
        }
    }

    /// Insert a display hint for an expression
    pub fn insert(&mut self, id: ExprId, hint: DisplayHint) {
        self.hints.insert(id, hint);
    }

    /// Get the display hint for an expression, if any
    pub fn get(&self, id: ExprId) -> Option<&DisplayHint> {
        self.hints.get(&id)
    }

    /// Check if there are any hints
    pub fn is_empty(&self) -> bool {
        self.hints.is_empty()
    }

    /// Number of hints stored
    pub fn len(&self) -> usize {
        self.hints.len()
    }
}
