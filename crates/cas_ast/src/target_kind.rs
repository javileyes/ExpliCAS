//! Compiler-checked rule dispatch types.
//!
//! [`TargetKind`] replaces the stringly-typed `"Add"` / `"Mul"` / … dispatch
//! that previously powered rule indexing.  A typo in a variant name is now a
//! compile error, and `TargetKindSet` eliminates the per-rule `Vec<&str>`
//! allocation.

use crate::Expr;
use std::fmt;

// =============================================================================
// TargetKind enum
// =============================================================================

/// One-to-one mapping to `Expr` discriminants, used for rule dispatch.
///
/// Adding a variant to `Expr` without updating this enum and
/// [`TargetKind::from_expr`] will cause a compile error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum TargetKind {
    Add = 0,
    Sub = 1,
    Mul = 2,
    Div = 3,
    Pow = 4,
    Neg = 5,
    Function = 6,
    Number = 7,
    Variable = 8,
    Constant = 9,
    Matrix = 10,
    SessionRef = 11,
    Hold = 12,
}

impl TargetKind {
    /// Total number of variants (used for iteration bounds).
    pub const COUNT: usize = 13;

    /// Convert an `Expr` reference to its `TargetKind`.
    ///
    /// This is exhaustive — the compiler will reject a missing arm when
    /// `Expr` gains a new variant.
    #[inline]
    pub fn from_expr(expr: &Expr) -> Self {
        match expr {
            Expr::Add(..) => Self::Add,
            Expr::Sub(..) => Self::Sub,
            Expr::Mul(..) => Self::Mul,
            Expr::Div(..) => Self::Div,
            Expr::Pow(..) => Self::Pow,
            Expr::Neg(..) => Self::Neg,
            Expr::Function(..) => Self::Function,
            Expr::Number(..) => Self::Number,
            Expr::Variable(..) => Self::Variable,
            Expr::Constant(..) => Self::Constant,
            Expr::Matrix { .. } => Self::Matrix,
            Expr::SessionRef(..) => Self::SessionRef,
            Expr::Hold(..) => Self::Hold,
        }
    }
}

impl fmt::Display for TargetKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Add => "Add",
            Self::Sub => "Sub",
            Self::Mul => "Mul",
            Self::Div => "Div",
            Self::Pow => "Pow",
            Self::Neg => "Neg",
            Self::Function => "Function",
            Self::Number => "Number",
            Self::Variable => "Variable",
            Self::Constant => "Constant",
            Self::Matrix => "Matrix",
            Self::SessionRef => "SessionRef",
            Self::Hold => "Hold",
        })
    }
}

// =============================================================================
// TargetKindSet — bitflag set
// =============================================================================

/// A compact set of [`TargetKind`] values stored as a `u16` bitmask.
///
/// # Usage
///
/// ```ignore
/// use crate::target_kind::{TargetKind, TargetKindSet};
///
/// let set = TargetKindSet::of(TargetKind::Add) | TargetKindSet::of(TargetKind::Sub);
/// assert!(set.contains(TargetKind::Add));
/// assert!(!set.contains(TargetKind::Mul));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TargetKindSet(u16);

impl TargetKindSet {
    /// The empty set (no targets).
    pub const EMPTY: Self = Self(0);

    /// Create a singleton set containing exactly one `TargetKind`.
    #[inline]
    pub const fn of(kind: TargetKind) -> Self {
        Self(1u16 << kind as u8)
    }

    /// Test whether this set contains `kind`.
    #[inline]
    pub const fn contains(self, kind: TargetKind) -> bool {
        self.0 & (1u16 << kind as u8) != 0
    }

    /// Union of two sets.
    #[inline]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Iterate over all `TargetKind` values in this set (low to high).
    pub fn iter(self) -> impl Iterator<Item = TargetKind> {
        const ALL: [TargetKind; TargetKind::COUNT] = [
            TargetKind::Add,
            TargetKind::Sub,
            TargetKind::Mul,
            TargetKind::Div,
            TargetKind::Pow,
            TargetKind::Neg,
            TargetKind::Function,
            TargetKind::Number,
            TargetKind::Variable,
            TargetKind::Constant,
            TargetKind::Matrix,
            TargetKind::SessionRef,
            TargetKind::Hold,
        ];
        let bits = self.0;
        ALL.into_iter()
            .filter(move |k| bits & (1u16 << *k as u8) != 0)
    }
}

impl std::ops::BitOr for TargetKindSet {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        self.union(rhs)
    }
}

impl fmt::Display for TargetKindSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let names: Vec<_> = self.iter().map(|k| k.to_string()).collect();
        write!(f, "{{{}}}", names.join(", "))
    }
}

// =============================================================================
// Convenience constants — one per variant
// =============================================================================

impl TargetKindSet {
    pub const ADD: Self = Self::of(TargetKind::Add);
    pub const SUB: Self = Self::of(TargetKind::Sub);
    pub const MUL: Self = Self::of(TargetKind::Mul);
    pub const DIV: Self = Self::of(TargetKind::Div);
    pub const POW: Self = Self::of(TargetKind::Pow);
    pub const NEG: Self = Self::of(TargetKind::Neg);
    pub const FUNCTION: Self = Self::of(TargetKind::Function);
    pub const NUMBER: Self = Self::of(TargetKind::Number);
    pub const VARIABLE: Self = Self::of(TargetKind::Variable);
    pub const CONSTANT: Self = Self::of(TargetKind::Constant);
    pub const MATRIX: Self = Self::of(TargetKind::Matrix);
    pub const SESSION_REF: Self = Self::of(TargetKind::SessionRef);
    pub const HOLD: Self = Self::of(TargetKind::Hold);

    /// Common multi-kind set: Add + Sub
    pub const ADD_SUB: Self = Self(Self::ADD.0 | Self::SUB.0);
}
