//! Display formatting for expressions.
//!
//! This module provides display implementations for expressions:
//! - [`DisplayExpr`]: Basic expression display (ASCII and Unicode pretty modes)
//! - [`DisplayExprWithHints`]: Display with rendering hints (roots, fractions, etc.)
//! - [`RawDisplayExpr`]: Debug-style raw AST display
//!
//! # Architecture: `display_context` Split
//!
//! Display context is split across two crates by design:
//!
//! - **`cas_ast::display_context`** — owns the **types** ([`DisplayHint`],
//!   [`DisplayContext`]) because they are pure data structures with no engine
//!   dependency. Any crate that needs to *read* display hints can depend on
//!   `cas_ast` alone.
//!
//! - **`cas_engine::display_context`** — owns the **builder**
//!   (`build_display_context`) because constructing hints requires access to
//!   [`Step`](crate) types defined in `cas_engine`. This function scans
//!   simplification steps for sqrt/root patterns and propagates `AsRoot` hints
//!   to the corresponding `Pow` expressions.
//!
//! This split keeps `cas_ast` dependency-free while allowing `cas_engine` to
//! perform the analysis that requires step-level context.

mod expr;
mod hints;
mod ordering;
mod styled;

// Re-export public items from submodules
pub use expr::{DisplayExpr, RawDisplayExpr};
pub use hints::DisplayExprWithHints;
pub use ordering::{cmp_term_for_display, DisplayFactor, FractionDisplayView, OrderingMode};
pub use styled::DisplayExprStyled;

use std::sync::atomic::{AtomicBool, Ordering};

// ============================================================================
// Pretty Output Configuration
// ============================================================================

/// Global flag controlling pretty Unicode output (∛, x², ·) vs ASCII (sqrt, ^, *)
/// Default: false (ASCII mode for tests compatibility)
/// CLI enables pretty mode for end users
static PRETTY_OUTPUT: AtomicBool = AtomicBool::new(false);

/// Enable pretty Unicode output (∛, x², ·)
pub fn enable_pretty_output() {
    PRETTY_OUTPUT.store(true, Ordering::SeqCst);
}

/// Disable pretty output, use ASCII (sqrt, ^, *)
pub fn disable_pretty_output() {
    PRETTY_OUTPUT.store(false, Ordering::SeqCst);
}

/// Check if pretty output is enabled
pub fn is_pretty_output() -> bool {
    PRETTY_OUTPUT.load(Ordering::SeqCst)
}

/// Get the multiplication symbol based on pretty mode
pub fn mul_symbol() -> &'static str {
    if PRETTY_OUTPUT.load(Ordering::SeqCst) {
        "·"
    } else {
        " * "
    }
}

// ============================================================================
// Unicode Pretty Output Helpers
// ============================================================================

/// Convert a digit (0-9) to its Unicode superscript equivalent
fn digit_to_superscript(d: u32) -> char {
    match d {
        0 => '⁰',
        1 => '¹',
        2 => '²',
        3 => '³',
        4 => '⁴',
        5 => '⁵',
        6 => '⁶',
        7 => '⁷',
        8 => '⁸',
        9 => '⁹',
        _ => '?',
    }
}

/// Convert an integer to a Unicode superscript string
/// Examples: 2 → "²", 12 → "¹²", 100 → "¹⁰⁰"
pub fn number_to_superscript(n: u64) -> String {
    if n == 0 {
        return "⁰".to_string();
    }

    let mut result = String::new();
    let mut num = n;
    let mut digits = Vec::new();

    while num > 0 {
        digits.push((num % 10) as u32);
        num /= 10;
    }

    for d in digits.into_iter().rev() {
        result.push(digit_to_superscript(d));
    }

    result
}

/// Get the root prefix for a given index
/// Pretty mode: 2 → "√", 3 → "∛", 4 → "∜", 5 → "⁵√"
/// ASCII mode: 2 → "sqrt", 3 → "cbrt", n → "root(,n)"
pub fn unicode_root_prefix(index: u64) -> String {
    if !is_pretty_output() {
        // ASCII mode - will be handled by the caller using sqrt() format
        return match index {
            2 => "sqrt".to_string(),
            n => format!("{}√", n), // Fallback, normally caller handles ASCII
        };
    }

    match index {
        2 => "√".to_string(),
        3 => "∛".to_string(),
        4 => "∜".to_string(),
        n => format!("{}√", number_to_superscript(n)),
    }
}
