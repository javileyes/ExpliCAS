//! Assumption collection and reporting infrastructure.
//!
//! This module provides dedup-aware assumption tracking for domain assumptions
//! made during simplification (e.g., "x ≠ 0" when cancelling x/x).
//!
//! # Architecture
//!
//! - `AssumptionKey`: Hashable key for deduplication (kind + fingerprint)
//! - `AssumptionEvent`: What a Rewrite emits when making an assumption
//! - `AssumptionRecord`: What gets exported to JSON (with count)
//! - `AssumptionCollector`: Collects events, deduplicates, produces records
//! - `AssumptionReporting`: Controls visibility (Off/Summary/Trace)
//!
//! # Anti-Cascade Contract
//!
//! Multiple uses of the same assumption (e.g., x/x + x/x + x/x) produce
//! a single AssumptionRecord with count=3, not 3 separate warnings.

use cas_ast::{Context, ExprId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

// =============================================================================
// AssumptionReporting - Visibility Control
// =============================================================================

/// Controls how assumptions are reported to the user.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AssumptionReporting {
    /// No assumptions shown (hard off - not in JSON)
    #[default]
    Off,
    /// Deduped summary list at end
    Summary,
    /// Future: Include step locations and trace info
    Trace,
}

impl AssumptionReporting {
    /// Parse from string (for REPL commands)
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "off" => Some(Self::Off),
            "summary" => Some(Self::Summary),
            "trace" => Some(Self::Trace),
            _ => None,
        }
    }
}

impl std::fmt::Display for AssumptionReporting {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Off => write!(f, "off"),
            Self::Summary => write!(f, "summary"),
            Self::Trace => write!(f, "trace"),
        }
    }
}

// =============================================================================
// AssumptionKey - Hashable Dedup Key
// =============================================================================

/// Hashable key for assumption deduplication.
///
/// Uses expression fingerprint (hash of canonical display) to identify
/// the same assumption about the same expression across multiple rule applications.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AssumptionKey {
    /// Assumed expression is non-zero (e.g., for x/x → 1)
    NonZero { expr_fingerprint: u64 },
    /// Assumed expression is positive (e.g., for √x² → x)
    Positive { expr_fingerprint: u64 },
    /// Assumed expression is defined (e.g., for a-a → 0 when a has division)
    Defined { expr_fingerprint: u64 },
    /// Assumed argument is in principal range for inverse trig composition
    InvTrigPrincipalRange {
        func: &'static str,
        arg_fingerprint: u64,
    },
    /// Assumed principal branch for complex multi-valued functions
    ComplexPrincipalBranch {
        func: &'static str,
        arg_fingerprint: u64,
    },
}

impl AssumptionKey {
    /// Get the kind as a string for JSON output
    pub fn kind(&self) -> &'static str {
        match self {
            Self::NonZero { .. } => "nonzero",
            Self::Positive { .. } => "positive",
            Self::Defined { .. } => "defined",
            Self::InvTrigPrincipalRange { .. } => "principal_range",
            Self::ComplexPrincipalBranch { .. } => "principal_branch",
        }
    }
}

// =============================================================================
// Fingerprint Helper
// =============================================================================

/// Compute a stable fingerprint for an expression based on its canonical display.
///
/// This is used to deduplicate assumptions about the same expression.
pub fn expr_fingerprint(ctx: &Context, expr: ExprId) -> u64 {
    use std::collections::hash_map::DefaultHasher;

    let display = format!(
        "{}",
        cas_ast::display::DisplayExpr {
            context: ctx,
            id: expr
        }
    );
    let mut hasher = DefaultHasher::new();
    display.hash(&mut hasher);
    hasher.finish()
}

// =============================================================================
// AssumptionEvent - What Rewrite Emits
// =============================================================================

/// An assumption event emitted by a Rewrite.
///
/// This is what rules produce; the collector aggregates these into records.
#[derive(Debug, Clone)]
pub struct AssumptionEvent {
    /// Hashable key for deduplication
    pub key: AssumptionKey,
    /// Canonical display of the expression (for output)
    pub expr_display: String,
    /// Human-readable message (not used for dedup)
    pub message: String,
}

impl AssumptionEvent {
    /// Create a NonZero assumption event
    pub fn nonzero(ctx: &Context, expr: ExprId) -> Self {
        let fp = expr_fingerprint(ctx, expr);
        let display = format!(
            "{}",
            cas_ast::display::DisplayExpr {
                context: ctx,
                id: expr
            }
        );
        Self {
            key: AssumptionKey::NonZero {
                expr_fingerprint: fp,
            },
            expr_display: display.clone(),
            message: format!("Assumed {} ≠ 0", display),
        }
    }

    /// Create a Positive assumption event
    pub fn positive(ctx: &Context, expr: ExprId) -> Self {
        let fp = expr_fingerprint(ctx, expr);
        let display = format!(
            "{}",
            cas_ast::display::DisplayExpr {
                context: ctx,
                id: expr
            }
        );
        Self {
            key: AssumptionKey::Positive {
                expr_fingerprint: fp,
            },
            expr_display: display.clone(),
            message: format!("Assumed {} > 0", display),
        }
    }

    /// Create a Defined assumption event (for expressions with potential undefined points)
    pub fn defined(ctx: &Context, expr: ExprId) -> Self {
        let fp = expr_fingerprint(ctx, expr);
        let display = format!(
            "{}",
            cas_ast::display::DisplayExpr {
                context: ctx,
                id: expr
            }
        );
        Self {
            key: AssumptionKey::Defined {
                expr_fingerprint: fp,
            },
            expr_display: display.clone(),
            message: format!("Assumed {} is defined", display),
        }
    }

    /// Create an InvTrig principal range assumption
    pub fn inv_trig_principal_range(ctx: &Context, func: &'static str, arg: ExprId) -> Self {
        let fp = expr_fingerprint(ctx, arg);
        let display = format!(
            "{}",
            cas_ast::display::DisplayExpr {
                context: ctx,
                id: arg
            }
        );
        Self {
            key: AssumptionKey::InvTrigPrincipalRange {
                func,
                arg_fingerprint: fp,
            },
            expr_display: display.clone(),
            message: format!("Assumed {} is in principal range of {}", display, func),
        }
    }

    /// Create a complex principal branch assumption
    pub fn complex_principal_branch(ctx: &Context, func: &'static str, arg: ExprId) -> Self {
        let fp = expr_fingerprint(ctx, arg);
        let display = format!(
            "{}",
            cas_ast::display::DisplayExpr {
                context: ctx,
                id: arg
            }
        );
        Self {
            key: AssumptionKey::ComplexPrincipalBranch {
                func,
                arg_fingerprint: fp,
            },
            expr_display: display.clone(),
            message: format!("Used principal branch of {}({})", func, display),
        }
    }

    /// Create an assumption from a legacy domain_assumption string.
    ///
    /// Parses strings like:
    /// - "Assuming denominator ≠ 0" → NonZero with "denominator" as display
    /// - "Assuming expression is defined" → Defined with "expression" as display
    /// - Other strings → Defined with truncated message as display
    pub fn from_legacy_string(message: &str) -> Self {
        use std::collections::hash_map::DefaultHasher;

        // Parse known patterns
        let (kind, expr_display) = if message.contains("≠ 0") || message.contains("!= 0") {
            // NonZero assumption
            let expr = if message.contains("denominator") {
                "denominator"
            } else if message.contains("x") {
                "x"
            } else {
                "expr"
            };
            ("nonzero", expr.to_string())
        } else if message.contains("defined") {
            // Defined assumption
            ("defined", "expression".to_string())
        } else if message.contains("> 0") || message.contains("positive") {
            // Positive assumption
            ("positive", "expr".to_string())
        } else if message.contains("principal") || message.contains("range") {
            // Principal range assumption
            ("principal_range", "arg".to_string())
        } else {
            // Unknown - treat as Defined
            ("defined", "expression".to_string())
        };

        // Create fingerprint from the message itself (since we don't have expression)
        let mut hasher = DefaultHasher::new();
        message.hash(&mut hasher);
        let fp = hasher.finish();

        let key = match kind {
            "nonzero" => AssumptionKey::NonZero {
                expr_fingerprint: fp,
            },
            "positive" => AssumptionKey::Positive {
                expr_fingerprint: fp,
            },
            "principal_range" => AssumptionKey::InvTrigPrincipalRange {
                func: "trig",
                arg_fingerprint: fp,
            },
            _ => AssumptionKey::Defined {
                expr_fingerprint: fp,
            },
        };

        Self {
            key,
            expr_display,
            message: message.to_string(),
        }
    }
}

// =============================================================================
// AssumptionRecord - JSON Output
// =============================================================================

/// An assumption record for JSON output.
///
/// This is the deduplicated, counted result of collecting AssumptionEvents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssumptionRecord {
    /// Type of assumption (nonzero, positive, defined, etc.)
    pub kind: String,
    /// Canonical display of the expression
    pub expr: String,
    /// Human-readable message
    pub message: String,
    /// How many times this assumption was used
    pub count: u32,
}

// =============================================================================
// AssumptionCollector - Dedup Aggregator
// =============================================================================

/// Collects assumption events with deduplication.
///
/// Multiple events with the same key are counted, not repeated.
#[derive(Debug, Clone, Default)]
pub struct AssumptionCollector {
    map: HashMap<AssumptionKey, (AssumptionEvent, u32)>,
}

impl AssumptionCollector {
    /// Create a new empty collector
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    /// Note an assumption event (deduplicates by key)
    pub fn note(&mut self, event: AssumptionEvent) {
        let key = event.key.clone();
        self.map
            .entry(key)
            .and_modify(|(_, count)| *count += 1)
            .or_insert((event, 1));
    }

    /// Check if any assumptions were collected
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Get the count of unique assumptions
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Finish collection and produce sorted records
    ///
    /// Order is stable: by kind, then by expr_display
    pub fn finish(self) -> Vec<AssumptionRecord> {
        let mut records: Vec<AssumptionRecord> = self
            .map
            .into_iter()
            .map(|(_, (event, count))| AssumptionRecord {
                kind: event.key.kind().to_string(),
                expr: event.expr_display,
                message: event.message,
                count,
            })
            .collect();

        // Stable order: by kind, then by expr
        records.sort_by(|a, b| a.kind.cmp(&b.kind).then_with(|| a.expr.cmp(&b.expr)));

        records
    }

    /// Format assumptions as a summary line for REPL output
    pub fn summary_line(&self) -> Option<String> {
        if self.map.is_empty() {
            return None;
        }

        let mut items: Vec<String> = self
            .map
            .iter()
            .map(|(_, (event, count))| {
                if *count > 1 {
                    format!("{}({}) (×{})", event.key.kind(), event.expr_display, count)
                } else {
                    format!("{}({})", event.key.kind(), event.expr_display)
                }
            })
            .collect();

        // Stable order
        items.sort();

        Some(format!("⚠ Assumptions: {}", items.join(", ")))
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::Context;

    #[test]
    fn test_dedup_same_expr() {
        let mut ctx = Context::new();
        let x = ctx.var("x");

        let mut collector = AssumptionCollector::new();

        // Note the same assumption 3 times
        collector.note(AssumptionEvent::nonzero(&ctx, x));
        collector.note(AssumptionEvent::nonzero(&ctx, x));
        collector.note(AssumptionEvent::nonzero(&ctx, x));

        let records = collector.finish();

        assert_eq!(records.len(), 1, "Should dedup to single record");
        assert_eq!(records[0].count, 3, "Count should be 3");
        assert_eq!(records[0].kind, "nonzero");
        assert_eq!(records[0].expr, "x");
    }

    #[test]
    fn test_different_exprs_separate() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");

        let mut collector = AssumptionCollector::new();

        collector.note(AssumptionEvent::nonzero(&ctx, x));
        collector.note(AssumptionEvent::nonzero(&ctx, y));

        let records = collector.finish();

        assert_eq!(records.len(), 2, "Different exprs should be separate");
    }

    #[test]
    fn test_different_kinds_separate() {
        let mut ctx = Context::new();
        let x = ctx.var("x");

        let mut collector = AssumptionCollector::new();

        collector.note(AssumptionEvent::nonzero(&ctx, x));
        collector.note(AssumptionEvent::positive(&ctx, x));

        let records = collector.finish();

        assert_eq!(records.len(), 2, "Different kinds should be separate");
    }

    #[test]
    fn test_stable_order() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let z = ctx.var("z");

        let mut collector = AssumptionCollector::new();

        // Add in arbitrary order
        collector.note(AssumptionEvent::positive(&ctx, z));
        collector.note(AssumptionEvent::nonzero(&ctx, y));
        collector.note(AssumptionEvent::nonzero(&ctx, x));

        let records = collector.finish();

        // Should be sorted: nonzero before positive, then by expr
        assert_eq!(records[0].kind, "nonzero");
        assert_eq!(records[0].expr, "x");
        assert_eq!(records[1].kind, "nonzero");
        assert_eq!(records[1].expr, "y");
        assert_eq!(records[2].kind, "positive");
        assert_eq!(records[2].expr, "z");
    }

    #[test]
    fn test_summary_line() {
        let mut ctx = Context::new();
        let x = ctx.var("x");

        let mut collector = AssumptionCollector::new();
        collector.note(AssumptionEvent::nonzero(&ctx, x));
        collector.note(AssumptionEvent::nonzero(&ctx, x));

        let summary = collector.summary_line();

        assert!(summary.is_some());
        let line = summary.unwrap();
        assert!(line.contains("nonzero(x)"));
        assert!(line.contains("×2"));
    }

    #[test]
    fn test_empty_collector() {
        let collector = AssumptionCollector::new();

        assert!(collector.is_empty());
        assert_eq!(collector.len(), 0);
        assert!(collector.summary_line().is_none());

        let records = collector.finish();
        assert!(records.is_empty());
    }

    #[test]
    fn test_reporting_from_str() {
        assert_eq!(
            AssumptionReporting::from_str("off"),
            Some(AssumptionReporting::Off)
        );
        assert_eq!(
            AssumptionReporting::from_str("summary"),
            Some(AssumptionReporting::Summary)
        );
        assert_eq!(
            AssumptionReporting::from_str("trace"),
            Some(AssumptionReporting::Trace)
        );
        assert_eq!(AssumptionReporting::from_str("invalid"), None);
    }
}
