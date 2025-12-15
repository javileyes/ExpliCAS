//! Cycle detection infrastructure for detecting ping-pong patterns (A↔B).
//!
//! This module provides:
//! - `expr_fingerprint`: Structural hash of an expression tree
//! - `CycleDetector`: Ring buffer for detecting short cycles (period 1-8)
//! - `CycleInfo`: Information about a detected cycle

use cas_ast::{Context, Expr, ExprId};
use std::collections::HashMap;

/// Information about a detected cycle
#[derive(Debug, Clone)]
pub struct CycleInfo {
    /// Which phase detected the cycle
    pub phase: crate::phase::SimplifyPhase,
    /// Cycle period (1=self-loop, 2=A↔B, 3=A→B→C→A, etc.)
    pub period: usize,
    /// At which rewrite step the cycle was detected
    pub at_step: usize,
}

/// Memoization cache for fingerprints (cleared per phase)
pub type FingerprintMemo = HashMap<ExprId, u64>;

/// Ring buffer cycle detector for short cycles (period 1-8)
#[derive(Debug)]
pub struct CycleDetector {
    /// Ring buffer of recent fingerprints
    buf: [u64; 64],
    /// Current position in buffer
    pos: usize,
    /// Number of items in buffer
    len: usize,
    /// Current step count
    step: usize,
    /// Phase this detector is tracking
    phase: crate::phase::SimplifyPhase,
    /// Maximum cycle period to detect (default: 8)
    max_period: usize,
}

impl CycleDetector {
    /// Create a new cycle detector for a phase
    pub fn new(phase: crate::phase::SimplifyPhase) -> Self {
        Self {
            buf: [0u64; 64],
            pos: 0,
            len: 0,
            step: 0,
            phase,
            max_period: 8,
        }
    }

    /// Observe a new fingerprint and check for cycles.
    /// Returns Some(CycleInfo) if a cycle is detected.
    pub fn observe(&mut self, hash: u64) -> Option<CycleInfo> {
        self.step += 1;

        // Check for cycles BEFORE storing the new hash
        let result = self.check_for_cycles(hash);

        // Store in ring buffer AFTER checking
        self.buf[self.pos] = hash;
        self.pos = (self.pos + 1) % 64;
        if self.len < 64 {
            self.len += 1;
        }

        result
    }

    /// Check for cycles given the new hash (before it's stored)
    fn check_for_cycles(&self, hash: u64) -> Option<CycleInfo> {
        // Need at least 1 item to detect self-loop
        if self.len < 1 {
            return None;
        }

        // Check for self-loop (period 1): A → A
        let prev1 = self.get_back(1);
        if hash == prev1 {
            return Some(CycleInfo {
                phase: self.phase,
                period: 1,
                at_step: self.step,
            });
        }

        // Check for 2-cycle (period 2): A → B → A → B
        // Need: h[now] == h[n-2] && h[n-1] == h[n-3]
        if self.len >= 3 {
            let prev2 = self.get_back(2);
            let prev3 = self.get_back(3);
            // prev1 is B, hash is A. For 2-cycle: hash == prev2 (previous A) and prev1 == prev3
            if hash == prev2 && prev1 == prev3 {
                return Some(CycleInfo {
                    phase: self.phase,
                    period: 2,
                    at_step: self.step,
                });
            }
        }

        // Check for period 3-8 cycles
        for period in 3..=self.max_period.min(self.len) {
            if self.check_period(period, hash) {
                return Some(CycleInfo {
                    phase: self.phase,
                    period,
                    at_step: self.step,
                });
            }
        }

        None
    }

    /// Check if there's a cycle of the given period.
    /// Called BEFORE current hash is stored, so buffer has len items.
    fn check_period(&self, period: usize, current_hash: u64) -> bool {
        // Need: period positions back + (period-1) more for verification
        // With hash not yet stored, need at least 2*period - 1 items
        if self.len < 2 * period - 1 {
            return false;
        }

        // Check if current matches `period` positions back
        let back_n = self.get_back(period);
        if current_hash != back_n {
            return false;
        }

        // Verify the pattern holds for previous positions too
        // (require at least one full cycle before current)
        for i in 1..period {
            let a = self.get_back(i);
            let b = self.get_back(i + period);
            if a != b {
                return false;
            }
        }

        true
    }

    /// Get the hash at position `n` items back (1 = previous, 2 = two back, etc.)
    fn get_back(&self, n: usize) -> u64 {
        debug_assert!(n >= 1 && n <= self.len);
        let idx = (self.pos + 64 - n) % 64;
        self.buf[idx]
    }

    /// Reset the detector for a new phase
    pub fn reset(&mut self, phase: crate::phase::SimplifyPhase) {
        self.buf = [0u64; 64];
        self.pos = 0;
        self.len = 0;
        self.step = 0;
        self.phase = phase;
    }
}

// ==================== Fingerprinting ====================

/// TAG constants for mixing expression types
const TAG_NUM: u64 = 0x1234567890ABCDEF;
const TAG_VAR: u64 = 0xFEDCBA0987654321;
const TAG_CONST: u64 = 0xABCDEF1234567890;
const TAG_ADD: u64 = 0x2345678901BCDEF0;
const TAG_SUB: u64 = 0x3456789012CDEF01;
const TAG_MUL: u64 = 0x4567890123DEF012;
const TAG_DIV: u64 = 0x5678901234EF0123;
const TAG_POW: u64 = 0x6789012345F01234;
const TAG_NEG: u64 = 0x7890123456012345;
const TAG_FUNC: u64 = 0x8901234567123456;
const TAG_MATRIX: u64 = 0x9012345678234567;

/// Mix two u64 values (splitmix64-style)
#[inline]
fn mix(tag: u64, a: u64, b: u64) -> u64 {
    let mut x = tag
        .wrapping_add(a)
        .wrapping_add(b.wrapping_mul(0x9E3779B97F4A7C15));
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}

/// Mix single value with tag
#[inline]
fn mix1(tag: u64, a: u64) -> u64 {
    mix(tag, a, 0)
}

/// Hash a string (FNV-1a style)
fn hash_str(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for byte in s.bytes() {
        h ^= byte as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Compute structural fingerprint of an expression.
/// Uses memoization for efficiency on DAGs.
pub fn expr_fingerprint(ctx: &Context, root: ExprId, memo: &mut FingerprintMemo) -> u64 {
    if let Some(&h) = memo.get(&root) {
        return h;
    }

    let h = match ctx.get(root) {
        Expr::Number(q) => {
            // Hash both numer and denom
            let n = q.numer().to_string();
            let d = q.denom().to_string();
            mix(TAG_NUM, hash_str(&n), hash_str(&d))
        }
        Expr::Variable(name) => mix1(TAG_VAR, hash_str(name)),
        Expr::Constant(c) => mix1(TAG_CONST, hash_str(&format!("{:?}", c))),
        Expr::Add(a, b) => {
            let ha = expr_fingerprint(ctx, *a, memo);
            let hb = expr_fingerprint(ctx, *b, memo);
            mix(TAG_ADD, ha, hb)
        }
        Expr::Sub(a, b) => {
            let ha = expr_fingerprint(ctx, *a, memo);
            let hb = expr_fingerprint(ctx, *b, memo);
            mix(TAG_SUB, ha, hb)
        }
        Expr::Mul(a, b) => {
            let ha = expr_fingerprint(ctx, *a, memo);
            let hb = expr_fingerprint(ctx, *b, memo);
            mix(TAG_MUL, ha, hb)
        }
        Expr::Div(a, b) => {
            let ha = expr_fingerprint(ctx, *a, memo);
            let hb = expr_fingerprint(ctx, *b, memo);
            mix(TAG_DIV, ha, hb)
        }
        Expr::Pow(a, b) => {
            let ha = expr_fingerprint(ctx, *a, memo);
            let hb = expr_fingerprint(ctx, *b, memo);
            mix(TAG_POW, ha, hb)
        }
        Expr::Neg(x) => {
            let hx = expr_fingerprint(ctx, *x, memo);
            mix1(TAG_NEG, hx)
        }
        Expr::Function(name, args) => {
            let mut h = mix1(TAG_FUNC, hash_str(name));
            for arg in args {
                let ha = expr_fingerprint(ctx, *arg, memo);
                h = mix(h, ha, args.len() as u64);
            }
            h
        }
        Expr::Matrix { rows, cols, data } => {
            let mut h = mix(TAG_MATRIX, *rows as u64, *cols as u64);
            for elem in data {
                let he = expr_fingerprint(ctx, *elem, memo);
                h = mix(h, he, data.len() as u64);
            }
            h
        }
        // SessionRef is a leaf - hash the id
        Expr::SessionRef(id) => mix1(TAG_VAR, *id),
    };

    memo.insert(root, h);
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phase::SimplifyPhase;

    #[test]
    fn test_cycle_detector_self_loop() {
        let mut detector = CycleDetector::new(SimplifyPhase::Core);

        assert!(detector.observe(100).is_none());

        // A → A (self-loop)
        let info = detector.observe(100);
        assert!(info.is_some());
        let info = info.unwrap();
        assert_eq!(info.period, 1);
        assert_eq!(info.at_step, 2);
    }

    #[test]
    fn test_cycle_detector_period_2() {
        let mut detector = CycleDetector::new(SimplifyPhase::Transform);

        assert!(detector.observe(100).is_none()); // A
        assert!(detector.observe(200).is_none()); // B
        assert!(detector.observe(100).is_none()); // A (need more)

        // A → B → A → B (2-cycle)
        let info = detector.observe(200);
        assert!(info.is_some());
        let info = info.unwrap();
        assert_eq!(info.period, 2);
    }

    #[test]
    fn test_cycle_detector_period_3() {
        let mut detector = CycleDetector::new(SimplifyPhase::Core);

        detector.observe(100); // A
        detector.observe(200); // B
        detector.observe(300); // C
        detector.observe(100); // A
        detector.observe(200); // B

        // A → B → C → A → B → C (3-cycle)
        let info = detector.observe(300);
        assert!(info.is_some());
        let info = info.unwrap();
        assert_eq!(info.period, 3);
    }

    #[test]
    fn test_cycle_detector_no_false_positive() {
        let mut detector = CycleDetector::new(SimplifyPhase::Core);

        // Distinct sequence - no cycles
        for i in 0..20 {
            assert!(detector.observe(i * 1000 + 1).is_none());
        }
    }

    #[test]
    fn test_fingerprint_different_expressions() {
        let mut ctx = cas_ast::Context::new();
        let mut memo = FingerprintMemo::new();

        let x = ctx.var("x");
        let y = ctx.var("y");
        let one = ctx.num(1);

        let x_plus_1 = ctx.add(Expr::Add(x, one));
        let y_plus_1 = ctx.add(Expr::Add(y, one));

        let h1 = expr_fingerprint(&ctx, x_plus_1, &mut memo);
        let h2 = expr_fingerprint(&ctx, y_plus_1, &mut memo);

        // Different expressions should have different fingerprints
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_fingerprint_same_expression() {
        let mut ctx = cas_ast::Context::new();
        let mut memo = FingerprintMemo::new();

        let x = ctx.var("x");
        let one = ctx.num(1);

        let expr1 = ctx.add(Expr::Add(x, one));
        let expr2 = ctx.add(Expr::Add(x, one)); // Same structure, different ExprId

        let h1 = expr_fingerprint(&ctx, expr1, &mut memo);
        memo.clear(); // Clear to force recompute
        let h2 = expr_fingerprint(&ctx, expr2, &mut memo);

        // Same structure should have same fingerprint
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_fingerprint_nested() {
        let mut ctx = cas_ast::Context::new();
        let mut memo = FingerprintMemo::new();

        let x = ctx.var("x");
        let one = ctx.num(1);
        let two = ctx.num(2);

        // (x + 1) * 2
        let x_plus_1 = ctx.add(Expr::Add(x, one));
        let expr = ctx.add(Expr::Mul(x_plus_1, two));

        let h = expr_fingerprint(&ctx, expr, &mut memo);

        // Should produce a valid hash
        assert_ne!(h, 0);
    }

    #[test]
    fn test_fingerprint_functions() {
        let mut ctx = cas_ast::Context::new();
        let mut memo = FingerprintMemo::new();

        let x = ctx.var("x");
        let sin_x = ctx.add(Expr::Function("sin".to_string(), vec![x]));
        let cos_x = ctx.add(Expr::Function("cos".to_string(), vec![x]));

        let h1 = expr_fingerprint(&ctx, sin_x, &mut memo);
        let h2 = expr_fingerprint(&ctx, cos_x, &mut memo);

        // sin(x) != cos(x)
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_cycle_detector_reset() {
        let mut detector = CycleDetector::new(SimplifyPhase::Core);

        detector.observe(100);
        detector.observe(200);
        detector.observe(100);

        // Reset for new phase
        detector.reset(SimplifyPhase::Transform);

        // Should start fresh
        assert!(detector.observe(100).is_none());
        assert!(detector.observe(100).is_some()); // Now detects self-loop
    }
}
