//! Cycle detection infrastructure for short oscillations (A↔B, etc.).

use cas_ast::{Context, Expr, ExprId};
use std::collections::HashMap;

/// Memoization cache for fingerprints (clear per phase/pass).
pub type FingerprintMemo = HashMap<ExprId, u64>;

/// Ring buffer cycle detector for short cycles.
#[derive(Debug)]
pub struct CycleDetector {
    buf: [u64; 64],
    pos: usize,
    len: usize,
    step: usize,
    phase: crate::simplify_phase::SimplifyPhase,
    max_period: usize,
}

impl CycleDetector {
    /// Create a new cycle detector for one simplification phase.
    pub fn new(phase: crate::simplify_phase::SimplifyPhase) -> Self {
        Self {
            buf: [0u64; 64],
            pos: 0,
            len: 0,
            step: 0,
            phase,
            max_period: 16,
        }
    }

    /// Observe one fingerprint and report cycle info if detected.
    pub fn observe(&mut self, hash: u64) -> Option<crate::cycle_models::CycleInfo> {
        self.step += 1;
        let result = self.check_for_cycles(hash);

        self.buf[self.pos] = hash;
        self.pos = (self.pos + 1) % 64;
        if self.len < 64 {
            self.len += 1;
        }

        result
    }

    fn check_for_cycles(&self, hash: u64) -> Option<crate::cycle_models::CycleInfo> {
        if self.len < 1 {
            return None;
        }

        let prev1 = self.get_back(1);
        if hash == prev1 {
            return Some(crate::cycle_models::CycleInfo {
                phase: self.phase,
                period: 1,
                at_step: self.step,
            });
        }

        if self.len >= 3 {
            let prev2 = self.get_back(2);
            let prev3 = self.get_back(3);
            if hash == prev2 && prev1 == prev3 {
                return Some(crate::cycle_models::CycleInfo {
                    phase: self.phase,
                    period: 2,
                    at_step: self.step,
                });
            }
        }

        for period in 3..=self.max_period.min(self.len) {
            if self.check_period(period, hash) {
                return Some(crate::cycle_models::CycleInfo {
                    phase: self.phase,
                    period,
                    at_step: self.step,
                });
            }
        }

        None
    }

    fn check_period(&self, period: usize, current_hash: u64) -> bool {
        if self.len < 2 * period - 1 {
            return false;
        }

        let back_n = self.get_back(period);
        if current_hash != back_n {
            return false;
        }

        for i in 1..period {
            let a = self.get_back(i);
            let b = self.get_back(i + period);
            if a != b {
                return false;
            }
        }

        true
    }

    fn get_back(&self, n: usize) -> u64 {
        debug_assert!(n >= 1 && n <= self.len);
        let idx = (self.pos + 64 - n) % 64;
        self.buf[idx]
    }

    pub fn reset(&mut self, phase: crate::simplify_phase::SimplifyPhase) {
        self.buf = [0u64; 64];
        self.pos = 0;
        self.len = 0;
        self.step = 0;
        self.phase = phase;
    }
}

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

#[inline]
fn mix(tag: u64, a: u64, b: u64) -> u64 {
    let mut x = tag
        .wrapping_add(a)
        .wrapping_add(b.wrapping_mul(0x9E3779B97F4A7C15));
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}

#[inline]
fn mix1(tag: u64, a: u64) -> u64 {
    mix(tag, a, 0)
}

fn hash_str(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for byte in s.bytes() {
        h ^= byte as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Compute structural fingerprint of an expression.
pub fn expr_fingerprint(ctx: &Context, root: ExprId, memo: &mut FingerprintMemo) -> u64 {
    if let Some(&h) = memo.get(&root) {
        return h;
    }

    let h = match ctx.get(root) {
        Expr::Number(q) => {
            let n = q.numer().to_string();
            let d = q.denom().to_string();
            mix(TAG_NUM, hash_str(&n), hash_str(&d))
        }
        Expr::Variable(sym_id) => mix1(TAG_VAR, hash_str(ctx.sym_name(*sym_id))),
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
            let mut h = mix1(TAG_FUNC, hash_str(ctx.sym_name(*name)));
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
        Expr::SessionRef(id) => mix1(TAG_VAR, *id),
        Expr::Hold(x) => {
            let hx = expr_fingerprint(ctx, *x, memo);
            mix1(TAG_NEG, hx)
        }
    };

    memo.insert(root, h);
    h
}
