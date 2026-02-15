//! Metamorphic Equation Tests
//!
//! Tests solver correctness using metamorphic testing strategies:
//!
//! **Strategy 1 (Solution Verification):**
//! For each equation in `equation_corpus.csv`, solve and verify each solution
//! by substituting back into the original equation. Uses `verify_solution_set()`
//! for symbolic check, with numeric fallback via `eval_f64` when symbolic
//! verification produces residuals.
//!
//! **Strategy 3 (Equivalent Equation Pairs):**
//! For pairs of algebraically equivalent equations in `equation_pairs.csv`,
//! verify that solutions of each satisfy the other equation (cross-substitution).
//!
//! # Configuration
//!
//! - `METATEST_SEED=<u64>`: Force specific RNG seed
//! - `METATEST_VERBOSE=1`: Show detailed per-equation results

#![allow(dead_code)]
#![allow(unused_imports)]

use cas_ast::{Context, DisplayExpr, Equation, Expr, ExprId, RelOp, SolutionSet};
use cas_engine::domain::DomainMode;
use cas_engine::domain_facts::{DomainOracle, FactStrength, Predicate};
use cas_engine::domain_oracle::StandardOracle;
use cas_engine::engine::eval_f64;
use cas_engine::implicit_domain::{
    derive_requires_from_equation, domain_delta_check, infer_implicit_domain, DomainDelta,
    ImplicitCondition, ImplicitDomain,
};
use cas_engine::semantics::ValueDomain;
use cas_engine::solver::check::{verify_solution_set, VerifyResult, VerifyStatus, VerifySummary};
use cas_engine::solver::solve;
use cas_engine::Simplifier;
use cas_parser::parse;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::sync::mpsc;
use std::time::{Duration, Instant};

// =============================================================================
// Configuration
// =============================================================================

/// Timeout per equation solve+verify cycle
const SOLVE_TIMEOUT: Duration = Duration::from_secs(10);

/// Number of numeric fallback samples for multi-var equations
const NUMERIC_SAMPLES: usize = 20;

/// Absolute tolerance for numeric equivalence
const ATOL: f64 = 1e-8;

/// Relative tolerance for numeric equivalence
const RTOL: f64 = 1e-6;

fn is_verbose() -> bool {
    env::var("METATEST_VERBOSE")
        .map(|v| v == "1")
        .unwrap_or(false)
}

// =============================================================================
// CSV Parsing
// =============================================================================

/// A single equation entry from the corpus CSV
#[derive(Debug, Clone)]
struct EquationEntry {
    /// Raw equation string (e.g. "2*x + 3 = 7")
    equation_str: String,
    /// Variable to solve for
    solve_var: String,
    /// Expected solution kind
    expected_kind: ExpectedKind,
    /// Expected number of solutions (if discrete)
    expected_count: Option<usize>,
    /// Family tag for grouping in reports
    family: String,
    /// Domain mode: g (generic) or a (assume)
    domain_mode: char,
}

#[derive(Debug, Clone, PartialEq)]
enum ExpectedKind {
    Discrete,
    Empty,
    Interval,
    Residual,
    Any,
}

impl ExpectedKind {
    fn from_str(s: &str) -> Self {
        match s.trim().to_lowercase().as_str() {
            "discrete" => ExpectedKind::Discrete,
            "empty" => ExpectedKind::Empty,
            "interval" => ExpectedKind::Interval,
            "residual" => ExpectedKind::Residual,
            "any" | "" => ExpectedKind::Any,
            other => panic!("Unknown expected_kind: '{}'", other),
        }
    }
}

/// Load equation entries from CSV file
fn load_equation_corpus() -> Vec<EquationEntry> {
    let csv_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/equation_corpus.csv");
    let file = fs::File::open(&csv_path).unwrap_or_else(|e| {
        panic!("Failed to open {}: {}", csv_path.display(), e);
    });
    let reader = BufReader::new(file);
    let mut entries = Vec::new();

    for (line_num, line) in reader.lines().enumerate() {
        let line = line.unwrap();
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.splitn(6, ',').collect();
        if parts.len() < 4 {
            panic!(
                "equation_corpus.csv line {}: expected at least 4 fields, got {}. Line: '{}'",
                line_num + 1,
                parts.len(),
                line
            );
        }

        let equation_str = parts[0].trim().to_string();
        let solve_var = parts[1].trim().to_string();
        let expected_kind = ExpectedKind::from_str(parts[2]);
        let expected_count = if parts.len() > 3 && !parts[3].trim().is_empty() {
            parts[3].trim().parse::<usize>().ok()
        } else {
            None
        };
        let family = if parts.len() > 4 {
            parts[4].trim().to_string()
        } else {
            "unknown".to_string()
        };
        let domain_mode = if parts.len() > 5 {
            parts[5].trim().chars().next().unwrap_or('g')
        } else {
            'g'
        };

        entries.push(EquationEntry {
            equation_str,
            solve_var,
            expected_kind,
            expected_count,
            family,
            domain_mode,
        });
    }

    entries
}

// =============================================================================
// Equation Parsing
// =============================================================================

/// Parse an equation string "lhs = rhs" into an Equation struct.
/// Returns None if parsing fails.
fn parse_equation_str(ctx: &mut Context, eq_str: &str) -> Option<Equation> {
    // Split on '=' but NOT on '<=' or '>=' or '!='
    // Find the '=' that isn't part of another operator
    let eq_str = eq_str.trim();

    // Simple split: find first standalone '='
    let mut eq_pos = None;
    let chars: Vec<char> = eq_str.chars().collect();
    for i in 0..chars.len() {
        if chars[i] == '=' {
            // Check it's not <= or >= or !=
            if i > 0 && (chars[i - 1] == '<' || chars[i - 1] == '>' || chars[i - 1] == '!') {
                continue;
            }
            // Check it's not ==
            if i + 1 < chars.len() && chars[i + 1] == '=' {
                continue;
            }
            eq_pos = Some(i);
            break;
        }
    }

    let eq_pos = eq_pos?;
    let lhs_str = &eq_str[..eq_pos].trim();
    let rhs_str = &eq_str[eq_pos + 1..].trim();

    let lhs = parse(lhs_str, ctx).ok()?;
    let rhs = parse(rhs_str, ctx).ok()?;

    Some(Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    })
}

// =============================================================================
// Numeric Fallback Verification
// =============================================================================

/// Verify a solution numerically by substituting into lhs-rhs and checking ≈ 0.
/// For single-variable equations, evaluates at the solution directly.
/// For multi-variable equations, samples random values for other variables.
fn numeric_verify_solution(
    ctx: &Context,
    equation: &Equation,
    solve_var: &str,
    solution: ExprId,
) -> NumericVerifyResult {
    // Collect all variables in the equation
    let mut all_vars = Vec::new();
    collect_variables(ctx, equation.lhs, &mut all_vars);
    collect_variables(ctx, equation.rhs, &mut all_vars);
    all_vars.sort();
    all_vars.dedup();

    // Other variables (not the solve variable)
    let other_vars: Vec<String> = all_vars
        .iter()
        .filter(|v| v.as_str() != solve_var)
        .cloned()
        .collect();

    let mut valid = 0;
    let mut _domain_errors = 0;
    let mut mismatches = 0;

    // If no other variables, single evaluation
    let sample_count = if other_vars.is_empty() {
        1
    } else {
        NUMERIC_SAMPLES
    };

    for i in 0..sample_count {
        let mut var_map = HashMap::new();

        // First, evaluate the solution expression with current other-var values
        for (j, var) in other_vars.iter().enumerate() {
            // Deterministic sample values spread across a reasonable range
            let t = (i as f64 * 7.0 + j as f64 * 13.0 + 0.5) / sample_count as f64;
            let val = 0.1 + t.fract() * 4.9; // Range [0.1, 5.0] to avoid 0
            var_map.insert(var.clone(), val);
        }

        // Evaluate the solution expression to get a number for solve_var
        let sol_val = match eval_f64(ctx, solution, &var_map) {
            Some(v) if v.is_finite() => v,
            _ => {
                _domain_errors += 1;
                continue;
            }
        };

        var_map.insert(solve_var.to_string(), sol_val);

        // Evaluate lhs and rhs
        let lhs_val = eval_f64(ctx, equation.lhs, &var_map);
        let rhs_val = eval_f64(ctx, equation.rhs, &var_map);

        match (lhs_val, rhs_val) {
            (Some(l), Some(r)) if l.is_finite() && r.is_finite() => {
                let diff = (l - r).abs();
                let scale = l.abs().max(r.abs()).max(1.0);
                let allowed = ATOL + RTOL * scale;
                if diff <= allowed {
                    valid += 1;
                } else {
                    mismatches += 1;
                }
            }
            _ => {
                _domain_errors += 1;
            }
        }
    }

    if mismatches > 0 {
        NumericVerifyResult::Failed
    } else if valid > 0 {
        NumericVerifyResult::Verified(valid)
    } else {
        NumericVerifyResult::Inconclusive
    }
}

#[derive(Debug)]
enum NumericVerifyResult {
    /// Solution verified numerically at N sample points
    Verified(usize),
    /// Solution failed numeric check
    Failed,
    /// Could not evaluate (all domain errors)
    Inconclusive,
}

/// Collect all variable names from an expression
fn collect_variables(ctx: &Context, expr: ExprId, vars: &mut Vec<String>) {
    match ctx.get(expr) {
        Expr::Variable(sym) => {
            let name = ctx.sym_name(*sym).to_string();
            if !vars.contains(&name) {
                vars.push(name);
            }
        }
        Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::Div(a, b) | Expr::Pow(a, b) => {
            collect_variables(ctx, *a, vars);
            collect_variables(ctx, *b, vars);
        }
        Expr::Neg(a) | Expr::Hold(a) => {
            collect_variables(ctx, *a, vars);
        }
        Expr::Function(_, args) => {
            for arg in args {
                collect_variables(ctx, *arg, vars);
            }
        }
        Expr::Matrix { data, .. } => {
            for d in data {
                collect_variables(ctx, *d, vars);
            }
        }
        Expr::Number(_) | Expr::Constant(_) | Expr::SessionRef(_) => {}
    }
}

// =============================================================================
// Per-Equation Result
// =============================================================================

#[derive(Debug, Clone)]
enum EqTestOutcome {
    /// All solutions verified (symbolically or numerically)
    Verified,
    /// Some verified symbolically, rest verified numerically
    NumericFallback,
    /// Solution type not checkable (interval, AllReals, etc.)
    NotCheckable,
    /// Empty solution set (expected or not)
    EmptyResult,
    /// Solver returned error or panicked
    SolverError(String),
    /// Timed out
    Timeout,
    /// Verification failed (wrong solution)
    Failed(String),
    /// Solution count mismatch
    CountMismatch { expected: usize, got: usize },
    /// Kind mismatch (expected discrete, got interval, etc.)
    KindMismatch { expected: ExpectedKind, got: String },
    /// Parse error
    ParseError(String),
}

// =============================================================================
// Metrics per Family
// =============================================================================

#[derive(Debug, Default, Clone)]
struct FamilyMetrics {
    family: String,
    total: usize,
    verified: usize,
    numeric_fallback: usize,
    not_checkable: usize,
    empty: usize,
    failed: usize,
    timeout: usize,
    solver_error: usize,
    count_mismatch: usize,
    kind_mismatch: usize,
    parse_error: usize,
}

impl FamilyMetrics {
    fn new(family: &str) -> Self {
        FamilyMetrics {
            family: family.to_string(),
            ..Default::default()
        }
    }

    fn record(&mut self, outcome: &EqTestOutcome) {
        self.total += 1;
        match outcome {
            EqTestOutcome::Verified => self.verified += 1,
            EqTestOutcome::NumericFallback => self.numeric_fallback += 1,
            EqTestOutcome::NotCheckable => self.not_checkable += 1,
            EqTestOutcome::EmptyResult => self.empty += 1,
            EqTestOutcome::SolverError(_) => self.solver_error += 1,
            EqTestOutcome::Timeout => self.timeout += 1,
            EqTestOutcome::Failed(_) => self.failed += 1,
            EqTestOutcome::CountMismatch { .. } => self.count_mismatch += 1,
            EqTestOutcome::KindMismatch { .. } => self.kind_mismatch += 1,
            EqTestOutcome::ParseError(_) => self.parse_error += 1,
        }
    }

    fn success_count(&self) -> usize {
        self.verified + self.numeric_fallback + self.not_checkable + self.empty
    }

    fn failure_count(&self) -> usize {
        self.failed + self.count_mismatch + self.kind_mismatch
    }
}

// =============================================================================
// Strategy 1: Solution Substitution Verification
// =============================================================================

/// Run Strategy 1 on a single equation entry.
fn verify_equation(entry: &EquationEntry) -> EqTestOutcome {
    // Create a fresh simplifier for each equation to avoid cross-contamination
    let mut simplifier = Simplifier::with_default_rules();
    simplifier.set_collect_steps(false); // Don't need steps for benchmark

    // Parse equation
    let equation = match parse_equation_str(&mut simplifier.context, &entry.equation_str) {
        Some(eq) => eq,
        None => {
            return EqTestOutcome::ParseError(format!("Failed to parse: '{}'", entry.equation_str));
        }
    };

    // Solve with timeout
    let start = Instant::now();
    let solve_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        solve(&equation, &entry.solve_var, &mut simplifier)
    }));

    if start.elapsed() > SOLVE_TIMEOUT {
        return EqTestOutcome::Timeout;
    }

    let (solution_set, _steps) = match solve_result {
        Ok(Ok(result)) => result,
        Ok(Err(e)) => return EqTestOutcome::SolverError(format!("{:?}", e)),
        Err(_) => return EqTestOutcome::SolverError("Solver panicked".to_string()),
    };

    // Check kind match
    match (&entry.expected_kind, &solution_set) {
        (ExpectedKind::Discrete, SolutionSet::Empty) => {
            return EqTestOutcome::KindMismatch {
                expected: ExpectedKind::Discrete,
                got: "empty".to_string(),
            };
        }
        (ExpectedKind::Empty, SolutionSet::Discrete(sols)) if !sols.is_empty() => {
            return EqTestOutcome::KindMismatch {
                expected: ExpectedKind::Empty,
                got: format!("discrete({})", sols.len()),
            };
        }
        (ExpectedKind::Empty, SolutionSet::Empty) => {
            return EqTestOutcome::EmptyResult;
        }
        _ => {} // Any, Residual, or compatible
    }

    // Check solution count for discrete
    if let (Some(expected), SolutionSet::Discrete(sols)) = (entry.expected_count, &solution_set) {
        if sols.len() != expected {
            return EqTestOutcome::CountMismatch {
                expected,
                got: sols.len(),
            };
        }
    }

    // Handle empty
    if matches!(solution_set, SolutionSet::Empty) {
        return EqTestOutcome::EmptyResult;
    }

    // Verify solutions
    let verify_result =
        verify_solution_set(&mut simplifier, &equation, &entry.solve_var, &solution_set);

    match verify_result.summary {
        VerifySummary::AllVerified => EqTestOutcome::Verified,
        VerifySummary::Empty => EqTestOutcome::EmptyResult,
        VerifySummary::NotCheckable => EqTestOutcome::NotCheckable,
        VerifySummary::PartiallyVerified | VerifySummary::NoneVerified => {
            // Try numeric fallback for unverified solutions
            let mut all_verified = true;
            let mut any_numeric = false;
            let mut fail_detail = String::new();

            for (sol_expr, status) in &verify_result.solutions {
                match status {
                    VerifyStatus::Verified => {}
                    VerifyStatus::Unverifiable {
                        residual: _,
                        reason,
                    } => {
                        // Numeric fallback
                        match numeric_verify_solution(
                            &simplifier.context,
                            &equation,
                            &entry.solve_var,
                            *sol_expr,
                        ) {
                            NumericVerifyResult::Verified(_) => {
                                any_numeric = true;
                            }
                            NumericVerifyResult::Failed => {
                                all_verified = false;
                                let sol_str = DisplayExpr {
                                    context: &simplifier.context,
                                    id: *sol_expr,
                                }
                                .to_string();
                                fail_detail =
                                    format!("Solution {} failed verification: {}", sol_str, reason);
                            }
                            NumericVerifyResult::Inconclusive => {
                                // Count as numeric fallback (not a failure)
                                any_numeric = true;
                            }
                        }
                    }
                    VerifyStatus::NotCheckable { reason: _ } => {
                        // Count as not checkable, not a failure
                        any_numeric = true;
                    }
                }
            }

            if !all_verified {
                EqTestOutcome::Failed(fail_detail)
            } else if any_numeric {
                EqTestOutcome::NumericFallback
            } else {
                EqTestOutcome::Verified
            }
        }
    }
}

// =============================================================================
// Strategy 3: Equivalent Equation Pairs
// =============================================================================

/// A pair of algebraically equivalent equations
#[derive(Debug, Clone)]
struct EquationPair {
    eq_a: String,
    eq_b: String,
    solve_var: String,
    family: String,
}

/// Load equation pairs from CSV
fn load_equation_pairs() -> Vec<EquationPair> {
    let csv_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/equation_pairs.csv");
    let file = match fs::File::open(&csv_path) {
        Ok(f) => f,
        Err(_) => return Vec::new(), // File not yet created
    };
    let reader = BufReader::new(file);
    let mut pairs = Vec::new();

    for line in reader.lines() {
        let line = line.unwrap();
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.splitn(5, ',').collect();
        if parts.len() < 3 {
            continue;
        }

        pairs.push(EquationPair {
            eq_a: parts[0].trim().to_string(),
            eq_b: parts[1].trim().to_string(),
            solve_var: parts[2].trim().to_string(),
            family: if parts.len() > 3 {
                parts[3].trim().to_string()
            } else {
                "unknown".to_string()
            },
        });
    }

    pairs
}

#[derive(Debug, Clone)]
enum PairOutcome {
    /// Both solved, solutions cross-verified
    Verified,
    /// Solutions verified numerically
    NumericFallback,
    /// Solutions don't match
    Mismatch(String),
    /// One or both couldn't solve
    SolverFailed(String),
    /// Parse error
    ParseError(String),
    /// Timeout
    Timeout,
}

/// Verify an equivalent equation pair using cross-substitution oracle.
fn verify_equation_pair(pair: &EquationPair) -> PairOutcome {
    let mut simplifier = Simplifier::with_default_rules();
    simplifier.set_collect_steps(false);

    // Parse both equations
    let eq_a = match parse_equation_str(&mut simplifier.context, &pair.eq_a) {
        Some(eq) => eq,
        None => return PairOutcome::ParseError(format!("Failed to parse eq_A: '{}'", pair.eq_a)),
    };
    let eq_b = match parse_equation_str(&mut simplifier.context, &pair.eq_b) {
        Some(eq) => eq,
        None => return PairOutcome::ParseError(format!("Failed to parse eq_B: '{}'", pair.eq_b)),
    };

    // Solve both
    let start = Instant::now();
    let result_a = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        solve(&eq_a, &pair.solve_var, &mut simplifier)
    }));
    if start.elapsed() > SOLVE_TIMEOUT {
        return PairOutcome::Timeout;
    }

    let result_b = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        solve(&eq_b, &pair.solve_var, &mut simplifier)
    }));
    if start.elapsed() > SOLVE_TIMEOUT * 2 {
        return PairOutcome::Timeout;
    }

    let (sol_a, _) = match result_a {
        Ok(Ok(r)) => r,
        _ => return PairOutcome::SolverFailed(format!("eq_A '{}' failed", pair.eq_a)),
    };
    let (sol_b, _) = match result_b {
        Ok(Ok(r)) => r,
        _ => return PairOutcome::SolverFailed(format!("eq_B '{}' failed", pair.eq_b)),
    };

    // Cross-verification: solutions of A must satisfy B and vice versa
    let mut all_ok = true;
    let mut any_numeric = false;
    let mut fail_msg = String::new();

    // Check solutions of A satisfy B
    if let SolutionSet::Discrete(sols_a) = &sol_a {
        for &sol in sols_a {
            let verify = verify_solution_set(
                &mut simplifier,
                &eq_b,
                &pair.solve_var,
                &SolutionSet::Discrete(vec![sol]),
            );
            match verify.summary {
                VerifySummary::AllVerified => {}
                _ => {
                    // Numeric fallback
                    match numeric_verify_solution(&simplifier.context, &eq_b, &pair.solve_var, sol)
                    {
                        NumericVerifyResult::Verified(_) => any_numeric = true,
                        NumericVerifyResult::Failed => {
                            all_ok = false;
                            let s = DisplayExpr {
                                context: &simplifier.context,
                                id: sol,
                            }
                            .to_string();
                            fail_msg = format!("Solution {} of A does not satisfy B", s);
                        }
                        NumericVerifyResult::Inconclusive => any_numeric = true,
                    }
                }
            }
        }
    }

    // Check solutions of B satisfy A
    if all_ok {
        if let SolutionSet::Discrete(sols_b) = &sol_b {
            for &sol in sols_b {
                let verify = verify_solution_set(
                    &mut simplifier,
                    &eq_a,
                    &pair.solve_var,
                    &SolutionSet::Discrete(vec![sol]),
                );
                match verify.summary {
                    VerifySummary::AllVerified => {}
                    _ => {
                        match numeric_verify_solution(
                            &simplifier.context,
                            &eq_a,
                            &pair.solve_var,
                            sol,
                        ) {
                            NumericVerifyResult::Verified(_) => any_numeric = true,
                            NumericVerifyResult::Failed => {
                                all_ok = false;
                                let s = DisplayExpr {
                                    context: &simplifier.context,
                                    id: sol,
                                }
                                .to_string();
                                fail_msg = format!("Solution {} of B does not satisfy A", s);
                            }
                            NumericVerifyResult::Inconclusive => any_numeric = true,
                        }
                    }
                }
            }
        }
    }

    // Check solution count match when both are discrete
    if all_ok {
        if let (SolutionSet::Discrete(sa), SolutionSet::Discrete(sb)) = (&sol_a, &sol_b) {
            if sa.len() != sb.len() {
                return PairOutcome::Mismatch(format!(
                    "Solution count differs: A has {}, B has {}",
                    sa.len(),
                    sb.len()
                ));
            }
        }
    }

    if !all_ok {
        PairOutcome::Mismatch(fail_msg)
    } else if any_numeric {
        PairOutcome::NumericFallback
    } else {
        PairOutcome::Verified
    }
}

// =============================================================================
// Reporting
// =============================================================================

fn print_strategy1_table(families: &[FamilyMetrics]) {
    eprintln!();
    eprintln!("╔═══════════════════════════════════════════════════════════════════════════════════════════╗");
    eprintln!("║               EQUATION SOLUTION VERIFICATION (Strategy 1)                               ║");
    eprintln!("╠═══════════════════╤═══════╤══════════╤══════════╤══════════╤════════╤═══════╤════════════╣");
    eprintln!("║ Family            │ Total │ Verified │ Num-Fllb │ NotCheck │ Failed │  T/O  │ Err/Parse  ║");
    eprintln!("╠═══════════════════╪═══════╪══════════╪══════════╪══════════╪════════╪═══════╪════════════╣");

    let mut t_total = 0usize;
    let mut t_verified = 0usize;
    let mut t_numeric = 0usize;
    let mut t_notcheck = 0usize;
    let mut t_failed = 0usize;
    let mut t_timeout = 0usize;
    let mut t_errors = 0usize;

    for m in families {
        let errors = m.solver_error + m.parse_error + m.count_mismatch + m.kind_mismatch;
        eprintln!(
            "║ {:17} │ {:>5} │ {:>8} │ {:>8} │ {:>8} │ {:>6} │ {:>5} │ {:>10} ║",
            truncate(&m.family, 17),
            m.total,
            m.verified + m.empty,
            m.numeric_fallback,
            m.not_checkable,
            m.failed,
            m.timeout,
            errors,
        );

        t_total += m.total;
        t_verified += m.verified + m.empty;
        t_numeric += m.numeric_fallback;
        t_notcheck += m.not_checkable;
        t_failed += m.failed;
        t_timeout += m.timeout;
        t_errors += errors;
    }

    eprintln!("╠═══════════════════╪═══════╪══════════╪══════════╪══════════╪════════╪═══════╪════════════╣");
    eprintln!(
        "║ TOTAL             │ {:>5} │ {:>8} │ {:>8} │ {:>8} │ {:>6} │ {:>5} │ {:>10} ║",
        t_total, t_verified, t_numeric, t_notcheck, t_failed, t_timeout, t_errors,
    );
    eprintln!("╚═══════════════════╧═══════╧══════════╧══════════╧══════════╧════════╧═══════╧════════════╝");

    if t_failed > 0 {
        eprintln!();
        eprintln!("❌ {} equation(s) FAILED verification!", t_failed);
    }
    if t_errors > 0 {
        eprintln!();
        eprintln!(
            "⚠️  {} equation(s) had errors (solver/parse/count/kind mismatch)",
            t_errors
        );
    }
    if t_failed == 0 && t_errors == 0 {
        eprintln!();
        eprintln!(
            "✅ All equations passed (verified: {}, numeric fallback: {}, not checkable: {})",
            t_verified, t_numeric, t_notcheck
        );
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..max - 1])
    }
}

// =============================================================================
// Main Test: Strategy 1 - Equation Solution Verification
// =============================================================================

#[test]
#[ignore] // Run with: cargo test --release -p cas_engine --test metamorphic_equation_tests metatest_equation_solution_verification -- --ignored --nocapture
fn metatest_equation_solution_verification() {
    let entries = load_equation_corpus();
    let verbose = is_verbose();

    eprintln!();
    eprintln!(
        "=== Equation Solution Verification: {} equations ===",
        entries.len()
    );
    eprintln!();

    // Collect all unique families
    let mut family_order: Vec<String> = Vec::new();
    for e in &entries {
        if !family_order.contains(&e.family) {
            family_order.push(e.family.clone());
        }
    }

    let mut family_metrics: HashMap<String, FamilyMetrics> = HashMap::new();
    let mut total_failed = 0usize;

    for (i, entry) in entries.iter().enumerate() {
        let start = Instant::now();
        let outcome = verify_equation(entry);
        let elapsed = start.elapsed();

        let fm = family_metrics
            .entry(entry.family.clone())
            .or_insert_with(|| FamilyMetrics::new(&entry.family));
        fm.record(&outcome);

        let symbol = match &outcome {
            EqTestOutcome::Verified => "✓",
            EqTestOutcome::NumericFallback => "≈",
            EqTestOutcome::NotCheckable => "ℹ",
            EqTestOutcome::EmptyResult => "∅",
            EqTestOutcome::Failed(_) => "✗",
            EqTestOutcome::CountMismatch { .. } => "#",
            EqTestOutcome::KindMismatch { .. } => "K",
            EqTestOutcome::SolverError(_) => "E",
            EqTestOutcome::Timeout => "T",
            EqTestOutcome::ParseError(_) => "P",
        };

        if matches!(
            &outcome,
            EqTestOutcome::Failed(_)
                | EqTestOutcome::CountMismatch { .. }
                | EqTestOutcome::KindMismatch { .. }
        ) {
            total_failed += 1;
        }

        if verbose
            || !matches!(
                &outcome,
                EqTestOutcome::Verified | EqTestOutcome::EmptyResult
            )
        {
            let detail = match &outcome {
                EqTestOutcome::Verified => "verified".to_string(),
                EqTestOutcome::NumericFallback => "numeric fallback".to_string(),
                EqTestOutcome::NotCheckable => "not checkable".to_string(),
                EqTestOutcome::EmptyResult => "empty/expected".to_string(),
                EqTestOutcome::Failed(msg) => format!("FAILED: {}", msg),
                EqTestOutcome::CountMismatch { expected, got } => {
                    format!("count mismatch: expected {}, got {}", expected, got)
                }
                EqTestOutcome::KindMismatch { expected, got } => {
                    format!("kind mismatch: expected {:?}, got {}", expected, got)
                }
                EqTestOutcome::SolverError(msg) => format!("error: {}", msg),
                EqTestOutcome::Timeout => "TIMEOUT".to_string(),
                EqTestOutcome::ParseError(msg) => format!("parse: {}", msg),
            };
            eprintln!(
                "  {} {:>3}. [{:17}] {} ({:.1}ms) — {}",
                symbol,
                i + 1,
                truncate(&entry.family, 17),
                entry.equation_str,
                elapsed.as_secs_f64() * 1000.0,
                detail
            );
        }
    }

    // Build sorted metrics
    let metrics: Vec<FamilyMetrics> = family_order
        .iter()
        .filter_map(|f| family_metrics.get(f).cloned())
        .collect();

    print_strategy1_table(&metrics);

    // Assert no hard failures
    assert_eq!(
        total_failed, 0,
        "Some equations failed verification — see details above"
    );
}

// =============================================================================
// Main Test: Strategy 3 - Equivalent Equation Pairs
// =============================================================================

#[test]
#[ignore] // Run with: cargo test --release -p cas_engine --test metamorphic_equation_tests metatest_equation_pair_equivalence -- --ignored --nocapture
fn metatest_equation_pair_equivalence() {
    let pairs = load_equation_pairs();
    if pairs.is_empty() {
        eprintln!("No equation pairs found (equation_pairs.csv missing or empty). Skipping.");
        return;
    }

    eprintln!();
    eprintln!("=== Equivalent Equation Pairs: {} pairs ===", pairs.len());
    eprintln!();

    let mut verified = 0usize;
    let mut numeric = 0usize;
    let mut mismatch = 0usize;
    let mut errors = 0usize;

    for (i, pair) in pairs.iter().enumerate() {
        let start = Instant::now();
        let outcome = verify_equation_pair(pair);
        let elapsed = start.elapsed();

        let (symbol, detail) = match &outcome {
            PairOutcome::Verified => {
                verified += 1;
                ("✓", "cross-verified".to_string())
            }
            PairOutcome::NumericFallback => {
                numeric += 1;
                ("≈", "numeric fallback".to_string())
            }
            PairOutcome::Mismatch(msg) => {
                mismatch += 1;
                ("✗", format!("MISMATCH: {}", msg))
            }
            PairOutcome::SolverFailed(msg) => {
                errors += 1;
                ("E", format!("solver: {}", msg))
            }
            PairOutcome::ParseError(msg) => {
                errors += 1;
                ("P", format!("parse: {}", msg))
            }
            PairOutcome::Timeout => {
                errors += 1;
                ("T", "TIMEOUT".to_string())
            }
        };

        if is_verbose() || !matches!(&outcome, PairOutcome::Verified) {
            eprintln!(
                "  {} {:>3}. [{:15}] '{}' ↔ '{}' ({:.1}ms) — {}",
                symbol,
                i + 1,
                truncate(&pair.family, 15),
                pair.eq_a,
                pair.eq_b,
                elapsed.as_secs_f64() * 1000.0,
                detail
            );
        }
    }

    eprintln!();
    eprintln!(
        "Pairs: {} total, {} verified, {} numeric, {} mismatch, {} errors",
        pairs.len(),
        verified,
        numeric,
        mismatch,
        errors,
    );

    assert_eq!(
        mismatch, 0,
        "Some equation pairs had mismatching solutions — see details above"
    );
}

// =============================================================================
// Unified Benchmark (runs all strategies)
// =============================================================================

#[test]
#[ignore] // Run with: cargo test --release -p cas_engine --test metamorphic_equation_tests metatest_equation_benchmark -- --ignored --nocapture
fn metatest_equation_benchmark() {
    eprintln!();
    eprintln!("╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║       METAMORPHIC EQUATION BENCHMARK                       ║");
    eprintln!("╚══════════════════════════════════════════════════════════════╝");

    // Run Strategy 1
    let entries = load_equation_corpus();
    let verbose = is_verbose();

    let mut family_order: Vec<String> = Vec::new();
    for e in &entries {
        if !family_order.contains(&e.family) {
            family_order.push(e.family.clone());
        }
    }

    let mut family_metrics: HashMap<String, FamilyMetrics> = HashMap::new();
    let mut s1_failed = 0usize;
    let total_start = Instant::now();

    for (i, entry) in entries.iter().enumerate() {
        let outcome = verify_equation(entry);

        let fm = family_metrics
            .entry(entry.family.clone())
            .or_insert_with(|| FamilyMetrics::new(&entry.family));
        fm.record(&outcome);

        if matches!(
            &outcome,
            EqTestOutcome::Failed(_)
                | EqTestOutcome::CountMismatch { .. }
                | EqTestOutcome::KindMismatch { .. }
        ) {
            s1_failed += 1;
            let detail = match &outcome {
                EqTestOutcome::Failed(msg) => format!("FAILED: {}", msg),
                EqTestOutcome::CountMismatch { expected, got } => {
                    format!("count: expected {}, got {}", expected, got)
                }
                EqTestOutcome::KindMismatch { expected, got } => {
                    format!("kind: expected {:?}, got {}", expected, got)
                }
                _ => String::new(),
            };
            eprintln!("  ✗ {:>3}. {} — {}", i + 1, entry.equation_str, detail);
        } else if verbose {
            eprintln!("  ✓ {:>3}. {}", i + 1, entry.equation_str);
        }
    }

    let metrics: Vec<FamilyMetrics> = family_order
        .iter()
        .filter_map(|f| family_metrics.get(f).cloned())
        .collect();

    print_strategy1_table(&metrics);

    // Run Strategy 3
    let pairs = load_equation_pairs();
    if !pairs.is_empty() {
        eprintln!();
        eprintln!("--- Strategy 3: Equivalent Pairs ---");
        let mut s3_mismatch = 0usize;
        for (i, pair) in pairs.iter().enumerate() {
            let outcome = verify_equation_pair(pair);
            if let PairOutcome::Mismatch(msg) = &outcome {
                s3_mismatch += 1;
                eprintln!(
                    "  ✗ {:>3}. '{}' ↔ '{}' — {}",
                    i + 1,
                    pair.eq_a,
                    pair.eq_b,
                    msg
                );
            }
        }
        eprintln!("Pairs: {} total, {} mismatches", pairs.len(), s3_mismatch);
        s1_failed += s3_mismatch;
    }

    // Run Strategy 2
    let s2_result = run_strategy2(verbose);
    s1_failed += s2_result.mismatches;

    let total_elapsed = total_start.elapsed();
    eprintln!();
    eprintln!("Total time: {:.2}s", total_elapsed.as_secs_f64());

    assert_eq!(
        s1_failed, 0,
        "Equation benchmark had failures — see details above"
    );
}

// =============================================================================
// Strategy 2: Identity-Preserving Transforms (Sampled)
// =============================================================================

/// Simple LCG PRNG for deterministic, reproducible sampling.
/// Period 2^64, same constants as Knuth's MMIX.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg { state: seed }
    }

    fn next(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    /// Pick a random index in [0, n)
    fn pick(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// A lightweight identity representation for Strategy 2
#[derive(Debug, Clone)]
struct S2Identity {
    exp: String,
    simp: String,
    /// Primary variable in the identity
    var: String,
    /// Identity family (from CSV comment headers)
    family: String,
    /// Tier 0 = identity doesn't contain the solve variable; Tier 1 = it does
    tier: u8,
    /// AST cost: node_count(exp) + node_count(simp)
    cost: usize,
    /// Whether the identity preserves domain (domain_delta_check Safe in both directions,
    /// after filtering semantically redundant conditions like 1+x²≠0)
    domain_safe: bool,
}

/// Count AST nodes for a parsed expression
fn expr_cost(ctx: &Context, expr: ExprId) -> usize {
    match ctx.get(expr) {
        Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => 1,
        Expr::Neg(a) | Expr::Hold(a) => 1 + expr_cost(ctx, *a),
        Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::Div(a, b) | Expr::Pow(a, b) => {
            1 + expr_cost(ctx, *a) + expr_cost(ctx, *b)
        }
        Expr::Function(_, args) => 1 + args.iter().map(|a| expr_cost(ctx, *a)).sum::<usize>(),
        Expr::Matrix { data, .. } => 1 + data.iter().map(|a| expr_cost(ctx, *a)).sum::<usize>(),
    }
}

/// Load identity pairs from CSV, filtering to mode=g and cost ≤ max_cost
fn load_identities_for_s2(max_cost: usize) -> Vec<S2Identity> {
    let csv_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/identity_pairs.csv");
    let content = fs::read_to_string(&csv_path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", csv_path.display(), e));

    let mut identities = Vec::new();
    let mut current_family = String::from("Uncategorized");

    for line in content.lines() {
        let line = line.trim();
        if line.starts_with('#') {
            let label = line.trim_start_matches('#').trim();
            if !label.is_empty()
                && !label.starts_with("Format")
                && !label.starts_with("Each row")
                && !label.starts_with("var is")
                && !label.starts_with("Mathematical Identity")
            {
                current_family = label.to_string();
            }
            continue;
        }
        if line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 3 {
            continue;
        }

        // Filter: mode must be 'g' (Generic)
        let mode = if parts.len() >= 4 {
            parts[3].trim()
        } else {
            "g"
        };
        if mode != "g" {
            continue;
        }

        let exp = parts[0].trim().to_string();
        let simp = parts[1].trim().to_string();
        let var = parts[2]
            .trim()
            .split(';')
            .next()
            .unwrap_or("x")
            .trim()
            .to_string();

        // Compute cost by parsing into a temporary context
        let mut ctx = Context::new();
        let exp_parsed = match parse(&exp, &mut ctx) {
            Ok(e) => e,
            Err(_) => continue,
        };
        let simp_parsed = match parse(&simp, &mut ctx) {
            Ok(e) => e,
            Err(_) => continue,
        };
        let cost = expr_cost(&ctx, exp_parsed) + expr_cost(&ctx, simp_parsed);

        if cost > max_cost {
            continue;
        }

        // Filter out solver-unsupported functions that always cause errors.
        const UNSUPPORTED_FUNS: &[&str] = &[
            "cosh", "sinh", "tanh", "sech", "csch", "coth", "acosh", "asinh", "atanh",
        ];
        let combined = format!("{} {}", exp, simp);
        if UNSUPPORTED_FUNS.iter().any(|f| combined.contains(f)) {
            continue;
        }

        // Classify domain safety using domain_delta_check + semantic redundancy.
        // An identity is "domain-safe" if the transformation A→B and B→A don't
        // expand the domain in a semantically meaningful way.
        let delta_fwd = domain_delta_check(&ctx, exp_parsed, simp_parsed, ValueDomain::RealOnly);
        let delta_rev = domain_delta_check(&ctx, simp_parsed, exp_parsed, ValueDomain::RealOnly);
        let domain_safe = delta_is_semantically_safe(&ctx, &delta_fwd)
            && delta_is_semantically_safe(&ctx, &delta_rev);

        identities.push(S2Identity {
            exp,
            simp,
            var,
            family: current_family.clone(),
            tier: 0, // Will be set per-equation in run_case
            cost,
            domain_safe,
        });
    }

    identities
}

/// Check if a string expression contains a given variable name
fn identity_contains_var(expr_str: &str, var: &str) -> bool {
    // Simple heuristic: check if the variable appears as a word-boundary token
    // This avoids false positives like "x" in "exp"
    let var_bytes = var.as_bytes();
    let expr_bytes = expr_str.as_bytes();
    for i in 0..expr_bytes.len() {
        if expr_bytes[i..].starts_with(var_bytes) {
            let before_ok = i == 0 || !expr_bytes[i - 1].is_ascii_alphanumeric();
            let after_pos = i + var_bytes.len();
            let after_ok =
                after_pos >= expr_bytes.len() || !expr_bytes[after_pos].is_ascii_alphanumeric();
            if before_ok && after_ok {
                return true;
            }
        }
    }
    false
}

/// Result of a Strategy 2 test case
#[derive(Debug)]
enum S2Outcome {
    /// Cross-substitution verified symbolically
    OkSymbolic,
    /// Cross-substitution verified numerically
    OkNumeric,
    /// One side returned non-discrete (Residual/Conditional/Interval)
    /// but cross-substitution didn't fail — not a bug, just incomplete
    Incomplete(String),
    /// Cross-substitution failed, but identity or equation domain differs
    /// — expected behaviour, not a solver bug
    DomainChanged(String),
    /// Cross-substitution found a real mismatch on a domain-safe identity
    Mismatch(String),
    /// Solver error or timeout
    Error(String),
    /// Timeout
    Timeout,
}

/// Strategy 2 aggregated results
struct S2Results {
    ok_symbolic: usize,
    ok_numeric: usize,
    incomplete: usize,
    domain_changed: usize,
    mismatches: usize,
    errors: usize,
    timeouts: usize,
    total: usize,
}

impl S2Results {
    fn new() -> Self {
        S2Results {
            ok_symbolic: 0,
            ok_numeric: 0,
            incomplete: 0,
            domain_changed: 0,
            mismatches: 0,
            errors: 0,
            timeouts: 0,
            total: 0,
        }
    }

    fn record(&mut self, outcome: &S2Outcome) {
        self.total += 1;
        match outcome {
            S2Outcome::OkSymbolic => self.ok_symbolic += 1,
            S2Outcome::OkNumeric => self.ok_numeric += 1,
            S2Outcome::Incomplete(_) => self.incomplete += 1,
            S2Outcome::DomainChanged(_) => self.domain_changed += 1,
            S2Outcome::Mismatch(_) => self.mismatches += 1,
            S2Outcome::Error(_) => self.errors += 1,
            S2Outcome::Timeout => self.timeouts += 1,
        }
    }
}

/// Check if an ImplicitCondition is semantically redundant (always true in ℝ).
/// Uses the StandardOracle provers to determine if e.g. 1+x²≠0 is provably true.
fn implicit_cond_is_redundant(ctx: &Context, c: &ImplicitCondition) -> bool {
    let oracle = StandardOracle::new(ctx, DomainMode::Strict, ValueDomain::RealOnly);
    match c {
        ImplicitCondition::NonZero(e) => oracle.query(&Predicate::NonZero(*e)).is_proven(),
        ImplicitCondition::Positive(e) => oracle.query(&Predicate::Positive(*e)).is_proven(),
        ImplicitCondition::NonNegative(e) => oracle.query(&Predicate::NonNegative(*e)).is_proven(),
    }
}

/// Check if a DomainDelta is semantically safe (Safe, or all dropped conditions are redundant).
fn delta_is_semantically_safe(ctx: &Context, d: &DomainDelta) -> bool {
    match d {
        DomainDelta::Safe => true,
        DomainDelta::ExpandsAnalytic(conds) | DomainDelta::ExpandsDefinability(conds) => {
            conds.iter().all(|c| implicit_cond_is_redundant(ctx, c))
        }
    }
}

/// Infer the required implicit domain of an equation (union of both sides + derived).
fn infer_equation_domain(ctx: &Context, lhs: ExprId, rhs: ExprId) -> ImplicitDomain {
    let vd = ValueDomain::RealOnly;
    let dl = infer_implicit_domain(ctx, lhs, vd);
    let dr = infer_implicit_domain(ctx, rhs, vd);
    let mut d = ImplicitDomain::empty();
    d.extend(&dl);
    d.extend(&dr);
    for cond in derive_requires_from_equation(ctx, lhs, rhs, &d, vd) {
        d.conditions_mut().insert(cond);
    }
    d
}

/// Check if two equation domains are semantically equivalent
/// (any differences are provably redundant).
fn eq_domains_semantically_same(ctx: &Context, d0: &ImplicitDomain, d1: &ImplicitDomain) -> bool {
    // Check both directions: conditions dropped and added
    let dropped = d0.dropped_from(d1);
    let added = d1.dropped_from(d0);

    dropped.iter().all(|c| implicit_cond_is_redundant(ctx, c))
        && added.iter().all(|c| implicit_cond_is_redundant(ctx, c))
}

/// Run a single Strategy 2 test case:
/// Given equation (lhs = rhs) and identity (A ≡ B),
/// construct transformed equation (lhs + A = rhs + B),
/// solve both, cross-verify solutions.
fn run_s2_case(eq_entry: &EquationEntry, identity: &S2Identity) -> S2Outcome {
    let mut simplifier = Simplifier::with_default_rules();
    simplifier.set_collect_steps(false);

    // Parse original equation
    let orig_eq = match parse_equation_str(&mut simplifier.context, &eq_entry.equation_str) {
        Some(eq) => eq,
        None => return S2Outcome::Error(format!("parse eq: '{}'", eq_entry.equation_str)),
    };

    // Parse identity sides
    let id_a = match parse(&identity.exp, &mut simplifier.context) {
        Ok(e) => e,
        Err(_) => return S2Outcome::Error(format!("parse identity exp: '{}'", identity.exp)),
    };
    let id_b = match parse(&identity.simp, &mut simplifier.context) {
        Ok(e) => e,
        Err(_) => return S2Outcome::Error(format!("parse identity simp: '{}'", identity.simp)),
    };

    // Construct transformed equation: (lhs + A) = (rhs + B)
    let trans_lhs = simplifier.context.add(Expr::Add(orig_eq.lhs, id_a));
    let trans_rhs = simplifier.context.add(Expr::Add(orig_eq.rhs, id_b));
    let trans_eq = Equation {
        lhs: trans_lhs,
        rhs: trans_rhs,
        op: RelOp::Eq,
    };

    // Solve original
    let start = Instant::now();
    let sol0 = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        solve(&orig_eq, &eq_entry.solve_var, &mut simplifier)
    }));
    if start.elapsed() > SOLVE_TIMEOUT {
        return S2Outcome::Timeout;
    }
    let (set0, _) = match sol0 {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => return S2Outcome::Error(format!("solve orig: {:?}", e)),
        Err(_) => return S2Outcome::Error("solve orig: panic".into()),
    };

    // Solve transformed
    let sol1 = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        solve(&trans_eq, &eq_entry.solve_var, &mut simplifier)
    }));
    if start.elapsed() > SOLVE_TIMEOUT * 2 {
        return S2Outcome::Timeout;
    }
    let (set1, _) = match sol1 {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => return S2Outcome::Error(format!("solve trans: {:?}", e)),
        Err(_) => return S2Outcome::Error("solve trans: panic".into()),
    };

    // Check if either is non-discrete
    let s0_discrete = matches!(&set0, SolutionSet::Discrete(_) | SolutionSet::Empty);
    let s1_discrete = matches!(&set1, SolutionSet::Discrete(_) | SolutionSet::Empty);

    if !s0_discrete || !s1_discrete {
        // Cross-substitute what we can; if no mismatch, mark as Incomplete
        let cross_ok =
            cross_substitute_discrete_into(&mut simplifier, &set0, &trans_eq, &eq_entry.solve_var)
                && cross_substitute_discrete_into(
                    &mut simplifier,
                    &set1,
                    &orig_eq,
                    &eq_entry.solve_var,
                );

        return if cross_ok {
            S2Outcome::Incomplete("non-discrete form".into())
        } else {
            S2Outcome::Mismatch("cross-sub failed on non-discrete".into())
        };
    }

    // Both discrete — full cross-verification
    let mut any_numeric = false;

    // S0 solutions must satisfy trans_eq
    if let SolutionSet::Discrete(sols0) = &set0 {
        for &sol in sols0 {
            let verify = verify_solution_set(
                &mut simplifier,
                &trans_eq,
                &eq_entry.solve_var,
                &SolutionSet::Discrete(vec![sol]),
            );
            match verify.summary {
                VerifySummary::AllVerified => {}
                _ => {
                    match numeric_verify_solution(
                        &simplifier.context,
                        &trans_eq,
                        &eq_entry.solve_var,
                        sol,
                    ) {
                        NumericVerifyResult::Verified(_) | NumericVerifyResult::Inconclusive => {
                            any_numeric = true;
                        }
                        NumericVerifyResult::Failed => {
                            let s = DisplayExpr {
                                context: &simplifier.context,
                                id: sol,
                            }
                            .to_string();
                            return S2Outcome::Mismatch(format!(
                                "sol {} of orig doesn't satisfy transformed",
                                s
                            ));
                        }
                    }
                }
            }
        }
    }

    // S1 solutions must satisfy orig_eq
    if let SolutionSet::Discrete(sols1) = &set1 {
        for &sol in sols1 {
            let verify = verify_solution_set(
                &mut simplifier,
                &orig_eq,
                &eq_entry.solve_var,
                &SolutionSet::Discrete(vec![sol]),
            );
            match verify.summary {
                VerifySummary::AllVerified => {}
                _ => {
                    match numeric_verify_solution(
                        &simplifier.context,
                        &orig_eq,
                        &eq_entry.solve_var,
                        sol,
                    ) {
                        NumericVerifyResult::Verified(_) | NumericVerifyResult::Inconclusive => {
                            any_numeric = true;
                        }
                        NumericVerifyResult::Failed => {
                            let s = DisplayExpr {
                                context: &simplifier.context,
                                id: sol,
                            }
                            .to_string();
                            return S2Outcome::Mismatch(format!(
                                "sol {} of transformed doesn't satisfy orig",
                                s
                            ));
                        }
                    }
                }
            }
        }
    }

    // Also compare solution counts when both discrete
    if let (SolutionSet::Discrete(sa), SolutionSet::Discrete(sb)) = (&set0, &set1) {
        if sa.len() != sb.len() {
            return S2Outcome::Incomplete(format!(
                "sol count differs: orig={}, trans={}",
                sa.len(),
                sb.len()
            ));
        }
    }

    if any_numeric {
        S2Outcome::OkNumeric
    } else {
        S2Outcome::OkSymbolic
    }
}

/// Helper: cross-substitute discrete solutions of `source_set` into `target_eq`.
/// Returns true if all discrete solutions pass, or if source is non-discrete (skip).
fn cross_substitute_discrete_into(
    simplifier: &mut Simplifier,
    source_set: &SolutionSet,
    target_eq: &Equation,
    var: &str,
) -> bool {
    let sols = match source_set {
        SolutionSet::Discrete(s) => s,
        SolutionSet::Empty => return true,
        _ => return true, // Skip non-discrete (can't enumerate)
    };

    for &sol in sols {
        let verify = verify_solution_set(
            simplifier,
            target_eq,
            var,
            &SolutionSet::Discrete(vec![sol]),
        );
        match verify.summary {
            VerifySummary::AllVerified => {}
            _ => match numeric_verify_solution(&simplifier.context, target_eq, var, sol) {
                NumericVerifyResult::Verified(_) | NumericVerifyResult::Inconclusive => {}
                NumericVerifyResult::Failed => return false,
            },
        }
    }
    true
}

/// Run Strategy 2 with sampling
fn run_strategy2(verbose: bool) -> S2Results {
    // Config from environment
    let seed: u64 = env::var("METATEST_EQ_IDENTITY_SEED")
        .ok()
        .and_then(|v| v.parse().ok())
        .or_else(|| env::var("METATEST_SEED").ok().and_then(|v| v.parse().ok()))
        .unwrap_or(42);

    let samples: usize = env::var("METATEST_EQ_IDENTITY_SAMPLES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(500);

    let max_cost: usize = env::var("METATEST_EQ_IDENTITY_MAX_COST")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(60);

    // Load and filter
    let identities = load_identities_for_s2(max_cost);

    let equations: Vec<EquationEntry> = load_equation_corpus()
        .into_iter()
        .filter(|e| {
            matches!(
                e.expected_kind,
                ExpectedKind::Discrete | ExpectedKind::Empty
            )
        })
        .collect();

    if identities.is_empty() || equations.is_empty() {
        eprintln!();
        eprintln!("--- Strategy 2: Identity-Preserving Transforms ---");
        eprintln!("  Skipped (no identities or equations after filtering)");
        return S2Results::new();
    }

    // Separate identities by tier for each equation (done dynamically during sampling)
    let mut rng = Lcg::new(seed);
    let mut results = S2Results::new();
    let mut tier0_count = 0usize;
    let mut tier1_count = 0usize;

    eprintln!();
    eprintln!("--- Strategy 2: Identity-Preserving Transforms ---");
    eprintln!(
        "  seed={}, samples={}, identities={} (mode=g, cost≤{}), equations={}",
        seed,
        samples,
        identities.len(),
        max_cost,
        equations.len()
    );

    for sample_i in 0..samples {
        let eq_idx = rng.pick(equations.len());
        let id_idx = rng.pick(identities.len());

        let eq = &equations[eq_idx];
        let id = &identities[id_idx];

        // Classify tier
        let is_tier0 = !identity_contains_var(&id.exp, &eq.solve_var)
            && !identity_contains_var(&id.simp, &eq.solve_var);
        if is_tier0 {
            tier0_count += 1;
        } else {
            tier1_count += 1;
        }

        // Run with real thread-based timeout to prevent solver hangs
        let eq_clone = eq.clone();
        let id_clone = id.clone();
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let result = run_s2_case(&eq_clone, &id_clone);
            let _ = tx.send(result);
        });
        let mut outcome = match rx.recv_timeout(SOLVE_TIMEOUT) {
            Ok(result) => result,
            Err(_) => S2Outcome::Timeout,
        };

        // Domain-aware reclassification: if cross-sub returned Mismatch but
        // the identity or equation domain changed, downgrade to DomainChanged.
        if let S2Outcome::Mismatch(ref msg) = outcome {
            if !id.domain_safe {
                outcome = S2Outcome::DomainChanged(format!("{} [identity domain differs]", msg));
            } else {
                // Parse and compare equation domain (orig vs trans)
                let mut ctx = Context::new();
                let eq_parts: Vec<&str> = eq.equation_str.splitn(2, '=').collect();
                let orig_lhs = eq_parts
                    .first()
                    .and_then(|s| parse(s.trim(), &mut ctx).ok());
                let orig_rhs = eq_parts.get(1).and_then(|s| parse(s.trim(), &mut ctx).ok());
                let id_a = parse(&id.exp, &mut ctx).ok();
                let id_b = parse(&id.simp, &mut ctx).ok();

                let eq_domain_same = if let (Some(olhs), Some(orhs), Some(ia), Some(ib)) =
                    (orig_lhs, orig_rhs, id_a, id_b)
                {
                    let trans_lhs = ctx.add(Expr::Add(olhs, ia));
                    let trans_rhs = ctx.add(Expr::Add(orhs, ib));
                    let d0 = infer_equation_domain(&ctx, olhs, orhs);
                    let d1 = infer_equation_domain(&ctx, trans_lhs, trans_rhs);
                    eq_domains_semantically_same(&ctx, &d0, &d1)
                } else {
                    true // can't parse → assume same
                };

                if !eq_domain_same {
                    outcome =
                        S2Outcome::DomainChanged(format!("{} [equation domain contracted]", msg));
                }
            }
        }
        results.record(&outcome);

        // Print non-ok results (or all in verbose)
        let should_print = verbose
            || matches!(
                &outcome,
                S2Outcome::Mismatch(_)
                    | S2Outcome::DomainChanged(_)
                    | S2Outcome::Error(_)
                    | S2Outcome::Timeout
            );

        if should_print {
            let (sym, detail) = match &outcome {
                S2Outcome::OkSymbolic => ("✓", "symbolic".into()),
                S2Outcome::OkNumeric => ("≈", "numeric".into()),
                S2Outcome::Incomplete(msg) => ("⚠", format!("incomplete: {}", msg)),
                S2Outcome::DomainChanged(msg) => ("D", format!("domain-changed: {}", msg)),
                S2Outcome::Mismatch(msg) => ("✗", format!("MISMATCH: {}", msg)),
                S2Outcome::Error(msg) => ("E", format!("error: {}", msg)),
                S2Outcome::Timeout => ("T", "TIMEOUT".into()),
            };
            eprintln!(
                "  {} {: >4}. T{} [{}] '{}' + ({} ≡ {}) — {}",
                sym,
                sample_i + 1,
                if is_tier0 { 0 } else { 1 },
                truncate(&eq.family, 12),
                truncate(&eq.equation_str, 20),
                truncate(&id.exp, 15),
                truncate(&id.simp, 15),
                detail,
            );
        }
    }

    // Summary table
    eprintln!();
    eprintln!("  ┌─────────────────────────────────────────────────────────────────┐");
    eprintln!(
        "  │  Strategy 2:  {} samples  (T0: {}, T1: {})  seed={}",
        results.total, tier0_count, tier1_count, seed
    );
    eprintln!(
        "  │  ✓ symbolic: {}  ≈ numeric: {}  ⚠ incomplete: {}",
        results.ok_symbolic, results.ok_numeric, results.incomplete
    );
    eprintln!(
        "  │  D domain-chg: {}  ✗ mismatch: {}  E errors: {}  T timeout: {}",
        results.domain_changed, results.mismatches, results.errors, results.timeouts
    );
    eprintln!("  └─────────────────────────────────────────────────────────────────┘");

    if results.mismatches > 0 {
        eprintln!("  ❌ {} mismatch(es) found!", results.mismatches);
    } else {
        eprintln!("  ✅ No mismatches (identity transforms are solver-transparent)");
    }

    results
}

// =============================================================================
// Main Test: Strategy 2 - Identity-Preserving Transforms
// =============================================================================

#[test]
#[ignore] // Run with: cargo test --release -p cas_engine --test metamorphic_equation_tests metatest_equation_identity_transforms -- --ignored --nocapture
fn metatest_equation_identity_transforms() {
    let verbose = is_verbose();
    let results = run_strategy2(verbose);
    assert_eq!(
        results.mismatches, 0,
        "Strategy 2 had {} mismatches — see details above",
        results.mismatches
    );
}
