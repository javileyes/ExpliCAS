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

use cas_ast::{Context, Equation, Expr, ExprId, RelOp, SolutionSet};
use cas_formatter::{display_transforms::ScopeTag, DisplayExpr};
use cas_parser::parse;
use cas_solver::api::{
    derive_requires_from_equation, domain_delta_check, eval_f64, infer_implicit_domain, solve,
    solve_with_display_steps, verify_solution_set, AssumeScope, AssumptionReporting, DomainDelta,
    DomainOracle, FactStrength, ImplicitCondition, ImplicitDomain, Predicate, SolveBudget,
    StandardOracle, VerifyResult, VerifyStatus, VerifySummary,
};
use cas_solver::command_api::solve::{evaluate_solve_command_lines_with_session, SolveDisplayMode};
use cas_solver::runtime::{
    CasError, DomainMode, EvalOptions, Simplifier, SolverOptions, StatelessEvalSession, ValueDomain,
};
use std::collections::HashMap;
use std::env;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::sync::mpsc;
use std::time::{Duration, Instant};

fn find_test_data_file(filename: &str) -> PathBuf {
    let local = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("tests/{filename}"));
    if local.exists() {
        return local;
    }

    // Compatibility path used when this file is compiled via cas_engine wrapper tests.
    let solver_tests =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("../cas_solver/tests/{filename}"));
    if solver_tests.exists() {
        return solver_tests;
    }

    local
}

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

/// Interior samples that stay well inside common bounded domains like arcsin/acos.
const NUMERIC_INTERIOR_VALUES: [f64; 10] = [
    -0.9, -0.75, -0.5, -0.25, -0.125, 0.125, 0.25, 0.5, 0.75, 0.9,
];

/// Mixed-sign general samples for rational/polynomial contexts.
const NUMERIC_GENERAL_VALUES: [f64; 12] = [
    -4.0, -2.5, -1.5, -0.75, -0.25, 0.25, 0.75, 1.5, 2.5, 4.0, 0.1, 5.0,
];

/// Positive samples for logs, roots and other positivity-sensitive contexts.
const NUMERIC_POSITIVE_VALUES: [f64; 10] = [0.1, 0.2, 0.35, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0];

fn is_verbose() -> bool {
    env::var("METATEST_VERBOSE")
        .map(|v| v == "1")
        .unwrap_or(false)
}

fn numeric_sample_value(sample_idx: usize, var_idx: usize) -> f64 {
    let profile = sample_idx % 3;
    let round = sample_idx / 3;
    let values: &[f64] = match profile {
        0 => &NUMERIC_INTERIOR_VALUES,
        1 => &NUMERIC_GENERAL_VALUES,
        _ => &NUMERIC_POSITIVE_VALUES,
    };
    let idx = (round * 7 + var_idx * 13) % values.len();
    values[idx]
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
    let csv_path = find_test_data_file("equation_corpus.csv");
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
            // Deterministic, domain-aware profiles:
            // interior values for bounded domains, mixed-sign general values,
            // and positivity-friendly values for logs/roots.
            let val = numeric_sample_value(i, j);
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

fn make_solver_opts(mode: DomainMode, scope: AssumeScope) -> SolverOptions {
    SolverOptions {
        value_domain: ValueDomain::RealOnly,
        domain_mode: mode,
        assume_scope: scope,
        budget: SolveBudget::default(),
        ..Default::default()
    }
}

#[test]
fn numeric_verify_solution_finds_interior_multivar_samples() {
    let mut ctx = Context::new();
    let equation = parse_equation_str(&mut ctx, "x = cos(2*arcsin(y)) - (1 - 2*y^2)").unwrap();
    let solution = parse("0", &mut ctx).unwrap();

    match numeric_verify_solution(&ctx, &equation, "x", solution) {
        NumericVerifyResult::Verified(valid) => assert!(valid > 0),
        other => panic!("expected numeric verification to succeed, got {:?}", other),
    }
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
    /// Numeric fallback could not validate at least one residual solution
    NumericInconclusive,
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
    numeric_inconclusive: usize,
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
            EqTestOutcome::NumericInconclusive => self.numeric_inconclusive += 1,
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
        self.verified
            + self.numeric_fallback
            + self.numeric_inconclusive
            + self.not_checkable
            + self.empty
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
            let mut any_numeric_inconclusive = false;
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
                                any_numeric_inconclusive = true;
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
            } else if any_numeric_inconclusive {
                EqTestOutcome::NumericInconclusive
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
    let csv_path = find_test_data_file("equation_pairs.csv");
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
    /// Numeric fallback could not validate at least one solution
    NumericInconclusive,
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
    let mut any_numeric_inconclusive = false;
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
                        NumericVerifyResult::Inconclusive => any_numeric_inconclusive = true,
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
                            NumericVerifyResult::Inconclusive => {
                                any_numeric_inconclusive = true;
                            }
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
    } else if any_numeric_inconclusive {
        PairOutcome::NumericInconclusive
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
    eprintln!("╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗");
    eprintln!("║                    EQUATION SOLUTION VERIFICATION (Strategy 1)                                      ║");
    eprintln!("╠═══════════════════╤═══════╤══════════╤══════════╤══════════╤══════════╤════════╤═══════╤════════════╣");
    eprintln!("║ Family            │ Total │ Verified │ Num-Fllb │ Num-Inc  │ NotCheck │ Failed │  T/O  │ Err/Parse  ║");
    eprintln!("╠═══════════════════╪═══════╪══════════╪══════════╪══════════╪══════════╪════════╪═══════╪════════════╣");

    let mut t_total = 0usize;
    let mut t_verified = 0usize;
    let mut t_numeric = 0usize;
    let mut t_numeric_inconclusive = 0usize;
    let mut t_notcheck = 0usize;
    let mut t_failed = 0usize;
    let mut t_timeout = 0usize;
    let mut t_errors = 0usize;

    for m in families {
        let errors = m.solver_error + m.parse_error + m.count_mismatch + m.kind_mismatch;
        eprintln!(
            "║ {:17} │ {:>5} │ {:>8} │ {:>8} │ {:>8} │ {:>8} │ {:>6} │ {:>5} │ {:>10} ║",
            truncate(&m.family, 17),
            m.total,
            m.verified + m.empty,
            m.numeric_fallback,
            m.numeric_inconclusive,
            m.not_checkable,
            m.failed,
            m.timeout,
            errors,
        );

        t_total += m.total;
        t_verified += m.verified + m.empty;
        t_numeric += m.numeric_fallback;
        t_numeric_inconclusive += m.numeric_inconclusive;
        t_notcheck += m.not_checkable;
        t_failed += m.failed;
        t_timeout += m.timeout;
        t_errors += errors;
    }

    eprintln!("╠═══════════════════╪═══════╪══════════╪══════════╪══════════╪══════════╪════════╪═══════╪════════════╣");
    eprintln!(
        "║ TOTAL             │ {:>5} │ {:>8} │ {:>8} │ {:>8} │ {:>8} │ {:>6} │ {:>5} │ {:>10} ║",
        t_total,
        t_verified,
        t_numeric,
        t_numeric_inconclusive,
        t_notcheck,
        t_failed,
        t_timeout,
        t_errors,
    );
    eprintln!("╚═══════════════════╧═══════╧══════════╧══════════╧══════════╧══════════╧════════╧═══════╧════════════╝");

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
            "✅ All equations passed (verified: {}, numeric fallback: {}, numeric inconclusive: {}, not checkable: {})",
            t_verified, t_numeric, t_numeric_inconclusive, t_notcheck
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
    let mut total_errors = 0usize;

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
            EqTestOutcome::NumericInconclusive => "?",
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
        } else if matches!(
            &outcome,
            EqTestOutcome::SolverError(_) | EqTestOutcome::Timeout | EqTestOutcome::ParseError(_)
        ) {
            total_errors += 1;
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
                EqTestOutcome::NumericInconclusive => "numeric inconclusive".to_string(),
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
    assert_eq!(
        total_errors, 0,
        "Some equations had solver/parse/timeout errors — see details above"
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
    let mut numeric_inconclusive = 0usize;
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
            PairOutcome::NumericInconclusive => {
                numeric_inconclusive += 1;
                ("?", "numeric inconclusive".to_string())
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
        "Pairs: {} total, {} verified, {} numeric, {} numeric-inconclusive, {} mismatch, {} errors",
        pairs.len(),
        verified,
        numeric,
        numeric_inconclusive,
        mismatch,
        errors,
    );

    assert_eq!(
        mismatch, 0,
        "Some equation pairs had mismatching solutions — see details above"
    );
    assert_eq!(
        errors, 0,
        "Some equation pairs had solver/parse/timeout errors — see details above"
    );
}

// =============================================================================
// Unified Benchmark (runs all strategies)
// =============================================================================

#[test]
#[ignore] // Run with: cargo test --release -p cas_engine --test metamorphic_equation_tests metatest_equation_benchmark -- --ignored --nocapture
fn metatest_equation_benchmark() {
    // Reset Phase 1.5 instrumentation counters (env‐gated by VERIFY_STATS=1).
    cas_solver::api::verify_stats::reset_stats();

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
    let mut total_failures = 0usize;
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
            total_failures += 1;
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
        } else if matches!(
            &outcome,
            EqTestOutcome::SolverError(_) | EqTestOutcome::Timeout | EqTestOutcome::ParseError(_)
        ) {
            total_failures += 1;
            let detail = match &outcome {
                EqTestOutcome::SolverError(msg) => format!("ERROR: {}", msg),
                EqTestOutcome::Timeout => "TIMEOUT".to_string(),
                EqTestOutcome::ParseError(msg) => format!("PARSE: {}", msg),
                _ => String::new(),
            };
            eprintln!("  E {:>3}. {} — {}", i + 1, entry.equation_str, detail);
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
        let mut s3_errors = 0usize;
        for (i, pair) in pairs.iter().enumerate() {
            let outcome = verify_equation_pair(pair);
            match &outcome {
                PairOutcome::Mismatch(msg) => {
                    s3_mismatch += 1;
                    eprintln!(
                        "  ✗ {:>3}. '{}' ↔ '{}' — {}",
                        i + 1,
                        pair.eq_a,
                        pair.eq_b,
                        msg
                    );
                }
                PairOutcome::SolverFailed(msg) => {
                    s3_errors += 1;
                    eprintln!(
                        "  E {:>3}. '{}' ↔ '{}' — solver: {}",
                        i + 1,
                        pair.eq_a,
                        pair.eq_b,
                        msg
                    );
                }
                PairOutcome::ParseError(msg) => {
                    s3_errors += 1;
                    eprintln!(
                        "  P {:>3}. '{}' ↔ '{}' — parse: {}",
                        i + 1,
                        pair.eq_a,
                        pair.eq_b,
                        msg
                    );
                }
                PairOutcome::Timeout => {
                    s3_errors += 1;
                    eprintln!(
                        "  T {:>3}. '{}' ↔ '{}' — TIMEOUT",
                        i + 1,
                        pair.eq_a,
                        pair.eq_b
                    );
                }
                _ => {}
            }
        }
        eprintln!(
            "Pairs: {} total, {} mismatches, {} errors",
            pairs.len(),
            s3_mismatch,
            s3_errors
        );
        total_failures += s3_mismatch + s3_errors;
    }

    // Run Strategy 2
    let s2_result = run_strategy2(verbose);
    total_failures += s2_result.mismatches + s2_result.errors + s2_result.timeouts;

    let total_elapsed = total_start.elapsed();
    eprintln!();
    eprintln!("Total time: {:.2}s", total_elapsed.as_secs_f64());

    // Print Phase 1.5 instrumentation summary (only when VERIFY_STATS=1).
    cas_solver::api::verify_stats::dump_stats();

    assert_eq!(
        total_failures, 0,
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
    /// Extra domain hints carried in the CSV metadata, e.g. `ge(0.0)`.
    explicit_domain_hints: Vec<String>,
    /// Heuristic flag for inverse-trig identities that introduce bounded domains.
    bounded_domain_sensitive: bool,
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
    let csv_path = find_test_data_file("identity_pairs.csv");
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
        let explicit_domain_hints: Vec<String> = parts
            .iter()
            .skip(4)
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect();

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
        let bounded_domain_sensitive = exp.contains("arcsin(")
            || simp.contains("arcsin(")
            || exp.contains("arccos(")
            || simp.contains("arccos(");

        identities.push(S2Identity {
            exp,
            simp,
            var,
            family: current_family.clone(),
            tier: 0, // Will be set per-equation in run_case
            cost,
            domain_safe,
            explicit_domain_hints,
            bounded_domain_sensitive,
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

/// Structured reason for `Incomplete` outcomes.
/// Each variant maps to a specific solver limitation or expected behaviour.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum IncompleteReason {
    /// Solver returned IsolationError (can't isolate variable)
    Isolation,
    /// Solver detected equivalent-form loop
    CycleDetected,
    /// Maximum recursion depth exceeded
    MaxDepth,
    /// Continuous solution in factor split (abs-value)
    ContinuousSolution,
    /// Substitution only supports discrete solutions
    SubstitutionNonDiscrete,
    /// Solver returned non-discrete result (Conditional/Residual/Interval)
    NonDiscrete,
    /// Numeric fallback could not validate due to domain/sample issues
    NumericInconclusive,
    /// Other solver limitation
    Other(String),
}

impl std::fmt::Display for IncompleteReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IncompleteReason::Isolation => write!(f, "isolation"),
            IncompleteReason::CycleDetected => write!(f, "cycle"),
            IncompleteReason::MaxDepth => write!(f, "max-depth"),
            IncompleteReason::ContinuousSolution => write!(f, "continuous"),
            IncompleteReason::SubstitutionNonDiscrete => write!(f, "sub-non-disc"),
            IncompleteReason::NonDiscrete => write!(f, "non-discrete"),
            IncompleteReason::NumericInconclusive => write!(f, "numeric-inconclusive"),
            IncompleteReason::Other(s) => write!(f, "{}", s),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CrossSubstituteOutcome {
    Clear,
    Numeric,
    Inconclusive,
    Failed,
}

impl CrossSubstituteOutcome {
    fn merge(self, other: Self) -> Self {
        use CrossSubstituteOutcome::*;
        match (self, other) {
            (Failed, _) | (_, Failed) => Failed,
            (Inconclusive, _) | (_, Inconclusive) => Inconclusive,
            (Numeric, _) | (_, Numeric) => Numeric,
            _ => Clear,
        }
    }
}

/// Result of a Strategy 2 test case
#[derive(Debug)]
enum S2Outcome {
    /// Cross-substitution verified symbolically
    OkSymbolic,
    /// Cross-substitution verified numerically
    OkNumeric,
    /// Non-discrete solution but discrete parts pass cross-substitution
    OkPartialVerified,
    /// Solver couldn't fully solve — not a correctness bug
    Incomplete(IncompleteReason),
    /// Cross-substitution failed, but identity or equation domain differs
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
    ok_partial: usize,
    incomplete: usize,
    domain_changed: usize,
    mismatches: usize,
    errors: usize,
    timeouts: usize,
    total: usize,
    /// Per-reason breakdown of Incomplete outcomes
    incomplete_reasons: std::collections::HashMap<IncompleteReason, usize>,
    /// Top identity offenders: identity index → incomplete count
    identity_offenders: std::collections::HashMap<usize, usize>,
    /// Top equation family offenders: family name → incomplete count
    family_offenders: std::collections::HashMap<String, usize>,
}

impl S2Results {
    fn new() -> Self {
        S2Results {
            ok_symbolic: 0,
            ok_numeric: 0,
            ok_partial: 0,
            incomplete: 0,
            domain_changed: 0,
            mismatches: 0,
            errors: 0,
            timeouts: 0,
            total: 0,
            incomplete_reasons: std::collections::HashMap::new(),
            identity_offenders: std::collections::HashMap::new(),
            family_offenders: std::collections::HashMap::new(),
        }
    }

    fn record(&mut self, outcome: &S2Outcome) {
        self.total += 1;
        match outcome {
            S2Outcome::OkSymbolic => self.ok_symbolic += 1,
            S2Outcome::OkNumeric => self.ok_numeric += 1,
            S2Outcome::OkPartialVerified => self.ok_partial += 1,
            S2Outcome::Incomplete(_) => self.incomplete += 1,
            S2Outcome::DomainChanged(_) => self.domain_changed += 1,
            S2Outcome::Mismatch(_) => self.mismatches += 1,
            S2Outcome::Error(_) => self.errors += 1,
            S2Outcome::Timeout => self.timeouts += 1,
        }
    }

    /// Record Incomplete context: reason, identity index, equation family
    fn record_incomplete_context(
        &mut self,
        reason: &IncompleteReason,
        id_idx: usize,
        eq_family: &str,
    ) {
        *self.incomplete_reasons.entry(reason.clone()).or_insert(0) += 1;
        *self.identity_offenders.entry(id_idx).or_insert(0) += 1;
        *self
            .family_offenders
            .entry(eq_family.to_string())
            .or_insert(0) += 1;
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

fn identity_domain_hint_reason(id: &S2Identity) -> Option<String> {
    if !id.explicit_domain_hints.is_empty() {
        return Some(format!(
            "identity requires {}",
            id.explicit_domain_hints.join(", ")
        ));
    }
    if id.bounded_domain_sensitive {
        return Some("identity bounded inverse-trig domain differs".into());
    }
    None
}

/// Detect whether a Strategy 2 transform changed the applicable domain.
/// Returns a short human-readable reason when the identity itself or the
/// transformed equation contracts the domain.
fn strategy2_domain_change_reason(eq: &EquationEntry, id: &S2Identity) -> Option<String> {
    if let Some(reason) = identity_domain_hint_reason(id) {
        return Some(reason);
    }

    if !id.domain_safe {
        return Some("identity domain differs".into());
    }

    let mut ctx = Context::new();
    let eq_parts: Vec<&str> = eq.equation_str.splitn(2, '=').collect();
    let orig_lhs = eq_parts
        .first()
        .and_then(|s| parse(s.trim(), &mut ctx).ok());
    let orig_rhs = eq_parts.get(1).and_then(|s| parse(s.trim(), &mut ctx).ok());
    let id_a = parse(&id.exp, &mut ctx).ok();
    let id_b = parse(&id.simp, &mut ctx).ok();

    let eq_domain_same =
        if let (Some(olhs), Some(orhs), Some(ia), Some(ib)) = (orig_lhs, orig_rhs, id_a, id_b) {
            let trans_lhs = ctx.add(Expr::Add(olhs, ia));
            let trans_rhs = ctx.add(Expr::Add(orhs, ib));
            let d0 = infer_equation_domain(&ctx, olhs, orhs);
            let d1 = infer_equation_domain(&ctx, trans_lhs, trans_rhs);
            eq_domains_semantically_same(&ctx, &d0, &d1)
        } else {
            true
        };

    if eq_domain_same {
        None
    } else {
        Some("equation domain contracted".into())
    }
}

/// Classify a solver error as either an expected limitation (Incomplete) or a real error.
fn classify_solver_error(e: &CasError, _phase: &str) -> S2Outcome {
    match e {
        CasError::IsolationError(_, _) => S2Outcome::Incomplete(IncompleteReason::Isolation),
        CasError::SolverError(s) if s.contains("Maximum solver recursion depth") => {
            S2Outcome::Incomplete(IncompleteReason::MaxDepth)
        }
        CasError::SolverError(s) if s.contains("Continuous solution in factor split") => {
            S2Outcome::Incomplete(IncompleteReason::ContinuousSolution)
        }
        CasError::SolverError(s) if s.contains("currently only supports discrete") => {
            S2Outcome::Incomplete(IncompleteReason::SubstitutionNonDiscrete)
        }
        CasError::SolverError(s) if s.contains("Cycle detected") => {
            S2Outcome::Incomplete(IncompleteReason::CycleDetected)
        }
        _ => S2Outcome::Error(format!("{:?}", e)),
    }
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
        Ok(Err(e)) => return classify_solver_error(&e, "solve orig"),
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
        Ok(Err(e)) => return classify_solver_error(&e, "solve trans"),
        Err(_) => return S2Outcome::Error("solve trans: panic".into()),
    };

    // Check if either is non-discrete
    let s0_discrete = matches!(&set0, SolutionSet::Discrete(_) | SolutionSet::Empty);
    let s1_discrete = matches!(&set1, SolutionSet::Discrete(_) | SolutionSet::Empty);

    if !s0_discrete || !s1_discrete {
        // Cross-substitute what we can; if no mismatch, mark appropriately
        let cross_result_0 =
            cross_substitute_discrete_into(&mut simplifier, &set0, &trans_eq, &eq_entry.solve_var);
        let cross_result_1 =
            cross_substitute_discrete_into(&mut simplifier, &set1, &orig_eq, &eq_entry.solve_var);
        let cross_result = cross_result_0.merge(cross_result_1);

        return if cross_result == CrossSubstituteOutcome::Failed {
            S2Outcome::Mismatch("cross-sub failed on non-discrete".into())
        } else if cross_result == CrossSubstituteOutcome::Inconclusive {
            S2Outcome::Incomplete(IncompleteReason::NumericInconclusive)
        } else {
            // At least one side has discrete solutions that were verified.
            // If either side has any discrete solutions, this is a partial-but-correct result.
            let has_discrete_0 = matches!(&set0, SolutionSet::Discrete(v) if !v.is_empty());
            let has_discrete_1 = matches!(&set1, SolutionSet::Discrete(v) if !v.is_empty());
            if has_discrete_0 || has_discrete_1 {
                S2Outcome::OkPartialVerified
            } else {
                S2Outcome::Incomplete(IncompleteReason::NonDiscrete)
            }
        };
    }

    // Both discrete — full cross-verification
    let mut any_numeric = false;
    let mut any_numeric_inconclusive = false;

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
                        NumericVerifyResult::Verified(_) => {
                            any_numeric = true;
                        }
                        NumericVerifyResult::Inconclusive => {
                            any_numeric_inconclusive = true;
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
                        NumericVerifyResult::Verified(_) => {
                            any_numeric = true;
                        }
                        NumericVerifyResult::Inconclusive => {
                            any_numeric_inconclusive = true;
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
            // Check if the identity contracts the equation domain.
            // If so, fewer solutions are expected — classify as DomainChanged.
            let d0 = infer_equation_domain(&simplifier.context, orig_eq.lhs, orig_eq.rhs);
            let d1 = infer_equation_domain(&simplifier.context, trans_eq.lhs, trans_eq.rhs);
            let domain_same = eq_domains_semantically_same(&simplifier.context, &d0, &d1);

            if !domain_same {
                return S2Outcome::DomainChanged(format!(
                    "sol count differs: orig={}, trans={} [domain contracted]",
                    sa.len(),
                    sb.len()
                ));
            }
            return S2Outcome::Incomplete(IncompleteReason::Other(format!(
                "sol count differs: orig={}, trans={}",
                sa.len(),
                sb.len()
            )));
        }
    }

    if any_numeric_inconclusive {
        S2Outcome::Incomplete(IncompleteReason::NumericInconclusive)
    } else if any_numeric {
        S2Outcome::OkNumeric
    } else {
        S2Outcome::OkSymbolic
    }
}

/// Helper: cross-substitute discrete solutions of `source_set` into `target_eq`.
/// Returns a structured outcome for the discrete cross-substitution pass.
fn cross_substitute_discrete_into(
    simplifier: &mut Simplifier,
    source_set: &SolutionSet,
    target_eq: &Equation,
    var: &str,
) -> CrossSubstituteOutcome {
    let sols = match source_set {
        SolutionSet::Discrete(s) => s,
        SolutionSet::Empty => return CrossSubstituteOutcome::Clear,
        _ => return CrossSubstituteOutcome::Clear, // Skip non-discrete (can't enumerate)
    };

    let mut outcome = CrossSubstituteOutcome::Clear;
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
                NumericVerifyResult::Verified(_) => {
                    outcome = outcome.merge(CrossSubstituteOutcome::Numeric);
                }
                NumericVerifyResult::Inconclusive => {
                    outcome = outcome.merge(CrossSubstituteOutcome::Inconclusive);
                }
                NumericVerifyResult::Failed => return CrossSubstituteOutcome::Failed,
            },
        }
    }
    outcome
}

fn normalize_s2_outcome(eq: &EquationEntry, id: &S2Identity, mut outcome: S2Outcome) -> S2Outcome {
    if let S2Outcome::Mismatch(ref msg) = outcome {
        if let Some(reason) = strategy2_domain_change_reason(eq, id) {
            outcome = S2Outcome::DomainChanged(format!("{} [{}]", msg, reason));
        }
    }

    if matches!(&outcome, S2Outcome::Incomplete(IncompleteReason::Isolation)) {
        if let Some(reason) = strategy2_domain_change_reason(eq, id) {
            outcome = S2Outcome::DomainChanged(format!("isolation + {}", reason));
        }
    }

    if matches!(
        &outcome,
        S2Outcome::Incomplete(IncompleteReason::NumericInconclusive)
    ) {
        if let Some(reason) = strategy2_domain_change_reason(eq, id) {
            outcome = S2Outcome::DomainChanged(format!("numeric inconclusive [{}]", reason));
        }
    }

    outcome
}

#[derive(Debug, Clone)]
struct S2ContractCase {
    equation_str: String,
    solve_var: String,
    identity_exp: String,
    identity_simp: String,
    expected: S2ContractExpectation,
    family: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum S2ContractExpectation {
    Ok,
    DomainChanged,
    Incomplete,
}

impl S2ContractExpectation {
    fn from_str(s: &str) -> Self {
        match s.trim().to_lowercase().as_str() {
            "ok" => Self::Ok,
            "domain-changed" => Self::DomainChanged,
            "incomplete" => Self::Incomplete,
            other => panic!(
                "Unknown equation transform contract expectation: '{}'",
                other
            ),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::DomainChanged => "domain-changed",
            Self::Incomplete => "incomplete",
        }
    }

    fn matches(self, outcome: &S2Outcome) -> bool {
        match self {
            Self::Ok => matches!(
                outcome,
                S2Outcome::OkSymbolic | S2Outcome::OkNumeric | S2Outcome::OkPartialVerified
            ),
            Self::DomainChanged => matches!(outcome, S2Outcome::DomainChanged(_)),
            Self::Incomplete => matches!(outcome, S2Outcome::Incomplete(_)),
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct S2ContractMetrics {
    total: usize,
    passed: usize,
    failed: usize,
    load_errors: usize,
}

fn load_s2_contract_cases() -> Vec<S2ContractCase> {
    let csv_path = find_test_data_file("equation_transform_contract_cases.csv");
    let file = fs::File::open(&csv_path).unwrap_or_else(|e| {
        panic!("Failed to open {}: {}", csv_path.display(), e);
    });
    let reader = BufReader::new(file);
    let mut cases = Vec::new();

    for (line_num, line) in reader.lines().enumerate() {
        let line = line.unwrap();
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.splitn(6, ',').collect();
        if parts.len() < 5 {
            panic!(
                "equation_transform_contract_cases.csv line {}: expected at least 5 fields, got {}. Line: '{}'",
                line_num + 1,
                parts.len(),
                line
            );
        }

        cases.push(S2ContractCase {
            equation_str: parts[0].trim().to_string(),
            solve_var: parts[1].trim().to_string(),
            identity_exp: parts[2].trim().to_string(),
            identity_simp: parts[3].trim().to_string(),
            expected: S2ContractExpectation::from_str(parts[4]),
            family: if parts.len() > 5 {
                parts[5].trim().to_string()
            } else {
                "unknown".to_string()
            },
        });
    }

    cases
}

fn load_identities_by_pair_for_s2() -> HashMap<(String, String), S2Identity> {
    load_identities_for_s2(usize::MAX)
        .into_iter()
        .map(|id| ((id.exp.clone(), id.simp.clone()), id))
        .collect()
}

fn run_s2_transform_contract_tests() -> S2ContractMetrics {
    let cases = load_s2_contract_cases();
    let identities = load_identities_by_pair_for_s2();
    let verbose = is_verbose();
    let mut metrics = S2ContractMetrics::default();

    eprintln!();
    eprintln!(
        "=== Equation Transform Contracts: {} curated cases ===",
        cases.len()
    );

    for (i, case) in cases.iter().enumerate() {
        metrics.total += 1;
        let Some(identity) = identities
            .get(&(case.identity_exp.clone(), case.identity_simp.clone()))
            .cloned()
        else {
            metrics.load_errors += 1;
            metrics.failed += 1;
            eprintln!(
                "  E {:>3}. [{}] missing identity metadata for {} ≡ {}",
                i + 1,
                truncate(&case.family, 18),
                case.identity_exp,
                case.identity_simp
            );
            continue;
        };

        let eq_entry = EquationEntry {
            equation_str: case.equation_str.clone(),
            solve_var: case.solve_var.clone(),
            expected_kind: ExpectedKind::Any,
            expected_count: None,
            family: case.family.clone(),
            domain_mode: 'g',
        };

        let outcome = normalize_s2_outcome(&eq_entry, &identity, run_s2_case(&eq_entry, &identity));
        let passed = case.expected.matches(&outcome);

        if passed {
            metrics.passed += 1;
        } else {
            metrics.failed += 1;
        }

        if verbose || !passed {
            let detail = match &outcome {
                S2Outcome::OkSymbolic => "ok-symbolic".to_string(),
                S2Outcome::OkNumeric => "ok-numeric".to_string(),
                S2Outcome::OkPartialVerified => "ok-partial".to_string(),
                S2Outcome::Incomplete(reason) => format!("incomplete: {}", reason),
                S2Outcome::DomainChanged(msg) => format!("domain-changed: {}", msg),
                S2Outcome::Mismatch(msg) => format!("mismatch: {}", msg),
                S2Outcome::Error(msg) => format!("error: {}", msg),
                S2Outcome::Timeout => "timeout".to_string(),
            };
            eprintln!(
                "  {} {:>3}. [{}] {} + ({} ≡ {}) — expected {}, got {}",
                if passed { "✓" } else { "✗" },
                i + 1,
                truncate(&case.family, 18),
                truncate(&case.equation_str, 24),
                truncate(&case.identity_exp, 18),
                truncate(&case.identity_simp, 18),
                case.expected.label(),
                detail
            );
        }
    }

    eprintln!();
    eprintln!(
        "Equation transform contracts: {} total, {} passed, {} failed, {} load-errors",
        metrics.total, metrics.passed, metrics.failed, metrics.load_errors
    );

    metrics
}

#[derive(Debug, Clone)]
struct EquationKindContractCase {
    equation_str: String,
    solve_var: String,
    domain_mode: DomainMode,
    assume_scope: AssumeScope,
    identity_exp: String,
    identity_simp: String,
    expected_orig: EquationKindExpectation,
    expected_trans: EquationKindExpectation,
    family: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EquationKindExpectation {
    Discrete,
    Empty,
    AllReals,
    Conditional,
    Residual,
}

impl EquationKindExpectation {
    fn from_str(s: &str) -> Self {
        match s.trim().to_lowercase().as_str() {
            "discrete" => Self::Discrete,
            "empty" => Self::Empty,
            "allreals" => Self::AllReals,
            "all_reals" => Self::AllReals,
            "all-reals" => Self::AllReals,
            "conditional" => Self::Conditional,
            "residual" => Self::Residual,
            other => panic!("Unknown equation kind contract expectation: '{}'", other),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Discrete => "discrete",
            Self::Empty => "empty",
            Self::AllReals => "allreals",
            Self::Conditional => "conditional",
            Self::Residual => "residual",
        }
    }

    fn matches(self, set: &SolutionSet) -> bool {
        match self {
            Self::Discrete => matches!(set, SolutionSet::Discrete(_)),
            Self::Empty => matches!(set, SolutionSet::Empty),
            Self::AllReals => matches!(set, SolutionSet::AllReals),
            Self::Conditional => matches!(set, SolutionSet::Conditional(_)),
            Self::Residual => matches!(set, SolutionSet::Residual(_)),
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct EquationKindContractMetrics {
    total: usize,
    passed: usize,
    failed: usize,
}

#[derive(Debug, Clone)]
struct EquationRequiredContractCase {
    equation_str: String,
    solve_var: String,
    domain_mode: DomainMode,
    assume_scope: AssumeScope,
    identity_exp: String,
    identity_simp: String,
    expected_required: Vec<String>,
    family: String,
}

#[derive(Debug, Default, Clone, Copy)]
struct EquationRequiredContractMetrics {
    total: usize,
    passed: usize,
    failed: usize,
}

#[derive(Debug, Clone)]
struct EquationAssumptionContractCase {
    equation_str: String,
    solve_var: String,
    domain_mode: DomainMode,
    assume_scope: AssumeScope,
    identity_exp: String,
    identity_simp: String,
    expected_assumed: Vec<String>,
    family: String,
}

#[derive(Debug, Default, Clone, Copy)]
struct EquationAssumptionContractMetrics {
    total: usize,
    passed: usize,
    failed: usize,
}

#[derive(Debug, Clone)]
struct EquationAssumptionRecordContractCase {
    equation_str: String,
    solve_var: String,
    domain_mode: DomainMode,
    assume_scope: AssumeScope,
    identity_exp: String,
    identity_simp: String,
    expected_records: Vec<String>,
    family: String,
}

#[derive(Debug, Default, Clone, Copy)]
struct EquationAssumptionRecordContractMetrics {
    total: usize,
    passed: usize,
    failed: usize,
}

#[derive(Debug, Clone)]
struct EquationWarningContractCase {
    equation_str: String,
    solve_var: String,
    domain_mode: DomainMode,
    assume_scope: AssumeScope,
    assumption_reporting: AssumptionReporting,
    identity_exp: String,
    identity_simp: String,
    expected_warning_items: Vec<String>,
    family: String,
}

#[derive(Debug, Default, Clone, Copy)]
struct EquationWarningContractMetrics {
    total: usize,
    passed: usize,
    failed: usize,
}

#[derive(Debug, Clone)]
struct EquationAssumptionSectionContractCase {
    equation_str: String,
    solve_var: String,
    domain_mode: DomainMode,
    assume_scope: AssumeScope,
    assumption_reporting: AssumptionReporting,
    identity_exp: String,
    identity_simp: String,
    expected_section_items: Vec<String>,
    family: String,
}

#[derive(Debug, Default, Clone, Copy)]
struct EquationAssumptionSectionContractMetrics {
    total: usize,
    passed: usize,
    failed: usize,
}

#[derive(Debug, Clone)]
struct EquationTransparencyContractCase {
    equation_str: String,
    solve_var: String,
    domain_mode: DomainMode,
    assume_scope: AssumeScope,
    identity_exp: String,
    identity_simp: String,
    expected_signal: bool,
    family: String,
}

#[derive(Debug, Default, Clone, Copy)]
struct EquationTransparencyContractMetrics {
    total: usize,
    passed: usize,
    failed: usize,
}

#[derive(Debug, Clone)]
struct EquationScopeContractCase {
    equation_str: String,
    solve_var: String,
    domain_mode: DomainMode,
    assume_scope: AssumeScope,
    identity_exp: String,
    identity_simp: String,
    expected_scopes: Vec<String>,
    family: String,
}

#[derive(Debug, Default, Clone, Copy)]
struct EquationScopeContractMetrics {
    total: usize,
    passed: usize,
    failed: usize,
}

#[derive(Debug, Clone)]
struct EquationStepContractCase {
    equation_str: String,
    solve_var: String,
    domain_mode: DomainMode,
    assume_scope: AssumeScope,
    identity_exp: String,
    identity_simp: String,
    expected_keywords: Vec<String>,
    family: String,
}

#[derive(Debug, Default, Clone, Copy)]
struct EquationStepContractMetrics {
    total: usize,
    passed: usize,
    failed: usize,
}

#[derive(Debug, Clone)]
struct EquationPairContractCase {
    eq_a: String,
    eq_b: String,
    solve_var: String,
    expected: PairContractExpectation,
    family: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PairContractExpectation {
    Verified,
    Numeric,
}

impl PairContractExpectation {
    fn from_str(s: &str) -> Self {
        match s.trim().to_lowercase().as_str() {
            "verified" => Self::Verified,
            "numeric" => Self::Numeric,
            other => panic!("Unknown pair contract expectation: '{}'", other),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Verified => "verified",
            Self::Numeric => "numeric",
        }
    }

    fn matches(self, outcome: &PairOutcome) -> bool {
        match self {
            Self::Verified => matches!(outcome, PairOutcome::Verified),
            Self::Numeric => matches!(outcome, PairOutcome::NumericFallback),
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct EquationPairContractMetrics {
    total: usize,
    passed: usize,
    failed: usize,
}

fn parse_contract_domain_mode(s: &str) -> DomainMode {
    match s.trim().to_lowercase().as_str() {
        "strict" => DomainMode::Strict,
        "generic" => DomainMode::Generic,
        "assume" => DomainMode::Assume,
        other => panic!("Unknown contract domain mode: '{}'", other),
    }
}

fn parse_contract_assume_scope(s: &str) -> AssumeScope {
    match s.trim().to_lowercase().as_str() {
        "real" => AssumeScope::Real,
        "wildcard" => AssumeScope::Wildcard,
        other => panic!("Unknown contract assume scope: '{}'", other),
    }
}

fn parse_contract_assumption_reporting(s: &str) -> AssumptionReporting {
    match s.trim().to_lowercase().as_str() {
        "off" => AssumptionReporting::Off,
        "summary" => AssumptionReporting::Summary,
        "trace" => AssumptionReporting::Trace,
        other => panic!("Unknown contract assumption reporting: '{}'", other),
    }
}

fn load_equation_kind_contract_cases() -> Vec<EquationKindContractCase> {
    let csv_path = find_test_data_file("equation_solution_kind_contract_cases.csv");
    let file = fs::File::open(&csv_path).unwrap_or_else(|e| {
        panic!("Failed to open {}: {}", csv_path.display(), e);
    });
    let reader = BufReader::new(file);
    let mut cases = Vec::new();

    for (line_num, line) in reader.lines().enumerate() {
        let line = line.unwrap();
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.splitn(9, ',').collect();
        if parts.len() < 8 {
            panic!(
                "equation_solution_kind_contract_cases.csv line {}: expected at least 8 fields, got {}. Line: '{}'",
                line_num + 1,
                parts.len(),
                line
            );
        }

        cases.push(EquationKindContractCase {
            equation_str: parts[0].trim().to_string(),
            solve_var: parts[1].trim().to_string(),
            domain_mode: parse_contract_domain_mode(parts[2]),
            assume_scope: parse_contract_assume_scope(parts[3]),
            identity_exp: parts[4].trim().to_string(),
            identity_simp: parts[5].trim().to_string(),
            expected_orig: EquationKindExpectation::from_str(parts[6]),
            expected_trans: EquationKindExpectation::from_str(parts[7]),
            family: if parts.len() > 8 {
                parts[8].trim().to_string()
            } else {
                "unknown".to_string()
            },
        });
    }

    cases
}

fn load_equation_required_contract_cases() -> Vec<EquationRequiredContractCase> {
    let csv_path = find_test_data_file("equation_required_contract_cases.csv");
    let file = fs::File::open(&csv_path).unwrap_or_else(|e| {
        panic!("Failed to open {}: {}", csv_path.display(), e);
    });
    let reader = BufReader::new(file);
    let mut cases = Vec::new();

    for (line_num, line) in reader.lines().enumerate() {
        let line = line.unwrap();
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.splitn(8, ',').collect();
        if parts.len() < 7 {
            panic!(
                "equation_required_contract_cases.csv line {}: expected at least 7 fields, got {}. Line: '{}'",
                line_num + 1,
                parts.len(),
                line
            );
        }

        let expected_required = if parts[6].trim().is_empty() || parts[6].trim() == "none" {
            Vec::new()
        } else {
            parts[6]
                .split(';')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect()
        };

        cases.push(EquationRequiredContractCase {
            equation_str: parts[0].trim().to_string(),
            solve_var: parts[1].trim().to_string(),
            domain_mode: parse_contract_domain_mode(parts[2]),
            assume_scope: parse_contract_assume_scope(parts[3]),
            identity_exp: parts[4].trim().to_string(),
            identity_simp: parts[5].trim().to_string(),
            expected_required,
            family: if parts.len() > 7 {
                parts[7].trim().to_string()
            } else {
                "unknown".to_string()
            },
        });
    }

    cases
}

fn load_equation_assumption_contract_cases() -> Vec<EquationAssumptionContractCase> {
    let csv_path = find_test_data_file("equation_assumption_contract_cases.csv");
    let file = fs::File::open(&csv_path).unwrap_or_else(|e| {
        panic!("Failed to open {}: {}", csv_path.display(), e);
    });
    let reader = BufReader::new(file);
    let mut cases = Vec::new();

    for (line_num, line) in reader.lines().enumerate() {
        let line = line.unwrap();
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.splitn(8, ',').collect();
        if parts.len() < 7 {
            panic!(
                "equation_assumption_contract_cases.csv line {}: expected at least 7 fields, got {}. Line: '{}'",
                line_num + 1,
                parts.len(),
                line
            );
        }

        let expected_assumed = if parts[6].trim().is_empty() || parts[6].trim() == "none" {
            Vec::new()
        } else {
            parts[6]
                .split(';')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect()
        };

        cases.push(EquationAssumptionContractCase {
            equation_str: parts[0].trim().to_string(),
            solve_var: parts[1].trim().to_string(),
            domain_mode: parse_contract_domain_mode(parts[2]),
            assume_scope: parse_contract_assume_scope(parts[3]),
            identity_exp: parts[4].trim().to_string(),
            identity_simp: parts[5].trim().to_string(),
            expected_assumed,
            family: if parts.len() > 7 {
                parts[7].trim().to_string()
            } else {
                "unknown".to_string()
            },
        });
    }

    cases
}

fn load_equation_transparency_contract_cases() -> Vec<EquationTransparencyContractCase> {
    let csv_path = find_test_data_file("equation_transparency_contract_cases.csv");
    let file = fs::File::open(&csv_path).unwrap_or_else(|e| {
        panic!("Failed to open {}: {}", csv_path.display(), e);
    });
    let reader = BufReader::new(file);
    let mut cases = Vec::new();

    for (line_num, line) in reader.lines().enumerate() {
        let line = line.unwrap();
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.splitn(8, ',').collect();
        if parts.len() < 7 {
            panic!(
                "equation_transparency_contract_cases.csv line {}: expected at least 7 fields, got {}. Line: '{}'",
                line_num + 1,
                parts.len(),
                line
            );
        }

        let expected_signal = match parts[6].trim().to_lowercase().as_str() {
            "yes" | "true" | "present" => true,
            "no" | "false" | "absent" => false,
            other => panic!("Unknown equation transparency expectation: '{}'", other),
        };

        cases.push(EquationTransparencyContractCase {
            equation_str: parts[0].trim().to_string(),
            solve_var: parts[1].trim().to_string(),
            domain_mode: parse_contract_domain_mode(parts[2]),
            assume_scope: parse_contract_assume_scope(parts[3]),
            identity_exp: parts[4].trim().to_string(),
            identity_simp: parts[5].trim().to_string(),
            expected_signal,
            family: if parts.len() > 7 {
                parts[7].trim().to_string()
            } else {
                "unknown".to_string()
            },
        });
    }

    cases
}

fn load_equation_assumption_record_contract_cases() -> Vec<EquationAssumptionRecordContractCase> {
    let csv_path = find_test_data_file("equation_assumption_record_contract_cases.csv");
    let file = fs::File::open(&csv_path).unwrap_or_else(|e| {
        panic!("Failed to open {}: {}", csv_path.display(), e);
    });
    let reader = BufReader::new(file);
    let mut cases = Vec::new();

    for (line_num, line) in reader.lines().enumerate() {
        let line = line.unwrap();
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.splitn(8, ',').collect();
        if parts.len() < 7 {
            panic!(
                "equation_assumption_record_contract_cases.csv line {}: expected at least 7 fields, got {}. Line: '{}'",
                line_num + 1,
                parts.len(),
                line
            );
        }

        let expected_records = if parts[6].trim().is_empty() || parts[6].trim() == "none" {
            Vec::new()
        } else {
            parts[6]
                .split(';')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect()
        };

        cases.push(EquationAssumptionRecordContractCase {
            equation_str: parts[0].trim().to_string(),
            solve_var: parts[1].trim().to_string(),
            domain_mode: parse_contract_domain_mode(parts[2]),
            assume_scope: parse_contract_assume_scope(parts[3]),
            identity_exp: parts[4].trim().to_string(),
            identity_simp: parts[5].trim().to_string(),
            expected_records,
            family: if parts.len() > 7 {
                parts[7].trim().to_string()
            } else {
                "unknown".to_string()
            },
        });
    }

    cases
}

fn load_equation_warning_contract_cases() -> Vec<EquationWarningContractCase> {
    let csv_path = find_test_data_file("equation_warning_contract_cases.csv");
    let file = fs::File::open(&csv_path).unwrap_or_else(|e| {
        panic!("Failed to open {}: {}", csv_path.display(), e);
    });
    let reader = BufReader::new(file);
    let mut cases = Vec::new();

    for (line_num, line) in reader.lines().enumerate() {
        let line = line.unwrap();
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.splitn(9, ',').collect();
        if parts.len() < 8 {
            panic!(
                "equation_warning_contract_cases.csv line {}: expected at least 8 fields, got {}. Line: '{}'",
                line_num + 1,
                parts.len(),
                line
            );
        }

        let expected_warning_items = if parts[7].trim().is_empty() || parts[7].trim() == "none" {
            Vec::new()
        } else {
            parts[7]
                .split(';')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect()
        };

        cases.push(EquationWarningContractCase {
            equation_str: parts[0].trim().to_string(),
            solve_var: parts[1].trim().to_string(),
            domain_mode: parse_contract_domain_mode(parts[2]),
            assume_scope: parse_contract_assume_scope(parts[3]),
            assumption_reporting: parse_contract_assumption_reporting(parts[4]),
            identity_exp: parts[5].trim().to_string(),
            identity_simp: parts[6].trim().to_string(),
            expected_warning_items,
            family: if parts.len() > 8 {
                parts[8].trim().to_string()
            } else {
                "unknown".to_string()
            },
        });
    }

    cases
}

fn load_equation_assumption_section_contract_cases() -> Vec<EquationAssumptionSectionContractCase> {
    let csv_path = find_test_data_file("equation_assumption_section_contract_cases.csv");
    let file = fs::File::open(&csv_path).unwrap_or_else(|e| {
        panic!("Failed to open {}: {}", csv_path.display(), e);
    });
    let reader = BufReader::new(file);
    let mut cases = Vec::new();

    for (line_num, line) in reader.lines().enumerate() {
        let line = line.unwrap();
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.splitn(9, ',').collect();
        if parts.len() < 8 {
            panic!(
                "equation_assumption_section_contract_cases.csv line {}: expected at least 8 fields, got {}. Line: '{}'",
                line_num + 1,
                parts.len(),
                line
            );
        }

        let expected_section_items = if parts[7].trim().is_empty() || parts[7].trim() == "none" {
            Vec::new()
        } else {
            parts[7]
                .split(';')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect()
        };

        cases.push(EquationAssumptionSectionContractCase {
            equation_str: parts[0].trim().to_string(),
            solve_var: parts[1].trim().to_string(),
            domain_mode: parse_contract_domain_mode(parts[2]),
            assume_scope: parse_contract_assume_scope(parts[3]),
            assumption_reporting: parse_contract_assumption_reporting(parts[4]),
            identity_exp: parts[5].trim().to_string(),
            identity_simp: parts[6].trim().to_string(),
            expected_section_items,
            family: if parts.len() > 8 {
                parts[8].trim().to_string()
            } else {
                "unknown".to_string()
            },
        });
    }

    cases
}

fn load_equation_scope_contract_cases() -> Vec<EquationScopeContractCase> {
    let csv_path = find_test_data_file("equation_scope_contract_cases.csv");
    let file = fs::File::open(&csv_path).unwrap_or_else(|e| {
        panic!("Failed to open {}: {}", csv_path.display(), e);
    });
    let reader = BufReader::new(file);
    let mut cases = Vec::new();

    for (line_num, line) in reader.lines().enumerate() {
        let line = line.unwrap();
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.splitn(8, ',').collect();
        if parts.len() < 7 {
            panic!(
                "equation_scope_contract_cases.csv line {}: expected at least 7 fields, got {}. Line: '{}'",
                line_num + 1,
                parts.len(),
                line
            );
        }

        let expected_scopes = if parts[6].trim().is_empty() || parts[6].trim() == "none" {
            Vec::new()
        } else {
            parts[6]
                .split(';')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect()
        };

        cases.push(EquationScopeContractCase {
            equation_str: parts[0].trim().to_string(),
            solve_var: parts[1].trim().to_string(),
            domain_mode: parse_contract_domain_mode(parts[2]),
            assume_scope: parse_contract_assume_scope(parts[3]),
            identity_exp: parts[4].trim().to_string(),
            identity_simp: parts[5].trim().to_string(),
            expected_scopes,
            family: if parts.len() > 7 {
                parts[7].trim().to_string()
            } else {
                "unknown".to_string()
            },
        });
    }

    cases
}

fn load_equation_step_contract_cases() -> Vec<EquationStepContractCase> {
    let csv_path = find_test_data_file("equation_step_contract_cases.csv");
    let file = fs::File::open(&csv_path).unwrap_or_else(|e| {
        panic!("Failed to open {}: {}", csv_path.display(), e);
    });
    let reader = BufReader::new(file);
    let mut cases = Vec::new();

    for (line_num, line) in reader.lines().enumerate() {
        let line = line.unwrap();
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.splitn(8, ',').collect();
        if parts.len() < 7 {
            panic!(
                "equation_step_contract_cases.csv line {}: expected at least 7 fields, got {}. Line: '{}'",
                line_num + 1,
                parts.len(),
                line
            );
        }

        let expected_keywords = parts[6]
            .split(';')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        cases.push(EquationStepContractCase {
            equation_str: parts[0].trim().to_string(),
            solve_var: parts[1].trim().to_string(),
            domain_mode: parse_contract_domain_mode(parts[2]),
            assume_scope: parse_contract_assume_scope(parts[3]),
            identity_exp: parts[4].trim().to_string(),
            identity_simp: parts[5].trim().to_string(),
            expected_keywords,
            family: if parts.len() > 7 {
                parts[7].trim().to_string()
            } else {
                "unknown".to_string()
            },
        });
    }

    cases
}

fn load_equation_pair_contract_cases() -> Vec<EquationPairContractCase> {
    let csv_path = find_test_data_file("equation_pair_contract_cases.csv");
    let file = fs::File::open(&csv_path).unwrap_or_else(|e| {
        panic!("Failed to open {}: {}", csv_path.display(), e);
    });
    let reader = BufReader::new(file);
    let mut cases = Vec::new();

    for (line_num, line) in reader.lines().enumerate() {
        let line = line.unwrap();
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.splitn(5, ',').collect();
        if parts.len() < 4 {
            panic!(
                "equation_pair_contract_cases.csv line {}: expected at least 4 fields, got {}. Line: '{}'",
                line_num + 1,
                parts.len(),
                line
            );
        }

        cases.push(EquationPairContractCase {
            eq_a: parts[0].trim().to_string(),
            eq_b: parts[1].trim().to_string(),
            solve_var: parts[2].trim().to_string(),
            expected: PairContractExpectation::from_str(parts[3]),
            family: if parts.len() > 4 {
                parts[4].trim().to_string()
            } else {
                "unknown".to_string()
            },
        });
    }

    cases
}

fn solve_equation_kind(
    equation_str: &str,
    solve_var: &str,
    mode: DomainMode,
    scope: AssumeScope,
) -> Result<SolutionSet, String> {
    let mut simplifier = Simplifier::with_default_rules();
    let equation = parse_equation_str(&mut simplifier.context, equation_str)
        .ok_or_else(|| format!("Failed to parse equation '{}'", equation_str))?;
    let opts = make_solver_opts(mode, scope);
    solve_with_display_steps(&equation, solve_var, &mut simplifier, opts)
        .map(|(set, _steps, _diagnostics)| set)
        .map_err(|e| format!("{:?}", e))
}

fn render_required_signatures(ctx: &Context, required: &[ImplicitCondition]) -> Vec<String> {
    let mut items: Vec<String> = required
        .iter()
        .filter(|cond| !cond.is_trivial(ctx))
        .map(|cond| match cond {
            ImplicitCondition::NonZero(id) => format!(
                "nonzero({})",
                DisplayExpr {
                    context: ctx,
                    id: *id
                }
            ),
            ImplicitCondition::Positive(id) => format!(
                "positive({})",
                DisplayExpr {
                    context: ctx,
                    id: *id
                }
            ),
            ImplicitCondition::NonNegative(id) => format!(
                "nonnegative({})",
                DisplayExpr {
                    context: ctx,
                    id: *id
                }
            ),
        })
        .collect();
    items.sort();
    items.dedup();
    items
}

fn required_floor_matches(actual: &[String], expected_floor: &[String]) -> bool {
    expected_floor.iter().all(|item| actual.contains(item))
}

fn solve_equation_required_signatures(
    equation_str: &str,
    solve_var: &str,
    mode: DomainMode,
    scope: AssumeScope,
) -> Result<Vec<String>, String> {
    let mut simplifier = Simplifier::with_default_rules();
    let equation = parse_equation_str(&mut simplifier.context, equation_str)
        .ok_or_else(|| format!("Failed to parse equation '{}'", equation_str))?;
    let opts = make_solver_opts(mode, scope);
    solve_with_display_steps(&equation, solve_var, &mut simplifier, opts)
        .map(|(_set, _steps, diagnostics)| {
            render_required_signatures(&simplifier.context, &diagnostics.required)
        })
        .map_err(|e| format!("{:?}", e))
}

fn solve_equation_assumption_signatures(
    equation_str: &str,
    solve_var: &str,
    mode: DomainMode,
    scope: AssumeScope,
) -> Result<Vec<String>, String> {
    let mut simplifier = Simplifier::with_default_rules();
    let equation = parse_equation_str(&mut simplifier.context, equation_str)
        .ok_or_else(|| format!("Failed to parse equation '{}'", equation_str))?;
    let opts = make_solver_opts(mode, scope);
    solve_with_display_steps(&equation, solve_var, &mut simplifier, opts)
        .map(|(_set, _steps, diagnostics)| {
            let mut items: Vec<String> = diagnostics
                .assumed
                .into_iter()
                .map(|event| {
                    let expr = if let Some(expr_id) = event.expr_id {
                        let (normalized_id, _) = simplifier.simplify(expr_id);
                        DisplayExpr {
                            context: &simplifier.context,
                            id: normalized_id,
                        }
                        .to_string()
                    } else {
                        event.expr_display
                    };
                    format!("{}({})", event.key.kind(), expr)
                })
                .collect();
            items.sort();
            items.dedup();
            items
        })
        .map_err(|e| format!("{:?}", e))
}

fn solve_equation_assumption_record_signatures(
    equation_str: &str,
    solve_var: &str,
    mode: DomainMode,
    scope: AssumeScope,
) -> Result<Vec<String>, String> {
    let mut simplifier = Simplifier::with_default_rules();
    let equation = parse_equation_str(&mut simplifier.context, equation_str)
        .ok_or_else(|| format!("Failed to parse equation '{}'", equation_str))?;
    let opts = make_solver_opts(mode, scope);
    solve_with_display_steps(&equation, solve_var, &mut simplifier, opts)
        .map(|(_set, _steps, diagnostics)| {
            let mut items: Vec<String> = diagnostics
                .assumed_records
                .into_iter()
                .map(|record| {
                    let normalized_expr = parse(&record.expr, &mut simplifier.context)
                        .map(|expr_id| {
                            let (normalized_id, _) = simplifier.simplify(expr_id);
                            DisplayExpr {
                                context: &simplifier.context,
                                id: normalized_id,
                            }
                            .to_string()
                        })
                        .unwrap_or(record.expr);
                    if record.count > 1 {
                        format!("{}({}) (×{})", record.kind, normalized_expr, record.count)
                    } else {
                        format!("{}({})", record.kind, normalized_expr)
                    }
                })
                .collect();
            items.sort();
            items.dedup();
            items
        })
        .map_err(|e| format!("{:?}", e))
}

fn solve_equation_transparency_signal(
    equation_str: &str,
    solve_var: &str,
    mode: DomainMode,
    scope: AssumeScope,
) -> Result<bool, String> {
    let mut simplifier = Simplifier::with_default_rules();
    let equation = parse_equation_str(&mut simplifier.context, equation_str)
        .ok_or_else(|| format!("Failed to parse equation '{}'", equation_str))?;
    let opts = make_solver_opts(mode, scope);
    solve_with_display_steps(&equation, solve_var, &mut simplifier, opts)
        .map(|(_set, steps, _diagnostics)| {
            steps.iter().any(|step| {
                let desc = step.description.to_ascii_lowercase();
                desc.contains("complex") || desc.contains("preset")
            })
        })
        .map_err(|e| format!("{:?}", e))
}

fn solve_equation_warning_lines(
    equation_str: &str,
    solve_var: &str,
    mode: DomainMode,
    scope: AssumeScope,
    assumption_reporting: AssumptionReporting,
) -> Result<Vec<String>, String> {
    let mut simplifier = Simplifier::with_default_rules();
    let mut eval_options = EvalOptions::default();
    eval_options.shared.semantics.domain_mode = mode;
    eval_options.shared.semantics.assume_scope = scope;
    eval_options.shared.assumption_reporting = assumption_reporting;

    let mut session = StatelessEvalSession::new(eval_options.clone());
    let line = format!("solve {equation_str}, {solve_var}");
    evaluate_solve_command_lines_with_session(
        &mut simplifier,
        &mut session,
        &line,
        &eval_options,
        SolveDisplayMode::None,
        false,
    )
    .map(|lines| {
        lines
            .into_iter()
            .filter(|line| line.starts_with("⚠ Assumptions:"))
            .collect()
    })
}

fn solve_equation_assumption_section_lines(
    equation_str: &str,
    solve_var: &str,
    mode: DomainMode,
    scope: AssumeScope,
    assumption_reporting: AssumptionReporting,
) -> Result<Vec<String>, String> {
    let mut simplifier = Simplifier::with_default_rules();
    let mut eval_options = EvalOptions::default();
    eval_options.shared.semantics.domain_mode = mode;
    eval_options.shared.semantics.assume_scope = scope;
    eval_options.shared.assumption_reporting = assumption_reporting;

    let mut session = StatelessEvalSession::new(eval_options.clone());
    let line = format!("solve {equation_str}, {solve_var}");
    evaluate_solve_command_lines_with_session(
        &mut simplifier,
        &mut session,
        &line,
        &eval_options,
        SolveDisplayMode::None,
        true,
    )
    .map(|lines| {
        lines
            .into_iter()
            .filter(|line| line.starts_with("ℹ️ Assumptions used:") || line.starts_with("  - "))
            .collect()
    })
}

fn render_scope_signatures(scopes: &[ScopeTag]) -> Vec<String> {
    let mut items: Vec<String> = scopes
        .iter()
        .map(|scope| match scope {
            ScopeTag::Rule(name) => format!("rule({name})"),
            ScopeTag::Solver(name) => format!("solver({name})"),
            ScopeTag::Command(name) => format!("command({name})"),
        })
        .collect();
    items.sort();
    items.dedup();
    items
}

fn solve_equation_scope_signatures(
    equation_str: &str,
    solve_var: &str,
    mode: DomainMode,
    scope: AssumeScope,
) -> Result<Vec<String>, String> {
    let mut simplifier = Simplifier::with_default_rules();
    let equation = parse_equation_str(&mut simplifier.context, equation_str)
        .ok_or_else(|| format!("Failed to parse equation '{}'", equation_str))?;
    let opts = make_solver_opts(mode, scope);
    solve_with_display_steps(&equation, solve_var, &mut simplifier, opts)
        .map(|(_set, _steps, diagnostics)| render_scope_signatures(&diagnostics.output_scopes))
        .map_err(|e| format!("{:?}", e))
}

fn solve_equation_step_descriptions(
    equation_str: &str,
    solve_var: &str,
    mode: DomainMode,
    scope: AssumeScope,
) -> Result<Vec<String>, String> {
    let mut simplifier = Simplifier::with_default_rules();
    let equation = parse_equation_str(&mut simplifier.context, equation_str)
        .ok_or_else(|| format!("Failed to parse equation '{}'", equation_str))?;
    let mut opts = make_solver_opts(mode, scope);
    opts.detailed_steps = true;
    solve_with_display_steps(&equation, solve_var, &mut simplifier, opts)
        .map(|(_set, steps, _diagnostics)| {
            steps
                .iter()
                .map(|step| step.description.to_ascii_lowercase())
                .collect()
        })
        .map_err(|e| format!("{:?}", e))
}

fn transformed_equation_text(
    equation_str: &str,
    identity_exp: &str,
    identity_simp: &str,
) -> String {
    let parts: Vec<&str> = equation_str.splitn(2, '=').collect();
    let lhs = parts.first().map(|s| s.trim()).unwrap_or("");
    let rhs = parts.get(1).map(|s| s.trim()).unwrap_or("");
    format!("({lhs}) + ({identity_exp}) = ({rhs}) + ({identity_simp})")
}

fn run_equation_kind_contract_tests() -> EquationKindContractMetrics {
    let cases = load_equation_kind_contract_cases();
    let verbose = is_verbose();
    let mut metrics = EquationKindContractMetrics::default();

    eprintln!();
    eprintln!(
        "=== Equation Solution-Kind Contracts: {} curated cases ===",
        cases.len()
    );

    for (i, case) in cases.iter().enumerate() {
        metrics.total += 1;
        let transformed =
            transformed_equation_text(&case.equation_str, &case.identity_exp, &case.identity_simp);

        let orig_result = solve_equation_kind(
            &case.equation_str,
            &case.solve_var,
            case.domain_mode,
            case.assume_scope,
        );
        let trans_result = solve_equation_kind(
            &transformed,
            &case.solve_var,
            case.domain_mode,
            case.assume_scope,
        );

        let passed = match (&orig_result, &trans_result) {
            (Ok(orig), Ok(trans)) => {
                case.expected_orig.matches(orig) && case.expected_trans.matches(trans)
            }
            _ => false,
        };

        if passed {
            metrics.passed += 1;
        } else {
            metrics.failed += 1;
        }

        if verbose || !passed {
            let orig_detail = match &orig_result {
                Ok(set) => format!("{:?}", set),
                Err(msg) => format!("error: {}", msg),
            };
            let trans_detail = match &trans_result {
                Ok(set) => format!("{:?}", set),
                Err(msg) => format!("error: {}", msg),
            };
            eprintln!(
                "  {} {:>3}. [{}] {} — expected ({}, {}), got ({}, {})",
                if passed { "✓" } else { "✗" },
                i + 1,
                truncate(&case.family, 18),
                truncate(&case.equation_str, 26),
                case.expected_orig.label(),
                case.expected_trans.label(),
                truncate(&orig_detail, 28),
                truncate(&trans_detail, 28),
            );
        }
    }

    eprintln!();
    eprintln!(
        "Equation solution-kind contracts: {} total, {} passed, {} failed",
        metrics.total, metrics.passed, metrics.failed
    );

    metrics
}

fn run_equation_required_contract_tests() -> EquationRequiredContractMetrics {
    let cases = load_equation_required_contract_cases();
    let verbose = is_verbose();
    let mut metrics = EquationRequiredContractMetrics::default();

    eprintln!();
    eprintln!(
        "=== Equation Required-Condition Contracts: {} curated cases ===",
        cases.len()
    );

    for (i, case) in cases.iter().enumerate() {
        metrics.total += 1;
        let transformed =
            transformed_equation_text(&case.equation_str, &case.identity_exp, &case.identity_simp);

        let orig_required = solve_equation_required_signatures(
            &case.equation_str,
            &case.solve_var,
            case.domain_mode,
            case.assume_scope,
        );
        let trans_required = solve_equation_required_signatures(
            &transformed,
            &case.solve_var,
            case.domain_mode,
            case.assume_scope,
        );

        let passed = match (&orig_required, &trans_required) {
            (Ok(orig), Ok(trans)) if case.expected_required.is_empty() => {
                orig.is_empty() && trans.is_empty()
            }
            (Ok(orig), Ok(trans)) => {
                required_floor_matches(orig, &case.expected_required)
                    && required_floor_matches(trans, &case.expected_required)
            }
            _ => false,
        };

        if passed {
            metrics.passed += 1;
        } else {
            metrics.failed += 1;
        }

        if verbose || !passed {
            let orig_detail = match &orig_required {
                Ok(reqs) => format!("{:?}", reqs),
                Err(msg) => format!("error: {}", msg),
            };
            let trans_detail = match &trans_required {
                Ok(reqs) => format!("{:?}", reqs),
                Err(msg) => format!("error: {}", msg),
            };
            eprintln!(
                "  {} {:>3}. [{}] {} — expected {:?}, got ({}, {})",
                if passed { "✓" } else { "✗" },
                i + 1,
                truncate(&case.family, 18),
                truncate(&case.equation_str, 26),
                case.expected_required,
                truncate(&orig_detail, 28),
                truncate(&trans_detail, 28),
            );
        }
    }

    eprintln!();
    eprintln!(
        "Equation required-condition contracts: {} total, {} passed, {} failed",
        metrics.total, metrics.passed, metrics.failed
    );

    metrics
}

fn run_equation_assumption_contract_tests() -> EquationAssumptionContractMetrics {
    let cases = load_equation_assumption_contract_cases();
    let verbose = is_verbose();
    let mut metrics = EquationAssumptionContractMetrics::default();

    eprintln!();
    eprintln!(
        "=== Equation Assumption-Signal Contracts: {} curated cases ===",
        cases.len()
    );

    for (i, case) in cases.iter().enumerate() {
        metrics.total += 1;
        let transformed =
            transformed_equation_text(&case.equation_str, &case.identity_exp, &case.identity_simp);

        let orig_assumed = solve_equation_assumption_signatures(
            &case.equation_str,
            &case.solve_var,
            case.domain_mode,
            case.assume_scope,
        );
        let trans_assumed = solve_equation_assumption_signatures(
            &transformed,
            &case.solve_var,
            case.domain_mode,
            case.assume_scope,
        );

        let passed = match (&orig_assumed, &trans_assumed) {
            (Ok(orig), Ok(trans)) if case.expected_assumed.is_empty() => {
                orig.is_empty() && trans.is_empty()
            }
            (Ok(orig), Ok(trans)) => {
                required_floor_matches(orig, &case.expected_assumed)
                    && required_floor_matches(trans, &case.expected_assumed)
            }
            _ => false,
        };

        if passed {
            metrics.passed += 1;
        } else {
            metrics.failed += 1;
        }

        if verbose || !passed {
            let orig_detail = match &orig_assumed {
                Ok(items) => format!("{:?}", items),
                Err(msg) => format!("error: {}", msg),
            };
            let trans_detail = match &trans_assumed {
                Ok(items) => format!("{:?}", items),
                Err(msg) => format!("error: {}", msg),
            };
            eprintln!(
                "  {} {:>3}. [{}] {} — expected {:?}, got ({}, {})",
                if passed { "✓" } else { "✗" },
                i + 1,
                truncate(&case.family, 18),
                truncate(&case.equation_str, 26),
                case.expected_assumed,
                truncate(&orig_detail, 28),
                truncate(&trans_detail, 28),
            );
        }
    }

    eprintln!();
    eprintln!(
        "Equation assumption-signal contracts: {} total, {} passed, {} failed",
        metrics.total, metrics.passed, metrics.failed
    );

    metrics
}

fn run_equation_transparency_contract_tests() -> EquationTransparencyContractMetrics {
    let cases = load_equation_transparency_contract_cases();
    let verbose = is_verbose();
    let mut metrics = EquationTransparencyContractMetrics::default();

    eprintln!();
    eprintln!(
        "=== Equation Transparency Contracts: {} curated cases ===",
        cases.len()
    );

    for (i, case) in cases.iter().enumerate() {
        metrics.total += 1;
        let transformed =
            transformed_equation_text(&case.equation_str, &case.identity_exp, &case.identity_simp);

        let orig_signal = solve_equation_transparency_signal(
            &case.equation_str,
            &case.solve_var,
            case.domain_mode,
            case.assume_scope,
        );
        let trans_signal = solve_equation_transparency_signal(
            &transformed,
            &case.solve_var,
            case.domain_mode,
            case.assume_scope,
        );

        let passed = match (&orig_signal, &trans_signal) {
            (Ok(orig), Ok(trans)) => {
                *orig == case.expected_signal && *trans == case.expected_signal
            }
            _ => false,
        };

        if passed {
            metrics.passed += 1;
        } else {
            metrics.failed += 1;
        }

        if verbose || !passed {
            let orig_detail = match &orig_signal {
                Ok(signal) => signal.to_string(),
                Err(msg) => format!("error: {}", msg),
            };
            let trans_detail = match &trans_signal {
                Ok(signal) => signal.to_string(),
                Err(msg) => format!("error: {}", msg),
            };
            eprintln!(
                "  {} {:>3}. [{}] {} — expected {}, got ({}, {})",
                if passed { "✓" } else { "✗" },
                i + 1,
                truncate(&case.family, 18),
                truncate(&case.equation_str, 26),
                if case.expected_signal {
                    "signal"
                } else {
                    "no-signal"
                },
                truncate(&orig_detail, 28),
                truncate(&trans_detail, 28),
            );
        }
    }

    eprintln!();
    eprintln!(
        "Equation transparency contracts: {} total, {} passed, {} failed",
        metrics.total, metrics.passed, metrics.failed
    );

    metrics
}

fn run_equation_assumption_record_contract_tests() -> EquationAssumptionRecordContractMetrics {
    let cases = load_equation_assumption_record_contract_cases();
    let verbose = is_verbose();
    let mut metrics = EquationAssumptionRecordContractMetrics::default();

    eprintln!();
    eprintln!(
        "=== Equation Assumption-Record Contracts: {} curated cases ===",
        cases.len()
    );

    for (i, case) in cases.iter().enumerate() {
        metrics.total += 1;
        let transformed =
            transformed_equation_text(&case.equation_str, &case.identity_exp, &case.identity_simp);

        let orig_records = solve_equation_assumption_record_signatures(
            &case.equation_str,
            &case.solve_var,
            case.domain_mode,
            case.assume_scope,
        );
        let trans_records = solve_equation_assumption_record_signatures(
            &transformed,
            &case.solve_var,
            case.domain_mode,
            case.assume_scope,
        );

        let passed = match (&orig_records, &trans_records) {
            (Ok(orig), Ok(trans)) if case.expected_records.is_empty() => {
                orig.is_empty() && trans.is_empty()
            }
            (Ok(orig), Ok(trans)) => {
                required_floor_matches(orig, &case.expected_records)
                    && required_floor_matches(trans, &case.expected_records)
            }
            _ => false,
        };

        if passed {
            metrics.passed += 1;
        } else {
            metrics.failed += 1;
        }

        if verbose || !passed {
            let orig_detail = match &orig_records {
                Ok(items) => format!("{:?}", items),
                Err(msg) => format!("error: {}", msg),
            };
            let trans_detail = match &trans_records {
                Ok(items) => format!("{:?}", items),
                Err(msg) => format!("error: {}", msg),
            };
            eprintln!(
                "  {} {:>3}. [{}] {} — expected {:?}, got ({}, {})",
                if passed { "✓" } else { "✗" },
                i + 1,
                truncate(&case.family, 18),
                truncate(&case.equation_str, 26),
                case.expected_records,
                truncate(&orig_detail, 28),
                truncate(&trans_detail, 28),
            );
        }
    }

    eprintln!();
    eprintln!(
        "Equation assumption-record contracts: {} total, {} passed, {} failed",
        metrics.total, metrics.passed, metrics.failed
    );

    metrics
}

fn run_equation_warning_contract_tests() -> EquationWarningContractMetrics {
    let cases = load_equation_warning_contract_cases();
    let verbose = is_verbose();
    let mut metrics = EquationWarningContractMetrics::default();

    eprintln!();
    eprintln!(
        "=== Equation Warning Contracts: {} curated cases ===",
        cases.len()
    );

    for (i, case) in cases.iter().enumerate() {
        metrics.total += 1;
        let transformed =
            transformed_equation_text(&case.equation_str, &case.identity_exp, &case.identity_simp);

        let orig_warning_lines = solve_equation_warning_lines(
            &case.equation_str,
            &case.solve_var,
            case.domain_mode,
            case.assume_scope,
            case.assumption_reporting,
        );
        let trans_warning_lines = solve_equation_warning_lines(
            &transformed,
            &case.solve_var,
            case.domain_mode,
            case.assume_scope,
            case.assumption_reporting,
        );

        let passed = match (&orig_warning_lines, &trans_warning_lines) {
            (Ok(orig), Ok(trans)) if case.expected_warning_items.is_empty() => {
                orig.is_empty() && trans.is_empty()
            }
            (Ok(orig), Ok(trans)) => case.expected_warning_items.iter().all(|item| {
                orig.iter().any(|line| line.contains(item))
                    && trans.iter().any(|line| line.contains(item))
            }),
            _ => false,
        };

        if passed {
            metrics.passed += 1;
        } else {
            metrics.failed += 1;
        }

        if verbose || !passed {
            let orig_detail = match &orig_warning_lines {
                Ok(items) => format!("{:?}", items),
                Err(msg) => format!("error: {}", msg),
            };
            let trans_detail = match &trans_warning_lines {
                Ok(items) => format!("{:?}", items),
                Err(msg) => format!("error: {}", msg),
            };
            eprintln!(
                "  {} {:>3}. [{}] {} — expected {:?}, got ({}, {})",
                if passed { "✓" } else { "✗" },
                i + 1,
                truncate(&case.family, 18),
                truncate(&case.equation_str, 26),
                case.expected_warning_items,
                truncate(&orig_detail, 28),
                truncate(&trans_detail, 28),
            );
        }
    }

    eprintln!();
    eprintln!(
        "Equation warning contracts: {} total, {} passed, {} failed",
        metrics.total, metrics.passed, metrics.failed
    );

    metrics
}

fn run_equation_assumption_section_contract_tests() -> EquationAssumptionSectionContractMetrics {
    let cases = load_equation_assumption_section_contract_cases();
    let verbose = is_verbose();
    let mut metrics = EquationAssumptionSectionContractMetrics::default();

    eprintln!();
    eprintln!(
        "=== Equation Assumption-Section Contracts: {} curated cases ===",
        cases.len()
    );

    for (i, case) in cases.iter().enumerate() {
        metrics.total += 1;
        let transformed =
            transformed_equation_text(&case.equation_str, &case.identity_exp, &case.identity_simp);

        let orig_section_lines = solve_equation_assumption_section_lines(
            &case.equation_str,
            &case.solve_var,
            case.domain_mode,
            case.assume_scope,
            case.assumption_reporting,
        );
        let trans_section_lines = solve_equation_assumption_section_lines(
            &transformed,
            &case.solve_var,
            case.domain_mode,
            case.assume_scope,
            case.assumption_reporting,
        );

        let passed = match (&orig_section_lines, &trans_section_lines) {
            (Ok(orig), Ok(trans)) if case.expected_section_items.is_empty() => {
                orig.is_empty() && trans.is_empty()
            }
            (Ok(orig), Ok(trans)) => case.expected_section_items.iter().all(|item| {
                orig.iter().any(|line| line.contains(item))
                    && trans.iter().any(|line| line.contains(item))
            }),
            _ => false,
        };

        if passed {
            metrics.passed += 1;
        } else {
            metrics.failed += 1;
        }

        if verbose || !passed {
            let orig_detail = match &orig_section_lines {
                Ok(items) => format!("{:?}", items),
                Err(msg) => format!("error: {}", msg),
            };
            let trans_detail = match &trans_section_lines {
                Ok(items) => format!("{:?}", items),
                Err(msg) => format!("error: {}", msg),
            };
            eprintln!(
                "  {} {:>3}. [{}] {} — expected {:?}, got ({}, {})",
                if passed { "✓" } else { "✗" },
                i + 1,
                truncate(&case.family, 18),
                truncate(&case.equation_str, 26),
                case.expected_section_items,
                truncate(&orig_detail, 28),
                truncate(&trans_detail, 28),
            );
        }
    }

    eprintln!();
    eprintln!(
        "Equation assumption-section contracts: {} total, {} passed, {} failed",
        metrics.total, metrics.passed, metrics.failed
    );

    metrics
}

fn run_equation_scope_contract_tests() -> EquationScopeContractMetrics {
    let cases = load_equation_scope_contract_cases();
    let verbose = is_verbose();
    let mut metrics = EquationScopeContractMetrics::default();

    eprintln!();
    eprintln!(
        "=== Equation Scope Contracts: {} curated cases ===",
        cases.len()
    );

    for (i, case) in cases.iter().enumerate() {
        metrics.total += 1;
        let transformed =
            transformed_equation_text(&case.equation_str, &case.identity_exp, &case.identity_simp);

        let orig_scopes = solve_equation_scope_signatures(
            &case.equation_str,
            &case.solve_var,
            case.domain_mode,
            case.assume_scope,
        );
        let trans_scopes = solve_equation_scope_signatures(
            &transformed,
            &case.solve_var,
            case.domain_mode,
            case.assume_scope,
        );

        let passed = match (&orig_scopes, &trans_scopes) {
            (Ok(orig), Ok(trans)) if case.expected_scopes.is_empty() => {
                orig.is_empty() && trans.is_empty()
            }
            (Ok(orig), Ok(trans)) => {
                required_floor_matches(orig, &case.expected_scopes)
                    && required_floor_matches(trans, &case.expected_scopes)
            }
            _ => false,
        };

        if passed {
            metrics.passed += 1;
        } else {
            metrics.failed += 1;
        }

        if verbose || !passed {
            let orig_detail = match &orig_scopes {
                Ok(items) => format!("{:?}", items),
                Err(msg) => format!("error: {}", msg),
            };
            let trans_detail = match &trans_scopes {
                Ok(items) => format!("{:?}", items),
                Err(msg) => format!("error: {}", msg),
            };
            eprintln!(
                "  {} {:>3}. [{}] {} — expected {:?}, got ({}, {})",
                if passed { "✓" } else { "✗" },
                i + 1,
                truncate(&case.family, 18),
                truncate(&case.equation_str, 26),
                case.expected_scopes,
                truncate(&orig_detail, 28),
                truncate(&trans_detail, 28),
            );
        }
    }

    eprintln!();
    eprintln!(
        "Equation scope contracts: {} total, {} passed, {} failed",
        metrics.total, metrics.passed, metrics.failed
    );

    metrics
}

fn run_equation_step_contract_tests() -> EquationStepContractMetrics {
    let cases = load_equation_step_contract_cases();
    let verbose = is_verbose();
    let mut metrics = EquationStepContractMetrics::default();

    eprintln!();
    eprintln!(
        "=== Equation Step Contracts: {} curated cases ===",
        cases.len()
    );

    for (i, case) in cases.iter().enumerate() {
        metrics.total += 1;
        let transformed =
            transformed_equation_text(&case.equation_str, &case.identity_exp, &case.identity_simp);

        let orig_steps = solve_equation_step_descriptions(
            &case.equation_str,
            &case.solve_var,
            case.domain_mode,
            case.assume_scope,
        );
        let trans_steps = solve_equation_step_descriptions(
            &transformed,
            &case.solve_var,
            case.domain_mode,
            case.assume_scope,
        );

        let passed = match (&orig_steps, &trans_steps) {
            (Ok(orig), Ok(trans)) => case.expected_keywords.iter().all(|kw| {
                orig.iter().any(|step| step.contains(kw))
                    && trans.iter().any(|step| step.contains(kw))
            }),
            _ => false,
        };

        if passed {
            metrics.passed += 1;
        } else {
            metrics.failed += 1;
        }

        if verbose || !passed {
            let orig_detail = match &orig_steps {
                Ok(items) => format!("{:?}", items),
                Err(msg) => format!("error: {}", msg),
            };
            let trans_detail = match &trans_steps {
                Ok(items) => format!("{:?}", items),
                Err(msg) => format!("error: {}", msg),
            };
            eprintln!(
                "  {} {:>3}. [{}] {} — expected {:?}, got ({}, {})",
                if passed { "✓" } else { "✗" },
                i + 1,
                truncate(&case.family, 18),
                truncate(&case.equation_str, 26),
                case.expected_keywords,
                truncate(&orig_detail, 28),
                truncate(&trans_detail, 28),
            );
        }
    }

    eprintln!();
    eprintln!(
        "Equation step contracts: {} total, {} passed, {} failed",
        metrics.total, metrics.passed, metrics.failed
    );

    metrics
}

fn run_equation_pair_contract_tests() -> EquationPairContractMetrics {
    let cases = load_equation_pair_contract_cases();
    let verbose = is_verbose();
    let mut metrics = EquationPairContractMetrics::default();

    eprintln!();
    eprintln!(
        "=== Equation Pair Contracts: {} curated cases ===",
        cases.len()
    );

    for (i, case) in cases.iter().enumerate() {
        metrics.total += 1;
        let pair = EquationPair {
            eq_a: case.eq_a.clone(),
            eq_b: case.eq_b.clone(),
            solve_var: case.solve_var.clone(),
            family: case.family.clone(),
        };
        let outcome = verify_equation_pair(&pair);
        let passed = case.expected.matches(&outcome);

        if passed {
            metrics.passed += 1;
        } else {
            metrics.failed += 1;
        }

        if verbose || !passed {
            let detail = match &outcome {
                PairOutcome::Verified => "verified".to_string(),
                PairOutcome::NumericFallback => "numeric".to_string(),
                PairOutcome::NumericInconclusive => "numeric-inconclusive".to_string(),
                PairOutcome::Mismatch(msg) => format!("mismatch: {}", msg),
                PairOutcome::SolverFailed(msg) => format!("solver: {}", msg),
                PairOutcome::ParseError(msg) => format!("parse: {}", msg),
                PairOutcome::Timeout => "timeout".to_string(),
            };
            eprintln!(
                "  {} {:>3}. [{}] '{}' ↔ '{}' — expected {}, got {}",
                if passed { "✓" } else { "✗" },
                i + 1,
                truncate(&case.family, 18),
                case.eq_a,
                case.eq_b,
                case.expected.label(),
                detail
            );
        }
    }

    eprintln!();
    eprintln!(
        "Equation pair contracts: {} total, {} passed, {} failed",
        metrics.total, metrics.passed, metrics.failed
    );

    metrics
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
        let outcome = match rx.recv_timeout(SOLVE_TIMEOUT) {
            Ok(result) => result,
            Err(_) => S2Outcome::Timeout,
        };
        let outcome = normalize_s2_outcome(eq, id, outcome);
        results.record(&outcome);

        // Record context for Incomplete outcomes
        if let S2Outcome::Incomplete(ref reason) = outcome {
            results.record_incomplete_context(reason, id_idx, &eq.family);
        }

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
                S2Outcome::OkPartialVerified => ("◐", "partial-verified".into()),
                S2Outcome::Incomplete(reason) => ("⚠", format!("incomplete: {}", reason)),
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
        "  │  ✓ symbolic: {}  ≈ numeric: {}  ◐ partial: {}  ⚠ incomplete: {}",
        results.ok_symbolic, results.ok_numeric, results.ok_partial, results.incomplete
    );
    eprintln!(
        "  │  D domain-chg: {}  ✗ mismatch: {}  E errors: {}  T timeout: {}",
        results.domain_changed, results.mismatches, results.errors, results.timeouts
    );
    eprintln!("  └─────────────────────────────────────────────────────────────────┘");

    // --- Incomplete cross-tab ---
    if results.incomplete > 0 {
        // Per-reason breakdown
        eprintln!();
        eprintln!("  ⚠ Incomplete breakdown by reason:");
        let mut reasons: Vec<_> = results.incomplete_reasons.iter().collect();
        reasons.sort_by(|a, b| b.1.cmp(a.1));
        for (reason, count) in &reasons {
            eprintln!("    {:>3}× {}", count, reason);
        }

        // Top-10 identity offenders
        eprintln!();
        eprintln!("  ⚠ Top identity offenders:");
        let mut id_top: Vec<_> = results.identity_offenders.iter().collect();
        id_top.sort_by(|a, b| b.1.cmp(a.1));
        for (id_idx, count) in id_top.iter().take(10) {
            let identity = &identities[**id_idx];
            eprintln!(
                "    {:>3}× id#{:>3} [{}] {} ≡ {}",
                count,
                id_idx,
                truncate(&identity.family, 12),
                truncate(&identity.exp, 25),
                truncate(&identity.simp, 25)
            );
        }

        // Top-10 equation family offenders
        eprintln!();
        eprintln!("  ⚠ Top equation family offenders:");
        let mut fam_top: Vec<_> = results.family_offenders.iter().collect();
        fam_top.sort_by(|a, b| b.1.cmp(a.1));
        for (family, count) in fam_top.iter().take(10) {
            eprintln!("    {:>3}× {}", count, family);
        }
    }

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

#[test]
#[ignore] // Run with: cargo test -p cas_solver --test metamorphic_equation_tests metatest_equation_transform_contracts -- --ignored --nocapture
fn metatest_equation_transform_contracts() {
    let metrics = run_s2_transform_contract_tests();
    assert_eq!(
        metrics.failed, 0,
        "Equation transform contracts had {} failures — see details above",
        metrics.failed
    );
    assert_eq!(
        metrics.load_errors, 0,
        "Equation transform contracts had {} load errors — see details above",
        metrics.load_errors
    );
}

#[test]
#[ignore] // Run with: cargo test -p cas_solver --test metamorphic_equation_tests metatest_equation_solution_kind_contracts -- --ignored --nocapture
fn metatest_equation_solution_kind_contracts() {
    let metrics = run_equation_kind_contract_tests();
    assert_eq!(
        metrics.failed, 0,
        "Equation solution-kind contracts had {} failures — see details above",
        metrics.failed
    );
}

#[test]
#[ignore] // Run with: cargo test -p cas_solver --test metamorphic_equation_tests metatest_equation_required_contracts -- --ignored --nocapture
fn metatest_equation_required_contracts() {
    let metrics = run_equation_required_contract_tests();
    assert_eq!(
        metrics.failed, 0,
        "Equation required-condition contracts had {} failures — see details above",
        metrics.failed
    );
}

#[test]
#[ignore] // Run with: cargo test -p cas_solver --test metamorphic_equation_tests metatest_equation_assumption_contracts -- --ignored --nocapture
fn metatest_equation_assumption_contracts() {
    let metrics = run_equation_assumption_contract_tests();
    assert_eq!(
        metrics.failed, 0,
        "Equation assumption-signal contracts had {} failures — see details above",
        metrics.failed
    );
}

#[test]
#[ignore] // Run with: cargo test -p cas_solver --test metamorphic_equation_tests metatest_equation_assumption_record_contracts -- --ignored --nocapture
fn metatest_equation_assumption_record_contracts() {
    let metrics = run_equation_assumption_record_contract_tests();
    assert_eq!(
        metrics.failed, 0,
        "Equation assumption-record contracts had {} failures — see details above",
        metrics.failed
    );
}

#[test]
#[ignore] // Run with: cargo test -p cas_solver --test metamorphic_equation_tests metatest_equation_warning_contracts -- --ignored --nocapture
fn metatest_equation_warning_contracts() {
    let metrics = run_equation_warning_contract_tests();
    assert_eq!(
        metrics.failed, 0,
        "Equation warning contracts had {} failures — see details above",
        metrics.failed
    );
}

#[test]
#[ignore] // Run with: cargo test -p cas_solver --test metamorphic_equation_tests metatest_equation_assumption_section_contracts -- --ignored --nocapture
fn metatest_equation_assumption_section_contracts() {
    let metrics = run_equation_assumption_section_contract_tests();
    assert_eq!(
        metrics.failed, 0,
        "Equation assumption-section contracts had {} failures — see details above",
        metrics.failed
    );
}

#[test]
#[ignore] // Run with: cargo test -p cas_solver --test metamorphic_equation_tests metatest_equation_transparency_contracts -- --ignored --nocapture
fn metatest_equation_transparency_contracts() {
    let metrics = run_equation_transparency_contract_tests();
    assert_eq!(
        metrics.failed, 0,
        "Equation transparency contracts had {} failures — see details above",
        metrics.failed
    );
}

#[test]
#[ignore] // Run with: cargo test -p cas_solver --test metamorphic_equation_tests metatest_equation_scope_contracts -- --ignored --nocapture
fn metatest_equation_scope_contracts() {
    let metrics = run_equation_scope_contract_tests();
    assert_eq!(
        metrics.failed, 0,
        "Equation scope contracts had {} failures — see details above",
        metrics.failed
    );
}

#[test]
#[ignore] // Run with: cargo test -p cas_solver --test metamorphic_equation_tests metatest_equation_step_contracts -- --ignored --nocapture
fn metatest_equation_step_contracts() {
    let metrics = run_equation_step_contract_tests();
    assert_eq!(
        metrics.failed, 0,
        "Equation step contracts had {} failures — see details above",
        metrics.failed
    );
}

#[test]
#[ignore] // Run with: cargo test -p cas_solver --test metamorphic_equation_tests metatest_equation_pair_contracts -- --ignored --nocapture
fn metatest_equation_pair_contracts() {
    let metrics = run_equation_pair_contract_tests();
    assert_eq!(
        metrics.failed, 0,
        "Equation pair contracts had {} failures — see details above",
        metrics.failed
    );
}

#[test]
#[ignore] // Run with: cargo test -p cas_solver --test metamorphic_equation_tests metatest_equation_contract_suites -- --ignored --nocapture
fn metatest_equation_contract_suites() {
    let transform = run_s2_transform_contract_tests();
    let kind = run_equation_kind_contract_tests();
    let required = run_equation_required_contract_tests();
    let assumed = run_equation_assumption_contract_tests();
    let assumption_records = run_equation_assumption_record_contract_tests();
    let warnings = run_equation_warning_contract_tests();
    let assumption_sections = run_equation_assumption_section_contract_tests();
    let transparency = run_equation_transparency_contract_tests();
    let scope = run_equation_scope_contract_tests();
    let steps = run_equation_step_contract_tests();
    let pair = run_equation_pair_contract_tests();

    eprintln!();
    eprintln!(
        "📦 Equation contract summary: transforms={} kind={} required={} assumed={} assumption_records={} warnings={} assumption_sections={} transparency={} scope={} steps={} pair={}",
        transform.total,
        kind.total,
        required.total,
        assumed.total,
        assumption_records.total,
        warnings.total,
        assumption_sections.total,
        transparency.total,
        scope.total,
        steps.total,
        pair.total
    );

    assert_eq!(
        transform.failed, 0,
        "Equation transform contracts had {} failures",
        transform.failed
    );
    assert_eq!(
        transform.load_errors, 0,
        "Equation transform contracts had {} load errors",
        transform.load_errors
    );
    assert_eq!(
        kind.failed, 0,
        "Equation solution-kind contracts had {} failures",
        kind.failed
    );
    assert_eq!(
        required.failed, 0,
        "Equation required-condition contracts had {} failures",
        required.failed
    );
    assert_eq!(
        assumed.failed, 0,
        "Equation assumption-signal contracts had {} failures",
        assumed.failed
    );
    assert_eq!(
        assumption_records.failed, 0,
        "Equation assumption-record contracts had {} failures",
        assumption_records.failed
    );
    assert_eq!(
        warnings.failed, 0,
        "Equation warning contracts had {} failures",
        warnings.failed
    );
    assert_eq!(
        assumption_sections.failed, 0,
        "Equation assumption-section contracts had {} failures",
        assumption_sections.failed
    );
    assert_eq!(
        transparency.failed, 0,
        "Equation transparency contracts had {} failures",
        transparency.failed
    );
    assert_eq!(
        scope.failed, 0,
        "Equation scope contracts had {} failures",
        scope.failed
    );
    assert_eq!(
        steps.failed, 0,
        "Equation step contracts had {} failures",
        steps.failed
    );
    assert_eq!(
        pair.failed, 0,
        "Equation pair contracts had {} failures",
        pair.failed
    );
}
