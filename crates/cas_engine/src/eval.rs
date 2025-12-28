use crate::session::{EntryKind, ResolveError};
use crate::session_state::SessionState;
use crate::Simplifier;
use cas_ast::{Equation, Expr, ExprId, RelOp};

/// The central Engine struct that wraps the core Simplifier and potentially other components.
///
/// # Example
///
/// ```
/// use cas_engine::Engine;
/// use cas_parser::parse;
///
/// let mut engine = Engine::new();
/// let expr = parse("x + x", &mut engine.simplifier.context).unwrap();
/// let (result, _steps) = engine.simplifier.simplify(expr);
///
/// // Result is simplified
/// use cas_ast::display::DisplayExpr;
/// let output = format!("{}", DisplayExpr { context: &engine.simplifier.context, id: result });
/// assert!(output.contains("x")); // Contains x
/// ```
pub struct Engine {
    pub simplifier: Simplifier,
}

impl Engine {
    /// Create a new Engine with default rules.
    ///
    /// # Example
    ///
    /// ```
    /// use cas_engine::Engine;
    ///
    /// let engine = Engine::new();
    /// // Engine is ready to simplify expressions
    /// ```
    pub fn new() -> Self {
        Self {
            simplifier: Simplifier::with_default_rules(),
        }
    }

    /// Determine effective options, resolving Auto modes based on expression content.
    /// - ContextMode::Auto → IntegratePrep if contains integrate(), else Standard
    /// - ComplexMode::Auto → On if contains i, else Off
    /// - expand_policy forced Off in Solve mode (anti-contamination)
    fn effective_options(
        &self,
        opts: &crate::options::EvalOptions,
        expr: ExprId,
    ) -> crate::options::EvalOptions {
        use crate::options::{ComplexMode, ContextMode};
        use crate::phase::ExpandPolicy;

        let mut effective = opts.clone();

        // Resolve ContextMode::Auto
        if opts.context_mode == ContextMode::Auto {
            if crate::helpers::contains_integral(&self.simplifier.context, expr) {
                effective.context_mode = ContextMode::IntegratePrep;
            } else {
                effective.context_mode = ContextMode::Standard;
            }
        }

        // Resolve ComplexMode::Auto
        if opts.complex_mode == ComplexMode::Auto {
            if crate::helpers::contains_i(&self.simplifier.context, expr) {
                effective.complex_mode = ComplexMode::On;
            } else {
                effective.complex_mode = ComplexMode::Off;
            }
        }

        // CRITICAL: Force expand_policy Off in Solve mode
        // Auto-expansion can interfere with equation solving strategies
        if effective.context_mode == ContextMode::Solve {
            effective.expand_policy = ExpandPolicy::Off;
        }

        effective
    }
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug)]
pub enum EvalAction {
    Simplify,
    Expand,
    // Solve for a variable.
    Solve { var: String },
    // Check equivalence between two expressions
    Equiv { other: ExprId },
}

#[derive(Clone, Debug)]
pub struct EvalRequest {
    pub raw_input: String,
    pub parsed: ExprId,
    pub kind: EntryKind,
    pub action: EvalAction,
    pub auto_store: bool,
}

#[derive(Clone, Debug)]
pub enum EvalResult {
    Expr(ExprId),
    Set(Vec<ExprId>), // For Solve multiple roots
    Bool(bool),       // For Equiv
    None,             // For commands with no output
}

/// A domain assumption warning with its source rule.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DomainWarning {
    pub message: String,
    pub rule_name: String,
}

#[derive(Clone, Debug)]
pub struct EvalOutput {
    pub stored_id: Option<u64>,
    pub parsed: ExprId,
    pub resolved: ExprId,
    pub result: EvalResult,
    /// Domain warnings with deduplication and rule source.
    pub domain_warnings: Vec<DomainWarning>,
    pub steps: Vec<crate::Step>,
    pub solve_steps: Vec<crate::solver::SolveStep>,
}

/// Collect domain warnings from steps with deduplication.
/// Collects both legacy domain_assumption strings AND structured assumption_events.
fn collect_domain_warnings(steps: &[crate::Step]) -> Vec<DomainWarning> {
    use std::collections::HashSet;

    let mut seen: HashSet<String> = HashSet::new();
    let mut warnings = Vec::new();

    for step in steps {
        // Collect legacy domain_assumption strings
        if let Some(msg) = &step.domain_assumption {
            let msg_str = msg.to_string();
            if !seen.contains(&msg_str) {
                seen.insert(msg_str.clone());
                warnings.push(DomainWarning {
                    message: msg_str,
                    rule_name: step.rule_name.clone(),
                });
            }
        }

        // Collect structured assumption_events (priority channel)
        for event in &step.assumption_events {
            let msg_str = event.message.clone();
            if !seen.contains(&msg_str) {
                seen.insert(msg_str.clone());
                warnings.push(DomainWarning {
                    message: msg_str,
                    rule_name: step.rule_name.clone(),
                });
            }
        }
    }

    warnings
}

impl Engine {
    /// The main entry point for evaluating requests.
    /// Handles session storage, resolution, and action dispatch.
    pub fn eval(
        &mut self,
        state: &mut SessionState,
        req: EvalRequest,
    ) -> Result<EvalOutput, anyhow::Error> {
        // 1. Auto-store raw + parsed (unresolved)
        let stored_id = if req.auto_store {
            Some(state.store.push(req.kind, req.raw_input.clone()))
        } else {
            None
        };

        // 2. Resolve (state.resolve_all)
        // We resolve the parsed expression against the session state (Session refs #id and Environment vars)
        let resolved = match state.resolve_all(&mut self.simplifier.context, req.parsed) {
            Ok(r) => r,
            Err(ResolveError::CircularReference(msg)) => {
                return Err(anyhow::anyhow!("Circular reference detected: {}", msg))
            }
            Err(e) => return Err(anyhow::anyhow!("Resolution error: {}", e)),
        };

        // 3. Dispatch Action -> produce EvalResult
        let (result, domain_warnings, steps, solve_steps) = match req.action {
            EvalAction::Simplify => {
                // Determine effective context mode for this request
                let effective_opts = self.effective_options(&state.options, resolved);

                // Get cached profile (or build once and cache)
                let profile = state.profile_cache.get_or_build(&effective_opts);

                // Create simplifier from cached profile
                let mut ctx_simplifier = Simplifier::from_profile(profile);
                // Transfer the context (expressions)
                ctx_simplifier.context = std::mem::take(&mut self.simplifier.context);

                // Use simplify_with_stats to respect expand_policy and other options
                let simplify_opts = effective_opts.to_simplify_options();
                let (mut res, steps, _stats) =
                    ctx_simplifier.simplify_with_stats(resolved, simplify_opts);

                if effective_opts.const_fold == crate::const_fold::ConstFoldMode::Safe {
                    let mut budget = crate::budget::Budget::preset_cli();
                    let cfg = crate::semantics::EvalConfig::default();
                    if let Ok(fold_result) = crate::const_fold::fold_constants(
                        &mut ctx_simplifier.context,
                        res,
                        &cfg,
                        effective_opts.const_fold,
                        &mut budget,
                    ) {
                        res = fold_result.expr;
                    }
                }

                // Transfer context back
                self.simplifier.context = ctx_simplifier.context;

                // Collect domain assumptions from steps with deduplication
                let mut warnings = collect_domain_warnings(&steps);

                // Add warning if i is used in RealOnly mode
                if effective_opts.value_domain == crate::semantics::ValueDomain::RealOnly
                    && crate::helpers::contains_i(&self.simplifier.context, resolved)
                {
                    let i_warning = DomainWarning {
                        message:
                            "To use complex arithmetic (i² = -1), run: semantics set value complex"
                                .to_string(),
                        rule_name: "Imaginary Usage Warning".to_string(),
                    };
                    // Only add if not already present
                    if !warnings.iter().any(|w| w.message == i_warning.message) {
                        warnings.push(i_warning);
                    }
                }

                (EvalResult::Expr(res), warnings, steps, vec![])
            }
            EvalAction::Expand => {
                // Treating Expand as Simplify for now, as Simplifier has no explicit expand mode yet exposed cleanly
                let (res, steps) = self.simplifier.simplify(resolved);
                let warnings = collect_domain_warnings(&steps);
                (EvalResult::Expr(res), warnings, steps, vec![])
            }
            EvalAction::Solve { var } => {
                // Construct proper Equation for solver
                // If resolved is "Equal(lhs, rhs)", use that.
                // Otherwise assume "resolved == 0".

                // We must peek at the resolved expression structure
                let eq_to_solve = match self.simplifier.context.get(resolved) {
                    Expr::Function(name, args) if name == "Equal" && args.len() == 2 => {
                        Equation {
                            lhs: args[0],
                            rhs: args[1],
                            op: RelOp::Eq, // Assuming strict equality for Solve for now
                        }
                    }
                    Expr::Function(name, args) if name == "Less" && args.len() == 2 => Equation {
                        lhs: args[0],
                        rhs: args[1],
                        op: RelOp::Lt,
                    },
                    Expr::Function(name, args) if name == "Greater" && args.len() == 2 => {
                        Equation {
                            lhs: args[0],
                            rhs: args[1],
                            op: RelOp::Gt,
                        }
                    }
                    // Handle other ops if needed, or default to Expr = 0
                    _ => {
                        use num_traits::Zero;
                        let zero = self
                            .simplifier
                            .context
                            .add(Expr::Number(num_rational::BigRational::zero()));
                        Equation {
                            lhs: resolved,
                            rhs: zero,
                            op: RelOp::Eq,
                        }
                    }
                };

                // Call solver
                let sol_result = crate::solver::solve(&eq_to_solve, &var, &mut self.simplifier);

                match sol_result {
                    Ok((solution_set, solve_steps)) => {
                        // Convert SolutionSet to EvalResult::Set (of ExprIds)
                        // If Discrete, straightforward. If Continuous/AllReals, might need representation.
                        // EvalResult::Set expects Vec<ExprId>.
                        // SolutionSet::Discrete(Vec<ExprId>).
                        // SolutionSet::AllReals -> no standard Expr representation yet? Or maybe a special Expr?
                        // Let's rely on what solve produces.

                        use cas_ast::SolutionSet;
                        // For Solve, currently no domain warnings in steps
                        let warnings: Vec<DomainWarning> = vec![];

                        let eval_res = match solution_set {
                            SolutionSet::Discrete(sols) => EvalResult::Set(sols),
                            SolutionSet::AllReals => {
                                // Return empty list or special token?
                                // For now let's return EvalResult::None or handle via text output in CLI?
                                // The user requested EvalResult::Set(Vec<ExprId>).
                                // Let's create a special expression "AllReals"?
                                let id = self
                                    .simplifier
                                    .context
                                    .add(Expr::Variable("All Reals".to_string())); // Hacky
                                EvalResult::Set(vec![id])
                            }
                            SolutionSet::Empty => EvalResult::Set(vec![]),
                            SolutionSet::Continuous(interval) => {
                                // Convert interval to Expr representation?
                                // Interval has min, max.
                                // Let's return min and max in result set for now?? No that's confusing.
                                // Let's return the simplified interval boundaries as expressions.
                                // Or create a Tuple expr.
                                // "EvalResult::Set" implies discrete solutions.
                                // But User said "EvalResult::Set(Vec<ExprId>)".
                                // I'll return the interval bounds as 2 elements if continuous.
                                EvalResult::Set(vec![interval.min, interval.max])
                            }
                            SolutionSet::Union(intervals) => {
                                let mut bounds = Vec::new();
                                for interval in intervals {
                                    bounds.push(interval.min);
                                    bounds.push(interval.max);
                                }
                                EvalResult::Set(bounds)
                            }
                        };
                        (eval_res, warnings, vec![], solve_steps)
                    }
                    Err(e) => return Err(anyhow::anyhow!("Solver error: {}", e)),
                }
            }
            EvalAction::Equiv { other } => {
                let resolved_other = match state.resolve_all(&mut self.simplifier.context, other) {
                    Ok(r) => r,
                    Err(ResolveError::CircularReference(msg)) => {
                        return Err(anyhow::anyhow!(
                            "Circular reference detected in other: {}",
                            msg
                        ))
                    }
                    Err(e) => return Err(anyhow::anyhow!("Resolution error in other: {}", e)),
                };

                let are_eq = self.simplifier.are_equivalent(resolved, resolved_other);
                (EvalResult::Bool(are_eq), vec![], vec![], vec![])
            }
        };

        Ok(EvalOutput {
            stored_id,
            parsed: req.parsed,
            resolved,
            result,
            domain_warnings,
            steps,
            solve_steps,
        })
    }
}
