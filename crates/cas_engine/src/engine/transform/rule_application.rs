//! Rule application: `apply_rules` and `build_parent_context`.
//!
//! Contains the 303-line rule matching loop that iterates over specific and global
//! rule buckets, applying phase filtering, solve-safety checks, semantic equality,
//! anti-worsen budget, domain airbag, cycle detection, and profiling.

use super::*;
use crate::canonical_forms::normalize_core_with_cache;

impl<'a> LocalSimplificationTransformer<'a> {
    /// Build the ParentContext for the current node position.
    ///
    /// PERF: This is called once per `apply_rules()` invocation (i.e., once per node),
    /// and the result is shared by all rule attempts on that node. Previously, this
    /// construction was duplicated inside both the specific-rules and global-rules loops,
    /// causing O(rules × nodes × ancestors) redundant work.
    fn build_parent_context(&self) -> crate::parent_context::ParentContext {
        let mut ctx = if let Some(marks) = self.initial_parent_ctx.pattern_marks() {
            crate::parent_context::ParentContext::with_marks(marks.clone())
        } else {
            crate::parent_context::ParentContext::root()
        };
        // Copy expand_mode (enables BinomialExpansionRule when Simplifier::expand() is called)
        if self.initial_parent_ctx.is_expand_mode() {
            ctx = ctx.with_expand_mode_flag(true);
        }
        // Copy auto_expand (enables AutoExpandPowSumRule)
        if self.initial_parent_ctx.is_auto_expand() {
            ctx = ctx
                .with_auto_expand_flag(true, self.initial_parent_ctx.auto_expand_budget().cloned());
        }
        ctx = ctx.with_domain_mode(self.initial_parent_ctx.domain_mode());
        ctx = ctx.with_inv_trig(self.initial_parent_ctx.inv_trig_policy());
        ctx = ctx.with_value_domain(self.initial_parent_ctx.value_domain());
        ctx = ctx.with_goal(self.initial_parent_ctx.goal());
        if let Some(root) = self.initial_parent_ctx.root_expr() {
            ctx = ctx.with_root_expr_only(root);
        }
        // Propagate context_mode and simplify_purpose for Solve mode blocking
        ctx = ctx.with_context_mode(self.initial_parent_ctx.context_mode());
        ctx = ctx.with_simplify_purpose(self.initial_parent_ctx.simplify_purpose());
        // Propagate implicit_domain for domain-aware simplifications
        ctx = ctx.with_implicit_domain(self.initial_parent_ctx.implicit_domain().cloned());
        // Build ancestor chain from stack (for Div tracking)
        for &ancestor in &self.ancestor_stack {
            ctx = ctx.extend_with_div_check(ancestor, self.context);
        }
        ctx = ctx.with_autoexpand_binomials(self.initial_parent_ctx.autoexpand_binomials());
        ctx = ctx.with_heuristic_poly(self.initial_parent_ctx.heuristic_poly());
        ctx
    }

    /// Returns `true` if the rule should be skipped due to disabled-list,
    /// phase mismatch, or (optionally) solve-safety constraints.
    ///
    /// `check_solve_safety`: `true` for specific rules, `false` for global rules
    /// (the original code only applied solve-safety filtering to specific rules).
    fn should_skip_rule(&mut self, rule: &dyn crate::rule::Rule, check_solve_safety: bool) -> bool {
        if self.disabled_rules.contains(rule.name()) {
            self.profiler
                .record_rejected_disabled(self.current_phase, rule.name());
            return true;
        }
        let phase_mask = self.current_phase.mask();
        if !rule.allowed_phases().contains(phase_mask) {
            self.profiler
                .record_rejected_phase(self.current_phase, rule.name());
            return true;
        }
        if check_solve_safety {
            match self.simplify_purpose {
                crate::solve_safety::SimplifyPurpose::Eval => {}
                crate::solve_safety::SimplifyPurpose::SolvePrepass => {
                    if !rule.solve_safety().safe_for_prepass() {
                        return true;
                    }
                }
                crate::solve_safety::SimplifyPurpose::SolveTactic => {
                    let domain_mode = self.initial_parent_ctx.domain_mode();
                    if !rule.solve_safety().safe_for_tactic(domain_mode) {
                        return true;
                    }
                }
            }
        }
        false
    }

    pub(super) fn apply_rules(&mut self, mut expr_id: ExprId) -> ExprId {
        // Note: This loop pattern with early returns is intentional for structured exit points
        #[allow(clippy::never_loop)]
        loop {
            let mut changed = false;
            let target_kind = crate::target_kind::TargetKind::from_expr(self.context.get(expr_id));

            // PERF: Build ParentContext ONCE per node, shared by all rules.
            // Previously this was rebuilt inside the rule loop (O(rules × nodes × ancestors)).
            let parent_ctx = self.build_parent_context();

            // println!("apply_rules for {:?} variant: {}", expr_id, variant);
            // Try specific rules
            if let Some(specific_rules) = self.rules.get(&target_kind) {
                for rule in specific_rules {
                    if self.should_skip_rule(rule.as_ref(), true) {
                        continue;
                    }

                    if let Some(mut rewrite) = rule.apply(self.context, expr_id, &parent_ctx) {
                        // Check semantic equality - skip if no real change
                        // EXCEPTION: Didactic rules should always generate steps
                        // even if result is semantically equivalent (e.g., sqrt(12) → 2*√3)
                        let is_didactic_rule = rule.name() == "Evaluate Numeric Power"
                            || rule.name() == "Sum Exponents";

                        if !is_didactic_rule {
                            use crate::semantic_equality::SemanticEqualityChecker;
                            let checker = SemanticEqualityChecker::new(self.context);
                            if checker.are_equal(expr_id, rewrite.new_expr) {
                                debug!(
                                    "{}[DEBUG] Rule '{}' produced semantically equal result, skipping",
                                    self.indent(),
                                    rule.name()
                                );
                                self.profiler
                                    .record_rejected_semantic(self.current_phase, rule.name());
                                continue;
                            }
                        }

                        // ANTI-WORSEN BUDGET: Reject rewrites that grow expression beyond threshold.
                        // This is a GLOBAL SAFETY NET against exponential explosion (e.g., sin(16*x) expansion).
                        // Budget policy: Block if BOTH:
                        // - Absolute growth > 30 nodes
                        // - Relative growth > 1.5x (50% larger)
                        // Exception: expand_mode bypasses this check (user explicitly requested expansion)
                        // Exception: budget_exempt rewrites bypass this check (intentional expansion rules)
                        if !parent_ctx.is_expand_mode()
                            && !parent_ctx.in_auto_expand_context()
                            && !rewrite.budget_exempt
                            && crate::helpers::rewrite_worsens_too_much(
                                self.context,
                                expr_id,
                                rewrite.new_expr,
                                30,  // max_growth_abs
                                1.5, // max_growth_ratio
                            )
                        {
                            debug!(
                                "{}[DEBUG] Rule '{}' blocked by anti-worsen budget (expression grew too much)",
                                self.indent(),
                                rule.name()
                            );
                            continue;
                        }

                        // Domain Delta Airbag: skip_unless_analytic=true for specific rules
                        if self.check_domain_airbag(
                            rule.as_ref(),
                            &parent_ctx,
                            expr_id,
                            &mut rewrite,
                            true,
                        ) {
                            continue;
                        }

                        // Record rule application with delta_nodes for health metrics
                        let delta = if self.profiler.is_health_enabled() {
                            let before = crate::helpers::count_all_nodes(self.context, expr_id);
                            let after =
                                crate::helpers::count_all_nodes(self.context, rewrite.new_expr);
                            after as i64 - before as i64
                        } else {
                            0
                        };
                        self.profiler
                            .record_with_delta(self.current_phase, rule.name(), delta);

                        // TRACE: Log applied rules for debugging cycles
                        if *CAS_TRACE_RULES_ENABLED {
                            use std::io::Write;
                            if let Ok(mut f) = std::fs::OpenOptions::new()
                                .create(true)
                                .append(true)
                                .open("/tmp/rule_trace.log")
                            {
                                let node_count_before =
                                    crate::helpers::node_count(self.context, expr_id);
                                let node_count_after =
                                    crate::helpers::node_count(self.context, rewrite.new_expr);
                                let _ = writeln!(
                                    f,
                                    "APPLIED depth={} rule={} nodes={}->{}",
                                    self.current_depth,
                                    rule.name(),
                                    node_count_before,
                                    node_count_after
                                );
                                let _ = f.flush();
                            }
                        }

                        // println!(
                        //     "Rule '{}' applied: {:?} -> {:?}",
                        //     rule.name(),
                        //     expr_id,
                        //     rewrite.new_expr
                        // );
                        debug!(
                            "{}[DEBUG] Rule '{}' applied: {:?} -> {:?}",
                            self.indent(),
                            rule.name(),
                            expr_id,
                            rewrite.new_expr
                        );
                        expr_id = self.record_rewrite_step(rule.as_ref(), expr_id, &rewrite);

                        // Budget tracking: count this rewrite (charged at end of pass)
                        self.rewrite_count += 1;

                        // Note: Rule application tracking for rationalization is now handled by phase, not flag
                        // Apply canonical normalization to prevent loops
                        expr_id = normalize_core_with_cache(
                            self.context,
                            expr_id,
                            &mut self.normalize_cache,
                        );

                        // V2.14.30: Always-On Cycle Detection with blocklist
                        // Reset detector if phase changed since last initialization
                        if self.cycle_phase != Some(self.current_phase) {
                            self.cycle_detector = Some(crate::cycle_detector::CycleDetector::new(
                                self.current_phase,
                            ));
                            self.cycle_phase = Some(self.current_phase);
                            self.fp_memo.clear();
                            // Note: blocked_rules persists across phases (conservative)
                        }

                        let h = crate::cycle_detector::expr_fingerprint(
                            self.context,
                            expr_id,
                            &mut self.fp_memo,
                        );
                        if let Some(detector) = self.cycle_detector.as_mut() {
                            if let Some(info) = detector.observe(h) {
                                // Emit cycle event for the registry
                                let expr_str = format!(
                                    "{}",
                                    cas_ast::DisplayExpr {
                                        context: self.context,
                                        id: expr_id,
                                    }
                                );
                                crate::cycle_events::register_cycle_event(
                                    crate::cycle_events::CycleEvent {
                                        phase: self.current_phase,
                                        period: info.period,
                                        level: crate::cycle_events::CycleLevel::IntraNode,
                                        rule_name: rule.name().to_string(),
                                        expr_fingerprint: h,
                                        expr_display: crate::cycle_events::truncate_display(
                                            &expr_str, 120,
                                        ),
                                        rewrite_step: info.at_step,
                                    },
                                );
                                // Add to blocklist to prevent re-entry
                                let rule_name_static = rule.name();
                                if self.blocked_rules.insert((h, rule_name_static.to_string())) {
                                    // First time seeing this (fingerprint, rule) - emit hint
                                    // But don't emit for constants/numbers (they're not meaningful cycles)
                                    let is_constant = matches!(
                                        self.context.get(expr_id),
                                        cas_ast::Expr::Number(_) | cas_ast::Expr::Constant(_)
                                    );
                                    if !is_constant {
                                        crate::domain::register_blocked_hint(crate::domain::BlockedHint {
                                        key: crate::assumptions::AssumptionKey::Defined {
                                            expr_fingerprint: h,
                                        },
                                        expr_id,
                                        rule: rule_name_static.to_string(),
                                        suggestion: "cycle detected; consider disabling heuristic rules or tightening budget",
                                    });
                                    }
                                }
                                self.last_cycle = Some(info);
                                // Treat as fixed-point: stop this phase early
                                self.current_depth -= 1;
                                return expr_id;
                            }
                        }

                        changed = true;
                        break;
                    }
                }
            }

            if changed {
                return self.transform_expr_recursive(expr_id);
            }

            // Try global rules
            for rule in self.global_rules {
                if self.should_skip_rule(rule.as_ref(), false) {
                    continue;
                }

                // PERF: Reuse the parent_ctx built once per apply_rules() call
                if let Some(mut rewrite) = rule.apply(self.context, expr_id, &parent_ctx) {
                    // Fast path: if rewrite produces identical ExprId, skip entirely
                    if rewrite.new_expr == expr_id {
                        continue;
                    }

                    // Semantic equality check to prevent infinite loops
                    // Skip rewrites that produce semantically equivalent results without improvement
                    let is_didactic_rule =
                        rule.name() == "Evaluate Numeric Power" || rule.name() == "Sum Exponents";

                    if !is_didactic_rule {
                        use crate::semantic_equality::SemanticEqualityChecker;
                        let checker = SemanticEqualityChecker::new(self.context);
                        if checker.are_equal(expr_id, rewrite.new_expr) {
                            // Provably equal - only accept if it improves normal form
                            if !crate::helpers::nf_score_after_is_better(
                                self.context,
                                expr_id,
                                rewrite.new_expr,
                            ) {
                                continue; // Skip - no improvement
                            }
                        }
                    }

                    // Domain Delta Airbag: skip_unless_analytic=false for global rules (unconditional)
                    if self.check_domain_airbag(
                        rule.as_ref(),
                        &parent_ctx,
                        expr_id,
                        &mut rewrite,
                        false,
                    ) {
                        continue;
                    }

                    // Record rule application for profiling
                    self.profiler.record(self.current_phase, rule.name());

                    debug!(
                        "{}[DEBUG] Global Rule '{}' applied: {:?} -> {:?}",
                        self.indent(),
                        rule.name(),
                        expr_id,
                        rewrite.new_expr
                    );
                    expr_id = self.record_rewrite_step(rule.as_ref(), expr_id, &rewrite);

                    // Budget tracking: count this rewrite (charged at end of pass)
                    self.rewrite_count += 1;

                    // Note: Rule application tracking for rationalization is now handled by phase, not flag
                    // Apply canonical normalization to prevent loops
                    expr_id =
                        normalize_core_with_cache(self.context, expr_id, &mut self.normalize_cache);
                    changed = true;
                    break;
                }
            }

            if changed {
                return self.transform_expr_recursive(expr_id);
            }

            self.current_depth -= 1;
            return expr_id;
        }
    }
}
