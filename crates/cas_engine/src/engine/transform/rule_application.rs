//! Rule application: `apply_rules` and `build_parent_context`.
//!
//! Contains the 303-line rule matching loop that iterates over specific and global
//! rule buckets, applying phase filtering, solve-safety checks, semantic equality,
//! anti-worsen budget, domain airbag, cycle detection, and profiling.

use super::*;
use crate::canonical_forms::normalize_core_with_cache;
use cas_solver_core::solve_safety_policy::{safe_for_prepass, safe_for_tactic_with_domain_flags};

impl<'a> LocalSimplificationTransformer<'a> {
    /// Build the ParentContext for the current node position.
    ///
    /// PERF: This is called once per `apply_rules()` invocation (i.e., once per node),
    /// and the result is shared by all rule attempts on that node. Previously, this
    /// construction was duplicated inside both the specific-rules and global-rules loops,
    /// causing O(rules × nodes × ancestors) redundant work.
    fn build_parent_context(&self) -> crate::parent_context::ParentContext {
        self.initial_parent_ctx
            .clone()
            .with_runtime_ancestors(&self.ancestor_stack, self.context)
    }

    /// Returns `true` if the rule should be skipped due to disabled-list,
    /// phase mismatch, or (optionally) solve-safety constraints.
    ///
    /// `check_solve_safety`: `true` for specific rules, `false` for global rules
    /// (the original code only applied solve-safety filtering to specific rules).
    fn should_skip_rule(&mut self, rule: &dyn crate::rule::Rule, check_solve_safety: bool) -> bool {
        if !self.phase_prefiltered && self.disabled_rules.contains(rule.name()) {
            self.profiler
                .record_rejected_disabled(self.current_phase, rule.name());
            return true;
        }
        if !self.phase_prefiltered {
            let phase_mask = self.current_phase.mask();
            if !rule.allowed_phases().contains(phase_mask) {
                self.profiler
                    .record_rejected_phase(self.current_phase, rule.name());
                return true;
            }
        }
        if check_solve_safety {
            match self.simplify_purpose {
                crate::SimplifyPurpose::Eval => {}
                crate::SimplifyPurpose::SolvePrepass => {
                    if !safe_for_prepass(rule.solve_safety()) {
                        return true;
                    }
                }
                crate::SimplifyPurpose::SolveTactic => {
                    let domain_mode = self.initial_parent_ctx.domain_mode();
                    if !safe_for_tactic_with_domain_flags(
                        rule.solve_safety(),
                        matches!(domain_mode, crate::DomainMode::Assume),
                        matches!(domain_mode, crate::DomainMode::Strict),
                    ) {
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
                        let is_didactic_rule =
                            cas_solver_core::step_rules::is_always_keep_step_rule_name(rule.name());

                        if !is_didactic_rule {
                            use cas_math::semantic_equality::SemanticEqualityChecker;
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
                            && cas_math::expr_complexity::rewrite_worsens_too_much(
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
                            let before =
                                cas_math::expr_nf_scoring::count_all_nodes(self.context, expr_id);
                            let after = cas_math::expr_nf_scoring::count_all_nodes(
                                self.context,
                                rewrite.new_expr,
                            );
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
                                let node_count_before = cas_math::expr_complexity::node_count_tree(
                                    self.context,
                                    expr_id,
                                );
                                let node_count_after = cas_math::expr_complexity::node_count_tree(
                                    self.context,
                                    rewrite.new_expr,
                                );
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
                        expr_id = self.record_rewrite_step(rule.as_ref(), expr_id, rewrite);

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
                            self.cycle_detector =
                                Some(cas_solver_core::cycle_detection::CycleDetector::new(
                                    self.current_phase,
                                ));
                            self.cycle_phase = Some(self.current_phase);
                            self.fp_memo.clear();
                            // Note: blocked_rules persists across phases (conservative)
                        }

                        let h = cas_solver_core::cycle_detection::expr_fingerprint(
                            self.context,
                            expr_id,
                            &mut self.fp_memo,
                        );
                        if let Some(detector) = self.cycle_detector.as_mut() {
                            if let Some(info) = detector.observe(h) {
                                // Emit cycle event for the registry
                                cas_solver_core::cycle_event_registry::register_cycle_event_for_expr(
                                    self.context,
                                    expr_id,
                                    self.current_phase,
                                    info.period,
                                    cas_solver_core::cycle_models::CycleLevel::IntraNode,
                                    rule.name(),
                                    h,
                                    info.at_step,
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
                                        crate::register_blocked_hint(crate::BlockedHint {
                                        key: crate::AssumptionKey::Defined {
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
                        cas_solver_core::step_rules::is_always_keep_step_rule_name(rule.name());

                    if !is_didactic_rule {
                        use cas_math::semantic_equality::SemanticEqualityChecker;
                        let checker = SemanticEqualityChecker::new(self.context);
                        if checker.are_equal(expr_id, rewrite.new_expr) {
                            // Provably equal - only accept if it improves normal form
                            if !cas_math::expr_nf_scoring::nf_score_after_is_better(
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
                    expr_id = self.record_rewrite_step(rule.as_ref(), expr_id, rewrite);

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
