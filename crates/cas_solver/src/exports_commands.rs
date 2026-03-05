pub use crate::budget_runtime_types::{Budget, BudgetExceeded, Metric, Operation, PassStats};
pub use crate::display_eval_steps::DisplayEvalSteps;
pub use crate::engine_runtime_types::{
    AutoExpandBinomials, BranchMode, CasError, ComplexMode, ContextMode, Engine, EvalAction,
    EvalOptions, EvalOutput, EvalRequest, EvalResult, HeuristicPoly, RuleProfiler,
    SharedSemanticConfig, Simplifier, SimplifyOptions, StepsMode,
};
pub use crate::error_runtime_types::error;
pub use crate::phase_runtime_types::{
    ExpandBudget, ExpandPolicy, PhaseBudgets, PhaseMask, PhaseStats, PipelineStats, SimplifyPhase,
};
pub use crate::rule_runtime_types::{
    LogExpansionRule, Orchestrator, ParentContext, Rewrite, Rule, SimpleRule,
};
pub use crate::rules_runtime_types::rules;
pub use crate::semantics_command_eval::evaluate_semantics_command_line;
pub use crate::semantics_command_parse::parse_semantics_command_input;
pub use crate::semantics_command_types::{SemanticsCommandInput, SemanticsCommandOutput};
pub use crate::semantics_preset_apply::{
    apply_semantics_preset_by_name, apply_semantics_preset_by_name_to_options,
    apply_semantics_preset_state_to_options, evaluate_semantics_preset_args_to_options,
    semantics_preset_state_from_options,
};
pub use crate::semantics_preset_catalog::{find_semantics_preset, semantics_presets};
pub use crate::semantics_preset_format::{
    format_semantics_preset_application_lines, format_semantics_preset_help_lines,
    format_semantics_preset_list_lines,
};
pub use crate::semantics_preset_types::{
    SemanticsPreset, SemanticsPresetApplication, SemanticsPresetApplyError,
    SemanticsPresetCommandOutput, SemanticsPresetState,
};
pub use crate::semantics_set_apply::{
    apply_semantics_set_args_to_options, apply_semantics_set_state_to_options,
    evaluate_semantics_set_args_to_overview_lines,
};
pub use crate::semantics_set_parse::evaluate_semantics_set_args;
pub use crate::semantics_set_types::{semantics_set_state_from_options, SemanticsSetState};
pub use crate::semantics_view_format::{
    format_semantics_axis_lines, format_semantics_overview_lines,
    format_semantics_unknown_subcommand_message, semantics_help_message,
};
pub use crate::semantics_view_types::{semantics_view_state_from_options, SemanticsViewState};
pub use crate::set_command_apply::apply_set_command_plan;
pub use crate::set_command_eval::evaluate_set_command_input;
pub use crate::set_command_format::{format_set_help_text, format_set_option_value};
pub use crate::set_command_parse::parse_set_command_input;
pub use crate::set_command_types::{
    SetCommandApplyEffects, SetCommandInput, SetCommandPlan, SetCommandResult, SetCommandState,
    SetDisplayMode,
};
pub use crate::show_command::{
    evaluate_show_command_lines, evaluate_show_command_lines_with, ShowCommandContext,
};
pub use crate::simplifier_setup_build::build_simplifier_with_rule_config;
pub use crate::simplifier_setup_toggle::apply_simplifier_toggle_config;
pub use crate::simplifier_setup_types::{
    set_simplifier_toggle_rule, SimplifierRuleConfig, SimplifierToggleConfig,
};
pub use crate::solution_display::{
    display_interval, display_solution_set, is_pure_residual_otherwise,
};
pub use crate::solve_command_errors::{
    format_solve_command_error_message, format_solve_prepare_error_message,
};
pub use crate::solve_command_eval_core::{
    evaluate_solve_command_with_session, prepare_solve_eval_request, PreparedSolveEvalRequest,
    SolveCommandEvalError, SolveCommandEvalOutput,
};
pub use crate::solve_command_session_eval::{
    evaluate_solve_command_lines_with_session, evaluate_solve_command_message_with_session,
};
pub use crate::solve_display_lines::format_solve_command_eval_lines;
pub use crate::solve_display_result::{format_solve_result_line, requires_result_expr_anchor};
pub use crate::solve_display_steps::format_solve_steps_lines;
pub use crate::solve_input_parse_parse::{
    parse_solve_command_input, parse_solve_invocation_check, parse_timeline_command_input,
};
pub use crate::solve_input_parse_prepare::{
    prepare_solve_expr_and_var, prepare_timeline_solve_equation, resolve_solve_var,
};
pub use crate::solve_input_types::{SolveCommandInput, SolvePrepareError, TimelineCommandInput};
pub use crate::solve_render_config::{
    solve_render_config_from_eval_options, SolveCommandRenderConfig, SolveDisplayMode,
};
pub use crate::solve_safety::{RequirementDescriptor, RuleSolveSafetyExt, SolveSafety};
pub use crate::solve_verify_display::format_verify_summary_lines;
pub use crate::solver_entrypoints::*;
pub use crate::standard_oracle::{oracle_allows_with_hint, StandardOracle};
pub use crate::step_runtime_types::{ImportanceLevel, PathStep, Step, StepCategory};
pub use crate::steps_command_eval::{apply_steps_command_update, evaluate_steps_command_input};
pub use crate::steps_command_format::{
    format_steps_collection_set_message, format_steps_current_message,
    format_steps_display_set_message, format_steps_unknown_mode_message,
};
pub use crate::steps_command_parse::parse_steps_command_input;
pub use crate::steps_command_types::{
    StepsCommandApplyEffects, StepsCommandInput, StepsCommandResult, StepsCommandState,
    StepsDisplayMode,
};
pub use crate::substitute::{
    detect_substitute_strategy, substitute_auto, substitute_auto_with_strategy,
    substitute_power_aware, substitute_with_steps, SubstituteOptions, SubstituteStrategy,
};
pub use crate::substitute_command_eval::{
    evaluate_substitute_command_lines, evaluate_substitute_invocation_lines,
    evaluate_substitute_invocation_message, evaluate_substitute_invocation_user_message,
};
pub use crate::substitute_command_format::{
    format_substitute_eval_lines, format_substitute_parse_error_message,
    substitute_render_mode_from_display_mode,
};
pub use crate::substitute_command_types::{
    SubstituteEvalOutput, SubstituteParseError, SubstituteRenderMode, SubstituteSimplifyEvalOutput,
};
pub use crate::substitute_subcommand_eval::evaluate_substitute_subcommand;
pub use crate::substitute_subcommand_json::evaluate_substitute_subcommand_json_canonical;
pub use crate::substitute_subcommand_text::parse_substitute_json_text_lines;
pub use crate::substitute_subcommand_types::{SubstituteCommandMode, SubstituteSubcommandOutput};
pub use crate::symbolic_transforms::{apply_weierstrass_recursive, expand_log_recursive};
pub use crate::telescoping::{telescope, TelescopingResult, TelescopingStep};
pub use crate::timeline_command_eval::evaluate_timeline_command_with_session;
pub use crate::timeline_types::{
    TimelineCommandEvalError, TimelineCommandEvalOutput, TimelineSimplifyEvalError,
    TimelineSimplifyEvalOutput, TimelineSolveEvalError, TimelineSolveEvalOutput,
};
pub use crate::types::{
    DisplaySolveSteps, SolveCtx, SolveDiagnostics, SolveDomainEnv, SolveStep, SolveSubStep,
    SolverOptions,
};
pub use crate::unary_command_eval::{
    evaluate_unary_command_lines, evaluate_unary_command_message,
    evaluate_unary_function_command_lines,
};
pub use crate::unary_display::format_unary_function_eval_lines;
pub use crate::vars_command_display::{
    evaluate_vars_command_lines, evaluate_vars_command_lines_with_context,
};
pub use crate::weierstrass_command::{
    evaluate_weierstrass_command_lines, evaluate_weierstrass_invocation_lines,
    evaluate_weierstrass_invocation_message, parse_weierstrass_invocation_input,
    weierstrass_usage_message,
};
pub use cas_ast::ordering::compare_expr;
pub use cas_ast::target_kind;
pub use cas_formatter::visualizer;
pub use cas_math::canonical_forms;
pub use cas_math::evaluator_f64::{
    eval_f64, eval_f64_checked, EvalCheckedError, EvalCheckedOptions,
};
pub use cas_math::expr_nary::{add_terms_no_sign, add_terms_signed, Sign};
pub use cas_math::expr_predicates::is_zero_expr as is_zero;
pub use cas_math::factor::factor;
pub use cas_math::limit_types::{Approach, LimitOptions, PreSimplifyMode};
pub use cas_math::number_theory_support::GcdResult;
pub use cas_math::pattern_marks;
pub use cas_math::poly_store::{try_get_poly_result_term_count, try_render_poly_result};
pub use cas_math::rationalize::{rationalize_denominator, RationalizeConfig, RationalizeResult};
pub use cas_math::rationalize_policy::{AutoRationalizeLevel, RationalizeOutcome};
pub use cas_math::telescoping_dirichlet::{
    try_dirichlet_kernel_identity as try_dirichlet_kernel_identity_pub, DirichletKernelResult,
};
pub use cas_session_core::eval::{EvalSession, EvalStore};
pub use cas_solver_core::assume_scope::AssumeScope;
pub use cas_solver_core::assumption_model::AssumptionRecord;
pub use cas_solver_core::assumption_model::{
    assumption_condition_text, assumption_key_dedupe_fingerprint, blocked_hint_suggestion,
    collect_assumption_records, collect_assumption_records_from_iter, collect_blocked_hint_items,
    format_assumption_records_conditions, format_assumption_records_section_lines,
    format_blocked_hint_condition, format_blocked_simplifications_section_lines,
    group_blocked_hint_conditions_by_rule, AssumptionCollector, AssumptionEvent, AssumptionKey,
    AssumptionKind,
};
pub use cas_solver_core::assumption_reporting::AssumptionReporting;
pub use cas_solver_core::blocked_hint::BlockedHint;
pub use cas_solver_core::blocked_hint_store::{
    clear_blocked_hints, register_blocked_hint, take_blocked_hints,
};
pub use cas_solver_core::const_fold_types::{ConstFoldMode, ConstFoldResult};
pub use cas_solver_core::diagnostics_model::{Diagnostics, RequireOrigin, RequiredItem};
pub use cas_solver_core::domain_assumption_classification::classify_assumption;
pub use cas_solver_core::domain_cancel_decision::CancelDecision;
pub use cas_solver_core::domain_condition::{
    filter_requires_for_display, ImplicitCondition, ImplicitDomain, RequiresDisplayLevel,
};
pub use cas_solver_core::domain_context::DomainContext;
pub use cas_solver_core::domain_facts_model::{DomainFact, FactStrength, Predicate};
pub use cas_solver_core::domain_inference::{AnalyticExpansionResult, DomainDelta};
pub use cas_solver_core::domain_inference_counter::{
    get as infer_domain_calls_get, reset as infer_domain_calls_reset,
};
pub use cas_solver_core::domain_mode::DomainMode;
pub use cas_solver_core::domain_normalization::{
    normalize_and_dedupe_conditions, normalize_condition, normalize_condition_expr,
    render_conditions_normalized,
};
pub use cas_solver_core::domain_oracle_model::DomainOracle;
pub use cas_solver_core::domain_proof::Proof;
pub use cas_solver_core::domain_warning::DomainWarning;
pub use cas_solver_core::engine_events::{EngineEvent, StepListener};
pub use cas_solver_core::equivalence::EquivalenceResult;
pub use cas_solver_core::eval_config::EvalConfig;
pub use cas_solver_core::inverse_trig_policy::InverseTrigPolicy;
pub use cas_solver_core::isolation_utils::contains_var;
pub use cas_solver_core::solve_budget::SolveBudget;
pub use cas_solver_core::solve_infer::infer_solve_variable;
pub use cas_solver_core::solve_safety_policy::ConditionClass;
pub use cas_solver_core::solve_safety_policy::ProvenanceKind as Provenance;
pub use cas_solver_core::solve_safety_policy::SimplifyPurpose;
pub use cas_solver_core::verification::{VerifyResult, VerifyStatus, VerifySummary};
pub use cas_solver_core::verify_stats;
pub use cas_solver_core::{branch_policy::BranchPolicy, value_domain::ValueDomain};
