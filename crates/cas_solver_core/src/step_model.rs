//! Shared step-trace data model.

use crate::soundness_label::SoundnessLabel;
use crate::step_types::{ImportanceLevel, PathStep, StepCategory, SubStep};
use cas_ast::ExprId;
use smallvec::SmallVec;
use smol_str::SmolStr;

/// Didactic metadata for a step.
///
/// Boxed behind `Option<Box<StepMeta>>` in `Step` so that the core struct
/// stays small and the allocation is skipped when not needed.
#[derive(Debug, Clone, Default)]
pub struct StepMeta {
    /// Path from root to the transformed node (kept for debugging/reference).
    pub path: SmallVec<[PathStep; 8]>,
    /// String representation of after (for display).
    pub after_str: Option<String>,
    /// Optional: specific pattern matched for "Rule: X -> Y" display.
    pub before_local: Option<ExprId>,
    /// Optional: specific result of the pattern.
    pub after_local: Option<ExprId>,
    /// Structured assumption events propagated from rewrite metadata.
    pub assumption_events: smallvec::SmallVec<[crate::assumption_model::AssumptionEvent; 1]>,
    /// Required conditions for validity (implicit-domain preservation).
    pub required_conditions: Vec<crate::domain_condition::ImplicitCondition>,
    /// Optional polynomial proof data for identity cancellation.
    pub poly_proof: Option<cas_math::multipoly_display::PolynomialProofData>,
    /// True if this step was created from a chained rewrite.
    pub is_chained: bool,
    /// Educational sub-steps explaining rule application.
    pub substeps: Vec<SubStep>,
}

#[derive(Debug, Clone)]
pub struct Step {
    pub description: SmolStr,
    pub rule_name: SmolStr,
    /// The local expression before transformation.
    pub before: ExprId,
    /// The local expression after transformation.
    pub after: ExprId,
    /// Complete root expression before this step's transformation.
    pub global_before: Option<ExprId>,
    /// Complete root expression after this step's transformation.
    pub global_after: Option<ExprId>,
    /// Importance level for step filtering.
    pub importance: ImportanceLevel,
    /// Category of step for grouping.
    pub category: StepCategory,
    /// Mathematical soundness classification for this transformation.
    pub soundness: SoundnessLabel,
    /// Didactic metadata — lazily allocated, only present when steps are displayed.
    pub meta: Option<Box<StepMeta>>,
}

impl cas_formatter::DisplayStepLike for Step {
    fn rule_name(&self) -> &str {
        &self.rule_name
    }

    fn before(&self) -> cas_ast::ExprId {
        self.before
    }

    fn after(&self) -> cas_ast::ExprId {
        self.after
    }

    fn global_before(&self) -> Option<cas_ast::ExprId> {
        self.global_before
    }

    fn global_after(&self) -> Option<cas_ast::ExprId> {
        self.global_after
    }
}

impl Step {
    /// Path from root to the transformed node.
    #[inline]
    pub fn path(&self) -> &[PathStep] {
        self.meta.as_ref().map_or(&[], |m| &m.path)
    }

    /// String representation of the after expression.
    #[inline]
    pub fn after_str(&self) -> Option<&str> {
        self.meta.as_ref().and_then(|m| m.after_str.as_deref())
    }

    /// Focused before sub-expression (for "Rule: X -> Y" display).
    #[inline]
    pub fn before_local(&self) -> Option<ExprId> {
        self.meta.as_ref().and_then(|m| m.before_local)
    }

    /// Focused after sub-expression (for "Rule: X -> Y" display).
    #[inline]
    pub fn after_local(&self) -> Option<ExprId> {
        self.meta.as_ref().and_then(|m| m.after_local)
    }

    /// Structured assumption events.
    #[inline]
    pub fn assumption_events(&self) -> &[crate::assumption_model::AssumptionEvent] {
        self.meta.as_ref().map_or(&[], |m| &m.assumption_events)
    }

    /// Required conditions for validity.
    #[inline]
    pub fn required_conditions(&self) -> &[crate::domain_condition::ImplicitCondition] {
        self.meta.as_ref().map_or(&[], |m| &m.required_conditions)
    }

    /// Polynomial proof data for identity cancellation.
    #[inline]
    pub fn poly_proof(&self) -> Option<&cas_math::multipoly_display::PolynomialProofData> {
        self.meta.as_ref().and_then(|m| m.poly_proof.as_ref())
    }

    /// Whether this step was created from a chained rewrite.
    #[inline]
    pub fn is_chained(&self) -> bool {
        self.meta.as_ref().is_some_and(|m| m.is_chained)
    }

    /// Educational sub-steps.
    #[inline]
    pub fn substeps(&self) -> &[SubStep] {
        self.meta.as_ref().map_or(&[], |m| &m.substeps)
    }

    /// Get a mutable reference to the meta, creating it if absent.
    #[inline]
    pub fn meta_mut(&mut self) -> &mut StepMeta {
        self.meta
            .get_or_insert_with(|| Box::new(StepMeta::default()))
    }
}

impl Step {
    pub fn new(
        description: &str,
        rule_name: &str,
        before: ExprId,
        after: ExprId,
        path: impl Into<SmallVec<[PathStep; 8]>>,
        _context: Option<&cas_ast::Context>,
    ) -> Self {
        let path = path.into();
        Self {
            description: SmolStr::new(description),
            rule_name: SmolStr::new(rule_name),
            before,
            after,
            global_before: None,
            global_after: None,
            importance: ImportanceLevel::Low,
            category: StepCategory::General,
            soundness: SoundnessLabel::Equivalence,
            meta: Some(Box::new(StepMeta {
                path,
                ..Default::default()
            })),
        }
    }

    /// Create a compact step without display formatting (for StepsMode::Compact).
    pub fn new_compact(description: &str, rule_name: &str, before: ExprId, after: ExprId) -> Self {
        Self {
            description: SmolStr::new(description),
            rule_name: SmolStr::new(rule_name),
            before,
            after,
            global_before: None,
            global_after: None,
            importance: ImportanceLevel::Low,
            category: StepCategory::General,
            soundness: SoundnessLabel::Equivalence,
            meta: None,
        }
    }

    /// Create a step with complete global snapshots before and after transformation.
    #[allow(clippy::too_many_arguments)]
    pub fn with_snapshots(
        description: &str,
        rule_name: &str,
        before: ExprId,
        after: ExprId,
        path: impl Into<SmallVec<[PathStep; 8]>>,
        context: Option<&cas_ast::Context>,
        global_before: ExprId,
        global_after: ExprId,
    ) -> Self {
        let mut step = Self::new(description, rule_name, before, after, path, context);
        step.global_before = Some(global_before);
        step.global_after = Some(global_after);
        step
    }

    /// Get the importance/significance of this step.
    pub fn get_importance(&self) -> ImportanceLevel {
        if self.before == self.after {
            return ImportanceLevel::Trivial;
        }
        if !self.assumption_events().is_empty() {
            return ImportanceLevel::Medium;
        }
        self.importance
    }
}
