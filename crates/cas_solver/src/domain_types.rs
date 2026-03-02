/// Classification of side conditions for domain-mode gating.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConditionClass {
    /// Small holes (for example, non-zero denominator).
    Definability,
    /// Bigger analytic restrictions (for example, positivity/ranges).
    Analytic,
}

pub(crate) fn condition_class_from_engine(value: cas_engine::ConditionClass) -> ConditionClass {
    match value {
        cas_engine::ConditionClass::Definability => ConditionClass::Definability,
        cas_engine::ConditionClass::Analytic => ConditionClass::Analytic,
    }
}

impl From<cas_engine::ConditionClass> for ConditionClass {
    fn from(value: cas_engine::ConditionClass) -> Self {
        condition_class_from_engine(value)
    }
}

/// Origin of a domain fact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Provenance {
    /// Inherited from input structure.
    Intrinsic,
    /// Structurally proven by the engine.
    Proven,
    /// Assumed by mode policy or user choice.
    Assumed,
    /// Introduced as a side condition by a rule.
    Introduced,
}

impl From<cas_engine::Provenance> for Provenance {
    fn from(value: cas_engine::Provenance) -> Self {
        match value {
            cas_engine::Provenance::Intrinsic => Self::Intrinsic,
            cas_engine::Provenance::Proven => Self::Proven,
            cas_engine::Provenance::Assumed => Self::Assumed,
            cas_engine::Provenance::Introduced => Self::Introduced,
        }
    }
}
