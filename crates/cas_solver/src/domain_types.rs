/// Classification of side conditions for domain-mode gating.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConditionClass {
    /// Small holes (for example, non-zero denominator).
    Definability,
    /// Bigger analytic restrictions (for example, positivity/ranges).
    Analytic,
}

impl ConditionClass {
    /// Converts to the engine representation.
    #[inline]
    pub fn into_engine(self) -> cas_engine::ConditionClass {
        self.into()
    }
}

impl From<cas_engine::ConditionClass> for ConditionClass {
    fn from(value: cas_engine::ConditionClass) -> Self {
        match value {
            cas_engine::ConditionClass::Definability => Self::Definability,
            cas_engine::ConditionClass::Analytic => Self::Analytic,
        }
    }
}

impl From<ConditionClass> for cas_engine::ConditionClass {
    fn from(value: ConditionClass) -> Self {
        match value {
            ConditionClass::Definability => cas_engine::ConditionClass::Definability,
            ConditionClass::Analytic => cas_engine::ConditionClass::Analytic,
        }
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

impl Provenance {
    /// Converts to the engine representation.
    #[inline]
    pub fn into_engine(self) -> cas_engine::Provenance {
        self.into()
    }
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

impl From<Provenance> for cas_engine::Provenance {
    fn from(value: Provenance) -> Self {
        match value {
            Provenance::Intrinsic => cas_engine::Provenance::Intrinsic,
            Provenance::Proven => cas_engine::Provenance::Proven,
            Provenance::Assumed => cas_engine::Provenance::Assumed,
            Provenance::Introduced => cas_engine::Provenance::Introduced,
        }
    }
}
