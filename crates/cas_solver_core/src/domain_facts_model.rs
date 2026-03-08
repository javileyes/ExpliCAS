use cas_ast::ExprId;

/// A domain predicate about an expression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Predicate {
    /// Expression != 0 (Definability class)
    NonZero(ExprId),
    /// Expression > 0 (Analytic class)
    Positive(ExprId),
    /// Expression >= 0 (Analytic class)
    NonNegative(ExprId),
    /// Expression is defined at this point (Definability class)
    Defined(ExprId),
}

impl Predicate {
    /// Get the condition class for this predicate.
    #[inline]
    pub fn condition_class(&self) -> crate::solve_safety_policy::ConditionClass {
        predicate_condition_class(self)
    }

    /// Get the expression this predicate is about.
    #[inline]
    pub fn expr(&self) -> ExprId {
        match self {
            Predicate::NonZero(e)
            | Predicate::Positive(e)
            | Predicate::NonNegative(e)
            | Predicate::Defined(e) => *e,
        }
    }

    /// Human-readable description of the predicate.
    pub fn describe(&self) -> &'static str {
        match self {
            Predicate::NonZero(_) => "≠ 0",
            Predicate::Positive(_) => "> 0",
            Predicate::NonNegative(_) => "≥ 0",
            Predicate::Defined(_) => "is defined",
        }
    }

    /// Short label for logging/debugging.
    pub fn label(&self) -> &'static str {
        match self {
            Predicate::NonZero(_) => "nonzero",
            Predicate::Positive(_) => "positive",
            Predicate::NonNegative(_) => "nonnegative",
            Predicate::Defined(_) => "defined",
        }
    }
}

/// Origin of a domain fact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Provenance {
    /// Inherited from input AST structure.
    Intrinsic,
    /// Structurally proven by the engine.
    Proven,
    /// Assumed by user or mode policy.
    Assumed,
    /// Introduced by a rule as a new side condition.
    Introduced,
}

/// Strength of evidence for a domain fact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FactStrength {
    /// Property is provably true.
    Proven,
    /// Property status is unknown (conservative).
    Unknown,
    /// Property is provably false.
    Disproven,
}

impl FactStrength {
    /// Returns true if the property is proven.
    #[inline]
    pub fn is_proven(self) -> bool {
        matches!(self, FactStrength::Proven)
    }

    /// Returns true if the property is unknown.
    #[inline]
    pub fn is_unknown(self) -> bool {
        matches!(self, FactStrength::Unknown)
    }

    /// Returns true if the property is disproven.
    #[inline]
    pub fn is_disproven(self) -> bool {
        matches!(self, FactStrength::Disproven)
    }
}

/// A complete domain assertion combining predicate, provenance, and strength.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DomainFact {
    /// What is being asserted.
    pub predicate: Predicate,
    /// Where the fact came from.
    pub provenance: Provenance,
    /// How strong the evidence is.
    pub strength: FactStrength,
}

/// Convert a `Proof` value to `FactStrength`.
#[inline]
pub fn proof_to_strength(proof: crate::domain_proof::Proof) -> FactStrength {
    use crate::domain_proof::Proof;
    match proof {
        Proof::Proven | Proof::ProvenImplicit => FactStrength::Proven,
        Proof::Unknown => FactStrength::Unknown,
        Proof::Disproven => FactStrength::Disproven,
    }
}

/// Convert `FactStrength` back to `Proof`.
#[inline]
pub fn strength_to_proof(strength: FactStrength) -> crate::domain_proof::Proof {
    use crate::domain_proof::Proof;
    match strength {
        FactStrength::Proven => Proof::Proven,
        FactStrength::Unknown => Proof::Unknown,
        FactStrength::Disproven => Proof::Disproven,
    }
}

/// Map a `Predicate` to its `ConditionClass`.
#[inline]
pub fn predicate_condition_class(pred: &Predicate) -> crate::solve_safety_policy::ConditionClass {
    use crate::solve_safety_policy::ConditionClass;
    match pred {
        Predicate::NonZero(_) | Predicate::Defined(_) => ConditionClass::Definability,
        Predicate::Positive(_) | Predicate::NonNegative(_) => ConditionClass::Analytic,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        predicate_condition_class, proof_to_strength, strength_to_proof, FactStrength, Predicate,
    };
    use crate::domain_proof::Proof;
    use crate::solve_safety_policy::ConditionClass;

    #[test]
    fn predicate_metadata_helpers_work() {
        let mut ctx = cas_ast::Context::default();
        let x = ctx.var("x");
        let p = Predicate::NonZero(x);
        assert_eq!(p.expr(), x);
        assert_eq!(p.describe(), "≠ 0");
        assert_eq!(p.label(), "nonzero");
        assert_eq!(p.condition_class(), ConditionClass::Definability);
        assert_eq!(predicate_condition_class(&p), ConditionClass::Definability);
    }

    #[test]
    fn fact_strength_helpers_work() {
        assert!(FactStrength::Proven.is_proven());
        assert!(FactStrength::Unknown.is_unknown());
        assert!(FactStrength::Disproven.is_disproven());
    }

    #[test]
    fn proof_strength_roundtrip_contract() {
        assert_eq!(proof_to_strength(Proof::Proven), FactStrength::Proven);
        assert_eq!(
            proof_to_strength(Proof::ProvenImplicit),
            FactStrength::Proven
        );
        assert_eq!(proof_to_strength(Proof::Unknown), FactStrength::Unknown);
        assert_eq!(proof_to_strength(Proof::Disproven), FactStrength::Disproven);

        assert_eq!(strength_to_proof(FactStrength::Proven), Proof::Proven);
        assert_eq!(strength_to_proof(FactStrength::Unknown), Proof::Unknown);
        assert_eq!(strength_to_proof(FactStrength::Disproven), Proof::Disproven);
    }
}
