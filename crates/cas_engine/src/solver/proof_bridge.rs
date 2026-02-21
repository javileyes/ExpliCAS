//! Shared conversions from engine domain proofs to solver-core proof/status types.

use cas_solver_core::linear_solution::NonZeroStatus;
use cas_solver_core::log_domain::ProofStatus;
use cas_solver_core::proof_status::proof_status_to_nonzero_status;

/// Convert engine proof outcome to solver-core proof status.
pub(crate) fn proof_to_status(proof: crate::domain::Proof) -> ProofStatus {
    match proof {
        crate::domain::Proof::Proven | crate::domain::Proof::ProvenImplicit => ProofStatus::Proven,
        crate::domain::Proof::Unknown => ProofStatus::Unknown,
        crate::domain::Proof::Disproven => ProofStatus::Disproven,
    }
}

/// Convert engine proof outcome to NonZeroStatus used by solver result builders.
pub(crate) fn proof_to_nonzero_status(proof: crate::domain::Proof) -> NonZeroStatus {
    proof_status_to_nonzero_status(proof_to_status(proof))
}
