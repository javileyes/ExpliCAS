//! Helpers to map external proof enums into solver-core proof statuses.

use crate::linear_solution::NonZeroStatus;
use crate::log_domain::ProofStatus;
use crate::proof_status::proof_status_to_nonzero_status;

/// Map an external proof value to [`ProofStatus`].
pub fn map_external_proof_status_with<P, FIsProven, FIsDisproven>(
    proof: P,
    is_proven: FIsProven,
    is_disproven: FIsDisproven,
) -> ProofStatus
where
    FIsProven: Fn(&P) -> bool,
    FIsDisproven: Fn(&P) -> bool,
{
    if is_proven(&proof) {
        ProofStatus::Proven
    } else if is_disproven(&proof) {
        ProofStatus::Disproven
    } else {
        ProofStatus::Unknown
    }
}

/// Map an external proof value to [`NonZeroStatus`].
pub fn map_external_nonzero_status_with<P, FIsProven, FIsDisproven>(
    proof: P,
    is_proven: FIsProven,
    is_disproven: FIsDisproven,
) -> NonZeroStatus
where
    FIsProven: Fn(&P) -> bool,
    FIsDisproven: Fn(&P) -> bool,
{
    let status = map_external_proof_status_with(proof, is_proven, is_disproven);
    proof_status_to_nonzero_status(status)
}

#[cfg(test)]
mod tests {
    use super::{map_external_nonzero_status_with, map_external_proof_status_with};
    use crate::linear_solution::NonZeroStatus;
    use crate::log_domain::ProofStatus;

    #[derive(Debug, Clone, Copy)]
    enum ExternalProof {
        Yes,
        No,
        Maybe,
    }

    #[test]
    fn map_external_proof_status_maps_three_valued_domain() {
        let proven = map_external_proof_status_with(
            ExternalProof::Yes,
            |p| matches!(p, ExternalProof::Yes),
            |p| matches!(p, ExternalProof::No),
        );
        let disproven = map_external_proof_status_with(
            ExternalProof::No,
            |p| matches!(p, ExternalProof::Yes),
            |p| matches!(p, ExternalProof::No),
        );
        let unknown = map_external_proof_status_with(
            ExternalProof::Maybe,
            |p| matches!(p, ExternalProof::Yes),
            |p| matches!(p, ExternalProof::No),
        );

        assert_eq!(proven, ProofStatus::Proven);
        assert_eq!(disproven, ProofStatus::Disproven);
        assert_eq!(unknown, ProofStatus::Unknown);
    }

    #[test]
    fn map_external_nonzero_status_composes_with_proof_mapping() {
        let out = map_external_nonzero_status_with(
            ExternalProof::Yes,
            |p| matches!(p, ExternalProof::Yes),
            |p| matches!(p, ExternalProof::No),
        );
        assert_eq!(out, NonZeroStatus::NonZero);
    }
}
