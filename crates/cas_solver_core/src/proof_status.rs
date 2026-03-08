use crate::linear_solution::NonZeroStatus;
use crate::log_domain::ProofStatus;

/// Convert log-domain proof status into NonZeroStatus used by solution builders.
pub fn proof_status_to_nonzero_status(status: ProofStatus) -> NonZeroStatus {
    match status {
        ProofStatus::Proven => NonZeroStatus::NonZero,
        ProofStatus::Disproven => NonZeroStatus::Zero,
        ProofStatus::Unknown => NonZeroStatus::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::proof_status_to_nonzero_status;
    use crate::linear_solution::NonZeroStatus;
    use crate::log_domain::ProofStatus;

    #[test]
    fn proven_maps_to_nonzero() {
        assert_eq!(
            proof_status_to_nonzero_status(ProofStatus::Proven),
            NonZeroStatus::NonZero
        );
    }

    #[test]
    fn disproven_maps_to_zero() {
        assert_eq!(
            proof_status_to_nonzero_status(ProofStatus::Disproven),
            NonZeroStatus::Zero
        );
    }

    #[test]
    fn unknown_maps_to_unknown() {
        assert_eq!(
            proof_status_to_nonzero_status(ProofStatus::Unknown),
            NonZeroStatus::Unknown
        );
    }
}
