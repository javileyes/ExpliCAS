use crate::phase::SimplifyPhase;

#[test]
fn test_phase_order() {
    assert_eq!(SimplifyPhase::Core.next(), Some(SimplifyPhase::Transform));
    assert_eq!(
        SimplifyPhase::Transform.next(),
        Some(SimplifyPhase::Rationalize)
    );
    assert_eq!(
        SimplifyPhase::Rationalize.next(),
        Some(SimplifyPhase::PostCleanup)
    );
    assert_eq!(SimplifyPhase::PostCleanup.next(), None);
}

#[test]
fn test_phase_properties() {
    assert!(SimplifyPhase::Core.is_core_phase());
    assert!(!SimplifyPhase::Core.allows_distribution());
    assert!(!SimplifyPhase::Core.allows_rationalization());

    assert!(!SimplifyPhase::Transform.is_core_phase());
    assert!(SimplifyPhase::Transform.allows_distribution());
    assert!(!SimplifyPhase::Transform.allows_rationalization());

    assert!(!SimplifyPhase::Rationalize.is_core_phase());
    assert!(!SimplifyPhase::Rationalize.allows_distribution());
    assert!(SimplifyPhase::Rationalize.allows_rationalization());

    assert!(SimplifyPhase::PostCleanup.is_core_phase());
    assert!(!SimplifyPhase::PostCleanup.allows_distribution());
}

#[test]
fn test_all_phases() {
    let phases = SimplifyPhase::all();
    assert_eq!(phases.len(), 4);
    assert_eq!(phases[0], SimplifyPhase::Core);
    assert_eq!(phases[3], SimplifyPhase::PostCleanup);
}
