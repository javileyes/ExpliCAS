use crate::cycle_detector::{expr_fingerprint, CycleDetector, FingerprintMemo};
use crate::phase::SimplifyPhase;
use cas_ast::Expr;

#[test]
fn test_cycle_detector_self_loop() {
    let mut detector = CycleDetector::new(SimplifyPhase::Core);

    assert!(detector.observe(100).is_none());

    // A -> A (self-loop)
    let info = detector.observe(100).expect("self loop should be detected");
    assert_eq!(info.period, 1);
    assert_eq!(info.at_step, 2);
}

#[test]
fn test_cycle_detector_period_2() {
    let mut detector = CycleDetector::new(SimplifyPhase::Transform);

    assert!(detector.observe(100).is_none()); // A
    assert!(detector.observe(200).is_none()); // B
    assert!(detector.observe(100).is_none()); // A (need more)

    // A -> B -> A -> B (2-cycle)
    let info = detector.observe(200).expect("2-cycle should be detected");
    assert_eq!(info.period, 2);
}

#[test]
fn test_cycle_detector_period_3() {
    let mut detector = CycleDetector::new(SimplifyPhase::Core);

    detector.observe(100); // A
    detector.observe(200); // B
    detector.observe(300); // C
    detector.observe(100); // A
    detector.observe(200); // B

    // A -> B -> C -> A -> B -> C (3-cycle)
    let info = detector.observe(300).expect("3-cycle should be detected");
    assert_eq!(info.period, 3);
}

#[test]
fn test_cycle_detector_no_false_positive() {
    let mut detector = CycleDetector::new(SimplifyPhase::Core);

    // Distinct sequence - no cycles
    for i in 0..20 {
        assert!(detector.observe(i * 1000 + 1).is_none());
    }
}

#[test]
fn test_fingerprint_different_expressions() {
    let mut ctx = cas_ast::Context::new();
    let mut memo = FingerprintMemo::new();

    let x = ctx.var("x");
    let y = ctx.var("y");
    let one = ctx.num(1);

    let x_plus_1 = ctx.add(Expr::Add(x, one));
    let y_plus_1 = ctx.add(Expr::Add(y, one));

    let h1 = expr_fingerprint(&ctx, x_plus_1, &mut memo);
    let h2 = expr_fingerprint(&ctx, y_plus_1, &mut memo);

    // Different expressions should have different fingerprints
    assert_ne!(h1, h2);
}

#[test]
fn test_fingerprint_same_expression() {
    let mut ctx = cas_ast::Context::new();
    let mut memo = FingerprintMemo::new();

    let x = ctx.var("x");
    let one = ctx.num(1);

    let expr1 = ctx.add(Expr::Add(x, one));
    let expr2 = ctx.add(Expr::Add(x, one)); // Same structure, different ExprId

    let h1 = expr_fingerprint(&ctx, expr1, &mut memo);
    memo.clear(); // Clear to force recompute
    let h2 = expr_fingerprint(&ctx, expr2, &mut memo);

    // Same structure should have same fingerprint
    assert_eq!(h1, h2);
}

#[test]
fn test_fingerprint_nested() {
    let mut ctx = cas_ast::Context::new();
    let mut memo = FingerprintMemo::new();

    let x = ctx.var("x");
    let one = ctx.num(1);
    let two = ctx.num(2);

    // (x + 1) * 2
    let x_plus_1 = ctx.add(Expr::Add(x, one));
    let expr = ctx.add(Expr::Mul(x_plus_1, two));

    let h = expr_fingerprint(&ctx, expr, &mut memo);

    // Should produce a valid hash
    assert_ne!(h, 0);
}

#[test]
fn test_fingerprint_functions() {
    let mut ctx = cas_ast::Context::new();
    let mut memo = FingerprintMemo::new();

    let x = ctx.var("x");
    let sin_x = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![x]);
    let cos_x = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![x]);

    let h1 = expr_fingerprint(&ctx, sin_x, &mut memo);
    let h2 = expr_fingerprint(&ctx, cos_x, &mut memo);

    // sin(x) != cos(x)
    assert_ne!(h1, h2);
}

#[test]
fn test_cycle_detector_reset() {
    let mut detector = CycleDetector::new(SimplifyPhase::Core);

    detector.observe(100);
    detector.observe(200);
    detector.observe(100);

    // Reset for new phase
    detector.reset(SimplifyPhase::Transform);

    // Should start fresh
    assert!(detector.observe(100).is_none());
    assert!(detector.observe(100).is_some()); // Now detects self-loop
}
