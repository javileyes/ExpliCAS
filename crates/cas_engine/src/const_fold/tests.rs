use super::*;
use cas_ast::Context;

#[test]
fn test_off_mode_is_noop() {
    let mut ctx = Context::new();
    let expr = ctx.num(42);
    let cfg = EvalConfig::default();
    let mut budget = Budget::preset_unlimited();

    let result = fold_constants(&mut ctx, expr, &cfg, ConstFoldMode::Off, &mut budget).unwrap();
    assert_eq!(result.expr, expr);
    assert_eq!(result.folds_performed, 0);
}

#[test]
fn test_literal_unchanged() {
    let mut ctx = Context::new();
    let expr = ctx.num(42);
    let cfg = EvalConfig::default();
    let mut budget = Budget::preset_unlimited();

    let result = fold_constants(&mut ctx, expr, &cfg, ConstFoldMode::Safe, &mut budget).unwrap();
    assert_eq!(result.expr, expr);
}
