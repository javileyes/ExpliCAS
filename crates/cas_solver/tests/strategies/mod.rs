use cas_ast::{Constant, Context, Expr, ExprId};
use proptest::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════════
// TEST COMPLEXITY CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════
//
// These parameters control the complexity of generated expressions for property tests.
// Use different profiles to stress-test the simplification engine and identify
// optimization opportunities in rule orchestration.
//
// PARAMETERS:
// - DEPTH: Maximum recursion depth (2^DEPTH possible nesting levels)
// - SIZE:  Maximum number of nodes in the generated expression tree
// - ITEMS: Maximum items per collection (affects function arguments)
//
// PROFILES:
// - SAFE:    (2, 8, 4)   - Fast, never stack overflows
// - NORMAL:  (3, 15, 6)  - Balanced, may overflow on small stacks
// - STRESS:  (4, 20, 10) - Original values, will stress the engine
// - EXTREME: (5, 30, 15) - For debugging specific bottlenecks
//
// To run stress tests with adequate stack:
//   RUST_MIN_STACK=16777216 cargo test --package cas_engine --test property_tests
//
// To debug a specific failing case, add PROPTEST_MAX_SHRINK_ITERS=0 to see the
// original expression before shrinking.
// ═══════════════════════════════════════════════════════════════════════════════

/// Test profile for property-based testing
#[derive(Clone, Copy, Debug)]
pub struct TestProfile {
    pub depth: u32,
    pub size: u32,
    pub items: u32,
    pub name: &'static str,
}

impl TestProfile {
    /// Safe profile - guaranteed no stack overflow
    pub const SAFE: TestProfile = TestProfile {
        depth: 2,
        size: 8,
        items: 4,
        name: "SAFE",
    };

    /// Normal profile - balanced complexity
    pub const NORMAL: TestProfile = TestProfile {
        depth: 3,
        size: 15,
        items: 6,
        name: "NORMAL",
    };

    /// Stress profile - original values, will stress the engine
    pub const STRESS: TestProfile = TestProfile {
        depth: 4,
        size: 20,
        items: 10,
        name: "STRESS",
    };

    /// Extreme profile - for debugging specific bottlenecks
    pub const EXTREME: TestProfile = TestProfile {
        depth: 5,
        size: 30,
        items: 15,
        name: "EXTREME",
    };
}

// ═══════════════════════════════════════════════════════════════════════════════
// ACTIVE PROFILE SELECTION
// Set via environment variable: STRESS_PROFILE=SAFE|NORMAL|STRESS|EXTREME
// Example: STRESS_PROFILE=STRESS cargo test --package cas_engine --test stress_test
// ═══════════════════════════════════════════════════════════════════════════════

/// Default profile when STRESS_PROFILE env var is not set
pub const DEFAULT_PROFILE: TestProfile = TestProfile::SAFE;

/// Get the active test profile from environment variable STRESS_PROFILE
/// Falls back to DEFAULT_PROFILE if not set or invalid
pub fn get_active_profile() -> TestProfile {
    match std::env::var("STRESS_PROFILE").as_deref() {
        Ok("SAFE") => TestProfile::SAFE,
        Ok("NORMAL") => TestProfile::NORMAL,
        Ok("STRESS") => TestProfile::STRESS,
        Ok("EXTREME") => TestProfile::EXTREME,
        Ok(other) => {
            eprintln!(
                "[WARN] Unknown STRESS_PROFILE='{}', using {:?}. Valid: SAFE|NORMAL|STRESS|EXTREME",
                other, DEFAULT_PROFILE.name
            );
            DEFAULT_PROFILE
        }
        Err(_) => DEFAULT_PROFILE,
    }
}

#[derive(Clone, Debug)]
pub enum RecursiveExpr {
    Number(i64),
    Variable(String),
    Constant(Constant),
    Add(Box<RecursiveExpr>, Box<RecursiveExpr>),
    Sub(Box<RecursiveExpr>, Box<RecursiveExpr>),
    Mul(Box<RecursiveExpr>, Box<RecursiveExpr>),
    Div(Box<RecursiveExpr>, Box<RecursiveExpr>),
    Pow(Box<RecursiveExpr>, Box<RecursiveExpr>),
    Neg(Box<RecursiveExpr>),
    Function(String, Vec<RecursiveExpr>),
}

/// Create a recursive expression generator with the active profile (from env var)
#[allow(dead_code)]
pub fn arb_recursive_expr() -> impl Strategy<Value = RecursiveExpr> {
    arb_recursive_expr_with_profile(get_active_profile())
}

/// Create a recursive expression generator with a specific profile
pub fn arb_recursive_expr_with_profile(
    profile: TestProfile,
) -> impl Strategy<Value = RecursiveExpr> {
    let leaf = prop_oneof![
        (-10i64..10).prop_map(RecursiveExpr::Number),
        "[a-z]".prop_map(RecursiveExpr::Variable),
        Just(RecursiveExpr::Constant(Constant::Pi)),
        Just(RecursiveExpr::Constant(Constant::E)),
    ];

    leaf.prop_recursive(profile.depth, profile.size, profile.items, |inner| {
        prop_oneof![
            (inner.clone(), inner.clone())
                .prop_map(|(l, r)| RecursiveExpr::Add(Box::new(l), Box::new(r))),
            (inner.clone(), inner.clone())
                .prop_map(|(l, r)| RecursiveExpr::Sub(Box::new(l), Box::new(r))),
            (inner.clone(), inner.clone())
                .prop_map(|(l, r)| RecursiveExpr::Mul(Box::new(l), Box::new(r))),
            (inner.clone(), inner.clone())
                .prop_map(|(l, r)| RecursiveExpr::Div(Box::new(l), Box::new(r))),
            (inner.clone(), inner.clone())
                .prop_map(|(l, r)| RecursiveExpr::Pow(Box::new(l), Box::new(r))),
            inner.clone().prop_map(|e| RecursiveExpr::Neg(Box::new(e))),
            inner
                .clone()
                .prop_map(|e| RecursiveExpr::Function("sin".to_string(), vec![e])),
            inner
                .clone()
                .prop_map(|e| RecursiveExpr::Function("cos".to_string(), vec![e])),
            inner
                .clone()
                .prop_map(|e| RecursiveExpr::Function("ln".to_string(), vec![e])),
        ]
    })
}

pub fn to_context(re: RecursiveExpr) -> (Context, ExprId) {
    let mut ctx = Context::new();
    let id = add_recursive(&mut ctx, re);
    (ctx, id)
}

fn add_recursive(ctx: &mut Context, re: RecursiveExpr) -> ExprId {
    match re {
        RecursiveExpr::Number(n) => ctx.num(n),
        RecursiveExpr::Variable(s) => ctx.var(&s),
        RecursiveExpr::Constant(c) => ctx.add(Expr::Constant(c)),
        RecursiveExpr::Add(l, r) => {
            let lid = add_recursive(ctx, *l);
            let rid = add_recursive(ctx, *r);
            ctx.add(Expr::Add(lid, rid))
        }
        RecursiveExpr::Sub(l, r) => {
            let lid = add_recursive(ctx, *l);
            let rid = add_recursive(ctx, *r);
            ctx.add(Expr::Sub(lid, rid))
        }
        RecursiveExpr::Mul(l, r) => {
            let lid = add_recursive(ctx, *l);
            let rid = add_recursive(ctx, *r);
            ctx.add(Expr::Mul(lid, rid))
        }
        RecursiveExpr::Div(l, r) => {
            let lid = add_recursive(ctx, *l);
            let rid = add_recursive(ctx, *r);
            ctx.add(Expr::Div(lid, rid))
        }
        RecursiveExpr::Pow(l, r) => {
            let lid = add_recursive(ctx, *l);
            let rid = add_recursive(ctx, *r);
            ctx.add(Expr::Pow(lid, rid))
        }
        RecursiveExpr::Neg(e) => {
            let eid = add_recursive(ctx, *e);
            ctx.add(Expr::Neg(eid))
        }
        RecursiveExpr::Function(name, args) => {
            let arg_ids: Vec<ExprId> = args.into_iter().map(|a| add_recursive(ctx, a)).collect();
            ctx.call(&name, arg_ids)
        }
    }
}
