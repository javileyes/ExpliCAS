use proptest::prelude::*;
use cas_ast::{Expr, Context, ExprId, Constant};

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

pub fn arb_recursive_expr() -> impl Strategy<Value = RecursiveExpr> {
    let leaf = prop_oneof![
        (-10i64..10).prop_map(RecursiveExpr::Number),
        "[a-z]".prop_map(RecursiveExpr::Variable),
        Just(RecursiveExpr::Constant(Constant::Pi)),
        Just(RecursiveExpr::Constant(Constant::E)),
    ];

    leaf.prop_recursive(
        4, // levels deep
        20, // max size
        10, // items per collection
        |inner| prop_oneof![
            (inner.clone(), inner.clone()).prop_map(|(l, r)| RecursiveExpr::Add(Box::new(l), Box::new(r))),
            (inner.clone(), inner.clone()).prop_map(|(l, r)| RecursiveExpr::Sub(Box::new(l), Box::new(r))),
            (inner.clone(), inner.clone()).prop_map(|(l, r)| RecursiveExpr::Mul(Box::new(l), Box::new(r))),
            (inner.clone(), inner.clone()).prop_map(|(l, r)| RecursiveExpr::Div(Box::new(l), Box::new(r))),
            (inner.clone(), inner.clone()).prop_map(|(l, r)| RecursiveExpr::Pow(Box::new(l), Box::new(r))),
            inner.clone().prop_map(|e| RecursiveExpr::Neg(Box::new(e))),
            inner.clone().prop_map(|e| RecursiveExpr::Function("sin".to_string(), vec![e])),
            inner.clone().prop_map(|e| RecursiveExpr::Function("cos".to_string(), vec![e])),
            inner.clone().prop_map(|e| RecursiveExpr::Function("ln".to_string(), vec![e])),
        ]
    )
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
        },
        RecursiveExpr::Sub(l, r) => {
            let lid = add_recursive(ctx, *l);
            let rid = add_recursive(ctx, *r);
            ctx.add(Expr::Sub(lid, rid))
        },
        RecursiveExpr::Mul(l, r) => {
            let lid = add_recursive(ctx, *l);
            let rid = add_recursive(ctx, *r);
            ctx.add(Expr::Mul(lid, rid))
        },
        RecursiveExpr::Div(l, r) => {
            let lid = add_recursive(ctx, *l);
            let rid = add_recursive(ctx, *r);
            ctx.add(Expr::Div(lid, rid))
        },
        RecursiveExpr::Pow(l, r) => {
            let lid = add_recursive(ctx, *l);
            let rid = add_recursive(ctx, *r);
            ctx.add(Expr::Pow(lid, rid))
        },
        RecursiveExpr::Neg(e) => {
            let eid = add_recursive(ctx, *e);
            ctx.add(Expr::Neg(eid))
        },
        RecursiveExpr::Function(name, args) => {
            let arg_ids: Vec<ExprId> = args.into_iter().map(|a| add_recursive(ctx, a)).collect();
            ctx.add(Expr::Function(name, arg_ids))
        },
    }
}
