//! Serializable `cas_ast::Context` snapshots.
//!
//! Kept in `cas_session_core` so stateful runtimes can persist/restore
//! expression arenas without duplicating AST serialization glue.

use serde::{Deserialize, Serialize};

/// Serializable Context representation.
/// Since `cas_ast::Context` doesn't have serde, we serialize the nodes array manually.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextSnapshot {
    /// The expression nodes in arena order (`ExprId.index() = position`).
    pub nodes: Vec<ExprNodeSnapshot>,
}

/// Serializable `Expr` variant - mirrors `cas_ast::Expr`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExprNodeSnapshot {
    Number {
        num: String,
        den: String,
    }, // BigRational as strings for portability
    Constant(ConstantSnapshot),
    Variable(String),
    Add(u32, u32), // ExprId indices
    Sub(u32, u32),
    Mul(u32, u32),
    Div(u32, u32),
    Pow(u32, u32),
    Neg(u32),
    Function(String, Vec<u32>),
    Matrix {
        rows: usize,
        cols: usize,
        data: Vec<u32>,
    },
    SessionRef(u64),
    Hold(u32), // ExprId index for inner expression
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstantSnapshot {
    Pi,
    E,
    Infinity,
    Undefined,
    I,
    Phi,
}

impl ContextSnapshot {
    pub fn from_context(ctx: &cas_ast::Context) -> Self {
        use cas_ast::Expr;

        let nodes = ctx
            .nodes
            .iter()
            .map(|expr| match expr {
                Expr::Number(r) => ExprNodeSnapshot::Number {
                    num: r.numer().to_string(),
                    den: r.denom().to_string(),
                },
                Expr::Constant(c) => ExprNodeSnapshot::Constant(ConstantSnapshot::from(c)),
                Expr::Variable(sym_id) => {
                    ExprNodeSnapshot::Variable(ctx.sym_name(*sym_id).to_string())
                }
                Expr::Add(l, r) => ExprNodeSnapshot::Add(l.index() as u32, r.index() as u32),
                Expr::Sub(l, r) => ExprNodeSnapshot::Sub(l.index() as u32, r.index() as u32),
                Expr::Mul(l, r) => ExprNodeSnapshot::Mul(l.index() as u32, r.index() as u32),
                Expr::Div(l, r) => ExprNodeSnapshot::Div(l.index() as u32, r.index() as u32),
                Expr::Pow(l, r) => ExprNodeSnapshot::Pow(l.index() as u32, r.index() as u32),
                Expr::Neg(inner) => ExprNodeSnapshot::Neg(inner.index() as u32),
                Expr::Function(fn_id, args) => ExprNodeSnapshot::Function(
                    ctx.sym_name(*fn_id).to_string(),
                    args.iter().map(|a| a.index() as u32).collect(),
                ),
                Expr::Matrix { rows, cols, data } => ExprNodeSnapshot::Matrix {
                    rows: *rows,
                    cols: *cols,
                    data: data.iter().map(|d| d.index() as u32).collect(),
                },
                Expr::SessionRef(id) => ExprNodeSnapshot::SessionRef(*id),
                Expr::Hold(inner) => ExprNodeSnapshot::Hold(inner.index() as u32),
            })
            .collect();

        Self { nodes }
    }

    pub fn into_context(self) -> cas_ast::Context {
        use cas_ast::{Context, Expr, ExprId};

        let node_capacity = self.nodes.len();
        let mut ctx = Context::with_restore_capacity(node_capacity);

        // Reconstruct nodes in the same order to preserve ExprId stability.
        for node in self.nodes {
            let expr = match node {
                ExprNodeSnapshot::Number { num, den } => {
                    Expr::Number(parse_snapshot_rational(&num, &den))
                }
                ExprNodeSnapshot::Constant(c) => Expr::Constant(c.into()),
                ExprNodeSnapshot::Variable(s) => Expr::Variable(ctx.intern_symbol(&s)),
                ExprNodeSnapshot::Add(l, r) => Expr::Add(ExprId::from_raw(l), ExprId::from_raw(r)),
                ExprNodeSnapshot::Sub(l, r) => Expr::Sub(ExprId::from_raw(l), ExprId::from_raw(r)),
                ExprNodeSnapshot::Mul(l, r) => Expr::Mul(ExprId::from_raw(l), ExprId::from_raw(r)),
                ExprNodeSnapshot::Div(l, r) => Expr::Div(ExprId::from_raw(l), ExprId::from_raw(r)),
                ExprNodeSnapshot::Pow(l, r) => Expr::Pow(ExprId::from_raw(l), ExprId::from_raw(r)),
                ExprNodeSnapshot::Neg(inner) => Expr::Neg(ExprId::from_raw(inner)),
                ExprNodeSnapshot::Function(name, args) => {
                    let fn_id = ctx.intern_symbol(&name);
                    Expr::Function(fn_id, args.into_iter().map(ExprId::from_raw).collect())
                }
                ExprNodeSnapshot::Matrix { rows, cols, data } => Expr::Matrix {
                    rows,
                    cols,
                    data: data.into_iter().map(ExprId::from_raw).collect(),
                },
                ExprNodeSnapshot::SessionRef(id) => Expr::SessionRef(id),
                ExprNodeSnapshot::Hold(inner) => Expr::Hold(ExprId::from_raw(inner)),
            };
            // Use raw push to preserve exact structure without re-canonicalization.
            ctx.nodes.push(expr);
        }

        ctx
    }
}

fn parse_snapshot_rational(num: &str, den: &str) -> num_rational::BigRational {
    use num_bigint::BigInt;
    use num_rational::BigRational;

    if let (Ok(n), Ok(d)) = (num.parse::<i64>(), den.parse::<i64>()) {
        return BigRational::new(BigInt::from(n), BigInt::from(d));
    }

    let n: BigInt = num.parse().unwrap_or_default();
    let d: BigInt = den.parse().unwrap_or_else(|_| BigInt::from(1));
    BigRational::new(n, d)
}

impl From<&cas_ast::Constant> for ConstantSnapshot {
    fn from(c: &cas_ast::Constant) -> Self {
        use cas_ast::Constant;
        match c {
            Constant::Pi => ConstantSnapshot::Pi,
            Constant::E => ConstantSnapshot::E,
            Constant::Infinity => ConstantSnapshot::Infinity,
            Constant::Undefined => ConstantSnapshot::Undefined,
            Constant::I => ConstantSnapshot::I,
            Constant::Phi => ConstantSnapshot::Phi,
        }
    }
}

impl From<ConstantSnapshot> for cas_ast::Constant {
    fn from(c: ConstantSnapshot) -> Self {
        use cas_ast::Constant;
        match c {
            ConstantSnapshot::Pi => Constant::Pi,
            ConstantSnapshot::E => Constant::E,
            ConstantSnapshot::Infinity => Constant::Infinity,
            ConstantSnapshot::Undefined => Constant::Undefined,
            ConstantSnapshot::I => Constant::I,
            ConstantSnapshot::Phi => Constant::Phi,
        }
    }
}

#[cfg(test)]
mod tests;
