mod node;
mod walk;

use std::collections::HashMap;

use cas_ast::{Context, ExprId};

use crate::EvalConfig;

pub(super) struct IterativeFolder<'a> {
    ctx: &'a mut Context,
    cfg: &'a EvalConfig,
    memo: HashMap<ExprId, ExprId>,
    nodes_created: u64,
    folds_performed: u64,
}

impl<'a> IterativeFolder<'a> {
    pub(super) fn new(ctx: &'a mut Context, cfg: &'a EvalConfig) -> Self {
        Self {
            ctx,
            cfg,
            memo: HashMap::new(),
            nodes_created: 0,
            folds_performed: 0,
        }
    }

    pub(super) fn nodes_created(&self) -> u64 {
        self.nodes_created
    }

    pub(super) fn folds_performed(&self) -> u64 {
        self.folds_performed
    }
}
