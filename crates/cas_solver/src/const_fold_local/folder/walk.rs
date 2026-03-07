use cas_ast::ExprId;

use crate::{Budget, CasError, Metric, Operation};

use super::IterativeFolder;
use crate::const_fold_local::tree::get_children;

#[derive(Clone, Copy)]
struct Frame {
    id: ExprId,
    // 0 = push children, 1 = fold node
    state: u8,
}

impl<'a> IterativeFolder<'a> {
    pub(in super::super) fn fold(
        &mut self,
        root: ExprId,
        budget: &mut Budget,
    ) -> Result<ExprId, CasError> {
        let mut stack = vec![Frame { id: root, state: 0 }];

        while let Some(frame) = stack.pop() {
            if frame.state == 0 {
                if self.memo.contains_key(&frame.id) {
                    continue;
                }

                stack.push(Frame {
                    id: frame.id,
                    state: 1,
                });

                let children = get_children(self.ctx, frame.id);
                for child in children.into_iter().rev() {
                    stack.push(Frame {
                        id: child,
                        state: 0,
                    });
                }
            } else {
                let folded = self.try_fold_node(frame.id);
                self.memo.insert(frame.id, folded);
                if folded != frame.id {
                    self.folds_performed += 1;
                }
                budget.charge(Operation::ConstFold, Metric::Iterations, 1)?;
            }
        }

        Ok(self.memo.get(&root).copied().unwrap_or(root))
    }
}
