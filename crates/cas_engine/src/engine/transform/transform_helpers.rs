//! Expression transform helpers: binary, pow, div, and function simplification.
//!
//! These methods are extracted with `#[inline(never)]` to reduce the stack frame
//! size of `transform_expr_recursive`.

use super::*;

impl<'a> LocalSimplificationTransformer<'a> {
    /// Transform binary expression (Add/Sub/Mul) by simplifying children.
    /// Extracted to reduce stack frame size in transform_expr_recursive.
    #[inline(never)]
    pub(super) fn transform_binary(
        &mut self,
        id: ExprId,
        l: ExprId,
        r: ExprId,
        op: BinaryOp,
    ) -> ExprId {
        if self.steps_mode != StepsMode::Off {
            self.current_path.push(crate::step::PathStep::Left);
        }
        self.ancestor_stack.push(id);
        let new_l = self.transform_expr_recursive(l);
        self.ancestor_stack.pop();
        if self.steps_mode != StepsMode::Off {
            self.current_path.pop();
        }

        if self.steps_mode != StepsMode::Off {
            self.current_path.push(crate::step::PathStep::Right);
        }
        self.ancestor_stack.push(id);
        let new_r = self.transform_expr_recursive(r);
        self.ancestor_stack.pop();
        if self.steps_mode != StepsMode::Off {
            self.current_path.pop();
        }

        if new_l != l || new_r != r {
            let expr = match op {
                BinaryOp::Add => Expr::Add(new_l, new_r),
                BinaryOp::Sub => Expr::Sub(new_l, new_r),
                BinaryOp::Mul => Expr::Mul(new_l, new_r),
                BinaryOp::Div => Expr::Div(new_l, new_r),
            };
            self.context.add(expr)
        } else {
            id
        }
    }

    /// Transform Pow expression with early detection for sqrt-of-square patterns.
    /// Extracted with #[inline(never)] to reduce stack frame size.
    #[inline(never)]
    pub(super) fn transform_pow(&mut self, id: ExprId, base: ExprId, exp: ExprId) -> ExprId {
        // EARLY DETECTION: sqrt-of-square pattern (u^2)^(1/2) -> |u|
        // Must check BEFORE recursing into children to prevent binomial expansion
        if crate::helpers::is_half(self.context, exp) {
            // Try (something^2)^(1/2) -> |something|
            if let Some(result) = self.try_sqrt_of_square(id, base) {
                return result;
            }
            // Try (u * u)^(1/2) -> |u|
            if let Some(result) = self.try_sqrt_of_product(id, base) {
                return result;
            }
        }

        // Check if this Pow is canonical before recursing into children
        if crate::canonical_forms::is_canonical_form(self.context, id) {
            debug!(
                "Skipping simplification of canonical Pow: {:?}",
                self.context.get(id)
            );
            return id;
        }

        // Simplify children
        if self.steps_mode != StepsMode::Off {
            self.current_path.push(crate::step::PathStep::Base);
        }
        self.ancestor_stack.push(id);
        let new_b = self.transform_expr_recursive(base);
        self.ancestor_stack.pop();
        if self.steps_mode != StepsMode::Off {
            self.current_path.pop();
        }

        if self.steps_mode != StepsMode::Off {
            self.current_path.push(crate::step::PathStep::Exponent);
        }
        self.ancestor_stack.push(id);
        let new_e = self.transform_expr_recursive(exp);
        self.ancestor_stack.pop();
        if self.steps_mode != StepsMode::Off {
            self.current_path.pop();
        }

        if new_b != base || new_e != exp {
            self.context.add(Expr::Pow(new_b, new_e))
        } else {
            id
        }
    }

    /// Try to simplify (u^2)^(1/2) -> |u|
    #[inline(never)]
    fn try_sqrt_of_square(&mut self, id: ExprId, base: ExprId) -> Option<ExprId> {
        if let Expr::Pow(inner_base, inner_exp) = self.context.get(base) {
            if let Expr::Number(n) = self.context.get(*inner_exp) {
                if n.is_integer() && *n == num_rational::BigRational::from_integer(2.into()) {
                    let abs_expr = self.context.call("abs", vec![*inner_base]);
                    self.record_step(
                        "sqrt(u^2) = |u|",
                        "Simplify Square Root of Square",
                        id,
                        abs_expr,
                    );
                    return Some(self.transform_expr_recursive(abs_expr));
                }
            }
        }
        None
    }

    /// Try to simplify (u * u)^(1/2) -> |u|
    #[inline(never)]
    fn try_sqrt_of_product(&mut self, id: ExprId, base: ExprId) -> Option<ExprId> {
        if let Expr::Mul(left, right) = self.context.get(base) {
            if crate::ordering::compare_expr(self.context, *left, *right)
                == std::cmp::Ordering::Equal
            {
                let abs_expr = self.context.call("abs", vec![*left]);
                self.record_step(
                    "sqrt(u * u) = |u|",
                    "Simplify Square Root of Product",
                    id,
                    abs_expr,
                );
                return Some(self.transform_expr_recursive(abs_expr));
            }
        }
        None
    }

    /// Transform Function expression by simplifying children.
    /// Extracted with #[inline(never)] to reduce stack frame size.
    #[inline(never)]
    pub(super) fn transform_function(
        &mut self,
        id: ExprId,
        fn_id: SymbolId,
        args: Vec<ExprId>,
    ) -> ExprId {
        let name = self.context.sym_name(fn_id);
        // Check if this function is canonical before recursing into children
        if (name == "sqrt" || name == "abs")
            && crate::canonical_forms::is_canonical_form(self.context, id)
        {
            debug!(
                "Skipping simplification of canonical Function: {:?}",
                self.context.get(id)
            );
            return id;
        }

        // HoldAll semantics: do NOT simplify arguments for these functions
        if is_hold_all_function(name) {
            debug!(
                "HoldAll function, skipping child simplification: {:?}",
                self.context.get(id)
            );
            return id;
        }

        // Simplify children
        let mut new_args = Vec::with_capacity(args.len());
        let mut changed = false;
        for (i, arg) in args.iter().enumerate() {
            if self.steps_mode != StepsMode::Off {
                self.current_path.push(crate::step::PathStep::Arg(i));
            }
            self.ancestor_stack.push(id);
            let new_arg = self.transform_expr_recursive(*arg);
            self.ancestor_stack.pop();
            if self.steps_mode != StepsMode::Off {
                self.current_path.pop();
            }

            if new_arg != *arg {
                changed = true;
            }
            new_args.push(new_arg);
        }

        if changed {
            self.context.add(Expr::Function(fn_id, new_args))
        } else {
            id
        }
    }

    /// Transform Div expression with early detection for difference-of-squares pattern.
    /// Extracted with #[inline(never)] to reduce stack frame size.
    #[inline(never)]
    pub(super) fn transform_div(&mut self, id: ExprId, l: ExprId, r: ExprId) -> ExprId {
        // EARLY DETECTION: (A² - B²) / (A ± B) pattern
        if let Some(early_result) = crate::rules::algebra::try_difference_of_squares_preorder(
            self.context,
            id,
            l,
            r,
            self.steps_mode != StepsMode::Off,
            &mut self.steps,
            &self.current_path,
        ) {
            // Note: don't decrement depth here - transform_expr_recursive manages it
            return self.transform_expr_recursive(early_result);
        }

        // Simplify children
        if self.steps_mode != StepsMode::Off {
            self.current_path.push(crate::step::PathStep::Left);
        }
        self.ancestor_stack.push(id);
        let new_l = self.transform_expr_recursive(l);
        self.ancestor_stack.pop();
        if self.steps_mode != StepsMode::Off {
            self.current_path.pop();
        }

        if self.steps_mode != StepsMode::Off {
            self.current_path.push(crate::step::PathStep::Right);
        }
        self.ancestor_stack.push(id);
        let new_r = self.transform_expr_recursive(r);
        self.ancestor_stack.pop();
        if self.steps_mode != StepsMode::Off {
            self.current_path.pop();
        }

        if new_l != l || new_r != r {
            self.context.add(Expr::Div(new_l, new_r))
        } else {
            id
        }
    }
}
