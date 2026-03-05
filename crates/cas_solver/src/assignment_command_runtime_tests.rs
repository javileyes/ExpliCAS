#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use cas_ast::ExprId;

    use crate::{
        evaluate_assignment_command_message_with_context, evaluate_assignment_command_with_context,
        evaluate_let_assignment_command_message_with_context, AssignmentApplyContext, Simplifier,
    };

    #[derive(Default)]
    struct MockAssignmentContext {
        bindings: HashMap<String, ExprId>,
    }

    impl AssignmentApplyContext for MockAssignmentContext {
        fn assignment_unset_binding(&mut self, name: &str) -> bool {
            self.bindings.remove(name).is_some()
        }

        fn assignment_set_binding(&mut self, name: String, expr: ExprId) {
            self.bindings.insert(name, expr);
        }

        fn assignment_resolve_state_refs(
            &self,
            _ctx: &mut cas_ast::Context,
            expr: ExprId,
        ) -> Result<ExprId, String> {
            Ok(expr)
        }

        fn assignment_is_reserved_name(&self, _name: &str) -> bool {
            false
        }
    }

    #[test]
    fn evaluate_assignment_command_with_context_returns_output() {
        let mut context = MockAssignmentContext::default();
        let mut simplifier = Simplifier::with_default_rules();
        let output = evaluate_assignment_command_with_context(
            &mut context,
            &mut simplifier,
            "a",
            "x + x",
            false,
        )
        .expect("output");
        assert_eq!(output.name, "a");
    }

    #[test]
    fn evaluate_assignment_command_message_with_context_formats_eager() {
        let mut context = MockAssignmentContext::default();
        let mut simplifier = Simplifier::with_default_rules();
        let message = evaluate_assignment_command_message_with_context(
            &mut context,
            &mut simplifier,
            "a",
            "x + x",
            false,
        )
        .expect("message");
        assert!(message.starts_with("a = "));
    }

    #[test]
    fn evaluate_let_assignment_command_message_with_context_formats_lazy() {
        let mut context = MockAssignmentContext::default();
        let mut simplifier = Simplifier::with_default_rules();
        let message = evaluate_let_assignment_command_message_with_context(
            &mut context,
            &mut simplifier,
            "a := x + x",
        )
        .expect("message");
        assert!(message.starts_with("a :="));
    }
}
