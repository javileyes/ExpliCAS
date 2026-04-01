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
        functions: HashMap<String, (Vec<String>, ExprId)>,
    }

    impl AssignmentApplyContext for MockAssignmentContext {
        fn assignment_unset_binding(&mut self, name: &str) -> bool {
            self.bindings.remove(name).is_some()
        }

        fn assignment_set_binding(&mut self, name: String, expr: ExprId) {
            self.bindings.insert(name, expr);
        }

        fn assignment_unset_function(&mut self, name: &str) -> bool {
            self.functions.remove(name).is_some()
        }

        fn assignment_set_function(&mut self, name: String, params: Vec<String>, expr: ExprId) {
            self.functions.insert(name, (params, expr));
        }

        fn assignment_resolve_session_refs(
            &self,
            _ctx: &mut cas_ast::Context,
            expr: ExprId,
        ) -> Result<ExprId, String> {
            Ok(expr)
        }

        fn assignment_substitute_bindings_with_shadow(
            &self,
            _ctx: &mut cas_ast::Context,
            expr: ExprId,
            _shadow: &[&str],
        ) -> ExprId {
            expr
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

    #[test]
    fn evaluate_assignment_command_with_context_supports_function_target() {
        let mut context = MockAssignmentContext::default();
        let mut simplifier = Simplifier::with_default_rules();
        let output = evaluate_assignment_command_with_context(
            &mut context,
            &mut simplifier,
            "f(x)",
            "x + 1",
            true,
        )
        .expect("output");
        assert_eq!(output.name, "f(x)");
        assert!(output.lazy);
        assert!(context.functions.contains_key("f"));
    }
}
