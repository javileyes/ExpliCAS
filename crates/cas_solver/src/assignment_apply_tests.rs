#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use cas_ast::ExprId;

    use crate::{
        apply_assignment_with_context, AssignmentApplyContext, AssignmentError, Simplifier,
    };

    #[derive(Default)]
    struct MockAssignmentContext {
        bindings: HashMap<String, ExprId>,
        functions: HashMap<String, (Vec<String>, ExprId)>,
        reserved: HashSet<String>,
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

        fn assignment_is_reserved_name(&self, name: &str) -> bool {
            self.reserved.contains(name)
        }
    }

    #[test]
    fn apply_assignment_rejects_empty_name() {
        let mut context = MockAssignmentContext::default();
        let mut simplifier = Simplifier::with_default_rules();
        let error = apply_assignment_with_context(&mut context, &mut simplifier, "", "1", false)
            .expect_err("error");
        assert_eq!(error, AssignmentError::EmptyName);
    }

    #[test]
    fn apply_assignment_rejects_reserved_name() {
        let mut context = MockAssignmentContext::default();
        context.reserved.insert("pi".to_string());
        let mut simplifier = Simplifier::with_default_rules();
        let error = apply_assignment_with_context(&mut context, &mut simplifier, "pi", "1", false)
            .expect_err("error");
        assert_eq!(error, AssignmentError::ReservedName("pi".to_string()));
    }

    #[test]
    fn apply_assignment_stores_eager_simplified_expression() {
        let mut context = MockAssignmentContext::default();
        let mut simplifier = Simplifier::with_default_rules();

        let expr_id =
            apply_assignment_with_context(&mut context, &mut simplifier, "a", "x + x", false)
                .expect("ok");
        let rendered = format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: &simplifier.context,
                id: expr_id
            }
        );
        assert_eq!(rendered.replace(' ', ""), "2*x");
    }

    #[test]
    fn apply_assignment_stores_lazy_unsimplified_expression() {
        let mut context = MockAssignmentContext::default();
        let mut simplifier = Simplifier::with_default_rules();

        let expr_id =
            apply_assignment_with_context(&mut context, &mut simplifier, "a", "x + x", true)
                .expect("ok");
        let rendered = format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: &simplifier.context,
                id: expr_id
            }
        );
        assert_eq!(rendered, "x + x");
    }

    #[test]
    fn apply_assignment_stores_function_definition_without_colliding_with_variables() {
        let mut context = MockAssignmentContext::default();
        let mut simplifier = Simplifier::with_default_rules();

        let expr_id =
            apply_assignment_with_context(&mut context, &mut simplifier, "f(x)", "x + 1", true)
                .expect("ok");
        let rendered = format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: &simplifier.context,
                id: expr_id
            }
        );
        assert_eq!(rendered, "x + 1");
        assert!(!context.bindings.contains_key("f"));
        assert_eq!(
            context.functions.get("f"),
            Some(&(vec!["x".to_string()], expr_id))
        );
    }
}
