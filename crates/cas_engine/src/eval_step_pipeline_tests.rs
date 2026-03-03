#[cfg(test)]
mod tests {
    use crate::eval_step_pipeline::to_display_steps;
    use crate::step::Step;
    use cas_ast::Context;

    #[test]
    fn removes_no_op_steps() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");

        let steps = vec![
            Step::new("Real change", "SomeRule", x, y, vec![], Some(&ctx)),
            Step::new("No-op", "NoOpRule", x, x, vec![], Some(&ctx)),
            Step::new("Another change", "AnotherRule", y, x, vec![], Some(&ctx)),
        ];

        let display = to_display_steps(steps);

        assert_eq!(display.len(), 2, "No-op step should be filtered out");
        assert_eq!(display[0].description, "Real change");
        assert_eq!(display[1].description, "Another change");
    }

    #[test]
    fn empty_input_produces_empty_output() {
        let display = to_display_steps(vec![]);
        assert!(display.is_empty());
    }

    #[test]
    fn all_no_ops_produces_empty_output() {
        let mut ctx = Context::new();
        let x = ctx.var("x");

        let steps = vec![
            Step::new("No-op 1", "Rule1", x, x, vec![], Some(&ctx)),
            Step::new("No-op 2", "Rule2", x, x, vec![], Some(&ctx)),
        ];

        let display = to_display_steps(steps);
        assert!(display.is_empty(), "All no-ops should produce empty result");
    }
}
