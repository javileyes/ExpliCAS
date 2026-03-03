#[cfg(test)]
mod tests {
    use crate::extract_visualize_command_tail;

    #[test]
    fn extract_visualize_command_tail_accepts_alias() {
        assert_eq!(extract_visualize_command_tail("viz x+1"), "x+1");
    }
}
