//! Compatibility shim for timeline path helpers.
//! Path traversal utilities now live in `cas_formatter::path`.

pub(super) use cas_formatter::path::{
    diff_find_all_paths_to_expr, diff_find_path_to_expr, diff_find_paths_by_structure,
    extract_add_terms, find_path_to_expr, navigate_to_subexpr,
};
