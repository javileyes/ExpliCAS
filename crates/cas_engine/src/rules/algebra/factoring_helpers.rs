//! Helper functions for factoring rules.
//!
//! Contains conjugate pair detection, negation checking, and structural zero
//! verification utilities used by the factoring rules.

pub(super) use cas_math::expr_relations::conjugate_add_sub_pair as is_conjugate_pair;
pub(super) use cas_math::expr_relations::conjugate_nary_add_sub_pair as is_nary_conjugate_pair;
pub(super) use cas_math::expr_relations::is_structurally_zero;
