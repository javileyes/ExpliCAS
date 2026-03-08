use std::hash::{Hash, Hasher};

pub(super) fn hash_number<H: Hasher>(n: &num_rational::BigRational, hasher: &mut H) {
    0u8.hash(hasher);
    n.numer().to_string().hash(hasher);
    n.denom().to_string().hash(hasher);
}

pub(super) fn hash_variable<H: Hasher, T: Hash>(name: &T, hasher: &mut H) {
    1u8.hash(hasher);
    name.hash(hasher);
}

pub(super) fn hash_constant<H: Hasher, T: core::fmt::Debug>(c: &T, hasher: &mut H) {
    2u8.hash(hasher);
    format!("{:?}", c).hash(hasher);
}

pub(super) fn hash_session_ref<H: Hasher>(id: &u64, hasher: &mut H) {
    11u8.hash(hasher);
    id.hash(hasher);
}
