use crate::health_suite_types::{Category, HealthCase};

pub(super) fn select_suite(suite: Vec<HealthCase>, filter: Option<Category>) -> Vec<HealthCase> {
    match filter {
        Some(cat) => suite.into_iter().filter(|c| c.category == cat).collect(),
        None => suite,
    }
}
