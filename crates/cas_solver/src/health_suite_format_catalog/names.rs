use crate::health_category::Category;

pub(super) fn category_name_list() -> Vec<&'static str> {
    Category::all().iter().map(|c| c.as_str()).collect()
}
