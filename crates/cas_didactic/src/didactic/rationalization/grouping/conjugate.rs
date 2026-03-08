pub(super) struct GroupedRationalizationData {
    pub grouped_sum: String,
    pub grouped_sum_with_parens: String,
    pub grouped_conjugate: String,
}

pub(super) fn build_grouped_rationalization_data(
    group_terms: &[String],
    last_term: &str,
) -> GroupedRationalizationData {
    let grouped_sum = if group_terms.len() > 1 {
        format!("({}) + {}", group_terms.join(" + "), last_term)
    } else {
        format!("{} + {}", group_terms.join(" + "), last_term)
    };

    let grouped_conjugate = if group_terms.len() > 1 {
        format!("({}) - {}", group_terms.join(" + "), last_term)
    } else {
        format!("{} - {}", group_terms.join(" + "), last_term)
    };

    let grouped_sum_with_parens = if group_terms.len() > 1 {
        format!("({})", group_terms.join(" + "))
    } else {
        group_terms.join(" + ")
    };

    GroupedRationalizationData {
        grouped_sum,
        grouped_sum_with_parens,
        grouped_conjugate,
    }
}
