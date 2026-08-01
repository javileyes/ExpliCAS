//! `metamorphic_simplification_tests`: familia `general`.
//!
//! Ver la cabecera de `metamorphic_simplification_tests.rs` para el contexto.

use super::*;

/// Log file path (relative to project root)
pub(super) fn log_file_path() -> PathBuf {
    // Try to use project root, fallback to current dir
    let base = env::var("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."));
    base.parent().unwrap_or(&base).join("metatest_log.jsonl")
}

/// Check if a shape signature contains negative exponent patterns
pub(super) fn shape_has_neg_exp(shape: &str) -> bool {
    shape.contains("INT_NEG")
}

/// Check if a shape signature is a Div structure
pub(super) fn shape_has_div(shape: &str) -> bool {
    shape.contains("Div(")
}

/// Generate a random expression using only "safe" operations.
///
/// Safe operations (no domain issues):
/// - Variables from the provided list
/// - Small integer constants (-3 to 3)
/// - Add, Sub, Mul
/// - Pow with small non-negative integer exponents (0-4)
/// - sin, cos (total functions)
///
/// NOT included (domain issues):
/// - Division
/// - log, ln, sqrt, root
/// - Negative exponents
fn gen_expr(vars: &[&str], depth: usize, rng: &mut Lcg) -> String {
    if depth == 0 || vars.is_empty() {
        // Leaf: variable or constant
        if vars.is_empty() || rng.pick(4) == 0 {
            // Constant
            rng.pick_i32(-3, 3).to_string()
        } else {
            // Variable
            let idx = rng.pick(vars.len() as u32) as usize;
            vars[idx].to_string()
        }
    } else {
        match rng.pick(10) {
            0 | 1 => {
                // Add
                format!(
                    "({}) + ({})",
                    gen_expr(vars, depth - 1, rng),
                    gen_expr(vars, depth - 1, rng)
                )
            }
            2 | 3 => {
                // Sub
                format!(
                    "({}) - ({})",
                    gen_expr(vars, depth - 1, rng),
                    gen_expr(vars, depth - 1, rng)
                )
            }
            4 | 5 => {
                // Mul
                format!(
                    "({}) * ({})",
                    gen_expr(vars, depth - 1, rng),
                    gen_expr(vars, depth - 1, rng)
                )
            }
            6 => {
                // Pow with small positive exponent (avoid 0 to prevent 0^0=undefined)
                let base = gen_expr(vars, depth - 1, rng);
                let exp = [1, 2, 3, 4][rng.pick(4) as usize];
                format!("({})^({})", base, exp)
            }
            7 => {
                // sin (total function)
                format!("sin({})", gen_expr(vars, depth - 1, rng))
            }
            8 => {
                // cos (total function)
                format!("cos({})", gen_expr(vars, depth - 1, rng))
            }
            _ => {
                // Bias toward leaves to avoid size explosion
                gen_expr(vars, 0, rng)
            }
        }
    }
}

/// Check symbolic equivalence using are_equivalent_extended with bucket gating.
///
/// Uses the engine's equivalence API which tracks soundness labels and
/// introduced requires, then gates the result based on bucket.
///
/// V2.15.8: Adds polynomial normalization fallback when equivalence is Unknown.
/// This enables proving identities like (x+1)^5 ≡ x^5 + 5x^4 + ... without
/// requiring simplify to auto-expand binomials.
pub(super) fn check_symbolic_equiv_bucket_aware(
    simplifier: &mut Simplifier,
    exp_expr: ExprId,
    simp_expr: ExprId,
    bucket: Bucket,
) -> SymbolicResult {
    // Fast path: structural comparison after simplification
    let (exp_simplified, _) = simplifier.simplify(exp_expr);
    let (simp_simplified, _) = simplifier.simplify(simp_expr);

    if cas_solver::runtime::compare_expr(&simplifier.context, exp_simplified, simp_simplified)
        == std::cmp::Ordering::Equal
    {
        return SymbolicResult::Pass;
    }

    // Slow path: full equivalence check with tracking
    let eq = simplifier.are_equivalent_extended(exp_expr, simp_expr);

    // V2.15.8: Polynomial normalization fallback for Unknown results
    // This catches cases like (x+1)^5 vs expanded polynomial where simplify
    // doesn't expand but the expressions are polynomially equivalent
    let eq = if matches!(eq, EquivalenceResult::Unknown) {
        if let Some(poly_result) =
            check_polynomial_equivalence(&simplifier.context, exp_simplified, simp_simplified)
        {
            poly_result
        } else {
            eq
        }
    } else {
        eq
    };

    match (&bucket, eq) {
        // Unconditional bucket: only pure True counts as symbolic pass
        (Bucket::Unconditional, EquivalenceResult::True) => SymbolicResult::Pass,
        (Bucket::Unconditional, EquivalenceResult::ConditionalTrue { requires }) => {
            SymbolicResult::Conditional(requires) // NOT symbolic pass, falls to numeric
        }
        (Bucket::Unconditional, EquivalenceResult::False) => SymbolicResult::Fail,
        (Bucket::Unconditional, EquivalenceResult::Unknown) => SymbolicResult::Unknown,

        // ConditionalRequires: conditional counts as pass
        (Bucket::ConditionalRequires, EquivalenceResult::True) => SymbolicResult::Pass,
        (Bucket::ConditionalRequires, EquivalenceResult::ConditionalTrue { requires }) => {
            SymbolicResult::PassConditional(requires)
        }
        (Bucket::ConditionalRequires, EquivalenceResult::False) => SymbolicResult::Fail,
        (Bucket::ConditionalRequires, EquivalenceResult::Unknown) => SymbolicResult::Unknown,

        // BranchSensitive: skip symbolic except for pure True
        (Bucket::BranchSensitive, EquivalenceResult::True) => SymbolicResult::Pass,
        (Bucket::BranchSensitive, _) => SymbolicResult::SkipSymbolic,
    }
}

/// V2.15.8: Check if two expressions are equivalent as polynomials.
/// Returns Some(True) if they canonicalize to the same polynomial,
/// None if either expression is not a polynomial (contains trig, log, etc.)
fn check_polynomial_equivalence(ctx: &Context, a: ExprId, b: ExprId) -> Option<EquivalenceResult> {
    use cas_math::multipoly::{multipoly_from_expr, PolyBudget};

    // Use a generous budget for polynomial equivalence checking
    // (higher than normal since this is for testing, not runtime)
    let budget = PolyBudget {
        max_terms: 500,       // Allow up to 500 terms (covers (x+1)^8 etc.)
        max_total_degree: 15, // Allow up to degree 15
        max_pow_exp: 10,      // Allow exponents up to 10
    };

    let pa = multipoly_from_expr(ctx, a, &budget).ok()?;
    let pb = multipoly_from_expr(ctx, b, &budget).ok()?;

    if pa == pb {
        Some(EquivalenceResult::True)
    } else {
        // Polynomials are different - this is a definite non-equivalence
        Some(EquivalenceResult::False)
    }
}

/// Truncate an identity string for display (avoids log bloat)
pub(super) fn truncate_identity(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}

/// Format f64 for stable display (avoids "2" vs "2.0" inconsistency)
pub(super) fn fmt_f64(x: f64) -> String {
    if x.fract().abs() < 1e-12 {
        format!("{:.0}", x)
    } else {
        format!("{:.6}", x)
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string()
    }
}

/// Parse METATEST_TRANSFORMS env var.
/// Format: "scale:2,scale:-1,shift:1,square"
/// Returns defaults (scale:2, scale:-1) if not set.
pub(super) fn parse_meta_transforms_from_env() -> Vec<MetaTransform> {
    let raw = env::var("METATEST_TRANSFORMS").ok().unwrap_or_default();
    let raw = raw.trim();

    if raw.is_empty() {
        // Defaults: scale(2), scale(-1) - very safe transforms
        return vec![MetaTransform::Scale(2.0), MetaTransform::Scale(-1.0)];
    }

    parse_meta_transforms(raw)
}

/// Parse transform spec string
fn parse_meta_transforms(spec: &str) -> Vec<MetaTransform> {
    let mut out = Vec::new();

    for (idx, item) in spec.split(',').enumerate() {
        let item = item.trim();
        if item.is_empty() {
            continue;
        }

        // "square" without parameter
        if item.eq_ignore_ascii_case("square") {
            out.push(MetaTransform::Square);
            continue;
        }

        // Format with ':'
        let (kind, val_str) = match item.split_once(':') {
            Some(parts) => parts,
            None => panic!(
                "Invalid METATEST_TRANSFORMS item #{}: '{}'. Expected 'scale:<num>', 'shift:<num>', or 'square'. Full spec: '{}'",
                idx + 1, item, spec
            ),
        };

        let kind = kind.trim();
        let val_str = val_str.trim();

        let val: f64 = val_str.parse().unwrap_or_else(|e| {
            panic!(
                "Invalid numeric value in METATEST_TRANSFORMS item #{}: '{}'. Error: {}. Full spec: '{}'",
                idx + 1, item, e, spec
            )
        });

        if !val.is_finite() {
            panic!(
                "Value must be finite in METATEST_TRANSFORMS item #{}: '{}'. Full spec: '{}'",
                idx + 1,
                item,
                spec
            );
        }

        match kind.to_lowercase().as_str() {
            "scale" => out.push(MetaTransform::Scale(val)),
            "shift" => out.push(MetaTransform::Shift(val)),
            _ => panic!(
                "Unknown transform kind in METATEST_TRANSFORMS item #{}: '{}'. Supported: scale, shift, square. Full spec: '{}'",
                idx + 1, item, spec
            ),
        }
    }

    // Dedup (stable order)
    let mut seen: Vec<MetaTransform> = Vec::new();
    out.retain(|t| {
        if seen.iter().any(|x| x == t) {
            false
        } else {
            seen.push(t.clone());
            true
        }
    });

    if out.is_empty() {
        panic!(
            "METATEST_TRANSFORMS parsed to empty list. Spec: '{}'. Example: 'scale:2,scale:-1,shift:1,square'",
            spec
        );
    }

    out
}

/// Collect all addends from a flattened Add tree (recursive)
/// Returns vec of ExprIds in order they appear in the tree
pub(super) fn collect_addends(ctx: &Context, expr: ExprId) -> Vec<ExprId> {
    match ctx.nodes.get(expr.index()) {
        Some(cas_ast::Expr::Add(a, b)) => {
            let mut result = collect_addends(ctx, *a);
            result.extend(collect_addends(ctx, *b));
            result
        }
        _ => vec![expr],
    }
}

/// Collect only the immediate top-level addends of an Add node.
/// This preserves contextual grouping like `(A + B)` vs `(C + D)` before
/// flattening nested sums inside each side.
pub(super) fn collect_shallow_addends(ctx: &Context, expr: ExprId) -> Vec<ExprId> {
    match ctx.nodes.get(expr.index()) {
        Some(cas_ast::Expr::Add(a, b)) => vec![*a, *b],
        _ => vec![expr],
    }
}

pub(super) fn collect_signed_shallow_addends(ctx: &Context, expr: ExprId) -> Vec<(i8, ExprId)> {
    match ctx.nodes.get(expr.index()) {
        Some(cas_ast::Expr::Add(a, b)) => vec![(1, *a), (1, *b)],
        Some(cas_ast::Expr::Sub(a, b)) => vec![(1, *a), (-1, *b)],
        _ => vec![(1, expr)],
    }
}

/// Collect all factors from a flattened Mul tree (recursive)
pub(super) fn collect_factors(ctx: &Context, expr: ExprId) -> Vec<ExprId> {
    match ctx.nodes.get(expr.index()) {
        Some(cas_ast::Expr::Mul(a, b)) => {
            let mut result = collect_factors(ctx, *a);
            result.extend(collect_factors(ctx, *b));
            result
        }
        _ => vec![expr],
    }
}

/// Rebuild Add tree from terms (left-associative)
pub(super) fn rebuild_add(ctx: &mut Context, terms: &[ExprId]) -> ExprId {
    if terms.is_empty() {
        ctx.add_raw(cas_ast::Expr::Number(
            num_rational::BigRational::from_integer(0.into()),
        ))
    } else if terms.len() == 1 {
        terms[0]
    } else {
        let mut result = terms[0];
        for &term in &terms[1..] {
            result = ctx.add_raw(cas_ast::Expr::Add(result, term));
        }
        result
    }
}

/// Rebuild Mul tree from factors (left-associative)
pub(super) fn rebuild_mul(ctx: &mut Context, factors: &[ExprId]) -> ExprId {
    if factors.is_empty() {
        ctx.add_raw(cas_ast::Expr::Number(
            num_rational::BigRational::from_integer(1.into()),
        ))
    } else if factors.len() == 1 {
        factors[0]
    } else {
        let mut result = factors[0];
        for &factor in &factors[1..] {
            result = ctx.add_raw(cas_ast::Expr::Mul(result, factor));
        }
        result
    }
}

/// Stable hash for an expression (FNV-1a based, deterministic)
fn stable_expr_hash(ctx: &Context, expr: ExprId) -> u64 {
    const FNV_OFFSET: u64 = 14695981039346656037;
    const FNV_PRIME: u64 = 1099511628211;

    fn hash_combine(hash: u64, byte: u8) -> u64 {
        (hash ^ (byte as u64)).wrapping_mul(FNV_PRIME)
    }

    fn hash_u64(hash: u64, val: u64) -> u64 {
        let mut h = hash;
        for i in 0..8 {
            h = hash_combine(h, ((val >> (i * 8)) & 0xff) as u8);
        }
        h
    }

    fn hash_expr(ctx: &Context, expr: ExprId, h: u64) -> u64 {
        match ctx.nodes.get(expr.index()) {
            Some(cas_ast::Expr::Number(n)) => {
                let mut h = hash_combine(h, b'N');
                for b in n.to_string().bytes() {
                    h = hash_combine(h, b);
                }
                h
            }
            Some(cas_ast::Expr::Variable(sym_id)) => {
                let mut h = hash_combine(h, b'V');
                // sym_id is a SymbolId (usize), need to convert to string representation
                // Since we don't have Context here, use the raw id as bytes
                for b in sym_id.to_string().bytes() {
                    h = hash_combine(h, b);
                }
                h
            }
            Some(cas_ast::Expr::Add(a, b)) => {
                let h = hash_combine(h, b'+');
                let h = hash_expr(ctx, *a, h);
                hash_expr(ctx, *b, h)
            }
            Some(cas_ast::Expr::Mul(a, b)) => {
                let h = hash_combine(h, b'*');
                let h = hash_expr(ctx, *a, h);
                hash_expr(ctx, *b, h)
            }
            Some(cas_ast::Expr::Pow(base, exp)) => {
                let h = hash_combine(h, b'^');
                let h = hash_expr(ctx, *base, h);
                hash_expr(ctx, *exp, h)
            }
            Some(cas_ast::Expr::Function(name_id, args)) => {
                let mut h = hash_combine(h, b'F');
                for b in ctx.sym_name(*name_id).bytes() {
                    h = hash_combine(h, b);
                }
                for arg in args {
                    h = hash_expr(ctx, *arg, h);
                }
                h
            }
            Some(cas_ast::Expr::Neg(inner)) => {
                let h = hash_combine(h, b'-');
                hash_expr(ctx, *inner, h)
            }
            Some(cas_ast::Expr::Sub(a, b)) => {
                let h = hash_combine(h, b'S');
                let h = hash_expr(ctx, *a, h);
                hash_expr(ctx, *b, h)
            }
            Some(cas_ast::Expr::Div(a, b)) => {
                let h = hash_combine(h, b'/');
                let h = hash_expr(ctx, *a, h);
                hash_expr(ctx, *b, h)
            }
            Some(cas_ast::Expr::Constant(c)) => {
                let mut h = hash_combine(h, b'C');
                for b in format!("{:?}", c).bytes() {
                    h = hash_combine(h, b);
                }
                h
            }
            _ => hash_combine(h, b'?'),
        }
    }

    hash_expr(ctx, expr, FNV_OFFSET)
}

/// Deterministic shuffle based on expr hash (Fisher-Yates with seeded PRNG)
pub(super) fn shuffle_vec<T>(items: &mut [T], seed: u64) {
    let mut rng = seed;
    for i in (1..items.len()).rev() {
        // Simple LCG PRNG
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let j = (rng as usize) % (i + 1);
        items.swap(i, j);
    }
}

/// Shuffle an expression by permuting Add/Mul children deterministically
/// Only touches commutative nodes (Add, Mul), preserves structure of other nodes
fn shuffle_expr(ctx: &mut Context, expr: ExprId) -> ExprId {
    let seed = stable_expr_hash(ctx, expr);
    shuffle_expr_seeded(ctx, expr, seed)
}

pub(super) fn build_nvar_slice_anchors(
    ctx: &Context,
    a: ExprId,
    b: ExprId,
    vars: &[String],
    filters: &[FilterSpec],
    config: &MetatestConfig,
    seed: f64,
) -> Vec<(String, f64)> {
    let profile_order = choose_numeric_sample_profile_order_exprs(ctx, a, b);
    let (denom_guards, analytic_guards) = collect_numeric_precheck_guards(ctx, a, b);
    const OFFSETS: [f64; 6] = [0.0, 0.125, 0.25, 0.375, 0.5, 0.75];

    for offset in OFFSETS {
        let effective_seed = (seed + offset).fract();
        let anchors = vars
            .iter()
            .enumerate()
            .map(|(idx, var)| {
                let filter = filters.get(idx).unwrap_or(&FilterSpec::None);
                (
                    var.clone(),
                    sample_nvar_slice_anchor_filtered(
                        config,
                        idx,
                        effective_seed,
                        filter,
                        profile_order.as_ref(),
                    ),
                )
            })
            .collect::<Vec<_>>();

        let var_map = anchors.iter().cloned().collect::<HashMap<String, f64>>();

        if !sample_violates_numeric_precheck_guards(ctx, &denom_guards, &analytic_guards, &var_map)
        {
            return anchors;
        }
    }

    vars.iter()
        .enumerate()
        .map(|(idx, var)| {
            let filter = filters.get(idx).unwrap_or(&FilterSpec::None);
            (
                var.clone(),
                sample_nvar_slice_anchor_filtered(
                    config,
                    idx,
                    seed,
                    filter,
                    profile_order.as_ref(),
                ),
            )
        })
        .collect()
}

pub(super) fn normal_forms_visibly_equal(ctx: &Context, lhs: ExprId, rhs: ExprId) -> bool {
    if cas_solver::runtime::compare_expr(ctx, lhs, rhs) == std::cmp::Ordering::Equal {
        return true;
    }

    let lhs_text = expr_text(ctx, lhs);
    let rhs_text = expr_text(ctx, rhs);
    if lhs_text != rhs_text {
        return false;
    }

    let mut fresh = Context::new();
    match (parse(&lhs_text, &mut fresh), parse(&rhs_text, &mut fresh)) {
        (Ok(lhs_reparsed), Ok(rhs_reparsed)) => {
            cas_solver::runtime::compare_expr(&fresh, lhs_reparsed, rhs_reparsed)
                == std::cmp::Ordering::Equal
        }
        _ => false,
    }
}

pub(super) fn mask_includes(mask: u32, index: usize) -> bool {
    (mask & (1u32 << index)) != 0
}

pub(super) fn build_group_from_mask(ctx: &mut Context, terms: &[ExprId], mask: u32) -> ExprId {
    let mut selected = Vec::new();
    for (idx, &term) in terms.iter().enumerate() {
        if mask_includes(mask, idx) {
            selected.push(term);
        }
    }
    rebuild_add(ctx, &selected)
}

pub(super) fn filter_terms_by_mask(
    terms: &[ExprId],
    mask: u32,
    keep_selected: bool,
) -> Vec<ExprId> {
    let mut out = Vec::new();
    for (idx, &term) in terms.iter().enumerate() {
        let selected = mask_includes(mask, idx);
        if selected == keep_selected {
            out.push(term);
        }
    }
    out
}

pub(super) fn curated_pair_corpus() -> &'static CuratedPairCorpus {
    static CURATED: OnceLock<CuratedPairCorpus> = OnceLock::new();
    CURATED.get_or_init(|| {
        let mut raw = HashSet::new();
        let mut alpha = HashSet::new();

        let mut insert_pair = |lhs: &str, rhs: &str| {
            let lhs_raw = normalize_pair_text(lhs);
            let rhs_raw = normalize_pair_text(rhs);
            raw.insert((lhs_raw.clone(), rhs_raw.clone()));
            raw.insert((rhs_raw, lhs_raw));

            if let (Some(lhs_alpha), Some(rhs_alpha)) = (
                alpha_normalize_pair_text(lhs),
                alpha_normalize_pair_text(rhs),
            ) {
                alpha.insert((lhs_alpha.clone(), rhs_alpha.clone()));
                alpha.insert((rhs_alpha, lhs_alpha));
            }
        };

        for pair in load_contextual_pairs()
            .into_iter()
            .chain(load_contextual_rational_pairs().into_iter())
            .chain(load_contextual_trig_pairs().into_iter())
            .chain(load_contextual_polynomial_pairs().into_iter())
            .chain(load_contextual_radical_pairs().into_iter())
            .chain(load_residual_pairs().into_iter())
        {
            insert_pair(&pair.lhs, &pair.rhs);
        }
        for pair in load_identity_pairs()
            .into_iter()
            .chain(load_substitution_identities().into_iter())
        {
            insert_pair(&pair.exp, &pair.simp);
        }

        CuratedPairCorpus { raw, alpha }
    })
}

pub(super) fn residual_pair_corpus() -> &'static CuratedPairCorpus {
    static RESIDUAL: OnceLock<CuratedPairCorpus> = OnceLock::new();
    RESIDUAL.get_or_init(|| {
        let mut raw = HashSet::new();
        let mut alpha = HashSet::new();

        let mut insert_pair = |lhs: &str, rhs: &str| {
            let lhs_raw = normalize_pair_text(lhs);
            let rhs_raw = normalize_pair_text(rhs);
            raw.insert((lhs_raw.clone(), rhs_raw.clone()));
            raw.insert((rhs_raw, lhs_raw));

            if let (Some(lhs_alpha), Some(rhs_alpha)) = (
                alpha_normalize_pair_text(lhs),
                alpha_normalize_pair_text(rhs),
            ) {
                alpha.insert((lhs_alpha.clone(), rhs_alpha.clone()));
                alpha.insert((rhs_alpha, lhs_alpha));
            }
        };

        for pair in load_residual_pairs() {
            insert_pair(&pair.lhs, &pair.rhs);
        }

        CuratedPairCorpus { raw, alpha }
    })
}

pub(super) fn strip_wrapping_parens(mut text: &str) -> &str {
    loop {
        if !(text.starts_with('(') && text.ends_with(')')) {
            return text;
        }
        let mut depth = 0usize;
        let mut wraps_entire_expr = true;
        for (idx, ch) in text.char_indices() {
            match ch {
                '(' => depth += 1,
                ')' => {
                    depth = depth.saturating_sub(1);
                    if depth == 0 && idx + ch.len_utf8() != text.len() {
                        wraps_entire_expr = false;
                        break;
                    }
                }
                _ => {}
            }
        }
        if !wraps_entire_expr {
            return text;
        }
        text = &text[1..text.len() - 1];
    }
}

pub(super) fn abs_square_identity_matches(lhs_text: &str, rhs_text: &str) -> bool {
    fn side_matches(abs_side: &str, plain_side: &str) -> bool {
        let Some(abs_inner) = abs_side
            .strip_prefix('|')
            .and_then(|s| s.strip_suffix("|^2"))
        else {
            return false;
        };
        let Some(plain_inner) = plain_side.strip_suffix("^2") else {
            return false;
        };
        strip_wrapping_parens(abs_inner) == strip_wrapping_parens(plain_inner)
    }

    let lhs = normalize_metamorphic_text(lhs_text);
    let rhs = normalize_metamorphic_text(rhs_text);
    side_matches(&lhs, &rhs) || side_matches(&rhs, &lhs)
}

pub(super) fn atan_double_angle_identity_matches(lhs_text: &str, rhs_text: &str) -> bool {
    fn side_matches(inv_side: &str, rational_side: &str) -> bool {
        let inv_side = strip_wrapping_parens(inv_side);
        let (prefix, numerator_prefix) =
            if inv_side.starts_with("sin(2*arctan(") && inv_side.ends_with("))") {
                ("sin(2*arctan(", "2*")
            } else if inv_side.starts_with("cos(2*arctan(") && inv_side.ends_with("))") {
                ("cos(2*arctan(", "1-")
            } else {
                return false;
            };

        let inner = &inv_side[prefix.len()..inv_side.len() - 2];
        let expected = if numerator_prefix == "2*" {
            format!("2*{inner}/(1+{inner}^2)")
        } else {
            format!("(1-{inner}^2)/(1+{inner}^2)")
        };

        strip_wrapping_parens(rational_side) == strip_wrapping_parens(&expected)
    }

    let lhs = normalize_metamorphic_text(lhs_text);
    let rhs = normalize_metamorphic_text(rhs_text);
    side_matches(&lhs, &rhs) || side_matches(&rhs, &lhs)
}

pub(super) fn three_linear_shift_anchor_partner_identity_matches(
    lhs_text: &str,
    rhs_text: &str,
) -> bool {
    fn side_matches(factored_side: &str, expanded_side: &str) -> bool {
        let Some((factored_base, factored_partner)) =
            extract_three_linear_shift_anchor_and_partner_text(factored_side)
        else {
            return false;
        };
        let Some((expanded_base, expanded_partner)) =
            extract_three_linear_shift_expanded_and_partner_text(expanded_side)
        else {
            return false;
        };
        if factored_base != expanded_base {
            return false;
        }

        matches_double_angle_arcsin_partner_text(&factored_partner, &expanded_partner)
            || matches_small_radical_product_partner_text(&factored_partner, &expanded_partner)
    }

    let lhs = normalize_metamorphic_text(lhs_text);
    let rhs = normalize_metamorphic_text(rhs_text);
    side_matches(&lhs, &rhs) || side_matches(&rhs, &lhs)
}

pub(super) fn small_pow_anchor_partner_identity_matches(lhs_text: &str, rhs_text: &str) -> bool {
    fn side_matches(factored_side: &str, expanded_side: &str) -> bool {
        let factored_factors =
            split_top_level_mul_factors_text(strip_wrapping_parens(factored_side));
        let expanded_factors =
            split_top_level_mul_factors_text(strip_wrapping_parens(expanded_side));
        if factored_factors.len() != 2 || expanded_factors.len() < 2 {
            return false;
        }

        for factored_anchor_index in 0..factored_factors.len() {
            let factored_anchor = strip_wrapping_parens(factored_factors[factored_anchor_index]);
            let factored_partner =
                strip_wrapping_parens(factored_factors[1 - factored_anchor_index]);

            for expanded_anchor_index in 0..expanded_factors.len() {
                let expanded_anchor =
                    strip_wrapping_parens(expanded_factors[expanded_anchor_index]);
                let remaining_partner_factors = expanded_factors
                    .iter()
                    .enumerate()
                    .filter_map(|(index, factor)| {
                        (index != expanded_anchor_index).then_some(strip_wrapping_parens(factor))
                    })
                    .collect::<Vec<_>>();
                if remaining_partner_factors.is_empty() {
                    continue;
                }
                let expanded_partner = if remaining_partner_factors.len() == 1 {
                    remaining_partner_factors[0].to_string()
                } else {
                    remaining_partner_factors.join("*")
                };

                let anchors_match =
                    prove_zero_from_curated_pair_corpus_text(factored_anchor, expanded_anchor)
                        || prove_zero_from_residual_pair_corpus_text(
                            factored_anchor,
                            expanded_anchor,
                        )
                        || prove_equiv_expr_texts_fresh(factored_anchor, expanded_anchor)
                        || prove_zero_via_wire_eval(factored_anchor, expanded_anchor);
                if !anchors_match {
                    continue;
                }

                let partners_match =
                    matches_double_angle_arcsin_partner_text(factored_partner, &expanded_partner)
                        || matches_small_radical_product_partner_text(
                            factored_partner,
                            &expanded_partner,
                        )
                        || prove_zero_from_curated_pair_corpus_text(
                            factored_partner,
                            &expanded_partner,
                        )
                        || prove_zero_from_residual_pair_corpus_text(
                            factored_partner,
                            &expanded_partner,
                        )
                        || prove_zero_via_wire_eval(factored_partner, &expanded_partner);
                if partners_match {
                    return true;
                }
            }
        }

        false
    }

    let lhs = normalize_metamorphic_text(lhs_text);
    let rhs = normalize_metamorphic_text(rhs_text);
    side_matches(&lhs, &rhs) || side_matches(&rhs, &lhs)
}

pub(super) fn sum_of_squares_anchor_partner_identity_matches(
    lhs_text: &str,
    rhs_text: &str,
) -> bool {
    fn side_matches(factored_side: &str, expanded_side: &str) -> bool {
        let Some((factored_anchor, factored_partner)) =
            extract_sum_of_squares_anchor_and_partner_text(factored_side)
        else {
            return false;
        };

        let expanded_factors =
            split_top_level_mul_factors_text(strip_wrapping_parens(expanded_side));
        if expanded_factors.len() != 2 {
            return false;
        }

        for anchor_index in 0..2 {
            let partner_index = 1 - anchor_index;
            let expanded_anchor = strip_wrapping_parens(expanded_factors[anchor_index]);
            let expanded_partner = strip_wrapping_parens(expanded_factors[partner_index]);
            if !prove_equiv_expr_texts_fresh(&factored_anchor, expanded_anchor) {
                continue;
            }
            if strip_wrapping_parens(&factored_partner) == expanded_partner
                || prove_zero_from_curated_pair_corpus_text(&factored_partner, expanded_partner)
                || prove_equiv_expr_texts_fresh(&factored_partner, expanded_partner)
            {
                return true;
            }
        }

        false
    }

    let lhs = normalize_metamorphic_text(lhs_text);
    let rhs = normalize_metamorphic_text(rhs_text);
    side_matches(&lhs, &rhs) || side_matches(&rhs, &lhs)
}

pub(super) fn half_angle_identity_matches(lhs_text: &str, rhs_text: &str) -> bool {
    fn side_matches(half_side: &str, direct_side: &str) -> bool {
        let half_side = strip_wrapping_parens(half_side);
        let (prefix, suffix, direct_prefix) =
            if half_side.starts_with("2*sin(") && half_side.ends_with("/2)^2") {
                ("2*sin(", "/2)^2", "1-cos(")
            } else if half_side.starts_with("2*cos(") && half_side.ends_with("/2)^2") {
                ("2*cos(", "/2)^2", "1+cos(")
            } else {
                return false;
            };

        let direct_side = strip_wrapping_parens(direct_side);
        let Some(direct_arg) = direct_side
            .strip_prefix(direct_prefix)
            .and_then(|s| s.strip_suffix(')'))
        else {
            return false;
        };
        let inner_raw = &half_side[prefix.len()..half_side.len() - suffix.len()];
        strip_wrapping_parens(direct_arg) == strip_wrapping_parens(inner_raw)
    }

    let lhs = normalize_metamorphic_text(lhs_text);
    let rhs = normalize_metamorphic_text(rhs_text);
    side_matches(&lhs, &rhs) || side_matches(&rhs, &lhs)
}

pub(super) fn nf_converges_in_child_process(lhs: &str, rhs: &str) -> bool {
    let Ok(current_exe) = std::env::current_exe() else {
        return false;
    };

    let mut child = match std::process::Command::new(current_exe)
        // `super::`, not `crate::`: cas_engine's compatibility wrapper includes
        // this harness one module deeper, where `crate::runners` does not exist.
        .arg(super::runners::CHILD_NF_CONVERGENCE.filter())
        .arg("--ignored")
        .arg("--exact")
        .arg("--nocapture")
        .env(METATEST_CHILD_NF_LHS_ENV, lhs)
        .env(METATEST_CHILD_NF_RHS_ENV, rhs)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
    {
        Ok(child) => child,
        Err(_) => return false,
    };

    let timeout = std::time::Duration::from_millis(METATEST_CHILD_NF_TIMEOUT_MS);
    let start = std::time::Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return status.success(),
            Ok(None) => {
                if start.elapsed() >= timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    return false;
                }
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            Err(_) => {
                let _ = child.kill();
                let _ = child.wait();
                return false;
            }
        }
    }
}

pub(super) fn encode_child_vars(vars: &[String]) -> String {
    vars.join("|")
}

pub(super) fn decode_child_vars(spec: &str) -> Vec<String> {
    if spec.is_empty() {
        Vec::new()
    } else {
        spec.split('|').map(str::to_string).collect()
    }
}

pub(super) fn encode_child_filters(filters: &[FilterSpec]) -> String {
    filters
        .iter()
        .map(FilterSpec::as_str)
        .collect::<Vec<_>>()
        .join("|")
}

pub(super) fn decode_child_filters(spec: &str) -> Vec<FilterSpec> {
    if spec.is_empty() {
        Vec::new()
    } else {
        spec.split('|')
            .map(|item| parse_filter_spec(item, 0))
            .collect()
    }
}

pub(super) fn known_symbolic_residual_reason(
    _lhs_text: &str,
    _rhs_text: &str,
) -> Option<&'static str> {
    None
}

#[test]
fn curated_pair_corpus_rejects_unlisted_pair() {
    let lhs = "x + 1";
    let rhs = "x + 2";
    assert!(!prove_zero_from_curated_pair_corpus_text(lhs, rhs));
}

#[test]
fn pair_symbolic_proof_accepts_curated_and_abs_square_pairs() {
    let curated_pair = IdentityPair {
        exp: "1/a + 1/(a+1)".to_string(),
        simp: "(2*a+1)/(a*(a+1))".to_string(),
        vars: vec!["a".to_string()],
        mode: DomainRequirement::Generic,
        bucket: Bucket::ConditionalRequires,
        branch_mode: BranchMode::PrincipalStrict,
        filter_spec: FilterSpec::None,
        family: "Addition of fractions".to_string(),
    };
    assert!(pair_is_symbolically_proved(&curated_pair));

    let abs_square_pair = IdentityPair {
        exp: "|cos(x)|^2".to_string(),
        simp: "cos(x)^2".to_string(),
        vars: vec!["x".to_string()],
        mode: DomainRequirement::Generic,
        bucket: Bucket::ConditionalRequires,
        branch_mode: BranchMode::PrincipalStrict,
        filter_spec: FilterSpec::None,
        family: "Absolute value and powers".to_string(),
    };
    assert!(pair_is_symbolically_proved(&abs_square_pair));
}

#[test]
fn three_linear_shift_anchor_partner_identity_matches_double_angle_inverse_trig_pair() {
    let lhs = "((x+1)*(x+2)*(x+3)) * (sin(2*arcsin(u)))";
    let rhs = "(x^3 + 6*x^2 + 11*x + 6) * (2*u*sqrt(1-u^2))";
    assert!(three_linear_shift_anchor_partner_identity_matches(lhs, rhs));
    assert!(three_linear_shift_anchor_partner_identity_matches(rhs, lhs));
}

#[test]
fn three_linear_shift_anchor_partner_identity_matches_small_radical_product_pair() {
    let lhs = "((x+1)*(x+2)*(x+3)) * (sqrt(u)*sqrt(4*u))";
    let rhs = "(x^3 + 6*x^2 + 11*x + 6) * (2*u)";
    assert!(three_linear_shift_anchor_partner_identity_matches(lhs, rhs));
    assert!(three_linear_shift_anchor_partner_identity_matches(rhs, lhs));
}

#[test]
fn small_pow_anchor_partner_identity_matches_double_angle_inverse_trig_pair() {
    let lhs = "((x-1)^5) * (sin(2*arcsin(u)))";
    let rhs = "(x^5 - 5*x^4 + 10*x^3 - 10*x^2 + 5*x - 1) * (2*u*sqrt(1-u^2))";
    assert!(small_pow_anchor_partner_identity_matches(lhs, rhs));
    assert!(small_pow_anchor_partner_identity_matches(rhs, lhs));
}

#[test]
fn sum_of_squares_anchor_partner_identity_matches_sum_of_cubes_pair() {
    let lhs = "((x^2 + y^2)*(a^2 + b^2)) * (u^3 + v^3)";
    let rhs = "((x*a + y*b)^2 + (x*b - y*a)^2) * ((u+v)*(u^2-u*v+v^2))";
    assert!(sum_of_squares_anchor_partner_identity_matches(lhs, rhs));
    assert!(sum_of_squares_anchor_partner_identity_matches(rhs, lhs));
}

#[test]
fn sum_of_squares_anchor_partner_identity_matches_higher_degree_difference_pair() {
    let lhs = "((x^2 + y^2)*(a^2 + b^2)) * (u^6 - 1)";
    let rhs = "((x*a + y*b)^2 + (x*b - y*a)^2) * ((u^2+u+1)*(u^2-u+1)*(u+1)*(u-1))";
    assert!(sum_of_squares_anchor_partner_identity_matches(lhs, rhs));
    assert!(sum_of_squares_anchor_partner_identity_matches(rhs, lhs));
}

#[test]
fn sum_of_squares_anchor_partner_identity_matches_identical_partner_pair() {
    let lhs = "((x^2 + y^2)*(a^2 + b^2)) * (u^2 + v^2)";
    let rhs = "((x*a + y*b)^2 + (x*b - y*a)^2) * (u^2 + v^2)";
    assert!(sum_of_squares_anchor_partner_identity_matches(lhs, rhs));
    assert!(sum_of_squares_anchor_partner_identity_matches(rhs, lhs));
}

#[test]
fn child_hint_uses_three_linear_shift_partner_identity_matcher() {
    let lhs = "((x+1)*(x+2)*(x+3)) * (sin(2*arcsin(u)))";
    let rhs = "(x^3 + 6*x^2 + 11*x + 6) * (2*u*sqrt(1-u^2))";
    assert!(prove_zero_from_engine_texts_child_hint(lhs, rhs));
}

#[test]
fn child_hint_uses_small_pow_anchor_partner_identity_matcher() {
    let lhs = "((x-1)^5) * (sin(2*arcsin(u)))";
    let rhs = "(x^5 - 5*x^4 + 10*x^3 - 10*x^2 + 5*x - 1) * (2*u*sqrt(1-u^2))";
    assert!(prove_zero_from_engine_texts_child_hint(lhs, rhs));
}

#[test]
fn child_hint_uses_sum_of_squares_anchor_partner_identity_matcher() {
    let lhs = "((x^2 + y^2)*(a^2 + b^2)) * (u^3 - 1)";
    let rhs = "((x*a + y*b)^2 + (x*b - y*a)^2) * ((u - 1)*(u^2 + u + 1))";
    assert!(prove_zero_from_engine_texts_child_hint(lhs, rhs));
}

#[test]
fn residual_pair_corpus_detects_inverse_trig_rational_ctx_pair() {
    let lhs = "sin(arctan((1/u + 1/(u+1))))";
    let rhs = "(1/u + 1/(u+1))/sqrt(1 + (1/u + 1/(u+1))^2)";

    assert!(prove_zero_from_residual_pair_corpus_text(lhs, rhs));
}

/// Alpha-rename a variable in an expression string.
/// Simple text replacement - works for our test expressions.
fn alpha_rename(expr: &str, from: &str, to: &str) -> String {
    // Use word boundaries to avoid replacing 'x' inside 'exp' etc.
    // Simple approach: replace 'x' followed by non-alphanumeric or end
    let mut result = expr.to_string();

    // Replace patterns like "x)" "x+" "x-" "x*" "x/" "x^" "x," "x " "|x|" and standalone "x"
    let patterns = [
        (format!("{})", from), format!("{})", to)),
        (format!("{}+", from), format!("{}+", to)),
        (format!("{}-", from), format!("{}-", to)),
        (format!("{}*", from), format!("{}*", to)),
        (format!("{}/", from), format!("{}/", to)),
        (format!("{}^", from), format!("{}^", to)),
        (format!("{},", from), format!("{},", to)),
        (format!("{} ", from), format!("{} ", to)),
        (format!("({})", from), format!("({})", to)),
        (format!("({}", from), format!("({}", to)),
        // Absolute value: |x|
        (format!("|{}|", from), format!("|{}|", to)),
        (format!("|{}", from), format!("|{}", to)),
        (format!("{}|", from), format!("{}|", to)),
    ];

    for (pat, rep) in &patterns {
        result = result.replace(pat, rep);
    }

    // Handle end of string
    if result.ends_with(from) {
        let len = result.len();
        result.replace_range(len - from.len().., to);
    }

    result
}

pub(super) fn alpha_rename_many(expr: &str, renames: &[(String, String)]) -> String {
    let mut result = expr.to_string();
    let staged: Vec<(String, String, String)> = renames
        .iter()
        .enumerate()
        .map(|(idx, (from, to))| (from.clone(), format!("__tmp_var_{idx}__"), to.clone()))
        .collect();

    for (from, temp, _) in &staged {
        result = alpha_rename(&result, from, temp);
    }

    for (_, temp, to) in &staged {
        result = alpha_rename(&result, temp, to);
    }

    result
}

pub(super) fn identity_filters(pair: &IdentityPair) -> Vec<FilterSpec> {
    pair.vars
        .iter()
        .enumerate()
        .map(|(idx, _)| {
            if idx == 0 {
                pair.filter_spec.clone()
            } else {
                FilterSpec::None
            }
        })
        .collect()
}

pub(super) fn push_nf_first_mul_div_skip_example(
    examples: &mut Vec<String>,
    max_examples: usize,
    details: NfFirstMulDivSkipExample<'_>,
) {
    if examples.len() >= max_examples {
        return;
    }

    let diagnose = |label: &str, text: &str| match parse_text_probe(text) {
        Ok(()) => format!("{label}: OK"),
        Err(err) => format!("{label}: ERR {err} :: {text}"),
    };

    let lines = [
        format!(
            "[{}] [{}] ({}) * [{}] ({})",
            details.op_name,
            details.pair1.family,
            details.pair1.exp,
            details.pair2.family,
            details.pair2.exp
        ),
        diagnose("pair1.exp", &details.pair1.exp),
        diagnose("pair1.simp", &details.pair1.simp),
        diagnose("pair2.exp.orig", &details.pair2.exp),
        diagnose("pair2.simp.orig", &details.pair2.simp),
        diagnose("pair2.exp.renamed", details.pair2_exp),
        diagnose("pair2.simp.renamed", details.pair2_simp),
        diagnose("combined.exp", details.combined_exp),
        diagnose("combined.simp", details.combined_simp),
    ];

    examples.push(lines.join("\n        "));
}

/// Assert that combining two identity pairs preserves equivalence.
/// Given: Exp1 ≡ Simp1 and Exp2 ≡ Simp2
/// Verify: Exp1 + Exp2' ≡ Simp1 + Simp2' (where Exp2' is alpha-renamed)
///
/// This tests for interaction bugs between different simplification rules.
fn assert_metamorphic_combine(
    test_name: &str,
    pair1: TestPair,
    pair2: TestPair,
    op: &str, // "+", "-", or "*"
) {
    let config = metatest_config();

    // Alpha-rename pair2 to avoid variable collisions
    // x -> u, y -> v
    let pair2_exp = alpha_rename(pair2.exp, pair2.var, "u");
    let pair2_simp = alpha_rename(pair2.simp, pair2.var, "u");

    // Build combined expressions
    let combined_exp = format!("({}) {} ({})", pair1.exp, op, pair2_exp);
    let combined_simp = format!("({}) {} ({})", pair1.simp, op, pair2_simp);

    // Variables: original from pair1, renamed from pair2
    let vars = if pair1.var == pair2.var {
        vec![pair1.var, "u"] // pair2 was renamed to u
    } else {
        vec![pair1.var, pair2.var]
    };

    // Parse expressions
    let mut simplifier = Simplifier::with_default_rules();
    let exp_parsed = match parse(&combined_exp, &mut simplifier.context) {
        Ok(e) => e,
        Err(err) => {
            eprintln!("Parse error in combine test: {} - {:?}", combined_exp, err);
            return;
        }
    };
    let simp_parsed = match parse(&combined_simp, &mut simplifier.context) {
        Ok(e) => e,
        Err(err) => {
            eprintln!("Parse error in combine test: {} - {:?}", combined_simp, err);
            return;
        }
    };

    // Simplify both sides
    let (exp_simplified, _) = simplifier.simplify(exp_parsed);
    let (simp_simplified, _) = simplifier.simplify(simp_parsed);

    // Verify numeric equivalence
    let check_result = if vars.len() == 1 {
        check_numeric_equiv_1var(
            &simplifier.context,
            exp_simplified,
            simp_simplified,
            vars[0],
            &config,
        )
    } else {
        check_numeric_equiv_2var(
            &simplifier.context,
            exp_simplified,
            simp_simplified,
            vars[0],
            vars[1],
            &config,
            &FilterSpec::None,
            &FilterSpec::None,
        )
    };

    if let Err(err) = check_result {
        panic!(
            "Combination Metatest FAILED: {}\n\
             pair1: {} ≡ {}\n\
             pair2: {} ≡ {} (renamed: {} ≡ {})\n\
             combined_exp: {}\n\
             combined_simp: {}\n\
             Error: {}",
            test_name,
            pair1.exp,
            pair1.simp,
            pair2.exp,
            pair2.simp,
            pair2_exp,
            pair2_simp,
            combined_exp,
            combined_simp,
            err
        );
    }
}

/// Assert that combining THREE identity pairs preserves equivalence.
/// Given: Exp1 ≡ Simp1, Exp2 ≡ Simp2, Exp3 ≡ Simp3
/// Verify: Exp1 + Exp2 + Exp3 ≡ Simp1 + Simp2 + Simp3
///
/// Uses alpha-renaming: pair2 uses 'u', pair3 uses 'v'
fn assert_metamorphic_combine_triple(
    test_name: &str,
    pair1: TestPair,
    pair2: TestPair,
    pair3: TestPair,
    op: &str,
) {
    let config = metatest_config();

    // Alpha-rename pairs to avoid collisions
    // pair1: x, pair2: u, pair3: v
    let pair2_exp = alpha_rename(pair2.exp, pair2.var, "u");
    let pair2_simp = alpha_rename(pair2.simp, pair2.var, "u");
    let pair3_exp = alpha_rename(pair3.exp, pair3.var, "v");
    let pair3_simp = alpha_rename(pair3.simp, pair3.var, "v");

    // Build combined expressions: (exp1 op exp2) op exp3
    let combined_exp = format!(
        "(({}) {} ({})) {} ({})",
        pair1.exp, op, pair2_exp, op, pair3_exp
    );
    let combined_simp = format!(
        "(({}) {} ({})) {} ({})",
        pair1.simp, op, pair2_simp, op, pair3_simp
    );

    // Variables: x, u, v (all different now)
    let vars = [pair1.var, "u", "v"];

    // Parse expressions
    let mut simplifier = Simplifier::with_default_rules();
    let exp_parsed = match parse(&combined_exp, &mut simplifier.context) {
        Ok(e) => e,
        Err(err) => {
            eprintln!(
                "Parse error in triple combine test: {} - {:?}",
                combined_exp, err
            );
            return;
        }
    };
    let simp_parsed = match parse(&combined_simp, &mut simplifier.context) {
        Ok(e) => e,
        Err(err) => {
            eprintln!(
                "Parse error in triple combine test: {} - {:?}",
                combined_simp, err
            );
            return;
        }
    };

    // Simplify both sides
    let (exp_simplified, _) = simplifier.simplify(exp_parsed);
    let (simp_simplified, _) = simplifier.simplify(simp_parsed);

    // For 3 variables, we need a different check - use sampling approach
    // Check each variable independently with others fixed at sample values
    let (lo, hi) = config.sample_range;
    let mut valid = 0usize;
    let samples_per_dim = 5; // 5^3 = 125 samples

    for i in 0..samples_per_dim {
        for j in 0..samples_per_dim {
            for k in 0..samples_per_dim {
                let t1 = (i as f64 + 0.5) / samples_per_dim as f64;
                let t2 = (j as f64 + 0.5) / samples_per_dim as f64;
                let t3 = (k as f64 + 0.5) / samples_per_dim as f64;
                let x = lo + (hi - lo) * t1;
                let y = lo + (hi - lo) * t2;
                let z = lo + (hi - lo) * t3;

                let mut var_map = HashMap::new();
                var_map.insert(vars[0].to_string(), x);
                var_map.insert(vars[1].to_string(), y);
                var_map.insert(vars[2].to_string(), z);

                let va = eval_f64(&simplifier.context, exp_simplified, &var_map);
                let vb = eval_f64(&simplifier.context, simp_simplified, &var_map);

                if let (Some(va), Some(vb)) = (va, vb) {
                    if va.is_nan() || vb.is_nan() || va.is_infinite() || vb.is_infinite() {
                        continue;
                    }
                    valid += 1;

                    let diff = (va - vb).abs();
                    let scale = va.abs().max(vb.abs()).max(1.0);
                    let allowed = config.atol + config.rtol * scale;

                    if diff > allowed {
                        panic!(
                            "Triple Combination Metatest FAILED: {}\n\
                             pair1: {} ≡ {}\n\
                             pair2: {} ≡ {}\n\
                             pair3: {} ≡ {}\n\
                             at x={}, u={}, v={}\n\
                             a={:.15}, b={:.15}, diff={:.3e}",
                            test_name,
                            pair1.exp,
                            pair1.simp,
                            pair2.exp,
                            pair2.simp,
                            pair3.exp,
                            pair3.simp,
                            x,
                            y,
                            z,
                            va,
                            vb,
                            diff
                        );
                    }
                }
            }
        }
    }

    if valid < 10 {
        eprintln!(
            "Warning: triple combine {} had only {} valid samples",
            test_name, valid
        );
    }
}

/// Check approximate equality modulo a period (for branch-sensitive comparisons)
///
/// Returns true if the circular distance between a and b (mod period) is within tolerance.
/// Used for arctan identities (mod π) and general trig (mod 2π).
/// Handles NaN/Inf by returning false.
#[allow(dead_code)]
pub(super) fn approx_eq_mod_period(a: f64, b: f64, period: f64, atol: f64, rtol: f64) -> bool {
    // Handle non-finite values
    if !a.is_finite() || !b.is_finite() || !period.is_finite() || period <= 0.0 {
        return false;
    }

    // Calculate circular distance
    let diff = (a - b).rem_euclid(period);
    let circular_dist = diff.min(period - diff);

    let scale = a.abs().max(b.abs()).max(1.0);
    let allowed = atol + rtol * scale;

    circular_dist <= allowed
}

/// Filter: |x| < bound
#[allow(dead_code)]
fn filter_abs_lt(bound: f64) -> impl Fn(f64) -> bool {
    move |x| x.abs() < bound
}

/// Filter: keep samples away from singularities
#[allow(dead_code)]
fn filter_away_from(singularities: Vec<f64>, eps: f64) -> impl Fn(f64) -> bool {
    move |x| singularities.iter().all(|&s| (x - s).abs() > eps)
}

/// Filter: |x| < bound AND away from singularities
#[allow(dead_code)]
fn filter_abs_lt_and_away(bound: f64, singularities: Vec<f64>, eps: f64) -> impl Fn(f64) -> bool {
    move |x| x.abs() < bound && singularities.iter().all(|&s| (x - s).abs() > eps)
}

/// Parse filter spec from CSV string
/// Valid formats:
///   "" / empty / "none" → None
///   "abs_lt(0.9)" → AbsLt { limit: 0.9 }
///   "away_from(1.57;-1.57;eps=0.01)" → AwayFrom { centers: [1.57, -1.57], eps: 0.01 }
///   "abs_lt_and_away(0.9;1.0;-1.0;eps=0.1)" → AbsLtAndAway { limit: 0.9, centers: [1.0, -1.0], eps: 0.1 }
///   "gt(0.0)" → Gt { limit: 0.0 }
///   "ge(0.0)" → Ge { limit: 0.0 }
///   "lt(1.0)" → Lt { limit: 1.0 }
///   "le(1.0)" → Le { limit: 1.0 }
///   "range(0.1;3.0)" → Range { min: 0.1, max: 3.0 }
pub(super) fn parse_filter_spec(spec: &str, line_num: usize) -> FilterSpec {
    let spec = spec.trim();
    if spec.is_empty() || spec.eq_ignore_ascii_case("none") {
        return FilterSpec::None;
    }

    // abs_lt(limit)
    if spec.starts_with("abs_lt(") && spec.ends_with(')') {
        let inner = &spec[7..spec.len() - 1];
        let limit: f64 = inner.parse().unwrap_or_else(|_| {
            panic!("Invalid abs_lt limit at line {}: '{}'", line_num, spec);
        });
        return FilterSpec::AbsLt { limit };
    }

    // away_from(c1;c2;...;eps=<val>)
    if spec.starts_with("away_from(") && spec.ends_with(')') {
        let inner = &spec[10..spec.len() - 1];
        return parse_away_from_inner(inner, line_num, spec);
    }

    // abs_lt_and_away(limit;c1;c2;...;eps=<val>)
    if spec.starts_with("abs_lt_and_away(") && spec.ends_with(')') {
        let inner = &spec[16..spec.len() - 1];
        let parts: Vec<&str> = inner.split(';').collect();
        if parts.is_empty() {
            panic!("Invalid abs_lt_and_away at line {}: '{}'", line_num, spec);
        }
        let limit: f64 = parts[0].parse().unwrap_or_else(|_| {
            panic!(
                "Invalid abs_lt_and_away limit at line {}: '{}'",
                line_num, spec
            );
        });
        let remaining = parts[1..].join(";");
        let away = parse_away_from_inner(&remaining, line_num, spec);
        match away {
            FilterSpec::AwayFrom { centers, eps } => {
                return FilterSpec::AbsLtAndAway {
                    limit,
                    centers,
                    eps,
                };
            }
            _ => panic!("Invalid abs_lt_and_away at line {}: '{}'", line_num, spec),
        }
    }

    // gt(limit) - x > limit
    if spec.starts_with("gt(") && spec.ends_with(')') {
        let inner = &spec[3..spec.len() - 1];
        let limit: f64 = inner.parse().unwrap_or_else(|_| {
            panic!("Invalid gt limit at line {}: '{}'", line_num, spec);
        });
        return FilterSpec::Gt { limit };
    }

    // ge(limit) - x >= limit
    if spec.starts_with("ge(") && spec.ends_with(')') {
        let inner = &spec[3..spec.len() - 1];
        let limit: f64 = inner.parse().unwrap_or_else(|_| {
            panic!("Invalid ge limit at line {}: '{}'", line_num, spec);
        });
        return FilterSpec::Ge { limit };
    }

    // lt(limit) - x < limit
    if spec.starts_with("lt(") && spec.ends_with(')') {
        let inner = &spec[3..spec.len() - 1];
        let limit: f64 = inner.parse().unwrap_or_else(|_| {
            panic!("Invalid lt limit at line {}: '{}'", line_num, spec);
        });
        return FilterSpec::Lt { limit };
    }

    // le(limit) - x <= limit
    if spec.starts_with("le(") && spec.ends_with(')') {
        let inner = &spec[3..spec.len() - 1];
        let limit: f64 = inner.parse().unwrap_or_else(|_| {
            panic!("Invalid le limit at line {}: '{}'", line_num, spec);
        });
        return FilterSpec::Le { limit };
    }

    // range(min;max) - min <= x <= max
    if spec.starts_with("range(") && spec.ends_with(')') {
        let inner = &spec[6..spec.len() - 1];
        let parts: Vec<&str> = inner.split(';').collect();
        if parts.len() != 2 {
            panic!(
                "Invalid range at line {}: '{}'. Expected range(min;max)",
                line_num, spec
            );
        }
        let min: f64 = parts[0].trim().parse().unwrap_or_else(|_| {
            panic!("Invalid range min at line {}: '{}'", line_num, spec);
        });
        let max: f64 = parts[1].trim().parse().unwrap_or_else(|_| {
            panic!("Invalid range max at line {}: '{}'", line_num, spec);
        });
        if min > max {
            panic!(
                "Invalid range at line {}: '{}'. min ({}) > max ({})",
                line_num, spec, min, max
            );
        }
        return FilterSpec::Range { min, max };
    }

    panic!(
        "Unknown filter_spec at line {}: '{}'. \
         Expected: abs_lt(<f64>), away_from(<f64>;...; eps=<f64>), abs_lt_and_away(...), \
         gt(<f64>), ge(<f64>), lt(<f64>), le(<f64>), range(<min>;<max>), or none",
        line_num, spec
    );
}

/// Parse the inner part of away_from: "c1;c2;...;eps=<val>"
fn parse_away_from_inner(inner: &str, line_num: usize, spec: &str) -> FilterSpec {
    let parts: Vec<&str> = inner.split(';').collect();
    let mut centers = Vec::new();
    let mut eps = 0.01; // default

    for part in parts {
        let part = part.trim();
        if let Some(eps_str) = part.strip_prefix("eps=") {
            eps = eps_str.parse().unwrap_or_else(|_| {
                panic!("Invalid eps value at line {}: '{}'", line_num, spec);
            });
        } else if !part.is_empty() {
            let c: f64 = part.parse().unwrap_or_else(|_| {
                panic!(
                    "Invalid center value '{}' at line {}: '{}'",
                    part, line_num, spec
                );
            });
            centers.push(c);
        }
    }

    FilterSpec::AwayFrom { centers, eps }
}

fn normalize_inconclusive_reason_label(reason: &str) -> String {
    let reason = reason.trim();
    if reason.starts_with("Too few valid samples:") {
        "too few valid samples".to_string()
    } else if reason.starts_with("Direct n-var check failed but deterministic slices passed") {
        "n-var slices rescued after direct miss".to_string()
    } else if reason.starts_with("Direct n-var check remained inconclusive") {
        "n-var direct check remained inconclusive".to_string()
    } else if reason == "No free vars for numeric check" {
        "no free vars for numeric check".to_string()
    } else if reason.starts_with("Unsupported contextual numeric arity:") {
        "unsupported contextual numeric arity".to_string()
    } else {
        reason.to_string()
    }
}

pub(super) fn record_inconclusive_reason(
    counts: &mut HashMap<String, usize>,
    kind: &str,
    reason: &str,
) {
    let label = if kind == "domain_frontier" {
        format!("domain-frontier: {}", reason.trim())
    } else {
        normalize_inconclusive_reason_label(reason)
    };
    *counts.entry(label).or_default() += 1;
}

pub(super) fn print_inconclusive_breakdown(counts: &HashMap<String, usize>) {
    if counts.is_empty() {
        return;
    }

    eprintln!("   ◐ Inconclusive by reason:");
    let mut sorted: Vec<_> = counts.iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
    for (cause, count) in sorted {
        eprintln!("      - {}: {}", cause, count);
    }
}

/// Generate stable ID for an identity (hash of canonical representation)
pub(super) fn generate_identity_id(pair: &IdentityPair) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    pair.exp.hash(&mut hasher);
    pair.simp.hash(&mut hasher);
    pair.vars.join(";").hash(&mut hasher);
    format!("{:?}", pair.mode).hash(&mut hasher);
    format!("{:?}", pair.bucket).hash(&mut hasher);
    format!("{:?}", pair.branch_mode).hash(&mut hasher);
    pair.filter_spec.as_str().hash(&mut hasher);

    format!("{:016x}", hasher.finish())
}

/// Escape string for JSON output
pub(super) fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

/// Unescape JSON string
pub(super) fn unescape_json(s: &str) -> String {
    s.replace("\\\"", "\"")
        .replace("\\\\", "\\")
        .replace("\\n", "\n")
        .replace("\\r", "\r")
        .replace("\\t", "\t")
}

/// Baseline file path
pub(super) fn baseline_file_path() -> PathBuf {
    let base = env::var("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."));

    let local = base.join("tests/baselines/metatest_baseline.jsonl");
    if local.exists() {
        return local;
    }

    // Compatibility path when this test is compiled via cas_engine wrapper tests.
    if let Some(parent) = base.parent() {
        let solver_path = parent.join("cas_solver/tests/baselines/metatest_baseline.jsonl");
        if solver_path.exists() {
            return solver_path;
        }
    }

    local
}

/// Generate deterministic hash of test configuration for baseline validation
pub(super) fn generate_config_hash(config: &MetatestConfig) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    config.eval_samples.hash(&mut hasher);
    config.min_valid.hash(&mut hasher);
    // Hash floats as bits for determinism
    config.atol.to_bits().hash(&mut hasher);
    config.rtol.to_bits().hash(&mut hasher);
    config.sample_range.0.to_bits().hash(&mut hasher);
    config.sample_range.1.to_bits().hash(&mut hasher);

    format!("{:016x}", hasher.finish())
}

/// Config header line for baseline file
pub(super) fn config_header_json(config: &MetatestConfig) -> String {
    format!(
        r#"{{"_type":"config","cfg_hash":"{}","samples":{},"min_valid":{},"atol":{},"rtol":{},"range":[{},{}]}}"#,
        generate_config_hash(config),
        config.eval_samples,
        config.min_valid,
        config.atol,
        config.rtol,
        config.sample_range.0,
        config.sample_range.1
    )
}

/// Category ranking for regression detection (higher = worse)
fn category_rank(cat: &str) -> u8 {
    match cat {
        "Ok" => 0,
        "Fragile" => 1,
        "NeedsFilter" => 2,
        "ConfigError" => 3,
        "BugSignal" => 4,
        _ => 5,
    }
}

pub(super) fn check_regression(
    baseline: &IdentitySnapshot,
    current: &IdentitySnapshot,
) -> Option<RegressionResult> {
    let mut reasons = Vec::new();

    // 1. Category worsened
    if category_rank(&current.category) > category_rank(&baseline.category) {
        reasons.push(format!(
            "category {} → {}",
            baseline.category, current.category
        ));
    }

    // 2. asymmetric went from 0 to >0
    if baseline.asymmetric_invalid == 0 && current.asymmetric_invalid > 0 {
        reasons.push(format!("asymmetric 0 → {}", current.asymmetric_invalid));
    }

    // 3. invalid_rate increased by >5%
    let base_rate = baseline.invalid_rate();
    let curr_rate = current.invalid_rate();
    if curr_rate > base_rate + 0.05 {
        reasons.push(format!(
            "invalid_rate {:.1}% → {:.1}%",
            base_rate * 100.0,
            curr_rate * 100.0
        ));
    }

    // 4. filtered_rate increased by >20% (absolute)
    let base_filt = baseline.filtered_rate();
    let curr_filt = current.filtered_rate();
    if curr_filt > base_filt + 0.20 {
        reasons.push(format!(
            "filtered_rate {:.1}% → {:.1}%",
            base_filt * 100.0,
            curr_filt * 100.0
        ));
    }

    // 5. mismatches went from 0 to >0 (for non-BranchSensitive)
    if baseline.mismatches == 0 && current.mismatches > 0 {
        reasons.push(format!("mismatches 0 → {}", current.mismatches));
    }

    if reasons.is_empty() {
        None
    } else {
        Some(RegressionResult {
            id: current.id.clone(),
            exp: truncate_identity(&current.exp, 50),
            reasons,
        })
    }
}

/// Check fragility level based on bucket-specific thresholds
///
/// Thresholds (warning/fail):
/// - Unconditional: 10% / 25% (pure identities should rarely hit poles)
/// - ConditionalRequires: 30% / 50% (some poles expected)
/// - BranchSensitive: 40% / 60% (more tolerance for complex cases)
#[allow(dead_code)]
fn fragility_level_for_bucket(stats: &NumericEquivStats, bucket: Bucket) -> FragilityLevel {
    let rate = stats.invalid_rate();

    let (warn_threshold, fail_threshold) = match bucket {
        Bucket::Unconditional => (0.10, 0.25),
        Bucket::ConditionalRequires => (0.30, 0.50),
        Bucket::BranchSensitive => (0.40, 0.60),
    };

    if rate >= fail_threshold {
        FragilityLevel::Fail
    } else if rate >= warn_threshold {
        FragilityLevel::Warning
    } else {
        FragilityLevel::Ok
    }
}

/// Get minimum valid samples required based on bucket type
#[allow(dead_code)]
fn min_valid_for_bucket(bucket: Bucket, total_samples: usize) -> usize {
    let ratio = match bucket {
        Bucket::Unconditional => 0.70,       // 70% for pure identities
        Bucket::ConditionalRequires => 0.50, // 50% for conditional
        Bucket::BranchSensitive => 0.35,     // 35% for branch-sensitive
    };
    ((total_samples as f64) * ratio).ceil() as usize
}

/// Get legacy bucket from environment variable (for migration flexibility)
/// METATEST_LEGACY_BUCKET=unconditional|conditional_requires (default)
pub(super) fn legacy_bucket_from_env() -> Bucket {
    match env::var("METATEST_LEGACY_BUCKET").ok().as_deref() {
        Some("unconditional") => Bucket::Unconditional,
        _ => Bucket::ConditionalRequires,
    }
}

/// Validate filter spec - fail-fast if malformed
/// Valid formats: "", "abs_lt(0.9)", "away_from(1.0;-1.0;eps=0.1)", "abs_lt_and_away(...)"
#[allow(dead_code)]
fn validate_filter_spec(spec: &str, line_num: usize) {
    if spec.is_empty() {
        return; // Empty is valid (no filter)
    }

    // Basic syntax check: must start with known function name and have balanced parens
    let valid_prefixes = ["abs_lt(", "away_from(", "abs_lt_and_away("];
    let has_valid_prefix = valid_prefixes.iter().any(|p| spec.starts_with(p));
    let has_balanced_parens =
        spec.chars().filter(|&c| c == '(').count() == spec.chars().filter(|&c| c == ')').count();
    let ends_with_paren = spec.ends_with(')');

    if !has_valid_prefix || !has_balanced_parens || !ends_with_paren {
        panic!(
            "Invalid filter_spec at line {}: '{}'. \
             Expected: abs_lt(<f64>), away_from(<f64>;...; eps=<f64>), or abs_lt_and_away(...)",
            line_num, spec
        );
    }
}

/// Parse bucket from string
pub(super) fn parse_bucket(s: &str) -> Bucket {
    match s.to_lowercase().as_str() {
        "unconditional" | "u" => Bucket::Unconditional,
        "branch_sensitive" | "branch" | "b" => Bucket::BranchSensitive,
        _ => Bucket::ConditionalRequires, // Default
    }
}

/// Parse branch mode from string
pub(super) fn parse_branch_mode(s: &str) -> BranchMode {
    match s.to_lowercase().as_str() {
        "modulo_pi" | "mod_pi" => BranchMode::ModuloPi,
        "modulo_2pi" | "mod_2pi" => BranchMode::Modulo2Pi,
        "principal_with_filter" | "filter" => BranchMode::PrincipalWithFilter,
        _ => BranchMode::PrincipalStrict, // Default
    }
}

pub(super) fn top_normalization_gap_hotspots(
    metrics: &[ComboMetrics],
    limit: usize,
) -> Vec<(String, usize, usize, usize)> {
    let mut rows: Vec<_> = metrics
        .iter()
        .filter_map(|m| {
            let burden = m.proved_difference + m.proved_composed;
            (burden > 0).then(|| (m.op.clone(), burden, m.proved_difference, m.proved_composed))
        })
        .collect();
    rows.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    rows.truncate(limit);
    rows
}

#[test]
fn top_normalization_gap_hotspots_prefers_diff_plus_composed_burden() {
    let mk = |op: &str, quotient: usize, diff: usize, composed: usize| ComboMetrics {
        op: op.to_string(),
        pairs: 0,
        families: 0,
        combos: 0,
        nf_convergent: 0,
        proved_quotient: quotient,
        proved_difference: diff,
        proved_composed: composed,
        numeric_only: 0,
        inconclusive: 0,
        failed: 0,
        skipped: 0,
        timeouts: 0,
        cycle_events_total: 0,
        known_symbolic_residuals: 0,
        numeric_only_causes: HashMap::new(),
        inconclusive_causes: HashMap::new(),
        domain_frontier_examples: Vec::new(),
    };

    let top = top_normalization_gap_hotspots(
        &[
            mk("mul", 2253, 502, 76),
            mk("div", 253, 5, 0),
            mk("⇄sub", 441, 0, 0),
            mk("add", 129, 0, 0),
            mk("tie-b", 0, 2, 1),
            mk("tie-a", 0, 1, 2),
        ],
        4,
    );

    assert_eq!(
        top,
        vec![
            ("mul".to_string(), 578, 502, 76),
            ("div".to_string(), 5, 5, 0),
            ("tie-a".to_string(), 3, 1, 2),
            ("tie-b".to_string(), 3, 2, 1),
        ]
    );
}

pub(super) fn effective_combo_cap(total_combos: usize, requested_cap: Option<usize>) -> usize {
    requested_cap
        .filter(|n| *n > 0)
        .map(|limit| total_combos.min(limit))
        .unwrap_or(total_combos)
}

#[test]
fn effective_combo_cap_honors_requested_slice_without_exceeding_total() {
    assert_eq!(effective_combo_cap(11175, None), 11175);
    assert_eq!(effective_combo_cap(11175, Some(500)), 500);
    assert_eq!(effective_combo_cap(11175, Some(20000)), 11175);
}

#[test]
fn effective_combo_window_respects_start_and_cap_inside_total() {
    assert_eq!(effective_combo_window(1000, None, None), (0, 1000));
    assert_eq!(effective_combo_window(1000, Some(200), None), (200, 800));
    assert_eq!(
        effective_combo_window(1000, Some(200), Some(150)),
        (200, 150)
    );
    assert_eq!(
        effective_combo_window(1000, Some(1200), Some(150)),
        (1000, 0)
    );
}

pub(super) fn should_report_combo_progress(
    verbose: bool,
    total_combos: usize,
    processed_combos: usize,
    progress_every: usize,
) -> bool {
    verbose
        && progress_every > 0
        && total_combos >= progress_every
        && processed_combos > 0
        && processed_combos.is_multiple_of(progress_every)
}

#[test]
fn nf_first_div_binomial_square_pair_matches_under_engine_profile() {
    let mut engine = Engine::new();
    let simplifier = &mut engine.simplifier;
    let lhs = parse("(exp(0)) / (u^2 + 2*u + 1)", &mut simplifier.context).expect("lhs parse");
    let rhs = parse("(1) / ((u+1)^2)", &mut simplifier.context).expect("rhs parse");
    let opts = cas_solver::runtime::SimplifyOptions::default();

    let (lhs_simp_raw, _, _) = simplifier.simplify_with_stats(lhs, opts.clone());
    let lhs_simp = fold_constants_safe(&mut simplifier.context, lhs_simp_raw);
    let (rhs_simp_raw, _, _) = simplifier.simplify_with_stats(rhs, opts);
    let rhs_simp = fold_constants_safe(&mut simplifier.context, rhs_simp_raw);

    assert!(
        normal_forms_visibly_equal(&simplifier.context, lhs_simp, rhs_simp),
        "lhs_nf={} rhs_nf={}",
        DisplayExpr {
            context: &simplifier.context,
            id: lhs_simp,
        },
        DisplayExpr {
            context: &simplifier.context,
            id: rhs_simp,
        }
    );
}

pub(super) fn print_combo_progress(op_name: &str, snapshot: &ComboProgressSnapshot) {
    let pct = if snapshot.total_combos == 0 {
        0.0
    } else {
        snapshot.processed_combos as f64 / snapshot.total_combos as f64 * 100.0
    };
    eprintln!(
        "⏳ Progress [{}]: {}/{} ({:.1}%) | NF {} | Proved {} | Numeric {} | Inconcl {} | Skip {} | T/O {} | Failed {}",
        op_name,
        snapshot.processed_combos,
        snapshot.total_combos,
        pct,
        snapshot.nf_convergent,
        snapshot.proved_symbolic,
        snapshot.numeric_only,
        snapshot.inconclusive,
        snapshot.skipped,
        snapshot.timeouts,
        snapshot.failed
    );
}

/// Stratified sampling: guarantees ≥1 identity per CSV family.
///
/// Phase 1: Pick 1 representative per family using Lcg RNG.
/// Phase 2: Fill remaining `max_pairs - num_families` slots from un-selected pairs.
/// The final selection is shuffled for combo ordering randomization.
pub(super) fn stratified_select(
    all_pairs: Vec<IdentityPair>,
    max_pairs: usize,
    seed: u64,
) -> Vec<IdentityPair> {
    use std::collections::BTreeMap;

    let mut rng = Lcg::new(seed);

    // Group indices by family (BTreeMap for deterministic order)
    let mut family_groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (i, pair) in all_pairs.iter().enumerate() {
        family_groups
            .entry(pair.family.clone())
            .or_default()
            .push(i);
    }

    let num_families = family_groups.len();
    let mut selected_indices: Vec<usize> = Vec::with_capacity(max_pairs);
    let mut used = vec![false; all_pairs.len()];

    // Phase 1: Pick 1 representative per family
    for indices in family_groups.values() {
        let pick = rng.pick(indices.len() as u32) as usize;
        let idx = indices[pick];
        selected_indices.push(idx);
        used[idx] = true;
    }

    // Phase 2: Fill remaining slots from un-selected pairs
    if max_pairs > num_families {
        let remaining = max_pairs - num_families;
        // Collect un-selected indices and shuffle them
        let mut pool: Vec<usize> = (0..all_pairs.len()).filter(|i| !used[*i]).collect();
        // Fisher-Yates shuffle on pool
        for i in (1..pool.len()).rev() {
            let j = rng.pick((i + 1) as u32) as usize;
            pool.swap(i, j);
        }
        for &idx in pool.iter().take(remaining) {
            selected_indices.push(idx);
        }
    }

    // Truncate if max_pairs < num_families (best-effort: not all families covered)
    selected_indices.truncate(max_pairs);

    // Final shuffle for combo ordering randomization
    for i in (1..selected_indices.len()).rev() {
        let j = rng.pick((i + 1) as u32) as usize;
        selected_indices.swap(i, j);
    }

    // Build result
    selected_indices
        .into_iter()
        .map(|i| all_pairs[i].clone())
        .collect()
}

/// Test shuffle with dual check: semantic (numeric) + structural (exact)
pub(super) fn test_shuffle_dual(expr_str: &str, var: &str) -> ShuffleResult {
    let mut simplifier = Simplifier::new();

    // Parse - skip if syntax not supported
    let expr = match parse(expr_str, &mut simplifier.context) {
        Ok(e) => e,
        Err(_) => return ShuffleResult::ParseSkip,
    };

    // Simplify original
    let (simplified_original, _) = simplifier.simplify(expr);

    // Shuffle and simplify
    let shuffled = shuffle_expr(&mut simplifier.context, expr);
    let (simplified_shuffled, _) = simplifier.simplify(shuffled);

    // 1. Structural check (Debug representation)
    let original_debug = format!("{:?}", simplifier.context.get(simplified_original));
    let shuffled_debug = format!("{:?}", simplifier.context.get(simplified_shuffled));
    let structural_match = original_debug == shuffled_debug;

    // 2. Semantic check (numeric evaluation at a few points)
    let semantic_match = check_numeric_equiv_quick(
        &simplifier.context,
        simplified_original,
        simplified_shuffled,
        var,
    );

    match (structural_match, semantic_match) {
        (true, true) => ShuffleResult::Ok,
        (false, true) => ShuffleResult::StructuralDiff("different debug repr".to_string()),
        (_, false) => ShuffleResult::SemanticFail("numeric mismatch after shuffle".to_string()),
    }
}

/// Test that simplify(E) == simplify(shuffle(E)) for a single expression
fn test_shuffle_invariance(expr_str: &str, _label: &str) -> Result<(), String> {
    // Create simplifier (which owns Context)
    let mut simplifier = Simplifier::new();

    // Parse the expression
    let expr = match parse(expr_str, &mut simplifier.context) {
        Ok(e) => e,
        Err(_) => return Err("parse failed".to_string()),
    };

    // Simplify original
    let (simplified_original, _) = simplifier.simplify(expr);
    let original_str = format!("{:?}", simplifier.context.get(simplified_original));

    // Shuffle the expression
    let shuffled = shuffle_expr(&mut simplifier.context, expr);

    // Simplify shuffled
    let (simplified_shuffled, _) = simplifier.simplify(shuffled);
    let shuffled_str = format!("{:?}", simplifier.context.get(simplified_shuffled));

    // Compare (structural equality via Debug representation)
    if original_str != shuffled_str {
        return Err(format!(
            "shuffle mismatch: '{}' vs '{}'",
            original_str, shuffled_str
        ));
    }

    Ok(())
}

/// Test that A(T(x)) ≡ B(T(x)) for a specific transform
pub(super) fn test_transform_identity(
    pair: &IdentityPair,
    var: &str,
    transform: &MetaTransform,
    min_valid_factor: f64,
) -> TransformResult {
    let mut simplifier = Simplifier::new();

    // Parse expressions
    let exp = match parse(&pair.exp, &mut simplifier.context) {
        Ok(e) => e,
        Err(_) => return TransformResult::Skip("parse exp failed".to_string()),
    };
    let simp = match parse(&pair.simp, &mut simplifier.context) {
        Ok(e) => e,
        Err(_) => return TransformResult::Skip("parse simp failed".to_string()),
    };

    // Simplify both
    let (exp_simplified, _) = simplifier.simplify(exp);
    let (simp_simplified, _) = simplifier.simplify(simp);

    // Sample and evaluate with transform + composed filter
    let samples: Vec<f64> = (-50..=50).map(|i| (i as f64) * 0.2).collect();

    let min_valid = ((samples.len() as f64) * 0.9 * min_valid_factor) as usize;

    let mut valid = 0;
    let mut matching = 0;
    let mut _filtered_out = 0;

    for &x in &samples {
        // Apply transform: x' = T(x)
        let x_prime = transform.apply_f64(x);

        // Composed filter: check if x' passes the original filter
        if !pair.filter_spec.accept(x_prime) {
            _filtered_out += 1;
            continue;
        }

        // Evaluate at x'
        let mut vars = HashMap::new();
        vars.insert(var.to_string(), x_prime);

        let va = eval_f64(&simplifier.context, exp_simplified, &vars);
        let vb = eval_f64(&simplifier.context, simp_simplified, &vars);

        match (va, vb) {
            (Some(a), Some(b)) if a.is_finite() && b.is_finite() => {
                valid += 1;
                let diff = (a - b).abs();
                let rel = diff / a.abs().max(1e-10);
                if diff < 1e-6 || rel < 1e-6 {
                    matching += 1;
                }
            }
            _ => {} // Skip invalid evaluations
        }
    }

    // Check results
    if valid < min_valid {
        // Inconclusive - not enough valid samples
        return TransformResult::Skip(format!("only {}/{} valid", valid, min_valid));
    }

    if matching != valid {
        return TransformResult::Fail(format!("mismatch {}/{} valid", matching, valid));
    }

    TransformResult::Pass
}

/// Load contextual direct pairs from CSV
pub(super) fn parse_filter_specs(spec: &str, vars_len: usize, line_num: usize) -> Vec<FilterSpec> {
    let spec = spec.trim();
    if spec.is_empty() {
        return vec![FilterSpec::None; vars_len];
    }

    let mut filters: Vec<FilterSpec> = spec
        .split('|')
        .map(|part| parse_filter_spec(part.trim(), line_num))
        .collect();

    if filters.len() > vars_len {
        panic!(
            "Too many filter specs at line {}: expected at most {}, got {}",
            line_num,
            vars_len,
            filters.len()
        );
    }

    filters.resize(vars_len, FilterSpec::None);
    filters
}

pub(super) fn parse_direct_pairs(file_name: &str) -> Vec<ContextualPair> {
    let csv_path = find_test_data_file(file_name);
    let content = std::fs::read_to_string(csv_path)
        .unwrap_or_else(|_| panic!("Failed to read {}", file_name));

    let mut pairs = Vec::new();
    let mut current_family = String::from("Uncategorized");
    for (line_idx, line) in content.lines().enumerate() {
        let line_num = line_idx + 1;
        let line = line.trim();
        if line.starts_with('#') {
            let label = line.trim_start_matches('#').trim();
            if !label.is_empty() && !label.starts_with("Format") && !label.starts_with("Each row") {
                current_family = label.to_string();
            }
            continue;
        }
        if line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.splitn(4, ',').collect();
        if parts.len() >= 3 {
            let vars: Vec<String> = parts[2]
                .trim()
                .split(';')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            let filters = if parts.len() >= 4 {
                parse_filter_specs(parts[3], vars.len(), line_num)
            } else {
                vec![FilterSpec::None; vars.len()]
            };
            pairs.push(ContextualPair {
                lhs: parts[0].trim().to_string(),
                rhs: parts[1].trim().to_string(),
                vars,
                filters,
                family: current_family.clone(),
            });
        }
    }
    pairs
}

pub(super) fn load_residual_pairs() -> Vec<ContextualPair> {
    static RESIDUAL_PAIRS: OnceLock<Vec<ContextualPair>> = OnceLock::new();
    RESIDUAL_PAIRS
        .get_or_init(|| parse_direct_pairs("residual_pairs.csv"))
        .clone()
}

pub(super) fn load_idempotence_expressions() -> Vec<IdempotenceExpr> {
    let csv_path = find_test_data_file("idempotence_expressions.csv");
    let content =
        std::fs::read_to_string(csv_path).expect("Failed to read idempotence_expressions.csv");

    let mut exprs = Vec::new();
    let mut current_family = String::from("Uncategorized");
    for (line_idx, line) in content.lines().enumerate() {
        let line_num = line_idx + 1;
        let line = line.trim();
        if line.starts_with('#') {
            let label = line.trim_start_matches('#').trim();
            if !label.is_empty() && !label.starts_with("Format") && !label.starts_with("Goal:") {
                current_family = label.to_string();
            }
            continue;
        }
        if line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.splitn(3, ',').collect();
        if parts.len() < 2 {
            panic!(
                "idempotence_expressions.csv line {}: expected at least expr,vars. Line: '{}'",
                line_num, line
            );
        }

        let vars: Vec<String> = parts[1]
            .trim()
            .split(';')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        let filters = if parts.len() >= 3 {
            parse_filter_specs(parts[2], vars.len(), line_num)
        } else {
            vec![FilterSpec::None; vars.len()]
        };

        exprs.push(IdempotenceExpr {
            expr: parts[0].trim().to_string(),
            vars,
            filters,
            family: current_family.clone(),
        });
    }

    exprs
}

pub(super) fn parse_complex_mode_label(
    label: &str,
    csv_name: &str,
    line_num: usize,
) -> cas_solver::runtime::ComplexMode {
    match label.trim().to_lowercase().as_str() {
        "auto" => cas_solver::runtime::ComplexMode::Auto,
        "off" => cas_solver::runtime::ComplexMode::Off,
        "on" => cas_solver::runtime::ComplexMode::On,
        other => panic!(
            "{} line {}: invalid complex mode '{}'",
            csv_name, line_num, other
        ),
    }
}

pub(super) fn complex_mode_label(mode: cas_solver::runtime::ComplexMode) -> &'static str {
    match mode {
        cas_solver::runtime::ComplexMode::Auto => "auto",
        cas_solver::runtime::ComplexMode::Off => "off",
        cas_solver::runtime::ComplexMode::On => "on",
    }
}

pub(super) fn parse_const_fold_mode_label(
    label: &str,
    csv_name: &str,
    line_num: usize,
) -> cas_solver::api::ConstFoldMode {
    match label.trim().to_lowercase().as_str() {
        "off" => cas_solver::api::ConstFoldMode::Off,
        "safe" => cas_solver::api::ConstFoldMode::Safe,
        other => panic!(
            "{} line {}: invalid const-fold mode '{}'",
            csv_name, line_num, other
        ),
    }
}

pub(super) fn const_fold_mode_label(mode: cas_solver::api::ConstFoldMode) -> &'static str {
    match mode {
        cas_solver::api::ConstFoldMode::Off => "off",
        cas_solver::api::ConstFoldMode::Safe => "safe",
    }
}

pub(super) fn parse_inv_trig_policy_label(
    label: &str,
    csv_name: &str,
    line_num: usize,
) -> cas_solver::runtime::InverseTrigPolicy {
    match label.trim().to_lowercase().as_str() {
        "strict" => cas_solver::runtime::InverseTrigPolicy::Strict,
        "principal" | "principalvalue" => cas_solver::runtime::InverseTrigPolicy::PrincipalValue,
        other => panic!(
            "{} line {}: invalid inverse trig policy '{}'",
            csv_name, line_num, other
        ),
    }
}

pub(super) fn inv_trig_policy_label(value: cas_solver::runtime::InverseTrigPolicy) -> &'static str {
    match value {
        cas_solver::runtime::InverseTrigPolicy::Strict => "strict",
        cas_solver::runtime::InverseTrigPolicy::PrincipalValue => "principal",
    }
}

pub(super) fn load_warnings_contract_expressions() -> Vec<WarningsContractExpr> {
    let csv_path = find_test_data_file("warnings_contract_expressions.csv");
    let content = std::fs::read_to_string(csv_path)
        .expect("Failed to read warnings_contract_expressions.csv");

    let mut exprs = Vec::new();
    let mut current_family = String::from("Uncategorized");
    for (line_idx, line) in content.lines().enumerate() {
        let line_num = line_idx + 1;
        let line = line.trim();
        if line.starts_with('#') {
            let label = line.trim_start_matches('#').trim();
            if !label.is_empty()
                && !label.starts_with("Format")
                && !label.starts_with("Expressions")
            {
                current_family = label.to_string();
            }
            continue;
        }
        if line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.rsplitn(3, ',').collect();
        if parts.len() != 3 {
            panic!(
                "warnings_contract_expressions.csv line {}: expected expr,mode,expect_warning. Line: '{}'",
                line_num, line
            );
        }

        let expr = parts[2].trim().to_string();
        let mode = parse_domain_mode_label(parts[1], "warnings_contract_expressions.csv", line_num);
        let expect_warning = match parts[0].trim().to_lowercase().as_str() {
            "yes" | "true" => true,
            "no" | "false" => false,
            other => panic!(
                "warnings_contract_expressions.csv line {}: invalid expect_warning '{}'",
                line_num, other
            ),
        };

        exprs.push(WarningsContractExpr {
            expr,
            mode,
            expect_warning,
            family: current_family.clone(),
        });
    }

    exprs
}

pub(super) fn load_transparency_signal_contract_expressions() -> Vec<TransparencySignalContractExpr>
{
    let csv_path = find_test_data_file("transparency_signal_contract_expressions.csv");
    let content = std::fs::read_to_string(csv_path)
        .expect("Failed to read transparency_signal_contract_expressions.csv");

    let mut exprs = Vec::new();
    let mut current_family = String::from("Uncategorized");
    for (line_idx, line) in content.lines().enumerate() {
        let line_num = line_idx + 1;
        let line = line.trim();
        if line.starts_with('#') {
            let label = line.trim_start_matches('#').trim();
            if !label.is_empty()
                && !label.starts_with("Format")
                && !label.starts_with("Expressions")
            {
                current_family = label.to_string();
            }
            continue;
        }
        if line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.rsplitn(3, ',').collect();
        if parts.len() != 3 {
            panic!(
                "transparency_signal_contract_expressions.csv line {}: expected expr,mode,expect_signal. Line: '{}'",
                line_num, line
            );
        }

        let expr = parts[2].trim().to_string();
        let mode = parse_domain_mode_label(
            parts[1],
            "transparency_signal_contract_expressions.csv",
            line_num,
        );
        let expect_signal = match parts[0].trim().to_lowercase().as_str() {
            "yes" | "true" => true,
            "no" | "false" => false,
            other => panic!(
                "transparency_signal_contract_expressions.csv line {}: invalid expect_signal '{}'",
                line_num, other
            ),
        };

        exprs.push(TransparencySignalContractExpr {
            expr,
            mode,
            expect_signal,
            family: current_family.clone(),
        });
    }

    exprs
}

pub(super) fn load_branch_transparency_contract_expressions() -> Vec<BranchTransparencyContractExpr>
{
    let csv_path = find_test_data_file("branch_transparency_contract_expressions.csv");
    let content = std::fs::read_to_string(csv_path)
        .expect("Failed to read branch_transparency_contract_expressions.csv");

    let mut exprs = Vec::new();
    let mut current_family = String::from("Uncategorized");
    for (line_idx, line) in content.lines().enumerate() {
        let line_num = line_idx + 1;
        let line = line.trim();
        if line.starts_with('#') {
            let label = line.trim_start_matches('#').trim();
            if !label.is_empty()
                && !label.starts_with("Format")
                && !label.starts_with("Expressions")
            {
                current_family = label.to_string();
            }
            continue;
        }
        if line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.rsplitn(4, ',').collect();
        if parts.len() != 4 {
            panic!(
                "branch_transparency_contract_expressions.csv line {}: expected expr,mode,inv_trig,expect_signal. Line: '{}'",
                line_num, line
            );
        }

        let expr = parts[3].trim().to_string();
        let mode = parse_domain_mode_label(
            parts[2],
            "branch_transparency_contract_expressions.csv",
            line_num,
        );
        let inv_trig = parse_inv_trig_policy_label(
            parts[1],
            "branch_transparency_contract_expressions.csv",
            line_num,
        );
        let expect_signal = match parts[0].trim().to_lowercase().as_str() {
            "yes" | "true" => true,
            "no" | "false" => false,
            other => panic!(
                "branch_transparency_contract_expressions.csv line {}: invalid expect_signal '{}'",
                line_num, other
            ),
        };

        exprs.push(BranchTransparencyContractExpr {
            expr,
            mode,
            inv_trig,
            expect_signal,
            family: current_family.clone(),
        });
    }

    exprs
}

pub(super) fn load_semantic_behavior_contract_expressions() -> Vec<SemanticBehaviorContractExpr> {
    let csv_path = find_test_data_file("semantic_behavior_contract_expressions.csv");
    let content = std::fs::read_to_string(csv_path)
        .expect("Failed to read semantic_behavior_contract_expressions.csv");

    let mut exprs = Vec::new();
    let mut current_family = String::from("Uncategorized");
    for (line_idx, line) in content.lines().enumerate() {
        let line_num = line_idx + 1;
        let line = line.trim();
        if line.starts_with('#') {
            let label = line.trim_start_matches('#').trim();
            if !label.is_empty()
                && !label.starts_with("Format")
                && !label.starts_with("Expressions")
            {
                current_family = label.to_string();
            }
            continue;
        }
        if line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.rsplitn(5, ',').collect();
        if parts.len() != 5 {
            panic!(
                "semantic_behavior_contract_expressions.csv line {}: expected expr,value_domain,mode,match_kind,expected. Line: '{}'",
                line_num, line
            );
        }

        let expr = parts[4].trim().to_string();
        let value_domain = parse_value_domain_label(
            parts[3],
            "semantic_behavior_contract_expressions.csv",
            line_num,
        );
        let mode = parse_domain_mode_label(
            parts[2],
            "semantic_behavior_contract_expressions.csv",
            line_num,
        );
        let expectation = match parts[1].trim().to_lowercase().as_str() {
            "exact" => SemanticBehaviorExpectation::Exact(parts[0].trim().to_string()),
            "contains_all" => SemanticBehaviorExpectation::ContainsAll(
                parts[0]
                    .split(';')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect(),
            ),
            other => panic!(
                "semantic_behavior_contract_expressions.csv line {}: invalid match_kind '{}'",
                line_num, other
            ),
        };

        exprs.push(SemanticBehaviorContractExpr {
            expr,
            value_domain,
            mode,
            expectation,
            family: current_family.clone(),
        });
    }

    exprs
}

pub(super) fn load_complex_mode_behavior_contract_expressions(
) -> Vec<ComplexModeBehaviorContractExpr> {
    let csv_path = find_test_data_file("complex_mode_behavior_contract_expressions.csv");
    let content = std::fs::read_to_string(csv_path)
        .expect("Failed to read complex_mode_behavior_contract_expressions.csv");

    let mut exprs = Vec::new();
    let mut current_family = String::from("Uncategorized");
    for (line_idx, line) in content.lines().enumerate() {
        let line_num = line_idx + 1;
        let line = line.trim();
        if line.starts_with('#') {
            let label = line.trim_start_matches('#').trim();
            if !label.is_empty()
                && !label.starts_with("Format")
                && !label.starts_with("Expressions")
            {
                current_family = label.to_string();
            }
            continue;
        }
        if line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.rsplitn(5, ',').collect();
        if parts.len() != 5 {
            panic!(
                "complex_mode_behavior_contract_expressions.csv line {}: expected expr,value_domain,complex_mode,match_kind,expected. Line: '{}'",
                line_num, line
            );
        }

        let expr = parts[4].trim().to_string();
        let value_domain = parse_value_domain_label(
            parts[3],
            "complex_mode_behavior_contract_expressions.csv",
            line_num,
        );
        let complex_mode = parse_complex_mode_label(
            parts[2],
            "complex_mode_behavior_contract_expressions.csv",
            line_num,
        );
        let expectation = match parts[1].trim().to_lowercase().as_str() {
            "exact" => SemanticBehaviorExpectation::Exact(parts[0].trim().to_string()),
            "contains_all" => SemanticBehaviorExpectation::ContainsAll(
                parts[0]
                    .split(';')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect(),
            ),
            other => panic!(
                "complex_mode_behavior_contract_expressions.csv line {}: invalid match_kind '{}'",
                line_num, other
            ),
        };

        exprs.push(ComplexModeBehaviorContractExpr {
            expr,
            value_domain,
            complex_mode,
            expectation,
            family: current_family.clone(),
        });
    }

    exprs
}

pub(super) fn load_const_fold_behavior_contract_expressions() -> Vec<ConstFoldBehaviorContractExpr>
{
    let csv_path = find_test_data_file("const_fold_behavior_contract_expressions.csv");
    let content = std::fs::read_to_string(csv_path)
        .expect("Failed to read const_fold_behavior_contract_expressions.csv");

    let mut exprs = Vec::new();
    let mut current_family = String::from("Uncategorized");
    for (line_idx, line) in content.lines().enumerate() {
        let line_num = line_idx + 1;
        let line = line.trim();
        if line.starts_with('#') {
            let label = line.trim_start_matches('#').trim();
            if !label.is_empty()
                && !label.starts_with("Format")
                && !label.starts_with("Expressions")
            {
                current_family = label.to_string();
            }
            continue;
        }
        if line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.rsplitn(5, ',').collect();
        if parts.len() != 5 {
            panic!(
                "const_fold_behavior_contract_expressions.csv line {}: expected expr,value_domain,const_fold_mode,match_kind,expected. Line: '{}'",
                line_num, line
            );
        }

        let expr = parts[4].trim().to_string();
        let value_domain = parse_value_domain_label(
            parts[3],
            "const_fold_behavior_contract_expressions.csv",
            line_num,
        );
        let const_fold_mode = parse_const_fold_mode_label(
            parts[2],
            "const_fold_behavior_contract_expressions.csv",
            line_num,
        );
        let expectation = match parts[1].trim().to_lowercase().as_str() {
            "exact" => SemanticBehaviorExpectation::Exact(parts[0].trim().to_string()),
            "contains_all" => SemanticBehaviorExpectation::ContainsAll(
                parts[0]
                    .split(';')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect(),
            ),
            other => panic!(
                "const_fold_behavior_contract_expressions.csv line {}: invalid match_kind '{}'",
                line_num, other
            ),
        };

        exprs.push(ConstFoldBehaviorContractExpr {
            expr,
            value_domain,
            const_fold_mode,
            expectation,
            family: current_family.clone(),
        });
    }

    exprs
}

pub(super) fn load_semantic_axes_contract_expressions() -> Vec<SemanticAxesContractExpr> {
    let csv_path = find_test_data_file("semantic_axes_contract_expressions.csv");
    let content = std::fs::read_to_string(csv_path)
        .expect("Failed to read semantic_axes_contract_expressions.csv");

    let mut exprs = Vec::new();
    let mut current_family = String::from("Uncategorized");
    for (line_idx, line) in content.lines().enumerate() {
        let line_num = line_idx + 1;
        let line = line.trim();
        if line.starts_with('#') {
            let label = line.trim_start_matches('#').trim();
            if !label.is_empty()
                && !label.starts_with("Format")
                && !label.starts_with("Expressions")
            {
                current_family = label.to_string();
            }
            continue;
        }
        if line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.rsplitn(5, ',').collect();
        if parts.len() != 5 {
            panic!(
                "semantic_axes_contract_expressions.csv line {}: expected expr,value_domain,mode,expect_requires,expect_warning. Line: '{}'",
                line_num, line
            );
        }

        let expr = parts[4].trim().to_string();
        let value_domain =
            parse_value_domain_label(parts[3], "semantic_axes_contract_expressions.csv", line_num);
        let mode =
            parse_domain_mode_label(parts[2], "semantic_axes_contract_expressions.csv", line_num);
        let expect_requires = match parts[1].trim().to_lowercase().as_str() {
            "yes" | "true" => true,
            "no" | "false" => false,
            other => panic!(
                "semantic_axes_contract_expressions.csv line {}: invalid expect_requires '{}'",
                line_num, other
            ),
        };
        let expect_warning = match parts[0].trim().to_lowercase().as_str() {
            "yes" | "true" => true,
            "no" | "false" => false,
            other => panic!(
                "semantic_axes_contract_expressions.csv line {}: invalid expect_warning '{}'",
                line_num, other
            ),
        };

        exprs.push(SemanticAxesContractExpr {
            expr,
            value_domain,
            mode,
            expect_requires,
            expect_warning,
            family: current_family.clone(),
        });
    }

    exprs
}

pub(super) fn simplify_with_metadata_on_axes(
    input: &str,
    mode: cas_solver::runtime::DomainMode,
    value_domain: cas_solver::runtime::ValueDomain,
) -> Result<SimplifyMetadata, String> {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().shared.semantics.domain_mode = mode;
    state.options_mut().shared.semantics.value_domain = value_domain;

    let parsed = parse(input, &mut engine.simplifier.context)
        .map_err(|e| format!("parse failed for '{}': {:?}", input, e))?;
    let req = EvalRequest {
        raw_input: input.to_string(),
        parsed,
        action: EvalAction::Simplify,
        auto_store: false,
    };

    let output = engine
        .eval(&mut state, req)
        .map_err(|e| format!("eval failed for '{}': {:?}", input, e))?;

    let result = match &output.result {
        EvalResult::Expr(e) => DisplayExpr {
            context: &engine.simplifier.context,
            id: *e,
        }
        .to_string(),
        other => {
            return Err(format!(
                "unexpected eval result for '{}': {:?}",
                input, other
            ));
        }
    };

    let mut required: Vec<String> = output
        .required_conditions
        .iter()
        .map(|cond| cond.display(&engine.simplifier.context))
        .collect();
    required.sort();
    required.dedup();

    let mut warnings: Vec<String> = output
        .domain_warnings
        .iter()
        .map(|w| w.message.clone())
        .collect();
    warnings.sort();
    warnings.dedup();

    Ok(SimplifyMetadata {
        result,
        required,
        warnings,
    })
}

pub(super) fn simplify_with_complex_mode_behavior(
    input: &str,
    value_domain: cas_solver::runtime::ValueDomain,
    complex_mode: cas_solver::runtime::ComplexMode,
) -> Result<String, String> {
    let mut ctx = Context::new();
    let expr =
        parse(input, &mut ctx).map_err(|e| format!("parse failed for '{}': {:?}", input, e))?;

    let opts = cas_solver::runtime::EvalOptions {
        complex_mode,
        shared: cas_solver::runtime::SharedSemanticConfig {
            context_mode: cas_solver::runtime::ContextMode::Standard,
            semantics: cas_solver::runtime::EvalConfig {
                value_domain,
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    };

    let mut simplifier = cas_solver::runtime::Simplifier::with_profile(&opts);
    simplifier.context = ctx;
    let simplify_opts = opts.to_simplify_options();
    let (result, _steps) = simplifier.simplify_with_options(expr, simplify_opts);

    Ok(format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    ))
}

pub(super) fn fold_with_const_fold_behavior(
    input: &str,
    value_domain: cas_solver::runtime::ValueDomain,
    const_fold_mode: cas_solver::api::ConstFoldMode,
) -> Result<String, String> {
    let mut ctx = Context::new();
    let expr =
        parse(input, &mut ctx).map_err(|e| format!("parse failed for '{}': {:?}", input, e))?;

    let cfg = cas_solver::runtime::EvalConfig {
        value_domain,
        ..Default::default()
    };
    let mut budget = cas_solver::runtime::Budget::preset_unlimited();
    let result =
        cas_solver::api::fold_constants(&mut ctx, expr, &cfg, const_fold_mode, &mut budget)
            .map_err(|e| format!("const_fold failed for '{}': {:?}", input, e))?;

    Ok(format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: &ctx,
            id: result.expr
        }
    ))
}

pub(super) fn simplify_with_transparency_metadata(
    input: &str,
    mode: cas_solver::runtime::DomainMode,
) -> Result<SimplifyTransparencyMetadata, String> {
    simplify_with_transparency_metadata_with_inv_trig(
        input,
        mode,
        cas_solver::runtime::InverseTrigPolicy::Strict,
    )
}

pub(super) fn simplify_with_transparency_metadata_with_inv_trig(
    input: &str,
    mode: cas_solver::runtime::DomainMode,
    inv_trig: cas_solver::runtime::InverseTrigPolicy,
) -> Result<SimplifyTransparencyMetadata, String> {
    let mut engine = Engine::new();
    let mut state = SessionState::new();
    state.options_mut().shared.semantics.domain_mode = mode;
    state.options_mut().shared.semantics.inv_trig = inv_trig;

    let parsed = parse(input, &mut engine.simplifier.context)
        .map_err(|e| format!("parse failed for '{}': {:?}", input, e))?;
    let req = EvalRequest {
        raw_input: input.to_string(),
        parsed,
        action: EvalAction::Simplify,
        auto_store: false,
    };

    let output = engine
        .eval(&mut state, req)
        .map_err(|e| format!("eval failed for '{}': {:?}", input, e))?;

    let result = match &output.result {
        EvalResult::Expr(e) => DisplayExpr {
            context: &engine.simplifier.context,
            id: *e,
        }
        .to_string(),
        other => {
            return Err(format!(
                "unexpected eval result for '{}': {:?}",
                input, other
            ));
        }
    };

    let mut warnings: Vec<String> = output
        .domain_warnings
        .iter()
        .map(|w| w.message.clone())
        .collect();
    warnings.sort();
    warnings.dedup();

    let mut assumption_signals: Vec<String> = output
        .steps
        .iter()
        .flat_map(|step| step.assumption_events().iter())
        .filter(|event| {
            event.kind.should_display()
                || matches!(
                    event.kind,
                    cas_solver::api::AssumptionKind::DerivedFromRequires
                )
        })
        .map(|event| {
            format!(
                "{}|{}|{}",
                event.kind.label(),
                event.key.kind(),
                event.message
            )
        })
        .collect();
    assumption_signals.sort();
    assumption_signals.dedup();

    Ok(SimplifyTransparencyMetadata {
        result,
        warnings,
        assumption_signals,
    })
}

pub(super) fn simplify_generic_with_metadata(input: &str) -> Result<SimplifyMetadata, String> {
    simplify_with_metadata_on_axes(
        input,
        cas_solver::runtime::DomainMode::Generic,
        cas_solver::runtime::ValueDomain::RealOnly,
    )
}

pub(super) fn semantic_behavior_matches(
    expectation: &SemanticBehaviorExpectation,
    actual: &str,
) -> bool {
    match expectation {
        SemanticBehaviorExpectation::Exact(expected) => actual == expected,
        SemanticBehaviorExpectation::ContainsAll(needles) => {
            needles.iter().all(|needle| actual.contains(needle))
        }
    }
}

pub(super) fn semantic_behavior_label(expectation: &SemanticBehaviorExpectation) -> String {
    match expectation {
        SemanticBehaviorExpectation::Exact(expected) => format!("exact '{}'", expected),
        SemanticBehaviorExpectation::ContainsAll(parts) => {
            format!("contains_all {:?}", parts)
        }
    }
}

#[test]
fn safe_window_parametrized_proof_closes_log_square_and_sqrt_product_pairs() {
    assert!(prove_zero_from_safe_window_parametrized_texts(
        "ln((-u)^2)",
        "2*ln((-u))"
    ));
    assert!(prove_zero_from_safe_window_parametrized_texts(
        "ln((2*u)^2)",
        "2*ln((2*u))"
    ));
    assert!(prove_zero_from_safe_window_parametrized_texts(
        "ln((1-u)^2)",
        "2*ln((1-u))"
    ));
    assert!(prove_zero_from_safe_window_parametrized_texts(
        "(cos(3*pi/8))*(sqrt(u)*sqrt(4*u))",
        "(sqrt(2-sqrt(2))/2)*(2*u)"
    ));
    assert!(prove_zero_from_safe_window_parametrized_texts(
        "(sin(2*arcsin(x)))*(sqrt(u)*sqrt(4*u))",
        "(2*x*sqrt(1-x^2))*(2*u)"
    ));
    assert!(prove_zero_from_safe_window_parametrized_texts(
        "((exp(x)-exp(-x))/2)*(sin(2*arcsin(u)))",
        "(sinh(x))*(2*u*sqrt(1-u^2))"
    ));
    assert!(prove_zero_from_safe_window_parametrized_texts(
        "(tanh(x))*(sin(2*arcsin(u)))",
        "((exp(x)-exp(-x))/(exp(x)+exp(-x)))*(2*u*sqrt(1-u^2))"
    ));
    assert!(prove_zero_from_safe_window_parametrized_texts(
        "(sin(2*arcsin(x)))*(abs(sin(u/2)))",
        "(2*x*sqrt(1-x^2))*(sqrt((1-cos(u))/2))"
    ));
}

#[test]
fn normalize_inconclusive_reason_label_collapses_known_prefixes() {
    assert_eq!(
        normalize_inconclusive_reason_label("Too few valid samples: 0 / 20"),
        "too few valid samples"
    );
    assert_eq!(
        normalize_inconclusive_reason_label(
            "Direct n-var check remained inconclusive (2 slices inconclusive): Too few valid samples: 0 / 20"
        ),
        "n-var direct check remained inconclusive"
    );
    assert_eq!(
        normalize_inconclusive_reason_label("Unsupported contextual numeric arity: 0"),
        "unsupported contextual numeric arity"
    );
}

#[test]
fn build_nvar_slice_anchors_respects_filters_with_profiles() {
    let mut ctx = Context::new();
    let lhs = parse("arcsin(x/2)+y+z", &mut ctx).expect("parse lhs");
    let rhs = parse("arcsin(x/2)+y+z", &mut ctx).expect("parse rhs");
    let vars = vec!["x".to_string(), "y".to_string(), "z".to_string()];
    let filters = vec![
        FilterSpec::Range {
            min: -0.5,
            max: 0.5,
        },
        FilterSpec::None,
        FilterSpec::None,
    ];
    let anchors = build_nvar_slice_anchors(
        &ctx,
        lhs,
        rhs,
        &vars,
        &filters,
        &metatest_config(),
        0.618_033_988_749_894_8,
    );
    let map = anchors.into_iter().collect::<HashMap<String, f64>>();

    assert!(
        (-0.5..=0.5).contains(&map["x"]),
        "expected filtered x anchor inside [-0.5,0.5], got {}",
        map["x"]
    );
}
