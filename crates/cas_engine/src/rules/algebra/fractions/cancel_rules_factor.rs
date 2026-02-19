//! Factor-based cancellation and rationalization rules.
//!
//! Contains `CancelCommonFactorsRule` (cancel shared factors in num/den),
//! `RationalizeProductDenominatorRule` (rationalize product denominators),
//! and supporting helper functions.

use crate::build::mul2_raw;
use crate::define_rule;
use crate::phase::PhaseMask;
use crate::rule::Rewrite;
use cas_ast::Expr;
use cas_math::expr_terms::contains_irrational;
use cas_math::fraction_factors::collect_mul_factors_flat as collect_mul_factors;
use cas_math::root_forms::extract_root_base_and_index as extract_root_base;
use num_traits::{One, Zero};

define_rule!(
    RationalizeProductDenominatorRule,
    "Rationalize Product Denominator",
    None,
    PhaseMask::RATIONALIZE,
    |ctx, expr| {
        use cas_ast::views::FractionParts;

        // Handle fractions with product denominators containing roots
        let fp = FractionParts::from(&*ctx, expr);
        if !fp.is_fraction() {
            return None;
        }

        let (num, den, _) = fp.to_num_den(ctx);

        let factors = collect_mul_factors(ctx, den);

        // Find a root factor
        let mut root_factor = None;
        let mut non_root_factors = Vec::new();

        for &factor in &factors {
            if extract_root_base(ctx, factor).is_some() && root_factor.is_none() {
                root_factor = Some(factor);
            } else {
                non_root_factors.push(factor);
            }
        }

        let root = root_factor?;

        // Don't apply if denominator is ONLY a root (handled elsewhere or simpler)
        if non_root_factors.is_empty() {
            // Just sqrt(n) in denominator - still rationalize
            if let Some((radicand, _index)) = extract_root_base(ctx, root) {
                // Check if radicand is a binomial (Add or Sub) - these can cause infinite loops
                // when both numerator and denominator have binomial radicals like sqrt(x+y)/sqrt(x-y)
                let is_binomial_radical =
                    matches!(ctx.get(radicand), Expr::Add(_, _) | Expr::Sub(_, _));
                if is_binomial_radical && contains_irrational(ctx, num) {
                    return None;
                }

                // Don't rationalize if radicand is a simple number - power rules handle these better
                // e.g., sqrt(2) / 2^(1/3) should simplify via power combination to 2^(1/6)
                if matches!(ctx.get(radicand), Expr::Number(_)) {
                    return None;
                }

                // 1/sqrt(n) -> sqrt(n)/n
                let new_num = mul2_raw(ctx, num, root);
                let new_den = radicand;
                let new_expr = ctx.add(Expr::Div(new_num, new_den));
                return Some(Rewrite::new(new_expr).desc("Rationalize: multiply by √n/√n"));
            }
            return None;
        }

        // Don't apply if radicand is a simple number - power rules can handle these better
        // e.g., 2*sqrt(2) / (2*2^(1/3)) should simplify via power combination, not rationalization
        if let Some((radicand, _index)) = extract_root_base(ctx, root) {
            if matches!(ctx.get(radicand), Expr::Number(_)) {
                return None;
            }
        }

        // We have: num / (other_factors * root) where root = radicand^(1/index)
        // To rationalize, we need to multiply by radicand^((index-1)/index) / radicand^((index-1)/index)
        // This gives: root * radicand^((index-1)/index) = radicand^(1/index + (index-1)/index) = radicand^1 = radicand
        //
        // For sqrt (index=2): multiply by radicand^(1/2) to get radicand^(1/2 + 1/2) = radicand
        // For cbrt (index=3): multiply by radicand^(2/3) to get radicand^(1/3 + 2/3) = radicand

        if let Some((radicand, index)) = extract_root_base(ctx, root) {
            // Compute the conjugate exponent: (index - 1) / index
            // For square root (index=2): conjugate = 1/2, so conjugate_power = radicand^(1/2) = sqrt(radicand)
            // For cube root (index=3): conjugate = 2/3, so conjugate_power = radicand^(2/3)

            // Get index as integer if possible
            let index_val = if let Expr::Number(n) = ctx.get(index) {
                if n.is_integer() {
                    Some(n.to_integer())
                } else {
                    None
                }
            } else {
                None
            };

            // Only handle integer indices for now
            let index_int = index_val?;
            if index_int <= num_bigint::BigInt::from(1) {
                return None; // Not a valid root index
            }

            // Build conjugate exponent (index - 1) / index
            let one = num_bigint::BigInt::from(1);
            let conjugate_num = &index_int - &one;
            let conjugate_exp = num_rational::BigRational::new(conjugate_num, index_int);
            let conjugate_exp_id = ctx.add(Expr::Number(conjugate_exp));

            // conjugate_power = radicand^((index-1)/index)
            let conjugate_power = ctx.add(Expr::Pow(radicand, conjugate_exp_id));

            // New numerator: num * conjugate_power
            let new_num = mul2_raw(ctx, num, conjugate_power);

            // Build new denominator: other_factors * radicand (since root * conjugate_power = radicand)
            let mut new_den = radicand;
            for &factor in &non_root_factors {
                new_den = mul2_raw(ctx, new_den, factor);
            }

            let new_expr = ctx.add(Expr::Div(new_num, new_den));
            return Some(Rewrite::new(new_expr).desc("Rationalize product denominator"));
        }

        None
    }
);

define_rule!(
    CancelCommonFactorsRule,
    "Cancel Common Factors",
    solve_safety: crate::solve_safety::SolveSafety::NeedsCondition(
        crate::assumptions::ConditionClass::Definability
    ),
    |ctx, expr, parent_ctx| {
        use crate::domain_facts::Predicate;

        // Capture domain mode once at start
        let domain_mode = parent_ctx.domain_mode();

        let (mut num_factors, mut den_factors) = match ctx.get(expr) {
            Expr::Div(n, d) => (collect_mul_factors(ctx, *n), collect_mul_factors(ctx, *d)),
            Expr::Pow(b, e) => {
                let (b, e) = (*b, *e);
                if let Expr::Number(n) = ctx.get(e) {
                    if n.is_integer() && *n == num_rational::BigRational::from_integer((-1).into())
                    {
                        (vec![ctx.num(1)], collect_mul_factors(ctx, b))
                    } else {
                        return None;
                    }
                } else {
                    return None;
                }
            }
            Expr::Mul(_, _) => {
                let factors = collect_mul_factors(ctx, expr);
                let mut nf = Vec::new();
                let mut df = Vec::new();
                for f in factors {
                    if let Expr::Pow(b, e) = ctx.get(f) {
                        if let Expr::Number(n) = ctx.get(*e) {
                            if n.is_integer()
                                && *n == num_rational::BigRational::from_integer((-1).into())
                            {
                                df.extend(collect_mul_factors(ctx, *b));
                                continue;
                            }
                        }
                    }
                    nf.push(f);
                }
                if df.is_empty() {
                    return None;
                }
                (nf, df)
            }
            _ => return None,
        };
        // NOTE: Pythagorean identity simplification (k - k*sin² → k*cos²) has been
        // extracted to TrigPythagoreanSimplifyRule for pedagogical clarity.
        // CancelCommonFactorsRule now does pure factor cancellation.

        let mut changed = false;
        let mut assumption_events: smallvec::SmallVec<[crate::assumptions::AssumptionEvent; 1]> = Default::default();
        let mut i = 0;
        while i < num_factors.len() {
            let nf = num_factors[i];
            // println!("Processing num factor: {:?}", ctx.get(nf));
            let mut found = false;
            for j in 0..den_factors.len() {
                let df = den_factors[j];

                // Check exact match
                if crate::ordering::compare_expr(ctx, nf, df) == std::cmp::Ordering::Equal {
                    // DOMAIN GATE: use canonical helper
                    let decision = crate::domain_oracle::oracle_allows_with_hint(
                        ctx,
                        domain_mode,
                        parent_ctx.value_domain(),
                        &Predicate::NonZero(nf),
                        "Cancel Common Factors",
                    );
                    if !decision.allow {
                        continue; // Skip this pair in strict mode
                    }
                    // Record assumption if made
                    if decision.assumption.is_some() {
                        assumption_events.push(
                            crate::assumptions::AssumptionEvent::nonzero(ctx, nf)
                        );
                    }
                    den_factors.remove(j);
                    found = true;
                    changed = true;
                    break;
                }

                // Check power cancellation: nf = x^n, df = x^m
                // Case 1: nf = base^n, df = base. (integer n only to preserve rationalized forms)
                let nf_pow = if let Expr::Pow(b, e) = ctx.get(nf) {
                    Some((*b, *e))
                } else {
                    None
                };
                if let Some((b, e)) = nf_pow {
                    if crate::ordering::compare_expr(ctx, b, df) == std::cmp::Ordering::Equal {
                        if let Expr::Number(n) = ctx.get(e) {
                            // Guard: only integer exponents - skip fractional to preserve rationalized forms
                            // E.g., sqrt(x)/x should NOT become x^(-1/2) as this undoes rationalization
                            if !n.is_integer() {
                                // Skip this pair, continue to next
                            } else {
                                let new_exp = n - num_rational::BigRational::one();
                                if new_exp.is_zero() {
                                    // x^1 / x = 1, remove both factors
                                    let decision = crate::domain_oracle::oracle_allows_with_hint(
                                        ctx,
                                        domain_mode,
                                        parent_ctx.value_domain(),
                                        &Predicate::NonZero(b),
                                        "Cancel Common Factors",
                                    );
                                    if !decision.allow {
                                        continue; // Skip in strict mode
                                    }
                                    // Record assumption if made
                                    if decision.assumption.is_some() {
                                        assumption_events.push(
                                            crate::assumptions::AssumptionEvent::nonzero(ctx, b)
                                        );
                                    }
                                    den_factors.remove(j);
                                    found = true; // Remove num factor too
                                    changed = true;
                                    break;
                                }
                                let new_term = if new_exp.is_one() {
                                    b
                                } else {
                                    let exp_node = ctx.add(Expr::Number(new_exp));
                                    ctx.add(Expr::Pow(b, exp_node))
                                };
                                num_factors[i] = new_term;
                                den_factors.remove(j);
                                found = false; // Modified num factor
                                changed = true;
                                break;
                            }
                        }
                    }
                }

                // Case 2: nf = base, df = base^m. (integer m only to preserve rationalized forms)
                let df_pow = if let Expr::Pow(b, e) = ctx.get(df) {
                    Some((*b, *e))
                } else {
                    None
                };
                if let Some((b, e)) = df_pow {
                    if crate::ordering::compare_expr(ctx, nf, b) == std::cmp::Ordering::Equal {
                        if let Expr::Number(m) = ctx.get(e) {
                            // Guard: only integer exponents - skip fractional to preserve rationalized forms
                            // E.g., x/sqrt(x) with fractional exp handled by QuotientOfPowersRule
                            if !m.is_integer() {
                                // Skip this pair, continue to next
                            } else {
                                let new_exp = m - num_rational::BigRational::one();
                                if new_exp.is_zero() {
                                    // x / x^1 = 1, remove both factors
                                    let decision = crate::domain_oracle::oracle_allows_with_hint(
                                        ctx,
                                        domain_mode,
                                        parent_ctx.value_domain(),
                                        &Predicate::NonZero(b),
                                        "Cancel Common Factors",
                                    );
                                    if !decision.allow {
                                        continue; // Skip in strict mode
                                    }
                                    den_factors.remove(j);
                                    found = true; // Remove num factor too
                                    changed = true;
                                    break;
                                }
                                let new_term = if new_exp.is_one() {
                                    b
                                } else {
                                    let exp_node = ctx.add(Expr::Number(new_exp));
                                    ctx.add(Expr::Pow(b, exp_node))
                                };
                                den_factors[j] = new_term;
                                found = true; // Remove num factor
                                changed = true;
                                break;
                            }
                        }
                    }
                }

                // Case 3: nf = base^n, df = base^m (integer exponents only)
                // Fractional exponents are handled atomically by QuotientOfPowersRule
                if let Some((b_n, e_n)) = nf_pow {
                    if let Some((b_d, e_d)) = df_pow {
                        if crate::ordering::compare_expr(ctx, b_n, b_d) == std::cmp::Ordering::Equal
                        {
                            if let (Expr::Number(n), Expr::Number(m)) = (ctx.get(e_n), ctx.get(e_d))
                            {
                                // Skip fractional exponents - QuotientOfPowersRule handles them
                                if !n.is_integer() || !m.is_integer() {
                                    // Continue to next factor, don't process this pair
                                } else if n > m {
                                    let new_exp = n - m;
                                    let new_term = if new_exp.is_one() {
                                        b_n
                                    } else {
                                        let exp_node = ctx.add(Expr::Number(new_exp));
                                        ctx.add(Expr::Pow(b_n, exp_node))
                                    };
                                    num_factors[i] = new_term;
                                    den_factors.remove(j);
                                    found = false;
                                    changed = true;
                                    break;
                                } else if m > n {
                                    let new_exp = m - n;
                                    let new_term = if new_exp.is_one() {
                                        b_d
                                    } else {
                                        let exp_node = ctx.add(Expr::Number(new_exp));
                                        ctx.add(Expr::Pow(b_d, exp_node))
                                    };
                                    den_factors[j] = new_term;
                                    found = true;
                                    changed = true;
                                    break;
                                } else {
                                    // x^n / x^n (n == m), remove both factors
                                    let decision = crate::domain_oracle::oracle_allows_with_hint(
                                        ctx,
                                        domain_mode,
                                        parent_ctx.value_domain(),
                                        &Predicate::NonZero(b_n),
                                        "Cancel Common Factors",
                                    );
                                    if !decision.allow {
                                        continue; // Skip in strict mode
                                    }
                                    // Record assumption if made
                                    if decision.assumption.is_some() {
                                        assumption_events.push(
                                            crate::assumptions::AssumptionEvent::nonzero(ctx, b_n)
                                        );
                                    }
                                    den_factors.remove(j);
                                    found = true;
                                    changed = true;
                                    break;
                                } // end else for integer exponents
                            }
                        }
                    }
                }
            }
            if found {
                num_factors.remove(i);
            } else {
                i += 1;
            }
        }

        if changed {
            // Reconstruct
            let new_num = if num_factors.is_empty() {
                ctx.num(1)
            } else {
                let mut n = num_factors[0];
                for &f in num_factors.iter().skip(1) {
                    n = mul2_raw(ctx, n, f);
                }
                n
            };
            let new_den = if den_factors.is_empty() {
                ctx.num(1)
            } else {
                let mut d = den_factors[0];
                for &f in den_factors.iter().skip(1) {
                    d = mul2_raw(ctx, d, f);
                }
                d
            };

            let new_expr = ctx.add(Expr::Div(new_num, new_den));
            return Some(Rewrite::new(new_expr)
                .desc("Cancel common factors")
                .local(expr, new_expr)
                .assume_all(assumption_events));
        }

        None
    }
);
