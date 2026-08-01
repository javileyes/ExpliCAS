//! Polynomial comparison helpers for lightweight equivalence checks.
//!
//! These utilities compare expressions by converting to canonical `MultiPoly`
//! under a tight budget, instead of relying on AST shape.

use crate::multipoly::{multipoly_from_expr, PolyBudget};
use cas_ast::{Context, ExprId};

fn compare_budget() -> PolyBudget {
    PolyBudget {
        max_terms: 100,
        max_total_degree: 10,
        max_pow_exp: 5,
    }
}

/// Budget for the exact zero-decision of `poly_is_zero`.
///
/// Wider than `compare_budget` on purpose: the clientele is expand()-produced
/// flat numerators (fraction-combine zero checks), whose degree routinely
/// exceeds 10 while staying cheap to convert (monomials don't multiply out —
/// `max_pow_exp` only gates `Pow(Add, n)` expansion). The caps still bound
/// adversarial inputs; beyond them the answer is a conservative `false`
/// ("cannot PROVE zero"), never a guess.
fn zero_check_budget() -> PolyBudget {
    PolyBudget {
        max_terms: 256,
        max_total_degree: 64,
        max_pow_exp: 8,
    }
}

/// Decide EXACTLY whether `expr` is the identically-zero polynomial.
///
/// This is the only sanctioned way to CONFIRM "is zero" for expressions with
/// variables (R2: soundness gates are exact — probes may refute, never
/// confirm). Returns `false` when the expression is not a polynomial within
/// budget (surds, π/e, trig, |·|, oversized): that is "cannot prove zero",
/// not "nonzero".
pub(crate) fn poly_is_zero(ctx: &Context, expr: ExprId) -> bool {
    // Rational-function normalization: `num/den ≡ 0 ⟺ num ≡ 0 ∧ den ≢ 0`.
    // Semi-combined fraction numerators carry embedded `Div` nodes (the
    // Bernoulli-IVP Gate-1 shape); the plain polynomial converter would
    // reject them with `NonConstantDivision` and lose a true zero.
    matches!(
        crate::multipoly::rational_multipoly_from_expr(ctx, expr, &zero_check_budget()),
        Ok((num, den)) if num.is_zero() && !den.is_zero()
    )
}

/// `poly_is_zero` over the polynomial CLOSURE of opaque atoms: maximal
/// non-polynomial subtrees that denote a (pointwise) well-defined value —
/// function calls, the finite constants π/e/φ/i, non-integer powers — are
/// replaced by fresh independent indeterminates before conversion.
///
/// Soundness: if the result is the zero polynomial with the atoms treated as
/// INDEPENDENT indeterminates, the original is identically zero for the
/// actual values of those atoms (substitution instance of the zero
/// polynomial). The converse is false (`sin²+cos²−1`, `√x·√x−x` have hidden
/// relations), so a nonzero polynomial here proves NOTHING — this helper is
/// confirmation-only and the caller must treat `false` as "cannot prove".
///
/// `Infinity`/`Undefined`/`Matrix` are NEVER atomized: an `∞` indeterminate
/// would "prove" `∞ − ∞ = 0`. Their presence bails the whole check.
pub(crate) fn poly_is_zero_opaque(ctx: &mut Context, expr: ExprId) -> bool {
    if poly_is_zero(ctx, expr) {
        return true;
    }
    // FOLDS (all faithful exact identities on the expression's own domain,
    // never mere atom independence):
    // 1. e^(k·g) = (e^g)^k — the Bernoulli-IVP Gate-1 residue mixes e^x with
    //    e^(2x) and is zero only THROUGH that relation.
    // 2. B^(p/q) = s^(p·d/q) with s = B^(1/d), d = lcm of the fractional-
    //    exponent denominators of base B (and B itself = s^d, √B = s^(d/2)):
    //    the arctan(√x) verification residue mixes √x, x^(1/2), x^(3/2), x.
    // 3. tan/cot/sec/csc and their hyperbolic siblings become sin/cos ratios,
    //    so cosh²·tanh² = sinh² closes polynomially.
    // The folds happen INSIDE `atomize_non_poly` (which walks Div too — the
    // Laurent utilities in `opaque_atoms` don't, and the Gate-1 numerator
    // carries its e^x inside embedded fractions).
    // Nombres sintéticos vía el asignador canónico (cas_ast::fresh_names,
    // clase L15). Se precalculan aquí porque atomize/collect reciben bases
    // numéricas simples; el conjunto ya queda marcado y es monótono.
    let mut taken = cas_ast::fresh_names::taken_variable_names(ctx, &[expr]);
    // Bases de LOTE (esquema `prefix{base+i}`): exigen la semántica max+1 de
    // `fresh_suffix_base` — con huecos en taken, el primer-libre de
    // `alloc_indexed_name` colisionaría más adelante.
    let atom_base = cas_ast::fresh_names::fresh_suffix_base(&taken, "__polyzero_atom_");
    let root_base = cas_ast::fresh_names::fresh_suffix_base(&taken, "__polyzero_root_");
    let mut exps = Vec::new();
    collect_e_exponents_full(ctx, expr, &mut exps, 0);
    let expfold = if exps.is_empty() {
        None
    } else {
        crate::opaque_atoms::find_exp_base(ctx, &exps, 16).map(|g| {
            let expatom_name = if taken.contains("__polyzero_expatom") {
                cas_ast::fresh_names::alloc_indexed_name(&mut taken, "__polyzero_expatom_", 0)
            } else {
                "__polyzero_expatom".to_string()
            };
            let u = ctx.var(&expatom_name);
            (g, u)
        })
    };
    let powfold = collect_fractional_power_bases(ctx, expr, root_base);
    let mut atoms: rustc_hash::FxHashMap<ExprId, ExprId> = rustc_hash::FxHashMap::default();
    let Some(atomized) = atomize_non_poly(ctx, expr, &mut atoms, expfold, &powfold, atom_base)
    else {
        return false;
    };
    // Nothing changed ⟹ same expression the direct attempt already rejected.
    if atomized == expr {
        return false;
    }
    // 4. Relation reduction on the final polynomials (all exact identities):
    //    - Pythagorean pairs: cos² = 1 − sin², cosh² = 1 + sinh², whenever
    //      both members over the SAME argument became atoms — closes the
    //      `3 − 3tanh⁴ − 3tanh²/cosh² − 3/cosh²` residues.
    //    - Constant-radical powers: s^d = c for atoms s ≡ c^(1/d) over a
    //      RATIONAL c (√2·√2 = 2; `1/(x⁴−4)` antiderivatives mix √2 and
    //      2^(3/2)) — no independent-atom view can see either family.
    let mut relations = collect_pythagorean_pairs(ctx, &atoms);
    relations.extend(collect_const_radical_relations(ctx, &atoms));
    relations.extend(collect_multiple_angle_relations(ctx, &atoms));
    // Defining relations of the var-base radical atoms: s_B^d = B expressed
    // through the REMAINING folds (`s_{x+1}² = s_x²+1`). Without them, two
    // radicals over algebraically-related bases stay independent and true
    // zeros like `√(x/(x+1))·√(x+1)·√x − x`-shaped residues never close.
    for (base, (d, s_var)) in &powfold {
        let mut others = powfold.clone();
        others.remove(base);
        let Some(base_atomized) =
            atomize_non_poly(ctx, *base, &mut atoms, expfold, &others, atom_base)
        else {
            continue;
        };
        let cas_ast::Expr::Variable(sv) = ctx.get(*s_var) else {
            continue;
        };
        relations.push(AtomRelation::PolyPower {
            s_name: ctx.sym_name(*sv).to_string(),
            d: *d,
            value_expr: base_atomized,
        });
    }
    let budget = zero_check_budget();
    let Ok((num, den)) = crate::multipoly::rational_multipoly_from_expr(ctx, atomized, &budget)
    else {
        return false;
    };
    let Some(num_red) = reduce_by_relations(ctx, &num, &relations, &budget) else {
        return false;
    };
    let Some(den_red) = reduce_by_relations(ctx, &den, &relations, &budget) else {
        return false;
    };
    num_red.is_zero() && !den_red.is_zero()
}

/// Exponent as a rational NUMBER, seeing through one `Neg` wrapper: the
/// parser stores `x^(-3/2)` as `Pow(x, Neg(Number(3/2)))`, and every
/// exponent match on bare `Number` silently misses those.
fn exp_number(ctx: &Context, exp: ExprId) -> Option<num_rational::BigRational> {
    // The canonical foldable-form extractor (ledger lesson: matching bare
    // `Number` literals misses `Neg(Div(3,2))`-shaped exponents entirely).
    crate::numeric_eval::as_rational_const(ctx, exp)
}

/// One substitution rule for the final-polynomial reduction.
enum AtomRelation {
    /// `c_var² = 1 + sign·s_var²` (trig `sign=-1`, hyperbolic `sign=+1`).
    Pythagoras {
        c_name: String,
        s_name: String,
        sign: i64,
    },
    /// `s_var^d = value` for an atom `s ≡ value^(1/d)` over rational `value`.
    RadicalPower {
        s_name: String,
        d: u32,
        value: num_rational::BigRational,
    },
    /// `s_var^d = B` where `B` is the (atomized) defining expression of a
    /// var-base radical atom — e.g. `s_{x+1}² = s_x² + 1` when both `√x` and
    /// `√(x+1)` were folded. Converted to a polynomial at reduce time over
    /// the current variable list.
    PolyPower {
        s_name: String,
        d: u32,
        value_expr: ExprId,
    },
}

/// Atoms that are radicals of RATIONAL constants. At atomization every
/// `c^(p/q)` and `sqrt(c)` folds to `c^w · s^r` with `s` keyed by the
/// CANONICAL node `Pow(c, 1/q)`, so here each such atom yields one relation
/// `s^q = c`.
fn collect_const_radical_relations(
    ctx: &Context,
    atoms: &rustc_hash::FxHashMap<ExprId, ExprId>,
) -> Vec<AtomRelation> {
    use num_traits::{Signed, ToPrimitive};
    let mut out = Vec::new();
    for (key, var) in atoms {
        let cas_ast::Expr::Pow(base, exp) = ctx.get(*key) else {
            continue;
        };
        let cas_ast::Expr::Number(c) = ctx.get(*base) else {
            continue;
        };
        let cas_ast::Expr::Number(e) = ctx.get(*exp) else {
            continue;
        };
        if e.numer() != &num_bigint::BigInt::from(1) {
            continue;
        }
        let Some(q) = e.denom().to_u32().filter(|q| (2..=12).contains(q)) else {
            continue;
        };
        if c.is_negative() {
            continue;
        }
        let cas_ast::Expr::Variable(sv) = ctx.get(*var) else {
            continue;
        };
        out.push(AtomRelation::RadicalPower {
            s_name: ctx.sym_name(*sv).to_string(),
            d: q,
            value: c.clone(),
        });
    }
    out
}

/// Find atom pairs `(cos-like, sin-like)` over the same argument and emit the
/// substitution `cos² ↦ 1 ∓ sin²` as (cos_var_name, sin_var_name, sign) with
/// `sign = -1` for trig (cos² = 1 − sin²) and `+1` for hyperbolic
/// (cosh² = 1 + sinh²).
fn collect_pythagorean_pairs(
    ctx: &mut Context,
    atoms: &rustc_hash::FxHashMap<ExprId, ExprId>,
) -> Vec<AtomRelation> {
    use cas_ast::BuiltinFn;
    let mut out = Vec::new();
    let keys: Vec<ExprId> = atoms.keys().copied().collect();
    for key in keys {
        let cas_ast::Expr::Function(f, args) = ctx.get(key).clone() else {
            continue;
        };
        if args.len() != 1 {
            continue;
        }
        let (sin_b, sign) = if ctx.is_builtin(f, BuiltinFn::Cos) {
            (BuiltinFn::Sin, -1i64)
        } else if ctx.is_builtin(f, BuiltinFn::Cosh) {
            (BuiltinFn::Sinh, 1i64)
        } else {
            continue;
        };
        let sin_node = ctx.call_builtin(sin_b, args);
        let Some(sin_var) = atoms.get(&sin_node) else {
            continue;
        };
        let cos_var = atoms[&key];
        let (cas_ast::Expr::Variable(cv), cas_ast::Expr::Variable(sv)) =
            (ctx.get(cos_var).clone(), ctx.get(*sin_var).clone())
        else {
            continue;
        };
        out.push(AtomRelation::Pythagoras {
            c_name: ctx.sym_name(cv).to_string(),
            s_name: ctx.sym_name(sv).to_string(),
            sign,
        });
    }
    out
}

/// Multiple-angle relations: an atom `sin(k·a)`/`cos(k·a)` coexisting with
/// the atoms `sin(a)` AND `cos(a)` substitutes to its Chebyshev-style
/// polynomial in them (recurrence sin(ka) = sin((k−1)a)cos(a) +
/// cos((k−1)a)sin(a) and the cosine analogue) — exact identities, emitted as
/// `d = 1` power relations. Closes `sin(8x) − P(sin x, cos x)` verification
/// residues of the product-to-sum integrals.
fn collect_multiple_angle_relations(
    ctx: &mut Context,
    atoms: &rustc_hash::FxHashMap<ExprId, ExprId>,
) -> Vec<AtomRelation> {
    use cas_ast::BuiltinFn;
    let mut out = Vec::new();
    let entries: Vec<(ExprId, ExprId)> = atoms.iter().map(|(k, v)| (*k, *v)).collect();
    for (big_key, big_var) in &entries {
        let cas_ast::Expr::Function(bf, bargs) = ctx.get(*big_key).clone() else {
            continue;
        };
        if bargs.len() != 1 {
            continue;
        }
        let big_is_sin = ctx.is_builtin(bf, BuiltinFn::Sin);
        let big_is_cos = ctx.is_builtin(bf, BuiltinFn::Cos);
        if !big_is_sin && !big_is_cos {
            continue;
        }
        for (small_key, _) in &entries {
            if small_key == big_key {
                continue;
            }
            let cas_ast::Expr::Function(sf, sargs) = ctx.get(*small_key).clone() else {
                continue;
            };
            let small_is_sincos =
                ctx.is_builtin(sf, BuiltinFn::Sin) || ctx.is_builtin(sf, BuiltinFn::Cos);
            if sargs.len() != 1 || !small_is_sincos {
                continue;
            }
            let a = sargs[0];
            let Some(k) = exp_integer_multiple_of(ctx, bargs[0], a) else {
                continue;
            };
            if !(2..=16).contains(&k) {
                continue;
            }
            let sin_a = ctx.call_builtin(BuiltinFn::Sin, vec![a]);
            let cos_a = ctx.call_builtin(BuiltinFn::Cos, vec![a]);
            let sv = atoms.get(&sin_a).copied();
            let Some(cv) = atoms.get(&cos_a).copied() else {
                continue;
            };
            let value_expr = if big_is_cos && sv.is_none() {
                // cos(k·a) = T_k(cos a): pure-cosine Chebyshev recurrence
                // T_k = 2c·T_{k−1} − T_{k−2} — no sine atom required (the
                // cos·cos product-to-sum residues never mention sin).
                let two = ctx.num(2);
                let mut t_prev = ctx.num(1); // T_0
                let mut t_cur = cv; // T_1
                for _ in 1..k {
                    let two_c = ctx.add(cas_ast::Expr::Mul(two, cv));
                    let m = ctx.add(cas_ast::Expr::Mul(two_c, t_cur));
                    let next = ctx.add(cas_ast::Expr::Sub(m, t_prev));
                    t_prev = t_cur;
                    t_cur = next;
                }
                t_cur
            } else {
                let Some(sv) = sv else {
                    continue;
                };
                // Angle-addition recurrence over both atoms.
                let mut sin_k = sv;
                let mut cos_k = cv;
                for _ in 1..k {
                    let t1 = ctx.add(cas_ast::Expr::Mul(sin_k, cv));
                    let t2 = ctx.add(cas_ast::Expr::Mul(cos_k, sv));
                    let new_sin = ctx.add(cas_ast::Expr::Add(t1, t2));
                    let t3 = ctx.add(cas_ast::Expr::Mul(cos_k, cv));
                    let t4 = ctx.add(cas_ast::Expr::Mul(sin_k, sv));
                    let new_cos = ctx.add(cas_ast::Expr::Sub(t3, t4));
                    sin_k = new_sin;
                    cos_k = new_cos;
                }
                if big_is_sin {
                    sin_k
                } else {
                    cos_k
                }
            };
            let cas_ast::Expr::Variable(bvar) = ctx.get(*big_var) else {
                continue;
            };
            out.push(AtomRelation::PolyPower {
                s_name: ctx.sym_name(*bvar).to_string(),
                d: 1,
                value_expr,
            });
            break;
        }
    }
    out
}

/// Reduce `p` modulo the collected exact relations: Pythagorean pairs
/// (`c² ↦ 1 + sign·s²`) and constant-radical powers (`s^d ↦ value`). Each
/// substitution is a faithful identity; the result is `None` only on budget
/// blowup.
fn reduce_by_relations(
    ctx: &Context,
    p: &crate::multipoly::MultiPoly,
    relations: &[AtomRelation],
    budget: &PolyBudget,
) -> Option<crate::multipoly::MultiPoly> {
    let mut current = p.clone();
    // Relations may chain (s2² → s3²+1 introduces s3 which another relation
    // reduces); a few bounded rounds reach the fixpoint.
    for _round in 0..3 {
        let before = current.clone();
        current = reduce_round(ctx, &current, relations, budget)?;
        if current == before {
            break;
        }
    }
    Some(current)
}

fn reduce_round(
    ctx: &Context,
    p: &crate::multipoly::MultiPoly,
    relations: &[AtomRelation],
    budget: &PolyBudget,
) -> Option<crate::multipoly::MultiPoly> {
    use crate::multipoly::MultiPoly;
    use num_rational::BigRational;
    let mut current = p.clone();
    for rel in relations {
        match rel {
            AtomRelation::RadicalPower { s_name, d, value } => {
                let Some(s_idx) = current.var_index(s_name) else {
                    continue;
                };
                let mut acc_terms = Vec::with_capacity(current.terms.len());
                for (coeff, mono) in &current.terms {
                    let e = mono.get(s_idx).copied().unwrap_or(0);
                    let (w, r) = (e / d, e % d);
                    let mut new_coeff = coeff.clone();
                    for _ in 0..w {
                        new_coeff *= value;
                    }
                    let mut new_mono = mono.clone();
                    new_mono[s_idx] = r;
                    acc_terms.push((new_coeff, new_mono));
                }
                let mut acc = MultiPoly::zero(current.vars.clone());
                for (coeff, mono) in acc_terms {
                    let t = MultiPoly {
                        vars: current.vars.clone(),
                        terms: vec![(coeff, mono)],
                    };
                    acc = acc.add(&t).ok()?;
                }
                current = acc;
            }
            AtomRelation::PolyPower {
                s_name,
                d,
                value_expr,
            } => {
                let Some(s_idx) = current.var_index(s_name) else {
                    continue;
                };
                let Ok(value) = crate::multipoly::multipoly_from_expr_with_vars(
                    ctx,
                    *value_expr,
                    &current.vars,
                    budget,
                ) else {
                    continue;
                };
                let mut acc = MultiPoly::zero(current.vars.clone());
                for (coeff, mono) in &current.terms {
                    let e = mono.get(s_idx).copied().unwrap_or(0);
                    let (w, r) = (e / d, e % d);
                    let mut base_mono = mono.clone();
                    base_mono[s_idx] = r;
                    let mut term_poly = MultiPoly {
                        vars: current.vars.clone(),
                        terms: vec![(coeff.clone(), base_mono)],
                    };
                    for _ in 0..w {
                        term_poly = term_poly.mul(&value, budget).ok()?;
                    }
                    acc = acc.add(&term_poly).ok()?;
                }
                current = acc;
            }
            AtomRelation::Pythagoras {
                c_name,
                s_name,
                sign,
            } => {
                let Some(c_idx) = current.var_index(c_name) else {
                    continue;
                };
                let Some(s_idx) = current.var_index(s_name) else {
                    continue;
                };
                let mut s2_mono = vec![0u32; current.vars.len()];
                s2_mono[s_idx] = 2;
                let repl = MultiPoly {
                    vars: current.vars.clone(),
                    terms: vec![
                        (
                            BigRational::from_integer(1.into()),
                            vec![0; current.vars.len()],
                        ),
                        (BigRational::from_integer((*sign).into()), s2_mono),
                    ],
                };
                let mut acc = MultiPoly::zero(current.vars.clone());
                for (coeff, mono) in &current.terms {
                    let e = mono.get(c_idx).copied().unwrap_or(0);
                    let mut base_mono = mono.clone();
                    base_mono[c_idx] = e % 2;
                    let base = MultiPoly {
                        vars: current.vars.clone(),
                        terms: vec![(coeff.clone(), base_mono)],
                    };
                    let mut term_poly = base;
                    for _ in 0..(e / 2) {
                        term_poly = term_poly.mul(&repl, budget).ok()?;
                    }
                    acc = acc.add(&term_poly).ok()?;
                }
                current = acc;
            }
        }
    }
    Some(current)
}

/// For every non-constant base `B` that appears with a FRACTIONAL rational
/// exponent (`B^(p/q)` or `sqrt(B)`), compute `d` = lcm of the denominators
/// (capped at 12) and mint the atom `s ≡ B^(1/d)`. Every occurrence of `B`
/// itself then rewrites to `s^d` — a faithful reparametrization on the
/// domain where the fractional powers exist (d even ⟹ B ≥ 0 there; d odd ⟹
/// real d-th root is a bijection).
fn collect_fractional_power_bases(
    ctx: &mut Context,
    expr: ExprId,
    root_base: usize,
) -> rustc_hash::FxHashMap<ExprId, (u32, ExprId)> {
    use num_traits::ToPrimitive;
    fn lcm(a: u32, b: u32) -> u32 {
        let g = {
            let (mut x, mut y) = (a, b);
            while y != 0 {
                let t = x % y;
                x = y;
                y = t;
            }
            x
        };
        (a / g).saturating_mul(b)
    }
    fn scan(ctx: &Context, e: ExprId, out: &mut rustc_hash::FxHashMap<ExprId, u32>, depth: usize) {
        if depth > 64 {
            return;
        }
        match ctx.get(e) {
            cas_ast::Expr::Pow(base, exp) => {
                if let Some(n) = exp_number(ctx, *exp) {
                    if !n.is_integer() {
                        if let Some(q) = n.denom().to_u32() {
                            if (2..=12).contains(&q)
                                && !cas_ast::collect_variables(ctx, *base).is_empty()
                            {
                                let d = out.entry(*base).or_insert(1);
                                *d = lcm(*d, q).min(12);
                            }
                        }
                    }
                }
                scan(ctx, *base, out, depth + 1);
                scan(ctx, *exp, out, depth + 1);
            }
            cas_ast::Expr::Function(f, args) => {
                if ctx.is_builtin(*f, cas_ast::BuiltinFn::Sqrt)
                    && args.len() == 1
                    && !cas_ast::collect_variables(ctx, args[0]).is_empty()
                {
                    let d = out.entry(args[0]).or_insert(1);
                    *d = lcm(*d, 2).min(12);
                }
                for a in args {
                    scan(ctx, *a, out, depth + 1);
                }
            }
            cas_ast::Expr::Add(l, r)
            | cas_ast::Expr::Sub(l, r)
            | cas_ast::Expr::Mul(l, r)
            | cas_ast::Expr::Div(l, r) => {
                scan(ctx, *l, out, depth + 1);
                scan(ctx, *r, out, depth + 1);
            }
            cas_ast::Expr::Neg(i) | cas_ast::Expr::Hold(i) => scan(ctx, *i, out, depth + 1),
            _ => {}
        }
    }
    let mut denoms: rustc_hash::FxHashMap<ExprId, u32> = rustc_hash::FxHashMap::default();
    scan(ctx, expr, &mut denoms, 0);
    denoms.retain(|_, d| *d >= 2);
    let mut out = rustc_hash::FxHashMap::default();
    for (i, (base, d)) in denoms.into_iter().enumerate() {
        let s = ctx.var(&format!("__polyzero_root_{}", root_base + i));
        out.insert(base, (d, s));
    }
    out
}

/// Collect exponents of `e^(...)` nodes over the FULL tree (including `Div`,
/// which the Laurent-oriented `opaque_atoms::collect_exp_exponents` skips).
/// Function arguments are NOT entered: functions are atomized whole.
fn collect_e_exponents_full(ctx: &Context, expr: ExprId, out: &mut Vec<ExprId>, depth: usize) {
    if depth > 64 {
        return;
    }
    match ctx.get(expr) {
        cas_ast::Expr::Pow(base, exp) => {
            if matches!(
                ctx.get(*base),
                cas_ast::Expr::Constant(cas_ast::Constant::E)
            ) {
                out.push(*exp);
            } else {
                collect_e_exponents_full(ctx, *base, out, depth + 1);
                collect_e_exponents_full(ctx, *exp, out, depth + 1);
            }
        }
        cas_ast::Expr::Add(l, r)
        | cas_ast::Expr::Sub(l, r)
        | cas_ast::Expr::Mul(l, r)
        | cas_ast::Expr::Div(l, r) => {
            collect_e_exponents_full(ctx, *l, out, depth + 1);
            collect_e_exponents_full(ctx, *r, out, depth + 1);
        }
        cas_ast::Expr::Neg(inner) | cas_ast::Expr::Hold(inner) => {
            collect_e_exponents_full(ctx, *inner, out, depth + 1)
        }
        _ => {}
    }
}

/// `exp == k·g` for integer `1 ≤ k ≤ 16`? Returns `k`. Mirrors the Laurent
/// extractor but local to the zero-decider's contract.
fn exp_integer_multiple_of(ctx: &Context, exp: ExprId, g: ExprId) -> Option<i64> {
    use cas_ast::ordering::compare_expr;
    use num_traits::ToPrimitive;
    if compare_expr(ctx, exp, g) == std::cmp::Ordering::Equal {
        return Some(1);
    }
    if let cas_ast::Expr::Mul(l, r) = ctx.get(exp) {
        for (num_side, rest) in [(*l, *r), (*r, *l)] {
            if let cas_ast::Expr::Number(n) = ctx.get(num_side) {
                if n.is_integer() {
                    if let Some(k) = n.to_integer().to_i64() {
                        if (1..=16).contains(&k)
                            && compare_expr(ctx, rest, g) == std::cmp::Ordering::Equal
                        {
                            return Some(k);
                        }
                    }
                }
            }
        }
    }
    None
}

/// Rebuild `expr` with every maximal non-polynomial-but-defined subtree
/// replaced by a fresh opaque variable (one per DISTINCT normalized subtree —
/// the arena interns structurally, so `ExprId` identity is structural
/// identity). Three faithful folds run first (see `poly_is_zero_opaque`):
/// e^(k·g) ↦ u^k, fractional powers of a base ↦ integer powers of s=B^(1/d)
/// (including bare `B` ↦ s^d and `sqrt(B)` ↦ s^(d/2)), and quotient
/// trig/hyperbolic functions ↦ sin/cos ratios. Function arguments are
/// normalized RECURSIVELY before interning the atom key, so `sinh(√x−b)` and
/// `sinh(x^(1/2)−b)` share one atom. Returns `None` when the expression
/// contains a node that must never be atomized (`Infinity`, `Undefined`,
/// matrices, session refs).
fn atomize_non_poly(
    ctx: &mut Context,
    expr: ExprId,
    atoms: &mut rustc_hash::FxHashMap<ExprId, ExprId>,
    expfold: Option<(ExprId, ExprId)>,
    powfold: &rustc_hash::FxHashMap<ExprId, (u32, ExprId)>,
    atom_base: usize,
) -> Option<ExprId> {
    use cas_ast::Constant;
    fn atom_for(
        ctx: &mut Context,
        atoms: &mut rustc_hash::FxHashMap<ExprId, ExprId>,
        key: ExprId,
        atom_base: usize,
    ) -> ExprId {
        if let Some(v) = atoms.get(&key) {
            return *v;
        }
        let name = format!("__polyzero_atom_{}", atom_base + atoms.len());
        let v = ctx.var(&name);
        atoms.insert(key, v);
        v
    }
    // Bare occurrence of a fractional-power base: B ↦ s^d.
    if let Some((d, s)) = powfold.get(&expr) {
        let dn = ctx.num(*d as i64);
        return Some(ctx.add(cas_ast::Expr::Pow(*s, dn)));
    }
    let node = ctx.get(expr).clone();
    match node {
        cas_ast::Expr::Number(_) | cas_ast::Expr::Variable(_) => Some(expr),
        cas_ast::Expr::Constant(c) => match c {
            Constant::Infinity | Constant::Undefined => None,
            _ => Some(atom_for(ctx, atoms, expr, atom_base)),
        },
        cas_ast::Expr::Add(a, b)
        | cas_ast::Expr::Sub(a, b)
        | cas_ast::Expr::Mul(a, b)
        | cas_ast::Expr::Div(a, b) => {
            let na = atomize_non_poly(ctx, a, atoms, expfold, powfold, atom_base)?;
            let nb = atomize_non_poly(ctx, b, atoms, expfold, powfold, atom_base)?;
            if na == a && nb == b {
                return Some(expr);
            }
            let rebuilt = match ctx.get(expr) {
                cas_ast::Expr::Add(..) => cas_ast::Expr::Add(na, nb),
                cas_ast::Expr::Sub(..) => cas_ast::Expr::Sub(na, nb),
                cas_ast::Expr::Mul(..) => cas_ast::Expr::Mul(na, nb),
                _ => cas_ast::Expr::Div(na, nb),
            };
            Some(ctx.add(rebuilt))
        }
        cas_ast::Expr::Neg(inner) => {
            let ni = atomize_non_poly(ctx, inner, atoms, expfold, powfold, atom_base)?;
            if ni == inner {
                return Some(expr);
            }
            Some(ctx.add(cas_ast::Expr::Neg(ni)))
        }
        cas_ast::Expr::Pow(base, exp) => {
            // Exp-fold first: e^(k·g) ↦ u^k (faithful identity).
            if let Some((g, u)) = expfold {
                if matches!(ctx.get(base), cas_ast::Expr::Constant(Constant::E)) {
                    if let Some(k) = exp_integer_multiple_of(ctx, exp, g) {
                        if k == 1 {
                            return Some(u);
                        }
                        let kn = ctx.num(k);
                        return Some(ctx.add(cas_ast::Expr::Pow(u, kn)));
                    }
                }
            }
            // Product/quotient radical split: (A·B)^(p/q) ↦ A^(p/q)·B^(p/q)
            // (and Div alike), ONLY when each factor is a nonnegative
            // rational constant or a base whose OWN fractional powers already
            // appear in the expression (their presence pins the domain where
            // the multiplicativity identity holds; for odd q it holds
            // unconditionally). The rebuilt parts re-enter this atomizer and
            // fold through the constant/base arms.
            if let Some(n) = exp_number(ctx, exp) {
                if !n.is_integer() {
                    let radicand = ctx.get(base).clone();
                    if let cas_ast::Expr::Mul(fa, fb) | cas_ast::Expr::Div(fa, fb) = radicand {
                        use num_traits::{Signed, ToPrimitive};
                        let q_odd = n.denom().to_u32().is_some_and(|q| q % 2 == 1);
                        let splittable = |ctx: &Context, e: ExprId| -> bool {
                            if q_odd || powfold.contains_key(&e) {
                                return true;
                            }
                            matches!(ctx.get(e), cas_ast::Expr::Number(c) if !c.is_negative())
                        };
                        if splittable(ctx, fa) && splittable(ctx, fb) {
                            let exp_node = exp;
                            let pa = ctx.add(cas_ast::Expr::Pow(fa, exp_node));
                            let pb = ctx.add(cas_ast::Expr::Pow(fb, exp_node));
                            let rebuilt = if matches!(ctx.get(expr), cas_ast::Expr::Pow(b, _) if matches!(ctx.get(*b), cas_ast::Expr::Mul(..)))
                            {
                                cas_ast::Expr::Mul(pa, pb)
                            } else {
                                cas_ast::Expr::Div(pa, pb)
                            };
                            let rebuilt_id = ctx.add(rebuilt);
                            return atomize_non_poly(
                                ctx, rebuilt_id, atoms, expfold, powfold, atom_base,
                            );
                        }
                    }
                }
            }
            // Power-fold: B^(p/q) ↦ s^(p·d/q) when the multiple is integral.
            if let Some((d, s)) = powfold.get(&base) {
                if let Some(n) = exp_number(ctx, exp) {
                    use num_traits::ToPrimitive;
                    let scaled = &n * num_rational::BigRational::from_integer((*d).into());
                    if scaled.is_integer() {
                        if let Some(k) = scaled.to_integer().to_i64() {
                            if k.unsigned_abs() <= 64 {
                                let s = *s;
                                let kn = ctx.num(k);
                                return Some(ctx.add(cas_ast::Expr::Pow(s, kn)));
                            }
                        }
                    }
                }
            }
            // Constant-radical fold: c^(p/q) ↦ c^w · s^r with w = p div q,
            // r = p mod q and s the atom keyed by the CANONICAL Pow(c, 1/q)
            // node — so sqrt(2), 2^(1/2) and 2^(3/2) all share one atom and
            // the relation s^q = c (emitted later) closes their algebra.
            if let (cas_ast::Expr::Number(c), Some(n)) =
                (ctx.get(base).clone(), exp_number(ctx, exp))
            {
                use num_traits::{Signed, ToPrimitive};
                if !n.is_integer() && !c.is_negative() {
                    if let (Some(pn), Some(q)) = (n.numer().to_i64(), n.denom().to_u32()) {
                        if (2..=12).contains(&q) && pn.unsigned_abs() <= 64 {
                            // Canonical square-root normalization (q = 2):
                            // √(num/den) = (s/den)·√m with m the SQUAREFREE
                            // part of num·den — so √(5/4) and √5 share one
                            // atom (denested residues die without this).
                            if q == 2 && pn == 1 {
                                use num_traits::Zero as _;
                                let m0 = c.numer() * c.denom();
                                let mut sq = num_bigint::BigInt::from(1);
                                let mut m = m0.clone();
                                let mut pdiv = num_bigint::BigInt::from(2);
                                let limit = num_bigint::BigInt::from(10_000);
                                while &pdiv * &pdiv <= m && pdiv <= limit {
                                    let p2 = &pdiv * &pdiv;
                                    while (&m % &p2).is_zero() {
                                        m /= &p2;
                                        sq *= &pdiv;
                                    }
                                    pdiv += 1;
                                }
                                // leftover perfect square (large prime²)?
                                if let Some(r) = crate::const_sign::exact_nth_root(
                                    &num_rational::BigRational::from_integer(m.clone()),
                                    2,
                                ) {
                                    sq *= r.to_integer();
                                    m = num_bigint::BigInt::from(1);
                                }
                                let coef = num_rational::BigRational::new(sq, c.denom().clone());
                                if m == num_bigint::BigInt::from(1) {
                                    return Some(ctx.add(cas_ast::Expr::Number(coef)));
                                }
                                let m_node = ctx.add(cas_ast::Expr::Number(
                                    num_rational::BigRational::from_integer(m),
                                ));
                                let half = num_rational::BigRational::new(1.into(), 2.into());
                                let half_node = ctx.add(cas_ast::Expr::Number(half));
                                let key = ctx.add(cas_ast::Expr::Pow(m_node, half_node));
                                let atom = atom_for(ctx, atoms, key, atom_base);
                                let coef_node = ctx.add(cas_ast::Expr::Number(coef));
                                return Some(ctx.add(cas_ast::Expr::Mul(coef_node, atom)));
                            }
                            // Perfect root: 4^(1/2) IS 2 — fold to the exact
                            // rational instead of minting an atom whose
                            // linear occurrences no even-power relation can
                            // resolve.
                            if let Some(root) = crate::const_sign::exact_nth_root(&c, q) {
                                let mut val = num_rational::BigRational::from_integer(1.into());
                                if pn >= 0 {
                                    for _ in 0..pn {
                                        val *= &root;
                                    }
                                } else {
                                    for _ in 0..(-pn) {
                                        val /= &root;
                                    }
                                }
                                return Some(ctx.add(cas_ast::Expr::Number(val)));
                            }
                            let w = pn.div_euclid(q as i64);
                            let r = pn.rem_euclid(q as i64) as u32;
                            let inv_q = num_rational::BigRational::new(1.into(), (q as i64).into());
                            let c_node = ctx.add(cas_ast::Expr::Number(c.clone()));
                            let invq_node = ctx.add(cas_ast::Expr::Number(inv_q));
                            let key = ctx.add(cas_ast::Expr::Pow(c_node, invq_node));
                            let s = atom_for(ctx, atoms, key, atom_base);
                            let mut cw = num_rational::BigRational::from_integer(1.into());
                            if w >= 0 {
                                for _ in 0..w {
                                    cw *= &c;
                                }
                            } else {
                                for _ in 0..(-w) {
                                    cw /= &c;
                                }
                            }
                            let cw_node = ctx.add(cas_ast::Expr::Number(cw));
                            let sr = if r == 1 {
                                s
                            } else {
                                let rn = ctx.num(r as i64);
                                ctx.add(cas_ast::Expr::Pow(s, rn))
                            };
                            return Some(ctx.add(cas_ast::Expr::Mul(cw_node, sr)));
                        }
                    }
                }
            }
            // Integer-exponent powers (any sign — the rational-function
            // converter supports negatives) stay structural; anything else
            // is a single opaque atom.
            let is_int_exp = exp_number(ctx, exp).is_some_and(|n| n.is_integer());
            if is_int_exp {
                let nb = atomize_non_poly(ctx, base, atoms, expfold, powfold, atom_base)?;
                if nb == base {
                    return Some(expr);
                }
                return Some(ctx.add(cas_ast::Expr::Pow(nb, exp)));
            }
            Some(atom_for(ctx, atoms, expr, atom_base))
        }
        cas_ast::Expr::Function(f, args) => {
            // Denesting: sqrt(a + b·√n) with rational a,b,n denests exactly
            // iff a² − b²·n is a perfect square d², into
            // √((a+d)/2) + sign(b)·√((a−d)/2) — both plain rational sqrts
            // that re-enter this atomizer and share the √n-family atoms.
            // Closes surd-root verification residues like
            // sqrt((3−√5)/2) + (1−√5)/2 (the (1−√5)/2 root of √(x+1) = −x).
            if ctx.is_builtin(f, cas_ast::BuiltinFn::Sqrt)
                && args.len() == 1
                && cas_ast::collect_variables(ctx, args[0]).is_empty()
            {
                if let Some((a, b, Some(rad))) =
                    crate::root_forms::as_linear_surd_expr(ctx, args[0])
                {
                    use num_traits::{Signed, Zero};
                    if !b.is_zero() {
                        if let Some(n) = crate::numeric_eval::as_rational_const(ctx, rad) {
                            let disc = &a * &a - &b * &b * &n;
                            if !disc.is_negative() {
                                if let Some(d) = crate::const_sign::exact_nth_root(&disc, 2) {
                                    let half = num_rational::BigRational::new(1.into(), 2.into());
                                    let p1 = (&a + &d) * &half;
                                    let p2 = (&a - &d) * &half;
                                    if !p1.is_negative() && !p2.is_negative() {
                                        let p1n = ctx.add(cas_ast::Expr::Number(p1));
                                        let p2n = ctx.add(cas_ast::Expr::Number(p2));
                                        let s1 =
                                            ctx.call_builtin(cas_ast::BuiltinFn::Sqrt, vec![p1n]);
                                        let s2 =
                                            ctx.call_builtin(cas_ast::BuiltinFn::Sqrt, vec![p2n]);
                                        let comb = if b.is_positive() {
                                            ctx.add(cas_ast::Expr::Add(s1, s2))
                                        } else {
                                            ctx.add(cas_ast::Expr::Sub(s1, s2))
                                        };
                                        return atomize_non_poly(
                                            ctx, comb, atoms, expfold, powfold, atom_base,
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // sqrt(c) over a RATIONAL constant: same canonical atom as
            // c^(1/2) (the constant-radical fold above).
            if ctx.is_builtin(f, cas_ast::BuiltinFn::Sqrt) && args.len() == 1 {
                if let cas_ast::Expr::Number(c) = ctx.get(args[0]).clone() {
                    use num_traits::Signed;
                    if !c.is_negative() {
                        let half = num_rational::BigRational::new(1.into(), 2.into());
                        let c_node = ctx.add(cas_ast::Expr::Number(c));
                        let half_node = ctx.add(cas_ast::Expr::Number(half));
                        let key = ctx.add(cas_ast::Expr::Pow(c_node, half_node));
                        return Some(atom_for(ctx, atoms, key, atom_base));
                    }
                }
            }
            // sqrt(B) with a folded base: ↦ s^(d/2) (d is even by lcm).
            if ctx.is_builtin(f, cas_ast::BuiltinFn::Sqrt) && args.len() == 1 {
                if let Some((d, s)) = powfold.get(&args[0]) {
                    if d % 2 == 0 {
                        let s = *s;
                        let half = ctx.num((*d / 2) as i64);
                        return Some(ctx.add(cas_ast::Expr::Pow(s, half)));
                    }
                }
            }
            // Normalize arguments recursively so structurally-different
            // spellings of the same argument intern to the same atom key.
            let mut nargs = Vec::with_capacity(args.len());
            let mut changed = false;
            for a in &args {
                let na = atomize_non_poly(ctx, *a, atoms, expfold, powfold, atom_base)?;
                changed |= na != *a;
                nargs.push(na);
            }
            let key = if changed {
                ctx.add(cas_ast::Expr::Function(f, nargs.clone()))
            } else {
                expr
            };
            // ln(e^w) = w: unconditional real identity (ln∘exp = id on all
            // of ℝ) — `sin(ln(e^x))` must share its atom with `sin(x)`.
            // And ln(e) = 1 (the pipeline sometimes leaves `x·ln(e)`).
            if ctx.is_builtin(f, cas_ast::BuiltinFn::Ln) && nargs.len() == 1 {
                if matches!(
                    ctx.get(nargs[0]),
                    cas_ast::Expr::Constant(cas_ast::Constant::E)
                ) {
                    return Some(ctx.num(1));
                }
                if let cas_ast::Expr::Pow(b, w) = ctx.get(nargs[0]).clone() {
                    if matches!(ctx.get(b), cas_ast::Expr::Constant(cas_ast::Constant::E)) {
                        return Some(w);
                    }
                }
                if let cas_ast::Expr::Function(g, gargs) = ctx.get(nargs[0]).clone() {
                    if ctx.is_builtin(g, cas_ast::BuiltinFn::Exp) && gargs.len() == 1 {
                        return Some(gargs[0]);
                    }
                }
            }
            // Log-of-rational fold: ln(c) for POSITIVE rational c becomes the
            // exact integer combination of ln(prime) atoms via trial-division
            // factorization (ln(8) = 3·ln(2); ln(9/4) = 2·ln(3) − 2·ln(2)).
            // The interval oracle can bound but never PROVE zero for
            // transcendental residues like 3·ln(2) − ln(8); this fold makes
            // them polynomial identities between prime-log atoms. Bounded:
            // primes ≤ 10⁴, leftover cofactors get their own atom.
            if ctx.is_builtin(f, cas_ast::BuiltinFn::Ln) && nargs.len() == 1 {
                if let cas_ast::Expr::Number(c) = ctx.get(nargs[0]).clone() {
                    use num_traits::{One, Signed, Zero};
                    if c.is_positive() && !c.is_one() {
                        let mut terms: Vec<(i64, num_bigint::BigInt)> = Vec::new();
                        let push_factors =
                            |n: &num_bigint::BigInt, sign: i64, terms: &mut Vec<(i64, num_bigint::BigInt)>| {
                                let mut m = n.clone();
                                let mut p = num_bigint::BigInt::from(2);
                                let limit = num_bigint::BigInt::from(10_000);
                                while &p * &p <= m && p <= limit {
                                    let mut e = 0i64;
                                    while (&m % &p).is_zero() {
                                        m /= &p;
                                        e += 1;
                                    }
                                    if e > 0 {
                                        terms.push((sign * e, p.clone()));
                                    }
                                    p += 1;
                                }
                                if m > num_bigint::BigInt::one() {
                                    terms.push((sign, m));
                                }
                            };
                        push_factors(c.numer(), 1, &mut terms);
                        push_factors(c.denom(), -1, &mut terms);
                        let mut acc: Option<ExprId> = None;
                        for (e, prime) in terms {
                            let p_node = ctx.add(cas_ast::Expr::Number(
                                num_rational::BigRational::from_integer(prime),
                            ));
                            let ln_p = ctx.call_builtin(cas_ast::BuiltinFn::Ln, vec![p_node]);
                            let atom = atom_for(ctx, atoms, ln_p, atom_base);
                            let coef = ctx.num(e);
                            let term = ctx.add(cas_ast::Expr::Mul(coef, atom));
                            acc = Some(match acc {
                                None => term,
                                Some(prev) => ctx.add(cas_ast::Expr::Add(prev, term)),
                            });
                        }
                        if let Some(total) = acc {
                            return Some(total);
                        }
                    }
                }
            }
            // Exponential lowering: with an active e-fold base g and a
            // normalized argument k·g, the DEFINITIONS sinh(k·g) =
            // (u^k − u^(−k))/2 and cosh(k·g) = (u^k + u^(−k))/2 (u = e^g)
            // are faithful identities; the rational converter handles the
            // negative powers. Closes exp×hyperbolic residues like
            // e^(2x) − 2·sinh(x)·e^x − 1.
            if let Some((g, u)) = expfold {
                if nargs.len() == 1 {
                    let is_sinh = ctx.is_builtin(f, cas_ast::BuiltinFn::Sinh);
                    let is_cosh = ctx.is_builtin(f, cas_ast::BuiltinFn::Cosh);
                    if is_sinh || is_cosh {
                        if let Some(k) = exp_integer_multiple_of(ctx, nargs[0], g) {
                            let kp = ctx.num(k);
                            let kn = ctx.num(-k);
                            let up = ctx.add(cas_ast::Expr::Pow(u, kp));
                            let un = ctx.add(cas_ast::Expr::Pow(u, kn));
                            let comb = if is_sinh {
                                ctx.add(cas_ast::Expr::Sub(up, un))
                            } else {
                                ctx.add(cas_ast::Expr::Add(up, un))
                            };
                            let two = ctx.num(2);
                            return Some(ctx.add(cas_ast::Expr::Div(comb, two)));
                        }
                    }
                }
            }
            // Quotient-family fold: tan = sin/cos, cot = cos/sin,
            // sec = 1/cos, csc = 1/sin, tanh = sinh/cosh. Faithful on the
            // expression's own domain (tan defined ⟹ cos ≠ 0), and it turns
            // cosh²·tanh² = sinh² into a polynomial identity between atoms.
            use cas_ast::BuiltinFn;
            let ratio = if ctx.is_builtin(f, BuiltinFn::Tan) {
                Some((Some(BuiltinFn::Sin), BuiltinFn::Cos))
            } else if ctx.is_builtin(f, BuiltinFn::Cot) {
                Some((Some(BuiltinFn::Cos), BuiltinFn::Sin))
            } else if ctx.is_builtin(f, BuiltinFn::Tanh) {
                Some((Some(BuiltinFn::Sinh), BuiltinFn::Cosh))
            } else if ctx.is_builtin(f, BuiltinFn::Sec) {
                Some((None, BuiltinFn::Cos))
            } else if ctx.is_builtin(f, BuiltinFn::Csc) {
                Some((None, BuiltinFn::Sin))
            } else {
                None
            };
            if let Some((num_b, den_b)) = ratio {
                let den_node = ctx.call_builtin(den_b, nargs.clone());
                let den_atom = atom_for(ctx, atoms, den_node, atom_base);
                let num_atom = match num_b {
                    Some(nb) => {
                        let num_node = ctx.call_builtin(nb, nargs);
                        atom_for(ctx, atoms, num_node, atom_base)
                    }
                    None => ctx.num(1),
                };
                return Some(ctx.add(cas_ast::Expr::Div(num_atom, den_atom)));
            }
            Some(atom_for(ctx, atoms, key, atom_base))
        }
        cas_ast::Expr::Hold(_) => Some(atom_for(ctx, atoms, expr, atom_base)),
        // Matrices / session refs: never atomize, never confirm.
        _ => None,
    }
}

/// Relation between two polynomial expressions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignRelation {
    /// `a == b`
    Same,
    /// `a == -b`
    Negated,
}

/// Compare two expressions as polynomials (ignoring AST structure/order).
///
/// Returns `false` if conversion fails for either side.
pub fn poly_eq(ctx: &Context, a: ExprId, b: ExprId) -> bool {
    let budget = compare_budget();

    let pa = match multipoly_from_expr(ctx, a, &budget) {
        Ok(p) => p,
        Err(_) => return false,
    };
    let pb = match multipoly_from_expr(ctx, b, &budget) {
        Ok(p) => p,
        Err(_) => return false,
    };

    pa == pb
}

/// Compare two expressions to detect if they are equal or negated.
///
/// Returns:
/// - `Some(SignRelation::Same)` if `a == b`
/// - `Some(SignRelation::Negated)` if `a == -b`
/// - `None` otherwise
pub(crate) fn poly_relation(ctx: &Context, a: ExprId, b: ExprId) -> Option<SignRelation> {
    let budget = compare_budget();

    let pa = multipoly_from_expr(ctx, a, &budget).ok()?;
    let pb = multipoly_from_expr(ctx, b, &budget).ok()?;

    if pa == pb {
        return Some(SignRelation::Same);
    }

    if pa == pb.neg() {
        return Some(SignRelation::Negated);
    }

    None
}

/// True when `a == λ·b` for some rational `λ < 0`, i.e. `a` and `b` are non-zero
/// polynomials pointing in opposite directions.
///
/// Then `a > 0 ∧ b > 0` is unsatisfiable (`a` and `b` always have opposite signs),
/// which is how `solve(log(b, -k·x) = log(b, x) + …)` collapses to "No solution":
/// its recorded domain conditions `Positive(-k·x) ∧ Positive(x)` are contradictory.
/// This generalises [`poly_relation`]'s `Negated` (`λ = -1`) case to any negative
/// rational multiple (e.g. `-8·x` vs `x`). Returns `false` (no claim) when either
/// side is not polynomial-convertible or is zero.
pub fn poly_negatively_proportional(ctx: &Context, a: ExprId, b: ExprId) -> bool {
    use num_traits::Signed;

    let budget = compare_budget();
    let (Ok(pa), Ok(pb)) = (
        multipoly_from_expr(ctx, a, &budget),
        multipoly_from_expr(ctx, b, &budget),
    ) else {
        return false;
    };
    if pa.is_zero() || pb.is_zero() {
        return false;
    }
    // For proportional polynomials the leading terms (same monomial order)
    // correspond, so λ = (lead coeff a) / (lead coeff b) is the only candidate.
    let (Some(ta), Some(tb)) = (pa.leading_term_lex(), pb.leading_term_lex()) else {
        return false;
    };
    let lambda = &ta.0 / &tb.0;
    if !lambda.is_negative() {
        return false;
    }
    // Exact proof: a - λ·b == 0  (and a wrong λ from a non-proportional pair makes
    // this non-zero, so the test never yields a false positive).
    matches!(pa.sub(&pb.mul_scalar(&lambda)), Ok(diff) if diff.is_zero())
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    #[test]
    fn poly_eq_matches_commutative_forms() {
        let mut ctx = Context::new();
        let a = parse("x + y", &mut ctx).expect("parse a");
        let b = parse("y + x", &mut ctx).expect("parse b");
        assert!(poly_eq(&ctx, a, b));
    }

    #[test]
    fn poly_relation_detects_negation() {
        let mut ctx = Context::new();
        let a = parse("x - y", &mut ctx).expect("parse a");
        let b = parse("y - x", &mut ctx).expect("parse b");
        assert_eq!(poly_relation(&ctx, a, b), Some(SignRelation::Negated));
    }

    #[test]
    fn poly_negatively_proportional_detects_negative_multiples() {
        let mut ctx = Context::new();
        // -8x = -8·x  (the log(2,-8x)=log(2,x)+k domain pair)
        let a = parse("-8*x", &mut ctx).expect("a");
        let b = parse("x", &mut ctx).expect("b");
        assert!(poly_negatively_proportional(&ctx, a, b));
        // exact negation (λ = -1) is still covered
        let c = parse("1 - x", &mut ctx).expect("c");
        let d = parse("x - 1", &mut ctx).expect("d");
        assert!(poly_negatively_proportional(&ctx, c, d));
        // a multivariable negative multiple
        let e = parse("-3*x - 3*y", &mut ctx).expect("e");
        let f = parse("x + y", &mut ctx).expect("f");
        assert!(poly_negatively_proportional(&ctx, e, f));
    }

    #[test]
    fn poly_negatively_proportional_rejects_compatible_and_unrelated() {
        let mut ctx = Context::new();
        // positive multiple: x>0 and 2x>0 are compatible, NOT contradictory
        let a = parse("2*x", &mut ctx).expect("a");
        let b = parse("x", &mut ctx).expect("b");
        assert!(!poly_negatively_proportional(&ctx, a, b));
        // unrelated variables are not proportional
        let c = parse("x", &mut ctx).expect("c");
        let d = parse("y", &mut ctx).expect("d");
        assert!(!poly_negatively_proportional(&ctx, c, d));
        // not proportional at all
        let e = parse("x + 1", &mut ctx).expect("e");
        let f = parse("x", &mut ctx).expect("f");
        assert!(!poly_negatively_proportional(&ctx, e, f));
    }

    #[test]
    fn opaque_zero_gate_survives_adversarial_internal_names() {
        // Clase L15: los átomos sintéticos (__polyzero_*) no comprobaban
        // colisión con variables ya presentes. Un árbol que CONTIENE esos
        // nombres fusionaba átomo y variable y el gate declaraba cero
        // diferencias que no lo son.
        let mut ctx = Context::new();
        let e1 = cas_parser::parse("sin(x) - __polyzero_atom_0", &mut ctx).expect("e1");
        assert!(
            !poly_is_zero_opaque(&mut ctx, e1),
            "sin(x) - __polyzero_atom_0 NO es identicamente cero"
        );
        let e2 = cas_parser::parse("e^x - __polyzero_expatom", &mut ctx).expect("e2");
        assert!(
            !poly_is_zero_opaque(&mut ctx, e2),
            "e^x - __polyzero_expatom NO es identicamente cero"
        );
        let e3 = cas_parser::parse("x^(1/2) - __polyzero_root_0", &mut ctx).expect("e3");
        assert!(
            !poly_is_zero_opaque(&mut ctx, e3),
            "sqrt(x) - __polyzero_root_0 NO es identicamente cero"
        );
        // y los ceros de verdad siguen cerrando aunque el nombre este presente
        let e4 = cas_parser::parse("(sin(x)^2 + cos(x)^2 - 1) * __polyzero_atom_0", &mut ctx)
            .expect("e4");
        assert!(poly_is_zero_opaque(&mut ctx, e4), "pitagorica escalada = 0");
    }
}
