# Engine Improvement Scorecard

- Generated: 2026-07-29T16:45:49.024169+00:00
- Git branch: main
- Git commit: `9aabf418f638166912836cd5b491edd6428b353f`
- Profile: `pressure`

## Generated Discovery Ledger

- Purpose: keep failed generated candidates visible without promoting them to live corpus.
- Observe-only discoveries: total=10
- By area: calculus / integration:4, calculus / runtime:3, calculus / differentiation:2, calculus / robustness:1
- Recent 1: `calculus / integration` - 2026-06-08 - Discovery observe-only: polynomial cosecant/cotangent source-return still emits depth pressure
- Recent 2: `calculus / differentiation` - 2026-06-06 - Observe-only discovery: exact-square atanh scaled-root runtime is not caused by the global empty-domain check
- Recent 3: `calculus / differentiation` - 2026-06-06 - Observe-only discovery: exact-square inverse-root diff runtime is not fixed by raw target preservation

## Calculus Support Matrix Signal

- Dimension: public calculus behavior, support-matrix coverage, result simplification, domain conditions, trace quality, presentation, and verification residuals.
- Interpretation: matrix-oriented calculus lanes; classify failures by command, family, argument regime, domain regime, trace regime, presentation regime, or reusable pre-calculus dependency before adding isolated cases.
- Matrix axes: command, family, argument regime, domain regime, trace regime, presentation regime, and residual verification.
- `diff_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=263
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=380

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=976.99ms avg_case_ms=9.77 simplify=274.74ms avg_simplify_ms=2.75, sum total=200 failed=0 elapsed=857.39ms avg_case_ms=4.29 simplify=277.52ms avg_simplify_ms=1.39, product total=100 failed=0 elapsed=599.59ms avg_case_ms=6.00 simplify=172.01ms avg_simplify_ms=1.72, difference total=50 failed=0 elapsed=394.15ms avg_case_ms=7.88 simplify=119.11ms avg_simplify_ms=2.38
- Engine hotspots: sum simplify=277.52ms avg_simplify_ms=1.39 wall=857.39ms, shifted_quotient simplify=274.74ms avg_simplify_ms=2.75 wall=976.99ms, product simplify=172.01ms avg_simplify_ms=1.72 wall=599.59ms, difference simplify=119.11ms avg_simplify_ms=2.38 wall=394.15ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=976.99ms avg_case_ms=9.77 avg_simplify_ms=2.75, sum@0+100 failed=0 elapsed=625.74ms avg_case_ms=6.26 avg_simplify_ms=1.95, product@0+100 failed=0 elapsed=599.59ms avg_case_ms=6.00 avg_simplify_ms=1.72, difference@0+50 failed=0 elapsed=394.15ms avg_case_ms=7.88 avg_simplify_ms=2.38, sum@700+100 failed=0 elapsed=231.65ms avg_case_ms=2.32 avg_simplify_ms=0.82
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.94ms median_wire=17.00ms median_wall=64.43ms, product@0+100 #175 product runs=3 median_simplify=15.19ms median_wire=15.23ms median_wall=58.29ms, sum@0+100 #173 sum runs=3 median_simplify=14.99ms median_wire=15.04ms median_wall=57.96ms, difference@0+50 #174 difference runs=3 median_simplify=15.12ms median_wire=15.17ms median_wall=58.57ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=13.15ms median_wire=13.22ms median_wall=49.55ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.83s | passed=450 failed=0 total=450 avg_case=6.289ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.35s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.85s | passed=1 failed=0 |
