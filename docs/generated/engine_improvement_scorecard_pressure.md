# Engine Improvement Scorecard

- Generated: 2026-06-27T22:48:42.815597+00:00
- Git branch: main
- Git commit: `5529f20b13e30854b9b7c170d03ff591a477bba0`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=353

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=832.48ms avg_case_ms=8.32 simplify=240.82ms avg_simplify_ms=2.41, sum total=200 failed=0 elapsed=748.71ms avg_case_ms=3.74 simplify=262.06ms avg_simplify_ms=1.31, product total=100 failed=0 elapsed=513.87ms avg_case_ms=5.14 simplify=153.67ms avg_simplify_ms=1.54, difference total=50 failed=0 elapsed=349.99ms avg_case_ms=7.00 simplify=114.37ms avg_simplify_ms=2.29
- Engine hotspots: sum simplify=262.06ms avg_simplify_ms=1.31 wall=748.71ms, shifted_quotient simplify=240.82ms avg_simplify_ms=2.41 wall=832.48ms, product simplify=153.67ms avg_simplify_ms=1.54 wall=513.87ms, difference simplify=114.37ms avg_simplify_ms=2.29 wall=349.99ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=832.48ms avg_case_ms=8.32 avg_simplify_ms=2.41, sum@0+100 failed=0 elapsed=539.36ms avg_case_ms=5.39 avg_simplify_ms=1.79, product@0+100 failed=0 elapsed=513.87ms avg_case_ms=5.14 avg_simplify_ms=1.54, difference@0+50 failed=0 elapsed=349.99ms avg_case_ms=7.00 avg_simplify_ms=2.29, sum@700+100 failed=0 elapsed=209.35ms avg_case_ms=2.09 avg_simplify_ms=0.83
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.32ms median_wire=13.39ms median_wall=50.62ms, product@0+100 #175 product runs=3 median_simplify=12.31ms median_wire=12.37ms median_wall=47.00ms, difference@0+50 #174 difference runs=3 median_simplify=12.26ms median_wire=12.32ms median_wall=46.52ms, sum@0+100 #173 sum runs=3 median_simplify=12.10ms median_wire=12.15ms median_wall=46.88ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=11.25ms median_wire=11.33ms median_wall=42.76ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.45s | passed=450 failed=0 total=450 avg_case=5.444ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.54s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.13s | passed=1 failed=0 |
