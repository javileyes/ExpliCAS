# Engine Improvement Scorecard

- Generated: 2026-06-26T09:30:40.540775+00:00
- Git branch: main
- Git commit: `f2cc06a54f64d8828ebdccb4969751b2194c2790`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=782.67ms avg_case_ms=7.83 simplify=223.63ms avg_simplify_ms=2.24, sum total=200 failed=0 elapsed=703.61ms avg_case_ms=3.52 simplify=241.62ms avg_simplify_ms=1.21, product total=100 failed=0 elapsed=486.88ms avg_case_ms=4.87 simplify=142.35ms avg_simplify_ms=1.42, difference total=50 failed=0 elapsed=326.48ms avg_case_ms=6.53 simplify=105.03ms avg_simplify_ms=2.10
- Engine hotspots: sum simplify=241.62ms avg_simplify_ms=1.21 wall=703.61ms, shifted_quotient simplify=223.63ms avg_simplify_ms=2.24 wall=782.67ms, product simplify=142.35ms avg_simplify_ms=1.42 wall=486.88ms, difference simplify=105.03ms avg_simplify_ms=2.10 wall=326.48ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=782.67ms avg_case_ms=7.83 avg_simplify_ms=2.24, sum@0+100 failed=0 elapsed=506.59ms avg_case_ms=5.07 avg_simplify_ms=1.66, product@0+100 failed=0 elapsed=486.88ms avg_case_ms=4.87 avg_simplify_ms=1.42, difference@0+50 failed=0 elapsed=326.48ms avg_case_ms=6.53 avg_simplify_ms=2.10, sum@700+100 failed=0 elapsed=197.02ms avg_case_ms=1.97 avg_simplify_ms=0.76
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.92ms median_wire=12.99ms median_wall=49.22ms, product@0+100 #175 product runs=3 median_simplify=11.63ms median_wire=11.68ms median_wall=44.25ms, sum@0+100 #173 sum runs=3 median_simplify=11.74ms median_wire=11.79ms median_wall=44.70ms, difference@0+50 #174 difference runs=3 median_simplify=11.60ms median_wire=11.65ms median_wall=44.13ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.73ms median_wire=10.80ms median_wall=40.35ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.30s | passed=450 failed=0 total=450 avg_case=5.111ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.45s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.05s | passed=1 failed=0 |
