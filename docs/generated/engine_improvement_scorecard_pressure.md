# Engine Improvement Scorecard

- Generated: 2026-08-01T22:01:01.437003+00:00
- Git branch: main
- Git commit: `c332e5cee5552c98618fe948b83982c9f556b369`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=391

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=977.02ms avg_case_ms=9.77 simplify=273.79ms avg_simplify_ms=2.74, sum total=200 failed=0 elapsed=855.59ms avg_case_ms=4.28 simplify=274.36ms avg_simplify_ms=1.37, product total=100 failed=0 elapsed=618.19ms avg_case_ms=6.18 simplify=177.41ms avg_simplify_ms=1.77, difference total=50 failed=0 elapsed=400.19ms avg_case_ms=8.00 simplify=117.78ms avg_simplify_ms=2.36
- Engine hotspots: sum simplify=274.36ms avg_simplify_ms=1.37 wall=855.59ms, shifted_quotient simplify=273.79ms avg_simplify_ms=2.74 wall=977.02ms, product simplify=177.41ms avg_simplify_ms=1.77 wall=618.19ms, difference simplify=117.78ms avg_simplify_ms=2.36 wall=400.19ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=977.02ms avg_case_ms=9.77 avg_simplify_ms=2.74, product@0+100 failed=0 elapsed=618.19ms avg_case_ms=6.18 avg_simplify_ms=1.77, sum@0+100 failed=0 elapsed=617.65ms avg_case_ms=6.18 avg_simplify_ms=1.89, difference@0+50 failed=0 elapsed=400.19ms avg_case_ms=8.00 avg_simplify_ms=2.36, sum@700+100 failed=0 elapsed=237.94ms avg_case_ms=2.38 avg_simplify_ms=0.85
- Steady-state engine reruns: product@0+100 #175 product runs=3 median_simplify=14.91ms median_wire=14.96ms median_wall=57.44ms, shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.79ms median_wire=16.86ms median_wall=65.77ms, sum@0+100 #173 sum runs=3 median_simplify=16.34ms median_wire=16.39ms median_wall=63.10ms, difference@0+50 #174 difference runs=3 median_simplify=15.23ms median_wire=15.27ms median_wall=58.04ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.90ms median_wire=12.97ms median_wall=49.11ms
- Steady-state dominant expressions: product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.85s | passed=450 failed=0 total=450 avg_case=6.333ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.30s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.73s | passed=1 failed=0 |
