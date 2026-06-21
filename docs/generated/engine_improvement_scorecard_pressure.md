# Engine Improvement Scorecard

- Generated: 2026-06-21T15:31:37.148468+00:00
- Git branch: main
- Git commit: `834d5cd4905002aa1dbb724f97856d084ad1ddcb`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=813.13ms avg_case_ms=8.13 simplify=234.50ms avg_simplify_ms=2.34, sum total=200 failed=0 elapsed=737.94ms avg_case_ms=3.69 simplify=254.55ms avg_simplify_ms=1.27, product total=100 failed=0 elapsed=497.69ms avg_case_ms=4.98 simplify=144.51ms avg_simplify_ms=1.45, difference total=50 failed=0 elapsed=339.03ms avg_case_ms=6.78 simplify=109.41ms avg_simplify_ms=2.19
- Engine hotspots: sum simplify=254.55ms avg_simplify_ms=1.27 wall=737.94ms, shifted_quotient simplify=234.50ms avg_simplify_ms=2.34 wall=813.13ms, product simplify=144.51ms avg_simplify_ms=1.45 wall=497.69ms, difference simplify=109.41ms avg_simplify_ms=2.19 wall=339.03ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=813.13ms avg_case_ms=8.13 avg_simplify_ms=2.34, sum@0+100 failed=0 elapsed=524.68ms avg_case_ms=5.25 avg_simplify_ms=1.73, product@0+100 failed=0 elapsed=497.69ms avg_case_ms=4.98 avg_simplify_ms=1.45, difference@0+50 failed=0 elapsed=339.03ms avg_case_ms=6.78 avg_simplify_ms=2.19, sum@700+100 failed=0 elapsed=213.26ms avg_case_ms=2.13 avg_simplify_ms=0.81
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.41ms median_wire=13.48ms median_wall=51.13ms, sum@0+100 #173 sum runs=3 median_simplify=12.10ms median_wire=12.16ms median_wall=45.31ms, difference@0+50 #174 difference runs=3 median_simplify=11.75ms median_wire=11.80ms median_wall=45.04ms, product@0+100 #175 product runs=3 median_simplify=11.74ms median_wire=11.79ms median_wall=44.54ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.99ms median_wire=11.07ms median_wall=41.45ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.39s | passed=450 failed=0 total=450 avg_case=5.311ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.51s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.07s | passed=1 failed=0 |
