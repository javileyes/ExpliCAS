# Engine Improvement Scorecard

- Generated: 2026-06-28T15:28:07.398870+00:00
- Git branch: main
- Git commit: `c1c8434089e64d7405555d6f1a69e8c3c46edfc2`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=824.41ms avg_case_ms=8.24 simplify=238.28ms avg_simplify_ms=2.38, sum total=200 failed=0 elapsed=742.97ms avg_case_ms=3.71 simplify=256.34ms avg_simplify_ms=1.28, product total=100 failed=0 elapsed=497.65ms avg_case_ms=4.98 simplify=148.19ms avg_simplify_ms=1.48, difference total=50 failed=0 elapsed=339.61ms avg_case_ms=6.79 simplify=110.65ms avg_simplify_ms=2.21
- Engine hotspots: sum simplify=256.34ms avg_simplify_ms=1.28 wall=742.97ms, shifted_quotient simplify=238.28ms avg_simplify_ms=2.38 wall=824.41ms, product simplify=148.19ms avg_simplify_ms=1.48 wall=497.65ms, difference simplify=110.65ms avg_simplify_ms=2.21 wall=339.61ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=824.41ms avg_case_ms=8.24 avg_simplify_ms=2.38, sum@0+100 failed=0 elapsed=530.40ms avg_case_ms=5.30 avg_simplify_ms=1.75, product@0+100 failed=0 elapsed=497.65ms avg_case_ms=4.98 avg_simplify_ms=1.48, difference@0+50 failed=0 elapsed=339.61ms avg_case_ms=6.79 avg_simplify_ms=2.21, sum@700+100 failed=0 elapsed=212.57ms avg_case_ms=2.13 avg_simplify_ms=0.81
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.09ms median_wire=13.17ms median_wall=50.17ms, sum@0+100 #173 sum runs=3 median_simplify=11.73ms median_wire=11.79ms median_wall=44.93ms, difference@0+50 #174 difference runs=3 median_simplify=11.76ms median_wire=11.82ms median_wall=45.08ms, product@0+100 #175 product runs=3 median_simplify=11.77ms median_wire=11.83ms median_wall=45.13ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.85ms median_wire=10.93ms median_wall=40.92ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.41s | passed=450 failed=0 total=450 avg_case=5.356ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.48s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.07s | passed=1 failed=0 |
