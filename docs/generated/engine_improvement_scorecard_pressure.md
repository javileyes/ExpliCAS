# Engine Improvement Scorecard

- Generated: 2026-07-30T17:46:44.419290+00:00
- Git branch: main
- Git commit: `11614d980110ecc1cd50c9224ad1c661f2e9c323`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=973.28ms avg_case_ms=9.73 simplify=272.51ms avg_simplify_ms=2.73, sum total=200 failed=0 elapsed=838.65ms avg_case_ms=4.19 simplify=268.52ms avg_simplify_ms=1.34, product total=100 failed=0 elapsed=588.66ms avg_case_ms=5.89 simplify=169.45ms avg_simplify_ms=1.69, difference total=50 failed=0 elapsed=382.27ms avg_case_ms=7.65 simplify=112.26ms avg_simplify_ms=2.25
- Engine hotspots: shifted_quotient simplify=272.51ms avg_simplify_ms=2.73 wall=973.28ms, sum simplify=268.52ms avg_simplify_ms=1.34 wall=838.65ms, product simplify=169.45ms avg_simplify_ms=1.69 wall=588.66ms, difference simplify=112.26ms avg_simplify_ms=2.25 wall=382.27ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=973.28ms avg_case_ms=9.73 avg_simplify_ms=2.73, sum@0+100 failed=0 elapsed=611.69ms avg_case_ms=6.12 avg_simplify_ms=1.87, product@0+100 failed=0 elapsed=588.66ms avg_case_ms=5.89 avg_simplify_ms=1.69, difference@0+50 failed=0 elapsed=382.27ms avg_case_ms=7.65 avg_simplify_ms=2.25, sum@700+100 failed=0 elapsed=226.96ms avg_case_ms=2.27 avg_simplify_ms=0.81
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.55ms median_wire=16.61ms median_wall=63.40ms, sum@0+100 #173 sum runs=3 median_simplify=14.91ms median_wire=14.96ms median_wall=57.19ms, product@0+100 #175 product runs=3 median_simplify=14.95ms median_wire=15.00ms median_wall=56.73ms, difference@0+50 #174 difference runs=3 median_simplify=15.04ms median_wire=15.08ms median_wall=57.16ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.70ms median_wire=12.77ms median_wall=49.03ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.78s | passed=450 failed=0 total=450 avg_case=6.178ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.14s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.74s | passed=1 failed=0 |
