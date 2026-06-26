# Engine Improvement Scorecard

- Generated: 2026-06-26T22:55:06.136892+00:00
- Git branch: main
- Git commit: `c256fcc0ccf1aea3b9f135c8bf2ad786ef146932`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=793.90ms avg_case_ms=7.94 simplify=227.56ms avg_simplify_ms=2.28, sum total=200 failed=0 elapsed=716.44ms avg_case_ms=3.58 simplify=248.20ms avg_simplify_ms=1.24, product total=100 failed=0 elapsed=510.41ms avg_case_ms=5.10 simplify=151.30ms avg_simplify_ms=1.51, difference total=50 failed=0 elapsed=341.60ms avg_case_ms=6.83 simplify=110.43ms avg_simplify_ms=2.21
- Engine hotspots: sum simplify=248.20ms avg_simplify_ms=1.24 wall=716.44ms, shifted_quotient simplify=227.56ms avg_simplify_ms=2.28 wall=793.90ms, product simplify=151.30ms avg_simplify_ms=1.51 wall=510.41ms, difference simplify=110.43ms avg_simplify_ms=2.21 wall=341.60ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=793.90ms avg_case_ms=7.94 avg_simplify_ms=2.28, sum@0+100 failed=0 elapsed=514.40ms avg_case_ms=5.14 avg_simplify_ms=1.69, product@0+100 failed=0 elapsed=510.41ms avg_case_ms=5.10 avg_simplify_ms=1.51, difference@0+50 failed=0 elapsed=341.60ms avg_case_ms=6.83 avg_simplify_ms=2.21, sum@700+100 failed=0 elapsed=202.04ms avg_case_ms=2.02 avg_simplify_ms=0.79
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.08ms median_wire=13.15ms median_wall=50.05ms, product@0+100 #175 product runs=3 median_simplify=11.75ms median_wire=11.80ms median_wall=45.16ms, difference@0+50 #174 difference runs=3 median_simplify=11.84ms median_wire=11.89ms median_wall=45.16ms, sum@0+100 #173 sum runs=3 median_simplify=11.91ms median_wire=11.96ms median_wall=44.48ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.69ms median_wire=10.77ms median_wall=40.74ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.36s | passed=450 failed=0 total=450 avg_case=5.244ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.49s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.06s | passed=1 failed=0 |
