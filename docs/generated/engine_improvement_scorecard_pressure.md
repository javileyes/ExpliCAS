# Engine Improvement Scorecard

- Generated: 2026-06-11T15:35:50.048419+00:00
- Git branch: main
- Git commit: `ee4b4f10df4c63c27e2d9b0a031df06d168b18f6`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=344

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=784.23ms avg_case_ms=7.84 simplify=223.59ms avg_simplify_ms=2.24, sum total=200 failed=0 elapsed=700.87ms avg_case_ms=3.50 simplify=235.57ms avg_simplify_ms=1.18, product total=100 failed=0 elapsed=476.76ms avg_case_ms=4.77 simplify=136.96ms avg_simplify_ms=1.37, difference total=50 failed=0 elapsed=322.82ms avg_case_ms=6.46 simplify=102.75ms avg_simplify_ms=2.05
- Engine hotspots: sum simplify=235.57ms avg_simplify_ms=1.18 wall=700.87ms, shifted_quotient simplify=223.59ms avg_simplify_ms=2.24 wall=784.23ms, product simplify=136.96ms avg_simplify_ms=1.37 wall=476.76ms, difference simplify=102.75ms avg_simplify_ms=2.05 wall=322.82ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=784.23ms avg_case_ms=7.84 avg_simplify_ms=2.24, sum@0+100 failed=0 elapsed=512.48ms avg_case_ms=5.12 avg_simplify_ms=1.66, product@0+100 failed=0 elapsed=476.76ms avg_case_ms=4.77 avg_simplify_ms=1.37, difference@0+50 failed=0 elapsed=322.82ms avg_case_ms=6.46 avg_simplify_ms=2.05, sum@700+100 failed=0 elapsed=188.39ms avg_case_ms=1.88 avg_simplify_ms=0.70
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=14.17ms median_wire=14.24ms median_wall=64.89ms, sum@0+100 #173 sum runs=3 median_simplify=11.55ms median_wire=11.60ms median_wall=43.89ms, difference@0+50 #174 difference runs=3 median_simplify=11.52ms median_wire=11.58ms median_wall=43.53ms, product@0+100 #175 product runs=3 median_simplify=11.58ms median_wire=11.62ms median_wall=46.20ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.66ms median_wire=10.73ms median_wall=39.84ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.29s | passed=450 failed=0 total=450 avg_case=5.089ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.82s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 17.11s | passed=1 failed=0 |
