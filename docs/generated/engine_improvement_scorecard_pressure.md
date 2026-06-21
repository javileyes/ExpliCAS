# Engine Improvement Scorecard

- Generated: 2026-06-21T06:51:58.443634+00:00
- Git branch: main
- Git commit: `b049fa6ce0a33dfb83a6f2ea73690eb93c944964`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=785.70ms avg_case_ms=7.86 simplify=224.25ms avg_simplify_ms=2.24, sum total=200 failed=0 elapsed=699.06ms avg_case_ms=3.50 simplify=237.52ms avg_simplify_ms=1.19, product total=100 failed=0 elapsed=476.07ms avg_case_ms=4.76 simplify=137.05ms avg_simplify_ms=1.37, difference total=50 failed=0 elapsed=329.03ms avg_case_ms=6.58 simplify=104.52ms avg_simplify_ms=2.09
- Engine hotspots: sum simplify=237.52ms avg_simplify_ms=1.19 wall=699.06ms, shifted_quotient simplify=224.25ms avg_simplify_ms=2.24 wall=785.70ms, product simplify=137.05ms avg_simplify_ms=1.37 wall=476.07ms, difference simplify=104.52ms avg_simplify_ms=2.09 wall=329.03ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=785.70ms avg_case_ms=7.86 avg_simplify_ms=2.24, sum@0+100 failed=0 elapsed=506.57ms avg_case_ms=5.07 avg_simplify_ms=1.65, product@0+100 failed=0 elapsed=476.07ms avg_case_ms=4.76 avg_simplify_ms=1.37, difference@0+50 failed=0 elapsed=329.03ms avg_case_ms=6.58 avg_simplify_ms=2.09, sum@700+100 failed=0 elapsed=192.49ms avg_case_ms=1.92 avg_simplify_ms=0.73
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.21ms median_wire=13.28ms median_wall=50.32ms, sum@0+100 #173 sum runs=3 median_simplify=11.81ms median_wire=11.86ms median_wall=44.51ms, product@0+100 #175 product runs=3 median_simplify=11.53ms median_wire=11.58ms median_wall=44.37ms, difference@0+50 #174 difference runs=3 median_simplify=11.53ms median_wire=11.58ms median_wall=43.83ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.50ms median_wire=10.58ms median_wall=40.06ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.29s | passed=450 failed=0 total=450 avg_case=5.089ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.41s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.03s | passed=1 failed=0 |
