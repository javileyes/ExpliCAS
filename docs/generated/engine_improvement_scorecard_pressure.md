# Engine Improvement Scorecard

- Generated: 2026-06-20T20:53:37.980867+00:00
- Git branch: main
- Git commit: `b93bf998a1a3057f074b82954bc464c4d2216474`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=784.31ms avg_case_ms=7.84 simplify=224.18ms avg_simplify_ms=2.24, sum total=200 failed=0 elapsed=701.83ms avg_case_ms=3.51 simplify=237.11ms avg_simplify_ms=1.19, product total=100 failed=0 elapsed=477.57ms avg_case_ms=4.78 simplify=137.42ms avg_simplify_ms=1.37, difference total=50 failed=0 elapsed=328.51ms avg_case_ms=6.57 simplify=105.30ms avg_simplify_ms=2.11
- Engine hotspots: sum simplify=237.11ms avg_simplify_ms=1.19 wall=701.83ms, shifted_quotient simplify=224.18ms avg_simplify_ms=2.24 wall=784.31ms, product simplify=137.42ms avg_simplify_ms=1.37 wall=477.57ms, difference simplify=105.30ms avg_simplify_ms=2.11 wall=328.51ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=784.31ms avg_case_ms=7.84 avg_simplify_ms=2.24, sum@0+100 failed=0 elapsed=507.57ms avg_case_ms=5.08 avg_simplify_ms=1.65, product@0+100 failed=0 elapsed=477.57ms avg_case_ms=4.78 avg_simplify_ms=1.37, difference@0+50 failed=0 elapsed=328.51ms avg_case_ms=6.57 avg_simplify_ms=2.11, sum@700+100 failed=0 elapsed=194.26ms avg_case_ms=1.94 avg_simplify_ms=0.72
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.86ms median_wire=12.93ms median_wall=49.15ms, difference@0+50 #174 difference runs=3 median_simplify=11.66ms median_wire=11.71ms median_wall=44.41ms, sum@0+100 #173 sum runs=3 median_simplify=11.83ms median_wire=11.88ms median_wall=44.78ms, product@0+100 #175 product runs=3 median_simplify=11.69ms median_wire=11.75ms median_wall=45.29ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.82ms median_wire=10.89ms median_wall=40.37ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.29s | passed=450 failed=0 total=450 avg_case=5.089ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.44s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.04s | passed=1 failed=0 |
