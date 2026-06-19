# Engine Improvement Scorecard

- Generated: 2026-06-19T14:07:34.424954+00:00
- Git branch: main
- Git commit: `b74acfdfb6c19b5513aaa00de2a9e3c086a66213`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=786.49ms avg_case_ms=7.86 simplify=225.52ms avg_simplify_ms=2.26, sum total=200 failed=0 elapsed=703.49ms avg_case_ms=3.52 simplify=239.62ms avg_simplify_ms=1.20, product total=100 failed=0 elapsed=484.29ms avg_case_ms=4.84 simplify=139.89ms avg_simplify_ms=1.40, difference total=50 failed=0 elapsed=332.90ms avg_case_ms=6.66 simplify=106.43ms avg_simplify_ms=2.13
- Engine hotspots: sum simplify=239.62ms avg_simplify_ms=1.20 wall=703.49ms, shifted_quotient simplify=225.52ms avg_simplify_ms=2.26 wall=786.49ms, product simplify=139.89ms avg_simplify_ms=1.40 wall=484.29ms, difference simplify=106.43ms avg_simplify_ms=2.13 wall=332.90ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=786.49ms avg_case_ms=7.86 avg_simplify_ms=2.26, sum@0+100 failed=0 elapsed=508.02ms avg_case_ms=5.08 avg_simplify_ms=1.66, product@0+100 failed=0 elapsed=484.29ms avg_case_ms=4.84 avg_simplify_ms=1.40, difference@0+50 failed=0 elapsed=332.90ms avg_case_ms=6.66 avg_simplify_ms=2.13, sum@700+100 failed=0 elapsed=195.47ms avg_case_ms=1.95 avg_simplify_ms=0.74
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.16ms median_wire=13.24ms median_wall=49.63ms, difference@0+50 #174 difference runs=3 median_simplify=11.75ms median_wire=11.80ms median_wall=44.50ms, product@0+100 #175 product runs=3 median_simplify=11.78ms median_wire=11.84ms median_wall=44.74ms, sum@0+100 #173 sum runs=3 median_simplify=11.74ms median_wire=11.79ms median_wall=44.69ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.63ms median_wire=10.70ms median_wall=40.23ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.31s | passed=450 failed=0 total=450 avg_case=5.133ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.47s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.04s | passed=1 failed=0 |
