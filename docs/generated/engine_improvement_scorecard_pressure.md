# Engine Improvement Scorecard

- Generated: 2026-06-25T23:26:26.269683+00:00
- Git branch: main
- Git commit: `97c4c6859004575edd3f641c5a25413819fd63df`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=786.78ms avg_case_ms=7.87 simplify=225.17ms avg_simplify_ms=2.25, sum total=200 failed=0 elapsed=696.99ms avg_case_ms=3.48 simplify=239.83ms avg_simplify_ms=1.20, product total=100 failed=0 elapsed=484.31ms avg_case_ms=4.84 simplify=142.30ms avg_simplify_ms=1.42, difference total=50 failed=0 elapsed=325.64ms avg_case_ms=6.51 simplify=106.02ms avg_simplify_ms=2.12
- Engine hotspots: sum simplify=239.83ms avg_simplify_ms=1.20 wall=696.99ms, shifted_quotient simplify=225.17ms avg_simplify_ms=2.25 wall=786.78ms, product simplify=142.30ms avg_simplify_ms=1.42 wall=484.31ms, difference simplify=106.02ms avg_simplify_ms=2.12 wall=325.64ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=786.78ms avg_case_ms=7.87 avg_simplify_ms=2.25, sum@0+100 failed=0 elapsed=504.72ms avg_case_ms=5.05 avg_simplify_ms=1.66, product@0+100 failed=0 elapsed=484.31ms avg_case_ms=4.84 avg_simplify_ms=1.42, difference@0+50 failed=0 elapsed=325.64ms avg_case_ms=6.51 avg_simplify_ms=2.12, sum@700+100 failed=0 elapsed=192.27ms avg_case_ms=1.92 avg_simplify_ms=0.74
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.93ms median_wire=13.00ms median_wall=49.33ms, product@0+100 #175 product runs=3 median_simplify=11.81ms median_wire=11.86ms median_wall=45.35ms, sum@0+100 #173 sum runs=3 median_simplify=11.70ms median_wire=11.74ms median_wall=44.21ms, difference@0+50 #174 difference runs=3 median_simplify=11.51ms median_wire=11.56ms median_wall=44.01ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.57ms median_wire=10.65ms median_wall=40.06ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.29s | passed=450 failed=0 total=450 avg_case=5.089ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.45s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.06s | passed=1 failed=0 |
