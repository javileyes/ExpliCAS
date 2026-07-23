# Engine Improvement Scorecard

- Generated: 2026-07-23T19:20:17.958708+00:00
- Git branch: main
- Git commit: `94f378a92e62c9ab1c813229c336d4b05ddaecfe`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=367

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=1.01s avg_case_ms=10.14 simplify=286.26ms avg_simplify_ms=2.86, sum total=200 failed=0 elapsed=888.87ms avg_case_ms=4.44 simplify=294.16ms avg_simplify_ms=1.47, product total=100 failed=0 elapsed=622.61ms avg_case_ms=6.23 simplify=177.95ms avg_simplify_ms=1.78, difference total=50 failed=0 elapsed=397.46ms avg_case_ms=7.95 simplify=121.83ms avg_simplify_ms=2.44
- Engine hotspots: sum simplify=294.16ms avg_simplify_ms=1.47 wall=888.87ms, shifted_quotient simplify=286.26ms avg_simplify_ms=2.86 wall=1.01s, product simplify=177.95ms avg_simplify_ms=1.78 wall=622.61ms, difference simplify=121.83ms avg_simplify_ms=2.44 wall=397.46ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=1.01s avg_case_ms=10.14 avg_simplify_ms=2.86, sum@0+100 failed=0 elapsed=655.73ms avg_case_ms=6.56 avg_simplify_ms=2.10, product@0+100 failed=0 elapsed=622.61ms avg_case_ms=6.23 avg_simplify_ms=1.78, difference@0+50 failed=0 elapsed=397.46ms avg_case_ms=7.95 avg_simplify_ms=2.44, sum@700+100 failed=0 elapsed=233.15ms avg_case_ms=2.33 avg_simplify_ms=0.84
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=17.10ms median_wire=17.17ms median_wall=64.91ms, product@0+100 #175 product runs=3 median_simplify=15.67ms median_wire=15.73ms median_wall=59.39ms, difference@0+50 #174 difference runs=3 median_simplify=15.08ms median_wire=15.13ms median_wall=58.39ms, sum@0+100 #173 sum runs=3 median_simplify=17.37ms median_wire=17.42ms median_wall=62.00ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=13.05ms median_wire=13.12ms median_wall=50.15ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.92s | passed=450 failed=0 total=450 avg_case=6.489ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.81s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.92s | passed=1 failed=0 |
