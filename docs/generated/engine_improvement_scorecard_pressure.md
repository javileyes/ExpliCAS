# Engine Improvement Scorecard

- Generated: 2026-07-20T01:15:55.765691+00:00
- Git branch: main
- Git commit: `69e671d743b5a4c645168ee9d08dbb04984b3ef7`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=969.33ms avg_case_ms=9.69 simplify=270.92ms avg_simplify_ms=2.71, sum total=200 failed=0 elapsed=890.66ms avg_case_ms=4.45 simplify=285.20ms avg_simplify_ms=1.43, product total=100 failed=0 elapsed=601.70ms avg_case_ms=6.02 simplify=172.81ms avg_simplify_ms=1.73, difference total=50 failed=0 elapsed=397.19ms avg_case_ms=7.94 simplify=121.37ms avg_simplify_ms=2.43
- Engine hotspots: sum simplify=285.20ms avg_simplify_ms=1.43 wall=890.66ms, shifted_quotient simplify=270.92ms avg_simplify_ms=2.71 wall=969.33ms, product simplify=172.81ms avg_simplify_ms=1.73 wall=601.70ms, difference simplify=121.37ms avg_simplify_ms=2.43 wall=397.19ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=969.33ms avg_case_ms=9.69 avg_simplify_ms=2.71, sum@0+100 failed=0 elapsed=656.83ms avg_case_ms=6.57 avg_simplify_ms=2.02, product@0+100 failed=0 elapsed=601.70ms avg_case_ms=6.02 avg_simplify_ms=1.73, difference@0+50 failed=0 elapsed=397.19ms avg_case_ms=7.94 avg_simplify_ms=2.43, sum@700+100 failed=0 elapsed=233.84ms avg_case_ms=2.34 avg_simplify_ms=0.84
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.78ms median_wire=16.85ms median_wall=64.39ms, sum@0+100 #173 sum runs=3 median_simplify=16.51ms median_wire=16.56ms median_wall=62.81ms, difference@0+50 #174 difference runs=3 median_simplify=15.29ms median_wire=15.34ms median_wall=58.97ms, product@0+100 #175 product runs=3 median_simplify=15.37ms median_wire=15.43ms median_wall=57.74ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.88ms median_wire=12.95ms median_wall=49.00ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.86s | passed=450 failed=0 total=450 avg_case=6.356ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.50s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.92s | passed=1 failed=0 |
