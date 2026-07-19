# Engine Improvement Scorecard

- Generated: 2026-07-19T19:12:53.839550+00:00
- Git branch: main
- Git commit: `d7ae0c57ba5f183ee3f2c126d00a15df36cecd4c`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=981.18ms avg_case_ms=9.81 simplify=275.51ms avg_simplify_ms=2.76, sum total=200 failed=0 elapsed=888.06ms avg_case_ms=4.44 simplify=283.02ms avg_simplify_ms=1.42, product total=100 failed=0 elapsed=599.96ms avg_case_ms=6.00 simplify=172.61ms avg_simplify_ms=1.73, difference total=50 failed=0 elapsed=400.54ms avg_case_ms=8.01 simplify=122.37ms avg_simplify_ms=2.45
- Engine hotspots: sum simplify=283.02ms avg_simplify_ms=1.42 wall=888.06ms, shifted_quotient simplify=275.51ms avg_simplify_ms=2.76 wall=981.18ms, product simplify=172.61ms avg_simplify_ms=1.73 wall=599.96ms, difference simplify=122.37ms avg_simplify_ms=2.45 wall=400.54ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=981.18ms avg_case_ms=9.81 avg_simplify_ms=2.76, sum@0+100 failed=0 elapsed=655.50ms avg_case_ms=6.56 avg_simplify_ms=2.00, product@0+100 failed=0 elapsed=599.96ms avg_case_ms=6.00 avg_simplify_ms=1.73, difference@0+50 failed=0 elapsed=400.54ms avg_case_ms=8.01 avg_simplify_ms=2.45, sum@700+100 failed=0 elapsed=232.55ms avg_case_ms=2.33 avg_simplify_ms=0.83
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.93ms median_wire=17.00ms median_wall=68.98ms, sum@0+100 #173 sum runs=3 median_simplify=15.45ms median_wire=15.49ms median_wall=58.51ms, difference@0+50 #174 difference runs=3 median_simplify=15.34ms median_wire=15.39ms median_wall=59.21ms, product@0+100 #175 product runs=3 median_simplify=15.41ms median_wire=15.47ms median_wall=58.36ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.80ms median_wire=12.87ms median_wall=48.62ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.87s | passed=450 failed=0 total=450 avg_case=6.378ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.46s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.91s | passed=1 failed=0 |
