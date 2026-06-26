# Engine Improvement Scorecard

- Generated: 2026-06-26T08:39:20.094188+00:00
- Git branch: main
- Git commit: `fb4cbeffe2255e4b571e680653964f5b71cbf748`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=814.61ms avg_case_ms=8.15 simplify=234.25ms avg_simplify_ms=2.34, sum total=200 failed=0 elapsed=707.00ms avg_case_ms=3.54 simplify=242.63ms avg_simplify_ms=1.21, product total=100 failed=0 elapsed=488.80ms avg_case_ms=4.89 simplify=144.21ms avg_simplify_ms=1.44, difference total=50 failed=0 elapsed=333.43ms avg_case_ms=6.67 simplify=107.37ms avg_simplify_ms=2.15
- Engine hotspots: sum simplify=242.63ms avg_simplify_ms=1.21 wall=707.00ms, shifted_quotient simplify=234.25ms avg_simplify_ms=2.34 wall=814.61ms, product simplify=144.21ms avg_simplify_ms=1.44 wall=488.80ms, difference simplify=107.37ms avg_simplify_ms=2.15 wall=333.43ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=814.61ms avg_case_ms=8.15 avg_simplify_ms=2.34, sum@0+100 failed=0 elapsed=511.01ms avg_case_ms=5.11 avg_simplify_ms=1.67, product@0+100 failed=0 elapsed=488.80ms avg_case_ms=4.89 avg_simplify_ms=1.44, difference@0+50 failed=0 elapsed=333.43ms avg_case_ms=6.67 avg_simplify_ms=2.15, sum@700+100 failed=0 elapsed=195.99ms avg_case_ms=1.96 avg_simplify_ms=0.76
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.32ms median_wire=13.40ms median_wall=50.82ms, product@0+100 #175 product runs=3 median_simplify=11.71ms median_wire=11.77ms median_wall=44.56ms, difference@0+50 #174 difference runs=3 median_simplify=11.81ms median_wire=11.86ms median_wall=45.31ms, sum@0+100 #173 sum runs=3 median_simplify=11.88ms median_wire=11.94ms median_wall=44.90ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.93ms median_wire=11.00ms median_wall=40.70ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.34s | passed=450 failed=0 total=450 avg_case=5.200ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.42s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.04s | passed=1 failed=0 |
