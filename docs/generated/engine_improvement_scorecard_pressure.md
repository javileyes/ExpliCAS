# Engine Improvement Scorecard

- Generated: 2026-07-18T19:23:50.739307+00:00
- Git branch: main
- Git commit: `2d67fc4d7425f9d82eb5cee7f90060ceb39252e8`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=998.48ms avg_case_ms=9.98 simplify=282.03ms avg_simplify_ms=2.82, sum total=200 failed=0 elapsed=892.88ms avg_case_ms=4.46 simplify=286.99ms avg_simplify_ms=1.43, product total=100 failed=0 elapsed=601.34ms avg_case_ms=6.01 simplify=172.90ms avg_simplify_ms=1.73, difference total=50 failed=0 elapsed=398.57ms avg_case_ms=7.97 simplify=121.21ms avg_simplify_ms=2.42
- Engine hotspots: sum simplify=286.99ms avg_simplify_ms=1.43 wall=892.88ms, shifted_quotient simplify=282.03ms avg_simplify_ms=2.82 wall=998.48ms, product simplify=172.90ms avg_simplify_ms=1.73 wall=601.34ms, difference simplify=121.21ms avg_simplify_ms=2.42 wall=398.57ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=998.48ms avg_case_ms=9.98 avg_simplify_ms=2.82, sum@0+100 failed=0 elapsed=664.32ms avg_case_ms=6.64 avg_simplify_ms=2.05, product@0+100 failed=0 elapsed=601.34ms avg_case_ms=6.01 avg_simplify_ms=1.73, difference@0+50 failed=0 elapsed=398.57ms avg_case_ms=7.97 avg_simplify_ms=2.42, sum@700+100 failed=0 elapsed=228.56ms avg_case_ms=2.29 avg_simplify_ms=0.82
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=17.02ms median_wire=17.08ms median_wall=64.54ms, sum@0+100 #173 sum runs=3 median_simplify=15.25ms median_wire=15.30ms median_wall=58.15ms, difference@0+50 #174 difference runs=3 median_simplify=15.27ms median_wire=15.32ms median_wall=58.12ms, product@0+100 #175 product runs=3 median_simplify=16.58ms median_wire=16.63ms median_wall=60.12ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.93ms median_wire=13.00ms median_wall=48.94ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.89s | passed=450 failed=0 total=450 avg_case=6.422ms |
| `calculus_diff_exhaustive_contract` | `pass` | 13.06s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.96s | passed=1 failed=0 |
