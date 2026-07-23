# Engine Improvement Scorecard

- Generated: 2026-07-23T21:32:13.121242+00:00
- Git branch: main
- Git commit: `d443ee0024f7b9086db808e2f3e7f7e35c85d4d8`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=994.97ms avg_case_ms=9.95 simplify=280.29ms avg_simplify_ms=2.80, sum total=200 failed=0 elapsed=874.29ms avg_case_ms=4.37 simplify=287.38ms avg_simplify_ms=1.44, product total=100 failed=0 elapsed=608.07ms avg_case_ms=6.08 simplify=174.75ms avg_simplify_ms=1.75, difference total=50 failed=0 elapsed=400.47ms avg_case_ms=8.01 simplify=121.86ms avg_simplify_ms=2.44
- Engine hotspots: sum simplify=287.38ms avg_simplify_ms=1.44 wall=874.29ms, shifted_quotient simplify=280.29ms avg_simplify_ms=2.80 wall=994.97ms, product simplify=174.75ms avg_simplify_ms=1.75 wall=608.07ms, difference simplify=121.86ms avg_simplify_ms=2.44 wall=400.47ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=994.97ms avg_case_ms=9.95 avg_simplify_ms=2.80, sum@0+100 failed=0 elapsed=644.75ms avg_case_ms=6.45 avg_simplify_ms=2.05, product@0+100 failed=0 elapsed=608.07ms avg_case_ms=6.08 avg_simplify_ms=1.75, difference@0+50 failed=0 elapsed=400.47ms avg_case_ms=8.01 avg_simplify_ms=2.44, sum@700+100 failed=0 elapsed=229.54ms avg_case_ms=2.30 avg_simplify_ms=0.83
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.88ms median_wire=16.96ms median_wall=64.71ms, sum@0+100 #157 sum runs=3 median_simplify=9.71ms median_wire=9.76ms median_wall=36.56ms, difference@0+50 #174 difference runs=3 median_simplify=15.41ms median_wire=15.46ms median_wall=58.63ms, sum@0+100 #173 sum runs=3 median_simplify=15.71ms median_wire=15.76ms median_wall=67.21ms, product@0+100 #175 product runs=3 median_simplify=15.38ms median_wire=15.44ms median_wall=59.04ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #157 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^3) + ln(y^2) - ln(x^3 * y^2)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.88s | passed=450 failed=0 total=450 avg_case=6.400ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.58s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.93s | passed=1 failed=0 |
