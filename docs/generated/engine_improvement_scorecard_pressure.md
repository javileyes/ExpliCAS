# Engine Improvement Scorecard

- Generated: 2026-07-18T14:15:24.724151+00:00
- Git branch: main
- Git commit: `1396a6f078fa5124b124008cebd3f56d51dfffc8`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=985.51ms avg_case_ms=9.86 simplify=273.32ms avg_simplify_ms=2.73, sum total=200 failed=0 elapsed=876.16ms avg_case_ms=4.38 simplify=291.96ms avg_simplify_ms=1.46, product total=100 failed=0 elapsed=597.75ms avg_case_ms=5.98 simplify=171.55ms avg_simplify_ms=1.72, difference total=50 failed=0 elapsed=401.30ms avg_case_ms=8.03 simplify=121.51ms avg_simplify_ms=2.43
- Engine hotspots: sum simplify=291.96ms avg_simplify_ms=1.46 wall=876.16ms, shifted_quotient simplify=273.32ms avg_simplify_ms=2.73 wall=985.51ms, product simplify=171.55ms avg_simplify_ms=1.72 wall=597.75ms, difference simplify=121.51ms avg_simplify_ms=2.43 wall=401.30ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=985.51ms avg_case_ms=9.86 avg_simplify_ms=2.73, sum@0+100 failed=0 elapsed=646.79ms avg_case_ms=6.47 avg_simplify_ms=2.10, product@0+100 failed=0 elapsed=597.75ms avg_case_ms=5.98 avg_simplify_ms=1.72, difference@0+50 failed=0 elapsed=401.30ms avg_case_ms=8.03 avg_simplify_ms=2.43, sum@700+100 failed=0 elapsed=229.37ms avg_case_ms=2.29 avg_simplify_ms=0.82
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.56ms median_wire=16.63ms median_wall=63.27ms, sum@0+100 #173 sum runs=3 median_simplify=15.09ms median_wire=15.15ms median_wall=57.79ms, difference@0+50 #174 difference runs=3 median_simplify=15.18ms median_wire=15.23ms median_wall=58.12ms, product@0+100 #175 product runs=3 median_simplify=15.23ms median_wire=15.28ms median_wall=58.15ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.92ms median_wire=12.99ms median_wall=49.04ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.86s | passed=450 failed=0 total=450 avg_case=6.356ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.47s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.90s | passed=1 failed=0 |
