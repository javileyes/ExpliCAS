# Engine Improvement Scorecard

- Generated: 2026-06-13T19:36:09.659672+00:00
- Git branch: main
- Git commit: `705ef64776d6af5d3e657c2beb99ddf76eeb97b4`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=344

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=896.45ms avg_case_ms=8.96 simplify=290.57ms avg_simplify_ms=2.91, sum total=200 failed=0 elapsed=696.38ms avg_case_ms=3.48 simplify=233.12ms avg_simplify_ms=1.17, product total=100 failed=0 elapsed=474.14ms avg_case_ms=4.74 simplify=135.91ms avg_simplify_ms=1.36, difference total=50 failed=0 elapsed=326.39ms avg_case_ms=6.53 simplify=103.34ms avg_simplify_ms=2.07
- Engine hotspots: shifted_quotient simplify=290.57ms avg_simplify_ms=2.91 wall=896.45ms, sum simplify=233.12ms avg_simplify_ms=1.17 wall=696.38ms, product simplify=135.91ms avg_simplify_ms=1.36 wall=474.14ms, difference simplify=103.34ms avg_simplify_ms=2.07 wall=326.39ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=896.45ms avg_case_ms=8.96 avg_simplify_ms=2.91, sum@0+100 failed=0 elapsed=505.54ms avg_case_ms=5.06 avg_simplify_ms=1.62, product@0+100 failed=0 elapsed=474.14ms avg_case_ms=4.74 avg_simplify_ms=1.36, difference@0+50 failed=0 elapsed=326.39ms avg_case_ms=6.53 avg_simplify_ms=2.07, sum@700+100 failed=0 elapsed=190.84ms avg_case_ms=1.91 avg_simplify_ms=0.71
- Steady-state engine reruns: shifted_quotient@0+100 #316 shifted_quotient runs=3 median_simplify=2.81ms median_wire=2.85ms median_wall=10.14ms, shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.90ms median_wire=12.97ms median_wall=49.37ms, difference@0+50 #174 difference runs=3 median_simplify=11.56ms median_wire=11.61ms median_wall=44.51ms, sum@0+100 #173 sum runs=3 median_simplify=11.58ms median_wire=11.63ms median_wall=44.21ms, product@0+100 #175 product runs=3 median_simplify=11.70ms median_wire=11.75ms median_wall=44.41ms
- Steady-state dominant expressions: shifted_quotient@0+100 #316 shifted_quotient expr=((tan(x) + cot(x) - sec(x)*csc(x)) + 1)/((ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + 1), shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.39s | passed=450 failed=0 total=450 avg_case=5.311ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.83s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.92s | passed=1 failed=0 |
