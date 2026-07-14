# Engine Improvement Scorecard

- Generated: 2026-07-14T00:07:15.710662+00:00
- Git branch: main
- Git commit: `a81b2bcc9602e75a329e8e61a06421b4153d78de`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=357

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=959.12ms avg_case_ms=9.59 simplify=264.58ms avg_simplify_ms=2.65, sum total=200 failed=0 elapsed=854.88ms avg_case_ms=4.27 simplify=273.93ms avg_simplify_ms=1.37, product total=100 failed=0 elapsed=579.22ms avg_case_ms=5.79 simplify=164.56ms avg_simplify_ms=1.65, difference total=50 failed=0 elapsed=394.18ms avg_case_ms=7.88 simplify=119.36ms avg_simplify_ms=2.39
- Engine hotspots: sum simplify=273.93ms avg_simplify_ms=1.37 wall=854.88ms, shifted_quotient simplify=264.58ms avg_simplify_ms=2.65 wall=959.12ms, product simplify=164.56ms avg_simplify_ms=1.65 wall=579.22ms, difference simplify=119.36ms avg_simplify_ms=2.39 wall=394.18ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=959.12ms avg_case_ms=9.59 avg_simplify_ms=2.65, sum@0+100 failed=0 elapsed=627.26ms avg_case_ms=6.27 avg_simplify_ms=1.92, product@0+100 failed=0 elapsed=579.22ms avg_case_ms=5.79 avg_simplify_ms=1.65, difference@0+50 failed=0 elapsed=394.18ms avg_case_ms=7.88 avg_simplify_ms=2.39, sum@700+100 failed=0 elapsed=227.62ms avg_case_ms=2.28 avg_simplify_ms=0.82
- Steady-state engine reruns: sum@0+100 #173 sum runs=3 median_simplify=14.90ms median_wire=14.95ms median_wall=57.48ms, shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.32ms median_wire=16.39ms median_wall=63.55ms, difference@0+50 #174 difference runs=3 median_simplify=14.95ms median_wire=14.99ms median_wall=57.83ms, product@0+100 #175 product runs=3 median_simplify=16.58ms median_wire=16.64ms median_wall=59.28ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.05ms median_wire=12.13ms median_wall=46.66ms
- Steady-state dominant expressions: sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.79s | passed=450 failed=0 total=450 avg_case=6.200ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.33s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.90s | passed=1 failed=0 |
