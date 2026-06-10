# Engine Improvement Scorecard

- Generated: 2026-06-10T21:14:25.515009+00:00
- Git branch: main
- Git commit: `f0f0c52c40b346e7cec68f15cd852b9b8292aae6`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=767.29ms avg_case_ms=7.67 simplify=216.80ms avg_simplify_ms=2.17, sum total=200 failed=0 elapsed=701.11ms avg_case_ms=3.51 simplify=232.83ms avg_simplify_ms=1.16, product total=100 failed=0 elapsed=471.19ms avg_case_ms=4.71 simplify=135.02ms avg_simplify_ms=1.35, difference total=50 failed=0 elapsed=322.39ms avg_case_ms=6.45 simplify=102.18ms avg_simplify_ms=2.04
- Engine hotspots: sum simplify=232.83ms avg_simplify_ms=1.16 wall=701.11ms, shifted_quotient simplify=216.80ms avg_simplify_ms=2.17 wall=767.29ms, product simplify=135.02ms avg_simplify_ms=1.35 wall=471.19ms, difference simplify=102.18ms avg_simplify_ms=2.04 wall=322.39ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=767.29ms avg_case_ms=7.67 avg_simplify_ms=2.17, sum@0+100 failed=0 elapsed=509.46ms avg_case_ms=5.09 avg_simplify_ms=1.62, product@0+100 failed=0 elapsed=471.19ms avg_case_ms=4.71 avg_simplify_ms=1.35, difference@0+50 failed=0 elapsed=322.39ms avg_case_ms=6.45 avg_simplify_ms=2.04, sum@700+100 failed=0 elapsed=191.65ms avg_case_ms=1.92 avg_simplify_ms=0.71
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.55ms median_wire=12.62ms median_wall=47.86ms, difference@0+50 #174 difference runs=3 median_simplify=11.34ms median_wire=11.39ms median_wall=43.31ms, sum@0+100 #173 sum runs=3 median_simplify=11.40ms median_wire=11.45ms median_wall=43.32ms, product@0+100 #175 product runs=3 median_simplify=11.08ms median_wire=11.12ms median_wall=42.84ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=11.48ms median_wire=11.56ms median_wall=41.83ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.26s | passed=450 failed=0 total=450 avg_case=5.022ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.82s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 17.16s | passed=1 failed=0 |
