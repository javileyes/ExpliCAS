# Engine Improvement Scorecard

- Generated: 2026-07-27T01:59:27.822004+00:00
- Git branch: main
- Git commit: `1e244f98d43eaf9b29f4ace5729aaedb7d69b795`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=380

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=945.63ms avg_case_ms=9.46 simplify=264.52ms avg_simplify_ms=2.65, sum total=200 failed=0 elapsed=832.83ms avg_case_ms=4.16 simplify=271.27ms avg_simplify_ms=1.36, product total=100 failed=0 elapsed=581.54ms avg_case_ms=5.82 simplify=166.56ms avg_simplify_ms=1.67, difference total=50 failed=0 elapsed=386.01ms avg_case_ms=7.72 simplify=117.56ms avg_simplify_ms=2.35
- Engine hotspots: sum simplify=271.27ms avg_simplify_ms=1.36 wall=832.83ms, shifted_quotient simplify=264.52ms avg_simplify_ms=2.65 wall=945.63ms, product simplify=166.56ms avg_simplify_ms=1.67 wall=581.54ms, difference simplify=117.56ms avg_simplify_ms=2.35 wall=386.01ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=945.63ms avg_case_ms=9.46 avg_simplify_ms=2.65, sum@0+100 failed=0 elapsed=610.09ms avg_case_ms=6.10 avg_simplify_ms=1.91, product@0+100 failed=0 elapsed=581.54ms avg_case_ms=5.82 avg_simplify_ms=1.67, difference@0+50 failed=0 elapsed=386.01ms avg_case_ms=7.72 avg_simplify_ms=2.35, sum@700+100 failed=0 elapsed=222.73ms avg_case_ms=2.23 avg_simplify_ms=0.80
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.02ms median_wire=16.09ms median_wall=62.05ms, sum@0+100 #173 sum runs=3 median_simplify=14.67ms median_wire=14.71ms median_wall=55.78ms, difference@0+50 #174 difference runs=3 median_simplify=14.77ms median_wire=14.82ms median_wall=56.60ms, product@0+100 #175 product runs=3 median_simplify=14.76ms median_wire=14.81ms median_wall=56.60ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.56ms median_wire=12.63ms median_wall=47.45ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.75s | passed=450 failed=0 total=450 avg_case=6.111ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.07s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.83s | passed=1 failed=0 |
