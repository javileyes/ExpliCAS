# Engine Improvement Scorecard

- Generated: 2026-07-28T11:21:44.915193+00:00
- Git branch: main
- Git commit: `023c17643e3196def7edd81e327ab585a61ed338`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=944.47ms avg_case_ms=9.44 simplify=262.96ms avg_simplify_ms=2.63, sum total=200 failed=0 elapsed=827.72ms avg_case_ms=4.14 simplify=269.03ms avg_simplify_ms=1.35, product total=100 failed=0 elapsed=582.58ms avg_case_ms=5.83 simplify=166.81ms avg_simplify_ms=1.67, difference total=50 failed=0 elapsed=386.78ms avg_case_ms=7.74 simplify=117.86ms avg_simplify_ms=2.36
- Engine hotspots: sum simplify=269.03ms avg_simplify_ms=1.35 wall=827.72ms, shifted_quotient simplify=262.96ms avg_simplify_ms=2.63 wall=944.47ms, product simplify=166.81ms avg_simplify_ms=1.67 wall=582.58ms, difference simplify=117.86ms avg_simplify_ms=2.36 wall=386.78ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=944.47ms avg_case_ms=9.44 avg_simplify_ms=2.63, sum@0+100 failed=0 elapsed=605.56ms avg_case_ms=6.06 avg_simplify_ms=1.90, product@0+100 failed=0 elapsed=582.58ms avg_case_ms=5.83 avg_simplify_ms=1.67, difference@0+50 failed=0 elapsed=386.78ms avg_case_ms=7.74 avg_simplify_ms=2.36, sum@700+100 failed=0 elapsed=222.16ms avg_case_ms=2.22 avg_simplify_ms=0.79
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.19ms median_wire=16.26ms median_wall=62.05ms, difference@0+50 #174 difference runs=3 median_simplify=14.67ms median_wire=14.72ms median_wall=57.02ms, sum@0+100 #173 sum runs=3 median_simplify=14.79ms median_wire=14.83ms median_wall=56.11ms, product@0+100 #175 product runs=3 median_simplify=14.84ms median_wire=14.88ms median_wall=56.70ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.59ms median_wire=12.66ms median_wall=47.63ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.74s | passed=450 failed=0 total=450 avg_case=6.089ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.01s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.84s | passed=1 failed=0 |
