# Engine Improvement Scorecard

- Generated: 2026-07-30T23:10:19.406856+00:00
- Git branch: main
- Git commit: `8ddba41eb5d0042952137d92165e78be9cf0649e`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=986.67ms avg_case_ms=9.87 simplify=276.03ms avg_simplify_ms=2.76, sum total=200 failed=0 elapsed=871.63ms avg_case_ms=4.36 simplify=278.55ms avg_simplify_ms=1.39, product total=100 failed=0 elapsed=608.47ms avg_case_ms=6.08 simplify=174.53ms avg_simplify_ms=1.75, difference total=50 failed=0 elapsed=395.93ms avg_case_ms=7.92 simplify=115.91ms avg_simplify_ms=2.32
- Engine hotspots: sum simplify=278.55ms avg_simplify_ms=1.39 wall=871.63ms, shifted_quotient simplify=276.03ms avg_simplify_ms=2.76 wall=986.67ms, product simplify=174.53ms avg_simplify_ms=1.75 wall=608.47ms, difference simplify=115.91ms avg_simplify_ms=2.32 wall=395.93ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=986.67ms avg_case_ms=9.87 avg_simplify_ms=2.76, sum@0+100 failed=0 elapsed=637.60ms avg_case_ms=6.38 avg_simplify_ms=1.95, product@0+100 failed=0 elapsed=608.47ms avg_case_ms=6.08 avg_simplify_ms=1.75, difference@0+50 failed=0 elapsed=395.93ms avg_case_ms=7.92 avg_simplify_ms=2.32, sum@700+100 failed=0 elapsed=234.03ms avg_case_ms=2.34 avg_simplify_ms=0.84
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=17.01ms median_wire=17.07ms median_wall=65.19ms, sum@0+100 #173 sum runs=3 median_simplify=15.29ms median_wire=15.34ms median_wall=58.10ms, difference@0+50 #174 difference runs=3 median_simplify=15.45ms median_wire=15.49ms median_wall=58.96ms, product@0+100 #175 product runs=3 median_simplify=15.12ms median_wire=15.16ms median_wall=57.98ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=13.28ms median_wire=13.35ms median_wall=50.64ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.86s | passed=450 failed=0 total=450 avg_case=6.356ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.15s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.73s | passed=1 failed=0 |
