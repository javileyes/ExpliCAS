# Engine Improvement Scorecard

- Generated: 2026-07-23T16:48:36.353055+00:00
- Git branch: main
- Git commit: `975c781c6c116cafb233057f7f89c79669f7dbc5`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=1.00s avg_case_ms=10.00 simplify=279.61ms avg_simplify_ms=2.80, sum total=200 failed=0 elapsed=873.64ms avg_case_ms=4.37 simplify=285.16ms avg_simplify_ms=1.43, product total=100 failed=0 elapsed=606.67ms avg_case_ms=6.07 simplify=173.77ms avg_simplify_ms=1.74, difference total=50 failed=0 elapsed=402.91ms avg_case_ms=8.06 simplify=122.99ms avg_simplify_ms=2.46
- Engine hotspots: sum simplify=285.16ms avg_simplify_ms=1.43 wall=873.64ms, shifted_quotient simplify=279.61ms avg_simplify_ms=2.80 wall=1.00s, product simplify=173.77ms avg_simplify_ms=1.74 wall=606.67ms, difference simplify=122.99ms avg_simplify_ms=2.46 wall=402.91ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=1.00s avg_case_ms=10.00 avg_simplify_ms=2.80, sum@0+100 failed=0 elapsed=642.68ms avg_case_ms=6.43 avg_simplify_ms=2.02, product@0+100 failed=0 elapsed=606.67ms avg_case_ms=6.07 avg_simplify_ms=1.74, difference@0+50 failed=0 elapsed=402.91ms avg_case_ms=8.06 avg_simplify_ms=2.46, sum@700+100 failed=0 elapsed=230.96ms avg_case_ms=2.31 avg_simplify_ms=0.83
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.81ms median_wire=16.89ms median_wall=64.58ms, difference@0+50 #174 difference runs=3 median_simplify=15.55ms median_wire=15.60ms median_wall=58.51ms, sum@0+100 #173 sum runs=3 median_simplify=15.18ms median_wire=15.23ms median_wall=58.67ms, product@0+100 #175 product runs=3 median_simplify=15.87ms median_wire=15.93ms median_wall=59.34ms, shifted_quotient@0+100 #160 shifted_quotient runs=3 median_simplify=11.11ms median_wire=11.18ms median_wall=42.49ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.88s | passed=450 failed=0 total=450 avg_case=6.400ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.58s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.92s | passed=1 failed=0 |
