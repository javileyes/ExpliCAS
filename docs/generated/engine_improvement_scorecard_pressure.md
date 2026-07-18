# Engine Improvement Scorecard

- Generated: 2026-07-18T09:18:13.666457+00:00
- Git branch: main
- Git commit: `88402a4488b622f77da2dc06d2161fc44e22ec4e`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=1.02s avg_case_ms=10.16 simplify=285.84ms avg_simplify_ms=2.86, sum total=200 failed=0 elapsed=885.79ms avg_case_ms=4.43 simplify=287.77ms avg_simplify_ms=1.44, product total=100 failed=0 elapsed=623.67ms avg_case_ms=6.24 simplify=179.00ms avg_simplify_ms=1.79, difference total=50 failed=0 elapsed=412.29ms avg_case_ms=8.25 simplify=126.16ms avg_simplify_ms=2.52
- Engine hotspots: sum simplify=287.77ms avg_simplify_ms=1.44 wall=885.79ms, shifted_quotient simplify=285.84ms avg_simplify_ms=2.86 wall=1.02s, product simplify=179.00ms avg_simplify_ms=1.79 wall=623.67ms, difference simplify=126.16ms avg_simplify_ms=2.52 wall=412.29ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=1.02s avg_case_ms=10.16 avg_simplify_ms=2.86, sum@0+100 failed=0 elapsed=653.86ms avg_case_ms=6.54 avg_simplify_ms=2.03, product@0+100 failed=0 elapsed=623.67ms avg_case_ms=6.24 avg_simplify_ms=1.79, difference@0+50 failed=0 elapsed=412.29ms avg_case_ms=8.25 avg_simplify_ms=2.52, sum@700+100 failed=0 elapsed=231.93ms avg_case_ms=2.32 avg_simplify_ms=0.85
- Steady-state engine reruns: sum@0+100 #173 sum runs=3 median_simplify=15.59ms median_wire=15.64ms median_wall=59.33ms, shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=17.30ms median_wire=17.37ms median_wall=66.03ms, difference@0+50 #174 difference runs=3 median_simplify=15.63ms median_wire=15.68ms median_wall=60.45ms, product@0+100 #175 product runs=3 median_simplify=15.45ms median_wire=15.51ms median_wall=59.39ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.71ms median_wire=12.79ms median_wall=49.17ms
- Steady-state dominant expressions: sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.94s | passed=450 failed=0 total=450 avg_case=6.533ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.88s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.94s | passed=1 failed=0 |
