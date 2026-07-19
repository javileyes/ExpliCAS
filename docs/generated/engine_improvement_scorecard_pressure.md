# Engine Improvement Scorecard

- Generated: 2026-07-19T21:53:09.015049+00:00
- Git branch: main
- Git commit: `59d5dc3d02277e714c04d469a251e186653351ab`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=988.11ms avg_case_ms=9.88 simplify=276.94ms avg_simplify_ms=2.77, sum total=200 failed=0 elapsed=886.46ms avg_case_ms=4.43 simplify=285.80ms avg_simplify_ms=1.43, product total=100 failed=0 elapsed=606.72ms avg_case_ms=6.07 simplify=174.24ms avg_simplify_ms=1.74, difference total=50 failed=0 elapsed=395.81ms avg_case_ms=7.92 simplify=120.67ms avg_simplify_ms=2.41
- Engine hotspots: sum simplify=285.80ms avg_simplify_ms=1.43 wall=886.46ms, shifted_quotient simplify=276.94ms avg_simplify_ms=2.77 wall=988.11ms, product simplify=174.24ms avg_simplify_ms=1.74 wall=606.72ms, difference simplify=120.67ms avg_simplify_ms=2.41 wall=395.81ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=988.11ms avg_case_ms=9.88 avg_simplify_ms=2.77, sum@0+100 failed=0 elapsed=653.84ms avg_case_ms=6.54 avg_simplify_ms=2.03, product@0+100 failed=0 elapsed=606.72ms avg_case_ms=6.07 avg_simplify_ms=1.74, difference@0+50 failed=0 elapsed=395.81ms avg_case_ms=7.92 avg_simplify_ms=2.41, sum@700+100 failed=0 elapsed=232.62ms avg_case_ms=2.33 avg_simplify_ms=0.83
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.88ms median_wire=16.95ms median_wall=64.03ms, sum@0+100 #173 sum runs=3 median_simplify=15.12ms median_wire=15.17ms median_wall=57.59ms, difference@0+50 #174 difference runs=3 median_simplify=15.83ms median_wire=15.89ms median_wall=60.71ms, product@0+100 #175 product runs=3 median_simplify=17.88ms median_wire=17.94ms median_wall=60.27ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=13.33ms median_wire=13.41ms median_wall=50.11ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.88s | passed=450 failed=0 total=450 avg_case=6.400ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.50s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.92s | passed=1 failed=0 |
