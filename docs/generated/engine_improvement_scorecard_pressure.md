# Engine Improvement Scorecard

- Generated: 2026-07-13T08:13:01.154386+00:00
- Git branch: main
- Git commit: `c1383f09c0717b4ec9461a05843709e892c93571`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=952.41ms avg_case_ms=9.52 simplify=262.79ms avg_simplify_ms=2.63, sum total=200 failed=0 elapsed=830.41ms avg_case_ms=4.15 simplify=268.69ms avg_simplify_ms=1.34, product total=100 failed=0 elapsed=592.37ms avg_case_ms=5.92 simplify=168.25ms avg_simplify_ms=1.68, difference total=50 failed=0 elapsed=393.16ms avg_case_ms=7.86 simplify=119.08ms avg_simplify_ms=2.38
- Engine hotspots: sum simplify=268.69ms avg_simplify_ms=1.34 wall=830.41ms, shifted_quotient simplify=262.79ms avg_simplify_ms=2.63 wall=952.41ms, product simplify=168.25ms avg_simplify_ms=1.68 wall=592.37ms, difference simplify=119.08ms avg_simplify_ms=2.38 wall=393.16ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=952.41ms avg_case_ms=9.52 avg_simplify_ms=2.63, sum@0+100 failed=0 elapsed=604.64ms avg_case_ms=6.05 avg_simplify_ms=1.88, product@0+100 failed=0 elapsed=592.37ms avg_case_ms=5.92 avg_simplify_ms=1.68, difference@0+50 failed=0 elapsed=393.16ms avg_case_ms=7.86 avg_simplify_ms=2.38, sum@700+100 failed=0 elapsed=225.77ms avg_case_ms=2.26 avg_simplify_ms=0.81
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.67ms median_wire=16.74ms median_wall=63.66ms, product@0+100 #175 product runs=3 median_simplify=14.87ms median_wire=14.91ms median_wall=58.15ms, difference@0+50 #174 difference runs=3 median_simplify=15.18ms median_wire=15.23ms median_wall=57.75ms, sum@0+100 #173 sum runs=3 median_simplify=15.01ms median_wire=15.06ms median_wall=57.26ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.65ms median_wire=12.72ms median_wall=48.51ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.77s | passed=450 failed=0 total=450 avg_case=6.156ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.99s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.88s | passed=1 failed=0 |
