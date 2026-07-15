# Engine Improvement Scorecard

- Generated: 2026-07-15T21:46:27.047308+00:00
- Git branch: main
- Git commit: `4fd29a4ca9ec64cca88717fb98d6ecf84a172d19`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=365

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=1.04s avg_case_ms=10.39 simplify=289.93ms avg_simplify_ms=2.90, sum total=200 failed=0 elapsed=908.48ms avg_case_ms=4.54 simplify=300.47ms avg_simplify_ms=1.50, product total=100 failed=0 elapsed=627.64ms avg_case_ms=6.28 simplify=182.33ms avg_simplify_ms=1.82, difference total=50 failed=0 elapsed=415.81ms avg_case_ms=8.32 simplify=126.26ms avg_simplify_ms=2.53
- Engine hotspots: sum simplify=300.47ms avg_simplify_ms=1.50 wall=908.48ms, shifted_quotient simplify=289.93ms avg_simplify_ms=2.90 wall=1.04s, product simplify=182.33ms avg_simplify_ms=1.82 wall=627.64ms, difference simplify=126.26ms avg_simplify_ms=2.53 wall=415.81ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=1.04s avg_case_ms=10.39 avg_simplify_ms=2.90, sum@0+100 failed=0 elapsed=673.77ms avg_case_ms=6.74 avg_simplify_ms=2.15, product@0+100 failed=0 elapsed=627.64ms avg_case_ms=6.28 avg_simplify_ms=1.82, difference@0+50 failed=0 elapsed=415.81ms avg_case_ms=8.32 avg_simplify_ms=2.53, sum@700+100 failed=0 elapsed=234.71ms avg_case_ms=2.35 avg_simplify_ms=0.86
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.92ms median_wire=17.01ms median_wall=66.25ms, sum@0+100 #173 sum runs=3 median_simplify=16.14ms median_wire=16.19ms median_wall=60.16ms, difference@0+50 #174 difference runs=3 median_simplify=15.87ms median_wire=15.93ms median_wall=68.00ms, product@0+100 #175 product runs=3 median_simplify=15.57ms median_wire=15.63ms median_wall=60.11ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=13.76ms median_wire=13.84ms median_wall=51.73ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.99s | passed=450 failed=0 total=450 avg_case=6.644ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.96s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.97s | passed=1 failed=0 |
