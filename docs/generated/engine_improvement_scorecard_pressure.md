# Engine Improvement Scorecard

- Generated: 2026-06-13T18:31:46.968642+00:00
- Git branch: main
- Git commit: `d4bc4626c13d0eb0c0462d3d453072b50fd735d4`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=783.30ms avg_case_ms=7.83 simplify=222.78ms avg_simplify_ms=2.23, sum total=200 failed=0 elapsed=687.01ms avg_case_ms=3.44 simplify=230.16ms avg_simplify_ms=1.15, product total=100 failed=0 elapsed=475.04ms avg_case_ms=4.75 simplify=136.97ms avg_simplify_ms=1.37, difference total=50 failed=0 elapsed=329.08ms avg_case_ms=6.58 simplify=104.25ms avg_simplify_ms=2.08
- Engine hotspots: sum simplify=230.16ms avg_simplify_ms=1.15 wall=687.01ms, shifted_quotient simplify=222.78ms avg_simplify_ms=2.23 wall=783.30ms, product simplify=136.97ms avg_simplify_ms=1.37 wall=475.04ms, difference simplify=104.25ms avg_simplify_ms=2.08 wall=329.08ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=783.30ms avg_case_ms=7.83 avg_simplify_ms=2.23, sum@0+100 failed=0 elapsed=498.59ms avg_case_ms=4.99 avg_simplify_ms=1.61, product@0+100 failed=0 elapsed=475.04ms avg_case_ms=4.75 avg_simplify_ms=1.37, difference@0+50 failed=0 elapsed=329.08ms avg_case_ms=6.58 avg_simplify_ms=2.08, sum@700+100 failed=0 elapsed=188.42ms avg_case_ms=1.88 avg_simplify_ms=0.70
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.04ms median_wire=13.11ms median_wall=49.46ms, sum@0+100 #173 sum runs=3 median_simplify=11.62ms median_wire=11.67ms median_wall=44.55ms, difference@0+50 #174 difference runs=3 median_simplify=12.13ms median_wire=12.19ms median_wall=44.26ms, product@0+100 #175 product runs=3 median_simplify=11.48ms median_wire=11.53ms median_wall=43.98ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.72ms median_wire=10.79ms median_wall=39.94ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.27s | passed=450 failed=0 total=450 avg_case=5.044ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.82s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.92s | passed=1 failed=0 |
