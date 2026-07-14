# Engine Improvement Scorecard

- Generated: 2026-07-14T01:00:15.546558+00:00
- Git branch: main
- Git commit: `928728f8f9c293722233b20cd8aba1e54ecaa414`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=949.50ms avg_case_ms=9.50 simplify=261.64ms avg_simplify_ms=2.62, sum total=200 failed=0 elapsed=847.94ms avg_case_ms=4.24 simplify=274.87ms avg_simplify_ms=1.37, product total=100 failed=0 elapsed=588.16ms avg_case_ms=5.88 simplify=167.44ms avg_simplify_ms=1.67, difference total=50 failed=0 elapsed=388.82ms avg_case_ms=7.78 simplify=117.61ms avg_simplify_ms=2.35
- Engine hotspots: sum simplify=274.87ms avg_simplify_ms=1.37 wall=847.94ms, shifted_quotient simplify=261.64ms avg_simplify_ms=2.62 wall=949.50ms, product simplify=167.44ms avg_simplify_ms=1.67 wall=588.16ms, difference simplify=117.61ms avg_simplify_ms=2.35 wall=388.82ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=949.50ms avg_case_ms=9.50 avg_simplify_ms=2.62, sum@0+100 failed=0 elapsed=620.96ms avg_case_ms=6.21 avg_simplify_ms=1.94, product@0+100 failed=0 elapsed=588.16ms avg_case_ms=5.88 avg_simplify_ms=1.67, difference@0+50 failed=0 elapsed=388.82ms avg_case_ms=7.78 avg_simplify_ms=2.35, sum@700+100 failed=0 elapsed=226.98ms avg_case_ms=2.27 avg_simplify_ms=0.81
- Steady-state engine reruns: sum@0+100 #173 sum runs=3 median_simplify=14.88ms median_wire=14.93ms median_wall=56.90ms, shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.12ms median_wire=16.19ms median_wall=62.96ms, product@0+100 #175 product runs=3 median_simplify=14.70ms median_wire=14.74ms median_wall=57.04ms, difference@0+50 #174 difference runs=3 median_simplify=15.34ms median_wire=15.40ms median_wall=60.32ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.71ms median_wire=12.79ms median_wall=47.44ms
- Steady-state dominant expressions: sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.77s | passed=450 failed=0 total=450 avg_case=6.156ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.34s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.90s | passed=1 failed=0 |
