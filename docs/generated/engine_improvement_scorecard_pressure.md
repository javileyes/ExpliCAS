# Engine Improvement Scorecard

- Generated: 2026-06-21T19:54:23.814066+00:00
- Git branch: main
- Git commit: `291c8ed96853409eb7c2a345cf3c559f21ada632`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=353

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=797.66ms avg_case_ms=7.98 simplify=229.00ms avg_simplify_ms=2.29, sum total=200 failed=0 elapsed=704.60ms avg_case_ms=3.52 simplify=239.73ms avg_simplify_ms=1.20, product total=100 failed=0 elapsed=481.87ms avg_case_ms=4.82 simplify=138.71ms avg_simplify_ms=1.39, difference total=50 failed=0 elapsed=331.98ms avg_case_ms=6.64 simplify=106.12ms avg_simplify_ms=2.12
- Engine hotspots: sum simplify=239.73ms avg_simplify_ms=1.20 wall=704.60ms, shifted_quotient simplify=229.00ms avg_simplify_ms=2.29 wall=797.66ms, product simplify=138.71ms avg_simplify_ms=1.39 wall=481.87ms, difference simplify=106.12ms avg_simplify_ms=2.12 wall=331.98ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=797.66ms avg_case_ms=7.98 avg_simplify_ms=2.29, sum@0+100 failed=0 elapsed=508.43ms avg_case_ms=5.08 avg_simplify_ms=1.66, product@0+100 failed=0 elapsed=481.87ms avg_case_ms=4.82 avg_simplify_ms=1.39, difference@0+50 failed=0 elapsed=331.98ms avg_case_ms=6.64 avg_simplify_ms=2.12, sum@700+100 failed=0 elapsed=196.17ms avg_case_ms=1.96 avg_simplify_ms=0.74
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.19ms median_wire=13.25ms median_wall=50.18ms, sum@0+100 #173 sum runs=3 median_simplify=11.92ms median_wire=11.97ms median_wall=44.76ms, product@0+100 #175 product runs=3 median_simplify=11.80ms median_wire=11.85ms median_wall=44.99ms, difference@0+50 #174 difference runs=3 median_simplify=11.61ms median_wire=11.66ms median_wall=44.46ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.48ms median_wire=10.56ms median_wall=39.85ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.32s | passed=450 failed=0 total=450 avg_case=5.156ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.46s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.03s | passed=1 failed=0 |
