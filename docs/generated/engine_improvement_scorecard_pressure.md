# Engine Improvement Scorecard

- Generated: 2026-06-15T14:21:04.585761+00:00
- Git branch: main
- Git commit: `e3d949843bf5c245086712706a055b07beea19a7`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=792.71ms avg_case_ms=7.93 simplify=226.35ms avg_simplify_ms=2.26, sum total=200 failed=0 elapsed=694.13ms avg_case_ms=3.47 simplify=232.44ms avg_simplify_ms=1.16, product total=100 failed=0 elapsed=484.17ms avg_case_ms=4.84 simplify=138.69ms avg_simplify_ms=1.39, difference total=50 failed=0 elapsed=327.29ms avg_case_ms=6.55 simplify=103.72ms avg_simplify_ms=2.07
- Engine hotspots: sum simplify=232.44ms avg_simplify_ms=1.16 wall=694.13ms, shifted_quotient simplify=226.35ms avg_simplify_ms=2.26 wall=792.71ms, product simplify=138.69ms avg_simplify_ms=1.39 wall=484.17ms, difference simplify=103.72ms avg_simplify_ms=2.07 wall=327.29ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=792.71ms avg_case_ms=7.93 avg_simplify_ms=2.26, sum@0+100 failed=0 elapsed=502.85ms avg_case_ms=5.03 avg_simplify_ms=1.61, product@0+100 failed=0 elapsed=484.17ms avg_case_ms=4.84 avg_simplify_ms=1.39, difference@0+50 failed=0 elapsed=327.29ms avg_case_ms=6.55 avg_simplify_ms=2.07, sum@700+100 failed=0 elapsed=191.28ms avg_case_ms=1.91 avg_simplify_ms=0.71
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.87ms median_wire=12.94ms median_wall=49.34ms, difference@0+50 #174 difference runs=3 median_simplify=12.12ms median_wire=12.18ms median_wall=45.26ms, product@0+100 #175 product runs=3 median_simplify=11.64ms median_wire=11.69ms median_wall=44.77ms, sum@0+100 #173 sum runs=3 median_simplify=11.70ms median_wire=11.75ms median_wall=44.47ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=11.22ms median_wire=11.30ms median_wall=42.94ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.30s | passed=450 failed=0 total=450 avg_case=5.111ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.85s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.93s | passed=1 failed=0 |
