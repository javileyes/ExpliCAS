# Engine Improvement Scorecard

- Generated: 2026-06-12T08:18:39.316405+00:00
- Git branch: main
- Git commit: `52a7062091a7523f645cd370a5836e9b8089de18`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=767.17ms avg_case_ms=7.67 simplify=217.47ms avg_simplify_ms=2.17, sum total=200 failed=0 elapsed=681.82ms avg_case_ms=3.41 simplify=227.90ms avg_simplify_ms=1.14, product total=100 failed=0 elapsed=470.66ms avg_case_ms=4.71 simplify=134.07ms avg_simplify_ms=1.34, difference total=50 failed=0 elapsed=327.02ms avg_case_ms=6.54 simplify=103.38ms avg_simplify_ms=2.07
- Engine hotspots: sum simplify=227.90ms avg_simplify_ms=1.14 wall=681.82ms, shifted_quotient simplify=217.47ms avg_simplify_ms=2.17 wall=767.17ms, product simplify=134.07ms avg_simplify_ms=1.34 wall=470.66ms, difference simplify=103.38ms avg_simplify_ms=2.07 wall=327.02ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=767.17ms avg_case_ms=7.67 avg_simplify_ms=2.17, sum@0+100 failed=0 elapsed=493.22ms avg_case_ms=4.93 avg_simplify_ms=1.58, product@0+100 failed=0 elapsed=470.66ms avg_case_ms=4.71 avg_simplify_ms=1.34, difference@0+50 failed=0 elapsed=327.02ms avg_case_ms=6.54 avg_simplify_ms=2.07, sum@700+100 failed=0 elapsed=188.60ms avg_case_ms=1.89 avg_simplify_ms=0.70
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.63ms median_wire=12.69ms median_wall=48.28ms, difference@0+50 #174 difference runs=3 median_simplify=11.45ms median_wire=11.50ms median_wall=43.44ms, sum@0+100 #173 sum runs=3 median_simplify=11.23ms median_wire=11.28ms median_wall=43.35ms, product@0+100 #175 product runs=3 median_simplify=11.38ms median_wire=11.43ms median_wall=43.53ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.10ms median_wire=10.17ms median_wall=38.98ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.25s | passed=450 failed=0 total=450 avg_case=5.000ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.81s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 17.21s | passed=1 failed=0 |
