# Engine Improvement Scorecard

- Generated: 2026-06-11T17:32:09.185688+00:00
- Git branch: main
- Git commit: `8b494beef3e17f3a35271a1b87c4cdc13e04ea1b`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=777.30ms avg_case_ms=7.77 simplify=219.04ms avg_simplify_ms=2.19, sum total=200 failed=0 elapsed=702.35ms avg_case_ms=3.51 simplify=233.08ms avg_simplify_ms=1.17, product total=100 failed=0 elapsed=473.79ms avg_case_ms=4.74 simplify=135.93ms avg_simplify_ms=1.36, difference total=50 failed=0 elapsed=327.80ms avg_case_ms=6.56 simplify=104.18ms avg_simplify_ms=2.08
- Engine hotspots: sum simplify=233.08ms avg_simplify_ms=1.17 wall=702.35ms, shifted_quotient simplify=219.04ms avg_simplify_ms=2.19 wall=777.30ms, product simplify=135.93ms avg_simplify_ms=1.36 wall=473.79ms, difference simplify=104.18ms avg_simplify_ms=2.08 wall=327.80ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=777.30ms avg_case_ms=7.77 avg_simplify_ms=2.19, sum@0+100 failed=0 elapsed=512.71ms avg_case_ms=5.13 avg_simplify_ms=1.63, product@0+100 failed=0 elapsed=473.79ms avg_case_ms=4.74 avg_simplify_ms=1.36, difference@0+50 failed=0 elapsed=327.80ms avg_case_ms=6.56 avg_simplify_ms=2.08, sum@700+100 failed=0 elapsed=189.64ms avg_case_ms=1.90 avg_simplify_ms=0.70
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=14.20ms median_wire=14.27ms median_wall=53.83ms, difference@0+50 #174 difference runs=3 median_simplify=11.59ms median_wire=11.66ms median_wall=43.97ms, product@0+100 #175 product runs=3 median_simplify=11.34ms median_wire=11.39ms median_wall=43.39ms, sum@0+100 #173 sum runs=3 median_simplify=12.14ms median_wire=12.20ms median_wall=46.95ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.69ms median_wire=12.77ms median_wall=45.05ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.28s | passed=450 failed=0 total=450 avg_case=5.067ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.83s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 17.12s | passed=1 failed=0 |
