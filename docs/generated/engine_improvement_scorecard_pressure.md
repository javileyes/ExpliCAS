# Engine Improvement Scorecard

- Generated: 2026-06-15T17:18:10.670480+00:00
- Git branch: main
- Git commit: `9264417eb3212de8ff8ddf3f07b2ae8bd2f9003b`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=790.01ms avg_case_ms=7.90 simplify=225.62ms avg_simplify_ms=2.26, sum total=200 failed=0 elapsed=695.29ms avg_case_ms=3.48 simplify=233.32ms avg_simplify_ms=1.17, product total=100 failed=0 elapsed=480.48ms avg_case_ms=4.80 simplify=137.85ms avg_simplify_ms=1.38, difference total=50 failed=0 elapsed=327.08ms avg_case_ms=6.54 simplify=103.65ms avg_simplify_ms=2.07
- Engine hotspots: sum simplify=233.32ms avg_simplify_ms=1.17 wall=695.29ms, shifted_quotient simplify=225.62ms avg_simplify_ms=2.26 wall=790.01ms, product simplify=137.85ms avg_simplify_ms=1.38 wall=480.48ms, difference simplify=103.65ms avg_simplify_ms=2.07 wall=327.08ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=790.01ms avg_case_ms=7.90 avg_simplify_ms=2.26, sum@0+100 failed=0 elapsed=499.67ms avg_case_ms=5.00 avg_simplify_ms=1.61, product@0+100 failed=0 elapsed=480.48ms avg_case_ms=4.80 avg_simplify_ms=1.38, difference@0+50 failed=0 elapsed=327.08ms avg_case_ms=6.54 avg_simplify_ms=2.07, sum@700+100 failed=0 elapsed=195.63ms avg_case_ms=1.96 avg_simplify_ms=0.72
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.23ms median_wire=13.29ms median_wall=49.70ms, product@0+100 #175 product runs=3 median_simplify=11.69ms median_wire=11.74ms median_wall=44.25ms, sum@0+100 #173 sum runs=3 median_simplify=12.27ms median_wire=12.32ms median_wall=47.63ms, difference@0+50 #174 difference runs=3 median_simplify=12.62ms median_wire=12.73ms median_wall=46.71ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.87ms median_wire=10.95ms median_wall=40.92ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.29s | passed=450 failed=0 total=450 avg_case=5.089ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.82s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.93s | passed=1 failed=0 |
