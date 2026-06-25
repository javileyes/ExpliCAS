# Engine Improvement Scorecard

- Generated: 2026-06-25T07:35:12.199768+00:00
- Git branch: main
- Git commit: `e956e2b13a877b508a98f608db6fead459c43b73`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=795.66ms avg_case_ms=7.96 simplify=228.35ms avg_simplify_ms=2.28, sum total=200 failed=0 elapsed=697.08ms avg_case_ms=3.49 simplify=236.42ms avg_simplify_ms=1.18, product total=100 failed=0 elapsed=479.51ms avg_case_ms=4.80 simplify=138.34ms avg_simplify_ms=1.38, difference total=50 failed=0 elapsed=327.65ms avg_case_ms=6.55 simplify=104.43ms avg_simplify_ms=2.09
- Engine hotspots: sum simplify=236.42ms avg_simplify_ms=1.18 wall=697.08ms, shifted_quotient simplify=228.35ms avg_simplify_ms=2.28 wall=795.66ms, product simplify=138.34ms avg_simplify_ms=1.38 wall=479.51ms, difference simplify=104.43ms avg_simplify_ms=2.09 wall=327.65ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=795.66ms avg_case_ms=7.96 avg_simplify_ms=2.28, sum@0+100 failed=0 elapsed=503.95ms avg_case_ms=5.04 avg_simplify_ms=1.64, product@0+100 failed=0 elapsed=479.51ms avg_case_ms=4.80 avg_simplify_ms=1.38, difference@0+50 failed=0 elapsed=327.65ms avg_case_ms=6.55 avg_simplify_ms=2.09, sum@700+100 failed=0 elapsed=193.13ms avg_case_ms=1.93 avg_simplify_ms=0.72
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.26ms median_wire=13.33ms median_wall=49.72ms, difference@0+50 #174 difference runs=3 median_simplify=11.68ms median_wire=11.73ms median_wall=44.72ms, product@0+100 #175 product runs=3 median_simplify=11.73ms median_wire=11.78ms median_wall=45.25ms, sum@0+100 #173 sum runs=3 median_simplify=12.35ms median_wire=12.40ms median_wall=47.60ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.85ms median_wire=10.93ms median_wall=40.51ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.30s | passed=450 failed=0 total=450 avg_case=5.111ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.43s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.03s | passed=1 failed=0 |
