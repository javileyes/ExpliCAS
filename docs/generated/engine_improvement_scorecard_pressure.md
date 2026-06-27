# Engine Improvement Scorecard

- Generated: 2026-06-27T07:18:13.204067+00:00
- Git branch: main
- Git commit: `fc2c3577ca6d398b37cf063f9e4174fead31f87e`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=810.64ms avg_case_ms=8.11 simplify=232.54ms avg_simplify_ms=2.33, sum total=200 failed=0 elapsed=707.02ms avg_case_ms=3.54 simplify=244.33ms avg_simplify_ms=1.22, product total=100 failed=0 elapsed=484.08ms avg_case_ms=4.84 simplify=142.09ms avg_simplify_ms=1.42, difference total=50 failed=0 elapsed=327.30ms avg_case_ms=6.55 simplify=105.73ms avg_simplify_ms=2.11
- Engine hotspots: sum simplify=244.33ms avg_simplify_ms=1.22 wall=707.02ms, shifted_quotient simplify=232.54ms avg_simplify_ms=2.33 wall=810.64ms, product simplify=142.09ms avg_simplify_ms=1.42 wall=484.08ms, difference simplify=105.73ms avg_simplify_ms=2.11 wall=327.30ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=810.64ms avg_case_ms=8.11 avg_simplify_ms=2.33, sum@0+100 failed=0 elapsed=508.23ms avg_case_ms=5.08 avg_simplify_ms=1.68, product@0+100 failed=0 elapsed=484.08ms avg_case_ms=4.84 avg_simplify_ms=1.42, difference@0+50 failed=0 elapsed=327.30ms avg_case_ms=6.55 avg_simplify_ms=2.11, sum@700+100 failed=0 elapsed=198.79ms avg_case_ms=1.99 avg_simplify_ms=0.77
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.02ms median_wire=13.10ms median_wall=49.63ms, sum@0+100 #173 sum runs=3 median_simplify=11.75ms median_wire=11.80ms median_wall=44.48ms, product@0+100 #175 product runs=3 median_simplify=11.63ms median_wire=11.68ms median_wall=44.41ms, difference@0+50 #174 difference runs=3 median_simplify=11.71ms median_wire=11.76ms median_wall=44.20ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.69ms median_wire=10.76ms median_wall=40.26ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.33s | passed=450 failed=0 total=450 avg_case=5.178ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.44s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.06s | passed=1 failed=0 |
