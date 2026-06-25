# Engine Improvement Scorecard

- Generated: 2026-06-25T23:47:18.739986+00:00
- Git branch: main
- Git commit: `224f9373569d1ad076d7b00194747f328b3c6d81`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=787.24ms avg_case_ms=7.87 simplify=224.79ms avg_simplify_ms=2.25, sum total=200 failed=0 elapsed=711.39ms avg_case_ms=3.56 simplify=246.68ms avg_simplify_ms=1.23, product total=100 failed=0 elapsed=484.26ms avg_case_ms=4.84 simplify=142.38ms avg_simplify_ms=1.42, difference total=50 failed=0 elapsed=327.79ms avg_case_ms=6.56 simplify=105.42ms avg_simplify_ms=2.11
- Engine hotspots: sum simplify=246.68ms avg_simplify_ms=1.23 wall=711.39ms, shifted_quotient simplify=224.79ms avg_simplify_ms=2.25 wall=787.24ms, product simplify=142.38ms avg_simplify_ms=1.42 wall=484.26ms, difference simplify=105.42ms avg_simplify_ms=2.11 wall=327.79ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=787.24ms avg_case_ms=7.87 avg_simplify_ms=2.25, sum@0+100 failed=0 elapsed=512.99ms avg_case_ms=5.13 avg_simplify_ms=1.70, product@0+100 failed=0 elapsed=484.26ms avg_case_ms=4.84 avg_simplify_ms=1.42, difference@0+50 failed=0 elapsed=327.79ms avg_case_ms=6.56 avg_simplify_ms=2.11, sum@700+100 failed=0 elapsed=198.41ms avg_case_ms=1.98 avg_simplify_ms=0.76
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.87ms median_wire=13.95ms median_wall=53.50ms, difference@0+50 #174 difference runs=3 median_simplify=11.82ms median_wire=11.87ms median_wall=45.08ms, product@0+100 #175 product runs=3 median_simplify=11.72ms median_wire=11.77ms median_wall=44.73ms, sum@0+100 #173 sum runs=3 median_simplify=11.63ms median_wire=11.68ms median_wall=44.37ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.71ms median_wire=10.78ms median_wall=40.63ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.31s | passed=450 failed=0 total=450 avg_case=5.133ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.42s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.04s | passed=1 failed=0 |
