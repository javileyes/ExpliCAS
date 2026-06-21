# Engine Improvement Scorecard

- Generated: 2026-06-21T05:39:13.704730+00:00
- Git branch: main
- Git commit: `2f6a7f073c0b5ebbe0f4b836f4ed990b793c52fe`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=783.77ms avg_case_ms=7.84 simplify=224.50ms avg_simplify_ms=2.24, sum total=200 failed=0 elapsed=700.36ms avg_case_ms=3.50 simplify=236.68ms avg_simplify_ms=1.18, product total=100 failed=0 elapsed=498.33ms avg_case_ms=4.98 simplify=144.58ms avg_simplify_ms=1.45, difference total=50 failed=0 elapsed=330.20ms avg_case_ms=6.60 simplify=105.95ms avg_simplify_ms=2.12
- Engine hotspots: sum simplify=236.68ms avg_simplify_ms=1.18 wall=700.36ms, shifted_quotient simplify=224.50ms avg_simplify_ms=2.24 wall=783.77ms, product simplify=144.58ms avg_simplify_ms=1.45 wall=498.33ms, difference simplify=105.95ms avg_simplify_ms=2.12 wall=330.20ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=783.77ms avg_case_ms=7.84 avg_simplify_ms=2.24, sum@0+100 failed=0 elapsed=506.36ms avg_case_ms=5.06 avg_simplify_ms=1.65, product@0+100 failed=0 elapsed=498.33ms avg_case_ms=4.98 avg_simplify_ms=1.45, difference@0+50 failed=0 elapsed=330.20ms avg_case_ms=6.60 avg_simplify_ms=2.12, sum@700+100 failed=0 elapsed=194.00ms avg_case_ms=1.94 avg_simplify_ms=0.72
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.07ms median_wire=13.13ms median_wall=49.82ms, product@0+100 #175 product runs=3 median_simplify=11.69ms median_wire=11.74ms median_wall=44.33ms, difference@0+50 #174 difference runs=3 median_simplify=11.63ms median_wire=11.68ms median_wall=44.18ms, sum@0+100 #173 sum runs=3 median_simplify=11.70ms median_wire=11.75ms median_wall=44.52ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.69ms median_wire=10.77ms median_wall=40.42ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.31s | passed=450 failed=0 total=450 avg_case=5.133ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.42s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.03s | passed=1 failed=0 |
