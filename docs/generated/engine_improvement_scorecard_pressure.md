# Engine Improvement Scorecard

- Generated: 2026-06-21T19:41:20.696525+00:00
- Git branch: main
- Git commit: `de3a5b67ce26ba2c87357ccf94c5b6b06137dfc6`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=791.81ms avg_case_ms=7.92 simplify=226.59ms avg_simplify_ms=2.27, sum total=200 failed=0 elapsed=702.45ms avg_case_ms=3.51 simplify=238.35ms avg_simplify_ms=1.19, product total=100 failed=0 elapsed=487.52ms avg_case_ms=4.88 simplify=140.15ms avg_simplify_ms=1.40, difference total=50 failed=0 elapsed=328.31ms avg_case_ms=6.57 simplify=105.18ms avg_simplify_ms=2.10
- Engine hotspots: sum simplify=238.35ms avg_simplify_ms=1.19 wall=702.45ms, shifted_quotient simplify=226.59ms avg_simplify_ms=2.27 wall=791.81ms, product simplify=140.15ms avg_simplify_ms=1.40 wall=487.52ms, difference simplify=105.18ms avg_simplify_ms=2.10 wall=328.31ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=791.81ms avg_case_ms=7.92 avg_simplify_ms=2.27, sum@0+100 failed=0 elapsed=505.59ms avg_case_ms=5.06 avg_simplify_ms=1.65, product@0+100 failed=0 elapsed=487.52ms avg_case_ms=4.88 avg_simplify_ms=1.40, difference@0+50 failed=0 elapsed=328.31ms avg_case_ms=6.57 avg_simplify_ms=2.10, sum@700+100 failed=0 elapsed=196.87ms avg_case_ms=1.97 avg_simplify_ms=0.74
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.07ms median_wire=13.14ms median_wall=49.55ms, sum@0+100 #173 sum runs=3 median_simplify=11.73ms median_wire=11.79ms median_wall=44.65ms, product@0+100 #175 product runs=3 median_simplify=11.65ms median_wire=11.70ms median_wall=44.63ms, difference@0+50 #174 difference runs=3 median_simplify=11.81ms median_wire=11.86ms median_wall=45.20ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.90ms median_wire=10.97ms median_wall=40.50ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.31s | passed=450 failed=0 total=450 avg_case=5.133ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.42s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.05s | passed=1 failed=0 |
