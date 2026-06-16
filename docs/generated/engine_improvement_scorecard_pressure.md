# Engine Improvement Scorecard

- Generated: 2026-06-16T00:18:20.760556+00:00
- Git branch: main
- Git commit: `ac879deee83803887b915ee4bc1ca0b19de7b76e`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=794.40ms avg_case_ms=7.94 simplify=226.59ms avg_simplify_ms=2.27, sum total=200 failed=0 elapsed=702.58ms avg_case_ms=3.51 simplify=235.71ms avg_simplify_ms=1.18, product total=100 failed=0 elapsed=481.31ms avg_case_ms=4.81 simplify=138.48ms avg_simplify_ms=1.38, difference total=50 failed=0 elapsed=328.20ms avg_case_ms=6.56 simplify=104.58ms avg_simplify_ms=2.09
- Engine hotspots: sum simplify=235.71ms avg_simplify_ms=1.18 wall=702.58ms, shifted_quotient simplify=226.59ms avg_simplify_ms=2.27 wall=794.40ms, product simplify=138.48ms avg_simplify_ms=1.38 wall=481.31ms, difference simplify=104.58ms avg_simplify_ms=2.09 wall=328.20ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=794.40ms avg_case_ms=7.94 avg_simplify_ms=2.27, sum@0+100 failed=0 elapsed=506.21ms avg_case_ms=5.06 avg_simplify_ms=1.63, product@0+100 failed=0 elapsed=481.31ms avg_case_ms=4.81 avg_simplify_ms=1.38, difference@0+50 failed=0 elapsed=328.20ms avg_case_ms=6.56 avg_simplify_ms=2.09, sum@700+100 failed=0 elapsed=196.37ms avg_case_ms=1.96 avg_simplify_ms=0.73
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.09ms median_wire=13.16ms median_wall=49.42ms, difference@0+50 #174 difference runs=3 median_simplify=11.62ms median_wire=11.67ms median_wall=44.33ms, sum@0+100 #173 sum runs=3 median_simplify=11.95ms median_wire=12.00ms median_wall=45.25ms, product@0+100 #175 product runs=3 median_simplify=11.64ms median_wire=11.70ms median_wall=44.54ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.68ms median_wire=10.76ms median_wall=39.99ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.31s | passed=450 failed=0 total=450 avg_case=5.133ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.85s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.94s | passed=1 failed=0 |
