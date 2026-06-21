# Engine Improvement Scorecard

- Generated: 2026-06-21T10:26:40.868100+00:00
- Git branch: main
- Git commit: `6c060186c90c917a30014b5269df31271c50c0d4`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=783.95ms avg_case_ms=7.84 simplify=224.09ms avg_simplify_ms=2.24, sum total=200 failed=0 elapsed=701.17ms avg_case_ms=3.51 simplify=238.65ms avg_simplify_ms=1.19, product total=100 failed=0 elapsed=476.73ms avg_case_ms=4.77 simplify=137.19ms avg_simplify_ms=1.37, difference total=50 failed=0 elapsed=329.35ms avg_case_ms=6.59 simplify=105.05ms avg_simplify_ms=2.10
- Engine hotspots: sum simplify=238.65ms avg_simplify_ms=1.19 wall=701.17ms, shifted_quotient simplify=224.09ms avg_simplify_ms=2.24 wall=783.95ms, product simplify=137.19ms avg_simplify_ms=1.37 wall=476.73ms, difference simplify=105.05ms avg_simplify_ms=2.10 wall=329.35ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=783.95ms avg_case_ms=7.84 avg_simplify_ms=2.24, sum@0+100 failed=0 elapsed=506.32ms avg_case_ms=5.06 avg_simplify_ms=1.66, product@0+100 failed=0 elapsed=476.73ms avg_case_ms=4.77 avg_simplify_ms=1.37, difference@0+50 failed=0 elapsed=329.35ms avg_case_ms=6.59 avg_simplify_ms=2.10, sum@700+100 failed=0 elapsed=194.85ms avg_case_ms=1.95 avg_simplify_ms=0.73
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.10ms median_wire=13.18ms median_wall=49.95ms, difference@0+50 #174 difference runs=3 median_simplify=11.71ms median_wire=11.76ms median_wall=44.77ms, sum@0+100 #173 sum runs=3 median_simplify=11.76ms median_wire=11.81ms median_wall=44.15ms, product@0+100 #175 product runs=3 median_simplify=11.61ms median_wire=11.66ms median_wall=43.94ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.78ms median_wire=10.85ms median_wall=40.79ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.29s | passed=450 failed=0 total=450 avg_case=5.089ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.45s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.03s | passed=1 failed=0 |
