# Engine Improvement Scorecard

- Generated: 2026-06-21T12:29:17.938138+00:00
- Git branch: main
- Git commit: `908c9f723e152fa1543d09180aed940ecff426fb`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=786.71ms avg_case_ms=7.87 simplify=224.33ms avg_simplify_ms=2.24, sum total=200 failed=0 elapsed=701.87ms avg_case_ms=3.51 simplify=237.89ms avg_simplify_ms=1.19, product total=100 failed=0 elapsed=480.35ms avg_case_ms=4.80 simplify=140.76ms avg_simplify_ms=1.41, difference total=50 failed=0 elapsed=345.78ms avg_case_ms=6.92 simplify=112.03ms avg_simplify_ms=2.24
- Engine hotspots: sum simplify=237.89ms avg_simplify_ms=1.19 wall=701.87ms, shifted_quotient simplify=224.33ms avg_simplify_ms=2.24 wall=786.71ms, product simplify=140.76ms avg_simplify_ms=1.41 wall=480.35ms, difference simplify=112.03ms avg_simplify_ms=2.24 wall=345.78ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=786.71ms avg_case_ms=7.87 avg_simplify_ms=2.24, sum@0+100 failed=0 elapsed=506.78ms avg_case_ms=5.07 avg_simplify_ms=1.66, product@0+100 failed=0 elapsed=480.35ms avg_case_ms=4.80 avg_simplify_ms=1.41, difference@0+50 failed=0 elapsed=345.78ms avg_case_ms=6.92 avg_simplify_ms=2.24, sum@700+100 failed=0 elapsed=195.10ms avg_case_ms=1.95 avg_simplify_ms=0.72
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.19ms median_wire=13.26ms median_wall=52.26ms, sum@0+100 #173 sum runs=3 median_simplify=11.65ms median_wire=11.71ms median_wall=44.49ms, difference@0+50 #174 difference runs=3 median_simplify=11.51ms median_wire=11.56ms median_wall=44.40ms, product@0+100 #175 product runs=3 median_simplify=11.67ms median_wire=11.72ms median_wall=44.33ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.57ms median_wire=10.64ms median_wall=40.01ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.32s | passed=450 failed=0 total=450 avg_case=5.156ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.42s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.03s | passed=1 failed=0 |
