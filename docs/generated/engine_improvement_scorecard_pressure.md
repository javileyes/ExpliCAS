# Engine Improvement Scorecard

- Generated: 2026-06-21T11:25:12.403037+00:00
- Git branch: main
- Git commit: `78b7fe37b4c78181ea56c5b464748efe1dad31ce`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=794.06ms avg_case_ms=7.94 simplify=227.78ms avg_simplify_ms=2.28, sum total=200 failed=0 elapsed=713.70ms avg_case_ms=3.57 simplify=242.80ms avg_simplify_ms=1.21, product total=100 failed=0 elapsed=481.60ms avg_case_ms=4.82 simplify=138.34ms avg_simplify_ms=1.38, difference total=50 failed=0 elapsed=335.04ms avg_case_ms=6.70 simplify=107.93ms avg_simplify_ms=2.16
- Engine hotspots: sum simplify=242.80ms avg_simplify_ms=1.21 wall=713.70ms, shifted_quotient simplify=227.78ms avg_simplify_ms=2.28 wall=794.06ms, product simplify=138.34ms avg_simplify_ms=1.38 wall=481.60ms, difference simplify=107.93ms avg_simplify_ms=2.16 wall=335.04ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=794.06ms avg_case_ms=7.94 avg_simplify_ms=2.28, sum@0+100 failed=0 elapsed=513.88ms avg_case_ms=5.14 avg_simplify_ms=1.67, product@0+100 failed=0 elapsed=481.60ms avg_case_ms=4.82 avg_simplify_ms=1.38, difference@0+50 failed=0 elapsed=335.04ms avg_case_ms=6.70 avg_simplify_ms=2.16, sum@700+100 failed=0 elapsed=199.82ms avg_case_ms=2.00 avg_simplify_ms=0.75
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.21ms median_wire=13.28ms median_wall=50.13ms, difference@0+50 #174 difference runs=3 median_simplify=11.86ms median_wire=11.92ms median_wall=44.94ms, sum@0+100 #173 sum runs=3 median_simplify=11.93ms median_wire=11.98ms median_wall=45.04ms, product@0+100 #175 product runs=3 median_simplify=11.83ms median_wire=11.88ms median_wall=44.98ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.69ms median_wire=10.76ms median_wall=40.33ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.32s | passed=450 failed=0 total=450 avg_case=5.156ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.48s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.07s | passed=1 failed=0 |
