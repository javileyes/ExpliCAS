# Engine Improvement Scorecard

- Generated: 2026-06-18T14:50:03.430304+00:00
- Git branch: main
- Git commit: `b45e971ddb3f1adbe78f7db7328e68e907a5d268`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=801.88ms avg_case_ms=8.02 simplify=229.18ms avg_simplify_ms=2.29, sum total=200 failed=0 elapsed=713.60ms avg_case_ms=3.57 simplify=241.13ms avg_simplify_ms=1.21, product total=100 failed=0 elapsed=505.99ms avg_case_ms=5.06 simplify=145.89ms avg_simplify_ms=1.46, difference total=50 failed=0 elapsed=335.11ms avg_case_ms=6.70 simplify=105.51ms avg_simplify_ms=2.11
- Engine hotspots: sum simplify=241.13ms avg_simplify_ms=1.21 wall=713.60ms, shifted_quotient simplify=229.18ms avg_simplify_ms=2.29 wall=801.88ms, product simplify=145.89ms avg_simplify_ms=1.46 wall=505.99ms, difference simplify=105.51ms avg_simplify_ms=2.11 wall=335.11ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=801.88ms avg_case_ms=8.02 avg_simplify_ms=2.29, sum@0+100 failed=0 elapsed=513.80ms avg_case_ms=5.14 avg_simplify_ms=1.67, product@0+100 failed=0 elapsed=505.99ms avg_case_ms=5.06 avg_simplify_ms=1.46, difference@0+50 failed=0 elapsed=335.11ms avg_case_ms=6.70 avg_simplify_ms=2.11, sum@700+100 failed=0 elapsed=199.80ms avg_case_ms=2.00 avg_simplify_ms=0.74
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.42ms median_wire=13.49ms median_wall=50.68ms, product@0+100 #175 product runs=3 median_simplify=13.27ms median_wire=13.33ms median_wall=48.67ms, difference@0+50 #174 difference runs=3 median_simplify=13.62ms median_wire=13.68ms median_wall=51.04ms, sum@0+100 #173 sum runs=3 median_simplify=13.64ms median_wire=13.70ms median_wall=51.48ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.11ms median_wire=12.20ms median_wall=45.66ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.36s | passed=450 failed=0 total=450 avg_case=5.244ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.56s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 1.06s | passed=1 failed=0 |
