# Engine Improvement Scorecard

- Generated: 2026-07-10T18:46:49.623964+00:00
- Git branch: main
- Git commit: `d3b41fe104af42a7da598e51c81f2e6e16ea7128`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=357

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=959.93ms avg_case_ms=9.60 simplify=265.33ms avg_simplify_ms=2.65, sum total=200 failed=0 elapsed=842.26ms avg_case_ms=4.21 simplify=271.91ms avg_simplify_ms=1.36, product total=100 failed=0 elapsed=586.79ms avg_case_ms=5.87 simplify=166.31ms avg_simplify_ms=1.66, difference total=50 failed=0 elapsed=384.81ms avg_case_ms=7.70 simplify=115.92ms avg_simplify_ms=2.32
- Engine hotspots: sum simplify=271.91ms avg_simplify_ms=1.36 wall=842.26ms, shifted_quotient simplify=265.33ms avg_simplify_ms=2.65 wall=959.93ms, product simplify=166.31ms avg_simplify_ms=1.66 wall=586.79ms, difference simplify=115.92ms avg_simplify_ms=2.32 wall=384.81ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=959.93ms avg_case_ms=9.60 avg_simplify_ms=2.65, sum@0+100 failed=0 elapsed=608.66ms avg_case_ms=6.09 avg_simplify_ms=1.89, product@0+100 failed=0 elapsed=586.79ms avg_case_ms=5.87 avg_simplify_ms=1.66, difference@0+50 failed=0 elapsed=384.81ms avg_case_ms=7.70 avg_simplify_ms=2.32, sum@700+100 failed=0 elapsed=233.61ms avg_case_ms=2.34 avg_simplify_ms=0.83
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.84ms median_wire=16.91ms median_wall=65.03ms, product@0+100 #175 product runs=3 median_simplify=14.65ms median_wire=14.70ms median_wall=56.82ms, sum@0+100 #173 sum runs=3 median_simplify=15.24ms median_wire=15.29ms median_wall=58.53ms, difference@0+50 #174 difference runs=3 median_simplify=14.50ms median_wire=14.54ms median_wall=55.99ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=13.21ms median_wire=13.28ms median_wall=49.14ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.77s | passed=450 failed=0 total=450 avg_case=6.156ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.99s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.86s | passed=1 failed=0 |
