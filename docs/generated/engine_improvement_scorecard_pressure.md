# Engine Improvement Scorecard

- Generated: 2026-07-10T23:40:34.836152+00:00
- Git branch: main
- Git commit: `3193b7eaa4afe69df637370d12c01f9522200abc`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=946.50ms avg_case_ms=9.47 simplify=264.07ms avg_simplify_ms=2.64, sum total=200 failed=0 elapsed=834.28ms avg_case_ms=4.17 simplify=267.31ms avg_simplify_ms=1.34, product total=100 failed=0 elapsed=596.84ms avg_case_ms=5.97 simplify=169.67ms avg_simplify_ms=1.70, difference total=50 failed=0 elapsed=393.48ms avg_case_ms=7.87 simplify=118.25ms avg_simplify_ms=2.36
- Engine hotspots: sum simplify=267.31ms avg_simplify_ms=1.34 wall=834.28ms, shifted_quotient simplify=264.07ms avg_simplify_ms=2.64 wall=946.50ms, product simplify=169.67ms avg_simplify_ms=1.70 wall=596.84ms, difference simplify=118.25ms avg_simplify_ms=2.36 wall=393.48ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=946.50ms avg_case_ms=9.47 avg_simplify_ms=2.64, sum@0+100 failed=0 elapsed=607.95ms avg_case_ms=6.08 avg_simplify_ms=1.87, product@0+100 failed=0 elapsed=596.84ms avg_case_ms=5.97 avg_simplify_ms=1.70, difference@0+50 failed=0 elapsed=393.48ms avg_case_ms=7.87 avg_simplify_ms=2.36, sum@700+100 failed=0 elapsed=226.33ms avg_case_ms=2.26 avg_simplify_ms=0.81
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.63ms median_wire=16.70ms median_wall=63.89ms, difference@0+50 #174 difference runs=3 median_simplify=15.02ms median_wire=15.06ms median_wall=57.97ms, sum@0+100 #173 sum runs=3 median_simplify=15.00ms median_wire=15.05ms median_wall=57.56ms, product@0+100 #175 product runs=3 median_simplify=14.87ms median_wire=14.92ms median_wall=57.37ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.51ms median_wire=12.60ms median_wall=47.94ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.77s | passed=450 failed=0 total=450 avg_case=6.156ms |
| `calculus_diff_exhaustive_contract` | `pass` | 2.00s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.87s | passed=1 failed=0 |
