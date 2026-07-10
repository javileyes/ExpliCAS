# Engine Improvement Scorecard

- Generated: 2026-07-10T10:59:28.279159+00:00
- Git branch: main
- Git commit: `f270df18daa5e71f9c0bf1f79a479e641fa5640d`
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
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=934.09ms avg_case_ms=9.34 simplify=258.51ms avg_simplify_ms=2.59, sum total=200 failed=0 elapsed=831.45ms avg_case_ms=4.16 simplify=268.59ms avg_simplify_ms=1.34, product total=100 failed=0 elapsed=578.33ms avg_case_ms=5.78 simplify=164.37ms avg_simplify_ms=1.64, difference total=50 failed=0 elapsed=383.66ms avg_case_ms=7.67 simplify=115.98ms avg_simplify_ms=2.32
- Engine hotspots: sum simplify=268.59ms avg_simplify_ms=1.34 wall=831.45ms, shifted_quotient simplify=258.51ms avg_simplify_ms=2.59 wall=934.09ms, product simplify=164.37ms avg_simplify_ms=1.64 wall=578.33ms, difference simplify=115.98ms avg_simplify_ms=2.32 wall=383.66ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=934.09ms avg_case_ms=9.34 avg_simplify_ms=2.59, sum@0+100 failed=0 elapsed=608.34ms avg_case_ms=6.08 avg_simplify_ms=1.89, product@0+100 failed=0 elapsed=578.33ms avg_case_ms=5.78 avg_simplify_ms=1.64, difference@0+50 failed=0 elapsed=383.66ms avg_case_ms=7.67 avg_simplify_ms=2.32, sum@700+100 failed=0 elapsed=223.11ms avg_case_ms=2.23 avg_simplify_ms=0.79
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.89ms median_wire=16.96ms median_wall=64.23ms, difference@0+50 #174 difference runs=3 median_simplify=14.63ms median_wire=14.68ms median_wall=56.74ms, sum@0+100 #173 sum runs=3 median_simplify=14.88ms median_wire=14.92ms median_wall=58.11ms, product@0+100 #175 product runs=3 median_simplify=14.59ms median_wire=14.64ms median_wall=55.89ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.25ms median_wire=12.32ms median_wall=47.14ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.73s | passed=450 failed=0 total=450 avg_case=6.067ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.98s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.90s | passed=1 failed=0 |
