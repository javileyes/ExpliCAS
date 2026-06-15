# Engine Improvement Scorecard

- Generated: 2026-06-15T07:43:18.580842+00:00
- Git branch: main
- Git commit: `f82ec9b71cc717543cfcc12d53d0e82dd62c8559`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=351

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=807.15ms avg_case_ms=8.07 simplify=231.73ms avg_simplify_ms=2.32, sum total=200 failed=0 elapsed=699.71ms avg_case_ms=3.50 simplify=233.60ms avg_simplify_ms=1.17, product total=100 failed=0 elapsed=485.78ms avg_case_ms=4.86 simplify=139.97ms avg_simplify_ms=1.40, difference total=50 failed=0 elapsed=327.03ms avg_case_ms=6.54 simplify=103.79ms avg_simplify_ms=2.08
- Engine hotspots: sum simplify=233.60ms avg_simplify_ms=1.17 wall=699.71ms, shifted_quotient simplify=231.73ms avg_simplify_ms=2.32 wall=807.15ms, product simplify=139.97ms avg_simplify_ms=1.40 wall=485.78ms, difference simplify=103.79ms avg_simplify_ms=2.08 wall=327.03ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=807.15ms avg_case_ms=8.07 avg_simplify_ms=2.32, sum@0+100 failed=0 elapsed=508.58ms avg_case_ms=5.09 avg_simplify_ms=1.62, product@0+100 failed=0 elapsed=485.78ms avg_case_ms=4.86 avg_simplify_ms=1.40, difference@0+50 failed=0 elapsed=327.03ms avg_case_ms=6.54 avg_simplify_ms=2.08, sum@700+100 failed=0 elapsed=191.13ms avg_case_ms=1.91 avg_simplify_ms=0.71
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=13.09ms median_wire=13.16ms median_wall=49.57ms, product@0+100 #175 product runs=3 median_simplify=11.83ms median_wire=11.89ms median_wall=44.60ms, sum@0+100 #173 sum runs=3 median_simplify=11.66ms median_wire=11.71ms median_wall=44.79ms, difference@0+50 #174 difference runs=3 median_simplify=11.89ms median_wire=11.94ms median_wall=44.56ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=10.78ms median_wire=10.85ms median_wall=40.50ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.32s | passed=450 failed=0 total=450 avg_case=5.156ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.87s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.93s | passed=1 failed=0 |
