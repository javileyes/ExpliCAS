# Engine Improvement Scorecard

- Generated: 2026-08-01T22:36:37.806776+00:00
- Git branch: main
- Git commit: `cfa6430ca6132d55ab893bad4e5317e8a89d085d`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=391

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=961.74ms avg_case_ms=9.62 simplify=269.10ms avg_simplify_ms=2.69, sum total=200 failed=0 elapsed=858.54ms avg_case_ms=4.29 simplify=275.72ms avg_simplify_ms=1.38, product total=100 failed=0 elapsed=604.74ms avg_case_ms=6.05 simplify=174.34ms avg_simplify_ms=1.74, difference total=50 failed=0 elapsed=392.29ms avg_case_ms=7.85 simplify=114.94ms avg_simplify_ms=2.30
- Engine hotspots: sum simplify=275.72ms avg_simplify_ms=1.38 wall=858.54ms, shifted_quotient simplify=269.10ms avg_simplify_ms=2.69 wall=961.74ms, product simplify=174.34ms avg_simplify_ms=1.74 wall=604.74ms, difference simplify=114.94ms avg_simplify_ms=2.30 wall=392.29ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=961.74ms avg_case_ms=9.62 avg_simplify_ms=2.69, sum@0+100 failed=0 elapsed=623.54ms avg_case_ms=6.24 avg_simplify_ms=1.92, product@0+100 failed=0 elapsed=604.74ms avg_case_ms=6.05 avg_simplify_ms=1.74, difference@0+50 failed=0 elapsed=392.29ms avg_case_ms=7.85 avg_simplify_ms=2.30, sum@700+100 failed=0 elapsed=235.00ms avg_case_ms=2.35 avg_simplify_ms=0.84
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=16.56ms median_wire=16.63ms median_wall=63.65ms, product@0+100 #175 product runs=3 median_simplify=14.88ms median_wire=14.93ms median_wall=56.53ms, difference@0+50 #174 difference runs=3 median_simplify=14.78ms median_wire=14.83ms median_wall=56.66ms, sum@0+100 #173 sum runs=3 median_simplify=15.05ms median_wire=15.09ms median_wall=57.10ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=12.79ms median_wire=12.85ms median_wall=48.46ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), product@0+100 #175 product expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.82s | passed=450 failed=0 total=450 avg_case=6.267ms |
| `calculus_diff_exhaustive_contract` | `pass` | 12.19s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.79s | passed=1 failed=0 |
