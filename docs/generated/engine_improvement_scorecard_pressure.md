# Engine Improvement Scorecard

- Generated: 2026-06-13T09:35:35.501558+00:00
- Git branch: main
- Git commit: `61db6b9c63cc8e58422745742e5ef13cbf726984`
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
- `integrate_exhaustive`: passed=1 failed=0 ignored=0 filtered_out=344

## Mixed Zero Pressure

- Dimension: raw engine pressure on composed zero-target expressions through the canonical eval path.
- Interpretation: better runtime proxy than unified `proved-composed` counts for mixed additive/multiplicative workloads.
- Harness: fixed corpus windows, not a full sweep, so pressure stays reproducible and cheap enough for routine iteration.
- Composition hotspots: shifted_quotient total=100 failed=0 elapsed=781.32ms avg_case_ms=7.81 simplify=221.99ms avg_simplify_ms=2.22, sum total=200 failed=0 elapsed=691.20ms avg_case_ms=3.46 simplify=231.61ms avg_simplify_ms=1.16, product total=100 failed=0 elapsed=471.69ms avg_case_ms=4.72 simplify=134.87ms avg_simplify_ms=1.35, difference total=50 failed=0 elapsed=323.25ms avg_case_ms=6.46 simplify=102.83ms avg_simplify_ms=2.06
- Engine hotspots: sum simplify=231.61ms avg_simplify_ms=1.16 wall=691.20ms, shifted_quotient simplify=221.99ms avg_simplify_ms=2.22 wall=781.32ms, product simplify=134.87ms avg_simplify_ms=1.35 wall=471.69ms, difference simplify=102.83ms avg_simplify_ms=2.06 wall=323.25ms
- Window slices: shifted_quotient@0+100 failed=0 elapsed=781.32ms avg_case_ms=7.81 avg_simplify_ms=2.22, sum@0+100 failed=0 elapsed=498.77ms avg_case_ms=4.99 avg_simplify_ms=1.60, product@0+100 failed=0 elapsed=471.69ms avg_case_ms=4.72 avg_simplify_ms=1.35, difference@0+50 failed=0 elapsed=323.25ms avg_case_ms=6.46 avg_simplify_ms=2.06, sum@700+100 failed=0 elapsed=192.43ms avg_case_ms=1.92 avg_simplify_ms=0.71
- Steady-state engine reruns: shifted_quotient@0+100 #176 shifted_quotient runs=3 median_simplify=12.72ms median_wire=12.79ms median_wall=49.21ms, sum@0+100 #173 sum runs=3 median_simplify=11.45ms median_wire=11.49ms median_wall=43.52ms, difference@0+50 #174 difference runs=3 median_simplify=11.67ms median_wire=11.73ms median_wall=43.90ms, product@0+100 #175 product runs=3 median_simplify=11.53ms median_wire=11.58ms median_wall=45.11ms, shifted_quotient@0+100 #4 shifted_quotient runs=3 median_simplify=11.48ms median_wire=11.56ms median_wall=43.10ms
- Steady-state dominant expressions: shifted_quotient@0+100 #176 shifted_quotient expr=((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^2 - y^2) - ln(x - y) - ln(x + y)) + 1), sum@0+100 #173 sum expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y)), difference@0+50 #174 difference expr=(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) - (ln(x^2 - y^2) - ln(x - y) - ln(x + y))

| Suite | Status | Elapsed | Key metrics |
| --- | --- | --- | --- |
| `simplify_zero_mixed` | `pass` | 2.27s | passed=450 failed=0 total=450 avg_case=5.044ms |
| `calculus_diff_exhaustive_contract` | `pass` | 1.82s | passed=1 failed=0 |
| `calculus_integrate_exhaustive_contract` | `pass` | 0.92s | passed=1 failed=0 |
